"""LangGraph multi-agent workflow for question generation and evaluation"""

from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import MCQQuestion
from app.core.config import settings
import json
import logging

logger = logging.getLogger(__name__)


class QuestionGenerationState(TypedDict):
    """State for the question generation workflow"""
    document_id: str
    query: str
    num_questions: int
    context: str
    generated_questions: List[Dict]
    evaluation_results: List[Dict]
    iteration: int
    max_iterations: int
    feedback_history: List[str]
    approved: bool
    quality_warnings: List[str]


class MultiAgentWorkflow:
    """LangGraph workflow with Question Generator and Evaluator agents"""

    def __init__(self, vector_store):
        """
        Initialize the multi-agent workflow

        Args:
            vector_store: VectorStore instance for retrieval
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(QuestionGenerationState)

        # Add nodes
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_questions", self.generate_questions)
        workflow.add_node("evaluate_questions", self.evaluate_questions)
        workflow.add_node("regenerate_questions", self.regenerate_questions)

        # Set entry point
        workflow.add_edge(START, "retrieve_context")
        # Add edges
        workflow.add_edge("retrieve_context", "generate_questions")
        workflow.add_edge("generate_questions", "evaluate_questions")
        workflow.add_edge("regenerate_questions", "evaluate_questions")

        # Add conditional edge from evaluate
        workflow.add_conditional_edges(
            "evaluate_questions",
            self.should_continue,
            {
                "regenerate": "regenerate_questions",
                "end": END
            }
        )

        return workflow.compile()

    def retrieve_context(self, state: QuestionGenerationState
                         ) -> QuestionGenerationState:
        """Retrieve relevant context from vector DB"""
        logger.info(f"Retrieving context for query: {state['query']}")

        try:
            collection = self.vector_store.get_collection(state["document_id"])

            if not collection:
                raise ValueError(f"Document {state['document_id']} not found")

            # Perform similarity search
            results = self.vector_store.similarity_search(
                collection=collection,
                query=state["query"],
                top_k=settings.RETRIEVAL_TOP_K
            )

            # Check if results are relevant enough
            if not results or len(results) == 0:
                logger.warning(f"No relevant context found for query: {state['query']}")
                state["context"] = ""
                state["quality_warnings"] = [
                    f"No relevant content found for topic: '{state['query']}'",
                    "This topic does not appear to exist in the document.",
                    "Please try a different topic or upload a relevant document."
                ]
                return state

            # Check relevance score (if available from vector store)
            # Most vector stores return a distance/similarity score
            min_relevance_threshold = settings.MIN_RELEVANCE_THRESHOLD
            relevant_results = []

            for result in results:
                # Check if result has a distance/score
                distance = result.get("distance", 0)
                # Lower distance = more similar (for most distance metrics)
                # If distance is too high, content is not relevant
                if distance < min_relevance_threshold or distance == 0:
                    relevant_results.append(result)

            # If no relevant results after filtering
            if len(relevant_results) == 0:
                logger.warning(
                    f"Retrieved {len(results)} chunks but none are "
                    f"sufficiently relevant for query: {state['query']}"
                )
                state["context"] = ""
                state["quality_warnings"] = [
                    f"No sufficiently relevant content found for topic: '{state['query']}'",
                    "The document does not contain substantial information about this topic.",
                    "Please try a different topic that exists in the document."
                ]
                return state

            # Combine retrieved chunks
            context_parts = []
            for result in relevant_results:
                section = result["metadata"].get("section", "Unknown")
                text = result["text"]
                context_parts.append(f"[{section}]\n{text}")

            state["context"] = "\n\n".join(context_parts)
            logger.info(
                f"Retrieved {len(relevant_results)} relevant context chunks "
                f"(from {len(results)} total)"
            )

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            state["context"] = ""
            state["quality_warnings"] = [
                f"Error retrieving context: {str(e)}"
            ]

        return state

    def generate_questions(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Generate MCQ questions using LLM"""
        logger.info(
            f"Generating {state['num_questions']} questions "
            f"(iteration {state['iteration']})"
        )

        # Check if context is empty (topic not found in document)
        if not state["context"] or state["context"].strip() == "":
            logger.warning("No context available - cannot generate questions")
            state["generated_questions"] = []
            if not state.get("quality_warnings"):
                state["quality_warnings"] = [
                    "Cannot generate questions: No relevant content found"
                ]
            return state

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert question generator. Generate high-quality Multiple Choice Questions (MCQs) based on the provided context.

Requirements:
1. Each question must have exactly 4 options (A, B, C, D)
2. Only one option should be correct
3. Questions should be clear and unambiguous
4. Distractors (wrong answers) should be plausible but clearly incorrect
5. Questions should test understanding, not just recall
6. Base all questions strictly on the provided context

Return a JSON array of questions with this exact format:
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0,
    "explanation": "Explanation of why this is correct",
    "source_section": "Section name from context"
  }}
]

{feedback}"""),
            ("user", """Context:
{context}

Topic/Query: {query}

Generate {num_questions} high-quality MCQ questions based on the above context.""")
        ])

        feedback_text = ""
        if state["feedback_history"]:
            feedback_text = "\nPrevious feedback from evaluator:\n" + "\n".join(state["feedback_history"])

        try:
            # Generate questions
            messages = prompt.format_messages(
                context=state["context"],
                query=state["query"],
                num_questions=state["num_questions"],
                feedback=feedback_text
            )

            response = self.llm.invoke(messages)
            content = response.content

            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            questions = json.loads(content)

            state["generated_questions"] = questions
            logger.info(f"Generated {len(questions)} questions")

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            state["generated_questions"] = []

        return state

    def evaluate_questions(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Evaluate generated questions"""
        logger.info(f"Evaluating {len(state['generated_questions'])} questions")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert question evaluator. Evaluate the quality of MCQ questions based on these criteria:

1. Correctness (40%): Answer must be derivable from the context
2. Clarity (25%): Question is unambiguous and well-worded
3. Distractor Quality (25%): Wrong options are plausible but clearly incorrect
4. Coverage (10%): Question tests important concepts

For each question, provide:
- Overall score (0.0 to 1.0)
- Approved (true/false) - approve if score >= {min_score}
- Specific feedback for improvement

Return a JSON array:
[
  {{
    "question_index": 0,
    "score": 0.85,
    "approved": true,
    "feedback": "Detailed feedback",
    "criteria_scores": {{
      "correctness": 0.9,
      "clarity": 0.8,
      "distractor_quality": 0.85,
      "coverage": 0.8
    }}
  }}
]"""),
            ("user", """Context:
{context}

Questions to evaluate:
{questions}

Evaluate each question strictly.""")
        ])

        try:
            messages = prompt.format_messages(
                context=state["context"],
                questions=json.dumps(state["generated_questions"], indent=2),
                min_score=settings.MIN_QUESTION_SCORE
            )

            response = self.llm.invoke(messages)
            content = response.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            evaluations = json.loads(content)

            state["evaluation_results"] = evaluations

            # Count approved questions
            approved_count = sum(1 for e in evaluations if e.get("approved", False))
            logger.info(f"Approved {approved_count}/{len(evaluations)} questions")

            # Increment iteration
            state["iteration"] += 1

        except Exception as e:
            logger.error(f"Error evaluating questions: {e}")
            state["evaluation_results"] = []
            state["iteration"] += 1

        return state

    def regenerate_questions(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Regenerate questions based on feedback"""
        logger.info("Regenerating questions based on feedback")

        # Collect feedback from rejected questions
        feedback_items = []
        for eval_result in state["evaluation_results"]:
            if not eval_result.get("approved", False):
                idx = eval_result.get("question_index", 0)
                feedback = eval_result.get("feedback", "")
                feedback_items.append(f"Question {idx + 1}: {feedback}")

        if feedback_items:
            state["feedback_history"].extend(feedback_items)

        # Regenerate (will use feedback in generate_questions)
        return self.generate_questions(state)

    def should_continue(self, state: QuestionGenerationState) -> str:
        """Decide whether to continue or end workflow"""

        # If no context was found, end immediately
        if not state["context"] or state["context"].strip() == "":
            logger.warning("No context available - ending workflow")
            return "end"

        # Check if we have approved questions
        if not state["evaluation_results"]:
            if state["iteration"] < state["max_iterations"]:
                return "regenerate"
            else:
                return "end"

        approved_count = sum(1 for e in state["evaluation_results"] if e.get("approved", False))
        all_approved = approved_count == len(state["evaluation_results"])

        # Check termination conditions
        if all_approved:
            state["approved"] = True
            return "end"

        if state["iteration"] >= state["max_iterations"]:
            # Max iterations reached, return with warnings
            state["approved"] = False
            state["quality_warnings"] = [
                f"Reached maximum iterations ({state['max_iterations']})",
                f"Only {approved_count}/{len(state['evaluation_results'])} questions approved",
                "Returning best available questions with warnings"
            ]
            return "end"

        # Continue regenerating
        return "regenerate"

    def run(
        self,
        document_id: str,
        query: str,
        num_questions: int
    ) -> Dict:
        """
        Run the complete workflow

        Args:
            document_id: Document to query
            query: Topic/concept for questions
            num_questions: Number of questions to generate

        Returns:
            Dict with questions and evaluation results
        """
        logger.info(f"Starting question generation workflow for document: {document_id}")

        # Initialize state
        initial_state: QuestionGenerationState = {
            "document_id": document_id,
            "query": query,
            "num_questions": num_questions,
            "context": "",
            "generated_questions": [],
            "evaluation_results": [],
            "iteration": 0,
            "max_iterations": settings.MAX_ITERATIONS,
            "feedback_history": [],
            "approved": False,
            "quality_warnings": []
        }

        # Run workflow
        final_state = self.workflow.invoke(initial_state)

        # Format results
        approved_questions = []
        for i, question in enumerate(final_state["generated_questions"]):
            eval_result = final_state["evaluation_results"][i] if i < len(final_state["evaluation_results"]) else None

            if eval_result and eval_result.get("approved", False):
                approved_questions.append(MCQQuestion(
                    question=question["question"],
                    options=question["options"],
                    correct_answer=question["correct_answer"],
                    explanation=question["explanation"],
                    source_section=question.get("source_section", "Unknown")
                ))

        # If no approved questions, return all with warnings
        if not approved_questions and final_state["generated_questions"]:
            for question in final_state["generated_questions"]:
                approved_questions.append(MCQQuestion(
                    question=question["question"],
                    options=question["options"],
                    correct_answer=question["correct_answer"],
                    explanation=question["explanation"],
                    source_section=question.get("source_section", "Unknown")
                ))

        # Compile evaluation summary
        evaluation_summary = {
            "total_generated": len(final_state["generated_questions"]),
            "total_approved": sum(1 for e in final_state["evaluation_results"] if e.get("approved", False)),
            "average_score": sum(e.get("score", 0) for e in final_state["evaluation_results"]) / len(final_state["evaluation_results"]) if final_state["evaluation_results"] else 0
        }

        return {
            "questions": approved_questions,
            "context": final_state["context"],
            "evaluation_summary": evaluation_summary,
            "quality_warnings": final_state["quality_warnings"] if final_state["quality_warnings"] else None,
            "iterations_used": final_state["iteration"]
        }
