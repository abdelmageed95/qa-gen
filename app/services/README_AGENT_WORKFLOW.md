# Agent Workflow: Multi-Agent Question Generation System

## Overview

The `agent_workflow.py` module implements a **LangGraph-based multi-agent system** for generating and evaluating high-quality Multiple Choice Questions (MCQs) from documents. It uses a state machine architecture with two cooperating AI agents that iteratively improve question quality through a feedback loop.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MultiAgentWorkflow                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Question    │───▶│  Evaluator   │───▶│  Regenerator │   │
│  │  Generator   │    │   Agent      │    │    Agent     │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         ▲                    │                    │         │
│         │                    │                    │         │
│         └────────────────────┴────────────────────┘         │
│                     Feedback Loop                           │
└─────────────────────────────────────────────────────────────┘
```

### LangGraph State Machine

The workflow is implemented as a **StateGraph** with the following nodes and edges:

```
START
  │
  ▼
retrieve_context ────────────────────────────┐
  │                                          │
  ▼                                          │
generate_questions                           │
  │                                          │
  ▼                                          │
evaluate_questions ◄─────┐                   │
  │                      │                   │
  ├─ should_continue ────┤                   │
  │                      │                   │
  ├─ regenerate ────▶ regenerate_questions   │
  │                      │                   │
  └─ end ──────────────▶ END                 │
                                             │
         (no context found) ─────────────────┘
```

## Workflow Nodes

### 1. Retrieve Context (`retrieve_context`)

**Purpose**: Retrieves relevant document chunks using vector similarity search.

**Process**:
- Queries the vector database with the user's topic/query
- Uses cosine similarity to find the most relevant chunks
- Filters results based on relevance threshold (`MIN_RELEVANCE_THRESHOLD`)
- Combines top-k chunks with section metadata

**Quality Controls**:
- Checks if any results were found
- Validates relevance scores against threshold
- Sets quality warnings if topic not found in document
- Returns empty context with warnings if no relevant content exists

**Code Reference**: `app/services/agent_workflow.py:79-155`

**Example Output**:
```python
state["context"] = """
[Introduction]
Artificial Intelligence (AI) is the simulation of human intelligence...

[Machine Learning]
Machine learning is a subset of AI that enables systems to learn...
"""
```

### 2. Generate Questions (`generate_questions`)

**Purpose**: Uses LLM to generate MCQ questions based on retrieved context.

**Process**:
- Constructs a detailed prompt with requirements and context
- Includes feedback from previous iterations (if any)
- Calls OpenAI API with temperature=0.7 for creative variety
- Parses JSON response containing question array

**Question Format**:
```json
{
  "question": "What is the primary goal of machine learning?",
  "options": [
    "To replace human intelligence",
    "To enable systems to learn from data",
    "To create robots",
    "To store information"
  ],
  "correct_answer": 1,
  "explanation": "Machine learning focuses on creating systems that can learn and improve from experience without being explicitly programmed.",
  "source_section": "Machine Learning"
}
```

**Quality Requirements** (enforced by prompt):
1. Exactly 4 options (A, B, C, D)
2. Only one correct answer
3. Clear, unambiguous questions
4. Plausible distractors (wrong answers)
5. Tests understanding, not just recall
6. Strictly based on provided context

**Code Reference**: `app/services/agent_workflow.py:157-237`

### 3. Evaluate Questions (`evaluate_questions`)

**Purpose**: Critically evaluates each generated question using a separate LLM call.

**Evaluation Criteria**:
1. **Correctness (40%)**: Answer must be derivable from context
2. **Clarity (25%)**: Question is unambiguous and well-worded
3. **Distractor Quality (25%)**: Wrong options are plausible but incorrect
4. **Coverage (10%)**: Tests important concepts

**Evaluation Output**:
```json
{
  "question_index": 0,
  "score": 0.85,
  "approved": true,
  "feedback": "Good question with clear options...",
  "criteria_scores": {
    "correctness": 0.9,
    "clarity": 0.8,
    "distractor_quality": 0.85,
    "coverage": 0.8
  }
}
```

**Approval Logic**:
- Questions with `score >= MIN_QUESTION_SCORE` are approved
- Rejected questions receive specific feedback for improvement

**Code Reference**: `app/services/agent_workflow.py:239-312`

### 4. Regenerate Questions (`regenerate_questions`)

**Purpose**: Regenerates rejected questions using evaluator feedback.

**Process**:
- Collects feedback from all rejected questions
- Appends feedback to `feedback_history`
- Calls `generate_questions` again with accumulated feedback
- Preserves context across iterations

**Feedback Example**:
```
Question 1: Distractor 'C' is too obviously wrong. Make it more plausible.
Question 3: Question is ambiguous. Clarify what is being asked.
```

**Code Reference**: `app/services/agent_workflow.py:314-330`

### 5. Decision Node (`should_continue`)

**Purpose**: Determines whether to continue iteration or end workflow.

**Decision Logic**:
```python
if no_context_found:
    return "end"  # Cannot generate without context

if all_questions_approved:
    return "end"  # Success!

if iteration >= max_iterations:
    return "end"  # Max attempts reached, return best available

return "regenerate"  # Try again with feedback
```

**Termination Conditions**:
1. All questions approved (success)
2. Maximum iterations reached
3. No context available

**Code Reference**: `app/services/agent_workflow.py:332-366`

## State Management

### QuestionGenerationState

The workflow maintains a typed state dictionary that flows through all nodes:

```python
class QuestionGenerationState(TypedDict):
    # Input parameters
    document_id: str           # Document identifier for retrieval
    query: str                 # Topic/concept to generate questions about
    num_questions: int         # Requested number of questions

    # Retrieved data
    context: str               # Combined relevant document chunks

    # Generated content
    generated_questions: List[Dict]      # Current batch of questions
    evaluation_results: List[Dict]       # Evaluation scores and feedback

    # Iteration tracking
    iteration: int             # Current iteration number (0-indexed)
    max_iterations: int        # Maximum allowed iterations
    feedback_history: List[str]          # Accumulated feedback from evaluator

    # Quality control
    approved: bool             # Whether all questions meet quality standards
    quality_warnings: List[str]          # Warnings about quality issues
```

## Quality Control Mechanisms

### 1. Context Relevance Filtering

```python
# app/services/agent_workflow.py:110-133
MIN_RELEVANCE_THRESHOLD = 0.7  # Configurable in settings

for result in results:
    distance = result.get("distance", 0)
    if distance < MIN_RELEVANCE_THRESHOLD:
        relevant_results.append(result)
```

If no relevant context is found, the workflow terminates early with descriptive warnings.

### 2. Multi-Criteria Evaluation

Each question is evaluated on 4 weighted criteria:
- **Correctness (40%)**: Most important
- **Clarity (25%)**: Prevents ambiguity
- **Distractor Quality (25%)**: Ensures learning value
- **Coverage (10%)**: Tests important concepts

### 3. Iterative Improvement

The feedback loop allows up to `MAX_ITERATIONS` attempts to improve questions:

```
Iteration 1: Generate → Evaluate → 3/5 approved
Iteration 2: Regenerate with feedback → Evaluate → 4/5 approved
Iteration 3: Regenerate → Evaluate → 5/5 approved ✓
```

### 4. Graceful Degradation

If `MAX_ITERATIONS` is reached without full approval:
- Returns the best available questions
- Includes quality warnings
- Provides evaluation summary with scores

## Configuration

The workflow uses these settings from `app/core/config.py`:

```python
OPENAI_MODEL = "gpt-4"                # LLM model for both agents
RETRIEVAL_TOP_K = 5                   # Number of context chunks to retrieve
MIN_RELEVANCE_THRESHOLD = 0.7         # Minimum similarity score
MIN_QUESTION_SCORE = 0.75             # Minimum score for approval (0.0-1.0)
MAX_ITERATIONS = 3                    # Maximum regeneration attempts
```

## Usage Example

```python
from app.services.agent_workflow import MultiAgentWorkflow
from app.services.vector_store import VectorStore

# Initialize
vector_store = VectorStore()
workflow = MultiAgentWorkflow(vector_store)

# Run workflow
result = workflow.run(
    document_id="doc_123",
    query="machine learning algorithms",
    num_questions=5
)

# Access results
questions = result["questions"]              # List[MCQQuestion]
context = result["context"]                  # Retrieved context
summary = result["evaluation_summary"]       # Evaluation stats
warnings = result["quality_warnings"]        # Quality issues (if any)
iterations = result["iterations_used"]       # Number of iterations
```

## Output Format

### Success Response

```python
{
    "questions": [
        MCQQuestion(...),  # Only approved questions
        MCQQuestion(...),
        ...
    ],
    "context": "Retrieved document chunks...",
    "evaluation_summary": {
        "total_generated": 5,
        "total_approved": 5,
        "average_score": 0.87
    },
    "quality_warnings": None,
    "iterations_used": 2
}
```

### Partial Success (Some Questions Approved)

```python
{
    "questions": [MCQQuestion(...), MCQQuestion(...)],  # 2 approved
    "context": "Retrieved document chunks...",
    "evaluation_summary": {
        "total_generated": 5,
        "total_approved": 2,
        "average_score": 0.72
    },
    "quality_warnings": [
        "Reached maximum iterations (3)",
        "Only 2/5 questions approved",
        "Returning best available questions with warnings"
    ],
    "iterations_used": 3
}
```

### No Context Found

```python
{
    "questions": [],
    "context": "",
    "evaluation_summary": {
        "total_generated": 0,
        "total_approved": 0,
        "average_score": 0
    },
    "quality_warnings": [
        "No relevant content found for topic: 'quantum computing'",
        "This topic does not appear to exist in the document.",
        "Please try a different topic or upload a relevant document."
    ],
    "iterations_used": 0
}
```

## Advantages of Multi-Agent Architecture

### 1. Separation of Concerns
- **Generator**: Focuses on creativity and question creation
- **Evaluator**: Focuses on quality assessment and critique
- Each agent has a specialized role with distinct prompts

### 2. Quality Assurance
- Independent evaluation prevents generator from "grading its own homework"
- Objective quality metrics across multiple criteria
- Consistent standards across all questions

### 3. Iterative Improvement
- Feedback loop allows learning from mistakes
- Questions get progressively better with each iteration
- Evaluator provides specific, actionable feedback

### 4. Robustness
- Handles edge cases (no context, poor questions, etc.)
- Graceful degradation with partial results
- Informative warnings guide users to better inputs

### 5. Observability
- Structured logging at each step
- Detailed evaluation scores and feedback
- Iteration tracking for debugging

## Dependencies

- **LangGraph**: State machine orchestration
- **LangChain**: LLM abstraction and prompting
- **OpenAI API**: GPT-4 for generation and evaluation
- **Vector Store**: Document retrieval (ChromaDB/Qdrant/etc.)

## Error Handling

The workflow handles errors at each node:

```python
try:
    # Node operation
except Exception as e:
    logger.error(f"Error in node: {e}")
    # Set appropriate state values
    # Continue workflow gracefully
```

Errors are logged and converted to user-friendly warnings in the final output.

## Performance Considerations

- **API Calls**: 2-6 LLM calls per workflow run (1 generate + 1 evaluate per iteration)
- **Token Usage**: Proportional to context size and number of questions
- **Latency**: ~10-30 seconds typical (depends on LLM response time)
- **Optimization**: Consider caching context retrieval for repeat queries

## Logging

The workflow provides detailed logging at INFO level:

```
INFO - Retrieving context for query: machine learning
INFO - Retrieved 5 relevant context chunks (from 5 total)
INFO - Generating 5 questions (iteration 0)
INFO - Generated 5 questions
INFO - Evaluating 5 questions
INFO - Approved 3/5 questions
INFO - Regenerating questions based on feedback
INFO - Generating 5 questions (iteration 1)
...
```

## Future Enhancements

Potential improvements to consider:

1. **Parallel Evaluation**: Evaluate questions in parallel for speed
2. **Adaptive Iterations**: Stop early if quality plateaus
3. **Question Diversity**: Ensure questions cover different sections
4. **Difficulty Levels**: Generate questions at various difficulty levels
5. **Human-in-the-Loop**: Allow manual approval/rejection
6. **Answer Validation**: Cross-verify answers against source text
7. **Batch Processing**: Generate questions for multiple topics simultaneously

## References

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **State Machine Pattern**: Enables complex, stateful AI workflows
- **MCQ Best Practices**: Based on educational assessment literature
