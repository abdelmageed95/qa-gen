"""Endpoint for question generation"""

from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import QuestionRequest, QuestionResponse
from app.services.vector_store import VectorStore
from app.services.document_manager import DocumentManager
from app.services.agent_workflow import MultiAgentWorkflow
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def get_vector_store():
    """Dependency for vector store"""
    return VectorStore()


def get_document_manager():
    """Dependency for document manager"""
    return DocumentManager()


def get_agent_workflow(vector_store: VectorStore = Depends(get_vector_store)):
    """Dependency for agent workflow"""
    return MultiAgentWorkflow(vector_store=vector_store)


@router.post("/generate/questions", response_model=QuestionResponse)
async def generate_questions(
    request: QuestionRequest,
    doc_manager: DocumentManager = Depends(get_document_manager),
    agent_workflow: MultiAgentWorkflow = Depends(get_agent_workflow)
):
    """
    Generate MCQ questions from a document

    Process:
    1. Resolve filename to document_id
    2. Retrieve relevant content from vector DB
    3. Generate questions using Question Generator Agent
    4. Evaluate questions using Evaluator Agent
    5. Return approved questions (with warnings if quality threshold not met)

    Returns:
        QuestionResponse with generated questions and evaluation summary
    """
    logger.info(
        f"Received question generation request for filename: {request.filename}"
    )

    # Resolve filename to document_id
    document_id = doc_manager.get_document_id_by_filename(request.filename)

    if not document_id:
        raise HTTPException(
            status_code=404,
            detail=f"Document with filename '{request.filename}' not found"
        )

    try:
        # Run multi-agent workflow
        logger.info(f"Running multi-agent workflow for query: {request.query}")
        result = agent_workflow.run(
            document_id=document_id,
            query=request.query,
            num_questions=request.num_questions
        )

        # Return response
        context = result["context"]
        truncated_context = (
            context[:500] + "..." if len(context) > 500 else context
        )

        return QuestionResponse(
            questions=result["questions"],
            retrieval_context=truncated_context,
            evaluation_summary=result["evaluation_summary"],
            quality_warnings=result["quality_warnings"],
            iterations_used=result["iterations_used"],
            total_questions_generated=len(result["questions"])
        )

    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating questions: {str(e)}"
        )
