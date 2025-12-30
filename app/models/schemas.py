"""Pydantic models for API request/response schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Ingest Endpoint Models
class IngestResponse(BaseModel):
    """Response from PDF ingestion"""

    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    upload_date: str


# Question Generation Models
class QuestionRequest(BaseModel):
    """Request for generating questions"""

    filename: str = Field(..., description="Filename of the uploaded document")
    query: str = Field(..., min_length=3, description="Topic or concept for question generation")
    num_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate")


class MCQQuestion(BaseModel):
    """Multiple Choice Question"""

    question: str
    options: List[str] = Field(..., min_length=4, max_length=4, description="Four answer options")
    correct_answer: int = Field(..., ge=0, le=3, description="Index of correct answer (0-3)")
    explanation: str
    source_section: str
    difficulty: Optional[str] = None


class EvaluationResult(BaseModel):
    """Evaluation result for a question"""

    question_index: int
    score: float = Field(..., ge=0.0, le=1.0)
    approved: bool
    feedback: str
    criteria_scores: Dict[str, float]


class QuestionResponse(BaseModel):
    """Response with generated questions"""

    questions: List[MCQQuestion]
    retrieval_context: str
    evaluation_summary: Dict[str, Any]
    quality_warnings: Optional[List[str]] = None
    iterations_used: int
    total_questions_generated: int


# Document Registry Models
class DocumentMetadata(BaseModel):
    """Metadata for a registered document"""

    document_id: str
    filename: str
    upload_date: datetime
    total_pages: int
    total_chunks: int
    collection_name: str


# Agent State Models (for internal use)
class AgentState(BaseModel):
    """State for LangGraph agent workflow"""

    document_id: str
    query: str
    num_questions: int
    context: str = ""
    generated_questions: List[MCQQuestion] = []
    evaluation_results: List[EvaluationResult] = []
    iteration: int = 0
    max_iterations: int = 3
    feedback_history: List[str] = []
    approved: bool = False


# Health Check
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    chroma_status: str
