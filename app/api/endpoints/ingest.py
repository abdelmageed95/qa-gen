"""Endpoint for PDF ingestion"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.models.schemas import IngestResponse
from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.services.document_manager import DocumentManager
from app.models.schemas import DocumentMetadata

from app.core.config import settings
import uuid
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def get_pdf_processor():
    """Dependency for PDF processor"""
    return PDFProcessor()


def get_vector_store():
    """Dependency for vector store"""
    return VectorStore()


def get_document_manager():
    """Dependency for document manager"""
    return DocumentManager()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    vector_store: VectorStore = Depends(get_vector_store),
    doc_manager: DocumentManager = Depends(get_document_manager)
):
    """
    Upload and process a PDF file

    Process:
    1. Save uploaded file
    2. Extract text and chunk document
    3. Create embeddings and store in ChromaDB
    4. Return document metadata

    Returns:
        IngestResponse with document_id and metadata
    """
    logger.info(f"Received PDF upload: {file.filename}")

    # Validate file type ( comment this check out if you want to accept other types of files)
    # if not str(file.filename).endswith('.pdf'):
    #     raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Prevent duplicate filenames
    existing_doc_id = doc_manager.get_document_id_by_filename(str(file.filename))
    if existing_doc_id:
        raise HTTPException(status_code=400, detail="A document with this filename already exists")
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())

    # Create upload directory if it doesn't exist
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    ext = Path(str(file.filename)).suffix.lower()
    file_path = upload_dir / f"{document_id}{ext}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved PDF to {file_path}")

        # 1. Process PDF
        logger.info("Processing PDF...")
        chunks, total_pages = pdf_processor.process_pdf(str(file_path))

        # 2. Create vector store collection and add documents to it
        logger.info("Creating vector store collection...")
        collection = vector_store.create_collection(document_id)
        logger.info("Adding documents to vector store...")
        total_chunks = vector_store.add_documents(
            collection=collection,
            chunks=chunks,
            document_id=document_id
        )

        # 3. Register document
        logger.info("Registering document...")
        doc_manager.register_document(
            document_id=document_id,
            filename=str(file.filename),
            total_pages=total_pages,
            total_chunks=total_chunks,
            collection_name=f"doc_{document_id}"
        )

        logger.info(f"Successfully ingested document {document_id}")

        # Return response
        return IngestResponse(
            document_id=document_id,
            filename=str(file.filename),
            total_pages=total_pages,
            total_chunks=total_chunks,
            upload_date=str(doc_manager.get_document(document_id).upload_date)
        )

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")

        # Cleanup on error
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.delete("/documents/{document_name}", response_model=dict)
async def delete_document(
    document_name: str,
    vector_store: VectorStore = Depends(get_vector_store),
    doc_manager: DocumentManager = Depends(get_document_manager)
):
    """
    Delete a document and its associated data

    Process:
    1. Delete ChromaDB collection
    2. Remove document from registry
    3. Delete uploaded file

    Returns:
        dict with deletion status
    """
    document_id = doc_manager.get_document_id_by_filename(document_name)
    logger.info(f"Received request to delete document: {document_id}")

    try:
        # 1. Delete vector store collection
        collection = vector_store.get_collection(document_id)
        if collection:
            vector_store.delete_collection(document_id)
            logger.info(f"Deleted vector store collection for document {document_id}")
        else:
            logger.warning(f"No vector store collection found for document {document_id}")

        # 2. Remove document from registry
        doc_manager.delete_document(document_id)
        logger.info(f"Deleted document {document_id} from registry")

        # 3. Delete uploaded file
        upload_path = Path(settings.UPLOAD_DIR) / f"{document_id}.pdf"
        if upload_path.exists():
            upload_path.unlink()
            logger.info(f"Deleted uploaded file for document {document_id}")
        else:
            logger.warning(f"No uploaded file found for document {document_id}")

        return {"detail": f"Document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
    
@router.get("/documents/list", response_model=list[DocumentMetadata])
async def list_documents(
    doc_manager: DocumentManager = Depends(get_document_manager)
):
    """
    List all ingested documents

    Returns:
        List of documents with their metadata
    """
    logger.info("Listing all ingested documents")
    try:
        documents = doc_manager.list_documents()
        return documents

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")