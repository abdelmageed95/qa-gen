"""Document registry manager for tracking uploaded documents"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from app.models.schemas import DocumentMetadata
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document registry using JSON file storage"""

    def __init__(self, registry_path: str = ''):
        """
        Initialize document manager

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path or settings.DOCUMENT_REGISTRY_PATH)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        """Ensure registry file and directory exist"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({})
            logger.info(f"Created new document registry at {self.registry_path}")

    def _load_registry(self) -> Dict:
        """Load registry from file"""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return {}

    def _save_registry(self, registry: Dict):
        """Save registry to file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def register_document(
        self,
        document_id: str,
        filename: str,
        total_pages: int,
        total_chunks: int,
        collection_name: str
    ) -> DocumentMetadata:
        """
        Register a new document

        Args:
            document_id: Unique document identifier
            filename: Original filename
            total_pages: Number of pages
            total_chunks: Number of chunks
            collection_name: ChromaDB collection name

        Returns:
            DocumentMetadata object
        """
        logger.info(f"Registering document: {document_id}")

        registry = self._load_registry()

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            upload_date=datetime.now(),
            total_pages=total_pages,
            total_chunks=total_chunks,
            collection_name=collection_name
        )

        # Convert to dict for JSON storage
        registry[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "upload_date": metadata.upload_date.isoformat(),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "collection_name": collection_name
        }

        self._save_registry(registry) # dump updated registry
        logger.info(f"Document {document_id} registered successfully")

        return metadata

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata

        Args:
            document_id: Document identifier

        Returns:
            DocumentMetadata or None if not found
        """
        registry = self._load_registry()

        if document_id not in registry:
            logger.warning(f"Document {document_id} not found in registry")
            return None

        doc_data = registry[document_id]

        return DocumentMetadata(
            document_id=doc_data["document_id"],
            filename=doc_data["filename"],
            upload_date=datetime.fromisoformat(doc_data["upload_date"]),
            total_pages=doc_data["total_pages"],
            total_chunks=doc_data["total_chunks"],
            collection_name=doc_data["collection_name"]
        )

    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all registered documents

        Returns:
            List of DocumentMetadata objects
        """
        registry = self._load_registry()

        documents = []
        for doc_id, doc_data in registry.items():
            documents.append(DocumentMetadata(
                document_id=doc_data["document_id"],
                filename=doc_data["filename"],
                upload_date=datetime.fromisoformat(doc_data["upload_date"]),
                total_pages=doc_data["total_pages"],
                total_chunks=doc_data["total_chunks"],
                collection_name=doc_data["collection_name"]
            ))

        return documents

    def delete_document(self, document_id: str) -> bool:
        """
        Delete document from registry

        Args:
            document_id: Document identifier

        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting document from registry: {document_id}")

        registry = self._load_registry()

        if document_id in registry:
            del registry[document_id]
            self._save_registry(registry)
            logger.info(f"Document {document_id} deleted from registry")
            return True
        else:
            logger.warning(f"Document {document_id} not found in registry")
            return False

    def document_exists(self, document_id: str) -> bool:
        """
        Check if document exists in registry

        Args:
            document_id: Document identifier

        Returns:
            True if document exists
        """
        registry = self._load_registry()
        return document_id in registry

    def get_document_id_by_filename(self, filename: str) -> str:
        """
        Get document ID by filename

        Args:
            filename: Original filename

        Returns:
            Document ID or None if not found
        """
        registry = self._load_registry()

        for doc_id, doc_data in registry.items():
            if doc_data.get("filename") == filename:
                logger.info(f"Found document ID {doc_id} for filename {filename}")
                return doc_id

        logger.warning(f"No document found with filename {filename}")
        return ''
