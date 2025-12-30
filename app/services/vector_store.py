"""ChromaDB vector store service"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for document embeddings"""

    def __init__(self):
        """Initialize ChromaDB client and embeddings"""
        logger.info(f"Initializing ChromaDB with persist directory: {settings.CHROMA_PERSIST_DIR}")

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )

        # Initialize OpenAI embeddings
        # self.embeddings = OpenAIEmbeddings(
        #     model=settings.EMBEDDING_MODEL,
        #     openai_api_key=settings.OPENAI_API_KEY
        # )

        self.embeddings = SentenceTransformer(settings.EMBEDDING_MODEL)
    def create_collection(self, document_id: str) -> chromadb.Collection:
        """
        Create or get a collection for a document

        Args:
            document_id: Unique identifier for the document

        Returns:
            ChromaDB collection instance
        """
        collection_name = f"doc_{document_id}" # collection per document
        logger.info(f"Creating/getting collection: {collection_name}")

        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"document_id": document_id}
            )
            return collection
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def get_collection(self, document_id: str) -> Optional[chromadb.Collection]:
        """
        Get existing collection for a document

        Args:
            document_id: Unique identifier for the document

        Returns:
            ChromaDB collection or None if not found
        """
        collection_name = f"doc_{document_id}"

        try:
            collection = self.client.get_collection(name=collection_name)
            return collection
        except Exception as e:
            logger.warning(f"Collection not found for document {document_id}: {e}")
            return None

    def add_documents(
        self,
        collection: chromadb.Collection,
        chunks: List[Dict],
        document_id: str
    ) -> int:
        """
        Add document chunks to collection with embeddings

        Args:
            collection: ChromaDB collection
            chunks: List of text chunks with metadata
            document_id: Document identifier

        Returns:
            Number of chunks added
        """
        logger.info(f"Adding {len(chunks)} chunks to collection")

        try:
            # Get the texts from chunks
            texts = [chunk["text"] for chunk in chunks]
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "section": chunk["metadata"].get("section", ""),
                    "page": chunk["metadata"].get("page", 0),
                    "chunk_index": i,
                    "hierarchy_level": chunk["metadata"].get("hierarchy_level", 1)
                }
                metadatas.append(metadata)

                # Generate unique ID
                ids.append(f"{document_id}_chunk_{i}")

            # Generate embeddings
            logger.info("Generating embeddings...")
            #embeddings = self.embeddings.embed_documents(texts)
            embeddings = self.embeddings.encode(texts).tolist()

            # Add to ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully added {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def similarity_search(
        self,
        collection: chromadb.Collection,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Perform similarity search on collection

        Args:
            collection: ChromaDB collection
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search results with text and metadata
        """
        k =  settings.RETRIEVAL_TOP_K or top_k
        logger.info(f"Performing similarity search with query: {query[:50]}...")

        try:
            # Generate query embedding
            #query_embedding = self.embeddings.embed_query(query)
            query_embedding = self.embeddings.encode(query).tolist()
            # Search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0
                    })

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

    def list_collections(self) -> List[str]:
        """
        List all collections in the database

        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def delete_collection(self, document_id: str) -> bool:
        collection_name = f"doc_{document_id}"
        logger.info(f"Deleting collection: {collection_name}")

        try:
            self.client.delete_collection(name=collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def get_collection_count(self, collection: chromadb.Collection) -> int:
        try:
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    def health_check(self) -> bool:
        """
        Check if ChromaDB is healthy

        Returns:
            True if healthy
        """
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False()
