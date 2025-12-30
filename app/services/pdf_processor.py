"""PDF processing service for text extraction and chunking"""

import pdfplumber
from typing import List, Dict, Tuple
from landingai_ade import LandingAIADE

from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF chunking using LandingAI DPT or legacy fallback"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def _chunk_with_landingai(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """
        Use LandingAI's DPT for intelligent semantic chunking

        Returns:
            Tuple of (chunks, total_pages)
        """
        try:
            

            logger.info(f"Using LandingAI DPT for chunking: {pdf_path}")

            # Initialize LandingAI client
            ade = LandingAIADE() # Agentic Document Extraction client

            # Parse PDF with DPT
            response = ade.parse(
                document_url=pdf_path,
                model=settings.LANDINGAI_DPT_MODEL
            )

            chunks = []
            total_pages = 0

            # Process chunks from DPT response
            if hasattr(response, 'chunks') and response.chunks: 
                for i, chunk in enumerate(response.chunks):
                    # Extract chunk type and content
                    chunk_type = getattr(chunk, 'type', 'text')
                    # Use markdown attribute for text content
                    chunk_text = getattr(chunk, 'markdown', '')

                    # Get page number from grounding
                    page_num = 1
                    if hasattr(chunk, 'grounding') and chunk.grounding:
                        page_num = getattr(
                            chunk.grounding, 'page', 0
                        ) + 1  # 0-indexed

                    # Only use text, table, and figure chunks for Q&A
                    if chunk_type in ['text', 'table','figure'] and chunk_text.strip():
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "section": f"Chunk {i+1} ({chunk_type})",
                                "page": page_num,
                                "chunk_index": i,
                                "hierarchy_level": 1,
                                "chunk_type": chunk_type,
                                "source": "landingai_dpt"
                            }
                        })

                        # Track max page number
                        total_pages = max(total_pages, page_num)

            logger.info(
                f"LandingAI DPT created {len(chunks)} semantic chunks"
            )
            return chunks, total_pages

        except Exception as e:
            logger.error(f"Error using LandingAI DPT: {e}")
            raise

    def _extract_text_simple(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """
        Simple PDF text extraction (legacy fallback)

        Returns:
            Tuple of (pages_data, total_pages)
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")

        pages_data = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_data.append({
                            "page_number": page_num,
                            "text": page_text
                        })

                return pages_data, total_pages

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise

    def _chunk_by_pages(self, pages_data: List[Dict]) -> List[Dict]:
        """
        Chunk document by pages with smart text splitting

        This is the fallback chunking strategy
        """
        chunks = []

        for page in pages_data:
            page_text = page["text"]
            page_num = page["page_number"]

            if len(page_text) > self.chunk_size:
                # Split large pages into smaller chunks
                sub_chunks = self._split_text(page_text)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        "text": sub_chunk,
                        "metadata": {
                            "section": f"Page {page_num}",
                            "page": page_num,
                            "chunk_index": i,
                            "hierarchy_level": 1
                        }
                    })
            else:
                # Keep small pages as single chunks
                chunks.append({
                    "text": page_text.strip(),
                    "metadata": {
                        "section": f"Page {page_num}",
                        "page": page_num,
                        "chunk_index": 0,
                        "hierarchy_level": 1
                    }
                })

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap at sentence boundaries"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence ending
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """
        Main processing pipeline

        Returns:
            Tuple of (chunks, total_pages)
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Check if LandingAI chunking is enabled
        if (settings.USE_LANDINGAI_CHUNKING and
                settings.VISION_AGENT_API_KEY):
            logger.info("Using LandingAI DPT for intelligent chunking")
            try:
                # Set API key as environment variable
                os.environ['VISION_AGENT_API_KEY'] = (
                    settings.VISION_AGENT_API_KEY
                )

                chunks, total_pages = self._chunk_with_landingai(pdf_path)
                return chunks, total_pages

            except Exception as e:
                logger.warning(
                    f"LandingAI DPT failed: {e}. "
                    f"Falling back to legacy chunking"
                )
                # Fall through to legacy chunking

        # Legacy chunking (fallback or default)
        logger.info("Using legacy page-based chunking")

        # Extract text ( fallback if LandingAI fails )
        pages_data, total_pages = self._extract_text_simple(pdf_path)

        # Create chunks
        chunks = self._chunk_by_pages(pages_data)

        return chunks, total_pages
