import os
import logging
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

logger = logging.getLogger(__name__)

# Global state
_embedding_model: Optional[SentenceTransformer] = None
_pinecone_index = None
_initialized = False


def initialize() -> bool:
    global _embedding_model, _pinecone_index, _initialized

    api_key = os.getenv('PINECONE_API_KEY')

    if not api_key:
        logger.warning("PINECONE_API_KEY not set - RAG disabled")
        _initialized = False
        return False

    try:
        logger.info("Initializing RAG retriever...")
        logger.info("Loading embedding model (all-mpnet-base-v2)...")
        _embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=api_key)
        _pinecone_index = pc.Index("dpo-qa-index")

        _initialized = True
        logger.info("RAG retriever initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize RAG retriever: {e}")
        _initialized = False
        return False


def search(query: str, namespace: str, top_k: int = 3) -> List[Dict[str, str]]:
    if not _initialized or _embedding_model is None or _pinecone_index is None:
        logger.debug("RAG not initialized - returning empty results")
        return []

    try:
        query_embedding = _embedding_model.encode(query, convert_to_numpy=True).tolist()

        results = _pinecone_index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )

        context_items = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            if 'question' in metadata and 'answer' in metadata:
                context_items.append({
                    'question': metadata['question'],
                    'answer': metadata['answer']
                })

        logger.info(f"Retrieved {len(context_items)} context items from namespace '{namespace}'")
        return context_items

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return []
