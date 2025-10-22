import os
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


def load_data(filepath: str) -> Tuple[Dict[str, List], int]:
    """Load and group Q&A items by knowledge base"""
    print(f"Loading {filepath}...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON array of items")

    grouped = defaultdict(list)
    for item in data:
        try:
            kb_name = item['question']['dataset']
            grouped[kb_name].append(item)
        except KeyError as e:
            print(f"Warning: Skipping malformed item - missing key: {e}")
            continue

    total = sum(len(items) for items in grouped.values())
    print(f"Found {total} items across {len(grouped)} knowledge bases\n")

    return dict(grouped), total


def initialize_embedding_model() -> SentenceTransformer:
    """Load sentence transformer model"""
    print("Initializing embedding model (all-mpnet-base-v2)...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("Model loaded successfully\n")
    return model


def connect_to_pinecone(api_key: str) -> any:
    """Connect to Pinecone index"""
    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=api_key)
    index = pc.Index("dpo-qa-index")

    print("Connected to index: dpo-qa-index\n")
    return index


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """Generate embeddings for a batch of texts"""
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def create_vectors(items: List[Dict], kb_name: str, model: SentenceTransformer) -> List[Tuple]:
    """Create vector tuples (id, embedding, metadata) for all items in a KB"""
    print(f"  Generating embeddings...", end=" ", flush=True)

    vectors = []
    questions = []

    for item in items:
        try:
            question_text = item['question']['full_text']
            questions.append(question_text)
        except KeyError as e:
            print(f"\n  Warning: Skipping item - missing key: {e}")
            continue

    embeddings = generate_embeddings(questions, model)
    print("Done")

    for i, item in enumerate(items):
        try:
            question_id = item['question']['id']
            question_text = item['question']['full_text']
            answer_text = item.get('answer_0', '')

            vector_id = f"{kb_name}_{question_id}"
            embedding = embeddings[i]
            metadata = {
                "question": question_text,
                "answer": answer_text,
                "dataset": kb_name
            }

            vectors.append((vector_id, embedding, metadata))
        except (KeyError, IndexError) as e:
            print(f"\n  Warning: Skipping item - error: {e}")
            continue

    return vectors


def upload_in_batches(index, vectors: List[Tuple], kb_name: str, batch_size: int = 100):
    """Upload vectors to Pinecone in batches"""
    total_batches = (len(vectors) + batch_size - 1) // batch_size

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        batch_count = len(batch)

        print(f"  Uploading batch {batch_num}/{total_batches} ({batch_count} vectors)...", end=" ", flush=True)

        try:
            index.upsert(
                vectors=batch,
                namespace=kb_name
            )
            print("Done")
        except Exception as e:
            print(f"Failed - {e}")
            raise


def main():
    start_time = time.time()

    # Check environment variable
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("ERROR: PINECONE_API_KEY environment variable not set")
        print("Please set it with: export PINECONE_API_KEY='your-api-key'")
        return

    try:
        # Load data
        grouped_data, total_items = load_data('sample_data.json')

        # Initialize model
        model = initialize_embedding_model()

        # Connect to Pinecone
        index = connect_to_pinecone(api_key)

        # Process each knowledge base
        summary = {}
        for kb_name in sorted(grouped_data.keys()):
            items = grouped_data[kb_name]
            item_count = len(items)

            print(f"Processing {kb_name} ({item_count} items)...")

            # Create vectors with embeddings
            vectors = create_vectors(items, kb_name, model)

            # Upload in batches
            upload_in_batches(index, vectors, kb_name)

            summary[kb_name] = len(vectors)
            print(f"  {kb_name} complete: {len(vectors)} vectors uploaded\n")

        # Display final summary
        elapsed = time.time() - start_time
        print("=" * 50)
        print("SUMMARY:")
        for kb_name, count in sorted(summary.items()):
            print(f"  {kb_name}: {count} vectors")
        print()
        print(f"TOTAL: {sum(summary.values())} vectors uploaded to Pinecone")
        print(f"Time taken: {elapsed:.1f} seconds")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}")
        raise


if __name__ == "__main__":
    main()
