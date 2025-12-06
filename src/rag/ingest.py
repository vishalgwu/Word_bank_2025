from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

def build_qdrant_index(collection_name, summaries):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embedded local Qdrant
    client = QdrantClient(path="qdrant_local", prefer_grpc=False)

    # Delete old collection if exists
    try:
        client.delete_collection(collection_name)
    except:
        pass

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,      # MiniLM embedding size
            distance=Distance.COSINE
        ),
    )

    points = []
    for i, text in enumerate(summaries):
        vec = model.encode(text).tolist()
        points.append(
            PointStruct(
                id=i,                # numeric ID (IMPORTANT)
                vector=vec,
                payload={"text": text}
            )
        )

    client.upsert(collection_name=collection_name, points=points)

    return client
