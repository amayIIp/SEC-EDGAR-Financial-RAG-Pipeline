from __future__ import annotations 
import uuid 
from typing import Any, List 
from qdrant_client import QdrantClient 
from qdrant_client.http import models as qmodels 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import Chunk 
log = get_logger(__name__)
class QdrantIndexManager:
    """
    Manages collection creation, payload indexing, and batch vector uploads in Qdrant.
    """
    def __init__(self) -> None:
        self.client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        self.collection_name = cfg.qdrant.collection_name
    def create_collection(self, dimension: int, force: bool = False) -> None:
        """
        Creates the Qdrant collection sized to the embedding dimension if it does not exist.
        If force=True, deletes the existing collection first.
        """
        existing_collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in existing_collections)
        if force and collection_exists:
            log.info("deleting_existing_qdrant_collection", collection=self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)
            collection_exists = False
        if collection_exists:
            log.info("qdrant_collection_exists", collection=self.collection_name)
            return
        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "dot": qmodels.Distance.DOT,
            "euclidean": qmodels.Distance.EUCLID
        }
        q_distance = distance_map.get(cfg.qdrant.distance.lower(), qmodels.Distance.COSINE)
        log.info(
            "creating_qdrant_collection",
            collection=self.collection_name,
            dimension=dimension,
            distance=cfg.qdrant.distance
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=q_distance
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=cfg.qdrant.hnsw_m,
                ef_construct=cfg.qdrant.hnsw_ef_construct
            )
        )
        for field in ["ticker", "cik", "filing_type", "filing_date", "fiscal_year"]:
            schema_type = (
                qmodels.PayloadSchemaType.INTEGER
                if field == "fiscal_year"
                else qmodels.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema_type
            )
        log.info("qdrant_collection_created_successfully", collection=self.collection_name)
    def bulk_upsert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """
        Uploads a batch of Chunk models and their corresponding embedding vectors to Qdrant.
        """
        if not chunks or not vectors:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must equal the number of vectors.")
        points: List[qmodels.PointStruct] = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = chunk.model_dump()
            point = qmodels.PointStruct(
                id=point_id,
                vector=vectors[i],
                payload=payload
            )
            points.append(point)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        log.info("qdrant_bulk_upsert_complete", count=len(points))
