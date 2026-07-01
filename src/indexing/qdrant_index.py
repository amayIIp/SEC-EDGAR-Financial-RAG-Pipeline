# src/indexing/qdrant_index.py
# This module manages the Qdrant vector database storage and indexing.
#
# =========================================================================================
# Advanced Concept: Qdrant Vector Storage and Filtering
# Qdrant is a specialized vector database designed to hold high-dimensional vectors.
# 1. Collection: Equivalent to a table in SQL. It groups vectors of a fixed size.
#    Since dimensions differ (OpenAI is 1536, BGE is 1024), we keep them in separate collections.
# 2. Point: A record containing a unique UUID, the vector embedding, and a "payload" (metadata).
# 3. Payload Indexes: By default, filtering a vector query on metadata (like ticker = "AAPL")
#    requires scanning every point. By building payload indexes in Qdrant on fields like
#    ticker, CIK, and filing_date, Qdrant can filter using inverted indices before executing
#    the vector distance math, maintaining sub-millisecond retrieval speeds.
# 4. Idempotency: We convert the string chunk_id into a deterministic UUIDv5 so that
#    re-indexing updating the point in-place rather than creating duplicates.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
import uuid # Standard library module to generate deterministic UUIDs.
from typing import Any, List # Type helpers.
from qdrant_client import QdrantClient # Official Qdrant Python SDK client.
from qdrant_client.http import models as qmodels # Object models used by Qdrant.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger setup.
from src.shared.models import Chunk # Shared models.

log = get_logger(__name__)

class QdrantIndexManager:
    """
    Manages collection creation, payload indexing, and batch vector uploads in Qdrant.
    """

    def __init__(self) -> None:
        # Establish connection client with Qdrant server.
        self.client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        self.collection_name = cfg.qdrant.collection_name

    def create_collection(self, dimension: int, force: bool = False) -> None:
        """
        Creates the Qdrant collection sized to the embedding dimension if it does not exist.
        If force=True, deletes the existing collection first.
        """
        # Delete existing collection if force deletion is requested.
        if force and self.client.collection_exists(collection_name=self.collection_name):
            log.info("deleting_existing_qdrant_collection", collection=self.collection_name)
            self.client.delete_collection(collection_name=self.collection_name)

        # Skip creation if the collection already exists.
        if self.client.collection_exists(collection_name=self.collection_name):
            log.info("qdrant_collection_exists", collection=self.collection_name)
            return

        # Map distance metric configuration string to Qdrant models Distance object.
        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "dot": qmodels.Distance.DOT,
            "euclidean": qmodels.Distance.EUCLIDEAN
        }
        # Fetch the distance metric from config. Default to cosine.
        q_distance = distance_map.get(cfg.qdrant.distance.lower(), qmodels.Distance.COSINE)

        log.info(
            "creating_qdrant_collection",
            collection=self.collection_name,
            dimension=dimension,
            distance=cfg.qdrant.distance
        )

        # Create the collection with standard configuration.
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

        # Build payload indexes on filterable keyword fields to speed up queries.
        for field in ["ticker", "cik", "filing_type", "filing_date", "fiscal_year"]:
            # Deduce appropriate index schema type.
            schema_type = (
                qmodels.PayloadSchemaType.INTEGER
                if field == "fiscal_year"
                else qmodels.PayloadSchemaType.KEYWORD
            )
            # Create payload index.
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

        # Ensure lengths match.
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must equal the number of vectors.")

        points: List[qmodels.PointStruct] = []
        
        # Loop through chunks to construct PointStruct records.
        for i, chunk in enumerate(chunks):
            # Generate a deterministic UUIDv5 based on the chunk_id string.
            # Qdrant requires IDs to be UUIDs or 64-bit integers.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            # Map Chunk fields to payload dictionary.
            payload = chunk.model_dump()
            
            # Create the point.
            point = qmodels.PointStruct(
                id=point_id,
                vector=vectors[i],
                payload=payload
            )
            points.append(point)

        # Upload the batch of points to Qdrant.
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        log.info("qdrant_bulk_upsert_complete", count=len(points))
