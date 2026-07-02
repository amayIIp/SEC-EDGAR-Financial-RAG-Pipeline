from __future__ import annotations 
import re 
from typing import List 
import numpy as np 
import tiktoken 
from sentence_transformers import SentenceTransformer 
from src.chunking.base_chunker import BaseChunker 
from src.shared.config import cfg 
from src.shared.models import Chunk, ChunkingStrategy, ParsedFiling, ParsedSection 
class SemanticChunker(BaseChunker):
    """
    Splits text sections into sentences, embeds them, and splits where semantic similarity
    between adjacent sentences falls below a threshold.
    """
    def __init__(self, similarity_threshold: float = None, min_tokens: int = None) -> None:
        self.threshold = similarity_threshold if similarity_threshold is not None            else cfg.chunking.semantic.similarity_threshold
        self.min_tokens = min_tokens if min_tokens is not None            else cfg.chunking.semantic.min_chunk_tokens
        self.max_tokens = cfg.chunking.semantic.max_chunk_tokens
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
        self.model = SentenceTransformer(cfg.chunking.semantic.sentence_model)
        self.sentence_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits a text paragraph into a list of clean, non-empty sentences.
        """
        raw_sentences = self.sentence_regex.split(text)
        return [s.strip() for s in raw_sentences if s.strip()]
    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes the cosine similarity between two vectors.
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))
    def _group_sentences_semantically(self, sentences: List[str]) -> List[str]:
        """
        Encodes sentences and groups them by semantic similarity boundaries.
        """
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]
        embeddings = self.model.encode(
            sentences,
            batch_size=cfg.chunking.semantic.sentence_encode_batch_size,
            show_progress_bar=False
        )
        groups: List[str] = [] 
        current_group: List[str] = [sentences[0]] 
        for i in range(len(sentences) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i+1])
            if similarity < self.threshold:
                groups.append(" ".join(current_group))
                current_group = [sentences[i+1]]
            else:
                current_group.append(sentences[i+1])
        if current_group:
            groups.append(" ".join(current_group))
        return groups
    def _merge_small_chunks(self, groups: List[str]) -> List[str]:
        """
        Merges adjacent chunks that are smaller than min_tokens.
        """
        merged_chunks: List[str] = []
        current_chunk = ""
        for group in groups:
            if not current_chunk:
                current_chunk = group
                continue
            curr_tokens = len(self.encoder.encode(current_chunk))
            group_tokens = len(self.encoder.encode(group))
            if curr_tokens < self.min_tokens and (curr_tokens + group_tokens) <= self.max_tokens:
                current_chunk += " " + group
            else:
                merged_chunks.append(current_chunk)
                current_chunk = group
        if current_chunk:
            merged_chunks.append(current_chunk)
        return merged_chunks
    def chunk(self, parsed_filing: ParsedFiling) -> List[Chunk]:
        """
        Chunks the parsed filing using semantic boundary calculations.
        """
        chunks: List[Chunk] = [] 
        chunk_index = 0 
        for section in parsed_filing.sections:
            if section.content.strip():
                sentences = self._split_into_sentences(section.content)
                raw_groups = self._group_sentences_semantically(sentences)
                semantic_splits = self._merge_small_chunks(raw_groups)
                for split in semantic_splits:
                    context_prefix = f"[{section.item_id}. {section.title}]\n\n"
                    fused_text = context_prefix + split
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.SEMANTIC.value,
                        chunk_index=chunk_index
                    )
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=fused_text,
                        strategy=ChunkingStrategy.SEMANTIC,
                        chunk_index=chunk_index,
                        is_table=False,
                        ticker=parsed_filing.metadata.ticker,
                        cik=parsed_filing.metadata.cik,
                        filing_type=parsed_filing.metadata.filing_type,
                        filing_date=parsed_filing.metadata.filing_date,
                        fiscal_year=parsed_filing.metadata.fiscal_year,
                        accession_number=parsed_filing.metadata.accession_number,
                        section=section.item_id,
                        section_title=section.title,
                        token_count=len(self.encoder.encode(fused_text))
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            for table in section.tables:
                caption = table.get("caption", "").strip() or "Financial Table"
                table_text = f"[{section.item_id}. {section.title}] - Table: {caption}\n\n{table.get('raw_text', '')}"
                if table.get("raw_text", "").strip():
                    chunk_id = Chunk.make_id(
                        ticker=parsed_filing.metadata.ticker,
                        accession_number=parsed_filing.metadata.accession_number,
                        strategy=ChunkingStrategy.SEMANTIC.value,
                        chunk_index=chunk_index
                    )
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        text=table_text,
                        strategy=ChunkingStrategy.SEMANTIC,
                        chunk_index=chunk_index,
                        is_table=True,
                        ticker=parsed_filing.metadata.ticker,
                        cik=parsed_filing.metadata.cik,
                        filing_type=parsed_filing.metadata.filing_type,
                        filing_date=parsed_filing.metadata.filing_date,
                        fiscal_year=parsed_filing.metadata.fiscal_year,
                        accession_number=parsed_filing.metadata.accession_number,
                        section=section.item_id,
                        section_title=section.title,
                        token_count=len(self.encoder.encode(table_text))
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        return chunks
