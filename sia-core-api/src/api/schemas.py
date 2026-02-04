"""
Pydantic schemas for request/response models and error handling.

Author: Lorena Calvo-Bartolomé
Date: 04/02/2026
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


# ======================================================
# Enums
# ======================================================
class QueryOperator(str, Enum):
    """Query operators for Solr queries."""
    AND = "AND"
    OR = "OR"


class ProcessingModule(str, Enum):
    """Available processing modules."""
    INGESTION = "ingestion"
    SUMMARIZATION = "summarization"
    TOPIC_MODELING = "topic_modeling"
    EMBEDDINGS = "embeddings"
    ALL = "all"


class DataSource(str, Enum):
    """Available data sources for ingestion."""
    PLACE = "PLACE"
    TED = "TED"
    BDNS = "BDNS"


# ======================================================
# Base Response Models
# ======================================================
class BaseResponse(BaseModel):
    """Base response model with success indicator."""
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response."""
    success: bool = False
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Collection not found",
                "error_code": "NOT_FOUND",
                "details": {"collection": "my_collection"}
            }
        }


class PaginatedResponse(BaseModel):
    """Base paginated response model."""
    success: bool = True
    data: List[Any]
    total: Optional[int] = None
    start: int = 0
    rows: int = 10


class DataResponse(BaseModel):
    """Generic response with data field."""
    success: bool = True
    data: Any


# ======================================================
# Health/Ping Response
# ======================================================
class PingResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = "pong"
    timestamp: str
    service: str = "NP Tools API"


class HealthResponse(BaseModel):
    """Response for detailed health check endpoint."""
    status: str
    timestamp: str
    service: str = "NP Tools API"
    version: str = "2.0.0"
    components: Dict[str, Any]


# ======================================================
# INFRASTRUCTURE ADMINISTRATION SERVICES
# ======================================================
class CollectionResponse(BaseResponse):
    """Response for collection operations."""
    collection: Optional[str] = None


class CollectionListResponse(BaseResponse):
    """Response for listing collections."""
    collections: List[str] = []


class SolrQueryParams(BaseModel):
    """Parameters for raw Solr queries."""
    collection: str = Field(..., description="Collection name to query")
    q: str = Field(..., description="Query string using standard query syntax")
    q_op: Optional[QueryOperator] = Field(
        None, alias="q.op", description="Default operator (AND/OR)")
    fq: Optional[str] = Field(None, description="Filter query")
    sort: Optional[str] = Field(
        None, description="Sort order (e.g., 'field asc')")
    start: Optional[int] = Field(0, ge=0, description="Offset for pagination")
    rows: Optional[int] = Field(
        10, ge=1, le=1000, description="Number of rows to return")
    fl: Optional[str] = Field(None, description="Fields to return")
    df: Optional[str] = Field(None, description="Default field")

    class Config:
        populate_by_name = True



class MetadataFieldsResponse(BaseResponse):
    """Response for metadata fields."""
    fields: List[str] = []


class DisplayConfigResponse(BaseResponse):
    """Response for complete display configuration."""
    metadata_displayed: List[str] = []
    searchable_fields: List[str] = []
    active_filters: List[str] = []


class CorpusListResponse(BaseResponse):
    """Response for listing corpora."""
    corpora: List[str] = []


class ModelsListResponse(BaseResponse):
    """Response for listing models."""
    models: Dict[str, List[Dict[str, int]]] = {}


class CorpusModelsResponse(BaseResponse):
    """Response for corpus models."""
    models: Dict[str, List[Dict[str, int]]] = {}


# ======================================================
# ENRICHMENT & INGESTION SERVICES
# ======================================================
class DownloadRequest(BaseModel):
    """Request for bulk data download."""
    source: DataSource = Field(..., description="Data source")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class TextExtractionRequest(BaseModel):
    """Request for PDF text extraction and normalization."""
    document_ids: List[str] = Field(..., description="Document IDs to process")
    normalize: bool = Field(True, description="Apply text normalization")


class SummarizationRequest(BaseModel):
    """Request for document summarization."""
    document_ids: List[str] = Field(..., description="Document IDs to summarize")
    focus_dimensions: Optional[List[str]] = Field(None, description="Focus dimensions for summary")
    include_traceability: bool = Field(True, description="Include traceability to original text")


class MetadataExtractionRequest(BaseModel):
    """Request for automatic metadata extraction."""
    document_ids: List[str] = Field(..., description="Document IDs to process")
    metadata_fields: List[str] = Field(..., description="Metadata fields to extract")
    validation: bool = Field(True, description="Validate extracted metadata")


class AIRelevanceRequest(BaseModel):
    """Request for AI relevance classification."""
    document_ids: List[str] = Field(..., description="Document IDs to classify")
    output_format: str = Field("binary", description="Output format: 'binary' or 'score'")


class IngestionRequest(BaseModel):
    """Request for triggering data ingestion pipeline."""
    corpus_name: str = Field(..., description="Name for the corpus")
    source_path: Optional[str] = Field(None, description="Path to source data")
    extract_text: bool = Field(True, description="Extract text from PDFs")
    normalize: bool = Field(True, description="Apply text normalization")
    modules: List[ProcessingModule] = Field(
        default=[ProcessingModule.ALL],
        description="Processing modules to execute"
    )
    async_execution: bool = Field(
        default=True,
        description="Execute asynchronously"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "corpus_name": "procurement_2024",
                "extract_text": True,
                "normalize": True,
                "modules": ["ingestion", "embeddings"],
                "async_execution": True
            }
        }


class IngestionResponse(BaseResponse):
    """Response for ingestion operations."""
    job_id: Optional[str] = None
    status: str = "queued"
    documents_processed: int = 0


class TopicModelTrainingRequest(BaseModel):
    """Request for topic model training."""
    corpus_name: str = Field(..., description="Corpus for topic modeling")
    model_name: str = Field(..., description="Name for the model")
    num_topics: int = Field(
        10, ge=2, description="Number of topics")


class EmbeddingsGenerationRequest(BaseModel):
    """Request for embeddings generation."""
    corpus_name: str = Field(..., description="Corpus for embedding generation")
    model_type: str = Field("bert", description="Embedding model: bert, sentence-transformers")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs or None for all")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size for processing")


class SummarizationRequest(BaseModel):
    """Request for document summarization."""
    corpus_name: str = Field(..., description="Corpus to summarize")
    model_type: str = Field("default", description="Summarization model")
    max_length: Optional[int] = Field(
        None, description="Maximum summary length")


class ProcessingJobResponse(BaseResponse):
    """Response for processing jobs."""
    job_id: Optional[str] = None
    status: str = "queued"
    progress: Optional[float] = None
    results: Optional[Dict[str, Any]] = None


# --- On-Demand Inference (External Documents) ---
class OnDemandInferenceRequest(BaseModel):
    """
    Request for on-demand inference on external (non-indexed) documents.
    Allows real-time processing and comparison with existing index.
    """
    text: str = Field(..., description="Text/document to process")
    operations: List[str] = Field(
        default=["embeddings"],
        description="Operations: embeddings, topics, summary, all"
    )
    model_name: Optional[str] = Field(
        None, description="Model name (required for topic inference)")
    compare_with_index: bool = Field(
        True,
        description="Compare results with existing index to find similar documents"
    )
    corpus_collection: Optional[str] = Field(
        None,
        description="Corpus collection to compare against"
    )
    start: Optional[int] = Field(
        0, ge=0, description="Offset for comparison results")
    rows: Optional[int] = Field(
        10, description="Number of similar documents to return")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Suministro de equipos informáticos para oficinas...",
                "operations": ["embeddings", "topics"],
                "cpv": "30",
                "granularity": "high",
                "compare_with_index": True,
                "corpus_collection": "np_corpus",
                "rows": 10
            }
        }


class OnDemandInferenceResponse(BaseResponse):
    """Response for on-demand inference."""
    embeddings: Optional[List[float]] = None
    topic_distribution: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    similar_documents: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: Optional[int] = None


# ======================================================
# EXPLOITATION SERVICES
# ======================================================
class MetadataSearchQuery(BaseModel):
    """Query for exact metadata search."""
    corpus_collection: str = Field(...,
                                   description="Corpus collection to search")
    field: str = Field(..., description="Metadata field to search")
    value: str = Field(..., description="Value to search for")
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(10, description="Number of rows")


class TextSearchQuery(BaseModel):
    """Query for text search in searchable fields."""
    corpus_collection: str = Field(..., description="Corpus collection")
    query_string: str = Field(..., description="Search string")
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(None, description="Number of rows")


class TopicLabelsQuery(BaseModel):
    """Query for topic labels."""
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(None, description="Number of rows")


class TopicDocumentsQuery(BaseModel):
    """Query for documents by topic."""
    corpus_collection: str = Field(..., description="Corpus collection")
    topic_id: str = Field(..., description="Topic ID")
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(None, description="Number of rows")


class TopicInfoResponse(BaseResponse):
    """Response for topic information."""
    topics: List[Dict[str, Any]] = []
    model_info: Optional[Dict[str, Any]] = None


class SemanticSearchQuery(BaseModel):
    """Query for semantic similarity search."""
    corpus_collection: str = Field(..., description="Corpus collection")
    query_text: str = Field(...,
                            description="Text to search for similar documents")
    search_method: str = Field(
        "embeddings",
        description="Method: embeddings (BERT), topic_model, word2vec"
    )
    keyword_filter: Optional[str] = Field(
        None, description="Optional keyword filter")
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(None, description="Number of rows")

    class Config:
        json_schema_extra = {
            "example": {
                "corpus_collection": "np_corpus",
                "query_text": "Contrato de suministro de material de oficina",
                "search_method": "embeddings",
                "rows": 10
            }
        }


class SimilarDocumentsQuery(BaseModel):
    """Query for finding similar documents by document ID."""
    corpus_collection: str = Field(..., description="Corpus collection")
    doc_id: str = Field(..., description="Reference document ID")
    model_name: str = Field(...,
                            description="Model for similarity computation")
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(None, description="Number of rows")


class DocumentMetadataResponse(BaseResponse):
    """Response for document metadata."""
    doc_id: str
    metadata: Dict[str, Any] = {}


class DocumentThetasResponse(BaseResponse):
    """Response for document-topic distribution."""
    doc_id: str
    model_name: str
    thetas: List[Dict[str, Any]] = []


class IndicatorsResponse(BaseResponse):
    """Response for corpus indicators."""
    total_documents: int = 0
    years_available: List[int] = []
    document_count_by_year: Optional[Dict[int, int]] = None
    topics_distribution: Optional[Dict[str, Any]] = None


class TemporalSearchQuery(BaseModel):
    """Query for temporal/year-based search."""
    corpus_collection: str = Field(..., description="Corpus collection")
    year: Optional[int] = Field(None, description="Specific year")
    start_year: Optional[int] = Field(None, description="Start of range")
    end_year: Optional[int] = Field(None, description="End of range")
    sort_by: str = Field("date:desc", description="Sort specification")
    keyword: Optional[str] = Field("*", description="Keyword filter")
    searchable_field: Optional[str] = Field("*", description="Field to search")
    start: Optional[int] = Field(0, ge=0, description="Offset")
    rows: Optional[int] = Field(None, description="Number of rows")


class RecommendationQuery(BaseModel):
    """Query for document recommendations."""
    corpus_collection: str = Field(..., description="Corpus collection")
    doc_id: Optional[str] = Field(
        None, description="Document ID for similarity-based")
    query_text: Optional[str] = Field(
        None, description="Text for semantic recommendations")
    recommendation_type: str = Field(
        "similar",
        description="Type: similar (by doc_id), semantic (by text), topic_based"
    )
    limit: int = Field(10, ge=1, le=100, description="Max recommendations")


class RecommendationResponse(BaseResponse):
    """Response for recommendations."""
    recommendations: List[Dict[str, Any]] = []
    total: int = 0
    recommendation_type: str = ""


class SearchResponse(BaseResponse):
    """Generic response for search operations."""
    results: List[Dict[str, Any]] = []
    total: int = 0
    start: int = 0
    rows: int = 10
    query_time_ms: Optional[int] = None
