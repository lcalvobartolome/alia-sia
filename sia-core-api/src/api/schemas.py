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
# Health Check Response
# ======================================================
class HealthResponse(BaseModel):
    """Health check response including Solr status."""
    status: str = Field(..., description="Overall status: 'healthy' or 'unhealthy'")
    timestamp: str = Field(..., description="ISO timestamp of check")
    solr_connected: bool = Field(..., description="Solr connection status")
    version: str = Field("1.0.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-02-04T10:30:00Z",
                "solr_connected": True,
                "version": "1.0.0"
            }
        }


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


# ======================================================
# TEXT EXTRACTION SCHEMAS
# ======================================================
class TextExtractionBatchRequest(BaseModel):
    """Request for batch PDF text extraction from parquet file."""
    parquet_path: str = Field(..., description="Path to the parquet file containing documents")
    text_column: str = Field("pdf_path", description="Column name containing PDF paths")
    id_column: str = Field("doc_id", description="Column name containing document IDs")
    normalize: bool = Field(True, description="Apply text normalization")
    output_path: Optional[str] = Field(None, description="Path to save extracted text")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/documents/batch_2024.parquet",
                "text_column": "pdf_path",
                "id_column": "doc_id",
                "normalize": True
            }
        }


class TextExtractionSingleRequest(BaseModel):
    """Request for single document PDF text extraction (JSON body, for path or base64)."""
    document_id: Optional[str] = Field(None, description="Document ID (auto-generated if not provided)")
    pdf_path: Optional[str] = Field(None, description="Path to the PDF file (if not in system)")
    pdf_content: Optional[str] = Field(None, description="Base64 encoded PDF content")
    normalize: bool = Field(True, description="Apply text normalization")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "DOC-2024-001",
                "pdf_path": "/data/pdfs/document.pdf",
                "normalize": True
            }
        }


# ======================================================
# SUMMARIZATION SCHEMAS
# ======================================================
class SummarizationBatchRequest(BaseModel):
    """Request for batch document summarization from parquet file."""
    parquet_path: str = Field(..., description="Path to the parquet file containing documents")
    text_column: str = Field("text", description="Column name containing document text")
    id_column: str = Field("doc_id", description="Column name containing document IDs")
    focus_dimensions: Optional[List[str]] = Field(None, description="Focus dimensions for summary")
    include_traceability: bool = Field(True, description="Include traceability to original text")
    output_path: Optional[str] = Field(None, description="Path to save summaries")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/documents/batch_2024.parquet",
                "text_column": "full_text",
                "id_column": "doc_id",
                "focus_dimensions": ["technical", "economic"],
                "include_traceability": True
            }
        }


class SummarizationSingleRequest(BaseModel):
    """Request for single document summarization."""
    document_id: Optional[str] = Field(None, description="Document ID (if indexed)")
    text: Optional[str] = Field(None, description="Raw text to summarize (if not indexed)")
    focus_dimensions: Optional[List[str]] = Field(None, description="Focus dimensions for summary")
    include_traceability: bool = Field(True, description="Include traceability to original text")
    max_length: Optional[int] = Field(None, description="Maximum summary length")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "El presente contrato tiene por objeto...",
                "focus_dimensions": ["technical"],
                "include_traceability": True,
                "max_length": 500
            }
        }


# ======================================================
# METADATA EXTRACTION SCHEMAS
# ======================================================
class MetadataExtractionBatchRequest(BaseModel):
    """Request for batch automatic metadata extraction from parquet file."""
    parquet_path: str = Field(..., description="Path to the parquet file containing documents")
    text_column: str = Field("text", description="Column name containing document text")
    id_column: str = Field("doc_id", description="Column name containing document IDs")
    metadata_fields: List[str] = Field(..., description="Metadata fields to extract")
    validation: bool = Field(True, description="Validate extracted metadata")
    output_path: Optional[str] = Field(None, description="Path to save extracted metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/documents/batch_2024.parquet",
                "text_column": "full_text",
                "id_column": "doc_id",
                "metadata_fields": ["organization", "budget", "deadline"],
                "validation": True
            }
        }


class MetadataExtractionSingleRequest(BaseModel):
    """Request for single document metadata extraction."""
    document_id: Optional[str] = Field(None, description="Document ID (if indexed)")
    text: Optional[str] = Field(None, description="Raw text to extract metadata from (if not indexed)")
    metadata_fields: List[str] = Field(..., description="Metadata fields to extract")
    validation: bool = Field(True, description="Validate extracted metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "El presente contrato tiene por objeto...",
                "metadata_fields": ["organization", "budget", "deadline"],
                "validation": True
            }
        }


# ======================================================
# AI RELEVANCE CLASSIFICATION SCHEMAS
# ======================================================
class AIRelevanceBatchRequest(BaseModel):
    """Request for batch AI relevance classification from parquet file."""
    parquet_path: str = Field(..., description="Path to the parquet file containing documents")
    text_column: str = Field("text", description="Column name containing document text")
    id_column: str = Field("doc_id", description="Column name containing document IDs")
    output_format: str = Field("binary", description="Output format: 'binary' or 'score'")
    output_path: Optional[str] = Field(None, description="Path to save classification results")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/documents/batch_2024.parquet",
                "text_column": "full_text",
                "id_column": "doc_id",
                "output_format": "score"
            }
        }


class AIRelevanceSingleRequest(BaseModel):
    """Request for single document AI relevance classification."""
    document_id: Optional[str] = Field(None, description="Document ID (if indexed)")
    text: Optional[str] = Field(None, description="Raw text to classify (if not indexed)")
    output_format: str = Field("binary", description="Output format: 'binary' or 'score'")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Desarrollo de sistema de inteligencia artificial...",
                "output_format": "score"
            }
        }


# ======================================================
# TOPIC INFERENCE SCHEMAS
# ======================================================
class TopicInferenceBatchRequest(BaseModel):
    """Request for batch topic inference from parquet file."""
    parquet_path: str = Field(..., description="Path to the parquet file containing documents")
    text_column: str = Field("text", description="Column name containing document text")
    id_column: str = Field("doc_id", description="Column name containing document IDs")
    model_name: str = Field(..., description="Name of the trained topic model to use")
    output_path: Optional[str] = Field(None, description="Path to save inference results")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/documents/batch_2024.parquet",
                "text_column": "full_text",
                "id_column": "doc_id",
                "model_name": "topic_model_v1"
            }
        }


class TopicInferenceSingleRequest(BaseModel):
    """Request for single document topic inference."""
    text: str = Field(..., description="Text to analyze for topic distribution")
    model_name: str = Field(..., description="Name of the trained topic model to use")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "El presente contrato tiene por objeto...",
                "model_name": "topic_model_v1"
            }
        }


# ======================================================
# EMBEDDINGS GENERATION SCHEMAS
# ======================================================
class EmbeddingsBatchRequest(BaseModel):
    """Request for batch embeddings generation from parquet file."""
    parquet_path: str = Field(..., description="Path to the parquet file containing documents")
    text_column: str = Field("text", description="Column name containing document text")
    id_column: str = Field("doc_id", description="Column name containing document IDs")
    model_type: str = Field("sentence-transformers", description="Embedding model type")
    batch_size: int = Field(32, ge=1, le=128, description="Batch size for processing")
    output_path: Optional[str] = Field(None, description="Path to save embeddings")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/documents/batch_2024.parquet",
                "text_column": "full_text",
                "id_column": "doc_id",
                "model_type": "sentence-transformers",
                "batch_size": 32
            }
        }


class EmbeddingsSingleRequest(BaseModel):
    """Request for single document embeddings generation."""
    document_id: Optional[str] = Field(None, description="Document ID (if indexed)")
    text: Optional[str] = Field(None, description="Raw text to generate embeddings for (if not indexed)")
    model_type: str = Field("sentence-transformers", description="Embedding model type")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "El presente contrato tiene por objeto...",
                "model_type": "sentence-transformers"
            }
        }


# Legacy schemas (kept for backwards compatibility)
class TextExtractionRequest(BaseModel):
    """DEPRECATED: Use TextExtractionBatchRequest or TextExtractionSingleRequest."""
    document_ids: List[str] = Field(..., description="Document IDs to process")
    normalize: bool = Field(True, description="Apply text normalization")


class SummarizationRequest(BaseModel):
    """DEPRECATED: Use SummarizationBatchRequest or SummarizationSingleRequest."""
    document_ids: List[str] = Field(..., description="Document IDs to summarize")
    focus_dimensions: Optional[List[str]] = Field(None, description="Focus dimensions for summary")
    include_traceability: bool = Field(True, description="Include traceability to original text")


class MetadataExtractionRequest(BaseModel):
    """DEPRECATED: Use MetadataExtractionBatchRequest or MetadataExtractionSingleRequest."""
    document_ids: List[str] = Field(..., description="Document IDs to process")
    metadata_fields: List[str] = Field(..., description="Metadata fields to extract")
    validation: bool = Field(True, description="Validate extracted metadata")


class AIRelevanceRequest(BaseModel):
    """DEPRECATED: Use AIRelevanceBatchRequest or AIRelevanceSingleRequest."""
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


class ProcessingJobResponse(BaseResponse):
    """Response for processing jobs."""
    job_id: Optional[str] = None
    status: str = "queued"
    progress: Optional[float] = None
    results: Optional[Dict[str, Any]] = None


# --- On-Demand Inference (External Documents) ---
class OnDemandInferenceBatchRequest(BaseModel):
    """
    Request for batch on-demand inference on external (non-indexed) documents.
    Processes multiple documents from a parquet file.
    """
    parquet_path: str = Field(..., description="Path to parquet file with documents to process")
    text_column: str = Field("text", description="Column name containing the text")
    id_column: Optional[str] = Field(None, description="Column name for document IDs")
    operations: List[str] = Field(
        default=["embeddings"],
        description="Operations: embeddings, topics, summary, all"
    )
    model_name: Optional[str] = Field(
        None, description="Model name (required for topic inference)")
    compare_with_index: bool = Field(
        False,
        description="Compare results with existing index to find similar documents"
    )
    corpus_collection: Optional[str] = Field(
        None,
        description="Corpus collection to compare against (required if compare_with_index=True)"
    )
    rows_per_doc: Optional[int] = Field(
        5, description="Number of similar documents to return per document")

    class Config:
        json_schema_extra = {
            "example": {
                "parquet_path": "/data/external_docs.parquet",
                "text_column": "content",
                "id_column": "doc_id",
                "operations": ["embeddings", "topics"],
                "model_name": "my_topic_model",
                "compare_with_index": True,
                "corpus_collection": "np_corpus",
                "rows_per_doc": 5
            }
        }


class OnDemandInferenceSingleRequest(BaseModel):
    """
    Request for on-demand inference on a single external (non-indexed) document.
    Allows real-time processing and comparison with existing index.
    """
    text: str = Field(..., description="Text/document to process")
    document_id: Optional[str] = Field(None, description="Optional document identifier")
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
                "text": "Supply of computer equipment for offices...",
                "document_id": "EXT-001",
                "operations": ["embeddings", "topics"],
                "model_name": "my_topic_model",
                "compare_with_index": True,
                "corpus_collection": "np_corpus",
                "rows": 10
            }
        }


class OnDemandInferenceResponse(BaseResponse):
    """Response for single on-demand inference."""
    document_id: Optional[str] = None
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
