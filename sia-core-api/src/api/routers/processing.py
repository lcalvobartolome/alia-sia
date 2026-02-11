"""
Enrichment and Data Ingestion (Processing APIs)

Set of services that orchestrate and trigger the processing pipeline:
- Ingestion and processing trigger: download, text extraction, normalization
- AI inference services: summaries, topic models, embeddings
- On-demand inference: real-time processing of external documents

Endpoints can be executed:
1. Sequentially within the ingestion pipeline (configuration controlled)
2. Independently for on-demand calculations

Response conventions:
- Single endpoints return a typed response extending ResponseBase (e.g. TextExtractionSingleResponse).
- Batch endpoints return BatchProcessingResponse (success/message + job_id).
- All error responses use ErrorResponse.

Author: Lorena Calvo-Bartolome
Date: 27/03/2023
Modified: 04/02/2026 (Migrated to FastAPI and reorganized)
"""

import uuid
import base64
from pathlib import Path

from fastapi import APIRouter, Body, Path as PathParam, Request # type: ignore

from src.api.exceptions import (
    APIException,
    ConflictException,
    NotFoundException,
    ProcessingException,
    ValidationException,
    error_responses,
)
from src.api.schemas import (
    CorpusIndexRequest,
    ModelIndexRequest,
    ResponseBase,
    BatchProcessingResponse,
    IndexingResponse,
    TopicModelTrainingRequest,
    PipelineRequest,
    # Single processing responses
    TextExtractionSingleResponse,
    SummarizationSingleResponse,
    MetadataExtractionSingleResponse,
    AIRelevanceSingleResponse,
    TopicInferenceSingleResponse,
    EmbeddingsSingleResponse,
    OnDemandInferenceSingleResponse,
    # Batch processing request schemas
    TextExtractionBatchRequest,
    SummarizationBatchRequest,
    MetadataExtractionBatchRequest,
    AIRelevanceBatchRequest,
    TopicInferenceBatchRequest,
    EmbeddingsBatchRequest,
    OnDemandInferenceBatchRequest,
    # Single document processing request schemas
    TextExtractionSingleRequest,
    SummarizationSingleRequest,
    MetadataExtractionSingleRequest,
    AIRelevanceSingleRequest,
    TopicInferenceSingleRequest,
    EmbeddingsSingleRequest,
    OnDemandInferenceSingleRequest,
)

# ======================================================
# Router
# ======================================================
router = APIRouter(
    prefix="/processing",
    tags=["2. Data Enrichment and Ingestion"],
)

# ======================================================
# Corpus Management
# ======================================================
@router.post(
    "/corpora",
    response_model=IndexingResponse,
    status_code=201,
    summary="Index corpus",
    description="Index a corpus from a parquet file. The corpus has been previously preprocessed and stored in the file system.",
    responses=error_responses(
        ConflictException, ProcessingException,
        ConflictException="Corpus already exists",
    ),
)
async def index_corpus(
    request: Request,
    body: CorpusIndexRequest = Body(...),
) -> IndexingResponse:
    """Index a corpus from parquet file."""
    sc = request.app.state.solr_client
    try:
        sc.index_corpus(body.corpus_name)
        return IndexingResponse(
            success=True,
            message=f"Corpus '{body.corpus_name}' indexed successfully",
            status="completed"
        )
    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(str(e))


@router.delete(
    "/corpora/{corpus_name}",
    response_model=ResponseBase,
    summary="Delete corpus",
    description="Delete a corpus and all its associated data from the system.",
    responses=error_responses(
        NotFoundException, ProcessingException,
        NotFoundException="Corpus not found",
    ),
)
async def delete_corpus(
    request: Request,
    corpus_name: str = PathParam(..., description="Name of the corpus to delete"),
) -> ResponseBase:
    """Delete a corpus from the system."""
    sc = request.app.state.solr_client
    try:
        sc.delete_corpus(corpus_name)
        return ResponseBase(
            success=True,
            message=f"Corpus '{corpus_name}' deleted successfully"
        )
    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# Model Management
# ======================================================
@router.post(
    "/models",
    response_model=IndexingResponse,
    status_code=201,
    summary="Index topic model",
    description="Index a trained topic model in Solr.",
    responses=error_responses(
        ConflictException, ProcessingException,
        ConflictException="Model already exists",
    ),
)
async def index_model(
    request: Request,
    body: ModelIndexRequest = Body(...),
) -> IndexingResponse:
    """Index a topic model in Solr."""
    sc = request.app.state.solr_client
    try:
        sc.index_model(body.model_name)
        return IndexingResponse(
            success=True,
            message=f"Model '{body.model_name}' indexed successfully",
            status="completed"
        )
    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(str(e))


@router.delete(
    "/models/{model_name}",
    response_model=ResponseBase,
    summary="Delete topic model",
    description="Delete a topic model from the system.",
    responses=error_responses(
        NotFoundException, ProcessingException,
        NotFoundException="Model not found",
    ),
)
async def delete_model(
    request: Request,
    model_name: str = PathParam(..., description="Name of the model to delete"),
) -> ResponseBase:
    """Delete a topic model."""
    sc = request.app.state.solr_client
    try:
        sc.delete_model(model_name)
        return ResponseBase(
            success=True,
            message=f"Model '{model_name}' deleted successfully"
        )
    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# On-Demand Inference
# ======================================================
@router.post(
    "/inference/on-demand/batch",
    response_model=BatchProcessingResponse,
    summary="Batch on-demand inference",
    description="Process multiple external PDF documents (not indexed) from a parquet file. Operations: text extraction, summarization, metadata enrichment, topic inference, embeddings.",
    responses=error_responses(
        ValidationException, ProcessingException,
        ValidationException="Invalid request",
    ),
)
async def on_demand_inference_batch(
    req: Request,
    body: OnDemandInferenceBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Perform on-demand inference on multiple external documents from a parquet file."""
    return BatchProcessingResponse(
        success=True,
        message="Batch on-demand inference started.",
        job_id=f"job_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/inference/on-demand/single",
    response_model=OnDemandInferenceSingleResponse,
    summary="Single document on-demand inference",
    description="Real-time processing of a single external PDFdocument (not indexed). Operations: text extraction, summarization, metadata enrichment, topic inference, embeddings.",
    responses=error_responses(
        ValidationException, NotFoundException, ProcessingException,
        ValidationException="Invalid request",
        NotFoundException="Model not found",
    ),
)
async def on_demand_inference_single(
    req: Request,
    body: OnDemandInferenceSingleRequest = Body(...),
) -> OnDemandInferenceSingleResponse:
    """Perform on-demand inference on a single external document."""
    sc = req.app.state.solr_client
    doc_id = body.document_id or f"EXT-{uuid.uuid4().hex[:8].upper()}"

    try:
        return OnDemandInferenceSingleResponse(
            success=True,
            document_id=doc_id,
            message="On-demand inference completed "
        )

    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# Full Pipeline (ingestion orchestration)
# ======================================================
@router.post(
    "/pipeline",
    response_model=BatchProcessingResponse,
    summary="Run ingestion pipeline",
    description=(
        "Orchestrate a full or partial ingestion pipeline for a corpus. "
        "Select which modules to execute via the 'modules' list: they "
        "will always run in dependency order: download -> pdf_extraction -> "
        "summarization | metadata_enrichment | ai_relevance_classification | "
        "topic_modeling | embeddings -> ingestion.  Per-module parameters are "
        "provided in the 'config' object."
    ),
    responses=error_responses(
        ValidationException, ProcessingException,
        ValidationException="Invalid pipeline configuration",
    ),
)
async def run_pipeline(
    request: Request,
    body: PipelineRequest = Body(...),
) -> BatchProcessingResponse:
    """Run the processing pipeline with the selected modules."""
    job_id = f"pipeline_{uuid.uuid4().hex[:8]}"
    return BatchProcessingResponse(
        success=True,
        message=f"Pipeline started for corpus '{body.corpus_name}' with modules: {[m.value for m in body.modules]}",
        job_id=job_id,
        status="queued"
    )


# ======================================================
# MODULE 2: PDF Document Parsing
# ======================================================
@router.post(
    "/pdf/extract-text/batch",
    response_model=BatchProcessingResponse,
    summary="Batch text extraction and normalization",
    description="Extract textual content from multiple PDFs in a parquet file and apply normalization.",
    responses=error_responses(
        ValidationException, ProcessingException,
        ValidationException="Invalid request",
    ),
)
async def extract_text_batch(
    request: Request,
    body: TextExtractionBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Extract and normalize text from multiple PDF documents in a parquet file."""
    return BatchProcessingResponse(
        success=True,
        message="Batch text extraction started ",
        job_id=f"extract_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/pdf/extract-text/single",
    response_model=TextExtractionSingleResponse,
    summary="Single document text extraction (path or base64)",
    description="Extract textual content from a PDF file already in the server (by path) or from base64 encoded content.",
    responses=error_responses(
        ValidationException, NotFoundException, ProcessingException,
        ValidationException="Invalid request",
        NotFoundException="File not found",
    ),
)
async def extract_text_single(
    request: Request,
    body: TextExtractionSingleRequest = Body(...),
) -> TextExtractionSingleResponse:
    """Extract and normalize text from a single PDF document (path or base64)."""
    doc_id = body.document_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    
    if not body.pdf_path and not body.pdf_content:
        raise ValidationException("Either 'pdf_path' or 'pdf_content' must be provided.")
    
    try:
        if body.pdf_path:
            pdf_path = Path(body.pdf_path)
            if not pdf_path.exists():
                raise NotFoundException(f"PDF file not found: {body.pdf_path}")
            if not pdf_path.suffix.lower() == '.pdf':
                raise ValidationException(f"Invalid file type. Expected PDF, got: {pdf_path.suffix}")
            
            pdf_content = pdf_path.read_bytes()
            
        elif body.pdf_content:
            try:
                pdf_content = base64.b64decode(body.pdf_content)
            except Exception:
                raise ValidationException("Invalid base64 encoded PDF content.")
        
        # TODO: Implement actual text extraction from pdf_content
        return TextExtractionSingleResponse(
            success=True,
            message=f"Text extraction completed for document '{doc_id}'",
            data="[Extracted text placeholder — pending implementation]"
        )
        
    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(f"Error processing PDF: {str(e)}")


# ======================================================
# MODULE 3: Automatic Summary Generation
# ======================================================
@router.post(
    "/summarization/batch",
    response_model=BatchProcessingResponse,
    summary="Batch summary generation",
    description="Generate summaries for multiple documents from a parquet file using LLMs.",
    responses=error_responses(ProcessingException),
)
async def generate_summaries_batch(
    request: Request,
    body: SummarizationBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Generate automatic summaries for multiple documents in a parquet file."""
    # TODO: Implement batch summary generation with LLM
    return BatchProcessingResponse(
        success=True,
        message="Batch summary generation started ",
        job_id=f"summary_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/summarization/single",
    response_model=SummarizationSingleResponse,
    summary="Single document summary generation",
    description="Generate summary for a single document using LLMs.",
    responses=error_responses(ProcessingException),
)
async def generate_summary_single(
    request: Request,
    body: SummarizationSingleRequest = Body(...),
) -> SummarizationSingleResponse:
    """Generate automatic summary for a single document."""
    # TODO: Implement single summary generation with LLM
    return SummarizationSingleResponse(
        success=True,
        message="Summary generation completed ",
        data=None,
        grounding=None
    )


# ======================================================
# MODULE 4: Automatic Metadata Enrichment
# ======================================================
@router.post(
    "/metadata/extract/batch",
    response_model=BatchProcessingResponse,
    summary="Batch metadata extraction",
    description="Extract metadata from multiple documents in a parquet file using LLMs.",
    responses=error_responses(ProcessingException),
)
async def extract_metadata_batch(
    request: Request,
    body: MetadataExtractionBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Extract structured metadata from multiple documents in a parquet file."""
    return BatchProcessingResponse(
        success=True,
        message="Batch metadata extraction started ",
        job_id=f"metadata_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/metadata/extract/single",
    response_model=MetadataExtractionSingleResponse,
    summary="Single document metadata extraction",
    description="Extract metadata from a single document using LLMs.",
    responses=error_responses(ProcessingException),
)
async def extract_metadata_single(
    request: Request,
    body: MetadataExtractionSingleRequest = Body(...),
) -> MetadataExtractionSingleResponse:
    """Extract structured metadata from a single document."""
    return MetadataExtractionSingleResponse(
        success=True,
        message="Metadata extraction completed ",
        data=None
    )


# ======================================================
# MODULE 5: Identification of AI-Relevant Actions
# ======================================================
@router.post(
    "/ai-relevance/classify/batch",
    response_model=BatchProcessingResponse,
    summary="Batch AI relevance classification",
    description="Classify multiple documents from a parquet file by AI relevance.",
    responses=error_responses(ProcessingException),
)
async def classify_ai_relevance_batch(
    request: Request,
    body: AIRelevanceBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Classify multiple documents by AI relevance from a parquet file."""
    return BatchProcessingResponse(
        success=True,
        message="Batch AI relevance classification started ",
        job_id=f"relevance_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/ai-relevance/classify/single",
    response_model=AIRelevanceSingleResponse,
    summary="Single document AI relevance classification",
    description="Classify a single document by AI relevance.",
    responses=error_responses(ProcessingException),
)
async def classify_ai_relevance_single(
    request: Request,
    body: AIRelevanceSingleRequest = Body(...),
) -> AIRelevanceSingleResponse:
    """Classify a single document by AI relevance."""
    return AIRelevanceSingleResponse(
        success=True,
        message="AI relevance classification completed ",
        data=None
    )


# ======================================================
# MODULE 6: Topic modeling
# ======================================================
@router.post(
    "/topic-modeling/train",
    response_model=BatchProcessingResponse,
    summary="Train topic model",
    description="Topic model training for unsupervised thematic classification.",
    responses=error_responses(ProcessingException),
)
async def train_topic_model(
    request: Request,
    body: TopicModelTrainingRequest = Body(...),
) -> BatchProcessingResponse:
    """Train topic model for unsupervised thematic classification."""
    # TODO: Implement topic model training
    return BatchProcessingResponse(
        success=True,
        message="Topic model training started ",
        job_id=f"train_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/topic-modeling/infer/batch",
    response_model=BatchProcessingResponse,
    summary="Batch topic inference",
    description="Infer topic distributions for multiple documents from a parquet file.",
    responses=error_responses(ProcessingException),
)
async def infer_topics_batch(
    request: Request,
    body: TopicInferenceBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Infer topic distributions for multiple documents in a parquet file."""
    # TODO: Implement batch topic inference
    return BatchProcessingResponse(
        success=True,
        message="Batch topic inference started",
        job_id=f"infer_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/topic-modeling/infer/single",
    response_model=TopicInferenceSingleResponse,
    summary="Single document topic inference",
    description="Infer topic distribution for a single text.",
    responses=error_responses(
        NotFoundException, ProcessingException,
        NotFoundException="Model not found",
    ),
)
async def infer_topic_single(
    request: Request,
    body: TopicInferenceSingleRequest = Body(...),
) -> TopicInferenceSingleResponse:
    """Infer topic distribution for a single text."""
    sc = request.app.state.solr_client

    try:
        result = sc.do_Q22(model_name=body.model_name, text_to_infer=body.text)
        return TopicInferenceSingleResponse(
            success=True,
            message="Topic inference completed",
            data=result
        )

    except APIException:
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# MODULE 7: Generation of document embeddings
# ======================================================
@router.post(
    "/embeddings/batch",
    response_model=BatchProcessingResponse,
    summary="Batch embeddings generation",
    description="Generate contextualized embeddings for multiple documents from a parquet file.",
    responses=error_responses(ProcessingException),
)
async def generate_embeddings_batch(
    request: Request,
    body: EmbeddingsBatchRequest = Body(...),
) -> BatchProcessingResponse:
    """Generate contextualized embeddings for multiple documents in a parquet file."""
    return BatchProcessingResponse(
        success=True,
        message="Batch embeddings generation started ",
        job_id=f"embed_{uuid.uuid4().hex[:8]}",
        status="queued"
    )


@router.post(
    "/embeddings/single",
    response_model=EmbeddingsSingleResponse,
    summary="Single document embeddings generation",
    description="Generate contextualized embeddings for a single document.",
    responses=error_responses(ProcessingException),
)
async def generate_embeddings_single(
    request: Request,
    body: EmbeddingsSingleRequest = Body(...),
) -> EmbeddingsSingleResponse:
    """Generate contextualized embeddings for a single document."""
    return EmbeddingsSingleResponse(
        success=True,
        message="Embeddings generation completed ",
        data=None
    )