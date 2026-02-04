"""
Enrichment and Data Ingestion (Processing APIs)

Set of services that orchestrate and trigger the processing pipeline:
- Ingestion and processing trigger: download, text extraction, normalization
- AI inference services: summaries, topic models, embeddings
- On-demand inference: real-time processing of external documents

Endpoints can be executed:
1. Sequentially within the ingestion pipeline (configuration controlled)
2. Independently for on-demand calculations

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modified: 04/02/2026 (Migrated to FastAPI and reorganized)
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Query, Request  # type: ignore
from src.api.exceptions import (NotFoundException, ProcessingException, ValidationException)

from ..schemas import (
    BaseResponse,
    DataSource,
    EmbeddingsGenerationRequest,
    ErrorResponse,
    IngestionResponse,
    OnDemandInferenceRequest,
    OnDemandInferenceResponse,
    ProcessingJobResponse,
    SummarizationRequest,
    TopicModelTrainingRequest,
    DownloadRequest,
    TextExtractionRequest,
    MetadataExtractionRequest,
    AIRelevanceRequest,
)

router = APIRouter(
    prefix="/processing",
    tags=["2. Data Enrichment and Ingestion"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# ======================================================
#
# ======================================================
@router.post(
    "/corpus/index",
    response_model=IngestionResponse,
    summary="Index corpus",
    description="Index a corpus from a parquet file. The corpus has been previously preprocessed and stored in the system.",
)
async def index_corpus(
    request: Request,
    corpus_name: str = Query(...,
                             description="Name of the corpus to index (without extension)"),
) -> IngestionResponse:
    """Index a corpus from parquet file."""
    sc = request.app.state.solr_client
    try:
        sc.index_corpus(corpus_name)
        return IngestionResponse(
            success=True,
            message=f"Corpus '{corpus_name}' indexed successfully",
            status="completed"
        )
    except Exception as e:
        raise ProcessingException(str(e))


@router.delete(
    "/corpus/{corpus_name}",
    response_model=BaseResponse,
    summary="Delete corpus",
    description="Delete a corpus and all its associated data from the system.",
)
async def delete_corpus(
    request: Request,
    corpus_name: str,
) -> BaseResponse:
    """Delete a corpus from the system."""
    sc = request.app.state.solr_client
    try:
        sc.delete_corpus(corpus_name)
        return BaseResponse(
            success=True,
            message=f"Corpus '{corpus_name}' deleted successfully"
        )
    except Exception as e:
        raise ProcessingException(str(e))


@router.post(
    "/models/index",
    response_model=ProcessingJobResponse,
    summary="Index topic model",
    description="Index a trained topic model in Solr.",
)
async def index_model(
    request: Request,
    model_name: str = Query(...,
                            description="Name of the model folder"),
) -> ProcessingJobResponse:
    """Index a topic model in Solr."""
    sc = request.app.state.solr_client
    try:
        sc.index_model(model_name)
        return ProcessingJobResponse(
            success=True,
            message=f"Model '{model_name}' indexed successfully",
            status="completed"
        )
    except Exception as e:
        raise ProcessingException(str(e))


@router.delete(
    "/models/{model_name}",
    response_model=BaseResponse,
    summary="Delete topic model",
    description="Delete a topic model from the system.",
)
async def delete_model(
    request: Request,
    model_name: str,
) -> BaseResponse:
    """Delete a topic model."""
    sc = request.app.state.solr_client
    try:
        sc.delete_model(model_name)
        return BaseResponse(
            success=True,
            message=f"Model '{model_name}' deleted successfully"
        )
    except Exception as e:
        raise ProcessingException(str(e))


@router.post(
    "/inference/on-demand",
    response_model=OnDemandInferenceResponse,
    summary="On-demand inference",
    description="Real-time processing of external documents (not indexed). Operations can be disabled as needed."
)
async def on_demand_inference(
    req: Request,
    request: OnDemandInferenceRequest = Body(...),
) -> OnDemandInferenceResponse:
    """
    Perform on-demand inference on an external document.
    """
    sc = req.app.state.solr_client
    response = OnDemandInferenceResponse(success=True)

    try:
        # @TODO: Implement
        return response

    except (ValidationException, NotFoundException):
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# MODULE 1
# ======================================================
@router.post(
    "/ingestion/download",
    response_model=BaseResponse,
    summary="Download data",
    description="Download data from a specific source with configurable filters.",
)
async def download(
    request: Request,
    download_request: DownloadRequest = Body(...),
) -> BaseResponse:
    """Perform download from data source."""
    # TODO: Implement download logic
    return BaseResponse(
        success=True,
        message="Download initiated (pending implementation)"
    )


# ======================================================
# MODULE 2: PDF Document Parsing
# ======================================================
@router.post(
    "/pdf/extract-text",
    response_model=Dict[str, Any],
    summary="Text extraction and normalization",
    description="Extract textual content from PDFs and apply normalization (cleaning, structuring).",
)
async def extract_and_normalize_text(
    request: Request,
    extraction_request: TextExtractionRequest = Body(...),
) -> Dict[str, Any]:
    """Extract and normalize text from PDF documents."""
    # TODO: Implement text extraction and normalization
    return {
        "success": True,
        "message": "Text extraction started (pending implementation)",
        "documents_queued": len(extraction_request.document_ids)
    }

# ======================================================
# MODULE 3: Automatic Summary Generation
# ======================================================
@router.post(
    "/summarization/generate",
    response_model=Dict[str, Any],
    summary="Automatic summary generation",
    description="Document summary generation based on LLMs",
)
async def generate_summaries(
    request: Request,
    summary_request: SummarizationRequest = Body(...),
) -> Dict[str, Any]:
    """Generate automatic summaries with configurable focus and traceability."""
    # TODO: Implement summary generation with LLM
    return {
        "success": True,
        "message": "Summary generation started (pending implementation)",
        "documents_queued": len(summary_request.document_ids),
        "focus_dimensions": summary_request.focus_dimensions or ["general"],
        "traceability": summary_request.include_traceability
    }

# ======================================================
# MODULE 4: Automatic Metadata Enrichment
# ======================================================
@router.post(
    "/metadata/auto-extract",
    response_model=Dict[str, Any],
    summary="Automatic metadata extraction",
    description="Normalized metadata extraction using LLMs"
)
async def extract_metadata(
    request: Request,
    metadata_request: MetadataExtractionRequest = Body(...),
) -> Dict[str, Any]:
    """Extract structured metadata from documents using LLMs."""
    # TODO: Implement metadata extraction with LLM
    return {
        "success": True,
        "message": "Metadata extraction started (pending implementation)",
        "documents_queued": len(metadata_request.document_ids),
        "fields_to_extract": metadata_request.metadata_fields,
        "validation_enabled": metadata_request.validation
    }

# ======================================================
# MODULE 5: Identification of AI-Relevant Actions
# ======================================================
@router.post(
    "/ai-relevance/classify",
    response_model=Dict[str, Any],
    summary="AI relevance classification",
    description="Automatic detection and classification of AI-related actions.",
)
async def classify_ai_relevance(
    request: Request,
    relevance_request: AIRelevanceRequest = Body(...),
) -> Dict[str, Any]:
    """Classify documents by AI relevance."""
    # TODO: Implement AI relevance classification
    return {
        "success": True,
        "message": "AI relevance classification started (pending implementation)",
        "documents_queued": len(relevance_request.document_ids),
        "output_format": relevance_request.output_format
    }

# ======================================================
# MODULE 6: Thematic Classification with Topic Models
# ======================================================
@router.post(
    "/topic-modeling/train",
    response_model=Dict[str, Any],
    summary="Train topic model",
    description="Topic model training for unsupervised thematic classification.",
)
async def train_topic_model(
    request: Request,
    training_request: TopicModelTrainingRequest = Body(...),
) -> Dict[str, Any]:
    """Train topic model for unsupervised thematic classification."""
    # TODO: Implement topic model training
    return {
        "success": True,
        "message": "Topic model training started (pending implementation)",
        "corpus": training_request.corpus_name,
        "model": training_request.model_name,
        "num_topics": training_request.num_topics
    }


@router.post(
    "/topic-modeling/infer",
    summary="Infer text topics",
    description="""
    Infer topic distribution for a given text.
    
    This endpoint can be invoked independently outside the preprocessing flow
    to obtain thematic information from external documents.
    """,
)
async def infer_topic(
    request: Request,
    text_to_infer: str = Body(..., description="Text to analyze"),
    model_name: str = Body(..., description="Model name"),
) -> Dict[str, Any]:
    """Infer topic distribution for a text."""
    sc = request.app.state.solr_client

    try:
        result = sc.do_Q22(model_name=model_name, text_to_infer=text_to_infer)
        return {"success": True, "data": result}

    except (ValidationException, NotFoundException):
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# MODULE 7: Generation of document embeddings
# ======================================================
@router.post(
    "/embeddings/generate-contextual",
    response_model=Dict[str, Any],
    summary="Generate contextualized embeddings",
    description="""
    **Contextualized embeddings generation with SentenceTransformers.**
    
    Generates high-quality dense vector representations for semantic search,
    less interpretable but more accurate in distance measures.
    """,
)
async def generate_contextual_embeddings(
    request: Request,
    embeddings_request: EmbeddingsGenerationRequest = Body(...),
) -> Dict[str, Any]:
    """Generate contextualized embeddings using SentenceTransformers."""
    # TODO: Implement contextualized embeddings generation
    return {
        "success": True,
        "message": "Contextualized embeddings generation started (pending implementation)",
        "corpus": embeddings_request.corpus_name,
        "model_type": embeddings_request.model_type,
        "representation_type": "contextualized_embeddings",
        "batch_size": embeddings_request.batch_size
    }
