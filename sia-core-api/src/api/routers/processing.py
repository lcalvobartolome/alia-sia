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
import uuid
import base64
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, Query, Request, UploadFile  # type: ignore
from src.api.exceptions import (NotFoundException, ProcessingException, ValidationException)

from ..schemas import (
    BaseResponse,
    ErrorResponse,
    IngestionResponse,
    OnDemandInferenceBatchRequest,
    OnDemandInferenceSingleRequest,
    OnDemandInferenceResponse,
    ProcessingJobResponse,
    TopicModelTrainingRequest,
    DownloadRequest,
    # Batch processing schemas
    TextExtractionBatchRequest,
    SummarizationBatchRequest,
    MetadataExtractionBatchRequest,
    AIRelevanceBatchRequest,
    TopicInferenceBatchRequest,
    EmbeddingsBatchRequest,
    # Single document processing schemas
    TextExtractionSingleRequest,
    SummarizationSingleRequest,
    MetadataExtractionSingleRequest,
    AIRelevanceSingleRequest,
    TopicInferenceSingleRequest,
    EmbeddingsSingleRequest,
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
    "/inference/on-demand/batch",
    response_model=Dict[str, Any],
    summary="Batch on-demand inference",
    description="Process multiple external documents (not indexed) from a parquet file. Operations: embeddings, topics, summary."
)
async def on_demand_inference_batch(
    req: Request,
    request: OnDemandInferenceBatchRequest = Body(...),
) -> Dict[str, Any]:
    """
    Perform on-demand inference on multiple external documents from a parquet file.
    """
    # TODO: Implement batch on-demand inference
    return {
        "success": True,
        "message": "Batch on-demand inference started (pending implementation)",
        "parquet_path": request.parquet_path,
        "operations": request.operations,
        "model_name": request.model_name,
        "compare_with_index": request.compare_with_index
    }


@router.post(
    "/inference/on-demand/single",
    response_model=OnDemandInferenceResponse,
    summary="Single document on-demand inference",
    description="Real-time processing of a single external document (not indexed). Operations: embeddings, topics, summary."
)
async def on_demand_inference_single(
    req: Request,
    request: OnDemandInferenceSingleRequest = Body(...),
) -> OnDemandInferenceResponse:
    """
    Perform on-demand inference on a single external document.
    """
    sc = req.app.state.solr_client
    doc_id = request.document_id or f"EXT-{uuid.uuid4().hex[:8].upper()}"

    try:
        # TODO: Implement actual inference logic
        return OnDemandInferenceResponse(
            success=True,
            document_id=doc_id,
            message="On-demand inference completed (pending implementation)"
        )

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
    "/pdf/extract-text/batch",
    response_model=Dict[str, Any],
    summary="Batch text extraction and normalization",
    description="Extract textual content from multiple PDFs in a parquet file and apply normalization.",
)
async def extract_text_batch(
    request: Request,
    extraction_request: TextExtractionBatchRequest = Body(...),
) -> Dict[str, Any]:
    """Extract and normalize text from multiple PDF documents in a parquet file."""
    # TODO: Implement batch text extraction and normalization
    return {
        "success": True,
        "message": "Batch text extraction started (pending implementation)",
        "parquet_path": extraction_request.parquet_path,
        "normalize": extraction_request.normalize
    }


@router.post(
    "/pdf/extract-text/single",
    response_model=Dict[str, Any],
    summary="Single document text extraction",
    description="Extract textual content from a single PDF and apply normalization. Supports file upload.",
)
async def extract_text_single(
    request: Request,
    file: Optional[UploadFile] = File(None, description="PDF file to upload"),
    document_id: Optional[str] = Form(None, description="Document ID (auto-generated if not provided)"),
    normalize: bool = Form(True, description="Apply text normalization"),
) -> Dict[str, Any]:
    """
    Extract and normalize text from a single PDF document.
    
    Upload a PDF file directly using multipart/form-data.
    """
    # Generate document_id if not provided
    doc_id = document_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    
    if not file:
        raise ValidationException("No file uploaded. Please provide a PDF file.")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise ValidationException(f"Invalid file type. Expected PDF, got: {file.filename}")
    
    # Get upload directory from config
    upload_dir = Path(request.app.state.config.get("restapi", "path_uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file to mounted volume
    try:
        pdf_content = await file.read()
        file_size = len(pdf_content)
        
        # Save with document_id as filename
        saved_path = upload_dir / f"{doc_id}.pdf"
        with open(saved_path, "wb") as f:
            f.write(pdf_content)
            
    except Exception as e:
        raise ProcessingException(f"Error saving uploaded file: {str(e)}")
    
    # TODO: Implement actual text extraction from pdf_content
    return {
        "success": True,
        "message": "Text extraction completed (pending implementation)",
        "document_id": doc_id,
        "filename": file.filename,
        "saved_path": str(saved_path),
        "file_size_bytes": file_size,
        "normalize": normalize
    }


@router.post(
    "/pdf/extract-text/single/from-path",
    response_model=Dict[str, Any],
    summary="Single document text extraction from path or base64",
    description="Extract textual content from a PDF file already in the server (by path) or from base64 encoded content.",
)
async def extract_text_single_from_path(
    request: Request,
    extraction_request: TextExtractionSingleRequest = Body(...),
) -> Dict[str, Any]:
    """
    Extract and normalize text from a single PDF document.
    
    Use this endpoint when:
    - The PDF is already on the server (provide pdf_path)
    - You have the PDF as base64 encoded string (provide pdf_content)
    
    For direct file upload, use /pdf/extract-text/single instead.
    """
    # Generate document_id if not provided
    doc_id = extraction_request.document_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    
    # Validate that at least one source is provided
    if not extraction_request.pdf_path and not extraction_request.pdf_content:
        raise ValidationException("Either 'pdf_path' or 'pdf_content' must be provided.")
    
    file_size = 0
    source_type = ""
    
    try:
        if extraction_request.pdf_path:
            # Read from server path
            pdf_path = Path(extraction_request.pdf_path)
            if not pdf_path.exists():
                raise NotFoundException(f"PDF file not found: {extraction_request.pdf_path}")
            if not pdf_path.suffix.lower() == '.pdf':
                raise ValidationException(f"Invalid file type. Expected PDF, got: {pdf_path.suffix}")
            
            pdf_content = pdf_path.read_bytes()
            file_size = len(pdf_content)
            source_type = "path"
            
        elif extraction_request.pdf_content:
            # Decode from base64
            try:
                pdf_content = base64.b64decode(extraction_request.pdf_content)
                file_size = len(pdf_content)
                source_type = "base64"
            except Exception:
                raise ValidationException("Invalid base64 encoded PDF content.")
        
        # TODO: Implement actual text extraction from pdf_content
        return {
            "success": True,
            "message": "Text extraction completed (pending implementation)",
            "document_id": doc_id,
            "source_type": source_type,
            "source_path": extraction_request.pdf_path,
            "file_size_bytes": file_size,
            "normalize": extraction_request.normalize
        }
        
    except (ValidationException, NotFoundException):
        raise
    except Exception as e:
        raise ProcessingException(f"Error processing PDF: {str(e)}")


# ======================================================
# MODULE 3: Automatic Summary Generation
# ======================================================
@router.post(
    "/summarization/generate/batch",
    response_model=Dict[str, Any],
    summary="Batch summary generation",
    description="Generate summaries for multiple documents from a parquet file using LLMs.",
)
async def generate_summaries_batch(
    request: Request,
    summary_request: SummarizationBatchRequest = Body(...),
) -> Dict[str, Any]:
    """Generate automatic summaries for multiple documents in a parquet file."""
    # TODO: Implement batch summary generation with LLM
    return {
        "success": True,
        "message": "Batch summary generation started (pending implementation)",
        "parquet_path": summary_request.parquet_path,
        "focus_dimensions": summary_request.focus_dimensions or ["general"],
        "traceability": summary_request.include_traceability
    }


@router.post(
    "/summarization/generate/single",
    response_model=Dict[str, Any],
    summary="Single document summary generation",
    description="Generate summary for a single document using LLMs.",
)
async def generate_summary_single(
    request: Request,
    summary_request: SummarizationSingleRequest = Body(...),
) -> Dict[str, Any]:
    """Generate automatic summary for a single document."""
    # TODO: Implement single summary generation with LLM
    return {
        "success": True,
        "message": "Summary generation completed (pending implementation)",
        "document_id": summary_request.document_id,
        "focus_dimensions": summary_request.focus_dimensions or ["general"],
        "traceability": summary_request.include_traceability
    }


# ======================================================
# MODULE 4: Automatic Metadata Enrichment
# ======================================================
@router.post(
    "/metadata/auto-extract/batch",
    response_model=Dict[str, Any],
    summary="Batch metadata extraction",
    description="Extract metadata from multiple documents in a parquet file using LLMs.",
)
async def extract_metadata_batch(
    request: Request,
    metadata_request: MetadataExtractionBatchRequest = Body(...),
) -> Dict[str, Any]:
    """Extract structured metadata from multiple documents in a parquet file."""
    # TODO: Implement batch metadata extraction with LLM
    return {
        "success": True,
        "message": "Batch metadata extraction started (pending implementation)",
        "parquet_path": metadata_request.parquet_path,
        "fields_to_extract": metadata_request.metadata_fields,
        "validation_enabled": metadata_request.validation
    }


@router.post(
    "/metadata/auto-extract/single",
    response_model=Dict[str, Any],
    summary="Single document metadata extraction",
    description="Extract metadata from a single document using LLMs.",
)
async def extract_metadata_single(
    request: Request,
    metadata_request: MetadataExtractionSingleRequest = Body(...),
) -> Dict[str, Any]:
    """Extract structured metadata from a single document."""
    # TODO: Implement single metadata extraction with LLM
    return {
        "success": True,
        "message": "Metadata extraction completed (pending implementation)",
        "document_id": metadata_request.document_id,
        "fields_to_extract": metadata_request.metadata_fields,
        "validation_enabled": metadata_request.validation
    }


# ======================================================
# MODULE 5: Identification of AI-Relevant Actions
# ======================================================
@router.post(
    "/ai-relevance/classify/batch",
    response_model=Dict[str, Any],
    summary="Batch AI relevance classification",
    description="Classify multiple documents from a parquet file by AI relevance.",
)
async def classify_ai_relevance_batch(
    request: Request,
    relevance_request: AIRelevanceBatchRequest = Body(...),
) -> Dict[str, Any]:
    """Classify multiple documents by AI relevance from a parquet file."""
    # TODO: Implement batch AI relevance classification
    return {
        "success": True,
        "message": "Batch AI relevance classification started (pending implementation)",
        "parquet_path": relevance_request.parquet_path,
        "output_format": relevance_request.output_format
    }


@router.post(
    "/ai-relevance/classify/single",
    response_model=Dict[str, Any],
    summary="Single document AI relevance classification",
    description="Classify a single document by AI relevance.",
)
async def classify_ai_relevance_single(
    request: Request,
    relevance_request: AIRelevanceSingleRequest = Body(...),
) -> Dict[str, Any]:
    """Classify a single document by AI relevance."""
    # TODO: Implement single AI relevance classification
    return {
        "success": True,
        "message": "AI relevance classification completed (pending implementation)",
        "document_id": relevance_request.document_id,
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
    "/topic-modeling/infer/batch",
    response_model=Dict[str, Any],
    summary="Batch topic inference",
    description="Infer topic distributions for multiple documents from a parquet file.",
)
async def infer_topics_batch(
    request: Request,
    inference_request: TopicInferenceBatchRequest = Body(...),
) -> Dict[str, Any]:
    """Infer topic distributions for multiple documents in a parquet file."""
    # TODO: Implement batch topic inference
    return {
        "success": True,
        "message": "Batch topic inference started (pending implementation)",
        "parquet_path": inference_request.parquet_path,
        "model_name": inference_request.model_name
    }


@router.post(
    "/topic-modeling/infer/single",
    response_model=Dict[str, Any],
    summary="Single document topic inference",
    description="Infer topic distribution for a single text.",
)
async def infer_topic_single(
    request: Request,
    inference_request: TopicInferenceSingleRequest = Body(...),
) -> Dict[str, Any]:
    """Infer topic distribution for a single text."""
    sc = request.app.state.solr_client

    try:
        result = sc.do_Q22(model_name=inference_request.model_name, text_to_infer=inference_request.text)
        return {"success": True, "data": result}

    except (ValidationException, NotFoundException):
        raise
    except Exception as e:
        raise ProcessingException(str(e))


# ======================================================
# MODULE 7: Generation of document embeddings
# ======================================================
@router.post(
    "/embeddings/generate-contextual/batch",
    response_model=Dict[str, Any],
    summary="Batch embeddings generation",
    description="Generate contextualized embeddings for multiple documents from a parquet file.",
)
async def generate_embeddings_batch(
    request: Request,
    embeddings_request: EmbeddingsBatchRequest = Body(...),
) -> Dict[str, Any]:
    """Generate contextualized embeddings for multiple documents in a parquet file."""
    # TODO: Implement batch embeddings generation
    return {
        "success": True,
        "message": "Batch embeddings generation started (pending implementation)",
        "parquet_path": embeddings_request.parquet_path,
        "model_type": embeddings_request.model_type,
        "batch_size": embeddings_request.batch_size
    }


@router.post(
    "/embeddings/generate-contextual/single",
    response_model=Dict[str, Any],
    summary="Single document embeddings generation",
    description="Generate contextualized embeddings for a single document.",
)
async def generate_embeddings_single(
    request: Request,
    embeddings_request: EmbeddingsSingleRequest = Body(...),
) -> Dict[str, Any]:
    """Generate contextualized embeddings for a single document."""
    # TODO: Implement single embeddings generation
    return {
        "success": True,
        "message": "Embeddings generation completed (pending implementation)",
        "document_id": embeddings_request.document_id,
        "model_type": embeddings_request.model_type
    }
