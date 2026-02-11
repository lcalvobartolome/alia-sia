"""
Exploitation services (Service APIs)

Set of services for public and mass consumption by the PortalIA.

It integrates the following functionalities:

- Multimodal search: exact metadata queries, thematic search, semantic search
- Calculation of indicators
- Recommendation services

Response conventions:
- All responses extend ResponseBase (success + message).
- Query endpoints use DataResponse (success + message + data).
- Error responses use ErrorResponse.

Author: Lorena Calvo-Bartolome
Date: 27/03/2023
Modified: 04/02/2026 (Migrated to FastAPI and reorganized)
"""

from fastapi import APIRouter, Body, Path, Request  # type: ignore

from src.api.schemas import (
    DataResponse,
    # Search request schemas
    SemanticSearchByTextRequest,
    ThematicSearchByTextRequest,
    WordSimilaritySearchRequest,
    SimilarByDocumentRequest,
    #TemporalSearchRequest,
    MetadataFilter,
)
from src.api.exceptions import (
    APIException,
    SolrException,
    NotFoundException,
    ValidationException,
    error_responses,
)

router = APIRouter(
    prefix="/exploitation",
    tags=["3. Exploitation Services"],
)


# ======================================================
# Helper: build Solr filter query from MetadataFilter
# ======================================================
def _build_filter_query(
    filter_query: str | None,
    filters: MetadataFilter | None,
) -> str | None:
    """
    Combine a raw Solr ``fq`` string with structured ``MetadataFilter``
    into a single filter query.

    Returns ``None`` if neither is active.
    """
    parts: list[str] = []

    if filter_query:
        parts.append(filter_query)

    if filters is not None:
        if filters.year is not None:
            parts.append(f"year:{filters.year}")
        if filters.cpv is not None:
            parts.append(f"cpv:{filters.cpv}")
        if filters.extra:
            for key, value in filters.extra.items():
                parts.append(f"{key}:{value}")

    return " AND ".join(parts) if parts else None


# ======================================================
# Metadata queries
# ======================================================
@router.get(
    "/corpora/{corpus_collection}/documents/{doc_id}",
    response_model=DataResponse,
    summary="Get document metadata",
    description="Retrieve all metadata associated with a specific document.",
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Document or corpus not found",
    ),
)
async def get_document_metadata(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
    doc_id: str = Path(..., description="Document ID"),
) -> DataResponse:
    """Get document metadata by ID."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q6(corpus_col=corpus_collection, doc_id=doc_id)
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/corpora/{corpus_collection}/metadata-fields",
    response_model=DataResponse,
    summary="Get corpus metadata fields",
    description="Returns the list of metadata fields available in a corpus.",
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus not found",
    ),
)
async def get_corpus_metadata_fields(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
) -> DataResponse:
    """Get metadata fields of a corpus."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q2(corpus_col=corpus_collection)
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Semantic Search
# ======================================================
@router.post(
    "/corpora/{corpus_collection}/semantic/by-text",
    response_model=DataResponse,
    summary="Semantic search by text",
    description=(
        "Semantic search for documents similar to a given text. Uses BERT "
        "embeddings to find documents semantically related to the query, "
        "regardless of exact word matches. Results can be filtered by year, "
        "CPV code, and additional metadata."
    ),
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus not found",
    ),
)
async def semantic_search_by_text(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
    body: SemanticSearchByTextRequest = Body(...),
) -> DataResponse:
    """Semantic search using BERT embeddings."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q21(
            corpus_col=corpus_collection,
            search_doc=body.query_text,
            embedding_model="bert",
            filter_query=_build_filter_query(body.filter_query, body.filters),
            start=body.pagination.start,
            rows=body.pagination.rows,
        )
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.post(
    "/corpora/{corpus_collection}/thematic/by-text",
    response_model=DataResponse,
    summary="Thematic similarity by text",
    description=(
        "Find thematically similar documents to a given text. Uses topic "
        "model inference to find documents with similar thematic content."
    ),
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus or model not found",
    ),
)
async def similar_docs_by_text_tm(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
    body: ThematicSearchByTextRequest = Body(...),
) -> DataResponse:
    """Find similar documents using topic model inference."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q14(
            corpus_col=corpus_collection,
            model_name=body.model_name,
            text_to_infer=body.query_text,
            filter_query=_build_filter_query(body.filter_query, body.filters),
            start=body.pagination.start,
            rows=body.pagination.rows,
        )
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.post(
    "/corpora/{corpus_collection}/semantic/by-word",
    response_model=DataResponse,
    summary="Search by word similarity",
    description=(
        "Find documents related to a word using Word2Vec. Uses word embeddings "
        "to find documents containing terms semantically related to the search word."
    ),
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus or model not found",
    ),
)
async def docs_related_to_word(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
    body: WordSimilaritySearchRequest = Body(...),
) -> DataResponse:
    """Find documents related to a word using Word2Vec embeddings."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q20(
            corpus_col=corpus_collection,
            model_name=body.model_name,
            search_word=body.word,
            embedding_model="word2vec",
            filter_query=_build_filter_query(body.filter_query, body.filters),
            start=body.pagination.start,
            rows=body.pagination.rows,
        )
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Similarity by Document ID(s)
# ======================================================
@router.post(
    "/corpora/{corpus_collection}/semantic/by-document",
    response_model=DataResponse,
    summary="Semantically similar documents by document ID(s)",
    description=(
        "Find documents semantically similar to one or more existing indexed "
        "documents. Accepts a list of document IDs and returns results "
        "aggregated across all reference documents."
    ),
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Document, corpus or model not found",
    ),
)
async def similar_documents_by_id(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
    body: SimilarByDocumentRequest = Body(...),
) -> DataResponse:
    """Find documents semantically similar to one or more existing documents."""
    sc = request.app.state.solr_client
    try:
        # TODO: implement multi-doc aggregation; for now process each ID
        all_results = []
        for doc_id in body.doc_ids:
            result = sc.do_Q21_by_doc(
                corpus_col=corpus_collection,
                doc_id=doc_id,
                filter_query=_build_filter_query(body.filter_query, body.filters),
                start=body.pagination.start,
                rows=body.pagination.rows,
            )
            all_results.extend(result if isinstance(result, list) else [result])
        return DataResponse(success=True, data=all_results)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.post(
    "/corpora/{corpus_collection}/thematic/by-document",
    response_model=DataResponse,
    summary="Thematically similar documents by document ID(s)",
    description=(
        "Find thematically similar documents to one or more existing indexed "
        "documents using topic model distributions."
    ),
    responses=error_responses(
        ValidationException, NotFoundException, SolrException,
        ValidationException="model_name is required for thematic similarity",
        NotFoundException="Document, corpus or model not found",
    ),
)
async def similar_docs_by_doc_tm(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
    body: SimilarByDocumentRequest = Body(...),
) -> DataResponse:
    """Find thematically similar documents to one or more existing documents."""
    if not body.model_name:
        raise ValidationException("model_name is required for thematic similarity")

    sc = request.app.state.solr_client
    try:
        all_results = []
        for doc_id in body.doc_ids:
            result = sc.do_Q15(
                corpus_col=corpus_collection,
                model_name=body.model_name,
                doc_id=doc_id,
                filter_query=_build_filter_query(body.filter_query, body.filters),
                start=body.pagination.start,
                rows=body.pagination.rows,
            )
            all_results.extend(result if isinstance(result, list) else [result])
        return DataResponse(success=True, data=all_results)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Temporal Search
# ======================================================
#@router.post(
#    "/corpora/{corpus_collection}/temporal/by-year",
#    response_model=DataResponse,
#    summary="Documents by year",
#    description="Retrieve documents filtered by publication year with optional metadata filters.",
#    responses=error_responses(
#        NotFoundException, SolrException,
#        NotFoundException="Corpus not found",
#    ),
#)
#async def get_documents_by_year(
#    request: Request,
#    corpus_collection: str = Path(..., description="Corpus collection name"),
#    body: TemporalSearchRequest = Body(...),
#) -> DataResponse:
#    """Get documents from a specific year."""
#    sc = request.app.state.solr_client
#    try:
#        result = sc.do_Q30(
#            corpus_col=corpus_collection,
#            year=body.year,
#            filter_query=_build_filter_query(body.filter_query, body.filters),
#            start=body.pagination.start,
#            rows=body.pagination.rows,
#        )
#        return DataResponse(success=True, data=result)
#    except APIException:
#        raise
#    except Exception as e:
#        raise SolrException(str(e))
    
# ======================================================
# Indicators and Statistics
# ======================================================
@router.get(
    "/collections/{collection}/count",
    response_model=DataResponse,
    summary="Count documents",
    description="Get the total number of documents in a collection.",
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Collection not found",
    ),
)
async def get_document_count(
    request: Request,
    collection: str = Path(..., description="Collection name"),
) -> DataResponse:
    """Get document count."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q3(col=collection)
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/corpora/{corpus_collection}/years",
    response_model=DataResponse,
    summary="Get available years",
    description="List all years with documents in the corpus.",
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus not found",
    ),
)
async def get_available_years(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
) -> DataResponse:
    """Get list of available years."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q31(corpus_col=corpus_collection)
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))
    
# couple of dummy endpoints for indicators to be implemented in the future
@router.get(
    "/corpora/{corpus_collection}/indicators/indicator1",
    response_model=DataResponse,
    summary="Indicator 1",
    description="Calculate indicator 1 for the corpus.",
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus not found",
    ),
)
async def calculate_indicator_1(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
) -> DataResponse:
    """Calculate indicator 1."""
    sc = request.app.state.solr_client
    try:
        result = {"indicator1": 42}  # Placeholder result
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))
    
@router.get(
    "/corpora/{corpus_collection}/indicators/indicator2",
    response_model=DataResponse,
    summary="Indicator 2",
    description="Calculate indicator 2 for the corpus.",
    responses=error_responses(
        NotFoundException, SolrException,
        NotFoundException="Corpus not found",
    ),
)
async def calculate_indicator_2(
    request: Request,
    corpus_collection: str = Path(..., description="Corpus collection name"),
) -> DataResponse:
    """Calculate indicator 2."""
    sc = request.app.state.solr_client
    try:
        result = {"indicator2": 3.14}  # Placeholder result
        return DataResponse(success=True, data=result)
    except APIException:
        raise
    except Exception as e:
        raise SolrException(str(e))