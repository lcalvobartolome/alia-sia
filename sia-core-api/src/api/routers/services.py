"""
Exploitation services (Service APIs)

Set of services for public and mass consumption by the PortalIA.

It integrates the following functionalities:

- Multimodal search: exact metadata queries, thematic search, semantic search
- Calculation of indicators
- Recommendation services

Author: Lorena Calvo-Bartolome
Date: 27/03/2023
Modified: 04/02/2026 (Migrated to FastAPI and reorganized)
"""

from typing import Optional
from fastapi import APIRouter, Request, Query  # type: ignore

from src.api.schemas import (
    ErrorResponse,
    DataResponse,
)
from src.api.exceptions import (
    SolrException,
    NotFoundException,
)
from src.core.clients.np_solr_client import SIASolrClient

router = APIRouter(
    prefix="/search",
    tags=["3. Exploitation Services"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# ======================================================
# Search by Metadata (Exact Queries)
# ======================================================
@router.get(
    "/documents/by-string",
    summary="Search documents by text",
    description="Allows searching for documents containing a specific string in the SearchableField fields of the corpus.",
)
async def search_docs_by_string(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    string: str = Query(..., description="Search string"),
    start: Optional[int] = Query(0, ge=0, description="Pagination offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Find documents containing a text string."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q7(
            corpus_col=corpus_collection,
            string=string,
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/documents/{doc_id}/metadata",
    summary="Get document metadata",
    description="Retrieve all metadata associated with a specific document.",
)
async def get_document_metadata(
    request: Request,
    doc_id: str,
    corpus_collection: str = Query(..., description="Corpus collection"),
) -> DataResponse:
    """Get document metadata by ID."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q6(corpus_col=corpus_collection, doc_id=doc_id)
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/documents/{doc_id}/thetas",
    summary="Get document topic distribution",
    description="""
    Retrieve the document-topic distribution (thetas) of a document 
    according to a specific topic model.
    """,
)
async def get_document_thetas(
    request: Request,
    doc_id: str,
    corpus_collection: str = Query(..., description="Corpus collection"),
    model_name: str = Query(..., description="Topic model name"),
) -> DataResponse:
    """Get topic distribution of a document."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q1(
            corpus_col=corpus_collection,
            doc_id=doc_id,
            model_name=model_name
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/corpus/metadata-fields",
    summary="Get corpus metadata fields",
    description="Returns the list of metadata fields available in a corpus.",
)
async def get_corpus_metadata_fields(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
) -> DataResponse:
    """Get metadata fields of a corpus."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q2(corpus_col=corpus_collection)
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Thematic Search
# ======================================================
@router.get(
    "/topics/labels",
    summary="Get topic labels",
    description="""
    Get labels of all topics in a model.
    
    Allows exploring available themes in a topic model.
    """,
)
async def get_topic_labels(
    request: Request,
    model_name: str = Query(..., description="Model name"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Get topic labels from a model."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q8(model_col=model_name, start=start, rows=rows)
        return {"success": True, "data": result}

    except (SolrException, NotFoundException):
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/topics/{topic_id}/documents",
    summary="Get top documents from a topic",
    description="""
    Retrieve documents with highest probability for a specific topic.
    
    Allows exploring the most representative documents of each theme.
    """,
)
async def get_topic_top_documents(
    request: Request,
    topic_id: str,
    corpus_collection: str = Query(..., description="Corpus collection"),
    model_name: str = Query(..., description="Model name"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Get top documents from a topic."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q9(
            corpus_col=corpus_collection,
            model_name=model_name,
            topic_id=topic_id,
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}

    except (SolrException, NotFoundException):
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/topics/model-info",
    summary="Get topic model information",
    description="Retrieve detailed information about a topic model.",
)
async def get_model_info(
    request: Request,
    model_name: str = Query(..., description="Model name"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Get information about a topic model."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q10(
            model_col=model_name,
            start=start,
            rows=rows,
            only_id=False
        )
        return {"success": True, "data": result}

    except (SolrException, NotFoundException):
        raise
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Semantic Search
# ======================================================
@router.get(
    "/semantic/by-text",
    summary="Semantic search by text",
    description="""
    Semantic search for documents similar to a given text.
    
    Uses BERT embeddings to find documents semantically 
    related to the query, regardless of exact word matches.
    
    This is the recommended method for high-quality semantic search.
    """,
)
async def semantic_search_by_text(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    query_text: str = Query(..., description="Query text"),
    keyword_filter: Optional[str] = Query(
        None, description="Optional keyword filter"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(10, description="Number of results"),
) -> DataResponse:
    """Semantic search using BERT embeddings."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q21(
            corpus_col=corpus_collection,
            search_doc=query_text,
            embedding_model="bert",
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/semantic/similar-by-topic-model",
    summary="Thematic similarity by text",
    description="""
    Find similar documents using topic models.
    
    Infers the topic distribution of the given text and finds 
    documents with similar thematic distributions.
    
    Useful for structured and explainable thematic search.
    """,
)
async def similar_docs_by_text_tm(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    model_name: str = Query(..., description="Topic model name"),
    text_to_infer: str = Query(..., description="Query text"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(10, description="Number of results"),
) -> DataResponse:
    """Find similar documents using topic model inference."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q14(
            corpus_col=corpus_collection,
            model_name=model_name,
            text_to_infer=text_to_infer,
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}

    except (NotFoundException,):
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/semantic/by-word",
    summary="Search by word similarity",
    description="""
    Find documents related to a word using Word2Vec.
    
    Uses word embeddings to find documents containing
    terms semantically related to the search word.
    
    Ideal for thematic exploration based on key concepts.
    """,
)
async def docs_related_to_word(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    model_name: str = Query(..., description="Embeddings model name"),
    word: str = Query(..., description="Search keyword"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(10, description="Number of results"),
) -> DataResponse:
    """Find documents related to a word using Word2Vec embeddings."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q20(
            corpus_col=corpus_collection,
            model_name=model_name,
            search_word=word,
            embedding_model="word2vec",
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}

    except (NotFoundException,):
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/semantic/by-document",
    summary="Documents similar to existing indexed document",
    description="""
    Find documents similar to an already indexed one.
    
    Calculates similarity based on topic distributions between documents.
    """,
)
async def similar_documents_by_id(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    doc_id: str = Query(..., description="Reference document ID"),
    model_name: str = Query(...,
                            description="Model for similarity calculation"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Find documents similar to an existing one."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q5(
            corpus_col=corpus_collection,
            model_name=model_name,
            doc_id=doc_id,
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Indicators and Statistics
# ======================================================
@router.get(
    "/indicators/document-count",
    summary="Count documents",
    description="Get the total number of documents in a collection.",
)
async def get_document_count(
    request: Request,
    collection: str = Query(..., description="Collection name"),
) -> DataResponse:
    """Get document count."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q3(col=collection)
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/indicators/years",
    summary="Get available years",
    description="List all years with documents in the corpus.",
)
async def get_available_years(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
) -> DataResponse:
    """Get list of available years."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q31(corpus_col=corpus_collection)
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Temporal Search
# ======================================================
@router.get(
    "/temporal/by-year",
    summary="Documents by year",
    description="Retrieve documents filtered by publication year.",
)
async def get_documents_by_year(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    year: int = Query(..., description="Year to filter"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Get documents from a specific year."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q30(
            corpus_col=corpus_collection,
            year=year,
            start=start,
            rows=rows
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/temporal/advanced",
    summary="Advanced temporal search",
    description="""
    Advanced search with temporal filters and sorting.
    
    Allows:
    - Filter by year range
    - Sort by multiple fields
    - Combine with keyword search
    """,
)
async def advanced_temporal_search(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    start_year: Optional[int] = Query(None, description="Start year of range"),
    end_year: Optional[int] = Query(None, description="End year of range"),
    sort_by: Optional[str] = Query(
        "date:desc",
        description="Sorting: field:order (e.g., 'date:desc,title:asc')"
    ),
    keyword: Optional[str] = Query("", description="Search keyword"),
    searchable_field: Optional[str] = Query(
        "", description="Field to search in"),
    start: Optional[int] = Query(0, ge=0, description="Offset"),
    rows: Optional[int] = Query(None, description="Number of results"),
) -> DataResponse:
    """Temporal search with advanced filters."""
    sort_by_order = []
    if sort_by:
        for sort_item in sort_by.split(','):
            if ':' in sort_item:
                field, order = sort_item.strip().split(':', 1)
                sort_by_order.append((field.strip(), order.strip()))
            else:
                sort_by_order.append((sort_item.strip(), 'desc'))
    else:
        sort_by_order = [("date", "desc")]

    sc = request.app.state.solr_client
    try:
        result = sc.do_Q32(
            corpus_col=corpus_collection,
            start=start,
            rows=rows,
            sort_by_order=sort_by_order,
            start_year=start_year,
            end_year=end_year,
            keyword=keyword,
            searchable_field=searchable_field
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Recommendation Services
# ======================================================
@router.get(
    "/recommendations/by-document",
    summary="Document-based recommendations",
    description="""
    Get recommended documents similar to a given one.
    
    Uses the topic distribution of the reference document 
    to find the most related documents.
    """,
)
async def get_recommendations_by_document(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    doc_id: str = Query(..., description="Reference document ID"),
    model_name: str = Query(...,
                            description="Model for similarity calculation"),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of recommendations"),
) -> DataResponse:
    """Get recommendations based on a document."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q5(
            corpus_col=corpus_collection,
            model_name=model_name,
            doc_id=doc_id,
            start=0,
            rows=limit
        )
        return {
            "success": True,
            "recommendations": result,
            "total": len(result) if isinstance(result, list) else 1,
            "recommendation_type": "document_similarity"
        }
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/recommendations/by-text",
    summary="Text-based recommendations",
    description="""
    Get recommended documents similar to a given text.
    
    Uses semantic search to find documents related 
    to the provided query text.
    """,
)
async def get_recommendations_by_text(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    query_text: str = Query(..., description="Query text"),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of recommendations"),
) -> DataResponse:
    """Get recommendations based on text."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q21(
            corpus_col=corpus_collection,
            search_doc=query_text,
            embedding_model="bert",
            start=0,
            rows=limit
        )
        return {
            "success": True,
            "recommendations": result,
            "total": len(result) if isinstance(result, list) else 1,
            "recommendation_type": "semantic_similarity"
        }
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/recommendations/by-topic",
    summary="Topic-based recommendations",
    description="""
    Get recommended documents from a specific topic.
    
    Returns the most representative documents of a given theme.
    """,
)
async def get_recommendations_by_topic(
    request: Request,
    corpus_collection: str = Query(..., description="Corpus collection"),
    model_name: str = Query(..., description="Model name"),
    topic_id: str = Query(..., description="Topic ID"),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of recommendations"),
) -> DataResponse:
    """Get recommendations based on a topic."""
    sc = request.app.state.solr_client
    try:
        result = sc.do_Q9(
            corpus_col=corpus_collection,
            model_name=model_name,
            topic_id=topic_id,
            start=0,
            rows=limit
        )
        return {
            "success": True,
            "recommendations": result,
            "total": len(result) if isinstance(result, list) else 1,
            "recommendation_type": "topic_based"
        }

    except (SolrException, NotFoundException):
        raise
    except Exception as e:
        raise SolrException(str(e))
