"""
Infrastructure Administration (Configuration APIs)

This module groups low-level operations reserved for technical management
and dynamic system configuration:

- Solr collections management: creation, listing, reloading and deletion of indexes
- Display configuration: metadata, visible fields, active filters
- API key management: generation, listing, revocation

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modified: 04/02/2026 (Migrated to FastAPI and reorganized)
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Request, Query # type: ignore
from src.api.schemas import (
    BaseResponse,
    CollectionResponse,
    CollectionListResponse,
    MetadataFieldsResponse,
    CorpusListResponse,
    ModelsListResponse,
    CorpusModelsResponse,
    ErrorResponse,
    QueryOperator,
)
from src.api.exceptions import (
    ConflictException,
    NotFoundException,
    SolrException
)
from src.api.auth import (
    api_key_manager,
    verify_api_key,
    verify_master_key,
    APIKeyCreate,
    APIKeyResponse,
    APIKeyListResponse,
    APIKeyInfo,
)

router = APIRouter(
    prefix="/admin",
    tags=["1. Infrastructure Administration"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        409: {"model": ErrorResponse, "description": "Conflict"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


# ======================================================
# Management of Solr Collections
# ======================================================
@router.post(
    "/collections/create",
    response_model=CollectionResponse,
    status_code=201,
    summary="Crear colección Solr",
    description="Crea una nueva colección en Apache Solr.",
)
async def create_collection(
    request: Request,
    collection: str = Query(..., description="Name of the collection to create"),
) -> CollectionResponse:
    """Create a new Solr collection."""
    sc = request.app.state.solr_client
    try:
        _, status_code = sc.create_collection(col_name=collection)
        
        if status_code == 409:
            raise ConflictException("Collection", collection)
        
        return CollectionResponse(
            success=True,
            message=f"Collection '{collection}' created successfully",
            collection=collection
        )
    except ConflictException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.post(
    "/collections/delete",
    response_model=CollectionResponse,
    summary="Delete Solr collection",
    description="Deletes an existing collection from Apache Solr.",
)
async def delete_collection(
    request: Request,
    collection: str = Query(..., description="Name of the collection to delete"),
) -> CollectionResponse:
    """Delete an existing Solr collection."""
    sc = request.app.state.solr_client
    try:
        sc.delete_collection(col_name=collection)
        return CollectionResponse(
            success=True,
            message=f"Collection '{collection}' deleted successfully",
            collection=collection
        )
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/collections/list",
    response_model=CollectionListResponse,
    summary="List Solr collections",
    description="Returns the list of all available collections in Solr.",
)
async def list_collections(
    request: Request
) -> CollectionListResponse:
    """List all available Solr collections."""
    sc = request.app.state.solr_client
    try:
        collections = sc.list_collections()
        return CollectionListResponse(
            success=True,
            collections=collections
        )
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/collections/query",
    summary="Execute Solr query",
    description="Executes a direct query against a Solr collection using standard syntax.",
)
async def execute_raw_query(
    request: Request,
    collection: str = Query(..., description="Collection to query"),
    q: str = Query(..., description="Query string (Solr syntax)"),
    q_op: Optional[QueryOperator] = Query(
        None, 
        alias="q.op",
        description="Default operator (AND/OR)"
    ),
    fq: Optional[str] = Query(None, description="Query filter"),
    sort: Optional[str] = Query(None, description="Sorting (e.g., 'field asc')"),
    start: Optional[int] = Query(0, ge=0, description="Pagination offset"),
    rows: Optional[int] = Query(10, ge=1, le=1000, description="Number of rows"),
    fl: Optional[str] = Query(None, description="Fields to return (comma-separated)"),
    df: Optional[str] = Query(None, description="Default field"),
) -> Dict[str, Any]:
    """Execute Solr query with full parameters."""
    sc = request.app.state.solr_client
    query_values = {
        "q_op": q_op.value if q_op else None,
        "fq": fq,
        "sort": sort,
        "start": start,
        "rows": rows,
        "fl": fl,
        "df": df
    }
    query_values = {k: v for k, v in query_values.items() if v is not None}
    
    try:
        code, results = sc.execute_query(q=q, col_name=collection, **query_values)
        return {
            "success": True,
            "data": results.docs,
            "num_found": getattr(results, 'num_found', len(results.docs))
        }
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# Visualization Configuration Management
# ======================================================
@router.get(
    "/corpora/list",
    response_model=CorpusListResponse,
    summary="List all corpora",
    description="Returns the list of all corpora indexed in the system.",
)
async def list_all_corpora(
    request: Request
) -> CorpusListResponse:
    """List all available corpora."""
    sc = request.app.state.solr_client
    try:
        corpus_lst, code = sc.list_corpus_collections()
        if code != 200:
            raise SolrException(f"Error listing corpora (code: {code})")
        return CorpusListResponse(
            success=True,
            corpora=corpus_lst
        )
    except SolrException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/corpora/{corpus_col}/models",
    response_model=CorpusModelsResponse,
    summary="List topic models of a corpus",
    description="Lists all topic models associated with a specific corpus.",
)
async def list_corpus_models(
    request: Request,
    corpus_col: str,
) -> CorpusModelsResponse:
    """List models associated with a corpus."""
    sc = request.app.state.solr_client
    try:
        models_lst, code = sc.get_corpus_models(corpus_col=corpus_col)
        if code != 200:
            raise SolrException(f"Error getting models (code: {code})")
                
        return CorpusModelsResponse(
            success=True,
            models=models_lst
        )
    except SolrException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/models/list",
    response_model=ModelsListResponse,
    summary="Listar todos los modelos",
    description="Lista todos los modelos de tópicos disponibles en el sistema.",
)
async def list_all_models(
    request: Request
) -> ModelsListResponse:
    """List all topic models."""
    sc = request.app.state.solr_client
    try:
        models_lst, code = sc.list_model_collections()
        if code != 200:
            raise SolrException(f"Error listing models (code: {code})")
        
        
        return ModelsListResponse(
            success=True,
            models=models_lst
        )
    except SolrException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/config/metadata-displayed/{corpus_col}",
    response_model=MetadataFieldsResponse,
    summary="Get visible metadata fields",
    description="Gets the list of metadata fields configured to be displayed in the interface. This configuration allows evolving the visualization independently of the code.",
)
async def get_metadata_displayed(
    request: Request,
    corpus_col: str,
) -> MetadataFieldsResponse:
    """Get MetadataDisplayed fields from a corpus."""
    sc = request.app.state.solr_client
    try:
        fields_lst, code = sc.get_corpus_MetadataDisplayed(corpus_col=corpus_col)
        if code != 200:
            raise SolrException(f"Error getting metadata fields (code: {code})")
        return MetadataFieldsResponse(
            success=True,
            fields=fields_lst
        )
    except SolrException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/config/searchable-fields/{corpus_col}",
    response_model=MetadataFieldsResponse,
    summary="Get searchable fields",
    description="Gets the list of fields enabled for search in a corpus.",
)
async def get_searchable_fields(
    request: Request,
    corpus_col: str,
) -> MetadataFieldsResponse:
    """Get SearchableField fields from a corpus."""
    sc = request.app.state.solr_client
    try:
        fields_lst, code = sc.get_corpus_SearchableField(corpus_col=corpus_col)
        if code != 200:
            raise SolrException(f"Error getting fields (code: {code})")
        return MetadataFieldsResponse(
            success=True,
            fields=fields_lst
        )
    except SolrException:
        raise
    except Exception as e:
        raise SolrException(str(e))


@router.post(
    "/config/searchable-fields/{corpus_col}/add",
    response_model=BaseResponse,
    summary="Add searchable fields",
    description="Adds new fields to the list of searchable fields of a corpus.",
)
async def add_searchable_fields(
    request: Request,
    corpus_col: str,
    searchable_fields: str = Query(..., description="Fields to add (comma-separated)"),
) -> BaseResponse:
    """Add SearchableField fields to a corpus."""
    sc = request.app.state.solr_client
    try:
        sc.modify_corpus_SearchableFields(
            SearchableFields=searchable_fields,
            corpus_col=corpus_col,
            action="add"
        )
        return BaseResponse(
            success=True,
            message=f"Fields {searchable_fields} added to '{corpus_col}'"
        )
    except Exception as e:
        raise SolrException(str(e))


@router.post(
    "/config/searchable-fields/{corpus_col}/delete",
    response_model=BaseResponse,
    summary="Eliminar campos de búsqueda",
    description="Removes fields from the list of searchable fields of a corpus.",
)
async def delete_searchable_fields(
    request: Request,
    corpus_col: str,
    searchable_fields: str = Query(..., description="Fields to remove (comma-separated)"),
) -> BaseResponse:
    """Remove SearchableField fields from a corpus."""
    sc = request.app.state.solr_client
    try:
        sc.modify_corpus_SearchableFields(
            SearchableFields=searchable_fields,
            corpus_col=corpus_col,
            action="remove"
        )
        return BaseResponse(
            success=True,
            message=f"Fields removed from '{corpus_col}'"
        )
    except Exception as e:
        raise SolrException(str(e))


@router.get(
    "/config/searchable-fields/all",
    response_model=MetadataFieldsResponse,
    summary="Get all searchable fields",
    description="Gets all searchable fields available in all corpora.",
)
async def get_all_searchable_fields(
    request: Request
) -> MetadataFieldsResponse:
    """Get all SearcheableField fields from the system."""
    sc = request.app.state.solr_client
    try:
        fields, code = sc.get_all_searchable_fields()
        if code != 200:
            raise SolrException(f"Error getting fields (code: {code})")
        return MetadataFieldsResponse(
            success=True,
            fields=fields
        )
    except SolrException:
        raise
    except Exception as e:
        raise SolrException(str(e))


# ======================================================
# API Key Management (requires master key)
# ======================================================
@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=201,
    summary="Generate new API key",
    description="""
    Generate a new API key. **Requires master key authentication.**
    
    The generated API key will only be shown once in the response.
    Store it securely as it cannot be retrieved later.
    """,
    dependencies=[Depends(verify_master_key)],
)
async def create_api_key(
    request: Request,
    key_request: APIKeyCreate,
) -> APIKeyResponse:
    """Generate a new API key."""
    return api_key_manager.generate_key(key_request.name)


@router.get(
    "/api-keys",
    response_model=APIKeyListResponse,
    summary="List all API keys",
    description="""
    List all API keys with their metadata. **Requires master key authentication.**
    
    Note: The actual key values are not returned for security reasons.
    """,
    dependencies=[Depends(verify_master_key)],
)
async def list_api_keys(
    request: Request,
) -> APIKeyListResponse:
    """List all API keys."""
    keys = api_key_manager.list_keys()
    return APIKeyListResponse(keys=keys, total=len(keys))


@router.get(
    "/api-keys/{key_id}",
    response_model=APIKeyInfo,
    summary="Get API key info",
    description="Get information about a specific API key. **Requires master key authentication.**",
    dependencies=[Depends(verify_master_key)],
)
async def get_api_key(
    request: Request,
    key_id: str,
) -> APIKeyInfo:
    """Get info about a specific API key."""
    key_info = api_key_manager.get_key_info(key_id)
    if not key_info:
        raise NotFoundException(f"API key with ID '{key_id}' not found")
    return key_info


@router.post(
    "/api-keys/{key_id}/revoke",
    response_model=BaseResponse,
    summary="Revoke API key",
    description="""
    Revoke an API key (soft delete). **Requires master key authentication.**
    
    The key will be deactivated but kept in records.
    """,
    dependencies=[Depends(verify_master_key)],
)
async def revoke_api_key(
    request: Request,
    key_id: str,
) -> BaseResponse:
    """Revoke an API key."""
    if api_key_manager.revoke_key(key_id):
        return BaseResponse(success=True, message=f"API key '{key_id}' revoked successfully")
    raise NotFoundException(f"API key with ID '{key_id}' not found")


@router.delete(
    "/api-keys/{key_id}",
    response_model=BaseResponse,
    summary="Delete API key",
    description="Permanently delete an API key. **Requires master key authentication.**",
    dependencies=[Depends(verify_master_key)],
)
async def delete_api_key(
    request: Request,
    key_id: str,
) -> BaseResponse:
    """Permanently delete an API key."""
    if api_key_manager.delete_key(key_id):
        return BaseResponse(success=True, message=f"API key '{key_id}' deleted successfully")
    raise NotFoundException(f"API key with ID '{key_id}' not found")
