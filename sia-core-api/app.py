"""
SIA-Core API: RESTful API of the SIA-Core system, which is organized into three functional blocks:
1. Infrastructure Administration - Technical management and configuration
2. Data Enrichment and Ingestion - Processing pipeline orchestration
3. Exploitation Services - Public consumption by the AI Portal

Author: Lorena Calvo-Bartolomé
Date: 27/03/2023
Modified: 04/02/2026 (Migrated to FastAPI and reorganized)
"""

import logging
import pathlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.openapi.utils import get_openapi  # type: ignore
from src.api.exceptions import (APIException, api_exception_handler,
                                generic_exception_handler,
                                http_exception_handler)
from src.api.routers.admin import router as admin_router
from src.api.routers.processing import router as processing_router
from src.api.routers.services import router as exploitation_router
from src.api.schemas import HealthResponse, PingResponse
from src.core.clients.np_solr_client import SIASolrClient


# ======================================================
# Loaders (version, description, tags)
# ======================================================
def load_version() -> str:
    """Load application version from file."""
    version_path = pathlib.Path(__file__).parent / "docs" / "version.txt"
    try:
        return version_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "1.0.0"


def load_api_description() -> str:
    """Load API description from markdown file."""
    docs_path = pathlib.Path(__file__).parent / "docs" / "api_description.md"
    try:
        return docs_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "SIA-Core API - Sistema de Inteligencia y Análisis"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SIA-Core-API")

# get version
VERSION = load_version()

# ======================================================
# Application lifespan
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting SIA-Core API...")

    try:
        config_path = pathlib.Path(__file__).parent / "config" / "config.cf"
        app.state.solr_client = SIASolrClient(
            logger, config_file=str(config_path))
        logger.info("Solr client initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Solr client: {e}")
        raise

    yield

    logger.info("Closing SIA-Core API...")


# ======================================================
# Application Configuration
# ======================================================
app = FastAPI(
    title="SIA-Core API",
    description=load_api_description(),
    version=VERSION,
    contact={
        "name": "Lorena Calvo-Bartolomé",
        "email": "lcalvo@pa.uc3m.es",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ======================================================
# CORS Middleware
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # @beauseant
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# Exception Handlers
# ======================================================
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# ======================================================
# Routers
# ======================================================
[app.include_router(router) for router in [admin_router,
                                           processing_router, exploitation_router]]

# ======================================================
# Root and Health Endpoints
# ======================================================
@app.get(
    "/",
    summary="Raíz de la API",
    description="Información básica sobre la API y enlaces a documentación.",
    tags=["Health"],
)
async def root():
    """Root endpoint with basic API information and documentation links."""
    return {
        "name": "SIA-Core API",
        "version": VERSION,
        "description": "API RESTful del Sistema de Inteligencia y Análisis de Contratación y Ayudas Públicas (SIA).",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "blocks": {
            "1_admin": "/admin - Administración de Infraestructura",
            "2_processing": "/processing - Enriquecimiento e Ingesta",
            "3_exploitation": "/api - Servicios de Explotación"
        }
    }


@app.get(
    "/ping",
    response_model=PingResponse,
    summary="Health Check Simple",
    description="Verificación simple de que la API está funcionando.",
    tags=["Health"],
    responses={
        200: {
            "description": "API funcionando correctamente",
            "content": {
                "application/json": {
                    "example": {
                        "status": "pong",
                        "timestamp": "2026-02-04T12:00:00.000000Z",
                        "service": "NP Tools API"
                    }
                }
            }
        }
    }
)
async def ping() -> PingResponse:
    """Health check endpoint."""
    return PingResponse(
        status="pong",
        timestamp=datetime.now(timezone.utc).isoformat(),
        service="NP Tools API"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check Detallado",
    description="Estado detallado de salud incluyendo conectividad con Solr.",
    tags=["Health"],
)
async def health_check(request: Request) -> HealthResponse:
    """Detailed health check with component status."""
    try:
        sc = request.app.state.solr_client
        collections = sc.list_collections()
        solr_status = "healthy"
        solr_collections_count = len(
            collections) if isinstance(collections, list) else 0
    except Exception as e:
        solr_status = f"unhealthy: {str(e)}"
        solr_collections_count = 0

    overall_status = "healthy" if solr_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        service="SIA- API",
        version=VERSION,
        components={
            "api": "healthy",
            "solr": solr_status,
            "solr_collections_count": solr_collections_count
        }
    )


# ======================================================
# Custom OpenAPI Schema
# ======================================================
def custom_openapi():
    """Generate custom OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
        license_info=app.license_info,
    )
    
    # Add servers
    openapi_schema["servers"] = [
        {"url": "/", "description": "Servidor actual"},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ======================================================
# Main Entry Point
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10083,
        reload=True,
        log_level="info",
    )
