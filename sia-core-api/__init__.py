"""
FastAPI routers package.

Organized in three blocks:
1. admin - Infrastructure Administration (Configuration APIs)
2. processing - Data Enrichment & Ingestion (Processing APIs)
3. services - Exploitation Services (Service APIs)
"""

from .admin import router as admin_router
from .processing import router as processing_router
from .services import router as services_router

__all__ = [
    "admin_router",
    "processing_router",
    "services_router",
]
