"""
API Key Authentication and Management

Provides API key generation, validation, and management functionality.
Keys are stored in a JSON file for persistence.

Author: Lorena Calvo-Bartolomé
Date: 04/02/2026
"""

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


# ======================================================
# Configuration
# ======================================================
API_KEY_HEADER_NAME = "X-API-Key"
MASTER_KEY = os.getenv("SIA_MASTER_KEY", "master-key-change-in-production")
API_KEYS_FILE = Path(os.getenv("API_KEYS_FILE", "/config/api_keys.json"))

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


# ======================================================
# Schemas
# ======================================================
class APIKeyInfo(BaseModel):
    """Information about an API key."""
    key_id: str = Field(..., description="Unique identifier for the key")
    name: str = Field(..., description="Descriptive name for the key")
    created_at: str = Field(..., description="ISO timestamp of creation")
    last_used: Optional[str] = Field(None, description="ISO timestamp of last use")
    is_active: bool = Field(True, description="Whether the key is active")


class APIKeyCreate(BaseModel):
    """Request to create a new API key."""
    name: str = Field(..., description="Descriptive name for the key (e.g., 'frontend-prod', 'data-team')")


class APIKeyResponse(BaseModel):
    """Response when creating a new API key."""
    key_id: str
    name: str
    api_key: str = Field(..., description="The API key (only shown once!)")
    created_at: str


class APIKeyListResponse(BaseModel):
    """Response listing all API keys."""
    keys: List[APIKeyInfo]
    total: int


# ======================================================
# API Key Storage
# ======================================================
class APIKeyManager:
    """Manages API keys with file-based persistence."""
    
    def __init__(self, keys_file: Path = API_KEYS_FILE):
        self.keys_file = keys_file
        self._keys: Dict[str, dict] = {}
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load keys from file."""
        if self.keys_file.exists():
            try:
                with open(self.keys_file, "r") as f:
                    self._keys = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._keys = {}
        else:
            self._keys = {}
            self._save_keys()
    
    def _save_keys(self) -> None:
        """Save keys to file."""
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.keys_file, "w") as f:
            json.dump(self._keys, f, indent=2)
    
    def generate_key(self, name: str) -> APIKeyResponse:
        """Generate a new API key."""
        key_id = secrets.token_hex(8)  # 16 char ID
        api_key = secrets.token_hex(32)  # 64 char key
        created_at = datetime.now(timezone.utc).isoformat()
        
        self._keys[api_key] = {
            "key_id": key_id,
            "name": name,
            "created_at": created_at,
            "last_used": None,
            "is_active": True
        }
        self._save_keys()
        
        return APIKeyResponse(
            key_id=key_id,
            name=name,
            api_key=api_key,
            created_at=created_at
        )
    
    def validate_key(self, api_key: str) -> bool:
        """Validate an API key and update last_used."""
        if api_key in self._keys and self._keys[api_key]["is_active"]:
            self._keys[api_key]["last_used"] = datetime.now(timezone.utc).isoformat()
            self._save_keys()
            return True
        return False
    
    def list_keys(self) -> List[APIKeyInfo]:
        """List all API keys (without the actual key values)."""
        return [
            APIKeyInfo(
                key_id=data["key_id"],
                name=data["name"],
                created_at=data["created_at"],
                last_used=data["last_used"],
                is_active=data["is_active"]
            )
            for data in self._keys.values()
        ]
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key by its ID."""
        for api_key, data in self._keys.items():
            if data["key_id"] == key_id:
                self._keys[api_key]["is_active"] = False
                self._save_keys()
                return True
        return False
    
    def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key by its ID."""
        for api_key, data in list(self._keys.items()):
            if data["key_id"] == key_id:
                del self._keys[api_key]
                self._save_keys()
                return True
        return False
    
    def get_key_info(self, key_id: str) -> Optional[APIKeyInfo]:
        """Get info about a specific key by ID."""
        for data in self._keys.values():
            if data["key_id"] == key_id:
                return APIKeyInfo(**data)
        return None


# Global instance
api_key_manager = APIKeyManager()


# ======================================================
# Authentication Dependencies
# ======================================================
async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Verify API key from header."""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key missing. Provide header: X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Master key also works as a valid API key (for convenience)
    if api_key == MASTER_KEY:
        return api_key
    
    # Check against managed keys
    if api_key_manager.validate_key(api_key):
        return api_key
    
    raise HTTPException(
        status_code=403,
        detail="Invalid API key",
    )


async def verify_master_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Verify master key for admin operations."""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Master key required for admin operations",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != MASTER_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid master key. Admin operations require the master key.",
        )
    
    return api_key
