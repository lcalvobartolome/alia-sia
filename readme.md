# ALIA-SIA

Intelligence and Analysis System for Public Procurement and Aid (from Spanish, *"Sistema de Inteligencia y Análisis de Contratación y Ayudas Públicas"*).

- [ALIA-SIA](#alia-sia)
  - [Documentation](#documentation)
  - [Services](#services)
  - [Instructions for deployment](#instructions-for-deployment)
    - [1. Create env file with the following structure](#1-create-env-file-with-the-following-structure)
    - [2. Actualize folder with data and GPU resources in docker-compose.yaml](#2-actualize-folder-with-data-and-gpu-resources-in-docker-composeyaml)
    - [3. Build and start services](#3-build-and-start-services)
    - [4. Generate an API key](#4-generate-an-api-key)
  - [Docker → Podman deployment (moving pre-built images to another machine)](#docker--podman-deployment-moving-pre-built-images-to-another-machine)
    - [1. On the source machine: build and export the images](#1-on-the-source-machine-build-and-export-the-images)
    - [2. Copy what the target machine needs](#2-copy-what-the-target-machine-needs)
    - [3. Load the images and start the stack](#3-load-the-images-and-start-the-stack)
  - [API Authentication](#api-authentication)
    - [Configuration](#configuration)
    - [API Key Management](#api-key-management)
      - [Generate a new API key](#generate-a-new-api-key)
      - [List all API keys](#list-all-api-keys)
      - [Revoke an API key](#revoke-an-api-key)
      - [Delete an API key](#delete-an-api-key)
    - [Using API Keys](#using-api-keys)
      - [With Swagger UI](#with-swagger-ui)
      - [With curl](#with-curl)
    - [Public Endpoints (no authentication required)](#public-endpoints-no-authentication-required)
  - [Commands](#commands)
    - [To index a corpus:](#to-index-a-corpus)
    - [To launch the extract pipeline:](#to-launch-the-extract-pipeline)

## Documentation

- **Swagger UI**: `http://<host>:10083/docs`
- **ReDoc**: `http://<host>:10083/redoc`
- **OpenAPI JSON**: `http://<host>:10083/openapi.json`

## Services

| Service | Port | Description |
|---------|------|-------------|
| sia-core-api | 10083 | Main REST API |
| solr | 10085 | Apache Solr search engine |
| zoo | 10086/10087 | Zookeeper for Solr Cloud |

## Instructions for deployment

### 1. Create env file with the following structure

Create a `.env` file in the project root:

```bash
# Master key for API key management (admin operations)
SIA_MASTER_KEY=your-secure-master-key-here

# CORS allowed origins (comma-separated). Use "*" for development only.
CORS_ORIGINS=http://<host>:3000,https://your-frontend.com

# GitHub token to clone private pipeline repository during Docker build
GITHUB_TOKEN=your-github-token-here
```

### 2. Actualize folder with data and GPU resources in docker-compose.yaml

```yaml
networks:
  sia-net:
    name: sia-net
services:
  sia-core-api:
    build:
      context: ./sia-core-api
      args:
        GITHUB_TOKEN: ${GITHUB_TOKEN}
    container_name: sia-core-api
    ports:
      - 10083:10083
    environment:
      #NVIDIA_DRIVER_CAPABILITIES: compute,utility # needed in Lt2 >>---
      SOLR_URL: http://solr:8983
      SIA_MASTER_KEY: ${SIA_MASTER_KEY:-master-key-change-in-production}
      API_KEYS_FILE: /config/api_keys.json
      CORS_ORIGINS: ${CORS_ORIGINS:-http://<host>:3000,http://<host>:8080}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - solr
    volumes:
      - {folder_with_data}:/mnt/data >>---
      - ./sia-config:/config
      - ./db/data/sqlite3/pipeline_jobs.db:/data/pipeline_jobs.db
    deploy:
      resources:
        limits:
          memory: 100GB
        # remove the following for Lt2 >>---
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["2"]
              capabilities: [gpu]
```

### 3. Build and start services

```bash
docker compose up -d --build
```

To follow the logs:

```bash
docker compose logs -f sia-core-api
```

### 4. Generate an API key

Once the API is running, use the master key to generate an API key for regular access; see [API Authentication](#api-authentication) below.

## Docker → Podman deployment (moving pre-built images to another machine)

Use this when you build the images once (with Docker) and then need to run
the stack on a different machine that only has Podman, without rebuilding
anything there (e.g. no internet access, no `GITHUB_TOKEN`, no build tools).

A ready-to-use `docker-compose.podman.yaml` is kept at the project root next
to `docker-compose.yaml` specifically for this. The two files build the exact
same three images with the exact same tags (`sia-core-api:latest`,
`sia-solr:9.1.1`, `sia-solr-config:latest`).

### 1. On the source machine: build and export the images

`zookeeper` and `alpine` are pulled from Docker Hub rather than built, so if
the target machine has internet access Podman will just pull them itself and
you can drop them from the command below. Include them only if the target
machine is offline / air-gapped:

```bash
docker compose build
docker save -o sia-images.tar \
  sia-core-api:latest sia-solr:9.1.1 sia-solr-config:latest \
  zookeeper:latest alpine:latest
```

### 2. Copy what the target machine needs

- `sia-images.tar`
- The whole project folder (needed for the bind-mounted config/data:
  `sia-config/`, `solr-config/`, `db/`, `docker-compose.podman.yaml`, `.env`)

### 3. Load the images and start the stack

```bash
podman load -i sia-images.tar
podman compose -f docker-compose.podman.yaml up -d
```

Podman finds `sia-core-api:latest`, `sia-solr:9.1.1` and `sia-solr-config:latest`
already loaded and starts the containers directly.

## API Authentication

The API uses a two-tier authentication system:

1. **Master Key**: For administrative operations (generating/managing API keys)
2. **API Keys**: For regular API access (all other endpoints)

### Configuration

Set the following environment variables (in `.env` file or docker-compose):

```bash
SIA_MASTER_KEY=your-secure-master-key-here
CORS_ORIGINS=http://<host>:3000,https://your-frontend.com
```

### API Key Management

Use the master key to manage API keys through the admin endpoints.

#### Generate a new API key

```bash
curl -X POST "http://<host>:10083/admin/api-keys" \
  -H "X-API-Key: your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "frontend-production"}'
```

Response:

```json
{
  "key_id": "a1b2c3d4",
  "name": "frontend-production",
  "api_key": "abc123...xyz789",  // Save this! Only shown once
  "created_at": "2026-02-04T12:00:00Z"
}
```

#### List all API keys

```bash
curl -X GET "http://<host>:10083/admin/api-keys" \
  -H "X-API-Key: your-master-key"
```

#### Revoke an API key

```bash
curl -X POST "http://<host>:10083/admin/api-keys/{key_id}/revoke" \
  -H "X-API-Key: your-master-key"
```

#### Delete an API key

```bash
curl -X DELETE "http://<host>:10083/admin/api-keys/{key_id}" \
  -H "X-API-Key: your-master-key"
```

### Using API Keys

#### With Swagger UI

1. Open Swagger: `http://<host>:10083/docs`
2. Click the **"Authorize"** button (🔓 lock icon, top right)
3. Enter your API key in the `X-API-Key` field
4. Click **"Authorize"** then **"Close"**
5. Now all requests from Swagger will include the API key automatically

#### With curl

```bash
curl -X GET "http://<host>:10083/api/documents/search?query=test" \
  -H "X-API-Key: your-api-key"
```

### Public Endpoints (no authentication required)

- `GET /` - API info
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## Commands

### To index a corpus:

```bash
curl -X 'POST' \
  'http://<host>:<port>/processing/corpora' \
  -H 'accept: application/json' \
  -H 'X-API-Key: <your-api-key>' \
  -H 'Content-Type: application/json' \
  -d '{
  "corpus_name": "<corpus_name>"
}'
```

### To launch the extract pipeline:

```bash
curl -X 'POST' \
  'http://<host>:<port>/processing/alia-pipeline/extract' \
  -H 'accept: application/json' \
  -H 'X-API-Key: <your-api-key>' \
  -H 'Content-Type: application/json' \
  -d '{
  "base_dir": "<data_dir>",
  "tipo": "<tipo>",
  "calculate_on": "<field_name>",
  "llm_model_gen": "<ollama_model>",
  "embed_model": "<huggingface_embed_model>",
  "file_workers": <num_file_workers>,
  "row_workers": <num_row_workers>,
  "semantic_threshold": <threshold>,
  "mallet": "<path_to_mallet>",
  "ollama_host": "<ollama_host_url>"
}'
```
