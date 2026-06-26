# ALIA-SIA

Sistema de Inteligencia y Análisis de Contratación y Ayudas Públicas.


## Documentation

- **Swagger UI**: http://kumo01:10083/docs
- **ReDoc**: http://kumo01:10083/redoc
- **OpenAPI JSON**: http://kumo01:10083/openapi.json

## API Authentication

The API uses a two-tier authentication system:

1. **Master Key**: For administrative operations (generating/managing API keys)
2. **API Keys**: For regular API access (all other endpoints)

### Configuration

Set the following environment variables (in `.env` file or docker-compose):

```bash
SIA_MASTER_KEY=your-secure-master-key-here
CORS_ORIGINS=http://kumo01:3000,https://your-frontend.com
```

### API Key Management

Use the master key to manage API keys through the admin endpoints.

#### Generate a new API key

```bash
curl -X POST "http://kumo01:10083/admin/api-keys" \
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
curl -X GET "http://kumo01:10083/admin/api-keys" \
  -H "X-API-Key: your-master-key"
```

#### Revoke an API key

```bash
curl -X POST "http://kumo01:10083/admin/api-keys/{key_id}/revoke" \
  -H "X-API-Key: your-master-key"
```

#### Delete an API key

```bash
curl -X DELETE "http://kumo01:10083/admin/api-keys/{key_id}" \
  -H "X-API-Key: your-master-key"
```

### Using API Keys

#### With Swagger UI

1. Open Swagger: http://kumo01:10083/docs
2. Click the **"Authorize"** button (🔓 lock icon, top right)
3. Enter your API key in the `X-API-Key` field
4. Click **"Authorize"** then **"Close"**
5. Now all requests from Swagger will include the API key automatically

#### With curl

```bash
curl -X GET "http://kumo01:10083/api/documents/search?query=test" \
  -H "X-API-Key: your-api-key"
```

### Public Endpoints (no authentication required)

- `GET /` - API info
- `GET /health` - Health check
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

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
CORS_ORIGINS=http://kumo01:3000,https://your-frontend.com

# GitHub token to clone private pipeline repository during Docker build
GITHUB_TOKEN=your-github-token-here
```

### 2. Actualize folder with data and GPU resources in docker-compose.yaml
```docker
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
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:3000,http://localhost:8080}
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

### 3. Download the Solr plugin

```bash
rm -rf solr-plugins/NP-solr-dist-plugin
mkdir -p solr-plugins/NP-solr-dist-plugin
wget -O solr-plugins/NP-solr-dist-plugin/NP-solr-dist-plugin.jar \
  https://github.com/nextprocurement/NP-solr-dist-plugin/raw/main/NP-solr-dist-plugin.jar
```

### 4. Build and start services

```bash
docker compose up -d --build
```

To follow the logs:

```bash
docker compose logs -f sia-core-api
```

### 5. Generate an API key

Once the API is running, use the master key to generate an API key for regular access:

```bash
curl -X POST "http://<host>:<port>/admin/api-keys" \
  -H "X-API-Key: your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-client"}'
```

---

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
