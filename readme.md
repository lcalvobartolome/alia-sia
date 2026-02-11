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