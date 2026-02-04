# General Description

RESTful API for the Public Procurement and Grants Intelligence and Analysis System (SIA)

---

## Functional Blocks

### 1. Infrastructure Administration (`/admin`)

Low-level operations for technical management and dynamic system configuration:

- **Solr Collection Management**: Creation, listing, reloading and deletion of indexes
- **Display Configuration**: Metadata and search field management

### 2. Data Enrichment and Ingestion (`/processing`)

Services that orchestrate the information processing pipeline:

- **Ingestion Trigger**: Corpus and model indexing
- **AI Inference Services**: Thematic analysis, embeddings, semantic similarity

### 3. Exploitation Services (`/search`)

Block oriented to public consumption by the AI Portal:

- **Multimodal Search**: ...
- **Indicators and Statistics**: ...
- **Recommendations**: ...

---

## Error Handling

All error responses follow a standardized format:

```json
{
    "success": false,
    "error": "Error description",
    "error_code": "ERROR_CODE",
    "details": {}
}
```

## Error Codes

| Code | Description |
|--------|-------------|
| `BAD_REQUEST` | Invalid request parameters |
| `VALIDATION_ERROR` | Validation error |
| `NOT_FOUND` | Resource not found |
| `CONFLICT` | Resource already exists |
| `INTERNAL_ERROR` | Internal server error |
| `SOLR_ERROR` | Solr-specific error |
