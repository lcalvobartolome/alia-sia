# General Description

RESTful API for the Public Procurement and Grants Intelligence and Analysis System (SIA)

---

## Functional Blocks

### 1. Infrastructure Administration (`/admin`)

This block includes low-level operations for technical management and dynamic system configuration. It allows the creation, listing, reloading and deletion of Solr indexes, as well as the management of metadata and search field configurations used across the system.

### 2. Data Enrichment and Ingestion (`/processing`)

This block contains services that orchestrate the information processing pipeline. It enables corpus and model indexing through ingestion triggers and provides AI-powered inference services such as thematic analysis, embeddings generation and semantic similarity processing.

### 3. Exploitation Services (`/exploitation`)

This block provides the services consumed by PortalIA. It supports different types of search, including exact metadata queries, thematic search and semantic search, and also offers indicator calculation and recommendation services to enhance data exploration and decision-making.