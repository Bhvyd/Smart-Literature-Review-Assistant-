
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    query TEXT,
    status TEXT NOT NULL,
    error TEXT,
    degraded TEXT,
    embedding_model TEXT,
    n_papers INTEGER DEFAULT 0,
    n_chunks INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id),
    arxiv_id TEXT,
    openalex_id TEXT,
    doi TEXT,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    year INTEGER,
    venue TEXT,
    pdf_url TEXT,
    citation_count INTEGER,
    source TEXT NOT NULL,
    parse_status TEXT DEFAULT 'not_attempted',
    parse_note TEXT,
    has_fulltext INTEGER DEFAULT 0,
    created_at TEXT,
    UNIQUE (collection_id, arxiv_id),
    UNIQUE (collection_id, openalex_id),
    UNIQUE (collection_id, doi)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id),
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    n_tokens INTEGER,
    page INTEGER,
    section TEXT,
    UNIQUE (paper_id, seq)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    model TEXT NOT NULL,
    UNIQUE (owner_type, owner_id)
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id),
    src_paper_id TEXT NOT NULL REFERENCES papers(id),
    dst_paper_id TEXT,
    dst_openalex_id TEXT,
    dst_title TEXT,
    dst_doi TEXT,
    resolved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS projections (
    collection_id TEXT PRIMARY KEY REFERENCES collections(id),
    method TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id),
    question TEXT NOT NULL,
    answer TEXT,
    citations TEXT,
    refused INTEGER DEFAULT 0,
    model TEXT,
    latency_ms INTEGER,
    retrieval_snapshot TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    collection_id TEXT REFERENCES collections(id),
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0,
    message TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);
