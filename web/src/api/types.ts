export type CollectionStatus = 'building' | 'ready' | 'failed'
export type JobStatus = 'queued' | 'running' | 'done' | 'failed'
export type Mode = 'lexical' | 'tfidf' | 'dense' | 'hybrid'
export type Sort = 'relevance' | 'influence' | 'year'

export interface Health {
  status: string
  version: string
  embedding_model: string | null
  generation_available: boolean
  generation_reason: string | null
  generation_configured: boolean

  generation_last_error?: { code: string; message: string } | null
  refusal_tau?: number | null
  refusal_tau_mode?: string | null

  generation_provider?: string | null
  generation_model?: string | null

  generation_config_note?: string | null
}

export interface CollectionJob {
  id: string
  status: JobStatus

  progress: number

  message: string | null
}

export interface Collection {
  id: string
  mode: 'topic' | 'upload' | string
  query: string | null
  status: CollectionStatus
  error: string | null

  degraded?: string[]

  job?: CollectionJob | null
  embedding_model: string | null
  n_papers: number
  n_chunks: number
  created_at: string
  updated_at: string
}

export interface DeletedCollection {
  collection_id: string
  deleted: Record<string, number>
}

export interface Job {
  id: string
  collection_id: string | null
  kind: string
  status: JobStatus
  progress: number
  message: string | null
  error: string | null
  created_at: string
  finished_at: string | null
}

export interface Paper {
  paper_id: string
  title: string
  year: number | null
  citation_count: number | null
  score: number | null
  best_chunk_id: string | null
  abstract: string | null
  authors: string[]
  venue: string | null
  pdf_url: string | null
  arxiv_id: string | null
  parse_status: string
  parse_note: string | null
  has_fulltext: number
  source?: string

  source_url?: string | null
}

export interface PaperDetail {
  paper_id: string
  collection_id: string
  arxiv_id: string | null
  openalex_id: string | null
  doi: string | null
  title: string
  abstract: string | null
  authors: string[]
  year: number | null
  venue: string | null
  pdf_url: string | null
  citation_count: number | null
  source: string
  parse_status: string
  parse_note: string | null
  has_fulltext: number
  chunk_count: number
  source_url?: string | null
}

export interface PapersPage {
  papers: Paper[]
  mode: string
  degraded: string[]
  total: number
  offset: number
  limit: number
}

export interface MapPoint {
  paper_id: string
  x: number
  y: number
  cluster: number
}

export interface MapPayload {
  points: MapPoint[]
  method: string
  params_hash: string
  excluded: number
  cluster_note: string | null
  vector_source?: 'title_abstract' | 'chunk_mean' | string
}

export interface MapLabels {
  labels: Record<string, string>
  unavailable: { code: string; message: string } | null

  terms: Record<string, string[]>

  distinct: boolean

  collisions: [number, number][]

  unnamed: number[]

  uninformative: number[]

  ungrounded: Record<string, string[]>
}

export interface GraphNode {
  id: string
  title: string
  year: number | null
  citation_count: number | null
  resolved: boolean
  pagerank?: number | null

  cluster?: number | null
}

export interface GraphViewEntry {
  id: string
  title: string
  in_collection: boolean
  score: number
}

export type GraphViews =
  | {
      available: true
      pagerank: GraphViewEntry[]
      authority: GraphViewEntry[]
      hub: GraphViewEntry[]
      notes: string[]
      method: { pagerank: string; hits: string }
    }
  | { available: false; reason: string }

export interface GraphEdge {
  source: string
  target: string
  resolved: boolean
}

export interface GraphRanking {
  paper_id: string
  score: number
  complete: boolean
}

export interface GraphClusters {
  available: boolean

  reason: string | null

  ids: number[]

  unplaced: number
}

export interface GraphPayload {
  nodes: GraphNode[]
  edges: GraphEdge[]
  rankings: GraphRanking[]
  unidentified: number
  reason?: string
  needed?: number
  have?: number

  views?: GraphViews | Record<string, never>
  components?: { full?: number; internal?: number }

  clusters?: GraphClusters
}

export interface AskCitation {
  n: number
  paper_id: string
  chunk_id: string
  title: string
  year: number | null
  arxiv_id: string | null
  url: string | null
  quote: string
}

export type RefusalReason =
  | 'no_retrieval_hits'
  | 'below_threshold'
  | 'model_declined'
  | 'answer_had_no_valid_citation'

export interface AskResult {
  question: string
  answer: string | null
  refused: boolean
  refusal_reason: RefusalReason | string | null
  answer_unavailable: { code: string; message: string } | null
  citations: AskCitation[]
  confidence: number | null
  confidence_basis: string
  citation_anomaly: number[]
  retrieval: {
    mode: string
    degraded: string[]
    top_score: number | null
    n_passages: number
    tau: number | null
    tau_calibrated: boolean
  }
  model: string
  cached: boolean
  latency_ms: number
  answer_id?: string | number
}

export interface PaperSummary {
  paper_id: string
  title: string
  summary: string
  model: string
  cached: boolean
  sources: string[]
  unsupported_numbers: string[]
  ai_generated: true
}

export interface RelatedWork {
  paragraph: string
  citations: { key: string; paper_id: string; title: string; year: number | null; url: string | null }[]
  unknown_citations: string[]
  uncited_papers: string[]
  word_count: number
  ordered_by: string
  ai_assisted_draft: true
  label: string
  model: string
  cached: boolean
}

export type RelatedWorkRequest = { paper_ids: string[] } | { cluster_id: number }

export interface ComparedPaper {
  paper_id: string
  title: string
  year: number | null
  url: string | null
}

export type Verdict = 'agree' | 'conflict' | 'superseded' | 'no_comparable_claims'

export interface Comparison {
  paper_a: ComparedPaper
  paper_b: ComparedPaper
  verdict: Verdict | string

  quote_a: string | null
  quote_b: string | null

  source_a?: string | null
  source_b?: string | null
  reason: string | null
  note?: string
  model?: string
  cached?: boolean
}

export interface Contradictions {
  comparisons: Comparison[]
  n_pairs_requested: number

  n_pairs_possible: number
  unavailable: { code: string; message: string } | null

  incomplete_pair: { paper_a: ComparedPaper; paper_b: ComparedPaper } | null
  method: string
  note: string
  ai_generated: true
}

export type ContradictionRequest = RelatedWorkRequest
