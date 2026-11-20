
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SEED = 42
MIN_PAPERS_FOR_MAP = 15
PROJECTION_METHOD = "umap"
CLUSTER_DIM = 10
CLUSTER_UMAP_MIN_DIST = 0.0
CLUSTER_MIN_SAMPLES = 5
NOISE_RETRY_FRACTION = 0.6
CLUSTER_MIN_RETRY_SIZE = 3


VECTOR_SOURCE_PAPER = "title_abstract"
VECTOR_SOURCE_CHUNKS = "chunk_mean"


class MapDependencyMissing(Exception):
    pass


class MapUnavailable(Exception):
    def __init__(self, reason: str, *, needed: int, have: int) -> None:
        super().__init__(f"{reason}: need {needed}, have {have}")
        self.reason = reason
        self.needed = needed
        self.have = have


@dataclass(frozen=True)
class MapPoint:
    paper_id: str
    x: float
    y: float
    cluster: int


@dataclass(frozen=True)
class MapPayload:
    points: list[MapPoint]
    method: str
    params_hash: str
    excluded: int
    cluster_note: str | None = None
    vector_source: str = VECTOR_SOURCE_CHUNKS

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [
                {
                    "paper_id": point.paper_id,
                    "x": point.x,
                    "y": point.y,
                    "cluster": point.cluster,
                }
                for point in self.points
            ],
            "method": self.method,
            "params_hash": self.params_hash,
            "excluded": self.excluded,
            "cluster_note": self.cluster_note,
            "vector_source": self.vector_source,
        }


@dataclass(frozen=True)
class PaperVectors:
    paper_ids: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    excluded: int = 0


def _unit(vector: Sequence[float]) -> list[float] | None:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0 or not math.isfinite(norm):
        return None
    return [value / norm for value in vector]


def paper_vectors(conn: Any, collection_id: str) -> tuple[list[str], list[list[float]], int]:
    rows = conn.execute(
        """
        SELECT c.paper_id, c.id AS chunk_id
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        WHERE p.collection_id=?
        ORDER BY c.paper_id, c.seq
        """,
        (collection_id,),
    ).fetchall()
    if not rows:
        total = conn.execute(
            "SELECT count(*) FROM papers WHERE collection_id=?", (collection_id,)
        ).fetchone()[0]
        return [], [], int(total)

    vector_rows = conn.execute(
        """
        SELECT e.owner_id, e.dim, e.vec
        FROM embeddings e
        JOIN chunks c ON c.id = e.owner_id
        JOIN papers p ON p.id = c.paper_id
        WHERE e.owner_type='chunk' AND p.collection_id=?
        """,
        (collection_id,),
    ).fetchall()

    from store.repo import unpack_vector

    by_chunk = {
        row["owner_id"]: unpack_vector(bytes(row["vec"]), int(row["dim"]))
        for row in vector_rows
    }

    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        paper_id = row["paper_id"]
        vector = by_chunk.get(row["chunk_id"])
        if vector is None:
            continue
        running = sums.get(paper_id)
        if running is None:
            sums[paper_id] = list(vector)
        else:
            if len(running) != len(vector):
                raise ValueError(
                    f"paper {paper_id} has chunks with mismatched dimensions: "
                    f"{len(running)} vs {len(vector)}"
                )
            for index, value in enumerate(vector):
                running[index] += value
        counts[paper_id] = counts.get(paper_id, 0) + 1

    paper_ids: list[str] = []
    vectors: list[list[float]] = []
    for paper_id in sorted(sums):
        count = counts[paper_id]
        mean = [value / count for value in sums[paper_id]]
        unit = _unit(mean)
        if unit is None:
            continue
        paper_ids.append(paper_id)
        vectors.append(unit)

    total_papers = conn.execute(
        "SELECT count(*) FROM papers WHERE collection_id=?", (collection_id,)
    ).fetchone()[0]
    excluded = int(total_papers) - len(paper_ids)
    return paper_ids, vectors, excluded


def abstract_vectors(
    conn: Any, collection_id: str
) -> tuple[list[str], list[list[float]], int]:
    from store.repo import unpack_vector

    rows = conn.execute(
        """
        SELECT e.owner_id, e.dim, e.vec
        FROM embeddings e JOIN papers p ON p.id = e.owner_id
        WHERE e.owner_type='paper' AND p.collection_id=?
        ORDER BY e.owner_id
        """,
        (collection_id,),
    ).fetchall()
    paper_ids: list[str] = []
    vectors: list[list[float]] = []
    for row in rows:
        unit = _unit(unpack_vector(bytes(row["vec"]), int(row["dim"])))
        if unit is not None:
            paper_ids.append(row["owner_id"])
            vectors.append(unit)
    total = conn.execute(
        "SELECT count(*) FROM papers WHERE collection_id=?", (collection_id,)
    ).fetchone()[0]
    return paper_ids, vectors, int(total) - len(paper_ids)


def select_vectors(
    conn: Any, collection_id: str
) -> tuple[list[str], list[list[float]], int, str]:
    ids, vectors, excluded = abstract_vectors(conn, collection_id)
    if ids and excluded == 0:
        return ids, vectors, excluded, VECTOR_SOURCE_PAPER
    ids, vectors, excluded = paper_vectors(conn, collection_id)
    return ids, vectors, excluded, VECTOR_SOURCE_CHUNKS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def params_hash(
    paper_ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    *,
    excluded: int = 0,
    method: str = PROJECTION_METHOD,
    seed: int = UMAP_SEED,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    source: str = VECTOR_SOURCE_CHUNKS,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{method}|{seed}|{n_neighbors}|{len(vectors)}|{excluded}|{source}"
        f"|cd{CLUSTER_DIM}|ms{CLUSTER_MIN_SAMPLES}|cmd{CLUSTER_UMAP_MIN_DIST}|md{UMAP_MIN_DIST}"
        f"|nr{NOISE_RETRY_FRACTION}|rs{CLUSTER_MIN_RETRY_SIZE}".encode("utf-8")
    )
    for paper_id in paper_ids:
        digest.update(paper_id.encode("utf-8"))
        digest.update(b"\x00")
    for row in vectors:
        for value in row:
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()[:32]


def project_2d(
    vectors: Sequence[Sequence[float]],
    *,
    method: str = PROJECTION_METHOD,
    seed: int = UMAP_SEED,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    n_components: int = 2,
    min_dist: float = UMAP_MIN_DIST,
) -> list[tuple[float, ...]]:
    if method != PROJECTION_METHOD:
        raise ValueError(f"unknown projection method {method!r}")
    if len(vectors) < 3:
        raise ValueError("a projection needs at least three points")
    n_components = max(2, min(n_components, len(vectors) - 2))

    try:
        import numpy as np
        import umap
    except ImportError as exc:
        raise MapDependencyMissing(
            f"the similarity map needs umap-learn and numpy ({exc}); "
            "pip install umap-learn hdbscan numpy scikit-learn"
        ) from exc

    matrix = np.asarray(vectors, dtype="float32")
    neighbours = max(2, min(n_neighbors, len(vectors) - 1))
    reducer = umap.UMAP(
        n_neighbors=neighbours,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        init="random",
        random_state=seed,
        n_jobs=1,
        compatibility_layout=True,
    )
    embedding = reducer.fit_transform(matrix)
    return [tuple(float(value) for value in row) for row in embedding]


def assign_clusters(
    points: Sequence[Sequence[float]], *, min_cluster_size: int
) -> list[int]:
    try:
        import hdbscan
        import numpy as np
    except ImportError as exc:
        raise MapDependencyMissing(
            f"clustering needs hdbscan and numpy ({exc}); "
            "pip install umap-learn hdbscan numpy scikit-learn"
        ) from exc

    if not points:
        return []
    labels = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size),
        min_samples=CLUSTER_MIN_SAMPLES,
        metric="euclidean",
    ).fit_predict(np.asarray(points, dtype="float32"))
    return [int(label) for label in labels]


def _noise_fraction(labels: Sequence[int]) -> float:
    if not labels:
        return 1.0
    return sum(1 for label in labels if label < 0) / len(labels)


def _cluster_with_retry(
    coordinates: Sequence[Sequence[float]],
    clusterer: Callable[..., list[int]],
    count: int,
) -> tuple[list[int], str | None]:
    min_cluster_size = max(5, count // 50)
    labels = list(clusterer(coordinates, min_cluster_size=min_cluster_size))
    if len(labels) != count:
        raise ValueError(f"clusterer returned {len(labels)} labels for {count} papers")

    if _noise_fraction(labels) <= NOISE_RETRY_FRACTION:
        return labels, None

    smaller = max(CLUSTER_MIN_RETRY_SIZE, min_cluster_size // 2)
    if smaller < min_cluster_size:
        retried = list(clusterer(coordinates, min_cluster_size=smaller))
        if len(retried) != count:
            raise ValueError(
                f"clusterer returned {len(retried)} labels for {count} papers"
            )
        if _noise_fraction(retried) <= NOISE_RETRY_FRACTION:
            return retried, None
        labels = retried

    return [-1] * count, "clusters_unreliable"


def build_map(
    conn: Any,
    collection_id: str,
    *,
    min_papers: int = MIN_PAPERS_FOR_MAP,
    method: str = PROJECTION_METHOD,
    seed: int = UMAP_SEED,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    projector: Callable[..., list[tuple[float, ...]]] | None = None,
    clusterer: Callable[..., list[int]] | None = None,
) -> MapPayload:
    paper_ids, vectors, excluded, source = select_vectors(conn, collection_id)
    have = len(paper_ids)
    if have < min_papers:
        raise MapUnavailable("too_few_papers", needed=min_papers, have=have)

    fingerprint = params_hash(
        paper_ids,
        vectors,
        excluded=excluded,
        method=method,
        seed=seed,
        n_neighbors=n_neighbors,
        source=source,
    )
    cached = conn.execute(
        "SELECT method, params_hash, payload FROM projections WHERE collection_id=?",
        (collection_id,),
    ).fetchone()
    if cached is not None and cached["params_hash"] == fingerprint:
        stored = json.loads(cached["payload"])
        return MapPayload(
            points=[MapPoint(**point) for point in stored["points"]],
            method=stored["method"],
            params_hash=stored["params_hash"],
            excluded=stored["excluded"],
            cluster_note=stored.get("cluster_note"),
            vector_source=stored.get("vector_source", VECTOR_SOURCE_CHUNKS),
        )

    coordinates = (projector or project_2d)(
        vectors, method=method, seed=seed, n_neighbors=n_neighbors
    )
    if len(coordinates) != have:
        raise ValueError(
            f"projection returned {len(coordinates)} points for {have} papers"
        )

    if projector is None:
        cluster_space = project_2d(
            vectors,
            method=method,
            seed=seed,
            n_neighbors=n_neighbors,
            n_components=CLUSTER_DIM,
            min_dist=CLUSTER_UMAP_MIN_DIST,
        )
    else:
        cluster_space = coordinates
    labels, cluster_note = _cluster_with_retry(
        cluster_space, clusterer or assign_clusters, have
    )

    payload = MapPayload(
        points=[
            MapPoint(paper_id=paper_id, x=float(x), y=float(y), cluster=int(label))
            for paper_id, (x, y), label in zip(paper_ids, coordinates, labels)
        ],
        method=f"{method}-hdbscan",
        params_hash=fingerprint,
        excluded=excluded,
        cluster_note=cluster_note,
        vector_source=source,
    )

    with conn:
        conn.execute(
            """
            INSERT INTO projections (collection_id, method, params_hash, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(collection_id) DO UPDATE SET
                method=excluded.method,
                params_hash=excluded.params_hash,
                payload=excluded.payload,
                created_at=excluded.created_at
            """,
            (
                collection_id,
                payload.method,
                payload.params_hash,
                json.dumps(payload.as_dict()),
                _now(),
            ),
        )
    return payload
