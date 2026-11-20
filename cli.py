
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from services.collection import CollectionBuildError, build_collection
from store.db import connect, init_db


def _build(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    db_path = Path(args.db) if args.db else settings.db_path
    init_db(db_path)

    conn = connect(db_path)
    try:
        report = build_collection(
            conn,
            query=args.query,
            limit=args.limit,
            settings=settings,
            log=lambda message: print(message, file=sys.stderr, flush=True),
            reuse=not args.fresh,
        )
    except CollectionBuildError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "collection_id": report.collection_id,
                "n_papers": report.n_papers,
                "n_chunks": report.n_chunks,
                "n_embeddings": report.n_embeddings,
                "n_edges": report.n_edges,
                "degraded": report.degraded,
                "db": str(db_path),
            },
            indent=2,
        )
    )
    return 0


def _stats(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    db_path = Path(args.db) if args.db else settings.db_path
    if not db_path.is_file():
        print(json.dumps({"error": f"no database at {db_path}"}), file=sys.stderr)
        return 1

    conn = connect(db_path)
    try:
        collection = conn.execute(
            "SELECT * FROM collections WHERE id=?", (args.collection,)
        ).fetchone()
        if collection is None:
            print(json.dumps({"error": "collection not found"}), file=sys.stderr)
            return 1

        parse_status = {
            row["parse_status"]: row["n"]
            for row in conn.execute(
                "SELECT parse_status, count(*) AS n FROM papers WHERE collection_id=? GROUP BY parse_status",
                (args.collection,),
            )
        }
        top = [
            {"title": row["title"], "arxiv_id": row["arxiv_id"], "citations": row["citation_count"]}
            for row in conn.execute(
                """
                SELECT title, arxiv_id, citation_count FROM papers
                WHERE collection_id=?
                ORDER BY citation_count IS NULL, citation_count DESC
                LIMIT 5
                """,
                (args.collection,),
            )
        ]
        edges = conn.execute(
            "SELECT count(*) AS n, coalesce(sum(resolved), 0) AS resolved FROM edges WHERE collection_id=?",
            (args.collection,),
        ).fetchone()

        print(
            json.dumps(
                {
                    "collection_id": collection["id"],
                    "query": collection["query"],
                    "status": collection["status"],
                    "error": collection["error"],
                    "n_papers": collection["n_papers"],
                    "n_chunks": collection["n_chunks"],
                    "parse_status": parse_status,
                    "n_edges": edges["n"],
                    "n_edges_resolved": edges["resolved"],
                    "top_papers": top,
                },
                indent=2,
            )
        )
    finally:
        conn.close()
    return 0


SECTION_KEYS = {
    "refusal": "refusal_end_to_end",
    "summaries": "summary_rouge",
    "cost": "cost_per_query",
    "bootstrap": "retrieval_bootstrap",
}


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _make_llm(settings: Settings, db_path: Path):
    from services.llm import LLMClient

    return LLMClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        db_path=db_path,
        provider=settings.llm_provider,
    )


def _eval(args: argparse.Namespace) -> int:
    from evaluation import harness
    from services.query_embedder import QueryEmbedder

    settings = Settings.from_env()
    db_path = Path(args.db) if args.db else settings.db_path
    init_db(db_path)
    conn = connect(db_path)
    embedder = QueryEmbedder(model_dir=str(settings.models_dir))
    vector_fn = embedder.vector

    def read_json(path: str) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    try:
        name = args.section
        if name == "latency":
            result = harness.latency(conn, args.collection, vector_fn)
        elif name == "faiss":
            result = harness.faiss_oracle(conn, args.collection, vector_fn)
        elif name == "clustering":
            result = harness.clustering(conn, args.collection)
        elif name == "graph":
            result = harness.graph_corroboration(conn, args.collection, settings.cache_dir)
        elif name == "pool":
            queries = [
                line.strip()
                for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            pool = harness.retrieval_pool(conn, args.collection, queries, vector_fn)
            Path(args.out).write_text(json.dumps(pool, indent=2), encoding="utf-8")
            print(f"wrote {len(queries)} queries to {args.out}; label them, then run `eval retrieval`")
            return 0
        elif name == "retrieval":
            result = harness.retrieval_metrics(conn, args.collection, read_json(args.labels), vector_fn)
        elif name == "tau":
            labelled = read_json(args.labels)["questions"]
            result = harness.calibrate_tau(conn, args.collection, labelled, vector_fn)
        elif name == "faithfulness-pool":
            pool = harness.faithfulness_pool(conn, args.collection)
            Path(args.out).write_text(json.dumps(pool, indent=2), encoding="utf-8")
            print(f"wrote {len(pool['claims'])} claims to {args.out}; adjudicate, then `eval faithfulness`")
            return 0
        elif name == "faithfulness":
            result = harness.faithfulness_score(read_json(args.labels))
        elif name == "bootstrap":
            result = harness.retrieval_bootstrap(conn, args.collection, read_json(args.labels), vector_fn)
        elif name == "refusal":
            result = harness.refusal_end_to_end(
                conn, args.collection, read_json(args.labels), _make_llm(settings, db_path), vector_fn,
                refusal_tau=settings.refusal_tau, tau_mode=settings.refusal_tau_mode,
            )
        elif name == "summaries":
            bert_fn = lambda c, r: harness.bertscore_f1(c, r, model_type=args.bert_model)
            result = harness.summary_scores(
                conn, args.collection, _make_llm(settings, db_path), n=args.n or 20, bert_fn=bert_fn
            )
        elif name == "cost":
            with tempfile.TemporaryDirectory() as scratch:
                fresh = Path(scratch) / "cache.sqlite3"
                init_db(fresh)
                result = harness.cost_per_query(
                    conn, args.collection, read_json(args.labels), _make_llm(settings, fresh), vector_fn,
                    n_answers=args.n or 10, n_summaries=args.n or 10,
                )
        else:
            raise SystemExit(f"unknown section {name}")
    except harness.NoLabels as exc:
        print(json.dumps({"error": str(exc), "hint": "this section needs a labels file with judgements "
                          "and, for dense modes, query vectors"}), file=sys.stderr)
        return 2
    finally:
        conn.close()

    key = SECTION_KEYS.get(name, name)
    reason = harness.unusable_reason(key, result)
    if reason:
        print(json.dumps({"error": reason, "section": key, "stored_report": "left unchanged"}), file=sys.stderr)
        return 3
    harness.save_section(settings.data_dir, args.collection, key, result)
    print(json.dumps(result, indent=2))
    return 0


def _snapshot(args: argparse.Namespace) -> int:
    from store import snapshot

    settings = Settings.from_env()
    db_path = Path(args.db) if args.db else settings.db_path
    try:
        if args.action == "export":
            report = Path(settings.data_dir) / "eval" / f"{args.collection}.json"
            manifest = snapshot.export_collection(db_path, args.collection, args.out, evaluation_json=report)
            print(json.dumps({"written": args.out, "counts": manifest["counts"], "db_sha256": manifest["db_sha256"]}, indent=2))
        elif args.action == "verify":
            manifest = snapshot.verify(args.zip)
            print(json.dumps({"ok": True, "collection": manifest["collection"], "counts": manifest["counts"]}, indent=2))
        else:
            result = snapshot.import_snapshot(args.zip, db_path, replace=args.replace, data_dir=settings.data_dir)
            print(json.dumps({"imported_collection": result["collection_id"], "rows": result["imported"]}, indent=2))
    except snapshot.SnapshotError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli", description="Build, inspect, evaluate and snapshot paper collections."
    )
    parser.add_argument("--db", help="SQLite path (default: from settings)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a collection from a topic query")
    build.add_argument("--query", required=True)
    build.add_argument("--limit", type=int, default=100)
    build.add_argument(
        "--fresh",
        action="store_true",
        help="create a new collection instead of rebuilding the ready one for this query",
    )
    build.set_defaults(func=_build)

    stats = subparsers.add_parser("stats", help="inspect a stored collection")
    stats.add_argument("--collection", required=True)
    stats.set_defaults(func=_stats)

    evaluate = subparsers.add_parser("eval", help="measure the system on a stored collection")
    evaluate.add_argument(
        "section",
        choices=["latency", "faiss", "clustering", "graph", "pool", "retrieval", "tau",
                 "faithfulness-pool", "faithfulness", "bootstrap", "refusal", "summaries", "cost"],
    )
    evaluate.add_argument("--collection", required=True)
    evaluate.add_argument("--queries", help="text file, one query per line (pool)")
    evaluate.add_argument("--labels", help="JSON with judgements (retrieval, bootstrap, tau, faithfulness, refusal, cost)")
    evaluate.add_argument("--out", help="where to write a pool for labelling")
    evaluate.add_argument("--n", type=_positive_int, help="sample size (summaries: 20; cost: that many answers and summaries, 10)")
    evaluate.add_argument("--bert-model", default="roberta-large", help="BERTScore model (summaries)")
    evaluate.set_defaults(func=_eval)

    snap = subparsers.add_parser("snapshot", help="export / verify / import an offline snapshot of a collection")
    snap.add_argument("action", choices=["export", "verify", "import"])
    snap.add_argument("zip", nargs="?", help="snapshot file (verify, import)")
    snap.add_argument("--collection", help="collection id (export)")
    snap.add_argument("--out", help="output zip (export)")
    snap.add_argument("--replace", action="store_true", help="overwrite an existing collection with the same id (import)")
    snap.set_defaults(func=_snapshot)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
