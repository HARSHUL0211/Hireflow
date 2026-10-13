import argparse
import json


def build_pipeline():
    # imported here so that `python main.py --help` is instant
    from core.pipeline import RecruitmentPipeline

    return RecruitmentPipeline()


def cmd_index(pipeline, args):
    count = pipeline.ingest_resumes()
    print(f"Indexed {count} resumes.")
    stats = pipeline.hybrid.get_index_stats()
    print(f"BM25 resumes: {stats['bm25_resumes']} | "
          f"FAISS vectors: {stats['vector_stats']['total_vector_count']} | "
          f"hybrid ready: {stats['hybrid_ready']}")


def cmd_search(pipeline, args):
    from utils.schemas import JobDescription

    if not pipeline.is_indexed():
        print("Index is empty - indexing resumes first...")
        pipeline.ingest_resumes()

    results = pipeline.search(
        args.query,
        top_k=args.top_k,
        search_mode=args.mode,
        required_skills=args.skills,
        locations=[args.location] if args.location else None,
        min_experience=args.min_exp,
    )
    if not results:
        print("No candidates found.")
        return

    for i, r in enumerate(results, start=1):
        m = r["metadata"]
        print(f"{i}. {r['name']:<22} score={r['combined_score']:<7} {m.get('title', '')} | "
              f"{m.get('experience', '?')} yrs | {m.get('location', '')}")
        print(f"   skills: {', '.join(m.get('skills', [])[:8])}")
    print(f"\nmode={results[0]['search_mode']}  queries={results[0]['queries_used']}")

    if args.rerank:
        jd = JobDescription(
            jd_id="cli",
            title=args.query,
            text=args.query,
            required_skills=args.skills or [],
            min_experience=int(args.min_exp) if args.min_exp is not None else None,
        )
        print("\n--- AI evaluation ---")
        for e in pipeline.evaluate_candidates(results, jd):
            print(f"{e.candidate_name}: {e.fit_score}/100 [{e.evidence['method']}]")
            print(f"  {e.summary}")
            print(f"  strengths: {e.strengths}")
            print(f"  gaps: {e.gaps}")


def cmd_evaluate(pipeline, args):
    if not pipeline.is_indexed():
        pipeline.ingest_resumes()
    metrics = pipeline.evaluator.evaluate_search_quality(
        args.query, args.skills or [], search_mode=args.mode, use_ragas=args.ragas
    )
    print(json.dumps(metrics.to_dict(), indent=2))


def cmd_stats(pipeline, args):
    print(json.dumps(pipeline.status(), indent=2, default=str))


def cmd_reset(pipeline, args):
    pipeline.reset_indexes()
    print("Indexes cleared. Your PDFs were not touched. Run `python main.py index` to rebuild.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HireFlow - AI recruitment platform")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="ingest and index every PDF in data/resumes").set_defaults(fn=cmd_index)

    s = sub.add_parser("search", help="search for candidates")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--mode", choices=["auto", "shallow", "deep"], default="auto")
    s.add_argument("--skills", nargs="*", help="required skills, e.g. --skills SQL Excel")
    s.add_argument("--location", help="location must contain this text")
    s.add_argument("--min-exp", type=float, help="minimum years of experience")
    s.add_argument("--rerank", action="store_true", help="add an AI fit-score evaluation")
    s.set_defaults(fn=cmd_search)

    e = sub.add_parser("evaluate", help="score search quality")
    e.add_argument("query")
    e.add_argument("--skills", nargs="*", help="skills a good result should have")
    e.add_argument("--mode", choices=["shallow", "deep"], default="deep")
    e.add_argument("--ragas", action="store_true", help="use RAGAS (slower, uses many API calls)")
    e.set_defaults(fn=cmd_evaluate)

    sub.add_parser("stats", help="show index status").set_defaults(fn=cmd_stats)
    sub.add_parser("reset", help="wipe the indexes (PDFs are kept)").set_defaults(fn=cmd_reset)
    return ap


def main():
    args = build_parser().parse_args()
    pipeline = build_pipeline()
    args.fn(pipeline, args)


if __name__ == "__main__":
    main()