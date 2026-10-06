"""
run_es_bench.py - provision benchmark against Elasticsearch (dev via port-forward, or the local stand-in).

    python run_es_bench.py --es http://127.0.0.1:9201 --out <results dir> [--firm-id N] [--local-es URL]

Sections (same numbering idea as provision_search_benchmark.ipynb):
  1. The live index (read-only): version, docs, vector mapping, effective index_options, segments.
  2. Load a sample (read-only): one firm's unmerged provisions with vectors + clean description.
  3. Embedding health (provision_bench.embedding_health).
  4. fuzz.ratio ground truth: how many neighbours each provision has above each store floor (K, d).
  5. Candidate recall: do cosine neighbours (float32 and compressed, incl. TurboQuant) or trigram
     candidates contain the pairs with fuzz.ratio >= floor?  <- the question that matters for the PD.
  6. Compression vs exact cosine (provision_bench.run_method), for completeness with the original.
  7. Live kNN on the index (read-only): latency and overlap with exact search.
  8. Elasticsearch's own quantization (index_options) on a LOCAL throwaway cluster only.
Writes CSVs and REPORT.md into --out. Nothing is written to the source cluster.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

import es_provision_bench as eb
import provision_bench as pb

FLOORS = (30, 40, 50, 60, 70, 80)
KS = (10, 25, 50, 100, 200, 500)


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if len(df) else "_(none)_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es", required=True, help="source cluster (read-only), e.g. port-forwarded dev")
    ap.add_argument("--index", default=eb.PD_INDEX)
    ap.add_argument("--firm-id", type=int, help="default: the firm with the most provisions")
    ap.add_argument("--limit", type=int, help="cap the sample size")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--local-es", help="LOCAL cluster for section 8 (never the source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = [f"# Provision benchmark (Elasticsearch) {a.label}\n", f"Run: {time.strftime('%Y-%m-%d %H:%M')} · source `{a.es}` · index `{a.index}`\n"]
    es = Elasticsearch(a.es, request_timeout=300)

    # 1 ---------------------------------------------------------------------------------
    info = eb.index_info(es, a.index)
    firms = eb.largest_firms(es, a.index)
    firm = a.firm_id if a.firm_id is not None else (firms[0][0] if firms else None)
    rep += ["## 1. Index\n", "```json\n" + json.dumps(info, indent=1) + "\n```\n",
            f"Largest firms (firm_id, provisions with a vector): {firms}\n", f"Firm used: **{firm}**\n"]
    print("1 index", info["docs"], "docs; firm", firm)

    # 2 ---------------------------------------------------------------------------------
    t = time.time()
    ids, _, X_raw, texts = eb.load_sample(es, a.index, firm_id=firm, limit=a.limit)
    X = pb.normalize(X_raw)
    n = len(ids)
    g = np.random.default_rng(0)
    qidx = np.sort(g.choice(n, min(a.queries, n), replace=False))
    rep += ["## 2. Sample\n", f"{n} provisions loaded in {time.time() - t:.1f}s; {len(qidx)} query provisions.\n"]
    print("2 sample", n)

    # 3 ---------------------------------------------------------------------------------
    S_true, I_true = pb.exact_topk(X, qidx, min(max(KS), n - 1))
    health, _, _ = pb.embedding_health(X_raw, X, S_true, I_true)
    rep += ["## 3. Embedding health\n", md(pd.DataFrame([health]).T.reset_index().rename(columns={"index": "metric", 0: "value"})) + "\n"]

    # 4 ---------------------------------------------------------------------------------
    t = time.time()
    F = eb.fuzz_truth(texts, qidx)
    hist = pd.DataFrame(eb.fuzz_score_histogram(F, FLOORS))
    hist.to_csv(out / "4_fuzz_neighbours_by_floor.csv", index=False)
    rep += ["## 4. fuzz.ratio neighbours per provision (the K / d the options page needs)\n",
            f"Exact fuzz.ratio of {len(qidx)} provisions against all {n} ({time.time() - t:.1f}s).\n", md(hist) + "\n"]
    print("4 fuzz truth done")

    # 5 ---------------------------------------------------------------------------------
    rows = []
    kmax = min(max(KS), n - 1)
    methods = [pb.Float32(), pb.ScalarInt8(), pb.Binary1(), pb.TurboQuantRef(4), pb.TurboQuantRef(2)]
    for m in methods:
        m.build(X)
        _, I = m.search(X, qidx, kmax)
        for k in KS:
            if k > kmax:
                continue
            for f in FLOORS:
                rows.append(dict(candidates=f"cosine top-k · {m.name}", k=k, **eb.candidate_recall(F, f, I[:, :k])))
    M, sizes = eb.trigram_matrix(texts)
    T = eb.trigram_similarity(M, sizes, qidx)
    It = eb.candidates_from_scores(T, kmax)
    for k in KS:
        if k > kmax:
            continue
        for f in FLOORS:
            rows.append(dict(candidates="trigram top-k (pg_trgm <->)", k=k, **eb.candidate_recall(F, f, It[:, :k])))
    for cut in (0.1, 0.2, 0.3, 0.4):
        C = eb.candidates_above(T, cut)
        avg = float((C >= 0).sum(1).mean())
        for f in FLOORS:
            rows.append(dict(candidates=f"trigram % cutoff {cut}", k=round(avg, 1), **eb.candidate_recall(F, f, C)))
    rec = pd.DataFrame(rows)
    rec.to_csv(out / "5_candidate_recall.csv", index=False)
    piv = rec[rec.floor.isin([30, 50, 70])].pivot_table(index=["candidates", "k"], columns="floor", values="pair_recall").reset_index()
    piv.columns = [str(c) if c in ("candidates", "k") else f"recall fuzz>={c}" for c in piv.columns]
    rep += ["## 5. Candidate recall vs fuzz.ratio\n",
            "Share of the pairs with fuzz.ratio >= floor that are among each provision's candidates. "
            "`k` = candidates per provision (for `%` cutoffs: the average number returned). "
            f"Sample size n = {n}, so k / n is the share of the firm scored.\n", md(piv) + "\n"]
    print("5 recall done")

    # 6 ---------------------------------------------------------------------------------
    k10 = min(10, n - 1)
    comp = []
    for m in [pb.Float32(), pb.ScalarInt8(), pb.Binary1(), pb.TurboQuantRef(4), pb.TurboQuantRef(2)]:
        comp += pb.run_method(m, X, qidx, I_true[:, :k10], k=k10)
    comp = pd.DataFrame(comp); comp.to_csv(out / "6_compression_vs_exact_cosine.csv", index=False)
    rep += ["## 6. Compression vs exact cosine (original notebook section 4)\n", md(comp) + "\n"]

    # 7 ---------------------------------------------------------------------------------
    live = []
    lq = qidx[:min(200, len(qidx))]
    for nc in (100, 500):
        I_live, lat = eb.live_knn(es, X, ids, lq, k10, nc, index=a.index, firm_id=firm)
        live.append(dict(num_candidates=nc, queries=len(lq), p50_ms=round(float(np.median(lat)), 1),
                         p95_ms=round(float(np.quantile(lat, 0.95)), 1),
                         overlap_with_exact_top10=round(pb.recall_at_k(I_live, I_true[:len(lq), :k10], k10), 4)))
    live = pd.DataFrame(live); live.to_csv(out / "7_live_knn.csv", index=False)
    rep += ["## 7. Live kNN on the index (read-only)\n",
            f"Effective index_options: `{info['vector_index_options_effective']}`.\n", md(live) + "\n"]
    print("7 live done")

    # 8 ---------------------------------------------------------------------------------
    if a.local_es:
        les = Elasticsearch(a.local_es, request_timeout=900)
        types = []
        for it in ("hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw", "flat"):
            name = f"bench_{it}"
            b = eb.build_local_index(les, name, X, it)
            for ov in ((None, 3.0) if it in ("int4_hnsw", "bbq_hnsw") else (None,)):
                I10, lat = eb.local_knn(les, name, X, lq, k10, 100, rescore_oversample=ov)
                Ik, _ = eb.local_knn(les, name, X, qidx, min(200, n - 1), min(1000, n), rescore_oversample=ov)
                r70 = eb.candidate_recall(F, 70, Ik)["pair_recall"]; r30 = eb.candidate_recall(F, 30, Ik)["pair_recall"]
                types.append(dict(index_type=it, rescore_oversample=ov, **b, p50_ms=round(float(np.median(lat)), 1),
                                  overlap_with_exact_top10=round(pb.recall_at_k(I10, I_true[:len(lq), :k10], k10), 4),
                                  **{"fuzz>=70 recall @200": r70, "fuzz>=30 recall @200": r30}))
            les.options(ignore_status=404).indices.delete(index=name)
        types = pd.DataFrame(types); types.to_csv(out / "8_es_index_types.csv", index=False)
        rep += ["## 8. Elasticsearch quantization (local throwaway cluster)\n", md(types) + "\n"]
        print("8 local types done")

    (out / "REPORT.md").write_text("\n".join(rep))
    print("report:", out / "REPORT.md")


if __name__ == "__main__":
    main()
