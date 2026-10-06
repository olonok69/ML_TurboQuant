"""
memo_bench.py - can vector compression (TurboQuant & co.) help comment-memo response suggestions?

Rebuilds the suggestion retrieval path locally and measures it:
  * corpus: comment-memo questions/responses from extracted memo JSONs ({"chains": [{"query_response": [...]}]}),
    text normalised like the platform (visible HTML text, lower-cased), embedded with text-embedding-3-large (3072-d);
  * index: Qdrant (pin the platform's version), cosine, firm payload; search EXACT with a firm filter, own thread
    excluded, limit 50, score floor 0.5 then a recall retry at 0.42 - the platform's settings.

Sections
  0. Known-answer checks: Qdrant exact search == numpy exact search; the score floor filters exactly.
  1. Corpus.
  2. Score distribution around the 0.5 / 0.42 floors (how fragile the decisions are).
  3. Compression quality: top-10 overlap and suggestion decisions that flip at the floors
     (scalar int8, binary, TurboQuant 4/2-bit offline; Qdrant's own scalar/binary on the real collection).
  4. Latency of each step of one suggestion request (embedding call, Qdrant calls as the platform makes them).
  5. Scale: exact search latency and RAM vs collection size (synthetic copies of the real vectors).
  6. Optional: one LLM relevance-check-sized call, for scale against the vector search.

    python memo_bench.py --json-root <folder> --out <dir outside git> [--env-file .env] [--scale 10000,50000,200000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provision_bench as pb  # noqa: E402

MODEL, DIM = "text-embedding-3-large", 3072
FLOOR, RECALL_FLOOR, LIMIT = 0.5, 0.42, 50
COLL = "memo_bench"


def visible_lower(html: str) -> str:
    from bs4 import BeautifulSoup
    return " ".join(s.strip() for s in BeautifulSoup(html, "html.parser").stripped_strings).lower()


def load_corpus(root: Path):
    """Unique comments with (doc, chain) as thread and doc as matter. Returns DataFrame."""
    rows, seen = [], set()
    skip = {"node_modules", "knowledge-graph", "_prepull", "machine-sync"}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            p = Path(dp) / f
            if p.suffix != ".json" or p.stat().st_size > 30e6:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            d = d["data"] if isinstance(d.get("data"), dict) else d
            if not isinstance(d.get("chains"), list):
                continue
            doc = f.split(".doc")[0]
            for ch in d["chains"]:
                if not isinstance(ch, dict):
                    continue
                for it in ch.get("query_response") or []:
                    t = visible_lower(it.get("text") or "")
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    rows.append(dict(doc=doc, thread=f"{doc}#{ch.get('number')}", type=(it.get("type") or "").upper(), text=t))
    df = pd.DataFrame(rows)
    df["thread_id"] = df.thread.astype("category").cat.codes
    df["matter_id"] = df.doc.astype("category").cat.codes
    return df


def embed(texts, cache: Path, batch=100):
    import tiktoken
    from openai import OpenAI
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    have = {}
    if cache.exists():
        z = np.load(cache)
        have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in have]
    if todo:
        enc, cl = tiktoken.get_encoding("cl100k_base"), OpenAI()
        for a in range(0, len(todo), batch):
            idx = todo[a:a + batch]
            r = cl.embeddings.create(model=MODEL, input=[enc.decode(enc.encode(texts[i])[:8191]) for i in idx])
            for i, d in zip(idx, r.data):
                have[keys[i]] = np.asarray(d.embedding, np.float32)
        np.savez(cache, keys=np.array(list(have)), vecs=np.stack(list(have.values())))
    return np.stack([have[k] for k in keys])


# ---------------------------------------------------------------------------------- Qdrant


def qclient(url):
    from qdrant_client import QdrantClient
    assert "localhost" in url or "127.0.0.1" in url, "this benchmark only writes to a local Qdrant"
    return QdrantClient(url=url, timeout=600)


def build_collection(qc, name, X, firm, thread, quant=None, on_disk=False):
    from qdrant_client import models as m
    if qc.collection_exists(name):
        qc.delete_collection(name)
    qcfg = None
    if quant == "scalar":
        qcfg = m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, quantile=0.99, always_ram=True))
    elif quant == "binary":
        qcfg = m.BinaryQuantization(binary=m.BinaryQuantizationConfig(always_ram=True))
    qc.create_collection(name, vectors_config=m.VectorParams(size=X.shape[1], distance=m.Distance.COSINE, on_disk=on_disk),
                         quantization_config=qcfg)
    qc.create_payload_index(name, "firm_id", m.PayloadSchemaType.INTEGER)
    B = 300  # 3072-d floats as JSON: 1000 points = 65 MB > Qdrant 32 MB request limit
    for a in range(0, len(X), B):
        qc.upsert(name, points=m.Batch(ids=list(range(a, min(a + B, len(X)))), vectors=X[a:a + B].tolist(),
                                       payloads=[{"firm_id": int(firm[i]), "thread_id": int(thread[i])} for i in range(a, min(a + B, len(X)))]),
                  wait=True)


def platform_search(qc, name, qv, firm_id, exclude_thread, floor=FLOOR, exact=True, quant_ignore=None, check_exists=True):
    """One search exactly as the platform makes it: collection_exists, then query_points EXACT with
    must firm_id / must_not thread_id, limit 50, score_threshold."""
    from qdrant_client import models as m
    if check_exists:
        qc.collection_exists(name)
    params = m.SearchParams(exact=exact)
    if quant_ignore is not None:
        params.quantization = m.QuantizationSearchParams(ignore=quant_ignore, rescore=not quant_ignore)
    flt = m.Filter(must=[m.FieldCondition(key="firm_id", match=m.MatchValue(value=int(firm_id)))],
                   must_not=[m.FieldCondition(key="thread_id", match=m.MatchValue(value=int(exclude_thread)))])
    r = qc.query_points(name, query=qv.tolist(), query_filter=flt, search_params=params, limit=LIMIT,
                        score_threshold=floor, with_payload=True)
    return r.points


def np_search(X, q, thread, exclude_thread, floor=FLOOR):
    s = X @ X[q]
    s[thread == exclude_thread] = -2
    order = np.argsort(-s)[:LIMIT]
    return [(int(i), float(s[i])) for i in order if s[i] >= floor]


# ---------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qdrant", default="http://127.0.0.1:6333")
    ap.add_argument("--env-file")
    ap.add_argument("--scale", default="10000,50000,200000")
    ap.add_argument("--llm-model", default="", help="optional: time a relevance-check-sized call with this chat model")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = [f"# Comment-memo suggestion benchmark\n", f"Run {time.strftime('%Y-%m-%d %H:%M')} · model {MODEL} · Qdrant {a.qdrant}\n"]
    fails = []

    # 1 corpus --------------------------------------------------------------------------
    df = load_corpus(Path(a.json_root))
    X_raw = embed(df.text.tolist(), out / "memo_embeddings.npz")
    X = pb.normalize(X_raw)
    thread, firm = df.thread_id.to_numpy(), np.ones(len(df), int)
    qrows = np.flatnonzero(df.type.to_numpy() == "Q")
    rep += ["## 1. Corpus\n", f"{len(df)} unique comments ({(df.type == 'Q').sum()} questions, {(df.type == 'R').sum()} responses) "
            f"from {df.doc.nunique()} memo documents, {df.thread.nunique()} threads. Queries = the questions.\n"]
    print("1 corpus", len(df))

    qc = qclient(a.qdrant)
    build_collection(qc, COLL, X, firm, thread)

    # 0 known-answer checks -------------------------------------------------------------
    agree = 0
    for q in qrows[:200]:
        got = [(p.id, p.score) for p in platform_search(qc, COLL, X[q], 1, thread[q])]
        want = np_search(X, q, thread, thread[q])
        agree += [i for i, _ in got] == [i for i, _ in want] and all(abs(s1 - s2) < 1e-4 for (_, s1), (_, s2) in zip(got, want))
    ok = agree == min(200, len(qrows))
    rep += ["## 0. Known-answer checks\n", f"Qdrant exact search (firm filter, own thread excluded, floor {FLOOR}) equals numpy exact "
            f"search on {agree}/{min(200, len(qrows))} queries (ids, order and scores to 1e-4): **{'PASS' if ok else 'FAIL'}**\n"]
    if not ok:
        fails.append("C0 qdrant==numpy")
    print("0 canary", agree)

    # 2 score distribution --------------------------------------------------------------
    best, n50, n42 = [], [], []
    for q in qrows:
        s = X @ X[q]; s[thread == thread[q]] = -2
        best.append(s.max()); n50.append(int((s >= FLOOR).sum())); n42.append(int((s >= RECALL_FLOOR).sum()))
    best, n50, n42 = map(np.array, (best, n50, n42))
    S_true, I_true = pb.exact_topk(X, qrows, LIMIT)
    near = float(np.mean(np.abs(S_true - FLOOR) < 0.01))
    dist = dict(queries=len(qrows), top1_median=round(float(np.median(best)), 3), top1_p10=round(float(np.quantile(best, .1)), 3),
                top1_p90=round(float(np.quantile(best, .9)), 3), queries_with_any_ge_050_pct=round(100 * float((n50 > 0).mean()), 1),
                queries_needing_recall_retry_pct=round(100 * float((n50 == 0).mean()), 1),
                avg_results_ge_050=round(float(n50.mean()), 1), share_of_top50_scores_within_001_of_floor_pct=round(100 * near, 2))
    health, _, _ = pb.embedding_health(X_raw, X, S_true, I_true)
    rep += ["## 2. Scores around the floors\n", pd.DataFrame([dist]).T.reset_index().to_markdown(index=False, headers=["metric", "value"]) + "\n",
            "Embedding health:\n", pd.DataFrame([health]).T.reset_index().to_markdown(index=False, headers=["metric", "value"]) + "\n"]

    # 3 compression quality --------------------------------------------------------------
    a_, b_, s_ = pb.candidate_pairs(qrows, S_true, I_true, 0.3)
    comp = []
    for meth in [pb.ScalarInt8(), pb.Binary1(), pb.TurboQuantRef(4), pb.TurboQuantRef(2)]:
        meth.build(X)
        _, I = meth.search(X, qrows, 10)
        row = dict(method=meth.name, bytes_per_vector=round(meth.nbytes() / len(X)), top10_overlap=round(pb.recall_at_k(I, I_true[:, :10], 10), 4))
        if meth.reconstructs:
            approx = pb.approx_pair_scores(meth, X, a_, b_)
            for fl in (FLOOR, RECALL_FLOOR):
                f = pb.merge_flips(s_, approx, fl)
                row[f"flips@{fl} (added/dropped of {f['merges_exact']})"] = f"{f['false_merges']}/{f['missed_merges']}"
        comp.append(row)
    for quant in ("scalar", "binary"):
        build_collection(qc, COLL + "_" + quant, X, firm, thread, quant=quant)
        added = dropped = same_top10 = 0
        for q in qrows:
            ex = {p.id for p in platform_search(qc, COLL, X[q], 1, thread[q], check_exists=False)}
            qq = platform_search(qc, COLL + "_" + quant, X[q], 1, thread[q], quant_ignore=False, check_exists=False)
            got = {p.id for p in qq}
            added += len(got - ex); dropped += len(ex - got)
        comp.append(dict(method=f"Qdrant {quant} quantization (exact search, rescore on)", bytes_per_vector=None, top10_overlap=None,
                         **{f"flips@{FLOOR} (added/dropped of {sum(n50)})": f"{added}/{dropped}"}))
        qc.delete_collection(COLL + "_" + quant)
    comp = pd.DataFrame(comp); comp.to_csv(out / "3_compression.csv", index=False)
    rep += ["## 3. Compression quality\n", "`flips` = suggestions that would be added / dropped at the floor because the score is "
            "computed on compressed vectors (pairs from each question's exact top-50 with cosine ≥ 0.3).\n", comp.to_markdown(index=False) + "\n"]
    print("3 compression done")

    # 4 latency per step ----------------------------------------------------------------
    from openai import OpenAI
    cl = OpenAI(); emb_ms = []
    for t in df.text.iloc[qrows[:20]]:
        t0 = time.perf_counter(); cl.embeddings.create(model=MODEL, input=[t]); emb_ms.append((time.perf_counter() - t0) * 1000)
    ex_ms, call_ms = [], []
    for q in qrows[:300]:
        t0 = time.perf_counter(); qc.collection_exists(COLL); t1 = time.perf_counter()
        platform_search(qc, COLL, X[q], 1, thread[q], check_exists=False); t2 = time.perf_counter()
        ex_ms.append((t1 - t0) * 1000); call_ms.append((t2 - t1) * 1000)
    p = lambda v: f"{np.median(v):.1f} / {np.quantile(v, .95):.1f}"
    lat = pd.DataFrame([dict(step="OpenAI embedding of one comment (3-large, from this machine)", p50_p95_ms=p(emb_ms)),
                        dict(step="Qdrant collection_exists (made before every search)", p50_p95_ms=p(ex_ms)),
                        dict(step=f"Qdrant exact search, {len(df)} points, firm filter", p50_p95_ms=p(call_ms))])
    rep += ["## 4. One suggestion request, step by step (local Qdrant; network to a real cluster adds per call)\n", lat.to_markdown(index=False) + "\n"]

    # 5 scale ----------------------------------------------------------------------------
    g = np.random.default_rng(0); scale = []
    for N in [int(x) for x in a.scale.split(",") if x]:
        base = X[g.integers(0, len(X), N)]
        Xs = pb.normalize(base + g.normal(0, 0.02, base.shape).astype(np.float32))
        thr = g.integers(0, N // 5 + 1, N)
        for quant in (None, "scalar"):
            name = f"{COLL}_n{N}_{quant or 'float'}"
            t0 = time.time(); build_collection(qc, name, Xs, np.ones(N, int), thr, quant=quant); build = time.time() - t0
            ms = []
            for q in g.integers(0, N, 50):
                t0 = time.perf_counter()
                platform_search(qc, name, Xs[q], 1, thr[q], quant_ignore=None if quant is None else False, check_exists=False)
                ms.append((time.perf_counter() - t0) * 1000)
            scale.append(dict(points_in_firm=N, storage=quant or "float32", build_s=round(build, 1), exact_search_p50_ms=round(float(np.median(ms)), 1),
                              exact_search_p95_ms=round(float(np.quantile(ms, .95)), 1), vector_RAM_MB=round(N * (DIM * 4 if quant is None else DIM) / 2 ** 20)))
            qc.delete_collection(name)
        del Xs, base
        print("5 scale", N)
    for N in (100_000, 1_000_000):
        scale.append(dict(points_in_firm=N, storage="(projection) TurboQuant 4-bit", vector_RAM_MB=round(N * (DIM // 2 + 2) / 2 ** 20)))
    scale = pd.DataFrame(scale); scale.to_csv(out / "5_scale.csv", index=False)
    rep += ["## 5. Scale (all points in one firm = worst case for an exact, firm-filtered search)\n", scale.to_markdown(index=False) + "\n"]

    # 6 optional LLM call ----------------------------------------------------------------
    if a.llm_model:
        cands = "\n".join(f"{i + 1}. {df.text.iloc[j][:600]}" for i, j in enumerate(I_true[0, :12]))
        prompt = (f"Question:\n{df.text.iloc[qrows[0]][:800]}\n\nCandidate past responses:\n{cands}\n\n"
                  "Score each candidate 0-100 for how well it answers the question. Reply as JSON {\"scores\": [...]}.")
        ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            cl.chat.completions.create(model=a.llm_model, messages=[{"role": "user", "content": prompt}], reasoning_effort="low")
            ms.append((time.perf_counter() - t0) * 1000)
        rep += ["## 6. One relevance-check-sized LLM call (approximation of the platform's gate)\n",
                f"model `{a.llm_model}`, 12 candidates, reasoning low: p50 {np.median(ms):.0f} ms, max {max(ms):.0f} ms\n"]

    qc.delete_collection(COLL)
    (out / "REPORT.md").write_text("\n".join(rep))
    print("report", out / "REPORT.md", "FAILS:" if fails else "", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
