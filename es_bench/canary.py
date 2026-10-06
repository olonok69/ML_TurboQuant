"""
canary.py - known-answer checks for every instrument in es_provision_bench.py. Run before trusting
any benchmark number. Exits non-zero on the first failure.

Needs: the local Elasticsearch from docker-compose.yml (port 9201) and a throwaway Postgres with
pg_trgm for the trigram check:
    docker run --rm -d --name pg-trgm -e POSTGRES_PASSWORD=x postgres:16-alpine
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
from elasticsearch import Elasticsearch
from rapidfuzz import fuzz

import es_provision_bench as eb
import provision_bench as pb
from make_local_pd import create_pd_index

ES = "http://127.0.0.1:9201"
g = np.random.default_rng(7)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


# a small corpus with known near-duplicates (clean, ASCII, no punctuation, like provision_clean_description)
base = [
    "The Investor shall have the right to redeem upon 30 days notice",
    "Management Fee shall be 2 of NAV payable quarterly in advance",
    "The General Partner may establish side pocket accounts for illiquid investments",
    "Each Investor shall maintain the confidentiality of all information received from the Fund",
    "The Fund shall provide the Investor with quarterly unaudited financial statements",
]
texts = []
for b in base:
    texts += [b, b.replace("30", "90").replace("2 of", "20 of"), b + " This provision survives termination",
              " ".join(reversed(b.split()))]
texts += ["".join(g.choice(list("abcdefghij klmnop"), 60)) for _ in range(20)]

# C1 fuzz_truth equals rapidfuzz.fuzz.ratio pair by pair (the monolith's scorer, rounded like uint8 cdist)
qidx = np.arange(len(texts))
T = eb.fuzz_truth(texts, qidx)
ref = np.array([[0 if i == j else int(fuzz.ratio(texts[i], texts[j]) + 0.5) for j in range(len(texts))]
                for i in range(len(texts))])
diff = np.abs(T.astype(int) - ref)
check("C1 fuzz_truth == fuzz.ratio loop", diff.max() <= 1, f"max diff {diff.max()}, exact {np.mean(diff == 0):.3f}")

# C2 trigram similarity equals Postgres pg_trgm similarity()
pairs = [(i, j) for i in range(0, len(texts), 3) for j in range(1, len(texts), 4) if i != j][:60]
vals = ",".join(f"({k},$q${texts[i]}$q$,$q${texts[j]}$q$)" for k, (i, j) in enumerate(pairs))
sql = f"CREATE EXTENSION IF NOT EXISTS pg_trgm; SELECT k, similarity(a,b) FROM (VALUES {vals}) t(k,a,b) ORDER BY k;"
try:
    out = subprocess.run(["docker", "exec", "pg-trgm", "psql", "-U", "postgres", "-At", "-c", sql],
                         capture_output=True, text=True, check=True).stdout.split()
    pg = np.array([float(x.split("|")[1]) for x in out if "|" in x])
    M, sizes = eb.trigram_matrix(texts)
    ours = np.array([eb.trigram_similarity(M, sizes, np.array([i]))[0, j] for i, j in pairs])
    check("C2 trigram_similarity == pg_trgm similarity()", np.allclose(ours, pg, atol=1e-6),
          f"{len(pg)} pairs, max diff {np.abs(ours - pg).max():.2e}")
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    check("C2 trigram_similarity == pg_trgm similarity()", False, f"postgres not available: {e}")

# C3 candidate_recall on a hand-made case: 3 true pairs, candidates find 2
truth = np.array([[0, 80, 10, 75], [80, 0, 10, 10]], np.uint8)
cand = np.array([[1, 2, -1], [2, 3, -1]])
r = eb.candidate_recall(truth, 70, cand)
check("C3 candidate_recall hand case", (r["true_pairs"], r["found"], r["queries_fully_recalled"]) == (3, 1, 0), str(r))
cand2 = np.array([[1, 3, -1], [0, -1, -1]])
r2 = eb.candidate_recall(truth, 70, cand2)
check("C3b candidate_recall full", r2["pair_recall"] == 1.0 and r2["queries_fully_recalled"] == 2, str(r2))

# C4 cleaning matches the monolith's steps on a known input
html = "<p>The Investor&rsquo;s right (a) to <b>redeem</b>: 30&nbsp;days&#8217; notice.</p>"
want = "The Investors right a to redeem 30 days notice"
got = eb.monolith_clean_description(html)
check("C4 monolith_clean_description", " ".join(got.split()) == want, repr(got))

# C5/C6 Elasticsearch: a flat (exact) index must reproduce exact_topk; load_sample must return exactly the
# unmerged rows of the requested firm; live_knn on it must equal exact search.
es = Elasticsearch(ES, request_timeout=120)
d, n = 64, 600
X = pb.normalize(g.standard_normal((n, d)).astype(np.float32))
q = g.choice(n, 50, replace=False)
_, I_true = pb.exact_topk(X, q, 10)
eb.build_local_index(es, "canary_flat", X, "flat")
I_flat, _ = eb.local_knn(es, "canary_flat", X, q, 10, 10)
check("C5 ES flat kNN == exact top-10", pb.recall_at_k(I_flat, I_true, 10) == 1.0,
      f"recall {pb.recall_at_k(I_flat, I_true, 10)}")
eb.build_local_index(es, "canary_hnsw", X, "hnsw")
I_h, _ = eb.local_knn(es, "canary_hnsw", X, q, 10, 200)
check("C5b ES hnsw kNN close to exact", pb.recall_at_k(I_h, I_true, 10) >= 0.95, f"recall {pb.recall_at_k(I_h, I_true, 10)}")

name = "canary_pd"
create_pd_index(es, name, dims=d, index_options={"type": "flat"})  # exact: tests the instrument, not quantization
from elasticsearch import helpers  # noqa: E402

firm = np.where(np.arange(n) % 3 == 0, 7, 9)
merged = np.arange(n) % 10 == 5
txt = [f"text {i}" if i % 17 else "" for i in range(n)]
helpers.bulk(es, ({"_index": name, "_id": i, "_source": {"id": 1000 + i, "firm_id": int(firm[i]), "is_merged": bool(merged[i]),
                                                       eb.TEXT_FIELD: txt[i], eb.VECTOR_FIELD: X[i].tolist()}}
                  for i in range(n)))
es.indices.refresh(index=name)
ids, firms, Xs, ts = eb.load_sample(es, name, firm_id=7, page=37)
want_rows = [i for i in range(n) if firm[i] == 7 and not merged[i] and txt[i]]
check("C6 load_sample = unmerged rows of the firm with text", sorted(ids.tolist()) == [1000 + i for i in want_rows],
      f"got {len(ids)} want {len(want_rows)}")
order = np.argsort(ids); ids, Xs = ids[order], Xs[order]
check("C6b vectors round-trip", np.allclose(Xs, X[[i - 1000 for i in ids]], atol=1e-6))
qq = np.arange(0, len(ids), 5)
_, It = pb.exact_topk(Xs, qq, 5)
Il, lat = eb.live_knn(es, Xs, ids, qq, 5, 500, index=name, firm_id=7)
check("C6c live_knn (firm filter) == exact top-5", pb.recall_at_k(Il, It, 5) >= 0.99, f"recall {pb.recall_at_k(Il, It, 5)}")
info = eb.index_info(es, name)
check("C6d index_info reads dims/similarity", info["vector_dims"] == d and info["vector_similarity"] == "cosine", str(info))
create_pd_index(es, "canary_default", dims=d)
dflt = eb.index_info(es, "canary_default")["vector_index_options_effective"]
print("   NOTE default index_options for the monolith mapping on ES", info["es_version"], "=", dflt)
es.options(ignore_status=404).indices.delete(index="canary_default")

# C7 the local-only guard refuses a remote host
try:
    eb.build_local_index(Elasticsearch("http://dev-tenant-elasticsearch:9200"), "x", X, "flat")
    check("C7 refuses non-local cluster", False)
except AssertionError:
    check("C7 refuses non-local cluster", True)

for ix in ("canary_flat", "canary_hnsw", name):
    es.options(ignore_status=404).indices.delete(index=ix)
print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
