"""
es_provision_bench.py - Elasticsearch side of the provision benchmark.

Reuses provision_bench.py (exact search, compression methods, embedding health) and adds:
  * read-only access to the Provision Database index in Elasticsearch (config, sample, live kNN),
  * local throwaway indexes to compare Elasticsearch's own vector quantization (index_options),
  * the fuzz.ratio recall test: can cosine neighbours (or trigram candidates) find the pairs the
    product actually scores, i.e. rapidfuzz.fuzz.ratio >= floor on provision_clean_description?

Conventions follow provision_bench.py: X is float32 [n, d], normalized; queries are row indices.
Everything that talks to a REMOTE cluster is read-only (_mapping, _settings, _stats, _search, PIT).
Index creation and deletion only ever happen on the LOCAL cluster passed as `local_es`.
"""
from __future__ import annotations

import re
import string
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import provision_bench as pb  # noqa: E402  (exact_topk, methods, embedding_health)

PD_INDEX = "provision_database_elastic_search"
VECTOR_FIELD = "provision_clean_embedding"
TEXT_FIELD = "provision_clean_description"

# Elasticsearch 8.18 dense_vector index types (index_options.type). The *_flat ones are brute force.
ES_INDEX_TYPES = ("hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw", "flat", "int8_flat", "int4_flat", "bbq_flat")


# ----------------------------------------------------------------------------------------
# the monolith's text cleaning (common/utils.py remove_html_tags, only_html=False)
# ----------------------------------------------------------------------------------------


def monolith_clean_description(html: str) -> str:
    """Same steps as the monolith's provision_clean_description: strip HTML, NFKD->ASCII,
    drop string.punctuation, strip. Whitespace runs are kept (the monolith keeps them too)."""
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "html.parser").get_text()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.translate(str.maketrans("", "", string.punctuation)).strip()


# ----------------------------------------------------------------------------------------
# read-only views of a cluster
# ----------------------------------------------------------------------------------------


def index_info(es, index: str = PD_INDEX) -> dict:
    """Config facts that decide vector search cost. Read-only."""
    info = es.info()
    mapping = es.indices.get_mapping(index=index)[index]["mappings"]
    vec = mapping.get("properties", {}).get(VECTOR_FIELD, {})
    full = es.indices.get_field_mapping(index=index, fields=VECTOR_FIELD, include_defaults=True)
    vec_full = full[index]["mappings"].get(VECTOR_FIELD, {}).get("mapping", {}).get(VECTOR_FIELD, {})
    stats = es.indices.stats(index=index, metric=["docs", "store", "segments"])["indices"][index]["primaries"]
    with_vec = es.count(index=index, query={"exists": {"field": VECTOR_FIELD}})["count"]
    return dict(
        es_version=info["version"]["number"], index=index,
        docs=stats["docs"]["count"], docs_with_vector=with_vec,
        store_MB=round(stats["store"]["size_in_bytes"] / 2 ** 20, 1),
        segments=stats["segments"]["count"],
        vector_mapping_explicit=vec,
        vector_index_options_effective=vec_full.get("index_options"),
        vector_dims=vec_full.get("dims"), vector_similarity=vec_full.get("similarity"),
    )


def largest_firms(es, index: str = PD_INDEX, top: int = 10) -> list[tuple[int, int]]:
    r = es.search(index=index, size=0, query={"exists": {"field": VECTOR_FIELD}},
                  aggs={"f": {"terms": {"field": "firm_id", "size": top}}})
    return [(b["key"], b["doc_count"]) for b in r["aggregations"]["f"]["buckets"]]


def population_query(firm_id: int | None = None) -> dict:
    """The rows the monolith scores: unmerged, with a vector, optionally one firm."""
    must = [{"exists": {"field": VECTOR_FIELD}}]
    if firm_id is not None:
        must.append({"term": {"firm_id": firm_id}})
    return {"bool": {"must": must, "must_not": [{"term": {"is_merged": True}}]}}


def load_sample(es, index: str = PD_INDEX, firm_id: int | None = None, limit: int | None = None,
                with_text: bool = True, page: int = 500):
    """Pull ids, vectors (and text) with a point-in-time + search_after. Read-only.
    Mirrors the monolith's similarity population: unmerged rows with a non-empty clean description."""
    query = population_query(firm_id)
    fields = ["id", "firm_id", VECTOR_FIELD] + ([TEXT_FIELD] if with_text else [])
    pit = es.open_point_in_time(index=index, keep_alive="2m")["id"]
    ids, firms, vecs, texts, after = [], [], [], [], None
    try:
        while limit is None or len(ids) < limit:
            body = dict(size=page, query=query, _source=fields,
                        pit={"id": pit, "keep_alive": "2m"}, sort=[{"_shard_doc": "asc"}])
            if after:
                body["search_after"] = after
            hits = es.search(**body)["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                s = h["_source"]
                if with_text and not (s.get(TEXT_FIELD) or "").strip():
                    continue
                ids.append(s["id"]); firms.append(s.get("firm_id")); vecs.append(s[VECTOR_FIELD])
                texts.append(s.get(TEXT_FIELD, ""))
            after = hits[-1]["sort"]
    finally:
        es.close_point_in_time(id=pit)
    if limit is not None:
        ids, firms, vecs, texts = ids[:limit], firms[:limit], vecs[:limit], texts[:limit]
    return np.array(ids), np.array(firms), np.asarray(vecs, dtype=np.float32), texts


def live_knn(es, X: np.ndarray, ids: np.ndarray, qidx: np.ndarray, k: int, num_candidates: int,
             index: str = PD_INDEX, firm_id: int | None = None):
    """Run the live kNN query once per sampled provision (self excluded). Read-only.
    Returns row indices into `ids` (-1 when a hit is outside the sample) and per-query latency in ms."""
    pos = {int(v): i for i, v in enumerate(ids)}
    filt = population_query(firm_id)
    fetch = 2 * k + 1   # over-fetch: rows outside the sample (e.g. empty text) are dropped below
    out, lat = np.full((len(qidx), k), -1, np.int64), []
    for r, q in enumerate(qidx):
        body = dict(knn=dict(field=VECTOR_FIELD, query_vector=X[q].tolist(), k=fetch,
                             num_candidates=max(num_candidates, fetch), filter=filt),
                    _source=["id"], size=fetch)
        t = time.perf_counter(); hits = es.search(index=index, **body)["hits"]["hits"]
        lat.append((time.perf_counter() - t) * 1000)
        rows = [pos.get(int(h["_source"]["id"]), -1) for h in hits]
        rows = [x for x in rows if x != q and x >= 0][:k]
        out[r, :len(rows)] = rows
    return out, np.array(lat)


# ----------------------------------------------------------------------------------------
# local throwaway indexes (Elasticsearch's own quantization)
# ----------------------------------------------------------------------------------------


def build_local_index(local_es, name: str, X: np.ndarray, index_type: str, m: int = 16, ef_construction: int = 100):
    """Create (or replace) a LOCAL index with one quantization type, bulk-load X, force-merge to 1 segment."""
    from elasticsearch import helpers

    assert index_type in ES_INDEX_TYPES, index_type
    host = str(local_es.transport.node_pool.get().base_url)
    assert "localhost" in host or "127.0.0.1" in host, f"refusing to create indexes on a non-local cluster: {host}"
    opts = {"type": index_type}
    if index_type.endswith("hnsw"):
        opts.update(m=m, ef_construction=ef_construction)
    local_es.options(ignore_status=404).indices.delete(index=name)
    local_es.indices.create(index=name, settings={"number_of_shards": 1, "number_of_replicas": 0},
                            mappings={"properties": {"row": {"type": "integer"},
                                                     "v": {"type": "dense_vector", "dims": X.shape[1], "index": True,
                                                           "similarity": "cosine", "index_options": opts}}})
    t = time.time()
    helpers.bulk(local_es, ({"_index": name, "_id": i, "_source": {"row": i, "v": X[i].tolist()}}
                            for i in range(len(X))), chunk_size=500, request_timeout=300)
    local_es.indices.refresh(index=name)
    local_es.indices.forcemerge(index=name, max_num_segments=1, request_timeout=900)
    build_s = time.time() - t
    size = local_es.indices.stats(index=name, metric="store")["indices"][name]["primaries"]["store"]["size_in_bytes"]
    return dict(build_s=round(build_s, 1), store_MB=round(size / 2 ** 20, 1))


def local_knn(local_es, name: str, X: np.ndarray, qidx: np.ndarray, k: int, num_candidates: int,
              rescore_oversample: float | None = None):
    """kNN against a local index; returns row indices (self excluded) and latency ms."""
    out, lat = np.full((len(qidx), k), -1, np.int64), []
    for r, q in enumerate(qidx):
        knn = dict(field="v", query_vector=X[q].tolist(), k=k + 1, num_candidates=max(num_candidates, k + 1))
        if rescore_oversample:
            knn["rescore_vector"] = {"oversample": rescore_oversample}
        t = time.perf_counter()
        hits = local_es.search(index=name, knn=knn, _source=["row"], size=k + 1)["hits"]["hits"]
        lat.append((time.perf_counter() - t) * 1000)
        rows = [h["_source"]["row"] for h in hits if h["_source"]["row"] != q][:k]
        out[r, :len(rows)] = rows
    return out, np.array(lat)


# ----------------------------------------------------------------------------------------
# fuzz.ratio ground truth and candidate recall
# ----------------------------------------------------------------------------------------


def fuzz_truth(texts: list[str], qidx: np.ndarray, workers: int = -1) -> np.ndarray:
    """Exact fuzz.ratio of each query row against every row, exactly as the monolith computes it
    (cdist, scorer=fuzz.ratio, no processor, uint8). Self is set to 0. Shape [len(qidx), n]."""
    S = cdist([texts[q] for q in qidx], texts, scorer=fuzz.ratio, dtype=np.uint8, workers=workers)
    S[np.arange(len(qidx)), qidx] = 0
    return S


_WORD = re.compile(r"[A-Za-z0-9]+")


def pg_trigrams(text: str) -> set[str]:
    """pg_trgm's show_trgm(): lower-case, split into alphanumeric words, pad each word with two spaces in
    front and one behind, take every 3-character window, as a set. (Exact for ASCII, which is what
    provision_clean_description holds after the monolith's to_ascii step.)"""
    out = set()
    for w in _WORD.findall(text.lower()):
        p = "  " + w + " "
        out.update(p[i:i + 3] for i in range(len(p) - 2))
    return out


def trigram_matrix(texts: list[str]):
    """Binary sparse doc x trigram matrix plus trigram counts per doc."""
    vocab, rows, cols = {}, [], []
    for r, t in enumerate(texts):
        for g in pg_trigrams(t):
            cols.append(vocab.setdefault(g, len(vocab))); rows.append(r)
    M = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
    return M, np.asarray(M.sum(1)).ravel()


def trigram_similarity(M, sizes, qidx: np.ndarray) -> np.ndarray:
    """pg_trgm similarity() = |A & B| / |A | B| of each query row against all rows. Self set to -1."""
    inter = (M[qidx] @ M.T).toarray()
    union = sizes[qidx][:, None] + sizes[None, :] - inter
    S = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    S[np.arange(len(qidx)), qidx] = -1
    return S


def candidate_recall(truth: np.ndarray, floor: int, cand_rows: np.ndarray) -> dict:
    """truth [q, n] fuzz scores; cand_rows [q, c] candidate row indices (-1 = empty slot).
    Pair recall = share of all (query, j) pairs with fuzz >= floor that appear among the candidates."""
    hit = tot = full = with_any = 0
    for r in range(len(truth)):
        t = set(np.flatnonzero(truth[r] >= floor).tolist())
        if not t:
            continue
        c = set(cand_rows[r][cand_rows[r] >= 0].tolist())
        found = len(t & c); hit += found; tot += len(t); with_any += 1; full += found == len(t)
    return dict(floor=floor, true_pairs=tot, found=hit, pair_recall=round(hit / tot, 4) if tot else float("nan"),
                queries_with_pairs=with_any, queries_fully_recalled=full,
                query_full_recall_pct=round(100 * full / with_any, 1) if with_any else float("nan"))


def candidates_from_scores(S: np.ndarray, k: int) -> np.ndarray:
    """Top-k columns by score (descending), self assumed already masked."""
    k = min(k, S.shape[1] - 1)
    part = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.take_along_axis(-S, part, 1).argsort(1)
    return np.take_along_axis(part, order, 1)


def candidates_above(S: np.ndarray, cutoff: float, cap: int | None = None) -> np.ndarray:
    """All columns with score >= cutoff (the pg_trgm `%` operator); padded with -1."""
    rows = [np.flatnonzero(s >= cutoff) for s in S]
    width = max(1, max(len(r) for r in rows))
    out = np.full((len(S), width), -1, np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def fuzz_score_histogram(truth: np.ndarray, floors=(30, 40, 50, 60, 70, 80, 90)) -> list[dict]:
    """Average neighbours per provision above each floor (K in the options page), from the sample."""
    n = truth.shape[1] - 1
    return [dict(floor=f, avg_neighbours_K=round(float((truth >= f).sum(1).mean()), 1),
                 density_d_pct=round(100 * float((truth >= f).sum(1).mean()) / max(n, 1), 3)) for f in floors]
