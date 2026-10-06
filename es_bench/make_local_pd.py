"""
make_local_pd.py - build a LOCAL stand-in for the Provision Database Elasticsearch index.

Used while the dev cluster is not reachable. Reads provision descriptions from extracted side-letter
JSON files, applies the monolith's cleaning (provision_clean_description) and de-duplication (one PD
row per distinct clean description), embeds with the monolith's model (OpenAI text-embedding-3-small,
1536-d), and indexes into a local Elasticsearch under the same index name and field names, so
run_es_bench.py runs unchanged against this or against dev.

    python make_local_pd.py --json-root <folder> [--es http://127.0.0.1:9201] [--limit N]

The OpenAI key is read from OPENAI_API_KEY (environment or --env-file). Embeddings are cached in
--cache (an .npz next to the data, never in git) so re-runs cost nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from es_provision_bench import PD_INDEX, TEXT_FIELD, VECTOR_FIELD, monolith_clean_description

EMBED_MODEL = "text-embedding-3-small"
MAX_EMBED_TOKENS = 8191

# The subset of the monolith's ProvisionDatabaseDocument mapping that the benchmark reads.
PD_MAPPING = {"properties": {
    "id": {"type": "integer"}, "firm_id": {"type": "integer"}, "is_merged": {"type": "boolean"},
    "provision_description": {"type": "text"}, TEXT_FIELD: {"type": "text"},
    VECTOR_FIELD: {"type": "dense_vector", "dims": 1536, "index": True, "similarity": "cosine"},
}}


def create_pd_index(local_es, name: str = PD_INDEX, dims: int = 1536, index_options: dict | None = None):
    """Monolith mapping. index_options=None keeps the monolith's choice (none set -> the ES default)."""
    host = str(local_es.transport.node_pool.get().base_url)
    assert "localhost" in host or "127.0.0.1" in host, f"refusing to write to a non-local cluster: {host}"
    mapping = json.loads(json.dumps(PD_MAPPING))
    mapping["properties"][VECTOR_FIELD]["dims"] = dims
    if index_options:
        mapping["properties"][VECTOR_FIELD]["index_options"] = index_options
    local_es.options(ignore_status=404).indices.delete(index=name)
    local_es.indices.create(index=name, settings={"number_of_shards": 1, "number_of_replicas": 0}, mappings=mapping)


def collect_descriptions(root: Path) -> list[str]:
    """Every provision description in every extracted-JSON file under root (same shapes as the
    get-sl-upload-status payload: {"provisions": [...]} or {"data": {"provisions": [...]}})."""
    out = []
    skip = {"node_modules", "knowledge-graph", "_prepull", "machine-sync"}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            p = Path(dp) / f
            if p.suffix != ".json" or p.stat().st_size > 20e6:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            provs = d.get("provisions") if isinstance(d, dict) else None
            if provs is None and isinstance(d, dict) and isinstance(d.get("data"), dict):
                provs = d["data"].get("provisions")
            if not isinstance(provs, list):
                continue
            for pr in provs:
                if isinstance(pr, dict) and isinstance(pr.get("description"), str) and pr["description"].strip():
                    out.append(pr["description"])
    return out


def embed(texts: list[str], cache: Path, batch: int = 100) -> np.ndarray:
    import tiktoken
    from openai import OpenAI

    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in have]
    if todo:
        enc, client = tiktoken.get_encoding("cl100k_base"), OpenAI()
        for a in range(0, len(todo), batch):
            idx = todo[a:a + batch]
            # The monolith sends the full text; the API rejects > 8191 tokens. Truncate here and report it.
            inp = [enc.decode(enc.encode(texts[i])[:MAX_EMBED_TOKENS]) for i in idx]
            resp = client.embeddings.create(model=EMBED_MODEL, input=inp)
            for i, d in zip(idx, resp.data):
                have[keys[i]] = np.asarray(d.embedding, np.float32)
            print(f"  embedded {min(a + batch, len(todo))}/{len(todo)}")
        np.savez(cache, keys=np.array(list(have.keys())), vecs=np.stack(list(have.values())))
    return np.stack([have[k] for k in keys])


def main():
    from dotenv import load_dotenv
    from elasticsearch import Elasticsearch, helpers

    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", required=True)
    ap.add_argument("--es", default="http://127.0.0.1:9201")
    ap.add_argument("--cache", default="local_pd_embeddings.npz")
    ap.add_argument("--env-file")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--firm-id", type=int, default=1)
    a = ap.parse_args()
    if a.env_file:
        load_dotenv(a.env_file)

    raw = collect_descriptions(Path(a.json_root))
    seen, rows = set(), []
    for html in raw:
        clean = monolith_clean_description(html)
        if clean and clean not in seen:          # PD keeps one row per distinct clean description
            seen.add(clean); rows.append((html, clean))
    if a.limit:
        rows = rows[:a.limit]
    print(f"descriptions read: {len(raw)}, distinct clean descriptions (PD rows): {len(rows)}")

    X = embed([c for _, c in rows], Path(a.cache))
    es = Elasticsearch(a.es, request_timeout=120)
    create_pd_index(es)
    helpers.bulk(es, ({"_index": PD_INDEX, "_id": i + 1,
                       "_source": {"id": i + 1, "firm_id": a.firm_id, "is_merged": False,
                                   "provision_description": h, TEXT_FIELD: c, VECTOR_FIELD: X[i].tolist()}}
                      for i, (h, c) in enumerate(rows)), chunk_size=200)
    es.indices.refresh(index=PD_INDEX)
    print(f"indexed {es.count(index=PD_INDEX)['count']} docs into local {PD_INDEX}")


if __name__ == "__main__":
    main()
