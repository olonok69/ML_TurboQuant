# es_bench: the provision benchmark against Elasticsearch

The Qdrant benchmark (`../provision_search_benchmark.*`) assumes a Qdrant `provisions` collection and
a cosine auto-merge threshold. Neither exists in the platform. Provision vectors live in the
Provision Database index in **Elasticsearch** (`provision_database_elastic_search`, field
`provision_clean_embedding`, OpenAI `text-embedding-3-small`, 1536-d), and the similarity the product
shows and stores is **`rapidfuzz.fuzz.ratio`** on `provision_clean_description`. This folder benchmarks
that reality and asks the question that matters for the PD similarity matrix: **can cosine
neighbours (float or compressed, TurboQuant included) or trigram candidates find the pairs the
product scores with fuzz.ratio, without scoring every pair?**

Runs locally (CPU). Reuses `../provision_bench.py` and `../turboquant_core.py` unchanged.

| File | What it is |
|---|---|
| `docker-compose.yml` | Local Elasticsearch 8.18.0 single node, security off, 2 GB heap, port 9201 (same as dev) |
| `es_provision_bench.py` | Library: read-only index views, sample loading (PIT), live kNN, local index types, fuzz.ratio truth, pg_trgm-exact trigram similarity, candidate recall |
| `canary.py` | Known-answer checks for every instrument (12 checks, incl. trigram vs a real Postgres `pg_trgm`). Run first |
| `make_local_pd.py` | Builds a local stand-in of the PD index from extracted side-letter JSONs (monolith cleaning, dedup, model) |
| `run_es_bench.py` | The benchmark: sections 1-8, writes CSVs + `REPORT.md` |

## Run

```bash
python -m venv ../.venv   # or: uv venv ../.venv; packages: torch (cpu) numpy scipy rapidfuzz
                          # elasticsearch==8.19.0 beautifulsoup4 openai pandas python-dotenv tiktoken tabulate
docker compose up -d
docker run --rm -d --name pg-trgm -e POSTGRES_PASSWORD=x postgres:16-alpine   # only for canary C2
../.venv/bin/python canary.py                                                  # must print ALL PASS
```

Against **dev** (needs a Kubernetes access entry on `nonprod-dev`; port-forward the service
`dev-<tenant>-elasticsearch:9200`):

```bash
kubectl -n <ns> port-forward svc/<dev-tenant>-elasticsearch 9200:9200
../.venv/bin/python run_es_bench.py --es http://127.0.0.1:9200 --local-es http://127.0.0.1:9201 --out <dir outside git>
```

Against the **local stand-in** (while dev is unreachable):

```bash
OPENAI_API_KEY=... ../.venv/bin/python make_local_pd.py --json-root <folder of extracted JSONs> --cache <dir>/emb.npz
../.venv/bin/python run_es_bench.py --es http://127.0.0.1:9201 --local-es http://127.0.0.1:9201 --out <dir>
```

## Rules

* The source cluster is **read-only**: `_mapping`, `_settings`, `_stats`, `_count`, `_search`, PIT. Index
  creation/deletion is guarded to `localhost` only (canary C7).
* Section 7 sends at most 400 sequential kNN queries to the source. Dev Elasticsearch is a single
  2 GB-heap node, so run it outside working hours or lower `--queries`.
* Provision text and vectors are client data: results and caches go to a folder outside git
  (`.gitignore` covers `es_bench/results/` and `*.npz`).
