# memo_bench: can vector compression help comment-memo response suggestions?

The memo half of the case study (`docs/guide/Case_Study_Provision_Similarity_EN.md`, verdict B). Rebuilds the
suggestion retrieval locally and measures it: corpus from extracted memo JSONs (`{"chains": [{"query_response": [...]}]}`),
`text-embedding-3-large` (3,072-d), Qdrant exact search with firm filter, own thread excluded, limit 50, floors 0.5 / 0.42.

```bash
docker run -d --name qdrant-bench -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.17.1
OPENAI_API_KEY=... ../.venv/bin/python memo_bench.py --json-root <folder of memo JSONs> --out <dir outside git> \
    [--scale 10000,50000,200000] [--llm-model <chat model>]
```

Section 0 is a known-answer check (Qdrant exact search must equal numpy exact search) and the script exits non-zero
if it fails. It only writes to a local Qdrant. Memo text and embeddings are client data: keep `--out` outside git.
Text normalisation is visible text, lower-cased; the platform's query-side legal expansion is not reproduced.
