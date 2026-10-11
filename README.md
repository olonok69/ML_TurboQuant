# TurboQuant demos

| File | What it is |
|---|---|
| `llm_kv_cache_demo.ipynb` / `.py` | KV-cache compression demo for Google Colab (A100 or L4 GPU). Llama-3.1-8B-Instruct, the paper's model (gated: needs an `HF_TOKEN` Colab secret). |
| `vector_search_demo.ipynb` / `.py` | Vector search benchmark (CPU is fine). DBpedia OpenAI-1536, 100k vectors by default. |
| `turboquant_core.py` | Pure-PyTorch reference implementation used by both (the notebooks embed it; the `.py` scripts import it from this folder). |
| `provision_search_benchmark.ipynb` / `.py` + `provision_bench.py` | Separate from the workshop demos: benchmarks our Qdrant provision embeddings (live collection checks, embedding health, compression options incl. TurboQuant, auto-merge stability, labelled evaluation). Colab; read-only against production. |
| `docs/guide/` | Workshop technical guide in English (`TurboQuant_Technical_Guide_EN.md`) and Spanish (`TurboQuant_Guia_Tecnica_ES.md`), with SVG figures. `make_figures.py` regenerates the figures. |
| `docs/guide/Case_Study_Provision_Similarity_EN.md` / `Caso_Estudio_Similitud_Provisiones_ES.md` | Workshop case study: a real check where vector compression does **not** help (edit-distance similarity, Elasticsearch embeddings), with the checklist to run before compressing. |
| `es_bench/` | The benchmark behind the case study: Elasticsearch instead of Qdrant, fuzz.ratio recall, known-answer canary. See `es_bench/README.md`. |

Colab: File > Upload notebook, pick the runtime (A100 or L4 for the LLM demo; the 8B model does not fit a free T4), Runtime > Run all. If the LLM notebook upgrades `transformers` to v5, restart the session once and run again.

The two demo notebooks are edited and executed in Colab, so they are the source: `python build_notebooks.py` exports them to the `.py` scripts and leaves the saved outputs alone (`--rebuild NAME` goes the other way and drops the outputs).

## Colab results (11 October 2026)

**LLM demo, Llama-3.1-8B-Instruct on an A100, WikiText-2.** Perplexity (FP16 5.71): TurboQuant 4-bit 5.81, 3.5-bit 5.86, 3-bit 6.10, 2.5-bit 7.38, 2-bit 9.36; INT4 with scales 5.78, INT3 6.14, INT2 17.20. KV cache for an 8,084-token prompt: 1,010 MB in FP16, 260 MB with TurboQuant 4-bit, 158 MB at 2.5 bits, 134 MB at 2 bits (INT4: 316 MB, INT2: 189 MB). Needle in a haystack (2k to 8k tokens, 3 depths, 9 prompts): TurboQuant 4 and 3-bit 9 of 9, 2-bit 8, 2.5-bit 7, INT2 3. Decode speed at 4k context: 26.9 tok/s FP16, 10.4 TurboQuant 4-bit, 6.6 TurboQuant 2.5-bit (reference code, no fused kernels).

**Vector demo, DBpedia OpenAI-1536 (100k + 1k), Colab CPU, 1 thread:**

| Method | Bits | Build (s) | QPS | R1@1 | 10@10 |
|---|---|---|---|---|---|
| turbovec | 4 | 3.6 | 189 | 0.971 | 0.967 |
| TurboQuant_mse (PyTorch) | 4 | 25.3 | 88 | 0.966 | 0.961 |
| FAISS RaBitQ | 4 | 39.6 | 40 | 0.965 | 0.965 |
| FAISS PQ LUT256 | 4 | 816.6 | 8 | 0.965 | 0.960 |
| FAISS PQ-FastScan | 4 | 46.4 | 161 | 0.959 | 0.954 |
| FAISS SQ4 | 4 | 0.8 | 13 | 0.941 | 0.948 |
| turbovec | 2 | 3.0 | 357 | 0.892 | 0.895 |
| TurboQuant_mse (PyTorch) | 2 | 22.9 | 113 | 0.879 | 0.883 |
| FAISS RaBitQ | 2 | 21.3 | 87 | 0.874 | 0.894 |
| FAISS PQ LUT256 | 2 | 397.5 | 17 | 0.873 | 0.895 |
| FAISS PQ-FastScan | 2 | 16.1 | 352 | 0.882 | 0.884 |

Exact float32 search: 155 QPS. turbovec 4-bit top-100 re-ranked with exact vectors: R1@1 = 1.000. 100k vectors streamed into turbovec in 1.8 s.

## Sandbox results (before the Colab runs; CPU, Hugging Face Hub not reachable)

* Theory check: measured MSE for unit vectors (d = 128) is 0.360 / 0.116 / 0.034 / 0.0093 at 1-4 bits, matching the paper's 0.36 / 0.117 / 0.03 / 0.009. `TurboQuant_prod` is unbiased (+0.1%); MSE is biased by -36% / -12% / -3.5% at 1 / 2 / 3 bits.
* Both notebooks executed end to end with `nbclient`. The LLM notebook ran with a random-weight Qwen2 model (no model downloads possible), so its quality and needle numbers there are meaningless; they are produced for real on Colab. Measured memory is exact: 4.125 bits/channel at 4-bit, 2.5 at the 2.5-bit outlier split, 5.0 for INT4 with scales.
* Vector search, 100k database + 1k queries, 384-d embeddings of real text (Reuters, Gutenberg, Brown; TF-IDF + SVD), 4 CPU threads:

| Method | Bits | Build (s) | QPS | R1@1 | 10@10 |
|---|---|---|---|---|---|
| turbovec | 4 | 0.9 | 10,008 | 0.944 | 0.951 |
| TurboQuant_mse (PyTorch) | 4 | 0.9 | 950 | 0.912 | 0.934 |
| FAISS RaBitQ | 4 | 1.4 | 862 | 0.871 | 0.891 |
| FAISS PQ LUT256 | 4 | 83.4 | 390 | 0.818 | 0.870 |
| FAISS SQ4 | 4 | 0.06 | 470 | 0.814 | 0.856 |
| FAISS PQ-FastScan | 4 | 4.4 | 3,784 | 0.728 | 0.808 |
| turbovec | 2 | 0.8 | 9,340 | 0.799 | 0.832 |
| TurboQuant_mse (PyTorch) | 2 | 0.7 | 1,039 | 0.715 | 0.802 |
| FAISS RaBitQ | 2 | 0.9 | 1,717 | 0.623 | 0.704 |
| FAISS PQ LUT256 | 2 | 6.6 | 930 | 0.612 | 0.724 |
| FAISS PQ-FastScan | 2 | 1.6 | 6,342 | 0.536 | 0.649 |

Exact float32 search: 857 QPS. turbovec 4-bit top-100 re-ranked with exact vectors: R1@1 = 1.000.

Note: the paper's "2.5-bit" split (32 channels @3b + 96 @2b) is 2.25 bits/channel of codes; with the fp16 norms it measures exactly 2.5.
