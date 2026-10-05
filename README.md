# TurboQuant demos

| File | What it is |
|---|---|
| `llm_kv_cache_demo.ipynb` / `.py` | KV-cache compression demo for Google Colab (T4 GPU). Qwen2.5-1.5B-Instruct by default. |
| `vector_search_demo.ipynb` / `.py` | Vector search benchmark (CPU is fine). DBpedia OpenAI-1536, 100k vectors by default. |
| `turboquant_core.py` | Pure-PyTorch reference implementation used by both (the notebooks embed it; the `.py` scripts import it from this folder). |
| `docs/guide/` | Workshop technical guide in English (`TurboQuant_Technical_Guide_EN.md`) and Spanish (`TurboQuant_Guia_Tecnica_ES.md`), with SVG figures. `make_figures.py` regenerates the figures. |

Colab: File > Upload notebook, pick the runtime (T4 for the LLM demo), Runtime > Run all. If the LLM notebook upgrades `transformers` to v5, restart the session once and run again.

## What was verified before delivery (sandbox, CPU, Hugging Face Hub not reachable)

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
