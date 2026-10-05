"""Rebuild the Colab notebooks from the .py sources (stdlib only). Run in the folder holding the files."""
import json, re
core = open("turboquant_core.py", encoding="utf-8").read()
bench = open("provision_bench.py", encoding="utf-8").read()
for name in ["llm_kv_cache_demo", "vector_search_demo", "provision_search_benchmark"]:
    src = open(f"{name}.py", encoding="utf-8").read()
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append({"id": f"c{len(cells)}", "cell_type": "markdown", "metadata": {}, "source": text})
        else:
            if "Needs turboquant_core.py in the same folder" in body:
                body = "%%writefile turboquant_core.py\n" + core
            elif "Needs provision_bench.py in the same folder" in body:
                body = "%%writefile provision_bench.py\n" + bench
            elif body.startswith("# !pip"):
                body = body.replace("# !pip", "!pip", 1)
            cells.append({"id": f"c{len(cells)}", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
    if name.startswith("llm"):
        meta.update({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}})
    json.dump({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, open(f"{name}.ipynb", "w", encoding="utf-8"), indent=1)
    print(name, len(cells), "cells")
