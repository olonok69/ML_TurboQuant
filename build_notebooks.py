"""Keep the Colab notebooks and their .py script versions in step (stdlib only). Run in the folder holding the files.

The two workshop demos (llm_kv_cache_demo, vector_search_demo) are edited and executed in Colab, so the
notebook is the source: `python build_notebooks.py` exports them to .py and leaves the executed notebooks alone.
The provision benchmark is still written as a script: its notebook is rebuilt from the .py.
`python build_notebooks.py --rebuild NAME` rebuilds any notebook from its .py (this drops the saved outputs).
"""
import json, re, sys
NOTEBOOK_SOURCED = ["llm_kv_cache_demo", "vector_search_demo"]
SCRIPT_SOURCED = ["provision_search_benchmark"]
EMBEDDED = {"turboquant_core.py": "# Needs turboquant_core.py in the same folder (in the notebook this cell writes it).",
            "provision_bench.py": "# Needs provision_bench.py in the same folder (in the notebook this cell writes it)."}


def build(name):
    core = open("turboquant_core.py", encoding="utf-8").read()
    bench = open("provision_bench.py", encoding="utf-8").read()
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
        meta.update({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}})
    json.dump({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, open(f"{name}.ipynb", "w", encoding="utf-8"), indent=1)
    print("rebuilt", f"{name}.ipynb", len(cells), "cells")


def export(name):
    nb = json.load(open(f"{name}.ipynb", encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        body = "".join(c["source"]).strip("\n")
        if c["cell_type"] == "markdown":
            out.append("# %% [markdown]\n" + "\n".join("# " + l if l else "#" for l in body.splitlines()))
            continue
        first, _, rest = body.partition("\n")
        if first.startswith("%%writefile ") and first.split()[1] in EMBEDDED:
            body = EMBEDDED[first.split()[1]]
        else:   # notebook-only syntax stays as comments so the script still runs
            body = "\n".join("# " + l if l.lstrip().startswith(("!", "%")) else l for l in body.splitlines())
        out.append("# %%\n" + body)
    open(f"{name}.py", "w", encoding="utf-8").write("\n\n".join(out) + "\n")
    print("exported", f"{name}.py", len(nb["cells"]), "cells")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rebuild"]:
        for n in sys.argv[2:]:
            build(n)
    else:
        for n in NOTEBOOK_SOURCED:
            export(n)
        for n in SCRIPT_SOURCED:
            build(n)
