"""Assemble the notebook and save executed cells atomically as each stage completes."""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NOTEBOOK = ROOT / "main.ipynb"
SOURCES = ROOT / "tools" / "asbd_stages"
STARTER = OUT / "starter_notebook.ipynb"
OUT.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fresh", action="store_true", help="Re-extract every sampled PDF from the source files.")
arguments = parser.parse_args()
os.environ["ASBD_FRESH_EXTRACTION"] = "1" if arguments.fresh else "0"

if not STARTER.exists():
    STARTER.write_bytes(NOTEBOOK.read_bytes())

starter = nbformat.read(STARTER, as_version=4)
notebook = nbformat.v4.new_notebook(metadata=starter.metadata)
current = nbformat.read(NOTEBOOK, as_version=4)
notebook.cells.append(nbformat.v4.new_markdown_cell(current.cells[0].source))
for stage in sorted(SOURCES.glob("*.py")):
    title, description, code = stage.read_text(encoding="utf-8").split("\n", 2)
    notebook.cells.extend([
        nbformat.v4.new_markdown_cell("## " + title[2:] + "\n\n" + description[2:]),
        nbformat.v4.new_code_cell(code.strip())
    ])

def atomic(path, contents):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(contents, encoding="utf-8")
    os.replace(tmp, path)

started = datetime.now(timezone.utc).isoformat()
completed = []

def save(state, label):
    atomic(NOTEBOOK, nbformat.writes(notebook))
    atomic(OUT / "execution_progress.json", json.dumps({
        "state": state, "stage": label, "utc": datetime.now(timezone.utc).isoformat(),
        "stages_in_run": (len(notebook.cells) - 1) // 2, "started_utc": started,
        "fresh_extraction": arguments.fresh, "completed": completed
    }, indent=2))
    print(state, label, flush=True)

# A checkpoint contains only outputs produced in this run.
save("started", "Initializing kernel")

client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
with client.setup_kernel():
    for index in range(2, len(notebook.cells), 2):
        label = notebook.cells[index - 1].source.splitlines()[0][3:]
        print("running", label, flush=True)
        try:
            client.execute_cell(notebook.cells[index], index, execution_count=index // 2)
        except BaseException:
            save("failed", label)
            raise
        completed.append({"stage": label, "finished_utc": datetime.now(timezone.utc).isoformat()})
        save("saved", label)

save("run_complete", label)

