"""Benchmark the existing implementations on additional distinct judgment PDFs."""
import ast
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from math import ceil
from pathlib import Path

import pandas as pd
import pymupdf as fitz
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
STAGES = ROOT / "tools" / "asbd_stages"
SEED, EXTRA_PDFS = 43, 6000


def load_definitions(stage, names, kind=ast.FunctionDef):
    # The notebook stages execute workloads at import time. Load only their shared definitions.
    path = STAGES / stage
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, kind) and
                (node.name if kind is ast.FunctionDef else getattr(node.targets[0], "id", None)) in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), globals())


def main():
    base = pd.read_csv(OUT / "transactions.csv", keep_default_na=False)
    original = pd.read_csv(OUT / "pdf_sample_manifest.csv")
    meta = pd.read_csv(OUT / "metadata_clean.csv", usecols=["pdf_filename", "is_pdf_available"])
    pool = meta.loc[meta.is_pdf_available & ~meta.pdf_filename.isin(original.pdf_filename),
                   "pdf_filename"].drop_duplicates()
    sample = pool.sample(n=EXTRA_PDFS, random_state=SEED).tolist()
    journal = OUT / "spark_extended_pdf_sample.csv"
    if journal.exists():
        saved = pd.read_csv(journal, keep_default_na=False)
        assert saved.pdf_filename.tolist() == sample[:len(saved)], "Checkpoint does not match the sample"
        start = len(saved)
    else:
        start = 0
    with journal.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if not start and stream.tell() == 0:
            writer.writerow(["pdf_filename", "status", "truncated", "concepts"])
        for index, name in enumerate(sample[start:], start + 1):
            result = extract_judgment(ROOT / "data" / "pdfs" / name)
            concepts = sorted(key for key, pattern in patterns.items() if pattern.search(result["text"]))
            writer.writerow([name, result["status"], result["truncated"], " | ".join(concepts)])
            stream.flush()
            if index % 500 == 0:
                print(f"Saved extraction checkpoint {index:,}/{len(sample):,}", flush=True)
    extracted = pd.read_csv(journal, keep_default_na=False)
    assert extracted.pdf_filename.tolist() == sample and not extracted.pdf_filename.isin(original.pdf_filename).any()
    extra = extracted.loc[extracted.status.eq("ok"), ["pdf_filename", "concepts"]]
    combined = pd.concat([base[["pdf_filename", "concepts"]], extra], ignore_index=True)
    assert combined.pdf_filename.is_unique and len(extra) > 0
    combined.to_csv(OUT / "spark_extended_transactions.csv", index=False)
    baskets = [tuple(value.split(" | ")) if value else () for value in combined.concepts]
    benchmark(baskets, len(base), extracted)


def benchmark(baskets, baseline_size, extracted):
    path = OUT / "spark_extended_benchmark_runs.csv"
    rows = []
    namespace = dict(globals())
    # Reuse notebook engine initialization and warm-up without its original benchmark loop.
    spark_path = STAGES / "07_spark.py"
    startup = spark_path.read_text(encoding="utf-8").split("write_json(OUT", 1)[0]
    started = time.perf_counter()
    exec(compile(startup, str(spark_path), "exec"), namespace)
    spark = namespace["spark"]
    measure = namespace["measure_spark"]
    setup_seconds = time.perf_counter() - started
    try:
        warm = [("x",), ("x", "y"), ("y",), ()]
        started = time.perf_counter()
        assert measure(spark, warm, "FP-Growth")[0] == counts_from_fp(local_fp_growth(warm), len(warm))
        assert measure(spark, warm, "A-Close")[0] == local_a_close(warm)
        warmup_seconds = time.perf_counter() - started
        for repeat in range(1, 4):
            started = time.perf_counter()
            fp_items = local_fp_growth(baskets)
            fp_counts = counts_from_fp(fp_items, len(baskets))
            local_fp_seconds = time.perf_counter() - started
            fp_spark, spark_seconds, prep = measure(spark, baskets, "FP-Growth")
            assert fp_spark == fp_counts
            rows.append(("FP-Growth", repeat, local_fp_seconds, spark_seconds, prep, len(fp_counts), None))
            save_runs(path, rows, len(baskets))
            started = time.perf_counter()
            local_closed = local_a_close(baskets)
            local_close_seconds = time.perf_counter() - started
            spark_closed, spark_seconds, prep = measure(spark, baskets, "A-Close")
            assert spark_closed == local_closed
            assert {item: count for item, count in local_closed[0].items() if item} == fp_counts
            rows.append(("A-Close", repeat, local_close_seconds, spark_seconds, prep, len(fp_counts), len(local_closed[2])))
            runs = save_runs(path, rows, len(baskets))
            print(f"Saved both verified comparisons, repeat {repeat}/3", flush=True)
    finally:
        spark.stop()
    # Verify every reported support and closure directly against the combined baskets.
    sets = [frozenset(basket) for basket in baskets]
    for item, count in fp_counts.items():
        matches = [basket for basket in sets if item <= basket]
        assert len(matches) == count and bool(item in local_closed[2]) == (frozenset.intersection(*matches) == item)
    rules = association_rules(fp_items, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules.loc[rules.lift.ge(MIN_LIFT)]
    closed_rules = sum((row.antecedents | row.consequents) in local_closed[2] for row in rules.itertuples())
    summary = runs.groupby("algorithm", sort=False).agg(
        local_seconds=("local_seconds", "median"), spark_seconds=("spark_seconds", "median"),
        spark_prepare_seconds=("spark_prepare_seconds", "median"),
        frequent=("frequent", "first"), closed=("closed", "first")).reset_index()
    summary["speedup"] = summary.local_seconds / summary.spark_seconds
    summary.insert(1, "documents", len(baskets))
    summary.to_csv(OUT / "spark_extended_benchmark_summary.csv", index=False)
    (OUT / "spark_extended_summary.json").write_text(json.dumps({
        "seed": SEED, "sampled": len(extracted), "baseline_documents": baseline_size,
        "added_readable": len(baskets) - baseline_size, "documents": len(baskets),
        "statuses": extracted.status.value_counts().to_dict(),
        "empty_baskets": sum(not basket for basket in baskets), "repetitions": 3,
        "support": MIN_SUPPORT, "confidence": MIN_CONFIDENCE, "lift": MIN_LIFT,
        "fp_rules": len(rules), "closed_union_rules": closed_rules,
        "spark_version": namespace["spark_version"], "master": "local[2]",
        "setup_seconds": setup_seconds, "warmup_seconds": warmup_seconds,
        "verified_matches": len(runs)
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def save_runs(path, rows, documents):
    runs = pd.DataFrame(rows, columns=["algorithm", "repeat", "local_seconds", "spark_seconds",
                                       "spark_prepare_seconds", "frequent", "closed"])
    runs.insert(1, "documents", documents)
    runs["equal"] = True
    runs.to_csv(path, index=False)
    return runs


if __name__ == "__main__":
    parameters = json.loads((OUT / "fp_summary.json").read_text(encoding="utf-8"))
    MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = (parameters[key] for key in
                                           ("min_support", "min_confidence", "min_lift"))
    load_definitions("01_metadata.py", {"MAX_TEXT_CHARS_PER_DOCUMENT"}, ast.Assign)
    load_definitions("01_metadata.py", {"normalize_document"})
    load_definitions("02_extract.py", {"vocabulary", "patterns"}, ast.Assign)
    assert vocabulary == json.loads((OUT / "legal_vocabulary.json").read_text(encoding="utf-8"))
    load_definitions("02_extract.py", {"extract_judgment"})
    load_definitions("04_fp.py", {"local_fp_growth", "counts_from_fp"})
    load_definitions("05_aclose.py", {"mine_generators", "supports_from_closures", "local_a_close"})
    main()
