"""Render the report, validate saved results, and package the verified deliverables."""
import argparse
from collections import Counter
from csv import DictReader
import json
from math import isclose
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat
import pandas as pd
import pymupdf as fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NOTEBOOK = ROOT / "main.ipynb"
PDF = ROOT / "ASBD_EndSem_Report.pdf"
RENDER = ROOT / "screenshots" / "report_review"
AUDIT = ROOT / "FINAL_AUDIT.md"
ARCHIVE = ROOT / "ASBD_EndSem_Submission.zip"

def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))

def render_and_check():
    RENDER.mkdir(parents=True, exist_ok=True)
    with fitz.open(PDF) as document:
        assert len(document) == 11
        sheet = Image.new("RGB", (1280, ((len(document)+3)//4)*468), "#e7edf2")
        draw = ImageDraw.Draw(sheet)
        images = 0
        for index, page in enumerate(document):
            words = page.get_text("words")
            assert len(words) > 30, f"Nearly empty report page {index+1}"
            assert all(25 <= w[0] < w[2] <= page.rect.width - 25 for w in words)
            assert all(20 <= w[1] < w[3] <= page.rect.height - 18 for w in words)
            images += len(page.get_image_info())
            png = RENDER / f"page_{index+1:02}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(png)
            with Image.open(png) as image:
                image.thumbnail((299, 424), Image.Resampling.LANCZOS)
                x, y = (index % 4) * 320 + 10, (index // 4) * 468 + 12
                sheet.paste(image, (x, y))
                draw.text((x, y + 429), f"Page {index+1}", fill="#172f46")
        assert images == 13
    sheet.save(RENDER / "contact_sheet.png")

def package():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    stages = sorted((ROOT / "tools" / "asbd_stages").glob("*.py"))
    assert [c.execution_count for c in code] == list(range(1, len(stages) + 1))
    assert [c.source for c in code] == [
        path.read_text(encoding="utf-8").split("\n", 2)[2].strip() for path in stages
    ]
    assert all(all(o.output_type != "error" for o in c.outputs) for c in code)
    checkpoint = read_json("execution_progress.json")
    assert checkpoint["state"] == "run_complete"
    assert len(checkpoint["completed"]) == checkpoint["stages_in_run"] == len(stages)
    descriptive = read_json("descriptive_figures.json")
    all_figures = read_json("all_figures.json")
    assert len(descriptive) == len({item["type"] for item in descriptive}) == 10
    assert len(all_figures) == 13 and all(item["inference"].strip() for item in all_figures)
    for item in all_figures:
        with Image.open(ROOT / "figures" / item["file"]) as plot:
            plot.verify()

    meta = read_json("metadata_summary.json")
    extraction = read_json("pdf_summary.json")
    fp = read_json("fp_summary.json")
    closure = read_json("aclose_summary.json")
    cluster = read_json("kmedoids_summary.json")
    spark = read_json("spark_summary.json")
    assert meta["rows"] == meta["matched_rows"] + meta["unmatched_rows"]
    assert extraction["sampled"] == sum(extraction["statuses"].values())
    assert extraction["sampled"] == extraction["cache_hits"] + extraction["live_pdf_attempts"]
    if checkpoint["fresh_extraction"]:
        assert extraction["cache_hits"] == 0
    assert all(meta["missing"][key] >= count for key, count in meta["zero_placeholders"].items())
    assert fp["transactions"] == extraction["readable"] and len(fp["interpretations"]) >= 3
    assert fp["items"] == closure["frequent"] and closure["closure_verified"]
    assert closure["full_data_counts_match_fp"] and closure["exhaustive_tests"] >= 1
    baskets = [frozenset(value.split(" | ")) if value else frozenset()
               for value in pd.read_csv(OUT / "transactions.csv", keep_default_na=False).concepts]
    sweep = pd.read_csv(OUT / "fp_support_sensitivity.csv").set_index("min_support")
    assert list(sweep.index) == [.015, .02, .03]
    assert sweep.documents.eq(len(baskets)).all()
    assert sweep.empty_baskets.eq(sum(not basket for basket in baskets)).all()
    assert sweep.min_confidence.eq(fp["min_confidence"]).all() and sweep.min_lift.eq(fp["min_lift"]).all()
    assert (sweep.loc[.02, "frequent_itemsets"], sweep.loc[.02, "rules"]) == (fp["items"], fp["rules"])
    itemsets = pd.read_csv(OUT / "fp_itemsets.csv")
    counts = {frozenset(row.itemset.split(" | ")): row.support_count for row in itemsets.itertuples()}
    assert len(counts) == fp["items"] and len(baskets) == fp["transactions"]
    assert all(sum(item <= basket for basket in baskets) == count for item, count in counts.items())
    closed_table = pd.read_csv(OUT / "aclose_closed_itemsets.csv")
    closed = {frozenset(row.itemset.split(" | ")): row.support_count for row in closed_table.itertuples()}
    expected_closed = {item: count for item, count in counts.items()
                       if all(not item < other or counts[other] != count for other in counts)}
    assert closed == expected_closed and len(closed) == closure["closed"]
    for name in ("fp_rules.csv", "aclose_rules.csv"):
        for rule in pd.read_csv(OUT / name).itertuples():
            left = frozenset(rule.antecedent.split(" | "))
            right = frozenset(rule.consequent.split(" | "))
            joint = counts[left | right]
            confidence = joint / counts[left]
            assert isclose(rule.support, joint / len(baskets))
            assert isclose(rule.confidence, confidence)
            assert isclose(rule.lift, confidence * len(baskets) / counts[right])
            assert rule.support >= fp["min_support"] and rule.confidence >= fp["min_confidence"]
            assert rule.lift >= fp["min_lift"]
            if name == "aclose_rules.csv":
                assert left | right in closed
    assignments = pd.read_csv(OUT / "kmedoids_assignments.csv")
    assert len(assignments) == cluster["sample"]
    assert assignments.groupby("cluster").is_medoid.sum().eq(1).all()
    assert assignments.cluster.nunique() == cluster["selected_k"]
    trials = pd.read_csv(OUT / "kmedoids_trials.csv")
    best = trials.loc[trials.silhouette.idxmax()]
    assert best.k == cluster["selected_k"] and isclose(best.silhouette, cluster["silhouette"])
    clusters = pd.read_csv(OUT / "kmedoids_clusters.csv").set_index("cluster")
    medoids = assignments.loc[assignments.is_medoid].set_index("cluster")
    assert clusters.medoid_record.sort_index().equals(medoids.record_id.sort_index())
    assert clusters.medoid_diary_no.sort_index().equals(medoids.diary_no.sort_index())
    tax_cluster = clusters.loc[clusters.characteristic_terms.str.contains("tax, assessee")].iloc[0]
    timings = pd.read_csv(OUT / "spark_benchmark_runs.csv")
    assert spark["all_supports_equal"] and timings.equal.all()
    assert timings[["local_seconds", "spark_seconds"]].gt(0).all().all()
    assert timings.spark_prepare_seconds.between(0, timings.spark_seconds).all()
    assert len(timings) == 2 * spark["repetitions"] * len(spark["workloads"])
    assert set(timings.algorithm) == {"FP-Growth", "A-Close"}
    assert timings.groupby(["algorithm", "documents"]).repeat.nunique().eq(spark["repetitions"]).all()
    assert len(spark["workloads"]) == 2
    summary = pd.read_csv(OUT / "spark_benchmark_summary.csv")
    evidence = read_json("aclose_workload_comparison.json")
    example = evidence["example"]
    smaller = frozenset(example["itemset"].split(" | "))
    larger = frozenset(example["closure"].split(" | "))
    prefix = baskets[:example["documents"]]
    matches = [basket for basket in prefix if smaller <= basket]
    assert smaller < larger == frozenset.intersection(*matches)
    assert example["support_count"] == len(matches)
    assert isclose(example["support"], len(matches) / len(prefix))
    for result in evidence["workloads"]:
        spark_result = summary.loc[summary.algorithm.eq("A-Close") &
                                   summary.documents.eq(result["documents"])].iloc[0]
        assert (result["frequent"], result["closed"]) == (spark_result.frequent, spark_result.closed)
        assert result["nonclosed"] == result["frequent"] - result["closed"]
        assert isclose(result["reduction"], result["nonclosed"] / result["frequent"])
    medians = timings.groupby(["algorithm", "documents"])[["local_seconds", "spark_seconds"]].median()
    for row in summary.itertuples():
        measured = medians.loc[row.algorithm, row.documents]
        assert isclose(row.local_seconds, measured.local_seconds)
        assert isclose(row.spark_seconds, measured.spark_seconds)
    assert all(isclose(row.speedup, row.local_seconds / row.spark_seconds)
               for row in summary.itertuples())
    # Verify the exported distribution independently of the notebook's pandas merge.
    with (OUT / "metadata_clean.csv").open(encoding="utf-8", newline="") as stream:
        records = list(DictReader(stream))
    by_id = {row["record_id"]: row for row in records}
    with (OUT / "pdf_sample_features.csv").open(encoding="utf-8", newline="") as stream:
        readable = [row for row in DictReader(stream) if row["status"] == "ok"]
    mining_keys = pd.read_csv(OUT / "transactions.csv", usecols=["record_id", "pdf_filename"],
                              dtype={"record_id": str})
    readable_keys = {(row["record_id"], row["pdf_filename"]) for row in readable}
    assert len(records) == len(by_id) == meta["rows"]
    assert len(readable) == len(readable_keys) == len(mining_keys) == fp["transactions"]
    assert readable_keys == set(mining_keys.itertuples(index=False, name=None))
    assert all(by_id[row["record_id"]]["pdf_filename"] == row["pdf_filename"] and
               by_id[row["record_id"]]["case_category"] == row["case_category"] for row in readable)
    full_counts = Counter(row["case_category"] for row in records)
    readable_counts = Counter(by_id[row["record_id"]]["case_category"] for row in readable)
    mix = pd.read_csv(OUT / "sample_representativeness.csv")
    assert len(mix) == len(full_counts) and mix.case_category.is_unique
    assert set(mix.case_category) == set(full_counts)
    assert mix.full_count.is_monotonic_decreasing
    for row in mix.itertuples():
        assert row.full_count == full_counts[row.case_category]
        assert row.readable_count == readable_counts[row.case_category]
        full_pct = 100 * row.full_count / len(records)
        readable_pct = 100 * row.readable_count / len(readable)
        assert isclose(row.full_pct, full_pct, rel_tol=0, abs_tol=5.1e-9)
        assert isclose(row.readable_pct, readable_pct, rel_tol=0, abs_tol=5.1e-9)
        assert isclose(row.difference_pp, readable_pct - full_pct, rel_tol=0, abs_tol=5.1e-9)
    largest_gap = mix.loc[mix.difference_pp.abs().idxmax()]

    with fitz.open(PDF) as document:
        limitations_page = " ".join(document[10].get_text().split())
        report = " ".join(" ".join(page.get_text().split()) for page in document)
        pages = len(document)
        benchmark_page = " ".join(document[9].get_text().split())
    assert pages == 11 and "\ufffd" not in report and "×" not in benchmark_page
    assert f"at most {abs(largest_gap.difference_pp):.2f} percentage points" in limitations_page
    for row in mix.head(3).itertuples():
        values = f"{row.case_category} {row.full_pct:.2f}% to {row.readable_pct:.2f}% ({row.difference_pp:+.2f} pp)"
        assert values in limitations_page, values
    for row in summary.itertuples():
        values = f"{row.algorithm} {row.documents:,} {row.local_seconds:.4f} {row.spark_seconds:.4f} {row.speedup:.4f}x"
        assert values in benchmark_page, values
    extension_text = ""
    if (OUT / "spark_extended_summary.json").is_file():
        extension = read_json("spark_extended_summary.json")
        extra_sample = pd.read_csv(OUT / "spark_extended_pdf_sample.csv", keep_default_na=False)
        original_sample = pd.read_csv(OUT / "pdf_sample_manifest.csv")
        original_tx = pd.read_csv(OUT / "transactions.csv", keep_default_na=False)
        extended_tx = pd.read_csv(OUT / "spark_extended_transactions.csv", keep_default_na=False)
        added = extra_sample.loc[extra_sample.status.eq("ok"), ["pdf_filename", "concepts"]]
        expected = pd.concat([original_tx[["pdf_filename", "concepts"]], added], ignore_index=True)
        pd.testing.assert_frame_equal(extended_tx, expected)
        assert len(extra_sample) == extension["sampled"] and extra_sample.pdf_filename.is_unique
        assert not extra_sample.pdf_filename.isin(original_sample.pdf_filename).any()
        assert extended_tx.pdf_filename.is_unique and len(extended_tx) == extension["documents"]
        assert len(added) == extension["added_readable"] == len(extended_tx) - len(original_tx)
        assert (extension["support"], extension["confidence"], extension["lift"]) == (
            fp["min_support"], fp["min_confidence"], fp["min_lift"])
        extended = pd.read_csv(OUT / "spark_extended_benchmark_summary.csv")
        extended_runs = pd.read_csv(OUT / "spark_extended_benchmark_runs.csv")
        assert set(extended.algorithm) == {"FP-Growth", "A-Close"}
        assert len(extended_runs) == extension["verified_matches"] == 2 * extension["repetitions"]
        assert extended_runs.equal.all() and extended_runs.documents.eq(len(extended_tx)).all()
        for row in extended.itertuples():
            runs = extended_runs.loc[extended_runs.algorithm.eq(row.algorithm)]
            assert runs.repeat.nunique() == extension["repetitions"]
            assert runs[["local_seconds", "spark_seconds"]].gt(0).all().all()
            assert runs.spark_prepare_seconds.between(0, runs.spark_seconds).all()
            assert isclose(row.local_seconds, runs.local_seconds.median())
            assert isclose(row.spark_seconds, runs.spark_seconds.median())
            assert isclose(row.speedup, row.local_seconds / row.spark_seconds)
            assert f"{row.algorithm} {row.local_seconds:.4f} {row.spark_seconds:.4f} {row.speedup:.4f}x" in benchmark_page
        extension_text = (
            f"- The larger workload added {len(added):,} readable PDFs from {len(extra_sample):,} newly sampled filenames. "
            f"All {len(extended_runs)} paired runs matched across {len(extended_tx):,} baskets. "
            f"Spark was faster for {int(extended.speedup.gt(1).sum())} of the {len(extended)} algorithms. "
            "The original benchmark timings were preserved.\n"
        )
        extension_text += "".join(
            f"- Larger workload, {row.algorithm}. Local {row.local_seconds:.4f} s, "
            f"Spark {row.spark_seconds:.4f} s, speedup {row.speedup:.4f}x.\n"
            for row in extended.itertuples()
        )
    for result in evidence["workloads"]:
        values = " ".join(str(result[key]) for key in ("frequent", "closed", "nonclosed"))
        assert f"{result['documents']:,} {values} {result['reduction']:.2%}" in report
    assert f"{example['support_count']} documents ({example['support']:.1%} support)" in report
    assert f"{tax_cluster.medoid_diary_no} (CSV row {tax_cluster.medoid_record:,})" in report
    assert f"silhouette of {best.silhouette:.3f} indicates weak separation" in report
    for chart in descriptive:
        inference = chart["inference"].replace("→", " to ").replace("—", "-").replace("–", "-").replace("×", "x")
        assert " ".join(inference.split()) in report, chart["file"]
    for support, row in sweep.iterrows():
        left = frozenset(row.antecedent.split(" | "))
        right = frozenset(row.consequent.split(" | "))
        joint = counts[left | right]
        assert isclose(row.rule_support, joint / len(baskets))
        assert isclose(row.rule_confidence, joint / counts[left])
        assert isclose(row.rule_lift, row.rule_confidence * len(baskets) / counts[right])
        assert f"{support:.1%} gives {row.frequent_itemsets} itemsets and {row.rules} rules" in report
    for value in (f'{meta["rows"]:,}', f'{extraction["readable"]:,}', f'{fp["items"]:,}',
                  f'{closure["closed"]:,}', f'{cluster["silhouette"]:.3f}',
                  "FP-Growth", "A-Close", "K-Medoids", "Local and PySpark benchmarks",
                  "25CS5004 / CS 5010", "November 2025", "Kancheepuram"):
        assert value in report, value

    benchmark_lines = "\n".join(
        f"- {row.algorithm}, {row.documents:,} baskets. Local {row.local_seconds:.4f} s, "
        f"Spark {row.spark_seconds:.4f} s, speedup {row.speedup:.4f}x."
        for row in summary.itertuples())
    AUDIT.write_text(f"""# November 2025 end-semester audit

## Findings and corrections

- The missing-value policy previously counted placeholder identifiers as valid. The audit masks {meta['zero_placeholders']['case_no']:,} case numbers and {meta['zero_placeholders']['diary_no']:,} diary IDs. The original diary field remains available for exact PDF linkage.
- A bare Section 302 citation is ambiguous. The murder pattern now requires a literal mention.
- A-Close now prunes candidates with equal support to any immediate subset before the next join and intersects supporting baskets to compute closures. It passed {closure['exhaustive_tests']} exhaustive-reference tests. The saved FP supports and closed itemsets were independently checked, including confidence and lift for all exported rules.
- The Spark comparison now includes input preparation, mining and result collection, starting from the same Python baskets. Session startup and warm-up were measured separately.
- The report and notebook identify the supplied November 2025 paper. All {len(code)} stages completed and match the bundled Python source. The {pages}-page report was rendered and inspected.

## Verified results

- Metadata checks account for {meta['rows']:,} rows, {meta['unmatched_rows']} missing exact matches and {meta['pre_1950']} disputed chronology dates. The extraction stage attempted {extraction['live_pdf_attempts']:,} live PDF reads and used {extraction['cache_hits']:,} cached records. It produced {extraction['readable']:,} readable documents.
- Ten distinct descriptive chart types and three algorithm figures have data-specific inferences.
- FP-Growth produced {fp['items']:,} frequent itemsets and {fp['rules']:,} rules. A-Close produced {closure['closed']:,} closed itemsets and {closure['closed_union_rules']:,} rules with closed unions.
- K-Medoids selected k={cluster['selected_k']} with silhouette {cluster['silhouette']:.3f}. The notebook verified medoid membership and objective cost, and the exported assignments contain one medoid per cluster.
- All {len(timings)} paired Spark runs agreed with local results. Spark was faster in {int(summary.speedup.gt(1).sum())} of {len(summary)} median comparisons.
{benchmark_lines}

## Final submission verification

- The README evidence guide links all eleven requirements and checks to report pages, notebook sections and bundled files. The supplemental FP-Growth sweep holds the 3,843 transactions, including 501 empty baskets, and the confidence and lift filters fixed. Its 2% run reproduced the original itemsets and rules, including their metrics. The sweep is recorded in `outputs/fp_support_sensitivity.csv` and summarized on page 7.
- The final notebook cell was re-executed from saved inputs. An independent record-ID lookup verified the exact {len(readable):,} original mining PDFs against {len(records):,} metadata records, including every category count and percentage. The largest difference was {largest_gap.case_category} ({largest_gap.difference_pp:+.2f} percentage points). Page 11 and `outputs/sample_representativeness.csv` document the comparison. Prior notebook cells, mining outputs, and benchmark measurements were preserved.
- Page 10 now uses ASCII `x` for speedup ratios in the table and caption. The text layer contains all four benchmark rows at the precision shown and has no replacement glyphs. Visual inspection covered all 11 pages and all 13 figures.
- A-Close and report cells were executed from saved inputs. Six other executed cells were preserved, including K-Medoids and both Spark tests. All ten descriptive inferences appear in the report, and the benchmark table matches the two raw repetitions per workload and algorithm.
- The first {evidence['workloads'][0]['documents']:,} transactions produced {evidence['workloads'][0]['frequent']} frequent itemsets and {evidence['workloads'][0]['closed']} closed ones, a {evidence['workloads'][0]['reduction']:.2%} reduction. Direct basket intersection verified that {{{example['itemset'].replace(' | ', ', ')}}} and its strict closure {{{example['closure'].replace(' | ', ', ')}}} share {example['support_count']} supporting documents ({example['support']:.1%}). The full workload retains all {closure['closed']} frequent itemsets.
- The selected k={cluster['selected_k']} scored highest among the four tested values. The final report identifies weak separation and uses medoid {tax_cluster.medoid_diary_no} and the characteristic terms of cluster {tax_cluster.name} to illustrate its tax-assessment vocabulary.
- The original Spark review checked all {len(baskets):,} available transaction documents and per-run preparation measurements. On that sample, preparation alone took a median {timings.loc[(timings.documents.eq(len(baskets))) & timings.algorithm.eq("FP-Growth"), "spark_prepare_seconds"].median():.4f} s for FP-Growth and {timings.loc[(timings.documents.eq(len(baskets))) & timings.algorithm.eq("A-Close"), "spark_prepare_seconds"].median():.4f} s for A-Close. The report explains data conversion, partitioning, JVM task scheduling and driver collection.
{extension_text}- The ZIP was checked for integrity and contains byte-identical copies of the latest notebook, PDF, source and supporting results.

## Limits and decisions

- Seed 42 selects a uniform sample of unique, exactly matched PDF names. English phrase patterns and extraction without OCR can miss mentions. The character cap can truncate documents and separate filenames can refer to versions of one case.
- The three malformed dates are retained with evidence and omitted from chronology charts. Candidate filenames for unmatched rows were logged without substitution.
- Cluster labels are exploratory. The benchmark measures one Windows machine in local Spark mode.
- The shell reported a Windows cleanup error after the last stage saved. The runner recorded completion, all eight notebook cells have successful outputs, and post-run checks passed.
- All ten descriptive types appear in the locally available coursework. Lab 4 uses bar, histogram, scatter, line, box and violin plots. Lab 3 uses area charts, heatmaps, treemaps and Pareto charts.
- Raw CSV/PDF inputs, temporary caches, and readiness/demo files are omitted from the archive. Re-execution uses the originally supplied `data/` directory.
""", encoding="utf-8")

    sources = [NOTEBOOK, PDF, AUDIT, ROOT / "README.md", ROOT / "tools" / "run_asbd.py",
               ROOT / "tools" / "finalize_asbd.py", OUT / "execution_progress.json", *stages]
    if extension_text:
        sources.append(ROOT / "tools" / "benchmark_larger_spark.py")
    prefixes = ("metadata_", "pdf_sample_", "pdf_summary", "fp_", "aclose_", "concept_", "transaction",
                "kmedoids_", "spark_benchmark_", "spark_summary", "spark_figure",
                "descriptive_figures", "all_figures", "legal_vocabulary", "sample_representativeness")
    if extension_text:
        prefixes += ("spark_extended_",)
    sources += [path for path in sorted(OUT.iterdir())
                if path.suffix in {".csv", ".json"} and path.name.startswith(prefixes)]
    sources += [ROOT / "figures" / item["file"] for item in all_figures]
    tmp = ARCHIVE.with_suffix(".tmp")
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(tmp) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {path.relative_to(ROOT).as_posix() for path in sources}
        assert all(archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes() for path in sources)
    tmp.replace(ARCHIVE)
    print(f"Verified {len(sources)} files in {ARCHIVE.name}")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reviewed", action="store_true",
                    help="Package after the rendered report has been visually reviewed.")
args = parser.parse_args()
if args.reviewed:
    package()
else:
    render_and_check()

