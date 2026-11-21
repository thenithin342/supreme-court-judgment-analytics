# 8. Report from executed results and notebook checks
# Build the academic PDF directly from the tables and figures produced above. A separate final audit checks the saved notebook, renders the PDF and verifies the submission archive.
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (Image as ReportImage, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Compare the full record mix with the exact readable workload used for mining.
sample_metadata = pd.read_csv(OUT / "metadata_clean.csv", usecols=["record_id", "case_category"])
sample_features = pd.read_csv(OUT / "pdf_sample_features.csv", usecols=["record_id", "status"])
readable_categories = sample_features.loc[sample_features.status.eq("ok"), ["record_id"]].merge(
    sample_metadata, on="record_id", how="left", validate="one_to_one"
)
assert sample_metadata.case_category.notna().all() and readable_categories.case_category.notna().all()
assert len(sample_metadata) == metadata["rows"] and len(readable_categories) == n_tx

category_mix = pd.DataFrame({"full_count": sample_metadata.case_category.value_counts()})
category_mix.index.name = "case_category"
category_mix["readable_count"] = readable_categories.case_category.value_counts().reindex(
    category_mix.index, fill_value=0
)
category_mix["full_pct"] = 100 * category_mix["full_count"] / len(sample_metadata)
category_mix["readable_pct"] = 100 * category_mix["readable_count"] / len(readable_categories)
category_mix["difference_pp"] = category_mix["readable_pct"] - category_mix["full_pct"]
category_mix.to_csv(OUT / "sample_representativeness.csv", float_format="%.8f")
main_groups = ", ".join(
    f"{category} {row.full_pct:.2f}% to {row.readable_pct:.2f}% ({row.difference_pp:+.2f} pp)"
    for category, row in category_mix.head(3).iterrows()
)
largest_gap = category_mix["difference_pp"].abs().max()
print("Case-category comparison (difference = readable percentage minus full percentage)")
print(category_mix.round(4).to_string())

PDF_PATH = ROOT / "ASBD_EndSem_Report.pdf"
ink, muted, accent = [colors.HexColor(c) for c in ("#172f46", "#4b6174", "#13736f")]
paper, line = colors.HexColor("#eef5f8"), colors.HexColor("#cbd9e2")
styles = {
    "tag": ParagraphStyle("tag", fontName="Helvetica-Bold", fontSize=9, leading=12,
                          textColor=accent, spaceAfter=7),
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=30, leading=34,
                            textColor=ink, spaceAfter=12),
    "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=22, leading=26,
                              textColor=ink, spaceAfter=8),
    "sub": ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=12, leading=15,
                          textColor=ink, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.2,
                           textColor=ink, spaceAfter=7),
    "caption": ParagraphStyle("caption", fontName="Helvetica", fontSize=8.5, leading=11.5,
                              textColor=muted, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.3, leading=10.6, textColor=ink),
    "cell_head": ParagraphStyle("cell_head", fontName="Helvetica-Bold", fontSize=8.2, leading=10.4,
                               textColor=colors.white),
    "metric": ParagraphStyle("metric", fontName="Helvetica-Bold", fontSize=18, leading=22,
                             textColor=ink, alignment=TA_CENTER),
    "metric_label": ParagraphStyle("metric_label", fontName="Helvetica", fontSize=8.5, leading=10.5,
                                   textColor=muted, alignment=TA_CENTER),
}
story = []

def rich(value, style):
    safe = str(value).replace("→", " to ").replace("—", "-").replace("–", "-").replace("×", "x")
    return Paragraph(escape(safe), styles[style])

def add(value, style="body"):
    story.append(rich(value, style))

def section(tag, title):
    story.append(PageBreak())
    add(tag, "tag")
    add(title, "section")

def table(headers, rows, widths):
    contents = [[rich(x, "cell_head") for x in headers]]
    contents += [[rich(x, "cell") for x in row] for row in rows]
    result = Table(contents, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ink),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, paper]),
        ("LINEBELOW", (0, -1), (-1, -1), .45, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([result, Spacer(1, 8)])

def figure(item, width=487):
    number = item["file"].split("_", 1)[0]
    image = ReportImage(str(FIGURES / item["file"]), width=width, height=width*3.65/8.2)
    image.hAlign = "CENTER"
    story.append(image)
    add(f"Figure {number}. {item['type']}. "
        f"{item['basis']} {item['inference']}", "caption")

add("Indian Institute of Information Technology Design & Manufacturing Kancheepuram, Chennai-127", "tag")
add("25CS5004 / CS 5010  |  Analytics & Systems of Big Data Practice", "sub")
add("End-semester examination  |  November 2025  |  Maximum marks 50", "caption")
add("Indian Supreme Court judgments", "title")
add("Metadata coverage, legal-concept co-occurrence and text clustering")
metrics = [[f"{metadata['rows']:,}", f"{metadata['matched_rows']/metadata['rows']:.2%}",
            f"{pdf_summary['readable']:,}", f"{cluster_summary['sample']:,}"],
           ["Metadata records", "Exact PDF match", "Readable sampled PDFs", "Clustered documents"]]
cards = Table([[rich(x, "metric") for x in metrics[0]], [rich(x, "metric_label") for x in metrics[1]]],
              colWidths=[124.5] * 4, hAlign="LEFT")
cards.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), paper),
                           ("TOPPADDING", (0, 0), (-1, 0), 9),
                           ("BOTTOMPADDING", (0, 1), (-1, 1), 9)]))
story.extend([cards, Spacer(1, 8)])
add("Scope and data preparation", "sub")
add(f"The source CSV contains {metadata['columns']} original fields and the archive contains "
    f"{metadata['pdf_files']:,} PDF files. The analysis uses dates and case numbers for chronology and categories, "
    "party and advocate fields for coverage, judge and bench fields for completeness, and matching PDFs for text. "
    "Whitespace and missing markers were standardized. J and judgment map to one Judgment category.")
d = metadata["repeats"]["diary_no"]
l = metadata["repeats"]["temp_link"]
add(f"The CSV contains {metadata['exact_duplicate_rows']:,} completely duplicate rows. "
    f"{d['groups']:,} diary IDs repeat ({d['extra_rows']:,} extra occurrences), and {l['groups']:,} links repeat "
    f"({l['extra_rows']:,} extra occurrences). {d['groups_with_different_files']:,} repeated diary groups and "
    f"{l['groups_with_different_files']:,} repeated link groups point to different constructed filenames. "
    "All rows remain in the metadata analysis. The PDF sampling frame uses unique filenames.")
add(f"The exact filename rule leaves {metadata['unmatched_rows']} unmatched rows. "
    f"{metadata['same_link_candidates']} have a same-link match under a different diary number and "
    f"{metadata['near_name_candidates']} have a similar filename under the same diary number. "
    "The similar filenames were logged for review and left unmatched because some point to a different language or document.")
add("Derived fields and disputed dates", "sub")
add(f"Dates use day-first parsing. Case groups use leading case-number patterns. Bench size is populated in "
    f"{metadata['reliable_bench_sizes']:,} records with explicit judge titles and compatible comma separators. "
    "Petitioner groups flag explicit government names and preserve unclassified and missing values. "
    "PDF presence is a filename check. Page and word counts are derived for the sampled documents.")
table(["Record", "CSV date", "Case year", "PDF date header"], [
    (f"{row.record_id:,}", row.judgment_dates, row.case_no.rsplit(" ", 1)[-1],
     row.pdf_date_evidence.split(" | ", 1)[0].removeprefix("DATE OF JUDGMENT: "))
    for row in review_dates.itertuples()
], [68, 86, 65, 279])
add(f"The {metadata['pre_1950']} disputed dates are retained in the metadata and disclosed here. "
    "The PDF headers are malformed and the case dates contradict 1902. The chronology figures omit those records "
    f"without assigning replacement dates. {metadata['unparsed_dates']} dates failed to parse.", "caption")

pairs = [("Case mix and chronology", 0, 1), ("Length and case groups", 2, 3),
         ("Word-count distributions and monthly coverage", 4, 5), ("Length and cumulative coverage", 6, 7),
         ("Petitioner labels and missingness", 8, 9)]
for heading, first, second in pairs:
    section("DESCRIPTIVE ANALYTICS", heading)
    figure(artifacts[first])
    figure(artifacts[second])

section("ASSOCIATION RULES", "FP-Growth")
empty_files = int(extracted.error.fillna("").str.startswith("EmptyFileError:").sum())
empty_text = pdf_summary["statuses"].get("empty_text_or_scanned", 0)
add(f"PyMuPDF read one document at a time from a uniform sample of {pdf_summary['sampled']:,} "
    f"of {pdf_summary['sampling_frame']:,} distinct linked PDFs (seed {SEED}). "
    f"{pdf_summary['sampled']-pdf_summary['readable']:,} did not yield usable text, including "
    f"{empty_files:,} empty files and {empty_text:,} files with an empty text layer. "
    f"{pdf_summary['truncated']:,} were truncated at {pdf_summary['text_char_cap']:,} characters. "
    "The extraction budget applies to text features. Page count comes from the PDF itself.")
add(f"One distinct-concept basket per readable document gave {n_tx:,} transactions, including "
    f"{fp_summary['empty']:,} empty baskets. {fp_summary['observed_vocabulary']} concepts occurred, with a mean "
    f"of {fp_summary['mean_items']:.2f} and a maximum of {fp_summary['max_items']} per document. "
    "Word-boundary phrase patterns capture mentions and omit broad terms such as Supreme Court and judgment.")
size_distribution = ", ".join(f"L{length}={count}" for length, count in fp_lengths.items())
add(f"At {MIN_SUPPORT:.0%} minimum support, FP-Growth found {len(fp_counts):,} frequent itemsets ({size_distribution}). "
    f"{len(fp_rules):,} rules met {MIN_CONFIDENCE:.0%} minimum confidence and {MIN_LIFT:.2f} minimum lift. "
    "Support divides joint occurrence by all readable documents. Confidence conditions on the antecedent. "
    "Lift divides confidence by the consequent's baseline prevalence.")
table(["Antecedent", "Consequent", "Support", "Confidence", "Lift"], [
    (r.antecedent, r.consequent, f"{r.support:.1%}", f"{r.confidence:.1%}", f"{r.lift:.2f}")
    for r in fp_rules.head(4).itertuples()
], [150, 149, 60, 73, 66])
figure(artifacts[10], width=430)
for rule in strong_rules:
    add(rule, "caption")
add("Read these rules as measures of textual co-occurrence.", "caption")
support_sweep_path = OUT / "fp_support_sensitivity.csv"
if support_sweep_path.is_file():
    support_sweep = pd.read_csv(support_sweep_path)
    baseline = support_sweep.loc[support_sweep.min_support.eq(MIN_SUPPORT)].iloc[0]
    if (baseline.documents, baseline.frequent_itemsets, baseline.rules) == (n_tx, len(fp_counts), len(fp_rules)):
        sweep_results = ", ".join(
            f"{r.min_support:.1%} gives {r.frequent_itemsets} itemsets and {r.rules} rules"
            for r in support_sweep.itertuples()
        )
        add(f"Support sensitivity on the same baskets and fixed confidence and lift thresholds. "
            f"{sweep_results.capitalize()}. Higher support excludes rarer patterns.", "caption")

section("CLOSED ITEMSETS", "A-Close")
add("A-Close joins frequent minimal generators level by level. If a candidate has the same support as an "
    "immediate subset, it is pruned before the next join. Locally, transaction-ID bit masks count the "
    "matching baskets. Intersecting the contents of those baskets gives the generator's closure.")
add(f"The complete lattice contains {len(fp_counts):,} nonempty frequent itemsets and "
    f"{len(closed_counts):,} closed itemsets. The implementation found {len(generators):,} generators "
    f"including the empty generator and {len(a_rules):,} rules whose unions are closed.")
table(["Closed concept set", "Length", "Count", "Support"], [
    (r.itemset, r.length, r.support_count, f"{r.support:.1%}")
    for r in closed_table.sort_values("support_count", ascending=False).head(7).itertuples()
], [294, 59, 70, 75])
add("Selected rules from closed unions", "sub")
table(["Antecedent", "Consequent", "Support", "Confidence", "Lift"], [
    (r.antecedent, r.consequent, f"{r.support:.1%}", f"{r.confidence:.1%}", f"{r.lift:.2f}")
    for r in a_rules.head(4).itertuples()
], [150, 149, 60, 73, 66])
add(f"Exhaustive subset enumeration verified {aclose_summary['exhaustive_tests']} small test datasets. "
    "On the judgment sample, every returned closure was checked against its supporting baskets. "
    "The full frequent support dictionary equals FP-Growth's, and an independent same-support extension check agrees on the closed sets.")
add(f"Closed representation at {MIN_SUPPORT:.0%} minimum support", "sub")
table(["Documents", "Frequent", "Closed", "Nonclosed", "Reduction"], [
    (f"{row['documents']:,}", row["frequent"], row["closed"], row["nonclosed"], f"{row['reduction']:.2%}")
    for row in comparison
], [100, 95, 95, 100, 108])
smaller_label = "{" + closure_example["itemset"].replace(" | ", ", ") + "}"
larger_label = "{" + closure_example["closure"].replace(" | ", ", ") + "}"
add(f"In the first {closure_example['documents']:,} transactions, {smaller_label} and its closure "
    f"{larger_label} both occur in {closure_example['support_count']} documents "
    f"({closure_example['support']:.1%} support). Every basket containing the smaller set also contains "
    "the additional concept, so the strict superset preserves its support and the smaller set is nonclosed.")

section("TEXT CLUSTERING", "K-Medoids")
add("The judgment-type spellings normalize to a single value. K-Medoids groups documents by their "
    "text and identifies an actual judgment as the representative of each group.")
add(f"The first {CLUSTER_POOL} sampled PDFs formed the candidate pool. After filtering unsuccessful extraction, "
    f"known vernacular filenames, documents under {cluster_summary['min_words']} words and zero-feature rows, "
    f"{len(cluster_data):,} documents remained. TF-IDF used {tfidf.shape[1]:,} unigrams and bigrams. "
    f"Truncated SVD, unit normalization and cosine distance support the medoid-swap search.")
table(["k", "Silhouette", "Total cosine distance", "Starts"], [
    (r.k, f"{r.silhouette:.3f}", f"{r.total_cosine_distance:.2f}", r.starts)
    for r in trial_table.itertuples()
], [55, 125, 180, 138])
table(["Cluster", "Size", "Medoid diary / CSV row", "Characteristic terms"], [
    (r.cluster, r.size, f"{r.medoid_diary_no} / {r.medoid_record:,}", r.characteristic_terms)
    for r in cluster_table.itertuples()
], [50, 47, 123, 278])
figure(artifacts[11], width=453)
tax_cluster = next(row for row in cluster_table.itertuples()
                   if {"tax", "assessment"} <= set(row.characteristic_terms.split(", ")))
cluster_interpretation = (
    f"k={selected_k} had the highest silhouette among the tested values "
    f"({', '.join(str(k) for k in trial_table.k)}). The search covers only those values. "
    f"The silhouette of {best_score:.3f} indicates weak separation, so the text clusters are exploratory. "
    f"For example, cluster {tax_cluster.cluster} has {tax_cluster.size} documents and medoid diary "
    f"{tax_cluster.medoid_diary_no} (CSV row {tax_cluster.medoid_record:,}). Characteristic terms such as "
    "tax, assessee and assessment suggest a tax-assessment theme."
)
add(cluster_interpretation, "caption")

section("DISTRIBUTED ANALYTICS", "Local and PySpark benchmarks")
add(f"Spark {spark_version} ran on two local threads. Session startup and the first action took "
    f"{startup_seconds:.2f} s. A separate warm-up took {warmup_seconds:.2f} s. "
    "Both were timed before the benchmark.")
add(f"Each table entry is the median of {repetitions} runs. Both implementations start with the same "
    "Python baskets and finish with results in driver memory. Times include local encoding or Spark input "
    "preparation, mining and result collection as applicable. Spark runs native FP-Growth and a custom "
    "A-Close with distributed support counting. Speedup is local time divided by Spark time.")
table(["Algorithm", "Baskets", "Local s", "Spark s", "Speedup", "Frequent / closed"], [
    (r.algorithm, f"{r.documents:,}", f"{r.local_seconds:.4f}", f"{r.spark_seconds:.4f}",
     f"{r.speedup:.4f}x", f"{r.frequent} / {int(r.closed) if pd.notna(r.closed) else '-'}")
    for r in timing.itertuples()
], [87, 65, 64, 64, 71, 147])
figure(artifacts[12], width=453)
faster = int(timing.speedup.gt(1).sum())
add(f"Spark was faster in {faster} of {len(timing)} algorithm/workload comparisons. "
    "All local and Spark runs agreed on frequent itemsets and support counts. The two A-Close "
    "implementations also agreed on generator-to-closure mappings. In local mode, the JVM schedules tasks "
    "on the same machine. Converting Python baskets, repartitioning and collecting results on the driver "
    "add overhead. A-Close also launches jobs at each candidate level. The measured workloads were too "
    "small to offset these costs. A multi-machine cluster requires separate measurements.")
extended_path = OUT / "spark_extended_summary.json"
if extended_path.is_file():
    from json import loads

    extension = loads(extended_path.read_text(encoding="utf-8"))
    extended_timing = pd.read_csv(OUT / "spark_extended_benchmark_summary.csv")
    add("Larger workload on the same machine", "sub")
    add(f"A new sample of {extension['sampled']:,} distinct PDF filenames outside the original sample "
        f"added {extension['added_readable']:,} readable documents. Both engines mined the same "
        f"{extension['documents']:,} baskets. The table reports medians of {extension['repetitions']} paired "
        "runs per algorithm, including input preparation and result collection.")
    table(["Algorithm", "Local s", "Spark s", "Speedup"], [
        (r.algorithm, f"{r.local_seconds:.4f}", f"{r.spark_seconds:.4f}", f"{r.speedup:.4f}x")
        for r in extended_timing.itertuples()
    ], [168, 110, 110, 110])
    larger_faster = int(extended_timing.speedup.gt(1).sum())
    add(f"All {extension['verified_matches']} paired runs agreed. Spark was faster for "
        f"{larger_faster} of {len(extended_timing)} algorithms at this workload.", "caption")

section("CONCLUSION", "What the results support")
add(f"The CSV is suitable for record-level counts and dates after review. {categories.index[0]} is the largest "
    f"derived case category, with {categories.iloc[0]:,} records. The full metadata retains all {len(meta):,} source rows. "
    "Bench, author, language and advocate fields have uneven coverage and should be interpreted with their denominators.")
add(f"The text sample supports reproducible co-mention estimates for the documented vocabulary. "
    f"A-Close verifies {len(closed_counts):,} closed itemsets among {len(fp_counts):,} frequent itemsets. "
    f"K-Medoids selects k={selected_k} with a silhouette of {best_score:.3f}, indicating weak separation. "
    "Cluster terms describe exploratory vocabulary patterns.")
add("Limitations and reproducibility", "sub")
add("The CSV counts cover only records in the supplied archive. Exact filenames establish linkage, while separate files can be "
    "translations or versions of the same case. Metadata fields and chronology are incomplete in ways that may vary over time. "
    "The benchmark uses a single machine, local Spark, a fixed vocabulary and measured workload sizes.")
add("Text extraction omits OCR. English concept patterns miss untranslated forms and paraphrases, can match citations or "
    "negated statements, and cannot identify which side advanced a claim. The character cap can suppress late mentions. "
    "The clustering language filter relies in part on filename markers, and the PCA plot only shows two projected coordinates.")
add(f"Case-category shares for the {len(sample_metadata):,} metadata records and {len(readable_categories):,} "
    f"readable mining PDFs differed by at most {largest_gap:.2f} percentage points. "
    f"Full-to-readable shares for the three largest groups were {main_groups}. The broad case-category mix is similar. "
    "Exact filename matching, sampling unique files and excluding unreadable PDFs can still affect coverage "
    "by year, language or document type.")
add("Source and outputs", "sub")
source_url = "https://www.kaggle.com/datasets/vangap/indian-supreme-court-judgments"
story.append(Paragraph('Dataset supplied with the paper: <link href="' + source_url +
                       '" color="#13736f"><u>Indian Supreme Court Judgments (vangap, Kaggle)</u></link>.',
                       styles["body"]))
add("The workflow uses the supplied judgments.csv and PDFs. main.ipynb contains the executed analysis. "
    "The accompanying CSV and JSON files contain full tables, denominators, benchmark repeats, vocabulary and figure inferences.")

def footer(canvas, doc):
    width, _ = A4
    canvas.saveState()
    canvas.setStrokeColor(line)
    canvas.line(48, 42, width - 48, 42)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(muted)
    canvas.drawString(48, 28, "25CS5004 / CS 5010   |   NOVEMBER 2025   |   INDIAN SUPREME COURT JUDGMENTS")
    canvas.drawRightString(width - 48, 28, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=42, leftMargin=42,
                        topMargin=42, bottomMargin=54, title="November 2025 End-Semester | Indian Supreme Court judgments")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
assert len(artifacts) == 13 and all((FIGURES / item["file"]).is_file() for item in artifacts)
with fitz.open(PDF_PATH) as report:
    assert len(report) >= 10 and all(page.get_text().strip() for page in report)
    print(f"Saved {PDF_PATH.name}: {len(report)} pages")
print(cluster_interpretation)
write_json(OUT / "all_figures.json", artifacts)

