# 1. Metadata inventory, cleaning and selection
# Establish the full metadata denominator before sampling. Clean textual fields and placeholder identifiers, derive case categories and chronology fields, and report exact PDF matches. Keep repeated diary identifiers and disputed dates available for audit.
from pathlib import Path
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import get_close_matches
from itertools import combinations

import numpy as np
import pandas as pd
import pymupdf as fitz
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIGURES = ROOT / "figures"
for folder in (OUT, FIGURES):
    folder.mkdir(exist_ok=True)
SEED = 42
MAX_TEXT_CHARS_PER_DOCUMENT = 250_000
MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT = 0.02, 0.50, 1.10


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize_document(text):
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


csv_path = DATA / "judgments.csv"
pdf_dir = DATA / "pdfs"
raw = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
meta = raw.apply(lambda col: col.str.replace(r"\s+", " ", regex=True).str.strip())
for column in meta:
    meta[column] = meta[column].mask(
        meta[column]
        .str.casefold()
        .isin({"", "na", "n/a", "null", "none", "nan", "not available", "-", "--"})
    )
# The archive embeds even the missing diary marker in real filenames.
diary_for_linkage = meta.diary_no.copy()
zero_placeholders = {}
for column in ("diary_no", "case_no"):
    marker = meta[column].str.fullmatch(r"-\s*0", na=False)
    zero_placeholders[column] = int(marker.sum())
    meta[column] = meta[column].mask(marker)
missing = meta.isna().sum().rename("missing")
pd.concat([missing, (100 * missing / len(meta)).rename("percent")], axis=1).to_csv(
    OUT / "metadata_missingness.csv"
)
meta.insert(0, "record_id", np.arange(1, len(meta) + 1))
raw_types = meta["Judgement_type"].value_counts(dropna=False)
types = meta["Judgement_type"].str.casefold()
meta["Judgement_type"] = types.replace(
    {"j": "Judgment", "judgment": "Judgment", "judgement": "Judgment"}
).fillna("Unknown")
meta["language"] = meta.language.str.upper().fillna("Unknown")
meta["judgment_date"] = pd.to_datetime(
    meta.judgment_dates, format="%d-%m-%Y", dayfirst=True, errors="coerce"
)
meta["judgment_year"] = meta.judgment_date.dt.year.astype("Int64")
meta["judgment_month"] = meta.judgment_date.dt.month.astype("Int64")
meta["judgment_decade"] = meta.judgment_year // 10 * 10
meta["date_review"] = np.select(
    [
        meta.judgment_date.isna(),
        meta.judgment_year.lt(1950).fillna(False).to_numpy(dtype=bool),
    ],
    ["Unparsed", "Pre-1950"],
    default="Recorded",
)

types_by_prefix = {
    "CA": "Civil appeal",
    "CRLA": "Criminal appeal",
    "APPEALCRL": "Criminal appeal",
    "WPC": "Civil writ petition",
    "WPCRL": "Criminal writ petition",
    "SLPC": "Civil special leave",
    "SLPCRL": "Criminal special leave",
    "TPC": "Civil transfer petition",
    "TPCRL": "Criminal transfer petition",
    "CONMTPETC": "Civil contempt petition",
    "RPC": "Civil review",
    "RPCRL": "Criminal review",
    "ARBITPETITON": "Arbitration petition",
    "ARBITPETITION": "Arbitration petition",
    "TCC": "Civil transferred case",
    "SPLREF": "Special reference",
    "ORGNLSUIT": "Original suit",
    "SMWC": "Civil suo motu writ",
    "SMWCRL": "Criminal suo motu writ",
    "CURATIVEPETC": "Civil curative petition",
}
case_prefix = meta.case_no.str.extract(
    r"^\s*(.*?)(?:\s+No\.?|\s*[-\d]|$)", expand=False
)
case_prefix = case_prefix.str.upper().str.replace(r"[^A-Z]", "", regex=True)
meta["case_category"] = case_prefix.map(types_by_prefix).fillna("Other/Unknown")

# Legacy NAME,INITIALS fields are ambiguous. Only count explicit modern judge titles.
title = r"\b(?:CHIEF\s+JUSTICE|JUSTICE)\s+(?=[A-Z])"
titles = meta.bench.str.upper().str.count(title)
commas = meta.bench.str.count(",")
meta["bench_size"] = titles.where(titles.gt(0) & commas.eq(titles - 1)).astype("Int64")
government = meta.pet.str.match(
    r"^(?:THE\s+)?(?:STATE OF\b|UNION OF INDIA\b|GOVERNMENT OF\b)", case=False, na=False
)
meta["petitioner_category"] = np.select(
    [government.to_numpy(dtype=bool), meta.pet.isna()],
    ["Explicit government name", "Missing"],
    default="Unclassified",
)

pdf_names = {
    p.name for p in pdf_dir.iterdir() if p.is_file() and p.suffix.casefold() == ".pdf"
}
pdf_dir_size = len(pdf_names)
meta["source_diary_no"] = raw.diary_no
meta["pdf_filename"] = (
    diary_for_linkage + "___" + meta.temp_link.str.replace("/", "__", regex=False)
)
meta["is_pdf_available"] = meta.pdf_filename.isin(pdf_names)
unmatched = meta.loc[
    ~meta.is_pdf_available, ["record_id", "diary_no", "temp_link", "pdf_filename"]
].copy()
by_suffix = defaultdict(list)
unmatched_suffixes = set(
    unmatched.temp_link.dropna().str.replace("/", "__", regex=False)
)
for filename in sorted(pdf_names):
    _, _, suffix = filename.partition("___")
    if suffix in unmatched_suffixes:
        by_suffix[suffix].append(filename)
unmatched["same_link_other_diary"] = unmatched.temp_link.map(
    lambda link: " | ".join(by_suffix.get(link.replace("/", "__"), [])),
    na_action="ignore",
)
unmatched["near_filename"] = unmatched.pdf_filename.map(
    lambda name: " | ".join(
        get_close_matches(
            name,
            (p for p in pdf_names if p.startswith(name.partition("___")[0] + "___")),
            n=2,
            cutoff=0.75,
        )
    ),
    na_action="ignore",
)
unmatched.to_csv(OUT / "metadata_unmatched_pdfs.csv", index=False)

review_dates = meta.loc[
    meta.date_review.ne("Recorded"),
    [
        "record_id",
        "diary_no",
        "case_no",
        "judgment_dates",
        "pdf_filename",
        "is_pdf_available",
    ],
].copy()


def first_page_date_lines(name):
    with fitz.open(pdf_dir / name) as doc:
        lines = "\n".join(
            page.get_text() for page in (doc[i] for i in range(min(2, len(doc))))
        ).splitlines()
    header = [normalize_document(line) for line in lines if "DATE OF JUDGMENT:" in line]
    years = [
        normalize_document(line)
        for line in lines
        if re.search(r"\b(?:19|20)\d{2}\b", line)
    ]
    return " | ".join(header + years[:8])


review_dates["pdf_date_evidence"] = [
    first_page_date_lines(row.pdf_filename) if row.is_pdf_available else ""
    for row in review_dates.itertuples()
]
review_dates.to_csv(OUT / "metadata_date_review.csv", index=False)

duplicate_metrics = {}
for key in ("diary_no", "temp_link"):
    group = (
        meta.loc[meta[key].notna()]
        .groupby(key)
        .agg(
            records=("record_id", "size"),
            distinct_cases=("case_no", "nunique"),
            distinct_dates=("judgment_dates", "nunique"),
            distinct_files=("pdf_filename", "nunique"),
        )
    )
    repeated = group.loc[group.records.gt(1)].sort_values("records", ascending=False)
    repeated.to_csv(OUT / f"metadata_repeated_{key}.csv")
    duplicate_metrics[key] = {
        "groups": len(repeated),
        "rows": int(repeated.records.sum()),
        "extra_rows": int((repeated.records - 1).sum()),
        "groups_with_different_files": int(repeated.distinct_files.gt(1).sum()),
    }

selected = [
    "record_id",
    "source_diary_no",
    "diary_no",
    "Judgement_type",
    "case_no",
    "pet",
    "res",
    "pet_adv",
    "res_adv",
    "bench",
    "judgement_by",
    "judgment_dates",
    "temp_link",
    "language",
    "judgment_date",
    "judgment_year",
    "judgment_month",
    "judgment_decade",
    "date_review",
    "case_category",
    "bench_size",
    "petitioner_category",
    "pdf_filename",
    "is_pdf_available",
]
meta[selected].to_csv(OUT / "metadata_clean.csv", index=False)
date_counts = meta.date_review.value_counts()
metadata = {
    "rows": len(meta),
    "columns": len(raw.columns),
    "csv_bytes": csv_path.stat().st_size,
    "pdf_files": pdf_dir_size,
    "matched_rows": int(meta.is_pdf_available.sum()),
    "matched_unique_files": int(
        meta.loc[meta.is_pdf_available, "pdf_filename"].nunique()
    ),
    "unmatched_rows": len(unmatched),
    "same_link_candidates": int(
        unmatched.same_link_other_diary.fillna("").ne("").sum()
    ),
    "near_name_candidates": int(unmatched.near_filename.fillna("").ne("").sum()),
    "exact_duplicate_rows": int(raw.duplicated().sum()),
    "repeats": duplicate_metrics,
    "missing": {k: int(v) for k, v in missing.items()},
    "zero_placeholders": zero_placeholders,
    "raw_diary_repeat_occurrences": int(raw.diary_no.duplicated().sum()),
    "raw_types": {str(k): int(v) for k, v in raw_types.items()},
    "unparsed_dates": int(date_counts.get("Unparsed", 0)),
    "pre_1950": int(date_counts.get("Pre-1950", 0)),
    "earliest_raw_year": int(meta.judgment_year.min()),
    "latest_raw_year": int(meta.judgment_year.max()),
    "reliable_bench_sizes": int(meta.bench_size.count()),
    "explicit_government_names": int(government.sum()),
}
assert metadata["matched_rows"] + len(unmatched) == len(raw)
assert meta.record_id.is_unique and meta.case_category.notna().all()
write_json(OUT / "metadata_summary.json", metadata)
display(
    pd.DataFrame(
        {
            k: [v]
            for k, v in metadata.items()
            if k
            in (
                "rows",
                "columns",
                "pdf_files",
                "matched_rows",
                "unmatched_rows",
                "pre_1950",
            )
        }
    )
)
display(review_dates)
display(unmatched)
