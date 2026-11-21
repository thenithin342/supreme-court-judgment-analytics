# 2. PDF text and legal-concept transactions
# Sample 4,000 unique matching filenames uniformly without replacement, using seed 42. Keep readable documents with no dictionary hit as empty baskets. The first 600 sampled filenames form the K-Medoids candidate pool.
from hashlib import sha256

PDF_SAMPLE = 4000
CLUSTER_POOL = 600
vocabulary = {
    "anticipatory bail": r"anticipatory bail",
    "bail application": r"bail application",
    "criminal procedure": r"code of criminal procedure|cr\.?\s*p\.?\s*c\.?",
    "penal code": r"indian penal code|i\.?\s*p\.?\s*c\.?",
    "evidence act": r"(?:indian )?evidence act",
    "first information report": r"first information report|f\.?\s*i\.?\s*r\.?",
    "prosecution witness": r"prosecution witness(?:es)?",
    "eyewitness": r"eye[- ]?witness(?:es)?|ocular evidence",
    "medical evidence": r"medical evidence|post[- ]mortem report",
    "dying declaration": r"dying declaration",
    "circumstantial evidence": r"circumstantial evidence",
    "benefit of doubt": r"benefit of (?:the )?doubt|reasonable doubt",
    "murder": r"murder",
    "common intention": r"common intention",
    "life imprisonment": r"life imprisonment|imprisonment for life",
    "death sentence": r"death sentence|death penalty",
    "arbitration": r"arbitration|arbitral tribunal|arbitral award",
    "specific performance": r"specific performance",
    "sale deed": r"sale deed",
    "land acquisition": r"land acquisition",
    "market value": r"market value",
    "compensation": r"compensation",
    "motor accident": r"motor accident(?:s)?|motor vehicles act",
    "negligence": r"negligence|negligent driving",
    "income tax": r"income[- ]tax",
    "sales tax": r"sales tax",
    "excise duty": r"excise duty|central excise",
    "customs duty": r"customs duty|customs act",
    "service law": r"service rules|service jurisprudence|conditions of service",
    "disciplinary proceeding": r"disciplinary proceedings?|departmental enquir(?:y|ies)|departmental inquir(?:y|ies)",
    "natural justice": r"natural justice",
    "industrial dispute": r"industrial disputes?|labour court",
    "reinstatement": r"reinstatement|re[- ]instatement",
    "writ petition": r"writ petition",
    "fundamental rights": r"fundamental rights?",
    "article 14": r"article\s+14",
    "article 21": r"article\s+21",
    "limitation": r"period of limitation|limitation act|barred by limitation",
    "res judicata": r"res judicata",
    "contract": r"contract act|breach of contract",
    "insurance": r"insurance policy|insurance company",
    "burden of proof": r"burden of proof",
    "tenant": r"tenancy|tenant|rent control",
}
patterns = {key: re.compile(r"(?<!\w)(?:" + value + r")(?!\w)", re.I)
            for key, value in vocabulary.items()}
write_json(OUT / "legal_vocabulary.json", vocabulary)

def extract_judgment(pdf_path, max_chars=MAX_TEXT_CHARS_PER_DOCUMENT):
    # Reuse the starter's one-document extraction budget and record truncation.
    result = {"status": "ok", "page_count": None, "text": "", "error": None, "truncated": False}
    try:
        chunks, remaining = [], max_chars
        with fitz.open(pdf_path) as doc:
            result["page_count"] = len(doc)
            for page in doc:
                segment = page.get_text("text")
                if len(segment) > remaining:
                    chunks.append(segment[:remaining])
                    result["truncated"] = True
                    break
                chunks.append(segment)
                remaining -= len(segment)
        result["text"] = normalize_document("\n".join(chunks))
        if not result["text"]:
            result["status"] = "empty_text_or_scanned"
    except (RuntimeError, ValueError, OSError) as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

pool = (meta.loc[meta.is_pdf_available, ["record_id", "diary_no", "pdf_filename", "case_category"]]
        .drop_duplicates("pdf_filename"))
sample = pool.sample(n=min(PDF_SAMPLE, len(pool)), random_state=SEED).reset_index(drop=True)
sample.to_csv(OUT / "pdf_sample_manifest.csv", index=False)
sample_paths = [pdf_dir / name for name in sample.pdf_filename]
file_versions = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sample_paths]
cache_key = sha256(repr((3, file_versions, tuple(vocabulary.items()), MAX_TEXT_CHARS_PER_DOCUMENT,
                         CLUSTER_POOL)).encode()).hexdigest()[:16]
cache = OUT / f"pdf_extract_{cache_key}.jsonl"
cached = {}
fresh_extraction = os.environ.get("ASBD_FRESH_EXTRACTION") == "1"
if cache.exists() and not fresh_extraction:
    with cache.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A torn final write can be rebuilt from the sample manifest.
                continue
            cached[record["pdf_filename"]] = record

cache_hits = sum(path.name in cached for path in sample_paths)
with cache.open("w" if fresh_extraction else "a", encoding="utf-8") as stream:
    if not fresh_extraction:
        stream.write("\n")
    for index, path in enumerate(sample_paths):
        if path.name not in cached:
            result = extract_judgment(path)
            text = result.pop("text")
            result.update(pdf_filename=path.name, word_count=len(text.split()),
                          concepts=[key for key, pattern in patterns.items() if pattern.search(text)],
                          cluster_text=text if index < CLUSTER_POOL else "")
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            cached[path.name] = result
        if (index + 1) % 500 == 0:
            print(f"PDF progress: {index + 1:,}/{len(sample):,}")
            write_json(OUT / "extraction_progress.json",
                       {"processed": index + 1, "sampled": len(sample), "cache_hits": cache_hits})

extracted = sample.merge(pd.DataFrame(cached.values()), on="pdf_filename", validate="one_to_one")
readable = extracted.loc[extracted.status.eq("ok")].copy()
def csv_features(frame):
    return frame.drop(columns="cluster_text").assign(
        concepts=frame.concepts.map(" | ".join),
        error=frame.error.str.replace(r"filename=.*$", "filename recorded separately", regex=True))

csv_features(extracted).to_csv(OUT / "pdf_sample_features.csv", index=False)
pdf_summary = {"seed": SEED, "sampling_frame": len(pool), "sampled": len(sample),
               "readable": len(readable), "statuses": {k: int(v) for k, v in extracted.status.value_counts().items()},
               "truncated": int(extracted.truncated.sum()), "text_char_cap": MAX_TEXT_CHARS_PER_DOCUMENT,
               "cluster_pool": CLUSTER_POOL, "cache_hits": cache_hits,
               "live_pdf_attempts": len(sample) - cache_hits}
assert len(extracted) == len(sample) and extracted.pdf_filename.is_unique
assert len(readable) > 0 and readable.word_count.gt(0).all()
write_json(OUT / "pdf_summary.json", pdf_summary)
display(pd.DataFrame([pdf_summary]).drop(columns="statuses"))
display(csv_features(extracted).drop(columns="error").head(8))

