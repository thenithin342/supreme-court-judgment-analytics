# 7. Local versus PySpark benchmarks
# Both timers start with the same in-memory baskets and end with comparable Python result dictionaries. Input preparation is included in each run. The Spark session starts once, before warm-up and benchmarking.
import tempfile
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.fpm import FPGrowth

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if os.name == "nt":
    # Spark's Windows launcher cannot use Codex's temp path with CMD metacharacters.
    spark_temp = OUT / "spark_temp"
    spark_temp.mkdir(exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(spark_temp)
    tempfile.tempdir = str(spark_temp)
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(spark_temp))

def spark_baskets(spark, baskets, name):
    # JVM-only parsing avoids Python worker launch on Windows.
    payload = F.lit(json.dumps(baskets, separators=(",", ":")))
    return spark.range(1).select(F.explode(F.from_json(payload, "array<array<string>>")).alias(name))

def spark_fp(frame):
    rows = (FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT,
                     minConfidence=MIN_CONFIDENCE, numPartitions=2)
            .fit(frame).freqItemsets.collect())
    return {frozenset(row.items): int(row.freq) for row in rows}

def spark_a_close(spark, frame, n):
    # Each match row is one candidate and a basket containing all its terms.
    def matching_baskets(candidates):
        candidate_frame = spark_baskets(spark, candidates, "candidate")
        return frame.crossJoin(F.broadcast(candidate_frame)).where(
            F.size(F.array_except(F.col("candidate"), F.col("items"))) == 0)

    def count_spark(candidates, cutoff):
        result = (matching_baskets(candidates).groupBy("candidate").count()
                  .where(F.col("count") >= cutoff).collect())
        return {tuple(row.candidate): int(row["count"]) for row in result}

    vocabulary = [row.term for row in frame.select(F.explode("items").alias("term")).distinct().collect()]
    counts = mine_generators(vocabulary, n, count_spark)
    # A term belongs to the closure iff every basket supporting the generator
    # also contains the term. SQL counts those co-occurrences across partitions.
    occurrences = (matching_baskets([sorted(g) for g in counts])
                   .select("candidate", F.explode("items").alias("term"))
                   .groupBy("candidate", "term").count().collect())
    common = {generator: set() for generator in counts}
    for row in occurrences:
        generator = frozenset(row.candidate)
        if row["count"] == counts[generator]:
            common[generator].add(row.term)
    return supports_from_closures(counts, {g: frozenset(terms) for g, terms in common.items()})

def measure_spark(spark, baskets, algorithm):
    start = time.perf_counter()
    frame = spark_baskets(spark, baskets, "items").repartition(2).cache()
    try:
        assert frame.count() == len(baskets)
        prepared = time.perf_counter()
        result = spark_fp(frame) if algorithm == "FP-Growth" else spark_a_close(spark, frame, len(baskets))
        elapsed = time.perf_counter() - start
        return result, elapsed, prepared - start
    finally:
        frame.unpersist(blocking=True)

spark_setup_start = time.perf_counter()
spark = (SparkSession.builder.master("local[2]").appName("ASBD-judgment-analysis")
         .config("spark.ui.enabled", "false")
         .config("spark.driver.bindAddress", "127.0.0.1")
         .config("spark.driver.memory", "768m")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.adaptive.enabled", "false")
         .config("spark.sql.crossJoin.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark_version = spark.version
assert spark.range(4).count() == 4
startup_seconds = time.perf_counter() - spark_setup_start
write_json(OUT / "spark_startup.json", {"version": spark_version, "startup_seconds": startup_seconds})
bench_rows = []
prep_rows = []
workloads = sorted({min(1000, n_tx), n_tx})
repetitions = 2

try:
    # Warm one FPGrowth and one candidate-count path before taking measurements.
    warm_baskets = [("x",), ("x", "y"), ("y",), ()]
    assert measure_spark(spark, warm_baskets, "FP-Growth")[0] == counts_from_fp(local_fp_growth(warm_baskets), 4)
    assert measure_spark(spark, warm_baskets, "A-Close")[0] == local_a_close(warm_baskets)
    warmup_seconds = time.perf_counter() - spark_setup_start - startup_seconds
    for n in workloads:
        baskets = transactions[:n]
        for attempt in range(1, repetitions + 1):
            t = time.perf_counter()
            local_fp = counts_from_fp(local_fp_growth(baskets), n)
            local_seconds = time.perf_counter() - t
            distributed_fp, spark_seconds, prep = measure_spark(spark, baskets, "FP-Growth")
            assert distributed_fp == local_fp
            bench_rows.append({"algorithm": "FP-Growth", "documents": n, "repeat": attempt,
                               "local_seconds": local_seconds, "spark_seconds": spark_seconds,
                               "spark_prepare_seconds": prep, "frequent": len(local_fp), "closed": None, "equal": True})
            pd.DataFrame(bench_rows).to_csv(OUT / "spark_benchmark_runs.csv", index=False)

            t = time.perf_counter()
            local_closed = local_a_close(baskets)
            local_seconds = time.perf_counter() - t
            distributed_closed, spark_seconds, prep = measure_spark(spark, baskets, "A-Close")
            assert distributed_closed == local_closed
            assert {s: c for s, c in distributed_closed[0].items() if s} == distributed_fp
            bench_rows.append({"algorithm": "A-Close", "documents": n, "repeat": attempt,
                               "local_seconds": local_seconds, "spark_seconds": spark_seconds,
                               "spark_prepare_seconds": prep, "frequent": len(local_fp), "closed": len(local_closed[2]), "equal": True})
            pd.DataFrame(bench_rows).to_csv(OUT / "spark_benchmark_runs.csv", index=False)
            print(f"Saved verified benchmark: n={n:,}, repetition {attempt}/{repetitions}")
        prep_rows.append({"documents": n, "spark_cache_seconds": float(np.median(
            [row["spark_prepare_seconds"] for row in bench_rows if row["documents"] == n]))})
finally:
    spark.stop()

benchmark = pd.DataFrame(bench_rows)
timing = benchmark.groupby(["algorithm", "documents"], sort=False).agg(
    local_seconds=("local_seconds", "median"), spark_seconds=("spark_seconds", "median"),
    frequent=("frequent", "first"), closed=("closed", "first"), equal=("equal", "all")).reset_index()
timing["speedup"] = timing.local_seconds / timing.spark_seconds
timing.to_csv(OUT / "spark_benchmark_summary.csv", index=False)
spark_summary = {"version": spark_version, "master": "local[2]", "startup_seconds": startup_seconds,
                 "warmup_seconds": warmup_seconds, "preparation": prep_rows,
                 "repetitions": repetitions, "workloads": workloads, "all_supports_equal": bool(timing.equal.all())}
write_json(OUT / "spark_summary.json", spark_summary)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), layout="constrained", sharey=True)
for ax, (algorithm, group) in zip(axes, timing.groupby("algorithm", sort=False)):
    positions = np.arange(len(group))
    ax.bar(positions - .18, group.local_seconds, width=.34, color=BLUE, label="Local")
    ax.bar(positions + .18, group.spark_seconds, width=.34, color=TEAL, label="Spark")
    ax.set(title=algorithm, yscale="log", xlabel="Transactions",
           xticks=positions, xticklabels=[f"{n:,}" for n in group.documents])
axes[0].set_ylabel("Median seconds (log scale)")
axes[0].legend(frameon=False)
fig.suptitle("Measured local and PySpark runtime", fontsize=13, fontweight="bold")
full = timing[timing.documents.eq(n_tx)]
runtime_inference = "Full-sample local/Spark runtime ratios are " + ", ".join(
    f"{row.algorithm} {row.speedup:.3g}×" for row in full.itertuples()) + ". Values below 1 indicate slower Spark execution."
save_chart(fig, "13_runtime_comparison.png", "Runtime bars", "Local and Spark benchmark",
           f"{repetitions} measured runs per algorithm and workload after startup.", runtime_inference)
write_json(OUT / "spark_figure.json", artifacts[-1])
display(pd.DataFrame([{"Spark": spark_version, "Startup (s)": startup_seconds, "Warm-up (s)": warmup_seconds}]))
display(timing.style.format({"local_seconds": "{:.4f}", "spark_seconds": "{:.4f}", "speedup": "{:.4f}", "closed": "{:.0f}"}, na_rep="—"))

