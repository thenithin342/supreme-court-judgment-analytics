# 4. Local FP-Growth
# Encode each readable PDF as a binary basket of legal concepts. Mine frequent sets at 2% support, then retain rules with at least 50% confidence and 1.10 lift. Interpret the rules as textual co-occurrence.
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

transactions = [tuple(sorted(set(items))) for items in readable.concepts]
n_tx = len(transactions)


def local_fp_growth(baskets):
    encoder = TransactionEncoder().fit(baskets)
    one_hot = pd.DataFrame.sparse.from_spmatrix(
        encoder.transform(baskets, sparse=True), columns=encoder.columns_
    )
    items = fpgrowth(one_hot, min_support=MIN_SUPPORT, use_colnames=True)
    return items


def counts_from_fp(items, n):
    return {
        item: round(fraction * n)
        for item, fraction in zip(items.itemsets, items.support)
    }


def label_set(itemset):
    return " | ".join(sorted(itemset))


def item_frame(counts, n):
    rows = [
        {
            "itemset": label_set(item),
            "length": len(item),
            "support_count": count,
            "support": count / n,
        }
        for item, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["length", "support_count", "itemset"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def tidy_rules(frame):
    return frame.assign(
        antecedent=frame.antecedents.map(label_set),
        consequent=frame.consequents.map(label_set),
    )[["antecedent", "consequent", "support", "confidence", "lift"]].sort_values(
        ["lift", "confidence", "support", "antecedent", "consequent"],
        ascending=[False, False, False, True, True],
        ignore_index=True,
    )


fp_items = local_fp_growth(transactions)
fp_counts = counts_from_fp(fp_items, n_tx)
fp_table = item_frame(fp_counts, n_tx)
fp_table.to_csv(OUT / "fp_itemsets.csv", index=False)
fp_lengths = fp_table.groupby("length").size().rename("itemsets")
fp_lengths.to_csv(OUT / "fp_itemsets_by_length.csv")
fp_rules_raw = association_rules(
    fp_items, metric="confidence", min_threshold=MIN_CONFIDENCE
)
fp_rules_raw = fp_rules_raw.loc[fp_rules_raw.lift.ge(MIN_LIFT)]
fp_rules = tidy_rules(fp_rules_raw)
fp_rules.to_csv(OUT / "fp_rules.csv", index=False)
for row in fp_rules_raw.itertuples():
    joint = fp_counts[row.antecedents | row.consequents]
    assert np.isclose(row.support, joint / n_tx)
    assert np.isclose(row.confidence, joint / fp_counts[row.antecedents])
    assert np.isclose(row.lift, row.confidence * n_tx / fp_counts[row.consequents])

concept_freq = Counter(item for tx in transactions for item in tx)
concept_table = pd.DataFrame(
    concept_freq.most_common(), columns=["concept", "documents"]
)
concept_table["support"] = concept_table.documents / n_tx
concept_table.to_csv(OUT / "concept_frequencies.csv", index=False)
readable[["record_id", "pdf_filename", "concepts"]].assign(
    concepts=readable.concepts.map(" | ".join)
).to_csv(OUT / "transactions.csv", index=False)
basket_lengths = pd.Series([len(tx) for tx in transactions], name="items")
basket_lengths.value_counts().sort_index().rename("baskets").to_csv(
    OUT / "transaction_sizes.csv"
)
strong_rules = []
# Favor simple, mutually distinct concept pairs in the narrative.
seen_pairs = set()
for row in fp_rules.itertuples(index=False):
    pair = frozenset((row.antecedent, row.consequent))
    if " | " in row.antecedent or " | " in row.consequent or pair in seen_pairs:
        continue
    seen_pairs.add(pair)
    strong_rules.append(
        f"{row.antecedent.capitalize()} → {row.consequent}: {row.confidence:.1%} of documents mentioning "
        f"the antecedent also mention the consequent, against {row.confidence/row.lift:.1%} "
        f"in the entire readable sample (support {row.support:.1%}, lift {row.lift:.2f})."
    )
    if len(strong_rules) == 3:
        break

fp_summary = {
    "transactions": n_tx,
    "empty": int(basket_lengths.eq(0).sum()),
    "observed_vocabulary": len(concept_freq),
    "mean_items": float(basket_lengths.mean()),
    "max_items": int(basket_lengths.max()),
    "min_support": MIN_SUPPORT,
    "min_confidence": MIN_CONFIDENCE,
    "min_lift": MIN_LIFT,
    "items": len(fp_table),
    "itemsets_by_length": {int(k): int(v) for k, v in fp_lengths.items()},
    "rules": len(fp_rules),
    "interpretations": strong_rules,
}
write_json(OUT / "fp_summary.json", fp_summary)
fig, ax = chart()
points = ax.scatter(
    fp_rules.support,
    fp_rules.confidence,
    c=fp_rules.lift,
    cmap="viridis",
    s=20,
    alpha=0.75,
)
ax.set(title="FP-Growth rule quality", xlabel="Support", ylabel="Confidence")
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.colorbar(points, ax=ax, label="Lift")
save_chart(
    fig,
    "11_fp_rule_quality.png",
    "Rule-quality scatter",
    "FP-Growth rule quality",
    f"{len(fp_rules):,} rules across {n_tx:,} baskets.",
    f"Confidence ranges from {fp_rules.confidence.min():.1%} to {fp_rules.confidence.max():.1%}. "
    f"Lift ranges from {fp_rules.lift.min():.2f} to {fp_rules.lift.max():.2f}.",
)
write_json(OUT / "fp_figure.json", artifacts[-1])
display(
    pd.DataFrame([fp_summary]).drop(columns=["interpretations", "itemsets_by_length"])
)
display(fp_table.head(10))
display(fp_rules.head(10))
for interpretation in strong_rules:
    display(Markdown(interpretation))
