# 5. A-Close and independent closure verification
# Only frequent minimal generators contribute candidates to the next level. Intersecting each generator's matching baskets yields its closure and preserves its frequency.
from math import ceil

def mine_generators(items, n, count_candidates, level_log=None):
    cutoff = ceil(n * MIN_SUPPORT)
    candidates = [(item,) for item in sorted(items)]
    counts = {frozenset(): n}
    while candidates:
        observed = count_candidates(candidates, cutoff)
        frequent = {terms: int(count) for terms, count in observed.items() if count >= cutoff}
        # Equal support with an immediate subset gives the same matching baskets.
        # Every superset inherits this redundancy, so discard the candidate before joining.
        level = {}
        for terms, count in frequent.items():
            itemset = frozenset(terms)
            if all(counts[itemset - {term}] > count for term in terms):
                level[terms] = count
                counts[itemset] = count
        if level_log is not None:
            level_log.append({"length": len(candidates[0]), "candidates_counted": len(candidates),
                              "frequent_candidates": len(frequent), "minimal_generators": len(level)})
        if not level:
            break
        groups = defaultdict(list)
        for itemset in sorted(level):
            groups[itemset[:-1]].append(itemset[-1])
        candidates = [
            prefix + suffix for prefix, choices in groups.items() for suffix in combinations(choices, 2)
            if all(subset in level for subset in combinations(prefix + suffix, len(prefix) + 1))
        ]
    return counts

def supports_from_closures(generator_counts, mapping):
    # Every frequent set is contained in its own closure. Its support is the
    # maximum support of its closed supersets. Expand for rule denominators and QA.
    closed = {closure: generator_counts[g] for g, closure in mapping.items() if closure}
    counts = {frozenset(): generator_counts[frozenset()]}
    for closure, count in sorted(closed.items(), key=lambda item: -item[1]):
        for length in range(1, len(closure) + 1):
            for subset in combinations(closure, length):
                counts.setdefault(frozenset(subset), count)
    return counts, mapping, closed

def local_a_close(baskets, level_log=None):
    # An integer bit mask records the transaction IDs containing each concept.
    universe = (1 << len(baskets)) - 1
    vertical = defaultdict(int)
    for index, basket in enumerate(baskets):
        bit = 1 << index
        for term in basket:
            vertical[term] |= bit

    def hits(candidate):
        relevant = universe
        for term in candidate:
            relevant &= vertical[term]
        return relevant

    def count_local(candidates, cutoff):
        return {candidate: hits(candidate).bit_count() for candidate in candidates}

    counts = mine_generators(vertical, len(baskets), count_local, level_log)
    mapping = {}
    for generator in counts:
        relevant = hits(generator)
        closure = frozenset(term for term, tids in vertical.items() if (tids & relevant) == relevant)
        mapping[generator] = closure
    return supports_from_closures(counts, mapping)

def exhaustive_closed(baskets):
    universe = sorted(set().union(*baskets))
    cutoff = ceil(len(baskets) * MIN_SUPPORT)
    reference = {}
    for size in range(1, len(universe) + 1):
        for terms in combinations(universe, size):
            subset = frozenset(terms)
            hits = [basket for basket in baskets if subset <= basket]
            if len(hits) >= cutoff and set.intersection(*hits) == subset:
                reference[subset] = len(hits)
    return reference

test_baskets = [{"always", "a", "b"}, {"always", "a", "b", "c"},
                {"always", "c"}, {"always", "d", "c"}, {"always", "d"}]
test_log = []
test_counts, test_generators, test_closed = local_a_close(test_baskets, test_log)
assert test_closed == exhaustive_closed(test_baskets)
assert test_generators[frozenset()] == frozenset({"always"})
assert frozenset({"a", "b"}) not in test_generators
assert test_counts[frozenset({"a", "b", "c"})] == 1
assert test_log[1]["candidates_counted"] == 6  # 4 retained singletons, "always" was pruned.
assert all(row["length"] < 3 for row in test_log)  # Redundant pair blocks extension.
rng = np.random.default_rng(SEED)
for _ in range(12):
    check = [set(np.array(list("abcd"))[row]) for row in rng.random((20, 4)) < .45]
    assert local_a_close(check)[2] == exhaustive_closed(check)

generator_levels = []
a_counts, generators, closed_counts = local_a_close(transactions, generator_levels)
pd.DataFrame(generator_levels).to_csv(OUT / "aclose_levels.csv", index=False)
assert {item: count for item, count in a_counts.items() if item} == fp_counts
tx_sets = [set(tx) for tx in transactions]
for closure, count in closed_counts.items():
    hits = [basket for basket in tx_sets if closure <= basket]
    assert len(hits) == count and set.intersection(*hits) == closure
# In a downward-closed lattice, a same-support strict superset implies a same-support immediate extension.
nonclosed = {item - {term}
             for item, count in a_counts.items() for term in item
             if a_counts[item - {term}] == count}
assert set(closed_counts) == set(fp_counts) - nonclosed

closed_table = item_frame(closed_counts, n_tx)
closed_table.to_csv(OUT / "aclose_closed_itemsets.csv", index=False)
pd.DataFrame([{"generator": label_set(g), "closure": label_set(c), "support_count": a_counts[g]}
              for g, c in generators.items()]).to_csv(OUT / "aclose_generators.csv", index=False)
rule_rows = []
for closed, joint in closed_counts.items():
    for size in range(1, len(closed)):
        for left in map(frozenset, combinations(sorted(closed), size)):
            right = closed - left
            confidence = joint / a_counts[left]
            lift = confidence * n_tx / a_counts[right]
            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rule_rows.append({"antecedents": left, "consequents": right,
                                  "support": joint / n_tx, "confidence": confidence, "lift": lift})
a_rules = tidy_rules(pd.DataFrame(rule_rows))
a_rules.to_csv(OUT / "aclose_rules.csv", index=False)
pairs_fp = {(r.antecedent, r.consequent) for r in fp_rules.itertuples()}
assert {(r.antecedent, r.consequent) for r in a_rules.itertuples()} <= pairs_fp
aclose_summary = {"frequent": len(fp_counts), "closed": len(closed_counts),
                  "generators_including_empty": len(generators), "closed_union_rules": len(a_rules),
                  "closed_by_length": {int(k): int(v) for k, v in closed_table.groupby("length").size().items()},
                  "exhaustive_tests": 13, "full_data_counts_match_fp": True, "closure_verified": True}
write_json(OUT / "aclose_summary.json", aclose_summary)
display(pd.DataFrame([aclose_summary]).drop(columns="closed_by_length"))
display(closed_table.sort_values("support_count", ascending=False).head(8))
display(a_rules.head(8))

# Compare the benchmark prefix with the full sample using the same support threshold.
prefix = transactions[:1000]
prefix_counts, _, prefix_closed = local_a_close(prefix)
prefix_fp = counts_from_fp(local_fp_growth(prefix), len(prefix))
assert {item: count for item, count in prefix_counts.items() if item} == prefix_fp
assert prefix_closed == {item: count for item, count in prefix_fp.items()
                         if not any(item < other and count == support for other, support in prefix_fp.items())}
comparison = []
for documents, frequent, closed in ((len(prefix), len(prefix_fp), len(prefix_closed)),
                                    (n_tx, len(fp_counts), len(closed_counts))):
    comparison.append({"documents": documents, "frequent": frequent, "closed": closed,
                       "nonclosed": frequent - closed, "reduction": (frequent - closed) / frequent})

smaller = frozenset({"benefit of doubt", "murder"})
hits = [set(basket) for basket in prefix if smaller.issubset(basket)]
larger = frozenset(set.intersection(*hits))
assert smaller < larger and prefix_counts[smaller] == prefix_closed[larger] == len(hits)
closure_example = {"itemset": label_set(smaller), "closure": label_set(larger),
                   "documents": len(prefix), "support_count": len(hits), "support": len(hits) / len(prefix)}
write_json(OUT / "aclose_workload_comparison.json", {"workloads": comparison, "example": closure_example})
display(pd.DataFrame(comparison).style.format({"reduction": "{:.2%}"}))
display(pd.DataFrame([closure_example]).style.format({"support": "{:.1%}"}))

