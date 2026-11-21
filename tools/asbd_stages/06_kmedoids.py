# 6. K-Medoids on judgment text
# Use TF-IDF, truncated SVD and unit normalization. Partitioning Around Medoids searches one-for-one medoid swaps in cosine distance. Select k by the highest silhouette across k=2 through 5.
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

candidates = readable.loc[readable.cluster_text.ne("") & readable.word_count.ge(100)].copy()
candidates = candidates.loc[~candidates.pdf_filename.str.contains("supremecourt_vernacular", regex=False)].copy()
cluster_data = candidates.reset_index(drop=True)
stopwords = sorted(ENGLISH_STOP_WORDS | {
    "court", "high", "supreme", "judgment", "judgement", "justice", "learned", "said", "order",
    "hon", "ble", "versus", "case", "appellant", "respondent", "petitioner", "appeal", "page",
})
vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=3, max_df=.85, max_features=8500,
                            ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                            sublinear_tf=True, dtype=np.float32)
tfidf = vectorizer.fit_transform(cluster_data.cluster_text)
nonempty = np.asarray(tfidf.getnnz(axis=1)).ravel() > 0
cluster_data = cluster_data.loc[nonempty].reset_index(drop=True)
tfidf = tfidf[nonempty]
with threadpool_limits(limits=2):
    svd = TruncatedSVD(n_components=60, random_state=SEED)
    vectors = normalize(svd.fit_transform(tfidf))
    distance = cosine_distances(vectors).astype(np.float64)
distance = (distance + distance.T)/2
np.fill_diagonal(distance, 0)

def pam(distance, initial):
    medoids = np.array(initial, dtype=int)
    current = distance[:, medoids].min(axis=1).sum()
    while True:
        active = distance[:, medoids]
        outside = np.setdiff1d(np.arange(len(distance)), medoids)
        best_cost = current
        best_swap = None
        for j in range(len(medoids)):
            retained = np.delete(active, j, axis=1).min(axis=1)
            costs = np.minimum(retained[:, None], distance[:, outside]).sum(axis=0)
            choice = int(costs.argmin())
            if costs[choice] < best_cost - 1e-9:
                best_cost = float(costs[choice])
                best_swap = j, outside[choice]
        if best_swap is None:
            labels = distance[:, medoids].argmin(axis=1)
            labels[medoids] = np.arange(len(medoids))
            return medoids, labels, float(current)
        j, candidate = best_swap
        medoids[j] = candidate
        current = best_cost

simple = np.abs(np.array([0, 1, 9, 10])[:, None] - np.array([0, 1, 9, 10])[None, :])
assert pam(simple, [0, 1])[2] == 2

rng = np.random.default_rng(SEED)
trials = []
solutions = {}
for k in range(2, 6):
    fits = [pam(distance, rng.choice(len(cluster_data), size=k, replace=False)) for _ in range(3)]
    medoids, labels, loss = min(fits, key=lambda result: result[2])
    score = float(silhouette_score(distance, labels, metric="precomputed"))
    trials.append({"k": k, "silhouette": score, "total_cosine_distance": loss, "starts": len(fits)})
    solutions[k] = medoids, labels, loss

trial_table = pd.DataFrame(trials)
trial_table.to_csv(OUT / "kmedoids_trials.csv", index=False)
selected_k = int(trial_table.loc[trial_table.silhouette.idxmax(), "k"])
medoids, labels, loss = solutions[selected_k]
assert len(set(medoids)) == selected_k and np.array_equal(labels[medoids], np.arange(selected_k))
assert np.isclose(distance[np.arange(len(labels)), medoids[labels]].sum(), loss)

feature_names = vectorizer.get_feature_names_out()
cluster_rows = []
for cluster, medoid in enumerate(medoids):
    member = labels == cluster
    cohort_mean = np.asarray(tfidf[member].mean(axis=0)).ravel()
    outside_mean = np.asarray(tfidf[~member].mean(axis=0)).ravel()
    term_order = np.argsort(cohort_mean - outside_mean)[::-1][:8]
    cluster_rows.append({
        "cluster": cluster + 1, "size": int(member.sum()),
        "medoid_record": int(cluster_data.record_id.iloc[medoid]),
        "medoid_diary_no": str(cluster_data.diary_no.iloc[medoid]),
        "medoid_pdf": cluster_data.pdf_filename.iloc[medoid],
        "characteristic_terms": ", ".join(feature_names[term_order]),
    })
cluster_table = pd.DataFrame(cluster_rows)
cluster_table.to_csv(OUT / "kmedoids_clusters.csv", index=False)
with threadpool_limits(limits=2):
    projector = PCA(n_components=2)
    projection = projector.fit_transform(vectors)
cluster_data[["record_id", "diary_no", "pdf_filename", "case_category"]].assign(
    cluster=labels + 1, is_medoid=np.isin(np.arange(len(labels)), medoids),
    pc1=projection[:, 0], pc2=projection[:, 1]).to_csv(OUT / "kmedoids_assignments.csv", index=False)
best_score = float(trial_table.silhouette.max())
cluster_summary = {"sample": len(cluster_data), "selected_k": selected_k, "silhouette": best_score,
                   "distance": "cosine", "tfidf_features": tfidf.shape[1],
                   "svd_dimensions": vectors.shape[1], "svd_variance": float(svd.explained_variance_ratio_.sum()),
                   "plot_variance": float(projector.explained_variance_ratio_.sum()), "min_words": 100}
write_json(OUT / "kmedoids_summary.json", cluster_summary)
fig, ax = chart()
for cluster in range(selected_k):
    member = labels == cluster
    ax.scatter(*projection[member].T, s=13, alpha=.56, label=f"Cluster {cluster+1} (n={member.sum()})")
ax.scatter(*projection[medoids].T, marker="X", s=85, color="#172a3a", label="Medoid")
ax.set(title="K-Medoids clusters in two PCA coordinates", xlabel="PC 1", ylabel="PC 2")
ax.legend(frameon=False, fontsize=8)
save_chart(fig, "12_kmedoids_projection.png", "Cluster projection", "K-Medoids clusters",
           f"{len(cluster_data):,} documents. Distances use all {vectors.shape[1]} SVD coordinates.",
           f"k={selected_k} has the highest silhouette ({best_score:.3f}). The plot retains "
           f"{cluster_summary['plot_variance']:.1%} of variance in the normalized SVD features.")
write_json(OUT / "kmedoids_figure.json", artifacts[-1])
display(trial_table)
display(cluster_table.drop(columns="medoid_pdf"))

