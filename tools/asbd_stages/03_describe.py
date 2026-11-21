# 3. Descriptive analysis
# Use ten complementary plot types to describe case mix, chronology, PDF length and missingness. Identify the population behind each plot. Exclude the three disputed dates from chronology and use only readable documents for text-length figures.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#20354a",
        "xtick.color": "#334d62",
        "ytick.color": "#334d62",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "font.size": 10,
    }
)
BLUE, TEAL, GOLD = "#2f6597", "#15817b", "#bf781a"
artifacts = []


def chart():
    return plt.subplots(figsize=(8.2, 3.65), layout="constrained")


def save_chart(fig, filename, kind, title, basis, inference):
    path = FIGURES / filename
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)
    observation = {
        "file": filename,
        "type": kind,
        "title": title,
        "basis": basis,
        "inference": inference,
    }
    artifacts.append(observation)
    display(Image(filename=str(path)))
    display(Markdown(f"**{basis}** {inference}"))


categories = meta.case_category.value_counts()
basis_csv = f"Full CSV ({len(meta):,} records)."
basis_text = f"Readable PDFs in the seed-{SEED} sample ({len(readable):,} documents)."
chrono = meta.loc[meta.date_review.eq("Recorded")].copy()
basis_time = f"Recorded chronology ({len(chrono):,} rows after date review)."
annual = chrono.groupby("judgment_year").size()
annual = annual.reindex(
    range(int(annual.index.min()), int(annual.index.max()) + 1), fill_value=0
)

fig, ax = chart()
categories.head(8).sort_values().plot.barh(ax=ax, color=BLUE)
ax.set(title="Leading case categories", xlabel="Judgment records", ylabel="")
save_chart(
    fig,
    "01_case_categories.png",
    "Bar",
    "Leading case categories",
    basis_csv,
    f"{categories.index[0]} leads with {categories.iloc[0]:,} records ({categories.iloc[0]/len(meta):.1%}). "
    f"{categories['Other/Unknown']:,} case numbers remain unclassified.",
)

fig, ax = chart()
ax.plot(annual.index, annual, color=TEAL, linewidth=1.8)
ax.set(title="Records by judgment year", xlabel="Judgment year", ylabel="Records")
save_chart(
    fig,
    "02_yearly_records.png",
    "Line",
    "Records by judgment year",
    basis_time,
    f"The largest annual count is {annual.max():,} in {annual.idxmax()}. "
    f"The final recorded year, {annual.index[-1]}, has {annual.iloc[-1]:,} rows.",
)

fig, ax = chart()
ax.hist(
    readable.word_count,
    bins=np.geomspace(1, max(2, readable.word_count.max()) + 1, 36),
    color=BLUE,
)
ax.set(
    xscale="log",
    title="Extracted words per PDF",
    xlabel="Words (log scale)",
    ylabel="PDFs",
)
save_chart(
    fig,
    "03_word_histogram.png",
    "Histogram",
    "Extracted words per PDF",
    basis_text,
    f"The median is {readable.word_count.median():,.0f} words and the 90th percentile is "
    f"{readable.word_count.quantile(.9):,.0f}. {pdf_summary['truncated']:,} sampled PDFs reached the character cap.",
)

group_names = list(readable.case_category.value_counts().head(4).index)
pdf_groups = [readable.loc[readable.case_category.eq(group)] for group in group_names]
fig, ax = chart()
ax.boxplot(
    [g.page_count for g in pdf_groups],
    tick_labels=group_names,
    patch_artist=True,
    boxprops={"facecolor": "#a9c9e7"},
)
ax.set(yscale="log", title="PDF pages by case group", ylabel="Pages (log scale)")
save_chart(
    fig,
    "04_pages_box.png",
    "Box plot",
    "PDF pages by case group",
    basis_text,
    "Median page counts for the displayed groups are "
    + ", ".join(
        f"{name} {group.page_count.median():g} (n={len(group):,})"
        for name, group in zip(group_names, pdf_groups)
    )
    + ".",
)

fig, ax = chart()
log_lengths = [np.log10(g.word_count.to_numpy()) for g in pdf_groups]
parts = ax.violinplot(log_lengths, showmedians=True)
for body in parts["bodies"]:
    body.set_facecolor(TEAL)
    body.set_alpha(0.65)
ax.set(
    xticks=np.arange(1, len(group_names) + 1),
    xticklabels=group_names,
    title="Shape of word-count distributions",
    ylabel="log10 extracted words",
)
iqrs = {
    name: float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    for name, values in zip(group_names, log_lengths)
}
widest = max(iqrs, key=iqrs.get)
save_chart(
    fig,
    "05_words_violin.png",
    "Violin",
    "Shape of word-count distributions",
    basis_text,
    f"{widest} has the widest middle half of log-lengths among the displayed groups "
    f"(IQR {iqrs[widest]:.2f} log10 units).",
)

months = pd.crosstab(chrono.judgment_year, chrono.judgment_month).reindex(
    index=annual.index, columns=range(1, 13), fill_value=0
)
fig, ax = chart()
heat = ax.imshow(
    months.T,
    origin="lower",
    aspect="auto",
    cmap="YlGnBu",
    extent=[annual.index.min() - 0.5, annual.index.max() + 0.5, 0.5, 12.5],
)
ax.set(
    title="Month-by-year judgment counts",
    xlabel="Judgment year",
    ylabel="Month",
    yticks=[1, 3, 5, 7, 9, 12],
)
fig.colorbar(heat, ax=ax, label="Records")
peak_year, peak_month = months.stack().idxmax()
save_chart(
    fig,
    "06_month_heatmap.png",
    "Heatmap",
    "Month-by-year judgment counts",
    basis_time,
    f"The highest individual month-year cell is {peak_year}-{peak_month:02d} with "
    f"{months.loc[peak_year, peak_month]:,} records.",
)

fig, ax = chart()
ax.scatter(
    readable.page_count,
    readable.word_count,
    s=11,
    alpha=0.23,
    color=TEAL,
    edgecolors="none",
)
ax.set(
    xscale="log",
    yscale="log",
    title="Page count versus extracted words",
    xlabel="PDF pages (log scale)",
    ylabel="Extracted words (log scale)",
)
rho = readable[["page_count", "word_count"]].corr(method="spearman").iloc[0, 1]
save_chart(
    fig,
    "07_pages_and_words.png",
    "Scatter",
    "Page count versus extracted words",
    basis_text,
    f"Page count and extracted word count have Spearman rank correlation {rho:.3f}.",
)

fig, ax = chart()
cumulative = annual.cumsum()
ax.fill_between(annual.index, cumulative, color="#90b8d5")
ax.set(
    title="Cumulative records in the CSV",
    xlabel="Judgment year",
    ylabel="Cumulative records",
)
midpoint_year = int(cumulative.index[cumulative.ge(len(chrono) / 2)][0])
save_chart(
    fig,
    "08_cumulative_records.png",
    "Area",
    "Cumulative records in the CSV",
    basis_time,
    f"The cumulative total first reached half of the chronology records in {midpoint_year}.",
)

fig, ax = chart()
pet_counts = meta.petitioner_category.value_counts()
palette = {
    "Unclassified": "#d4e3ef",
    "Explicit government name": "#77b5af",
    "Missing": "#efc879",
}


def treemap(items, x, y, width, height):
    # Split the weighted rectangles along their longest side.
    if len(items) == 1:
        label, count = items[0]
        ax.add_patch(
            Rectangle(
                (x, y), width, height, facecolor=palette[label], ec="white", lw=1.5
            )
        )
        if width * height > 0.035:
            ax.text(
                x + width / 2,
                y + height / 2,
                f"{label}\n{count:,}",
                ha="center",
                va="center",
                color="#173651",
            )
        return
    first, rest = items[:1], items[1:]
    fraction = first[0][1] / sum(n for _, n in items)
    if width > height:
        treemap(first, x, y, width * fraction, height)
        treemap(rest, x + width * fraction, y, width * (1 - fraction), height)
    else:
        treemap(first, x, y, width, height * fraction)
        treemap(rest, x, y + height * fraction, width, height * (1 - fraction))


items = list(pet_counts.items())
treemap(items, 0, 0, 1, 1)
ax.set(xlim=(0, 1), ylim=(0, 1), title="Petitioner-name classification")
ax.set_axis_off()
ax.legend(
    [Rectangle((0, 0), 1, 1, facecolor=palette[label]) for label, _ in items],
    [f"{label} ({count:,})" for label, count in items],
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.11),
    frameon=False,
    fontsize=9,
)
save_chart(
    fig,
    "09_petitioner_treemap.png",
    "Treemap",
    "Petitioner-name classification",
    basis_csv,
    f"The strict name pattern flags {pet_counts['Explicit government name']:,} records "
    f"({pet_counts['Explicit government name']/len(meta):.1%}) and leaves "
    f"{pet_counts['Unclassified']:,} present names unclassified.",
)

fig, ax = chart()
impact = missing.loc[missing.gt(0)].sort_values(ascending=False)
pos = np.arange(len(impact))
ax.bar(pos, impact, color=BLUE)
ax.set(
    title="Concentration of missing metadata",
    ylabel="Missing cells",
    xticks=pos,
    xticklabels=impact.index,
)
ax.tick_params(axis="x", labelrotation=29)
twin = ax.twinx()
twin.plot(pos, impact.cumsum() / impact.sum(), "o-", color=GOLD, markersize=3)
twin.set(ylim=(0, 1.08), ylabel="Cumulative share")
twin.yaxis.set_major_formatter(PercentFormatter(1))
save_chart(
    fig,
    "10_missingness_pareto.png",
    "Pareto",
    "Concentration of missing metadata",
    basis_csv,
    f"{impact.index[0]} and {impact.index[1]} account for {impact.iloc[:2].sum()/impact.sum():.1%} "
    f"of the {impact.sum():,} missing cells across original fields.",
)

assert len({item["type"] for item in artifacts}) == 10
write_json(OUT / "descriptive_figures.json", artifacts.copy())
