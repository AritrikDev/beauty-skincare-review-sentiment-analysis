
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "sephora-analysis/datasets"
reviews = pd.read_csv(f"{BASE}/skincare_products_reviews.csv", index_col=0, low_memory=False)
products = pd.read_csv(f"{BASE}/product_info.csv", low_memory=False)
print("1. SAMPLE PROFILE")
print("rows:", len(reviews), "| columns:", reviews.shape[1], "| products in product file:", len(products))
print("unique products:", reviews.product_id.nunique(),
      "| brands:", reviews.brand_name.nunique(),
      "| authors:", reviews.author_id.nunique())
print("date range:", reviews.submission_time.min(), "to", reviews.submission_time.max())
print("missing % per column:")
print((reviews.isna().mean() * 100).round(2).to_string())
n_raw = len(reviews)
reviews = reviews.drop_duplicates(["author_id", "product_id", "review_text"])
print("\n2. CLEANING")
print("duplicate (author, product, text) rows removed:", n_raw - len(reviews))
reviews = reviews[reviews.review_text.notna()].copy()
print("rows after removing missing review text:", len(reviews))
reviews["words"] = reviews.review_text.str.split().str.len()
print("reviews with 3 words or fewer:", int((reviews.words <= 3).sum()))
print("\n3. DESCRIPTIVE STATISTICS")
print("rating share % :", (reviews.rating.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print("mean rating:", round(reviews.rating.mean(), 2))
print("words: mean", round(reviews.words.mean(), 1), "median", reviews.words.median(),
      "Q1", reviews.words.quantile(.25), "Q3", reviews.words.quantile(.75), "max", reviews.words.max())
print("median words by rating:", reviews.groupby("rating").words.median().to_dict())
print("% recommended by rating:", (reviews.groupby("rating").is_recommended.mean() * 100).round(1).to_dict())
per_product = reviews.groupby("product_id").size()
print("reviews per product: median", per_product.median(), "max", per_product.max())
brand_share = reviews.groupby("brand_name").size().sort_values(ascending=False) / len(reviews) * 100
print("top brand:", brand_share.index[0], round(brand_share.iloc[0], 1), "% | top 5 brands:", round(brand_share.head(5).sum(), 1), "%")
def rating_label(x):
    return "negative" if x <= 2 else ("neutral" if x == 3 else "positive")
reviews["rating_label"] = reviews.rating.map(rating_label)
print("\n4. RATING-BASED LABELS %:", (reviews.rating_label.value_counts(normalize=True) * 100).round(1).to_dict())
sia = SentimentIntensityAnalyzer()
reviews["vader"] = reviews.review_text.map(lambda t: sia.polarity_scores(t)["compound"])
reviews["vader_label"] = reviews.vader.map(lambda c: "negative" if c <= -0.05 else ("positive" if c >= 0.05 else "neutral"))
order = ["negative", "neutral", "positive"]
ct = pd.crosstab(reviews.rating_label, reviews.vader_label).reindex(index=order, columns=order).fillna(0)
print("\n5. RATING LABEL (rows) vs VADER LABEL (columns), row %")
print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())
agree = sum(ct.values[i][i] for i in range(3)) / ct.values.sum() * 100
rho, _ = stats.spearmanr(reviews.rating, reviews.vader)
chi2, p_chi, _, _ = stats.chi2_contingency(ct.values)
cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
print("agreement %:", round(agree, 1), "| Cohen kappa:", round(cohen_kappa_score(reviews.rating_label, reviews.vader_label), 3),
      "| Spearman rho:", round(rho, 3), "| Cramer V:", round(cramers_v, 3))
hi, lo = reviews[reviews.rating >= 4], reviews[reviews.rating <= 2]
print("4-5 star reviews scored negative by VADER: %.1f%% (n=%d of %d)" % ((hi.vader_label == "negative").mean() * 100, (hi.vader_label == "negative").sum(), len(hi)))
print("1-2 star reviews scored positive by VADER: %.1f%% (n=%d of %d)" % ((lo.vader_label == "positive").mean() * 100, (lo.vader_label == "positive").sum(), len(lo)))
cats = products[["product_id", "primary_category", "secondary_category"]].drop_duplicates("product_id")
reviews = reviews.merge(cats, on="product_id", how="left")
top = reviews.secondary_category.value_counts().head(6).index.tolist()
sub = reviews[reviews.secondary_category.isin(top)]
print("\n6. CATEGORY COMPARISON (top 6 secondary categories)")
for c in top:
    s = sub[sub.secondary_category == c]
    n, neg = len(s), (s.rating <= 2).sum()
    ph, z = neg / n, 1.96
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    print(f"{c:20s} n={n:6d} mean={s.rating.mean():.2f}  1-2 star={ph*100:.1f}% (95% CI {100*(centre-half):.1f}-{100*(centre+half):.1f})")
H, p_kw = stats.kruskal(*[sub[sub.secondary_category == c].rating.values for c in top])
ct2 = pd.crosstab(sub.secondary_category, sub.rating_label)
chi2b, _, _, _ = stats.chi2_contingency(ct2.values)
print("Kruskal-Wallis H=%.1f (epsilon squared=%.4f) | chi-square Cramer V=%.3f" % (H, H / (len(sub) - 1), np.sqrt(chi2b / (ct2.values.sum() * (min(ct2.shape) - 1)))))
stop = set("""the a an and or but if of to in on for with is it this that i my me you your was were are be been am as at by from so have has had not no just very really more most can will would could do does did they them their there these those its it's i'm i've don't didn't doesn't than then too also one use used using product products skin after when what which about out up get got like love""".split())
tokens = lambda s: [w for w in re.findall(r"[a-z']+", s.lower()) if w not in stop and len(w) > 2]
c_lo, c_hi = Counter(), Counter()
for t in lo.review_text: c_lo.update(set(tokens(t)))
for t in hi.review_text: c_hi.update(set(tokens(t)))
rows = []
for w in set(c_lo) | set(c_hi):
    a, b = c_lo[w], c_hi[w]
    if a + b < 150:
        continue
    pl, ph_ = (a + .5) / (len(lo) + 1), (b + .5) / (len(hi) + 1)
    rows.append((w, np.log((pl / (1 - pl)) / (ph_ / (1 - ph_))), a / len(lo) * 100, b / len(hi) * 100))
rows.sort(key=lambda x: -x[1])
print("\n7. WORDS MORE COMMON IN 1-2 STAR REVIEWS (word, % of 1-2 star reviews, % of 4-5 star reviews)")
for w, _, pl, ph_ in rows[:15]:
    print(f"{w:15s} {pl:5.1f}  {ph_:5.1f}")
rp = (reviews.rating.value_counts(normalize=True).sort_index() * 100)
wm = reviews.groupby("rating").words.median()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(rp.index.astype(str), rp.values, color="black")
for i, v in enumerate(rp.values): ax[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
ax[0].set(xlabel="Star rating", ylabel="Share of reviews (%)", title="Rating distribution", ylim=(0, 75))
ax[1].bar(wm.index.astype(str), wm.values, color="dimgray")
for i, v in enumerate(wm.values): ax[1].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=8)
ax[1].set(xlabel="Star rating", ylabel="Median review length (words)", title="Review length by rating")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("pilot_figure.png", dpi=200)
print("\nfigure saved: pilot_figure.png")
reviews["year"] = pd.to_datetime(reviews.submission_time).dt.year
rec = reviews.sort_values("submission_time").iloc[len(reviews) // 2]
print("\n8. ONE REAL RECORD (median date)")
for k in ["rating", "is_recommended", "submission_time", "skin_type", "product_name", "brand_name", "price_usd", "words"]:
    print(f"{k:16s}", rec[k])
print("review text:", rec.review_text)
yearly = reviews.groupby("year").agg(n=("rating", "size"), mean=("rating", "mean"),
                                     low=("rating", lambda x: (x <= 2).mean() * 100)).round(2)
print("\n9. REVIEWS, MEAN RATING AND 1-2 STAR SHARE BY YEAR (2023 stops on", reviews.submission_time.max(), ")")
print(yearly.to_string())
g = reviews.groupby(["product_id", "product_name", "brand_name"]).agg(
    n=("rating", "size"), mean=("rating", "mean"),
    low=("rating", lambda x: (x <= 2).mean() * 100), price=("price_usd", "first")).reset_index()
g = g[g.n >= 100]
print("\n10. PRODUCTS WITH 100+ REVIEWS IN THE SAMPLE:", len(g))
print("highest mean rating:"); print(g.sort_values(["mean", "n"], ascending=[False, False]).head(6).round(2).to_string(index=False))
print("lowest mean rating:"); print(g.sort_values(["mean", "n"], ascending=[True, False]).head(6).round(2).to_string(index=False))
print("Spearman correlation, price vs mean rating (these products):", round(g.price.corr(g["mean"], method="spearman"), 3))
dup_any = reviews.duplicated(["author_id", "review_text"], keep=False)
per_review_ids = reviews[dup_any].groupby(["author_id", "review_text"]).product_id.nunique()
print("\nCROSS-PRODUCT DUPLICATES: rows =", int(dup_any.sum()), "(%.1f%%)" % (dup_any.mean() * 100),
      "| distinct reviews =", len(per_review_ids), "| max product IDs per review =", int(per_review_ids.max()))
dedup = reviews.drop_duplicates(["author_id", "review_text"])
print("rows after keeping one copy:", len(dedup), "| mean rating before/after:", round(reviews.rating.mean(), 3), round(dedup.rating.mean(), 3))
themes = {
    "Skin reaction": r"break(?:ing|s)? out|broke me out|breakout|irritat|rash|burn|sting|itch|allerg|hives|redness",
    "No visible effect": r"didn'?t work|did not work|no difference|no results?|no effect|doesn'?t work|does not work|not effective|did nothing|didn'?t do anything|no change|nothing happened",
    "Price and value": r"price|expensive|overpriced|pricey|money|cost|cheap|worth",
    "Returns and refunds": r"return|refund|exchange",
    "Texture": r"sticky|greasy|oily|thick|heavy|tacky|gritty|pill(?:s|ing)|clump|watery|runny",
    "Scent": r"smell|scent|fragrance|perfume|odou?r",
    "Packaging and damage": r"leak|spill|packag|pump|broken|dented|damaged|seal",
}
print("\n11. KEYWORD THEMES: % of reviews mentioning the theme")
print(f"{'theme':24s}{'1-2 star':>10s}{'4-5 star':>10s}{'ratio':>8s}")
for name, rx in themes.items():
    a = lo.review_text.str.contains(rx, case=False, regex=True).mean() * 100
    b = hi.review_text.str.contains(rx, case=False, regex=True).mean() * 100
    print(f"{name:24s}{a:10.1f}{b:10.1f}{a / b:8.1f}")
cand = reviews[reviews.words.between(20, 80)]
print("\n12. 1-STAR REVIEWS WITH THE MOST POSITIVE VADER SCORE (20-80 words)")
for _, x in cand[cand.rating == 1].sort_values(["vader", "author_id"], ascending=[False, True]).head(4).iterrows():
    print(f"[{x.rating} star, VADER {x.vader:.2f}] {x.brand_name} | {x.product_name}\n    {x.review_text}\n")
print("5-STAR REVIEWS WITH THE MOST NEGATIVE VADER SCORE (20-80 words)")
for _, x in cand[cand.rating == 5].sort_values(["vader", "author_id"]).head(4).iterrows():
    print(f"[{x.rating} star, VADER {x.vader:.2f}] {x.brand_name} | {x.product_name}\n    {x.review_text}\n")
y2 = yearly.loc[2015:]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].bar(y2.index.astype(str), y2.n.values, color="black")
for i, v in enumerate(y2.n.values): ax[0].text(i, v + 150, f"{v:,}", ha="center", fontsize=7)
ax[0].set(xlabel="Year (2023 ends in March)", ylabel="Reviews in sample", title="Reviews per year")
ax[1].plot(y2.index.astype(str), y2.low.values, color="black", marker="o")
for i, v in enumerate(y2.low.values): ax[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)
ax[1].set(xlabel="Year (2023 ends in March)", ylabel="1-2 star share (%)", title="Share of 1-2 star reviews", ylim=(0, 24))
for a in ax:
    a.spines[["top", "right"]].set_visible(False); a.tick_params(axis="x", labelsize=7)
plt.tight_layout(); plt.savefig("pilot_figure2.png", dpi=200)
print("figure saved: pilot_figure2.png")
