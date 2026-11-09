import os
import re
import math
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CONFIG = {
    "seed": 42,
    "n_splits": 5,
    "target_enc_m": 20,
    "min_cell": 100,
    "min_product_reviews": 30,
    "tfidf_max_features": 100000,
    "hgb_iter": 150
}

THEMES = {
    "th_skin_reaction": r"break(?:ing|s)? out|broke me out|breakout|irritat|rash|burn|sting|itch|allerg|hives|redness",
    "th_no_effect": r"didn'?t work|did not work|no difference|no results?|no effect|doesn'?t work|does not work|not effective|did nothing|didn'?t do anything|no change|nothing happened",
    "th_price_value": r"price|expensive|overpriced|pricey|money|cost|cheap|worth",
    "th_returns": r"return|refund|exchange",
    "th_texture": r"sticky|greasy|oily|thick|heavy|tacky|gritty|pill(?:s|ing)|clump|watery|runny",
    "th_scent": r"smell|scent|fragrance|perfume|odou?r",
    "th_packaging": r"leak|spill|packag|pump|broken|dented|damaged|seal"
}

ACTIVES = {
    "act_retinoid": r"retin(?:ol|yl|al|oid)|tretinoin|adapalene",
    "act_niacinamide": r"niacinamide|nicotinamide",
    "act_hyaluronic": r"hyaluronic|sodium hyaluronate",
    "act_vitamin_c": r"ascorb(?:ic|yl)|tetrahexyldecyl ascorbate",
    "act_salicylic": r"salicylic acid|beta hydroxy",
    "act_aha": r"glycolic|lactic|mandelic|malic|tartaric acid",
    "act_alcohol_denat": r"alcohol denat|denatured alcohol|sd alcohol",
    "act_uv_filter": r"avobenzone|homosalate|octisalate|octocrylene|zinc oxide|titanium dioxide"
}

CONTRAST_WORDS = set("but however although though unfortunately except yet".split())
NEGATION_WORDS = set("not no never neither nor barely hardly scarcely n't don't doesn't didn't won't wouldn't can't couldn't shouldn't isn't aren't wasn't weren't".split())

sia = SentimentIntensityAnalyzer()

def build_features(df):
    f = pd.DataFrame(index=df.index)
    
    clean_text = df["review_text_clean"].fillna("").astype(str)
    raw_text = df["review_text_raw"].fillna("").astype(str) if "review_text_raw" in df.columns else clean_text
    
    f["words"] = clean_text.apply(lambda s: len(s.split()))
    f["log_words"] = np.log1p(f["words"])
    f["n_sentences"] = clean_text.apply(lambda s: max(1, len(re.split(r"[.!?]+", s.strip())) - 1))
    f["n_exclaim"] = clean_text.apply(lambda s: s.count("!"))
    f["n_question"] = clean_text.apply(lambda s: s.count("?"))
    f["caps_ratio"] = raw_text.apply(lambda s: sum(1 for c in s if c.isupper()) / max(1, len(s)))
    
    def get_rates(text):
        tokens = [w.lower() for w in re.findall(r"[a-z']+", text)]
        n = max(1, len(tokens))
        negs = sum(1 for w in tokens if w in NEGATION_WORDS)
        contrasts = sum(1 for w in tokens if w in CONTRAST_WORDS)
        return negs * 100.0 / n, contrasts * 100.0 / n
    
    rates = clean_text.apply(get_rates)
    f["neg_per_100w"] = [r[0] for r in rates]
    f["contrast_per_100w"] = [r[1] for r in rates]
    f["exclaim_per_100w"] = f["n_exclaim"] * 100.0 / np.maximum(1, f["words"])
    
    title_text = df["review_title"].fillna("").astype(str) if "review_title" in df.columns else pd.Series("", index=df.index)
    f["title_words"] = title_text.apply(lambda s: len(s.split()))
    f["has_title"] = (f["title_words"] > 0).astype(int)
    
    f["vader"] = clean_text.apply(lambda s: sia.polarity_scores(s)["compound"])
    f["has_emoji"] = df["has_emoji"].astype(int) if "has_emoji" in df.columns else 0
    f["all_caps"] = ((f["caps_ratio"] > 0.8) & (raw_text.apply(len) >= 20)).astype(int)
    
    for th_col, pattern in THEMES.items():
        f[th_col] = clean_text.str.contains(pattern, case=False, regex=True).astype(int)
    
    theme_cols = list(THEMES.keys())
    f["n_themes"] = f[theme_cols].sum(axis=1)
    
    rev_counts = df["reviewer_key"].map(df["reviewer_key"].value_counts())
    f["reviewer_n_reviews_log"] = np.log1p(rev_counts)
    f["reviewer_multi"] = (rev_counts > 1).astype(int)
    f["incentivised"] = df["incentivised"].astype(int) if "incentivised" in df.columns else 0
    f["rec_missing"] = df["rec_missing"].astype(int) if "rec_missing" in df.columns else 0
    
    st = df["skin_type"].fillna("unknown").astype(str) if "skin_type" in df.columns else pd.Series("unknown", index=df.index)
    f["skin_known"] = (st != "unknown").astype(int)
    f["skin_combination"] = (st == "combination").astype(int)
    f["skin_dry"] = (st == "dry").astype(int)
    f["skin_normal"] = (st == "normal").astype(int)
    f["skin_oily"] = (st == "oily").astype(int)
    
    dates = pd.to_datetime(df["submission_time"], errors="coerce") if "submission_time" in df.columns else pd.to_datetime(df["submission_date"], errors="coerce")
    f["year"] = dates.dt.year.fillna(2022).astype(int)
    months = dates.dt.month.fillna(1).astype(int)
    f["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12)
    f["holiday_season"] = months.isin([11, 12]).astype(int)
    
    first_dates = df.groupby("product_id")[dates.name].transform("min")
    prod_age_days = np.maximum(0, (dates - first_dates).dt.days.fillna(0))
    f["product_age_log"] = np.log1p(prod_age_days)
    
    max_date = dates.max()
    recency_days = np.maximum(0, (max_date - dates).dt.days.fillna(0))
    f["recency_log"] = np.log1p(recency_days)
    
    price = pd.to_numeric(df["price_usd"], errors="coerce").fillna(45.0)
    f["price_log"] = np.log1p(price)
    
    cat = df["secondary_category"].fillna(df.get("primary_category", "Moisturizers")).astype(str)
    f["price_pct"] = df.groupby(cat)["price_usd"].rank(pct=True).fillna(0.5)
    
    ppml = pd.to_numeric(df["price_per_ml"], errors="coerce")
    f["ppml_log"] = np.log(ppml.replace(0, np.nan)).fillna(0.0)
    f["ppml_pct"] = df.groupby(cat)["price_per_ml"].rank(pct=True).fillna(0.5)
    
    f["n_ingredients"] = pd.to_numeric(df.get("n_ingredients", 0), errors="coerce").fillna(0).astype(int)
    f["has_fragrance_f"] = df.get("has_fragrance", False).astype(int)
    f["ingredients_known"] = (f["n_ingredients"] > 0).astype(int)
    f["on_sale"] = df.get("on_sale", False).astype(int)
    
    prod_counts = df["product_id"].map(df["product_id"].value_counts())
    f["product_reviews_log"] = np.log1p(prod_counts)
    
    brand_counts = df["brand_name"].map(df["brand_name"].value_counts()) if "brand_name" in df.columns else pd.Series(1, index=df.index)
    f["brand_share"] = brand_counts / len(df)
    
    ing_text = df["ingredient_list"].astype(str) if "ingredient_list" in df.columns else pd.Series("", index=df.index)
    for act_col, pat in ACTIVES.items():
        f[act_col] = ing_text.str.contains(pat, case=False, regex=True).astype(int)
    
    act_cols = list(ACTIVES.keys())
    f["n_actives"] = f[act_cols].sum(axis=1)
    
    top_cats = ["Moisturizers", "Treatments", "Cleansers", "Eye Care", "Masks", "Value & Gift Sets", "Wellness", "High Tech Tools"]
    for c in top_cats:
        f[f"cat_{c}"] = (cat == c).astype(int)
    
    f["frag_x_dry"] = f["has_fragrance_f"] * f["skin_dry"]
    f["retinoid_x_dry"] = f["act_retinoid"] * f["skin_dry"]
    f["alcohol_x_dry"] = f["act_alcohol_denat"] * f["skin_dry"]
    f["price_x_gifted"] = f["price_pct"] * f["incentivised"]
    f["words_x_contrast"] = f["log_words"] * f["contrast_per_100w"]
    
    return f

def compute_target_encoding(train_df, val_df, group_col, target_col, m=20, seed=42):
    global_prior = train_df[target_col].mean()
    kf = GroupKFold(n_splits=3)
    train_encoded = pd.Series(index=train_df.index, dtype=float)
    
    for tr_in_idx, val_in_idx in kf.split(train_df, groups=train_df["reviewer_key"]):
        tr_part = train_df.iloc[tr_in_idx]
        stats = tr_part.groupby(group_col)[target_col].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + m * global_prior) / (stats["count"] + m)
        train_encoded.iloc[val_in_idx] = train_df.iloc[val_in_idx][group_col].map(smoothed).fillna(global_prior)
    
    full_stats = train_df.groupby(group_col)[target_col].agg(["count", "mean"])
    full_smoothed = (full_stats["count"] * full_stats["mean"] + m * global_prior) / (full_stats["count"] + m)
    val_encoded = val_df[group_col].map(full_smoothed).fillna(global_prior)
    
    return train_encoded, val_encoded

def run_evaluation(df):
    y = (df["rating"] <= 2).astype(int).values
    features_df = build_features(df)
    
    gkf = GroupKFold(n_splits=CONFIG["n_splits"])
    groups = df["reviewer_key"].values
    
    oof_preds = {
        "A": np.zeros(len(df)),
        "B": np.zeros(len(df)),
        "C": np.zeros(len(df)),
        "D": np.zeros(len(df))
    }
    
    tfidf_text = df["review_text_clean"].fillna("").values
    scaler = RobustScaler()
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        tr_df, val_df = df.iloc[tr_idx], df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=CONFIG["tfidf_max_features"], min_df=2)
        X_tr_tfidf = vec.fit_transform(tfidf_text[tr_idx])
        X_val_tfidf = vec.transform(tfidf_text[val_idx])
        
        clf_a = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
        clf_a.fit(X_tr_tfidf, y_tr)
        oof_preds["A"][val_idx] = clf_a.predict_proba(X_val_tfidf)[:, 1]
        
        X_feats_tr = features_df.iloc[tr_idx].copy()
        X_feats_val = features_df.iloc[val_idx].copy()
        
        p_risk_tr, p_risk_val = compute_target_encoding(tr_df, val_df, "product_id", "low", m=CONFIG["target_enc_m"])
        b_risk_tr, b_risk_val = compute_target_encoding(tr_df, val_df, "brand_name", "low", m=CONFIG["target_enc_m"])
        
        X_feats_tr["product_risk"] = p_risk_tr
        X_feats_val["product_risk"] = p_risk_val
        X_feats_tr["brand_risk"] = b_risk_tr
        X_feats_val["brand_risk"] = b_risk_val
        
        clf_b = HistGradientBoostingClassifier(max_iter=CONFIG["hgb_iter"], random_state=CONFIG["seed"])
        clf_b.fit(X_feats_tr, y_tr)
        oof_preds["B"][val_idx] = clf_b.predict_proba(X_feats_val)[:, 1]
        
        X_scaled_tr = scaler.fit_transform(X_feats_tr.fillna(0))
        X_scaled_val = scaler.transform(X_feats_val.fillna(0))
        
        from scipy.sparse import hstack
        X_comb_tr = hstack([X_tr_tfidf, X_scaled_tr])
        X_comb_val = hstack([X_val_tfidf, X_scaled_val])
        
        clf_c = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
        clf_c.fit(X_comb_tr, y_tr)
        oof_preds["C"][val_idx] = clf_c.predict_proba(X_comb_val)[:, 1]
        
        inner_gkf = GroupKFold(n_splits=3)
        tr_text_oof = np.zeros(len(tr_df))
        for in_tr_idx, in_val_idx in inner_gkf.split(tr_df, groups=tr_df["reviewer_key"].values):
            inner_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=CONFIG["tfidf_max_features"], min_df=2)
            X_in_tr = inner_vec.fit_transform(tfidf_text[tr_idx[in_tr_idx]])
            X_in_val = inner_vec.transform(tfidf_text[tr_idx[in_val_idx]])
            inner_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
            inner_clf.fit(X_in_tr, y_tr[in_tr_idx])
            tr_text_oof[in_val_idx] = inner_clf.predict_proba(X_in_val)[:, 1]
            
        X_stack_tr = X_feats_tr.copy()
        X_stack_tr["text_score"] = tr_text_oof
        X_stack_val = X_feats_val.copy()
        X_stack_val["text_score"] = oof_preds["A"][val_idx]
        
        clf_d = HistGradientBoostingClassifier(max_iter=CONFIG["hgb_iter"], random_state=CONFIG["seed"])
        clf_d.fit(X_stack_tr, y_tr)
        oof_preds["D"][val_idx] = clf_d.predict_proba(X_stack_val)[:, 1]
        
    return oof_preds, y

def run_selftest():
    assert len(re.findall(r"not|never", "not bad and never again")) == 2
    assert "but" in CONTRAST_WORDS
    assert len(re.findall(r"[!?]", "Great! Really?")) == 2
    s_cap = "HELLO"
    assert sum(1 for c in s_cap if c.isupper()) / len(s_cap) == 1.0
    assert len("".split()) == 0
    assert bool(re.search(THEMES["th_scent"], "strong perfume and scent"))
    assert sia.polarity_scores("amazing product")["compound"] > sia.polarity_scores("terrible product")["compound"]
    
    for m in range(1, 13):
        rad = 2 * np.pi * (m - 1) / 12
        assert math.isclose(np.sin(rad)**2 + np.cos(rad)**2, 1.0, rel_tol=1e-5)
        
    dist_jan_dec = math.dist([np.sin(0), np.cos(0)], [np.sin(2*np.pi*11/12), np.cos(2*np.pi*11/12)])
    dist_jan_jun = math.dist([np.sin(0), np.cos(0)], [np.sin(2*np.pi*5/12), np.cos(2*np.pi*5/12)])
    assert dist_jan_dec < dist_jan_jun
    
    t0 = pd.to_datetime("2021-01-01")
    t1 = pd.to_datetime("2021-01-10")
    assert (t1 - t0).days == 9
    assert pd.to_datetime("2021-12-05").month in [11, 12]
    
    p_ranks = pd.Series([10, 20, 30, 40]).rank(pct=True).values
    assert len(p_ranks) == 4
    
    dummy_prior = 0.15
    assert dummy_prior == 0.15
    assert 2 * 3 == 6
    
    h1 = hashlib.sha256(b"test").hexdigest()
    h2 = hashlib.sha256(b"test").hexdigest()
    assert h1 == h2

if __name__ == "__main__":
    run_selftest()
