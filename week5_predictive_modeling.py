import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def generate_synthetic_skincare_corpus(n_samples=6000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n_reviewers = 1500
    reviewer_keys = rng.integers(10000, 10000 + n_reviewers, size=n_samples)
    reviewer_bias = {rk: rng.normal(0.0, 0.35) for rk in np.unique(reviewer_keys)}
    user_offset = np.array([reviewer_bias[rk] for rk in reviewer_keys])

    subcategories = rng.choice(
        ["Eye Care", "Facial Masks", "Serums", "Moisturizers", "Cleansers"],
        size=n_samples,
        p=[0.14, 0.16, 0.28, 0.25, 0.17],
    )
    price_quartile = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])
    incentivized = rng.binomial(1, 0.14, size=n_samples)
    fragrance_present = rng.binomial(1, 0.42, size=n_samples)
    days_since_launch = rng.integers(1, 900, size=n_samples)
    log_word_count = rng.normal(3.8, 0.65, size=n_samples).clip(1.0, 6.2)
    negation_per_100w = rng.exponential(2.2, size=n_samples).clip(0.0, 18.0)
    contrast_per_100w = rng.exponential(1.4, size=n_samples).clip(0.0, 12.0)
    vader_compound = rng.normal(0.35, 0.48, size=n_samples).clip(-1.0, 1.0)
    th_skin_reaction = rng.binomial(1, 0.11, size=n_samples)
    th_packaging_defect = rng.binomial(1, 0.07, size=n_samples)
    th_no_efficacy = rng.binomial(1, 0.15, size=n_samples)

    eye_care_flag = (subcategories == "Eye Care").astype(int)
    mask_flag = (subcategories == "Facial Masks").astype(int)
    q1_price_flag = (price_quartile == 1).astype(int)
    q4_price_flag = (price_quartile == 4).astype(int)

    logit = (
        -2.15
        + 0.42 * eye_care_flag
        + 0.25 * mask_flag
        + 0.28 * q1_price_flag
        - 1.65 * incentivized
        - 1.85 * vader_compound
        + 0.11 * negation_per_100w
        + 0.16 * contrast_per_100w
        + 1.35 * th_skin_reaction
        + 0.95 * th_packaging_defect
        + 0.85 * th_no_efficacy
        + 0.55 * (eye_care_flag * fragrance_present)
        + 0.48 * (q4_price_flag * th_no_efficacy)
        + user_offset
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target_low_rating = rng.binomial(1, prob)

    positive_phrases = [
        "holy grail skincare staple cleared my skin",
        "gentle hydrating formula no breakout at all",
        "absorbs smoothly worth every rupee great glow",
        "never irritated my sensitive skin repurchase again",
        "lightweight texture calm barrier zero redness",
    ]
    negative_phrases = [
        "complete waste of money broke me out badly",
        "stung my eyes and burned my skin barrier",
        "arrived broken pump defect terrible packaging",
        "wanted to love this however zero visible efficacy",
        "strong fragrance caused redness and dry patches",
    ]

    texts = []
    for idx in range(n_samples):
        if target_low_rating[idx] == 1:
            base = negative_phrases[rng.integers(0, len(negative_phrases))]
            extra = positive_phrases[rng.integers(0, len(positive_phrases))] if rng.random() < 0.2 else ""
        else:
            base = positive_phrases[rng.integers(0, len(positive_phrases))]
            extra = negative_phrases[rng.integers(0, len(negative_phrases))] if rng.random() < 0.1 else ""
        texts.append(f"{base} {subcategories[idx].lower()} {extra}".strip())

    df = pd.DataFrame({
        "reviewer_key": reviewer_keys,
        "review_text": texts,
        "subcategory": subcategories,
        "eye_care_flag": eye_care_flag,
        "mask_flag": mask_flag,
        "price_quartile": price_quartile,
        "q1_price_flag": q1_price_flag,
        "q4_price_flag": q4_price_flag,
        "incentivized": incentivized,
        "fragrance_present": fragrance_present,
        "days_since_launch": days_since_launch,
        "log_word_count": log_word_count,
        "negation_per_100w": negation_per_100w,
        "contrast_per_100w": contrast_per_100w,
        "vader_compound": vader_compound,
        "th_skin_reaction": th_skin_reaction,
        "th_packaging_defect": th_packaging_defect,
        "th_no_efficacy": th_no_efficacy,
        "eye_x_fragrance": eye_care_flag * fragrance_present,
        "q4_x_no_efficacy": q4_price_flag * th_no_efficacy,
        "target_low_rating": target_low_rating,
    })
    return df


def compute_oof_brand_target_encoding(train_df, val_df, col, target_col, smoothing=20.0):
    prior = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)
    train_encoded = train_df[col].map(smoothed).fillna(prior).to_numpy()
    val_encoded = val_df[col].map(smoothed).fillna(prior).to_numpy()
    return train_encoded, val_encoded


def build_text_baseline_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=15000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
    ])


def build_dense_lr_pipeline(numeric_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
    ])


def build_hgb_pipeline():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
    )


def evaluate_predictions(y_true, y_prob, threshold=0.44):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def run_week5_modeling_benchmark(df):
    dense_cols = [
        "eye_care_flag",
        "mask_flag",
        "price_quartile",
        "q1_price_flag",
        "q4_price_flag",
        "incentivized",
        "fragrance_present",
        "days_since_launch",
        "log_word_count",
        "negation_per_100w",
        "contrast_per_100w",
        "vader_compound",
        "th_skin_reaction",
        "th_packaging_defect",
        "th_no_efficacy",
        "eye_x_fragrance",
        "q4_x_no_efficacy",
    ]

    y = df["target_low_rating"].to_numpy()
    groups = df["reviewer_key"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    oof_dense_lr = np.zeros(len(df))
    oof_dense_hgb = np.zeros(len(df))
    oof_text_lr = np.zeros(len(df))
    oof_stacked_hgb = np.zeros(len(df))

    for train_idx, val_idx in gkf.split(df, y, groups=groups):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        tr_sub_te, va_sub_te = compute_oof_brand_target_encoding(
            train_df, val_df, "subcategory", "target_low_rating", smoothing=20.0
        )
        train_df["subcat_target_enc"] = tr_sub_te
        val_df["subcat_target_enc"] = va_sub_te

        fold_dense_cols = dense_cols + ["subcat_target_enc"]
        X_tr_dense = train_df[fold_dense_cols]
        X_va_dense = val_df[fold_dense_cols]
        y_tr = train_df["target_low_rating"].to_numpy()
        y_va = val_df["target_low_rating"].to_numpy()

        model_dense_lr = build_dense_lr_pipeline(fold_dense_cols)
        model_dense_lr.fit(X_tr_dense, y_tr)
        oof_dense_lr[val_idx] = model_dense_lr.predict_proba(X_va_dense)[:, 1]

        model_dense_hgb = build_hgb_pipeline()
        model_dense_hgb.fit(X_tr_dense, y_tr)
        oof_dense_hgb[val_idx] = model_dense_hgb.predict_proba(X_va_dense)[:, 1]

        model_text_lr = build_text_baseline_pipeline()
        model_text_lr.fit(train_df["review_text"], y_tr)
        oof_text_lr[val_idx] = model_text_lr.predict_proba(val_df["review_text"])[:, 1]

        inner_gkf = GroupKFold(n_splits=4)
        inner_oof_text = np.zeros(len(train_df))
        tr_groups = train_df["reviewer_key"].to_numpy()
        for in_tr_idx, in_va_idx in inner_gkf.split(train_df, y_tr, groups=tr_groups):
            inner_text_model = clone(build_text_baseline_pipeline())
            inner_text_model.fit(train_df["review_text"].iloc[in_tr_idx], y_tr[in_tr_idx])
            inner_oof_text[in_va_idx] = inner_text_model.predict_proba(
                train_df["review_text"].iloc[in_va_idx]
            )[:, 1]

        X_tr_stack = X_tr_dense.copy()
        X_tr_stack["oof_text_prob"] = inner_oof_text
        X_va_stack = X_va_dense.copy()
        X_va_stack["oof_text_prob"] = oof_text_lr[val_idx]

        meta_hgb = build_hgb_pipeline()
        meta_hgb.fit(X_tr_stack, y_tr)
        oof_stacked_hgb[val_idx] = meta_hgb.predict_proba(X_va_stack)[:, 1]

    results = {
        "Model_A_Dense_LogisticRegression": evaluate_predictions(y, oof_dense_lr, threshold=0.50),
        "Model_B_Dense_HistGradientBoosting": evaluate_predictions(y, oof_dense_hgb, threshold=0.44),
        "Model_C_Text_TFIDF_LogisticRegression": evaluate_predictions(y, oof_text_lr, threshold=0.50),
        "Model_D_Stacked_Hybrid_HistGradientBoosting": evaluate_predictions(y, oof_stacked_hgb, threshold=0.44),
    }
    return results, oof_stacked_hgb


def run_subgroup_fairness_audit(df, oof_probs, threshold=0.44):
    y = df["target_low_rating"].to_numpy()
    subgroups = {
        "Eye_Care": df["eye_care_flag"] == 1,
        "Facial_Masks": df["mask_flag"] == 1,
        "Budget_Quartile_Q1": df["q1_price_flag"] == 1,
        "Luxury_Quartile_Q4": df["q4_price_flag"] == 1,
        "Organic_Buyers": df["incentivized"] == 0,
        "Incentivized_Gifted": df["incentivized"] == 1,
    }
    audit = {}
    for name, mask in subgroups.items():
        idx = mask.to_numpy()
        y_sub = y[idx]
        p_sub = oof_probs[idx]
        y_hat = (p_sub >= threshold).astype(int)
        audit[name] = {
            "count": int(idx.sum()),
            "base_dissatisfaction_rate": round(float(y_sub.mean()), 4),
            "pr_auc": round(float(average_precision_score(y_sub, p_sub)), 4),
            "recall_at_threshold": round(float(recall_score(y_sub, y_hat, zero_division=0)), 4),
        }
    return audit


def run_self_tests(df, results):
    assert len(df) == 6000
    assert "target_low_rating" in df.columns
    assert set(df["target_low_rating"].unique()).issubset({0, 1})
    for model_name, metrics in results.items():
        assert 0.0 <= metrics["pr_auc"] <= 1.0, f"Invalid PR-AUC for {model_name}"
        assert 0.5 <= metrics["roc_auc"] <= 1.0, f"Invalid ROC-AUC for {model_name}"
        assert 0.0 <= metrics["brier_score"] <= 1.0, f"Invalid Brier for {model_name}"


if __name__ == "__main__":
    corpus_df = generate_synthetic_skincare_corpus()
    benchmark_results, stacked_probs = run_week5_modeling_benchmark(corpus_df)
    fairness_audit = run_subgroup_fairness_audit(corpus_df, stacked_probs)
    run_self_tests(corpus_df, benchmark_results)

    manifest = {
        "pipeline_stage": "Week 5 Predictive Modeling & Algorithm Selection",
        "random_seed": RANDOM_SEED,
        "corpus_rows": len(corpus_df),
        "positive_rate": round(float(corpus_df["target_low_rating"].mean()), 4),
        "dataset_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(corpus_df, index=True).values.tobytes()
        ).hexdigest(),
        "benchmark_metrics": benchmark_results,
        "subgroup_fairness_audit": fairness_audit,
    }
    print(json.dumps(manifest, indent=2))
