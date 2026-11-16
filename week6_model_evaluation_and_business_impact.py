import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def generate_evaluation_corpus(n_samples=6000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n_reviewers = 1500
    reviewer_keys = rng.integers(10000, 10000 + n_reviewers, size=n_samples)
    reviewer_bias = {rk: rng.normal(0.0, 0.35) for rk in np.unique(reviewer_keys)}
    user_offset = np.array([reviewer_bias[rk] for rk in reviewer_keys])

    subcategories = rng.choice(
        ["Eye Care", "Facial Masks", "Anti-Aging Serums", "Moisturizers", "Cleansers"],
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

    return pd.DataFrame({
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


def compute_smoothed_target_encoding(train_df, val_df, col, target_col, smoothing=20.0):
    prior = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)
    return (
        train_df[col].map(smoothed).fillna(prior).to_numpy(),
        val_df[col].map(smoothed).fillna(prior).to_numpy(),
    )


def build_text_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=15000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
    ])


def build_meta_hgb_model():
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


def fit_and_evaluate_stacked_pipeline(df):
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

    oof_probs = np.zeros(len(df))
    oof_organic_counterfactual = np.zeros(len(df))
    fold_metrics = []
    last_fold_artifacts = None

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups=groups), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        tr_te, va_te = compute_smoothed_target_encoding(
            train_df, val_df, "subcategory", "target_low_rating", smoothing=20.0
        )
        train_df["subcat_target_enc"] = tr_te
        val_df["subcat_target_enc"] = va_te

        feature_cols = dense_cols + ["subcat_target_enc"]
        y_tr = train_df["target_low_rating"].to_numpy()
        y_va = val_df["target_low_rating"].to_numpy()

        text_model = build_text_model()
        text_model.fit(train_df["review_text"], y_tr)
        val_text_prob = text_model.predict_proba(val_df["review_text"])[:, 1]

        inner_gkf = GroupKFold(n_splits=4)
        inner_oof_text = np.zeros(len(train_df))
        tr_groups = train_df["reviewer_key"].to_numpy()
        for in_tr, in_va in inner_gkf.split(train_df, y_tr, groups=tr_groups):
            in_clf = clone(build_text_model())
            in_clf.fit(train_df["review_text"].iloc[in_tr], y_tr[in_tr])
            inner_oof_text[in_va] = in_clf.predict_proba(train_df["review_text"].iloc[in_va])[:, 1]

        X_tr_stack = train_df[feature_cols].copy()
        X_tr_stack["oof_text_prob"] = inner_oof_text
        X_va_stack = val_df[feature_cols].copy()
        X_va_stack["oof_text_prob"] = val_text_prob

        meta_model = build_meta_hgb_model()
        meta_model.fit(X_tr_stack, y_tr)
        val_preds = meta_model.predict_proba(X_va_stack)[:, 1]
        oof_probs[val_idx] = val_preds

        X_va_organic = X_va_stack.copy()
        X_va_organic["incentivized"] = 0
        oof_organic_counterfactual[val_idx] = meta_model.predict_proba(X_va_organic)[:, 1]

        fold_metrics.append({
            "fold": fold_id,
            "pr_auc": round(float(average_precision_score(y_va, val_preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_va, val_preds)), 4),
            "brier_score": round(float(brier_score_loss(y_va, val_preds)), 4),
        })
        last_fold_artifacts = (meta_model, X_va_stack, y_va)

    return oof_probs, oof_organic_counterfactual, fold_metrics, last_fold_artifacts


def evaluate_operating_thresholds(y_true, y_prob, thresholds=(0.28, 0.44, 0.68)):
    summary = {}
    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        net_roi_per_10k = ((tp * 0.30 * 150.0) - (fp * 2.0) + (tn * 2.0)) * (10000.0 / len(y_true))
        summary[f"threshold_{tau:.2f}"] = {
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "estimated_net_value_per_10k_reviews_usd": round(float(net_roi_per_10k), 2),
        }
    return summary


def compute_decile_calibration(y_true, y_prob):
    df_cal = pd.DataFrame({"y": y_true, "p": y_prob})
    df_cal["decile"] = pd.qcut(df_cal["p"], q=10, labels=False, duplicates="drop") + 1
    rows = []
    for dec, grp in df_cal.groupby("decile"):
        rows.append({
            "decile": int(dec),
            "count": int(len(grp)),
            "mean_predicted_prob": round(float(grp["p"].mean()), 4),
            "observed_complaint_rate": round(float(grp["y"].mean()), 4),
            "calibration_gap": round(float(grp["y"].mean() - grp["p"].mean()), 4),
        })
    return rows


def compute_permutation_feature_importance(model, X_val, y_val, n_repeats=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    base_pr_auc = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    importances = {}
    for col in X_val.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            perm_score = average_precision_score(y_val, model.predict_proba(X_perm)[:, 1])
            drops.append(base_pr_auc - perm_score)
        importances[col] = round(float(np.mean(drops)), 4)
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def run_counterfactual_gifting_audit(df, raw_probs, organic_probs):
    audit = {}
    for subcat, grp in df.groupby("subcategory"):
        idx = grp.index.to_numpy()
        audit[subcat] = {
            "review_count": int(len(idx)),
            "observed_complaint_rate": round(float(grp["target_low_rating"].mean()), 4),
            "mean_predicted_raw_risk": round(float(raw_probs[idx].mean()), 4),
            "mean_counterfactual_organic_risk": round(float(organic_probs[idx].mean()), 4),
        }
    return audit


def run_capstone_verification_tests(y_true, oof_probs, calibration_rows):
    assert 0.50 <= average_precision_score(y_true, oof_probs) <= 1.0
    assert 0.70 <= roc_auc_score(y_true, oof_probs) <= 1.0
    assert len(calibration_rows) >= 5
    prec, rec, _ = precision_recall_curve(y_true, oof_probs)
    assert len(prec) == len(rec)


if __name__ == "__main__":
    df_corpus = generate_evaluation_corpus()
    y_all = df_corpus["target_low_rating"].to_numpy()

    oof_predictions, oof_organic_preds, cv_folds, last_artifacts = fit_and_evaluate_stacked_pipeline(df_corpus)
    meta_clf, X_val_last, y_val_last = last_artifacts

    threshold_report = evaluate_operating_thresholds(y_all, oof_predictions)
    calibration_report = compute_decile_calibration(y_all, oof_predictions)
    feature_importance_report = compute_permutation_feature_importance(meta_clf, X_val_last, y_val_last)
    gifting_bias_report = run_counterfactual_gifting_audit(df_corpus, oof_predictions, oof_organic_preds)
    run_capstone_verification_tests(y_all, oof_predictions, calibration_report)

    evaluation_manifest = {
        "capstone_stage": "Week 6 Model Evaluation, Reporting & Business Impact Communication",
        "random_seed": RANDOM_SEED,
        "dataset_rows": len(df_corpus),
        "dataset_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(df_corpus, index=True).values.tobytes()
        ).hexdigest(),
        "overall_metrics": {
            "pr_auc": round(float(average_precision_score(y_all, oof_predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_all, oof_predictions)), 4),
            "brier_score": round(float(brier_score_loss(y_all, oof_predictions)), 4),
        },
        "group_kfold_splits": cv_folds,
        "operating_thresholds": threshold_report,
        "decile_calibration": calibration_report,
        "top_permutation_drivers": dict(list(feature_importance_report.items())[:10]),
        "counterfactual_gifting_audit": gifting_bias_report,
    }
    print(json.dumps(evaluation_manifest, indent=2))
