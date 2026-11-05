"""
Week 3: data cleaning and preprocessing pipeline for beauty and skincare review data.

Data: the example files in the public GitHub repository nadyinky/sephora-analysis (folder "datasets"):
  - skincare_products_reviews.csv   (reviews)
  - product_info.csv                (products)
They are from the same data family as the Kaggle dataset "Sephora Products and Skincare Reviews",
but they are a SAMPLE (about 50,000 reviews), not the full dataset.

Design rules used in every step
  1. Never overwrite raw data: raw files stay untouched and the raw text is kept in its own column.
  2. Flag first, delete only when a row is unusable (no text, duplicate, impossible value).
  3. Every step writes one line to an audit log (rows in, rows out, cells changed, note).
  4. Everything is driven by the CONFIG dictionary and a fixed random seed, so a rerun gives the same result.

How to run
  Google Colab:  !git clone https://github.com/nadyinky/sephora-analysis
                 !pip install -r requirements.txt
                 !python week3_cleaning_pipeline.py
  Local:         create a virtual environment, pip install -r requirements.txt, git clone the repository
                 into the same folder as this script, then run: python week3_cleaning_pipeline.py
  Add the word "selftest" (python week3_cleaning_pipeline.py selftest) to run only the self-test.
"""
import ast
import hashlib
import html
import re
import sys
import time
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import ftfy                                              # repairs broken text encoding and curly quotes
from rapidfuzz import fuzz, process                      # fuzzy matching of near-duplicate names
from scipy import stats                                  # median absolute deviation for robust outlier scores
from sklearn.preprocessing import RobustScaler           # scaling that is not distorted by outliers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score

# ----------------------------------------------------------------------------- configuration
CONFIG = {
    "data_dir": "sephora-analysis/datasets",
    "reviews_file": "skincare_products_reviews.csv",
    "products_file": "product_info.csv",
    "out_dir": "week3_outputs",
    "seed": 42,
    "min_words": 4,                 # fewer words than this -> flagged "too_short" (kept, but not used for themes)
    "iqr_k": 1.5,                   # Tukey fence multiplier for review length (on log scale)
    "robust_z": 3.5,                # |robust z| above this -> outlier flag (median and MAD based)
    "winsor_q": 0.99,               # upper cap for heavy-tailed counts used as model features
    "smoothing": 1.0,               # Laplace smoothing for helpfulness
    "detect_language": True,        # takes about 2 to 3 minutes on 50,000 reviews
    "lang_min_words": 5,
    "lang_min_prob": 0.95,
    "english_stopword_share_max": 0.10,
    "drop_personal_attributes": True,   # data minimisation: drop skin tone, eye colour, hair colour from the output
    "salt": "week3-demo-salt",      # used to pseudonymise reviewer IDs
}

REVIEW_COLS = ["author_id", "rating", "is_recommended", "helpfulness", "total_feedback_count",
               "total_neg_feedback_count", "total_pos_feedback_count", "submission_time", "review_text",
               "review_title", "skin_tone", "eye_color", "skin_type", "hair_color", "product_id",
               "product_name", "brand_name", "price_usd"]
PRODUCT_COLS = ["product_id", "product_name", "brand_name", "size", "ingredients", "price_usd",
                "sale_price_usd", "primary_category", "secondary_category"]

# ----------------------------------------------------------------------------- audit log helpers
LOG = []


def log_step(step, rows_in, rows_out, cells_changed=0, note=""):
    """Append one line to the audit log."""
    LOG.append({"step": step, "rows_in": int(rows_in), "rows_out": int(rows_out),
                "rows_removed": int(rows_in - rows_out), "cells_changed": int(cells_changed), "note": note})


def sha256(path):
    """Fingerprint of a raw file, so anyone can check that the same input was used."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- stage A: load and validate
def validate_schema(df, required, name):
    """Fail early and clearly if expected columns are missing (documentation and files can disagree)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing expected columns {missing}")


def load_raw(cfg):
    base = Path(cfg["data_dir"])
    rev = pd.read_csv(base / cfg["reviews_file"], index_col=0, low_memory=False)
    prod = pd.read_csv(base / cfg["products_file"], low_memory=False)
    validate_schema(rev, REVIEW_COLS, "reviews")
    validate_schema(prod, PRODUCT_COLS, "products")
    log_step("A1 load raw files", len(rev), len(rev), 0,
             f"reviews sha256={sha256(base / cfg['reviews_file'])[:12]}, "
             f"products sha256={sha256(base / cfg['products_file'])[:12]}, product rows={len(prod)}")
    return rev, prod


# ----------------------------------------------------------------------------- stage B: types and ranges
def standardise_types(df):
    n0 = len(df)
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    bad_rating = ~df["rating"].between(1, 5) | df["rating"].isna()
    df["submission_date"] = pd.to_datetime(df["submission_time"], errors="coerce")
    bad_date = df["submission_date"].isna() | (df["submission_date"] > pd.Timestamp.today())
    df["is_recommended"] = df["is_recommended"].astype("boolean")   # nullable boolean keeps "unknown" as <NA>
    for c in ["total_feedback_count", "total_neg_feedback_count", "total_pos_feedback_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    bad_price = df["price_usd"].isna() | (df["price_usd"] <= 0)
    drop = bad_rating | bad_date | bad_price
    df = df.loc[~drop].copy()
    df["rating"] = df["rating"].astype(int)
    df["year"] = df["submission_date"].dt.year
    log_step("B1 types and impossible values", n0, len(df), int(drop.sum()),
             f"invalid rating={int(bad_rating.sum())}, invalid or future date={int(bad_date.sum())}, "
             f"non-positive or missing price={int(bad_price.sum())}")
    return df


# ----------------------------------------------------------------------------- stage C: text standardisation
TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
REPEAT = re.compile(r"([A-Za-z])\1{2,}")                  # soooo -> soo
SPACES = re.compile(r"\s+")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def clean_text(s):
    """Repair encoding and curly quotes, remove markup and URLs, tidy spaces and stretched letters.
    Case is NOT changed here: VADER treats capitals as emphasis, and lower-casing is done later inside
    the model feature step."""
    if not isinstance(s, str):
        return s
    s = ftfy.fix_text(s)                                  # mojibake repair + curly quotes -> straight quotes
    s = html.unescape(s)                                  # &amp; -> &
    s = TAG.sub(" ", s)
    s = URL.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = REPEAT.sub(r"\1\1", s)
    return SPACES.sub(" ", s).strip()


def standardise_text(df):
    n = len(df)
    df = df.copy()
    df["review_text_raw"] = df["review_text"]
    df["review_text_clean"] = df["review_text_raw"].map(clean_text)
    changed = int((df["review_text_raw"].fillna("") != df["review_text_clean"].fillna("")).sum())
    curly = int(df["review_text_raw"].fillna("").str.contains("\u2019|\u2018|\u201c|\u201d").sum())
    df["review_title"] = df["review_title"].map(clean_text)
    log_step("C1 text standardisation", n, n, changed,
             f"texts changed={changed}; with curly quotes before={curly}; "
             f"with curly quotes after={int(df['review_text_clean'].fillna('').str.contains(chr(0x2019)).sum())}")
    return df


# ----------------------------------------------------------------------------- stage D: duplicates
def remove_duplicates(df):
    n0 = len(df)
    df = df[df["review_text_clean"].notna() & (df["review_text_clean"].str.len() > 0)].copy()
    n_blank = n0 - len(df)
    log_step("D1 remove reviews with no text", n0, len(df), 0, "review text is the essential field")
    n1 = len(df)
    df = df.drop_duplicates(["author_id", "product_id", "review_text_clean"], keep="first")
    log_step("D2 exact duplicates", n1, len(df), 0, "same reviewer, same product, same text")
    # the same review posted under several product IDs (for example full size and mini size)
    n2 = len(df)
    ids = df.groupby(["author_id", "review_text_clean"])["product_id"].transform("nunique")
    df["n_product_ids"] = ids.astype(int)
    df = df.drop_duplicates(["author_id", "review_text_clean"], keep="first").copy()
    log_step("D3 same review under several product IDs", n2, len(df), 0,
             f"kept the first copy; {int((ids > 1).sum())} rows belonged to a repeated review")
    return df


# ----------------------------------------------------------------------------- stage E: missing values
def handle_missing(df):
    n = len(df)
    df = df.copy()
    notes = []
    # E1 title: missing means the reviewer wrote no title -> empty string + indicator
    df["has_title"] = df["review_title"].notna()
    df["review_title"] = df["review_title"].fillna("")
    notes.append(f"title missing={int((~df['has_title']).sum())}")
    # E2 helpfulness is missing exactly when nobody voted: structural, so do not impute a number
    structural = df["helpfulness"].isna() & (df["total_feedback_count"] == 0)
    other = int((df["helpfulness"].isna() & ~structural).sum())
    a = CONFIG["smoothing"]
    df["helpfulness_smoothed"] = (df["total_pos_feedback_count"] + a) / (df["total_feedback_count"] + 2 * a)
    notes.append(f"helpfulness missing={int(df['helpfulness'].isna().sum())} (all structural: {other == 0})")
    # E3 recommendation flag: keep <NA>, add indicator (the value is not random: 5-star reviews miss it most)
    df["rec_missing"] = df["is_recommended"].isna()
    notes.append(f"is_recommended missing={int(df['rec_missing'].sum())}")
    # E4 reviewer profile fields: 'unknown' category, so rows are kept and the gap stays visible
    cat_missing = 0
    for c in ["skin_type", "skin_tone", "eye_color", "hair_color"]:
        cat_missing += int(df[c].isna().sum())
        df[c] = df[c].fillna("unknown")
    notes.append(f"profile fields set to 'unknown'={cat_missing}")
    log_step("E1 missing values", n, len(df), cat_missing + int(df["rec_missing"].sum()), "; ".join(notes))
    return df


# ----------------------------------------------------------------------------- stage F: categories and names
def standardise_categories(df):
    n = len(df)
    df = df.copy()
    before = df["eye_color"].nunique()
    for c in ["skin_type", "skin_tone", "eye_color", "hair_color"]:
        df[c] = df[c].astype(str).str.strip()
    # camelCase labels such as lightMedium -> light medium; the code 'notSureST' means the reviewer was not sure
    df["skin_tone"] = df["skin_tone"].replace({"notSureST": "not sure"})
    df["skin_tone"] = df["skin_tone"].str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True)
    for c in ["skin_type", "skin_tone", "eye_color", "hair_color"]:
        df[c] = df[c].str.lower()
    df["eye_color"] = df["eye_color"].replace({"grey": "gray"})      # Grey (294) and gray (83) are the same colour
    # brand names: one canonical display name per normalised key
    df["brand_key"] = df["brand_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    display = df.groupby("brand_key")["brand_name"].agg(lambda s: s.value_counts().index[0])
    df["brand_display"] = df["brand_key"].map(display)
    keys = sorted(df["brand_key"].unique())
    pairs = []
    for i, k in enumerate(keys):
        for cand, score, _ in process.extract(k, keys[i + 1:], scorer=fuzz.ratio, limit=3, score_cutoff=92):
            pairs.append((k, cand, round(score, 1)))
    log_step("F1 categories and brand names", n, len(df), 0,
             f"eye_color labels {before} -> {df['eye_color'].nunique()}; brands {df['brand_name'].nunique()} -> "
             f"{df['brand_key'].nunique()}; fuzzy brand pairs with score >= 92: {len(pairs)}")
    return df


# ----------------------------------------------------------------------------- stage G: product enrichment
SIZE_ML = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mL|ml)\b")
SIZE_OZ = re.compile(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz\b", re.I)
SIZE_G = re.compile(r"(\d+(?:\.\d+)?)\s*g\b")
SIZE_COUNT = re.compile(r"(\d+)\s*(pads|vials|sheets|patches|count|ct|pieces|masks|wipes|capsules|pack)", re.I)
MULTIPACK = re.compile(r"^\s*(\d+)\s*[x\u00d7]\s*\d")      # '4 x 0.25 oz/ 7.5 mL' means four items of 7.5 mL
OZ_TO_ML = 29.5735        # 1 US fluid ounce in millilitres


def parse_size(s):
    """Turn strings such as '1.7 oz/ 50 mL' or '0.5 oz/ 14 g' into (value, unit)."""
    if not isinstance(s, str):
        return np.nan, "missing"
    pack = MULTIPACK.search(s)
    k = int(pack.group(1)) if pack else 1                       # multiply multi-packs by the number of items
    m = SIZE_ML.search(s)
    if m:
        return float(m.group(1)) * k, "ml"
    g = SIZE_G.search(s)
    oz = SIZE_OZ.search(s)
    if g:
        return float(g.group(1)) * k, "g"
    if oz:
        return float(oz.group(1)) * OZ_TO_ML * k, "ml_from_oz"  # ounces only: converted, flagged as converted
    c = SIZE_COUNT.search(s)
    if c:
        return float(c.group(1)), "count"
    return np.nan, "unparsed"


COMMA_OUTSIDE_BRACKETS = re.compile(r",(?![^()]*\))")


def parse_ingredients(raw):
    """The ingredient column holds the text of a Python list, sometimes with variant labels such as
    'Capri Eau de Parfum:'. Return (list of ingredient names of the first variant, number of variants)."""
    if not isinstance(raw, str):
        return [], 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # some ingredient texts contain stray backslashes
            items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        items = [raw]
    variants, current = [], []
    for it in items:
        it = str(it).strip()
        if it.endswith(":") and "," not in it:             # a variant label, start a new variant
            if current:
                variants.append(current)
            current = []
        else:
            current.append(it)
    if current:
        variants.append(current)
    if not variants:
        return [], 0
    joined = " ".join(variants[0])
    names = [SPACES.sub(" ", x).strip(" .;") for x in COMMA_OUTSIDE_BRACKETS.split(joined)]
    return [x for x in names if x], len(variants)


def robust_z(x):
    """Robust z-score: (x - median) / (1.4826 * MAD). Not distorted by the outliers it looks for."""
    mad = stats.median_abs_deviation(x, nan_policy="omit")
    return (x - np.nanmedian(x)) / (1.4826 * mad + 1e-12)


def clean_products(prod, cfg=CONFIG):
    n = len(prod)
    prod = prod.copy()
    parsed = prod["size"].map(parse_size)
    prod["size_value"] = [p[0] for p in parsed]
    prod["size_unit"] = [p[1] for p in parsed]
    ml = prod["size_unit"].isin(["ml", "ml_from_oz"]) & (prod["size_value"] > 0)
    prod["price_per_ml"] = np.where(ml, prod["price_usd"] / prod["size_value"].where(ml), np.nan)
    ing = prod["ingredients"].map(parse_ingredients)
    prod["ingredients_clean"] = [i[0] for i in ing]
    prod["n_variants"] = [i[1] for i in ing]
    prod["n_ingredients"] = prod["ingredients_clean"].map(len)
    prod["has_fragrance"] = prod["ingredients_clean"].map(
        lambda L: any(re.search(r"\b(parfum|fragrance)\b", x, re.I) for x in L))
    prod["on_sale"] = prod["sale_price_usd"].notna()
    # outliers: robust z-score of log price per mL inside each primary category (flag only)
    prod["ppml_outlier"] = False
    lp = np.log(prod["price_per_ml"])
    for cat, idx in prod.groupby("primary_category").groups.items():
        sub = lp.loc[idx]
        if sub.notna().sum() >= 20:
            prod.loc[idx, "ppml_outlier"] = (robust_z(sub).abs() > cfg["robust_z"]).fillna(False)
    prod["price_outlier"] = False
    lpr = np.log(prod["price_usd"])
    for cat, idx in prod.groupby("primary_category").groups.items():
        sub = lpr.loc[idx]
        if len(sub) >= 20:
            prod.loc[idx, "price_outlier"] = (robust_z(sub).abs() > cfg["robust_z"]).fillna(False)
    dup_names = prod.duplicated(["product_name", "brand_name", "size"], keep=False)
    prod["dup_product_flag"] = dup_names
    log_step("G1 product table", n, len(prod), 0,
             f"size parsed={int(prod['size_unit'].isin(['ml', 'g', 'ml_from_oz', 'count']).sum())} of "
             f"{int(prod['size'].notna().sum())}; ingredients parsed={int((prod['n_ingredients'] > 0).sum())}; "
             f"price per mL outliers={int(prod['ppml_outlier'].sum())}; price outliers={int(prod['price_outlier'].sum())}; "
             f"rows identical on name+brand+size={int(dup_names.sum())}")
    return prod


# ----------------------------------------------------------------------------- stage H and I: outliers and flags
DISCLOSURE = re.compile(
    r"received (?:this|it|the product|these|a|an)?.{0,40}(?:for free|free of charge|complimentary|at no cost)"
    r"|\bgifted\b|in exchange for|exchange for (?:my )?(?:honest|unbiased)|free sample|influenster|voxbox", re.I)
EN_STOP = set("the a an and or but of to in on for with is it this that i my me you your was are be have has had not "
              "very so as at by from just can will would they them their there these those its it's i'm i've don't "
              "didn't doesn't than then too also one use used using after when what which about".split())


def add_flags(df, cfg=CONFIG):
    n = len(df)
    df = df.copy()
    df["words"] = df["review_text_clean"].str.split().str.len()
    df["log_words"] = np.log1p(df["words"])
    q1, q3 = df["log_words"].quantile([0.25, 0.75])
    lo, hi = q1 - cfg["iqr_k"] * (q3 - q1), q3 + cfg["iqr_k"] * (q3 - q1)
    df["len_outlier"] = (df["log_words"] < lo) | (df["log_words"] > hi)
    df["too_short"] = df["words"] < cfg["min_words"]
    df["all_caps"] = df["review_text_clean"].map(lambda s: len(s) >= 20 and s.isupper())
    df["has_emoji"] = df["review_text_raw"].fillna("").str.contains(EMOJI)
    df["had_url_or_html"] = df["review_text_raw"].fillna("").str.contains(URL) | df["review_text_raw"].fillna("").str.contains(TAG)
    df["incentivised"] = df["review_text_clean"].str.contains(DISCLOSURE)
    rec = df["is_recommended"]
    df["rating_rec_conflict"] = (((df["rating"] == 5) & (rec == False)) |
                                 ((df["rating"] <= 2) & (rec == True))).fillna(False).astype(bool)  # <NA> is not a conflict
    cap = df["total_feedback_count"].quantile(cfg["winsor_q"])
    df["feedback_w"] = df["total_feedback_count"].clip(upper=cap)               # winsorised copy for models
    log_step("H1 outlier and quality flags", n, len(df), 0,
             f"length fence {np.expm1(lo):.1f} to {np.expm1(hi):.0f} words (flagged {int(df['len_outlier'].sum())}); "
             f"too short (<{cfg['min_words']} words)={int(df['too_short'].sum())}; all caps={int(df['all_caps'].sum())}; "
             f"emoji={int(df['has_emoji'].sum())}; url or html={int(df['had_url_or_html'].sum())}; "
             f"feedback count capped at {cap:.0f}")
    log_step("H2 domain flags", n, len(df), 0,
             f"incentivised or gifted={int(df['incentivised'].sum())} ({df['incentivised'].mean() * 100:.1f}%); "
             f"rating and recommendation conflict={int(df['rating_rec_conflict'].sum())}")
    return df


def flag_non_english(df, cfg=CONFIG):
    """Language detection is unreliable on short, emphatic English, so a review is flagged only when the
    detector is confident AND the text has hardly any common English words."""
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = cfg["seed"]
    n = len(df)
    flags, tested = np.zeros(n, dtype=bool), 0
    raw_non_en = 0
    for i, (text, w) in enumerate(zip(df["review_text_clean"], df["words"])):
        if w < cfg["lang_min_words"]:
            continue
        tested += 1
        try:
            best = detect_langs(text)[0]
        except Exception:
            continue
        if best.lang != "en":
            raw_non_en += 1
            toks = re.findall(r"[a-z']+", text.lower())
            share = sum(t in EN_STOP for t in toks) / max(len(toks), 1)
            if best.prob >= cfg["lang_min_prob"] and share < cfg["english_stopword_share_max"]:
                flags[i] = True
    df = df.copy()
    df["non_english"] = flags
    log_step("I1 language check", n, n, 0,
             f"tested={tested}; detector said not English={raw_non_en}; flagged after confidence and stop-word check="
             f"{int(flags.sum())}")
    return df


# ----------------------------------------------------------------------------- stage J: minimisation and output
def minimise_and_finish(df, products, cfg=CONFIG):
    n = len(df)
    df = df.copy()
    pcols = ["product_id", "primary_category", "secondary_category", "price_per_ml", "ppml_outlier",
             "price_outlier", "n_ingredients", "has_fragrance", "on_sale"]
    df = df.merge(products[pcols], on="product_id", how="left", validate="m:1")
    df["rating_label"] = np.select([df["rating"] <= 2, df["rating"] == 3], ["negative", "neutral"], "positive")
    df["reviewer_key"] = df["author_id"].map(lambda a: hashlib.sha256((cfg["salt"] + str(a)).encode()).hexdigest()[:12])
    dropped = []
    if cfg["drop_personal_attributes"]:
        dropped = ["skin_tone", "eye_color", "hair_color"]
    df = df.drop(columns=["author_id", "review_text", "submission_time"] + dropped)
    log_step("J1 data minimisation", n, len(df), 0,
             f"reviewer IDs hashed; columns dropped: {dropped}; product fields merged for {int(df['primary_category'].notna().sum())} reviews")
    return df


def validate_output(df):
    """Hard checks. If one fails the pipeline stops instead of silently producing bad data."""
    checks = {
        "rating between 1 and 5": df["rating"].between(1, 5).all(),
        "no empty review text": (df["review_text_clean"].str.len() > 0).all(),
        "no duplicate reviewer+text": not df.duplicated(["reviewer_key", "review_text_clean"]).any(),
        "dates not in the future": (df["submission_date"] <= pd.Timestamp.today()).all(),
        "helpfulness in [0,1] when present": df["helpfulness"].dropna().between(0, 1).all(),
        "smoothed helpfulness in (0,1)": df["helpfulness_smoothed"].between(0, 1).all(),
        "no curly apostrophes left": not df["review_text_clean"].str.contains("\u2019").any(),
        "product fields joined": df["primary_category"].notna().all(),
    }
    failed = [k for k, v in checks.items() if not bool(v)]
    log_step("K1 validation checks", len(df), len(df), 0, f"{len(checks) - len(failed)} of {len(checks)} passed")
    if failed:
        raise AssertionError(f"validation failed: {failed}")
    return checks


# ----------------------------------------------------------------------------- the pipeline
def run_pipeline(cfg=CONFIG, verbose=True):
    LOG.clear()
    t0 = time.time()
    rev, prod = load_raw(cfg)
    raw = rev.copy()
    df = standardise_types(rev)
    df = standardise_text(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = standardise_categories(df)
    products = clean_products(prod, cfg)
    df = add_flags(df, cfg)
    if cfg["detect_language"]:
        df = flag_non_english(df, cfg)
    else:
        df["non_english"] = False
    df = minimise_and_finish(df, products, cfg)
    checks = validate_output(df)
    out = Path(cfg["out_dir"])
    out.mkdir(exist_ok=True)
    df.to_csv(out / "cleaned_reviews.csv", index=False)
    products.drop(columns=["ingredients_clean"]).to_csv(out / "cleaned_products.csv", index=False)
    pd.DataFrame(LOG).to_csv(out / "cleaning_log.csv", index=False)
    if verbose:
        print(f"\nPipeline finished in {time.time() - t0:.0f} s. Output folder: {out}/")
        print_log()
    return raw, df, products


def print_log():
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    for row in LOG:
        print(f"[{row['step']}] rows {row['rows_in']} -> {row['rows_out']} (removed {row['rows_removed']}), "
              f"cells changed {row['cells_changed']} | {row['note']}")


# ----------------------------------------------------------------------------- scaling for models
def make_model_features(df, train_index, test_index):
    """Log-transform heavy-tailed numbers, then scale with RobustScaler. The scaler learns the median and IQR
    from the TRAINING rows only and is then applied to the test rows (no leakage)."""
    cols = {"price_usd": np.log, "feedback_w": np.log1p, "words": np.log1p}
    X = pd.DataFrame({k: f(df[k].astype(float)) for k, f in cols.items()}, index=df.index)
    scaler = RobustScaler().fit(X.loc[train_index])
    return (pd.DataFrame(scaler.transform(X.loc[train_index]), index=train_index, columns=X.columns),
            pd.DataFrame(scaler.transform(X.loc[test_index]), index=test_index, columns=X.columns),
            X.loc[train_index], scaler)


def scaling_demo(df, cfg=CONFIG):
    """Show what log transform + RobustScaler do to heavy-tailed numbers (fit on training rows only)."""
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=cfg["seed"]).split(df, groups=df["reviewer_key"]))
    train_idx, test_idx = df.index[tr], df.index[te]
    Xtr, Xte, raw_train, _ = make_model_features(df, train_idx, test_idx)
    print("\nSCALING DEMO (training rows %d, test rows %d, split by reviewer)" % (len(train_idx), len(test_idx)))
    for c in Xtr.columns:
        q1, med, q3 = raw_train[c].quantile([0.25, 0.5, 0.75])
        t1, tmed, t3 = Xte[c].quantile([0.25, 0.5, 0.75])
        print(f"  {c:12s} after log: median {med:.2f}, IQR {q3 - q1:.2f} | after scaling: train median {Xtr[c].median():.2f}, "
              f"train IQR {Xtr[c].quantile(.75) - Xtr[c].quantile(.25):.2f}, test median {tmed:.2f}, test IQR {t3 - t1:.2f}")
    top = df["price_usd"].max(), df["price_usd"].median()
    print(f"  raw price: median {top[1]:.0f}, maximum {top[0]:.0f} dollars; scaled maximum {Xtr['price_usd'].max():.2f}")


# ----------------------------------------------------------------------------- impact of cleaning
def impact_report(raw, df, cfg=CONFIG):
    """Measure what cleaning changed in the analysis from Week 2 (rating versus text sentiment)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    lab = lambda c: "negative" if c <= -0.05 else ("positive" if c >= 0.05 else "neutral")
    print("\nIMPACT 1: rating versus VADER sentiment, raw text against cleaned text")
    keep = raw.loc[raw["review_text"].notna()].copy()
    keep["rating_label"] = np.select([keep["rating"] <= 2, keep["rating"] == 3], ["negative", "neutral"], "positive")
    keep["v_raw"] = keep["review_text"].map(lambda t: sia.polarity_scores(t)["compound"])
    keep["v_clean"] = keep["review_text"].map(lambda t: sia.polarity_scores(clean_text(t))["compound"])
    keep["l_raw"], keep["l_clean"] = keep["v_raw"].map(lab), keep["v_clean"].map(lab)
    for name, col in [("raw", "l_raw"), ("clean", "l_clean")]:
        lo, hi = keep[keep["rating"] <= 2], keep[keep["rating"] >= 4]
        print(f"  {name:5s}: kappa={cohen_kappa_score(keep['rating_label'], keep[col]):.3f} | "
              f"1-2 star scored positive={(lo[col] == 'positive').mean() * 100:.1f}% | "
              f"4-5 star scored negative={(hi[col] == 'negative').mean() * 100:.1f}%")
    changed = keep[keep["l_raw"] != keep["l_clean"]]
    print(f"  reviews whose VADER label changed after cleaning: {len(changed)} of {len(keep)} "
          f"({len(changed) / len(keep) * 100:.1f}%)")
    print(f"  of these, positive -> negative: {int(((changed.l_raw == 'positive') & (changed.l_clean == 'negative')).sum())}, "
          f"negative -> positive: {int(((changed.l_raw == 'negative') & (changed.l_clean == 'positive')).sum())}")
    ex = keep.assign(d=keep.v_clean - keep.v_raw).sort_values("d").iloc[0]
    print(f"  biggest single change: rating {ex.rating}, score {ex.v_raw:+.2f} -> {ex.v_clean:+.2f}, "
          f"text: {clean_text(ex.review_text)[:110]}")

    print("\nIMPACT 2: incentivised (gifted or free sample) reviews")
    g = df[df["incentivised"]]
    r = df[~df["incentivised"]]
    print(f"  flagged={len(g)} ({len(g) / len(df) * 100:.1f}%); mean rating {g['rating'].mean():.2f} vs {r['rating'].mean():.2f}; "
          f"five-star share {(g['rating'] == 5).mean() * 100:.1f}% vs {(r['rating'] == 5).mean() * 100:.1f}%; "
          f"1-2 star share {(g['rating'] <= 2).mean() * 100:.1f}% vs {(r['rating'] <= 2).mean() * 100:.1f}%")
    print(f"  overall mean rating {df['rating'].mean():.3f}; without incentivised reviews {r['rating'].mean():.3f}")

    print("\nIMPACT 3: what the cleaned ingredient field makes possible (descriptive, not causal)")
    ing = df[df["n_ingredients"] > 0]
    for flag in [True, False]:
        s = ing[ing["has_fragrance"] == flag]
        k, n = int((s["rating"] <= 2).sum()), len(s)
        ph, z = k / n, 1.96
        den = 1 + z * z / n
        centre = (ph + z * z / (2 * n)) / den
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        print(f"  fragrance listed={flag!s:5s}: reviews={n}; 1-2 star share {ph * 100:.1f}% (95% CI {100 * (centre - half):.1f} to "
              f"{100 * (centre + half):.1f}); mean rating {s['rating'].mean():.2f}")

    print("\nIMPACT 4: honest evaluation. Duplicate leakage between training and test rows")
    base = raw.loc[raw["review_text"].notna()].copy()
    base["y"] = np.where(base["rating"] >= 4, 1, np.where(base["rating"] <= 2, 0, -1))
    base = base[base["y"] >= 0]
    clean = df[df["rating"] != 3].copy()
    clean["y"] = (clean["rating"] >= 4).astype(int)

    def run(frame, text_col, group_col, grouped):
        rng = np.random.RandomState(cfg["seed"])
        if grouped:
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=cfg["seed"]).split(frame, groups=frame[group_col]))
        else:
            perm = rng.permutation(len(frame))
            cut = int(len(frame) * 0.8)
            tr, te = perm[:cut], perm[cut:]
        tr_text, te_text = frame[text_col].iloc[tr], frame[text_col].iloc[te]
        overlap = int(te_text.isin(set(tr_text)).sum())
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        Xtr = vec.fit_transform(tr_text)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg["seed"]).fit(Xtr, frame["y"].iloc[tr])
        f1 = f1_score(frame["y"].iloc[te], clf.predict(vec.transform(te_text)), average="macro")
        return overlap, len(te), f1

    o1, n1, f1a = run(base, "review_text", None, False)
    o2, n2, f1b = run(clean, "review_text_clean", "reviewer_key", True)
    print(f"  raw data, random split:              {o1} of {n1} test reviews also appear word for word in training; macro-F1={f1a:.3f}")
    print(f"  cleaned data, split by reviewer:     {o2} of {n2} test reviews also appear word for word in training; macro-F1={f1b:.3f}")


# ----------------------------------------------------------------------------- self-test with injected errors
def run_self_test():
    """Simulate raw, dirty data with known errors and check that every cleaning step catches them."""
    print("SELF-TEST: injected errors")
    dirty = pd.DataFrame({
        "author_id": ["a1", "a2", "a3", "a4", "a4", "a5", "a6", "a7", "a8", "a9", "a10"],
        "rating": [5, 0, 7, 4, 4, 1, 3, 5, 2, 4, 5],
        "is_recommended": [1.0, 1.0, np.nan, 1.0, 1.0, 0.0, np.nan, 0.0, 1.0, np.nan, 1.0],
        "helpfulness": [0.9, np.nan, 0.5, 1.0, 1.0, np.nan, 0.0, 0.8, 0.3, np.nan, 1.0],
        "total_feedback_count": [10, 0, 4, 2, 2, 0, 3, 5, 10, 0, 1],
        "total_neg_feedback_count": [1, 0, 2, 0, 0, 0, 3, 1, 7, 0, 0],
        "total_pos_feedback_count": [9, 0, 2, 2, 2, 0, 0, 4, 3, 0, 1],
        "submission_time": ["2021-05-01", "2021-05-02", "2021-05-03", "2021-06-01", "2021-06-01", "2022-02-03",
                            "2099-01-01", "2020-01-05", "2020-02-05", "2020-03-05", "31/02/2022"],
        "review_text": ["I don\u00e2\u20ac\u2122t like it, but <b>it works</b> https://x.co", "Fine", "Nice cream",
                        "Great product, sooooo smooth!!!", "Great product, sooooo smooth!!!", "TERRIBLE, BROKE ME OUT AND ITS HORRIBLE",
                        "I don\u2019t love the smell", "I received this product for free from Influenster. Love it", None,
                        "Works well on my skin", "Great"],
        "review_title": [None, "ok", None, "Good", "Good", None, "meh", "yes", "no", None, "ok"],
        "skin_tone": ["lightMedium", "fair", None, "mediumTan", "mediumTan", "deep", None, "light", "light", "fair", None],
        "eye_color": ["Grey", "gray", None, "brown", "brown", "blue", None, "green", "hazel", "brown", None],
        "skin_type": ["dry", None, "oily", "combination", "combination", "dry", "normal", None, "oily", "dry", "dry"],
        "hair_color": ["brown", "black", None, "blonde", "blonde", "red", None, "brown", "gray", "black", None],
        "product_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P1", "P2", "P3", "P1", "P1"],
        "product_name": ["Cream A", "Cream A", "Serum B", "Serum B", "Serum B mini", "Mask C", "Cream A", "Serum B", "Mask C", "Cream A", "Cream A"],
        "brand_name": ["tarte", "TARTE", "Fresh", "fresh", "fresh", "Kiehl's", "tarte", "Fresh", "Kiehl's", "tarte", "tarte"],
        "price_usd": [30.0, 30.0, 45.0, 45.0, 15.0, 22.0, 30.0, 45.0, 22.0, -5.0, 30.0],
    })
    LOG.clear()
    d = standardise_types(dirty)
    d = standardise_text(d)
    d = remove_duplicates(d)
    d = handle_missing(d)
    d = standardise_categories(d)
    d = add_flags(d, {**CONFIG})
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    check("invalid ratings 0 and 7 removed", d["rating"].between(1, 5).all() and 0 not in d["rating"].values)
    check("impossible dates (31/02/2022 and year 2099) removed", d["submission_date"].notna().all() and (d["submission_date"] < pd.Timestamp("2030-01-01")).all())
    check("negative price removed", (d["price_usd"] > 0).all())
    check("mojibake repaired (don\u00e2\u20ac\u2122t -> don't)", "don't like it" in d["review_text_clean"].iloc[0])
    check("HTML tags and URL removed", "<b>" not in d["review_text_clean"].iloc[0] and "http" not in d["review_text_clean"].iloc[0])
    check("curly apostrophe replaced by straight apostrophe", not d["review_text_clean"].str.contains("\u2019").any())
    check("stretched letters shortened (sooooo -> soo)", d["review_text_clean"].str.contains("soo smooth").any() and not d["review_text_clean"].str.contains("sooo").any())
    check("exact duplicate removed", d.duplicated(["author_id", "product_id", "review_text_clean"]).sum() == 0)
    check("same review under two product IDs kept once", d.duplicated(["author_id", "review_text_clean"]).sum() == 0)
    check("empty review text removed", d["review_text_clean"].str.len().gt(0).all())
    check("eye colour Grey and gray merged", set(d["eye_color"]) <= {"gray", "brown", "blue", "green", "hazel", "unknown"} and "grey" not in set(d["eye_color"]))
    check("skin tone lightMedium -> light medium", "light medium" in set(d["skin_tone"]))
    check("brand names tarte/TARTE and Fresh/fresh merged", d["brand_key"].nunique() == d["brand_display"].nunique())
    check("helpfulness left empty when nobody voted", d.loc[d["total_feedback_count"] == 0, "helpfulness"].isna().all())
    check("smoothed helpfulness between 0 and 1", d["helpfulness_smoothed"].between(0, 1).all())
    check("incentivised review flagged", d["incentivised"].sum() == 1)
    check("all-caps review flagged", d["all_caps"].sum() == 1)
    check("rating and recommendation conflict flagged", d["rating_rec_conflict"].sum() >= 1)
    prod = pd.DataFrame({
        "product_id": ["P1", "P2", "P3"], "product_name": ["Cream A", "Serum B", "Mask C"], "brand_name": ["tarte", "Fresh", "Kiehl's"],
        "size": ["1.7 oz/ 50 mL", "0.5 oz/ 14 g", "n/a"],
        "ingredients": ["['Cream A:', 'Water, Glycerin, Parfum (Fragrance)', 'Serum:', 'Water, Niacinamide']", "['Water, Alcohol Denat., Niacinamide']", None],
        "price_usd": [30.0, 45.0, 22.0], "sale_price_usd": [np.nan, 40.0, np.nan], "primary_category": ["Skincare"] * 3, "secondary_category": ["Moisturizers", "Treatments", "Masks"]})
    p = clean_products(prod)
    check("size '1.7 oz/ 50 mL' parsed as 50 mL", p.loc[0, "size_value"] == 50 and p.loc[0, "size_unit"] == "ml")
    check("size '0.5 oz/ 14 g' parsed as 14 g", p.loc[1, "size_value"] == 14 and p.loc[1, "size_unit"] == "g")
    check("size 'n/a' marked unparsed, not guessed", p.loc[2, "size_unit"] == "unparsed")
    check("ingredient list text parsed, variant label removed", p.loc[0, "ingredients_clean"] == ["Water", "Glycerin", "Parfum (Fragrance)"] and p.loc[0, "n_variants"] == 2)
    check("fragrance flag set", bool(p.loc[0, "has_fragrance"]) and not bool(p.loc[1, "has_fragrance"]))
    check("multi-pack '3 x 0.17 oz/ 5 mL' parsed as 15 mL", parse_size("3 x 0.17 oz/ 5 mL") == (15.0, "ml"))
    check("recommendation flag has no missing values", d["rating_rec_conflict"].notna().all())
    check("skin tone code notSureST -> not sure", "not sure" in set(standardise_categories(handle_missing(d.assign(skin_tone=["notSureST"] * len(d))))["skin_tone"]))
    check("missing ingredients stay empty", p.loc[2, "n_ingredients"] == 0)
    print(f"SELF-TEST: {sum(results)} of {len(results)} checks passed")
    if not all(results):
        raise SystemExit(1)
    return len(results)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    print("Versions:", f"pandas {pd.__version__}, numpy {np.__version__}, ftfy {ftfy.__version__}")
    run_self_test()
    if "selftest" not in sys.argv:
        raw, cleaned, products = run_pipeline()
        impact_report(raw, cleaned)
        scaling_demo(cleaned)
