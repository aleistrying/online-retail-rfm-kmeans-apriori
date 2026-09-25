"""
CBD-3333: Data Mining and Analysis - Assignment 1
Practical Data Mining with Python (Online Retail dataset)

Covers:
  Task 1 - Data Preprocessing            (30 marks)
  Task 2 - Customer Clustering (K-Means)  (35 marks)
  Task 3 - Association Rule Mining        (35 marks)

------------------------------------------------------------------------------
WHY THIS IS A .py SCRIPT (and not run inside the VS Code notebook):
The previous OOM crashes came from (a) densifying a 541,909 x 4,108 one-hot
matrix (~17.8 GB) and (b) the VS Code Jupyter UI holding everything in RAM on a
15 GB machine. Running this in a plain terminal ('python assignment-1-solution.py')
avoids the notebook overhead, and the guardrails below stop any single runaway
allocation from taking down your whole desktop again.

The clustering uses CUSTOMER-LEVEL (RFM) features (~4,300 rows), never the giant
transaction-level one-hot matrix, so it stays tiny. The association mining keeps
everything SPARSE. Nothing here should exceed ~1-2 GB of RAM.
------------------------------------------------------------------------------
"""

# =============================================================================
# 0. MEMORY / CPU GUARDRAILS  (must run BEFORE numpy-backed imports)
# =============================================================================
import os

# Cap math-library thread pools so the script can't peg all 24 cores and freeze
# the desktop. 4 threads is plenty for this workload.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import resource

# Hard ceiling on this process's address space. If any operation tries to
# allocate past this (e.g. an accidental dense one-hot), Python raises a
# *catchable* MemoryError INSTEAD of the Linux OOM-killer nuking Chrome/VS Code.
# Everything legitimate in this script stays well under 2 GB, so 10 GB is safe
# headroom on a 15 GB machine and still blocks the ~17.8 GB densification bomb.
MEM_LIMIT_GB = 10
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_GB * 1024**3, _hard))
    print(f"[guard] Address-space cap set to {MEM_LIMIT_GB} GB "
          f"(runaway allocations now fail safely instead of crashing the PC).")
except (ValueError, OSError) as e:  # pragma: no cover
    print(f"[guard] could not set memory cap: {e}")

import gc
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: save figures to files, open no GUI windows (lighter)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# ----- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_XLSX = os.path.join(HERE, "online_retail.xlsx")
CACHE_PARQUET = os.path.join(HERE, "online_retail_clean.parquet")  # speeds up re-runs
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _savefig(name):
    """Save the current figure to outputs/ and close it (frees memory)."""
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close("all")
    print(f"   saved figure -> outputs/{name}")


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# =============================================================================
# TASK 1 - DATA PREPROCESSING (30 marks)
# =============================================================================
def task1_preprocess():
    section("TASK 1 - DATA PREPROCESSING")

    # --- Load (cache to parquet so the slow 23 MB xlsx is read only once) -----
    raw = pd.read_excel(RAW_XLSX)  # openpyxl backend; one-time cost
    print(f"Raw shape: {raw.shape}")
    print("Columns:", list(raw.columns))

    # Normalise column names (UCI file sometimes ships as InvoiceNo/Invoice etc.)
    raw.columns = [c.strip() for c in raw.columns]

    # --- (a) Handle missing values (5) ---------------------------------------
    print("\n[a] Missing values per column:\n", raw.isna().sum())
    # CustomerID is required for customer clustering; Description for itemsets.
    df = raw.dropna(subset=["CustomerID", "Description"]).copy()
    df["CustomerID"] = df["CustomerID"].astype("int64")
    print(f"   after dropping rows missing CustomerID/Description: {df.shape}")

    # Business cleaning: drop cancelled invoices ('C' prefix) and non-positive
    # quantity/price (returns & data errors) - standard for this dataset.
    before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"   removed {before - len(df):,} cancellation/non-positive rows")

    # --- (b) Convert categorical -> numerical (5) ----------------------------
    # StockCode mixes ints and strings -> force to str so encoders don't choke.
    df["StockCode"] = df["StockCode"].astype(str)
    # Low-cardinality 'Country' -> compact integer code (fine for models).
    df["Country_Code"] = df["Country"].astype("category").cat.codes

    # Demonstrate CORRECT one-hot encoding kept SPARSE. This is the operation
    # that previously OOM'd: pd.DataFrame(sparse_matrix, ...) DENSIFIES to GBs.
    # The fix is pd.DataFrame.sparse.from_spmatrix(...), which stays in MBs.
    enc = OneHotEncoder(sparse_output=True)
    onehot = enc.fit_transform(df[["StockCode", "Country"]])      # scipy sparse
    onehot_df = pd.DataFrame.sparse.from_spmatrix(                # <-- the fix
        onehot, columns=enc.get_feature_names_out(["StockCode", "Country"]),
        index=df.index,
    )
    dense_gb = onehot.shape[0] * onehot.shape[1] * 8 / 1e9
    sparse_mb = onehot.data.nbytes / 1e6
    print(f"[b] One-hot shape {onehot.shape}: "
          f"dense would be ~{dense_gb:.1f} GB, sparse uses ~{sparse_mb:.1f} MB "
          f"(this is what caused the earlier crash if densified).")
    del onehot, onehot_df            # not needed downstream; free immediately
    gc.collect()

    # --- (c) Remove duplicate records (5) ------------------------------------
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"[c] Removed {dups:,} duplicate rows -> {df.shape}")

    # --- (d) Feature engineering: TotalPrice (5) -----------------------------
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    print("[d] Added TotalPrice, Year, Month features.")

    # --- (e) Normalise numerical features (5) --------------------------------
    num_cols = ["Quantity", "UnitPrice", "TotalPrice"]
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]),
                          columns=[f"{c}_z" for c in num_cols], index=df.index)
    df = pd.concat([df, scaled], axis=1)
    print("[e] Standardized numeric features (mean~0, std~1):")
    print(scaled.describe().loc[["mean", "std"]].round(3))

    # --- (f) Summary statistics + visualizations (5) -------------------------
    print("\n[f] Summary statistics:")
    print(df[["Quantity", "UnitPrice", "TotalPrice"]].describe().round(2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, num_cols):
        sns.histplot(df[col].clip(upper=df[col].quantile(0.99)), bins=40, ax=ax)
        ax.set_title(f"Distribution of {col} (99th-pct clipped)")
    _savefig("task1_distributions.png")

    top_country = df["Country"].value_counts().head(10)
    plt.figure(figsize=(9, 4))
    sns.barplot(x=top_country.values, y=top_country.index)
    plt.title("Top 10 countries by number of records")
    plt.xlabel("records")
    _savefig("task1_top_countries.png")

    top_products = (df.groupby("Description")["Quantity"].sum()
                      .sort_values(ascending=False).head(10))
    plt.figure(figsize=(9, 4))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title("Top 10 products by total quantity sold")
    plt.xlabel("units sold")
    _savefig("task1_top_products.png")

    # cache cleaned data for fast re-runs
    df.to_parquet(CACHE_PARQUET, index=False)
    print(f"\nCleaned dataset cached -> {os.path.basename(CACHE_PARQUET)} "
          f"(re-runs will be much faster).")
    return df


# =============================================================================
# TASK 2 - CUSTOMER CLUSTERING WITH K-MEANS (35 marks)
# =============================================================================
def task2_clustering(df):
    section("TASK 2 - CUSTOMER CLUSTERING (K-MEANS)")

    # --- (1) Select relevant features: RFM (5) -------------------------------
    # RFM = Recency, Frequency, Monetary - the standard behavioural features for
    # retail customer segmentation. Aggregating 500k+ rows -> ~4,300 customers,
    # so clustering is fast and memory-trivial.
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    rfm = rfm[rfm["Monetary"] > 0]
    print(f"[1] Built RFM table for {len(rfm):,} customers.")
    print(rfm.describe().round(2))

    # --- (2) Normalize features before K-Means (5) ---------------------------
    # RFM is heavily right-skewed -> log1p first, then standardize so each
    # feature contributes equally to the Euclidean distance K-Means uses.
    rfm_log = np.log1p(rfm)
    X = StandardScaler().fit_transform(rfm_log)
    print("[2] Applied log1p + StandardScaler to RFM features.")

    # --- (3) Elbow method for optimal k (5) ----------------------------------
    ks = range(1, 11)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), inertias, "o-")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("inertia (within-cluster SSE)")
    plt.title("Elbow Method")
    _savefig("task2_elbow.png")

    # --- (6) Silhouette score to pick & evaluate k (5) -----------------------
    sil = {}
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        sil[k] = silhouette_score(X, labels)
    best_k = max(sil, key=sil.get)
    print("[3/6] Silhouette by k:",
          {k: round(v, 3) for k, v in sil.items()})
    print(f"      -> best k by silhouette = {best_k} (score {sil[best_k]:.3f})")

    # --- (4) Apply K-Means and interpret clusters (5) ------------------------
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    rfm["Cluster"] = km.fit_predict(X)
    profile = rfm.groupby("Cluster").agg(
        Customers=("Recency", "size"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    ).round(1)
    print("\n[4] Cluster profiles (mean RFM per cluster):")
    print(profile)
    # Plain-language interpretation: lower Recency = more recent; higher
    # Frequency/Monetary = more valuable. The cluster with low Recency + high
    # Frequency + high Monetary = "best / loyal" customers; high Recency + low
    # Frequency + low Monetary = "lapsed / low-value".

    # --- (5) Visualize clustering results (5) --------------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster",
                    palette="tab10", s=25, alpha=0.6)
    plt.xscale("log"); plt.yscale("log")
    plt.title(f"Customer segments (k={best_k}) - Frequency vs Monetary")
    _savefig("task2_clusters_freq_monetary.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster",
                    palette="tab10", s=25, alpha=0.6)
    plt.yscale("log")
    plt.title(f"Customer segments (k={best_k}) - Recency vs Monetary")
    _savefig("task2_clusters_recency_monetary.png")

    # --- (7) Business discussion (printed for the report) (5) ----------------
    print("""
[7] Business use of segmentation:
    - Loyal/high-value clusters  -> VIP perks, early access, retention focus.
    - Recent low-frequency        -> onboarding nudges to drive a 2nd purchase.
    - Lapsed (high Recency)       -> win-back campaigns / discounts.
    - Low-value frequent          -> upsell / bundle offers to raise basket size.
    Targeting spend by segment beats one-size-fits-all marketing.""")

    rfm.to_csv(os.path.join(OUT_DIR, "task2_rfm_clusters.csv"))
    del X, rfm_log
    gc.collect()
    return rfm


# =============================================================================
# TASK 3 - ASSOCIATION RULE MINING (APRIORI) (35 marks)
# =============================================================================
def _basket_from(transactions):
    """Build a SPARSE one-hot basket matrix from a list-of-items-per-invoice.
    Kept sparse end-to-end so it never densifies into GBs."""
    te = TransactionEncoder()
    ary = te.fit_transform(transactions, sparse=True)             # scipy sparse
    return pd.DataFrame.sparse.from_spmatrix(ary, columns=te.columns_)


def task3_association(df):
    section("TASK 3 - ASSOCIATION RULE MINING (APRIORI)")

    # --- (1) Convert to transaction (basket) format (5) ----------------------
    transactions = (df.groupby("InvoiceNo")["Description"]
                      .apply(lambda s: list(set(s))))            # unique items / invoice
    print(f"[1] {len(transactions):,} transactions (invoices).")
    basket = _basket_from(transactions)
    print(f"    Sparse basket matrix: {basket.shape} "
          f"(kept sparse - dense would be "
          f"~{basket.shape[0]*basket.shape[1]*1/1e6:.0f} MB+ and grow fast).")

    # --- (2)+(3) Apriori at the REQUIRED thresholds (5+5) ---------------------
    # NOTE: min_support=0.15 means an itemset must appear in >=15% of ALL
    # baskets. With ~4,000 distinct products that is very strict - typically
    # only a handful of single items qualify and often NO pairs do, so the
    # 0.5-confidence rule set may be empty. That is itself the lesson in step 4.
    print("\n[2/3] Apriori @ min_support=0.15 (assignment requirement):")
    freq = apriori(basket, min_support=0.15, use_colnames=True, low_memory=True)
    print(f"      frequent itemsets found: {len(freq)}")
    if len(freq):
        print(freq.sort_values("support", ascending=False).head(10).to_string(index=False))

    rules = pd.DataFrame()
    if len(freq):
        rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    print(f"      rules @ confidence>=0.5: {len(rules)}")

    # --- Supplementary run so we can actually SHOW top-10 rules ---------------
    # Restrict to the 80 most popular products and lower support so meaningful
    # multi-item rules emerge (bounded -> still fast & memory-safe).
    if len(rules) < 10:
        print("\n[supplementary] Too few rules at 0.15. Re-running on the top-80 "
              "products with min_support=0.02 to illustrate real rules:")
        top_items = (df["Description"].value_counts().head(80).index)
        df_top = df[df["Description"].isin(top_items)]
        tx_top = (df_top.groupby("InvoiceNo")["Description"]
                        .apply(lambda s: list(set(s))))
        basket_top = _basket_from(tx_top)
        freq = apriori(basket_top, min_support=0.02, use_colnames=True, low_memory=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.5)
        print(f"      frequent itemsets: {len(freq)}, rules: {len(rules)}")

    # --- (5) Display & interpret the top 10 rules (5) ------------------------
    if len(rules):
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        show = rules.head(10).copy()
        show["antecedents"] = show["antecedents"].apply(lambda s: ", ".join(list(s)))
        show["consequents"] = show["consequents"].apply(lambda s: ", ".join(list(s)))
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print("\n[5] Top 10 rules (by lift):")
        print(show[cols].round(3).to_string(index=False))
        rules[cols].to_csv(os.path.join(OUT_DIR, "task3_rules.csv"), index=False)
    else:
        print("\n[5] No rules met the thresholds - see discussion in step 4.")

    # --- (6) Visualize most frequent itemsets (5) ----------------------------
    if len(freq):
        single = freq[freq["itemsets"].apply(len) == 1].copy()
        single["item"] = single["itemsets"].apply(lambda s: list(s)[0])
        single = single.sort_values("support", ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        sns.barplot(x=single["support"], y=single["item"])
        plt.title("Most frequent items by support")
        plt.xlabel("support")
        _savefig("task3_frequent_items.png")

    # --- (4) Discussion: support vs confidence (printed) (5) -----------------
    print("""
[4] Choosing support & confidence (and why not to maximize them):
    - SUPPORT = how often an itemset appears. High support -> only the few most
      common items survive; you miss niche-but-valuable patterns. Too low ->
      combinatorial explosion (memory/time) and noisy, coincidental itemsets.
    - CONFIDENCE = P(consequent | antecedent). High confidence can be
      MISLEADING when the consequent is just popular on its own (e.g. a rule
      "X -> bestseller" can have high confidence with no real association).
      That is why LIFT (>1 = positive association) is used to filter rules.
    - Maximizing both yields very few, obvious rules (e.g. "buy milk -> buy
      bread") with little actionable insight; balancing them surfaces useful,
      non-trivial co-purchase patterns.

[7] Business use: place associated products near each other / bundle them,
    drive "frequently bought together" recommendations, and time promotions so
    a high-lift anchor product pulls through its associated items.""")

    del basket
    gc.collect()


# =============================================================================
# MAIN
# =============================================================================
def main():
    if os.path.exists(CACHE_PARQUET):
        print(f"[cache] Loading cleaned data from {os.path.basename(CACHE_PARQUET)} "
              f"(delete it to force a full re-preprocess).")
        df = pd.read_parquet(CACHE_PARQUET)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        section("TASK 1 - DATA PREPROCESSING (loaded from cache)")
        print(f"Cleaned shape: {df.shape}")
    else:
        df = task1_preprocess()

    task2_clustering(df)
    task3_association(df)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    section("DONE")
    print(f"All tasks complete. Peak memory used: {peak_mb:,.0f} MB "
          f"(cap was {MEM_LIMIT_GB*1024} MB).")
    print(f"Figures and CSVs are in: {OUT_DIR}")


if __name__ == "__main__":
    main()
