# Credit Card Customer Segmentation with K-Means

##### Segmenting 8,950 cardholders by spending, cash-advance, and repayment behavior — then naming what each group actually looks like.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-kmeans-dbscan-clustering.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)]()
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)]()
[![Model - KMeans](https://img.shields.io/badge/Model-KMeans-brightgreen)]()
[![Model - Hierarchical](https://img.shields.io/badge/Model-Hierarchical-yellow)]()
[![Model - DBSCAN](https://img.shields.io/badge/Model-DBSCAN-blue)]()

**Author:** **MD. Sakib Al Hasan** · Data Science Portfolio Project · **[Live demo](https://customer-segmentation-kmeans-dbscan-clustering.streamlit.app/)**

This project applies the full unsupervised workflow to the classic **Credit Card Dataset for Clustering**: clean 3.5% missing values, log-transform 12 skewed features, reduce 15 columns to 9 with PCA, then compare K-Means against Hierarchical and DBSCAN. The winner, **K-Means with k=2**, splits the portfolio into two clearly readable groups — 60% purchase-active customers and 40% who lean heavily on cash advances and carry large balances.
![1_cover_v2.png](visuals/1_cover_v3.png)

## Live Demo

**[Open the app: customer-segmentation-kmeans-dbscan-clustering.streamlit.app](https://customer-segmentation-kmeans-dbscan-clustering.streamlit.app/)**

Enter a customer's 6-month card usage (balance, purchases, cash advances, payments and so on) and the app returns the segment, that segment's share of the portfolio, and the suggested business action. Two built-in example customers, one per segment, let you try it in one click.

<!-- Take a screenshot of the running app, save it as visuals/app_demo.png, then delete the two comment markers around the image line below:
![App screenshot](visuals/app_demo.png)
-->

Under the hood the app repeats the exact training pipeline on the input: `log1p` on the 12 skewed features → `StandardScaler` → K-Means (k=2). Code: [`app.py`](app.py). The saved models are loaded from [`models/`](models).

> Free Streamlit apps go to sleep after a few days without visitors. If you see a "Wake up" button, click it and wait a few seconds.

## Problem Statement

Every card issuer's portfolio has a mix of customers who *use* their card productively (buying things, paying partially or in full) and customers who are essentially borrowing at the cash-advance rate. Without segmentation, marketing and risk teams treat everyone the same — the bank advertises rewards to people who never buy, and hands out limit increases to customers who are already carrying risky high balances. That's wasted spend on one side and silent credit-risk creep on the other.

**THE QUESTION:** *Can we determine, purely from 6 months of usage behavior, which customers fall into each behavior segment — and what should the bank do differently for each group?*

The dataset is the Kaggle **Credit Card Dataset for Clustering** ([source](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)) — 8,950 rows × 18 columns, one row per active cardholder, covering a 6-month observation window. `CUST_ID` is the only non-behavioral column and is dropped. Every other feature is numeric: balance carried, purchase patterns (amounts, frequency, transaction counts), cash-advance behavior, repayment amounts, full-payment rate, and tenure.

## Methodology

**1. Cleaning & missing values.** Near-clean dataset. `MINIMUM_PAYMENTS` was 3.5% (313 rows) missing and `CREDIT_LIMIT` had a single hole (0.01%). Both are right-skewed, so both were filled with their median rather than the mean. No duplicate rows.

**2. Feature selection (redundancy).** A correlation scan flagged two pairs above 0.85:

| Pair | Correlation | Action |
|---|---|---|
| PURCHASES – ONEOFF_PURCHASES | 0.917 | Dropped ONEOFF_PURCHASES |
| PURCHASES_FREQUENCY – PURCHASES_INSTALLMENTS_FREQUENCY | 0.863 | Dropped PURCHASES_INSTALLMENTS_FREQUENCY |

Keeping both members of almost-identical pairs would have silently doubled the weight of that signal in the distance metric. That leaves **15 features**.

**3. Skew handling.** Spending and balance data is heavily right-skewed — `MINIMUM_PAYMENTS` hit a skewness of **13.9** and `PURCHASES` **8.1**. Since distance-based clustering is sensitive to tails, the **12** most skewed columns (>1.0) were transformed with `log1p`. After the transform, skew on every transformed column dropped below ≈1.8 (most close to 0). Outliers were *not* deleted: a customer with a large cash advance is real behavior, not a typo.

**4. Scaling & dimensionality.** All features standardized with `StandardScaler` (required for K-Means — columns range from ≈0.0 frequencies to thousands of dollars). PCA on the standardized data: **9 of 15 components capture 95% of the variance**, confirming the dimensionality is genuinely compact.

**5. Choosing k.** K-Means ran for k = 2–10, scored with silhouette:

| k | Inertia | Silhouette |
|---|---|---|
| **2** | 98,950 | **0.2600** |
| 3 | 83,583 | 0.2249 |
| 4 | 74,029 | 0.2289 |
| 5 | 67,375 | 0.2240 |
| 6 | 62,526 | 0.2146 |
| 7 | 58,494 | 0.2203 |
| 8 | 55,429 | 0.1873 |
| 9 | 52,916 | 0.1839 |
| 10 | 50,697 | 0.1894 |

The elbow is a smooth slide rather than a sharp bend, but the silhouette score is unambiguous: it peaks at **k=2**. The data genuinely wants a coarse split, which is typical of noisy, overlapping human behavioral data.

![Elbow and silhouette score for k = 2 to 10](visuals/elbow_silhouette.png)

**6. Algorithm comparison.** Three algorithms at the same settings on the standardized features:

| Model | Clusters | Silhouette | Noise |
|---|---|---|---|
| **K-Means** | **2** | **0.2600** | 0% |
| Hierarchical (Ward) | 2 | 0.2323 | 0% |
| DBSCAN (eps=2.0) | 4 | 0.0725 | 3.7% |

K-Means wins. DBSCAN additionally discards 3.7% of customers as noise and only reaches 0.07 — on dense overlapping behavior data, density-based clustering is the wrong tool, and now we know it's wrong with a number attached.

## Results

K-Means k=2 was the final model. The truth about the score: a 0.26 silhouette is **moderate**, not spectacular — these customers overlap more than they separate, and the README shouldn't pretend otherwise. What the split loses in elegance it gains in readability.

![Customer segments in PCA space](visuals/pca_clusters.png)

| Segment | Share | Definition |
|---|---|---|
| **Cluster 0 — Purchase-active** | 60.4% | Buy regularly (avg 21 purchase transactions, 0.69 purchase frequency), moderate balances (≈$938), little cash-advance use (≈$124), pay in full 23% of months |
| **Cluster 1 — Cash-advance & balance carriers** | 39.6% | Carry high balances (≈$2,519), take large cash advances (≈$2,282, avg 7.6 advance transactions), buy rarely (≈$319), pay in full just 4% of months |

Key differences in raw-feature means:

| Feature | Cluster 0 | Cluster 1 |
|---|---|---|
| BALANCE | $937.91 | $2,519.35 |
| PURCHASES | $1,452.39 | $318.66 |
| CASH_ADVANCE | $123.64 | $2,282.22 |
| PAYMENTS | $1,575.81 | $1,972.92 |
| PRC_FULL_PAYMENT | 0.23 | 0.04 |
| MINIMUM_PAYMENTS | $645.69 | $1,148.51 |

![Average raw values per segment](visuals/segment_profile.png)

Both groups have near-identical credit limits (≈$4,450–4,530) and tenure (≈11.5 months), which makes the divergence in *use* of the card that much more striking — same limits, completely different behavior.

## Business Implications

> **39.6% of the portfolio relies on cash advances at 2.3× the portfolio average ($2,282 vs $979) while paying its balance in full just 4% of months.**

| Segment | Recommended Action |
|---|---|
| Purchase-active (60.4%) | Grow with them: rewards, limit increases, point-linked offers. They transact 21×/month — interchange revenue is here. |
| Cash-advance & balance carriers (39.6%) | Manage risk first: tighter limit policy, targeting the balance-carrier profile before it converts to a write-off. They are also the interest-income segment, so balance-transfer products fit. |

**Suggested next step:** This is behavior snapshots, not outcomes. The validation that actually matters is linking these two segments to *realized* defaults, fees, and interest income over the following 6–12 months. Re-run on a fresh monthly snapshot and check whether cluster assignment (and cluster drift) predicts default; if it does, the segments become a feature in a supervised credit-risk model, not just a dashboard insight.

## Repository Structure

```
Customer-Segmentation-Kmeans-dbscan-clustering/
├── README.md                    # this file
├── LICENSE                      # MIT license (code)
├── app.py                       # Streamlit app (live demo)
├── requirements.txt             # lean dependencies for the Streamlit app
├── requirements-dev.txt         # full pinned environment for the notebook
├── data/
│   └── CC_GENERAL.csv           # raw dataset (8,950 × 18)
├── notebooks/
│   └── ds_workflow.ipynb        # full EDA + clustering, executed with outputs
├── models/
│   ├── kmeans_model.pkl         # final K-Means (k=2) model
│   ├── scaler.pkl               # fitted StandardScaler
│   └── pca.pkl                  # fitted PCA
└── visuals/
    ├── elbow_silhouette.png     # choosing k
    ├── pca_clusters.png         # segments in PCA space
    └── segment_profile.png      # average raw values per segment
```

## Environment & Reproducibility

- Python 3.13 for the notebook. The deployed app resolves its own dependencies from `requirements.txt`.
- Notebook packages (pinned in `requirements-dev.txt`): pandas 3.0.3, numpy 2.2.6, scikit-learn 1.9.0, matplotlib 3.11.0, seaborn 0.13.2, scipy 1.18.0, joblib 1.5.3, Jupyter stack

```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/ds_workflow.ipynb   # run top-to-bottom
```

The notebook reads `../data/CC_GENERAL.csv`, so start it from inside the `notebooks/` folder.

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## License

- **Code** (notebook, saved models, this README): [MIT License](LICENSE) — free to reuse and adapt.
- **Data**: published on Kaggle under the license listed on its [dataset page](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) (CC BY 4.0) — review the current terms there before republishing the data.

## Contact

**MD. Sakib Al Hasan** — open to discussions about customer analytics, unsupervised learning, and credit-risk applications.

[![Email](https://img.shields.io/badge/Email-sakibzzz641%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:sakibzzz641@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-sakibzzz641-121013?style=flat&logo=github&logoColor=white)](https://github.com/sakibzzz641)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sakibzzz641-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/sakibzzz641)
