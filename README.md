# Market Basket Analysis using Apriori Algorithm on Retail Transactions

This repository contains an end-to-end implementation of Association Rule Mining (ARM) applied to point-of-sale grocery transactions using the Apriori algorithm. The project focuses on transforming granular log-level transaction data into transactional baskets, computing frequent itemsets, and extracting actionable association rules based on statistical thresholds.

---

## 1. Problem Formulation & Objectives

Point-of-sale (POS) transactional databases typically record items on individual rows rather than grouping them per checkout basket. To uncover cross-selling opportunities, optimize inventory placement, and design targeted product bundles, these logs must be structured into atomic customer transactions.

### Key Objectives:

* Aggregate sequential line-item records into unique transactional baskets indexed by customer ID and purchase date.
* Encode high-cardinality basket lists into a sparse boolean transaction matrix.
* Discover frequent itemsets using the Apriori property (downward closure).
* Generate and rank association rules based on predefined statistical metrics:
* Minimum Support ($\text{min\_support}$): 0.0052 ($0.52\%$)
* Minimum Confidence ($\text{min\_confidence}$): `0.12` ($12.0\%$)



---

## 2. Mathematical Framework

The Apriori algorithm evaluates relationships of the form $X \rightarrow Y$, where $X$ (antecedent) and $Y$ (consequent) are disjoint itemsets ($X \cap Y = \emptyset$).

* **Support:** Proportion of transactions in the database $D$ containing itemset $X \cup Y$:

$$\text{Support}(X \rightarrow Y) = P(X \cup Y) = \frac{\sigma(X \cup Y)}{\vert{}D\vert{}}$$


* **Confidence:** Conditional probability that a transaction contains $Y$ given that it contains $X$:

$$\text{Confidence}(X \rightarrow Y) = P(Y \mid X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$


* **Lift:** Ratio of observed joint support to expected support under independence:

$$\text{Lift}(X \rightarrow Y) = \frac{\text{Confidence}(X \rightarrow Y)}{\text{Support}(Y)} = \frac{P(X \cup Y)}{P(X)P(Y)}$$


* $\text{Lift} = 1$: Independent items.
* $\text{Lift} > 1$: Positive association.
* $\text{Lift} < 1$: Negative association / substitution effect.



---

## 3. Dataset & Preprocessing Pipeline

* **Dataset:** `Groceries_dataset.csv`
* **Raw Records:** 38,765 purchase entries across 3,898 unique loyalty members.
* **Granularity:** `Member_number`, `Date`, `itemDescription`.

### Transaction Engineering

Raw records represent single-item entries. Baskets are defined by grouping records sharing identical `(Member_number, Date)` pairs:

1. Deduplication per customer-day trip using Python `set` operations.
2. Conversion into a transactional list of sets ($N = 14,963$ distinct transaction baskets).
3. One-hot boolean encoding via `mlxtend.preprocessing.TransactionEncoder` to produce a sparse $14963 \times 167$ matrix.

---

## 4. Implementation

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Ingest raw log data
df = pd.read_csv('data/raw/Groceries_dataset.csv')

# 2. Reconstruct discrete transactions (Member + Date level)
transactions = (
    df.groupby(['Member_number', 'Date'])['itemDescription']
    .apply(lambda items: sorted(list(set(items))))
    .tolist()
)

# 3. Transform to boolean matrix
te = TransactionEncoder()
te_matrix = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_matrix, columns=te.columns_)

# 4. Generate frequent itemsets (min_support = 0.0052)
frequent_itemsets = apriori(
    df_encoded, 
    min_support=0.0052, 
    use_colnames=True
)

# 5. Mine association rules (min_confidence = 0.12)
rules = association_rules(
    frequent_itemsets, 
    metric="confidence", 
    min_threshold=0.12
)

# 6. Rank rules by confidence
top_rules = rules.sort_values(by='confidence', ascending=False).head(5)

```

---

## 5. Results & Top Extracted Rules

Filtering with $\text{min\_support} \ge 0.0052$ and $\text{min\_confidence} \ge 0.12$ yields the following top 5 rules ordered by confidence:

| Rank | Antecedent ($X$) | Consequent ($Y$) | Support | Confidence | Lift |
| --- | --- | --- | --- | --- | --- |
| 1 | `bottled beer` | `whole milk` | 0.007151 | **15.78%** | 0.9993 |
| 2 | `sausage` | `whole milk` | 0.008955 | **14.84%** | 0.9397 |
| 3 | `newspapers` | `whole milk` | 0.005614 | **14.43%** | 0.9139 |
| 4 | `domestic eggs` | `whole milk` | 0.005280 | **14.23%** | 0.9013 |
| 5 | `frankfurter` | `whole milk` | 0.005280 | **13.98%** | 0.8854 |

### Findings & Analytical Insights:

* **Dairy Centrality:** `whole milk` serves as the dominant high-frequency anchor across retail visits.
* **Breakfast & Deli Affinity:** Breakfast staples (`domestic eggs`) and processed meats (`sausage`, `frankfurter`) demonstrate the highest conditional co-occurrence with fresh dairy.
* **Lift Analysis:** The lift values hover close to $1.0$, indicating that while conditional confidence is high due to the sheer baseline popularity of milk, these pairings represent regular pantry replenishment behavior rather than strong isolated affinity clusters.

---

## 6. Repository Layout

```text
├── data/
│   └── raw/
│       └── Groceries_dataset.csv
├── notebooks/
│   └── 02_groceries_apriori_association_rules.ipynb
├── docs/
│   └── Groceries-Market-Basket-Analysis-Apriori-Rules.pdf
├── .gitignore
├── requirements.txt
└── README.md

```

---

## 7. Setup & Reproduction

1. **Clone repository:**
```bash
git clone https://github.com/<your-username>/groceries-market-basket-analysis.git
cd groceries-market-basket-analysis

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run notebook:**
```bash
jupyter notebook notebooks/02_groceries_apriori_association_rules.ipynb

```

Technologies Used

- Python

- Pandas & NumPy (Vectorized Data Manipulation)

- Mlxtend (TransactionEncoder, apriori, association_rules)

- Google Colab & Git / GitHub
