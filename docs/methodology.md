# Methodology

## 1. Data
| Shop | Type | Customers | Entries | Source |
|---|---|---|---|---|
| S1 | Pharmacy | 83 | 2,460 | PDF statements from a free accounting app |
| S2 | Grocery | 29 | 2,395 | PDF statements |
| S3 | Grocery | 8 | 3,105 | Handwritten ledger, digitised manually |
| S4 | Vegetable store | 24 | 2,120 | Handwritten ledger, digitised manually |

Credit sales only (cash sales are not recorded). Entry types: debt 9,407 · payment 631 · return 36 · transfer 6.
Customer and shop names were removed and replaced by IDs (C001…C144, S1…S4); the anonymised ledgers are published in `data/shop_ledgers.csv`. The ID-to-name map is kept privately.

**Cleaning decisions**
- Handwritten ledgers recorded day/month without year (≈51% of rows); years were inferred going backwards from the last entry (2026) and when a month drops by five or more, the year changes. Two ledgers remain uncertain.
- Six impossible dates (29/30 February) were set to 28 February; one zero-amount entry removed; a lab-test file found among pharmacy statements was excluded.
- Payment vs. return vs. transfer was derived from entry descriptions; returns and transfers reduce the balance but are not counted as payments.
- Extraction check: PDF debit and credit totals matched the printed totals; balances matched the source.

## 2. Target (no leakage by construction)
A **snapshot** is taken every 60 days from each customer's first entry. It is kept when the customer has ≥ 3 entries, a positive balance, and bought within the last 90 days.
- **Features:** only entries dated ≤ t.
- **Label:** 1 if there is **no payment in (t, t + 180 days]**.
- Snapshots whose 180-day window is not complete before the collection date are dropped.

Result: 470 snapshots from 119 customers, 45% positive.

## 3. Features (12)
| Feature | Meaning |
|---|---|
| recency | days since last entry (log) |
| rec_rel | absence relative to the customer's own usual gap |
| since_pay | days since last payment |
| n_pay | number of payments |
| age_days | account age |
| n_ev | number of entries |
| bal_frac | balance / total credit |
| pay_ratio | total paid / total credit |
| ev_trend | entries in last 90 days vs the 90 before |
| expected_pay_overdue | days beyond the customer's usual payment interval |
| pay_cov90 | paid vs bought in the last 90 days |
| debt_trend | credit volume trend |

Shop type and shop ID are **not** features. Features alone can identify the shop (AUC 0.97), so we also report within-shop AUC and leave-one-shop-out results. A feature describing an "opening balance" was removed because it mostly encoded the shop.

## 4. Validation
- **Customer-grouped stratified 5-fold CV**, repeated; no customer in both train and test (asserted in code).
- **Leave-one-shop-out**, **time split** (train on earlier snapshots, test on later ones).
- **Held-out set:** 25% of customers (28) set aside before the 480-configuration experiment grid; the winners were tested on them once.
- **Shuffled-label check** (should be ≈ 0.5) and a check that deleting all future rows leaves features unchanged.
- **Metrics:** AUC and average precision (ranking, threshold-free); balanced accuracy and F1 at thresholds chosen on development data; Brier score for probability quality. Plain accuracy alone is misleading here.
- **Bootstrap by customer** for 95% confidence intervals.

## 5. Unsupervised learning
Customer-level features at the collection date, standardised; K-Means for k = 2…6 chosen by silhouette; PCA to 2 components for visualisation. Segments found: regular payers (≈80% repaid), active customers with growing debt, customers who left long ago with most of their debt unpaid, and a few very heavy buyers.

## 6. From prediction to action
- **Expected loss = P(no payment in 180 days) × current balance** orders the follow-up list.
- Levels: high (≥ money-optimal threshold, 0.79 on the real data), medium (≥ 0.5), low; customers absent > 90 days with a balance are listed separately ("left with debt").
- Thresholds are chosen to maximise money saved under configurable costs (goods cost 85%, margin 15%). Pausing new credit saves little (customers who stop paying also stop buying), so the main value is **prioritising collection**.
- A churn model ("no purchase in 60 days", AUC 0.78) was **not deployed**: the simple rule matched it (0.785).

## 7. Responsible use
No automatic refusals; log every decision (otherwise the data becomes biased towards customers who were granted credit); monitor any new shop for three months; retrain every 120 days or when `monitor.py` asks; names stay with the shop owner.
