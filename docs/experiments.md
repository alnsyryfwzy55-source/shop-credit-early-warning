# All experiments (real data)

Every experiment was scored on **real shop customers the model never saw in training**. External or synthetic data was added to training folds only.

## 1. Problem framing
| Framing | Result | Verdict |
|---|---|---|
| Classify a **new** customer (loyal / late / fled) from the first 3–8 entries | accuracy 0.43–0.47 vs 0.36–0.40 by chance; binary AUC 0.49–0.56 on PDF shops | not predictable |
| First "fled vs rest" result, AUC 0.69 | leaked shop identity (inferred year, first-entry month) | discarded |
| Including dormant customers | AUC 0.94 — misleading, a dormant customer stays dormant | discarded |
| **Active debtor: no payment in next 180 days** | **AUC 0.77** | adopted |
| Same, horizon 30 / 60 / 90 / 120 days | 0.708 / 0.725 / 0.747 / 0.743 | 180 days best |
| Churn: no purchase in next 60 days | 0.78 (rule 0.785) | rule used instead |

## 2. Models (customer-grouped CV, 12 core features, development customers)
| Model | AUC |
|---|---|
| Extra Trees | 0.774 |
| Soft voting ET + LR + SVM | 0.773 |
| SVM (RBF) | 0.766 |
| Random Forest | 0.760 |
| Logistic Regression | 0.759 |
| LDA | 0.758 |
| KNN | 0.743 |
| Gradient Boosting | 0.738 |
| HistGradientBoosting | 0.727 |
| Naive Bayes | 0.725 |
| AdaBoost | 0.723 |
| MLP | 0.706 |

- Nested hyper-parameter search: no gain (0.740–0.768).
- Stacking (2 to 9 base models, group-aware): 0.763–0.767.
- Error analysis: mean rank correlation between models 0.86; a median 67% of any two models' errors are shared; 11% of snapshots are misclassified by all nine models. Ensembles cannot fix errors that all models share.

## 3. Extra information
| Addition | AUC | Verdict |
|---|---|---|
| 12 extra behavioural features | ≈ −0.01 in almost every setting | no |
| Entry descriptions (cash loans, named payer, tobacco, carried balances; 92 customers) | ET 0.790 → 0.799, LR 0.781 → 0.770 | inconsistent |
| Customer age (57 customers) | ET 0.671 → 0.670, LR 0.634 → 0.619 | no |
| Housing / occupation (34 customers) | too few to test | — |

## 4. External and synthetic data (training only)
| Source | Combined AUC | Source-only model on our customers | Verdict |
|---|---|---|---|
| Reference: our data only | 0.769 | — | — |
| SMOTE ×1 / ×3, jitter, Gaussian synthetic | 0.755–0.773 | — | no gain |
| Generated ledgers (200 synthetic customers) | 0.675–0.743 | 0.481 | worse |
| Give Me Some Credit (Kaggle, 150k), 4 shared features | 0.51–0.58 | 0.505 | worse |
| B2B invoices (Kaggle, 50k) converted to ledgers | 0.742–0.757 | 0.694 | worse |
| Domain adaptation: most shop-like invoices | 0.748–0.758 | — | worse |
| OneFi (Nigerian lender, 4,359 customers) | 0.770–0.772 | 0.57 | neutral |
| Kenya Financial Diaries (household shop credit) | 0.750–0.756 | 0.689 | worse |
| Kenya shuffled to random pseudo-customers | 0.727 | — | worse |
| Kenya split into pseudo-customers | 0.772 (but creates false "stopped paying" labels: 63% positive) | — | invalid |

Interesting finding: models trained **only** on Kenyan households or on B2B invoices rank our customers at AUC ≈ 0.69 — "stopping begets stopping" is a general pattern — but they do not add to a model trained on local data.

## 5. Selection bias check (480-configuration grid)
91 development customers for the grid, 28 held-out customers tested once.

| Configuration | Dev AUC | Held-out AUC |
|---|---|---|
| Reference (Extra Trees, no extra data) | 0.774 | 0.698 |
| Best on dev: HistGB + Kenya + invoices | 0.779 | 0.654 |
| ET + all sources 25% | 0.779 | 0.664 |
| ET + OneFi 100% | 0.778 | 0.704 |
| Rule: days since last purchase | — | 0.712 |

The configurations that looked best on development data fell on fresh customers: picking the maximum of many tries inflates the number.

## 6. Learning curve and reality check
- Training customers 23 → 47 → 71 → 95: AUC 0.731 → 0.755 → 0.765 → 0.772 (≈ +0.02 per doubling).
- Shop owners' labels (committed / late / fled, 34 customers) vs model risk: AUC 0.67–0.73 overall, 0.75 in the vegetable store; days since last purchase: 0.79–0.84; unpaid share: 0.41–0.52. Owners judge by absence, as the model does.
- Backtest: a model trained on data up to 2026-03-01 scored 35 active customers; 180 days later AUC 0.91 (low level: 8% did not pay, medium: 70%). Small sample.
