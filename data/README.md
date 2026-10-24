# Data — `shop_ledgers.csv`

Real credit ledgers collected in the field from four shops, **anonymised**: customer names and shop names were removed and replaced by IDs. The map from IDs to names is kept privately and is not part of this repository.

| Shop ID | Type | Customers | Entries | Period | Source |
|---|---|---|---|---|---|
| S1 | pharmacy | 83 | 2,460 | 2022-05-15 → 2026-10-22 | PDF statements from an accounting app |
| S2 | grocery | 29 | 2,395 | 2025-02-10 → 2026-10-22 | PDF statements |
| S3 | grocery | 8 | 3,105 | 2024-12-01 → 2026-10-21 | handwritten notebook, digitised by hand |
| S4 | vegetable_stall | 24 | 2,120 | 2023-01-01 → 2026-10-22 | handwritten notebook, digitised by hand |

Credit sales only (cash sales were never written in these ledgers). Data collected on 2026-10-23.

## Columns
| Column | Meaning |
|---|---|
| customer_id | anonymous customer ID (C001…C144); one ledger = one customer account |
| store_id | anonymous shop ID (S1…S4) — used for validation only, never as a model feature |
| shop_type | pharmacy / grocery / vegetable_stall |
| seq | order of the entry in the ledger (may differ from date order by a few days) |
| date | YYYY-MM-DD |
| date_year_inferred | 1 = the year was not written in the notebook and was inferred; 0 = written in the source |
| kind | `debt` credit purchase · `payment` · `return` goods returned · `transfer` moved between two accounts |
| debt_amount | credit taken (≥ 0) |
| payment_amount | amount paid / returned / transferred (≥ 0) |
| running_balance_calc | balance after the entry (debt − payments so far; positive = customer owes) |

## Cleaning notes
- **Years in handwritten notebooks (≈51% of rows).** Entries were written as day/month. Years were inferred backwards from the last entry (2026): a drop of five or more months means the previous year. When the first entry carried a written year, the sequence went forward from it. Two notebooks (C119, C136) remain uncertain.
- Six impossible dates (29/30 February) were set to 28 February; one zero-amount entry was removed; a lab-test file found among the pharmacy statements was excluded.
- `kind` was derived from entry descriptions; in the handwritten notebooks every negative entry is a payment (returns cannot be distinguished there). Returns and transfers reduce the balance but are not counted as payments.
- The first entry of some handwritten notebooks is a large carried-over balance from an older notebook.
- Some balances become negative for a moment (returns, transfers, over-payment) — expected.
- Extraction check: PDF debit/credit totals matched the printed totals and balances matched the source.

## Use your own ledger
Any business can use the code with a CSV that has at least: `customer_id, store_id, date, kind, debt_amount, payment_amount` (`seq` recommended). `running_balance_calc` is recomputed by the code.

## Use and citation
Free for education and research. Please cite: *Fawzi AL-Nusairi, "Early Warning for Late and Unpaid Customer Credit", 2026, https://github.com/alnsyryfwzy55-source/shop-credit-early-warning*.
