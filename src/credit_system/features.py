"""Feature engineering shared by training, scoring and monitoring.

Input ledger CSV columns (see data/README.md):
    customer_id, store_id, seq, date (YYYY-MM-DD), kind (debt|payment|return|transfer),
    debt_amount, payment_amount, running_balance_calc
Every feature at cutoff t uses ONLY rows dated <= t.
"""
import numpy as np
import pandas as pd

L = np.log1p
FEATURES = ['recency', 'rec_rel', 'since_pay', 'n_pay', 'age_days', 'n_ev', 'bal_frac', 'pay_ratio',
            'ev_trend', 'expected_pay_overdue', 'pay_cov90', 'debt_trend']

RISK_HORIZON = 180      # target 1: no payment at all in the next 180 days
CHURN_HORIZON = 60      # target 2: no purchase at all in the next 60 days
ACTIVE_DAYS = 90        # risk model applies to customers with balance > 0 who bought within 90 days
CHURN_ACTIVE_DAYS = 60  # churn model applies to customers who bought within 60 days
MIN_EVENTS = 3


def load_ledger(path):
    df = pd.read_csv(path, parse_dates=['date'])
    need = {'customer_id', 'store_id', 'date', 'kind', 'debt_amount', 'payment_amount'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'ledger is missing columns: {sorted(missing)}')
    df[['debt_amount', 'payment_amount']] = df[['debt_amount', 'payment_amount']].fillna(0.0)
    sort_cols = ['customer_id', 'seq'] if 'seq' in df.columns else ['customer_id', 'date']
    df = df.sort_values(sort_cols, kind='stable').reset_index(drop=True)
    # recompute the balance so the system never depends on a pre-computed column
    df['running_balance_calc'] = (df.debt_amount - df.payment_amount).groupby(df.customer_id).cumsum()
    return df


def features_at(h, t, d0):
    """h = this customer's rows with date <= t (date-sorted). Returns (model features, raw values for explanations)."""
    pay = h[h.kind == 'payment']
    td = max(h.debt_amount.sum(), 1.0)
    bal = float(h.running_balance_calc.iloc[-1])
    r90 = h[h.date > t - pd.Timedelta(days=90)]
    p90 = h[(h.date <= t - pd.Timedelta(days=90)) & (h.date > t - pd.Timedelta(days=180))]
    lastpay = pay.date.max() if len(pay) else d0
    gaps = h.date.diff().dt.days.abs().dropna()
    med_gap = float(gaps.median()) if len(gaps) else 30.0
    rec = int((t - h.date.max()).days)
    since_pay = int((t - lastpay).days)
    pay_gap = float(pay.date.diff().dt.days.abs().dropna().mean()) if len(pay) > 1 else 400.0
    f = dict(recency=L(rec), rec_rel=L(rec / max(med_gap, 1)), since_pay=L(since_pay), n_pay=len(pay),
             age_days=L((t - d0).days), n_ev=L(len(h)), bal_frac=bal / td, pay_ratio=pay.payment_amount.sum() / td,
             ev_trend=(len(r90) + 1) / (len(p90) + 1),
             expected_pay_overdue=L(max(since_pay - pay_gap, 0) + 1),
             pay_cov90=r90.payment_amount.sum() / max(r90.debt_amount.sum(), 1),
             debt_trend=L(r90.debt_amount.sum() + 1) - L(p90.debt_amount.sum() + 1))
    ud = pd.Series(h.date.dt.normalize().unique()).sort_values().diff().dt.days.dropna()   # rhythm between distinct visit days
    raw = dict(days_since_purchase=rec, usual_gap_days=round(float(ud.median()), 1) if len(ud) else None, days_since_payment=since_pay,
               has_paid=len(pay) > 0, n_payments=len(pay), balance=round(bal, 2), total_debt=round(td, 2),
               paid_share=round(pay.payment_amount.sum() / td, 3), usual_pay_gap_days=round(pay_gap, 1) if len(pay) > 1 else None,
               events_last_90=len(r90), events_prev_90=len(p90), debt_last_90=round(r90.debt_amount.sum(), 2),
               account_age_days=int((t - d0).days))
    return f, raw


def risk_eligible(h, t):
    return len(h) >= MIN_EVENTS and h.running_balance_calc.iloc[-1] > 0 and (t - h.date.max()).days <= ACTIVE_DAYS


def churn_eligible(h, t):
    return len(h) >= MIN_EVENTS and (t - h.date.max()).days <= CHURN_ACTIVE_DAYS


def build_training_snapshots(df, now, step=60):
    """Historical snapshots with both labels. A snapshot needs its full future window to be observed before `now`."""
    rows = []
    for cid, g in df.groupby('customer_id'):
        g = g.sort_values('date', kind='stable')
        d0 = g.date.min()
        t = d0 + pd.Timedelta(days=step)
        while t <= now - pd.Timedelta(days=CHURN_HORIZON):
            h = g[g.date <= t]
            if len(h) >= MIN_EVENTS:
                f, raw = features_at(h, t, d0)
                base = dict(customer_id=cid, store_id=g.store_id.iloc[0], t=t, **f,
                            future_debt_180=float(g[(g.date > t) & (g.date <= t + pd.Timedelta(days=RISK_HORIZON))].debt_amount.sum()),
                            balance=raw['balance'])
                if risk_eligible(h, t) and t <= now - pd.Timedelta(days=RISK_HORIZON):
                    fut = g[(g.date > t) & (g.date <= t + pd.Timedelta(days=RISK_HORIZON))]
                    rows.append(dict(base, task='risk', y=int(not (fut.kind == 'payment').any())))
                if churn_eligible(h, t):
                    fut = g[(g.date > t) & (g.date <= t + pd.Timedelta(days=CHURN_HORIZON))]
                    rows.append(dict(base, task='churn', y=int((fut.debt_amount > 0).sum() == 0)))
            t += pd.Timedelta(days=step)
    return pd.DataFrame(rows)


def current_snapshot(df, now):
    """One row per customer as of `now`, with raw values for explanations."""
    rows = []
    for cid, g in df.groupby('customer_id'):
        g = g.sort_values('date', kind='stable')
        h = g[g.date <= now]
        if len(h) < MIN_EVENTS:
            continue
        f, raw = features_at(h, now, g.date.min())
        rows.append(dict(customer_id=cid, store_id=g.store_id.iloc[0], **f, **{'raw_' + k: v for k, v in raw.items()},
                         risk_eligible=risk_eligible(h, now), churn_eligible=churn_eligible(h, now)))
    return pd.DataFrame(rows)
