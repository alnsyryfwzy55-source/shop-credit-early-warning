"""Train both models, validate them honestly, choose money-based thresholds, save everything for production.

Usage:  python src/credit_system/train.py LEDGER.csv [--now YYYY-MM-DD] [--out models/]
Costs (edit to match the shop, all as a share of the purchase amount):
    LOSS_RATE      cost of goods lost when credit is never repaid           (default 0.85 = goods cost)
    MARGIN         profit lost if a good customer is pushed away            (default 0.15)
    """
import argparse, json, os, sys, warnings
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import features as F

LOSS_RATE, MARGIN = 0.85, 0.15


def make_model(seed=0):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()),
                     ('et', ExtraTreesClassifier(300, min_samples_leaf=10, max_features=0.5,
                                                 class_weight='balanced_subsample', random_state=seed, n_jobs=-1))])


def oof_predictions(A, reps=5):
    X, y, g = A[F.FEATURES], A.y.values, A.customer_id.values
    P = []
    for r in range(reps):
        p = np.zeros(len(A))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
            assert not set(g[tr]) & set(g[te]), 'customer leakage'
            p[te] = make_model(r).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        P.append(p)
    return np.mean(P, 0), [roc_auc_score(y, p) for p in P]


def evaluate(A, name):
    y = A.y.values
    p, aucs = oof_predictions(A)
    m = dict(task=name, snapshots=len(A), customers=int(A.customer_id.nunique()), positive_rate=round(y.mean(), 3),
             auc=round(float(np.mean(aucs)), 3), auc_sd=round(float(np.std(aucs)), 3), avg_precision=round(average_precision_score(y, p), 3),
             brier=round(brier_score_loss(y, p), 3), brier_baseline=round(y.mean() * (1 - y.mean()), 3),
             rule_days_since_purchase_auc=round(roc_auc_score(y, A.recency), 3))
    rng = np.random.default_rng(0); cust = A.customer_id.unique(); boot = []
    for _ in range(500):
        idx = np.concatenate([np.where(A.customer_id.values == c)[0] for c in rng.choice(cust, len(cust))])
        if len(set(y[idx])) > 1: boot.append(roc_auc_score(y[idx], p[idx]))
    m['auc_ci95'] = [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])]
    m['leave_one_store_out'] = {}
    for st in sorted(A.store_id.unique()):
        te = (A.store_id == st).values
        if te.sum() >= 30 and len(set(y[te])) > 1 and len(set(y[~te])) > 1:
            m['leave_one_store_out'][st] = round(roc_auc_score(y[te], make_model().fit(A.loc[~te, F.FEATURES], y[~te]).predict_proba(A.loc[te, F.FEATURES])[:, 1]), 3)
    cut = A.t.quantile(0.7); tr = (A.t < cut).values
    if len(set(y[~tr])) > 1:
        m['time_split'] = dict(train_before=str(cut.date()), auc=round(roc_auc_score(y[~tr], make_model().fit(A.loc[tr, F.FEATURES], y[tr]).predict_proba(A.loc[~tr, F.FEATURES])[:, 1]), 3),
                               rule_auc=round(roc_auc_score(y[~tr], A.recency[~tr]), 3))
    return m, p


def money_thresholds(A, p):
    """Money view on held-out predictions.
    (1) 'Pause new credit' tier: acting on a customer who would NOT pay saves LOSS_RATE x their next-180-day credit purchases;
        acting on a good customer costs MARGIN x the same amount. The high threshold maximises that saving.
    (2) Collection priority: expected loss = risk x current balance. We report how much of the balance that was really
        left unpaid sits in the top 20% of customers ranked by expected loss, versus ranking by balance or by the simple rule."""
    y, fd, bal = A.y.values, A.future_debt_180.values, A.balance.values
    def value(thr):
        act = p >= thr
        return float((act & (y == 1)) @ fd * LOSS_RATE - (act & (y == 0)) @ fd * MARGIN)
    grid = np.round(np.arange(0.30, 0.96, 0.01), 2); v = [value(t) for t in grid]
    hi = float(grid[int(np.argmax(v))]); best = max(v)
    k = int((p >= hi).sum()); act_rule = np.zeros(len(A), bool); act_rule[np.argsort(-A.recency.values)[:k]] = True
    rule_value = float((act_rule & (y == 1)) @ fd * LOSS_RATE - (act_rule & (y == 0)) @ fd * MARGIN)
    lost = bal * y; top = max(int(0.2 * len(A)), 1)
    def captured(score): return round(float(lost[np.argsort(-score)[:top]].sum() / max(lost.sum(), 1)), 3)
    return dict(threshold_medium=0.5, threshold_high=hi, share_high=round(float((p >= hi).mean()), 3), share_medium_or_high=round(float((p >= 0.5).mean()), 3),
                precision_high=round(float(((p >= hi) & (y == 1)).sum() / max(k, 1)), 3),
                pause_credit_saving=round(best), pause_credit_saving_rule_same_size=round(rule_value),
                future_credit_loss_without_system=round(float(fd[y == 1].sum()) * LOSS_RATE),
                unpaid_balance_total=round(float(lost.sum())),
                top20pct_captures_unpaid_balance=dict(expected_loss_model=captured(p * bal), balance_only=captured(bal),
                                                      rule_x_balance=captured(A.recency.values * bal), random=0.2),
                costs=dict(LOSS_RATE=LOSS_RATE, MARGIN=MARGIN))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ledger'); ap.add_argument('--now', default=None); ap.add_argument('--out', default='models')
    a = ap.parse_args()
    df = F.load_ledger(a.ledger); now = pd.Timestamp(a.now) if a.now else df.date.max().normalize() + pd.Timedelta(days=1)
    S = F.build_training_snapshots(df, now)
    os.makedirs(a.out, exist_ok=True); meta = dict(trained_on=str(date.today()), data_until=str(now.date()), features=F.FEATURES,
                                                   ledger_rows=len(df), customers=int(df.customer_id.nunique()))
    for task in ('risk', 'churn'):
        A = S[S.task == task].reset_index(drop=True)
        m, p = evaluate(A, task)
        if task == 'risk':
            m['money'] = money_thresholds(A, p)
            # shuffled-label sanity check: must be ~0.5, otherwise something leaks
            ys = np.random.default_rng(1).permutation(A.y.values); B = A.assign(y=ys); m['shuffled_label_auc'] = round(float(np.mean(oof_predictions(B, reps=2)[1])), 3)
        if task == 'risk':   # the churn model did not beat the simple rule, so production uses the rule for churn (documented in metadata)
            joblib.dump(make_model().fit(A[F.FEATURES], A.y.values), os.path.join(a.out, 'risk_model.joblib'))
        else:
            m['decision'] = 'not deployed: simple rule (days since purchase vs usual rhythm) is as good or better; scoring uses the rule'
        # reference feature distributions for drift monitoring
        m['feature_reference'] = {f: [float(q) for q in A[f].quantile([0.1, 0.25, 0.5, 0.75, 0.9])] for f in F.FEATURES}
        meta[task] = m
        print(json.dumps({k: v for k, v in m.items() if k != 'feature_reference'}, ensure_ascii=False, indent=1))
    json.dump(meta, open(os.path.join(a.out, 'metadata.json'), 'w'), ensure_ascii=False, indent=1)
    print('saved models + metadata to', a.out)


if __name__ == '__main__':
    main()
