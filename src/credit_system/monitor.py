"""Monthly health check: did past scores come true? Has the data drifted? Is it time to retrain?

Usage:  python src/credit_system/monitor.py LEDGER.csv [--now YYYY-MM-DD] [--models models/] [--log logs/]
For every score log whose 180-day window has finished, compares predictions with what really happened.
"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import features as F

AUC_ALERT, DRIFT_ALERT, MAX_MODEL_AGE_DAYS = 0.65, 0.35, 120
# counts and account age grow naturally as accounts get older, so they are not used as drift signals
DRIFT_FEATURES = [f for f in F.FEATURES if f not in ('n_ev', 'age_days', 'n_pay')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ledger'); ap.add_argument('--now', default=None)
    ap.add_argument('--models', default='models'); ap.add_argument('--log', default='logs')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.models, 'metadata.json')))
    df = F.load_ledger(a.ledger); now = pd.Timestamp(a.now) if a.now else df.date.max().normalize() + pd.Timedelta(days=1)
    df = df[df.date <= now]
    alerts = []
    # 1) outcome check for finished windows
    for path in sorted(glob.glob(os.path.join(a.log, 'scores_*.csv'))):
        s = pd.read_csv(path); d = pd.Timestamp(s.scored_at.iloc[0])
        s = s[s.risk_eligible & s.risk.notna()]
        if d + pd.Timedelta(days=F.RISK_HORIZON) > now or len(s) == 0:
            print(f'{d.date()}: window not finished yet ({(d + pd.Timedelta(days=F.RISK_HORIZON)).date()})'); continue
        win = df[(df.date > d) & (df.date <= d + pd.Timedelta(days=F.RISK_HORIZON))]
        paid = set(win[win.kind == 'payment'].customer_id)
        s['y'] = (~s.customer_id.isin(paid)).astype(int)
        line = f'{d.date()}: {len(s)} customers, real non-payers {s.y.mean():.0%}'
        if s.y.nunique() > 1:
            auc = roc_auc_score(s.y, s.risk); line += f', AUC={auc:.3f}, Brier={brier_score_loss(s.y, s.risk):.3f}'
            if auc < AUC_ALERT: alerts.append(f'AUC {auc:.2f} below {AUC_ALERT} for scores of {d.date()}')
        for lv in ('عالٍ', 'متوسط', 'منخفض'):
            g = s[s.level == lv]
            if len(g): line += f' | {lv}: {len(g)} → {g.y.mean():.0%} did not pay'
        print(line)
    # 2) drift: share of current customers outside the training 10-90% range, per feature
    C = F.current_snapshot(df, now); C = C[C.risk_eligible]
    if len(C):
        ref = meta['risk']['feature_reference']; drift = {}
        for f in DRIFT_FEATURES:
            lo, hi_ = ref[f][0], ref[f][4]
            drift[f] = round(float(((C[f] < lo) | (C[f] > hi_)).mean()), 2)   # ~0.20 expected without drift
        worst = sorted(drift.items(), key=lambda x: -x[1])[:4]
        print('drift (share outside training 10-90% range, ~0.20 is normal):', dict(worst))
        alerts += [f'feature {f} drifted ({v:.0%} outside normal range)' for f, v in worst if v > DRIFT_ALERT]
    # 3) model age
    age = (now - pd.Timestamp(meta['data_until'])).days
    if age > MAX_MODEL_AGE_DAYS: alerts.append(f'model trained on data until {meta["data_until"]} ({age} days old)')
    print('\nRETRAIN RECOMMENDED: ' + ' ; '.join(alerts) if alerts else '\nOK: no retraining needed')


if __name__ == '__main__':
    main()
