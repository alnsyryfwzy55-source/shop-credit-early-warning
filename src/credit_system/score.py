"""Daily/weekly scoring: who needs follow-up, why, and what to do.

Usage:  python src/credit_system/score.py LEDGER.csv [--now YYYY-MM-DD] [--models models/] [--out follow_up.csv] [--log logs/]
Output: a follow-up list sorted by collection priority (customer IDs only), and a copy of the scores in logs/
        so monitor.py can later check them against what really happened.
The system never refuses anyone automatically: it ranks and explains; the shop owner decides.
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import features as F

CHURN_MULT, CHURN_MIN_DAYS = 3, 14      # rule: away 3x longer than usual and at least 14 days


def reasons(r):
    out = []
    d, g = r.raw_days_since_purchase, r.raw_usual_gap_days
    if g is not None and not pd.isna(g) and d >= max(2 * g, 7):
        out.append(f'لم يشترِ منذ {d} يومًا وعادته زيارة كل {g:g} يوم تقريبًا')
    elif d >= 30:
        out.append(f'لم يشترِ منذ {d} يومًا')
    if not r.raw_has_paid:
        out.append('لم يسدد أي مبلغ منذ بداية حسابه')
    else:
        pg = r.raw_usual_pay_gap_days
        if pg and r.raw_days_since_payment > 1.5 * pg and r.raw_days_since_payment >= 30:
            out.append(f'تأخر عن سداده المعتاد: آخر سداد قبل {r.raw_days_since_payment} يومًا وعادته كل {pg:g} يوم')
        elif r.raw_days_since_payment >= 60:
            out.append(f'آخر سداد قبل {r.raw_days_since_payment} يومًا')
    if r.raw_paid_share < 0.3 and r.raw_total_debt > 0:
        out.append(f'سدد {r.raw_paid_share:.0%} فقط من مجموع ما أخذه')
    if r.raw_events_prev_90 >= 4 and r.raw_events_last_90 < 0.5 * r.raw_events_prev_90:
        out.append(f'نشاطه انخفض: {r.raw_events_last_90} قيود في آخر 3 أشهر مقابل {r.raw_events_prev_90} قبلها')
    if not out:
        out.append('يشتري ويسدد بانتظام' if r.raw_has_paid else 'حساب حديث')
    return '؛ '.join(out[:3])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ledger'); ap.add_argument('--now', default=None)
    ap.add_argument('--models', default='models'); ap.add_argument('--out', default='follow_up.csv'); ap.add_argument('--log', default='logs')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.models, 'metadata.json'))); model = joblib.load(os.path.join(a.models, 'risk_model.joblib'))
    hi, mid = meta['risk']['money']['threshold_high'], meta['risk']['money']['threshold_medium']
    df = F.load_ledger(a.ledger); now = pd.Timestamp(a.now) if a.now else df.date.max().normalize() + pd.Timedelta(days=1)
    df = df[df.date <= now]
    C = F.current_snapshot(df, now)
    C['risk'] = np.nan
    el = C.risk_eligible.values
    if el.any():
        C.loc[el, 'risk'] = model.predict_proba(C.loc[el, F.FEATURES])[:, 1]
    away = C.raw_days_since_purchase
    C['churn_alert'] = (away >= np.maximum(CHURN_MULT * C.raw_usual_gap_days.fillna(30), CHURN_MIN_DAYS)) & (away <= F.ACTIVE_DAYS)
    dormant = (~C.risk_eligible) & (C.raw_balance > 0) & (away > F.ACTIVE_DAYS)

    def level(r):
        if r.risk_eligible:
            return 'عالٍ' if r.risk >= hi else 'متوسط' if r.risk >= mid else 'منخفض'
        return 'متوقف بدين' if r.raw_balance > 0 and r.raw_days_since_purchase > F.ACTIVE_DAYS else 'لا دين قائم'
    C['level'] = C.apply(level, axis=1)
    ACTION = {'عالٍ': 'تواصل مباشر الآن، وإيقاف الآجل الجديد مؤقتًا حتى يسدد جزءًا',
              'متوسط': 'تذكير ودّي، وعدم رفع سقف الدين',
              'منخفض': 'لا إجراء',
              'متوقف بدين': 'متابعة تحصيل: غائب أكثر من 3 أشهر وعليه رصيد (خارج نطاق النموذج)',
              'لا دين قائم': 'لا إجراء'}
    C['action'] = C.level.map(ACTION)
    C['expected_loss'] = np.where(C.risk_eligible, C.risk * C.raw_balance.clip(lower=0), np.where(dormant, C.raw_balance, 0.0))
    C['why'] = C.apply(reasons, axis=1)
    # active customers (model) first by expected loss, then dormant debtors by balance, then the rest
    C['group'] = np.select([C.risk_eligible, dormant], [0, 1], 2)
    C = C.sort_values(['group', 'expected_loss'], ascending=[True, False])
    C['priority'] = np.arange(1, len(C) + 1)
    out = C[['priority', 'customer_id', 'store_id', 'level', 'risk', 'raw_balance', 'expected_loss', 'churn_alert', 'why', 'action']].rename(columns={
        'priority': 'أولوية_التحصيل', 'customer_id': 'رقم_العميل', 'store_id': 'المحل', 'level': 'المستوى', 'risk': 'احتمال_عدم_السداد_180_يومًا',
        'raw_balance': 'الرصيد', 'expected_loss': 'الخسارة_المتوقعة', 'churn_alert': 'تنبيه_توقف_عن_الشراء', 'why': 'الأسباب', 'action': 'الإجراء_المقترح'})
    out['احتمال_عدم_السداد_180_يومًا'] = out['احتمال_عدم_السداد_180_يومًا'].round(2)
    out['الخسارة_المتوقعة'] = out['الخسارة_المتوقعة'].round(0)
    out['تنبيه_توقف_عن_الشراء'] = out['تنبيه_توقف_عن_الشراء'].map({True: 'نعم', False: ''})
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    os.makedirs(a.log, exist_ok=True)
    C.assign(scored_at=str(now.date()))[['scored_at', 'customer_id', 'store_id', 'risk_eligible', 'risk', 'level', 'raw_balance', 'churn_alert'] + F.FEATURES] \
        .to_csv(os.path.join(a.log, f'scores_{now.date()}.csv'), index=False)
    print(f'scored {len(C)} customers as of {now.date()} (model trained {meta["trained_on"]}, data until {meta["data_until"]})')
    print(out['المستوى'].value_counts().to_string())
    print('churn alerts:', int(C.churn_alert.sum()), '| total balance', round(C.raw_balance.clip(lower=0).sum()), '| expected loss', round(C.expected_loss.sum()))
    print('saved', a.out)


if __name__ == '__main__':
    main()
