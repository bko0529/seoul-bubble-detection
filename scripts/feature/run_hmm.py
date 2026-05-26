"""
HMM 레짐 탐지 실행 스크립트
입력:  data/processed/lstm_ae_scores.csv
출력:  data/processed/hmm_regime.csv
"""
import warnings, os, time
warnings.filterwarnings('ignore')
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from hmmlearn.hmm import GaussianHMM

plt.rcParams['font.family']        = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']  = False
plt.rcParams['figure.dpi']          = 110

print("=" * 60)
print("  HMM 레짐 탐지")
print("=" * 60)

SEED       = 42
ENC        = 'utf-8-sig'
SIDO_LIST  = ['서울', '경기', '인천']
SIDO_COLORS = {'서울': '#E63946', '경기': '#2A9D8F', '인천': '#E9C46A'}
REGIME_COLORS = {'버블': '#E63946', '과열': '#F4A261', '정상': '#ADB5BD'}
np.random.seed(SEED)

HMM_FEATURES = [
    '매매중위_YoY', '전세가율', 'PIR', 'base_rate',
    '매매_vol12', 'bsi_realestate', 'recon_error',
]

# ── 1. 데이터 로드
df = pd.read_csv('data/processed/lstm_ae_scores.csv', encoding=ENC)
df['ym_dt'] = pd.to_datetime(df['ym'].astype(str), format='%Y%m')

df_hmm = df.copy().sort_values(['시도', 'ym']).reset_index(drop=True)
# ffill/bfill 시도별
for sido in SIDO_LIST:
    idx = df_hmm[df_hmm['시도'] == sido].index
    df_hmm.loc[idx, HMM_FEATURES] = df_hmm.loc[idx, HMM_FEATURES].ffill().bfill().values

print(f"[1] 데이터 로드: {df_hmm.shape}  NaN={df_hmm[HMM_FEATURES].isnull().sum().sum()}")

# ── 2. HMM 유틸리티
def fit_hmm(X, n_components, n_iter=200):
    model = GaussianHMM(n_components=n_components, covariance_type='diag',
                        n_iter=n_iter, random_state=SEED)
    model.fit(X)
    return model

def compute_bic(model, X):
    n, d  = X.shape
    n_c   = model.n_components
    k     = (n_c-1)*2 + n_c * d * 2
    logL  = model.score(X) * n
    return -2 * logL + k * np.log(n)

def map_states(sub_df, states, n_states):
    stats = []
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0: continue
        yoy    = sub_df.loc[mask, '매매중위_YoY'].mean()
        jeonse = sub_df.loc[mask, '전세가율'].mean()
        pir    = sub_df.loc[mask, 'PIR'].mean()
        score  = yoy - jeonse * 0.3 + pir * 1.5
        stats.append({'state': s, 'score': score,
                      'avg_YoY': yoy, 'avg_jeonse': jeonse, 'avg_PIR': pir,
                      'count': mask.sum()})
    stats_df = pd.DataFrame(stats).sort_values('score', ascending=False)
    ranks = stats_df['state'].tolist()
    if n_states >= 3:
        names = ['버블', '과열'] + ['정상'] * (n_states - 2)
    else:
        names = ['과열', '정상']
    return {ranks[i]: names[i] for i in range(len(ranks))}, stats_df

# ── 3. 서울 BIC 최적화
seoul = df_hmm[df_hmm['시도'] == '서울'].sort_values('ym').reset_index(drop=True)
scaler_s = StandardScaler()
X_s = scaler_s.fit_transform(seoul[HMM_FEATURES].values)

bic_res = []
for n in range(2, 6):
    m   = fit_hmm(X_s, n)
    bic = compute_bic(m, X_s)
    bic_res.append({'n': n, 'BIC': round(bic, 2)})
    print(f"  n={n}  BIC={bic:.2f}")

bic_df = pd.DataFrame(bic_res)
bic_best = int(bic_df.loc[bic_df['BIC'].idxmin(), 'n'])
# 경제적 해석 가능성: 최대 4개 상태로 제한 (정상/과열/버블1/버블2)
N_STATES = min(bic_best, 4)
print(f"[2] BIC 최적={bic_best}  -> 해석 가능성 기준 n_states={N_STATES}")

# BIC 그래프
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(bic_df['n'], bic_df['BIC'], marker='o', color='#264653', lw=2, ms=8)
ax.axvline(N_STATES, color='#E63946', ls='--', lw=1.2, alpha=0.8, label=f'최적 n={N_STATES}')
ax.set_xlabel('상태 수'); ax.set_ylabel('BIC')
ax.set_title(f'HMM 상태 수 선택 — BIC (서울)', fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_hmm_01_bic.png', bbox_inches='tight')
plt.close()
print("  fig_hmm_01_bic.png 저장")

# ── 4. 시도별 HMM 학습
hmm_models, hmm_scalers, results_list = {}, {}, []
for sido in SIDO_LIST:
    sub = df_hmm[df_hmm['시도'] == sido].sort_values('ym').reset_index(drop=True)
    scaler = StandardScaler()
    X      = scaler.fit_transform(sub[HMM_FEATURES].values)
    hmm_scalers[sido] = scaler
    model  = fit_hmm(X, N_STATES)
    states = model.predict(X)
    hmm_models[sido] = model

    state_map, stats_df = map_states(sub, states, N_STATES)
    regime_labels = np.array([state_map[s] for s in states])

    print(f"\n[{sido}] 상태 매핑: {state_map}")
    for _, row in stats_df.iterrows():
        print(f"  S{int(row['state'])}: count={int(row['count'])}  YoY={row['avg_YoY']:.1f}  전세={row['avg_jeonse']:.1f}  PIR={row['avg_PIR']:.2f}")

    sub_r = sub[['ym','시도','ym_dt','bubble_label',
                 '매매중위가격','매매중위_YoY','전세가율','PIR',
                 'base_rate','매매_vol12','bsi_realestate',
                 'recon_error','anomaly']].copy()
    sub_r['hmm_state']  = states
    sub_r['hmm_regime'] = regime_labels
    results_list.append(sub_r)

result_df = pd.concat(results_list, ignore_index=True)
print(f"\n[3] 결과 shape: {result_df.shape}")

# ── 5. 전이 행렬 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, sido in zip(axes, SIDO_LIST):
    model  = hmm_models[sido]
    sub    = result_df[result_df['시도'] == sido]
    uniq   = sorted(sub['hmm_state'].unique())
    labels = [f"S{s}\n{sub[sub['hmm_state']==s]['hmm_regime'].mode()[0]}" for s in uniq]
    trans  = model.transmat_[np.ix_(uniq, uniq)]
    sns.heatmap(trans, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.5, ax=ax, annot_kws={'size':9})
    ax.set_title(f'{sido} 국면 전이 확률', fontsize=11)
    ax.set_xlabel('다음'); ax.set_ylabel('현재')
plt.suptitle('HMM 상태 전이 행렬 (시도별)', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/fig_hmm_02_transmat.png', bbox_inches='tight')
plt.close()
print("[4] fig_hmm_02_transmat.png 저장")

# ── 6. 레짐 시계열
fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
for ax, sido in zip(axes, SIDO_LIST):
    sub = result_df[result_df['시도'] == sido].sort_values('ym_dt')
    ax2 = ax.twinx()
    ax2.plot(sub['ym_dt'], sub['매매중위가격'], color=SIDO_COLORS[sido], lw=1.8, alpha=0.6)
    ax2.set_ylabel('매매중위(만원)', color=SIDO_COLORS[sido], fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    ax2.tick_params(axis='y', labelcolor=SIDO_COLORS[sido])
    for i, (_, row) in enumerate(sub.iterrows()):
        col = REGIME_COLORS.get(row['hmm_regime'], '#EEEEEE')
        ax.axvspan(row['ym_dt'] - pd.Timedelta(days=15),
                   row['ym_dt'] + pd.Timedelta(days=15), alpha=0.35, color=col, zorder=0)
    for _, row in sub[sub['bubble_label'] == 2].iterrows():
        ax.axvspan(row['ym_dt'] - pd.Timedelta(days=15),
                   row['ym_dt'] + pd.Timedelta(days=15),
                   alpha=0, edgecolor='#E63946', lw=2, fill=False, zorder=3)
    ax.set_yticks([])
    b_cnt = int((sub['bubble_label'] == 2).sum())
    hmm_b = int((sub['hmm_regime'] == '버블').sum())
    ax.set_title(f'{sido}  |  HMM 버블={hmm_b}개월  GT 버블={b_cnt}개월', fontsize=11)
    ax.grid(axis='x', alpha=0.2)
patches = [mpatches.Patch(color=c, label=r, alpha=0.6) for r, c in REGIME_COLORS.items()]
fig.legend(handles=patches, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5,1.01))
plt.suptitle('HMM 시장 레짐 시계열', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/fig_hmm_03_regime_ts.png', bbox_inches='tight')
plt.close()
print("[5] fig_hmm_03_regime_ts.png 저장")

# ── 7. 성능 평가
eval_df   = result_df[result_df['bubble_label'].notna()].copy()
y_true    = (eval_df['bubble_label'] == 2).astype(int)
y_hmm     = (eval_df['hmm_regime'] == '버블').astype(int)
y_ae      = eval_df['anomaly'].fillna(0).astype(int)

print("\n=== HMM 버블(2) 탐지 성능 ===")
print(classification_report(y_true, y_hmm, target_names=['비버블','버블'], digits=3, zero_division=0))
print("=== LSTM-AE 버블(2) 탐지 성능 ===")
print(classification_report(y_true, y_ae,  target_names=['비버블','버블'], digits=3, zero_division=0))

both = (y_hmm == 1) & (y_ae == 1)
bp = ((eval_df['bubble_label']==2) & both).sum() / max(1, both.sum())
br = ((eval_df['bubble_label']==2) & both).sum() / max(1, y_true.sum())
print(f"AND 앙상블 Precision={bp:.3f}  Recall={br:.3f}  (겹침={both.sum()}건)")

# 혼동 행렬
from sklearn.metrics import confusion_matrix
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (y_p, title) in zip(axes, [(y_hmm,'HMM'),(y_ae,'LSTM-AE')]):
    cm = confusion_matrix(y_true, y_p)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['정상(예측)','버블(예측)'],
                yticklabels=['정상(실제)','버블(실제)'])
    ax.set_title(f'{title} 버블 탐지', fontsize=11)
plt.suptitle('버블(2) 탐지: HMM vs LSTM-AE', fontsize=12)
plt.tight_layout()
plt.savefig('notebooks/fig_hmm_04_compare.png', bbox_inches='tight')
plt.close()
print("[6] fig_hmm_04_compare.png 저장")

# ── 8. 피처 분포
seoul_res = result_df[result_df['시도'] == '서울'].copy()
plot_features = ['매매중위_YoY','전세가율','PIR','base_rate','매매_vol12','bsi_realestate']
feat_labels   = ['가격YoY(%)','전세가율(%)','PIR(배)','기준금리(%)','변동성','BSI']
regime_order  = [r for r in ['정상','과열','버블'] if r in seoul_res['hmm_regime'].unique()]
colors_used   = [REGIME_COLORS[r] for r in regime_order]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat, label in zip(axes.flatten(), plot_features, feat_labels):
    data = [seoul_res[seoul_res['hmm_regime']==r][feat].dropna().values for r in regime_order]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='white', lw=2))
    for patch, color in zip(bp['boxes'], colors_used):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(regime_order, fontsize=10)
    ax.set_title(label, fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.suptitle('서울 국면별 피처 분포 (HMM)', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/fig_hmm_05_feature_dist.png', bbox_inches='tight')
plt.close()
print("[7] fig_hmm_05_feature_dist.png 저장")

# ── 9. 저장
base_df  = pd.read_csv('data/processed/lstm_ae_scores.csv', encoding=ENC)
merge_cols = result_df[['ym','시도','hmm_state','hmm_regime']]
save_df    = base_df.merge(merge_cols, on=['ym','시도'], how='left')
save_df.to_csv('data/processed/hmm_regime.csv', index=False, encoding='utf-8-sig')
print(f"\n[8] 저장: data/processed/hmm_regime.csv  {save_df.shape}")

print("\n" + "=" * 60)
print("  완료")
print("=" * 60)
