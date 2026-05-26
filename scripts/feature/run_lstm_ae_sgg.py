"""
LSTM Autoencoder 버블 이상 탐지 — 구 단위 (SGG)
─────────────────────────────────────────────────
입력:  data/processed/features_sgg_final.csv
출력:  data/processed/lstm_ae_sgg_scores.csv
       models/lstm_ae_sgg_서울.pth
       models/lstm_ae_sgg_경기.pth
       models/lstm_ae_sgg_인천.pth
       notebooks/fig_ae_sgg_01_anomaly_freq.png
       notebooks/fig_ae_sgg_02_error_dist.png

학습 전략:
  - 정규화: 구별 MinMaxScaler (정상 데이터로만 fit)
  - 시퀀스: 구별 독립 생성 (시계열 연속성 보장)
  - 모델:   시도별 1개 (서울/경기/인천) — 같은 시도 내 구들 공유
  - HPO:    Optuna 10 trials × 50 epochs
  - 임계값: 시도별 정상 시퀀스 재구성 오차 95th percentile
"""
import sys, warnings, os, random, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.makedirs("models", exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']       = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi']        = 110

print("=" * 60)
print("  LSTM Autoencoder 버블 이상 탐지 — 구 단위 (SGG)")
print("=" * 60)

# ── 상수
SEED      = 42
DEVICE    = torch.device("cpu")
ENC       = "utf-8-sig"
SEQ_LEN   = 12
SIDO_LIST = ["서울", "경기", "인천"]
SIDO_COLORS = {"서울": "#E63946", "경기": "#2A9D8F", "인천": "#E9C46A"}

FEATURES = [
    "매매중위_MoM", "매매중위_YoY", "매매_vol12", "price_zscore_24m",
    "전세가율_cap", "is_kangtong", "거래량_ratio",
    "base_rate", "mortgage_rate_chg_yoy", "m2_yoy_pct", "cpi_yoy_pct",
    "bsi_realestate", "PIR", "미분양_yoy", "가계대출비중_yoy",
]
N_FEATURES = len(FEATURES)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════
# 1. 데이터 로드 & 결측 처리
# ══════════════════════════════════════════════════════════════
print("\n[1] 데이터 로드")
df = pd.read_csv("data/processed/features_sgg_final.csv", encoding=ENC)
df["ym"] = df["ym"].astype(int)
df_filled = df.copy()
df_filled[FEATURES] = df_filled[FEATURES].fillna(0)
print(f"  shape: {df.shape}  |  구={df['구'].nunique()}개  기간={df['ym'].min()}~{df['ym'].max()}")
print(f"  피처 NaN(처리 후): {df_filled[FEATURES].isnull().sum().sum()}")


# ══════════════════════════════════════════════════════════════
# 2. 구별 정규화 (정상 데이터로만 fit)
# ══════════════════════════════════════════════════════════════
print("\n[2] 구별 MinMaxScaler 정규화")
scalers_gu    = {}
df_scaled_list = []

for (sido_name, gu), grp in df_filled.groupby(["시도", "구"], sort=False):
    grp = grp.sort_values("ym").reset_index(drop=True)
    nm  = grp["bubble_label"] == 0
    sc  = MinMaxScaler()
    sc.fit(grp.loc[nm if nm.sum() >= 10 else grp.index, FEATURES])
    scalers_gu[(sido_name, gu)] = sc
    s = grp.copy()
    s[FEATURES] = sc.transform(grp[FEATURES])
    df_scaled_list.append(s)

df_scaled = pd.concat(df_scaled_list, ignore_index=True)
print(f"  완료: {df_scaled.shape}")


# ══════════════════════════════════════════════════════════════
# 3. 시퀀스 생성
# ══════════════════════════════════════════════════════════════
print("\n[3] 시퀀스 생성 (SEQ_LEN={})".format(SEQ_LEN))

def make_sequences(data_df, seq_len, normal_only=True):
    seqs, labels, yms, sidos, gus = [], [], [], [], []
    for (sido_name, gu), grp in data_df.groupby(["시도", "구"], sort=False):
        sub  = grp.sort_values("ym").reset_index(drop=True)
        vals = sub[FEATURES].values.astype(np.float32)
        lbls = sub["bubble_label"].fillna(-1).values
        yms_ = sub["ym"].values
        for i in range(len(sub) - seq_len + 1):
            wl = lbls[i:i+seq_len]
            if normal_only and not np.all(wl == 0):
                continue
            seqs.append(vals[i:i+seq_len])
            labels.append(lbls[i+seq_len-1])
            yms.append(yms_[i+seq_len-1])
            sidos.append(sido_name)
            gus.append(gu)
    return np.array(seqs), np.array(labels), np.array(yms), sidos, gus

train_seqs, train_labels, train_yms, train_sidos, train_gus = make_sequences(df_scaled, SEQ_LEN, True)
all_seqs,   all_labels,   all_yms,   all_sidos,   all_gus   = make_sequences(df_scaled, SEQ_LEN, False)

print(f"  학습(정상) 시퀀스: {train_seqs.shape}")
print(f"  전체    시퀀스: {all_seqs.shape}")
for s in SIDO_LIST:
    tr_n = sum(1 for x in train_sidos if x == s)
    al_n = sum(1 for x in all_sidos   if x == s)
    print(f"    {s}: 학습={tr_n}  전체={al_n}")

# 시도별 인덱스
train_sido_idx = {s: np.where(np.array(train_sidos) == s)[0] for s in SIDO_LIST}
all_sido_idx   = {s: np.where(np.array(all_sidos)   == s)[0] for s in SIDO_LIST}


# ══════════════════════════════════════════════════════════════
# 4. 모델 정의
# ══════════════════════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = torch.tensor(seqs, dtype=torch.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]

class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features, hidden_size, latent_dim, num_layers, dropout):
        super().__init__()
        dp = dropout if num_layers > 1 else 0.0
        self.enc_lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, dropout=dp)
        self.enc_fc   = nn.Linear(hidden_size, latent_dim)
        self.dec_fc   = nn.Linear(latent_dim, hidden_size)
        self.dec_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dp)
        self.out_fc   = nn.Linear(hidden_size, n_features)

    def encode(self, x):
        _, (h, _) = self.enc_lstm(x)
        return self.enc_fc(h[-1])

    def decode(self, z, seq_len):
        d = self.dec_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.dec_lstm(d)
        return self.out_fc(out)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z, x.size(1)), z

def recon_loss(recon, x): return nn.functional.mse_loss(recon, x)

def train_one_epoch(model, loader, opt):
    model.train(); total = 0.0
    for x in loader:
        x = x.to(DEVICE); recon, _ = model(x)
        loss = recon_loss(recon, x)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); total = 0.0
    for x in loader:
        x = x.to(DEVICE); recon, _ = model(x)
        total += recon_loss(recon, x).item() * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def get_errors(model, loader):
    model.eval(); errors = []
    for x in loader:
        x = x.to(DEVICE); recon, _ = model(x)
        mse = ((recon - x)**2).mean(dim=-1).mean(dim=-1)
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)

def train_model(params, ds, n_epochs=120, patience=15, val_ratio=0.15, verbose=False):
    torch.manual_seed(SEED)
    nv = max(1, int(len(ds)*val_ratio)); nt = len(ds)-nv
    tr_ds, vl_ds = random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(SEED))
    tr_ld = DataLoader(tr_ds, batch_size=params["batch_size"], shuffle=True)
    vl_ld = DataLoader(vl_ds, batch_size=params["batch_size"])
    model = LSTMAutoEncoder(N_FEATURES, params["hidden_size"], params["latent_dim"],
                             params["num_layers"], params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, n_epochs+1):
        tl = train_one_epoch(model, tr_ld, opt)
        vl = evaluate(model, vl_ld)
        sch.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: print(f"    Early stop @ epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, best_val


# ══════════════════════════════════════════════════════════════
# 5. 시도별 HPO + 학습 + 스코어링
# ══════════════════════════════════════════════════════════════
t0          = time.time()
result_rows = []

for sido_name in SIDO_LIST:
    print(f"\n{'='*55}")
    print(f"  [{sido_name}] LSTM-AE")
    print(f"{'='*55}")

    tr_idx   = train_sido_idx[sido_name]
    al_idx   = all_sido_idx[sido_name]
    tr_seqs_ = train_seqs[tr_idx]
    al_seqs_ = all_seqs[al_idx]
    print(f"  학습 시퀀스: {len(tr_seqs_)}  전체: {len(al_seqs_)}")

    train_ds = SeqDataset(tr_seqs_)
    all_ds_  = SeqDataset(al_seqs_)

    # ── HPO
    def objective(trial):
        p = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "latent_dim":  trial.suggest_categorical("latent_dim",  [8, 16, 32]),
            "num_layers":  trial.suggest_int("num_layers", 1, 2),
            "dropout":     trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
            "lr":          trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "batch_size":  trial.suggest_categorical("batch_size", [16, 32]),
        }
        _, vl = train_model(p, train_ds, n_epochs=50, patience=8)
        return vl

    t1 = time.time()
    print(f"  Optuna HPO 시작 (10 trials) ...")
    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"  HPO 완료: {time.time()-t1:.1f}s  val_loss={study.best_value:.6f}")
    print(f"  최적 파라미터: {best_params}")

    # ── 최종 학습
    t2 = time.time()
    print(f"  최종 학습 (120 epochs, early stop patience=15) ...")
    final_model, final_val = train_model(
        best_params, train_ds, n_epochs=120, patience=15, verbose=True)
    print(f"  학습 완료: {time.time()-t2:.1f}s  val_loss={final_val:.5f}")

    # ── 임계값
    tr_loader = DataLoader(train_ds, batch_size=64)
    al_loader = DataLoader(all_ds_,  batch_size=64)
    tr_errors = get_errors(final_model, tr_loader)
    al_errors = get_errors(final_model, al_loader)
    threshold = float(np.percentile(tr_errors, 95))
    print(f"  임계값(95th): {threshold:.5f}  |  정상 평균={tr_errors.mean():.5f}")

    # ── 모델 저장
    torch.save({
        "model_state": final_model.state_dict(),
        "params":      best_params,
        "features":    FEATURES,
        "seq_len":     SEQ_LEN,
        "threshold":   threshold,
    }, f"models/lstm_ae_sgg_{sido_name}.pth")
    print(f"  models/lstm_ae_sgg_{sido_name}.pth 저장")

    # ── 결과 수집
    al_sidos_sub = [all_sidos[i] for i in al_idx]
    al_gus_sub   = [all_gus[i]   for i in al_idx]
    al_yms_sub   = [all_yms[i]   for i in al_idx]
    al_lbls_sub  = [all_labels[i] for i in al_idx]

    for i in range(len(al_seqs_)):
        result_rows.append({
            "ym":           al_yms_sub[i],
            "시도":          al_sidos_sub[i],
            "구":            al_gus_sub[i],
            "recon_error":  float(al_errors[i]),
            "bubble_label": float(al_lbls_sub[i]) if al_lbls_sub[i] >= 0 else np.nan,
            "anomaly":      int(al_errors[i] > threshold),
        })

# ══════════════════════════════════════════════════════════════
# 6. 결과 DataFrame
# ══════════════════════════════════════════════════════════════
print("\n[6] 결과 수집 & 성능 평가")
result_df = pd.DataFrame(result_rows)

# 전체 성능
eval_df = result_df[result_df["bubble_label"].notna()].copy()
y_true  = (eval_df["bubble_label"] == 2).astype(int)
y_pred  = eval_df["anomaly"]

print("\n=== 구 단위 LSTM-AE 버블(2) 탐지 성능 (전체) ===")
print(classification_report(y_true, y_pred, target_names=["비버블","버블"], digits=3, zero_division=0))
try:
    auc = roc_auc_score(y_true, eval_df["recon_error"])
    print(f"AUC-ROC: {auc:.4f}")
except Exception: pass

# 시도별 성능
print("\n[시도별 성능]")
for sido_name in SIDO_LIST:
    sub_e = eval_df[eval_df["시도"] == sido_name]
    if len(sub_e) == 0: continue
    yt = (sub_e["bubble_label"] == 2).astype(int)
    yp = sub_e["anomaly"]
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    f  = f1_score(yt, yp, zero_division=0)
    print(f"  {sido_name}: Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}  "
          f"(GT버블={yt.sum()}, 탐지={yp.sum()})")


# ══════════════════════════════════════════════════════════════
# 7. 시각화
# ══════════════════════════════════════════════════════════════
print("\n[7] 시각화")

# ── 그림1: 이상 탐지 빈도 상위 15개 구
bubble_freq = (
    result_df[result_df["anomaly"] == 1]
    .groupby(["시도", "구"])
    .size()
    .reset_index(name="anomaly_months")
    .sort_values("anomaly_months", ascending=False)
    .head(15)
)
fig, ax = plt.subplots(figsize=(12, 6))
labels_  = [f"{r['구']} ({r['시도']})" for _, r in bubble_freq.iterrows()]
colors_  = [SIDO_COLORS[r["시도"]] for _, r in bubble_freq.iterrows()]
ax.barh(labels_[::-1], bubble_freq["anomaly_months"].values[::-1],
        color=colors_[::-1], alpha=0.8, edgecolor='white')
ax.set_xlabel("이상 탐지 개월 수", fontsize=11)
ax.set_title("LSTM-AE 이상 탐지 빈도 상위 15개 구", fontsize=13)
ax.grid(axis="x", alpha=0.3)
from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=s) for s, c in SIDO_COLORS.items()]
ax.legend(handles=legend_handles, fontsize=9)
plt.tight_layout()
plt.savefig("notebooks/fig_ae_sgg_01_anomaly_freq.png", bbox_inches="tight")
plt.close()
print("  fig_ae_sgg_01_anomaly_freq.png 저장")

# ── 그림2: 시도별 재구성 오차 분포 (정상 vs 버블)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, sido_name in zip(axes, SIDO_LIST):
    sub = eval_df[eval_df["시도"] == sido_name]
    normal_err = sub[sub["bubble_label"] == 0]["recon_error"]
    bubble_err = sub[sub["bubble_label"] == 2]["recon_error"]
    ax.hist(normal_err, bins=40, alpha=0.6, color="#ADB5BD", label="정상")
    ax.hist(bubble_err, bins=40, alpha=0.7, color="#E63946", label="버블")
    ax.set_title(f"{sido_name} 재구성 오차 분포", fontsize=11)
    ax.set_xlabel("Reconstruction Error"); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.suptitle("LSTM-AE 재구성 오차: 정상 vs 버블", fontsize=13)
plt.tight_layout()
plt.savefig("notebooks/fig_ae_sgg_02_error_dist.png", bbox_inches="tight")
plt.close()
print("  fig_ae_sgg_02_error_dist.png 저장")


# ══════════════════════════════════════════════════════════════
# 8. 저장
# ══════════════════════════════════════════════════════════════
print("\n[8] 저장")
base_df  = pd.read_csv("data/processed/features_sgg_final.csv", encoding=ENC)
merge_df = result_df[["ym", "시도", "구", "recon_error", "anomaly"]]
save_df  = base_df.merge(merge_df, on=["ym", "시도", "구"], how="left")
save_df.to_csv("data/processed/lstm_ae_sgg_scores.csv", index=False, encoding=ENC)

print(f"\n✅ 저장: data/processed/lstm_ae_sgg_scores.csv  {save_df.shape}")
print(f"   anomaly 컬럼 커버리지: {save_df['anomaly'].notna().sum()}/{len(save_df)} "
      f"({save_df['anomaly'].notna().mean()*100:.1f}%)")
print(f"\n총 소요시간: {time.time()-t0:.1f}s")
print("\n" + "=" * 60)
print("  완료")
print("=" * 60)
