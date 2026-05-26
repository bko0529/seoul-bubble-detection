"""
LSTM Autoencoder 버블 이상 탐지 — 실행 스크립트
입력:  data/processed/features_sido_monthly.csv
출력:  data/processed/lstm_ae_scores.csv
       models/lstm_ae_best.pth
"""
import sys, warnings, os, random, time
warnings.filterwarnings('ignore')
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.makedirs("models", exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 60)
print("  LSTM Autoencoder 버블 이상 탐지")
print("=" * 60)

# ── 상수
SEED       = 42
DEVICE     = torch.device("cpu")
ENC        = "utf-8-sig"
SEQ_LEN    = 12
SIDO_LIST  = ["서울", "경기", "인천"]
FEATURES   = [
    "매매중위_MoM","매매중위_YoY","매매_vol12",
    "전세가율","PIR","소득대비전세율",
    "base_rate","rate_spread","mortgage_rate_chg_yoy",
    "m2_yoy_pct","cpi_yoy_pct","bsi_realestate",
    "미분양_yoy","인허가_yoy","차주당대출_yoy","가계대출비중_yoy",
]
N_FEATURES = len(FEATURES)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── 1. 데이터 로드 & 정규화
df = pd.read_csv("data/processed/features_sido_monthly.csv", encoding=ENC)
df_filled = df.copy()
df_filled[FEATURES] = df_filled[FEATURES].fillna(0)

scalers, df_scaled_list = {}, []
for sido in SIDO_LIST:
    sub = df_filled[df_filled["시도"]==sido].sort_values("ym").reset_index(drop=True)
    nm  = sub["bubble_label"] == 0
    sc  = MinMaxScaler(); sc.fit(sub.loc[nm, FEATURES])
    scalers[sido] = sc
    s = sub.copy(); s[FEATURES] = sc.transform(sub[FEATURES])
    df_scaled_list.append(s)
df_scaled = pd.concat(df_scaled_list, ignore_index=True)
print(f"[1] 정규화 완료: {df_scaled.shape}")

# ── 2. 시퀀스 구성
def make_sequences(data_df, seq_len, normal_only=True):
    seqs, labels, yms, sidos = [], [], [], []
    for sido in SIDO_LIST:
        sub = data_df[data_df["시도"]==sido].sort_values("ym").reset_index(drop=True)
        vals = sub[FEATURES].values.astype(np.float32)
        lbls = sub["bubble_label"].values
        yms_ = sub["ym"].values
        for i in range(len(sub)-seq_len+1):
            wl = lbls[i:i+seq_len]
            if normal_only and not np.all(wl==0): continue
            seqs.append(vals[i:i+seq_len])
            labels.append(lbls[i+seq_len-1])
            yms.append(yms_[i+seq_len-1])
            sidos.append(sido)
    return np.array(seqs), np.array(labels), np.array(yms), sidos

train_seqs, train_labels, train_yms, train_sidos = make_sequences(df_scaled, SEQ_LEN, True)
all_seqs,   all_labels,   all_yms,   all_sidos   = make_sequences(df_scaled, SEQ_LEN, False)
print(f"[2] 학습 시퀀스: {train_seqs.shape}  전체: {all_seqs.shape}")

class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = torch.tensor(seqs, dtype=torch.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]

train_ds = SeqDataset(train_seqs)
all_ds   = SeqDataset(all_seqs)

# ── 3. 모델 정의
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
        x = x.to(DEVICE); recon, _ = model(x); loss = recon_loss(recon, x)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); total = 0.0
    for x in loader:
        x = x.to(DEVICE); recon, _ = model(x); total += recon_loss(recon, x).item() * len(x)
    return total / len(loader.dataset)

@torch.no_grad()
def get_errors(model, loader):
    model.eval(); errors = []
    for x in loader:
        x = x.to(DEVICE); recon, _ = model(x)
        mse = ((recon - x)**2).mean(dim=-1).mean(dim=-1)
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)

def train_model(params, ds, n_epochs=100, patience=15, val_ratio=0.15, verbose=False):
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
    history = []
    for epoch in range(1, n_epochs+1):
        tl = train_one_epoch(model, tr_ld, opt)
        vl = evaluate(model, vl_ld)
        sch.step(vl); history.append((tl, vl))
        if vl < best_val:
            best_val = vl; best_state = {k: v.clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: print(f"  Early stop @ {epoch}")
                break
    model.load_state_dict(best_state)
    return model, best_val, history

# ── 4. Optuna HPO (15 trials)
def objective(trial):
    p = {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "latent_dim":  trial.suggest_categorical("latent_dim",  [8, 16, 32]),
        "num_layers":  trial.suggest_int("num_layers", 1, 2),
        "dropout":     trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        "lr":          trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", [16, 32]),
    }
    _, vl, _ = train_model(p, train_ds, n_epochs=50, patience=8)
    return vl

t0 = time.time()
print("[3] Optuna HPO 시작 (15 trials) ...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=15)
best_params = study.best_params
print(f"    완료: {time.time()-t0:.1f}s  val_loss={study.best_value:.6f}")
print(f"    최적: {best_params}")

# ── 5. 최종 학습
print("[4] 최종 모델 학습 (200 epochs, early stopping) ...")
t1 = time.time()
final_model, final_val, history = train_model(
    best_params, train_ds, n_epochs=200, patience=20, verbose=True)
print(f"    완료: {time.time()-t1:.1f}s  val_loss={final_val:.5f}")

# ── 6. 재구성 오차 + 임계값
all_loader   = DataLoader(all_ds,   batch_size=64)
train_loader = DataLoader(train_ds, batch_size=64)
all_errors   = get_errors(final_model, all_loader)
train_errors = get_errors(final_model, train_loader)
threshold    = float(np.percentile(train_errors, 95))
print(f"[5] 임계값(95th): {threshold:.5f}  |  정상오차 평균={train_errors.mean():.5f}")

# ── 7. 결과 DataFrame
result_df = pd.DataFrame({
    "ym":           all_yms,
    "시도":          all_sidos,
    "recon_error":  all_errors,
    "bubble_label": all_labels,
    "anomaly":      (all_errors > threshold).astype(int),
})

# ── 8. 성능 평가
eval_df  = result_df[~np.isnan(result_df["bubble_label"])]
y_true   = (eval_df["bubble_label"] == 2).astype(int)
y_pred   = eval_df["anomaly"]
print(f"\n[6] 버블(2) 탐지 성능:")
print(classification_report(y_true, y_pred, target_names=["비버블","버블"], digits=3, zero_division=0))
try:
    auc = roc_auc_score(y_true, eval_df["recon_error"])
    print(f"AUC-ROC: {auc:.4f}")
except: pass

# ── 9. 저장
save_df = df.merge(result_df[["ym","시도","recon_error","anomaly"]], on=["ym","시도"], how="left")
save_df.to_csv("data/processed/lstm_ae_scores.csv", index=False, encoding="utf-8-sig")
print(f"\n[7] 저장: data/processed/lstm_ae_scores.csv  {save_df.shape}")

torch.save({
    "model_state": final_model.state_dict(),
    "params":      best_params,
    "features":    FEATURES,
    "seq_len":     SEQ_LEN,
    "threshold":   threshold,
}, "models/lstm_ae_best.pth")
print("[7] 저장: models/lstm_ae_best.pth")
print(f"\n총 소요시간: {time.time()-t0:.1f}s")
print("=" * 60)
