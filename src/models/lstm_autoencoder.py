"""
LSTM Autoencoder: 시계열 이상 탐지
재구성 오차가 임계값 초과 시 버블 신호 발생
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]  # (batch, hidden)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, seq_len: int,
                 num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # z: (batch, hidden)
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z)
        return self.fc(out)  # (batch, seq_len, output_size)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 latent_size: int = 32, seq_len: int = 12, num_layers: int = 2):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.expand = nn.Linear(latent_size, hidden_size)
        self.decoder = LSTMDecoder(hidden_size, input_size, seq_len, num_layers)

    def forward(self, x):
        z = self.encoder(x)
        z = torch.relu(self.bottleneck(z))
        z = torch.relu(self.expand(z))
        return self.decoder(z)


def make_sequences(arr: np.ndarray, seq_len: int = 12) -> np.ndarray:
    """슬라이딩 윈도우로 시퀀스 생성"""
    seqs = []
    for i in range(len(arr) - seq_len + 1):
        seqs.append(arr[i:i + seq_len])
    return np.array(seqs, dtype=np.float32)


def train(model: LSTMAutoencoder, X: np.ndarray, epochs: int = 100,
          lr: float = 1e-3, batch_size: int = 32, device: str = "cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    seqs = make_sequences(X, model.decoder.seq_len)
    dataset = TensorDataset(torch.from_numpy(seqs))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss / len(loader):.6f}")
    return model


def compute_anomaly_scores(model: LSTMAutoencoder, X: np.ndarray,
                            device: str = "cpu") -> np.ndarray:
    """각 시점의 재구성 오차 반환"""
    model.eval()
    seqs = make_sequences(X, model.decoder.seq_len)
    tensor = torch.from_numpy(seqs).to(device)
    with torch.no_grad():
        recon = model(tensor).cpu().numpy()
    mse = np.mean((seqs - recon) ** 2, axis=(1, 2))
    return mse


def get_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(scores, percentile))
