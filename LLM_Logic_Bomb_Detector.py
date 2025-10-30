import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

def generate_synthetic_sample():
    conditional_count = np.random.poisson(5)
    date_check = int(np.random.rand() < 0.05)
    rare_api_calls = np.random.poisson(0.8)
    dynamic_trigger_flag = int(np.random.rand() < 0.04)
    file_modifications = np.random.poisson(1.0)
    unusual_time_exec = int(np.random.rand() < 0.1)
    entropy = min(1.0, max(0.0, np.random.normal(0.3, 0.15)))
    
    rule1 = date_check and (rare_api_calls >= 1 or dynamic_trigger_flag)
    rule2 = (conditional_count > 20) and (rare_api_calls > 3)
    rule3 = (dynamic_trigger_flag and unusual_time_exec and file_modifications > 2)
    label = int(rule1 or rule2 or rule3)
    
    conditional_count = int(max(0, np.random.normal(conditional_count, 2.0)))
    rare_api_calls = int(max(0, np.random.normal(rare_api_calls, 1.0)))
    file_modifications = int(max(0, np.random.normal(file_modifications, 1.0)))
    entropy = float(min(1.0, max(0.0, entropy + np.random.normal(0, 0.05))))
    
    features = {
        "conditional_count": float(conditional_count),
        "date_check": float(date_check),
        "rare_api_calls": float(rare_api_calls),
        "dynamic_trigger_flag": float(dynamic_trigger_flag),
        "file_modifications": float(file_modifications),
        "unusual_time_exec": float(unusual_time_exec),
        "entropy": entropy
    }
    return features, label

def generate_dataset(n_samples=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data = []
    labels = []
    for _ in range(n_samples):
        f, l = generate_synthetic_sample()
        data.append(f)
        labels.append(l)
    df = pd.DataFrame(data)
    df["label"] = labels
    return df

FEATURE_COLUMNS = [
    "conditional_count",
    "date_check",
    "rare_api_calls",
    "dynamic_trigger_flag",
    "file_modifications",
    "unusual_time_exec",
    "entropy"
]

class LogicBombDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.X = df[FEATURE_COLUMNS].values.astype(np.float32)
        self.y = df["label"].values.astype(np.float32).reshape(-1,1)
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-6
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

class PredicateNet(nn.Module):
    def __init__(self, input_dim, n_predicates=6, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_predicates)
        )
    
    def forward(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits)

class LogicalClauseLayer(nn.Module):
    def __init__(self, n_predicates, n_clauses=8):
        super().__init__()
        self.raw_w = nn.Parameter(torch.randn(n_clauses, n_predicates) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_clauses))
        self.n_clauses = n_clauses
        self.n_predicates = n_predicates

    def forward(self, preds):
        w = torch.nn.functional.softplus(self.raw_w)
        clause_inputs = torch.matmul(preds, w.t())
        clause_activations = torch.sigmoid(clause_inputs - self.bias)
        or_soft = 1.0 - torch.prod(1.0 - clause_activations + 1e-8, dim=1, keepdim=True)
        return or_soft, clause_activations

class LNNDetector(nn.Module):
    def __init__(self, input_dim, n_predicates=6, n_clauses=8):
        super().__init__()
        self.pred_net = PredicateNet(input_dim, n_predicates=n_predicates)
        self.clause_layer = LogicalClauseLayer(n_predicates, n_clauses=n_clauses)

    def forward(self, x):
        preds = self.pred_net(x)
        or_soft, clause_acts = self.clause_layer(preds)
        return or_soft, preds, clause_acts

def train_model(model, train_dl, val_dl, lr=1e-3, epochs=12, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out, _, _ = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_dl.dataset)
        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                out, _, _ = model(xb)
                ys.append(yb.cpu().numpy())
                ps.append(out.cpu().numpy())
        ys = np.vstack(ys)
        ps = np.vstack(ps)
        preds = (ps >= 0.5).astype(int)
        acc = accuracy_score(ys, preds)
        try:
            auc = roc_auc_score(ys, ps)
        except Exception:
            auc = float("nan")
        print(f"Epoch {epoch:02d}  loss={total_loss:.4f}  val_acc={acc:.4f}  val_auc={auc:.4f}")
    return model

def explain_sample(model: LNNDetector, raw_feature_vector: Dict[str,float], dataset_obj: LogicBombDataset, top_k_clauses=3):
    model.eval()
    x = np.array([raw_feature_vector[c] for c in FEATURE_COLUMNS], dtype=np.float32)
    x = (x - dataset_obj.mean) / dataset_obj.std
    xt = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        prob, preds, clause_acts = model(xt)
    prob = float(prob.item())
    preds = preds.squeeze(0).cpu().numpy()
    clause_acts = clause_acts.squeeze(0).cpu().numpy()
    print("=== Explanation for sample ===")
    print("Raw features:")
    for k,v in raw_feature_vector.items():
        print(f"  {k}: {v}")
    print("\nPredicate (soft truth) values:")
    for i, val in enumerate(preds):
        print(f"  P{i+1}: {val:.3f}")
    print("\nClause activations:")
    for j,act in enumerate(clause_acts):
        print(f"  Clause{j+1}: {act:.3f}")
    top_idx = np.argsort(-clause_acts)[:top_k_clauses]
    print("\nTop clauses contributing to final decision:")
    for idx in top_idx:
        print(f"  Clause{idx+1}: activation={clause_acts[idx]:.3f}")
    print(f"\nFinal probability (logic-bomb present) = {prob:.3f}")
    return prob, preds, clause_acts

def main():
    print("Generating synthetic dataset...")
    df = generate_dataset(n_samples=6000)
    train_df = df.sample(frac=0.7, random_state=1)
    rest = df.drop(train_df.index)
    val_df = rest.sample(frac=0.5, random_state=1)
    test_df = rest.drop(val_df.index)

    train_ds = LogicBombDataset(train_df)
    val_ds = LogicBombDataset(val_df)
    test_ds = LogicBombDataset(test_df)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    input_dim = len(FEATURE_COLUMNS)
    model = LNNDetector(input_dim=input_dim, n_predicates=6, n_clauses=8)
    print("Training model...")
    model = train_model(model, train_dl, val_dl, lr=3e-3, epochs=18, device="cpu")

    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in test_dl:
            out, _, _ = model(xb)
            ys.append(yb.numpy())
            ps.append(out.numpy())
    ys = np.vstack(ys)
    ps = np.vstack(ps)
    preds = (ps >= 0.5).astype(int)
    acc = accuracy_score(ys, preds)
    try:
        auc = roc_auc_score(ys, ps)
    except Exception:
        auc = float("nan")
    print(f"\nTest Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    sample_rows = test_df.sample(n=5, random_state=2).to_dict(orient="records")
    for r in sample_rows:
        raw = {c: r[c] for c in FEATURE_COLUMNS}
        explain_sample(model, raw, train_ds)
        print("\n-----------------------------\n")

if __name__ == "__main__":
    main()
    