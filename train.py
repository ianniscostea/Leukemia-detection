# train.py - Leukemia Detection Training & Experiment Pipeline with Auto-Download
import os
import random
import glob
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import kagglehub

print("=== STARTING TRAINING & EXPERIMENT PIPELINE ===")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

os.makedirs('./models', exist_ok=True)
os.makedirs('./data', exist_ok=True)

# --- AUTOMATIC DATASET DOWNLOADER VIA KAGGLE ---
def ensure_kaggle_dataset():
    original_dir = Path('./data/Original')
    if not original_dir.exists():
        print("Dataset not found locally. Downloading automatically via Kaggle (`mehradaria/leukemia`)...")
        download_path = kagglehub.dataset_download("mehradaria/leukemia")
        print(f"Dataset downloaded to cache: {download_path}")
        return download_path
    else:
        print("Dataset already exists locally.")
        return './data'

DATASET_ROOT = ensure_kaggle_dataset()

# 1. Data Discovery & DataFrame Construction
paths = glob.glob(os.path.join(DATASET_ROOT, '**/*.*'), recursive=True)
paths = [p for p in paths if p.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
print("Total images found:", len(paths))

rows = []
search_base = DATASET_ROOT if os.path.exists(os.path.join(DATASET_ROOT, 'Original')) else './data'
for c in ['Benign', 'Early', 'Pre', 'Pro']:
    for p in glob.glob(os.path.join(search_base, f'Original/{c}/*.jpg')):
        rows.append({'path': p, 'stage': c, 'label': 0 if c == 'Benign' else 1})
df = pd.DataFrame(rows)

print(df.stage.value_counts())
print("Share malignant:", round(df.label.mean(), 3))

# 2. Feature Extraction & Block Split Validation Checks
df = df.sort_values(['path']).reset_index(drop=True)
df['num'] = df.path.str.extract(r'-(\d+)\.jpg')[0].astype(int)
df = df.sort_values(['stage', 'num']).reset_index(drop=True)
df['block_id'] = df.stage + '_' + (df.groupby('stage').cumcount() // 30).astype(str)

feat = ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std']

def colour_stats(p):
    im = np.asarray(Image.open(p).convert('RGB').resize((64, 64)), dtype=float)
    return [*im.mean(axis=(0, 1)), *im.std(axis=(0, 1))]

stats = np.array([colour_stats(p) for p in df.path])
df[['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std']] = stats

rand_train, rand_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df.stage)
rng = np.random.default_rng(42)
ids = df.block_id.unique()
test_blocks = set(rng.choice(ids, size=int(0.2 * len(ids)), replace=False))
block_test = df.index[df.block_id.isin(test_blocks)]
block_train = df.index.difference(block_test)

def evaluate(name, tr, te):
    Xtr, ytr = df.loc[tr, feat], df.loc[tr, 'label']
    Xte, yte = df.loc[te, feat], df.loc[te, 'label']
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000)).fit(Xtr, ytr)
    pred = m.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    print(f"{name}: test size {len(yte)} | benign in test {tn + fp}")
    print(f"   colour-only accuracy: {(pred == yte).mean():.3f} | leukemia recall: {recall_score(yte, pred):.3f}")

evaluate("RANDOM split", rand_train, rand_test)
evaluate("BLOCK split ", block_train, block_test)

# 3. Tensor Conversion for Deep Learning
print("Loading images into tensor memory...")
imgs = np.stack([np.asarray(Image.open(p).convert('RGB').resize((224, 224))) for p in df.path])
X_all = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
y_all = torch.tensor(df.label.values)
del imgs

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

AUG = {'bright': False, 'degrade': False}

def prep(xb, train=False):
    xb = xb.to(device).float() / 255.
    if train:
        m = torch.rand(xb.shape[0], device=device) < 0.5
        xb[m] = xb[m].flip(3)
        m = torch.rand(xb.shape[0], device=device) < 0.5
        xb[m] = xb[m].flip(2)
        if AUG['bright']:
            f = torch.empty(xb.shape[0], 1, 1, 1, device=device).uniform_(0.7, 1.3)
            xb = (xb * f).clamp(0, 1)
        if AUG['degrade']:
            if np.random.rand() < 0.5:
                s = float(np.random.uniform(0.5, 3.0))
                xb = TF.gaussian_blur(xb, kernel_size=int(2 * round(3 * s) + 1), sigma=s)
            if np.random.rand() < 0.5:
                sc = float(np.random.uniform(0.25, 0.75))
                small = nn.functional.interpolate(xb, scale_factor=sc, mode='bilinear', align_corners=False)
                xb = nn.functional.interpolate(small, size=(224, 224), mode='bilinear', align_corners=False)
            xb = xb.clamp(0, 1)
    return (xb - mean) / std

def predict_probs(model, idx, bs=128):
    model.eval()
    idx = np.asarray(idx)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = prep(X_all[torch.as_tensor(idx[i:i + bs])])
            out.append(torch.softmax(model(xb), 1)[:, 1].cpu())
    return torch.cat(out).numpy()

def train_model(tr, va, epochs=5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 2)
    model = model.to(device)
    tr = np.asarray(tr)
    n1 = int(y_all[torch.as_tensor(tr)].sum()); n0 = len(tr) - n1
    w = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 32):
            b = torch.as_tensor(perm[i:i + 32])
            xb, yb = prep(X_all[b], train=True), y_all[b].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        auc = roc_auc_score(y_all[torch.as_tensor(va)].numpy(), predict_probs(model, va))
        print(f"  epoch {ep + 1}: validation AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model

# 4. Cluster Split Setup
feat_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feat_model.fc = nn.Identity()
feat_model = feat_model.to(device).eval()

feats = []
with torch.no_grad():
    for i in range(0, len(df), 128):
        feats.append(feat_model(prep(X_all[i:i + 128])).cpu())
feats = torch.cat(feats).numpy()

Z = PCA(n_components=50, random_state=0).fit_transform(feats)
df['cluster'] = KMeans(n_clusters=80, n_init=10, random_state=0).fit_predict(Z)

def cluster_split(seed):
    cl = df.cluster.unique().copy()
    np.random.default_rng(seed).shuffle(cl)
    n_te, n_va = int(0.2 * len(cl)), int(0.15 * len(cl))
    te = df.index[df.cluster.isin(set(cl[:n_te]))].values
    va = df.index[df.cluster.isin(set(cl[n_te:n_te + n_va]))].values
    tr = np.setdiff1d(df.index.values, np.concatenate([te, va]))
    return tr, va, te

for seed in range(50):
    tr, va, te = cluster_split(seed)
    nb_va = int((y_all[torch.as_tensor(va)] == 0).sum())
    nb_te = int((y_all[torch.as_tensor(te)] == 0).sum())
    if nb_va >= 30 and nb_te >= 60:
        break
print(f"Cluster split found with seed {seed} | train {len(tr)}, val {len(va)}, test {len(te)}")

# 5. Train and Save
print("\n--- Training Original Model ---")
AUG['bright'] = False; AUG['degrade'] = False
final_model = train_model(tr, va, epochs=5, seed=0)
torch.save(final_model.state_dict(), './models/final_model.pt')

print("\n--- Training Brightness-Augmented Model ---")
AUG['bright'] = True; AUG['degrade'] = False
aug_model = train_model(tr, va, epochs=5, seed=0)
torch.save(aug_model.state_dict(), './models/aug_model.pt')

print("\n--- Training Robust Model ---")
AUG['bright'] = True; AUG['degrade'] = True
np.random.seed(0)
robust_model = train_model(tr, va, epochs=8, seed=0)
torch.save(robust_model.state_dict(), './models/robust_model.pt')

np.savez('./models/final_split.npz', tr=tr, va=va, te=te)
print("Training complete! Models saved to ./models/")
