# inference.py - Evaluation, Robustness Testing & Polished Demo Runner
import os
import math
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, roc_auc_score
from IPython.display import display
import matplotlib.pyplot as plt
import kagglehub

print("=== LEUKEMIA DETECTION INFERENCE & JUDGE DEMO ===")
print("Disclaimer: This project is a prototype and is not medical advice.[cite: 16, 22]")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Auto-locate dataset (downloads `mehradaria/leukemia` if missing)
original_dir = Path('./data/Original')
if not original_dir.exists():
    dataset_root = kagglehub.dataset_download("mehradaria/leukemia")
else:
    dataset_root = './data'

search_base = dataset_root if os.path.exists(os.path.join(dataset_root, 'Original')) else './data'
rows = []
for c in ['Benign', 'Early', 'Pre', 'Pro']:
    for p in glob.glob(os.path.join(search_base, f'Original/{c}/*.jpg')):
        rows.append({'path': p, 'stage': c, 'label': 0 if c == 'Benign' else 1})
df = pd.DataFrame(rows)
df = df.sort_values(['path']).reset_index(drop=True)
df['num'] = df.path.str.extract(r'-(\d+)\.jpg')[0].astype(int)
df = df.sort_values(['stage', 'num']).reset_index(drop=True)

print("Loading test dataset tensor memory...")
imgs = np.stack([np.asarray(Image.open(p).convert('RGB').resize((224, 224))) for p in df.path])
X_all = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
y_all = torch.tensor(df.label.values)
del imgs

splits = np.load('./models/final_split.npz')
tr, va, te = splits['tr'], splits['va'], splits['te']

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def prep(xb, train=False):
    xb = xb.to(device).float() / 255.
    return (xb - mean) / std

def load_resnet():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 2)
    return m.to(device)

final_model = load_resnet(); final_model.load_state_dict(torch.load('./models/final_model.pt', map_location=device))
aug_model = load_resnet(); aug_model.load_state_dict(torch.load('./models/aug_model.pt', map_location=device))
robust_model = load_resnet(); robust_model.load_state_dict(torch.load('./models/robust_model.pt', map_location=device))

MODELS = {'robust': robust_model, 'original': final_model, 'bright-aug': aug_model}
SPLIT_OF = {int(i): nm for nm, ids_ in (('train', tr), ('val', va), ('test', te)) for i in ids_}

def predict_probs(model, idx, bs=128):
    model.eval()
    idx = np.asarray(idx)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = prep(X_all[torch.as_tensor(idx[i:i + bs])])
            out.append(torch.softmax(model(xb), 1)[:, 1].cpu())
    return torch.cat(out).numpy()

def referral_table(p, y):
    pred = (p >= 0.5).astype(int)
    rows = []
    for conf in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
        sure = np.maximum(p, 1 - p) >= conf
        rows.append(dict(confidence_cutoff=conf,
                         referred_percent=round(100 * (~sure).mean(), 1),
                         missed_leukemia=int(((pred == 0) & (y == 1) & sure).sum()),
                         false_alarms=int(((pred == 1) & (y == 0) & sure).sum()),
                         errors_caught_by_referral=int(((pred != y) & ~sure).sum())))
    return pd.DataFrame(rows)

y_te = y_all[torch.as_tensor(te)].numpy()
y_va = y_all[torch.as_tensor(va)].numpy()

print("\n--- RUNNING ROBUSTNESS & REFERRAL TESTS FOR JUDGES ---")
p_va_r, p_te_r = predict_probs(robust_model, va), predict_probs(robust_model, te)
print("ROBUST MODEL - TEST REFERRAL TABLE:")
display(referral_table(p_te_r, y_te))

# =========================================================================
# POLISHED DEMO RUNNER CONFIGURATION
# =========================================================================
N = 6                  # Number of demo images
WHERE = 'test'         # 'test', 'val', 'train', or 'all'
STAGE = None           # None, or 'Benign' / 'Early' / 'Pre' / 'Pro'
ONLY_ERRORS = False    # True = showcase model error cases
CHANGES = []           # e.g., [('blur', 2)] or ['corner']
MODEL = 'robust'       # 'robust', 'original', or 'bright-aug'
CUTOFF = 0.9           # Human referral confidence threshold
COLS = 3               # Grid columns
# =========================================================================

print(f"\n--- GENERATING POLISHED DEMO GRID (Model: {MODEL}, Target: {WHERE}) ---")
m = MODELS[MODEL]
pool = np.asarray({'test': te, 'val': va, 'train': tr, 'all': df.index.values}[WHERE])
if STAGE is not None:
    keep = [STAGE] if isinstance(STAGE, str) else list(STAGE)
    pool = pool[df.loc[pool, 'stage'].isin(keep).values]

idx = np.random.default_rng(42).choice(pool, size=min(N, len(pool)), replace=False)
xb = X_all[torch.as_tensor(idx)].clone()
meta = pd.DataFrame({
    'file': df.loc[idx, 'path'].str.split('/').str[-1].values, 
    'true_stage': df.loc[idx, 'stage'].values,
    'true_label': df.loc[idx, 'label'].values, 
    'split': [SPLIT_OF.get(int(i), '?') for i in idx]
})

def probs_of(model, xb, bs=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(xb), bs):
            out.append(torch.softmax(model(prep(xb[i:i + bs])), 1)[:, 1].cpu())
    return torch.cat(out).numpy()

p = probs_of(m, xb); conf = np.maximum(p, 1 - p); pred = (p >= 0.5).astype(int)
known = meta.true_label.notna().values; truth = meta.true_label.fillna(-1).astype(int).values

verdict, outcome = [], []
for k in range(len(p)):
    ref = conf[k] < CUTOFF
    verdict.append('REFER TO HUMAN' if ref else ('LEUKEMIA' if pred[k] else 'BENIGN'))
    if not known[k]: outcome.append('no label')
    elif ref: outcome.append('referred')
    elif pred[k] == truth[k]: outcome.append('correct')
    else: outcome.append('MISSED LEUKEMIA' if truth[k] == 1 else 'FALSE ALARM')

colour = ['black' if o == 'no label' else 'darkorange' if o == 'referred' else 'forestgreen' if o == 'correct' else 'crimson' for o in outcome]

rows_cnt = math.ceil(len(xb) / COLS)
fig, axes = plt.subplots(rows_cnt, COLS, figsize=(5 * COLS, 5.2 * rows_cnt))
axes = np.atleast_1d(axes).ravel()
for ax in axes: ax.axis('off')

for ax, img, v, pp, st, c in zip(axes, xb, verdict, p, meta.true_stage, colour):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"Verdict: {v}\nP(Leukemia) = {pp:.2f} | True: {st}", fontsize=10, fontweight='bold', color=c)
    ax.patch.set_edgecolor(c)
    ax.patch.set_linewidth(3)

fig.suptitle(f"Clinical AI Prototype Demo | Cutoff: {CUTOFF}", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

tbl = meta.drop(columns='true_label').copy()
tbl['P_leukemia'] = p.round(3)
tbl['confidence'] = conf.round(3)
tbl['verdict'] = verdict
tbl['outcome'] = outcome
display(tbl)
print("Demo Summary Statistics:", pd.Series(outcome).value_counts().to_dict())
