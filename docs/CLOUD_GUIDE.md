# Cloud Guide — Colab & Kaggle for heavy models and fine-tuning

Local development is on an Apple **M3 Pro / MPS (no CUDA)**. Frozen feature extraction of
the CNNs and the ViT‑Base/L foundation models runs fine locally. Reach for a cloud GPU
when either:

1. a backbone is too slow/large on MPS for comfortable iteration (e.g. ViT‑Giant), or
2. you need **end‑to‑end fine‑tuning** (gradients through the backbone) — MPS is too slow
   and some ops fall back to CPU, so a CUDA GPU is the practical choice.

Everything below is designed so the cloud step produces artifacts the **local** pipeline
consumes unchanged. You never run the whole project in the cloud — just the GPU‑heavy step.

---

## The contract with the local pipeline (read this first)

The local scripts are config‑driven and backbone‑agnostic. Two file contracts matter:

**1. Frozen features** → `python scripts/train_path_a.py --config <cfg>` consumes a cache
directory (`cfg.features.cache_dir`) containing, per split:

| File | Shape / columns | Notes |
|---|---|---|
| `{backbone}_{split}.npy` | `(N, D)` float32 | one row per patch, in the split's img_path order |
| `{split}_paths.csv` | column `img_path` | **same row order** as the `.npy` (the alignment key) |

`split` ∈ {`train`, `test`}; `backbone` is each name in `cfg.features.backbones`.

**2. Final predictions** (fine‑tuning, or any end‑to‑end training) →
`python scripts/make_submission.py --config <cfg>` reads `cfg.outputs.oof_path`, an
`.npz` with keys:

```
y_true       int   (N_train,)      # ground-truth labels, train order
oof_probs    float (N_train,)      # out-of-fold predicted probabilities
test_probs   float (N_test,)       # fold-averaged test probabilities
test_paths   str   (N_test,)       # test img_paths, aligned to test_probs
tau          float scalar          # OOF-optimized decision threshold
temperature  float scalar          # 1.0 if no calibration
```

Match these exactly and the local submission/run‑log/cataloging all work as‑is.

---

## Getting the data into the cloud

- **Kaggle:** in the competition's notebook the data is already mounted under
  `/kaggle/input/<competition>/`. Otherwise zip `train/`, `test/`, `train.csv`,
  `dummyTest.csv` and upload as a **private Kaggle Dataset**, then "Add Data" to the notebook.
- **Colab:** zip the data, upload to Google Drive, then mount:
  ```python
  from google.colab import drive; drive.mount('/content/drive')
  DATA = '/content/drive/MyDrive/be224b_data'   # contains train/ test/ train.csv dummyTest.csv
  ```

---

## Workflow A — Frozen feature extraction on a cloud GPU

Use for heavy frozen backbones (e.g. H‑optimus‑0). Produces the `.npy` + paths CSV the
local `train_path_a.py` expects.

```python
# --- setup (Colab/Kaggle) ---
!pip -q install torch timm transformers albumentations opencv-python-headless scikit-learn pandas numpy pillow

# Gated models (H-optimus, UNI, Virchow, GigaPath) need HF auth — accept the license on the
# model page first, then:
from huggingface_hub import login; login()        # paste a READ token (Kaggle: store as a Secret)

import os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA = '/kaggle/input/<competition>'              # or your Colab Drive path
CACHE = 'features_hoptimus'; os.makedirs(CACHE, exist_ok=True)
dev = 'cuda'

# H-optimus-0: ViT-Giant via timm, 1536-d, NON-ImageNet normalization
import timm
model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                          init_values=1e-5, dynamic_img_size=False, num_classes=0).to(dev).eval()
MEAN, STD = (0.707223,0.578729,0.703617), (0.211883,0.230117,0.177517)
tf = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

class DS(Dataset):
    def __init__(s, paths): s.p = paths
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        img = np.asarray(Image.open(os.path.join(DATA, s.p[i])).convert('RGB'))
        return tf(image=img)['image']

def flip4(x):  # 4-way flip TTA (orig, hflip, vflip, h+v)
    return [x, torch.flip(x,[-1]), torch.flip(x,[-2]), torch.flip(x,[-2,-1])]

@torch.no_grad()
def extract(split, paths):
    loader = DataLoader(DS(paths), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    feats = []
    for imgs in loader:
        imgs = imgs.to(dev, non_blocking=True)
        acc = None
        for v in flip4(imgs):                      # average over TTA views
            with torch.autocast('cuda', dtype=torch.float16):
                f = model(v).float()
            acc = f if acc is None else acc + f
        feats.append((acc/4).cpu().numpy())
    arr = np.concatenate(feats).astype(np.float32)
    np.save(f'{CACHE}/h_optimus_{split}.npy', arr)
    pd.DataFrame({'img_path': paths}).to_csv(f'{CACHE}/{split}_paths.csv', index=False)
    print(split, arr.shape)

extract('train', pd.read_csv(f'{DATA}/train.csv')['img_path'].tolist())
extract('test',  pd.read_csv(f'{DATA}/dummyTest.csv')['img_path'].tolist())
```

**Bring the features home:** download `features_hoptimus/` (Colab: zip + `files.download`;
Kaggle: it's in `/kaggle/working`, or save the notebook output as a Dataset). Drop it into
the local repo at `outputs/features_hoptimus/`, then run locally **unchanged**:

```bash
python scripts/train_path_a.py    --config configs/path_c.yaml
python scripts/make_submission.py --config configs/path_c.yaml
```

> Tip: this same cell extracts *any* frozen backbone — swap the model/normalization
> (Phikon: `transformers.AutoModel.from_pretrained("owkin/phikon")`, CLS token
> `out.last_hidden_state[:,0]`, ImageNet norm) and the `CACHE`/backbone name.

---

## Workflow B — Phikon end‑to‑end fine‑tuning (needs a CUDA GPU)

Unfreeze Phikon with **discriminative learning rates** (head ≫ backbone) and train through
the backbone. Emits the `.npz` `make_submission.py` expects.

```python
!pip -q install torch transformers albumentations opencv-python-headless scikit-learn pandas numpy pillow
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA='/kaggle/input/<competition>'; dev='cuda'
IMG_MEAN,IMG_STD=(0.485,0.456,0.406),(0.229,0.224,0.225)
train_aug = A.Compose([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomRotate90(p=.5),
                       A.Normalize(IMG_MEAN,IMG_STD), ToTensorV2()])
eval_tf   = A.Compose([A.Normalize(IMG_MEAN,IMG_STD), ToTensorV2()])

class DS(Dataset):
    def __init__(s, paths, labels, tf): s.p,s.y,s.tf=paths,labels,tf
    def __len__(s): return len(s.p)
    def __getitem__(s,i):
        img=np.asarray(Image.open(f'{DATA}/{s.p[i]}').convert('RGB'))
        y=0. if s.y is None else float(s.y[i])
        return s.tf(image=img)['image'], torch.tensor(y)

class PhikonClassifier(nn.Module):
    def __init__(s):
        super().__init__()
        s.backbone = AutoModel.from_pretrained("owkin/phikon")
        s.head = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.2), nn.Linear(768,1))
    def forward(s,x):
        cls = s.backbone(pixel_values=x).last_hidden_state[:,0]
        return s.head(cls).squeeze(1)

df = pd.read_csv(f'{DATA}/train.csv'); paths=df.img_path.tolist(); y=df.label.values
test_paths = pd.read_csv(f'{DATA}/dummyTest.csv').img_path.tolist()
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); test_acc = np.zeros(len(test_paths))

for fold,(tr,va) in enumerate(skf.split(paths, y)):
    model = PhikonClassifier().to(dev)
    # discriminative LRs: head fast, backbone slow
    opt = torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-5},
                             {'params':model.head.parameters(),    'lr':3e-4}], weight_decay=1e-2)
    EPOCHS=6; steps=EPOCHS*((len(tr)//32)+1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5,3e-4], total_steps=steps, pct_start=1/EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.BCEWithLogitsLoss()
    tl = DataLoader(DS([paths[i] for i in tr], y[tr], train_aug), batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(DS([paths[i] for i in va], y[va], eval_tf),  batch_size=64, num_workers=2)
    best_auc, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in tl:
            xb,yb=xb.to(dev),yb.to(dev)*0.9+0.05          # label smoothing 0.05
            opt.zero_grad()
            with torch.autocast('cuda',dtype=torch.float16):
                loss=crit(model(xb),yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        model.eval(); ps=[]
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            for xb,_ in vl: ps.append(torch.sigmoid(model(xb.to(dev))).float().cpu().numpy())
        auc=roc_auc_score(y[va], np.concatenate(ps))
        if auc>best_auc: best_auc, best_state = auc, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'fold{fold} ep{ep} valAUC {auc:.4f}')
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        vp=[]; 
        for xb,_ in vl: vp.append(torch.sigmoid(model(xb.to(dev))).float().cpu().numpy())
        oof[va]=np.concatenate(vp)
        tl2=DataLoader(DS(test_paths,None,eval_tf),batch_size=64,num_workers=2); tp=[]
        for xb,_ in tl2: tp.append(torch.sigmoid(model(xb.to(dev))).float().cpu().numpy())
        test_acc += np.concatenate(tp)/5

# threshold on OOF (maximize F1) + save in the schema make_submission.py expects
grid=np.linspace(0.05,0.95,181); tau=float(grid[np.argmax([f1_score(y,(oof>=t)) for t in grid])])
np.savez('path_b_finetune.npz', y_true=y, oof_probs=oof, test_probs=test_acc,
         test_paths=np.array(test_paths), tau=tau, temperature=1.0)
print('OOF AUROC', roc_auc_score(y,oof), 'tau', tau)
```

Download `path_b_finetune.npz` into `outputs/oof_preds/`, point a config's
`outputs.oof_path` at it, and run `python scripts/make_submission.py --config <cfg>` locally.

---

## Colab vs Kaggle

| | Colab | Kaggle Notebooks |
|---|---|---|
| GPUs | T4 (free), L4 / A100 (Pro) | P100 or **T4 ×2**, free |
| Quota | session-based, can disconnect | ~**30 GPU‑hours/week**, persistent `/kaggle/working` |
| Data | mount Google Drive | attach a Dataset / competition data directly |
| Secrets | n/a (use `login()`) | **Add‑ons → Secrets** for your HF token |
| Best for | quick A100 runs | longer jobs, easy competition data, free 2×T4 |

For this project's scale (≈10k patches), a single T4 fine‑tunes Phikon in well under an hour.
