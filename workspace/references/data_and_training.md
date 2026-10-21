# Data Format and Training Guide

## Classification Task Data Loading

The data is a NumPy compressed file (e.g., A1.npz / B1.npz), containing a sparse matrix in CSR format. Key fields:

| Field                              | Meaning                   | Shape             |
| ------------------------------- | ------------------------- | ----------------- |
| attr\_data/indices/indptr/shape | Node feature matrix (CSR sparse) | (num_nodes, feature_dim) |
| adj\_data/indices/indptr/shape  | Adjacency matrix (CSR sparse)    | (num_nodes, num_nodes)  |
| labels                          | Node labels, test nodes = -1     | (num_nodes,)      |
| train\_idx                      | Node indices available for train/val | (N\_train,) |
| test\_idx                       | Test node indices to predict     | (N\_test,)  |

**Loading example ideas**:

```
d = np.load("A1.npz")   # For B track use B1.npz
# CSR -> Dense: sp.csr_matrix((data, indices, indptr), shape=shape).toarray()
# Features: attr_* -> (N, F) dense float32
# Adjacency: adj_* -> (N, N) sparse (keep CSR for model)
# labels: read directly, -1 are test nodes (exclude from num_classes calculation)
```

## Recommendation Task Data Files

### train.csv (training set, 40000 rows)

| Column              | Meaning               | Example          |
| ----------------- | --------------------- | ---------------- |
| uid               | User ID               | u000001          |
| target\_iid       | Target item (label)   | i020992          |
| item\_seq\_raw    | Raw interaction sequence  | "i20595,i20595"  |
| item\_seq\_dedup  | Deduplicated sequence     | "i20595"         |
| item\_seq\_counts | Sequence with counts      | "i20595:2"       |

### test.csv (test set, 10000 rows)

Same as train.csv but without the target\_iid column.

### user.csv (user features, 50000 rows)

uid + u\_cat\_01 ~ u\_cat\_08 (8 anonymous categorical features, values are integers)

### item.csv (item features, 2156 rows)

iid + i\_cat\_01/02/03 (categorical features) + i\_bucket\_01 (bucket feature)

## Output Format Specification

### Classification Task Output CSV (A1.csv / B1.csv)

```
test_idx,label
18,3
19,7
```

- test\_idx must match the official test\_idx
- label is an integer class number (based on actual number of classes in the data; A track typically 0-9)
- No quotes, no extra columns

### Recommendation Task Output CSV (A2.csv / B2.csv)

```
uid,prediction
u000001,"i000001,i000002,i000003,i000004,i000005,i000006,i000007,i000008,i000009,i000010"
```

- uid must match the official test.csv
- prediction is 10 iids separated by English commas, sorted by confidence descending
- Because it contains commas, the prediction column will be auto-quoted by CSV
- No duplicate iids for the same user

## Training Script Specification

Every training script written by the Agent must be a fully self-contained .py file (no dependency on any pre-written bootstrap):

1. Explicitly import all required libraries: `import numpy as np; import pandas as pd; import torch; from pathlib import Path` etc.
2. Use `Path(__file__).resolve().parent.parent` to locate the project root, or use relative paths directly `workspace/data/...`
3. Use `torch.cuda.is_available()` to detect GPU, define `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`
4. Include complete data loading logic
5. Include complete model definition (PyTorch nn.Module)
6. Include complete training loop (train/val split, early stopping, LR scheduler)
7. After training, generate the corresponding output CSV (e.g., A1.csv / B1.csv) into `workspace/outputs/...`
8. **Save a full checkpoint** (do not save bare state\_dict only). Save immediately upon best validation, use a timestamp directory variable, never hardcode path strings:
   ```python
   # Create save_dir before training starts
   timestamp = time.strftime("%Y%m%d_%H%M%S")
   save_dir = OUT_DIR / timestamp
   save_dir.mkdir(parents=True, exist_ok=True)

   # Save full checkpoint upon best validation (inside the training loop)
   torch.save({
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'epoch': best_epoch,
       'best_val_acc': best_acc,      # classification uses best_val_acc
       'best_val_ndcg': best_ndcg,    # recommendation uses best_val_ndcg + best_val_hr
       'best_val_hr': best_hr,
       'config': {                     # all hyperparameters, keys in UPPERCASE
       }
   }, save_dir / "best_model.pt")

   # Also save a standalone config file (to easily inspect hyperparameters without loading the model)
   import json
   json.dump({
       **config_dict,                    # same hyperparameters as above
       'best_epoch': best_epoch,
       'best_val_acc': best_acc,        # or best_val_ndcg / best_val_hr
   }, open(save_dir / "config.json", "w"), indent=2)

   # Loading during inference
   ckpt = torch.load(save_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
   model.load_state_dict(ckpt['model_state_dict'])
   ```
   Note: PyTorch 2.6+ defaults to `weights_only=True`, loading a checkpoint containing numpy values will raise UnpicklingError, so you must explicitly pass `weights_only=False`.
9. Print key metrics (val\_acc or val\_ndcg) for trajectory logging

## Common Error Troubleshooting

### CUDA: device-side assert triggered / embedding index out of bounds

**Root cause**: The `vocab_size` of the Embedding layer `nn.Embedding(vocab_size, dim)` is too small. Feature values go out of range after +1 or +2.
**Diagnosis method**: Print `max(index)`, `len(embedding_weight)`, check the difference. Common off-by-one: `i_bucket_01` values start from 1, verify whether `int(max_value) + 1 == vocab_size` after checking the max value in the data.
**Fix**: Set vocab\_size to `int(max_value) + 2` to leave enough headroom (padding 0 + all values).

### RuntimeError: mat1 and mat2 shapes cannot be multiplied (AxB and CxD)

**Root cause**: Input/output dimension mismatch in a fully connected layer. Usually the input dimension of `nn.Linear` was not adjusted after reshape/flatten.
**Fix**: Use edit\_file to change the in\_features of Linear, or add `.reshape()` inside forward.

### ValueError: Must have equal len keys and value when setting with an iterable

**Root cause**: Using `df.iloc[i, col] = []` to assign an empty list to a single cell causes pandas to treat `[]` as an iterable and unpack it.
**Fix**: Use `df.at[idx, col] = ''` or `df.loc[idx, col] = ''` instead.

### FileNotFoundError: workspace\data\rec\train.csv

**Root cause**: 99% of the time this is because `cd workspace && python code/xxx.py` was used, changing the working directory.
**Fix**: Do not use cd; execute directly with `python workspace/code/xxx.py` from the project root.

- Move models and tensors to GPU with `.to(DEVICE)`
- Data preprocessing (pandas/numpy operations) should be done on CPU
- Do not make batch size too large (~8GB VRAM).
- For classification tasks using full-graph GNN, pay attention to the adjacency matrix size (N x N can be very large)

