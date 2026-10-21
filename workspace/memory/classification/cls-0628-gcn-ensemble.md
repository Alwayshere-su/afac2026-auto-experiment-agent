# GCN Multi-Seed Ensemble for B1 Classification

## Experiment: Full Four-Phase Pipeline on B1

### Hypothesis
GCN would be the dominant architecture on this medium-sized graph (7650 nodes, 512 dense features, 8 classes) due to its stability with proper regularization. Multi-seed ensembling would be the primary source of score improvement, with bias calibration needed to handle class imbalance.

### Data Characteristics
- 7650 nodes, 512 dense features, 8 classes, 6120 train / 1530 test
- Directed, asymmetric graph — converted to undirected (A + A^T, binarized)
- No self-loops — added manually
- Class imbalance: C1=22.2%, C6=25.1%, C0=5.0%, C7=4.2%
- No isolated nodes (min degree=8)
- RTX 3090 24GB — batch_size not a concern for full-graph training

### Phase 1: Architecture Screening
| Architecture | val_acc | Notes |
|---|---|---|
| GCN | 0.4502 | Best, used DropEdge=0.2, LS=0.05, SWA |
| GAT | 0.3995 | heads=8, DropEdge=0.3, LS=0.1 — significantly behind GCN |
| SGC | 0.3742 | K=2, simple baseline |
| GraphSAGE | 0.3693 | Mean aggregation |
| APPNP | 0.3366 | K=10, alpha=0.1 |
| GIN | 0.2704 | MLP kernel, near-random |

Top 3: GCN, GAT, SGC

### Phase 2: Grid Search (Partial)
- GCN grid: lr=[0.001,0.003,0.005,0.01], hd=[128,256,512], do=[0.3,0.5,0.7], de=[0,0.1,0.2], ls=[0,0.05], wd=[1e-4,5e-4]
- Best GCN config: lr=0.001, hd=256, do=0.3, de=0.2, ls=0.05, wd=5e-4, nl=2 — val_acc=0.4502
- Killed early at 80/448 configs — no improvement beyond Phase 1 default

### Phase 3: Multi-Seed Ensemble
- 10 GCN seeds with best config: val_acc range [0.4175, 0.4510], mean 0.4367
- 5 more GCN seeds: val_acc range [0.4126, 0.4404]
- 3 GAT seeds: val_acc [0.3546, 0.3840] — excluded
- 3 SGC seeds: val_acc [0.2810, 0.4020] — excluded
- 5 diverse-config GCN seeds: val_acc [0.4134, 0.4559] — best single seed 0.4559 (lr=0.005, hd=128, do=0.3, wd=1e-4)

### Phase 4: Bias Calibration
- Conservative bias [C0=0.60, C7=0.60] gave C0=4.9%, C7=3.3% but C6=34.7%
- Adding C1/C6 penalty [-0.3, -0.5] aligned with training distribution
- Final bias: [0.65, -0.3, 0, 0, 0, 0, -0.5, 0.60]
- Label propagation made distribution worse (reinforced majority)

### Final Ensemble
- 15 GCN seeds (threshold ≥ 0.42)
- Equal-weight softmax probability averaging
- Bias calibration: [C0=+0.65, C1=-0.3, C6=-0.5, C7=+0.60]
- Distribution: C0=7.1%, C1=22.2%, C2=7.1%, C3=11.4%, C4=11.9%, C5=9.1%, C6=24.6%, C7=6.7%
- Matches training distribution closely

### GAT Grid Search
- Best GAT config: lr=0.001, hd=64, heads=4, do=0.3, de=0.2, ls=0.15 — val_acc=0.4289
- 5 GAT seeds with this config: val_acc [0.3930, 0.4191] — below GCN threshold, excluded from ensemble
- GAT fundamentally cannot match GCN on this dataset

### Mechanism Analysis
**Why GCN dominated**: The graph is medium-sized with dense features (512 fully dense dimensions). GCN's simple neighborhood averaging works well here — the features carry strong signal and graph structure provides meaningful smoothing. GAT's attention mechanism may overfit on this graph size with imbalanced classes.

**Why GAT underperformed**: Despite DropEdge + Label Smoothing, GAT's multi-head attention produced lower scores. The directed-to-undirected conversion may lose directional information that GAT could exploit. Alternatively, the class imbalance (C1+C6 = 47% of train) may cause attention heads to focus on majority classes.

**Why bias needed both boost AND penalty**: GCN naturally over-predicts majority classes (C6=36% vs train 25%). Simply boosting minority classes (C0, C7) doesn't fix this — the model confidently predicts C1/C6 for ambiguous nodes near the decision boundary. Penalizing the over-predicted classes pushes these ambiguous nodes toward minority classes, better matching the expected test distribution.

**Why label propagation failed**: On a graph where majority classes dominate, propagating labels reinforces this dominance. The LP matrix (D^-0.5 A D^-0.5) spreads probability mass from high-confidence majority nodes to neighbors, further suppressing minority predictions.

### Key Technical Decisions
1. Undirected conversion (A + A^T, binarize) + self-loops
2. Feature z-score standardization
3. SWA (start at 60% of epochs, running average with 1/n decay)
4. Label smoothing (0.05)
5. DropEdge (0.2) for GCN
6. CosineAnnealingLR with T_max = epochs * 1.3
7. AdamW (lr=0.001, wd=5e-4)
8. Early stopping (patience=50, warmup=30)
9. Gradient clipping (max_norm=1.0)
10. Degree-aware train/val split
