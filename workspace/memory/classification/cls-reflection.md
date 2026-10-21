# Reflection Log

## Final Ensemble Result
- 12 GCN seeds (filtered from 18, threshold 0.425)
- Bias: [0.65, -0.3, 0, 0, 0, 0, -0.5, 0.6]
- Distribution matches training well: C1=22.2% (train 22.2%), C6=24.7% (train 25.1%)
- All classes ≥ 3% ✓

## Key Learnings

### Architecture Ranking (Phase 1)
- GCN (0.4502) >> GAT (0.3995) > SGC (0.3742) > GraphSAGE (0.3693) > APPNP (0.3366) > GIN (0.2704)
- GCN dominates on this dataset. GAT underperformed despite DropEdge + Label Smoothing.
- Graph is directed and asymmetric — converting to undirected was essential.

### Grid Search (Phase 2)
- Best GCN config: lr=0.001, hd=256, do=0.3, de=0.2, ls=0.05, wd=5e-4, nl=2
- Lower lr (0.001) was better than Phase 1 default (0.01)
- Grid search with 448 configs was too aggressive — killed early

### Multi-Seed Ensemble (Phase 3-5)
- 10 initial GCN seeds: mean 0.4367, range [0.4175, 0.4510]
- Added 5 more: mean improved
- Best single seed: 0.4559 (lr=0.005, hd=128, do=0.3)
- Filtering low-quality seeds (threshold 0.425) improved ensemble
- GAT seeds (0.35-0.38) too low quality to include
- SGC seeds similarly too low

### Bias Calibration (Phase 4)
- Conservative bias (C0=0.60, C7=0.60) was insufficient
- Adding C1/C6 penalty (-0.3, -0.5) aligned distribution with training
- Label propagation made distribution worse (amplified majority classes)
- Key insight: must penalize over-predicted classes (C1, C6), not just boost under-predicted (C0, C7)

### What Didn't Work
- Label propagation: reinforced majority classes, hurt distribution
- Graph structure features: no improvement in val_acc
- GAT/SGC ensembles: too low quality to contribute
- K-core decomposition: too slow
- Enhanced features (degree, PageRank, etc.): similar quality to base

### Hypothesis Check
- Original hypothesis: GCN would lead, multi-seed ensemble would be the main gain ✓
- GAT underperformance was unexpected — may need different tuning on this dataset
- Bias calibration needed both boost (C0, C7) AND penalty (C1, C6) for good distribution
