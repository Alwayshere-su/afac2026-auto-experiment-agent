# Sequence Recommendation Method Reference

## Data Characteristics

**Must do in the first round: Use execute_code to explore data yourself and obtain the following key statistics——**

- Training set/test set sizes, total number of products, user/product feature column names and vocab size
- Sequence fields: `item_seq_raw` (original), `item_seq_dedup` (deduplicated), `item_seq_count` (with counts)
- **Cold-start ratio**: proportion of users with empty sequences in the test set (affects validation truncation distribution design)
- **Repurchase rate**: proportion of targets that appear in the user's historical sequence (affects repurchase boost strategy)
- **Target concentration**: how many different target products appear in training (affects loss selection and long-tail strategy)
- Evaluation metric: NDCG@10

## Optional Methods

### Sequence-based Models (GRU4Rec type)

- Core idea: treat user interaction sequences as temporal data, model sequence patterns with RNN/GRU
- Input: sequence of user's most recent N interacted product IDs (padded to fixed length)
- Output: probability distribution over next interacted product (num_items+1 classification, +1 for padding)
- Key hyperparameters: embed_dim (product embedding dimension), hidden_dim (GRU hidden layer dimension), num_layers, seq_length (sequence truncation length)
- Training techniques: sequence augmentation (randomly truncate prefixes to simulate cold-start scenarios), Focal Loss (focus on hard samples)

### User Feature-based Models (UserFeat type)

- Core idea: ignore sequence, predict product preferences using only user demographic features
- Naturally suitable for cold-start users (no sequence)
- Input: user's 8 categorical features → Embedding → MLP → product probabilities
- Key hyperparameters: embed_dim (feature embedding dimension), hidden_dims (MLP layer dimension list)
- Can incorporate DCN (Deep & Cross Network) cross layers to enhance feature interaction

### Fusion Strategies

- **Hard Switch**: empty sequence → UserFeat, non-empty sequence → sequence model
- **Soft Blend**: weighted mix of two model logits by sequence length
  - Shorter sequence → higher UserFeat weight
  - Longer sequence → higher sequence model weight
- **BlendNet** (recommended): feed normalized sequence length into a small MLP (e.g. 1→32→1, Sigmoid), output fusion weight alpha, fused = alpha × gru_logits + (1-alpha) × user_logits. Lets the model adaptively learn fusion weights, better than fixed formulas.
- **Per-Bucket Soft Blend** (alternative, more stable): bucket by sequence length (empty seq / len=1 / len=2-3 / len>=4), each bucket uses a different fixed fusion weight. Empty sequence bucket has highest UF weight (nearly pure UF), long sequence bucket has highest GRU weight. Advantages over BlendNet: no learning needed, behavior is predictable, easy to debug. If BlendNet output is constant and doesn't vary with seq_len, can fall back to this approach.
- **Multi-Seed Ensemble**: train multiple models with different random seeds, weighted average of logits
- **Popularity Prior**: for cold-start users, mix in the global product popularity distribution from the training set
- **Co-visitation**: count conditional probability of "last interacted product A → target product B" as an additional signal. Can be extended to multi-dimensional: co-vis probabilities for last item, second-to-last, recent window respectively. Bayesian smoothing can reduce sparse noise but will weaken signal strength.
- **Repurchase Boost**: weight products that appeared in the user's history higher (high-frequency products more likely to be repurchased)

### Post-processing Parameter Search

- **Must use grid search, no greedy search allowed**: post-processing parameters have strong interactions (e.g. co-vis weight and threshold are coupled). Searching parameters one by one greedily will fall into local optima. Should do coarse-grained grid search in parameter space (e.g. 5-10 values per dimension), then fine-search around the optimal region.
- **Fixed validation set split seed**: must use the same train/val split seed across experiments, otherwise scores are not comparable.
- **Co-visitation statistics use training set only, and the validation subset for grid search must be split off before building co-vis**: the co-vis matrix must not touch the validation set or test set during construction, otherwise information leakage causes inflated local scores. **More importantly**: the holdout validation subset used for grid search must be split from the training set before building co-vis — if co-vis is built first and then holdout is split, the holdout users' targets are already written into the co-vis matrix, and post-processing directly pushes the answer to the top, inflating NDCG by 0.1+.

## Training Strategy Reference

- **Validation set construction (critical!)**: cannot just do binary truncation (empty vs full) — this misses users with intermediate-length sequences, causing inflated local NDCG. Must simulate the **full distribution** of sequence lengths in the test set, with multi-level truncation in both training and validation. See "Validation Set Distribution Simulation" below for details. **Validation set must use different users from the training set** — same users with only changed truncation lengths still means the model has seen this person, causing inflated Val NDCG. Correct approach: split out a portion of users from the training subset for validation, with the two sets having no overlapping users.
- **Optimizer**: AdamW, moderate learning rate and weight decay
- **Early Stopping**: stop when validation NDCG stops improving
- **Batch Size**: adjust by GPU memory; too large or too small both hinder convergence
- **Sequence Parsing**: convert "i00001,i00002" to int list [1,2], use 0 for padding

### Validation Set Distribution Simulation (Extremely Important)

The test set's sequence length distribution is not binary (empty vs full), but a continuous distribution. The truncation strategy during validation and training must simulate this distribution, otherwise:

- **Binary truncation only**: users with sequences use full history for prediction → inflated local NDCG
- **No multi-level truncation during training**: model only trains on full sequences → no generalization ability for short sequences

**Must do in first round: estimate the distribution proportions by analyzing the sequence length distribution of users in test.csv** (empty sequence ratio, len=1 ratio, short sequence ratio, full sequence ratio), and set validation truncation distribution accordingly. Do not hardcode fixed proportions — different datasets may have different distributions.

**Correct multi-level truncation strategy** (validation set, fixed seed ensures consistency across epochs):

- Divide users into multiple levels according to estimated distribution proportions: fully empty / keep only last 1 / keep last 2-3 / keep full sequence

**Sequence augmentation during training** (random each epoch, data augmentation):

- Training augmentation is more aggressive than validation (more short sequences), aiming to ensure the model sees enough samples at all lengths
- Design proportions for each level based on the estimated distribution

**Implementation notes**:

- Validation truncation must pre-generate a mask with a fixed seed in `__init__`, ensuring consistency across epochs
- Training augmentation randomly decides in `__getitem__`, different each epoch, serving as data augmentation
- Multi-level truncation is crucial for fusion strategy learning — if the model only trains on "empty" and "full" lengths, the fusion weight is ineffective for intermediate lengths

## Iterative Optimization Strategy

Recommendation task iteration has a clear convergence path; avoid repeatedly changing direction:

**Correct workflow: Architecture exploration → Intra-architecture grid hyperparameter search → Add seed ensemble → Post-processing grid re-search**

This is the standard four-stage workflow, must be strictly executed in order, no skipping stages. Each new dataset starts again from Stage 0.

**Stage 0: Architecture Exploration (Mandatory in First Round!)**

- Fix a single seed, train one seed per architecture prototype and compare no-PP NDCG
- The following four dimensions form the core design space. The Agent decides the number of prototypes and specific combinations independently; suggest starting with 4-6 prototypes covering complementary directions:
  1. **Sequence column**: `raw` (retain duplicates, preserve interaction intensity signals) vs `dedup` (deduplication, prevents model from over-focusing on a few high-frequency repeated products, mitigates preference for popular products)
  2. **Loss function**: `CE` (standard cross-entropy, requires label_smoothing to prevent overfitting) vs `Focal` (down-weights easy samples, automatically focuses on hard samples. When there are many products, highly concentrated targets, and severe long-tail, Focal is naturally more suitable — because most products have never been targets, CE will be dominated by high-frequency targets)
  3. **Product features**: with (concatenate i_cat_01/02/03 + i_bucket_01 embeddings, increase product representation information) vs without (only item ID embedding, simplified model, fewer parameters, prevents dependence on product popularity features rather than user behavior)
  4. **Fusion method**: `BlendNet` (adaptive learning of fusion weight — feeds normalized sequence length into MLP to output α, model autonomously discovers optimal fusion ratio. Advantage: flexible. Disadvantage: α may not vary with seq_len and degenerate to a constant) vs `soft_blend` (bucket by sequence length, each bucket has fixed fusion weight, Stage 3 grid search determines. Advantage: stable and predictable, no learning dependency)
- The above four dimensions are not isolated — they have complementary relationships. E.g., dedup + Focal naturally pair together (dedup mitigates high-frequency bias, Focal handles remaining long-tail), no product features + high dropout naturally pair together (simplified representation needs stronger regularization compensation). When designing prototypes, package these dimensions into internally coherent combinations, rather than comparing each dimension independently
- Goal: find the best architecture direction on this dataset, not stack seeds
- Forbidden: training one architecture then adding seeds right away — if the initial architecture is wrong, all subsequent optimization starts from the wrong foundation
- **⚠️ Data distribution differs per dataset (product count, cold-start ratio, repurchase rate). The optimal architecture from the previous dataset may not apply to the new dataset. Must re-run Stage 0, do not directly reuse old conclusions.**

**Stage 1: Intra-architecture Grid Hyperparameter Search**

- Lock in the optimal architecture selected from Stage 0, do not change model structure
- Grid search training hyperparameters on a single seed (seed fixed): LR, WD, DO, label_smoothing, etc.
- Goal: find the single-model configuration with the highest no-PP NDCG under this architecture
- Forbidden: using different seeds for hyperparameter search — seed variation introduces noise, making it impossible to tell whether hyperparameters or seeds are driving the effect
- **⚠️ If no-pp improvement after grid search < 0.01 → the current architecture direction is wrong, not a hyperparameter problem. Immediately return to Stage 0 and try a different architecture prototype from scratch.** Searching hyperparameters on the wrong architecture is just wasting time — the ceiling is locked, no amount of configuration search will break through. If the current prototype can't deliver big improvements via search, go try the next one. Only proceed to Stage 2 if all prototypes have been searched with no breakthrough.
- **Grid hyperparameter search ≠ trying two values per axis**: each dimension should have at least 5-8 values, covering at least 3-5 dimensions searched simultaneously. Trying two values per axis is not a grid, it's guessing. If manual search is too inefficient, write a script to auto-run full combinations and compare results.
- **Core search dimensions and ranges**: the following dimensions have relatively large impact on single-model no-pp NDCG; Stage 1 grid search should cover them as much as possible —
  - **LR (learning rate)**: cover at least 3 orders of magnitude, at least 3 points within each order. The optimal LR could be on any order of magnitude
  - **WD (weight decay)**: cover at least 3 orders of magnitude. WD and LR have strong interaction — high LR needs low WD to cooperate, low LR can tolerate high WD. If WD only searches the high-value range, gradients are suppressed and the model simply cannot learn
  - **Dropout**: cover the full range from low to high. Single-layer GRU internal dropout = 0 (PyTorch enforced), external dropout is the only regularization mechanism
  - **embed_dim**: product embedding dimension, affects product distinguishability. More products need higher dimensions, cover at least 2-3 tiers
  - **hidden_dim**: GRU hidden layer dimension, affects sequence modeling capacity. Cover at least 2-3 tiers
  - **num_layers**: GRU layer count (1 or 2). Increasing layers increases capacity and overfitting risk simultaneously
  - **seq_length**: cover at least small/medium/large three tiers. Longer sequences provide more historical information
  - **batch_size**: affects training stability and convergence speed, cover at least 2-3 tiers
  - **label_smoothing**: smoothing coefficient for CE (Focal typically doesn't need it), cover at least 3-4 values
  - Focal Loss's gamma is also a search dimension: do grid comparison across multiple gamma values. gamma=0 is equivalent to CE — so this search automatically includes CE vs Focal comparison
- When full combination search is infeasible, use a two-stage strategy: ① coarse search with 2-3 values per dimension to lock in approximate range ② fine search around the optimal region. Or use random sampling instead of full combinations
- **⚠️ If no-pp NDCG improvement after grid search < 0.01?** The search range hasn't covered the optimal region — extend one order of magnitude in both directions and re-search. If no improvement even after extension, go back to Stage 0 and switch architecture prototype. Don't repeatedly fine-tune within the same parameter range — that's just looking for signal in noise
- **Fallback principle**: If multiple prototypes from Stage 0 have extremely close no-PP NDCG (e.g. <0.003 gap), even if B slightly beats A, if B's Stage 1 search shows little progress, the runner-up prototype should also go through Stage 1 grid search. Close scores mean close potential — don't completely abandon the runner-up prototype just because of a marginal initial gap. Whoever has the highest score after Stage 1 search is the true optimal architecture.

**Stage 2: Add Seed Ensemble**

- Lock in Stage 1's optimal configuration, only vary random seeds
- Train multiple seeds and do logits average ensemble
- Seed count 3-7, marginal returns approach zero beyond 7

**Stage 3: Post-processing Grid Re-search**

- After seed count changes, optimal post-processing parameters (cw/rb/pm/pb) also change
- Re-do grid search for post-processing parameters on the ensemble logits
- Generate final A2.csv

**Forbidden behaviors:**

- Switch to a new architecture / new loss / new fusion method each round → can never converge
- Assume direction is correct just because local score went up → look at online, not local
- Greedy search for post-processing parameters → must do joint grid search
- Switch to a new batch of seeds each round → old seeds that haven't gone through post-processing grid search are wasted if discarded

**Convergence signal:** When post-processing parameters show flat performance in grid search (wide optimal region), and adding more seeds shows diminishing marginal returns approaching zero, the current architecture has reached its ceiling.

### Training Strategy

- **Cold-start simulation (must do in both training AND validation!)**: During training, randomly truncate sequences to empty at the test set's cold-start ratio, letting the model learn to handle empty-sequence users. **Validation must also do the same proportion of cold-start simulation** — all users in train.csv have sequences; if validation doesn't truncate, local NDCG will be severely overestimated.
- **Fixed validation split seed**: ensures validation set is completely consistent across different experiments, scores are comparable.
- **Gradient clipping + AdamW + early stopping**: standard training stability measures.

### Multi-Seed Ensemble

- Same architecture, same hyperparameters, only change random seed to train multiple models
- During inference, take arithmetic mean of logits from multiple models (not voting on ranking results)
- Marginal returns remain positive when seed count increases from 3 to 7.
- **Seed count cap at 7**: marginal returns approach zero beyond 7 seeds. Don't endlessly add seeds — adding seeds is a means, not an end.
- After adding seeds, post-processing parameters must be re-grid-searched (different seed count → different optimal post-processing parameters)

### The Correct Way to Improve Ensembles: Optimize Generalization of Individual Seeds

Adding more seeds only reduces variance, doesn't solve the problem of poor single-model generalization. **More effective than adding seeds: pick the best-performing 2-3 seeds from the ensemble, individually debug their training methods, improve their generalization, then fuse again.**

Directions for debugging individual seeds: (bracketed values are just examples, you can try other values too)

- **Increase regularization**: raise dropout, increase weight decay, reduce label smoothing
- **Adjust learning rate schedule**: cosine annealing, warmup, more aggressive learning rate decay
- **Data augmentation**: change cold-start truncation probability, randomly shuffle sequence order, randomly drop products from sequences
- **Compare checkpoints from different epochs**: best epoch doesn't necessarily have the best generalization, try weights from earlier/later epochs
- **Loss function variants**: Focal Loss (focus on hard samples), Label Smoothing cross-validation

Workflow:

1. Multi-seed ensemble → get baseline
2. Pick the 2-3 seeds with the best holdout performance from the ensemble
3. Run regularization/training strategy ablation experiments on these 2-3 seeds individually (single variable!)
4. Replace the corresponding seeds in the original ensemble with the improved seeds
5. Re-grid-search post-processing parameters
6. Compare no-post-processing NDCG of old vs new ensemble — if improved, direction is right; if not, try a different seed or a different debugging direction

**Key point**: Large local→online gap indicates poor model generalization. Narrowing the gap is more important than inflating local numbers.

### Post-processing Pipeline

Post-processing has significant impact on the final score.

**Core principle: Post-processing is only done once at the final ensemble stage, NOT per individual model!**

```
Correct workflow:                          Wrong workflow:
train(seed=42) ─┐                         train(seed=42) → post-process → separate A2
train(seed=123) ┤                         train(seed=123) → post-process → separate A2
train(seed=456) ┤                         train(seed=456) → post-process → separate A2
    ...         │                             ...
    Average all model logits                  (Waste: co-vis repeatedly built
       ↓                                           cw/rb repeatedly searched
    Post-process (once only!)                      Single-model optimal ≠ ensemble optimal)
       ↓
    A2.csv
```

Doing post-processing per individual model is a waste of time and directionally wrong — the optimal cw/rb found for a single model doesn't apply after ensemble, and the co-visitation matrix gets built repeatedly. Post-processing should target the fused logits after ensemble with a one-time grid search.

#### 1. Co-visitation (Co-occurrence Signal)

- **Construction method**: count conditional probability of "user's last N interacted products → target" in the training set
- **Exponential decay weighting**: closer interactions have higher weight (decay^position)
- **Minimum count filtering**: filter out source products with occurrence count below threshold to reduce noise
- **Application method**: additive enhancement of model logits, weight determined by grid search

#### 2. Repurchase Boost

- Products that have appeared in the user's historical sequence get a fixed bias added to the final score
- Leverages data characteristic: in scenarios with high repurchase rate, historical products have a higher probability of reappearing

#### 3. Grid Search (Critical!)

- **Holdout must be split off before model training!** Two layers of isolation, both indispensable: ① split a portion from train.csv as holdout ② use the remaining portion to train the model (model never sees holdout users) ③ use the remaining portion to build co-visitation matrix ④ grid search cw/rb on holdout. **Common mistake**: use full train to train model → then split holdout from it to search post-processing → holdout users have already been seen by the model → no-post-processing NDCG is higher than training val NDCG. If cw=0, rb=0 yields holdout NDCG much higher than training val NDCG, this is the leakage.
- **Co-visitation matrix isolation**: if co-vis is built before splitting the validation set, validation set users' targets are already written into the co-vis matrix — post-processing directly pushes the correct answer to the top, inflating NDCG by 0.1+, equivalent to self-deception.
- Correct workflow: train.csv → split a portion as holdout (untouched by model and post-processing) → remaining portion used for model training + co-visitation matrix construction → grid search cw/rb on holdout.
- Search dimensions: co-visitation weight (cw) × repurchase boost (rb)
- **Must grid, no greedy**: cw and rb have strong interaction; greedy per-dimension search falls into local optima
- **cw upper limit control**: cw should not be too large. If the optimal cw from grid search hits the ceiling (search upper bound), it means co-visitation is replacing the model's predictions rather than supplementing them — the model's own ranking is being completely overridden. In this case, prioritize improving model quality so that post-processing only does fine-tuning rather than dominating.
- Post-processing gain should have a reasonable upper bound. If the improvement is too large, suspect co-vis weight is too high or there is leakage.
- **⚠️ Mandatory investigation of anomalous gains**: if post-processing NDCG improvement is too large (far beyond reasonable range), immediately stop and investigate the following four items; do not directly trust the results:
  1. **Co-visitation nested loop bug (most common!)**: check if `apply_covisitation` has an outer `for pos in range(n_positions)` + inner `for p in range(n_positions)` double loop — the inner loop already aggregates all positions, the outer loop adds again, amplifying the boost by ~`n_positions` times (i.e. 3x). Correct approach is **single loop**: for each position p, take `co_vis[p][seq[-p-1]]`, weighted sum, then `boosted[i] += cw * boost_vec`, only once.
  2. **Whether co-vis matrix touched holdout users**: check if the DataFrame passed to co-vis construction truly only contains `train_sub` (without holdout uids). If co-vis is built before splitting holdout, holdout users' targets are already written into the matrix, and post-processing directly pushes the answer to the top.
  3. **Repurchase boost formula reasonableness**: check for any extraneous amplification factors (e.g. `len(set(seq))` making boost scale linearly with user browsing breadth). Repurchase boost should be `rb * (item_count / total_count)`, with no additional multipliers.
  4. **Whether grid search cw hits the ceiling**: if the optimal cw exactly equals the search upper bound, the true optimum may be even higher — co-vis is replacing the model's predictions, and model quality itself is insufficient.
- After finding the optimal combination, fine-search around it

#### 4. Advanced Co-visitation Techniques

Basic co-visitation only uses the last N products for conditional probability; there are two improvement directions:

**Bayesian Shrinkage**

- Problem: co-visitation statistics for low-frequency source products (few occurrences in training set) are unreliable — conditional probability from small samples has high noise
- Solution: shrink co-visitation probability toward the global target product distribution
  - `P_smooth(target|src) = alpha * P_MLE + (1-alpha) * P_global`
  - `alpha = n / (n + lambda)`, where n is the total occurrence count of the source product, lambda is shrinkage strength
  - High-frequency source products → alpha close to 1 → trust own statistics
  - Low-frequency source products → alpha close to 0 → rely on global prior
- Implementation notes: lambda is a hyperparameter controlling shrinkage strength. Larger lambda means low-frequency products rely more on the global distribution

**Multi-position Co-visitation**

- Basic version only uses the last product (position=-1); multi-position version simultaneously uses the last 1st, 2nd, 3rd positions
- Build independent co-visitation matrices per position (`P(target | product at position_i)`)
- Apply with multi-position superposition, closer positions have higher weight (exponential decay)
- Recommended 2-4 positions; too many introduces noise
- **⚠️ Implementation iron rule — must use single loop**: multi-position superposition is a **single for loop**, not a double loop. Correct implementation:
  ```python
  # ✅ Correct: single loop, each position contributes its own co_vis[p], weighted sum, then += once
  for i in range(len(seqs)):
      boost_vec = np.zeros(num_items + 1)
      for p in range(min(n_positions, len(seqs[i]))):
          s = seqs[i][-(p + 1)]
          if 0 < s < co_vis[p].shape[0]:
              boost_vec += (decay ** p) * co_vis[p][s]   # co_vis[p] is the matrix for position p
      boosted[i] += cw * boost_vec                        # ← only added once!
  ```
  **Common mistake**: outer `for pos in range(n_positions)` + inner `for p in range(n_positions)` nested → inner loop already aggregates all positions, outer loop repeats → boost amplified by ~n_positions times. After writing the code, self-check with "is no-post-processing NDCG → post-processing NDCG improvement ≤ 0.06".

#### 5. Pop Match (Popularity Distribution Matching)

**Core idea**: models naturally tend to predict popular products (they have high frequency in training), causing long-tail products' targets to not make the ranking. Pop Match suppresses this bias by subtracting `alpha * log(P_target)` from logits.

**Operation**: count the probability P(item) of each product being a target in the training set, take the logarithm as a penalty term, subtract from the fused logits:

```
adjusted_logits = fused_logits - alpha * log(P_target)
```

- Popular products → large P → log(P) close to 0 → small penalty, basically unaffected
- Unpopular products → small P → log(P) is a large negative number → subtracting it increases logits → ranking improves
- Products that have never been targets → set P to `1/(total_samples)` → receive maximum boost

**Bucketed alpha**: users with different sequence lengths need different strengths of pop match. Empty-sequence users need more popularity signal (smaller or even negative alpha), long-sequence users need more debiasing (larger alpha). Recommend using different alphas per sequence-length bucket.

**Effect**: Pop Match is one of the single most beneficial post-processing techniques, especially on data with concentrated targets (severe long-tail).

#### 6. Bucket-Differentiated Post-processing

Post-processing parameters should not treat all users uniformly. Bucket by sequence length, use independent parameters per bucket, enabling fine-grained matching for different user groups:

- **Bucket划分**: empty sequence / len=1 / len=2-3 / len>=4 (adjust based on data distribution)
- **Differentiated parameters**:
  - Blend weight: empty-sequence bias toward UF, long-sequence bias toward GRU
  - Repurchase boost: long-sequence users have experienced more products, boost can be higher
  - Pop match alpha: different buckets have different tolerance for popularity bias
  - Co-visitation boost: only users with sequences can leverage co-vis signals
- **Search strategy**: first coarse search with unified parameters, then fine-search optimal values per bucket. Interactions between bucket parameters are weak, so buckets can be optimized individually

### Cold-Start Specialized Handling

Cold-start users (empty sequence) rely entirely on UserFeat and statistical signals. The following techniques specifically enhance prediction quality for cold-start users:

**A. Global Popularity Fallback (Pop Blend)**

- For empty-sequence users, mix pure UF logits with global product popularity distribution
- `final = (1-beta) * uf_logits + beta * pop_logits`
- Rationale: when there is no behavioral signal, popularity is a reasonable prior. But UF already has predictive power, so beta should not be too large — it plays a supplementary rather than dominant role

**B. User Feature Conditional Probability (Feature-based Cold-Start)**

- Count the target product distribution for each user feature value in the training set: `P(target_item | user_feature_k = value_v)`
- For empty-sequence users, look up each of their user feature values to get multiple conditional distributions, take the average, and mix into final logits with a certain weight
- This supplements UF — UF learns nonlinear interactions, while feature conditional probabilities are stable pure-counting signals. The two combined are more robust than UF alone
- When implementing, convert to log space (log-probability) to be consistent with logits scale

**C. Sequence Augmentation (During Training)**

- When training GRU4Rec, don't only use full sequences; randomly truncate prefixes following a distribution consistent with the test set
- Truncate not only to empty, but also to very short (len=1~3), so the model sees various lengths
- This is key to enabling GRU to output reasonable logits in cold-start scenarios — if the model has never seen an empty sequence during training, GRU's output for an empty sequence is undefined

### UserFeat Independent Training

Joint training of UserFeat with GRU4Rec has advantages (BlendNet can learn), but also risks: UF's gradients are dominated by GRU, and UF's own parameters don't get fully optimized. An alternative is **training UF independently**:

- UF uses an independent optimizer (typically needs a higher learning rate than joint training)
- UF uses independent early stopping and scheduler
- Training objectives can be: ① classification (predict target item ID) ② **Embedding regression**: first train a GRU to extract item embedding matrix for all products, then have UF not predict discrete item IDs but directly regress item embedding vectors (MSE or cosine distance loss). Regression mode naturally avoids popular product bias because similarity computation doesn't consider item frequency.
- After independent training, UF weights are frozen; only GRU does multi-seed training. At inference, fuse with fixed rules (e.g. bucketed blend)

### Sequence Enhancement Techniques (During Training)

Do data augmentation on sequences when training GRU4Rec to improve model generalization and robustness to different sequence lengths:

- **Random prefix truncation**: simulate the test set's sequence length distribution (fully empty, very short, medium, etc.), letting the model see various lengths
- **Random product dropout**: randomly drop products from sequences with low probability, simulating incomplete observations
- **Random local shuffling**: swap adjacent products with small probability, reducing overfitting to strict temporal order
- **Compare different seq_column**: item_seq_raw (retain duplicates), item_seq_dedup (deduplicated), item_seq_counts (frequency-weighted expansion) can serve as inputs from different perspectives
- Augmentation is only done during training, not during validation or inference

### Repurchase Boost Frequency Weighting

Basic repurchase boost applies the same fixed bias to all historical products. A more refined approach is:

- **Frequency-Weighted**: products that appear more times in the user's history get larger boost. Because the target is often one of the user's most frequently interacted products
- Operation: count each product's occurrence frequency in the user's sequence, boost = base_boost * (freq_i / total_freq) * n_unique_items
- Difference from uniform boost: uniform boost gives the same bonus to products appearing once vs 10 times; frequency-weighted focuses more on high-frequency products
- Suitable for scenarios with high repurchase rate

### Model Tuning Diagnostic Workflow

When post-processing gain is too large (cw hits ceiling) or single-model no-post-processing NDCG is far below a reasonable baseline, first investigate the model itself using the following workflow; don't rush to add seeds or tune post-processing:

**Diagnostic signal → Direction → Single-variable validation**

| Signal                            | Possible Cause                | Try (change one thing at a time!)                |
| ----------------------------- | ------------------- | ---------------------------- |
| cw optimal value consistently hits ceiling                | Model ranking signal too weak, post-processing replacing model   | Reduce model + increase regularization                 |
| Single-model no-PP NDCG far below reasonable reference           | Inappropriate model capacity              | Shrink model first (shallow layers/narrow hidden), then gradually increase           |
| Adding seeds shows sharply diminishing marginal returns                   | Large single-model variance, severe homogenization         | Increase regularization (dropout/weight_decay) |
| Large variance between Val and holdout NDCG       | Overfitting to validation set              | Increase dropout, reduce model size              |
| Long-tail targets almost never recommended             | Loss dominated by high-frequency targets  | Switch to Focal Loss                |
| Cold-start user NDCG extremely low                 | Insufficient UF training or weak product representation      | Train UF independently, increase UF learning rate            |
| BlendNet alpha doesn't vary with seq_len | BlendNet hasn't learned effective fusion    | Fall back to bucketed fixed fusion weights                  |
| Top-k predictions concentrated on few high-frequency products          | Model overly dependent on product popularity rather than user behavior   | Remove product category features (simplify product representation), switch to deduplicated sequence column     |
| NDCG variance across sequence lengths extremely large         | Short-sequence users have insufficient info but long-sequence users also not fully utilized | Increase sequence truncation length (let more history of long-sequence users participate in encoding)    |

**Debugging principles**:

- **Shrink model first, then add regularization**: when encountering weak signals, the intuition is "model is not big enough." But on small datasets (tens of thousands of samples), overly large models easily overfit the validation set, appearing as decent val NDCG but poor holdout. First try reducing layers, shrinking hidden dimensions, increasing dropout; use a single seed to run through and compare no-PP NDCG.
- **Change only one variable at a time**: changing multiple parameters simultaneously makes attribution impossible. Change one → run one seed → compare no-PP NDCG → confirm direction → then change the next. **Exception: "shrink model" is one variable** — reducing layers, shrinking hidden dimensions, increasing dropout should be changed together because they are complementary (small models need strong regularization, large models need weak regularization). Changing them separately (e.g. only reducing layers without changing dropout) leads to underfitting or overfitting, producing misleading signals.
- **Judge by no-post-processing NDCG**: post-processing masks the real changes in model quality. When evaluating model changes, only look at no-PP NDCG.
- **Don't skip "shrink model" and jump straight to complex approaches**: multi-task learning, loss weighting, item2vec — these are considered only after the model is saturated. If the baseline model has never undergone architecture optimization, finish basic tuning first.
- **"Shrink model" is the highest-priority direction**: if the following directions have been tried one by one and all are ineffective — larger model, switching sequence column, Focal Loss, tuning label_smoothing — the problem is not in training techniques but in model capacity itself. At this point, must return to the first diagnostic table entry: reduce model + increase regularization (all three together).
- **Avoid premature seed addition**: when single-model architecture is not yet locked in, the benefit of adding seeds is illusory — seeds mask architecture problems. First lock in a good single-model configuration, then add seeds for ensemble.

**Sequence column selection**:

- `item_seq_raw` (original with duplicates) and `item_seq_dedup` (deduplicated) provide different perspectives. Deduplicated sequences prevent the model from over-focusing on a few high-frequency repeated products; original sequences retain interaction intensity signals. If top-k predictions are concentrated on high-frequency products, try switching sequence columns.

### Key Insights

1. **BlendNet is better than fixed weighting**: adaptively learned fusion weights outperform hand-crafted formulas based on sequence length (e.g. `seq_len/total_len`), because the model can discover nonlinear relationships.
2. **Multi-seed ensemble is a stable source of gain**: without changing architecture or adding parameters, simply adding seeds stably improves results — the most reliable iteration direction.
3. **Post-processing parameters have larger online impact than local**: for the same set of model weights, different post-processing parameters can cause online NDCG fluctuations of 0.02+. But don't be fooled by post-processing scores — the larger the post-processing gain, the weaker the model itself.
4. **Local→online mapping is not linear**: a local improvement of 0.01 does not equal an online improvement of 0.01. Overfitting to the validation set (repeated hyperparameter tuning) leads to inflated local scores without online improvement.
5. **No-post-processing NDCG is the model's true level**: the larger the post-processing gain, the weaker the model itself. Different datasets have different repurchase distributions; approaches that rely on post-processing as a crutch have poor generalization. **The goal of improving the model is no-post-processing NDCG, not post-processing NDCG.** If two consecutive rounds of experiments show no improvement in no-post-processing NDCG, the current architecture has reached its bottleneck — change the architecture rather than tuning post-processing parameters.

## Common Pitfalls

- **Empty sequence handling**: empty sequences are all-zero padding; the model must be able to handle this (item 0 is the padding token)
- **Validation distribution mismatch**: binary truncation (empty vs full) ≠ real test set distribution. The real test set has many intermediate-length users (len=1-3); using only binary truncation leads to inflated local NDCG. Must use multi-level truncation to simulate the full distribution; see "Validation Set Distribution Simulation" section for details.
- **Model training data leakage (diagnostic formula)**: if `holdout NDCG at cw=0, rb=0 − training val NDCG` is abnormally large, it means holdout users were already seen during model training. This is failure of two-layer isolation: the training script used the full train.csv, then the inference script split holdout from it — the model has already memorized holdout users' patterns. **Correct order**: train.csv → first split and save a portion (holdout, untouched going forward) → train model + build co-vis on the remaining portion → search post-processing on holdout.
- **Evaluation distribution inconsistency causing misdiagnosis as leakage**: abnormal gap could also be due to **different evaluation methods** rather than data leakage. Common scenario: training validation set uses multi-level truncation, but holdout evaluation uses full sequences. This causes holdout NDCG to be far higher than training val NDCG, but there's no information leakage — just different test difficulty. **Diagnostic method**: first confirm whether holdout users are truly outside the training set (check uid set intersection). If truly isolated, then check whether holdout evaluation uses the same truncation levels as training validation. **Solution**: holdout evaluation should also use the same multi-level truncation distribution as training validation (fixed seed) to make them comparable. Don't conclude leakage just from the gap value — first check isolation, then check truncation.
- **Local↑ Online↓ = Overfitting**: when model improvements raise local scores but online scores don't rise or even drop, it indicates overfitting to the validation set. At this point, should return to the historically best online configuration, rather than continuing to pile up local numbers in the current direction. Repeatedly tuning post-processing parameters on the same validation set is also a source of overfitting — the more rounds of parameter tuning, the lower the local→online conversion rate.
- **Product IDs start from 1**: i000001 corresponds to index 1, index 0 is reserved for padding
- **Output format**: the prediction column uses English commas to separate 10 iids, in descending order of confidence
- **Deduplication**: the same user's recommendation list should not contain duplicate product IDs

## Training Script Self-Check Checklist (Must verify each item before writing code!)

After writing the training script and before launching training, must confirm each of the following items one by one.

**Reusable utilities**: `workspace/code/eval_utils.py` provides evaluation and truncation functions; import and use directly, don't rewrite each time:

- `compute_ndcg_at_k(logits, targets, k)` — NumPy version, for inference/post-processing. Automatically detects and removes padding position
- `compute_ndcg_at_k_torch(logits, targets, k)` — PyTorch version, for validation during training
- `compute_hr_at_k(logits, targets, k)` / `compute_hr_at_k_torch(logits, targets, k)` — HitRate
- `make_trunc_mask(n, seed, dist)` — generate multi-level truncation mask, returns list[int]. Call during validation set `__init__`, truncate by level in `__getitem__`

### Model Structure

- [ ] **⚠️ When taking the last hidden state from GRU4Rec, must use** **`[:, -1, :]` (take the last position of the sequence), cannot use** **`seq_len - 1`** **indexing.** Sequences are left-padded: `[0,0,...,0, real_1,...,real_n]`. If using `gru_out[:, seq_len-1, :]`, with seq_len=10 you'd take position 9 (padding), not position 49 (last real item). `[:, -1, :]` always takes the last position, and has no effect for empty sequences (all padding). **Common mistake**: thinking seq_len is "the position of the last valid element," when it's actually "the number of valid elements" — under left-padding, the last valid element is always at the end of the sequence.
- [ ] **⚠️ Forbidden to use `pack_padded_sequence`!** Sequences are left-padded (padding at the front), `pack_padded_sequence` assumes right-padding — it takes the first `seq_len` positions of each sequence as valid, which ends up exactly taking the padding, skipping the real items. Correct approach: directly pass the full sequence through GRU (`gru_out, _ = self.gru(item_emb)`), no pack/unpack. GRU naturally handles padding — padding positions are 0 embedding, having minimal impact on hidden state. **Common mistake**: the Agent thinks `pack_padded_sequence` is a "better implementation," adds it, and the score plummets from 0.50 to 0.39.
- [ ] **⚠️ Does GRU4Rec forward actually use product category features** (not just defining `item_feat_embeds` and `feat_proj` but only using `item_id_embed` in forward)? Check: does forward look up `item_feat_embeds` for each item in the seq, does it call `feat_proj`, does it add to `item_id_embed` before feeding to GRU. **Common mistake**: everything is neatly defined in `__init__`, but not called anywhere in `forward` — product features are completely wasted, directly losing no-PP NDCG.
- [ ] **⚠️ Are product feature initialization methods (e.g. `_build_item_feat_table`) called after model creation?** Common pattern: an initialization method is defined in `__init__`, forward checks `hasattr(self, 'item_feat_ids')` to see if the buffer exists → if not, silently `return 0`. But `main()` forgets to call the method after creating the model → buffer is never registered → forward check fails every time → silently returns 0 → all architectures with product features effectively degrade to no product features, making Stage 0 architecture comparison completely invalid (with features vs without features NDCG is exactly the same). **Check method**: globally search for the method name in `main()` or the training script, confirm it is called after model creation and before training starts.
- [ ] Does UserFeatModel's `get_all_item_vectors` include product category features (not just ID embedding)?
- [ ] **⚠️ Do GRU4Rec and UserFeatModel share the same item representation (not just having UserFeatModel use** **`output.weight`** **for dot product)?** `output.weight` is the classification head weight (hidden_dim dimensions), `item_id_embed` is embedding (embed_dim dimensions). The two have different dimensions, different semantics, and gradients never flow through the same matrix — fusing two unrelated logits has no synergy, instead introduces noise, losing 0.03-0.04. Correct approach: UserFeatModel reuses GRU's `item_id_embed` + the same item vector after product feature projection.
- [ ] **⚠️ In GRU forward, does the item embedding pass through a Dropout layer before going into GRU?** If `_get_item_emb()` or forward doesn't call `self.dropout(item_emb)` after embedding and before GRU, the entire sequence encoding path lacks input-side regularization. Single-layer GRU (`num_layers=1`) internal dropout is forced to 0 by PyTorch. If this external dropout is also missing → zero regularization on the GRU path → overfits training set → holdout NDCG directly loses 0.02-0.03. **Check method**: search for `self.dropout` in forward, confirm it appears before the GRU call and is actually executed.

### Training Configuration

- [ ] Does the optimizer include parameters of all three sub-networks (GRU4Rec + UserFeat + BlendNet)?
- [ ] Is training loss computed using BlendNet's fused logits (not gru_logits or user_logits separately)?
- [ ] Does `CrossEntropyLoss` have `label_smoothing` parameter set (not the parameterless `CrossEntropyLoss()`)?
- [ ] Does validation set truncation use **multi-level distribution** (empty sequence / very short / short / full, multiple levels), rather than binary (empty vs full)? Are the proportions of each level estimated from the test.csv sequence length distribution?
- [ ] Is validation set truncation pre-generated with a fixed seed mask in `__init__` (ensuring cross-epoch evaluation consistency)?
- [ ] Is multi-level sequence augmentation done during training (randomly preserving 0-2/3-10/11-30/full, four lengths), rather than only binary truncation?
- [ ] **⚠️ Val→Holdout gap monitoring (overfitting prevention)**: The gap between single-model training val NDCG and holdout NDCG is the core metric for measuring overfitting. After training completes, must immediately evaluate single-model no-pp NDCG on holdout, compute gap = val_ndcg - holdout_ndcg. Small gap is normal, medium means mild overfitting needing more regularization, **large means severe overfitting requiring investigation**. Investigation directions (by priority): ① Is training augmentation sufficient (is aug_seq enabled, does augmentation intensity cover empty-sequence and short-sequence scenarios); ② Is weight_decay too low; ③ Is dropout too low; ④ Is early stopping patience too large, causing selection of an overfitted epoch; ⑤ Is the train/validation user split random, and is the validation set proportion sufficient.

### Post-processing

### Data Isolation

- [ ] Does the training script split holdout and save the uid list before training (not retroactively split after training)?
- [ ] **⚠️ Is the validation set split from** **`train_sub`** **again, rather than directly using** **`holdout_df`?** This is the most frequent mistake. `val_ds = ValDataset(holdout_df, ...)` means the model uses holdout for early stopping — every time it selects the best epoch, it's peeking at holdout. Correct approach: `train_sub` is further randomly split into a true training set + validation set; holdout is untouched throughout. Validation set is for early stopping and LR scheduler; holdout is only for the single final evaluation after training completes. If validation set = holdout, three consequences: ① gap analysis completely invalid (val≈holdout, gap always ~0); ② model selection is hijacked by holdout's truncation distribution; ③ during post-processing grid search, holdout has already been indirectly optimized by the model.
- [ ] Does the inference script reuse the holdout uid list saved during training (not randomly split a new one)?
- [ ] Does holdout evaluation use the same multi-level truncation as training validation (otherwise gap is inflated, post-processing parameters are inaccurate)?
- [ ] **⚠️ Holdout evaluation completeness (must do before grid hyperparameter search!)**: Before any evaluation on holdout, must complete the following three checks — all are indispensable:
  1. **Truncation consistency**: does `InferDataset` receive `trunc_mask` during holdout inference. No truncation → holdout no-pp NDCG inflated.
  2. **UID count**: print `len(holdout_uids)`, confirm it equals `train user count × holdout ratio`, not 0 and not the full set.
  3. **Post-processing zero-gain verification**: run a round of post-processing grid search on holdout. If no-pp NDCG is abnormally high and the post-processing search yields optimal cw=0, rb=0, pm=0, it means the no-pp evaluation itself has problems — post-processing can't find room for improvement because the baseline has been contaminated or the evaluation method is wrong.
- [ ] **⚠️ Abnormally low holdout no-pp NDCG investigation**: When holdout no-pp NDCG is noticeably lower than the historical baseline, the following four items are the most frequent root causes; investigate one by one in descending probability:
  1. **Inconsistent holdout_uids (most common!)**: Is the `holdout_uids.npy` loaded by the inference script the same file saved by the training script? If each training script independently splits its own holdout (without reusing an existing uid list), different scripts have different holdout splits, and evaluation baselines are incomparable. Correct approach: the first training script splits holdout and saves `holdout_uids.npy` to `OUT_DIR`; all subsequent scripts (training/inference) load this one copy. Print `holdout_uids[:5]` and compare with training logs for confirmation.
  2. **Inconsistent truncation mask seed**: is the seed in `make_trunc_mask(len(holdout_sub), seed=...)` fixed? Different seed → different truncation distribution → incomparable evaluation results. Must use the same fixed seed, consistent with historical inference scripts.
  3. **Model weight loading failure**: when loading `best_model.pt`, are the return values of load_state_dict and missing/unexpected keys checked? A model's config doesn't match the inference script → weights silently fail to load → random weight inference → NDCG plummets. After each model load, print `mcfg.get('best_val_ndcg', '?')` to confirm it matches training time.
  4. **Sequence column mismatch**: does `InferDataset`'s `seq_col` match the one used during training? Wrong column → evaluating a different task → NDCG abnormal.
- [ ] If gap > 0.05, did you first check uid intersection + truncation consistency, rather than directly concluding leakage?

### Post-processing

- [ ] Did you **not** directly generate A2.csv in the individual model training script?
- [ ] Did you **not** hardcode repurchase boost values in the training script (e.g. `logits += 0.1`)?
- [ ] Is the co-visitation matrix built from the **training subset** (not the full train.csv)?
- [ ] Was the holdout validation subset for post-processing grid search split off **before** model training and co-vis construction?
- [ ] Is Pop Match implemented (subtract alpha * log(P_target) from logits to suppress popularity bias)?
- [ ] Does co-visitation use Bayesian shrinkage (low-frequency source products shrunk toward global prior)?
- [ ] **⚠️ Bucketing must be based on truncated sequence length, not original sequence length!** Holdout users all have full sequences (only shortened by trunc_mask), if bucketed by original length, bucket 0 (empty sequence) users will never appear in holdout → bucket 0's post-processing parameters are never validated on holdout → their effect on truly empty-sequence users in the test set is completely blind guessing. Correct approach: when bucketing for post-processing, use the truncated sequence length (i.e. `seq_len` after passing through trunc_mask) to determine which bucket the user belongs to. This way, users truncated to empty in holdout can enter bucket 0, allowing bucket 0 parameters to be effectively validated.
- [ ] Do post-processing parameters use different values per sequence-length bucket (rather than one-size-fits-all)?
- [ ] Does co-visitation use multi-position superposition (conditional probabilities for last 1/2/3 products, exponentially decayed weighting)?
- [ ] Does repurchase boost support frequency weighting (high-frequency products get more boost, rather than uniform bias for all products)?
- [ ] Does the cw from co-visitation grid search hit the ceiling? If optimal cw exactly equals the search range upper bound, co-vis is replacing the model rather than supplementing it — need to expand search range to confirm true optimal value, or prioritize improving model quality
- [ ] **⚠️** **Is** **`apply_covisitation`** **only a single loop?** Check the code — is there an outer `for pos in range(n_positions)` + inner `for p in range(n_positions)` nesting? Inner loop aggregates all positions, outer loop adds again = boost amplified ~3x. Correct implementation: single for loop, `boost_vec` is `cw *`-ed only once, `+=`-ed only once.
- [ ] **Is post-processing NDCG improvement within reasonable range?** If no-post-processing → post-processing improvement is too large, must investigate the four items in the anomalous gain checklist above one by one; only accept results after confirming no bugs and no leakage.
- [ ] **⚠️** **Is** **`compute_ndcg_at_k`** **called with full batch rather than sample-by-sample loop?** This is an extremely subtle bug: `compute_ndcg_at_k`'s padding auto-detection relies on `logits.shape[1] == targets.max() + 1`. When called sample-by-sample, each sample's `targets.max()` is just that sample's target value, not equal to num_items → padding column (index 0) is not stripped → participates in top-10 competition → NDCG severely underestimated. **Correct approach**: `compute_ndcg_at_k(logits, targets, k=10)` called once with full batch, letting auto-detection work properly.
- [ ] **⚠️ Does the grid search use full sequences or truncated sequences?** Model inference uses truncation for holdout (simulating test distribution), but co-visitation and repurchase boost must use **full original sequences** — all products in the user's history should participate in co-vis queries and repurchase statistics, not just the last 1-3 products after truncation. Check the code: the `sequences` parameter passed to `grid_search()` or post-processing function, does it come from full sequences directly parsed from `holdout_df`, rather than truncated sequences returned by `compute_logits()`? If empty-truncated users get no co-vis/repurchase signal, the grid-searched parameters are ineffective for real scenarios.
- [ ] Is there specialized enhancement for cold-start users (pop blend, feature conditional probability)?
- [ ] **⚠️ A2.csv format validation (must do before submission!)**: After generating A2.csv, must validate format. ① **Item ID zero-padding**: iid format must be `i` + 6-digit zero-padded number (`i000001` format), not 5-digit. Code check: `f'i{idx:06d}'` not `f'i{idx:05d}'`. ② uid matches official test.csv with no additions, deletions, or reordering. ③ prediction column has 10 iids comma-separated, no duplicate iids. ④ Compare headers and row count with `sample_submission.csv`.


## Round A Lessons Learned

The following conclusions come from the Agent's autonomous multi-round experiments on Round A, validated online (best online NDCG@10=0.5095, local 7-model ensemble 0.5285). **Note: The data scale difference between Rounds A and B for recommendation is huge. The following provides methodological-level experience reference; specific hyperparameters need to be re-searched on the new dataset.**

### ⚠️ Key Differences Between Round B and Round A (Must Read in First Round)

| | Round A | Round B |
|------|------|------|
| Product count | 2156 | **14065** (6.5x!) |
| User count | 50000 | 50000 |
| Training users | 40000 | 40000 |
| Test users | 10000 | 10000 |

**Round B has 6.5x more products than Round A**, which means:
- Model output layer expands from ~2156 to ~14065 classes, significantly slower training and inference
- Single-model training time increases substantially; fewer total experiment rounds can be run within 2 hours
- Product embedding parameter count increases, requiring corresponding adjustment of embed_dim/hidden_dim search ranges
- More severe long-tail: among 14065 products, most are rarely interacted with; loss function choice (CE vs Focal) has larger impact
- Round A hyperparameters may fail on Round B due to scale differences; must re-search

### Core Architecture

- **GRU4Rec + UserFeat + BlendNet** is the validated baseline architecture direction
- Pure sequence models cannot handle cold-start users (empty sequences); user feature models are necessary
- Adaptive fusion weights (BlendNet) outperform fixed-formula fusion, but need verification that they vary with sequence length

### Post-processing Pipeline (Methodology)

Post-processing directions validated effective on Round A (specific parameters need grid search determination on Round B):
1. Pop Match: suppresses popularity bias, significant effect on severe long-tail scenarios (more applicable to Round B's 14065 products)
2. Co-visitation (multi-position): count conditional probability of last N products → target
3. Repurchase Boost (frequency-weighted): weight historical high-frequency products
4. Bayesian shrinkage: shrink co-vis of low-frequency source products toward global prior
5. Bucket-differentiated: bucket by sequence length, independent post-processing parameters per bucket

### Post-processing Parameter Search

- Must use grid search (cw/rb/pm have strong interactions)
- cw hitting ceiling = co-vis replacing model predictions → prioritize improving model quality
- Holdout for grid search must be split before model training and co-vis construction

### Multi-Seed Ensemble

- 3-7 seeds, arithmetic mean of logits
- Must re-grid-search post-processing parameters after adding seeds
- Optimizing individual seed generalization is more effective than simply adding more seeds

### Most Common Bugs

1. Co-visitation nested loop → boost amplification → anomalous post-processing gain
2. Holdout data leakage (build co-vis before splitting holdout) → inflated NDCG
3. Validation set = holdout (`val_ds = ValDataset(holdout_df)`) → early stopping peeks at holdout
4. Truncation inconsistency (holdout evaluation without multi-level truncation)
5. GRU hidden state taken at wrong position (using `seq_len-1` index instead of `[:, -1, :]`)
6. Product features defined but not called in forward
7. Models not sharing item representation (GRU and UF each use their own embedding)
8. pack_padded_sequence (real items skipped under left-padding)
9. Sample-by-sample call to compute_ndcg_at_k (padding detection fails)
10. Bucketing based on original sequence length (empty-sequence post-processing parameters never validated)

### Local→Online Mapping Experience

- Nonlinear mapping relationship between local NDCG and online
- Repeated hyperparameter tuning on validation set → local inflated, online doesn't rise
- No-post-processing NDCG is the model's true level
- Two consecutive rounds with no no-PP NDCG improvement → current architecture has reached bottleneck

