# Graph Node Classification Methods Reference

## Data Characteristics

- Graph structure is stored in CSR sparse format (adjacency matrix + node feature matrix)
- Training nodes ~80%, test nodes ~20%
- Labels with -1 indicate test nodes (not real categories); exclude these when computing num_classes
- Node features may be sparse and need to be converted to dense matrices for model use

## Basic Architecture Code Writing Guide (Core!)

Phase 1 horizontal comparison requires writing training scripts for multiple architectures (GCN/GAT/GraphSAGE/GIN/APPNP/SGC). The correct approach is to use **a single universal training framework + replacing only the model definition part for each architecture**, rather than writing a training loop from scratch for each architecture. Below are the conventions and common pitfalls to follow when writing this framework.

### 1. Model Class Design Conventions

All GNN model classes should follow a uniform interface convention so they can share the same training and inference logic:

**1.1 Constructor Parameters**

Each model class's `__init__` signature should ideally be unified as `(in_dim, hidden_dim, out_dim, num_layers, dropout, activation)` or add architecture-specific parameters on top of that. This way the training loop can pass parameters uniformly.

**1.2 General Rules for Layer Stacking**

Use `nn.ModuleList` to manage multiple convolutional layers, building them according to these rules:

- Single layer: Conv maps directly from in_dim to out_dim
- Multiple layers: First layer in_dim → hidden_dim, middle layers hidden_dim → hidden_dim, last layer hidden_dim → out_dim

**1.3 Inter-layer Processing**

After each convolutional layer (except the last), apply in order: optional normalization (e.g. LayerNorm) → activation function → Dropout. The last layer should not have activation or Dropout — output directly.

**1.4 Forward Output**

The final output should uniformly use log_softmax (not softmax or raw logits), so it can directly work with NLL loss during training.

**1.5 Architecture-Specific Key Points**

- **GCN**: Can add LayerNorm to improve training stability. Correctly pass the edge_weight parameter in forward.
- **GAT**: When using multi-head attention in intermediate layers, the output dimension must be multiplied by the number of heads; the last layer needs heads=1 to compress the dimension back to out_dim. Note that the input dimension of the first layer is in_dim (not multiplied by heads).
- **GraphSAGE**: Full-graph training is fine when the graph is not large; no need to sample neighbors; use mean aggregation.
- **GIN**: The "aggregation function" of GINConv is an MLP (not a single-layer Linear). You need to build a small two-layer or three-layer network with activations using Sequential as the convolution kernel.
- **APPNP**: The structure differs from stacking Conv layers — only two Linear layers are needed for feature transformation, followed by a propagation layer for PageRank-style diffusion. The propagation steps K and restart probability alpha are unique hyperparameters of this architecture.
- **SGC**: The simplest architecture; a single SGConv layer completes feature transformation + propagation. No need to stack multiple layers.

### 2. Key Points for Writing the Training Loop

The following lists the key considerations for each step in the training flow in chronological order — no pseudocode templates are provided; you need to independently design the specific implementation.

**2.1 Data Preparation**

When converting the CSR-format adjacency matrix to the edge_index format usable by PyG, use `torch_geometric.utils.from_scipy_sparse_matrix`. Note that the edge_weight returned by this function may be None (depending on whether the CSR matrix has explicit weight storage); you must check for None before calling `.to(device)`.

**2.2 Train/Validation Split**

You must use the `split_train_val` function from `workspace/code/cls/eval_utils.py`; do not write your own train_test_split. This function guarantees stratified splitting (stratify by label) and a fixed random seed, ensuring that validation scores across experiments are comparable.

**2.3 Fix Random Seeds**

Fix all random seeds (`torch.manual_seed` + `torch.cuda.manual_seed_all`) before creating the model. Otherwise, running the same hyperparameters twice will produce different results, making it impossible to determine whether the parameters are truly effective.

**2.4 Operation Order for Each Epoch**

The following lists the key steps and error-prone details for each epoch in chronological order:

**(a) DropEdge (optional)**: Randomly drop a portion of edges before entering training mode. This should only affect the training forward pass; validation and inference must not use the edge-dropped graph.

**(b) Training Forward Pass**: Switch to train mode, clear gradients, perform a full-graph forward pass using the (possibly edge-dropped) edge_index. Take the output corresponding to train_sub_idx nodes as log_probs.

**(c) Loss Calculation**: Since the model outputs log_softmax, use negative log-likelihood directly for the loss. If using label smoothing, you need to manually construct the smoothed target distribution — distribute a portion of probability mass evenly to all non-target classes. Note that when distributing the smoothing value to other classes, the denominator is num_classes - 1 (not num_classes).

**(d) Backpropagation**: After computing the loss, first do gradient clipping (`clip_grad_norm_`) to prevent gradient explosion common in GNN training, then execute optimizer.step() and scheduler.step(). If using CosineAnnealingLR, step once per batch (not once per epoch).

**(e) SWA (Stochastic Weight Averaging)**: In the later stages of training, perform a moving average of model parameters, maintaining a separate copy of smoothed weights independent of the current training. The decay factor should decrease with the number of accumulations (new model weight = 1/n, old SWA weight = (n-1)/n), rather than using a fixed decay rate.

**(f) Validation**: Switch to eval mode, perform a full-graph forward pass using the **original full edges** (not edge-dropped). Take the output at val_sub_idx to compute accuracy. Note that Dropout does not take effect in eval mode, so no manual toggling is needed.

**(g) Early Stopping and Checkpoint**: If validation accuracy exceeds the historical best, save the current model parameters to CPU (using `.cpu().clone()`, not storing a GPU reference), and reset the patience counter to zero; otherwise increment the counter. You can give the model a few warm-up epochs at the start of training before beginning the count.

**2.5 Model Selection After Training**

After training completes, validate once more using the SWA-smoothed weights: if the SWA result is better than the best checkpoint saved during training, use the SWA weights; otherwise load the best checkpoint.

**2.6 Early Training Collapse Detection**

If the validation accuracy in the first few epochs is close to random guessing level (approximately 1/N for an N-class task), it means the model has learned nothing at all. This usually indicates a fundamental error in the architecture configuration (such as dimension mismatch, features not correctly passed in). You should terminate and troubleshoot early rather than waiting hundreds of rounds.

**2.7 Optimizer and Scheduler Selection**

- Use AdamW (not Adam) as the optimizer; it decouples weight decay from the learning rate, making GNN training more stable
- Use CosineAnnealingLR for learning rate scheduling; T_max can be set slightly larger than the actual number of epochs, and eta_min should be set to a fraction of the initial learning rate

### 3. Key Points for Writing Inference

**3.1 Predicting Class Labels**

Perform a single full-graph forward pass and take the argmax of the output corresponding to test_idx. Note that you should use the original full edge_index and must be within a `torch.no_grad()` context.

**3.2 Predicting Probabilities (for Ensembling)**

Multi-seed ensembling requires softmax probability weighted averaging. If the model forward outputs log_softmax (with negative values), you need to use exp to convert them back to probabilities before ensembling, then weight by each model's weight and sum, and finally argmax to get the label. Directly using log_softmax values for weighted averaging will produce completely wrong results.

**3.3 Full-Graph Probabilities (for Label Propagation)**

Post-processing methods like label propagation require probability matrices for all nodes (not just test_idx). In this case, do not argmax the logits; instead take softmax over all nodes.

### 4. Organizing the Horizontal Comparison Script

Phase 1 is recommended to use a single script to complete the comparison of all architectures:

- Define a universal training function that accepts a model and config and returns a validation score
- Write all six architecture model classes in the same script
- Loop through the architecture list, calling the same training function for each architecture

This ensures that all architectures use exactly the same training logic (optimizer, scheduler, early stopping, SWA, data split), with differences coming only from the model architecture itself, making the comparison fair.

### 5. Most Common Bugs (Self-Check Item by Item After Writing)

1. **edge_weight not checked for None**: After CSR-to-edge_index conversion, edge_weight may be None; calling `.to(device)` directly will cause an AttributeError
2. **Validation used the edge-dropped graph**: Validation must use the original full edge_index; DropEdge only applies during training
3. **Confusing log_softmax and softmax**: Use log_softmax output directly for loss calculation; ensembling and label propagation require converting to probabilities first (exp or softmax); the two values are completely non-interchangeable
4. **Wrong label smoothing denominator**: When distributing the smoothing value to other classes, divide by `num_classes - 1`, not `num_classes`
5. **Fixed seed set after build_model**: Model parameter initialization is determined when `__init__` executes; the seed must be set before creating the model
6. **Forgot model.eval() before validation**: Dropout layers behave differently in train and eval modes; forgetting to switch to eval will cause validation scores to be low and unstable
7. **Checkpoint stored a GPU reference**: When saving the best model, parameters must first be `.cpu().clone()`, otherwise the checkpoint will change when subsequent training updates parameters on GPU
8. **num_classes computed incorrectly**: Must take unique values on `labels[labels != -1]`; -1 is a placeholder label for test nodes, not a real category
9. **Very low val_acc in early epochs not investigated**: If val_acc in the first few training epochs is close to 1/number_of_classes (random level), it means the model has a fundamental error; running hundreds more rounds won't help
10. **Directly using log_softmax for weighted ensembling**: log_softmax values are negative and not a probability distribution; must use exp to restore them before weighting
11. **Mixing boolean indexing and fancy indexing causing label misalignment (Critical! Subtle!)**: This is one of the most insidious bugs — all scores are universally low (e.g. accuracy < 0.40 for a 10-class task), but the code doesn't error. Root cause: sklearn `train_test_split` shuffles the index order. If you use the shuffled indices returned by split for both: ① `torch.zeros(N)[idx]=True; log_probs[mask]` (boolean mask indexing, returns in ascending node ID order) and ② `labels[idx]` (fancy indexing, returns in the original order of idx), the two orders are inconsistent — predictions and labels are randomly shuffled in pairing, and the model learns nonsense. Symptoms: all architecture scores are very low, rankings are meaningless (GIN ranking first is pure noise), running more epochs doesn't help. Fix: immediately `np.sort()` the split results, or uniformly use integer indexing (no boolean masks). **Self-check method**: print the first 5 train_sub_idx and the node IDs selected by the corresponding mask, and visually confirm the order is consistent

### 6. Training Script Completeness Checklist (Check Off Each Item After Writing Code — None Can Be Omitted)

The following items are all scoring techniques verified by hand-written predictors — missing any one will cause significantly lower scores (cumulative loss can exceed 0.02). After writing the training script, confirm each item:

**□ SWA (Stochastic Weight Averaging)**

- Start performing moving average of model parameters in the second half of training (e.g. after the first half of epochs)
- Decay method: on the n-th accumulation, new weight = 1/n of current model, old SWA weight = (n-1)/n
- After all training is done, validate once more with SWA weights, take the better of (SWA, best_checkpoint) as the final model
- Missing SWA directly loses approximately 0.005-0.01

**□ Label Smoothing**

- Training loss cannot be bare NLL loss; must include label smoothing
- Distribute a small portion of probability mass from the target class evenly to all non-target classes (denominator is num_classes - 1)
- Effective for all architectures; GAT especially depends on this
- Missing label smoothing directly loses approximately 0.003-0.005

**□ DropEdge (Must Enable for GCN and GAT)**

- Randomly drop a portion of edges before training each epoch; use full edges for validation and inference
- GAT is especially sensitive to this — GAT without DropEdge severely overfits, and its score may be lower than GCN
- APPNP doesn't need it (the propagation layer has built-in smoothing); SGC doesn't need it
- Missing DropEdge causes GAT to directly lose approximately 0.005-0.01

**□ Sufficient Training Epochs**

- GNNs typically need many epochs to fully converge on this dataset
- The number of epochs should not be too few; recommended reference value: far greater than a multiple of the early stopping patience
- When used with early stopping, training will automatically stop after convergence; setting a larger epoch count won't waste much time

**□ Learning Rate in a Reasonable Range**

- Learning rate should not be too high; GNNs are sensitive to this
- GAT's learning rate should be lower than other architectures (multi-head attention is more sensitive to high lr)
- If val_acc oscillates wildly or fails to converge, first suspect that lr is too high

**□ CosineAnnealingLR Configured Correctly**

- T_max should be set larger than the actual number of epochs (so the entire training process only experiences a portion of the cosine curve's descending segment), not equal to the number of epochs
- eta_min should be set as a proportion of the initial lr (e.g. 10%), not hardcoded as an absolute value

**□ Gradient Clipping**

- After each backward pass and before optimizer.step, perform gradient clipping on parameters
- Gradient explosion is a common issue in GNN training; not clipping can cause training instability

**□ Early Stopping Has a Warm-Up Period**

- Do not trigger early stopping for the first several epochs, giving the model sufficient time to leave the random initialization region
- Start counting patience only after the warm-up period ends

**□ Adjacency Matrix Preprocessing**

- Convert directed graph to undirected graph (A + A^T, binarize)
- Add self-loops (A + I)
- These two steps directly affect the message-passing quality of GCN/GAT/APPNP

**□ Feature Standardization**

- Apply z-score standardization to input features (subtract mean, divide by standard deviation)
- If concatenating additional graph structure features, standardize the original features and graph structure features separately before concatenation

**□ Boolean Index and Fancy Index Consistency (Fatal! Must Check Before Training)**

- Check whether the code has both `mask[idx]=True; output[mask]` and `labels[idx]` occurring simultaneously
- If idx comes from sklearn's `train_test_split`, it shuffles the order by default — boolean masks return in ascending node ID order, fancy indexing returns in shuffled order; the two are misaligned
- Fix: immediately `np.sort()` the split results, or uniformly use one type of indexing
- Verification method: print `train_sub_idx[:5]` and the first 5 labels you actually feed to the loss before training, and visually confirm the order is consistent

## Optional Methods

### GCN (Graph Convolutional Network)

- Core idea: Each layer performs mean aggregation of neighbor node features, followed by linear transformation + activation
- Layers: Typically 2-3 layers; too deep causes over-smoothing (all node representations converge)
- Key hyperparameters: hidden_dim (hidden dimension), dropout (anti-overfitting), lr, weight_decay
- Training techniques: DropEdge (random edge dropping), SWA, label smoothing, gradient clipping

### GAT (Graph Attention Network)

- Core idea: Introduces attention mechanism on top of GCN; neighbor nodes have different aggregation weights
- Multi-head attention: Multiple attention heads in parallel; intermediate layers concatenate (dim × heads), last layer outputs single head
- Key hyperparameters: heads (number of attention heads), hidden_dim
- Training techniques: DropEdge + Label Smoothing are especially effective for this architecture

### GraphSAGE

- Core idea: Neighbor sampling + aggregation (mean/max/LSTM pool), suitable for large graphs
- Supports inductive learning: can generalize to unseen nodes
- No need for sampling in full-graph training; use mean aggregation

### GIN (Graph Isomorphism Network)

- Core idea: Learns aggregation function through MLP; strong expressive power; theoretically can distinguish different graph structures
- Key implementation: GINConv internally uses an MLP (multi-layer Linear + activation), not a single-layer Linear
- Can use the eps parameter to control the ratio of self-features to neighbor features

### APPNP (Approximate Personalized Propagation)

- Core idea: Decouples feature transformation from information propagation. First uses MLP for feature transformation, then uses PageRank-style propagation
- Structure differs from stacked Conv: Two Linear layers + one propagation layer (APPNPLayer)
- Key hyperparameters: K (propagation steps), alpha (restart probability, controls the weight of distant information)

### SGC (Simplified Graph Convolution)

- Core idea: Removes GCN's non-linear activation, keeping only linear transformation + propagation
- Few parameters, fast training, suitable as a quick baseline; implemented with a single SGConv layer

## Training Strategy Reference

- **Optimizer**: AdamW (not Adam) — decoupled weight decay is generally better
- **Learning Rate Schedule**: CosineAnnealingLR, schedule once per step
- **Gradient Clipping**: `clip_grad_norm_(max_norm=1.0)`; GNN training is prone to gradient explosion
- **Early Stopping**: Stop when validation set no longer improves
- **Validation Set Split**: **Must use the `split_train_val` function from `workspace/code/cls/eval_utils.py`**. This function implements stratified splitting (stratify by label) + fixed seed=42 + 20% validation ratio. The validation ratio must not be reduced — too few validation nodes leads to noisy scores, making it easy to overfit to the validation set
- **Evaluation Metric**: Accuracy (classification accuracy), using `eval_utils.compute_accuracy()`
- **Multi-Seed Evaluation**: Use `eval_utils.evaluate_seeds(val_accs)` to get mean ± standard deviation; **using the mean of top-k seeds is forbidden** (that's cherry-picking, overfitting to the validation set)
- **SWA (Stochastic Weight Averaging)**: Perform moving average of model parameters in the later stages of training; validate once with the SWA model before finishing; take the better of (SWA, best_checkpoint)
- **Label Smoothing**: Manually construct smoothed targets (the model outputs log_softmax, use NLL loss directly); denominator is num_classes-1

## Recommended Methods

### Recommended Experiment Flow

Proceed in the following four-phase order:

**Phase 1: Architecture Screening**

- Using **the same fixed seed**, train multiple different base architectures sequentially
- **Must cover all six architectures: GCN, GAT, GraphSAGE, APPNP, SGC, GIN**, running at least one configuration for each. Skipping any architecture is not allowed
- **Iron rule**: If a certain architecture throws a bug/error, you must fix the bug and rerun. **It is absolutely forbidden to skip an architecture on the grounds of "instability" or "poor performance."** Errors are code problems, not architecture problems
- **Code organization**: Write a single script containing all model classes + one universal training function, looping through architectures and calling the same training logic. Do not write separate training scripts for each architecture
- **Architecture-Specific Default Configuration Iron Rule (Extremely Important!)**: Each architecture needs different training techniques to demonstrate its true capabilities under fair conditions. **You must not give all architectures the same set of default hyperparameters** — that would test "which architecture can withstand a crude configuration" rather than "which architecture is stronger." You must specify the required default training techniques for each architecture:
  - **GCN**: Needs DropEdge regularization; needs SWA for improved generalization; moderate use of label smoothing
  - **GAT**: Extremely sensitive to overfitting; **must enable DropEdge** (this is GAT's core regularization technique; without it, severe overfitting occurs); **must use Label Smoothing**; give a reasonable initial value for the number of attention heads; needs SWA; learning rate should be lower than other architectures (multi-head attention is more sensitive to high lr)
  - **GraphSAGE**: Relatively robust; basic configuration + SWA is sufficient
  - **GIN**: MLP convolution kernel needs a reasonably designed number of layers; needs SWA + label smoothing
  - **APPNP**: Structure differs from stacked Conv; propagation steps K and restart probability alpha must be given reasonable initial values; does not need DropEdge (propagation layer has built-in smoothing effect); basic lr is sufficient
  - **SGC**: Simplest architecture; does not need DropEdge/SWA/label smoothing; basic configuration, run through quickly
    Key principle: **The goal of Phase 1 is to compare the expressive power of the architectures themselves; you must give each architecture the default configuration it truly needs, so they compete under fair conditions.** One-size-fits-all parameters + missing key training techniques = unfair ranking = risk of missing the optimal architecture. If a certain architecture's ranking contradicts your prior knowledge (e.g. GAT is significantly behind APPNP), first check whether it was given the training techniques it needs, rather than directly concluding that the architecture is weak.
- Select the **top 3 architectures by validation set performance** to proceed to the next phase. Phase 1's default configurations cannot achieve full fairness (hyperparameters like lr significantly affect rankings); taking only one is too risky — the true optimal architecture might be suppressed by a high default lr. Taking the top 3 into Phase 2, fine-tuning each separately and then comparing horizontally, will determine the truly optimal architecture.

**Phase 2: Single-Seed Hyperparameter Tuning (Fine-Tune the Top 3 Architectures Separately)**

- Lock in the **top 3 architectures** selected from Phase 1, and search key hyperparameters for each architecture separately on **the same fixed seed**
- **Grid Search Iron Rule (Extremely Important!)**: Hyperparameter tuning **must use exhaustive grid search**; each hyperparameter needs to systematically cover its reasonable value space. **It is absolutely forbidden** to try only 2-3 configurations and think that's enough, or to stop when you find a "decent" score. Phase 2 is the highest-ROI phase in the entire workflow; cutting corners on search here directly loses leaderboard score.

  Correct grid search approach:
  - **Each hyperparameter should cover at least 3-4 different values**, forming a complete search grid. For example, lr should search multiple orders of magnitude from low to high; hidden_dim should cover multiple scales from small to large; dropout should be evenly distributed from low to high
  - **The search space must be wide enough** — covering from "conservative" to "aggressive" (e.g. lr's low values ensure convergence, high values test the upper limit; hidden_dim's small values test for overfitting, large values test capacity limits)
  - **Combinatorial coverage**: Do not tune each parameter independently (e.g. "tune lr first, then tune hidden_dim with lr fixed"); do Cartesian product-style full combination search. The "optimal" found by independently tuning one parameter is often not optimal when combined with other parameters
  - **At least 30 configurations per architecture**, depending on the product of search dimensions. If you stop after searching only a few configurations, you're cutting corners — go back and complete the grid
  - **Horizontal comparison only after searching all configurations**; it is not allowed to terminate early upon finding a configuration that "looks good" mid-search
- Focus on searching: hidden_dim, dropout, lr, weight_decay, number of layers, etc.
- **Learning rate is the most important search dimension in Phase 2** — Phase 1's default lr is typically too high; tuning should start searching from lower initial values
- For GAT, pay extra attention to: number of attention heads, DropEdge ratio, label smoothing value; lr should be searched lower
- For APPNP, pay extra attention to: propagation steps K, restart probability alpha
- For GCN, pay extra attention to: DropEdge ratio, whether to use LayerNorm; number of layers should not exceed 2
- Simultaneously determine the optimal adjacency matrix processing method (symmetric normalization/row normalization, making undirected, self-loops) and graph structure feature combinations
- After each architecture obtains its optimal single-model configuration, compare the optimal single models of the 3 architectures horizontally and select the **one with the highest validation score** to proceed to Phase 3

**Phase 3: Multi-Seed Ensembling**

- Lock in the optimal configuration selected from Phase 2 (architecture + hyperparameters + features + adjacency processing), **changing only the random seed**
- Train multiple seeds (recommended no more than 10); the validation score of each seed should not deviate too far from the mean
- **All trained seeds participate in ensembling with equal weights** — do NOT filter seeds by validation score (this is cherry-picking, overfitting to validation set). The only valid reason to exclude a seed is if it failed to train (val_acc near random level, indicating a bug). If a seed trained successfully, it participates.
- **Ensembling method**: Take the softmax probabilities of each model and do simple averaging (equal weights, not weighted by val score), then argmax to get the label
- This is the phase with the highest scoring gain

**Phase 4: Ensemble Post-Processing**

- Perform post-processing (e.g. label propagation/smoothing) on the logits of the multi-seed ensemble
- Requires the full-graph probability matrix (predict_full_proba), not just the test_idx output
- Gains are typically small, serving as the icing on the cake; should not take too much time

### Prediction Distribution Calibration (Important!)

Even with high val_acc, GCN's prediction distribution often heavily favors majority classes (very few predictions for classes 0 and 7). Bias calibration is needed to fix the distribution.

**Correct Approach**:
1. Calibrate on **the ensembled logits**, not seed by seed
2. **Search bias on the training set** — the training set has 5000+ nodes with at least 250+ samples per class, far more reliable than the 900-node validation set
3. Optimization objective: After adding bias to training set logits, check that the prediction distribution satisfies each class >= 3%; do not check accuracy on training set (training set labels are known; the model may overfit there). **However, you MUST verify on the validation set that bias did not cause accuracy to drop significantly — if val accuracy drops >0.02 after bias, the bias is too aggressive and will destroy test accuracy.**
4. **Bias magnitude safety limit**: Bias should only nudge ambiguous cases, not override confident predictions. After applying bias: (a) count how many predictions changed compared to no-bias, if >15% changed → bias is too strong, reduce it; (b) bias values should stay within [-0.5, +0.8] — values like +1.5 or -1.0 are almost certainly destructive.
5. **Distribution target is training distribution, NOT uniform**: Classes C1 and C6 may naturally be >20% because they account for >20% of training data. Predictions at similar proportions are EXPECTED and CORRECT. The goal is to prevent minority classes from being entirely suppressed (0%), not to force majority classes below their natural frequency.

**Forbidden Approaches**:
- Do not calibrate seed by seed to a uniform distribution (1/8) — this destroys the model's true prediction signal; val appears falsely high but the leaderboard score crashes
- Do not tune bias on the validation set (900 nodes) — minority classes have only 40 samples; bias is unreliable
- Do not unconstrainedly pursue uniform class distribution — some classes account for only 4-5% in training; forcing them to 12.5% means misclassifying majority class samples into minority classes

### Seed Diversity Ensembling

More effective than single-model fine-tuning:

- **Multi-seed ensembling** is the primary means of score improvement; even the most fine-tuned single seed cannot match the balanced ensemble of multiple medium-quality seeds
- **Seed count should not be too high**: Too many seeds introduce low-quality models that drag down the mean; marginal returns diminish
- **All seeds participate equally**: Do NOT filter by validation score — every successfully trained seed (val_acc meaningfully above random) gets equal weight in the ensemble. Filtering seeds by val score = cherry-picking = overfitting to validation set = online score drops
- **Balanced mixing**: Use simple equal-weight averaging (not weighted by val score), avoiding a single seed dominating the result
- **Single-seed fine-tuning ceiling**: Repeatedly fine-tuning on one seed yields diminishing returns; it's better to invest computation in training multiple high-quality seeds

### Graph Structure Feature Enhancement

Hand-crafted graph structure features provide clear benefits to the model:

- **Node degree and its log transform**: Reflects connection density
- **PageRank**: Reflects global importance
- **Multi-hop neighbor degree** (2-hop and beyond): Reflects extended neighborhood information
- **K-core decomposition value**: Reflects the node's core/periphery position in the graph
- **Feature norm and feature density**: Reflect the strength of the node's own attributes

These features are model-independent and can be concatenated to the original features. Feature preprocessing methods have a significant impact on effectiveness.

### Adjacency Matrix Processing

- **Symmetric normalization** (D^(-1/2) A D^(-1/2)) is generally better than row normalization
- **Make undirected**: Convert directed graph to undirected graph (A + A^T), retaining binary edge weights
- **Add self-loops**: Add the identity matrix to the adjacency matrix, preserving the node's own information
- **Propagation parameters**: Information propagation steps and restart probability need to be tuned for the dataset; different configurations may be complementary

### Cross-Configuration Fusion

Models trained with different configurations (e.g. different propagation parameters, different normalization methods) may be complementary when ensembled:

- Two configurations that perform moderately individually may, when fused, surpass their respective optimal results
- Complementarity comes from different configurations capturing different aspects of graph structure information
- If the difference between configurations is too large, fusion gains decrease; look for configuration pairs that are "similar but different"

### Ensemble Post-Processing

Further post-processing (e.g. label propagation/smoothing) can be done on the ensemble logits, requiring the full-graph probability matrix. However, gains are typically small, serving as icing on the cake rather than the primary means of score improvement.

## Common Pitfalls

- **One-size-fits-all default configuration causing distorted rankings (Frequent mistake!)**: Giving all architectures the same set of hyperparameters in Phase 1 easily produces incorrect rankings. Symptom: GAT is significantly behind APPNP, contrary to intuition. Root cause: GAT without DropEdge + Label Smoothing is competing with one hand tied; APPNP is naturally robust and doesn't need these. Solution: Each architecture uses its own dedicated default configuration — GAT must enable DropEdge + Label Smoothing + lower lr; GCN must enable DropEdge + SWA; APPNP should be given reasonable K/alpha initial values. When rankings are anomalous, first check whether the configuration is unfair; don't directly conclude the architecture is weak
- **Label -1 handling**: Exclude -1 when computing num_classes (`labels[labels != -1]`); mask out -1 nodes during training; labels in test_idx are -1, not real categories
- **Sparse feature conversion**: CSR → toarray() may cause memory explosion; pay attention to feature dimension × number of nodes
- **Full-graph training vs. batch training**: GCN-type models typically do a single full-graph forward pass; batch processing requires attention to adjacency matrix slicing
- **Validation set overfitting (Most common error!)**: The validation set has only about 2,200 nodes; repeatedly selecting seeds and comparing configurations on it will cause val to "improve" continuously while the leaderboard score drops instead of rising. Symptom: val rises but leaderboard drops. Solution: ① Use the mean of all seeds to compare configurations; do not pick top-k ② Treat very small val improvements as noise ③ If val is rising but the leaderboard is dropping, immediately stop fine-tuning and return to a simpler configuration
- **edge_weight is empty**: When converting CSR matrix to edge_index, edge_weight may be None; must check for None before `.to(device)`, otherwise AttributeError
- **Training and validation used different edges**: DropEdge only takes effect during training; validation and inference must use the original full edges
- **Wrong probability used in ensembling**: The model outputs log_softmax (negative values); when ensembling, you must first use exp to restore to probabilities before weighted averaging; directly using log_softmax values for weighting produces completely wrong results

## Lessons Learned from Leaderboard A

The following conclusions come from the Agent's autonomous 18-round experiments on Leaderboard A (~130 models), verified on the leaderboard. **Note: Leaderboard A and B have different data distributions (different numbers of nodes, feature dimensions, and classes). The following serves as methodological reference only; specific hyperparameters must be re-searched and determined on the new dataset and must not be directly copied.**

### Key Differences Between Leaderboard B and A (Must Read for First Round)

**Do NOT rely on hardcoded numbers below — run data exploration yourself in round 1 to get actual values.** The A榜 numbers are provided for scale comparison only:

| <br /> | Leaderboard A (reference) | Leaderboard B |
| ------ | ------------- | ------------- |
| Number of Nodes | 13752 | **Discover via explore_data.py** |
| Feature Dimension | 767 | **Discover via explore_data.py** |
| Number of Classes | 10 | **Discover via explore_data.py** |
| Training Nodes | 11001 | **Discover via explore_data.py** |
| Test Nodes | 2751 | **Discover via explore_data.py** |

**Leaderboard B has different data scale from A. All hyperparameters (hidden_dim/lr/dropout, etc.) must be re-searched on B榜 — never copy A榜 values.**

### Adjacency Matrix Processing

**Must strictly follow the data README requirements**: "Please use the adjacency relationships saved in the file as-is, without additionally assuming they are necessarily undirected, symmetric, or have no self-loops." First use a script to check the graph's directedness, symmetry, and self-loop status, then decide the preprocessing method based on the actual structure.

**Recommended to convert the adjacency matrix to an undirected graph (A + A^T, binarized) before training**. Use PyG GCNConv default settings (add_self_loops=True, symmetric normalization); do not manually implement convolution layers. Making undirected is a standard GNN preprocessing step — first analyze and confirm the graph structure, then do preprocessing to adapt to the architecture, which does not conflict with "using adjacency relationships as-is."

### Architecture Rankings (Leaderboard A Reference)

- On Leaderboard A, GAT significantly led GCN/GraphSAGE/APPNP/SGC/GIN under fair configuration
- **Leaderboard B may have different architecture rankings — always trust your own Phase 1 results, not A榜 history.** Different graph sizes, feature dimensions, and class distributions can change which architecture works best. If your Phase 1 finds GCN > GAT, that is valid data — do not waste time second-guessing it
- Phase 1 should still follow the experiment specifications and comprehensively compare all architectures; do not skip any

### Hyperparameter Search Experience (Methodology, Not Specific Values)

Technical directions that were significantly effective for GAT on Leaderboard A (specific values need to be re-searched on Leaderboard B):

- **DropEdge**: GAT is extremely sensitive to the edge-drop ratio; safe values differ across datasets; must grid search
- **SWA**: Enable Stochastic Weight Averaging in the second half of training; consistently improves score
- **Label Smoothing**: GAT depends on this; the smoothing coefficient needs to be searched
- **Activation Function**: ReLU performed well on Leaderboard A; other activation functions can be used as comparison items
- **Learning Rate**: GAT is sensitive to high lr; recommended to start searching from conservative values

### Seed Diversity Ensembling

- Single seeds have a ceiling; balanced multi-seed ensembling is the key means to break through the upper limit
- **All successfully trained seeds participate with equal weights** — do NOT filter by validation score (this overfits to validation set). Only exclude seeds that clearly failed to train (val_acc near random)
- Ensembling method: softmax probability simple averaging (equal weights, not weighted by val score)
- More seeds is not always better; balance diversity and quality

### Methods Verified Ineffective on Leaderboard A

The following methods were experimentally verified on Leaderboard A to provide no scoring benefit or have unstable effects; they can be deprioritized on Leaderboard B:

- Pseudo-labels (high confidence ≠ correct; low-quality labels pollute training)
- Temperature scaling (no effect)
- Repeated fine-tuning of a single seed (extremely low ROI)
- KNN pseudo-edges, SVD/PCA dimensionality reduction

### Isolated Nodes

Feature propagation (multi-round smoothing of original features) benefited both isolated and connected nodes on Leaderboard A. However, the specific number of propagation steps and implementation method need to be verified on Leaderboard B data.

### Common Failure Points

- Inappropriate DropEdge ratio → model collapse (val near random level); must grid search safe values for the current dataset
- Mixing boolean indexing and fancy indexing → label misalignment; all model scores < 1/number_of_classes; no error but everything is ruined
- Directly using log_softmax values for weighted ensembling → completely wrong; must first use exp to convert to probabilities
- Single seed proportion too high (>40%) → ensemble is dominated by a single model; diversity advantage is lost
- Frequently switching architectures → unable to converge; should complete the four-phase workflow before switching direction
