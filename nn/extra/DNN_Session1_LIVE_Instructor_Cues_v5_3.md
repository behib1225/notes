# DNN Session 1 — live instructor cue sheet (corrected)

**For the corrected 72-slide deck:** `DNN_Session1_150min_v5_3_TEACHING_CORRECTED.pptx`. **Use during class;** the full combined guide is for preparation. Approximate timing preserves 150 minutes, with 10-minute break at 1:20. It gives 12 minutes to **slides 30–34**. Appendix slides **55–72 are optional**, not scheduled.

**Delivery principles:** Ask before revealing numerical answers; on slide 30, put givens on the board while showing slide 29. On slide 43, ask predictions on slide 42. Separate mathematical identities from toy-model assumptions and three-seed experimental observations.


## Part 1 — foundations

| Slide · time | Essential words to say | Action / accuracy check |
|---|---|---|
| **1 · 0:00–0:02**<br>Deep neural networks: from one neuron to networks that train | Promise three questions: what networks compute, how they learn, why they generalize. | Point to teal forward / orange backward. |
| **2 · 0:02–0:03**<br>By the end of today you can… | Targets are actions: derive, compute shapes, diagnose, explain. | Ask prerequisite check; do not reteach calculus unless needed. |
| **3 · 0:03–0:05**<br>Today's plan, and one example we will follow all session | Return throughout to Fashion-MNIST 784→128→64→10. | Write architecture visibly; distinguish experiment from theorem. |
| **4 · 0:05**<br>Foundations: what a network computes | Begin with the geometry of a single decision boundary. | Ask: can every dataset be split by one line? |
| **5 · 0:05–0:08**<br>One neuron: a weighted sum and a threshold | A neuron computes z=wᵀx+b and thresholds the score. | Draw the separating line and perpendicular w; assume w≠0 for distance. |
| **6 · 0:08–0:10**<br>Linear models can only draw straight boundaries | Sigmoid changes score interpretation, not a single-neuron boundary. | Contrast 86% and 100% TRAIN accuracy—not generalization. |
| **7 · 0:10–0:13**<br>XOR: no straight line separates the classes | XOR demands contradictory affine inequalities. | Let students try to draw a line before formal proof. |
| **8 · 0:13–0:15**<br>A better representation makes XOR easy | Adding x₁x₂ makes XOR linearly separable in feature space. | Evaluate all four rows of x₁+x₂−2x₁x₂. |
| **9 · 0:15–0:17**<br>The neural-network idea: learn the features too | Networks learn useful features jointly with the classifier. | Draw input → h=g(Wx+b) → class score. |
| **10 · 0:17–0:20**<br>Smooth activations give gradients: the sigmoid neuron | Step is nondifferentiable; sigmoid is smooth, but saturates. | Derive σ′=σ(1−σ)≤¼. |
| **11 · 0:20–0:22**<br>Without a nonlinearity, depth collapses to one layer | Stacked affine layers collapse to one affine mapping. | Multiply two matrices; show how biases combine. |
| **12 · 0:22–0:24**<br>Quick check 1 | Quick check 1: identity hidden layers keep hyperplane boundary. | Vote → discuss → revote; C. |
| **13 · 0:24–0:27**<br>Sigmoid and tanh saturate; ReLU does not for z > 0 | Sigmoid/tanh can saturate; active ReLU has slope one. | Use derivative plots; ReLU can die for z<0. |
| **14 · 0:27–0:30**<br>Two ReLU units solve XOR by remapping the space | Two ReLU units build a representation that separates XOR. | Calculate h₁=ReLU(x₁+x₂), h₂=ReLU(x₁+x₂−1). |
| **15 · 0:30–0:32**<br>A ReLU network computes a patchwork of linear pieces | A ReLU MLP is affine within each fixed activation pattern. | Trace one decision-boundary segment; 146 regions is observed on grid. |
| **16 · 0:32–0:35**<br>Universal approximation, stated honestly | Universal approximation is existence, not learnability or generalization. | Contrast fixed constructed curve and optimized network. |
| **17 · 0:35–0:37**<br>The forward pass repeats one rule per layer | Each layer repeats z=Wa+b, a=g(z). | Check weight shape: (out,in) for column-vector math. |
| **18 · 0:37–0:39**<br>Running example: a 784–128–64–10 MLP | Image batch: (32,1,28,28) → (32,784) → … → (32,10). | Follow batch dimension; labels are class IDs. |
| **19 · 0:39–0:41**<br>In PyTorch terms: a stack of layers that returns logits | PyTorch returns logits; total parameters = 109,386. | Compute 784×128+128 for first layer. |
| **20 · 0:41–0:43**<br>Quick check 2 | Quick check 2: weight storage is (128,784). | Vote; answer B; each output neuron has 784 weights. |

## Part 2 — learning

| Slide · time | Essential words to say | Action / accuracy check |
|---|---|---|
| **21 · 0:43**<br>How a network learns | Switch to loss → gradients → optimization. | Preview the scalar chain-rule calculation. |
| **22 · 0:43–0:46**<br>Most losses are negative log-likelihoods | Noise/target likelihood suggests MSE, BCE, or multiclass CE. | Gaussian fixed σ → MSE; classification losses accept logits. |
| **23 · 0:46–0:48**<br>Softmax, and the first-loss sanity check | Softmax normalizes logits; near-uniform K-class CE is ln K. | Compute ln 10≈2.30; sanity check, not absolute first-loss rule. |
| **24 · 0:48–0:50**<br>Cross-entropy keeps learning when MSE stalls | CE cancels the output sigmoid derivative in logit gradient. | Write CE: σ−y vs half-MSE: (σ−y)σ′. |
| **25 · 0:50–0:52**<br>Quick check 3 | QUICK CHECK 3: model has Softmax then CrossEntropyLoss. | Vote; B remove extra Softmax. The code makes answer determinate. |
| **26 · 0:52–0:55**<br>Pass raw logits to the loss: softmax twice fails silently | Explain double-softmax loss floor ln(K−1+e)−1. | For K=10, ≈1.46. Accuracy alone can hide this bug. |
| **27 · 0:55–0:57**<br>Choose the activation by the layer's role | Activation choice depends on hidden-layer and output role. | Classifier + CE needs raw logits; no softmax in model. |
| **28 · 0:57–1:00**<br>Gradient descent: the sharpest direction caps η | Steepest Hessian eigendirection caps GD step on convex quadratic. | J=½λθ² gives 0<η<2/λ. Not universal for DNN. |
| **29 · 1:00–1:01**<br>Backpropagation: every gradient for ≈ 3× a forward pass | Reverse-mode differentiation shares work across many parameters. | Keep ≈3 forward-pass cost as a rough operational heuristic. |
| **30 · 1:01–1:03**<br>Worked example: forward, then backward, by hand | Before SHOWING: calculate forward pass from parameters on board. | Keep slide 29 showing; then reveal slide 30 with answers: 1.1, 0.7503, 0.3002, 0.5745, 0.0905. |
| **31 · 1:03–1:06**<br>Worked example: backward error signals | Walk backward: loss slope → δ₂ → upstream gradient → δ₁. | Write −0.4255 → −0.1040 → −0.0832 → −0.0156. |
| **32 · 1:06–1:08**<br>Worked example: parameter gradients | Weight gradient = receiving δ × incoming activation. | w₁≈−0.0312; w₂≈−0.0780; bias gradients δ. |
| **33 · 1:08–1:10**<br>Quick check 4 | Quick check 4: w₂=0 blocks first-layer gradient. | Vote; B; cross out the zero-weight backward path. |
| **34 · 1:10–1:13**<br>The four equations of backpropagation | Generalize the scalar trace into δ recurrence and outer product. | Derive Wᵀδ and δaᵀ; check shapes. Softmax+CE gives p−y separately. |
| **35 · 1:13–1:14**<br>In PyTorch, autograd runs these equations for you | Autograd implements chain rule; optimizer uses .grad. | Gradients accumulate; distinguish eval() and no_grad(). |
| **36 · 1:14–1:15**<br>Feeding the network: splits, standardization, mini-batches | Prevent leakage using correct splits and train-only scaling. | Explain dataset shuffling vs validation preprocessing. |
| **37 · 1:15–1:17**<br>The training loop: five steps, every mini-batch | Training loop: clear, forward, loss, backward, update. | Mark where validation, scheduler and logging fit. |
| **38 · 1:17–1:20**<br>Running example: training curves after 20 epochs | Loss and accuracy tell different stories over 20 epochs. | Ask why val CE rises as val accuracy stays near 90%. |
| **39 · 1:20–1:30**<br>Ten-minute break | BREAK: preserve full ten minutes. | Restart at 1:30. |

## Part 3 — train and generalize

| Slide · time | Essential words to say | Action / accuracy check |
|---|---|---|
| **40 · 1:30**<br>Making deep networks train and generalize | Part 3 asks: what makes depth train and generalize? | Recall multiplying Jacobians. |
| **41 · 1:30–1:34**<br>Depth multiplies: gradients vanish or explode | Backprop multiplies layer Jacobians; norms may contract or grow. | A single layer with active ReLUs is not a guarantee. |
| **42 · 1:34–1:39**<br>Initialize to keep the pre-activation variance constant | Derive He scaling for ReLU via preserved second moment. | E[ReLU(Z)²]≈½E[Z²] under symmetric Z; this is not variance equality. |
| **43 · 1:39–1:42**<br>Simulation: 20 ReLU layers, four initializations | Before SHOWING: predict scale behavior on slide 42. | Then show simulation; one seed, measures activations not gradients. |
| **44 · 1:42–1:44**<br>Quick check 5 | Quick check 5: Xavier + deep ReLU tends to shrink forward RMS. | B under idealized equal-width assumptions; caution variance vs RMS. |
| **45 · 1:44–1:49**<br>Residual connections: why deeper can still train | Residual block y=x+F(x) has Jacobian I+J_F. | Direct path aids but does not guarantee nonvanishing gradients. |
| **46 · 1:49–1:53**<br>Ravines defeat plain SGD; momentum and Adam help | On narrow ravines momentum damps oscillation; Adam rescales coords. | Toy losses tuned individually; no universal optimizer ranking. |
| **47 · 1:53–1:55**<br>Quick check 6 | Quick check 6: run became unstable with momentum; isolate η. | A test smaller η; NEVER teach required η/10. Quadratic momentum can permit larger stable η. |
| **48 · 1:55–1:59**<br>Adam = momentum + per-parameter steps; use AdamW | Adam adapts per parameter; AdamW decouples weight decay. | Adam default coupled decay in PyTorch; AdamW decoupled. |
| **49 · 1:59–2:03**<br>Diagnose the gap first, then pick remedies | Diagnose training vs validation curves before remedies. | Check data/pipeline and baseline before architectural complexity. |
| **50 · 2:03–2:07**<br>Weight decay and dropout | Weight decay and dropout change training behavior differently. | Distinguish L2 vs AdamW decay; dropout off in eval. |
| **51 · 2:07–2:10**<br>Data augmentation: free data, if the label survives | Augmentation assumes image transformation preserves labels. | One valid and one invalid transform; train-only stochastic aug. |
| **52 · 2:10–2:14**<br>Running example: 1,000 images overfit | Three-seed observations: CE 0.57→0.53; accuracy +0.4 pp inconclusive. | More data showed +8.4 pp here; no universal or significance claim. |
| **53 · 2:14–2:16**<br>Quick check 7 | Quick check 7: verified splits plus relevant pretrained model. | B: evaluate transfer learning; promising experiment, not guaranteed best. |
| **54 · 2:20–2:30**<br>Today in three sentences | Close: representation, chain-rule learning, reliable generalization. | 109,386 params; near-ln10 initial CE; one-minute muddiest point. |

## Appendix quick-reference (not in core 150 minutes)

| Slides | Use only when needed |
|---|---|
| **55 — Appendix A** | Today’s deeper review / background |
| **56 — Quick-check answers (1–4)** | Answer and reasoning review |
| **57 — Quick-check answers (5–7)** | Answer and reasoning review |
| **58 — Notation used all session** | Today’s deeper review / background |
| **59 — Why deep? Composition multiplies what each layer can do** | Today’s deeper review / background |
| **60 — From 1943 to today: backprop was not the bottleneck** | Today’s deeper review / background |
| **61 — When not to reach for a neural network** | Today’s deeper review / background |
| **62 — Verify hand-written gradients with centred differences** | Today’s deeper review / background |
| **63 — Session 2 preview** | PyTorch Session 2 preview / reference |
| **64 — Batch normalization: standardize, then rescale** | PyTorch Session 2 preview / reference |
| **65 — Normalization variants: which axes, which statistics** | PyTorch Session 2 preview / reference |
| **66 — Double descent: test error can fall again** | PyTorch Session 2 preview / reference |
| **67 — PyTorch loss functions and target formats** | PyTorch Session 2 preview / reference |
| **68 — Six training mistakes that may not raise an exception** | PyTorch Session 2 preview / reference |
| **69 — Step each scheduler at the right frequency** | PyTorch Session 2 preview / reference |
| **70 — A complete training recipe (illustrative settings)** | PyTorch Session 2 preview / reference |
| **71 — References and further reading (1/2)** | PyTorch Session 2 preview / reference |
| **72 — References and further reading (2/2)** | PyTorch Session 2 preview / reference |

## Instructor-only checks before teaching

1. **Momentum math:** For SPD quadratic and heavy-ball update `v_t=βv_(t−1)+∇J(θ_t); θ_(t+1)=θ_t−ηv_t`, stability needs `0<η<2(1+β)/λ_max` with `0≤β<1`. Constant-gradient velocity amplification `1/(1−β)` does not imply a mandated learning-rate reduction.
2. **PyTorch loss interfaces:** `CrossEntropyLoss` expects raw logits; class-index labels are usually `(B,)` `int64`. `BCEWithLogitsLoss` also expects logits.
3. **He initialization:** ReLU propagation arguments usually control `E[a²]` (RMS), not exact `Var(a)` when activation has nonzero mean.
4. **Experimental caution:** In the supplied results JSON, the three-seed mean best validation loss is about `0.5722` without and `0.5343` with combined regularization; validation accuracy means are about `80.66%` and `81.09%`. These are reported runs, not significance estimates.
5. **Timing:** If questions take too long, shorten the review of cost heuristics and optimizer analogies—not the hand calculation or mini-batch shape checks.

**API sources:** https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html · https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html · https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

**Verification scope:** The deck was rendered to verify edited slides. The experiment JSON was inspected for the small-data summary; full model-training jobs were not rerun.
