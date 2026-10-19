# Instructor teaching guide — revised and expanded

**Companion deck:** `DNN_Session1_150min_v5_3_TEACHING_CORRECTED.pptx` — 72 slides: core class slides **1–54**, review appendix **55–62**, Session 2 preview **63–70**, references **71–72**.  
**Reference date:** **18 October 2026**. Slide titles below are taken from the actual deck, not guessed from individual answer boxes or code blocks.  
**Scope:** This is an **instructor companion**, not a replacement for the deck’s built-in timing/speaker notes. It adds teachable derivations, board prompts, pitfalls, and question-and-answer preparation.

## How to use in a 150-minute class

The corrected PowerPoint and speaking script use the following 150-minute allocation (backprop slides 30–34 receive 12 minutes): **0:00–0:43 foundations; 0:43–1:20 learning; 1:20–1:30 break; 1:30–2:20 train and generalize; 2:20–2:30 wrap-up/buffer**. The additional derivations below are a *bank*: teach the most relevant one per difficult slide, and reserve the remaining Q&A for discussion and office hours. To stay within time, use the worked XOR, MSE-vs-CE, scalar-backprop, He initialization, and AdamW/diagnosis demonstrations as anchors rather than presenting every optional question.

Keep the four provenance categories separate: **EXPERIMENT** (trained model with stated seed(s)); **SIMULATION** (numeric simulation); **COMPUTED** (deterministic calculation); **CONSTRUCTED** (hand-built example). Source scripts and results files were supplied in `DNN_Session1_figure_scripts(1).zip`. **The updated guide was checked against the deck, selected official API documentation, and selected numerical records in the supplied results JSON; full training jobs were not independently rerun.** These plotted accuracy and loss values are not population-level guarantees.

### Most important instructor-level corrections

1. **Slides 25 and 53:** Quiz prompts have been revised in the corrected PowerPoint: slide 25 presents the actual double-softmax code error; slide 53 specifies evaluation checks and a suitable pretrained model, and asks for a promising experiment rather than a guaranteed winner.
2. **Slides 42–44:** For symmetric pre-ReLU inputs, `E[ReLU(Z)^2] = ½ E[Z²]`; this is a *second-moment* identity, not `Var(ReLU(Z))=½ Var(Z)`.
3. **Slides 24 and 34:** Cross-entropy simplifies the **output-logit** derivative; it does not delete the hidden-layer Jacobian products. Softmax is not an elementwise activation.
4. **Slides 28 and 29:** Quadratic convergence and 3× forward-pass cost are useful but limited models, not unrestricted theorems about deep networks.
5. **Slides 48 and 50:** PyTorch `Adam(..., decoupled_weight_decay=False)` is coupled by default; `AdamW` (or `Adam(..., decoupled_weight_decay=True)`) uses decoupled weight decay. They are not the same as optimizing an explicit L2 penalty with Adam.
6. **Slide 52:** Three reported runs show a 0.57→0.53 mean best-validation-loss difference; the 0.4 percentage-point accuracy change is inconclusive, and no statistical reliability is claimed.
7. **Slide 47:** Constant-gradient velocity amplification is **not** a stability theorem or a reason to reduce learning rate by a fixed factor of ten; consult the derivation in the speaking notes.
8. **Slides 29–38:** Timing reallocates more time to the worked backpropagation and matrix equations without extending the session.
9. **Slides 69–70:** `OneCycleLR` requires `max_lr` and an update-count specification. A scheduler’s step frequency depends on the scheduler, and `OneCycleLR` usually advances **after each optimizer update**.

---

### Classroom time repair (Part 2)

| Slide(s) | Revised minutes | Teaching priority |
|---|---|---|
| 28–29 | 0:57–1:01 | Local quadratic step-size picture and reverse-mode motivation; be brief on operation-count heuristics. |
| 30 | 1:01–1:03 | Students compute from board **before** showing slide with answers. |
| 31–32 | 1:03–1:08 | Trace local derivatives and parameter gradients. |
| 33–34 | 1:08–1:13 | Vote, then derive matrix backprop with dimension checks. |
| 35–38 | 1:13–1:20 | Autograd, data pipeline, training loop, and validation curves. |

The ten-minute break remains 1:20–1:30, and wrap-up remains 2:20–2:30. Optional mathematical discussion should not crowd out the worked calculation.

## Slide-by-slide teaching notes and anticipated student questions
### Slide 1 — Deep neural networks: from one neuron to networks that train

**Instructor notes.** Frame the three acts: representations, gradient-based learning, and reliable training. Signal colours help students distinguish forward computations from backward gradients.

**Boardwork / teaching move.** **Say:** ‘By the end, you should be able to explain why a network works *and* diagnose why it fails.’ **Show:** point to the teal forward-pass / orange backward-pass key; keep that distinction consistent.

**Likely student question:** Are neural networks fundamentally different from logistic regression?

**Suggested answer:** They generalize it: hidden layers learn nonlinear representations; a classifier can still be a linear readout in the learned space.

**Further anticipated questions and prepared answers**

- **Q: Does a network “reason” like a person?**  **A:** Not implied. In this lecture it is a trainable mathematical function.

### Slide 2 — By the end of today you can…

**Instructor notes.** Treat the six learning objectives as measurable outcomes. Tell students they will hand-compute gradients and interpret training curves, not merely recognize buzzwords.

**Boardwork / teaching move.** **Do:** State the prerequisites as testable skills: one matrix-vector product and one chain-rule derivative. Ask for a show of hands on which feels less familiar so the algebra can be paced accordingly.

**Likely student question:** Do we have to implement PyTorch today?

**Suggested answer:** No. The deck uses PyTorch vocabulary to bridge to the next practical session; focus today on reasoning and shapes.

**Further anticipated questions and prepared answers**

- **Q: Will we prove a universal-approximation theorem today?**  **A:** Only the informal statement and its limitations are covered; formal proof is outside this session.

### Slide 3 — Today's plan, and one example we will follow all session

**Instructor notes.** Introduce the same Fashion-MNIST 784–128–64–10 network throughout. Distinguish its reproducible training results from conceptual illustrations and toy simulations.

**Boardwork / teaching move.** **Anchor:** Write `784 → 128 → 64 → 10` in a corner of the board and reuse it for tensor shapes, losses, optimization, and validation. The numbers are architecture dimensions, not learning results.

**Likely student question:** Why choose Fashion-MNIST?

**Suggested answer:** It is a manageable ten-class 28×28 benchmark: harder than handwritten digits, but small enough for repeatable classroom experiments.

**Further anticipated questions and prepared answers**

- **Q: Are the same images used for train and validation?**  **A:** No. The benchmark uses a held-out subset of the original training portion for validation.

### Slide 4 — Foundations: what a network computes

**Instructor notes.** Start with what a neuron computes, then show the geometric limitation, then build a representation that defeats the limitation.

**Boardwork / teaching move.** **Ask first:** ‘What can a single decision boundary look like?’ Elicit a line before introducing higher-dimensional hyperplanes.

**Likely student question:** What does 'foundation' mean here?

**Suggested answer:** Understanding the mapping from inputs to outputs before optimizing its parameters.

**Further anticipated questions and prepared answers**

- **Q: Is there learning inside a neuron during the forward pass?**  **A:** No. Forward propagation computes using current weights; updates happen during optimization.

### Slide 5 — One neuron: a weighted sum and a threshold

**Instructor notes.** Derive z=wᵀx+b and threshold(z). In 2-D, w is perpendicular to the line z=0; b moves the line without rotating it when w is fixed. Signed distance is z/‖w‖ (w≠0). Avoid implying this is a biological model.

**Boardwork / teaching move.** **Board derivation:** Let `w=(2,−1), b=−3`; the boundary is `2x₁−x₂−3=0`, and `w·(x−x′)=0` for two points on that boundary. Translate the line by changing only `b`. The signed-distance formula requires `w ≠ 0`.

**Likely student question:** Does the bias change the boundary's angle?

**Suggested answer:** Not if w is held fixed: changing b translates the line. Changing w changes its orientation.

**Further anticipated questions and prepared answers**

- **Q: Is the signed distance still defined if all weights are zero?**  **A:** No. If w=0 there is no proper separating hyperplane, and the formula divides by zero.
- **Q: Can two very different pairs of w and b give the same boundary?**  **A:** Yes. Multiplying both by any positive constant leaves the boundary and its positive side unchanged; a negative constant reverses which side is positive.

### Slide 6 — Linear models can only draw straight boundaries

**Instructor notes.** Compare the SAME two-moons data and emphasize the distinction between a linear boundary and a nonlinear activation. These 86% and 100% figures refer to the plotted *training runs*, not guarantees of test accuracy.

**Boardwork / teaching move.** **Probe the figure:** Ask students which of the two-moons points a single straight separator necessarily misses. Distinguish a nonlinear *probability map* from a nonlinear *classification boundary*. These percentages are training accuracies for these runs.

**Likely student question:** But logistic regression uses a nonlinear sigmoid—why a straight line?

**Suggested answer:** Because sigmoid is monotone, σ(z)>0.5 exactly when z>0. The boundary is still wᵀx+b=0.

**Further anticipated questions and prepared answers**

- **Q: If the threshold is set to 0.8 rather than 0.5, does the boundary curve?**  **A:** No. A fixed sigmoid threshold t in (0,1) corresponds to the affine level set wᵀx+b=log[t/(1−t)].
- **Q: Would a kernel logistic model still be limited to a straight input-space boundary?**  **A:** No. Nonlinear input features or kernels can create curved input-space boundaries even with a linear readout in feature space.

### Slide 7 — XOR: no straight line separates the classes

**Instructor notes.** Let students try separating the four XOR corners. Then add the inequalities to obtain the contradiction: b≤0, w₁+b>0, w₂+b>0, w₁+w₂+b≤0 cannot all hold.

**Boardwork / teaching move.** **Two-minute proof:** Write the four required inequalities for `(0,0),(0,1),(1,0),(1,1)`. Adding the two positive examples gives `w₁+w₂+2b>0`; using `b≤0` implies `w₁+w₂+b>0`, contradicting the negative `(1,1)` case.

**Likely student question:** Could another threshold value solve XOR?

**Suggested answer:** A different threshold can be absorbed into b; it still gives one half-space and cannot separate XOR.

**Further anticipated questions and prepared answers**

- **Q: Can two threshold neurons solve XOR?**  **A:** With an appropriate hidden layer and output combination, yes; one threshold neuron alone cannot.
- **Q: Does changing the bias solve XOR?**  **A:** No. Any bias merely moves a straight separating line; the contradiction covers every b.

### Slide 8 — A better representation makes XOR easy

**Instructor notes.** Map (x₁,x₂)↦(x₁,x₂,x₁x₂). Now x₁+x₂−2x₁x₂ gives the XOR label exactly on binary inputs. The added feature was engineered by us, not learned.

**Boardwork / teaching move.** **Show the coordinate trick:** Table the four rows of `(x₁,x₂,x₁x₂)` and verify `x₁+x₂−2x₁x₂`. Emphasize that this is a *hand-designed* feature. Do not suggest the mapping is linear in the original inputs.

**Likely student question:** Is XOR itself linear in the new features?

**Suggested answer:** Yes, its values on the four binary inputs follow a linear expression in the augmented features; use a 0.5 threshold to classify.

**Further anticipated questions and prepared answers**

- **Q: Why not always manually add every interaction feature?**  **A:** Their number grows quickly with dimensionality; feature engineering also requires task insight. Learned features can scale better, but they add optimization and data requirements.

### Slide 9 — The neural-network idea: learn the features too

**Instructor notes.** Contrast feature engineering with learning h=g(Wx+c) jointly with a task-specific readout. The 'edge detector' claim is a common vision example, not a guarantee for all modalities.

**Boardwork / teaching move.** **Bridge:** Contrast (i) engineer `x₁x₂`, then fit a linear classifier with (ii) learn useful hidden activations using task loss. Representations need not correspond one-to-one with human concepts.

**Likely student question:** Does the network discover human-readable features automatically?

**Suggested answer:** Not necessarily: representations are shaped by the training objective and may be distributed and hard to interpret.

**Further anticipated questions and prepared answers**

- **Q: Can learned representations transfer to a different task?**  **A:** Sometimes, especially when pretraining and target tasks share structure; transfer is empirical, not guaranteed.

### Slide 10 — Smooth activations give gradients: the sigmoid neuron

**Instructor notes.** The hard step has zero derivative almost everywhere. Sigmoid yields differentiable probability predictions when linear *log-odds* are assumed: p=1/(1+e^(−z)). Stress that the perceptron can be trained by its special mistake-driven rule, not ordinary backprop through steps.

**Boardwork / teaching move.** **Derive:** From `log[p/(1−p)]=z` obtain `p=eᶻ/(1+eᶻ)`. On the board differentiate to `σ′(z)=σ(z)(1−σ(z))`; have students identify its maximum at `σ=1/2`.

**Likely student question:** Why not keep the step for all hidden neurons?

**Suggested answer:** Standard gradient-based backprop receives zero or undefined local derivatives, so it cannot efficiently train arbitrary stacks of steps.

**Further anticipated questions and prepared answers**

- **Q: Is a step function nonlinear?**  **A:** Yes, although it is non-differentiable at zero and flat elsewhere.
- **Q: If sigmoid has a gradient everywhere, why can training still stall?**  **A:** Its derivative approaches zero in saturated regions, and products through layers can attenuate backpropagated signals.

### Slide 11 — Without a nonlinearity, depth collapses to one layer

**Instructor notes.** Multiply two affine maps explicitly: W₂(W₁x+b₁)+b₂=(W₂W₁)x+(W₂b₁+b₂). Induction collapses any number of affine layers. Note that skip connections without nonlinearities also remain affine.

**Boardwork / teaching move.** **Substitution, not assertion:** Write `z₂=W₂(W₁x+b₁)+b₂`. Circle `W₂W₁` and `W₂b₁+b₂`. The composition remains **affine**, not necessarily strictly linear (bias matters).

**Likely student question:** Why is a nonlinearity needed between layers rather than just at the end?

**Suggested answer:** An output sigmoid turns a single affine score into a probability but cannot curve its class boundary.

**Further anticipated questions and prepared answers**

- **Q: Does the collapse theorem apply when biases are present?**  **A:** Yes. A composition of affine maps remains affine.
- **Q: Could a network with only linear layers still be useful?**  **A:** It can factorize a linear map or impose rank constraints, but it cannot represent nonlinear input–output dependencies.

### Slide 12 — Quick check 1

**Instructor notes.** Correct option C. Have students explain the difference between a nonlinear output value and a nonlinear *boundary*; with a 0.5 threshold an output sigmoid preserves the affine boundary.

**Boardwork / teaching move.** **Peer instruction:** After the vote, request a reason from a student who chose ‘curved’ and a counterargument from someone who chose C. Mark the threshold `σ(z)=0.5 ⇔ z=0`.

**Likely student question:** Would a different sigmoid threshold bend the boundary?

**Suggested answer:** No; for any fixed threshold in (0,1) it is a different level set of the same affine score, hence still a hyperplane.

**Further anticipated questions and prepared answers**

- **Q: Would putting sigmoid on hidden layers change the answer?**  **A:** Yes. Nonlinear hidden activations generally make curved input-space boundaries possible.

### Slide 13 — Sigmoid and tanh saturate; ReLU does not for z > 0

**Instructor notes.** Use the derivative plots to distinguish saturation from vanishing through matrix products. The bound 0.25¹⁰ concerns only sigmoid derivative factors, not the total network gradient. ReLU has unit slope for positive z but zero for negative z.

**Boardwork / teaching move.** **Use both panels:** Point to sigmoid output near 1 and then to derivative near 0. For a scalar chain, sketch `δ₁ = (w₂g′(z₁))···δ_L` to demonstrate products; warn that weights can also amplify gradients.

**Likely student question:** Does ReLU guarantee no vanishing gradient?

**Suggested answer:** No. Inactive units have zero derivative, and the weight matrices can still shrink or distort the gradient.

**Further anticipated questions and prepared answers**

- **Q: Why does tanh often train better than sigmoid in older networks?**  **A:** It is zero-centered and has slope 1 at zero, although it still saturates for large |z|.
- **Q: Does ReLU have a derivative at exactly zero?**  **A:** No classical derivative there. Implementations select a subgradient convention; PyTorch uses zero for ReLU at zero.
- **Q: Can a dying ReLU ever recover?**  **A:** If the unit is inactive for every training example and its only paths through the activation have zero derivative, gradient descent alone cannot change its incoming weights on those examples. Other architecture paths or changes in its input distribution may alter this conclusion.

### Slide 14 — Two ReLU units solve XOR by remapping the space

**Instructor notes.** Trace the four XOR inputs through h₁=ReLU(x₁+x₂) and h₂=ReLU(x₁+x₂−1); output h₁−2h₂ gives 0,1,1,0. These hidden units are 'OR-like' and 'AND-like', not binary gates on arbitrary real inputs.

**Boardwork / teaching move.** **Trace with a table:** Fill columns `x₁, x₂, h₁, h₂, y` for each XOR corner. Draw the mapped points in hidden coordinates and show a **straight** separator there.

**Likely student question:** Why do two different XOR points map to the same hidden location?

**Suggested answer:** The representation needs only preserve information relevant to the label, not preserve every input distinction.

**Further anticipated questions and prepared answers**

- **Q: Is the output y=h₁−2h₂ itself a sigmoid probability?**  **A:** No. In this construction y is a real-valued expression exactly equal to the binary XOR target at the four corners. A threshold at 0.5 yields the labels.
- **Q: Does gradient descent necessarily learn OR-like and AND-like hidden units?**  **A:** No. Many parameterizations represent XOR, and the optimizer has no reason to choose this one.

### Slide 15 — A ReLU network computes a patchwork of linear pieces

**Instructor notes.** Within each fixed on/off pattern of ReLUs the mapping is affine; therefore a zero-logit decision boundary is locally straight. The observed 146 activation regions are grid-dependent and are a *lower bound* on regions in that window, not a theorem about this architecture.

**Boardwork / teaching move.** **Animate mentally:** For a fixed set of active ReLUs, replace each ReLU by either identity or zero; all remaining operations are affine. A decision curve is assembled from locally straight pieces. The reported region count comes from a finite grid.

**Likely student question:** Are all ReLU networks globally linear?

**Suggested answer:** No. They are piecewise affine; changing which neurons are active changes the affine map.

**Further anticipated questions and prepared answers**

- **Q: How many linear regions can a ReLU network have?**  **A:** It depends on architecture, input dimension and learned weights. The 146 is a count inferred from this finite-grid visualization, not a universal maximum.
- **Q: Can two different activation patterns compute the same affine map?**  **A:** Yes. Activation patterns distinguish computational regions; maps can coincide by parameter symmetries or degeneracies.

### Slide 16 — Universal approximation, stated honestly

**Instructor notes.** Use the constructed ReLU interpolants to motivate the theorem: any continuous function on a compact domain can be approximated uniformly by a sufficiently wide one-hidden-layer ReLU network. Existence does not imply feasible width, trainability, or generalization.

**Boardwork / teaching move.** **Teach the quantifiers:** Write `∀ continuous f on compact K, ∀ ε>0, ∃ finite-width network F such that sup_K|F−f|<ε`. Underline ‘there exists’ and contrast with ‘gradient descent finds it’. The slide's curves are deliberately constructed, not fitted by learning.

**Likely student question:** If a shallow network is universal, why build a deep network?

**Suggested answer:** Depth can represent some compositional functions much more efficiently, and architectural inductive biases often make training and reuse easier.

**Further anticipated questions and prepared answers**

- **Q: Does universal approximation guarantee exact representation of a function?**  **A:** No. It says approximation to any prescribed positive error under the theorem assumptions.
- **Q: Does this include discontinuous targets under the uniform-norm statement?**  **A:** Not generally on compact domains: continuous ReLU networks cannot uniformly approximate a genuinely discontinuous target arbitrarily closely.
- **Q: Can it approximate a high-dimensional function efficiently?**  **A:** Not guaranteed. Required width may grow very rapidly with input dimension; structural assumptions can change the rates.
- **Q: What is the difference between expressive power and generalization?**  **A:** Expressive power asks whether parameters exist that fit a function; generalization asks whether a learned model performs well on new data.

**Accuracy caution.** Theory establishes existence only; neither the measured constructed-example errors nor an architecture’s generalization follows from the theorem.

### Slide 17 — The forward pass repeats one rule per layer

**Instructor notes.** For each layer, compute z[l]=W[l]a[l−1]+b[l] and a[l]=g[l](z[l]). With column vectors, W[l] has shape (n_l,n_(l−1)). The output activation depends on the task.

**Boardwork / teaching move.** **Shape proof:** Every row of `W[l]` gives one destination neuron's weighted sum. Draw `(n_l, n_{l−1}) @ (n_{l−1},1) = (n_l,1)`; when switching to batch-row PyTorch layout, transpose the weight.

**Likely student question:** Why does the first layer count as layer 1?

**Suggested answer:** The input is designated layer 0 because it contains no learned affine transformation.

**Further anticipated questions and prepared answers**

- **Q: What does a bias vector contribute to the shape?**  **A:** A per-output additive term with shape (n_l,1) for a single column-vector example.
- **Q: Is every layer required to use the same activation?**  **A:** No. Hidden layers and outputs can use different functions to suit their roles.

### Slide 18 — Running example: a 784–128–64–10 MLP

**Instructor notes.** Trace B×1×28×28→B×784→B×128→B×64→B×10. Batch axis B remains unchanged throughout this sequential MLP, not necessarily across every architecture (e.g., batching operations). Output values are raw class scores.

**Boardwork / teaching move.** **Read shapes aloud:** `B×1×28×28 → B×784 → B×128 → B×64 → B×10`. Ask whether flattening changes the *information* or only its organization. Distinguish each logit from a probability.

**Likely student question:** What does a logit of 4 mean?

**Suggested answer:** A raw score, not a 400% probability. Relative logits determine probabilities after softmax.

**Further anticipated questions and prepared answers**

- **Q: Why not apply softmax in the final layer?**  **A:** For multiclass cross-entropy, PyTorch expects raw logits and computes stable log-softmax internally.
- **Q: Why does flattening lose spatial structure for an MLP?**  **A:** Flattening retains pixel values but eliminates explicit local-neighborhood structure in the architecture; fully connected weights do not share across locations.

### Slide 19 — In PyTorch terms: a stack of layers that returns logits

**Instructor notes.** Count parameters aloud: 784×128+128=100,480; 128×64+64=8,256; 64×10+10=650; total 109,386. PyTorch stores nn.Linear weight as (out,in) and batches inputs as rows.

**Boardwork / teaching move.** **Count on board:** The first layer contributes `784·128+128`; let students compute the remaining layers in pairs. The first fully connected layer holds ~91.9% of parameters even though most of the network's depth comes later.

**Likely student question:** Why do we transpose the weight matrix in PyTorch's expression?

**Suggested answer:** Each output unit's weights occupy one row; a B×in batch multiplies Wᵀ (in×out) to produce B×out.

**Further anticipated questions and prepared answers**

- **Q: Does the 109,386 count depend on batch size?**  **A:** No. Parameters belong to layers; batch size changes activations and compute per update, not the number of weights.
- **Q: Is 92% an exact fraction?**  **A:** It is rounded: 100,480/109,386 is about 91.86%.

### Slide 20 — Quick check 2

**Instructor notes.** Correct B: (128,784). Ask how many entries are weights and how many additional entries are biases.

**Boardwork / teaching move.** **Peer check:** When votes split between `(784,128)` and `(128,784)`, ask ‘how many *output* units are there?’ One row of parameters belongs to each output.

**Likely student question:** Is layer.bias.shape also (128,784)?

**Suggested answer:** No, biases have shape (128,), one scalar per output unit.

**Further anticipated questions and prepared answers**

- **Q: What is the shape of the layer output for 32 examples?**  **A:** (32,128) for a (32,784) input.

### Slide 21 — How a network learns

**Instructor notes.** Bridge from a deterministic forward pass to selecting a loss, taking gradients, and training on data.

**Boardwork / teaching move.** **Bridge:** Write `x → model → logits → loss J → gradients → optimizer step`. Ask which boxes define a model, a training objective, and an algorithm.

**Likely student question:** If a network's predictions are wrong, which weights should change?

**Suggested answer:** All parameters connected to the loss may receive gradients; backprop calculates each sensitivity.

**Further anticipated questions and prepared answers**

- **Q: Does loss measure whether the answer is “right”?**  **A:** It produces a training signal and often measures confidence as well as correctness; it need not equal classification accuracy.

### Slide 22 — Most losses are negative log-likelihoods

**Instructor notes.** Derive MSE from homoscedastic Gaussian observation noise with *fixed* variance, binary CE from Bernoulli, multiclass CE from categorical likelihood. These are modeling choices, not the only valid losses. PyTorch classification losses expect logits.

**Boardwork / teaching move.** **Likelihood-to-loss derivation:** Gaussian `y|x ~ N(fθ(x), σ²)` gives `−log p = (y−fθ(x))²/(2σ²)+constant` for fixed σ. Bernoulli gives `−[y log p+(1−y)log(1−p)]`; categorical gives `−log p_y`. The table lists common NLL losses, not every possible loss function.

**Likely student question:** Why can cross-entropy be better than MSE for class labels?

**Suggested answer:** It matches a probabilistic label model and avoids an extra saturating sigmoid derivative in the output logit gradient.

**Further anticipated questions and prepared answers**

- **Q: Is MSE *exactly* a Gaussian NLL?**  **A:** Up to scaling and additive constants if the conditional variance is fixed. If variance is learned, the log-variance term matters.
- **Q: Does one-hot encoding turn a categorical target into ten independent Bernoulli variables?**  **A:** No. The standard single-label ten-class model is one categorical observation with probabilities summing to one.
- **Q: What if an example may belong to multiple classes simultaneously?**  **A:** Use independent binary logits and BCEWithLogitsLoss, not single-label softmax CE.
- **Q: Are all losses negative log-likelihoods?**  **A:** No. Margin losses, contrastive objectives, task-specific metrics and some robust losses are not simply the NLLs shown here.

**Accuracy caution.** “Most losses” is an overstatement if read literally; the slide illustrates three common likelihood-based losses.

### Slide 23 — Softmax, and the first-loss sanity check

**Instructor notes.** Compute softmax for logits [2,1,0.1], illustrating relative scores and shift invariance. Uniform logits yield ln(10)≈2.303 cross-entropy for a ten-class problem; the deck's 2.63 is a measured run, not a universal initial-loss target.

**Boardwork / teaching move.** **Board calculation:** With logits `(2,1,0.1)`, use `e²/(e²+e¹+e^0.1)` to estimate the first softmax probability (~0.659). Subtracting the largest logit before exponentiating leaves each probability unchanged and prevents overflow.

**Likely student question:** Does larger absolute logit always mean larger predicted probability?

**Suggested answer:** Only relative to other logits of that same example; adding the same constant to all logits changes nothing.

**Further anticipated questions and prepared answers**

- **Q: Why is the uniform-class baseline loss ln K rather than K?**  **A:** Cross-entropy for the true class is −ln(1/K)=ln K.
- **Q: If all logits are 1, is softmax nonuniform?**  **A:** No. Every logit is equal, so all K probabilities are 1/K.
- **Q: What does an initial loss of 2.63 prove?**  **A:** Only that this particular initialization/data/target combination has that average loss; it is not a diagnostic guarantee.

**Accuracy caution.** An unusual first loss is a diagnostic clue, not proof of a particular bug.

### Slide 24 — Cross-entropy keeps learning when MSE stalls

**Instructor notes.** For L=½(σ(z)−y)², derivative is (σ−y)σ(1−σ). For Bernoulli CE, derivative is σ−y. The useful cancellation occurs at the final sigmoid/softmax, not through all earlier layers.

**Boardwork / teaching move.** **Derivative side-by-side:** Set `p=σ(z), y=1`. With `L_MSE=½(p−y)²`, chain rule gives `(p−y)p(1−p)`. With `L_CE=−[y log p+(1−y)log(1−p)]`, differentiating via `p` cancels `p(1−p)` to give `p−y`. State explicitly that hidden-layer derivatives do not disappear.

**Likely student question:** Does cross-entropy solve vanishing gradients in deep sigmoid hidden layers?

**Suggested answer:** No. It removes the final activation factor in the output error, but hidden layers can still saturate.

**Further anticipated questions and prepared answers**

- **Q: What does ‘sigmoid derivative cancellation’ really mean?**  **A:** When differentiating Bernoulli cross-entropy composed with sigmoid, the algebra simplifies the final-logit derivative to p−y.
- **Q: Does CE always create a large gradient for every parameter when the prediction is wrong?**  **A:** No. The final-logit gradient can be large while hidden-layer Jacobians still make upstream parameter gradients small.
- **Q: What changes for softmax with K classes?**  **A:** For one example with one-hot target, each logit derivative is p_k−y_k. Averaging a batch multiplies the summed gradients by 1/B.
- **Q: Why might MSE still be used on outputs?**  **A:** It is standard for many continuous regression targets and useful pedagogically; it is usually not the preferred likelihood-based loss for Bernoulli labels.

**Accuracy caution.** Cross-entropy does not solve vanishing gradients through hidden layers.

### Slide 25 — Quick check 3

**Instructor notes.** In the revised slide, the **code** ends with Softmax and feeds that output to `CrossEntropyLoss`. The uniquely supported fix is B: remove the redundant output softmax. A loss plateau around 1.5 by itself is not diagnostic.

**Boardwork / teaching move.** Point to the code and ask what the loss expects. Then explain the bound `log(K−1+e)−1≈1.46` (K=10) as a consequence of the bug, not a universal signature.

**Likely student question:** Does a training-loss plateau alone prove the 'softmax twice' bug?

**Suggested answer:** No. The revised quiz identifies the code misuse explicitly; an observed plateau alone has many possible causes.

**Further anticipated questions and prepared answers**

- **Q: Could a very low learning rate produce a plateau at 1.5?**  **A:** Yes. So could inadequate training, data errors, excessive regularization, label noise, or model limitations.
- **Q: What experiment distinguishes the double-softmax mistake?**  **A:** Inspect the final model operation and loss call, remove the extra softmax, and compare loss on the same fixed logits/labels.
- **Q: Why is option D not a general correction?**  **A:** CrossEntropyLoss accepts integer class IDs, usually preferred; it also supports correctly formatted probability targets.

**Accuracy caution.** The revised prompt names the actual code misuse. An isolated loss plateau still cannot prove its cause.

### Slide 26 — Pass raw logits to the loss: softmax twice fails silently

**Instructor notes.** CrossEntropyLoss internally uses stable log-softmax on logits. If p=softmax(z) is incorrectly passed instead, components of p lie in [0,1] and the best possible one-hot per-example loss has limit log(e+K−1)−1≈1.461 for K=10. This is a double *normalization*, not literally applying one softmax twice in the source.

**Boardwork / teaching move.** **Prove the loss floor:** Passing an already-normalized vector `p∈Δ^{K−1}` to CE produces `log(∑_j e^{p_j})−p_y`. Best case is `p_y→1` and all other `p_j→0`, giving `log(e+K−1)−1 ≈ 1.461` for K=10. This is a lower bound for the *buggy input pipeline*, not for correct CE on logits.

**Likely student question:** Why might accuracy still look high after this bug?

**Suggested answer:** The argmax is preserved by softmax, but gradients and probabilistic calibration differ; optimization can still learn a decision boundary.

**Further anticipated questions and prepared answers**

- **Q: How do you compute the numerical floor for K=10?**  **A:** ln(e+9)−1 ≈ 1.461.
- **Q: Does placing softmax before CE always make accuracy terrible?**  **A:** No. The argmax of a fixed score vector is unchanged by softmax, and training can still update parameters, though the gradient and probability calibration can be poor.
- **Q: Is softmax itself numerically unstable in all circumstances?**  **A:** No. Libraries have stable softmax implementations; the central problem is applying CE to probabilities when it expects logits.
- **Q: Should I ever calculate probabilities?**  **A:** Yes, for reporting/visualization/inference, after training loss computation, with suitable calibration checks.

**Accuracy caution.** The floor formula applies only when the input supplied to CE is a simplex probability vector; correct logits CE has no positive loss floor.

### Slide 27 — Choose the activation by the layer's role

**Instructor notes.** Differentiate role-specific choices: ReLU/variants inside MLPs; GELU or gated SwiGLU in transformers; raw output logits for CrossEntropyLoss/BCEWithLogitsLoss; identity for unconstrained regression. Positive-output link functions can be chosen by target distribution.

**Boardwork / teaching move.** **Explain by target semantics:** In hidden layers choose transformations with useful derivatives; at the classifier output send raw logits to CE/BCEWithLogits. For unconstrained real regression the output is usually identity; a positive target can require a positive link, provided the corresponding likelihood is appropriate.

**Likely student question:** Why can regression need a different output activation?

**Suggested answer:** Its target can be real-valued, strictly positive, bounded, or categorical; output constraints should match those semantics.

**Further anticipated questions and prepared answers**

- **Q: Why is softmax not a hidden-layer default?**  **A:** It couples all coordinates and constrains them to a simplex; ordinary hidden transformations typically do not need that constraint.
- **Q: Can a regression output use sigmoid?**  **A:** Yes if the target is truly bounded in [0,1] and the modeling choice is appropriate; an unconstrained real target should not be forced into that interval.

### Slide 28 — Gradient descent: the sharpest direction caps η

**Instructor notes.** Taylor expansion explains a short move along −∇J. On J=½(θ₁²+10θ₂²), the high-curvature coordinate obeys θ'=(1−10η)θ, requiring 0<η<0.2 for convergence from generic nonzero initial conditions. Do not present this as a global neural-network theorem.

**Boardwork / teaching move.** **Quadratic on board:** For `J(θ)=½λθ²`, `∇J=λθ` and one step gives `θ_{t+1}=(1−ηλ)θ_t`. Convergence from nonzero θ requires `|1−ηλ|<1`, i.e. `0<η<2/λ`. With λ_max=10, the exact limit is 0.2 *for this fixed positive-definite quadratic*, not for arbitrary deep networks.

**Likely student question:** Why not always use a tiny learning rate?

**Suggested answer:** It may be stable but prohibitively slow; curvature changes during training, and adaptive schedules often improve efficiency.

**Further anticipated questions and prepared answers**

- **Q: Is the negative gradient always the steepest local descent?**  **A:** For a Euclidean norm constraint on infinitesimal steps, yes. Different geometry or preconditioning leads to other steepest-descent directions.
- **Q: What happens exactly at η=2/λ for a scalar quadratic?**  **A:** The iterate alternates sign without converging when initialized away from zero.
- **Q: Does this criterion hold globally in a neural network?**  **A:** No. Network curvature varies with parameters, and stochastic minibatch gradients make the fixed-quadratic result a local intuition.
- **Q: Does a smaller learning rate guarantee better test performance?**  **A:** No. Optimization speed and implicit regularization may change; the validation objective must still be checked.

**Accuracy caution.** The η<2/λ_max limit is exact for the given positive-definite quadratic with full-gradient descent, not a universal network stability criterion.

### Slide 29 — Backpropagation: every gradient for ≈ 3× a forward pass

**Instructor notes.** Backprop is reverse-mode automatic differentiation, distinct from the optimizer. For a single scalar loss and many parameters it shares intermediate derivatives. The ≈3× forward-work figure is an order-of-magnitude rule, hardware and layer dependent.

**Boardwork / teaching move.** **Draw a shared graph:** If many weights feed one scalar loss, backprop propagates a single output sensitivity backward while reusing intermediate activations. Contrast with parameter-wise finite differences. ‘≈3× forward’ is only a cost heuristic.

**Likely student question:** Would numerical finite differences work for checking gradients?

**Suggested answer:** Yes, for a few chosen parameters; it is too expensive to compute a complete gradient for a large network.

**Further anticipated questions and prepared answers**

- **Q: Why does backprop sum over multiple graph paths?**  **A:** The multivariate chain rule adds contributions whenever one intermediate value affects the loss by more than one downstream route.
- **Q: Does backprop change weights itself?**  **A:** No. It computes gradients; an optimizer performs parameter updates.
- **Q: Is its runtime exactly three forward passes?**  **A:** No. It is a rough rule of thumb; operation type, GPU kernels, memory and checkpointing change the cost.

**Accuracy caution.** Runtime multiplier varies by model and hardware; use approximate language.

### Slide 30 — Worked example: forward, then backward, by hand

**Instructor notes.** The slide visibly contains the answers (`z₁=1.1`, `a₁≈0.7503`, `z₂≈0.3002`, `a₂≈0.5745`, `L≈0.0905`). Therefore ask students to calculate **before showing slide 30**.

**Boardwork / teaching move.** While slide 29 is still projected, write `x=2`, `y=1`, `w₁=0.5`, `b₁=0.1`, `w₂=0.8`, `b₂=−0.3`, sigmoid at each neuron, and `L=½(a₂−y)²` on the board; let students work, then advance. No click-to-reveal animation is assumed.

**Likely student question:** Why use MSE and sigmoid here when you recommend BCE for classification?

**Suggested answer:** To make every local chain-rule factor visible in a tiny worked example, not as the recommended classification training setup.

**Further anticipated questions and prepared answers**

- **Q: What is a preactivation?**  **A:** The weighted sum plus bias before applying sigmoid/ReLU, written z.
- **Q: Why is the loss ½ times the squared error?**  **A:** The factor ½ makes the derivative a₂−y; its omission would scale this toy gradient by two.

### Slide 31 — Worked example: forward, then backward, by hand

**Instructor notes.** Define δᵢ=∂L/∂zᵢ. From ∂L/∂a₂≈−0.4255 obtain δ₂≈−0.1040; pass through w₂=0.8 and σ'(z₁)≈0.187 to get δ₁≈−0.0156. The 6.7× attenuation applies to this scalar path.

**Boardwork / teaching move.** **Use local derivatives:** In orange write `∂L/∂a₂ = a₂−y`, `δ₂=(a₂−y) a₂(1−a₂)`, then `δ₁=(w₂δ₂)a₁(1−a₁)`. Circle each factor and show numerically where the ~6.7 attenuation occurs.

**Likely student question:** Why multiply by w₂ in the backward pass?

**Suggested answer:** Because z₂=w₂a₁+b₂, so ∂z₂/∂a₁=w₂; the chain rule includes this local derivative.

**Further anticipated questions and prepared answers**

- **Q: What does δ₂ represent?**  **A:** The derivative of the scalar loss with respect to the *preactivation* z₂, not the difference between output and label in general.
- **Q: Where does the factor 0.187 come from?**  **A:** It is σ′(z₁)=a₁(1−a₁)≈0.7503·0.2497.
- **Q: Is the 6.7 factor a universal decay per sigmoid layer?**  **A:** No. It depends on the particular activation and weight values on this path.

### Slide 32 — Worked example: forward, then backward, by hand

**Instructor notes.** Conclude ∂L/∂w₁=δ₁x≈−0.0312 and ∂L/∂w₂=δ₂a₁≈−0.0780. Bias derivatives are δ₁ and δ₂. The negative sign means increasing these weights locally lowers the loss for this example.

**Boardwork / teaching move.** **Connect to weights:** A weight changes a preactivation by the *value that flows through it*. For scalar connections, `∂L/∂w₂=δ₂a₁`, `∂L/∂b₂=δ₂`; for the first layer, replace `a₁` by `x`.

**Likely student question:** Why is the weight gradient 'error times input'?

**Suggested answer:** Because z=wa+b, ∂z/∂w=a; multiplying by δ=∂L/∂z yields ∂L/∂w=δa.

**Further anticipated questions and prepared answers**

- **Q: Why does ∂L/∂b₂ equal δ₂?**  **A:** Because ∂z₂/∂b₂=1.
- **Q: If the gradient is negative, should gradient descent decrease that parameter?**  **A:** No. The update subtracts η times the negative gradient, so it increases that parameter for this local step.

### Slide 33 — Quick check 4

**Instructor notes.** Correct B: setting w₂=0 severs this hidden activation's influence on the output, so δ₁=0 and ∂L/∂w₁=0 at that instant. Clarify that w₂ itself can receive a nonzero gradient and become nonzero on later steps.

**Boardwork / teaching move.** **Separating connectivity from future learning:** At `w₂=0`, first-layer sensitivity through that path is zero *on this step*, but `∂L/∂w₂` can still be nonzero, so future steps can reconnect the path.

**Likely student question:** Will the first layer *never* learn after setting w₂=0?

**Suggested answer:** Not necessarily. The update of w₂ may make it nonzero; then first-layer gradients can resume.

**Further anticipated questions and prepared answers**

- **Q: Can the zero hidden gradient coexist with a nonzero output gradient?**  **A:** Yes. The zero connection blocks sensitivity to earlier activations, while its own parameter has derivative δ₂ a₁.
- **Q: Does removing a connection mean its parameter gradient must be zero?**  **A:** Not if the connection remains a trainable parameter currently set to zero; setting the parameter to zero is not the same as removing it from the computation graph.

### Slide 34 — The four equations of backpropagation

**Instructor notes.** Allow about **three minutes** here, following three preceding minutes for the backward numeric trace. Define `δ[l]=∂L/∂z[l]`, establish the recurrence with `Wᵀ`, and derive the outer-product weight gradient from its entries.

**Boardwork / teaching move.** Show `∂L/∂W_ij[l]=δ_i[l]a_j[l−1]`; stacking entries yields `δ[l]a[l−1]ᵀ`, matching the weight's shape. For row-oriented PyTorch mini-batches, mean-reduced gradient is `Δᵀ A_prev/B` if Δ contains per-example unaveraged δ. Softmax requires its full Jacobian; softmax+CE simplifies to `p−y` for one example.

**Likely student question:** Why transpose W during backprop?

**Suggested answer:** The forward layer maps previous activations to current units, while the backward Jacobian-vector product maps current sensitivities back to previous ones.

**Further anticipated questions and prepared answers**

- **Q: Why define δ rather than differentiate every weight directly?**  **A:** Every incoming weight to one neuron shares the same downstream sensitivity δ; reusing it avoids repeated chain-rule work.
- **Q: Why does W transpose in the hidden-layer recurrence?**  **A:** Each upstream activation influences many downstream units. The gradient with respect to that activation is the sum over those units, represented by Wᵀδ.
- **Q: Why is the weight gradient an outer product?**  **A:** Each entry ∂L/∂W_ij equals δ_i times a_prev,j, so the matrix of all entries is δ a_prevᵀ.
- **Q: How does mini-batching change the formulas?**  **A:** For batch-row matrices A_prev∈R^{B×n_prev}, Δ∈R^{B×n_next}, mean-reduced gradients are (Δᵀ A_prev)/B, if each row of Δ holds the unreduced per-example δ.
- **Q: Is softmax just another elementwise activation for BP1?**  **A:** No. Its outputs interact. The Jacobian has off-diagonal terms, though the softmax+CE combination simplifies to p−y.

**Accuracy caution.** The first output-error equation assumes elementwise activation; softmax needs its Jacobian, even though softmax + CE simplifies.

### Slide 35 — In PyTorch, autograd runs these equations for you

**Instructor notes.** Explain leaf tensor gradients and the computational graph. `.backward()` accumulates gradients, so use `zero_grad()` unless accumulation is intentional. Validation uses `eval()` to change module behavior and `no_grad()` (or inference mode) to avoid graph construction—two different functions.

**Boardwork / teaching move.** **Practical distinction:** `.backward()` computes and **accumulates** gradients; `optimizer.step()` changes parameters. `model.eval()` changes Dropout/BatchNorm behavior but *does not disable autograd*; wrap evaluation in `torch.no_grad()` or, where appropriate, inference mode.

**Likely student question:** Does `model.eval()` disable gradient tracking?

**Suggested answer:** No. It switches behavior of modules such as Dropout and BatchNorm; use a no-grad context separately for inference.

**Further anticipated questions and prepared answers**

- **Q: What is a leaf tensor?**  **A:** A tensor created as an independent input/parameter to the computation graph; trainable leaf parameters normally accumulate their `.grad`.
- **Q: Why use model(x) instead of model.forward(x)?**  **A:** The module call manages registered hooks and normal module call behavior.
- **Q: Does torch.no_grad() put a model in eval mode?**  **A:** No. It disables recording operations for autograd within its scope, but Dropout/BatchNorm still follow the module training/eval flag.

### Slide 36 — Feeding the network: splits, standardization, mini-batches

**Instructor notes.** Partition Fashion-MNIST as 55k train, 5k validation, 10k held-out test. Fit standardization on training data only. 55,000/128 gives 429 full batches plus one partial batch, or 430 optimizer updates if `drop_last=False`.

**Boardwork / teaching move.** **Prevent leakage:** Show `train → fit scaler & parameters`, `validation → tune/checkpoint`, `test → final report`. Use the 55k train split, not val/test pixels, for preprocessing statistics. `DataLoader(drop_last=False)` can produce one final short minibatch.

**Likely student question:** Why can't I use the test set to choose a learning rate?

**Suggested answer:** Then it is no longer an unbiased final evaluation; model-selection information leaks into the reported test score.

**Further anticipated questions and prepared answers**

- **Q: Why is the test set held out even though we already have validation?**  **A:** Validation guides choices; test estimates performance after those choices are fixed.
- **Q: Can standardization use statistics from an unlabeled test set?**  **A:** For this supervised benchmark’s held-out evaluation protocol, no; fitting preprocessing on the test set leaks information. Some transductive tasks explicitly permit unlabeled test data, but they are a different protocol.

### Slide 37 — The training loop: five steps, every mini-batch

**Instructor notes.** Memorize the five mini-batch operations in order: clear, forward, loss, backward, update. Then evaluate without weight updates using `model.eval()` plus `torch.no_grad()`.

**Boardwork / teaching move.** **Chant the loop:** `zero_grad → logits=model(x) → loss → backward → step`. Draw the boundary between each minibatch. If gradients are intentionally accumulated, scale losses appropriately and call step only after the planned accumulation.

**Likely student question:** What happens if I forget to clear `.grad`?

**Suggested answer:** PyTorch adds new gradients to existing values; that may be intended gradient accumulation, but is wrong for ordinary one-batch updates.

**Further anticipated questions and prepared answers**

- **Q: Why does PyTorch accumulate gradients by default?**  **A:** It supports intentional accumulation across operations and mini-batches; users must clear or reset at the intended update boundary.
- **Q: How many optimizer steps per epoch are there?**  **A:** Normally one per minibatch: with 55,000 images, batch 128, and drop_last=False, ceil(55000/128)=430.

### Slide 38 — Running example: training curves after 20 epochs

**Instructor notes.** Compare validation *loss* against validation *accuracy*, not just training loss. Best validation loss may occur well before last epoch because confidently wrong predictions raise CE without changing argmax decisions.

**Boardwork / teaching move.** **Interpret two metrics:** Cross-entropy cares *how much* probability is given to the correct label; accuracy depends only on top-1 rank. A more confident wrong prediction can increase CE without changing accuracy. Restore the best validation checkpoint, not blindly the final epoch.

**Likely student question:** Should I always stop at epoch 5?

**Suggested answer:** No. Five is one seed's observed best epoch; choose a validation checkpoint in each run, possibly with patience or a schedule.

**Further anticipated questions and prepared answers**

- **Q: Can validation accuracy stay unchanged while validation loss gets worse?**  **A:** Yes. If the same examples remain right/wrong but the model becomes more confidently wrong, cross-entropy increases.
- **Q: Is the epoch with the best validation accuracy always the same as the best CE epoch?**  **A:** No. Different metrics select different checkpoints.
- **Q: Can repeated use of validation itself overfit?**  **A:** Yes. Extensive hyperparameter selection can overfit validation; preserve an untouched test set.

### Slide 39 — Back at 1:30

**Instructor notes.** During break pose the deep-plain-network question: worse training performance indicates optimization failure, not standard overfitting.

**Boardwork / teaching move.** **Break prompt:** Ask students to choose among optimization failure, representation failure and overfitting; ask for a *training-set* diagnostic that would distinguish them.

**Likely student question:** If deep nets can represent shallower nets, why can they fit training data worse?

**Suggested answer:** Representational containment does not guarantee the optimizer finds good parameters within a finite training budget.

**Further anticipated questions and prepared answers**

- **Q: Could more depth ever harm training optimization?**  **A:** Yes, even though the larger model is expressive; initialization, parameterization, normalization and optimization matter.

### Slide 40 — Making deep networks train and generalize

**Instructor notes.** Preview four connected diagnoses: gradient scaling, loss curvature, optimization, and generalization.

**Boardwork / teaching move.** **Reset:** Separate three axes of failure on the board: (i) expressiveness, (ii) finding good parameters, (iii) generalizing to unseen examples. The next slides mostly address (ii) and (iii).

**Likely student question:** Is overfitting the same as optimization failure?

**Suggested answer:** No. Overfitting is good training fit but weak validation performance; optimization failure means training error remains too high.

**Further anticipated questions and prepared answers**

- **Q: Which problem does dropout address?**  **A:** It is principally a generalization regularizer, not a universal remedy for failure to fit training examples.

### Slide 41 — Depth multiplies: gradients vanish or explode

**Instructor notes.** Products of layer Jacobians can shrink or grow gradients; the simple scalar powers are intuition. Log per-layer norms with care—unequal parameter counts and scaling also affect norms. NaNs can have causes other than exploding gradients.

**Boardwork / teaching move.** **Unroll a Jacobian product:** For `a[l]=g(W[l]a[l−1]+b)`, each reverse step multiplies by a transposed weight and a diagonal activation-derivative matrix. Show scalar analogues `0.5³⁰≈9×10⁻¹⁰` and `1.5³⁰≈1.9×10⁵`, but stress that matrix products depend on singular directions, not just one scalar average.

**Likely student question:** If gradient norms decrease toward the input, is that definitive proof of vanishing gradients?

**Suggested answer:** It is a clue. Compare normalization, layer size, activation statistics, losses, and whether those layers actually fail to update.

**Further anticipated questions and prepared answers**

- **Q: How is vanishing different from a zero gradient at the optimum?**  **A:** Vanishing describes an inability of early parameters to receive a useful learning signal even while loss remains high; near-optimal small gradients can be appropriate.
- **Q: Will gradient clipping cure vanishing gradients?**  **A:** No. Clipping limits overly large norms; it does not amplify small ones.
- **Q: Do NaNs prove exploding gradients?**  **A:** No. Invalid arithmetic, a faulty loss, data infinities, and numerical overflow can also cause NaNs.
- **Q: Should we compare raw layer gradient norms without considering layer size?**  **A:** Not blindly. Include relative update-to-weight norms, activation distributions, and comparable parameters or normalization.

**Accuracy caution.** Small early-layer gradient norms can have many causes; look at relative update scale and activation/weight statistics.

### Slide 42 — Initialize to keep the pre-activation variance constant

**Instructor notes.** Derive Var(z)≈n_in Var(w)E[a²] under zero-mean independent weights. A symmetric pre-ReLU variable has E[ReLU(z)²]=½E[z²]; He scaling Var(w)=2/n_in stabilizes the second-moment propagation approximately, not every realized activation variance.

**Boardwork / teaching move.** **Worked numeric derivation:** Suppose a preactivation Z has symmetric standard-normal distribution, so `E[Z²]=1`, `E[ReLU(Z)²]=½`, and `Var(ReLU(Z))=½−1/(2π)≈0.341`. For 256 independent incoming activations and independent zero-mean weights, `Var(z_next)=256·Var(w)·E[a²]`. Xavier at equal widths sets `Var(w)=1/256`, producing `Var(z_next)=½`; He sets `2/256`, producing `Var(z_next)=1`. **The exact factor-of-two argument is about the second moment E[a²], not the centered variance of post-ReLU activations.**

**Likely student question:** Why can we initialize biases at zero but not all weights?

**Suggested answer:** Zero biases do not make neurons identical when weights differ; all-identical weights yield symmetry and identical updates within a layer.

**Further anticipated questions and prepared answers**

- **Q: Why do zero-mean random weights matter in the variance calculation?**  **A:** They make E[z]=0 and remove cross terms under the independence assumptions, simplifying Var(z).
- **Q: Why can biases be zero?**  **A:** Distinct randomly initialized incoming weights break symmetry even when every bias begins at zero.
- **Q: Why does ½ appear for ReLU?**  **A:** A symmetric continuous preactivation has equal probability of positive and negative signs; ReLU discards the negative half of its squared magnitude.
- **Q: Is Var(ReLU(Z)) exactly Var(Z)/2?**  **A:** No. For Z~N(0,1), E[ReLU(Z)²]=1/2 but Var(ReLU(Z))=1/2−1/(2π)≈0.341.
- **Q: Why does fan_out matter in other initializers?**  **A:** Backward gradient stability depends on how many outputs a unit feeds; Xavier balances forward and backward fan counts in its intended setting.

**Accuracy caution.** Do **not** state that a zero-mean Gaussian’s post-ReLU variance is half its pre-ReLU variance; its *second moment* is half.

### Slide 43 — Simulation: 20 ReLU layers, four initializations

**Instructor notes.** Ask for predictions before revealing the plot: show slide 42, not slide 43, while students reason about small, Xavier, He and oversized weights. The plotted answer is visible as soon as slide 43 appears.

**Boardwork / teaching move.** Contrast expected signal-propagation scaling with one observed simulation; do not infer gradient stability or training success solely from forward activation spread.

**Likely student question:** Does He initialization guarantee a 100-layer network will train?

**Suggested answer:** No. Finite-width fluctuations, correlations, optimization, architecture, and backward conditioning still matter.

**Further anticipated questions and prepared answers**

- **Q: Why can simulated He activations drift despite the correct expectation?**  **A:** Finite widths, random correlations and a single seed create sampling fluctuations, so expected-scale arguments are not per-run guarantees.
- **Q: Would LayerNorm or residual connections change this story?**  **A:** Yes. Both change signal propagation and conditioning; the plotted initialization-only comparison excludes those mechanisms.
- **Q: Does this plot directly measure gradients?**  **A:** No. It measures forward activation spread; backward conditioning has related but distinct calculations.

**Accuracy caution.** Forward activation spread alone is not direct evidence of backward gradient magnitudes.

### Slide 44 — Quick check 5

**Instructor notes.** Correct B in the slide's equal-width, independent-initialization idealization. More precisely, ReLU halves the *second moment* when its input distribution is symmetric; an approximate RMS ratio 1/√2 is better wording than an exact standard-deviation rule.

**Boardwork / teaching move.** **Clarify the quiz:** Correct B **under the slide's symmetric-preactivation, independent-weight, equal-width approximation**. Xavier with fan_in=fan_out has `Var(w)=1/fan_in`; each *squared signal scale* shrinks by ~1/2 and the RMS scale by ~1/√2 per layer. Do not call this an exact statement about each realized centered standard deviation.

**Likely student question:** Why is the answer approximate?

**Suggested answer:** Post-ReLU activations are not zero-mean; standard deviation and root-mean-square are different quantities, and finite layers are correlated.

**Further anticipated questions and prepared answers**

- **Q: Is the first layer’s input standard deviation guaranteed to drop by exactly 1/√2?**  **A:** No. At the input-to-first-ReLU transition the mean shifts and centered variance differs; the clean recursion describes expected squared scale under the idealization.
- **Q: Would Leaky ReLU use exactly the same He factor?**  **A:** No. The gain depends on the negative slope; a common fan-in choice is 2/[(1+α²)fan_in] for squared-signal preservation.
- **Q: Does a successful tanh network automatically fail after switching to ReLU with Xavier?**  **A:** No. This is a signal-propagation prediction under specified assumptions, not a guarantee about any trained instance.

**Accuracy caution.** Treat “shrinks by √2 per layer” as an idealized RMS/typical-scale statement.

### Slide 45 — Residual connections: why deeper can still train

**Instructor notes.** Derive the block Jacobian I+J_F, yielding a direct gradient route. ResNets addressed the **degradation of training error** in deeper plain nets; the identity branch facilitates but does not mathematically guarantee optimization.

**Boardwork / teaching move.** **Derive the skip-path gradient:** If `h=x+F(x)`, then `∂h/∂x=I+J_F`; hence `∇_x L=(I+J_F)ᵀ∇_h L`. The identity term gives a direct route, but cancellation with `J_F` and products of many blocks can still complicate training. ResNet degradation concerns *training* error, not simply overfitting.

**Likely student question:** If F(x)=0, does a block lose representational capacity?

**Suggested answer:** It becomes an identity map at that setting but can learn nonzero corrections if training benefits from them.

**Further anticipated questions and prepared answers**

- **Q: Why does the Jacobian contain an identity matrix?**  **A:** Differentiating x+F(x) with respect to x gives I+J_F.
- **Q: Can residual connections guarantee a nonvanishing final gradient?**  **A:** No. Different paths can interact or cancel, and initialization/normalization/depth still matter.
- **Q: What if input and output dimensions differ?**  **A:** The skip must include a projection or dimension-matching operation; a literal identity addition requires compatible shapes.
- **Q: Is the degradation problem the same as overfitting?**  **A:** No. The original result showed worse *training* errors for deeper plain networks.

**Accuracy caution.** An identity gradient term does not guarantee healthy overall Jacobian products.

### Slide 46 — Ravines defeat plain SGD; momentum and Adam help

**Instructor notes.** Compare a rotated quadratic ravine over 40 updates with optimizer-specific tuned learning rates. Momentum averages persistent directions; Adam combines momentum and coordinate-wise historical squared-gradient scaling. The toy does not establish a universal optimizer ranking.

**Boardwork / teaching move.** **Ravine experiment:** Sketch an elongated bowl and show how plain SGD oscillates across the steep direction. Momentum damps alternating gradients while reinforcing persistent ones. Adam tracks a first moment `m` and second raw moment `v` of gradients and rescales coordinates; it does **not** compute Hessian eigenvalues. Rankings here are 40-step toy outcomes with tuned step sizes.

**Likely student question:** Does Adam estimate the Hessian curvature?

**Suggested answer:** No. Its second moment tracks squared gradients, not Hessian eigenvalues; sometimes the resulting scaling helps optimization.

**Further anticipated questions and prepared answers**

- **Q: Why is a narrow ravine troublesome?**  **A:** One learning rate must be small enough for steep curvature, causing slow movement along low-curvature directions.
- **Q: Is Adam always better than SGD with momentum?**  **A:** No. The chart is a toy with tuned rates; generalization and compute efficiency depend on architecture and training recipe.
- **Q: Does per-coordinate scaling solve every rotated ravine?**  **A:** No. Diagonal scaling is not rotation-invariant and cannot fully undo cross-coordinate curvature.
- **Q: Can momentum overshoot?**  **A:** Yes. It can amplify unstable motion if the learning rate is not retuned.

**Accuracy caution.** This 2D 40-step toy does not rank optimizers generally.

### Slide 47 — Quick check 6

**Instructor notes.** The corrected answer A is **test a smaller learning rate and examine the actual loss curve** after a particular momentum-enabled run becomes unstable. This is controlled troubleshooting, not a general claim that momentum requires lowering η, and certainly not a 10× rule.

**Boardwork / teaching move.** For constant `g`, `v_t=βv_{t−1}+g` approaches `g/(1−β)`. But on a positive-definite quadratic the conventional heavy-ball recurrence has the stability condition `0<η<2(1+β)/λ_max` for `0≤β<1`—potentially a *larger* stable η than plain gradient descent's `2/λ_max`. Curvature and gradients vary in actual deep-network training, so retune and measure.

**Likely student question:** Does setting momentum to 0.9 always require dividing learning rate by ten?

**Suggested answer:** No. The constant-gradient limit is not a stability condition; on SPD quadratics conventional momentum can allow a larger stable step. Tune empirically for the actual training run.

**Further anticipated questions and prepared answers**

- **Q: What is the limit of v_t for constant g under β=0.9?**  **A:** With v₀=0, v_t=g(1−0.9^t)/(1−0.9)→10g.
- **Q: Why might batch-size changes be less diagnostic initially?**  **A:** They change gradient noise and optimization dynamics. Holding batch size fixed while varying η better isolates the effect of the new optimizer setting.

**Accuracy caution.** No predetermined scale factor for learning-rate retuning follows from constant-gradient amplification.

### Slide 48 — Adam = momentum + per-parameter steps; use AdamW

**Instructor notes.** Explain m_t, v_t, zero-initialization bias correction, and θ update with m̂/(√v̂+ε). Adam's conventional coupled L2 penalty differs from AdamW's decoupled parameter shrinkage. Verify the deck's current PyTorch API wording (Adam's decoupled_weight_decay flag). Excluding bias/norm from decay is a common recipe, not a universal law.

**Boardwork / teaching move.** **Put Adam beside AdamW:** `m_t=β₁m_{t−1}+(1−β₁)g_t`, `v_t=β₂v_{t−1}+(1−β₂)g_t²`, corrected for zero initialization, and `θ←θ−η m̂/(√v̂+ε)`. With ordinary Adam's coupled decay, replace `g_t` by `g_t+λθ` *before* the moments. AdamW applies a separate shrink `θ←(1−ηλ)θ` and computes the moments from the unpenalized loss gradient. PyTorch Adam's `decoupled_weight_decay=True` uses AdamW semantics; `False` is the documented default.

**Likely student question:** When should I use Adam rather than AdamW?

**Suggested answer:** AdamW is a useful general default when weight decay is intended; SGD with momentum can be competitive after careful tuning.

**Further anticipated questions and prepared answers**

- **Q: What does the second moment v_t measure?**  **A:** An exponential average of **squared gradients**, not the variance after subtracting the mean and not the Hessian.
- **Q: Why are bias corrections needed?**  **A:** Zero initial moving averages would underestimate their steady-state scale at early steps; dividing by 1−β_i^t corrects the initialization bias in the stationary-gradient expectation.
- **Q: Does Adam(weight_decay=λ) now automatically mean AdamW?**  **A:** No. In PyTorch 2.14, `decoupled_weight_decay=False` is the default; set it True or use `AdamW`.
- **Q: Does AdamW exactly minimize L+λ||θ||²/2?**  **A:** No, not as the adaptive Adam update of that penalized objective. Decoupled shrinkage and coupled L2 are distinct in Adam.
- **Q: Why are biases/normalization parameters commonly excluded from decay?**  **A:** Shrinking scale/offset terms is often unnecessary or harmful in common architectures; this is a tuning convention, not a mathematical law.

**Accuracy caution.** PyTorch defaults and API statements are version-specific; see official documentation references.

### Slide 49 — Diagnose the gap first, then pick remedies

**Instructor notes.** Start diagnosis by overfitting a small batch. High training and validation error suggest model, optimization, or data issues; low train/high val suggests generalization. The listed remedies are a practical menu, not a universal rank-order—pretraining may precede new data for images.

**Boardwork / teaching move.** **Diagnostic decision tree:** (1) Can the model memorize ~100 clean examples with augmentation/dropout disabled and sufficient steps? If not, debug data/labels/loss/optimizer. (2) If training is good but val is poor, try validation hygiene and generalization measures. (3) If both are poor, tune optimization/architecture before adding dropout.

**Likely student question:** If training accuracy is 100%, should I make the model smaller?

**Suggested answer:** Not automatically. Preserve capacity and try sound validation, appropriate augmentation, data, transfer learning, and regularization first.

**Further anticipated questions and prepared answers**

- **Q: What if a model cannot overfit 100 examples?**  **A:** Verify clean labels, output–loss compatibility, gradient flow, optimizer ownership, learning rate, and disabling augmentations/regularization during the sanity test.
- **Q: What if both training and validation losses are high?**  **A:** Look first for undercapacity, poor features, optimization issues, preprocessing errors or mismatched labels rather than blindly adding dropout.
- **Q: Is smaller model always the worst option?**  **A:** No. On deployment-constrained, low-data or highly noisy tasks it can be very sensible; the deck’s order is heuristic.

**Accuracy caution.** Overfitting a clean tiny subset requires appropriate loss, sufficient training and disabled augmentation/regularization to be informative.

### Slide 50 — Weight decay and dropout

**Instructor notes.** Show AdamW-style decay θ←(1−ηλ)θ minus adaptive gradient step. Do NOT equate this update literally with minimizing L+λ‖θ‖²/2 under Adam: that equivalence holds for ordinary SGD, not AdamW. Explain inverted dropout: training activations are Bernoulli-masked and divided by 1−p.

**Boardwork / teaching move.** **Two mechanisms:** Decoupled AdamW decay shrinks parameters each step by `(1−ηλ)` separately from the adaptive gradient update; it is **not** identical to adding `(λ/2)‖θ‖²` to an Adam objective. Dropout masks activations with Bernoulli keep-probability `q=1−p` and divides survivors by `q`; conditional expectation is unchanged, but this does not mean the expected *network output* always equals the unmasked output after nonlinearities.

**Likely student question:** Does dropout scale activations during test time?

**Suggested answer:** With inverted dropout (as in PyTorch), no. They were rescaled during training, and dropout is disabled in eval mode.

**Further anticipated questions and prepared answers**

- **Q: Is AdamW weight decay equivalent to L2 loss regularization?**  **A:** No, unlike conventional SGD with a simple update. AdamW decouples shrinkage from moment-adapted gradients.
- **Q: Does inverted dropout guarantee the same expected prediction?**  **A:** It preserves the conditional expectation of an individual masked activation, not necessarily the expected downstream nonlinear network prediction.
- **Q: Should every layer use p=0.5?**  **A:** No. Dropout needs architecture- and data-specific tuning; the deck’s rates are examples, not mandates.
- **Q: Does weight decay always produce smoother input–output functions?**  **A:** Not universally. Simple norm-based sensitivity bounds exist under assumptions, while normalization and parameter symmetries complicate the relation.
- **Q: Why is dropout turned off at inference?**  **A:** The trained deterministic evaluation network uses unmasked activations; PyTorch has already scaled surviving units during training.

**Accuracy caution.** Do not conflate AdamW’s decoupled decay with Adam’s coupled L2 objective. Dropout does not preserve expected output through arbitrary nonlinear downstream operations.

### Slide 51 — Data augmentation: free data, if the label survives

**Instructor notes.** For Fashion-MNIST demonstrate horizontal flip and modest crops, explaining label-preservation assumptions and why augmentations add *variations*, not new independent labeled examples. Compute normalization statistics from the training subset used in the experiment.

**Boardwork / teaching move.** **Have students judge validity:** Flip a T-shirt horizontally (usually preserves label) versus flip a digit 6 (may not). Augmented samples reuse source images and are correlated; call them stochastic variations, not literally 55× independent observations.

**Likely student question:** Why shouldn't I augment validation images randomly?

**Suggested answer:** It changes the evaluation distribution and adds randomness. For ordinary evaluation, use deterministic preprocessing consistent with deployment.

**Further anticipated questions and prepared answers**

- **Q: Is horizontal flipping safe for every dataset?**  **A:** No. It can change the meaning of letters, digits, medical images or directions; check the task’s invariances.
- **Q: Can a crop remove the target object?**  **A:** Yes. An overly aggressive crop invalidates the original label and may hurt training.
- **Q: Should test-time augmentation never be used?**  **A:** It can be an explicitly declared evaluation/ensembling method, but it is distinct from the standard held-out evaluation shown in the deck.

**Accuracy caution.** Avoid the phrase “free independent data”; transformations do not create independent labels.

### Slide 52 — Running example: 1,000 images overfit

**Instructor notes.** Treat the three reported Fashion-MNIST seeds as limited empirical evidence. Mean best validation CE changes 0.57→0.53 with combined regularization, but a 0.4 percentage-point accuracy difference is inconclusive. The reported 55,000-vs-1,000-image difference is +8.4 points in these particular runs.

**Boardwork / teaching move.** Label statements explicitly as observed sample means, not statistically established universal effects. Compare validation loss and accuracy as different metrics, and note the lack of independent reruns.

**Likely student question:** Does this prove more data always beats dropout?

**Suggested answer:** No. It is this dataset, architecture and experiment; data quality, domain shift and cost also matter.

**Further anticipated questions and prepared answers**

- **Q: Does 0.4 percentage points establish a generalization improvement?**  **A:** Not in this three-seed comparison; the stated variability is comparable to the difference.
- **Q: Why might loss improve more clearly than accuracy?**  **A:** Cross-entropy rewards better calibrated/probabilistic predictions even when the predicted top class is unchanged.
- **Q: Can we conclude that larger datasets always win by 8.4 points?**  **A:** No. The result compares two sample sizes in one Fashion-MNIST experiment.

**Accuracy caution.** Three seeds and one task do not support a general effectiveness ranking.

### Slide 53 — Quick check 7

**Instructor notes.** The revised question supplies two facts previously missing: evaluation pipeline/splits have been checked, and a pretrained model from a related image domain is available. B is a **promising test** to run, not a uniquely optimal intervention or a guaranteed improvement.

**Boardwork / teaching move.** Ask what could cause the gap (overfitting, shift, leakage, mislabeling, preprocessing issues) and which have been checked. Compare the pretrained baseline against alternatives using the same validation protocol.

**Likely student question:** Could augmentation be better than transfer learning?

**Suggested answer:** Yes. With no suitable pretrained model, strong label-preserving augmentation may be the best first intervention.

**Further anticipated questions and prepared answers**

- **Q: Why is training 99.8% and validation 71% a generalization warning?**  **A:** The model fits observed training labels very well but predicts poorly on held-out examples; investigate data shift and the generalization gap.
- **Q: Does a pretrained model necessarily help?**  **A:** No. Negative transfer is possible if the source and target distributions or representation needs differ, even when the images look related.
- **Q: What if this is not an image problem or there is no pretrained model?**  **A:** Test label-preserving augmentation, data collection/quality improvements, regularization and simpler baselines; answer B is then not justified.
- **Q: Would more epochs fix this?**  **A:** Not automatically; with already near-perfect training accuracy, more epochs can worsen overfitting unless a schedule/regularization changes.

**Accuracy caution.** The revised question assumes a related pretrained model; transfer learning must still be evaluated rather than presumed best.

### Slide 54 — Today in three sentences

**Instructor notes.** Close the loop: nonlinear representation, probabilistically motivated loss plus backprop, and optimizer/data/generalization engineering. Use a 30-second retrieval prompt to have students say one diagnostic for a stalled model.

**Boardwork / teaching move.** **Retrieval closure:** Ask three students to finish: ‘Nonlinearity helps because…’, ‘Loss + backprop help because…’, ‘A high validation gap calls for…’. Distinguish an NLL-derived loss from the many tasks with other loss choices.

**Likely student question:** What should I remember before the coding session?

**Suggested answer:** Write the forward shapes; use logits with the matching loss; train with zero_grad→forward→loss→backward→step; validate separately.

**Accuracy caution.** Not all legitimate losses are NLLs; read the wrap-up as referring to the likelihood-based examples covered.

### Slide 55 — Appendix A

**Instructor notes.** Explain that Appendices A and B are reference material, not part of the timed lecture. Point to the exact pages students can consult after class.

**Boardwork / teaching move.** **Guide students:** These are optional reference slides, not a hidden extension of the 150-minute lesson. Tell students which pages answer the quiz and which preview Session 2.

**Likely student question:** Are normalization and double descent examinable today?

**Suggested answer:** They are previews in Appendix B; follow the course's actual assessment scope.

**Further anticipated questions and prepared answers**

- **Q: Are the appendix plots new proof of the lecture’s claims?**  **A:** No. They contain explanatory derivations, numerical simulations and forward-looking topics of different evidentiary strength.

### Slide 56 — Quick-check answers (1–4)

**Instructor notes.** Reveal the rationales behind quick checks 1–4, especially the distinct roles of output activations, weight-array shape, and connectivity in backprop.

**Boardwork / teaching move.** **Debrief conceptually:** For each quick check, ask ‘which operation fixes the conclusion?’: an output sigmoid, a weight transpose, a change in connectivity or a wrong loss input.

**Likely student question:** What common idea ties these answers together?

**Suggested answer:** Follow operations and shapes instead of memorizing slogans.

**Further anticipated questions and prepared answers**

- **Q: Why is output softmax a mistake only with the particular loss?**  **A:** Because CrossEntropyLoss already expects logits. A different explicitly probability-based objective can be designed, but it is not this API usage.

### Slide 57 — Quick-check answers (5–7)

**Instructor notes.** Debrief Q5 (He-vs-Xavier under a specific signal-propagation model), Q6 (retune unstable momentum experiments; no universal η division), and Q7 (test domain-relevant transfer learning after checks).

**Boardwork / teaching move.** Have students distinguish mathematical guarantees from experiments and propose a counterexample to each oversimplified rule.

**Likely student question:** Which issue is about representation, which about optimization, which about generalization?

**Suggested answer:** Insufficient features/capacity: representation; unstable steps/gradients: optimization; a widening train–val gap: generalization.

**Accuracy caution.** Appendix answer 7 is a conditional experimental recommendation, not a universal optimality result.

### Slide 58 — Notation used all session

**Instructor notes.** Anchor notation: single-example a vectors are columns, PyTorch batches examples by rows. Label W[l] and δ[l] dimensions before writing the four BP equations.

**Boardwork / teaching move.** **Map conventions explicitly:** `a` is column `(n,1)` in derivations, while a minibatch is a row-stacked tensor `(B,n)` in PyTorch. Include a dimension next to each variable before evaluating any expression.

**Likely student question:** Why do textbooks and PyTorch sometimes write transposes in opposite places?

**Suggested answer:** They use different conventions for how examples are arranged; both are equivalent when shapes are consistent.

**Further anticipated questions and prepared answers**

- **Q: If PyTorch uses row batches, do the backprop formulas change mathematically?**  **A:** No. They are transposed representations of the same derivatives.

### Slide 59 — Why deep? Composition multiplies what each layer can do

**Instructor notes.** Explain depth separation as *efficiency of representation*, not universal impossibility for shallow networks. Cite the supplied depth-separation studies, and qualify any exponential/polynomial claim with family and assumptions.

**Boardwork / teaching move.** **Qualification:** Depth separation results construct function families under specific width/depth/precision assumptions. They demonstrate potential efficiency advantages, not ‘deep always generalizes better’.

**Likely student question:** If width can replace depth, why is depth worthwhile?

**Suggested answer:** For some structured function families, compositional depth requires exponentially fewer units than a shallow alternative.

**Further anticipated questions and prepared answers**

- **Q: Does universal approximation contradict depth separation?**  **A:** No. Existence of a wide shallow approximation and *efficient size* of an approximation are different questions.
- **Q: Does a separation theorem predict better test accuracy?**  **A:** No. It is about representational complexity for constructed function families.

### Slide 60 — From 1943 to today: backprop was not the bottleneck

**Instructor notes.** Clarify historical timeline: reverse-mode differentiation predates the 1986 popularization of neural-network backprop; 2012-era success was also driven by data, compute and training techniques.

**Boardwork / teaching move.** **Timeline caution:** Separate discoveries of reverse-mode differentiation, its application to neural networks, and the compute/data/activation advances that made large-scale training practical.

**Likely student question:** Was backprop invented in 1986?

**Suggested answer:** No. Reverse-mode methods and neural-network uses predate 1986; 1986 papers helped popularize the approach.

**Further anticipated questions and prepared answers**

- **Q: What changed around 2012 for vision networks?**  **A:** The combination of better hardware, datasets, architectures and training techniques made large neural networks practical; no single event explains it.

### Slide 61 — When not to reach for a neural network

**Instructor notes.** Emphasize comparison against boosted trees, logistic regression and classical models on tabular/small-data problems; domain constraints matter more than an AI-first preference.

**Boardwork / teaching move.** **Model-choice exercise:** Ask for an honest baseline for 300 rows of tabular data versus millions of labeled images. Consider pretrained models for tabular data as an option to benchmark, not a universal replacement for boosted trees.

**Likely student question:** Should a neural network always be the baseline?

**Suggested answer:** No. Start with a strong simple baseline appropriate to the task, then demonstrate a real advantage before increasing complexity.

**Further anticipated questions and prepared answers**

- **Q: Are tree models always superior on tabular data?**  **A:** No. They are useful strong baselines, but representation, sample size, pretraining and task requirements affect the ranking.

### Slide 62 — Verify hand-written gradients with centred differences

**Instructor notes.** Centred finite differences compare analytical and numerical derivatives on deterministic float64 computations. Near a ReLU kink, finite differences can disagree because the function is not differentiable there; tolerances are heuristics.

**Boardwork / teaching move.** **Numerical check:** For a chosen scalar parameter θ, use `(J(θ+h)−J(θ−h))/(2h)`. Compare to the analytical derivative with a *scale-aware* relative error. Disable stochasticity; avoid nonsmooth kinks; small h can also suffer cancellation.

**Likely student question:** Why not use gradient checking for every weight on every batch?

**Suggested answer:** Each checked coordinate costs two extra forward passes, so it is for debugging small subsets.

**Further anticipated questions and prepared answers**

- **Q: Can too small a finite-difference step be worse?**  **A:** Yes. Subtracting nearly equal floating-point losses can produce roundoff error; choose a sensible step and use float64.
- **Q: Why do ReLU kinks cause disagreement?**  **A:** At the kink there is no unique classical derivative; symmetric finite differences average sides while autograd picks a subgradient convention.

**Accuracy caution.** Gradient-checking numerical tolerances depend on conditioning, nondifferentiability, step size and precision.

### Slide 63 — Session 2 preview

**Instructor notes.** Signal the transition to a future session: normalization, double descent, PyTorch loss details, silent bugs, schedulers and a modern recipe.

**Boardwork / teaching move.** **Transition:** Tell students that normalization, double descent and schedulers are advanced previews; today's fundamentals remain prerequisite for the hands-on lab.

**Likely student question:** What is the main prerequisite?

**Suggested answer:** Be able to read a forward pass, loss, gradients, and train/validation curves.

**Further anticipated questions and prepared answers**

- **Q: Is Session 2 required to understand this session?**  **A:** No. This session’s mathematical sequence is self-contained given the stated prerequisites.

### Slide 64 — Batch normalization: standardize, then rescale

**Instructor notes.** BatchNorm normalizes over batch (and spatial positions for convolutions), then learns γ and β. Training uses batch statistics and tracks running estimates; evaluation uses running estimates. Its benefits are not definitively explained by 'internal covariate shift'.

**Boardwork / teaching move.** **Trace two modes:** In training BN normalizes with the current minibatch and updates running estimates. In eval, default `track_running_stats=True` uses stored estimates. The running variance uses an unbiased mini-batch estimator in PyTorch, whereas training-time normalization uses the biased one.

**Likely student question:** Does BatchNorm use the same statistics during training and inference?

**Suggested answer:** Not usually. Standard BatchNorm updates running statistics during training and uses them in eval mode.

**Further anticipated questions and prepared answers**

- **Q: Does BatchNorm always use stored statistics during eval?**  **A:** Only when track_running_stats=True and the running buffers exist; otherwise it uses current-batch statistics in both modes.
- **Q: What is PyTorch’s BatchNorm momentum parameter doing?**  **A:** It weights the newest statistic in the running estimate, unlike the classical optimization momentum convention.
- **Q: Is “internal covariate shift” the established mechanism of BN?**  **A:** No. It was the original motivation; alternative explanations emphasize changes in optimization and smoothness, and the causal story is still debated.

**Accuracy caution.** BatchNorm default behavior assumes running statistics are enabled.

### Slide 65 — Normalization variants: which axes, which statistics

**Instructor notes.** Explain normalizing axes: BatchNorm across examples, LayerNorm within per-example features, GroupNorm within groups of channels, InstanceNorm per-example per-channel spatial dimensions. RMSNorm does not subtract a mean.

**Boardwork / teaching move.** **Axes on board:** For image `(N,C,H,W)`, BN pools over N,H,W per channel; GN pools within a channel group per example; LN and RMSNorm use their configured normalized feature dimensions per example. RMSNorm uses root-mean-square and *does not subtract a mean*.

**Likely student question:** Which normalization is natural with variable-length transformers?

**Suggested answer:** LayerNorm or RMSNorm is common because it does not depend on other examples' batch statistics.

**Further anticipated questions and prepared answers**

- **Q: Why does BatchNorm often suffer with tiny batches?**  **A:** Batch estimates become noisy or may be undefined for some tensor shapes; GroupNorm/LayerNorm can avoid dependence on batch composition.
- **Q: Does RMSNorm force features to zero mean?**  **A:** No. It rescales by RMS without mean subtraction.

### Slide 66 — Double descent: test error can fall again

**Instructor notes.** The slide's double descent is a seeded least-squares simulation with an interpolation peak, not a universal law of every neural network. Strong regularization often suppresses the peak.

**Boardwork / teaching move.** **Interpret the simulation, not a universal law:** Plot train/test error against the number of random features. Near p≈n, interpolation can amplify noise; in some overparameterized regimes a minimum-norm solution generalizes better again. Regularization may remove the peak; outcome depends on data and algorithm.

**Likely student question:** How can interpolating models generalize?

**Suggested answer:** The chosen interpolating solution and the data's structure matter; for example a minimum-norm solution can generalize despite fitting training data.

**Further anticipated questions and prepared answers**

- **Q: Is interpolation always the point where p=n?**  **A:** Not universally. Here p and n count random features/examples in a particular simulation; effective capacity varies with model and data.
- **Q: Does a larger interpolating model always improve test error?**  **A:** No. Data structure, implicit bias, noise and regularization determine the shape.
- **Q: Does double descent invalidate the need for validation?**  **A:** No. Held-out evaluation is essential precisely because simple capacity rules can fail.

**Accuracy caution.** Double descent is a phenomenon, not an inevitable shape for every dataset/model.

### Slide 67 — PyTorch loss functions and target formats

**Instructor notes.** Loss interfaces: CrossEntropyLoss expects logits and usually integer class labels, but probability targets are supported; BCEWithLogitsLoss expects float binary targets; MSELoss requires aligned shapes to avoid broadcasting.

**Boardwork / teaching move.** **API quick test:** For B=8 and K=10, logits have `(8,10)`. Class-index targets are `(8,)` int64; probability targets are `(8,10)` float and must be valid distributions even though PyTorch does not enforce that requirement.

**Likely student question:** Why should labels usually be integer class IDs for multiclass CE?

**Suggested answer:** It is compact and uses the optimized class-index path; use distribution targets when you actually need soft labels.

**Further anticipated questions and prepared answers**

- **Q: Does PyTorch check whether float CE targets sum to one?**  **A:** No. Probability targets must be checked by the user.
- **Q: Why is `(B,1)` versus `(B,)` dangerous for MSE?**  **A:** Broadcasting may compute a B×B grid of pairwise errors instead of B matched pairs.

**Accuracy caution.** Soft probability targets are supported; PyTorch does not validate that they are valid distributions.

### Slide 68 — Six training mistakes that may not raise an exception

**Instructor notes.** Walk through 'silent' bugs: output softmax, unintended accumulation, wrong eval mode, unregistered modules kept in a plain list, target broadcasting, and optimizer not owning intended parameters. Watch warnings and tests.

**Boardwork / teaching move.** **Debug in a fixed order:** Assert tensor shapes and dtypes, inspect `.grad` for key parameters, verify optimizer param IDs, compare train/eval, then inspect losses and predictions. A returned scalar loss proves very little about correctness.

**Likely student question:** How can a model produce a loss yet never learn a layer?

**Suggested answer:** If a layer is not registered or its parameters are missing from the optimizer, it can be used in forward without being updated.

**Further anticipated questions and prepared answers**

- **Q: Can a parameter be used in a forward pass but absent from the optimizer?**  **A:** Yes. Unregistered modules or optimizer created before a model replacement can have that effect.
- **Q: Is numerical gradient accumulation always wrong?**  **A:** No. It is valid when performed intentionally and normalized to the desired effective batch objective.

### Slide 69 — Step each scheduler at the right frequency

**Instructor notes.** Contrast per-batch schedules (OneCycleLR) with per-epoch schedules; a misplaced scheduler.step changes the effective learning-rate trajectory without an exception.

**Boardwork / teaching move.** **Show cadence:** OneCycleLR steps after *each optimizer step*; ReduceLROnPlateau steps after evaluating a validation metric, commonly once per epoch. Logging only once per epoch can miss a OneCycleLR peak within an epoch; inspect a per-step trace when debugging.

**Likely student question:** Can I call every scheduler.step() once per epoch?

**Suggested answer:** No. Follow the scheduler's documented stepping semantics and inspect logged rates.

**Further anticipated questions and prepared answers**

- **Q: Does OneCycleLR step once per epoch?**  **A:** No. The documented policy changes the rate after every optimizer step / training minibatch.
- **Q: When should ReduceLROnPlateau.step be called?**  **A:** After the validation metric is available, often once per epoch: scheduler.step(val_loss).
- **Q: What happens when gradient accumulation reduces optimizer steps?**  **A:** Specify `total_steps` or `steps_per_epoch` in **optimizer updates**, not in raw dataloader batches, and advance the scheduler only when the optimizer updates.

**Accuracy caution.** For schedulers, a “step” normally means an optimizer update rather than an arbitrary dataloader iteration.

### Slide 70 — A complete training recipe (illustrative settings)

**Instructor notes.** Treat the pseudocode as an illustrative recipe rather than universal defaults. Clarify OneCycleLR max_lr and batch stepping, gradient clipping between backward and update, no decay for selected parameter groups, and checkpointing on validation.

**Boardwork / teaching move.** **Make the pseudocode executable:** The deck's `OneCycleLR(pct_start=0.1)` line alone omits required `max_lr` and total-step information. Supply `max_lr` **and** either `total_steps` or `(epochs, steps_per_epoch)`; call `scheduler.step()` after `optimizer.step()` once per optimizer update. Note that `max_lr` is a *peak*: default starting LR is `max_lr/25`.

**Likely student question:** Why isn't one recipe best for all datasets?

**Suggested answer:** Optimizer, regularization, batch size and schedule interact with model architecture, data scale and compute budget.

**Further anticipated questions and prepared answers**

- **Q: Which OneCycleLR arguments are mandatory?**  **A:** `optimizer` and `max_lr`, plus either `total_steps` or both `epochs` and `steps_per_epoch`.
- **Q: If I construct AdamW with lr=3e-4 and OneCycleLR(max_lr=3e-4), does the initial LR stay 3e-4?**  **A:** No. By default OneCycleLR begins at max_lr/div_factor=3e-4/25=1.2e-5 and subsequently rises to max_lr.
- **Q: How do I avoid unexpected cycling of AdamW β₁?**  **A:** Set `cycle_momentum=False` if you want beta₁ held fixed, or document the default momentum cycling.
- **Q: Why choose the checkpoint by validation loss rather than training loss?**  **A:** Training loss rewards fitting known examples; the held-out validation criterion estimates generalization and determines the selected checkpoint.
- **Q: Is clipping at 1.0 universally necessary?**  **A:** No. It is a tunable safety measure; check actual gradient norms and optimization behavior.

**Accuracy caution.** The PowerPoint itself shows pseudocode, not a fully specified runnable OneCycleLR constructor.

### Slide 71 — References and further reading (1/2)

**Instructor notes.** Point students to the cited textbooks, PyTorch documentation and foundational representation/initialization sources.

**Boardwork / teaching move.** **Reading strategy:** Recommend one textbook exposition before jumping to primary optimization proofs; official PyTorch API pages should be the arbiter for code behavior as of the stated version.

**Likely student question:** What is a good reading order?

**Suggested answer:** Start with the neural-network chapters of Dive into Deep Learning or Understanding Deep Learning, then primary papers for specialized claims.

**Further anticipated questions and prepared answers**

- **Q: Can I use a research paper’s code defaults unchanged?**  **A:** Not safely. API versions and datasets differ; reproduce conditions and validate settings.

### Slide 72 — References and further reading (2/2)

**Instructor notes.** Connect readings to the three lecture acts: autodiff/backprop, optimization, and generalization; use official framework docs for API behavior.

**Boardwork / teaching move.** **Provenance:** Distinguish the deck's own seeded runs from published theoretical results. Do not tell students that a paper establishes the accuracy numbers on these slides unless those numbers were actually published there.

**Likely student question:** Are research-paper hyperparameters universal recommendations?

**Suggested answer:** No; they are settings demonstrated under particular experimental conditions.

---

## Reference and verification notes

Core concepts are consistent with Goodfellow, Bengio & Courville (2016), *Deep Learning*; Prince (2023), *Understanding Deep Learning*; Zhang et al. (2023), *Dive into Deep Learning*; and the papers listed on presentation slides 71–72. Specific API checks used the official PyTorch 2.14 documentation for `Adam` and `CrossEntropyLoss` (accessed 18 October 2026):

- `https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html`
- `https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`

**Reproducibility note.** The companion ZIP contains `small_figures.py`, `fmnist_experiments.py`, `calibration_check.py` and stored JSON results. Descriptions above preserve the slide authors’ experiments; this guide does not claim to have re-executed the training runs.

**Further anticipated questions and prepared answers**

- **Q: Why use official API docs as well as papers?**  **A:** Papers explain methods and results; versioned API documentation specifies current parameter behavior.


## Instructor implementation inset — slide 70 (correct OneCycleLR setup)

The following is a **minimal coherent PyTorch example** showing the actual scheduler API and call order. It assumes `model` is on `device`, `train_loader` and `val_loader` yield `(x,y)`, each `y` is an integer class index, and batches are moved to `device`. Adjust hyperparameters to the task; `max_lr=3e-4` is the **peak**, not the starting learning rate. This inset *fills in* missing arguments from the deck’s pseudocode; it is not a reproduction of the deck’s Fashion-MNIST runs.

```python
import torch
from torch import nn

EPOCHS = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
    cycle_momentum=False,  # simplify the teaching example: keep AdamW betas fixed
)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # raw logits: (B, K)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # one scheduler step per optimizer update

    model.eval()
    val_sum, val_count = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            batch_loss = loss_fn(model(x), y)
            val_sum += batch_loss.item() * x.shape[0]
            val_count += x.shape[0]
    val_loss = val_sum / val_count
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
```

**Instructor caveats:** This illustration does not implement per-parameter no-decay groups; excluding biases and normalization scale/shift from decay is a common choice in many architectures. This example presumes *one optimizer step for every dataloader batch* and does not show mixed precision or gradient accumulation. If you accumulate gradients over several batches, set the scheduler's total steps to the number of **optimizer updates** and step the scheduler only when the optimizer steps. To resume a one-cycle schedule correctly, also checkpoint/restore optimizer and scheduler state and track epoch/step. PyTorch defaults `div_factor=25`, so here the initial LR is `3e-4/25=1.2e-5`. The example selects the best epoch by mean per-example validation loss; choose a metric aligned with the task.

## Evidence and versioned references (checked for October 2026)

**Source attribution:** All slide-specific experiment/simulation numbers above are taken from the submitted deck's own speaker notes and its supplied artifacts, not independently reproduced measurements. The following published documents support **general mathematical or API** claims, not the deck's exact numerical runs. Use the version displayed on the official documentation site when teaching.

**Official PyTorch documentation**

- [CrossEntropyLoss: logits, class-index and probability targets](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (including warning that probability targets are not validated).
- [BCEWithLogitsLoss: numerically stable binary-logit loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
- [Linear and functional.linear: input/weight/output shapes](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html).
- [Autograd mechanics and why eval() is not no_grad()](https://docs.pytorch.org/docs/stable/notes/autograd).
- [Initialization: Xavier and Kaiming gain, fan_in/fan_out](https://docs.pytorch.org/docs/stable/nn.init).
- [Adam: decoupled_weight_decay=False default](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
- [AdamW: decoupled weight decay](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
- [Dropout: inverse-keep-probability training scaling](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).
- [BatchNorm1d: train/eval stats and biased/unbiased variance distinction](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
- [OneCycleLR: max_lr, total_steps or epochs + steps_per_epoch, step every batch](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html).
- [ReduceLROnPlateau: metric-driven stepping](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
- [clip_grad_norm_: clipping after backward](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).

**Peer-reviewed foundational literature / course references**

- Cybenko (1989), *Approximation by superpositions of a sigmoidal function*; Leshno et al. (1993), *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function* — universal approximation with assumptions.
- Glorot & Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*; He et al. (2015), *Delving Deep into Rectifiers* — Xavier/He initialization.
- He et al. (2016), *Deep Residual Learning for Image Recognition*; He et al. (2016), *Identity Mappings in Deep Residual Networks* — degradation and residual pathways.
- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*; Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* — Adam and AdamW.
- Ioffe & Szegedy (2015), *Batch Normalization*; Santurkar et al. (2018), *How Does Batch Normalization Help Optimization?* — mechanism and open interpretations.
- Belkin et al. (2019), *Reconciling modern machine-learning practice and the classical bias–variance trade-off*; Nakkiran et al. (2020), *Deep Double Descent* — double descent is conditional, not inevitable.
- Xavier/He, residual and optimization citations already appear in the deck's slides 71–72. For a unified student-friendly narrative: Goodfellow, Bengio & Courville (2016), *Deep Learning*; Prince (2023), *Understanding Deep Learning*; and the free *Dive into Deep Learning* textbook.
