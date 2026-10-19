# Deep Neural Networks · Session 1 — Instructor notes

Companion to **DNN_Session1_150min_v7.1.pptx** (55 lecture slides + appendices, 150 minutes).
Audience: undergraduate engineers who know classical machine learning (linear and logistic regression, trees, k-NN, train/validation/test splits) but not deep learning. PyTorch is taught in a later session; all code in this session is pseudocode.

**How each slide entry is organized**

- **Time** — the slot from the speaker notes.
- **Goal** — the one idea students must leave the slide with.
- **Say / show** — the core of what to say (the deck's SAY line, expanded).
- **Watch for** — common misconceptions and slips.
- **Likely questions** — questions students tend to ask, with answers you can give in under a minute.

**Version note.** In v7.1 the quick checks are numbered in presentation order: QC4 = ravine step size (slide 28), QC5 = w₂ = 0 (slide 37), QC6 = Xavier with ReLU (slide 46). The answer slides (57–58) follow the same numbering.

**Timing overview**

| Block | Slides | Time |
|---|---|---|
| Opening | 1–3 | 0:00–0:05 |
| Part 1 · What a network computes | 4–19 | 0:05–0:40 |
| Part 2 · How it learns | 20–41 | 0:40–1:29 |
| Break | 42 | 1:29–1:39 |
| Part 3 · Train and generalize | 43–54 | 1:39–2:19 |
| Wrap-up, exit ticket, questions | 55 | 2:19–2:30 |

If you run late, the safest places to save time are slide 16 (show the figure, skip the details), slide 30 (one sentence), and slide 52 (do weight decay, mention dropout briefly). Do not cut the worked example (33–36) or the quick checks; they are where students actually learn.

**Peer-instruction routine for every quick check:** students vote silently (fingers or clickers) → if the vote is split (roughly 30–70% correct), give 60 seconds to convince a neighbour → revote → explain. If more than ~70% are right on the first vote, explain briefly and move on.

---

## Opening

### Slide 1 · Deep neural networks: from one neuron to networks that train
**Time:** 0:00–0:02

**Goal:** Students know the three-part arc and the colour code.

**Say / show:** Three questions today: what a network computes, how it learns, how to make it train and generalize. Teal = forward signal and data; orange = gradients flowing backward, and warnings. In data plots, teal and orange are simply the two classes.

**Likely questions**
- *Is a neural network a model of the brain?* — Only loosely inspired by it. Real neurons spike, dendrites compute nonlinearly, and there is no established evidence that the brain learns by backpropagation. Treat a "neuron" here as a mathematical function.
- *How is deep learning different from the machine learning I already know?* — In classical ML you usually design features and fit a model on top. A neural network learns the features and the model together from the data, by gradient descent on one loss. That pays off most with large datasets of images, audio and text; on tabular data, boosted trees are still very strong (appendix A6).

### Slide 2 · By the end of today you can…
**Time:** 0:02–0:03

**Goal:** Six concrete, testable objectives; each quick check maps to one.

**Say / show:** Read the six objectives quickly. Point to the prerequisites line; if anyone is unsure about the chain rule or matrix–vector products, suggest a 10-minute review before next session.

**Likely questions**
- *Do we need PyTorch today?* — No. Everything today is pseudocode so the ideas are not tied to one library. The next session translates it into PyTorch.
- *What does "MLP" mean?* — Multi-layer perceptron: layers of units, each fully connected to the previous layer, with a nonlinearity between layers. Defined on slide 15.

### Slide 3 · Today's plan, and one example we will follow all session
**Time:** 0:03–0:05

**Goal:** One running example (Fashion-MNIST, 784–128–64–10 MLP) ties all three parts together.

**Say / show:** We build the network in Part 1, train it in Part 2, and diagnose and regularize it in Part 3. Every curve for it comes from a script with fixed seeds.

**Likely questions**
- *Why Fashion-MNIST and not MNIST?* — MNIST digits are too easy (a small MLP reaches about 98% test accuracy), so differences between methods are hard to see. Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017) has the same format (28 × 28 greyscale, 10 classes, 60,000 training and 10,000 test images) but is harder.
- *Why an MLP and not a convolutional network?* — An MLP needs only the ideas from today. Convolutional networks, which exploit the image's spatial structure, come in a later session and do much better on images.

---

## Part 1 · What a network computes (0:05–0:40)

### Slide 4 · Part 1 divider
**Time:** 0:05

**Say / show:** Part 1 is about representation: what a network computes. Nothing learns yet.

### Slide 5 · One neuron: a weighted sum and a threshold
**Time:** 0:05–0:08

**Goal:** One neuron computes z = wᵀx + b and answers one question: which side of a hyperplane is x on?

**Say / show:** Walk the diagram left to right. Then the geometry: w is perpendicular to the boundary z = 0; the bias shifts the boundary without rotating it; |b|/‖w‖ is the distance from the origin.

**Watch for:** Students calling this neuron "linear". The step function is nonlinear; only its *boundary* is linear. Keep the word "linear" for the boundary.

**Likely questions**
- *Why is w perpendicular to the boundary?* — Take two points x and x′ on the boundary. Both satisfy wᵀx + b = 0, so wᵀ(x − x′) = 0: w is perpendicular to every direction that lies in the boundary.
- *Why do we need the bias?* — Without b, the boundary wᵀx = 0 must pass through the origin. The bias lets it sit anywhere.
- *Is this the perceptron?* — The threshold unit is McCulloch & Pitts (1943). Rosenblatt's perceptron (1958) adds a learning rule for the weights.

### Slide 6 · Smooth activations give gradients: the sigmoid neuron
**Time:** 0:08–0:11

**Goal:** A step has zero slope, so gradient descent cannot train a stack of steps. The sigmoid is smooth, and a single sigmoid neuron is exactly logistic regression.

**Say / show:** The logit z is the log-odds; solving log(p/(1 − p)) = z for p gives the sigmoid. Its slope σ(1 − σ) is at most 1/4, at z = 0 — remember this number, it returns twice. Scaling w and b by k keeps the boundary but makes the switch k times steeper.

**Likely questions**
- *Why the sigmoid and not another S-shaped curve?* — It is exactly what you get if you assume the log-odds are linear in x. Other choices exist (probit regression uses the Gaussian CDF); they behave similarly.
- *Why is the maximum slope 1/4?* — σ′ = σ(1 − σ) is a product of two numbers that sum to 1, which is largest when both equal 1/2: 1/2 × 1/2 = 1/4.
- *The perceptron learned with a step function. How?* — With its own update rule (add or subtract the misclassified input), not with gradients. It converges in finitely many steps when the data are linearly separable (Novikoff, 1962), but it does not extend to stacked layers.
- *If I double w and b, do predictions change?* — The hard prediction (which side) does not change. The probabilities do: they become more extreme.

### Slide 7 · On raw inputs, linear models draw only straight boundaries
**Time:** 0:11–0:13

**Goal:** The motivating picture: a straight line stops at 86% on the two moons; a small network bends the boundary.

**Say / show:** Same 300 points, same optimizer, same loss. Logistic regression predicts class 1 when z > 0, so its boundary is a straight line. Ask: why can't one neuron bend?

**Watch for:** Students concluding that 100% training accuracy means the bigger model is "better". That is a Part 3 question.

**Likely questions**
- *Could logistic regression with extra features (x₁², x₁x₂, …) solve this?* — Yes. A linear model on nonlinear features draws a boundary that is no longer a single straight line in the original space (usually a curve). That is exactly where slides 9–10 go.
- *Is 100% training accuracy good?* — Not by itself. Training accuracy measures fit, not generalization. Part 3 is about the difference.
- *What optimizer trained these?* — Full-batch Adam, a variant of gradient descent introduced in Part 2; both models used the same one.

### Slide 8 · XOR: no straight line separates the classes
**Time:** 0:13–0:16

**Goal:** A short proof that no single threshold neuron computes XOR.

**Say / show:** Give the room 20 seconds to try to draw a separating line. Then the four inequalities: adding rows 2 and 3 and using b ≤ 0 gives w₁ + w₂ + b > 0, contradicting row 4.

**Likely questions**
- *Would a different activation fix it?* — Not for a single neuron. Any monotone activation thresholded at a fixed level still switches on one hyperplane. The fix is more units.
- *Why does XOR matter in practice?* — It is the simplest case of an interaction: the effect of one input depends on the value of another. Real data are full of interactions.
- *Who proved this?* — Minsky & Papert (1969) analysed the limits of single-layer perceptrons in detail; their book is often linked to a drop in neural-network research funding in the 1970s, though historians debate how large that effect was.

### Slide 9 · A better representation makes XOR easy
**Time:** 0:16–0:18

**Goal:** The data did not change; the representation did. This is the central idea of the field.

**Say / show:** Add the feature x₁x₂. In 3-D the four points are separated by a plane, and y = x₁ + x₂ − 2x₁x₂ exactly.

**Likely questions**
- *Is this what kernels do in SVMs?* — Yes. A kernel computes inner products in a feature space without building the features explicitly; a degree-2 polynomial kernel includes products like x₁x₂.
- *Then why not always use kernel methods?* — Exact kernel methods need an n × n kernel matrix (memory grows with n², training time typically with n³), which is hard for large datasets, and the kernel is chosen by hand rather than learned.
- *What does the boundary look like in the original 2-D plane?* — x₁ + x₂ − 2x₁x₂ = ½ rearranges to (x₁ − ½)(x₂ − ½) = 0: the two crossing lines x₁ = ½ and x₂ = ½ (a degenerate hyperbola). Class 1 is where exactly one coordinate exceeds ½. It is not a single straight line, which is why slide 7 says "on raw inputs".

### Slide 10 · The neural-network idea: learn the features too
**Time:** 0:18–0:20

**Goal:** Give the feature map its own parameters and train features and classifier together by gradient descent on one loss.

**Say / show:** h = g(Wx + c) is a learned representation; ŷ = wᵀh + b is a linear model on top. Three consequences: features are shaped by the loss, early features transfer across tasks, and layers compose (that is what "deep" means).

**Likely questions**
- *Is the hidden layer like PCA?* — No. PCA is linear and unsupervised; it keeps directions of high variance. A hidden layer is nonlinear and is trained to reduce the task's loss.
- *How many hidden units should I use?* — It is a hyperparameter, chosen on the validation set. More units give more capacity, and need more data or regularization.
- *Why would features transfer between tasks?* — Early layers of image networks learn generic detectors such as edges and textures, which are useful for many tasks; transferability decreases in later, more task-specific layers (Yosinski et al., 2014).

### Slide 11 · Without a nonlinearity, depth collapses to one layer
**Time:** 0:20–0:22

**Goal:** A stack of linear layers equals one linear layer.

**Say / show:** Substitute each layer into the next: W′ = W[3]W[2]W[1] and one combined bias. Ten linear layers still cannot do XOR.

**Likely questions**
- *Does the bias change this?* — No. The biases combine into a single bias b′.
- *Then why do researchers study "deep linear networks"?* — The function class is the same as one linear layer, but the training dynamics are not, so they are a tractable model for studying optimization.
- *Can a narrow middle layer help?* — It can only restrict the map: rank(W[3]W[2]W[1]) is at most the smallest factor's rank.

### Slide 12 · Quick check 1
**Time:** 0:22–0:24 · **Answer: C** (only a hyperplane)

**Explain:** The identity hidden layers collapse into one affine map z = W′x + b′. The output sigmoid is monotone, so ŷ > 0.5 exactly when z > 0: a hyperplane, whatever the width or depth.

**Distractors:** A catches "the output sigmoid adds curvature"; B (piecewise-linear) is the correct answer for ReLU networks, a useful misconception to surface now.

**Likely questions**
- *But the sigmoid is nonlinear!* — It is, but it only relabels the score; because it is monotone it cannot move where the score crosses zero. Nonlinearity helps only *between* layers.

### Slide 13 · Sigmoid and tanh saturate; ReLU does not for z > 0
**Time:** 0:24–0:27

**Goal:** Backprop multiplies one activation slope per layer; sigmoid's slope ≤ 0.25 shrinks gradients, ReLU's slope is exactly 1 when active.

**Say / show:** Left: the functions; right: their derivatives. Ten sigmoid layers contribute at most 0.25¹⁰ ≈ 10⁻⁶ from the slopes alone. ReLU: slope 1 when on, 0 when off, so a unit that is off for every input stops learning ("dying ReLU").

**Watch for:** "ReLU is linear, so it can't help." ReLU is piecewise linear; the kink at 0 is what lets networks bend boundaries.

**Likely questions**
- *What is the derivative of ReLU at exactly 0?* — Mathematically undefined. Frameworks pick a value; PyTorch uses 0. In practice inputs are almost never exactly 0, so it does not matter.
- *Is tanh still used?* — Yes, in some places, e.g. inside LSTM cells together with sigmoid gates. For hidden layers of feed-forward networks, ReLU-family activations are the default.
- *Can a dead ReLU come back to life?* — A unit whose input is negative for every example gets zero gradient for its own weights. If earlier layers change, its input can become positive again; first-layer units, whose inputs are the fixed data, normally cannot recover. Lowering the learning rate or using Leaky ReLU (appendix B8) reduces the problem.
- *Why did ReLU take over around 2010–2012?* — It trains deep networks faster than saturating activations (Glorot et al., 2011; AlexNet, Krizhevsky et al., 2012), together with GPUs, more data and better initialization.

### Slide 14 · Two ReLU units solve XOR by remapping the space
**Time:** 0:27–0:30

**Goal:** The hidden layer moves the points so that one straight line separates them.

**Say / show:** h₁ = ReLU(x₁ + x₂) is OR-like, h₂ = ReLU(x₁ + x₂ − 1) is AND-like, y = h₁ − 2h₂. Trace the four corners: 0, 1, 1, 2 − 2 = 0. In hidden space (0,1) and (1,0) land on the same point, and the line h₁ − 2h₂ = ½ separates the classes.

**Likely questions**
- *Would gradient descent find these exact weights?* — Not necessarily. Many weight settings solve XOR (e.g. ReLU(x₁ − x₂) + ReLU(x₂ − x₁) = |x₁ − x₂|). With only two hidden units, training can also get stuck; a few more units make it reliable.
- *Is y a probability?* — No, it is a raw output. Threshold it at ½, or put a sigmoid on it and train with cross-entropy.
- *Where does the −2 come from?* — y = OR − 2·AND: at (1,1) OR = 2 and AND = 1, so y = 0.

### Slide 15 · The forward pass repeats one rule per layer
**Time:** 0:30–0:32

**Goal:** Any depth is the same two lines per layer: z = Wa + b, a = g(z). For a classifier the output is a vector of logits.

**Say / show:** Layer 0 is the input (no parameters), so this network has L = 3 layers. Shape check: W[l] has one row per unit of layer l, so its shape is (n[l], n[l−1]). For a classifier, the last layer has no activation: a[L] = z[L] are the raw scores (logits); the loss turns them into probabilities.

**Likely questions**
- *Why isn't the input layer counted?* — It has no weights; it just holds x.
- *Why (n[l], n[l−1]) and not the transpose?* — With column vectors, W·a must map n[l−1] numbers to n[l] numbers. Code that stores a batch as rows computes X Wᵀ + b with the same W (slide 18).
- *What exactly is a logit?* — The raw, unnormalized score for a class. In the binary case it is the log-odds, as on slide 6.
- *Why no softmax at the output?* — The loss applies softmax (or sigmoid) internally, in a numerically stable way. Slide 25 shows what goes wrong if you apply it twice.

### Slide 16 · A ReLU network computes a patchwork of linear pieces
**Time:** 0:32–0:34

**Goal:** Inside each patch every ReLU is fixed on or off, so the network is exactly linear there; the decision boundary is a polyline that follows curves.

**Say / show:** 146 regions were found on a grid for this 2–12–12–1 network (tiny regions can be missed, so the true count is at least this). Offer playground.tensorflow.org as a home demo.

**Likely questions**
- *How many linear regions can a network have?* — The maximum grows exponentially with depth (Montúfar et al., 2014), but trained networks typically have far fewer regions than the maximum (Hanin & Rolnick, 2019).
- *Can straight pieces really fit a smooth curve?* — With enough pieces, as closely as you like on a bounded region; see universal approximation (appendix A3).

### Slide 17 · Running example: a 784–128–64–10 MLP
**Time:** 0:34–0:36

**Goal:** Read shapes top to bottom; the batch dimension (128) never changes.

**Watch for:** "Batch" has not been defined yet. Say: a batch is a group of examples processed together in one step; slide 31 explains mini-batches.

**Say / show:** 70,000 images, 10 classes; we hold out 5,000 of the 60,000 training images for validation. Flatten (128, 1, 28, 28) → (128, 784), then 128, 64, 10 units. Output: 10 logits per image.

**Likely questions**
- *Doesn't flattening throw away the image structure?* — Yes. The MLP treats pixel 1 and pixel 2 as unrelated inputs. Convolutional networks keep the 2-D structure; that is a later session, and it is one reason this MLP stops near 89%.
- *Why 128 and 64 units?* — Reasonable defaults, not magic. They are hyperparameters you would tune on validation.
- *Why do we need a validation set if we have a test set?* — To choose hyperparameters (learning rate, epochs, size) without peeking at the test set, which must stay untouched for the final, honest estimate.

### Slide 18 · In pseudocode: a stack of layers that returns logits
**Time:** 0:36–0:38

**Goal:** Count parameters; see that 92% of them are in layer 1.

**Say / show:** Each layer has n[l]·n[l−1] weights plus n[l] biases: 100,480 + 8,256 + 650 = 109,386. Batches are rows, so a layer computes X Wᵀ + b.

**Likely questions**
- *Is 109,386 parameters a lot?* — Small by modern standards: ResNet-50 has about 25.6 million; large language models have billions.
- *Why is the first layer so big?* — Every one of the 784 pixels connects to every one of 128 units. Convolutions share weights across positions, which is one reason they use far fewer parameters for images.
- *Are more parameters always better?* — No. Whether a bigger model helps depends on data and regularization (Part 3; appendix B3 on double descent).

### Slide 19 · Quick check 2
**Time:** 0:38–0:40 · **Answer: B** (128, 784)

**Explain:** One row per unit of layer 1, one column per input. Ask a pair who chose B to explain the "one row per unit" rule; A (784, 128) is the most common wrong answer.

**Likely questions**
- *But in code I've seen weights stored as (784, 128).* — Some libraries store the transpose and compute X W instead of X Wᵀ. The maths is the same; check your library's convention. PyTorch's Linear(784, 128) stores (128, 784).

---

## Part 2 · How a network learns (0:40–1:29)

### Slide 20 · Part 2 divider
**Time:** 0:40

**Say / show:** A loss turns "wrong" into a number; gradient descent decides which way and how far to move; backpropagation gets every gradient at once; then we read real training curves.

### Slide 21 · Most losses are negative log-likelihoods
**Time:** 0:40–0:43

**Goal:** Choose a noise model for the targets; the loss follows.

**Say / show:** Gaussian noise → squared error. Bernoulli label → binary cross-entropy. Categorical label → cross-entropy, which is −log of the probability of the true class. Classification losses take the raw logits z = a[L].

**Likely questions**
- *What is a likelihood?* — The probability (or density) of the observed targets under the model, viewed as a function of the parameters. Maximizing it picks the parameters that make the data most probable.
- *Why take the log?* — It turns a product over examples into a sum, avoids numerical underflow, and has the same maximizer because log is increasing.
- *Why the ½ in squared error?* — It cancels the 2 from differentiation. It only rescales the gradient, which is equivalent to changing the learning rate.
- *Why average over the batch instead of summing?* — So the gradient's size does not depend on the batch size, and the same learning rate works for different batch sizes.
- *What about absolute error?* — It is the negative log-likelihood under Laplace noise; it predicts the median rather than the mean and is less sensitive to outliers. Huber loss is a compromise between the two.

### Slide 22 · Softmax, and the first-loss sanity check
**Time:** 0:43–0:45

**Goal:** Softmax turns logits into probabilities; an untrained K-class model should start near ln K.

**Say / show:** Worked numbers: logits (2, 1, 0.1) → (0.659, 0.242, 0.099). −log 0.9 = 0.105, −log 0.5 = 0.693, −log 0.01 = 4.61. For 10 classes ln 10 = 2.30; our model started at 2.63 because He initialization gives logits of order 1, not exactly 0.

**Likely questions**
- *Why exponentials?* — They make every output positive, preserve the order of the logits, and correspond to the multi-class version of logistic regression (log-probabilities are linear in the logits up to a constant).
- *Adding the same number to all logits changes nothing?* — Correct: softmax(z + c) = softmax(z). Only differences between logits matter. Fused losses use this to subtract the largest logit for numerical stability.
- *Can we trust softmax outputs as probabilities?* — Not automatically. Modern networks are often overconfident (Guo et al., 2017). Calibration has to be measured.
- *What if my first loss is 10 or more?* — The logits differ hugely at initialization: check input scaling, the initialization, and the labels.

### Slide 23 · Cross-entropy keeps learning when MSE stalls
**Time:** 0:45–0:47

**Goal:** With MSE on a sigmoid, the gradient carries σ′(z), which vanishes exactly when the model is confidently wrong; cross-entropy cancels it.

**Say / show:** σ(z) = 0.0001 with y = 1: MSE gives ∂L/∂z ≈ −0.0001, cross-entropy gives ≈ −1. With cross-entropy, ∂L/∂z = σ(z) − y.

**Likely questions**
- *Is MSE ever used for classification?* — It can work with careful tuning (Hui & Belkin, 2021, report competitive results with squared loss), but cross-entropy is the standard choice and the one to use by default.
- *Why is cross-entropy convex for logistic regression but MSE is not?* — With a linear logit, log-loss is a convex function of w. Squared error composed with a sigmoid is not convex in w. (For deep networks neither loss is convex in the weights.)

### Slide 24 · Quick check 3
**Time:** 0:47–0:49 · **Answer: B** (remove the softmax layer; pass raw logits)

**Explain:** Do not advance until after the revote: the next slide reveals the answer. The loss applies log-softmax itself, so a softmax layer at the end of the model applies softmax twice.

### Slide 25 · Pass raw logits to the loss: softmax twice fails silently
**Time:** 0:49–0:51

**Goal:** A bug that raises no error: the loss cannot go below ln(K − 1 + e) − 1 ≈ 1.46, and the probabilities are badly miscalibrated.

**Say / show:** Same model, data and optimizer; only difference is an extra softmax. Training loss stalls near 1.63; validation accuracy 82–88% versus ≈ 89%; the buggy model reports ≈ 99% mean confidence.

**Likely questions**
- *Where does 1.46 come from?* — After the first softmax the "logits" are probabilities in [0, 1]. The best case is 1 for the true class and 0 elsewhere, giving loss −log(e / (e + K − 1)) = ln(K − 1 + e) − 1; for K = 10 that is 1.46.
- *If the loss is stuck, why is accuracy still decent?* — Softmax preserves the order of its inputs, so the predicted class (argmax) can still be right. What breaks is the size of the gradients and the meaning of the probabilities.
- *What is log-sum-exp?* — log Σⱼ e^{zⱼ}, computed as c + log Σⱼ e^{zⱼ − c} with c = max zⱼ, so no exponent overflows. Fused cross-entropy losses do this for you.
- *What is ECE (in the notes)?* — Expected calibration error: group predictions by confidence and average the gap between confidence and accuracy, weighted by group size. Zero means perfectly calibrated.

### Slide 26 · Gradient descent: the sharpest direction caps η
**Time:** 0:51–0:54

**Goal:** Step against the gradient; on a bowl, gradient descent is stable only if η < 2/λ_max.

**Say / show:** First-order Taylor: the change in loss ≈ ∇Jᵀ Δθ, most negative opposite the gradient, for small steps. On J = ½λθ², θ ← (1 − ηλ)θ converges iff |1 − ηλ| < 1, i.e. η < 2/λ. With curvatures 1 and 10 the steep direction caps η at 0.2; η = 0.21 diverges.

**Likely questions**
- *What is λ in a real network?* — The largest eigenvalue of the Hessian (the matrix of second derivatives) of the loss — the curvature in the sharpest direction. It changes during training.
- *How do I pick η in practice?* — Try values on a log scale (e.g. 10⁻⁴ to 10⁻¹) and compare validation curves, or run a learning-rate range test (Smith, 2017). Symptoms: nan in a few steps → far too large; oscillating → too large; glacial → too small.
- *Why not use the exact best step, e.g. Newton's method?* — Newton's method needs the Hessian, which has (number of parameters)² entries — about 10¹⁰ for our small 109k-parameter model — far too large to store or invert.

### Slide 27 · Ravines defeat plain SGD; momentum and Adam help
**Time:** 0:54–0:57

**Goal:** One step size cannot suit a steep and a gentle direction at once. Momentum averages out the bouncing; Adam rescales each parameter's step.

**Say / show:** Rotated ravine (curvatures 1 and 25), 40 steps, learning rates tuned per optimizer. A 2-D toy: rankings on real networks vary.

**Likely questions**
- *What does β in momentum do?* — It sets how much of the previous velocity is kept. β = 0.9 averages roughly the last 1/(1 − β) = 10 gradients; along a consistent direction, steps grow up to 10× larger than plain SGD with the same η.
- *What is Nesterov momentum?* — A variant that evaluates the gradient at the point momentum is about to move to; often slightly better, same idea.
- *So is Adam always the best?* — No. Here it won one tuned 40-step toy. On real problems the ranking depends on the task and tuning; SGD with momentum remains competitive for image classification.

### Slide 28 · Quick check 4
**Time:** 0:57–0:59 · **Answer: B** (2/25 = 0.08)

**Explain:** η < 2/λ_max; rotating the ravine does not change its curvatures. At η just below 0.08 the gentle direction shrinks only by a factor 0.92 per step: the crawl momentum and Adam address.

**Distractors:** A uses the gentle curvature; C (1/λ_max) is the step that solves the steep direction in one step, not the stability limit; D confuses "convex" with "converges for any step".

### Slide 29 · Adam = momentum + per-parameter steps; use AdamW
**Time:** 0:59–1:02

**Goal:** Adam keeps running averages of the gradient and its square, corrects them for starting at zero, and steps by their ratio. AdamW applies weight decay directly to the weights.

**Say / show:** Defaults β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸. Start with AdamW (weight decay 0.01–0.1), excluding biases and normalization scales from decay. Glossary for this audience: Transformers are the architecture behind language models; CNNs are image models — both later in the course.

**Likely questions**
- *Why the bias correction?* — m and v start at 0, so early averages are biased toward 0. Dividing by (1 − βᵗ) removes that bias; its effect fades as t grows.
- *What is ε for?* — It prevents division by zero when a parameter's gradients have been tiny.
- *What is the difference between "Adam with weight decay" and AdamW?* — With plain SGD, an L2 penalty and weight decay are the same. In Adam, an L2 penalty's gradient gets divided by √v̂, so parameters with large gradients are barely regularized. AdamW subtracts ηλθ directly, independent of the adaptive scaling (Loshchilov & Hutter, 2019).
- *Which learning rate?* — 10⁻³ is a common default for Adam; 3 × 10⁻⁴ is a common starting point for AdamW in Transformer training. Always check on validation.
- *Are there newer optimizers?* — Yes. For example, Muon has been reported to train large language models with about half the compute of AdamW (Liu et al., 2025). AdamW remains the standard default and the right one to learn first.

### Slide 30 · A network's loss is not one bowl
**Time:** 1:02–1:04

**Goal:** A network's loss is non-convex: different starts reach different valleys, and flat regions slow progress. In large networks the valleys gradient descent reaches are usually about equally good.

**Say / show:** Left: three starts, three valleys with nearly equal loss. Right: near a saddle point the loss stalls, then drops. Practical lesson: the seed sets the initial weights and batch order, so run several seeds.

**Watch for:** Taking the 1-D picture literally. In high dimensions most critical points are saddles rather than bad minima; the picture is a cartoon.

**Likely questions**
- *Should we worry about getting stuck in a bad local minimum?* — In large networks this is rarely the main problem; saddle points and flat regions are a bigger obstacle (Dauphin et al., 2014). Mini-batch noise and momentum help escape them.
- *Why are there many equivalent solutions?* — Swapping two hidden units (and their weights) gives the same function, so every solution has many copies.
- *How many seeds should I run?* — At least three for any claim, and report mean and spread. More if differences are small.
- *Do we want the global minimum?* — Not necessarily. Zero training loss can mean memorizing the training set. What matters is validation and test performance.

### Slide 31 · Feeding the network: splits, standardization, mini-batches
**Time:** 1:04–1:07

**Goal:** Train/validation/test roles; standardize with training statistics only; mini-batches.

**Say / show:** 55,000 images, batch 128 → 430 updates per epoch (the last batch has 88 images). The noise in a mini-batch gradient has a standard deviation that falls as 1/√m.

**Likely questions**
- *Why standardize the inputs?* — Features on very different scales make the loss surface badly conditioned (a ravine), and standard initializations assume inputs of roughly unit scale.
- *Why compute the mean and std on the training set only?* — Using validation or test data leaks information about them into training, which makes your estimates optimistic.
- *What batch size should I use?* — 32–512 is typical. Larger batches give less noisy gradients and use the hardware better; when you increase the batch, the learning rate usually needs to increase too (e.g. the linear scaling rule with warmup, Goyal et al., 2017).
- *Epoch vs iteration?* — An epoch is one pass over the training data; an iteration is one update (one mini-batch).

### Slide 32 · Backpropagation: all gradients for ≈ 3 forward passes
**Time:** 1:07–1:09

**Goal:** Backprop is the chain rule organized so shared products are computed once; one backward pass gives every gradient.

**Say / show:** Finite differences need about one forward pass per parameter: 10⁸ for 10⁸ parameters. Backprop costs about 3 forward-pass equivalents in total. In the diagram, the orange product ∂J/∂c · ∂c/∂b is computed once and reused for ∂J/∂w₂ and ∂J/∂w₁.

**Likely questions**
- *Why does the backward pass cost about twice the forward pass?* — For each matrix multiplication, the backward pass computes two products of similar size: the gradient with respect to the layer's input and with respect to its weights.
- *Is backprop the same as the chain rule?* — Yes, applied in reverse order on the computation graph so that intermediate products are shared (reverse-mode automatic differentiation).
- *Who invented it?* — Reverse-mode differentiation: Linnainmaa (1970); applied to neural networks: Werbos (1974); popularized by Rumelhart, Hinton & Williams (1986).
- *Does backprop update the weights?* — No. It only computes gradients; the optimizer (SGD, Adam) uses them.

### Slide 33 · Worked example: your turn, forward pass first
**Time:** 1:09–1:12 (3-minute exercise)

**Goal:** Students compute the forward pass by hand.

**Say / show:** Pairs are fine. Loss: binary cross-entropy; with y = 1 it is L = −ln a₂. Answers on the next slide: z₁ = 1.1, a₁ = 0.7503, z₂ = 0.3002, a₂ = 0.5745, L = 0.5543.

**Watch for:** Forgetting b₂ = −0.3; using log₁₀ instead of ln (gives 0.2407).

**Likely questions**
- *Why ln and not log₁₀?* — Cross-entropy uses the natural log by convention (loss in "nats"). Another base only rescales the loss and gradients by a constant.
- *How do I compute σ(0.3002) without a calculator?* — You need one; e^(−0.3002) ≈ 0.7407, so σ ≈ 1/1.7407 ≈ 0.5745.

### Slide 34 · Worked example: forward, then backward, by hand (build 2 of 4: check the forward pass)
**Time:** 1:12–1:13

**Say / show:** Check: 1.1, 0.7503, 0.3002, 0.5745, 0.5543. Ask who matched to three decimals. Keep these values: the backward pass reuses every one of them.

### Slide 35 · Worked example: forward, then backward, by hand (build 3 of 4: error signals)
**Time:** 1:13–1:16

**Goal:** Compute the error signals and see the output σ′ cancel; see δ shrink through one sigmoid layer.

**Say / show:** ∂L/∂a₂ = −1/a₂ = −1.7407. Multiply by σ′(z₂) = 0.2445: δ₂ = a₂ − y = −0.4255 — the cancellation from slide 23. Back through w₂: ∂L/∂a₁ = −0.3404. Multiply by σ′(z₁) = 0.1874: δ₁ = −0.0638, 6.7 times smaller.

**Likely questions**
- *Why are the gradients negative?* — Increasing the weights would push a₂ toward the target 1 and lower the loss. Gradient descent subtracts η × (negative number), so it increases them.
- *Why did δ shrink?* — The factor w₂ · σ′(z₁) = 0.8 × 0.187 ≈ 0.15. With σ′ ≤ 0.25 and weights not much larger than 1, every sigmoid layer shrinks the error like this; twenty layers leave almost nothing — vanishing gradients (Part 3).
- *Why define δ on z rather than on a?* — Because then the parameter gradients are one line each (next slide): weight gradient = δ × input, bias gradient = δ.

### Slide 36 · Worked example: forward, then backward, by hand (build 4 of 4: parameter gradients)
**Time:** 1:16–1:18

**Say / show:** ∂L/∂w₂ = δ₂a₁ = −0.3192, ∂L/∂b₂ = −0.4255, ∂L/∂w₁ = δ₁x = −0.1276, ∂L/∂b₁ = −0.0638. Weight gradient = error × input; bias gradient = the error itself. All values check against finite differences.

**Likely questions**
- *What happens after one update?* — With η = 0.1, for example, w₁ becomes 0.5 + 0.1 × 0.1276 ≈ 0.5128 and w₂ becomes 0.8 + 0.1 × 0.3192 ≈ 0.8319; a₂ rises and the loss falls.
- *In code, would I compute −ln a₂ myself?* — No. You would pass the logit z₂ to a fused binary-cross-entropy-with-logits loss, which is numerically stable.

### Slide 37 · Quick check 5
**Time:** 1:18–1:20 · **Answer: B** (exactly 0)

**Explain:** Layer 1's error arrives only through w₂: ∂L/∂a₁ = δ₂ · w₂ = 0, so δ₁ = 0 and ∂L/∂w₁ = δ₁x = 0. Vanishing gradients and dead units block learning by the same mechanism. The next slide's BP2 sends errors back through Wᵀ — exactly the path w₂ = 0 blocks.

### Slide 38 · The four equations of backpropagation
**Time:** 1:20–1:23

**Goal:** The same rules for any fully connected layer.

**Say / show:** BP1: output error = loss gradient ⊙ slope. BP2: send the error back through Wᵀ, gate by g′. BP3: bias gradient = δ. BP4: weight gradient = δ a[l−1]ᵀ (outer product). For softmax + cross-entropy, δ[L] = ŷ − y directly.

**Likely questions**
- *Why the transpose in BP2?* — The forward pass maps n[l] numbers to n[l+1]; the backward pass must map n[l+1] errors back to n[l], and each unit's error is the sum over the units it feeds, weighted by the same weights — that is Wᵀ.
- *Why the elementwise product ⊙?* — Activations are applied elementwise, so each unit's slope multiplies only its own error.
- *Why doesn't BP1 work for softmax?* — Softmax is not elementwise: each output depends on every logit, so its derivative is a full matrix. Combined with cross-entropy it simplifies to ŷ − y.
- *What about a batch?* — With examples as rows, the weight gradient is Δᵀ A, which sums the outer products over the batch (divide by m if the loss is a mean).

### Slide 39 · Automatic differentiation runs these equations for you
**Time:** 1:23–1:24

**Goal:** Frameworks record the forward computation and run reverse-mode differentiation; the numbers match the hand calculation (−0.1276, −0.3192).

**Say / show:** Four consequences: gradients accumulate (clear them every step); activations are stored for the backward pass (memory grows with batch size); evaluate without gradient tracking; the numbers match.

**Likely questions**
- *Is autodiff the same as numerical differentiation?* — No. It is exact up to floating-point rounding; it is neither finite differences nor symbolic algebra.
- *Why do gradients accumulate instead of being overwritten?* — By design, so you can sum gradients over several small batches before one update ("gradient accumulation") when a large batch does not fit in memory.
- *Why does training need more memory than inference?* — The forward activations must be kept until the backward pass uses them (BP4). Gradient checkpointing trades memory for recomputation.

### Slide 40 · The training loop: five steps, every mini-batch
**Time:** 1:24–1:26

**Goal:** Clear, forward, loss, backward, update. Training mode to learn; evaluation mode without gradient tracking to validate.

**Say / show:** Dropout is introduced in Part 3; for now, "training mode" just means "layers that behave differently in training are switched on".

**Likely questions**
- *Does the order of "clear gradients" matter?* — It must happen before the backward pass of the current step; putting it first is the common convention.
- *Why validate every epoch?* — To track generalization and to keep the best checkpoint (next slide).

### Slide 41 · Running example: training curves after 20 epochs
**Time:** 1:26–1:29

**Goal:** Training loss keeps falling; validation loss is best at epoch 5 and then rises, while validation accuracy still creeps up. Keep the checkpoint that is best on the metric you care about.

**Say / show:** Adam, η = 10⁻³, batch 128, He initialization. The best epoch moves with the seed (5, 15, 9), so select it on validation; do not hard-code a number of epochs.

**Likely questions**
- *How can validation loss rise while accuracy also rises?* — Cross-entropy punishes confident mistakes heavily. The model becomes more confident on the examples it gets wrong, raising the loss, while still getting a few more examples right, raising accuracy.
- *So is it overfitting?* — Mildly, yes: it is starting to fit the training set in ways that do not generalize. Part 3 diagnoses and treats it.
- *Which checkpoint should I keep?* — The one that is best on the metric you will be judged on: validation loss if you need good probabilities, validation accuracy if you only need the right class.

### Slide 42 · Break (1:29–1:39)
**Say / show:** Leave the question on screen: a 30-layer plain MLP trains worse than a 10-layer one *on the training set*. Is that overfitting?

**Answer (give at the start of Part 3):** No — training loss is worse too, so it is an optimization problem: the deeper network is harder to train. Two fixes today: initialization and residual connections.

---

## Part 3 · Making deep networks train and generalize (1:39–2:19)

### Slide 43 · Part 3 divider
**Time:** 1:39

**Say / show:** Answer the break question first (not overfitting; an optimization problem). Then the plan: keep signals and gradients at scale (initialization), make very deep networks trainable (residual connections), diagnose the gap, close the gap.

### Slide 44 · Depth multiplies: gradients vanish or explode
**Time:** 1:39–1:43

**Goal:** Unrolling BP2 gives a product of one factor per layer; products shrink or blow up geometrically.

**Say / show:** 0.9³⁰ ≈ 0.04, 0.5³⁰ ≈ 10⁻⁹, 1.5³⁰ ≈ 2 × 10⁵. Vanishing is silent (loss plateaus too high); exploding shows up as inf or nan. Diagnose by logging each layer's gradient norm.

**Likely questions**
- *Why are recurrent networks especially prone to this?* — They multiply by the same weight matrix at every time step, so the product's behaviour is governed by that one matrix raised to a high power.
- *What is gradient clipping?* — If the gradient's norm exceeds a threshold, rescale it down to the threshold before the update. It prevents single huge steps; it does not fix vanishing gradients.
- *How do I log per-layer gradient norms?* — After the backward pass, compute the norm of each layer's weight gradient and record it every few steps. Early layers orders of magnitude below late ones means vanishing.

### Slide 45 · Initialize to keep the pre-activation variance constant
**Time:** 1:43–1:48

**Goal:** Var(z) = n_in · Var(w) · E[x²]; keep it constant across layers. ReLU halves the mean square, so He initialization uses Var(w) = 2/n_in.

**Say / show:** Table: He for ReLU-family, Xavier for tanh, LeCun for SELU/linear. Never all-zero weights. Biases may start at 0.

**Likely questions**
- *Why not start all weights at zero?* — Every unit in a layer would compute the same thing and receive the same gradient, so they would stay identical forever. Randomness breaks this symmetry.
- *Normal or uniform distribution?* — Either, as long as the variance is right. A uniform U(−a, a) has variance a²/3.
- *What do libraries use by default?* — It varies. PyTorch's default for Linear and Conv layers is a Kaiming-uniform variant with Var(w) = 1/(3 n_in), smaller than He's 2/n_in; set He explicitly for deep ReLU networks trained from scratch.
- *Does this also keep the backward gradients stable?* — The same argument applied to the backward pass uses fan-out instead of fan-in; He et al. (2015) show either choice works for their networks. Xavier averages the two.

### Slide 46 · Quick check 6
**Time:** 1:48–1:50 · **Answer: B** (shrinks by about √2 per layer)

**Explain:** Xavier with n_in = n_out = 256 gives Var(w) = 1/256, half of He's 2/256. ReLU halves the mean square each layer, so the standard deviation falls by √2 per layer: by (1/√2)¹⁹ ≈ 1.4 × 10⁻³ from layer 1 to layer 20. The next slide shows it.

**Distractors:** C has the right mechanism with the wrong sign (that is what std 1.0 does); D misses that the factor compounds.

### Slide 47 · Simulation: 20 ReLU layers, four initializations
**Time:** 1:50–1:53

**Goal:** Only He keeps the signal at a stable scale.

**Say / show:** He: 0.82 at layer 1, 1.14 at layer 20. Xavier: 1.1 × 10⁻³ at layer 20 (theory ≈ 8 × 10⁻⁴). std 0.01 vanishes to 10⁻¹⁹; std 1.0 explodes to 10²¹.

**Likely questions**
- *Why does He's value drift from 0.82 to 1.14?* — Finite width: each layer multiplies by a random factor close to, but not exactly, 1, and these fluctuations accumulate. With wider layers the drift shrinks.
- *Why is layer 1 at 0.82 and not 1?* — The plot shows the standard deviation *after* ReLU. For Gaussian z with variance 2, ReLU(z) has std √(2 × (½ − 1/(2π))) ≈ 0.83; the mean square, not the std, is what He keeps constant.

### Slide 48 · Residual connections: why deeper can still train
**Time:** 1:53–1:58

**Goal:** x_{l+1} = x_l + F(x_l): a block learns a correction, and "doing nothing" is easy. This is what lets very deep networks train.

**Say / show:** Three points: deeper plain networks trained worse in He et al.'s runs and residual ones did not; F → 0 makes a block the identity; residual connections are in ResNets (images) and Transformers (language). The Jacobian I + ∂F/∂x always contains a path with factor 1.

**Likely questions**
- *What if x_l and F(x_l) have different sizes?* — Use a projection on the shortcut (a linear layer, or a 1 × 1 convolution in CNNs) to match the size.
- *Is this the same as the skip connections in U-Net?* — Related but different: U-Net concatenates earlier features; residual blocks add them.
- *Do residual connections guarantee that deep networks train?* — No guarantee, but in practice they make very deep networks trainable: ResNet-152 on ImageNet (2015), and a 1,001-layer ResNet on CIFAR-10 (He et al., 2016b).
- *Wasn't the problem vanishing gradients?* — Partly, but He et al. (2016) saw the degradation even in well-initialized networks with batch normalization, where gradients were healthy. Residual connections help in both ways: an easy identity, and a gradient path with factor 1.

### Slide 49 · Diagnose the gap first, then pick remedies
**Time:** 1:58–2:02

**Goal:** The training–validation gap tells you which problem you have; our running example is in row 2 (early overfitting).

**Say / show:** Rung 1 (keep the best checkpoint) was done on the 20-epoch curves; rung 2 (more data) is the 55× bar on slide 53; next slides take rungs 3, 4, then 5 and 6. First check: can the model overfit 100 examples? If not, suspect a bug.

**Likely questions**
- *My validation loss is lower than my training loss. Is something wrong?* — Often not: dropout and augmentation make training harder and are off at evaluation. But also check for leakage (validation examples duplicated in training) or a validation set that is simply easier.
- *Why is "smaller model" last?* — Large, well-regularized models usually do at least as well as small ones; shrinking the model is a blunt tool (appendix B3).
- *Early stopping vs keeping the best checkpoint?* — Keeping the best checkpoint is nearly free. Early stopping additionally halts training when validation stops improving; it saves time but needs a "patience" matched to the learning-rate schedule.

### Slide 50 · Data augmentation: free data, if the label survives
**Time:** 2:02–2:05

**Goal:** Transforms create new training examples, but only if the label survives. Never augment validation or test data.

**Say / show:** Go through the table: flips suit clothes but not digits or text; rotation confuses 6 and 9. Foreshadow slide 53: for our MLP, this recipe did not help.

**Likely questions**
- *Why didn't augmentation help our MLP?* — An MLP treats each pixel position as a separate input, so a shifted or flipped image is effectively a new pattern it must learn from scratch. Convolutional networks share weights across positions and usually benefit much more.
- *Isn't "test-time augmentation" a thing?* — Yes, as a deliberate prediction technique: average the model's predictions over several augmented copies of each test image. That is different from changing the evaluation data; the evaluation set itself stays fixed.
- *What are Mixup and CutMix?* — Mixup trains on weighted blends of two images and their labels; CutMix pastes a patch of one image into another and mixes labels by area. Both produce soft (probability) targets.

### Slide 51 · Transfer learning: reuse features learned on big data
**Time:** 2:05–2:08

**Goal:** Reuse a network pretrained on a large dataset; add a new output layer trained on your data; optionally fine-tune gently.

**Say / show:** Three steps: freeze and train the head; fine-tune with a small learning rate; match the original inputs (size, standardization).

**Likely questions**
- *Where do pretrained models come from?* — Model libraries such as torchvision, timm and the Hugging Face Hub publish weights trained on large datasets (e.g. ImageNet).
- *Could I use an ImageNet model on greyscale Fashion-MNIST?* — Yes: repeat the grey channel three times and resize to the expected input size. It usually helps, though the domain gap (photos vs small clothing icons) limits the gain.
- *Freeze or fine-tune?* — Little data and data similar to the original: freeze. More data, or data that differ more: fine-tune all layers with a smaller learning rate (a common rule of thumb is about 10× smaller than training from scratch).
- *Is this how large AI models are used?* — Yes, the same principle at much larger scale: pretrain once on huge data, then adapt (fine-tune) to many tasks.

### Slide 52 · Weight decay and dropout
**Time:** 2:08–2:13

**Goal:** Weight decay shrinks weights every step; dropout randomly switches units off during training and is off at test time.

**Say / show:** W ← (1 − ηλ)W − η∇J. Inverted dropout divides survivors by 1 − p so the expected activation matches test time. Typical p: 0.5 for large fully connected layers, 0.1 in many Transformer recipes, little or none in convolutional networks with normalization. Do not decay biases.

**Likely questions**
- *Is weight decay the same as L2 regularization?* — For plain SGD, yes (with λ scaled appropriately). For Adam, no — that is why AdamW exists (slide 29).
- *Why divide by (1 − p)?* — So that each unit's expected output during training equals its output at test time, when nothing is dropped.
- *Is dropout ever used at test time?* — Normally no. "Monte Carlo dropout" deliberately keeps it on and averages several predictions to estimate uncertainty (Gal & Ghahramani, 2016).
- *Why does weight decay help?* — Smaller weights generally make the function less sensitive to small input changes; it can also be read as a Gaussian prior on the weights. Its exact effect depends on the architecture and optimizer.

### Slide 53 · Running example: more data beats every trick
**Time:** 2:13–2:17

**Goal:** On 1,000 images, dropout, weight decay and augmentation all land within noise of each other; 55× more data adds 8.5 points.

**Say / show:** Test accuracy at the best-validation checkpoint, mean of 3 seeds: none 79.8%, dropout + weight decay 80.2%, crop + flip 79.6%, all three 79.3%, full 55,000 images 88.3%. Be candid: augmentation did not help this MLP.

**Likely questions**
- *Why report test accuracy here but validation accuracy earlier?* — The checkpoint is chosen on validation; the test set is then used once to report the result. Reporting validation accuracy at a checkpoint chosen on validation is slightly optimistic.
- *Is 80.2% really better than 79.8%?* — Not convincingly. With 10,000 test images the standard error of an accuracy near 80% is about 0.4 points, and seed-to-seed spread is about 0.5 points. Three seeds cannot separate them.
- *Why is "all three" the worst?* — With that much regularization the model was still improving at epoch 150 (best epochs 142–147): it is under-trained, not useless. Longer training might change the ranking.
- *Would a CNN do better on 1,000 images?* — Typically yes, because weight sharing builds in knowledge about images. That comparison comes in a later session.

### Slide 54 · Quick check 7
**Time:** 2:17–2:19 · **Answer: B** (evaluate transfer learning with the related pretrained model)

**Explain:** The checks rule out a broken pipeline and a related pretrained model exists, so transfer learning is a promising, controlled experiment — not a guaranteed winner. A train/validation gap alone cannot distinguish overfitting from data shift, leakage or label problems.

---

## Wrap-up (2:19–2:30)

### Slide 55 · Today in three sentences
**Time:** 2:19–2:30

**Say / show:** One sentence per part. Then the one-minute exit ticket: "What was the muddiest point today?" Read the cards before next session and open with the top two. Use the remaining minutes for questions.

**Homework answers (pen and paper)**
- Worked example with x = 1: z₁ = 0.6, a₁ = 0.6457, z₂ = 0.2165, a₂ = 0.5539, L = −ln a₂ = 0.5907.
- 784–256–10 parameters: 784 × 256 + 256 = 200,960; 256 × 10 + 10 = 2,570; total 203,530.
- First loss for 100 classes ≈ ln 100 = 4.61.
- BP2 for one hidden layer: δ[1] = (W[2]ᵀ δ[2]) ⊙ g′(z[1]).
- Training loop: clear gradients, forward, loss, backward, update — every mini-batch.

---

## Appendix A · Review of today

Appendix slides are not presented; use them to answer questions or as reading.

### Slide 56 · Appendix A index
A1 quick-check answers, A2 notation, A3 universal approximation, A4 why deep, A5 history, A6 when not to use a neural network, A7 gradient checking.

### Slides 57–58 · A1 Quick-check answers
1 C · 2 B · 3 B · 4 B · 5 B · 6 B · 7 B. Ask students to explain the reason, not to memorize letters. (Several answers are B by coincidence; consider reordering the options in a future version so the pattern does not tip students off.)

### Slide 59 · A2 Notation
Square brackets index layers, round brackets index examples. a[L] is the network output (logits for a classifier); ŷ is the prediction (a value for regression, probabilities for classification).

**Likely question:** *Why are derivations column vectors but code uses rows?* — Maths texts treat one example as a column; data libraries store a batch as a table with one example per row. Same W, so a layer is Wx + b for one example and X Wᵀ + b for a batch.

### Slide 60 · A3 Universal approximation, stated honestly
**Key points:** For any continuous f on a compact set and any ε > 0, a one-hidden-layer network with enough units and a non-polynomial activation (e.g. ReLU, sigmoid) is within ε of f everywhere on that set (Cybenko, 1989; Hornik, 1991; Leshno et al., 1993).

**Likely questions**
- *So one hidden layer is enough for everything?* — For *existence* on a bounded domain, yes. But the required width can be enormous, gradient descent may not find the weights, and fitting the training data is not the same as generalizing.
- *Why must the activation be non-polynomial?* — A network whose activations are polynomials of bounded degree computes only polynomials of bounded degree, which cannot approximate every continuous function.

### Slide 61 · A4 Why deep?
**Key points:** Composing a "tent" map k times with 2 ReLUs per layer gives 2ᵏ linear pieces; a shallow network needs roughly one unit per piece. Telgarsky (2016) proves functions computable with Θ(k³) layers of constant width need Ω(2ᵏ) units at depth O(k).

**Likely question:** *Does this prove deep networks are better on real data?* — No. These are worst-case constructions; the benefit of depth on real data is an empirical (well-supported) observation.

### Slide 62 · A5 History
**Likely question:** *If backprop was known in 1986, why did deep learning only take off around 2012?* — Data (ImageNet, 2009), GPUs, and training techniques (ReLU, better initialization, dropout, later normalization and residual connections) had to catch up. Most of those techniques are in today's Part 3. The 2024 Nobel Prizes went to Hopfield and Hinton (Physics) and to Baker, Hassabis and Jumper (Chemistry).

### Slide 63 · A6 When not to reach for a neural network
**Key points:** Small data, tabular data, decisions that must be audited, no baseline yet. For tabular data, boosted trees (XGBoost, LightGBM, CatBoost) are hard to beat (Grinsztajn et al., 2022); tabular foundation models such as TabPFN-3 (Prior Labs, May 2026; up to about 1 million rows × 200 features, fewer rows when there are more features) are reported by their developer to beat tuned boosted trees on many benchmarks — verify on your own data.

**Likely question:** *Should I use TabPFN instead of XGBoost?* — Try both against a simple baseline on your validation set. Check the licence: the TabPFN-3 model weights are released under a non-commercial licence.

### Slide 64 · A7 Gradient checking
**Key points:** Centred differences (J(θ + ε) − J(θ − ε))/2ε have error O(ε²), versus O(ε) for one-sided differences. Use float64, dropout off, a random subset of parameters, and a relative error: ≤ 10⁻⁷ correct; above 10⁻² probably a bug; up to about 10⁻⁴ is usually fine for losses with kinks such as ReLU.

**Likely question:** *Why not use a very small ε like 10⁻¹²?* — Floating-point rounding then dominates: subtracting two nearly equal numbers loses precision. ε around 10⁻⁵–10⁻⁶ in float64 is typical.

---

## Appendix B · Going further (later sessions)

### Slide 65 · Appendix B index
B1–B3 normalization and double descent; B4–B7 support the coding sessions; B8 activations by layer role.

### Slide 66 · B1 Batch normalization
**Key points:** Standardize each feature (channel) over the mini-batch, then apply a learned scale γ and shift β. Training uses batch statistics and updates running averages; evaluation uses the running averages.

**Likely questions**
- *Why does BatchNorm behave differently at evaluation?* — At test time you may have one example, so batch statistics are meaningless; the running averages make predictions deterministic and independent of the other examples.
- *Why does it help?* — The original "internal covariate shift" explanation has been questioned (Santurkar et al., 2018); a leading explanation is that it smooths the loss landscape, allowing larger learning rates. The question is not fully settled.

### Slide 67 · B2 Normalization variants
**Likely question:** *Why do Transformers use LayerNorm instead of BatchNorm?* — LayerNorm normalizes each example on its own, so it does not depend on batch size or on other sequences in the batch, which suits variable-length sequences and small per-device batches. Many recent language models use RMSNorm, which skips the mean subtraction.

### Slide 68 · B3 Double descent
**Key points:** Test error can follow the classical U, peak where the model can just fit the training data (parameters ≈ examples), then fall again for much larger models (Belkin et al., 2019; deep networks: Nakkiran et al., 2020). A suitable ridge penalty removes the peak here.

**Likely question:** *So bigger is always better?* — No. The slide's own caveat: with 5-dimensional inputs the classical optimum beats the largest unregularized model. It depends on data, model and regularization.

### Slide 69 · B4 Losses and target formats
**Likely question:** *Class indices or one-hot targets?* — Cross-entropy losses accept integer class indices (shape (m,)) or class probabilities (shape (m, K)); probabilities are needed for label smoothing, Mixup and CutMix. For squared error, make prediction and target shapes identical: (m, 1) vs (m,) broadcasts to (m, m) with at most a warning.

### Slide 70 · B5 Six silent training mistakes
Double softmax; unintended gradient accumulation; no evaluation mode; layers stored where the framework cannot see them; optimizer built for a different model; shape mismatch in squared error. Students should check this list before asking "why doesn't my model learn?".

### Slide 71 · B6 Learning-rate schedules
**Likely question:** *Why does it matter how often I step the scheduler?* — The schedule is defined in steps or epochs. Stepping a per-batch schedule once per epoch (or vice versa) silently stretches or compresses it, and no error is raised. Print the learning rate every epoch.

### Slide 72 · B7 A complete training recipe
**Likely questions**
- *Why warm up the learning rate?* — Early in training, gradients and Adam's second-moment estimates are unreliable; starting with a small learning rate and increasing it avoids early instability, especially for Transformers and large batches.
- *What is label smoothing?* — Replace the one-hot target by a softened one: with smoothing 0.1 and K classes, the true class gets 0.9 + 0.1/K and every other class gets 0.1/K. This discourages overconfident logits.

### Slide 73 · B8 Choose the activation by the layer's role
**Key points:** Hidden layers: ReLU by default. Classifier output: no activation (raw logits). Regression output: identity. Positive output: softplus, or predict log y. GELU = z·Φ(z) and SiLU = z·σ(z) are smooth ReLU variants used in Transformers.

**Likely question:** *Why GELU in Transformers?* — It is smooth and was found empirically to work well in models such as BERT and GPT; many recent language models use gated variants (e.g. SwiGLU, Shazeer 2020). The differences from ReLU are usually modest.

### Slides 74–75 · References
Primary sources for every claim in the deck, plus three textbooks (Goodfellow et al., 2016; Prince, 2023; Zhang et al., 2023) suitable as course reading.

---

## General questions students often ask

- *How deep is "deep"?* — There is no formal threshold; informally, more than one or two hidden layers. Today's running example (two hidden layers) is shallow by modern standards; the standard ImageNet ResNets have 18 to 152 layers.
- *How much data do I need?* — It depends on the task and model. Rules of thumb are unreliable; plot validation performance against training-set size (a learning curve) — if it is still rising, more data will help. Transfer learning lowers the requirement substantially.
- *Is deep learning always better than classical ML?* — No. It excels with large datasets of unstructured data (images, audio, text). For small or tabular datasets, regularized linear models and boosted trees are often as good or better, and easier to interpret.
- *Can I trust a network's predicted probabilities?* — Not without checking. Measure calibration on held-out data; slide 25 shows how a bug can make probabilities badly overconfident.
- *Why is training random? I got a different result from my friend.* — Random initialization, data shuffling and (on GPUs) some non-deterministic operations. Fix seeds for reproducibility and compare methods over several seeds.
