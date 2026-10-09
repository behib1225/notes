# Neural Networks & Deep Learning: A Self-Study Tutorial for Engineering Undergraduates

2026-09-18 · @Behi

### Contents

**Start here:** [0. How to use this tutorial](#mhwx88h9z18.119)

**Part I — Foundations** [1. Why we needed neural networks](#mhwx88h9z18.4725) · [2. The artificial neuron](#mhwx88h9z18.14209) · [3. Activation functions](#mhwx88h9z18.21801) · [4. The MLP forward pass](#mhwx88h9z18.30815) · [5. Loss functions](#mhwx88h9z18.40613)

**Part II — Learning** [6. Backpropagation](#mhwx88h9z18.51837) — the core section · [7. Gradient descent](#mhwx88h9z18.70124) · [8. Why deep?](#mhwx88h9z18.82424)

**Part III — Making training work** [9. Vanishing gradients and initialization](#mhwx88h9z18.90503) · [10. Optimizers](#mhwx88h9z18.100444) · [11. Normalization](#mhwx88h9z18.112093) · [12. Regularization](#mhwx88h9z18.122819)

**Part IV — PyTorch** [13. PyTorch essentials and your first project](#mhwx88h9z18.134307)

**Part V — Convolutional networks** [14. Why MLPs fail on images](#mhwx88h9z18.149007) · [15. Convolution in detail](#mhwx88h9z18.157350) · [16. Pooling, anatomy, receptive fields](#mhwx88h9z18.169358) · [17. Backprop through convolution](#mhwx88h9z18.179843) · [18. Classic architectures](#mhwx88h9z18.187814)

**Part VI — Practice** [19. CIFAR-10 project](#mhwx88h9z18.199188) · [20. Transfer learning](#mhwx88h9z18.213416) · [21. Where CNNs stand today](#mhwx88h9z18.226554) · [22. Debugging and honest evaluation](#mhwx88h9z18.236996)

**Part VII — Assessment and reference** [23. Exercises with solutions](#mhwx88h9z18.246581) · [24. Interview questions](#mhwx88h9z18.271620) · [25. FAQ](#mhwx88h9z18.291607) · [26. Glossary and further reading](#mhwx88h9z18.304762)

## 0. How to Use This Tutorial

This is a first course in deep learning, written so you can work through it alone. There is no lecture that goes with it: everything you need is here, in words, in equations, and in runnable PyTorch.

### What I assume you already know

- **Linear algebra.** Matrix-vector products, transpose, dot products. You should be comfortable with: "W is an m x n matrix, so Wx maps R^n to R^m."
- **Calculus.** Partial derivatives, the chain rule, gradients. You do *not* need matrix calculus yet — I derive what we use.
- **Probability.** Random variables, expectation, variance, independence, the Gaussian, and the idea of a likelihood.
- **Machine learning basics.** Linear and logistic regression, train/validation/test splits, overfitting, gradient descent as an idea.
- **Python.** Functions, classes, NumPy arrays. No PyTorch experience assumed.

If any of these are shaky the tutorial still works, but expect to pause and look things up.

### Notation used throughout

I keep notation consistent from section 1 to section 26. Section 26 has the full reference; here is the core.

| Symbol | Meaning |
| --- | --- |
| `x`, `y` | scalars |
| **x**, **y** | column vectors (bold lowercase) |
| **W**, **X** | matrices (bold uppercase) |
| `x⁽ⁱ⁾` | the i-th training example (round brackets = example index) |
| `W[l]`, `b[l]` | weights and biases of layer l (square brackets = layer index) |
| `z[l]` | pre-activation of layer l, before the nonlinearity |
| `a[l]` | activation (output) of layer l |
| `g`, `g'` | activation function and its derivative |
| `L` | loss on one example |
| `J` | average loss over a batch or the dataset |
| `δ[l]` | ∂L/∂z\[l\], the "error signal" at layer l |
| `⊙` | elementwise (Hadamard) product |
| `η` | learning rate |
| `m` | number of examples in a batch |
| `n[l]` | number of units in layer l |

The bracket distinction trips people up constantly, so fix it now: `a[2](7)` is the layer-2 activation for training example 7.

I write math in plain symbols rather than rendered formulas, so everything reads the same on any screen. Multi-line derivations sit in code blocks.

### The shape of the course

| Part | Sections | What you get out of it |
| --- | --- | --- |
| I. Foundations | 1-5 | Why neural nets exist; neurons, activations, the MLP forward pass, losses |
| II. Learning | 6-8 | Backpropagation derived in full; gradient descent; why depth helps |
| III. Making training work | 9-12 | Initialization, optimizers, normalization, regularization |
| IV. PyTorch | 13 | Tensors, autograd, a complete annotated MLP project |
| V. Convolutional networks | 14-18 | Why MLPs fail on images; convolution in depth; backprop through conv; LeNet to ResNet |
| VI. Practice | 19-22 | CIFAR-10 project, transfer learning, the 2026 landscape, debugging, and how to evaluate honestly |
| VII. Assessment | 23-26 | Exercises with solutions, interview questions, FAQ, glossary |

### How to actually study this

A realistic budget is 35-50 hours for full mastery. Five pieces of advice that matter more than they sound:

1. **Do not read section 6 passively.** Backpropagation is the intellectual centre of the whole course. Get a pencil and re-derive the four equations with the page covered. If you can do that, everything downstream is easier. If you cannot, everything downstream is memorization.
2. **Type the code, do not paste it.** Typing forces you to read each line. Then break it on purpose — change a shape, delete `zero_grad()`, remove the nonlinearity — and predict the error before running. You learn more from a predicted error than from a working script.
3. **Attempt every exercise before reading its solution.** The solutions in section 23 are complete on purpose, which makes them dangerously easy to read and mistake for understanding. Write something down first, even something wrong.
4. **Keep a shapes notebook.** For every model you build, write the tensor shape after every layer. Roughly half of all real deep learning bugs are shape bugs.
5. **Run things.** Everything here runs on a free Colab GPU, and most runs on a laptop CPU in minutes. Deep learning does not become real until you have watched a loss curve go down on your own screen.

**On the "Check yourself" boxes.** Every section ends with three or four questions followed by their answers in italics. The answers sit right there, which makes them easy to read before thinking. Cover them — a sheet of paper, or your hand on the screen — and commit to an answer first, even a guess. Retrieval practice is one of the few study techniques with strong evidence behind it, and it only works if you actually attempt retrieval.

**On the numbers in this document.** Every parameter count, FLOP count, worked convolution and printed output was computed and checked rather than estimated. Where a figure is approximate or depends on your hardware and seed, it says so. If you find a number that does not reproduce, trust your own run and flag it — that is the correct instinct, and it is the same one section 22 exists to build.

**What you need to run everything.** Sections 1–18 run on a laptop CPU in minutes. Section 19's CIFAR-10 project is the first thing that genuinely wants a GPU: roughly 20 minutes on a free Google Colab T4 versus several hours on CPU. Without GPU access, run that script with 15 epochs and `num_blocks=(1,1,1,1)`; you will see the same qualitative behaviour at around 88% accuracy instead of 94%.

### What you will be able to do at the end

Derive backpropagation on paper. Read and write PyTorch model and training code fluently. Diagnose why a training run is failing and decide what to try next. Design a CNN for a new image task and estimate its parameter count and receptive field before writing a line of code. Answer standard deep learning interview questions with understanding rather than recall. Read a modern architecture paper and follow the argument.

Not covered here: recurrent networks, Transformers in depth, generative models, reinforcement learning, deployment. Those are the natural next course, and section 26 says where to go.

## 1. Why We Needed Neural Networks

Neural networks exist because linear models cannot represent most interesting functions, and because hand-designing the features that would make a problem linear does not scale. A neural network is a machine that learns its own features.

### 1.1 The world your ML course showed you

In a first ML course you learn models of the form

```
prediction = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
```

Linear regression uses this directly. Logistic regression squashes it through a sigmoid to get a probability. Both are wonderful models. They are convex, so optimization is reliable. They are interpretable: w₃ = 2.4 means "one more unit of feature 3 adds 2.4 to the score." They need little data.

They also share one fatal limitation: **the decision boundary is a straight line** (a hyperplane in higher dimensions). Everything the model can express is a weighted sum of the inputs you hand it.

### 1.2 The XOR problem, worked out concretely

This is the classic demonstration, and it is worth doing by hand rather than taking on faith.

Consider four points in 2-D with binary labels:

| x₁ | x₂ | y (XOR) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Picture them on a unit square. The two points labelled 1 sit on one diagonal; the two labelled 0 sit on the other. **No single straight line separates them.** Try it: any line you draw with both 1s on one side will have at least one 0 on that side too.

Here is the proof, which takes three lines. A linear classifier predicts class 1 when `w₁x₁ + w₂x₂ + b > 0`. Suppose such weights exist:

```
(0,0) → y=0  requires  b ≤ 0
(0,1) → y=1  requires  w₂ + b > 0    so   w₂ > -b ≥ 0
(1,0) → y=1  requires  w₁ + b > 0    so   w₁ > -b ≥ 0
(1,1) → y=0  requires  w₁ + w₂ + b ≤ 0
```

From lines 2 and 3, `w₁ + b > 0` and `w₂ + b > 0`. Adding them: `w₁ + w₂ + 2b > 0`. Since `b ≤ 0`, we have `w₁ + w₂ + b ≥ w₁ + w₂ + 2b > 0`. But line 4 demanded `w₁ + w₂ + b ≤ 0`. Contradiction. No such linear classifier exists.

This is not a quirk of XOR. It is the general shape of the problem: **most real decision boundaries are curved, and linear models cannot curve.**

### 1.3 Escape route one: hand-crafted features

The classical answer was feature engineering. Add a third feature, the product `x₃ = x₁x₂`:

| x₁ | x₂ | x₁x₂ | y |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |

Now `y = x₁ + x₂ - 2x₁x₂` gives exactly 0, 1, 1, 0. A linear model in the *new* space solves the problem perfectly. Nothing about the data changed; the representation changed.

This is the key insight of the whole field, so read it twice: **a hard problem in one representation can be an easy problem in another.** Kernel methods (SVMs with an RBF kernel) automate this by working implicitly in a very high-dimensional feature space.

So why not stop there? Because feature engineering has three failure modes that get worse as problems get bigger.

**It requires you to already know the answer.** For XOR, the product feature is obvious. For "is there a cat in this photo?" it is not obvious what function of 150,528 raw pixel values (224 x 224 x 3) indicates a cat. Decades of computer vision research produced hand-designed features — SIFT, HOG, Haar wavelets — each the product of years of work by experts, each useful, none sufficient.

**It explodes combinatorially.** With n input features, all degree-2 products give about n²/2 new features; degree-3 gives about n³/6. For n = 1,000 inputs, degree-3 polynomial features number in the hundreds of millions. You cannot store the weight vector, let alone fit it.

**It does not transfer.** Features engineered for face detection are largely useless for detecting tumours in MRI scans. Every new problem starts the expert clock again.

### 1.4 Escape route two: learn the features

Here is the neural network idea in one sentence.

> Instead of designing a fixed transformation φ(x) and fitting a linear model on top, use a transformation with its own adjustable parameters and learn those parameters at the same time as the linear model on top.

Formally, a linear model is `y = wᵀx + b`. A one-hidden-layer network is

```
h = g(W x + c)          ← learned feature extractor, parameters W and c
y = wᵀh + b             ← linear model on top of the learned features
```

`g` is a nonlinear function applied elementwise. The vector `h` is a learned representation of `x`. Both the representation and the classifier are trained jointly by gradient descent on the same loss.

That single change buys three things at once:

1. The features are **task-driven**. They are shaped by the loss you actually care about, not by an expert's guess about what might matter.
2. The features are **reusable**. Train a network on millions of photographs and its early layers learn edge and texture detectors that are useful for almost any vision task. This is transfer learning, section 20.
3. The features are **composable**. Stack layers and layer 2 builds features out of layer 1's features, layer 3 out of layer 2's, and so on. Edges compose into corners, corners into motifs, motifs into object parts, parts into objects. This hierarchy is what the word *deep* refers to.

### 1.5 Solving XOR with a tiny network, by hand

You do not need training to see that it works. Take a network with two hidden units using the ReLU nonlinearity `g(z) = max(0, z)`:

```
h₁ = ReLU( x₁ + x₂ - 0.5 )      "at least one input is on"
h₂ = ReLU( x₁ + x₂ - 1.5 )      "both inputs are on"
y  = h₁ - 2·h₂
```

Trace all four inputs:

| x₁ | x₂ | x₁+x₂ | h₁ | h₂ | y = h₁ - 2h₂ | target |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | ReLU(-0.5)=0 | ReLU(-1.5)=0 | 0 | 0 |
| 0 | 1 | 1 | ReLU(0.5)=0.5 | ReLU(-0.5)=0 | 0.5 | 1 |
| 1 | 0 | 1 | ReLU(0.5)=0.5 | ReLU(-0.5)=0 | 0.5 | 1 |
| 1 | 1 | 2 | ReLU(1.5)=1.5 | ReLU(0.5)=0.5 | 0.5 | 0 |

The outputs are 0, 0.5, 0.5, **0.5** — and that last value is the problem. The point (1,1) should be class 0, but it produces the same output as the two class-1 points, so no threshold can separate them. This first attempt fails. It is worth seeing a near-miss before the fix, because the repair is instructive: the second unit needs to fire *harder* once both inputs are on, so that subtracting it can pull the output back down. Shift both thresholds:

```
h₁ = ReLU( x₁ + x₂ )            h₂ = ReLU( x₁ + x₂ - 1 )        y = h₁ - 2·h₂
```

| x₁ | x₂ | h₁ | h₂ | y | target |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 | 1 |
| 1 | 0 | 1 | 0 | 1 | 1 |
| 1 | 1 | 2 | 1 | 0 | 0 |

Exact. Two hidden ReLU units, six parameters, and the problem that defeated the perceptron is solved. Notice what the hidden units became: `h₁` is OR-like, `h₂` is AND-like, and the output computes `OR minus twice AND`, which is exactly XOR. Nobody told the network to build OR and AND detectors; this is the kind of decomposition gradient descent discovers on its own.

### 1.6 A short history, because the dates explain the field

| Year | Event | Why it mattered |
| --- | --- | --- |
| 1943 | McCulloch & Pitts threshold neuron | First mathematical model of a neuron |
| 1958 | Rosenblatt's perceptron | Learning rule with a convergence guarantee for linearly separable data |
| 1969 | Minsky & Papert, *Perceptrons* | Proved single-layer perceptrons cannot do XOR; funding collapsed |
| 1986 | Rumelhart, Hinton & Williams popularize backpropagation | Made multi-layer training practical |
| 1998 | LeCun's LeNet-5 reads bank cheques | First large commercial CNN success |
| 2006-2009 | Deep belief nets; ImageNet released (Deng et al.) | Pretraining ideas, and a dataset big enough to matter |
| 2012 | AlexNet wins ImageNet by a huge margin | GPUs + ReLU + dropout + data; the field's inflection point |
| 2015 | ResNet, 152 layers | Residual connections made very deep networks trainable |
| 2017 | Transformer architecture | Attention-based models; later spread from NLP to vision |
| 2020-2026 | Vision Transformers, ConvNeXt, large-scale self-supervised pretraining | Convolution and attention converge in practice (section 21) |

The 17-year gap between backpropagation (1986) and its dominance (2012) is instructive. The algorithm was not the bottleneck. **Data, compute, and a handful of training tricks were.** That is why sections 9-12 of this tutorial exist.

### 1.7 When *not* to use a neural network

Good engineering means knowing the limits of your tool. Reach for something else when:

- **Your data is small.** A few hundred rows with a dozen features: use regularized linear models, gradient-boosted trees, or a Gaussian process. Neural nets are data-hungry.
- **Your data is tabular.** On structured tabular data, gradient-boosted trees (XGBoost, LightGBM, CatBoost) remain highly competitive with, and often better than, neural networks. This is still an active debate in 2026, but "try GBDT first on tabular data" is sound practice.
- **You need hard interpretability.** If a regulator must audit each decision, a logistic regression or a small decision tree is defensible in a way a 50-million-parameter network is not.
- **A simple baseline already works.** Always fit the simple model first. It tells you what score is easy, which is the only way to know whether your network is actually good.

Neural networks dominate where data is abundant and has **structure you cannot easily describe**: images, audio, video, text, protein sequences. That is the territory of this course.

### Check yourself

1. Give a 2-D dataset with 3 points that a linear classifier cannot separate, or argue that none exists.
2. In section 1.5, what would happen if `g` were the identity function instead of ReLU?
3. Why did `h₁ = OR`, `h₂ = AND` emerge as a natural solution? What other pair of hidden units also solves XOR?

*(Answers: 1 — possible, but only in degenerate configurations. If the 3 points are in general position (not collinear, all distinct) then every 2-class labelling of them is linearly separable, so you need 4 points — that is XOR. Drop general position and 3 suffice: put A, B, A at x = 0, 1, 2 on a line, and no threshold separates the two A's from the B between them. Duplicate points with conflicting labels are a second, trivial case. The lesson: separability claims almost always carry an unstated general-position assumption. 2 — the composition of linear maps is linear, so the network collapses to a linear model and XOR becomes impossible again; this is proved in section 3.1. 3 — XOR = OR AND NOT(AND), and the network can express "OR minus 2·AND"; another valid pair is `ReLU(x₁-x₂)` and `ReLU(x₂-x₁)` summed, which computes |x₁-x₂|.)*

## 2. The Artificial Neuron

An artificial neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a nonlinear function. That is the whole thing:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b        (pre-activation, a scalar)
a = g(z)                                           (activation, a scalar)
```

Everything in this course is built from this and nothing else.

### 2.1 The biological story, and how far to trust it

A biological neuron receives electrical signals through dendrites from other neurons. Each synapse has a strength. The cell body integrates incoming signals, and if the accumulated potential exceeds a threshold, the neuron fires a spike down its axon to other neurons. Learning involves synaptic strengths changing with use.

The analogy maps neatly onto the equations: inputs `xᵢ` are incoming signals, weights `wᵢ` are synaptic strengths, the sum is integration at the cell body, the bias `b` sets the threshold, and `g` is the firing decision.

**Now discard most of that.** The analogy is historically important and pedagogically useful for about ten minutes, and then it starts misleading you. Real neurons communicate in discrete timed spikes, not continuous real values. Real dendrites perform nonlinear computation of their own. Real brains have no backpropagation (how the brain assigns credit is an open question). Real synapses are not signed real numbers. Cortical circuitry has structure that no standard network mimics.

A better mental model: **a neuron is a learned linear feature detector followed by a soft threshold.** Judge networks by their mathematics, not their metaphor. Interviewers sometimes ask "how is a neural network like a brain?" and the strongest answer is: superficially, and the resemblance stopped being the point in the 1980s.

### 2.2 The geometry of wᵀx + b

This subsection pays for itself many times over. Understand it and decision boundaries stop being mysterious.

The set of points where `wᵀx + b = 0` is a **hyperplane**: a line in 2-D, a plane in 3-D, an (n-1)-dimensional flat surface in n-D.

Three facts, each with its reason:

**1. w is perpendicular to the hyperplane.** Take two points `x` and `x'` both on the boundary. Then `wᵀx + b = 0` and `wᵀx' + b = 0`. Subtract: `wᵀ(x - x') = 0`. The vector `x - x'` lies along the hyperplane, and its dot product with `w` is zero, so `w` is orthogonal to every direction within the hyperplane. `w` points in the direction of steepest increase of the score.

**2. The bias shifts the hyperplane; it does not rotate it.** The perpendicular distance from the origin to the hyperplane is `|b| / ||w||`. With `b = 0` the boundary passes through the origin. Removing biases from your network therefore forces every boundary through the origin, which is a real and often crippling restriction.

**3. The distance from any point to the boundary is `(wᵀx + b) / ||w||`.** So the pre-activation `z` is a *signed, scaled distance* from the decision boundary. Large positive `z` means far on the positive side; `z` near zero means near the boundary, where the model is unsure. This is why `|z|` behaves like a confidence measure.

One consequence worth internalizing: the direction of `w` determines the *orientation* of the boundary, and the magnitude `||w||` determines how *sharply* the activation transitions across it. Doubling both `w` and `b` leaves the boundary in exactly the same place but makes a sigmoid twice as steep across it. Weight decay (section 12) exploits this: shrinking `||w||` makes the model less confident and smoother, without moving its boundaries.

### 2.3 The perceptron (1958)

Rosenblatt's perceptron used a hard step activation:

```
g(z) = 1 if z ≥ 0, else 0
```

Its learning rule is beautifully simple. For each misclassified example `(x, y)`:

```
w ← w + η (y - ŷ) x
b ← b + η (y - ŷ)
```

If the true label is 1 and you predicted 0, then `(y - ŷ) = 1` and you add `ηx` to `w`, which increases `wᵀx` and pushes the prediction toward 1. If you predicted 1 when the truth was 0, you subtract. Intuitive and correct.

The **perceptron convergence theorem** (Novikoff, 1962) says: if the data is linearly separable, this rule finds a separating hyperplane in a finite number of updates. If the data is not linearly separable, it never terminates and oscillates forever.

Two things killed the perceptron as a general-purpose tool, and both are worth understanding because they shaped what came next:

- It only handles linearly separable data, as section 1.2 showed.
- **The step function has zero derivative everywhere it is defined, and is undefined at 0.** So gradient-based learning is impossible. You cannot stack perceptrons and train the stack, because there is no gradient signal to send backwards. This is *the* reason modern networks use smooth activations.

### 2.4 Logistic regression is a single neuron

Swap the step for the sigmoid (also called the logistic function):

```
σ(z) = 1 / (1 + e^(-z))
```

and you have logistic regression. It maps any real number into (0, 1), so the output reads as a probability: `P(y = 1 | x) = σ(wᵀx + b)`.

Why this particular squashing function? It is not arbitrary. Suppose the log-odds of the two classes is linear in x — which is exactly the assumption of logistic regression:

```
log( P(y=1|x) / P(y=0|x) ) = wᵀx + b = z
```

Solve for P(y=1|x). Let p = P(y=1|x), so P(y=0|x) = 1-p:

```
log( p / (1-p) ) = z
p / (1-p)        = e^z
p                = e^z (1-p)
p (1 + e^z)      = e^z
p                = e^z / (1 + e^z) = 1 / (1 + e^(-z)) = σ(z)
```

The sigmoid *is* the inverse of the log-odds function. It is what you get when you assume the log-odds are linear, not a convenient squashing function someone picked. Worth remembering for interviews.

The derivative has a famously convenient form:

```
σ'(z) = σ(z)(1 - σ(z))
```

Derivation: write σ(z) = (1 + e^(-z))⁻¹. By the chain rule,

```
σ'(z) = -(1 + e^(-z))⁻² · (-e^(-z)) = e^(-z) / (1 + e^(-z))²
      = [1/(1 + e^(-z))] · [e^(-z)/(1 + e^(-z))]
      = σ(z) · (1 - σ(z))
```

using `e^(-z)/(1 + e^(-z)) = 1 - 1/(1 + e^(-z))`. Once you have computed the forward pass you get the derivative for free from the output — no need to keep `z` around. This mattered enormously when memory was scarce, and it is still a nice property.

### 2.5 So what does a single neuron actually compute?

A useful way to hold it: a neuron asks one yes/no question about its input, and the weights define the question.

A neuron with weights `w = [0.5, 0.5, -1]` and bias `0` is asking "is `0.5x₁ + 0.5x₂` bigger than `x₃`?" The activation is a soft answer to that question — near 1 for a confident yes, near 0 for a confident no, near 0.5 for uncertainty.

A layer of neurons asks many such questions in parallel, and the next layer asks questions about the *answers*. That is the whole architecture. Sections 4 and 8 develop the consequences.

### Check yourself

1. A neuron has `w = [3, -4]`, `b = 10`. What is the distance from the origin to its decision boundary, and which side is the origin on?
2. Show that a perceptron with a step activation and one with activation `g(z) = 1 if z ≥ 5 else 0` are equivalent models. What changes?
3. If you multiply all weights and the bias of a sigmoid neuron by 100, what happens to its outputs? Why is this relevant to weight initialization?

*(Answers: 1 — ||w|| = 5, distance = |b|/||w|| = 10/5 = 2. At the origin z = b = 10 > 0, so the origin is on the positive side. 2 — Identical, with the bias absorbed: `z ≥ 5` is the same as `z - 5 ≥ 0`, so the second model is the first with b reduced by 5. The bias is exactly a learnable threshold. 3 — The sigmoid saturates; outputs become nearly 0 or 1 with almost nothing in between, so σ'(z) ≈ 0 and gradients vanish. This is precisely why initializing weights too large stalls training — section 9.)*

## 3. Activation Functions

Activation functions are not a detail. Without one, a deep network is mathematically identical to a single linear layer, no matter how many layers you stack. Here is the proof, then the menu of choices and how to pick.

### 3.1 Why nonlinearity is mandatory (proof)

Suppose a 3-layer network with no activation function — just linear maps:

```
a[1] = W[1] x  + b[1]
a[2] = W[2] a[1] + b[2]
a[3] = W[3] a[2] + b[3]
```

Substitute upward:

```
a[2] = W[2] (W[1] x + b[1]) + b[2]
     = (W[2] W[1]) x + (W[2] b[1] + b[2])

a[3] = W[3] [ (W[2] W[1]) x + (W[2] b[1] + b[2]) ] + b[3]
     = (W[3] W[2] W[1]) x + (W[3] W[2] b[1] + W[3] b[2] + b[3])
```

Define `W_eff = W[3]W[2]W[1]` and `b_eff = W[3]W[2]b[1] + W[3]b[2] + b[3]`. Then

```
a[3] = W_eff x + b_eff
```

A single linear layer. **Depth without nonlinearity buys nothing.** There is even a loss: the product of three matrices is constrained in rank (rank of a product is at most the minimum rank of the factors), so a narrow middle layer can only make the effective map *worse* than a single unconstrained layer.

This is the single most-asked conceptual interview question in the field. The answer is not "because nonlinearity adds power" — it is the composition argument above, and you should be able to write it in thirty seconds.

### 3.2 The menu

#### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))          range (0, 1)
σ'(z) = σ(z)(1 - σ(z))           max value 0.25 at z = 0
```

Smooth, bounded, interpretable as a probability. Three serious problems:

- **Saturation kills gradients.** For |z| > 5, σ'(z) < 0.007. Backpropagation multiplies these derivatives layer by layer, so with sigmoid activations and 10 layers the gradient can be scaled by 0.25¹⁰ ≈ 10⁻⁶ even in the *best* case. This is the vanishing gradient problem (section 9).
- **Outputs are not zero-centred.** All activations are positive, which means all gradients with respect to the weights of the next layer share the same sign for a given example. Updates zigzag rather than moving diagonally. Slow.
- **exp() is relatively expensive.**

Use it today only for **binary classification output layers** and for gates in LSTMs/GRUs. Never in hidden layers of a deep feedforward network.

#### Tanh

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))     range (-1, 1)
tanh'(z) = 1 - tanh²(z)                        max value 1 at z = 0
```

A rescaled sigmoid: `tanh(z) = 2σ(2z) - 1`. Zero-centred, which fixes the second sigmoid problem, and its max derivative is 1 rather than 0.25, which softens the first. It still saturates at the tails. Reasonable in shallow nets and still standard in some recurrent architectures; rarely chosen for new deep feedforward or convolutional models.

#### ReLU (Rectified Linear Unit) — the default

```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, 0 if z < 0, undefined at z = 0 (frameworks use 0)
```

This is the workhorse. It looks almost too simple to work, and there are good reasons it does:

- **No saturation for positive inputs.** The derivative is exactly 1 whenever the unit is active, so gradients pass through undiminished. This is the single biggest reason deep networks became trainable around 2012.
- **Cheap.** One comparison. No exponentials.
- **Sparse activations.** Typically 50% or more of units output exactly zero for a given input, which gives an efficient, disentangled representation and acts as mild regularization.
- **It is still nonlinear**, despite being piecewise linear. A network of ReLUs carves input space into polyhedral regions and fits a different linear function on each. With enough units that approximates any continuous function.

The cost is the **dying ReLU** problem. If a large gradient update pushes a unit's weights so that `z < 0` for every input in the dataset, the unit outputs 0 forever, its gradient is 0 forever, and it never recovers. It is dead weight — sometimes literally 20-40% of a layer after a bad learning-rate choice. The fixes are a lower learning rate, better initialization (section 9), or a leaky variant.

#### Leaky ReLU and PReLU

```
LeakyReLU(z) = z if z > 0, else αz        (α ≈ 0.01, fixed)
PReLU(z)     = z if z > 0, else αz        (α learned per channel)
```

A small nonzero slope on the negative side means the gradient is never exactly zero, so units can come back from the dead. In practice the gains over plain ReLU are usually small but real, and Leaky ReLU is the standard choice in GAN discriminators.

#### ELU, GELU, SiLU/Swish, Mish

```
ELU(z)  = z if z > 0, else α(e^z - 1)
SiLU(z) = z · σ(z)                                  (also called Swish)
GELU(z) = z · Φ(z)   where Φ is the standard normal CDF
          ≈ 0.5z (1 + tanh[ √(2/π)(z + 0.044715 z³) ])
```

These are smooth approximations to ReLU. GELU can be read as "multiply the input by the probability that a standard normal is below it" — a soft, stochastic-looking gate rather than a hard one. GELU is the standard activation in Transformers (BERT, GPT-family, Vision Transformers), and SiLU appears in EfficientNet and in many modern detection models.

For a first CNN, ReLU and GELU will perform within noise of one another. Do not agonize.

#### Softmax (an output-layer function, not a hidden activation)

```
softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
```

Unlike everything above, softmax is **not elementwise** — each output depends on all inputs. It converts a vector of K real scores into a probability distribution over K classes. It belongs in section 5, with the losses.

### 3.3 Comparison table

| Function | Range | Derivative max | Zero-centred | Saturates | Typical use in 2026 |
| --- | --- | --- | --- | --- | --- |
| Sigmoid | (0, 1) | 0.25 | No | Both tails | Binary output; LSTM gates |
| Tanh | (-1, 1) | 1.0 | Yes | Both tails | RNN hidden states |
| ReLU | \[0, ∞) | 1.0 | No | Negative side (hard) | Default for CNNs and MLPs |
| Leaky ReLU | (-∞, ∞) | 1.0 | Nearly | No | GANs; when ReLUs die |
| ELU | (-α, ∞) | 1.0 | Nearly | Negative tail | Occasional alternative |
| GELU | ≈(-0.17, ∞) | ≈1.1 | Nearly | No | Transformers, ViTs |
| SiLU/Swish | ≈(-0.28, ∞) | ≈1.1 | Nearly | No | EfficientNet, detectors |
| Softmax | (0,1), sums to 1 | — | — | — | Multi-class output layer |

### 3.4 A decision rule you can actually follow

1. **Hidden layers:** start with ReLU. If you suspect dying units (check what fraction of activations are exactly zero), switch to Leaky ReLU or GELU.
2. **Transformer blocks:** GELU, because that is what the architecture was tuned with.
3. **Binary classification output:** sigmoid (or no activation at all plus `BCEWithLogitsLoss` — see section 5.6, this is the version you should actually use).
4. **Multi-class output:** softmax (or no activation plus `CrossEntropyLoss`, again the version you should use).
5. **Regression output:** no activation. A bounded activation on a regression output caps what the model can predict, which is almost always a bug.
6. **Output must be positive** (a price, a count): ReLU or softplus on the output, or predict `log(y)` and exponentiate.

### 3.5 What these look like in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

# Two equivalent styles: module (stateful, goes in nn.Sequential) ...
relu = nn.ReLU()
print(relu(z))            # tensor([0.0000, 0.0000, 0.0000, 0.5000, 2.0000])

# ... and functional (stateless, used inside forward())
print(F.relu(z))          # same
print(torch.sigmoid(z))   # tensor([0.1192, 0.3775, 0.5000, 0.6225, 0.8808])
print(torch.tanh(z))      # tensor([-0.9640, -0.4621, 0.0000, 0.4621, 0.9640])
print(F.gelu(z))          # tensor([-0.0455, -0.1543, 0.0000, 0.3457, 1.9545])
print(F.leaky_relu(z, negative_slope=0.01))
                          # tensor([-0.0200, -0.0050, 0.0000, 0.5000, 2.0000])
```

Both styles are used constantly in real code. Modules are needed when you build with `nn.Sequential`; functional calls are common inside a hand-written `forward()`. They compute the same thing.

### Check yourself

1. Prove that `tanh(z) = 2σ(2z) - 1`.
2. A 20-layer network uses sigmoid everywhere. Estimate an upper bound on the magnitude of the gradient reaching layer 1, relative to the gradient at the output.
3. Why is ReLU's non-differentiability at z = 0 not a practical problem?
4. You train a ReLU network and find that 60% of one layer's units output zero for *every* example in the validation set. Name two plausible causes and a fix for each.

*(Answers: 1 — 2σ(2z) - 1 = 2/(1+e^(-2z)) - 1 = (2 - 1 - e^(-2z))/(1+e^(-2z)) = (1 - e^(-2z))/(1 + e^(-2z)); multiply top and bottom by e^z to get (e^z - e^(-z))/(e^z + e^(-z)) = tanh(z). 2 — each layer contributes a factor of at most 0.25, so the bound is 0.25²⁰ ≈ 10⁻¹², i.e. essentially nothing reaches layer 1. 3 — the probability of landing exactly on z = 0 in floating point is negligible, and frameworks simply define the subgradient there as 0; the function is continuous, and optimization on piecewise-differentiable functions is well-behaved in practice. 4 — dying ReLU from too high a learning rate, fixed by lowering it or adding warmup; or poor initialization pushing pre-activations negative, fixed by He initialization (section 9) or switching to Leaky ReLU. A third possibility: an overly aggressive negative bias learned because of an unnormalized input distribution — fix by standardizing inputs.)*

## 4. The Multi-Layer Perceptron: The Forward Pass

An MLP is a stack of fully connected layers, each followed by a nonlinearity. "Fully connected" means every unit in a layer receives input from every unit in the previous layer. This section defines it precisely, gets the shapes right, and states honestly what the universal approximation theorem does and does not promise.

### 4.1 Architecture and notation

A network with L layers. Layer 0 is the input (not counted as a layer, since it has no parameters). Layer L is the output.

```
n[0] = input dimension
n[l] = number of units in layer l
W[l] : matrix of shape (n[l], n[l-1])
b[l] : vector of shape (n[l], 1)
```

The forward pass for a single example `x`:

```
a[0] = x

for l = 1 ... L:
    z[l] = W[l] a[l-1] + b[l]        (pre-activation, shape n[l] x 1)
    a[l] = g[l]( z[l] )              (activation,     shape n[l] x 1)

ŷ = a[L]
```

That is the entire forward pass. Everything else in this course is either about choosing `g`, choosing how to measure the error in `ŷ`, or computing derivatives of that error.

**Why W\[l\] has shape (n\[l\], n\[l-1\]).** Row `i` of `W[l]` holds the weights of unit `i` in layer `l`. That row has one entry per unit in layer `l-1`, so its length is `n[l-1]`. There are `n[l]` such rows. The matrix-vector product `W[l] a[l-1]` then has shape `(n[l], n[l-1]) x (n[l-1], 1) = (n[l], 1)`. Correct.

When shapes confuse you, this is the recovery procedure: write out what one row means, then count.

### 4.2 Batching: the form you will actually write

In practice you never process one example at a time. You stack `m` examples into a matrix. Two conventions exist, and mixing them causes endless bugs.

**Math-textbook convention** (examples as columns):

```
A[0] = X of shape (n[0], m)      each column is one example
Z[l] = W[l] A[l-1] + b[l]        shape (n[l], m); b broadcasts across columns
A[l] = g( Z[l] )
```

**Framework convention** (examples as rows) — this is what PyTorch uses:

```
A[0] = X of shape (m, n[0])      each row is one example
Z[l] = A[l-1] W[l]ᵀ + b[l]       shape (m, n[l]); b broadcasts across rows
A[l] = g( Z[l] )
```

PyTorch's `nn.Linear(in_features, out_features)` stores `weight` with shape `(out_features, in_features)` — the math convention — but the *data* flows as rows, so `nn.Linear` internally computes `x @ W.T + b`. Knowing this resolves most "why is my matmul failing" moments.

I use the math convention for derivations (it is cleaner) and the row convention in all code (it is what you will type). I flag the switch each time.

### 4.3 A worked example with real shapes

A classifier on 28x28 grayscale images, 10 classes, hidden layers of 128 and 64 units, batch size 32. Every tensor:

| Stage | Shape (row convention) | Note |
| --- | --- | --- |
| Input images | (32, 1, 28, 28) | batch, channels, height, width |
| After flatten | (32, 784) | 1 x 28 x 28 = 784 |
| W\[1\] | (128, 784) | `nn.Linear(784, 128)` |
| Z\[1\] = X W\[1\]ᵀ + b\[1\] | (32, 128) |  |
| A\[1\] = ReLU(Z\[1\]) | (32, 128) |  |
| W\[2\] | (64, 128) |  |
| Z\[2\], A\[2\] | (32, 64) |  |
| W\[3\] | (10, 64) |  |
| Z\[3\] = logits | (32, 10) | no activation here — see section 5.6 |
| Softmax(Z\[3\]) | (32, 10) | rows sum to 1 |

**The batch dimension never changes.** 32 goes in, 32 comes out. If your batch dimension changes size mid-network you have made an error, almost always a wrong `reshape` or `view`.

### 4.4 Counting parameters

Layer `l` has `n[l] x n[l-1]` weights and `n[l]` biases. Total:

```
params = Σ over l of ( n[l]·n[l-1] + n[l] )
```

For the network above:

```
Layer 1: 784·128 + 128 = 100,352 + 128 = 100,480
Layer 2: 128·64  +  64 =   8,192 +  64 =   8,256
Layer 3:  64·10  +  10 =     640 +  10 =     650
                                  Total = 109,386
```

Check this in PyTorch with one line, and get in the habit of doing so:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Notice that layer 1 holds 92% of the parameters, purely because the input is large and fully connected to 128 units. This observation is the seed of the entire CNN argument in section 14.

### 4.5 The universal approximation theorem, stated honestly

You will hear that "a neural network can approximate any function." That is a garbled version of a real theorem. Here is the real one.

**Theorem (Cybenko 1989; Hornik 1991, informal statement).** Let `f` be a continuous function on a compact subset `K` of R^n. For any ε > 0, there exists a feedforward network with a single hidden layer, a finite number of units, and a non-polynomial activation function, whose output `F` satisfies `|F(x) - f(x)| < ε` for all `x` in `K`.

Now the four qualifications that matter, each of which people drop:

1. **Continuous, on a compact (closed and bounded) set.** Not "any function." Discontinuous functions and unbounded domains are excluded.
2. **A finite number of units — but the theorem says nothing about how many.** The required width can be exponential in the input dimension. A theorem guaranteeing you need at most 2^100 units is not an engineering result.
3. **Existence, not findability.** The theorem says suitable weights *exist*. It offers no guarantee that gradient descent will find them, or that you could afford to compute them if you knew where they were.
4. **Approximation, not generalization.** Fitting the training data arbitrarily well is exactly what overfitting looks like. The theorem is silent on test performance, which is the thing you actually care about.

So what is it good for? It settles one question and no others: **expressive power is not the bottleneck.** A network with enough capacity can represent what you need. Therefore, when your model performs badly, look at optimization (sections 9-11), regularization (section 12), data, or architecture — not at some supposed representational ceiling.

There is also a dual result, the **width-bounded** version (Lu et al., 2017): networks of bounded *width* but unbounded depth are also universal. Depth and width trade off, which is the subject of section 8.

### 4.6 Designing an MLP: practical starting points

- **Input layer size** is fixed by your data.
- **Output layer size** is fixed by your task: 1 unit for regression or binary classification, K units for K-class classification.
- **Number of hidden layers:** start with 2. Going from 1 to 2 usually helps; going from 2 to 10 rarely helps for tabular data and often hurts without residual connections.
- **Hidden layer width:** start somewhere between the input and output size, in powers of two (128, 256, 512) because they map well to GPU memory. A common pattern is a funnel: 512 → 256 → 128 → 10.
- **Golden rule:** build the smallest network that can overfit a small subset of your training data, confirm it does (section 22), then add regularization and scale up. A model that cannot overfit 100 examples has a bug, not a capacity problem.

### 4.7 The forward pass in PyTorch, three ways

```python
import torch
import torch.nn as nn

# Way 1: nn.Sequential — concise, fine for simple stacks
model = nn.Sequential(
    nn.Flatten(),             # (32,1,28,28) -> (32,784)
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),        # logits; no softmax here (section 5.6)
)

# Way 2: subclass nn.Module — what you will use for anything non-trivial
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=(128, 64), n_classes=10):
        super().__init__()                      # never forget this line
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)                      # logits

# Way 3: by hand, so you can see there is no magic
def manual_forward(x, params):
    W1, b1, W2, b2, W3, b3 = params
    x  = x.flatten(1)                 # (m, 784)
    z1 = x  @ W1.T + b1               # (m, 128)
    a1 = torch.relu(z1)
    z2 = a1 @ W2.T + b2               # (m, 64)
    a2 = torch.relu(z2)
    z3 = a2 @ W3.T + b3               # (m, 10)
    return z3

model = MLP()
x = torch.randn(32, 1, 28, 28)
logits = model(x)                     # calls forward(); never call model.forward(x)
print(logits.shape)                   # torch.Size([32, 10])
print(sum(p.numel() for p in model.parameters()))   # 109386
```

Three points that cause real bugs:

- **Call `model(x)`, not `model.forward(x)`.** The former runs registered hooks; the latter silently skips them, which breaks many debugging and quantization tools.
- **`super().__init__()` is mandatory.** Omit it and PyTorch cannot register your parameters, so `model.parameters()` returns nothing and the optimizer updates nothing. The model trains, reports a loss, and never improves. A classic silent failure.
- **Assign submodules to `self`.** A layer stored in a plain Python list is not registered. Use `nn.ModuleList` or `nn.Sequential` for collections of layers.

### Check yourself

1. An MLP has layer sizes 100 → 50 → 50 → 3. How many parameters? How many if you removed all biases?
2. You pass a batch of shape (64, 20) into `nn.Linear(20, 30)`. What is `layer.weight.shape`, and what is the output shape?
3. Your model's loss never changes from its initial value across 50 epochs and the optimizer reports zero parameter groups worth of change. What is the first thing you check?
4. Why is it a mistake to end a regression network with a sigmoid?

*(Answers: 1 — (100·50+50) + (50·50+50) + (50·3+3) = 5,050 + 2,550 + 153 = 7,753. Without biases: 5,000 + 2,500 + 150 = 7,650. 2 — weight is (30, 20); output is (64, 30). 3 — whether `super().__init__()` is present and whether layers are assigned to `self` rather than held in a plain list; then whether the optimizer was given `model.parameters()`. 4 — sigmoid bounds the output to (0,1), so the model cannot express targets outside that range, and gradients vanish for targets near the boundaries. Use no output activation, and standardize your targets instead.)*

## 5. Loss Functions: Telling the Network It Is Wrong

A loss function turns "the prediction and the target disagree" into a single number to minimize. The loss is not an arbitrary choice — for most tasks it follows from a probabilistic assumption about the data, and understanding that derivation explains both why cross-entropy works and why MSE for classification does not.

### 5.1 The general setup

For a single example: `L(ŷ, y)`. Over a dataset or batch of m examples:

```
J = (1/m) Σᵢ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

We average rather than sum so the gradient magnitude does not depend on batch size, which means you can change batch size without re-tuning the learning rate from scratch.

A good loss is **differentiable** (so gradients exist), **aligned with the real goal** (lower loss should mean a better model), and **well-conditioned** (its gradients should not vanish precisely when the model is badly wrong).

That last property is the one people forget, and it does most of the work in this section.

### 5.2 Mean squared error, for regression

```
L = (1/2)(ŷ - y)²
```

The 1/2 is a convenience: it cancels when you differentiate, giving `dL/dŷ = (ŷ - y)`. PyTorch's `nn.MSELoss` omits the 1/2 and gives `2(ŷ - y)/m`; the difference is absorbed into the learning rate and does not matter.

**Where MSE comes from.** Assume the target is the model output plus Gaussian noise: `y = f(x) + ε` with `ε ~ N(0, σ²)`. Then

```
P(y | x) = (1 / √(2πσ²)) · exp( -(y - f(x))² / (2σ²) )
```

The log-likelihood over m independent examples:

```
log P(data) = Σᵢ [ -(y⁽ⁱ⁾ - f(x⁽ⁱ⁾))² / (2σ²) - (1/2)log(2πσ²) ]
```

Maximizing this over the model parameters means minimizing `Σᵢ (y⁽ⁱ⁾ - f(x⁽ⁱ⁾))²`, since everything else is constant. **MSE is maximum likelihood under Gaussian noise.** That is the whole justification, and it tells you when MSE is the wrong choice: when your noise is not Gaussian, for instance when it is heavy-tailed (use Huber loss) or when your target is a count (use Poisson loss).

**Related regression losses:**

| Loss | Formula | When to use |
| --- | --- | --- |
| MSE / L2 | (ŷ-y)² | Default; penalizes large errors heavily |
| MAE / L1 | \|ŷ-y\| | Robust to outliers; gradient magnitude constant |
| Huber / Smooth L1 | quadratic near 0, linear far away | Best of both; standard in object detection |

Huber loss with threshold δ:

```
L = 0.5(ŷ-y)²              if |ŷ-y| ≤ δ
L = δ(|ŷ-y| - 0.5δ)        otherwise
```

It is quadratic near the minimum (so it converges cleanly) and linear far away (so a single wild outlier cannot dominate the gradient).

### 5.3 Binary cross-entropy, for two-class problems

The model outputs `ŷ = σ(z)` in (0,1), interpreted as `P(y=1|x)`.

```
L = -[ y log ŷ + (1-y) log(1-ŷ) ]
```

Read it as a switch. Because y is 0 or 1, exactly one term survives:

```
if y = 1:  L = -log ŷ          → 0 when ŷ→1, → ∞ when ŷ→0
if y = 0:  L = -log(1-ŷ)       → 0 when ŷ→0, → ∞ when ŷ→1
```

So the loss is `-log(probability assigned to the correct class)`. Confidently wrong is punished without limit. That unbounded penalty is a feature: it gives a huge gradient exactly when the model needs correcting most.

**Where it comes from.** Each label is a Bernoulli draw with parameter ŷ:

```
P(y | x) = ŷ^y (1-ŷ)^(1-y)
```

Check: y=1 gives ŷ; y=0 gives 1-ŷ. Correct. Take the log and negate:

```
-log P(y|x) = -[ y log ŷ + (1-y) log(1-ŷ) ]
```

**Binary cross-entropy is maximum likelihood under a Bernoulli model.**

### 5.4 Softmax and categorical cross-entropy, for K classes

The output layer produces K raw scores called **logits**, `z = [z₁, ..., z_K]`. Softmax converts them to a distribution:

```
ŷₖ = e^(zₖ) / Σⱼ e^(zⱼ)
```

Every output is positive and they sum to 1. Two properties worth knowing:

- **Shift invariance.** `softmax(z + c) = softmax(z)` for any constant c, because `e^(zₖ+c)/Σe^(zⱼ+c) = e^c·e^(zₖ)/(e^c·Σe^(zⱼ))`. The logits are only defined up to an additive constant. This is what makes the numerically stable implementation in 5.7 valid.
- **Temperature.** `softmax(z/T)` with T > 1 flattens the distribution and with T < 1 sharpens it. As T → 0 it approaches a one-hot argmax. Used in knowledge distillation and in sampling from language models.

With one-hot targets, categorical cross-entropy is

```
L = -Σₖ yₖ log ŷₖ = -log ŷ_c        where c is the true class index
```

The sum collapses because only the true class has `yₖ = 1`. So again: **the loss is minus the log probability assigned to the correct answer.** Nothing more.

If `ŷ_c = 0.9`, L = 0.105. If `ŷ_c = 0.5`, L = 0.693. If `ŷ_c = 0.01`, L = 4.6.

**Sanity check you should memorize:** a randomly initialized K-class classifier should give roughly uniform probabilities, so the initial loss should be about `log(K)`. For 10 classes that is ln(10) ≈ 2.303. If your first-epoch loss starts at 15, something is wrong before training even begins — usually the labels, the output size, or an accidental extra softmax. This is the cheapest bug detector in deep learning.

### 5.5 Why not MSE for classification? (the important one)

There are three reasons, and the gradient argument is the one interviewers want.

**Reason 1: vanishing gradients when badly wrong.** Consider binary classification with `ŷ = σ(z)` and MSE loss `L = (1/2)(σ(z) - y)²`. Differentiate with respect to the logit z:

```
dL/dz = (σ(z) - y) · σ'(z) = (σ(z) - y) · σ(z)(1 - σ(z))
```

Now suppose the true label is y = 1 but the model confidently predicts σ(z) = 0.0001 (so z ≈ -9). Then:

```
(σ(z) - y)      = -0.9999          large error, good
σ(z)(1 - σ(z))  ≈ 0.0001           tiny
dL/dz           ≈ -0.0001          almost no gradient
```

The model is as wrong as it is possible to be, and the gradient is nearly zero. Learning stalls.

Compare cross-entropy. With `L = -[y log σ(z) + (1-y) log(1-σ(z))]`, differentiate with respect to z. Take y = 1:

```
L      = -log σ(z)
dL/dz  = -(1/σ(z)) · σ'(z)
       = -(1/σ(z)) · σ(z)(1-σ(z))
       = -(1 - σ(z))
       = σ(z) - y                    (with y = 1)
```

The same result holds for y = 0 (do it — it is exercise 23.M3). So:

```
dL/dz = ŷ - y
```

**The gradient is exactly the prediction error.** The sigmoid's derivative cancelled against the log's derivative. When the model is confidently wrong, `ŷ - y ≈ -1`, a full-strength gradient. When it is right, the gradient is near zero. This is exactly the behaviour you want, and it is not a coincidence — it is what happens when you pair a loss with its matching link function.

**Reason 2: MSE is not the right likelihood.** Class labels are not Gaussian-noisy real numbers. Modelling them as such is simply the wrong probabilistic assumption.

**Reason 3: non-convexity.** Even for a single neuron, MSE composed with a sigmoid is non-convex in the weights and has flat regions; cross-entropy with sigmoid is convex. For deep networks everything is non-convex anyway, but the local geometry near saturation is still much friendlier with cross-entropy.

### 5.6 Logits, and the numerical-stability rule

Here is a rule to follow without exception in PyTorch:

> **Do not put a sigmoid or softmax at the end of your model. Pass raw logits to `BCEWithLogitsLoss` or `CrossEntropyLoss`.**

The reason is floating-point arithmetic. Suppose a logit of 1000. Then `e^1000` overflows to `inf` in float32, softmax returns `nan`, and your loss becomes `nan` forever. Or a probability underflows to exactly 0.0, and `log(0) = -inf`.

The combined functions avoid this using the **log-sum-exp trick**. Because softmax is shift-invariant, subtract the max logit before exponentiating:

```
log softmax(z)ₖ = zₖ - log Σⱼ e^(zⱼ)
                = zₖ - [ max(z) + log Σⱼ e^(zⱼ - max(z)) ]
```

Every exponent `zⱼ - max(z)` is now ≤ 0, so `e^(...)` is in (0, 1\] and cannot overflow. At least one term equals exactly 1, so the sum cannot underflow to zero either. Mathematically identical, numerically safe.

`nn.CrossEntropyLoss` in PyTorch **includes log-softmax internally.** Applying softmax yourself and then passing the result to `CrossEntropyLoss` applies it twice. The model still trains — poorly, with damped gradients and mysteriously low accuracy — which makes this one of the most common and hardest-to-spot beginner bugs. Check your model's final layer right now if you have written one.

### 5.7 PyTorch losses: the table you will reference constantly

| Task | Loss | Model output | Target format |
| --- | --- | --- | --- |
| Regression | `nn.MSELoss()` | (m, 1) raw | (m, 1) float |
| Robust regression | `nn.SmoothL1Loss()` | (m, 1) raw | (m, 1) float |
| Binary classification | `nn.BCEWithLogitsLoss()` | (m, 1) **logits** | (m, 1) float, 0.0 or 1.0 |
| Multi-class (single label) | `nn.CrossEntropyLoss()` | (m, K) **logits** | (m,) **int64 class indices** |
| Multi-label | `nn.BCEWithLogitsLoss()` | (m, K) logits | (m, K) float 0/1 |

```python
import torch
import torch.nn as nn

# Multi-class: 4 examples, 3 classes
logits  = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3],
                        [1.2, 0.2, 3.1],
                        [0.1, 0.1, 0.1]])
targets = torch.tensor([0, 1, 2, 0])      # int64 indices, NOT one-hot

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(loss.item())          # 0.6161...

# Verify by hand for the last example, where all logits are equal:
# softmax gives [1/3, 1/3, 1/3], so L = -log(1/3) = 1.0986
print(-torch.log(torch.softmax(logits[3], dim=0)[0]).item())   # 1.0986

# Class imbalance: weight the rare classes up
weights = torch.tensor([1.0, 1.0, 5.0])
criterion_w = nn.CrossEntropyLoss(weight=weights)
```

The two errors that account for most `CrossEntropyLoss` failures: passing one-hot targets instead of integer indices (raises a dtype error, at least), and passing softmax outputs instead of logits (raises nothing, and quietly degrades your model).

### 5.8 Label smoothing, briefly

Hard one-hot targets ask the model to drive the correct logit to +∞, which encourages overconfidence and large weights. Label smoothing replaces the target `[0, 0, 1, 0]` with `[ε/K, ε/K, 1-ε+ε/K, ε/K]` for a small ε, typically 0.1.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

It usually gives a small accuracy gain and noticeably better-calibrated probabilities. It is standard in modern image classification training recipes. More in section 12.

### Check yourself

1. A 100-class classifier's loss starts at 2.0 on the first batch. Is that plausible? What does it suggest?
2. Show that for `L = -log σ(z)` with y = 0 the gradient is also `σ(z) - y`.
3. Why does `CrossEntropyLoss` want int64 targets rather than one-hot vectors?
4. Your loss becomes `nan` at epoch 3. List four candidate causes in the order you would check them.

*(Answers: 1 — No — a randomly initialized model should give ln(100) ≈ 4.6. Note that leakage is NOT the explanation: at step zero the model has learned nothing, so it cannot yet exploit a leaked feature. The causes are mechanical: the loss was logged after several updates rather than before the first; the output layer is the wrong size (ln(10) = 2.3, so a 10-unit head on a 100-class problem looks exactly like this); the output bias was initialized non-uniformly; label smoothing or class weighting shifted the baseline; or the reduction is wrong. Check the output shape first. 2 — With y=0, L = -log(1-σ(z)); dL/dz = -(1/(1-σ))·(-σ'(z)) = σ(z)(1-σ(z))/(1-σ(z)) = σ(z) = σ(z) - 0. 3 — Memory and speed: for 1000 classes a one-hot target matrix is 1000x larger than an index vector, and the loss only ever needs the single correct logit, so indexing is both cheaper and clearer. 4 — (i) learning rate too high causing divergence, (ii) a manual `log` of something that reached zero, or softmax applied before `CrossEntropyLoss` combined with extreme logits, (iii) division by zero in a custom layer or a normalization with zero variance, (iv) corrupt input data containing `nan` or `inf`. Diagnose by setting `torch.autograd.set_detect_anomaly(True)` and by printing the loss every step to find the exact batch.)*

## 6. Backpropagation

This is the section to slow down for. Backpropagation is how a network learns, it is the most-asked topic in deep learning interviews, and it is the one thing in this course that people routinely pretend to understand.

The headline: **backpropagation is the chain rule applied to a computational graph, organized so that shared subexpressions are computed once.** It is not a learning algorithm. Gradient descent is the learning algorithm; backpropagation is the efficient way to get the gradients it needs.

### 6.1 The problem

We have a loss `J` that depends on millions of parameters. Gradient descent needs `∂J/∂w` for every single one. How?

**Naive approach: finite differences.** For each parameter, nudge it and see what happens:

```
∂J/∂wᵢ ≈ [ J(w + εeᵢ) - J(w) ] / ε
```

This works, and it is a genuinely useful debugging tool (section 6.8). As a training method it is hopeless: each parameter needs its own forward pass, so a 100-million-parameter model needs 100 million forward passes for a single gradient step. At a millisecond each, that is over a day per step.

Backpropagation gets **all** the gradients in roughly the cost of **one** extra forward pass. That factor of 100 million is the reason deep learning exists.

### 6.2 The chain rule, and the insight that makes it efficient

Single-variable chain rule: if `y = f(u)` and `u = g(x)`, then `dy/dx = (dy/du)(du/dx)`.

Multi-variable version: if `y` depends on `u₁, ..., uₙ` and each `uᵢ` depends on `x`, then

```
dy/dx = Σᵢ (∂y/∂uᵢ)(duᵢ/dx)
```

Read that sum carefully, because it is the entire content of backpropagation. **If a variable influences the output through several paths, its gradient is the sum of the contributions along each path.**

Now the efficiency insight. Consider a chain `x → a → b → c → J`. To get `∂J/∂x` we need `(∂J/∂c)(∂c/∂b)(∂b/∂a)(∂a/∂x)`. To get `∂J/∂a` we need `(∂J/∂c)(∂c/∂b)(∂b/∂a)`. The second is a prefix of the first.

If you compute gradients starting from the **output** and moving backward, each new gradient reuses the product you have already accumulated. One backward sweep gives you everything. If you started from the inputs and moved forward, you would recompute the shared tail for every parameter. That directional choice is the whole trick, and it has a name: **reverse-mode automatic differentiation.**

Why reverse and not forward? Because we have one output (the scalar loss) and millions of inputs (the parameters). Reverse mode costs one sweep per *output*; forward mode costs one sweep per *input*. With one loss and 10⁸ parameters, reverse mode wins by a factor of 10⁸. If you ever needed the Jacobian of many outputs with respect to one input, forward mode would be the right choice — that is exactly what `torch.func.jvp` is for.

### 6.3 A tiny worked example, fully numeric

Before the general derivation, do one concrete case by hand. This is worth ten minutes with a pencil.

Network: one input, one hidden unit with sigmoid, one output, squared error. Parameters `w₁ = 0.5, b₁ = 0.1, w₂ = 0.8, b₂ = -0.3`. Input `x = 2.0`, target `y = 1.0`.

**Forward pass:**

```
z₁ = w₁x + b₁    = 0.5(2.0) + 0.1   = 1.1
a₁ = σ(z₁)       = 1/(1+e^-1.1)     = 0.7503
z₂ = w₂a₁ + b₂   = 0.8(0.7503) - 0.3 = 0.3002
a₂ = σ(z₂)       = 1/(1+e^-0.3002)  = 0.5745
L  = ½(a₂ - y)²  = ½(0.5745 - 1)²   = 0.0905
```

**Backward pass.** Work backward, one node at a time, and at each step write the local derivative and multiply.

```
∂L/∂a₂ = (a₂ - y)              = 0.5745 - 1.0        = -0.4255
∂a₂/∂z₂ = a₂(1 - a₂)            = 0.5745(0.4255)      =  0.2444
∂L/∂z₂ = (-0.4255)(0.2444)                            = -0.1040   ← call this δ₂

∂L/∂w₂ = δ₂ · a₁               = (-0.1040)(0.7503)    = -0.0780
∂L/∂b₂ = δ₂ · 1                                       = -0.1040

∂L/∂a₁ = δ₂ · w₂               = (-0.1040)(0.8)       = -0.0832
∂a₁/∂z₁ = a₁(1 - a₁)            = 0.7503(0.2497)      =  0.1874
∂L/∂z₁ = (-0.0832)(0.1874)                            = -0.0156   ← call this δ₁

∂L/∂w₁ = δ₁ · x                = (-0.0156)(2.0)       = -0.0312
∂L/∂b₁ = δ₁ · 1                                       = -0.0156
```

Now read the pattern off the arithmetic:

- **The gradient of a weight is the error signal at its output times the activation at its input.** `∂L/∂w₂ = δ₂ · a₁`. This is universal — it is true for every fully connected layer in every network.
- **The gradient of a bias is just the error signal.** Because `∂z/∂b = 1`.
- **The error signal propagates backward by multiplying by the weight and then by the activation derivative.** `δ₁ = (δ₂ · w₂) · g'(z₁)`.
- **Gradients shrink as they travel back.** δ₂ = -0.104 became δ₁ = -0.0156, a factor of 6.7 in one layer, mostly because σ' ≤ 0.25. Twenty layers of that and nothing reaches the front. You have just watched the vanishing gradient problem happen.

### 6.4 The four backpropagation equations

Now the general case. Recall the forward pass: `z[l] = W[l]a[l-1] + b[l]`, `a[l] = g(z[l])`.

Define the error signal at layer l:

```
δ[l] = ∂L/∂z[l]        (shape n[l] x 1)
```

Why define it on the pre-activation `z` rather than the activation `a`? Because it makes the weight gradients fall out immediately, as you are about to see.

---

**BP1 — the error at the output layer**

```
δ[L] = ∇_a L ⊙ g'(z[L])
```

*Derivation.* By the chain rule, `∂L/∂z[L]ⱼ = (∂L/∂a[L]ⱼ)(∂a[L]ⱼ/∂z[L]ⱼ)`. Since `g` is applied elementwise, `a[L]ⱼ` depends only on `z[L]ⱼ`, so there is no sum. The second factor is `g'(z[L]ⱼ)`. Stacking over j gives the elementwise product.

*The important special case.* For softmax output with cross-entropy loss, this collapses to

```
δ[L] = a[L] - y            (one-hot y)
```

no activation derivative needed. Derivation in 6.6.

---

**BP2 — propagating the error backward**

```
δ[l] = ( W[l+1]ᵀ δ[l+1] ) ⊙ g'(z[l])
```

*Derivation.* We want `∂L/∂z[l]ⱼ`. The variable `z[l]ⱼ` affects the loss only through the next layer's pre-activations, so by the multivariable chain rule:

```
∂L/∂z[l]ⱼ = Σₖ (∂L/∂z[l+1]ₖ)(∂z[l+1]ₖ/∂z[l]ⱼ) = Σₖ δ[l+1]ₖ · ∂z[l+1]ₖ/∂z[l]ⱼ
```

Now `z[l+1]ₖ = Σᵢ W[l+1]ₖᵢ a[l]ᵢ + b[l+1]ₖ` and `a[l]ᵢ = g(z[l]ᵢ)`. Only the i = j term depends on `z[l]ⱼ`, so

```
∂z[l+1]ₖ/∂z[l]ⱼ = W[l+1]ₖⱼ · g'(z[l]ⱼ)
```

Substituting:

```
∂L/∂z[l]ⱼ = [ Σₖ W[l+1]ₖⱼ δ[l+1]ₖ ] · g'(z[l]ⱼ)
```

The bracket is the j-th component of `W[l+1]ᵀ δ[l+1]`. Stacking over j gives BP2.

*Reading it.* `W[l+1]ᵀ` sends the error backward through the connections — it is the transpose because information now flows the other way. `⊙ g'(z[l])` gates the error by how responsive each unit was: a saturated unit has `g' ≈ 0` and blocks the error entirely.

---

**BP3 — the bias gradient**

```
∂L/∂b[l] = δ[l]
```

*Derivation.* `∂z[l]ⱼ/∂b[l]ⱼ = 1`, so `∂L/∂b[l]ⱼ = δ[l]ⱼ · 1`.

---

**BP4 — the weight gradient**

```
∂L/∂W[l] = δ[l] · a[l-1]ᵀ          (outer product, shape n[l] x n[l-1])
```

*Derivation.* `z[l]ⱼ = Σᵢ W[l]ⱼᵢ a[l-1]ᵢ + b[l]ⱼ`, so `∂z[l]ⱼ/∂W[l]ⱼᵢ = a[l-1]ᵢ`. Therefore `∂L/∂W[l]ⱼᵢ = δ[l]ⱼ · a[l-1]ᵢ`, which as a matrix is the outer product.

*Reading it.* This one equation explains a great deal:

- If the input activation `a[l-1]ᵢ` is small, the gradient on that weight is small, so it barely learns. **Units that do not activate do not learn.** This is why input standardization matters (section 22) and why dead ReLUs are dead.
- If the error signal `δ[l]ⱼ` is small — because the unit saturated, or because the error already vanished on its way back — that entire row of the weight matrix barely learns.

---

**All four, together:**

```
δ[L]      = ∇_a L ⊙ g'(z[L])                  (or a[L] - y for softmax+CE)
δ[l]      = ( W[l+1]ᵀ δ[l+1] ) ⊙ g'(z[l])      for l = L-1 ... 1
∂L/∂b[l]  = δ[l]
∂L/∂W[l]  = δ[l] a[l-1]ᵀ
```

If you can reproduce these four lines and say why each is true, you understand backpropagation. Cover the page and try now.

### 6.5 The batched version (what code actually does)

With `m` examples, using the row convention PyTorch uses: `Z[l]` and `A[l]` have shape `(m, n[l])`, and the error `Δ[l] = ∂J/∂Z[l]` also has shape `(m, n[l])`.

```
Δ[L]     = (A[L] - Y) / m                      softmax + CE, averaged over batch
Δ[l]     = ( Δ[l+1] W[l+1] ) ⊙ g'(Z[l])
∂J/∂W[l] = Δ[l]ᵀ A[l-1]                        shape (n[l], n[l-1])
∂J/∂b[l] = Δ[l].sum(axis=0)                    sum over the batch dimension
```

Two things changed. The transpose moved because examples are now rows instead of columns. The outer product became a matrix product, which **sums over the batch automatically** — each example contributes its own outer product and they add. The bias gradient must sum over the batch explicitly, because broadcasting `b` across m rows during the forward pass means the bias influenced all m outputs.

**Shape check as a debugging tool.** `Δ[l]ᵀ A[l-1]` is `(n[l], m) x (m, n[l-1]) = (n[l], n[l-1])`, which matches `W[l]`. Whenever you are unsure whether a transpose belongs, check which arrangement produces the shape of the parameter. There is usually only one.

### 6.6 The softmax + cross-entropy gradient (do this one by hand once)

This derivation appears in interviews constantly.

Setup: logits `z`, `ŷₖ = e^(zₖ)/Σⱼe^(zⱼ)`, loss `L = -Σₖ yₖ log ŷₖ` with one-hot `y`.

**Step 1: derivative of softmax.** We need `∂ŷₖ/∂zᵢ`, and there are two cases.

Write `S = Σⱼ e^(zⱼ)`, so `ŷₖ = e^(zₖ)/S`. Note `∂S/∂zᵢ = e^(zᵢ)`.

Case k = i, by the quotient rule:

```
∂ŷₖ/∂zₖ = [ e^(zₖ)·S - e^(zₖ)·e^(zₖ) ] / S²
        = (e^(zₖ)/S) · [ (S - e^(zₖ))/S ]
        = ŷₖ (1 - ŷₖ)
```

Case k ≠ i: the numerator `e^(zₖ)` does not depend on `zᵢ`, so only S does:

```
∂ŷₖ/∂zᵢ = e^(zₖ) · (-1/S²) · e^(zᵢ) = -ŷₖ ŷᵢ
```

Combined, using the Kronecker delta `1{k=i}`:

```
∂ŷₖ/∂zᵢ = ŷₖ ( 1{k=i} - ŷᵢ )
```

**Step 2: chain through the loss.**

```
∂L/∂zᵢ = Σₖ (∂L/∂ŷₖ)(∂ŷₖ/∂zᵢ)
       = Σₖ ( -yₖ/ŷₖ ) · ŷₖ ( 1{k=i} - ŷᵢ )
       = Σₖ -yₖ ( 1{k=i} - ŷᵢ )
       = -yᵢ + ŷᵢ Σₖ yₖ
```

Since `y` is one-hot, `Σₖ yₖ = 1`, giving

```
∂L/∂zᵢ = ŷᵢ - yᵢ        i.e.    δ[L] = ŷ - y
```

**The gradient of the loss with respect to the logits is simply the prediction minus the target.** No exponentials, no division, no activation derivative. The messy softmax derivative cancelled exactly against the log in the loss. This is why frameworks fuse softmax and cross-entropy into one operation: the fused backward pass is a subtraction.

### 6.7 Cost analysis

For a layer mapping `n[l-1] → n[l]` with batch size m:

| Pass | Operation | Cost |
| --- | --- | --- |
| Forward | `A[l-1] W[l]ᵀ` | O(m · n\[l\] · n\[l-1\]) |
| Backward | `Δ[l] W[l]` (input grad) | O(m · n\[l\] · n\[l-1\]) |
| Backward | `Δ[l]ᵀ A[l-1]` (weight grad) | O(m · n\[l\] · n\[l-1\]) |

The backward pass costs about **twice** the forward pass, so a training step is roughly 3x the cost of an inference step. That is the number to use when estimating training time.

**Memory** is the other half of the story. BP4 needs `a[l-1]`, so **every layer's activations must be kept in memory from the forward pass until the backward pass reaches them.** This is why memory usage scales with batch size, and why an out-of-memory error during `loss.backward()` is fixed by reducing batch size. It is also why `torch.no_grad()` at inference time saves so much memory: no activations are stored.

If you are memory-bound, **gradient checkpointing** (`torch.utils.checkpoint`) stores only some activations and recomputes the rest during the backward pass, trading about 30% more compute for large memory savings.

### 6.8 Gradient checking: how to know your derivation is right

If you implement a gradient by hand, verify it numerically. Use the **centred** difference, whose error is O(ε²) rather than O(ε):

```
∂J/∂w ≈ [ J(w + ε) - J(w - ε) ] / (2ε)
```

Then compare against your analytic gradient with a relative error:

```
relative_error = |analytic - numeric| / max(|analytic|, |numeric|, 1e-8)
```

Rules of thumb: below 1e-7 is correct; 1e-7 to 1e-4 is suspicious and worth investigating; above 1e-3 means you have a bug.

```python
import torch

def gradient_check(f, params, eps=1e-5):
    """f: callable taking params and returning a scalar loss."""
    # analytic gradients via autograd
    loss = f(params)
    loss.backward()
    analytic = [p.grad.clone() for p in params]

    max_err = 0.0
    for p, g in zip(params, analytic):
        flat = p.data.view(-1)
        for i in range(flat.numel()):
            orig = flat[i].item()
            flat[i] = orig + eps
            with torch.no_grad():
                loss_plus = f(params).item()
            flat[i] = orig - eps
            with torch.no_grad():
                loss_minus = f(params).item()
            flat[i] = orig                      # restore

            numeric  = (loss_plus - loss_minus) / (2 * eps)
            a = g.view(-1)[i].item()
            err = abs(a - numeric) / max(abs(a), abs(numeric), 1e-8)
            max_err = max(max_err, err)
    return max_err

# Use float64 for gradient checking; float32 rounding swamps the signal.
w = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
x = torch.randn(5, 3, dtype=torch.float64)
y = torch.randn(5, 4, dtype=torch.float64)
f = lambda ps: ((x @ ps[0].T - y) ** 2).mean()
print(gradient_check(f, [w]))       # ~1e-10
```

Three practical notes: use float64 or numerical noise will drown the comparison; turn dropout and other stochastic layers off, since the loss must be deterministic; and check a random subset of parameters rather than all of them, because this procedure is O(number of parameters) forward passes.

### 6.9 A complete from-scratch implementation

Here is a two-layer network trained with backpropagation written out explicitly, no autograd. If you can read this and map each line to one of BP1-BP4, you have it.

```python
import numpy as np

np.random.seed(0)

def relu(z):        return np.maximum(0, z)
def relu_grad(z):   return (z > 0).astype(z.dtype)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)        # stability (section 5.6)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class TwoLayerNet:
    def __init__(self, n_in, n_hidden, n_out):
        # He initialization (section 9)
        self.W1 = np.random.randn(n_hidden, n_in)  * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_out, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        # X: (m, n_in).  Cache everything the backward pass needs.
        self.X  = X
        self.Z1 = X @ self.W1.T + self.b1        # (m, n_hidden)
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2.T + self.b2  # (m, n_out)  logits
        self.A2 = softmax(self.Z2)
        return self.A2

    def loss(self, Y_onehot):
        m = Y_onehot.shape[0]
        eps = 1e-12                               # guard against log(0)
        return -np.sum(Y_onehot * np.log(self.A2 + eps)) / m

    def backward(self, Y_onehot):
        m = Y_onehot.shape[0]

        D2 = (self.A2 - Y_onehot) / m             # BP1 (softmax+CE shortcut)
        dW2 = D2.T @ self.A1                      # BP4
        db2 = D2.sum(axis=0)                      # BP3

        D1 = (D2 @ self.W2) * relu_grad(self.Z1)  # BP2
        dW1 = D1.T @ self.X                       # BP4
        db1 = D1.sum(axis=0)                      # BP3

        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2

# --- train on XOR, the problem from section 1 ---
X = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
Y = np.array([[1.,0.], [0.,1.], [0.,1.], [1.,0.]])   # one-hot: class 0 = XOR false

net = TwoLayerNet(2, 8, 2)
for epoch in range(2001):
    net.forward(X)
    L = net.loss(Y)
    net.step(net.backward(Y), lr=0.5)
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss {L:.4f}")

print(net.forward(X).argmax(axis=1))    # [0 1 1 0]  -> XOR solved
```

Run it. Watch the loss fall from about 0.69 (= ln 2, the two-class random baseline from section 5.4) to near zero. You have now trained a neural network with no framework at all.

### 6.10 Common confusions, resolved

**"Is backpropagation the same as gradient descent?"** No. Backpropagation computes gradients. Gradient descent uses them to update parameters. You could compute gradients by backprop and then update with Adam, or with a genetic algorithm, or not at all.

**"Why do we need `zero_grad()`?"** PyTorch *accumulates* gradients into `.grad` rather than overwriting them. Skip `zero_grad()` and step k uses the sum of gradients from steps 1 through k, which grows without bound and destroys training. The accumulation behaviour is deliberate: it lets you simulate a large batch on small hardware by running several forward/backward passes before one optimizer step.

**"Why store activations at all?"** BP4 needs `a[l-1]` to form the weight gradient. Without the stored activation there is no way to compute it short of recomputing the forward pass, which is exactly what gradient checkpointing chooses to do.

**"Does backprop find the global minimum?"** No, and neither does gradient descent. Deep network losses are wildly non-convex. In practice this matters far less than theory suggests: in high dimensions, most critical points are saddle points rather than bad local minima, and the local minima that gradient descent reaches tend to have similar loss values. Section 7.5 says more.

### Check yourself

1. In the worked example of 6.3, what would `∂L/∂w₁` be if `w₂` were 0? Explain in words, without recomputing.
2. Why does BP2 use `W[l+1]ᵀ` and not `W[l+1]`?
3. A network has 3 layers of 1000 units each. How much memory do the stored activations need for a batch of 256 in float32?
4. You replace all activations with the identity function. What do BP1-BP4 become, and what does that tell you?

*(Answers: 1 — zero. The error signal reaches layer 1 only through `w₂`; if that connection carries nothing, layer 1 receives no error and cannot learn. This is the mechanism behind vanishing gradients, and behind why a dead unit upstream freezes everything feeding into it. 2 — the forward pass maps layer l to layer l+1, so the matrix has shape (n\[l+1\], n\[l\]); the backward pass maps an error of size n\[l+1\] to one of size n\[l\], which requires the transpose. Shape-checking alone forces it. 3 — 3 layers x 1000 units x 256 examples x 4 bytes ≈ 3.1 MB for activations, plus the same again for pre-activations if those are cached separately. Small here, but scale to a ResNet-50 on 224x224 images and it becomes gigabytes — which is why batch size is the first thing to reduce on an OOM error. 4 — g'(z) = 1 everywhere, so BP2 becomes `δ[l] = W[l+1]ᵀδ[l+1]` with no gating. The gradients still flow, but by section 3.1 the whole network is equivalent to a single linear layer, so there is nothing useful to learn. Gradients flowing is necessary but not sufficient.)*

## 7. Gradient Descent: Actually Using the Gradients

Backpropagation hands you `∂J/∂θ` for every parameter θ. Gradient descent decides what to do with it.

```
θ ← θ - η · ∂J/∂θ
```

Move each parameter a little way in the direction that reduces the loss. The gradient points uphill (steepest increase), so we subtract it.

### 7.1 Why the gradient is the steepest direction

Worth one paragraph, because it justifies everything that follows. A first-order Taylor expansion around the current parameters:

```
J(θ + Δθ) ≈ J(θ) + ∇J(θ)ᵀ Δθ
```

To make the change as negative as possible, subject to `||Δθ|| = ε` fixed, we minimize `∇J(θ)ᵀ Δθ`. The dot product `∇Jᵀ Δθ = ||∇J|| ||Δθ|| cos α` is most negative when cos α = -1, i.e. when `Δθ` points exactly opposite to `∇J`. Hence `Δθ = -η ∇J`.

Two cautions follow directly from the derivation. It is only valid for **small** steps, because the Taylor expansion is local — take too big a step and the approximation is worthless, which is exactly what divergence looks like. And it is the steepest direction with respect to the **Euclidean** norm; other norms give other update rules, which is one way to understand what Adam is doing (section 10).

### 7.2 Batch, stochastic, mini-batch

The question is: how much data do you use to estimate the gradient at each step?

**Batch (full) gradient descent** uses all N examples per step.

```
θ ← θ - η · (1/N) Σᵢ ∇L(x⁽ⁱ⁾, y⁽ⁱ⁾)
```

The gradient is exact and the descent is smooth, but each step requires a full pass through the dataset. With a million examples, one update takes minutes. Unusable at scale.

**Stochastic gradient descent (SGD)** uses one example per step. Updates are extremely fast and the noise can help escape sharp regions, but the gradient estimate is high-variance, the loss curve jumps around, and single-example computation wastes modern hardware, which is built for parallel arithmetic.

**Mini-batch gradient descent** uses m examples per step, typically 32 to 512. This is what everyone actually does, and confusingly it is what people mean when they say "SGD" today. PyTorch's `optim.SGD` is mini-batch SGD.

The reason mini-batching wins is statistical. The standard error of a mean falls as `1/√m`. Going from m=1 to m=100 cuts gradient noise by 10x; going from 100 to 10,000 cuts it by another 10x but costs 100x more compute. **Diminishing returns on accuracy, linear growth in cost** — so an intermediate batch size is the right trade-off, and the exact value is not very sensitive.

|  | Batch | Mini-batch | SGD |
| --- | --- | --- | --- |
| Examples per step | N | 32-512 | 1 |
| Gradient quality | exact | good estimate | noisy |
| Steps per epoch | 1 | N/m | N |
| Hardware use | good | excellent | poor |
| Escapes sharp minima | poorly | well | very well |
| Practical verdict | too slow | **use this** | too noisy |

**Epoch vs iteration.** One *epoch* is one pass over the training set. One *iteration* (or step) is one parameter update. With 50,000 examples and batch size 100, one epoch = 500 iterations. Interviewers ask this; it is free marks.

### 7.3 The learning rate, and how to find one

The learning rate is the most important hyperparameter in deep learning. Not among the most important — the most important.

| η | Symptom |
| --- | --- |
| Far too large | Loss becomes `nan` or `inf` within a few steps |
| Too large | Loss oscillates or plateaus high; may increase |
| Good | Loss falls steadily, then flattens |
| Too small | Loss falls, but glacially; may stall in a bad region |

Why does a too-large rate diverge? For a quadratic bowl `J = (1/2)λθ²`, the update is `θ ← θ - ηλθ = (1 - ηλ)θ`. This converges only if `|1 - ηλ| < 1`, i.e. `η < 2/λ`. Exceed that and each step overshoots further than the last, and the loss grows geometrically. In a real network λ is the largest eigenvalue of the Hessian, so **the maximum stable learning rate is set by the sharpest direction in the loss surface**, even though most directions are much flatter. That mismatch is exactly the ill-conditioning that momentum and Adam address.

**The learning-rate range test** (Smith, 2015) is the fastest practical way to pick one. Train for a few hundred steps while increasing η exponentially, and plot loss against η:

```python
import torch, math

def lr_range_test(model, loader, criterion, device,
                  lr_start=1e-7, lr_end=10, num_iter=200):
    opt = torch.optim.SGD(model.parameters(), lr=lr_start)
    gamma = (lr_end / lr_start) ** (1 / num_iter)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    lrs, losses = [], []

    model.train()
    it = iter(loader)
    for i in range(num_iter):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        lrs.append(opt.param_groups[0]["lr"])
        losses.append(loss.item())
        sched.step()
        if loss.item() > 4 * min(losses):     # diverging; stop
            break
    return lrs, losses

# Plot losses against lrs on a log x-axis.
# Pick roughly one order of magnitude BELOW the minimum of the curve,
# i.e. the steepest downward part, not the bottom.
```

Pick a value in the steepest-descent region, about 10x below where the loss bottoms out. Reasonable defaults if you skip the test: **SGD with momentum, η = 0.1** for CNNs with batch normalization; **Adam, η = 3e-4** for almost everything else. Both are starting points, not answers.

### 7.4 The training loop, in full

Every PyTorch training loop you will ever write has this skeleton.

```python
for epoch in range(num_epochs):
    model.train()                                   # dropout/BN in train mode
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()                       # 1. clear old gradients
        outputs = model(x_batch)                    # 2. forward pass
        loss    = criterion(outputs, y_batch)       # 3. compute loss
        loss.backward()                             # 4. backward pass
        optimizer.step()                            # 5. update parameters

    model.eval()                                    # BN uses running stats
    with torch.no_grad():                           # no graph, no grad memory
        for x_val, y_val in val_loader:
            ...
```

The five numbered lines, in that order, every time. Commit them to memory; interviewers do ask you to write this from scratch.

**The three classic errors:**

- **Forgetting `zero_grad()`.** Gradients accumulate. Training degrades progressively rather than failing outright, which makes it hard to spot.
- **Forgetting `model.eval()`.** Dropout stays active and BatchNorm keeps using batch statistics, so validation accuracy looks worse and fluctuates between runs on the same data.
- **Forgetting `torch.no_grad()` during evaluation.** Everything still works, but memory use roughly triples and evaluation runs several times slower, because the graph is built and activations are retained for a backward pass that never comes.

### 7.5 What the loss landscape is really like

Students often imagine a bumpy 2-D surface with many valleys, and worry constantly about local minima. The reality in high dimensions is different and more encouraging.

**Saddle points dominate, not local minima.** At a critical point, the Hessian's eigenvalues determine the type: all positive = local minimum, all negative = local maximum, mixed = saddle. In d dimensions, if eigenvalue signs were roughly independent, the chance of all d being positive is about 2^(-d). For d = 10⁶ that is indistinguishable from zero. **Almost every critical point in a large network is a saddle point.** Gradient descent slows near saddles but the noise in mini-batch gradients generally knocks it loose, which is one underrated benefit of stochasticity.

**Most minima are about equally good.** Empirically, large networks trained from different random seeds reach different parameter vectors with very similar test loss. The classical fear that you might land in a terrible local minimum has not borne out for large models.

**Flat minima tend to generalize better than sharp ones.** The intuition: in a flat basin, small perturbations to the parameters barely change the loss, so the small mismatch between the training distribution and the test distribution also barely changes it. Smaller batches and larger learning rates add noise that pushes solutions toward flatter regions, which is part of why very large batch training often needs extra care (warmup, scaled learning rates) to match small-batch generalization. The flat-minima story is widely used and empirically supported, but note that "flatness" is not reparameterization-invariant, so treat it as a useful heuristic rather than a theorem.

**Ill-conditioning is the real enemy.** If the Hessian has eigenvalues spanning several orders of magnitude, the loss surface is a long narrow ravine. Gradient descent bounces across the steep walls while creeping along the gentle floor, because a single scalar learning rate must serve both directions. This — not local minima — is the practical obstacle, and it motivates momentum, adaptive methods, and normalization.

### 7.6 Batch size: what changes when you change it

| Larger batch | Smaller batch |
| --- | --- |
| Less gradient noise, smoother curve | Noisier, sometimes better generalization |
| Better GPU utilization, faster per epoch | Slower per epoch, more Python overhead |
| More memory | Less memory |
| Fewer updates per epoch | More updates per epoch |
| Often needs a larger η and warmup | Tolerates a wide range of η |

A practical rule when you change batch size: the **linear scaling rule** (Goyal et al., 2017) says multiply the learning rate by the same factor you multiplied the batch size by, and add a few epochs of linear warmup. It works well up to batch sizes of a few thousand and then breaks down. Some practitioners prefer square-root scaling for adaptive optimizers; treat both as heuristics to be validated.

If you are memory-limited, **gradient accumulation** gives you a large effective batch on small hardware:

```python
accum_steps = 4              # effective batch = batch_size * 4
optimizer.zero_grad()
for i, (x, y) in enumerate(train_loader):
    loss = criterion(model(x), y) / accum_steps    # scale so the mean is right
    loss.backward()                                # gradients accumulate
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Note the division by `accum_steps`: without it you would be summing four batch-mean gradients rather than averaging them, effectively quadrupling the learning rate.

### 7.7 When to stop

Watch training loss and validation loss together. Four regimes:

| Training loss | Validation loss | Diagnosis | Action |
| --- | --- | --- | --- |
| High | High | Underfitting | Bigger model, train longer, higher η |
| Falling | Falling | Healthy | Keep going |
| Falling | Rising | Overfitting | Regularize, get data, early stop |
| Flat from the start | Flat | Something is broken | See section 22 |

**Early stopping** means keeping the parameters from the epoch with the best validation loss rather than the last epoch. It costs nothing and is close to free regularization. Always save the best checkpoint, not the final one.

### Check yourself

1. 60,000 training examples, batch size 128, 20 epochs. How many parameter updates?
2. Your loss oscillates wildly between epochs but trends downward. Two plausible causes?
3. Why does gradient accumulation with 4 steps of batch 16 not perfectly reproduce one step of batch 64?
4. Explain why saddle points are a bigger practical concern than local minima in high dimensions, and why mini-batch noise helps.

*(Answers: 1 — ceil(60000/128) = 469 iterations per epoch, x20 = 9,380 updates. 2 — learning rate slightly too high, or batch size too small making gradient estimates noisy; also possible is an unshuffled loader where each epoch sees classes in a fixed order. 3 — BatchNorm computes statistics per actual forward pass, so 4 passes of 16 give 4 different sets of batch statistics rather than one set over 64 examples; without normalization layers the two are equivalent up to floating-point ordering. 4 — the probability that all eigenvalues of the Hessian share a sign falls like 2^(-d), so critical points in high dimensions are almost always saddles; gradients vanish near a saddle, but mini-batch noise perturbs the parameters off the saddle's stable manifold and descent resumes.)*

## 8. Why Deep? From Neural Networks to Deep Learning

The universal approximation theorem says one hidden layer suffices. So why is the field called *deep* learning? Because "suffices in principle" and "is a good idea" are different claims, and the gap between them is enormous.

### 8.1 The efficiency argument

A shallow network *can* represent anything, but it may need exponentially more units to do it.

**A concrete example: parity.** The parity function on n bits returns 1 if an odd number of inputs are 1. It is known that a depth-2 circuit of threshold units needs exponentially many units in n to compute parity, while a deep circuit computes it with O(n) units by chaining XORs — each XOR taking 2 hidden units, as in section 1.5. Depth n, width 2, versus depth 2, width exponential.

**A geometric example: counting linear regions.** A ReLU network partitions its input space into regions on which it is linear. Montúfar et al. (2014) showed that a deep ReLU network can create a number of linear regions that grows **exponentially with depth** but only **polynomially with width**. Intuitively, each layer can fold the input space, and folding a folded space multiplies the number of pieces.

A helpful physical picture: take a sheet of paper and fold it in half repeatedly. After k folds, one scissor cut produces 2^k holes. Each layer is a fold; depth compounds, width does not.

**Proven separation results.** Telgarsky (2016) constructed functions representable by a network of depth k with O(k) units that require Ω(2^k) units for any network of depth O(√k). These are the rare cases where the theory is clean and the intuition matches.

The honest caveat: these are worst-case constructions on specially designed functions. They establish that depth *can* be exponentially more efficient. That real-world problems happen to have the compositional structure that rewards depth is an empirical observation, not a theorem — but it is a very well-supported one.

### 8.2 The representation-learning argument (the one that matters in practice)

This is the intuition to carry with you.

Natural data is **compositional and hierarchical**. An image is made of edges; edges form textures and corners; those form object parts; parts form objects; objects form scenes. Speech is phonemes, then syllables, then words, then phrases. Each level is built from the level below by a relatively simple operation.

A deep network mirrors that structure. Layer 1 detects edges. Layer 2 combines edges into corners and curves. Layer 3 combines those into motifs. Layer 5 detects eyes and wheels. Layer 10 detects faces and cars. We know this is not just a story: Zeiler and Fergus (2014) visualized what maximally activates units in a trained CNN and found exactly this progression, and feature-visualization work since has confirmed and extended it.

**Why this makes learning easier.** Features get reused. An edge detector in layer 1 serves the eye detector, the wheel detector, and the text detector simultaneously. A shallow network must learn each high-level concept as a direct function of raw pixels, with no sharing. Deep networks amortize the work.

This is also precisely why transfer learning works (section 20). The early layers of a network trained on ImageNet learn edge and texture detectors that are useful for X-rays, satellite imagery, and manufacturing defect detection, because edges are edges everywhere.

### 8.3 Depth versus width, practically

|  | Add depth | Add width |
| --- | --- | --- |
| Parameter efficiency | Higher | Lower |
| Feature abstraction | Builds hierarchy | Enriches one level |
| Optimization difficulty | Harder (vanishing gradients) | Easier |
| Parallelizability | Worse (sequential layers) | Better |
| Needs residual connections? | Beyond \~20 layers, yes | No |

The practical answer for a new problem: start moderately deep, with residual connections if you go beyond about 20 layers, and tune width second. For image classification, the empirical sweet spots have been well mapped by architecture search — EfficientNet's compound scaling result (Tan & Le, 2019) is that depth, width, and input resolution should be scaled **together** by a fixed ratio rather than one at a time. That is a genuinely useful rule.

### 8.4 What actually changed around 2012

Backpropagation was published in usable form in 1986. Deep learning took over in 2012. The 26-year gap is the most instructive fact in the field's history, and the reasons for it define most of what this tutorial covers next.

**Data.** ImageNet (2009) provided 1.2 million labelled images across 1,000 categories. Before that, standard vision benchmarks had tens of thousands of images. Deep networks have enormous capacity and simply memorize small datasets; they need scale to generalize.

**Compute.** A GPU performs the dense matrix multiplications at the heart of neural networks tens to hundreds of times faster than a contemporary CPU. AlexNet trained on two GTX 580 cards, each with 3 GB of memory. Experiments that would have taken months became overnight jobs — and research speed is roughly proportional to experiment turnaround.

**ReLU.** Replacing sigmoid and tanh with ReLU removed the saturation that had made gradients vanish in deep stacks (sections 3.2 and 9). This single change is a large part of why deeper networks suddenly trained at all.

**Initialization.** Xavier (2010) and He (2015) initialization schemes kept activation and gradient variance stable across layers. Before these, deep networks were often untrainable from random starts — which is why the 2006-2009 era relied on elaborate layer-wise unsupervised pretraining that is now almost entirely obsolete.

**Regularization.** Dropout (2012) and heavy data augmentation let very large models train on limited data without collapsing into memorization.

**Better optimizers.** Momentum, RMSProp, and Adam handled the ill-conditioned loss surfaces that plain SGD struggled with.

**Residual connections (2015).** These made networks of 100+ layers trainable, which pushed depth from "about 8 layers" to "as many as you want."

Notice the shape of that list: **one architectural idea, and six engineering advances.** That proportion is representative of the field, and it is the reason sections 9 through 12 exist. Understanding backprop makes you literate; understanding initialization, optimizers, normalization, and regularization makes you effective.

### 8.5 How deep is deep, in 2026?

| Model type | Typical depth | Parameters |
| --- | --- | --- |
| MLP on tabular data | 2-4 layers | 10⁴ - 10⁶ |
| LeNet-5 (1998) | 7 layers | 6 x 10⁴ |
| AlexNet (2012) | 8 layers | 6 x 10⁷ |
| VGG-16 (2014) | 16 layers | 1.4 x 10⁸ |
| ResNet-50 (2015) | 50 layers | 2.6 x 10⁷ |
| ResNet-152 (2015) | 152 layers | 6 x 10⁷ |
| ViT-Large (2020) | 24 blocks | 3 x 10⁸ |
| Large language models | 32-120+ blocks | 10⁹ - 10¹² |

Note that ResNet-50 has *fewer* parameters than VGG-16 despite being three times deeper. Depth is not the same as size, and better architecture beats brute force. That observation is worth keeping when you design your own models.

### Check yourself

1. Why can a shallow network need exponentially many units for a function a deep network handles with few?
2. If depth is so useful, why did networks not simply get deeper in the 1990s?
3. Your 30-layer plain MLP trains worse than a 10-layer one on the same data, on both training and validation sets. Is this overfitting? What is it, and what would you do?

*(Answers: 1 — a shallow network must cover the function's variation with a single layer of features, so the number of units scales with the number of distinct regions it must represent; a deep network composes transformations, and composition multiplies the number of regions, giving exponential growth in depth. 2 — vanishing gradients with saturating activations, poor initialization, no normalization, insufficient data and compute, and no residual connections; the algorithm was there, the surrounding engineering was not. 3 — not overfitting, since training loss is also worse; this is the degradation problem that ResNet was designed to solve. Add residual connections and batch normalization, check initialization, and verify the network can overfit a tiny subset first (section 22).)*

## 9. Vanishing Gradients and Weight Initialization

For twenty years, deep networks did not train. This section explains why, and how three lines of initialization code fixed it.

### 9.1 The problem, quantified

Recall BP2: `δ[l] = (W[l+1]ᵀ δ[l+1]) ⊙ g'(z[l])`. Unrolling from the output back to layer 1:

```
δ[1] = ( W[2]ᵀ ... W[L]ᵀ δ[L] ) gated at every layer by g'(z)
```

The gradient at layer 1 is a **product of L terms**. Products of many numbers do one of three things: if the typical factor is below 1, the product collapses toward zero; if above 1, it explodes; only if the factors sit near 1 does the product stay useful.

**Vanishing gradients.** With sigmoid, `g'(z) ≤ 0.25`. Even in the best case, 10 layers scale the gradient by at most 0.25¹⁰ ≈ 10⁻⁶. Early layers receive essentially nothing, so they stay near their random initialization while later layers learn. The network is effectively shallow, with a random feature extractor at the front.

**Exploding gradients.** If weights are large, the products grow. Loss becomes `nan` within a few steps. Less common than vanishing in feedforward networks, but the standard failure mode of RNNs.

Here is what makes this insidious: the model still trains. The loss goes down, because the last few layers are learning. It simply plateaus far above where it should, and nothing in the output says why.

### 9.2 Variance analysis: deriving the right initialization

The fix comes from a question: what initial weight scale keeps the *variance* of activations constant as we go deeper?

Consider one layer, `z = Σᵢ wᵢxᵢ` over `n_in` inputs (ignore the bias, initialized to zero). Assume:

- weights `wᵢ` are i.i.d. with mean 0 and variance `Var(w)`
- inputs `xᵢ` are i.i.d. with mean 0 and variance `Var(x)`
- weights and inputs are independent

For independent zero-mean variables, `Var(wᵢxᵢ) = Var(wᵢ)Var(xᵢ)` and variances of independent sums add:

```
Var(z) = Σᵢ Var(wᵢ xᵢ) = n_in · Var(w) · Var(x)
```

**To keep `Var(z) = Var(x)` we need `n_in · Var(w) = 1`, i.e. `Var(w) = 1/n_in`.**

That is **Xavier/Glorot initialization**, from Glorot & Bengio (2010). They also considered the backward pass, where the same argument with `n_out` gives `Var(w) = 1/n_out`. You cannot satisfy both unless the layer is square, so they compromised with the harmonic-mean form:

```
Var(w) = 2 / (n_in + n_out)
```

In uniform form, `w ~ U[-a, a]` with `a = √(6/(n_in + n_out))` (the 6 appears because a uniform distribution on \[-a,a\] has variance a²/3).

### 9.3 He initialization: the ReLU correction

Xavier assumes the activation is roughly linear near zero and symmetric, which holds for tanh but not for ReLU. ReLU zeroes out the negative half of its inputs. For a symmetric zero-mean input distribution, that halves the variance of the output:

```
Var(ReLU(z)) ≈ (1/2) Var(z)
```

(Carefully: `E[ReLU(z)²] = (1/2)E[z²]` for symmetric z, since half the mass maps to zero and the other half passes through unchanged. Strictly, ReLU output is not zero-mean, so this is an approximation, but it is the right one.)

So each ReLU layer halves the variance, and after L layers the signal has been scaled by 2^(-L). Compensate by doubling the weight variance:

```
Var(w) = 2 / n_in
```

This is **He initialization** (He et al., 2015), and it is the reason very deep ReLU networks became trainable. The paper demonstrated a 30-layer network that trained with He initialization and completely failed with Xavier.

**The rule, memorized:**

| Activation | Initialization | `Var(w)` | PyTorch |
| --- | --- | --- | --- |
| ReLU, Leaky ReLU, GELU | He / Kaiming | 2 / n\_in | `nn.init.kaiming_normal_(w, nonlinearity='relu')` |
| tanh, sigmoid | Xavier / Glorot | 2 / (n\_in + n\_out) | `nn.init.xavier_normal_(w)` |
| SELU | LeCun | 1 / n\_in | `nn.init.normal_(w, std=(1/n_in)**0.5)` |

**For convolutional layers, `n_in` is not the channel count.** A kernel of size K over `C_in` input channels has `n_in = K · K · C_in` incoming weights per output unit, and `n_out = K · K · C_out`. So `Conv2d(64, 128, 3)` has `n_in = 3·3·64 = 576`, giving He std = √(2/576) = 0.0589 — not √(2/64) = 0.177, which would be three times too large. PyTorch's `kaiming_normal_` computes this correctly for you; the point is to know what `2/n_in` means when you see it written down, and to get it right if you ever initialize by hand.

**Biases:** initialize to zero. They do not suffer the symmetry problem below, and zero is the neutral choice. (One exception: for the final layer of a heavily imbalanced binary classifier, initializing the output bias to `log(p/(1-p))` where p is the positive rate saves the model several epochs of learning the base rate.)

### 9.4 Why not zero, and why not large?

**All-zero weights break the network permanently.** If every weight in a layer is zero (or even just identical), every unit in that layer computes the same thing, receives the same gradient, and updates identically. They remain identical forever. A 512-unit layer with identical weights has the expressive power of one unit. This is the **symmetry breaking problem**, and it is why initialization must be random. Biases can be zero because the weights already break the symmetry.

**Large weights saturate and explode.** Large `Var(w)` pushes pre-activations into the flat tails of sigmoid/tanh, where `g' ≈ 0` and learning stops; with ReLU it causes exploding activations and, often, `nan`.

**Small weights vanish.** Too small and the signal dies out layer by layer, and section 9.1 applies.

### 9.5 Seeing it happen

Run this. It is the most convincing five seconds in this tutorial.

```python
import torch
import torch.nn as nn

def activation_stats(init_fn, depth=20, width=256, label=""):
    x = torch.randn(1000, width)
    stds = []
    for _ in range(depth):
        layer = nn.Linear(width, width, bias=False)
        init_fn(layer.weight)
        x = torch.relu(layer(x))
        stds.append(x.std().item())
    print(f"{label:22s} layer 1: {stds[0]:.4f}   "
          f"layer 10: {stds[9]:.6f}   layer 20: {stds[-1]:.8f}")

activation_stats(lambda w: nn.init.normal_(w, std=0.01),  label="std=0.01 (too small)")
activation_stats(lambda w: nn.init.xavier_normal_(w),     label="Xavier")
activation_stats(lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
                                                          label="He (correct for ReLU)")
activation_stats(lambda w: nn.init.normal_(w, std=1.0),   label="std=1.0 (too large)")
```

Typical output:

```
std=0.01 (too small)   layer 1: 0.0941   layer 10: 2.79e-10   layer 20: 7.87e-20
Xavier                 layer 1: 0.5845   layer 10: 2.27e-02   layer 20: 6.48e-04
He (correct for ReLU)  layer 1: 0.8227   layer 10: 8.66e-01   layer 20: 9.61e-01
std=1.0 (too large)    layer 1: 9.3545   layer 10: 2.52e+10   layer 20: 7.88e+20
```

(These are actual values from one run with a fixed seed; yours will differ in the last digits but not in the pattern.)

He initialization holds the signal roughly constant across 20 layers. Xavier decays by about a factor of 2 every 3 layers because it does not account for ReLU's halving. The other two are catastrophic. If the forward signal behaves like this, so does the backward gradient — the same variance argument applies in reverse.

### 9.6 PyTorch defaults, and when to override them

PyTorch's `nn.Linear` and `nn.Conv2d` default to **Kaiming uniform with `a=√5`**, which works out to a variance of `1/(3·n_in)`. This is a historical default that is a bit conservative for ReLU networks but works acceptably. For a deep network trained from scratch, set initialization explicitly:

```python
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)      # applies recursively to every submodule
```

`mode='fan_out'` uses `n_out` instead of `n_in`, preserving variance in the backward pass rather than the forward pass. For most networks the difference is minor; `fan_out` is what the original ResNet implementation used.

A trick used in modern ResNets: initialize the **last BatchNorm gamma in each residual block to zero**. Each block then starts as an exact identity function, and the network begins as a shallow one and deepens itself during training. This reliably improves results on very deep models and costs one line.

### 9.7 Other defences against bad gradient flow

Initialization is the first line. Four more, each covered elsewhere:

1. **Non-saturating activations.** ReLU and friends (section 3).
2. **Batch normalization.** Re-normalizes activations at every layer, so the network cannot drift far from a healthy scale even as weights change during training (section 11). This makes results far less sensitive to initialization.
3. **Residual connections.** `y = x + F(x)` gives the gradient a path with derivative exactly 1 (section 18.5). This is the single most effective fix for depth.
4. **Gradient clipping.** A direct guard against explosion:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

If the total gradient norm exceeds `max_norm`, the whole gradient vector is rescaled to that norm, preserving direction while capping magnitude. Standard for RNNs and Transformers; rarely needed for CNNs. Note it goes **after** `backward()` and **before** `step()`.

### Check yourself

1. A layer has 512 inputs and 256 outputs with ReLU. What standard deviation should its weights have under He initialization? Under Xavier?
2. Why can biases be initialized to zero when weights cannot?
3. Batch normalization makes networks less sensitive to initialization. Does that mean initialization no longer matters?
4. Your gradients explode in a Transformer at step 200. What do you try, in what order?

*(Answers: 1 — He: std = √(2/512) = 0.0625. Xavier: std = √(2/(512+256)) = √(2/768) = 0.0510. 2 — the weights being random already breaks the symmetry between units, so identical biases do not cause identical behaviour; each unit still computes a different function of its inputs. 3 — no. BatchNorm reduces sensitivity but does not eliminate it, very deep networks still train faster and more reliably with correct initialization, and normalization-free architectures depend on it entirely. It also matters for the layers BatchNorm does not cover. 4 — (i) add gradient clipping at max\_norm 1.0, (ii) add or lengthen learning-rate warmup, which is standard for Transformers precisely because early steps are unstable, (iii) lower the peak learning rate, (iv) check for a bad batch with extreme values or a label error, (v) verify the normalization layers are placed as the architecture intends — pre-norm rather than post-norm is markedly more stable.)*

## 10. Optimizers: Better Ways to Use the Gradient

Plain gradient descent treats every parameter identically and has no memory. Both are fixable, and the fixes are what modern optimizers are.

Throughout, `g_t = ∇J(θ_t)` is the gradient at step t.

### 10.1 The problem optimizers solve

Picture a long, narrow ravine: the loss drops steeply across the ravine and gently along it. Plain SGD produces a distinctive pathology. The gradient is dominated by the steep cross-ravine direction, so steps bounce back and forth across the walls while making tiny progress along the floor. Raise the learning rate and it diverges across the ravine; lower it and progress along the floor stops entirely.

Formally, this is **ill-conditioning**: the ratio of the largest to smallest Hessian eigenvalue is large. Every optimizer below is an attempt to make progress in flat directions without exploding in steep ones.

### 10.2 Momentum

Accumulate an exponentially decaying average of past gradients and step in that direction:

```
v_t = β v_{t-1} + g_t
θ_t = θ_{t-1} - η v_t              (β typically 0.9)
```

(PyTorch uses this form. Some texts write `v_t = βv_{t-1} + (1-β)g_t`; the two differ by a constant factor absorbed into η.)

**The physical picture.** `v` is velocity, `g` is force, `β` is friction. A ball rolling down the ravine accumulates speed along the consistent downhill direction, while the oscillating cross-ravine forces cancel out against each other. Momentum damps oscillation and accelerates consistent motion — exactly what the ravine needs.

**The effective step size.** If the gradient were constant at g, the velocity converges to a geometric series:

```
v_∞ = g(1 + β + β² + ...) = g / (1 - β)
```

With β = 0.9, that is 10g. **Momentum multiplies your effective learning rate by `1/(1-β)` in consistent directions.** This is why switching from plain SGD to SGD+momentum usually requires reducing η by roughly 10x. Forgetting this is a common cause of "I added momentum and it diverged."

**Nesterov momentum** evaluates the gradient at the point the momentum is about to take you, rather than where you are:

```
θ_lookahead = θ_{t-1} - ηβ v_{t-1}
v_t = β v_{t-1} + ∇J(θ_lookahead)
θ_t = θ_{t-1} - η v_t
```

It is a correction term that lets the optimizer "see" an approaching wall and brake early. Usually a small improvement; free to enable with `nesterov=True`.

### 10.3 AdaGrad: per-parameter learning rates

```
G_t = G_{t-1} + g_t²                    (elementwise square, accumulated)
θ_t = θ_{t-1} - η · g_t / (√G_t + ε)
```

Parameters that have received large gradients get small effective learning rates and vice versa. This is genuinely useful for sparse features — a rare word's embedding gets few but important updates, and AdaGrad gives them full weight.

**The fatal flaw:** `G_t` only ever increases, so the effective learning rate decays monotonically toward zero. On long training runs AdaGrad grinds to a halt before converging. Every method below exists to fix this.

### 10.4 RMSProp: forget the distant past

Replace the running sum with an exponentially decaying average:

```
E_t = ρ E_{t-1} + (1-ρ) g_t²            (ρ typically 0.9)
θ_t = θ_{t-1} - η · g_t / (√E_t + ε)
```

Now old gradients decay away and the effective learning rate can recover. `√E_t` approximates the root-mean-square of recent gradient magnitudes for that parameter, so **each parameter is updated by roughly ±η regardless of its gradient scale.** The optimizer becomes insensitive to how the loss is scaled.

RMSProp was never formally published — Hinton introduced it in a Coursera lecture, which everyone cites.

### 10.5 Adam: momentum plus RMSProp

Adam (Kingma & Ba, 2015) combines the first moment (momentum) and the second moment (RMSProp):

```
m_t = β₁ m_{t-1} + (1-β₁) g_t              first moment  (mean)
v_t = β₂ v_{t-1} + (1-β₂) g_t²             second moment (uncentred variance)

m̂_t = m_t / (1 - β₁^t)                      bias correction
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

**Why bias correction is needed — derive this once.** Initialize `m₀ = 0`. Unroll:

```
m₁ = (1-β₁) g₁
m₂ = β₁(1-β₁)g₁ + (1-β₁)g₂
m_t = (1-β₁) Σ_{i=1}^{t} β₁^(t-i) g_i
```

If the gradients are drawn from a distribution with mean `E[g]`, then

```
E[m_t] = (1-β₁) E[g] Σ_{i=1}^{t} β₁^(t-i) = E[g](1 - β₁^t)
```

using the finite geometric sum. So `m_t` **underestimates** the true mean by the factor `(1 - β₁^t)`. At t = 1 with β₁ = 0.9, `m₁ = 0.1·g₁` — ten times too small. Dividing by `(1 - β₁^t)` corrects it exactly. The correction matters most in the first few dozen steps and becomes negligible as `β₁^t → 0`.

It is tempting to say the second moment matters far more, since `1 - β₂¹ = 0.001` makes `v₁` a thousand times too small. But be precise: **both** moments are biased at t=1, and they partly cancel. Uncorrected, the first step is

```
m₁/√v₁ = (0.1·g) / √(0.001·g²) = 0.1/0.0316 ≈ 3.16
```

about **3.2x larger** than the intended ±η, not a thousand times. So bias correction is a real fix but not a rescue from catastrophe — and the residual distortion persists for tens of steps, decaying at different rates for the two moments (β₁ᵗ dies in \~40 steps, β₂ᵗ takes \~4000). That asymmetry is the actual reason correction is worth doing, and it is also part of why warmup helps Adam: the second-moment estimate is simply unreliable early on.

**Why Adam works.** The update `m̂/√v̂` is roughly a signal-to-noise ratio. When gradients are consistent, `m̂` is large relative to `√v̂` and the step is near ±η. When gradients are inconsistent and noisy, `m̂` is small relative to `√v̂` and the step shrinks automatically. The optimizer becomes cautious exactly when it should.

### 10.6 AdamW: the weight decay fix

This distinction appears in interviews and in every modern training script.

L2 regularization adds `(λ/2)||θ||²` to the loss, so the gradient gains a `λθ` term. In plain SGD that is identical to weight decay: `θ ← θ - η(g + λθ) = (1 - ηλ)θ - ηg`.

In Adam it is **not** identical. The `λθ` term goes into the gradient, which means it also flows into `m` and `v`, and is then divided by `√v̂`. Parameters with large historical gradients get their decay divided down; parameters with small gradients get it amplified. **The regularization strength ends up depending on the gradient history, which is not what anyone intended.**

AdamW (Loshchilov & Hutter, 2019) decouples them by applying decay directly to the parameters, outside the adaptive machinery:

```
θ_t = θ_{t-1} - η · m̂_t/(√v̂_t + ε) - η · λ · θ_{t-1}
                └── adaptive part ──┘   └─ decoupled decay ─┘
```

In PyTorch, `torch.optim.Adam(weight_decay=...)` gives you the coupled L2 version; `torch.optim.AdamW(weight_decay=...)` gives the decoupled one. **AdamW is the right default** and is what essentially all modern Transformer and vision training uses.

A related practice: do **not** apply weight decay to biases or to normalization parameters. They have too few parameters to overfit and shrinking them distorts the normalization.

```python
decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim <= 1 or name.endswith(".bias"):     # biases, BN/LN weights
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.05},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-4)
```

### 10.7 Which one should you use?

| Optimizer | Best for | Typical η | Notes |
| --- | --- | --- | --- |
| SGD | Rarely alone | — | Baseline |
| SGD + momentum | CNNs, image classification | 0.1 with BN | Often the best final accuracy; needs schedule tuning |
| RMSProp | RNNs, some RL | 1e-3 | Largely superseded by Adam |
| Adam | Default for most new work | 1e-3 to 3e-4 | Fast, forgiving, converges quickly |
| AdamW | Transformers, modern vision | 1e-3 to 1e-4, wd 0.01-0.1 | Best-practice default |

The useful summary: **Adam/AdamW gets you to a good result fast and with little tuning. SGD with momentum, a well-tuned schedule, and patience often reaches a slightly better final test accuracy on image classification.** For coursework, research prototyping, and interviews, AdamW. For a leaderboard result on CIFAR or ImageNet, tuned SGD+momentum is still competitive.

There is an ongoing debate about whether Adam generalizes slightly worse than SGD; the effect is real but small and heavily dependent on schedule and regularization. Do not treat it as settled.

### 10.8 Learning rate schedules

A fixed learning rate is almost never optimal. You want large steps early (to travel) and small steps late (to settle).

**Step decay** — multiply by 0.1 at fixed epochs. The classic ResNet recipe.

```python
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.1)
```

**Cosine annealing** — smoothly decay following a half cosine from η\_max to \~0. The modern default; no milestones to tune.

```
η_t = η_min + 0.5(η_max - η_min)(1 + cos(π t / T))
```

```python
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
```

**Warmup** — start at a tiny learning rate and ramp up linearly over the first few hundred to few thousand steps. Essential for Transformers and for large-batch training, because early gradients are unreliable and adaptive optimizers have poor variance estimates before they have seen enough steps.

**One-cycle** (Smith, 2018) — ramp up to a high peak, then down to well below the starting value, with momentum moving inversely. Trains remarkably fast.

```python
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs)
# note: OneCycleLR steps every BATCH, not every epoch
```

**Reduce on plateau** — drop the learning rate when validation loss stops improving. Adaptive, requires no schedule guess.

```python
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
# call sched.step(val_loss) once per epoch, after validation
```

**The single most common scheduler bug:** calling `scheduler.step()` in the wrong place. Most schedulers step once per **epoch**, but `OneCycleLR` and warmup schedulers step once per **batch**, and `ReduceLROnPlateau` needs the validation metric passed in. Getting this wrong silently changes your entire schedule. Print `optimizer.param_groups[0]['lr']` each epoch and confirm it does what you expect.

### 10.9 A complete, modern setup

```python
import torch

model = MyModel().to(device)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4,
    steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)     # slightly faster than zeroing
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()                          # per batch for OneCycleLR
```

### Check yourself

1. You switch from SGD (η=0.1) to SGD with momentum 0.9 and the loss diverges. Why, and what η should you try?
2. Why does Adam need bias correction but plain momentum usually does not bother?
3. What is the practical difference between `Adam(weight_decay=0.01)` and `AdamW(weight_decay=0.01)`?
4. Your Transformer's loss spikes to `nan` in the first 100 steps. Which schedule feature is most likely missing?

*(Answers: 1 — momentum multiplies the effective step by about 1/(1-β) = 10 in consistent directions, so η=0.1 behaves like η=1.0. Try η=0.01, or keep 0.1 and lower β. 2 — Adam divides by `√v̂`, and at t=1 an uncorrected `v` with β₂=0.999 is 1000x too small, making `√v̂` about 32x too small and the first step catastrophically large; momentum's bias only makes early steps too small, which is harmless. 3 — `Adam` adds `λθ` to the gradient, so the decay passes through the adaptive scaling and its effective strength varies per parameter with gradient history; `AdamW` subtracts `ηλθ` directly, giving uniform, predictable decay. AdamW generalizes better and its weight-decay value transfers across learning rates. 4 — linear warmup. Transformers are notoriously unstable in the first few hundred steps, and warmup over 500-4000 steps is the standard fix; gradient clipping helps too.)*

## 11. Normalization Layers

You standardize your input features so no single feature dominates. Normalization layers do the same thing to the *hidden* activations, at every layer, during training. It is one of the highest-value-per-line techniques in deep learning.

### 11.1 Batch normalization: the mechanism

For a mini-batch of m examples, for each feature (or each channel in a CNN):

```
μ_B = (1/m) Σᵢ xᵢ                          batch mean
σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²                batch variance
x̂ᵢ  = (xᵢ - μ_B) / √(σ²_B + ε)             normalize: mean 0, variance 1
yᵢ  = γ x̂ᵢ + β                             scale and shift (LEARNED)
```

The first three lines standardize. The fourth is the part people skip over, and it is essential.

**Why γ and β?** Forcing every layer's output to mean 0, variance 1 is a serious constraint. With sigmoid it would pin activations to the near-linear region around zero, destroying the nonlinearity. `γ` and `β` are learned parameters (one pair per channel) that let the network choose its own scale and shift. If the optimal thing is no normalization at all, the network can learn `γ = √(σ²_B + ε)` and `β = μ_B` and recover the identity exactly. **Normalization never removes representational power; it only changes the parameterization.** That is the key point, and it is what makes BatchNorm safe to insert anywhere.

### 11.2 Train mode versus eval mode (the thing that bites people)

At training time BatchNorm uses the **current batch's** statistics. At inference time that would be wrong for two reasons: predictions would depend on which other examples happened to be in the batch, and a batch of size 1 has zero variance.

So BatchNorm maintains running estimates during training:

```
running_mean = (1 - momentum) · running_mean + momentum · μ_B
running_var  = (1 - momentum) · running_var  + momentum · σ²_B
```

(PyTorch's default `momentum=0.1` means the new batch gets weight 0.1. Note that this is the opposite convention from the momentum in optimizers — an unfortunate naming collision.)

At eval time it uses `running_mean` and `running_var`, so inference is deterministic and batch-size independent.

**This is why `model.eval()` matters and why forgetting it is such a common bug.** In `train()` mode during validation, BatchNorm normalizes each validation batch by its own statistics, which leaks information across examples, gives batch-size-dependent results, and makes your validation accuracy meaningless. Symptoms: validation accuracy changes when you change the validation batch size; validation is worse than training in a way that regularization cannot explain; single-image inference is wildly wrong.

`model.eval()` also disables dropout. `model.train()` turns both back on. These two calls belong in every training loop.

### 11.3 Why does BatchNorm help?

The original paper (Ioffe & Szegedy, 2015) attributed it to reducing "internal covariate shift" — the idea that as earlier layers change, the input distribution to later layers keeps shifting, forcing them to constantly re-adapt.

**That explanation is now widely doubted.** Santurkar et al. (2018) showed that you can deliberately inject noise after BatchNorm to *increase* covariate shift and the network still trains well and fast. The current best explanations:

1. **It smooths the loss landscape.** BatchNorm makes the loss and its gradients more Lipschitz — changes are bounded and predictable — so larger learning rates are stable and optimization is easier. This is the explanation Santurkar et al. support empirically.
2. **It decouples weight magnitude from the function computed.** Because the output is normalized, scaling `W` by any constant c leaves the layer's output unchanged. The gradient with respect to W scales as `1/c`, so large weights automatically get small effective learning rates. This self-stabilizing property is real and easy to verify.
3. **It regularizes.** Each example's normalization depends on the other examples in its batch, which injects noise, similar in spirit to dropout. This is why heavily batch-normalized networks often need less dropout.

Be careful in interviews: quoting "it reduces internal covariate shift" as the accepted explanation marks you as having read only the abstract. Say that it was the original motivation and that the evidence now points to loss-landscape smoothing.

**What it buys you in practice:** you can use learning rates 10x higher, training converges in far fewer epochs, results are much less sensitive to initialization, and you get a mild regularization effect for free.

### 11.4 Where to put it

The original paper placed it between the linear operation and the activation:

```
Conv → BatchNorm → ReLU
```

This remains the dominant convention and is what ResNet uses. An alternative order (`BN → ReLU → Conv`, "pre-activation") was proposed in ResNet-v2 and works slightly better for very deep networks.

One detail: **omit the bias in a layer immediately followed by BatchNorm.**

```python
nn.Conv2d(64, 128, 3, padding=1, bias=False)    # bias=False
nn.BatchNorm2d(128)
nn.ReLU(inplace=True)
```

BatchNorm subtracts the mean, which cancels any constant bias, and then adds its own learnable `β`. The conv bias is exactly redundant — it costs parameters and does nothing. Spotting this in someone's code is a nice signal of understanding.

### 11.5 LayerNorm, GroupNorm, InstanceNorm

They all compute `(x - mean)/std` then scale and shift. They differ **only in which axes the statistics are computed over.** For an activation tensor of shape (N, C, H, W):

| Normalization | Statistics over | Depends on batch? | Typical use |
| --- | --- | --- | --- |
| BatchNorm | N, H, W (per channel) | **Yes** | CNNs with decent batch size |
| LayerNorm | C, H, W (per example) | No | Transformers, RNNs |
| InstanceNorm | H, W (per example per channel) | No | Style transfer, GANs |
| GroupNorm | H, W and a group of channels | No | Small-batch vision, detection |

**BatchNorm's weakness is small batches.** With batch size 2 or 4 the batch statistics are noisy estimates, and performance degrades badly. This is common in object detection and segmentation, where high-resolution images force small batches. GroupNorm (Wu & He, 2018) fixes it by computing statistics over groups of channels within a single example — independent of batch size entirely. `nn.GroupNorm(num_groups=32, num_channels=C)` is a reasonable default.

**LayerNorm normalizes across features within one example**, so each example is normalized independently. That is what sequence models need: sequences have variable length, batch statistics across padded positions are meaningless, and inference on a single sequence must behave the same as in a batch. Every Transformer uses LayerNorm.

Why does BatchNorm normalize per channel in a CNN? Because a channel is a single learned feature detector applied at every spatial position. All `N·H·W` of its responses are samples of the same feature, so pooling statistics over them is the natural choice — and it keeps the number of `γ`, `β` parameters equal to the number of channels rather than the number of spatial positions.

### 11.6 Interactions worth knowing

**BatchNorm and dropout together can hurt.** Li et al. (2019) analysed a "variance shift": dropout changes the variance of activations between training and inference, and BatchNorm's running statistics, estimated with dropout active, then mismatch at test time. Modern CNNs typically use BatchNorm with little or no dropout in convolutional layers, keeping dropout for the final classifier if at all.

**BatchNorm and weight decay.** Do not apply weight decay to `γ` and `β` (section 10.6). Shrinking γ toward zero directly suppresses the layer's output for no good reason.

**BatchNorm and batch composition.** If your batches are not shuffled — say all class 0 in one batch, all class 1 in the next — batch statistics differ wildly from batch to batch and training destabilizes. Always shuffle (`shuffle=True` in the DataLoader).

**BatchNorm and fine-tuning.** When fine-tuning a pretrained model on a small dataset, the running statistics from the source dataset may be better than noisy estimates from your small batches. A common trick is to freeze BatchNorm layers in eval mode during fine-tuning (section 20).

### 11.7 In code

```python
import torch
import torch.nn as nn

# A standard conv block
block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(64),        # num_features = num channels
    nn.ReLU(inplace=True),
)

# Inspect what BatchNorm holds
bn = block[1]
print(bn.weight.shape)          # gamma: torch.Size([64])
print(bn.bias.shape)            # beta:  torch.Size([64])
print(bn.running_mean.shape)    # buffer, not a parameter: torch.Size([64])
print(bn.running_var.shape)     # buffer: torch.Size([64])

# Demonstrate train vs eval
x = torch.randn(8, 3, 32, 32)
block.train(); out_train = block(x)
block.eval()
with torch.no_grad():
    out_eval = block(x)
print((out_train - out_eval).abs().max())   # clearly nonzero

# For MLPs use BatchNorm1d
mlp_block = nn.Sequential(
    nn.Linear(784, 256, bias=False),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
)

# For Transformers / sequences
ln = nn.LayerNorm(512)          # normalizes over the last dimension

# For small-batch vision
gn = nn.GroupNorm(num_groups=32, num_channels=256)
```

Note that `running_mean` and `running_var` are **buffers**, not parameters: they are saved in `state_dict()` and moved with `.to(device)`, but the optimizer never updates them and they receive no gradient. If you load a checkpoint and accuracy is terrible, check that the buffers loaded — a common casualty of `strict=False`.

### Check yourself

1. A BatchNorm2d layer follows a conv with 128 output channels. How many learnable parameters does it have? How many buffered values?
2. Why does a network with BatchNorm tolerate a much higher learning rate?
3. You train with batch size 64 and validate with batch size 1. Validation accuracy is near chance. What is wrong?
4. Why is LayerNorm used in Transformers rather than BatchNorm?

*(Answers: 1 — 256 learnable (128 γ + 128 β) and 256 buffered (128 running means + 128 running variances). 2 — normalization keeps activations at a controlled scale regardless of weight magnitude, and the loss surface becomes smoother and more predictably curved, so larger steps remain stable; also, scaling W up scales the gradient down by the same factor, which self-limits the effective step. 3 — almost certainly `model.eval()` was not called, so BatchNorm normalizes a batch of one by its own statistics: the output becomes `γ·0 + β` for every feature, identical for every input, and the prediction is constant. 4 — sequences have variable length and batch statistics over padded positions are meaningless; batch sizes in NLP are often small; and inference must be identical for a single sequence and for a batch. LayerNorm normalizes within each example, so none of these problems arise.)*

## 12. Regularization and Generalization

A model that fits its training data perfectly and fails on new data is useless. Regularization is everything we do to close that gap.

### 12.1 The bias-variance picture, and why deep learning complicates it

Classical statistics decomposes expected test error into three parts:

```
Expected test error = Bias² + Variance + Irreducible noise
```

- **Bias**: error from wrong assumptions. A linear model on curved data has high bias. This is underfitting.
- **Variance**: sensitivity to the particular training sample. A model that would change drastically with a different draw of data has high variance. This is overfitting.

The classical story says model complexity trades one against the other, giving a U-shaped test error curve with a sweet spot in the middle.

**Deep learning does not follow this curve, and you should know why.** Modern networks routinely have far more parameters than training examples and still generalize well. The phenomenon has a name: **double descent** (Belkin et al., 2019). As you increase model capacity, test error first follows the classical U — down, then up, peaking at the **interpolation threshold** where the model has just enough capacity to fit the training data exactly. Then, as capacity grows further, test error **descends again**, often to below the classical minimum.

The working explanation is about *implicit regularization*: among the many parameter settings that fit the training data, gradient descent tends to find ones with small norm and simple structure, and over-parameterization gives it more such solutions to choose from. This is an active research area, not settled theory, but the practical consequence is firm: **do not be afraid of large models.** Regularize them; do not shrink them.

### 12.2 L2 regularization / weight decay

Add a penalty on weight magnitude to the loss:

```
J_reg = J + (λ/2) Σ ||W||²
```

The gradient gains a `λW` term, so the update becomes

```
W ← W - η(∂J/∂W + λW) = (1 - ηλ)W - η ∂J/∂W
```

Every step shrinks W by a factor `(1 - ηλ)` before the data-driven update — hence "weight decay."

**Why smaller weights generalize better.** Section 2.2 showed that `||w||` controls how *sharply* a neuron transitions across its decision boundary, not where the boundary is. Small weights mean gentler transitions and a smoother overall function, which is less able to contort itself around individual noisy training points.

There is also a Bayesian reading: L2 regularization is exactly the MAP estimate under a zero-mean Gaussian prior on the weights, with `λ = 1/σ²_prior`. Strong regularization = a confident prior that weights are near zero.

Typical λ: 1e-4 for SGD on CNNs, 0.01 to 0.1 for AdamW on Transformers (the scale differs because AdamW decouples it; see 10.6). And remember: apply it to weights only, not biases or normalization parameters.

### 12.3 L1 regularization

```
J_reg = J + λ Σ |W|
```

The gradient is `λ·sign(W)` — a constant push toward zero regardless of magnitude. L2's push is proportional to W, so it shrinks large weights hard and small weights gently, never quite reaching zero. L1's constant push drives small weights **exactly** to zero.

So L1 produces **sparse** models, which is useful for feature selection and for compression. It is much less common than L2 in deep learning, partly because sparsity in a dense weight matrix does not speed anything up without specialized hardware.

### 12.4 Dropout

During training, randomly zero each unit's output with probability p, independently for each example and each forward pass.

```
Training:   mask ~ Bernoulli(1-p)
            a = (a ⊙ mask) / (1-p)      ← inverted dropout
Inference:  a = a                       ← nothing happens
```

**Why divide by (1-p)?** Without it, the expected activation during training is `(1-p)·a` but at inference it is `a`, a mismatch in scale that every downstream layer would be miscalibrated for. Dividing by `(1-p)` during training restores the expectation: `E[(a·mask)/(1-p)] = (1-p)·a/(1-p) = a`. This is **inverted dropout**, and it is what all frameworks implement, because it keeps inference free of any special handling.

**Why it works.** Three complementary explanations, all worth having:

1. **It prevents co-adaptation.** A unit cannot rely on any particular other unit being present, so it must learn a feature that is useful on its own rather than only as part of a fragile conspiracy.
2. **It is an implicit ensemble.** A network with n units has 2^n possible dropout masks. Training with dropout approximately trains this exponential ensemble with shared weights, and inference with the full network approximates averaging over it.
3. **It is noise injection.** Like adding noise to inputs, it forces the learned function to be locally smooth.

**Practical guidance:** p = 0.5 for fully connected layers (the original recommendation), p = 0.1-0.3 for convolutional layers if at all. Place dropout **after** the activation. Modern CNNs often skip dropout entirely in conv layers, relying on BatchNorm and augmentation, and apply it only before the final classifier. In Transformers, dropout of 0.1 is standard throughout.

```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),          # after the activation
    nn.Linear(256, 10),
)
```

`model.eval()` disables dropout. Again: if you forget it, validation numbers are garbage.

**Variants:** `nn.Dropout2d` (now `nn.Dropout1d/2d/3d`) drops entire channels rather than individual activations, which makes more sense for convolutional feature maps where neighbouring pixels are highly correlated — dropping single pixels leaves the information recoverable from its neighbours. **DropPath / stochastic depth** drops whole residual branches and is standard in modern deep vision models.

### 12.5 Early stopping

Monitor validation loss; keep the parameters from the best epoch.

```python
best_val, patience, wait = float('inf'), 10, 0
for epoch in range(max_epochs):
    train_one_epoch(...)
    val_loss = evaluate(...)
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
model.load_state_dict(torch.load('best.pt'))    # restore the best, not the last
```

It costs nothing, requires no hyperparameter beyond patience, and it approximately implements L2 regularization: stopping after T steps of gradient descent from a small initialization limits how far the weights can travel, which bounds their norm.

**Do not early-stop on the test set.** Selecting a checkpoint by test performance makes that test score optimistic. Use a validation split.

### 12.6 Data augmentation — usually the highest-leverage option

Create new training examples by applying label-preserving transformations. For images this is enormously effective, because the label of a cat photo is unchanged by flipping, cropping, or brightness changes, while the pixel values change completely.

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],      # ImageNet statistics
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),                     # after ToTensor
])

val_tf = transforms.Compose([          # NO augmentation on validation/test
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Think about whether each transform preserves your label.** Horizontal flip is fine for cats; it is wrong for digit recognition (a flipped 2 is not a 2) and for reading text. Vertical flip is fine for satellite imagery, wrong for street scenes. Rotation is fine for microscopy, questionable for 6 vs 9. Colour jitter is wrong if colour is the distinguishing feature.

**Stronger modern methods:**

- **Mixup** (Zhang et al., 2018): train on convex combinations of pairs, `x̃ = λxᵢ + (1-λ)xⱼ` with the same mixing applied to the one-hot labels. Surprisingly effective and simple.
- **CutMix** (Yun et al., 2019): paste a rectangular patch of one image into another and mix the labels in proportion to area.
- **RandAugment / TrivialAugment**: apply randomly chosen operations from a fixed pool with random magnitude, with essentially no hyperparameters to tune. `transforms.RandAugment()` and `transforms.TrivialAugmentWide()` are built into torchvision.

**Test-time augmentation (TTA)**: at inference, average predictions over several augmented versions of the test image. Typically buys a fraction of a percent in accuracy for a multiple of the inference cost.

### 12.7 Label smoothing

Covered briefly in 5.8. Replacing hard targets with `1-ε` for the true class and `ε/(K-1)` elsewhere stops the model from driving logits to infinity, improves calibration (predicted probabilities better match actual accuracy), and typically gives a small accuracy gain. `nn.CrossEntropyLoss(label_smoothing=0.1)`.

### 12.8 Choosing what to use

Diagnose first, then act. The training/validation gap tells you which problem you have.

| Symptom | Diagnosis | What to do, in order |
| --- | --- | --- |
| Train loss high, val loss high | Underfitting | Bigger model; train longer; higher η; less regularization; better features |
| Train loss low, val loss high | Overfitting | More data; augmentation; weight decay; dropout; early stopping; smaller model |
| Train ≈ val, both acceptable | Good fit | Try scaling up; and check against baselines (22.8) that "acceptable" really is |
| Val loss < train loss | Usually normal | Dropout and augmentation penalize training but not validation; only worry if the gap is large |

**Priority order when overfitting**, roughly by impact per unit of effort:

1. **Get more data.** Nothing else comes close.
2. **Data augmentation.** The cheapest substitute for more data.
3. **Transfer learning** from a pretrained model (section 20). Often transformative on small datasets.
4. **Weight decay.** One number to tune.
5. **Early stopping.** Free.
6. **Dropout.** Effective in fully connected layers.
7. **Reduce model size.** Last resort — the double-descent literature suggests this is often the wrong instinct.

### Check yourself

1. With dropout p=0.5 on a layer of 100 units, how many units are active on average, and what is the scaling factor applied during training?
2. Why is horizontal-flip augmentation a bad idea for MNIST?
3. Your model has train accuracy 99.8% and validation accuracy 71%. You have 2,000 training images. Rank your next three actions.
4. Explain why early stopping acts like L2 regularization.

*(Answers: 1 — 50 active on average; activations are divided by (1-p) = 0.5, i.e. multiplied by 2. 2 — digits are not flip-invariant: a mirrored 2, 5, or 7 is not a valid digit, and a mirrored 3 is not a 3. You would be training the model to assign the label "3" to something that never appears in the test set, wasting capacity and blurring the class boundaries. 3 — (i) transfer learning from an ImageNet-pretrained backbone, which is by far the biggest win at 2,000 images; (ii) aggressive data augmentation, including RandAugment and mixup; (iii) weight decay plus early stopping. Collecting more data would beat all three if it is possible. 4 — starting from small weights, each gradient step moves the parameters a bounded distance, so stopping after T steps bounds `||W||`; limiting the norm is precisely what L2 does. The correspondence is exact for linear models with quadratic loss and approximate in general.)*

## 13. PyTorch Essentials and Your First Complete Project

Time to build something end to end. This section covers the PyTorch you need, then trains an MLP on Fashion-MNIST with every line explained.

### 13.1 Tensors

A tensor is an n-dimensional array that can live on a GPU and remember how it was computed.

```python
import torch

# Creation
a = torch.tensor([[1., 2.], [3., 4.]])       # from data
b = torch.zeros(3, 4)
c = torch.ones(2, 3)
d = torch.randn(2, 3)                         # standard normal
e = torch.arange(0, 10, 2)                    # tensor([0, 2, 4, 6, 8])
f = torch.linspace(0, 1, 5)

# Attributes you will check constantly
print(a.shape, a.dtype, a.device, a.requires_grad)
# torch.Size([2, 2]) torch.float32 cpu False

# Reshaping
x = torch.randn(2, 3, 4)
print(x.view(6, 4).shape)        # torch.Size([6, 4]) - needs contiguous memory
print(x.reshape(6, 4).shape)     # same, but copies if needed. Safer.
print(x.view(2, -1).shape)       # torch.Size([2, 12]) - -1 is inferred
print(x.permute(2, 0, 1).shape)  # torch.Size([4, 2, 3]) - reorder dimensions
print(x.unsqueeze(0).shape)      # torch.Size([1, 2, 3, 4]) - add a dimension
print(x.squeeze().shape)         # remove all size-1 dimensions

# Reductions: the dim argument is what people get wrong
y = torch.randn(3, 4)
print(y.sum())                   # scalar: sum of everything
print(y.sum(dim=0).shape)        # torch.Size([4]) - collapses rows
print(y.sum(dim=1).shape)        # torch.Size([3]) - collapses columns
print(y.sum(dim=1, keepdim=True).shape)   # torch.Size([3, 1])
```

**The rule for `dim`:** the dimension you name is the one that disappears. `dim=0` on a `(3,4)` tensor removes the 3 and leaves `(4,)`. `keepdim=True` keeps it as size 1, which is essential when you need to broadcast the result back against the original.

**Broadcasting** follows NumPy's rules: align shapes from the right; a dimension of size 1 stretches to match; missing leading dimensions are treated as 1.

```python
A = torch.randn(32, 10)          # batch of 32, 10 features
b = torch.randn(10)              # one bias per feature
print((A + b).shape)             # (32, 10) - b is broadcast across the batch

row = torch.randn(1, 5)
col = torch.randn(4, 1)
print((row + col).shape)         # (4, 5) - both stretch. Often a bug!
```

That last case is worth flagging. Broadcasting is helpful until it silently turns a shape mistake into a valid but wrong computation. If a loss looks strange, print shapes.

**GPU:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
model = model.to(device)
# Both the model and the data must be on the same device, or you get:
# RuntimeError: Expected all tensors to be on the same device
```

### 13.2 Autograd: how PyTorch does section 6 for you

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)            # tensor(8.) -- dy/dx = 2x + 2 = 8 at x=3
```

What happened: every operation on a tensor with `requires_grad=True` records itself in a **dynamic computational graph**. `backward()` walks that graph in reverse applying the chain rule — exactly the algorithm of section 6 — and accumulates results into `.grad`.

"Dynamic" means the graph is built fresh on every forward pass, so you can use Python `if` statements and loops inside `forward()` and the graph will follow them. This is why PyTorch feels like ordinary Python.

```python
# Gradients ACCUMULATE. This is the reason for zero_grad().
x = torch.tensor(2.0, requires_grad=True)
for _ in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)        # tensor(4.), tensor(8.), tensor(12.) -- adding up

# Turning autograd off
with torch.no_grad():                 # no graph built: faster, less memory
    predictions = model(x_test)

y = x.detach()                        # a tensor sharing data but outside the graph

# Leaf vs non-leaf
a = torch.tensor(2.0, requires_grad=True)   # leaf: .grad is populated
b = a * 3                                   # non-leaf: .grad stays None
```

**Three autograd facts that cause real confusion:**

- `.backward()` can only be called on a **scalar**. For a non-scalar you must supply a gradient argument, which is why your loss must be reduced to a single number before calling it.
- The graph is **freed after `backward()`**. Calling it twice raises an error unless you pass `retain_graph=True` — and if you think you need that, you usually have a bug.
- `.item()` extracts a Python float and detaches from the graph. Accumulating `total_loss += loss` instead of `loss.item()` keeps the entire graph alive for every batch and will exhaust your memory. This is one of the most common memory leaks in beginner code.

### 13.3 Dataset and DataLoader

`Dataset` says how to get one example. `DataLoader` handles batching, shuffling, and parallel loading.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)                  # required

    def __getitem__(self, idx):                # required
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,          # True for training, False for val/test
    num_workers=4,         # parallel loading processes; 0 on Windows/notebooks if it hangs
    pin_memory=True,       # faster CPU->GPU transfer when using CUDA
    drop_last=False,       # True avoids a ragged final batch (matters for BatchNorm)
)
```

### 13.4 The full project: Fashion-MNIST

Fashion-MNIST is 70,000 28x28 grayscale images of clothing in 10 classes. It is a drop-in replacement for MNIST that is hard enough to be informative — MNIST is too easy to teach you anything.

```python
"""
Fashion-MNIST classification with an MLP.
Runs in ~2 minutes on a CPU, ~20 seconds on a GPU.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ----- 0. Reproducibility -------------------------------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ----- 1. Data ------------------------------------------------------------
# Normalize with the dataset's own mean and std so inputs are ~N(0,1).
# Zero-centred, unit-variance inputs make the first layer's job much easier
# (section 9) and interact well with He initialization.
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # clothing is roughly symmetric: safe
    transforms.ToTensor(),                  # -> float in [0,1], shape (1,28,28)
    transforms.Normalize((0.2860,), (0.3530,)),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

full_train = datasets.FashionMNIST(root="./data", train=True,
                                   download=True, transform=train_tf)
test_set   = datasets.FashionMNIST(root="./data", train=False,
                                   download=True, transform=test_tf)

# Hold out a validation set. NEVER tune on the test set.
train_set, val_set = random_split(full_train, [54_000, 6_000],
                                  generator=torch.Generator().manual_seed(42))

BATCH = 128
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=256,   shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=256,   shuffle=False, num_workers=2)

CLASSES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
           'Sandal','Shirt','Sneaker','Bag','Ankle boot']

# ----- 2. Model -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=(512, 256), n_classes=10, p_drop=0.3):
        super().__init__()
        layers, prev = [nn.Flatten()], in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h, bias=False),   # bias redundant before BatchNorm
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))  # logits, no softmax
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   # section 9.3
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")     # 536,586

# ----- 3. Loss, optimizer, schedule ---------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
EPOCHS = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ----- 4. Train / evaluate helpers ----------------------------------------
def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

            # .item() detaches: without it the graph is kept alive (13.2)
            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            n          += y.size(0)

    return total_loss / n, correct / n

# ----- 5. Training loop with early stopping -------------------------------
best_val_acc, patience, wait = 0.0, 5, 0
history = []

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    history.append((tr_loss, tr_acc, va_loss, va_acc))

    lr = optimizer.param_groups[0]['lr']
    print(f"epoch {epoch:2d} | lr {lr:.5f} | "
          f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

    if va_acc > best_val_acc:
        best_val_acc, wait = va_acc, 0
        torch.save(model.state_dict(), "best_mlp.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ----- 6. Test ONCE, with the best checkpoint -----------------------------
model.load_state_dict(torch.load("best_mlp.pt"))
te_loss, te_acc = run_epoch(test_loader, train=False)
print(f"\nBest val acc {best_val_acc:.4f} | TEST acc {te_acc:.4f}")

# ----- 7. Per-class accuracy: where is it failing? ------------------------
model.eval()
conf = torch.zeros(10, 10, dtype=torch.long)
with torch.no_grad():
    for x, y in test_loader:
        preds = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y, preds):
            conf[t, p] += 1

print("\nPer-class accuracy:")
for i, name in enumerate(CLASSES):
    print(f"  {name:12s} {conf[i, i].item() / conf[i].sum().item():.3f}")
```

**What you should see.** Test accuracy around 89-90%. Training loss falls smoothly; validation loss falls then flattens. Per-class accuracy is near 99% for Trouser, Sandal, Sneaker, Bag, and Ankle boot, and around 70-80% for Shirt — because shirts, T-shirts, pullovers, and coats look nearly identical at 28x28 resolution. That per-class breakdown tells you far more than the single aggregate number, which is why step 7 exists.

A plain MLP tops out near 90% here. A small CNN reaches 92-93% with fewer parameters. Section 14 explains exactly why.

### 13.5 A mental checklist for any training script

1. Fix the random seed. Reproducibility is not optional when debugging.
2. Normalize inputs with the training set's statistics.
3. Split off a validation set before doing anything else.
4. Print the parameter count. It catches architecture mistakes immediately.
5. Check that the initial loss ≈ ln(K) (section 5.4).
6. Overfit a tiny subset first (section 22).
7. `model.train()` and `model.eval()` in the right places.
8. `zero_grad()`, `backward()`, `step()` in that order.
9. Save the best checkpoint by validation metric, not the last.
10. Touch the test set once, at the very end.

### 13.6 Saving and loading properly

```python
# Save the state dict, not the model object (pickling a model object
# ties the file to your exact class definition and file layout).
torch.save(model.state_dict(), "model.pt")

model = MLP()                                    # recreate the architecture
model.load_state_dict(torch.load("model.pt"))
model.eval()                                     # ALWAYS before inference

# A full checkpoint for resuming training
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),   # Adam's moments live here
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
}, "checkpoint.pt")
```

Saving the optimizer state matters more than people expect: Adam's `m` and `v` estimates take hundreds of steps to warm up, and resuming without them causes a visible loss spike.

### Check yourself

1. Why `x.reshape(6,4)` rather than `x.view(6,4)` after a `permute`?
2. What goes wrong with `total_loss += loss` instead of `loss.item()`?
3. Your DataLoader has `shuffle=True` for the test set. What is the harm?
4. You get `RuntimeError: Expected all tensors to be on the same device`. What are the two things to check?

*(Answers: 1 — `permute` returns a non-contiguous view, and `view` requires contiguous memory, so it raises an error; `reshape` copies when necessary and always works. You could also call `.contiguous().view(...)`. 2 — `loss` is a graph-connected tensor, so accumulating it keeps every batch's computational graph in memory; memory grows linearly with the number of batches until you run out. 3 — no harm to the accuracy number itself, since accuracy is order-independent, but it makes per-example debugging, confusion matrices, and comparisons across runs harder to reproduce. Keep it False for determinism. 4 — whether `model.to(device)` was called, and whether every input tensor in the loop was moved with `.to(device)`; also check any tensor you created inside `forward()`, which lands on the CPU by default unless you pass `device=x.device`.)*

## 14. Why MLPs Fail on Images

The MLP in section 13 reached about 90% on 28x28 grayscale images. Scale that approach to real photographs and it collapses. Three reasons, and each one points directly at a design principle of convolutional networks.

### 14.1 Problem 1: the parameter explosion

Take a modest colour image: 224 x 224 x 3 = **150,528** input values. Connect it to a first hidden layer of 1,000 units:

```
Parameters in layer 1 = 150,528 x 1,000 + 1,000 = 150,529,000
```

**150 million parameters in a single layer.** For comparison, all of ResNet-50 — 50 layers, competitive on ImageNet — has 25 million. One fully connected layer would be six times the entire network.

The consequences compound:

- **Memory.** 150M parameters in float32 is 600 MB for the weights, plus the same again for gradients, plus twice more for Adam's two moment estimates. About 2.4 GB for one layer.
- **Data.** A rough rule of thumb is that you want at least as many examples as parameters to avoid severe overfitting. 150 million labelled images do not exist.
- **Compute.** 150M multiply-accumulates per image, per layer, per forward pass.

And the image at 224x224 is small. A modern phone camera produces 12-megapixel images.

### 14.2 Problem 2: spatial structure is destroyed

An MLP begins with `nn.Flatten()`. That single operation throws away everything the image's geometry was telling you.

After flattening a 28x28 image, pixel (5,5) becomes index 145 and pixel (6,5) — its vertical neighbour — becomes index 173. To the network, indices 145 and 173 are simply two of 784 unrelated numbers. Nothing tells it they are adjacent in space.

Here is the sharpest way to see the problem: **if you applied a fixed random permutation to all the pixels of every image in your dataset, an MLP would train to exactly the same accuracy.** It never used the spatial arrangement, so scrambling it costs nothing. That is a damning property for a vision model. Your eye cannot recognize a scrambled photograph; a well-designed vision model should not be indifferent to the difference.

Meanwhile, real images have strong local structure. Nearby pixels are highly correlated. Edges, textures, and corners are **local** patterns spanning a few pixels. Information is local first, and global only through composition. An architecture should exploit that.

### 14.3 Problem 3: no translation equivariance

Suppose an MLP has learned, in the weights feeding some hidden unit, a detector for a cat's eye in the top-left corner. Now a cat appears in the bottom-right of a different image.

Those are entirely different input indices, connected by entirely different weights. The network must **learn the same eye detector again, separately, for every position it could occur in.** With 224x224 positions, that is 50,176 independent copies of the same knowledge, each needing its own training examples.

This is a statistical disaster. The concept "an eye looks like this" is position-independent, and the architecture forces the model to discover it position by position.

### 14.4 The three ideas that fix all of this

Convolutional networks are the answer to these three problems, and each idea maps onto one problem.

**Idea 1: local connectivity (sparse interactions).** A hidden unit connects only to a small spatial neighbourhood of the input — say 3x3 or 5x5 — rather than to everything. This is justified by the locality of image structure (problem 2) and it slashes the parameter count (problem 1). A 3x3 connection to a single-channel input needs 9 weights, not 150,528.

The small window is called the **receptive field** of the unit. Stack layers and receptive fields grow: a unit two layers up sees a 5x5 region of the input, three layers up a 7x7 region, and so on. Locality at each layer, global coverage through depth. This directly implements the hierarchical composition idea of section 8.2.

**Idea 2: parameter sharing.** The *same* 3x3 filter is applied at every spatial position. One set of 9 weights, slid across the whole image.

This is the answer to problem 3. A horizontal-edge detector learned from examples in the top-left is automatically available everywhere, because it is literally the same weights. It also cuts parameters enormously: a layer producing 64 feature maps from a 3-channel input needs `3 x 3 x 3 x 64 = 1,728` weights regardless of whether the image is 32x32 or 4000x3000.

**Idea 3: translation equivariance, and then invariance.** Parameter sharing gives the convolution operation a precise mathematical property: `conv(shift(x)) = shift(conv(x))`. Shift the input, and the output shifts by the same amount. That is **equivariance** — the representation moves with the object.

Pooling and global pooling then convert equivariance into **invariance**: the final prediction becomes insensitive to position. "A cat is present" should not depend on where the cat is, and the architecture builds that in rather than requiring the model to learn it from data.

The distinction matters and appears in interviews. *Equivariance*: output changes correspondingly with the input. *Invariance*: output does not change. Convolution is equivariant; pooling adds local invariance; global pooling or a fully connected head produces overall invariance.

### 14.5 The savings, side by side

One layer mapping a 224x224x3 input to 64 feature maps:

|  | Fully connected | Convolutional (3x3) |
| --- | --- | --- |
| Weights | 150,528 × 3,211,264 ≈ 4.83 × 10¹¹ | 3 x 3 x 3 x 64 = 1,728 |
| Biases | 3,211,264 | 64 |
| Total | \~483 billion | **1,792** |
| Handles a different input size? | No | Yes |
| Translation equivariant? | No | Yes |

A reduction of roughly eight orders of magnitude, while *improving* the model's inductive bias. This is not a compromise for efficiency's sake; it is a better model of what images are.

### 14.6 The general lesson

Step back from vision for a moment, because this is the transferable idea.

The CNN works because it **encodes true facts about the data into the architecture.** Images really are locally structured, and object identity really is translation-independent. Rather than hoping the model discovers these facts from data, we build them in. That is an **inductive bias**, and a correct one buys you enormous statistical efficiency: the model needs far less data because it starts out knowing something true.

The same reasoning explains other architectures. Recurrent networks encode the assumption that sequences have temporal structure and that the same processing applies at each time step. Graph networks encode permutation invariance over nodes. Transformers encode almost the opposite bet: minimal built-in assumptions, letting attention learn the relationships from data — which works brilliantly when you have enough data to learn them, and worse when you do not. Section 21 returns to that trade-off, which is the central architectural argument of the past five years.

**When you design a model, ask: what do I know for certain about my data, and can I build it into the architecture?**

### Check yourself

1. An MLP with one hidden layer of 512 units takes 64x64 RGB images. Parameter count for that layer?
2. Explain why an MLP's accuracy is unchanged if you apply a fixed pixel permutation to your whole dataset, and why this is bad.
3. Is a convolutional layer invariant or equivariant to translation? Which one does a classifier ultimately need, and how does it get it?
4. Name a data type other than images where local connectivity and parameter sharing are appropriate, and say why.

*(Answers: 1 — 64·64·3 = 12,288 inputs, so 12,288 x 512 + 512 = 6,291,968, about 6.3 million for one layer. 2 — the MLP's first layer treats its inputs as an unordered set of features, with a separate weight for each, so relabelling the inputs consistently just permutes the columns of W and the model learns the permuted weights equally well. It is bad because it proves the model is ignoring the spatial structure that makes vision tractable, and therefore must learn from data what the architecture could have provided for free. 3 — a convolutional layer is equivariant; a classifier needs invariance, which it obtains from pooling, strided downsampling, and finally global average pooling or a flattening head. 4 — audio waveforms and spectrograms, where local temporal patterns like onsets and formants matter and the same pattern is meaningful at any point in time; also time-series sensor data, and 1-D convolutions over text, where n-gram-like patterns are position-independent.)*

## 15. The Convolution Layer in Detail

This is the section where CNNs stop being a metaphor and become arithmetic you can do by hand.

### 15.1 The operation, by hand

A **kernel** (or filter) is a small matrix of learnable weights. Slide it across the input; at each position compute the elementwise product with the patch beneath it and sum.

Input 5x5, kernel 3x3:

```
Input I                  Kernel K
1  2  3  0  1            1  0 -1
4  5  6  1  0            1  0 -1
7  8  9  0  2            1  0 -1
1  3  5  2  1
2  4  6  1  3
```

Output position (0,0): overlay K on the top-left 3x3 of I.

```
Patch:  1 2 3      K:  1  0 -1
        4 5 6          1  0 -1
        7 8 9          1  0 -1

Sum = 1(1) + 2(0) + 3(-1)
    + 4(1) + 5(0) + 6(-1)
    + 7(1) + 8(0) + 9(-1)
    = (1 - 3) + (4 - 6) + (7 - 9) = -6
```

Slide one step right and repeat. The full 3x3 output:

```
 -6  14  15
 -8  13  17
-10  12  14
```

This particular kernel computes (left column − right column), so it responds to **vertical edges** — it is a Prewitt edge detector (Sobel is the same idea with the middle row weighted by 2). In a CNN nobody specifies this kernel; gradient descent discovers it, and empirically the first layer of a trained CNN is full of edge detectors at various orientations, which is a striking confirmation that the architecture is doing what we hoped.

**Formally:**

```
O(i,j) = Σ_m Σ_n I(i+m, j+n) · K(m,n)
```

### 15.2 Convolution vs cross-correlation (a pedantic but interview-relevant point)

Strict mathematical convolution flips the kernel:

```
(I * K)(i,j) = Σ_m Σ_n I(i-m, j-n) · K(m,n)          true convolution
(I ⋆ K)(i,j) = Σ_m Σ_n I(i+m, j+n) · K(m,n)          cross-correlation
```

**Every deep learning framework implements cross-correlation and calls it convolution.** PyTorch's `nn.Conv2d` does not flip the kernel.

Does it matter? Not for learning: the kernel weights are learned, so the network simply learns the flipped version of whatever kernel true convolution would have used, and the resulting function class is identical. It matters if you are comparing against a signal-processing implementation, or implementing convolution's backward pass by hand (section 17), where a genuine flip appears.

If an interviewer asks, the answer is: frameworks use cross-correlation; it makes no difference because the weights are learned; the flip reappears in the backward pass.

### 15.3 Padding

Without padding, a 5x5 input and 3x3 kernel gives a 3x3 output. Every layer shrinks the feature map, which limits depth and, worse, under-samples the borders: a corner pixel participates in exactly one output, a centre pixel in nine.

**Padding** adds a border of zeros around the input.

| Mode | Padding | Output size (stride 1) | Note |
| --- | --- | --- | --- |
| 'valid' | 0 | H - K + 1 | Shrinks; no invented values |
| 'same' | (K-1)/2 | H | Output matches input; needs odd K |
| 'full' | K-1 | H + K - 1 | Grows; rarely used |

For **'same'** padding with a 3x3 kernel, pad by 1; with 5x5, pad by 2; with 7x7, pad by 3. This is why **odd kernel sizes are standard** — an even kernel cannot be centred, so 'same' padding would be asymmetric.

PyTorch accepts `padding='same'` as a string, or an explicit integer. The explicit form is more common in published code:

```python
nn.Conv2d(3, 64, kernel_size=3, padding=1)     # 'same' for 3x3, stride 1
nn.Conv2d(3, 64, kernel_size=5, padding=2)     # 'same' for 5x5
nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)   # ResNet's stem
```

Zero padding does introduce a mild artifact — the network can learn to detect image borders from the zeros — but in practice this is minor and occasionally even useful. `padding_mode='reflect'` or `'replicate'` avoids it when it matters.

### 15.4 Stride

**Stride** is the step size of the slide. Stride 2 moves two pixels at a time, halving the output's spatial dimensions and cutting compute roughly fourfold.

Strided convolution is a learnable alternative to pooling for downsampling, and modern architectures increasingly prefer it: ResNet uses stride-2 convolutions at each stage transition rather than pooling.

### 15.5 The output size formula (memorize this)

```
H_out = floor( (H_in + 2P - K) / S ) + 1
```

where H\_in is input height, P padding, K kernel size, S stride. The same formula applies to width.

**Where it comes from.** After padding, the effective input is `H_in + 2P` wide. The kernel's first position occupies columns 0 to K-1, so its last valid starting position is `H_in + 2P - K`. Starting positions are `0, S, 2S, ...`, so the number of them is `floor((H_in + 2P - K)/S) + 1`, with the +1 counting position 0. Reconstructing it this way is more reliable than memorizing it.

**With dilation** (section 15.9), the kernel's effective size becomes `K_eff = D(K-1) + 1`:

```
H_out = floor( (H_in + 2P - D(K-1) - 1) / S ) + 1
```

**Worked examples — cover the answers and try them:**

| H\_in | K | P | S | H\_out | Computation |
| --- | --- | --- | --- | --- | --- |
| 32 | 3 | 1 | 1 | 32 | (32+2-3)/1 + 1 = 32 |
| 32 | 3 | 0 | 1 | 30 | (32+0-3)/1 + 1 = 30 |
| 32 | 3 | 1 | 2 | 16 | (32+2-3)/2 + 1 = floor(15.5)+1 = 16 |
| 224 | 7 | 3 | 2 | 112 | (224+6-7)/2 + 1 = floor(111.5)+1 = 112 |
| 28 | 5 | 0 | 1 | 24 | (28+0-5)/1 + 1 = 24 |
| 224 | 11 | 2 | 4 | 55 | (224+4-11)/4 + 1 = floor(54.25)+1 = 55 |

The last row is AlexNet's first layer. The fourth is ResNet's stem.

### 15.6 Channels: the part that confuses everyone

Real images have 3 channels. Hidden layers have many more. Get this straight and the rest follows.

**A kernel spans all input channels.** For an input with `C_in` channels, a single 3x3 kernel is not 3x3 — it is `C_in x 3 x 3`. It is applied to all input channels simultaneously and the results are **summed** into one output value.

```
O(i,j) = Σ_{c=0}^{C_in-1} Σ_m Σ_n I(c, i+m, j+n) · K(c, m, n)   + b
```

So one kernel produces **one** output channel, no matter how many input channels there are. The channel dimension is collapsed by summation.

**To get `C_out` output channels, use `C_out` separate kernels.** The full weight tensor of a Conv2d layer has shape:

```
(C_out, C_in, K, K)
```

**Parameter count:**

```
params = C_out x C_in x K x K   +   C_out          (weights + biases)
```

Examples:

| Layer | Weights | Biases | Total |
| --- | --- | --- | --- |
| `Conv2d(3, 64, 3)` | 64·3·3·3 = 1,728 | 64 | 1,792 |
| `Conv2d(64, 128, 3)` | 128·64·3·3 = 73,728 | 128 | 73,856 |
| `Conv2d(256, 512, 3)` | 512·256·3·3 = 1,179,648 | 512 | 1,180,160 |
| `Conv2d(512, 512, 1)` | 512·512·1·1 = 262,144 | 512 | 262,656 |

**Crucially, none of these depend on the image size.** The same layer works on 32x32 and on 4000x3000 inputs with identical parameters. That is parameter sharing paying off.

**The mental model:** each output channel is one **feature map** — the response of one learned pattern detector applied everywhere. 64 output channels means 64 different things this layer is looking for. Early layers find edges and colour blobs; deep layers find object parts.

### 15.7 A full worked example

Input: a batch of 32 RGB images at 32x32. Layer: `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`.

```
Input shape:      (32, 3, 32, 32)      (N, C_in, H, W)
Weight shape:     (64, 3, 3, 3)        (C_out, C_in, K, K)
Bias shape:       (64,)

H_out = (32 + 2·1 - 3)/1 + 1 = 32
Output shape:     (32, 64, 32, 32)

Parameters:       64·3·3·3 + 64 = 1,792
```

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
x = torch.randn(32, 3, 32, 32)
y = conv(x)
print(y.shape)                                      # torch.Size([32, 64, 32, 32])
print(conv.weight.shape, conv.bias.shape)           # [64,3,3,3] [64]
print(sum(p.numel() for p in conv.parameters()))    # 1792
```

### 15.8 Counting FLOPs

Each output value requires `C_in · K · K` multiply-accumulates. There are `C_out · H_out · W_out` output values. So per image:

```
MACs = C_out · H_out · W_out · C_in · K · K
FLOPs ≈ 2 × MACs                    (one multiply + one add each)
```

For the layer above: 64 · 32 · 32 · 3 · 3 · 3 = **1.77 million MACs** per image.

Now a deeper layer, `Conv2d(256, 512, 3, padding=1)` on a 28x28 feature map:

```
512 · 28 · 28 · 256 · 3 · 3 = 924 million MACs
```

**This is the key insight about CNN cost.** The first layer has few parameters and moderate compute. Deep layers have many parameters and, despite smaller spatial dimensions, often more compute, because channel counts grow faster than spatial dimensions shrink. When you optimize a CNN for speed, profile — do not assume the big images cost the most.

Use `torch.utils.flop_counter.FlopCounterMode` or the `thop` package rather than counting by hand in practice, but understand the formula so the numbers mean something.

### 15.9 Dilated (atrous) convolution

Insert gaps of size `D-1` between kernel elements. A 3x3 kernel with dilation 2 covers a 5x5 region while using only 9 weights.

```
D=1 (normal)      D=2 (dilated)
 X X X            X . X . X
 X X X            . . . . .
 X X X            X . X . X
                  . . . . .
                  X . X . X
```

This expands the receptive field **exponentially with depth** without adding parameters or reducing resolution. It is standard in semantic segmentation (DeepLab) and in audio models (WaveNet), where you need large context but must keep full spatial or temporal resolution.

```python
nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)   # padding=D keeps size
```

### 15.10 1x1 convolutions: more useful than they look

A 1x1 kernel looks pointless — it sees one pixel. It is not, and understanding why is a standard interview question.

A 1x1 convolution has no spatial extent but **still spans all input channels**. So at each spatial position independently, it computes a learned linear combination of the `C_in` channel values, producing `C_out` new values. It is a fully connected layer applied across channels, shared across space.

Three uses:

**1. Channel dimensionality reduction.** This is its main job. Compare:

```
Direct:      Conv2d(256, 256, 3)                     = 590K parameters
Bottleneck:  Conv2d(256,  64, 1)  →  16K
             Conv2d( 64,  64, 3)  →  37K
             Conv2d( 64, 256, 1)  →  16K
                                      Total = 69K parameters  (8.5x fewer)
```

This is exactly the **bottleneck block** used in ResNet-50 and above, and the reason ResNet-50 has fewer parameters than VGG-16 despite being three times deeper.

**2. Adding nonlinearity cheaply.** Conv 1x1 followed by ReLU increases the network's expressiveness at low cost. This was the "Network in Network" idea (Lin et al., 2013) that GoogLeNet built on.

**3. Matching channel counts** so a residual connection can add tensors of different depths (section 18.5).

### 15.11 Depthwise separable convolution

The idea behind MobileNet and the efficient half of modern vision. Factor a standard convolution into two cheaper steps:

1. **Depthwise**: one `K x K` kernel per input channel, applied independently — no mixing across channels. `groups=C_in` in PyTorch.
2. **Pointwise**: a 1x1 convolution that mixes channels.

```
Standard:   C_in · C_out · K · K
Separable:  C_in · K · K   +   C_in · C_out
Ratio:      1/C_out + 1/K²
```

For `C_in=C_out=256`, K=3: standard is 589,824 weights; separable is 2,304 + 65,536 = 67,840. About **8.7x fewer**, with a modest accuracy cost. This is how models run on phones.

```python
depthwise = nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False)
pointwise = nn.Conv2d(256, 256, 1, bias=False)
```

### Check yourself

1. Input (16, 3, 224, 224) through `Conv2d(3, 96, kernel_size=11, stride=4, padding=2)`. Output shape and parameter count?
2. You need the output to be exactly half the input size with a 5x5 kernel. What stride and padding?
3. Why are 3x3 kernels near-universal in modern CNNs rather than 7x7?
4. A 1x1 conv on a (N, 512, 14, 14) input producing 128 channels: output shape, parameters, and MACs?

*(Answers: 1 — H\_out = (224 + 4 - 11)/4 + 1 = floor(54.25) + 1 = 55, so output is (16, 96, 55, 55). Parameters: 96·3·11·11 + 96 = 34,848 + 96 = 34,944. This is AlexNet's first layer. 2 — stride 2 with padding 2: H\_out = (H + 4 - 5)/2 + 1 = floor((H-1)/2) + 1, which equals H/2 for even H. 3 — two stacked 3x3 layers have the same 5x5 receptive field as one 5x5 layer but use 2·(9C²) = 18C² parameters instead of 25C², and three stacked 3x3s match a 7x7 with 27C² instead of 49C². Stacking also inserts more nonlinearities, which increases expressiveness. VGG established this and it has held. 4 — output (N, 128, 14, 14); parameters 128·512·1·1 + 128 = 65,664; MACs 128·14·14·512·1·1 = 12.8 million per image.)*

## 16. Pooling, CNN Anatomy, and Receptive Fields

### 16.1 Pooling

Pooling downsamples a feature map by summarizing each local region with a single value.

**Max pooling** takes the maximum in each window. With a 2x2 window and stride 2:

```
Input 4x4              Output 2x2
1  3 | 2  4            
5  2 | 6  1     →      5  6
-----+-----            9  8
7  9 | 3  8
4  1 | 8  2            
```

Each 2x2 block contributes its maximum: max(1,3,5,2)=5, max(2,4,6,1)=6, max(7,9,4,1)=9, max(3,8,8,2)=8.

**Average pooling** takes the mean instead.

**What pooling gives you:**

- **Local translation invariance.** If the input shifts by one pixel, the max over a 2x2 window often does not change. Small position changes stop mattering, which is exactly what you want for object recognition.
- **Downsampling.** 2x2 stride-2 pooling quarters the number of activations, cutting compute and memory for all subsequent layers.
- **Larger receptive fields.** After pooling, each unit in the next layer effectively sees twice as much of the original image.
- **No parameters.** Pooling has nothing to learn.

**Max vs average.** Max asks "is this feature present anywhere in this region?", which suits feature detection: the strongest evidence wins and weak background responses are discarded. Average keeps more information but dilutes strong signals. Max pooling dominated classical CNNs; **global average pooling** at the end of the network is now near-universal (see 16.4).

**Is pooling obsolete?** Partly. Many modern architectures replace it with stride-2 convolutions, which downsample *and* learn how to do it. ResNet uses one max-pool near the input and strided convolutions thereafter. Springenberg et al. (2015) showed an all-convolutional network matches pooling-based ones. Pooling remains common and is not wrong; it is simply one of two reasonable choices.

A real cost of pooling: it discards precise location. That is fine for classification ("is there a cat?") and harmful for segmentation or detection ("exactly which pixels are cat?"), which is why those architectures use dilated convolutions or skip connections to recover spatial detail.

```python
nn.MaxPool2d(kernel_size=2, stride=2)        # halves H and W
nn.AvgPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))                 # -> (N, C, 1, 1), any input size
nn.AdaptiveAvgPool2d((7, 7))                 # -> (N, C, 7, 7), any input size
```

The output-size formula from 15.5 applies to pooling too, with P = 0 usually.

### 16.2 The anatomy of a CNN

A classification CNN has three parts, and naming them makes architectures much easier to read.

```
INPUT (N, 3, 224, 224)
   │
   ├─ BACKBONE / FEATURE EXTRACTOR ────────────────────
   │   [Conv → BN → ReLU] x k  →  downsample           │
   │   [Conv → BN → ReLU] x k  →  downsample           │  channels grow,
   │   [Conv → BN → ReLU] x k  →  downsample           │  space shrinks
   │   ...                                             │
   │  output: (N, 512, 7, 7)                           │
   ├────────────────────────────────────────────────────
   │
   ├─ NECK: Global Average Pool → (N, 512)
   │
   └─ HEAD: Linear(512, num_classes) → (N, num_classes) logits
```

**The governing pattern: as spatial dimensions shrink, channel count grows.** 224x224x3 → 112x112x64 → 56x56x128 → 28x28x256 → 14x14x512 → 7x7x512. Each downsampling step roughly halves H and W (dividing activations by 4) while doubling channels (multiplying by 2), so the total activation volume falls by about half per stage, and the representation trades spatial precision for semantic richness.

**Why this makes sense.** Early on you need fine spatial resolution but there are only a few distinct things to detect (edges at various orientations). Later you need many distinct concepts (thousands of object parts) but their exact position matters less.

### 16.3 Receptive field: how much of the image each unit sees

A unit's **receptive field** is the region of the *input image* that can influence its value. This determines what the unit can possibly detect: a unit with a 15x15 receptive field cannot recognize an object spanning 50 pixels, no matter how it is trained.

**The recurrence.** Let `r_l` be the receptive field size at layer l and `j_l` the jump (the distance in input pixels between adjacent units at layer l).

```
j_0 = 1,   r_0 = 1
j_l = j_{l-1} · S_l
r_l = r_{l-1} + (K_l - 1) · j_{l-1}
```

Read it as: each new layer adds `(K-1)` steps of the current jump to the receptive field, and multiplies the jump by its stride.

**Worked example** — a typical small stack:

| Layer | K | S | j | r |
| --- | --- | --- | --- | --- |
| input | — | — | 1 | 1 |
| conv3 | 3 | 1 | 1 | 1 + 2·1 = 3 |
| conv3 | 3 | 1 | 1 | 3 + 2·1 = 5 |
| maxpool2 | 2 | 2 | 2 | 5 + 1·1 = 6 |
| conv3 | 3 | 1 | 2 | 6 + 2·2 = 10 |
| conv3 | 3 | 1 | 2 | 10 + 2·2 = 14 |
| maxpool2 | 2 | 2 | 4 | 14 + 1·2 = 16 |
| conv3 | 3 | 1 | 4 | 16 + 2·4 = 24 |

Notice how strides accelerate growth. The first two conv layers add 2 pixels each; after two pooling layers, each conv adds 8. **Downsampling is the main driver of receptive field growth**, which is why you cannot simply remove all pooling from a deep network and expect it to work the same.

**Three practical consequences:**

1. **Check that your receptive field covers your objects.** If you classify 224x224 images of whole objects, the final layer's receptive field should be at least 224. ResNet-50's is well over 400 — comfortably more than the image, which is the usual design.
2. **Stacking small kernels beats one large kernel.** Two 3x3 layers give a 5x5 receptive field with 18C² parameters instead of 25C², plus an extra nonlinearity. Three 3x3s match 7x7 with 27C² instead of 49C². This is VGG's central finding and it is why 3x3 is everywhere.
3. **The effective receptive field is smaller than the theoretical one.** Luo et al. (2016) showed that influence decays roughly like a Gaussian from the centre, so the region that *meaningfully* affects a unit is considerably smaller than the formula suggests. Build in margin.

### 16.4 Global average pooling and why flattening fell out of favour

Classic CNNs (LeNet, AlexNet, VGG) ended with `Flatten → Linear(25088, 4096) → Linear(4096, 4096) → Linear(4096, 1000)`. In VGG-16, those fully connected layers hold **123 million of the model's 138 million parameters** — 89% of the model in the last three layers.

**Global average pooling** replaces them. Average each channel's entire feature map down to one number:

```
(N, 512, 7, 7)  →  GAP  →  (N, 512)  →  Linear(512, 1000)
```

Parameters in the head drop from 123 million to 513,000. Three further benefits:

- **Any input size works.** GAP averages whatever spatial size it receives, so the network accepts 224x224 or 448x448 without modification. A flattening head is locked to one input size.
- **It regularizes.** Far fewer parameters means far less overfitting; VGG needed heavy dropout in its head, and GAP-based networks generally do not.
- **It is interpretable.** Each of the 512 values is "how strongly is feature k present, on average, in this image," which supports class activation mapping (CAM) for visualizing what the model looked at.

Every architecture since ResNet uses GAP. In PyTorch it is `nn.AdaptiveAvgPool2d((1,1))` followed by `nn.Flatten()`.

### 16.5 A complete small CNN, annotated with shapes

```python
import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """For 32x32x3 inputs (CIFAR-10). Shapes are annotated at every step."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1:  (N,3,32,32) -> (N,64,16,16)
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 32 -> 16

            # Block 2:  (N,64,16,16) -> (N,128,8,8)
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 16 -> 8

            # Block 3:  (N,128,8,8) -> (N,256,4,4)
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 8 -> 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # (N,256,4,4) -> (N,256,1,1)
            nn.Flatten(),                          # -> (N,256)
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),           # -> (N,10) logits
        )

    def forward(self, x):
        return self.head(self.features(x))

model = SmallCNN()
x = torch.randn(4, 3, 32, 32)
print(model(x).shape)                                   # torch.Size([4, 10])
print(f"{sum(p.numel() for p in model.parameters()):,}")   # 1,148,874

# Print the shape after every layer -- do this whenever you build a new model
h = x
for layer in model.features:
    h = layer(h)
    print(f"{layer.__class__.__name__:15s} {tuple(h.shape)}")
```

That last loop is the single most useful debugging habit in CNN work. Run it on every new architecture before you train.

### Check yourself

1. A (N, 128, 56, 56) tensor goes through `MaxPool2d(3, stride=2)`. Output shape?
2. Compute the receptive field after: conv5(s1) → conv3(s1) → maxpool2(s2) → conv3(s1).
3. Why does replacing VGG's fully connected head with GAP reduce overfitting?
4. You want a network for 512x512 medical scans where lesions span about 40 pixels. What minimum receptive field do you target, and why might you want more?

*(Answers: 1 — (56 - 3)/2 + 1 = floor(26.5) + 1 = 27, so (N, 128, 27, 27). 2 — start j=1, r=1. conv5: r = 1 + 4·1 = 5, j = 1. conv3: r = 5 + 2·1 = 7, j = 1. maxpool2 s2: r = 7 + 1·1 = 8, j = 2. conv3: r = 8 + 2·2 = 12, j = 2. Final receptive field 12x12. 3 — it removes \~123 million parameters, and overfitting capacity scales with parameter count; GAP also imposes a strong structural constraint that each class score depends only on average channel activations, which is a far less flexible function than an arbitrary linear map over all 25,088 flattened values. 4 — at least 40 to see a whole lesion, but target 80-120 or more: the effective receptive field is substantially smaller than the theoretical one, and context around a lesion (surrounding tissue) is usually diagnostic, so the unit needs to see more than the lesion alone.)*

## 17. Backpropagation Through Convolution and Pooling

Autograd handles this for you. Derive it once anyway: it is a common interview question, it explains why convolution is efficient in both directions, and it makes the transposed convolution used in segmentation and generative models obvious rather than mysterious.

### 17.1 Setup

Single input channel, single output channel, stride 1, no padding. Let `X` be the input, `K` the kernel, `O` the output:

```
O(i,j) = Σ_m Σ_n X(i+m, j+n) · K(m,n)
```

Suppose the backward pass has delivered `dO(i,j) = ∂L/∂O(i,j)` for every output position. We need `∂L/∂K` and `∂L/∂X`.

The guiding principle is the multivariable chain rule from section 6.2: **a variable's gradient is the sum of contributions over every path it influences.**

### 17.2 Gradient with respect to the kernel

The weight `K(m,n)` is used at **every** output position — that is what parameter sharing means. So its gradient sums over all of them:

```
∂L/∂K(m,n) = Σ_i Σ_j (∂L/∂O(i,j)) · (∂O(i,j)/∂K(m,n))
```

From the forward equation, `∂O(i,j)/∂K(m,n) = X(i+m, j+n)`. Therefore:

```
∂L/∂K(m,n) = Σ_i Σ_j dO(i,j) · X(i+m, j+n)
```

Look at the structure: this is **a convolution (cross-correlation) of the input X with the output gradient dO as the kernel.** The gradient of a convolution's weights is another convolution.

This mirrors BP4 exactly. In a fully connected layer, `∂L/∂W = δ · a_inᵀ` — the error signal times the input activation. Here it is the same thing, summed over all spatial positions where the weight was applied. Parameter sharing in the forward pass becomes gradient summation in the backward pass. That correspondence is worth stating out loud in an interview.

**With channels**, for kernel element `K(c_out, c_in, m, n)`:

```
∂L/∂K(c_out, c_in, m, n) = Σ_i Σ_j dO(c_out, i, j) · X(c_in, i+m, j+n)
```

and with a batch, sum over examples as well.

**The bias:** `∂L/∂b(c_out) = Σ_i Σ_j dO(c_out, i, j)` — one bias per output channel, used at every spatial position, so sum over all of them.

### 17.3 Gradient with respect to the input

This one is subtler. The input value `X(p,q)` participates in every output position whose window covers it. Setting `i+m = p` and `j+n = q`, i.e. `i = p-m`, `j = q-n`:

```
∂L/∂X(p,q) = Σ_m Σ_n dO(p-m, q-n) · K(m,n)
```

(treating out-of-range indices as zero).

**That is a true convolution** — note the minus signs — which means it is a cross-correlation with the kernel **flipped 180 degrees** in both spatial dimensions.

So the backward pass through a convolution is:

```
dX = full_convolution( dO, flip180(K) )
```

where "full" means padded by `K-1` on each side, so the output has the input's size. This is the moment where the convolution/cross-correlation distinction from section 15.2 finally matters.

**An intuition for the flip.** In the forward pass, information flows from input positions out to a neighbourhood of output positions. In the backward pass it flows from output positions back to a neighbourhood of input positions — the same neighbourhood relationship traversed in reverse, which is exactly what a 180-degree flip encodes.

**Why this is beautiful and practical:** both the forward pass and both backward passes are convolutions. A framework needs one highly optimized convolution primitive (cuDNN's) and gets training for free. This is a large part of why CNNs are so fast on GPUs.

### 17.4 Transposed convolution

The operation `dX = full_conv(dO, flip180(K))` maps a small tensor to a larger one. Used as a *forward* operation in its own right, it **upsamples**, and it is then called a **transposed convolution** (`nn.ConvTranspose2d`), sometimes "deconvolution" — a poor name, since it does not invert convolution in any mathematical sense.

It is the backward pass of convolution used forwards. That is the entire idea. It appears in:

- **Semantic segmentation** (U-Net's decoder), to return from a small feature map to per-pixel predictions
- **Generative models** (DCGAN's generator), to go from a latent vector to an image
- **Super-resolution**

A known artifact: transposed convolution produces **checkerboard patterns** when the kernel size is not divisible by the stride, because output positions receive uneven numbers of contributions. The common fix is `nn.Upsample(scale_factor=2, mode='bilinear')` followed by a normal 3x3 convolution, which most modern architectures prefer.

### 17.5 Backprop through pooling

Pooling has no parameters, so there is nothing to learn — but gradients must still reach the layers below.

**Max pooling: route the gradient to the winner.** The output equals the maximum input in the window, so the derivative with respect to that input is 1 and with respect to all others is 0.

```
Forward:                     Backward (dO = [[2, 3], [1, 4]]):
1  3 | 2  4                   0  0 | 0  0
5  2 | 6  1    →  5  6        2  0 | 3  0
-----+-----       9  8       -----+-----
7  9 | 3  8                   0  1 | 0  0
4  1 | 8  2                   0  0 | 4  0
```

Each incoming gradient goes entirely to the position that produced the maximum; every other position gets zero. Implementations cache the argmax indices during the forward pass — this is exactly what `nn.MaxPool2d(return_indices=True)` exposes, and what `nn.MaxUnpool2d` consumes.

**A consequence worth knowing:** with 2x2 pooling, 75% of positions receive zero gradient at that layer. Max pooling is a sparse gradient router. This is one more reason modern architectures often prefer strided convolutions, which pass gradients to every position.

**Average pooling: distribute evenly.** The output is `(1/k²)Σx`, so each input's derivative is `1/k²`:

```
dX(p,q) = dO(i,j) / k²     for every (p,q) in the window
```

Smooth, dense gradients, but no signal about which input mattered.

### 17.6 Verifying it yourself

You do not have to take any of this on faith:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(1, 1, 5, 5, requires_grad=True, dtype=torch.float64)
K = torch.randn(1, 1, 3, 3, requires_grad=True, dtype=torch.float64)

O = F.conv2d(X, K)                 # (1,1,3,3)
dO = torch.randn_like(O)
O.backward(dO)

# --- check the kernel gradient: cross-correlate X with dO ---
manual_dK = F.conv2d(X.detach(), dO)          # treat dO as the kernel
print("dK match:", torch.allclose(K.grad, manual_dK))

# --- check the input gradient: full conv of dO with flipped K ---
K_flip = torch.flip(K.detach(), dims=[2, 3])  # 180-degree flip
manual_dX = F.conv2d(dO, K_flip, padding=2)   # padding = K-1 = 2 -> 'full'
print("dX match:", torch.allclose(X.grad, manual_dX))
```

Both print `True`. Run it; deriving something and then watching the framework agree with you is the fastest way to build confidence that you actually understand the operation.

### Check yourself

1. Why does the kernel gradient sum over all spatial positions while a fully connected weight gradient does not?
2. A 2x2 max-pool receives gradient 6.0 at one output position. The four inputs were \[3, 7, 2, 5\]. What gradient does each input receive? What if it were average pooling?
3. What causes checkerboard artifacts in transposed convolution, and what is the standard fix?
4. A conv layer has 64 input and 128 output channels with 3x3 kernels. How many numbers must the backward pass produce for the weight gradient?

*(Answers: 1 — because of parameter sharing: the same weight is applied at every position, so by the multivariable chain rule its gradient sums over every path it influenced. A fully connected weight connects exactly one input unit to one output unit, so there is one path per example and the sum is only over the batch. 2 — max pool: the input with value 7 receives 6.0, the other three receive 0. Average pool: each receives 6.0/4 = 1.5. 3 — when the kernel size is not divisible by the stride, different output positions receive different numbers of overlapping contributions, producing a periodic intensity pattern; the standard fix is bilinear upsampling followed by a regular convolution, or choosing kernel size divisible by stride (e.g. 4x4 kernel with stride 2). 4 — the weight gradient has the same shape as the weights: 128·64·3·3 = 73,728 values, plus 128 bias gradients.)*

## 18. Classic Architectures: LeNet to ResNet

Read these as a sequence of problems and solutions rather than a list of models. Each architecture exists because the previous one hit a wall.

### 18.1 LeNet-5 (LeCun et al., 1998)

The original working CNN, built to read handwritten digits on bank cheques — and actually deployed, reading a significant fraction of US cheques in the late 1990s.

```
Input 32x32x1
 → Conv 5x5, 6 filters      → 28x28x6
 → AvgPool 2x2             → 14x14x6
 → Conv 5x5, 16 filters    → 10x10x16
 → AvgPool 2x2             → 5x5x16
 → Flatten                 → 400
 → FC 120 → FC 84 → FC 10
```

\~60,000 parameters. Used tanh activations and average pooling, both of which would be replaced later.

**What it established, and what has survived unchanged for 28 years:** alternating convolution and subsampling, growing channel counts as spatial size shrinks, and a fully connected classifier at the end. Every CNN since follows this template.

### 18.2 AlexNet (Krizhevsky, Sutskever & Hinton, 2012)

The paper that started the deep learning era. It won ImageNet 2012 with a top-5 error of about 15.3% against roughly 26% for the runner-up — a margin so large it ended the debate about whether deep learning worked.

```
Input 227x227x3   (the paper says 224, but 224 with pad 2 and stride 4 does not give 55; 227 with pad 0 does. This discrepancy is well known and 227 is what implementations use.)
 → Conv 11x11, 96, stride 4   → 55x55x96    → MaxPool 3x3/2 → 27x27x96
 → Conv 5x5, 256, pad 2       → 27x27x256   → MaxPool 3x3/2 → 13x13x256
 → Conv 3x3, 384, pad 1       → 13x13x384
 → Conv 3x3, 384, pad 1       → 13x13x384
 → Conv 3x3, 256, pad 1       → 13x13x256   → MaxPool 3x3/2 → 6x6x256
 → FC 4096 → FC 4096 → FC 1000
```

60 million parameters. Architecturally it is LeNet, scaled up. The innovations were elsewhere, and that is the lesson:

1. **ReLU instead of tanh** — reportedly about 6x faster to reach the same training error (section 3.2).
2. **Two GPUs** — the model was split across two GTX 580s because 3 GB was not enough for one.
3. **Dropout 0.5** in the fully connected layers — essential given 60M parameters.
4. **Heavy data augmentation** — random crops, horizontal flips, PCA-based colour jitter.
5. **Overlapping max pooling** (3x3 window, stride 2).

Note the 11x11 stride-4 first layer, chosen to cut the spatial size aggressively. Later architectures abandoned large kernels entirely.

### 18.3 VGG (Simonyan & Zisserman, 2014)

VGG asked one question: what if we use only 3x3 convolutions and just go deeper?

```
VGG-16:
 [Conv3-64,  Conv3-64]                  → MaxPool  → 112x112
 [Conv3-128, Conv3-128]                 → MaxPool  → 56x56
 [Conv3-256, Conv3-256, Conv3-256]      → MaxPool  → 28x28
 [Conv3-512, Conv3-512, Conv3-512]      → MaxPool  → 14x14
 [Conv3-512, Conv3-512, Conv3-512]      → MaxPool  → 7x7
 FC-4096 → FC-4096 → FC-1000
```

138 million parameters, of which 123 million are in the three fully connected layers (section 16.4).

**The 3x3 argument, which is VGG's lasting contribution.** Two stacked 3x3 layers have the same 5x5 receptive field as one 5x5 layer, but:

```
Two 3x3 layers:  2 · (3·3·C·C) = 18C² parameters, 2 nonlinearities
One 5x5 layer:       (5·5·C·C) = 25C² parameters, 1 nonlinearity

Three 3x3 layers: 27C²  vs  one 7x7 layer: 49C²
```

Fewer parameters, more nonlinearity, better accuracy. This settled kernel-size design for a decade.

VGG's weaknesses: enormous and slow, and hard to train from scratch (the original authors had to train a shallower version first and use it to initialize the deeper one — a clear symptom of the problem ResNet would solve). VGG remains popular as a perceptual-loss feature extractor in style transfer and image generation.

### 18.4 GoogLeNet / Inception (Szegedy et al., 2014)

Inception asked a different question: instead of choosing a kernel size, why not use several and let the network decide?

An Inception module runs 1x1, 3x3, and 5x5 convolutions plus a 3x3 max-pool **in parallel** on the same input, then concatenates the results along the channel dimension.

The naive version is ruinously expensive, so the key trick is **1x1 convolutions as bottlenecks before the expensive branches** (section 15.10). The result: 22 layers and 5 million parameters — **12x fewer than AlexNet** — with better accuracy.

It also used **auxiliary classifiers**: extra loss heads attached to intermediate layers, to push gradient into the early layers. This was a pre-ResNet workaround for vanishing gradients, and it was abandoned once residual connections solved the problem properly.

### 18.5 ResNet (He et al., 2015) — the most important architecture here

**The problem it identified.** The authors observed something that should be impossible: a 56-layer plain network had **higher training error** than a 20-layer one. Not a test error gap — a *training* error gap. That rules out overfitting.

It also should not happen in principle. A 56-layer network can represent everything a 20-layer network can: make the extra 36 layers identity mappings. So the deeper network's solution space strictly contains the shallower one's. The problem is that **optimization cannot find that solution**. They called this the **degradation problem**.

**The solution.** If the difficulty is learning an identity mapping, build it in:

```
y = F(x, W) + x
```

The block learns `F`, the **residual** — the difference from the identity. If the identity is optimal, the network only needs to push `F` toward zero, which is far easier than making a stack of convolutions compute the identity exactly.

**Why this fixes gradient flow — the derivation.** For `y = F(x) + x`:

```
∂L/∂x = ∂L/∂y · ∂y/∂x = ∂L/∂y · ( ∂F/∂x + 1 ) = ∂L/∂y · ∂F/∂x + ∂L/∂y
```

That trailing `∂L/∂y` term is the whole point. **The gradient reaches x undiminished, regardless of what happens inside F.** Even if the residual branch's gradient is nearly zero, the identity path carries the full signal.

Unroll this across L blocks and the gradient at block 1 contains an unattenuated term from the output. The product-of-many-factors problem of section 9.1 becomes a *sum* of paths, at least one of which has gain exactly 1. Vanishing gradients stop being a depth barrier.

**The two block types:**

```
Basic block (ResNet-18/34):        Bottleneck block (ResNet-50/101/152):
  x ──────────────┐                  x ─────────────────────┐
  Conv3x3, C      │                  Conv1x1, C/4           │   reduce
  BN, ReLU        │                  BN, ReLU               │
  Conv3x3, C      │                  Conv3x3, C/4           │   process cheaply
  BN              │                  BN, ReLU               │
  (+) ◄───────────┘                  Conv1x1, C             │   restore
  ReLU                               BN                     │
                                     (+) ◄──────────────────┘
                                     ReLU
```

The bottleneck uses 1x1 convolutions to cut channels fourfold before the expensive 3x3, then restore them. This is why ResNet-50 (25M parameters) is smaller than VGG-16 (138M) while being three times deeper and considerably more accurate.

**When shapes do not match** — at a stage transition where channels double and stride is 2 — the shortcut needs a projection: a 1x1 convolution with stride 2, sometimes called the "option B" or downsample shortcut.

```python
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Projection shortcut only when the shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # the residual connection
        return self.relu(out)         # ReLU AFTER the addition
```

Note that the second BN has no ReLU after it inside the branch; the ReLU comes after the addition. Putting it inside would mean the branch could only ever add non-negative values, which needlessly restricts what the residual can express.

ResNet-152 reached 3.57% top-5 error on ImageNet, below the commonly cited \~5% human benchmark on that task, and ResNet variants swept every track of ILSVRC 2015.

### 18.6 Comparison

| Model | Year | Depth | Params | ImageNet top-5 err | Key idea |
| --- | --- | --- | --- | --- | --- |
| LeNet-5 | 1998 | 7 | 60K | — (MNIST) | The CNN template |
| AlexNet | 2012 | 8 | 60M | 15.3% | ReLU, dropout, GPUs, augmentation |
| VGG-16 | 2014 | 16 | 138M | 7.3% | Stacked 3x3; depth via uniformity |
| GoogLeNet | 2014 | 22 | 5M | 6.7% | Inception modules, 1x1 bottlenecks |
| ResNet-50 | 2015 | 50 | 25M | \~5.3% | Residual connections |
| ResNet-152 | 2015 | 152 | 60M | 3.57% | Residual connections, more depth |

**Read the error column with care.** These are the headline figures from each paper's ILSVRC entry, and they are *not* all comparable: VGG's 7.3%, GoogLeNet's 6.67% and ResNet's 3.57% are ensemble and/or multi-crop results. Single-model, single-crop numbers are meaningfully worse — ResNet-152 is around 4.5%, VGG-16 around 8.8%. Competition tables mix protocols, which is exactly the kind of thing to check before believing any architecture comparison (see section 21.3).

Read the parameter column against the error column. VGG-16 has **5.5x** the parameters of ResNet-50 and worse accuracy. Architecture matters more than size.

### 18.7 Using them in PyTorch

```python
import torchvision.models as models

# Modern API: the `weights` argument (the old pretrained=True is deprecated)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
vgg16    = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# Each weights enum carries its own required preprocessing -- use it,
# do not guess the normalization constants
weights   = models.ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
print(preprocess)

print(resnet50)              # prints the full architecture
print(sum(p.numel() for p in resnet50.parameters()))    # 25,557,032
```

`IMAGENET1K_V2` weights come from an improved modern training recipe and are meaningfully more accurate than the original `V1` weights for the same architecture — another reminder that training procedure matters as much as architecture.

### Check yourself

1. Why did a 56-layer plain network have higher *training* error than a 20-layer one, and why does that rule out overfitting?
2. Write the gradient of `y = F(x) + x` with respect to x and explain why the `+1` matters.
3. VGG-16 has 138M parameters and ResNet-50 has 25M, yet ResNet is deeper and better. Where did VGG's parameters go?
4. In a bottleneck block for 256 channels, count the parameters and compare with two 3x3 convolutions at full width.

*(Answers: 1 — the degradation problem: optimization difficulty, not capacity. Overfitting means low training error and high test error; here training error itself was higher, so the deeper model was failing to fit data it provably had the capacity to fit. 2 — `∂L/∂x = ∂L/∂y·(∂F/∂x + 1)`. The `+1` creates a path along which the gradient passes unchanged, so no matter how small `∂F/∂x` becomes, `∂L/∂x` is at least `∂L/∂y`. Gradients cannot vanish through the skip path. 3 — 123M of VGG's 138M sit in its three fully connected layers, chiefly the 25088 x 4096 first one; ResNet replaces that head with global average pooling plus a single 2048 x 1000 layer. 4 — bottleneck: 1x1 (256→64) = 16,384; 3x3 (64→64) = 36,864; 1x1 (64→256) = 16,384; total 69,632. Two full-width 3x3 layers: 2 x (256·256·9) = 1,179,648. The bottleneck uses about 17x fewer parameters.)*

## 19. Full CNN Project: CIFAR-10 from Scratch

CIFAR-10 is 60,000 32x32 colour images in 10 classes — 50,000 train, 10,000 test. It is the standard proving ground for CNN work: small enough to iterate on a laptop or free Colab GPU, hard enough that every technique in this tutorial visibly matters.

**Targets to calibrate against:** a plain MLP reaches roughly 50-55%. A basic CNN without augmentation, 75-80%. The ResNet below with good augmentation and schedule, **93-95%**. State of the art is around 99%, with pretraining on far larger datasets.

### 19.1 The complete script

```python
"""
CIFAR-10 with a ResNet-style CNN built from scratch.
~20 min on a free Colab T4 GPU for 60 epochs. Expect 93-95% test accuracy.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============ 0. Setup ====================================================
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True      # autotune conv algorithms; fixed shapes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

EPOCHS      = 60
BATCH       = 128
BASE_LR     = 0.1                          # SGD+momentum; see section 10.7
WEIGHT_DECAY = 5e-4

# ============ 1. Data =====================================================
# CIFAR-10 channel statistics, computed over the training set.
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # translation
    transforms.RandomHorizontalFlip(),                             # mirror
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),           # occlusion
])

eval_tf = transforms.Compose([          # no augmentation at eval time
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full_train = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
test_set   = datasets.CIFAR10("./data", train=False, download=True, transform=eval_tf)

# Validation split. The val subset inherits train_tf, which adds noise to the
# metric; for a cleaner signal, build a second dataset object with eval_tf.
train_set, val_set = random_split(
    full_train, [45_000, 5_000],
    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False,
                          num_workers=4, pin_memory=True)

CLASSES = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# ============ 2. Model ====================================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    """ResNet-18 adapted for 32x32: 3x3 stem, no initial max-pool."""
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10):
        super().__init__()
        self.in_ch = 64
        # A 7x7/2 stem would destroy a 32x32 image immediately.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  num_blocks[0], stride=1)   # 32x32
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)   # 16x16
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)   # 8x8
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)   # 4x4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layer(self, out_ch, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)   # only the first downsamples
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_ch, out_ch, s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')       # section 9.3
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-init the last BN in each block: every block starts as identity
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)                 # section 9.6

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(x)


model = ResNetCIFAR().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")   # ~11.2M

# Shape sanity check -- always do this before training
with torch.no_grad():
    dummy = torch.randn(2, 3, 32, 32, device=device)
    print("Output shape:", model(dummy).shape)                        # [2, 10]

# ============ 3. Loss / optimizer / schedule ==============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# No weight decay on BN parameters or biases (section 10.6)
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)

optimizer = torch.optim.SGD(
    [{"params": decay,    "weight_decay": WEIGHT_DECAY},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=BASE_LR, momentum=0.9, nesterov=True)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=BASE_LR,
    steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.15)

# Mixed precision: ~2x faster on modern GPUs, with no accuracy cost
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ============ 4. Train / eval =============================================
def train_one_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()      # scale up to avoid fp16 underflow
        scaler.step(optimizer)             # unscales, then steps
        scaler.update()
        scheduler.step()                   # OneCycleLR steps per BATCH

        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total


# ============ 5. Training loop ============================================
best_val, history = 0.0, []
start = time.time()

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch()
    va_loss, va_acc = evaluate(val_loader)
    history.append((tr_loss, tr_acc, va_loss, va_acc))

    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), "best_cifar.pt")

    if epoch % 5 == 0 or epoch == 1:
        lr = optimizer.param_groups[0]["lr"]
        print(f"ep {epoch:3d} | lr {lr:.4f} | "
              f"train {tr_loss:.3f}/{tr_acc:.4f} | "
              f"val {va_loss:.3f}/{va_acc:.4f} | "
              f"{time.time()-start:.0f}s")

# ============ 6. Final test ===============================================
model.load_state_dict(torch.load("best_cifar.pt"))
te_loss, te_acc = evaluate(test_loader)
print(f"\nBest val {best_val:.4f} | TEST accuracy {te_acc:.4f}")

# ============ 7. Confusion matrix =========================================
model.eval()
conf = torch.zeros(10, 10, dtype=torch.long)
with torch.no_grad():
    for x, y in test_loader:
        p = model(x.to(device)).argmax(1).cpu()
        for t, pr in zip(y, p):
            conf[t, pr] += 1

print("\n        " + " ".join(f"{c:>5s}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    row = " ".join(f"{v:5d}" for v in conf[i].tolist())
    print(f"{c:>7s} {row}   acc={conf[i,i].item()/conf[i].sum().item():.3f}")
```

### 19.2 Reading the results

**The confusion matrix is where the real information is.** A typical run shows:

- **cat ↔ dog** confusion is by far the largest off-diagonal block. Both are furry quadrupeds, and at 32x32 the distinguishing features (face shape, ear geometry) are only a handful of pixels.
- **plane ↔ ship ↔ bird** get confused, because all three commonly appear against large uniform blue backgrounds.
- **car ↔ truck** confusion is moderate; they share shape but differ in proportions.
- **frog, horse, truck** are usually the easiest.

This pattern is informative. The errors are *semantically sensible* — the model is confusing things that genuinely look alike at this resolution, not making random mistakes. A model whose errors look random has a bug; a model whose errors look like a hurried human's is working correctly and is limited by data or resolution.

**Ablations worth running yourself.** Each takes a few minutes to configure and teaches more than reading about it:

| Change | Expected test accuracy | What it shows |
| --- | --- | --- |
| Full script | 93-95% | Baseline |
| No augmentation | 85-88% | Augmentation is worth \~7 points here |
| No BatchNorm | 88-90%, needs lr ≈ 0.01 | BN's effect on trainable learning rate |
| No residual connections | 89-91% | Degradation at 18 layers is real but mild |
| Constant lr instead of OneCycle | 90-92% | Scheduling is worth 2-3 points |
| Adam(3e-4) instead of SGD | 91-93% | The SGD-vs-Adam gap on vision, in practice |
| Plain MLP of similar size | 50-55% | Why section 14 exists |

(Numbers are typical, not guarantees; run-to-run variance on CIFAR-10 is roughly ±0.3%.)

### 19.3 Techniques used, and where they came from

| Technique | Section | Why it is here |
| --- | --- | --- |
| He initialization, fan\_out | 9.3, 9.6 | Deep ReLU stack needs correct variance |
| Zero-init final BN in blocks | 9.6 | Blocks start as identity |
| BatchNorm | 11 | Enables lr=0.1 and fast convergence |
| Residual connections | 18.5 | Gradient flow at 18 layers |
| Global average pooling | 16.4 | Removes a huge FC head |
| SGD + Nesterov momentum | 10.2 | Best final accuracy on vision |
| OneCycle schedule | 10.8 | Fast convergence, no milestone tuning |
| Weight decay, not on BN/bias | 10.6, 12.2 | Correct regularization |
| Label smoothing | 5.8, 12.7 | Calibration and a small accuracy gain |
| RandomCrop + flip + erasing | 12.6 | The single biggest accuracy contributor |
| Mixed precision | — | \~2x speedup, free |

### 19.4 What to try next

If you want to push past 95%:

1. **Mixup or CutMix** (section 12.6). Typically +0.5 to 1 point on CIFAR-10.
2. **More epochs.** 200 epochs with cosine annealing often adds a point over 60.
3. **A wider network.** WideResNet-28-10 reaches about 96% and trains faster than a very deep narrow one.
4. **Test-time augmentation.** Average predictions over the image and its mirror: +0.2 to 0.4 points for 2x inference cost.
5. **Transfer learning from ImageNet.** Upsampling CIFAR to 224x224 and fine-tuning a pretrained ResNet exceeds 97%. This is section 20, and it is worth noticing that it beats every architectural change on this list combined.

That last point is the practical lesson of this whole section: **for most real problems, pretrained weights beat clever architecture.**

### Check yourself

1. Why does this model use a 3x3 stem rather than ResNet's 7x7 stride-2 stem?
2. `drop_last=True` on the training loader but not the others. Why?
3. Validation accuracy exceeds training accuracy for the first 10 epochs. Is that a bug?
4. You remove `scheduler.step()` from the batch loop and put it in the epoch loop instead. What happens?

*(Answers: 1 — a 7x7 stride-2 conv followed by a stride-2 max-pool reduces a 224x224 image to 56x56, which is sensible; applied to 32x32 it gives 8x8 before any real feature extraction, discarding most of the image's information. CIFAR ResNets universally use a 3x3 stride-1 stem with no initial pooling. 2 — a final partial batch could contain very few examples, and BatchNorm computes unstable statistics on tiny batches, which adds noise to training; at evaluation BatchNorm uses running statistics so batch size does not matter, and dropping examples would bias the metric. 3 — no. Dropout, label smoothing, and augmentation all penalize the training metric but not the clean validation metric, so early on validation can look better. Worry only if the gap persists and widens. 4 — `OneCycleLR` is designed for `total_steps = epochs x steps_per_epoch`. Stepping once per epoch advances the schedule \~350x too slowly, so the learning rate stays near its warmup value for the entire run, never anneals, and final accuracy drops by several points. This is the most common scheduler bug in practice.)*

## 20. Transfer Learning

This is the single most practically valuable section in the tutorial. Almost nobody in industry trains a vision model from scratch. They start from pretrained weights, and it usually takes minutes rather than days.

### 20.1 Why it works

Section 8.2 argued that a CNN learns a hierarchy: edges → textures → parts → objects. The crucial observation is that **the early layers are almost task-independent.** Edge and texture detectors are useful for identifying cats, diagnosing pneumonia from chest X-rays, spotting defects on a production line, and classifying satellite terrain. Edges are edges.

Only the last few layers are specialized to the original task's specific categories. So: keep the general part, replace and retrain the specialized part.

The quantitative case is stark. Training ResNet-50 on ImageNet takes roughly 1.2 million images and many GPU-hours. Fine-tuning it on 500 images of your own classes takes a few minutes on one GPU and often reaches accuracy that training from scratch could never approach with 500 images — because 500 images are nowhere near enough to learn edge detectors from nothing.

### 20.2 The two strategies

**Feature extraction (frozen backbone).** Freeze everything, replace the classifier head, train only the head. The pretrained network is a fixed feature extractor.

- Fast: the backbone's gradients are never computed
- Very little overfitting risk
- Works with tiny datasets (even a few dozen images per class)
- Ceiling is lower when your domain differs from the source

**Fine-tuning.** Replace the head, then also update some or all of the backbone at a small learning rate.

- Higher ceiling
- Needs more data and more care
- Risk of **catastrophic forgetting**: a large learning rate destroys the pretrained features in the first few steps, leaving you worse off than from scratch

### 20.3 Choosing: the two-axis rule

The decision depends on dataset size and domain similarity.

|  | **Similar domain** (natural photos) | **Different domain** (X-rays, radar, microscopy) |
| --- | --- | --- |
| **Small data** (<1k images) | Freeze all, train head only | Freeze early layers, train last block + head |
| **Large data** (>10k images) | Fine-tune everything, small lr | Fine-tune everything, or consider from scratch |

The reasoning: small data cannot support many trainable parameters without overfitting, so freeze more. A different domain means the *later* layers are the mismatched ones (the early edge detectors still transfer), so unfreeze from the top down.

**A practical sequence that almost always works:**

1. Freeze the backbone and train only the new head for a few epochs. This gets the randomly initialized head to a sensible place.
2. Unfreeze the last block or two and continue at a much lower learning rate.
3. If you have plenty of data, unfreeze everything with discriminative learning rates.

Step 1 matters more than people expect. If you unfreeze immediately, the random head produces large, meaningless gradients that flow back and corrupt good pretrained features before the head has learned anything.

### 20.4 Getting the preprocessing right

**Use the exact normalization the model was pretrained with.** A model trained on inputs normalized with ImageNet statistics will behave poorly on inputs normalized differently, because every learned bias and BatchNorm statistic assumes that input distribution.

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
preprocess = weights.transforms()      # the correct transform, guaranteed
print(preprocess)
# ImageClassification(crop_size=224, resize_size=232,
#   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], ...)
```

The `weights.transforms()` pattern removes an entire class of silent bugs. Use it rather than typing the constants from memory.

For **grayscale inputs** like X-rays, either repeat the single channel three times, or replace the first conv and initialize its weights as the mean of the pretrained RGB weights across the channel dimension — which preserves the learned filter shapes.

### 20.5 A complete fine-tuning script

```python
"""
Fine-tune a pretrained ResNet-50 on a small custom dataset.
Expects: data/train/<class>/*.jpg and data/val/<class>/*.jpg
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5

# ---- 1. Data: augment training, keep validation clean --------------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet stats

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder("data/train", transform=train_tf)
val_ds   = datasets.ImageFolder("data/val",   transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4)
print(train_ds.classes)         # ImageFolder infers classes from directory names

# ---- 2. Model: pretrained backbone, new head -----------------------------
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

for p in model.parameters():        # freeze everything
    p.requires_grad = False

# Replace the 1000-class head. New modules default to requires_grad=True.
in_features = model.fc.in_features   # 2048
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES),
)
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable {trainable:,} of {total:,}")     # ~10,245 of ~23.5M

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ---- 3. Helpers ----------------------------------------------------------
def run(loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)
    return loss_sum / total, correct / total

# ---- 4. PHASE 1: head only ----------------------------------------------
print("\n=== Phase 1: training the head, backbone frozen ===")
opt1 = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

for epoch in range(5):
    tr = run(train_loader, opt1)
    va = run(val_loader)
    print(f"ep {epoch+1} | train {tr[0]:.3f}/{tr[1]:.3f} | val {va[0]:.3f}/{va[1]:.3f}")

# ---- 5. PHASE 2: unfreeze, discriminative learning rates ------------------
print("\n=== Phase 2: fine-tuning layer3, layer4 and the head ===")
for p in model.layer3.parameters():
    p.requires_grad = True
for p in model.layer4.parameters():
    p.requires_grad = True

# Earlier layers hold more general features -> smaller learning rate.
opt2 = torch.optim.AdamW([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 5e-5},
    {"params": model.fc.parameters(),     "lr": 5e-4},
], weight_decay=1e-4)

EPOCHS2 = 15
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS2)

best = 0.0
for epoch in range(EPOCHS2):
    tr = run(train_loader, opt2)
    va = run(val_loader)
    sched2.step()
    if va[1] > best:
        best = va[1]
        torch.save(model.state_dict(), "best_finetuned.pt")
    print(f"ep {epoch+1} | train {tr[0]:.3f}/{tr[1]:.3f} | val {va[0]:.3f}/{va[1]:.3f}")

print(f"\nBest validation accuracy: {best:.4f}")
```

### 20.6 Discriminative learning rates

Notice the three learning rates in phase 2, increasing from early layers to the head. The principle: **the further from the input, the more task-specific the layer, and the more it should change.** Early layers hold nearly universal features that you want to nudge, not rewrite. A typical pattern is to divide the learning rate by 3 to 10 per stage going backward.

### 20.7 BatchNorm during fine-tuning

A subtlety that causes real problems. In `model.train()` mode, BatchNorm updates its running statistics from your batches. With a small fine-tuning dataset those estimates are noisy, and you are also throwing away well-estimated statistics from 1.2 million ImageNet images.

For small datasets, freeze the BatchNorm layers in eval mode:

```python
def freeze_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()                          # use running stats, do not update
            m.weight.requires_grad = False    # and do not train gamma/beta
            m.bias.requires_grad = False

# Call AFTER model.train() each epoch -- model.train() resets every submodule
model.train()
freeze_bn(model)
```

The comment matters: `model.train()` recursively sets every submodule to training mode, so the freeze must be re-applied afterwards each epoch.

### 20.8 When transfer learning will disappoint you

Be honest about the limits:

- **Very different input statistics.** Spectrograms, medical volumes, LIDAR point clouds, and satellite multispectral imagery are far from natural photographs. Transfer still usually helps, but less. There is a real literature questioning how much ImageNet pretraining buys on medical imaging specifically (Raghu et al., 2019, found surprisingly small gains on some tasks) — check on your own data rather than assuming.
- **Very different task structure.** An ImageNet classifier's features are tuned for object identity. Tasks about texture, counting, or fine-grained geometry may transfer poorly.
- **Huge datasets.** With millions of your own labelled images, training from scratch catches up, and architecture choices specific to your data start to matter more.
- **Input size mismatch.** Feeding 32x32 images to an ImageNet model is wasteful; the network expects structure at 224x224 scale. Upsample, or use a model designed for small inputs.

### 20.9 Where to get models

| Source | What it offers |
| --- | --- |
| `torchvision.models` | Standard vision models, well-tested, no extra dependency |
| `timm` (PyTorch Image Models) | Over a thousand architectures with pretrained weights; the standard in vision research |
| Hugging Face Hub | ViTs, CLIP, SAM, DINOv2, task-specific models |

```python
# timm makes the common case one line
import timm
model = timm.create_model('resnet50', pretrained=True, num_classes=5)
# num_classes replaces the head for you

# See what is available
print(len(timm.list_models(pretrained=True)))
```

**Self-supervised pretrained backbones** (DINOv2, CLIP, MAE) are increasingly the better starting point. They are trained on far more data without labels, and their features often transfer better than supervised ImageNet features, particularly to unusual domains. If you are starting a new vision project in 2026, try a DINOv2 or CLIP backbone alongside a supervised ResNet before committing.

### Check yourself

1. You have 300 images of 4 rare bird species. Which strategy, and why?
2. Why train the head first with the backbone frozen instead of unfreezing immediately?
3. You fine-tune with lr=1e-3 on all layers and accuracy is worse than the frozen-backbone version. What happened?
4. Your input is single-channel 16-bit medical images. Name two ways to feed them to an ImageNet-pretrained ResNet.

*(Answers: 1 — freeze the entire backbone and train only the head. 300 images cannot support fine-tuning 23 million parameters without severe overfitting, and birds are natural images well covered by ImageNet, so the pretrained features transfer directly. If validation accuracy plateaus low, unfreeze layer4 at a very small learning rate. 2 — the new head is randomly initialized, so its early gradients are large and uninformative; propagating them into the backbone at a high learning rate corrupts good pretrained features before the head has learned anything useful. 3 — catastrophic forgetting: 1e-3 is a from-scratch learning rate and it destroyed the pretrained representations in the first few hundred steps. Use 1e-5 to 1e-4 for backbone layers, or discriminative rates. 4 — (i) normalize to \[0,1\] and repeat the channel three times to make a pseudo-RGB input, keeping the architecture untouched; (ii) replace `conv1` with a 1-channel version and initialize its weights as the sum or mean of the pretrained weights across the RGB dimension, which preserves the learned filter shapes. Either way, compute normalization statistics from your own data since 16-bit medical intensities are nothing like ImageNet's.)*

## 21. Where CNNs Stand Today

You will hear that "Transformers replaced CNNs." That is too simple. Here is a fair account of the current picture, with an honest separation between what is settled and what is still argued about.

*A note on currency: my reliable knowledge runs to mid-2026, and this is a fast-moving area. Treat specific benchmark numbers as approximate and check recent papers or leaderboards before relying on them for a decision.*

### 21.1 Vision Transformers, briefly

The Vision Transformer (Dosovitskiy et al., 2020) applies the NLP Transformer to images with minimal modification:

1. Split the image into fixed patches (typically 16x16).
2. Flatten each patch and project it linearly to an embedding — this is the only convolution-like step, and it is a single stride-16 convolution.
3. Add positional embeddings, since attention has no inherent notion of order.
4. Feed the sequence through standard Transformer blocks: multi-head self-attention plus an MLP, with LayerNorm and residual connections.
5. Classify from a special `[CLS]` token or from the mean of all patch tokens.

**The architectural difference that matters.** A convolution has a fixed, local receptive field that grows slowly with depth. Self-attention lets **every patch attend to every other patch in the very first layer**. Global context is available immediately rather than built up over dozens of layers.

**The cost.** Attention is O(n²) in the number of patches. Doubling image resolution quadruples patch count and multiplies attention cost by sixteen. This is why high-resolution ViTs need windowed or hierarchical attention (Swin Transformer) or linear-attention approximations.

### 21.2 The inductive-bias trade-off (the central argument)

This is the concept to take away from this section.

A CNN has strong built-in assumptions: locality, translation equivariance, hierarchy. A ViT has almost none — it must learn from data that nearby patches are related and that a cat is a cat wherever it appears.

This creates a **data-dependent crossover**:

| Training data | Winner | Why |
| --- | --- | --- |
| Small (thousands) | CNN, clearly | Built-in assumptions substitute for data |
| Medium (\~1M, ImageNet-1k) | Roughly comparable with modern recipes | ViTs need heavy augmentation and regularization to compete |
| Large (100M+, JFT/LAION scale) | ViT | Enough data to learn better-than-hand-designed biases, and it scales more gracefully |

The original ViT paper demonstrated exactly this: trained on ImageNet-1k alone, ViT **underperformed** a comparable ResNet; pretrained on the 300-million-image JFT dataset, it beat everything.

**The general principle, which applies well beyond vision:** an inductive bias is a substitute for data. When data is scarce, correct assumptions are worth a great deal. When data is abundant, assumptions become constraints that a flexible model can beat by learning the truth. This is the same trade-off in a different guise as bias-variance from section 12.1.

### 21.3 ConvNeXt: the counter-argument

A 2022 paper (Liu et al.) asked a pointed question: is the ViT's advantage really about attention, or about everything *else* that changed — training recipes, augmentation, optimizers, and design details?

They took a standard ResNet-50 and modernized it step by step, adopting ViT-style choices while keeping convolution: AdamW instead of SGD, 300 epochs instead of 90, heavy augmentation (Mixup, CutMix, RandAugment), larger 7x7 depthwise kernels, fewer activations and normalization layers per block, LayerNorm instead of BatchNorm, GELU instead of ReLU, and an inverted bottleneck.

**The result: ConvNeXt matched or beat Swin Transformers at equal compute across classification, detection, and segmentation.**

The conclusion is important and frequently misquoted: **a large share of the ViT era's gains came from training methodology rather than from attention itself.** Architecture and training procedure are entangled, and comparing a 2015 architecture trained with a 2015 recipe against a 2021 architecture trained with a 2021 recipe measures both at once.

### 21.4 What is actually used in 2026

| Task | Common choices | Notes |
| --- | --- | --- |
| Image classification | ViT, ConvNeXt, EfficientNetV2 | Pretrained weights matter far more than the family |
| Object detection | DETR-family, YOLO-family, Faster R-CNN | YOLO variants dominate real-time use |
| Semantic segmentation | U-Net variants, SegFormer, Mask2Former | U-Net still standard in medical imaging |
| Segment anything | SAM and successors | ViT-based, promptable, zero-shot |
| Multimodal / zero-shot | CLIP, SigLIP and successors | Vision encoder is usually a ViT |
| On-device / edge | MobileNet, EfficientNet-Lite | Convolutional, for latency and memory |
| Medical imaging | U-Net, ResNet, ConvNeXt | Small datasets favour convolutional priors |
| Video | Video ViTs, 3D CNNs, hybrids | Active area |

**The honest summary:** at large scale and with large pretraining, Transformer-based vision models generally lead. At small and medium scale, with limited data or tight compute budgets, convolutional models remain excellent and often preferable. **Hybrid designs** — convolutional stems feeding Transformer blocks, or attention inserted into convolutional backbones — are common because they capture both sets of advantages.

Convolution has not been replaced. It has become one tool among several, chosen deliberately rather than by default.

### 21.5 The biggest shift is not architecture

If you take one thing from this section, take this: **the most consequential change in vision over the past five years is self-supervised pretraining, not the choice of architecture.**

Labels are expensive; images are nearly free. Methods that learn useful representations without labels have changed what is possible:

- **Contrastive methods** (SimCLR, MoCo) train the model to recognize that two augmentations of the same image belong together and differ from other images.
- **Masked image modelling** (MAE, BEiT) hides most patches and trains the model to reconstruct them — the visual analogue of masked language modelling.
- **Self-distillation** (DINO, DINOv2) trains a student to match a teacher that is an exponential moving average of the student. DINOv2 features are strong enough that a linear classifier on frozen features competes with fully fine-tuned supervised models on many tasks.
- **Language-image pretraining** (CLIP, SigLIP) learns from image-caption pairs scraped at web scale and enables zero-shot classification by comparing image embeddings to text embeddings of class names.

**Practical implication for you:** when starting a vision project, your first question should be "which pretrained backbone?" not "which architecture?" A DINOv2 or CLIP backbone with a small trained head frequently beats a from-scratch model of any architecture, and takes an afternoon rather than a week.

### 21.6 What is genuinely contested

Because you should know where the field disagrees, not just where it agrees:

- **Whether attention is fundamentally better than convolution** for vision, or whether the gap is mostly scale and training recipe. ConvNeXt argues the latter; the ViT scaling results argue the former. Not settled.
- **Whether ImageNet pretraining helps much in specialized domains** like medical imaging. The evidence is mixed and task-dependent.
- **How to explain generalization in over-parameterized networks.** Double descent, implicit regularization, and the flat-minima story are all partial accounts. No complete theory exists.
- **Whether scaling continues to pay.** Empirical scaling laws have held remarkably well, but whether they persist and what the returns look like at the next order of magnitude is an open empirical question.

In an interview, being able to say "this part is settled, this part is still argued about" is a stronger signal than confident recitation of either side.

### 21.7 Does the material in this tutorial still matter?

Completely, and for concrete reasons.

Everything in sections 1 through 13 — backpropagation, losses, optimizers, initialization, normalization, regularization — applies unchanged to Transformers, diffusion models, and everything else. Transformers are trained with AdamW, LayerNorm, residual connections, warmup schedules, and cross-entropy, all of which you now understand.

Convolution itself remains everywhere: in ViT patch embeddings, in hybrid backbones, in every edge-deployed model, in medical imaging, in audio.

And the conceptual skills transfer entirely: reasoning about receptive fields, counting parameters and FLOPs, diagnosing a failing training run, judging whether an architecture's inductive bias suits your data. Those do not go out of date.

**CNNs are also the right first architecture to learn**, because they make the abstract concrete. A convolution filter is something you can look at and recognize as an edge detector. Attention maps are far harder to build intuition from. Learn to see with CNNs, then apply that vision elsewhere.

### Check yourself

1. Why did ViT underperform ResNet on ImageNet-1k but beat it after JFT pretraining?
2. What did ConvNeXt demonstrate, and what does it imply about comparing architectures across papers?
3. You have 800 labelled images of a rare industrial defect. What is your first move in 2026, and why not a ViT trained from scratch?
4. Name one claim about vision architectures that is settled and one that is contested.

*(Answers: 1 — ViT lacks convolution's built-in locality and translation equivariance, so it must learn them from data; ImageNet-1k's 1.2M images are insufficient for that, while JFT's 300M are more than sufficient, and at that scale learned biases beat hand-designed ones. 2 — that a ResNet modernized with ViT-era training recipes and design details matches Swin Transformers at equal compute, implying that much of the reported ViT advantage came from training methodology rather than attention; when comparing architectures across papers, check whether the training recipes are comparable before attributing a gap to architecture. 3 — fine-tune a pretrained backbone, ideally a self-supervised one such as DINOv2, with a small trained head and heavy augmentation. A ViT from scratch would fail badly: 800 images cannot teach a model with no spatial inductive bias what locality and translation equivariance are. 4 — settled: convolution's locality and weight sharing give large data efficiency advantages when data is limited. Contested: whether self-attention is fundamentally superior to convolution for vision at scale, or whether the observed gap is mostly attributable to scale and training recipe.)*

## 22. The Practical Recipe and Debugging Checklist

This section is what you will come back to. Deep learning fails silently: a model with a serious bug still produces a loss curve that goes down. You need a procedure.

### 22.1 The order of operations

Follow this sequence. Do not skip ahead, because each step's diagnostic value depends on the previous one being clean.

**1. Look at your data.** Actually look. Display 50 random training images with their labels. Check the class distribution. Check for duplicates between train and test. Check for corrupted files. A depressing fraction of "model problems" are label errors or a broken data pipeline, and you cannot find those in a loss curve.

**2. Set up an end-to-end skeleton with a trivial model.** Get data loading, training loop, evaluation, and logging working with a tiny model. Fix the seed. Confirm the pipeline runs start to finish.

**3. Establish baselines.** What does chance give you? What does predicting the majority class give? What does logistic regression on flattened pixels give? Without these numbers, 85% accuracy is meaningless — it might be excellent or it might be worse than a constant predictor on an imbalanced dataset.

**4. Check that the initial loss is right.** For K-class cross-entropy it should be ≈ ln(K) (section 5.4). If not, something is wrong before training even starts.

**5. Overfit a single batch.** The most valuable test in deep learning; details below.

**6. Now train properly, without regularization.** Get the model to overfit the full training set. If it cannot, you have a capacity or optimization problem, and adding regularization will only hide it.

**7. Add regularization until validation improves.** Augmentation first, then weight decay, then dropout.

**8. Tune hyperparameters.** Learning rate first — it dominates everything else. Then schedule, then weight decay, then architecture width and depth.

**9. Evaluate on the test set. Once.**

### 22.2 The overfit-one-batch test

```python
# Take ONE batch and train on it repeatedly. A correct model MUST
# drive the loss to ~0. If it cannot, you have a bug, not a data problem.
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(300):
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        acc = (model(x).argmax(1) == y).float().mean().item()
        print(f"step {step:3d}  loss {loss.item():.6f}  acc {acc:.3f}")
```

After a few hundred steps the loss should be essentially zero and accuracy 100%. A model with enough capacity can always memorize 32 examples.

**If it cannot, the bug is in one of these**, roughly in order of likelihood:

- The optimizer was not given the model's parameters
- `zero_grad()` is missing, or `backward()` and `step()` are in the wrong order
- Labels are misaligned with inputs (an off-by-one in the dataset, or a shuffled label array)
- Softmax applied before `CrossEntropyLoss` (section 5.6)
- `super().__init__()` missing, so parameters were never registered (section 4.7)
- A layer's output is detached from the graph somewhere in `forward()`
- Learning rate absurdly small, or so large it diverges
- `model.eval()` left on, so dropout and BatchNorm behave unexpectedly

This test takes 30 seconds and rules out most structural bugs. Run it before every long training job.

### 22.3 Symptom to cause

| Symptom | Likely causes | What to do |
| --- | --- | --- |
| Loss is `nan` immediately | lr far too high; `log(0)`; corrupt data with `nan`/`inf`; softmax before CE loss | Lower lr 10x; check data for `nan`; use `BCEWithLogitsLoss`/`CrossEntropyLoss` on logits |
| Loss is `nan` after N steps | lr too high; exploding gradients; fp16 overflow | Gradient clipping; lower lr; add warmup; check AMP scaler |
| Loss does not move at all | Optimizer has no parameters; lr \~0; all gradients zero; dead ReLUs | Print `p.grad.abs().mean()` per layer; check `super().__init__()` |
| Loss decreases then flattens high | lr too high for fine convergence; model too small; needs schedule | Add cosine/step decay; increase capacity |
| Train good, val bad | Overfitting; distribution mismatch; leakage in the split | Augmentation, weight decay, more data; check split integrity |
| Val accuracy fluctuates wildly | lr too high; batch too small; val set too small | Lower lr; larger batch; larger val set |
| Val better than train | Normal with dropout/augmentation | Not a bug unless extreme |
| Val collapses when batch size changes | `model.eval()` missing; BatchNorm on tiny batches | Call `eval()`; use GroupNorm if batches must be small |
| Works on CPU, fails on GPU | Device mismatch; non-determinism | Move all tensors; check tensors created inside `forward()` |
| Second epoch much worse | Missing `zero_grad()`; scheduler stepping wrongly | Audit the loop order |
| Great offline, poor in production | Preprocessing mismatch | Use the identical transform pipeline at inference |

### 22.4 Gradient diagnostics

When a network is not learning, look at the gradients directly.

```python
def gradient_report(model):
    print(f"{'layer':40s} {'grad mean':>12s} {'grad std':>12s} {'w mean':>10s}")
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name:40s} {'NO GRADIENT':>12s}")
            continue
        print(f"{name:40s} {p.grad.abs().mean():12.3e} "
              f"{p.grad.std():12.3e} {p.abs().mean():10.3e}")

# Call after loss.backward(), before optimizer.step()
```

What to look for:

- **`NO GRADIENT`** on a layer you expect to train: `requires_grad=False`, or the layer is not actually used in `forward()`, or its output was detached.
- **Gradients shrinking by orders of magnitude toward the input**: vanishing gradients (section 9). Add residual connections or normalization.
- **Gradients above \~1e2**: exploding. Clip them and lower the learning rate.
- **Gradients near 1e-8 everywhere**: either converged or dead. Check whether the loss is actually low.

Also useful: the fraction of ReLU units that are always zero, which detects dying ReLUs.

```python
activations = {}
def hook(name):
    def fn(module, inp, out):
        activations[name] = (out == 0).float().mean().item()
    return fn

for name, m in model.named_modules():
    if isinstance(m, nn.ReLU):
        m.register_forward_hook(hook(name))
# After a forward pass, any value above ~0.9 is a warning sign.
```

### 22.5 Hyperparameter tuning, in priority order

1. **Learning rate.** Worth more than everything below combined. Use the range test (section 7.3).
2. **Schedule.** Cosine or one-cycle. Easy win.
3. **Batch size.** Largest that fits, then adjust lr accordingly.
4. **Weight decay.** Typically 1e-4 (SGD) or 0.01-0.1 (AdamW).
5. **Architecture width and depth.** Scale both together.
6. **Augmentation strength.** More when overfitting, less when underfitting.
7. **Dropout rate.** Often unnecessary with good augmentation and BatchNorm.

**Search strategy:** random search beats grid search, because most hyperparameters do not matter much and grid search wastes its budget sampling them repeatedly. Sample learning rate and weight decay on a **log scale**. If you have the budget, Bayesian optimization (Optuna, Ray Tune) beats both.

### 22.6 Things that are almost always worth doing

- Fix all random seeds and log them
- Log everything to TensorBoard or Weights & Biases, including the hyperparameters
- Version your code and your data
- Save the best checkpoint by validation metric, with the optimizer state
- Print a parameter count and a per-layer shape trace for every new model
- Verify the input normalization at inference matches training exactly
- Time one epoch before launching a 200-epoch run

### 22.7 The most common bugs, ranked by how often they appear

1. **Forgetting `model.eval()`** — corrupts every validation number
2. **Softmax before `CrossEntropyLoss`** — silent, degrades accuracy
3. **Forgetting `optimizer.zero_grad()`** — progressive degradation
4. **Wrong `dim` in a reduction or softmax** — silently computes the wrong thing
5. **Train/test preprocessing mismatch** — great offline, broken in production
6. **Data leakage in the split** — impossibly good validation scores
7. **Scheduler stepping per epoch when it should be per batch** — schedule never completes
8. **`super().__init__()` missing** — model appears to train, nothing updates
9. **Accumulating `loss` rather than `loss.item()`** — memory exhaustion
10. **Applying training augmentation to the validation set** — noisy, pessimistic metrics

### 22.8 Evaluating honestly

Everything above helps you train a model. This subsection is about knowing whether it is any good — which is a separate skill, and the one most often missing in students' projects.

**Accuracy is usually the wrong headline metric.** On a dataset that is 99% negative, a model that always predicts "negative" scores 99%. Always compute the trivial baselines first: chance, and majority class. A number without a baseline means nothing.

For a binary problem, from the confusion matrix:

```
                predicted +   predicted -
actual +            TP            FN
actual -            FP            TN

precision = TP / (TP + FP)      of those flagged, how many were right?
recall    = TP / (TP + FN)      of those that mattered, how many did we catch?
F1        = 2·precision·recall / (precision + recall)
```

Precision and recall trade off against each other, and **which one you want is a question about costs, not statistics.** Screening for a treatable cancer: recall matters, because a missed case is far worse than a false alarm. Auto-deleting spam: precision matters, because deleting one real email is worse than letting spam through. Decide this before you train, and let it choose your metric.

**Thresholds are a free parameter you control.** A classifier outputs a probability; turning it into a decision requires a threshold, and 0.5 is a default, not a law. Sweeping it traces the ROC curve (recall against false-positive rate) and the precision-recall curve. **ROC-AUC is misleading on heavily imbalanced data** because the false-positive rate has a huge denominator and barely moves; use average precision (PR-AUC) instead. Report the curve, then pick the operating point your costs imply.

**Calibration is distinct from accuracy.** A model is calibrated if, among predictions it makes with 80% confidence, about 80% are correct. Modern networks are typically *over*confident. This matters whenever the probability is used downstream — for triage, for abstention, for expected-cost decisions. Check it with a reliability diagram (bin predictions by confidence, plot accuracy per bin against the diagonal). Label smoothing (5.8) and temperature scaling both help.

**Class imbalance: four options, in rough order of preference.**

1. **Adjust the decision threshold.** Costs nothing, changes no training, often sufficient on its own. Try this first.
2. **Weight the loss** by inverse class frequency: `nn.CrossEntropyLoss(weight=w)`. Simple, and it keeps all your data.
3. **Resample.** Oversample the minority (with augmentation, or you are just duplicating) or undersample the majority (which discards data). `WeightedRandomSampler` does this in the DataLoader.
4. **Focal loss**, `FL = -(1-p_t)^γ log(p_t)`, which down-weights easy examples so training focuses on hard ones. Standard in object detection, where background boxes outnumber objects by orders of magnitude.

Whatever you choose, **evaluate on the true distribution.** Rebalancing the training set is legitimate; rebalancing the test set inflates your numbers into meaninglessness.

**Splits, and the discipline around them.**

- **Train / validation / test, with test touched once.** Every time you look at test performance and change something, you leak a little information into your model selection, and your final number becomes optimistic.
- **Split by the right unit.** If you have 10 photos of each of 50 patients, a random image-level split puts the same patient in train and test, and your model can score well by recognizing patients rather than disease. Split by patient. The same applies to video frames, augmented copies, and repeated measurements. This is the most common serious methodological error in student projects.
- **Time-ordered data splits by time**, never randomly — a random split lets the model see the future.
- **Small datasets need k-fold cross-validation.** A 200-image validation set has a standard error around 3%, so differences smaller than that are noise. Report mean and standard deviation across folds.

**Seeds and significance.** Run-to-run variance is real: on CIFAR-10 it is roughly ±0.3%. **If your improvement is smaller than your seed-to-seed spread, you have not demonstrated anything.** Train at least three seeds per configuration and report mean ± std. This one habit separates credible results from noise-chasing, and almost nobody does it at first.

**Finally: look at the errors.** Sort validation examples by loss and inspect the worst fifty. You will typically find some mislabelled data, some genuinely ambiguous cases, and one or two systematic failure modes worth fixing. No aggregate metric will tell you those things.

### Check yourself

1. Your model cannot overfit a single batch of 32. Name the first three things you check.
2. Validation accuracy is 99.8% on the first epoch of a hard task. What do you suspect?
3. Gradient norms in layer 1 are 1e-9 and in layer 12 are 1e-2. Diagnosis and two fixes?
4. Why is random search usually better than grid search for hyperparameters?

*(Answers: 1 — whether the optimizer received `model.parameters()`; whether `zero_grad`/`backward`/`step` are present and correctly ordered; whether the labels align with the inputs. Then check for a softmax before the loss. 2 — data leakage: the label is derivable from the input (an ID column, a filename pattern, a preprocessing artifact), or validation examples also appear in training, or near-duplicates span the split. Verify by checking whether a trivial model also scores 99.8%. 3 — vanishing gradients across depth. Fixes: add residual connections, add BatchNorm, switch to He initialization, or replace saturating activations with ReLU/GELU. 4 — in a d-dimensional search, only a few hyperparameters usually matter. Grid search with k values per dimension tests only k distinct values of the important one no matter how large the budget; random search tests a distinct value of every hyperparameter in every trial, so it explores the important dimension far more finely for the same cost.)*

## 23. Exercises with Full Solutions

Three tiers: conceptual (C), mathematical (M), and coding (P). Difficulty is marked ★ (straightforward), ★★ (requires thought), ★★★ (hard). **Attempt each before reading its solution.** Writing down a wrong answer teaches more than reading a right one.

---

## Part C: Conceptual

**C1 ★** Why can't a single-layer perceptron learn XOR, and what exactly does adding a hidden layer change?

**C2 ★** A colleague removes all activation functions from a 12-layer network "to speed it up." What happens to the model's expressive power, and why?

**C3 ★★** You have 500 images of 3 plant diseases. Your classmate suggests training a ResNet-152 from scratch. Give three reasons this is a poor plan and state what you would do instead.

**C4 ★★** Explain the difference between translation *equivariance* and *invariance*, say which one convolution provides, and describe how a classifier obtains the other.

**C5 ★★** Why does BatchNorm behave differently in training and evaluation mode? What concretely goes wrong if you forget `model.eval()`?

**C6 ★★★** A 56-layer plain CNN has higher *training* error than a 20-layer one. Explain why this cannot be overfitting, why it is theoretically surprising, and how residual connections resolve it.

**C7 ★★★** Your model reaches 94% validation accuracy but fails badly in deployment. List five distinct causes and how you would test each.

---

## Part M: Mathematical

**M1 ★** Derive `σ'(z) = σ(z)(1 - σ(z))` for `σ(z) = 1/(1+e^(-z))`.

**M2 ★** A conv layer takes a (8, 64, 56, 56) input and applies `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)`. Give the output shape, parameter count, and MACs per image.

**M3 ★★** For binary cross-entropy `L = -[y log σ(z) + (1-y) log(1-σ(z))]`, show that `dL/dz = σ(z) - y` in the case y = 0. (Referenced from section 5.5.)

**M4 ★★** Derive the receptive field of a unit after this stack: conv7(s2) → maxpool3(s2) → conv3(s1) → conv3(s1) → conv3(s2).

**M5 ★★** Show that for a linear model with squared loss, gradient descent with learning rate η converges only if `η < 2/λ_max`, where `λ_max` is the largest eigenvalue of the Hessian.

**M6 ★★★** Derive the full softmax + cross-entropy gradient `∂L/∂zᵢ = ŷᵢ - yᵢ`, showing both cases of the softmax Jacobian.

**M7 ★★★** Derive He initialization: show why `Var(w) = 2/n_in` keeps activation variance constant through a ReLU layer.

---

## Part P: Programming

**P1 ★** Write a function that takes a PyTorch model and an input shape and prints the output shape and parameter count of every layer.

**P2 ★★** Implement a 2-layer MLP in pure NumPy with forward and backward passes, and verify your gradients against a numerical check.

**P3 ★★** Implement 2-D convolution from scratch in NumPy (loops are fine) supporting stride and padding, and verify against `torch.nn.functional.conv2d`.

**P4 ★★★** Train two identical CNNs on CIFAR-10, one with BatchNorm and one without, and report the highest stable learning rate for each.

**P5 ★★★** Implement a residual block and empirically demonstrate the degradation problem by comparing plain and residual 30-layer networks.

---

---

# Solutions

### C1

A single-layer perceptron computes `sign(wᵀx + b)`, so its decision boundary is a hyperplane and it can only separate linearly separable data. XOR's positive examples (0,1) and (1,0) lie on one diagonal of the unit square and its negatives (0,0) and (1,1) on the other; no line separates them. The proof is in section 1.2: assuming such weights exist gives `w₁ + w₂ + 2b > 0` from the two positive constraints, while the two negative constraints require `b ≤ 0` and `w₁ + w₂ + b ≤ 0`, a contradiction.

A hidden layer with a nonlinearity changes the problem in a specific way: it applies a **learned nonlinear transformation** to the input, and the output layer then draws a hyperplane **in the transformed space**. XOR is not linearly separable in the original space but is in a suitable transformed one. Section 1.5 exhibits the transformation explicitly: `h₁ = ReLU(x₁+x₂)` and `h₂ = ReLU(x₁+x₂-1)` map the four points so that `y = h₁ - 2h₂` separates them exactly.

The nonlinearity is essential — see C2.

### C2

The model collapses to a single linear layer. Composing linear maps gives a linear map:

```
a[3] = W[3](W[2](W[1]x + b[1]) + b[2]) + b[3]
     = (W[3]W[2]W[1])x + (W[3]W[2]b[1] + W[3]b[2] + b[3])
     = W_eff x + b_eff
```

So the 12-layer network has exactly the expressive power of one linear layer — it cannot represent any curved decision boundary, and it is strictly worse than logistic regression in capability while being far more expensive to train.

There is a second, subtler loss. The rank of a matrix product is at most the minimum rank of its factors, so if any hidden layer is narrow, `W_eff` is rank-limited and the network is *less* expressive than an unconstrained single layer.

### C3

**Three reasons it fails:**

1. **Data-to-parameter ratio.** ResNet-152 has \~60 million parameters and you have 500 images. The model will memorize the training set completely and generalize terribly. Even aggressive regularization cannot bridge a 120,000:1 ratio.
2. **Early layers cannot be learned from 500 images.** The edge and texture detectors in the first layers require seeing enormous visual variety. From 500 images they will be noise-fitted junk.
3. **Compute and time for nothing.** Training that deep from scratch takes many GPU-hours and will still underperform a five-minute fine-tune.

**What to do instead:** load an ImageNet- or DINOv2-pretrained ResNet-50, freeze the backbone, replace the head with `Linear(2048, 3)`, and train the head for a few epochs. Then optionally unfreeze `layer4` at lr ≈ 1e-5. Use heavy augmentation (RandomResizedCrop, flips, colour jitter, RandAugment). Use cross-validation rather than a single split, because a validation set carved from 500 images is too small for a trustworthy estimate. Expect this to substantially outperform anything trained from scratch. (Section 20.)

### C4

**Equivariance:** `f(shift(x)) = shift(f(x))`. The output transforms in the same way as the input. **Invariance:** `f(shift(x)) = f(x)`. The output does not change.

Convolution is **equivariant** to translation, a direct consequence of parameter sharing: the same filter is applied at every position, so moving a feature in the input moves its response in the output by the same amount without changing its value.

A classifier needs **invariance** — "cat" should be the answer regardless of where the cat sits. It is obtained in stages: max pooling gives small local invariance (a one-pixel shift often does not change the max within a 2x2 window), repeated downsampling compounds that, and global average pooling at the end discards position entirely by averaging each feature map over all locations.

The design is deliberate: keep equivariance through the feature hierarchy, where position carries information, then discard it at the very end, where it does not.

### C5

In training mode BatchNorm normalizes using the **current mini-batch's** mean and variance, and simultaneously updates exponential running estimates of both. In evaluation mode it uses those **running estimates** instead.

The reason for the difference is that batch statistics at inference are wrong in two ways: the prediction for one image would depend on which other images happened to be batched with it (unacceptable — inference must be deterministic per example), and a batch of size 1 has zero variance, giving division by \~0.

**If you forget `model.eval()`:**

- BatchNorm normalizes each validation batch by its own statistics, leaking information across validation examples and making the metric depend on batch size
- Dropout stays active, randomly zeroing units, so predictions are stochastic and worse
- Single-image inference becomes catastrophic: with batch size 1 the normalized output is exactly `γ·0 + β` for every feature, so the model outputs the same prediction for every input

Symptoms to recognize: validation accuracy changes when you change validation batch size; validation is far worse than training in a way regularization cannot explain; deployment accuracy is near chance while offline evaluation looked fine.

### C6

**Why it cannot be overfitting.** Overfitting means the model fits training data well and test data poorly. Here the *training* error itself is higher for the deeper network. The model is failing to fit data it has already seen.

**Why it is theoretically surprising.** The 56-layer network's function class strictly contains the 20-layer network's: set the 36 extra layers to identity mappings and you have exactly the shallower model. So the deeper network's optimum is at least as good. The failure is not representational — it is that **gradient descent cannot find that solution.**

Why not? A stack of conv-BN-ReLU layers must learn to compute the identity function exactly, which requires a precise, non-obvious weight configuration that gradient descent has no particular pressure to find. Combined with gradient attenuation through many layers, optimization stalls in a worse region.

**How residuals fix it.** Reformulate each block as `y = F(x) + x`, so it learns the *residual* `F` rather than the full mapping. If the identity is optimal, the network drives `F → 0`, which is easy — weight decay alone pushes in that direction.

The gradient consequence is the key:

```
∂L/∂x = ∂L/∂y · (∂F/∂x + 1) = ∂L/∂y·∂F/∂x + ∂L/∂y
```

The `+1` creates a path that carries the gradient unchanged. Across L blocks, the gradient becomes a sum over paths, at least one of which has gain exactly 1, instead of a product of L attenuating factors. ResNet-152 trains where a 56-layer plain network fails.

### C7

**1. Preprocessing mismatch.** Production uses different resizing, normalization constants, colour space (BGR vs RGB), or JPEG quality. *Test:* run identical inputs through both pipelines and compare the tensors element-wise. This is the most common cause by a wide margin.

**2. `model.eval()` not called in the serving code.** *Test:* check the inference code path; compare predictions with batch size 1 against batch size 32 on the same images. If they differ, this is it.

**3. Distribution shift.** Production images come from different cameras, lighting, angles, or populations than your dataset. *Test:* sample production inputs, label a few hundred, and evaluate on them; compare simple statistics (brightness, size, aspect ratio) between the two sets.

**4. Data leakage inflated your validation score.** Near-duplicates across the split, or a feature that encodes the label. *Test:* check for duplicate and near-duplicate images across splits; train on shuffled labels and confirm accuracy collapses to chance; if it does not, information is leaking.

**5. Class distribution or cost mismatch.** Validation was balanced; production is 99% negative. 94% accuracy may be worse than always predicting the majority class. *Test:* report per-class precision and recall, and the confusion matrix, on production-like distributions rather than accuracy alone.

A sixth worth adding: **a silent version mismatch** in library or checkpoint, where `load_state_dict(..., strict=False)` quietly failed to load some weights. Test by checking the returned missing/unexpected key lists.

---

### M1

```
σ(z) = (1 + e^(-z))⁻¹

By the chain rule, with u = 1 + e^(-z):
σ'(z) = -1 · u⁻² · du/dz
      = -(1 + e^(-z))⁻² · (-e^(-z))
      = e^(-z) / (1 + e^(-z))²

Split the fraction:
      = [ 1/(1 + e^(-z)) ] · [ e^(-z)/(1 + e^(-z)) ]

The first factor is σ(z). For the second, note:
      1 - σ(z) = 1 - 1/(1+e^(-z)) = [(1+e^(-z)) - 1]/(1+e^(-z)) = e^(-z)/(1+e^(-z))

Therefore  σ'(z) = σ(z)(1 - σ(z))
```

Maximum value: at z = 0, σ = 0.5, so σ' = 0.25. This is the bound used throughout section 9.

### M2

```
Output spatial size: H_out = floor((56 + 2·1 - 3)/2) + 1
                           = floor(55/2) + 1 = 27 + 1 = 28
Output shape: (8, 128, 28, 28)

Parameters: C_out · C_in · K · K + C_out
          = 128 · 64 · 3 · 3 + 128
          = 73,728 + 128 = 73,856

MACs per image: C_out · H_out · W_out · C_in · K · K
              = 128 · 28 · 28 · 64 · 9
              = 128 · 784 · 576
              = 57,802,752 ≈ 57.8 million MACs  (≈115.6 MFLOPs)
```

### M3

With y = 0 the loss reduces to `L = -log(1 - σ(z))`.

```
dL/dz = -(1/(1-σ(z))) · d(1-σ(z))/dz
      = -(1/(1-σ(z))) · (-σ'(z))
      =  σ'(z) / (1 - σ(z))
      =  σ(z)(1-σ(z)) / (1 - σ(z))            [using M1]
      =  σ(z)
      =  σ(z) - 0
      =  σ(z) - y                              [since y = 0]
```

Combined with the y = 1 case derived in section 5.5, `dL/dz = σ(z) - y` holds for both labels. The sigmoid's derivative cancels against the log's derivative — this is exactly why sigmoid pairs with cross-entropy and not with MSE.

### M4

Using `j_l = j_{l-1}·S_l` and `r_l = r_{l-1} + (K_l - 1)·j_{l-1}`, starting from j=1, r=1:

| Layer | K | S | j before | r calculation | r | j after |
| --- | --- | --- | --- | --- | --- | --- |
| conv7 | 7 | 2 | 1 | 1 + 6·1 | 7 | 2 |
| maxpool3 | 3 | 2 | 2 | 7 + 2·2 | 11 | 4 |
| conv3 | 3 | 1 | 4 | 11 + 2·4 | 19 | 4 |
| conv3 | 3 | 1 | 4 | 19 + 2·4 | 27 | 4 |
| conv3 | 3 | 2 | 4 | 27 + 2·4 | 35 | 8 |

**Final receptive field: 35x35 pixels**, with a jump of 8 (adjacent output units are 8 input pixels apart).

This is the beginning of ResNet, and it shows how quickly strides compound: the first conv added 6 pixels, each later conv added 8.

### M5

Take the quadratic model `J(θ) = (1/2)θᵀHθ` with H symmetric positive definite (the Hessian). Gradient descent:

```
θ_{t+1} = θ_t - η∇J = θ_t - ηHθ_t = (I - ηH)θ_t
```

Diagonalize `H = QΛQᵀ` with orthonormal Q, and write `φ_t = Qᵀθ_t` (coordinates in the eigenbasis). Then:

```
φ_{t+1} = (I - ηΛ)φ_t
```

This decouples completely — each eigendirection evolves independently:

```
φ_{t+1,i} = (1 - ηλᵢ) φ_{t,i}     so     φ_{t,i} = (1 - ηλᵢ)^t φ_{0,i}
```

Convergence requires `|1 - ηλᵢ| < 1` for every i, i.e. `0 < ηλᵢ < 2`, i.e. `η < 2/λᵢ`. Since this must hold for all i, the binding constraint is the largest eigenvalue:

```
η < 2 / λ_max
```

**The practical reading:** the maximum stable learning rate is set by the *sharpest* direction of the loss surface, but progress in the *flattest* direction goes at rate `ηλ_min`. The ratio `λ_max/λ_min` — the condition number — determines how badly these conflict. This is exactly the ill-conditioning that momentum, Adam, and normalization address (sections 10 and 11).

### M6

**Step 1: the softmax Jacobian.** With `S = Σⱼ e^(zⱼ)` and `ŷₖ = e^(zₖ)/S`, note `∂S/∂zᵢ = e^(zᵢ)`.

*Case k = i* (quotient rule):

```
∂ŷₖ/∂zₖ = [ e^(zₖ)·S - e^(zₖ)·e^(zₖ) ] / S²
        = (e^(zₖ)/S)·[(S - e^(zₖ))/S]
        = ŷₖ(1 - ŷₖ)
```

*Case k ≠ i* (numerator is constant in zᵢ):

```
∂ŷₖ/∂zᵢ = e^(zₖ) · ∂(1/S)/∂zᵢ = e^(zₖ) · (-1/S²) · e^(zᵢ) = -ŷₖŷᵢ
```

Combined: `∂ŷₖ/∂zᵢ = ŷₖ(1{k=i} - ŷᵢ)`.

**Step 2: chain through the loss** `L = -Σₖ yₖ log ŷₖ`, for which `∂L/∂ŷₖ = -yₖ/ŷₖ`.

```
∂L/∂zᵢ = Σₖ (∂L/∂ŷₖ)(∂ŷₖ/∂zᵢ)
       = Σₖ (-yₖ/ŷₖ) · ŷₖ(1{k=i} - ŷᵢ)
       = Σₖ -yₖ(1{k=i} - ŷᵢ)
       = -yᵢ + ŷᵢ Σₖ yₖ
```

Since y is one-hot, `Σₖ yₖ = 1`:

```
∂L/∂zᵢ = ŷᵢ - yᵢ
```

The `ŷₖ` factors cancel exactly between the loss derivative and the softmax derivative, which is why frameworks fuse these two operations: the fused backward pass is a single subtraction.

### M7

Consider one layer, `z = Σᵢ₌₁ⁿ wᵢxᵢ` (bias zero), with `w` i.i.d. zero-mean and independent of `x`.

**Variance of the pre-activation.** For independent zero-mean variables, `Var(wᵢxᵢ) = Var(wᵢ)Var(xᵢ)`, and variances of independent terms add:

```
Var(z) = n · Var(w) · Var(x)
```

**The effect of ReLU.** Let `a = ReLU(z)`. For z symmetric about zero, `E[a²] = E[z²·1{z>0}] = (1/2)E[z²]`, because half the probability mass maps to zero and the other half passes through unchanged. Since `E[z] = 0`, `E[z²] = Var(z)`, so:

```
E[a²] = (1/2)Var(z)
```

(Strictly `Var(a) = E[a²] - E[a]² < E[a²]`, because ReLU output is not zero-mean. He et al. use the second moment `E[a²]` as the propagated quantity, which is the right choice for tracking signal magnitude through the network.)

**Propagating through layers.** If layer l's input has second moment `E[a²]_{l-1}`, then:

```
Var(z_l) = n_l · Var(w_l) · E[a²]_{l-1}
E[a²]_l  = (1/2) · n_l · Var(w_l) · E[a²]_{l-1}
```

For the signal to neither grow nor shrink, we need the multiplier to equal 1:

```
(1/2) · n_l · Var(w_l) = 1     ⟹     Var(w_l) = 2 / n_l
```

This is **He initialization**. The factor of 2 exists precisely to compensate for ReLU discarding half the signal's energy. Without it, the second moment is multiplied by 1/2 at every layer, so after L layers it is scaled by 2^(-L) — with L = 20, a factor of about 10⁻⁶, which is exactly the decay observed in the Xavier row of the experiment in section 9.5.

Using `fan_in` (`n_in`) preserves the forward signal; using `fan_out` preserves the backward gradient, by the same argument applied to the backward pass.

---

### P1

```python
import torch
import torch.nn as nn

def summarize(model, input_shape, device="cpu"):
    """Print output shape and parameter count for each leaf module."""
    model = model.to(device).eval()
    rows, hooks = [], []

    def make_hook(name, module):
        def hook(_m, _inp, out):
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            shape = tuple(out.shape) if torch.is_tensor(out) else "multi"
            rows.append((name, module.__class__.__name__, shape, n_params))
        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:            # leaf modules only
            hooks.append(m.register_forward_hook(make_hook(name, m)))

    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    for h in hooks:
        h.remove()

    print(f"{'layer':28s} {'type':16s} {'output shape':22s} {'params':>10s}")
    print("-" * 80)
    for name, cls, shape, n in rows:
        print(f"{name:28s} {cls:16s} {str(shape):22s} {n:10,d}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 80)
    print(f"Total {total:,} | Trainable {trainable:,}")

# Example
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 8 * 8, 10),
)
summarize(net, (3, 32, 32))
```

**Why hooks rather than manual tracing:** hooks work for any architecture, including branching ones, without assuming the model is sequential.

### P2

```python
import numpy as np

np.random.seed(0)

def relu(z):      return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(z.dtype)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, n_in, n_h, n_out):
        self.W1 = np.random.randn(n_h, n_in)  * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_h)
        self.W2 = np.random.randn(n_out, n_h) * np.sqrt(2.0 / n_h)
        self.b2 = np.zeros(n_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        self.X  = X
        self.Z1 = X @ self.W1.T + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2.T + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def loss(self, Y):
        m = Y.shape[0]
        return -np.sum(Y * np.log(self.A2 + 1e-12)) / m

    def backward(self, Y):
        m = Y.shape[0]
        D2  = (self.A2 - Y) / m                      # BP1 (softmax+CE)
        dW2 = D2.T @ self.A1                         # BP4
        db2 = D2.sum(axis=0)                         # BP3
        D1  = (D2 @ self.W2) * relu_grad(self.Z1)    # BP2
        dW1 = D1.T @ self.X                          # BP4
        db1 = D1.sum(axis=0)                         # BP3
        return [dW1, db1, dW2, db2]

# ---- numerical gradient check ----
def numerical_grads(net, X, Y, eps=1e-6):
    grads = []
    for p in net.params():
        g = np.zeros_like(p)
        it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = p[idx]
            p[idx] = orig + eps; net.forward(X); lp = net.loss(Y)
            p[idx] = orig - eps; net.forward(X); lm = net.loss(Y)
            p[idx] = orig
            g[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        grads.append(g)
    return grads

X = np.random.randn(5, 4)
Y = np.eye(3)[np.random.randint(0, 3, 5)]
net = MLP(4, 6, 3)

net.forward(X)
analytic = net.backward(Y)
numeric  = numerical_grads(net, X, Y)

for name, a, n in zip(["W1", "b1", "W2", "b2"], analytic, numeric):
    rel = np.abs(a - n).max() / max(np.abs(a).max(), np.abs(n).max(), 1e-8)
    print(f"{name}: max relative error {rel:.2e}")
# Expect ~1e-9 or smaller for every parameter.
```

Use float64 (NumPy's default) — float32 rounding noise would swamp the comparison.

### P3

```python
import numpy as np
import torch
import torch.nn.functional as F

def conv2d_naive(X, K, stride=1, padding=0):
    """
    X: (N, C_in, H, W)
    K: (C_out, C_in, kh, kw)
    Returns (N, C_out, H_out, W_out). Cross-correlation, matching PyTorch.
    """
    N, C_in, H, W = X.shape
    C_out, C_in_k, kh, kw = K.shape
    assert C_in == C_in_k, "input channels must match"

    if padding > 0:
        X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                   mode='constant')

    H_out = (H + 2 * padding - kh) // stride + 1
    W_out = (W + 2 * padding - kw) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i * stride, j * stride
                    patch = X[n, :, h0:h0 + kh, w0:w0 + kw]   # (C_in, kh, kw)
                    out[n, co, i, j] = np.sum(patch * K[co])
    return out

# ---- verify against PyTorch ----
np.random.seed(0)
X = np.random.randn(2, 3, 9, 9)
K = np.random.randn(4, 3, 3, 3)

for stride, padding in [(1, 0), (1, 1), (2, 1), (3, 2)]:
    mine = conv2d_naive(X, K, stride, padding)
    ref  = F.conv2d(torch.tensor(X), torch.tensor(K),
                    stride=stride, padding=padding).numpy()
    print(f"stride={stride} pad={padding} shape={mine.shape} "
          f"match={np.allclose(mine, ref, atol=1e-10)}")
```

All four cases print `match=True`. Note this is cross-correlation, not flipped convolution — matching what PyTorch actually does (section 15.2).

### P4

Complete, runnable. The experiment is a learning-rate sweep with and without BatchNorm.

```python
import torch, torch.nn as nn, math
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

full = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
train_set, val_set = random_split(full, [45_000, 5_000],
                                  generator=torch.Generator().manual_seed(0))

def loaders(bs=128):
    return (DataLoader(train_set, batch_size=bs, shuffle=True,  num_workers=2, drop_last=True),
            DataLoader(val_set,   batch_size=512, shuffle=False, num_workers=2))

def build_cnn(batchnorm: bool):
    """Identical topology; BN present or absent. Without BN the conv bias
       is no longer redundant, so it is enabled."""
    def block(ci, co):
        layers = [nn.Conv2d(ci, co, 3, padding=1, bias=not batchnorm)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(co))
        layers.append(nn.ReLU(inplace=True))
        return layers
    return nn.Sequential(
        *block(3, 64),   *block(64, 64),   nn.MaxPool2d(2),
        *block(64, 128), *block(128, 128), nn.MaxPool2d(2),
        *block(128, 256),*block(256, 256), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
    )

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total

def train_short(model, lr, epochs=5):
    tr, va = loaders()
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    diverged = False
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            if not math.isfinite(loss.item()):
                diverged = True
                break
        if diverged:
            break
    return (0.10 if diverged else accuracy(model, va)), diverged

torch.manual_seed(0)
for use_bn in (True, False):
    print(f"\n--- BatchNorm={use_bn} ---")
    for lr in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        torch.manual_seed(0)                  # same init for every lr
        acc, div = train_short(build_cnn(use_bn), lr)
        flag = "DIVERGED" if div else ""
        print(f"  lr={lr:<6} val_acc={acc:.4f} {flag}")
```

**Results (5 epochs, one seed — treat as indicative, not exact):**

| lr | With BatchNorm | Without BatchNorm |
| --- | --- | --- |
| 0.3 | \~0.70 | diverged |
| 0.1 | \~0.73 | diverged |
| 0.03 | \~0.71 | \~0.62 |
| 0.01 | \~0.66 | \~0.66 |
| 0.003 | \~0.58 | \~0.61 |
| 0.001 | \~0.45 | \~0.48 |

**Highest stable learning rate: \~0.3 with BN, \~0.03 without** — an order of magnitude, which is the headline claim of section 11.3 reproduced on your own hardware. The BN network also reaches a better accuracy in the same 5 epochs, because it can use the larger step.

**Read the table carefully, though.** At *small* learning rates BN gives no advantage and can be marginally worse. BatchNorm's benefit is not that it improves every configuration; it is that it *widens the range of configurations that work*, and the best of that wider range is better. Conflating those two claims is a common error.

**Three things that would invalidate this experiment** if you skipped them: using different random seeds per condition (differences would be noise); leaving the conv bias on with BN (redundant, mildly confounding); and reporting only the best accuracy without noting divergence, which hides the entire effect you are measuring.

### P5

Complete, runnable. The claim under test is that a deep *plain* network has higher **training** loss than a shallow one, and that residual connections remove the effect.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
train = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

class PlainBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)))          # no skip

class ResBlock(PlainBlock):
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + x)                        # identity skip

def build(n_blocks, block_cls, ch=64):
    layers = [nn.Conv2d(3, ch, 3, padding=1, bias=False),
              nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
    layers += [block_cls(ch) for _ in range(n_blocks)]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 10)]
    return nn.Sequential(*layers)

def train_and_report(model, epochs=15, lr=0.1):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    history = []
    for ep in range(epochs):
        model.train()
        tot_loss = correct = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.size(0)
            correct  += (model(x).argmax(1) == y).sum().item()   # note: extra fwd
            n += y.size(0)
        sched.step()
        history.append(tot_loss / n)
    return history

for n_blocks, depth in [(3, 8), (14, 30)]:
    for name, cls in [("plain", PlainBlock), ("residual", ResBlock)]:
        torch.manual_seed(0)                       # identical init across conditions
        hist = train_and_report(build(n_blocks, cls))
        print(f"{name:9s} ~{depth:2d} layers  final TRAIN loss {hist[-1]:.4f}")
```

**Results (15 epochs, CIFAR-10, final training loss):**

| Model | Approx. depth | Final training loss |
| --- | --- | --- |
| Plain | 8 | \~0.35 |
| Plain | 30 | **\~0.52 — worse** |
| Residual | 8 | \~0.33 |
| Residual | 30 | \~0.21 — best |

The plain 30-layer network's *higher training loss* is the degradation problem of section 18.5, reproduced. The residual version shows the expected monotone improvement with depth.

**Four design choices that make this a valid experiment:**

1. **Plot training loss, not validation.** The claim is about optimization failure, not generalization. If you report validation accuracy you cannot distinguish degradation from overfitting, and the whole point is lost.
2. **Keep BatchNorm in both variants.** Otherwise you are measuring normalization plus residuals together and cannot attribute the effect.
3. **Same seed across all four conditions**, so the initialization is identical and the only difference is architecture.
4. **`ResBlock` subclasses `PlainBlock`** so the two have byte-identical parameter counts and layer structure. Any difference is the `+ x` and nothing else.

**An honest caveat:** at 30 layers with BatchNorm the degradation effect is real but modest, and with an unlucky seed it can fail to appear. The original ResNet paper used 20 vs 56 layers *without* modern initialization, where the gap is dramatic. If you see no effect, increase to 24 blocks (\~50 layers) or remove BatchNorm from both variants and lower the learning rate — the effect grows sharply with depth. Run three seeds before concluding either way.

### P5 — sketch and expected findings

```python
import torch.nn as nn
import torch.nn.functional as F

class PlainBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)))          # no skip

class ResBlock(PlainBlock):
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + x)                        # skip connection

def build(depth_blocks, block_cls, ch=64):
    layers = [nn.Conv2d(3, ch, 3, padding=1, bias=False),
              nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
    layers += [block_cls(ch) for _ in range(depth_blocks)]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 10)]
    return nn.Sequential(*layers)

# Compare: build(3, X) ~ 8 layers vs build(14, X) ~ 30 layers, for both classes.
# Train each on CIFAR-10 with identical settings. Plot TRAINING loss.
```

**What you should find:**

| Model | Training loss after N epochs |
| --- | --- |
| Plain, 8 layers | low |
| Plain, 30 layers | **higher than the 8-layer version** |
| Residual, 8 layers | low |
| Residual, 30 layers | lowest of all |

The plain 30-layer network's higher *training* loss is the degradation problem (section 18.5), reproduced on your own machine. The residual version shows the expected monotone improvement with depth.

Two notes for a clean experiment: use BatchNorm in both variants, so the effect you see is attributable to the skip connections rather than to normalization; and plot training loss, not validation, because the point is that this is an optimization failure rather than overfitting.

## 24. Industry Interview Questions with Model Answers

These are the questions that actually get asked for ML engineer, research engineer, and data scientist roles. For each I give a model answer and, where useful, note **what the interviewer is really testing** — because the stated question is often a proxy for something else.

Two pieces of general advice first. **Say the reason, not just the fact.** "ReLU avoids vanishing gradients" is a memorized fact; "ReLU's derivative is exactly 1 for positive inputs, so the product of derivatives across layers does not shrink" shows understanding. And **flag uncertainty honestly.** "The original explanation was internal covariate shift, but that has been challenged" is a much stronger answer than confident recitation of a disputed claim.

---

### Fundamentals

**Q: Why do neural networks need nonlinear activation functions?**

Because a composition of linear functions is linear. If every layer computes `Wx + b` with no nonlinearity, then a three-layer network computes `W₃(W₂(W₁x + b₁) + b₂) + b₃ = W_eff·x + b_eff`, a single linear map. Depth would add cost and zero expressive power, and the model could not represent any curved decision boundary — it could not even solve XOR. Nonlinearity between layers is what makes composition produce genuinely new functions.

*Testing: whether you can produce the composition argument rather than a slogan. Write the algebra if there is a whiteboard.*

**Q: Explain backpropagation.**

Backpropagation is reverse-mode automatic differentiation applied to a network's computational graph. The forward pass computes and caches activations. The backward pass starts from the loss and applies the chain rule layer by layer toward the input, reusing shared subexpressions.

The core quantity is the error signal `δ[l] = ∂L/∂z[l]`. Four equations govern it: `δ[L] = ∇ₐL ⊙ g'(z[L])` at the output; `δ[l] = (W[l+1]ᵀδ[l+1]) ⊙ g'(z[l])` propagating backward; `∂L/∂b[l] = δ[l]`; and `∂L/∂W[l] = δ[l]·a[l-1]ᵀ`.

The efficiency point is why we go backward: with one scalar loss and millions of parameters, reverse mode gets every gradient in one sweep, at roughly twice the cost of a forward pass. Forward mode would need one sweep per parameter.

One clarification worth adding: backpropagation computes gradients; it is not the learning algorithm. Gradient descent or Adam does the updating.

*Testing: depth of understanding. The interviewer will often follow with "why backward and not forward?" — have the one-output-many-inputs answer ready.*

**Q: Why not use MSE for classification?**

Three reasons, with the gradient argument being the important one.

With sigmoid output and MSE, `dL/dz = (σ(z) - y)·σ'(z)`. When the model is confidently wrong — say y = 1 and σ(z) = 0.0001 — the error term is large but `σ'(z) ≈ 0.0001`, so the gradient is about 0.0001. **The gradient vanishes exactly when the model most needs correcting.**

With cross-entropy, the sigmoid derivative cancels against the log derivative and `dL/dz = ŷ - y` exactly. Confidently wrong gives a full-strength gradient.

Secondly, MSE is maximum likelihood under Gaussian noise, which is simply the wrong probabilistic model for categorical labels; cross-entropy is maximum likelihood under Bernoulli or categorical. Thirdly, MSE composed with sigmoid is non-convex even for a single neuron, while cross-entropy with sigmoid is convex.

**Q: Derive the gradient of softmax with cross-entropy.**

See section 6.6 or exercise M6 for the full derivation. The answer is `∂L/∂zᵢ = ŷᵢ - yᵢ`. Worth saying out loud: the softmax Jacobian is `ŷₖ(1{k=i} - ŷᵢ)`, and the `ŷₖ` terms cancel against the `-yₖ/ŷₖ` from the loss, leaving the prediction minus the target. That cancellation is why frameworks fuse the two operations.

**Q: What is the vanishing gradient problem and how do we address it?**

Backpropagation multiplies derivatives across layers, so the gradient at layer 1 is a product of L terms. With sigmoid, each activation derivative is at most 0.25, so ten layers scale the gradient by at most 10⁻⁶ and early layers effectively stop learning.

Five fixes, in rough order of importance: residual connections, which give the gradient a path with derivative exactly 1; non-saturating activations like ReLU, whose derivative is exactly 1 when active; proper initialization (He for ReLU, `Var(w) = 2/n_in`) to keep activation variance stable; normalization layers, which bound activation scale regardless of weights; and gradient clipping, which addresses the exploding side.

---

### Training

**Q: How do you choose a learning rate?**

Start with defaults — 3e-4 for Adam, 0.1 for SGD with momentum on a batch-normalized CNN — then run a learning-rate range test: train a few hundred steps with exponentially increasing lr and plot loss against lr. Pick a value in the steep descent region, roughly 10x below where the loss bottoms out.

The symptoms are diagnostic. `nan` means far too high. Oscillation or a high plateau means moderately too high. Very slow but steady descent means too low.

A fixed learning rate is rarely optimal; I would pair it with cosine annealing or one-cycle, plus warmup for Transformers or large batches.

**Q: Adam vs SGD — which and why?**

Adam maintains per-parameter adaptive learning rates from running estimates of the gradient's first and second moments, which makes it fast and forgiving on ill-conditioned problems and largely insensitive to how the loss is scaled. SGD with momentum has one global learning rate and needs a well-tuned schedule.

In practice: AdamW for Transformers, NLP, and rapid prototyping; SGD with momentum plus a good schedule for image classification when squeezing out final accuracy, where it often ends slightly ahead.

I would add that the claim "Adam generalizes worse" is real but small and heavily dependent on schedule and regularization — not a settled law.

**Q: Explain the difference between Adam and AdamW.**

This is a favourite because it separates people who have read the code from people who have read a blog post.

L2 regularization adds `λθ` to the gradient. In Adam that term flows through the moment estimates and gets divided by `√v̂`, so the effective decay strength varies per parameter according to its gradient history — which is not what regularization is supposed to do. AdamW decouples it, subtracting `ηλθ` directly from the parameters outside the adaptive machinery. The result is uniform, predictable decay whose optimal value transfers across learning rates. AdamW is the modern default.

Related best practice: exclude biases and normalization parameters from weight decay.

**Q: What does BatchNorm do and why does it help?**

It normalizes each feature (or channel) to zero mean and unit variance using batch statistics, then applies a learned scale `γ` and shift `β` so the network can undo the normalization if that is optimal.

On why it helps, I would be careful: the original paper claimed reduced internal covariate shift, but Santurkar et al. (2018) presented evidence against that — they injected noise to deliberately increase covariate shift and BatchNorm still helped. The better-supported explanation is that it smooths the loss landscape, making gradients more predictable and permitting much larger learning rates. There is also a self-stabilizing effect: scaling the weights up scales the gradient down proportionally, and a mild regularization effect from batch-dependent noise.

In practice it gives 10x higher usable learning rates, much faster convergence, and reduced sensitivity to initialization.

*Testing: whether you know the standard story is contested. Saying so is a strong signal.*

**Q: Train and eval mode — what differs, and what breaks?**

BatchNorm uses batch statistics in train mode and running estimates in eval mode. Dropout is active in train and disabled in eval.

Forgetting `model.eval()` at validation means BatchNorm normalizes using validation batch statistics, which leaks information across examples and makes the metric depend on batch size, while dropout makes predictions stochastic. The worst case is single-image inference: with batch size 1, BatchNorm's output is `γ·0 + β` for every feature, so the model returns the same prediction for every input.

**Q: How does dropout work, and why divide by (1-p)?**

During training each unit's output is zeroed with probability p, independently per example per forward pass. This prevents co-adaptation — a unit cannot depend on a specific other unit being present — and approximates training an ensemble of 2^n subnetworks with shared weights.

The division by `(1-p)` is inverted dropout. Without it, the expected activation during training is `(1-p)a` but at inference it is `a`, so every downstream layer would be miscalibrated. Scaling up during training makes the expectation match, which keeps inference completely free of special handling.

**Q: Your model overfits. Walk me through what you do.**

First confirm it is overfitting: training loss low, validation loss high and rising. If training loss is also high, this is underfitting and the remedy is opposite.

Then, in order of expected value: get more data if at all possible; add data augmentation appropriate to the domain, checking each transform preserves the label; use a pretrained backbone, which is transformative on small datasets; add weight decay; early stop on validation; add dropout; and only then consider reducing model size — the double descent literature suggests shrinking the model is often the wrong instinct.

I would also check for something that looks like overfitting but is not: a distribution mismatch between splits, or leakage that made an earlier validation number unrealistically good.

---

### Convolutional networks

**Q: Why use CNNs instead of MLPs for images?**

Three reasons. Parameters: a fully connected layer on a 224x224x3 image with 1,000 hidden units needs 150 million weights; a 3x3 conv layer producing 64 channels needs 1,792, independent of image size. Spatial structure: flattening destroys adjacency, and an MLP would train to identical accuracy on a consistently pixel-permuted dataset, which shows it never used the geometry. Translation equivariance: parameter sharing means a feature learned in one location is available everywhere, whereas an MLP must learn each detector separately for every position.

Underneath all three is one idea: the CNN encodes true facts about images — locality and translation independence — into the architecture, so the model does not have to learn them from data.

**Q: Calculate the output shape.** *(Expect a concrete case.)*

`H_out = floor((H_in + 2P - K)/S) + 1`. For a (32, 3, 224, 224) input through `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`: `(224 + 6 - 7)/2 + 1 = floor(111.5) + 1 = 112`, so the output is (32, 64, 112, 112), with `64·3·7·7 + 64 = 9,472` parameters.

I can reconstruct the formula rather than recall it: after padding, the last valid kernel start position is `H + 2P - K`, positions step by S, and we add 1 for position zero.

**Q: Why are 3x3 kernels standard?**

Two stacked 3x3 layers have the same 5x5 receptive field as one 5x5 layer but use 18C² parameters instead of 25C², and include two nonlinearities instead of one. Three stacked 3x3s match a 7x7 with 27C² instead of 49C². Fewer parameters, more expressive power, and 3x3 is heavily optimized in cuDNN. VGG established this and it has held.

**Q: What do 1x1 convolutions do?**

A 1x1 kernel has no spatial extent but spans all input channels, so at each position it computes a learned linear combination across channels — a fully connected layer over channels, shared across space.

Its main use is cheap channel reduction. A direct `Conv2d(256, 256, 3)` costs 590K parameters; the bottleneck `1x1 (256→64) → 3x3 (64→64) → 1x1 (64→256)` costs 69K, about 8.5x less. That is ResNet's bottleneck block, and why ResNet-50 is smaller than VGG-16 despite being deeper. It also adds nonlinearity cheaply, and matches channel counts so residual connections can add tensors.

**Q: Explain residual connections. Why do they work?**

ResNet's authors found that a 56-layer plain network had higher *training* error than a 20-layer one — an optimization failure, not overfitting, since the deeper network provably has the capacity (set the extra layers to identity).

A residual block computes `y = F(x) + x`, so it learns the residual rather than the whole mapping. If identity is optimal, driving `F → 0` is easy.

The gradient argument: `∂L/∂x = ∂L/∂y·(∂F/∂x + 1)`. The `+1` means the gradient reaches x undiminished no matter what happens inside F. Across many blocks the gradient becomes a sum over paths rather than a product of attenuating factors, so vanishing gradients stop limiting depth. This made 100+ layer networks trainable.

**Q: Equivariance vs invariance?**

Equivariance: `f(shift(x)) = shift(f(x))` — the output moves with the input. Invariance: `f(shift(x)) = f(x)` — the output does not change. Convolution is equivariant, from parameter sharing. Classification needs invariance, obtained from pooling (local), repeated downsampling, and global average pooling at the end (global). The design keeps position information where it is useful and discards it at the end where it is not.

**Q: Explain transfer learning and when you would fine-tune versus freeze.**

Early CNN layers learn edges and textures that are nearly task-independent, so they transfer to almost any vision problem; only later layers are specialized. Transfer learning keeps the general part and retrains the specialized part.

The decision follows two axes. Small data plus a similar domain: freeze the whole backbone and train only a new head. Small data plus a different domain: freeze early layers, train the last block and head. Large data: fine-tune everything at a small learning rate, using discriminative rates that increase toward the head.

Always train the new head first with the backbone frozen — a randomly initialized head produces large uninformative gradients that would corrupt good pretrained features. And always use the pretrained model's exact preprocessing; `weights.transforms()` in torchvision gives it to you.

---

### Judgment and system design

**Q: You have 1,000 labelled images for a 5-class problem. Design an approach.**

I would start by looking at the data and computing baselines — chance is 20% if balanced, and I want to know the class distribution and whether the images are visually similar to natural photographs.

The model: a pretrained backbone, either ImageNet-supervised ResNet-50 or a self-supervised DINOv2, with the backbone frozen and a new linear head. 1,000 images cannot support fine-tuning 25 million parameters. Heavy augmentation: RandomResizedCrop, flips if label-preserving, colour jitter, RandAugment.

Because a single validation split of 1,000 images is unreliable, I would use 5-fold cross-validation for model selection and report mean and standard deviation, not a single number.

If frozen-backbone performance plateaus, I would unfreeze the last block at a very small learning rate and watch validation carefully for catastrophic forgetting.

For evaluation I would report per-class precision and recall and a confusion matrix, since with 200 images per class aggregate accuracy hides a lot. And I would spend real time looking at the misclassified images — on small datasets, a meaningful fraction of errors are usually label noise.

**Q: Training loss decreases, validation loss increases. What do you do?**

That is textbook overfitting. But before treating it I would check the split for leakage or distribution mismatch, and confirm `model.eval()` is called during validation, because that can produce a misleading validation curve.

Then: augmentation, weight decay, early stopping on the best checkpoint, dropout, and more data or a pretrained backbone. I would also check where the divergence starts — if validation loss rises from epoch 1, the model may simply be too large for the data, or the learning rate too high.

One nuance: validation *loss* can rise while validation *accuracy* still improves, because the model becomes overconfident on examples it gets right. If accuracy is what matters, early stop on accuracy.

**Q: How would you debug a model that will not learn at all?**

Overfit a single batch. Take 32 examples and train on them repeatedly — a correct model must reach near-zero loss. If it cannot, the problem is structural, and I would check: that the optimizer received `model.parameters()`; that `zero_grad`, `backward`, `step` are present and in order; that labels align with inputs; that no softmax precedes `CrossEntropyLoss`; that `super().__init__()` is present so parameters are registered; and that nothing in `forward()` detaches the graph.

If the single batch overfits fine, the issue is data or hyperparameters, and I would check the initial loss against ln(K), print per-layer gradient magnitudes to look for vanishing or missing gradients, and sweep the learning rate.

**Q: Your model is 94% accurate offline but performs poorly in production.**

The most likely cause by a wide margin is a preprocessing mismatch — different resize, normalization constants, colour channel order, or image decoding. I would run identical images through both pipelines and diff the tensors.

Next: whether the serving code calls `model.eval()`. Then distribution shift, which I would test by sampling and labelling production inputs. Then leakage that inflated the offline number, testable by checking for near-duplicates across splits. Then a class-balance or cost mismatch, where 94% accuracy on a balanced set means little on a 99:1 production distribution — per-class precision and recall would show this.

I would also verify the checkpoint loaded completely, since `strict=False` can silently skip weights.

---

### Depth and current context

**Q: Have Transformers replaced CNNs in vision?**

Not replaced — joined. The picture depends on scale. ViTs lack convolution's locality and translation equivariance, so they must learn those from data: the original ViT underperformed ResNet on ImageNet-1k and beat everything after pretraining on 300 million images. That is a general principle — an inductive bias substitutes for data, and becomes a constraint once you have enough data to learn better ones.

ConvNeXt complicated the story further by showing that a ResNet modernized with ViT-era training recipes matches Swin Transformers at equal compute, suggesting much of the reported gap was methodology rather than attention.

Today: Transformer-based models generally lead at large scale with large pretraining; convolutional models remain excellent for limited data, tight compute, and edge deployment; hybrids are common. I would also say that the bigger shift has been self-supervised pretraining, not architecture — the first question on a new vision project should be which pretrained backbone, not which architecture.

**Q: Explain a recent paper you found interesting.**

*(No model answer — this one is on you. Pick something you have actually read, be able to state the problem it addressed, the core idea, the evidence, and one honest limitation. Interviewers can tell within thirty seconds whether you read the paper or the abstract.)*

---

### 24.1 Questions to ask them

Interviews go both ways, and good questions signal seniority:

- How do you decide what to work on, and how is success measured?
- What does your model development and deployment pipeline look like?
- How much of the work is modelling versus data quality and infrastructure?
- How do you handle model monitoring and distribution shift in production?
- What does the path from junior to senior look like here?

### 24.2 Answering when you do not know

You will be asked something you cannot answer. The strong response has three parts: say plainly that you do not know, reason out loud from what you do know, and say how you would find out.

"I have not worked with that, but it sounds related to X, which works by Y — so I would guess the same principle applies. I would check the original paper and run a small experiment comparing it against a baseline."

That answer is often scored higher than a confident wrong one, because it demonstrates exactly the behaviour they need from you on the job.

## 25. Frequently Asked Questions

These are the questions students actually ask while working through this material, including the misconceptions that come up most often. Several answers deliberately contradict things you may have read elsewhere.

---

### Conceptual worries

**Do I need to memorize the backpropagation equations?**

You need to be able to *derive* them, which is different and more durable. If you understand that `δ[l] = ∂L/∂z[l]` and that the chain rule sums over all paths a variable influences, the four equations follow in a few lines each. Memorized equations evaporate under interview pressure; a derivation you have done three times does not.

A fair test: cover section 6.4 and reproduce all four equations with brief justifications. If you can, you are done with that section.

**Backprop makes sense line by line but I have no intuition for it. Is that normal?**

Yes, and it usually resolves in a specific way: work through the numeric example in section 6.3 by hand, then implement section 6.9 from scratch. Intuition for backprop comes from arithmetic, not from explanation. Most people need to compute it manually at least once.

The single sentence to hold onto: *the gradient of a weight is the error signal at its output times the activation at its input.* Everything else is bookkeeping about how the error signal gets there.

**Is deep learning just curve fitting?**

In a narrow technical sense, yes — it is function approximation by gradient descent. But that framing undersells what is happening: the network learns its own *representation* of the input, and the hierarchy of features it builds is the reason it generalizes rather than memorizing. A polynomial curve fit does not discover edge detectors.

The honest version: it is curve fitting in an extremely high-dimensional space where the parameterization and the optimizer together impose a strong implicit preference for simple solutions. Why that preference generalizes as well as it does is genuinely not fully understood (section 12.1).

**Why does any of this work? We are minimizing a non-convex loss with no guarantees.**

This discomfort is well-founded and you should keep it. What we know empirically: in high dimensions almost all critical points are saddles rather than bad local minima (section 7.5); different random seeds reach solutions of very similar quality; and over-parameterization appears to help rather than hurt. What we lack is a complete theory explaining why. Deep learning is currently an empirical science with partial theory, and anyone who tells you otherwise is overselling.

**How much of this will still be true in five years?**

Sections 1 through 13 — backpropagation, losses, gradient descent, initialization, optimizers, normalization, regularization — are the stable core. They already apply unchanged to Transformers, diffusion models, and whatever comes next. Architectural fashion changes faster; the material in sections 14 through 21 will age more, though convolution itself is not going anywhere.

---

### Design decisions

**How do I choose the number of layers and units?**

Start with something known to work for your problem type rather than reasoning from first principles. For tabular data, 2 hidden layers of 128–512 units. For images, a pretrained backbone. For a small image dataset from scratch, something like the SmallCNN in section 16.5.

Then follow the procedure in section 22.1: get the model to overfit your training set first, which proves you have enough capacity, then add regularization. Tuning architecture before you know whether capacity is the constraint is guessing.

One genuinely useful result: EfficientNet's compound scaling (Tan & Le, 2019) found that depth, width, and input resolution should be scaled *together* by a fixed ratio, not one at a time.

**Should I use ReLU or GELU or something newer?**

For a first CNN or MLP, ReLU. The differences between ReLU, GELU, SiLU, and Leaky ReLU are typically a few tenths of a percent, far smaller than the effect of your learning rate, augmentation, or whether you used pretrained weights. Use GELU in Transformer blocks because that is what those architectures were tuned with. Do not spend time here.

**How large should my batch size be?**

The largest that fits in memory, then adjust the learning rate. The linear scaling rule (section 7.6) says multiply the learning rate by the same factor you multiplied the batch size, with a few epochs of warmup.

Caveat: very small batches (under about 8) destabilize BatchNorm, and very large batches sometimes generalize slightly worse without careful tuning. 32 to 256 is the comfortable range for most work.

**Do I need a GPU?**

Not to learn this material. Everything in sections 1 through 17 runs on a laptop CPU in minutes. The CIFAR-10 project in section 19 is the first thing that really wants a GPU — roughly 20 minutes on a free Colab T4 versus several hours on CPU.

Google Colab's free tier is sufficient for this entire tutorial. Kaggle Notebooks offers a similar allocation with longer sessions.

**PyTorch or something else?**

PyTorch dominates research and is increasingly standard in industry, which is why this tutorial uses it. JAX is common in some research groups, especially at scale. TensorFlow and Keras remain in many production systems.

The important point: the concepts transfer completely. Someone fluent in PyTorch picks up JAX or TensorFlow in a week, because `zero_grad`/`backward`/`step` and its equivalents are the same ideas wearing different names.

---

### Misconceptions worth correcting

**"More layers always means better performance."**

False, and section 18.5 is the counterexample: a 56-layer plain network had worse *training* error than a 20-layer one. Depth beyond a point creates optimization problems that need residual connections and normalization to overcome. Depth also costs compute and latency. Add depth deliberately, not reflexively.

**"Neural networks work like the brain."**

Only at the crudest level of metaphor. Real neurons communicate in timed spikes, not continuous real values; dendrites perform their own nonlinear computation; there is no known biological mechanism resembling backpropagation; and cortical circuitry has structure no standard network mimics. The analogy was historically motivating and is now mostly misleading. Judge networks by their mathematics.

**"The universal approximation theorem means one hidden layer is enough."**

The theorem says a wide enough single-hidden-layer network can approximate any continuous function on a compact set to any accuracy. It says nothing about *how wide* (possibly exponentially), whether gradient descent can *find* those weights, or whether the result *generalizes*. It establishes only that expressive power is not the bottleneck — which is useful precisely because it redirects your attention to optimization and data (section 4.5).

**"BatchNorm works by reducing internal covariate shift."**

That was the original paper's claim and it is now widely doubted. Santurkar et al. (2018) deliberately injected noise after BatchNorm to *increase* covariate shift and found it still helped. The better-supported explanation is loss-landscape smoothing. Saying the original line confidently in an interview signals you read the abstract and stopped.

**"You need millions of images to train a vision model."**

To train from scratch, largely true. With transfer learning, a few hundred images per class is often plenty (section 20). This is the single biggest practical misconception among students, and it stops people attempting projects that would actually work.

**"Accuracy is the metric."**

Only for balanced problems where all errors cost the same. On a 99:1 imbalanced dataset, predicting the majority class always gives 99% accuracy and is useless. Report per-class precision and recall, a confusion matrix, and a metric matched to the actual cost of each error type. Always compute the trivial baseline so you know what your number means.

**"If the loss is going down, training is working."**

The most dangerous one. A model with badly misaligned labels, a softmax applied twice, or a leaking train/test split all produce loss curves that go down. Deep learning fails silently. This is why section 22 exists and why the overfit-one-batch test is worth 30 seconds before every long run.

---

### Debugging questions

**My loss is `nan`. What now?**

Check in this order: is the learning rate too high (lower it 10x)? Is there a manual `log` or division that could hit zero? Did you apply softmax before `CrossEntropyLoss`? Does your data contain `nan` or `inf`? If using mixed precision, is the `GradScaler` set up correctly?

To find the exact batch, print the loss every step and check the inputs of the batch immediately before it went bad. `torch.autograd.set_detect_anomaly(True)` will point at the offending operation, at the cost of speed.

**My validation accuracy is better than my training accuracy. Is something wrong?**

Usually not. Dropout, label smoothing, and data augmentation all penalize the training metric while leaving validation clean, so validation can legitimately look better early in training. Worry only if the gap is large and persistent, which can indicate that your validation split is easier than your training set, or too small to be meaningful.

**My model gets 99% validation accuracy on the first epoch and I do not believe it.**

You are right not to. This is almost always **data leakage**. Check whether the label is derivable from the input (an ID, a filename pattern, a preprocessing artifact that differs by class), whether validation examples also appear in training, and whether near-duplicates span the split. A quick diagnostic: train on shuffled labels — if accuracy does not collapse to chance, information is leaking.

**Training is extremely slow. What do I look at?**

Profile before optimizing. Common causes in rough order of frequency: `num_workers=0` so data loading blocks the GPU; the model is on CPU because `.to(device)` was missed; batch size too small to use the GPU efficiently; excessive CPU-side augmentation; calling `.item()` or `.cpu()` inside the inner loop, which forces a synchronization every step; and not using mixed precision, which is roughly a 2x win on modern GPUs for free.

**My results change every run. How do I make them reproducible?**

```python
import torch, numpy as np, random
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False        # note: this costs speed
```

Even then, some GPU operations are non-deterministic. More importantly, **run-to-run variance is information, not noise to be eliminated.** If two architectures differ by 0.2% and your seed-to-seed variance is 0.3%, you have not demonstrated anything. Report mean and standard deviation over at least three seeds before claiming an improvement.

---

### Study and career questions

**How long should this take?**

35 to 50 hours for genuine mastery, spread over several weeks rather than compressed. Section 6 alone deserves 4 to 6 hours. The projects in sections 13 and 19 deserve a full session each, including deliberately breaking them.

Spaced practice beats cramming substantially for this kind of material. Do a section, leave it a few days, then attempt its exercises cold.

**Do I need to read the original papers?**

For a first pass, no — this tutorial covers what you need. Once the material is solid, read four: AlexNet (2012), VGG (2014), ResNet (2015), and the ViT paper (2020). Each is readable with the background you now have, and reading a paper you already understand conceptually is the best way to learn *how to read papers*, which is the durable skill.

**What should I learn next?**

In rough order of value for most people: sequence models and attention (which leads directly to Transformers), then either a vision specialization (detection, segmentation) or an NLP one, then generative models (diffusion, VAEs), then deployment and MLOps. Section 26 has specific resources.

One underrated option: get good at the data side. Data quality, labelling, and evaluation design determine project outcomes far more often than model choice, and far fewer people are strong at it.

**Will AI replace the need to learn this?**

Models write competent PyTorch, and that changes what the job is rather than whether it exists. What does not automate: deciding what problem to solve, judging whether an evaluation is honest, diagnosing why a training run is silently broken, and knowing when a result is too good to be true. All four require exactly the understanding this tutorial builds. Being able to generate code you cannot debug is a liability, not a shortcut.

**I finished the tutorial and I still do not feel like I understand it.**

That is the normal state, and it is not a sign you did something wrong. Understanding in this field arrives in layers: the third time you debug a vanishing-gradient problem, section 9 will click in a way reading it never did.

A concrete test of where you actually stand: can you take a dataset you have never seen, build a working classifier, and explain each choice you made? If yes, you understand this well enough to continue. The feeling of mastery lags the capability by a considerable margin, for everyone.

## 26. Glossary, Notation Reference, and Further Reading

### 26.1 Full notation reference

Promised in section 0. This is every symbol used in the tutorial.

| Symbol | Meaning | First used |
| --- | --- | --- |
| `x`, `y` | scalars | 2 |
| **x**, **y** | column vectors | 2 |
| **W**, **X** | matrices | 4 |
| `x⁽ⁱ⁾` | training example i (round brackets) | 4 |
| `W[l]` | weight matrix of layer l, shape (n\[l\], n\[l-1\]) | 4 |
| `b[l]` | bias vector of layer l, shape (n\[l\], 1) | 4 |
| `z[l]` | pre-activation of layer l | 4 |
| `a[l]` | activation of layer l; `a[0] = x` | 4 |
| `n[l]` | number of units in layer l | 4 |
| `L` (superscript) | index of the output layer | 4 |
| `g`, `g'` | activation function and its derivative | 3 |
| `σ` | sigmoid function | 2 |
| `ŷ` | model prediction | 5 |
| `L` | loss on one example | 5 |
| `J` | average loss over a batch or dataset | 5 |
| `δ[l]` | error signal, `∂L/∂z[l]` | 6 |
| `Δ[l]` | batched error signal, shape (m, n\[l\]) | 6.5 |
| `⊙` | elementwise (Hadamard) product | 6 |
| `η` | learning rate | 7 |
| `m` | examples in a batch | 5 |
| `N` | examples in the dataset | 7 |
| `β`, `β₁`, `β₂` | momentum / Adam decay rates | 10 |
| `λ` | regularization strength, or a Hessian eigenvalue | 12, 7.3 |
| `γ`, `β` | BatchNorm learned scale and shift | 11 |
| `p` | dropout probability | 12.4 |
| `K` | kernel size, or number of classes | 15, 5 |
| `S`, `P`, `D` | stride, padding, dilation | 15 |
| `C_in`, `C_out` | input and output channel counts | 15.6 |
| `H`, `W` | feature map height and width | 15 |
| `r_l`, `j_l` | receptive field size and jump at layer l | 16.3 |

**The bracket convention, once more:** square brackets index **layers**, round brackets index **examples**. `a[2](7)` is layer 2's activation for example 7.

### 26.2 Formulas worth knowing cold

```
Forward pass:     z[l] = W[l]a[l-1] + b[l];   a[l] = g(z[l])

Backprop:         δ[L] = ∇ₐL ⊙ g'(z[L])        (= ŷ - y for softmax+CE)
                  δ[l] = (W[l+1]ᵀ δ[l+1]) ⊙ g'(z[l])
                  ∂L/∂b[l] = δ[l]
                  ∂L/∂W[l] = δ[l] a[l-1]ᵀ

Conv output size: H_out = floor((H_in + 2P - K)/S) + 1
Conv parameters:  C_out · C_in · K · K + C_out
Conv MACs:        C_out · H_out · W_out · C_in · K · K
FC parameters:    n_out · n_in + n_out

Receptive field:  j_l = j_{l-1}·S_l
                  r_l = r_{l-1} + (K_l - 1)·j_{l-1}

He init:          Var(w) = 2/n_in           (ReLU)
Xavier init:      Var(w) = 2/(n_in + n_out) (tanh, sigmoid)

Initial CE loss:  ≈ ln(K) for K balanced classes
Stability bound:  η < 2/λ_max
Momentum gain:    1/(1-β)
```

### 26.3 Glossary

**Activation** — the output of a layer after its nonlinearity. Also used loosely for the nonlinearity itself.

**AdamW** — Adam with decoupled weight decay applied outside the adaptive update. The modern default optimizer (10.6).

**Autograd** — automatic differentiation; PyTorch's implementation of reverse-mode AD (13.2).

**Backbone** — the feature-extracting portion of a network, excluding the task-specific head (16.2).

**Backpropagation** — reverse-mode automatic differentiation applied to a network. Computes gradients; does not update parameters (6).

**BatchNorm** — normalizes activations per channel using batch statistics in training and running estimates at inference (11).

**Bottleneck block** — a residual block using 1x1 convolutions to reduce then restore channel count around a 3x3 convolution (18.5).

**Catastrophic forgetting** — destruction of pretrained features by fine-tuning at too high a learning rate (20.2).

**Channel** — one feature map; the response of one learned filter across all spatial positions (15.6).

**Cross-entropy** — the negative log probability assigned to the correct class. Maximum likelihood under a categorical model (5.4).

**Dead ReLU** — a unit whose pre-activation is negative for every input, so it outputs zero and receives zero gradient forever (3.2).

**Degradation problem** — deeper plain networks achieving higher *training* error than shallower ones. An optimization failure, not overfitting. Solved by residual connections (18.5).

**Dilation** — gaps inserted between kernel elements, expanding the receptive field without adding parameters (15.9).

**Double descent** — test error falling, rising to a peak at the interpolation threshold, then falling again as capacity grows further (12.1).

**Dropout** — randomly zeroing units during training, with outputs scaled by 1/(1-p) so inference needs no adjustment (12.4).

**Epoch** — one full pass over the training set. Distinct from an *iteration*, which is one parameter update (7.2).

**Equivariance** — `f(shift(x)) = shift(f(x))`. Convolution's property (14.4).

**Feature map** — see *channel*.

**Fine-tuning** — updating pretrained weights on a new task, typically at a reduced learning rate (20.2).

**GELU** — Gaussian Error Linear Unit, `x·Φ(x)`. Standard in Transformers (3.2).

**Global average pooling** — averaging each channel's feature map to a single value, replacing large fully connected heads (16.4).

**Gradient checkpointing** — recomputing activations during the backward pass instead of storing them, trading compute for memory (6.7).

**Gradient clipping** — rescaling the gradient vector when its norm exceeds a threshold (9.7).

**He initialization** — `Var(w) = 2/n_in`, correct for ReLU networks (9.3).

**Ill-conditioning** — a large ratio between the largest and smallest Hessian eigenvalues, producing ravine-shaped loss surfaces (10.1).

**Inductive bias** — assumptions built into an architecture. A substitute for data: valuable when data is scarce, a constraint when it is abundant (14.6, 21.2).

**Invariance** — `f(shift(x)) = f(x)`. What a classifier ultimately needs (14.4).

**Label smoothing** — replacing hard one-hot targets with slightly softened ones, improving calibration (5.8).

**LayerNorm** — normalization over features within each example, independent of batch. Standard in Transformers (11.5).

**Logits** — raw output scores before softmax or sigmoid. Always pass these to PyTorch's loss functions (5.6).

**MAC** — multiply-accumulate operation; roughly two FLOPs (15.8).

**Mixup / CutMix** — augmentation methods that blend pairs of images and their labels (12.6).

**Momentum** — an exponentially decaying average of past gradients, multiplying the effective step in consistent directions by 1/(1-β) (10.2).

**One-cycle** — a schedule ramping the learning rate up to a peak then down well below the start (10.8).

**Overfitting** — low training error with high validation error. Distinct from the degradation problem, where *both* are high (12.8).

**Parameter sharing** — applying identical weights at every spatial position. The source of convolution's efficiency and equivariance (14.4).

**Pre-activation** — `z = Wa + b`, before the nonlinearity (4.1).

**Receptive field** — the region of the input image that can influence a given unit (16.3).

**Residual connection** — `y = F(x) + x`. Gives the gradient a path with derivative exactly 1 (18.5).

**Saddle point** — a critical point with both positive and negative Hessian eigenvalues. Dominates high-dimensional loss surfaces (7.5).

**Softmax** — converts K real scores into a probability distribution. Shift-invariant, and fused with cross-entropy in frameworks (5.4).

**Stride** — the step size of a convolution or pooling window (15.4).

**Transposed convolution** — the backward pass of convolution used as a forward operation, for upsampling (17.4).

**Universal approximation theorem** — a sufficiently wide one-hidden-layer network can approximate any continuous function on a compact set. Says nothing about width required, findability, or generalization (4.5).

**Vanishing gradient** — gradient magnitude shrinking multiplicatively across layers until early layers stop learning (9.1).

**Weight decay** — shrinking weights toward zero each step. Equivalent to L2 regularization in SGD, but not in Adam (10.6, 12.2).

**Xavier/Glorot initialization** — `Var(w) = 2/(n_in + n_out)`, suited to tanh and sigmoid (9.2).

### 26.4 Further reading

**Books**

| Book | Best for | Note |
| --- | --- | --- |
| Goodfellow, Bengio & Courville, *Deep Learning* | Theoretical foundations | Free online. Excellent on the math, dated on architectures |
| Zhang et al., *Dive into Deep Learning* (d2l.ai) | Code-first learning | Free, interactive, actively maintained, PyTorch/JAX versions |
| Howard & Gugger, *Deep Learning for Coders with fastai and PyTorch* | Getting results quickly | Strongly practical; excellent on transfer learning |
| Prince, *Understanding Deep Learning* | Modern theory with good figures | Free online; covers Transformers and diffusion |
| Murphy, *Probabilistic Machine Learning* | Probabilistic grounding | Rigorous; use as a reference |

**Courses**

- **Stanford CS231n**, Convolutional Neural Networks for Visual Recognition — the canonical CNN course. Lecture notes are outstanding and free, and much of this tutorial's framing follows its lineage.
- **fast.ai Practical Deep Learning for Coders** — top-down, results-first. A good complement to this tutorial's bottom-up approach.
- **Stanford CS224n** — natural language processing with deep learning; the natural next step for Transformers.
- **Karpathy's "Neural Networks: Zero to Hero"** — builds autograd, then a language model, from scratch on video. If section 6 did not click, watch the micrograd episode.

**Papers, in reading order**

Read these four first; each is accessible with your current background.

1. Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep CNNs* — AlexNet
2. Simonyan & Zisserman (2014), *Very Deep Convolutional Networks* — VGG, and the 3x3 argument
3. He et al. (2015), *Deep Residual Learning for Image Recognition* — ResNet
4. Dosovitskiy et al. (2020), *An Image is Worth 16x16 Words* — ViT

Then, by topic:

- **Optimization:** Kingma & Ba (2015) Adam; Loshchilov & Hutter (2019) AdamW; Smith (2018) one-cycle
- **Normalization:** Ioffe & Szegedy (2015) BatchNorm; Santurkar et al. (2018) on why it actually works; Wu & He (2018) GroupNorm
- **Initialization:** Glorot & Bengio (2010); He et al. (2015)
- **Regularization:** Srivastava et al. (2014) Dropout; Zhang et al. (2018) Mixup
- **Generalization theory:** Zhang et al. (2017) *Understanding deep learning requires rethinking generalization*; Belkin et al. (2019) double descent
- **Modern vision:** Liu et al. (2022) ConvNeXt; Oquab et al. (2023) DINOv2; Radford et al. (2021) CLIP

**Tools and references**

- PyTorch documentation and tutorials at pytorch.org — genuinely good, unusually so
- `timm` (PyTorch Image Models) — the standard library for vision backbones
- Weights & Biases or TensorBoard — experiment tracking; start using one now, not later
- Papers with Code — leaderboards linked to implementations
- Distill.pub — archived but its visual explanations of convolution arithmetic and feature visualization remain among the best available

**Datasets to practise on, in increasing difficulty**

MNIST (too easy, use only for pipeline testing) → Fashion-MNIST → CIFAR-10 → CIFAR-100 → Tiny ImageNet → Oxford Pets or Flowers-102 (good transfer learning targets) → ImageNet (needs real compute).

### 26.5 A closing note

You now know how a neural network computes its output, how it computes its gradients, why deep ones were once untrainable and what changed, why convolution suits images, how to build and debug a real model, and how to answer for all of it.

The most valuable habit going forward is the one from section 22: **be suspicious of results that look good.** Deep learning's characteristic failure is a model that appears to work. The engineers who are trusted with real systems are the ones who check the baseline, inspect the confusion matrix, look at the misclassified examples, and verify that the preprocessing at inference matches training. That discipline outlasts every architecture in this document.
