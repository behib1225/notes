# DNN Session 1 — What to say on every slide

**Companion to:** `DNN_Session1_150min_v5_3_TEACHING_CORRECTED.pptx` and the corrected instructor guide  
**Audience:** undergraduate engineering students with introductory linear algebra, calculus and Python  
**Format:** Suggested **spoken wording**, followed by optional teaching actions. This is a teaching script, not a transcript to read mechanically.  
**Timing:** Same 150-minute session. Slides 30–34 now receive 12 minutes, with time recovered from slide 29 and the surrounding transition; no changes to break or ending. Slides **55–72 are appendices**, not additional material to deliver within the scheduled 150 minutes.

## How to use these notes

Read the **Say** paragraphs aloud in your own voice, pause at questions, and work through the figures or equations at the indicated points. A proposed statement of **what an experiment shows** refers specifically to the numbers reported in the supplied slides; the experiment scripts were not rerun for this document. Mathematical and API statements are distinguished from empirical claims. Timing assumes brief discussion, boardwork, voting and student responses rather than nonstop reading. If time is short, retain the core idea and its teaching action. **Do not simply speed through the backpropagation derivation.** Use the separate live-presentation cue sheet for prompts rather than reading this long script verbatim.

### Important corrections in this edition

- **Momentum is not a universal 10× learning-rate multiplier.** A constant-gradient velocity limit is not a stability bound. After an unstable optimizer change, test a smaller learning rate and inspect the curves; do not impose a fixed ratio.
- **Quiz slides 25, 47 and 53 and answer slides 56–57 are revised in the companion PowerPoint.** The question on slide 25 now identifies the actual code error; slide 53 explicitly checks evaluation validity and assumes a relevant pretrained model exists.
- **Reveal discipline:** Before opening slide 30, put its parameters on the board and let students calculate. Before opening slide 43, solicit predictions based on slide 42. The PowerPoint contains visible numerical answers, not click-to-reveal animations.
- **Empirical caution:** The 0.57 to 0.53 loss change on slide 52 was observed in three supplied runs. This has not been independently reproduced here and is not a significance test.

---

## Part 1 — Foundations (slides 1–20, 0:00–0:43)

### Slide 1 — Deep neural networks: from one neuron to networks that train | 0:00–0:02

**Say:**
> “Today we will build a deep neural network from first principles. We will begin with one neuron, discover why one neuron has limited expressive power, and see how multiple neurons produce useful nonlinear representations. Then we will answer the learning question: how do the network's parameters change when its predictions are wrong? Finally, we will address a practical question: why do some deep networks train well while others fail or overfit? I will use **teal** for information moving forward through a network and **orange** for derivatives moving backward. By the end, a neural network should feel like a sequence of understandable mathematical operations rather than a black box.”

**Do:** Point to the three parts. State that code implementation is covered in the subsequent PyTorch session; this session supplies the reasoning behind it.

### Slide 2 — By the end of today you can… | 0:02–0:03

**Say:**
> “These are things I want you to be able to **do**, not just definitions to memorize. You should be able to prove why XOR defeats a single threshold neuron, compute the tensor shapes and parameter count of an MLP, explain why classification usually uses cross-entropy, and calculate a simple backward pass. You should also be able to look at a training curve and propose an evidence-based next step. I will check these skills with short questions during the lecture. Some of you may already know the software syntax; today the important skill is to explain **why** the syntax is correct.”

**Do:** Ask for a show of hands on familiarity with matrix multiplication, chain rule and logistic regression; do not spend lecture time reteaching prerequisites unless many students need it.

### Slide 3 — Today's plan, and one example we will follow all session | 0:03–0:05

**Say:**
> “We will not learn a completely different example every ten minutes. Our recurring model is an MLP for **Fashion-MNIST**: a grayscale image has 28 by 28 pixels, so after flattening it has 784 inputs. Two hidden layers have 128 and 64 units, and the output has 10 class scores. We will first trace the 784–128–64–10 network, then define its loss and training procedure, and finally inspect its training and validation curves. The plots tagged ‘experiment’ come from the scripts supplied with the slides. Those are results of particular runs—not claims that every run will look identical. There is a ten-minute break after Part 2.”

**Do:** Write `784 → 128 → 64 → 10` on the board and keep referring to it.

### Slide 4 — Foundations: what a network computes | 0:05

**Say:**
> “Before asking how a network learns, we need to understand what it can represent. Imagine a classifier that can draw only one straight separating line. What kinds of problems would be easy for it, and what kinds would be impossible? We will discover that the problem is not simply the number of inputs or the number of training iterations. It is the **representation** the classifier sees. Our first task is to identify the shape of a single neuron's decision boundary; then we will build a representation in which a formerly impossible problem becomes easy.”

**Do:** Wait for one student to suggest that some data cannot be separated by a straight line.

### Slide 5 — One neuron: a weighted sum and a threshold | 0:05–0:08

**Say:**
> “A threshold neuron computes one score, **z equals w transpose x plus b**, and returns one if the score is positive and zero otherwise. The weights tell us which directions increase the score. In two input dimensions, setting z to zero gives a straight line; in higher dimensions, a hyperplane. The weight vector points perpendicular to that boundary, and the bias moves the boundary parallel to itself when the weights stay fixed. Why? Because at any two points on the boundary, the change in the dot product with w is zero. If w is nonzero, the signed distance to the boundary is z divided by the length of w. Notice an important distinction: the **step activation is nonlinear**, but the boundary of this one threshold neuron is still straight.”

**Do:** Draw `2x₁ − x₂ − 3 = 0`, its normal vector `(2, −1)`, and a parallel line produced by changing only the bias. Mention that signed distance requires `w ≠ 0`.

### Slide 6 — Linear models can only draw straight boundaries | 0:08–0:10

**Say:**
> “Look at the two interlocking moons. Logistic regression does not predict directly from a straight line: it first computes a linear score and then passes it through a sigmoid. So why is its **classification boundary** straight? Because sigmoid crosses probability one half exactly when its input crosses zero. Applying a monotone sigmoid changes how we interpret the score, but it does not bend the set of points where the score is zero. On the particular training data shown, logistic regression reaches about 86 percent accuracy, while a small MLP fits the training examples perfectly. That is a demonstration of expressive capacity, **not** evidence that this MLP will necessarily generalize better. Linear classifiers remain excellent baselines; we need more flexibility only when the problem requires it.”

**Do:** Point to misclassified dots on the linear plot. Contrast **training accuracy** with performance on unseen data.

### Slide 7 — XOR: no straight line separates the classes | 0:10–0:13

**Say:**
> “XOR outputs one when exactly one of its two binary inputs is one. Try drawing a single straight line that puts both positive examples on one side and both negative examples on the other. You cannot. Let us prove that rather than trusting the picture. If the point `(0,0)` has class zero, then `b ≤ 0`. The positive points `(1,0)` and `(0,1)` require `w₁+b > 0` and `w₂+b > 0`. Add those inequalities: `w₁+w₂+2b > 0`. Since `b ≤ 0`, we conclude `w₁+w₂+b > 0`. But `(1,1)` is supposed to have class zero, requiring the opposite inequality. The requirements contradict each other. No choice of weights fixes that; additional training cannot fix a representational impossibility.”

**Do:** Give students 20 seconds to draw a line before writing the four inequalities.

### Slide 8 — A better representation makes XOR easy | 0:13–0:15

**Say:**
> “Instead of changing the labels, change how we describe the inputs. Add a third feature: the product `x₁x₂`. The original XOR truth table remains the same, but in the new three-dimensional feature space the four points can be separated by a plane. In fact, the expression `x₁ + x₂ − 2x₁x₂` gives the exact XOR output on binary inputs. This is a powerful idea: a task that requires a complicated decision boundary in one space may require only a simple separator after a useful transformation. Traditional feature engineering supplies that transformation by hand. Neural networks offer a way to **learn** the transformation from data.”

**Do:** Substitute `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` into the expression with the class.

### Slide 9 — The neural-network idea: learn the features too | 0:15–0:17

**Say:**
> “Imagine classifying a photograph from its raw pixels. We could try inventing a feature for every edge, texture, arrangement and object part, but manual engineering quickly becomes difficult. Instead, let the model learn an intermediate representation: `h = g(Wx + c)`. A later layer uses `h` to make the prediction. Both the feature-producing weights and the final classifier weights are adjusted using the **same task loss**. Early layers may learn simple patterns; later layers can combine them into more useful representations. This compositional feature learning is a central motivation for deep networks. It does not guarantee that every learned feature will be reusable, but transfer learning often benefits from this structure.”

**Do:** Draw `pixels → learned features → class scores`; ask which part would change when training for a different classification task.

### Slide 10 — Smooth activations give gradients: the sigmoid neuron | 0:17–0:20

**Say:**
> “A hard step function is fine for making a yes-or-no prediction, but it is unsuitable for ordinary backpropagation: its derivative is zero almost everywhere and undefined at the threshold. Gradient-based learning needs information about **how** an output changes when a parameter changes. Sigmoid gives a smooth alternative: `σ(z)=1/(1+e^(−z))`. Logistic regression can be derived by assuming the **log odds** are linear in the inputs. Sigmoid has derivative `σ(z)(1−σ(z))`, which is largest at `z=0` and never exceeds one quarter. That small derivative will become a problem when we multiply derivatives across many layers. One more useful point: scaling both w and b by the same positive number keeps the 0.5 decision boundary but changes how sharply probability changes near it.”

**Do:** Point out the flat sigmoid tails. Distinguish perceptron learning from differentiation through a step: a perceptron can be trained by a special update rule even though the step derivative is zero.

### Slide 11 — Without a nonlinearity, depth collapses to one layer | 0:20–0:22

**Say:**
> “Suppose I stack three fully connected layers but remove the hidden activation functions. The first layer computes `W₁x+b₁`; the second multiplies by another matrix and adds another bias; and so on. Substitute the first expression into the second and collect terms. Everything becomes **one matrix times x plus one bias**. Ten such layers are still an affine map, so extra depth does not let them solve XOR. The hidden nonlinearities prevent this collapse. Note that the output sigmoid alone does not help: it may turn a score into a probability, but it does not change the linear classification boundary.”

**Do:** Expand `W₂(W₁x+b₁)+b₂ = (W₂W₁)x+(W₂b₁+b₂)` on the board.

### Slide 12 — Quick check 1 | 0:22–0:24

**Say:**
> “The network here has five layers and a sigmoid at the end, but every hidden activation is the identity. Do not answer based only on the word ‘five.’ Ask what mathematical function the hidden stack computes. Can the final sigmoid turn the classifier's separating hyperplane into a curved boundary? Vote for the best answer individually; if you disagree with a neighbour, explain the chain of reasoning and vote again.”

**After voting, say:**
> “The answer is **C**. The hidden layers collapse to a single affine score, and the sigmoid threshold is equivalent to thresholding that score. Depth alone is not enough.”

**Do:** Keep answers hidden until the first vote.

### Slide 13 — Sigmoid and tanh saturate; ReLU does not for z > 0 | 0:24–0:27

**Say:**
> “The plot on the right is especially important: it shows the derivatives that backpropagation will multiply. Sigmoid has a derivative no greater than 0.25, and it becomes tiny in either saturated tail. Ten sigmoid derivatives multiply to at most about one millionth, **before accounting for the weights**, so a deep stack can struggle to send learning signals backward. Tanh is centred at zero and has a larger slope near zero, but it too saturates. ReLU, defined as the larger of zero and z, has derivative one on the positive side and zero on the negative side. That makes positive activations friendly to gradient flow, although weight matrices can still shrink or amplify gradients. A unit that is negative for every training example can become effectively inactive—a ‘dying ReLU.’”

**Do:** Trace the derivative curves rather than only the activation curves. Explain that ReLU is not differentiable at zero; frameworks choose a subgradient convention there.

### Slide 14 — Two ReLU units solve XOR by remapping the space | 0:27–0:30

**Say:**
> “Now we can solve XOR using a tiny network. The first hidden unit computes `ReLU(x₁+x₂)`; the second computes `ReLU(x₁+x₂−1)`. The output is the first value minus twice the second. On binary inputs, evaluate each corner: zero for `(0,0)`, one for `(1,0)` and `(0,1)`, and zero for `(1,1)`. Look at the right-hand plot: the two positive inputs have been mapped to the **same hidden-space point**, which is fine because they share a label. The output layer needs only a straight separating line in that hidden space. The hidden layer did the difficult work by transforming the representation. These weights were constructed to make the idea visible; training need not discover exactly this decomposition.”

**Do:** Compute the four hidden feature pairs `[(0,0),(1,0),(1,0),(2,1)]` in a table; point out the line `h₁−2h₂=1/2`.

### Slide 15 — A ReLU network computes a patchwork of linear pieces | 0:30–0:32

**Say:**
> “ReLU switches between two simple rules: return zero or return the input. For a fixed combination of on-and-off ReLU units, the whole network is an **affine function** of its inputs. But as we move through the input space, some units switch state and the affine rule changes. That creates a patchwork of linear regions. The coloured regions in this figure represent different activation patterns, and the black decision boundary consists of straight segments that can follow a curved shape overall. The script detected 146 regions on this finite grid, not necessarily the complete mathematical count. More layers can create complex boundaries by combining many individually simple pieces.”

**Do:** Trace the black polyline through several differently coloured regions. If using TensorFlow Playground, treat it as an optional demonstration, not a requirement.

### Slide 16 — Universal approximation, stated honestly | 0:32–0:35

**Say:**
> “At first the small ReLU network is a poor approximation to the grey curve. Add more appropriately placed ReLU units and the teal piecewise-linear approximation improves. This illustrates a famous existence theorem: with sufficiently many hidden units and a suitable non-polynomial activation, a one-hidden-layer network can approximate **any continuous function on a compact domain** as closely as desired. But be careful with what this theorem does *not* say. It does not say the required network is small, that gradient descent will find the weights, that finite training data identify the true function, or that fitting examples guarantees generalization. Universal approximation is a statement about **expressive possibility**, not an end-to-end recipe for successful machine learning.”

**Do:** Show decreasing approximation error across the three constructed panels. Emphasize “constructed, not trained” and “compact domain.”

### Slide 17 — The forward pass repeats one rule per layer | 0:35–0:37

**Say:**
> “We are ready to write an MLP as one repeated rule. The input is `a[0]=x`. At layer l, first compute a **pre-activation**, `z[l]=W[l]a[l−1]+b[l]`; then apply the chosen activation, `a[l]=g[l](z[l])`. Continue until the output layer. When doing these calculations by hand, treat one example as a column vector. If layer l has `n[l]` units and the previous layer has `n[l−1]`, the weight matrix must have shape `n[l] by n[l−1]`. One row of weights belongs to one output neuron. This same shape rule will save you from many programming mistakes.”

**Do:** Ask students for the weight shape of a `4 → 3` layer: `(3,4)`.

### Slide 18 — Running example: a 784–128–64–10 MLP | 0:37–0:39

**Say:**
> “Return to Fashion-MNIST. A mini-batch contains 32 images, each with one channel and dimensions 28 by 28. The starting tensor shape is `(32,1,28,28)`. Flattening each image—not the whole batch—gives `(32,784)`. The next linear layer produces 128 features per image; the next produces 64; the last produces **10 scores**, one per class. These scores are called **logits**. They are not yet probabilities, and we intentionally leave softmax out of the model. Notice that the leading batch dimension stays 32 at every stage.”

**Do:** Cover the displayed shapes one at a time and have students predict the next. Clarify that the full dataset has 60,000 original training images; this deck holds out 5,000 for validation.

### Slide 19 — In PyTorch terms: a stack of layers that returns logits | 0:39–0:41

**Say:**
> “PyTorch expresses this model as a sequence: flatten, linear, ReLU, linear, ReLU, and one final linear layer. There is **no output softmax** because the loss function will handle the logits correctly. How many parameters are learned? In a fully connected layer there is one weight for every input–output connection and one bias per output unit. That is `n_out × n_in + n_out`. Our three layers have 100,480, 8,256 and 650 parameters, totalling **109,386**. About 92 percent are in the first layer because each of 128 neurons sees all 784 pixels. Finally, PyTorch stores the weight matrix as `(out_features,in_features)` and multiplies a batch from the right by its transpose.”

**Do:** Write `784×128+128`, `128×64+64`, `64×10+10`; point to `layer.weight.shape` in the subsequent check.

### Slide 20 — Quick check 2 | 0:41–0:43

**Say:**
> “`nn.Linear(784,128)` takes 784 inputs and produces 128 outputs. Which weight shape lets every output neuron have one weight for each input? Think in terms of **rows of neurons**, not the order in which numbers are written in the constructor. Vote now.”

**After voting, say:**
> “The answer is **B: `(128,784)`**. A batch of shape `(32,784)` multiplied by the transposed weights gives `(32,128)`. The first dimension counts output neurons.”

**Do:** Ask a student to multiply the shapes on the board.

---
## Part 2 — Learning (slides 21–38, 0:43–1:20)

### Slide 21 — How a network learns | 0:43

**Say:**
> “So far, the weights were given to us. In practice we want to learn them. Learning requires three connected ideas: a **loss** measuring what the model gets wrong, an **optimizer** deciding how to change the weights, and a way to compute the **gradient** of that loss with respect to every weight. The forward pass makes predictions. The loss turns prediction quality into a number. Backpropagation supplies gradients. The optimizer uses them to produce the next set of weights. Keep these as distinct operations in your mental picture.”

**Do:** Under the existing `784 → 128 → 64 → 10`, write `predictions → loss → gradients → parameter update`.

### Slide 22 — Most losses are negative log-likelihoods | 0:43–0:46

**Say:**
> “A loss function should reflect what kind of outcome we are predicting. If a real-valued target equals a prediction plus **Gaussian noise of fixed variance**, maximizing its likelihood is equivalent to minimizing squared error. If the target is one of two classes, a **Bernoulli** model leads to binary cross-entropy. If the target is one of several mutually exclusive classes, a **categorical** model leads to multiclass cross-entropy. So a loss is not merely a punishment someone invented; often it comes from the probability model we assume for the labels. In PyTorch, `MSELoss` is for ordinary squared-error regression, `BCEWithLogitsLoss` is for binary or multilabel classification, and `CrossEntropyLoss` is for mutually exclusive classes. The two classification losses expect **raw logits**.”

**Do:** Ask whether ‘number of shoes sold’ necessarily fits Gaussian noise; use counts as an example where another likelihood may be more appropriate. Say “many common losses,” rather than claiming every useful loss is a negative log-likelihood.

### Slide 23 — Softmax, and the first-loss sanity check | 0:46–0:48

**Say:**
> “Our model produces ten logits, not ten probabilities. Softmax converts logits into positive numbers that sum to one by exponentiating and normalizing them. If all ten logits were equal, each class would receive probability one tenth. Cross-entropy for a correct-class probability of one tenth is `−log(0.1)`, about **2.30**. That gives a useful initial sanity check: an untrained ten-class classifier whose logits are small and similar should start near that loss. In the deck's recorded runs, the first loss is a little higher, around 2.6 to 3.0, because the random logits are not identical. If the first loss is 10 or 15, investigate the scale of the logits, the inputs and the targets. This is a diagnostic clue, not a hard law.”

**Do:** Compute `−log(0.9)≈0.105` and `−log(0.01)≈4.61`. Ask why two predictions with identical accuracy can have different losses.

### Slide 24 — Cross-entropy keeps learning when MSE stalls | 0:48–0:50

**Say:**
> “Here is why the pairing of output activation and loss matters. Suppose the true binary label is one, but the sigmoid predicts only 0.0001. If we use **half the squared error**, the gradient with respect to the logit is prediction error times the sigmoid derivative. Because sigmoid is saturated, that product is roughly **minus 0.0001**: a tiny correction despite a very wrong prediction. With binary cross-entropy, the algebra simplifies. The sigmoid derivative cancels with the derivative introduced by the logarithm, giving `∂L/∂z = σ(z) − y`. That is approximately **minus one**, so the output receives a strong corrective signal. For softmax combined with multiclass cross-entropy, the analogous logit gradient is predicted probabilities minus target probabilities. This advantage is at the **output logits**; it does not guarantee that gradients remain healthy through all earlier hidden layers.”

**Do:** Write both derivatives side by side. State explicitly that the plotted squared-error formula uses `L = ½(ŷ−y)²`; PyTorch `MSELoss` omits the ½, which changes the numeric gradient by a factor of two.

### Slide 25 — Quick check 3 | 0:50–0:52

**Say:**
> “Here is the actual code pattern, not just a training-curve symptom: a ten-class model ends with `Softmax(dim=1)`, and its output is passed to `nn.CrossEntropyLoss()`. Which change fixes the API mismatch? Think about whether this loss expects probabilities or raw scores.”

**After voting, say:**
> “**B: remove the final softmax and pass raw logits to `CrossEntropyLoss`.** This conclusion follows from the code, unlike an isolated training-loss plateau, which could have many causes. The next slide shows how the extra softmax can create a loss floor of about 1.46 for ten classes—even when classification accuracy looks reasonable.”

**Do:** Vote before advancing to slide 26. Explain that a plateau near 1.5 is a *clue*, never proof, unless the code confirms it.

### Slide 26 — Pass raw logits to the loss: softmax twice fails silently | 0:52–0:55

**Say:**
> “Let us unpack that bug. `CrossEntropyLoss` treats its input as logits and applies the log-softmax operation internally, using a numerically stable calculation. If you feed it probabilities that have **already** passed through softmax, those values are restricted to the interval from zero to one. Even the most extreme probability vector cannot create an arbitrarily dominant true-class logit in this second softmax. For K classes, the smallest possible resulting loss per example is `log(K−1+e)−1`. With ten classes that is about **1.46**, not zero. The experiments shown here demonstrate how misleading this can be: accuracy may remain respectable, but the reported loss has an artificial floor and the model's own probability estimates can be badly calibrated. The reliable rule is simple: **return raw logits from the model; let the fused loss handle softmax and the logarithm**.”

**Do:** Point from the bugged curve to its loss floor. Say reported confidence and calibration figures belong to these particular runs; do not present them as universal outcomes.

### Slide 27 — Choose the activation by the layer's role | 0:55–0:57

**Say:**
> “An activation is not one choice we repeat everywhere. In a conventional MLP or CNN, ReLU is a useful hidden-layer default; alternatives like Leaky ReLU can help if many units become permanently inactive. Many Transformers use GELU or a gated feed-forward design such as SwiGLU. The classifier's final layer, however, should usually return **raw logits** for PyTorch's fused classification loss. Unbounded regression often uses an identity output. If the predicted quantity must be positive, we can choose a positive transformation such as Softplus—or parameterize its logarithm. These are architectural choices informed by what the output represents. A gated unit like SwiGLU is not merely another scalar activation; it combines multiple learned projections.”

**Do:** Ask students to distinguish “hidden-layer activation” from “output activation required by the task.” Clarify that raw logits are a training-interface convention for these fused PyTorch losses, not a ban on computing probabilities at inference.

### Slide 28 — Gradient descent: the sharpest direction caps η | 0:57–1:00

**Say:**
> “Once we have a loss, we need to change the parameters to reduce it. The gradient points in the direction of fastest local increase, so a small step in the **negative gradient direction** should decrease the loss. ‘Small’ matters. On a quadratic bowl, the gradient gets steeper in the high-curvature direction. If we take a large step, we overshoot, bounce across the valley, and may diverge. For the quadratic pictured, the largest curvature is 10, so fixed-step gradient descent converges only when `0 < η < 2/10`, or **η below 0.2**. A step size of 0.21 diverges on this example. Real neural-network losses are not globally convex quadratics, so this is an instructive **local stability model**, not a universal upper bound for every training run.”

**Do:** Derive `θ_new=(1−ηλ)θ` for `J(θ)=½λθ²` and the condition `|1−ηλ|<1`. Show overshoot on the figure.

### Slide 29 — Backpropagation: every gradient for ≈ 3× a forward pass | 1:00–1:01

**Say:**
> “Imagine a network with a hundred million parameters. If we change one weight at a time to estimate its derivative numerically, we need a huge number of extra forward evaluations. That is useful for checking a few derivatives, but not for training. Backpropagation uses the **chain rule**, moving backward through a computation graph while reusing intermediate results. Because one scalar loss depends on many parameters, reverse-mode differentiation is particularly efficient. The rule of thumb on this slide is that a forward pass plus backward pass can cost on the order of **three forward passes** in common dense networks. It is only a rough computational estimate; the real ratio depends on the operations, implementation and hardware. Remember: backpropagation **computes gradients**. SGD or Adam **updates parameters** using those gradients.”

**Do:** Draw a four-node chain and point out how one backward sweep reuses already computed derivatives rather than starting over for every weight.

### Slide 30 — Worked example: forward, then backward, by hand | 1:01–1:03

**Say:**
> “We will compute the forward pass of the smallest network in today's lecture. First copy the data **from my board**, not from the slide: `x=2`, `y=1`, `w₁=0.5`, `b₁=0.1`, `w₂=0.8`, `b₂=−0.3`, and `L=½(a₂−y)²`. Both neurons use sigmoid. Compute `z₁`, `a₁`, `z₂`, `a₂` and the loss in that order. You have one minute. What is the number entering the first sigmoid, and why?”

**After students calculate, show slide 30 and say:**
> “We get `z₁=1.1`, `a₁≈0.7503`, `z₂≈0.3002`, `a₂≈0.5745`, `L≈0.0905`. This is only a forward pass: the parameters have not changed. Squared error with a sigmoid is deliberately used to expose each derivative in our hand calculation; binary cross-entropy is generally a better classification training objective.”

**Do:** Keep slide 29 displayed while writing the parameters on the board; *then* advance to slide 30 with its answers already visible. Do not promise an animation that the deck does not contain.

### Slide 31 — Worked example: backward error signals | 1:03–1:06

**Say:**
> “Backpropagation starts at the final loss and repeatedly applies the chain rule. Write `δ₂ = ∂L/∂z₂`. First, `∂L/∂a₂ = a₂−y ≈ −0.4255`. Multiply by `σ′(z₂)=a₂(1−a₂)` and obtain `δ₂ ≈ −0.1040`. To move through the connection to the hidden neuron, multiply by `w₂=0.8`: `∂L/∂a₁≈−0.0832`. Finally multiply by `σ′(z₁)=a₁(1−a₁)` to get `δ₁≈−0.0156`. Each product has a meaning: downstream effect times local sensitivity.”

**Do:** Reuse the slide-30 diagram. Colour the backward arrows orange and write the local multiplier above each edge. Ask students why `δ₁` can be much smaller than `δ₂`; emphasize that a single example illustrates a *possible mechanism*, not an inevitable fate of deep networks.

### Slide 32 — Worked example: parameter gradients | 1:06–1:08

**Say:**
> “We now know `δ₁` and `δ₂`, the derivatives of the loss with respect to each neuron’s pre-activation. What is the gradient of the output weight? Because `z₂ = w₂a₁+b₂`, the chain rule gives `∂L/∂w₂=δ₂a₁≈−0.0780`. Similarly, `∂L/∂w₁=δ₁x≈−0.0312`. The bias gradient is just δ because a bias enters the pre-activation with coefficient one. The reusable rule is **receiving-neuron error times incoming activation**.”

**Do:** Have students supply the four gradients: `w₁≈−0.0312`, `b₁≈−0.0156`, `w₂≈−0.0780`, `b₂≈−0.1040`. Draw one weight arrow and annotate `∂z/∂w = input`.

### Slide 33 — Quick check 4 | 1:08–1:10

**Say:**
> “Keep the original network but set the second weight, the connection from hidden activation to output, to zero. Now ask: does changing the first weight have *any effect* on the final prediction? Trace the computational path rather than doing arithmetic. If the pathway from hidden neuron to output has a zero coefficient, what happens to the first layer's loss gradient?”

**After voting, say:**
> “The answer is **B: exactly zero**. The chain rule includes a factor of `w₂`; when `w₂=0`, the derivative of the output with respect to the earlier hidden activation is zero. So the first weight receives no loss gradient on this example. Blocking a path blocks learning along that path.”

**Do:** Cross out the connection between the hidden and output neurons on the diagram.

### Slide 34 — The four equations of backpropagation | 1:10–1:13

**Say:**
> “We just calculated four gradients without memorizing formulas. Now give that procedure reusable matrix notation. Define `δ[l]=∂L/∂z[l]`. For an **elementwise** output activation, `δ[L] = (∂L/∂a[L]) ⊙ g′(z[L])`. To move one hidden layer backward, `δ[l] = (W[l+1]ᵀδ[l+1]) ⊙ g′(z[l])`. Why the transpose? Forward propagation distributes each input to the next layer; backward propagation must *sum contributions back* to each previous neuron. For biases, `∂L/∂b[l]=δ[l]`. For weights, every entry is `δ_i a_j`, so `∂L/∂W[l] = δ[l] a[l−1]ᵀ`, an outer product.”

**Then add:**
> “Check the shapes: if `W[l]` has `(n_l,n_{l−1})`, then `δ[l] a[l−1]ᵀ` has exactly that shape. For a batch mean, sum the per-example outer products and divide by batch size. **Softmax is not elementwise**; when combined with cross-entropy, derive its output result separately: `δ[L]=p−y` per example (before batch averaging).”

**Do:** Build **one row** of the weight-gradient matrix from the scalar example; draw how `Wᵀ` collects downstream contributions. Pause on shape matching; do not read four equations in one uninterrupted burst.

### Slide 35 — In PyTorch, autograd runs these equations for you | 1:13–1:14

**Say:**
> “PyTorch makes these derivatives automatic, not mysterious. A tensor with `requires_grad=True` participates in a computation graph. Calling `loss.backward()` traverses that graph backward and computes the relevant derivatives. The resulting gradients agree with our hand-calculated values, approximately minus 0.0312 and minus 0.0780 for the two weights. There are three practical consequences. First, gradients **accumulate** in `.grad`, so clear them before a normal training step. Second, backward needs information from the forward pass, so training uses more memory than inference. Third, `model.eval()` and `torch.no_grad()` do **different jobs**: evaluation mode changes the behaviour of layers such as dropout; no-grad disables gradient recording. Use both for ordinary validation.”

**Do:** Highlight `loss.backward()` and `.grad`; explain that the slide is PyTorch-flavoured pseudocode, not a ready-to-run cell as printed.

### Slide 36 — Feeding the network: splits, standardization, mini-batches | 1:14–1:15

**Say:**
> “Training is not just an optimizer loop; the data procedure matters. Use training examples to fit weights, validation examples to select learning rates and checkpoints, and reserve the test set for final evaluation. **Estimate normalization statistics on training data only**; carrying information from validation or test into preprocessing produces leakage. In our example, there are 55,000 training images, 5,000 validation images and 10,000 test images. An epoch is one pass through the training set. With batches of 128, that is `ceil(55,000/128)=430` updates per epoch. A mini-batch estimates the full gradient with some sampling noise but makes efficient use of modern hardware. Under usual sampling assumptions, its noise standard deviation falls roughly as one over the square root of batch size.”

**Do:** Distinguish **epoch**, **mini-batch**, **optimizer step**, and **validation check**. Explain that no validation example should be used to update model weights.

### Slide 37 — The training loop: five steps, every mini-batch | 1:15–1:17

**Say:**
> “The five-step training loop is the procedure you will actually implement. First, clear the old gradients. Second, run the input batch forward through the model. Third, compute the scalar loss from the logits and labels. Fourth, call backward to populate gradients. Fifth, let the optimizer update the parameters. Repeat that sequence **once per training mini-batch**. At the start of the epoch, set the model to training mode. For validation, switch to evaluation mode and disable gradient tracking. Notice that the validation data are used to **measure** performance, not to call `optimizer.step()`. If you understand what each of these calls does, the PyTorch syntax will be much easier to debug.”

**Do:** Have students say which line *computes* gradients (`backward`) and which line *uses* them (`step`).

### Slide 38 — Running example: training curves after 20 epochs | 1:17–1:20

**Say:**
> “These plots finally bring everything together. In the recorded Fashion-MNIST run, training loss continues to decrease, reaching about 0.14. The validation loss initially decreases but reaches its best value around **epoch five** and then begins to rise, while validation accuracy stays near 90 percent. Why can accuracy remain stable while cross-entropy gets worse? Accuracy only asks whether the top class is correct; cross-entropy also cares **how confident** the model is. Becoming more confident on a few wrong predictions can raise loss without changing the number of correct predictions. That is why we save the checkpoint with the best **validation metric**, not automatically the final epoch. Across the recorded seeds, the best epoch changes, so do not memorize ‘epoch five’ as a general rule.”

**Do:** Point first to training loss, then validation loss, then validation accuracy. Ask: “Would a longer training run necessarily give a better final model?”

### Slide 39 — Ten-minute break | 1:20–1:30

**Say:**
> “Take ten minutes. Here is the question I want you to think about: suppose a 30-layer plain network obtains **higher training error** than a similar 10-layer network. Is that ordinary overfitting? Overfitting usually means performance gets worse on unseen examples while training performance becomes very good. So if the **training** fit gets worse with depth, what else could be going wrong? After the break we will examine gradient flow, initialization and residual connections.”

**Do:** Resume at 1:30. Do not use the break to rush ahead with additional lecture content.

---
## Part 3 — Making deep networks train and generalize (slides 40–54, 1:30–2:30)

### Slide 40 — Making deep networks train and generalize | 1:30

**Say:**
> “Let us answer the break question first. If the **training** error is worse for a deeper network, that is not ordinary overfitting. It suggests an **optimization** problem: we cannot find a good solution even for the training data. Increasing depth makes both forward signals and backward gradients pass through many transformations. We therefore need techniques that keep their scales manageable, make optimization less sensitive to direction, and help us distinguish an optimization failure from a generalization failure. The roadmap is initialization, residual connections, optimizers, and finally regularization and diagnosis.”

**Do:** Contrast “cannot fit training data” with “can fit training data, cannot generalize.”

### Slide 41 — Depth multiplies: gradients vanish or explode | 1:30–1:34

**Say:**
> “Look back at the backpropagation equation: to move a gradient toward an earlier layer, we multiply by a transposed weight matrix and an activation derivative. Repeat that through 20 or 30 layers and we get a **product of many factors**. If their typical scale is below one, the gradient can shrink dramatically; if above one, it can explode. Think of repeatedly multiplying by 0.5: after 30 repetitions, the result is around one billionth of where it started. Repeatedly multiplying by 1.5 creates a huge value instead. In real networks these are **matrices**, not just scalar multipliers, so directions and singular values matter, but the analogy captures the central danger. To diagnose this, log gradient norms by layer. Initialization, residual connections and normalization can improve the situation; gradient clipping is especially useful for large gradients.”

**Do:** Calculate `0.5³⁰≈9.3×10⁻¹⁰`. Warn that an early small gradient is not by itself proof of failure; compare trends and learning behaviour.

### Slide 42 — Initialize to keep the pre-activation variance constant | 1:34–1:39

**Say:**
> “When we initialize weights, choosing numbers that are simply ‘small and random’ is not enough. We want the scale of signals to remain reasonably stable as they pass through many layers. For independent zero-mean weights, the variance of the next pre-activation is approximately the number of incoming connections times the weight variance times the **second moment** of the incoming activations. Why second moment? Because after ReLU the activations are no longer zero-centred. If a pre-activation is symmetric about zero, ReLU removes its negative half; its **squared output has half the original second moment**. He or Kaiming initialization compensates with weight variance about `2/n_in`. Xavier initialization instead uses `2/(n_in+n_out)` and is a common starting point for approximately symmetric activations such as tanh. All-zero **weights** are a mistake: identical hidden units then receive identical gradients and cannot specialize.”

**Do:** Derive `Var(z)≈n_in Var(w) E[a²]`; for ReLU use `E[ReLU(z)²]=½ E[z²]` when z is symmetric and zero-mean. Add that `Var(ReLU(z))` itself is **not** half `Var(z)`—this is the key subtlety. The table's suggestion of He for GELU is a useful heuristic, not the exact ReLU derivation.

### Slide 43 — Simulation: 20 ReLU layers, four initializations | 1:39–1:42

**Say:**
> “Before advancing from slide 42, predict which weight scale will preserve a reasonable signal through many ReLU layers, and which scales will shrink or explode. Now inspect the plotted results. Small weights produce rapidly shrinking activations; very large weights can create growth; Xavier is too small for this plain, repeated-ReLU setting; He initialization keeps the second-moment scale roughly controlled under its assumptions. Remember this figure is **one numerical simulation** of forward activations, not a proof that only He can work or that the network will train successfully.”

**Do:** Ask for predictions **while slide 42 is still visible**, then display slide 43. Point out that forward activation spread is not a direct measurement of backward gradients.

### Slide 44 — Quick check 5 | 1:42–1:44

**Say:**
> “A 30-layer network originally used tanh and Xavier initialization. We change all hidden activations to ReLU but keep exactly the same initialization. What do we expect for the scale of the forward signal? Remember: for equal layer widths, Xavier gives weight variance `1/n`, while the ReLU-based He formula gives `2/n`. Which way does the mismatch push the signal, and does that mismatch compound with depth? Vote and explain your reasoning to a neighbour.”

**After voting, say:**
> “The answer is **B** under the assumptions of the simple variance-propagation model. The mean-square signal is roughly halved per layer, corresponding to an RMS-scale factor around `1/√2` per layer. Exact **activation standard deviations** differ somewhat because ReLU outputs have a nonzero mean. The appropriate fix is to revisit initialization, for example using He for a plain ReLU stack.”

**Do:** Avoid presenting `1/√2` as an exact empirical standard-deviation multiplier in arbitrary architectures.

### Slide 45 — Residual connections: why deeper can still train | 1:44–1:49

**Say:**
> “The next advance is not merely better random weights. A **residual block** computes `x + F(x)`: the input passes directly to the output while the learned branch adds a correction. This makes a useful behaviour easy to represent. If an extra block is unnecessary, it can learn `F(x)≈0` and approximate the identity rather than having to construct identity through several nonlinear layers. The backward view is equally important. The block's Jacobian is `I + ∂F/∂x`, so the gradient has an **identity path** in addition to the learned path. This often makes very deep networks substantially easier to optimize. But do not say that residual connections eliminate vanishing or exploding gradients by theorem: products of these Jacobians can still be poorly conditioned. ResNets helped resolve the observed degradation of training performance in deeper plain networks, and residual connections are now standard in Transformer blocks too.”

**Do:** Draw a curved bypass around two operations and write `∂(x+F(x))/∂x = I + J_F`. Explain why the experimental degradation problem need not be caused solely by vanishing gradients.

### Slide 46 — Ravines defeat plain SGD; momentum and Adam help | 1:49–1:53

**Say:**
> “Consider minimizing a loss shaped like a long, narrow valley. Across the valley the slope changes rapidly; along the valley progress is slow. Plain SGD uses one learning rate for every direction, so a learning rate large enough to move quickly along the valley may bounce too much across it. **Momentum** keeps a running average of gradients: directions that consistently point the same way accumulate, while oscillating directions partly cancel. **Adam** also tracks the squared gradients and rescales updates coordinate by coordinate. The graph is a controlled, rotated two-dimensional quadratic example, with learning rates tuned separately for each method. Adam reaches the lowest loss here after 40 steps. That result describes this toy experiment, **not a universal ranking**. Real training outcomes depend on model, data, schedule and tuning.”

**Do:** Trace the three optimizer paths on the contours. Explain that Adam's coordinatewise scaling is not automatically the same as full Hessian curvature correction.

### Slide 47 — Quick check 6 | 1:53–1:55

**Say:**
> “Imagine switching from plain SGD to momentum with `β=0.9` while keeping `η=0.1`. The new run becomes unstable. Which *first controlled troubleshooting experiment* would you try? We should test a smaller learning rate, keeping other settings fixed, and compare the loss traces. Vote.”

**After voting, say:**
> “**A: try a smaller learning rate and inspect the results.** But do not memorize `η_new=η_old/10`. The fact that `v_t=βv_{t−1}+g` tends to `g/(1−β)` **when the gradient stays constant** does not tell us the stability limit when the gradient changes with the parameters. Indeed, for positive-definite quadratics and conventional heavy-ball momentum, stability allows `η < 2(1+β)/λ_max`, compared with `2/λ_max` for plain gradient descent. Real neural-network training is not that quadratic model either; the practical lesson is to **retune and measure**.”

**Do:** Present the 10× velocity observation only as a conditional arithmetic fact. For advanced students, write the 1-D quadratic recurrence `θ_{t+1}=(1+β−ηλ)θ_t−βθ_{t−1}`; do not suggest a universal 10× reduction.

### Slide 48 — Adam = momentum + per-parameter steps; use AdamW | 1:55–1:59

**Say:**
> “Adam maintains two running quantities for every parameter: a smoothed gradient, called the first moment, and a smoothed squared gradient, called the second moment. It corrects their initial bias toward zero and divides the first by a square-root scale derived from the second. This gives adaptive coordinatewise steps. **AdamW** changes how we apply weight decay. With plain SGD **without momentum**, adding an L2 penalty and applying multiplicative weight decay are equivalent after matching the coefficients. With momentum, their equivalence requires additional care. In Adam, adding `λw` to the gradient is **not equivalent** to shrinking the weights separately, because the adaptive denominator also acts on that penalty. AdamW decouples the decay from the gradient moments. In current PyTorch, `Adam` defaults to coupled decay when `weight_decay` is set, whereas `AdamW`, or `Adam(decoupled_weight_decay=True)`, uses decoupled decay. I would start with AdamW as a practical baseline for many new projects, but tune its learning rate and decay; SGD with momentum remains strong in many CNN recipes.”

**Do:** Show two arrows: `gradient → adaptive Adam update` and a separate `weight → multiplicative decay`. Say the stated learning rates and decay ranges are starting points, not validated optimum settings for this dataset.

### Slide 49 — Diagnose the gap first, then pick remedies | 1:59–2:03

**Say:**
> “Before changing the model, determine which failure you actually have. If **training and validation losses are both high**, the model may be underfitting or failing to optimize; investigate data, labels, capacity and learning rate. If training is excellent but validation is much worse, you have a generalization problem. If both are good, do not add complexity without a reason. My favourite debugging test is to train on a tiny sample—perhaps 100 examples—with augmentation and strong regularization disabled. A sufficiently capable correctly implemented model should often be able to nearly memorize it. If it cannot, look for a bug before searching for exotic optimizers. For overfitting, first save the best validation checkpoint, then consider more useful data, appropriate augmentation or transfer learning, and tune weight decay or dropout. The order is a **decision guide**, not a rigid universal ranking.”

**Do:** Ask students to classify the Fashion-MNIST curves from slide 38. Explain that a lower validation loss than training loss can occur when train-time dropout/augmentation is active.

### Slide 50 — Weight decay and dropout | 2:03–2:07

**Say:**
> “We now have two complementary forms of regularization. **Weight decay** discourages large weights by shrinking them during optimization; in decoupled weight decay the shrinkage is separate from the data-loss gradient. Smaller weights can limit how sensitively some networks respond to changes in input, but the effect depends on architecture and normalization, so ‘small weights always mean simple functions’ would be too strong. In typical implementations we apply decay to weight matrices while often excluding biases and normalization scale/shift parameters. **Dropout** works differently: during training it randomly sets selected activations to zero, then scales the survivors so the **expected** activation stays the same. At evaluation time dropout is disabled. It discourages reliance on fragile combinations of units, although the best dropout rate depends heavily on the task.”

**Do:** For `p=0.5`, show an activation of `2` being kept and scaled to `4`, or dropped to `0`; the expected value remains `2`. Clarify this equality concerns **expected activations**, not exact equality of model predictions, because later nonlinear operations matter.

### Slide 51 — Data augmentation: free data, if the label survives | 2:07–2:10

**Say:**
> “Suppose we have only a small number of labeled clothing images. An augmentation applies a random transformation that produces a new training view **without changing what the class means**. A mild horizontal flip might preserve whether something is a shoe or shirt; the same flip could reverse the meaning of text, and rotating a six could make it look like a nine. So augmentation is not automatically safe—its **label-preserving assumptions** must match the task. The code sketch uses a random crop and horizontal flip before conversion to a float image and normalization. The normalization constants should come from the **training images**. Apply random augmentation to the **training** set, not to validation or test measurements; those should use a fixed, appropriate evaluation pipeline. Stronger approaches such as Mixup and CutMix may even change the targets into soft probability labels.”

**Do:** Ask students to propose one valid and one invalid transformation for a digit-recognition dataset. Explain that a crop can cease being label-preserving if it removes the entire object.

### Slide 52 — Running example: 1,000 images overfit | 2:10–2:14

**Say:**
> “When this MLP is trained on 1,000 Fashion-MNIST examples, its training loss becomes very small but validation loss eventually rises: a classic gap between fitting training examples and performing on held-out examples. In the **three reported seeds**, adding dropout and weight decay changes the best validation cross-entropy mean from roughly 0.57 to 0.53. That is a result observed in these runs; with only three seeds we have not established a reliable general advantage. The accuracy difference is only around **0.4 percentage points** and is inconclusive. In these particular conditions, training with 55,000 rather than 1,000 images shows a much larger, roughly **8.4-point** validation-accuracy difference. Neither comparison proves that more data or any specific regularizer always wins.”

**Do:** Differentiate cross-entropy from accuracy. Ask students to name confounders (training schedule, label quality, data distribution) before claiming causality; do not use the phrase *statistically significant* for these runs.

### Slide 53 — Quick check 7 | 2:14–2:16

**Say:**
> “You have 2,000 labeled images. Training accuracy is 99.8 percent, but validation accuracy is 71 percent. We have checked that preprocessing and labels are consistent and that training and validation follow the intended evaluation distribution. We also have a pretrained model trained on **related images**. Of these choices, which is a promising **next experiment to evaluate**, not a guaranteed cure?”

**After voting, say:**
> “**B: evaluate transfer learning** using the relevant pretrained model. With few labeled examples, previously learned features can be useful. The train–validation gap indicates a generalization problem, but alone it would not tell us whether pretraining, more data, augmentation, model simplification or other approaches will win. Without the stated data checks and a suitable pretrained model, there is no unique ‘best’ option.”

**Do:** Ask which evidence could overturn the choice: mismatch of source and target images, noisy labels, or validation shift. Compare results on the **same** untouched validation split.

### Slide 54 — Today in three sentences | 2:20–2:30

**Say:**
> “Here is the entire lecture in three ideas. First, **representation**: nonlinear hidden layers let a network learn features that a simple classifier can use, even when the original input space is difficult. Second, **learning**: a loss evaluates the current prediction, backpropagation computes gradients efficiently, and an optimizer uses those gradients to update weights. Third, **successful training and generalization**: suitable initialization, residual connections and optimizers help us fit the training data; validation curves and regularization tell us whether we are learning patterns that transfer beyond those examples. Before the PyTorch session, try constructing the 784–128–64–10 model, checking its **109,386 parameters**, checking the scale of its initial loss, overfitting a small sample, and plotting training versus validation loss. If a network misbehaves, start by testing the basics, not by adding complexity.”

**Exit ticket — say:**
> “On one line, tell me which idea remains least clear: feature representations, loss functions, backpropagation, gradient flow, or generalization. I will use your answers to plan our next session.”

**Do:** Leave time for questions and the one-minute exit ticket. Explain that slides 55–72 are review and Session 2 preview; they are not another 18 scheduled lecture slides.

---
## Appendix A — Review of today's lecture (slides 55–62; use only if needed)

### Slide 55 — Appendix A

**Say:**
> “The next section is for review, not new required material during today's 150-minute lecture. It collects the quick-check answers, a notation reference, an explanation of why depth can be useful beyond universal approximation, a short history, guidance on when **not** to use a neural network, and a numerical method for checking your derivatives. You can return to these pages while implementing the model or preparing for the next class.”

**Do:** If presenting after class, identify which appendix items answer student exit-ticket questions.

### Slide 56 — Quick-check answers (1–4)

**Say:**
> “Review the first four quick checks by reconstructing their logic. **One:** affine hidden layers collapse to an affine function. **Two:** `Linear(784,128).weight` has shape `(128,784)`. **Three:** the inspected program incorrectly feeds already-softmaxed probabilities to `CrossEntropyLoss`; remove the final softmax and pass logits. This is a **code-based** diagnosis now, not a deduction from an arbitrary plateau. **Four:** if `w₂=0` in our worked network, the backpropagated gradient to `w₁` is zero.”

**Do:** Have students explain *why* one answer is correct rather than reciting the letter choices. Slide 25 was revised so its result is supported by the supplied code.

### Slide 57 — Quick-check answers (5–7)

**Say:**
> “The remaining checks concern optimization and generalization. **Five:** Xavier is smaller than He in the equal-width ReLU example, so the forward RMS signal tends to shrink under the idealized assumptions. **Six:** when adding momentum makes a particular run unstable, try retuning the learning rate and inspect the curves; **there is no universal divide-by-ten rule**. **Seven:** with the evaluation pipeline checked and a related pretrained model available, transfer learning is a sensible experiment to compare with your baseline, not a guaranteed best solution.”

**Do:** Invite students to give one case where transfer learning could hurt and one case where momentum could tolerate a larger learning rate than plain gradient descent.

### Slide 58 — Notation used all session

**Say:**
> “Here is a compact decoder for our mathematical notation. Superscripts in **square brackets** identify network layers. Parentheses identify **training examples**. `z[l]` is the vector before an activation, while `a[l]` is the vector after it. `δ[l]` means the derivative of the loss with respect to `z[l]`. We often use `L` for one example's loss and `J` for an average over a mini-batch. In hand derivations, an individual example is a **column vector** and `W` multiplies it on the left. In PyTorch, mini-batches store examples as **rows**, so `nn.Linear` performs `x @ W.T + b`. These are the same mathematical transformation in two tensor layouts.”

**Do:** Put an example `(batch, in_features) @ (in_features, out_features) → (batch, out_features)` under the formulas.

### Slide 59 — Why deep? Composition multiplies what each layer can do

**Say:**
> “If a wide one-hidden-layer network can approximate every continuous function on a compact domain, why do we need depth? The theorem allows the required width to become enormous. Depth can represent some **compositional structures** more efficiently. For example, repeatedly applying a simple tent-shaped function creates more and more linear pieces—doubling the number of pieces at each composition. Formal depth-separation results construct function families that a deeper network represents compactly but substantially shallower networks require many more units to approximate. The important caveat is that these are **existence and worst-case results**. They do not mean that every deeper network beats every shallow network on real data; the benefit must be tested alongside optimization, data quality and architecture.”

**Do:** Sketch a one-peak tent map and then a two- or four-peak composition. Avoid treating parity as a generic exponential-depth-separation example without specifying computational model and restrictions.

### Slide 60 — From 1943 to today: backprop was not the bottleneck

**Say:**
> “Today's methods emerged gradually. Early threshold-neuron models gave a simple mathematical unit; perceptrons made learning algorithms concrete; work on multilayer differentiation supplied backpropagation decades before very deep networks became broadly practical. Backpropagation was popularized for neural networks in the 1980s, but success at scale also required better data, computational hardware, architectures and optimization practices. LeNet demonstrated useful convolutional networks, AlexNet helped popularize GPU-trained deep vision in 2012, residual networks enabled much greater depth, and Transformers shifted the frontier in sequence modeling. This is not a history of one magic invention. It is an interaction between **ideas, data, compute and engineering**.”

**Do:** Correct the overly literal interpretation of the slide title: backpropagation was **not the only bottleneck**, but it was an important enabling algorithm.

### Slide 61 — When not to reach for a neural network

**Say:**
> “Neural networks are powerful, but starting with one is not always the best engineering decision. With a few hundred structured rows, try regularized linear models or tree ensembles first. For many tabular classification and regression problems, strong gradient-boosted trees remain competitive, and pretrained tabular models are another option. If every decision must be audited in detail, simpler models may be easier to explain and maintain. On the other hand, deep networks often shine when learning directly from images, audio, text or other high-dimensional signals where useful features are hard to hand-design. The default rule is: **establish a simple, reproducible baseline, then justify the added complexity**.”

**Do:** Discuss dataset size, latency, cost, evaluation metric, deployment constraints and interpretability together; never suggest that model choice follows from sample count alone.

### Slide 62 — Verify hand-written gradients with centred differences

**Say:**
> “If we implement a custom backward operation, how can we check it? Choose one parameter, perturb it slightly upward and downward, and calculate the difference between the two loss values divided by twice the perturbation. This **centred finite difference** approximates the derivative. Compare it with the backpropagated gradient using a normalized relative error; check a **sample** of parameters rather than every weight in a large model. For reliable tests, make the computation deterministic, turn off dropout and randomness, and use float64 if possible. Very small relative errors are reassuring; large ones suggest a bug. Take care around ReLU's nondifferentiable point: a finite difference that crosses a kink may disagree even when the chosen backpropagation convention is implemented correctly.”

**Do:** Write `(L(θ+ε)−L(θ−ε))/(2ε)` and distinguish numerical **gradient checking** from an optimizer performing gradient descent.

---

## Appendix B — Preview of Session 2 (slides 63–70; optional)

### Slide 63 — Session 2 preview

**Say:**
> “These slides point toward the practical PyTorch session. We will compare normalization layers, revisit the relationship between model size and generalization, match PyTorch losses to the correct target shapes, and examine bugs that do not necessarily crash a training loop. We will also look at schedulers and assemble a complete training recipe. Think of this as a reference checklist for your first serious training experiment, not another collection of facts to learn in the remaining minutes today.”

**Do:** Link each preview topic to an earlier challenge: normalization to gradient flow; target formats to cross-entropy; schedulers to learning-rate stability.

### Slide 64 — Batch normalization: standardize, then rescale

**Say:**
> “Batch normalization standardizes intermediate features using statistics from a mini-batch and then applies a learned scale, gamma, and shift, beta. That learned rescaling means normalization does not force all learned features to remain mean zero and variance one. In **training mode**, BatchNorm uses the current batch's statistics and updates running estimates. In **evaluation mode**, it normally uses those running estimates instead. That is a concrete reason why forgetting `model.eval()` can change evaluation results. BatchNorm has helped train deep CNNs at larger learning rates. Its original explanation involved reducing ‘internal covariate shift,’ but later research questioned that as the primary mechanism. A frequently discussed alternative is that normalization can make optimization better behaved, though the full story is nuanced.”

**Do:** Draw `conv → BatchNorm → ReLU`. Mention PyTorch uses the biased variance for training-time normalization but stores an unbiased running-variance estimate; the running-statistics update factor named `momentum` is not the same concept as optimizer momentum.

### Slide 65 — Normalization variants: which axes, which statistics

**Say:**
> “Different normalization methods answer one question differently: **over which coordinates do we estimate a scale?** BatchNorm in a CNN typically uses the batch and spatial axes for each channel, so training behaviour depends on the other examples in that batch. LayerNorm normalizes within each example and is widely used in Transformers. GroupNorm forms groups of channels per example and is helpful when batch sizes are small. InstanceNorm treats each example and channel separately across spatial positions and is common in some style-transfer pipelines. RMSNorm uses a root-mean-square scale **without subtracting the mean**. These are not interchangeable operations, and their relative merits depend on architecture and batch size.”

**Do:** Identify the axes for a tensor of shape `(N,C,H,W)`. Explain that standard LayerNorm axes are specified by `normalized_shape`; saying it *always* normalizes “all features” is an oversimplification.

### Slide 66 — Double descent: test error can fall again

**Say:**
> “Classical intuition says a model that is too small underfits, then improves, then becomes too flexible and overfits—a U-shaped error curve. But in some modern regimes, test error gets worse near the point where the model can **just fit the training data**, then improves again when the model becomes far larger. That pattern is called **double descent**. The slide shows a deliberately simplified simulation using random features and least squares, not a universal property of deep networks. Near the interpolation threshold, fitting noisy observations can lead to unstable solutions; further overparameterization can admit better-behaved interpolating solutions, depending on the learning rule and data. Regularization can remove or change the peak. This is one reason ‘always shrink an overfitting network’ is too simplistic.”

**Do:** Point out the simulated `p≈n` threshold and label the results **simulation, not law**. Remind students that increasing model size can also worsen generalization on other problems.

### Slide 67 — PyTorch loss functions and target formats

**Say:**
> “This table is worth keeping beside your notebook. For scalar regression with `MSELoss` or `HuberLoss`, predictions and targets should have the **same shape** and a floating-point type. For binary classification, `BCEWithLogitsLoss` takes raw logits and floating targets between zero and one. For a K-class single-label problem, `CrossEntropyLoss` takes `(batch,K)` logits and usually `(batch,)` **integer class indices**. It can also accept `(batch,K)` **floating probability targets** when needed, for example for mixed labels. Multilabel classification is different from multiclass: labels are independently present or absent, so multiple sigmoid logits with `BCEWithLogitsLoss` are usually appropriate. Do not apply sigmoid or softmax to the logits before these fused classification losses.”

**Do:** Contrast “cat OR dog” (multiclass) with “has wheels AND is red” (multilabel). Verify loss shapes explicitly in the next notebook session.

### Slide 68 — Six training mistakes that may not raise an exception

**Say:**
> “Debugging deep learning often means detecting **silent semantic errors**. A second softmax may still produce plausible accuracy. Forgetting to clear gradients causes unintended accumulation. Omitting `model.eval()` means dropout or BatchNorm may still behave as if training. Putting layers in a plain Python list instead of `ModuleList` can leave their parameters **unregistered**. Replacing a model after constructing its optimizer can leave the optimizer updating obsolete parameters. Finally, mismatched regression shapes such as `(batch,1)` and `(batch,)` may broadcast into an unintended matrix; PyTorch may warn, but not stop execution. A loss curve on its own is not evidence that the intended model is being trained.”

**Do:** Ask students to name a one-line check for each bug, such as parameter counts, printed gradient norms or comparing shapes.

### Slide 69 — Step each scheduler at the right frequency

**Say:**
> “A learning-rate scheduler changes the optimizer's learning rate during training, but not every scheduler should be stepped at the same time. **OneCycleLR** is designed to step once per optimizer update—a mini-batch step. A warmup schedule counted in optimizer steps is likewise stepped per batch. **CosineAnnealingLR** can be configured in epochs or steps; its `T_max` must use the same unit as the stepping frequency. **MultiStepLR** is often configured around epochs. **ReduceLROnPlateau** is normally called after validation with the metric being monitored, for example validation loss. If we step at the wrong frequency, the code may run but implement a completely different learning-rate curve.”

**Do:** Recommend recording actual learning-rate values, not merely the scheduler name, in experiment logs.

### Slide 70 — A complete training recipe (illustrative settings)

**Say:**
> “This slide joins the pieces into one practical recipe. Construct a model, choose the correct logits-based classification loss, set up an optimizer such as **AdamW**, and optionally use a learning-rate schedule. For a one-cycle schedule, the real PyTorch constructor must receive `max_lr` and **either** `total_steps` **or** both `epochs` and `steps_per_epoch`. This slide deliberately abbreviates that call and is **pseudocode**, not runnable code as printed. In each training batch, clear gradients, compute logits and loss, call backward, optionally clip the gradient norm, then step the optimizer **before stepping OneCycleLR**. Validate using evaluation mode and no gradient tracking, and save the best checkpoint. Hyperparameters such as weight decay 0.05, label smoothing 0.1 and clipping at 1.0 are **illustrations**, not universal defaults.”

**Do:** Show a correct minimal setup such as `OneCycleLR(optimizer, max_lr=3e-4, epochs=num_epochs, steps_per_epoch=len(train_loader), pct_start=0.1)`. Note that `cycle_momentum=True` also changes momentum-related optimizer settings by default where supported.

---

## Sources and further reading (slides 71–72)

### Slide 71 — References and further reading (1/2)

**Say:**
> “If you want one systematic textbook explanation, begin with Goodfellow, Bengio and Courville's *Deep Learning* or Simon Prince's *Understanding Deep Learning*. For an implementation-oriented treatment, *Dive into Deep Learning* and the Stanford CS231n materials are useful. The papers listed here support the history of neural networks, universal approximation, ReLU and feature representation, while the Fashion-MNIST paper documents the dataset we have used. You do not need to read all of these in order. Choose a topic where your reasoning feels shaky, find the corresponding source, and verify the assumptions behind the result.”

**Do:** Prioritize one text and one implementation reference rather than presenting every citation aloud. Emphasize primary papers as deeper follow-ups, not necessary first readings.

### Slide 72 — References and further reading (2/2)

**Say:**
> “The second reference slide maps more closely to the techniques we just learned: reverse-mode differentiation and backpropagation, learning-rate dynamics, initialization, residual networks, Adam and AdamW, normalization, dropout, augmentation and double descent. The historical papers often answer **why a method was invented**, while modern PyTorch documentation tells us its **current software contract**. Both matter. When an implementation detail conflicts with a remembered formula, check the installed version's documentation and reproduce a small numerical example. That habit is part of doing reliable machine learning.”

**Do:** Close by restating the difference between mathematical claims, experimental observations and library-specific API behaviour.

---

## Important mathematical note for the instructor: momentum stability

For the **conventional heavy-ball** update `v_t=βv_{t−1}+∇J(θ_t)`, `θ_{t+1}=θ_t−ηv_t`, and an SPD quadratic `J=½ θᵀHθ`, one Hessian eigendirection (`λ>0`) obeys `θ_{t+1}=(1+β−ηλ)θ_t−βθ_{t−1}`. For `0≤β<1`, its characteristic roots are inside the unit disk iff `0<ηλ<2(1+β)`. Thus `0<η<2(1+β)/λ_max` ensures asymptotic convergence on this model. This **does not** provide a global bound for general neural networks, stochastic momentum, nonstationary curvature, or different optimizer conventions. For example, with `λ_max=10` and `β=0.9`, the quadratic bounds are `η<0.2` without momentum and `η<0.38` with this form of momentum. That is why the constant-gradient `10g` observation cannot justify a universal LR decrease.

## Version-checked technical reference points (October 2026)

The following points were checked against the current **PyTorch 2.14 / stable documentation**. They are specifically relevant to slides **19–20, 22–27, 35, 37, 48, 64, 67–70**:

1. `nn.Linear(in_features, out_features)` stores weights with shape `(out_features, in_features)` and maps row-oriented batches through `x @ W.T + b`. See https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html .
2. `nn.CrossEntropyLoss` expects logits and supports class-index targets or probability-distribution targets. See https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html .
3. `torch.optim.Adam(..., decoupled_weight_decay=False)` uses the coupled form by default; setting it to `True` gives AdamW-style decoupling. See https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html and https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html .
4. `OneCycleLR` steps **per batch** and requires `max_lr` plus a defined training length (`total_steps` or `epochs` and `steps_per_epoch`). See https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html .
5. `model.eval()` does **not** disable autograd; `no_grad()` and inference mode control gradient recording separately. See https://docs.pytorch.org/docs/stable/notes/autograd.html and https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html .
6. BatchNorm uses training mini-batch statistics and, with running statistics enabled, stored estimates at evaluation. See https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html .

**Scope note:** Experimental loss values, accuracies, best epochs, linear-region counts and simulation figures quoted above are those **reported by the supplied slide deck**. This document has not independently executed the supplied `fmnist_experiments.py` or `small_figures.py` scripts. Where estimates rely on assumptions—for example fixed-Gaussian noise, symmetric random pre-activations, quadratic optimization or a suitable pretrained model—those conditions are made explicit.

**Checked source for momentum implementation:** https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html (note PyTorch supports momentum conventions and Nesterov variants; the derivation above explicitly specifies its recurrence).
