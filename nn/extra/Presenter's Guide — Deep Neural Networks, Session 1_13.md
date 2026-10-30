# Presenter's Guide — Deep Neural Networks, Session 1

Oct 25, 2026 · @Ben

## How to use this guide

For each of the deck's 59 slides this guide gives what to say, the technical points to state precisely, and the questions students are likely to ask, with answers. Timings assume 150 minutes with one 10-minute break; the slide-by-slide timings finish Part 3 at 2:08, leaving 7 minutes of buffer before the 2:15 shown on slide 3; slides 49–59 are an appendix for reference and questions, not for presenting.

| Segment | Slides | Clock | Minutes |
| --- | --- | --- | --- |
| Opening | 1–3 | 0:00–0:05 | 5 |
| Part 1 · What a network computes | 4–19 | 0:05–0:45 | 40 |
| Part 2 · How a network learns | 20–29 | 0:45–1:10 | 25 |
| Break | 30 | 1:10–1:20 | 10 |
| Part 3 · Backpropagation and curves | 31–47 | 1:20–2:15 | 55 |
| Wrap-up, exit ticket, questions | 48 | 2:15–2:30 | 15 |
| Appendix (reference) | 49–59 | — | — |

Each slide entry has the same parts:

- **Say** — a suggested script in plain speech. Paraphrase freely.
- **Get it right** — statements that must be precise, and the caveats behind them.
- **If students ask** — likely questions with answers you can give in under a minute.
- **Tip** — delivery advice, where useful.

**Colour code on the slides:** teal = forward signal and data; orange = gradients and the backward pass; red = warnings and failure modes. In plots, filled violet = class 1 and hollow grey = class 0. Point this out once on slide 1; students use it for the rest of the session.

**Numbers to know by heart.** All come from the shipped NumPy scripts (exp\_fmnist.py, figs.py) with fixed seeds, so they reproduce exactly.

| Quantity | Value |
| --- | --- |
| Running-example parameters (784–128–64–10) | 109,386 (layer 1: 100,480 = 92%) |
| Data split | 55,000 train · 5,000 validation · 10,000 test |
| Input standardization (training split) | μ = 0.286, σ = 0.353 |
| Updates per epoch at B = 256 | 215 |
| First-batch loss, 3 seeds | 2.29–2.33 (ln 10 = 2.30) |
| Epoch with lowest validation loss, seeds 0/1/2 | 9 / 12 / 15 |
| Test accuracy, softmax regression (3-seed mean) | 84.5% |
| Test accuracy, MLP (3-seed mean; runs 88.4–88.7%) | 88.5% |
| Worked example: z₁, a₁, z₂, a₂, L | 1.1, 0.7503, 0.3002, 0.5745, 0.5543 |
| Worked example: δ₂, ∂L/∂a₁, δ₁ | −0.4255, −0.3404, −0.0638 |
| Worked example: loss after one step, η = 0.5 | 0.4166 |
| Gradient check of the NumPy MLP | max relative error 3 × 10⁻⁹ |

## Before class

Ten minutes of setup prevents most problems; the voting method and the worked-example handout matter most.

- [ ] Open the deck in Presenter View and check that the equations (images) and fonts (Cambria, Calibri) display correctly on the room's computer.
- [ ] Decide how students vote in the five quick checks: A–D cards, fingers (1–4), or a polling tool. Have it ready before slide 11.
- [ ] Print or post the worked-example setup (slide 33): w₁ = 0.5, b₁ = 0.1, w₂ = 0.8, b₂ = −0.3, x = 2.0, y = 1. Students need a calculator or phone; tell them at the start.
- [ ] Have a whiteboard or tablet ready for three live drawings: the XOR square (slide 7), the chain rule with a branch (slide 38), and a small matrix W with one column circled (slide 39).
- [ ] Optional: open playground.tensorflow.org in a browser tab (spiral data, ReLU) for slide 14 if you are ahead of time.
- [ ] Put the source scripts (exp\_fmnist.py, figs.py) on the course page; students will ask where the numbers come from.
- [ ] Prepare index cards or a form for the exit ticket on slide 48.
- [ ] Set a visible timer for the break and the paired activities (slides 13 and 33).

**If you are running late**, cut in this order: slide 14 (patchwork) to 30 seconds; slide 29 (momentum and Adam) to one sentence; slide 42 (autodiff) to its last box; quick check 2 to a show of hands without revote. Never cut slides 33–36 (the worked example) or 38–39 (branching and Wᵀδ): they carry the session's hardest idea.

**If you are running early**, use the playground demo on slide 14, derive p − y on the board from Appendix A3, or ask students to predict the curves on slide 46 before revealing them.

## Opening · slides 1–3

The opening sets three expectations in five minutes: one running example, active participation every few slides, and a colour code that carries meaning.

### Slide 1 · Title (0:00–0:02)

**Say:** "Welcome. Today we go from a single neuron to a network that learns. By the end you'll have computed a forward pass, run backpropagation by hand, and read real training curves. We stay framework-neutral today; PyTorch starts next session. One thing to notice now: colours mean something. Teal is the forward signal, orange is gradients flowing backward, red is a warning. In plots, filled violet is class 1 and hollow grey is class 0."

**Get it right:**

- Say that every number in the deck comes from seeded NumPy scripts shipped with the course. That earns trust and invites students to reproduce results.

**If students ask:**

- **Q: Will we use PyTorch today?** A: No. Today is the mathematics and the logic; the pseudocode maps one-to-one onto PyTorch next session.
- **Q: Do I need a GPU for this course?** A: Not for today's model. The running example trains in well under a minute per seed on a laptop CPU in NumPy.

**Tip:** Ask everyone to have a calculator or phone ready for the paired exercise at about 1:23.

### Slide 2 · Learning outcomes (0:02–0:04)

**Say:** "Four outcomes, each checked during the session by a quick question. Two are about what a network computes, one about how it learns, one about backpropagation and reading curves. You already know logistic regression, matrix–vector products and the chain rule; today we only stack them."

**Get it right:**

- Prerequisites stated on the slide: matrix–vector products, chain rule, logistic regression and its log-likelihood, train/validation/test splits, NumPy. If many students are shaky on the chain rule, spend 30 seconds on one example (d/dx of σ(3x)) now; it pays off in Part 3.

**If students ask:**

- **Q: Are the quick checks graded?** A: No. They are for learning: vote, argue with a neighbour, revote. Research on peer instruction (Mazur, 1997; Crouch & Mazur, 2001) shows the discussion is where most learning happens.

### Slide 3 · Plan and running example (0:04–0:05)

**Say:** "Three parts and a break. One example all session: Fashion-MNIST, 70,000 small grey images of clothing in 10 classes, and one network with layers of 784, 128, 64 and 10 units. We build it in Part 1, train it in Part 2, and diagnose it in Part 3."

**Get it right:**

- Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017) was designed as a harder drop-in replacement for MNIST: same 28×28 format, 60,000 training and 10,000 test images.
- The deck splits the official 60,000 training images into 55,000 train and 5,000 validation with a fixed seed; the 10,000 test images are used only for final numbers.

**If students ask:**

- **Q: Why not MNIST digits?** A: MNIST is too easy (a linear model reaches about 92% and small MLPs about 98%), so differences between models are hard to see. Fashion-MNIST leaves room: a linear model gets about 84%, our MLP about 88.5%.
- **Q: Why a fully connected network for images?** A: It is the simplest network that shows every core idea. We will see its limit on slide 45; convolutional networks come later.

## Part 1 · What a network computes · slides 4–19

Part 1 builds one argument: a single neuron is logistic regression and draws one straight boundary; XOR shows that is not enough; a hidden layer with a nonlinearity fixes it; and an MLP repeats that rule layer by layer.

### Slide 4 · Part 1 divider (0:05)

**Say:** "Four questions for the next 40 minutes. Keep them in mind; we'll tick them off."

**Tip:** Read the questions aloud and move on in under 30 seconds.

### Slide 5 · Same raw inputs: line vs bend (0:05–0:07)

**Say:** "Same 400 points, same two input coordinates. Logistic regression can only draw a straight line, so it gets 87% on fresh test points. A small network with two hidden layers bends its boundary and gets 97%. The question for the next half hour is how it bends."

**Get it right:**

- The numbers are on 2,000 fresh test points, not the training data: logistic regression 87.5% train / 86.9% test; MLP 2–12–12–1 with ReLU 97.0% train / 96.7% test.
- Be fair to classical ML: kernel SVMs and hand-made polynomial features also give curved boundaries. What is new is that the network learns the transformation from data.

**If students ask:**

- **Q: Isn't this just a kernel SVM?** A: A kernel SVM uses a fixed feature map chosen in advance (through the kernel). A network learns its feature map from the data along with the classifier. On two moons both work; on images, learned features win by a wide margin.
- **Q: Is 97% overfitting?** A: No: train and test accuracy are almost equal (97.0% vs 96.7%). The remaining errors are points that overlap because of the noise in the data.

### Slide 6 · One neuron = logistic regression (0:07–0:10)

**Say:** "A neuron computes a weighted sum plus a bias, z = wᵀx + b, then applies an activation. With the sigmoid, this is exactly logistic regression: z is the log-odds of class 1. Geometrically, z = 0 is a straight line in 2-D (a hyperplane in general). The weight vector w is perpendicular to it and points toward the positive side. The bias slides the line without rotating it."

**Get it right:**

- Distance from the origin to the boundary is |b| / ‖w‖ (for w ≠ 0). The signed distance of a point x is (wᵀx + b) / ‖w‖.
- The decision boundary at probability 0.5 is z = 0, because σ(0) = 0.5 and σ is strictly increasing.
- Scaling w and b by the same k > 0 leaves the boundary unchanged but makes the sigmoid k times steeper. This is why unregularized logistic regression on separable data drives ‖w‖ to infinity.

**If students ask:**

- **Q: Is this how real neurons work?** A: Only loosely. McCulloch and Pitts (1943) modelled a neuron as a threshold unit. Real neurons have spikes, timing, dendritic computation and much more. Treat "neuron" as a name, not a claim about biology.
- **Q: Why the sigmoid and not a hard step?** A: The step has zero derivative everywhere except at 0, where it is undefined, so gradient descent gets no signal. The perceptron learning rule (Rosenblatt, 1958) can train a single step unit on linearly separable data, but it does not extend to stacked layers.
- **Q: Why is z the log-odds?** A: Solve p = σ(z) for z: z = log(p / (1 − p)). Logistic regression models the log-odds as linear in x.

### Slide 7 · XOR is not linearly separable (0:10–0:13)

**Say:** "Thirty seconds: try to draw one straight line with both violet points on one side and both grey points on the other. … You can't. Here's the proof. Write the condition each point imposes. Adding rows 2 and 3 gives w₁ + w₂ + 2b > 0. Since b ≤ 0 from row 1, w₁ + w₂ + b > −b ≥ 0. That contradicts row 4."

**Get it right:**

- The proof covers any single affine threshold, and therefore also a sigmoid neuron thresholded at 0.5, because the sigmoid is monotone.
- Minsky and Papert (1969) made this limitation of single-layer perceptrons famous. Multi-layer networks could represent XOR then too; what was missing was a practical way to train them.

**If students ask:**

- **Q: Could one neuron with a different activation solve XOR?** A: Yes, if the activation is not monotone. With z = x₁ + x₂ − 1 and g(z) = 1 − z², the outputs on (0,0), (0,1), (1,0), (1,1) are 0, 1, 1, 0. The impossibility proof is for one linear threshold, which covers the usual monotone activations (step, sigmoid, tanh, ReLU).
- **Q: Why does XOR matter? It's a toy.** A: It is the smallest example of a pattern where the label depends on an interaction between inputs, not on each input separately. Images are full of such interactions.

**Tip:** Draw the unit square on the board and let a student try a line before showing the proof.

### Slide 8 · Fix 1: a hand-made feature (0:13–0:15)

**Say:** "Add a third feature, the product x₁x₂. In the three-dimensional space (x₁, x₂, x₁x₂) a single plane separates the classes. In fact, on the four corners, y = x₁ + x₂ − 2x₁x₂ exactly, which is linear in the new features."

**Get it right:**

- This is classical feature engineering. A degree-2 polynomial kernel would include x₁x₂ implicitly.
- The point to land: changing the representation turned an impossible problem into an easy one. The catch: for 784 pixels, which products of which pixels should we add?

**If students ask:**

- **Q: Why not add all pairwise products?** A: For 784 inputs that is 784·783/2 = 306,936 new features, and you might also need triple products. Kernels avoid computing them explicitly but scale poorly with the number of training examples. Networks learn a compact set of useful features instead.

### Slide 9 · Fix 2: learn the features (0:15–0:17)

**Say:** "Here is the neural-network idea. Instead of designing h by hand, compute it with learnable parameters: h = g(W⁽¹⁾x + b⁽¹⁾). Then put the logistic regression you already know on top of h. Crucially, both parts are trained together, by gradient descent on one loss."

**Get it right:**

- The output neuron is literally logistic regression, but on learned features h instead of raw inputs x.
- "Deep" means stacking this idea: layer 2 builds features out of layer-1 features.
- End with the open question on the third card: what must g be? It sets up the next slide.

**If students ask:**

- **Q: How does the network know which features to learn?** A: It doesn't decide in advance. Gradient descent changes W⁽¹⁾ in whatever direction lowers the loss, so features that help the final prediction emerge. You'll see learned features from our running example on slide 45.
- **Q: Do learned features transfer to other tasks?** A: Often, especially early layers of image networks (Yosinski et al., 2014). That is the basis of transfer learning, covered in a later session.

### Slide 10 · Without a nonlinearity, layers collapse (0:17–0:19)

**Say:** "Suppose g(z) = z. Substitute layer 1 into layer 2: you get W⁽²⁾W⁽¹⁾x plus a constant. A product of matrices is just a matrix. So two linear layers equal one linear layer, and by induction so do ten. The nonlinearity between layers is what buys expressive power."

**Get it right:**

- The collapse holds with biases: b′ = W⁽²⁾b⁽¹⁾ + b⁽²⁾.
- A stack of linear layers can still differ from one layer in rank: if a middle layer is narrow, W′ has rank at most that width. It is still linear.

**If students ask:**

- **Q: Then why do linear layers appear back-to-back in some architectures?** A: A narrow linear bottleneck (low-rank factorization) can save parameters or act as a projection, but it adds no expressive power beyond a single linear map of that rank.
- **Q: Is a nonlinearity only at the output enough?** A: No. That is exactly the next quick check.

### Slide 11 · Quick check 1 (0:19–0:23)

**Run it:** Silent vote (30 s), neighbour discussion (60 s), revote. Then explain.

**Answer: C — only a hyperplane.** The identity hidden layers collapse into one affine map z = W′x + b′. The sigmoid is nonlinear but strictly increasing, so ŷ > 0.5 exactly when z > 0. The boundary is a hyperplane, whatever the depth or widths.

**Common wrong answers:**

- A ("any smooth curve"): students see "sigmoid" and think "nonlinear, so curved". The output nonlinearity changes the probabilities, not the shape of the 0.5 boundary.
- D ("depends on widths"): widths change the rank of W′, never its linearity.

### Slide 12 · Three common activations (0:23–0:25)

**Say:** "Sigmoid squashes to between 0 and 1, tanh to between −1 and 1, and ReLU keeps positive values and zeroes the negatives. For hidden layers, ReLU is the standard default: it's cheap and its slope is exactly 1 wherever it's active. Why the flat tails of sigmoid and tanh matter for learning, you'll measure yourselves in Part 3."

**Get it right:**

- ReLU(z) = max(0, z). Its derivative is 1 for z > 0 and 0 for z < 0; at exactly 0 it is undefined and frameworks use 0 by convention.
- Sigmoid is still the right choice at a binary output, where it turns a logit into a probability.
- ReLU's rise: Nair & Hinton (2010) and Glorot, Bordes & Bengio (2011); AlexNet (Krizhevsky et al., 2012) made it standard. Many modern Transformer models use smooth variants such as GELU (Hendrycks & Gimpel, 2016) or SiLU/Swish, often inside gated units such as SwiGLU (Shazeer, 2020).

**If students ask:**

- **Q: Isn't ReLU linear? How can it bend anything?** A: It is piecewise linear: linear on each side of 0, but not linear overall, because ReLU(a + b) ≠ ReLU(a) + ReLU(b) in general. That kink, repeated across many units, is enough (next two slides).
- **Q: What is a "dead" ReLU?** A: A unit whose pre-activation is negative for every training input. Its gradient is then zero, so it may never recover. Too large a learning rate can cause it; Leaky ReLU (small slope for z < 0) is one remedy.
- **Q: Which activation is best?** A: There is no universal best. ReLU is a strong default for MLPs and CNNs; GELU or SiLU variants are common in Transformers. The differences are usually smaller than those from learning rate or data.

### Slide 13 · Two ReLUs solve XOR (0:25–0:29)

**Run it:** Pairs, 2 minutes: compute h₁, h₂ and s for the four inputs.

**Answers:**

| Input (x₁, x₂) | h₁ = ReLU(x₁ + x₂) | h₂ = ReLU(x₁ + x₂ − 1) | s = h₁ − 2h₂ |
| --- | --- | --- | --- |
| (0, 0) | 0 | 0 | 0 |
| (0, 1) | 1 | 0 | 1 |
| (1, 0) | 1 | 0 | 1 |
| (1, 1) | 2 | 1 | 0 |

**Say:** "Look at the right-hand panel. The hidden layer moves the points: both class-1 points land on the same spot, (1, 0). Now a single straight line separates the classes. The hidden layer changes the representation; the output neuron just draws the line."

**Get it right:**

- In hidden space the output's decision line is s = 1/2, that is h₁ − 2h₂ = 1/2.
- This network is exact on the four binary corners. It is not a probability model on arbitrary real inputs.
- h₁ behaves like "at least one input is on" (OR-like) and h₂ like "both are on" (AND-like); XOR = OR minus 2·AND on these points.

**If students ask:**

- **Q: Would gradient descent find these exact weights?** A: Usually it finds a different solution that also works; many weight settings solve XOR. With very few hidden units, training can also get stuck, which is one reason networks are usually made wider than the bare minimum.

### Slide 14 · Many ReLUs: a patchwork of linear pieces (0:29–0:31)

**Say:** "This is the network from slide 5. Each coloured patch is a region where every ReLU is either on or off. Inside one patch, the whole network is exactly linear. So the decision boundary is a chain of straight segments, and with enough segments it can follow any curve."

**Get it right:**

- The count 67 is the number of distinct on/off patterns of the 24 ReLUs found on a 900 × 600 grid in this window. Tiny regions can be missed, so it is a lower bound.
- Universal approximation (Cybenko, 1989; Hornik, 1991; Leshno et al., 1993): one hidden layer with enough units of a non-polynomial activation can approximate any continuous function on a compact set to any accuracy. It is an existence result: it says nothing about how many units, whether training finds them, or whether the result generalizes (Appendix A4).

**If students ask:**

- **Q: If one hidden layer is enough, why go deep?** A: Depth can be exponentially more efficient: some functions that a deep ReLU network represents with few units need exponentially many units in a shallow one (Telgarsky, 2016; Montúfar et al., 2014 on linear regions). In practice, deep networks also learn reusable hierarchies of features.
- **Q: Does the number of regions grow with more units?** A: Yes. For ReLU networks it can grow polynomially with width and exponentially with depth in the best case (Montúfar et al., 2014). Trained networks typically use far fewer regions than the maximum.

**Tip:** If you are ahead, open playground.tensorflow.org, pick the spiral dataset and ReLU, and add hidden units live.

### Slide 15 · The forward pass, one rule per layer (0:31–0:34)

**Say:** "Every layer does the same two things: an affine map z = W a + b, then the activation. The last layer has no activation; its outputs are the logits. Check the shapes every time: W⁽ˡ⁾ has one row per unit of layer l and one column per unit of the layer below."

**Get it right:**

- Notation: z = pre-activation, a = activation, a⁽⁰⁾ = x, and z⁽ᴸ⁾ = logits.
- Column-vector convention for derivations; code stores examples as rows, so it computes A Wᵀ + b (slide 17).

**If students ask:**

- **Q: Why no activation on the last layer?** A: The loss turns logits into probabilities itself, in a numerically stable way (slide 18, Appendix A3). For regression the output should be unbounded, so no activation either.
- **Q: What does "a layer" mean in code?** A: A weight matrix and a bias vector, plus the activation applied after it. In PyTorch next session that is nn.Linear followed by nn.ReLU.

### Slide 16 · Running example: 784–128–64–10 (0:34–0:36)

**Say:** "Each 28-by-28 image is flattened into 784 numbers. A mini-batch of 256 images becomes a 256 × 784 matrix, one image per row. Two hidden layers of 128 and 64 ReLU units, then 10 logits, one per class. Predict: is W⁽¹⁾ 784 × 128 or 128 × 784?"

**Get it right:**

- The batch size 256 was chosen so that it never equals a layer width; shape mistakes are then visible.
- The input shape (256, 28, 28) is a batch of 256 grey images. In PyTorch it is usually (256, 1, 28, 28), with an explicit channel dimension; flattening gives (256, 784) either way.

**If students ask:**

- **Q: Why 128 and 64? How do you choose widths?** A: These are hyperparameters, chosen by experiment on the validation set. Shrinking widths toward the output is a common habit, not a rule.
- **Q: Doesn't flattening lose information?** A: It keeps every pixel value but throws away which pixels are neighbours. The MLP has to learn spatial structure from scratch, which is why convolutional networks do better (slide 45).

### Slide 17 · 109,386 parameters (0:36–0:38)

**Say:** "Answer: 128 × 784, one row per hidden unit. Count weights plus biases layer by layer: 100,480 + 8,256 + 650 = 109,386. The first layer holds 92% of them, because every hidden unit connects to every pixel."

**Get it right:**

- Per layer: n⁽ˡ⁾·n⁽ˡ⁻¹⁾ weights + n⁽ˡ⁾ biases. 128·784 + 128 = 100,480; 64·128 + 64 = 8,256; 10·64 + 10 = 650.
- With examples as rows, Z = A W⁽ˡ⁾ᵀ + b⁽ˡ⁾ᵀ: (256 × 784)(784 × 128) = 256 × 128. Same weights, transposed product.

**If students ask:**

- **Q: In PyTorch, is the weight stored as 784 × 128?** A: No. nn.Linear(784, 128).weight has shape (128, 784), matching this slide, and computes x @ weight.T + bias.
- **Q: Is 109,386 parameters a lot?** A: Small by modern standards. It is roughly twice the number of training images. Language models in 2026 have billions to trillions of parameters.
- **Q: Do more parameters mean more overfitting?** A: Not automatically. Overfitting depends on data, training time and regularization as well. Very large networks can generalize well (see double descent, a later session).

### Slide 18 · Logits and softmax (0:38–0:41)

**Say:** "The last layer gives 10 raw scores, called logits. Softmax exponentiates and normalizes them: every probability is positive, they sum to 1, and the ranking is kept. Logits 2.0, 1.0, 0.1 become 0.66, 0.24, 0.10. One rule for code: the model returns raw logits, and the loss applies softmax itself."

**Get it right:**

- With K = 2 classes and logits (z, 0), softmax gives eᶻ / (eᶻ + 1) = σ(z). Softmax generalizes the sigmoid.
- Softmax is unchanged if you add the same constant to every logit. That is what makes the stable log-sum-exp trick possible (Appendix A3).
- The silent bug: softmax in the model and again inside the loss. Training runs without error but optimizes a different objective; for 10 classes the loss cannot go below about 1.46.

**If students ask:**

- **Q: Why exponentiate? Why not divide by the sum directly?** A: Logits can be negative, and exponentiating makes them positive. It also matches the log-linear (multinomial logistic) model, so the log-probabilities are simply logits minus a normalizer.
- **Q: Are softmax outputs true probabilities?** A: They sum to 1, but they are often over-confident. How well they match real frequencies is called calibration (Guo et al., 2017). Slide 44 shows the model becoming over-confident.
- **Q: What is "temperature"?** A: Dividing logits by T before softmax. T > 1 flattens the distribution, T < 1 sharpens it. It appears in calibration and in sampling from language models.

### Slide 19 · Quick check 2 (0:41–0:45)

**Run it:** Paper and pencil, vote, discuss, revote.

**Answer: B — 203,530.** Layer 1: 784·256 + 256 = 200,960. Layer 2: 256·10 + 10 = 2,570. Total 203,530.

**Common wrong answers:**

- A (203,264) forgets the biases.
- C (200,960) is layer 1 only; D (2,570) is layer 2 only.

## Part 2 · How a network learns · slides 20–30

Part 2 turns "wrong" into a number (the loss), shows why cross-entropy is the right number for classification, and shows how gradient descent lowers it, including why the step size has a ceiling.

### Slide 20 · Part 2 divider (0:45)

**Say:** "We can compute outputs. Now: how do 109,386 numbers get good values?"

### Slide 21 · A loss is one number with a slope (0:45–0:48)

**Say:** "Follow one image. The network gives this sneaker probability 0.70 for 'sneaker'. The loss is −log 0.70, about 0.36. Average over the training set to get J(θ). The data are fixed, so J depends only on the parameters. Why not train on accuracy directly? Look at the plot: as one weight changes, the error rate is flat and then jumps. Its slope is zero almost everywhere, so it gives no direction. The loss is smooth."

**Get it right:**

- −ln 0.70 = 0.3567. Natural logarithms throughout.
- The plot: one sigmoid neuron on 12 points, sweeping one weight with the bias fixed. The loss minimum and the best error rate need not be at the same weight; that is normal.
- Accuracy is what we report; the loss is what we optimize.

**If students ask:**

- **Q: Why average rather than sum?** A: The average keeps the loss on the same scale whatever the dataset or batch size, so a learning rate that works for one batch size is less sensitive to it. Summing just rescales the gradient.
- **Q: Can't we approximate accuracy smoothly?** A: Cross-entropy is in effect that smooth surrogate. Other surrogates exist (hinge loss in SVMs), but cross-entropy also gives probabilities and matches maximum likelihood.

### Slide 22 · Cross-entropy is the negative log-likelihood (0:48–0:51)

**Say:** "You fitted logistic regression by maximum likelihood. Same principle here: assume a probability model for y given x, then minimize the average negative log-likelihood. Gaussian noise gives squared error. A Bernoulli label gives binary cross-entropy. A categorical label gives multi-class cross-entropy: minus the log of the probability of the true class."

**Get it right:**

- Gaussian with fixed variance: −log p(y | x) = (y − ŷ)² / (2σ²) + const, so minimizing it is minimizing squared error.
- Maximizing a product of probabilities = minimizing a sum of negative logs. The log turns products into sums and avoids numerical underflow.
- Not every loss is a negative log-likelihood (hinge and ranking losses are not), but the common ones are.
- In code, all three losses take raw outputs: mse(ŷ, y), bce\_with\_logits(z, y), cross\_entropy(z, y). In PyTorch: nn.MSELoss, nn.BCEWithLogitsLoss, nn.CrossEntropyLoss.

**If students ask:**

- **Q: Why is it called "cross-entropy"?** A: For a true distribution q and a predicted distribution p, the cross-entropy is H(q, p) = −Σ q\_k log p\_k. With a one-hot label q, it reduces to −log p of the true class. It equals the entropy of q plus the KL divergence from q to p, so minimizing it minimizes the KL divergence.
- **Q: Binary or multi-class for two classes?** A: Equivalent. One sigmoid output with binary cross-entropy matches two softmax outputs with cross-entropy (slide 18: softmax over (z, 0) is σ(z)).
- **Q: What about multi-label problems (several classes true at once)?** A: Use one sigmoid and binary cross-entropy per class, not softmax.

### Slide 23 · Sanity check: first loss ≈ ln 10 (0:51–0:53)

**Say:** "Before you train, look at the very first loss. If the network starts with all ten logits close together, every class gets probability about 1/10, so the loss is about ln 10 = 2.30. Ours: 2.29 to 2.33 across three seeds. If you see 8 or 15, something is off before training has even started."

**Get it right:**

- Setup: He-normal initialization (standard deviation √(2/fan\_in)) in the two hidden layers, standard deviation 0.01 in the output layer, zero biases. Small output weights make the initial logits small.
- If the output layer also used He initialization, the initial logits would have a standard deviation of about 1.4 and the first loss would be roughly 3 (from a simulation), noticeably above ln 10.
- This is step one of Karpathy's (2019) "A recipe for training neural networks": verify the loss at initialization.

**If students ask:**

- **Q: What does PyTorch do by default?** A: nn.Linear initializes weights uniformly with a scale of about 1/√fan\_in (Kaiming-uniform with a = √5). On standardized inputs that also gives small logits and a first loss close to 2.30.
- **Q: What is the expected first loss with 100 classes?** A: ln 100 ≈ 4.61. It is homework.
- **Q: What if the classes are imbalanced?** A: Then a good initial output bias is the log of each class's frequency, and the expected first loss is the entropy of the class distribution, which is below ln K.

### Slide 24 · Cross-entropy vs squared error (0:53–0:56)

**Say:** "Take a sigmoid output and a true label y = 1. With cross-entropy, the gradient with respect to the logit is σ(z) − y: about −1 when the model is confidently wrong, a strong push. With squared error there is an extra factor σ(1 − σ), which goes to zero in the tails. Confidently wrong examples barely move. That's the red curve."

**Get it right:**

- Squared error here is L = ½(σ(z) − y)², so ∂L/∂z = (σ − y)·σ(1 − σ). At σ(z) = 0.0001, y = 1: cross-entropy gives −0.9999; squared error gives about −0.0001.
- The red curve peaks at σ = 1/3, that is z ≈ −0.69, where its value is 4/27 ≈ 0.148.
- Squared error is still the right loss for regression. The issue is squared error on a sigmoid or softmax output.

**If students ask:**

- **Q: Is cross-entropy convex?** A: In the logits, yes; for logistic regression it is convex in the weights too. For a network it is not convex in the weights, because the logits are a nonlinear function of them.
- **Q: Does anyone train classifiers with squared error?** A: Sometimes, and it can work with care, but cross-entropy on logits is the standard default.

### Slide 25 · Gradient descent (0:56–0:59)

**Say:** "The gradient points uphill, in the direction where J rises fastest. So step the opposite way, scaled by the learning rate η. A first-order Taylor expansion shows why it works: for a small enough step, J changes by −η‖∇J‖², which is negative."

**Get it right:**

- "Small enough" is the catch: the Taylor approximation ignores curvature. The next slide makes the limit exact on a quadratic.
- The gradient here has 109,386 components, one per parameter. Computing all of them efficiently is Part 3.

**If students ask:**

- **Q: Why the negative gradient and not some other downhill direction?** A: Among all directions of the same small length, the negative gradient gives the steepest first-order decrease (Cauchy–Schwarz). Other descent directions also work; Newton's method uses curvature to pick a better one, but is far too expensive for millions of parameters.
- **Q: How do I choose η?** A: Try values spaced by factors of about 3–10 (for example 1e−4 to 1e−1), watch the training loss for the first few hundred steps, and pick the largest value that decreases it smoothly. Adam with η = 10⁻³ is a common starting point for small networks.
- **Q: Does gradient descent find the global minimum?** A: For a convex loss with a suitable η, yes. For networks there is no guarantee, but in practice large networks usually reach low training loss.

### Slide 26 · How big a step? (0:59–1:02)

**Say:** "On a quadratic bowl, gradient descent decouples along the curvature directions: each coordinate is multiplied by (1 − ηλ) every step. It converges only if every factor has magnitude below 1, which means η < 2 / λmax. The sharpest direction sets the ceiling; the flattest direction then sets the speed."

**Get it right:**

- Here λ = 1 and 10, so η must be below 0.2. η = 0.02: θ₂ factor 0.8, θ₁ factor 0.98 (slow). η = 0.15: θ₂ factor −0.5 (oscillates but shrinks), θ₁ factor 0.85. η = 0.21: θ₂ factor −1.1, grows.
- The λ's are the eigenvalues of the Hessian. Their ratio, λmax / λmin, is the condition number: large means slow progress along flat directions.
- Real losses are not quadratic, but locally the same mechanism explains oscillation, blow-up and slow progress.

**If students ask:**

- **Q: Do real networks respect η < 2/λmax?** A: Interestingly, full-batch gradient descent on networks often drives the sharpness (λmax) up to about 2/η and then hovers there while the loss keeps falling, the "edge of stability" (Cohen et al., 2021). The quadratic rule is a guide, not a law, for networks.
- **Q: What if the loss is convex — can I use any η?** A: No. This bowl is convex and η = 0.21 still diverges. That is quick check 3.

### Slide 27 · Quick check 3 (1:02–1:05)

**Run it:** Vote, discuss, revote. Ask students to name the curvatures first.

**Answer: B — η < 2/50 = 0.04.** The curvatures are 4 and 50, and gradient descent converges from every start if and only if 0 < η < 2/λmax = 0.04.

**Common wrong answers:**

- A (0.5) uses the flat direction; θ₂ would diverge.
- C (0.02) is safe but not the full range; η = 0.03 also converges.
- D: convexity guarantees one minimum, not that every step size reaches it.

### Slide 28 · Standardize, then mini-batches (1:05–1:08)

**Say:** "Two practical points. First, standardize the inputs using statistics from the training split only. Second, computing the exact gradient over 55,000 images for every step is wasteful. A random mini-batch of 256 gives a noisy but unbiased estimate at a fraction of the cost. Shuffle every epoch. That is stochastic gradient descent."

**Get it right:**

- μ = 0.286, σ = 0.353 on pixel values scaled to \[0, 1\], computed on the 55,000 training images, one scalar for all pixels.
- ⌈55,000 / 256⌉ = 215 updates per epoch; the last batch has 216 images.
- If examples are sampled independently, the standard deviation of the mini-batch gradient estimate falls like 1/√B. Four times the batch halves the noise but costs four times as much per step.
- Epoch = one pass over the training data. Iteration (step) = one parameter update.

**If students ask:**

- **Q: Why not compute μ and σ on all the data?** A: Using validation or test data in preprocessing leaks information about them into training, so validation scores become optimistic. The rule: fit every preprocessing step on the training split only.
- **Q: Why shuffle?** A: Without shuffling, consecutive batches can be correlated (for example, sorted by class), which biases each step. Shuffling makes each mini-batch a fair sample.
- **Q: Is a bigger batch better?** A: It gives a less noisy gradient and uses hardware better, but fewer updates per epoch. Very large batches usually need a larger learning rate and warm-up (Goyal et al., 2017). Some noise can even help generalization, though this is debated.
- **Q: Why powers of two like 256?** A: Habit and slightly better hardware efficiency. Any batch size works.

### Slide 29 · Momentum and Adam (1:08–1:10)

**Say:** "Plain gradient descent uses one step size for every direction. In a narrow ravine it bounces across and creeps along. Momentum keeps a running sum of past gradients, so steady directions build speed and zig-zags cancel. Adam adds a separate step size for each parameter, from a running average of squared gradients. We train our running example with Adam, η = 10⁻³."

**Get it right:**

- The toy: a rotated quadratic with curvatures 1 and 25, each method's η tuned separately on a grid for 40 steps. Losses after 40 steps: gradient descent 5.9 × 10⁻³, momentum 1.5 × 10⁻³, Adam 3.6 × 10⁻⁴. Do not present this as a ranking that holds in general.
- Momentum as written (v ← βv + g, θ ← θ − ηv) is the PyTorch form. Adam: Kingma & Ba (2015); full equations in Appendix A6.

**If students ask:**

- **Q: Which optimizer should I use?** A: AdamW is the most common default, especially for Transformers. SGD with momentum remains strong for image CNNs. Newer optimizers such as Muon (Jordan et al., 2024) have been used for large language-model training (Liu et al., 2025), but they are not yet a general default. Tune the learning rate first; it matters more than the optimizer choice.
- **Q: What is the difference between Adam and AdamW?** A: How weight decay is applied. Adam with an L2 penalty adds it through the gradient, where Adam's per-parameter scaling distorts it. AdamW applies the decay directly to the weights (Loshchilov & Hutter, 2019). Session 2.
- **Q: Does Adam need a learning rate?** A: Yes. Adam rescales each parameter's step, but η still sets the overall step size.

### Slide 30 · Break (1:10–1:20)

**Say:** "Ten minutes. While you stretch, think about this: a network with 10⁸ parameters, one millisecond per forward pass. How long does one gradient take if you nudge each parameter separately?"

**Answer (give it after the break):** 10⁸ forward passes × 1 ms = 10⁵ s ≈ 28 hours, for a single gradient, and it grows linearly with the number of parameters. Backpropagation gets all of them for roughly the cost of one forward and one backward pass.

## Part 3 · Backpropagation and curves · slides 31–47

Part 3 is the hardest stretch: compute gradients by hand on a chain, generalize to branching and matrices, then use the full training loop to read real curves.

### Slide 31 · Part 3 divider (1:20)

**Say:** "Answer to the break question: about 28 hours for one gradient. We need something radically better. That is backpropagation."

### Slide 32 · Backpropagation: one backward sweep (1:20–1:23)

**Say:** "Finite differences need one extra forward pass per parameter. Backpropagation applies the chain rule from the loss backward. Look at the three gradients: they share the orange prefix, ∂J/∂c, then ∂J/∂c·∂c/∂b, and so on. Computed once from the output end and reused, all gradients cost about one backward pass."

**Get it right:**

- Backprop is reverse-mode automatic differentiation. History: Linnainmaa (1970) for reverse-mode differentiation; Werbos (1974) proposed it for networks; Rumelhart, Hinton & Williams (1986) made it widely known.
- Cost: for dense layers the backward pass takes roughly twice the arithmetic of the forward pass (a gradient for the weights and one for the inputs), so a training step costs about three forward passes. In theory, reverse mode costs a small constant multiple of evaluating the function, independent of the number of parameters (Griewank & Walther, 2008).
- Backprop computes gradients. An optimizer (SGD, Adam) uses them. Students often conflate the two.

**If students ask:**

- **Q: Why "reverse" mode? Is there a forward mode?** A: Forward mode pushes derivatives from inputs to outputs and costs about one pass per input parameter. Reverse mode costs about one pass per output. With one scalar loss and millions of parameters, reverse mode wins by a factor of millions.
- **Q: What does it cost in memory?** A: Reverse mode must keep the forward activations until the backward pass uses them. That is why training needs much more memory than inference. Gradient checkpointing trades memory for recomputation (Chen et al., 2016).
- **Q: Is this how the brain learns?** A: Unknown, and probably not exactly. Biologically plausible alternatives are an active research area; for engineering it does not matter.

### Slide 33 · Worked example: your turn (1:23–1:27)

**Run it:** Pairs, three minutes. Four decimals, natural logs.

**Expected:** z₁ = 1.1000, a₁ = 0.7503, z₂ = 0.3002, a₂ = 0.5745, L = 0.5543.

**Tip:** Circulate. Common slips: forgetting b₂, using log base 10, or computing σ(z) as 1/(1 + e^z). If a pair finishes early, ask them to predict the sign of ∂L/∂w₂ before the reveal (negative: increasing w₂ raises a₂ toward y = 1, lowering the loss).

### Slide 34 · Forward pass revealed (1:27–1:28)

**Say:** "Check your numbers. And keep them: the backward pass reuses every one."

**Get it right:**

- z₂ = 0.8 × 0.7503 − 0.3 = 0.3002; a₂ = σ(0.3002) = 0.5745; L = −ln 0.5745 = 0.5543.
- The caching point is real: training stores activations for the backward pass; inference does not.

### Slide 35 · Backward: the error signal δ (1:28–1:31)

**Say:** "Define δ as the derivative of the loss with respect to a pre-activation z. For sigmoid with binary cross-entropy, δ₂ = a₂ − y, here −0.4255. Send it back through the weight w₂ to get ∂L/∂a₁ = −0.3404, then multiply by the sigmoid's slope at z₁, 0.1874, to get δ₁ = −0.0638. One sigmoid layer made the error signal 6.7 times smaller."

**Get it right:**

- σ′(z₁) = a₁(1 − a₁) = 0.7503 × 0.2497 = 0.1874.
- |δ₁ / δ₂| = w₂·σ′(z₁) = 0.8 × 0.1874 = 0.150, that is 6.7 times smaller.
- The derivative of BCE with respect to a₂ is −y/a₂ + (1 − y)/(1 − a₂) = −1.7407; times σ′(z₂) = 0.2445 gives −0.4255 = a₂ − y. The simplification a₂ − y is why the sigmoid and BCE are always fused in code.
- The sigmoid's slope is at most 1/4, so each sigmoid layer multiplies the error by at most |w|/4. Over many layers this can vanish. It also depends on the weights: large weights can make gradients grow instead (exploding gradients).

**If students ask:**

- **Q: Why define δ on z rather than on a?** A: Because both parameter gradients come straight from δ: ∂L/∂w = δ × (input) and ∂L/∂b = δ. It also makes the recursion in BP2 clean.
- **Q: Does ReLU fix vanishing gradients?** A: It removes the activation's shrinking factor for active units (slope 1), which helps a lot. Gradients can still vanish or explode through the weights; initialization, normalization and residual connections handle that (Session 2).

### Slide 36 · Gradients and one step (1:31–1:34)

**Say:** "Each weight's gradient is the error at its output times the activation at its input; each bias gradient is the error itself. Take one step with η = 0.5 and recompute: the loss falls from 0.5543 to 0.4166."

**Get it right:**

- ∂L/∂w₂ = δ₂·a₁ = −0.4255 × 0.7503 = −0.3192; ∂L/∂b₂ = −0.4255; ∂L/∂w₁ = δ₁·x = −0.0638 × 2 = −0.1276; ∂L/∂b₁ = −0.0638.
- After the step: w₁ = 0.5638, b₁ = 0.1319, w₂ = 0.9596, b₂ = −0.0872. New loss 0.4166 (verified numerically).
- All gradients are negative, so every parameter increases. That makes sense: increasing any of them raises a₂ toward the target y = 1.
- A decrease is guaranteed only for a small enough η.

**If students ask:**

- **Q: Why is the bias gradient just δ?** A: z = wx + b, so ∂z/∂b = 1: the bias behaves like a weight on a constant input of 1.
- **Q: Would a larger η be better here?** A: For this one step a somewhat larger η would lower the loss more, but too large overshoots. Try η = 5 at home and watch what happens.

### Slide 37 · Quick check 4 (1:34–1:37)

**Run it:** Vote, discuss, revote. Ask: "Which path connects w₁ to L?"

**Answer: A — exactly 0.** w₁ affects L only through a₁ → z₂, and ∂L/∂a₁ = w₂·δ₂ = 0, so δ₁ = 0 and ∂L/∂w₁ = δ₁·x = 0.

**Follow-up worth saying:** ∂L/∂w₂ = δ₂·a₁ is not zero, so w₂ moves away from 0 at the next step and w₁ starts learning again. A zero on the only path blocks learning: the same mechanism as dead ReLUs and vanishing gradients. It is also why we never initialize all weights to zero (in a layer of identical units with identical weights, every unit gets the same gradient and they stay identical).

### Slide 38 · Branching: gradients add over paths (1:37–1:40)

**Say:** "Our worked example was a chain: each value fed exactly one thing. In a real network each hidden activation feeds every unit of the next layer. The multivariable chain rule says: add the contributions from every path. Here a feeds z\_A with weight 2 and z\_B with weight −1; the errors arriving are 0.3 and 0.5. So ∂L/∂a = 2 × 0.3 + (−1) × 0.5 = 0.1."

**Get it right:**

- This is the step most first-time students miss between the scalar example and the matrix equations. Spend the time.
- Forward: a is copied to both units. Backward: the two error signals come back and are added. In automatic differentiation, "copy forward" always means "add backward".

**If students ask:**

- **Q: Why add, not multiply?** A: A small change Δa changes z\_A by w\_A·Δa and z\_B by w\_B·Δa, and both changes affect L. To first order the effects on L add: ΔL ≈ (δ\_A w\_A + δ\_B w\_B)Δa.
- **Ask them:** What would ∂L/∂a be if w\_B were +1? (Answer: 1.1.)

### Slide 39 · The sum over paths is Wᵀδ (1:40–1:43)

**Say:** "Rows of W index the units above, columns the units below. Unit j's outgoing weights are column j. So the sum over paths, Σ\_k W\_kj δ\_k, is column j of W dotted with δ, and for all j at once that is Wᵀδ. Then gate by the activation's slope, elementwise. For ReLU the slope is 1 for active units and 0 for inactive ones: inactive units pass nothing back."

**Get it right:**

- Shapes: W is n\_above × n\_below, δ is n\_above × 1, so Wᵀδ is n\_below × 1, one number per unit below.
- Forward multiplies by W; backward multiplies by Wᵀ. The transpose is not a convention; it falls out of the sum over paths.

**If students ask:**

- **Q: Is Wᵀ the inverse of W?** A: No. Wᵀ does not undo W; it routes each error back along the same connections it came through, weighted by the same weights.
- **Q: What if the activation is softmax, which isn't elementwise?** A: Then the gating step needs the full Jacobian of softmax. Combined with cross-entropy it simplifies to p − y (slide 40, Appendix A3).

**Tip:** Draw a 3 × 4 matrix on the board and circle column 2 before revealing the slide.

### Slide 40 · The four equations of backpropagation (1:43–1:46)

**Say:** "These are exactly the steps you did by hand, written for vectors. BP1 starts the error at the output. BP2 sends it back one layer: through Wᵀ, then gated by g′. BP3 and BP4 turn error signals into parameter gradients. With softmax and cross-entropy, BP1 simplifies to p − y."

**Get it right:**

- δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ for one example. For the mean loss over a batch, average the per-example gradients.
- The labels BP1–BP4 follow Nielsen's "Neural Networks and Deep Learning" (2015).
- The right-hand column maps each equation onto the worked example; point to it.

**If students ask:**

- **Q: Do I need to memorize these?** A: Understand them well enough to derive them: BP2 and BP4 are the chain rule plus "copy forward, add backward". Frameworks implement them for you, but you will debug better if you know what they compute.

### Slide 41 · From one weight to a layer and a batch (1:46–1:49)

**Say:** "For a layer, BP4 is an outer product: the column of output errors times the row of input activations gives a matrix the same shape as W. For a batch stored as rows, stack errors into Δ and activations into A; then ΔᵀA sums the outer products over the batch, and dividing by B gives the mean-loss gradient."

**Get it right:**

- Example: δ = (0.4, −0.2), a = (2, 1, 3) gives \[\[0.8, 0.4, 1.2\], \[−0.4, −0.2, −0.6\]\].
- Running example, layer 1: Δ is 256 × 128 and A is 256 × 784, so ΔᵀA / B is 128 × 784, the shape of W⁽¹⁾. This is exactly what exp\_fmnist.py implements.

**If students ask:**

- **Q: Why does the batch gradient become one matrix multiplication?** A: Σᵢ δ⁽ⁱ⁾a⁽ⁱ⁾ᵀ is by definition ΔᵀA when examples are rows. One large matrix product runs far faster on GPUs than many small outer products; that is why mini-batches are hardware-friendly.

### Slide 42 · Autodiff and gradient checking (1:49–1:51)

**Say:** "Frameworks record the operations of the forward pass as a graph and apply these equations backward: automatic differentiation. When you write gradients by hand, check them against centred finite differences on a few parameters."

**Get it right:**

- Our NumPy MLP's hand-written backward pass agrees with centred differences to a relative error of about 3 × 10⁻⁹ (float64, ε = 10⁻⁵).
- Rules of thumb from the CS231n notes: relative error below about 10⁻⁷ is correct; around 10⁻⁴ is acceptable for functions with kinks such as ReLU; above about 10⁻² is almost certainly a bug.
- In PyTorch, gradients accumulate in .grad across backward() calls; optimizer.zero\_grad() clears them (by default it sets them to None since PyTorch 2.0).

**If students ask:**

- **Q: Why centred differences rather than (J(θ + ε) − J(θ))/ε?** A: The centred version has error of order ε², the one-sided version of order ε, so it is far more accurate for the same ε.
- **Q: Why not use finite differences for training, then?** A: Two forward passes per parameter: 10⁸ parameters means 2 × 10⁸ passes per step (slide 32). Fine for checking a few parameters, hopeless for training.
- **Q: Why does PyTorch accumulate gradients at all?** A: So you can sum gradients over several small batches before one update (gradient accumulation), useful when a big batch does not fit in memory.

### Slide 43 · The training loop (1:51–1:54)

**Say:** "Everything so far in one loop. For each mini-batch: clear old gradients, forward pass to logits, loss, backward pass, optimizer update. After each epoch, evaluate on the validation set with no gradient tracking and no updates, and keep the checkpoint with the lowest validation loss."

**Get it right:**

- This is exactly what exp\_fmnist.py does: Adam, η = 10⁻³, B = 256, 20 epochs.
- Training mode vs evaluation mode matters once the model has dropout or batch normalization (Session 2).
- Validation data never change the weights directly; they choose the checkpoint and hyperparameters. The test set is used once, at the end.

**If students ask:**

- **Q: Isn't choosing the checkpoint on validation data a kind of overfitting?** A: Slightly: each choice made on validation data makes the validation score a bit optimistic. That is exactly why a separate test set is kept and touched only once.
- **Q: How many epochs should I train?** A: Until validation loss stops improving for a while (patience), then keep the best checkpoint. For our model that is about 9–15 epochs depending on the seed.

### Slide 44 · Our model's real training curves (1:54–1:57)

**Say:** "Seed 0. Training loss falls steadily. Validation loss falls, flattens, and creeps back up after epoch 9. From then on the model is fitting details of the training set that don't carry over. Validation accuracy still inches up, because accuracy only counts whether the top class is right, while the loss also punishes over-confident mistakes. We keep the checkpoint with the lowest validation loss: early stopping."

**Get it right:**

- Seed 0, epoch 20: training loss 0.145, validation loss 0.315; the lowest validation loss is 0.298 at epoch 9. Validation accuracy goes from 89.0% at epoch 9 to 89.6% at epoch 20; training accuracy reaches 94.8%.
- The best epoch varies by seed: 9, 12 and 15 for seeds 0, 1 and 2. One run can mislead.
- The y-axis of the accuracy chart starts at 80%.

**If students ask:**

- **Q: Should I select on validation loss or validation accuracy?** A: Whichever matches your goal. If you need well-calibrated probabilities, loss; if you only report accuracy, accuracy. They can disagree by a few epochs, as here.
- **Q: Why is training loss below validation loss from the start?** A: Training loss is measured on data the model is being fitted to. A gap is normal; a gap that keeps widening is the warning sign.
- **Q: Isn't early stopping a hack?** A: It is a standard regularizer. For linear models with gradient descent it behaves much like L2 regularization (Goodfellow et al., 2016, §7.8).

### Slide 45 · What the network learned (1:57–2:00)

**Say:** "Left: softmax regression, the linear model, learns exactly one template per class; you can see trousers, a sandal, a bag. The MLP's first layer learns 128 features shared by all classes; here are 24 of them. Right: learning those features is worth about 4 points of test accuracy over the linear model, 84.5% to 88.5%. Honest caveat: flattening throws away the image's 2-D layout. Convolutional networks keep it and do better; that's a later session."

**Get it right:**

- Our numbers: official 10,000 test images, checkpoint with the lowest validation loss, mean of 3 seeds. Softmax regression 84.5% (84.45–84.50%); MLP 88.5% (88.36–88.70%).
- The k-NN (85.4%) and random forest (87.3%) bars are the best settings reported in the Fashion-MNIST paper's benchmark (Xiao et al., 2017), under a different protocol. Treat them as reference points, not a controlled comparison.
- Weight images: purple and orange mark positive and negative weights; each image is scaled to its own range.

**If students ask:**

- **Q: Why do the MLP features look so noisy?** A: An MLP has no built-in notion that neighbouring pixels belong together, and it stopped training after about 9–15 epochs. Regularization or more data smooths them; convolutional layers build in locality from the start.
- **Q: What is the best accuracy anyone gets on Fashion-MNIST?** A: Simple two-layer CNNs reach about 90–92%, and strong modern CNNs report around 94–96%. Part of the remaining error comes from genuinely ambiguous classes such as shirt vs T-shirt vs pullover.
- **Q: Only 4 points better: was the network worth it?** A: On this data with this architecture, modestly. That is an honest result and the reason the next architectures (CNNs) exist. Always fit a strong simple baseline first.

### Slide 46 · Read the curves: diagnosis (2:00–2:04)

**Say:** "Three real runs of the same code. Left: only two hidden units. Training and validation loss are both high and close together: underfitting, so make the model bigger or train longer. Middle: our full network on just 1,000 images. Training loss goes almost to zero while validation loss turns up early: overfitting. Get more data, regularize, stop early. Right: the running example, a small gap with both low."

**Get it right:**

- 784–2–10, epoch 20: training loss 1.01, validation loss 1.04, test accuracy 61.7%, still slowly falling.
- 1,000 images: training loss 0.007 at epoch 100; validation loss lowest (0.55) at epoch 24, then 0.83; training accuracy 100%, validation accuracy 81%.
- Before any diagnosis: can the model drive the loss to almost zero on one small batch? If not, suspect a bug first.
- Underfitting and overfitting are about the gap and the levels together, not either alone.

**If students ask:**

- **Q: Can a model underfit and overfit at the same time?** A: Different parts of it, in a sense, and the labels are a simplification. Use them as guides to the next experiment, not verdicts.
- **Q: Doesn't more parameters always mean more overfitting?** A: No. In the "double descent" regime, very large models can generalize better again (Belkin et al., 2019; Nakkiran et al., 2020). Session 2 discusses it.
- **Q: What remedies come next?** A: Session 2: initialization, normalization, residual connections, weight decay, dropout and data augmentation.

### Slide 47 · Quick check 5 (2:04–2:08)

**Run it:** Remind them of the sanity check on slide 23 before the revote if the first vote splits.

**Answer: C — look for a bug.** A loss stuck at 2.30 = ln 10 with 10% accuracy means uniform predictions: nothing has been learned. That is not ordinary underfitting; even the linear softmax model reaches a validation loss near 0.40 on this data. Test whether the model can drive the loss to almost zero on one small batch.

**Likely causes to name:** the optimizer was built on a different model's parameters; the learning rate is zero or tiny; gradients are cleared between backward() and the update step; inputs and labels are misaligned or shuffled separately; a ReLU layer is dead everywhere.

**Common wrong answers:** B (more layers) treats a bug as a capacity problem; A and D assume something has been fitted, but nothing has.

## Wrap-up and appendix · slides 48–59

The wrap-up closes the loop on the four outcomes; the appendix slides are for questions and self-study, not for presenting.

### Slide 48 · Today in three sentences (2:08–2:15, then questions to 2:30)

**Say:** Read the three sentences slowly, one per part. Then: "Homework is on the right. Before you leave, one minute on the exit ticket: your muddiest point today, and one sentence on why g must be nonlinear."

**Homework answers (for you, not the slide):**

- Worked example with x = 1.0: z₁ = 0.6, a₁ = 0.6457, z₂ = 0.2165, a₂ = 0.5539, L = 0.5907. After one step with η = 0.5 the loss falls to 0.4555.
- 784–256–128–10 has 200,960 + 32,896 + 1,290 = 235,146 parameters. First loss for 100 classes ≈ ln 100 = 4.61.
- BP2 for two units feeding two units: ∂L/∂a\_j = W\_1j δ₁ + W\_2j δ₂ for j = 1, 2, i.e. Wᵀδ, then multiply elementwise by g′(z).

**Tip:** Collect the exit tickets and open the next session with the three most common muddiest points. Expect backprop's Wᵀ and the train/validation distinction to top the list.

### Slide 49 · Appendix divider

Not presented. Point students to it as reference material.

### Slides 50–51 · A1 · Quick-check answers

For self-study after class. The full answers and the reasoning behind the common wrong choices are under each quick check in this guide (slides 11, 19, 27, 37, 47).

### Slide 52 · A2 · Notation

**Get it right:** Square brackets index layers (W⁽ˡ⁾); round brackets index examples (x⁽ⁱ⁾). B is the mini-batch size. Derivations use column vectors (z = Wa + b); code stores examples as rows (Z = AWᵀ + bᵀ).

**If students ask:**

- **Q: Why do textbooks use different notation?** A: There is no standard. Goodfellow et al. (2016), Nielsen (2015) and Prince (2023) differ slightly. Learn to translate by checking shapes.

### Slide 53 · A3 · Why the output error is p − y

**Say (if asked):** "Write the loss as −z\_y + log Σⱼ e^{zⱼ}. Differentiating the first term gives −1 for the true class and 0 otherwise; differentiating the log-sum-exp gives softmax, p\_k. So the gradient is p − y with y one-hot."

**Get it right:**

- Log-sum-exp is computed stably by subtracting the largest logit c first: log Σ e^{zⱼ} = c + log Σ e^{zⱼ − c}. Every exponent is then ≤ 0, so nothing overflows.
- This is why frameworks fuse softmax and cross-entropy and expect logits.

### Slide 54 · A4 · Universal approximation, stated honestly

**Get it right:**

- Statement: for any continuous f on a compact set K and any ε > 0, some network with one hidden layer of finitely many units with a non-polynomial activation is within ε of f everywhere on K (Cybenko, 1989, for sigmoids; Hornik, 1991; Leshno et al., 1993 for the general condition).
- Four caveats on the slide: only continuous functions on bounded regions; the width can be exponential in the input dimension in the worst case; existence does not mean gradient descent finds it; fitting is not generalizing.
- The figure is hand-built piecewise-linear interpolation with ReLU units, not a trained network.

**If students ask:**

- **Q: Then why do we need anything beyond one hidden layer?** A: Efficiency. Depth can represent some functions with exponentially fewer units (Telgarsky, 2016), and deep networks train well in practice.
- **Q: Does the identity network of quick check 1 count?** A: No. Its activation is a polynomial (degree 1), which the theorem excludes. That is the theoretical side of slide 10.

### Slide 55 · A5 · Gradient checking

**Get it right:**

- Use float64, ε around 10⁻⁵, a deterministic loss (dropout off), and check a random subset of parameters: each check costs two forward passes.
- With ReLU, a nudge can cross a kink at z = 0 and spoil one check; relative errors around 10⁻⁴ are then acceptable.

### Slide 56 · A6 · Adam in full, and AdamW

**Get it right:**

- Defaults β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ (Kingma & Ba, 2015). They are common starting points, not optimal for every model.
- Bias correction divides by (1 − βᵗ) because m and v start at zero and would otherwise be too small in early steps.
- AdamW decouples weight decay from the adaptive step (Loshchilov & Hutter, 2019). In PyTorch, torch.optim.AdamW has a default weight decay of 0.01; Adam with weight\_decay > 0 applies L2 through the gradient unless decoupled\_weight\_decay=True is set.

**If students ask:**

- **Q: What does ε do?** A: It prevents division by zero when v̂ is tiny. It rarely matters, but some large models tune it.

### Slide 57 · A7 · History, 1943 to today

**Get it right:**

- 1943 McCulloch & Pitts; 1958 Rosenblatt; 1969 Minsky & Papert; 1970 Linnainmaa; 1986 Rumelhart, Hinton & Williams; 1998 LeNet-5 (LeCun et al.); 2012 AlexNet; 2015 ResNet (He et al., published at CVPR 2016); 2017 Transformer (Vaswani et al.); 2024 Nobel Prizes.
- 2024 Nobel Prize in Physics: John Hopfield and Geoffrey Hinton, for foundational work enabling machine learning with artificial neural networks. 2024 Nobel Prize in Chemistry: David Baker (computational protein design) and Demis Hassabis and John Jumper (protein structure prediction, AlphaFold).
- The slide's message: backprop existed by 1986; deep learning took off around 2012 when large labelled datasets (ImageNet, 2009), GPUs and training techniques came together.

### Slide 58 · A8 · When not to use a neural network

**Get it right:**

- On medium-sized tabular datasets, gradient-boosted trees (XGBoost, LightGBM, CatBoost) often outperform deep networks (Grinsztajn, Oyallon & Varoquaux, 2022).
- Pretrained tabular foundation models changed this picture. TabPFN (Hollmann et al., Nature, 2025) beat tuned baselines on datasets up to 10,000 samples; Prior Labs released TabPFN-3 in May 2026 (up to about 1 million rows) and TabPFN-3.5 in September 2026. Their strongest comparisons with boosted trees are reported by the vendor, so check them on your own data.

**If students ask:**

- **Q: So should I always try a network?** A: Fit a strong simple baseline first: logistic regression or boosted trees. It tells you what score is easy and whether the extra complexity pays off. Our linear baseline here was 84.5%.

### Slide 59 · References

Point students to Goodfellow, Bengio & Courville (2016), Prince (2023, free online) and Nielsen (2015, free online) for today's material. The scripts that produced every number are in the course materials.

## Common misconceptions to watch for

These are the beliefs students most often bring or form in this session; correct them when you hear them, using the slide listed.

| Misconception | Correction | Slide |
| --- | --- | --- |
| "A sigmoid at the output makes the boundary curved." | A monotone output function keeps the 0.5 boundary where z = 0, a hyperplane if z is linear in x. | 10, 11 |
| "More layers always mean more power." | Only with a nonlinearity between them. Linear layers collapse into one. | 10 |
| "ReLU is linear, so it can't bend anything." | It is piecewise linear. Many kinks together form a patchwork that can follow any continuous curve. | 12, 14 |
| "Backpropagation is the learning algorithm." | Backprop computes gradients. The optimizer (SGD, Adam) uses them to update the weights. | 32 |
| "The gradient flows back through W." | It flows back through Wᵀ: the sum over paths, Σ\_k W\_kj δ\_k, is a product with the transpose. | 38, 39 |
| "When a value feeds two units, take one path." | Add the contributions of all paths (multivariable chain rule). | 38 |
| "Softmax should be the last layer of the model." | The model returns logits; the loss applies softmax. Doing both is a silent bug. | 18 |
| "Accuracy is the thing we minimize." | Accuracy has zero slope almost everywhere. We minimize a smooth loss and report accuracy. | 21 |
| "A convex loss is safe with any learning rate." | On a convex quadratic, η ≥ 2/λmax still diverges. | 26, 27 |
| "Validation data are just more training data." | They never update the weights; they select checkpoints and hyperparameters. The test set is used once. | 43 |
| "Rising validation loss means accuracy is falling." | Accuracy can still inch up while the loss rises from over-confident mistakes. | 44 |
| "Loss stuck at ln K means the model is too small." | It means nothing is being learned at all; look for a bug first. | 47 |
| "Initialize all weights to zero to be safe." | Identical units then get identical gradients and stay identical (and with ReLU, zero gradients). Break the symmetry with random initialization. | 37 |

## Running the room

The five quick checks and two paired exercises carry about a third of the learning; protect their time and run them the same way every time.

**Quick checks (peer instruction):**

1. Read the question aloud. Give 30 seconds of silent thinking, then a simultaneous vote (cards, fingers or a poll), so nobody copies a neighbour.
2. If 30–70% are correct, have students convince a neighbour for 60 seconds, then revote. Below 30%, re-teach the idea briefly before discussion. Above 70%, explain quickly and move on (Crouch & Mazur, 2001).
3. Ask one student who changed their answer to explain why. Wrong answers explained aloud are useful; thank the student.
4. Reveal the answer and spend 30 seconds on the most popular wrong option.

**Paired exercises (slides 13 and 33):** announce the time limit, circulate, and look for the common slips listed under each slide. Reveal when about two thirds of pairs are done.

**Inclusion:**

- Say the colour code aloud as well as showing it; about 1 in 12 men and 1 in 200 women have some colour-vision deficiency. Plots also use shape (filled vs hollow) for classes, so name both.
- Read every equation in words the first time it appears ("z equals w transpose x plus b").
- Use cold-calling sparingly and only after paired discussion, so students answer for their pair rather than alone.
- Repeat student questions into the microphone before answering, for the room and any recording.

**Pacing checkpoints:** be on slide 20 by 0:45, on the break slide by 1:10, on slide 38 by 1:37, and on slide 46 by 2:00. If you miss one by more than five minutes, use the cut list in Before class.

## Sources

All Fashion-MNIST numbers come from the course scripts (exp\_fmnist.py, figs.py), run with fixed seeds. External facts were checked against these pages:

- [Fashion-MNIST paper, Xiao, Rasul & Vollgraf (2017)](https://arxiv.org/abs/1708.07747) — dataset and k-NN / random-forest benchmark figures.
- [Fashion-MNIST README and benchmark table](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md) — user-submitted CNN results.
- [Prior Labs TabPFN changelog](https://docs.priorlabs.ai/changelog.md) — TabPFN-3 (May 2026) and TabPFN-3.5 (September 2026).
- [PyTorch Adam source](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py) and [AdamW source](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py) — decoupled\_weight\_decay option and the AdamW default weight decay of 0.01.
- [PyTorch 2.14.0 release announcement](https://dev-discuss.pytorch.org/t/pytorch-2-14-0-general-availability/3431) — current release as of October 2026.

Standard references cited in the notes: Goodfellow, Bengio & Courville (2016), Deep Learning; Nielsen (2015), Neural Networks and Deep Learning; Prince (2023), Understanding Deep Learning; Griewank & Walther (2008), Evaluating Derivatives; Karpathy (2019), A recipe for training neural networks; Crouch & Mazur (2001), Am. J. Phys. 69; Cybenko (1989); Hornik (1991); Leshno et al. (1993); Telgarsky (2016); Montúfar et al. (2014); Kingma & Ba (2015); Loshchilov & Hutter (2019); Cohen et al. (2021); Goyal et al. (2017); Guo et al. (2017); Chen et al. (2016); Belkin et al. (2019); Nakkiran et al. (2020); Grinsztajn, Oyallon & Varoquaux (2022); Hollmann et al. (2025).
