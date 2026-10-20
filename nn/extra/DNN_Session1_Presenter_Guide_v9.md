# Presenter Guide — Deep Neural Networks, Session 1
## "From one neuron to networks that train" · 150 minutes · undergraduate engineering

**Deck:** `DNN_Session1_150min_v9.pptx` (75 slides: 52 lecture slides, Appendix A, Appendix B, references)
**Written for:** a TA or colleague presenting this deck for the first time
**Fact-checked:** October 2026

---

## How to use this guide

Each lecture slide has four parts:

- **Time and goal.** When the slide starts and ends on the session clock, and the one idea students must leave with.
- **What to say.** A full script you can read or paraphrase. Text in *[italics and brackets]* is a stage direction, not something to say aloud.
- **Clarifications for you.** Background, derivations, and traps, so you can answer confidently. You do not need to say all of this.
- **Questions students may ask.** Likely questions with short, correct answers.

The appendix slides (A1–A10, B1–B8) are reference material, not part of the 150 minutes. They get a shorter treatment at the end, for when a student asks a question that one of them answers.

---

## Before the session

1. **Do the worked example yourself on paper** (slides 31–34), with a calculator, before class. It's the part of the lecture where you most need to be fluent.
2. **Tell students to bring a calculator** (a phone is fine). They need one for slides 31–34.
3. **Decide how students will vote in quick checks**: four fingers, A–D cards, or a clicker/poll tool. There are six quick checks, each about four minutes.
4. **Reserve whiteboard space** for the XOR inequalities (slide 8) and the worked example (slides 31–34).
5. **Bring index cards** (or set up a one-question online form) for the exit ticket on slide 52.
6. **Optional:** open `playground.tensorflow.org` in a browser tab in case a student asks what a ReLU network "looks like" (Appendix A3).
7. **Ask the course lead where the experiment scripts are posted.** Slide 3 tells students that every curve for the running example comes from a script they can rerun, and someone will ask.

### The colour code (used on every slide)

| Colour | Meaning |
|---|---|
| Teal | Forward signal and data (input → output) |
| Orange | Gradients flowing backward |
| Red | Warnings, bugs, the "bad" run in a chart |
| Navy / grey | Neutral |
| Teal and gold in scatter plots | The two classes |

### The peer-instruction protocol (all six quick checks)

1. Read the question aloud. Give about 30 seconds of silent thinking.
2. **Everyone votes at once** (so nobody copies). Look at the spread of votes, but don't reveal the answer.
3. If the votes are split (roughly 30–70% correct), say: *"Find someone who voted differently and convince them. One minute."*
4. Revote. Then ask one student who changed their vote to explain why.
5. Explain the answer, including why each wrong option is tempting.
6. If almost everyone is right on the first vote, skip the discussion and explain briefly. If almost everyone is wrong, skip the discussion and teach the idea again.

The answers are also in Appendix A1 (slides 54–55), so students can review them later. Don't show those slides in class.

---

## Session clock

| Clock | Slides | Segment |
|---|---|---|
| 0:00–0:05 | 1–3 | Opening |
| 0:05–0:36 | 4–17 | **Part 1 · Foundations** (Quick check 1 at 0:21) |
| 0:36–1:04 | 18–28 | **Part 2 · Learning**: losses, data, gradient descent, optimizers (Quick checks 2 and 3) |
| 1:04–1:14 | 29 | **Break** |
| 1:14–1:43 | 30–39 | **Part 2 (cont.)**: backpropagation, worked example, training loop (Quick check 4) |
| 1:43–2:19 | 40–51 | **Part 3 · Train and generalize** (Quick checks 5 and 6) |
| 2:19–2:30 | 52 | Wrap-up, exit ticket, open questions |

### If you are running late

Protect the quick checks and the worked example; they are where students actually learn. If you need time, trim these places, in this order:

1. **Slide 27 (optimizers):** state the one-line takeaway and move on. Adam is covered in full in A6.
2. **Slide 37 (autodiff):** say "the framework runs BP1–BP4 for you and reproduces our numbers; clear gradients every step," and move on.
3. **Slide 9 (3-D XOR):** show the table and the equation only.
4. **Slides 47–48 (augmentation, transfer learning):** one minute each. Give the rule and one example.
5. **Quick checks:** if the first vote is above about 70% correct (the same threshold as the protocol), skip the discussion round. This saves about two minutes each.

Never skip slide 24 (the double-softmax bug). Many students will make that mistake in their first project.

---

# OPENING

## Slide 1 — Title: "Deep neural networks: from one neuron to networks that train"
**Time:** 0:00–0:02 · **Goal:** set expectations and teach the colour code.

### What to say
"Good morning, everyone, and welcome to the first session on deep neural networks. Over the next two and a half hours we will go from a single artificial neuron, which you have essentially already met as logistic regression, all the way to multi-layer networks that we train on real images. We will finish with what it takes to make those networks train reliably and generalize to new data.

The session has three parts. First, what a network computes. Second, how it learns. Third, how to make it train and generalize.

One thing will help you all session: the colours on the slides mean something. *[Point to the bottom right of the slide.]* Teal is the forward signal, data flowing from input to output. Orange is gradients, information flowing backward while the network learns. Red is a warning: a bug, a failure, or the bad run on a chart. Look at the network on this slide: the teal nodes compute forward, and the orange path is the route a gradient takes backward.

There is a ten-minute break a little after the one-hour mark."

### Clarifications for you
- Today is all concepts, pen-and-paper, and **framework-neutral pseudocode**. Nobody needs a GPU or a framework installed.
- The audience is assumed to know classical machine learning (logistic regression, train/validation/test splits), calculus, and basic probability, but not deep learning.

### Questions students may ask
- **"Do we need a GPU or a framework for today?"** No. Today is about ideas. The code on the slides is pseudocode, and the homework uses NumPy only.
- **"Is deep learning the same thing as AI?"** No. Deep learning is one family of machine-learning methods: models built from many layers of learned transformations. Most current AI systems (image recognition, speech, language models) are built on it, but AI as a field is broader.
- **"Will we build language models like ChatGPT?"** Not today. Today's foundations (neurons, losses, gradient descent, backpropagation, initialization, residual connections, regularization) are exactly what those systems are built on. Specific architectures such as Transformers come in later sessions.

---

## Slide 2 — "By the end of today you can…"
**Time:** 0:02–0:03 · **Goal:** the six objectives are skills students will *do*, not topics.

### What to say
"Here are six things you will be able to do by the end of today, two for each part. Notice the verbs: explain, compute, derive, run, choose, diagnose. We will check every one of these in class, mostly with quick votes.

Part 1: explain why linear models fail on XOR and how a hidden layer fixes it, and compute an MLP's forward pass, its shapes and its parameter count.
Part 2: derive a loss from a noise model and explain what momentum and Adam fix, and run backpropagation by hand.
Part 3: choose an initialization and explain residual connections, and diagnose underfitting and overfitting and pick remedies in a sensible order.

*[Point to the bottom box.]* I'm assuming you are comfortable with matrix–vector products, the chain rule, logistic regression, train/validation/test splits, and basic Python and NumPy. If the chain rule or matrix shapes feel rusty, tell me at the break. Those two carry most of today."

### Clarifications for you
The layout runs column by column: objectives 1–2 sit under Part 1, 3–4 under Part 2, and 5–6 under Part 3. Here is where each objective is checked:

| Objective | Checked on |
|---|---|
| 1 XOR and hidden layers | Slides 8–14, Quick check 1 (slide 12) |
| 2 Forward pass, shapes, parameters | W[1] prediction (slide 16), parameter count (slide 17) |
| 3 Loss from noise model; momentum/Adam | Derivation on slide 20, Quick checks 2 and 3 |
| 4 Backprop by hand | Slides 31–34, Quick check 4 |
| 5 Initialization; residuals | Quick check 5, slide 45 |
| 6 Diagnosis and remedies | Slide 46, Quick check 6 |

### Questions students may ask
- **"Will all of this be on the exam?"** Defer to the course syllabus. A safe answer: "The six objectives are the core skills; the appendix slides are for reference."
- **"How good does my linear algebra need to be?"** Good enough to multiply a matrix by a vector and to check that shapes agree. We do shape checks explicitly on slides 15–17.

---

## Slide 3 — "Today's plan, and one example we will follow all session"
**Time:** 0:03–0:05 · **Goal:** show the clock and introduce the running example.

### What to say
"The bar at the top is our clock. Part 1 runs until about 36 minutes in. Part 2 starts before the break and continues after it. Part 3 starts at about 1:43, and we wrap up at 2:30.

To tie everything together, we will follow one example all session. *[Point to the images.]* This is Fashion-MNIST: small 28-by-28 grey-scale pictures of clothing, in 10 classes such as T-shirt, trouser, sneaker and bag. Our model is a network with layer sizes 784, 128, 64 and 10. We build it in Part 1, train it in Part 2, and diagnose and improve it in Part 3. Every curve you see for this model comes from an actual training run, made with scripts you can rerun yourselves.

A word on how we work: about every 20–30 minutes there is a quick check, a multiple-choice question. Everyone votes at the same time. If the room disagrees, you convince your neighbour, and then we vote again. This works better than me just talking, so please really commit to an answer."

### Clarifications for you
- 784 = 28 × 28. The image is flattened into a vector of 784 pixel values.
- **Why Fashion-MNIST rather than MNIST:** it has the same format and size (70,000 images, 28 × 28, 10 classes), but it is harder. On MNIST, simple models already score so high that differences between methods are hard to see.

### Questions students may ask
- **"Why not ordinary MNIST?"** MNIST is so easy that even simple models do very well, so it can't show the differences we care about. Fashion-MNIST is a drop-in replacement that is harder.
- **"What does 784–128–64–10 mean?"** The widths of the layers: 784 inputs (one per pixel), then 128 hidden units, then 64 hidden units, then 10 outputs (one score per class).
- **"Can I train this on my laptop?"** Yes. It has about 109,000 parameters, so it trains on a CPU in minutes.

---

# PART 1 · FOUNDATIONS (0:05–0:36)

## Slide 4 — Part 1 title: "Foundations: what a network computes"
**Time:** 0:05 (about 30 seconds) · **Goal:** frame Part 1 with four questions.

### What to say
"Part 1 answers four questions. Why can't a linear model bend its decision boundary? What does one neuron compute, and why can't it solve a problem called XOR? Why must a nonlinearity sit between layers? And how does a multi-layer network turn inputs into outputs? In Part 1 nothing learns yet. We only ask what a network *can compute*. Learning is Part 2."

### Clarifications for you
Keep this short. It is a signpost, not content.

---

## Slide 5 — "On raw inputs, linear models draw only straight boundaries"
**Time:** 0:05–0:07 · **Goal:** pose the problem before the mechanism. Logistic regression's boundary is a straight line, and some data need a curve.

### What to say
"Let's start with a model you already know: logistic regression. *[Point to the formula.]* It computes a weighted sum z = wᵀx + b and passes it through a sigmoid to get a probability ŷ.

Here are 300 points from a dataset called 'two moons', two interlocking half-circles. On the left, logistic regression draws a straight line and reaches 86% training accuracy. On the right, a network with two hidden layers bends its boundary around the moons and gets every training point right.

Why can't logistic regression do the same? *[Pause and let a student try.]* It predicts class 1 when ŷ is above 0.5, which happens exactly when z > 0. So its boundary is the set where wᵀx + b = 0, and that is a straight line: a hyperplane in more dimensions. No choice of w or b can bend a line. The rest of Part 1 is about how we keep what's good about this model and still bend the boundary."

### Clarifications for you
- **"On raw inputs"** is deliberate. With engineered features such as x₁², x₁x₂ or polynomial terms, logistic regression can draw curved boundaries in the original space (slide 9 shows exactly this). The limitation is "linear in the features you give it."
- 100% *training* accuracy shows capacity, not generalization. If a student raises overfitting, praise the instinct and say Part 3 is about exactly that.
- Linear models remain valuable: their training problem is convex, their weights are interpretable, and they need little data.

### Questions students may ask
- **"Isn't 86% pretty good?"** For this toy dataset it's the best *any* straight line can do, and that's the point: the limit comes from the model's form, not from the training. On harder data, that limit costs more.
- **"Isn't the network overfitting at 100%?"** Possibly. With this little noise, a smooth curve can fit every point legitimately. Training accuracy alone can't tell you. We need validation data, which comes in Part 3.
- **"Couldn't I use a kernel SVM or add polynomial features?"** Yes, and that works here. Slides 9–10 explain why neural networks learn such features instead of having them hand-designed.

---

## Slide 6 — "One neuron: a weighted sum and a threshold"
**Time:** 0:07–0:10 · **Goal:** one neuron answers one question: which side of one hyperplane is x on? Know the geometry of w and b.

### What to say
"Let's look closely at one artificial neuron. *[Point to the diagram on the left.]* It takes inputs x₁ to xₙ, multiplies each by a weight, adds a bias b, and gets a score z = wᵀx + b. The simplest neuron then applies a step: output 1 if z > 0, otherwise 0. This idea goes back to the threshold unit of McCulloch and Pitts in 1943. Rosenblatt's perceptron in 1958 added real-valued weights that are *learned* from data.

Now the geometry. *[Point to the right-hand figure.]* The neuron's boundary is the line where z = 0.
One: the weight vector w is perpendicular to that line, and it points toward increasing z. Here's why: take any two points x and x′ on the line. Both give z = 0, so wᵀx − wᵀx′ = 0, which means wᵀ(x − x′) = 0. So w is perpendicular to every direction along the line.
Two: the bias shifts the line without rotating it. The distance from the origin to the line is |b| / ‖w‖. More generally, (wᵀx + b) / ‖w‖ is the *signed distance* from any point x to the line.
Three: the neuron fires on one side only, the shaded side.

So one neuron answers exactly one question: which side of one line is this point on?"

### Clarifications for you
- **Wording trap:** the step function *is* nonlinear, yet the boundary is still a straight line. Avoid calling this neuron "linear." The precise statement: its *decision boundary* is linear because the step is applied to a linear score. This distinction pays off on slides 8, 11 and 12.
- The proof that w is perpendicular to the boundary works in any dimension (there the boundary is a hyperplane).
- **Signed distance derivation:** write x = x₀ + t·w/‖w‖ with x₀ on the boundary. Then wᵀx + b = t‖w‖, so t = (wᵀx + b)/‖w‖.
- **The biology analogy is useful for about one minute.** Real neurons spike over time, dendrites compute nonlinearly, and there is no evidence that brains run backpropagation.

### Questions students may ask
- **"What if b = 0?"** The line passes through the origin.
- **"If I double w and b, does the boundary change?"** No. z = 0 defines the same set. Only the *steepness* of a smooth activation changes, which is the next slide.
- **"Is this how real neurons work?"** Only loosely. It's a useful abstraction, not a biological model.
- **"Why divide by ‖w‖?"** Because z grows when you scale w, but the geometric distance doesn't. Dividing by ‖w‖ turns the score into a true distance.

---

## Slide 7 — "Smooth activations give gradients: the sigmoid neuron"
**Time:** 0:10–0:12 · **Goal:** a step has zero slope, so gradient descent can't train it. The sigmoid fixes this, and the result is logistic regression. Remember σ′ ≤ ¼.

### What to say
"The step neuron has a problem. *[Point to the red box.]* Its derivative is zero everywhere it is defined. Nudge a weight slightly and the output usually doesn't change at all. A stack of step neurons therefore gives no gradient, and gradient descent can't learn.

The fix is to smooth the step. Suppose the log-odds of class 1 is linear: log(p / (1 − p)) = z. Solve for p and you get p = 1 / (1 + e⁻ᶻ). That's the sigmoid, σ(z). *[Point to the teal curve.]* We call z the **logit**, and we'll use that word all session.

The sigmoid's derivative has a neat form: σ′(z) = σ(z)(1 − σ(z)). *[Point to the orange bump.]* It peaks at z = 0, where it equals exactly one quarter. Please remember that ¼. It comes back twice today.

And if you scale w and b by a factor k, the boundary stays put but the sigmoid switches k times more sharply. A sigmoid neuron with this loss is exactly logistic regression. You already know this model."

### Clarifications for you
- **Algebra for the board:** p/(1−p) = eᶻ ⇒ p = eᶻ/(1+eᶻ) = 1/(1+e⁻ᶻ).
- **Derivative:** σ′ = e⁻ᶻ/(1+e⁻ᶻ)² = σ(1−σ). The product σ(1−σ) is largest at σ = ½, where it equals ¼.
- **Perceptron history:** Rosenblatt's learning rule trains a *single* step neuron without gradients. It converges on linearly separable data (Novikoff, 1962). It does not extend to hidden layers, because a hidden step unit gives no signal about how to change it.
- **Sharpness:** as k → ∞ the sigmoid approaches the step. Small weights mean gentle switching, which is the intuition behind weight decay in Part 3.

### Questions students may ask
- **"The step's derivative at 0 is undefined. Isn't that a problem?"** It is undefined at a single point and zero everywhere else. The real problem is the zero everywhere else: it gives no learning signal.
- **"Couldn't we just pretend the step has a slope?"** That trick exists. It's called the *straight-through estimator* and is used in some quantized networks. It's an approximation, outside today's scope.
- **"Why the sigmoid and not another S-shaped curve?"** It's what you get if you assume the log-odds are linear, which gives it a probabilistic meaning. Other smooth curves (tanh, for example) also work, as we'll see.
- **"Why do you keep emphasizing ¼?"** Because backpropagation multiplies one σ′ factor per sigmoid layer. A factor of at most ¼ per layer shrinks gradients fast (slides 13 and 33).

---

## Slide 8 — "XOR: no straight line separates the classes"
**Time:** 0:12–0:15 · **Goal:** a proof by contradiction that one neuron can't compute XOR.

### What to say
"Here's the simplest problem a single neuron can't solve: XOR, 'exclusive or'. The output is 1 when exactly one input is 1. *[Point to the plot.]* The filled teal points, (0,1) and (1,0), are class 1. The hollow gold points, (0,0) and (1,1), are class 0.

Your task, twenty seconds: try to draw one straight line with both teal points on one side and both gold points on the other. *[Wait.]* The dashed lines are some attempts. Each one gets a point wrong.

We can prove it's impossible. The neuron predicts class 1 when w₁x₁ + w₂x₂ + b > 0. Each point gives one inequality. *[Point to the four rows.]*
(0,0) must be class 0, so b ≤ 0.
(0,1) must be class 1, so w₂ + b > 0.
(1,0) must be class 1, so w₁ + b > 0.
(1,1) must be class 0, so w₁ + w₂ + b ≤ 0.

Add rows 2 and 3: w₁ + w₂ + 2b > 0, so w₁ + w₂ + b > −b. Since b ≤ 0, −b is at least zero. So w₁ + w₂ + b > 0. That contradicts row 4. No weights and bias can work. Minsky and Papert made limitations like this precise for single-layer perceptrons in 1969.

Notice the step was nonlinear all along, and it didn't help. One unit still draws one line. The fix isn't a different activation on one unit. The fix is more units."

### Clarifications for you
- The same proof applies to a **sigmoid** neuron, because "ŷ > 0.5" is equivalent to "z > 0". It applies to any neuron whose activation is monotone and is thresholded.
- **Historical nuance:** Minsky and Papert's 1969 book is often blamed for a decline in neural-network funding in the 1970s. Historians debate how much it caused. It was known then that multi-layer networks could represent XOR. What was missing was a practical way to *train* them.

### Questions students may ask
- **"Does this also hold for a sigmoid neuron?"** Yes. Classifying with ŷ > 0.5 is the same as z > 0, so the same inequalities apply.
- **"What if the activation weren't monotone, say g(z) = z²?"** Good thinking! Then one unit *can* do it: with z = x₁ − x₂, z² equals XOR exactly on the four corners. The proof assumes you threshold a linear score. In practice we use monotone activations and get flexibility from *more units*, which is more general.
- **"Why does XOR matter? It's tiny."** It's the smallest example of a pattern that depends on an *interaction* between inputs. Real data are full of interactions, and one linear boundary can't capture them.

---

## Slide 9 — "A better representation makes XOR easy"
**Time:** 0:15–0:17 · **Goal:** the central idea of the field: a hard problem in one representation can be easy in another.

### What to say
"Here's a trick. Keep x₁ and x₂, and add one new feature: their product, x₁x₂. *[Point to the table.]* Only the corner (1,1) gets a 1 in the new column.

Now plot the four points in three dimensions. *[Point to the 3-D figure.]* A flat plane separates the two classes. In fact, the linear formula y = x₁ + x₂ − 2x₁x₂ reproduces XOR exactly. Check it: (1,1) gives 1 + 1 − 2 = 0.

The data didn't change. The representation did. That's the central idea of this whole field: a problem that is hard in one representation can be easy in another."

### Clarifications for you
- **Back in the original 2-D plane,** thresholding this model at ½ gives the boundary x₁ + x₂ − 2x₁x₂ = ½. That simplifies to (x₁ − ½)(x₂ − ½) = 0: two crossing lines, x₁ = ½ and x₂ = ½. This curved, non-linear boundary is why slide 5 says "on *raw* inputs."
- **Kernel methods** (for example, an SVM with a polynomial or RBF kernel) apply this idea implicitly, without ever writing the new features out.

### Questions students may ask
- **"How did you know to pick x₁x₂?"** I knew the answer in advance, and that's exactly the weakness the next slide addresses: for real problems we don't know which features to build.
- **"Is this the kernel trick?"** It's the same idea: map to a space where the classes become linearly separable. A kernel does it implicitly.
- **"Why not just add all products of features, always?"** The number of features explodes. That's on the next slide.

---

## Slide 10 — "The neural-network idea: learn the features too"
**Time:** 0:17–0:19 · **Goal:** a neural network learns the feature map and the linear model together, by gradient descent on one loss.

### What to say
"Who designs features like x₁x₂? For real problems, nobody can. Which function of the 150,528 numbers in a 224-by-224 colour image means 'cat'? And brute force doesn't scale: just the degree-3 products of 1,000 inputs are about 170 million features.

So here's the neural-network idea. *[Point to the diagram.]* Give the feature extractor its own parameters: h = g(W[1]x + b[1]), where g is some nonlinear function applied to each element. Put a linear model on top: ŷ = w[2]ᵀh + b[2]. Then train *both* together, by gradient descent on one loss.

Three consequences. *[Point to the three boxes.]*
**Task-driven:** the features are shaped by the loss we care about, not by an expert's guess.
**Reusable:** features like edge and texture detectors, learned on one vision task, transfer to others. We'll use that in Part 3.
**Composable:** layer 2 builds on layer 1's features, layer 3 on layer 2's. That hierarchy is what 'deep' means."

### Clarifications for you
- **Notation:** the superscripts [1] and [2] index *layers*. They are not powers.
- For classification, the "linear model on top" produces logits. The sigmoid or softmax is applied in the loss (slides 15 and 20).
- **The arithmetic:** 224 × 224 × 3 = 150,528. Degree-3 monomials of 1,000 variables number about 1,000³/6 ≈ 1.7 × 10⁸.

### Questions students may ask
- **"Are the learned features interpretable?"** Sometimes. Early layers of image networks often learn edge- and colour-detectors. Deeper features are often hard to interpret, and interpreting them is an active research area.
- **"What is g?"** Coming in two slides. First we'll see why g must be there at all.
- **"Is this just logistic regression on learned features?"** For a binary classifier with one hidden layer, yes, exactly. That's a good way to think about it.

---

## Slide 11 — "Without a nonlinearity, depth collapses to one layer"
**Time:** 0:19–0:21 · **Goal:** stacking linear layers buys nothing. The nonlinearity g between layers is essential.

### What to say
"Why do we need g? Suppose we leave it out and stack three linear layers. *[Point to the top diagram.]* Substitute each layer into the next. *[Point to the equation.]* We get W[3]W[2]W[1] times x, plus some combined bias. The product of three matrices is just one matrix, W′. So three linear layers equal one linear layer. So do ten, or a hundred. They still can't solve XOR.

So depth only pays off when there is a nonlinearity g between the layers."

### Clarifications for you
- **The combined bias**, for completeness: b′ = W[3]W[2]b[1] + W[3]b[2] + b[3].
- **Rank:** rank(W[3]W[2]W[1]) ≤ the smallest rank among the factors. A narrow middle layer can therefore only *restrict* the overall linear map, never enrich it.
- Deep *linear* networks are still studied in research, because their *training dynamics* are non-trivial even though the function they compute is linear. Mention this only if asked.

### Questions students may ask
- **"Isn't ReLU linear for positive inputs?"** Piecewise, yes. But the kink at zero makes the whole function nonlinear, and that's enough. Stacked ReLU layers do not collapse.
- **"Is a network with no activations ever useful?"** Occasionally. A linear bottleneck network is a low-rank factorization, and a linear autoencoder learns the same subspace as PCA. But it can't represent anything a single linear map can't.
- **"Does a sigmoid on the output count as a nonlinearity?"** That's exactly the next question. Let's vote.

---

## Slide 12 — Quick check 1 (identity hidden layers, sigmoid output)
**Time:** 0:21–0:25 · **Goal:** make students apply slide 11. An output nonlinearity alone doesn't bend the boundary.

### What to say
"Quick check. A 5-layer network uses g(z) = z, the identity, in every hidden layer, and a sigmoid on the output. Which decision boundaries can it draw? A: any smooth curve. B: piecewise-linear curves. C: only a straight line, a hyperplane. D: it depends on the layer widths.

Think silently for thirty seconds. Then everyone votes at once on my signal." *[Run the peer-instruction protocol. Don't reveal the answer before the revote.]*

**After the revote:** "The answer is **C**. The hidden layers are all linear, so they compose into one affine map, z = W′x + b′. That's slide 11. Then the sigmoid is applied. But the sigmoid is monotone: ŷ > 0.5 exactly when z > 0. So the boundary is still the hyperplane W′x + b′ = 0, no matter how wide or deep the network is."

### Clarifications for you
- **Why each distractor is tempting:**
  - **A** catches students who think the curved sigmoid curves the *boundary*. It changes the probabilities near the boundary, not the boundary's shape.
  - **B** is the correct answer for *ReLU* hidden layers (Appendix A3). It's a useful misconception to surface now.
  - **D** confuses capacity in general with this specific case. Width doesn't matter when everything is linear.
- **Connection to theory:** the universal approximation theorem requires a *non-polynomial* activation (Appendix A4). The identity is a polynomial, so this network is not universal.

### Questions students may ask
- **"But the sigmoid IS nonlinear. Why is the boundary still straight?"** Because a boundary is where the output crosses a threshold. A monotone function crosses 0.5 at exactly one value of z, so the boundary is z = 0. The sigmoid changes how confident the model is, not where it draws the line.
- **"What if the output activation weren't monotone?"** Then the boundary could be several parallel hyperplanes. But we never do that in practice. We get flexibility from hidden nonlinearities instead.

---

## Slide 13 — "Sigmoid and tanh saturate; ReLU does not for z > 0"
**Time:** 0:25–0:28 · **Goal:** choose g by its *derivative*, because backprop multiplies one derivative per layer. Know dying ReLU.

### What to say
"So which g should we use? *[Point to the left chart.]* The three classic choices: sigmoid, tanh and ReLU. ReLU is max(0, z): zero for negative inputs, the identity for positive ones.

*[Point to the right chart.]* What matters for learning is the *derivative*, because backpropagation multiplies one derivative per layer, as we'll see in Part 2.

**Saturation kills gradients.** The sigmoid's slope is at most 0.25. At z = 6 it's about 0.0025, essentially flat. Stack ten sigmoid layers and the activation slopes alone contribute at most 0.25¹⁰, about one in a million. tanh is better centred, with a slope up to 1 at zero, but its tails flatten out too.

**ReLU has slope exactly 1 whenever it's active.** So the activation itself doesn't shrink the gradient. The weights still can, which is Part 3. ReLU is also cheap to compute and gives sparse activations.

**The price is dying ReLU.** If a unit's input is negative for *every* example, it always outputs zero and gets zero gradient through the activation. It can stay switched off for good. That typically happens after a too-large learning rate."

### Clarifications for you
- **"From σ′ alone":** the weights also enter the backward product. Large weights could compensate, but large weights push sigmoids deeper into saturation, so it doesn't work in practice.
- **Useful identities:** σ′(6) = σ(6)(1 − σ(6)) ≈ 0.0025. 0.25¹⁰ ≈ 9.5 × 10⁻⁷. tanh(z) = 2σ(2z) − 1.
- **ReLU at exactly z = 0:** the derivative is undefined there. Frameworks use 0 by convention, and it essentially never matters.
- **History:** ReLU, together with better initialization, GPUs and more data, made deep networks practical around 2010–2012 (Glorot et al., 2011; AlexNet, 2012).
- **Remedies for dying ReLU:** lower the learning rate, use good initialization, or use Leaky ReLU (a small slope for z < 0). Smooth variants such as GELU and SiLU are standard in Transformers (Appendix B8).

### Questions students may ask
- **"If slope 1 is good, why not use the identity everywhere?"** Because then the network collapses to one linear layer (slide 11). ReLU keeps slope 1 where it's active *and* has a kink that makes it nonlinear.
- **"How can ReLU, made of straight pieces, produce curved boundaries?"** It produces *piecewise-linear* boundaries with many pieces, which can follow curves closely (Appendix A3).
- **"Which activation should I use?"** ReLU is the default for MLPs and CNNs. GELU or SiLU for Transformers. Sigmoid or tanh mainly for gates (in LSTMs) and special outputs, not for hidden layers.
- **"Is tanh still used?"** Yes, inside recurrent networks (LSTM/GRU gates use both sigmoid and tanh) and in some output layers that need a range of (−1, 1).

---

## Slide 14 — "Two ReLU units solve XOR by remapping the space"
**Time:** 0:28–0:30 · **Goal:** see a hidden layer change the representation so that a straight line works.

### What to say
"Now we can solve XOR. *[Point to the network.]* There are two hidden ReLU units, and all input weights are 1. h₁ = ReLU(x₁ + x₂) behaves like OR. h₂ = ReLU(x₁ + x₂ − 1) behaves like AND. The output is y = h₁ − 2h₂.

Trace the four corners with me. *[Write on the board.]*
(0,0): h₁ = 0, h₂ = 0, so y = 0.
(0,1): h₁ = 1, h₂ = ReLU(0) = 0, so y = 1.
(1,0): same, y = 1.
(1,1): h₁ = 2, h₂ = 1, so y = 2 − 2 = 0.
That's XOR.

Now the picture that matters. *[Point to the right plot, 'hidden space'.]* After the hidden layer, (0,1) and (1,0) land on the *same* point, (1,0). The corner (1,1) moves to (2,1). In this hidden space, one straight line separates the classes. The output neuron is still just a line. The hidden layer moved the points so that a line works."

### Clarifications for you
- y is a score. Classify with y > ½, which corresponds to the line h₁ − 2h₂ = ½ drawn in hidden space.
- **Biases:** 0 for h₁ and −1 for h₂.
- **This solution isn't unique.** For example, ReLU(x₁ − x₂) + ReLU(x₂ − x₁) = |x₁ − x₂| also equals XOR on the corners. Gradient descent would find *some* solution, not necessarily this one.

### Questions students may ask
- **"Would gradient descent find these exact weights?"** Not necessarily. Many solutions exist. With only two hidden units, training from a random start can sometimes get stuck, which is one reason we use more units than the minimum.
- **"Could one hidden ReLU unit do it?"** Not with a plain linear output on top: the output would be a monotone function of a single linear score, so the boundary is again one line. With a direct connection from the inputs to the output it *can*: y = x₁ + x₂ − 2·ReLU(x₁ + x₂ − 1). Skip connections like this return on slide 45.
- **"Why −2?"** h₁ counts (1,1) as 2. Subtracting twice the AND unit cancels it back to 0.

---

## Slide 15 — "The forward pass repeats one rule per layer"
**Time:** 0:30–0:32 · **Goal:** the general forward pass, and the shape rule "W has one row per unit."

### What to say
"The XOR network had one hidden layer. Any depth works the same way: two lines per layer. *[Point to the algorithm.]* Start with a[0] = x. For each layer l: compute the pre-activation z[l] = W[l]a[l−1] + b[l], a weighted sum plus bias. Then apply the activation, a[l] = g(z[l]).

For a classifier, the last layer has *no* activation. Its output is a vector of raw scores called **logits**, one per class. The probabilities are computed inside the loss, which is Part 2.

*[Point to the shape check.]* Shapes: W[l] has shape (units in this layer, units in the previous layer). Row i of W holds unit i's weights, one per input. So W times a column vector of length n[l−1] gives a column of length n[l]. If shapes ever confuse you, write down what *one row* means, then count."

### Clarifications for you
- The input is "layer 0" and has no parameters, so the network in the figure has **L = 3** layers.
- **For regression,** the last layer is also the identity, and its output *is* the prediction ŷ.
- **Notation:** square brackets index layers, and round brackets index training examples (Appendix A2).

### Questions students may ask
- **"Why isn't the input counted as a layer?"** It has no weights. Counting layers by weight matrices is the common convention.
- **"Why no activation on the output?"** The loss applies sigmoid or softmax internally, in a numerically stable way. Adding your own causes a nasty bug (slides 23–24).
- **"What's the difference between z and a?"** z is the weighted sum before the nonlinearity. a is the value after it, and it's what the next layer sees.
- **"Can different layers use different activations?"** Yes. Each layer has its own g[l]. Typically every hidden layer uses ReLU and the output uses none.

---

## Slide 16 — "Running example: a 784–128–64–10 MLP"
**Time:** 0:32–0:34 · **Goal:** read the shapes of a real network. The batch dimension never changes. Predict W[1]'s shape.

### What to say
"Here's our running example. Fashion-MNIST has 70,000 grey-scale images of 28 by 28 pixels, in 10 classes. We use 55,000 for training, hold out 5,000 for validation, and keep 10,000 for testing.

Read the shapes top to bottom for a *batch* of 128 images. A batch is just a group of examples processed together. *[Point down the stack.]* The input is (128, 1, 28, 28): 128 images, 1 colour channel, 28 by 28 pixels. Flatten gives (128, 784). The first linear layer with ReLU gives (128, 128). The second gives (128, 64). The final linear layer gives (128, 10): ten logits per image.

Notice that the first number, 128, never changes. If it ever does in your code, a reshape is wrong.

*[Point to 'To beat'.]* What should we aim for? Classical methods you know, on the same data: k-nearest-neighbours gets 85.3% test accuracy, and a random forest gets 87.5%. Keep those in mind.

Now a prediction. *[Point to the bottom-left box.]* What's the shape of W[1], the first weight matrix? Hands up for (784, 128)… and for (128, 784)… Answer on the next slide."

### Clarifications for you
- The original dataset is 60,000 training plus 10,000 test images. The 5,000 validation images are carved out of the 60,000.
- The "1" in (128, 1, 28, 28) is the channel dimension: 1 for grey, 3 for RGB.
- **The most common wrong answer is (784, 128).** Don't correct it yet; slide 17 does.
- **Payoff later (slide 50):** the full-data MLP reaches about 88.3%, roughly a tie with the random forest. Be honest about it when it comes up: on small grey images, a plain MLP isn't much better than a forest. Convolutional networks (later sessions) are where deep learning clearly pulls ahead.

### Questions students may ask
- **"Doesn't flattening throw away the image structure?"** Yes, and that's a real weakness. The MLP doesn't know which pixels are neighbours. Convolutional networks fix this and come later in the course.
- **"Why 128 and 64 units?"** They're hyperparameters, chosen by convention and a bit of trial. Shrinking widths toward the output is a common pattern, not a rule.
- **"Why a batch of 128?"** It's a common size that uses hardware efficiently while giving many updates per epoch (slide 25).
- **"Why does a neural network only roughly tie a random forest?"** Because this MLP ignores the spatial structure of images, and the dataset is small and simple. Architecture matters, and baselines keep us honest.

---

## Slide 17 — "In pseudocode: a stack of layers that returns logits"
**Time:** 0:34–0:36 · **Goal:** reveal W[1] = (128, 784), connect the maths to batched code, and count parameters.

### What to say
"The same network as pseudocode: flatten, linear 784 to 128 with ReLU, linear 128 to 64 with ReLU, and linear 64 to 10. That's raw logits: *no softmax at the end*. Remember that.

The answer to the prediction: **W[1] is (128, 784)**, one row per unit. In code, a batch stores examples as *rows*, so the layer computes X Wᵀ + b. X is (128, 784), Wᵀ is (784, 128), and the result is (128, 128). It's the same W as in the maths, applied to a whole batch at once.

Now count the parameters. Each layer has (outputs × inputs) weights plus one bias per output.
Layer 1: 784 × 128 + 128 = 100,480.
Layer 2: 128 × 64 + 64 = 8,256.
Layer 3: 64 × 10 + 10 = 650.
Total: 109,386. *[Point to the bar chart.]* 92% of them are in the first layer, because every one of 784 pixels connects to every one of 128 units. Keep that in mind. It's a big part of why image models use convolutions instead."

### Clarifications for you
- **Shape algebra:** in the maths, z = W a with a as a column vector. In code, Z = X Wᵀ + b, where the bias of shape (128,) broadcasts across the 128 rows. These are the same computation, transposed.
- **PyTorch, if asked:** `nn.Linear(784, 128).weight.shape` is `(128, 784)`, and the layer computes `x @ W.T + b`.
- **Memory:** 109,386 float32 parameters × 4 bytes ≈ 0.44 MB.

### Questions students may ask
- **"Why store W as (out, in) if the code transposes it anyway?"** It matches the maths (one row per unit) and is a common library convention. Either layout works if you're consistent.
- **"Is 109k parameters a lot?"** It's tiny by modern standards. Image models have millions, and large language models have billions.
- **"Why would most parameters in the first layer be a problem?"** It makes the model data-hungry and blind to spatial structure. A convolution reuses a small set of weights at every position, which cuts parameters dramatically.

---

# PART 2 · LEARNING (0:36–1:43, break at 1:04)

## Slide 18 — Part 2 title: "How a network learns"
**Time:** 0:36 (about 30 seconds) · **Goal:** frame Part 2.

### What to say
"Part 1 built a network that computes. Part 2 makes it learn. Four questions: how do we turn 'wrong' into a number? Which way, and how far, should the weights move? How do we get every gradient at once, efficiently? And what do real training curves tell us? We'll take a break partway through."

---

## Slide 19 — "A loss is one number, and it has a slope"
**Time:** 0:36–0:39 · **Goal:** a loss turns 'wrong' into one smooth number. We train on the loss, not on accuracy, because accuracy is flat.

### What to say
"*[Point along the top row.]* A sneaker image goes into our network, with all 109,386 parameters, which we'll call θ. The network says P(sneaker) = 0.70. The loss compares that with the true label and returns one number: −log 0.70, which is 0.36. That's cross-entropy. The loss is near zero when the model is confidently right, and it grows without limit as the probability of the true class approaches zero.

*[Point to the box on the right.]* Average the per-example loss over the training set and you get J(θ), the cost. The data are fixed, so J is a function of the parameters alone. Training means finding the θ that minimizes J.

Here's a fair question: we care about accuracy, so why not maximize accuracy directly? *[Point to the chart.]* This chart is computed for one sigmoid neuron on ten points, sweeping one weight w. The red staircase is the error rate. It's flat between jumps: its slope is zero almost everywhere, so it gives no direction to move. The teal cross-entropy changes smoothly, and its slope tells us which way is downhill.

So: we train on the loss, and we report accuracy."

### Clarifications for you
- **Logs are natural logs all session.** −ln 0.70 = 0.357.
- **Vocabulary:** L(ŷ, y) is the per-example loss. J(θ) = (1/m) Σ L is the average, also called the cost or objective. People loosely call both "the loss."
- **Reading the chart:**
  - The error rate steps down as w increases, because more points get classified correctly.
  - The cross-entropy keeps falling even where the error rate is flat, because correct predictions become *more confident*.
  - Past its minimum, the cross-entropy rises again, because one misclassified point becomes more confidently wrong. Accuracy can't see that, but the loss can.

### Questions students may ask
- **"Why the negative log specifically?"** It comes from maximum likelihood, which is the next slide. It also turns a product of probabilities over the dataset into a sum.
- **"Can the loss reach zero?"** Only in the limit, as the true-class probability approaches 1. In practice, never exactly.
- **"Is a lower loss always higher accuracy?"** Usually, but not always. Slide 39 shows the validation loss rising while validation accuracy still improves.
- **"Why does the teal curve go back up on the right?"** One point is on the wrong side. Increasing w makes the model more confident about it, and confident mistakes are punished hard.

---

## Slide 20 — "Most losses are negative log-likelihoods"
**Time:** 0:39–0:42 · **Goal:** derive squared error from Gaussian noise. See BCE and cross-entropy as the same idea for discrete labels.

### What to say
"Where do losses come from? Usually from a probability model of the targets. The recipe: assume how y is generated given x, write down its likelihood, and minimize the negative log-likelihood (NLL).

Regression, first row. Assume y = f(x) + Gaussian noise with variance σ². Over to you, twenty seconds: what is −log p(y | x)? *[Wait, then write:]* The Gaussian density is (1/√(2πσ²)) · exp(−(y − ŷ)²/2σ²). Take minus the log: (y − ŷ)²/2σ², plus ½ log(2πσ²). With σ fixed, that last term is a constant. So maximizing the likelihood means minimizing the **squared error**. MSE isn't arbitrary: it's what Gaussian noise implies.

Binary classification: the label is a coin flip with probability ŷ = σ(z). The NLL is **binary cross-entropy**.
Multi-class: the label is one of K categories with probabilities ŷ = softmax(z). The NLL is **cross-entropy**, which is −log of the probability given to the true class c.

*[Point to the 'In pseudocode' column.]* In code, the classification losses take the raw logits z and apply sigmoid or softmax internally."

### Clarifications for you
- **Why a sum:** with i.i.d. examples the likelihood is a product over examples, and its log is a sum, which gives J = (1/m) Σ L.
- **The ½:** the familiar ½(y − ŷ)² is the σ = 1 case. Library "mse" functions usually compute the plain mean of (ŷ − y)², without the ½. A constant factor only rescales the gradient, which is the same as changing the learning rate.
- **When MSE is wrong:**
  - Heavy-tailed noise → Huber loss. Laplace noise gives absolute error (L1).
  - Counts → Poisson NLL.
  - Noise whose variance depends on x → predict σ as well and use the Gaussian NLL.
- **BCE from Bernoulli:** p(y | x) = ŷʸ (1 − ŷ)¹⁻ʸ. Taking −log gives −[y log ŷ + (1 − y) log(1 − ŷ)].

### Questions students may ask
- **"Is cross-entropy related to KL divergence?"** Yes. CE(y, ŷ) = H(y) + KL(y ‖ ŷ). With one-hot labels H(y) = 0, so minimizing cross-entropy is minimizing KL divergence.
- **"What if the noise isn't Gaussian?"** Then choose the loss that matches it (see above). The NLL recipe tells you which.
- **"Why do the losses want logits, not probabilities?"** For numerical stability and correctness. Slide 24 shows what happens otherwise.
- **"Why is binary cross-entropy a special case of cross-entropy?"** With K = 2 classes, softmax over two logits equals a sigmoid of their difference, and the two formulas agree.

---

## Slide 21 — "Softmax, and the first-loss sanity check"
**Time:** 0:42–0:44 · **Goal:** compute softmax, and learn to check that the first loss is about ln K.

### What to say
"Softmax turns K logits into probabilities. Exponentiate each one, then divide by the sum. *[Point to the bar charts.]* Logits 2.0, 1.0 and 0.1 become 0.66, 0.24 and 0.10. They're all positive, they sum to 1, and the largest logit gets the largest probability.

Cross-entropy is then minus the log of the true class's probability. *[Point to the bottom line.]* If the true class gets 0.9, the loss is 0.105. If it gets 0.5, the loss is 0.69. If it gets 0.01, the loss is 4.61.

Here's a sanity check you should run on every new model. *[Point to the teal box.]* Before training, an untrained model should be roughly uniform: about 1/K for each class. So the very first loss should be about −log(1/K) = ln K. For 10 classes that's ln 10 = 2.30. Our running example started at 2.63, close enough.

*[Point to the red box.]* If your first loss is 10 or 15, something is wrong before you've trained at all. Check your input scaling, your initialization and your labels."

### Clarifications for you
- **The arithmetic:** e² = 7.389, e¹ = 2.718, e⁰·¹ = 1.105. The sum is 11.21, which gives 0.659, 0.242 and 0.099.
- **Shift invariance:** softmax(z + c) = softmax(z) for any constant c. Only *differences* between logits matter. This is what makes the stable implementation on slide 24 possible.
- **Why 2.63 rather than 2.30:** a standard random initialization gives logits of order 1, not exactly zero, so the first loss is usually somewhat above ln K. Values 10× too high mean the logits differ hugely, usually because the inputs weren't standardized or the weights are too large.
- A first loss far *below* ln K before any training is usually suspicious: for example, the loss was computed after an update, or labels leak into the inputs. **One legitimate exception:** if the output biases are deliberately initialized to the log class frequencies (a common trick for imbalanced data), the first loss equals the entropy of the class distribution, which is below ln K.
- For two classes, softmax([z₁, z₂])₁ = σ(z₁ − z₂).

### Questions students may ask
- **"Why exponentiate? Why not just divide by the sum?"** Logits can be negative, and exponentials are always positive. Exponentials also make the model shift-invariant, and they pair with the log in cross-entropy to give a very clean gradient (ŷ − y, slide 36).
- **"Why is it called softmax?"** It's a smooth, differentiable version of 'pick the maximum': the largest logit gets most of the probability.
- **"What if my first loss is slightly below ln K?"** That can happen by chance. Worry only about large deviations.

---

## Slide 22 — "Cross-entropy keeps learning when MSE stalls"
**Time:** 0:44–0:46 · **Goal:** for classification, cross-entropy's gradient is largest when the model is confidently wrong. MSE's is not.

### What to say
"Why not just use squared error for classification? Look at the gradient with respect to the logit z, for a sigmoid output whose true label is 1.

*[Point to the red box.]* With squared error, the gradient is (σ(z) − y) times σ′(z). Suppose the model is confidently wrong and predicts 0.0001. The error is almost −1, which is large. But σ′ there is about 0.0001, so the gradient is about −0.0001. Learning stalls exactly when the model is most wrong.

*[Point to the teal box.]* With cross-entropy, the σ′ factor cancels, and the gradient is simply σ(z) − y: about −1 in the same case. Full strength when it's needed most.

*[Point to the chart.]* On the left, where the model is confidently wrong, the teal cross-entropy gradient is near 1 and the orange MSE gradient is near zero. Use cross-entropy for classification."

### Clarifications for you
- **The slide assumes L = ½(σ − y)².** A library MSE without the ½ doubles the gradient to about −0.0002, which is still negligible.
- **Why σ′ cancels:** ∂/∂z of −log σ(z) is −(1 − σ) = σ − 1, the case y = 1. The same cancellation for softmax plus cross-entropy gives ŷ − y. Students will see this again in the worked example (δ₂ = a₂ − y, slide 33).
- **Two more reasons to prefer cross-entropy:**
  - Labels aren't "real values plus Gaussian noise," so MSE uses the wrong likelihood.
  - For a single neuron, cross-entropy∘sigmoid is convex in the weights, while MSE∘sigmoid is not.

### Questions students may ask
- **"Is squared error ever used for classification?"** Occasionally, applied to raw linear outputs (no sigmoid), where the stall problem doesn't arise. Research has found it competitive in some settings. The specific failure here is MSE *after a saturating sigmoid*.
- **"Cross-entropy goes to infinity as p → 0. Doesn't the gradient explode?"** No. The *loss* grows, but its gradient with respect to the logit is σ − y, which is bounded between −1 and 1.
- **"Why is the cross-entropy gradient small on the right?"** There the model is already confidently right, so there's little left to learn. That's the behaviour we want.

---

## Slide 23 — Quick check 2 (softmax into cross_entropy)
**Time:** 0:46–0:50 · **Goal:** students must know that the cross-entropy function applies softmax itself.

### What to say
"Quick check. Training loss stalls near 1.6 while validation accuracy is about 85%. The model ends in a softmax layer, and that output goes into cross_entropy(·, y). Which change fixes it? A: remove the final softmax and pass raw logits. B: raise the learning rate. C: train for more epochs. D: convert the labels to one-hot." *[Run the protocol. Don't advance until the revote: the next slide gives the answer away.]*

**After the revote:** "The answer is **A**. The cross_entropy function already applies a (log-)softmax to its input. With a softmax at the end of the model, softmax is applied *twice*. No error is raised, but the loss can't get below a floor, which is what you see. Remove the model's softmax and pass raw logits."

### Clarifications for you
- **B and C** treat the plateau as an optimization problem, the classic misreading. Training longer or faster won't break through a floor built into the maths.
- **D** confuses target formats. Integer class labels are the normal input to cross-entropy functions.
- The question stem deliberately doesn't say that the loss applies softmax. That's the knowledge being tested.

### Questions students may ask
- **"Why is accuracy still about 85% if there's a bug?"** Softmax is monotone, so the class with the largest score is unchanged. The model still learns *which* class is likely. What breaks is the loss value and the probabilities (next slide).
- **"How would I catch this in my own code?"** Check whether your model's outputs already sum to 1 before they enter the loss. Read the last line of your model definition. A loss that plateaus well above zero for no obvious reason is a red flag.

---

## Slide 24 — "Pass raw logits to the loss: softmax twice fails silently"
**Time:** 0:50–0:52 · **Goal:** see the bug's cost on the running example, and know why fused losses are stable.

### What to say
"Here's what that bug does to our running example. *[Point to the chart.]* Same model, data and optimizer; the only difference is a softmax at the end of the model. The correct run's training loss falls to about 0.14. The buggy run sits flat, around 1.63, just above the dashed line at 1.46.

*[Point to the red box.]* That line is a hard floor. With softmax applied twice, the loss for K classes can never go below ln(K − 1 + e) − 1. For 10 classes that's 1.46.

The sneaky part: validation accuracy is still 82–88% across three seeds, compared with about 89% without the bug. Easy to miss. But the model reports about 99% confidence, so its probabilities are wrong.

*[Point to the bottom box.]* Why do libraries want logits? Inside a fused cross-entropy loss is the log-sum-exp trick. It subtracts the largest logit before exponentiating, which is allowed because softmax ignores shifts. Every exponent is then zero or negative, so nothing overflows. If you exponentiated a logit of 1,000 directly in 32-bit floats, you'd get infinity.

Rule: end the model in raw logits, and let the loss apply log-softmax once."

### Clarifications for you
- **Deriving the floor (worth a minute on the board if students are curious):** after the first softmax every output p is in (0, 1) and they sum to 1. The loss is −log softmax(p)_c.
  - Best case: p_c = 1 and all others 0. The second softmax gives e/(e + K − 1), so the loss is ln(K − 1 + e) − 1 = 1.46 for K = 10.
  - Worst case: p_c = 0 and one other class gets 1. The loss is ln(K − 1 + e) = 2.46.
- **Why the buggy model becomes overconfident:** to approach the floor, the *first* softmax must push p_c toward 1, so the reported probabilities saturate near 100% even on examples the model gets wrong.
- **Overflow:** float32 overflows above about 3.4 × 10³⁸, which is roughly e⁸⁸. So even a logit of about 89 overflows if exponentiated naively.

### Questions students may ask
- **"Does the same bug exist for binary classification?"** Yes. Applying a sigmoid and then a "with-logits" binary loss applies the sigmoid twice.
- **"How do I get probabilities when I need them?"** Apply softmax yourself *after* the model, at prediction time, outside the loss.
- **"Why does the loss floor stop at 1.46 and not 0?"** The second softmax can't output more than e/(e + K − 1) for the true class, because its inputs are confined to [0, 1].

---

## Slide 25 — "Feeding the network: standardize, then mini-batches"
**Time:** 0:52–0:54 · **Goal:** standardize with training statistics only. Define mini-batch, epoch, iteration and SGD.

### What to say
"Before we step downhill, how do the data reach the model? Two rules.

First, **standardize the inputs** using the mean and standard deviation of the *training* split: subtract the mean, divide by the standard deviation. Never compute them on validation or test data. That leaks information about data you're supposed to be blind to.

Quick reminder of the three splits. Training data fit the parameters, the weights and biases. Validation data choose *hyperparameters*, things you set yourself like the learning rate, batch size and layer sizes, and choose which checkpoint to keep. Test data you touch once, at the very end.

Second, **mini-batches**. *[Point to the table.]* Computing the gradient on all 55,000 images for every step is exact but slow. One image at a time is noisy and wastes the hardware. A mini-batch of, say, 128 is the sweet spot.

*[Point to the code box.]* Each **epoch**, one pass over the data, we shuffle the 55,000 training images and cut them into batches of 128. That's 430 updates per epoch. Each update is called an **iteration**. Validation runs in bigger batches, with no shuffling and no updates.

*[Point to the formula.]* The noise in a mini-batch gradient shrinks only as 1 over the square root of the batch size. A batch 100 times bigger gives a gradient only 10 times less noisy.

Gradient descent on mini-batches is called **stochastic gradient descent, SGD**."

### Clarifications for you
- 55,000 / 128 = 429.7, so 429 full batches plus one of 88. That's 430 updates, unless the last partial batch is dropped.
- **The 1/√m claim** concerns the *standard deviation* of the mini-batch gradient. Its variance falls as 1/m (for independent samples).
- **Validation can use bigger batches** because evaluation stores no activations for a backward pass, so it needs less memory.
- **Typical batch sizes:** 32–512. Bigger batches often need a larger learning rate, and there is no free lunch.

### Questions students may ask
- **"Why does standardizing help?"** It puts all inputs on a similar scale, which improves the loss's conditioning (next slide). It also matches the assumptions behind good initializations (Part 3).
- **"Isn't using the training mean also a form of leakage?"** No. It's part of fitting the model, and fitting is allowed to use training data. You then apply the *same* numbers to validation and test data.
- **"Why shuffle every epoch?"** So that batches differ each epoch and the model doesn't learn an artefact of the data order.
- **"Which batch size should I use?"** Start with something like 64–256 that fits in memory. If you change it a lot, retune the learning rate.

---

## Slide 26 — "Gradient descent: the sharpest direction caps η"
**Time:** 0:54–0:57 · **Goal:** the update rule, why the negative gradient, and why the steepest curvature limits the learning rate.

### What to say
"The update rule: θ ← θ − η times the gradient of J. η is the learning rate, the step size.

*Why* the negative gradient? *[Point to the Taylor box.]* For a small step Δθ, the change in loss is approximately ∇Jᵀ Δθ. That's most negative when Δθ points exactly opposite the gradient. Note the word *small*: this is a local approximation.

How large can η be? *[Point to the right box.]* Take the simplest bowl, J = ½λθ², where λ is the curvature. One step gives θ ← (1 − ηλ)θ. This shrinks only if |1 − ηλ| < 1, which means η < 2/λ.

*[Point to the three panels.]* Here's a 2-D bowl with curvature 1 along one axis and 10 along the other. So η must be below 2/10 = 0.2. At 0.02 we crawl. At 0.15 we zig-zag but converge. At 0.21 we diverge.

The lesson: the *steepest* direction caps the step size for *every* direction. That's why progress along the gentle direction is slow. Fixing that is the next slide."

### Clarifications for you
- **In many dimensions,** the curvatures are the eigenvalues of the Hessian, and the largest one caps η.
- **Why 0.15 zig-zags:** along the steep axis, 1 − 0.15 × 10 = −0.5. The sign flips every step while the magnitude halves.
- **Learning-rate symptoms:**
  - NaN within a few steps → far too large.
  - Oscillating or stuck high → too large.
  - Steady fall that then flattens → good.
  - Glacial → too small.
- **Starting points:** about 3 × 10⁻⁴ to 10⁻³ for Adam/AdamW. About 0.1 for SGD with momentum on CNNs with batch normalization.
- **For curious students:** in real networks, full-batch gradient descent often operates at the "edge of stability" (Cohen et al., ICLR 2021). The sharpness rises to about 2/η and the loss keeps decreasing non-monotonically. The 2/λ rule is a local, quadratic picture.

### Questions students may ask
- **"Why is the negative gradient the steepest direction?"** Among all steps of fixed length, ∇Jᵀ Δθ is most negative when Δθ points opposite ∇J (the Cauchy–Schwarz inequality).
- **"How do I know λ_max for a real network?"** You usually don't. You tune η by experiment. λ_max can be estimated with power iteration using Hessian–vector products, but this is rarely needed in practice.
- **"Why not use Newton's method, which accounts for curvature?"** The Hessian has (number of parameters)² entries. For 10⁸ parameters that's 10¹⁶ numbers. Far too many to store or invert.

---

## Slide 27 — "Ravines defeat plain GD; momentum and Adam help"
**Time:** 0:57–1:00 · **Goal:** intuition for momentum and Adam, with honest caveats.

### What to say
"A ravine is steep across and gentle along. *[Point to the left panel.]* Plain gradient descent bounces between the walls and creeps along the floor. One step size can't suit both directions.

*[Middle panel.]* **Momentum** keeps a running sum of past gradients: v ← βv + g, then θ ← θ − ηv. The back-and-forth components cancel out, and the consistent along-the-valley component builds up speed, like a heavy ball rolling downhill.

*[Right panel.]* **Adam** divides each parameter's step by a running root-mean-square of that parameter's own gradients. So every coordinate takes steps of roughly size η, whether its gradients are large or small.

After 40 steps, each with its learning rate tuned: plain GD reaches a loss of 8.6 × 10⁻³, momentum 5.8 × 10⁻³, and Adam 6.9 × 10⁻⁴.

But this is a 2-D toy. On real networks, the ranking depends on the problem and the tuning. Don't conclude that Adam always wins."

### Clarifications for you
- **These runs are full-batch and deterministic,** so they're labelled GD, not SGD.
- **The ravine is rotated** (curvatures 1 and 25), so it doesn't line up with the coordinate axes. That would unfairly favour Adam's per-coordinate scaling.
- **Momentum:** with a constant gradient g, the velocity approaches g/(1 − β) = 10g for β = 0.9. So along a consistent direction it moves up to 10× faster than plain GD at the same η. Retune η when you switch optimizers.
- **Adam's full update** (with bias correction) is in Appendix A6. The current default is **AdamW**.

### Questions students may ask
- **"Should I always use Adam?"** AdamW is the usual first choice for most new work, including Transformers. SGD with momentum and a tuned schedule remains competitive for image classifiers.
- **"What is β?"** How much past gradients are remembered. With β = 0.9, the effective memory is about 10 steps.
- **"Does Adam need a tuned learning rate?"** Yes. Its defaults are a good start, not a guarantee.
- **"What's Nesterov momentum?"** A variant that evaluates the gradient at the look-ahead position. It's often slightly better and available as an option in most libraries.

---

## Slide 28 — Quick check 3 (largest stable η on the ravine)
**Time:** 1:00–1:04 · **Goal:** apply η < 2/λ_max.

### What to say
"Quick check. Plain gradient descent on the ravine you just saw, with curvatures 1 and 25. It converges only if η is below: A, 2/1 = 2. B, 1/25 = 0.04. C, no limit, because the loss is convex. D, 2/25 = 0.08." *[Run the protocol.]*

**After the revote:** "The answer is **D**. Convergence requires η < 2/λ_max, and the steepest curvature is 25, so η < 0.08. Rotating the ravine doesn't change its curvatures. Look back: the tuned GD learning rate on the previous slide was 0.075, just under the limit. And at that rate the gentle direction shrinks only by a factor of about 0.92 per step. That's the crawl that momentum and Adam fix."

### Clarifications for you
- **A** uses the gentle curvature instead of the steep one.
- **B, 1/λ_max,** solves the steep direction in one step exactly, but it isn't the *stability* limit.
- **C** confuses "one minimum" (convexity) with "any step size converges."
- **At exactly η = 2/λ_max,** the steep component flips sign with constant size: it oscillates forever and doesn't converge.
- **Heavy-ball momentum** on a quadratic is stable for η < 2(1 + β)/λ_max. Real networks have no fixed rule.

### Questions students may ask
- **"Why doesn't rotation change the limit?"** Curvatures are eigenvalues of the Hessian, and rotating the coordinates doesn't change eigenvalues.
- **"Is 0.08 itself OK?"** No. It's the boundary: the steep direction oscillates without shrinking. You need strictly less.

---

## Slide 29 — Break: "Back at 1:14"
**Time:** 1:04–1:14 · **Goal:** rest, with a question that leads straight into backpropagation.

### What to say
"Ten-minute break. We're back at 1:14. While you stretch, think about the question on the screen. A network has 10⁸ parameters. To measure one gradient, you nudge that one parameter and rerun the forward pass, which takes one millisecond. How long does *one* training step take? And what would you do instead?"

### Clarifications for you
- **Answer (open with it after the break):** 10⁸ × 1 ms = 10⁵ seconds ≈ 28 hours for one step, or twice that with centred differences. The fix is backpropagation: all gradients for about the cost of 3 forward passes.
- Start again on time. The second half is dense.

---

## Slide 30 — "Backpropagation: all gradients for ≈ 3 forward passes"
**Time:** 1:14–1:16 · **Goal:** backprop is the chain rule organized to reuse shared work. It computes gradients; it is not the learning rule.

### What to say
"Welcome back. The break question: 10⁸ parameters, one forward pass each, one millisecond each. That's 10⁵ seconds, about 28 hours, for a *single* training step. *[Point to the red box.]* That's finite differences: 10⁸ forward passes per step.

*[Point to the teal box.]* Backpropagation gets *all* 10⁸ gradients for about the cost of three forward passes: one forward, plus a backward pass costing about two. How?

*[Point to the chain diagram.]* Here's a chain: x goes to a, then b, then c, then the loss J. Each step uses one parameter, w₁ to w₄. Look at the gradient for w₂ and for w₁. Both contain the same orange product: ∂J/∂c times ∂c/∂b. Backprop computes that product *once*, while sweeping backward from J, and reuses it for every parameter further down. That reuse is all of backpropagation. It's called reverse-mode automatic differentiation.

Three points. *[Point to the right column.]*
One: it's just the chain rule. And if a value feeds two places, its gradient is the *sum* of what comes back from both.
Two: one backward sweep is ideal for our situation: one loss, many parameters.
Three: backprop is *not* a learning rule. It computes gradients. SGD or Adam uses them."

### Clarifications for you
- **Why the backward pass costs about 2× the forward:** for each linear layer the backward pass needs two matrix products, one for the gradient with respect to the input and one for the gradient with respect to the weights. The forward pass needs one.
- **Forward mode** would need one sweep per input parameter. It wins only when there are few inputs and many outputs.
- **The branch rule, as an example:** y = u·x with u = x gives dy/dx = x + x = 2x, the sum over both paths.
- **The memory cost:** the backward pass needs the forward activations, so they are stored (slide 37).
- **History:** reverse-mode differentiation is due to Linnainmaa (1970). Werbos proposed it for networks (1974). It was popularized by Rumelhart, Hinton and Williams (1986).

### Questions students may ask
- **"Does backprop find the best weights?"** No. It only computes the gradient at the current weights. The optimizer decides what to do with it.
- **"Is this how the brain learns?"** Almost certainly not in this exact form. It's an open research question whether the brain does something similar.
- **"Are finite differences ever useful?"** Yes, for *checking* a hand-written gradient on a few parameters (Appendix A10).
- **"Why is it called 'reverse' mode?"** Because it starts at the output (the loss) and works backward toward the inputs.

---

## Slide 31 — Worked example: "your turn, forward pass first"
**Time:** 1:16–1:20 · **Goal:** students compute a forward pass themselves (build 1 of 4).

### What to say
"Your turn. *[Point to the network.]* Two neurons in a chain, both with sigmoid activations, and a binary cross-entropy loss. The setup: w₁ = 0.5, b₁ = 0.1, w₂ = 0.8, b₂ = −0.3. One example: x = 2.0, label y = 1.

Compute z₁, a₁, z₂, a₂ and the loss L, to four decimals, with your neighbour. Three minutes. Tips on the slide: e^(−1.1) ≈ 0.3329 and e^(−0.3002) ≈ 0.7407. Since y = 1, the loss simplifies to L = −ln a₂. Natural log, please." *[Circulate. At 2.5 minutes, give a 30-second warning.]*

### Clarifications for you
- **Answers:** z₁ = 1.1, a₁ = 0.7503, z₂ = 0.3002, a₂ = 0.5745, L = 0.5543.
- **Common slips:**
  - Forgetting b₂ = −0.3 (they get z₂ ≈ 0.60).
  - Using log₁₀ instead of ln (they get L ≈ 0.24).
  - Rounding a₁ too early. Keep four or more decimals.

### Questions students may ask
- **"Which log?"** Natural log, ln, as on every slide today.
- **"Why does the loss simplify?"** With y = 1, the second term (1 − y) ln(1 − a₂) is zero.

---

## Slide 32 — Worked example: check the forward pass
**Time:** 1:20–1:21 · **Goal:** students verify their numbers (build 2 of 4).

### What to say
"Check your numbers. z₁ = 0.5 × 2.0 + 0.1 = 1.1. a₁ = σ(1.1) = 0.7503. z₂ = 0.8 × 0.7503 − 0.3 = 0.3002. a₂ = σ(0.3002) = 0.5745. L = −ln 0.5745 = 0.5543. Hands up if you matched to three decimals. *[Acknowledge.]* Keep these values. The backward pass reuses every one of them."

### Questions students may ask
- **"I got 0.5542 / 0.5544."** That's rounding. Fine.

---

## Slide 33 — Worked example: backward, one local derivative at a time
**Time:** 1:21–1:24 · **Goal:** compute the error signals δ backward. See σ′ cancel at the output and the error shrink through a sigmoid.

### What to say
"Now backward. We'll track δ, the gradient of the loss with respect to each *pre-activation* z.

Step 1. *[Point to the first line.]* The loss's derivative with respect to a₂: for binary cross-entropy that's −y/a₂ + (1 − y)/(1 − a₂). With y = 1, it's −1/a₂ = −1.7407.

Step 2. Multiply by the sigmoid's slope at z₂, which is a₂(1 − a₂) = 0.2445. That gives δ₂ = −0.4255. Notice that δ₂ equals exactly a₂ − y. The sigmoid's slope cancelled out, just like on the cross-entropy slide.

Step 3. Send the error back through the weight: ∂L/∂a₁ = δ₂ × w₂ = −0.3404.

Step 4. Multiply by the first sigmoid's slope, a₁(1 − a₁) = 0.1874. That gives δ₁ = −0.0638.

*[Point to '6.7×'.]* Compare δ₁ with δ₂: the error shrank by a factor of 6.7 through ONE sigmoid layer, because it was multiplied by w₂ × σ′(z₁) = 0.8 × 0.187. Twenty such layers and almost nothing reaches the front. That's vanishing gradients, in miniature."

### Clarifications for you
- **The cancellation:** (−1/a₂) × a₂(1 − a₂) = −(1 − a₂) = a₂ − 1 = a₂ − y.
- **The shrink factor:** 0.4255 / 0.0638 ≈ 6.67 = 1/(0.8 × 0.1874).
- **Negative gradients** mean that *increasing* that parameter reduces the loss. That makes sense: we want a₂ closer to 1.

### Questions students may ask
- **"Why do we define δ on z rather than on a?"** It makes the parameter gradients one line each (next slide): ∂L/∂w = δ × input.
- **"Will errors always shrink layer by layer?"** With sigmoids, usually, since σ′ ≤ ¼. A weight larger than 4 could compensate, but large weights push sigmoids into saturation. With ReLU and good initialization, the error can keep its size (Part 3).
- **"Why are all the δ values negative?"** Every one of them says "increase me to reduce the loss," which is consistent with pushing a₂ toward the label 1.

---

## Slide 34 — Worked example: parameter gradients and one update
**Time:** 1:24–1:27 · **Goal:** weight gradient = error × input; bias gradient = error. One step lowers the loss.

### What to say
"From the δs, the parameter gradients take one line each. *[Point to the box.]*
∂L/∂w₂ = δ₂ × a₁ = −0.3192, and ∂L/∂b₂ = δ₂ = −0.4255.
∂L/∂w₁ = δ₁ × x = −0.1276, and ∂L/∂b₁ = δ₁ = −0.0638.
The pattern: **a weight's gradient is the error at its output times the activation at its input. A bias's gradient is the error itself.**

*[Point to the dark box.]* Now take one gradient step with η = 0.5. Each parameter moves *against* its gradient: w₁ goes from 0.5 to 0.5638, b₁ from 0.1 to 0.1319, w₂ from 0.8 to 0.9596, and b₂ from −0.3 to −0.0872. Recompute the forward pass with the new values and the loss falls from 0.5543 to 0.4166. That's the whole point: the gradients exist to make that number go down."

### Clarifications for you
- **Check of the new loss:** z₁ = 1.2595, a₁ = 0.7789, z₂ = 0.6602, a₂ = 0.6593, L = 0.4166.
- **All parameters update simultaneously,** using gradients computed at the *old* values.
- **In real code** you'd pass the logit z₂ to a fused "binary cross-entropy with logits" loss rather than computing ln a₂ yourself.

### Questions students may ask
- **"Why is the bias gradient just δ?"** Because z = wa + b, so ∂z/∂b = 1.
- **"Should we update w₂ before computing w₁'s gradient?"** No. Compute every gradient first, then update all parameters together.
- **"Would a bigger η be better?"** Maybe for one step. Too large and you overshoot and the loss rises (slide 26).

---

## Slide 35 — Quick check 4 (set w₂ = 0)
**Time:** 1:27–1:31 · **Goal:** the error reaches earlier layers only through the weights. A zero path blocks it.

### What to say
"Quick check. In the worked example, set w₂ = 0, everything else unchanged. What is ∂L/∂w₁? A: exactly 0. B: unchanged, −0.1276. C: larger in magnitude. D: undefined." *[Run the protocol.]*

**After the revote:** "The answer is **A**. Layer 1's error arrives *only* through w₂: ∂L/∂a₁ = δ₂ × w₂ = 0. So δ₁ = 0, and ∂L/∂w₁ = δ₁ × x = 0. Vanishing gradients and dead units block learning by exactly this mechanism: the path back is (nearly) zero. The next slide's general equation sends the error back through the weights, and that's exactly the path w₂ = 0 blocks."

### Clarifications for you
- **B** assumes gradients don't depend on downstream weights. **C** might come from thinking "smaller w₂, bigger change." **D** might come from a feeling that zero breaks things.
- **A nice extension:** w₂ itself still gets a nonzero gradient. With w₂ = 0, a₂ = σ(−0.3) = 0.4256, so δ₂ = −0.5744 and ∂L/∂w₂ = δ₂ × a₁ ≈ −0.431. After one update w₂ is no longer zero, and layer 1 starts receiving gradient again. A single zero weight is not permanently stuck, unlike a dead ReLU or an all-zero initialization.

### Questions students may ask
- **"Then is w₂ stuck at 0 forever?"** No. Its own gradient is about −0.43, so it moves on the first step.
- **"Does this relate to dropout?"** Yes. Dropout zeroes some units, which blocks their backward paths for that step, so different paths get trained on different steps.

---

## Slide 36 — "The four equations of backpropagation"
**Time:** 1:31–1:34 · **Goal:** the general matrix rules BP1–BP4, and why the transpose appears.

### What to say
"The same rules for any fully connected layer. Define the error signal δ[l] as the gradient of the loss with respect to the pre-activation z[l].

**BP1, the output error:** the loss's gradient with respect to the output, times each unit's own slope. The circle-dot means elementwise multiplication.

**BP2, sending the error back:** multiply by the *transpose* of the next layer's weights, then gate by the slope g′. Why the transpose? Unit j feeds every unit k in the layer above, through weight W_kj. So its gradient collects from all of them: ∂L/∂a_j = Σ_k W_kj δ_k. Summing over k with j fixed walks down a *column* of W, and a column of W is a row of Wᵀ. Also, forward maps n[l] numbers to n[l+1], so backward has to map n[l+1] numbers back to n[l]. Wᵀ has exactly that shape.

**BP3:** the bias gradient is the error signal itself.

**BP4:** the weight gradient is the error times the input activation, as an outer product. That's exactly the 'error × input' pattern from our worked example.

*[Point to the bottom line.]* One special case. Softmax isn't elementwise, so BP1 doesn't apply directly. But for softmax with cross-entropy, the output error works out to simply ŷ − y."

### Clarifications for you
- **Shapes** (column vectors, one example):
  - δ[l] is (n[l], 1).
  - W[l+1]ᵀ is (n[l], n[l+1]).
  - BP4 gives (n[l], 1) × (1, n[l−1]) = (n[l], n[l−1]), which matches W[l]'s shape. That makes a good check.
- **With a batch** (examples as rows, error matrix Δ of shape (m, n[l]), activations A of shape (m, n[l−1])): dW = Δᵀ A sums the outer products over the batch. Divide by m if the loss is a mean.
- **The scalar worked example is BP1–BP4 with every matrix of size 1×1:** δ₂ (BP1 with the cancellation), δ₁ = w₂δ₂σ′ (BP2), ∂L/∂b = δ (BP3), and ∂L/∂w = δ·input (BP4).

### Questions students may ask
- **"Why the transpose and not the inverse?"** We aren't undoing the forward map. We're collecting how much each unit contributed to the downstream units. That is a sum over the weights leaving the unit, which is a transpose.
- **"What does ⊙ mean?"** Elementwise (Hadamard) product: multiply matching entries.
- **"Where does ŷ − y come from for softmax?"** From differentiating −log softmax(z)_c with respect to each z_k: ∂L/∂z_k = ŷ_k − 1[k = c].
- **"Do I need to memorize these?"** Understand them well enough to rederive BP2 for one hidden layer. That's in the homework.

---

## Slide 37 — "Automatic differentiation runs these equations for you"
**Time:** 1:34–1:36 · **Goal:** frameworks automate BP1–BP4. Know four practical consequences.

### What to say
"You won't write BP1–BP4 by hand in practice. *[Point to the code.]* Mark the parameters as 'track gradients'. Run the forward pass; every operation is recorded in a graph. Call backward on the loss, and the framework runs reverse-mode differentiation through that graph. The results, −0.1276 and −0.3192, are exactly our hand-computed numbers.

Four consequences. *[Point to the right column.]*
One: gradients **accumulate**. Each backward call *adds* to the stored gradients, so clear them before every step.
Two: **activations are stored** for the backward pass, so memory grows with batch size. If you run out of memory during training, reduce the batch size first.
Three: when **evaluating**, turn gradient tracking off. No graph, no stored activations, less memory.
Four: same numbers as by hand. Autodiff isn't approximate."

### Clarifications for you
- **Accumulation is deliberate:** it lets you sum gradients from several small batches to simulate a larger one ("gradient accumulation"). It becomes a bug only when it's unintended.
- **Gradient checkpointing** recomputes activations during the backward pass instead of storing them. It costs roughly one extra forward pass and saves memory.
- **PyTorch names, if asked:** `requires_grad=True`, `loss.backward()`, `w.grad`, `optimizer.zero_grad()`, `torch.no_grad()`.

### Questions students may ask
- **"Does autodiff use finite differences or symbolic maths?"** Neither. It applies the exact derivative of each elementary operation, numerically, at the current values, and chains them.
- **"What about ReLU at exactly zero, where there's no derivative?"** Frameworks use a fixed convention (0 for ReLU). It essentially never matters.
- **"Why would anyone *want* gradients to accumulate?"** To train with a large effective batch that doesn't fit in memory.

---

## Slide 38 — "The training loop: five steps, every mini-batch"
**Time:** 1:36–1:38 · **Goal:** memorize the loop and the train/eval distinction.

### What to say
"Everything so far fits in one loop. For every mini-batch: **one**, clear old gradients. **Two**, forward pass to get logits. **Three**, compute the loss. **Four**, backward, to get gradients. **Five**, update the parameters. *[Trace the circle on the right.]* Clear, forward, loss, backward, update. Please commit this to memory; you'll write it many times.

Around it: before training, put the model in *training mode*. That matters for layers like dropout, which we meet in Part 3. For validation, switch to *evaluation mode* and turn off gradient tracking."

### Clarifications for you
- **Training vs evaluation mode** matters for dropout (on vs off) and batch normalization (batch statistics vs running averages; Appendix B1). For a plain MLP without these layers it changes nothing, but it's a habit to build now.
- **The "update" line** is `optimizer.step()` in real code. The optimizer may be SGD, momentum or AdamW.

### Questions students may ask
- **"Can I clear gradients after the update instead of before?"** Yes. What matters is that they're cleared exactly once per step, unless you're accumulating on purpose.
- **"Why validate every epoch?"** To watch for overfitting and to keep the best checkpoint (next slide).
- **"Why does evaluation mode matter if my model has no dropout?"** For this model, it doesn't change anything. It's a habit that prevents bugs later.

---

## Slide 39 — "Running example: training curves after 20 epochs"
**Time:** 1:38–1:43 · **Goal:** read real curves. The validation loss can rise while accuracy improves. Keep the best checkpoint, and compare seeds.

### What to say
"Here's our running example after 20 epochs. *[Point to the left chart.]* The training loss keeps falling, to about 0.14. The validation loss is lowest at epoch 5, about 0.33, and then *rises* to about 0.38.

*[Point to the right chart.]* But validation accuracy keeps creeping up, to almost 90% at epoch 20. How can the loss rise while accuracy rises? The model is becoming **overconfident**. The validation examples it gets wrong, it gets wrong with growing confidence, and cross-entropy punishes confident mistakes heavily. That's the mildest form of overfitting.

Three lessons.
One: keep the checkpoint that's best on the metric you care about. By validation loss that's epoch 5; by validation accuracy it's epoch 20. Not blindly the last one.
Two: always plot both curves, training and validation.
Three: one run can mislead. With other random seeds, the best-loss epoch was 15 and 9, not 5. So choose the checkpoint on validation; don't hard-code a number of epochs.

Part 3 diagnoses and treats this."

### Clarifications for you
- **Why training accuracy at epoch 1 (82.5%) is *below* validation accuracy (85.1%):** the most likely reason is that the training metric is a running average over the epoch, collected while the weights were still improving, whereas the validation metric is measured once at the end of the epoch. The slide doesn't say how the training metric was computed, so present this as the likely explanation, not a certainty. It is a common and harmless pattern.
- **The random seed** sets the initial weights and the order of batches. That's why results vary between runs.
- **Small wiggles** in the validation curves are noise from the finite validation set and the stochastic training.

### Questions students may ask
- **"Why is validation accuracy higher than training accuracy at the start?"** See above: training accuracy is a running average during the epoch.
- **"So which checkpoint should I keep?"** The one that's best on the metric you'll report or care about, decided on validation data. If you care about calibrated probabilities, use the loss. If you care about accuracy, use accuracy.
- **"Is this overfitting?"** Mildly. The gap between training and validation loss is growing. Slide 46 classifies it.
- **"Should I stop training at epoch 5?"** You can stop early, but keeping the best checkpoint gives you the same model and lets you see what happens later.

---

# PART 3 · TRAIN AND GENERALIZE (1:43–2:19)

## Slide 40 — Part 3 title, with a warm-up question
**Time:** 1:43–1:44 · **Goal:** separate *optimization* problems (can't fit training data) from *generalization* problems (can't transfer to new data).

### What to say
"Part 3: making deep networks train, then making them generalize. Four topics: initialization, residual connections, diagnosing the gap, and closing it.

Warm-up question. *[Read the box.]* A 30-layer plain MLP trains *worse* than a 10-layer one, *on the training set*. Is that overfitting? Show of hands: yes… no… *[Count.]*

It's **not** overfitting. Overfitting means good training performance and poor validation performance. Here the *training* loss itself is worse. The deeper network is harder to *optimize*. Over the next few slides we'll see two reasons, each with a fix: the signal scale (initialization) and the degradation problem (residual connections)."

### Questions students may ask
- **"Isn't a bigger model always at least as good on training data?"** In principle, yes: a 30-layer network *could* copy the 10-layer one and set the extra layers to the identity. In practice gradient descent doesn't find that solution easily. That's the puzzle residual connections solve.

---

## Slide 41 — "Depth multiplies: gradients vanish or explode"
**Time:** 1:44–1:47 · **Goal:** backprop is a product of per-layer factors. Products shrink or grow geometrically. Know the symptoms and the diagnostic.

### What to say
"Unroll BP2 all the way down to layer 1. *[Point to the equation.]* The error at layer 1 is a product of one factor per layer: a weight matrix, gated by activation slopes.

Products of many numbers behave dramatically. *[Point to the chart; note the log scale.]* If each factor is 1.5, after 30 layers you've multiplied by about 200,000. With 1.1 the growth is milder but still compounds. At exactly 1.0 the product stays flat. With 0.9 you're down to about 0.04. With 0.5 you're at about one in a billion.

*[Point to the right boxes.]*
**Vanishing:** early layers barely learn and stay near random. The loss falls, then plateaus too high, and no error message tells you why.
**Exploding:** the loss spikes, or jumps to infinity or NaN.
**To diagnose,** log the gradient norm of each layer. If the early layers' norms are orders of magnitude smaller than the late layers', the gradients are vanishing.

*[Point to the bottom.]* The remedies: He initialization, next; residual connections, which add a path with factor exactly 1; gradient clipping, for explosions; and normalization layers (Appendix B)."

### Clarifications for you
- **The "per-layer factor" is a cartoon.** In reality each factor is a matrix, and what matters is how it scales vectors (its singular values). The picture is still right: typical gains below 1 compound into vanishing, and gains above 1 into exploding.
- **The numbers:** 0.9³⁰ ≈ 0.042, 0.5³⁰ ≈ 9.3 × 10⁻¹⁰, 1.5³⁰ ≈ 1.9 × 10⁵.
- **Exploding gradients** are classically a recurrent-network problem. Gradient clipping caps the gradient norm and treats the symptom.

### Questions students may ask
- **"Which is worse, vanishing or exploding?"** Exploding is loud: you see NaN. Vanishing is silent: training just stalls, so it's harder to spot.
- **"Does ReLU solve vanishing gradients?"** It removes the shrinkage from the *activation* when units are active. The weights can still shrink or grow the signal, so initialization still matters.
- **"Does gradient clipping fix vanishing gradients?"** No. It only caps large gradients.
- **"How do I log per-layer gradient norms?"** After the backward call, compute the norm of each parameter's gradient and record or plot it for each layer.

---

## Slide 42 — "Initialize to keep the pre-activation variance constant"
**Time:** 1:47–1:51 · **Goal:** derive the scale of He initialization. Choose the initialization by activation. Never initialize to all zeros.

### What to say
"First fix: start with the right weight scale. Look at one unit: z = Σ wᵢxᵢ. *[Point to the derivation.]* If the weights are independent, zero-mean and independent of the inputs, then Var(z) = n_in × Var(w) × E[x²]. For zero-mean inputs, E[x²] = Var(x). So keeping Var(z) = Var(x), layer after layer, requires Var(w) = 1/n_in. That's LeCun initialization.

ReLU changes things. *[Point to the second box.]* It zeroes the negative half of a symmetric input, so it *halves* the mean square: E[ReLU(z)²] = ½ E[z²]. To compensate, double the variance: Var(w) = 2/n_in. That's **He**, or Kaiming, initialization, the standard choice for ReLU networks.

*[Point to the table.]* The rule of thumb: ReLU or GELU → He, 2/n_in. tanh → Xavier/Glorot, 2/(n_in + n_out). SELU or linear → LeCun, 1/n_in.

*[Point to the code box.]* In code you specify the *standard deviation*, √(2/n_in). Biases can start at 0.

*[Point to the red box.]* Never initialize all weights to zero, or to any identical values. Every unit in a layer would compute the same thing and receive the same gradient, so they'd stay identical forever."

### Clarifications for you
- **ReLU outputs are not zero-mean.** So the derivation tracks the *second moment* E[x²], not the variance, through the ReLU. That's why the He box uses E[ReLU(z)²].
- **Xavier** averages the forward requirement (1/n_in) and the backward requirement (1/n_out). It assumes an activation that is roughly linear and symmetric near 0, like tanh.
- **Convolutions:** n_in = kernel height × width × input channels. For example, 3 × 3 × 64 = 576 gives a He std of about 0.059.
- **Library defaults:** PyTorch's default for linear and conv layers is Kaiming-uniform with a = √5, which works out to Var(w) = 1/(3·n_in). That's smaller than He. For deep ReLU networks trained from scratch, set He initialization explicitly (`nn.init.kaiming_normal_(w, nonlinearity='relu')`).
- **Normal vs uniform:** either works if the variance matches. A uniform distribution on [−a, a] has variance a²/3, so He-uniform uses a = √(6/n_in).

### Questions students may ask
- **"Why does symmetry matter so much?"** Identical units get identical gradients, so they make identical updates and never become different. You'd effectively have one unit per layer.
- **"Why is it OK for biases to start at zero?"** The random weights already break the symmetry.
- **"Do I still need careful initialization with Adam or batch normalization?"** It matters less with normalization layers, but it still matters, especially for very deep networks.
- **"Why do we track the forward signal when we care about gradients?"** The backward pass multiplies by the same matrices (transposed), so a scale that keeps the forward signal stable keeps the gradients roughly stable too.

---

## Slide 43 — Quick check 5 (ReLU with Xavier initialization)
**Time:** 1:51–1:55 · **Goal:** predict, before seeing the simulation, what a mismatched initialization does.

### What to say
"Predict before you see the simulation. A tanh network trains well. You switch every activation to ReLU but keep Xavier initialization: 20 layers, all 256 units wide. What happens to the forward signal? A: nothing, Xavier works for any activation. B: it grows by about √2 per layer and explodes. C: it shrinks by about √2 per layer and vanishes. D: only the output layer is affected." *[Run the protocol.]*

**After the revote:** "The answer is **C**. With 256 in and 256 out, Xavier gives Var(w) = 2/(256 + 256) = 1/256. That's half of He's 2/256. ReLU halves the mean square at every layer, and Xavier doesn't compensate. So each layer multiplies the mean square by ½, which multiplies the standard deviation by 1/√2. After 19 more layers that's (1/√2)¹⁹, about 1/700. Next slide: the simulation."

### Clarifications for you
- **B** has the right mechanism but the wrong direction. Explosion happens with a *too-large* initialization (std 1.0 on the next slide).
- **A** reflects the belief that Xavier is universal. **D** misses that the factor compounds through every layer.

### Questions students may ask
- **"Why √2 and not 2?"** The *variance* halves, so the *standard deviation* shrinks by √2.
- **"Is a factor of 1/700 really a problem?"** At 20 layers it's serious, and it compounds further with depth. Gradients shrink similarly, so early layers barely learn.

---

## Slide 44 — "Simulation: 20 ReLU layers, four initializations"
**Time:** 1:55–1:57 · **Goal:** see the prediction confirmed. Only He keeps the signal stable.

### What to say
"Here's the simulation: 20 ReLU layers of width 256, 1,000 random inputs, measuring the standard deviation of the activations after each layer. Note the log scale.

*[Point to the chart.]* The teal line, He, stays flat around 1. The navy line, Xavier, falls steadily: it halves about every two layers and reaches about 10⁻³ by layer 20, just as you predicted.

*[Point to the table.]* The table also shows two extremes that would wreck the chart's scale. A tiny standard deviation of 0.01 vanishes to about 10⁻¹⁹. A standard deviation of 1.0 explodes to about 10²¹. Of these four, only He keeps the signal stable through 20 ReLU layers."

### Clarifications for you
- **The chart shows only two of the four lines,** so the subtle Xavier case stays visible. The extremes are in the table. Point this out, since the title says "four."
- **Why He's layer-1 value is 0.82 rather than 1:**
  - With standard-normal inputs and He initialization, the pre-activation variance is 2, so its std is 1.41.
  - The std of ReLU(z) for a zero-mean Gaussian z is about 0.58 × std(z), and 0.58 × 1.41 ≈ 0.82.
  - The value then stays roughly flat. The small drift to 1.14 at layer 20 is finite-width randomness.
- **The extremes:** for std 1.0, each layer multiplies the mean square by 256 × 1 × ½ = 128, about 11× in std per layer. For std 0.01, the factor is 256 × 10⁻⁴ × ½ = 0.0128, about 0.11× in std per layer.

### Questions students may ask
- **"Does this matter for our three-layer running example?"** Much less, because three factors can't compound far. It becomes critical for deep networks.
- **"Are there other good initializations?"** Yes. For example, scaled orthogonal initialization. This slide compares only these four.
- **"Do the gradients behave the same way?"** Roughly, yes. The backward pass multiplies by the same weight matrices.

---

## Slide 45 — "Residual connections: why deeper can still train"
**Time:** 1:57–2:01 · **Goal:** xₗ₊₁ = xₗ + F(xₗ). A block only learns a correction, and doing nothing is easy. This answers the second half of the warm-up question.

### What to say
"Back to the warm-up. Even with good initialization and normalization, He and colleagues showed, in the ResNet paper (released in 2015, published at CVPR 2016), that deeper *plain* networks reached *higher training error* than shallower ones. They called it the degradation problem. It's not overfitting, because it's on the training set.

Their fix is beautifully simple. *[Point to the diagram.]* Instead of asking a block to learn the whole mapping, let it learn a *correction* F(x), and add the input back: xₗ₊₁ = xₗ + F(xₗ). The curved arrow is the identity 'skip' connection.

Why does this help? *[Point to the numbered boxes.]*
One: deeper residual networks didn't degrade; plain ones did.
Two: if a block isn't useful, it only has to push F toward zero to become the identity. Extra blocks are cheap to ignore.
Three: this idea is now everywhere: ResNets for images since 2015, and every Transformer block in language models.

*[Point to the derivative.]* The gradient view: the derivative of a block is the identity matrix plus ∂F/∂x. So the backward product always contains a path with factor exactly 1, straight through the skips."

### Clarifications for you
- **The original paper's nuance:** He et al. argued that degradation was *unlikely* to be caused by vanishing gradients, since their plain networks used batch normalization and had healthy gradient norms. They conjectured that deep plain networks are just very hard to optimize. For students, keep the message simple: residual connections make very deep networks trainable.
- **Residuals ease optimization; they don't guarantee it.**
- **Shapes must match** to add x and F(x). When they don't (a change in width or resolution), the skip path uses a learned projection (a linear layer or 1 × 1 convolution).
- **Notation:** here xₗ means the activation entering block l, the same role a[l] played earlier.
- **A practical trick:** initialize the last layer of each residual branch (or its normalization scale) to zero, so every block starts as the identity.

### Questions students may ask
- **"What if x and F(x) have different sizes?"** Use a projection on the skip path so the sizes match.
- **"Doesn't adding x make the network linear?"** No. F contains nonlinearities. The sum of the identity and a nonlinear function is nonlinear.
- **"Why not just initialize a plain deep network to the identity?"** Partly possible, but the residual form keeps the identity path *throughout* training, not just at the start. Empirically it works much better.
- **"Is this related to DenseNet or U-Net skip connections?"** Same family. DenseNet *concatenates* earlier features instead of adding them. U-Net skips across the encoder and decoder.

---

## Slide 46 — "Diagnose the gap first, then pick remedies"
**Time:** 2:01–2:04 · **Goal:** use training vs validation to diagnose, then apply remedies in a sensible order.

### What to say
"The model trains. Does it generalize? Compare training and validation performance. *[Walk through the table.]*
Training error high and validation error high: **underfitting**. Use a bigger model, train longer, check the learning rate.
Training low, validation much higher: **overfitting**. Use the ladder on the right.
Both similar and both good: a good fit. Scale up, and compare against baselines.
Loss flat from the very start: probably a **bug or a bad learning rate**.

*[Point to the red box.]* A first check before anything else: can your model overfit about 100 examples, reaching near-zero training loss? If it can't, suspect a bug or a setup error, like the learning rate, or regularization accidentally left on.

*[Point to the teal box.]* Our running example: training loss 0.14 and falling, validation loss up from 0.33 at epoch 5 to 0.38. Which row? *[Let students answer.]* Row 2, early overfitting.

*[Point to the ladder.]* When overfitting, a typical order of remedies. One: keep the best-validation checkpoint; we already did. Two: get more data, if you can. Three: data augmentation. Four: transfer learning. Five: weight decay. Six: dropout. And last: shrink the model. The next slides take rungs 3 to 6."

### Clarifications for you
- **Validation loss slightly *below* training loss is often normal** when dropout or augmentation are on, because they make training harder but are off at evaluation.
- **Keeping the best checkpoint is not the same as early stopping.** Keeping the checkpoint is nearly free. Halting training early is optional, and its patience must suit the learning-rate schedule, because validation loss often improves again after a learning-rate drop.
- **The ladder is typical, not fixed.** For images, transfer learning from a pretrained model is often the first thing practitioners try.
- **Shrinking the model is last** because large, well-regularized models usually do at least as well as small ones (see double descent, Appendix B3).

### Questions students may ask
- **"Why is 'smaller model' the last resort?"** Large, well-regularized models usually generalize at least as well, and shrinking also limits what the model can learn.
- **"What if training is bad AND there's a big gap?"** Fix underfitting first: make sure the model can fit the training data, then address the gap.
- **"How much more data do I need?"** There's no formula. Plotting performance against training-set size (a learning curve) tells you whether more data is still helping.
- **"Why overfit 100 examples on purpose?"** It's a cheap test that the whole pipeline (data, model, loss, optimizer) works. Any working model should memorize a tiny set.

---

## Slide 47 — "Data augmentation: free data, if the label survives"
**Time:** 2:04–2:06 · **Goal:** augment with transforms that preserve the label, and never augment validation or test data.

### What to say
"Rung 3: data augmentation. Create new training examples by transforming existing ones: a shirt is still a shirt after a small crop or a flip. But *only* use transforms that keep the label. *[Walk through the table.]*
A horizontal flip is fine for clothes and photos, but not for digits or text.
A vertical flip is fine for satellite or cell images, but not for street scenes.
Rotation is fine for microscopy, but it turns a 6 into a 9.
Colour jitter is fine for most objects, unless colour *is* the class.

*[Point to the code box.]* For Fashion-MNIST: each epoch, a fresh random crop of 80–100% resized to 28 by 28, a left-right flip with probability one half, then scaling to [0, 1] and standardizing with the training mean and std, 0.286 and 0.353. Validation and test data are only scaled and standardized.

*[Point to the red box.]* **Never augment validation or test data.** You want to measure performance on real examples."

### Clarifications for you
- **Augmentation is applied on the fly,** with a new random draw every epoch, so the model rarely sees the exact same image twice. No extra disk space is needed.
- **Be candid now or on slide 50:** for this MLP on flattened pixels, this recipe did not help. An MLP treats each pixel position as a separate input, so a shifted image looks entirely new to it. Convolutional networks share weights across positions and benefit much more.
- **Stronger methods (later sessions):** Mixup (blending two images, and their labels, pixel by pixel) and CutMix (pasting a rectangular patch of one image into another and mixing the labels in proportion to the patch's area), RandAugment, and label smoothing (softening the targets rather than the inputs).

### Questions students may ask
- **"What about augmenting at test time and averaging the predictions?"** That's a different, legitimate technique called test-time augmentation. It's part of the *model's prediction procedure*. What you must not do is change the *evaluation data* itself.
- **"How do I choose transforms?"** Ask: would a human still give this image the same label? Domain knowledge decides.
- **"Does augmentation slow training?"** Each epoch costs a bit more, and models often need more epochs to converge. It's usually worth it for image models.

---

## Slide 48 — "Transfer learning: reuse features learned on big data"
**Time:** 2:06–2:08 · **Goal:** reuse a pretrained feature extractor, either freezing it or fine-tuning it gently, and match its input preprocessing.

### What to say
"Rung 4: transfer learning. Remember from Part 1 that learned features are reusable. Take a network already trained on a huge dataset, for images typically ImageNet. Keep its feature extractor, here called F_pre. Replace its output layer with a new one sized for your classes, and train on your small dataset. *[Point to the three boxes.]*
One: **freeze and train the head.** Keep the pretrained part fixed and train only the new output layer. Fast, and best when you have very little data.
Two: **fine-tune with a small learning rate.** Then gently update all layers. It helps when your images differ more from the original data. A learning rate around ten times smaller than from scratch is a common start.
Three: **match the original inputs.** Resize and standardize exactly as the pretrained model expects. A mismatch silently degrades the features."

### Clarifications for you
- **ImageNet-pretrained image models** usually expect 224 × 224 RGB images standardized with the ImageNet channel means and stds. Using one on Fashion-MNIST would mean resizing and repeating the grey channel three times. That's possible, but overkill for this dataset.
- **The same idea underlies fine-tuning language models,** including parameter-efficient methods such as LoRA (later sessions).

### Questions students may ask
- **"Where do pretrained models come from?"** Public model hubs: torchvision, timm, Hugging Face, among others.
- **"What if my domain is very different, like X-rays or satellite images?"** Pretraining still often helps. Fine-tune more layers, and compare against training from scratch on your validation set. Domain-specific pretrained models exist for many fields.
- **"Is fine-tuning ChatGPT-style models the same thing?"** Conceptually, yes: start from pretrained weights and adapt them to a narrower task or dataset.
- **"Which layers should I unfreeze first?"** Usually the later ones, since they're the most task-specific. Early layers (edges, textures) transfer best.

---

## Slide 49 — "Weight decay and dropout"
**Time:** 2:08–2:12 · **Goal:** the mechanics of L2/AdamW weight decay and inverted dropout, and their typical settings.

### What to say
"Rungs 5 and 6. *[Point to the left.]* **Weight decay** adds λ/2 times the squared size of the weights to the loss. With plain gradient descent, each update then becomes: first shrink W by a factor (1 − ηλ), then take the normal gradient step. So weights decay toward zero unless the data push back.

The intuition: smaller weights give gentler, less sensitive functions. Remember the sigmoid neuron: the size of w set how sharply it switched. Apply decay to weights only, not to biases or normalization scales. And if you use Adam, use **AdamW**, which applies this shrink directly. I'll say why in a moment.

*[Point to the right.]* **Dropout:** during training, each unit is zeroed with probability p, independently for every example and every pass. The survivors are divided by (1 − p), so the *expected* activation is the same as at test time. At test time dropout is off: evaluation mode switches it off. Typical values: 0.5 for large fully connected layers, about 0.1 in many Transformer recipes, and little or none in conv nets with normalization."

### Clarifications for you
- **AdamW vs "Adam + L2":** with Adam, an L2 penalty added to the loss is divided by Adam's per-parameter scaling, so weights with large gradient history are barely regularized. AdamW decouples the decay and applies it directly (Loshchilov and Hutter, 2019). In PyTorch, `AdamW` is decoupled. `Adam(weight_decay=…)` is coupled L2 unless you enable its decoupled option.
- **p is the probability of *dropping* a unit.** Some older texts use the keep-probability instead, so check which one a library means.
- **Why dropout regularizes** (several complementary views): units can't co-adapt; training implicitly averages many weight-sharing sub-networks (an ensemble); and it injects noise.
- **Typical λ:** about 10⁻⁴ to 5 × 10⁻⁴ for SGD on CNNs; 0.01–0.1 with AdamW.
- **With normalization layers,** weight decay mostly changes the *effective learning rate* of the weights before them, since those layers make the output insensitive to weight scale. Mention this only to advanced students.

### Questions students may ask
- **"Why divide by (1 − p)?"** So that the expected value of each activation is the same in training and testing. Otherwise the next layer would see systematically larger inputs at test time.
- **"Why not decay biases?"** Biases only shift the function and don't make it more sensitive to inputs. Decaying them mostly hurts fit.
- **"Is weight decay the same as L2 regularization?"** For plain SGD, yes. For Adam, no, which is why AdamW exists.
- **"Does dropout slow training?"** Usually yes. The noisier training needs more epochs.
- **"Can I combine dropout with batch normalization?"** You can, but it needs care. Many modern conv nets with batch norm use little or no dropout.

---

## Slide 50 — "Regularizers tame the loss; more data lifts accuracy"
**Time:** 2:12–2:15 · **Goal:** an honest experiment. Regularizers changed the curves but not the accuracy. More data mattered most.

### What to say
"Do these remedies actually help? We tested them on a hard case: the same MLP trained on only 1,000 images.

*[Point to the left chart.]* Validation loss over 150 epochs. With no regularization, the grey line, the loss bottoms out early and then climbs: overfitting. Dropout plus weight decay, augmentation, and all three combined each flatten that climb.

*[Point to the right chart.]* But look at test accuracy, measured at the best-validation checkpoint and averaged over three seeds. No regularization: 79.8%. Dropout plus weight decay: 80.2%. Crop and flip: 79.6%. All three: 79.3%. Every variant is within about half a point of the baseline, which is within the noise. And the bottom bar: training on all 55,000 images gives 88.3%, more than 8 points better.

A likely reason for the tie: keeping the best-validation checkpoint *already* stops before the loss starts climbing. So flattening the climb barely changes which model we keep. The honest lesson: more data beat every regularizer here. Regularizers matter more for bigger models, longer training, and architectures where augmentation fits, like convolutional networks."

### Clarifications for you
- **The noise floor:** the standard error of an accuracy near 80% on 10,000 test images is √(0.8 × 0.2 / 10,000) ≈ 0.4 percentage points. So differences of 0.5 points aren't meaningful.
- **The weight decay was deliberately strong,** λ = 0.5 with AdamW at η = 10⁻³, giving a shrink of 5 × 10⁻⁴ per step. It's one setting per method, not a tuned comparison.
- **"All three" may simply need more epochs.** Heavily regularized models learn more slowly.
- **Why augmentation didn't help here:** see slide 47. An MLP has no weight sharing across positions.
- **The callback to slide 16:** 88.3% with full data is roughly a tie with the random forest (87.5%) and about 3 points above k-NN (85.3%).

### Questions students may ask
- **"So regularization is useless?"** No. This experiment shows that for this small MLP, with checkpoint selection, they didn't change test accuracy. In other settings, especially CNNs with augmentation and larger models, they matter a lot. Always test on your own problem.
- **"Why does the 'all three' curve start so high?"** Heavy regularization slows learning, so early in training its loss is higher. It was still improving at the end.
- **"How do you know 0.4 points is noise?"** From the standard error of a proportion estimated on 10,000 examples, about 0.4 points. That's why we also average over seeds.

---

## Slide 51 — Quick check 6 (2,000 images, 99.8% train vs 71% validation)
**Time:** 2:15–2:19 · **Goal:** apply the diagnosis table and the remedy ladder.

### What to say
"Last quick check. You have 2,000 labelled images. Training accuracy is 99.8%, validation 71%. Data and evaluation checks pass. Which is the best *first* experiment? A: train for more epochs. B: fine-tune a model pretrained on a related task. C: remove regularization so it fits better. D: shrink the network." *[Run the protocol.]*

**After the revote:** "The answer is **B**. 99.8% versus 71% is a large generalization gap, and the question rules out a broken pipeline, so this is overfitting. With only 2,000 images, starting from features learned on a large related dataset is the strongest first try. It's still an experiment: compare it on the same validation set, ideally alongside augmentation.

A and C would *widen* the gap: the model already fits the training set almost perfectly. D, shrinking the network, is the last rung of the ladder."

### Clarifications for you
- **A gap alone can't distinguish overfitting from data shift, leakage or label problems.** That's why the question says the checks pass. Make this point: real diagnosis starts by checking the data.

### Questions students may ask
- **"Why not data augmentation first?"** It's also a very good choice, and in practice you'd likely combine it with a pretrained model. With only 2,000 images, pretrained features usually give the biggest jump.
- **"What if no related pretrained model exists?"** Then augmentation, regularization and collecting more data are the main options.

---

## Slide 52 — Wrap-up: "Today in three sentences"
**Time:** 2:19–2:30 · **Goal:** consolidate the day, set the homework, and collect the exit ticket.

### What to say
"Today in three sentences. *[Read each one.]*
**Part 1:** a network learns its own features: linear layers interleaved with nonlinearities. Without the nonlinearity, depth collapses.
**Part 2:** losses are negative log-likelihoods; curvature caps the learning rate, and momentum and Adam help; backprop gets every gradient for the price of about three forward passes.
**Part 3:** He initialization and residual connections let deep networks train; diagnose the gap first, then add data before regularizers.

*[Point to the 'Before next session' box.]* Before next time:
Redo the worked example with x = 1, and take one gradient step with η = 0.5.
Code it in NumPy, and check your gradients with centred differences.
Count the parameters of a 784–256–10 network, and predict its first loss for 100 classes.
Derive BP2 for one hidden layer.
In one sentence: why can a residual block 'do nothing'?

*[Point to the dark box.]* Last thing, one minute: on a card, write the muddiest point of today, the thing that's least clear to you. No names needed. I'll start next session with the most common ones."

*[Use the remaining minutes for open questions.]*

### Clarifications for you: homework answers
- **Worked example with x = 1:**
  - Forward: z₁ = 0.6, a₁ = 0.6457, z₂ = 0.2165, a₂ = 0.5539, L = 0.5907.
  - Gradients: δ₂ = −0.4461, ∂L/∂w₂ = −0.2880, ∂L/∂b₂ = −0.4461; δ₁ = −0.0816, ∂L/∂w₁ = −0.0816 (since x = 1), ∂L/∂b₁ = −0.0816.
  - After one step with η = 0.5: w₁ = 0.5408, b₁ = 0.1408, w₂ = 0.9440, b₂ = −0.0770. The new loss is 0.4555.
- **NumPy check:** hand and code gradients should agree to four decimals. A centred-difference check in float64 should give relative error below about 10⁻⁷.
- **Parameters of 784–256–10:** (784 × 256 + 256) + (256 × 10 + 10) = 200,960 + 2,570 = **203,530**.
- **First loss for 100 classes:** about ln 100 = **4.61**.
- **BP2 for one hidden layer:** δ[1] = (W[2]ᵀ δ[2]) ⊙ g′(z[1]).
- **Residual "do nothing":** if F(x) = 0, then xₗ₊₁ = xₗ: the block is the identity, which is easy to reach by driving its weights toward zero.
- **Exit tickets:** read them before next session and open with the top two muddy points.

### Questions students may ask
- **"Is the homework graded?"** Defer to the course policy.
- **"What do we do next session?"** Turn today's pseudocode into real framework code. Check the course plan for specifics.
- **"Which textbook should I read?"** See the references slide. *Understanding Deep Learning* (Prince, 2023) and *Dive into Deep Learning* are both freely available online and match today's level well.

---

# APPENDIX A · Review of today (slides 53–64)

These slides are **not presented in the 150 minutes**. Use them when a student asks a question they answer, in office hours, or as pointers for review. Each entry gives the slide's purpose, a short script if you show it, and the questions it tends to answer.

## Slide 53 — Appendix A contents
A table of contents: A1 quick-check answers through A10 gradient checking. Tell students the appendix is review material for *today's* lecture.

## Slides 54–55 — A1 · Quick-check answers
**Purpose:** the answers and reasons for all six quick checks, for self-study. **Don't show these in class before the votes.**

| Quick check | Answer | One-line reason |
|---|---|---|
| 1 | C | Linear hidden layers compose into one affine map, and a monotone output doesn't bend the boundary. |
| 2 | A | cross_entropy applies log-softmax itself; a softmax in the model applies it twice, giving a floor of ≈ 1.46. |
| 3 | D | Convergence needs η < 2/λ_max = 2/25 = 0.08. |
| 4 | A | Layer 1's error arrives only through w₂, so it is 0. |
| 5 | C | Xavier gives half of He's variance, so with ReLU the std shrinks by √2 per layer. |
| 6 | B | A big gap with checked data: fine-tune a related pretrained model first. |

Tell students: "Learn the *reasons*, not the letters."

## Slide 56 — A2 · Notation
**Purpose:** a symbol table. Square brackets index layers; round brackets index examples. In derivations one example is a column vector; in code a batch stores examples as rows, so a layer computes X Wᵀ + b.
**Likely question:** "Why do the maths and the code look transposed?" Both describe the same computation. The maths uses columns for one example; code uses rows for a batch.

## Slide 57 — A3 · A ReLU network computes a patchwork of linear pieces
**Purpose:** what a ReLU network actually computes. **When to use:** if someone asks "how do straight-line ReLUs make curves?" (slides 13 or 14).
**Say if used:** "This is a real 2–12–12–1 ReLU network trained on the two moons. Each coloured patch is a region where every ReLU is either on or off in one fixed pattern, and inside a patch the whole network is exactly linear. We found 146 patches on a fine grid, and that's a lower bound, since tiny ones can be missed. So the decision boundary, in black, is made of straight pieces, yet it follows the curve. More units and layers give more pieces."
**Live demo:** playground.tensorflow.org. Choose the XOR or spiral dataset with ReLU activations, and add hidden units.
**Likely question:** "Is the boundary ever truly curved?" Not for a ReLU network: it is piecewise linear, though it can have very many small pieces. Smooth activations (tanh, GELU) give smooth boundaries.

## Slide 58 — A4 · Universal approximation, stated honestly
**Purpose:** what the theorem says and doesn't say. **When to use:** "Can a neural network fit any function?"
**Say if used:** "Informally: any continuous function on a bounded, closed region can be approximated as closely as you like by a network with one hidden layer, if it has enough units and a non-polynomial activation such as ReLU or sigmoid. The pictures are hand-built ReLU networks, not trained ones. With 3, 6 and 24 units, the maximum error falls from 0.65 to 0.14 to 0.01. Four caveats: it covers continuous functions only, not 'anything'. The width needed can be huge, exponential in the input dimension in the worst case. The theorem says the network exists, not that gradient descent finds it. And fitting isn't generalizing."
**Key references:** Cybenko (1989) for sigmoids; Hornik (1991); Leshno et al. (1993), which established the non-polynomial condition.
**Likely question:** "Then why do we need depth?" That's A5.

## Slide 59 — A5 · Why deep? Composition multiplies what each layer can do
**Purpose:** why depth can be more efficient than width.
**Say if used:** "A 'tent' function built from two ReLUs folds the interval in half. Compose it k times and you get 2ᵏ linear pieces with constant width, while a one-hidden-layer network needs roughly one unit per piece. Telgarsky (2016) proved separations of this kind: some functions computed by deep, narrow networks need exponentially many units in shallow ones. The honest caveat: these are worst-case constructions. That real data reward depth is an empirical observation, though a well-supported one."
**Likely question:** "Isn't parity the classic example?" Not a good one for networks. n-bit parity needs only about n hidden threshold units at depth 2. The exponential lower bounds for parity concern restricted AND/OR circuits.

## Slide 60 — A6 · Adam = momentum + per-parameter steps; use AdamW
**Purpose:** the full Adam and AdamW updates and default settings. **When to use:** questions after slide 27 or slide 49.
**Say if used:** "Adam keeps a running mean of the gradient, which is momentum, and a running mean of the squared gradient, which sets the scale. It corrects both for starting at zero, then steps by their ratio. Consistent gradients give steps of about η; noisy ones give smaller steps. AdamW adds weight decay *directly* to the update instead of through the loss. Start with AdamW: weight decay 0.01–0.1, and no decay on biases or normalization parameters."
**Defaults on the slide:** β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸. Typical η: 3 × 10⁻⁴ for AdamW, and 0.1 at batch size 256 for SGD with momentum on CNNs.
**Fact-checked (October 2026):** PyTorch's `AdamW` uses decoupled decay with default `weight_decay=0.01`. `Adam(weight_decay=…)` is coupled L2 unless you pass `decoupled_weight_decay=True`, an option in recent PyTorch versions. "Check your library" on the slide is good advice.
**Likely questions:**
- **"What's bias correction for?"** m and v start at zero, so early on they're biased toward zero. Dividing by (1 − βᵗ) removes that bias.
- **"Is Adam always better than SGD?"** No. SGD with momentum and a good schedule remains competitive for image classification, and results depend on the recipe.
- **"What about newer optimizers?"** Matrix-aware optimizers such as Muon have been used in large-scale language-model pretraining and can beat AdamW at scale. AdamW remains the right default to learn and to use first.

## Slide 61 — A7 · A network's loss is not one bowl
**Purpose:** non-convexity, and why training still works.
**Say if used:** "Logistic regression's loss is one convex bowl. A network's loss has many valleys, flat regions and saddle points. Left: plain gradient descent from three starting points ends in three different valleys of nearly equal loss, a one-dimensional cartoon. Right: near a saddle point the loss drops, stalls on a plateau, then drops again. In large networks, the valleys gradient descent reaches tend to be about equally good. Influential work (Dauphin et al., 2014) argues that saddles and plateaus, not bad local minima, are the main obstacle. That is well-supported evidence, not a theorem. Practical lesson: run several seeds."
**Likely question:** "How can there be many equally good solutions?" Partly from symmetry: swapping two hidden units gives exactly the same function with different weights. In high dimensions, there are many routes down.

## Slide 62 — A8 · From 1943 to today: backprop was not the bottleneck
**Purpose:** a historical timeline. Backprop was popularized in 1986, but deep learning took off around 2012, once data (ImageNet), GPUs and training techniques caught up.
**Facts for you:** McCulloch & Pitts (1943); Rosenblatt (1958); Minsky & Papert (1969); Rumelhart, Hinton & Williams (1986); LeNet-5 (1998); AlexNet (2012); ResNet (2015); Transformer (2017); AlphaFold 2, GPT-3, ChatGPT (2020–22). In 2024 the Nobel Prize in Physics went to Hopfield and Hinton, and the Nobel Prize in Chemistry to Baker, Hassabis and Jumper. Reverse-mode differentiation predates 1986 (Linnainmaa 1970; Werbos 1974).
**Likely question:** "If backprop existed in 1986, why did it take until 2012?" There wasn't enough data or compute, and training deep networks was unreliable until better activations, initialization, regularization and architectures arrived. Most of those are what Part 3 covers.

## Slide 63 — A9 · When not to reach for a neural network
**Purpose:** judgment. Neural networks win on abundant data whose structure is hard to describe by hand (images, audio, text, molecules). Other tools often win on small or tabular data, or when decisions must be audited.
**Say if used:** "With a few hundred rows, use regularized linear models, boosted trees or a Gaussian process. On tables, gradient-boosted trees such as XGBoost, LightGBM and CatBoost are hard to beat. Tabular foundation models such as TabPFN now rival them on many datasets. If every decision must be explained, a logistic regression or small tree is easier to defend. And always fit a simple baseline first."
**Fact-checked (October 2026):** TabPFN-3 was released on 12 May 2026 and TabPFN-3.5 on 15 September 2026. The package code moved to Apache 2.0 in September 2026. The *model weights* of TabPFN-3 and 3.5 remain under non-commercial or research-and-evaluation licences, and commercial use needs a licence from Prior Labs. Prior Labs reports first-place results on public benchmarks such as TabArena, but these are the vendor's own numbers. Tell students to verify them on their own data.
**Likely question:** "Why do trees beat networks on tables?" Tabular features are heterogeneous and often irregular, which suits axis-aligned splits. Networks are biased toward smooth functions, which suit images and audio better. Grinsztajn et al. (NeurIPS 2022) documented this.

## Slide 64 — A10 · Verify hand-written gradients with centred differences
**Purpose:** gradient checking for the homework.
**Say if used:** "Nudge one parameter up and down by a small ε, rerun the forward pass, and compare (J(θ + ε) − J(θ − ε)) / 2ε with your backprop gradient, using *relative* error. Below about 10⁻⁷: correct. Above 10⁻²: probably a bug. In between: investigate. With ReLU kinks, up to about 10⁻⁴ is usually fine. Use float64 and turn dropout off. Check a random subset of parameters, because each check costs two forward passes."
**Likely question:** "Why centred and not one-sided differences?" The centred difference has error proportional to ε² instead of ε, so it's far more accurate for the same ε.

---

# APPENDIX B · Going further (slides 65–73)

For later sessions. Mention that they exist; don't teach them today unless asked.

## Slide 65 — Appendix B contents
B1–B3: normalization and double descent. B4–B7: practical material for the coding sessions, ending with a full training recipe. B8: activations by layer role.

## Slide 66 — B1 · Batch normalization
**Core idea:** for each feature (each channel in a CNN), standardize over the mini-batch, then apply a learned scale γ and shift β. Training mode uses the current batch's statistics and updates running averages. Evaluation mode uses the running averages, so predictions are deterministic.
**Why it helps:** the original "internal covariate shift" explanation is now doubted (Santurkar et al., 2018). A leading, still-debated explanation is that it smooths the loss landscape, so larger learning rates stay stable.
**Likely question:** "What happens if I forget evaluation mode?" Predictions then depend on the other examples in the batch, and validation results change with validation batch size. This is a classic silent bug.

## Slide 67 — B2 · Normalization variants
**Core idea:** the same "standardize, then rescale" step, computed over different axes.
- **BatchNorm:** over the batch and spatial axes, per channel. Depends on the batch. Used in CNNs with decent batch sizes.
- **LayerNorm:** over all features of one example. Used in Transformers.
- **GroupNorm:** over groups of channels. Used in small-batch vision.
- **InstanceNorm:** per example and channel. Used in style transfer.
- **RMSNorm:** like LayerNorm without subtracting the mean. Used in many recent language models.

**Likely question:** "Why do Transformers use LayerNorm?" Sequences vary in length, batches can be small, and LayerNorm doesn't depend on other examples in the batch.

## Slide 68 — B3 · Double descent
**Core idea:** test error as a function of model size can follow the classical U, spike at the point where the model can just barely interpolate the training data (here, number of features p equals number of samples n = 60), then fall again for much larger models. Ridge regularization removes the spike.
**Caveat to state:** it's not a law. With different data, the classical small model can win. It shows why "bigger and regularized" often beats "small," and why shrinking the model is the last rung of the ladder.
**Likely question:** "Why does the error spike exactly at p = n?" At p ≈ n the model can only just interpolate the noisy training points. The fitting problem is badly conditioned, so the single interpolating solution needs enormous weights and swings wildly between the data points. With p ≫ n there are many exact fits; the minimum-norm one (which least squares picks, and which gradient descent from zero finds) keeps the weights small, which often generalizes better. Small norm is not the same as smoothness, and ridge regularization removes the spike by limiting the norm directly.

## Slide 69 — B4 · Loss functions and target formats
**Core idea:** a table of which loss goes with which output and target shape.

| Task | Loss | Model output | Target |
|---|---|---|---|
| Regression | mse | (m, 1) raw | (m, 1) float |
| Binary | bce_with_logits | (m, 1) logits | (m, 1) float 0/1 |
| Multi-class | cross_entropy | (m, K) logits | (m,) integer class indices, or (m, K) probabilities |
| Multi-label | bce_with_logits | (m, K) logits | (m, K) float 0/1 |

Classification losses take raw logits.
**Trap to mention:** for MSE, mismatched shapes (m, 1) vs (m,) broadcast to (m, m). The code runs, gives a wrong loss, and at most prints a warning.

## Slide 70 — B5 · Six training mistakes that may not raise an exception
**Core idea:** a checklist for "why doesn't my model learn?"
1. Softmax applied twice.
2. Gradients never cleared.
3. No evaluation mode for validation.
4. Layers stored in a plain Python list, so the framework doesn't register them and they never update.
5. The optimizer was built for a model that was later re-created.
6. A shape mismatch in MSE.

Tell students to check this list before asking for debugging help.

## Slide 71 — B6 · Learning-rate schedules
**Core idea:** each schedule must be stepped at the right frequency.
- One-cycle and warmup: every batch.
- Step decay: every epoch.
- Reduce-on-plateau: every epoch, fed the validation loss.
- Cosine decay: either, as long as its period is counted in the same units.

A wrong frequency raises no error and silently changes the schedule. Print the learning rate every epoch.

## Slide 72 — B7 · A complete training recipe (illustrative settings)
**Core idea:** everything together:
- AdamW with weight decay.
- A one-cycle schedule (warmup, then cosine decay), stepped every batch.
- Cross-entropy with label smoothing.
- Gradient clipping after the backward pass and before the update.
- Evaluation under no-gradient mode, keeping the best checkpoint.

**Say:** these hyperparameters are illustrative starting points, not recommendations. Validate them.
**Facts if asked (PyTorch):** `OneCycleLR`'s default warmup fraction is 0.3 (the slide sets 0.1). With Adam-family optimizers it also cycles β₁ by default, which surprises people. `zero_grad()` sets gradients to `None` by default since PyTorch 2.0.

## Slide 73 — B8 · Choose the activation by the layer's role
**Core idea:**
- Hidden layers: ReLU, or Leaky ReLU if many units die.
- Classifier output: none (raw logits).
- Regression output: none (identity).
- Strictly positive output: softplus, or predict log y.

The chart also shows GELU and SiLU, smooth ReLU variants used in Transformers. They are smooth, not saturation-free: both flatten toward 0 for very negative inputs.

## Slides 74–75 — References
Primary sources and textbooks. Not presented. Point students here for further reading. Good starting points at this level: Prince, *Understanding Deep Learning* (2023); Zhang et al., *Dive into Deep Learning*; Goodfellow, Bengio & Courville, *Deep Learning* (2016); the Stanford CS231n notes; Karpathy's "A recipe for training neural networks" (2019).

---

# Review notes for the presenter

While writing this guide I checked the deck's numbers and claims. Everything below was verified.

- **Arithmetic:**
  - The worked example (slides 31–34) and the one-step update (loss 0.5543 → 0.4166).
  - The homework answers (x = 1).
  - The softmax example (0.659 / 0.242 / 0.099).
  - The double-softmax floor (1.46) and ceiling (2.46).
  - The parameter counts (109,386 and 203,530).
  - The 430 updates per epoch.
  - The stability limits (0.2 on slide 26, 0.08 on slide 28).
  - The He/Xavier simulation scales.
  - The noise floor on slide 50 (≈ 0.4 points).
- **Time-sensitive facts (as of October 2026):** the TabPFN-3 and 3.5 release dates and licensing (A9), and the PyTorch AdamW / Adam decoupled-weight-decay behaviour (A6).

Small things to be ready for, none of them errors:

1. **Slide 44's title says "four initializations," but the chart plots only two** (He and Xavier). The other two are in the table. Say so out loud.
2. **Slide 45 switches notation** from a[l] to xₗ for a block's input. Mention that they mean the same thing.
3. **Slide 22 uses the ½(σ − y)² convention for MSE.** Library MSE functions usually omit the ½, which doubles the gradient but doesn't change the conclusion.
4. **Several experiment numbers** (the baselines on slide 16, the curves on slides 24, 39 and 50) come from the course's scripts and can't be checked from the slides alone. Find out where the scripts are posted, because slide 3 promises students they can rerun them.

## Corrections made on a second review

A second, critical pass found these problems in the first version of this guide. They are now fixed.

1. **Slide 6:** the first version credited the weighted-sum-plus-bias unit entirely to McCulloch and Pitts. Their 1943 unit had binary inputs and fixed weights; learned real-valued weights came with Rosenblatt's perceptron.
2. **Slide 21:** the first version called any first loss far below ln K "suspicious." It now notes the legitimate exception: output biases initialized to the log class frequencies.
3. **Slide 39:** the first version stated the reason for training accuracy < validation accuracy at epoch 1 as fact. The slide doesn't say how training accuracy was measured, so it's now presented as the most likely explanation.
4. **Slide 47:** the first version described CutMix as "blending." CutMix pastes a patch from one image into another; Mixup is the method that blends.
5. **Appendix B3:** the first version called the minimum-norm interpolant "smooth." The accurate statement: the spike at p ≈ n comes from ill-conditioning (huge weights), and minimum norm is not the same as smoothness.
6. **Appendix A7:** the first version stated "saddles, not bad minima, are the main obstacle" as settled. It is now attributed to Dauphin et al. (2014) as strong evidence, not a theorem.
7. **Slide 45:** the year of the ResNet result is now consistent: released 2015, published at CVPR 2016.
8. **"Running late" section:** the quick-check skip threshold (80%) contradicted the protocol (70%). Now both say 70%.
