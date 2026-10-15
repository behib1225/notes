# Convergence in Probability and Statistics: A Self-Study Tutorial

Oct 15, 2026 · @Behi

## How to use this tutorial

By the end of this tutorial you will be able to name the four modes of convergence of random variables, prove which one holds for a given sequence, explain how they relate, and pick the right one for an engineering problem. You will also understand exactly what the Law of Large Numbers and the Central Limit Theorem promise, and what they do not.

**What you need.** An elementary probability course (random variables, PMF, PDF, CDF, expectation, variance, independence, the normal distribution, moment generating functions) and calculus (limits, series, Taylor expansion). No measure theory is used. Wherever the rigorous theory would need measure theory, we say so honestly and give the intuitive version.

**How to study it.** Read each Part in order, because each one builds on the previous. Every Part has the same rhythm: an intuitive story, a precise definition, a recipe for proving it, then worked examples that go from easy to tricky. Before reading a solution, cover it and try the problem yourself for at least five minutes. Run the Python code in Part 13 alongside Parts 8 to 12; seeing the theorems happen on screen is the fastest way to believe them.

**Practice as you go.** Parts 1 to 12 each open with a short list of goals and close with a summary in three lines and a *Check yourself* block of exercises with full solutions, graded from routine to tricky. Do these before moving on. Part 14 then mixes topics across Parts, and Part 15 tests recall.

### Roadmap

| Part | Topic | Level |
| --- | --- | --- |
| 1 | Why we need convergence of random variables | Beginner |
| 2 | The toolkit: Markov and Chebyshev inequalities, useful limits | Beginner |
| 3 | Convergence in probability | Beginner |
| 4 | Almost sure convergence and the Borel–Cantelli lemmas | Intermediate |
| 5 | Convergence in mean square and in Lᵖ | Intermediate |
| 6 | Convergence in distribution | Intermediate |
| 7 | How the modes relate: the hierarchy and its counterexamples | Intermediate |
| 8 | The Weak Law of Large Numbers | Intermediate |
| 9 | The Strong Law of Large Numbers | Advanced |
| 10 | The Central Limit Theorem | Intermediate |
| 11 | Continuous mapping theorem, Slutsky's theorem, delta method | Advanced |
| 12 | Engineering applications | All levels |
| 13 | Python simulations with plots | All levels |
| 14 | Practice problems with full solutions | All levels |
| 15 | Self-check quizzes | All levels |
| 16 | One-page cheat sheet | Review |

### Notation used throughout

| Symbol | Meaning |
| --- | --- |
| X₁, X₂, X₃, … or (Xₙ) | A sequence of random variables, indexed by n = 1, 2, 3, … |
| ω | One outcome of the underlying random experiment (one "run of the universe") |
| Xₙ(ω) | The number that Xₙ takes when the outcome is ω |
| ε | A small positive tolerance, such as 0.01 |
| Fₙ(x), F(x) | CDFs: Fₙ(x) = P(Xₙ ≤ x) and F(x) = P(X ≤ x) |
| μ, σ² | Mean and variance of one observation |
| Sₙ = X₁ + … + Xₙ | Partial sum of the first n observations |
| X̄ₙ = Sₙ / n | Sample mean of the first n observations |
| iid | Independent and identically distributed |
| Φ(z) | Standard normal CDF |
| a.s. | Almost surely, meaning "with probability one" |
| \|x\|, also printed ∣x∣ inside tables | Absolute value of x (both symbols mean the same thing) |
| Lᵖ, Lʳ (p, r ≥ 1) | Convergence of the p-th (or r-th) absolute moment of the error, Part 5; L² is mean square |
| ∀, ∃ | "For every" and "there exists" (used in the side-by-side table of Section 3.1) |
| φ(t) | Characteristic function E\[e^{itX}\], the Fourier transform of the density (Section 2.7) |
| l.i.m. | "Limit in the mean": an older name for the mean square limit (Part 5) |
| A i.o. | "Aₙ infinitely often": the event that infinitely many of the events Aₙ occur (Section 4.3) |
| Superconsistent | An estimator whose error shrinks faster than the usual 1/√n, for example like 1/n (Problem 5 of Part 14) |

The four modes are written as follows. In sentences we use words ("Xₙ converges to X in probability"); in displayed formulas we use the arrows below.

```latex
X_n \xrightarrow{\text{a.s.}} X \qquad X_n \xrightarrow{P} X \qquad X_n \xrightarrow{L^p} X \;\;(\text{mean square: } X_n \xrightarrow{\text{m.s.}} X \text{ when } p=2) \qquad X_n \xrightarrow{d} X
```

## Part 1: Why we need convergence of random variables

Engineers constantly rely on "more data gives a better answer", but a random quantity never stops being random, so "getting closer" needs a careful definition. This Part explains why there is more than one sensible definition.

**Goals.** After this Part you can (1) state the ε–N definition of a limit for ordinary numbers, (2) explain why a random variable is a function of the outcome ω, and (3) describe each of the four modes of convergence in words.

### 1.1 Questions that convergence answers

Each of these everyday engineering questions is secretly a question about the limit of a sequence of random variables:

- **Measurement.** You measure a voltage n times and average the readings. Why does the average approach the true voltage, and how fast?
- **Estimation.** A reliability engineer estimates a failure rate from n lifetimes. Does the estimate get arbitrarily accurate as n grows?
- **Approximation.** Why is the sum of many small independent errors approximately normal, even when each error is uniform or exponential?
- **Simulation.** A Monte Carlo program averages 10⁶ random samples. Why can we trust a single long run, and how big is its error?

In every case we have a sequence X₁, X₂, X₃, … (averages, estimates, sums) and want to say it "converges". The trouble is that each Xₙ is random, so we must decide what closeness means.

### 1.2 Refresher: convergence of ordinary numbers

A sequence of real numbers a₁, a₂, … converges to a if, for every tolerance ε > 0, the sequence eventually stays within ε of a:

```latex
a_n \to a \iff \text{for every } \varepsilon>0 \text{ there is an } N \text{ such that } |a_n-a|<\varepsilon \text{ for all } n \ge N.
```

Read it as a game. Your opponent names a tolerance ε, however tiny. You must answer with an index N after which every term is within ε of a. If you can always win, the sequence converges.

**Example 1.1.** aₙ = 1/n converges to 0. Given ε, choose N larger than 1/ε. Then for n ≥ N we have |aₙ − 0| = 1/n ≤ 1/N < ε.

**Example 1.2.** aₙ = (−1)ⁿ does not converge. It keeps jumping between −1 and 1, so for ε = 0.5 no single value a has all late terms within 0.5 of it.

**Example 1.3 (functions).** Let fₙ(x) = xⁿ on \[0, 1\]. For each fixed x < 1, xⁿ → 0. At x = 1, fₙ(1) = 1 for all n. So fₙ converges at every single point x, but the limit is 0 for x < 1 and 1 at x = 1. This is called pointwise convergence. Keep it in mind: a random variable is a function, and almost sure convergence will turn out to be pointwise convergence that is allowed to fail on a set of probability zero.

### 1.3 A random variable is a function of the outcome

This single idea unlocks the whole subject. A random experiment produces an outcome ω. A random variable X is a rule that turns the outcome into a number X(ω). A sequence of random variables is a sequence of such rules, all driven by the same outcome ω.

So once nature picks ω, you get an ordinary sequence of numbers X₁(ω), X₂(ω), X₃(ω), … that you can study with the calculus of Section 1.2. A different ω gives a different number sequence. Think of ω as one complete history of an experiment that runs forever, for example the full infinite list of coin tosses.

**Example 1.4.** Toss a fair coin forever. Let Xₙ be the fraction of heads in the first n tosses. If ω = (H, T, H, H, T, …), then X₁(ω) = 1, X₂(ω) = 1/2, X₃(ω) = 2/3, X₄(ω) = 3/4, X₅(ω) = 3/5, and so on. For this ω the number sequence appears to drift toward 1/2. For the very strange outcome ω = (H, H, H, …), Xₙ(ω) = 1 for every n. The Law of Large Numbers will say such strange outcomes have total probability zero.

### 1.4 The million-engineers picture

Imagine one million engineers, each running the same infinitely long experiment independently. Engineer number k gets her own outcome ωₖ and so her own number sequence X₁(ωₖ), X₂(ωₖ), …. The four modes of convergence are four different ways of saying "the sequence settles down to X":

| Mode | What it says about the million engineers |
| --- | --- |
| Almost sure | Essentially every engineer sees her own number sequence converge, in the ordinary calculus sense. |
| In probability | Pick any fixed large time n. Only a tiny fraction of engineers are more than ε away at that time, but which engineers are off may keep changing forever. |
| Mean square | The average, over all engineers, of the squared error (Xₙ − X)² goes to zero. |
| In distribution | The histogram of the million values of Xₙ looks more and more like the histogram of X. No individual engineer needs to be close to anything. |

These are genuinely different. Part 7 gives sequences that satisfy one mode and fail another. Before that, Parts 3 to 6 study each mode on its own.

**Example 1.5 (a sequence that converges in every sense).** Let X be any random variable and Xₙ = (1 + 1/n)X. For every outcome ω, Xₙ(ω) = (1 + 1/n)X(ω) is an ordinary number sequence that converges to X(ω). This is the strongest possible behaviour, called sure convergence, and it implies every other mode. (The Lᵖ modes of Part 5 are defined only when the p-th moments exist; here E\[(Xₙ − X)²\] = E\[X²\]/n² → 0 needs E\[X²\] < ∞.) Most interesting sequences, like running averages of noisy measurements, are not this well behaved, which is why we need the weaker modes.

**Example 1.6 (deterministic sequences are a special case).** A constant c can be viewed as a random variable that equals c for every outcome. If Xₙ = 1/n for every outcome, then the random sequence is just the number sequence 1/n, and every mode of convergence reduces to ordinary convergence to 0. Our definitions must agree with calculus in this case, and they do.

### Summary and check yourself: Part 1

**In three lines.** A random variable is a function of the outcome ω, so a random sequence Xₙ produces one ordinary number sequence for every outcome. We can ask whether each path converges (almost surely), whether large errors become improbable (in probability), whether the average squared error vanishes (mean square), or whether only the histograms match (in distribution). These four questions can have different answers.

**Exercises.**

1. Let aₙ = (n + 1)/n. Show aₙ → 1 by finding, for ε = 0.01, an index N that wins the ε–N game.
2. Does cₙ = n/(n + 1) + (−1)ⁿ converge? Justify.
3. A coin is tossed forever and the outcome begins ω = (T, H, H, T, H, …). Let Xₙ be the fraction of heads in the first n tosses. Compute X₁, …, X₅ for this ω.
4. Let X \~ N(0, 1). (a) For Xₙ = X + (−1)ⁿ/n, for which outcomes ω does Xₙ(ω) converge, and to what? (b) Same question for Yₙ = (−1)ⁿX.
5. Match each statement about a million engineers with a mode of convergence. (a) "My own running average has settled at 3.2." (b) "At n = 1000, only 0.1% of us are off by more than 0.01." (c) "The histogram of our 10⁶ values of X₁₀₀₀ looks like a bell curve." (d) "The average of our squared errors is 10⁻⁶ and still falling."

**Solutions.**

1. |aₙ − 1| = 1/n, and 1/n < 0.01 exactly when n > 100. So N = 101 works: every n ≥ 101 gives |aₙ − 1| < 0.01. (Any larger N also works.)
2. No. For even n, cₙ = n/(n + 1) + 1 → 2; for odd n, cₙ = n/(n + 1) − 1 → 0. The terms keep returning near 2 and near 0, so no single number a has all late terms within ε = 0.5 of it.
3. X₁ = 0/1 = 0, X₂ = 1/2, X₃ = 2/3, X₄ = 2/4 = 1/2, X₅ = 3/5.
4. (a) For every ω, Xₙ(ω) = X(ω) + (−1)ⁿ/n → X(ω), because (−1)ⁿ/n → 0. This is sure convergence. (b) Yₙ(ω) alternates between X(ω) and −X(ω), so it converges only when X(ω) = 0. That set of outcomes has probability P(X = 0) = 0, so Yₙ does not converge almost surely. (It does converge in distribution, as Problem 10 in Part 14 shows.)
5. (a) almost surely (a single path), (b) in probability (one fixed time), (c) in distribution (histogram only), (d) mean square (average squared error).

## Part 2: The toolkit

Almost every convergence proof in this tutorial uses only the handful of tools below: two inequalities, the union bound, and a few limits from calculus. Learn them now and the later proofs become short.

**Goals.** After this Part you can (1) bound tail probabilities with Markov's and Chebyshev's inequalities, (2) compute the mean, variance and standard error of a sample mean, (3) split a mean squared error into variance plus bias², and (4) decide whether the standard series and limits used later converge.

### 2.1 Markov's inequality

If Y ≥ 0 and a > 0, then

```latex
P(Y \ge a) \le \frac{E[Y]}{a}.
```

**In words.** A non-negative quantity cannot often be much bigger than its average. If the average height of a fence is 1 m, at most 1/3 of its length can be 3 m or taller.

**Proof (continuous case; the discrete case replaces integrals by sums).** Since Y ≥ 0, split the integral for E\[Y\] at a and throw away the lower part:

```latex
E[Y]=\int_0^\infty y f(y)\,dy \;\ge\; \int_a^\infty y f(y)\,dy \;\ge\; \int_a^\infty a f(y)\,dy = a\,P(Y\ge a).
```

**Generalised Markov.** Applying Markov to Y = |Z|ᵖ and a = εᵖ gives, for any p > 0,

```latex
P(|Z|\ge\varepsilon) \le \frac{E|Z|^p}{\varepsilon^p}.
```

This is the bridge from "moments are small" to "large deviations are unlikely". We use it constantly.

### 2.2 Chebyshev's inequality

If X has mean μ and finite variance σ², then for every ε > 0

```latex
P(|X-\mu|\ge\varepsilon) \le \frac{\sigma^2}{\varepsilon^2}.
```

**Proof.** Apply generalised Markov with Z = X − μ and p = 2, and note E\[(X − μ)²\] = σ².

**In words.** Small variance forces the random variable to stay near its mean. Chebyshev works for every distribution, which makes it a universal but often loose bound.

**Example 2.1 (resistor measurements).** A meter's readings have mean μ (the true resistance) and standard deviation 2 Ω. You average n = 100 readings. How likely is the average to be off by 0.5 Ω or more?

The sample mean X̄ has mean μ and variance σ²/n = 4/100 = 0.04. Chebyshev gives

```latex
P(|\bar X-\mu|\ge 0.5)\le \frac{0.04}{0.25}=0.16.
```

So at least 84% of the time, the average is within 0.5 Ω. Part 10 shows the true probability is about 1.2% when the CLT applies, so Chebyshev is conservative, but it needs no assumption about the shape of the noise.

**Example 2.2 (Markov on a lifetime).** A component's lifetime T has mean 1000 h. Without knowing anything else, P(T ≥ 5000 h) ≤ 1000/5000 = 0.2. If T is actually exponential, the exact value is e⁻⁵ ≈ 0.0067, which shows how loose Markov can be.

### 2.3 Mean and variance of the sample mean

If X₁, …, Xₙ are uncorrelated with common mean μ and variance σ², then

```latex
E[\bar X_n]=\mu, \qquad \operatorname{Var}(\bar X_n)=\frac{1}{n^2}\sum_{i=1}^n \operatorname{Var}(X_i)=\frac{\sigma^2}{n}.
```

The standard deviation of the average, σ/√n, is called the standard error. It shrinks like 1/√n: to halve the error you need four times the data. This one formula drives Parts 8, 10 and 12.

### 2.4 Bias–variance decomposition

For any random variable Y and any constant c,

```latex
E[(Y-c)^2]=\operatorname{Var}(Y)+\big(E[Y]-c\big)^2.
```

**Proof.** Write Y − c = (Y − E\[Y\]) + (E\[Y\] − c), square it, and take expectations. The cross term is 2(E\[Y\] − c)·E\[Y − E\[Y\]\] = 0.

In estimation language: mean squared error = variance + bias². We use this in Part 5.

### 2.5 Union bound and independence

For any events A₁, A₂, … (independent or not),

```latex
P\Big(\bigcup_{n} A_n\Big)\le \sum_n P(A_n).
```

For independent events, P(A₁ and A₂ and … and Aₘ) = P(A₁)·P(A₂)·…·P(Aₘ).

Also useful: P(|X + Y| > ε) ≤ P(|X| > ε/2) + P(|Y| > ε/2), because if |X + Y| > ε then at least one of |X|, |Y| must exceed ε/2. This is the triangle inequality combined with the union bound.

### 2.6 Limits and series you will need

| Fact | Where it is used |
| --- | --- |
| (1 + x/n)ⁿ → eˣ for every real x | Maxima, minima, Poisson and exponential limits |
| (1 + x/n + o(1/n))ⁿ → eˣ, where o(1/n) means a term that is small compared with 1/n | CLT proof by moment generating functions |
| 1 − x ≤ e⁻ˣ for all real x | Second Borel–Cantelli lemma |
| Σ 1/nᵖ converges if p > 1 and diverges if p ≤ 1 | Deciding almost sure convergence |
| Σ rⁿ converges if 0 ≤ r < 1 | Exponentially small tail probabilities |
| Σ 1/(n ln n) diverges, Σ 1/(n (ln n)²) converges (integral test) | Borderline Borel–Cantelli problems |
| Tail of a convergent series: Σₙ≥N aₙ → 0 as N → ∞ | First Borel–Cantelli lemma |

### 2.7 Moment generating functions

The moment generating function (MGF) of X is M(t) = E\[e^{tX}\]. Two facts matter here. First, if two MGFs agree on an interval around t = 0, the distributions are equal. Second, near t = 0,

```latex
M(t)=1+t\,E[X]+\frac{t^2}{2}E[X^2]+o(t^2).
```

The MGF of an independent sum is the product of the MGFs. For N(0, 1), M(t) = e^{t²/2}.

**Characteristic functions: the engineer's Fourier transform.** An MGF can be infinite: for the Cauchy distribution of Example 8.4, E\[e^{tX}\] = ∞ for every t ≠ 0. The characteristic function

```latex
\varphi(t)=E\big[e^{itX}\big]=\int_{-\infty}^{\infty} e^{itx}f(x)\,dx
```

never has this problem, because |e^{itx}| = 1, so it exists for every distribution. It is the Fourier transform of the density (up to the sign convention), and the Fourier rules you already know are exactly the probability rules you need: the transform of a convolution is the product of the transforms (the density of an independent sum is the convolution of the densities, so φ\_{X+Y} = φ\_X·φ\_Y); scaling the variable rescales the transform (φ\_{aX}(t) = φ\_X(at)); and the transform determines the signal (two distributions with the same φ are equal). Every MGF argument in this tutorial can be repeated with φ in place of M, and that is how the rigorous proofs of the CLT (Part 10) and of the Cauchy fact in Example 8.4 are done; we keep MGFs for readability. For N(0, 1), φ(t) = e^{−t²/2}; for the standard Cauchy, φ(t) = e^{−|t|}, which is not differentiable at 0, the Fourier-domain signature of a missing mean.

### Summary and check yourself: Part 2

**In three lines.** Markov's inequality turns a small mean of a non-negative quantity into a small tail probability, and Chebyshev's inequality does the same with a small variance. The sample mean has variance σ²/n, so its standard error shrinks like 1/√n. Mean squared error = variance + bias², and the series Σ 1/nᵖ (p > 1) and Σ rⁿ (r < 1) converge while Σ 1/n diverges.

**Exercises.**

1. A packet delay Y ≥ 0 has mean 2 ms. Bound P(Y ≥ 10 ms) without any other information.
2. A supply voltage X has mean 50 V and standard deviation 5 V. Give a lower bound for P(40 V < X < 60 V) that holds for every distribution. Compare it with the value if X is normal.
3. Let Z \~ N(0, 1), for which E\[Z²\] = 1 and E\[Z⁴\] = 3. Bound P(|Z| ≥ 2) with generalised Markov using p = 2 and p = 4, and compare with the exact value 0.0455.
4. Compute (1 − 3/n)ⁿ for n = 100 and compare it with its limit.
5. Decide whether each series converges: (a) Σ 1/n^1.01, (b) Σ e^{−√n}, (c) Σ 1/(n ln n) for n ≥ 2, (d) Σ 0.99ⁿ.
6. Single readings have standard deviation 10 mV. What is the standard error of the mean of 25 readings? How many readings give a standard error of 0.5 mV?
7. An estimator has mean μ + 0.2 and variance 0.05. What is its mean squared error?

**Solutions.**

1. Markov: P(Y ≥ 10) ≤ E\[Y\]/10 = 2/10 = 0.2.
2. Chebyshev with ε = 10: P(|X − 50| ≥ 10) ≤ 25/100 = 0.25, so P(40 < X < 60) ≥ 0.75. If X is normal, the event is |Z| < 2 and its probability is about 0.954. The universal bound is safe but far from the normal value.
3. With p = 2: P(|Z| ≥ 2) ≤ 1/4 = 0.25. With p = 4: P(|Z| ≥ 2) ≤ 3/16 ≈ 0.19. Higher moments can give sharper bounds for light-tailed variables; both are still well above the exact 0.0455.
4. (0.97)¹⁰⁰ = e^{100 ln 0.97} ≈ e^{−3.046} ≈ 0.0476, close to the limit e⁻³ ≈ 0.0498.
5. (a) Converges: a p-series with p = 1.01 > 1. (b) Converges: for n ≥ 100 we have √n ≥ 2 ln n, so e^{−√n} ≤ e^{−2 ln n} = 1/n², and Σ 1/n² converges. (c) Diverges: by the integral test, ∫ dx/(x ln x) = ln(ln x) → ∞. (d) Converges: a geometric series with ratio 0.99 < 1.
6. Standard error = 10/√25 = 2 mV. Solving 10/√n = 0.5 gives √n = 20, so n = 400 readings. A four-fold reduction in error costs sixteen times the data.
7. MSE = variance + bias² = 0.05 + 0.2² = 0.09.

## Part 3: Convergence in probability

Convergence in probability says that for any tolerance you choose, the chance of being outside that tolerance shrinks to zero as n grows. It is the workhorse mode of statistics: "a consistent estimator" means exactly this.

**Goals.** After this Part you can (1) state the definition of convergence in probability and read it in words, (2) prove it by computing or bounding P(|Xₙ − X| > ε), and (3) recognise sequences that fail it, including ones whose distribution never changes.

### 3.1 Definition

Xₙ converges to X in probability if, for every ε > 0,

```latex
\lim_{n\to\infty} P\big(|X_n-X|>\varepsilon\big)=0 \qquad\text{equivalently}\qquad \lim_{n\to\infty} P\big(|X_n-X|\le\varepsilon\big)=1.
```

**How to read it.** Fix a tolerance ε. For each n, compute the number pₙ(ε) = P(|Xₙ − X| > ε), the probability of an "unacceptable" error at time n. This is now an ordinary sequence of numbers. Convergence in probability means this number sequence goes to 0, for every choice of ε.

**Three things to notice.**

1. It is a statement about each fixed time n separately. It says nothing about whether one engineer's whole path stays close (that is almost sure convergence, Part 4).
2. The event |Xₙ − X| > ε involves Xₙ and X together, so Xₙ and X must be defined on the same experiment. When X is a constant c, this is automatic, and that is the most common case in practice.
3. Using > ε or ≥ ε makes no difference, since the condition must hold for every ε.

**The definitions side by side.** Writing the three definitions with their quantifiers in the same order makes the differences visible. Read ∀ as "for every" and ∃ as "there exists"; δ is a second small tolerance, for probabilities.

| Mode | Statement | Who gets an N? |
| --- | --- | --- |
| Ordinary numbers (and sure convergence) | ∀ε > 0 ∃N: ∀n ≥ N, ∣aₙ − a∣ < ε. For a random sequence: this holds for every outcome ω. | One N for the whole future, chosen after ε |
| Almost surely (Part 4) | P( ∀ε > 0 ∃N: ∀n ≥ N, ∣Xₙ − X∣ < ε ) = 1 | Each outcome ω gets its own N; only a probability-zero set of outcomes has none |
| In probability (this Part) | ∀ε > 0 ∀δ > 0 ∃N: ∀n ≥ N, P(∣Xₙ − X∣ > ε) < δ | One N, but it only controls the probability at each single time n; which outcomes are off may change from one n to the next |

The almost-sure line puts the probability *outside* the whole "for every ε there is an N" statement: a single path must win the ε–N game of Section 1.2. The in-probability line puts the probability *inside*: at each time n separately, few paths are off. Part 4 makes this precise and shows the two are different.

### 3.2 The proof recipe

1. Write down the "bad event" |Xₙ − X| > ε.
2. Either compute its probability exactly (easy when you know the distribution of Xₙ − X), or bound it with Markov or Chebyshev.
3. Show the result goes to 0 as n → ∞, for every fixed ε > 0.

### 3.3 Worked examples

**Example 3.1 (a rare blip).** Let Xₙ be Bernoulli with P(Xₙ = 1) = 1/n and P(Xₙ = 0) = 1 − 1/n. Show Xₙ → 0 in probability.

*Solution.* Take any ε > 0. If ε ≥ 1, the event |Xₙ| > ε is impossible, so its probability is 0. If 0 < ε < 1, then |Xₙ| > ε happens exactly when Xₙ = 1, so P(|Xₙ| > ε) = 1/n → 0. Either way the limit is 0.

**Example 3.2 (shrinking uniform).** Let Xₙ \~ Uniform(0, 1/n). Show Xₙ → 0 in probability.

*Solution.* For any ε > 0, once n > 1/ε the whole interval (0, 1/n) lies below ε, so P(|Xₙ| > ε) = 0 for all such n. The limit is 0.

**Example 3.3 (exponential with growing rate).** Let Xₙ \~ Exponential with rate n (mean 1/n). Then

```latex
P(|X_n-0|>\varepsilon)=P(X_n>\varepsilon)=e^{-n\varepsilon}\to 0.
```

**Example 3.4 (the maximum of uniforms).** Let U₁, U₂, … be iid Uniform(0, 1) and Mₙ = max(U₁, …, Uₙ). Show Mₙ → 1 in probability.

*Solution.* Since Mₙ ≤ 1 always, the bad event |Mₙ − 1| > ε is the event Mₙ < 1 − ε (take 0 < ε < 1). The maximum is below 1 − ε only if every Uᵢ is, and by independence

```latex
P(M_n<1-\varepsilon)=P(U_1<1-\varepsilon)\cdots P(U_n<1-\varepsilon)=(1-\varepsilon)^n\to 0.
```

*Engineering meaning:* the largest of n noise samples drawn from a bounded range creeps up to the top of the range. This is how the maximum is used to estimate the unknown upper end of a uniform distribution (Practice Problem 5).

**Example 3.5 (using Chebyshev with a bias).** Suppose E\[Xₙ\] = μ + 1/n and Var(Xₙ) = 1/n, with no other information about the distribution. Show Xₙ → μ in probability.

*Solution.* We cannot compute the probability exactly, so we bound it. By generalised Markov with p = 2 and the bias–variance decomposition of Section 2.4,

```latex
P(|X_n-\mu|>\varepsilon)\le\frac{E[(X_n-\mu)^2]}{\varepsilon^2}=\frac{\operatorname{Var}(X_n)+(\text{bias})^2}{\varepsilon^2}=\frac{1/n+1/n^2}{\varepsilon^2}\to 0.
```

The lesson is general: **if the bias and the variance both go to zero, you have convergence in probability.**

**Example 3.6 (rare but huge values).** Let Xₙ = n with probability 1/n and Xₙ = 0 otherwise. For 0 < ε, P(|Xₙ| > ε) ≤ 1/n → 0, so Xₙ → 0 in probability. But E\[Xₙ\] = n·(1/n) = 1 for every n, which does not go to 0.

*Lesson:* convergence in probability does not control expectations. A very rare but very large value can keep the mean away from the limit. Part 5 introduces a stronger mode that does control the mean.

**Example 3.7 (a non-example: independent coin flips).** Let X₁, X₂, … be iid Bernoulli(1/2). Show this sequence does not converge in probability to anything.

*Solution.* Suppose, for contradiction, that Xₙ → X in probability for some X. Then the differences Xₙ − Xₙ₊₁ would go to 0 in probability, because

```latex
P(|X_n-X_{n+1}|>\varepsilon)\le P(|X_n-X|>\varepsilon/2)+P(|X_{n+1}-X|>\varepsilon/2)\to 0.
```

But Xₙ and Xₙ₊₁ are independent fair bits, so they differ with probability 1/2 for every n. With ε = 1/2, P(|Xₙ − Xₙ₊₁| > 1/2) = 1/2, which does not go to 0. Contradiction.

*Lesson:* having the same distribution at every n is not enough. Convergence in probability requires the values themselves to get close.

### 3.4 Useful properties

If Xₙ → X and Yₙ → Y in probability, then:

- Xₙ + Yₙ → X + Y and aXₙ → aX in probability (proof: the union bound trick of Section 2.5).
- XₙYₙ → XY in probability, and g(Xₙ) → g(X) for any continuous g (the continuous mapping theorem, Part 11). Both are stated here without proof; the product rule follows from the CMT applied to the pair (Xₙ, Yₙ).
- The limit is unique: if Xₙ → X and Xₙ → X′ in probability, then P(X = X′) = 1.

### 3.5 When is it used?

Convergence in probability is the natural language for **consistency of estimators**: an estimator θ̂ₙ is consistent for θ if θ̂ₙ → θ in probability, meaning that with enough data, the estimate is within any tolerance you like with probability close to 1. The Weak Law of Large Numbers (Part 8) is the most famous convergence-in-probability theorem.

### Summary and check yourself: Part 3

**In three lines.** Xₙ → X in probability when, for every fixed tolerance ε, the probability of an error larger than ε goes to 0. Prove it by computing that probability exactly, or by bounding it with Markov or Chebyshev; vanishing bias and variance are enough. It controls each fixed time n, not whole paths, and it does not force means to converge.

**Exercises.**

1. Let U \~ Uniform(0, 1) and Xₙ = Uⁿ. Show Xₙ → 0 in probability.
2. Let Xₙ \~ Poisson(1/n). Show Xₙ → 0 in probability.
3. Suppose E\[Xₙ\] = 2 + 3/n and Var(Xₙ) = 4/√n. Show Xₙ → 2 in probability.
4. Let X₁, X₂, … be iid N(0, 1). Does Xₙ → 0 in probability?
5. Prove directly from the definition: if Xₙ → X in probability, then 3Xₙ + 1 → 3X + 1 in probability.
6. A defect rate p is estimated by the fraction p̂ of defective units among n = 100 tested. Use Chebyshev to bound P(|p̂ − p| > 0.1) without knowing p.

**Solutions.**

1. For 0 < ε < 1, P(Uⁿ > ε) = P(U > ε^{1/n}) = 1 − ε^{1/n}. Since ε^{1/n} = e^{(ln ε)/n} → e⁰ = 1, this probability → 0. (For ε ≥ 1 the probability is 0.)
2. Xₙ takes values 0, 1, 2, …, so for 0 < ε < 1, P(|Xₙ| > ε) = P(Xₙ ≥ 1) = 1 − e^{−1/n} → 0. For ε ≥ 1 the probability is even smaller.
3. By the bias–variance decomposition and generalised Markov, P(|Xₙ − 2| > ε) ≤ \[Var(Xₙ) + (3/n)²\]/ε² = (4/√n + 9/n²)/ε² → 0.
4. No. P(|Xₙ − 0| > 1) = P(|Z| > 1) ≈ 0.317 for every n, which does not go to 0. Each Xₙ is a fresh standard normal; nothing shrinks.
5. |(3Xₙ + 1) − (3X + 1)| = 3|Xₙ − X|, so P(|(3Xₙ + 1) − (3X + 1)| > ε) = P(|Xₙ − X| > ε/3) → 0, because ε/3 is just another positive tolerance.
6. Var(p̂) = p(1 − p)/100 ≤ 0.25/100, since p(1 − p) ≤ 1/4. Chebyshev gives P(|p̂ − p| > 0.1) ≤ (0.25/100)/0.01 = 0.25. With 100 units you can only guarantee a 75% chance of being within ±0.1; Part 10 shows the CLT gives a much more optimistic and realistic answer.

## Part 4: Almost sure convergence

Almost sure convergence says that the single path you actually observe will, with probability one, converge in the ordinary calculus sense. It is the mode that justifies trusting one long experiment or one long simulation run.

**Goals.** After this Part you can (1) state almost sure convergence as convergence of individual paths, (2) explain the difference between "unlikely at one time" and "eventually stays", (3) use the two Borel–Cantelli lemmas to prove or disprove almost sure convergence, and (4) analyse the standard examples, including the sliding hump.

### 4.1 Definition

Xₙ converges to X almost surely (also "with probability 1" or "a.s.") if

```latex
P\Big(\big\{\omega:\ \lim_{n\to\infty} X_n(\omega)=X(\omega)\big\}\Big)=1.
```

**How to read it.** Freeze the outcome ω. You now have an ordinary number sequence X₁(ω), X₂(ω), … and an ordinary number X(ω). Ask the calculus question: does the sequence converge to that number? Collect all outcomes where the answer is yes. If that collection has probability 1, the convergence is almost sure.

**Why "almost"?** Some outcomes may misbehave, as long as together they have probability zero. In Example 1.4 the outcome "heads forever" is perfectly possible to write down, yet its probability is (1/2)·(1/2)·… = 0. Almost sure convergence allows exceptions like this. Sure convergence (every single outcome converges, as in Example 1.5) is stronger but rarely needed.

### 4.2 The "eventually stays" picture

Compare the two modes for a fixed tolerance ε:

- **In probability:** at each single time n, the chance of being more than ε away is small for large n.
- **Almost surely:** the chance of being more than ε away at **any** time after N is small for large N.

Formally, almost sure convergence is equivalent to the following, for every ε > 0:

```latex
\lim_{N\to\infty} P\Big(|X_n-X|>\varepsilon \text{ for at least one } n\ge N\Big)=0
\quad\Longleftrightarrow\quad
P\big(|X_n-X|>\varepsilon \text{ for infinitely many } n\big)=0.
```

In words: with probability one, each path makes only finitely many excursions outside any ε-band, and then stays inside it forever. Convergence in probability allows a path to leave the band again and again, as long as the departures become rarer at each fixed time.

**Why the two statements are equivalent (a proof without measure theory).** It uses one fact about probabilities beyond the union bound, the *continuity of probability*: if events B₁ ⊇ B₂ ⊇ B₃ ⊇ … shrink down to an event B (B is the set of outcomes that lie in every Bₙ), then P(Bₙ) → P(B). This is the rule that turns statements about every finite N into one statement about the whole infinite future.

1. *Translate non-convergence into events.* The path Xₙ(ω) fails to converge to X(ω) exactly when there is a tolerance, which we can take of the form 1/k, such that |Xₙ(ω) − X(ω)| > 1/k for infinitely many n. So the set of bad outcomes is the union, over k = 1, 2, 3, …, of the events Dₖ = {|Xₙ − X| > 1/k infinitely often}.
2. *Almost sure convergence ⇔ every Dₖ has probability 0.* If each P(Dₖ) = 0, the union bound gives P(bad outcomes) ≤ Σₖ P(Dₖ) = 0, which is almost sure convergence. Conversely, each Dₖ is contained in the set of bad outcomes, so almost sure convergence forces P(Dₖ) = 0 for every k. Because every ε > 0 exceeds some 1/k, and the events {|Xₙ − X| > ε i.o.} shrink as ε grows, this is the same as the right-hand statement above: P(|Xₙ − X| > ε i.o.) = 0 for every ε > 0.
3. *The i.o. event is the limit of the tail events.* Fix ε and let T\_N = {|Xₙ − X| > ε for at least one n ≥ N}. These tail events shrink as N grows (T₁ ⊇ T₂ ⊇ …), and an outcome lies in every T\_N exactly when it leaves the ε-band infinitely often. So the T\_N shrink down to {|Xₙ − X| > ε i.o.}, and continuity of probability gives P(T\_N) → P(|Xₙ − X| > ε i.o.). Hence the left-hand statement (P(T\_N) → 0) holds exactly when the right-hand one does. ∎

The same trick of replacing "every ε > 0" by the countable list ε = 1, 1/2, 1/3, … reappears in the proof of the Strong Law (Part 9, step 3).

&#91;image: Fifteen running-mean paths: left panel marks the single time n = 200, right panel shades the whole tail n ≥ 200\]

The figure shows 15 running-mean paths of Exponential(1) data, made the same way as Simulation 1 in Part 13. Convergence in probability asks how many paths are outside the band at one time slice (left: 3 of 15 at n = 200). Almost sure convergence asks how many leave the band at any time in the whole tail (right: 4 of 15 for n ≥ 200). The tail event always contains the slice event, so it is at least as likely; that is why almost sure convergence is the stronger requirement.

A very practical sufficient condition follows from this (it is the first Borel–Cantelli lemma, proved in Section 4.3):

```latex
\sum_{n=1}^{\infty} P\big(|X_n-X|>\varepsilon\big)<\infty \ \text{ for every } \varepsilon>0 \quad\Longrightarrow\quad X_n\xrightarrow{\text{a.s.}}X.
```

**Read this condition carefully: summability is sufficient, not necessary.** Summable bad probabilities, such as 1/n² or e^{−εn}, guarantee almost sure convergence. Non-summable ones, such as 1/n, settle nothing on their own:

- If the Xₙ are **independent**, a divergent sum does rule out almost sure convergence (the second Borel–Cantelli lemma, Section 4.3).
- **Without independence**, the sequence may still converge almost surely. Take ω \~ Uniform(0, 1) and Xₙ = 1 when ω < 1/n, else 0. Then P(Xₙ = 1) = 1/n is not summable, yet for every ω > 0 we have Xₙ(ω) = 0 as soon as n > 1/ω, so Xₙ → 0 almost surely. The bad events here are nested (each one inside the previous), so a path that escapes one escapes all later ones.

### 4.3 The Borel–Cantelli lemmas

The two Borel–Cantelli lemmas decide whether infinitely many events from a sequence A₁, A₂, … will occur. The first proves almost sure convergence; the second, under independence, disproves it.

**The event "infinitely often".** The event "Aₙ infinitely often", written Aₙ i.o., is the set of outcomes ω that belong to infinitely many of the Aₙ. Its complement, "only finitely many Aₙ occur", means that from some time on none of them occur.

```latex
\{A_n \text{ i.o.}\}=\bigcap_{N=1}^{\infty}\ \bigcup_{n\ge N}A_n \qquad\text{(for every } N\text{, some } A_n \text{ with } n\ge N \text{ occurs).}
```

*Example.* Toss a coin forever and let Aₙ = {toss n is heads}. The outcome (H, T, H, T, …) is in "Aₙ i.o."; the outcome (H, H, T, T, T, T, …) is not, because heads stop after toss 2.

*The link to convergence.* With Aₙ = {|Xₙ − X| > ε}, Section 4.2 says exactly this: Xₙ → X almost surely if and only if P(Aₙ i.o.) = 0 for every ε > 0.

**Lemma (first Borel–Cantelli lemma, BC1).** If Σ P(Aₙ) < ∞, then P(Aₙ i.o.) = 0. No independence is needed.

*Proof, step by step.*

1. Fix any N. If ω is in infinitely many Aₙ, it is certainly in at least one Aₙ with n ≥ N. So {Aₙ i.o.} is contained in the union of A\_N, A\_{N+1}, …
2. Probability is monotone, and the union bound applies: P(Aₙ i.o.) ≤ P(A\_N or A\_{N+1} or …) ≤ Σₙ≥N P(Aₙ).
3. The right side is the tail of a convergent series, so it tends to 0 as N → ∞.
4. The left side does not depend on N, and it is at most a quantity that can be made as small as we like. So it equals 0. ∎

*In words.* Let C be the number of events that occur. Then E\[C\] = Σ P(Aₙ), because C is a sum of indicators. A count with a finite mean cannot be infinite with positive probability, so only finitely many events occur.

**Pause here before the second lemma.** Cover the table "Why independence matters" below and predict: for independent events with P(Aₙ) = 1/n, do you expect infinitely many of them to occur? And for P(Aₙ) = 1/n²? Write your guess down, then read on and check it.

**Lemma (second Borel–Cantelli lemma, BC2).** If the Aₙ are independent and Σ P(Aₙ) = ∞, then P(Aₙ i.o.) = 1.

*Proof, step by step.*

1. Fix N and M > N. By independence, the probability that none of A\_N, …, A\_M occurs is the product (1 − P(A\_N))·…·(1 − P(A\_M)).
2. Use 1 − x ≤ e⁻ˣ on each factor: the product is at most exp(−\[P(A\_N) + … + P(A\_M)\]).
3. Let M → ∞. The sum in the exponent diverges, so the bound tends to e^{−∞} = 0. Hence, for every N, "no Aₙ after N" has probability 0, that is, "some Aₙ after N occurs" has probability 1.
4. An event of probability 1 for each N = 1, 2, 3, … (countably many) holds for all N at once with probability 1. "Some Aₙ after every N" is exactly "Aₙ i.o.". ∎

```latex
P\Big(\bigcap_{n=N}^{M}A_n^c\Big)=\prod_{n=N}^{M}\big(1-P(A_n)\big)\le\exp\Big(-\sum_{n=N}^{M}P(A_n)\Big)\xrightarrow[M\to\infty]{}0.
```

**Why independence matters.** Two dependent examples with Σ P(Aₙ) = ∞ give opposite answers, so a divergent sum alone decides nothing:

| Events | Σ P(Aₙ) | P(Aₙ i.o.) |
| --- | --- | --- |
| Independent, P(Aₙ) = 1/n | ∞ | 1 (by BC2) |
| Aₙ = A for every n, with P(A) = 1/2 | ∞ | 1/2 (i.o. exactly when A occurs) |
| Aₙ = {U < 1/n}, U \~ Uniform(0, 1) | ∞ | 0 (each U > 0 escapes once n > 1/U) |
| Any events with P(Aₙ) = 1/n² | π²/6, finite | 0 (by BC1) |

**The two lemmas as a recipe for almost sure convergence.** Take Aₙ = {|Xₙ − X| > ε}.

- Σ P(Aₙ) < ∞ for every ε > 0 ⇒ Xₙ → X almost surely (BC1, any dependence).
- The Xₙ are independent, X = c is a constant, and Σ P(|Xₙ − c| > ε) = ∞ for some ε > 0 ⇒ Xₙ does not converge to c almost surely (BC2).
- The sum diverges and the Xₙ are dependent ⇒ use another argument, usually the direct pointwise one.

Three more worked applications of the two lemmas, on borderline sums such as Σ 1/(n ln n) and on how high noise peaks grow over a long record, are collected in Section 9.1. Read them now if you want more practice before moving on; nothing in Parts 5 to 8 depends on them.

### 4.4 Proof recipes

- **Direct (pointwise) method.** Describe Xₙ(ω) explicitly, find the set of ω where the number sequence converges, and show it has probability 1.
- **Summability method (proves).** Show Σ P(|Xₙ − X| > ε) < ∞ for every ε > 0, then apply BC1. Sufficient, not necessary.
- **Independence method (disproves).** If the Xₙ are independent and Σ P(|Xₙ − c| > ε) = ∞ for some ε > 0, BC2 shows the path leaves the ε-band infinitely often, so there is no almost sure convergence to c.
- **Monotone method.** If Xₙ(ω) is monotone in n for every outcome and Xₙ → X in probability, then Xₙ → X almost surely. (Why: a monotone number sequence always has a limit, possibly ±∞, so Xₙ(ω) → some Y(ω) for every outcome. Then Xₙ → Y almost surely, hence in probability (Part 7), and limits in probability are unique (Section 3.4), so Y = X with probability 1.)

### 4.5 Worked examples

**Example 4.1 (the classic pointwise example).** Let ω \~ Uniform\[0, 1\], X(ω) = ω and Xₙ(ω) = ω + ωⁿ. Show Xₙ → X almost surely.

*Solution.* Fix ω. If 0 ≤ ω < 1, then ωⁿ → 0, so Xₙ(ω) → ω = X(ω). If ω = 1, then Xₙ(1) = 2 for all n, which does not converge to X(1) = 1. So the sequence converges for every ω in \[0, 1), a set of probability 1. The single bad point ω = 1 has probability 0. Hence Xₙ → X almost surely, but not surely. Compare with Example 1.3: this is pointwise convergence with one harmless exception.

**Example 4.2 (exponentials with growing rate, revisited).** Let Xₙ \~ Exponential(rate n), independent or not. In Example 3.3 we found P(Xₙ > ε) = e^{−nε}. This is a geometric series with ratio e^{−ε} < 1, so

```latex
\sum_{n=1}^{\infty}e^{-n\varepsilon}=\frac{e^{-\varepsilon}}{1-e^{-\varepsilon}}<\infty,
```

and by the summability method Xₙ → 0 almost surely.

**Example 4.3 (maximum of uniforms, revisited).** In Example 3.4, Mₙ = max(U₁, …, Uₙ) and P(|Mₙ − 1| > ε) = (1 − ε)ⁿ, which is summable. So Mₙ → 1 almost surely. The monotone method gives the same conclusion: adding a sample can only increase the maximum, so each path Mₙ(ω) is non-decreasing and bounded by 1.

**Example 4.4 (a sequence that converges in probability but not almost surely).** Let X₁, X₂, … be independent, with Xₙ \~ Bernoulli(1/n). By Example 3.1, Xₙ → 0 in probability.

But Σ 1/n = ∞, and because the Xₙ are independent, the second Borel–Cantelli lemma (Section 4.3) says that with probability 1 infinitely many Xₙ equal 1. A path that keeps returning to 1 cannot converge to 0. So Xₙ does **not** converge to 0 almost surely.

*Interpretation:* each engineer sees a 1 at time n with chance only 1/n, yet every engineer keeps seeing occasional 1s forever (around ln n of them by time n). Contrast this with Xₙ \~ Bernoulli(1/n²): now Σ 1/n² < ∞, so with probability 1 only finitely many 1s ever appear, and Xₙ → 0 almost surely.

**Example 4.5 (the sliding hump).** Let ω \~ Uniform\[0, 1\]. Build indicator functions in blocks. Block 1 has the single interval \[0, 1\]. Block 2 splits \[0, 1\] into halves \[0, 1/2\] and \[1/2, 1\]. Block k splits \[0, 1\] into k intervals of length 1/k. List all these intervals one after another, and let Xₙ(ω) = 1 if ω lies in the n-th interval of the list, and 0 otherwise.

- In probability: P(Xₙ = 1) is the length of the n-th interval, 1/k for an interval in block k. As n → ∞, k → ∞, so P(|Xₙ| > ε) → 0. Thus Xₙ → 0 in probability.
- Almost surely: every ω lies in at least one interval of every block, so Xₙ(ω) = 1 infinitely often. No path converges to 0. Thus Xₙ does not converge almost surely, for any ω at all.

The hump gets thinner (so the chance of being hit at a fixed time shrinks), but it sweeps across every point again and again.

&#91;image: The sliding hump: shaded intervals for n = 1 to 21, with the intervals containing ω = 0.3 shown in red\]

Each column is one Xₙ, equal to 1 on its shaded interval of ω. The heights, which are the probabilities P(Xₙ = 1), shrink like 1/k. But follow the dashed line ω = 0.3: it passes through a red interval in every block, so the path X₁(0.3), X₂(0.3), … returns to 1 once per block, forever.

### 4.6 When is it used?

Almost sure convergence is what you want when you observe **one realisation over a long time**: a single long simulation, one long data record, one adaptive filter running forever. The Strong Law of Large Numbers (Part 9) guarantees that the running average you watch on your own screen converges, not just that "most people's" averages are close at a fixed time.

### Summary and check yourself: Part 4

**In three lines.** Xₙ → X almost surely when, with probability 1, the observed path converges in the ordinary calculus sense; equivalently, every ε-band is left only finitely often. Summable bad probabilities prove it (BC1, any dependence); for independent Xₙ, a divergent sum disproves it (BC2). For dependent sequences with a divergent sum, argue path by path.

**Exercises.**

1. Let ω \~ Uniform\[0, 1\] and Xₙ(ω) = nωⁿ. Does Xₙ → 0 almost surely?
2. Let U \~ Uniform(0, 1) and Xₙ = Uⁿ (Exercise 3.1). Show Xₙ → 0 almost surely, in two different ways.
3. Independent Xₙ \~ Bernoulli(n^{−1.1}). Does Xₙ → 0 almost surely?
4. Independent Xₙ \~ Bernoulli(1/√n). Decide convergence to 0 in probability and almost surely.
5. Independent Xₙ \~ Exponential with rate ln(n + 1), so the mean is 1/ln(n + 1) → 0. Show Xₙ → 0 in probability but not almost surely. Compare with Example 4.2, where the rate is n.
6. A fair coin is tossed forever. Let Aₙ be the event that tosses n, n + 1, …, n + 9 are all heads. Show that, with probability 1, a run of ten heads occurs infinitely often. (Careful: the Aₙ are not independent.)
7. True or false: "If P(|Xₙ − X| > ε) = 1/n for every n and every small ε, then Xₙ cannot converge to X almost surely."

**Solutions.**

1. Yes. For 0 ≤ ω < 1, nωⁿ → 0 (an exponential decay beats the linear factor n; for example, take logarithms: ln n + n ln ω → −∞). At ω = 1, Xₙ = n → ∞. The single bad point ω = 1 has probability 0.
2. *Pointwise:* for every U < 1, Uⁿ → 0, and P(U = 1) = 0. *Monotone:* Uⁿ decreases in n for every outcome, and it converges in probability (Exercise 3.1), so by the monotone method it converges almost surely.
3. Yes. Σ n^{−1.1} is a p-series with p = 1.1 > 1, so it converges, and BC1 says only finitely many Xₙ equal 1.
4. In probability: P(Xₙ = 1) = 1/√n → 0, so yes. Almost surely: Σ 1/√n = ∞ and the Xₙ are independent, so by BC2 infinitely many Xₙ equal 1 with probability 1. No almost sure convergence.
5. P(Xₙ > ε) = e^{−ε ln(n + 1)} = (n + 1)^{−ε} → 0, so Xₙ → 0 in probability. For any ε ≤ 1, Σ (n + 1)^{−ε} = ∞, and the Xₙ are independent, so BC2 gives Xₙ > ε infinitely often: no almost sure convergence. With rate n (Example 4.2), P(Xₙ > ε) = e^{−nε} is summable, and convergence is almost sure. The difference is how fast the bad probabilities vanish: polynomially and too slowly versus exponentially.
6. The events Aₙ overlap, so use only A₁, A₁₁, A₂₁, …, which involve disjoint blocks of ten tosses and are therefore independent. Each has probability 2^{−10} = 1/1024, and Σ 1/1024 = ∞. By BC2, infinitely many of these blocks are all heads, with probability 1. Choosing an independent subsequence is a standard trick.
7. False. The probabilities alone do not decide it. With independent Xₙ it is true (BC2), but with Xₙ = 1{U < 1/n} the same probabilities give almost sure convergence (Section 4.2).

## Part 5: Convergence in mean square and in Lᵖ

Mean square convergence says the average squared error goes to zero. For engineers it is the most natural mode of all, because the mean squared error is the power of the error signal.

**Goals.** After this Part you can (1) state Lᵖ and mean square convergence, (2) prove mean square convergence to a constant from a bias and a variance, (3) explain why Lᵖ convergence implies convergence in probability and controls means, and (4) build sequences that converge in one Lᵖ but not another.

### 5.1 Definition

Let p ≥ 1. Xₙ converges to X in Lᵖ ("in p-th mean") if E|Xₙ|ᵖ and E|X|ᵖ are finite and

```latex
\lim_{n\to\infty} E\big[|X_n-X|^p\big]=0.
```

The two cases you will meet most are p = 1 ("in mean", E|Xₙ − X| → 0) and p = 2 ("in mean square", written m.s. or l.i.m. for "limit in the mean"):

```latex
X_n\xrightarrow{\text{m.s.}}X \iff E\big[(X_n-X)^2\big]\to 0.
```

**How to read it.** For each n, compute one number: the average over all outcomes of the error raised to the power p. If that sequence of numbers goes to 0, you have Lᵖ convergence. Unlike the previous two modes, this is a statement about an expectation, so large but rare errors are penalised according to their size.

**Engineering reading.** If Xₙ − X is an error signal, E\[(Xₙ − X)²\] is its mean power. Mean square convergence means "the error power goes to zero". This is the mode used in signal processing, Wiener filtering, adaptive filters, and least squares.

### 5.2 The key tool: bias and variance

When the limit is a constant c, the bias–variance decomposition of Section 2.4 gives a clean test:

```latex
E[(X_n-c)^2]=\operatorname{Var}(X_n)+\big(E[X_n]-c\big)^2\to 0
\iff E[X_n]\to c \ \text{ and } \ \operatorname{Var}(X_n)\to 0.
```

Both terms are non-negative, so the sum goes to 0 exactly when each term does. **Mean square convergence to a constant = vanishing bias + vanishing variance.**

### 5.3 Two facts about Lᵖ convergence

**Fact 1: Lᵖ implies in probability.** By generalised Markov,

```latex
P(|X_n-X|>\varepsilon)\le\frac{E|X_n-X|^p}{\varepsilon^p}\to 0.
```

This is often the easiest way to prove convergence in probability: show the mean squared error goes to zero.

**Fact 2: higher p is stronger.** If 1 ≤ p < r and Xₙ → X in Lʳ, then Xₙ → X in Lᵖ. The general proof uses Jensen's (or Lyapunov's) inequality. For p = 1, r = 2 it follows from the Cauchy–Schwarz inequality, or from Var(|Y|) ≥ 0:

```latex
E|Y|\le\sqrt{E[Y^2]}, \quad\text{so}\quad E|X_n-X|\le\sqrt{E[(X_n-X)^2]}\to 0.
```

So mean square convergence implies convergence in mean. The converse fails (Example 5.5).

**A bonus: Lᵖ convergence controls means.** If Xₙ → X in L¹, then |E\[Xₙ\] − E\[X\]| ≤ E|Xₙ − X| → 0, so E\[Xₙ\] → E\[X\]. If Xₙ → X in mean square, then also E\[Xₙ²\] → E\[X²\]. Convergence in probability alone guarantees neither (Example 3.6).

### 5.4 Worked examples

**Example 5.1 (the sample mean).** Let X₁, X₂, … be uncorrelated with mean μ and variance σ². Then E\[X̄ₙ\] = μ (no bias) and Var(X̄ₙ) = σ²/n, so

```latex
E\big[(\bar X_n-\mu)^2\big]=\frac{\sigma^2}{n}\to 0.
```

The sample mean converges to μ in mean square, and therefore in probability. That is already a proof of the Weak Law of Large Numbers (Part 8).

**Example 5.2 (shrinking uniform).** Let Xₙ \~ Uniform(0, 1/n). Then

```latex
E[X_n^2]=\int_0^{1/n} x^2\, n\,dx=\frac{1}{3n^2}\to 0,
```

so Xₙ → 0 in mean square.

**Example 5.3 (maximum of uniforms).** Let Mₙ = max(U₁, …, Uₙ) with Uᵢ iid Uniform(0, 1). Its CDF is xⁿ on \[0, 1\], so its density is n xⁿ⁻¹ and E\[Mₙᵏ\] = n/(n + k). Hence

```latex
E[(1-M_n)^2]=1-2\frac{n}{n+1}+\frac{n}{n+2}=\frac{2}{(n+1)(n+2)}\to 0.
```

So Mₙ → 1 in mean square. Notice the error power falls like 1/n², much faster than the 1/n of the sample mean. (Check the algebra: put everything over (n + 1)(n + 2); the numerator is (n + 1)(n + 2) − 2n(n + 2) + n(n + 1) = 2.)

**Example 5.4 (rare huge values fail).** Let Xₙ = n with probability 1/n and 0 otherwise (Example 3.6). Then E|Xₙ − 0| = 1 for every n, so Xₙ does not converge to 0 in L¹, and E\[Xₙ²\] = n → ∞, so not in mean square either. Yet Xₙ → 0 in probability. The rare value n is so large that it dominates the average error.

**Example 5.5 (L¹ but not mean square).** Let Xₙ = √n with probability 1/n and 0 otherwise. Then

```latex
E|X_n|=\sqrt n\cdot\frac1n=\frac{1}{\sqrt n}\to 0, \qquad E[X_n^2]=n\cdot\frac1n=1\not\to 0.
```

So Xₙ → 0 in L¹ but not in L². Squaring punishes large errors more, which is why mean square convergence is the stronger requirement.

**Example 5.6 (a biased estimator that still converges).** Let Yₙ = X̄ₙ + 1/n, where X̄ₙ is the sample mean of iid data with mean μ and variance σ². Bias = 1/n → 0, variance = σ²/n → 0, so E\[(Yₙ − μ)²\] = σ²/n + 1/n² → 0. A small bias that vanishes is harmless.

### 5.5 When is it used?

- Whenever performance is measured by **mean squared error**: estimator design, least squares, filtering, quantisation noise.
- As a **convenient route to convergence in probability**: computing a mean and a variance is usually easier than computing a probability.
- For **random processes and signals**: "the time average converges in mean square" is the standard notion of ergodicity in the mean (Part 12).

Caution: mean square convergence and almost sure convergence do not imply each other. Part 7 has the counterexamples.

### Summary and check yourself: Part 5

**In three lines.** Xₙ → X in Lᵖ when the average p-th power of the error goes to 0; mean square (p = 2) means the error power vanishes. To a constant, mean square convergence is exactly "bias → 0 and variance → 0". It implies convergence in probability and convergence of means, but rare huge values can block it even when convergence in probability holds.

**Exercises.**

1. Let Xₙ \~ Exponential with rate n. Show Xₙ → 0 in mean square.
2. Let Xₙ = nᵃ with probability 1/n and 0 otherwise, where a > 0. For which a does Xₙ → 0 (i) in probability, (ii) in L¹, (iii) in mean square?
3. A "shrinkage" estimator of μ is θ̂ₙ = (n/(n + 1))·X̄ₙ, from iid data with mean μ and variance σ². Find its bias and variance, and decide whether θ̂ₙ → μ in mean square.
4. Prove that if Xₙ → X in mean square, then E\[Xₙ\] → E\[X\].
5. (Quantisation.) An analogue value X is rounded to the nearest multiple of Δₙ = 1/n, giving Xₙ. Show Xₙ → X in every mode, and bound the quantisation error power E\[(Xₙ − X)²\].

**Solutions.**

1. E\[Xₙ\] = 1/n and Var(Xₙ) = 1/n², so E\[Xₙ²\] = 1/n² + 1/n² = 2/n² → 0.
2. (i) P(Xₙ ≠ 0) = 1/n → 0: in probability for every a. (ii) E|Xₙ| = nᵃ/n = n^{a−1} → 0 exactly when a < 1. (iii) E\[Xₙ²\] = n^{2a}/n = n^{2a−1} → 0 exactly when a < 1/2. So for 1/2 ≤ a < 1 the sequence converges in L¹ but not in mean square, and for a ≥ 1 in neither.
3. E\[θ̂ₙ\] = nμ/(n + 1), so bias = −μ/(n + 1) → 0. Var(θ̂ₙ) = (n/(n + 1))²·σ²/n → 0. By Section 5.2, θ̂ₙ → μ in mean square. A little bias can even lower the MSE for small n, which is the idea behind shrinkage estimators.
4. |E\[Xₙ\] − E\[X\]| = |E\[Xₙ − X\]| ≤ E|Xₙ − X| ≤ √(E\[(Xₙ − X)²\]) → 0. The last step is Fact 2 of Section 5.3.
5. Rounding to the nearest grid point gives |Xₙ − X| ≤ Δₙ/2 = 1/(2n) for every outcome. So Xₙ → X surely (hence almost surely, in probability and in distribution) and E\[(Xₙ − X)²\] ≤ 1/(4n²) → 0. (For a smooth distribution the error is close to uniform on \[−Δ/2, Δ/2\], giving the familiar quantisation noise power Δ²/12.)

## Part 6: Convergence in distribution

Convergence in distribution says that probabilities computed from Xₙ approach the probabilities computed from X. It is about the shape of the distribution, not about the values themselves, and it is what the Central Limit Theorem delivers.

**Goals.** After this Part you can (1) state convergence in distribution through CDFs and explain why jump points are excluded, (2) prove it with the CDF, PMF or MGF method, (3) recognise when no limit distribution exists because probability escapes to infinity, and (4) explain why it says nothing about the values themselves.

### 6.1 Definition

Let Fₙ be the CDF of Xₙ and F the CDF of X. Xₙ converges to X in distribution (also "in law" or "weakly") if

```latex
\lim_{n\to\infty}F_n(x)=F(x)\quad\text{at every } x \text{ where } F \text{ is continuous.}
```

**How to read it.** For each fixed x, P(Xₙ ≤ x) is a number. As n grows, that number should approach P(X ≤ x). Picture the CDF curves of X₁, X₂, X₃, … gradually lying on top of the CDF of X.

**The key surprise.** Only distributions are compared. Xₙ and X need not be defined on the same experiment, and Xₙ need not be close to X in value. That is why this mode is called weak. It is exactly enough to approximate probabilities, which is all you need to build confidence intervals and tests.

**An equivalent definition through expectations.** Xₙ → X in distribution if and only if

```latex
E[g(X_n)]\to E[g(X)] \quad\text{for every bounded continuous function } g
```

(stated without proof; it is part of the "portmanteau theorem"). The CDF definition asks the same thing only for the step function g(y) = 1{y ≤ x}, which is bounded but jumps at y = x; that is why the CDF version must exclude the points where F jumps, and the expectation version needs no such exception. Three things this form buys you. (1) It makes the Part 1 picture precise: "the histograms match" means that averages of every smooth, bounded function of the values match. (2) It proves the in-distribution case of the continuous mapping theorem (Section 11.1): if h is continuous, then g∘h is bounded and continuous whenever g is, so E\[g(h(Xₙ))\] → E\[g(h(X))\], which is exactly h(Xₙ) → h(X) in distribution. (3) It shows what is *not* covered: unbounded g such as g(y) = y or g(y) = y². That gap is why means and variances need the extra condition of Section 7.3 (uniform integrability) before they can be read off a limit distribution.

### 6.2 Why we ignore jump points

**Example 6.1.** Let Xₙ = 1/n with probability 1 (a constant). Common sense says Xₙ → 0. The limit X = 0 has CDF F(x) = 0 for x < 0 and F(x) = 1 for x ≥ 0, which jumps at x = 0.

For every n, Fₙ(0) = P(1/n ≤ 0) = 0, while F(0) = 1. So Fₙ(0) does not converge to F(0). At any other x the CDFs do converge: for x < 0, Fₙ(x) = 0 = F(x); for x > 0, once 1/n ≤ x we have Fₙ(x) = 1 = F(x).

The only failure is at the jump point x = 0, where an arbitrarily small shift of the values flips the CDF. The definition deliberately ignores such points, so Xₙ → 0 in distribution, as it should.

### 6.3 Ways to prove convergence in distribution

| Method | Statement | Best for |
| --- | --- | --- |
| CDF method | Show Fₙ(x) → F(x) at every continuity point of F | Maxima, minima, explicit CDFs |
| PMF method | For integer-valued Xₙ and X: show P(Xₙ = k) → P(X = k) for every integer k | Binomial to Poisson |
| MGF method (continuity theorem) | If Mₙ(t) → M(t) for all t in an interval around 0, and M is the MGF of X, then Xₙ → X in distribution | Sums of independent variables, the CLT |
| Density method | If densities fₙ(x) → f(x) for every x, then Xₙ → X in distribution (Scheffé's theorem) | Families like t → normal |

*Why the PMF method works.* For counts (values 0, 1, 2, …), Fₙ(x) = P(Xₙ = 0) + … + P(Xₙ = ⌊x⌋) is a finite sum, and a finite sum of convergent terms converges to the sum of the limits, which is F(x). *The density method* is only sufficient: Example 6.6 converges in distribution even though its densities do not converge. *The MGF method* has a more general version with the characteristic functions E\[e^{itX}\] of Section 2.7, which always exist (Lévy's continuity theorem).

### 6.4 Worked examples

**Example 6.2 (binomial to Poisson, PMF method).** Let Xₙ \~ Binomial(n, λ/n) for a fixed λ > 0. For a fixed integer k ≥ 0,

```latex
P(X_n=k)=\binom{n}{k}\Big(\frac{\lambda}{n}\Big)^k\Big(1-\frac{\lambda}{n}\Big)^{n-k}
=\frac{\lambda^k}{k!}\cdot\underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to 1}\cdot\underbrace{\Big(1-\frac{\lambda}{n}\Big)^{n}}_{\to e^{-\lambda}}\cdot\underbrace{\Big(1-\frac{\lambda}{n}\Big)^{-k}}_{\to 1}\to e^{-\lambda}\frac{\lambda^k}{k!}.
```

So Xₙ → Poisson(λ) in distribution. *Engineering meaning:* the number of defective chips in a large batch with a small defect rate, or the number of bit errors in a long frame, is approximately Poisson.

**Example 6.3 (the same result by MGFs).** The MGF of Binomial(n, p) is (1 − p + peᵗ)ⁿ. With p = λ/n,

```latex
M_n(t)=\Big(1+\frac{\lambda(e^t-1)}{n}\Big)^n\to e^{\lambda(e^t-1)},
```

which is the Poisson(λ) MGF. By the continuity theorem, Xₙ → Poisson(λ) in distribution.

**Example 6.4 (extreme values, CDF method).** Let Mₙ = max(U₁, …, Uₙ) with Uᵢ iid Uniform(0, 1). We know Mₙ → 1. How far below 1 is it, on the right scale? Let Yₙ = n(1 − Mₙ). For x ≥ 0 and n > x,

```latex
P(Y_n>x)=P\Big(M_n<1-\frac{x}{n}\Big)=\Big(1-\frac{x}{n}\Big)^n\to e^{-x}.
```

So n(1 − Mₙ) → Exponential(1) in distribution. The gap between the largest sample and the top of the range is about 1/n, with an exponential shape. Notice the pattern that appears again in the CLT: Mₙ − 1 → 0, but the gap blown up by the right factor (here n) has a non-trivial limiting distribution.

**Example 6.5 (discrete becomes continuous).** Let Xₙ be uniform on the n points {1/n, 2/n, …, n/n}. For 0 ≤ x ≤ 1, Fₙ(x) = ⌊nx⌋/n, where ⌊·⌋ rounds down. Since nx − 1 < ⌊nx⌋ ≤ nx, we get x − 1/n < Fₙ(x) ≤ x, so Fₙ(x) → x. Hence Xₙ → Uniform(0, 1) in distribution, even though every Xₙ is discrete. This is why a fine quantiser's output looks continuous.

&#91;image: Left: CDFs of n(1 − Mₙ) for n = 1, 2, 5, 20 approaching the exponential CDF. Right: staircase CDFs of the discrete uniform approaching the straight-line uniform CDF\]

This is what convergence in distribution looks like: the curves Fₙ settle onto F. On the left (Example 6.4), the CDF of n(1 − Mₙ) approaches 1 − e⁻ˣ; by n = 20 the two are hard to tell apart. On the right (Example 6.5), staircases with ever finer steps approach the straight line of the Uniform(0, 1) CDF.

**Example 6.6 (wiggly densities).** Let Xₙ have density fₙ(x) = 1 − cos(2πnx) on \[0, 1\]. The densities keep oscillating between 0 and 2 and never converge. But

```latex
F_n(x)=x-\frac{\sin(2\pi n x)}{2\pi n}\to x,
```

so Xₙ → Uniform(0, 1) in distribution. The CDF integrates the wiggles away.

**Example 6.7 (same distribution, values far apart).** Let X \~ N(0, 1) and Xₙ = −X for every n. Since −X is also N(0, 1), every Xₙ has exactly the CDF of X, so Xₙ → X in distribution trivially. But |Xₙ − X| = 2|X|, which does not shrink at all. So Xₙ does **not** converge to X in probability.

*Lesson:* convergence in distribution says nothing about the values. It is a statement about histograms only.

**Example 6.8 (mass escaping to infinity: no limit at all).** Let Xₙ \~ Uniform(0, n), for example a waiting time whose possible range keeps growing. For any fixed x ≥ 0 and n > x, Fₙ(x) = x/n → 0. So the pointwise limit is G(x) = 0 for every x.

But G is not a CDF: a CDF must climb to 1 as x → ∞. All the probability has drifted off to infinity, and Xₙ does **not** converge in distribution to anything. The same happens for Xₙ = n (Fₙ(x) = 0 once n > x) and for Xₙ \~ N(0, n), where Fₙ(x) = Φ(x/√n) → 1/2 for every x: half the probability escapes to +∞ and half to −∞.

*Lesson:* finding a pointwise limit of the CDFs is not the end of the job. **Always check that the limit is a genuine CDF**, rising from 0 to 1.

**Tightness.** A sequence is called tight if no probability escapes to infinity: for every ε > 0 there is a bound M with P(|Xₙ| > M) < ε for all n. Every sequence that converges in distribution is tight. Conversely, every tight sequence has a subsequence that converges in distribution (Prokhorov's theorem, stated without proof). A practical sufficient check: if E|Xₙ| ≤ C for all n, Markov gives P(|Xₙ| > M) ≤ C/M, which is small for large M, so the sequence is tight. In Example 6.8, E\[Xₙ\] = n/2 is unbounded, and indeed the sequence is not tight.

### 6.5 The one important exception: constant limits

If the limit is a constant c, convergence in distribution and in probability are the same:

```latex
X_n\xrightarrow{d}c \iff X_n\xrightarrow{P}c.
```

The proof is in Part 7. Intuitively, a constant has no "shape" to match, so matching its distribution forces the values to pile up at c.

### 6.6 When is it used?

Whenever you **approximate a probability** using a simpler distribution:

- the CLT's normal approximation for sums, averages and estimators (Part 10),
- the Poisson approximation for counts of rare events,
- large-sample distributions of test statistics (t, chi-square) used to compute p-values and confidence intervals,
- extreme-value distributions for maximum loads, peak noise and floods.

### Summary and check yourself: Part 6

**In three lines.** Xₙ → X in distribution when Fₙ(x) → F(x) at every continuity point of F; only the shapes of the distributions are compared. Prove it with CDFs, PMFs or MGFs, and check that the limit really is a CDF (no mass escaping to infinity). It is the weakest mode, except that convergence in distribution to a constant is the same as convergence in probability.

**Exercises.**

1. A transmitter retries in time slots until the first success; each slot succeeds independently with probability λ/n, so the number of slots Gₙ is Geometric(λ/n) with P(Gₙ > k) = (1 − λ/n)ᵏ. Show Gₙ/n → Exponential(λ) in distribution.
2. Let Xₙ \~ N(μₙ, σₙ²). (a) If μₙ → μ and σₙ → σ > 0, find the limit in distribution. (b) What happens if instead σₙ → 0?
3. Let Xₙ \~ Uniform(−n, n). Does Xₙ converge in distribution?
4. Let Xₙ \~ Poisson(n), with MGF exp(n(eᵗ − 1)). Use MGFs to show (Xₙ − n)/√n → N(0, 1) in distribution.
5. Let Xₙ = n with probability 1/n and 0 otherwise. Find the limit in distribution, and compare E\[Xₙ\] with the mean of the limit.
6. Let Xₙ \~ Binomial(n, 1/2). Does Xₙ/n converge in distribution? To what?
7. (Spot the jump point.) Let Xₙ = 1/n with probability 1/2 and Xₙ = −1/n with probability 1/2. Find the limit in distribution, find every x at which Fₙ(x) fails to converge to the limit's CDF, and say whether that failure matters.

**Solutions.**

1. For x > 0, P(Gₙ/n > x) = P(Gₙ > nx) = (1 − λ/n)^{⌊nx⌋}. Write ⌊nx⌋ = nx − θ with 0 ≤ θ < 1. Then (1 − λ/n)^{nx} → e^{−λx} and (1 − λ/n)^{−θ} → 1, so P(Gₙ/n > x) → e^{−λx}, the Exponential(λ) tail. Many rare-chance slots, measured in units of n slots, give an exponential waiting time.
2. (a) Fₙ(x) = Φ((x − μₙ)/σₙ) → Φ((x − μ)/σ), since Φ is continuous. So Xₙ → N(μ, σ²). (b) For x > μ, (x − μₙ)/σₙ → +∞ and Fₙ(x) → 1; for x < μ, Fₙ(x) → 0. That is the CDF of the constant μ (its jump at μ is ignored). So Xₙ → μ in distribution, hence in probability.
3. No. For |x| ≤ n, Fₙ(x) = (x + n)/(2n) → 1/2 for every x. The constant 1/2 is not a CDF; the probability escapes to both infinities.
4. M(t) = E\[e^{t(Xₙ − n)/√n}\] = e^{−t√n}·exp(n(e^{t/√n} − 1)). Expand e^{t/√n} = 1 + t/√n + t²/(2n) + t³/(6n^{3/2}) + …, so n(e^{t/√n} − 1) = t√n + t²/2 + t³/(6√n) + …. Adding −t√n, the exponent is t²/2 + t³/(6√n) + … → t²/2, the log-MGF of N(0, 1). (This is the CLT in disguise: Poisson(n) is a sum of n iid Poisson(1) variables.)
5. Fₙ(x) = 0 for x < 0 and Fₙ(x) ≥ 1 − 1/n for x ≥ 0, so Fₙ(x) → 1 for x ≥ 0: the limit is the constant 0. But E\[Xₙ\] = 1 for every n, while the limit has mean 0. Convergence in distribution does not make means converge; Section 7.3 gives the extra condition that does.
6. Yes, to the constant 1/2: E\[Xₙ/n\] = 1/2 and Var(Xₙ/n) = 1/(4n) → 0, so Xₙ/n → 1/2 in mean square, hence in probability and in distribution.
7. |Xₙ| = 1/n for every outcome, so Xₙ → 0 surely, hence in every mode, and the limit in distribution is the constant 0, with CDF F(x) = 0 for x < 0 and F(x) = 1 for x ≥ 0. For x > 0, once 1/n ≤ x both values of Xₙ are ≤ x, so Fₙ(x) = 1 = F(x); for x < 0, once 1/n < |x| both values exceed x, so Fₙ(x) = 0 = F(x). At x = 0, however, Fₙ(0) = P(Xₙ ≤ 0) = P(Xₙ = −1/n) = 1/2 for every n, which does not converge to F(0) = 1. That is the one jump point of F, so the definition ignores it, and the failure does not matter: Xₙ → 0 in distribution, as it must. (Compare Example 6.1, where Fₙ(0) = 0; the value at the jump can be anything.)

## Part 7: How the modes relate

Almost sure and Lᵖ convergence each imply convergence in probability, which implies convergence in distribution; nothing else holds in general. The diagram below is the single most important picture in this tutorial. Memorise it, and memorise one counterexample for every missing arrow.

**Goals.** After this Part you can (1) draw the hierarchy of the four modes from memory, (2) prove each true implication, (3) give a counterexample for each false one, (4) state when a weaker mode can be upgraded, including when means converge, and (5) classify any sequence across all four modes.

&#91;embedded content: hierarchy of the four modes of convergence\]

Read the solid arrows as "always implies". Every route ends at convergence in distribution, the weakest mode; the dashed arrows go back up only under the condition written beside them.

### 7.1 The always-true implications, with proofs

**(a) Almost sure ⇒ in probability.** For a single time n, being more than ε away at time n is one particular way of being more than ε away at some time m ≥ n. So

```latex
P(|X_n-X|>\varepsilon)\le P\big(|X_m-X|>\varepsilon \text{ for some } m\ge n\big)\to 0,
```

where the limit is the "eventually stays" characterisation of Section 4.2.

**(b) Lᵖ ⇒ in probability.** Generalised Markov: P(|Xₙ − X| > ε) ≤ E|Xₙ − X|ᵖ / εᵖ → 0.

**(c) Lʳ ⇒ Lᵖ for r > p ≥ 1.** Section 5.3, Fact 2.

**(d) In probability ⇒ in distribution.** Fix a point x where F is continuous, and a small ε > 0. The plan is to trap Fₙ(x) between two numbers that are both close to F(x).

1. *Upper bound.* Suppose Xₙ ≤ x. Either X ≤ x + ε, or X > x + ε; in the second case X and Xₙ are more than ε apart. So the event {Xₙ ≤ x} is contained in {X ≤ x + ε} ∪ {|Xₙ − X| > ε}, and the union bound gives

```latex
F_n(x)\le F(x+\varepsilon)+P(|X_n-X|>\varepsilon).
```

2. *Lower bound.* Suppose X ≤ x − ε. Either Xₙ ≤ x, or Xₙ > x; in the second case they are more than ε apart. So F(x − ε) ≤ Fₙ(x) + P(|Xₙ − X| > ε), which rearranges to

```latex
F(x-\varepsilon)-P(|X_n-X|>\varepsilon)\le F_n(x).
```

3. *Let n → ∞.* The probability terms tend to 0 by assumption. So for large n, Fₙ(x) is squeezed between F(x − ε) and F(x + ε), up to an error that vanishes.
4. *Let ε → 0.* Because F is continuous at x, both F(x − ε) and F(x + ε) approach F(x). The squeeze leaves no room: Fₙ(x) → F(x). ∎

*Where continuity was used:* only in step 4. At a jump point, F(x − ε) and F(x + ε) stay apart however small ε is, and the squeeze fails, exactly as in Example 6.1.

**(e) In distribution to a constant ⇒ in probability.** Let X = c. Its CDF jumps only at c, so c − ε and c + ε are continuity points. Then

```latex
P(|X_n-c|>\varepsilon)\le P(X_n\le c-\varepsilon)+P(X_n> c+\varepsilon)=F_n(c-\varepsilon)+1-F_n(c+\varepsilon)\to 0+1-1=0.
```

### 7.2 The missing arrows: one counterexample each

Every example below was already worked out in Parts 3 to 6. Here they are collected so you can see the whole map.

| Claim that fails | Counterexample | Why it fails |
| --- | --- | --- |
| In probability ⇒ almost surely | Independent Xₙ \~ Bernoulli(1/n) (Example 4.4), or the sliding hump (Example 4.5) | P(Xₙ = 1) → 0, but every path hits 1 infinitely often |
| In probability ⇒ mean square (or L¹) | Xₙ = n with probability 1/n, else 0 (Examples 3.6 and 5.4) | A rare huge value keeps E∣Xₙ∣ = 1 |
| Almost surely ⇒ mean square (or L¹) | ω \~ Uniform(0, 1), Xₙ = n when ω < 1/n, else 0 | For each ω > 0, Xₙ(ω) = 0 once n > 1/ω, so Xₙ → 0 a.s.; but E∣Xₙ∣ = n·(1/n) = 1 |
| Mean square ⇒ almost surely | Independent Xₙ \~ Bernoulli(1/n), or the sliding hump | E\[Xₙ²\] = 1/n → 0 (hump: interval length → 0), but paths keep returning to 1 |
| L¹ ⇒ mean square | Xₙ = √n with probability 1/n, else 0 (Example 5.5) | E∣Xₙ∣ = 1/√n → 0 but E\[Xₙ²\] = 1 |
| In distribution ⇒ in probability | X \~ N(0, 1), Xₙ = −X (Example 6.7) | Same distribution, but the distance ∣Xₙ − X∣ = 2∣X∣ never shrinks |

Notice two "star" examples. Independent Bernoulli(1/n) is the standard proof that probability and mean square do not imply almost sure. Rare-but-huge spikes are the standard proof that probability and almost sure do not imply mean square.

### 7.3 Partial converses: when a weaker mode upgrades

These are the dashed arrows in the diagram. We state them without full proof. The four bullets are used throughout the tutorial; the paragraph on uniform integrability after them is more advanced and may be skipped on a first reading, as long as you keep its one-line message: never read a mean or a variance off a limit distribution without an extra condition.

- **Constant limit.** In distribution to a constant c ⇒ in probability to c (proved above in (e)).
- **Subsequence.** If Xₙ → X in probability, some subsequence Xₙ₁, Xₙ₂, … converges almost surely. Idea: pick nₖ so that P(|Xₙₖ − X| > 1/k) < 1/2ᵏ; these probabilities are summable, so Borel–Cantelli applies. In the sliding hump, the subsequence of first intervals \[0, 1/k\] of each block converges to 0 for every ω > 0.
- **Domination.** If Xₙ → X in probability (or a.s.) and |Xₙ| ≤ Y for all n with E\[Yᵖ\] < ∞, then Xₙ → X in Lᵖ. The special case |Xₙ| ≤ M (bounded) is the one engineers use most: bounded signals cannot have rare huge spikes. This is the probabilistic form of the dominated convergence theorem.
- **Fast convergence in probability.** If Σ P(|Xₙ − X| > ε) < ∞ for every ε, then convergence is almost sure (Borel–Cantelli, Section 4.3). Likewise, Σ E\[(Xₙ − X)²\] < ∞ implies almost sure convergence, by Markov plus Borel–Cantelli.

**When do means converge? Uniform integrability.** Convergence in probability or in distribution does not make E\[Xₙ\] → E\[X\] (Example 3.6, Exercise 6.5). What is missing is a guarantee that the far tails cannot carry a non-vanishing share of the mean. A sequence is **uniformly integrable** if

```latex
\lim_{M\to\infty}\ \sup_{n}\ E\big[\,|X_n|\;\mathbf{1}\{|X_n|>M\}\big]=0,
```

where 1{·} equals 1 when the event occurs and 0 otherwise, and sup over n means the largest value over all n. In words: beyond a large enough level M, the tails contribute almost nothing to the mean, uniformly in n.

*Theorem (stated without proof).* If Xₙ → X in distribution and the Xₙ are uniformly integrable, then E\[Xₙ\] → E\[X\]. If Xₙ → X in probability and the Xₙ are uniformly integrable, then Xₙ → X in L¹.

*Two practical sufficient conditions.* (i) Domination: |Xₙ| ≤ Y for all n, with E\[Y\] < ∞. (ii) A bounded higher moment: E|Xₙ|^{1+δ} ≤ C for all n and some δ > 0, for example bounded variances. For (ii), on the event |Xₙ| > M we have |Xₙ| ≤ |Xₙ|^{1+δ}/M^δ, so the tail contribution is at most C/M^δ → 0.

*Check on Example 3.6.* For Xₙ = n with probability 1/n, and any n > M, E\[|Xₙ|·1{|Xₙ| > M}\] = n·(1/n) = 1. The supremum is 1 for every M, so the sequence is not uniformly integrable, which is exactly why E\[Xₙ\] = 1 fails to converge to 0.

*Engineering reading.* Using a limiting distribution to compute a mean or a variance (for instance, reading the variance of an estimator off its CLT limit) is justified only under a condition like this. Bounded signals satisfy it automatically.

### 7.4 A decision guide for exercises

When asked "in which modes does Xₙ converge?", work through these steps in order.

1. **Guess the limit.** Look at what Xₙ does for typical outcomes, or compute E\[Xₙ\] and Var(Xₙ).
2. **Check distribution first.** Find Fₙ and its limit. If there is no limit in distribution, there is no convergence of any kind.
3. **Check probability.** Compute or bound P(|Xₙ − X| > ε). If the limit is a constant, step 2 already settles it.
4. **Check Lᵖ.** Compute E|Xₙ − X|ᵖ. Watch for rare huge values.
5. **Check almost surely.** Try the pointwise picture, then summability of P(|Xₙ − X| > ε). If the events are independent and the sum diverges, the second Borel–Cantelli lemma says it fails.
6. **Cross-check with the diagram.** Your answers must respect the arrows; for example, if you found almost sure convergence but not convergence in probability, you made a mistake.

### Summary and check yourself: Part 7

**In three lines.** Almost sure and Lᵖ convergence each imply convergence in probability, which implies convergence in distribution; no other implication holds in general. The standard counterexamples are rare large spikes (which break Lᵖ) and recurring blips such as independent Bernoulli(1/n) or the sliding hump (which break almost sure convergence). Weaker modes upgrade only under extra conditions: a constant limit, a subsequence, domination or uniform integrability.

**Classification drill.** For each sequence, decide whether it converges to the stated limit almost surely (a.s.), in probability (P), in mean square (m.s.) and in distribution (d). Here ω \~ Uniform(0, 1) is one shared outcome, and Z \~ N(0, 1). Fill in all 48 answers before looking at the key.

| # | Sequence | Candidate limit |
| --- | --- | --- |
| 1 | Xₙ = 1/n | 0 |
| 2 | Xₙ = ωⁿ | 0 |
| 3 | Xₙ = n when ω < 1/n, else 0 | 0 |
| 4 | Xₙ = √n when ω < 1/n, else 0 | 0 |
| 5 | Independent Xₙ \~ Bernoulli(1/n) | 0 |
| 6 | Independent Xₙ \~ Bernoulli(1/n²) | 0 |
| 7 | Independent Xₙ = n with probability 1/n, else 0 | 0 |
| 8 | Sliding hump (Example 4.5) | 0 |
| 9 | Xₙ \~ N(0, 1/n), with any dependence between the Xₙ | 0 |
| 10 | Xₙ = (−1)ⁿ, not random | none |
| 11 | Xₙ = (−1)ⁿZ | Z |
| 12 | Independent Xₙ \~ N(0, 1) | N(0, 1) |

**Answer key.**

| # | a.s. | P | m.s. | d | Key reason |
| --- | --- | --- | --- | --- | --- |
| 1 | yes | yes | yes | yes | Not random: ordinary convergence |
| 2 | yes | yes | yes | yes | ωⁿ → 0 for every ω < 1; E\[ω²ⁿ\] = 1/(2n + 1) → 0 |
| 3 | yes | yes | no | yes | Path is 0 once n > 1/ω; E\[Xₙ²\] = n |
| 4 | yes | yes | no | yes | Same paths; E\[Xₙ²\] = 1, though E∣Xₙ∣ = 1/√n → 0 (L¹ holds) |
| 5 | no | yes | yes | yes | Σ 1/n = ∞ with independence (BC2); E\[Xₙ²\] = 1/n |
| 6 | yes | yes | yes | yes | Σ 1/n² < ∞ (BC1); E\[Xₙ²\] = 1/n² |
| 7 | no | yes | no | yes | BC2 again; E\[Xₙ²\] = n |
| 8 | no | yes | yes | yes | Every ω is hit once per block; E\[Xₙ²\] = interval length → 0 |
| 9 | yes | yes | yes | yes | P(∣Xₙ∣ > ε) ≤ 2e^{−nε²/2} is summable, and BC1 needs no independence; E\[Xₙ²\] = 1/n |
| 10 | no | no | no | no | The numbers alternate, and the CDFs alternate between two step functions |
| 11 | no | no | no | yes | ∣Xₙ − Z∣ = 2∣Z∣ for odd n; but every Xₙ is N(0, 1) |
| 12 | no | no | no | yes | Same distribution at every n, but fresh independent values (Example 3.7 argument) |

**Self-check.** Each row must respect the hierarchy: a "yes" in a.s. or m.s. forces "yes" in P and d. If one of your rows violates this, find the mistake before reading on.

**Proof exercise (the subsequence upgrade of Section 7.3).** Suppose Xₙ → X in probability. Prove, using only the definition and BC1, that there is a subsequence n₁ < n₂ < n₃ < … along which Xₙₖ → X almost surely.

*Solution.* Fix k. Because P(|Xₙ − X| > 1/k) → 0 as n → ∞, there is an index beyond which this probability is below 2⁻ᵏ; choose nₖ to be such an index, and larger than nₖ₋₁ so that the subsequence increases. Let Aₖ = {|Xₙₖ − X| > 1/k}. Then Σₖ P(Aₖ) < Σₖ 2⁻ᵏ = 1 < ∞, so by BC1, with probability 1 only finitely many Aₖ occur. For such an outcome ω there is a K with |Xₙₖ(ω) − X(ω)| ≤ 1/k for all k ≥ K, and since 1/k → 0 this says Xₙₖ(ω) → X(ω). That is almost sure convergence along the subsequence. No independence was used, which is why the result holds for every sequence; the full sequence may still fail, as the sliding hump shows.

## Part 8: The Weak Law of Large Numbers

The Weak Law of Large Numbers (WLLN) says the sample mean converges in probability to the true mean. It is the mathematical reason averaging works, and the reason a probability can be read as a long-run frequency.

**Goals.** After this Part you can (1) state and prove the Weak Law of Large Numbers with Chebyshev, (2) use it to size an experiment, (3) apply it to functions of the data such as squares, and (4) spot when it fails: common-mode error, growing variances and heavy tails.

### 8.1 Statement

**Theorem (WLLN, finite variance version).** Let X₁, X₂, … be uncorrelated (independent is enough), each with mean μ and variance σ² < ∞. Then

```latex
\bar X_n=\frac{X_1+\cdots+X_n}{n}\xrightarrow{P}\mu \qquad\text{(and also } \bar X_n\xrightarrow{\text{m.s.}}\mu\text{)}.
```

**Proof.** By Section 2.3, E\[X̄ₙ\] = μ and Var(X̄ₙ) = σ²/n. Chebyshev's inequality gives

```latex
P(|\bar X_n-\mu|>\varepsilon)\le\frac{\sigma^2}{n\varepsilon^2}\to 0.
```

That is the entire proof: three lines, using only the variance of a sum.

**Stronger version (Khinchin).** If the Xᵢ are iid, the conclusion holds as long as E|X₁| < ∞, that is, the mean μ = E\[X₁\] exists and is finite, even if the variance is infinite. This is exactly the assumption of the Strong Law in Part 9. The proof needs a truncation argument and is beyond our scope.

### 8.2 What the WLLN means, and what it does not mean

**Frequency interpretation.** Let A be an event with probability p, and let Xᵢ = 1 if A occurs on trial i and 0 otherwise. Then X̄ₙ is the relative frequency of A in n trials, and the WLLN says it converges in probability to p. This connects the abstract number P(A) with what you measure in the lab.

**Not the gambler's fallacy.** The WLLN does not say that after many heads, tails become more likely to "balance things out". The coins have no memory. What happens is dilution, not compensation. The total excess Sₙ − nμ typically grows like σ√n. Dividing by n shrinks it like σ/√n. Early runs of luck are not cancelled; they are simply swamped by later data.

**Fixed n, not the whole path.** The WLLN is a statement about each fixed n. It does not by itself say that your running average, once close, will stay close forever. That stronger statement is the Strong Law (Part 9).

### 8.3 Worked examples

**Example 8.1 (how many samples, by Chebyshev).** A polling engineer wants to estimate the fraction p of defective units, so that P(|p̂ − p| ≥ 0.05) ≤ 0.05. Here p̂ = X̄ₙ with Xᵢ \~ Bernoulli(p), so Var(p̂) = p(1 − p)/n ≤ 1/(4n), since p(1 − p) is at most 1/4 (at p = 1/2). Chebyshev gives

```latex
P(|\hat p-p|\ge 0.05)\le\frac{1}{4n(0.05)^2}=\frac{100}{n}\le 0.05 \iff n\ge 2000.
```

Part 10 redoes this with the CLT and needs only 385 units. Chebyshev is safe for any distribution, but it pays for that generality with a much larger sample.

**Example 8.2 (non-identical but independent).** A sensor network has n sensors measuring the same temperature μ. Sensor i has noise variance σᵢ² ≤ 4 °C², all different, all unbiased and independent. Does the plain average converge to μ?

*Solution.* E\[X̄ₙ\] = μ, and Var(X̄ₙ) = (σ₁² + … + σₙ²)/n² ≤ 4n/n² = 4/n → 0. By Chebyshev, X̄ₙ → μ in probability. Identical distributions are not needed; bounded variances and uncorrelated errors are enough.

**Example 8.3 (correlated noise can break it).** Suppose every sensor shares a common calibration error B with variance τ² > 0, so Xᵢ = μ + B + Nᵢ, where the Nᵢ are independent with variance σ². Then X̄ₙ = μ + B + N̄ₙ, and N̄ₙ → 0 in probability, so X̄ₙ → μ + B, not μ. Var(X̄ₙ) = τ² + σ²/n → τ², not 0. *Lesson:* averaging removes independent noise, never common-mode error. Calibrate first.

**Example 8.4 (when the law fails: Cauchy noise).** Let X₁, X₂, … be iid standard Cauchy, with density 1/(π(1 + x²)). This distribution has no mean: E|X| = ∞. A remarkable fact is that X̄ₙ is again standard Cauchy, for every n: by the Fourier rules of Section 2.7, the characteristic function of X̄ₙ = (X₁ + … + Xₙ)/n is \[φ(t/n)\]ⁿ = \[e^{−|t|/n}\]ⁿ = e^{−|t|}, identical to that of a single sample. Averaging a million Cauchy samples is no better than using one! The WLLN's assumptions fail, and so does its conclusion. Heavy-tailed impulsive noise (lightning, some radio interference) can behave this way, which is why robust estimators like the median are used.

**Example 8.5 (estimating a second moment).** Let X₁, X₂, … be iid with E\[X⁴\] < ∞. Then Yᵢ = Xᵢ² are iid with mean E\[X²\] and finite variance, so by the WLLN

```latex
\frac1n\sum_{i=1}^n X_i^2\xrightarrow{P}E[X^2].
```

Applied to a zero-mean noise signal, this says the measured average power converges to the true power. The same trick (apply the WLLN to a function of the data) works for any moment.

### Summary and check yourself: Part 8

**In three lines.** For uncorrelated data with common mean μ and bounded variances, X̄ₙ → μ in probability and in mean square, by a three-line Chebyshev argument. Averaging works by dilution of early luck, not by compensation. It fails when errors are shared across samples, when variances grow too fast, or when the mean does not exist.

**Exercises.**

1. Bulb lifetimes have mean μ (unknown) and standard deviation 200 h. How many bulbs must be tested so that P(|X̄ − μ| ≥ 20 h) ≤ 0.05, (a) by Chebyshev, (b) by the CLT approximation of Part 10?
2. X₁, X₂, … are iid Uniform(0, 1). Find the limits in probability of (a) (1/n)ΣXᵢ², (b) (1/n)Σcos(2πXᵢ).
3. X₁, X₂, … are independent with common mean μ and Var(Xᵢ) = i. Does the Chebyshev proof of the WLLN work? If the Xᵢ are also normal, does X̄ₙ → μ in probability?
4. Repeat Exercise 3 with Var(Xᵢ) = √i.
5. A fair coin shows 520 heads in its first 1000 tosses. What is the expected number of heads after a total of 10⁶ tosses, and the expected fraction? What does this say about "the law of averages"?
6. In Example 8.3, take a calibration error with τ = 0.1 °C and independent noise with σ = 1 °C. Compute the standard deviation of the average for n = 100 and as n → ∞.

**Solutions.**

1. (a) Chebyshev: σ²/(nε²) = 200²/(n·20²) = 100/n ≤ 0.05 gives n ≥ 2000. (b) CLT: 1.96·200/√n ≤ 20 gives √n ≥ 19.6, so n ≥ 385.
2. (a) Yᵢ = Xᵢ² are iid with mean ∫₀¹ x² dx = 1/3 and finite variance, so the average → 1/3. (b) The mean is ∫₀¹ cos(2πx) dx = 0, so the average → 0. Any function g with E\[g(X)²\] < ∞ can be averaged this way.
3. Var(X̄ₙ) = (1 + 2 + … + n)/n² = (n + 1)/(2n) → 1/2, not 0, so Chebyshev gives nothing. If the Xᵢ are normal, X̄ₙ \~ N(μ, (n + 1)/(2n)), and P(|X̄ₙ − μ| > ε) → P(|N(0, 1/2)| > ε) > 0. The law fails: later, noisier samples swamp the average.
4. Σᵢ≤ₙ √i ≤ n·√n, so Var(X̄ₙ) ≤ n^{3/2}/n² = 1/√n → 0, and Chebyshev proves X̄ₙ → μ in probability. Variances may grow, as long as their sum grows more slowly than n².
5. The remaining 999,000 tosses are expected to give 499,500 heads, so the expected total is 500,020, an expected fraction of 0.50002. The excess of 20 heads is not cancelled (it is still expected to be +20); it is diluted from 2% of the tosses to 0.002%.
6. Var(X̄ₙ) = τ² + σ²/n = 0.01 + 0.01 = 0.02 for n = 100, a standard deviation of 0.141 °C. As n → ∞ the standard deviation falls only to τ = 0.1 °C. Beyond about n = 100, more samples barely help; better calibration does.

## Part 9: The Strong Law of Large Numbers

The Strong Law of Large Numbers (SLLN) says that the running average you actually observe converges to the true mean with probability one. This Part first puts the Borel–Cantelli lemmas of Section 4.3 to work on three engineering examples, then uses them to prove the SLLN.

**Goals.** After this Part you can (1) apply BC1 and BC2 to rare-event and peak-noise questions, (2) state the SLLN and explain how it differs from the weak law, (3) follow the fourth-moment proof and say why Chebyshev alone is not enough, and (4) decide which law justifies a given engineering claim.

### 9.1 Borel–Cantelli in action: worked examples

**Example 9.1 (the Bernoulli pair, settled).** Independent Xₙ \~ Bernoulli(pₙ), events Aₙ = {Xₙ = 1}.

- pₙ = 1/n²: Σ 1/n² = π²/6 < ∞, so by BC1 only finitely many 1s appear, and Xₙ → 0 almost surely.
- pₙ = 1/n: Σ 1/n = ∞ and the events are independent, so by BC2 infinitely many 1s appear with probability 1, and Xₙ does not converge to 0 almost surely, even though it does in probability.

**Example 9.2 (a borderline case).** Independent events with P(Aₙ) = 1/(n ln n) for n ≥ 2. By the integral test, ∫ dx/(x ln x) = ln(ln x) → ∞, so the sum diverges and BC2 gives P(Aₙ i.o.) = 1. With P(Aₙ) = 1/(n (ln n)²) for n ≥ 3 instead (at n = 2 this formula exceeds 1), ∫ dx/(x (ln x)²) = −1/ln x converges, so BC1 gives P(Aₙ i.o.) = 0.

**Example 9.3 (how high does noise peak?).** Let X₁, X₂, … be iid Exponential(1), a model for the power of independent noise samples. How large do the peaks get? For a constant c > 0,

```latex
P(X_n>c\ln n)=e^{-c\ln n}=\frac{1}{n^c}.
```

If c > 1 the sum converges, so by BC1 the noise exceeds c ln n only finitely often. If c = 1 the sum is Σ 1/n = ∞, and by BC2 the noise exceeds ln n infinitely often. Conclusion: with probability 1, the peak noise grows like ln n, so Xₙ/ln n has largest limit point exactly 1. *Engineering meaning:* in a long record, the worst noise sample keeps growing, but only logarithmically slowly. Detection thresholds for long observation windows must be raised accordingly.

### 9.2 The Strong Law of Large Numbers

**Theorem (SLLN, Kolmogorov).** Let X₁, X₂, … be iid with E|X₁| < ∞ and mean μ. Then

```latex
\bar X_n\xrightarrow{\text{a.s.}}\mu, \qquad\text{that is,}\qquad P\Big(\lim_{n\to\infty}\bar X_n=\mu\Big)=1.
```

**Why Chebyshev is not enough.** Chebyshev gives P(|X̄ₙ − μ| > ε) ≤ σ²/(nε²). The sum Σ 1/n diverges, so BC1 cannot be applied. We need a bound that decreases faster than 1/n.

**Proof sketch under a finite fourth moment.** Assume additionally E\[X₁⁴\] < ∞, and without loss of generality μ = 0 (otherwise subtract μ from every Xᵢ). Write Sₙ = X₁ + … + Xₙ.

*Strategy.* We need a bound on P(|X̄ₙ| > ε) that is summable in n, so that BC1 applies. Chebyshev uses the second moment and gives a bound of order 1/n, which is not summable. Using the fourth moment instead gives order 1/n², which is. The real work is showing that E\[Sₙ⁴\] grows only like n², not like n⁴.

*Step 1: compute E\[Sₙ⁴\].* Expand (X₁ + … + Xₙ)⁴ into a sum of products XᵢXⱼXₖXₗ. Because the Xᵢ are independent with mean 0, any product in which some index appears exactly once has expectation 0 (for example E\[X₁³X₂\] = E\[X₁³\]·E\[X₂\] = 0). Only two kinds of terms survive:

- Xᵢ⁴: there are n of them, each with expectation E\[X₁⁴\].
- Xᵢ²Xⱼ² with i ≠ j: each unordered pair {i, j} appears 4!/(2!·2!) = 6 times in the expansion, and there are n(n − 1)/2 pairs, so 3n(n − 1) terms, each with expectation σ⁴.

```latex
E[S_n^4]=n\,E[X_1^4]+3n(n-1)\sigma^4\le C n^2 \quad\text{for a constant } C.
```

*Check the count with n = 2.* (X₁ + X₂)⁴ = X₁⁴ + 4X₁³X₂ + 6X₁²X₂² + 4X₁X₂³ + X₂⁴. The terms 4X₁³X₂ and 4X₁X₂³ have expectation 0, leaving 2E\[X₁⁴\] + 6σ⁴, which is the formula with n = 2. For the bound, n E\[X₁⁴\] + 3n(n − 1)σ⁴ ≤ n²(E\[X₁⁴\] + 3σ⁴), so C = E\[X₁⁴\] + 3σ⁴ works.

*Step 2: fourth-moment Markov bound.*

```latex
P(|\bar X_n|>\varepsilon)=P(|S_n|>n\varepsilon)\le\frac{E[S_n^4]}{n^4\varepsilon^4}\le\frac{C}{n^2\varepsilon^4}.
```

*Step 3: Borel–Cantelli.* Σ C/(n²ε⁴) < ∞, so by BC1, |X̄ₙ| > ε happens only finitely often, with probability 1. Since this holds for every ε (take ε = 1, 1/2, 1/3, …, countably many), X̄ₙ → 0 almost surely. ∎

*Why it works.* The Chebyshev bound σ²/(nε²) and the fourth-moment bound C/(n²ε⁴) both tend to 0, so both prove the weak law. Only the second is summable, and summability is exactly what BC1 needs for the strong law. A stronger moment assumption buys a faster-decaying tail bound.

The general proof with only E|X₁| < ∞ uses truncation and a clever subsequence argument (Etemadi's proof); Practice Problem 14 lets you try the subsequence idea yourself.

### 9.3 Weak versus strong law, side by side

|  | Weak law | Strong law |
| --- | --- | --- |
| Mode | In probability | Almost surely |
| Statement in words | At any fixed large n, the average is very likely close to μ | With probability 1, the whole path of averages eventually enters every ε-band around μ and stays there |
| Typical proof | Chebyshev (3 lines) | Fourth moment + Borel–Cantelli, or truncation |
| Minimal iid condition | E∣X₁∣ < ∞, a finite mean (Khinchin) | E∣X₁∣ < ∞, the same finite mean (Kolmogorov) |
| What it justifies | "Most experimenters' averages are accurate at time n" | "My single long simulation run converges" |

For iid data with a finite mean, both laws hold. Notice that the two laws need exactly the same iid assumption, E|X₁| < ∞: the strong law is not bought with a stronger hypothesis, only with a harder proof. The strong law is the stronger promise, as the hierarchy diagram of Part 7 says it must be. (For completeness: a weak law can survive even without a mean, under the weaker tail condition n·P(|X₁| > n) → 0, provided the average is centred at E\[X₁·1{|X₁| ≤ n}\] instead of at μ; we never need this refinement.)

### Summary and check yourself: Part 9

**In three lines.** BC1 and BC2 answer "how often?" questions about rare events, such as how high the noise peaks over a long record. The SLLN says X̄ₙ → μ almost surely for iid data with a finite mean, so a single long run converges. Its simplest proof combines a summable fourth-moment bound with BC1.

**Exercises.**

1. X₁, X₂, … are iid Uniform(−1, 1). Hoeffding's inequality gives P(|X̄ₙ| ≥ ε) ≤ 2e^{−nε²/2}. Use it to prove X̄ₙ → 0 almost surely without the SLLN.
2. Noise samples X₁, X₂, … are iid N(0, 1), and P(Z > t) ≤ e^{−t²/2} for t ≥ 0. Show that for any c > 1, with probability 1, Xₙ ≤ c√(2 ln n) for all large n. About how large is the biggest of 10⁶ noise samples, in units of σ?
3. Show that the relative frequency of an event A in independent repetitions converges to P(A) almost surely.
4. A fair coin is tossed n times, with Sₙ heads. The SLLN gives Sₙ/n → 1/2. Does Sₙ − n/2 → 0? About how large is ∣Sₙ − n/2∣ typically for n = 10⁶?
5. Monte Carlo estimates I = ∫₀¹ u^{−1/2} du = 2 by averaging g(Uᵢ) = Uᵢ^{−1/2}. Does a single run converge almost surely? Is the usual error bar 1.96σ/√n valid?

**Solutions.**

1. Σ 2e^{−nε²/2} is a geometric series with ratio e^{−ε²/2} < 1, so it converges for every ε > 0. By BC1, |X̄ₙ| ≥ ε only finitely often, so X̄ₙ → 0 almost surely. Any summable tail bound proves a strong law.
2. P(Xₙ > c√(2 ln n)) ≤ e^{−c² ln n} = n^{−c²}, which is summable for c > 1. By BC1, only finitely many Xₙ exceed c√(2 ln n). For n = 10⁶, √(2 ln 10⁶) = √27.6 ≈ 5.3, which is an upper envelope that the peak eventually stays below. The typical peak is somewhat lower: the expected largest of a million Gaussian samples is about 4.9σ (E\[max\] ≈ 4.86σ), because √(2 ln n) overshoots by a correction of order ln(ln n)/√(ln n) that vanishes only slowly. Either way the peak grows like √(ln n): a thousand times more samples raise it by only about 1σ.
3. Let Yᵢ = 1 if A occurs on trial i and 0 otherwise. The Yᵢ are iid with E|Yᵢ| ≤ 1 < ∞ and mean P(A), so the SLLN gives (Y₁ + … + Yₙ)/n → P(A) almost surely.
4. No. Sₙ − n/2 = n(Sₙ/n − 1/2) is a small error multiplied by a large n. By the CLT (Part 10) it has standard deviation √n/2, so for n = 10⁶ it is typically around 500 heads, and it does not shrink. The fraction converges; the count difference does not.
5. E|g(U)| = ∫₀¹ u^{−1/2} du = 2 < ∞, so the SLLN applies and a single run converges almost surely. But E\[g(U)²\] = ∫₀¹ u^{−1} du = ∞: the variance is infinite, the CLT does not apply, and σ/√n error bars are meaningless. Convergence is guaranteed but slow and erratic; a change of variables to remove the singularity is the engineering fix.

## Part 10: The Central Limit Theorem

The Central Limit Theorem (CLT) says that the error of a sample mean, magnified by √n, converges in distribution to a normal distribution, whatever the shape of the original data. The Law of Large Numbers tells you where the average goes; the CLT tells you how its error is distributed on the way.

**Goals.** After this Part you can (1) state the CLT precisely and explain the √n scaling, (2) follow its MGF proof, (3) use it to approximate probabilities and choose sample sizes, with a continuity correction for integer data, and (4) judge when the approximation is trustworthy.

### 10.1 Statement

**Theorem (CLT, iid version).** Let X₁, X₂, … be iid with mean μ and variance 0 < σ² < ∞. Then

```latex
Z_n=\frac{\bar X_n-\mu}{\sigma/\sqrt n}=\frac{S_n-n\mu}{\sigma\sqrt n}\xrightarrow{d}N(0,1),
\qquad\text{i.e.}\qquad
P(Z_n\le z)\to\Phi(z)\ \text{ for every real } z.
```

Since Φ is continuous everywhere, the convergence holds at every z. Equivalent forms that you will see:

```latex
\sqrt n(\bar X_n-\mu)\xrightarrow{d}N(0,\sigma^2), \qquad \bar X_n\ \approx\ N\Big(\mu,\frac{\sigma^2}{n}\Big), \qquad S_n\ \approx\ N(n\mu,\ n\sigma^2).
```

The first is the precise limit statement. The last two are the practical approximations for large n (the symbol ≈ means "approximately distributed as"; a sequence cannot converge to a distribution that depends on n).

### 10.2 Three scales: why √n?

Look at the error X̄ₙ − μ multiplied by different powers of n:

| Quantity | Behaviour as n → ∞ | Theorem |
| --- | --- | --- |
| X̄ₙ − μ | Converges to 0 | Law of Large Numbers |
| √n(X̄ₙ − μ) | Stays random, with a non-degenerate N(0, σ²) limit | Central Limit Theorem |
| n(X̄ₙ − μ) | Spreads out without limit (its variance nσ² → ∞) | No limit |

The variance of X̄ₙ − μ is σ²/n, so multiplying by √n is exactly what keeps the variance at σ². Any slower magnification gives 0, any faster gives an explosion. The CLT says that at the one "just right" scale, the shape is always Gaussian.

### 10.3 Proof sketch with moment generating functions

Assume the MGF of X₁ exists near 0 (the general proof replaces M by the characteristic function of Section 2.7 and is otherwise identical), and standardise so that μ = 0 and σ = 1 (replace Xᵢ by (Xᵢ − μ)/σ). Then Zₙ = Sₙ/√n and, by independence,

```latex
M_{Z_n}(t)=E\big[e^{tS_n/\sqrt n}\big]=\Big[M\Big(\frac{t}{\sqrt n}\Big)\Big]^n.
```

By Section 2.7 with E\[X\] = 0 and E\[X²\] = 1, M(s) = 1 + s²/2 + o(s²). Put s = t/√n:

```latex
M_{Z_n}(t)=\Big[1+\frac{t^2}{2n}+o\Big(\frac1n\Big)\Big]^n\longrightarrow e^{t^2/2},
```

by the limit (1 + x/n + o(1/n))ⁿ → eˣ from Section 2.6. Since e^{t²/2} is the MGF of N(0, 1), the continuity theorem gives Zₙ → N(0, 1) in distribution. ∎

Notice what the proof used: only the mean and the variance survive in the limit. Every higher-order feature of the original distribution (skewness, bumps, discreteness) is washed out. That is why the normal distribution is universal.

### 10.4 The CLT is only convergence in distribution

Zₙ does not converge in probability to anything. To see why, compare Zₙ with Z₂ₙ, which shares the first n observations with it. Write the second block of n observations as its own standardised sum, Z′ₙ = (Xₙ₊₁ + … + X₂ₙ − nμ)/(σ√n). It is independent of Zₙ (different observations) and, by the CLT, also converges in distribution to N(0, 1). Since S₂ₙ − 2nμ = (Sₙ − nμ) + (Xₙ₊₁ + … + X₂ₙ − nμ),

```latex
Z_{2n}=\frac{S_{2n}-2n\mu}{\sigma\sqrt{2n}}=\frac{Z_n+Z'_n}{\sqrt2},
\qquad\text{so}\qquad
Z_n-Z_{2n}=\Big(1-\frac{1}{\sqrt2}\Big)Z_n-\frac{1}{\sqrt2}\,Z'_n .
```

Because Zₙ and Z′ₙ are independent, the pair (Zₙ, Z′ₙ) converges jointly in distribution to a pair (U, V) of independent N(0, 1) variables (the joint CDF is the product of the two marginal CDFs, each of which converges; see also Section 11.2). The continuous mapping theorem applied to the function (u, v) ↦ (1 − 1/√2)u − v/√2 then gives

```latex
Z_n-Z_{2n}\xrightarrow{d}N\Big(0,\ \big(1-\tfrac{1}{\sqrt2}\big)^2+\tfrac12\Big)=N\big(0,\ 2-\sqrt2\big)\approx N(0,\,0.59).
```

Now suppose Zₙ converged in probability to some random variable Z. Then Zₙ − Z₂ₙ would converge in probability to Z − Z = 0 (Problem 6 of Part 14), hence in distribution to the constant 0. But we just showed it converges in distribution to a non-degenerate normal, and a sequence has at most one limit in distribution. Contradiction. In particular Corr(Zₙ, Z₂ₙ) = Cov(Sₙ, S₂ₙ)/(σ²n√2) = 1/√2 ≈ 0.71 for every n: each path stays only partly tied to its own past, forever.

*A caution about the argument.* It was essential to produce a non-degenerate limit distribution. Observing only that Var(Zₙ − Z₂ₙ) = 2 − √2 does not go to 0 would prove nothing, because convergence in probability does not control variances (Example 3.6: Xₙ → 0 in probability with Var(Xₙ) → ∞).

So the histogram of Zₙ settles down to a bell curve, while each individual path of Zₙ keeps wandering forever. This is the clearest real-world example of convergence in distribution without convergence in probability.

### 10.5 Worked examples

**Example 10.1 (sum of dice).** Roll 100 fair dice. One die has mean 3.5 and variance 35/12 ≈ 2.917. Find P(S ≥ 380).

*Solution.* S has mean 350 and variance 100 × 35/12 ≈ 291.7, so standard deviation ≈ 17.08. S is integer-valued, so use a continuity correction (treat the integer 380 as the interval \[379.5, 380.5\]):

```latex
P(S\ge 380)\approx P\Big(Z\ge\frac{379.5-350}{17.08}\Big)=P(Z\ge 1.727)\approx 1-0.958=0.042.
```

**Example 10.2 (normal approximation to the binomial).** X \~ Binomial(100, 0.5) has mean 50 and standard deviation 5. Then

```latex
P(X\le 45)\approx\Phi\Big(\frac{45.5-50}{5}\Big)=\Phi(-0.9)\approx 0.184,
```

which matches the exact binomial value 0.184 to three decimals. Without the continuity correction you would get Φ(−1) ≈ 0.159, noticeably worse.

**Example 10.3 (polling sample size, compare with Example 8.1).** We want P(|p̂ − p| ≤ 0.05) ≈ 0.95. By the CLT, p̂ ≈ N(p, p(1 − p)/n), so we need 1.96·√(p(1 − p)/n) ≤ 0.05. Using the worst case p(1 − p) = 1/4,

```latex
n\ge\Big(\frac{1.96}{0.05}\Big)^2\cdot\frac14=384.2 \quad\Rightarrow\quad n=385.
```

Chebyshev required 2000. The CLT exploits the Gaussian shape, so it needs about five times fewer samples, at the price of being an approximation.

**Example 10.4 (tolerance stacking).** Twenty-five resistors are connected in series. Each has mean 100 Ω and standard deviation 2 Ω, with an unknown (perhaps uniform) distribution, independently. The total has mean 2500 Ω and standard deviation √25 × 2 = 10 Ω. The probability that the total lies within 2500 ± 20 Ω is approximately P(|Z| ≤ 2) ≈ 0.954. This "statistical tolerancing" is far less pessimistic than the worst case, which would add all 25 tolerances in the same direction.

**Example 10.5 (Example 2.1 revisited).** The average of 100 readings with σ = 2 Ω has standard error 0.2 Ω, so P(|X̄ − μ| ≥ 0.5) ≈ P(|Z| ≥ 2.5) ≈ 0.012. Chebyshev's bound was 0.16.

### 10.6 How large must n be? Caveats

- **The rate.** The Berry–Esseen theorem bounds the worst CDF error: sup over z of |P(Zₙ ≤ z) − Φ(z)| ≤ C·ρ/(σ³√n), where ρ = E|X − μ|³ and C is a universal constant known to be below 0.48 (the best proven value has been lowered several times; C = 0.48 is safe to use). The error shrinks like 1/√n, and skewed or heavy-tailed data need larger n.
- **The rule of thumb n ≥ 30** is only a rough guide. Symmetric data (uniform) look normal already at n ≈ 5 to 10. Very skewed data (exponential, lognormal, rare-event Bernoulli) may need hundreds. For proportions, a common check is np ≥ 10 and n(1 − p) ≥ 10.
- **Finite variance is essential.** For Cauchy data (Example 8.4) the CLT fails completely.
- **Tails are the weakest part.** The CLT approximates central probabilities well, but relative errors for very small tail probabilities (10⁻⁶, typical of bit error rates) can be huge. Use exact methods or large-deviation bounds there.
- **Generalisations.** The Lindeberg–Feller and Lyapunov CLTs allow independent but non-identical terms, provided no single term dominates the sum. The multivariate CLT gives a joint normal limit for vectors of averages.

### Summary and check yourself: Part 10

**In three lines.** For iid data with finite variance, √n(X̄ₙ − μ) → N(0, σ²) in distribution, so X̄ₙ ≈ N(μ, σ²/n) and Sₙ ≈ N(nμ, nσ²) for large n. Only the mean and variance survive in the limit, which is why the normal law is universal. The approximation is convergence in distribution only, improves like 1/√n, and is weakest for skewed data and far tails.

**Exercises.**

1. A DSP sums 48 values, each rounded to the nearest integer; the rounding errors are independent Uniform(−0.5, 0.5). Approximate the probability that the total rounding error is within ±3.
2. A link sends 10,000 bits, each in error independently with probability 0.01. Approximate P(more than 120 errors).
3. Single voltage readings have σ = 0.5 V. How many readings give a 95% interval for the mean of half-width at most 0.1 V?
4. S is the sum of n iid Exponential(1) lifetimes. Use the CLT to approximate (a) P(S > 15) for n = 10, (b) P(S > 110) for n = 100. The exact values are 0.0699 and 0.1583. What do you conclude?
5. Show that Var(√n(X̄ₙ − μ)) = σ² for every n, and explain why this makes √n the right scale.
6. Data with σ = 1.5 are averaged over n = 400 samples. Approximate P(|X̄ − μ| ≤ 0.1).

**Solutions.**

1. Each error has mean 0 and variance 1/12, so the total has mean 0 and variance 48/12 = 4, standard deviation 2. P(|S| ≤ 3) ≈ P(|Z| ≤ 1.5) ≈ 0.866.
2. The count X has mean 100 and standard deviation √(10,000 × 0.01 × 0.99) ≈ 9.95. P(X > 120) = P(X ≥ 121) ≈ P(Z ≥ (120.5 − 100)/9.95) = P(Z ≥ 2.06) ≈ 0.020.
3. 1.96 × 0.5/√n ≤ 0.1 gives √n ≥ 9.8, so n ≥ 96.04, that is n = 97 readings.
4. (a) Mean 10, standard deviation √10 ≈ 3.16: P(S > 15) ≈ P(Z > 1.58) ≈ 0.057, against the exact 0.070, a relative error of about 19%. (b) Mean 100, standard deviation 10: P(S > 110) ≈ P(Z > 1) ≈ 0.159, against 0.158. The exponential is right-skewed, so its upper tail is heavier than the normal one; the CLT underestimates it at n = 10 and becomes accurate by n = 100.
5. Var(√n(X̄ₙ − μ)) = n·Var(X̄ₙ) = n·σ²/n = σ². Any smaller power of n would send the variance to 0 (LLN territory), and any larger power would send it to ∞. Only √n keeps a non-trivial limit.
6. The standard error is 1.5/√400 = 0.075, so P(|X̄ − μ| ≤ 0.1) ≈ P(|Z| ≤ 1.33) ≈ 0.82.

## Part 11: Continuous mapping, Slutsky's theorem and the delta method

These three results let you transfer the LLN and the CLT from the sample mean to almost any statistic you build from it: ratios, square roots, studentised statistics, and nonlinear functions of estimates. They are the everyday tools of applied statistics and of engineering error propagation.

**Goals.** After this Part you can (1) pass limits through continuous functions, (2) combine a sequence converging in distribution with one converging to a constant (Slutsky), (3) justify the large-sample t-interval, and (4) use the delta method to propagate errors through nonlinear formulas, and recognise when it breaks down.

### 11.1 The continuous mapping theorem (CMT)

**Theorem.** Let g be a continuous function (it is enough that g is continuous at every point where X can land, with probability 1). Then convergence passes through g, in each of the three modes:

```latex
X_n\xrightarrow{\text{a.s.}}X\Rightarrow g(X_n)\xrightarrow{\text{a.s.}}g(X),\qquad
X_n\xrightarrow{P}X\Rightarrow g(X_n)\xrightarrow{P}g(X),\qquad
X_n\xrightarrow{d}X\Rightarrow g(X_n)\xrightarrow{d}g(X).
```

**Why it is true (almost sure case).** For each good outcome ω, the numbers Xₙ(ω) converge to X(ω), and a continuous function of a convergent number sequence converges: g(Xₙ(ω)) → g(X(ω)). That is ordinary calculus.

**Warning.** The CMT does not hold for Lᵖ convergence without extra conditions. For example, Xₙ → X in mean square does not imply Xₙ² → X² in mean square, because squaring can create rare huge values.

**Example 11.1.** By the LLN, X̄ₙ → μ. By the CMT, X̄ₙ² → μ², e^{X̄ₙ} → e^μ, and 1/X̄ₙ → 1/μ if μ ≠ 0 (all in probability, and almost surely by the SLLN).

**Example 11.2.** By the CLT, Zₙ → Z \~ N(0, 1) in distribution. With g(x) = x², Zₙ² → Z² in distribution, and Z² has the chi-square distribution with 1 degree of freedom. So n(X̄ₙ − μ)²/σ² is approximately χ²₁ for large n.

### 11.2 Combining two sequences: a trap

For convergence in probability (or almost surely), sums and products behave as you expect: if Xₙ → X and Yₙ → Y, then Xₙ + Yₙ → X + Y and XₙYₙ → XY.

For convergence in distribution, this is **false** in general. Knowing the two marginal distributions tells you nothing about how Xₙ and Yₙ are related.

**Example 11.3.** Let Z \~ N(0, 1), Xₙ = Z and Yₙ = −Z. Both converge in distribution to N(0, 1). If sums behaved nicely we would expect Xₙ + Yₙ → N(0, 2), but Xₙ + Yₙ = 0 for every n.

**Joint versus marginal convergence.** What repairs this is a statement about the *pair*. Say the random vector (Xₙ, Yₙ) converges in distribution to (X, Y) when the joint CDFs P(Xₙ ≤ x, Yₙ ≤ y) converge to P(X ≤ x, Y ≤ y) at every continuity point of the limit. Then Xₙ + Yₙ → X + Y and XₙYₙ → XY in distribution, by the continuous mapping theorem applied to the continuous functions (x, y) ↦ x + y and (x, y) ↦ xy. Marginal convergence of Xₙ alone and of Yₙ alone does not give joint convergence: in Example 11.3 the pair (Z, −Z) lives on the line y = −x, while the pair you would guess from the marginals, two independent N(0, 1) variables, fills the whole plane. Joint convergence does follow from the marginals in two important cases: when Xₙ and Yₙ are independent for every n (the joint CDF is then the product of the marginal CDFs, and products of convergent sequences converge), and when one of the two sequences converges to a constant, which is Slutsky's theorem below. Keep this rule in mind whenever you add or multiply two limits in distribution: ask first why the pair converges.

Slutsky's theorem identifies the important case where combining does work: when one of the two sequences converges to a **constant**.

### 11.3 Slutsky's theorem

**Theorem.** If Xₙ → X in distribution and Yₙ → c in probability, where c is a constant, then

```latex
X_n+Y_n\xrightarrow{d}X+c,\qquad Y_nX_n\xrightarrow{d}cX,\qquad \frac{X_n}{Y_n}\xrightarrow{d}\frac{X}{c}\ \ (c\ne 0).
```

**Intuition.** A sequence that converges to a constant becomes, in the limit, a non-random number. Adding or multiplying by a nearly fixed number cannot destroy the shape of a distribution; it only shifts or rescales it. The constant limit removes the dependence problem of Example 11.3.

**Example 11.4 (the large-sample t-statistic, the most important application).** In practice σ is unknown, so we replace it with the sample standard deviation Sₙ and form

```latex
T_n=\frac{\bar X_n-\mu}{S_n/\sqrt n}=\underbrace{\frac{\bar X_n-\mu}{\sigma/\sqrt n}}_{Z_n\ \xrightarrow{d}\ N(0,1)}\cdot\underbrace{\frac{\sigma}{S_n}}_{\xrightarrow{P}\ 1}.
```

*Step 1: Sₙ → σ in probability.* Write the sample variance as

```latex
S_n^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X_n)^2=\frac{n}{n-1}\Big[\frac1n\sum_{i=1}^n X_i^2-\bar X_n^2\Big].
```

By the WLLN applied to Yᵢ = Xᵢ², the first average converges to E\[X²\]. (Which WLLN? The three-line Chebyshev proof of Part 8 needs Var(Yᵢ) < ∞, that is E\[X⁴\] < ∞. With only a finite variance, Khinchin's version of the WLLN, which needs just E|Yᵢ| = E\[X²\] < ∞, gives the same conclusion, so the result below holds for every population with finite variance.) By the WLLN and the CMT, X̄ₙ² → μ². Also n/(n − 1) → 1. So Sₙ² → E\[X²\] − μ² = σ² in probability, and by the CMT (square root is continuous) Sₙ → σ, hence σ/Sₙ → 1.

*Step 2: Slutsky.* Tₙ = Zₙ · (σ/Sₙ) → N(0, 1) · 1 = N(0, 1) in distribution.

*Consequence:* for any population with finite variance, X̄ₙ ± 1.96 Sₙ/√n is an approximate 95% confidence interval for μ when n is large. Normality of the data is not required; it is required only for the exact small-sample t-distribution.

**Numerical example.** A batch of n = 64 resistors gives X̄ = 99.6 Ω and S = 2.4 Ω, from an unknown distribution. The large-sample 95% interval is 99.6 ± 1.96 × 2.4/√64 = 99.6 ± 0.59, that is (99.01, 100.19) Ω. The nominal value 100 Ω lies inside it, so the data are consistent with it at the 5% level. (The exact t-quantile with 63 degrees of freedom is about 2.00 rather than 1.96, which would widen the interval only to ±0.60 Ω; the difference vanishes as n grows.)

### 11.4 The delta method

Often we estimate θ well, but we really care about g(θ): a rate 1/θ, a power V²/R, a log-ratio in decibels. The delta method gives the approximate distribution of g(θ̂).

**Theorem (delta method).** Suppose √n(Tₙ − θ) → N(0, τ²) in distribution, and g is differentiable at θ with g′(θ) ≠ 0. Then

```latex
\sqrt n\big(g(T_n)-g(\theta)\big)\xrightarrow{d}N\big(0,\ [g'(\theta)]^2\tau^2\big),
\qquad\text{so}\qquad \operatorname{se}\big(g(T_n)\big)\approx|g'(\theta)|\cdot\operatorname{se}(T_n).
```

**Proof, step by step (assume g′ is continuous near θ).**

1. *Tₙ → θ in probability.* Write Tₙ − θ = (1/√n)·√n(Tₙ − θ). The second factor converges in distribution to N(0, τ²); the first is a number tending to 0. By Slutsky, the product converges in distribution to 0·N(0, τ²) = 0, a constant, and therefore in probability (Section 6.5).
2. *Linearise.* By the mean value theorem, for each outcome there is a point θ̃ₙ between Tₙ and θ with g(Tₙ) − g(θ) = g′(θ̃ₙ)(Tₙ − θ). Multiplying by √n:

```latex
\sqrt n\big(g(T_n)-g(\theta)\big)=g'(\tilde\theta_n)\cdot\sqrt n(T_n-\theta).
```

3. *The slope settles.* Since θ̃ₙ lies between Tₙ and θ, |θ̃ₙ − θ| ≤ |Tₙ − θ| → 0 in probability. Because g′ is continuous at θ, the continuous mapping theorem gives g′(θ̃ₙ) → g′(θ) in probability.
4. *Combine.* The right side is (a sequence → the constant g′(θ) in probability) × (a sequence → N(0, τ²) in distribution). Slutsky's product rule gives the limit g′(θ)·N(0, τ²) = N(0, g′(θ)²τ²). ∎

**Intuition.** Near θ, a smooth function looks like a straight line with slope g′(θ). Small random errors in Tₙ pass through that line, so they get multiplied by g′(θ). A linear function of a normal variable is normal. This is exactly the first-order error propagation formula from your physics lab, now with a theorem behind it.

**Example 11.5 (failure rate from lifetimes).** Lifetimes are iid Exponential(λ), with mean 1/λ and variance 1/λ². The natural estimator is λ̂ = 1/X̄ₙ. The CLT gives √n(X̄ₙ − 1/λ) → N(0, 1/λ²). With g(x) = 1/x, g′(x) = −1/x², so g′(1/λ) = −λ². The delta method gives

```latex
\sqrt n(\hat\lambda-\lambda)\xrightarrow{d}N\big(0,\ \lambda^4\cdot\tfrac{1}{\lambda^2}\big)=N(0,\lambda^2), \qquad \operatorname{se}(\hat\lambda)\approx\frac{\hat\lambda}{\sqrt n}.
```

*Numbers.* With n = 50 failures and mean lifetime 200 h, λ̂ = 0.005 per hour and se ≈ 0.005/√50 ≈ 0.000707. An approximate 95% confidence interval is 0.005 ± 1.96 × 0.000707, that is (0.0036, 0.0064) failures per hour. Plugging λ̂ in for λ inside the standard error is justified by Slutsky.

**Example 11.6 (power from voltage measurements).** You measure a voltage 25 times; single readings have standard deviation 0.1 V, and the average is V̄ = 5.00 V, so se(V̄) = 0.1/√25 = 0.02 V. The power in a 10 Ω load is P = V²/R = 2.5 W. With g(v) = v²/10, g′(5) = 2 × 5/10 = 1 W/V, so se(P) ≈ 1 × 0.02 = 0.02 W, and an approximate 95% interval is 2.50 ± 0.04 W.

**Example 11.7 (when g′(θ) = 0: a caution).** Let p̂ be the sample proportion and g(p) = p(1 − p), the Bernoulli variance. At p = 1/2, g′(1/2) = 1 − 2p = 0, and the first-order delta method gives a degenerate limit N(0, 0). Look closer: g(p̂) − 1/4 = −(p̂ − 1/2)², exactly. Since √n(p̂ − 1/2) → N(0, 1/4), the CMT gives

```latex
n\big(g(\hat p)-\tfrac14\big)=-\big[\sqrt n(\hat p-\tfrac12)\big]^2\xrightarrow{d}-\tfrac14\,\chi^2_1.
```

The error is now of order 1/n, not 1/√n, and its limit is not normal. This is the second-order delta method. When the slope vanishes, look at the curvature.

### Summary and check yourself: Part 11

**In three lines.** Continuous functions preserve almost sure, in-probability and in-distribution limits. Slutsky's theorem lets you add, multiply or divide a sequence converging in distribution by one converging to a constant, which justifies plugging in estimated standard deviations. The delta method propagates a √n-normal error through a smooth g: the standard error is multiplied by |g′(θ)|, unless g′(θ) = 0.

**Exercises.**

1. X̄ₙ → 2 in probability. Find the limit in probability of (X̄ₙ² + 1)/X̄ₙ.
2. Zₙ → N(0, 1) in distribution and Yₙ → 3 in probability. Find the limits in distribution of (a) Zₙ + Yₙ, (b) YₙZₙ, (c) (Zₙ + Yₙ)/Yₙ.
3. In n = 400 trials a component fails 120 times, so p̂ = 0.3. Give an approximate 95% interval for the log-odds ln(p/(1 − p)).
4. A received power is estimated as X̄ = 100 mW with standard error 5 mW. Engineers report it in dBm, g(x) = 10 log₁₀(x / 1 mW). Find the estimate and its approximate standard error in dB.
5. Let Z \~ N(0, 1) and Xₙ = Yₙ = Z for every n. Both converge in distribution to N(0, 1). What is the limit of Xₙ + Yₙ? Why does this not contradict Slutsky's theorem?
6. iid data have mean μ = 0 and variance σ². What does the delta method say about g(X̄ₙ) = X̄ₙ²? Find a scaling that gives a non-degenerate limit.
7. (A discontinuous g.) Let Xₙ \~ N(0, 1/n), so Xₙ → 0 in probability, and let g(x) = 1 for x > 0 and g(x) = 0 for x ≤ 0 (a comparator, or a hard decision). Does g(Xₙ) → g(0) = 0 in probability? Why does the CMT not apply?
8. (A joint argument.) X₁, X₂, … are iid with mean μ and variance σ²; Y₁, Y₂, … are iid with mean ν and variance τ², independent of the X's. Find the limit in distribution of √n(X̄ₙȲₙ − μν).

**Solutions.**

1. g(x) = (x² + 1)/x is continuous at x = 2, so by the CMT the limit is (4 + 1)/2 = 2.5.
2. By Slutsky: (a) Z + 3 \~ N(3, 1); (b) 3Z \~ N(0, 9); (c) (Z + 3)/3 \~ N(1, 1/9).
3. With g(p) = ln(p/(1 − p)), g′(p) = 1/(p(1 − p)), so the asymptotic variance is p(1 − p)/\[p(1 − p)\]² = 1/(p(1 − p)), and se ≈ 1/√(n p̂(1 − p̂)) = 1/√(400 × 0.21) ≈ 0.109. The estimate is ln(0.3/0.7) ≈ −0.847, and the interval is −0.847 ± 1.96 × 0.109, that is (−1.061, −0.633).
4. The estimate is 10 log₁₀(100) = 20 dBm. g′(x) = 10/(x ln 10), so g′(100) ≈ 0.0434 dB per mW, and se ≈ 0.0434 × 5 ≈ 0.22 dB. A 95% interval is 20 ± 0.43 dBm.
5. Xₙ + Yₙ = 2Z \~ N(0, 4), not N(0, 2). Slutsky needs one of the two sequences to converge to a constant; here both limits are random and perfectly dependent, so marginal limits say nothing about the sum (Section 11.2).
6. g′(0) = 0, so the first-order delta method gives the degenerate limit √n·X̄ₙ² → 0. Go to second order: n X̄ₙ² = \[√n X̄ₙ\]² → σ²·Z² by the CLT and the CMT, so n X̄ₙ²/σ² → χ²₁ in distribution. The error is of order 1/n, as in Example 11.7.
7. No. P(g(Xₙ) = 1) = P(Xₙ > 0) = 1/2 for every n, so P(|g(Xₙ) − 0| > 1/2) = 1/2, which does not go to 0; g(Xₙ) stays a fair coin forever. The CMT requires g to be continuous at the limit point, and 0 is exactly where g jumps. Engineering reading: a consistent estimate passed through a hard threshold placed at the true value gives an inconsistent decision. A decision rule needs a margin between the threshold and the value it is testing.
8. Write X̄ₙȲₙ − μν = (X̄ₙ − μ)Ȳₙ + μ(Ȳₙ − ν), so √n(X̄ₙȲₙ − μν) = √n(X̄ₙ − μ)·Ȳₙ + μ·√n(Ȳₙ − ν). The triple (√n(X̄ₙ − μ), √n(Ȳₙ − ν), Ȳₙ) converges jointly in distribution to (U, V, ν) with U \~ N(0, σ²) and V \~ N(0, τ²) independent: the first two coordinates are independent of each other (Section 11.2), and the third converges to a constant (Slutsky, in vector form). The CMT with the continuous function (u, v, y) ↦ uy + μv gives √n(X̄ₙȲₙ − μν) → νU + μV \~ N(0, ν²σ² + μ²τ²). This is the delta method in two variables with g(x, y) = xy: the variance is (∂g/∂x)²σ² + (∂g/∂y)²τ² at (μ, ν). The independence of the X's and the Y's is what lets the two normal limits be added; with dependence a covariance term appears.

## Part 12: Engineering applications

Each mode of convergence answers a different engineering question, and the table below maps the common applications to the mode they rely on. The sections after it work through four applications in detail.

**Goals.** After this Part you can (1) match an engineering question to the right mode of convergence, (2) check an estimator for consistency, (3) size a Monte Carlo run or a signal-averaging experiment, and (4) account for correlation when averaging a single long record.

| Application | Question asked | Mode used | Key theorem |
| --- | --- | --- | --- |
| Estimator design | Does the estimate get arbitrarily accurate with more data? | In probability (consistency) | WLLN, CMT |
| Confidence intervals, hypothesis tests | What is the approximate distribution of the estimate's error? | In distribution | CLT, Slutsky, delta method |
| Monte Carlo simulation | Does my one long run converge, and how big is its error? | Almost surely + in distribution | SLLN, CLT |
| Signal averaging, filtering | How fast does the error power fall? | Mean square | Variance of a sum |
| Time averages of random signals | Does one long record reveal the ensemble mean? | Mean square or almost surely | Ergodic theorems |
| Rare-event counts | Can I approximate the count of defects or bit errors? | In distribution | Binomial → Poisson |
| Adaptive algorithms (LMS, stochastic gradient) | Do the weights settle at the optimum? | Mean square or almost surely | Stochastic approximation |

### 12.1 Consistency of estimators

An estimator θ̂ₙ of a parameter θ is **consistent** if θ̂ₙ → θ in probability (weakly consistent), or **strongly consistent** if the convergence is almost sure. Consistency is the minimum we should demand: an estimator that does not improve with unlimited data is useless.

**The standard test.** By Section 5.2, if bias(θ̂ₙ) → 0 and Var(θ̂ₙ) → 0, then θ̂ₙ → θ in mean square, hence in probability. Alternatively, write θ̂ₙ as a continuous function of sample averages and use the WLLN plus the CMT.

**Example 12.1 (consistent, inconsistent, biased).** Data X₁, …, Xₙ iid with mean μ and variance σ².

- θ̂ₙ = X̄ₙ: unbiased, variance σ²/n → 0. Consistent.
- θ̂ₙ = X₁ (ignore all data but the first): unbiased, but its variance σ² never shrinks. Not consistent. *Unbiasedness alone is not enough.*
- σ̂ₙ² = (1/n)Σ(Xᵢ − X̄ₙ)²: biased, since its mean is (n − 1)σ²/n, but the bias σ²/n → 0, and it is a continuous function of the averages (1/n)ΣXᵢ² and X̄ₙ, so it is consistent. *A vanishing bias is harmless.*
- Mₙ = max(X₁, …, Xₙ) for Uniform(0, θ) data: biased low, yet consistent, and its error shrinks like 1/n rather than 1/√n (Practice Problem 5).

Maximum likelihood estimators are, under mild regularity conditions, consistent and asymptotically normal, √n(θ̂ₙ − θ) → N(0, 1/I(θ)) in distribution, where I(θ) is the Fisher information of one observation. This combines everything in Parts 8 to 11.

**The histogram converges too (Glivenko–Cantelli).** Let F̂ₙ(x) be the fraction of X₁, …, Xₙ that are ≤ x, the *empirical CDF*; it is the staircase you draw from the sorted data. For each fixed x, F̂ₙ(x) is a relative frequency, so the SLLN gives F̂ₙ(x) → F(x) almost surely. The Glivenko–Cantelli theorem says more: the convergence is uniform in x,

```latex
\sup_{x}\,\big|\hat F_n(x)-F(x)\big|\xrightarrow{\text{a.s.}}0
```

(stated without proof). In engineering terms, with enough data the empirical distribution of a signal, as a histogram or an empirical CDF, is a trustworthy picture of the true distribution everywhere at once, not just at one level x. Every sample quantile, every empirical bit-error curve and every distribution-based test (such as Kolmogorov–Smirnov) rests on this.

### 12.2 Monte Carlo integration

To compute I = ∫₀¹ g(x) dx, note that I = E\[g(U)\] for U \~ Uniform(0, 1). Draw U₁, …, Uₙ and use

```latex
\hat I_n=\frac1n\sum_{i=1}^n g(U_i).
```

- **SLLN:** Îₙ → I almost surely, so a single long run converges.
- **CLT:** Îₙ ≈ N(I, σ\_g²/n), where σ\_g² = Var(g(U)), so the error is about σ\_g/√n and an approximate 95% interval is Îₙ ± 1.96 sₙ/√n, with sₙ the sample standard deviation of the g(Uᵢ) (Slutsky justifies using sₙ).
- **Rate:** 1/√n, whatever the dimension of the integral. Halving the error costs four times the samples. In high dimensions this beats grid-based numerical integration, whose error rate deteriorates with dimension.

**Example 12.2 (estimating π).** Draw points (U, V) uniformly in the unit square and let Y = 4 if U² + V² ≤ 1, else 0. Then E\[Y\] = 4 × (π/4) = π and Var(Y) = 16 × (π/4)(1 − π/4) = π(4 − π) ≈ 2.697, so σ\_Y ≈ 1.642. For a 95% error of ±0.01,

```latex
n\approx\Big(\frac{1.96\times1.642}{0.01}\Big)^2\approx 1.04\times10^5 \text{ points}.
```

For one more correct decimal (±0.001) you need 100 times more, about 10⁷ points. Part 13 simulates this.

### 12.3 Signal averaging

A repeatable signal s (an evoked brain response, a radar echo, an ultrasound A-scan) is buried in noise. Record n sweeps, Yᵢ = s + Nᵢ, where the noise samples Nᵢ are independent with mean 0 and variance σ², and average them:

```latex
\bar Y_n=s+\bar N_n,\qquad E[(\bar Y_n-s)^2]=\frac{\sigma^2}{n}\to 0.
```

The estimate converges to s in mean square. The signal power is unchanged while the noise power drops by a factor of n, so the SNR improves by 10·log₁₀(n) dB: 100 sweeps buy 20 dB, 10,000 sweeps buy 40 dB. The amplitude SNR improves by √n.

Caveats that come straight from the theory: noise components that repeat in every sweep (mains hum locked to the trigger, a common-mode offset) are not averaged away (Example 8.3), and correlated noise gives less than the full √n gain (next section).

### 12.4 Time averages and correlated data

Often you have a single long record X₁, X₂, …, Xₙ of a stationary random signal with mean μ, and the samples are correlated. Does the time average converge to μ?

For a wide-sense stationary sequence with autocovariance C(k) = Cov(Xᵢ, Xᵢ₊ₖ), the variance of the time average is Var(X̄ₙ) = (1/n)·Σ\_{|k|\<n} (1 − |k|/n)·C(k), and the time average converges to μ in mean square if and only if (1/n)·Σₖ₌₀ⁿ⁻¹ C(k) → 0 (a theorem of Slutsky, stated here without proof). A simple sufficient condition is C(k) → 0 as k → ∞: samples far apart in time become uncorrelated. A process with this property is called mean-ergodic.

**Example 12.3 (AR(1) noise).** Take C(k) = σ²ρ^|k| with |ρ| < 1, a common model for low-pass filtered noise. Then C(k) → 0, so the time average converges. For large n,

```latex
\operatorname{Var}(\bar X_n)\approx\frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}.
```

The effective number of independent samples is n(1 − ρ)/(1 + ρ). With ρ = 0.9, 1,900 correlated samples carry only as much information as 100 independent ones. *Lesson:* oversampling a slowly varying noise does not reduce the error proportionally.

**Example 12.4 (a non-ergodic signal).** Let Xᵢ = A for all i, where A is a random amplitude chosen once (say a random DC offset). Then C(k) = Var(A) for every k, never decaying, and the time average equals A forever. It converges, but to the random A, not to the ensemble mean E\[A\]. One record cannot reveal the ensemble mean of such a process.

### 12.5 Adaptive algorithms (a preview)

Adaptive filters (LMS), stochastic gradient descent in machine learning, and recursive estimators update an estimate with each new noisy sample: wₙ₊₁ = wₙ − aₙ·(noisy gradient). With a fixed step size a, LMS converges in the mean, and its mean squared error settles at a floor proportional to a. With step sizes that shrink so that Σ aₙ = ∞ and Σ aₙ² < ∞ (for example aₙ = 1/n), the Robbins–Monro theory gives almost sure convergence to the optimum. The first condition makes sure the steps can travel any distance; the second makes the accumulated noise summable, exactly the Borel–Cantelli spirit of Section 4.3.

**Example 12.5 (a one-weight LMS filter, worked through).** The simplest adaptive estimator tracks a constant level μ from noisy samples Xₙ = μ + Nₙ, with iid zero-mean noise of variance σ², by the recursion wₙ₊₁ = wₙ + aₙ(Xₙ − wₙ), starting from w₁ = 0. Each update moves the estimate toward the newest sample by a fraction aₙ of the current error.

*Fixed step a, 0 < a < 1.* Subtract μ from both sides: the error eₙ = wₙ − μ obeys eₙ₊₁ = (1 − a)eₙ + aNₙ. Taking expectations, E\[eₙ₊₁\] = (1 − a)E\[eₙ\], so E\[wₙ\] → μ geometrically: LMS "converges in the mean". Taking variances (eₙ depends only on past noise, so it is independent of Nₙ), Var(eₙ₊₁) = (1 − a)²Var(eₙ) + a²σ², whose fixed point is

```latex
\operatorname{Var}(e_\infty)=\frac{a^2\sigma^2}{1-(1-a)^2}=\frac{a\,\sigma^2}{2-a}.
```

This does not go to 0: the mean squared error settles at a floor proportional to a (the "misadjustment"), so with a fixed step wₙ does not converge to μ in any mode. A small a gives a low floor but slow tracking; this is the step-size trade-off of every adaptive filter.

*Decreasing step aₙ = 1/n.* Now wₙ₊₁ = wₙ + (Xₙ − wₙ)/n, and by induction wₙ₊₁ = X̄ₙ, the running sample mean (check: w₂ = X₁, w₃ = X₁ + (X₂ − X₁)/2 = (X₁ + X₂)/2, and in general wₙ₊₁ = ((n − 1)X̄ₙ₋₁ + Xₙ)/n = X̄ₙ). So the Robbins–Monro step size turns LMS into the sample mean: the SLLN gives wₙ → μ almost surely, the CLT gives its √n-normal error, and the mean squared error σ²/n falls to zero. The two conditions Σ aₙ = ∞ and Σ aₙ² < ∞ are exactly what let the steps travel any distance while the accumulated noise, whose variance is of order Σ aₙ²σ², stays finite.

### Summary and check yourself: Part 12

**In three lines.** Consistency is convergence in probability; confidence intervals and tests rest on convergence in distribution; error power is mean square; trusting one long run or record is almost sure (ergodic) convergence. Averaging n independent samples cuts error power by n, that is, 10 log₁₀ n dB, at any dimension. Correlation reduces the effective number of samples, and shared errors never average out.

**Exercises.**

1. Is X̄ₙ² a consistent estimator of μ²? Is it unbiased? Suggest an unbiased alternative.
2. An evoked response has an SNR of −10 dB in a single sweep. How many sweeps must be averaged to reach +10 dB? And to reach +20 dB?
3. A stationary noise record has AR(1) correlation with ρ = 0.5. How many consecutive samples carry the same information about the mean as 100 independent samples?
4. A Monte Carlo simulation estimates a failure probability p ≈ 10⁻⁴. How many runs are needed for a relative error of ±10% at 95% confidence?
5. Which step-size sequences satisfy the Robbins–Monro conditions Σaₙ = ∞ and Σaₙ² < ∞: (a) aₙ = 1/n, (b) aₙ = 1/√n, (c) aₙ = 1/n², (d) aₙ = 0.01?
6. Every unit in a factory shares the same random calibration offset B, chosen once. Will a very long time average of one unit's readings reveal the ensemble mean E\[reading\]? Explain using Section 12.4.

**Solutions.**

1. Consistent: X̄ₙ → μ in probability and x² is continuous, so X̄ₙ² → μ² (CMT). Biased: E\[X̄ₙ²\] = Var(X̄ₙ) + μ² = μ² + σ²/n. An unbiased alternative is X̄ₙ² − Sₙ²/n, since E\[Sₙ²\] = σ². The bias σ²/n vanishes anyway, so both are consistent.
2. The SNR must rise by 20 dB, a factor of 100 in power, so n = 100 sweeps. Reaching +20 dB needs 30 dB of gain, n = 1000 sweeps.
3. The effective sample size is n(1 − ρ)/(1 + ρ) = n/3, so about 300 correlated samples are worth 100 independent ones.
4. p̂ has standard error √(p(1 − p)/n), so the relative error is √((1 − p)/(np)). Require 1.96·√((1 − p)/(np)) ≤ 0.1, giving n ≥ 384.2(1 − p)/p ≈ 3.8 × 10⁶ runs. Rare events are expensive to simulate directly, which motivates importance sampling.
5. (a) Yes: Σ1/n = ∞ and Σ1/n² < ∞. (b) No: Σaₙ² = Σ1/n = ∞, so the accumulated noise does not settle. (c) No: Σ1/n² < ∞, so the steps cannot travel far enough. (d) No: Σ0.0001 = ∞; a constant step converges only to a noisy neighbourhood, as with fixed-step LMS.
6. No. The record is like Example 12.4: its autocovariance includes Var(B) at every lag and never decays, so the time average converges to μ + B for that unit, not to the ensemble mean μ + E\[B\]. Averaging across many units, each with its own offset, is needed.

## Part 13: Python simulations with plots

Seven short, self-contained scripts let you watch every major theorem of this tutorial happen. Each figure below was produced by running the script exactly as printed; your plots will differ in detail only if you change the random seed.

**What a simulation can and cannot show.** Every simulation stops after finitely many steps. It can illustrate a mode of convergence and catch a wrong formula, but it can never prove that something happens forever. Statements such as "the path never leaves the band again" are guaranteed by the theorems, not by the plots.

**Setup.** Install the three libraries once with `pip install numpy scipy matplotlib`, then save each script as a .py file and run it, or paste it into a Jupyter notebook cell. Every script fixes a seed with `np.random.default_rng(seed)` so results are reproducible. Change the seed to see a fresh random world.

**How to use them.** For each simulation, first predict what the plot will look like using the theory, then run it, then do the "Try this" experiment.

### Simulation 1: the Strong Law, one path at a time (Parts 4 and 9)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation 1: Strong Law of Large Numbers -- running means of Exponential(1) data
rng = np.random.default_rng(1)
mu, eps = 1.0, 0.1                 # true mean and tolerance band
n_max, n_paths = 10_000, 20
n = np.arange(1, n_max + 1)

plt.figure(figsize=(8, 4.5))
for _ in range(n_paths):
    x = rng.exponential(scale=mu, size=n_max)      # one engineer's data
    running_mean = np.cumsum(x) / n                # X-bar_1, X-bar_2, ..., X-bar_n
    plt.plot(n, running_mean, lw=0.8, alpha=0.8)
plt.axhline(mu, color="k", lw=1.5, label="true mean μ = 1")
plt.axhline(mu + eps, color="k", ls="--", lw=1, label="ε-band, ε = 0.1")
plt.axhline(mu - eps, color="k", ls="--", lw=1)
plt.xscale("log"); plt.ylim(0, 2.5)
plt.xlabel("n (log scale)"); plt.ylabel("running mean")
plt.title("20 independent paths of the running mean: each one enters the band and stays")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig("sim1_lln_paths.png", dpi=130); plt.show()
```

&#91;image: Twenty running-mean paths of Exponential(1) data converging into the band 1 ± 0.1\]

**What to look for.** Each coloured line is one engineer's own running mean, the "million engineers" picture of Part 1 with 20 engineers. Early on the paths wander far outside the dashed ε-band. By n ≈ 1000 every path has entered the band, and none leaves it again within this run's 10,000 steps. The plot cannot show "forever"; the Strong Law is what guarantees it. That is almost sure convergence: each individual path converges.

**Try this.** Narrow the band to ε = 0.02 and extend n\_max to 10⁶. Every path still enters and stays, but later. Estimate from the plot roughly when, and compare with the standard error 1/√n.

### Simulation 2: the Weak Law, probability versus bounds (Parts 2, 8 and 10)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation 2: Weak law -- estimate P(|X-bar_n - mu| > eps) and compare with Chebyshev and the CLT
rng = np.random.default_rng(2)
mu, sigma, eps = 1.0, 1.0, 0.1     # Exponential(1): mean 1, variance 1
ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
reps = 20_000                      # number of simulated "engineers" per n

p_sim = []
for n in ns:
    means = rng.exponential(scale=mu, size=(reps, n)).mean(axis=1)
    p_sim.append(np.mean(np.abs(means - mu) > eps))
p_sim = np.array(p_sim)
p_cheb = np.minimum(1, sigma**2 / (ns * eps**2))           # Chebyshev bound
p_clt = 2 * (1 - norm.cdf(eps * np.sqrt(ns) / sigma))      # CLT approximation

for n, a, b, c in zip(ns, p_sim, p_cheb, p_clt):
    print(f"n={n:5d}  simulated={a:.4f}  Chebyshev bound={b:.4f}  CLT approx={c:.4f}")

plt.figure(figsize=(8, 4.5))
plt.plot(ns, p_sim, "o-", label="simulated probability")
plt.plot(ns, p_cheb, "s--", label="Chebyshev bound σ²/(nε²)")
plt.plot(ns, p_clt, "^:", label="CLT approximation 2[1 − Φ(ε√n/σ)]")
plt.xscale("log"); plt.yscale("log"); plt.ylim(1e-5, 1.5)
plt.xlabel("n (log scale)"); plt.ylabel("P(|X̄ₙ − μ| > 0.1)  (log scale)")
plt.title("Weak law: the probability of a large error goes to 0")
plt.legend(); plt.tight_layout()
plt.savefig("sim2_prob_vs_bounds.png", dpi=130); plt.show()
```

&#91;image: Simulated probability of an error above 0.1 versus n, with the Chebyshev bound and the CLT approximation\]

**What to look for.** The simulated probability of an error above 0.1 falls from 0.75 at n = 10 to 0.0014 at n = 1000 (at n = 2000 none of the 20,000 runs failed, so that point is off the log scale). The CLT curve sits right on top of the simulation; Chebyshev is a valid upper bound but is 1 (useless) until n = 100 and about 70 times too large at n = 1000. This is the WLLN made visible, and it shows why the CLT, not Chebyshev, is used for sample-size planning.

**Try this.** Replace the exponential data with `rng.uniform(0, 2, size=(reps, n))` (mean 1, variance 1/3) and change sigma accordingly. The CLT curve should fit even better, because uniform data are symmetric.

### Simulation 3: the Central Limit Theorem (Part 10)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation 3: Central Limit Theorem for skewed Exponential(1) data
rng = np.random.default_rng(3)
mu, sigma, reps = 1.0, 1.0, 50_000
z = np.linspace(-4, 4, 400)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, n in zip(axes, [1, 5, 30, 200]):
    means = rng.exponential(scale=mu, size=(reps, n)).mean(axis=1)
    Zn = (means - mu) / (sigma / np.sqrt(n))       # standardised sample mean
    ax.hist(Zn, bins=80, range=(-4, 6), density=True, alpha=0.6, label="simulated Zₙ")
    ax.plot(z, norm.pdf(z), "k", lw=1.5, label="N(0,1) density")
    ax.set_title(f"n = {n}"); ax.set_xlim(-4, 6)
axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
fig.suptitle("CLT: histograms of the standardised mean of Exponential(1) data")
plt.tight_layout()
plt.savefig("sim3_clt_histograms.png", dpi=130); plt.show()
```

&#91;image: Histograms of the standardised mean for n = 1, 5, 30 and 200, approaching the normal curve\]

**What to look for.** At n = 1 the histogram is the skewed exponential shape, cut off at −1. At n = 5 it is still visibly skewed to the right. By n = 30 it is close to the bell curve, and by n = 200 it is nearly indistinguishable. Each panel shows a histogram over 50,000 engineers at one fixed n: convergence in distribution is a statement about these histograms, not about any one path.

**Try this.** Replace the exponential with a much more skewed distribution, `rng.lognormal(0, 1.5, size=(reps, n))`, and set mu and sigma to its true values, `mu = np.exp(1.5**2 / 2)` and `sigma = np.sqrt((np.exp(1.5**2) - 1) * np.exp(1.5**2))`. How large must n be now? This is why "n ≥ 30" is only a rule of thumb.

### Simulation 4: in probability versus almost surely (Parts 4, 7 and 9)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation 4: in probability vs almost surely
# Independent X_n ~ Bernoulli(1/n)   -> converges in probability, NOT almost surely
# Independent X_n ~ Bernoulli(1/n^2) -> converges almost surely
rng = np.random.default_rng(4)
n_max, n_paths = 100_000, 10
n = np.arange(1, n_max + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, power, name in [(axes[0], 1, "Bernoulli(1/n)"), (axes[1], 2, "Bernoulli(1/n²)")]:
    for _ in range(n_paths):
        x = rng.random(n_max) < 1 / n**power       # X_n = 1 with probability 1/n^power
        ax.step(n, np.cumsum(x), where="post", lw=1)
    if power == 1:
        ax.plot(n, np.log(n) + 0.5772, "k--", lw=1.5, label="E[count] ≈ ln n + 0.577")
        ax.legend(loc="upper left")
    ax.set_xscale("log"); ax.set_xlabel("n (log scale)")
    ax.set_title(f"{name}: number of 1s seen up to time n")
axes[0].set_ylabel("cumulative number of 1s")
plt.tight_layout()
plt.savefig("sim4_as_vs_prob.png", dpi=130); plt.show()
```

&#91;image: Cumulative count of 1s for Bernoulli(1/n), which keeps growing like ln n, versus Bernoulli(1/n²), which stops after one or two\]

**What to look for.** Left panel: every path keeps collecting new 1s forever, roughly ln n of them by time n, exactly as the second Borel–Cantelli lemma predicts. Each new 1 is a departure from the limit 0, so no path converges, even though P(Xₙ = 1) = 1/n → 0. Right panel: every path gets one or two 1s near the start (X₁ = 1 always, since 1/1² = 1) and then no more within the 10⁵ steps shown; the first Borel–Cantelli lemma guarantees, with probability 1, that only finitely many ever occur. This is the first Borel–Cantelli lemma, and it is almost sure convergence.

**Try this.** Use probabilities 1/(n ln n) and 1/(n (ln n)²) for n ≥ 3 (Example 9.2). In theory the first count grows without bound (like ln ln n) and the second stops. Can you see the difference on screen? (See the answer at the end of this Part.)

### Simulation 5: when the Law of Large Numbers fails (Part 8)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation 5: the LLN fails without a mean -- Cauchy vs normal running means
rng = np.random.default_rng(5)
n_max = 100_000
n = np.arange(1, n_max + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for _ in range(5):
    axes[0].plot(n, np.cumsum(rng.standard_normal(n_max)) / n, lw=0.9)
    axes[1].plot(n, np.cumsum(rng.standard_cauchy(n_max)) / n, lw=0.9)
axes[0].set_title("Normal data: running means settle at 0")
axes[1].set_title("Cauchy data: running means never settle")
for ax in axes:
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("n (log scale)"); ax.set_ylabel("running mean")
axes[0].set_ylim(-2, 2); axes[1].set_ylim(-10, 10)
plt.tight_layout()
plt.savefig("sim5_cauchy.png", dpi=130); plt.show()
```

&#91;image: Running means of normal data settling at zero, next to Cauchy running means that keep jumping\]

**What to look for.** The normal running means squeeze onto 0. The Cauchy running means keep making sudden jumps, even after 100,000 samples: a single enormous observation can move the average of everything before it. With no finite mean, there is nothing for the average to converge to.

**Try this.** Replace the running mean of the Cauchy data with the running median (for example, compute np.median(x\[:k\]) at k = 10, 100, 1000, …). The median does converge to 0. Robust statistics exist for exactly this situation.

### Simulation 6: Monte Carlo estimate of π (Part 12)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulation 6: Monte Carlo estimate of pi -- error shrinks like 1/sqrt(n)
rng = np.random.default_rng(6)
n_max = 10**7
u, v = rng.random(n_max), rng.random(n_max)
y = 4.0 * (u**2 + v**2 <= 1.0)                 # Y = 4 inside the quarter circle, else 0
n = np.arange(1, n_max + 1)
estimate = np.cumsum(y) / n

idx = np.unique(np.logspace(1, 7, 300).astype(int)) - 1
sigma_y = np.sqrt(np.pi * (4 - np.pi))         # standard deviation of one Y, about 1.642
print("final estimate:", estimate[-1], " error:", abs(estimate[-1] - np.pi))

plt.figure(figsize=(8, 4.5))
plt.loglog(n[idx], np.abs(estimate[idx] - np.pi), lw=1, label="|estimate − π| (one run)")
plt.loglog(n[idx], 1.96 * sigma_y / np.sqrt(n[idx]), "k--", label="95% CLT error bound 1.96σ/√n")
plt.xlabel("number of points n"); plt.ylabel("absolute error")
plt.title("Monte Carlo π: 100× more points buys one more decimal")
plt.legend(); plt.tight_layout()
plt.savefig("sim6_monte_carlo_pi.png", dpi=130); plt.show()
```

&#91;image: Absolute error of a Monte Carlo estimate of pi versus number of points, on log–log axes, staying under the 1/√n CLT line\]

**What to look for.** On log–log axes, the 1/√n bound is a straight line of slope −1/2. The actual error of this one run zig-zags (it touches zero whenever the estimate crosses π) but stays under the line almost all the time. After 10⁷ points, this run gave 3.14205, an error of about 0.0005, consistent with the CLT prediction 1.96 × 1.642/√10⁷ ≈ 0.001 for a 95% error bound.

**Try this.** Run 200 independent shorter runs of n = 10⁴ points each and plot the histogram of the 200 estimates. You should see a bell curve centred at π with standard deviation close to 1.642/100 ≈ 0.016. That is the CLT for Monte Carlo.

### Simulation 7: checking the delta method (Part 11)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation 7: delta method for the failure-rate estimator lambda-hat = 1 / X-bar
rng = np.random.default_rng(7)
lam, reps = 0.005, 50_000                        # true rate: 0.005 failures per hour
z = np.linspace(-4, 4, 400)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, n in zip(axes, [10, 50, 500]):
    xbar = rng.exponential(scale=1 / lam, size=(reps, n)).mean(axis=1)
    lam_hat = 1 / xbar
    W = np.sqrt(n) * (lam_hat - lam) / lam      # delta method: approximately N(0,1)
    ax.hist(W, bins=80, range=(-4, 6), density=True, alpha=0.6, label="simulated")
    ax.plot(z, norm.pdf(z), "k", lw=1.5, label="N(0,1)")
    ax.set_title(f"n = {n}"); ax.set_xlim(-4, 6)
    print(f"n={n}: mean={W.mean():.3f}, sd={W.std():.3f}")
axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
fig.suptitle("Delta method: √n(λ̂ − λ)/λ approaches N(0, 1)")
plt.tight_layout()
plt.savefig("sim7_delta_method.png", dpi=130); plt.show()
```

&#91;image: Histograms of the standardised failure-rate estimator for n = 10, 50 and 500, approaching the normal curve\]

**What to look for.** The delta method of Example 11.5 predicts that √n(λ̂ − λ)/λ is approximately N(0, 1). The printed summary for this run was mean 0.348 and standard deviation 1.228 at n = 10, mean 0.146 and standard deviation 1.036 at n = 50, and mean 0.040 and standard deviation 1.002 at n = 500. At small n, λ̂ = 1/X̄ is biased upward and right-skewed, because 1/x is convex; the approximation improves steadily, exactly as an asymptotic (large-n) result should.

**Try this.** The convexity bias can be corrected: for exponential data, E\[1/X̄ₙ\] = nλ/(n − 1). Rerun with lam\_hat = (n − 1) / (n \* xbar) and check that the mean of W moves to 0.

### Simulation 8: a non-normal limit, binomial to Poisson (Part 6)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

# Simulation 8: Binomial(n, lambda/n) -> Poisson(lambda), a non-normal limit in distribution
lam = 3.0
k = np.arange(0, 13)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, n in zip(axes, [6, 20, 200]):
    pb, pp = binom.pmf(k, n, lam / n), poisson.pmf(k, lam)
    ax.bar(k - 0.2, pb, width=0.4, label=f"Binomial({n}, {lam}/{n})")
    ax.bar(k + 0.2, pp, width=0.4, label="Poisson(3)")
    kk = np.arange(0, n + 1)   # total variation distance over the whole support
    tv = 0.5 * (np.abs(binom.pmf(kk, n, lam / n) - poisson.pmf(kk, lam)).sum() + poisson.sf(n, lam))
    ax.set_title(f"n = {n}: total variation distance = {tv:.3f}")
    ax.set_xlabel("k"); ax.legend(fontsize=8)
    print(f"n={n}: TV distance = {tv:.4f}")
axes[0].set_ylabel("P(X = k)")
fig.suptitle("Binomial(n, 3/n) converges in distribution to Poisson(3)")
plt.tight_layout(); plt.savefig("sim8_binomial_poisson.png", dpi=130); plt.show()
```

&#91;image: Bar charts comparing Binomial(n, 3/n) with Poisson(3) for n = 6, 20 and 200\]

**What to look for.** This is convergence in distribution with a limit that is not normal, and it is computed exactly rather than simulated. The total variation distance is half the sum of the absolute differences of the two PMFs; it is the largest possible difference between the two distributions' probabilities of any event. It falls from 0.165 at n = 6 to 0.040 at n = 20 and 0.004 at n = 200, roughly like 1/n. For integer-valued variables, this is the PMF method of Section 6.3 made visible.

**Try this.** Replace Poisson(3) with the CLT's normal approximation N(3, 3(1 − 3/n)) and compute P(X ≤ k) both ways for small k. For n = 200 and λ = 3, the Poisson approximation is far better than the normal one, because the mean is small and the distribution is skewed.

### Answers to the Try this tasks

The numbers below come from running each modified script with the seeds printed in this Part; your results will be similar but not identical if you change seeds.

1. **Simulation 1 with ε = 0.02 and n\_max = 10⁶.** The last time each of the 20 paths left the band ranged from about 900 to about 15,000, with a median near 4,300. This is consistent with theory: the standard error 1/√n equals 0.02 at n = 2,500, and a path needs ε to be two or three standard errors before it settles, which happens around n ≈ 10⁴ to 2 × 10⁴.
2. **Simulation 2 with Uniform(0, 2) data.** Simulated versus CLT values were 0.591 vs 0.584 at n = 10, 0.221 vs 0.221 at n = 50, 0.083 vs 0.083 at n = 100 and 0.015 vs 0.014 at n = 200. The fit is closer than for the skewed exponential, as expected for symmetric data.
3. **Simulation 3 with lognormal(0, 1.5) data.** The skewness of this distribution is about 33.5, and the skewness of a sample mean is the original skewness divided by √n. Exponential data, with skewness 2, reach a skewness of 0.5 at n = 16; lognormal data need (33.5/0.5)² ≈ 4,500 samples for the same. At n = 30 the histogram is still strongly skewed: "n ≥ 30" can fail badly.
4. **Simulation 4 with borderline probabilities.** Starting at n = 3, the expected number of events up to 10⁶ is about 2.70 for 1/(n ln n) and about 1.00 for 1/(n (ln n)²); up to 10⁸ it is about 2.99 and 1.01. The first count does grow forever, but so slowly that no computer run can tell the two cases apart. Only the Borel–Cantelli lemmas can.
5. **Simulation 5 with running medians.** The running median of the Cauchy data was 0.265, 0.030, 0.011, −0.008 and 0.005 at n = 10, 10², 10³, 10⁴ and 10⁵. It converges to 0, the centre of the Cauchy distribution, even though the mean does not exist. The sample median's error shrinks like 1/√n here.
6. **Simulation 6 with 200 runs of 10⁴ points.** The 200 estimates had mean 3.1410 and standard deviation 0.0165, matching the predicted 1.642/√10⁴ ≈ 0.0164. Their histogram is bell-shaped around π: the CLT for Monte Carlo.
7. **Simulation 7 with the bias-corrected estimator (n − 1)/(n X̄).** The mean of W became −0.003, 0.001 and −0.005 at n = 10, 50 and 500: the bias is gone. The standard deviation at n = 10 is still about 1.10 rather than 1, and the histogram is still skewed, so small-sample intervals need care even after the bias correction.
8. **Simulation 8 with the normal approximation.** For n = 200, the Poisson CDF is within 0.003 of the exact binomial CDF at every k from 0 to 8, while the normal approximation, even with a continuity correction, is off by up to 0.036 and puts about 4% of its probability below 0. Rule: for counts of rare events, use Poisson; for sums of many moderate contributions, use the normal.

## Part 14: Practice problems with full solutions

Fourteen problems, ordered from routine to challenging, cover every Part of the tutorial. Try each one for at least five minutes before reading its solution; the solution is most useful when you have a half-finished attempt to compare it with.

| Problems | Level | Main skills |
| --- | --- | --- |
| 1 to 4 | Beginner | Checking each mode directly |
| 5 to 9 | Intermediate | Limit distributions, proofs from the definitions, CLT and delta method calculations |
| 10 to 14 | Advanced | Counterexamples, extreme values, Slutsky, Borel–Cantelli, a proof of the strong law |

### Problem 1 (beginner)

Let Xₙ \~ Uniform(−1/n, 1/n). In which of the four modes does Xₙ converge to 0?

**Solution.** All four.

- *Almost surely (in fact surely):* |Xₙ(ω)| ≤ 1/n for every outcome, so Xₙ(ω) → 0 for every ω by the squeeze theorem.
- *In probability:* for any ε > 0, once n > 1/ε the event |Xₙ| > ε is impossible, so its probability is 0.
- *In mean square:* E\[Xₙ²\] = Var(Xₙ) = (2/n)²/12 = 1/(3n²) → 0.
- *In distribution:* follows from convergence in probability.

### Problem 2 (beginner)

Let X₁, X₂, … be independent with Xₙ = n with probability 1/n² and Xₙ = 0 otherwise. Determine convergence to 0 almost surely, in probability, in L¹ and in mean square.

**Solution.**

- *In probability:* P(|Xₙ| > ε) ≤ 1/n² → 0. Yes.
- *Almost surely:* Σ P(Xₙ ≠ 0) = Σ 1/n² < ∞, so by the first Borel–Cantelli lemma only finitely many Xₙ are non-zero, with probability 1. Yes.
- *In L¹:* E|Xₙ| = n·(1/n²) = 1/n → 0. Yes.
- *In mean square:* E\[Xₙ²\] = n²·(1/n²) = 1 for every n. No.

This sequence converges almost surely and in L¹ but not in mean square: the spikes are rare enough to stop eventually, and small enough on average for L¹, but squaring makes them too costly.

### Problem 3 (beginner)

Let Xₙ \~ N(1/n, 1/n). (a) Show Xₙ → 0 in mean square. (b) Can you decide almost sure convergence from this information? (c) Decide it if the Xₙ are independent.

**Solution.**

(a) Bias–variance: E\[Xₙ²\] = Var(Xₙ) + (E\[Xₙ\])² = 1/n + 1/n² → 0.

(b) No. Almost sure convergence depends on how the Xₙ are related to each other (their joint distribution), and we are only told each one's marginal distribution. Convergence in probability, in Lᵖ and in distribution can be decided from marginals when the limit is a constant; almost sure convergence cannot.

(c) If they are independent, write Xₙ = 1/n + Zₙ/√n with Zₙ \~ N(0, 1). For n > 2/ε, |Xₙ| > ε requires |Zₙ| > (ε/2)√n, so P(|Xₙ| > ε) ≤ P(|Z| > ε√n/2) ≤ 2e^{−ε²n/8} (a standard Gaussian tail bound). This is summable, so Xₙ → 0 almost surely by the first Borel–Cantelli lemma. (In fact the summability argument does not even need independence.)

### Problem 4 (beginner)

Let Y₁, Y₂, … be iid Exponential(λ) lifetimes of the components in a series system, and let Mₙ = min(Y₁, …, Yₙ), the system lifetime. Show Mₙ → 0 almost surely and in mean square, and find the limit distribution of nMₙ.

**Solution.** P(Mₙ > t) = P(all Yᵢ > t) = (e^{−λt})ⁿ = e^{−nλt}, so Mₙ \~ Exponential(nλ).

- *Mean square:* E\[Mₙ²\] = 2/(nλ)² → 0.
- *Almost surely:* P(Mₙ > ε) = e^{−nλε} is summable (geometric). Alternatively, Mₙ is non-increasing in n and converges in probability, so the monotone method applies.
- *Scaled limit:* P(nMₙ > x) = P(Mₙ > x/n) = e^{−λx} for every n. So nMₙ \~ Exponential(λ) exactly, and trivially converges in distribution to Exponential(λ).

*Engineering meaning:* a series system of n identical exponential components fails n times faster, with lifetime of order 1/(nλ).

### Problem 5 (intermediate)

X₁, …, Xₙ are iid Uniform(0, θ) with θ unknown (for example, the unknown maximum delay of a network link). Show Mₙ = max(Xᵢ) is a consistent estimator of θ, and find the limit distribution of n(θ − Mₙ).

**Solution.** For 0 < ε < θ,

```latex
P(|M_n-\theta|>\varepsilon)=P(M_n<\theta-\varepsilon)=\Big(1-\frac{\varepsilon}{\theta}\Big)^n\to 0,
```

so Mₙ is consistent (and, since this is summable, strongly consistent). For x ≥ 0,

```latex
P\big(n(\theta-M_n)>x\big)=P\Big(M_n<\theta-\frac{x}{n}\Big)=\Big(1-\frac{x}{n\theta}\Big)^n\to e^{-x/\theta}.
```

So n(θ − Mₙ) → Exponential with mean θ, in distribution. The error θ − Mₙ is of order θ/n, much smaller than the 1/√n error of a sample-mean estimator such as 2X̄ₙ. Estimators with this faster rate are called superconsistent. Note the CLT does not apply here: Mₙ is a maximum, not an average, and its limit is not normal.

### Problem 6 (intermediate)

Prove from the definition: if Xₙ → X and Yₙ → Y in probability, then Xₙ + Yₙ → X + Y in probability.

**Solution.** By the triangle inequality, |(Xₙ + Yₙ) − (X + Y)| ≤ |Xₙ − X| + |Yₙ − Y|. If the left side exceeds ε, at least one term on the right exceeds ε/2. By the union bound,

```latex
P\big(|(X_n+Y_n)-(X+Y)|>\varepsilon\big)\le P\big(|X_n-X|>\tfrac{\varepsilon}{2}\big)+P\big(|Y_n-Y|>\tfrac{\varepsilon}{2}\big)\to 0+0.
```

Since ε was arbitrary, the claim follows. Compare with Example 11.3: the analogous statement for convergence in distribution is false.

### Problem 7 (intermediate)

A production run has n = 400 independent components, each defective with probability 0.02. Approximate P(more than 12 defective) using (a) the normal approximation with continuity correction and (b) the Poisson approximation. (c) Which is better, and why?

**Solution.** Let X \~ Binomial(400, 0.02): mean np = 8, variance np(1 − p) = 7.84, standard deviation 2.8. We need P(X ≥ 13).

(a) Normal: P(X ≥ 13) ≈ P(Z ≥ (12.5 − 8)/2.8) = P(Z ≥ 1.607) ≈ 0.054.

(b) Poisson(8): P(X ≥ 13) = 1 − Σₖ₌₀¹² e⁻⁸8ᵏ/k! ≈ 0.064.

(c) The exact binomial value is 0.0619. The Poisson approximation (error 0.002) beats the normal one (error 0.008). With p small and np = 8 moderate, the binomial is right-skewed; the Poisson limit (Example 6.2) captures that skew, while the symmetric normal curve does not. Rule: rare events → Poisson; many moderate-probability events → normal.

### Problem 8 (intermediate)

You want to estimate I = ∫₀¹ eˣ dx by Monte Carlo with error at most ±0.001 at 95% confidence. How many samples are needed? (The true value is e − 1 ≈ 1.71828, which is useful for checking.)

**Solution.** With U \~ Uniform(0, 1), I = E\[eᵁ\]. The variance of one sample is

```latex
\sigma^2=E[e^{2U}]-(E[e^U])^2=\frac{e^2-1}{2}-(e-1)^2\approx 3.1945-2.9525=0.2420, \qquad \sigma\approx 0.492.
```

By the CLT, the 95% error is about 1.96σ/√n. Setting this to 0.001 gives

```latex
n\approx\Big(\frac{1.96\times0.492}{0.001}\Big)^2\approx 9.3\times10^5.
```

In practice σ is unknown; you would estimate it from a pilot run of a few thousand samples (Slutsky justifies plugging in the estimate).

### Problem 9 (intermediate)

In n = 200 trials, a communication link fails 40 times, so p̂ = 0.2. Engineers report reliability as odds of failure, r = p/(1 − p). Estimate r and give an approximate 95% confidence interval using the delta method.

**Solution.** r̂ = 0.2/0.8 = 0.25. By the CLT, √n(p̂ − p) → N(0, p(1 − p)). With g(p) = p/(1 − p), g′(p) = 1/(1 − p)². The delta method gives asymptotic variance

```latex
[g'(p)]^2\,p(1-p)=\frac{p(1-p)}{(1-p)^4}=\frac{p}{(1-p)^3}, \qquad \operatorname{se}(\hat r)\approx\sqrt{\frac{0.2}{0.8^3\times 200}}\approx 0.0442.
```

The approximate 95% interval is 0.25 ± 1.96 × 0.0442, that is (0.163, 0.337).

### Problem 10 (advanced)

Let X \~ N(0, 1) and Xₙ = (−1)ⁿX. Show that Xₙ converges in distribution but not in probability.

**Solution.** Because N(0, 1) is symmetric, −X has the same distribution as X. So every Xₙ is N(0, 1), and Xₙ → N(0, 1) in distribution trivially.

Suppose Xₙ → Y in probability for some Y. Then, as in Example 3.7, Xₙ₊₁ − Xₙ → 0 in probability. But Xₙ₊₁ − Xₙ = ±2X, so |Xₙ₊₁ − Xₙ| = 2|X|, and P(2|X| > 1) = P(|X| > 0.5) ≈ 0.617 for every n. This does not go to 0. Contradiction.

### Problem 11 (advanced, extreme values)

Let X₁, X₂, … be iid Exponential(1) (for example, normalised noise power samples) and Mₙ = max(X₁, …, Xₙ). Show that Mₙ − ln n converges in distribution, and identify the limit.

**Solution.** For fixed x and n large enough that ln n + x > 0,

```latex
P(M_n-\ln n\le x)=\big[P(X_1\le \ln n+x)\big]^n=\big(1-e^{-\ln n-x}\big)^n=\Big(1-\frac{e^{-x}}{n}\Big)^n\to \exp\!\big(-e^{-x}\big).
```

The limit G(x) = exp(−e^{−x}) is a continuous CDF on the whole real line, the **Gumbel distribution**. So Mₙ − ln n → Gumbel in distribution. This sharpens Example 9.3: the peak grows like ln n, and the fluctuations around ln n have a Gumbel shape. Gumbel laws are used to design for maximum floods, wind loads and peak interference.

### Problem 12 (advanced, Slutsky)

X₁, X₂, … are iid with mean μ ≠ 0 and variance σ². Find the limit in distribution of (a) Wₙ = Zₙ(1 + 1/√n) + 3/n, where Zₙ = √n(X̄ₙ − μ)/σ, and (b) Vₙ = √n(X̄ₙ − μ)/X̄ₙ.

**Solution.**

(a) Zₙ → N(0, 1) by the CLT, (1 + 1/√n) → 1 and 3/n → 0 (constants, hence in probability). By Slutsky, Wₙ → 1·N(0, 1) + 0 = N(0, 1).

(b) √n(X̄ₙ − μ) → N(0, σ²) by the CLT, and X̄ₙ → μ in probability by the WLLN. Since μ ≠ 0, Slutsky's division rule gives Vₙ → N(0, σ²)/μ = N(0, σ²/μ²). (You could also get this from the delta method with g(x) = 1 − μ/x, which has g′(μ) = 1/μ; this is a good check.)

### Problem 13 (advanced, Borel–Cantelli)

A data link has independent frame errors, with frame n in error with probability pₙ. For each case, decide whether errors stop eventually (finitely many errors) with probability 1: (a) pₙ = 2⁻ⁿ, (b) pₙ = 1/n, (c) pₙ = 1/(n ln n) for n ≥ 2, (d) pₙ = 1/(n (ln n)²) for n ≥ 3, (e) pₙ = 0.001 for all n.

**Solution.** Use BC1 when Σpₙ < ∞ (errors stop) and BC2 when Σpₙ = ∞ (errors never stop, since the frames are independent).

| Case | Σ pₙ | Conclusion |
| --- | --- | --- |
| (a) 2⁻ⁿ | = 1, finite | Finitely many errors, with probability 1 |
| (b) 1/n | Diverges (harmonic) | Infinitely many errors, with probability 1 |
| (c) 1/(n ln n) | Diverges (integral is ln ln x) | Infinitely many errors, with probability 1 |
| (d) 1/(n (ln n)²) | Finite (integral is −1/ln x) | Finitely many errors, with probability 1 |
| (e) 0.001 | Diverges | Infinitely many errors, with probability 1 |

Case (e) is the realistic one: any fixed error rate, however small, produces infinitely many errors over an infinite time. Cases (c) and (d) show that the boundary between "stops" and "never stops" is extremely fine.

### Problem 14 (advanced): a full proof of the strong law with finite variance

Let X₁, X₂, … be iid, non-negative, with mean μ and finite variance σ². (a) Show that X̄ₙ → μ almost surely along the subsequence n = k² (k = 1, 2, 3, …). (b) Use non-negativity to fill in the gaps and conclude X̄ₙ → μ almost surely. (c) Remove the non-negativity assumption.

**Solution.**

(a) By Chebyshev, P(|X̄\_{k²} − μ| > ε) ≤ σ²/(k²ε²). Since Σ 1/k² < ∞, the first Borel–Cantelli lemma gives X̄\_{k²} → μ almost surely. The trick: along the full sequence the Chebyshev bounds σ²/(nε²) are not summable, but along the sparse subsequence n = k² they are.

(b) Because Xᵢ ≥ 0, the partial sums Sₙ are non-decreasing in n. For k² ≤ n < (k + 1)²,

```latex
\frac{S_{k^2}}{(k+1)^2}\ \le\ \frac{S_n}{n}\ \le\ \frac{S_{(k+1)^2}}{k^2}
\quad\Longleftrightarrow\quad
\bar X_{k^2}\cdot\frac{k^2}{(k+1)^2}\ \le\ \bar X_n\ \le\ \bar X_{(k+1)^2}\cdot\frac{(k+1)^2}{k^2}.
```

As n → ∞, k → ∞, and the ratios k²/(k + 1)² and (k + 1)²/k² both tend to 1. By part (a), both outer terms tend to μ almost surely. By the squeeze theorem, applied outcome by outcome, X̄ₙ → μ almost surely.

(c) Write Xᵢ = Xᵢ⁺ − Xᵢ⁻, where Xᵢ⁺ = max(Xᵢ, 0) and Xᵢ⁻ = max(−Xᵢ, 0) are non-negative, iid, with finite variances. Apply (b) to each, and subtract (almost sure limits can be subtracted): X̄ₙ → E\[X⁺\] − E\[X⁻\] = μ almost surely. ∎

This proves the SLLN under a finite variance, using only Chebyshev, Borel–Cantelli and a squeeze. Kolmogorov's full version needs only a finite mean.

## Part 15: Self-check quizzes

Twenty-one quick questions test whether the ideas have really landed; aim for at least 18 correct before moving on. Answer all of them first, then check the answer key at the end, which explains every answer and points back to the relevant Part.

### Quiz A: definitions and intuition (Parts 1 to 6)

1. Which mode of convergence is a statement only about the CDFs of Xₙ and X, and not about their values?
   1. Almost surely
   2. In probability
   3. In mean square
   4. In distribution
2. Independent Xₙ, with Xₙ = n with probability 1/n and 0 otherwise. Toward 0, Xₙ converges:
   1. In probability (and in distribution) only
   2. In probability and in L¹
   3. In every mode
   4. In no mode
3. True or false: if Xₙ → X almost surely, then E\[Xₙ\] → E\[X\].
4. Which condition is equivalent to Xₙ → c in mean square, for a constant c?
   1. E\[Xₙ\] → c
   2. Var(Xₙ) → 0
   3. E\[Xₙ\] → c and Var(Xₙ) → 0
   4. P(Xₙ = c) → 1
5. Independent Xₙ \~ Bernoulli(pₙ). For which pₙ does Xₙ → 0 almost surely?
   1. pₙ = 1/n
   2. pₙ = 1/√n
   3. pₙ = 1/n^1.5
   4. pₙ = 1/ln n
6. Why does the definition of convergence in distribution ignore the points where F jumps?
   1. To make proofs shorter
   2. So that Xₙ = 1/n converges to 0, as it obviously should
   3. Because CDFs are undefined at jumps
   4. Because jumps have probability zero

### Quiz B: relationships and the limit theorems (Parts 7 to 10)

7. Xₙ → c in distribution, where c is a constant. What else must be true?
   1. Nothing else
   2. Xₙ → c in probability
   3. Xₙ → c almost surely
   4. Xₙ → c in mean square
8. Which statement of the CLT is correct for iid data with mean μ and variance σ²?
   1. X̄ₙ → N(μ, σ²) in distribution
   2. √n(X̄ₙ − μ) → N(0, σ²) in distribution
   3. √n(X̄ₙ − μ) → 0 in probability
   4. X̄ₙ → μ in distribution, at rate 1/n
9. True or false: the standardised mean Zₙ = √n(X̄ₙ − μ)/σ converges in probability to a standard normal random variable.
10. To estimate a proportion within ±0.05 with 95% confidence, Chebyshev and the CLT require roughly:
    1. 385 and 2000 samples
    2. 2000 and 385 samples
    3. 400 and 400 samples
    4. 100 and 385 samples
11. Why can't Chebyshev plus the first Borel–Cantelli lemma prove the strong law directly along the full sequence?
    1. Chebyshev needs independence
    2. The bounds σ²/(nε²) are not summable
    3. Borel–Cantelli needs independence
    4. Chebyshev only gives bounds above 1
12. The sliding hump sequence of Example 4.5 converges to 0:
    1. Almost surely only
    2. In probability and in mean square, but not almost surely
    3. In every mode
    4. In distribution only
13. iid Cauchy observations. What happens to their running mean?
    1. It converges to 0
    2. It converges, but slowly
    3. It never settles, because the mean does not exist
    4. It converges in distribution to a normal

### Quiz C: tools and applications (Parts 11 and 12)

14. True or false: if Xₙ → X in distribution and Yₙ → Y in distribution, then Xₙ + Yₙ → X + Y in distribution.
15. Slutsky's theorem requires that:
    1. Both sequences converge almost surely
    2. The second sequence converges in probability to a constant
    3. The two sequences are independent
    4. The limits are normal
16. X̄ estimates μ = 2 with standard error 0.1. By the delta method, the standard error of X̄² is approximately:
    1. 0.01
    2. 0.1
    3. 0.2
    4. 0.4
17. Averaging 400 independent noisy sweeps improves the power SNR by about:
    1. 13 dB
    2. 20 dB
    3. 26 dB
    4. 52 dB
18. Which estimator of the mean μ is NOT consistent?
    1. X̄ₙ
    2. X̄ₙ + 1/n
    3. X₁ (the first observation only)
    4. The average of the last half of the sample

### Quiz D: the fine print (Sections 4.2, 4.3, 6.4 and 7.3)

19. Let Xₙ \~ Uniform(0, n). Then Xₙ:
    1. Converges in distribution to 0
    2. Converges in distribution to Uniform(0, 1)
    3. Does not converge in distribution
    4. Converges in mean square to n/2
20. You know only that P(|Xₙ − X| > ε) = 1/n for every n and every small ε. Which statement is correct?
    1. Xₙ → X almost surely
    2. Xₙ cannot converge to X almost surely
    3. Xₙ → X in probability; almost sure convergence depends on how the Xₙ are related
    4. Xₙ → X in mean square
21. Xₙ → X in distribution. Which extra condition guarantees E\[Xₙ\] → E\[X\]?
    1. The Xₙ are independent
    2. The Xₙ are uniformly integrable
    3. X is normal
    4. Each Xₙ has a finite mean

### Answer key

| Q | Answer | Why |
| --- | --- | --- |
| 1 | 4 | Convergence in distribution compares CDFs only (Part 6). |
| 2 | 1 | P(Xₙ ≠ 0) = 1/n → 0, but E∣Xₙ∣ = 1 for all n (Examples 3.6 and 5.4). With independence and Σ 1/n = ∞, BC2 says the spikes recur forever, so there is no almost sure convergence either. |
| 3 | False | ω \~ Uniform(0, 1), Xₙ = n when ω < 1/n: converges to 0 almost surely, yet E\[Xₙ\] = 1 (Part 7). |
| 4 | 3 | E\[(Xₙ − c)²\] = Var(Xₙ) + bias² (Section 5.2). |
| 5 | 3 | Only Σ 1/n^1.5 is finite; the other sums diverge, and BC2 applies (Section 4.3). |
| 6 | 2 | At x = 0 the CDF of 1/n is 0 while the limit's CDF is 1 (Example 6.1). |
| 7 | 2 | Convergence in distribution to a constant is equivalent to convergence in probability (Section 7.1 (e)). |
| 8 | 2 | The √n scaling keeps the variance at σ² (Section 10.2). Option 1 is wrong because X̄ₙ → μ, a constant. |
| 9 | False | Corr(Zₙ, Z₂ₙ) = 1/√2, so the paths never settle (Section 10.4). |
| 10 | 2 | Examples 8.1 and 10.3. |
| 11 | 2 | Σ 1/n diverges; fourth moments or a subsequence trick are needed (Section 9.2, Problem 14). |
| 12 | 2 | P(Xₙ = 1) and E\[Xₙ²\] equal the interval length, which shrinks, but every ω is hit infinitely often (Example 4.5). |
| 13 | 3 | X̄ₙ is standard Cauchy for every n (Example 8.4, Simulation 5). |
| 14 | False | Xₙ = Z, Yₙ = −Z gives Xₙ + Yₙ = 0 (Example 11.3). |
| 15 | 2 | Section 11.3. |
| 16 | 4 | g(x) = x², g′(2) = 4, so se ≈ 4 × 0.1 = 0.4 (Section 11.4). |
| 17 | 3 | 10·log₁₀(400) ≈ 26 dB (Section 12.3). |
| 18 | 3 | Its variance σ² never shrinks (Example 12.1). The half-sample average has variance 2σ²/n → 0, so it is consistent, just inefficient. |
| 19 | 3 | Fₙ(x) = x/n → 0 for every x, which is not a CDF: the probability escapes to infinity (Example 6.8). |
| 20 | 3 | Summability is sufficient, not necessary; independent Xₙ fail (BC2), nested events succeed (Section 4.2). |
| 21 | 2 | Finite means are not enough: Xₙ = n with probability 1/n has mean 1 but converges to 0 (Section 7.3). |

**Scoring guide.** 19 to 21: excellent, go to the cheat sheet. 14 to 18: reread the Sections referenced by your mistakes and redo their Check yourself exercises. Below 14: redo Parts 3 to 7, including the classification drill at the end of Part 7, before continuing.

## Part 16: One-page cheat sheet

Everything you must remember fits on one page: four definitions, one hierarchy, eleven theorems and six pitfalls. The counterexamples are collected in Section 7.2, and the classification drill at the end of Part 7 is the best final rehearsal.

### The four modes

| Mode | Definition | In words | How to prove it | Typical use |
| --- | --- | --- | --- | --- |
| Almost surely | P(Xₙ → X) = 1 | Every path (except a probability-zero set) converges | Pointwise argument; Σ P(∣Xₙ − X∣ > ε) < ∞ (sufficient only); monotone + in probability | SLLN, one long simulation or record |
| In probability | P(∣Xₙ − X∣ > ε) → 0 for every ε > 0 | At a fixed large n, a large error is unlikely | Exact probability; Markov or Chebyshev; MSE → 0 | Consistency, WLLN |
| In Lᵖ (mean square if p = 2) | E∣Xₙ − X∣ᵖ → 0 | Average error (power) goes to 0 | To a constant c: bias → 0 and variance → 0 | MSE, filtering, signal averaging |
| In distribution | Fₙ(x) → F(x) at continuity points of F | Histograms match; values need not be close | CDF, PMF, MGF limits | CLT, Poisson approximation, confidence intervals, tests |

**Hierarchy.** Almost surely ⇒ in probability ⇒ in distribution. Lʳ ⇒ Lᵖ ⇒ in probability, for r > p ≥ 1. In distribution to a constant ⇒ in probability. Almost sure and Lᵖ do not imply each other.

### The key theorems

| Theorem | Assumptions | Conclusion |
| --- | --- | --- |
| Markov | Y ≥ 0, a > 0 | P(Y ≥ a) ≤ E\[Y\]/a |
| Chebyshev | Finite variance σ² | P(∣X − μ∣ ≥ ε) ≤ σ²/ε² |
| Weak LLN | Uncorrelated, mean μ, variance σ² | X̄ₙ → μ in probability and in mean square |
| Strong LLN | iid, E∣X₁∣ < ∞ | X̄ₙ → μ almost surely |
| CLT | iid, 0 < σ² < ∞ | √n(X̄ₙ − μ) → N(0, σ²) in distribution |
| Borel–Cantelli 1 | Σ P(Aₙ) < ∞ | Only finitely many Aₙ occur, with probability 1 |
| Borel–Cantelli 2 | Independent, Σ P(Aₙ) = ∞ | Infinitely many Aₙ occur, with probability 1 |
| Continuous mapping | g continuous at the values X takes (a jump at the limit point breaks it); Xₙ → X a.s., in probability or in distribution | g(Xₙ) → g(X) in the same mode |
| Slutsky | Xₙ → X in distribution, Yₙ → c in probability | Xₙ + Yₙ → X + c, YₙXₙ → cX, Xₙ/Yₙ → X/c (c ≠ 0) |
| Delta method | √n(Tₙ − θ) → N(0, τ²), g′(θ) ≠ 0 | √n(g(Tₙ) − g(θ)) → N(0, g′(θ)²τ²) |
| Uniform integrability | Xₙ → X in distribution; tails uniformly negligible, e.g. E∣Xₙ∣^{1+δ} ≤ C | E\[Xₙ\] → E\[X\] |

### Pitfalls to avoid

- Summable bad probabilities prove almost sure convergence; a divergent sum disproves it only for independent Xₙ.
- Convergence in distribution ignores values, needs a genuine CDF as its limit, and does not let you add or multiply two limits unless the pair converges jointly, which a constant limit (Slutsky) or independence guarantees.
- Convergence in probability or in distribution does not make means converge; you need Lᵖ, a bound or uniform integrability.
- Averaging removes independent noise, never shared (common-mode) error, and correlated samples count for less.
- The LLN needs a finite mean and the CLT a finite variance; heavy tails (Cauchy) break them.
- The CLT is an approximation that improves like 1/√n: check skewness, use a continuity correction for counts, and distrust it far in the tails.
