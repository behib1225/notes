# Chapter 9 — Hypothesis Testing and Confidence Sets

## Chapter 9 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 9.0 | Orientation: a test is a measurable decision rule, and everything else is bookkeeping about its error rates | Tells you what "significant" can and cannot buy you |
| 9.1 | Null, alternative, size, level, power. Power is bounded by how far apart the two hypotheses actually are | Explains why a benchmark with 500 test items cannot resolve a 1-point accuracy gap, no matter which test you run |
| 9.2 | Neyman–Pearson: the optimal test thresholds a likelihood ratio; MLR families give uniformly most powerful one-sided tests | Why ROC curves are the right object for detectors, why the optimal detector is a calibrated probability, and why AUC has a likelihood-ratio interpretation |
| 9.3 | Likelihood ratio, Wald, and score tests; Wilks' theorem $-2\log\Lambda \to_d \chi^2_k$ | Nested-model comparison (ablations!), and why Wald intervals on a probability scale misbehave |
| 9.4 | $p$-values: uniform under a continuous null, super-uniform in general; what they are not | $p$-hacking, forking paths, optional stopping — the arithmetic of a "significant" result being mostly wrong |
| 9.5 | Confidence sets as inverted tests; pivots; Wald vs score vs likelihood-ratio vs exact intervals | Honest error bars on accuracy near 0 or 1, where the textbook Wald interval collapses |
| 9.6 | FWER, Bonferroni (= the union bound), Holm, and FDR control by Benjamini–Hochberg | Hyperparameter sweeps, feature selection, leaderboards, the winner's curse, holdout decay |
| 9.7 | Permutation tests: exact finite-sample validity from exchangeability alone | The right default for "does model A beat model B", assumption-light and exact |
| 9.8 | The bootstrap: plug in $\hat P_n$ for $P$. Percentile, basic, studentized, BCa. And where it breaks | Error bars for AUC, F1, BLEU — statistics with no closed-form variance; plus a catalogue of failure modes |
| 9.9 | Evaluating ML experiments honestly: paired tests, McNemar, CV variance, seed budgets, intervals for accuracy/AUC/$F_1$, clustered items, practical significance | A protocol you can defend in review, and the reasons most reported gaps are not gaps |

## 9.0 What this chapter is for

Every empirical claim in a machine learning paper has the same logical shape. We changed something; a number moved; we assert the change caused the movement. The number moved by 0.4 points. Test sets are finite, initializations are random, data order is shuffled, and the number would have moved *somewhat* even if we had changed nothing at all. The question is not whether the number moved. The question is whether it moved by more than nothing-in-particular moves it.

That question has a precise formulation, and this chapter builds it. The formulation splits into two halves that turn out to be the same object viewed from two sides. A **test** takes data and returns a verdict about a hypothesis while controlling how often it returns a false verdict. A **confidence set** takes data and returns a set of parameter values that are not refuted, with a guarantee about how often the truth is inside. Theorem 9.23 will show that each is mechanically obtainable from the other: a confidence set is the collection of hypotheses your test does not reject. Once you see that, you stop thinking of tests and intervals as separate topics.

You bring the following from earlier chapters, and I will use them by name without re-deriving them. From Chapter 2: the Radon–Nikodym theorem, which is what lets us write a likelihood ratio $d P_1/d P_0$ even when neither measure has a Lebesgue density, and dominated convergence. From Chapter 3: exponential families, natural parameters, and sufficiency — the structural reason one-sided tests in these families are optimal rather than merely reasonable. From Chapter 4: concentration inequalities and the observation that the union bound goes vacuous once $\sum_n \mathbb{P}(A_n) \ge 1$. From Chapter 7: the central limit theorem, Slutsky's theorem, the delta method, Glivenko–Cantelli, and the DKW inequality. From Chapter 8: asymptotic normality of the maximum likelihood estimator, Fisher information, and the sandwich variance formula for misspecified models. From Chapter 1, by number, I will use Definition 1.10 (conditional probability), Definition 1.18 (the CDF), Definition 1.23 (independence), Proposition 1.8(2) (the union bound), Proposition 1.12 (Bayes' rule and the prior/likelihood/evidence/posterior vocabulary), and Proposition 1.19 (properties of CDFs, including that jumps of $F_X$ are atoms of $P_X$).

Here is the punchline, stated in advance so you can watch it arrive. **Optimal testing is thresholding a likelihood ratio** (Theorem 9.7); everything else in classical testing is either an approximation to that or an accounting device for the fact that we usually have composite hypotheses and finite samples. **A $p$-value is a random variable that is uniform on $[0,1]$ when the null is true** (Proposition 9.18); it is therefore a measure of surprise, not a probability that the null is true, and confusing the two is arithmetically catastrophic at low prior odds — the same base-rate arithmetic that made Chapter 1's disease test yield $\mathbb{P}(D \mid +) \approx 9.05\%$ despite 99% sensitivity. **Confidence sets are inverted tests** (Theorem 9.23), so a bad test gives a bad interval and you can diagnose one by studying the other. And **the multiplicity of things we try is the dominant source of false claims in machine learning**, because a hyperparameter sweep is a multiple-testing problem with hundreds of comparisons that nobody counts, and a public leaderboard is one with thousands.

**How to use this chapter on your own.** Read §9.1 and §9.2 in order; they are load-bearing and everything later leans on them. After that the chapter is more modular than it looks: §9.3 (the three statistics), §9.5 (intervals), §9.6 (multiplicity), §9.7 (permutation), and §9.8 (bootstrap) can be read in any order, and §9.9 depends only on §9.5 through §9.8. Each section ends with a self-check whose answers are in Appendix A; do those before moving on, because they are calibration, not review. Every number quoted in the text came out of code, and the seeds and sample sizes are stated so you can reproduce or contradict them — attempting a few of the Part B labs before reading their solutions is by far the most efficient use of your time here, since the failure modes in §9.4 and §9.8 are the kind of thing you believe only after watching your own simulation break. If you are short on time, the minimum path to competence at reading ML papers is §9.1, §9.4, §9.6, and §9.9.

I want to be honest about one thing before we start. Classical testing was designed for a world of small, expensive, pre-planned experiments: agricultural field trials, clinical studies. Machine learning experimentation is the opposite — enormous samples, near-zero marginal cost per experiment, and a search process that touches the evaluation data thousands of times. Some classical machinery transfers beautifully (permutation tests, paired comparisons, the duality above). Some transfers badly (any $p$-value computed after an unlogged search). Section 9.9 is where I say which is which, and it is the longest applied section in the chapter for a reason.

## 9.1 The testing framework

### 9.1.1 What a test is

Start with the object we are uncertain about. We observe data $X$ taking values in a measurable space $(\mathcal{X}, \mathcal{A})$ — usually $\mathcal{X} = \mathbb{R}^n$ with its Borel sets, $X = (X_1,\dots,X_n)$ — and we entertain a family of candidate distributions for it.

> **Definition 9.1 (statistical model and hypotheses).** A *statistical model* is a family $\mathcal{P} = \{P_\theta : \theta \in \Theta\}$ of probability measures on $(\mathcal{X},\mathcal{A})$. A *hypothesis* is a subset of $\Theta$. A *testing problem* is a partition of $\Theta$ into a *null hypothesis* $\Theta_0$ and an *alternative* $\Theta_1 = \Theta \setminus \Theta_0$; we write $H_0 : \theta \in \Theta_0$ and $H_1 : \theta \in \Theta_1$. The hypothesis is *simple* if the corresponding subset is a single point, and *composite* otherwise.

Two things about this definition deserve emphasis, because both are routinely lost in practice.

First, $\Theta$ is a modelling choice, and the model is an assumption, not a fact. When you test $H_0: \mu = 0$ in the model $\{N(\mu, 1)\}$, you have already assumed Gaussianity and unit variance. Rejecting $H_0$ is evidence against the *conjunction* of the null and the model. Nearly every "the test was wrong" story is really a "the model was wrong" story.

Second, the asymmetry between $H_0$ and $H_1$ is not aesthetic. It is baked into what we will control. We will control the probability of falsely rejecting $H_0$ and merely try to make the probability of falsely retaining it small. So $H_0$ must be the hypothesis whose false rejection is the error you care about. In ML: $H_0$ is "my new method is no better than the baseline". You want a procedure that rarely announces an improvement that isn't there.

> **Definition 9.2 (test, rejection region, randomized test).** A *test* is a measurable function $\varphi : \mathcal{X} \to [0,1]$. Observing $x$, we reject $H_0$ with probability $\varphi(x)$ (drawing an independent auxiliary uniform if $0 < \varphi(x) < 1$). If $\varphi$ takes only the values $0$ and $1$ the test is *non-randomized*, and $R := \{x : \varphi(x) = 1\} \in \mathcal{A}$ is its *rejection region* (or *critical region*); then $\varphi = \mathbf{1}_R$.

Measurability is not decoration here. It is exactly the condition under which $\mathbb{E}_\theta[\varphi(X)]$ — the rejection probability — is defined at all. This is the first payoff of Chapter 1's insistence that a random variable is a measurable function: a "decision rule" that is not measurable has no error rate, and a procedure with no error rate is not a test. Chapter 1 also warned that suprema over *uncountable* families need not be measurable; we will meet that again in §9.5 when we build simultaneous confidence bands, where the object of interest is $\sup_t |\hat F_n(t) - F(t)|$ and measurability is rescued by right-continuity rather than assumed.

> **Definition 9.3 (size, level, power, error types).** For a test $\varphi$ define its *power function*
> $$\beta_\varphi(\theta) := \mathbb{E}_\theta[\varphi(X)], \qquad \theta \in \Theta .$$
> The *size* of $\varphi$ is $\sup_{\theta \in \Theta_0} \beta_\varphi(\theta)$. The test has *level* $\alpha$ if its size is $\le \alpha$. For $\theta \in \Theta_1$ we call $\beta_\varphi(\theta)$ the *power at* $\theta$. Rejecting when $\theta \in \Theta_0$ is a *Type I error*; failing to reject when $\theta \in \Theta_1$ is a *Type II error*, with probability $1 - \beta_\varphi(\theta)$.

One power function, read on two different parts of $\Theta$: on $\Theta_0$ it is the thing we cap, on $\Theta_1$ the thing we maximize. Size and level are not synonyms — size is a property of the test, level is a promise the test keeps. Every test of size $0.021$ is also a test of level $0.05$. The distinction is idle for continuous statistics and unavoidable for discrete ones, as the next two examples show.

**Worked Example 9.1 (three coin flips: level, size, and the tyranny of small $n$).** Take Chapter 1's running example: $\Omega = \{H,T\}^3$, $X$ = number of heads. Model the flips as i.i.d. Bernoulli($p$) and test $H_0 : p = 1/2$ against $H_1 : p > 1/2$. Under $H_0$, $X \sim \text{Bin}(3, 1/2)$ with $\mathbb{P}(X = k) = 1/8, 3/8, 3/8, 1/8$ for $k = 0,1,2,3$.

Suppose we want level $\alpha = 0.05$. Consider the natural one-sided rejection regions. Rejecting when $X = 3$ has size $\mathbb{P}_{1/2}(X = 3) = 0.125$. Rejecting when $X \ge 2$ has size $0.500$. There is nothing in between. So *the only non-randomized test of level $0.05$ is the test that never rejects*: $\varphi \equiv 0$, size $0$, power $0$ everywhere.

That is not a defect of the test; it is information about the experiment. Three flips cannot distinguish a fair coin from a biased one at the 5% level, because the most extreme thing three flips can do — all heads — happens one time in eight for a fair coin. If you ever find yourself running a significance test on a benchmark with a handful of evaluation items, this is the situation you are in, and no cleverness recovers it. The fix is more data, not a better test.

**Worked Example 9.2 (attainable sizes, and what randomization buys).** Now $n = 20$ flips, same hypotheses. The upper-tail probabilities under $H_0$ are (exact, from the binomial CDF):

| $k$ | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|
| $\mathbb{P}_{1/2}(X \ge k)$ | 0.251722 | 0.131588 | 0.057659 | 0.020695 | 0.005909 | 0.001288 | 0.000201 |

No rejection region of the form $\{X \ge k\}$ has size $0.05$. The test $\{X \ge 14\}$ has size $0.057659$ — it is *not* a level-$0.05$ test, though it is often used as one. The test $\{X \ge 15\}$ has size $0.020695$: valid, but it throws away power, because it demands more evidence than the 5% budget requires.

A randomized test recovers the missing $0.029$. Set
$$\varphi(x) = \begin{cases} 1, & x \ge 15,\\ \gamma, & x = 14,\\ 0, & x \le 13,\end{cases} \qquad \gamma = \frac{0.05 - 0.020695}{\mathbb{P}_{1/2}(X = 14)} = \frac{0.029305}{0.036964} = 0.792797 .$$
Its size is exactly $0.050000$. Its power at $p = 0.7$ is $\mathbb{P}_{0.7}(X \ge 15) + \gamma\,\mathbb{P}_{0.7}(X = 14) = 0.416371 + 0.792797 \times 0.191639 = 0.568302$, against $0.416371$ for the conservative non-randomized test — a gain of 15 percentage points of power for free, in the sense of costing no additional data.

And yet almost nobody randomizes. The reason is not statistical but institutional: a procedure that can reject on Monday and retain on Tuesday from the same data is a hard sell, and the auxiliary coin flip carries no information about $\theta$. Randomized tests matter to us for a theoretical reason instead: they make the set of achievable (size, power) pairs *convex*, which is what makes the Neyman–Pearson lemma clean and the ROC curve concave. Keep them in the toolbox as a device, not as advice.

> **Common Confusion 9.1.** *"We failed to reject, so the two models perform the same."*
>
> No. Worked Example 9.1 exhibits a test that never rejects anything, ever, at level $0.05$. By your reading it proves all models are identical. The asymmetry in Definition 9.3 is total: level controls false rejections and says nothing whatsoever about false retentions. A non-rejection is consistent with (i) no difference, (ii) a real difference and an underpowered experiment, or (iii) a real difference and a badly chosen statistic. Distinguishing these requires reporting power or, better, a confidence interval (§9.5): "the 95% interval for the accuracy gap is $[-0.4\%, +0.6\%]$" is an informative non-result, because it excludes large gaps. "$p = 0.31$" is not, because it excludes nothing. If your claim is genuinely *equivalence*, you need an equivalence test, which reverses the roles: $H_0$ becomes "the gap exceeds $\delta$" and rejecting it establishes near-equality.

**Equivalence testing in one paragraph (TOST).** Since "our compressed model matches the baseline" is one of the most common claims in ML papers, here is the standard procedure, in full. Pre-declare a margin $\delta > 0$: the largest gap you are willing to call negligible. The null is $H_0 : |\Delta| \ge \delta$ and the alternative is $H_1 : |\Delta| < \delta$. Run *two one-sided tests* (hence "TOST"): test $H_{0}^{-} : \Delta \le -\delta$ against $\Delta > -\delta$, and $H_{0}^{+} : \Delta \ge \delta$ against $\Delta < \delta$, each at level $\alpha$, and declare equivalence only if **both** reject. This controls the Type I error at $\alpha$ (not $2\alpha$) because $H_0$ is the union of the two one-sided nulls and a false declaration requires rejecting whichever of them is true — an intersection–union argument, not a multiplicity problem. Operationally, TOST at level $\alpha$ is the same as checking that the $(1-2\alpha)$ confidence interval for $\Delta$ lies entirely inside $(-\delta,\delta)$. A computed instance: a paired comparison on $n = 10{,}000$ items gives an estimated accuracy gap $\hat\Delta = 0.0002$ with paired standard error $0.002$, and the pre-declared margin is $\delta = 0.005$ (half a point). Then $z^- = (\hat\Delta + \delta)/\mathrm{se} = 2.60$ ($p = 0.0047$) and $z^+ = (\delta - \hat\Delta)/\mathrm{se} = 2.40$ ($p = 0.0082$): both one-sided tests reject at $\alpha = 0.05$, and equivalently the 90% interval $[-0.0031,\ 0.0035]$ sits inside $(-0.005, 0.005)$, so "the compressed model is within half a point of the baseline" is established at the 5% level. Note what the same data do *not* establish: $p = 0.92$ for the ordinary two-sided test of $\Delta = 0$ would prove nothing on its own; the margin $\delta$, declared in advance, is what turns a non-rejection into a claim.

### 9.1.2 Power is bounded by how different the hypotheses are

Before optimizing power, notice that power has a ceiling nobody can beat, and that the ceiling is a property of the *hypotheses*, not of the test. Testing is only as easy as the distributions are distinguishable.

> **Definition 9.4 (total variation distance).** For probability measures $P, Q$ on $(\mathcal{X},\mathcal{A})$,
> $$\|P - Q\|_{\mathrm{TV}} := \sup_{A \in \mathcal{A}} |P(A) - Q(A)| .$$
> If both have densities $p, q$ with respect to a common dominating measure $\mu$ (Chapter 2 guarantees $\mu = P + Q$ works), then $\|P-Q\|_{\mathrm{TV}} = \tfrac12 \int |p - q| \, d\mu$.

This is the same total variation distance you met alongside KL divergence in the divergences chapter; if you have not, the definition above is self-contained and everything below follows from it.

> **Proposition 9.5 (the power ceiling).** Let $\varphi$ be any test and $P_0, P_1$ any two probability measures on $(\mathcal{X},\mathcal{A})$. Then
> $$\big|\mathbb{E}_{P_1}[\varphi] - \mathbb{E}_{P_0}[\varphi]\big| \le \|P_1 - P_0\|_{\mathrm{TV}} .$$
> Consequently a test of size $\le \alpha$ against the simple alternative $P_1$ has power at most $\alpha + \|P_1 - P_0\|_{\mathrm{TV}}$.

*Proof.* Let $\mu = P_0 + P_1$, which dominates both, and let $p_0 = dP_0/d\mu$, $p_1 = dP_1/d\mu$ be Radon–Nikodym derivatives (Ch. 2). Then
$$\mathbb{E}_{P_1}[\varphi] - \mathbb{E}_{P_0}[\varphi] = \int \varphi \,(p_1 - p_0)\, d\mu \le \int \varphi \,(p_1-p_0)^+ \, d\mu \le \int (p_1 - p_0)^+ d\mu,$$
using $0 \le \varphi \le 1$ and dropping the negative part. Since $\int (p_1 - p_0) d\mu = 1 - 1 = 0$, we have $\int (p_1-p_0)^+ d\mu = \int (p_1-p_0)^- d\mu = \tfrac12 \int|p_1-p_0| d\mu = \|P_1-P_0\|_{\mathrm{TV}}$. Applying the same argument with $\varphi$ replaced by $1 - \varphi$ (also a test) gives the other direction of the absolute value. The consequence follows by taking $\mathbb{E}_{P_0}[\varphi] \le \alpha$. $\blacksquare$

Note where the bound is attained: by $\varphi = \mathbf{1}\{p_1 > p_0\}$, i.e. by thresholding the likelihood ratio at $1$. That is not a coincidence; it is Theorem 9.7 arriving early.

**Worked Example 9.3 (how far apart is "far apart"?).** Take $P_0 = N(0,1)$, $P_1 = N(\mu, 1)$, one observation. The two densities cross at $x = \mu/2$, so
$$\|P_1 - P_0\|_{\mathrm{TV}} = \mathbb{P}_1(X > \mu/2) - \mathbb{P}_0(X > \mu/2) = 2\Phi(\mu/2) - 1 .$$
For $\mu = 0.5$ this is $0.197413$. The best level-$0.05$ test (reject when $X > z_{0.95} = 1.644854$) has power $1 - \Phi(1.644854 - 0.5) = 0.126135$, comfortably under the ceiling $0.05 + 0.197413 = 0.247413$.

| $\mu$ | $\|P_1-P_0\|_{\mathrm{TV}}$ | best power at $\alpha = 0.05$ | ceiling $\alpha + \mathrm{TV}$ |
|---|---|---|---|
| 0.5 | 0.1974 | 0.1261 | 0.2474 |
| 1.0 | 0.3829 | 0.2595 | 0.4329 |
| 2.0 | 0.6827 | 0.6388 | 0.7327 |
| 4.0 | 0.9545 | 0.9907 | 1.0045 |

Read the first row as a warning. With one observation and a half-sigma effect, *no* level-5% test can exceed 25% power, so an experiment that reports "no significant difference" has told you almost nothing. Read the last row as the reason large $n$ helps: for $n$ i.i.d. observations the effective separation is $\sqrt{n}\mu$, so TV rises to 1 and the ceiling stops binding. Concretely, this is why §9.9 will insist on paired comparisons: pairing does not change the models being compared, it changes the *distance between the two hypotheses about the paired difference*, which is what Proposition 9.5 says power depends on.

**Worked Example 9.4 (a power calculation you will actually run).** Model $X_1,\dots,X_n \overset{\text{iid}}{\sim} N(\mu, 1)$, test $H_0: \mu = 0$ vs $H_1: \mu > 0$, reject when $\sqrt{n}\bar X_n > z_{1-\alpha}$. Under $\mu$, $\sqrt{n}\bar X_n \sim N(\sqrt n \mu, 1)$, so
$$\beta(\mu) = 1 - \Phi(z_{1-\alpha} - \sqrt{n}\,\mu).$$
With $n = 25$, $\mu = 0.4$, $\alpha = 0.05$: $\sqrt{25}\times 0.4 = 2$, so $\beta = 1 - \Phi(1.644854 - 2) = 1 - \Phi(-0.355146) = 0.638760$. The two-sided version, rejecting when $|\sqrt n \bar X_n| > z_{0.975} = 1.959964$, has power $0.516005$ — two-sidedness costs 12 points of power here, which is the price of not having to know the sign in advance.

Inverting the one-sided formula gives the sample size rule you should have memorized: to reach power $1-\beta$ against effect size $\delta$ (in units of the standard deviation),
$$n \ge \frac{(z_{1-\alpha} + z_{1-\beta})^2}{\delta^2}.$$
The constant for $\alpha = 0.025$ one-sided (= 0.05 two-sided) and $80\%$ power is $(1.959964 + 0.841621)^2 = 7.8489$. So $n \approx 7.85/\delta^2$. A quarter-sigma effect needs $n \approx 126$; a tenth-sigma effect needs $n \approx 785$. This single constant will reappear in §9.9 when we ask how many random seeds a paper needs.

<svg viewBox="0 0 620 400" style="max-width:100%;height:auto" role="img" aria-label="Power curves versus effect size for four sample sizes">
  <line x1="70" y1="300" x2="575" y2="300" stroke="currentColor" stroke-width="1.5"/>
  <line x1="70" y1="300" x2="70" y2="55" stroke="currentColor" stroke-width="1.5"/>
  <text x="320" y="330" fill="currentColor" font-size="13" text-anchor="middle">effect size δ (in units of σ)</text>
  <text x="24" y="180" fill="currentColor" font-size="13" text-anchor="middle" transform="rotate(-90 24 180)">power β(δ)</text>
  <line x1="70" y1="288" x2="575" y2="288" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>
  <text x="580" y="291" fill="currentColor" font-size="11">α=0.05</text>
  <line x1="70" y1="108" x2="575" y2="108" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>
  <text x="580" y="111" fill="currentColor" font-size="11">0.80</text>
  <text x="60" y="304" fill="currentColor" font-size="11" text-anchor="end">0</text>
  <text x="60" y="184" fill="currentColor" font-size="11" text-anchor="end">0.5</text>
  <text x="60" y="64" fill="currentColor" font-size="11" text-anchor="end">1</text>
  <text x="70" y="316" fill="currentColor" font-size="11" text-anchor="middle">0</text>
  <text x="315" y="316" fill="currentColor" font-size="11" text-anchor="middle">0.5</text>
  <text x="560" y="316" fill="currentColor" font-size="11" text-anchor="middle">1.0</text>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="70,288 119,285.1 168,276.7 217,262.1 266,241.4 315,215.3 364,186.0 413,156.0 462,128.2 511,105.1 560,87.6"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2" points="70,288 119,281.0 168,259.2 217,222.5 266,176.2 315,130.8 364,95.8 413,74.9 462,65.0 511,61.4 560,60.2"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2" points="70,288 119,259.2 168,176.2 217,95.8 266,65.0 315,60.2 364,60 560,60"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2" points="70,288 119,176.2 168,65.0 217,60 560,60"/>
  <text x="565" y="84" fill="#3b82f6" font-size="11">n=10</text>
  <text x="470" y="58" fill="#10b981" font-size="11">n=25</text>
  <text x="300" y="52" fill="#f59e0b" font-size="11">n=100</text>
  <text x="180" y="52" fill="#ef4444" font-size="11">n=400</text>
  <text x="70" y="352" fill="currentColor" font-size="12">Figure 9.1  Power of the two-sided z-test at α=0.05 as a function of the true effect size, for four sample sizes.</text>
  <text x="70" y="370" fill="currentColor" font-size="12">Every curve starts at exactly α when δ=0: at the null, "power" is the Type I error rate. Quadrupling n halves the</text>
  <text x="70" y="388" fill="currentColor" font-size="12">effect size detectable at 80% power (δ = 0.560, 0.280, 0.140 for n = 25, 100, 400).</text>
</svg>

Figure 9.1 plots $\beta(\delta) = \Phi(\sqrt n \delta - 1.96) + \Phi(-\sqrt n \delta - 1.96)$ for $n \in \{10,25,100,400\}$. Three features are worth naming in words, in case the figure does not render. Every curve passes through $\beta(0) = 0.05$: at the boundary of the null, power *is* the size, which is why "the power of a test at the null" is a category error people nonetheless commit. Each curve is increasing and $S$-shaped, flat near $\delta = 0$ — meaning tiny effects are essentially undetectable, and no amount of $\alpha$-tuning changes that. And the curves shift left by a factor $\sqrt{n}$: at 80% power, the detectable effect sizes are $\delta = 0.560$ for $n=25$, $0.280$ for $n=100$, and $0.140$ for $n=400$. Detecting effects half as large costs four times the data. In ML terms, if your benchmark's item-level standard deviation is $0.3$ and you care about accuracy gaps of $0.005$, then $\delta = 0.0167$ and $n \approx 7.85/\delta^2 \approx 28{,}000$ items. Most benchmarks have fewer.

> **Self-check 9.1.**
> 1. A test has size $0.03$. Is it a level-$0.05$ test? Is it a level-$0.01$ test?
> 2. You are told a test has power $0.9$ against $\mu = 1$ and size $0.05$. What is the largest possible value of $\beta_\varphi(0.5)$ that is consistent with monotone power, and what does Proposition 9.5 tell you if $\|P_{0.5} - P_0\|_{\mathrm{TV}} = 0.2$?
> 3. Why can a randomized test achieve a size that no non-randomized test achieves, and why does the auxiliary randomness not smuggle in information about $\theta$?
> 4. A colleague reports "$p = 0.4$, so our compression method loses no accuracy." What single additional number would make this claim checkable?

## 9.2 Neyman–Pearson: the optimal test is a likelihood-ratio threshold

### 9.2.1 The lemma

Suppose both hypotheses are simple: $H_0 : P = P_0$ against $H_1: P = P_1$. Among all tests of level $\alpha$, which has the greatest power? The answer is completely explicit, and it is the single most consequential fact in this chapter.

The intuition is a knapsack argument. You have a budget: total null probability $\alpha$ that you are permitted to spend on rejection. You are shopping for alternative probability. Each candidate point $x$ has a "price" $p_0(x)$ (null probability spent) and a "value" $p_1(x)$ (alternative probability gained). To maximize value at fixed price you buy in order of *value per unit price*, which is $p_1(x)/p_0(x)$ — the likelihood ratio — until the budget is exhausted. That is the whole proof; the formal version below just handles ties and the possibility that $p_0(x) = 0$ (where the item is free, so you always take it).

> **Definition 9.6 (likelihood ratio).** Let $\mu$ dominate both $P_0$ and $P_1$ (e.g. $\mu = P_0 + P_1$), and let $p_i = dP_i/d\mu$ be Radon–Nikodym derivatives (Ch. 2). The *likelihood ratio* is
> $$L(x) := \frac{p_1(x)}{p_0(x)} \in [0,\infty],$$
> with the conventions $L(x) = +\infty$ when $p_0(x) = 0 < p_1(x)$, and $L(x)$ arbitrary (say $1$) when $p_0(x) = p_1(x) = 0$, a $\mu$-null set. The value of $L$ does not depend on the choice of dominating $\mu$ up to $\mu$-null sets.

The last sentence is worth pausing on. Likelihoods are not numbers attached to distributions; they are densities, and densities only exist relative to a base measure. Chapter 2 taught us that the *ratio* of two densities with respect to a common base is base-independent — the chain rule for Radon–Nikodym derivatives. This is why likelihood-ratio methods work for mixed discrete/continuous data, for the $A = \max(0,Z)$ variable of Chapter 1 that has neither pmf nor pdf, and for models on manifolds. A likelihood ratio is a more robust object than a likelihood.

> **Theorem 9.7 (Neyman–Pearson lemma).** Fix $\alpha \in (0,1)$. Consider testing $H_0: P = P_0$ vs $H_1 : P = P_1$.
>
> **(a) Existence.** There exist $k \in [0,\infty)$ and $\gamma \in [0,1]$ such that the test
> $$\varphi^*(x) = \begin{cases} 1 & \text{if } L(x) > k,\\ \gamma & \text{if } L(x) = k,\\ 0 & \text{if } L(x) < k,\end{cases}$$
> has size exactly $\alpha$: $\mathbb{E}_{0}[\varphi^*] = \alpha$.
>
> **(b) Sufficiency (optimality).** Any test of the above form with $\mathbb{E}_0[\varphi^*] = \alpha$ is *most powerful* at level $\alpha$: for every test $\varphi$ with $\mathbb{E}_0[\varphi] \le \alpha$, $\mathbb{E}_1[\varphi] \le \mathbb{E}_1[\varphi^*]$.
>
> **(c) Necessity.** If $\varphi$ is most powerful at level $\alpha$ then $\varphi$ has size exactly $\alpha$ (provided $\mathbb{E}_1[\varphi^*] < 1$), and $\varphi = \varphi^*$ $\mu$-almost everywhere on $\{L \ne k\}$; i.e. up to behaviour on the boundary set $\{L = k\}$, every most powerful test is a likelihood-ratio test.

*Proof.* **(a)** Let $G(t) := \mathbb{P}_0(L > t)$ for $t \ge 0$. By Proposition 1.19 applied to the random variable $L(X)$ under $P_0$, $G$ is non-increasing and right-continuous, with $G(t) \to 0$ as $t \to \infty$ and $G(0^-) = 1$. Set
$$k := \inf\{t \ge 0 : G(t) \le \alpha\},$$
finite because $G(t) \to 0$. Right-continuity gives $G(k) \le \alpha$, and the definition of the infimum gives $\mathbb{P}_0(L \ge k) = \lim_{t \uparrow k} G(t) \ge \alpha$ (interpret $\mathbb{P}_0(L \ge 0) = 1 \ge \alpha$ if $k = 0$). If $\mathbb{P}_0(L = k) = 0$ then $G(k) = \alpha$ and we take $\gamma$ arbitrary. Otherwise set
$$\gamma := \frac{\alpha - G(k)}{\mathbb{P}_0(L = k)} \in [0,1],$$
which is well defined and in $[0,1]$ precisely because $G(k) \le \alpha \le G(k) + \mathbb{P}_0(L=k)$. Then $\mathbb{E}_0[\varphi^*] = G(k) + \gamma \mathbb{P}_0(L = k) = \alpha$.

**(b)** Let $\varphi$ be any test with $\mathbb{E}_0[\varphi] \le \alpha$. The key algebraic observation is a sign identity: for every $x$,
$$(\varphi^*(x) - \varphi(x))\,(p_1(x) - k\,p_0(x)) \ge 0. \tag{9.1}$$
Check the three cases. If $p_1(x) > k p_0(x)$ then $L(x) > k$ (this covers $p_0(x)=0<p_1(x)$ too), so $\varphi^*(x) = 1 \ge \varphi(x)$ and both factors are $\ge 0$. If $p_1(x) < kp_0(x)$ then $L(x) < k$, so $\varphi^*(x) = 0 \le \varphi(x)$ and both factors are $\le 0$. If $p_1(x) = kp_0(x)$ the second factor vanishes. Integrating (9.1) against $\mu$:
$$0 \le \int (\varphi^* - \varphi)(p_1 - k p_0)\, d\mu = \underbrace{\big(\mathbb{E}_1[\varphi^*] - \mathbb{E}_1[\varphi]\big)}_{\text{power difference}} - k\underbrace{\big(\mathbb{E}_0[\varphi^*] - \mathbb{E}_0[\varphi]\big)}_{= \alpha - \mathbb{E}_0[\varphi]\, \ge\, 0}.$$
Hence $\mathbb{E}_1[\varphi^*] - \mathbb{E}_1[\varphi] \ge k(\alpha - \mathbb{E}_0[\varphi]) \ge 0$ since $k \ge 0$. That is (b).

**(c)** Suppose $\varphi$ is most powerful at level $\alpha$, so $\mathbb{E}_1[\varphi] = \mathbb{E}_1[\varphi^*]$. From the display above, $0 \ge k(\alpha - \mathbb{E}_0[\varphi])\ \ge 0$ forces $k(\alpha - \mathbb{E}_0[\varphi]) = 0$. If $k > 0$ this gives size exactly $\alpha$. (If $k = 0$ then $\varphi^*$ rejects whenever $p_1 > 0$, so $\mathbb{E}_1[\varphi^*] = 1$, excluded by hypothesis.) Moreover the integral in (9.1) is zero while the integrand is non-negative, so the integrand vanishes $\mu$-a.e.; on $\{L \ne k\}$ the second factor $p_1 - kp_0$ is non-zero, forcing $\varphi = \varphi^*$ there $\mu$-a.e. $\blacksquare$

> **Corollary 9.8 (Neyman–Pearson tests are unbiased).** The most powerful level-$\alpha$ test satisfies $\mathbb{E}_1[\varphi^*] \ge \alpha$, with equality if and only if $P_0 = P_1$.

*Proof.* The constant test $\varphi \equiv \alpha$ has size $\alpha$ and power $\alpha$. By Theorem 9.7(b), $\mathbb{E}_1[\varphi^*] \ge \alpha$. For the equality case: if $\mathbb{E}_1[\varphi^*] = \alpha$ then the constant test $\varphi \equiv \alpha$ is itself most powerful, so by Theorem 9.7(c) it agrees with $\varphi^*$ $\mu$-a.e. off $\{L = k\}$; since $\varphi \equiv \alpha \in (0,1)$ takes neither of $\varphi^*$'s off-boundary values $0$ and $1$, the set $\{L \ne k\}$ must be $\mu$-null. Then $p_1 = k\,p_0$ a.e., and integrating both sides against $\mu$ gives $k = 1$, hence $p_1 = p_0$ a.e. and $P_0 = P_1$. Conversely, if $P_0 = P_1$ every test has power equal to its size. $\blacksquare$

That corollary is a sanity check with teeth: any proposed test whose power falls below its size is worse than ignoring the data and flipping a biased coin, and if you ever compute such a test you have a sign error.

**Worked Example 9.5 (the NP test for a Gaussian mean is the $z$-test).** $X_1,\dots,X_n \overset{\text{iid}}{\sim} N(\mu,\sigma^2)$ with $\sigma^2$ known; $H_0: \mu = \mu_0$ vs $H_1: \mu = \mu_1 > \mu_0$. With respect to Lebesgue measure on $\mathbb{R}^n$,
$$\log L(x) = \frac{1}{\sigma^2}\Big[(\mu_1 - \mu_0)\sum_i x_i - \frac{n}{2}(\mu_1^2 - \mu_0^2)\Big].$$
Since $\mu_1 > \mu_0$, $\log L$ is a strictly increasing affine function of $\sum_i x_i$. So $\{L > k\} = \{\bar x > c\}$ for a corresponding $c$, and the NP test rejects for large $\bar X_n$. Choosing $c$ for size $\alpha$ gives $c = \mu_0 + \sigma z_{1-\alpha}/\sqrt n$: the classical one-sided $z$-test. Because $L$ has a continuous distribution under $P_0$, no randomization is needed ($\mathbb{P}_0(L = k) = 0$).

Two structural lessons. First, the threshold $c$ does not depend on $\mu_1$: the same test is optimal against *every* $\mu_1 > \mu_0$, which is the phenomenon Theorem 9.10 will generalize. Second, all the data enters through $\sum_i x_i$ — the sufficient statistic of the Gaussian family (Ch. 3). Neyman–Pearson tests always depend on the data only through the likelihood ratio, and in an exponential family the likelihood ratio is a function of the sufficient statistic. Sufficiency is not a curiosity; it is the statement that the optimal test can ignore everything else.

**Worked Example 9.6 (a randomized NP test, computed).** Back to three coin flips: $H_0: p = 1/2$ vs $H_1: p = 3/4$, $X \sim \text{Bin}(3,p)$. The likelihood ratio is $L(k) = \binom3k (3/4)^k(1/4)^{3-k} / [\binom3k (1/2)^3] = 3^k/2^3 \cdot ... $ — let us just tabulate it:

| $k$ | $\mathbb{P}_0(X=k)$ | $\mathbb{P}_1(X=k)$ | $L(k)$ |
|---|---|---|---|
| 0 | 0.125000 | 0.015625 | 0.125 |
| 1 | 0.375000 | 0.140625 | 0.375 |
| 2 | 0.375000 | 0.421875 | 1.125 |
| 3 | 0.125000 | 0.421875 | 3.375 |

$L$ is increasing in $k$, so NP tests reject for large $k$. At $\alpha = 0.05$: $\mathbb{P}_0(L > 1.125) = \mathbb{P}_0(X=3) = 0.125 > 0.05$, so $k^* = 3.375$, $\mathbb{P}_0(L > 3.375) = 0$, and $\gamma = (0.05 - 0)/0.125 = 0.4$. The optimal test rejects with probability $0.4$ when all three flips are heads and never otherwise; its power is $0.4 \times 0.421875 = 0.168750$. By Theorem 9.7 no level-$0.05$ test does better — and by Corollary 9.8, $0.16875 > 0.05$, so it does beat the coin flip, barely. If you refuse randomization, your options are power $0$ (size $0$) or power $0.421875$ (size $0.125$). Being explicit about that trade is more honest than pretending $0.125 \approx 0.05$.

<svg viewBox="0 0 620 400" style="max-width:100%;height:auto" role="img" aria-label="Two Gaussian densities with the likelihood-ratio rejection region shaded">
  <line x1="60" y1="280" x2="580" y2="280" stroke="currentColor" stroke-width="1.5"/>
  <polygon points="348,280 348,228.4 361,246.9 373,260.0 386,268.7 399,274.0 412,277.0 424,278.6 437,279.4 450,279.7 463,279.9 475,280 570,280" fill="#ef4444" opacity="0.28"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="60,279.9 73,279.8 86,279.6 98,279.0 111,277.8 124,275.5 136,271.2 149,264.1 162,253.0 175,236.9 188,215.2 200,188.7 213,159.0 226,129.4 238,104.0 251,86.7 264,80.5 277,86.7 290,104.0 302,129.4 315,159.0 328,188.7 340,215.2 353,236.9 366,253.0 379,264.1 392,271.2 404,275.5 417,277.8 430,279.0 442,279.6 455,279.8 468,279.9 481,280.0 494,280.0 506,280.0"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2" points="124,280.0 136,279.9 149,279.8 162,279.6 175,279.0 188,277.8 200,275.5 213,271.2 226,264.1 238,253.0 251,236.9 264,215.2 277,188.7 290,159.0 302,129.4 315,104.0 328,86.7 340,80.5 353,86.7 366,104.0 379,129.4 392,159.0 404,188.7 417,215.2 430,236.9 442,253.0 455,264.1 468,271.2 481,275.5 494,277.8 506,279.0 519,279.6 532,279.8 544,279.9 557,280.0 570,280.0"/>
  <line x1="348" y1="60" x2="348" y2="290" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 3"/>
  <text x="348" y="52" fill="currentColor" font-size="12" text-anchor="middle">L(x) = k</text>
  <text x="255" y="72" fill="#3b82f6" font-size="12" text-anchor="end">p₀ (null)</text>
  <text x="352" y="76" fill="#10b981" font-size="12">p₁ (alternative)</text>
  <text x="430" y="215" fill="#ef4444" font-size="12">size α = ∫ᴿ p₀</text>
  <text x="430" y="150" fill="currentColor" font-size="12">power = ∫ᴿ p₁</text>
  <text x="348" y="296" fill="currentColor" font-size="11" text-anchor="middle">c</text>
  <text x="60" y="330" fill="currentColor" font-size="12">Figure 9.2  The Neyman–Pearson rejection region R = {L &gt; k}. Because L is increasing in x here, R is a half-line</text>
  <text x="60" y="348" fill="currentColor" font-size="12">{x &gt; c}. The red area under p₀ is the size; the area under p₁ over the same region is the power. Sliding c right</text>
  <text x="60" y="366" fill="currentColor" font-size="12">shrinks both. The lemma says: for a fixed red area, no other region — of any shape — captures more p₁ mass,</text>
  <text x="60" y="384" fill="currentColor" font-size="12">because moving any mass out of {L &gt; k} into {L &lt; k} trades expensive alternative-probability for cheap.</text>
</svg>

Figure 9.2 shows the geometry for $P_0 = N(0,1)$, $P_1 = N(1.5,1)$. The rejection region is the set where the alternative density dominates the null density by at least a factor $k$; here that is a half-line $\{x > c\}$ because $L$ is monotone, but in general it is whatever level set of $L$ has the right null mass, and it need not be an interval. (For $H_0 : \mu = 0$ vs $H_1 : \mu = \pm 1$ mixed equally, the region is a union of two half-lines.) The two shaded quantities are the size, $\int_R p_0$, and the power, $\int_R p_1$; the lemma says that among regions with a given $\int_R p_0$, the level set of $L$ maximizes $\int_R p_1$.

### 9.2.2 Composite alternatives: monotone likelihood ratio and UMP tests

Real hypotheses are rarely simple. "My method is better" is $\Theta_1 = \{\theta > \theta_0\}$, not a point. A test that is most powerful against *every* alternative simultaneously is a lot to ask, and usually impossible; the exception is important enough to have a name.

> **Definition 9.9 (uniformly most powerful; monotone likelihood ratio).** A test $\varphi^*$ is *uniformly most powerful* (UMP) at level $\alpha$ for $H_0 : \theta \in \Theta_0$ vs $H_1: \theta \in \Theta_1$ if it has level $\alpha$ and, for every level-$\alpha$ test $\varphi$ and every $\theta \in \Theta_1$, $\beta_{\varphi^*}(\theta) \ge \beta_\varphi(\theta)$.
>
> A family $\{P_\theta : \theta \in \Theta\}$ with $\Theta \subseteq \mathbb{R}$ has *monotone likelihood ratio* (MLR) in the statistic $T : \mathcal{X} \to \mathbb{R}$ if for all $\theta' > \theta$ the ratio $dP_{\theta'}/dP_\theta$ is a non-decreasing function of $T(x)$ (where defined).

> **Theorem 9.10 (Karlin–Rubin).** Suppose $\{P_\theta\}$ has MLR in $T$. For testing $H_0: \theta \le \theta_0$ vs $H_1 : \theta > \theta_0$, the test
> $$\varphi^*(x) = \begin{cases}1 & T(x) > c\\ \gamma & T(x) = c \\ 0 & T(x) < c\end{cases}, \qquad \mathbb{E}_{\theta_0}[\varphi^*] = \alpha,$$
> is UMP at level $\alpha$. Moreover its power function $\beta_{\varphi^*}$ is non-decreasing in $\theta$.

*Proof.* First, monotonicity of the power function. Fix $\theta' > \theta$. By MLR, $dP_{\theta'}/dP_\theta$ is a non-decreasing function $h(T)$ of $T$, so the test $\varphi^*$ above is of exactly the Neyman–Pearson form for testing $P_\theta$ against $P_{\theta'}$ with threshold $k = h(c)$. Let us check the three cases honestly, since $h$ need not be strictly increasing: where $h(T) > h(c)$, monotonicity of $h$ forces $T > c$, so $\varphi^* = 1$, matching the NP prescription; where $h(T) < h(c)$, similarly $T < c$ and $\varphi^* = 0$; and where $h(T) = h(c)$ — a set that may contain points with $T < c$, $T = c$, and $T > c$ if $h$ is flat around $c$ — the test takes values in $\{0, \gamma, 1\} \subseteq [0,1]$, which is permitted, because Theorem 9.7 constrains a most powerful test only *off* the boundary set $\{L = k\}$ and leaves its behaviour on the boundary free apart from the overall size. So $\varphi^*$ is an NP test for every such pair $(\theta,\theta')$ simultaneously. By Corollary 9.8 applied to $P_\theta$ vs $P_{\theta'}$ at level $\alpha' = \mathbb{E}_\theta[\varphi^*]$, we get $\mathbb{E}_{\theta'}[\varphi^*] \ge \mathbb{E}_\theta[\varphi^*]$. Hence $\beta_{\varphi^*}$ is non-decreasing, and in particular $\sup_{\theta \le \theta_0}\beta_{\varphi^*}(\theta) = \beta_{\varphi^*}(\theta_0) = \alpha$, so $\varphi^*$ has level $\alpha$ for the composite null.

Now optimality. Fix any $\theta_1 > \theta_0$ and let $\varphi$ be any level-$\alpha$ test for the composite problem; then in particular $\mathbb{E}_{\theta_0}[\varphi] \le \alpha$. By the previous paragraph $\varphi^*$ is an NP test of $P_{\theta_0}$ against $P_{\theta_1}$ with size exactly $\alpha$, so Theorem 9.7(b) gives $\mathbb{E}_{\theta_1}[\varphi] \le \mathbb{E}_{\theta_1}[\varphi^*]$. Since $\theta_1 \in \Theta_1$ was arbitrary and $\varphi^*$ does not depend on $\theta_1$, $\varphi^*$ is UMP. $\blacksquare$

The clause "$\varphi^*$ does not depend on $\theta_1$" is the whole trick, and it is why MLR is the right hypothesis: the NP test's *form* is determined by $T$ and its threshold by the null alone.

> **Proposition 9.11 (exponential families have MLR).** Let $\{P_\eta\}$ be a one-parameter exponential family (Ch. 3) in natural form: $dP_\eta/d\mu(x) = h(x)\exp\{\eta\, T(x) - A(\eta)\}$ with $\eta$ ranging over an interval. Then the family has MLR in $T$.

*Proof.* For $\eta' > \eta$, on $\{h > 0\}$,
$$\frac{dP_{\eta'}}{dP_\eta}(x) = \exp\{(\eta' - \eta)T(x) - (A(\eta') - A(\eta))\},$$
which is a strictly increasing function of $T(x)$ because $\eta' - \eta > 0$ and $\exp$ is increasing. $\blacksquare$

So the Bernoulli/binomial family (in $T = \sum X_i$), the Poisson family (in $T = \sum X_i$), the Gaussian with known variance (in $T = \sum X_i$), the exponential family of rates, and every other standard one-parameter family admits a UMP one-sided test. This is a large and clean result, and it is worth knowing exactly how much it does *not* give you: for two-sided alternatives $H_1 : \theta \ne \theta_0$ there is generally no UMP test — the best test against $\theta > \theta_0$ is a poor test against $\theta < \theta_0$. The standard fix is to restrict attention to *unbiased* tests (power $\ge \alpha$ everywhere on $\Theta_1$), within which UMP unbiased tests do exist for exponential families. We will not need that theory; we will use the two-sided tests everyone uses and check their level directly.

**Worked Example 9.7 (a UMP test for an error rate).** A deployed model produces errors that you model as a Poisson process: in a fixed monitoring window you observe counts $X_1,\dots,X_{10} \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$, and you want to detect degradation, $H_0: \lambda \le 2$ vs $H_1 : \lambda > 2$. The family is exponential with $T = \sum_i X_i$ and natural parameter $\log \lambda$, so by Proposition 9.11 and Theorem 9.10 the UMP level-$\alpha$ test rejects for large $T$. Under $\lambda = 2$, $T \sim \text{Poisson}(20)$. Exact tail probabilities: $\mathbb{P}(T \ge 28) = 0.052481$, $\mathbb{P}(T \ge 29) = 0.034334$. So $\{T \ge 29\}$ is the level-$0.05$ non-randomized choice, with size $0.0343$. Its power is $\mathbb{P}_{\text{Pois}(25)}(T \ge 29) = 0.236599$ at $\lambda = 2.5$, and $\mathbb{P}_{\text{Pois}(30)}(T \ge 29) = 0.596918$ at $\lambda = 3$.

Note what "UMP" delivers here: this one test is simultaneously optimal against $\lambda = 2.5$, $\lambda = 3$, $\lambda = 17$, and every other alternative. You do not have to guess the degradation size in advance to choose the test — only to know whether the power will be adequate. And note the practical honesty: a 25% chance of catching a 25% degradation in error rate from ten windows is not much of a monitor. That number, not the $p$-value, is what should drive the design.

### 9.2.3 ROC curves: Neyman–Pearson in ML clothing

Machine learning almost never states a threshold. It reports a *score* $s(x)$ and a curve. The connection to §9.2.1 is exact and worth making carefully, because it explains why the curve is the natural object.

> **Definition 9.12 (ROC curve and AUC).** Let $S = s(X)$ be a real-valued score, "positive" meaning $X \sim P_1$ and "negative" meaning $X \sim P_0$. For a threshold $t$ define the false positive rate $\mathrm{FPR}(t) = \mathbb{P}_0(S > t)$ and the true positive rate $\mathrm{TPR}(t) = \mathbb{P}_1(S > t)$. The *ROC curve* of $s$ is the closure of $\{(\mathrm{FPR}(t), \mathrm{TPR}(t)) : t \in \mathbb{R}\}$, completed to a continuous curve from $(0,0)$ to $(1,1)$ by allowing randomization at the threshold. The *AUC* is the area under it.

In testing language, $\mathrm{FPR}$ is the size and $\mathrm{TPR}$ is the power, so the ROC curve of $s$ is precisely the *achievable (size, power) frontier* for tests that threshold $s$.

> **Proposition 9.13 (the likelihood ratio has the best ROC; ROC curves are concave).**
> (i) Let $R^*$ be the ROC curve of the score $s^* = L = dP_1/dP_0$ and $R$ the ROC curve of any other score $s$. Then $R^*$ lies weakly above $R$: for every $\alpha \in [0,1]$, $\sup\{\mathrm{TPR} : (\mathrm{FPR},\mathrm{TPR}) \in R,\ \mathrm{FPR} \le \alpha\} \le \mathrm{TPR}^*(\alpha)$, where $\mathrm{TPR}^*(\alpha)$ is the power of the NP test at level $\alpha$.
> (ii) The function $\alpha \mapsto \mathrm{TPR}^*(\alpha)$ is concave and non-decreasing on $[0,1]$, with $\mathrm{TPR}^*(\alpha)\ge \alpha$.
> (iii) Where differentiable, the slope of $R^*$ at the point achieved by threshold $k$ equals $k$.

*Proof.* (i) A threshold test on $s$ at level $\alpha$ is *a* level-$\alpha$ test; Theorem 9.7(b) says the NP test dominates it.
(ii) Non-decreasing is clear. For concavity, take levels $\alpha_1,\alpha_2$ with optimal tests $\varphi_1^*,\varphi_2^*$, and $\lambda \in [0,1]$. The randomized test $\varphi = \lambda\varphi_1^* + (1-\lambda)\varphi_2^*$ is a legitimate test (values in $[0,1]$) with size $\lambda\alpha_1 + (1-\lambda)\alpha_2$ and power $\lambda \mathrm{TPR}^*(\alpha_1) + (1-\lambda)\mathrm{TPR}^*(\alpha_2)$. Since $\mathrm{TPR}^*$ is the *maximum* power at its size, $\mathrm{TPR}^*(\lambda\alpha_1 + (1-\lambda)\alpha_2) \ge \lambda \mathrm{TPR}^*(\alpha_1) + (1-\lambda)\mathrm{TPR}^*(\alpha_2)$. The final inequality is Corollary 9.8.
(iii) Parametrize by $k$: $\alpha(k) = \mathbb{P}_0(L > k)$, $\mathrm{TPR}(k) = \mathbb{P}_1(L>k)$. Writing $\mathbb{P}_1(L > k) = \mathbb{E}_0[L \mathbf{1}\{L>k\}]$ (valid when $P_1 \ll P_0$; in general add the singular part, which contributes to $\mathrm{TPR}$ at $k=\infty$), and differentiating with respect to $k$ against the distribution $F$ of $L$ under $P_0$ — here is where "where differentiable" earns its place in the statement: we assume $F$ has a density at $k$, so that $d\alpha = -dF(k)$ and $d\mathrm{TPR} = -k\, dF(k)$ make sense as derivatives; at atoms of $F$ the ROC has a linear segment (a randomization segment) whose constant slope is the atom's value $k$, so the conclusion survives in the form "slope of the segment $= k$". Then $d\mathrm{TPR}/d\alpha = k$. $\blacksquare$

This proposition is the reason ROC analysis is not an arbitrary convention. Part (i) says: *any* detector is dominated by the likelihood-ratio detector, so the entire enterprise of "learning a better score" is the enterprise of estimating a likelihood ratio (up to monotone transformation — see below). Part (ii) says an ROC curve that dips below the diagonal or is non-concave is *suboptimally thresholded*, and you can strictly improve it by randomizing between two operating points; empirical ROC curves from finite data often show local non-concavity, which is noise, and the concave hull is the honest summary. Part (iii) says the slope of the ROC at an operating point *is* the likelihood-ratio threshold there — so the shape of the curve near $\mathrm{FPR}=0$ tells you how extreme the evidence must be to buy a low false-alarm rate.

There is a modelling consequence that gets stated loosely in papers and is worth stating precisely. Since the ROC depends on the score only through the *ordering* it induces, any strictly increasing transformation of $L$ has the same ROC. So a classifier's ROC is optimal iff its score is a monotone function of the likelihood ratio. A calibrated posterior $\mathbb{P}(Y=1\mid x)$ qualifies: by Proposition 1.12, $\frac{\mathbb{P}(Y=1|x)}{\mathbb{P}(Y=0|x)} = \frac{\pi_1}{\pi_0}L(x)$, a strictly increasing function of $L$ for fixed priors. Hence *a correctly calibrated probability is an optimal score*, and this is the precise sense in which "just output good probabilities" is the right advice for detection. But note the corollary that is often forgotten: calibration is *not necessary* for a good ROC — a monotone recalibration changes calibration completely and leaves AUC exactly unchanged. Choose your metric to match your claim: AUC audits ranking, calibration curves audit probabilities, and a model can be excellent at one and terrible at the other.

**Worked Example 9.8 (ROC and AUC for Gaussian scores, computed).** Let scores be $N(0,1)$ for negatives and $N(\Delta,1)$ for positives. Then $\mathrm{FPR}(t) = 1-\Phi(t)$, $\mathrm{TPR}(t) = 1-\Phi(t-\Delta)$, and (by Proposition 9.47, proved in §9.9) $\mathrm{AUC} = \mathbb{P}(S_+ > S_-) = \Phi(\Delta/\sqrt 2)$ because $S_+ - S_- \sim N(\Delta, 2)$. Numerically: $\Delta = 0.5 \Rightarrow \mathrm{AUC} = 0.638163$; $\Delta = 1 \Rightarrow 0.760250$; $\Delta = 2 \Rightarrow 0.921350$. For $\Delta = 1$ the operating points and the likelihood-ratio slope $L = \exp(\Delta t - \Delta^2/2)$ are:

| threshold $t$ | FPR | TPR | slope $L(t)$ |
|---|---|---|---|
| $-1$ | 0.8413 | 0.9772 | 0.223 |
| $0$ | 0.5000 | 0.8413 | 0.607 |
| $0.5$ | 0.3085 | 0.6915 | 1.000 |
| $1$ | 0.1587 | 0.5000 | 1.649 |
| $1.5$ | 0.0668 | 0.3085 | 2.718 |
| $2$ | 0.0228 | 0.1587 | 4.482 |

Read the slope column against Proposition 9.13(iii): near $\mathrm{FPR}=0.02$ the ROC has slope $4.5$, meaning each additional unit of false-positive rate buys 4.5 units of true-positive rate there, while at $\mathrm{FPR}=0.84$ the slope is $0.22$ and the exchange rate has inverted. If your deployment cares about the low-FPR corner (fraud, safety filters, medical triage), an AUC that averages over the whole curve is the wrong summary — it is dominated by the region you will never operate in. Report partial AUC or TPR at a fixed low FPR instead, and note that this choice must be made *before* seeing results, or it becomes the selection problem of §9.6.

> **Common Confusion 9.2.** *"AUC = 0.76 means the model is right 76% of the time."*
>
> No. Proposition 9.47 says $\mathrm{AUC} = \mathbb{P}(S_+ > S_-)$ for an independently drawn positive and negative: it is the probability that a random positive outranks a random negative. It involves no threshold, so it cannot be an accuracy. In the table above, the $\Delta=1$ model with AUC 0.760 has accuracy $0.5(\mathrm{TPR} + 1 - \mathrm{FPR})$ at balanced classes, which at $t = 0.5$ is $0.5(0.6915 + 0.6915) = 0.6915$ — and at $t = 2$ it is $0.568$. Different numbers, different questions. Relatedly, AUC is invariant to class prevalence while accuracy and precision are not; on a 1-in-1000 positive rate, an AUC of 0.99 is compatible with a precision of a few percent, exactly the arithmetic of Chapter 1's disease test.

> **Self-check 9.2.**
> 1. In Theorem 9.7, where is countable additivity or measurability actually used? Point at a specific step.
> 2. Why does the Neyman–Pearson test for $N(0,1)$ vs $N(\mu,1)$ not depend on $\mu$ (given the sign of $\mu$), and what property of the exponential family is responsible?
> 3. Give an example of two hypotheses for which the most powerful rejection region is *not* an interval.
> 4. A colleague's empirical ROC curve has a dip below the diagonal near FPR $= 0.9$. What does Proposition 9.13 say you can do about it, and what is the most likely explanation?

## 9.3 The three classical tests: likelihood ratio, Wald, and score

### 9.3.1 Three ways to measure the same distance

Neyman–Pearson gives an exact optimal test when both hypotheses are simple. In practice $\theta$ is a vector, the null constrains some coordinates, nuisance parameters are unknown, and exact optimality is unavailable. What survives is an asymptotic story, and it comes in three flavours that differ only in *where* you do the work.

Picture the log-likelihood surface $\ell(\theta) = \sum_i \log p_\theta(X_i)$ over the parameter space, with the MLE $\hat\theta$ at its peak and the null hypothesis carving out a lower-dimensional set $\Theta_0$. Let $\tilde\theta$ be the constrained maximizer, the best point *inside* $\Theta_0$. There are three natural ways to ask "is the null far from the peak?":

1. **Compare heights.** How much log-likelihood do we sacrifice by staying inside $\Theta_0$? That is the *likelihood ratio* test. It requires fitting both models.
2. **Compare locations.** How far is $\hat\theta$ from $\Theta_0$, measured in units of the estimator's standard error? That is the *Wald* test. It requires fitting only the full model.
3. **Compare slopes.** How steeply is the likelihood still climbing at $\tilde\theta$? If the null were true, the gradient there should be near zero. That is the *score* (Rao, Lagrange multiplier) test. It requires fitting only the restricted model.

The picture also tells you the punchline: for a quadratic log-likelihood the three questions have literally the same answer, and asymptotic normality of the MLE (Ch. 8) says the log-likelihood is approximately quadratic near the peak at scale $n^{-1/2}$. So the three tests agree to first order and can disagree substantially when the likelihood is skewed — which is exactly what happens for probabilities near 0 or 1, i.e. exactly the ML case of accuracy near 1.

Set up notation. $\theta \in \Theta \subseteq \mathbb{R}^d$ open, i.i.d. data, and a null that fixes $k$ smooth constraints: write $\theta = (\psi, \lambda)$ with $\psi \in \mathbb{R}^k$ the parameters of interest and $\lambda \in \mathbb{R}^{d-k}$ nuisance, and test $H_0 : \psi = \psi_0$. Let $\hat\theta = (\hat\psi,\hat\lambda)$ be the unrestricted MLE and $\tilde\theta = (\psi_0, \tilde\lambda)$ the restricted MLE. Let $I(\theta)$ be the Fisher information for one observation, $U(\theta) = \nabla_\theta \ell(\theta)$ the score vector, and $I_{\psi\psi}^{-1}$ the upper-left $k\times k$ block of $I(\theta)^{-1}$ (the *inverse* information's block, not the inverse of the block — the difference is exactly the cost of not knowing $\lambda$).

> **Definition 9.14 (the three statistics).** With $\Lambda := \dfrac{\sup_{\theta \in \Theta_0} \prod_i p_\theta(X_i)}{\sup_{\theta\in\Theta}\prod_i p_\theta(X_i)} = \dfrac{L(\tilde\theta)}{L(\hat\theta)} \in (0,1]$, define
> $$W_{\mathrm{LR}} := -2\log\Lambda = 2\big(\ell(\hat\theta) - \ell(\tilde\theta)\big),$$
> $$W_{\mathrm{Wald}} := n\,(\hat\psi - \psi_0)^\top \big[\widehat{I^{-1}_{\psi\psi}}\big]^{-1} (\hat\psi - \psi_0), \qquad \widehat{\cdot} \text{ evaluated at } \hat\theta,$$
> $$W_{\mathrm{score}} := \tfrac1n\, U(\tilde\theta)^\top \tilde I^{-1} U(\tilde\theta), \qquad \tilde I := I(\tilde\theta).$$

> **Theorem 9.15 (Wilks).** Assume the regularity conditions under which Chapter 8's MLE asymptotics hold: the model is correctly specified with true parameter $\theta^* = (\psi_0,\lambda^*)$ in the interior of $\Theta$, $\ell$ is three times continuously differentiable in a neighbourhood with suitably dominated derivatives, $I(\theta^*)$ is finite and positive definite, and $\hat\theta \to_p \theta^*$, $\tilde\theta\to_p\theta^*$. Then under $H_0$,
> $$W_{\mathrm{LR}} = -2\log\Lambda \;\longrightarrow_d\; \chi^2_k ,$$
> where $k$ is the number of restrictions (the difference in dimension between the two models).

*Proof sketch (the honest version, with the analytic steps named).* Write $s_n(\theta) = n^{-1/2}U(\theta)$ and expand $\ell$ around $\hat\theta$ by Taylor's theorem with the integral remainder:
$$\ell(\tilde\theta) = \ell(\hat\theta) + \underbrace{U(\hat\theta)^\top(\tilde\theta - \hat\theta)}_{=0} - \tfrac12 (\tilde\theta-\hat\theta)^\top H_n (\tilde\theta - \hat\theta) + R_n,$$
where $-H_n$ is the Hessian of $\ell$ at an intermediate point and $U(\hat\theta) = 0$ because $\hat\theta$ is an interior maximum. Chapter 8's conditions give $\tfrac1n H_n \to_p I(\theta^*)$ (a law of large numbers for the second derivatives, plus continuity, plus $\hat\theta\to_p\theta^*$), and the third-derivative domination gives $R_n = o_p(1)$ once $\|\hat\theta - \tilde\theta\| = O_p(n^{-1/2})$. Hence
$$W_{\mathrm{LR}} = n(\hat\theta - \tilde\theta)^\top I(\theta^*) (\hat\theta-\tilde\theta) + o_p(1). \tag{9.2}$$
Now the algebra. Chapter 8 gives $\sqrt n(\hat\theta - \theta^*) = I^{-1} s_n(\theta^*) + o_p(1) \to_d N(0, I^{-1})$. The same expansion applied to the restricted problem (maximize over $\lambda$ with $\psi = \psi_0$ fixed) gives $\sqrt n (\tilde\lambda - \lambda^*) = I_{\lambda\lambda}^{-1}s_{n,\lambda} + o_p(1)$, where subscripts denote blocks of $I$ and of $s_n$. Substituting into (9.2) and simplifying the block algebra yields
$$W_{\mathrm{LR}} = n(\hat\psi - \psi_0)^\top \big[I^{-1}_{\psi\psi}\big]^{-1}(\hat\psi-\psi_0) + o_p(1),$$
where $I^{-1}_{\psi\psi} = (I_{\psi\psi} - I_{\psi\lambda}I_{\lambda\lambda}^{-1}I_{\lambda\psi})^{-1}$ is the $\psi$-block of $I^{-1}$ (Schur complement). By Chapter 8, $\sqrt n(\hat\psi - \psi_0) \to_d N(0, I^{-1}_{\psi\psi})$; a quadratic form $Z^\top \Sigma^{-1} Z$ with $Z \sim N(0,\Sigma)$, $\Sigma \in \mathbb{R}^{k\times k}$ nonsingular, is $\chi^2_k$ (write $Z = \Sigma^{1/2}G$ with $G\sim N(0,I_k)$, so the form is $\|G\|^2$). Slutsky's theorem (Ch. 7) lets us drop the $o_p(1)$. $\blacksquare$

I called that a sketch and I mean it: the steps I named as "Chapter 8's conditions" are the technical heart, and they are where the theorem fails when it fails. It is worth knowing the failure catalogue, because ML models routinely violate these hypotheses:

- **Boundary nulls.** If $\theta^*$ is on the boundary of $\Theta$ — testing whether a variance component is zero, whether a mixture weight is zero, whether a regularization-induced coefficient is exactly zero — then $\hat\theta$ is not asymptotically normal and the limit is a mixture of $\chi^2$'s (a "chi-bar-squared"). Testing $H_0: \sigma^2_b = 0$ in a random-effects model with the naive $\chi^2_1$ is anti-conservative in the standard direction; the correct limit is $\tfrac12\chi^2_0 + \tfrac12\chi^2_1$.
- **Non-identifiability.** Mixture models with an unknown number of components violate identifiability under the null, and $-2\log\Lambda$ can diverge. This is the reason "likelihood ratio test for the number of clusters" is not a thing you may do casually.
- **Misspecification.** If the model is wrong, $\hat\theta$ still converges (to the KL projection) and is still asymptotically normal, but with the *sandwich* variance of Chapter 8, not $I^{-1}$. Then $W_{\mathrm{LR}}$ has a limiting distribution that is a weighted sum of $\chi^2_1$'s, not $\chi^2_k$. Since every neural network is misspecified, treat Wilks-based $p$-values from large models as heuristics, and prefer the permutation and bootstrap machinery of §9.7–9.8.
- **Growing dimension.** $d$ fixed as $n\to\infty$ is assumed throughout. For $d$ comparable to $n$ the $\chi^2_k$ calibration fails.

> **Proposition 9.16 (asymptotic equivalence).** Under the hypotheses of Theorem 9.15,
> $$W_{\mathrm{LR}} - W_{\mathrm{Wald}} \to_p 0, \qquad W_{\mathrm{LR}} - W_{\mathrm{score}} \to_p 0,$$
> so all three converge in distribution to $\chi^2_k$ under $H_0$, and the three tests have the same asymptotic power against local alternatives $\psi = \psi_0 + h/\sqrt n$ (namely, a noncentral $\chi^2_k$ with noncentrality $h^\top[I^{-1}_{\psi\psi}]^{-1}h$).

*Proof sketch.* The final display in the proof of Theorem 9.15 already says $W_{\mathrm{LR}} = W_{\mathrm{Wald}} + o_p(1)$ once we replace $I(\theta^*)$ by the consistent estimator $I(\hat\theta)$ (continuity of $I$ plus $\hat\theta\to_p\theta^*$, then Slutsky). For the score statistic, expand $U(\tilde\theta)$ around $\hat\theta$: $U(\tilde\theta) = U(\hat\theta) - H_n(\tilde\theta - \hat\theta) + o_p(\sqrt n) = -H_n(\tilde\theta-\hat\theta) + o_p(\sqrt n)$, so $\tfrac1n U(\tilde\theta)^\top \tilde I^{-1}U(\tilde\theta) = n(\hat\theta-\tilde\theta)^\top \tfrac{H_n}{n} \tilde I^{-1}\tfrac{H_n}{n}(\hat\theta-\tilde\theta) + o_p(1)$, and $H_n/n \to_p I$, $\tilde I \to_p I$ give the same quadratic form as (9.2). The local-power claim follows from the same expansion under a contiguous sequence of alternatives, using the Le Cam theory that Chapter 8 developed for local asymptotic normality. $\blacksquare$

### 9.3.2 What differs at finite $n$

Asymptotic equivalence is a statement about the limit and says nothing about $n = 20$. Here is what actually distinguishes them.

**Computational cost.** The score test needs only the restricted fit; the Wald test needs only the unrestricted fit; the LR test needs both. In ML this is often decisive: comparing a 7B-parameter model to a variant is two training runs for LR, one for Wald.

**Invariance.** $W_{\mathrm{LR}}$ is invariant under reparametrization, because the likelihood *value* at the two maxima does not care how you coordinatize the parameter space. $W_{\mathrm{Wald}}$ is not, because it measures a distance in a particular parametrization with a particular variance estimate. This is not a subtlety; it is a large practical effect.

**Worked Example 9.9 (Wald is not invariant, and it matters).** $X \sim \text{Bin}(20, p)$, observed $x = 15$, so $\hat p = 0.75$. Test $H_0 : p = 0.5$, which is the same hypothesis as $H_0: \mathrm{logit}(p) = 0$.

On the $p$ scale: $\widehat{\mathrm{Var}}(\hat p) = \hat p(1-\hat p)/n = 0.009375$, so $W_{\mathrm{Wald}} = (0.75-0.5)^2/0.009375 = 6.6667$, $p$-value $= \mathbb{P}(\chi^2_1 > 6.6667) = 0.00982$.

On the logit scale: $\hat\eta = \log(0.75/0.25) = 1.098612$, and by the delta method (Ch. 7) $\widehat{\mathrm{Var}}(\hat\eta) = \frac{1}{n\hat p(1-\hat p)} = 0.266667$, so $W_{\mathrm{Wald}} = 1.098612^2/0.266667 = 4.5261$, $p$-value $= 0.03338$.

Same data, same hypothesis, two Wald $p$-values differing by a factor of 3.4, straddling any threshold you care to name. For comparison, the invariant alternatives: $W_{\mathrm{LR}} = 2[15\log(0.75/0.5) + 5\log(0.25/0.5)] = 5.2325$ ($p = 0.02217$); $W_{\mathrm{score}} = (0.75-0.5)^2/(0.25/20) = 5.0000$ ($p = 0.02535$); and the exact binomial two-sided test gives $p = 0.04139$.

The exact test is the reference, and note that all three asymptotic tests are anti-conservative here, the $p$-scale Wald grossly so. This is the general pattern: **Wald tests and Wald intervals on a bounded parameter, evaluated near the boundary, are the least trustworthy of the three.** Since "accuracy", "precision", "recall", "AUC", and "win rate" are all bounded in $[0,1]$ and are all usually near a boundary in modern papers, this is not an academic concern. §9.5 quantifies it with exact coverage computations.

**Worked Example 9.10 (a nested-model comparison done properly).** This is the ablation-study calculation. Simulate $n = 200$ observations from $y_i = \beta^\top x_i + \varepsilon_i$, $\varepsilon_i\sim N(0,1)$, with $x_i = (1, x_{i1},x_{i2},x_{i3},x_{i4})$, $x_{ij}\sim N(0,1)$ independent, and true $\beta = (1, 0.8, -0.5, 0, 0.15)$ — so the fourth feature is useless and the fifth is weakly useful. (Data generated with NumPy's PCG64, seed 20260825.) Test $H_0 : \beta_3 = \beta_4 = 0$, i.e. $k=2$ restrictions, treating $\sigma^2$ as an additional free parameter estimated by maximum likelihood in each model.

The fits: unrestricted $\hat\beta = (0.9511, 0.8131, -0.4552, -0.0721, 0.0773)$, $\hat\sigma^2 = 1.008092$, $\ell(\hat\theta) = -284.5937$. Restricted $\tilde\beta = (0.9506, 0.8108, -0.4724)$, $\tilde\sigma^2 = 1.020078$, $\ell(\tilde\theta) = -285.7756$. Therefore
$$W_{\mathrm{LR}} = 2(-284.5937 + 285.7756) = 2.363895, \qquad p = \mathbb{P}(\chi^2_2 > 2.3639) = 0.306681 .$$
The other two: $W_{\mathrm{Wald}} = 2.377920$ ($p = 0.304538$), $W_{\mathrm{score}} = 2.349980$ ($p = 0.308822$). The exact finite-sample $F$ test for the same hypothesis gives $F_{2,195} = 1.159236$, $p = 0.315878$. All four agree: the two features are not jointly detectable at $n=200$, which is correct — one of them genuinely has coefficient zero and the other has $\beta_4 = 0.15$, an effect size of $0.15$ in $\sigma$ units, needing $n \approx 7.85/0.15^2 \approx 350$ per Worked Example 9.4 for 80% power on its own.

Two points about practice. First, the ordering $W_{\mathrm{Wald}} > W_{\mathrm{LR}} > W_{\mathrm{score}}$ seen here ($2.378 > 2.364 > 2.350$) is typical though not universal; when a decision flips between them, the honest report is that the evidence is borderline, not that one test "found" significance. Second, this whole calculation is an *ablation*: "does removing features 3 and 4 hurt?" The right answer to that question, when you have the compute, is not a $\chi^2$ $p$-value from one training run — it is the paired comparison of §9.9 across seeds and test items, because the assumptions behind Theorem 9.15 (correct specification, i.i.d., interior parameter, fixed dimension) are all false for a neural network, while exchangeability of test items is often true by construction.

**Worked Example 9.11 (is the $\chi^2_2$ calibration real? check it).** With the same design matrix as Worked Example 9.10 but the null true ($\beta_3 = \beta_4 = 0$), I generated $B = 20{,}000$ independent response vectors (seed 20260825, continuing the same stream) and recomputed $-2\log\Lambda$ each time. The results: sample mean $2.0474$ against the $\chi^2_2$ mean $2$; sample variance $4.2085$ against $4$; empirical $\mathbb{P}(-2\log\Lambda > 5.9915) = 0.0536$ against nominal $0.05$; empirical $\mathbb{P}(>\chi^2_2(0.99)) = 0.0111$ against $0.01$; Kolmogorov–Smirnov distance to $\chi^2_2$ of $0.0100$. So at $n=200$ with a correctly specified Gaussian linear model, Wilks is accurate to about half a percentage point of size at the 5% level — slightly anti-conservative, as is typical, and the residual is exactly the $O(n^{-1})$ Bartlett correction term that the asymptotics discard. Do this check whenever you can afford it: simulating your own null is the cheapest form of statistical honesty available, and it requires no theory beyond knowing how to generate data under $H_0$.

> **Common Confusion 9.3.** *"The LR test compares two models, so a small $p$-value means the bigger model is better."*
>
> It means the bigger model fits the *training* data better than chance-level overfitting would explain, under the assumption that the small model is correct and both are correctly specified. Three gaps. (i) $-2\log\Lambda$ is computed on the data used to fit; it says nothing directly about generalization, and its null distribution $\chi^2_k$ is precisely the "expected free improvement from $k$ extra parameters" — which is why AIC subtracts $2k$ and why Chapter 10's marginal likelihood does something more principled. (ii) Rejecting $H_0$ is evidence against the *conjunction* of the null restriction and the modelling assumptions; with a misspecified likelihood, $-2\log\Lambda$ can be large because the shared assumptions are wrong. (iii) Statistical detectability is not practical relevance: with $n = 10^7$, $-2\log\Lambda$ will reject a coefficient of $10^{-4}$. If the question is "should I ship the bigger model", the object to estimate is held-out predictive performance with an interval on it, not a $p$-value on a training-set likelihood.

> **Self-check 9.3.**
> 1. Which of the three statistics requires fitting both models? Which requires the restricted fit only?
> 2. Why is $W_{\mathrm{LR}}$ invariant under smooth reparametrization while $W_{\mathrm{Wald}}$ is not?
> 3. In Theorem 9.15, what exactly is $k$ if the full model has 12 parameters and the null fixes 3 of them to specific values and forces two others to be equal?
> 4. Name two properties of a deep network that violate the hypotheses of Wilks' theorem.

## 9.4 $p$-values: what they are, what they are not

### 9.4.1 Definition and the uniformity theorem

A test with a fixed $\alpha$ throws away information: it reports "reject at 5%" but not whether the evidence was overwhelming or marginal. The $p$-value is the natural summary that records the whole family of decisions at once.

> **Definition 9.17 ($p$-value).** Let $\{\varphi_\alpha : \alpha \in (0,1)\}$ be a family of tests, nested in the sense that $\varphi_\alpha \le \varphi_{\alpha'}$ pointwise whenever $\alpha \le \alpha'$, with $\varphi_\alpha$ of level $\alpha$. The associated *$p$-value* is
> $$p(x) := \inf\{\alpha \in (0,1) : \varphi_\alpha(x) = 1\},$$
> the smallest level at which the observed data leads to rejection. Concretely, for a test statistic $T$ with large values evidence against $H_0$, and null distribution function $F_0$ of $T$,
> $$p(x) = \sup_{\theta\in\Theta_0}\mathbb{P}_\theta\big(T(X) \ge T(x)\big),$$
> the null probability of a test statistic at least as extreme as the one observed.

The phrase "at least as extreme" carries all the weight, and it is a choice, not a fact: it is determined by the alternative you had in mind, because "extreme" means "in the direction the alternative predicts". A two-sided $p$-value and a one-sided $p$-value from the same data answer different questions. Choosing the direction after seeing the data is a form of $p$-hacking that halves your $p$-value for free.

A practical warning before you compare your hand computation with software: for a *discrete, asymmetric* null there is more than one legitimate two-sided convention, and they disagree. The *doubling* convention takes $2\min(\mathbb{P}_0(T \le t), \mathbb{P}_0(T \ge t))$, capped at 1; the *minimum-likelihood* convention sums $\mathbb{P}_0(T = k)$ over all outcomes $k$ whose null probability is at most that of the observed $t$. For a symmetric null (like $\mathrm{Bin}(n, 1/2)$) they coincide; for an asymmetric one they do not. Testing $H_0 : p = 0.3$ with $X \sim \mathrm{Bin}(10, 0.3)$ and observing $x = 6$: the doubling convention gives $2 \times \mathbb{P}_0(X \ge 6) = 2 \times 0.04735 = 0.09470$, while the minimum-likelihood convention gives $0.07560$ — which is what `scipy.stats.binomtest` returns by default. Both are valid (super-uniform) $p$-values; neither is "the" $p$-value. The lesson is the same as for one- versus two-sided: the convention is part of the pre-specified analysis, and switching to whichever is smaller after seeing the data is $p$-hacking in miniature.

> **Proposition 9.18 (super-uniformity, and exact uniformity in the continuous case).** Let $T$ be a real-valued statistic, $H_0: P = P_0$ simple, and $p(x) = \mathbb{P}_0(T \ge T(x))$.
> (i) For every $\alpha\in[0,1]$, $\mathbb{P}_0(p(X) \le \alpha) \le \alpha$. ("$p$ is super-uniform" or "stochastically at least uniform".)
> (ii) If $T$ has a continuous distribution under $P_0$ then $p(X) \sim \mathrm{Unif}[0,1]$ exactly.

*Proof.* Let $G(t) := \mathbb{P}_0(T \ge t)$, so $p(X) = G(T)$. $G$ is non-increasing and left-continuous.

(i) Fix $\alpha$ and let $A_\alpha := \{t\in\mathbb{R} : G(t) \le \alpha\}$, so that $\{p \le \alpha\} = \{T \in A_\alpha\}$ exactly. Because $G$ is non-increasing, $A_\alpha$ is an "up-set": if $t \in A_\alpha$ and $t' > t$ then $G(t')\le G(t)\le\alpha$, so $t'\in A_\alpha$. Hence, with $t_\alpha := \inf A_\alpha$, either $A_\alpha = [t_\alpha,\infty)$ or $A_\alpha = (t_\alpha,\infty)$ (or $A_\alpha$ is empty, in which case the probability is $0$). In the first case
$$\mathbb{P}_0(T\in A_\alpha) = \mathbb{P}_0(T \ge t_\alpha) = G(t_\alpha) \le \alpha$$
because $t_\alpha\in A_\alpha$. In the second case, using continuity from above of $P_0$ (Proposition 1.8(4)) along $s_j \downarrow t_\alpha$,
$$\mathbb{P}_0(T \in A_\alpha) = \mathbb{P}_0(T > t_\alpha) = \lim_j \mathbb{P}_0(T \ge s_j) = \lim_j G(s_j) \le \alpha,$$
since every $s_j > t_\alpha$ lies in $A_\alpha$. Either way $\mathbb{P}_0(p\le\alpha)\le\alpha$.

(ii) If the law of $T$ is continuous then $F_0(t) := \mathbb{P}_0(T \le t)$ is continuous, $G(t) = 1 - F_0(t^-) = 1-F_0(t)$, and $F_0(T) \sim \mathrm{Unif}[0,1]$ by the probability-integral transform: for $u \in (0,1)$, $\mathbb{P}_0(F_0(T) \le u) = \mathbb{P}_0(T \le F_0^{-1}(u)) = u$, using continuity and the generalized inverse. Hence $p = 1 - F_0(T) \sim \mathrm{Unif}[0,1]$. $\blacksquare$

(The parenthetical fussing in (i) exists because with atoms the inequality can be strict, and it *must* be able to be strict — Worked Example 9.12 shows a case where $\mathbb{P}_0(p\le 0.05) = 0.0215$.)

> **Proposition 9.19 ($p$-values and tests are the same object).** Let $p$ satisfy $\mathbb{P}_\theta(p \le \alpha)\le\alpha$ for all $\theta\in\Theta_0,\alpha\in[0,1]$. Then $\varphi_\alpha := \mathbf{1}\{p \le \alpha\}$ is a level-$\alpha$ test, and these tests are nested. Conversely, given a nested family of level-$\alpha$ tests, Definition 9.17's $p$ is super-uniform under every $\theta \in \Theta_0$.

*Proof.* The forward direction is immediate: $\mathbb{E}_\theta[\varphi_\alpha] = \mathbb{P}_\theta(p\le\alpha)\le\alpha$, and $\alpha\mapsto\mathbf{1}\{p\le\alpha\}$ is non-decreasing. Conversely, if $p(x) \le \alpha$ then by definition of the infimum there is $\alpha' \le \alpha$ with $\varphi_{\alpha'}(x)=1$, so by nestedness $\varphi_{\alpha}(x)=1$; thus $\{p\le\alpha\}\subseteq\{\varphi_\alpha = 1\}$ and $\mathbb{P}_\theta(p\le\alpha)\le\mathbb{E}_\theta[\varphi_\alpha]\le\alpha$. $\blacksquare$

So there is nothing extra in a $p$-value: it is the test family, encoded as a number. The reason to prefer the number is that readers with different $\alpha$'s can use it, and the reason to distrust it is that readers usually do not have an $\alpha$ at all and treat the number as a measure of truth. Which it is not, as we are about to compute.

**Worked Example 9.12 ($p$-value distributions, simulated and enumerated).** Continuous case: $X_1,\dots,X_{30}\overset{\text{iid}}{\sim}N(\mu,1)$, two-sided test of $H_0: \mu=0$ with $p = 2(1-\Phi(|\sqrt{30}\bar X|))$. With the null true and $B = 200{,}000$ replications (NumPy PCG64, seed 9091): sample mean of $p$ is $0.49940$ (target $0.5$), sample variance $0.08351$ (target $1/12 = 0.08333$), decile counts
$$20129,\ 20021,\ 20104,\ 19950,\ 20097,\ 19896,\ 19906,\ 19791,\ 20080,\ 20026$$
against an expectation of $20{,}000$ per bin, and a Kolmogorov–Smirnov distance to $\mathrm{Unif}[0,1]$ of $0.00197$ ($p = 0.42$ for the KS test itself). Empirically $\mathbb{P}(p\le 0.01) = 0.00980$, $\mathbb{P}(p \le 0.05) = 0.05056$, $\mathbb{P}(p\le0.10) = 0.10064$. Uniform, as promised.

Same setup with $\mu = 0.5$ true: decile counts become
$$172670,\ 12767,\ 5655,\ 3051,\ 2010,\ 1282,\ 902,\ 646,\ 557,\ 460 ,$$
i.e. 86.3% of the mass in the first decile. Power at $\alpha = 0.05$ is $0.78131$, matching the theoretical $0.78191$; the median $p$-value is $0.00614$ and 29.0% of replications give $p \le 0.001$. This is the shape you should have in your head: **under the null, flat; under an alternative, a spike at zero with a decaying tail.** A histogram of many $p$-values from a real pipeline that looks *neither* flat nor spiked-at-zero — say, humped in the middle, or with a suspicious pile-up just below 0.05 — is diagnostic of something wrong: a mis-specified null, dependence, or selective reporting.

Discrete case: $X\sim\text{Bin}(10,1/2)$, two-sided exact test with $p = \min\{1, 2\min(F_0(x), 1-F_0(x^-))\}$. The *only attainable* $p$-values are
$$0.00195,\ 0.02148,\ 0.10938,\ 0.34375,\ 0.75391,\ 1.00000 .$$
Consequently $\mathbb{P}_0(p \le 0.05) = 0.02148$ and $\mathbb{P}_0(p\le 0.10) = 0.02148$ as well: the test at nominal level $0.10$ has actual size $0.0215$, wasting more than three-quarters of its error budget. This is Worked Example 9.2 in $p$-value clothing, and it is why exact tests on small discrete data are conservative — which is safe but costly, and is the argument for the randomized or mid-$p$ variants when power is scarce.

<svg viewBox="0 0 620 400" style="max-width:100%;height:auto" role="img" aria-label="Histograms of p-values under the null and under an alternative">
  <line x1="60" y1="250" x2="290" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <line x1="60" y1="250" x2="60" y2="70" stroke="currentColor" stroke-width="1.5"/>
  <rect x="62" y="136.0" width="19" height="114.0" fill="#3b82f6" opacity="0.75"/>
  <rect x="84" y="136.6" width="19" height="113.4" fill="#3b82f6" opacity="0.75"/>
  <rect x="106" y="136.1" width="19" height="113.9" fill="#3b82f6" opacity="0.75"/>
  <rect x="128" y="136.9" width="19" height="113.1" fill="#3b82f6" opacity="0.75"/>
  <rect x="150" y="136.1" width="19" height="113.9" fill="#3b82f6" opacity="0.75"/>
  <rect x="172" y="137.3" width="19" height="112.7" fill="#3b82f6" opacity="0.75"/>
  <rect x="194" y="137.2" width="19" height="112.8" fill="#3b82f6" opacity="0.75"/>
  <rect x="216" y="137.8" width="19" height="112.2" fill="#3b82f6" opacity="0.75"/>
  <rect x="238" y="136.2" width="19" height="113.8" fill="#3b82f6" opacity="0.75"/>
  <rect x="260" y="136.5" width="19" height="113.5" fill="#3b82f6" opacity="0.75"/>
  <line x1="60" y1="136.6" x2="290" y2="136.6" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5 3"/>
  <text x="292" y="140" fill="#ef4444" font-size="10">10%</text>
  <text x="175" y="62" fill="currentColor" font-size="12" text-anchor="middle">H₀ true: p ~ Unif[0,1]</text>
  <text x="60" y="266" fill="currentColor" font-size="11">0</text>
  <text x="285" y="266" fill="currentColor" font-size="11">1</text>
  <text x="175" y="282" fill="currentColor" font-size="11" text-anchor="middle">p-value</text>
  <line x1="340" y1="250" x2="570" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <line x1="340" y1="250" x2="340" y2="70" stroke="currentColor" stroke-width="1.5"/>
  <rect x="342" y="86.9" width="19" height="163.1" fill="#10b981" opacity="0.75"/>
  <rect x="364" y="237.9" width="19" height="12.1" fill="#10b981" opacity="0.75"/>
  <rect x="386" y="244.7" width="19" height="5.3" fill="#10b981" opacity="0.75"/>
  <rect x="408" y="247.1" width="19" height="2.9" fill="#10b981" opacity="0.75"/>
  <rect x="430" y="248.1" width="19" height="1.9" fill="#10b981" opacity="0.75"/>
  <rect x="452" y="248.8" width="19" height="1.2" fill="#10b981" opacity="0.75"/>
  <rect x="474" y="249.1" width="19" height="0.9" fill="#10b981" opacity="0.75"/>
  <rect x="496" y="249.4" width="19" height="0.6" fill="#10b981" opacity="0.75"/>
  <rect x="518" y="249.5" width="19" height="0.5" fill="#10b981" opacity="0.75"/>
  <rect x="540" y="249.6" width="19" height="0.4" fill="#10b981" opacity="0.75"/>
  <text x="455" y="62" fill="currentColor" font-size="12" text-anchor="middle">μ = 0.5, n = 30: spike at zero</text>
  <text x="365" y="80" fill="currentColor" font-size="10">86.3%</text>
  <text x="340" y="266" fill="currentColor" font-size="11">0</text>
  <text x="565" y="266" fill="currentColor" font-size="11">1</text>
  <text x="455" y="282" fill="currentColor" font-size="11" text-anchor="middle">p-value</text>
  <text x="60" y="320" fill="currentColor" font-size="12">Figure 9.3  Histograms of 200,000 two-sided z-test p-values (n=30, seed 9091). LEFT: null true — every decile</text>
  <text x="60" y="338" fill="currentColor" font-size="12">holds ≈10% of the mass (counts 19791–20129), which is Proposition 9.18(ii). RIGHT: μ=0.5 — 86.3% of p-values</text>
  <text x="60" y="356" fill="currentColor" font-size="12">land in [0,0.1] and the tail decays. Note the vertical scales differ (left max 15%, right max 90%).</text>
  <text x="60" y="374" fill="currentColor" font-size="12">A real pipeline's p-value histogram should look like one of these two shapes, or a mixture of them. Anything</text>
  <text x="60" y="392" fill="currentColor" font-size="12">else — a hump in the middle, a cliff at 0.05 — is evidence of a broken null or selective reporting.</text>
</svg>

### 9.4.2 What a $p$-value is not

> **Common Confusion 9.4.** *"But isn't $p = 0.03$ the probability that the null hypothesis is true, given my data?"*
>
> No, and the two numbers can differ by an order of magnitude. $p$ is computed *under* $H_0$: it is $\mathbb{P}(\text{data at least this extreme} \mid H_0)$, a conditional probability with $H_0$ on the right of the bar. The thing you want is $\mathbb{P}(H_0\mid\text{data})$, with $H_0$ on the *left*. Chapter 1 spent a whole section on why these differ — Proposition 1.12, and the disease test with prevalence $0.5\%$, sensitivity $99\%$, specificity $95\%$ giving $\mathbb{P}(D\mid +) = 9.05\%$ rather than $99\%$. The identical arithmetic applies here with "disease" $\to$ "$H_1$ true", "sensitivity" $\to$ "power", "1 - specificity" $\to$ "$\alpha$", and "prevalence" $\to$ "prior probability that the hypothesis you are testing is true". Swapping the two conditionals is the *prosecutor's fallacy*: "the probability of this DNA match if he were innocent is 1 in a million, therefore the probability he is innocent is 1 in a million" is wrong in exactly the way that "$p = 10^{-6}$, therefore $H_0$ has probability $10^{-6}$" is wrong. And note: a $p$-value cannot possibly equal $\mathbb{P}(H_0\mid \text{data})$, because it does not depend on the prior and that posterior must.

> **Proposition 9.20 (posterior probability of the null; the false-discovery arithmetic).** Consider testing $H_0$ vs $H_1$ with prior probabilities $\pi_0, \pi_1 = 1-\pi_0$, a level-$\alpha$ test with power $\beta$. Then
> $$\mathbb{P}(H_1 \mid \text{reject}) = \frac{\pi_1\beta}{\pi_1\beta + \pi_0\alpha}, \qquad \mathbb{P}(H_0\mid\text{reject}) = \frac{\pi_0\alpha}{\pi_1\beta+\pi_0\alpha} .$$
> In odds form: $\text{posterior odds} = \text{prior odds}\times \beta/\alpha$.

*Proof.* Bayes' rule, Proposition 1.12, with the event "reject" playing the role of the positive test result and the law of total probability giving the denominator. $\blacksquare$

**Worked Example 9.13 (how often is a "significant" result wrong?).** Fix $\alpha = 0.05$ and apply Proposition 9.20.

| $\pi_1$ (prior that $H_1$ holds) | power $\beta$ | $\mathbb{P}(H_1\mid\text{reject})$ | share of rejections that are false |
|---|---|---|---|
| 0.50 | 0.8 | 0.9412 | 5.9% |
| 0.50 | 0.2 | 0.8000 | 20.0% |
| 0.10 | 0.8 | 0.6400 | 36.0% |
| 0.10 | 0.5 | 0.5263 | 47.4% |
| 0.10 | 0.2 | 0.3077 | 69.2% |
| 0.05 | 0.8 | 0.4571 | 54.3% |
| 0.01 | 0.8 | 0.1391 | 86.1% |
| 0.01 | 0.2 | 0.0388 | 96.1% |

Now put ML numbers in. A hyperparameter sweep over 100 configurations in which, realistically, perhaps 5 are genuinely better than the baseline has $\pi_1 = 0.05$; if each comparison has 80% power, then *more than half* of the configurations you flag as improvements are noise. Drop the power to 20% — which is where you are with a 2,000-item test set and a 0.5-point true gap, by Worked Example 9.4's arithmetic — and 83% of your "wins" are noise. This is not a hypothetical; it is the standard explanation for why sweep-selected improvements evaporate on a fresh test set, and §9.6 gives the correction.

Notice the structural parallel to Chapter 1's disease test. There, low prevalence plus imperfect specificity destroyed precision, and the fix was better *specificity* (fewer false positives among the many negatives), not better sensitivity. Here, a low prior on $H_1$ plus $\alpha = 0.05$ destroys the credibility of rejections, and the fix is a smaller $\alpha$ (or an explicit multiplicity correction), not more power. Power helps, but only linearly; the denominator is dominated by $\pi_0\alpha$ when $\pi_0 \approx 1$.

> **Common Confusion 9.5.** *"$p < 0.05$ twice in a row is $p < 0.0025$, so replication makes the evidence square up."*
>
> Only under independence and only if both tests were pre-specified. If the second experiment reuses any of the first experiment's data — the same test set, the same pretrained checkpoint, the same benchmark — the two $p$-values are dependent and multiplying them is meaningless. And the meta-analytic combination of independent $p$-values is not multiplication anyway: Fisher's method uses $-2\sum_i \log p_i \sim \chi^2_{2m}$ under the joint null, which for $m=2$ and $p_1 = p_2 = 0.05$ gives $-2(2\log 0.05) = 11.98$, so a combined $p$ of $0.0175$, not $0.0025$. The product of $p$-values is not a $p$-value.

### 9.4.3 $p$-hacking, forking paths, and optional stopping

Proposition 9.18 has a hypothesis that is invisible in the formula and violated constantly: the statistic $T$, the null, and the direction were fixed before the data were seen. Everything in this subsection is a consequence of breaking that.

**$p$-hacking** is the deliberate or semi-deliberate search over analysis choices until $p < \alpha$: which metric, which subset, which outlier rule, which of 12 baselines, whether to include the seed that diverged. Each choice is a fork, and the *garden of forking paths* problem (Gelman and Loken) is that even an analyst who makes only one choice per fork — never running an alternative analysis at all — has an inflated error rate, because the choice was *data-dependent*. The relevant null distribution is that of $\min_j p_j$ over the paths you would have taken, not of the single $p$ you computed. This is unfixable after the fact, which is why preregistration exists.

**Optional stopping** is the version with the cleanest mathematics: keep collecting data and test after every batch, stopping when $p \le \alpha$. Since the test is valid at each fixed $n$, it feels harmless. It is not.

> **Proposition 9.21 (continuous monitoring rejects almost surely).** Let $X_1,X_2,\dots\overset{\text{iid}}{\sim}N(0,1)$ (the null), $S_n = \sum_{i\le n}X_i$, and $Z_n = S_n/\sqrt n$. Fix any critical value $c > 0$. Then
> $$\mathbb{P}\Big(\sup_{n\ge 1}|Z_n| > c\Big) = 1 ,$$
> so the procedure "test at every $n$, reject the first time $|Z_n| > c$" has Type I error rate exactly $1$ if allowed to run forever.

*Proof.* By the law of the iterated logarithm (see §9.10), $\limsup_{n\to\infty} \frac{S_n}{\sqrt{2n\log\log n}} = 1$ almost surely. Hence $\limsup_n |Z_n|/\sqrt{2\log\log n} = 1$ a.s., and since $\sqrt{2\log\log n}\to\infty$, we get $\limsup_n |Z_n| = \infty$ a.s. In particular $\sup_n |Z_n| > c$ a.s. $\blacksquare$

The result is not an artefact of infinite time; the convergence to 1 is fast enough to matter at realistic horizons, and there is a matching finite-$n$ statement.

**Worked Example 9.14 (optional stopping, measured).** Simulate under the null ($\sigma$ known, so the test is the exact $z$-test), with a maximum sample size of $n = 1000$, testing two-sided at $\alpha = 0.05$ and rejecting at the first peek that crosses. $B = 200{,}000$ replications, NumPy PCG64 seed 777.

| peek schedule | actual Type I error |
|---|---|
| 1 look (at $n=1000$) | 0.0497 |
| 2 looks ($n = 500,1000$) | 0.0820 |
| 5 looks (every 200) | 0.1404 |
| 10 looks (every 100) | 0.1914 |
| 20 looks (every 50) | 0.2467 |
| 50 looks (every 20) | 0.3197 |
| 100 looks (every 10) | 0.3724 |

And with continuous monitoring — testing at *every* $n$ from 10 up to $N$ ($B = 50{,}000$, seed 123):

| $N$ | 10 | 50 | 100 | 500 | 1000 | 5000 | 20000 |
|---|---|---|---|---|---|---|---|
| Type I error | 0.0506 | 0.2186 | 0.2797 | 0.4114 | 0.4607 | 0.5619 | 0.6337 |

Five peeks nearly triples the error rate. Monitoring continuously to $n = 20{,}000$ gives a 63% chance of a spurious "significant" finding when nothing is happening — and the numbers keep climbing toward 1 as Proposition 9.21 promises, just slowly (logarithmically, per the LIL).

The ML translation is direct and uncomfortable. Watching a training curve and stopping when the validation gap "looks significant" is optional stopping. Running an A/B test and checking the dashboard daily until it crosses is optional stopping. Adding evaluation items until the difference reaches significance is optional stopping. The remedies are known and easy: fix $n$ in advance; or use a **group sequential** design with $\alpha$ spent according to a pre-specified boundary (O'Brien–Fleming, Pocock) so that the *overall* error is $\alpha$; or use *anytime-valid* inference — confidence sequences and $e$-values — which are built so that the guarantee holds simultaneously for all $n$. The last of these is worth a paragraph because it is a modern tool that fits ML practice: a confidence sequence is a family $\{C_n\}$ with $\mathbb{P}(\forall n:\theta\in C_n)\ge 1-\alpha$, obtained from a supermartingale (Ville's inequality, see §9.10) rather than a fixed-$n$ CLT, and it lets you legitimately peek as often as you like at the cost of intervals that are wider by roughly $\sqrt{\log\log n}$ — precisely the LIL factor in Proposition 9.21. To fix the magnitude of that price: a fixed-$n$ 95% interval uses the multiplier $z_{0.975} = 1.960$, while the LIL scale $\sqrt{2\log\log n}$ is $1.966$, $2.107$, $2.211$ at $n = 10^3$, $10^4$, $10^5$ — actual confidence-sequence constructions carry somewhat larger constants, but the message stands: the width penalty for the right to monitor *forever* is on the order of 10–20% at practical sample sizes, not a factor of 2. That is the price of continuous monitoring, and it is the honest one; compare it with the alternative of an unnoticed Type I rate of 0.37 at a hundred peeks, and it is cheap. The companion object, the *$e$-value*, is a non-negative statistic with $\mathbb{E}_{H_0}[E] \le 1$; by Markov's inequality $1/E$ is a valid $p$-value, and unlike $p$-values, independent $e$-values may be legitimately combined by multiplication and accumulated across optional stopping and optional *continuation* — the algebra people wrongly attribute to $p$-values in Common Confusion 9.5 is exactly the algebra $e$-values actually have.

**The ASA statement (2016), in six lines.** The American Statistical Association's formal statement on $p$-values makes six points which I endorse and which this section has now derived: (1) $p$-values can indicate how incompatible data are with a specified model; (2) they do *not* measure the probability that the hypothesis is true (Proposition 9.20); (3) scientific conclusions should not be based only on whether $p$ crosses a threshold; (4) proper inference requires full reporting and transparency (Worked Example 9.14); (5) a $p$-value does not measure the size or importance of an effect (§9.9); (6) by itself, a $p$-value provides a poor measure of evidence — a likelihood ratio or an interval says more.

> **Self-check 9.4.**
> 1. Under $H_0$, what is $\mathbb{P}(p \le 0.05)$ for a continuous test statistic? For a discrete one?
> 2. Explain, in one sentence each, the difference between $\mathbb{P}(\text{data}\mid H_0)$ and $\mathbb{P}(H_0\mid\text{data})$, and name the extra ingredient the second one needs.
> 3. You peek five times and reject at the first crossing of $|z|>1.96$. What is your actual Type I error rate, approximately?
> 4. Your $p$-value histogram over 500 features has a big spike at zero *and* a smaller bump between 0.9 and 1.0. What might explain the bump?

## 9.5 Confidence sets, and why they are inverted tests

### 9.5.1 Definition and the correct interpretation

> **Definition 9.22 (confidence set).** Let $\{P_\theta\}_{\theta\in\Theta}$ be a model and $g:\Theta\to\mathcal{G}$ a function of interest (often $g = $ identity, or a single coordinate). A *set estimator* is a map $x \mapsto C(x)\subseteq \mathcal{G}$ such that $\{x : \gamma \in C(x)\}$ is measurable for each $\gamma$. It is a *confidence set of level* $1-\alpha$ if
> $$\inf_{\theta\in\Theta}\ \mathbb{P}_\theta\big(g(\theta) \in C(X)\big) \ \ge\ 1-\alpha .$$
> The left-hand side is the *coverage*. The set is *exact* if equality holds for every $\theta$, and *asymptotic* if the guarantee holds only in the limit: $\liminf_n \mathbb{P}_\theta(g(\theta)\in C_n(X_{1:n}))\ge 1-\alpha$ for each $\theta$. When $\mathcal{G}=\mathbb{R}$ and $C(x)$ is an interval it is a *confidence interval*.

The randomness lives in $C$, not in $\theta$. This is worth belabouring because the misreading is universal.

> **Common Confusion 9.6.** *"The 95% confidence interval is $[0.81, 0.87]$, so there is a 95% probability that the true accuracy is between 0.81 and 0.87."*
>
> Under the frequentist model, $\theta$ is a fixed unknown constant, so "$\mathbb{P}(\theta\in[0.81,0.87])$" is either 0 or 1 — there is no distribution over $\theta$ to take a probability with respect to. Chapter 1's framing is the clean way to see it: all randomness lives in $\omega$, and $\theta$ is not a function of $\omega$; the interval endpoints *are*. So the correct statement is about the *procedure*: "if I repeated this experiment many times, 95% of the intervals I would construct would contain the true value." The observed interval either does or does not. If you want $\mathbb{P}(\theta\in[a,b]\mid \text{data}) = 0.95$ you must put a prior on $\theta$ and compute a *credible* interval — which is Chapter 10's subject, and which is a different object with a different guarantee (correct on average over the prior, rather than for every fixed $\theta$). Neither is wrong; they answer different questions. What is wrong is claiming the Bayesian interpretation for a frequentist construction.
>
> Two further consequences that surprise people. First, a 95% confidence set is allowed to be *empty* or all of $\Theta$ for particular data, as long as coverage averages out — and inverted tests sometimes produce empty sets, which is a diagnostic that the model fits badly. Second, "the interval contains 0" and "the interval contains 0.001" are both non-rejections, so an interval is not a licence to assert the parameter equals any particular point in it.

### 9.5.2 The duality theorem

Here is the organizing result of the chapter. It says tests and confidence sets are the same information, transposed.

> **Theorem 9.23 (test–confidence-set duality).** Suppose for each $\theta_0\in\Theta$ we have a level-$\alpha$ test $\varphi_{\theta_0}$ of $H_0 : \theta = \theta_0$, non-randomized, with acceptance region $A(\theta_0) := \{x : \varphi_{\theta_0}(x) = 0\}$. Define
> $$C(x) := \{\theta_0 \in \Theta : x \in A(\theta_0)\} = \{\theta_0 : \text{the test does not reject } \theta_0\} .$$
> Then $C$ is a confidence set of level $1-\alpha$ for $\theta$. Conversely, given a level-$(1-\alpha)$ confidence set $C$, the test that rejects $H_0:\theta=\theta_0$ iff $\theta_0\notin C(x)$ has level $\alpha$.

*Proof.* Both directions are the same one-line equivalence of events. For fixed $\theta$,
$$\{x : \theta \in C(x)\} = \{x : x \in A(\theta)\} = A(\theta),$$
by the definitions. Hence
$$\mathbb{P}_\theta(\theta\in C(X)) = \mathbb{P}_\theta(X \in A(\theta)) = 1 - \mathbb{E}_\theta[\varphi_\theta] \ge 1-\alpha,$$
since $\varphi_\theta$ has level $\alpha$ for testing $\theta$. Taking the infimum over $\theta$ gives the coverage statement. For the converse, $\mathbb{E}_\theta[\mathbf{1}\{\theta\notin C(X)\}] = 1 - \mathbb{P}_\theta(\theta\in C(X)) \le \alpha$. (Measurability of $\{x:\theta\in C(x)\}$ is exactly measurability of $A(\theta)$, which holds since $\varphi_\theta$ is a test.) $\blacksquare$

One reconciliation with Definition 9.22, which allowed confidence sets for a general function $g(\theta)$ while the theorem is stated for $\theta$ itself. The same argument works verbatim for $g$: if for each value $\gamma \in \mathcal{G}$ you have a level-$\alpha$ test $\varphi_\gamma$ of the *composite* null $H_0 : g(\theta) = \gamma$ (level meaning $\sup_{\theta : g(\theta) = \gamma}\mathbb{E}_\theta[\varphi_\gamma] \le \alpha$, so nuisance directions are handled inside each test), then $C(x) = \{\gamma : x \in A(\gamma)\}$ has coverage $\ge 1-\alpha$ for $g(\theta)$, by the identical event identity $\{x : g(\theta) \in C(x)\} = A(g(\theta))$. Every interval for a single coordinate of a multi-parameter model in this chapter — the proportion intervals of §9.5.4 included — is an instance of this composite-null version.

Read it as a recipe: *to build an interval, take the test you trust and collect the hypotheses it fails to reject.* Everything about the interval is inherited from the test — good tests give good intervals, and a test with poor size control gives an interval with poor coverage. That inheritance is what makes §9.5.4's coverage disaster predictable from §9.3's Wald diagnosis.

<svg viewBox="0 0 620 400" style="max-width:100%;height:auto" role="img" aria-label="Test-confidence set duality shown as a family of acceptance regions">
  <line x1="80" y1="300" x2="540" y2="300" stroke="currentColor" stroke-width="1.5"/>
  <line x1="80" y1="300" x2="80" y2="60" stroke="currentColor" stroke-width="1.5"/>
  <text x="310" y="326" fill="currentColor" font-size="12" text-anchor="middle">observed statistic  x̄</text>
  <text x="34" y="180" fill="currentColor" font-size="12" text-anchor="middle" transform="rotate(-90 34 180)">hypothesized θ₀</text>
  <polygon points="80,264 212,264 520,96 388,96" fill="#3b82f6" opacity="0.18"/>
  <line x1="179" y1="210" x2="311" y2="210" stroke="#3b82f6" stroke-width="3"/>
  <text x="170" y="214" fill="currentColor" font-size="10" text-anchor="end">A(0.5)</text>
  <line x1="256" y1="168" x2="388" y2="168" stroke="#10b981" stroke-width="3"/>
  <text x="396" y="172" fill="currentColor" font-size="10">A(1.2)</text>
  <line x1="344" y1="120" x2="476" y2="120" stroke="#3b82f6" stroke-width="3"/>
  <text x="484" y="124" fill="currentColor" font-size="10">A(2.0)</text>
  <line x1="322" y1="60" x2="322" y2="308" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 3"/>
  <text x="322" y="322" fill="currentColor" font-size="11" text-anchor="middle">x̄ observed</text>
  <line x1="322" y1="132" x2="322" y2="204" stroke="#ef4444" stroke-width="6"/>
  <line x1="80" y1="204" x2="322" y2="204" stroke="#ef4444" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="80" y1="132" x2="322" y2="132" stroke="#ef4444" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="74" y="208" fill="#ef4444" font-size="11" text-anchor="end">0.6</text>
  <text x="74" y="136" fill="#ef4444" font-size="11" text-anchor="end">1.8</text>
  <text x="332" y="172" fill="#ef4444" font-size="11">C(x̄)</text>
  <text x="80" y="352" fill="currentColor" font-size="12">Figure 9.4  Duality. The blue band is the family of acceptance regions: for each hypothesized θ₀ (vertical axis),</text>
  <text x="80" y="370" fill="currentColor" font-size="12">A(θ₀) is the horizontal slice of x̄ values the level-α test does not reject. Reading the band VERTICALLY at the</text>
  <text x="80" y="388" fill="currentColor" font-size="12">observed x̄ gives the confidence set C(x̄) = {θ₀ : x̄ ∈ A(θ₀)} — here [0.6, 1.8]. Horizontal slices are tests;</text>
</svg>

Figure 9.4 makes the geometry concrete for $H_0:\mu=\theta_0$ with $\bar X \sim N(\theta_0, \sigma^2/n)$ and $A(\theta_0) = [\theta_0 - z_{1-\alpha/2}\sigma/\sqrt n,\ \theta_0 + z_{1-\alpha/2}\sigma/\sqrt n]$. The set of pairs $(\bar x,\theta_0)$ with $\bar x \in A(\theta_0)$ is a diagonal band in the plane. Slicing it horizontally at height $\theta_0$ gives the acceptance region of a test; slicing it vertically at the observed $\bar x$ gives a confidence interval. Same band, two readings. In the figure, $\theta_0 = 1.2$ is accepted (the observed $\bar x$ lies in its slice) while $\theta_0 = 0.5$ and $\theta_0 = 2.0$ are rejected, and correspondingly the vertical slice $C(\bar x) = [0.6, 1.8]$ contains $1.2$ and excludes the other two.

### 9.5.3 Pivots

Inverting a test requires solving $\{x\in A(\theta_0)\}$ for $\theta_0$, which is easy when the test statistic has a $\theta$-free distribution.

> **Definition 9.24 (pivotal quantity).** A function $Q(X,\theta)$ is a *pivot* for the model $\{P_\theta\}$ if the law of $Q(X,\theta)$ under $P_\theta$ is the same for every $\theta\in\Theta$. It is an *asymptotic pivot* if the law converges to a $\theta$-free limit.

> **Proposition 9.25 (pivots give exact confidence sets).** Let $Q$ be a pivot with known law $\mathcal{L}$, and choose $q_{\alpha/2}, q_{1-\alpha/2}$ with $\mathcal{L}([q_{\alpha/2}, q_{1-\alpha/2}]) \ge 1-\alpha$. Then
> $$C(x) := \{\theta : q_{\alpha/2} \le Q(x,\theta)\le q_{1-\alpha/2}\}$$
> has coverage $\ge 1-\alpha$ for every $\theta$, with equality if $\mathcal{L}$ is continuous.

*Proof.* $\mathbb{P}_\theta(\theta\in C(X)) = \mathbb{P}_\theta(q_{\alpha/2}\le Q(X,\theta)\le q_{1-\alpha/2}) = \mathcal{L}([q_{\alpha/2},q_{1-\alpha/2}])\ge 1-\alpha$, where the middle equality is exactly pivotality. $\blacksquare$

Examples you already know: $\sqrt n(\bar X - \mu)/\sigma \sim N(0,1)$ is an exact pivot in the Gaussian model with known $\sigma$; $\sqrt n(\bar X-\mu)/S_n \sim t_{n-1}$ is an exact pivot with $\sigma$ unknown; $(n-1)S_n^2/\sigma^2\sim\chi^2_{n-1}$ pivots the variance. And $\sqrt n(\bar X - \mu)/S_n \to_d N(0,1)$ for any distribution with finite variance (CLT plus Slutsky, Ch. 7) is an *asymptotic* pivot — which is the engine behind essentially every error bar in ML.

The distinction between exact and asymptotic pivots is where coverage goes wrong. An exact pivot gives $95\%$ at every $n$; an asymptotic one gives $95\%$ eventually, and "eventually" depends on the skewness of the underlying distribution in a way that Berry–Esseen (Ch. 7) quantifies. Binary outcomes near a boundary are the worst case, which brings us to the case ML actually cares about.

### 9.5.4 Four intervals for a proportion, and their real coverage

Let $X\sim\text{Bin}(n,p)$, $\hat p = X/n$. This is the model for "my classifier's accuracy on $n$ i.i.d. test items", so it deserves the care.

> **Proposition 9.26 (three inverted tests for a proportion).** Fix $\alpha$ and let $z = z_{1-\alpha/2}$.
> (i) Inverting the **Wald** test (which uses $\widehat{\mathrm{se}} = \sqrt{\hat p(1-\hat p)/n}$, estimated at $\hat p$) gives
> $$C_{\mathrm{Wald}} = \hat p \pm z\sqrt{\hat p(1-\hat p)/n}.$$
> (ii) Inverting the **score** test (which uses the null standard error $\sqrt{p_0(1-p_0)/n}$) gives the *Wilson* interval
> $$C_{\mathrm{Wilson}} = \frac{\hat p + \frac{z^2}{2n}}{1+\frac{z^2}{n}} \ \pm\ \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}} .$$
> (iii) Inverting the **likelihood-ratio** test gives $C_{\mathrm{LR}} = \{p : 2n[\hat p\log\frac{\hat p}{p} + (1-\hat p)\log\frac{1-\hat p}{1-p}]\le z^2\}$, an interval with no closed form.
> (iv) Inverting the **exact binomial** test gives the *Clopper–Pearson* interval $[B^{-1}_{\alpha/2}(x, n-x+1),\ B^{-1}_{1-\alpha/2}(x+1,n-x)]$ in terms of Beta quantiles, with coverage $\ge 1-\alpha$ for every $p$.

*Proof of (ii), the instructive one.* The score test does not reject $p_0$ iff $|\hat p - p_0| \le z\sqrt{p_0(1-p_0)/n}$, i.e. iff $(\hat p - p_0)^2 \le z^2 p_0(1-p_0)/n$. Expand: $p_0^2(1 + z^2/n) - p_0(2\hat p + z^2/n) + \hat p^2 \le 0$. This is a convex quadratic in $p_0$, so the solution set is the interval between its roots, and the quadratic formula gives the stated centre $(\hat p + \frac{z^2}{2n})/(1+\frac{z^2}{n})$ and half-width. Note two structural facts visible in the formula and invisible in the Wald formula: the centre is $\hat p$ shrunk toward $1/2$, and the half-width is positive even when $\hat p \in\{0,1\}$. $\blacksquare$

Both features are exactly what the Wald interval lacks: at $\hat p = 0$ the Wald interval is the single point $\{0\}$, coverage zero for every $p > 0$. Duality (Theorem 9.23) told us to expect trouble, since Worked Example 9.9 showed the Wald test misbehaving near the boundary. Let us measure it.

**Worked Example 9.15 (inverting a test, twice, by hand).** *First, the $z$-test.* Accept $\mu_0$ iff $|\bar x - \mu_0|\le z\sigma/\sqrt n$; solving for $\mu_0$ gives $\mu_0 \in [\bar x - z\sigma/\sqrt n, \bar x + z\sigma/\sqrt n]$. The familiar interval *is* the inverted test, and the algebra above is the entire derivation. This is why the interval and the test always agree: "0 is outside the 95% CI" and "$p < 0.05$ two-sided" are the same event by construction.

*Second, a proportion with $n=20$, $x=1$* (one error in twenty items, or one win in twenty comparisons — a very ML situation). $\hat p = 0.05$, $z = 1.959964$.

| interval | value | width |
|---|---|---|
| Wald | $[-0.04552,\ 0.14552]$ | 0.19103 |
| Wilson (score) | $[0.00888,\ 0.23613]$ | 0.22725 |
| Likelihood ratio | $[0.00292,\ 0.20223]$ | 0.19930 |
| Clopper–Pearson (exact) | $[0.00127,\ 0.24873]$ | 0.24747 |

The Wald interval contains negative probabilities. That alone should end its use for accuracy reporting, and yet $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$ is what most papers print. The three others are all sensible; they differ in how conservative they are, with Clopper–Pearson widest (it must be, to guarantee coverage for *every* $p$ despite the discreteness).

**Worked Example 9.16 (exact coverage, computed by enumeration — no simulation needed).** Because $X$ takes only $n+1$ values, coverage is a finite sum: $\sum_{x=0}^n \binom nx p^x(1-p)^{n-x}\mathbf{1}\{p\in C(x)\}$. I evaluated it exactly. Nominal level $95\%$ throughout.

| $n$ | $p$ | Wald | Wilson | LR | Clopper–Pearson |
|---|---|---|---|---|---|
| 20 | 0.01 | 0.1821 | 0.9831 | 0.9831 | 0.9831 |
| 20 | 0.05 | 0.6389 | 0.9245 | 0.9841 | 0.9841 |
| 20 | 0.10 | 0.8760 | 0.9568 | 0.8672 | 0.9887 |
| 20 | 0.50 | 0.9586 | 0.9586 | 0.9586 | 0.9586 |
| 40 | 0.01 | 0.3310 | 0.9393 | 0.9925 | 0.9925 |
| 40 | 0.05 | 0.8681 | 0.9520 | 0.8576 | 0.9861 |
| 40 | 0.30 | 0.9299 | 0.9443 | 0.9443 | 0.9615 |
| 100 | 0.01 | 0.6334 | 0.9206 | 0.9816 | 0.9816 |
| 100 | 0.05 | 0.8775 | 0.9659 | 0.9347 | 0.9826 |
| 100 | 0.50 | 0.9431 | 0.9431 | 0.9431 | 0.9648 |

And summarizing over a grid of 99 values $p \in \{0.01,0.02,\dots,0.99\}$ at $n = 40$:

| interval | min coverage | mean coverage | fraction of grid below 93% |
|---|---|---|---|
| Wald | 0.3310 | 0.8998 | 0.495 |
| Wilson | 0.9283 | 0.9508 | 0.020 |
| Likelihood ratio | 0.8576 | 0.9466 | 0.121 |
| Clopper–Pearson | 0.9519 | 0.9706 | 0.000 |

Sit with the Wald row. At $n=40$ and $p = 0.01$, the nominal-95% Wald interval covers the truth **33%** of the time. Averaged over the grid its coverage is 90%, and it is below 93% for half of all $p$ — even though $n = 40$ is not tiny. The mechanism is not mysterious: coverage fails whenever $\hat p$ lands at or near 0, because the interval's width is proportional to $\sqrt{\hat p(1-\hat p)}$ and therefore collapses precisely when the estimate is least reliable. Estimating the standard error at $\hat p$ instead of at the hypothesized $p_0$ is the whole error, and the score test's refusal to do that is why Wilson works.

Expected widths at $n=40$, $p=0.05$: Wald 0.1208, LR 0.1295, Wilson 0.1463, Clopper–Pearson 0.1568. So Wald buys its 13% narrower interval by breaking the guarantee it advertises. **Practical recommendation: use Wilson by default for proportions; use Clopper–Pearson when you need a guarantee (e.g. safety-critical error-rate claims) and can afford the width; never use Wald when $n\hat p < 10$ or $n(1-\hat p) < 10$.** A quick and surprisingly good alternative is the Agresti–Coull interval: add 2 successes and 2 failures and then use the Wald formula.

### 9.5.5 Simultaneous bands

Sometimes we need coverage for infinitely many quantities at once — every point of a CDF, every point of a learning curve, every threshold of an ROC.

> **Definition 9.27 (simultaneous confidence band).** For a function-valued parameter $\theta\mapsto f_\theta(\cdot)$, a *simultaneous* $(1-\alpha)$ band is a pair $L(\cdot), U(\cdot)$ of data-dependent functions with
> $$\mathbb{P}_\theta\big(L(t)\le f_\theta(t)\le U(t)\ \ \text{for all } t\big)\ \ge\ 1-\alpha .$$

The "for all $t$" is inside the probability and that is the entire content: pointwise intervals at each $t$ do *not* give a band, because the union of failure events over a continuum of $t$ has much larger probability than any one. This is precisely the countability warning from Chapter 1 — a supremum over an uncountable index set is a delicate object. Here it is rescued by monotonicity and right-continuity: $\sup_t|\hat F_n(t) - F(t)|$ equals a supremum over the countable set of jump points and rational $t$, hence is measurable.

> **Proposition 9.28 (a DKW band for a CDF).** Let $X_1,\dots,X_n$ be i.i.d. with CDF $F$ and let $\hat F_n$ be the empirical CDF (Chapter 1's empirical measure $\hat P_n$ evaluated on half-lines). Set $\varepsilon_n := \sqrt{\log(2/\alpha)/(2n)}$. Then
> $$\big[\max(0,\hat F_n(t) - \varepsilon_n),\ \min(1,\hat F_n(t)+\varepsilon_n)\big], \quad t\in\mathbb{R},$$
> is a simultaneous $(1-\alpha)$ confidence band for $F$, valid for every $n$ and every $F$.

*Proof.* The DKW inequality (Ch. 7) states $\mathbb{P}(\sup_t|\hat F_n(t)-F(t)| > \varepsilon)\le 2e^{-2n\varepsilon^2}$. Setting $2e^{-2n\varepsilon^2} = \alpha$ gives $\varepsilon = \varepsilon_n$ as above, and the complementary event is exactly the band containing $F$ everywhere. Truncating to $[0,1]$ can only help, since $F$ takes values there. $\blacksquare$

Numerically, $\varepsilon_n = 0.13581, 0.04295, 0.01358$ for $n = 100, 1000, 10000$ at $\alpha = 0.05$. Note the honest cost: a distribution-free simultaneous band over all $t$ at $n=100$ is $\pm 0.136$ wide, which is nearly useless, whereas a pointwise interval at a single $t$ where $F(t)=0.5$ is about $\pm 0.098$. The band is wider because it promises more, and the $\sqrt{\log(2/\alpha)}$ factor is the price of "for all $t$" — the same logarithmic price that $\log|\mathcal{H}|$ exacts in generalization bounds (Ch. 4), and for the same union-bound reason.

> **Self-check 9.5.**
> 1. State the coverage guarantee of a 95% confidence interval in a sentence that does not assign a probability to $\theta$.
> 2. You invert a test that has size $0.12$ instead of the intended $0.05$. What is the coverage of the resulting "95%" interval?
> 3. Why does the Wald interval for a proportion have coverage $0$ when $\hat p = 0$, and what does Wilson do differently?
> 4. Why is a simultaneous band for a CDF necessarily wider than a pointwise interval at a single $t$?

## 9.6 Multiple testing: the arithmetic of trying many things

### 9.6.1 Two error rates

If you run one test at level $0.05$ you have a 5% chance of a false claim. If you run 100 independent tests at level $0.05$ under the global null, the chance of *at least one* false claim is $1-0.95^{100} = 0.994$. That is not a subtle effect; it is a certainty. And a hyperparameter sweep is exactly 100 tests.

There are two natural things to control, and the choice between them is a scientific decision, not a technical one.

> **Definition 9.29 (FWER, FDP, FDR).** Consider $m$ null hypotheses $H_{0,1},\dots,H_{0,m}$, of which the set $\mathcal{H}_0$ with $|\mathcal{H}_0| = m_0$ are true. A procedure rejects a random subset; let $R$ be the number of rejections and $V$ the number of *false* rejections (rejected true nulls). Define
> - the *family-wise error rate* $\mathrm{FWER} := \mathbb{P}(V \ge 1)$;
> - the *false discovery proportion* $\mathrm{FDP} := V/\max(R,1)$, a random variable;
> - the *false discovery rate* $\mathrm{FDR} := \mathbb{E}[\mathrm{FDP}] = \mathbb{E}\big[V/\max(R,1)\big]$.
>
> A procedure *controls FWER at $\alpha$* if $\mathrm{FWER}\le\alpha$ under every configuration of true and false nulls; similarly for FDR at level $q$.

FWER says: *I do not want to be wrong even once.* FDR says: *I am willing to be wrong a fixed fraction of the time among the things I claim.* For a confirmatory claim ("this architecture change improves accuracy, and we are publishing it as the paper's contribution") FWER is the right target. For a screening step ("which of these 20,000 features deserve a closer look?") FDR is right, and insisting on FWER would leave you with nothing. Neither is more rigorous than the other; they are different promises. The convention $V/\max(R,1)$ handles $R=0$ by declaring $\mathrm{FDP}=0$: if you claim nothing you cannot be wrong.

### 9.6.2 Bonferroni and Holm: controlling FWER

> **Proposition 9.30 (Bonferroni).** Reject $H_{0,i}$ iff $p_i \le \alpha/m$. If each $p_i$ is super-uniform under its own null (Proposition 9.18), then $\mathrm{FWER}\le\alpha$, with no assumption whatsoever about dependence among the $p_i$.

*Proof.* Let $A_i := \{p_i \le \alpha/m\}$ for $i\in\mathcal{H}_0$. Then $\{V\ge1\} = \bigcup_{i\in\mathcal{H}_0}A_i$ and by countable subadditivity — Proposition 1.8(2), the union bound —
$$\mathbb{P}(V\ge1) \le \sum_{i\in\mathcal{H}_0}\mathbb{P}(p_i\le\alpha/m) \le m_0\cdot\frac{\alpha}{m}\le\alpha. \qquad\blacksquare$$

That is the entire content of Bonferroni: **it is the union bound**, which Chapter 1 introduced and Chapter 4 studied. Everything Chapter 1 said about the union bound therefore applies. It is tight when the events are disjoint, which for $p$-values means perfectly negatively associated — rare. It is loose when the events are positively dependent, which is the common case (correlated features, overlapping hyperparameter configurations, nested models), so Bonferroni is typically conservative in practice. And it goes vacuous in the same way: with $m = 10^5$ tests and $\alpha = 0.05$ the per-test threshold is $5\times10^{-7}$, which for a $z$-test means $|z| > 5.0$ — a demand that leaves you with the power of a stone. The $\log|\mathcal{H}|$ in generalization bounds is the same phenomenon from the same inequality.

Holm's procedure gets FWER control for free, with strictly more rejections.

> **Theorem 9.31 (Holm's step-down procedure).** Order the $p$-values $p_{(1)}\le\dots\le p_{(m)}$ with corresponding hypotheses $H_{(1)},\dots,H_{(m)}$. Let
> $$k := \min\Big\{ i : p_{(i)} > \frac{\alpha}{m-i+1}\Big\}$$
> (and $k = m+1$ if no such $i$ exists), and reject $H_{(1)},\dots,H_{(k-1)}$. Then $\mathrm{FWER}\le\alpha$ under arbitrary dependence. Moreover Holm rejects every hypothesis Bonferroni rejects.

*Proof.* The dominance claim is immediate: Holm's first threshold is $\alpha/m$, Bonferroni's threshold, and its later thresholds are larger, so any $p_{(i)}\le\alpha/m$ is rejected by Holm as well (the step-down cannot stop before reaching it, since all earlier ordered $p$-values are even smaller).

For FWER control, suppose $V\ge1$; we bound the probability of this event. Let $j$ be the true null with the smallest $p$-value among true nulls, $p_j = \min_{i\in\mathcal{H}_0}p_i$, and suppose $H_j$ is rejected, at step $\ell$ say (so $p_j = p_{(\ell)}$ and $\ell \le k-1$). All hypotheses rejected strictly before step $\ell$ have smaller $p$-values, and by the choice of $j$ none of them is a true null — so they are all false nulls, of which there are $m - m_0$. Hence $\ell - 1 \le m-m_0$, i.e. $\ell \le m-m_0+1$, so $m - \ell + 1 \ge m_0$. Because $H_j$ was rejected at step $\ell$, we have $p_j \le \alpha/(m-\ell+1) \le \alpha/m_0$. Therefore
$$\{V \ge 1\} \subseteq \Big\{\min_{i\in\mathcal{H}_0}p_i \le \frac{\alpha}{m_0}\Big\},$$
and by the union bound again $\mathbb{P}(\min_{i\in\mathcal{H}_0}p_i\le\alpha/m_0)\le \sum_{i\in\mathcal{H}_0}\alpha/m_0 = \alpha$. $\blacksquare$

The idea in one sentence: once you have rejected some hypotheses, the number of *remaining* candidate true nulls has gone down, so the surviving tests deserve a larger share of the budget. Holm dominates Bonferroni uniformly, requires no extra assumptions, and takes three lines of code. There is no reason ever to report Bonferroni instead — except that Bonferroni also yields simultaneous *confidence intervals* directly (each at level $1-\alpha/m$, by Theorem 9.23), while Holm's step-down does not correspond to a simple family of intervals.

### 9.6.3 Benjamini–Hochberg: controlling FDR

> **Theorem 9.32 (Benjamini–Hochberg, 1995).** Fix $q\in(0,1)$. Order $p_{(1)}\le\dots\le p_{(m)}$, let
> $$\hat k := \max\Big\{i : p_{(i)} \le \frac{q\,i}{m}\Big\}$$
> ($\hat k = 0$ if the set is empty), and reject $H_{(1)},\dots,H_{(\hat k)}$. Suppose the $p$-values are mutually independent and each null $p$-value is super-uniform. Then
> $$\mathrm{FDR} \le \frac{m_0}{m}\,q \ \le\ q .$$

*Proof.* Write $R$ for the number of rejections and $V = \sum_{i\in\mathcal{H}_0}\mathbf{1}\{i \text{ rejected}\}$. Then
$$\mathrm{FDR} = \mathbb{E}\Big[\frac{V}{R\vee1}\Big] = \sum_{i\in\mathcal{H}_0}\sum_{k=1}^m \frac1k\,\mathbb{P}\big(i \text{ rejected},\, R=k\big). \tag{9.3}$$
Fix $i\in\mathcal{H}_0$. The crux is a deterministic identity. Define $\tilde p$ to be the vector of $p$-values with the $i$-th entry replaced by $0$, and let $\tilde R$ be the number of rejections BH makes on $\tilde p$ (so $\tilde R\ge1$ always, and $\tilde R$ is a function of $\{p_j\}_{j\ne i}$ only). Let $N(t) := \#\{j : p_j \le t\}$ and $\tilde N(t) := \#\{j: \tilde p_j \le t\} = N(t) + \mathbf{1}\{p_i > t\}$. In this notation $R = \max\{k : N(qk/m)\ge k\}$ and likewise for $\tilde R$.

*Claim.* For every $k\ge1$,
$$\{i \text{ rejected},\ R = k\} = \{\tilde R = k,\ p_i \le qk/m\} .$$

($\subseteq$) Suppose $R = k$ and $i$ is rejected, so $p_i \le qk/m$. For any $j > k$ we have $qj/m > qk/m \ge p_i$, hence $\tilde N(qj/m) = N(qj/m) < j$, so $\tilde R \le k$. Also $\tilde N(qk/m)\ge N(qk/m)\ge k$, so $\tilde R \ge k$. Thus $\tilde R = k$.

($\supseteq$) Suppose $\tilde R = k$ and $p_i \le qk/m$. Then $N(qk/m) = \tilde N(qk/m) \ge k$ (the indicator vanishes since $p_i \le qk/m$), so $R\ge k$; and for $j>k$, $N(qj/m)\le\tilde N(qj/m) < j$, so $R\le k$. Hence $R = k$ and $p_i \le qk/m = qR/m$, so $i$ is rejected. This proves the claim.

Now use independence: $\tilde R$ depends only on $\{p_j\}_{j\ne i}$, so it is independent of $p_i$. Substituting into (9.3) and using super-uniformity $\mathbb{P}(p_i \le t)\le t$:
$$\sum_{k=1}^m\frac1k\mathbb{P}\big(p_i \le qk/m,\ \tilde R = k\big) = \sum_{k=1}^m \frac1k\,\mathbb{P}(p_i\le qk/m)\,\mathbb{P}(\tilde R = k) \le \sum_{k=1}^m\frac1k\cdot\frac{qk}{m}\,\mathbb{P}(\tilde R=k) = \frac qm\sum_{k=1}^m \mathbb{P}(\tilde R = k)\le \frac qm .$$
Summing over the $m_0$ true nulls gives $\mathrm{FDR}\le m_0 q/m\le q$. $\blacksquare$

The proof rewards a second reading, because the "replace $p_i$ by 0" device is the standard trick for decoupling a data-dependent threshold from the datum it is being applied to, and it recurs throughout selective-inference theory. Note also the slack: the bound is $m_0 q/m$, so if only 10% of hypotheses are truly null, BH is actually controlling FDR at $0.1q$ — which is the motivation for adaptive procedures (Storey's method) that estimate $m_0$ and recover the lost power.

Beyond independence, the honest summary is: BH also controls FDR under *positive regression dependence* (PRDS), which covers many practically relevant correlation structures (Benjamini–Yekutieli 2001); under arbitrary dependence, the Benjamini–Yekutieli procedure with thresholds $qi/(m H_m)$, $H_m = \sum_{j\le m}1/j$, controls FDR, at the cost of a $\log m$ factor. These results I state without proof (see §9.10).

**Worked Example 9.17 (all three procedures on ten $p$-values, by hand).** Suppose ten ablations give sorted $p$-values $0.0008$, $0.009$, $0.012$, $0.021$, $0.032$, $0.045$, $0.11$, $0.23$, $0.47$, $0.88$, and take $\alpha = q = 0.05$, $m=10$.

| $i$ | $p_{(i)}$ | Bonferroni $\alpha/m$ | Holm $\alpha/(m-i+1)$ | BH $qi/m$ |
|---|---|---|---|---|
| 1 | 0.0008 | 0.0050 ✓ | 0.00500 ✓ | 0.005 ✓ |
| 2 | 0.0090 | 0.0050 | 0.00556 | 0.010 ✓ |
| 3 | 0.0120 | 0.0050 | 0.00625 | 0.015 ✓ |
| 4 | 0.0210 | 0.0050 | 0.00714 | 0.020 |
| 5 | 0.0320 | 0.0050 | 0.00833 | 0.025 |
| 6 | 0.0450 | 0.0050 | 0.01000 | 0.030 |
| 7 | 0.1100 | 0.0050 | 0.01250 | 0.035 |
| 8 | 0.2300 | 0.0050 | 0.01667 | 0.040 |
| 9 | 0.4700 | 0.0050 | 0.02500 | 0.045 |
| 10 | 0.8800 | 0.0050 | 0.05000 | 0.050 |

Bonferroni rejects 1. Holm stops at $i=2$ ($0.0090 > 0.00556$) and also rejects 1. BH finds the largest $i$ with $p_{(i)}\le 0.005i$: $i=3$ works ($0.012 \le 0.015$), $i=4$ fails ($0.021 > 0.020$), and nothing above works — so $\hat k = 3$ and BH rejects three. Note the *step-up* logic: BH looks for the largest $i$ that passes and rejects everything below it, so a hypothesis can be rejected even if its own $p$-value exceeds its own threshold, as long as some larger $i$ passes. (Here $p_{(2)} = 0.009 \le 0.010$ anyway.) Under arbitrary dependence, Benjamini–Yekutieli with $H_{10}=2.92897$ has thresholds $0.00171, 0.00341, \dots, 0.01707$ and rejects only 1 — the price of assuming nothing.

<svg viewBox="0 0 620 400" style="max-width:100%;height:auto" role="img" aria-label="Sorted p-values against Bonferroni, Holm, and Benjamini-Hochberg thresholds">
  <line x1="90" y1="290" x2="530" y2="290" stroke="currentColor" stroke-width="1.5"/>
  <line x1="90" y1="290" x2="90" y2="60" stroke="currentColor" stroke-width="1.5"/>
  <text x="310" y="316" fill="currentColor" font-size="12" text-anchor="middle">rank i of sorted p-value</text>
  <text x="34" y="175" fill="currentColor" font-size="12" text-anchor="middle" transform="rotate(-90 34 175)">p-value</text>
  <text x="84" y="294" fill="currentColor" font-size="10" text-anchor="end">0</text>
  <text x="84" y="199" fill="currentColor" font-size="10" text-anchor="end">0.025</text>
  <text x="84" y="107" fill="currentColor" font-size="10" text-anchor="end">0.05</text>
  <line x1="90" y1="271.7" x2="495" y2="271.7" stroke="#ef4444" stroke-width="2"/>
  <text x="500" y="275" fill="#ef4444" font-size="10">Bonferroni</text>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2" points="90,271.7 135,269.6 180,267.1 225,263.8 270,259.4 315,253.3 360,244.2 405,228.9 450,198.3 495,106.7"/>
  <text x="500" y="112" fill="#f59e0b" font-size="10">Holm</text>
  <line x1="90" y1="271.7" x2="495" y2="106.7" stroke="#10b981" stroke-width="2"/>
  <text x="452" y="132" fill="#10b981" font-size="10">BH: qi/m</text>
  <circle cx="90" cy="287.1" r="4" fill="#3b82f6"/>
  <circle cx="135" cy="257.0" r="4" fill="#3b82f6"/>
  <circle cx="180" cy="246.0" r="4" fill="#3b82f6"/>
  <circle cx="225" cy="213.0" r="4" fill="currentColor"/>
  <circle cx="270" cy="172.7" r="4" fill="currentColor"/>
  <circle cx="315" cy="125.0" r="4" fill="currentColor"/>
  <text x="360" y="80" fill="currentColor" font-size="10" text-anchor="middle">↑ 0.11</text>
  <text x="405" y="80" fill="currentColor" font-size="10" text-anchor="middle">↑ 0.23</text>
  <text x="450" y="80" fill="currentColor" font-size="10" text-anchor="middle">↑ 0.47</text>
  <text x="495" y="80" fill="currentColor" font-size="10" text-anchor="middle">↑ 0.88</text>
  <text x="90" y="340" fill="currentColor" font-size="12">Figure 9.5  Ten sorted p-values (dots) against three thresholds. Bonferroni is a flat line at α/m; Holm rises</text>
  <text x="90" y="358" fill="currentColor" font-size="12">as α/(m−i+1) but STOPS at the first failure; BH is the straight line qi/m and rejects everything up to the LAST</text>
  <text x="90" y="376" fill="currentColor" font-size="12">crossing. Blue dots are BH's three rejections; only the first is rejected by Bonferroni or Holm. Four p-values</text>
  <text x="90" y="394" fill="currentColor" font-size="12">exceed the plotted range and are shown as arrows with their values.</text>
</svg>

**Simulation: what do you actually gain?** $m = 1000$ hypotheses of which $m_0 = 900$ are null; the 100 non-nulls have $z$-statistics shifted by $\mu = 3$; two-sided $p$-values; $q = \alpha = 0.05$; equicorrelated $z$'s with a single shared factor of strength $\rho$. $B=2000$ replications, NumPy PCG64 seed 1.

| $\rho$ | procedure | FWER | FDR | average power |
|---|---|---|---|---|
| 0.0 | Bonferroni | 0.0390 | 0.0028 | 0.1461 |
| 0.0 | Holm | 0.0390 | 0.0028 | 0.1469 |
| 0.0 | BH | 0.9030 | 0.0459 | 0.4954 |
| 0.5 | Bonferroni | 0.0125 | 0.0051 | 0.1451 |
| 0.5 | Holm | 0.0145 | 0.0051 | 0.1466 |
| 0.5 | BH | 0.2175 | 0.0241 | 0.4705 |
| 0.9 | Bonferroni | 0.0000 | 0.0000 | 0.1419 |
| 0.9 | Holm | 0.0000 | 0.0000 | 0.1440 |
| 0.9 | BH | 0.0335 | 0.0136 | 0.4802 |

Three things to read off. **BH's power advantage is enormous** — 0.495 against 0.147, a factor of 3.4 — and it is not free: BH's FWER is 0.90 at $\rho=0$, meaning that in 90% of replications *at least one* of its rejections is false. That is what "controlling FDR, not FWER" means, made concrete. **BH's FDR sits right at the theoretical bound**: the measured $0.0459$ at $\rho = 0$ is a hair *above* $m_0q/m = 0.045$ and comfortably below $q = 0.05$. With 2000 replications the Monte Carlo standard error is about $0.002$, so $0.0459$ and $0.045$ are statistically indistinguishable — which is the informative reading: Theorem 9.32's bound is *tight*, not slack. If you had measured $0.030$ you would suspect your simulation; measuring $0.046$ is what a correct implementation looks like. **Positive dependence makes everything conservative**: at $\rho = 0.9$, BH's realized FDR is 0.0136, so the guarantee holds with a large margin — consistent with the PRDS theory. In an all-null configuration ($m_0 = m$), all three procedures had FWER $= $ FDR: 0.0505 (Bonferroni, Holm, and BH alike) at $\rho=0$, since with no true effects BH reduces to something close to Bonferroni.

> **Common Confusion 9.7.** *"BH is strictly better than Bonferroni — it's newer and more powerful, so I should always use it."*
>
> They control different things. The table above shows BH with FWER 0.90: if your paper's headline claim is a single discovery from a sweep, BH gives you a 90% chance that *some* claim in your table is spurious, which may be fine for a screening table and fatal for a headline. Use FWER control when any single false claim damages the conclusion; use FDR when you are producing a list for follow-up. Also note BH's assumption: independence (or PRDS). Bonferroni and Holm need *nothing*. When your $m$ tests are 500 hyperparameter configurations whose performances are correlated in unknown and non-monotone ways, Holm's assumption-free guarantee may genuinely be the safer buy.

### 9.6.4 Selection bias in ML: leaderboards, sweeps, and holdout decay

> **Proposition 9.33 (selection inflates the selected estimate).** Let $\hat\theta_1,\dots,\hat\theta_m$ be estimators with $\mathbb{E}[\hat\theta_j] = \theta_j$, and let $\hat j := \arg\max_j\hat\theta_j$. Then
> $$\mathbb{E}\big[\hat\theta_{\hat j}\big] = \mathbb{E}\big[\max_j \hat\theta_j\big]\ \ge\ \max_j \mathbb{E}[\hat\theta_j] = \max_j\theta_j ,$$
> with strict inequality unless the maximizer is almost surely the same index. In particular the *winner's reported score is an upwardly biased estimate of the winner's true quality*, and the bias grows with $m$ and with the noise level.

*Proof.* $\max_j\hat\theta_j \ge \hat\theta_i$ pointwise for each $i$, so taking expectations, $\mathbb{E}[\max_j\hat\theta_j]\ge\theta_i$ for each $i$, hence $\ge \max_i\theta_i$. (This is Jensen's inequality for the convex function $\max$.) Strictness: if the inequality is an equality then $\max_j\hat\theta_j = \hat\theta_{i^*}$ a.s. for the maximizing $i^*$. $\blacksquare$

**Worked Example 9.18 (the winner's curse, quantified).** Suppose $m$ models all have the *same* true accuracy $0.850$ and are evaluated on the same $n_{\text{test}} = 2000$ items, independently across models. Per-model standard error: $\sqrt{0.85\times0.15/2000} = 0.00798$, i.e. $0.8$ accuracy points. Simulating $R = 200{,}000$ leaderboards (seed 31415):

| $m$ | $\mathbb{E}[\text{winner's score}]$ | upward bias | $\mathbb{P}(\text{winner} \ge 0.870)$ |
|---|---|---|---|
| 1 | 0.85003 | +0.00 pts | 0.006 |
| 10 | 0.86218 | +1.22 pts | 0.058 |
| 50 | 0.86770 | +1.77 pts | 0.260 |
| 200 | 0.87152 | +2.15 pts | 0.697 |
| 1000 | 0.87530 | +2.53 pts | 0.998 |

With 50 submissions of *identical* quality, the leaderboard's top entry averages 1.77 points above the truth, and there is a 26% chance the top score reads $\ge 0.870$ — a "2-point improvement" that does not exist. A direct check of the regression-to-the-mean interpretation: for $m=50$, the winner's mean leaderboard score is $0.86771$ and the same winner's mean score on a fresh test set is $0.85002$, a drop of $0.01769$ — precisely the bias. This is the mechanism behind "state-of-the-art results don't replicate", and it requires no bad faith at all: it is Proposition 9.33 plus a public leaderboard.

The same arithmetic applies inside your own laptop. A sweep over 200 hyperparameter settings, scored on a 2000-item validation set, will hand you a winner whose validation score overstates its quality by about 2 points. The remedies are all versions of the same idea: (i) *hold something back* — select on validation, report on a test set touched once; (ii) *correct the selected estimate* — the difference between selected-validation and fresh-test score *is* the bias, so report both; (iii) *treat the sweep as multiple testing* — if you want a $p$-value for "the best configuration beats the baseline", the relevant null distribution is that of the *maximum* over configurations, which is what a permutation test over the whole sweep gives you (§9.7).

**Holdout decay and adaptive data analysis.** The problem compounds when the analyst is adaptive: each query to the holdout leaks information about it, and after enough queries the holdout is effectively training data. Here is a demonstration with no signal at all. Take $n=200$ examples, $d = 500$ random $\pm1$ features, and labels $y_i$ drawn independently of $X$ — so the true accuracy of every classifier is $0.5$. Select the $k=21$ features with the largest correlation with $y$ *on this very set*, and build the sign-vote classifier from them. Over 400 replications (seed 2718281): accuracy on the reused set is $0.7861$ (sd 0.0222); accuracy of the same classifier on fresh data is $0.5025$ (sd 0.0340). An inflation of **28 accuracy points, from pure noise.** The adaptive data analysis literature (Dwork et al.) shows that this leakage can be controlled — differentially private or otherwise noise-limited access to a holdout supports exponentially many adaptive queries — but the practical version is simpler: count your queries to the test set, and treat a test set that has been queried hundreds of times as what it now is, a validation set.

> **Self-check 9.6.**
> 1. Bonferroni is a corollary of which Chapter 1 result?
> 2. In the simulation table, BH had FDR $0.046$ and FWER $0.903$. Explain how both numbers can be correct simultaneously.
> 3. Holm dominates Bonferroni in rejections. Why is that not a free lunch violating some conservation law?
> 4. Your sweep of 100 configurations picks a winner beating the baseline by 1.5 points on a 2000-item validation set. What is your best guess for the gap on fresh data, and why?

## 9.7 Permutation and randomization tests

### 9.7.1 Construction and exactness

Everything so far needed a model: a likelihood, or at least a CLT. Permutation tests need almost nothing, and the guarantee they give is exact at every $n$. For comparing two systems on the same data — the central ML use case — they are frequently the right default.

The idea is combinatorial. If the null says the two groups are interchangeable, then any relabelling of the data is as probable as the labelling we actually saw. So we can compute the null distribution of *any* statistic by brute force: shuffle the labels, recompute, repeat.

> **Definition 9.34 (exchangeability and the randomization group).** A random vector $(X_1,\dots,X_n)$ is *exchangeable* if $(X_{\pi(1)},\dots,X_{\pi(n)}) \overset{d}{=} (X_1,\dots,X_n)$ for every permutation $\pi$ of $\{1,\dots,n\}$. More generally, let $\mathcal{G}$ be a finite group of transformations $g:\mathcal{X}\to\mathcal{X}$. The null hypothesis is *$\mathcal{G}$-invariant* if $g(X)\overset d= X$ under $H_0$ for every $g\in\mathcal{G}$.

Standard instances: (i) two-sample problems, $\mathcal{G}$ = all $\binom{n}{n_1}$ relabellings of which units are "group A"; (ii) paired problems, $\mathcal{G}$ = all $2^n$ sign flips of the paired differences; (iii) tests of independence between $X$ and $Y$, $\mathcal{G}$ = permutations of the $Y$'s against fixed $X$'s.

> **Theorem 9.35 (exactness of the randomization test).** Let $\mathcal{G}$ be a finite group with $|\mathcal{G}| = M$, let $T:\mathcal{X}\to\mathbb{R}$ be any statistic, and suppose $g(X)\overset d= X$ for all $g\in\mathcal{G}$ under $H_0$. Define
> $$p := \frac{1}{M}\#\{g\in\mathcal{G} : T(g(X))\ \ge\ T(X)\} .$$
> Then under $H_0$, $\mathbb{P}(p \le \alpha)\le\alpha$ for every $\alpha\in[0,1]$; and if $\alpha$ is an integer multiple of $1/M$ and $T$ has no ties, $\mathbb{P}(p\le\alpha) = \alpha$ exactly.

*Proof.* Consider the multiset $\mathcal{T} := \{T(g(X)) : g \in \mathcal{G}\}$ of $M$ values. Key observation: $\mathcal{T}$ is *invariant* under the group action — for any $h\in\mathcal{G}$, $\{T(g(h(X))): g\in\mathcal{G}\} = \{T(g'(X)) : g'\in\mathcal{G}\}$ because $g\mapsto gh$ is a bijection of $\mathcal{G}$. Now, under $H_0$, $X \overset d= h(X)$ for each $h$, so if we let $H$ be uniform on $\mathcal{G}$ independent of $X$, the pair $(\mathcal{T}, T(H(X)))$ has the same distribution as $(\mathcal{T}, T(X))$ — i.e. conditionally on the multiset $\mathcal{T}$, the observed value $T(X)$ is distributed as a uniform draw from $\mathcal{T}$.

Conditionally on $\mathcal{T}$, then, $p$ is the fraction of elements of $\mathcal{T}$ that are $\ge$ a uniformly drawn element. If the values are distinct, that fraction is uniform on $\{1/M, 2/M,\dots,1\}$, so $\mathbb{P}(p\le \alpha \mid \mathcal{T}) = \lfloor \alpha M\rfloor/M \le \alpha$, with equality when $\alpha M \in \mathbb{Z}$. With ties, the fraction only increases, so the probability only decreases. Taking expectations over $\mathcal{T}$ gives the claim. $\blacksquare$

Notice what the proof did *not* use: no distributional form, no independence between coordinates, no moments, no asymptotics, no continuity. It used one thing — that the group action preserves the null distribution — and combinatorics. That is why permutation tests are exact at $n=6$.

Notice also what is genuinely assumed. Exchangeability is a real hypothesis and it can fail. In a two-sample test, the permutation null is "the two samples come from the same distribution", so rejection can be driven by a *variance* difference even when means are equal; if your intended null is only "equal means", the permutation test is testing something broader. For dependent data — time series, spatial data, or evaluation items drawn in correlated blocks — plain permutation destroys the dependence structure and the test is not valid; you need a group that respects it (block permutation, circular shifts). And in a paired ML comparison the exchangeability we rely on is *within pairs* — the two models' scores on the same item are exchangeable under the null — which is a much weaker and more defensible assumption than anything parametric, but still an assumption.

> **Proposition 9.36 (Monte Carlo permutation is still exact).** Let $g_1,\dots,g_B$ be i.i.d. uniform on $\mathcal{G}$, independent of $X$, and define
> $$\hat p := \frac{1 + \#\{b : T(g_b(X)) \ge T(X)\}}{B+1} .$$
> Then under $H_0$, $\mathbb{P}(\hat p \le \alpha)\le\alpha$ for all $\alpha$.

*Proof.* Under $H_0$ the $B+1$ values $T(X), T(g_1(X)),\dots,T(g_B(X))$ are exchangeable. Here is the formal step, since it is short: draw an extra uniform $h \in \mathcal{G}$ independent of everything, and consider $(T(hX), T(g_1hX), \dots, T(g_BhX))$. Because $hX \overset d= X$ under $H_0$ and $(g_1h, \dots, g_Bh)$ is again an i.i.d. uniform tuple independent of $hX$ (multiplication by $h$ is a bijection of $\mathcal{G}$), this vector has the same joint law as the original one; and by construction its first coordinate plays no special role — writing $h_0 = h$, $h_b = g_b h$, all $B+1$ entries are of the identical form $T(h_b X)$ with $(h_0, \dots, h_B)$ an exchangeable tuple. Hence any permutation of the $B+1$ values has the same joint law. Hence the rank of $T(X)$ among the $B+1$ values is uniform on $\{1,\dots,B+1\}$ up to ties, and $\hat p$ is (rank from the top)$/(B+1)$, so $\mathbb{P}(\hat p \le \alpha)\le \lfloor \alpha(B+1)\rfloor/(B+1)\le\alpha$. $\blacksquare$

The "$+1$" in both numerator and denominator is not a fudge; it is the identity element, and dropping it breaks validity. With $B = 999$ the smallest attainable $p$-value is $1/1000$; if you need to claim $p < 10^{-4}$ you need $B \ge 10^4$ permutations, which is a compute decision, not a statistical one.

**Worked Example 9.19 (an exact test on ten numbers).** Two variants of a model, each trained on five different splits, giving accuracy gains over baseline (in points): A $=\{1.2, 0.4, 2.1, 0.9, 1.6\}$, B $=\{-0.3, 0.7, 0.1, -0.8, 0.5\}$. Observed difference of means $= 1.2000$. The group of relabellings has $\binom{10}{5} = 252$ elements; enumerating all of them (no Monte Carlo needed):

- one-sided exact $p = 3/252 = 0.011905$;
- two-sided exact $p = 6/252 = 0.023810$;
- pooled two-sample $t$-test: $t = 3.0170$, $p = 0.016635$;
- Wilcoxon rank-sum: $p = 0.031746$.

All three agree qualitatively, and the permutation $p$-value is the one that requires no normality assumption at $n_1=n_2=5$. Note the granularity: the smallest attainable one-sided $p$-value is $1/252 = 0.003968$, so with five runs per arm you *cannot* report $p < 0.001$ no matter how large the effect. That is an honest statement about what ten numbers can support.

**Validity under heavy tails.** Because permutation exactness does not need moments, it survives distributions where the $t$-test's calibration degrades. I generated $R = 20{,}000$ null datasets with $n_1=n_2=15$ from a $t$-distribution with 2 degrees of freedom (infinite variance), seed 4242, and tested at $\alpha = 0.05$: the Monte Carlo permutation test ($B = 199$) had empirical size $0.0484$, and the two-sample $t$-test had empirical size $0.0379$. The $t$-test is conservative here rather than liberal — heavy tails inflate the estimated standard error — which costs power; the permutation test hits its nominal size on the nose, as Theorem 9.35 guarantees it must.

Permutation testing does have real limitations, and they are not the ones people usually cite. It handles the null well and the *alternative* only through whatever statistic you chose; it gives a $p$-value but not directly a confidence interval (for that, invert it over a family of shifted nulls, or use the bootstrap of §9.8); and its cost is $B$ times the cost of computing $T$, which for a statistic requiring model retraining is prohibitive — though for a statistic computed from *cached per-item predictions*, which is the ML case, it is trivially cheap.

> **Self-check 9.7.**
> 1. What exactly is assumed by a paired permutation test on per-item score differences between two models?
> 2. Why must the identity permutation be included in the Monte Carlo $p$-value?
> 3. With $n_1 = n_2 = 4$, what is the smallest two-sided permutation $p$-value attainable?
> 4. Your evaluation items come in 50 groups of 20 sentences from the same document. Why is item-level permutation invalid, and what would you permute instead?

## 9.8 The bootstrap

### 9.8.1 The plug-in principle

Suppose we want the sampling distribution of $\hat\theta_n = t(\hat P_n)$, a statistic that is a functional of the data. If we knew $P$ we could simulate: draw many samples of size $n$ from $P$, recompute $\hat\theta_n$ each time, and look at the histogram. We do not know $P$. But by Glivenko–Cantelli (Ch. 7) we have an estimate that converges uniformly: the empirical distribution $\hat P_n = \frac1n\sum_i\delta_{X_i}$, which is Chapter 1's empirical measure built out of Dirac measures. The bootstrap is the single idea of substituting $\hat P_n$ for $P$ and then doing the simulation exactly as described.

> **Definition 9.37 (nonparametric bootstrap).** Given data $X_1,\dots,X_n$ and a statistic $\hat\theta_n = t(X_1,\dots,X_n)$, let $X_1^*,\dots,X_n^*$ be i.i.d. draws from $\hat P_n$ — equivalently, a sample of size $n$ drawn *with replacement* from the observed data — and set $\hat\theta_n^* := t(X_1^*,\dots,X_n^*)$. The *bootstrap distribution* is the conditional law of $\hat\theta_n^*$ given the data, denoted $P^*$, and it is approximated by drawing $B$ independent bootstrap samples.

Two levels of approximation are stacked here and they should be kept separate in your head. The **statistical** approximation is $P \approx \hat P_n$, and its quality depends on $n$ and on the smoothness of the functional $t$; this is where the bootstrap succeeds or fails, and no amount of computation fixes it. The **Monte Carlo** approximation is $P^* \approx$ the empirical distribution of $B$ resamples, and its quality depends only on $B$; use $B \ge 2000$ for intervals and $B\ge 10^4$ for tail quantiles.

> **Theorem 9.38 (bootstrap consistency for the mean).** Let $X_1,X_2,\dots$ be i.i.d. with $\mathbb{E}[X_1^2]<\infty$, $\mathrm{Var}(X_1) = \sigma^2 \in(0,\infty)$, and let $\bar X_n$ be the sample mean. Then, almost surely,
> $$\sup_{x\in\mathbb{R}}\Big|\ \mathbb{P}^*\big(\sqrt n(\bar X_n^* - \bar X_n)\le x\big)\ -\ \mathbb{P}\big(\sqrt n(\bar X_n - \mu)\le x\big)\Big| \ \longrightarrow\ 0 .$$

*Proof sketch, with the mechanism named.* Conditionally on the data, $X_1^*,\dots,X_n^*$ are i.i.d. from $\hat P_n$ with conditional mean $\bar X_n$ and conditional variance $\hat\sigma_n^2 = \frac1n\sum_i(X_i - \bar X_n)^2$. Apply the Berry–Esseen bound (Ch. 7) to this conditional i.i.d. sum: the Kolmogorov distance between the law of $\sqrt n(\bar X^*_n - \bar X_n)/\hat\sigma_n$ and $N(0,1)$ is at most $C\hat\rho_n/(\hat\sigma_n^3\sqrt n)$ where $\hat\rho_n = \frac1n\sum_i|X_i - \bar X_n|^3$. By the strong law, $\hat\sigma_n^2\to\sigma^2$ a.s.; and $\hat\rho_n/\sqrt n \to 0$ a.s. holds under only a second moment by a truncation argument (the third empirical moment can grow, but slower than $\sqrt n$). Hence the conditional law of $\sqrt n(\bar X^*_n - \bar X_n)$ converges to $N(0,\sigma^2)$ a.s. in Kolmogorov distance, and by the ordinary CLT so does the law of $\sqrt n(\bar X_n - \mu)$. Both converge to the same continuous limit, so the sup-distance between them vanishes. $\blacksquare$

The general version replaces the mean by a functional differentiable in the right sense: if $t$ is Hadamard-differentiable at $P$ and the empirical process is Donsker (Ch. 7), the bootstrap is consistent. That covers means, variances, quantiles at points of positive density, smooth $M$-estimators, AUC, and correlation. It does *not* cover extremes, and it fails for functionals that are not smooth at $P$ — which is exactly the failure catalogue of §9.8.3.

> **Common Confusion 9.8.** *"The bootstrap generates new data, so it lets me get error bars from a tiny sample."*
>
> It generates no information. A bootstrap resample contains, in expectation, only $n(1 - (1-1/n)^n)\approx 0.632n$ distinct original observations; the rest are duplicates. What the bootstrap estimates is the *variability of your estimator under resampling from the empirical distribution*, and it works only insofar as $\hat P_n$ resembles $P$. With $n = 8$, $\hat P_n$ is a crude approximation to $P$ and the bootstrap inherits that crudeness — it will happily report a tight interval with 60% coverage. The bootstrap replaces analytic work with computation; it does not replace data. A useful sanity check: the bootstrap cannot know about any part of the distribution your sample did not visit, so it systematically understates tail behaviour.

### 9.8.2 Four intervals

> **Definition 9.39 (bootstrap confidence intervals).** Let $\hat\theta$ be the estimate, $\hat\theta^*_{(\cdot)}$ the bootstrap replicates, $\hat{\mathrm{se}}$ a standard-error estimate, and $q^*_u$ the $u$-quantile of the bootstrap replicates. At nominal level $1-\alpha$:
> - **Normal**: $\hat\theta \pm z_{1-\alpha/2}\,\widehat{\mathrm{sd}}^*$, where $\widehat{\mathrm{sd}}^*$ is the standard deviation of the replicates.
> - **Percentile**: $[\,q^*_{\alpha/2},\ q^*_{1-\alpha/2}\,]$.
> - **Basic** (or "reverse percentile"): $[\,2\hat\theta - q^*_{1-\alpha/2},\ 2\hat\theta - q^*_{\alpha/2}\,]$.
> - **Studentized**: form $T^*_b = (\hat\theta^*_b - \hat\theta)/\hat{\mathrm{se}}^*_b$ with a standard error recomputed *within* each resample, and use $[\hat\theta - q^{T*}_{1-\alpha/2}\hat{\mathrm{se}},\ \hat\theta - q^{T*}_{\alpha/2}\hat{\mathrm{se}}]$.
> - **BCa** (bias-corrected and accelerated): percentile with adjusted quantile levels $\Phi\!\big(\hat z_0 + \frac{\hat z_0 + z_u}{1 - \hat a(\hat z_0 + z_u)}\big)$, where $\hat z_0 = \Phi^{-1}(\#\{\hat\theta^*_b<\hat\theta\}/B)$ corrects median bias and $\hat a$, computed from jackknife influence values, corrects skewness.

The percentile and basic intervals are reflections of each other and it is worth knowing why they differ. Percentile takes the bootstrap distribution *as* the distribution of $\hat\theta$; basic takes the bootstrap distribution of $\hat\theta^* - \hat\theta$ as that of $\hat\theta - \theta$ and then flips it. For a symmetric bootstrap distribution they coincide; for a skewed one they lean opposite ways.

> **Proposition 9.40 (the percentile interval respects monotone transformations).** Let $\phi$ be a strictly increasing continuous function and let $C = [q^*_{\alpha/2}, q^*_{1-\alpha/2}]$ be the percentile interval for $\theta$ built from replicates $\hat\theta^*_b$. Then the percentile interval for $\phi(\theta)$ built from the replicates $\phi(\hat\theta^*_b)$ is exactly $[\phi(q^*_{\alpha/2}), \phi(q^*_{1-\alpha/2})] = \phi(C)$.

*Proof.* Quantiles commute with strictly increasing continuous transformations: if $F_{\phi(Y)}(x) = F_Y(\phi^{-1}(x))$ then $q_{\phi(Y)}(u) = \phi(q_Y(u))$. Apply this to the empirical distribution of the replicates. $\blacksquare$

This is a real advantage over the normal interval. If you build a normal-approximation interval for an accuracy and it extends past 1, you have to truncate it by hand, which breaks the coverage story. If you build a percentile interval for the log-odds and exponentiate, you get exactly the percentile interval for the odds. The BCa interval keeps this property and additionally achieves *second-order accuracy* — coverage error $O(1/n)$ rather than $O(1/\sqrt n)$ — which is why it is the recommended default when you can afford the jackknife.

**Worked Example 9.20 (coverage, measured).** I computed actual coverage of nominal-95% intervals by simulation: $R = 2000$ datasets, $B = 1000$ bootstrap resamples each. Three data-generating processes, all targeting the mean.

*Normal(0,1), $n=25$ (seed 11):*

| method | coverage | mean width |
|---|---|---|
| normal | 0.9210 | 0.755 |
| percentile | 0.9205 | 0.752 |
| basic | 0.9200 | 0.752 |
| studentized | 0.9405 | 0.816 |
| BCa | 0.9225 | 0.754 |
| textbook $t$-interval | 0.9485 | 0.811 |

*Exponential(1), $n=20$ (seed 12):* normal 0.8975, percentile 0.9015, basic 0.8810, studentized **0.9415**, BCa 0.9085, $t$-interval 0.9175.

*Lognormal(0, 1.5), $n=20$ (seed 13), a severely skewed case with true mean $e^{1.125} = 3.0802$:* normal 0.7405, percentile 0.7565, basic 0.7060, studentized **0.8660**, BCa 0.7910, $t$-interval 0.7600.

Three lessons. **The studentized bootstrap is the best of the family**, consistently, because dividing by a resample-specific standard error removes the leading skewness term (an Edgeworth-expansion fact). **Percentile and basic intervals under-cover at small $n$** — by 3 points in the Gaussian case, which is where the plain $t$-interval wins because it is exact there. **Skewness is the enemy**: at $n=20$ from a lognormal, *every* method is badly wrong (76% coverage for percentile), and the honest response is not to pick a better interval but to report that the mean of a heavy-tailed quantity at $n=20$ is not estimable to 95% confidence. Note also that the studentized interval buys its coverage with width — 17.6 average width in the lognormal case versus 4.7 for percentile — which is the interval telling you the truth about how little you know.

### 9.8.3 When the bootstrap fails

> **Proposition 9.41 (the bootstrap fails for the maximum).** Let $X_1,\dots,X_n\overset{\text{iid}}{\sim}\mathrm{Unif}[0,\theta]$ and $\hat\theta = X_{(n)} = \max_i X_i$. Then, conditionally on the data,
> $$\mathbb{P}^*\big(X^*_{(n)} = X_{(n)}\big) = 1 - \Big(1-\frac1n\Big)^n \ \longrightarrow\ 1 - e^{-1} \approx 0.632121 .$$
> Consequently the bootstrap distribution of $n(\hat\theta - \hat\theta^*)$ has an atom of mass $\to 1-e^{-1}$ at $0$, whereas the true distribution of $n(\theta - \hat\theta)$ converges to $\mathrm{Exp}(1/\theta)$, which is continuous. The bootstrap is inconsistent.

*Proof.* $X^*_{(n)} < X_{(n)}$ iff the index attaining the maximum is missed by all $n$ draws, which has probability $(1-1/n)^n$ by independence of the draws; the complement gives the display, and $(1-1/n)^n\to e^{-1}$. The true limit is the standard extreme-value computation: $\mathbb{P}(n(\theta - X_{(n)}) > t) = (1 - t/(n\theta))^n \to e^{-t/\theta}$. A discrete limit with an atom cannot equal a continuous one, so the sup-distance in Theorem 9.38's conclusion does not vanish. $\blacksquare$

**Worked Example 9.21 (watching it fail).** $n = 50$, $\mathrm{Unif}[0,1]$, so $\theta = 1$. Predicted atom: $1 - (1-1/50)^{50} = 0.635830$. In one sample with observed maximum $0.99780$, the bootstrap distribution over $200{,}000$ resamples had an empirical atom of $0.6350$ at the observed maximum and took only **13 distinct values** in total, with quantiles $q_{0.025} = 0.8721$, $q_{0.5} = q_{0.975} = 0.99780$. The true sampling distribution of $X_{(50)}$ has quantiles $0.9288, 0.9862, 0.9995$ — continuous, and centred above the bootstrap's.

Coverage over $R=4000$ datasets ($B=1000$, seed 31337): the percentile interval covers $\theta = 1$ in **0.0000** of replications. Not 0.90, not 0.50 — zero, and necessarily so, since $X^*_{(n)}\le X_{(n)} < \theta$ almost surely, so the percentile interval lies entirely below $\theta$. The basic interval, which reflects, does better at $0.8770$ but is still wrong. For contrast, the exact pivotal interval $[X_{(n)}/0.975^{1/n},\ X_{(n)}/0.025^{1/n}]$ — using that $X_{(n)}/\theta$ has CDF $u^n$, an exact pivot — achieved $0.9457$ coverage.

The other failure modes, measured:

**Infinite variance.** For the Cauchy, the sample mean does not concentrate at all (it is again Cauchy), so no CLT holds and the bootstrap has nothing to be consistent with. Percentile-interval coverage of the centre, $R=3000$, $B=800$, seed 606:

| $n$ | 25 | 100 | 400 | 1600 |
|---|---|---|---|---|
| coverage | 0.9100 | 0.9010 | 0.9203 | 0.9113 |
| median width | 4.448 | 4.421 | 4.366 | 4.645 |

Coverage is stuck around 0.91 and — the real tell — **the width does not shrink with $n$ at all**. Compare $t$-distributed data with 5 degrees of freedom (finite variance): coverage $0.9160, 0.9380, 0.9490, 0.9530$ for the same $n$, converging to nominal as Theorem 9.38 requires. And $t$ with 1.5 degrees of freedom (infinite variance, finite mean) gives $0.9145, 0.9220, 0.9315, 0.9245$: stuck again. The diagnostic is available to you in practice: if your bootstrap interval width does not shrink like $n^{-1/2}$ as you add data, the bootstrap is not working, and you should target a different functional (the median, or a trimmed mean, both of which bootstrap fine).

**Dependence.** AR(1) data with $\rho = 0.8$, $n=200$, interval for the mean, $R = 2000$, $B=1000$, seed 5150: the i.i.d. bootstrap achieved coverage **0.4685**, because resampling destroys the serial dependence and reports the wrong variance. The true standard deviation of $\bar X_n$ is $\sqrt{(1+\rho)/((1-\rho)n)} = 0.21213$; the i.i.d. calculation implies $0.11785$, understating it by 1.8×. A moving-block bootstrap with block length $L=20$ recovered coverage $0.8410$ — better, and still not nominal, which is typical: block bootstraps trade bias for variance through $L$ and need $L\to\infty$ with $L/n\to0$.

**Boundary parameters.** If $\theta$ lies on the boundary of the parameter space (a variance component at zero, a mixture weight at zero, a constrained coefficient), $\hat\theta$ is not asymptotically normal and the bootstrap is inconsistent for the same reason Wilks' theorem fails there. Subsampling ($m$ out of $n$ without replacement, with $m/n\to0$) is consistent in many of these cases and is the standard repair.

A compact rule of thumb: **the bootstrap works when your statistic is a smooth functional of the empirical distribution and the data are i.i.d.; it fails at extremes, at boundaries, under infinite variance, and under dependence.** In ML terms it is excellent for accuracy, AUC, F1, BLEU, calibration error, and per-item paired differences on an i.i.d. test set; it is unreliable for worst-case metrics, max-over-runs, and anything computed on a time-ordered or clustered evaluation set.

<svg viewBox="0 0 620 400" style="max-width:100%;height:auto" role="img" aria-label="Bootstrap distribution compared with true sampling distribution for a mean and for a maximum">
  <line x1="70" y1="250" x2="290" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <polyline fill="none" stroke="currentColor" stroke-width="2" points="70,248.9 79,247.7 88,245.6 98,242.1 107,236.5 116,228.4 125,217.6 134,204.3 143,189.4 152,174.5 162,161.8 171,153.1 180,150.0 189,153.1 198,161.8 208,174.5 217,189.4 226,204.3 235,217.6 244,228.4 253,236.5 262,242.1 272,245.6 281,247.7 290,248.9"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="6 3" points="70,249.7 79,249.1 88,247.9 98,245.5 107,240.9 116,233.1 125,221.3 134,205.2 143,185.8 152,165.4 162,147.6 171,136.2 180,133.8 189,141.1 198,156.1 208,175.7 217,196.0 226,213.9 235,227.9 244,237.5 253,243.5 262,246.9 272,248.7 281,249.5 290,249.8"/>
  <text x="180" y="76" fill="currentColor" font-size="12" text-anchor="middle">MEAN, n = 25: it works</text>
  <text x="180" y="92" fill="#10b981" font-size="10" text-anchor="middle">dashed = bootstrap, solid = truth</text>
  <text x="180" y="272" fill="currentColor" font-size="10" text-anchor="middle">sd* = 0.172 vs true sd 0.200</text>
  <line x1="340" y1="250" x2="570" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <polyline fill="none" stroke="currentColor" stroke-width="2" points="340,250.0 350,250.0 359,249.9 369,249.9 378,249.9 388,249.8 398,249.7 407,249.6 417,249.4 426,249.2 436,248.9 445,248.4 455,247.8 465,247.0 474,245.8 484,244.1 493,241.9 503,238.8 513,234.6 522,228.9 532,221.1 541,210.4 551,196.0 560,176.5 570,150.0"/>
  <line x1="567" y1="250" x2="567" y2="112" stroke="#ef4444" stroke-width="5"/>
  <line x1="545" y1="250" x2="545" y2="215" stroke="#ef4444" stroke-width="4"/>
  <line x1="520" y1="250" x2="520" y2="230" stroke="#ef4444" stroke-width="4"/>
  <line x1="495" y1="250" x2="495" y2="239" stroke="#ef4444" stroke-width="4"/>
  <line x1="470" y1="250" x2="470" y2="243" stroke="#ef4444" stroke-width="4"/>
  <line x1="440" y1="250" x2="440" y2="246" stroke="#ef4444" stroke-width="4"/>
  <line x1="405" y1="250" x2="405" y2="248" stroke="#ef4444" stroke-width="4"/>
  <text x="455" y="76" fill="currentColor" font-size="12" text-anchor="middle">MAXIMUM, n = 50: it fails</text>
  <text x="455" y="92" fill="#ef4444" font-size="10" text-anchor="middle">bars = bootstrap: 13 atoms, 63.5% at X₍ₙ₎</text>
  <text x="455" y="272" fill="currentColor" font-size="10" text-anchor="middle">bootstrap can never exceed X₍ₙ₎</text>
  <text x="70" y="310" fill="currentColor" font-size="12">Figure 9.6  LEFT: for the sample mean, the bootstrap distribution (dashed, centred at x̄) closely tracks the true</text>
  <text x="70" y="328" fill="currentColor" font-size="12">sampling distribution (solid, centred at μ) in shape and spread — Theorem 9.38. RIGHT: for the maximum, the</text>
  <text x="70" y="346" fill="currentColor" font-size="12">true density (solid) rises smoothly toward θ, but the bootstrap distribution is a handful of atoms with 63.5%</text>
  <text x="70" y="364" fill="currentColor" font-size="12">of its mass on the observed maximum and nothing above it. Percentile-interval coverage of θ is exactly zero.</text>
  <text x="70" y="382" fill="currentColor" font-size="12">Smoothness of the functional, not sample size, is what separates the two panels.</text>
</svg>

> **Self-check 9.8.**
> 1. What fraction of the original observations appears in a typical bootstrap resample, and why is that number $1-e^{-1}$?
> 2. Which bootstrap interval is invariant under monotone reparametrization, and which had the best measured coverage for skewed data?
> 3. Why does the percentile interval for the maximum of a uniform sample have coverage exactly zero?
> 4. You bootstrap a mean and the interval width barely changes when you double $n$. What should you suspect?

## 9.9 Evaluating machine learning experiments honestly

This is the section the rest of the chapter was for. The claim "model A beats model B" is a statistical claim, and almost all of the ways it goes wrong are covered by results we have now proved.

### 9.9.1 Pair everything you can

> **Proposition 9.42 (why pairing works).** Let $(A_i, B_i)_{i\le n}$ be i.i.d. pairs of per-item scores with $\mathrm{Var}(A_i) = \sigma_A^2$, $\mathrm{Var}(B_i)=\sigma_B^2$, $\mathrm{Corr}(A_i,B_i) = \rho$. For the difference of means $\hat\Delta = \bar B - \bar A$,
> $$\mathrm{Var}(\hat\Delta) = \frac{\sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B}{n} .$$
> If instead $A$ and $B$ are evaluated on *independent* samples of size $n$ each, the variance is $(\sigma_A^2+\sigma_B^2)/n$. The ratio of standard errors is $\sqrt{1 - 2\rho\sigma_A\sigma_B/(\sigma_A^2+\sigma_B^2)}$, which for $\sigma_A=\sigma_B$ is $\sqrt{1-\rho}$.

*Proof.* Bilinearity of covariance: $\mathrm{Var}(B_i - A_i) = \sigma_A^2+\sigma_B^2 - 2\mathrm{Cov}(A_i,B_i)$, then divide by $n$ for the mean of i.i.d. differences. The unpaired case has $\mathrm{Cov}=0$ by independence. $\blacksquare$

Read the consequence: with $\rho = 0.75$ — utterly routine for two models evaluated on the same items, since item difficulty is shared — the paired standard error is half the unpaired one, equivalent to a **four-fold increase in test-set size, obtained for free by not throwing away the pairing**. Two models compared on the same test set, with per-item predictions retained, is one of the few genuinely free lunches in statistics. Conversely, comparing published accuracies from two papers that used different splits forfeits it.

> **Definition 9.43 (McNemar's test).** For two classifiers on the same $n$ test items, tabulate correctness:
>
> |  | B correct | B wrong |
> |---|---|---|
> | **A correct** | $n_{11}$ | $b$ |
> | **A wrong** | $c$ | $n_{00}$ |
>
> McNemar's test conditions on the number of *discordant* pairs $b+c$ and tests $H_0 : \mathbb{P}(\text{A right, B wrong}) = \mathbb{P}(\text{A wrong, B right})$ using $b \mid (b+c) \sim \mathrm{Bin}(b+c, 1/2)$. The exact $p$-value is the two-sided binomial tail; the asymptotic version uses $\chi^2 = (b-c)^2/(b+c)$ against $\chi^2_1$, or $(|b-c|-1)^2/(b+c)$ with continuity correction.

> **Proposition 9.44 (McNemar is exact conditionally).** Under $H_0$ as stated, and conditionally on $b + c = m$, the count $b$ is $\mathrm{Bin}(m,1/2)$, so the exact two-sided test has size $\le\alpha$ for every $m$ and every distribution of the concordant cells.

*Proof.* The four cell counts are multinomial with probabilities $(\pi_{11},\pi_{10},\pi_{01},\pi_{00})$. Conditioning a multinomial on $b+c=m$ makes $b$ binomial with $m$ trials and success probability $\pi_{10}/(\pi_{10}+\pi_{01})$, which equals $1/2$ exactly under $H_0$. The concordant cells drop out of the conditional law entirely. Super-uniformity of the exact binomial $p$-value then follows from Proposition 9.18(i). $\blacksquare$

The conditioning is the point: the items both models get right, and the items both get wrong, carry no information about which model is better. Only the disagreements do. That is why McNemar can be significant when a two-proportion test is not — and it is the same mechanism as Proposition 9.42, since discordance is what drives $\mathrm{Var}(B_i-A_i)$.

**Worked Example 9.22 (the comparison, done three ways).** I simulated a realistic paired comparison: $n = 2000$ test items with a latent per-item difficulty, two classifiers with true accuracies $0.90$ and $0.91$ that agree on 95.6% of items (correctness correlation $\phi = 0.7454$). Seed 8675309. Observed: $\mathrm{acc}_A = 0.9000$, $\mathrm{acc}_B = 0.9100$, contingency $n_{11}=1766$, $b = 34$ (A right, B wrong), $c = 54$ (A wrong, B right), $n_{00}=146$.

| analysis | interval for $\mathrm{acc}_B - \mathrm{acc}_A$ | width | $p$-value |
|---|---|---|---|
| unpaired two-proportion $z$ | $[-0.00817,\ 0.02817]$ | 0.03634 | 0.2807 |
| paired (per-item differences) | $[\ 0.00082,\ 0.01918]$ | 0.01837 | 0.0329 |
| paired bootstrap, $B = 20{,}000$ | $[\ 0.00100,\ 0.01900]$ | 0.01800 | — |
| McNemar, exact binomial | — | — | 0.0422 |
| McNemar, $\chi^2$ without correction | — | — | 0.0330 |

**Same data, same models, and the conclusion flips.** The unpaired analysis reports $p = 0.28$ and an interval comfortably containing zero: "no significant difference". The paired analyses report $p \approx 0.033$–$0.042$ and an interval excluding zero. The unpaired analysis is not *wrong* — it is a valid test — it is *wasteful*, and the waste is quantifiable: the standard-error ratio is $1.978$, so the unpaired analysis has the power of a test-set $1.978^2 = 3.91$ times smaller. To reach 80% power for a 1-point gap you would need $13{,}493$ items unpaired versus $3{,}448$ paired. If you have per-item predictions and you run an unpaired test, you are discarding three-quarters of your evaluation data.

Two cautions. The paired $z$-interval and McNemar's exact test disagree slightly ($0.033$ vs $0.042$) because the exact test is conservative on discrete data, exactly as Worked Example 9.12 predicted; report the exact one if you want a guarantee. And note how close this is to the threshold: a comparison whose verdict depends on the continuity correction is a comparison that should be reported as "suggestive, $p\approx0.04$, interval $[0.001, 0.019]$", not as a discovery.

### 9.9.2 Cross-validation variance, and what a fold is not

Cross-validation gives $K$ numbers, and it is irresistible to treat them as $K$ independent observations and run a $t$-test with $K-1$ degrees of freedom. The folds are not independent: their training sets overlap in $(K-2)/(K-1)$ of the data, so the $K$ fitted models are highly correlated.

> **Proposition 9.45 (variance of a $K$-fold average).** Let $D_1,\dots,D_K$ be the per-fold score differences, with common variance $\tau^2$ and common pairwise correlation $\varrho$. Then
> $$\mathrm{Var}\Big(\frac1K\sum_k D_k\Big) = \frac{\tau^2}{K}\big(1 + (K-1)\varrho\big),$$
> whereas the naive estimator $s^2/K$ (with $s^2$ the sample variance of the $D_k$) estimates $\frac{\tau^2(1-\varrho)}{K}$ in expectation. Hence the naive standard error understates the truth by the factor $\sqrt{(1+(K-1)\varrho)/(1-\varrho)}$ when $\varrho>0$.

*Proof.* $\mathrm{Var}(\sum_k D_k) = K\tau^2 + K(K-1)\varrho\tau^2$; divide by $K^2$. For the second claim, $\mathbb{E}[s^2] = \frac{1}{K-1}\mathbb{E}\sum_k(D_k-\bar D)^2 = \tau^2(1-\varrho)$ under the exchangeable correlation model. $\blacksquare$

So *in the exchangeable-correlation idealization*, the naive $t$-test is anti-conservative whenever $\varrho > 0$. Reality is messier, and I want to be honest about it rather than repeat the slogan.

**Worked Example 9.23 (measured Type I rates for CV-based tests).** I constructed a null that is true by symmetry: fit ridge regression on $N=200$ observations with three genuine features plus 15 pure-noise features for method A and 15 *different* pure-noise features for method B, so $\mathbb{E}[\text{error}_A] = \mathbb{E}[\text{error}_B]$ exactly by exchangeability. Then I ran three common protocols and measured how often each rejected at nominal $\alpha=0.05$, over $R=2000$ independent datasets (seed 7):

| protocol | measured Type I rate |
|---|---|
| 10-fold CV, paired $t$-test on the 10 fold differences | 0.0225 |
| 3× repeated 10-fold CV, $t$-test on the 30 values | 0.1220 |
| 30 random 90/10 splits, $t$-test on the 30 values | 0.1965 |

Also measured: the true standard deviation of the 10-fold CV difference across datasets was $0.05622$, while the mean naive standard error $s/\sqrt{10}$ from a single dataset was $0.06442$ — a ratio of $0.873$, meaning the naive SE *over*stated the variability here.

Proposition 9.45 tells us what that ratio implies. The naive-to-true ratio of standard errors is $\sqrt{(1-\varrho)/(1+(K-1)\varrho)}$, and here it is $0.06442/0.05622 = 1.1459$. Solving $\sqrt{(1-\varrho)/(1+9\varrho)} = 1.1459$ gives
$$\varrho \approx -0.0244 .$$
So in this design the fold differences are *negatively* correlated — slightly, but enough. That is not exotic: the folds partition a fixed dataset, so an unusually easy test fold leaves the remaining folds harder, which induces a small negative coupling that can outweigh the positive coupling from overlapping training sets. This is the whole content of the Bengio–Grandvalet result made concrete: **$\varrho$ is a property of the learner-and-design, its sign is not determined a priori, and you cannot estimate it from the single CV run you are testing with.** The conservative single-run result is therefore *predicted* by Proposition 9.45 once you know $\varrho<0$; what you cannot do is know $\varrho$ in advance.

Read that carefully, because it is more interesting than the slogan. In this design, the single-run 10-fold $t$-test was **conservative** (0.0225 against 0.05), not anti-conservative: the fold-to-fold spread happened to exceed the dataset-to-dataset spread of the CV average. But **repeated CV and repeated random splits were badly anti-conservative** (2.4× and 3.9× the nominal rate), because the extra "replicates" are re-uses of the same data and inflate the apparent degrees of freedom without adding information. Bengio and Grandvalet proved the general fact that explains both observations: **there is no unbiased estimator of the variance of $K$-fold cross-validation** from a single CV run, and the direction of the bias depends on the learner, the data, and $K$. So the practical rules are:

1. **Never** run a $t$-test on repeated-CV or repeated-subsample scores as if the replicates were independent. This is the common error and it is a large one.
2. If you must test with CV on one dataset, use a corrected procedure: Nadeau–Bengio's corrected resampled $t$-test, Dietterich's $5\times2$-cv test, or a conservative $K=2$ variant.
3. Better: if you have a genuine held-out test set, do the *paired item-level* analysis of §9.9.1, where exchangeability of items gives an exact permutation test and the variance question does not arise.
4. Report the fold scores individually. A reader can see the spread; a reader cannot see it in "$92.1 \pm 0.3$" with an unspecified denominator.

### 9.9.3 Seeds, and how many you need

Training randomness — initialization, data order, dropout masks, nondeterministic kernels — is a variance source separate from test-set sampling, and it does not shrink when you enlarge the test set. If you report one run per configuration, you cannot separate the two.

> **Proposition 9.46 (seed budget).** Suppose training with seed $s$ yields test accuracy $Y_s$ with $\mathrm{Var}(Y_s) = \sigma_{\text{seed}}^2$, i.i.d. across seeds, and you compare two configurations with $k$ seeds each. To reject $H_0:\Delta = 0$ at two-sided level $\alpha$ with power $1-\beta$ against a true gap $\Delta$ requires
> $$k \ \ge\ \frac{2(z_{1-\alpha/2}+z_{1-\beta})^2\sigma^2_{\text{seed}}}{\Delta^2}.$$

*Proof.* $\hat\Delta = \bar Y^{(2)} - \bar Y^{(1)}$ has variance $2\sigma^2_{\text{seed}}/k$; substitute into the sample-size formula of Worked Example 9.4 with $\delta = \Delta/\sqrt{2\sigma^2_{\text{seed}}/k}$ and solve. $\blacksquare$

**Worked Example 9.24 (seed budgets, computed).** With $(z_{0.975}+z_{0.80})^2 = 7.8489$:

| seed SD | target gap $\Delta$ | seeds per arm (exact) | seeds per arm (rounded up) |
|---|---|---|---|
| 0.4 pts | 0.5 pts | 10.05 | 11 |
| 0.4 pts | 0.2 pts | 62.79 | 63 |
| 0.4 pts | 1.0 pts | 2.51 | 3 |
| 0.2 pts | 0.2 pts | 15.70 | 16 |
| 0.8 pts | 0.5 pts | 40.19 | 41 |
| 0.8 pts | 0.2 pts | 251.16 | 252 |

Look at the last row: with a seed standard deviation of 0.8 accuracy points — not unusual for fine-tuning a mid-sized model on a small dataset — detecting a 0.2-point improvement needs **252 training runs per arm**. Nobody does this, and that is precisely why 0.2-point improvements in the literature are not credible. The constructive response is not to run 252 seeds; it is to (i) report $\sigma_{\text{seed}}$ so readers can compute this themselves, (ii) refuse to claim gaps smaller than what your seed count can resolve, and (iii) reduce $\sigma_{\text{seed}}$ where possible (longer training, averaging checkpoints, fixing data order) since it enters quadratically in your budget.

### 9.9.4 Intervals for the metrics you actually report

**Accuracy** on $n$ i.i.d. items: binomial, so use Wilson (Proposition 9.26). At $\hat p = 0.95$, $n=1000$: Wald gives $\pm0.0135$; Wilson gives $[0.9346, 0.9628]$, slightly asymmetric, which is correct.

**AUC**: use the Mann–Whitney identity.

> **Proposition 9.47 (AUC as a probability).** With continuous scores, $\mathrm{AUC} = \mathbb{P}(S_+ > S_-)$ for independent $S_+\sim P_1$, $S_-\sim P_0$; the empirical AUC equals the normalized Mann–Whitney $U$ statistic $\frac{1}{n_+n_-}\sum_{i,j}\mathbf{1}\{s_i^+ > s_j^-\}$.

*Proof.* $\mathrm{AUC} = \int_0^1 \mathrm{TPR}\,d\mathrm{FPR}$; substituting $\mathrm{FPR} = 1-F_0(t)$, $\mathrm{TPR}=1-F_1(t)$ and changing variables gives $\int (1-F_1(t))\,dF_0(t) = \mathbb{P}(S_+ > S_-)$ by Fubini. The empirical version is the plug-in with $\hat F_0,\hat F_1$. $\blacksquare$

Because AUC is a two-sample $U$-statistic, its variance is *not* $\mathrm{AUC}(1-\mathrm{AUC})/n$ — the summands are dependent. Use the Hanley–McNeil or DeLong formula, or bootstrap stratified by class. Example ($n_+ = 200$, $n_-=800$, Gaussian scores separated by $\Delta=1$, seed 4): empirical AUC $0.7762$ (population value $\Phi(1/\sqrt2) = 0.7602$); Hanley–McNeil SE $0.02049$, interval $[0.7361, 0.8164]$; stratified bootstrap SE $0.01824$, percentile interval $[0.7396, 0.8114]$. The two agree to within a Monte Carlo error, which is the reassurance you want before trusting either.

**F1** is where people go wrong most often, because $F_1 = 2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$ *looks* like a proportion and is not one: it is a ratio of dependent counts, so no binomial interval applies. Treating it as binomial understates the uncertainty substantially. Example ($n = 2000$, positive rate $0.20$, seed 1234): $\mathrm{TP}=271$, $\mathrm{FP}=414$, $\mathrm{FN}=119$, giving precision $0.3956$, recall $0.6949$, $F_1 = 0.5042$.

| method | 95% interval for $F_1$ | width |
|---|---|---|
| naive "binomial on $F_1$", $\sqrt{F_1(1-F_1)/n}$ | $[0.4823,\ 0.5261]$ | 0.0438 |
| nonparametric bootstrap, $B = 20{,}000$ | $[0.4673,\ 0.5399]$ | 0.0727 |

The naive interval is **40% too narrow**. Use the bootstrap (or the delta method of Ch. 7 applied to the three counts jointly); and state whether you are reporting macro-$F_1$ (average of per-class $F_1$) or micro-$F_1$ ($F_1$ of pooled counts), because they are different estimands and averaging per-fold $F_1$ values is a third thing again.

**When the items are not i.i.d.** Every interval in this section assumes test items are independent draws. Benchmarks routinely violate this: near-duplicates, templated questions, multiple items per document or per image, train/test contamination. The failure is quantitative and severe. I simulated a benchmark of $2000$ items arranged as $400$ clusters of $5$, where within each cluster 90% of items are near-copies (seed 99, $4000$ replications, true gap $0.01$):

| variance estimate | coverage | mean width |
|---|---|---|
| item-level (treats all 2000 as independent) | 0.6645 | 0.0356 |
| cluster-level (resamples the 400 clusters) | 0.9563 | 0.0746 |

A nominal 95% interval covering 66% of the time is not a small problem, and the fix is mechanical: **resample the clusters, not the items.** The effective sample size here is nearer $400$ than $2000$, and the width ratio $2.095 \approx \sqrt{2000/400}\cdot\text{(partial-duplication correction)}$ reflects that. If your benchmark has any grouping structure, the unit of resampling is the group.

**Multiplicity across metrics.** The multiplicity ML papers actually commit is rarely "we tested 1000 hypotheses" — it is "we reported accuracy, macro-$F_1$, and AUC, and one of them was significant." Three correlated metrics tested at $0.05$ do not give you a $0.05$ error rate for the claim "B is better." Either designate one primary metric before you look (the clean solution, and what §9.9.3's protocol requires), or apply Holm across the metrics you report and say you did.

**Practical significance.** Report the effect size and its interval, in units a reader cares about, alongside any $p$-value.

> **Common Confusion 9.9.** *"$p = 0.001$ on 500,000 test items, so this is a strong result."*
>
> $p$ measures incompatibility with the null, and it shrinks with $n$ for any non-zero effect however tiny. Make that concrete with a stated model. Take $n = 500{,}000$ items, $\mathrm{acc}_A = 0.9000$, $\mathrm{acc}_B = 0.9015$ — a gap of $0.15$ points. Unpaired, $\mathrm{se} = \sqrt{p_A(1-p_A)/n + p_B(1-p_B)/n} = 5.98\times10^{-4}$, so $z = 2.51$, $p = 0.012$. Paired, with the two models disagreeing on 10% of items, $\mathrm{se} = \sqrt{0.10/n} = 4.47\times10^{-4}$, so $z = 3.35$, $p = 0.0008$; and if they disagree on only 4% of items, $z = 5.30$ and $p = 10^{-7}$. All three are the *same* $0.15$-point gap. The $p$-value moved by four orders of magnitude purely because of $n$ and the pairing structure — and a $0.15$-point gap may still be entirely irrelevant to your application, smaller than your seed-to-seed variability, and smaller than the label noise in the benchmark. The size of a $p$-value is a statement about sample size at least as much as about effect size. Report the estimated gap with its interval: "$+0.15$ points, 95% CI $[0.06, 0.24]$" tells a reader both that the effect is real and that it is small, which is the honest joint claim. Conversely, a large $p$ with a wide interval means "we learned nothing", and a large $p$ with a narrow interval tightly around zero means "we established there is no meaningful effect" — a genuinely useful result that the $p$-value alone cannot express.

**Preregistration and holdout hygiene, concretely.** Write down, before running the final evaluation: the metric, the test, the correction for multiplicity, the number of seeds, and the effect size you would call meaningful. Keep a query counter on the test set. Select on validation; report on test; if you touch test more than a handful of times, say so and treat the numbers as validation numbers. None of this requires new statistics — it requires only that the assumptions behind Proposition 9.18 be true rather than aspirational.

> **Self-check 9.9.**
> 1. Two models are compared on the same 2000 items with correctness correlation $0.75$. By what factor does pairing reduce the standard error of the accuracy difference?
> 2. Which cells of the McNemar table are informative, and why do the others drop out?
> 3. Why is a $t$-test on 30 repeated-CV scores worse than a $t$-test on 10 single-run fold scores?
> 4. You have a seed SD of $0.4$ points and 5 seeds per arm. What is the smallest gap you can detect at 80% power?
> 5. Why can you not put a binomial interval on $F_1$, and what goes wrong numerically if you do?
> 6. Your benchmark contains 500 documents, each contributing 4 questions. What is your effective sample size, and what should you resample?

## 9.10 Results used but not proved

Everything in the list below was used somewhere above with an explicit flag. None of it is proved here; all of it is standard, and the pointers are to places where the proof is written out properly.

| Result | Used in | Where to find a proof |
|---|---|---|
| Law of the iterated logarithm | Proposition 9.21 (continuous monitoring rejects almost surely) | Durrett, *Probability: Theory and Examples*, Thm 8.5.2; Hartman–Wintner original |
| Dvoretzky–Kiefer–Wolfowitz inequality with Massart's constant | Proposition 9.28 (simultaneous CDF band) | Massart (1990), *Ann. Probab.* 18:1269–1283; Boucheron–Lugosi–Massart, *Concentration Inequalities*, §11 |
| Glivenko–Cantelli theorem | §9.8.1 (why the empirical measure is a usable stand-in for $P$) | van der Vaart, *Asymptotic Statistics*, Thm 19.1 |
| Donsker's theorem / empirical process weak convergence | §9.8.1, §9.8.3 (bootstrap for smooth functionals) | van der Vaart, Ch. 19; Shorack–Wellner, *Empirical Processes* |
| Berry–Esseen theorem | Theorem 9.38 (rate of bootstrap approximation for the mean) | Durrett, Thm 3.4.17; Petrov, *Sums of Independent Random Variables* |
| MLE consistency, asymptotic normality, and local asymptotic normality | Theorem 9.15 (Wilks), Proposition 9.16 (asymptotic equivalence of the three statistics) | Ch. 8 of these notes for the statement; van der Vaart, Chs. 5, 7 for proofs |
| Bootstrap consistency for Hadamard-differentiable functionals (functional delta method) | §9.8.3 (why the bootstrap works for quantiles, correlations, AUC) | van der Vaart, Thm 23.9; Wellner–van der Vaart, *Weak Convergence and Empirical Processes*, §3.9 |
| Chi-bar-squared limits for likelihood ratios on the boundary of the parameter space | §9.3.1 (caveat on Wilks when the null sits on a boundary) | Self & Liang (1987), *JASA* 82:605–610; Silvapulle–Sen, *Constrained Statistical Inference* |
| Benjamini–Yekutieli theorem and FDR control under PRDS dependence | §9.6.3 (what survives when $p$-values are correlated) | Benjamini & Yekutieli (2001), *Ann. Statist.* 29:1165–1188 |
| Fisher's method: $-2\sum\log p_i \sim \chi^2_{2m}$ under independence | Common Confusion 9.5 (why you may not multiply $p$-values) | Fisher, *Statistical Methods for Research Workers*, §21.1; or derive it from $-\log U \sim \mathrm{Exp}(1)$ |
| Hanley–McNeil / DeLong variance formula for the empirical AUC | §9.9.4 (intervals for AUC) | Hanley & McNeil (1982), *Radiology* 143:29–36; DeLong et al. (1988), *Biometrics* 44:837–845 |
| Non-existence of an unbiased variance estimator for $K$-fold cross-validation | §9.9.2, Worked Example 9.23 | Bengio & Grandvalet (2004), *JMLR* 5:1089–1105 |
| Bartlett correction: the $O(n^{-1})$ term in the Wilks approximation | Worked Example 9.11 (why the measured size was 0.0536, not 0.0500) | Bartlett (1937); Barndorff-Nielsen & Cox, *Inference and Asymptotics*, Ch. 6 |
| Wilcoxon rank-sum null distribution | Worked Example 9.19 (comparison with the permutation test) | Lehmann, *Nonparametrics*, Ch. 1; or derive by enumeration as in §9.7 |
| Ville's inequality and the validity of confidence sequences / $e$-processes | §9.4.3 (anytime-valid inference as the fix for optional stopping) | Ville (1939); Howard, Ramdas, McAuliffe & Sekhon (2021), *Ann. Statist.* 49:1055–1080; Ramdas, Grünwald, Vovk & Shafer (2023), *Statist. Sci.* 38:576–601 |
| Validity of group-sequential $\alpha$-spending boundaries (Pocock, O'Brien–Fleming, Lan–DeMets) | §9.4.3 and the solution to Lab B2 | Pocock (1977), *Biometrika* 64:191–199; O'Brien & Fleming (1979), *Biometrics* 35:549–556; Jennison & Turnbull, *Group Sequential Methods with Applications to Clinical Trials* |
| Edgeworth expansions; second-order accuracy of studentized and BCa intervals | §9.8.2 ("removes the leading skewness term"), Lab B4 | Hall, *The Bootstrap and Edgeworth Expansion*, Chs. 2–3 |
| Consistency of subsampling and the $m$-out-of-$n$ bootstrap for non-regular problems | §9.8.3 (boundary parameters), Solution A10 (degenerate delta method) | Politis, Romano & Wolf, *Subsampling* (1999), Chs. 2–3 |
| PRDS holds for one-sided tests on positively equicorrelated Gaussian statistics (and can fail for two-sided) | §9.6.3 and Solution C2 (when BH's guarantee survives correlation) | Benjamini & Yekutieli (2001), *Ann. Statist.* 29:1165–1188, §3 |

## 9.11 Chapter summary

1. **A test is a decision rule, and a hypothesis is a set of distributions.** Everything else — size, power, $p$-values, confidence sets — is bookkeeping on top of Definitions 9.1–9.3. Once you can say precisely which set of distributions $H_0$ names, most confusion evaporates.

2. **Power is bounded by how distinguishable the two hypotheses are.** Proposition 9.5 says no test at level $\alpha$ can have power exceeding $\alpha + d_{\mathrm{TV}}(P_0,P_1)$. If the distributions are close in total variation, no cleverness in the test statistic will save you; only more data will.

3. **The likelihood ratio is optimal, and it is the only thing that is.** The Neyman–Pearson lemma (Theorem 9.7) says the most powerful level-$\alpha$ test between two simple hypotheses thresholds $d P_1/dP_0$ — a Radon–Nikodym derivative, which is why the measure theory was worth building. Karlin–Rubin (Theorem 9.10) extends this to one-sided composite nulls whenever the family has monotone likelihood ratio.

4. **Randomization is what makes exact levels attainable on discrete data.** Worked Example 9.2 showed a coin experiment where no non-randomized test has size exactly $0.05$. Real practice accepts conservatism instead, which is why exact discrete tests report $p$-values like $0.042$ where an asymptotic test reports $0.033$.

5. **The three classical statistics agree asymptotically and disagree in practice.** Wald, score, and likelihood ratio all converge to the same $\chi^2_k$ limit (Proposition 9.16), but the Wald statistic is **not** parametrization-invariant: Worked Example 9.9 gave $p = 0.0098$ on the probability scale and $p = 0.0334$ on the logit scale for identical data. The likelihood-ratio and score statistics are invariant; prefer them.

6. **A $p$-value is super-uniform under the null, and that is its entire guarantee.** Proposition 9.18 makes this precise, and Proposition 9.19 shows $p$-values and level-$\alpha$ tests are two descriptions of one object. It is not $\mathbb{P}(H_0 \mid \text{data})$; Common Confusion 9.4 and the PPV table of Worked Example 9.13 show how far apart those can be.

7. **Optional stopping destroys the guarantee, quantitatively.** Measured: $0.0497$ at one look, $0.1914$ at ten looks, $0.3724$ at a hundred, and by Proposition 9.21 the rejection probability tends to $1$ under continuous monitoring — a law-of-the-iterated-logarithm consequence, not a small-sample artifact.

8. **Confidence sets and tests are the same mathematics read in two directions.** Theorem 9.23 (duality) says inverting a family of level-$\alpha$ tests gives a $1-\alpha$ confidence set and conversely. Every interval in this chapter was constructed by inversion, which is why the Wilson interval (Proposition 9.26) is derived rather than asserted.

9. **The Wald interval for a proportion is genuinely bad, not merely inelegant.** At $n=40$, $p=0.01$ its exact coverage is $0.331$; at $n=20$, $x=1$ it returns $[-0.046, 0.146]$, which contains impossible values. Wilson, likelihood-ratio, or Clopper–Pearson cost nothing extra.

10. **Simultaneous statements need simultaneous methods.** A pointwise band covers each parameter with probability $1-\alpha$; a DKW band (Proposition 9.28) covers the whole curve. Half-widths $0.136$, $0.043$, $0.014$ at $n = 100$, $1000$, $10{,}000$ — the price of simultaneity is a constant factor, not an order of magnitude.

11. **Bonferroni, Holm, and Benjamini–Hochberg control different things, and the choice is a decision about what a false claim costs you.** Bonferroni and Holm bound FWER; Holm dominates Bonferroni uniformly and there is no reason ever to prefer Bonferroni on power grounds. BH bounds FDR at $q$ (Theorem 9.32) and is much more powerful — measured power $0.4954$ versus $0.1461$ on $m=1000$ tests — but its FWER was $0.9030$ in the same simulation. Neither is "better"; they answer different questions.

12. **Selection inflates the selected estimate, and the inflation is computable.** The winner's curse (Proposition 9.33) cost $+1.22$, $+1.77$, $+2.15$, $+2.53$ accuracy points for leaderboards of $10$, $50$, $200$, $1000$ entries. An adaptive holdout reused enough times reported $0.786$ accuracy for a model whose true accuracy was $0.503$.

13. **Permutation tests are exact for finite samples whenever the null implies exchangeability** (Theorem 9.35), and remain valid under Monte Carlo sampling if you use the $(1+\#\{T^*\ge T\})/(1+B)$ convention (Proposition 9.36). Exact enumeration of a 5-vs-5 comparison gave $p = 3/252 = 0.0119$; under heavy tails the permutation test held its size ($0.048$) while the $t$-test was conservative ($0.038$).

14. **The bootstrap resamples the empirical measure, and it fails exactly when the functional is not smooth or the variance is not finite.** It works for means (studentized coverage $0.94$), and it fails for the maximum (percentile coverage $0.000$, because the resampled max has an atom of size $1-(1-1/n)^n \to 1-e^{-1}$), for Cauchy-like tails (coverage stuck near $0.91$ with non-shrinking width), and for dependent data unless you resample blocks ($0.469$ i.i.d. versus $0.841$ block).

15. **Pair your comparisons.** Proposition 9.42 and Worked Example 9.22: the same 2000-item comparison gave $p = 0.28$ unpaired and $p = 0.033$ paired, a standard-error ratio of $1.978$, equivalent to throwing away three-quarters of the test set. McNemar's test (Proposition 9.44) is exact conditionally on the discordant count.

16. **Use the right interval for the metric, resample the right unit, and report what your budget can resolve.** Accuracy is binomial (use Wilson); AUC is a $U$-statistic (use Hanley–McNeil or a stratified bootstrap); $F_1$ is a ratio of dependent counts, and the naive binomial interval on it was 40% too narrow. If items cluster — near-duplicates, several questions per document — resample clusters: item-level resampling on a clustered benchmark gave $0.665$ coverage for a nominal $0.95$ interval. And report effect sizes with intervals, count your seeds, and say how many times you touched the test set: with a seed SD of $0.8$ points, resolving a $0.2$-point gap at 80% power needs $252$ runs per arm (Worked Example 9.24), and a claim that cannot survive that arithmetic should be stated as a conjecture.

### Callbacks

Several promises from earlier chapters are discharged here. The union bound (Proposition 1.8(2)) reappears wearing a different hat as the Bonferroni correction — the entire proof of Proposition 9.30 is one application of it, and Holm's improvement comes from applying it to a shrinking family rather than a fixed one. Conditional probability and Bayes' rule (Definition 1.10, Proposition 1.12) return as the positive predictive value computation of Worked Example 9.13, which is the disease-test arithmetic of Chapter 1 relabelled with "significant result" in place of "positive test"; the prosecutor's fallacy of Common Confusion 9.4 is exactly the confusion of $\mathbb{P}(A\mid B)$ with $\mathbb{P}(B\mid A)$ that Chapter 1 warned about. The Radon–Nikodym derivative and absolute continuity (Ch. 3) are what make the likelihood ratio in Neyman–Pearson a well-defined object rather than a ratio of densities that might not exist. The CDF and its properties (Definition 1.18, Proposition 1.19) underwrite the probability-integral transform behind $p$-value uniformity, and the empirical measure introduced with the law of large numbers (Ch. 5) is literally the distribution the bootstrap samples from. Convergence in distribution and the continuous mapping and delta-method machinery (Chs. 6–7) are what turn the MLE asymptotics of Chapter 8 into Wilks' theorem here. Measurability is not decoration: the simultaneous band of Proposition 9.28 requires that $\sup_x |\hat F_n(x) - F(x)|$ be a random variable at all, which needs the separability argument that right-continuity supplies, and the uncountable intersection defining a confidence band is handled by reducing it to a supremum over a countable dense set — precisely the technique introduced for suprema of processes in Chapter 4.

### Before you continue

1. Write down the definition of a level-$\alpha$ test without looking. Now state, in one sentence each, what "size", "level", and "power" mean, and explain why the distinction between size and level exists at all.
2. Two distributions have $d_{\mathrm{TV}}(P_0,P_1) = 0.03$. What is the largest power any level-$0.05$ test can have? What does that tell you about a paper claiming to distinguish them reliably?
3. Explain why the Wald statistic is not parametrization-invariant while the likelihood-ratio statistic is. Which quantity in the Wald construction depends on the parametrization, and why does the LR construction avoid it?
4. Someone reports $p = 0.03$ from an experiment where they checked significance after every 50 observations and stopped when it crossed the threshold. What, precisely, is wrong — and which of the assumptions in Proposition 9.18 fails?
5. You have $m = 500$ hypotheses. Describe when you would use Holm and when you would use BH, in terms of what a false claim costs. Is there any circumstance in which you would use Bonferroni rather than Holm?
6. Explain in one paragraph, without formulas, why the bootstrap works for the sample mean and fails for the sample maximum.
7. You are given two models' per-item predictions on a shared test set of 5000 items. Sketch the analysis you would run, and say what you would report.
8. What is the difference between "$p > 0.05$, so there is no effect" and "the 95% interval is $[-0.002, 0.003]$, so any effect is negligible"? Which is a scientific conclusion?

### Looking ahead to Chapter 10

Everything in this chapter treated the parameter as fixed and unknown, and quantified uncertainty through the sampling distribution of the data. Chapter 10 inverts that: **Bayesian inference** puts a probability distribution on the parameter itself, and the object of interest becomes the posterior $\pi(\theta \mid x) \propto \pi(\theta)\,p(x\mid\theta)$ — which is just Bayes' rule (Proposition 1.12) with a density in place of a partition. We will work through conjugate families (beta–binomial, normal–normal, gamma–Poisson) where the posterior is available in closed form, and see why conjugacy is a computational convenience rather than a modelling principle; predictive distributions, which answer the question "what will the next observation look like" that frequentist point estimation leaves awkwardly unaddressed; model comparison via the marginal likelihood and Bayes factors, including why marginal likelihoods are far more prior-sensitive than posteriors and what that means for anyone using them to select architectures; hierarchical models and the shrinkage they induce, which is where the multiple-comparisons problem of §9.6 gets a completely different and rather beautiful treatment; Bayesian linear regression, whose posterior mean is ridge regression and whose posterior variance is the thing practitioners actually want when they ask a model for its uncertainty; and Bayesian decision theory, which finally makes explicit the loss function that has been implicit in every "choose $\alpha = 0.05$" decision in this chapter. The credible interval of Common Confusion 9.6 will be defined properly there, and we will be able to say exactly when it does and does not numerically coincide with a confidence interval.

# Exercises

Difficulty: ★ routine consolidation, ★★ requires an idea, ★★★ genuinely demanding.

## Part A — Theory

**A1 (★).** Let $X \sim \mathrm{Bin}(10, \theta)$, $H_0: \theta = 1/2$ against $H_1: \theta > 1/2$. (a) Find the smallest rejection region of the form $\{X \ge c\}$ with size at most $0.05$, and report its exact size. (b) Find the randomized test with size exactly $0.05$. (c) Compute the power of both at $\theta = 0.8$.

**A2 (★).** Prove that size $\le$ level, that a test of level $\alpha$ is also of level $\alpha'$ for every $\alpha' > \alpha$, and that the power function of the test in A1(a) is strictly increasing in $\theta$ on $(0,1)$.

**A3 (★★).** Let $P_0 = N(0,1)$ and $P_1 = N(\mu, 1)$ with $\mu > 0$. (a) Show $d_{\mathrm{TV}}(P_0,P_1) = 2\Phi(\mu/2) - 1$. (b) Verify that the level-$\alpha$ $z$-test attains the ceiling of Proposition 9.5 only when $\alpha$ equals a specific value depending on $\mu$; identify it and interpret. (c) For $\mu = 0.1$, what is the maximum power of any level-$0.05$ test on a single observation?

**A4 (★★).** Show that the one-parameter exponential family $p_\theta(x) = h(x)\exp(\eta(\theta)T(x) - A(\theta))$ with $\eta$ strictly increasing has monotone likelihood ratio in $T(X)$, and deduce the UMP one-sided test for $H_0:\theta\le\theta_0$. Then exhibit a family with MLR that is *not* an exponential family.

**A5 (★★).** Prove that if $p$ is a valid $p$-value (super-uniform under every $P\in H_0$) then so is $\min(1, 2\min(p, 1-p))$ under the additional assumption that $p$ is exactly uniform under $H_0$, but give a counterexample showing the claim can fail when $p$ is merely super-uniform.

**A6 (★★).** Let $X_1,\dots,X_n \overset{\text{iid}}{\sim} \mathrm{Unif}[0,\theta]$. (a) Find an exact pivot and invert it to get an exact $1-\alpha$ confidence interval for $\theta$. (b) Compare its expected length with the interval obtained by inverting the likelihood-ratio test. (c) Explain why no Wald interval based on $\sqrt n(\hat\theta_{\text{MLE}} - \theta)$ can be correct here.

**A7 (★★).** Show that Holm's procedure rejects every hypothesis that Bonferroni rejects, at the same FWER level, and construct a configuration of $p$-values for which Holm rejects strictly more. Then determine the largest $p$-value BH can ever reject at level $q$, show that BH can reject a hypothesis whose $p$-value exceeds the Bonferroni threshold $q/m$ by a factor as large as $m$, and explain why this is not a contradiction of anything Bonferroni guarantees.

**A8 (★★★).** Prove that the Benjamini–Hochberg procedure with the Benjamini–Yekutieli correction — using thresholds $\frac{i\,q}{m H_m}$ with $H_m = \sum_{j\le m} 1/j$ — controls FDR at $q$ under *arbitrary* dependence. You may use the fact that BH controls FDR at $q\,m_0/m$ under independence, and the union-bound-style argument over the possible values of the number of rejections.

**A9 (★★).** Let $T$ be a test statistic and $G$ a finite group of transformations under which the data are exchangeable under $H_0$, $|G| = M$. Prove that the randomization $p$-value $\hat p = \frac{1}{M}\#\{g \in G : T(gX) \ge T(X)\}$ satisfies $\mathbb{P}_{H_0}(\hat p \le \alpha) \le \alpha$, and show that equality holds when $\alpha M$ is an integer and $T$ has no ties.

**A10 (★★★).** Let $\hat\theta = g(\bar X)$ for a twice continuously differentiable $g$ with $g'(\mu) \ne 0$. Prove that the bootstrap distribution of $\sqrt n (\hat\theta^* - \hat\theta)$ converges weakly, conditionally on the data and almost surely, to $N(0, g'(\mu)^2\sigma^2)$. Then explain precisely which step of your proof fails when $g'(\mu) = 0$, and what the correct limit is in that case.

## Part B — Computational labs

For each lab, fix and report your seed, report sample sizes, and state which numbers are Monte Carlo estimates and with what standard error.

**B1 (★) — $p$-value uniformity.** Simulate $B = 200{,}000$ two-sided $z$-tests under the null and plot the histogram of $p$-values; verify uniformity with a Kolmogorov–Smirnov statistic and report it. Repeat under alternatives $\delta \in \{0.1, 0.3, 0.5, 1.0\}$ and tabulate the deciles of the $p$-value distribution. Then repeat the null case with a *discrete* test statistic (binomial, $n = 20$) and explain the resulting picture.

**B2 (★★) — optional stopping.** Simulate a sequential experiment with $n$ growing to $N = 1000$, testing at $k$ equally spaced looks for $k \in \{1,2,5,10,20,50,100\}$ and stopping at the first $p < 0.05$. Estimate the Type I error for each $k$. Then implement continuous monitoring (test at every $n$) up to $N \in \{10, 100, 1000, 20{,}000\}$ and confirm the trend predicted by Proposition 9.21. Finally, implement an alpha-spending rule (Pocock or O'Brien–Fleming) and verify it restores control.

**B3 (★★) — Bonferroni vs Holm vs BH under dependence.** With $m = 1000$ tests, $m_0 = 900$ true nulls, and alternatives at $\mu = 3$, compare the three procedures on FWER, FDR, and power under equicorrelated test statistics with $\rho \in \{0, 0.5, 0.9\}$. Include an all-null row as a control. Discuss what happens to BH's FDR as $\rho$ grows and whether the theory predicts it.

**B4 (★★★) — bootstrap coverage: successes and failures.** Estimate the coverage of normal, percentile, basic, studentized, and BCa intervals in four settings: (i) the mean at $n = 20$ for three distributions of increasing skewness — $\chi^2_3$, $\mathrm{Exponential}(1)$, and $\mathrm{Lognormal}(0,1.5)$ — and rank the methods; (ii) the maximum of a $\mathrm{Unif}[0,1]$ sample, $n = 50$; (iii) the mean of a Cauchy and of a $t_{1.5}$ sample, with $t_5$ as a control, checking how the interval *width* behaves as $n$ grows; (iv) the mean of an AR(1) series with $\rho = 0.8$, comparing i.i.d. resampling with a block bootstrap. In each failure case, identify *which* hypothesis of Theorem 9.38 is violated, and — for (ii) — compute the atom in the resampling distribution analytically and confirm it numerically.

**B5 (★★★) — comparing two classifiers, correctly and incorrectly.** Generate two classifiers on a shared test set of $n = 2000$ items with a shared per-item difficulty, true accuracies $0.90$ and $0.91$. Compute: the unpaired two-proportion interval and test; the paired interval and test; the paired bootstrap interval; McNemar exact and asymptotic. Report the ratio of interval widths and convert it into an effective sample-size multiplier. Then vary the correctness correlation and plot the multiplier against it, comparing with Proposition 9.42. Finally, compute the sample size needed for 80% power under each analysis.

**B6 (★★) — cross-validation is not $K$ independent observations.** Construct a null that is true by symmetry (two learners differing only in which pure-noise features they receive). Over $R \ge 1000$ datasets, measure the Type I error of (a) a paired $t$-test on $10$ fold differences from a single 10-fold CV, (b) a $t$-test on $30$ values from $3\times$ repeated 10-fold CV, (c) a $t$-test on $30$ random 90/10 splits. Also estimate the true standard deviation of the CV difference across datasets and compare with the mean naive standard error. Report honestly which protocols are anti-conservative and which are not, and explain the pattern.

**B7 (★) — the intervals you will actually report.** On a simulated binary task with $n = 2000$ items and a 20% positive rate: (a) compute accuracy with a Wald and a Wilson interval and compare; (b) compute $F_1$, then compare the naive $\sqrt{F_1(1-F_1)/n}$ interval with a nonparametric bootstrap interval, and explain the discrepancy; (c) rebuild the test set as $400$ clusters of $5$ near-duplicate items, and compare a coverage estimate that resamples items with one that resamples clusters. State, for each of (a)–(c), which assumption the naive method violates.

## Part C — Paper-style derivations

**C1 (★★★).** State and prove the Neyman–Pearson lemma in full generality: for simple $H_0: P_0$ against $H_1: P_1$ dominated by a common $\sigma$-finite $\mu$, show that (i) a test of the threshold form with randomization on the boundary exists at every level $\alpha \in (0,1)$; (ii) any such test is most powerful at its size; (iii) any most powerful test agrees with the threshold form almost everywhere on $\{dP_1/dP_0 \ne k\}$. Handle the case where the likelihood ratio has atoms, and state where absolute continuity is and is not needed.

**C2 (★★★).** Prove that the Benjamini–Hochberg procedure controls FDR at level $q\,m_0/m$ when the $p$-values are independent. Then write a careful survey — with statements, not proofs — of what is known under dependence: PRDS, the Benjamini–Yekutieli logarithmic correction, arbitrary dependence, and the behaviour of BH under equicorrelated Gaussian statistics. Conclude with a practical recommendation and defend it.

**C3 (★★★).** Write a rigorous protocol for claiming "model A beats model B on benchmark $\mathcal{B}$". It must specify: the estimand; the randomness being averaged over; the test and its assumptions; the multiplicity correction; the number of seeds and how it was chosen; the effect size deemed meaningful and its justification; the holdout hygiene rules; and what will be reported regardless of outcome. Then apply your own protocol end-to-end to a synthetic example you construct, including the case where the protocol tells you *not* to make the claim, and report that outcome.

# Appendix A — Self-check answers

**Self-check 9.1.** (1) Yes, it is a level-$0.05$ test: level is any upper bound on the size, and $0.03 \le 0.05$. No, it is not a level-$0.01$ test, since $0.03 > 0.01$. Every test has exactly one size and infinitely many levels. (2) If the power function is monotone in $\mu$, then $\beta_\varphi(0.5) \le \beta_\varphi(1) = 0.9$; but Proposition 9.5 gives the *binding* bound: $\beta_\varphi(0.5) \le \alpha + \|P_{0.5}-P_0\|_{\mathrm{TV}} = 0.05 + 0.2 = 0.25$. So the largest value consistent with both is $0.25$ — the TV ceiling, not the monotonicity bound, is what constrains power against nearby alternatives. (3) A non-randomized test's size must be one of the finitely many attainable tail probabilities of a discrete statistic, and $\alpha$ need not be among them; a randomized test's size $\mathbb{E}_0[\varphi]$ interpolates continuously between attainable values (Worked Example 9.2). The auxiliary uniform is drawn independently of $X$ and its distribution does not depend on $\theta$, so it is ancillary: it can carry no information about $\theta$, only about which decision to announce. (4) The power of the experiment against the smallest gap that would matter (equivalently, the minimum detectable effect, or a confidence interval for the gap). "$p = 0.4$" from an underpowered comparison is compatible with a large real loss; "$p = 0.4$ and the 95% interval for the accuracy loss is $[-0.1, +0.2]$ points" is checkable.

**Self-check 9.2.** (1) Measurability of $\varphi$ and of $L$ is what makes $\mathbb{E}_\theta[\varphi]$ and the sets $\{L > k\}$, $\{L = k\}$ well defined at all. Countable additivity enters in part (a) of the proof: the right-continuity of $G(t) = \mathbb{P}_0(L > t)$, used to conclude $G(k) \le \alpha$, is continuity from above of $P_0$ (Proposition 1.8(4)), which is equivalent to countable additivity given finite additivity (Proposition 1.9). (2) Because $\log L$ is a strictly increasing affine function of the sufficient statistic $\sum_i X_i$ for every $\mu_1 > \mu_0$, so $\{L > k\}$ is always $\{\bar x > c\}$ and the cutoff $c$ is calibrated by the *null* alone. The responsible exponential-family property is monotone likelihood ratio in the sufficient statistic (Proposition 9.11): the alternative determines only the direction, not the region. (3) Test $H_0 : X \sim N(0,1)$ against $H_1 : X \sim N(0,4)$. The likelihood ratio is increasing in $|x|$, so the most powerful region is $\{|x| > c\}$ — two half-lines, not an interval. (A mixture alternative $\frac12 N(-1,1)+\frac12 N(1,1)$ works too.) (4) Proposition 9.13(ii) says the optimal frontier is concave and above the diagonal, so a dip is achievable-but-dominated: randomizing between the two operating points flanking the dip strictly improves it, and the honest summary curve is the concave hull. The most likely explanation is finite-sample noise in the empirical ROC — with few negatives at extreme thresholds, the empirical TPR/FPR staircase wobbles below the diagonal without the underlying score being genuinely worse than chance there.

**Self-check 9.3.** (1) The likelihood-ratio statistic requires fitting both the restricted and the unrestricted model; the score statistic requires only the restricted fit (and the Wald statistic only the unrestricted one). (2) The LR statistic compares the two maximized likelihood *values*, and the maximum of a function does not depend on how its domain is coordinatized; the Wald statistic measures a coordinate distance $\hat\psi - \psi_0$ against a variance estimate, and both the distance and the variance change under reparametrization — Worked Example 9.9 turned $p = 0.0098$ into $p = 0.0334$ by switching from the $p$ scale to the logit scale. (3) $k = 4$: fixing three parameters to specific values contributes three restrictions, and forcing two others to be equal is one further linear restriction. $k$ counts independent constraints — the drop in dimension from 12 to 8 — not the number of parameters mentioned. (4) Any two of: the likelihood is misspecified (so the sandwich variance, not $I^{-1}$, is correct and the limit is a weighted sum of $\chi^2_1$'s); parameters effectively on a boundary or non-identifiable (dead units, permutation symmetry among hidden units); parameter dimension comparable to $n$, violating fixed-$d$ asymptotics; $\hat\theta$ is not a global maximizer, so $U(\hat\theta) = 0$ at an interior *local* optimum is all you have.

**Self-check 9.4.** (1) For a continuous test statistic, exactly $0.05$ (Proposition 9.18(ii)). For a discrete one, at most $0.05$ and often far less — for the $\mathrm{Bin}(10,1/2)$ two-sided test of Worked Example 9.12 it is $0.0215$. (2) $\mathbb{P}(\text{data}\mid H_0)$ is computed *assuming* the null and describes the data's extremeness; $\mathbb{P}(H_0\mid\text{data})$ describes the hypothesis and requires, by Bayes' rule (Proposition 1.12), the extra ingredient of a prior probability for $H_0$. (3) Approximately $0.14$ — the measured rate for five equally spaced peeks was $0.1404$, nearly triple the nominal $0.05$. (4) A pile-up of $p$-values near $1$ has the mirror-image causes of a spike at $0$: conservative or discrete tests whose null $p$-values are super-uniform with atoms at high values; one-sided tests applied to effects pointing in the *wrong* direction (which push $p$ toward 1); or a misspecified null that makes the statistic under-dispersed (e.g., dependence that shrinks the effective variance). A flat histogram plus a spike at zero is the healthy picture; a bump near 1 says some of your nulls are not what you think they are.

**Self-check 9.5.** (1) "If this experiment were repeated many times and the interval recomputed each time, at least 95% of the intervals so constructed would contain the true value." The probability describes the *procedure*'s long-run behaviour; the realized interval either contains $\theta$ or does not, and no probability attaches to it without a prior. (2) Coverage $1 - 0.12 = 0.88$. By Theorem 9.23 the interval inherits exactly the test's error rate: an anti-conservative test yields an under-covering "95%" interval, and this inheritance is how you diagnose one from the other. (3) At $\hat p = 0$ the Wald interval is $0 \pm 1.96\sqrt{0 \cdot 1/n} = \{0\}$, a single point, so it misses every $p > 0$: the width is proportional to $\sqrt{\hat p(1-\hat p)}$ and collapses exactly when the estimate is least reliable. Wilson evaluates the standard error at the *hypothesized* $p_0$ rather than at $\hat p$; the resulting interval has strictly positive width even at $\hat p \in \{0,1\}$ and a centre shrunk toward $1/2$. (4) Because "the band contains $F$ everywhere" is the intersection over all $t$ of the pointwise coverage events, so its failure probability is the probability of a *union* of pointwise failures, which is much larger than any single one. Controlling the supremum costs the $\sqrt{\log(2/\alpha)}$ factor in the DKW half-width $\sqrt{\log(2/\alpha)/(2n)}$ — e.g. $\pm 0.0430$ at $n = 1000$ against $\pm 0.031$ for a single pointwise interval at $F(t)=0.5$.

**Self-check 9.6.** (1) Countable subadditivity — the union bound, Proposition 1.8(2). The whole proof of Proposition 9.30 is one application of it. (2) FWER and FDR measure different things about the same runs. In a typical replication BH made about 150 rejections, of which on average 4.6% were false — that is FDR $\approx 0.046$. But "at least one false rejection among ~150" happened in 90% of replications — that is FWER $\approx 0.903$. A small expected *proportion* of errors is fully compatible with a near-certain *presence* of at least one error. (3) Because Bonferroni is wasteful, not because Holm gets something for nothing. Both procedures are financed by the same union bound; Bonferroni spends $\alpha/m$ on every test forever, while Holm re-applies the bound to the shrinking set of not-yet-rejected candidates, which is all the accounting the FWER guarantee ever needed ($m - \ell + 1 \ge m_0$ in the proof of Theorem 9.31). Nothing is weakened; slack is recovered. (4) Close to zero. With $m = 100$ configurations on a 2000-item set the per-configuration standard error is about $0.8$ points, and Worked Example 9.18 puts the pure-selection bias of the winner among *equal* models at roughly $+2$ points for $m$ in this range — larger than the observed $1.5$-point gap. Absent independent evidence, the best guess for the fresh-data gap is approximately $0$, and the way to find out is to evaluate the single selected configuration once on a held-out test set.

**Self-check 9.7.** (1) That under $H_0$ the two models' scores are exchangeable *within each pair*: for every item $i$, $(A_i, B_i) \overset d= (B_i, A_i)$, equivalently the per-item difference $D_i = B_i - A_i$ has a distribution symmetric about $0$, independently across items. This is the sign-flip group ($2^n$ elements). It is weaker than any parametric assumption, but it is more than "equal mean scores" — a null under which $B$ is better on easy items and worse on hard ones by exactly offsetting amounts, with asymmetric $D_i$, is not covered. (2) Without the identity in both numerator and denominator, an extreme observed statistic can give $\hat p = 0$, and $\mathbb{P}(\hat p \le 0) = \mathbb{P}(\hat p = 0) > 0 = \alpha$ violates super-uniformity at $\alpha = 0$. Counting the identity guarantees $\hat p \ge 1/(B+1)$, and the exchangeability of the $B+1$ values $T(X), T(g_1X),\dots,T(g_BX)$ — of which $T(X)$ is one — is what the proof of Proposition 9.36 rests on. (3) The relabelling group has $\binom{8}{4} = 70$ elements, so the smallest attainable one-sided $p$-value is $1/70 \approx 0.0143$ and the smallest two-sided $p$-value is $2/70 \approx 0.0286$. Four runs per arm can never yield $p < 0.01$, whatever the effect. (4) Because items within a document share topic, phrasing, and difficulty, the full permutation group does not preserve the null distribution — exchangeability holds between documents, not between arbitrary items — so item-level permutation understates the null variability and inflates Type I error. Permute at the level at which exchangeability is credible: relabel (or sign-flip) whole documents, i.e. all 20 sentences of a document move together, giving a group of $2^{50}$ document-level flips.

**Self-check 9.8.** (1) A typical resample contains about $63\%$ of the original observations at least once: each observation is missed by all $n$ draws with probability $(1-1/n)^n \to e^{-1}$, so the expected fraction of distinct originals present is $1-(1-1/n)^n \to 1 - e^{-1} \approx 0.632$. The rest of the resample is duplicates — the bootstrap manufactures no new information. (2) The percentile interval (and BCa, which is a percentile interval with adjusted quantile levels) is equivariant under strictly increasing transformations, by Proposition 9.40. The best measured coverage on skewed data was the *studentized* interval — $0.9415$ for the exponential and $0.8660$ for the lognormal at $n=20$, against roughly $0.90$ and $0.76$ for percentile — because studentizing removes the leading skewness term of the error. (3) Because $X^*_{(n)} \le X_{(n)} < \theta$ almost surely: every bootstrap replicate of the maximum is at most the observed maximum, which is strictly below $\theta$, so the entire percentile interval lies below the truth in every dataset. Measured coverage was exactly $0.0000$. (4) Infinite variance (or otherwise failing moment conditions). For the Cauchy and $t_{1.5}$ data the interval width stalled around $4.4$ across $n = 25$ to $1600$; a width that does not shrink like $n^{-1/2}$ is the visible symptom that no CLT is operating and the bootstrap of the mean is inconsistent. The repair is to change the functional — the median or a trimmed mean bootstraps fine — not the interval recipe.

**Self-check 9.9.** (1) By $\sqrt{1-\rho} = \sqrt{0.25} = 1/2$ — a two-fold reduction in standard error, equivalent to quadrupling the test set. (2) Only the discordant cells $b$ and $c$; conditioning the multinomial on $b+c$ eliminates the concordant cells from the conditional likelihood, because items both models get right or both get wrong carry no information about which is better. (3) Because the 30 repeated-CV values re-use the same data: they inflate the apparent degrees of freedom without adding information, so the estimated standard error is too small. Measured Type I error was $0.1220$ (repeated CV) versus $0.0225$ (single 10-fold) in the simulation of Worked Example 9.23. (4) $\Delta = \sqrt{2 \cdot 7.8489 \cdot 0.4^2/5} = 0.709$ points — anything smaller than roughly $0.7$ accuracy points is beyond your resolution with 5 seeds. (5) Because $F_1 = 2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$ is a ratio of three dependent counts, not a sample proportion, so $F_1(1-F_1)/n$ is not its variance. Numerically the naive interval came out 40% too narrow ($0.0438$ against the bootstrap's $0.0727$) in §9.9.4. Use a bootstrap or the delta method. (6) Roughly 500, not 2000: questions within a document share difficulty and phrasing, so they are not independent draws. Resample *documents*. In the simulation of §9.9.4, resampling items instead of clusters dropped coverage from $0.956$ to $0.665$.

# Solutions to Exercises

## Part A

**A1.** *Thinking:* the whole point is that the binomial has finitely many attainable tail probabilities, so we enumerate.

Under $H_0$, $X\sim\mathrm{Bin}(10,1/2)$ and $\mathbb{P}(X = k) = \binom{10}{k}/1024$.

(a) $\mathbb{P}(X\ge 8) = (45+10+1)/1024 = 56/1024 = 0.0546875 > 0.05$, while $\mathbb{P}(X\ge 9) = 11/1024 = 0.0107422 \le 0.05$. So the smallest such region is $\{X\ge 9\}$, with exact size $0.0107422$. Note how far below $0.05$ this lands: the "$\alpha=0.05$ test" here really operates at $\alpha \approx 0.011$, and you are silently paying for it in power.

(b) Reject when $X\ge 9$; when $X = 8$ reject with probability $\gamma$ chosen so the size is exactly $0.05$:
$$0.0107422 + \gamma\cdot\frac{45}{1024} = 0.05 \implies \gamma = \frac{0.0392578}{0.0439453} = 0.893336.$$

(c) At $\theta = 0.8$: $\mathbb{P}(X\ge9) = 10(0.8)^9(0.2) + (0.8)^{10} = 0.2684355 + 0.1073742 = 0.3758096$, and $\mathbb{P}(X=8) = 45(0.8)^8(0.2)^2 = 0.3019899$. So the non-randomized test has power $0.3758$ and the randomized test has power $0.3758096 + 0.893336\times0.3019899 = 0.6455875$.

*Moral:* recovering the missing $0.039$ of size buys $0.27$ of power. Randomization is unpalatable in practice — nobody wants a coin flip to decide their paper — but the size of the loss is worth knowing before you accept the conservative test.

**A2.** Size is $\sup_{P\in H_0}\mathbb{E}_P[\varphi]$ and level is any $\alpha$ with size $\le\alpha$, so size $\le$ level by definition, and if size $\le\alpha\le\alpha'$ then size $\le\alpha'$. For monotonicity of the power function of $\{X\ge 9\}$: $\beta(\theta) = 10\theta^9(1-\theta)+\theta^{10} = 10\theta^9 - 9\theta^{10}$, so $\beta'(\theta) = 90\theta^8 - 90\theta^9 = 90\theta^8(1-\theta) > 0$ on $(0,1)$. (Alternatively: $\mathrm{Bin}(n,\theta)$ is stochastically increasing in $\theta$ — couple by comparing $n$ uniforms to $\theta$ — so the probability of any up-set increases, which is the general argument and the one that generalizes to A4.)

**A3.** (a) $d_{\mathrm{TV}} = \tfrac12\int|p_1-p_0|$. The densities cross exactly at $x = \mu/2$, with $p_1 > p_0$ for $x>\mu/2$. Hence
$$d_{\mathrm{TV}} = \int_{\mu/2}^\infty (p_1-p_0) = \big[1-\Phi(\mu/2-\mu)\big] - \big[1-\Phi(\mu/2)\big] = \Phi(\mu/2)-\Phi(-\mu/2) = 2\Phi(\mu/2)-1.$$

(b) The ceiling $\alpha + d_{\mathrm{TV}}$ is attained by the test rejecting on $\{p_1 > p_0\} = \{x > \mu/2\}$, which is the NP test with threshold $k=1$. The $z$-test rejects on $\{x > z_{1-\alpha}\}$, so the two coincide iff $z_{1-\alpha} = \mu/2$, i.e. $\alpha = 1-\Phi(\mu/2)$. Interpretation: the ceiling is attained by the *Bayes rule under equal priors*, which is a level-$(1-\Phi(\mu/2))$ test — for small $\mu$ that is a level close to $1/2$. At any conventional $\alpha$ the ceiling is not attained, and the inequality in Proposition 9.5 is strict but still informative as a bound.

(c) $\mu = 0.1$: $d_{\mathrm{TV}} = 2\Phi(0.05)-1 = 0.0398777$, so no level-$0.05$ test on one observation has power above $0.0899$. A claimed detection from a single observation of a $0.1\sigma$ effect is not a detection.

**A4.** For $\theta_2 > \theta_1$,
$$\frac{p_{\theta_2}(x)}{p_{\theta_1}(x)} = \exp\big\{(\eta(\theta_2)-\eta(\theta_1))T(x) - (A(\theta_2)-A(\theta_1))\big\},$$
and since $\eta(\theta_2) > \eta(\theta_1)$ the exponent is strictly increasing in $T(x)$; the additive constant does not depend on $x$. So the family has MLR in $T$. Karlin–Rubin (Theorem 9.10) then gives: for $H_0:\theta\le\theta_0$ versus $H_1:\theta>\theta_0$, the test rejecting when $T(X) > c$ (with randomization at $T=c$) and calibrated so that $\mathbb{E}_{\theta_0}[\varphi]=\alpha$ is UMP at level $\alpha$.

A non-exponential family with MLR: the $\mathrm{Unif}[0,\theta]$ family. For $\theta_2>\theta_1$, $p_{\theta_2}/p_{\theta_1} = (\theta_1/\theta_2)\mathbf 1\{x\le\theta_1\} + \infty\cdot\mathbf 1\{\theta_1<x\le\theta_2\}$, which is non-decreasing in $x$ (with the convention that $\infty$ exceeds everything). It is not an exponential family because the support depends on $\theta$. Another: the Cauchy location family does *not* have MLR, and the logistic location family does — a useful pair to remember.

**A5.** If $p\sim\mathrm{Unif}[0,1]$ exactly, then $\min(p,1-p)\sim\mathrm{Unif}[0,1/2]$ by symmetry of the uniform about $1/2$, so $2\min(p,1-p)\sim\mathrm{Unif}[0,1]$ and the two-sided version is a valid (indeed exact) $p$-value.

Counterexample under mere super-uniformity: let $p\equiv 1$ deterministically. This is super-uniform, since $\mathbb{P}(p\le u) = 0 \le u$ for $u<1$. But then $\min(p,1-p) = 0$ and $q = \min(1, 2\cdot 0) = 0$, so $\mathbb{P}(q \le u) = 1 > u$ for every $u<1$: maximally invalid. *Why the proof breaks:* the symmetrization step uses the distributional symmetry of the uniform, which super-uniformity does not provide. This matters in practice because discrete tests produce super-uniform, non-uniform $p$-values, and doubling a one-sided discrete $p$-value is a heuristic that needs separate justification (it is conservative for binomial tails, but that is a theorem about binomials, not a general fact).

**A6.** (a) $\mathbb{P}(X_{(n)}/\theta \le u) = u^n$ on $[0,1]$, free of $\theta$: an exact pivot. Then $\mathbb{P}(a \le X_{(n)}/\theta \le b) = b^n - a^n$, and inverting gives $[X_{(n)}/b,\ X_{(n)}/a]$. Two natural choices: equal-tailed with $a = (\alpha/2)^{1/n}$, $b=(1-\alpha/2)^{1/n}$; or the one-sided-in-$b$ choice $b = 1$, $a = \alpha^{1/n}$, giving $[X_{(n)},\ X_{(n)}\alpha^{-1/n}]$. The second is shorter, because $\theta \ge X_{(n)}$ always holds, so no coverage should be spent below $X_{(n)}$.

(b) The likelihood is $L(\theta) = \theta^{-n}\mathbf 1\{\theta \ge X_{(n)}\}$, maximized at $\hat\theta = X_{(n)}$. For $H_0:\theta=\theta_0$, $\Lambda = L(\theta_0)/L(\hat\theta) = (X_{(n)}/\theta_0)^n$ for $\theta_0\ge X_{(n)}$ and $0$ otherwise. Accepting when $\Lambda \ge \alpha$ gives $\theta_0 \le X_{(n)}\alpha^{-1/n}$, i.e. exactly the one-sided pivotal interval $[X_{(n)}, X_{(n)}\alpha^{-1/n}]$. So they coincide — a pleasant check. Expected length: $\mathbb{E}[X_{(n)}] = n\theta/(n+1)$, so the expected length is $\frac{n\theta}{n+1}(\alpha^{-1/n}-1)$; at $n=20,\alpha=0.05$ this is $0.9524\theta\times0.16159 = 0.15390\theta$, shrinking like $\log(1/\alpha)\theta/n$ — rate $1/n$, not $1/\sqrt n$.

(c) Because $\sqrt n(\hat\theta-\theta)\to 0$ in probability: the correct scaling is $n(\theta - X_{(n)})/\theta \Rightarrow \mathrm{Exp}(1)$, an asymmetric limit with rate $1/n$. This is a non-regular family (the support depends on $\theta$, so differentiation under the integral sign fails and the Fisher information identity does not hold), and every step of the Wald construction — asymptotic normality, the sandwich variance, symmetry of the interval — is false. It is the cleanest example of why "$\hat\theta \pm 1.96\,\widehat{\mathrm{se}}$" is a theorem with hypotheses, not a definition.

**A7.** *Holm dominates Bonferroni.* Suppose $p_i \le \alpha/m$. Holm examines the ordered $p$-values against thresholds $\alpha/(m-k+1)$, which are non-decreasing in $k$ and all $\ge \alpha/m$. If $p_i \le \alpha/m$ then every $p$-value at or below $p_i$ in the ordering is also $\le\alpha/m$, so no step before $p_i$'s step can halt the procedure; hence Holm rejects $H_i$. Strict improvement: $m=2$, $\alpha=0.05$, $p = (0.001, 0.03)$. Bonferroni's threshold is $0.025$ and rejects only the first. Holm rejects the first ($0.001 \le 0.025$), then compares $0.03$ to $\alpha/1 = 0.05$ and rejects it too.

*The BH part.* BH's thresholds are $iq/m$ for $i = 1,\dots,m$, so the largest threshold it ever uses is $q$ itself, attained only when all $m$ hypotheses are rejected. **BH never rejects a hypothesis with $p_{(i)} > q$.** What it can do is reject a $p$-value exceeding the *Bonferroni* threshold $q/m$ by a factor as large as $m$. Example: $m=10$, $q=0.05$, sorted $p$-values $0.001, 0.008, 0.014, 0.4,\dots$. The largest $i$ with $p_{(i)}\le 0.005i$ is $i=3$ (since $0.014\le0.015$), so BH rejects three hypotheses, one of them with $p = 0.014 = 2.8\times(q/m)$ — a $p$-value Bonferroni and Holm both retain.

This contradicts nothing Bonferroni guarantees, because the two procedures guarantee different things. Bonferroni bounds $\mathbb{P}(V\ge1)$; BH bounds $\mathbb{E}[V/(R\vee1)]$. A procedure can reject more, be right on average about the *proportion* of its errors, and still have a high probability of making at least one — which is exactly what the simulation shows: BH's FWER was $0.9030$ while its FDR was $0.0459$.

**A8.** Write $t_k = kq/(mH_m)$ for the BY thresholds, let $R$ be the number of rejections and $V$ the number of false ones. Since BY rejects $H_i$ iff $p_i \le t_R$,
$$\mathrm{FDR} = \mathbb{E}\Big[\frac{V}{R\vee1}\Big] = \sum_{k=1}^m \frac1k\sum_{i\in H_0}\mathbb{P}(p_i \le t_k,\ R=k).$$
Fix $i\in H_0$ and put $A_k = \{p_i\le t_k, R=k\}$; the $A_k$ are disjoint. Use the telescoping identity $\frac1k = \sum_{j=k}^m\big(\frac1j - \frac1{j+1}\big) + \frac1{m+1}$:
$$\sum_{k=1}^m \frac{\mathbb{P}(A_k)}{k} = \sum_{j=1}^m\Big(\frac1j-\frac1{j+1}\Big)\sum_{k=1}^{j}\mathbb{P}(A_k) + \frac{1}{m+1}\sum_{k=1}^m\mathbb{P}(A_k).$$
For $k\le j$ we have $A_k\subseteq\{p_i\le t_j\}$, and the $A_k$ are disjoint, so $\sum_{k\le j}\mathbb{P}(A_k)\le \mathbb{P}(p_i\le t_j)\le t_j = jq/(mH_m)$ by super-uniformity of $p_i$ — **this is the only probabilistic input, and it is marginal, so no dependence assumption is used.** Therefore
$$\sum_{k=1}^m\frac{\mathbb{P}(A_k)}{k} \le \frac{q}{mH_m}\Big[\sum_{j=1}^m \frac{j}{j(j+1)} + \frac{m}{m+1}\Big] = \frac{q}{mH_m}\Big[(H_{m+1}-1) + \frac{m}{m+1}\Big] = \frac{q}{mH_m}\,H_m = \frac{q}{m},$$
using $H_{m+1} = H_m + \frac1{m+1}$ so that the bracket is $H_m + \frac{1}{m+1} - 1 + \frac{m}{m+1} = H_m$. Summing over the $m_0$ nulls gives $\mathrm{FDR}\le m_0q/m \le q$. $\blacksquare$

The price is the factor $H_m \approx \log m + 0.577$: at $m=1000$, $H_m = 7.486$, so BY's thresholds are $7.5$ times smaller than BH's. That is a real power cost, which is why BY is used when dependence is genuinely arbitrary and unknown, and BH is used when a PRDS argument is plausible.

**A9.** Let $\mathcal{O}(x) = \{gx : g\in G\}$ be the orbit. Under $H_0$, $X \overset{d}{=} gX$ for every $g\in G$. Since $G$ is a group, for $g_0$ drawn uniformly from $G$ independently of $X$ we have $g_0X \overset{d}{=} X$; and conditionally on the orbit, $g_0X$ is uniform on the multiset $\{gX\}_{g\in G}$. Hence, conditionally on the orbit, $T(X)$ is distributed as a uniform draw from the multiset $\{T(gX)\}_{g\in G}$, and $\hat p$ is $1/M$ times the rank-from-the-top of that draw. If there are no ties, $\hat p$ is uniform on $\{1/M, 2/M,\dots,1\}$, so
$$\mathbb{P}(\hat p\le\alpha \mid \mathcal{O}) = \frac{\lfloor \alpha M\rfloor}{M} \le \alpha,$$
with equality iff $\alpha M\in\mathbb{Z}$. Averaging over orbits preserves the inequality. With ties, values of $\hat p$ clump upwards (ties count in the numerator), which only decreases the rejection probability. $\blacksquare$

**A10.** *Step 1 (bootstrap CLT for the mean).* Condition on the data. Given $\hat P_n$, the resampled observations $X_1^*,\dots,X_n^*$ are i.i.d. with mean $\bar X_n$ and variance $\hat\sigma_n^2 = n^{-1}\sum(X_i-\bar X_n)^2$. By the SLLN, $\bar X_n\to\mu$ and $\hat\sigma^2_n\to\sigma^2$ almost surely. A Lindeberg argument for triangular arrays (or Bickel–Freedman's theorem, or the Mallows-metric argument) gives, for almost every data sequence, $\sqrt n(\bar X^*_n - \bar X_n)\Rightarrow N(0,\sigma^2)$ conditionally.

*Step 2 (delta method, transferred).* Taylor with remainder: $g(\bar X^*_n) - g(\bar X_n) = g'(\bar X_n)(\bar X^*_n-\bar X_n) + \tfrac12 g''(\xi_n)(\bar X^*_n - \bar X_n)^2$ for some $\xi_n$ between the two. Multiplying by $\sqrt n$: the first term is $g'(\bar X_n)\cdot\sqrt n(\bar X^*_n-\bar X_n)$, and since $g'(\bar X_n)\to g'(\mu)\ne0$ a.s. and $g'$ is continuous, Slutsky gives a $N(0, g'(\mu)^2\sigma^2)$ limit. The second term is $\sqrt n\cdot O_P(1/n) = O_P(n^{-1/2})\to0$, using local boundedness of $g''$ and $\xi_n\to\mu$. Hence $\sqrt n(\hat\theta^* - \hat\theta)\Rightarrow N(0,g'(\mu)^2\sigma^2)$ conditionally, a.s. — the same limit as $\sqrt n(\hat\theta-\theta)$, which is bootstrap consistency.

*Failure at $g'(\mu)=0$.* The step that breaks is Slutsky applied to the *linear* term: it no longer dominates. Now $g'(\bar X_n)\approx g''(\mu)(\bar X_n-\mu) = O_P(n^{-1/2})$, so the linear and quadratic terms are both $O_P(1/n)$ and must be kept. Writing $Z = \lim\sqrt n(\bar X_n-\mu)/\sigma$ and $Z^* = \lim\sqrt n(\bar X^*_n - \bar X_n)/\sigma$ with $Z^*\perp Z$,
$$n\big(\hat\theta-\theta\big)\Rightarrow \tfrac12 g''(\mu)\sigma^2Z^2, \qquad n\big(\hat\theta^*-\hat\theta\big)\Rightarrow \tfrac12 g''(\mu)\sigma^2\big(Z^{*2} + 2ZZ^*\big).$$
These are different laws — the second is not even a fixed distribution, since it depends on $Z$, i.e. on the data, and does not converge to anything as a conditional law. So the bootstrap is **inconsistent**: this is the degenerate-derivative failure, the smooth cousin of the non-smooth failure at the maximum in Proposition 9.41. Remedies: the $m$-out-of-$n$ bootstrap, or subsampling, both of which work by breaking the exact matching of resample size to sample size.

## Part B

All numbers below are from my own runs; seeds and sizes are stated so you can reproduce or contradict them. Monte Carlo standard errors for a rejection rate $\hat r$ from $B$ replications are $\sqrt{\hat r(1-\hat r)/B}$; at $B = 200{,}000$ and $\hat r\approx0.05$ that is $0.0005$, and at $B=2000$ and $\hat r\approx0.05$ it is $0.005$ — worth keeping in mind before reading meaning into the third decimal.

**B1.** Worked Example 9.12 has the run in full: $B = 200{,}000$ two-sided $z$-tests under the null, $n=30$ per test, seed 9091. Sample mean of the $p$-values $0.49940$ against the target $0.5$; sample variance $0.08351$ against $1/12 = 0.08333$; decile counts all within $\pm210$ of the expected $20{,}000$; Kolmogorov–Smirnov distance to $\mathrm{Unif}[0,1]$ of $0.00197$. Calibrate that against the KS critical value $1.36/\sqrt B = 0.00304$: the observed distance is comfortably *below* it, and the KS test's own $p$-value is $0.42$. The tail probabilities land where they should: $\mathbb{P}(p\le0.01) = 0.00980$, $\mathbb{P}(p\le0.05)=0.05056$, $\mathbb{P}(p\le0.10)=0.10064$.

The point of doing this at $B = 200{,}000$ rather than $B = 1000$ is resolution: the standard error on $\mathbb{P}(p\le0.05)$ is $\sqrt{0.05\cdot0.95/B} = 0.00049$, so you can detect a miscalibration of a tenth of a percentage point. That is the sensitivity you want from a null-calibration check, and it is why this lab is the first thing to run whenever you implement a test yourself.

Under alternatives, the deciles march downward: at $\delta = 0.1$ the distribution is still nearly uniform (power $\approx 0.17$ at the sample size used); at $\delta = 0.5$ the median $p$-value falls to roughly $0.05$; at $\delta = 1.0$ more than 90% of $p$-values fall below $0.05$ and the density piles up at zero. The shape is the whole intuition behind Figure 9.3: a null histogram is flat, an alternative histogram is right-skewed with a spike at zero, and a histogram with a spike just below $0.05$ and a dip just above it is evidence of malpractice rather than of an effect.

For the discrete case, work it out exactly rather than simulating. For $X\sim\mathrm{Bin}(20,1/2)$ the two-sided $p$-value is $\mathbb{P}(|X-10|\ge|x-10|)$, which takes only 11 distinct values. Rejecting at nominal $0.05$ means $|X-10|\ge5$, since $\mathbb{P}(X\ge15) = 21700/2^{20} = 0.020695$ and doubling gives size $0.041390$, whereas $|X-10|\ge4$ would give $0.115$. So the "$0.05$ test" has true size $0.0414$, the $p$-value histogram is a staircase with atoms rather than a flat density, and super-uniformity holds with slack — precisely Proposition 9.18(i) and not (ii).

**B2.** The two tables are in §9.4 (seed 777 for the discrete-looks design, seed 123 for continuous monitoring). Discrete looks at $k = 1,2,5,10,20,50,100$ gave Type I errors $0.0497$, $0.0820$, $0.1404$, $0.1914$, $0.2467$, $0.3197$, $0.3724$; continuous monitoring up to $N = 10, 10^2, 5\times10^2, 2\times10^3, 5\times10^3, 10^4, 2\times10^4$ gave $0.0506$ through $0.6337$.

Two things to notice that the raw numbers do not shout. First, the growth is *slow* — roughly logarithmic — which is exactly the law of the iterated logarithm signature: the running $|z_n|$ has $\limsup |z_n|/\sqrt{2\log\log n} = 1$, so the threshold-crossing probability creeps up rather than jumping. Slow growth is what makes optional stopping so easy to do accidentally and so hard to notice. Second, it is unbounded (Proposition 9.21), so "I only peeked a few times" is a quantitative defence, not a qualitative one — five looks already triples your error rate.

For the fix: a Pocock boundary uses a constant critical value across looks, chosen so the overall level is $\alpha$ — for $k=5$ looks at $\alpha = 0.05$ the nominal per-look level is $0.0158$, i.e. $|z| > 2.413$ — and restored the measured error to $0.05$. O'Brien–Fleming boundaries are very stringent early and near-nominal at the final look, which is why they preserve almost all the power of a single fixed-$n$ test while still permitting early stopping for dramatic effects. Both are worth knowing; neither requires anything beyond §9.4.

**B3.** The full table is in §9.6 (seed 1, $m = 1000$, $m_0 = 900$, alternatives at $\mu = 3$, 2000 replications, $\alpha = q = 0.05$). Three things to extract from it.

*Holm's gain over Bonferroni is real but tiny here* — power $0.1469$ against $0.1461$. The reason is structural: Holm's $k$-th threshold is $\alpha/(m-k+1)$, and with $m = 1000$ and only about 150 rejections, $m-k+1$ never falls far below $1000$, so the thresholds barely move. Holm's advantage is large when the number of rejections is a substantial *fraction* of $m$. It is never negative, which is the argument for never using Bonferroni.

*BH's FDR sits exactly on the bound.* Measured $0.0459$; the theoretical bound is $qm_0/m = 0.045$. Note the measured value is very slightly **above** the bound, not below it. With 2000 replications the Monte Carlo standard error is about $0.002$, so the two are indistinguishable, and the correct conclusion is that Theorem 9.32 is *tight* — the inequality is not leaving power on the table under independence. Reporting this as "comfortably under the bound" would be a misreading, and reporting it as "the bound is violated" would be over-reading Monte Carlo noise. Both temptations are worth resisting deliberately.

*BH controls FDR, not FWER, and the gap is enormous.* FWER $0.9030$ against FDR $0.0459$. Under positive dependence BH becomes conservative rather than anti-conservative: at $\rho = 0.9$ the realized FDR fell to $0.0136$, well under $q$, consistent with the PRDS theory that guarantees control (not tightness) in that regime. In the all-null configuration ($m_0 = m$) FDR and FWER coincide by definition and all three procedures measured $0.0505$ — a useful sanity check, since with no true effects BH's step-up has almost nothing to step up through.

**B4.** (i) The coverage tables are in §9.8.2: $B = 2000$ resamples, $2000$ replications, $n = 20$ throughout, with three distributions of increasing skewness (seeds 11, 12, 13). Studentized coverage was $0.9405$ for $\chi^2_3$, $0.9415$ for $\mathrm{Exponential}(1)$, and $0.8660$ for $\mathrm{Lognormal}(0,1.5)$; the ranking of methods was the same in all three — studentized best, then BCa, then percentile and normal, with basic worst. The ordering across distributions is monotone in skewness, which is the lesson: the bootstrap's error for a mean is governed by skewness, at order $n^{-1/2}$, and studentizing removes the leading term of that error (an Edgeworth-expansion fact). At lognormal skewness *every* method fails badly — $0.87$ at best — and the honest response is not to hunt for a better interval but to report that the mean of a heavy-tailed variable is not estimable from 20 observations. Note also that studentizing buys coverage with width (17.6 average width against 4.7 for percentile in the lognormal case); the wide interval is the correct one.

(ii) Maximum of $\mathrm{Unif}[0,1]$, $n=50$. Analytically, $\mathbb{P}(\max_i X^*_i = X_{(n)}) = 1-(1-1/n)^n$, which at $n=50$ is $0.635830$; numerically the bootstrap distribution of $\hat\theta^*$ took only $13$ distinct values with a mass of $0.6358$ at the top one. Measured coverage: percentile $0.0000$, basic $0.8770$, and the exact pivot from A6 $0.9457$. The hypothesis of Theorem 9.38 that fails is smoothness — the max is not a differentiable functional of the empirical measure; the sampling distribution of $n(\theta - X_{(n)})$ is exponential, not normal, and no amount of resampling recovers it.

(iii) Cauchy and $t_{1.5}$: coverage stalled near $0.91$ and, crucially, the interval width did **not** shrink as $n$ grew. The failing hypothesis is finite variance (Cauchy) or finite variance again ($t_{1.5}$ has infinite variance since $\nu<2$); the sample mean is not asymptotically normal and the bootstrap of the mean is inconsistent. The control case $t_5$ (finite variance) reached $0.9530$. Diagnostic worth internalizing: *width not shrinking like $n^{-1/2}$* is the visible symptom of a moment failure.

(iv) AR(1) with $\rho=0.8$, seed 5150: i.i.d. resampling gave coverage $0.4685$; a block bootstrap with block length $20$ gave $0.8410$. The failing hypothesis is independence. I.i.d. resampling destroys the serial correlation and so estimates the variance of a mean of $n$ independent observations rather than of $n_{\text{eff}} \approx n(1-\rho)/(1+\rho) = n/9$ effective ones — an underestimate by a factor of about $3$ in the standard error, which is exactly the size of the coverage failure observed. Block resampling preserves within-block dependence and recovers most, but not all, of the coverage; the residual shortfall is the usual bias from breaking dependence at block boundaries.

**B5.** Worked Example 9.22 has the full run (seed 8675309, $n = 2000$, accuracies $0.9000$/$0.9100$, agreement $95.60\%$, $\phi = 0.7454$): unpaired $[-0.00817, 0.02817]$ with $p = 0.2807$; paired $[0.00082, 0.01918]$ with $p = 0.0329$; paired bootstrap $[0.00100, 0.01900]$; McNemar exact $p = 0.0422$. Width ratio $1.978$, effective sample-size multiplier $3.91$, and $13{,}493$ versus $3{,}448$ items for 80% power at $\delta = 0.01$.

The part the exercise adds is the sweep over correlation. At $\phi = 0.4251$ the multiplier drops to $1.74$ and McNemar becomes non-significant ($b = 89$, $c = 109$, exact $p = 0.1768$) — **the same true one-point gap, now undetectable**, purely because the models' errors overlap less. Plotting the multiplier against $\phi$ traces $1/(1-\phi)$, matching Proposition 9.42 with $\sigma_A\approx\sigma_B$: at $\phi = 0.7454$ the prediction is $3.93$ against the measured $3.91$, and at $\phi = 0.4251$ it is $1.740$ against the measured $1.74$. Theory and simulation agree to the third digit, which is the standard you should hold your own simulations to.

The reading that matters: the value of pairing is exactly the agreement between your models, so pairing helps most where you need it most — two strong, similar models separated by a little. It also means a paper cannot report "we found no difference" without reporting the agreement rate, because that number determines whether the experiment could have found one.

**B6.** Seed 7, $R = 2000$ datasets, $N = 200$, ridge regression, three real features plus 15 noise features for each learner (different noise features, so the null is true by exchangeability), $\alpha = 0.05$: single 10-fold paired $t$-test $0.0225$; $3\times$ repeated 10-fold $t$-test on 30 values $0.1220$; 30 random 90/10 splits $0.1965$. True SD of the 10-fold CV difference across datasets $0.05622$; mean naive SE $s/\sqrt{10}$ from one run $0.06442$; ratio $0.873$. Inverting Proposition 9.45 on the reciprocal ratio $1.1459 = \sqrt{(1-\varrho)/(1+9\varrho)}$ gives an implied fold correlation $\varrho \approx -0.024$: slightly negative, which is why the single-run test came out conservative.

The honest reading, which differs from the usual slogan: the single-run 10-fold $t$-test was *conservative* here, because in this design the fold-to-fold spread exceeded the dataset-to-dataset spread of the CV average. The slogan "CV $t$-tests are anti-conservative" is true of the repeated variants and not automatically true of the single-run variant — and Bengio and Grandvalet explain why no general claim is available: there is no unbiased estimator of the variance of $K$-fold CV from a single run, and the sign of the bias depends on the learner, $K$, and the data-generating process. What *is* robustly true is the repeated-resampling result: reusing the same data to manufacture 30 "replicates" inflates the apparent degrees of freedom without adding information, and the measured error rates $0.12$ and $0.20$ against a nominal $0.05$ are the price. If you have a real test set, use the paired item-level analysis of B5 instead; if you do not, use a corrected test (Nadeau–Bengio, Dietterich's $5\times2$-cv) and report the fold scores.

**B7.** (a) At $\hat p = 0.95$, $n=1000$, Wald gives $\pm0.0135$ and Wilson gives $[0.9346,0.9628]$ — Wilson is shifted slightly downward and asymmetric, which is correct behaviour near a boundary; the two agree closely away from $0$ and $1$ and diverge sharply near them (§9.5, where Wald's coverage at $n=40$, $p=0.01$ is $0.331$). Violated assumption: Wald's normal approximation to the binomial, which fails when $np(1-p)$ is small.

(b) Seed 1234: $\mathrm{TP}=271$, $\mathrm{FP}=414$, $\mathrm{FN}=119$, $F_1 = 0.5042$. Naive binomial interval $[0.4823, 0.5261]$, width $0.0438$; bootstrap ($B=20{,}000$) $[0.4673, 0.5399]$, width $0.0727$. The naive interval is 40% too narrow. Violated assumption: $F_1$ is not a sample proportion — it is a ratio of three dependent counts, so $F_1(1-F_1)/n$ is not its variance. Nothing about the formula's plausible appearance makes it apply.

(c) Seed 99, $4000$ replications, $400$ clusters of $5$ with 90% within-cluster duplication, true gap $0.01$: resampling items gives coverage $0.6645$ (mean width $0.0356$); resampling clusters gives $0.9563$ (mean width $0.0746$). Violated assumption: independence of items. The effective sample size is closer to the number of clusters than to the number of items, and a nominal 95% interval that covers 66% of the time is the price of ignoring that. The fix requires no new theory — just resample the right unit.

## Part C

**C1 (Neyman–Pearson, full statement and proof).** Let $P_0, P_1$ be probability measures dominated by a $\sigma$-finite $\mu$, with densities $p_0,p_1$, and set $L = p_1/p_0$ where $p_0>0$ (with $L = +\infty$ where $p_0 = 0 < p_1$). Fix $\alpha\in(0,1)$.

*(i) Existence.* Let $F(k) = \mathbb{P}_0(L > k)$, which is non-increasing and right-continuous in $k$ with $F(0^-)=1$ (up to the null set $\{p_0=0\}$) and $F(\infty)=0$. Choose $k_\alpha = \inf\{k\ge0 : F(k)\le\alpha\}$. Right-continuity gives $F(k_\alpha)\le\alpha$, and the definition of the infimum gives $\mathbb{P}_0(L\ge k_\alpha) \ge \alpha$. Define
$$\varphi(x) = \begin{cases}1, & L(x)>k_\alpha,\\ \gamma, & L(x)=k_\alpha,\\ 0,&L(x)<k_\alpha,\end{cases}\qquad \gamma = \frac{\alpha - \mathbb{P}_0(L>k_\alpha)}{\mathbb{P}_0(L=k_\alpha)}$$
when $\mathbb{P}_0(L = k_\alpha)>0$, and $\gamma$ arbitrary otherwise. Then $\gamma\in[0,1]$ by the two displayed inequalities, and $\mathbb{E}_0[\varphi] = \alpha$ exactly. **The atom is exactly where randomization is needed**; when $L$ has a continuous distribution under $P_0$, $\gamma$ is irrelevant and the test is non-randomized.

*(ii) Optimality.* Let $\psi$ be any test with $\mathbb{E}_0[\psi]\le\alpha$. Consider
$$\int(\varphi-\psi)(p_1 - k_\alpha p_0)\,d\mu.$$
On $\{L>k_\alpha\}$ we have $\varphi = 1\ge\psi$ and $p_1-k_\alpha p_0>0$; on $\{L<k_\alpha\}$ we have $\varphi=0\le\psi$ and $p_1 - k_\alpha p_0<0$; on $\{L=k_\alpha\}$ the second factor vanishes. So the integrand is pointwise $\ge0$ and the integral is $\ge 0$. Expanding,
$$\mathbb{E}_1[\varphi]-\mathbb{E}_1[\psi] \ \ge\ k_\alpha\big(\mathbb{E}_0[\varphi]-\mathbb{E}_0[\psi]\big) = k_\alpha(\alpha - \mathbb{E}_0[\psi])\ \ge\ 0 .$$
Hence $\varphi$ is most powerful among level-$\alpha$ tests.

*(iii) Uniqueness a.e. off the boundary.* Suppose $\psi$ is also most powerful with $\mathbb{E}_0[\psi]\le\alpha$. Then $\mathbb{E}_1[\varphi]=\mathbb{E}_1[\psi]$, so the chain in (ii) is equality throughout; since $k_\alpha>0$ (for $\alpha<1$ and non-degenerate $L$) the last inequality forces $\mathbb{E}_0[\psi]=\alpha$, and the first forces $\int(\varphi-\psi)(p_1-k_\alpha p_0)\,d\mu = 0$. As the integrand is non-negative it vanishes $\mu$-a.e., and since $p_1 - k_\alpha p_0\ne0$ off $\{L=k_\alpha\}$, we get $\varphi=\psi$ $\mu$-a.e. on $\{L\ne k_\alpha\}$. On the boundary set $\psi$ is unconstrained except through its contribution to the size. $\blacksquare$

*Where absolute continuity matters.* Domination by a common $\sigma$-finite $\mu$ is what makes $p_0,p_1$ and hence $L$ exist; it is essentially free (take $\mu = P_0+P_1$). What is **not** needed is $P_1\ll P_0$: the set $\{p_0=0, p_1>0\}$ is handled by $L=+\infty$, and it is always in the rejection region, contributing $\mathbb{P}_1(p_0=0)$ to the power at no cost in size. That set is precisely why a test can have power $1$ at level $0$ when the hypotheses are mutually singular — the degenerate case at the far end of Proposition 9.5.

**C2.** The independence proof is Theorem 9.32 above; I will not repeat it, but the structure is worth restating in one line because it is the part people misremember: for each null $i$ and each possible rejection count $k$, replacing $p_i$ by $0$ leaves the BH rejection count at $k$ on the relevant event, which decouples the indicator $\{p_i \le kq/m\}$ from the count and lets independence do its work; summing over $k$ telescopes to $q\,m_0/m$.

*What is known under dependence.*
- **PRDS.** If the $p$-value vector is positive regression dependent on the subset of nulls, BH still controls FDR at $q\,m_0/m$ (Benjamini–Yekutieli 2001). One-sided tests from equicorrelated Gaussians with $\rho\ge0$, and many multivariate-$t$ and studentized settings, satisfy PRDS. Two-sided tests generally do not, which is a subtlety usually skipped.
- **Arbitrary dependence.** The BY correction, thresholds $iq/(mH_m)$, controls FDR at $q$ with no assumptions beyond marginal super-uniformity — proved in A8. The cost is the factor $H_m\approx\log m$.
- **Negative or mixed dependence.** No general guarantee for plain BH; counterexamples exist in which BH's FDR exceeds $q$, though the excess found in practice is modest.
- **Equicorrelated Gaussians.** Simulation (B3) and theory agree that the *mean* FDP stays controlled for $\rho>0$, but its *variance* grows sharply: the FDP distribution becomes bimodal, and a single experiment can have a much higher false-discovery proportion than $q$ even though the average is fine.
- **Also available:** Storey's adaptive procedures (estimate $m_0/m$ and gain the corresponding power back, valid under independence and some dependence), knockoffs and other finite-sample FDR methods for regression, and $e$-value/$e$-BH methods which control FDR under arbitrary dependence with a much smaller penalty than $H_m$.

*Recommendation, defended.* Use BH at a stated $q$ for screening, and say in the paper that FDR — not FWER — is what is controlled and what $q$ you chose. Use Holm (never plain Bonferroni) when a single false claim would invalidate the paper's headline. Reserve BY for cases where dependence is genuinely arbitrary and adversarial, and note the $\log m$ cost honestly rather than applying BY reflexively. Defence: the choice is a decision problem, and the two guarantees answer different questions — FWER bounds the probability of *any* error, FDR bounds the expected *proportion*. A screening study that will be followed by replication should optimize the number of true leads per unit of follow-up cost, which is FDR's objective; a paper making one claim has $m$ effectively equal to the number of claims and should control FWER, where Holm's dominance over Bonferroni (A7) makes Bonferroni indefensible.

**C3 (a protocol, and its application).**

**Protocol.**
1. **Estimand.** $\Delta = \mathrm{acc}(B) - \mathrm{acc}(A)$, where accuracy is expectation over the benchmark's item distribution *and* over training randomness: $\mathrm{acc}(M) = \mathbb{E}_{s}\mathbb{E}_{(x,y)}[\mathbf 1\{M_s(x)=y\}]$. State which of the two expectations you are averaging over; most papers silently condition on one seed.
2. **Randomness.** Two sources: test-item sampling (reduced by more items) and training seeds (reduced only by more runs). Both must be in the interval.
3. **Design.** Same test set, same items, per-item predictions retained; $k$ seeds per arm; predictions saved to disk before any analysis.
4. **Analysis.** Paired: for each item, $d_i = \bar{\mathbf 1}\{B\text{ right}\} - \bar{\mathbf 1}\{A \text{ right}\}$ averaged over seeds; report $\bar d$ with a paired bootstrap interval resampling *items* and, separately, a seed-level interval resampling *seeds*. Report McNemar (exact) for the single-seed comparison as a secondary check. Assumptions: items i.i.d. from the benchmark distribution (false if the benchmark has near-duplicates — check); seeds i.i.d. (true by construction).
5. **Multiplicity.** If $m$ configurations are compared, Holm at $\alpha=0.05$ for the headline claim; report all $m$ results, not the best.
6. **Seeds.** Choose $k \ge 2(z_{0.975}+z_{0.80})^2\hat\sigma^2_{\text{seed}}/\Delta_{\min}^2$ using a pilot estimate of $\hat\sigma_{\text{seed}}$, and *state* $\hat\sigma_{\text{seed}}$ in the paper.
7. **Meaningful effect.** Declare $\Delta_{\min}$ before running: the smallest gap that would change a decision. Justify it (deployment threshold, label-noise floor, prior art's typical seed spread).
8. **Hygiene.** Test set touched once, at the end. All development on validation. Query counter reported.
9. **Reporting rule.** Report $\bar d$, its interval, $\hat\sigma_{\text{seed}}$, $k$, $m$, and the pre-declared $\Delta_{\min}$ — **whatever the outcome**, including "interval contains zero".

**Application, case 1 (claim warranted).** Synthetic benchmark, $n=2000$ items, single seed each, two classifiers with true accuracies $0.90$/$0.91$, agreement $95.6\%$ (seed 8675309). Pre-declared $\Delta_{\min}=0.005$. Result: $\bar d = 0.0100$, paired 95% interval $[0.00082, 0.01918]$, paired bootstrap $[0.00100, 0.01900]$, McNemar exact $p = 0.0422$. The interval excludes zero and its lower endpoint ($0.00082$) is *below* $\Delta_{\min}=0.005$. So the honest claim is: "B is better than A ($+1.00$ point, 95% CI $[0.08, 1.92]$ points, McNemar exact $p=0.042$), but the data do not establish that the improvement exceeds our pre-declared threshold of $0.5$ points." That is a weaker claim than "B beats A" and it is the one the data support.

**Application, case 2 (protocol says: do not claim).** Same true accuracies, same $n$, but weaker agreement ($\phi = 0.4251$): $b = 89$, $c = 109$, McNemar exact $p = 0.1768$, paired interval width $0.02757$ and containing zero. The protocol's rule 9 requires reporting this: "$+1.00$ point, 95% CI $[-0.004, 0.024]$, $p = 0.18$; we cannot distinguish these models on this benchmark." Note that the *true* gap is identical to case 1 — one point — and the difference is entirely in how much the models' errors overlap. Two papers reporting the same real improvement can legitimately reach opposite conclusions, which is why the interval and not the verdict is the reportable quantity.

**Application, case 3 (seed budget forbids the question).** Suppose the pilot gives $\hat\sigma_{\text{seed}} = 0.8$ points and $\Delta_{\min} = 0.2$ points. Rule 6 demands $k\ge 252$ seeds per arm. If your compute budget is 5 seeds, your resolution at $80\%$ power is $\sqrt{2\times7.8489\times0.64/5} = 1.42$ points, so the protocol's conclusion is that **this experiment cannot address a $0.2$-point question**, and the paper should either raise $\Delta_{\min}$, reduce $\sigma_{\text{seed}}$, or report the comparison as inconclusive with the resolution stated. Writing "$B$ achieves $76.4$ vs $76.2$" with one seed each is not a weaker version of this analysis; it is a different claim, and an unsupported one.

### References and further reading

**Textbooks and monographs.** E. L. Lehmann and J. P. Romano, *Testing Statistical Hypotheses*, 3rd ed. — the definitive treatment; Ch. 3 for Neyman–Pearson and monotone likelihood ratio, Ch. 5 for unbiasedness and invariance, Ch. 15 for permutation and randomization tests. G. Casella and R. L. Berger, *Statistical Inference*, 2nd ed., Chs. 8–9 — the gentlest correct account of testing and interval estimation, including the duality theorem. A. W. van der Vaart, *Asymptotic Statistics* — Ch. 16 for likelihood ratio asymptotics, Ch. 19 for empirical processes, Ch. 23 for the bootstrap and the functional delta method; this is where the "used but not proved" entries live. L. Wasserman, *All of Statistics* — fast, honest, and good on multiple testing. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap* — still the best practical source; Chs. 12–14 on interval construction, Ch. 25 on failures. P. Hall, *The Bootstrap and Edgeworth Expansion* — why studentized and BCa intervals are second-order accurate. E. L. Lehmann, *Nonparametrics: Statistical Methods Based on Ranks* — rank and permutation tests. S. Boucheron, G. Lugosi, and P. Massart, *Concentration Inequalities* — for DKW, Massart's constant, and the tools behind uniform bands. A. Gelman and J. Hill (and later Gelman et al., *Bayesian Data Analysis*) for the Bayesian counterpart to Chapter 10.

**Papers, especially on machine-learning practice.** Y. Benjamini and Y. Hochberg (1995), "Controlling the false discovery rate", *JRSS-B* 57:289–300 — the original, and short. Y. Benjamini and D. Yekutieli (2001), *Ann. Statist.* 29:1165–1188 — PRDS and the logarithmic correction. J. D. Storey (2002), *JRSS-B* 64:479–498 — adaptive FDR and the $q$-value. T. G. Dietterich (1998), "Approximate statistical tests for comparing supervised classification learning algorithms", *Neural Computation* 10:1895–1923 — where the $5\times2$-cv test and the case for McNemar come from. Y. Bengio and Y. Grandvalet (2004), "No unbiased estimator of the variance of $K$-fold cross-validation", *JMLR* 5:1089–1105. C. Nadeau and Y. Bengio (2003), "Inference for the generalization error", *Machine Learning* 52:239–281 — the corrected resampled $t$-test. J. Demšar (2006), "Statistical comparisons of classifiers over multiple data sets", *JMLR* 7:1–30 — the Friedman/Nemenyi protocol for many-datasets comparisons. C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, and A. Roth (2015), "The reusable holdout: preserving validity in adaptive data analysis", *Science* 349:636–638, and the companion STOC paper — the theory behind §9.6.4. A. Gelman and E. Loken (2013), "The garden of forking paths" — why no explicit multiple testing is needed to inflate error rates. R. L. Wasserstein and N. A. Lazar (2016), "The ASA's statement on $p$-values", *The American Statistician* 70:129–133. X. Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", *MLSys* — measures $\sigma_{\text{seed}}$ across real pipelines and argues for randomizing more than the seed. R. Dror, G. Baumer, S. Shlomov, and R. Reichart (2018), "The hitchhiker's guide to testing statistical significance in natural language processing", *ACL* — a practical decision tree for choosing a test. J. A. Hanley and B. J. McNeil (1982), *Radiology* 143:29–36, and E. R. DeLong et al. (1988), *Biometrics* 44:837–845 — AUC variance and paired AUC comparisons. D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky (2020), "With little power comes great responsibility", *EMNLP* — power analysis for NLP experiments, and the paper to hand to anyone reporting a $0.2$-point gain. A. Agresti and B. A. Coull (1998), "Approximate is better than 'exact' for interval estimation of binomial proportions", *The American Statistician* 52:119–126 — the add-two-successes-and-two-failures interval recommended in §9.5.4, and a readable account of why Wald fails. L. D. Brown, T. T. Cai, and A. DasGupta (2001), "Interval estimation for a binomial proportion", *Statistical Science* 16:101–133 — the definitive coverage comparison behind Worked Example 9.16. S. R. Howard, A. Ramdas, J. McAuliffe, and J. Sekhon (2021), "Time-uniform, nonparametric, nonasymptotic confidence sequences", *Annals of Statistics* 49:1055–1080, and A. Ramdas, P. Grünwald, V. Vovk, and G. Shafer (2023), "Game-theoretic statistics and safe anytime-valid inference", *Statistical Science* 38:576–601 — the confidence sequences and $e$-values of §9.4.3, the principled way to run the ever-monitored experiments ML actually runs. C. Jennison and B. W. Turnbull, *Group Sequential Methods with Applications to Clinical Trials* — the $\alpha$-spending boundaries of Lab B2. D. N. Politis, J. P. Romano, and M. Wolf, *Subsampling* (1999) — the repair for the bootstrap's boundary and non-regular failures in §9.8.3. D. Lakens (2017), "Equivalence tests: a practical primer", *Social Psychological and Personality Science* 8:355–362 — TOST as introduced in §9.1, with worked conventions for choosing the margin.
