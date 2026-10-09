# Chapter 4 — Inequalities and Concentration

## Chapter 4 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 4.1 Jensen's inequality | Convexity turns "expectation of a function" into "function of an expectation," in a known direction | $D_{\mathrm{KL}}\ge 0$; the ELBO is a genuine lower bound; log-likelihood estimators are biased downward |
| 4.2 Markov and Chebyshev | One moment buys a $1/t$ tail; two moments buy $1/t^2$ | The baseline every fancier bound is measured against; why $\delta$-dependence of $O(1/\sqrt{\delta})$ is unacceptable |
| 4.3 The Chernoff method | Apply Markov to $e^{\lambda X}$, then optimize $\lambda$ | The single technique behind every exponential tail bound in learning theory |
| 4.4 Sub-Gaussian and sub-exponential | A vocabulary for "tails at least as light as Gaussian / exponential" | Bounded losses are sub-Gaussian, so test-set error concentrates; gradient noise is usually only sub-exponential |
| 4.5 Hoeffding, Bernstein, Bennett | Exponential bounds for sums, with and without variance information; and how much slack is left | Confidence intervals for test error; why rare-event estimates need Bernstein, not Hoeffding |
| 4.6 Bounded differences | Concentration for any function that no single coordinate can move much | Stability of learning algorithms, cross-validation, differential privacy noise scales |
| 4.7 Finite classes | Hoeffding + union bound $\Rightarrow$ uniform convergence with $\log\lvert\mathcal H\rvert$ | The first real generalization bound; also the first to break |
| 4.8 Rademacher complexity | Replace counting hypotheses with measuring how well the class fits noise | Norm-based bounds for linear models and neural networks |
| 4.9 VC dimension | Combinatorial richness: how many labelings a class can realize | Uniform convergence for infinite classes; a useful bound for the empirical CDF, a useless one for networks |
| 4.10 Covering numbers and chaining | Approximate the class at many scales at once | Sharp rates for nonparametric classes; the tool behind modern margin bounds |
| 4.11 Why the bounds are vacuous for deep nets | Honest accounting of a real computation | What uniform convergence can and cannot explain; why this is an open problem |

## 4.0 Orientation: from "eventually" to "how much data"

Chapter 2 gave you the Lebesgue integral and with it the expectation $\mathbb{E}[X]$, defined for any random variable with $\mathbb{E}\lvert X\rvert<\infty$ regardless of whether $X$ has a density, a mass function, or neither. Chapter 3 gave you specific distributions to compute with, and in particular the moment generating function of an exponential family. This chapter answers the question that makes all of that machinery pay off in machine learning: *given $n$ samples, how far can an average be from its mean?*

That question has a qualitative answer you already know and a quantitative answer you do not. The qualitative answer is the law of large numbers: sample averages converge to expectations. It is a statement about a limit, and it is useless for engineering, because it does not tell you whether $n = 100$ or $n = 10^{8}$ is enough. Every claim in applied machine learning that has the shape "our model achieves $92.3\%$ accuracy" is a claim about a sample average — the average of a bounded loss over a test set — being close to a population quantity. The gap between those two numbers is exactly what this chapter bounds. When someone reports test accuracy on $10{,}000$ held-out examples without an error bar, this chapter tells you the error bar they omitted is roughly $\pm 1.4$ percentage points at $95\%$ confidence, and it tells you why that number and not another.

The technical arc is short and it repeats. Every bound in this chapter, with the sole exception of Jensen's inequality, is an application of one idea: **apply Markov's inequality to a cleverly transformed random variable.** Markov applied to $\lvert X-\mathbb{E}X\rvert^2$ gives Chebyshev. Markov applied to $e^{\lambda X}$ gives Chernoff, and then Hoeffding, Bernstein, Bennett, Azuma, and McDiarmid are all Chernoff plus a bound on the moment generating function. If you understand §4.2 and §4.3 you understand the mechanics of the entire chapter; §4.4 onwards is bookkeeping about which functions have which MGF bounds, and §4.7 onwards is bookkeeping about how many events you need to control at once.

What you should bring from earlier chapters: expectation as a Lebesgue integral, and the tail formula $\mathbb{E}[X]=\int_0^\infty \mathbb{P}(X>t)\,dt$ for non-negative $X$ (Ch. 2) — we will use it so often that you should be able to reproduce its proof from Tonelli's theorem on the spot. Fubini–Tonelli itself (Ch. 2), for interchanging expectations over a dataset with expectations over independent noise. Product measures, and in particular the fact that a dataset of $n$ i.i.d. samples has law $P^{\otimes n}$ on $\mathcal{Z}^n$ (Ch. 2) — this is the probability space on which every generalization bound lives. Moment generating functions and the exponential family (Ch. 3). $L^p$ spaces and the norms $\lVert\cdot\rVert_p$ (Ch. 2). From Chapter 1, by number: Proposition 1.8(2), the union bound, which is the load-bearing beam of §4.7 and the reason $\log\lvert\mathcal H\rvert$ appears in every finite-class bound; Definition 1.23 for independence; and the warning from Chapter 1 §1.8.3 that a supremum over an *uncountable* family of random variables need not be measurable, which is exactly the object $\sup_{h\in\mathcal H}\lvert\hat R_n(h)-R(h)\rvert$ that §4.7 onwards is about.

Here is the punchline, stated in advance so you can watch it assemble. For a *single* fixed hypothesis, the deviation between training error and true error is $O(\sqrt{\log(1/\delta)/n})$, and this is essentially optimal — Hoeffding gives it and the central limit theorem says you cannot do better than $\sqrt{\cdot/n}$. For a *finite* class of hypotheses you pay $\log\lvert\mathcal H\rvert$ inside the square root, which is a bargain: a million hypotheses cost you a factor of under four. For an *infinite* class you must replace counting with a geometric measure of complexity — Rademacher complexity, VC dimension, or metric entropy — and for the classes we actually use in deep learning, every such measure that anyone has computed produces a bound larger than $1$ on a probability, which is to say, no information at all. Section 4.11 computes one of those vacuous bounds explicitly for a small multilayer perceptron: the bound is $32.7$ where the truth is about $0.02$. That factor of roughly $1600$ is not a slack constant to be tightened by a cleverer proof; there are results suggesting no bound of this *form* can work. I will not pretend otherwise, and I will not offer you a resolution, because there isn't a settled one.

Two habits to build as you read. First, every inequality in this chapter has a *regime* in which it is nearly tight and a regime in which it is garbage, and the skill worth acquiring is knowing which is which — so every major bound here comes with a numerical comparison against the exact answer. Second, watch what each bound *charges you for*. Markov charges nothing and gives nothing. Hoeffding charges boundedness. Bernstein charges variance knowledge and refunds you enormously when the variance is small. McDiarmid charges you a Lipschitz-in-each-coordinate condition and buys concentration for functions far more complicated than sums. That accounting — what assumption bought what improvement — is the whole content of the subject.

Every numerical value reported in this chapter was computed by running code; simulation sizes and seeds are stated where relevant, and results are reported to a fixed number of decimals.
## 4.1 Jensen's inequality

### 4.1.1 The idea in one picture

Take a random variable and average it. Now take the same random variable, bend it through a function, and average that. Jensen's inequality says that if the function is *convex* — curving upward, like $x^2$ or $e^x$ or $-\log x$ — then bending before averaging gives you something at least as large as averaging before bending:
$$\mathbb{E}[\varphi(X)]\ \ge\ \varphi(\mathbb{E}[X]).$$

The reason is visible in a single picture and has nothing to do with probability. A convex function lies above every one of its tangent lines. A tangent line is *affine*, and affine functions commute with expectation exactly: $\mathbb{E}[aX+b]=a\mathbb{E}[X]+b$. So draw the tangent line at the point $\mu=\mathbb{E}[X]$. Since $\varphi\ge$ tangent everywhere, averaging preserves that, and averaging the tangent gives you its value at $\mu$, which is $\varphi(\mu)$ because the tangent touches there. That is the entire proof. Everything else is making "tangent line" precise for convex functions that might not be differentiable.

Equivalently, and this is the version to hold in your head for ML: **a convex function of an average is smaller than the average of the convex function, and the gap is a variance-like quantity that measures how spread out $X$ is.** When you see a bound in a paper that says "by Jensen," the author has replaced a spread-out random object by its mean and pocketed the difference in a known direction.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" role="img" aria-label="A convex parabola with a chord above it and a tangent line below it, illustrating the Jensen gap">
  <line x1="90" y1="320" x2="660" y2="320" stroke="currentColor" stroke-width="1"/>
  <line x1="360" y1="345" x2="360" y2="45" stroke="currentColor" stroke-width="0.5" stroke-dasharray="2,3"/>
  <polyline fill="none" stroke="currentColor" stroke-width="2" points="90.0,62.9 103.5,87.9 117.0,111.7 130.5,134.2 144.0,155.4 157.5,175.4 171.0,194.0 184.5,211.4 198.0,227.4 211.5,242.2 225.0,255.7 238.5,267.9 252.0,278.9 265.5,288.5 279.0,296.9 292.5,303.9 306.0,309.7 319.5,314.2 333.0,317.4 346.5,319.4 360.0,320.0 373.5,319.4 387.0,317.4 400.5,314.2 414.0,309.7 427.5,303.9 441.0,296.9 454.5,288.5 468.0,278.9 481.5,267.9 495.0,255.7 508.5,242.2 522.0,227.4 535.5,211.4 549.0,194.0 562.5,175.4 576.0,155.4 589.5,134.2 603.0,111.7 616.5,87.9 630.0,62.9"/>
  <line x1="225" y1="255.7" x2="630" y2="62.9" stroke="#3b82f6" stroke-width="2"/>
  <line x1="150" y1="320" x2="640" y2="320" stroke="#10b981" stroke-width="2" stroke-dasharray="6,4"/>
  <line x1="360" y1="320" x2="360" y2="191.4" stroke="#ef4444" stroke-width="3"/>
  <circle cx="225" cy="255.7" r="4" fill="currentColor"/>
  <circle cx="630" cy="62.9" r="4" fill="currentColor"/>
  <circle cx="360" cy="191.4" r="4" fill="#3b82f6"/>
  <circle cx="360" cy="320" r="4" fill="#10b981"/>
  <text x="640" y="55" font-size="13" fill="currentColor">φ(x) = x²</text>
  <text x="228" y="248" font-size="12" fill="currentColor">x = −1, p = 2/3</text>
  <text x="560" y="80" font-size="12" fill="currentColor">x = 2, p = 1/3</text>
  <text x="370" y="188" font-size="13" fill="#3b82f6">E[φ(X)] = 2</text>
  <text x="370" y="336" font-size="13" fill="#10b981">φ(E[X]) = 0</text>
  <text x="252" y="215" font-size="13" fill="#ef4444">Jensen gap = 2</text>
  <text x="200" y="308" font-size="12" fill="#10b981">supporting line at μ = 0</text>
  <text x="90" y="365" font-size="12.5" fill="currentColor">Figure 4.1. X takes the value −1 with probability 2/3 and 2 with probability 1/3, so E[X] = 0. The chord (blue)</text>
  <text x="90" y="381" font-size="12.5" fill="currentColor">through the two atoms sits at height E[φ(X)] = 2 above μ; the supporting line (green dashed) at μ sits at height</text>
  <text x="90" y="397" font-size="12.5" fill="currentColor">φ(μ) = 0. Convexity forces the curve between them, so E[φ(X)] ≥ φ(E[X]); the red segment is the gap.</text>
</svg>

### 4.1.2 Convexity and supporting lines

**Definition 4.1** (Convex function). Let $I\subseteq\mathbb{R}$ be an interval. A function $\varphi:I\to\mathbb{R}$ is *convex* if for all $x,y\in I$ and all $t\in[0,1]$,
$$\varphi(tx+(1-t)y)\ \le\ t\varphi(x)+(1-t)\varphi(y).$$
It is *strictly convex* if the inequality is strict whenever $x\ne y$ and $t\in(0,1)$. A function $\psi$ is *concave* if $-\psi$ is convex. For $\varphi:\mathbb{R}^d\to\mathbb{R}$ the same definition applies verbatim with $x,y\in\mathbb{R}^d$.

Geometrically: the chord lies above the graph. Note that convexity is a statement about all pairs of points, not about derivatives; $\varphi$ need not be differentiable (think $\varphi(x)=\lvert x\rvert$).

**Lemma 4.2** (Supporting line). Let $\varphi$ be convex on an interval $I$ and let $x_0$ be a point in the interior of $I$. Then there exists $c\in\mathbb{R}$ (a *subgradient* of $\varphi$ at $x_0$) such that
$$\varphi(x)\ \ge\ \varphi(x_0)+c\,(x-x_0)\qquad\text{for all }x\in I.$$
If $\varphi$ is strictly convex, the inequality is strict for $x\ne x_0$.

*Proof.* First, the **three-chord inequality**: for $u<v<w$ in $I$,
$$\frac{\varphi(v)-\varphi(u)}{v-u}\ \le\ \frac{\varphi(w)-\varphi(u)}{w-u}\ \le\ \frac{\varphi(w)-\varphi(v)}{w-v}.$$
To see the first inequality, write $v=tu+(1-t)w$ with $t=\frac{w-v}{w-u}\in(0,1)$. Convexity gives $\varphi(v)\le t\varphi(u)+(1-t)\varphi(w)$, hence
$$\varphi(v)-\varphi(u)\le (1-t)\big(\varphi(w)-\varphi(u)\big)=\frac{v-u}{w-u}\big(\varphi(w)-\varphi(u)\big),$$
and dividing by $v-u>0$ gives it. The second inequality follows from the same computation rearranged, subtracting $\varphi(w)$ instead of $\varphi(u)$.

Consequently the difference quotient $x\mapsto \frac{\varphi(x)-\varphi(x_0)}{x-x_0}$ is non-decreasing on $I\setminus\{x_0\}$. Therefore
$$c^-:=\sup_{x<x_0}\frac{\varphi(x)-\varphi(x_0)}{x-x_0}\ \le\ \inf_{x>x_0}\frac{\varphi(x)-\varphi(x_0)}{x-x_0}=:c^+ ,$$
and both are finite because $x_0$ is interior (so there are points on both sides). Pick any $c\in[c^-,c^+]$. For $x>x_0$, $c\le \frac{\varphi(x)-\varphi(x_0)}{x-x_0}$ gives $\varphi(x)\ge\varphi(x_0)+c(x-x_0)$; for $x<x_0$ the same rearrangement works because dividing by the negative number $x-x_0$ flips the inequality. For strict convexity, strict inequality in Definition 4.1 makes the difference quotient strictly increasing, so the two displayed suprema/infima are attained only in the limit and the supporting-line inequality is strict off $x_0$. $\blacksquare$

The multivariate version is identical in content — a convex function on $\mathbb{R}^d$ has a supporting *hyperplane* $\varphi(x)\ge\varphi(x_0)+\langle c,x-x_0\rangle$ at every interior point of its domain — but the existence proof needs a separating-hyperplane argument rather than monotone difference quotients. We will use the multivariate version only in passing, and it is listed in §4.12.

### 4.1.3 Jensen's inequality

**Theorem 4.3** (Jensen). Let $X$ be a random variable taking values in an interval $I$ with $\mathbb{E}\lvert X\rvert<\infty$, and let $\varphi:I\to\mathbb{R}$ be convex. Then $\mathbb{E}[\varphi(X)]$ is well defined in $(-\infty,+\infty]$ and
$$\mathbb{E}[\varphi(X)]\ \ge\ \varphi(\mathbb{E}[X]).$$
If $\varphi$ is strictly convex and $\mathbb{E}[\varphi(X)]<\infty$, equality holds if and only if $X=\mathbb{E}[X]$ a.s.

*Proof.* Write $\mu=\mathbb{E}[X]$; since $X\in I$ a.s. and $I$ is an interval, $\mu\in I$.

*Case 1: $\mu$ is an endpoint of $I$.* Say $\mu=\inf I$. Then $X\ge\mu$ a.s. and $\mathbb{E}[X-\mu]=0$, so by the fact that a non-negative random variable with zero mean is zero a.s. (Ch. 2), $X=\mu$ a.s., and both sides equal $\varphi(\mu)$.

*Case 2: $\mu$ is interior.* By Lemma 4.2 there is $c$ with $\varphi(x)\ge \varphi(\mu)+c(x-\mu)$ for all $x\in I$. Substituting $X$:
$$\varphi(X)\ \ge\ \varphi(\mu)+c(X-\mu)\qquad\text{a.s.}$$
The right-hand side is integrable, so the negative part of $\varphi(X)$ is integrable and $\mathbb{E}[\varphi(X)]$ is well defined in $(-\infty,+\infty]$. Taking expectations and using linearity and monotonicity of the Lebesgue integral (Ch. 2),
$$\mathbb{E}[\varphi(X)]\ \ge\ \varphi(\mu)+c\,\big(\mathbb{E}[X]-\mu\big)=\varphi(\mu).$$
For the equality case, let $D:=\varphi(X)-\varphi(\mu)-c(X-\mu)\ge0$. Equality in the theorem means $\mathbb{E}[D]=0$, hence $D=0$ a.s. But strict convexity makes the supporting-line inequality strict at every $x\ne\mu$ (Lemma 4.2), so $D>0$ off $\{X=\mu\}$; therefore $\mathbb{P}(X\ne\mu)=0$. $\blacksquare$

Two remarks worth internalizing. First, the direction is dictated entirely by convexity: for *concave* $\psi$ the inequality reverses, $\mathbb{E}[\psi(X)]\le\psi(\mathbb{E}[X])$. Since $\log$ is concave, $\mathbb{E}[\log X]\le\log\mathbb{E}[X]$ — this single line is responsible for most appearances of Jensen in machine learning, because log-likelihoods are everywhere. Second, no independence, no distributional assumption, and no finiteness of variance is needed. Jensen is a statement about a single random variable and a convex function.

> **Common Confusion 4.1**
>
> *"I can never remember which way Jensen goes, so I just write $\mathbb{E}[f(X)]\le f(\mathbb{E}[X])$ and hope."*
>
> Don't memorize; recover it in five seconds from a case you cannot get wrong. Take $X$ to be $\pm1$ with equal probability, so $\mathbb{E}[X]=0$. For $\varphi(x)=x^2$: $\mathbb{E}[X^2]=1$ and $\varphi(\mathbb{E}[X])=0$. So for convex $\varphi$, the *expectation of the function is bigger*. That's it. Now flip the sign for concave functions: $\mathbb{E}[\log X]\le\log\mathbb{E}[X]$, which you can also check with $X\in\{1,4\}$ equally likely: $\mathbb{E}[\log X]=\tfrac12\log 4=0.693$ while $\log \mathbb{E}[X]=\log 2.5=0.916$.
>
> The related trap is assuming $\mathbb{E}[1/X]=1/\mathbb{E}[X]$. Since $x\mapsto 1/x$ is convex on $(0,\infty)$, Jensen gives $\mathbb{E}[1/X]\ge 1/\mathbb{E}[X]$, with the gap large when $X$ is sometimes near zero. This is exactly why the *harmonic* mean of learning rates or of per-batch normalizers is not the reciprocal of the arithmetic mean, and why inverse-propensity weighting estimators have heavy tails.

**Worked Example 4.1** (The exponential gap, and why log-likelihood estimates are biased low). Let $X\sim N(0,1)$ and $\varphi(x)=e^x$, which is strictly convex. Then $\mathbb{E}[e^X]=e^{1/2}=1.6487$ (this is the Gaussian MGF at $\lambda=1$, Ch. 3), while $e^{\mathbb{E}[X]}=e^0=1$. The Jensen gap is $0.6487$ — a $65\%$ multiplicative error, from a variable whose standard deviation is only $1$. As a check, a Monte Carlo average of $e^{X}$ over $10^7$ draws with seed $0$ gives $1.6476$, consistent with $1.6487$ to within the $\pm 0.0007$ Monte Carlo standard error.

Now the ML consequence. In variational inference and in normalizing-flow/VAE evaluation, one estimates the marginal likelihood $p_\theta(x)=\mathbb{E}_{z\sim q}[w(z)]$ where $w(z)=p_\theta(x,z)/q(z)$ is an importance weight, and then reports $\log$ of the estimate. Let $\hat p=\frac1K\sum_{k=1}^K w(z_k)$, an unbiased estimator of $p_\theta(x)$. Because $\log$ is concave, Theorem 4.3 gives
$$\mathbb{E}\big[\log\hat p\big]\ \le\ \log\mathbb{E}[\hat p]=\log p_\theta(x).$$
So the reported number is a *lower bound in expectation*, never an over-estimate — reassuring for honest reporting, but it also means an under-trained proposal $q$ makes your model look worse than it is, and increasing $K$ monotonically tightens the bound. That monotonicity is precisely the content of the importance-weighted autoencoder objective; Chapter 9 revisits variational bounds in earnest.

**Worked Example 4.2** (Geometric versus arithmetic mean). Let $X$ be uniform on $\{1,2,3\}$. Then $\mathbb{E}[\log X]=\tfrac13(\log1+\log2+\log3)=0.597253$, so the geometric mean is $e^{0.597253}=1.817121$, while the arithmetic mean is $2$. Jensen for the concave $\log$ predicted exactly this ordering. The same computation with a different label is the *log-sum-exp* inequality: for any reals $a_1,\dots,a_m$ and weights $\pi_i\ge0$ summing to $1$,
$$\sum_i \pi_i a_i\ \le\ \log\sum_i \pi_i e^{a_i}\quad\text{(convexity of }\exp\text{)},$$
which is why `logsumexp` of a batch of logits dominates their weighted average, and why softmax-based losses are convex in the logits.

**Corollary 4.4** (Gibbs' inequality: non-negativity of KL). Let $P,Q$ be probability measures on a common space with $P\ll Q$, and let $p=\frac{dP}{dQ}$ be the Radon–Nikodym derivative (Ch. 2). Then
$$D_{\mathrm{KL}}(P\,\Vert\,Q):=\int \log\frac{dP}{dQ}\,dP=\mathbb{E}_Q\big[p\log p\big]\ \ge\ 0,$$
with equality if and only if $P=Q$. (If $P\not\ll Q$ we set $D_{\mathrm{KL}}(P\Vert Q)=+\infty$ by convention, so the conclusion holds in all cases.)

*Proof.* Let $\varphi(u)=u\log u$ on $[0,\infty)$, with $\varphi(0):=0$; it is continuous and strictly convex, since $\varphi''(u)=1/u>0$ on $(0,\infty)$. Under $Q$, the random variable $p$ satisfies $\mathbb{E}_Q[p]=\int p\,dQ=P(\text{whole space})=1$. Theorem 4.3 gives
$$D_{\mathrm{KL}}(P\Vert Q)=\mathbb{E}_Q[\varphi(p)]\ \ge\ \varphi\big(\mathbb{E}_Q[p]\big)=\varphi(1)=0.$$
Strict convexity gives equality iff $p=1$ $Q$-a.s., i.e. $P=Q$. $\blacksquare$

**Worked Example 4.3** (KL between coins; asymmetry; a preview of Pinsker). For Bernoulli laws, $D_{\mathrm{KL}}(\mathrm{Ber}(p)\Vert \mathrm{Ber}(q))=p\log\frac pq+(1-p)\log\frac{1-p}{1-q}$. Computed values:

| $p$ | $q$ | $D_{\mathrm{KL}}(\mathrm{Ber}(p)\Vert\mathrm{Ber}(q))$ | $2(p-q)^2$ |
|---|---|---|---|
| $0.9$ | $0.5$ | $0.368064$ | $0.32$ |
| $0.5$ | $0.9$ | $0.510826$ | $0.32$ |
| $0.6$ | $0.5$ | $0.020136$ | $0.02$ |
| $0.51$ | $0.5$ | $0.000200013$ | $0.0002$ |

Three things to notice. It is non-negative, as Corollary 4.4 demands. It is *not symmetric*: $0.368064\ne0.510826$. And in every row $D_{\mathrm{KL}}\ge 2(p-q)^2$, which is **Pinsker's inequality** ($D_{\mathrm{KL}}(P\Vert Q)\ge2\,d_{\mathrm{TV}}(P,Q)^2$) specialized to coins; the last two rows show it is nearly tight for $p\approx q$. Hold on to the number $2(p-q)^2$: it is exactly the exponent that will appear in Hoeffding's inequality in §4.5, and that is not a coincidence — Hoeffding's bound for coins is Pinsker's inequality applied to the exact Chernoff exponent, as Worked Example 4.7 will show.

The asymmetry has a practical face. Fitting $q$ by minimizing $D_{\mathrm{KL}}(P\Vert Q)$ (the maximum-likelihood direction, "forward KL") penalizes $q$ heavily for putting *low* mass where $P$ has mass, so $q$ spreads out and covers all modes. Minimizing $D_{\mathrm{KL}}(Q\Vert P)$ (the variational direction, "reverse KL") penalizes $q$ for putting mass where $P$ has none, so $q$ collapses onto one mode. Mode-covering versus mode-seeking is a corollary of which argument the logarithm sees.

**Corollary 4.5** (Weighted AM–GM). For $x_1,\dots,x_m>0$ and weights $\pi_i\ge0$ with $\sum_i\pi_i=1$,
$$\prod_i x_i^{\pi_i}\ \le\ \sum_i \pi_i x_i .$$

*Proof.* Apply Theorem 4.3 to the concave function $\log$ and the random variable $X$ with $\mathbb{P}(X=x_i)=\pi_i$: $\sum_i \pi_i\log x_i\le\log\sum_i\pi_i x_i$. Exponentiate. $\blacksquare$

**Corollary 4.6** (Lyapunov's moment inequality). For $0<p\le q$ and any random variable $X$,
$$\big(\mathbb{E}\lvert X\rvert^p\big)^{1/p}\ \le\ \big(\mathbb{E}\lvert X\rvert^q\big)^{1/q},$$
i.e. $\lVert X\rVert_p\le\lVert X\rVert_q$: the $L^p$ norms of Chapter 2 increase in $p$ on a probability space. In particular $\big(\mathbb{E}\lvert X\rvert\big)^2\le\mathbb{E}[X^2]$, so $\mathrm{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}X)^2\ge0$.

*Proof.* Let $Y=\lvert X\rvert^p\ge0$ and $\varphi(u)=u^{q/p}$, convex on $[0,\infty)$ because $q/p\ge1$. Theorem 4.3 gives $\mathbb{E}[Y^{q/p}]\ge(\mathbb{E}Y)^{q/p}$, that is $\mathbb{E}\lvert X\rvert^{q}\ge(\mathbb{E}\lvert X\rvert^p)^{q/p}$. Raise both sides to the power $1/q>0$. $\blacksquare$

This is the reason a bound on a high moment is stronger than a bound on a low one, and the reason the "moment growth" characterization of sub-Gaussianity in §4.4 controls everything below it for free.

### 4.1.4 The ELBO is a lower bound because $\log$ is concave

Chapter 2 promised that the variational lower bound used to train VAEs and to do approximate Bayesian inference is a *bound*, not an approximation. Here is the discharge of that promise.

**Proposition 4.7** (Evidence lower bound). Let $p_\theta(x,z)$ be a joint density with respect to a product base measure, with marginal $p_\theta(x)=\int p_\theta(x,z)\,dz>0$, and let $q(z)$ be any density with $q\ll p_\theta(\cdot\mid x)$. Define
$$\mathcal{L}(q,\theta;x):=\mathbb{E}_{z\sim q}\Big[\log p_\theta(x,z)-\log q(z)\Big].$$
Then
$$\log p_\theta(x)\ =\ \mathcal{L}(q,\theta;x)\ +\ D_{\mathrm{KL}}\big(q\,\Vert\,p_\theta(\cdot\mid x)\big)\ \ge\ \mathcal{L}(q,\theta;x),$$
with equality if and only if $q=p_\theta(\cdot\mid x)$ almost everywhere.

*Proof.* Write $p_\theta(x,z)=p_\theta(x)\,p_\theta(z\mid x)$, valid wherever $q>0$. Then
$$\mathcal{L}(q,\theta;x)=\mathbb{E}_q\Big[\log p_\theta(x)+\log p_\theta(z\mid x)-\log q(z)\Big]=\log p_\theta(x)-\mathbb{E}_q\Big[\log\frac{q(z)}{p_\theta(z\mid x)}\Big],$$
because $\log p_\theta(x)$ is a constant with respect to $z$ and $q$ integrates to one. The subtracted term is $D_{\mathrm{KL}}(q\Vert p_\theta(\cdot\mid x))$, which is $\ge0$ with equality iff the two agree, by Corollary 4.4. $\blacksquare$

Notice that the *identity* is pure algebra; Jensen enters only through Corollary 4.4 to sign the remainder. The alternative route — $\log p_\theta(x)=\log\mathbb{E}_q[p_\theta(x,z)/q(z)]\ge\mathbb{E}_q[\log(p_\theta(x,z)/q(z))]$ by concavity of $\log$ — gives the inequality in one line but hides the remainder term, and the remainder is the interesting part: it tells you the bound is loose exactly to the extent that your approximate posterior is wrong.

**Worked Example 4.4** (Watching the ELBO gap close). Take a two-state latent: $z\in\{1,2\}$ with prior $p(z)=(0.5,0.5)$, and an observation $x$ with likelihoods $p(x\mid z{=}1)=0.2$, $p(x\mid z{=}2)=0.8$. Then $p(x)=0.5\cdot0.2+0.5\cdot0.8=0.5$, so $\log p(x)=-0.693147$, and the exact posterior is $p(z\mid x)=(0.2,\,0.8)$ by Bayes' rule (Proposition 1.12). Now evaluate the ELBO for three choices of $q$:

| $q$ | $\mathcal{L}(q)$ | $D_{\mathrm{KL}}(q\Vert p(\cdot\mid x))$ | sum |
|---|---|---|---|
| $(0.5,0.5)$ | $-0.916291$ | $0.223144$ | $-0.693147$ |
| $(0.3,0.7)$ | $-0.721315$ | $0.028168$ | $-0.693147$ |
| $(0.2,0.8)$ | $-0.693147$ | $0.000000$ | $-0.693147$ |

The third column is the price of using the wrong posterior, and the sums confirm Proposition 4.7 to six decimals. Two lessons that transfer directly to practice: maximizing the ELBO in $q$ is *exactly* minimizing the reverse KL to the true posterior, and if your variational family cannot represent the posterior, the ELBO you report understates the log-likelihood by an amount you cannot see from the ELBO alone.

A caution before we leave Jensen. Everything above is unconditional. The version you actually need for SGD analysis and martingale arguments — $\mathbb{E}[\varphi(X)\mid \mathcal{G}]\ge\varphi(\mathbb{E}[X\mid\mathcal{G}])$ a.s. — is *conditional Jensen*, and it requires conditional expectation, which is Chapter 6's subject. It is true, it is proved there, and I will not use it in this chapter.

> **Self-check 4.1**
>
> 1. $X$ is uniform on $[0,1]$. Which is larger, $\mathbb{E}[e^{-X}]$ or $e^{-\mathbb{E}[X]}$? Compute both.
> 2. Where in the proof of Theorem 4.3 did we use that $\mathbb{E}\lvert X\rvert<\infty$?
> 3. A friend claims $D_{\mathrm{KL}}(P\Vert Q)=D_{\mathrm{KL}}(Q\Vert P)$ "by symmetry of the definition." Give the two-line refutation using Worked Example 4.3.
> 4. If $\varphi$ is affine, what does Theorem 4.3 say, and why is that consistent with the equality condition?
## 4.2 Markov and Chebyshev: everything from one moment, then two

### 4.2.1 Markov's inequality

The most useful inequality in probability is also the most obvious one. If a non-negative random variable has mean $1$, it cannot spend much time above $100$: if it were above $100$ with probability $1/50$, that alone would contribute $2$ to the mean. Formalizing this counting argument takes two lines.

**Theorem 4.8** (Markov's inequality). Let $X\ge0$ be a random variable and $t>0$. Then
$$\mathbb{P}(X\ge t)\ \le\ \frac{\mathbb{E}[X]}{t}.$$

*Proof.* Since $X\ge0$, we have the pointwise inequality $X\ \ge\ t\,\mathbf{1}_{\{X\ge t\}}$: on the event $\{X\ge t\}$ the left side is at least $t$ and the right side equals $t$; off that event the right side is $0\le X$. Both sides are non-negative measurable functions, so monotonicity of the integral (Ch. 2) gives $\mathbb{E}[X]\ge t\,\mathbb{E}[\mathbf{1}_{\{X\ge t\}}]=t\,\mathbb{P}(X\ge t)$. Divide by $t>0$. $\blacksquare$

That is the whole proof, and it is worth pausing on how little it assumed: no independence, no variance, no density, not even integrability (if $\mathbb{E}[X]=\infty$ the bound is vacuous but true). The generality is the point. Every other tail bound in this chapter is Theorem 4.8 applied to a transformed variable, and the art is choosing the transformation.

**Corollary 4.9** (Chebyshev's inequality). Let $X$ have $\mathbb{E}[X^2]<\infty$, mean $\mu$ and variance $\sigma^2$. For any $t>0$,
$$\mathbb{P}\big(\lvert X-\mu\rvert\ge t\big)\ \le\ \frac{\sigma^2}{t^2}.$$

*Proof.* The events $\{\lvert X-\mu\rvert\ge t\}$ and $\{(X-\mu)^2\ge t^2\}$ are literally the same set. Apply Theorem 4.8 to the non-negative variable $(X-\mu)^2$ at level $t^2$: $\mathbb{P}((X-\mu)^2\ge t^2)\le\mathbb{E}[(X-\mu)^2]/t^2=\sigma^2/t^2$. $\blacksquare$

The same trick with $\lvert X-\mu\rvert^k$ gives $\mathbb{P}(\lvert X-\mu\rvert\ge t)\le \mathbb{E}\lvert X-\mu\rvert^k/t^k$ for any $k\ge1$ — a *polynomial* improvement for each extra moment you know. Section 4.3 takes $k\to\infty$, in effect, by using $e^{\lambda X}$ instead, and gets an *exponential* improvement. That jump is the single most important structural fact in this chapter.

**Worked Example 4.5** (How weak is Markov, exactly?). Let $X\sim\mathrm{Exp}(1)$, so $\mathbb{E}[X]=1$ and $\mathbb{P}(X\ge t)=e^{-t}$.

| $t$ | true $\mathbb{P}(X\ge t)$ | Markov bound $1/t$ | ratio |
|---|---|---|---|
| $2$ | $0.135335$ | $0.500000$ | $3.7$ |
| $5$ | $0.006738$ | $0.200000$ | $29.7$ |
| $10$ | $0.0000454$ | $0.100000$ | $2203$ |

Markov decays like $1/t$; the truth decays like $e^{-t}$. The bound gets *relatively* worse without limit. And yet Markov cannot be improved as a general statement, which Proposition 4.10 makes precise.

**Proposition 4.10** (Markov's inequality is attained). For every $t>1$ there is a non-negative random variable $X$ with $\mathbb{E}[X]=1$ and $\mathbb{P}(X\ge t)=1/t$ exactly.

*Proof.* Let $X=t$ with probability $1/t$ and $X=0$ otherwise. Then $\mathbb{E}[X]=t\cdot(1/t)=1$ and $\mathbb{P}(X\ge t)=1/t$. $\blacksquare$

So the looseness in Worked Example 4.5 is not a defect of the proof; it is the price of using only one moment. The information "$\mathbb{E}[X]=1$" is compatible with both the exponential distribution and the two-point distribution above, and the two-point distribution really does have a $1/t$ tail. **If you want a better bound you must supply more information about the distribution.** That sentence is the design principle for the rest of the chapter: each subsequent section buys an improvement with a specific new assumption.

### 4.2.2 What Chebyshev says about sample means

**Corollary 4.11** (Chebyshev for sample means). Let $X_1,\dots,X_n$ be i.i.d. (so the dataset has law $P^{\otimes n}$, Ch. 2) with mean $\mu$ and variance $\sigma^2<\infty$, and let $\bar X_n=\frac1n\sum_{i=1}^nX_i$. Then for any $\varepsilon>0$,
$$\mathbb{P}\big(\lvert \bar X_n-\mu\rvert\ge\varepsilon\big)\ \le\ \frac{\sigma^2}{n\varepsilon^2}.$$
Equivalently, for confidence $1-\delta$ the deviation is at most $\varepsilon=\sigma/\sqrt{n\delta}$, and to guarantee $\varepsilon$ at confidence $1-\delta$ it suffices that
$$n\ \ge\ \frac{\sigma^2}{\delta\,\varepsilon^2}.$$

*Proof.* $\mathrm{Var}(\bar X_n)=\sigma^2/n$ by independence (Definition 1.23 plus bilinearity of covariance), so Corollary 4.9 applies with $t=\varepsilon$. Setting the right-hand side to $\delta$ and solving gives the other two forms. $\blacksquare$

This already proves the weak law of large numbers with a rate, and the rate $\varepsilon\asymp1/\sqrt n$ in $n$ is the correct one — the central limit theorem (Chapter 5 in this tutorial's arc; here we only need it as intuition) says the fluctuation of $\bar X_n$ is genuinely of order $\sigma/\sqrt n$ and no smaller. The problem is not the $n$-dependence. The problem is the $\delta$-dependence.

**Worked Example 4.6** (Why $\sqrt{1/\delta}$ is unaffordable and $\sqrt{\log(1/\delta)}$ is free). Estimate the bias of a coin ($\sigma^2=1/4$) to within $\varepsilon=0.01$. How many flips?

| target confidence $1-\delta$ | Chebyshev $n\ge\sigma^2/(\delta\varepsilon^2)$ | Hoeffding $n\ge\log(2/\delta)/(2\varepsilon^2)$ | CLT-based $n\approx z_{1-\delta/2}^2\sigma^2/\varepsilon^2$ |
|---|---|---|---|
| $95\%$ ($\delta=5\times10^{-2}$) | $50{,}000$ | $18{,}445$ | $9{,}604$ |
| $99.9\%$ ($\delta=10^{-3}$) | $2{,}500{,}000$ | $38{,}005$ | $27{,}069$ |
| $1-10^{-6}$ | $2{,}500{,}000{,}000$ | $72{,}544$ | $59{,}821$ |

Read the columns as costs. Chebyshev's requirement grows *linearly in $1/\delta$*: going from $95\%$ to six-nines confidence multiplies your data requirement by $50{,}000$. Hoeffding's requirement grows *logarithmically*: the same increase in confidence multiplies it by $3.9$. That is the entire practical reason exponential concentration inequalities matter. In machine learning you rarely control just one event; §4.7 will apply a tail bound simultaneously to $\lvert\mathcal H\rvert$ hypotheses at confidence $\delta/\lvert\mathcal H\rvert$ each, so the bound is *evaluated at exponentially small $\delta$ on purpose*. With Chebyshev, that costs a factor $\lvert\mathcal H\rvert$ inside the square root and the whole enterprise dies; with an exponential bound it costs $\log\lvert\mathcal H\rvert$ and the enterprise lives. The Hoeffding column is derived in §4.5; the third column is what the truth looks like asymptotically, and note that Hoeffding is within a factor of about $2$ of it — the exponential bounds are not merely qualitatively better, they are quantitatively close.

> **Common Confusion 4.2**
>
> *"Chebyshev is about the normal distribution, right? The two-sigma rule?"*
>
> No, and the difference matters. The "$95\%$ within two standard deviations" rule is a fact about the *Gaussian*. Chebyshev at $t=2\sigma$ gives only $\mathbb{P}(\lvert X-\mu\rvert\ge2\sigma)\le1/4$, i.e. at least $75\%$ within two standard deviations — much weaker, but true for *every* distribution with finite variance, including horrible ones. That universality is the value. When someone applies the two-sigma rule to a heavy-tailed quantity like a per-example loss, gradient norm, or revenue-per-user, they are assuming normality they have not checked; Chebyshev is the assumption-free fallback, and its weakness is honest.
>
> Similarly, neither Theorem 4.8 nor Corollary 4.9 requires independence anywhere. Independence enters only when we compute $\mathrm{Var}(\bar X_n)=\sigma^2/n$ in Corollary 4.11. If your samples are correlated — adjacent frames of a video, tokens in a document, two augmentations of the same image — that variance formula is wrong and every downstream bound is wrong, while Markov and Chebyshev themselves remain intact. Correlated data breaks the variance computation, not the inequality.

**A refinement worth knowing by name.** Chebyshev bounds a two-sided event, so using it for a one-sided question wastes half its strength. The one-sided repair is **Cantelli's inequality**: for $X$ with mean $0$ and variance $\sigma^2$, $\mathbb{P}(X\ge t)\le\sigma^2/(\sigma^2+t^2)$, which is never worse than $\sigma^2/t^2$ and strictly better everywhere — at $t=\sigma$ it gives $1/2$ where Chebyshev gives the vacuous $1$. The proof is a two-line optimization (shift $X$ by a constant before applying Chebyshev, then choose the constant), and it is Exercise A2, where you will also see that the improvement matters near the mean and evaporates in the far tail. Keep it in your toolkit for one-sided questions with only two moments: it is the sharpest possible bound under those hypotheses, just as Proposition 4.10 showed Markov is under one.

> **Self-check 4.2**
>
> 1. Give a random variable for which Chebyshev at $t=2\sigma$ is exactly tight.
> 2. In Corollary 4.11, what happens to the required $n$ if you halve $\varepsilon$? If you halve $\delta$?
> 3. $X\ge0$ has $\mathbb{E}[X]=3$. What does Markov say about $\mathbb{P}(X\ge2)$, and why is the answer useless?
> 4. Why is Chebyshev an instance of Markov rather than a separate theorem?

## 4.3 The Chernoff method: one procedure, four theorems

### 4.3.1 The template

Here is the idea. Markov is weak because $\mathbb{P}(X\ge t)\le\mathbb{E}[X]/t$ throws away everything about the distribution except one number. But we may apply Markov to *any* non-negative increasing transform of $X$ before comparing, and increasing transforms preserve the event. Pick the transform $x\mapsto e^{\lambda x}$ for $\lambda>0$. Then $\{X\ge t\}=\{e^{\lambda X}\ge e^{\lambda t}\}$, and Theorem 4.8 gives
$$\mathbb{P}(X\ge t)\ \le\ e^{-\lambda t}\,\mathbb{E}\big[e^{\lambda X}\big].$$
This holds for *every* $\lambda>0$, so we are free to pick the best one:
$$\mathbb{P}(X\ge t)\ \le\ \inf_{\lambda>0}\ e^{-\lambda t}\,\mathbb{E}\big[e^{\lambda X}\big].$$
Exponentiating before applying Markov is what converts polynomial decay into exponential decay, because the transform amplifies the far tail before we throw information away. The moment generating function $\mathbb{E}[e^{\lambda X}]$, which you met in Chapter 3 for exponential families, is exactly the object that knows about all moments at once.

**Definition 4.12** (MGF, CGF, Cramér transform). For a random variable $X$ let
$$M_X(\lambda):=\mathbb{E}\big[e^{\lambda X}\big]\in(0,\infty],\qquad \psi_X(\lambda):=\log M_X(\lambda),$$
the *moment generating function* and the *cumulant generating function*. The *Cramér transform* (Legendre–Fenchel conjugate of $\psi_X$) is
$$\psi_X^*(t):=\sup_{\lambda\ge0}\big(\lambda t-\psi_X(\lambda)\big).$$

**Lemma 4.13** (Basic properties of the CGF). Let $\Lambda:=\{\lambda\in\mathbb{R}:M_X(\lambda)<\infty\}$. Then:

1. $0\in\Lambda$ and $\psi_X(0)=0$;
2. $\Lambda$ is an interval and $\psi_X$ is convex on it;
3. if $X,Y$ are independent, $\psi_{X+Y}=\psi_X+\psi_Y$ on the common domain; consequently for i.i.d. $X_1,\dots,X_n$, $\psi_{S_n}(\lambda)=n\,\psi_X(\lambda)$ where $S_n=\sum_iX_i$;
4. $\psi_X^*(t)\ge0$ for all $t$, and $\psi^*_X(t)>0$ whenever $t>\mathbb{E}[X]$ and $M_X(\lambda)<\infty$ for some $\lambda>0$.

*Proof.* (1) $M_X(0)=\mathbb{E}[1]=1$.

(2) Let $\lambda=\theta\lambda_1+(1-\theta)\lambda_2$ with $\theta\in(0,1)$ and $\lambda_1,\lambda_2\in\Lambda$. Write $e^{\lambda X}=(e^{\lambda_1X})^{\theta}(e^{\lambda_2X})^{1-\theta}$ and apply Hölder's inequality (Ch. 2) with exponents $1/\theta$ and $1/(1-\theta)$:
$$M_X(\lambda)\le \big(\mathbb{E}[e^{\lambda_1X}]\big)^{\theta}\big(\mathbb{E}[e^{\lambda_2X}]\big)^{1-\theta}=M_X(\lambda_1)^\theta M_X(\lambda_2)^{1-\theta}<\infty.$$
So $\lambda\in\Lambda$ (hence $\Lambda$ is an interval) and taking logarithms gives $\psi_X(\lambda)\le\theta\psi_X(\lambda_1)+(1-\theta)\psi_X(\lambda_2)$, which is convexity.

(3) $\mathbb{E}[e^{\lambda(X+Y)}]=\mathbb{E}[e^{\lambda X}e^{\lambda Y}]=\mathbb{E}[e^{\lambda X}]\mathbb{E}[e^{\lambda Y}]$, where the factorization for independent variables follows from the product-measure form of Fubini's theorem (Ch. 2) applied to the joint law $P_X\otimes P_Y$. Take logs; iterate for $n$ terms.

(4) Taking $\lambda=0$ in the supremum gives $\psi^*_X(t)\ge0-\psi_X(0)=0$. For the strictness, note that by Jensen (Theorem 4.3, applied to the convex $e^{\lambda x}$) $\psi_X(\lambda)\ge\lambda\mathbb{E}[X]$; but more useful is the derivative computation: on the interior of $\Lambda$ one may differentiate under the integral sign (dominated convergence, Ch. 2) to get $\psi_X'(0)=\mathbb{E}[X]$. Hence for $t>\mathbb{E}[X]$ the function $\lambda\mapsto\lambda t-\psi_X(\lambda)$ has derivative $t-\mathbb{E}[X]>0$ at $\lambda=0$, so it is strictly positive for small $\lambda>0$, so the supremum is strictly positive. $\blacksquare$

**Theorem 4.14** (Chernoff–Cramér bound). For any random variable $X$ and any $t\in\mathbb{R}$,
$$\mathbb{P}(X\ge t)\ \le\ \exp\big(-\psi_X^*(t)\big)=\inf_{\lambda\ge0}\ e^{-\lambda t}M_X(\lambda).$$

*Proof.* For $\lambda>0$ the map $x\mapsto e^{\lambda x}$ is strictly increasing, so $\{X\ge t\}=\{e^{\lambda X}\ge e^{\lambda t}\}$. Apply Theorem 4.8 to the non-negative variable $e^{\lambda X}$ at level $e^{\lambda t}>0$:
$$\mathbb{P}(X\ge t)\le \frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda t}}=\exp\big(\psi_X(\lambda)-\lambda t\big).$$
Take the infimum over $\lambda>0$; including $\lambda=0$ changes nothing because it yields the trivial bound $1$. $\blacksquare$

**Proposition 4.15** (Chernoff for i.i.d. sums: the rate is linear in $n$). Let $X_1,\dots,X_n$ be i.i.d. copies of $X$, $\bar X_n$ their average, and $t>\mathbb{E}[X]$. Then
$$\mathbb{P}\big(\bar X_n\ge t\big)\ \le\ \exp\big(-n\,\psi_X^*(t)\big).$$

*Proof.* $\mathbb{P}(\bar X_n\ge t)=\mathbb{P}(S_n\ge nt)\le\inf_{\lambda\ge0}e^{-\lambda n t}M_{S_n}(\lambda)$ by Theorem 4.14. By Lemma 4.13(3), $M_{S_n}(\lambda)=e^{n\psi_X(\lambda)}$, so the bound is $\inf_{\lambda\ge0}\exp\big(-n(\lambda t-\psi_X(\lambda))\big)=\exp\big(-n\sup_{\lambda\ge0}(\lambda t-\psi_X(\lambda))\big)$. $\blacksquare$

Lemma 4.13(4) tells us $\psi^*_X(t)>0$ for $t$ above the mean, so this is genuine exponential decay in $n$ at a rate we can often compute exactly. **The procedure — write the MGF, subtract $\lambda t$, optimize $\lambda$, exponentiate — is used four more times in this chapter (Hoeffding, Bernstein, Bennett, Azuma) and you should be able to execute it from memory.**

### 4.3.2 The procedure in action

**Worked Example 4.7** (Chernoff for a fair coin, computed exactly). Let $S\sim\mathrm{Binomial}(100,\tfrac12)$; bound $\mathbb{P}(S\ge60)$. Here $X_i\in\{0,1\}$ with $M_X(\lambda)=\tfrac12(1+e^\lambda)$, so
$$\log\big(e^{-60\lambda}M_X(\lambda)^{100}\big)=-60\lambda+100\log\tfrac{1+e^\lambda}{2}.$$
Differentiate in $\lambda$: $-60+100\frac{e^\lambda}{1+e^\lambda}=0$, so $\frac{e^\lambda}{1+e^\lambda}=0.6$, giving
$$\lambda^\star=\log\frac{0.6}{0.4}=\log 1.5=0.405465 .$$
The mechanics are worth noticing: the optimal $\lambda$ is the one that *tilts* the coin from $p=0.5$ to $p=0.6$, i.e. makes the rare event typical. Substituting back and simplifying gives the exact Chernoff exponent
$$\mathbb{P}(\bar X_{100}\ge0.6)\ \le\ \exp\big(-100\,D_{\mathrm{KL}}(\mathrm{Ber}(0.6)\Vert\mathrm{Ber}(0.5))\big)=\exp(-2.013551)=0.133514 .$$
Compare all the bounds we have, plus the exact answer computed from the binomial CDF:

| method | bound on $\mathbb{P}(S\ge60)$ |
|---|---|
| Markov (on $S$, at $t=60$) | $0.833333$ |
| Chebyshev | $0.250000$ |
| Hoeffding (§4.5, $e^{-2n\varepsilon^2}$) | $0.135335$ |
| Chernoff, exact exponent | $0.133514$ |
| **truth** | $0.028444$ |

Three observations. First, the Chernoff bound is a factor of $4.7$ from the truth while Markov is off by a factor of $29$ — and the gap between Chernoff and truth is only a polynomial-in-$n$ factor (from the missing $1/\sqrt{n}$ in the central limit correction), not an exponential one; Chernoff exponents are asymptotically exact, a fact called Cramér's theorem in large-deviations theory. Second, Hoeffding's simpler bound $e^{-2n(0.1)^2}=0.135335$ is barely worse than the exact exponent $0.133514$, because $2(p-q)^2=0.02$ versus $D_{\mathrm{KL}}=0.020136$ — Hoeffding *is* Pinsker's inequality applied to the Chernoff exponent, as promised in Worked Example 4.3. Third, the whole computation used only Theorem 4.8 and one derivative.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" role="img" aria-label="The Chernoff objective as a function of lambda, with a minimum marked at lambda equals log 1.5">
  <line x1="100" y1="185" x2="660" y2="185" stroke="currentColor" stroke-width="1"/>
  <line x1="100" y1="45" x2="100" y2="330" stroke="currentColor" stroke-width="1"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="100.0,185.0 112.9,198.7 125.7,211.5 138.6,223.4 151.4,234.4 164.3,244.6 177.1,253.8 190.0,262.2 202.9,269.7 215.7,276.4 228.6,282.2 241.4,287.1 254.3,291.1 267.1,294.3 280.0,296.6 292.9,298.1 305.7,298.7 318.6,298.4 331.4,297.3 344.3,295.4 357.1,292.7 370.0,289.1 382.9,284.7 395.7,279.4 408.6,273.4 421.4,266.6 434.3,258.9 447.1,250.5 460.0,241.2 472.9,231.2 485.7,220.4 498.6,208.9 511.4,196.6 524.3,183.5 537.1,169.7 550.0,155.1 562.9,139.8 575.7,123.8 588.6,107.0 601.4,89.6 614.3,71.4 627.1,52.6 640.0,33.1"/>
  <line x1="308.5" y1="298.7" x2="308.5" y2="185" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="308.5" cy="298.7" r="5" fill="#10b981"/>
  <line x1="100" y1="298.7" x2="308.5" y2="298.7" stroke="#10b981" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="316" y="303" font-size="13" fill="#10b981">minimum: bound = 0.1335</text>
  <text x="250" y="180" font-size="12.5" fill="#10b981">λ* = log 1.5 = 0.4055</text>
  <text x="600" y="30" font-size="12.5" fill="#3b82f6">bound &gt; 1: useless</text>
  <text x="104" y="180" font-size="12" fill="currentColor">1</text>
  <text x="82" y="120" font-size="12" fill="currentColor">10</text>
  <text x="70" y="253" font-size="12" fill="currentColor">0.1</text>
  <text x="66" y="318" font-size="12" fill="currentColor">0.01</text>
  <text x="96" y="205" font-size="12" fill="currentColor">0</text>
  <text x="300" y="205" font-size="12" fill="currentColor">0.4</text>
  <text x="558" y="205" font-size="12" fill="currentColor">0.9</text>
  <text x="360" y="222" font-size="12.5" fill="currentColor">λ</text>
  <text x="100" y="358" font-size="12.5" fill="currentColor">Figure 4.2. The Chernoff objective e^(−λt)·M(λ)^n for S ~ Binomial(100, ½) at t = 60, on a logarithmic vertical</text>
  <text x="100" y="374" font-size="12.5" fill="currentColor">axis. At λ = 0 it equals 1 (Markov applied to a constant: no information). It decreases, reaches its minimum</text>
  <text x="100" y="390" font-size="12.5" fill="currentColor">0.1335 at λ* = log 1.5, then blows up. Choosing λ badly is worse than useless; optimizing is the whole method.</text>
</svg>

Figure 4.2 makes visible why we optimize rather than pick a convenient $\lambda$. At $\lambda=0$ the bound is exactly $1$ — Markov applied to the constant $1$, which says nothing. As $\lambda$ grows the exponential tilt initially helps, then the MGF factor $M(\lambda)^{100}$ overwhelms the $e^{-60\lambda}$ discount and the bound rises past $1$ again. There is a unique optimum because $\lambda\mapsto \lambda t-\psi(\lambda)$ is concave (Lemma 4.13(2)), so first-order conditions suffice — no need to check second derivatives.

**The multiplicative form, worth memorizing.** Running the computation above for a sum of independent Bernoulli variables — not necessarily with the same $p$ — and keeping the answer in *relative* rather than absolute terms gives the version you will meet most often in algorithm analysis. If $S=\sum_iB_i$ with $B_i\sim\mathrm{Ber}(p_i)$ independent and $\mu=\mathbb{E}[S]=\sum_ip_i$, then for every $\eta>0$
$$\mathbb{P}\big(S\ge(1+\eta)\mu\big)\ \le\ \Big(\frac{e^{\eta}}{(1+\eta)^{1+\eta}}\Big)^{\mu}=e^{-\mu\,h(\eta)},\qquad h(\eta)=(1+\eta)\log(1+\eta)-\eta,$$
and for $0<\eta<1$ the two convenient weakenings
$$\mathbb{P}\big(S\ge(1+\eta)\mu\big)\le e^{-\mu\eta^2/3},\qquad \mathbb{P}\big(S\le(1-\eta)\mu\big)\le e^{-\mu\eta^2/2}.$$
The derivation is Exercise A7, where you will get the same $h$ from a Poisson count and see why that is no accident. Note the shape: the exponent scales with the *expected count* $\mu$, not with $n$. This is the right tool whenever you are asking "I expect $50$ collisions; how likely are $100$?" — a question about a rare-ish count, where Hoeffding's $n$-based bound would be far too weak because it cannot see that $\mu\ll n$. Keep $h$ in view; it returns as Bennett's function in §4.5, for exactly the reason developed there.

> **Common Confusion 4.3**
>
> *"So Hoeffding tells me that with probability $95\%$ the true error is within $\pm0.0136$ of my measured test error. That means $\mathbb{P}(R\in[\hat R-0.0136,\ \hat R+0.0136])=0.95$, right?"*
>
> Careful. The true risk $R$ is a fixed number, not a random variable, so the statement "$\mathbb{P}(R\in\text{my interval})=0.95$" has no meaning under the probability measure we are using: the randomness lives in the sample (recall Chapter 1's insistence that $\omega$ is the random seed and everything else is a deterministic function of it). What the theorem says is a statement about the *procedure*: the random interval $[\hat R_n-\varepsilon,\hat R_n+\varepsilon]$ covers the fixed $R$ with probability at least $0.95$ over draws of the test set. Once you have drawn your particular test set and computed a particular interval, that interval either contains $R$ or it doesn't; what is calibrated is your long-run rate of being right, not this instance.
>
> This matters practically because it explains a real failure mode. If you evaluate $20$ model checkpoints on the same test set and report the best one's interval, that interval is *not* a $95\%$ interval for the selected model — you have run $20$ trials and kept the luckiest, so the coverage guarantee is destroyed. The fix is exactly §4.7's union bound: use $\delta/20$ per checkpoint. One honest caveat about that fix: the twenty checkpoints' losses are computed on the *same* test set and are typically highly correlated (adjacent checkpoints misclassify nearly the same examples), and the union bound ignores correlation entirely — it is valid regardless, but conservative, and the more similar the models, the more it overcharges. That conservatism is the price of a guarantee that holds with no assumptions on the dependence structure; Exercise A15 shows how to *exploit* the correlation instead when the question is a pairwise comparison. Getting a Bayesian statement — a probability distribution over $R$ given your data — requires a prior and is a different object entirely (Ch. 1's Proposition 1.12 and Chapter 9's PAC-Bayes).

> **Self-check 4.3**
>
> 1. Why does the Chernoff method restrict to $\lambda\ge0$ when bounding an upper tail?
> 2. Compute $\psi_X(\lambda)$ for $X\sim N(0,\sigma^2)$ and then $\psi^*_X(t)$. What tail bound results?
> 3. If $M_X(\lambda)=\infty$ for every $\lambda>0$, what does Theorem 4.14 give, and what does that tell you about heavy-tailed variables?
> 4. In Figure 4.2, why must the curve pass through the value $1$ at $\lambda=0$ no matter what $X$ and $t$ are?
## 4.4 Sub-Gaussian and sub-exponential variables

### 4.4.1 The right level of generality

Theorem 4.14 says: give me a bound on the MGF and I will give you a tail bound. So the question becomes, which random variables have MGFs bounded like a Gaussian's? Chapter 3 computed $\psi_X(\lambda)=\lambda^2\sigma^2/2$ for $X\sim N(0,\sigma^2)$. Running that through Theorem 4.14 gives $\psi^*(t)=t^2/(2\sigma^2)$ and hence $\mathbb{P}(X\ge t)\le e^{-t^2/(2\sigma^2)}$. That is the shape we want. Rather than reprove it for each distribution, we name the property.

**Definition 4.16** (Sub-Gaussian). A random variable $X$ with $\mathbb{E}[X]=0$ is *$\sigma^2$-sub-Gaussian* (equivalently, has *variance proxy* $\sigma^2$) if
$$\mathbb{E}\big[e^{\lambda X}\big]\ \le\ e^{\lambda^2\sigma^2/2}\qquad\text{for all }\lambda\in\mathbb{R}.$$
A non-centered $X$ is called sub-Gaussian if $X-\mathbb{E}[X]$ is.

Examples, all immediate: $N(0,\sigma^2)$ is $\sigma^2$-sub-Gaussian with equality; a Rademacher variable $\sigma\in\{\pm1\}$ with equal probabilities is $1$-sub-Gaussian (expand $\cosh\lambda=\sum\lambda^{2k}/(2k)!\le\sum\lambda^{2k}/(2^kk!)=e^{\lambda^2/2}$, using $(2k)!\ge2^kk!$); any bounded variable is sub-Gaussian, which is Lemma 4.19 below and is the single most-used fact in learning theory, because losses are bounded.

**Theorem 4.17** (Equivalent characterizations of sub-Gaussianity). Let $X$ be centered. The following are equivalent in the sense that each implies the others with a constant factor change in the parameter, and there are absolute constants $c,C>0$ making all the implications hold:

1. *(MGF)* $\mathbb{E}[e^{\lambda X}]\le e^{\lambda^2K_1^2/2}$ for all $\lambda\in\mathbb{R}$;
2. *(Tails)* $\mathbb{P}(\lvert X\rvert\ge t)\le 2e^{-t^2/(2K_2^2)}$ for all $t\ge0$;
3. *(Moment growth)* $\lVert X\rVert_p=(\mathbb{E}\lvert X\rvert^p)^{1/p}\le K_3\sqrt p$ for all $p\ge1$;
4. *(Orlicz norm)* $\mathbb{E}\big[e^{X^2/K_4^2}\big]\le 2$.

We prove (1) $\Rightarrow$ (2) and (2) $\Rightarrow$ (3), the two used in this chapter, and record the rest as a cited fact (§4.12).

*Proof of (1) $\Rightarrow$ (2), with $K_2=K_1$.* Apply Theorem 4.14: for $t\ge0$,
$$\mathbb{P}(X\ge t)\le\inf_{\lambda\ge0}\exp\Big(\tfrac{\lambda^2K_1^2}{2}-\lambda t\Big).$$
The exponent is a quadratic in $\lambda$ minimized at $\lambda^\star=t/K_1^2$, where its value is $-t^2/(2K_1^2)$. So $\mathbb{P}(X\ge t)\le e^{-t^2/(2K_1^2)}$. Since $-X$ also satisfies (1) with the same constant (replace $\lambda$ by $-\lambda$ in Definition 4.16), the same bound holds for $\mathbb{P}(-X\ge t)$, and adding the two gives (2). $\blacksquare$

*Proof of (2) $\Rightarrow$ (3), with $K_3=e^{1/e}\sqrt2\,K_2\le 2.05\,K_2$.* Take $p\ge2$. The tail formula for expectations (Ch. 2) applied to the non-negative variable $\lvert X\rvert^p$ gives
$$\mathbb{E}\lvert X\rvert^p=\int_0^\infty \mathbb{P}\big(\lvert X\rvert^p>s\big)\,ds=\int_0^\infty p\,t^{p-1}\,\mathbb{P}(\lvert X\rvert>t)\,dt,$$
where we substituted $s=t^p$. Insert (2) and substitute $u=t^2/(2K_2^2)$:
$$\mathbb{E}\lvert X\rvert^p\ \le\ 2p\int_0^\infty t^{p-1}e^{-t^2/(2K_2^2)}dt\ =\ p\,(2K_2^2)^{p/2}\,\Gamma\!\big(\tfrac p2\big).$$
Using $\Gamma(x)\le x^{x}$ for $x\ge1$ and $p^{1/p}\le e^{1/e}$,
$$\lVert X\rVert_p\ \le\ \sqrt2\,K_2\,p^{1/p}\big(\tfrac p2\big)^{1/2}\ \le\ e^{1/e}\,K_2\,\sqrt{p}.$$
For $1\le p<2$, Lyapunov's inequality (Corollary 4.6) gives $\lVert X\rVert_p\le\lVert X\rVert_2\le e^{1/e}\sqrt2 K_2\le e^{1/e}\sqrt2K_2\sqrt p$. $\blacksquare$

The equivalence is what makes "sub-Gaussian" a robust concept rather than a technical convenience: you can verify it whichever way is easiest for your object and use it whichever way is most convenient. Characterization (3) is the one that appears in matrix concentration and in random-matrix arguments; (4) defines the norm $\lVert X\rVert_{\psi_2}$ used in high-dimensional statistics.

**Proposition 4.18** (Closure properties). Let $X$ be $\sigma^2$-sub-Gaussian and $c\in\mathbb{R}$. Then $cX$ is $c^2\sigma^2$-sub-Gaussian. If $X_1,\dots,X_n$ are *independent* with $X_i$ being $\sigma_i^2$-sub-Gaussian, then $\sum_iX_i$ is $\big(\sum_i\sigma_i^2\big)$-sub-Gaussian.

*Proof.* For scaling, $\mathbb{E}[e^{\lambda cX}]\le e^{(\lambda c)^2\sigma^2/2}$. For sums, Lemma 4.13(3) gives $\psi_{\sum X_i}(\lambda)=\sum_i\psi_{X_i}(\lambda)\le\sum_i\lambda^2\sigma_i^2/2$. $\blacksquare$

So variance proxies add exactly like variances, and in particular the average of $n$ independent $\sigma^2$-sub-Gaussian variables is $(\sigma^2/n)$-sub-Gaussian. That one line plus Theorem 4.17(1)$\Rightarrow$(2) *is* Hoeffding's inequality; §4.5 just assembles it.

### 4.4.2 Bounded implies sub-Gaussian

**Lemma 4.19** (Hoeffding's lemma). Let $X$ be a random variable with $a\le X\le b$ a.s. and $\mathbb{E}[X]=0$. Then for all $\lambda\in\mathbb{R}$,
$$\mathbb{E}\big[e^{\lambda X}\big]\ \le\ \exp\Big(\frac{\lambda^2(b-a)^2}{8}\Big),$$
i.e. $X$ is $\sigma^2$-sub-Gaussian with variance proxy $\sigma^2=(b-a)^2/4$.

*Proof.* Since $X$ is bounded, $M_X(\lambda)=\mathbb{E}[e^{\lambda X}]\le e^{\lvert\lambda\rvert\max(\lvert a\rvert,\lvert b\rvert)}<\infty$ for every $\lambda$, and the same domination lets us differentiate under the integral sign twice (dominated convergence, Ch. 2). Write $\psi=\psi_X=\log M_X$. Then
$$\psi'(\lambda)=\frac{\mathbb{E}[Xe^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]},\qquad
\psi''(\lambda)=\frac{\mathbb{E}[X^2e^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]}-\Big(\frac{\mathbb{E}[Xe^{\lambda X}]}{\mathbb{E}[e^{\lambda X}]}\Big)^2 .$$
Now interpret these. Define the *tilted* probability measure $P_\lambda$ by
$$\frac{dP_\lambda}{dP}:=\frac{e^{\lambda X}}{\mathbb{E}[e^{\lambda X}]},$$
a legitimate probability measure since the density is positive with integral $1$ (this is a Radon–Nikodym derivative in the sense of Ch. 2). Then $\psi'(\lambda)=\mathbb{E}_{P_\lambda}[X]$ and $\psi''(\lambda)=\mathrm{Var}_{P_\lambda}(X)$.

Because $P_\lambda\ll P$ and $P(a\le X\le b)=1$, we also have $P_\lambda(a\le X\le b)=1$. Any random variable supported in $[a,b]$ has variance at most $(b-a)^2/4$: with $m=(a+b)/2$,
$$\mathrm{Var}(X)\le\mathbb{E}\big[(X-m)^2\big]\le\Big(\frac{b-a}{2}\Big)^2,$$
where the first step is the variational characterization of the variance (the mean minimizes mean squared deviation) and the second is $\lvert X-m\rvert\le(b-a)/2$ pointwise. Hence $\psi''(\lambda)\le(b-a)^2/4$ for every $\lambda$.

Finally, $\psi(0)=0$ and $\psi'(0)=\mathbb{E}[X]=0$, so Taylor's theorem with Lagrange remainder gives, for some $\xi$ between $0$ and $\lambda$,
$$\psi(\lambda)=\psi(0)+\lambda\psi'(0)+\frac{\lambda^2}{2}\psi''(\xi)\ \le\ \frac{\lambda^2}{2}\cdot\frac{(b-a)^2}{4}=\frac{\lambda^2(b-a)^2}{8}. \qquad\blacksquare$$

The tilting device in that proof is worth remembering for its own sake: **the derivatives of the CGF are the moments of the exponentially tilted distribution.** It is the same computation that makes the natural parameters of an exponential family (Ch. 3) so pleasant, and it reappears in Chapter 9 when we discuss the Gibbs posterior.

**Worked Example 4.8** (The variance proxy can be badly conservative). Let $X=B-p$ with $B\sim\mathrm{Ber}(p)$, so $X\in[-p,1-p]$ and $b-a=1$. Lemma 4.19 gives variance proxy $\sigma^2=1/4$ *regardless of $p$*. But the true variance is $p(1-p)$. At $p=0.01$:
$$\text{proxy}=0.25,\qquad \text{true variance}=0.0099,\qquad \text{ratio}=25.25 .$$
Directly checking the lemma numerically over $\lambda\in[0.01,8]$: the ratio $\psi_X(\lambda)\big/(\lambda^2/8)$ has maximum $1.0000$ at $p=0.5$, $0.7282$ at $p=0.1$, and $0.4184$ at $p=0.01$. So the lemma is *exactly tight* for a fair coin and loose by a factor of about $2.4$ in the exponent for a $1\%$ coin — and the looseness gets worse as $p\to0$.

This is not a curiosity. Rare-event estimation is the norm in machine learning: click-through rates, fraud, rare-class recall, per-token error rates on a good language model. Any bound that charges you variance proxy $1/4$ when your variance is $0.01$ will demand roughly $25\times$ more data than necessary. Section 4.5's Bernstein inequality exists to fix precisely this, by charging you the true variance plus a smaller correction term.

### 4.4.3 When Gaussian tails are too much to ask: sub-exponential variables

Not everything is sub-Gaussian, and the failures are not exotic. The square of a Gaussian is not sub-Gaussian; neither is an exponential variable, a Poisson count, or a product of two Gaussians. The obstruction is visible in the MGF: Definition 4.16 demands finiteness for *all* $\lambda\in\mathbb{R}$, and many useful variables have an MGF that blows up beyond a threshold.

**Definition 4.20** (Bernstein condition; sub-exponential). A centered random variable $X$ satisfies the *Bernstein condition* with parameters $(\nu^2,b)$, $\nu^2>0$, $b>0$, if
$$\psi_X(\lambda)=\log\mathbb{E}\big[e^{\lambda X}\big]\ \le\ \frac{\lambda^2\nu^2}{2(1-b\lambda)}\qquad\text{for }0\le\lambda<\frac1b .$$
Such an $X$ is called *sub-exponential*. Applying the condition to $-X$ controls the lower tail.

Compare with Definition 4.16: for small $\lambda$ the two agree to leading order, with $\nu^2$ playing the role of the variance proxy; the difference is the pole at $\lambda=1/b$, which encodes that the tail is only exponential, not Gaussian, far out.

**Proposition 4.21** (Two-regime tail bound). If $X$ satisfies the Bernstein condition with parameters $(\nu^2,b)$, then for all $t\ge0$,
$$\mathbb{P}(X\ge t)\ \le\ \exp\Big(-\frac{t^2}{2(\nu^2+bt)}\Big)\ \le\
\begin{cases}
\exp\big(-\dfrac{t^2}{4\nu^2}\big), & t\le \nu^2/b \quad\text{(Gaussian regime)},\\[2mm]
\exp\big(-\dfrac{t}{4b}\big), & t\ge \nu^2/b\quad\text{(exponential regime)} .
\end{cases}$$

*Proof.* By Theorem 4.14 and Definition 4.20, for any $0\le\lambda<1/b$,
$$\mathbb{P}(X\ge t)\le\exp\Big(\frac{\lambda^2\nu^2}{2(1-b\lambda)}-\lambda t\Big).$$
Rather than differentiate, substitute the guess $\lambda=\dfrac{t}{\nu^2+bt}$, which indeed lies in $[0,1/b)$. Then $1-b\lambda=\dfrac{\nu^2}{\nu^2+bt}$, so
$$\frac{\lambda^2\nu^2}{2(1-b\lambda)}=\frac{t^2}{(\nu^2+bt)^2}\cdot\frac{\nu^2}{2}\cdot\frac{\nu^2+bt}{\nu^2}=\frac{t^2}{2(\nu^2+bt)},\qquad \lambda t=\frac{t^2}{\nu^2+bt},$$
and the exponent becomes $\frac{t^2}{2(\nu^2+bt)}-\frac{t^2}{\nu^2+bt}=-\frac{t^2}{2(\nu^2+bt)}$, giving the first bound. (That guess is *not* the exact optimizer — it is chosen because it makes the algebra close cleanly. The true minimizer is available in closed form and gives a slightly better exponent: for $\nu^2=b=t=1$ it yields $-0.2679$ against the $-0.25$ we are about to report. Exercise A12 works this out, and the lesson is worth absorbing early: published constants in this subject are usually the product of a deliberate simplification, not of a tight optimization.) For the two regimes: if $bt\le\nu^2$ then $\nu^2+bt\le2\nu^2$, so the exponent is at most $-t^2/(4\nu^2)$; if $bt\ge\nu^2$ then $\nu^2+bt\le2bt$, so it is at most $-t/(4b)$. $\blacksquare$

The picture to carry: **a sub-exponential variable has a Gaussian tail near the mean and an exponential tail far out, and the crossover happens at $t\approx\nu^2/b$.** Near the mean — which is where sample averages live — you lose nothing relative to sub-Gaussian. This is why sub-exponential variables are almost as good as sub-Gaussian ones for statistical purposes, and why the Bernstein condition is the workhorse assumption in modern statistics.

**Proposition 4.22** (Squared Gaussians are sub-exponential). Let $Z\sim N(0,1)$ and $X=Z^2-1$. Then $X$ satisfies the Bernstein condition with $(\nu^2,b)=(2,2)$. Consequently, for $Z_1,\dots,Z_n$ i.i.d. standard normal and $t\ge0$,
$$\mathbb{P}\Big(\frac1n\sum_{i=1}^n Z_i^2-1\ \ge\ t\Big)\ \le\ \exp\Big(-\frac{nt^2}{2(2+2t)}\Big).$$

*Proof.* From Chapter 3, $\mathbb{E}[e^{\lambda Z^2}]=(1-2\lambda)^{-1/2}$ for $\lambda<1/2$ and $=\infty$ for $\lambda\ge1/2$. Hence for $0\le\lambda<1/2$,
$$\psi_X(\lambda)=-\lambda-\tfrac12\log(1-2\lambda)=-\lambda+\tfrac12\sum_{k\ge1}\frac{(2\lambda)^k}{k}=\sum_{k\ge2}\frac{(2\lambda)^k}{2k},$$
using the series $-\log(1-u)=\sum_{k\ge1}u^k/k$ for $\lvert u\rvert<1$ and noting the $k=1$ term cancels $-\lambda$. Since $2k\ge4$ for $k\ge2$,
$$\psi_X(\lambda)\ \le\ \frac14\sum_{k\ge2}(2\lambda)^k=\frac14\cdot\frac{4\lambda^2}{1-2\lambda}=\frac{\lambda^2}{1-2\lambda}=\frac{\lambda^2\cdot 2}{2(1-2\lambda)},$$
which is Definition 4.20 with $\nu^2=2$, $b=2$. The statement for the average follows because the Bernstein condition is preserved under sums of independent variables with $\nu^2$ and $b$ scaling as in Exercise A13, or directly: $\psi_{S_n}(\lambda)=n\psi_X(\lambda)$ and then optimize as in Proposition 4.21. $\blacksquare$

Numerically, the bound $\lambda^2/(1-2\lambda)$ is nearly exact: over $\lambda\in(0.001,0.499)$ the maximum of $\psi_X(\lambda)$ divided by the bound is $0.99933$, so almost nothing is lost.

**Worked Example 4.9** (The MGF cliff, and what the two-regime bound delivers). At $\lambda=0.4$, $\mathbb{E}[e^{0.4Z^2}]=(1-0.8)^{-1/2}=2.2361$; at $\lambda=0.5$ it is $+\infty$. No variance proxy can exist, so $Z^2$ is not sub-Gaussian for any $\sigma^2$. Here is what Proposition 4.21 gives for $X=Z^2-1$ against the exact $\chi^2_1$ tail:

| $t$ | true $\mathbb{P}(Z^2-1\ge t)$ | bound $e^{-t^2/(2(2+2t))}$ |
|---|---|---|
| $1$ | $1.573\times10^{-1}$ | $8.825\times10^{-1}$ |
| $5$ | $1.431\times10^{-2}$ | $3.529\times10^{-1}$ |
| $25$ | $3.414\times10^{-7}$ | $2.455\times10^{-3}$ |
| $100$ | $9.198\times10^{-24}$ | $1.779\times10^{-11}$ |

The bound is loose for a single variable — inevitably, since it is designed to survive summation. Now sum: for the average of $n=100$ i.i.d. copies,

| $t$ | true $\mathbb{P}(\frac1n\sum Z_i^2-1\ge t)$ | bound $e^{-nt^2/(2(2+2t))}$ |
|---|---|---|
| $0.2$ | $8.441\times10^{-2}$ | $4.346\times10^{-1}$ |
| $0.5$ | $9.039\times10^{-4}$ | $1.550\times10^{-2}$ |
| $1.0$ | $1.179\times10^{-8}$ | $3.727\times10^{-6}$ |

Now we are within one to three orders of magnitude of the truth on probabilities spanning eight orders of magnitude, which is the accuracy that matters for choosing sample sizes.

Where this bites in practice: $\frac1n\sum_iZ_i^2$ is the squared norm of a Gaussian vector divided by its dimension, so Proposition 4.22 is the concentration of $\lVert z\rVert^2/d$ for $z\sim N(0,I_d)$ — the statement that a high-dimensional Gaussian lives near a sphere of radius $\sqrt d$. That fact drives the initialization scaling of Chapter 5, the behavior of random projections, and the geometry of VAE latent spaces (where samples from the prior concentrate on a shell, so the decoder is never trained near the origin).

> **Common Confusion 4.4**
>
> *"If $X$ is $\sigma^2$-sub-Gaussian then $\sigma^2$ is its variance, and its tail is basically Gaussian, so I can use $z$-scores."*
>
> Two errors. First, $\sigma^2$ is an upper bound on the variance, not the variance. Taking $\lambda\to0$ in Definition 4.16 and matching second-order terms gives $\mathrm{Var}(X)\le\sigma^2$, and Worked Example 4.8 exhibits a gap of $25\times$. When a paper's bound depends on "the sub-Gaussian parameter," it may be enormously larger than the empirical variance you measured.
>
> Second, sub-Gaussian is an *upper* bound on the tail, not an approximation to it. A variable bounded in $[-1,1]$ is $1$-sub-Gaussian, and its tail beyond $t=1$ is exactly zero, not $e^{-t^2/2}$. Conversely nothing forces the tail to be as heavy as the bound. So you may never use sub-Gaussianity to compute a $p$-value or to say "the deviation is approximately normal with variance $\sigma^2$"; it licenses only one-sided statements of the form "the probability of exceeding $t$ is at most...".
>
> A related and more consequential misuse: many SGD convergence analyses assume the stochastic gradient noise is sub-Gaussian, because that assumption makes the proofs work. Empirical studies of gradient noise in deep networks report heavy tails that are not even sub-exponential, sometimes closer to an $\alpha$-stable law with $\alpha<2$. When a theorem's conclusion contradicts what you observe, this assumption is one of the first places to look.

> **Self-check 4.4**
>
> 1. Show directly from Definition 4.16 that a $\sigma^2$-sub-Gaussian variable has $\mathrm{Var}(X)\le\sigma^2$. (Hint: expand both sides to second order in $\lambda$.)
> 2. Is a variable uniform on $[-1,1]$ sub-Gaussian? With what proxy does Lemma 4.19 endow it, and what is its true variance?
> 3. Why does Definition 4.20 restrict to $\lambda<1/b$, and what goes wrong at $\lambda=1/b$ for $Z^2-1$?
> 4. In Proposition 4.21's two regimes, which regime governs when you are asking about a deviation of one standard deviation of the sample mean, i.e. $t\asymp\nu/\sqrt n$ for a sum of $n$ terms?
## 4.5 Hoeffding, Bernstein, Bennett

Everything is now in place. Each of the three classical inequalities below is Theorem 4.14 applied to a sum, with a different MGF bound plugged in, and each charges you a different assumption.

### 4.5.1 Hoeffding: boundedness is enough

**Theorem 4.23** (Hoeffding's inequality). Let $X_1,\dots,X_n$ be independent with $a_i\le X_i\le b_i$ a.s., and let $S_n=\sum_{i=1}^n\big(X_i-\mathbb{E}[X_i]\big)$. Then for all $t\ge0$,
$$\mathbb{P}(S_n\ge t)\ \le\ \exp\Big(-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\Big),$$
and the same bound holds for $\mathbb{P}(S_n\le-t)$; adding them gives a factor $2$ for the two-sided event. In particular, if all $X_i\in[0,1]$ and $\bar X_n$ denotes the average with $\mu=\mathbb{E}[\bar X_n]$,
$$\mathbb{P}\big(\lvert \bar X_n-\mu\rvert\ge\varepsilon\big)\ \le\ 2e^{-2n\varepsilon^2},$$
equivalently: with probability at least $1-\delta$, $\ \lvert\bar X_n-\mu\rvert\le\sqrt{\dfrac{\log(2/\delta)}{2n}}$.

*Proof.* Let $Y_i=X_i-\mathbb{E}[X_i]$, which is centered and takes values in an interval of length $b_i-a_i$. By Lemma 4.19, $Y_i$ is $\sigma_i^2$-sub-Gaussian with $\sigma_i^2=(b_i-a_i)^2/4$. By Proposition 4.18 (independence), $S_n=\sum_iY_i$ is $\sigma^2$-sub-Gaussian with $\sigma^2=\frac14\sum_i(b_i-a_i)^2$. By Theorem 4.17, (1)$\Rightarrow$(2) — or directly by Theorem 4.14 with the quadratic optimization done there —
$$\mathbb{P}(S_n\ge t)\le \exp\Big(-\frac{t^2}{2\sigma^2}\Big)=\exp\Big(-\frac{2t^2}{\sum_i(b_i-a_i)^2}\Big).$$
For the lower tail apply the same to $-S_n$. For the specialization, $(b_i-a_i)=1$ and $t=n\varepsilon$ give $\exp(-2n^2\varepsilon^2/n)=e^{-2n\varepsilon^2}$. Solving $2e^{-2n\varepsilon^2}=\delta$ for $\varepsilon$ gives the last form. $\blacksquare$

Note what is *not* assumed: the $X_i$ need not be identically distributed, and no variance appears anywhere. Note also what is: independence (through Proposition 4.18) and boundedness (through Lemma 4.19).

**Worked Example 4.10** (The error bar nobody prints). You evaluate a classifier on a held-out set of $n$ examples; the per-example loss is $0$–$1$, so it lies in $[0,1]$, and the test examples are i.i.d. from the deployment distribution. Then $\hat R_n$ is an average of independent $[0,1]$ variables with mean $R$, and Theorem 4.23 gives, at $\delta=0.05$:

| $n$ | Hoeffding radius $\sqrt{\log(2/\delta)/2n}$ |
|---|---|
| $100$ | $0.135810$ |
| $1{,}000$ | $0.042947$ |
| $10{,}000$ | $0.013581$ |
| $60{,}000$ | $0.005544$ |

So a test accuracy reported on $10{,}000$ examples carries a $\pm1.36$ percentage-point interval at $95\%$ confidence, *before* accounting for any model selection. This is the number promised in §4.0. It has immediate consequences for how you read a leaderboard: two models differing by $0.3$ points on a $10{,}000$-example test set are statistically indistinguishable by this bound, and even the paired comparison (which is tighter, because it uses the variance of the *difference* of losses — see Exercise A15) rarely rescues a $0.3$-point claim. A great deal of published progress on small benchmarks lies inside this interval.

Two honest caveats. First, Hoeffding is conservative here by a factor of about $2.6$ relative to the truth (Worked Example 4.12 quantifies this), so a normal-approximation interval is narrower and usually more informative; the value of Hoeffding is that it needs no asymptotics and it survives the union bounds of §4.7. Second, the i.i.d. assumption is often false in exactly the ways that matter: test sets built by scraping frequently contain near-duplicates of training examples, and benchmark test sets get reused thousands of times by the community, which is Common Confusion 4.3's selection problem at the scale of a whole field.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" role="img" aria-label="Log-scale comparison of Markov, Chebyshev, Hoeffding bounds and the true tail for the mean of 100 uniform variables">
  <line x1="100" y1="330" x2="660" y2="330" stroke="currentColor" stroke-width="1"/>
  <line x1="100" y1="55" x2="100" y2="330" stroke="currentColor" stroke-width="1"/>
  <polyline fill="none" stroke="currentColor" stroke-width="1.8" stroke-dasharray="5,3" points="116.9,60.2 170.1,60.8 223.3,61.4 276.5,61.9 329.7,62.5 382.9,63.0 436.1,63.5 489.3,64.1 542.5,64.6 595.7,65.0 640.0,65.4"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2" points="196.7,60.0 205.5,63.1 223.3,69.2 241.0,74.4 258.7,79.1 276.5,83.2 294.2,87.0 311.9,90.4 329.7,93.5 347.4,96.4 365.1,99.1 382.9,101.7 400.6,104.0 418.3,106.3 436.1,108.4 453.8,110.4 471.5,112.3 489.3,114.1 507.0,115.9 524.7,117.5 542.5,119.1 560.2,120.7 577.9,122.2 595.7,123.6 613.4,125.0 640.0,126.9"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="116.9,60.1 134.6,60.4 152.3,60.9 170.1,61.7 187.8,62.6 205.5,63.8 223.3,65.2 241.0,66.8 258.7,68.6 276.5,70.7 294.2,72.9 311.9,75.4 329.7,78.1 347.4,81.0 365.1,84.1 382.9,87.5 400.6,91.0 418.3,94.8 436.1,98.8 453.8,103.0 471.5,107.4 489.3,112.0 507.0,116.8 524.7,121.9 542.5,127.2 560.2,132.7 577.9,138.4 595.7,144.3 613.4,150.4 631.1,156.8 640.0,160.1"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="167.5,87.5 235.0,108.5 302.5,137.5 370.0,175.5 437.5,222.1 505.0,281.4"/>
  <circle cx="167.5" cy="87.5" r="3.5" fill="#10b981"/><circle cx="235.0" cy="108.5" r="3.5" fill="#10b981"/><circle cx="302.5" cy="137.5" r="3.5" fill="#10b981"/><circle cx="370.0" cy="175.5" r="3.5" fill="#10b981"/><circle cx="437.5" cy="222.1" r="3.5" fill="#10b981"/><circle cx="505.0" cy="281.4" r="3.5" fill="#10b981"/>
  <text x="560" y="58" font-size="12" fill="currentColor">Markov</text>
  <text x="560" y="140" font-size="12" fill="#f59e0b">Chebyshev</text>
  <text x="608" y="176" font-size="12" fill="#3b82f6">Hoeffding</text>
  <text x="512" y="296" font-size="12" fill="#10b981">truth (simulated)</text>
  <text x="66" y="64" font-size="11.5" fill="currentColor">1</text>
  <text x="60" y="154" font-size="11.5" fill="currentColor">10⁻²</text>
  <text x="60" y="244" font-size="11.5" fill="currentColor">10⁻⁴</text>
  <text x="60" y="334" font-size="11.5" fill="currentColor">10⁻⁶</text>
  <text x="228" y="348" font-size="11.5" fill="currentColor">t = 0.04</text>
  <text x="363" y="348" font-size="11.5" fill="currentColor">t = 0.08</text>
  <text x="498" y="348" font-size="11.5" fill="currentColor">t = 0.12</text>
  <text x="100" y="372" font-size="12.5" fill="currentColor">Figure 4.3. Upper-tail bounds on P(X̄₁₀₀ − 0.5 ≥ t) for the mean of 100 i.i.d. Uniform[0,1] variables. Markov is</text>
  <text x="100" y="388" font-size="12.5" fill="currentColor">flat and useless; Chebyshev decays polynomially; Hoeffding decays like exp(−200t²) and tracks the true tail's</text>
  <text x="100" y="399" font-size="11.5" fill="currentColor">shape (green, 2×10⁶ simulated draws, seed 20240419), staying above it by roughly two orders of magnitude.</text>
</svg>

Figure 4.3 is the whole story of §4.2–§4.5 in one image. On a logarithmic axis, "Markov" is a nearly horizontal line: it decays like $1/(0.5+t)$, which on this scale is invisible. Chebyshev is a straight line of slope $-2$ in $\log t$, still polynomial. Hoeffding is a downward parabola, and the truth is another downward parabola sitting about two orders of magnitude below it. Chebyshev actually *beats* Hoeffding for small $t$ — at $t=0.10$ Chebyshev gives $8.33\times10^{-2}$ against Hoeffding's $1.35\times10^{-1}$ — and the two cross between $t=0.10$ and $t=0.12$ (where the values are $5.79\times10^{-2}$ and $5.61\times10^{-2}$). By $t=0.15$ Hoeffding gives $1.11\times10^{-2}$ against Chebyshev's $3.70\times10^{-2}$, and the gap widens without limit thereafter.

### 4.5.2 Bernstein: paying for variance, refunded generously

**Theorem 4.24** (Bernstein's inequality). Let $X_1,\dots,X_n$ be independent with $\lvert X_i-\mathbb{E}[X_i]\rvert\le M$ a.s., and set $\sigma^2:=\frac1n\sum_{i=1}^n\mathrm{Var}(X_i)$. Then for all $t\ge0$,
$$\mathbb{P}\big(\bar X_n-\mu\ge t\big)\ \le\ \exp\Big(-\frac{n\,t^2}{2\big(\sigma^2+\tfrac{Mt}{3}\big)}\Big),$$
where $\mu=\mathbb{E}[\bar X_n]$. Equivalently, with probability at least $1-\delta$,
$$\bar X_n-\mu\ \le\ \sqrt{\frac{2\sigma^2\log(1/\delta)}{n}}\ +\ \frac{2M\log(1/\delta)}{3n}.$$

*Proof.* Let $Y=X_i-\mathbb{E}[X_i]$, so $\mathbb{E}[Y]=0$, $\lvert Y\rvert\le M$, $\mathrm{Var}(Y)=\sigma_i^2$. Expand the MGF and bound the moments using $\lvert Y\rvert^k\le M^{k-2}Y^2$ for $k\ge2$:
$$\mathbb{E}[e^{\lambda Y}]=1+\lambda\mathbb{E}[Y]+\sum_{k\ge2}\frac{\lambda^k\mathbb{E}[Y^k]}{k!}\ \le\ 1+\sigma_i^2\sum_{k\ge2}\frac{\lambda^kM^{k-2}}{k!}=1+\frac{\lambda^2\sigma_i^2}{2}\sum_{k\ge2}\frac{2\,(\lambda M)^{k-2}}{k!} .$$
Since $2/k!\le 3^{-(k-2)}$ for every $k\ge2$ (check $k=2,3$ with equality, then note $k!$ grows faster than $3^{k}$), for $0\le\lambda<3/M$ the sum is at most $\sum_{j\ge0}(\lambda M/3)^j=(1-\lambda M/3)^{-1}$. Hence, using $1+u\le e^u$,
$$\psi_Y(\lambda)\ \le\ \frac{\lambda^2\sigma_i^2}{2\big(1-\lambda M/3\big)},$$
which is exactly the Bernstein condition (Definition 4.20) with $\nu^2=\sigma_i^2$ and $b=M/3$. Summing over $i$ (Lemma 4.13(3)) gives $\psi_{S_n}(\lambda)\le\frac{\lambda^2n\sigma^2}{2(1-\lambda M/3)}$ where $S_n=n(\bar X_n-\mu)$, i.e. the Bernstein condition with $(\nu^2,b)=(n\sigma^2,M/3)$. Proposition 4.21 at level $s=nt$ gives
$$\mathbb{P}(S_n\ge nt)\le\exp\Big(-\frac{n^2t^2}{2(n\sigma^2+Mnt/3)}\Big)=\exp\Big(-\frac{nt^2}{2(\sigma^2+Mt/3)}\Big).$$
For the confidence form, set the right side to $\delta$ and solve the resulting quadratic in $t$, using $\sqrt{u+v}\le\sqrt u+\sqrt v$ to simplify; details are Exercise A14. $\blacksquare$

Read the two terms of the confidence form. The first, $\sqrt{2\sigma^2\log(1/\delta)/n}$, is the *variance term*: it is what the central limit theorem would give you, and it is what you want. The second, $2M\log(1/\delta)/(3n)$, is the *range term*: it decays like $1/n$ rather than $1/\sqrt n$, so it is negligible for large $n$. Bernstein's inequality is thus asymptotically as good as the CLT while remaining a finite-sample bound, and its improvement over Hoeffding is exactly the factor $\sigma/M$ — enormous when the variance is much smaller than the range.

**Theorem 4.25** (Bennett's inequality). In the setting of Theorem 4.24 — in fact under the weaker *one-sided* hypothesis $X_i-\mathbb{E}[X_i]\le M$ a.s., with no lower bound on the $X_i$ required —
$$\mathbb{P}\big(\bar X_n-\mu\ge t\big)\ \le\ \exp\Big(-\frac{n\sigma^2}{M^2}\,h\Big(\frac{Mt}{\sigma^2}\Big)\Big),\qquad h(u):=(1+u)\log(1+u)-u .$$
(Watch the proof: it uses only $Y\le M$, never $Y\ge-M$. This matters in practice — a non-negative loss that is bounded above but can dip far below its mean still satisfies the hypothesis for the upper tail.)

*Proof.* We need a sharper MGF bound than Bernstein's, and it rests on one small lemma about the function
$$\phi(u):=\frac{e^u-1-u}{u^2}\quad(u\ne0),\qquad \phi(0):=\tfrac12 .$$
**Claim: $\phi$ is non-decreasing on all of $\mathbb{R}$.** The clean proof is an integral representation:
$$\phi(u)=\int_0^1(1-t)\,e^{tu}\,dt .$$
To verify it, expand $e^{tu}$ and integrate term by term: $\int_0^1(1-t)t^k\,dt=\frac1{k+1}-\frac1{k+2}=\frac{1}{(k+1)(k+2)}$, so the right-hand side equals $\sum_{k\ge0}\frac{u^k}{k!(k+1)(k+2)}=\sum_{k\ge0}\frac{u^k}{(k+2)!}$, which is exactly the power series of $(e^u-1-u)/u^2$. Now the representation makes monotonicity immediate and *for every real $u$*, positive or negative: the integrand $(1-t)e^{tu}$ is non-negative and non-decreasing in $u$ for each $t\in[0,1]$, so the integral is too. (The power series alone would only give monotonicity for $u\ge0$, which is not enough below, since $\lambda Y$ takes negative values.)

Hence for $Y$ centered with $Y\le M$ and $\lambda>0$,
$$e^{\lambda Y}-1-\lambda Y=(\lambda Y)^2\phi(\lambda Y)\le (\lambda Y)^2\phi(\lambda M),$$
so taking expectations and using $\mathbb{E}[Y]=0$,
$$\mathbb{E}[e^{\lambda Y}]\ \le\ 1+\lambda^2\sigma_Y^2\phi(\lambda M)=1+\frac{\sigma_Y^2}{M^2}\big(e^{\lambda M}-1-\lambda M\big)\ \le\ \exp\Big(\frac{\sigma_Y^2}{M^2}\big(e^{\lambda M}-1-\lambda M\big)\Big).$$
Summing over the $n$ independent terms, with $v:=n\sigma^2$ and $s:=nt$, Theorem 4.14 gives
$$\mathbb{P}(S_n\ge s)\le\exp\Big(\frac{v}{M^2}\big(e^{\lambda M}-1-\lambda M\big)-\lambda s\Big).$$
Choose $\lambda^\star=\frac1M\log\big(1+\frac{Ms}{v}\big)$, so that $e^{\lambda^\star M}=1+u$ with $u=Ms/v$. Substituting and collecting terms yields exponent $-\frac{v}{M^2}\big[(1+u)\log(1+u)-u\big]$; with $s=nt$, $u=Mt/\sigma^2$ and $v/M^2=n\sigma^2/M^2$, this is the claim. $\blacksquare$

Bennett implies Bernstein, because $h(u)\ge\frac{u^2}{2(1+u/3)}$ (Exercise A8), but Bennett is sharper in the far tail, where $h(u)\approx u\log u$ grows faster than linearly. The practical ordering is: Hoeffding needs only a range; Bernstein needs range and variance; Bennett needs the same inputs as Bernstein and is never worse.

**Worked Example 4.11** (Rare events: Bernstein wins by fourteen orders of magnitude). Let $X_i\sim\mathrm{Ber}(0.01)$, $n=10^4$, so $M=1$ and $\sigma^2=0.0099$. Bound $\mathbb{P}(\bar X_n-p\ge t)$:

| $t$ | exact (binomial) | Hoeffding | Bernstein | Bennett |
|---|---|---|---|---|
| $0.005$ | $1.657\times10^{-6}$ | $6.065\times10^{-1}$ | $2.026\times10^{-5}$ | $1.818\times10^{-5}$ |
| $0.01$ | $5.630\times10^{-19}$ | $1.353\times10^{-1}$ | $3.898\times10^{-17}$ | $1.228\times10^{-17}$ |
| $0.02$ | $2.392\times10^{-59}$ | $3.355\times10^{-4}$ | $3.716\times10^{-53}$ | $2.129\times10^{-57}$ |

At $t=0.005$ Hoeffding says "this happens up to $61\%$ of the time" while the truth is one in six hundred thousand; Bernstein is off by a factor of $12$. At $t=0.02$ Bennett is within a factor of $10^{2}$ of a probability near $10^{-59}$, while Hoeffding is off by $10^{55}$. The reason is entirely Worked Example 4.8: Hoeffding is forced to use variance proxy $1/4$ when the variance is $0.0099$.

Inverting to confidence radii at $\delta=0.05$, $n=10^4$, $p=0.01$: Hoeffding gives $0.012239$, Bernstein gives $0.002635$ (a $4.6\times$ improvement), and the exact $95$th percentile of $\bar X_n-p$ is $0.0017$. In relative terms, Hoeffding claims your $1\%$ rate could really be as high as $2.2\%$; Bernstein says at most $1.26\%$; the truth is $1.17\%$. **If you are measuring a small rate — click-through, fraud, rare-class error, hallucination frequency — using Hoeffding rather than Bernstein can inflate your error bar past the quantity you are trying to measure.**

There is a catch, and it is the reason Hoeffding survives in practice: Theorem 4.24 requires $\sigma^2$, which you do not know. Plugging in the empirical variance invalidates the bound (the empirical variance is itself random). The clean fix:

**Proposition 4.26** (Empirical Bernstein; stated, not proved here). Let $X_1,\dots,X_n$ be i.i.d. in $[0,1]$ with mean $\mu$, and let $\hat V_n=\frac{1}{n(n-1)}\sum_{i<j}(X_i-X_j)^2$ be the (unbiased) sample variance. Then with probability at least $1-\delta$,
$$\mu\ \le\ \bar X_n+\sqrt{\frac{2\hat V_n\log(2/\delta)}{n}}+\frac{7\log(2/\delta)}{3(n-1)} .$$
This is the form to use when you want Bernstein's improvement from data alone; see §4.12 for the source.

> **Common Confusion 4.5**
>
> *"Why bother with any of this? I'll just use the CLT: $\hat R_n\pm1.96\,\hat\sigma/\sqrt n$."*
>
> For a single quantity with a decent sample size, that is often the *better* practical choice, and I want to say so plainly rather than pretend concentration inequalities dominate. The normal interval at $n=10^4$ for a $[0,1]$ loss with $\hat\sigma^2=0.09$ has half-width $0.0059$, versus Hoeffding's $0.0136$: less than half as wide, and approximately correctly calibrated.
>
> The three situations where the CLT is not enough. (i) **Finite-sample validity.** The CLT is an asymptotic statement; at $n=30$ with a skewed loss distribution, the nominal $95\%$ interval can have real coverage far below $95\%$, and nothing in the theorem tells you when $n$ is large enough. Hoeffding is exact for every $n$. (ii) **Many simultaneous claims.** As Worked Example 4.6 showed, the union bound forces you to evaluate at $\delta$ exponentially small, and there the CLT's Gaussian quantile $z_{1-\delta/2}\approx\sqrt{2\log(1/\delta)}$ happens to behave like Hoeffding's — but the CLT's *error* in the far tail is not controlled at all, whereas Hoeffding's bound is valid there by construction. This is the regime of all generalization bounds. (iii) **Non-sums.** From §4.6 onwards we bound functions that are not averages at all — the sample median, the output of a learning algorithm, a supremum over a hypothesis class — where no CLT applies.
>
> So: report normal intervals for single evaluations, and reach for concentration when you need finite-sample validity, simultaneity, or non-linear functionals.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" role="img" aria-label="Log-log plot of measured 95 percent deviation versus Hoeffding and Chebyshev radii as sample size grows">
  <line x1="100" y1="330" x2="660" y2="330" stroke="currentColor" stroke-width="1"/>
  <line x1="100" y1="45" x2="100" y2="330" stroke="currentColor" stroke-width="1"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="100.0,50.7 182.7,81.7 273.3,115.7 356.0,146.7 446.7,180.7 529.4,211.7 620.0,245.7"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6,3" points="100.0,53.6 182.7,84.6 273.3,118.6 356.0,149.6 446.7,183.6 529.4,214.6 620.0,248.6"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="100.0,100.2 182.7,131.1 273.3,165.1 356.0,196.2 446.7,230.1 529.4,261.3 620.0,295.2"/>
  <circle cx="100.0" cy="100.2" r="3.5" fill="#10b981"/><circle cx="273.3" cy="165.1" r="3.5" fill="#10b981"/><circle cx="446.7" cy="230.1" r="3.5" fill="#10b981"/><circle cx="620.0" cy="295.2" r="3.5" fill="#10b981"/>
  <text x="470" y="172" font-size="12" fill="#3b82f6">Hoeffding radius</text>
  <text x="470" y="207" font-size="12" fill="#f59e0b">Chebyshev radius</text>
  <text x="450" y="287" font-size="12" fill="#10b981">measured 95th percentile</text>
  <text x="52" y="59" font-size="11.5" fill="currentColor">0.4</text>
  <text x="52" y="137" font-size="11.5" fill="currentColor">0.1</text>
  <text x="45" y="267" font-size="11.5" fill="currentColor">0.01</text>
  <text x="88" y="348" font-size="11.5" fill="currentColor">n = 10</text>
  <text x="255" y="348" font-size="11.5" fill="currentColor">100</text>
  <text x="428" y="348" font-size="11.5" fill="currentColor">1000</text>
  <text x="596" y="348" font-size="11.5" fill="currentColor">10⁴</text>
  <text x="100" y="372" font-size="12.5" fill="currentColor">Figure 4.4. Measured 95th percentile of |X̄ₙ − 0.5| for i.i.d. Uniform[0,1] samples (10⁵ replications per n, seed</text>
  <text x="100" y="388" font-size="12.5" fill="currentColor">20240419) against the Hoeffding and Chebyshev radii at δ = 0.05. All three lines have slope −1/2 on this log-log</text>
  <text x="100" y="399" font-size="11.5" fill="currentColor">plot: the n-dependence is right. Hoeffding sits a constant factor 2.40 above the truth, uniformly in n.</text>
</svg>

**Worked Example 4.12** (Is Hoeffding's slack constant, or does it grow?). Figure 4.4 answers this by simulation: for i.i.d. $\mathrm{Uniform}[0,1]$ data, $10^5$ replications per $n$, seed $20240419$:

| $n$ | measured $95$th pct of $\lvert\bar X_n-\tfrac12\rvert$ | Hoeffding radius | ratio |
|---|---|---|---|
| $10$ | $0.17877$ | $0.42947$ | $2.40$ |
| $100$ | $0.05662$ | $0.13581$ | $2.40$ |
| $1{,}000$ | $0.01791$ | $0.04295$ | $2.40$ |
| $10{,}000$ | $0.00565$ | $0.01358$ | $2.40$ |

The ratio is $2.40$ at every sample size, to three digits. Hoeffding gets the *rate* exactly right and loses only a constant, and you can even predict the constant: the true radius is $\approx z_{0.975}\sigma/\sqrt n=1.96/\sqrt{12n}$ and Hoeffding's is $\sqrt{\log 40/(2n)}$, whose ratio is $\sqrt{\log 40 \cdot 12/2}/1.96=2.40$. That factor decomposes exactly: $\sqrt{0.25/(1/12)}=\sqrt3=1.732$ is the price of the variance proxy against the true variance, and $\sqrt{2\log(2/\delta)}/z_{0.975}=2.716/1.96=1.386$ is the price of an exponential tail bound against the Gaussian quantile. Their product is $2.400$.

A fuller table across $n$ and $\delta$, including the one-sided versions and the comparison with Chebyshev, appears in Lab B1, where you will reproduce these numbers yourself. (One warning to carry into that lab: the ratio quoted here, $2.40$, is for the *two-sided* radius $\sqrt{\log(2/\delta)/2n}$ against the two-sided empirical quantile. Lab B1 works one-sided, where the same comparison gives $2.58$ at $\delta=0.05$. Both are correct; the convention has to be stated, and papers that omit it are a common source of confusion.)

**How much is actually left on the table.** It would be easy to read Worked Example 4.12 as "Hoeffding is $2.4\times$ too weak, so a cleverer inequality would do better," and that reading is wrong in an important way. Two facts bound how much improvement is possible.

First, the *exponential rate* of the Chernoff bound is not improvable at all. Cramér's theorem (see §4.12) says that for i.i.d. summands,
$$\lim_{n\to\infty}\frac1n\log\mathbb{P}\big(\bar X_n\ge t\big)=-\psi^*(t),$$
a genuine limit, with the Chernoff bound of Theorem 4.14 supplying the upper half and a matching lower bound supplying the other. So the *shape* $e^{-n\psi^*(t)}$ is exactly right; there is no sharper exponential decay to find. Worked Example 4.7 saw this concretely: the Chernoff bound $0.133514$ and the truth $0.028444$ differ by a factor of $4.7$ at $n=100$, and that factor grows only polynomially in $n$, not exponentially — it is the $1/\sqrt{n}$ prefactor that Chernoff discards, not the rate.

Second, Hoeffding's remaining $2.4$ is two identifiable constants, both explained above: $\sqrt3$ for using variance proxy $1/4$ instead of the true $1/12$, and $1.386$ for the tail-bound-versus-Gaussian-quantile conversion. The first is removed by Bernstein when you know the variance; the second is *irreducible* for a bound that must hold at every $n$ with no asymptotics, because the Gaussian quantile is a limit statement and a finite-$n$ guarantee must pay something for it.

So the correct summary is: these inequalities are tight in rate, tight in their dependence on $n$ and $\delta$, and loose only by small constants traceable to specific hypotheses. Keep this in mind through §4.9, because it means the catastrophic failures in §4.11 *cannot* be blamed on weak inequalities. Something else is wrong there, and knowing that the tools are sharp is what makes §4.11's conclusion forced rather than merely suggested. So far, though, "the tools are sharp" has rested on a cited fact (Cramér's theorem). Exercise A19 lets you *prove* the sharpest instance yourself from scratch: a three-line lower bound on the Gaussian tail shows that for Gaussian data no bound of any form can have a smaller radius than $(1-o(1))\sqrt{2\log(1/\delta)}\,\sigma/\sqrt n$, so Hoeffding's dependence on both $n$ and $\delta$ is not merely good, it is unimprovable — and the same exercise shows that the $\sqrt{\log N}$ price of controlling $N$ things at once (§4.7, Lemma 4.39) is also genuinely necessary, not union-bound slack.

One forward pointer before we leave sums of scalars, because the papers this chapter prepares you to read use it constantly. Everything in §4.3–§4.5 extends to sums of independent random *matrices*: the Chernoff template survives with the scalar exponential replaced by the matrix exponential and the expectation by an operator-norm bound, yielding the **matrix Bernstein inequality** — for independent, centered, symmetric $d\times d$ matrices $X_i$ with $\lVert X_i\rVert\le M$, the operator norm $\lVert\sum_iX_i\rVert$ obeys a Bernstein-shaped tail with an extra factor of $d$ (entering, as always, only logarithmically, as $\log d$ — the union bound over dimensions in disguise). It is the standard tool for sample covariance matrices, random-feature approximations, and graph Laplacians; the canonical source is Tropp's monograph, listed in §4.12. We do not develop it here because Chapter 5 builds the multivariate machinery it needs, but you should recognize its shape on sight: it is this chapter's Theorem 4.24 wearing an operator norm.

> **Self-check 4.5**
>
> 1. In Theorem 4.23, why does the bound depend on $\sum_i(b_i-a_i)^2$ rather than on $n$ directly? What happens if one variable has a huge range and the rest are constants?
> 2. For $[0,1]$-valued losses, at what value of $\sigma^2$ do the leading terms of Bernstein and Hoeffding coincide?
> 3. Bennett needs the same inputs as Bernstein. Why would anyone ever use Bernstein?
> 4. You measure a $0.2\%$ error rate on $10^5$ test examples. Give the Hoeffding and Bernstein radii at $\delta=0.05$ (use $\sigma^2=p(1-p)$ with $p=0.002$), and say which one you would report and why.
## 4.6 Bounded differences: concentration beyond sums

### 4.6.1 Why we need to leave sums behind

Everything so far concerned $\bar X_n$, an average. But the quantities we care about in learning theory are usually not averages. The test error of the model *your training procedure produced* is a complicated function of all $n$ training points at once. The sample median is not an average. The largest eigenvalue of a sample covariance matrix is not an average. And the object at the heart of §4.7–§4.10,
$$\Phi(z_1,\dots,z_n)\ :=\ \sup_{h\in\mathcal H}\Big\lvert \hat R_n(h)-R(h)\Big\rvert,$$
is emphatically not an average — it is a supremum over a possibly uncountable family, and Chapter 1 §1.8.3 warned us it need not even be measurable without extra care.

The insight, due to McDiarmid, is that the *only* structural property that made Hoeffding work was this: changing one input changes the output a little. An average of $[0,1]$ variables changes by at most $1/n$ when one summand is replaced. Any function with that property concentrates just as well, whatever else it does.

**Definition 4.27** (Bounded differences). Let $f:\mathcal{Z}^n\to\mathbb{R}$ and $c_1,\dots,c_n\ge0$. We say $f$ has the *bounded differences property* with constants $c_i$ if for every $i$, every $z_1,\dots,z_n\in\mathcal Z$ and every $z_i'\in\mathcal Z$,
$$\big\lvert f(z_1,\dots,z_i,\dots,z_n)-f(z_1,\dots,z_i',\dots,z_n)\big\rvert\ \le\ c_i .$$
The vector $(c_i)$ measures the *sensitivity* of $f$ to each coordinate.

### 4.6.2 McDiarmid's inequality

The proof route is the *Doob martingale*: reveal the coordinates one at a time and track how the conditional expectation of $f$ moves. Each step moves by a bounded amount, so the whole walk concentrates. The martingale machinery — filtrations as accumulating information (a Chapter 1 theme), conditional expectation, and the conditional version of Hoeffding's lemma — is developed in Chapter 6. To keep this chapter self-contained but honest, I state the martingale inequality here and *do not prove it*; it is listed in §4.12 as a result used but not proved. Everything else in this section is proved in full.

**Theorem 4.28** (Azuma–Hoeffding, interval form; proof in Ch. 6). Let $(\mathcal F_k)_{k=0}^n$ be a filtration and let $(V_k)_{k=0}^n$ be a martingale with respect to it, i.e. $V_k$ is $\mathcal F_k$-measurable with $\mathbb{E}[V_k\mid\mathcal F_{k-1}]=V_{k-1}$. Suppose that for each $k$ there is an $\mathcal F_{k-1}$-measurable random variable $A_k$ and a constant $L_k$ with
$$A_k\ \le\ V_k-V_{k-1}\ \le\ A_k+L_k\qquad\text{a.s.}$$
Then for all $t\ge0$,
$$\mathbb{P}\big(V_n-V_0\ge t\big)\ \le\ \exp\Big(-\frac{2t^2}{\sum_{k=1}^nL_k^2}\Big).$$

The shape is Hoeffding's, and so is the proof: apply the conditional form of Lemma 4.19 at each step to get $\mathbb{E}[e^{\lambda(V_k-V_{k-1})}\mid\mathcal F_{k-1}]\le e^{\lambda^2L_k^2/8}$, then peel the exponentials off one at a time by the tower property. What makes it a Chapter 6 result rather than this chapter's is only that "conditional expectation given a $\sigma$-algebra" has not yet been constructed.

**Theorem 4.29** (McDiarmid's inequality / bounded differences inequality). Let $Z_1,\dots,Z_n$ be *independent* random variables taking values in $\mathcal Z$, and let $f:\mathcal Z^n\to\mathbb{R}$ have the bounded differences property with constants $c_1,\dots,c_n$. Write $Z=(Z_1,\dots,Z_n)$. Then for all $t\ge0$,
$$\mathbb{P}\big(f(Z)-\mathbb{E}[f(Z)]\ \ge\ t\big)\ \le\ \exp\Big(-\frac{2t^2}{\sum_ic_i^2}\Big),$$
and the same bound holds for the lower tail; hence with probability at least $1-\delta$,
$$\big\lvert f(Z)-\mathbb{E}[f(Z)]\big\rvert\ \le\ \sqrt{\frac{\big(\sum_ic_i^2\big)\log(2/\delta)}{2}} .$$

*Proof.* Set $\mathcal F_k=\sigma(Z_1,\dots,Z_k)$, with $\mathcal F_0$ trivial, and define the *Doob martingale*
$$V_k:=\mathbb{E}\big[f(Z)\mid\mathcal F_k\big],\qquad k=0,\dots,n,$$
so that $V_0=\mathbb{E}[f(Z)]$ and $V_n=f(Z)$ (the latter because $f(Z)$ is $\mathcal F_n$-measurable). That $(V_k)$ is a martingale is the tower property. It remains to show the increments lie in an interval of length $c_k$ determined by the past.

Because the $Z_i$ are independent, we can write $V_k$ explicitly. Define
$$g_k(z_1,\dots,z_k):=\mathbb{E}\big[f(z_1,\dots,z_k,Z_{k+1},\dots,Z_n)\big],$$
the expectation being over the remaining coordinates only. By Fubini's theorem for product measures (Ch. 2) and independence, $V_k=g_k(Z_1,\dots,Z_k)$ a.s. Then
$$V_k-V_{k-1}=g_k(Z_{1:k})-\int g_k(Z_{1:k-1},u)\,dP_{Z_k}(u)=:D_k .$$
Now define, with $Z_{1:k-1}$ held fixed (hence $\mathcal F_{k-1}$-measurably),
$$A_k:=\inf_{u}g_k(Z_{1:k-1},u)-\int g_k(Z_{1:k-1},u)\,dP_{Z_k}(u),\qquad B_k:=\sup_{u}g_k(Z_{1:k-1},u)-\int g_k(Z_{1:k-1},u)\,dP_{Z_k}(u).$$
Clearly $A_k\le D_k\le B_k$, and both $A_k,B_k$ are $\mathcal F_{k-1}$-measurable. Moreover, for any $u,v$, the bounded differences property applied inside the expectation gives
$$\big\lvert g_k(z_{1:k-1},u)-g_k(z_{1:k-1},v)\big\rvert=\Big\lvert\mathbb{E}\big[f(z_{1:k-1},u,Z_{k+1:n})-f(z_{1:k-1},v,Z_{k+1:n})\big]\Big\rvert\le c_k,$$
so $B_k-A_k\le c_k$. Theorem 4.28 with $L_k=c_k$ finishes the upper tail; apply it to $-f$ (which has the same constants) for the lower tail, then add. $\blacksquare$

That is the whole content: **independence plus low sensitivity implies concentration, with no requirement that $f$ be a sum, be smooth, or be computable in closed form.** McDiarmid is the reason learning theory can say anything at all about the output of a training algorithm.

**Worked Example 4.13** (McDiarmid recovers Hoeffding, and then goes further). First, the sanity check. For a fixed hypothesis $h$ and $[0,1]$-valued loss, $f(z_1,\dots,z_n)=\frac1n\sum_i\ell(h,z_i)$ changes by at most $1/n$ when one $z_i$ is replaced, so $c_i=1/n$ and $\sum_ic_i^2=1/n$. Theorem 4.29 gives $2\exp(-2nt^2)$, which is Theorem 4.23 exactly. McDiarmid is a strict generalization of Hoeffding, at the price of importing Theorem 4.28.

Now the step that Hoeffding cannot take. Let $\Phi(z_{1:n})=\sup_{h\in\mathcal H}\lvert\hat R_n(h)-R(h)\rvert$. Replace $z_i$ by $z_i'$. For each individual $h$, $\hat R_n(h)$ moves by at most $1/n$; a supremum of functions that each move by at most $1/n$ itself moves by at most $1/n$ (because $\sup_h a_h\le\sup_h(b_h+1/n)=\sup_hb_h+1/n$, and symmetrically). So $\Phi$ has bounded differences with $c_i=1/n$, and Theorem 4.29 gives
$$\mathbb{P}\Big(\Phi\ \ge\ \mathbb{E}[\Phi]+t\Big)\ \le\ e^{-2nt^2}\quad\text{i.e.}\quad \Phi\ \le\ \mathbb{E}[\Phi]+\sqrt{\frac{\log(1/\delta)}{2n}}\ \text{ w.p. } 1-\delta .$$
This is the pivot on which all of §4.8 turns: **the hard part of generalization is now computing $\mathbb{E}[\Phi]$, one deterministic number, rather than controlling a random supremum.** Rademacher complexity is precisely a tool for bounding $\mathbb{E}[\Phi]$. Note also that the size of $\mathcal H$ has completely disappeared from this step — infinite classes are fine here; the difficulty has been relocated, not solved.

(A measurability caveat, as promised in Chapter 1: for $\Phi$ to be a random variable at all when $\mathcal H$ is uncountable, one needs a condition such as $\mathcal H$ being pointwise separable — a countable subfamily whose pointwise closure is $\mathcal H$ — or continuity of $h\mapsto\ell(h,z)$ in a parametrization with a countable dense set. Chapter 1 §1.8.3 explained why countability is load-bearing. Papers almost never mention this; it is genuinely harmless for the classes used in practice, but it is an assumption, not a theorem.)

### 4.6.3 When McDiarmid fails: Efron–Stein

McDiarmid charges you the *worst case over all datasets*, and sometimes that is catastrophically pessimistic.

**Theorem 4.30** (Efron–Stein inequality; proof in Ch. 6). Let $Z_1,\dots,Z_n$ be independent, $Z=(Z_1,\dots,Z_n)$, and let $f$ be such that $f(Z)$ is square-integrable. Let $Z^{(i)}$ denote $Z$ with the $i$-th coordinate replaced by an independent copy $Z_i'$. Then
$$\mathrm{Var}\big(f(Z)\big)\ \le\ \frac12\sum_{i=1}^n\mathbb{E}\Big[\big(f(Z)-f(Z^{(i)})\big)^2\Big].$$

The right-hand side is an *average-case* sensitivity, not a worst case, and that is the entire difference. Efron–Stein bounds only the variance, not the tail; converting it to a tail bound requires Chebyshev (Corollary 4.9), so you get polynomial confidence dependence rather than exponential. There is a large literature on entropy methods that recovers exponential bounds under average-case conditions; §4.12 points to it.

**Worked Example 4.14** (The sample median: McDiarmid is useless, Efron–Stein is nearly exact). Let $Z_1,\dots,Z_n$ be i.i.d. $\mathrm{Uniform}[0,1]$ with $n=101$, and let $f$ be the sample median.

*McDiarmid.* What is $c_i$? In the worst case, moving one point moves the median from one order statistic to the next, and if the data are adversarially arranged (say $50$ points at $0$ and $50$ at $1$), moving the last point moves the median from $0$ to $1$. So $c_i=1$, the trivial bound, and $\sum_ic_i^2=101$. Theorem 4.29 then gives a confidence radius of
$$\sqrt{\frac{101\cdot\log(2/0.05)}{2}}\ =\ 13.65,$$
a bound on a quantity that lives in $[0,1]$. It is not merely loose; it is vacuous.

*Reality.* Simulating $2\times10^5$ datasets (seed $7$): the standard deviation of the sample median is $0.04927$, matching the asymptotic prediction $\frac{1}{2f(m)\sqrt n}=\frac{1}{2\sqrt{101}}=0.04975$, and the $95$th percentile of $\lvert f-\tfrac12\rvert$ is $0.09643$. The median concentrates perfectly well; McDiarmid simply cannot see it.

*Efron–Stein.* Estimating $\mathbb{E}[(f(Z)-f(Z^{(1)}))^2]$ over $2\times10^4$ datasets gives $9.342\times10^{-5}$, so Theorem 4.30 bounds the variance by $\frac{101}{2}\cdot9.342\times10^{-5}=0.004718$, against a true variance of $0.0024275$. The bound is off by a factor of $1.94$ — for a variance bound, that is excellent.

*Why the gap between the two approaches is so large.* The worst-case sensitivity is $1$, but the *typical* sensitivity is tiny: over $5000$ simulated datasets, the largest median shift achievable by moving a single point anywhere in $[0,1]$ averaged $0.01502$ and never exceeded $0.0852$. Bad configurations exist but have negligible probability. Whenever a McDiarmid-style bound looks absurd, ask whether you are being charged for a dataset that will never occur.

### 4.6.4 Sensitivity is stability is privacy

Definition 4.27's constants have two other names in machine learning, and recognizing them as the same object is worth more than any single theorem here.

**Definition 4.31** (Uniform stability). A learning algorithm $A$ mapping datasets $S\in\mathcal Z^n$ to hypotheses is *$\beta$-uniformly stable* with respect to a loss $\ell$ if for every pair of datasets $S,S'$ differing in exactly one example and every $z\in\mathcal Z$,
$$\big\lvert \ell\big(A(S),z\big)-\ell\big(A(S'),z\big)\big\rvert\ \le\ \beta .$$

**Proposition 4.32** (Stability implies generalization in expectation). Let $A$ be $\beta$-uniformly stable and let $S=(Z_1,\dots,Z_n)$ be i.i.d. Then
$$\Big\lvert\ \mathbb{E}\big[R(A(S))\big]-\mathbb{E}\big[\hat R_S(A(S))\big]\ \Big\rvert\ \le\ \beta .$$

*Proof.* Let $Z'$ be an independent fresh sample and let $S^{(i)}$ be $S$ with $Z_i$ replaced by $Z'$. By definition $R(A(S))=\mathbb{E}_{Z'}[\ell(A(S),Z')]$, so
$$\mathbb{E}\big[R(A(S))\big]-\mathbb{E}\big[\hat R_S(A(S))\big]=\frac1n\sum_{i=1}^n\Big(\mathbb{E}\big[\ell(A(S),Z')\big]-\mathbb{E}\big[\ell(A(S),Z_i)\big]\Big).$$
Fix $i$. The pair $(S,Z')$ and the pair $(S^{(i)},Z_i)$ have the same joint law: both consist of $n+1$ i.i.d. draws, with the roles of two of them exchanged (this is exchangeability of $P^{\otimes(n+1)}$, a consequence of the product structure from Ch. 2). Therefore
$$\mathbb{E}\big[\ell(A(S),Z_i)\big]=\mathbb{E}\big[\ell(A(S^{(i)}),Z')\big],$$
and hence
$$\mathbb{E}\big[\ell(A(S),Z')\big]-\mathbb{E}\big[\ell(A(S),Z_i)\big]=\mathbb{E}\Big[\ell(A(S),Z')-\ell(A(S^{(i)}),Z')\Big],$$
whose absolute value is at most $\beta$ because $S$ and $S^{(i)}$ differ in one example. Averaging over $i$ gives the claim. $\blacksquare$

Notice that this bound does *not* involve the size or complexity of the hypothesis class anywhere. It is a completely different route to generalization from §4.7–§4.10, and it is the route that gives non-vacuous results for gradient descent: a $T$-step SGD run on a convex, $L$-Lipschitz, $\gamma$-smooth objective is $O(L^2\sum_t\eta_t/n)$-uniformly stable, so *early stopping and small learning rates generalize*, a conclusion no VC bound will ever produce. Chapter 9 develops the stability–PAC-Bayes connection.

A warning before you extrapolate, because this is a place where the "obvious" next step fails. Proposition 4.32 is a statement *in expectation*, and you might expect a high-probability version to follow by feeding the generalization gap $R(A(S))-\hat R_S(A(S))$ into McDiarmid: it has bounded differences of order $\beta+1/n$, so Theorem 4.29 applies. It does — but the resulting radius is of order $\beta\sqrt n$ (the constants are $c_i\approx\beta$, and McDiarmid charges $\sqrt{\sum_ic_i^2}\approx\beta\sqrt n$), which for the typical stability level $\beta=\Theta(1/n)$ gives a useless $\Theta(1/\sqrt n)$ term that swallows the $\beta$-sized quantity you wanted to certify. Turning $\beta$-uniform stability into a high-probability bound of order $\beta\log n+\sqrt{\log(1/\delta)/n}$ was an open problem for nearly two decades, resolved by Feldman and Vondrák (2019) with tools beyond this chapter. The moral for reading papers: "stable in expectation" and "stable with high probability" are separated by real mathematics, and a paper that proves the first and quietly claims the second has skipped a hard step. The result is listed in §4.12.

**Worked Example 4.15** (The same constant, priced as privacy). In differential privacy, the *global sensitivity* of a statistic $f$ is $\Delta f=\max\lvert f(S)-f(S')\rvert$ over datasets differing in one record — literally Definition 4.27's $c_i$. The Laplace mechanism releases $f(S)+\mathrm{Lap}(\Delta f/\varepsilon)$ and achieves $\varepsilon$-differential privacy.

Take the mean of $n=1000$ values in $[0,1]$, so $\Delta f=1/n=0.001$. At $\varepsilon=1$ the noise has scale $b=0.001$ and standard deviation $b\sqrt2=0.001414$. Compare this with the *statistical* noise already present: for $\mathrm{Uniform}[0,1]$ data the sampling standard deviation of the mean is $\sqrt{(1/12)/1000}=0.009129$. The privacy noise is $15.5\%$ of the sampling noise — essentially free.

Now take the *median* instead. Its global sensitivity is $1$, as Worked Example 4.14 showed, so the Laplace mechanism would add noise of scale $1/\varepsilon$ — completely destroying a quantity in $[0,1]$. This is exactly the McDiarmid-versus-Efron–Stein gap again, and the practitioner's response is the same in both fields: use average-case rather than worst-case sensitivity (smooth sensitivity, the exponential mechanism, or propose-test-release). **One constant, three literatures: bounded differences in concentration, uniform stability in learning theory, global sensitivity in privacy.** When a paper in one of these fields proves a lemma bounding a sensitivity, it has proved a lemma for all three.

> **Self-check 4.6**
>
> 1. Does McDiarmid require the $Z_i$ to be identically distributed? Where would that assumption be used, if anywhere?
> 2. Give a function on $\{0,1\}^n$ with $c_i=1$ for every $i$ that genuinely does not concentrate, showing that McDiarmid's dependence on $\sum c_i^2$ cannot be improved in general.
> 3. In Worked Example 4.13, why does taking a supremum not increase the bounded-differences constants?
> 4. An algorithm that ignores its data entirely (always outputs the same $h$) is $0$-uniformly stable. What does Proposition 4.32 give, and why is it not a useful learning guarantee?
## 4.7 From concentration to generalization: finite classes

### 4.7.1 The setup, stated once and used for the rest of the chapter

**Definition 4.33** (Risk, empirical risk, ERM). Fix a measurable space $\mathcal Z$ of examples (typically $\mathcal Z=\mathcal X\times\mathcal Y$), a distribution $P$ on $\mathcal Z$, a hypothesis class $\mathcal H$ of measurable functions, and a loss $\ell:\mathcal H\times\mathcal Z\to[0,1]$. The *population risk* and *empirical risk* of $h\in\mathcal H$ on a sample $S=(Z_1,\dots,Z_n)\sim P^{\otimes n}$ are
$$R(h):=\mathbb{E}_{Z\sim P}\big[\ell(h,Z)\big],\qquad \hat R_n(h):=\frac1n\sum_{i=1}^n\ell(h,Z_i)=\mathbb{E}_{Z\sim\hat P_n}[\ell(h,Z)],$$
where $\hat P_n=\frac1n\sum_i\delta_{Z_i}$ is the empirical distribution (Ch. 1 notation). An *empirical risk minimizer* is any $\hat h\in\arg\min_{h\in\mathcal H}\hat R_n(h)$. We write $h^\star\in\arg\min_{h\in\mathcal H}R(h)$ for a population-optimal hypothesis in the class.

Two things are worth saying out loud. First, $\hat R_n(h)$ is what you can compute and $R(h)$ is what you care about; the entire subject is the gap. Second, $\hat R_n$ is an integral against the *random measure* $\hat P_n$, so a bound that holds for a fixed $h$ says nothing about $\hat h$, because $\hat h$ was chosen using the very sample we are averaging over. That is the trap, and the union bound is the escape.

### 4.7.2 The first generalization bound

**Theorem 4.34** (Uniform convergence for finite classes). Let $\mathcal H$ be finite with $\lvert\mathcal H\rvert=N$, let $\ell$ take values in $[0,1]$, and let $S\sim P^{\otimes n}$. Then for any $\delta\in(0,1)$, with probability at least $1-\delta$,
$$\sup_{h\in\mathcal H}\big\lvert\hat R_n(h)-R(h)\big\rvert\ \le\ \varepsilon(n,\delta):=\sqrt{\frac{\log N+\log(2/\delta)}{2n}} .$$

*Proof.* Fix $h\in\mathcal H$. The variables $\ell(h,Z_1),\dots,\ell(h,Z_n)$ are i.i.d. (a measurable function of i.i.d. variables, Proposition 1.15) with values in $[0,1]$ and mean $R(h)$. Theorem 4.23 gives, for any $t>0$,
$$\mathbb{P}\Big(\big\lvert\hat R_n(h)-R(h)\big\rvert\ge t\Big)\ \le\ 2e^{-2nt^2}=:\ \delta_h .$$
Let $A_h$ be that event. The event that *some* hypothesis fails is $\bigcup_{h\in\mathcal H}A_h$, and by the union bound (Proposition 1.8(2)),
$$\mathbb{P}\Big(\bigcup_{h\in\mathcal H}A_h\Big)\ \le\ \sum_{h\in\mathcal H}\mathbb{P}(A_h)\ \le\ 2Ne^{-2nt^2}.$$
Set the right-hand side equal to $\delta$ and solve for $t$: $t=\sqrt{\frac{\log(2N/\delta)}{2n}}=\sqrt{\frac{\log N+\log(2/\delta)}{2n}}$. On the complement of the union — an event of probability at least $1-\delta$ — every $h$ satisfies $\lvert\hat R_n(h)-R(h)\rvert<t$, hence so does the supremum, which is a maximum over a finite set and therefore measurable (Proposition 1.15). $\blacksquare$

**Corollary 4.35** (ERM is nearly optimal; sample complexity). In the setting of Theorem 4.34, with probability at least $1-\delta$ the empirical risk minimizer $\hat h$ satisfies
$$R(\hat h)\ \le\ R(h^\star)+2\varepsilon(n,\delta),$$
and consequently $n\ \ge\ \dfrac{2\big(\log N+\log(2/\delta)\big)}{\varepsilon^2}$ samples suffice for excess risk at most $\varepsilon$ with probability $1-\delta$.

*Proof.* Work on the event of Theorem 4.34. Then
$$R(\hat h)\ \le\ \hat R_n(\hat h)+\varepsilon\ \le\ \hat R_n(h^\star)+\varepsilon\ \le\ R(h^\star)+2\varepsilon,$$
where the first and third steps are uniform convergence applied to $\hat h$ and to $h^\star$ respectively, and the middle step is the definition of $\hat h$ as a minimizer. Setting $2\varepsilon(n,\delta)\le\varepsilon$ and solving gives the sample complexity. $\blacksquare$

Read the chain of three inequalities again, because it is the archetype of every ERM analysis: *go up* from population to empirical risk at $\hat h$, *use optimality* to swap $\hat h$ for $h^\star$, *come down* from empirical to population at $h^\star$. You pay $\varepsilon$ twice, once in each direction, and uniform convergence is needed only for the first step (where the hypothesis is sample-dependent); the third step could have used a single application of Hoeffding, since $h^\star$ is fixed in advance.

Now look at the shape of $\varepsilon(n,\delta)=\sqrt{(\log N+\log(2/\delta))/2n}$. Three separate miracles are packed into it.

The dependence on $\delta$ is $\sqrt{\log(1/\delta)}$, which is why Worked Example 4.6 mattered: with Chebyshev per hypothesis we would have gotten $\sqrt{N/(2n\delta)}$ — a factor $\sqrt N$ instead of $\sqrt{\log N}$ — and no bound would be usable for $N$ beyond a few dozen. **Exponential tails are precisely what makes the union bound affordable.**

The dependence on $N$ is *logarithmic*. Doubling the number of hypotheses adds $\log 2$ inside the square root. A class of one million hypotheses at $n=1000$ gives $\varepsilon=0.0936$; the same class at $n=10^5$ gives $\varepsilon=0.00936$. And the $\sqrt{\log N}$ is not an artifact of the union bound's crudeness: Exercise A19(c) proves a matching *lower* bound — for $N$ genuinely different (independent-fluctuation) hypotheses, a $\sqrt{\log N}$ price is unavoidable, and the union bound overcharges only when hypotheses are strongly correlated.

And the dependence on $n$ is the optimal $n^{-1/2}$.

**Worked Example 4.16** (A parameter-counting bound, and its collapse). Any hypothesis class implemented on a computer is finite: a network with $W$ real parameters stored as $32$-bit floats has at most $N=2^{32W}$ distinct hypotheses, so $\log N=32W\log 2$. This is a legitimate application of Theorem 4.34 — it requires nothing but boundedness of the loss and independence of the sample — and it is instructive to see what it gives.

Take a small MLP for MNIST: $784\to64\to10$ fully connected with biases, so
$$W=784\cdot64+64+64\cdot10+10=50{,}890 \quad\text{parameters},\qquad \log N=32\cdot50{,}890\cdot\log2=1.129\times10^6 .$$
At $n=60{,}000$ (the MNIST training set size) and $\delta=0.05$,
$$\varepsilon=\sqrt{\frac{1.129\times10^6+\log(40)}{2\cdot60{,}000}}\ =\ 3.067 .$$
The bound says the gap between training and test error is at most $3.067$. Since both are in $[0,1]$, the gap is at most $1$ trivially. **The bound is vacuous**, and by a wide margin: to reach $\varepsilon=0.05$ this bound needs $n\ge2.26\times10^8$, which is $3{,}763$ MNIST training sets.

Two conclusions, and it matters that they are different. (i) Parameter counting is a bad way to measure capacity — this is the honest one, and §4.8–§4.10 replace counting with geometry. (ii) But note also *what direction* the failure runs: the bound is an upper bound, so its vacuity tells us nothing about whether the network generalizes. It generalizes fine in practice (a network this size reaches roughly $2\%$ test error). The theory is silent, not wrong. Section 4.11 examines whether better theory can fix this, and reports that the question is open.

> **Common Confusion 4.6**
>
> *"For an infinite class I can just take a limit of finite classes, or use a fine discretization, and the union bound will still work."*
>
> The limit does not do what you want, and seeing exactly why is the entry ticket to §4.8–§4.10. The union bound (Proposition 1.8(2)) is countably additive-safe: $\mathbb{P}(\bigcup_nA_n)\le\sum_n\mathbb{P}(A_n)$ holds for countably many events. But the bound is *vacuous once $\sum_n\mathbb{P}(A_n)\ge1$*, which Chapter 1 flagged. With $N$ hypotheses each failing with probability $2e^{-2nt^2}$, the sum reaches $1$ as soon as $N\ge\frac12e^{2nt^2}$ — so at $n=1000$ and $t=0.05$ you may union over at most $e^{5}/2=74$ hypotheses before the bound says "probability at most $1$." Taking $N\to\infty$ gives $\infty$.
>
> Discretization is more subtle and it is genuinely the right idea, but it is not free. If you discretize a continuum class $\mathcal H$ to a finite $\varepsilon$-net $\mathcal H_\varepsilon$, you get a bound of the form $\varepsilon_{\text{approx}}+\sqrt{\log\lvert\mathcal H_\varepsilon\rvert/2n}$, and you must trade the approximation error against the log-cardinality. That trade-off is §4.10, and doing it *at one scale* is provably worse than doing it *at all scales simultaneously*, which is chaining. Also, and separately: for an uncountable $\mathcal H$ the supremum $\sup_h\lvert\hat R_n(h)-R(h)\rvert$ may fail to be measurable at all (Ch. 1 §1.8.3), so "the probability that uniform convergence fails" may not be defined without a separability assumption. Papers universally suppress this, and for parametric classes with continuous loss it is harmless — but it is why the theorems in the next sections carry hypotheses like "pointwise measurable class."

> **Self-check 4.7**
>
> 1. In Corollary 4.35, why do we pay $2\varepsilon$ rather than $\varepsilon$? Which of the three inequalities in the chain needs uniform convergence?
> 2. You have $\lvert\mathcal H\rvert=100$ and want $\varepsilon=0.05$ at $\delta=0.01$. How many samples does Corollary 4.35 require?
> 3. If you increase $\lvert\mathcal H\rvert$ from $10^3$ to $10^6$ at fixed $n$, by what factor does $\varepsilon$ grow?
> 4. Explain in one sentence why Theorem 4.34's proof would fail if we tried to apply Hoeffding to $\hat h$ directly.
## 4.8 Rademacher complexity

### 4.8.1 Measuring a class by how well it fits noise

Counting hypotheses is the wrong measure of capacity for two reasons. It is infinite for the classes we use, and it is *insensitive to the data*: a class of a million hypotheses that all agree on your dataset should cost you nothing, but $\log N$ charges you for all million.

The fix is to ask a different question. Instead of "how many hypotheses are there?", ask: **how well can this class fit pure noise?** A class that can match random $\pm1$ labels on your $n$ points is one whose empirical risk tells you nothing; a class that cannot is one whose empirical risk is meaningful. That intuition is made precise by attaching a random sign to each data point and seeing how large a correlation the class can achieve.

Throughout this section $\mathcal F$ denotes a class of functions $f:\mathcal Z\to\mathbb{R}$; in applications $\mathcal F=\ell\circ\mathcal H:=\{z\mapsto\ell(h,z):h\in\mathcal H\}$, the *loss class*. We add two symbols to the chapter's notation: $\sigma=(\sigma_1,\dots,\sigma_n)$ for i.i.d. Rademacher variables ($\mathbb{P}(\sigma_i=\pm1)=\tfrac12$, independent of everything else), and $\mathfrak{R}$ for Rademacher complexity.

**Definition 4.36** (Rademacher complexity). Let $\mathcal F$ be a class of functions $\mathcal Z\to\mathbb{R}$ and let $z_1,\dots,z_n\in\mathcal Z$ be fixed points. The *empirical Rademacher complexity* of $\mathcal F$ on those points is
$$\hat{\mathfrak{R}}_n(\mathcal F;z_{1:n}):=\mathbb{E}_\sigma\Big[\sup_{f\in\mathcal F}\frac1n\sum_{i=1}^n\sigma_i f(z_i)\Big],$$
and for $S=(Z_1,\dots,Z_n)\sim P^{\otimes n}$ the *expected* (or population) Rademacher complexity is
$$\mathfrak{R}_n(\mathcal F):=\mathbb{E}_S\big[\hat{\mathfrak{R}}_n(\mathcal F;Z_{1:n})\big]=\mathbb{E}_{S,\sigma}\Big[\sup_{f\in\mathcal F}\frac1n\sum_{i=1}^n\sigma_if(Z_i)\Big].$$
(We assume throughout that $\mathcal F$ is such that these suprema are measurable — see Common Confusion 4.6.)

Some orientation before the theorems. If $\mathcal F=\{f\}$ is a single function, $\mathbb{E}_\sigma[\frac1n\sum_i\sigma_if(z_i)]=0$: no class, no capacity. If $\mathcal F$ contains, for every sign pattern $s\in\{\pm1\}^n$, a function with $f(z_i)=s_i$, then the supremum is $1$ for every $\sigma$ and $\hat{\mathfrak{R}}_n=1$: maximal capacity, and this is what "shattering" will mean in §4.9. In general $\hat{\mathfrak{R}}_n$ interpolates, and the useful classes have $\hat{\mathfrak{R}}_n=O(1/\sqrt n)$.

### 4.8.2 Symmetrization

**Theorem 4.37** (Symmetrization). Let $\mathcal F$ be a class of functions $\mathcal Z\to\mathbb{R}$ and $S=(Z_1,\dots,Z_n)\sim P^{\otimes n}$. Write $Pf=\mathbb{E}_{Z\sim P}[f(Z)]$ and $P_nf=\frac1n\sum_if(Z_i)$. Then
$$\mathbb{E}_S\Big[\sup_{f\in\mathcal F}\big(Pf-P_nf\big)\Big]\ \le\ 2\,\mathfrak{R}_n(\mathcal F),$$
and the same bound holds with $\sup_f(P_nf-Pf)$ on the left. For the absolute value one has $\mathbb{E}_S\big[\sup_f\lvert P_nf-Pf\rvert\big]\le 2\,\mathfrak{R}_n(\mathcal F\cup(-\mathcal F))$, where $-\mathcal F=\{-f:f\in\mathcal F\}$.

*Proof.* Introduce a *ghost sample* $S'=(Z_1',\dots,Z_n')\sim P^{\otimes n}$, independent of $S$, and write $P_n'f=\frac1n\sum_if(Z_i')$. Since $Pf=\mathbb{E}_{S'}[P_n'f]$ for each fixed $f$,
$$\sup_{f}\big(Pf-P_nf\big)=\sup_f\ \mathbb{E}_{S'}\big[P_n'f-P_nf\big]\ \le\ \mathbb{E}_{S'}\Big[\sup_f\big(P_n'f-P_nf\big)\Big],$$
where the inequality is $\sup_f\mathbb{E}[\cdot]\le\mathbb{E}[\sup_f\cdot]$, valid because for each $f$ the integrand is dominated by the supremum. Taking $\mathbb{E}_S$ and using Fubini–Tonelli (Ch. 2) to combine the two expectations,
$$\mathbb{E}_S\Big[\sup_f(Pf-P_nf)\Big]\ \le\ \mathbb{E}_{S,S'}\Big[\sup_f\frac1n\sum_{i=1}^n\big(f(Z_i')-f(Z_i)\big)\Big].$$
Now the symmetrization step. Let $\sigma$ be i.i.d. Rademacher, independent of $(S,S')$. For any fixed $s\in\{\pm1\}^n$, the random vector
$$\Big(\big(Z_i',Z_i\big)\ \text{if } s_i=+1;\quad \big(Z_i,Z_i'\big)\ \text{if } s_i=-1\Big)_{i=1}^n$$
has the same law as $\big((Z_i',Z_i)\big)_{i=1}^n$, because the $2n$ variables are i.i.d. and we have merely swapped members of some pairs (the product measure $P^{\otimes 2n}$ is invariant under permutation of coordinates, Ch. 2). Hence for every fixed $s$,
$$\mathbb{E}_{S,S'}\Big[\sup_f\frac1n\sum_i\big(f(Z_i')-f(Z_i)\big)\Big]=\mathbb{E}_{S,S'}\Big[\sup_f\frac1n\sum_i s_i\big(f(Z_i')-f(Z_i)\big)\Big],$$
and averaging over $s$ drawn as $\sigma$ leaves the left side unchanged:
$$=\ \mathbb{E}_{\sigma,S,S'}\Big[\sup_f\frac1n\sum_i \sigma_i\big(f(Z_i')-f(Z_i)\big)\Big].$$
Finally split the supremum, using $\sup_f(a_f+b_f)\le\sup_fa_f+\sup_fb_f$:
$$\le\ \mathbb{E}_{\sigma,S'}\Big[\sup_f\frac1n\sum_i\sigma_if(Z_i')\Big]+\mathbb{E}_{\sigma,S}\Big[\sup_f\frac1n\sum_i(-\sigma_i)f(Z_i)\Big]=2\,\mathfrak{R}_n(\mathcal F),$$
where the last equality uses that $-\sigma$ has the same law as $\sigma$ and that $S,S'$ have the same law.

For the version with $\sup_f(P_nf-Pf)$ on the left, run the identical argument with the roles of $S$ and $S'$ exchanged; equivalently, apply the result just proved to the class $-\mathcal F$ and note that $\mathfrak{R}_n(-\mathcal F)=\mathfrak{R}_n(\mathcal F)$, because replacing $f$ by $-f$ inside the supremum is the same as replacing $\sigma$ by $-\sigma$, which has the same law. For the absolute value, observe the pointwise identity
$$\sup_{f\in\mathcal F}\big\lvert P_nf-Pf\big\rvert=\sup_{g\in\mathcal F\cup(-\mathcal F)}\big(P_ng-Pg\big),$$
since $\lvert x\rvert=\max(x,-x)$ and $P_n(-f)-P(-f)=-(P_nf-Pf)$; now apply the one-sided result to the class $\mathcal F\cup(-\mathcal F)$. $\blacksquare$

A caution on the absolute-value version, because textbooks routinely blur it: $\mathfrak{R}_n(\mathcal F\cup(-\mathcal F))$ is *not* in general equal to $\mathfrak{R}_n(\mathcal F)$, and the frequently written "$\mathbb{E}\sup_f\lvert P_nf-Pf\rvert\le2\mathfrak{R}_n(\mathcal F)$" needs an extra hypothesis (for instance, that $\mathcal F$ is closed under negation, or contains the zero function so that both one-sided suprema are non-negative and $\mathfrak{R}_n(\mathcal F\cup(-\mathcal F))\le2\mathfrak{R}_n(\mathcal F)$). In practice none of this matters, because the clean route to a two-sided *high-probability* statement is not through the two-sided expectation at all: apply the one-sided bound of Theorem 4.38 to $\mathcal F$ and to $-\mathcal F$ separately, at confidence $\delta/2$ each, and take a union bound over the two events. That is how every two-sided bound in this chapter should be read, and it costs only $\log 2$ inside a square root.

The trick deserves a name and a moment's admiration. We wanted to compare an empirical average to a population mean — an asymmetric comparison involving an unknown $P$. By introducing a second imaginary dataset and then randomly swapping members of matched pairs, we converted it into a *purely symmetric* quantity in which $P$ appears only as the distribution of the points, and the difficulty is now entirely about the class's ability to correlate with noise. **Symmetrization is why Rademacher complexity is the natural complexity measure, rather than one choice among many.**

### 4.8.3 The main bound

**Theorem 4.38** (Rademacher generalization bound). Let $\ell$ take values in $[0,1]$, let $\mathcal F=\ell\circ\mathcal H$, and let $S\sim P^{\otimes n}$. Then with probability at least $1-\delta$, *simultaneously for all* $h\in\mathcal H$,
$$R(h)\ \le\ \hat R_n(h)+2\,\mathfrak{R}_n(\mathcal F)+\sqrt{\frac{\log(1/\delta)}{2n}} ,$$
and also, with probability at least $1-\delta$, simultaneously for all $h\in\mathcal H$,
$$R(h)\ \le\ \hat R_n(h)+2\,\hat{\mathfrak{R}}_n(\mathcal F;Z_{1:n})+3\sqrt{\frac{\log(2/\delta)}{2n}} .$$

*Proof.* Let $\Phi(S):=\sup_{f\in\mathcal F}(Pf-P_nf)$. As computed in Worked Example 4.13, $\Phi$ has bounded differences with $c_i=1/n$ (replacing one $z_i$ moves each $P_nf$ by at most $1/n$ since $f\in[0,1]$, and a supremum inherits the bound). McDiarmid (Theorem 4.29, upper tail) gives: with probability at least $1-\delta$,
$$\Phi(S)\ \le\ \mathbb{E}[\Phi]+\sqrt{\frac{\log(1/\delta)}{2n}}\ \le\ 2\mathfrak{R}_n(\mathcal F)+\sqrt{\frac{\log(1/\delta)}{2n}},$$
the last step by Theorem 4.37. Since $R(h)-\hat R_n(h)\le\Phi(S)$ for every $h$, the first claim follows.

For the second, note that $S\mapsto\hat{\mathfrak{R}}_n(\mathcal F;Z_{1:n})$ also has bounded differences with $c_i=1/n$ (same argument, applied inside $\mathbb{E}_\sigma$), so McDiarmid's lower tail gives $\mathfrak{R}_n(\mathcal F)\le\hat{\mathfrak{R}}_n+\sqrt{\log(2/\delta)/2n}$ with probability $1-\delta/2$; combining with the first display at confidence $1-\delta/2$ and adding the two failure probabilities yields
$$R(h)\le\hat R_n(h)+2\hat{\mathfrak{R}}_n+2\sqrt{\frac{\log(2/\delta)}{2n}}+\sqrt{\frac{\log(2/\delta)}{2n}},$$
which is the claim. $\blacksquare$

The second form is the practically important one: $\hat{\mathfrak{R}}_n$ is computable from your data (estimate the expectation over $\sigma$ by sampling sign vectors and maximizing the correlation — that is a *training run* on random labels), so the bound is *data-dependent*. This is what makes Rademacher complexity more than a theoretical device.

> **Common Confusion 4.7**
>
> *"The empirical Rademacher complexity is computed from my data, so the bound adapts to my data. Then I can look at my data, pick a hypothesis class that fits it, and apply the bound to that class."*
>
> No. Read the quantifiers in Theorem 4.38: the class $\mathcal F$ is fixed *before* $S$ is drawn, and the probability is over $S$. "Data-dependent" means the *bound's numerical value* depends on $S$ — a fixed class can look simple on a lucky sample — not that the class may depend on $S$. If you choose $\mathcal H$ after seeing $S$, you have implicitly searched over a family of classes and the guarantee evaporates exactly as in Common Confusion 4.3.
>
> The legitimate version of what you want is a union bound over a *pre-declared* family of classes with a prior weight on each: fix nested classes $\mathcal H_1\subseteq\mathcal H_2\subseteq\cdots$ and weights $\pi_k$ with $\sum_k\pi_k\le1$, apply Theorem 4.38 to $\mathcal H_k$ at confidence $\delta\pi_k$, and you may then select $k$ after seeing the data at the cost of $\sqrt{\log(1/\pi_k)/2n}$. This is *structural risk minimization*, and it is the finite-class union bound wearing a different hat. PAC-Bayes (Chapter 9) is its continuous limit.

### 4.8.4 Computing Rademacher complexities

**Lemma 4.39** (Massart's finite class lemma). Let $A\subseteq\mathbb{R}^n$ be a finite set with $\lvert A\rvert=N\ge1$ and $\mathcal{R}:=\max_{a\in A}\lVert a\rVert_2$. Then
$$\mathbb{E}_\sigma\Big[\max_{a\in A}\ \langle\sigma,a\rangle\Big]\ \le\ \mathcal{R}\sqrt{2\log N}.$$

*Proof.* For each fixed $a$, $\langle\sigma,a\rangle=\sum_i\sigma_ia_i$ is a sum of independent variables, each $a_i\sigma_i$ being $a_i^2$-sub-Gaussian; by Proposition 4.18 it is $\lVert a\rVert_2^2$-sub-Gaussian, hence $\mathbb{E}_\sigma[e^{\lambda\langle\sigma,a\rangle}]\le e^{\lambda^2\mathcal R^2/2}$ for every $a\in A$. For $\lambda>0$, by Jensen (Theorem 4.3, applied to the convex $\exp$) and then monotonicity,
$$\exp\Big(\lambda\,\mathbb{E}_\sigma\big[\max_a\langle\sigma,a\rangle\big]\Big)\ \le\ \mathbb{E}_\sigma\Big[\exp\big(\lambda\max_a\langle\sigma,a\rangle\big)\Big]=\mathbb{E}_\sigma\Big[\max_a e^{\lambda\langle\sigma,a\rangle}\Big]\ \le\ \sum_{a\in A}\mathbb{E}_\sigma\big[e^{\lambda\langle\sigma,a\rangle}\big]\le Ne^{\lambda^2\mathcal R^2/2}.$$
Take logarithms and divide by $\lambda$:
$$\mathbb{E}_\sigma\Big[\max_a\langle\sigma,a\rangle\Big]\ \le\ \frac{\log N}{\lambda}+\frac{\lambda \mathcal R^2}{2}\qquad\text{for all }\lambda>0 .$$
Optimizing (the right side is minimized at $\lambda^\star=\sqrt{2\log N}/\mathcal R$) gives $\mathcal R\sqrt{2\log N}$. $\blacksquare$

Notice that this is the Chernoff procedure of §4.3 applied to a maximum rather than a tail, and that "sum over $A$" step is the union bound in disguise. The $\sqrt{\log N}$ is the same $\sqrt{\log N}$ as in Theorem 4.34, arrived at from a different direction — and Exercise A19(c) shows it cannot be removed: for $N$ independent Gaussian coordinates the expected maximum really is of order $\sqrt{\log N}$, from below as well as above.

**Worked Example 4.17** (Massart against the truth, and against Theorem 4.34). Numerically, with $A$ a random subset of $\{\pm1\}^{50}$ (so $\mathcal R=\sqrt{50}$) and $2\times10^4$ sampled sign vectors, seed $11$:

| $N$ | $\mathbb{E}_\sigma[\max_a\langle\sigma,a\rangle]$ (simulated) | Massart bound $\mathcal R\sqrt{2\log N}$ |
|---|---|---|
| $2$ | $4.211$ | $8.326$ |
| $10$ | $10.669$ | $15.174$ |
| $100$ | $17.537$ | $21.460$ |
| $1000$ | $22.525$ | $26.283$ |

Loose by a factor of $2$ for tiny $N$, and by only $17\%$ at $N=1000$: the lemma is asymptotically the right order.

Applying it to a finite loss class $\mathcal F$ with $\lvert\mathcal F\rvert=N$ and values in $[0,1]$: the vectors $a_f=(f(z_1),\dots,f(z_n))$ have $\lVert a_f\rVert_2\le\sqrt n$, so
$$\hat{\mathfrak{R}}_n(\mathcal F)=\frac1n\mathbb{E}_\sigma\max_{f}\langle\sigma,a_f\rangle\ \le\ \frac{\sqrt n\sqrt{2\log N}}{n}=\sqrt{\frac{2\log N}{n}} .$$
Feeding this into Theorem 4.38 gives a gap bound of $2\sqrt{2\log N/n}+\sqrt{\log(1/\delta)/2n}$, versus Theorem 4.34's $\sqrt{(\log N+\log(2/\delta))/2n}$. The direct union bound is better by a factor of about $4$ in the leading term. That is worth stating plainly: **on finite classes, Rademacher complexity is strictly worse than the union bound.** Its value is that it does not care whether $\mathcal F$ is finite.

**Lemma 4.40** (Talagrand's contraction lemma; stated, not proved here). Let $\phi:\mathbb{R}\to\mathbb{R}$ be $L$-Lipschitz with $\phi(0)=0$, and let $\mathcal F$ be a class of real-valued functions. Then for any points $z_{1:n}$,
$$\hat{\mathfrak{R}}_n(\phi\circ\mathcal F;z_{1:n})\ \le\ L\,\hat{\mathfrak{R}}_n(\mathcal F;z_{1:n}).$$
See §4.12 for a proof reference.

This is what lets us handle losses and activations: composing with a $1$-Lipschitz ReLU, or with a $1$-Lipschitz hinge or logistic loss, costs nothing.

**Proposition 4.41** (Norm-bounded linear class). Let $\mathcal F_{\mathrm{lin}}=\{x\mapsto\langle w,x\rangle:\lVert w\rVert_2\le B\}$ and let $x_1,\dots,x_n\in\mathbb{R}^d$ with $\max_i\lVert x_i\rVert_2\le X$. Then
$$\hat{\mathfrak{R}}_n(\mathcal F_{\mathrm{lin}};x_{1:n})=\frac{B}{n}\,\mathbb{E}_\sigma\Big\lVert\sum_{i=1}^n\sigma_ix_i\Big\rVert_2\ \le\ \frac{B\sqrt{\sum_i\lVert x_i\rVert^2}}{n}\ \le\ \frac{BX}{\sqrt n}.$$

*Proof.* For fixed $\sigma$, $\sup_{\lVert w\rVert\le B}\frac1n\sum_i\sigma_i\langle w,x_i\rangle=\frac1n\sup_{\lVert w\rVert\le B}\big\langle w,\sum_i\sigma_ix_i\big\rangle=\frac Bn\big\lVert\sum_i\sigma_ix_i\big\rVert_2$ by the definition of the dual norm (Cauchy–Schwarz, attained at $w=B v/\lVert v\rVert$). Taking $\mathbb{E}_\sigma$ gives the identity. For the first inequality, Jensen (Corollary 4.6 with $p=1,q=2$) gives $\mathbb{E}\lVert V\rVert\le(\mathbb{E}\lVert V\rVert^2)^{1/2}$, and
$$\mathbb{E}_\sigma\Big\lVert\sum_i\sigma_ix_i\Big\rVert^2=\sum_{i,j}\mathbb{E}[\sigma_i\sigma_j]\langle x_i,x_j\rangle=\sum_i\lVert x_i\rVert^2,$$
since $\mathbb{E}[\sigma_i\sigma_j]=\mathbf{1}_{\{i=j\}}$. The last inequality bounds each $\lVert x_i\rVert$ by $X$. $\blacksquare$

Combined with Theorem 4.38, this yields the classical margin-style bound for linear predictors: the generalization gap is $O(BX/\sqrt n)$, **with no dependence on the dimension $d$**. That is the single most important qualitative fact about norm-based capacity control, and it is why $\ell_2$ regularization, weight decay, and margin maximization work in high dimension. The dimension has not vanished by magic, though: it hides in $X=\max_i\lVert x_i\rVert_2$, which for raw data with $\Theta(1)$ per-coordinate scale grows like $\sqrt d$.

**Worked Example 4.18** (Estimating $\hat{\mathfrak{R}}_n$ by simulation, and confirming $\sqrt{d/n}$). Take $x_i\sim N(0,I_d)$ i.i.d., $B=1$, $n=200$, and estimate $\hat{\mathfrak{R}}_n=\frac1n\mathbb{E}_\sigma\lVert\sum_i\sigma_ix_i\rVert$ by averaging over $4000$ sampled sign vectors (seed $11$):

| $d$ | $\hat{\mathfrak{R}}_n$ (simulated) | $\sqrt{d/n}$ | Prop. 4.41 bound $\sqrt{\sum_i\lVert x_i\rVert^2}/n$ | ratio to $\sqrt{d/n}$ |
|---|---|---|---|---|
| $1$ | $0.05158$ | $0.07071$ | $0.06524$ | $0.730$ |
| $4$ | $0.13281$ | $0.14142$ | $0.14048$ | $0.939$ |
| $16$ | $0.27774$ | $0.28284$ | $0.28303$ | $0.982$ |
| $64$ | $0.55709$ | $0.56569$ | $0.55994$ | $0.985$ |
| $128$ | $0.79958$ | $0.80000$ | $0.80140$ | $0.999$ |

and at fixed $d=16$, varying $n$ (2000 sign vectors each):

| $n$ | $50$ | $100$ | $200$ | $400$ | $800$ | $1600$ |
|---|---|---|---|---|---|---|
| $\hat{\mathfrak{R}}_n$ | $0.53201$ | $0.38812$ | $0.27843$ | $0.19497$ | $0.13890$ | $0.09776$ |
| $\sqrt{d/n}$ | $0.56569$ | $0.40000$ | $0.28284$ | $0.20000$ | $0.14142$ | $0.10000$ |

Both scalings are confirmed: $\hat{\mathfrak{R}}_n\approx\sqrt{d/n}$ to within $2\%$ once $d\ge4$, and Proposition 4.41's upper bound is nearly tight (the Jensen step $\mathbb{E}\lVert V\rVert\le\sqrt{\mathbb{E}\lVert V\rVert^2}$ costs almost nothing because $\lVert V\rVert$ concentrates — which is Proposition 4.22 again). Note that the $\sqrt d$ here comes from $\lVert x_i\rVert\approx\sqrt d$ for Gaussian data, exactly as the remark after Proposition 4.41 said: for data normalized to $\lVert x\rVert=1$ the complexity would be $1/\sqrt n$, dimension-free.

**Proposition 4.42** (Two-layer network, norm-based). Let $\rho:\mathbb{R}\to\mathbb{R}$ be $1$-Lipschitz with $\rho(0)=0$ (e.g. ReLU), and let
$$\mathcal F_{2\text{-layer}}=\Big\{x\mapsto\sum_{j=1}^m u_j\,\rho\big(\langle w_j,x\rangle\big)\ :\ \lVert u\rVert_1\le B_2,\ \lVert w_j\rVert_2\le B_1\ \forall j\Big\}.$$
Then for any $x_{1:n}$ with $\max_i\lVert x_i\rVert_2\le X$,
$$\hat{\mathfrak{R}}_n\big(\mathcal F_{2\text{-layer}};x_{1:n}\big)\ \le\ \frac{2B_1B_2X}{\sqrt n},$$
independently of the width $m$.

*Proof.* Fix $\sigma$. Because the constraint on $u$ is an $\ell_1$ ball and the objective is linear in $u$, the supremum over $u$ is attained at a vertex:
$$\sup_{\lVert u\rVert_1\le B_2}\sum_j u_j c_j=B_2\max_j\lvert c_j\rvert,\qquad c_j:=\sum_i\sigma_i\rho(\langle w_j,x_i\rangle).$$
Taking also the supremum over the $w_j$, which are unconstrained relative to each other, all $m$ inner maxima become the same problem:
$$\hat{\mathfrak{R}}_n(\mathcal F_{2\text{-layer}})\ \le\ \frac{B_2}{n}\,\mathbb{E}_\sigma\sup_{\lVert w\rVert_2\le B_1}\Big\lvert\sum_i\sigma_i\rho(\langle w,x_i\rangle)\Big\rvert .$$
To remove the absolute value, let $G:=\{x\mapsto\rho(\langle w,x\rangle):\lVert w\rVert\le B_1\}$ and $T(\sigma):=\sup_{g\in G}\sum_i\sigma_ig(x_i)$. Since $w=0$ gives $g\equiv0$ (using $\rho(0)=0$), $T(\sigma)\ge0$ always, and
$$\sup_{g}\Big\lvert\sum_i\sigma_ig(x_i)\Big\rvert=\max\big(T(\sigma),T(-\sigma)\big)\le T(\sigma)+T(-\sigma).$$
As $-\sigma$ has the same law as $\sigma$, $\mathbb{E}_\sigma\sup_g\lvert\cdot\rvert\le2\,\mathbb{E}_\sigma T(\sigma)$. Now apply Lemma 4.40 with $\phi=\rho$ and $L=1$ to strip the activation, then Proposition 4.41:
$$\frac1n\mathbb{E}_\sigma T(\sigma)=\hat{\mathfrak{R}}_n(G)\ \le\ \hat{\mathfrak{R}}_n\big(\{x\mapsto\langle w,x\rangle:\lVert w\rVert\le B_1\}\big)\ \le\ \frac{B_1X}{\sqrt n}.$$
Combining, $\hat{\mathfrak{R}}_n(\mathcal F_{2\text{-layer}})\le B_2\cdot 2\cdot B_1X/\sqrt n$. $\blacksquare$

Now the important reading. The bound $2B_1B_2X/\sqrt n$ contains **no width and no dimension** — it is controlled entirely by norms. This is the template for all modern norm-based bounds for deep networks: peel off layers one at a time with contraction, and multiply the layerwise norms. The catch, which §4.11 will make numerical, is that the product of norms in a trained network is *large*, and it grows during training even as test error falls. With $X=10$, $n=60{,}000$ and modest norms $B_1=1$, $B_2=1$, the Rademacher term is $0.0817$ and the full gap bound $0.168$: non-vacuous, if barely. With $B_1=10$, $B_2=20$ — still unremarkable for a trained network — the same bound is $32.7$.

> **Self-check 4.8**
>
> 1. What is $\hat{\mathfrak{R}}_n(\mathcal F)$ if $\mathcal F$ consists of all functions $\mathcal Z\to[0,1]$?
> 2. Why does $\mathfrak{R}_n$ of a single-function class equal $0$ rather than something positive?
> 3. In Theorem 4.38, why does the second (empirical) form pay $3\sqrt{\log(2/\delta)/2n}$ rather than $\sqrt{\log(1/\delta)/2n}$?
> 4. Proposition 4.42 has no width dependence. Where would the width appear if you constrained $\lVert u\rVert_2\le B_2$ instead of $\lVert u\rVert_1\le B_2$?
## 4.9 VC dimension

### 4.9.1 Counting labelings instead of hypotheses

Rademacher complexity is the sharper tool, but it requires knowing something quantitative about the class — a norm bound, a covering estimate. Vapnik–Chervonenkis theory takes a purely combinatorial route that needs nothing but the class's ability to *label finite sets*, and it was historically first. The observation that starts it: a hypothesis class may be infinite, but on $n$ fixed points it can realize only finitely many labelings — at most $2^n$. If it realizes far fewer than $2^n$, it is effectively a small finite class, and Theorem 4.34 applies with $N$ = number of realizable labelings.

**Definition 4.43** (Restriction, growth function, shattering, VC dimension). Let $\mathcal H$ be a class of functions $\mathcal X\to\{0,1\}$. For points $x_{1:n}=(x_1,\dots,x_n)$, the *restriction* of $\mathcal H$ to $x_{1:n}$ is
$$\mathcal H\big|_{x_{1:n}}:=\big\{\big(h(x_1),\dots,h(x_n)\big):h\in\mathcal H\big\}\subseteq\{0,1\}^n .$$
The *growth function* (or shatter coefficient) is
$$\Pi_{\mathcal H}(n):=\max_{x_1,\dots,x_n\in\mathcal X}\big\lvert\mathcal H\big|_{x_{1:n}}\big\rvert\ \le\ 2^n .$$
We say $\mathcal H$ *shatters* $x_{1:n}$ if $\lvert\mathcal H|_{x_{1:n}}\rvert=2^n$, i.e. every labeling is realized. The *VC dimension* is
$$\mathrm{VC}(\mathcal H):=\sup\big\{n:\Pi_{\mathcal H}(n)=2^n\big\},$$
the size of the largest shattered set (possibly $\infty$).

The connection to §4.8 is immediate and worth stating before any theorems: shattering $x_{1:n}$ means the class contains, for every sign pattern, a hypothesis matching it, so by the remark after Definition 4.36 the empirical Rademacher complexity on those points is as large as possible. VC dimension is the largest $n$ at which the class can still fit *arbitrary* noise perfectly.

<svg viewBox="0 0 720 340" style="max-width:100%;height:auto" role="img" aria-label="Three points in the plane shattered by half-spaces, and four points that cannot be shattered">
  <text x="60" y="40" font-size="13" fill="currentColor">Half-spaces in R²: 3 points in general position ARE shattered</text>
  <line x1="60" y1="200" x2="290" y2="60" stroke="#3b82f6" stroke-width="2"/>
  <line x1="60" y1="120" x2="300" y2="215" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,3"/>
  <circle cx="120" cy="90" r="7" fill="#10b981"/>
  <circle cx="230" cy="140" r="7" fill="#ef4444"/>
  <circle cx="140" cy="195" r="7" fill="#ef4444"/>
  <text x="112" y="78" font-size="12" fill="#10b981">+</text>
  <text x="238" y="132" font-size="12" fill="#ef4444">−</text>
  <text x="148" y="187" font-size="12" fill="#ef4444">−</text>
  <text x="60" y="245" font-size="12" fill="currentColor">One of 8 labelings shown. Rotating and shifting the line</text>
  <text x="60" y="262" font-size="12" fill="currentColor">reaches all 8, so VC(half-spaces in R²) ≥ 3.</text>
  <text x="400" y="40" font-size="13" fill="currentColor">4 points are NOT shattered: the XOR labeling fails</text>
  <circle cx="450" cy="90" r="7" fill="#10b981"/>
  <circle cx="600" cy="90" r="7" fill="#ef4444"/>
  <circle cx="450" cy="200" r="7" fill="#ef4444"/>
  <circle cx="600" cy="200" r="7" fill="#10b981"/>
  <text x="440" y="78" font-size="12" fill="#10b981">+</text>
  <text x="608" y="82" font-size="12" fill="#ef4444">−</text>
  <text x="436" y="196" font-size="12" fill="#ef4444">−</text>
  <text x="608" y="196" font-size="12" fill="#10b981">+</text>
  <line x1="450" y1="90" x2="600" y2="200" stroke="currentColor" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="600" y1="90" x2="450" y2="200" stroke="currentColor" stroke-width="1" stroke-dasharray="3,3"/>
  <circle cx="525" cy="145" r="4" fill="#f59e0b"/>
  <text x="400" y="245" font-size="12" fill="currentColor">The two diagonals cross (amber dot): the convex hulls of the</text>
  <text x="400" y="262" font-size="12" fill="currentColor">+ pair and the − pair intersect, so no line separates them.</text>
  <text x="60" y="300" font-size="12.5" fill="currentColor">Figure 4.5. Left: half-spaces in the plane shatter any 3 points in general position, so VC ≥ 3. Right: for 4 points</text>
  <text x="60" y="316" font-size="12.5" fill="currentColor">the alternating (XOR) labeling is unrealizable, because a separating line would have to keep two intersecting</text>
  <text x="60" y="332" font-size="12.5" fill="currentColor">segments apart. Hence VC(half-spaces in R²) = 3 = d + 1. Radon's theorem generalizes the right-hand picture.</text>
</svg>

**Proposition 4.44** (VC dimensions of four basic classes).

1. *Thresholds* on $\mathbb{R}$: $\mathcal H=\{x\mapsto\mathbf{1}_{\{x\ge t\}}:t\in\mathbb{R}\}$ has $\mathrm{VC}=1$.
2. *Intervals* on $\mathbb{R}$: $\mathcal H=\{\mathbf{1}_{[a,b]}:a\le b\}$ has $\mathrm{VC}=2$.
3. *Half-spaces* in $\mathbb{R}^d$: $\mathcal H=\{x\mapsto\mathbf{1}_{\{\langle w,x\rangle+b\ge0\}}:w\in\mathbb{R}^d,b\in\mathbb{R}\}$ has $\mathrm{VC}=d+1$.
4. *Axis-aligned rectangles* in $\mathbb{R}^2$ has $\mathrm{VC}=4$.

*Proof.* (1) A single point $x_1$ is shattered ($t\le x_1$ gives label $1$; $t>x_1$ gives $0$). Two points $x_1<x_2$ cannot be: the labeling $(1,0)$ would need $x_1\ge t$ and $x_2<t$, contradicting $x_1<x_2$.

(2) Two points are shattered by choosing the interval to contain neither, one, the other, or both. For three points $x_1<x_2<x_3$, the labeling $(1,0,1)$ requires $a\le x_1$ and $x_3\le b$, hence $x_2\in[a,b]$, forcing label $1$ at $x_2$.

(3) *Lower bound:* take $x_0=0$ and $x_i=e_i$, $i=1,\dots,d$, a set of $d+1$ points. Given any labeling $y\in\{0,1\}^{d+1}$, set $w_i=1$ if $y_i=1$ and $w_i=-1$ otherwise for $i\ge1$, and $b=+\tfrac12$ if $y_0=1$, $b=-\tfrac12$ otherwise. Then $\langle w,x_0\rangle+b=b$ has the sign of $y_0$, and $\langle w,x_i\rangle+b=w_i\pm\tfrac12$ has the sign of $w_i$, hence of $y_i$. So all $2^{d+1}$ labelings are realized.
*Upper bound:* no set of $d+2$ points can be shattered. By **Radon's theorem** (any $d+2$ points in $\mathbb{R}^d$ can be split into two subsets whose convex hulls intersect — see §4.12), split the points into $A\uplus B$ with a common point $p\in\mathrm{conv}(A)\cap\mathrm{conv}(B)$. If some half-space labeled all of $A$ as $1$ and all of $B$ as $0$, then, because a half-space is convex and its complement is convex, $p$ would have to lie in both the half-space and its complement — a contradiction. Figure 4.5 (right) is this argument for $d=2$, $d+2=4$.

(4) *Lower bound:* the four points $(0,1),(1,0),(0,-1),(-1,0)$ are shattered — for any target subset, take the smallest axis-aligned rectangle containing exactly those points and check it excludes the others (there are $16$ cases, each immediate because each point is extreme in one of the four directions). *Upper bound:* given any $5$ points, choose one that is topmost, one bottommost, one leftmost, one rightmost (at most $4$ points, possibly with repeats); at least one point $q$ is not among them. Any rectangle containing the four extremes contains their bounding box, which contains $q$. So the labeling "all four extremes $1$, $q$ $0$" is unrealizable. $\blacksquare$

> **Common Confusion 4.8**
>
> *"VC dimension is basically the number of parameters."*
>
> It is a decent heuristic and a false theorem, and both halves matter. Half-spaces in $\mathbb{R}^d$ have $d+1$ parameters and VC dimension $d+1$, which is where the intuition comes from. But consider
> $$\mathcal H_{\sin}=\big\{x\mapsto\mathbf{1}_{\{\sin(\theta x)>0\}}\ :\ \theta\in\mathbb{R}\big\},$$
> a class with exactly **one** real parameter and **infinite** VC dimension: for any $n$, the points $x_i=2^{-i}$, $i=1,\dots,n$, can be shattered by choosing $\theta$ to encode the desired labels in the binary expansion of $\theta/\pi$. One parameter, unlimited capacity. In the other direction, a network with $W$ parameters constrained to a low-dimensional manifold, or quantized to few values, can have VC dimension far below $W$.
>
> The deeper point is that VC dimension measures a *combinatorial* property of the function class, not the parametrization, and two parametrizations of the same class always give the same VC dimension. When a paper says "the model has $10^9$ parameters so its capacity is enormous," that is a claim about a parametrization, not about a class — and §4.11 is about the fact that such claims have poor predictive value for generalization.

### 4.9.2 Sauer–Shelah: past the VC dimension, growth is polynomial

The magic of VC dimension is that finite VC dimension does not merely mean "cannot shatter large sets"; it forces the growth function to be *polynomial* in $n$, with degree equal to the VC dimension. Nothing in Definition 4.43 makes that obvious — a priori $\Pi_{\mathcal H}(n)$ could be $2^n/n$, say.

**Lemma 4.45** (Sauer–Shelah). If $\mathrm{VC}(\mathcal H)=d<\infty$ then for all $n\ge1$,
$$\Pi_{\mathcal H}(n)\ \le\ \sum_{i=0}^{d}\binom{n}{i}\ \le\ (n+1)^d ,$$
and for $n\ge d\ge1$ also $\Pi_{\mathcal H}(n)\le\big(\tfrac{en}{d}\big)^d$.

*Proof.* Fix points $x_{1:n}$ and write $\mathcal{A}:=\mathcal H|_{x_{1:n}}\subseteq\{0,1\}^n$; it suffices to show that any $\mathcal A\subseteq\{0,1\}^n$ that shatters no set of size $d+1$ satisfies $\lvert\mathcal A\rvert\le\sum_{i\le d}\binom ni$. (Here "$\mathcal A$ shatters $T\subseteq\{1,\dots,n\}$" means the projection of $\mathcal A$ onto the coordinates in $T$ is all of $\{0,1\}^{\lvert T\rvert}$.) Induct on $n$.

*Base cases.* If $n\le d$ then $\sum_{i\le d}\binom ni=2^n\ge\lvert\mathcal A\rvert$. If $d=0$, no single coordinate is shattered, so all elements of $\mathcal A$ agree in every coordinate and $\lvert\mathcal A\rvert\le1=\binom n0$.

*Inductive step.* Let $n>d\ge1$. Split off the last coordinate: define
$$\mathcal A':=\big\{v\in\{0,1\}^{n-1}\ :\ (v,0)\in\mathcal A\ \text{or}\ (v,1)\in\mathcal A\big\},\qquad
\mathcal A'':=\big\{v\in\{0,1\}^{n-1}\ :\ (v,0)\in\mathcal A\ \text{and}\ (v,1)\in\mathcal A\big\}.$$
Every element of $\mathcal A$ is counted once in $\mathcal A'$, and elements whose prefix appears with *both* last coordinates are counted once more in $\mathcal A''$; hence exactly
$$\lvert\mathcal A\rvert=\lvert\mathcal A'\rvert+\lvert\mathcal A''\rvert .$$
Now bound each. $\mathcal A'$ shatters no set of size $d+1$ within $\{1,\dots,n-1\}$ (a shattered set for $\mathcal A'$ lifts to one for $\mathcal A$), so by induction $\lvert\mathcal A'\rvert\le\sum_{i\le d}\binom{n-1}{i}$. And $\mathcal A''$ shatters no set of size $d$: if $\mathcal A''$ shattered $T$ with $\lvert T\rvert=d$, then $\mathcal A$ would shatter $T\cup\{n\}$, of size $d+1$ — because for each pattern on $T$ we have a $v\in\mathcal A''$ realizing it, and *both* $(v,0)$ and $(v,1)$ belong to $\mathcal A$, supplying both values on coordinate $n$. So by induction $\lvert\mathcal A''\rvert\le\sum_{i\le d-1}\binom{n-1}{i}$. Adding, and using Pascal's rule $\binom{n-1}{i}+\binom{n-1}{i-1}=\binom ni$:
$$\lvert\mathcal A\rvert\ \le\ \sum_{i=0}^{d}\binom{n-1}{i}+\sum_{i=0}^{d-1}\binom{n-1}{i}=\sum_{i=0}^d\Big[\binom{n-1}{i}+\binom{n-1}{i-1}\Big]=\sum_{i=0}^d\binom ni .$$
For the crude corollary $\sum_{i\le d}\binom ni\le(n+1)^d$ ($n\ge1$, $d\ge1$): since $\binom ni\le n^i$, it suffices to show $\sum_{i=0}^dn^i\le(n+1)^d$, which we prove by induction on $d$. At $d=1$ both sides equal $n+1$. Assuming it for $d$,
$$(n+1)^{d+1}=(n+1)^d(n+1)\ \ge\ \Big(\sum_{i=0}^dn^i\Big)(n+1)=\sum_{i=0}^dn^i+\sum_{i=0}^dn^{i+1}\ \ge\ \sum_{i=0}^{d+1}n^i .$$
For the sharper bound, assume $n\ge d\ge1$ so that $d/n\le1$; then $(d/n)^{i-d}\ge1$ for every $i\le d$, hence
$$\sum_{i\le d}\binom ni\ \le\ \Big(\frac nd\Big)^d\sum_{i\le d}\binom ni\Big(\frac dn\Big)^i\ \le\ \Big(\frac nd\Big)^d\Big(1+\frac dn\Big)^n\ \le\ \Big(\frac nd\Big)^de^d=\Big(\frac{en}{d}\Big)^d,$$
using the binomial theorem and $(1+x)^{1/x}\le e$. $\blacksquare$

**Worked Example 4.19** (How dramatic the collapse is). For $d=3$:

| $n$ | $\sum_{i\le3}\binom ni$ | $(n+1)^3$ | $2^n$ |
|---|---|---|---|
| $10$ | $176$ | $1{,}331$ | $1{,}024$ |
| $20$ | $1{,}351$ | $9{,}261$ | $1{,}048{,}576$ |
| $100$ | $166{,}751$ | $1{,}030{,}301$ | $\approx1.3\times10^{30}$ |
| $1000$ | $166{,}667{,}501$ | $1{,}003{,}003{,}001$ | $\approx1.1\times10^{301}$ |

At $n=1000$ a VC-dimension-$3$ class realizes about $1.7\times10^8$ labelings out of $10^{301}$ possible: a fraction of $10^{-293}$. And the quantity that enters our bounds is $\log\Pi_{\mathcal H}(n)$, which at $n=60{,}000$, $d=5$ equals $50.22$ (computed from the exact binomial sum $6{,}479{,}460{,}045{,}000{,}150{,}047{,}001$), against the crude estimate $5\log(60{,}001)=55.01$: the crude bound $(n+1)^d$ costs less than $10\%$.

### 4.9.3 The VC generalization bound

**Theorem 4.46** (VC bound via Rademacher). Let $\mathcal H$ be a class of functions $\mathcal X\to\{0,1\}$ with $\mathrm{VC}(\mathcal H)=d<\infty$, and let $\ell$ be the $0$–$1$ loss. Then
$$\mathfrak{R}_n(\ell\circ\mathcal H)\ \le\ \sqrt{\frac{2\log\Pi_{\mathcal H}(n)}{n}}\ \le\ \sqrt{\frac{2d\log(n+1)}{n}},$$
and consequently, with probability at least $1-\delta$, simultaneously for all $h\in\mathcal H$,
$$R(h)\ \le\ \hat R_n(h)+2\sqrt{\frac{2d\log(n+1)}{n}}+\sqrt{\frac{\log(1/\delta)}{2n}} .$$

*Proof.* Fix the sample $Z_{1:n}=(x_i,y_i)_{i=1}^n$. The loss vectors $a_h=\big(\mathbf{1}_{\{h(x_i)\ne y_i\}}\big)_{i=1}^n\in\{0,1\}^n$ take at most $\lvert\mathcal H|_{x_{1:n}}\rvert\le\Pi_{\mathcal H}(n)$ distinct values (distinct loss vectors require distinct label vectors), and $\lVert a_h\rVert_2\le\sqrt n$. Massart's lemma (Lemma 4.39) with $N=\Pi_{\mathcal H}(n)$ and $\mathcal R=\sqrt n$ gives
$$\hat{\mathfrak{R}}_n(\ell\circ\mathcal H)=\frac1n\mathbb{E}_\sigma\max_h\langle\sigma,a_h\rangle\le\frac{\sqrt n\sqrt{2\log\Pi_{\mathcal H}(n)}}{n}=\sqrt{\frac{2\log\Pi_{\mathcal H}(n)}{n}} .$$
This holds for every sample, so it holds in expectation, giving the bound on $\mathfrak{R}_n$; Lemma 4.45 supplies $\log\Pi_{\mathcal H}(n)\le d\log(n+1)$. Theorem 4.38 then gives the generalization statement. $\blacksquare$

The rate is $\sqrt{d\log n/n}$. A more careful argument (chaining, §4.10) removes the $\log n$ and gives the optimal $\sqrt{d/n}$; the classical Vapnik–Chervonenkis form, obtained by a direct union bound over the growth function rather than through Rademacher complexity, is
$$\varepsilon=\sqrt{\frac{8\big(d\log(2en/d)+\log(4/\delta)\big)}{n}},$$
valid for $n\ge d$. Note the constraint: **when $d>2en$ the logarithm's argument drops below $1$ and the expression stops making sense** — a signal, not a technicality, that uniform convergence has nothing to say once the VC dimension exceeds the sample size.

**A first genuine payoff: the empirical CDF converges uniformly.** Before we watch this machinery fail on neural networks, it is worth seeing it *succeed* on the oldest problem in nonparametric statistics, because this is the example that justifies the whole apparatus. Let $X_1,\dots,X_n$ be i.i.d. real-valued with CDF $F(t)=\mathbb{P}(X\le t)$, and let $\hat F_n(t)=\frac1n\sum_i\mathbf 1_{\{X_i\le t\}}$ be the empirical CDF. The class of events $\{X\le t\}$ is exactly the class of thresholds, which by Proposition 4.44(1) has $\mathrm{VC}=1$. Theorem 4.46 therefore gives, with probability at least $1-\delta$,
$$\sup_{t\in\mathbb{R}}\big\lvert\hat F_n(t)-F(t)\big\rvert\ \le\ 2\sqrt{\frac{2\log(n+1)}{n}}+\sqrt{\frac{\log(1/\delta)}{2n}},$$
which at $n=10{,}000$, $\delta=0.05$ evaluates to $0.0981$: **the entire CDF, at every one of uncountably many points simultaneously, is pinned to within ten percentage points.** No fixed-$t$ argument can give this — Hoeffding at a single $t$ gives $0.0136$, but a separate statement at each of uncountably many $t$ is not a statement about the supremum, which is precisely Common Confusion 4.6's point. Letting $n\to\infty$, the bound tends to $0$, which is the **Glivenko–Cantelli theorem**: $\sup_t\lvert\hat F_n-F\rvert\to0$ almost surely. The sharp constant is known — the Dvoretzky–Kiefer–Wolfowitz inequality gives $\sqrt{\log(2/\delta)/2n}=0.0136$ at these settings, so our VC route is about $7$ times loose — but the point is that a distribution-free, uniform-over-an-infinite-class guarantee exists at all, and that it is *non-vacuous at realistic sample sizes*. Hold this example next to Worked Example 4.20 below: the machinery is the same, and the difference in outcome is entirely a difference in VC dimension, $1$ versus $10^6$.

**Proposition 4.47** (VC dimension of neural networks; cited facts, not proved here). Let $\mathcal H$ be the class of binary classifiers computed by a feedforward network with $W$ weights and $L$ layers.

1. With threshold (sign) activations, $\mathrm{VC}(\mathcal H)=\Theta(W\log W)$.
2. With piecewise-linear (e.g. ReLU) activations, $\mathrm{VC}(\mathcal H)=O(WL\log W)$ and $\Omega(WL\log(W/L))$.
3. With sigmoid activations, $\mathrm{VC}(\mathcal H)=O(W^2k^2)$, where $k$ is the number of computational units (Karpinski–Macintyre), and $\mathrm{VC}(\mathcal H)=\Omega(W^2)$ (Koiran–Sontag). The exact polynomial order between these two bounds is not known.

Sources are in §4.12. What matters for us is that all three are *at least linear* in the parameter count, so for any network with more parameters than you have data — which is every network in modern practice — Theorem 4.46 is vacuous. That is not a subtlety to be optimized away; it is the subject of §4.11.

**Worked Example 4.20** (The VC bound for a linear model on MNIST-sized data, and then for an MLP). Take $n=60{,}000$, $\delta=0.05$, and use the Rademacher-derived form $\varepsilon=2\sqrt{2d\log(n+1)/n}+\sqrt{\log(1/\delta)/2n}$:

| class | $d=\mathrm{VC}$ | $\varepsilon$ at $n=60{,}000$ |
|---|---|---|
| thresholds on $\mathbb{R}$ | $1$ | $0.0433$ |
| intervals | $2$ | $0.0592$ |
| axis-aligned rectangles in $\mathbb{R}^2$ | $4$ | $0.0819$ |
| half-spaces in $\mathbb{R}^{10}$ | $11$ | $0.1320$ |
| linear classifier on raw MNIST pixels | $785$ | $1.0781$ |

The first four rows are meaningful bounds. The fifth is already vacuous: a plain *linear* classifier on $784$ pixels, trained on the full MNIST training set, gets no guarantee from VC theory at all. To reach $\varepsilon=0.1$ with $d=785$ this bound needs $n\approx1.01\times10^7$, about $170$ MNIST training sets. Meanwhile a logistic regression on MNIST has a train–test gap of roughly one percentage point.

And for the MLP of Worked Example 4.16 ($W=50{,}890$), Proposition 4.47(2) with $L=2$ suggests $\mathrm{VC}\approx W\log_2 W\approx7.96\times10^5$ up to constants. Then $d\gg n$: the classical form is not even defined ($2en/d=0.41<1$), and the Rademacher-derived form gives $\varepsilon=34.17$. A bound of $34$ on a quantity in $[0,1]$.

> **Self-check 4.9**
>
> 1. What is $\mathrm{VC}(\mathcal H)$ if $\mathcal H$ is the class of all functions $\mathcal X\to\{0,1\}$ on a finite $\mathcal X$ with $\lvert\mathcal X\rvert=k$?
> 2. Definition 4.43 defines $\Pi_{\mathcal H}(n)$ as a *maximum* over point configurations. Why a maximum rather than a typical value, and where does that choice cost us?
> 3. If $\mathcal H_1\subseteq\mathcal H_2$, how do their VC dimensions compare? Prove it in one line.
> 4. A class has $\mathrm{VC}=5$. What is the largest $n$ for which its growth function might still equal $2^n$?
## 4.10 Covering numbers and chaining

### 4.10.1 Approximating a class at one scale

VC dimension is a blunt instrument: it counts labelings, so it sees only which sign patterns are achievable and is blind to *how close* two hypotheses are. For real-valued classes — regression, margins, neural network outputs — we want a notion of size that respects a metric. That notion is the covering number.

**Definition 4.48** (Covering and packing numbers; metric entropy). Let $(T,\rho)$ be a metric space and $\varepsilon>0$. A set $T_\varepsilon\subseteq T$ is an *$\varepsilon$-net* (or $\varepsilon$-cover) if every $t\in T$ has some $s\in T_\varepsilon$ with $\rho(t,s)\le\varepsilon$. The *covering number* $N(T,\rho,\varepsilon)$ is the smallest cardinality of an $\varepsilon$-net. A set $S\subseteq T$ is *$\varepsilon$-separated* if $\rho(s,s')>\varepsilon$ for all distinct $s,s'\in S$; the *packing number* $M(T,\rho,\varepsilon)$ is the largest cardinality of an $\varepsilon$-separated set. The quantity $\log N(T,\rho,\varepsilon)$ is the *metric entropy* of $T$ at scale $\varepsilon$.

For function classes we use the *empirical* $L^2$ metric determined by the sample:
$$d_n(f,g):=\Big(\frac1n\sum_{i=1}^n\big(f(z_i)-g(z_i)\big)^2\Big)^{1/2}=\lVert f-g\rVert_{L^2(\hat P_n)} .$$
Two standard facts, both one-line arguments: $M(T,\rho,2\varepsilon)\le N(T,\rho,\varepsilon)\le M(T,\rho,\varepsilon)$ (so covering and packing numbers are interchangeable up to a factor of $2$ in the scale), and a maximal $\varepsilon$-separated set is automatically an $\varepsilon$-net.

**Proposition 4.49** (Covering numbers of the Euclidean ball). Let $B_2^d=\{x\in\mathbb{R}^d:\lVert x\rVert_2\le1\}$ with the Euclidean metric. For $0<\varepsilon\le1$,
$$\Big(\frac1\varepsilon\Big)^d\ \le\ N\big(B_2^d,\lVert\cdot\rVert_2,\varepsilon\big)\ \le\ \Big(\frac3\varepsilon\Big)^d .$$
Hence $\log N=\Theta\big(d\log(1/\varepsilon)\big)$.

*Proof.* *Lower bound (volume).* If balls of radius $\varepsilon$ centered at $N$ points cover $B_2^d$, comparing Lebesgue measures (Ch. 1, Definition 1.6 conventions; $\lambda_d$ scales as $\lambda_d(rA)=r^d\lambda_d(A)$) gives $\lambda_d(B_2^d)\le N\varepsilon^d\lambda_d(B_2^d)$, so $N\ge\varepsilon^{-d}$.

*Upper bound (packing).* Let $S$ be a maximal $\varepsilon$-separated subset of $B_2^d$; by maximality it is an $\varepsilon$-net, so $N\le\lvert S\rvert$. The balls of radius $\varepsilon/2$ around points of $S$ are pairwise disjoint and all contained in $(1+\varepsilon/2)B_2^d$. Comparing volumes,
$$\lvert S\rvert\Big(\frac\varepsilon2\Big)^d\lambda_d(B_2^d)\ \le\ \Big(1+\frac\varepsilon2\Big)^d\lambda_d(B_2^d)\quad\Longrightarrow\quad \lvert S\rvert\le\Big(\frac{2}{\varepsilon}+1\Big)^d\le\Big(\frac3\varepsilon\Big)^d,$$
the last step using $\varepsilon\le1$. $\blacksquare$

**Read the exponent carefully, because this is where most people give up on covering arguments too early.** The exponential dependence on $d$ is the curse of dimensionality in its purest form: at $d=100$, $\varepsilon=0.1$ the number of balls needed is between $10^{100}$ and $5.2\times10^{147}$, a count with no physical meaning. A reasonable first reaction is that any bound involving such a quantity must be worthless. That reaction is wrong, and the reason is worth stating flatly: **$N(\varepsilon)$ never enters a generalization bound; only $\log N(\varepsilon)$ does.** Covering numbers arrive through Massart's lemma (Lemma 4.39), which charges $\sqrt{2\log N/n}$ — a union bound over the net, and union bounds are logarithmic, exactly as in §4.7. At $d=100$, $\varepsilon=0.1$ we have $\log N\le d\log(3/\varepsilon)=340$, and at $n=10^6$ the resulting term is $\sqrt{2\cdot340/10^6}=0.026$. A count of $10^{147}$ has become a contribution of under three percentage points.

So metric entropy, not the covering number, is the currency, and it is only *linear* in $d$ and *logarithmic* in $1/\varepsilon$. Whenever you see an astronomical covering number in a paper, take its logarithm before forming an opinion.

**Proposition 4.50** (One-scale discretization bound). Let $\mathcal F$ be a class of functions with $B:=\sup_{f\in\mathcal F}\lVert f\rVert_{L^2(\hat P_n)}<\infty$. Then for every $\varepsilon>0$,
$$\hat{\mathfrak{R}}_n(\mathcal F)\ \le\ \varepsilon+B\sqrt{\frac{2\log N(\mathcal F,d_n,\varepsilon)}{n}},$$
and hence $\hat{\mathfrak{R}}_n(\mathcal F)\le\inf_{\varepsilon>0}\Big[\varepsilon+B\sqrt{2\log N(\mathcal F,d_n,\varepsilon)/n}\Big]$.

*Proof.* Let $\mathcal F_\varepsilon$ be a minimal $\varepsilon$-net of $\mathcal F$ in $d_n$. Given $f\in\mathcal F$ choose $\pi(f)\in\mathcal F_\varepsilon$ with $d_n(f,\pi(f))\le\varepsilon$. For any sign vector $\sigma$,
$$\frac1n\sum_i\sigma_if(z_i)=\frac1n\sum_i\sigma_i\pi(f)(z_i)+\frac1n\sum_i\sigma_i\big(f-\pi(f)\big)(z_i),$$
and by Cauchy–Schwarz the second term is at most
$$\frac1n\lVert\sigma\rVert_2\Big(\sum_i\big(f-\pi(f)\big)(z_i)^2\Big)^{1/2}=\frac1n\cdot\sqrt n\cdot\sqrt n\,d_n\big(f,\pi(f)\big)\le\varepsilon .$$
Taking suprema and then $\mathbb{E}_\sigma$ gives $\hat{\mathfrak{R}}_n(\mathcal F)\le\varepsilon+\hat{\mathfrak{R}}_n(\mathcal F_\varepsilon)$. Massart's lemma (Lemma 4.39) applied to the finite set of vectors $a_g=(g(z_i))_i$, $g\in\mathcal F_\varepsilon$, with $\lVert a_g\rVert_2=\sqrt n\lVert g\rVert_{L^2(\hat P_n)}\le\sqrt nB$, gives $\hat{\mathfrak{R}}_n(\mathcal F_\varepsilon)\le B\sqrt{2\log\lvert\mathcal F_\varepsilon\rvert/n}$. $\blacksquare$

This is the honest version of the discretization idea from Common Confusion 4.6, and it already recovers useful bounds: for a parametric class with $\log N(\varepsilon)\approx d\log(3/\varepsilon)$, optimizing over $\varepsilon$ gives $\hat{\mathfrak{R}}_n=O(\sqrt{d\log n/n})$. But there is a $\log n$ we should not be paying, and for nonparametric classes the loss is worse than a log factor.

### 4.10.2 Chaining: use all the scales at once

The inefficiency in Proposition 4.50 is that it treats every function in $\mathcal F$ as if it were $\varepsilon$ away from its representative *and* charges the full union-bound price $\sqrt{\log N(\varepsilon)}$ at that single scale. Chaining does something smarter: it approximates $f$ by a sequence of ever-finer representatives $f_0,f_1,f_2,\dots$ at scales $\varepsilon_k=2^{-k}$, writes
$$f=f_0+\sum_{k\ge1}\big(f_k-f_{k-1}\big),$$
and applies a union bound *separately at each scale*. The gain: the increment $f_k-f_{k-1}$ is small (norm $\approx3\cdot2^{-k}$), so the sub-Gaussian tail it needs is correspondingly tight, and the price $\sqrt{\log N(2^{-k})}$ is multiplied by that small norm. Summing a geometric-ish series of small terms beats paying the largest term once.

<svg viewBox="0 0 720 440" style="max-width:100%;height:auto" role="img" aria-label="A function class covered at three successively finer scales, with a chain of approximations to a target function">
  <ellipse cx="345" cy="175" rx="290" ry="150" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="610" y="170" font-size="13" fill="currentColor">class F</text>
  <circle cx="185" cy="160" r="75" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,4"/>
  <circle cx="360" cy="200" r="75" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,4"/>
  <circle cx="490" cy="175" r="75" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,4"/>
  <circle cx="185" cy="160" r="4" fill="#3b82f6"/>
  <circle cx="360" cy="200" r="4" fill="#3b82f6"/>
  <circle cx="490" cy="175" r="4" fill="#3b82f6"/>
  <circle cx="330" cy="170" r="36" fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="395" cy="225" r="36" fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="330" cy="170" r="3.5" fill="#10b981"/>
  <circle cx="395" cy="225" r="3.5" fill="#10b981"/>
  <circle cx="345" cy="185" r="16" fill="none" stroke="#f59e0b" stroke-width="1.5"/>
  <circle cx="345" cy="185" r="3" fill="#f59e0b"/>
  <circle cx="352" cy="192" r="4.5" fill="#ef4444"/>
  <text x="356" y="196" font-size="12.5" fill="#ef4444">f</text>
  <line x1="360" y1="200" x2="332" y2="172" stroke="currentColor" stroke-width="1.5"/>
  <line x1="330" y1="170" x2="345" y2="185" stroke="currentColor" stroke-width="1.5"/>
  <line x1="345" y1="185" x2="352" y2="192" stroke="currentColor" stroke-width="1.5"/>
  <text x="120" y="60" font-size="12" fill="#3b82f6">scale ε₀ = 1: N(1) centers</text>
  <text x="120" y="78" font-size="12" fill="#10b981">scale ε₁ = 1/2: N(1/2) centers</text>
  <text x="120" y="96" font-size="12" fill="#f59e0b">scale ε₂ = 1/4: N(1/4) centers</text>
  <text x="60" y="368" font-size="12.5" fill="currentColor">Figure 4.6. Chaining. Instead of approximating the target f (red) once at a single scale and paying the full union-</text>
  <text x="60" y="384" font-size="12.5" fill="currentColor">bound price √(log N(ε)) there, we follow a chain f₀ → f₁ → f₂ → f through nets of scale 1, 1/2, 1/4, …. Each link</text>
  <text x="60" y="400" font-size="12.5" fill="currentColor">is short, so the sub-Gaussian bound it needs is tight, and the total cost is a sum ∑ₖ 2⁻ᵏ√(log N(2⁻ᵏ)) — an integral</text>
  <text x="60" y="416" font-size="12.5" fill="currentColor">over scales rather than a single worst-case term. This is Dudley's entropy integral.</text>
</svg>

**Theorem 4.51** (Dudley's entropy integral; stated, with proof sketch). Let $\mathcal F$ be a class of functions with $\sup_{f\in\mathcal F}\lVert f\rVert_{L^2(\hat P_n)}\le B$ and $0\in\mathcal F$. Then
$$\hat{\mathfrak{R}}_n(\mathcal F)\ \le\ \frac{12}{\sqrt n}\int_0^{B}\sqrt{\log N(\mathcal F,d_n,\varepsilon)}\ d\varepsilon .$$

*Proof sketch.* Put $\varepsilon_k=B2^{-k}$ and let $\mathcal F_k$ be a minimal $\varepsilon_k$-net, with $\pi_k(f)$ the nearest element. For fixed $K$, decompose
$$\frac1n\sum_i\sigma_if(z_i)=\underbrace{\frac1n\sum_i\sigma_i\pi_0(f)(z_i)}_{\text{coarse}}+\sum_{k=1}^{K}\frac1n\sum_i\sigma_i\big(\pi_k(f)-\pi_{k-1}(f)\big)(z_i)+\underbrace{\frac1n\sum_i\sigma_i\big(f-\pi_K(f)\big)(z_i)}_{\le\ \varepsilon_K}.$$
The coarse term is handled by Massart at scale $\varepsilon_0$. For the $k$-th increment, $d_n(\pi_k(f),\pi_{k-1}(f))\le\varepsilon_k+\varepsilon_{k-1}=3\varepsilon_k$, and there are at most $\lvert\mathcal F_k\rvert\lvert\mathcal F_{k-1}\rvert\le\lvert\mathcal F_k\rvert^2$ possible increment vectors; Massart's lemma applied to that finite set of vectors, each of empirical $L^2$ norm at most $3\varepsilon_k$, bounds the $k$-th term's contribution by $3\varepsilon_k\sqrt{2\cdot2\log\lvert\mathcal F_k\rvert/n}$. Summing over $k$, letting $K\to\infty$ so the residual vanishes, and comparing the resulting sum $\sum_k\varepsilon_k\sqrt{\log N(\varepsilon_k)}$ to the integral $\int_0^B\sqrt{\log N(\varepsilon)}d\varepsilon$ (the summands are, up to the factor $\varepsilon_k-\varepsilon_{k+1}=\varepsilon_k/2$, a Riemann sum for a decreasing integrand) gives the stated form with an absolute constant, which can be taken to be $12$. *(End of sketch.)* Two steps were waved at rather than proved — the uniform justification of the $K\to\infty$ limit and the Riemann-sum comparison that produces the constant $12$ — which is why this result appears in §4.12 rather than carrying a proof box; full details are in the references given there.

**Worked Example 4.21** (Chaining versus one scale: a rate, not a constant). Compare the two bounds on the same entropy profile. All numbers below come from numerically minimizing Proposition 4.50's expression over a grid of $2\times10^5$ values of $\varepsilon$ and from numerically evaluating Dudley's integral.

*Parametric profile* $\log N(\varepsilon)=d\log(3/\varepsilon)$ with $d=10$, $B=1$. The Dudley integral is $\int_0^1\sqrt{10\log(3/\varepsilon)}\,d\varepsilon=4.4769$.

| $n$ | one scale (Prop. 4.50) | best $\varepsilon$ | Dudley integrand only, $I/\sqrt n$ | $\sqrt{d/n}$ |
|---|---|---|---|---|
| $10^3$ | $0.3333$ | $0.0333$ | $0.1416$ | $0.1000$ |
| $10^5$ | $0.0402$ | $0.0027$ | $0.0142$ | $0.0100$ |

The one-scale bound carries an extra $\sqrt{\log n}$-type factor; chaining's $\varepsilon$-dependence matches the optimal $\sqrt{d/n}$ up to a constant (and up to Theorem 4.51's unoptimized factor $12$, which for these $n$ makes the *numerical* Dudley bound larger — constants in Theorem 4.51 are not sharp, and no one uses them for numerical work).

*Nonparametric profile* $\log N(\varepsilon)=1/\varepsilon$, the entropy of a class like the $1$-Lipschitz functions on $[0,1]$. Here $\int_0^1\varepsilon^{-1/2}d\varepsilon=2$, so Dudley gives $24/\sqrt n$.

| $n$ | one scale | Dudley $24/\sqrt n$ | ratio |
|---|---|---|---|
| $10^3$ | $0.23811$ | $0.75895$ | $0.31$ |
| $10^6$ | $0.02381$ | $0.02400$ | $0.99$ |
| $10^9$ | $0.00238$ | $0.00076$ | $3.14$ |

Fitting slopes on a log-log scale over $n\in[10^3,10^9]$ gives exactly $-1/3$ for the one-scale bound and $-1/2$ for Dudley. **This is the point: single-scale discretization gives the wrong rate ($n^{-1/3}$), chaining gives the right one ($n^{-1/2}$).** The crossover in absolute terms happens near $n=10^6$ only because Theorem 4.51's constant is $12$ rather than something near $1$; the exponent is what matters and no amount of constant-tuning fixes an exponent.

Two closing remarks. First, Dudley's integral can diverge — if $\log N(\varepsilon)$ grows like $\varepsilon^{-2}$ then $\int\sqrt{\log N}$ is logarithmically divergent at $0$ and chaining as stated fails; such classes (e.g. all bounded measurable functions) genuinely do not admit uniform convergence. Second, chaining is not the end of the story: Talagrand's generic chaining replaces the entropy integral with the $\gamma_2$ functional and is sharp for Gaussian processes, where Dudley can lose a $\sqrt{\log}$ factor. That theory underlies the sharpest margin bounds for neural networks. Chapter 5 uses the Gaussian-process view of this material when it discusses GPs.

> **Self-check 4.10**
>
> 1. Why is the metric entropy, rather than the covering number itself, the quantity that enters generalization bounds?
> 2. In Proposition 4.50, what happens to the bound as $\varepsilon\to0$? As $\varepsilon\to B$?
> 3. A class has $\log N(\varepsilon)=d\log(1/\varepsilon)$. Explain, without computing, why Dudley's integral converges.
> 4. Give the covering-number-based reason that VC classes have $\hat{\mathfrak{R}}_n=O(\sqrt{d/n})$ without the $\log n$ of Theorem 4.46.
## 4.11 Why these bounds are vacuous for deep nets, and what that means

### 4.11.1 The accounting, done honestly

Everything in this chapter has been true. Every theorem was proved or carefully cited, every hypothesis stated. And when we apply the resulting machinery to the models people actually train, it tells us nothing. It is worth seeing exactly how much nothing, in one table, because the size of the failure is itself information.

Take the network from Worked Example 4.16: a two-layer MLP, $784\to64\to10$, so $W=50{,}890$ parameters, trained on $n=60{,}000$ MNIST examples, achieving about $2\%$ test error with a train–test gap of roughly one to two percentage points. Four bounds from this chapter, each at $\delta=0.05$:

**Worked Example 4.22** (Four bounds, one network). The last column says where each number comes from, and you should read that column before the numbers — two of these rows are computed from the architecture alone, one depends on quantities I have *assumed*, and the last is not something this chapter measured at all.

| bound | machinery | guaranteed gap | provenance |
|---|---|---|---|
| finite class by parameter counting | Thm. 4.34 with $\log\lvert\mathcal H\rvert=32W\log2$ | $3.067$ | computed from $W$ |
| VC dimension | Thm. 4.46 with $d\approx W\log_2W\approx7.96\times10^5$ | $34.17$ | computed from $W$ |
| classical VC form | $\sqrt{8(d\log(2en/d)+\log(4/\delta))/n}$ | **undefined** ($2en/d=0.41<1$) | computed from $W$ |
| norm-based Rademacher | Prop. 4.42, $2B_1B_2X/\sqrt n$ | $32.66$ | **assumed** $B_1=10$, $B_2=20$, $X=10$ |
| actual train–test gap | — | $\approx0.02$ | typical *reported* value; not measured here |

Be clear about that last row. I did not train this network; $2\%$ test error with a one-to-two-point train–test gap is what a two-layer MLP of this size routinely reports on MNIST, and Lab B4(e) asks you to measure it yourself rather than take my word for it. Nothing in the argument below depends on the exact value — any figure below $0.1$ produces the same conclusion — but a chapter that insists on computing its numbers should not quietly present a recalled one as data.

With that said: the bounds are $3$, $34$ and $33$ on a quantity confined to $[0,1]$. The best of them exceeds the gap by a factor of roughly $150$ and is still vacuous, and "vacuous" has a precise meaning — the bound $R(h)\le\hat R_n(h)+34$ is *logically implied by* $R(h)\le1$, so it conveys nothing beyond the definition of a probability.

The norm-based row deserves separate comment, because it is the one people hope will be rescued. Its inputs $B_1=\max_j\lVert w_j\rVert_2$, $B_2=\lVert v\rVert_1$ and $X=\max_i\lVert x_i\rVert_2$ are round numbers I chose, so the *value* $32.66$ carries no authority. What carries authority is the scaling $2B_1B_2X/\sqrt n$ together with an inversion: to get a gap below $0.1$ at $n=60{,}000$ with $X=10$ you need $B_1B_2\le0.582$. Trained networks do not have product norms below one, and weight norms typically *grow* during training even as test error falls. So this bound does not merely fail numerically — it moves in the wrong direction along the training trajectory, which is a different and more serious kind of failure, developed in §4.11.4 and Exercise C3(d).

### 4.11.2 Random labels: the experiment that rules out a whole family of explanations

One might hope the problem is looseness — that the true capacity of the trained network is small and our estimates of it are crude. Zhang, Bengio, Hardt, Recht and Vinyals (2017) closed off that hope with an experiment that requires no new theory to interpret. They took standard image classification architectures and standard training pipelines, replaced the labels with uniformly random ones, and trained. The networks reached **zero training error** on completely random labels, on datasets of tens of thousands of images. They also fit random *pixels*.

Read that through Definition 4.43. A class that achieves zero empirical error on every labeling of $n$ points shatters those points, so its VC dimension is at least $n$; equivalently its empirical Rademacher complexity on that sample is essentially maximal, near $1$ for the $0$–$1$ loss. So the estimate $\mathrm{VC}\gtrsim n$ used above is not an artifact of loose counting — it is *approximately correct for the class these architectures realize*. The capacity really is enormous.

Now the logic bites. Any explanation of generalization that depends only on the hypothesis class $\mathcal H$ — VC dimension, growth function, class-level Rademacher complexity, covering numbers of $\mathcal H$ — assigns the *same* value to the true-label and random-label runs, because it is the same architecture, the same class, the same $n$. But the outcomes differ completely: near-zero test error in one case, chance-level in the other. A quantity that is constant across two runs with different outcomes cannot explain the difference. This is not a statement that VC theory is wrong; it is a statement that class capacity is not the relevant variable.

Whatever does the explaining must therefore depend on something the two runs do *not* share: the labels' relationship to the inputs, the optimization trajectory, the properties of the particular solution found, or an interaction among these. Note also that training took much longer to fit random labels, and the resulting solutions had different geometry — evidence that the algorithm, not the class, is doing the selecting.

### 4.11.3 The main candidate repairs, and their status

Four research programs try to close the gap. I state each with what it buys and what it does not.

**Margin bounds.** Replace the $0$–$1$ loss by a margin-sensitive surrogate and control the class by products of layer-wise norms rather than parameter counts. Bartlett, Foster and Telgarsky (2017) give a spectrally-normalized bound scaling with $\prod_\ell\lVert W_\ell\rVert_\sigma$ times a sum of ratios of $(2,1)$-norms to spectral norms, divided by the margin and $\sqrt n$. This is genuine progress: the bound is *not* the same for true and random labels, since the margins and norms differ. It remains numerically vacuous for realistic networks — the norm products are large — and its correlation with observed generalization across architectures is mixed.

**Compression bounds.** If a trained network can be compressed to $k$ effective parameters without much loss of accuracy, one can bound the generalization gap using $k$ rather than $W$ (Arora, Ge, Neyshabur, Zhang, 2018). The bounds become tighter, occasionally near-nonvacuous, and they explain why *this* solution generalizes even though the class is huge. The cost: the guarantee is about the compressed classifier, and the compression procedure must itself be accounted for.

**PAC-Bayes.** Bound the expected risk of a posterior distribution over weights by its empirical risk plus $\sqrt{D_{\mathrm{KL}}(Q\Vert P)/2n}$-type terms. Dziugaite and Roy (2017) optimized such a bound directly and obtained the first *nonvacuous* generalization guarantees for stochastic neural networks on binary MNIST — values in the tens of percent, which is loose but genuinely informative. PAC-Bayes is the most successful of the four numerically. It is also the closest in spirit to the variational bounds of §4.1: the $D_{\mathrm{KL}}$ term is the same object as in Proposition 4.7, and Chapter 9 develops this properly. Its limitation is that the guarantee is for a distribution over weights, not the deterministic network you deploy, and tight bounds require optimizing the posterior against the data, which needs care to remain valid.

**The impossibility direction.** Nagarajan and Kolter (2019) argue that the problem may be structural. They exhibit settings where a network generalizes well, yet *every* bound of the form "uniform convergence over any hypothesis class containing the learned predictor" is provably vacuous — because the set of predictors reachable by the algorithm, even restricted to those that generalize, still has near-maximal complexity on adversarially-resampled data. If this is right, then no amount of cleverness in choosing $\mathcal H$ inside the uniform-convergence framework suffices, and the framework itself must be supplemented.

### 4.11.4 What to conclude, and what not to

**The question is open.** I am not going to tell you which of these programs will win, because nobody knows. What can be said cleanly:

The tools in this chapter are not broken. Hoeffding, Bernstein, McDiarmid and Rademacher complexity are exactly right, and tight, for the problems they were built for: estimating a fixed quantity from a sample, bounding the error of a model selected from a genuinely small class, quantifying the noise in a validation estimate, deriving sample complexities for kernel methods and linear models, calibrating differential privacy. Every one of those uses is load-bearing in practice, and §4.7's finite-class bound is the correct tool for hyperparameter selection over a grid.

What has failed is one specific application: explaining, from class capacity alone, why an overparametrized network trained by SGD generalizes. The failure is instructive rather than embarrassing. It tells us that the effective hypothesis class is not the architecture's class but something the algorithm carves out of it, and that characterizing *that* object — implicit regularization, the geometry of the solutions SGD prefers, the role of the data distribution's structure — is where the explanation must come from.

One practical consequence worth stating bluntly. Because no valid a priori bound on a deep network's test error is currently within three orders of magnitude of useful, **the held-out estimate is not a convenience but the only quantitative guarantee available** — and that estimate is exactly where this chapter's tools do work, giving $\pm1.36$ percentage points at $m=10{,}000$ for a *single, pre-committed* model (§4.5). The catch is "pre-committed," and Common Confusion 4.3 plus the finite-class bound of §4.7 are the honest accounting for what repeated looks cost.

> **Self-check 4.11**
>
> 1. State precisely why the random-labels experiment rules out class-capacity explanations. Which quantity is constant across the two runs?
> 2. The norm-based bound "moves in the wrong direction" during training. What does that mean, and why is it worse than merely being loose?
> 3. Is it correct to conclude from this section that VC theory is wrong? Answer carefully.
> 4. Which of the four repair programs currently produces nonvacuous numbers, and what is the object it bounds?
## 4.12 Results used but not proved

This chapter proved almost everything it used. The table below is the complete list of exceptions: every place where a result was invoked, or stated as part of the development, without a proof given here. "Ch. $k$" means the proof appears in this tutorial's Chapter $k$.

| Result | Used in | Where to find a proof |
|---|---|---|
| Supporting hyperplane theorem in $\mathbb{R}^d$ (multivariate version of Lemma 4.2) | §4.1, multivariate Jensen | Rockafellar, *Convex Analysis*, Thm. 18.1; Boyd & Vandenberghe §2.5 |
| Conditional Jensen inequality | §4.1 remark; needed for stability arguments | Ch. 6 (conditional expectation) |
| Pinsker's inequality $D_{\mathrm{KL}}(P\Vert Q)\ge2\lVert P-Q\rVert_{\mathrm{TV}}^2$ | §4.1 (WE 4.3), §4.3 (WE 4.7) | Boucheron–Lugosi–Massart §4.8; Tsybakov §2.4 |
| Theorem 4.17, implications $(3)\Rightarrow(1)$ and the Orlicz-norm equivalence $(4)$ | §4.4 | Vershynin, *HDP*, Prop. 2.5.2 |
| Proposition 4.26, empirical Bernstein inequality | §4.5 | Maurer & Pontil (2009), *Empirical Bernstein Bounds and Sample Variance Penalization* |
| Theorem 4.28, Azuma–Hoeffding inequality | §4.6 | Ch. 6 (martingales); Boucheron–Lugosi–Massart §6.1 |
| Theorem 4.30, Efron–Stein inequality | §4.6 (WE 4.14) | Ch. 6; Boucheron–Lugosi–Massart Thm. 3.1 |
| Entropy and log-Sobolev methods (exponential Efron–Stein, Talagrand's convex distance inequality) | §4.6, closing remark | Boucheron–Lugosi–Massart Ch. 6 and Ch. 7 |
| Lemma 4.40, Talagrand's contraction lemma | §4.8, §4.11 (margin bounds) | Ledoux & Talagrand, *Probability in Banach Spaces*, Thm. 4.12; Mohri et al. Lemma 5.7 |
| Radon's theorem (any $d+2$ points in $\mathbb{R}^d$ split into two sets with intersecting convex hulls) | §4.9, Prop. 4.44(3) upper bound | Matoušek, *Lectures on Discrete Geometry*, Thm. 1.1.4 |
| Proposition 4.47, VC dimensions of neural networks | §4.9, §4.11 | Bartlett, Harvey, Liaw & Mehrabian (2019) for ReLU; Baum & Haussler (1989) and Maass for threshold nets; Karpinski & Macintyre (1997) for the sigmoid upper bound; Koiran & Sontag (1997) for the $\Omega(W^2)$ sigmoid lower bound |
| Theorem 4.51, Dudley's entropy integral (sketched only; the constant and the Riemann-sum step are not fully justified here) | §4.10 | van der Vaart & Wellner §2.2; Vershynin, *HDP*, Thm. 8.1.6; Wainwright §5.3 |
| Generic chaining and the $\gamma_2$ functional; sharpness for Gaussian processes | §4.10, closing remark | Talagrand, *Upper and Lower Bounds for Stochastic Processes* |
| Scaling of Lebesgue measure, $\lambda_d(rA)=r^d\lambda_d(A)$ | §4.10, Prop. 4.49 volume argument | Ch. 1 (construction of Lebesgue measure); Folland §2.6 |
| Central limit theorem (used only to calibrate intuition and to compute the "CLT column" of comparison tables, never to prove a bound) | §4.2 (WE 4.6), §4.5 (CC 4.5) | Billingsley, *Probability and Measure*, §27; Durrett §3.4 |
| Cramér's theorem: the large-deviation *lower* bound matching Chernoff, so that $\frac1n\log\mathbb{P}(\bar X_n\ge t)\to-\psi^*(t)$ | §4.5, "How much is actually left on the table" | Dembo & Zeitouni, *Large Deviations Techniques*, §2.2; Boucheron–Lugosi–Massart §4.10 |
| Dvoretzky–Kiefer–Wolfowitz inequality (the sharp constant for the empirical CDF) and the Glivenko–Cantelli theorem as an a.s. statement | §4.9.3, empirical-CDF payoff | Massart (1990) for the sharp constant; van der Vaart & Wellner §2.1 |
| Minimax lower bounds showing $\Omega(\sqrt{d/n})$ is unavoidable for VC classes, so the rates of §4.8–§4.10 cannot be improved in order | §4.5 and §4.11 (used to argue the tools are sharp) | Devroye–Györfi–Lugosi Ch. 14; Shalev-Shwartz & Ben-David §6.4 (no-free-lunch and VC lower bound) |
| Haussler's bound $\log N(\mathcal F,d_n,\varepsilon)\lesssim d\log(1/\varepsilon)$ for VC classes | Self-check 4.10(4) | Haussler (1995); van der Vaart & Wellner §2.6 |
| High-probability bounds from uniform stability (order $\beta\log n+\sqrt{\log(1/\delta)/n}$, beating the naive McDiarmid rate $\beta\sqrt n$) | §4.6.4, remark after Proposition 4.32 | Feldman & Vondrák, "High probability generalization bounds for uniformly stable algorithms with nearly optimal rate," *COLT* 2019 |
| Matrix Bernstein inequality (operator-norm concentration for sums of independent random matrices) | §4.5, closing forward pointer | J. A. Tropp, *An Introduction to Matrix Concentration Inequalities*, Found. Trends Mach. Learn. 8 (2015), Thm. 1.6.2 |
| Smoothed sensitivity and the exponential mechanism for high-sensitivity queries | §4.6.4, Exercise A18(d) | Nissim, Raskhodnikova & Smith (2007); Dwork & Roth §3.4 |
| Empirical findings on deep networks: fitting random labels; margin, compression and PAC-Bayes bounds; the uniform-convergence impossibility construction | §4.11 | Zhang et al. (2017); Bartlett–Foster–Telgarsky (2017); Arora et al. (2018); Dziugaite & Roy (2017); Nagarajan & Kolter (2019) — see references |

## 4.13 Chapter summary

1. **Every concentration inequality is Markov's inequality applied to a cleverly chosen nonnegative function.** Chebyshev uses $(X-\mu)^2$, Chernoff uses $e^{\lambda X}$, and higher-moment bounds use $\lvert X-\mu\rvert^k$. The entire subject is a search for the transform that makes the tail visible (Theorem 4.8 and Definition 4.12).
2. **Jensen's inequality is the workhorse of machine learning's derivations, not just an analysis lemma.** It gives $D_{\mathrm{KL}}\ge0$, the ELBO, the AM–GM inequality, the monotonicity of $L^p$ norms, and the direction of the bias in importance-weighted estimators (Theorem 4.3 and its corollaries).
3. **For bounded summands, Chebyshev's advantage over Hoeffding at moderate confidence is small and temporary; at high confidence Hoeffding wins overwhelmingly, because $1/\delta$ loses to $\log(1/\delta)$.** State the convention before comparing — this summary uses *one-sided* radii, the two-sided versions differ by whether $\delta$ or $\delta/2$ enters. At $n=1000$, $\delta=0.05$, one-sided, Chebyshev's radius $\sigma/\sqrt{n\delta}=0.04082$ against Hoeffding's $\sqrt{\log(1/\delta)/2n}=0.03870$: essentially a tie (and under the two-sided convention Chebyshev is actually slightly *ahead*, $0.04082$ against $0.04295$; Cantelli's one-sided refinement, Exercise A2, closes even that). The comparison is decided not at $\delta=0.05$ but as $\delta$ shrinks: at $\delta=10^{-6}$ Chebyshev's radius has grown by a factor of $224$ while Hoeffding's has grown by a factor of $2.1$. Likewise at fixed $\delta$, Chebyshev wins for small deviations — at $n=100$, $t=0.10$ it gives $0.0833$ against Hoeffding's $0.1353$ — and Hoeffding overtakes by $t=0.12$ and thereafter wins by growing margins. The regime that matters for learning theory, exponentially small $\delta$ forced by union bounds, belongs entirely to Hoeffding (Worked Examples 4.6 and 4.7, Figure 4.3, Lab B1).
4. **Sub-Gaussianity is the right abstraction for "tails no heavier than a Gaussian," and it has four equivalent characterizations** — MGF growth, tail decay, moment growth, and finiteness of an Orlicz norm — which is why the concept is stable enough to build a theory on (Theorem 4.17).
5. **Hoeffding's lemma says any variable in $[a,b]$ is sub-Gaussian with proxy $(b-a)^2/4$ regardless of its actual variance, and Bernstein and Bennett repair the resulting looseness by charging the true variance instead,** replacing the exponent $t^2/2\sigma^2$ with $t^2/(2(\nu^2+bt/3))$. The gap is exactly the pessimism of the proxy — a factor of $25$ for a Bernoulli$(0.01)$ — and in the rare-event regime the repair is not cosmetic: at $p=0.01$, $n=10^4$, Hoeffding's radius is $0.0122$ and Bernstein's $0.0026$, against an exact $0.0017$ (Lemma 4.19, Theorems 4.24 and 4.25, Worked Examples 4.8 and 4.11).
6. **Hoeffding's slack is asymptotically a constant factor, not a widening gap.** Measurement gives a ratio of $2.40$ between Hoeffding's radius and the true $95$th percentile at every $n$ from $10$ to $10^4$, and that constant decomposes exactly as $\sqrt3$ (variance proxy) times $1.386$ (the $\sqrt{\log}$ versus normal quantile) (Worked Example 4.12, Figure 4.4).
7. **McDiarmid's inequality extends concentration from sums to any function with bounded differences,** which is what lets us concentrate suprema, medians, and the output of a learning algorithm. It is proved by building a Doob martingale over the coordinates and applying Hoeffding's lemma conditionally (Theorem 4.29).
8. **Bounded differences is a worst-case condition and can be catastrophically pessimistic.** For the sample median at $n=101$ it yields a radius of $13.65$ on a quantity in $[0,1]$, while Efron–Stein — which averages the perturbations instead of maximizing them — gives a variance bound within a factor of $1.94$ of the truth (Worked Example 4.14).
9. **Uniform convergence over a finite class costs $\sqrt{\log\lvert\mathcal H\rvert/2n}$, and the logarithm is the union bound's fingerprint.** This makes the finite-class bound the correct and genuinely useful tool for hyperparameter grids and model selection over enumerable sets (Theorem 4.34).
10. **Rademacher complexity replaces "how many hypotheses" with "how well can this class fit pure noise,"** is data-dependent, is estimable by simulation, and is tied to the generalization gap in both directions through symmetrization (Definition 4.36, Theorems 4.37 and 4.38).
11. **For linear classes the Rademacher complexity is $O(BX/\sqrt n)$ and for two-layer networks $O(B_1B_2X/\sqrt n)$ — dimension-free and width-free.** Norms, not parameter counts, are what these bounds actually charge for (Propositions 4.41, 4.42). Simulation confirms the $\sqrt{d/n}$ scaling to within about $2\%$ (Worked Example 4.18).
12. **Finite VC dimension forces the growth function to be polynomial of degree exactly the VC dimension**, and where the VC dimension is genuinely small the resulting bounds are useful: the empirical CDF is pinned to within $0.098$ uniformly over all thresholds at $n=10^4$, which in the limit is the Glivenko–Cantelli theorem. The mechanism is Sauer–Shelah — the collapse from $2^n$ to $\sum_{i\le d}\binom ni$, which at $n=1000$, $d=3$ means $1.7\times10^8$ labelings instead of $10^{301}$ (Lemma 4.45).
13. **Covering numbers refine VC dimension for real-valued classes, and chaining is not an optimization of constants but a correction of the exponent.** Single-scale discretization gives $n^{-1/3}$ for a nonparametric entropy profile where chaining gives $n^{-1/2}$; fitted log-log slopes are exactly $-1/3$ and $-1/2$ (Proposition 4.50, Theorem 4.51, Worked Example 4.21).
14. **Applied to real networks, every bound in this chapter is vacuous — off by two to three orders of magnitude — and the random-labels experiment shows this is not mere looseness.** Class capacity genuinely is near-maximal, so no class-capacity-only argument can explain generalization; and since §4.5 established that the underlying inequalities are tight in rate, the failure cannot be blamed on weak tools either. The question of what does explain it is open (§4.11).
15. **These inequalities are tight in rate, and only loosely tight in constants** — Cramér's theorem says the Chernoff exponent is exactly right, and Hoeffding's residual factor of $2.40$ decomposes into $\sqrt3$ (variance proxy) times $1.386$ (finite-$n$ tail versus Gaussian quantile). Two of these tightness claims are *proved* in this chapter, not merely cited: Exercise A19 derives the Gaussian tail lower bound from scratch, showing the radius $\sigma\sqrt{2\log(1/\delta)/n}$ and the price $\sqrt{\log N}$ for $N$ independent quantities are both unavoidable. This matters for reading §4.11: the vacuity there cannot be blamed on weak inequalities.
16. **The tools are nevertheless correct and load-bearing where their hypotheses hold** — validation-set error bars, sample-complexity calculations for linear and kernel methods, differential privacy calibration, and honest accounting for repeated test-set use. The lesson of §4.11 is about the limits of one application, not about the mathematics.

### Callbacks

This chapter discharged several promissory notes. Chapter 1's union bound (Proposition 1.8(2)) reappeared as the source of the $\log\lvert\mathcal H\rvert$ in Theorem 4.34, and Chapter 1's warning that the union bound goes vacuous once the summed probabilities exceed one was made quantitative in Common Confusion 4.6 — at $n=1000$, $t=0.05$ it tolerates only $74$ hypotheses. Chapter 1 §1.8.3's caution that a supremum over an uncountable family need not be measurable was honored rather than ignored: Worked Example 4.13 and Common Confusion 4.6 both flag where the hypothesis of separability or countability is doing real work in $\sup_{h\in\mathcal H}\lvert\hat R_n(h)-R(h)\rvert$. Chapter 2's tail formula $\mathbb{E}[X]=\int_0^\infty P(X>t)\,dt$ did the work in the moment-growth implication of Theorem 4.17; Fubini–Tonelli and the product measure $P^{\otimes n}$ as the law of a dataset underwrote symmetrization in Theorem 4.37; and Chapter 2's claim that the ELBO is a genuine lower bound was finally proved, in Proposition 4.7. Chapter 3's moment generating functions and exponential-family structure were the engine of §4.3 through §4.5, and the exponential tilting in Lemma 4.19's proof is the same construction that made the natural parameters of an exponential family tractable there. Two debts are deferred: the martingale results (Azuma–Hoeffding, Efron–Stein) go to Chapter 6, and the PAC-Bayes and variational bounds glimpsed in §4.11 to Chapter 9.

### Before you continue

Answer these from memory; if one resists, the section named in parentheses is where to look.

1. Explain in one sentence why Chernoff bounds are exponentially better than Chebyshev, in terms of what function is fed to Markov's inequality (§4.2, §4.3).
2. A random variable is supported on $[0,1]$ with variance $0.001$. What sub-Gaussian proxy does Hoeffding's lemma give, and by what factor does it overstate the variance? (§4.4)
3. State the two regimes of Bernstein's inequality and say which one governs when $t$ is small (§4.5).
4. You want the tail bound for $\max_i X_i$ rather than $\bar X_n$. Which theorem of this chapter applies, and what are the bounded-difference constants? (§4.6)
5. Why does the finite-class bound scale as $\sqrt{\log\lvert\mathcal H\rvert}$ rather than $\sqrt{\lvert\mathcal H\rvert}$? Name the step responsible (§4.7).
6. Define empirical Rademacher complexity, and explain what value it takes when the class shatters the sample (§4.8, §4.9).
7. Sauer–Shelah: state the bound and explain what quantity in the induction plays the role of "coordinate $n$ appears with both labels" (§4.9).
8. Give one sentence on why chaining improves the *rate* rather than the constant (§4.10).
9. Name one setting in this chapter where an infinite-class bound comes out *non-vacuous* at a realistic sample size, and one where it does not. For the second, say why "the inequality is just loose" is not an adequate diagnosis (§4.9, §4.5, §4.11).

### Looking ahead to Chapter 5

Chapter 5 turns from tails to shape. So far randomness has been essentially one-dimensional: single variables, sums of scalars, suprema of scalar processes. Chapter 5 develops the multivariate Gaussian in full — covariance as a genuine linear-algebraic object (a positive semidefinite operator, with its spectrum carrying meaning), the density and its degenerate cases, and the calculus of marginals and conditionals via Schur complements, which is the algebraic identity that makes Gaussian conditioning tractable at all. From that machinery two payoffs follow directly. First, the He and Xavier initialization schemes are *derived*, not quoted: propagating variances forward and backward through a layer and demanding they be preserved gives the familiar $2/n_{\text{in}}$ and $2/(n_{\text{in}}+n_{\text{out}})$ scalings, with the factor of two traced to the ReLU. Second, Gaussian processes appear as the infinite-dimensional limit of the same conditioning formulas, giving a Bayesian nonparametric model whose posterior is available in closed form — and, as §4.10 hinted, the object whose suprema chaining was built to control.
# Exercises

Difficulty: ★ routine application of a stated result; ★★ requires assembling two or more ideas; ★★★ a genuine derivation, of the sort a paper's appendix would contain. Full solutions, with the reasoning made explicit, follow Appendix A.

## Part A — Theory

**A1** ★ *Markov is tight.* Let $t>0$ and $a>0$. Construct a nonnegative random variable $X$ with $\mathbb{E}[X]=a$ for which $P(X\ge t)=a/t$ exactly, assuming $t\ge a$. Then explain in one sentence what property of your construction makes it extremal, and why no bounded-away-from-zero-and-$t$ variable can be extremal.

**A2** ★★ *One-sided Chebyshev (Cantelli).* Let $X$ have mean $0$ and variance $\sigma^2$. Prove that for all $t>0$,
$$P(X\ge t)\le\frac{\sigma^2}{\sigma^2+t^2}.$$
*Hint:* apply Chebyshev's inequality to $X+c$ for a well-chosen $c>0$, then optimize over $c$. Compare the result to the two-sided bound $\sigma^2/t^2$ at $t=\sigma$ and at $t=10\sigma$.

**A3** ★ *Jensen on two points, and the equality case.* Let $\varphi$ be convex, and let $X$ take values $a$ and $b$ with probabilities $1-p$ and $p$. Verify Jensen's inequality directly from the definition of convexity. Then show that if $\varphi$ is *strictly* convex, $a\ne b$ and $p\in(0,1)$, the inequality is strict. Which hypothesis fails in the classic "equality holds and yet $X$ is not constant" examples?

**A4** ★★ *A sub-Gaussian proxy dominates the variance.* Suppose $X$ satisfies $\mathbb{E}[e^{\lambda(X-\mathbb{E}X)}]\le e^{\lambda^2\sigma^2/2}$ for all $\lambda\in\mathbb{R}$. Show $\operatorname{Var}(X)\le\sigma^2$. *Hint:* expand both sides to order $\lambda^2$ as $\lambda\to0$ and justify the expansion. What does this tell you about Worked Example 4.8's factor of $25.25$?

**A5** ★★ *Expected maximum of sub-Gaussians.* Let $X_1,\dots,X_N$ be mean-zero, each sub-Gaussian with proxy $\sigma^2$ (no independence assumed). Prove
$$\mathbb{E}\Big[\max_{i\le N}X_i\Big]\le\sigma\sqrt{2\log N}.$$
Then explain in one sentence why this is the same computation as Massart's lemma, and why the absence of an independence assumption is essential for the application in §4.8.

**A6** ★★ *Hoeffding's lemma the easy way, for a symmetric sign.* Let $\sigma$ be a Rademacher variable. Show directly that $\log\mathbb{E}[e^{\lambda\sigma}]=\log\cosh\lambda\le\lambda^2/2$ for all $\lambda$, and confirm that this is exactly the proxy $(b-a)^2/4$ of Lemma 4.19 for $a=-1$, $b=1$. Where is the inequality tightest, and where does it waste the most?

**A7** ★★ *Chernoff for a Poisson count.* Let $X\sim\mathrm{Poisson}(\mu)$. Compute the CGF, carry out the Chernoff optimization for $P(X\ge(1+\eta)\mu)$ with $\eta>0$, and show
$$P\big(X\ge(1+\eta)\mu\big)\le\exp\big(-\mu\,h(\eta)\big),\qquad h(\eta)=(1+\eta)\log(1+\eta)-\eta .$$
Note that $h$ is exactly Bennett's function from Theorem 4.25, and say in one sentence why that is not a coincidence.

**A8** ★★★ *Bennett implies Bernstein.* Show that $h(u)=(1+u)\log(1+u)-u$ satisfies
$$h(u)\ \ge\ \frac{u^2}{2(1+u/3)}\qquad\text{for all }u\ge0,$$
and deduce Bernstein's inequality (Theorem 4.24) from Bennett's (Theorem 4.25). *Hint:* both sides vanish to second order at $u=0$; compare derivatives, or compare the functions $g(u)=h(u)\cdot2(1+u/3)/u^2$ against $1$ using a series argument.

**A9** ★★ *McDiarmid on the sample maximum.* Let $X_1,\dots,X_n$ be i.i.d. Uniform$[0,1]$ and $M_n=\max_iX_i$. Find the bounded-difference constants $c_i$ for $M_n$ as a function of $X_{1:n}$, apply Theorem 4.29, and compare the resulting deviation scale to the true one (recall $1-M_n$ is of order $1/n$). Explain precisely which feature of the bounded-differences hypothesis is responsible for the mismatch, and connect your answer to Worked Example 4.14.

**A10** ★ *VC dimension of unions of two intervals.* Let $\mathcal H$ be the class of indicators of unions of at most two closed intervals on $\mathbb{R}$. Prove $\mathrm{VC}(\mathcal H)=4$ by exhibiting a shattered set of size $4$ and an unrealizable labeling on every set of size $5$. Then guess and justify $\mathrm{VC}$ for unions of at most $k$ intervals.

**A11** ★★ *One parameter, infinite VC dimension.* Prove the claim in Common Confusion 4.8: the class $\mathcal H_{\sin}=\{x\mapsto\mathbf{1}_{\{\sin(\theta x)>0\}}:\theta\in\mathbb{R}\}$ has infinite VC dimension. *Hint:* take $x_i=2^{-i}$ for $i=1,\dots,n$ and build $\theta$ from the binary expansion of the target labels. State clearly where you use that $\theta$ ranges over an uncountable set, and say what this implies about "capacity control by counting parameters."

**A12** ★★ *How much the convenient $\lambda$ costs.* Proposition 4.21's proof bounds the tail by *guessing* $\lambda=t/(\nu^2+bt)$, which yields the exponent $-\frac{t^2}{2(\nu^2+bt)}$.
(a) Verify that substituting this $\lambda$ into $g(\lambda)=\frac{\lambda^2\nu^2}{2(1-b\lambda)}-\lambda t$ does give exactly that exponent.
(b) Now minimize $g$ properly on $(0,1/b)$. Show the exact minimizer is
$$\lambda^\star=\frac1b\Big(1-\sqrt{\tfrac{\nu^2}{\nu^2+2bt}}\Big),$$
and in particular that the guess is *not* optimal. *Hint:* $g'(\lambda)=0$ reduces to $b A\lambda^2-2A\lambda+2t=0$ with $A=\nu^2+2bt$; take the root in $(0,1/b)$.
(c) At $\nu^2=b=t=1$, evaluate $g$ at both. How much is lost, and is it worth the extra algebra in a bound you intend to quote?

**A13** ★ *The Bernstein condition adds like variance.* Let $X_1,\dots,X_n$ be independent and centered, with $X_i$ satisfying Definition 4.20 with parameters $(\nu_i^2,b)$ — the same $b$ for all $i$. Show $\sum_iX_i$ satisfies it with $\big(\sum_i\nu_i^2,\ b\big)$. Then specialize to the i.i.d. case and to the sample *mean*, and say what happens to $b$ when you divide by $n$.

**A14** ★★ *From tail bound to confidence radius.* Starting from Bernstein's tail bound $\exp\!\big(-\frac{nt^2}{2(\sigma^2+Mt/3)}\big)$, set it equal to $\delta$ and solve the resulting quadratic in $t$ exactly. Then show the clean upper bound
$$t\ \le\ \sqrt{\frac{2\sigma^2\log(1/\delta)}{n}}+\frac{2M\log(1/\delta)}{3n},$$
identifying where $\sqrt{u+v}\le\sqrt u+\sqrt v$ is used. Explain which of the two terms dominates for large $n$, and why that is the "right" behaviour.

**A15** ★★ *Why paired comparisons are tighter.* Two models $h_1,h_2$ are evaluated on the *same* test set of size $m$, with $0$–$1$ losses $L_i^{(1)},L_i^{(2)}\in\{0,1\}$. Let $D_i=L_i^{(1)}-L_i^{(2)}\in\{-1,0,1\}$.
(a) Apply Hoeffding to $\bar D_m$ directly and compare the radius to what you get by bounding each model's risk separately and subtracting.
(b) Show that if the two models agree on a fraction $1-q$ of examples, then $\operatorname{Var}(D_i)\le q$, and give the Bernstein radius in terms of $q$.
(c) Two models differ on $3\%$ of a $10{,}000$-example test set and differ in accuracy by $0.3$ points. At $\delta=0.05$, is the difference detectable? Compare with the $\pm1.36$-point interval of §4.5 and explain the discrepancy.

**A16** ★ *Metric entropy in your hands.* (a) Using Proposition 4.49, bound $\log N(B_2^d,\lVert\cdot\rVert_2,\varepsilon)$ for $(d,\varepsilon)=(2,0.5)$, $(10,0.1)$ and $(100,0.5)$, and evaluate. (b) For each, compute the contribution $\sqrt{2\log N/n}$ at $n=10^4$. (c) A colleague objects that a bound containing $N=10^{147}$ "obviously cannot be useful." Answer the objection in two sentences.

**A17** ★★★ *Chaining changes the exponent, not the constant.* Suppose a class has entropy profile $\log N(\varepsilon)=1/\varepsilon$ on $(0,1]$, with $B=1$.
(a) Minimize Proposition 4.50's bound $\varepsilon+\sqrt{2\log N(\varepsilon)/n}$ over $\varepsilon$ *analytically*, and show the optimum scales as $n^{-1/3}$ with an explicit constant.
(b) Evaluate Dudley's integral (Theorem 4.51) for this profile and show the bound scales as $n^{-1/2}$.
(c) Conclude that no improvement of the constant in (a) can match (b) for large $n$, and find the $n$ at which the two *numerical* bounds cross given Theorem 4.51's constant $12$.
(d) Now suppose instead $\log N(\varepsilon)=\varepsilon^{-2}$. Show Dudley's integral diverges, and explain in one sentence what that means about the class.

**A18** ★★ *Sensitivity, stability, privacy — the same constant three times.* A learning algorithm outputs the empirical mean of $n$ values in $[0,1]$.
(a) Compute its bounded-differences constants $c_i$ as a function of the sample, and state McDiarmid's radius at $\delta=0.05$, $n=1000$.
(b) Compute the $\ell_1$ sensitivity $\Delta f=\max\lvert f(S)-f(S')\rvert$ over neighbouring samples, and the scale of Laplace noise needed for $\varepsilon$-differential privacy.
(c) At $n=1000$, $\varepsilon=1$, compare the standard deviation of the privacy noise to the sampling standard deviation of the mean itself (take $\sigma^2=1/12$). At what $n$ does the privacy noise become negligible relative to sampling noise, and what does that tell you about when privacy is cheap?
(d) Repeat (b) for the *median* instead of the mean and explain why the answer is so much worse.

**A19** ★★★ *Lower bounds: proving that this chapter's upper bounds are unimprovable.* The chapter has claimed repeatedly that its inequalities are "tight in rate," resting on cited facts (Cramér's theorem, minimax bounds in §4.12). This exercise proves the two most important instances from scratch, so the claim is earned rather than borrowed.
(a) *(Gaussian tail lower bound.)* Let $Z\sim N(0,1)$ with density $\varphi(u)=e^{-u^2/2}/\sqrt{2\pi}$ and $\bar\Phi(u):=\mathbb{P}(Z\ge u)$. Prove that for all $u>0$,
$$\bar\Phi(u)\ \ge\ \frac{u}{1+u^2}\,\varphi(u).$$
*Hint:* verify by direct differentiation that $-\frac{d}{dt}\big[\tfrac{t}{1+t^2}\varphi(t)\big]\le\varphi(t)$ pointwise for $t>0$, then integrate from $u$ to $\infty$. (The companion upper bound $\bar\Phi(u)\le\varphi(u)/u$ follows by the same device applied to $\varphi(t)/t$.)
(b) *(No bound can beat $\sqrt{2\log(1/\delta)/n}$.)* Let $X_1,\dots,X_n$ be i.i.d. $N(\mu,\sigma^2)$, so $\bar X_n-\mu\sim N(0,\sigma^2/n)$ exactly (sums of independent Gaussians are Gaussian — Ch. 3, via MGFs). Show using (a) that any radius $t$ with $\mathbb{P}(\bar X_n-\mu\ge t)\le\delta$ must satisfy
$$t\ \ge\ \big(1-o(1)\big)\,\sigma\sqrt{\frac{2\log(1/\delta)}{n}}\qquad\text{as }\delta\to0,$$
and make the $o(1)$ explicit as a $\log\log$-type correction. Conclude that Hoeffding's radius $\sqrt{\log(1/\delta)/2n}$ for $[0,1]$-valued variables has exactly the necessary dependence on both $n$ and $\delta$ — the only slack is the constant ratio between the variance proxy and the true variance, precisely as Worked Example 4.12 measured.
(c) *(The $\sqrt{\log N}$ is real.)* Let $X_1,\dots,X_N$ be **independent** $N(0,\sigma^2)$. Using (a), prove that for all $N\ge50$,
$$\mathbb{E}\Big[\max_{i\le N}X_i\Big]\ \ge\ 0.43\,\sigma\sqrt{\log N},$$
matching Exercise A5's upper bound $\sigma\sqrt{2\log N}$ up to a constant. *Hint:* take $u=\sigma\sqrt{\log N}$, show $N\bar\Phi(\sqrt{\log N})\ge1$ for $N\ge50$, deduce $\mathbb{P}(\max\ge u)\ge1-e^{-1}$ from $(1-x)^N\le e^{-Nx}$, and handle the event $\{\max<u\}$ by bounding $\mathbb{E}[\max(0,-\max_iX_i)]\le\mathbb{E}[\max(0,-X_1)]$. Then answer: A5's *upper* bound needed no independence; where exactly does your *lower* bound use it, and what does that asymmetry tell you about when the $\log\lvert\mathcal H\rvert$ price in Theorem 4.34 is genuinely paid versus overcharged?

## Part B — Computational labs

Each lab asks for code and a short written interpretation. Report seeds and sample sizes; a number without a seed is not reproducible. Round consistently.

**B1** ★★ *Empirical tail versus three bounds.* Let $X_i$ be i.i.d. Uniform$[0,1]$ and $\bar X_n$ the sample mean.
(a) For $n\in\{10,100,1000,10^4\}$ and $\delta\in\{0.1,0.05,0.01\}$, estimate by Monte Carlo (at least $2\times10^5$ replicates) the smallest $t$ with $P(\bar X_n-\tfrac12\ge t)\le\delta$. Tabulate against the one-sided radii from Markov (applied to $\bar X_n$ itself), Chebyshev, Hoeffding, and the CLT approximation.
(b) Plot all four radii against $n$ on log-log axes and read off the slopes.
(c) Report the ratio Hoeffding/empirical in each cell. Is it approximately constant in $n$? In $\delta$? Explain what you find using Worked Example 4.12's decomposition.
(d) Markov's radius will behave oddly. Explain why, in terms of which hypothesis Markov is and is not using.

**B2** ★★ *Estimating Rademacher complexity for a linear class.* Let $\mathcal F=\{x\mapsto\langle w,x\rangle:\lVert w\rVert_2\le1\}$ and let $x_1,\dots,x_n$ be i.i.d. standard Gaussian in $\mathbb{R}^d$ (so $\lVert x_i\rVert_2\approx\sqrt d$).
(a) Show analytically that $\hat{\mathfrak{R}}_n(\mathcal F)=\frac1n\mathbb{E}_\sigma\lVert\sum_i\sigma_ix_i\rVert_2$, so no optimization over $w$ is needed.
(b) Estimate it by averaging over at least $4000$ random sign vectors, for $d\in\{1,4,16,64,128\}$ at $n=200$, and for $d=16$ at $n\in\{50,100,200,400,800,1600\}$.
(c) Compare to $\sqrt{d/n}$ and to the bound of Proposition 4.41 with $X=\max_i\lVert x_i\rVert_2$ measured from your sample. Fit the exponent of $n$ and of $d$.
(d) Your estimates should sit slightly *below* $\sqrt{d/n}$ in every cell. Prove that this must happen, and name the inequality responsible. Why does the gap shrink as $n$ grows?

**B3** ★★ *Bernstein versus Hoeffding for a rare event.* Let $X_i\sim$ Bernoulli$(p)$ with $p=0.01$ and $n=10^4$.
(a) Compute the exact $P(\bar X_n-p\ge t)$ from the Binomial CDF for $t$ on a grid, and the Hoeffding, Bernstein, and Bennett bounds.
(b) Tabulate, for $\delta=0.05$, the radii from each bound and the exact quantile.
(c) Repeat for $p\in\{0.5,0.1,0.01,0.001\}$ and plot the ratio Hoeffding-radius / Bernstein-radius against $p$. At which $p$ do they agree, and does that match Worked Example 4.8's computation of where the Hoeffding proxy is tight?
(d) Now use the *empirical* Bernstein bound of Proposition 4.26 with the variance estimated from a single simulated sample. How much do you lose relative to knowing $p$?

**B4** ★★★ *A vacuous bound for a real network.* Consider a fully connected ReLU network $784\to64\to10$ on MNIST-sized data ($n=60{,}000$, $\delta=0.05$).
(a) Count the parameters $W$ exactly (including biases).
(b) Compute the finite-class bound obtained by treating each float32 parameter as $32$ bits, i.e. $\log\lvert\mathcal H\rvert=32W\log2$, via Theorem 4.34.
(c) Compute the VC-based bound of Theorem 4.46 using $d=W\log_2W$ as a stand-in for Proposition 4.47(2), and also attempt the classical form $\sqrt{8(d\log(2en/d)+\log(4/\delta))/n}$. Report what happens to the latter and explain it.
(d) Solve for the $n$ that would make each bound equal $0.1$. Express the answer as a multiple of the MNIST training set size.
(e) If you have the means, train the network and report the actual train–test gap. Compute the ratio bound/measured for each bound, and write two sentences on what the ratio does and does not tell you.

## Part C — Paper-style derivations

**C1** ★★★ *Symmetrization.* Prove Theorem 4.37: for any class $\mathcal F$ of functions $\mathcal Z\to\mathbb{R}$ with $\mathbb{E}\sup_{f}\lvert\cdot\rvert$ finite,
$$\mathbb{E}\Big[\sup_{f\in\mathcal F}\big(\mathbb{E}f-\tfrac1n\textstyle\sum_if(Z_i)\big)\Big]\ \le\ 2\,\mathfrak{R}_n(\mathcal F).$$
Write the proof as a paper would: introduce the ghost sample and say what probability space it lives on, justify the exchange of supremum and expectation with the correct inequality direction, state where independence and identical distribution are each used, and identify the exact source of the factor $2$. Then answer: where in the argument would the proof break if the $Z_i$ were merely uncorrelated rather than independent?

**C2** ★★ *The finite-class bound, end to end, and its bottleneck.* Starting from Hoeffding's inequality and the union bound, derive Theorem 4.34 with all constants, for a loss taking values in $[0,1]$ and a class of size $N$. Then:
(a) State precisely which step becomes the bottleneck when $\mathcal H$ is infinite, and why each of the other steps survives.
(b) Show quantitatively that the failing step is not repairable by being cleverer about constants: given $n=1000$ and target $t=0.05$, compute the largest $N$ for which the union bound is not yet vacuous, and comment.
(c) Explain what Rademacher complexity and covering numbers each replace that step with, in one sentence each.

**C3** ★★★ *A norm-based bound for a two-layer network, evaluated.* Let
$$\mathcal F=\Big\{x\mapsto\sum_{j=1}^{m}v_j\,\rho\big(\langle w_j,x\rangle\big)\ :\ \lVert v\rVert_1\le B_2,\ \max_j\lVert w_j\rVert_2\le B_1\Big\}$$
with $\rho$ the ReLU and inputs satisfying $\lVert x\rVert_2\le X$.
(a) Derive $\mathfrak{R}_n(\mathcal F)\le2B_1B_2X/\sqrt n$, stating where you use the contraction lemma, the positive-homogeneity of the ReLU, and the fact that the Rademacher complexity of a convex hull equals that of its generators. Note carefully that $m$ does not appear.
(b) Evaluate the resulting generalization gap $2\mathfrak{R}_n+\sqrt{\log(2/\delta)/2n}$ at $n=60{,}000$, $\delta=0.1$, $X=10$, for $(B_1,B_2)\in\{(1,1),(2,5),(5,10),(10,20)\}$.
(c) Solve for the largest product $B_1B_2$ compatible with a gap of at most $0.1$.
(d) Trained networks routinely have product norms far above your answer to (c), and their weight norms *grow* during training while test error falls. Write a paragraph on what the gap between (b) and an observed test error near $0.02$ implies — distinguishing carefully between "the bound is loose," "the bound is measuring the wrong thing," and "the bound is false." Which of the three is it, and how do you know?
# Appendix A — Self-check answers

**Self-check 4.1.** (1) $\mathbb{E}[e^{-X}]=\int_0^1e^{-x}dx=1-e^{-1}=0.632121$ is larger than $e^{-\mathbb{E}X}=e^{-1/2}=0.606531$, as Jensen demands for the convex function $e^{-x}$. (2) In two places: finiteness of $\mathbb{E}\lvert X\rvert$ guarantees $\mathbb{E}[X]$ exists and is finite, so a supporting line *at the point $\mathbb{E}[X]$* is available; and it makes the expectation of the affine minorant $\varphi(\mathbb{E}X)+c(X-\mathbb{E}X)$ well-defined so we may take expectations of both sides. (3) Worked Example 4.3 computes $D_{\mathrm{KL}}(\mathrm{Ber}(0.9)\Vert\mathrm{Ber}(0.5))=0.368064$ and $D_{\mathrm{KL}}(\mathrm{Ber}(0.5)\Vert\mathrm{Ber}(0.9))=0.510826$. Two numbers, not equal; the definition is not symmetric in $P$ and $Q$ because only one of them appears inside the logarithm's numerator and only one supplies the integrating measure. (4) For affine $\varphi$, Jensen holds with equality in both directions, i.e. $\mathbb{E}[\varphi(X)]=\varphi(\mathbb{E}X)$. Consistent, because the equality condition requires strict convexity to force $X$ constant; an affine function is convex but not strictly so, and the supporting line coincides with $\varphi$ everywhere.

**Self-check 4.2.** (1) Take $X=\pm2\sigma$ with probability $1/8$ each and $X=0$ with probability $3/4$. Then $\operatorname{Var}(X)=4\sigma^2\cdot\tfrac14=\sigma^2$ and $P(\lvert X\rvert\ge2\sigma)=\tfrac14=\sigma^2/(2\sigma)^2$: exactly tight. (2) Corollary 4.11 requires $n\ge\sigma^2/(\delta\varepsilon^2)$. Halving $\varepsilon$ multiplies $n$ by $4$; halving $\delta$ multiplies $n$ by $2$. The second is the expensive one — contrast Hoeffding, where $\delta$ enters only through $\log(1/\delta)$. (3) Markov gives $P(X\ge2)\le3/2$. Useless because probabilities are at most $1$ anyway; this happens whenever the threshold is below the mean. (4) Because Chebyshev *is* Markov applied to the nonnegative variable $(X-\mu)^2$ at threshold $t^2$: $P(\lvert X-\mu\rvert\ge t)=P((X-\mu)^2\ge t^2)\le\mathbb{E}[(X-\mu)^2]/t^2$.

**Self-check 4.3.** (1) Because the event identity $\{X\ge t\}=\{e^{\lambda X}\ge e^{\lambda t}\}$ requires $x\mapsto e^{\lambda x}$ to be increasing, which needs $\lambda\ge0$. With $\lambda<0$ the map reverses order and the same computation bounds the *lower* tail. (2) $\psi_X(\lambda)=\lambda^2\sigma^2/2$, so $\psi_X^*(t)=\sup_\lambda(\lambda t-\lambda^2\sigma^2/2)=t^2/(2\sigma^2)$, giving $P(X\ge t)\le e^{-t^2/2\sigma^2}$ — the definitional sub-Gaussian bound. (3) The infimum is over an expression that is $+\infty$ for every $\lambda>0$, so Theorem 4.14 gives the trivial bound $1$. Heavy-tailed variables get nothing from the Chernoff method; one must use Markov on $\lvert X\rvert^k$ for whatever moments exist, obtaining polynomial rather than exponential decay. (4) At $\lambda=0$ the objective is $e^{0}\,\mathbb{E}[e^{0}]=1$, independent of $X$ and $t$. This is why the Chernoff bound can never be worse than trivial, and why any useful bound comes from the objective *decreasing* to the right of zero — which happens exactly when $t>\mathbb{E}X$, since the derivative at $0$ is $\psi'(0)-t=\mathbb{E}X-t$.

**Self-check 4.4.** (1) Both sides are finite and smooth near $\lambda=0$ (differentiation under the expectation is licensed by dominated convergence, Ch. 2). Expanding, $1+\lambda^2\operatorname{Var}(X)/2+o(\lambda^2)\le1+\lambda^2\sigma^2/2+o(\lambda^2)$; subtract $1$, divide by $\lambda^2>0$, let $\lambda\to0$: $\operatorname{Var}(X)\le\sigma^2$. (2) Yes — it is bounded, so Lemma 4.19 applies with $(b-a)^2/4=1$. Its true variance is $1/3$, so the proxy overstates by a factor of $3$. (3) Because the MGF of $Z^2-1$ is finite only for $\lambda<1/2$, and $b=2$ here; at $\lambda=1/b=1/2$ the integral $\int e^{z^2/2}e^{-z^2/2}dz$ fails to converge, so the CGF is $+\infty$ and no bound of that form exists. (4) The quadratic (sub-Gaussian) regime. For a sum of $n$ terms with $\nu^2\propto n$, a deviation $t\asymp\nu$ makes $bt\asymp b\sqrt n$, which is negligible against $\nu^2\asymp n$; so the exponent $t^2/(2(\nu^2+bt))\approx t^2/2\nu^2$. This is why CLT-scale deviations look Gaussian and Bernstein's improvement over Hoeffding shows up only in the far tail or in the small-variance regime.

**Self-check 4.5.** (1) Because each summand contributes its own range through Hoeffding's lemma, and the CGFs add (Lemma 4.13). If one variable has a huge range and the rest are constants, $\sum_i(b_i-a_i)^2$ is dominated by that one term and the bound degrades to a statement about a single variable — correctly, since there is no averaging happening. (2) At $\sigma^2=1/4$: Bernstein's leading exponent $nt^2/2\sigma^2$ becomes $2nt^2$, which is exactly Hoeffding's. That is the Hoeffding proxy for $[0,1]$, attained by a fair coin. (3) Because Bernstein's exponent can be inverted in closed form to give a confidence radius, while Bennett's $h$ must be inverted numerically; Bernstein also composes more transparently in longer arguments. Bennett is what you use when you want the sharpest number. (4) With $p=0.002$, $n=10^5$, $\delta=0.05$: Hoeffding's one-sided radius is $\sqrt{\log(1/\delta)/2n}=0.003870$; Bernstein's, using $\sigma^2=p(1-p)=0.001996$, is $0.000356$ — narrower by a factor of $10.87$. Report Bernstein, with the caveat that it uses $\sigma^2$, which you estimated from the same data; the fully honest choice is the empirical Bernstein bound of Proposition 4.26.

**Self-check 4.6.** (1) No. The proof of Theorem 4.29 uses only independence (to make the Doob martingale increments conditionally centered and to bound their conditional ranges); nowhere is a common distribution needed. This matters: McDiarmid applies to independent-but-not-identical data, e.g. stratified samples. (2) $f(x)=\sum_ix_i$ on $\{0,1\}^n$. Each $c_i=1$, so McDiarmid gives deviations of order $\sqrt n$, and for i.i.d. fair bits the true standard deviation is $\sqrt n/2$. Same order, so the $\sqrt{\sum_ic_i^2}$ scaling cannot be improved without extra assumptions. (3) Because for each fixed $h$ the map has differences at most $c_i$, and $\lvert\sup_hF_h(x)-\sup_hF_h(x')\rvert\le\sup_h\lvert F_h(x)-F_h(x')\rvert\le c_i$: the supremum of a family of functions is no more sensitive to one coordinate than the most sensitive member. (4) It gives $\mathbb{E}[R(A(S))-\hat R_n(A(S))]\le0$: a perfect generalization guarantee. Useless, because stability bounds the *gap*, not the risk. A constant predictor generalizes perfectly and may be wrong almost always. Any real guarantee needs stability *and* small empirical risk, and these pull against each other.

**Self-check 4.7.** (1) Because the chain is $R(\hat h)\le\hat R_n(\hat h)+\varepsilon\le\hat R_n(h^*)+\varepsilon\le R(h^*)+2\varepsilon$, and $\varepsilon$ is spent twice — once going from population to empirical at $\hat h$, once coming back at $h^*$. Only the *first* inequality needs uniform convergence, because $\hat h$ is data-dependent; the third concerns the fixed hypothesis $h^*$ and would follow from Hoeffding alone. The middle step is ERM optimality and costs nothing. (2) $n\ge(\log100+\log200)/(2\cdot0.05^2)=1980.7$, so $n=1981$. (3) $\varepsilon=\sqrt{(\log N+\log(2/\delta))/2n}$ grows from $0.0728$ to $0.0936$ at $n=1000$, $\delta=0.05$ — a factor of $1.286$. Tripling the number of digits in $\lvert\mathcal H\rvert$ costs less than $30\%$; this logarithmic cheapness is the whole point of the union bound. (4) Because $\hat h$ depends on the sample, $\hat R_n(\hat h)$ is not an average of independent terms whose common mean is $R(\hat h)$ — the summands and the function being averaged are correlated. Hoeffding's hypothesis (a fixed function of independent variables) fails.

**Self-check 4.8.** (1) $1/2$. Choosing $f(z_i)=1$ when $\sigma_i=+1$ and $f(z_i)=0$ otherwise achieves $\frac1n\lvert\{i:\sigma_i=1\}\rvert$, whose expectation is $1/2$. Maximal for $[0,1]$-valued classes; the class is useless for learning. (2) Because $\mathbb{E}_\sigma[\sigma_i]=0$ and there is no supremum to exploit: $\mathbb{E}_\sigma\frac1n\sum_i\sigma_if(z_i)=0$ exactly. Complexity in this sense is entirely about the *choice* available to an adversary fitting noise. (3) Because two steps each need their own confidence budget: McDiarmid on the supremum (constants $1/n$) and a second McDiarmid transferring $\hat{\mathfrak{R}}_n$ to $\mathfrak{R}_n$, with $\delta$ split in half — and the factor $2$ multiplying $\hat{\mathfrak{R}}_n$ doubles the second contribution, giving $1+2=3$. (4) Under $\lVert u\rVert_2\le B_2$ you must pass from an $\ell_1$ to an $\ell_2$ constraint by Cauchy–Schwarz, which introduces $\sqrt m$: the bound becomes $2B_1B_2X\sqrt m/\sqrt n$. Width-independence is a property of $\ell_1$ (i.e. sparsity-flavored) control of the output layer, not of two-layer networks as such.

**Self-check 4.9.** (1) $\mathrm{VC}(\mathcal H)=k$: the whole space is shattered, and no set of size $k+1$ exists. (2) A maximum, because uniform convergence must hold whatever configuration the data present, and the classical theory is distribution-free — the bound cannot depend on $P$. It costs us whenever the data distribution concentrates on configurations the class handles easily; empirical Rademacher complexity (§4.8) is the data-dependent repair. (3) $\mathrm{VC}(\mathcal H_1)\le\mathrm{VC}(\mathcal H_2)$, because $\mathcal H_1|_S\subseteq\mathcal H_2|_S$ for every $S$, so any $S$ shattered by $\mathcal H_1$ is shattered by $\mathcal H_2$. (4) $n=5$. At $n=6$, Sauer–Shelah gives $\Pi(6)\le\sum_{i\le5}\binom6i=63<64$, so the growth function must have fallen strictly below $2^n$ immediately after the VC dimension.

**Self-check 4.10.** (1) Because the covering number enters through Massart's lemma as $\sqrt{2\log N/n}$ — a union bound over the net — so only $\log N$ is charged. This is what makes the curse of dimensionality survivable: $N=10^{100}$ but $\log N=230$. (2) As $\varepsilon\to0$ the first term vanishes but $\log N(\varepsilon)\to\infty$, so the bound diverges; as $\varepsilon\to B$ the net collapses to a single point, $\log N\to0$, and the bound tends to the trivial $B$. The useful bound is at the interior optimum, which balances approximation error against union-bound cost. (3) Because $\sqrt{d\log(1/\varepsilon)}$ grows only logarithmically as $\varepsilon\to0$, and any function growing slower than $\varepsilon^{-1+\eta}$ is integrable on $(0,B]$. Divergence needs entropy growth like $\varepsilon^{-2}$ or faster. (4) VC classes satisfy $\log N(\mathcal F,d_n,\varepsilon)\lesssim d\log(1/\varepsilon)$ (Haussler's refinement of Sauer–Shelah). Then Dudley's integral is $\int_0^1\sqrt{d\log(3/\varepsilon)}\,d\varepsilon=O(\sqrt d)$, giving $\hat{\mathfrak{R}}_n=O(\sqrt{d/n})$ — the $\log n$ of Theorem 4.46 was an artifact of paying the union-bound price at a single scale.

**Self-check 4.11.** (1) VC dimension, growth function, covering numbers and class-level Rademacher complexity all depend only on the architecture, the class it realizes, and $n$ — all identical between the true-label and random-label runs. The outcomes differ (near-zero versus chance-level test error). A quantity that takes the same value in both cases cannot account for a difference between them. (2) The bound is proportional to a product of weight norms, and those norms grow as training proceeds while test error falls, so the guarantee gets *worse* exactly when generalization gets *better*. Worse than looseness: a loose bound can in principle be tightened by improving constants or the analysis, but a bound anti-correlated with its target is tracking the wrong quantity, and no constant-hunting fixes that. (3) No. Every theorem in §4.9 is true and every bound valid; the random-labels experiment actually *confirms* that the VC dimension is at least $n$, so the estimates were roughly right. What fails is the inference "small capacity, therefore generalization" as an *explanation* of deep learning, because capacity is not small. (4) PAC-Bayes, in the hands of Dziugaite and Roy, produces nonvacuous numbers. The object bounded is the expected risk of a *distribution* $Q$ over weights — a stochastic network — not the single deterministic network you deploy.

# Solutions to Exercises

## Part A

**A1.** Let $X=t$ with probability $a/t$ and $X=0$ otherwise (legitimate since $a/t\le1$). Then $\mathbb{E}[X]=t\cdot(a/t)=a$ and $P(X\ge t)=a/t$: Markov is attained.

*Why this is extremal.* Look at where Markov's proof loses. It writes $\mathbb{E}[X]=\mathbb{E}[X\mathbf 1_{\{X\ge t\}}]+\mathbb{E}[X\mathbf 1_{\{X<t\}}]\ge\mathbb{E}[X\mathbf 1_{\{X\ge t\}}]\ge t\,P(X\ge t)$. Two inequalities. The first is equality iff $X=0$ almost surely on $\{X<t\}$; the second iff $X=t$ almost surely on $\{X\ge t\}$. So equality forces $X\in\{0,t\}$ a.s., which is precisely the construction. Any variable placing mass strictly inside $(0,t)$ wastes expectation on outcomes that do not trigger the event; any variable placing mass above $t$ wastes expectation by exceeding the threshold more than necessary.

**A2.** For any $c>0$, the event $\{X\ge t\}$ implies $\{X+c\ge t+c\}$ and hence $\{\lvert X+c\rvert\ge t+c\}$. Markov applied to $(X+c)^2$ gives
$$P(X\ge t)\le\frac{\mathbb{E}[(X+c)^2]}{(t+c)^2}=\frac{\sigma^2+c^2}{(t+c)^2},$$
using $\mathbb{E}X=0$. Differentiating in $c$ and setting to zero: $2c(t+c)^2=2(t+c)(\sigma^2+c^2)$, i.e. $c(t+c)=\sigma^2+c^2$, i.e. $ct=\sigma^2$, so $c^\star=\sigma^2/t$. Substituting,
$$\frac{\sigma^2+\sigma^4/t^2}{(t+\sigma^2/t)^2}=\frac{\sigma^2(t^2+\sigma^2)/t^2}{(t^2+\sigma^2)^2/t^2}=\frac{\sigma^2}{\sigma^2+t^2}.$$

*Comparison.* At $t=\sigma$: Cantelli gives $1/2$, while two-sided Chebyshev gives $\sigma^2/t^2=1$ — vacuous. At $t=10\sigma$: $1/101=0.009901$ versus $0.01$. So the one-sided refinement is a genuine improvement precisely in the near-mean regime where Chebyshev is vacuous, and asymptotically irrelevant in the far tail. The moral: shifting before applying Markov is free, and free parameters should always be optimized.

**A3.** Convexity says exactly $\varphi((1-p)a+pb)\le(1-p)\varphi(a)+p\varphi(b)$ for $p\in[0,1]$. The left side is $\varphi(\mathbb{E}X)$ and the right side is $\mathbb{E}[\varphi(X)]$, so Jensen for two-point distributions *is* the definition of convexity — this is worth internalizing, because it shows Jensen contains no analysis beyond convexity itself; the general case is convexity plus a limiting/supporting-line argument. If $\varphi$ is strictly convex, $a\ne b$ and $p\in(0,1)$, the defining inequality is strict, hence so is Jensen's.

The classic "equality without constancy" examples always have $\varphi$ affine on an interval containing the support of $X$ — for instance $\varphi(x)=\lvert x\rvert$ with $X\ge0$, or $\varphi(x)=\max(x,0)$ with $X\ge0$. Strictness, not convexity, is the hypothesis that fails.

**A4.** Write $Y=X-\mathbb{E}X$ and $M(\lambda)=\mathbb{E}[e^{\lambda Y}]$. The hypothesis gives $M(\lambda)\le e^{\lambda^2\sigma^2/2}<\infty$ for all real $\lambda$, so the MGF is finite on a neighborhood of $0$; by the standard dominated-convergence argument (Ch. 2) $M$ is twice differentiable there with $M'(0)=\mathbb{E}[Y]=0$ and $M''(0)=\mathbb{E}[Y^2]=\operatorname{Var}(X)$. Taylor:
$$1+\tfrac{\lambda^2}{2}\operatorname{Var}(X)+o(\lambda^2)=M(\lambda)\le e^{\lambda^2\sigma^2/2}=1+\tfrac{\lambda^2}{2}\sigma^2+o(\lambda^2).$$
Subtract $1$, divide by $\lambda^2/2>0$, let $\lambda\to0$: $\operatorname{Var}(X)\le\sigma^2$.

*Reading Worked Example 4.8.* The Bernoulli$(0.01)$ has variance $0.0099$ and Hoeffding proxy $0.25$, a ratio of $25.25$. A4 says the inequality can only go this way — the proxy is always at least the variance — and puts no upper limit on the ratio. So a large ratio is not evidence of an error; it is evidence that a *variance-aware* inequality (Bernstein, Bennett) has room to improve on the sub-Gaussian one. Conversely, when the ratio is near $1$ (a fair coin), Bernstein cannot help.

**A5.** For any $\lambda>0$, by Jensen applied to the concave function $\log$,
$$\lambda\,\mathbb{E}\Big[\max_iX_i\Big]=\mathbb{E}\Big[\log e^{\lambda\max_iX_i}\Big]\le\log\mathbb{E}\Big[e^{\lambda\max_iX_i}\Big]=\log\mathbb{E}\Big[\max_ie^{\lambda X_i}\Big]\le\log\sum_{i=1}^N\mathbb{E}\big[e^{\lambda X_i}\big]\le\log\big(Ne^{\lambda^2\sigma^2/2}\big).$$
Hence $\mathbb{E}\max_iX_i\le\frac{\log N}{\lambda}+\frac{\lambda\sigma^2}{2}$ for every $\lambda>0$. The right side is minimized at $\lambda^\star=\sqrt{2\log N}/\sigma$, giving $\sigma\sqrt{2\log N}$.

This is Massart's lemma with the roles renamed: there the variables are $\langle\sigma,a\rangle$ for $a$ in a finite set, each sub-Gaussian with proxy $\lVert a\rVert_2^2$. Independence is never used — the step $\mathbb{E}\max_ie^{\lambda X_i}\le\sum_i\mathbb{E}e^{\lambda X_i}$ is just "the max is at most the sum of nonnegatives," a union bound in disguise. That is essential in §4.8, where the variables indexed by hypotheses are wildly dependent (two similar hypotheses give nearly the same loss vector); a method requiring independence across the class would be unusable.

**A6.** Compare power series. $\cosh\lambda=\sum_{k\ge0}\lambda^{2k}/(2k)!$ and $e^{\lambda^2/2}=\sum_{k\ge0}\lambda^{2k}/(2^kk!)$. Now
$$2^kk!=\prod_{j=1}^k(2j)\quad\text{and}\quad(2k)!=\prod_{j=1}^k(2j)\cdot\prod_{j=1}^k(2j-1)\ \ge\ \prod_{j=1}^k(2j)=2^kk!,$$
so every coefficient of $\cosh$ is at most the corresponding coefficient of $e^{\lambda^2/2}$. Both series have nonnegative terms, so $\cosh\lambda\le e^{\lambda^2/2}$ for all real $\lambda$; take logarithms.

Lemma 4.19 with $a=-1$, $b=1$ gives proxy $(b-a)^2/4=1$, i.e. $\psi(\lambda)\le\lambda^2/2$ — exactly what we just proved, so Hoeffding's lemma is sharp for the symmetric sign.

*Where it is tight and where it wastes.* Both sides are $\lambda^2/2-\lambda^4/12+O(\lambda^6)$ versus $\lambda^2/2+\lambda^4/8+O(\lambda^6)$, so the inequality is tight to second order at $\lambda=0$ — which is why Hoeffding gets the right *rate* for central deviations. For large $\lvert\lambda\rvert$, $\log\cosh\lambda=\lvert\lambda\rvert-\log2+o(1)$ grows linearly while $\lambda^2/2$ grows quadratically: enormous waste. Since large $\lambda$ is what the Chernoff optimization selects for large $t$, this is precisely the far-tail looseness that Bennett's inequality (whose proof keeps the true CGF shape) repairs.

**A7.** For $X\sim\mathrm{Poisson}(\mu)$, $M(\lambda)=\sum_{k\ge0}e^{\lambda k}e^{-\mu}\mu^k/k!=e^{-\mu}e^{\mu e^\lambda}$, so $\psi(\lambda)=\mu(e^\lambda-1)$, finite for all $\lambda$. The Chernoff bound with threshold $(1+\eta)\mu$ is
$$P\big(X\ge(1+\eta)\mu\big)\le\exp\Big(\mu(e^\lambda-1)-\lambda(1+\eta)\mu\Big).$$
Differentiate the exponent: $\mu e^\lambda-(1+\eta)\mu=0$ gives $\lambda^\star=\log(1+\eta)>0$, which is admissible. Substituting, $e^{\lambda^\star}=1+\eta$ and the exponent becomes
$$\mu\big(1+\eta-1\big)-\mu(1+\eta)\log(1+\eta)=-\mu\Big[(1+\eta)\log(1+\eta)-\eta\Big]=-\mu\,h(\eta).$$

*Why Bennett's function appears.* Not a coincidence: Bennett's proof bounds the CGF of a bounded, mean-zero summand by that of a suitably scaled and centered Poisson variable, and the Poisson is the extremal case for that bound. So Bennett's inequality *is* the Poisson tail bound, transplanted; $h$ is the Poisson large-deviation rate function. This also explains why Bennett is sharp for rare events (where counts are nearly Poisson) and why Bernstein, obtained by weakening $h$, loses exactly there.

**A8.** Let $F(u)=h(u)-\dfrac{u^2}{2(1+u/3)}=h(u)-\dfrac{3u^2}{6+2u}$ for $u\ge0$. We show $F\ge0$ by differentiating twice.

$F(0)=0$. Next, $h'(u)=\log(1+u)$ and
$$\frac{d}{du}\frac{3u^2}{6+2u}=\frac{6u(6+2u)-6u^2}{(6+2u)^2}=\frac{36u+6u^2}{(6+2u)^2}=\frac{3u(6+u)}{2(3+u)^2},$$
so $F'(u)=\log(1+u)-\dfrac{3u(6+u)}{2(3+u)^2}$ and $F'(0)=0$. Differentiating the second term once more,
$$\frac{d}{du}\frac{3(6u+u^2)}{2(3+u)^2}=\frac32\cdot\frac{(6+2u)(3+u)^2-(6u+u^2)\cdot2(3+u)}{(3+u)^4}=\frac32\cdot\frac{(6+2u)(3+u)-2(6u+u^2)}{(3+u)^3}=\frac{27}{(3+u)^3},$$
since $(6+2u)(3+u)-2(6u+u^2)=18+12u+2u^2-12u-2u^2=18$. Hence
$$F''(u)=\frac1{1+u}-\frac{27}{(3+u)^3}.$$
Now $(3+u)^3-27(1+u)$ vanishes at $u=0$ and has derivative $3(3+u)^2-27\ge0$ for $u\ge0$, so it is nonnegative; therefore $F''\ge0$ on $[0,\infty)$. Integrating twice from $0$ with $F(0)=F'(0)=0$ gives $F\ge0$, i.e. $h(u)\ge u^2/(2(1+u/3))$.

*Deduction.* Bennett's bound for a sum of centered variables bounded by $b$ with variance sum $\nu^2$ is $\exp\big(-\tfrac{\nu^2}{b^2}h(\tfrac{bt}{\nu^2})\big)$. Substituting the inequality with $u=bt/\nu^2$,
$$\frac{\nu^2}{b^2}\,h\!\Big(\frac{bt}{\nu^2}\Big)\ \ge\ \frac{\nu^2}{b^2}\cdot\frac{(bt/\nu^2)^2}{2\big(1+bt/(3\nu^2)\big)}=\frac{t^2}{2\big(\nu^2+bt/3\big)},$$
which is exactly Bernstein's exponent. So Bernstein is Bennett with $h$ replaced by a rational lower bound that is tight to second order at $0$ — losing nothing in the Gaussian regime and losing the $\log$ improvement in the far tail.

**A9.** *Constants.* Changing $x_i$ alone can move $\max_jx_j$ by at most $1$ (the whole range of $[0,1]$), and this is attained: from $x=(1,0,\dots,0)$ to $x'=(0,0,\dots,0)$ the max drops from $1$ to $0$. So $c_i=1$ for every $i$ and $\sum_ic_i^2=n$.

*McDiarmid.* $P(\lvert M_n-\mathbb{E}M_n\rvert\ge t)\le2e^{-2t^2/n}$, so the radius at $\delta=0.05$ is $\sqrt{n\log(2/\delta)/2}$, which at $n=100$ equals $13.58$ — on a quantity confined to $[0,1]$. Fully vacuous, and *increasing* in $n$.

*Truth.* $M_n$ has CDF $t^n$ on $[0,1]$, so $\mathbb{E}M_n=n/(n+1)$ and $\operatorname{Var}(M_n)=n/((n+1)^2(n+2))$, giving standard deviation $0.0098$ at $n=100$. The bound overstates by a factor of about $1{,}385$, and the true fluctuation *shrinks* like $1/n$ while the bound grows like $\sqrt n$.

*Diagnosis.* Bounded differences is a worst-case-over-configurations condition: $c_i$ must cover the most sensitive point of the cube. But $M_n$ is sensitive to coordinate $i$ only when $x_i$ is the maximum or would become it — an event of probability $O(1/n)$ under the product measure. McDiarmid charges the worst case at all $n$ coordinates simultaneously, so it pays $n\cdot1$ where the truth is $n\cdot O(1/n^2)$. This is exactly the pathology of Worked Example 4.14, where the median's McDiarmid radius was $13.65$ against a true standard deviation of $0.049$; and the repair is the same, namely Efron–Stein, which averages $\mathbb{E}[(f(X)-f(X^{(i)}))^2]$ over configurations instead of maximizing.

**A10.** *Shattering four points.* Take $x_1<x_2<x_3<x_4$, say $1,2,3,4$. Any subset $S\subseteq\{1,2,3,4\}$ consists of at most two maximal runs of consecutive elements — the most fragmented possibilities are $\{1,3\}$, $\{2,4\}$, $\{1,4\}$, $\{1,3,4\}$, and so on, all with two runs, because three runs on four positions would need positions $1,3,5$. Each run is realized by one interval, so at most two intervals suffice, and all $16$ subsets are realized.

*Failure at five.* With $x_1<\dots<x_5$, the labeling $\{x_1,x_3,x_5\}$ has three maximal runs. Any union of two intervals meets $\{x_1,\dots,x_5\}$ in at most two runs (an interval, being convex, picks out a contiguous block), so this labeling is unrealizable. Hence $\mathrm{VC}=4$.

*General $k$.* $\mathrm{VC}=2k$. Lower bound: $2k$ points can be labeled with at most $k$ runs (alternating is the worst case, giving exactly $k$ runs of the selected points), so $k$ intervals suffice. Upper bound: on $2k+1$ points, selecting the $k+1$ points in odd positions produces $k+1$ runs, which $k$ intervals cannot produce.

**A11.** Fix $n$ and take $x_i=2^{-i}$ for $i=1,\dots,n$. Given a target labeling $y\in\{0,1\}^n$, set
$$\theta=\pi\Big(1+\sum_{j=1}^n(1-y_j)2^j\Big).$$
Then
$$\theta x_i=\pi\Big(2^{-i}+\sum_{j=1}^n(1-y_j)2^{\,j-i}\Big).$$
Split the sum. For $j>i$, $2^{j-i}$ is an even integer, so those terms contribute integer multiples of $2\pi$ and leave $\sin$ unchanged. The term $j=i$ contributes $\pi(1-y_i)$. For $j<i$ the terms are fractional and sum to $f_i:=\sum_{j<i}(1-y_j)2^{-(i-j)}\le\sum_{k=1}^{i-1}2^{-k}=1-2^{-(i-1)}$. Therefore, modulo $2\pi$,
$$\theta x_i\equiv\pi\big(1-y_i\big)+\pi\big(2^{-i}+f_i\big),\qquad 0<2^{-i}+f_i\le1-2^{-i}<1 .$$
If $y_i=1$ the argument lies strictly in $(0,\pi)$, so $\sin(\theta x_i)>0$; if $y_i=0$ it lies strictly in $(\pi,2\pi)$, so $\sin(\theta x_i)<0$. Every labeling is realized, so $\{2^{-1},\dots,2^{-n}\}$ is shattered for every $n$, and $\mathrm{VC}(\mathcal H_{\sin})=\infty$.

*Where uncountability enters.* The construction needs $\theta$ to carry $n$ independent bits, and it does so because a single real number has infinitely many binary digits; the parameter set $\mathbb{R}$ is uncountable and no finite description bounds it. The implication for practice: parameter counting bounds capacity only when parameters are quantized or otherwise constrained. Worked Example 4.16's device — treating each weight as $32$ bits and getting $\lvert\mathcal H\rvert=2^{32W}$ — is exactly the fix, and it is why that bound is finite (if useless) while the VC dimension of an unquantized class can be infinite at $W=1$.

**A12.** *(a)* With $\lambda=t/D$, $D:=\nu^2+bt$, we get $1-b\lambda=(D-bt)/D=\nu^2/D$, so
$$g(\lambda)=\frac{(t/D)^2\nu^2}{2\nu^2/D}-\frac{t^2}{D}=\frac{t^2}{2D}-\frac{t^2}{D}=-\frac{t^2}{2D}=-\frac{t^2}{2(\nu^2+bt)} . \checkmark$$

*(b)* Differentiate: $g'(\lambda)=\dfrac{\nu^2\lambda(2-b\lambda)}{2(1-b\lambda)^2}-t$. Setting $g'=0$ and clearing denominators,
$$\nu^2\lambda(2-b\lambda)=2t(1-b\lambda)^2\ \Longleftrightarrow\ bA\lambda^2-2A\lambda+2t=0,\qquad A:=\nu^2+2bt,$$
after collecting terms. The roots are $\lambda=\frac1b\big(1\pm\sqrt{1-2bt/A}\big)$, and only the minus root lies in $(0,1/b)$. Since $1-2bt/A=\nu^2/A$, this is $\lambda^\star=\frac1b\big(1-\sqrt{\nu^2/(\nu^2+2bt)}\big)$.

*(c)* At $\nu^2=b=t=1$: the guess is $\lambda=1/2$ with $g=-0.2500$; the optimum is $\lambda^\star=1-\sqrt{1/3}=0.4226$ with $g=-0.2679$. So the convenient choice loses about $7\%$ of the exponent.

Is it worth it? Almost never. The exact exponent does not simplify, it does not compose (you cannot sum it over independent blocks and keep a closed form), and $7\%$ in an exponent is invisible next to the factor of $25$ that the variance proxy costs elsewhere (Worked Example 4.8). The general lesson: constants in this subject are artifacts of choosing tractable algebra, so treat a factor of $2$ in a published bound as noise, and reserve your attention for the *rate* and for which hypotheses are being used.

**A13.** Each $X_i$ satisfies $\psi_{X_i}(\lambda)\le\frac{\lambda^2\nu_i^2}{2(1-b\lambda)}$ for $\lvert\lambda\rvert<1/b$. CGFs of independent variables add (Lemma 4.13(3)), so
$$\psi_{\sum_iX_i}(\lambda)=\sum_i\psi_{X_i}(\lambda)\le\frac{\lambda^2\sum_i\nu_i^2}{2(1-b\lambda)},$$
valid on the same interval $\lvert\lambda\rvert<1/b$. So the parameters are $\big(\sum_i\nu_i^2,\,b\big)$: the variance-like parameter adds, and $b$ — which controls the *width* of the interval where the condition holds — does not change, because it is set by the worst individual summand, not by how many there are.

For the i.i.d. case, $S_n$ has $(n\nu^2,b)$. For the mean, use $\psi_{\bar X_n}(\lambda)=\psi_{S_n}(\lambda/n)$:
$$\psi_{\bar X_n}(\lambda)\le\frac{(\lambda/n)^2n\nu^2}{2(1-b\lambda/n)}=\frac{\lambda^2(\nu^2/n)}{2\big(1-(b/n)\lambda\big)},$$
so the mean satisfies the Bernstein condition with $\big(\nu^2/n,\ b/n\big)$. **Both** parameters shrink like $1/n$, and that is the structural reason the second (linear) regime of Proposition 4.21 becomes irrelevant as $n$ grows: the regime boundary sits at $t\asymp\nu^2/b$, which for the mean is $\asymp\nu^2/b$ — unchanged — while the deviations we care about shrink like $n^{-1/2}$, so we end up in the quadratic regime.

**A14.** Setting the bound equal to $\delta$ with $L:=\log(1/\delta)$ gives $nt^2=2L(\sigma^2+Mt/3)$, i.e.
$$nt^2-\frac{2LM}{3}t-2L\sigma^2=0\quad\Longrightarrow\quad t=\frac{LM}{3n}+\sqrt{\Big(\frac{LM}{3n}\Big)^2+\frac{2L\sigma^2}{n}},$$
taking the positive root. Now apply $\sqrt{u+v}\le\sqrt u+\sqrt v$ with $u=(LM/3n)^2$ and $v=2L\sigma^2/n$:
$$t\ \le\ \frac{LM}{3n}+\frac{LM}{3n}+\sqrt{\frac{2\sigma^2L}{n}}=\sqrt{\frac{2\sigma^2\log(1/\delta)}{n}}+\frac{2M\log(1/\delta)}{3n} .$$

The first term is $O(n^{-1/2})$ and the second $O(n^{-1})$, so for large $n$ the variance term dominates and the radius behaves exactly like a Gaussian confidence interval with the *true* variance — which is the right behaviour, because that is what the CLT says is achievable. The $O(1/n)$ term is the price of boundedness rather than Gaussianity; it matters only when $n$ is small or $\sigma^2$ is tiny (as at $p=0.01$ in Worked Example 4.11, where it contributed the $0.000861$ we saw in Lab B3).

**A15.** *(a)* $D_i\in[-1,1]$ has range $2$, so Hoeffding gives a two-sided radius $2\sqrt{\log(2/\delta)/2m}=0.02716$ at $m=10^4$, $\delta=0.05$. Bounding the two risks separately and subtracting requires a union bound over two events, i.e. $\delta/2$ each, giving $2\sqrt{\log(4/\delta)/2m}=0.02961$. So *Hoeffding on the difference barely helps* — a $9\%$ improvement. This is worth knowing, because "use a paired test" is often stated as though the pairing itself were the win. It is not; the win comes next.

*(b)* $D_i=0$ whenever the models agree, so $\mathbb{E}[D_i^2]\le\mathbb{P}(\text{disagree})=q$ and hence $\operatorname{Var}(D_i)\le q$. Bernstein (Theorem 4.24) with $\sigma^2=q$, $M=1$ gives radius
$$\frac{L}{3m}+\sqrt{\Big(\frac{L}{3m}\Big)^2+\frac{2qL}{m}},\qquad L=\log(2/\delta).$$

*(c)* At $q=0.03$, $m=10^4$, $\delta=0.05$: radius $=0.00483$, i.e. $0.48$ percentage points. A $0.3$-point difference is therefore **still not detectable** — but only just, and the interval has shrunk by a factor of $2.8$ relative to the $\pm1.36$ points of §4.5. Solving for what would suffice: the difference becomes detectable once disagreement drops below about $q=1.1\%$.

The discrepancy with §4.5 is entirely the variance. The $\pm1.36$-point interval is about the *absolute* risk of one model, whose loss is a Bernoulli with variance up to $1/4$. The paired quantity has variance at most $q=0.03$, an eight-fold reduction, and Bernstein converts that into a radius reduction of $\sqrt{1/4\,\big/\,0.03}\approx2.9$. Two morals: always report paired comparisons on shared test sets, and note that the honest interval depends on how *similar* the models are — two near-identical checkpoints can be separated by far less data than two unrelated architectures.

**A16.** *(a)* By Proposition 4.49, $d\log(1/\varepsilon)\le\log N\le d\log(3/\varepsilon)$:

| $(d,\varepsilon)$ | $\log N$ range | $N$ (upper) |
|---|---|---|
| $(2,0.5)$ | $[1.39,\ 3.58]$ | $36$ |
| $(10,0.1)$ | $[23.03,\ 34.01]$ | $5.9\times10^{14}$ |
| $(100,0.5)$ | $[69.31,\ 179.18]$ | $6\times10^{77}$ |

*(b)* At $n=10^4$, using the upper bounds, $\sqrt{2\log N/n}$ equals $0.0268$, $0.0825$ and $0.1893$ respectively.

*(c)* The bound contains $\log N$, not $N$, because covering numbers enter through a union bound over the net (Massart's lemma). Taking the logarithm of $6\times10^{77}$ leaves $179$, and $\sqrt{2\cdot179/10^4}=0.19$ — a perfectly ordinary number. Your colleague is reacting to a count that never appears in any inequality.

**A17.** *(a)* Minimize $F(\varepsilon)=\varepsilon+\sqrt{2/(\varepsilon n)}$. Then $F'(\varepsilon)=1-\tfrac12\sqrt{2/n}\,\varepsilon^{-3/2}=0$ gives $\varepsilon^\star=\big(\tfrac12\sqrt{2/n}\big)^{2/3}=2^{-1/3}n^{-1/3}$. At the optimum the second term equals $2\varepsilon^\star$, so
$$F(\varepsilon^\star)=3\varepsilon^\star=3\cdot2^{-1/3}n^{-1/3}=2.381\,n^{-1/3}.$$
(Check against Worked Example 4.21: at $n=10^3$ this is $0.23811$, at $n=10^6$ it is $0.023811$ — exactly the tabulated values, and the fitted slope $-1/3$ is now explained rather than observed.)

*(b)* $\int_0^1\sqrt{\log N(\varepsilon)}\,d\varepsilon=\int_0^1\varepsilon^{-1/2}d\varepsilon=2$, so Theorem 4.51 gives $\hat{\mathfrak{R}}_n\le24/\sqrt n$: rate $n^{-1/2}$.

*(c)* Any improvement of (a) multiplies $n^{-1/3}$ by a constant. Since $n^{-1/3}/n^{-1/2}=n^{1/6}\to\infty$, no constant can save it: for large enough $n$ the chaining bound wins by an unbounded factor. The crossover of the two *numerical* bounds is where $3\cdot2^{-1/3}n^{-1/3}=24n^{-1/2}$, i.e. $n^{1/6}=8\cdot2^{1/3}=10.079$, so $n\approx1.05\times10^6$ — consistent with the table in Worked Example 4.21, where the ratio was $0.99$ at $n=10^6$.

*(d)* With $\log N(\varepsilon)=\varepsilon^{-2}$ the integrand is $\varepsilon^{-1}$ and $\int_0^1\varepsilon^{-1}d\varepsilon$ diverges logarithmically, so Theorem 4.51 gives nothing. Interpretation: the class is too rich at fine scales for uniform convergence at rate $n^{-1/2}$ — the fine-scale detail contributes at every scale without ever becoming negligible. Classes with $\varepsilon^{-2}$ entropy (e.g. all uniformly bounded measurable functions, or Lipschitz functions in high dimension) genuinely fail to admit $\sqrt{}$-rate uniform convergence; the divergence is informative, not an artifact.

**A18.** *(a)* Changing one $x_i$ moves the mean by at most $1/n$, so $c_i=1/n$ for all $i$ and $\sum_ic_i^2=1/n$. McDiarmid gives radius $\sqrt{\log(2/\delta)/2n}=0.04295$ at $n=1000$, $\delta=0.05$ — which is exactly Hoeffding, as it must be (Worked Example 4.13).

*(b)* $\Delta f=\max\lvert f(S)-f(S')\rvert=1/n$ over neighbouring samples, so the Laplace mechanism adds noise of scale $\Delta f/\epsilon=1/(n\epsilon)$, with standard deviation $\sqrt2/(n\epsilon)$.

*(c)* At $n=1000$, $\epsilon=1$: privacy noise s.d. $=\sqrt2/1000=0.001414$, sampling noise s.d. $=\sqrt{1/(12n)}=0.009129$. The privacy noise is $15.5\%$ of the statistical noise you already had. Setting the two equal, $\sqrt2/n=\sqrt{1/(12n)}$ gives $n=24$: beyond a couple of dozen samples the privacy noise is already the smaller of the two, and the ratio decays like $n^{-1/2}$ because privacy noise shrinks like $1/n$ while sampling noise shrinks like $1/\sqrt n$. **Privacy is asymptotically free for low-sensitivity queries** — a genuinely useful fact, and the reason means, counts and histograms are the easy case in differential privacy.

*(d)* For the median, $\Delta f=1$: moving one point can jump the median across the whole range (consider $n$ odd with points clustered at $0$ and $1$). Laplace noise of scale $1/\epsilon$ then swamps a quantity confined to $[0,1]$. The sensitivity is worst-case over *all* neighbouring datasets, and although the median is stable for typical data — Worked Example 4.14 measured a typical single-point shift of $0.015$ — worst-case sensitivity does not see that. This is the same gap between worst-case and average-case perturbation that made McDiarmid useless for the median and Efron–Stein useful, and the fixes are analogous: smoothed sensitivity, or the exponential mechanism, which pay for typical rather than worst-case stability.

**A19.** *(a)* Let $h(t):=\dfrac{t}{1+t^2}\varphi(t)$. Differentiate, using $\varphi'(t)=-t\varphi(t)$ and $\frac{d}{dt}\frac{t}{1+t^2}=\frac{1-t^2}{(1+t^2)^2}$:
$$h'(t)=\varphi(t)\,\frac{1-t^2}{(1+t^2)^2}-\frac{t^2}{1+t^2}\varphi(t)=\varphi(t)\,\frac{(1-t^2)-t^2(1+t^2)}{(1+t^2)^2}=\varphi(t)\,\frac{1-2t^2-t^4}{(1+t^2)^2}.$$
Hence $-h'(t)=\varphi(t)\,\dfrac{t^4+2t^2-1}{(1+t^2)^2}\le\varphi(t)$, because the fraction's numerator $t^4+2t^2-1$ is strictly less than its denominator $(1+t^2)^2=t^4+2t^2+1$. Both $h$ and $\bar\Phi$ vanish at $+\infty$, so integrating the pointwise inequality $-h'\le\varphi$ from $u$ to $\infty$ gives $h(u)\le\bar\Phi(u)$, which is the claim. (For the companion upper bound: $-\frac{d}{dt}[\varphi(t)/t]=\varphi(t)(1+1/t^2)\ge\varphi(t)$, so the same integration gives $\bar\Phi(u)\le\varphi(u)/u$.) The two bounds pin the Gaussian tail to within the factor $(1+u^2)/u^2\to1$: for large $u$, $\bar\Phi(u)=e^{-u^2/2(1+o(1))}$ *exactly* — the sub-Gaussian tail bound of §4.4 is attained, not merely approached.

*(b)* Here $\bar X_n-\mu\sim N(0,\sigma^2/n)$, so with $v:=t\sqrt n/\sigma$, the requirement $\mathbb{P}(\bar X_n-\mu\ge t)\le\delta$ reads $\bar\Phi(v)\le\delta$. By (a),
$$\delta\ \ge\ \frac{v}{1+v^2}\cdot\frac{e^{-v^2/2}}{\sqrt{2\pi}}\quad\Longrightarrow\quad \frac{v^2}{2}\ \ge\ \log\frac1\delta-\log\Big(\sqrt{2\pi}\,\frac{1+v^2}{v}\Big).$$
The subtracted term is $O(\log v)=O(\log\log(1/\delta))$ along the relevant scaling, so
$$v\ \ge\ \sqrt{2\log(1/\delta)}\,\Big(1-O\Big(\tfrac{\log\log(1/\delta)}{\log(1/\delta)}\Big)\Big),\qquad\text{i.e.}\qquad t\ \ge\ \big(1-o(1)\big)\,\sigma\sqrt{\frac{2\log(1/\delta)}{n}}.$$
*Reading.* This is a lower bound on what *any* valid statement about Gaussian sample means must concede: the radius $\sigma\sqrt{2\log(1/\delta)/n}$ is forced by the distribution itself, not by a proof technique. Hoeffding's radius for $[0,1]$-valued data is $\sqrt{\log(1/\delta)/2n}=\tfrac12\sqrt{2\log(1/\delta)/n}$ — the necessary form with $\sigma$ replaced by the proxy $\tfrac12$. So the *shape* is unimprovable in both $n$ and $\delta$; all that can ever be recovered is the constant $\tfrac12/\sigma$ (which is $\sqrt3$ for Uniform$[0,1]$ data, exactly the factor Worked Example 4.12 isolated and Bernstein removes when the variance is known). This is the promised replacement of "cited fact" by proof for the central case; Cramér's theorem generalizes the same conclusion from Gaussian to arbitrary i.i.d. summands with finite MGF.

*(c)* Take $u=\sigma\sqrt{\log N}$ and write $p:=\bar\Phi(u/\sigma)=\bar\Phi(\sqrt{\log N})$. By (a) with $v=\sqrt{\log N}$, so $\varphi(v)=N^{-1/2}/\sqrt{2\pi}$,
$$Np\ \ge\ N\cdot\frac{\sqrt{\log N}}{1+\log N}\cdot\frac{N^{-1/2}}{\sqrt{2\pi}}=\frac{\sqrt N\,\sqrt{\log N}}{(1+\log N)\sqrt{2\pi}},$$
which is increasing in $N$ and at $N=50$ equals $7.071\times1.978/(4.912\times2.5066)=1.14\ge1$. Independence enters now and only now: $\mathbb{P}(\max_iX_i<u)=(1-p)^N\le e^{-Np}\le e^{-1}$, so $\mathbb{P}(\max\ge u)\ge1-e^{-1}\ge0.632$. Split the expectation on this event and its complement. On $\{\max\ge u\}$ the maximum contributes at least $u$; on the complement it can be negative, but $-\max_iX_i\le-X_1$, so $\mathbb{E}[\max(0,-\max_iX_i)]\le\mathbb{E}[\max(0,-X_1)]=\sigma/\sqrt{2\pi}=0.399\sigma$. Hence
$$\mathbb{E}\big[\max_iX_i\big]\ \ge\ 0.632\,u-0.399\,\sigma\ \ge\ \Big(0.632-\frac{0.399}{\sqrt{\log 50}}\Big)\sigma\sqrt{\log N}\ \ge\ 0.43\,\sigma\sqrt{\log N}$$
for $N\ge50$, using $\sqrt{\log N}\ge\sqrt{\log 50}=1.978$. (The sharp asymptotic constant is $\sqrt2$: $\mathbb{E}\max=\sigma\sqrt{2\log N}(1+o(1))$, so A5's upper bound is exact to leading order.)

*The asymmetry, and what it means for Theorem 4.34.* A5's upper bound $\sigma\sqrt{2\log N}$ used only marginal sub-Gaussianity — the variables may be arbitrarily dependent. The lower bound above used independence at exactly one step, $(1-p)^N\le e^{-Np}$, which needs the failure events to multiply. So: the $\log N$ (equivalently, the $\log\lvert\mathcal H\rvert$ of Theorem 4.34) is *genuinely necessary* when the hypotheses' fluctuations are close to independent — genuinely different hypotheses, spread over the class — and is an *overcharge* when they are strongly correlated, which is exactly the situation of twenty nearby checkpoints in Common Confusion 4.3, and exactly the slack that Rademacher complexity (which sees the correlations through the geometry of the loss vectors) is able to recover. The union bound cannot distinguish these two situations; that is both its universality and its cost.

## Part B

**B1.** *(a) and (b).* One-sided radii for $\bar X_n-\tfrac12$ with $X_i\sim\mathrm{Uniform}[0,1]$, from $200{,}000$ replicates, seed $20240419$. The comparison bounds are: Chebyshev $\sqrt{\sigma^2/(n\delta)}$ with $\sigma^2=1/12$; Hoeffding $\sqrt{\log(1/\delta)/(2n)}$; CLT $z_{1-\delta}\sigma/\sqrt n$. Representative cell at $n=1000$, $\delta=0.05$:

| quantity | radius |
|---|---|
| empirical | $0.01502$ |
| CLT | $0.01502$ |
| Hoeffding | $0.03870$ |
| Chebyshev | $0.04082$ |

The CLT approximation is indistinguishable from the measurement at three decimals, which is the honest summary of Common Confusion 4.5: for a sum of well-behaved bounded variables at $n=1000$ the normal approximation is *accurate*, while Hoeffding is *valid*. Slopes on log-log axes are $-1/2$ for all of empirical, CLT, Hoeffding, Chebyshev.

*(c)* Hoeffding/empirical is $2.58$ at $\delta=0.05$ and essentially constant in $n$ — both scale as $n^{-1/2}$, so the ratio cannot depend on $n$ except through the CLT's own convergence. It does move with $\delta$: the ratio is
$$\frac{\sqrt{\log(1/\delta)/2n}}{z_{1-\delta}\sigma/\sqrt n}=\frac{\sqrt{6\log(1/\delta)}}{z_{1-\delta}},$$
which is $2.90$ at $\delta=0.1$, $2.58$ at $\delta=0.05$, $2.26$ at $\delta=0.01$. Worked Example 4.12's decomposition explains the value: $\sqrt3$ from the variance proxy ($1/4$ against the true $1/12$) times $\sqrt{2\log(1/\delta)}/z_{1-\delta}=1.49$ at $\delta=0.05$, and $\sqrt3\times1.49=2.58$. Hoeffding's penalty is *two constants*, not a rate loss, and it shrinks as you demand higher confidence — the $\sqrt{\log(1/\delta)}$ tail shape is the correct one.

*(d)* Markov applied to $\bar X_n$ gives $P(\bar X_n\ge\tfrac12+t)\le\tfrac{1/2}{1/2+t}$, so requiring this to be at most $\delta$ needs $t\ge\tfrac12(1/\delta-1)$: that is $4.5$ at $\delta=0.1$ and $49.5$ at $\delta=0.01$, **independent of $n$**. Markov uses only the mean; it does not know the variables are independent, bounded, or averaged. No amount of data helps a bound that never looks at $n$. This is the cleanest possible demonstration that the value of concentration inequalities lies in the hypotheses they exploit, not in the algebra.

**B2.** *(a)* $\sup_{\lVert w\rVert_2\le1}\frac1n\sum_i\sigma_i\langle w,x_i\rangle=\frac1n\sup_{\lVert w\rVert_2\le1}\langle w,\sum_i\sigma_ix_i\rangle=\frac1n\lVert\sum_i\sigma_ix_i\rVert_2$, by the definition of the dual norm (the maximizer is $w=\sum\sigma_ix_i/\lVert\sum\sigma_ix_i\rVert_2$). So the inner optimization is closed-form and only the average over $\sigma$ needs simulation.

*(b)* Averaging over $4000$ sign vectors, seed $11$, $n=200$:

| $d$ | estimate | $\sqrt{d/n}$ |
|---|---|---|
| $1$ | $0.05158$ | $0.07071$ |
| $4$ | $0.13281$ | $0.14142$ |
| $16$ | $0.27774$ | $0.28284$ |
| $64$ | $0.55709$ | $0.56569$ |
| $128$ | $0.79958$ | $0.80000$ |

and at $d=16$, for $n=50,100,200,400,800,1600$: $0.53201$, $0.38812$, $0.27843$, $0.19497$, $0.13890$, $0.09776$.

*(c)* The fitted exponent in $n$ is $-0.500$ and in $d$ is $+0.500$, confirming $\hat{\mathfrak{R}}_n\approx\sqrt{d/n}$. Proposition 4.41 with $B=1$ and $X=\max_i\lVert x_i\rVert_2\approx\sqrt d$ (plus Gaussian fluctuation) gives $X/\sqrt n\approx\sqrt{d/n}$ — the bound is tight up to the max-versus-typical-norm discrepancy, which is small because $\lVert x_i\rVert_2$ concentrates around $\sqrt d$.

*(d)* Every estimate lies below $\sqrt{d/n}$, and must:
$$\mathbb{E}_\sigma\Big\lVert\sum_i\sigma_ix_i\Big\rVert_2\ \le\ \sqrt{\mathbb{E}_\sigma\Big\lVert\sum_i\sigma_ix_i\Big\rVert_2^2}=\sqrt{\sum_i\lVert x_i\rVert_2^2}\approx\sqrt{nd},$$
where the inequality is Jensen for the concave function $\sqrt{\cdot}$ (or Cauchy–Schwarz), and the middle equality uses $\mathbb{E}[\sigma_i\sigma_j]=\delta_{ij}$. So the measured quantity is the mean of a square root and the target is the square root of a mean; the gap is the Jensen gap, which is controlled by the variance of $\lVert\sum\sigma_ix_i\rVert_2^2$ relative to its mean. That relative variance shrinks as $n$ (or $d$) grows by concentration, which is exactly the pattern in the table: the ratio is $0.73$ at $d=1$ and $0.9995$ at $d=128$.

**B3.** *(a) and (b).* At $p=0.01$, $n=10^4$, $\delta=0.05$, one-sided radii:

| method | radius |
|---|---|
| Hoeffding | $0.012239$ |
| Bernstein | $0.002635$ |
| exact Binomial quantile | $0.0017$ |

Hoeffding is $7.2$ times too wide, Bernstein $1.55$ times. Comparing tail probabilities at a fixed $t$ makes the point even more starkly: at $t=0.005$ the exact tail is $1.657\times10^{-6}$, Bernstein gives $2.026\times10^{-5}$, and Hoeffding gives $0.6065$ — five orders of magnitude of slack, because Hoeffding believes the variance could be $0.25$ when it is $0.0099$.

*(c)* The ratio Hoeffding/Bernstein decreases toward $1$ as $p\to1/2$ and grows as $p\to0$. This is precisely Worked Example 4.8: the ratio of the true CGF to the Hoeffding proxy $\lambda^2/8$ has maximum $1.0000$ at $p=0.5$, $0.7282$ at $p=0.1$, $0.4184$ at $p=0.01$. Where the proxy is exact (fair coin) the two inequalities agree and Bernstein buys nothing; where the proxy is $25$-fold pessimistic the gain is large. The lesson is not "always use Bernstein" but "the gain equals the pessimism of the variance proxy," which you can estimate before doing any work.

*(d)* Empirical Bernstein (Proposition 4.26, with $\hat V_n\approx p(1-p)=0.0099$ estimated from one sample) gives $\sqrt{2\hat V_n\log(2/\delta)/n}+\tfrac{7\log(2/\delta)}{3(n-1)}=0.002703+0.000861=0.003563$, about $35\%$ wider than the oracle Bernstein radius $0.002635$ — and still $3.4$ times narrower than Hoeffding. The extra cost is the additive term $0.000861$, which is what you pay for not knowing the variance. That is a small price, and it is the version to use in practice, since a bound that requires the unknown $\sigma^2$ is not a bound you can report.

**B4.** *(a)* $W=784\cdot64+64+64\cdot10+10=50{,}176+64+640+10=50{,}890$.

*(b)* $\log\lvert\mathcal H\rvert=32W\log2=1{,}128{,}776$. Theorem 4.34 gives
$$\varepsilon=\sqrt{\frac{\log\lvert\mathcal H\rvert+\log(2/\delta)}{2n}}=\sqrt{\frac{1{,}128{,}776+3.689}{120{,}000}}=3.067 .$$

*(c)* With $d=W\log_2W=795{,}670$, Theorem 4.46 gives
$$\varepsilon=2\sqrt{\frac{2d\log(n+1)}{n}}+\sqrt{\frac{\log(1/\delta)}{2n}}=34.169 .$$
The classical form fails outright: $2en/d=2e\cdot60{,}000/795{,}670=0.410<1$, so $\log(2en/d)<0$ and the expression under the square root can be negative — the formula returns a NaN. This is not a bug to patch. The classical VC bound is derived under $n\ge d$; when $d>n$ the growth function is $2^n$, every labeling is achievable, and uniform convergence genuinely provides no information. The undefined value is the theory reporting its own inapplicability.

*(d)* Setting each bound to $0.1$: the finite-class bound needs $n=(1{,}128{,}776+\log40)/(2\cdot0.01)=5.64\times10^7$, which is $941$ MNIST training sets. The VC bound needs $n\approx1.49\times10^{10}$ — about $248{,}500$ MNIST training sets, or roughly two thousand images for every person in a large country.

*(e)* Such a network trained with standard settings reaches about $2\%$ test error with a train–test gap near $0.02$. Ratios: $3.067/0.02\approx153$ for parameter counting and $34.17/0.02\approx1{,}700$ for VC.

What the ratio tells you: how far a distribution-free, algorithm-free, capacity-based guarantee is from reality — three orders of magnitude, so no improvement in constants can rescue it. What it does *not* tell you: whether the gap is looseness or a wrong quantity. §4.11's random-label argument settles that; B4 alone cannot.

## Part C

**C1.** *Setup.* Let $Z_1,\dots,Z_n$ be i.i.d. $\sim P$ on $\mathcal Z$, realized on a probability space $(\Omega,\mathcal A,\mathbb P)$ carrying, in addition, a *ghost sample* $Z_1',\dots,Z_n'$ i.i.d. $\sim P$ independent of the first, and Rademacher signs $\sigma_1,\dots,\sigma_n$ independent of both. Concretely, work on the product space $(\mathcal Z^n\times\mathcal Z^n\times\{\pm1\}^n,\ P^{\otimes n}\otimes P^{\otimes n}\otimes\mathrm{Unif})$; the existence of this product measure is Chapter 2's product-measure construction, and it is where "introduce a ghost sample" acquires meaning. Write $\hat P_nf=\frac1n\sum_if(Z_i)$ and $\hat P_n'f=\frac1n\sum_if(Z_i')$. Assume $\mathbb{E}\sup_{f\in\mathcal F}\lvert\hat P_nf-Pf\rvert<\infty$ and that the supremum is measurable (e.g. $\mathcal F$ countable or separable in the relevant sense — Ch. 1 §1.8.3).

*Proof.* For each fixed $f$, $Pf=\mathbb{E}[\hat P_n'f\mid Z_{1:n}]=\mathbb{E}[\hat P_n'f]$, using that the ghost sample is independent of $Z_{1:n}$ and identically distributed. Hence
$$\mathbb{E}\sup_{f}\big(Pf-\hat P_nf\big)=\mathbb{E}\sup_f\ \mathbb{E}\big[\hat P_n'f-\hat P_nf\,\big\vert\,Z_{1:n}\big]\ \le\ \mathbb{E}\,\mathbb{E}\Big[\sup_f\big(\hat P_n'f-\hat P_nf\big)\,\Big\vert\,Z_{1:n}\Big]=\mathbb{E}\sup_f\big(\hat P_n'f-\hat P_nf\big),$$
where the inequality is $\sup\mathbb{E}\le\mathbb{E}\sup$ — valid in this direction because for each $f$, $\mathbb{E}[\cdot]\le\mathbb{E}[\sup_g\cdot]$, so the bound holds uniformly and hence for the supremum. (The reverse would be false; this is the only place the direction of the inequality is at issue.) Then
$$\hat P_n'f-\hat P_nf=\frac1n\sum_{i=1}^n\big(f(Z_i')-f(Z_i)\big),$$
and here is the key distributional fact: for any fixed sign vector $s\in\{\pm1\}^n$, the vector $\big(s_i(f(Z_i')-f(Z_i))\big)_{i\le n}$ has the same joint law as $\big(f(Z_i')-f(Z_i)\big)_{i\le n}$, because the pairs $(Z_i,Z_i')$ are **independent across $i$** and each pair is **exchangeable** (both coordinates are i.i.d. $P$), so swapping within any subset of coordinates preserves the law. Consequently the law of $\sup_f\frac1n\sum_i(f(Z_i')-f(Z_i))$ is unchanged if we insert $\sigma_i$, and averaging over $\sigma$,
$$\mathbb{E}\sup_f\big(\hat P_n'f-\hat P_nf\big)=\mathbb{E}\sup_f\frac1n\sum_i\sigma_i\big(f(Z_i')-f(Z_i)\big)\le\mathbb{E}\sup_f\frac1n\sum_i\sigma_if(Z_i')+\mathbb{E}\sup_f\frac1n\sum_i\big(-\sigma_i\big)f(Z_i)=2\,\mathfrak{R}_n(\mathcal F),$$
using subadditivity of the supremum and that $-\sigma$ has the same law as $\sigma$. $\blacksquare$

*Where each hypothesis is used.* Identical distribution: to write $Pf=\mathbb{E}\hat P_n'f$. Independence of the ghost sample from the original: same step, and to condition. Independence *across* $i$: only in the sign-insertion step — and this is the answer to the final question. If the $Z_i$ were merely uncorrelated, the joint law of $(Z_{1:n},Z'_{1:n})$ would not factor across coordinates, so flipping signs in a *subset* of coordinates need not preserve the joint law, and the symmetrization identity fails. (Uncorrelatedness is a statement about second moments of individual pairs; symmetrization needs invariance of an $n$-dimensional joint law.) Everything else in the proof survives dependence.

*Source of the factor $2$.* Splitting the symmetrized difference into two terms, each of which is a full Rademacher average. It is not an artifact of sloppiness — the ghost sample doubles the noise, and one factor is genuinely the price of comparing an empirical average to a population mean via a second independent empirical average.

**C2.** *Derivation.* Let $\mathcal H=\{h_1,\dots,h_N\}$ and let the loss $\ell(h,z)\in[0,1]$. Fix $j$. The variables $\ell(h_j,Z_1),\dots,\ell(h_j,Z_n)$ are i.i.d., bounded in $[0,1]$, with mean $R(h_j)$, so Theorem 4.23 (Hoeffding) gives, for each $t>0$,
$$P\big(\lvert\hat R_n(h_j)-R(h_j)\rvert\ge t\big)\le2e^{-2nt^2}.$$
Union bound (Proposition 1.8(2)) over $j=1,\dots,N$:
$$P\Big(\max_{j\le N}\lvert\hat R_n(h_j)-R(h_j)\rvert\ge t\Big)\le2Ne^{-2nt^2}.$$
Set the right side to $\delta$ and solve: $t=\sqrt{\dfrac{\log N+\log(2/\delta)}{2n}}$. So with probability at least $1-\delta$, for all $h\in\mathcal H$ simultaneously, $\lvert\hat R_n(h)-R(h)\rvert\le\varepsilon$ with that $\varepsilon$ — Theorem 4.34.

*(a) The bottleneck.* It is the union bound. Hoeffding's application is per-hypothesis and needs nothing about $\mathcal H$; it survives verbatim for infinite classes. The inversion is algebra. But the union bound produces a factor $N$, and $\log N=\infty$ for infinite $\mathcal H$. Note precisely what breaks: not the inequality $P(\cup A_j)\le\sum P(A_j)$, which is true for countably many events, but its *usefulness* — and for uncountable $\mathcal H$ even countable subadditivity is unavailable, so one must additionally worry whether $\sup_h\lvert\hat R_n(h)-R(h)\rvert$ is measurable at all (Ch. 1 §1.8.3).

*(b) Not a constants problem.* At $n=1000$, $t=0.05$, the union bound's total is $2Ne^{-2nt^2}=2Ne^{-5}$, which reaches $1$ at
$$N=\tfrac12e^{5}=74.2 .$$
So beyond $74$ hypotheses the bound says "the probability is at most something $\ge1$" — no information. Improving the constant $2$ in Hoeffding to, say, $1$ would move the threshold to $148$. Doubling $n$ moves it to $e^{10}/2=11{,}013$. Neither is remotely enough for a class parametrized by real vectors, where $N$ is infinite: the failure is qualitative, and any repair must stop enumerating hypotheses altogether.

*(c) The two repairs.* Rademacher complexity replaces the union bound with a single concentration step applied to the supremum itself (McDiarmid, constants $1/n$) plus symmetrization, so that the price becomes "how well can this class fit random signs on *these* $n$ points" — a quantity that is finite for infinite classes and measurable in the same breath. Covering numbers replace it with a union bound over a *finite $\varepsilon$-net* plus an explicit approximation error $\varepsilon$, so the enumerated set is a proxy of controllable size rather than the class itself; chaining then iterates this over all scales.

**C3.** *(a) Derivation.* Write $g(x)=\sum_{j\le m}v_j\rho(\langle w_j,x\rangle)$ and fix $\sigma$. The objective $\sum_jv_jc_j$, with $c_j:=\sum_i\sigma_i\rho(\langle w_j,x_i\rangle)$, is linear in $v$ over an $\ell_1$ ball, so its supremum is attained at a vertex: $\sup_{\lVert v\rVert_1\le B_2}\sum_jv_jc_j=B_2\max_j\lvert c_j\rvert$. This is where width-independence comes from — the $m$ inner problems are identical and unconstrained relative to each other, so taking $\sup$ over all the $w_j$ collapses them to a single problem:
$$\hat{\mathfrak{R}}_n(\mathcal F)\ \le\ \frac{B_2}{n}\,\mathbb{E}_\sigma\sup_{\lVert w\rVert_2\le B_1}\Big\lvert\sum_i\sigma_i\rho\big(\langle w,x_i\rangle\big)\Big\rvert .$$
Do not wave at the absolute value; removing it is where the factor $2$ is earned, and Proposition 4.42's proof does it properly. Set $G=\{x\mapsto\rho(\langle w,x\rangle):\lVert w\rVert_2\le B_1\}$ and $T(\sigma)=\sup_{g\in G}\sum_i\sigma_ig(x_i)$. Because $w=0$ is admissible and $\rho(0)=0$, the class contains the zero function, so $T(\sigma)\ge0$ for every $\sigma$; hence $\sup_g\lvert\sum_i\sigma_ig(x_i)\rvert=\max(T(\sigma),T(-\sigma))\le T(\sigma)+T(-\sigma)$, and since $-\sigma$ has the same law as $\sigma$, taking $\mathbb{E}_\sigma$ gives $\mathbb{E}_\sigma\sup_g\lvert\cdot\rvert\le2\,\mathbb{E}_\sigma T(\sigma)$. (The frequently-quoted shortcut "the Rademacher complexity of a convex hull equals that of its generators" is true, but on its own it does not license dropping the absolute value; the nonnegativity of $T$ is what does.) Now $\rho$ is $1$-Lipschitz with $\rho(0)=0$, so Talagrand's contraction lemma (Lemma 4.40) strips it off without cost:
$$\hat{\mathfrak{R}}_n\big(\{\rho(\langle w,\cdot\rangle)\}\big)\le\hat{\mathfrak{R}}_n\big(\{\langle w,\cdot\rangle:\lVert w\rVert_2\le B_1\}\big)=\frac{B_1}{n}\mathbb{E}_\sigma\Big\lVert\sum_i\sigma_ix_i\Big\rVert_2\le\frac{B_1}{n}\sqrt{\sum_i\lVert x_i\rVert^2}\le\frac{B_1X}{\sqrt n},$$
using the dual-norm identity, Jensen, and $\lVert x_i\rVert\le X$. Combining, $\hat{\mathfrak{R}}_n(\mathcal F)\le2B_1B_2X/\sqrt n$, and taking expectations gives the same for $\mathfrak{R}_n$. Positive homogeneity of the ReLU was used implicitly in normalizing $\lVert w_j\rVert\le B_1$ (one may rescale between layers), and **$m$ never appears**: the $\ell_1$ constraint on $v$ means adding hidden units cannot increase the bound.

*(b)* At $n=60{,}000$, $\delta=0.1$, $X=10$, the confidence term is $\sqrt{\log(2/\delta)/2n}=0.004996$ and the gap bound is $2\mathfrak{R}_n+0.004996$ with $\mathfrak{R}_n\le0.081650\,B_1B_2$:

| $(B_1,B_2)$ | $\mathfrak{R}_n$ bound | gap bound |
|---|---|---|
| $(1,1)$ | $0.08165$ | $0.16830$ |
| $(2,5)$ | $0.81650$ | $1.63799$ |
| $(5,10)$ | $4.08248$ | $8.16996$ |
| $(10,20)$ | $16.32993$ | $32.66486$ |

*(c)* Solving $0.163299\,B_1B_2+0.004996\le0.1$ gives $B_1B_2\le0.582$.

*(d) Loose, wrong quantity, or false?* Not false. Every step above is a theorem with hypotheses that the trained network satisfies (bounded inputs, finite norms, a $1$-Lipschitz activation); the conclusion holds with probability $1-\delta$. If you measured a gap of $0.5$, the bound of $32.7$ would be respected.

Loose, certainly — by orders of magnitude against a typical reported gap near $0.02$, with the caveat from §4.11 that the norms feeding (b) are assumed rather than measured. But looseness alone would be an invitation to sharpen constants, and that is not the situation. The decisive evidence is the *direction of travel*: as training proceeds, $B_1$ and $B_2$ grow (weight norms increase, often by large factors, under standard optimizers without aggressive regularization) while the test error falls. The bound therefore worsens monotonically along the trajectory on which generalization improves. A quantity anti-correlated with its target over the very trajectory of interest is not a loose estimate of that target; it is a measurement of something else. So: valid, loose, and measuring the wrong thing — with the third being the diagnosis that matters, and the reason §4.11 treats margin-normalized, compression-based and PAC-Bayes quantities (which are *not* constant across the random-label experiment and do not grow monotonically during training) as the more promising directions.

A final caution on (b): $X=10$ and the norm values are assumptions I supplied, not measurements. The defensible statement is the scaling $2B_1B_2X/\sqrt n$ together with the threshold from (c); anyone reporting a number here should report the measured norms alongside it, because the bound is linear in a product that varies by orders of magnitude across training runs.

### References and further reading

**Textbooks and monographs.**

- S. Boucheron, G. Lugosi and P. Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence*, Oxford University Press, 2013. The standard reference for everything in §4.2–§4.6. Chapter 2 for the Cramér–Chernoff method, Chapter 3 for Efron–Stein, Chapter 6 for the entropy method, Chapter 8 for Bennett and Bernstein.
- R. Vershynin, *High-Dimensional Probability: An Introduction with Applications in Data Science*, Cambridge University Press, 2018. The best first source for sub-Gaussian equivalences (Prop. 2.5.2), covering numbers (§4.2), and chaining (§8.1). Written with exactly this audience in mind.
- M. J. Wainwright, *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*, Cambridge University Press, 2019. Chapters 2 and 5 cover tail bounds and metric entropy with statistical applications; strong on the interaction between entropy and estimation rates.
- S. Shalev-Shwartz and S. Ben-David, *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press, 2014. The cleanest undergraduate treatment of PAC learning, VC dimension and Rademacher complexity; the source for the style of §4.7–§4.9.
- M. Mohri, A. Rostamizadeh and A. Talwalkar, *Foundations of Machine Learning*, 2nd ed., MIT Press, 2018. Complementary to the above, with careful statements of the contraction lemma and margin bounds.
- A. W. van der Vaart and J. A. Wellner, *Weak Convergence and Empirical Processes*, Springer, 1996. The definitive treatment of Dudley's entropy integral, bracketing entropy, and measurability of suprema.
- L. Devroye, L. Györfi and G. Lugosi, *A Probabilistic Theory of Pattern Recognition*, Springer, 1996. Classical VC theory in the context of classification, with sharp combinatorial arguments.
- M. Talagrand, *Upper and Lower Bounds for Stochastic Processes*, Springer, 2014. Generic chaining and the $\gamma_2$ functional; the successor to Dudley's bound.
- M. Ledoux and M. Talagrand, *Probability in Banach Spaces*, Springer, 1991. Original source for the contraction lemma (Thm. 4.12) used as Lemma 4.40.
- J. Matoušek, *Lectures on Discrete Geometry*, Springer, 2002. Radon's theorem and the combinatorial geometry behind VC dimensions of half-spaces.
- J. A. Tropp, *An Introduction to Matrix Concentration Inequalities*, Foundations and Trends in Machine Learning 8 (2015). The matrix Bernstein inequality mentioned at the end of §4.5; the natural next step after this chapter for anyone working with random matrices.

**Papers.**

- W. Hoeffding, "Probability inequalities for sums of bounded random variables," *JASA* 58 (1963), 13–30. The original, including the lemma of §4.4.
- G. Bennett, "Probability inequalities for the sum of independent random variables," *JASA* 57 (1962), 33–45.
- C. McDiarmid, "On the method of bounded differences," *Surveys in Combinatorics* (1989), 148–188.
- A. Maurer and M. Pontil, "Empirical Bernstein bounds and sample variance penalization," *COLT* 2009. Proposition 4.26.
- V. N. Vapnik and A. Ya. Chervonenkis, "On the uniform convergence of relative frequencies of events to their probabilities," *Theory Probab. Appl.* 16 (1971), 264–280.
- N. Sauer, "On the density of families of sets," *J. Combin. Theory A* 13 (1972), 145–147; S. Shelah, "A combinatorial problem," *Pacific J. Math.* 41 (1972), 247–261.
- P. L. Bartlett and S. Mendelson, "Rademacher and Gaussian complexities: risk bounds and structural results," *JMLR* 3 (2002), 463–482. The modern form of §4.8.
- O. Bousquet and A. Elisseeff, "Stability and generalization," *JMLR* 2 (2002), 499–526. Definition 4.31 and Proposition 4.32.
- V. Feldman and J. Vondrák, "High probability generalization bounds for uniformly stable algorithms with nearly optimal rate," *COLT* 2019. The high-probability upgrade of Proposition 4.32 discussed in §4.6.4.
- P. L. Bartlett, N. Harvey, C. Liaw and A. Mehrabian, "Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks," *JMLR* 20 (2019). Proposition 4.47(2).
- E. B. Baum and D. Haussler, "What size net gives valid generalization?" *Neural Computation* 1 (1989), 151–160; M. Karpinski and A. Macintyre, "Polynomial bounds for VC dimension of sigmoidal and general Pfaffian neural networks," *JCSS* 54 (1997); P. Koiran and E. D. Sontag, "Neural networks with quadratic VC dimension," *JCSS* 54 (1997), 190–198. Proposition 4.47(1) and (3): Karpinski–Macintyre supply the sigmoid upper bound, Koiran–Sontag the $\Omega(W^2)$ lower bound.
- C. Zhang, S. Bengio, M. Hardt, B. Recht and O. Vinyals, "Understanding deep learning requires rethinking generalization," *ICLR* 2017. The random-labels experiment of §4.11.
- P. L. Bartlett, D. J. Foster and M. Telgarsky, "Spectrally-normalized margin bounds for neural networks," *NeurIPS* 2017.
- S. Arora, R. Ge, B. Neyshabur and Y. Zhang, "Stronger generalization bounds for deep nets via a compression approach," *ICML* 2018.
- G. K. Dziugaite and D. M. Roy, "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data," *UAI* 2017.
- V. Nagarajan and J. Z. Kolter, "Uniform convergence may be unable to explain generalization in deep learning," *NeurIPS* 2019.
- C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," *Found. Trends Theor. Comput. Sci.* 9 (2014). Background for Worked Example 4.15.
