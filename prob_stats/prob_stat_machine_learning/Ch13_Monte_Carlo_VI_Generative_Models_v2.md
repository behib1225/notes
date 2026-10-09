# Chapter 13 — Monte Carlo, Variational Inference, and Generative Models

## Chapter 13 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 13.1 Monte Carlo estimation | Replace an integral by an average over samples; error is $O(n^{-1/2})$ with a constant that depends on the variance of the integrand, not on the dimension | Why every modern loss involving an expectation is estimated by sampling; how to put honest error bars on a reported evaluation number |
| 13.2 Importance sampling and variance reduction | Reweight samples from a proposal; then cut variance with control variates, antithetics, stratification, common random numbers, Rao–Blackwellization | Off-policy RL estimates, tighter ELBO gradients, variance-reduced policy gradients; diagnostics ($\widehat{\text{ESS}}$, $\hat k$) that tell you when a reweighted estimate is garbage |
| 13.3 Rejection sampling and its curse | Exact sampling by accept/reject; the acceptance rate decays exponentially in dimension | Why nobody samples a 10⁸-parameter posterior by rejection; the motivation for MCMC |
| 13.4 MCMC fundamentals | Transition kernels, invariance, detailed balance, ergodicity; Metropolis–Hastings and Gibbs | The workhorse for Bayesian neural networks, topic models, probabilistic programs (Stan, PyMC, NumPyro) |
| 13.5 Gradient-based MCMC | HMC, leapfrog, NUTS; Langevin (ULA/MALA) and SGLD | How gradients buy you dimension-scaling; the precise sense in which constant-step SGD looks like a sampler |
| 13.6 MCMC diagnostics | Autocorrelation, ESS, split-$\hat R$, divergences | How to audit somebody's (or your own) posterior sample and know when to distrust it |
| 13.7 Variational inference | The ELBO from three directions; mean-field, CAVI, SVI, natural gradients | The optimization-shaped alternative to sampling; the objective behind VAEs, variational dropout, and Bayes-by-backprop |
| 13.8 Variational autoencoders | Amortized inference, reparameterized gradients, the two ELBO terms, and the honest failure catalogue | Posterior collapse, blurry samples, $\beta$-VAE trade-offs, IWAE — diagnosed rather than folklore |
| 13.9 Score matching and diffusion | Scores dodge the normalizing constant; denoising score matching; the DDPM objective derived from a variational bound | Why diffusion models train stably where GANs did not, and how the measure-zero problem of Chapter 1 is dissolved |
| 13.10 Uncertainty and calibration | Aleatoric vs epistemic, calibration, ECE, proper scores, temperature scaling, conformal prediction | Deployable uncertainty: a finite-sample distribution-free coverage guarantee, and exactly what it does not promise |
| 13.11 The toolkit assembled | Every tool in this chapter mapped back to the chapter that built it | A map of the whole tutorial, plus an honest list of what is still open |

---

## 13.0 Orientation: the chapter where the tools get used

Twelve chapters ago we started with a sample space and a $\sigma$-algebra. Since then we have built a measure-theoretic integral, a theory of change of measure, a calculus for differentiating through expectations, Gaussian identities, conditional expectation, concentration inequalities, limit theorems, estimation theory, resampling, decision theory, divergences, and high-dimensional geometry. Each chapter ended with a promise that the machinery would pay for itself. This chapter is where the bill comes due.

The unifying problem is this. Modern probabilistic machine learning constantly needs quantities of the form
$$
\mathbb{E}_{P}[f] = \int f \, dP ,
$$
where $P$ is a distribution you can sample from but not integrate against in closed form (a posterior, a policy-induced trajectory distribution, a latent-variable model's marginal), and $f$ is something you can evaluate pointwise but not integrate (a loss, a reward, a log-likelihood). There are exactly three strategies, and this chapter develops all three: **sample and average** (Monte Carlo, §13.1–13.6), **replace the hard distribution by a tractable one and optimize** (variational inference, §13.7–13.8), or **learn the local geometry of the distribution instead of the distribution itself** (score matching and diffusion, §13.9). Section 13.10 then asks the question that decides whether any of this can be deployed: when a model reports a probability or an interval, is that number trustworthy?

You are assumed to bring the following, by name, without re-derivation. From Chapter 2: the Lebesgue integral and its convergence theorems, in particular the **Dominated Convergence Theorem** and the resulting licence to **differentiate under the expectation**; the **change-of-measure/importance-sampling identity** with its absolute-continuity requirement $P \ll Q$; the **reparameterization trick** and the **REINFORCE (score-function) estimator**, together with the fact that the score has mean zero; the **ELBO identity**; and **Radon–Nikodym derivatives** and the discipline of always naming a base measure. From Chapter 3: the **change-of-variables formula** with its Jacobian determinant, **normalizing flows**, and the **Gumbel-Softmax** relaxation. From Chapter 4: **concentration inequalities**, which we use for error bars that do not rely on asymptotics. From Chapter 5: **Gaussian identities**, especially conditioning and the product of two Gaussian densities. From Chapter 6: **conditional expectation** as an $L^2$ projection, **martingales**, and the **SGD convergence proof**. From Chapter 7: the **Central Limit Theorem** and **Slutsky's theorem**, which together justify Monte Carlo error bars. From Chapter 8: **EM**, which we will re-read as variational inference with an exact E-step, and **Rao–Blackwellization**. From Chapter 9: the **bootstrap** and the notion of **coverage**. From Chapter 10: **posteriors**, **predictive distributions**, the **Laplace approximation**, and **decision theory**. From Chapter 11: the **asymmetry of KL**, the **variational representations of $f$-divergences**, **Wasserstein distance**, and **rate–distortion**. From Chapter 12: **high-dimensional geometry** and the accompanying warning about over-reading concentration results. From Chapter 1, by number: Definition 1.16 (law/pushforward), Definition 1.17 (support), Definition 1.21 ($\sigma(X)$ as information), Definition 1.23 (independence), and the circle-generator computation showing that a pushforward of a low-dimensional latent has Lebesgue-null support and therefore no density.

Here is the punchline, stated in advance. **Sampling in high dimensions is not hard because integration is hard; it is hard because the mass concentrates on a thin, curved set that a blind proposal cannot find.** Every method in this chapter is a strategy for exploiting extra structure to find that set: importance sampling uses a hand-designed proposal, MCMC uses locality, HMC uses gradients, variational inference gives up exactness in exchange for an optimization problem, and diffusion models sidestep the problem entirely by smoothing the target until it is easy and then walking the smoothing backwards. The second punchline is about honesty: three of the four failures that opened Chapter 1 were failures of *representation* (no density, no common object, no gradient), and Chapter 2 fixed those; the failure that survives into modern practice is a failure of *validation*. A number produced by a sampler or a network is not a probability until something certifies it. Section 13.10 is about what can and cannot be certified.

A word on what this chapter does not do. It does not prove the ergodic theorem for general state spaces, nor Harris recurrence conditions, nor convergence rates for MCMC; those require a course in Markov chain theory on general state spaces, and I will state exactly what I am borrowing and where to find it. Everything else is proved.

---

## 13.1 Monte Carlo estimation

### 13.1.1 The estimator and why it works

The idea is the oldest one in computational statistics and it is embarrassingly simple: an integral is an average, so estimate it by an average.

Let $(\Omega, \mathcal{F}, P)$ be a probability space, let $X : \Omega \to \mathcal{X}$ be a random element with law $P_X$ (Definition 1.16), and let $f : \mathcal{X} \to \mathbb{R}$ be measurable with $\mathbb{E}|f(X)| < \infty$. Write $\theta := \mathbb{E}[f(X)] = \int f \, dP_X$.

**Definition 13.1 (Monte Carlo estimator).** Let $X_1, \dots, X_n$ be i.i.d. with law $P_X$. The *Monte Carlo estimator* of $\theta$ is
$$
\hat\theta_n := \frac{1}{n} \sum_{i=1}^n f(X_i).
$$

**Proposition 13.2 (unbiasedness, variance, and the $n^{-1/2}$ law).** Suppose $\mathbb{E}|f(X)| < \infty$. Then $\mathbb{E}[\hat\theta_n] = \theta$ and $\hat\theta_n \to_{as} \theta$. If in addition $\sigma^2 := \operatorname{Var}(f(X)) < \infty$, then
$$
\operatorname{Var}(\hat\theta_n) = \frac{\sigma^2}{n}, \qquad
\sqrt{\mathbb{E}\big[(\hat\theta_n - \theta)^2\big]} = \frac{\sigma}{\sqrt n}.
$$

*Proof.* Linearity of expectation gives unbiasedness; the strong law of large numbers (Ch. 7) gives almost-sure convergence, which needs only integrability. For the variance, independence makes the covariance terms vanish: $\operatorname{Var}(\sum_i f(X_i)) = n\sigma^2$, and dividing by $n^2$ gives $\sigma^2/n$. Since the estimator is unbiased, the mean squared error equals the variance. $\square$

Three features of this proposition are worth dwelling on, because together they explain why Monte Carlo took over computational statistics.

First, **the rate is $n^{-1/2}$ and nothing you do to $f$ or $\mathcal{X}$ changes the exponent.** Halving the error costs four times the samples. This is slow. Deterministic quadrature in one dimension is far better: Simpson's rule has error $O(n^{-4})$ for smooth integrands.

Second, **the constant is $\sigma$, and $\sigma$ does not know what dimension $\mathcal{X}$ has.** This is the crucial point. A product quadrature rule with $k$ nodes per axis in $d$ dimensions uses $k^d$ evaluations. Monte Carlo uses $\sigma^2/\varepsilon^2$ evaluations for error $\varepsilon$, whatever $d$ is. Somewhere around $d = 4$ or $5$ the crossover happens, and after that quadrature is not merely worse but unusable.

Third, **finite variance is a hypothesis, not a formality.** Without it, $\hat\theta_n$ still converges to $\theta$ (the law of large numbers only needs integrability) but the error bars below are meaningless, and in practice the estimator behaves erratically.

**Proposition 13.3 (Monte Carlo error bars).** Assume $\sigma^2 \in (0,\infty)$ and let $s_n^2 := \frac{1}{n-1}\sum_i (f(X_i) - \hat\theta_n)^2$. Then
$$
\frac{\sqrt n\,(\hat\theta_n - \theta)}{s_n} \to_d N(0,1),
\qquad\text{so}\qquad
P\Big( \theta \in \hat\theta_n \pm z_{1-\alpha/2}\, \tfrac{s_n}{\sqrt n} \Big) \to 1-\alpha .
$$

*Proof.* The CLT (Ch. 7) gives $\sqrt n (\hat\theta_n - \theta)/\sigma \to_d N(0,1)$. The sample variance is consistent, $s_n^2 \to_p \sigma^2$, by the law of large numbers applied to $f(X_i)^2$ (finite by assumption) plus the continuous mapping theorem. Slutsky's theorem (Ch. 7) lets us replace $\sigma$ by $s_n$ in the ratio. $\square$

This is the standard "$\pm$" you see in every experiments table, and it is worth being explicit about what it is: an *asymptotic* statement about the sampling variability of an average, conditional on the code being correct and the samples being independent. It says nothing about model misspecification, nothing about seed-to-seed variation in *training*, and — as we will see — it can be wildly optimistic when $\sigma^2 = \infty$.

**Worked Example 13.1 (a first estimate with an honest error bar).** Estimate $\theta = \mathbb{E}[e^U]$ for $U \sim \mathrm{Unif}(0,1)$. Exactly, $\theta = e - 1 = 1.718282$ and
$$
\sigma^2 = \mathbb{E}[e^{2U}] - \theta^2 = \frac{e^2-1}{2} - (e-1)^2 = 0.242036 .
$$
With $n = 100{,}000$ draws (NumPy PCG64, seed 0) the estimate is $\hat\theta_n = 1.717424$ with $s_n^2 = 0.241293$, giving a standard error $s_n/\sqrt n = 0.001553$ and a 95% interval $[1.714381, 1.720467]$. The truth is inside it, 0.55 standard errors from the centre. Note that the sample variance $0.241293$ estimated the true $0.242036$ to within $0.3\%$: with $10^5$ samples you learn the scale of your own uncertainty accurately, which is what makes Proposition 13.3 usable in practice.

**Worked Example 13.2 (the crossover with quadrature, concretely).** Suppose you want two-digit accuracy, $\varepsilon = 10^{-2}$, on an integral whose integrand has $\sigma \approx 1$. Monte Carlo needs about $\sigma^2/\varepsilon^2 = 10^4$ evaluations regardless of $d$. A tensor-product rule with a modest 10 nodes per axis needs $10^d$:

| $d$ | tensor grid, 10 nodes/axis | Monte Carlo for $\varepsilon = 10^{-2}$ |
|---|---|---|
| 1 | $10$ | $10^4$ |
| 3 | $10^3$ | $10^4$ |
| 5 | $10^5$ | $10^4$ |
| 10 | $10^{10}$ | $10^4$ |
| 20 | $10^{20}$ | $10^4$ |

At $d=3$ the grid wins comfortably. At $d = 5$ they are comparable. At $d = 20$ the grid needs more evaluations than there are seconds since the Big Bang ($\approx 4 \times 10^{17}$), while Monte Carlo has not noticed anything happened. Every expectation in a neural network — over a minibatch, over dropout masks, over latent variables, over a policy — lives at $d$ in the thousands or millions. That is the whole argument.

> **Common Confusion 13.1.** *"So Monte Carlo error is dimension-free. Dimension doesn't hurt sampling at all?"*
>
> The **rate** is dimension-free; the **constant** is not, and neither is your ability to draw the samples. Three things go wrong as $d$ grows, none of them visible in Proposition 13.2. (i) $\sigma^2 = \operatorname{Var}(f(X))$ often grows with $d$: for the importance weights of §13.2 it grows *exponentially* — Worked Example 13.7 computes it in closed form: a 20% per-coordinate scale mismatch multiplies $\mathbb{E}[w^2]$ by $\sim 10^6$ at $d = 1000$. (ii) Drawing $X_i \sim P_X$ may itself be impossible — that is what §13.3–13.5 are about; the "dimension-free" claim assumes exact i.i.d. draws, which for a posterior you do not have. (iii) The quantity you want may be an expectation of something rare, in which case $\sigma/\theta$ (the *relative* error) explodes even if $\sigma$ is small. Monte Carlo defeats the curse of dimensionality for *integration given samples*; it does not defeat it for *sampling*.

### 13.1.2 Relative error and rare events

For a probability $\theta = P(A) = \mathbb{E}[\mathbf{1}_A]$, the integrand is Bernoulli, so $\sigma^2 = \theta(1-\theta)$ and the *relative* standard error is
$$
\frac{\sigma/\sqrt n}{\theta} = \sqrt{\frac{1-\theta}{n\theta}} \approx \frac{1}{\sqrt{n\theta}} \quad\text{for small }\theta .
$$
To get 10% relative accuracy you need $n \approx 100/\theta$. For $\theta = 10^{-6}$ that is $10^8$ samples. This single formula explains why naive Monte Carlo is hopeless for tail probabilities — safety-critical failure rates, out-of-distribution detection thresholds, evaluation of rare harmful outputs from a language model — and why §13.2 exists.

**Worked Example 13.3 (naive Monte Carlo fails on a tail).** Let $Z \sim N(0,1)$ and $\theta = P(Z > 4) = 3.167124 \times 10^{-5}$. With $n = 10^5$ draws (seed 1), exactly 2 samples exceeded 4, giving $\hat\theta = 2.0 \times 10^{-5}$ — off by 37% — with an estimated relative standard error of 0.71. The estimator is technically unbiased and its interval is technically valid asymptotically, but with 2 effective observations we are in no sense in the asymptotic regime: the sampling distribution of $\hat\theta_n$ here is a scaled $\mathrm{Binomial}(10^5, \theta)$, which is Poisson-like with mean 3.17, and a Poisson with mean 3 is not a normal distribution. Section 13.2 will get four correct digits out of the same budget.

**Worked Example 13.4 (infinite variance silently destroys the error bars).** Let $f(u) = u^{-3/4}$ on $(0,1)$ with $U \sim \mathrm{Unif}(0,1)$. Then $\mathbb{E}[f(U)] = \int_0^1 u^{-3/4} du = 4$ is finite, but $\mathbb{E}[f(U)^2] = \int_0^1 u^{-3/2} du = \infty$. So $\hat\theta_n \to_{as} 4$, yet Proposition 13.3 does not apply. I ran 2000 independent replicates with $n = 1000$ each (seed 13) and computed the nominal 95% interval each time:

| integrand | true $\theta$ | empirical coverage of the nominal 95% interval | mean interval width |
|---|---|---|---|
| $e^u$ ($\sigma^2 = 0.242$) | $1.718282$ | $0.951$ | $0.0610$ |
| $u^{-3/4}$ ($\sigma^2 = \infty$) | $4$ | $0.648$ | $2.5908$ |

The first line is Proposition 13.3 working exactly as advertised. The second is what an invalid error bar looks like from the inside: the intervals are wide (mean width 2.59) and *still* miss the truth a third of the time, because occasional enormous samples inflate $s_n$ in some replicates and are absent in others. The lesson generalizes: **whenever you build an estimator by reweighting, check that the weights have finite second moment**, because heavy-tailed weights produce exactly this failure. Chapter 2 raised this caution for importance sampling; §13.2 gives it a diagnostic.

> **Self-check 13.1.**
> 1. You report a test accuracy of $0.812$ on $n = 2000$ examples. What is the standard error, and why is it legitimate to use Proposition 13.3 here even though "accuracy" is not an integral you set out to compute?
> 2. Your estimate of a failure probability is exactly $0$ after $10^6$ samples. What does the CLT interval say, and why is it useless? What would you report instead?
> 3. True or false: if $\hat\theta_n$ is unbiased and consistent then the interval $\hat\theta_n \pm 1.96 s_n/\sqrt n$ has asymptotic coverage $0.95$.
> 4. You double $n$. By what factor does the width of the interval shrink? How many samples buy you one extra decimal digit?

---

## 13.2 Importance sampling and variance reduction

### 13.2.1 The identity, restated with its hypothesis

Chapter 2 proved the change-of-measure identity: if $P \ll Q$ with Radon–Nikodym derivative $w = dP/dQ$, then for any $f \in L^1(P)$,
$$
\mathbb{E}_P[f] = \int f \, dP = \int f \, \frac{dP}{dQ} \, dQ = \mathbb{E}_Q[f\,w].
$$
Sampling $Y_1,\dots,Y_n \sim Q$ i.i.d. and averaging $f(Y_i) w(Y_i)$ therefore estimates $\mathbb{E}_P[f]$ without ever sampling $P$. When $P$ and $Q$ have densities $p$ and $q$ with respect to a common base measure $\mu$ (Chapter 2's discipline: always name the base measure), $w = p/q$ $\mu$-a.e. on $\{q > 0\}$.

**Proposition 13.4 (importance sampling: unbiasedness and variance).** Let $P \ll Q$, $w = dP/dQ$, and $\hat\theta_n^{\mathrm{IS}} := \frac1n \sum_i f(Y_i) w(Y_i)$ with $Y_i \sim Q$ i.i.d. Then $\hat\theta_n^{\mathrm{IS}}$ is unbiased for $\theta = \mathbb{E}_P[f]$, and
$$
\operatorname{Var}_Q\!\big(f w\big) = \mathbb{E}_Q\big[f^2 w^2\big] - \theta^2 = \mathbb{E}_P\big[f^2 w\big] - \theta^2 ,
$$
which is finite if and only if $\mathbb{E}_P[f^2 w] < \infty$.

*Proof.* Unbiasedness is the identity applied to $f$. For the variance, apply the identity to $f^2 w$: $\mathbb{E}_Q[f^2w^2] = \mathbb{E}_Q[(f^2 w) w] = \mathbb{E}_P[f^2 w]$. $\square$

Read the last expression carefully, because it is the whole story of importance sampling. The variance is controlled by $\mathbb{E}_P[f^2 w]$: it blows up when $w$ is large *in regions where $P$ puts mass*. In words: **the proposal must not be much lighter than the target anywhere the target lives.** A proposal with lighter tails than the target produces enormous rare weights, hence heavy-tailed $fw$, hence (Worked Example 13.4) an estimator whose error bars lie.

The requirement $P \ll Q$ is not a technicality either. If $P(A) > 0$ but $Q(A) = 0$, samples from $Q$ never visit $A$ and there is no reweighting that can repair the omission — the estimator is biased by $\int_A f \, dP$ and no amount of data reveals the problem. This is precisely the failure mode of off-policy evaluation in reinforcement learning when the behaviour policy assigns zero probability to actions the target policy would take.

**Proposition 13.5 (optimal proposal).** Among all $Q$ with $P \ll Q$, the variance of $\hat\theta_n^{\mathrm{IS}}$ is minimized by
$$
q^\star(x) \;\propto\; |f(x)|\, p(x),
$$
and if $f \ge 0$ the minimal variance is zero.

*Proof.* Write $q$ for the density of $Q$. We minimize $\mathbb{E}_Q[f^2w^2] = \int \frac{f^2p^2}{q}\,d\mu$ subject to $\int q \, d\mu = 1$, $q \ge 0$. By Cauchy–Schwarz,
$$
\Big(\int |f| p \, d\mu\Big)^2 = \Big(\int \frac{|f| p}{\sqrt q}\sqrt q \, d\mu\Big)^2 \le \int \frac{f^2p^2}{q} d\mu \cdot \int q \, d\mu = \mathbb{E}_Q[f^2w^2],
$$
with equality iff $|f|p/\sqrt q \propto \sqrt q$, i.e. $q \propto |f| p$. So the lower bound $(\mathbb{E}_P|f|)^2$ is attained exactly at $q^\star$. If $f \ge 0$ then $\mathbb{E}_P|f| = \theta$ and the second moment equals $\theta^2$, so the variance is zero. $\square$

The zero-variance solution is a joke with a serious point: to use $q^\star$ you must know its normalizing constant $\int |f| p$, which for $f \ge 0$ *is* the answer. But the form of $q^\star$ is genuinely useful guidance — put the proposal where $|f|p$ is big, not where $p$ alone is big. For a tail probability, $f = \mathbf{1}_{\{x > 4\}}$, so the optimal proposal lives entirely in the tail.

**Worked Example 13.5 (importance sampling rescues the tail).** Estimate $\theta = P(Z > 4) = 3.167124\times10^{-5}$ for $Z \sim N(0,1)$ using the shifted proposal $Q = N(4,1)$, so $w(y) = \varphi(y)/\varphi(y-4) = e^{-4y + 8}$. With $n = 10^5$ (seed 1):

| method | estimate | standard error | relative standard error |
|---|---|---|---|
| naive | $2.0\times10^{-5}$ | — (2 hits) | $\approx 0.71$ |
| IS with $N(4,1)$ | $3.161604\times10^{-5}$ | $2.12\times10^{-7}$ | $0.0067$ |

The relative error improved by a factor of about 100, i.e. a variance improvement of about $10^4$: the same accuracy would have required $10^9$ naive samples. Why it works: every proposal draw lands near the region of interest, and roughly half of them exceed 4, so the effective number of informative samples is $\sim 5\times10^4$ rather than 2.

### 13.2.2 Diagnostics: effective sample size and $\hat k$

Weights that are nearly constant are harmless; weights dominated by a handful of huge values are fatal. Two diagnostics quantify this.

**Definition 13.6 (effective sample size and the Pareto shape diagnostic).** Given weights $w_i = w(Y_i) > 0$, the *effective sample size* is
$$
\widehat{\mathrm{ESS}} := \frac{\big(\sum_{i} w_i\big)^2}{\sum_i w_i^2}
= \frac{n}{1 + \widehat{\mathrm{cv}}^2}, \qquad
\widehat{\mathrm{cv}}^2 := \frac{\frac1n\sum_i (w_i - \bar w)^2}{\bar w^2}.
$$
The *Pareto shape diagnostic* $\hat k$ is the maximum-likelihood-type estimate of the shape parameter $k$ of a generalized Pareto distribution fitted to the largest weights. A weight distribution with shape $k$ has finite moments of order $< 1/k$; so $\hat k < 1/2$ indicates finite variance, and $\hat k > 0.7$ is the conventional red flag.

$\widehat{\mathrm{ESS}}$ is exactly $n$ when all weights are equal and drops toward 1 when one weight dominates; the identity $\widehat{\mathrm{ESS}} = n/(1+\widehat{\mathrm{cv}}^2)$ is a two-line algebraic rearrangement and is worth checking once by hand. Note that $\widehat{\mathrm{ESS}}$ measures the mismatch between $Q$ and $P$ *globally*, ignoring $f$.

**Worked Example 13.6 (the diagnostics, validated and then applied).** First, does $\hat k$ work? I generated 20,000 draws from a generalized Pareto with known shape and ran the Zhang–Stephens estimator (5 independent repeats each, seed 7):

| true $k$ | mean $\hat k$ over 5 repeats |
|---|---|
| 0.2 | 0.202 |
| 0.5 | 0.505 |
| 0.7 | 0.640 |
| 1.0 | 0.970 |

Good enough to act on. Now apply both diagnostics to the two proposals below, each with $n = 10^5$ (seed 1), where the target is $N(0,1)$:

| proposal | $\widehat{\mathrm{ESS}}$ | $\hat k$ | verdict |
|---|---|---|---|
| $N(4,1)$ (for the tail problem) | $51.6$ | $1.34$ | terrible *as a global proposal for $N(0,1)$*, excellent for $f=\mathbf{1}_{\{x>4\}}$ |
| $N(0, 0.5^2)$ (too narrow) | $7647$ | $0.76$ | looks fine, is not: $\operatorname{Var}(w) = \infty$ |

Both rows are instructive. The $N(4,1)$ row shows that a tiny global ESS is *not* a bug when you have deliberately tilted the proposal toward a rare event; the diagnostics describe $Q$ versus $P$, not $Q$ versus $fP$. The $N(0,0.5^2)$ row is the genuinely dangerous case: the sample ESS is a healthy-looking 7647, but the exact second moment is
$$
\mathbb{E}_Q[w^2] = \int \frac{p^2}{q} \, dx \propto \int \exp\!\Big(-x^2 + \frac{x^2}{2 \cdot 0.25}\Big)\, dx = \int e^{+x^2}\, dx = \infty ,
$$
because the exponent is $+x^2$ at large $|x|$. The ESS did not catch it (with $10^5$ draws you rarely see the weights that carry the infinite second moment); $\hat k = 0.76 > 0.7$ did. This is why the Pareto diagnostic exists: **ESS is an estimate of a quantity that may not exist, while $\hat k$ estimates whether it exists.**

> **Common Confusion 13.2.** *"ESS = 7647 out of 100,000 means I have 7647 good samples, right?"*
>
> Not quite, in two ways. First, $\widehat{\mathrm{ESS}}$ is itself computed from the weights you happened to draw; if the weight distribution is heavy-tailed, $\widehat{\mathrm{ESS}}$ is unstable and typically optimistic, exactly as in the table above. Second, ESS is $f$-free. The correct statement is: the variance of the *self-normalized* estimator of $\mathbb{E}_P[f]$ is approximately $\operatorname{Var}_P(f)/\widehat{\mathrm{ESS}}$ *when the weights and $f$ are roughly uncorrelated*. If your $f$ happens to be supported where the weights are large (a tail probability with a tilted proposal), the effective sample size *for that $f$* can be far larger than the global ESS. Report both, and when in doubt estimate the variance of your actual estimator directly by replication.

**Worked Example 13.7 (weight degeneracy grows exponentially in dimension — computed exactly).** Common Confusion 13.1 claimed that the variance of importance weights can grow *exponentially* with dimension. Here is the cleanest case, and it needs no simulation because everything is a Gaussian integral. Target $P = N(0, I_d)$; proposal $Q = N(0, \sigma^2 I_d)$ — the right location and the right shape, merely the wrong scale, by the *same* modest factor in every coordinate. The weight factorizes, $w(y) = \prod_{j=1}^d w_1(y_j)$, and completing the square in one coordinate gives the per-coordinate second moment
$$
\rho(\sigma^2) := \mathbb{E}_Q\big[w_1^2\big] = \int \frac{\varphi(y)^2}{\varphi_\sigma(y)}\,dy
= \frac{\sigma^2}{\sqrt{2\sigma^2 - 1}} \qquad \text{for } \sigma^2 > \tfrac12, \quad = \infty \text{ otherwise},
$$
which is $\ge 1$ with equality only at $\sigma^2 = 1$. By independence across coordinates (Definition 1.23), $\mathbb{E}_Q[w^2] = \rho(\sigma^2)^d$: a fixed per-coordinate mismatch is *raised to the power $d$*. Since $\mathbb{E}_Q[w] = 1$, the population analogue of Definition 13.6's effective-sample-size fraction is $(\mathbb{E}_Q w)^2/\mathbb{E}_Q[w^2] = \rho^{-d}$. Exact values:

| $\sigma^2$ | $\rho(\sigma^2)$ | $\mathbb{E}_Q[w^2]$, $d=10$ | $d=100$ | $d=1000$ | ESS fraction at $d=1000$ |
|---|---|---|---|---|---|
| 1.2 | 1.014185 | 1.151 | 4.09 | $1.31\times10^{6}$ | $7.6\times10^{-7}$ |
| 2.0 | 1.154701 | 4.214 | $1.77\times10^{6}$ | $2.95\times10^{62}$ | $3.4\times10^{-63}$ |

Read the first row slowly: a 20% variance mismatch per coordinate — utterly invisible in any one-dimensional marginal plot, and giving a per-coordinate ESS fraction of 0.986 — costs a factor of about $10^6$ in effective sample size by $d = 1000$. The mechanism is Chapter 12's concentration, run in reverse. $\log w$ is a sum of $d$ i.i.d. terms, so it concentrates at its mean $-d\,D_{\mathrm{KL}}(Q_1\|P_1)$ with fluctuations of order $\sqrt d$; but the weight is the *exponential* of that sum, and the exponential of a quantity with $\sqrt d$ fluctuations is dominated by its rare upper tail. Almost every weight you draw is exponentially small, and almost all of the expectation sits in weights you will almost never see — which is also why the *empirical* $\widehat{\mathrm{ESS}}$ in this regime is typically a wild overestimate, exactly the failure mode Worked Example 13.6 exhibited. Note finally the threshold: for $\sigma^2 < 1/2$ the second moment is infinite in every dimension including $d=1$; Worked Example 13.6's dangerous $N(0, 0.5^2)$ proposal, with $\sigma^2 = 0.25$, sat on the wrong side of exactly this line. The two lessons compose: too-narrow proposals fail catastrophically in any dimension, and even generously wide, well-located proposals fail exponentially in high dimension. Hand-designed proposals do not scale; that is the second half of the argument for MCMC (§13.3 supplies the first).

### 13.2.3 Self-normalized importance sampling and its bias

In Bayesian practice you know the target only up to a constant: $p(\theta \mid x) = \tilde p(\theta)/Z$ with $\tilde p = p(x\mid\theta)p(\theta)$ computable and $Z$ unknown. Then $w \propto \tilde p / q$ is known only up to the same constant, and one uses the *self-normalized* estimator
$$
\hat\theta^{\mathrm{SN}}_n := \frac{\sum_{i=1}^n \tilde w_i f(Y_i)}{\sum_{i=1}^n \tilde w_i}, \qquad \tilde w_i = \tilde p(Y_i)/q(Y_i).
$$

**Proposition 13.7 (SNIS: consistency and $O(1/n)$ bias).** Assume $P \ll Q$, $\mathbb{E}_Q[\tilde w] < \infty$, $\mathbb{E}_Q[\tilde w |f|] < \infty$. Then $\hat\theta_n^{\mathrm{SN}} \to_{as} \theta$. If moreover $\operatorname{Var}_Q(\tilde w) < \infty$ and $\operatorname{Var}_Q(\tilde w f)<\infty$, then
$$
\mathbb{E}\big[\hat\theta^{\mathrm{SN}}_n\big] - \theta = -\frac{1}{n}\,\frac{\operatorname{Cov}_Q(\tilde w f, \tilde w)-\theta\operatorname{Var}_Q(\tilde w)}{\big(\mathbb{E}_Q \tilde w\big)^{2}} + o(1/n).
$$

*Proof sketch (the honest version).* Write $A_n = \frac1n\sum \tilde w_i f(Y_i)$, $B_n = \frac1n \sum \tilde w_i$, with $\mathbb{E}A_n = a = \theta\,\mathbb{E}_Q\tilde w$ and $\mathbb{E}B_n = b = \mathbb{E}_Q \tilde w$. The strong law gives $A_n \to a$, $B_n\to b$ a.s. and $b>0$, so $A_n/B_n \to a/b = \theta$ a.s. For the bias, expand $g(A,B) = A/B$ to second order around $(a,b)$: $g \approx \theta + \frac{A_n - a}{b} - \frac{a(B_n-b)}{b^2} - \frac{(A_n-a)(B_n-b)}{b^2} + \frac{a (B_n-b)^2}{b^3}$. The linear terms have mean zero; taking expectations of the quadratic terms gives $-\operatorname{Cov}(A_n,B_n)/b^2 + a\operatorname{Var}(B_n)/b^3$, and each of these is $1/n$ times the corresponding population quantity. Rearranging gives the display. Making the $o(1/n)$ rigorous requires a uniform-integrability argument (Ch. 7's tools plus a truncation), which is why I label this a sketch. $\square$

The bias is $O(1/n)$ while the standard deviation is $O(n^{-1/2})$, so bias is asymptotically negligible — but at small $n$, which is exactly the regime of a $K=5$ importance-weighted ELBO term, it is not.

**Worked Example 13.8 (SNIS bias, measured).** Target $P = N(1,1)$, proposal $Q = N(0,1)$, $f(x) = x$, so $\theta = 1$. Weights $\tilde w = e^{z - 1/2}$. Averaging over 40,000 replicates (seed 13):

| $m$ | mean of $\hat\theta^{\mathrm{SN}}$ | bias | $m\times$bias | sd |
|---|---|---|---|---|
| 25 | 0.9072 | $-0.0928$ | $-2.32$ | 0.3735 |
| 50 | 0.9529 | $-0.0471$ | $-2.36$ | 0.2893 |
| 100 | 0.9744 | $-0.0256$ | $-2.56$ | 0.2155 |
| 400 | 0.9941 | $-0.0059$ | $-2.37$ | 0.1142 |
| 1600 | 0.9984 | $-0.0016$ | $-2.56$ | 0.0579 |

The product $m \times \text{bias}$ is stable near $-2.4$, confirming the $1/m$ rate of Proposition 13.7, and the bias is *systematically toward the proposal* (the estimate is pulled from 1 toward 0). Keep this in mind in §13.8: the IWAE bound uses a self-normalized ratio with $K$ often as small as 5, and the resulting bias is not an accident but a designed one — it is what tightens the bound.

### 13.2.4 Variance reduction

Four classical techniques, each with a one-line summary: **control variates** subtract something you know the mean of; **antithetic variates** induce negative correlation; **stratification** removes between-stratum variance exactly; **Rao–Blackwellization** replaces a random quantity by its conditional expectation. All four appear in modern ML, usually unnamed.

**Proposition 13.8 (control variates).** Let $h$ be measurable with known mean $\mathbb{E}[h(X)] = \eta$ and finite variance. For $c \in \mathbb{R}$ define $f_c := f - c(h - \eta)$. Then $\mathbb{E}[f_c(X)] = \theta$ for every $c$, and
$$
\operatorname{Var}(f_c) = \operatorname{Var}(f) - 2c\operatorname{Cov}(f,h) + c^2\operatorname{Var}(h),
$$
minimized at $c^\star = \operatorname{Cov}(f,h)/\operatorname{Var}(h)$ with minimum $\operatorname{Var}(f)(1-\rho^2)$, where $\rho = \operatorname{Corr}(f,h)$.

*Proof.* Unbiasedness is linearity. The variance expression is the expansion of $\operatorname{Var}(f - ch)$. It is a strictly convex quadratic in $c$; setting the derivative $-2\operatorname{Cov}(f,h) + 2c\operatorname{Var}(h)$ to zero gives $c^\star$, and substituting yields $\operatorname{Var}(f) - \operatorname{Cov}(f,h)^2/\operatorname{Var}(h) = \operatorname{Var}(f)(1-\rho^2)$. $\square$

Two practical remarks. Using $\hat c$ estimated from the same samples introduces a small $O(1/n)$ bias and is standard practice; with several control variates $h_1,\dots,h_k$ the optimal coefficient vector is the least-squares regression of $f$ on the $h_j$'s, and the residual variance is $\operatorname{Var}(f)(1 - R^2)$. This is exactly the *baseline* in REINFORCE: the score-function estimator $\hat g = f(X)\nabla_\theta \log p_\theta(X)$ is modified to $(f(X) - b)\nabla_\theta \log p_\theta(X)$, which is legitimate because $\mathbb{E}[\nabla_\theta \log p_\theta(X)] = 0$ (the mean-zero score property, Ch. 2) — so $h = \nabla_\theta\log p_\theta(X)$ has known mean $\eta = 0$ and $b$ plays the role of $c$. Advantage estimation in policy-gradient methods is control variates with a learned $c^\star$.

**Proposition 13.9 (antithetic variates).** Let $U \sim \mathrm{Unif}(0,1)^d$ and let $\tilde U := \mathbf{1} - U$ (componentwise). Then $\tilde U \sim \mathrm{Unif}(0,1)^d$, and the estimator based on $m$ antithetic pairs (hence $n = 2m$ evaluations),
$$
\hat\theta^{\mathrm{anti}} = \frac1m \sum_{j=1}^m \frac{f(U_j) + f(\mathbf 1 - U_j)}{2},
$$
is unbiased with
$$
\operatorname{Var}(\hat\theta^{\mathrm{anti}}) = \frac{\sigma^2}{n}\,(1+\rho), \qquad \rho := \operatorname{Corr}\big(f(U), f(\mathbf 1 - U)\big).
$$
If $d = 1$ and $f$ is monotone, then $\rho \le 0$.

*Proof.* Marginal uniformity of $\mathbf1 - U$ is immediate. $\operatorname{Var}$ of one pair term is $\frac14(2\sigma^2 + 2\rho\sigma^2) = \frac{\sigma^2(1+\rho)}{2}$; dividing by $m = n/2$ gives the display. For the monotone claim, let $(U,V)$ be independent uniforms. If $f$ is non-decreasing then for any $u,v$, $(f(u)-f(v))\big(f(1-u)-f(1-v)\big) \le 0$, since $u > v$ forces $f(u)\ge f(v)$ and $f(1-u) \le f(1-v)$. Taking expectations over independent $U,V$ and expanding,
$$
0 \ge \mathbb{E}\big[(f(U)-f(V))(f(1-U)-f(1-V))\big] = 2\big(\mathbb{E}[f(U)f(1-U)] - \mathbb{E}[f(U)]\mathbb{E}[f(1-U)]\big) = 2\operatorname{Cov},
$$
using that $U, V$ are i.i.d. so $\mathbb{E}[f(U)f(1-V)] = \mathbb{E}[f(U)]\mathbb{E}[f(1-V)] = \mathbb{E}[f]^2$. $\square$

The bound $\rho \le 0$ requires monotonicity, and the requirement bites.

> **Common Confusion 13.3.** *"Antithetic sampling is free variance reduction, so I should always use it."*
>
> Consider $f(u) = (u - 1/2)^2$, which is symmetric about $1/2$ and therefore satisfies $f(1-u) = f(u)$ exactly. Then $\rho = +1$ and Proposition 13.9 gives $\operatorname{Var} = 2\sigma^2/n$: the antithetic estimator is exactly **twice as bad** as plain Monte Carlo, because each pair is two copies of one number. Measured with $n = 10^5$ (seed 13): plain per-evaluation variance $5.552\times10^{-3}$, antithetic-equivalent $1.111\times10^{-2}$, ratio $0.50$, sample correlation $\rho = 1.0000$. Antithetics help for monotone-ish integrands and hurt for symmetric ones. In ML this matters for "mirrored sampling" in evolution strategies, which works well for skewed objectives and does nothing for symmetric ones.

**Proposition 13.10 (stratification never hurts).** Partition $\mathcal X = \biguplus_{k=1}^K S_k$ with $p_k := P(X \in S_k) > 0$ known. Draw $n_k = n p_k$ independent samples from the conditional law $P(\cdot \mid S_k)$ in each stratum and set $\hat\theta^{\mathrm{str}} := \sum_k p_k \bar f_k$ where $\bar f_k$ is the within-stratum average. Then $\hat\theta^{\mathrm{str}}$ is unbiased and
$$
\operatorname{Var}(\hat\theta^{\mathrm{str}}) = \frac 1n \sum_k p_k \operatorname{Var}(f \mid S_k) = \frac{1}{n}\,\mathbb{E}\big[\operatorname{Var}(f \mid S)\big] \;\le\; \frac{\sigma^2}{n},
$$
where $S$ is the stratum label.

*Proof.* Unbiasedness is the law of total expectation. The variance is $\sum_k p_k^2 \operatorname{Var}(f\mid S_k)/n_k = \frac1n\sum_k p_k \operatorname{Var}(f\mid S_k)$. The inequality is the law of total variance (Ch. 6), $\sigma^2 = \mathbb{E}[\operatorname{Var}(f\mid S)] + \operatorname{Var}(\mathbb{E}[f\mid S])$, and the second term is $\ge 0$. $\square$

Stratification removes exactly the *between*-stratum variance. The same decomposition is **Rao–Blackwellization** (Ch. 8): replacing $f(X)$ by $\mathbb{E}[f(X)\mid T]$ for any statistic $T$ preserves the mean and cannot increase the variance, because conditional expectation is an $L^2$ projection (Ch. 6). In ML this is what "analytic KL in the ELBO" is: instead of estimating $\mathbb{E}_q[\log q - \log p]$ by sampling, integrate it exactly for Gaussian $q$ and prior. Same objective, strictly smaller gradient variance. Likewise **common random numbers** — reusing the same $\varepsilon$ draws when comparing two configurations — reduces the variance of the *difference* by inducing positive correlation; this is why the reparameterization trick (Ch. 2), which fixes the noise and moves the parameters, has so much lower variance than REINFORCE, which redraws.

**Worked Example 13.9 (control variate for $\mathbb{E}[e^U]$, derived and measured).** Take $h(u) = u$, $\eta = 1/2$, $\operatorname{Var}(h) = 1/12$. Then
$$
\operatorname{Cov}(e^U, U) = \mathbb{E}[Ue^U] - \tfrac12(e-1) = 1 - \tfrac12(e-1) = 0.140859,
$$
using $\int_0^1 u e^u du = 1$. So $c^\star = 0.140859 \times 12 = 1.690309$ and $\rho^2 = c^{\star 2}\operatorname{Var}(h)/\sigma^2 = 0.983720$, predicting a variance reduction factor $1/(1-\rho^2) = 61.4$.

**Worked Example 13.10 (all four methods on one integral).** Same target $\theta = e-1$. I ran 4000 independent replicates, each using exactly $N = 1200$ evaluations of $e^u$ (seed 99), and measured the variance of the estimator directly rather than trusting a formula:

| method | mean | bias | $\operatorname{Var}(\hat\theta)$ | $N\cdot\operatorname{Var}$ | reduction vs plain |
|---|---|---|---|---|---|
| plain MC | 1.718690 | $+4.1\times10^{-4}$ | $1.981\times10^{-4}$ | $2.377\times10^{-1}$ | $1.0\times$ |
| antithetic | 1.718327 | $+4.5\times10^{-5}$ | $6.764\times10^{-6}$ | $8.117\times10^{-3}$ | $29.8\times$ |
| control variate $h=u$ | 1.718307 | $+2.6\times10^{-5}$ | $3.237\times10^{-6}$ | $3.885\times10^{-3}$ | $62.3\times$ |
| control variates $h=(u,u^2)$ | 1.718284 | $+1.7\times10^{-6}$ | $2.313\times10^{-8}$ | $2.776\times10^{-5}$ | $8718\times$ |
| stratified, 40 strata | 1.718289 | $+7.3\times10^{-6}$ | $1.372\times10^{-7}$ | $1.647\times10^{-4}$ | $1470\times$ |

Check the predictions. Control variate: theory said $61.4\times$, measurement says $62.3\times$. Antithetic: $\rho = \operatorname{Corr}(e^U, e^{1-U})$ is computable exactly, $\operatorname{Cov} = e - (e-1)^2 = -0.234182$, so $\rho = -0.967670$ and theory predicts $1/(1+\rho) = 30.9\times$; measurement says $29.8\times$. Both confirmed. Two extra lessons: adding $u^2$ as a second control variate is spectacular here because $e^u$ is nearly quadratic on $[0,1]$ — the residual after regressing on $\{1,u,u^2\}$ is tiny — and stratification with $K$ strata gives $O(K^{-2})$ variance for smooth $f$, which is why 40 strata bought three orders of magnitude. In high dimensions stratification dies (you would need $K^d$ cells), control variates survive if you can find correlated computable functions, and this asymmetry is why learned baselines rather than grids dominate deep RL.

> **Self-check 13.2.**
> 1. Your importance weights have $\hat k = 1.2$. What can you say about $\operatorname{Var}(w)$, and what should you do?
> 2. Why is the REINFORCE baseline a control variate rather than a bias-inducing hack? Which chapter's result licenses it?
> 3. Give an $f$ for which stratification into 2 strata gives *zero* variance.
> 4. In off-policy evaluation, the behaviour policy plays action $a$ with probability 0 but the target policy plays it with probability 0.3. Which hypothesis of Proposition 13.4 fails, and is the resulting estimator biased or merely high-variance?

---

## 13.3 Rejection sampling and its curse

Importance sampling gives you weighted samples from $P$. Sometimes you want *unweighted* exact samples — for instance to feed them to code that has no notion of weights. Rejection sampling delivers exactly that, at a price that turns out to be unpayable in high dimensions.

**Definition 13.11 (rejection sampler).** Let $P, Q$ have densities $p, q$ with respect to a base measure $\mu$, and suppose there exists $M < \infty$ with $p(x) \le M q(x)$ for $\mu$-a.e. $x$. The *rejection sampler* repeats: draw $Y \sim Q$ and $U \sim \mathrm{Unif}(0,1)$ independently; if $U \le p(Y)/(M q(Y))$ return $Y$ ("accept"), else discard and repeat.

**Theorem 13.12 (correctness and cost).** The output of the rejection sampler has law exactly $P$; the acceptance probability at each trial is $1/M$; and the number of trials until the first acceptance is $\mathrm{Geometric}(1/M)$ with mean $M$, independent of the returned value.

*Proof.* Let $A := \{U \le p(Y)/(Mq(Y))\}$. Conditioning on $Y$ and integrating out $U$,
$$
P(A) = \mathbb{E}_Q\Big[\frac{p(Y)}{Mq(Y)}\Big] = \frac{1}{M}\int \frac{p}{q}\,q \, d\mu = \frac1M ,
$$
where the middle step needs $q > 0$ wherever $p>0$, which is implied by $p \le Mq$. For any measurable $B$,
$$
P(Y \in B \mid A) = \frac{P(\{Y\in B\}\cap A)}{P(A)} = \frac{\int_B \frac{p}{Mq} q \, d\mu}{1/M} = \int_B p \, d\mu = P(B).
$$
Trials are i.i.d., so the trial index of the first acceptance is geometric with success probability $1/M$; and because the conditional law of $Y$ given acceptance does not depend on the trial index, the returned value is independent of the number of trials. $\square$

Notice what the proof used: nothing about dimension, smoothness, or unimodality. Rejection sampling is exact and simple. Its only defect is $M$.

**Proposition 13.13 (the curse, quantified for two canonical cases).**
(i) Let $P = N(0, I_d)$ and $Q = N(0, s^2 I_d)$ with $s > 1$. Then the smallest valid constant is $M = s^{d}$, so the acceptance rate is $s^{-d}$.
(ii) Let $P = \mathrm{Unif}(B_d)$, the unit ball, and $Q = \mathrm{Unif}([-1,1]^d)$. Then the acceptance rate is $\operatorname{vol}(B_d)/2^d = \pi^{d/2}/(\Gamma(\tfrac d2 + 1)2^d)$.

*Proof.* (i) The ratio is
$$
\frac{p(x)}{q(x)} = \frac{(2\pi)^{-d/2}e^{-\|x\|^2/2}}{(2\pi s^2)^{-d/2}e^{-\|x\|^2/(2s^2)}} = s^{d}\exp\Big(-\frac{\|x\|^2}{2}\Big(1 - \frac1{s^2}\Big)\Big),
$$
which is maximized at $x = 0$ with value $s^d$; and $M = \sup p/q$ is the smallest constant that works. (ii) $p = \mathbf 1_{B_d}/\operatorname{vol}(B_d)$, $q = \mathbf 1_{[-1,1]^d}/2^d$, so $\sup p/q = 2^d/\operatorname{vol}(B_d)$ and the acceptance rate is the reciprocal. The volume formula for $B_d$ is standard (Ch. 12). $\square$

**Worked Example 13.11 (how fast the door closes).** Both cases, computed exactly:

| $d$ | $N(0,I_d)$ from $N(0,1.1^2 I_d)$: acceptance | expected trials | ball in cube: acceptance |
|---|---|---|---|
| 1 | $0.909$ | $1.10$ | $1.000$ |
| 2 | $0.826$ | $1.21$ | $0.785$ |
| 5 | $0.621$ | $1.61$ | $0.164$ |
| 10 | $0.386$ | $2.59$ | $2.49\times10^{-3}$ |
| 20 | $0.149$ | $6.73$ | $2.46\times10^{-8}$ |
| 50 | $8.52\times10^{-3}$ | $117$ | $1.54\times10^{-28}$ |
| 100 | $7.26\times10^{-5}$ | $1.38\times10^{4}$ | — |

Empirical check of the first column at $d = 5$: with $2\times10^5$ proposals (seed 2) the acceptance rate was $0.6219$ against the predicted $0.6209$, and the accepted sample had mean $0.0017$ and standard deviation $1.0010$ across all coordinates — exactly $N(0,I_5)$, as Theorem 13.12 promises.

Read the first column again. A proposal that is wrong by *ten percent in each coordinate's scale* — the mildest possible misspecification — costs a factor of $1.1^{d}$. At $d = 100$ that is 14,000 proposals per sample; at $d = 500$ it is $10^{20}$. And $d = 100$ is a small model. This is the fundamental reason MCMC exists: **rejection sampling requires a globally valid envelope, and in high dimensions no global envelope is tight.** MCMC gives up independence between samples in exchange for only ever needing *local* moves.

<svg viewBox="0 0 700 400" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Acceptance rate of rejection sampling collapsing with dimension">
  <line x1="60" y1="40" x2="60" y2="340" stroke="currentColor" stroke-width="1.5"/>
  <line x1="60" y1="340" x2="670" y2="340" stroke="currentColor" stroke-width="1.5"/>
  <text x="14" y="44" fill="currentColor" font-size="11">10⁰</text>
  <text x="14" y="144" fill="currentColor" font-size="11">10⁻¹⁰</text>
  <text x="14" y="244" fill="currentColor" font-size="11">10⁻²⁰</text>
  <text x="14" y="344" fill="currentColor" font-size="11">10⁻³⁰</text>
  <line x1="60" y1="140" x2="670" y2="140" stroke="currentColor" stroke-width="0.4" stroke-dasharray="3 4"/>
  <line x1="60" y1="240" x2="670" y2="240" stroke="currentColor" stroke-width="0.4" stroke-dasharray="3 4"/>
  <text x="300" y="368" fill="currentColor" font-size="12">dimension d</text>
  <text x="180" y="356" fill="currentColor" font-size="11">10</text>
  <text x="300" y="356" fill="currentColor" font-size="11">20</text>
  <text x="420" y="356" fill="currentColor" font-size="11">30</text>
  <text x="540" y="356" fill="currentColor" font-size="11">40</text>
  <text x="656" y="356" fill="currentColor" font-size="11">50</text>
  <polyline fill="none" stroke="#ef4444" stroke-width="2" points="72,40 96,43 120,48 144,54 168,62 192,70 216,80 240,89 264,100 288,111 312,122 336,133 360,145 384,158 408,170 432,183 456,197 480,210 504,224 528,238 552,252 576,266 600,281 624,296 648,311 660,318"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="72,40 96,41 120,42 144,43 168,44 192,45 216,45 240,46 264,47 288,48 312,49 336,50 360,50 384,51 408,52 432,53 456,54 480,54 504,55 528,56 552,57 576,58 600,59 624,59 648,60 660,61"/>
  <text x="470" y="292" fill="#ef4444" font-size="12">unit ball inside the cube</text>
  <text x="470" y="82" fill="#3b82f6" font-size="12">N(0,I) from N(0,1.1²I): rate 1.1⁻ᵈ</text>
  <text x="60" y="386" fill="currentColor" font-size="11.5">Figure 13.1 — Rejection sampling's acceptance rate on a log scale. Even a proposal wrong by only 10% per</text>
  <text x="60" y="399" fill="currentColor" font-size="11.5">coordinate (blue) decays geometrically; a naive envelope (red) is hopeless past d≈15. Exact values, not simulated.</text>
</svg>

> **Self-check 13.3.**
> 1. In Theorem 13.12, where exactly is $P \ll Q$ used, and what goes wrong if $p(x) > Mq(x)$ on a set of positive measure?
> 2. You accept 1 proposal in 500. What is the expected number of density evaluations to obtain 1000 samples?
> 3. Why does the returned sample's independence from the trial count matter for the correctness of a loop that returns "the first accepted draw"?
> 4. Rejection sampling needs $\sup p/q$; importance sampling does not. What does importance sampling pay instead?

---

## 13.4 MCMC fundamentals

### 13.4.1 The idea: give up independence, keep the target

Here is the shift in perspective. Instead of drawing independent samples from $\pi$ (the target; I switch to $\pi$ because in Bayesian applications the target is a posterior and $P$ is overloaded), build a *Markov chain* $X_0, X_1, X_2, \dots$ whose long-run distribution is $\pi$, and average along the trajectory. The samples are correlated, which costs you efficiency, but each step only requires a *local* proposal, which is what makes high dimensions survivable.

We work on a general measurable space $(\mathcal{X}, \mathcal{B})$, which for us will be $\mathbb{R}^d$ with $\mathcal B(\mathbb R^d)$ (Definition 1.6).

**Definition 13.14 (transition kernel).** A *(Markov) transition kernel* is a map $K : \mathcal X \times \mathcal B \to [0,1]$ such that (i) for each $x$, $K(x, \cdot)$ is a probability measure on $\mathcal B$; (ii) for each $B \in \mathcal B$, $x \mapsto K(x,B)$ is measurable. Given $K$ and an initial law $\mu_0$, the chain has $X_0 \sim \mu_0$ and $P(X_{t+1} \in B \mid X_t = x) = K(x,B)$. The law of $X_t$ is $\mu_t = \mu_0 K^t$, where $(\mu K)(B) := \int K(x,B)\,\mu(dx)$.

Note what a kernel is: a *conditional distribution*, exactly the object Chapter 6 built when it constructed conditional expectation, and the measurability requirement in (ii) is what makes $\mu K$ well defined.

**Definition 13.15 (invariance).** A probability measure $\pi$ is *invariant* (or stationary) for $K$ if $\pi K = \pi$, i.e. $\int K(x,B)\,\pi(dx) = \pi(B)$ for all $B \in \mathcal B$.

Invariance says: if $X_t \sim \pi$ then $X_{t+1}\sim\pi$. It is a fixed-point property, not a convergence property; a chain can have $\pi$ invariant and never approach it (start it at a point it cannot leave).

**Definition 13.16 (reversibility / detailed balance).** $K$ is *reversible* with respect to $\pi$ if for all $A, B \in \mathcal B$,
$$
\int_A K(x, B)\, \pi(dx) = \int_B K(y, A)\, \pi(dy).
$$
Equivalently, if $X_0 \sim \pi$ then $(X_0, X_1) \stackrel{d}{=} (X_1, X_0)$. If $K(x,dy) = k(x,y)\,\mu(dy)$ and $\pi(dx) = \pi(x)\mu(dx)$ for a common base measure $\mu$, this reads
$$
\pi(x)k(x,y) = \pi(y)k(y,x) \quad \text{for } \mu\otimes\mu\text{-a.e. }(x,y).
$$

**Proposition 13.17 (detailed balance implies invariance).** If $K$ is reversible w.r.t. $\pi$ then $\pi$ is invariant for $K$.

*Proof.* Take $A = \mathcal X$ in Definition 13.16: $\int_{\mathcal X} K(x,B)\pi(dx) = \int_B K(y,\mathcal X)\pi(dy) = \int_B 1 \, \pi(dy) = \pi(B)$. $\square$

That one-line proof is the engine of all of MCMC: detailed balance is a *pointwise, checkable* condition, and it hands you invariance for free. The converse is false — many useful chains (Gibbs with a fixed sweep order, HMC without a momentum flip, most "non-reversible" samplers) are invariant but not reversible.

Invariance alone is not enough for averaging to work. We need the chain to actually explore.

**Definition 13.18 (irreducibility, aperiodicity, Harris recurrence).** Let $\varphi$ be a nonzero measure on $\mathcal B$. The chain is *$\varphi$-irreducible* if for every $x$ and every $B$ with $\varphi(B) > 0$ there is $t \ge 1$ with $K^t(x,B) > 0$. It is *aperiodic* if there is no partition of the state space into $d \ge 2$ cyclically visited pieces. It is *Harris recurrent* if for every $B$ with $\varphi(B) > 0$ and every starting point, the chain visits $B$ infinitely often almost surely. (The phrasing of aperiodicity here is a deliberately informal gloss: on a general state space the precise definition runs through small sets and cycle decompositions, and lives in Meyn–Tweedie, Ch. 5. Nothing in this chapter turns on the finer points, and §13.12 records the borrowing.)

**Theorem 13.19 (ergodic theorem for Harris chains; stated, not proved).** Let $K$ have invariant probability measure $\pi$, and suppose the chain is $\pi$-irreducible, aperiodic, and Harris recurrent. Then for $\pi$-a.e. starting point (in fact every starting point, under Harris recurrence), $\|\mu_0 K^t - \pi\|_{TV} \to 0$, and for every $f \in L^1(\pi)$,
$$
\frac1T\sum_{t=1}^T f(X_t) \;\to_{as}\; \mathbb{E}_\pi[f].
$$
If in addition the chain is geometrically ergodic and $\mathbb{E}_\pi[f^2] < \infty$, a CLT holds:
$$
\sqrt T\Big(\frac1T\sum_{t\le T} f(X_t) - \mathbb{E}_\pi f\Big) \to_d N\big(0, \sigma_f^2 \tau_f\big),
$$
where $\sigma^2_f = \operatorname{Var}_\pi(f)$ and $\tau_f = 1 + 2\sum_{\ell\ge1}\rho_\ell$ is the *integrated autocorrelation time*, $\rho_\ell$ being the stationary lag-$\ell$ autocorrelation of $f(X_t)$.

This theorem is the licence for everything that follows, and it is the main result of this chapter that I am not proving; the proof (via regeneration/split chains) is a chapter of Meyn–Tweedie. It appears in §13.12's table. What you should take from it operationally: **MCMC averages are consistent, and their error bars are the Monte Carlo error bars of §13.1 inflated by $\tau_f$**, which is why §13.6 spends its time estimating $\tau_f$.

### 13.4.2 Metropolis–Hastings

Now the construction. We want a kernel with $\pi$ invariant, using only the ability to evaluate $\pi$ *up to a constant* — decisive, since a posterior is $\tilde\pi = \text{likelihood}\times\text{prior}$ with unknown normalizer.

**Definition 13.20 (Metropolis–Hastings kernel).** Let $\pi$ have density $\pi(x)$ w.r.t. $\mu$, and let $Q(x, dy) = q(x,y)\mu(dy)$ be a proposal kernel. Define the acceptance probability
$$
\alpha(x,y) := \min\Big\{1, \; \frac{\pi(y)\,q(y,x)}{\pi(x)\,q(x,y)}\Big\}
$$
(with $\alpha := 1$ when the denominator vanishes). The MH chain at $x$ draws $Y \sim Q(x,\cdot)$, accepts $X_{t+1} = Y$ with probability $\alpha(x,Y)$, and otherwise sets $X_{t+1} = x$. As a kernel,
$$
K(x, dy) = \alpha(x,y)q(x,y)\mu(dy) + r(x)\,\delta_x(dy), \qquad r(x) = 1 - \int \alpha(x,y)q(x,y)\mu(dy).
$$

The $\delta_x$ term (Chapter 1's Dirac measure) is not a technicality: **the MH kernel is genuinely a mixture of a continuous part and an atom at the current state.** It has no density with respect to Lebesgue measure. This is Chapter 1's Failure 2 — an object that is neither discrete nor continuous — appearing in the most-used algorithm in Bayesian computation, and it is exactly why we needed Chapter 2's framework of measures rather than densities to write the algorithm down honestly.

**Theorem 13.21 (MH is reversible w.r.t. $\pi$, hence leaves $\pi$ invariant).** The kernel of Definition 13.20 satisfies detailed balance with respect to $\pi$, and therefore $\pi K = \pi$. Moreover $\alpha$ depends on $\pi$ only through ratios, so an unnormalized $\tilde\pi = Z\pi$ gives the identical algorithm.

*Proof.* The atom contributes symmetrically: $\int_A r(x)\delta_x(B)\pi(dx) = \int_{A\cap B} r \, d\pi$, which is symmetric in $A, B$. So it suffices to check detailed balance for the continuous part, i.e. that
$$
\pi(x)q(x,y)\alpha(x,y) = \pi(y)q(y,x)\alpha(y,x) \quad \text{for a.e. }(x,y).
$$
Fix $x,y$ with $\pi(x)q(x,y) > 0$ and $\pi(y)q(y,x)>0$, and set $R := \dfrac{\pi(y)q(y,x)}{\pi(x)q(x,y)}$, so $\alpha(x,y) = \min\{1,R\}$ and $\alpha(y,x) = \min\{1,1/R\}$. If $R \le 1$: the left side is $\pi(x)q(x,y)R = \pi(y)q(y,x)$ and the right side is $\pi(y)q(y,x)\cdot 1$. If $R > 1$: the left side is $\pi(x)q(x,y)$ and the right side is $\pi(y)q(y,x)/R = \pi(x)q(x,y)$. Either way they agree. Using the identity $\min\{1,R\} = \min\{1,R\}$ symmetrically, one can write both cases at once as
$$
\pi(x)q(x,y)\alpha(x,y) = \min\{\pi(x)q(x,y),\, \pi(y)q(y,x)\},
$$
which is manifestly symmetric in $(x,y)$ — this is the cleanest way to see the theorem. If $\pi(x)q(x,y) = 0$ then both sides are $0$ (using the convention on $\alpha$ and that $\min$ of a pair containing 0 is 0). Proposition 13.17 then gives invariance. The ratio structure is evident from the formula for $\alpha$. $\square$

The symmetric form $\min\{\pi(x)q(x,y), \pi(y)q(y,x)\}$ is also the fastest way to *discover* the acceptance ratio rather than verify it: if you want a reversible kernel whose continuous part is $q$ times something, the largest symmetric function you can build pointwise from $\pi q$ is the minimum of the two directions, and dividing by $\pi(x)q(x,y)$ gives $\alpha$. Exercise C1 asks you to make that derivation precise.

One loose end should be tied before moving on, because a self-contained chapter must not leave it implicit: Theorem 13.21 delivers *invariance*, but Theorem 13.19 also demands $\pi$-irreducibility, aperiodicity, and Harris recurrence. For the chains actually used in this chapter these are checkable, in decreasing order of self-sufficiency. First, if the proposal density is strictly positive everywhere — as for the Gaussian random walk $q(x,y) = N(y;\, x, \sigma^2 I)$ or a Gaussian independence proposal — then from any $x$ the chain reaches any set $B$ of positive $\pi$-measure in a single step with positive probability, because $\alpha(x,y) > 0$ wherever $\pi(y) > 0$: that is $\pi$-irreducibility. Second, aperiodicity comes from the atom: at any $x$ with rejection probability $r(x) > 0$ — which occurs whenever $\pi$ is not constant over the proposal's reach — we have $K(x, \{x\}) > 0$, so the chain can idle in place, and an irreducible chain that can idle admits no cyclic partition. (Chains that never reject, such as Gibbs, need a separate but equally routine argument.) Third, Harris recurrence is the one genuinely technical item, and here Metropolis–Hastings is kind: **every $\pi$-irreducible MH chain is automatically Harris recurrent** (Tierney's corollary) — the pathologies that separate recurrence from Harris recurrence cannot occur for kernels of this form. That implication is borrowed without proof and recorded in §13.12. The practical summary: for a continuous, everywhere-positive target and an everywhere-positive proposal density, random-walk Metropolis satisfies *every* hypothesis of Theorem 13.19, so its failures — and §13.4.3–§13.4.4 catalogue them — are always failures of *rate*, never of the *fact* of convergence.

Two special cases dominate practice. If $q(x,y) = q(y,x)$ (a *symmetric* proposal, e.g. $Y = x + \sigma\varepsilon$ with $\varepsilon\sim N(0,I)$), the proposal densities cancel and $\alpha = \min\{1, \pi(y)/\pi(x)\}$: this is **random-walk Metropolis**. If the proposal is the exact conditional of the target for one block of coordinates, the acceptance probability is identically 1:

**Proposition 13.22 (Gibbs sampling is MH with acceptance 1).** Partition $x = (x_1,\dots,x_m)$ into blocks and let the proposal replace block $i$ by a draw from the exact conditional $\pi(x_i \mid x_{-i})$, leaving $x_{-i}$ fixed. Then $\alpha \equiv 1$, and the resulting kernel leaves $\pi$ invariant.

*Proof.* Here $q(x,y) = \pi(y_i\mid x_{-i})\mathbf 1\{y_{-i} = x_{-i}\}$. On the set where $y_{-i} = x_{-i}$,
$$
\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)} = \frac{\pi(y_i, x_{-i})\,\pi(x_i \mid x_{-i})}{\pi(x_i, x_{-i})\,\pi(y_i\mid x_{-i})}
= \frac{\pi(y_i\mid x_{-i})\pi(x_{-i})\pi(x_i\mid x_{-i})}{\pi(x_i\mid x_{-i})\pi(x_{-i})\pi(y_i\mid x_{-i})} = 1 .
$$
So $\alpha = 1$; each block update is reversible w.r.t. $\pi$ by Theorem 13.21, hence invariant. (A deterministic sweep through blocks composes invariant kernels, so it is invariant, though not reversible; a randomly chosen block gives a reversible kernel.) $\square$

*Blocked* Gibbs updates several coordinates jointly, which helps enormously when they are strongly dependent — for a Gaussian, updating a correlated pair jointly removes their correlation from the mixing problem entirely. *Collapsed* Gibbs integrates some variables out analytically before sampling the rest (this is Rao–Blackwellization again, Ch. 8): collapsed Gibbs for LDA integrates out the topic-word and document-topic distributions and samples only the discrete assignments, which is both faster per step and better mixing.

**Worked Example 13.12 (Gibbs on a correlated bivariate Gaussian, exactly).** Let $\pi = N\big(0, \begin{smallmatrix}1 & \rho\\ \rho & 1\end{smallmatrix}\big)$. The conditionals are $X_1 \mid x_2 \sim N(\rho x_2, 1-\rho^2)$ and symmetrically (Ch. 5's Gaussian conditioning). A full sweep gives $X_1^{(t+1)} = \rho X_2^{(t)} + \sqrt{1-\rho^2}\,\varepsilon$, then $X_2^{(t+1)} = \rho X_1^{(t+1)} + \sqrt{1-\rho^2}\,\varepsilon'$, so
$$
X_1^{(t+1)} = \rho^2 X_1^{(t)} + \text{noise},
$$
an AR(1) with autocorrelation $\rho^2$ per sweep. Hence $\tau_{f} = (1+\rho^2)/(1-\rho^2)$ for $f(x)=x_1$: at $\rho = 0.99$ that is $\tau \approx 100$, so 100,000 sweeps give an effective sample size of about 1000. The diagnosis is exact and it is the standard one: **Gibbs mixes slowly along directions of strong dependence, because it can only move parallel to the axes.**

### 13.4.3 Tuning: the step size is everything

**Worked Example 13.13 (random-walk MH on a banana).** Target: the twisted Gaussian
$$
\log \pi(x_1,x_2) = -\frac{x_1^2}{200} - \frac{1}{2}\big(x_2 - b(x_1^2 - 100)\big)^2 + \text{const}, \qquad b = 0.03,
$$
whose $x_1$-marginal is exactly $N(0,100)$ and which is strongly curved. I ran $N = 20{,}000$ iterations of random-walk MH with isotropic proposal $N(0,\sigma^2 I_2)$ from $x_0 = (0,0)$, seed 101, estimating $\tau$ by Geyer's initial-positive-sequence rule:

| proposal $\sigma$ | acceptance rate | ESS($x_1$) | ESS($x_2$) | ESS per 1000 density evaluations |
|---|---|---|---|---|
| 0.2 | 0.930 | 4.3 | 6.1 | 0.21 / 0.31 |
| 1.0 | 0.656 | 17.8 | 17.7 | 0.89 / 0.89 |
| 2.4 | 0.404 | 118.7 | 82.2 | 5.93 / 4.11 |
| 4.0 | 0.257 | 183.8 | 197.0 | 9.19 / 9.85 |
| 6.0 | 0.168 | 249.1 | 367.0 | 12.46 / 18.35 |
| 8.0 | 0.130 | 314.4 | 289.3 | 15.72 / 14.47 |
| 10.0 | 0.097 | 296.1 | 332.5 | 14.81 / 16.63 |
| 20.0 | 0.037 | 301.0 | 266.5 | 15.05 / 13.32 |

Read the two ends. At $\sigma = 0.2$ almost everything is accepted and the chain barely moves: 20,000 iterations bought 4 independent samples. At $\sigma = 20$ almost everything is rejected, but the rare accepted jumps are large, so per-evaluation efficiency is still respectable. The optimum here is broad, around $\sigma \in [6,20]$, i.e. acceptance rates of 4–17% — noticeably *lower* than the famous 0.234, because that figure is an asymptotic result for product targets in high dimension, and this is a curved 2-D target whose $x_1$ scale is 10. A useful corollary: **the folklore acceptance rate is a starting point, not a target.** Tune on ESS per evaluation, which is the quantity you actually care about.

<svg viewBox="0 0 720 430" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Trace plots and autocorrelation for Metropolis-Hastings at three step sizes">
  <text x="20" y="18" fill="currentColor" font-size="12">trace of x₁ (2000 iterations)</text>
  <text x="500" y="18" fill="currentColor" font-size="12">autocorrelation ρ(lag)</text>
  <line x1="20" y1="80" x2="440" y2="80" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 3"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="1" points="20,80 40,78 60,74 80,71 100,66 120,62 140,58 160,55 180,52 200,50 220,47 240,45 260,44 280,42 300,41 320,40 340,40 360,39 380,39 400,38 420,38 440,38"/>
  <text x="24" y="34" fill="#ef4444" font-size="11">σ = 0.2  (acc 0.93, ESS 4)  — random walk drifts, never returns</text>
  <line x1="20" y1="170" x2="440" y2="170" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 3"/>
  <polyline fill="none" stroke="#10b981" stroke-width="1" points="20,170 32,150 44,182 56,140 68,196 80,158 92,130 104,188 116,205 128,152 140,120 152,178 164,210 176,142 188,168 200,196 212,132 224,158 236,205 248,148 260,122 272,180 284,200 296,152 308,138 320,192 332,166 344,128 356,184 368,206 380,150 392,132 404,178 416,198 428,146 440,170"/>
  <text x="24" y="124" fill="#10b981" font-size="11">σ = 8  (acc 0.13, ESS 314)  — sweeps the whole marginal</text>
  <line x1="20" y1="290" x2="440" y2="290" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 3"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="1" points="20,290 60,290 60,262 140,262 140,318 220,318 220,275 300,275 300,300 380,300 380,268 440,268"/>
  <text x="24" y="244" fill="#f59e0b" font-size="11">σ = 60 (acc &lt;0.01) — long flat stretches: proposals rejected</text>
  <line x1="500" y1="60" x2="500" y2="300" stroke="currentColor" stroke-width="1.2"/>
  <line x1="500" y1="300" x2="700" y2="300" stroke="currentColor" stroke-width="1.2"/>
  <text x="470" y="64" fill="currentColor" font-size="10">1.0</text>
  <text x="478" y="304" fill="currentColor" font-size="10">0</text>
  <text x="590" y="320" fill="currentColor" font-size="10">lag</text>
  <path d="M500,60 C540,62 580,66 700,72" fill="none" stroke="#ef4444" stroke-width="2"/>
  <path d="M500,60 C520,180 560,280 700,296" fill="none" stroke="#10b981" stroke-width="2"/>
  <path d="M500,60 C560,120 620,200 700,250" fill="none" stroke="#f59e0b" stroke-width="2"/>
  <text x="60" y="352" fill="currentColor" font-size="11.5">Figure 13.2 — The same target, three step sizes. Too small: high acceptance, slow diffusion, autocorrelation</text>
  <text x="60" y="366" fill="currentColor" font-size="11.5">decays over thousands of lags. Well tuned: the trace looks like noise around the mean and the autocorrelation</text>
  <text x="60" y="380" fill="currentColor" font-size="11.5">falls to zero within a few lags. Too large: the chain sticks, producing flat plateaus and slow decay again.</text>
  <text x="60" y="394" fill="currentColor" font-size="11.5">Both failure modes give high autocorrelation, so the trace plot is what distinguishes them. Schematic;</text>
  <text x="60" y="408" fill="currentColor" font-size="11.5">the measured acceptance rates and ESS values are those of Worked Example 13.13.</text>
</svg>

> **Common Confusion 13.4.** *"My chain has $\pi$ as its invariant distribution, so my samples are from $\pi$."*
>
> Invariance (Definition 13.15) is a fixed point: *if* you were at $\pi$ you would stay. It says nothing about getting there. Consider a target that is a mixture of two well-separated Gaussians and a random-walk proposal with small $\sigma$: the chain is invariant for $\pi$, is even irreducible in principle (the Gaussian proposal has full support, so some day it will jump), yet in any feasible run it explores one mode and reports a posterior that is confidently wrong. Invariance is necessary; *rate* of convergence is what you actually need, and Theorem 13.19 gives it only under geometric ergodicity, which is exactly the hypothesis nobody can verify for a real posterior. This is why §13.6 is not optional.

> **Self-check 13.4.**
> 1. Show from Definition 13.20 that if the proposal is symmetric, $\alpha$ reduces to $\min\{1,\pi(y)/\pi(x)\}$, and explain why the normalizing constant of $\pi$ never appears.
> 2. Why does the MH kernel fail to have a density with respect to Lebesgue measure on $\mathbb{R}^d$? Which chapter's machinery lets you write it down anyway?
> 3. In Worked Example 13.12, what happens to $\tau$ as $\rho \to 1$, and what fix does blocked Gibbs provide?
> 4. Both a too-small and a too-large step size give high autocorrelation. Which diagnostic distinguishes them, and how?

---

### 13.4.4 Multiple modes: the failure no diagnostic catches

Everything so far tunes a sampler that has already found the right region. The remaining failure is different in kind, and it is the one you should fear most: if $\pi$ has well-separated modes, a local sampler initialized in one of them may never visit the others, and *every* diagnostic in §13.6 will report success. Four chains that all started in the same basin agree with each other perfectly. This is not a tuning problem — the step size is fine, the acceptance rate is fine, $\hat R$ is $1.00$ — and it is not detectable from within the run. Since it is a real limitation of the whole method, here are the honest options.

**Disperse and count.** Initialize chains from an over-dispersed distribution — much wider than your prior guess — and *report how many distinct modes were found*, not just $\hat R$. If eight chains from eight far-apart starts land in three clusters of final positions, you have learned something no convergence statistic will tell you. This is cheap and should be routine.

**Tempering.** Run auxiliary chains targeting $\pi_\beta \propto \pi^\beta$ for a ladder of inverse temperatures $\beta\in(0,1]$. Flattening $\pi$ lowers the barriers between modes, so the hot chains move between basins freely; a Metropolis swap between adjacent rungs (which is itself an MH step on the product space, and so needs no new theory) lets the cold chain inherit that mobility. Parallel tempering and simulated tempering are the two standard arrangements. The cost is a factor equal to the number of rungs, and the tuning problem moves to choosing the ladder.

**Change the geometry.** Many "modes" are artifacts of parameterization — label-switching in mixture models, scale/shift non-identifiability in hierarchical models, permutation symmetry in neural networks. Reparameterizing, imposing an ordering constraint, or marginalizing the offending discrete variable removes the multimodality rather than fighting it. When available this dominates every other option.

**Give up on the posterior.** For neural networks the number of symmetry-equivalent modes grows super-exponentially in width, and no sampler will visit a representative set. Deep ensembles — independent optimization runs treated as samples — are not a Bayesian posterior and should not be described as one, but they capture between-mode variability that a single well-mixed chain provably cannot. §13.10.1 returns to this.

None of these is a guarantee. The correct summary is that MCMC gives you asymptotically exact inference *conditional on irreducibility being effective at your sample size*, and multimodality is precisely the case where that condition fails invisibly.

---

## 13.5 Gradient-based MCMC

A random walk explores like a drunk: after $t$ steps of size $\sigma$ it has travelled $\sigma\sqrt t$. To cross a target of width $L$ it needs $t \approx (L/\sigma)^2$ steps, and $\sigma$ is capped by the *narrowest* direction, since a proposal larger than that gets rejected. That is the entire scaling problem, and gradients are the fix: $\nabla \log \pi$ tells you which way the mass is, so you can move *ballistically* instead of diffusively.

### 13.5.1 Hamiltonian Monte Carlo

Augment the state $x \in \mathbb{R}^d$ with a momentum $p \in \mathbb R^d$ and define the *Hamiltonian*
$$
H(x,p) := U(x) + \frac12 p^\top M^{-1} p, \qquad U(x) := -\log \pi(x),
$$
with $M$ a symmetric positive-definite mass matrix (take $M = I$ unless stated). The joint density $\propto e^{-H(x,p)} = \pi(x)\,N(p; 0, M)$ factorizes, so **the $x$-marginal of the joint target is exactly $\pi$, and $p$ is independent of $x$.** If we can sample the joint, we can throw away $p$ and we are done.

Hamilton's equations $\dot x = M^{-1}p$, $\dot p = -\nabla U(x)$ preserve $H$ exactly and preserve volume (Liouville). A perfect integrator would therefore move along level sets of $H$, and every proposal would be accepted. We cannot integrate exactly, so we use the *leapfrog* (Störmer–Verlet) scheme.

**Definition 13.23 (leapfrog integrator and the HMC kernel).** One leapfrog step of size $\epsilon$ from $(x,p)$:
$$
p^{1/2} = p - \tfrac{\epsilon}{2}\nabla U(x), \qquad
x' = x + \epsilon M^{-1} p^{1/2}, \qquad
p' = p^{1/2} - \tfrac{\epsilon}{2}\nabla U(x').
$$
The HMC kernel: draw $p \sim N(0,M)$; apply $L$ leapfrog steps to get $(x^\ast, p^\ast)$; negate the momentum, $(x^\ast, -p^\ast)$; accept $x^\ast$ with probability $\min\{1, \exp(H(x,p) - H(x^\ast,p^\ast))\}$, otherwise stay at $x$.

**Proposition 13.24 (leapfrog is volume-preserving and reversible).** Let $T_\epsilon$ be one leapfrog step and $F(x,p) := (x,-p)$. Then (i) $\det \nabla T_\epsilon = 1$; (ii) $F \circ T_\epsilon^{L} \circ F \circ T_\epsilon^L = \mathrm{id}$, i.e. $T^L_\epsilon$ followed by momentum flip is an involution.

*Proof.* (i) Each of the three sub-steps is a *shear*: the first updates $p$ by a function of $x$ alone, so its Jacobian is $\begin{pmatrix} I & 0\\ \ast & I\end{pmatrix}$ with determinant 1; the second updates $x$ by a function of $p^{1/2}$ alone, determinant 1; the third likewise. The composition has determinant $1\cdot1\cdot1 = 1$. (This is the change-of-variables formula from Ch. 3 applied three times.) (ii) Write out $T_\epsilon$ started from $(x', -p')$: the first half-kick gives $-p' - \frac\epsilon2\nabla U(x') = -(p^{1/2})$, the drift gives $x' - \epsilon M^{-1}p^{1/2} = x$, the second half-kick gives $-(p^{1/2}) - \frac\epsilon2 \nabla U(x) = -p$. So $T_\epsilon(F(T_\epsilon(x,p))) = F(x,p)$, and applying $F$ once more gives the identity; iterating over $L$ steps gives the claim. $\square$

**Theorem 13.25 (HMC leaves $\pi$ invariant).** The HMC kernel of Definition 13.23 is reversible with respect to the joint density $\propto e^{-H}$, hence leaves it invariant; consequently its $x$-marginal leaves $\pi$ invariant.

*Proof.* Let $\Phi := F\circ T_\epsilon^L$, an involution by Proposition 13.24(ii) with $|\det\nabla\Phi| = 1$ by (i) and $|\det \nabla F| = 1$. Consider the deterministic-proposal MH move $(x,p)\mapsto \Phi(x,p)$ with acceptance $\min\{1, e^{-H(\Phi(z))+H(z)}\}$. Detailed balance for the joint density $\rho(z) \propto e^{-H(z)}$ requires, for any test set,
$$
\rho(z)\,\alpha(z) \,\big|\det\nabla \Phi(z)\big|^{-1}= \rho(\Phi(z))\,\alpha(\Phi(z))
$$
in the sense of the change-of-variables identity (Ch. 3). With $\alpha(z) = \min\{1, \rho(\Phi(z))/\rho(z)\}$ and unit Jacobian, both sides equal $\min\{\rho(z), \rho(\Phi(z))\}$ — the same symmetric-minimum computation as in Theorem 13.21. Since $\Phi$ is an involution, $\Phi(\Phi(z)) = z$, so the reverse move is available with the correct probability, and the momentum flip is what makes it so. Finally, refreshing $p\sim N(0,M)$ is a Gibbs update of the momentum block, which leaves the joint invariant by Proposition 13.22. A composition of two joint-invariant kernels is joint-invariant, and marginalizing an invariant joint gives invariance of the marginal. In practice the flip is omitted in reporting because $p$ is discarded and $N(0,M)$ is symmetric. $\square$

That proof is worth re-reading for one reason: **the MH correction is not there to fix a modelling approximation, it is there to fix the integrator's discretization error.** The leapfrog scheme conserves $H$ only to $O(\epsilon^2)$ per step, and the acceptance ratio $e^{H(z) - H(z^\ast)}$ measures exactly the accumulated energy error. As $\epsilon \to 0$ acceptance $\to 1$; as $\epsilon$ grows past the stability threshold of the integrator, the energy error explodes and acceptance collapses — sharply, not gradually, because leapfrog on a quadratic is a linear map that becomes unstable at $\epsilon > 2/\sqrt{\lambda_{\max}}$.

Why HMC scales better: for a product target in $d$ dimensions, the energy error accumulates like $\sqrt d \epsilon^2$, so keeping acceptance constant requires $\epsilon \sim d^{-1/4}$, and covering a fixed trajectory length needs $L \sim d^{1/4}$ steps — a cost of $d^{1/4}$ gradient evaluations per nearly-independent sample against random-walk MH's $\sigma\sim d^{-1/2}$ giving $d$ density evaluations. The classic reference results are $d^{5/4}$ versus $d^{2}$ total cost; the exponents depend on assumptions, but the direction is robust and measurable.

**Worked Example 13.14 (dimension scaling, measured).** Target $N(0,I_d)$, monitoring the first coordinate. RWMH with the standard scaling $\sigma = 2.4/\sqrt d$; HMC with $\epsilon = 0.9\,d^{-1/4}$ and $L = \lceil 1/\epsilon\rceil$ (trajectory length $\approx 1$). $N = 20{,}000$ draws each, seeds $1000+d$:

| $d$ | RWMH acc. | ESS | ESS per 1000 evals | HMC acc. | $\epsilon$ | $L$ | ESS | ESS per 1000 grads |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.351 | 2652 | 132.6 | 0.921 | 0.757 | 2 | 17534 | 292.2 |
| 10 | 0.259 | 573 | 28.7 | 0.930 | 0.506 | 2 | 6400 | 106.7 |
| 50 | 0.233 | 123 | 6.15 | 0.932 | 0.338 | 3 | 5765 | 72.1 |
| 200 | 0.234 | 53 | 2.65 | 0.926 | 0.239 | 5 | 8583 | 71.5 |

From $d=2$ to $d=200$, RWMH's efficiency per evaluation fell by a factor of 50 — very nearly the predicted factor of $100 = d$-ratio, with the discrepancy attributable to the finite-$N$ estimate of $\tau$ — while HMC fell by a factor of 4 and then flattened. Also note that RWMH's tuned acceptance rate converged to $0.234$ as the theory says, a satisfying independent check on the code. And a caution: at $d = 2$ (Worked Example 13.13's banana) HMC's advantage per gradient evaluation was negligible, around 7 ESS per 1000 gradients versus 15 per 1000 densities for a well-tuned random walk. **HMC is not magic in low dimensions; it is indispensable in high ones.**

**NUTS in one paragraph.** Choosing $L$ by hand is fragile: too small and the trajectory does not traverse the distribution, too large and it doubles back, wasting gradients. The No-U-Turn Sampler builds the trajectory by repeatedly doubling it forwards or backwards in time (chosen at random) until the endpoints begin to approach each other — the "U-turn" criterion — and then samples a state from the resulting balanced binary tree using a slice/multinomial scheme designed so that detailed balance still holds. The subtlety, and the reason NUTS took a paper rather than a paragraph to get right, is that the stopping rule depends on the trajectory, so a naive implementation breaks reversibility; the doubling construction with a *symmetric* termination check restores it. NUTS with dual-averaging adaptation of $\epsilon$ is what Stan, PyMC, and NumPyro run by default, and it is the reason you can hand a moderately-sized hierarchical model to a probabilistic programming language and get a usable posterior without tuning anything.

<svg viewBox="0 0 700 380" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Random-walk Metropolis versus Hamiltonian Monte Carlo trajectories on a correlated Gaussian">
  <ellipse cx="180" cy="170" rx="120" ry="34" transform="rotate(-32 180 170)" fill="none" stroke="currentColor" stroke-width="0.8"/>
  <ellipse cx="180" cy="170" rx="80" ry="22" transform="rotate(-32 180 170)" fill="none" stroke="currentColor" stroke-width="0.8"/>
  <ellipse cx="180" cy="170" rx="40" ry="11" transform="rotate(-32 180 170)" fill="none" stroke="currentColor" stroke-width="0.8"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="1.3" points="150,190 158,184 156,186 164,180 162,182 170,176 168,178 174,171 172,174 180,167 178,169 186,163 184,165 190,159 188,161 195,155 193,157 199,151"/>
  <circle cx="150" cy="190" r="3" fill="#ef4444"/>
  <text x="40" y="60" fill="#ef4444" font-size="12">random-walk MH: 18 steps,</text>
  <text x="40" y="76" fill="#ef4444" font-size="12">many rejections (retraced points),</text>
  <text x="40" y="92" fill="#ef4444" font-size="12">net displacement ≈ one step size</text>
  <ellipse cx="520" cy="170" rx="120" ry="34" transform="rotate(-32 520 170)" fill="none" stroke="currentColor" stroke-width="0.8"/>
  <ellipse cx="520" cy="170" rx="80" ry="22" transform="rotate(-32 520 170)" fill="none" stroke="currentColor" stroke-width="0.8"/>
  <ellipse cx="520" cy="170" rx="40" ry="11" transform="rotate(-32 520 170)" fill="none" stroke="currentColor" stroke-width="0.8"/>
  <path d="M446,230 C470,214 486,202 500,192 C520,178 540,164 560,150 C578,138 592,128 604,120" fill="none" stroke="#3b82f6" stroke-width="1.6"/>
  <circle cx="446" cy="230" r="3" fill="#3b82f6"/>
  <circle cx="604" cy="120" r="3.5" fill="#3b82f6"/>
  <circle cx="470" cy="214" r="1.8" fill="#3b82f6"/><circle cx="500" cy="192" r="1.8" fill="#3b82f6"/>
  <circle cx="530" cy="171" r="1.8" fill="#3b82f6"/><circle cx="560" cy="150" r="1.8" fill="#3b82f6"/>
  <circle cx="585" cy="133" r="1.8" fill="#3b82f6"/>
  <text x="392" y="60" fill="#3b82f6" font-size="12">HMC: one trajectory, 18 gradient</text>
  <text x="392" y="76" fill="#3b82f6" font-size="12">evaluations, follows the ridge and</text>
  <text x="392" y="92" fill="#3b82f6" font-size="12">lands far away — accepted</text>
  <text x="40" y="320" fill="currentColor" font-size="11.5">Figure 13.3 — Same budget, same target (a correlated Gaussian; contours shown). The random walk must keep</text>
  <text x="40" y="334" fill="currentColor" font-size="11.5">its step below the narrow width of the ridge, so it diffuses: displacement grows like √t. HMC uses ∇log π to</text>
  <text x="40" y="348" fill="currentColor" font-size="11.5">travel along the ridge, so displacement grows like t, and the Metropolis correction only has to absorb the</text>
  <text x="40" y="362" fill="currentColor" font-size="11.5">leapfrog integrator's O(ε²) energy error. Schematic; measured efficiencies are in Worked Example 13.14.</text>
</svg>

### 13.5.2 Langevin dynamics, ULA, MALA, and SGLD

The overdamped Langevin diffusion is the stochastic differential equation
$$
dX_t = \tfrac12\nabla\log\pi(X_t)\,dt + dW_t ,
$$
with $W$ a standard $d$-dimensional Brownian motion. Its invariance property is the reason it is interesting.

**Proposition 13.26 (Langevin invariance, formal computation).** Suppose $\pi$ is a positive $C^2$ density on $\mathbb{R}^d$ and let $\rho_t$ solve the associated Fokker–Planck equation $\partial_t\rho = -\nabla\!\cdot\!\big(\tfrac12\rho\nabla\log\pi\big) + \tfrac12\Delta\rho$. Then $\rho \equiv \pi$ is a stationary solution.

*Proof.* Substitute $\rho = \pi$. Since $\pi\nabla\log\pi = \nabla\pi$, the drift term is $-\nabla\cdot(\frac12\nabla\pi) = -\frac12\Delta\pi$, which cancels the diffusion term $\frac12\Delta\pi$ exactly, giving $\partial_t \pi = 0$. $\square$

(I call this formal because it verifies stationarity of the PDE rather than proving existence/uniqueness of the diffusion and convergence to $\pi$; those require Itô calculus and a Lyapunov condition, and are flagged in §13.12.)

Discretizing with step $h$ gives the **unadjusted Langevin algorithm (ULA)**:
$$
X_{k+1} = X_k + \tfrac h2 \nabla\log\pi(X_k) + \sqrt h\,\xi_k, \qquad \xi_k \sim N(0,I) \text{ i.i.d.}
$$
Discretization destroys exactness. Adding a Metropolis–Hastings correction with $q(x,\cdot) = N(x + \frac h2\nabla\log\pi(x), hI)$ restores it: that is the **Metropolis-adjusted Langevin algorithm (MALA)**, and by Theorem 13.21 it is exact for any $h$.

**Proposition 13.27 (ULA's bias, exactly, on a Gaussian).** Let $\pi = N(0,1)$, so $\nabla\log\pi(x) = -x$. ULA gives $X_{k+1} = (1-\tfrac h2)X_k + \sqrt h \xi_k$, an AR(1) which for $0<h<4$ has a unique stationary law
$$
N\Big(0, \tfrac{1}{1 - h/4}\Big).
$$
Thus ULA targets a distribution with variance inflated by $1/(1-h/4) = 1 + h/4 + O(h^2)$: the bias is $O(h)$, not $O(h^2)$, and it does not vanish with more samples.

*Proof.* Stationary variance $v$ satisfies $v = (1-h/2)^2 v + h$, so $v(1 - (1-h/2)^2) = h$, i.e. $v(h - h^2/4) = h$ and $v = 1/(1-h/4)$. $\square$

**Worked Example 13.15 (ULA bias and MALA's fix, measured).** 600,000 ULA steps on $N(0,1)$ (50,000 discarded), seed 3:

| $h$ | ULA empirical variance | theory $1/(1-h/4)$ | MALA acceptance | MALA empirical variance |
|---|---|---|---|---|
| 0.1 | 1.0242 | 1.0256 | — | — |
| 0.5 | 1.1445 | 1.1429 | 0.972 | 1.0000 |
| 1.0 | 1.3314 | 1.3333 | 0.921 | 0.9970 |

Proposition 13.27 is confirmed to three decimals, and the point is stark: at $h=1$ ULA reports a posterior standard deviation 15% too large, *no matter how long you run it*, whereas MALA — one extra density evaluation per step — is exact. **Unadjusted samplers trade a controllable statistical error for an uncontrollable systematic one.** In practice this is often a deliberate and defensible trade (see below), but it should be a decision, not an accident.

**SGLD.** Now let $\pi(\theta) \propto p(\theta)\prod_{i=1}^n p(x_i\mid\theta)$ be a posterior over model parameters. The gradient $\nabla\log\pi(\theta) = \nabla\log p(\theta) + \sum_i \nabla\log p(x_i\mid\theta)$ costs a full pass over the data. *Stochastic gradient Langevin dynamics* replaces it with a minibatch estimate:
$$
\theta_{k+1} = \theta_k + \frac{h_k}{2}\Big(\nabla\log p(\theta_k) + \frac nB\sum_{i\in\mathcal B_k}\nabla\log p(x_i\mid\theta_k)\Big) + \sqrt{h_k}\,\xi_k .
$$
With $h_k \to 0$ such that $\sum h_k = \infty$, $\sum h_k^2 <\infty$, the minibatch noise is asymptotically dominated by the injected noise and the algorithm is (under conditions) consistent for $\pi$; with constant $h$ it is not, and it carries both the ULA discretization bias and a minibatch-noise bias.

### 13.5.3 Is SGD secretly a sampler? (What is theorem, what is analogy)

Look at SGD with constant step size $\eta$:
$$
\theta_{k+1} = \theta_k - \eta\,\hat g(\theta_k), \qquad \hat g = \nabla \hat R(\theta) + \text{noise},
$$
and at ULA/SGLD, $\theta_{k+1} = \theta_k - \frac h2\nabla U(\theta_k) + \sqrt h \xi_k$. They have the same shape: a drift down a potential plus noise. The suggestion writes itself — constant-step SGD is a sampler, and the distribution it samples is a *tempered* posterior $\propto e^{-U/T}$ with temperature set by $\eta$ and the gradient-noise covariance. This story is genuinely illuminating and it is also routinely overstated. Let us do the one case where everything is exact.

**Proposition 13.28 (constant-step SGD on a quadratic, exactly).** Let $U(\theta) = \frac a2\theta^2$ with $a>0$ and suppose the stochastic gradient is $\hat g(\theta) = a\theta + s\xi$, $\xi\sim N(0,1)$ i.i.d. (additive, state-independent noise). For $0 < \eta < 2/a$, the iterates $\theta_{k+1} = \theta_k - \eta(a\theta_k + s\xi_k)$ converge in law to
$$
N\Big(0, \; \frac{\eta s^2}{a\,(2-\eta a)}\Big) \;=\; N\Big(0, \frac{\eta s^2}{2a}\big(1 + O(\eta)\big)\Big).
$$

*Proof.* The recursion is AR(1) with coefficient $(1-\eta a)$ and innovation variance $\eta^2s^2$; $|1-\eta a| < 1$ iff $0<\eta<2/a$. Stationary variance: $v = (1-\eta a)^2 v + \eta^2 s^2 \Rightarrow v(2\eta a - \eta^2a^2) = \eta^2s^2 \Rightarrow v = \eta s^2/(a(2-\eta a))$. Gaussianity is inherited because an AR(1) with Gaussian innovations has Gaussian stationary law. $\square$

Comparing with the target $e^{-U/T} \propto N(0, T/a)$, the effective temperature is $T \approx \eta s^2/2$: **larger learning rate and noisier gradients both mean hotter, hence flatter, hence more spread-out stationary distributions.** That is the entire content of "SGD prefers flat minima" in this model, and it is a theorem.

**Worked Example 13.16 (the temperature, measured).** $a = 1$, $s=1$, 250,000 steps with the first 50,000 discarded, seed 13:

| $\eta$ | empirical stationary variance | $\eta s^2/(a(2-\eta a))$ |
|---|---|---|
| 0.02 | 0.01021 | 0.01010 |
| 0.10 | 0.05270 | 0.05263 |
| 0.40 | 0.25154 | 0.25000 |

> **Common Confusion 13.5.** *"So SGD is doing Bayesian inference and I don't need MCMC?"*
>
> Here is the honest ledger. **Theorem:** for a quadratic potential with additive isotropic Gaussian gradient noise, constant-step SGD has an explicit Gaussian stationary law whose covariance scales linearly with $\eta$ (Proposition 13.28). **Also theorem** (harder, and only under strong assumptions): for small $\eta$, SGD's iterates are approximated to $O(\eta)$ by an SDE $d\theta = -\nabla U\,dt + \sqrt{\eta}\,\Sigma(\theta)^{1/2}dW$. **Analogy, not theorem:** that this SDE samples the Bayesian posterior. It does not, in general, for three concrete reasons. (i) The noise covariance $\Sigma(\theta)$ of a minibatch gradient is *state dependent* and *anisotropic* — it is roughly the Fisher information — whereas the Langevin diffusion that has $\pi$ invariant requires isotropic noise, or else a compensating drift term. With state-dependent $\Sigma$ the stationary distribution is generally *not* $\propto e^{-2U/\eta}$ and has no closed form. (ii) SGD lacks the $\frac12$ factor and the injected noise scaling that make the drift-diffusion balance exact; the discretization bias of Proposition 13.27 is present and $O(\eta)$. (iii) Even where a stationary law exists, its relation to a *posterior* requires identifying $U$ with a log-posterior, but the training loss is usually an unregularized or differently-regularized empirical risk, so the "prior" is implicit and unknown. Chapter 12 warned specifically about reading geometric intuition (flatness, volume) as if it were a theorem about generalization; the same caution applies verbatim here. The defensible claim is: *constant-step SGD is a biased sampler of an implicitly defined stationary distribution whose spread grows with $\eta$ and with gradient noise.* Everything stronger requires assumptions you should state.

> **Self-check 13.5.**
> 1. Why does HMC need the momentum flip, given that the momentum is discarded anyway?
> 2. Leapfrog conserves $H$ to $O(\epsilon^2)$ per step. What does the acceptance probability of Definition 13.23 therefore measure?
> 3. In Proposition 13.27, what happens as $h\to4$, and what does that tell you about ULA on a target with a small eigenvalue in its Hessian?
> 4. Give two properties of minibatch gradient noise that break the analogy between SGD and a Langevin sampler.

---

## 13.6 MCMC diagnostics: how to audit a chain

You will never prove that a chain has converged. What you can do is run a battery of checks each of which would likely *fail* if it had not. This asymmetry — diagnostics can refute, not confirm — is the single most important thing to internalize here.

**Definition 13.29 (integrated autocorrelation time and effective sample size).** For a stationary chain and a scalar function $f$, let $\rho_\ell := \operatorname{Corr}(f(X_t), f(X_{t+\ell}))$. The *integrated autocorrelation time* is $\tau_f := 1 + 2\sum_{\ell=1}^\infty \rho_\ell$, and the *effective sample size* of a run of length $T$ is $\mathrm{ESS}_f := T/\tau_f$. By Theorem 13.19, $\operatorname{Var}\big(\frac1T\sum_t f(X_t)\big) \approx \operatorname{Var}_\pi(f)/\mathrm{ESS}_f$: the chain is worth $\mathrm{ESS}_f$ independent draws *for this $f$*.

Estimating $\tau_f$ requires truncating the sum, because the empirical $\hat\rho_\ell$ at large lags is pure noise and adding thousands of noise terms produces garbage. Two standard rules: Geyer's *initial positive sequence* (sum consecutive pairs $\hat\rho_{2m}+\hat\rho_{2m+1}$ until a pair is negative — this is what produced every ESS number in this chapter) and Stan's multi-chain estimator, which pools the autocorrelation estimate across chains and truncates similarly. Note "for this $f$": ESS is functional-specific, and the ESS for a tail quantile is typically far worse than for the mean, which is why modern reports quote both bulk-ESS and tail-ESS.

**Definition 13.30 (split-$\hat R$).** Run $m$ chains of length $2n$ from dispersed starting points; split each in half to get $M = 2m$ sequences of length $n$ with means $\bar f_j$ and variances $s_j^2$. Let $W := \frac1M\sum_j s_j^2$ (within) and $B := \frac{n}{M-1}\sum_j (\bar f_j - \bar f)^2$ (between). Then
$$
\widehat{\operatorname{Var}}^+ := \frac{n-1}{n}W + \frac{B}{n}, \qquad \hat R := \sqrt{\frac{\widehat{\operatorname{Var}}^+}{W}} .
$$

The intuition: $W$ underestimates the target variance if the chains are stuck in different places (each chain has only seen part of the distribution), while $\widehat{\operatorname{Var}}^+$ overestimates it (it includes the between-chain spread). If the chains have mixed, both estimate the same thing and $\hat R \approx 1$. *Splitting* each chain in half catches the case where all chains are drifting in the same direction — a common failure that unsplit $\hat R$ misses entirely. The modern threshold is $\hat R < 1.01$, tightened from the older $1.1$ because $1.1$ was demonstrably too loose.

**Worked Example 13.17 ($\hat R$ catching a badly tuned chain).** Four chains of 20,000 iterations on the banana of Worked Example 13.13, started at $x_1 \in \{0, 20, -20, 5\}$ (seeds 301–304 / 401–404), monitoring $x_1$:

| sampler | split-$\hat R$ for $x_1$ | verdict |
|---|---|---|
| RWMH, $\sigma = 0.2$ | 1.582 | catastrophic — the chains never met |
| RWMH, $\sigma = 2.4$ | 1.021 | borderline; would fail the 1.01 threshold |
| HMC, $\epsilon=0.2, L=20$ | 1.0009 | passes |

Compare with Worked Example 13.13: the $\sigma=0.2$ chain had ESS $\approx 4$, and $\hat R = 1.58$ flags it loudly. But notice the ordering of information: a *single* chain at $\sigma = 0.2$ produces a smooth, plausible-looking trace (Figure 13.2, top) and a sample mean and standard deviation that are not obviously absurd. It takes multiple dispersed starts to expose it. Run at least four chains, always.

**Divergence diagnostics for HMC.** A *divergent transition* is a leapfrog trajectory along which the energy error $|H(z^\ast) - H(z)|$ exceeds a threshold (Stan uses $10^3$), i.e. the integrator became unstable. Divergences are not merely rejected proposals; they are *biased* rejections, because they occur preferentially in regions of high curvature — the neck of a funnel-shaped hierarchical posterior, for instance. A run with divergences concentrated in one region of parameter space is telling you that the sampler cannot enter that region, so the reported posterior is truncated there. The standard responses are to lower $\epsilon$ (raise `adapt_delta`), reparameterize (non-centred parameterization for hierarchical scales), or change the model.

A minimal audit checklist, in the order I would run it: (1) plot the traces of several scalar summaries from $\ge 4$ dispersed chains and look for drift, plateaus, and stuck chains; (2) check split-$\hat R < 1.01$ for every monitored quantity; (3) check bulk- and tail-ESS $> 400$ or so, and report ESS per second and per gradient; (4) for HMC, check zero divergences and inspect any energy-error histogram (Betancourt's E-BFMI); (5) posterior predictive checks — simulate data from the fitted model and compare to the observed data; (6) if the model is small enough, cross-check against a second method (a long MALA run, a Laplace approximation from Ch. 10, or a fine grid in low dimensions).

> **Common Confusion 13.6.** *"$\hat R = 1.00$ and ESS = 5000, so my sample is from the posterior."*
>
> All the diagnostics test *consistency between things you observed*. If every chain is trapped in the same mode — because they were all initialized from the same optimizer solution, or because the modes are separated by a barrier no local proposal can cross — then $W \approx \widehat{\operatorname{Var}}^+$, $\hat R \approx 1$, the autocorrelation decays quickly, and ESS is large. Every diagnostic passes, and the answer is wrong. This is not a hypothetical: it is the normal state of affairs for Bayesian neural networks, where the posterior has an enormous number of symmetry-equivalent modes (permutations of hidden units) plus genuinely inequivalent ones, and no sampler in existence explores them all. The honest reading of a passing diagnostic suite is: *no evidence of non-convergence was found*, in exactly the sense of a failure to reject a null hypothesis (Ch. 9). Multi-modality is detected by dispersed initialization and by comparing independent runs, never by within-run statistics alone.

> **Self-check 13.6.**
> 1. Your chain has ESS 2000 for the posterior mean and 90 for the 97.5% quantile. Which number do you report next to a credible interval, and why are they different?
> 2. Explain in one sentence why splitting each chain in half makes $\hat R$ strictly more sensitive.
> 3. Why are HMC divergences worse than ordinary rejections?
> 4. Give a concrete situation in which every diagnostic in the checklist passes and the posterior sample is nevertheless badly wrong.

---

## 13.7 Variational inference: the ELBO from three directions

MCMC is asymptotically exact and computationally open-ended: you can always run longer. Variational inference makes the opposite trade — it is fast, it stops when the optimizer stops, and it is *wrong by an amount you cannot measure*. Given how central it is to modern generative modelling, it is worth deriving its objective three times, because the three derivations license different things.

Setup: a joint model $p(x,z) = p(x\mid z)p(z)$ with observed $x$ and latent $z$ (a per-datapoint latent, a parameter vector, or both). The quantities we want are the *evidence* $p(x) = \int p(x,z)\,dz$ and the *posterior* $p(z\mid x) = p(x,z)/p(x)$. Both are intractable. Fix a family $\mathcal Q$ of distributions over $z$ (all with densities with respect to a common base measure, per Chapter 2's discipline) and define, for $q \in \mathcal Q$,
$$
\mathcal L(q) := \mathbb{E}_{q}\big[\log p(x,Z) - \log q(Z)\big],
$$
with the convention $\log 0 = -\infty$, so that $\mathcal L(q) = -\infty$ whenever $q$ puts mass where $p(x,\cdot)$ vanishes.

**Theorem 13.31 (three derivations of the same object).** Assume the expectations below exist (possibly $-\infty$). Then:

**(a) Jensen.** $\log p(x) \ge \mathcal L(q)$ for *every* $q$. When $p(\cdot\mid x) \ll q$ — the proposal covers the posterior, which is §13.2's target-$\ll$-proposal condition with $q$ in the role of $Q$ — the bound comes from writing $\log p(x) = \log \mathbb{E}_q[p(x,Z)/q(Z)]$ and applying Jensen's inequality to the concave $\log$; without that condition the bound still holds, but with extra slack (see the proof).

**(b) Exact KL decomposition.** If $q \ll p(\cdot\mid x)$, then $\displaystyle \log p(x) = \mathcal L(q) + D_{\mathrm{KL}}\big(q \,\|\, p(\cdot\mid x)\big)$ — an identity, not an inequality. (If $q \not\ll p(\cdot\mid x)$ then $D_{\mathrm{KL}} = +\infty$ and $\mathcal L(q) = -\infty$, the identity degenerates to the uninformative $-\infty + \infty$, and only the bound in (a) survives.)

**(c) EM with an inexact E-step.** For a model $p_\theta(x,z)$, $\mathcal L(q,\theta) = \mathbb{E}_q[\log p_\theta(x,Z)] + H(q)$, and coordinate ascent in $(q,\theta)$ with the $q$-step solved *exactly* is precisely the EM algorithm (Ch. 8), whose E-step sets $q = p_\theta(\cdot\mid x)$ and whose M-step maximizes $\mathbb{E}_q[\log p_\theta]$.

*Proof.* (a) With $w(z) := p(x,z)/q(z)$ (well defined $q$-a.e., since $q > 0$ holds $q$-a.e. automatically),
$$
\mathbb{E}_q[w] = \int_{\{q > 0\}} p(x,z)\,dz \;\le\; \int p(x,z)\,dz = p(x),
$$
with *equality* if and only if $p(x,\cdot)$ puts no mass on $\{q = 0\}$, i.e. iff $p(\cdot\mid x) \ll q$. This is the importance-sampling identity of Ch. 2, and note carefully which way the absolute continuity points: **target $\ll$ proposal**, the same direction as $P \ll Q$ in §13.2 — not the reverse. Jensen for the concave $\log$ then gives
$$
\mathcal L(q) = \mathbb{E}_q[\log w] \;\le\; \log \mathbb{E}_q[w] \;\le\; \log p(x),
$$
so the lower bound holds unconditionally (if $q$ puts mass where $p(x,\cdot) = 0$, then $\mathcal L(q) = -\infty$ and the bound is trivial). When $p(\cdot\mid x) \ll q$ the second inequality is an equality and Jensen's inequality is the *only* source of the gap — which is what part (b) quantifies.

(b) Under $q \ll p(\cdot\mid x)$ the ratio $q(z)/p(z \mid x)$ is finite $q$-a.e. and the following manipulation is legitimate. Compute directly:
$$
D_{\mathrm{KL}}(q\|p(\cdot\mid x)) = \mathbb{E}_q\Big[\log\frac{q(Z)}{p(Z\mid x)}\Big] = \mathbb{E}_q\Big[\log\frac{q(Z)p(x)}{p(x,Z)}\Big] = \log p(x) - \mathcal L(q),
$$
where $\log p(x)$ passes through the expectation because it does not depend on $z$. Rearranging gives (b), and since $D_{\mathrm{KL}}\ge0$ (Ch. 11) this *re-proves* (a) with the gap identified.

(c) Substituting $\log p_\theta(x,Z) - \log q(Z)$ and splitting the expectation gives the stated form, with $H(q) = -\mathbb{E}_q\log q$ the differential entropy, which does not involve $\theta$. Maximizing over $q$ with $\theta$ fixed: by (b), $\mathcal L(q,\theta) = \log p_\theta(x) - D_{\mathrm{KL}}(q\|p_\theta(\cdot\mid x))$, maximized at $q = p_\theta(\cdot\mid x)$ with value $\log p_\theta(x)$ — the E-step. Maximizing over $\theta$ with $q$ fixed drops $H(q)$ and gives the M-step objective $\mathbb{E}_q[\log p_\theta(x,Z)]$. $\square$

Why keep all three? **(a)** generalizes: replace the single-sample ratio by a $K$-sample average and Jensen still applies, giving the tighter IWAE bound of §13.8. **(b)** tells you the *size and direction* of the error: the gap is exactly $D_{\mathrm{KL}}(q\|p(\cdot\mid x))$ — the *reverse* KL, whose mode-seeking behaviour (Ch. 11) predicts every characteristic pathology of VI. **(c)** tells you that VI is not a new idea but a relaxation of an old one, and that monotone ascent is available whenever a subproblem can be solved exactly.

One hypothesis is worth keeping straight, because the two absolute-continuity conditions above point in *opposite* directions and each belongs to a different part. Part (a)'s equality $\mathbb{E}_q[w] = p(x)$ needs $p(\cdot\mid x) \ll q$: if $q$ misses a region where the posterior has mass, $\mathbb{E}_q[w]$ falls strictly below $p(x)$ and the bound acquires slack that no amount of optimization can remove. Part (b) needs $q \ll p(\cdot\mid x)$: if $q$ puts mass where the posterior has none, the reverse KL is $+\infty$ and the ELBO is $-\infty$ — the objective assigns *infinite* penalty to overcovering the posterior and only finite penalty to undercovering it, which is the mode-seeking, zero-forcing behaviour of §13.7.2 and Chapter 11 stated as a support condition. For the families you will usually meet — a full-support Gaussian $q$ against a full-support posterior — both conditions hold trivially. For constrained supports (truncated likelihoods, mixtures with hard assignments, discrete latents with pruned states) checking the direction is not pedantry; it decides whether your ELBO is even finite.

<svg viewBox="0 0 700 250" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The ELBO and the evidence on a single vertical axis, with the gap equal to a KL divergence">
  <line x1="120" y1="30" x2="120" y2="200" stroke="currentColor" stroke-width="1.5"/>
  <line x1="105" y1="50" x2="330" y2="50" stroke="currentColor" stroke-width="2"/>
  <text x="340" y="54" fill="currentColor" font-size="12">log p(x) = −19.8160  (the evidence; a fixed number)</text>
  <line x1="105" y1="120" x2="330" y2="120" stroke="#3b82f6" stroke-width="2"/>
  <text x="340" y="124" fill="#3b82f6" font-size="12">ℒ(q) = −19.8367  (ELBO at the CAVI optimum)</text>
  <line x1="105" y1="175" x2="330" y2="175" stroke="#3b82f6" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="340" y="179" fill="#3b82f6" font-size="12">ℒ(q) = −19.9638  (ELBO after CAVI iteration 1)</text>
  <line x1="200" y1="50" x2="200" y2="120" stroke="#ef4444" stroke-width="1.5"/>
  <path d="M200,50 l-4,8 l8,0 z" fill="#ef4444"/><path d="M200,120 l-4,-8 l8,0 z" fill="#ef4444"/>
  <text x="206" y="90" fill="#ef4444" font-size="12">gap = D_KL(q ‖ p(·|x)) = 0.0207 nats</text>
  <text x="60" y="30" fill="currentColor" font-size="12">higher</text>
  <text x="60" y="205" fill="currentColor" font-size="12">lower</text>
  <text x="60" y="226" fill="currentColor" font-size="11.5">Figure 13.4 — Maximizing the ELBO pushes the blue line up toward the fixed black line; the remaining gap IS the</text>
  <text x="60" y="240" fill="currentColor" font-size="11.5">KL from q to the true posterior. Numbers are the measured values of Worked Example 13.18, in nats.</text>
</svg>

### 13.7.1 Mean field and CAVI

**Definition 13.32 (mean-field family).** Partition the latent vector as $z = (z_1,\dots,z_m)$. The *mean-field family* is $\mathcal Q_{\mathrm{MF}} = \{q : q(z) = \prod_{j=1}^m q_j(z_j)\}$, with each factor unrestricted. *Structured* variational families keep some dependencies (a tree, a chain, a low-rank Gaussian).

**Proposition 13.33 (CAVI update and monotone ascent).** Fix all factors but the $j$th. Then $\mathcal L$ is maximized over $q_j$ by
$$
q_j^\star(z_j) \;\propto\; \exp\Big(\mathbb{E}_{q_{-j}}\big[\log p(x, z)\big]\Big),
$$
where $q_{-j} = \prod_{k\ne j}q_k$, provided the right-hand side is normalizable. Iterating over $j$ produces a non-decreasing sequence of ELBO values, which therefore converges (to a local optimum, not necessarily the global one).

*Proof.* Write $\mathcal L(q) = \mathbb{E}_q[\log p(x,Z)] - \sum_k \mathbb{E}_{q_k}[\log q_k]$. As a functional of $q_j$ with the others fixed,
$$
\mathcal L = \int q_j(z_j)\,\underbrace{\mathbb{E}_{q_{-j}}[\log p(x,z)]}_{=: \log \tilde f(z_j) + c}\,dz_j - \int q_j\log q_j \,dz_j + \text{const}
= -D_{\mathrm{KL}}\big(q_j \,\|\, \tilde f / Z_{\tilde f}\big) + \text{const},
$$
where $\tilde f(z_j) := \exp\mathbb{E}_{q_{-j}}[\log p(x,z)]$ and $Z_{\tilde f} = \int\tilde f$. Since KL is minimized uniquely at equality (Ch. 11), the maximizer is $q_j = \tilde f/Z_{\tilde f}$. Each coordinate step weakly increases $\mathcal L$ by construction, and a bounded monotone sequence converges. $\square$

Two remarks. First, the update is "exponentiate the expected log joint", and for conditionally conjugate models it has a closed form: if $p(z_j \mid z_{-j}, x)$ is in an exponential family, then $q_j^\star$ is in the same family with natural parameter equal to the *expectation* of the conditional's natural parameter. Second, the ELBO increases monotonically, which gives you a free correctness test on your implementation — if your ELBO ever decreases, you have a bug. Use it.

**Worked Example 13.18 (CAVI for a Normal–Gamma model, with the ELBO audited).** Model: $x_i \mid \mu,\tau \sim N(\mu, \tau^{-1})$ i.i.d., $\mu\mid\tau \sim N(\mu_0, (\lambda_0\tau)^{-1})$, $\tau\sim\mathrm{Gamma}(a_0,b_0)$. This is conjugate, so the exact posterior is Normal–Gamma and $\log p(x)$ is available in closed form — perfect for measuring the VI gap. Mean-field family $q(\mu,\tau) = q(\mu)q(\tau)$ with, by Proposition 13.33,
$$
q(\mu) = N(\mu_n, \lambda_n^{-1}), \quad \lambda_n = (\lambda_0 + n)\mathbb{E}_q[\tau], \quad \mu_n = \frac{\lambda_0\mu_0 + n\bar x}{\lambda_0+n},
$$
$$
q(\tau) = \mathrm{Gamma}(a_n, b_n), \quad a_n = a_0 + \tfrac{n+1}{2}, \quad b_n = b_0 + \tfrac12\mathbb{E}_q\Big[\textstyle\sum_i (x_i - \mu)^2 + \lambda_0(\mu-\mu_0)^2\Big].
$$
Note the structure: $q(\mu)$ needs $\mathbb{E}_q[\tau] = a_n/b_n$, and $q(\tau)$ needs $\mathbb{E}_q[\mu]$ and $\mathbb{E}_q[\mu^2]$; the two updates chase each other. With $n=20$ points simulated from $N(2, 0.5^2)$ (seed 5, giving $\bar x = 1.9031$, $\sum(x_i-\bar x)^2 = 4.3629$) and hyperparameters $\mu_0 = 0, \lambda_0 = 0.1, a_0 = b_0 = 2$:

| iteration | ELBO | $\mu_n$ | $\operatorname{Var}_q(\mu) = 1/\lambda_n$ | $\mathbb{E}_q[\tau]$ |
|---|---|---|---|---|
| 1 | $-19.96383110$ | 1.89367 | 0.007960 | 2.81428 |
| 2 | $-19.83677331$ | 1.89367 | 0.017678 | 2.75373 |
| 3 | $-19.83665127$ | 1.89367 | 0.018067 | 2.75137 |
| 4 | $-19.83665108$ | 1.89367 | 0.018082 | 2.75127 |
| 5–8 | $-19.83665108$ | 1.89367 | 0.018083 | 2.75127 |

The ELBO increases at every step and converges in four iterations, as Proposition 13.33 requires. The exact evidence is $\log p(x) = -19.81596249$, so
$$
D_{\mathrm{KL}}\big(q\|p(\cdot\mid x)\big) = 0.02068859 \text{ nats},
$$
a genuinely small gap — mean field is doing well here because the posterior dependence between $\mu$ and $\tau$ is weak at $n=20$. But look at the variance: $\operatorname{Var}_q(\mu) = 0.018083$ against the exact $\operatorname{Var}(\mu\mid x) = b_n/((a_n-1)\lambda_n) = 0.019727$, a ratio of $0.9167$. **VI got the location right to five decimals and understated the uncertainty by 8%.** That pattern — good means, bad variances — is the signature of variational inference, and the next proposition explains it.

### 13.7.2 Why VI understates uncertainty

**Proposition 13.34 (mean-field on a correlated Gaussian).** Let $p = N(0,\Sigma)$ on $\mathbb R^2$ with $\Sigma = \begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}$, $|\rho|<1$, and let $q = q_1\otimes q_2$ minimize $D_{\mathrm{KL}}(q\|p)$ over product distributions. Then $q = N(0, (1-\rho^2)I_2)$, so each marginal variance is $1-\rho^2$ instead of the true 1, and
$$
D_{\mathrm{KL}}(q\|p) = -\tfrac12\log(1-\rho^2).
$$

*Proof.* Let $\Lambda := \Sigma^{-1} = \frac{1}{1-\rho^2}\begin{pmatrix}1&-\rho\\-\rho&1\end{pmatrix}$. Apply Proposition 13.33 with $p$ in place of $p(x,\cdot)$: $\log p(z) = -\frac12 z^\top\Lambda z + c$, so
$$
\log q_1^\star(z_1) = \mathbb{E}_{q_2}\big[-\tfrac12\Lambda_{11}z_1^2 - \Lambda_{12}z_1 Z_2\big] + c = -\tfrac12\Lambda_{11}z_1^2 - \Lambda_{12}z_1\mathbb{E}_{q_2}[Z_2] + c .
$$
By symmetry the fixed point has $\mathbb{E}[Z_2] = 0$, giving $q_1^\star = N(0, \Lambda_{11}^{-1}) = N(0, 1-\rho^2)$, and similarly for $q_2$. For the KL, with $D := (1-\rho^2)I$,
$$
2D_{\mathrm{KL}}(q\|p) = \operatorname{tr}(\Lambda D) - 2 + \log\frac{\det\Sigma}{\det D} = 2 - 2 + \log\frac{1-\rho^2}{(1-\rho^2)^2} = -\log(1-\rho^2). \square
$$

The mechanism is now transparent: **mean field replaces each marginal variance by the reciprocal of the corresponding precision-matrix diagonal, i.e. by the *conditional* variance rather than the marginal one.** Conditioning on the other coordinates always shrinks variance, so mean-field VI is *systematically* overconfident, and the more correlated the posterior, the worse:

**Worked Example 13.19 (how bad it gets).**

| $\rho$ | $\operatorname{Var}_q$ (true = 1) | $D_{\mathrm{KL}}(q\|p)$ |
|---|---|---|
| 0.50 | 0.7500 | 0.1438 nats |
| 0.90 | 0.1900 | 0.8304 nats |
| 0.99 | 0.0199 | 1.9585 nats |

At $\rho = 0.99$ the reported posterior standard deviation is $\sqrt{0.0199} = 0.141$ against the true 1: a credible interval seven times too narrow. Since strong posterior correlations are the norm in hierarchical models and in over-parameterized networks (where the posterior has flat directions and strongly coupled ones), the practical rule is: **treat mean-field posterior variances as lower bounds, and if the uncertainty matters, use a structured or full-covariance family, or check against MCMC.** Note the direction is a consequence of using *reverse* KL; forward KL ($D_{\mathrm{KL}}(p\|q)$, as in expectation propagation) over-estimates variance instead, matching moments rather than modes. Chapter 11's asymmetry discussion is not an academic point — it determines the sign of your error bars.

### 13.7.3 Stochastic and natural-gradient VI

Two extensions matter for scale. **Stochastic VI** notes that for $n$ i.i.d. datapoints with local latents $z_i$ and global parameters $\beta$, the ELBO is a sum over datapoints, so an unbiased gradient can be formed from a minibatch: subsample $B$ points, optimize their local factors, and take a scaled step on the global factor. This turns VI into SGD (Ch. 6's convergence theory applies) and is what made VI usable for topic models on millions of documents. **Natural-gradient VI** notes that the ELBO is a function on a manifold of distributions, and that the Euclidean gradient in the variational parameters is basis-dependent while the natural gradient — the gradient preconditioned by the inverse Fisher information of $q$ (Ch. 10) — is not. For conditionally conjugate models the natural gradient in the natural-parameter coordinates has a strikingly simple form: it is *the CAVI update minus the current value*. So CAVI is natural-gradient ascent with step size 1, and stochastic natural-gradient VI is CAVI with minibatches and a decaying step size. That identification is the cleanest justification I know for why CAVI converges in a handful of iterations while gradient ascent on the same objective can crawl.

> **Self-check 13.7.**
> 1. From Theorem 13.31(b), what happens to the ELBO if you enlarge $\mathcal Q$? Can the gap increase?
> 2. In Proposition 13.33, where is conjugacy used, and what breaks without it?
> 3. Your ELBO decreased at iteration 12 of CAVI. What do you conclude?
> 4. Mean-field VI reports a posterior standard deviation of 0.01 for a parameter. Give two reasons this could be far too small, and say which chapter's result predicts the direction of the error.

---

## 13.8 Variational autoencoders

### 13.8.1 Amortization: one network instead of $n$ optimizations

In §13.7 the variational parameters belonged to a single dataset. In a latent-variable *generative* model there is one latent $z_i$ per datapoint, so mean-field VI would require optimizing a separate $q_i$ for each of $n$ datapoints — and re-optimizing from scratch for every new test point. The variational autoencoder's central move is to *amortize*: train one network $x \mapsto (\mu_\phi(x), \sigma^2_\phi(x))$ that outputs the variational parameters as a function of the datapoint.

**Definition 13.35 (VAE).** Fix a latent dimension $k$ and a prior $p(z) = N(0,I_k)$. The *decoder* is a conditional density $p_\theta(x\mid z)$ (e.g. $N(x; g_\theta(z), \sigma^2 I_d)$ or a product of Bernoullis). The *encoder* defines an amortized variational family
$$
q_\phi(z\mid x) := N\big(z;\ \mu_\phi(x),\ \operatorname{diag}\sigma^2_\phi(x)\big),
$$
and the training objective is the ELBO summed over data,
$$
\mathcal L(\theta,\phi) = \sum_{i=1}^n \Big\{\underbrace{\mathbb{E}_{q_\phi(z\mid x_i)}\big[\log p_\theta(x_i \mid Z)\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(\cdot\mid x_i)\,\|\,p\big)}_{\text{regularization}}\Big\},
$$
maximized jointly over $\theta$ and $\phi$ by stochastic gradient ascent.

Amortization costs something. The optimal $q$ for datapoint $x_i$ is $p_\theta(z\mid x_i)$; the best the encoder can do is the best *function* of $x$ in its hypothesis class, so the total gap decomposes as
$$
\log p_\theta(x_i) - \mathcal L_i = \underbrace{D_{\mathrm{KL}}\big(q^\star_i\|p_\theta(\cdot\mid x_i)\big)}_{\text{approximation gap (family too small)}} + \underbrace{D_{\mathrm{KL}}\big(q_\phi(\cdot\mid x_i)\|p_\theta(\cdot\mid x_i)\big) - D_{\mathrm{KL}}\big(q_i^\star\|p_\theta(\cdot\mid x_i)\big)}_{\text{amortization gap}} ,
$$
where $q_i^\star$ is the best member of the family for that point. Both are nonnegative, and both are invisible from the training curve, which shows only their sum plus $\log p_\theta(x_i)$.

**Proposition 13.36 (the reparameterized gradient).** With $q_\phi(z\mid x) = N(\mu_\phi(x), \operatorname{diag}\sigma^2_\phi(x))$, write $Z = T_\phi(x,\varepsilon) := \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$ with $\varepsilon\sim N(0,I_k)$. Then for any $f$ with $\mathbb{E}\|\nabla_\phi f(T_\phi(x,\varepsilon))\|<\infty$ and satisfying the domination hypothesis of Ch. 2,
$$
\nabla_\phi\,\mathbb{E}_{q_\phi}\big[f(Z)\big] = \mathbb{E}_{\varepsilon\sim N(0,I)}\big[\nabla_\phi f\big(T_\phi(x,\varepsilon)\big)\big],
$$
and the single-sample estimator $\nabla_\phi f(T_\phi(x,\varepsilon))$ is unbiased.

*Proof.* Pushforward (Definition 1.16) rewrites the expectation over the $\phi$-dependent measure $q_\phi$ as an expectation over the $\phi$-*independent* measure $N(0,I)$: $\mathbb{E}_{q_\phi}[f(Z)] = \mathbb{E}_\varepsilon[f(T_\phi(x,\varepsilon))]$. Now the parameter appears only inside the integrand, so differentiating under the expectation is licensed by the Dominated Convergence Theorem argument of Ch. 2. Unbiasedness is then immediate. $\square$

This is the technical heart of the VAE and it is worth naming what makes it work: **the randomness was moved out of the thing being differentiated and into a fixed noise source** — which is exactly Chapter 1's insistence that a random variable is a deterministic function of $\omega$, with $\omega$ the seed. The alternative estimator, REINFORCE (Ch. 2), is also unbiased but has variance typically one to three orders of magnitude larger, because it does not use $\nabla f$ at all; the reparameterized estimator additionally enjoys common random numbers across the parameter update (§13.2.4). The Gaussian KL term is computed analytically rather than sampled — Rao–Blackwellization again (Proposition 13.10's mechanism), and for diagonal Gaussians it is the familiar
$$
D_{\mathrm{KL}}\big(N(\mu,\operatorname{diag}\sigma^2)\,\|\,N(0,I)\big) = \tfrac12\sum_{j=1}^k\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big).
$$

<svg viewBox="0 0 700 300" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="VAE computation graph with the reparameterized gradient path marked">
  <rect x="30" y="110" width="70" height="40" rx="4" fill="none" stroke="currentColor" stroke-width="1.4"/>
  <text x="56" y="135" fill="currentColor" font-size="13">x</text>
  <rect x="130" y="110" width="90" height="40" rx="4" fill="none" stroke="currentColor" stroke-width="1.4"/>
  <text x="146" y="135" fill="currentColor" font-size="12">encoder φ</text>
  <text x="248" y="108" fill="currentColor" font-size="12">μ(x)</text>
  <text x="244" y="166" fill="currentColor" font-size="12">σ(x)</text>
  <circle cx="330" cy="130" r="20" fill="none" stroke="#3b82f6" stroke-width="1.6"/>
  <text x="320" y="135" fill="#3b82f6" font-size="12">z</text>
  <rect x="300" y="222" width="60" height="34" rx="4" fill="none" stroke="#10b981" stroke-width="1.4"/>
  <text x="308" y="244" fill="#10b981" font-size="12">ε ~ N(0,I)</text>
  <rect x="410" y="110" width="90" height="40" rx="4" fill="none" stroke="currentColor" stroke-width="1.4"/>
  <text x="424" y="135" fill="currentColor" font-size="12">decoder θ</text>
  <rect x="540" y="110" width="130" height="40" rx="4" fill="none" stroke="currentColor" stroke-width="1.4"/>
  <text x="550" y="135" fill="currentColor" font-size="12">log p(x|z)</text>
  <line x1="100" y1="130" x2="130" y2="130" stroke="currentColor" stroke-width="1.2"/>
  <line x1="220" y1="120" x2="290" y2="112" stroke="currentColor" stroke-width="1.2"/>
  <line x1="220" y1="142" x2="290" y2="152" stroke="currentColor" stroke-width="1.2"/>
  <line x1="330" y1="222" x2="330" y2="152" stroke="#10b981" stroke-width="1.4" stroke-dasharray="4 3"/>
  <line x1="350" y1="130" x2="410" y2="130" stroke="currentColor" stroke-width="1.2"/>
  <line x1="500" y1="130" x2="540" y2="130" stroke="currentColor" stroke-width="1.2"/>
  <path d="M560,100 C480,60 300,58 200,100" fill="none" stroke="#ef4444" stroke-width="2"/>
  <path d="M200,100 l10,-3 l-2,8 z" fill="#ef4444"/>
  <text x="300" y="52" fill="#ef4444" font-size="12">gradient flows back THROUGH z = μ(x) + σ(x)⊙ε</text>
  <text x="360" y="240" fill="#10b981" font-size="12">noise enters here, not on the path to φ</text>
  <text x="30" y="278" fill="currentColor" font-size="11.5">Figure 13.5 — The VAE computation graph. Because z is written as a deterministic function of (x, φ, ε), the red</text>
  <text x="30" y="292" fill="currentColor" font-size="11.5">gradient path is an ordinary backprop path (Proposition 13.36); the green stochastic input is parameter-free.</text>
</svg>

### 13.8.2 The honest failure catalogue

**Posterior collapse.** The most-reported pathology: for some or all latent coordinates, $q_\phi(z_j\mid x)$ becomes identical to the prior for every $x$ — the KL term for that coordinate goes to zero, the coordinate carries no information about $x$, and the decoder ignores it. The mechanism is visible in the objective.

**Proposition 13.37 (collapse is an optimum, not a bug).** Suppose the decoder is such that $p_\theta(x \mid z) = p_\theta(x)$ does not depend on $z$ (for instance, the decoder weights on $z$ are zero). Then for the $\beta$-weighted objective $\mathcal L_\beta = \mathbb{E}_q[\log p_\theta(x\mid Z)] - \beta D_{\mathrm{KL}}(q\|p)$ with $\beta>0$, the optimal encoder is $q_\phi(\cdot\mid x) = p(\cdot)$ for every $x$, giving KL $=0$. Conversely, denoting by $I_q(X;Z)$ the mutual information under the joint $\hat P_n(x)q_\phi(z\mid x)$, we have the bound
$$
\mathbb{E}_{\hat P_n}\big[D_{\mathrm{KL}}(q_\phi(\cdot\mid X)\,\|\,p)\big] \;=\; I_q(X;Z) + D_{\mathrm{KL}}\big(q_\phi(z)\,\|\,p(z)\big) \;\ge\; I_q(X;Z),
$$
where $q_\phi(z) = \mathbb{E}_{\hat P_n}[q_\phi(z\mid X)]$ is the aggregate posterior. Hence driving the average KL to zero forces $I_q(X;Z) = 0$: the latent becomes independent of the data.

*Proof.* The first claim: with $z$ absent from the reconstruction term, $\mathcal L_\beta$ depends on $q$ only through $-\beta D_{\mathrm{KL}}(q\|p) \le 0$, maximized uniquely at $q = p$. For the identity, expand
$$
\mathbb{E}_{\hat P_n} D_{\mathrm{KL}}(q_\phi(\cdot\mid X)\|p) = \mathbb{E}_{\hat P_n}\mathbb{E}_{q_\phi(z\mid X)}\Big[\log\frac{q_\phi(z\mid X)}{q_\phi(z)} + \log\frac{q_\phi(z)}{p(z)}\Big] = I_q(X;Z) + D_{\mathrm{KL}}(q_\phi\|p),
$$
the first term being the definition of mutual information (Ch. 11) and the second being nonnegative. $\square$

So the KL term in the ELBO is *an information bottleneck*: it charges $I_q(X;Z)$ nats for every nat of information the latent carries. If the decoder is powerful enough to model $x$ without help (an autoregressive decoder on images or text is the classic case), the cheapest solution is to pay nothing and use nothing. Note also the second term: it says the ELBO also penalizes mismatch between the *aggregate* posterior and the prior, which is the "holes in the latent space" phenomenon — regions with prior mass but no encoder mass, from which samples look wrong.

**Worked Example 13.20 (inducing posterior collapse on purpose).** I trained a small VAE in NumPy (encoder $2\to32\to(\mu,\log\sigma^2)$ with $k=1$; decoder $1\to32\to2$; $\tanh$ activations; Gaussian decoder with fixed $\sigma = 0.1$; Adam, $3\times10^{-3}$, 8000 steps, batch 128, seed 0) on $n=1024$ points from the unit circle plus $0.05$ noise — Chapter 1's circle generator, now as data. Gradients were verified against central finite differences (max relative error $1.3\times10^{-8}$). Sweeping $\beta$:

| $\beta$ | $\mathbb{E}[-\log p_\theta(x\mid z)]$ (reconstruction, nats) | $D_{\mathrm{KL}}(q\|p)$ (nats) | ELBO |
|---|---|---|---|
| 0 | $-0.5512$ | 7.4096 | $-6.8584$ |
| 1 | $-0.1836$ | 2.9354 | $-2.7518$ |
| 4 | $1.7610$ | 2.1592 | $-3.9202$ |
| 20 | $11.5260$ | 1.1040 | $-12.6300$ |
| 100 | $47.3141$ | $0.000208$ | $-47.3143$ |

At $\beta = 100$ the KL is $2\times10^{-4}$ nats: total collapse. And the reconstruction term tells us what the model has become — a constant-output decoder that predicts the data mean would score $47.1927$, and the collapsed model scores $47.3141$. **The trained network is, to within numerical noise, a machine that ignores its input and outputs the mean of the data.** Meanwhile at $\beta = 0$ the KL is unbounded (7.4 nats and rising) and the "posterior" for each point is a spike: reconstruction is excellent, sampling from the prior is meaningless. The $\beta$ dial is precisely a rate–distortion trade-off (Ch. 11): rate $= I_q(X;Z) \le$ KL, distortion $=$ reconstruction loss, and $\beta$ selects a point on the frontier. There is no $\beta$ that is "correct" — there is a curve, and you choose a point on it according to what you want the latent for.

**Worked Example 13.21 (why the samples have a seam: topology, not optimization).** Look at the trained $\beta=1$ decoder along the latent axis, printing the decoder mean $g_\theta(z)$ and its radius:

| $z$ | $g_\theta(z)$ | radius |
|---|---|---|
| $-3$ | $(-0.905, -0.005)$ | 0.905 |
| $-2$ | $(-0.983, +0.021)$ | 0.983 |
| $-1$ | $(-0.896, -0.390)$ | 0.977 |
| $0$ | $(+0.571, +0.205)$ | **0.607** |
| $+1$ | $(+0.464, +0.881)$ | 0.995 |
| $+2$ | $(+0.855, +0.420)$ | 0.952 |
| $+3$ | $(+0.857, +0.307)$ | 0.910 |

Six of the seven points sit on the circle to within a few percent. One, at $z = 0$, has radius $0.607$ — it is *inside* the annulus, in a region with no data. This is not undertraining; it is a theorem. The decoder $g_\theta : \mathbb R \to \mathbb R^2$ is continuous, and $\mathbb{R}$ is not homeomorphic to the circle $S^1$; a continuous surjection from a line onto a circle must traverse the circle and come back, and any continuous map from a connected line whose image is close to $S^1$ must cross the interior somewhere or fail to cover. So a 1-dimensional-latent VAE with a continuous decoder *cannot* place all of its prior mass on the circle: there is always a seam, and samples drawn from prior mass near the seam land off the data manifold. Chapter 1 introduced $G(t) = (\cos t, \sin t)$ to show that a pushforward of a low-dimensional latent has Lebesgue-null support; the same example now shows that even matching the support requires the *topology* of the latent to be compatible. In practice this is one honest source of "bad samples" that no amount of training fixes, and it is why practitioners over-provision $k$.

**Blurry samples.** With decoder $p_\theta(x\mid z) = N(x; g_\theta(z),\sigma^2 I)$ the model distribution is a *convolution*: $p_\theta = \text{law of } g_\theta(Z) + \sigma\varepsilon$. Two consequences. First, for fixed $\sigma$ the reconstruction term is $\|x - g_\theta(z)\|^2/(2\sigma^2) + \text{const}$ — a squared error, whose minimizer given an ambiguous $z$ is a conditional *mean*, and the average of several plausible sharp images is a blurry image. Second, the model literally adds Gaussian noise at sample time; if you display $g_\theta(z)$ instead of a draw from $p_\theta(x\mid z)$, you are showing the conditional mean, not a sample from the model, which is standard practice and mildly dishonest. The rate–distortion reading (Ch. 11) is the sharpest one: at a fixed rate the optimal reconstruction under squared-error distortion *is* blurred, so blurriness is a property of the loss and the rate, not a failure of the network. Changing the decoder likelihood (discretized logistic mixtures, autoregressive decoders, adversarial or perceptual losses) changes the distortion measure and hence the character of the artifacts.

**IWAE: a tighter bound at a price.**

**Proposition 13.38 (importance-weighted ELBO).** For $K \ge 1$ define
$$
\mathcal L_K := \mathbb{E}_{Z_1,\dots,Z_K \sim q}\Big[\log\frac1K\sum_{j=1}^K \frac{p(x,Z_j)}{q(Z_j)}\Big].
$$
Then $\mathcal L_1 = \mathcal L(q)$, $\mathcal L_K \le \mathcal L_{K+1} \le \log p(x)$ for all $K$, and $\mathcal L_K \to \log p(x)$ as $K\to\infty$ if the weights are bounded in an appropriate sense.

*Proof of the bound and monotonicity.* Let $w_j := p(x,Z_j)/q(Z_j)$, so $\mathbb{E}[w_j] = p(x)$. Jensen gives $\mathcal L_K = \mathbb{E}[\log \bar w_K] \le \log\mathbb{E}[\bar w_K] = \log p(x)$ where $\bar w_K$ is the average. For monotonicity, note that $\bar w_K$ is a "more averaged" version of $\bar w_{K+1}$: precisely, conditioning on $(w_1,\dots,w_{K+1})$ and letting $I$ be a uniformly random subset of size $K$,
$$
\mathbb{E}_I\big[\bar w_I\big] = \bar w_{K+1},
$$
so by Jensen applied conditionally, $\mathbb{E}\log \bar w_{K+1} = \mathbb{E}\log \mathbb{E}_I[\bar w_I] \ge \mathbb{E}\,\mathbb{E}_I[\log \bar w_I] = \mathcal L_K$, the last equality by exchangeability. Convergence follows from $\bar w_K \to p(x)$ a.s. (law of large numbers) plus uniform integrability of $\log\bar w_K$. $\square$

The cost is not compute but *gradient signal*. The IWAE gradient with respect to $\phi$ is a self-normalized importance-weighted average (Proposition 13.7's estimator), and its signal-to-noise ratio for the *encoder* parameters degrades like $K^{-1/2}$: as the bound tightens, the encoder's gradient becomes relatively noisier, and at large $K$ the encoder can stop improving altogether. This is a rare and instructive case where a *better objective* trains *worse*, and the diagnosis came directly from a variance analysis of the estimator.

**Discrete latents.** If $z$ is categorical, Proposition 13.36 does not apply — there is no differentiable reparameterization of a discrete draw. Three options: REINFORCE with a learned baseline (unbiased, high variance; §13.2.4); the Gumbel-Softmax / concrete relaxation (Ch. 3), which reparameterizes a *continuous relaxation* of the categorical and is biased for temperature $>0$ but low-variance; or exact enumeration when the support is small (for $k$ categories, $k$ decoder passes, and the ELBO's expectation is a finite sum — Rao–Blackwellization taken to its limit).

> **Common Confusion 13.7.** *"The KL term is just a regularizer, so I can weight it however I like — and the ELBO is basically the log-likelihood, so I can compare ELBO numbers across papers."*
>
> Two errors with the same root. (i) The ELBO is a *bound*: $\mathcal L = \log p_\theta(x) - D_{\mathrm{KL}}(q_\phi\|p_\theta(\cdot\mid x))$. A model with a better ELBO can have a *worse* likelihood if its posterior is easier to approximate, and a model with a worse ELBO can have a better likelihood. Reporting ELBO as if it were likelihood is only defensible when the gap is small or bounded, which is why careful papers report IWAE bounds with large $K$ (Proposition 13.38) or annealed-importance-sampling estimates. (ii) Once you weight the KL by $\beta \ne 1$, the objective is *no longer a bound on any log-likelihood* — it is the Lagrangian of a rate-constrained problem (Ch. 11's rate–distortion). Worked Example 13.20 makes this vivid: the $\beta = 100$ row has "ELBO" $-47.31$, which is not a bound on anything about the data, it is a report that the model has thrown away its latent. Weight the KL if you want, but stop calling the result an ELBO and stop comparing it to one.

> **Self-check 13.8.**
> 1. Which two gaps separate a VAE's training objective from $\log p_\theta(x)$, and which one does *not* exist in non-amortized VI?
> 2. In Proposition 13.37, why does the identity imply that a per-datapoint KL of zero for all $x$ forces $I_q(X;Z) = 0$?
> 3. You increase $K$ in IWAE from 5 to 500 and the model gets worse. Give the mechanism.
> 4. Why is displaying $g_\theta(z)$ rather than a draw from $p_\theta(\cdot\mid z)$ mildly misleading, and how would you show the difference honestly?

---

## 13.9 Score matching and diffusion models

This chapter builds one generative model family in depth and treats the others as context, so it is worth placing them before starting. There are four broad strategies for learning a distribution you can sample from. **Autoregressive models** factor $p(x)=\prod_i p(x_i\mid x_{<i})$ and fit each conditional by maximum likelihood; the likelihood is exact and tractable, sampling is inherently sequential, and no measure-theoretic difficulty arises because the model is a product of ordinary conditional densities. **Normalizing flows** (Ch. 3) push a simple base law through a diffeomorphism and read off the density by change of variables; exact likelihood again, at the cost of architectural constraints and equal latent and data dimension. **Implicit models**, of which GANs are the canonical case, define the model as a pushforward $G_\#N(0,I_k)$ with $k<d$ and never write a density at all — which is what led to Failure 4 in Chapter 1 and is diagnosed in §13.9.3. **Latent-variable models with inference networks** — VAEs (§13.8) and diffusion models (this section) — introduce latents and optimize a bound.

Diffusion gets the long treatment here for two reasons that are pedagogical rather than partisan: its objective is *derived* from machinery this tutorial already built (a variational bound, Gaussian conjugacy, conditional expectation as $L^2$ projection), and it is the one construction that confronts Chapter 1's measure-zero obstruction head-on instead of assuming it away. Nothing below argues that diffusion is the best family for a given problem. Flow matching, the current generalization, appears in the closing “Where to go next” section.

### 13.9.1 The score: a way around the normalizing constant

An energy-based model $p_\theta(x) = e^{-E_\theta(x)}/Z_\theta$ is easy to specify and hard to fit, because $Z_\theta = \int e^{-E_\theta}$ is an intractable integral that depends on $\theta$, so $\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta\log Z_\theta$ needs the very expectation we cannot compute. The escape is to work with a different derivative.

**Definition 13.39 (score).** For a positive differentiable density $p$ on $\mathbb R^d$, the *(Stein) score* is $s(x) := \nabla_x \log p(x)$.

The point is trivial and decisive: $\nabla_x \log\big(e^{-E(x)}/Z\big) = -\nabla_x E(x)$, and $Z$ is gone, because $Z$ does not depend on $x$. (Warning on terminology: Chapter 10's "score" was $\nabla_\theta \log p_\theta(x)$, the derivative in the *parameter*, whose mean is zero. The score here is the derivative in the *data*, and its mean is not zero. The two are unrelated objects that share a name; papers rely on context.)

So: fit a network $s_\theta : \mathbb R^d \to \mathbb R^d$ to the score of the data distribution. The naive objective — squared error to the true score — is unavailable because we do not know the true score. Hyvärinen's identity removes it.

**Proposition 13.40 (score matching / integration by parts).** Let $p$ be a positive $C^1$ density on $\mathbb R^d$ with $\mathbb{E}_p\|\nabla\log p\|^2 < \infty$, and let $s_\theta$ be $C^1$ with $p(x)\,s_\theta(x) \to 0$ as $\|x\|\to\infty$ and suitable integrability. Then
$$
\tfrac12\,\mathbb{E}_{p}\big\|s_\theta(X) - \nabla\log p(X)\big\|^2
= \mathbb{E}_{p}\Big[\tfrac12\|s_\theta(X)\|^2 + \nabla\!\cdot\! s_\theta(X)\Big] + C,
$$
where $\nabla\cdot s_\theta = \sum_j \partial_j s_{\theta,j}$ is the divergence (the trace of the Jacobian) and $C = \frac12\mathbb{E}_p\|\nabla\log p\|^2$ does not depend on $\theta$.

*Proof.* Expand the square:
$$
\tfrac12\mathbb{E}\|s_\theta\|^2 - \mathbb{E}\big[s_\theta^\top \nabla\log p\big] + C .
$$
For the cross term, coordinate by coordinate, using $p\,\partial_j \log p = \partial_j p$ and integrating by parts,
$$
\int s_{\theta,j}(x)\,\partial_j p(x)\,dx = \big[s_{\theta,j}p\big]_{-\infty}^{\infty} - \int \partial_j s_{\theta,j}(x)\,p(x)\,dx = -\mathbb{E}_p[\partial_j s_{\theta,j}],
$$
the boundary term vanishing by hypothesis. Summing over $j$ gives $-\mathbb{E}[s_\theta^\top\nabla\log p] = +\mathbb{E}[\nabla\cdot s_\theta]$. $\square$

So one can fit a score without knowing it, at the cost of a divergence — which for a $d$-dimensional network requires $d$ backward passes (or a Hutchinson trace estimator, itself a Monte Carlo estimator with its own variance). For images, $d \sim 10^5$, so this is unusable. Worse, Proposition 13.40 assumed a *positive density on all of $\mathbb R^d$*, and Chapter 1 proved that real data concentrated on a low-dimensional manifold has no density at all. Both problems are solved by the same trick: add noise.

**Proposition 13.41 (denoising score matching; Vincent's identity).** Let $X \sim P_{\mathrm{data}}$ (any distribution, no density needed), let $\varepsilon\sim N(0,I_d)$ independent, fix $\sigma>0$, and let $\tilde X := X + \sigma\varepsilon$ with law $P_\sigma$, which has the everywhere-positive density $p_\sigma(\tilde x) = \int \varphi_\sigma(\tilde x - x)\,P_{\mathrm{data}}(dx)$. Then
$$
\nabla_{\tilde x}\log p_\sigma(\tilde x) = \mathbb{E}\Big[\frac{X - \tilde x}{\sigma^2}\;\Big|\;\tilde X = \tilde x\Big] = -\frac{1}{\sigma}\,\mathbb{E}\big[\varepsilon \mid \tilde X = \tilde x\big],
$$
and consequently, for any $s_\theta$,
$$
\mathbb{E}_{P_\sigma}\big\|s_\theta(\tilde X) - \nabla\log p_\sigma(\tilde X)\big\|^2
= \mathbb{E}_{X,\varepsilon}\Big\|s_\theta(X + \sigma\varepsilon) + \frac{\varepsilon}{\sigma}\Big\|^2 - C',
$$
with $C'$ independent of $\theta$. Hence minimizing the *denoising* objective on the right — a plain regression of the added noise — yields the score of the noised distribution.

*Proof.* Differentiate under the integral (justified by dominated convergence, Ch. 2, since $\varphi_\sigma$ and its gradient are dominated by integrable functions of $\tilde x$ locally):
$$
\nabla_{\tilde x} p_\sigma(\tilde x) = \int \nabla_{\tilde x}\varphi_\sigma(\tilde x - x)P_{\mathrm{data}}(dx) = \int \frac{x - \tilde x}{\sigma^2}\varphi_\sigma(\tilde x - x)P_{\mathrm{data}}(dx).
$$
Dividing by $p_\sigma(\tilde x)$ turns the right-hand side into an expectation under the posterior $P(dx\mid\tilde X = \tilde x) \propto \varphi_\sigma(\tilde x - x)P_{\mathrm{data}}(dx)$, giving the first display; the second form follows from $X - \tilde X = -\sigma\varepsilon$. For the objective, expand both squares and compare cross terms:
$$
\mathbb{E}\big\|s_\theta(\tilde X) + \tfrac\varepsilon\sigma\big\|^2 = \mathbb{E}\|s_\theta(\tilde X)\|^2 + \tfrac2\sigma\mathbb{E}\big[s_\theta(\tilde X)^\top \varepsilon\big] + \tfrac{d}{\sigma^2}.
$$
By the tower property (Ch. 6), $\mathbb{E}[s_\theta(\tilde X)^\top\varepsilon] = \mathbb{E}\big[s_\theta(\tilde X)^\top\mathbb{E}[\varepsilon\mid\tilde X]\big] = -\sigma\,\mathbb{E}\big[s_\theta(\tilde X)^\top\nabla\log p_\sigma(\tilde X)\big]$. The same cross term (with the same coefficient) appears in the expansion of $\mathbb{E}\|s_\theta - \nabla\log p_\sigma\|^2$, and the remaining terms differ by constants in $\theta$. $\square$

Two things just happened, and they are the reason diffusion models work. First, **the intractable divergence disappeared**: the objective is now an ordinary squared-error regression, one forward and one backward pass. Second, **the requirement of a density disappeared**: $P_{\mathrm{data}}$ may be supported on a circle, a union of manifolds, or a finite set of points; $P_\sigma$ always has a smooth positive density on all of $\mathbb R^d$ because convolution with a Gaussian is a smoothing operator. The price is that we learn the score of $P_\sigma$, not of $P_{\mathrm{data}}$ — and that price is paid by using many noise levels and annealing.

**Annealed Langevin sampling.** Given scores at a decreasing sequence $\sigma_1 > \sigma_2 > \dots > \sigma_T$, run a few ULA steps at $\sigma_1$ (where the target is nearly Gaussian and easy), use the result to initialize ULA at $\sigma_2$, and so on. Each stage's target is close to the previous one's, so each initialization is good — this is *simulated annealing implemented with scores*, and it is the algorithm of the original score-based generative models.

### 13.9.2 DDPM: deriving the objective, not quoting it

The denoising diffusion probabilistic model organizes the same idea as an explicit latent-variable model with a variational bound, which is why it belongs in this chapter next to VAEs: **a DDPM is a VAE with $T$ stochastic layers, a fixed (non-learned) encoder, and a Gaussian decoder at each layer.**

**Definition 13.42 (forward process).** Fix $T$ and $\beta_1,\dots,\beta_T \in (0,1)$. The *forward (noising) process* is the Markov chain
$$
q(x_t \mid x_{t-1}) = N\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big), \qquad t=1,\dots,T,
$$
started at $x_0 \sim P_{\mathrm{data}}$. Write $\alpha_t := 1-\beta_t$ and $\bar\alpha_t := \prod_{s\le t}\alpha_s$.

**Proposition 13.43 (closed-form marginals).** For every $t$, $q(x_t\mid x_0) = N\big(x_t; \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)$. Equivalently $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ with $\varepsilon\sim N(0,I)$.

*Proof.* Induction. True at $t=1$ by definition since $\bar\alpha_1 = \alpha_1$. Assume $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\eta$ with $\eta\sim N(0,I)$ independent of $x_0$. Then
$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{\beta_t}\,\xi = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\eta + \sqrt{1-\alpha_t}\,\xi,
$$
with $\xi \sim N(0,I)$ independent of $\eta$. The sum of two independent centred Gaussians is Gaussian (Ch. 5) with variance $\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\bar\alpha_{t-1} = 1-\bar\alpha_t$. $\square$

This closed form is what makes training cheap: to get a training pair at step $t$ you do not simulate $t$ steps, you sample $t$ uniformly and jump straight there.

**Worked Example 13.22 (verifying the closed form numerically).** With $T = 200$ and $\beta_t$ linear on $[10^{-4}, 0.06]$ (so $\bar\alpha_1 = 0.99990$, $\bar\alpha_T = 0.002167$), I iterated the one-step kernel of Definition 13.42 from $x_0 = (1,0)$ 20,000 times and compared with Proposition 13.43 (seeds $1000+t$):

| $t$ | iterated mean | iterated sd | closed-form mean | closed-form sd |
|---|---|---|---|---|
| 10 | $(0.9933, -0.0004)$ | $(0.1206, 0.1196)$ | $(0.9927, 0)$ | $0.1202$ |
| 50 | $(0.8304, -0.0009)$ | $(0.5576, 0.5620)$ | $(0.8288, 0)$ | $0.5596$ |
| 100 | $(0.4688, -0.0078)$ | $(0.8813, 0.8838)$ | $(0.4688, 0)$ | $0.8833$ |
| 200 | $(0.0415, -0.0017)$ | $(1.0021, 1.0054)$ | $(0.0466, 0)$ | $0.9989$ |

Agreement to three decimals throughout, and at $t=T$ the marginal is essentially $N(0,I)$ — which is what licenses starting the reverse process from pure noise. **Run this check on your own schedule before training anything**; a mismatch here is the single most common diffusion bug.

**Proposition 13.44 (the forward posterior is Gaussian and computable).** For $t\ge1$,
$$
q(x_{t-1}\mid x_t, x_0) = N\big(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\big),
$$
$$
\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0, \qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .
$$

*Proof.* By Bayes' rule (Proposition 1.12) and the Markov property,
$$
q(x_{t-1}\mid x_t,x_0) \propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0),
$$
a product of two Gaussian densities in $x_{t-1}$, hence Gaussian (Ch. 5). Collecting the quadratic coefficient:
$$
\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})} = \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})},
$$
whose reciprocal is $\tilde\beta_t$. The linear coefficient is $\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0$; multiplying by $\tilde\beta_t$ gives $\tilde\mu_t$ as displayed. $\square$

Now the model. Define the *reverse* process as a learned Markov chain
$$
p_\theta(x_{0:T}) := p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t), \qquad p(x_T) = N(0,I),
\quad p_\theta(x_{t-1}\mid x_t) = N\big(\mu_\theta(x_t,t),\ \Sigma_t\big),
$$
with $\Sigma_t$ fixed (either $\beta_t I$ or $\tilde\beta_t I$; both are used).

**Theorem 13.45 (the DDPM training objective).** With $q$ as in Definition 13.42 and $p_\theta$ as above,
$$
-\log p_\theta(x_0) \;\le\; \underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\|p(x_T)\big)}_{L_T:\ \text{no parameters}}
+ \sum_{t=2}^{T}\underbrace{\mathbb{E}_{q}\Big[D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)\Big]}_{L_{t-1}}
\;\underbrace{-\ \mathbb{E}_q\big[\log p_\theta(x_0\mid x_1)\big]}_{L_0}.
$$
Moreover, if $\Sigma_t = \sigma_t^2 I$ is fixed and one reparameterizes
$$
\mu_\theta(x_t,t) := \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\Big),
$$
then each $L_{t-1}$ equals
$$
L_{t-1} = \mathbb{E}_{x_0,\varepsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\big\|\varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\ t\big)\big\|^2\Big] + \text{const}.
$$
Dropping the $t$-dependent weight gives the DDPM "simple" loss $L_{\mathrm{simple}} = \mathbb{E}_{t,x_0,\varepsilon}\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$.

*Proof.* Start from the variational bound of Theorem 13.31(a) with latent $z = x_{1:T}$ and variational distribution $q(x_{1:T}\mid x_0)$ — which here is *fixed, not learned*:
$$
-\log p_\theta(x_0) \le \mathbb{E}_q\Big[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big] =: L .
$$
Expand both products:
$$
L = \mathbb{E}_q\Big[-\log p(x_T) - \sum_{t\ge1}\log\frac{p_\theta(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}\Big].
$$
For $t \ge 2$ rewrite $q(x_t\mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$ (Markov property) and apply Bayes' rule conditionally on $x_0$:
$$
q(x_t\mid x_{t-1},x_0) = \frac{q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)} .
$$
Substituting, the ratio $q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$ telescopes across $t = 2,\dots,T$ leaving $q(x_T\mid x_0)/q(x_1\mid x_0)$, and collecting terms gives
$$
L = \mathbb{E}_q\Big[\log\frac{q(x_T\mid x_0)}{p(x_T)} + \sum_{t=2}^T \log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)} - \log p_\theta(x_0\mid x_1)\Big],
$$
and taking the inner expectations turns the first two groups into the KL divergences displayed (the first is a KL because $x_T$'s conditional law is exactly $q(x_T\mid x_0)$; each summand is a KL after conditioning on $(x_t,x_0)$ and using the tower property).

For the second claim: both arguments of $L_{t-1}$ are Gaussians whose covariances are *fixed* — $q(x_{t-1}\mid x_t, x_0)$ has covariance $\tilde\beta_t I$ (Proposition 13.44) and $p_\theta(x_{t-1}\mid x_t)$ has covariance $\sigma_t^2 I$, and neither depends on $\theta$. The KL between two Gaussians (Ch. 5) splits into a mean term and covariance terms, and only the mean term involves $\theta$:
$$
D_{\mathrm{KL}} = \frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\big\|^2 + c_t,
\qquad c_t := \frac d2\Big(\frac{\tilde\beta_t}{\sigma_t^2} - 1 - \log\frac{\tilde\beta_t}{\sigma_t^2}\Big) \;\ge 0,
$$
where $c_t$ is one of the constants absorbed into the "$+\ \text{const}$" of the statement. For the choice $\sigma_t^2 = \tilde\beta_t$ the covariances literally match and $c_t = 0$; for $\sigma_t^2 = \beta_t$ they do not and $c_t > 0$ — harmless for training, since $\nabla_\theta c_t = 0$, but a genuine additive gap if you use the bound to *report* a likelihood.
Now express $\tilde\mu_t$ in terms of $(x_t,\varepsilon)$ rather than $(x_t,x_0)$. From Proposition 13.43, $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\varepsilon)/\sqrt{\bar\alpha_t}$. Substituting into Proposition 13.44 and simplifying (the algebra is worth doing once; use $\bar\alpha_t = \alpha_t\bar\alpha_{t-1}$ and $\beta_t = 1-\alpha_t$):
$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\Big).
$$
Comparing with the stated $\mu_\theta$, the two differ only by $\varepsilon$ versus $\varepsilon_\theta$, so
$$
\big\|\tilde\mu_t - \mu_\theta\big\|^2 = \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}\|\varepsilon - \varepsilon_\theta\|^2,
$$
which gives the displayed $L_{t-1}$. $\square$

Now the honest bookkeeping the phase demanded: **which parts of that are forced and which are choices?**

*Forced (consequences of the definitions):* the bound itself; the decomposition into $L_T + \sum L_{t-1} + L_0$; the Gaussian form and exact parameters of $q(x_{t-1}\mid x_t,x_0)$; the fact that with matched covariances each $L_{t-1}$ is a squared mean-difference; the algebraic identity converting $\tilde\mu_t$ into the $\varepsilon$-form; and hence the weighted denoising regression with weight $\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$.

*Modelling choices:* the Gaussian forward kernel with variance-preserving scaling $\sqrt{1-\beta_t}$; the schedule $\{\beta_t\}$ and the horizon $T$; the decision to make the reverse covariance *fixed* rather than learned; the $\varepsilon$-parameterization of $\mu_\theta$ (one could predict $x_0$, or $\tilde\mu$ directly, or the score — all are related by fixed affine maps but give different loss weightings and different conditioning); the choice $p(x_T) = N(0,I)$ (an approximation whose error is $L_T$, small only if $\bar\alpha_T\approx 0$); and — the big one — **dropping the $t$-dependent weight to obtain $L_{\mathrm{simple}}$**. That last step is not a consequence of anything; it is an empirical discovery that down-weighting small-$t$ terms (which the true weight up-weights enormously, since $\beta_t^2/(1-\bar\alpha_t) \to$ large as $t\to1$) produces better samples. In fact $L_{\mathrm{simple}}$ corresponds to a *different* weighting of the same family of denoising objectives, and it is no longer a bound on $-\log p_\theta(x_0)$. Papers that report likelihoods therefore train (or fine-tune) with the true weights, or with a learned reverse variance, and papers that report sample quality use $L_{\mathrm{simple}}$. **They are optimizing different objectives, and comparing their numbers requires care.**

The relation to §13.9.1 is now exact: since $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$, Proposition 13.41 with $\sigma^2 = (1-\bar\alpha_t)/\bar\alpha_t$ (after rescaling) gives
$$
\nabla_{x_t}\log q(x_t) = -\frac{\mathbb{E}[\varepsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}} \approx -\frac{\varepsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}} .
$$
**Predicting the noise is estimating the score, up to a known scalar.** DDPM and score-based models are the same model in different coordinates.

**Worked Example 13.23 (a DDPM from scratch, on Chapter 1's circle).** Data: exactly uniform on the unit circle, *no observation noise* — so $P_{\mathrm{data}}$ is supported on a Lebesgue-null set in $\mathbb R^2$ and has no density (Chapter 1's result). Network: MLP $\mathbb R^{9}\to128\to128\to\mathbb R^2$ with $\tanh$, taking $(x_t, \sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t}, t/T$, and four sinusoidal features of $t/T)$; trained with $L_{\mathrm{simple}}$, Adam $10^{-3}$, batch 256, 60,000 steps, seeds 3/11. Final loss (mean over last 500 steps) $0.4447$, against $2.0$ for a network that always predicted $0$ — the model is explaining most of the noise. Ancestral sampling of 5000 points (seed 77):

| quantity | model | truth |
|---|---|---|
| mean radius | $0.9984$ | $1$ |
| sd of radius | $0.0167$ | $0$ |
| angle histogram, 8 equal bins (expected 625 each) | 658, 670, 585, 785, 525, 478, 634, 665 | 625 each |
| $\chi^2$ statistic on 7 df | $101.8$ | — |

Two conclusions, one happy and one honest. The happy one: the model has learned the *manifold* almost exactly — samples sit on the circle to within 1.7% — despite the target having no density anywhere, which no likelihood-based method with a full-dimensional Gaussian output could do so cleanly. The honest one: the *density along* the manifold is only approximate. A $\chi^2$ of 101.8 on 7 degrees of freedom is overwhelming evidence against uniformity ($p < 10^{-17}$), so the model over-represents some arcs by up to 25%. A 2-layer MLP trained for 60k steps is not enough; and the general lesson is that **getting the support right and getting the density right are different achievements, and sample-quality metrics are much more sensitive to the former.** This is worth remembering when reading FID numbers.

### 13.9.3 How diffusion circumvents the measure-zero problem

Chapter 1 identified a structural obstruction, and Chapter 2 diagnosed it: if $P_\theta = (G_\theta)_\#N(0,I_k)$ with $k<d$, then $\operatorname{supp}(P_\theta)$ has Lebesgue measure zero, $P_\theta$ has no Lebesgue density, and any divergence that requires overlapping supports is degenerate — the Jensen–Shannon divergence between the model and the data is constant at $\log 2$ once the supports are disjoint, so the GAN generator's loss carries *no gradient signal about how to move*. That was Failure 4.

Diffusion dissolves the problem rather than solving it. Here is the statement, precisely:

- For every $t \ge 1$, the noised distribution $q_t := \mathrm{law}(x_t)$ is a Gaussian convolution of $P_{\mathrm{data}}$, hence has a strictly positive $C^\infty$ density on all of $\mathbb R^d$ **no matter what $P_{\mathrm{data}}$ is** — manifold-supported, discrete, fractal, anything.
- Therefore, at every $t\ge1$: the score $\nabla\log q_t$ exists everywhere; the KL divergences in Theorem 13.45 are all finite (both arguments are full-support Gaussians); the training target $\mathbb{E}[\varepsilon\mid x_t]$ is a well-defined conditional expectation (Ch. 6) and is what regression estimates; and the model and data distributions *always overlap*, so there is never a region where the loss is flat because the two distributions are mutually singular.
- The support mismatch is confined to the single limit $t\to0$, i.e. $\sigma\to0$, and at that limit it reappears exactly as a blow-up of the score.

**Worked Example 13.24 (the difficulty, localized to $\sigma\to0$).** Take $P_{\mathrm{data}}$ uniform on the unit circle in $\mathbb R^2$ and $p_\sigma$ its convolution with $N(0,\sigma^2I)$, computed by numerically integrating over the angle on a 20,000-point grid:

| $\sigma$ | $p_\sigma$ at $x = (1.1, 0)$ | $\|\nabla\log p_\sigma\|$ at $(1.1,0)$ | $\|\nabla\log p_\sigma\|$ at $(1.0,0)$ |
|---|---|---|---|
| 1.00 | $6.99\times10^{-2}$ | 0.62 | 0.554 |
| 0.20 | $2.68\times10^{-1}$ | 2.96 | 0.505 |
| 0.10 | $3.68\times10^{-1}$ | 10.46 | 0.501 |
| 0.05 | $1.64\times10^{-1}$ | 40.45 | 0.500 |
| 0.02 | $1.13\times10^{-5}$ | 250.45 | 0.500 |
| 0.01 | $1.17\times10^{-21}$ | 1000.45 | 0.500 |

Read the third column: at distance $0.1$ from the manifold, the score's magnitude is $\approx 0.1/\sigma^2$ — it diverges quadratically as $\sigma\to0$, exactly as it must if it is to push a point onto a measure-zero set in finite time. On the manifold (fourth column) the score stays bounded, converging to $0.5$, the curvature term. So the pathology is not everywhere: **it is a boundary layer of width $\sigma$ around the data manifold, and it is why every practical implementation stops at a positive $\sigma_{\min}$, clips or rescales the score, and parameterizes the network to predict $\varepsilon$ (bounded) rather than the score (unbounded).** The $\varepsilon$-parameterization is not cosmetic: it hides a $1/\sqrt{1-\bar\alpha_t}$ factor that would otherwise have to be represented by the network's output at exactly the noise levels where it is largest.

### 13.9.4 The SDE view, probability flow, and guidance

Letting $T\to\infty$ with $\beta_t = \beta(t)\Delta t$ turns Definition 13.42 into the *variance-preserving SDE*
$$
dx = -\tfrac12\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW_t ,
$$
whose time reversal (Anderson's theorem) is another SDE driven by the score:
$$
dx = \Big[-\tfrac12\beta(t)x - \beta(t)\nabla_x\log q_t(x)\Big]dt + \sqrt{\beta(t)}\,d\bar W_t .
$$
Two consequences matter.

**Proposition 13.46 (probability-flow ODE; stated).** The *deterministic* dynamics
$$
\frac{dx}{dt} = -\tfrac12\beta(t)\Big[x + \nabla_x\log q_t(x)\Big]
$$
has the same time-marginals $q_t$ as the SDE above. Consequently the map $x_T \mapsto x_0$ obtained by integrating it backwards is a *continuous normalizing flow* (Ch. 3), and the model's exact log-likelihood is available via the instantaneous change-of-variables formula,
$$
\log p_\theta(x_0) = \log p(x_T) + \int_0^T \nabla\!\cdot\!\Big(\tfrac12\beta(t)\big[x + s_\theta(x,t)\big]\Big)dt ,
$$
with the divergence estimated by Hutchinson's trick.

I state rather than prove this: the marginal-preservation claim is a Fokker–Planck computation of the same flavour as Proposition 13.26 but for the time-inhomogeneous case, and Anderson's time-reversal theorem is genuine stochastic analysis. It is listed in §13.12. What matters practically: the ODE view is why diffusion samplers can be run deterministically with 20–50 steps instead of 1000 (DDIM and higher-order ODE solvers), and why exact likelihoods are computable for a model trained by regression — Chapter 3's change-of-variables machinery is what converts a learned vector field into a density.

**Classifier-free guidance.** To sample from a conditional model $q(x\mid y)$ with extra emphasis on the condition, note that
$$
\nabla_x \log q(x\mid y) = \nabla_x\log q(x) + \nabla_x\log q(y\mid x),
$$
so amplifying the classifier term by $w$ corresponds to the *tilted* score
$$
s_w(x,y) := \nabla_x \log q(x) + w\,\big[\nabla_x\log q(x\mid y) - \nabla_x\log q(x)\big] = (1-w)s(x) + w\,s(x\mid y).
$$
In practice one trains a single network with the conditioning randomly dropped, so it learns both $s(x\mid y)$ and $s(x)$, and combines them at sampling time with $w>1$. Interpretation: $s_w$ is the score of a distribution $\propto q(x)\,q(y\mid x)^{w}$ — the conditional with a *sharpened* likelihood. Two honest caveats. First, that target is a valid distribution only if $q(x)q(y\mid x)^w$ is normalizable, which is not guaranteed for all $w$ and all models; empirically large $w$ produces over-saturated, low-diversity samples, which is the normalization pathology showing up as mode collapse. Second, the tilted score is not the score of any distribution *along the trajectory* unless the tilting is applied consistently at every $t$, and it is not: guided sampling is a heuristic whose fidelity/diversity trade-off is tuned, not derived. It works extremely well, and it is not a theorem.

<svg viewBox="0 0 720 340" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diffusion forward and reverse process across noise levels">
  <text x="30" y="20" fill="currentColor" font-size="12">forward q(xₜ|x₀): add noise  →  distribution gains full-dimensional support at every t &gt; 0</text>
  <circle cx="95" cy="95" r="42" fill="none" stroke="#10b981" stroke-width="2.2"/>
  <text x="66" y="152" fill="currentColor" font-size="11">t = 0 (data)</text>
  <text x="52" y="166" fill="currentColor" font-size="10.5">measure zero, no density</text>
  <circle cx="255" cy="95" r="42" fill="none" stroke="#10b981" stroke-width="9" stroke-opacity="0.28"/>
  <circle cx="255" cy="95" r="42" fill="none" stroke="#10b981" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="228" y="152" fill="currentColor" font-size="11">t = 50</text>
  <circle cx="415" cy="95" r="38" fill="none" stroke="#10b981" stroke-width="26" stroke-opacity="0.2"/>
  <text x="388" y="152" fill="currentColor" font-size="11">t = 100</text>
  <circle cx="580" cy="95" r="46" fill="#3b82f6" fill-opacity="0.16" stroke="#3b82f6" stroke-width="1"/>
  <text x="540" y="152" fill="currentColor" font-size="11">t = T ≈ N(0, I)</text>
  <path d="M145,95 L205,95" stroke="currentColor" stroke-width="1.2"/><path d="M205,95 l-7,-3 l0,6 z" fill="currentColor"/>
  <path d="M305,95 L365,95" stroke="currentColor" stroke-width="1.2"/><path d="M365,95 l-7,-3 l0,6 z" fill="currentColor"/>
  <path d="M465,95 L525,95" stroke="currentColor" stroke-width="1.2"/><path d="M525,95 l-7,-3 l0,6 z" fill="currentColor"/>
  <path d="M525,215 L465,215" stroke="#ef4444" stroke-width="1.4"/><path d="M465,215 l7,-3 l0,6 z" fill="#ef4444"/>
  <path d="M365,215 L305,215" stroke="#ef4444" stroke-width="1.4"/><path d="M305,215 l7,-3 l0,6 z" fill="#ef4444"/>
  <path d="M205,215 L145,215" stroke="#ef4444" stroke-width="1.4"/><path d="M145,215 l7,-3 l0,6 z" fill="#ef4444"/>
  <text x="150" y="205" fill="#ef4444" font-size="11">‖∇log qₜ‖ ~ 1/σ²</text>
  <text x="330" y="205" fill="#ef4444" font-size="11">ε_θ(xₜ,t) ≈ E[ε|xₜ]</text>
  <text x="490" y="205" fill="#ef4444" font-size="11">start from noise</text>
  <text x="30" y="245" fill="#ef4444" font-size="12">reverse p_θ(x_{t−1}|xₜ): subtract predicted noise; each step is a Gaussian whose mean uses the learned score</text>
  <text x="30" y="290" fill="currentColor" font-size="11.5">Figure 13.6 — Smoothing makes the problem well posed. At t = 0 the data may live on a measure-zero set (green</text>
  <text x="30" y="304" fill="currentColor" font-size="11.5">circle: no Lebesgue density, so no likelihood and, for a GAN, no JS gradient). For every t &gt; 0 the convolved</text>
  <text x="30" y="318" fill="currentColor" font-size="11.5">distribution has a positive smooth density everywhere, so scores, KLs and likelihoods all exist. The difficulty</text>
  <text x="30" y="332" fill="currentColor" font-size="11.5">returns only as t → 0, where the score diverges like 1/σ² (measured values: Worked Example 13.24).</text>
</svg>

> **Self-check 13.9.**
> 1. Why does Proposition 13.40 require a positive density on $\mathbb R^d$, and which hypothesis does Proposition 13.41 drop?
> 2. In Theorem 13.45, which term contains no parameters, and what does it being small require of the schedule?
> 3. Predicting $\varepsilon$ and predicting the score differ by a known factor. Why does the choice still matter numerically?
> 4. State precisely why a GAN's JS loss is flat on disjoint supports while a DDPM's loss is not.

---

## 13.10 Uncertainty quantification and calibration

Everything so far produces distributions. This section asks the deployment question: **when a model reports a number as a probability, or an interval as 90% likely, what makes that report true?**

### 13.10.1 Two kinds of uncertainty, and why the split is model-relative

**Definition 13.47 (aleatoric and epistemic uncertainty).** Fix a model class $\{p_\theta\}$ and a posterior (or any distribution) $\pi$ over $\theta$. The predictive distribution is $p(y\mid x) = \int p_\theta(y\mid x)\,\pi(d\theta)$. Decompose its uncertainty, for instance via the law of total variance (Ch. 6):
$$
\operatorname{Var}(Y\mid x) = \underbrace{\mathbb{E}_{\pi}\big[\operatorname{Var}_{\theta}(Y\mid x)\big]}_{\text{aleatoric}} + \underbrace{\operatorname{Var}_{\pi}\big(\mathbb{E}_\theta[Y\mid x]\big)}_{\text{epistemic}} .
$$
*Aleatoric* uncertainty is the noise the model believes is irreducible; *epistemic* is the model's uncertainty about which $\theta$ is right.

The decomposition is exact and useful — epistemic uncertainty should shrink with data, aleatoric should not — but notice what it is relative to: **the model class and the parameterization.** Label noise that one model calls aleatoric is, to a richer model with access to an extra feature, epistemic. There is no model-free fact of the matter about which is which. This is worth saying plainly because papers sometimes present the split as a property of the world; it is a property of a modelling choice. The practical content survives: an ensemble's *disagreement* is an epistemic signal and its *average predicted entropy* is an aleatoric one, and the two behave differently under distribution shift (disagreement grows, predicted entropy often does not, which is why ensembles detect shift better than single-model entropy).

In practice, epistemic uncertainty for a large model is estimated by one of four things, none of which is the posterior. A **Laplace approximation** fits a Gaussian at a mode using the Hessian (or a Fisher/Kronecker-factored surrogate) — cheap, post-hoc, and exactly the mean-field-style underestimate that Proposition 13.34 predicts when the covariance is diagonalized. **SWAG** and related methods fit a Gaussian to the trajectory of SGD iterates, which is the constant-step stationary distribution of §13.5.3 pressed into service as a posterior surrogate, with all the caveats of Common Confusion 13.5. **Deep ensembles** train several networks from different initializations; they are not samples from any posterior, but they capture between-mode variation that a single chain cannot (§13.4.4), and they are consistently the strongest baseline. **Last-layer or subnetwork Bayes** applies an exact method to a low-dimensional slice and treats the rest as fixed. Which to use is an empirical question; what matters for this chapter is that all four are *substitutes* for a posterior you cannot compute, and that reporting their spread as "the posterior uncertainty" is the overclaim to avoid.

### 13.10.2 Calibration

**Definition 13.48 (calibration).** A probabilistic classifier $\hat p : \mathcal X \to \Delta_K$ is *(perfectly) calibrated* if for every $c$ in the range of the confidence $\hat c(x) := \max_k \hat p_k(x)$,
$$
P\big(Y = \hat y(X) \;\big|\; \hat c(X) = c\big) = c, \qquad \hat y(x) := \arg\max_k \hat p_k(x).
$$
It is *fully (multi-class) calibrated* if $P(Y = k \mid \hat p(X) = q) = q_k$ for all $k$ and all $q$.

Calibration is a *weak* requirement: the constant predictor $\hat p \equiv$ the marginal class frequencies is perfectly calibrated and useless. What you want is calibration *and* sharpness (concentrated predictions). Proper scoring rules measure both at once.

**Definition 13.49 (reliability diagram and ECE).** Partition $[0,1]$ into bins $B_1,\dots,B_M$. A *reliability diagram* plots, for each bin, the mean confidence against the empirical accuracy of the examples whose confidence falls in that bin. The *expected calibration error* is estimated by
$$
\widehat{\mathrm{ECE}} := \sum_{m=1}^M \frac{|B_m|}{n}\Big|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\Big| .
$$

**Definition 13.50 (proper scoring rule).** A scoring rule $S(q, y)$ (loss of reporting distribution $q$ when $y$ occurs) is *proper* for a class of distributions if $\mathbb{E}_{Y\sim P}[S(P,Y)] \le \mathbb{E}_{Y\sim P}[S(q,Y)]$ for all $q$, and *strictly proper* if equality forces $q = P$. The log score $S(q,y) = -\log q_y$ and the Brier score $S(q,y) = \sum_k (q_k - \mathbf 1\{y=k\})^2$ are strictly proper.

**Proposition 13.51 (calibration–refinement decomposition).** Let $\hat p$ be a predictor with a discrete set of possible outputs, and for each output value $q$ let $\pi_q := P(Y = \cdot \mid \hat p(X) = q)$ be the true conditional class distribution given that report. For any proper scoring rule with associated divergence $d$ and entropy $H$,
$$
\mathbb{E}\big[S(\hat p(X), Y)\big] = \underbrace{\mathbb{E}\big[d\big(\pi_{\hat p(X)},\ \hat p(X)\big)\big]}_{\text{calibration}} + \underbrace{\mathbb{E}\big[H\big(\pi_{\hat p(X)}\big)\big]}_{\text{refinement / sharpness}} .
$$
For the Brier score, $d$ is squared Euclidean distance and $H(\pi) = \sum_k \pi_k(1-\pi_k)$; for the log score, $d = D_{\mathrm{KL}}$ and $H$ is Shannon entropy.

*Proof (Brier case, which is the transparent one).* Condition on the report $\hat p(X) = q$ and let $\pi = \pi_q$. Then
$$
\mathbb{E}\Big[\sum_k(q_k - \mathbf 1\{Y=k\})^2 \,\Big|\, \hat p(X)=q\Big] = \sum_k \big(q_k^2 - 2q_k\pi_k + \pi_k\big) = \sum_k (q_k - \pi_k)^2 + \sum_k \pi_k(1-\pi_k),
$$
using $\mathbb{E}[\mathbf 1\{Y=k\}\mid \cdot] = \pi_k$ and $\mathbb{E}[\mathbf 1\{Y=k\}^2\mid\cdot] = \pi_k$. Taking the outer expectation over the report gives the decomposition. The log-score case is the same computation with $-\log$ in place of the square and the tower property (Ch. 6). $\square$

This is the cleanest justification for reporting log-loss or Brier rather than ECE alone: **a proper score cannot be improved by miscalibration, and it simultaneously rewards sharpness, which ECE ignores entirely.**

**Temperature scaling.** Given a trained network with logits $z(x)$, fit a single scalar $T>0$ on held-out data by minimizing the negative log-likelihood of $\mathrm{softmax}(z(x)/T)$. Because a monotone transformation of the logits does not change the argmax, temperature scaling **never changes accuracy** — it only rescales confidence. It is the simplest possible recalibration map and, empirically, nearly always the best per unit of complexity.

**Worked Example 13.25 (an overconfident classifier, diagnosed and fixed).** I built a synthetic 5-class problem: $x\sim N(0,I_{10})$, true logits $Wx$ with $W$ fixed (seed 42), labels sampled from the true softmax, $n = 20{,}000$ examples split into 5000 calibration and 15,000 test. The "trained network" is the true model with logits multiplied by $2.5$ — a stand-in for the sharpening that overfitting produces. Reliability table on test, 15 equal-width bins (abbreviated):

| confidence bin | count | mean confidence | accuracy |
|---|---|---|---|
| (0.47, 0.53] | 689 | 0.504 | 0.388 |
| (0.60, 0.67] | 836 | 0.634 | 0.458 |
| (0.73, 0.80] | 999 | 0.767 | 0.481 |
| (0.87, 0.93] | 1711 | 0.902 | 0.606 |
| (0.93, 1.00] | 7413 | 0.984 | 0.834 |

Every row is below the diagonal: the model claims more than it delivers, and half the test set sits in the top bin claiming 98.4% while achieving 83.4%. Summary metrics:

| | accuracy | mean confidence | ECE (15 bins) | NLL | Brier |
|---|---|---|---|---|---|
| as trained | 0.6664 | 0.8503 | 0.1840 | 1.1303 | 0.5048 |
| after temperature scaling | 0.6664 | 0.6714 | 0.0111 | 0.8230 | 0.4424 |
| the true model $P$ | 0.6664 | — | 0.0097 | 0.8230 | — |

The fitted temperature was $T = 2.4838$, against the $2.5$ that generated the sharpening — recovered to $0.6\%$ from 5000 calibration points. Accuracy is unchanged to four decimals, exactly as promised, while ECE fell by a factor of 17 and both proper scores improved to the values of the *true* model. **Calibration was a one-parameter fix, and no amount of extra training would have found it, because the training loss was already minimized at the sharpened solution on the training set.** (Why are real networks overconfident? The short version: with enough capacity, minimizing NLL on the training set drives training-set confidences to 1, and the resulting logit scale transfers to test inputs where accuracy is lower. Weight decay, label smoothing, mixup, and early stopping all reduce it; none eliminate it.)

**Proposition 13.52 (binned ECE is biased upward).** Let $\hat p$ be perfectly calibrated. Then for any fixed binning with $M \ge 1$ bins and any $n$, $\mathbb{E}\big[\widehat{\mathrm{ECE}}\big] \ge 0$ with equality only in degenerate cases; in particular $\widehat{\mathrm{ECE}} > 0$ almost surely whenever some bin has finite count and $\operatorname{conf}(B_m)\notin\{0,1\}$.

*Proof.* Condition on the bin memberships and confidences. Within bin $m$ with $n_m$ points, $\operatorname{acc}(B_m)$ is an average of independent Bernoulli variables with means $\hat c_i$, so $\mathbb{E}[\operatorname{acc}(B_m)] = \operatorname{conf}(B_m)$ by calibration; but $\mathbb{E}|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)| > 0$ whenever the Bernoulli variance is positive, since a non-degenerate random variable has strictly positive mean absolute deviation. Averaging over bins with weights $n_m/n$ gives a strictly positive expectation. $\square$

**Worked Example 13.26 (how big is the bias?).** Using the *perfectly calibrated* true model from Worked Example 13.25, I computed $\widehat{\mathrm{ECE}}$ with 15 bins on random test subsets, 200 replicates each (seed 42):

| test-set size | mean $\widehat{\mathrm{ECE}}$ (truth: 0) |
|---|---|
| 100 | 0.1111 |
| 500 | 0.0491 |
| 2,000 | 0.0253 |
| 10,000 | 0.0149 |

At $n=100$ a perfectly calibrated model reports ECE $= 0.11$, which is *worse than the genuinely miscalibrated model's* 0.18 is bad. The bias decays roughly like $n^{-1/2}$ (as the mean absolute deviation of a binomial does), so it never vanishes at realistic evaluation sizes. **Consequences for reading papers:** an ECE of 0.02 on a 2000-example test set is indistinguishable from perfect calibration; ECE comparisons across papers with different $n$ or different bin counts are meaningless; and ECE differences smaller than the null bias should be ignored. Report a proper score (which is unbiased) alongside, use equal-*mass* rather than equal-width bins, and if you must compare ECEs, simulate the null.

<svg viewBox="0 0 700 330" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Reliability diagram before and after temperature scaling">
  <line x1="60" y1="40" x2="60" y2="240" stroke="currentColor" stroke-width="1.4"/>
  <line x1="60" y1="240" x2="260" y2="240" stroke="currentColor" stroke-width="1.4"/>
  <line x1="60" y1="240" x2="260" y2="40" stroke="currentColor" stroke-width="0.8" stroke-dasharray="4 4"/>
  <text x="20" y="44" fill="currentColor" font-size="10">acc 1</text>
  <text x="30" y="244" fill="currentColor" font-size="10">0</text>
  <text x="130" y="258" fill="currentColor" font-size="11">confidence</text>
  <text x="60" y="28" fill="#ef4444" font-size="12">before: ECE 0.184</text>
  <rect x="150" y="199" width="16" height="41" fill="#ef4444" fill-opacity="0.45"/>
  <rect x="178" y="184" width="16" height="56" fill="#ef4444" fill-opacity="0.45"/>
  <rect x="206" y="169" width="16" height="71" fill="#ef4444" fill-opacity="0.45"/>
  <rect x="234" y="73" width="16" height="167" fill="#ef4444" fill-opacity="0.45"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2" points="130,163 158,199 186,184 214,169 242,73"/>
  <circle cx="130" cy="163" r="2.5" fill="#ef4444"/><circle cx="158" cy="199" r="2.5" fill="#ef4444"/>
  <circle cx="186" cy="184" r="2.5" fill="#ef4444"/><circle cx="214" cy="169" r="2.5" fill="#ef4444"/>
  <circle cx="242" cy="73" r="2.5" fill="#ef4444"/>
  <line x1="420" y1="40" x2="420" y2="240" stroke="currentColor" stroke-width="1.4"/>
  <line x1="420" y1="240" x2="620" y2="240" stroke="currentColor" stroke-width="1.4"/>
  <line x1="420" y1="240" x2="620" y2="40" stroke="currentColor" stroke-width="0.8" stroke-dasharray="4 4"/>
  <text x="380" y="44" fill="currentColor" font-size="10">acc 1</text>
  <text x="390" y="244" fill="currentColor" font-size="10">0</text>
  <text x="490" y="258" fill="currentColor" font-size="11">confidence</text>
  <text x="420" y="28" fill="#10b981" font-size="12">after T = 2.4838: ECE 0.011</text>
  <polyline fill="none" stroke="#10b981" stroke-width="2" points="460,158 490,132 520,104 550,74 580,50"/>
  <circle cx="460" cy="158" r="2.5" fill="#10b981"/><circle cx="490" cy="132" r="2.5" fill="#10b981"/>
  <circle cx="520" cy="104" r="2.5" fill="#10b981"/><circle cx="550" cy="74" r="2.5" fill="#10b981"/>
  <circle cx="580" cy="50" r="2.5" fill="#10b981"/>
  <text x="60" y="286" fill="currentColor" font-size="11.5">Figure 13.7 — Reliability diagrams for Worked Example 13.25. Left: every bin sits below the diagonal (shaded gap =</text>
  <text x="60" y="300" fill="currentColor" font-size="11.5">confidence minus accuracy), with the top bin claiming 0.984 and delivering 0.834. Right: after dividing the logits by a</text>
  <text x="60" y="314" fill="currentColor" font-size="11.5">single scalar fitted on 5000 held-out points, the points lie on the diagonal. Accuracy is identical (0.6664) in both</text>
  <text x="60" y="328" fill="currentColor" font-size="11.5">panels: temperature scaling is monotone in the logits, so it cannot change any argmax.</text>
</svg>

### 13.10.3 Conformal prediction: a guarantee that needs no model assumptions

Calibration is about probabilities. Conformal prediction is about *sets*, and it offers the strongest guarantee available anywhere in this chapter: finite-sample, distribution-free coverage.

**Definition 13.53 (exchangeability).** Random variables $Z_1,\dots,Z_m$ are *exchangeable* if for every permutation $\pi$ of $\{1,\dots,m\}$, $(Z_{\pi(1)},\dots,Z_{\pi(m)}) \stackrel{d}{=} (Z_1,\dots,Z_m)$. I.i.d. implies exchangeable; exchangeable does not imply independent (draws without replacement from a finite population, or any conditionally-i.i.d. mixture, are exchangeable).

**Theorem 13.54 (split-conformal coverage).** Let $(X_1,Y_1),\dots,(X_n,Y_n),(X_{n+1},Y_{n+1})$ be exchangeable. Let $s : \mathcal X\times\mathcal Y\to\mathbb R$ be any *fixed* score function (fitted on data independent of these $n+1$ points), with larger values meaning worse conformity. Put $S_i := s(X_i,Y_i)$ for $i \le n$, let
$$
\hat q := S_{(\lceil (n+1)(1-\alpha)\rceil)} \quad(\text{the } \lceil (n+1)(1-\alpha)\rceil\text{-th smallest of } S_1,\dots,S_n),
$$
and define $C(x) := \{y : s(x,y) \le \hat q\}$. Then
$$
1-\alpha \;\le\; P\big(Y_{n+1}\in C(X_{n+1})\big) \;\le\; 1-\alpha+\frac{1}{n+1},
$$
the upper bound holding when the scores are almost surely distinct.

*Proof.* Let $k := \lceil (n+1)(1-\alpha)\rceil$ and note $Y_{n+1}\in C(X_{n+1})$ iff $S_{n+1} \le \hat q$, i.e. iff $S_{n+1}$ is at most the $k$-th smallest of $S_1,\dots,S_n$. Equivalently, in the pooled list $S_1,\dots,S_{n+1}$, the rank of $S_{n+1}$ is at most $k$. [Check: if $S_{n+1} \le S_{(k)}$ then at most $k-1$ of $S_1..S_n$ are $<S_{n+1}$ plus itself, so its pooled rank is $\le k$; conversely if the pooled rank is $\le k$ then at least $n+1-k$ of the others are $\ge S_{n+1}$, so $S_{n+1}\le S_{(k)}$.]

Because $s$ is a fixed function and the $n+1$ pairs are exchangeable, the scores $S_1,\dots,S_{n+1}$ are exchangeable. If they are almost surely distinct, the rank of $S_{n+1}$ in the pooled list is uniform on $\{1,\dots,n+1\}$: indeed, exchangeability makes all $(n+1)!$ orderings equally likely, hence all $n+1$ possible ranks for a designated element equally likely. Therefore
$$
P(\text{rank}(S_{n+1}) \le k) = \frac{k}{n+1} = \frac{\lceil (n+1)(1-\alpha)\rceil}{n+1} \in \Big[1-\alpha,\ 1-\alpha+\frac1{n+1}\Big),
$$
where the lower bound is $\lceil u\rceil \ge u$ and the upper is $\lceil u\rceil < u+1$. With ties, break them at random (or note that ties can only *increase* the coverage probability, preserving the lower bound). $\square$

Look at what the proof used: **exchangeability of the $n+1$ points, and nothing else.** No assumption on $P$, no assumption that the model is any good, no asymptotics, no smoothness. That is why conformal prediction has spread so fast. And look at what it did *not* use: any relation between $X$ and the score's shape. That omission is exactly the limitation.

**Worked Example 13.27 (verifying the guarantee, including its upper bound).** Regression setup: $x\sim\mathrm{Unif}(-3,3)$, $y = \sin x + \sigma(x)\xi$ with heteroscedastic $\sigma(x) = 0.15 + 1.5\,\mathbf 1\{x>0\}\,x/3$; score $s(x,y) = |y - \sin x|$ (I use the *true* regression function as the model so that all miscoverage comes from the noise, not from fit error). 200 replicates, 2000 test points each:

| $\alpha$ | $n_{\mathrm{cal}}$ | mean $\hat q$ | empirical coverage (sd over reps) | theoretical band $[1-\alpha,\ 1-\alpha+\tfrac1{n+1}]$ |
|---|---|---|---|---|
| 0.20 | 100 | 0.689 | 0.8023 (0.0398) | $[0.8000, 0.8099]$ |
| 0.20 | 1000 | 0.666 | 0.8006 (0.0155) | $[0.8000, 0.8010]$ |
| 0.10 | 100 | 1.214 | 0.9009 (0.0292) | $[0.9000, 0.9099]$ |
| 0.10 | 1000 | 1.178 | 0.8993 (0.0119) | $[0.9000, 0.9010]$ |
| 0.05 | 100 | 1.751 | 0.9514 (0.0206) | $[0.9500, 0.9599]$ |
| 0.05 | 1000 | 1.677 | 0.9500 (0.0087) | $[0.9500, 0.9510]$ |

Every empirical coverage lands in (or, at $\alpha=0.10, n=1000$, within Monte Carlo error of) the predicted band. Note the sd column: the *marginal* guarantee is about the average over calibration sets, and with $n_{\mathrm{cal}}=100$ the coverage of any *particular* calibration set varies with sd $\approx 0.03$. If you need a guarantee for your one realized calibration set, you need the beta-distribution refinement of the theorem, not the marginal statement.

**Worked Example 13.28 (the limitation, measured).** Same setup, $\alpha = 0.10$, $n_{\mathrm{cal}} = 2000$, 200,000 test points, giving $\hat q = 1.231$. Coverage *conditional on $x$*:

| region | coverage, score $|y-\hat\mu(x)|$ | coverage, normalized score $|y-\hat\mu(x)|/\hat\sigma(x)$ |
|---|---|---|
| $x\in(-3,-2]$ | 1.0000 | 0.9058 |
| $x\in(-2,-1]$ | 1.0000 | 0.9058 |
| $x\in(-1,0]$ | 1.0000 | 0.9035 |
| $x\in(0,1]$ | 0.9888 | 0.9068 |
| $x\in(1,2]$ | 0.8263 | 0.9037 |
| $x\in(2,3]$ | **0.6244** | 0.9075 |
| overall | 0.9065 | 0.9055 |

The theorem is satisfied — overall coverage 0.9065 — and yet in the noisiest region the intervals cover only 62% of the time, while in the quiet region they cover 100% and are absurdly wide. Marginal coverage was achieved by *averaging a systematic over-coverage against a systematic under-coverage*. The right column shows the fix: use a locally adaptive score $s(x,y) = |y-\hat\mu(x)|/\hat\sigma(x)$ with a fitted uncertainty model $\hat\sigma$ (here, the true $\sigma$). Coverage becomes essentially uniform across regions at 0.90. **The theorem's guarantee does not improve; what improves is the thing the theorem never promised.** This is the practical reason conformalized quantile regression and normalized scores are the default in serious applications.

Finally, distribution shift. Exchangeability of the calibration and test points is the *only* assumption, so it is also the only thing that can break — and it breaks whenever the test distribution differs from the calibration distribution.

**Same setup, shifted test:** calibrate on $x\sim\mathrm{Unif}(-3,3)$ but test on $x\sim\mathrm{Unif}(1,3)$ (the noisy half). Nominal 90%; measured coverage $0.7348$ over 100,000 test points. Nothing about the construction warns you: $\hat q$ is unchanged, the intervals look the same, and the guarantee has silently evaporated. Weighted conformal prediction restores validity under *covariate* shift if you can estimate the density ratio $dP^{\mathrm{test}}_X/dP^{\mathrm{cal}}_X$ (which is Chapter 2's change of measure again, with all the $P\ll Q$ caveats of §13.2), and there is no general fix for label shift or concept drift. In deployment terms: **conformal prediction converts "trust my model" into "trust that tomorrow's inputs look like yesterday's", which is a real and much smaller assumption, but it is not nothing.**

> **Common Confusion 13.8.** *"My conformal predictor has 90% coverage, so for this patient there's a 90% chance the true value is in the interval."*
>
> No. Theorem 13.54 is a statement about the *marginal* probability over the joint draw of calibration set and test point. Worked Example 13.28 exhibits a predictor with exactly 90% marginal coverage whose coverage is 62% for one identifiable subpopulation and 100% for another. If your test point is in the hard subpopulation, your interval is a 62% interval, and no amount of conformal machinery detects this — indeed *no* distribution-free method can provide non-trivial conditional coverage for all $x$ without additional assumptions (a known impossibility result: exact conditional coverage requires either infinite-width intervals or distributional assumptions). What you can do: condition on features you care about by running conformal separately within groups (Mondrian/group-conditional conformal), which buys you group-wise validity at the cost of splitting the calibration data; use normalized scores to make coverage approximately uniform; and report the *conditional* coverage you can measure on held-out data by region, as in the table above. Connect this to Chapter 9: the situation is exactly that of a confidence interval's coverage being a property of the procedure, not of the realized interval, and to Chapter 10: if your decision problem has asymmetric costs, a marginal-coverage set is not the Bayes-optimal action, and you should be doing decision theory with a predictive distribution rather than reading an interval.

> **Self-check 13.10.**
> 1. Why can temperature scaling never change a model's accuracy? What does it change?
> 2. A paper reports ECE 0.015 on 1000 test examples and claims near-perfect calibration. What is your objection?
> 3. In Theorem 13.54, where exactly is exchangeability used, and what replaces it if you have time-series data?
> 4. Give a concrete deployment scenario in which 90% marginal coverage is dangerous, and say what you would compute instead.

---

## 13.11 Closing: the toolkit assembled

Everything in this chapter was borrowed. Here is the ledger, tool by tool, so you can see the architecture of the twelve chapters that preceded it.

| Tool used here | Where it appears in this chapter | Chapter that built it |
|---|---|---|
| A random variable is a deterministic function of $\omega$ | the reparameterization trick's whole logic (Prop. 13.36) | Ch. 1 (Def. 1.13) |
| Pushforward / law | Prop. 13.36; the generator as $G_\#N(0,I)$ | Ch. 1 (Def. 1.16) |
| Support, and "measure zero $\ne$ impossible" | §13.9.3, Worked Ex. 13.21, 13.22, 13.23 | Ch. 1 (Def. 1.17) |
| Bayes' rule | Prop. 13.44 (the forward posterior) | Ch. 1 (Prop. 1.12) |
| Dirac measure, mixtures of atoms and densities | the MH kernel (Def. 13.20) | Ch. 1 + Ch. 2 |
| Lebesgue integral, DCT, differentiating under $\mathbb{E}$ | Props. 13.36, 13.41; §13.1 throughout | Ch. 2 |
| Change of measure, $P\ll Q$ | §13.2 entirely; weighted conformal | Ch. 2 |
| Radon–Nikodym, naming a base measure | §13.2.1; Def. 13.20's kernel decomposition | Ch. 2 |
| REINFORCE and the mean-zero score | §13.2.4 (baselines), §13.8.2 (discrete latents) | Ch. 2 |
| The ELBO identity | Thm. 13.31, Thm. 13.45 | Ch. 2 |
| Change of variables and Jacobians | Prop. 13.24 (leapfrog), Prop. 13.46 (probability flow) | Ch. 3 |
| Normalizing flows | Prop. 13.46 | Ch. 3 |
| Gumbel-Softmax | §13.8.2, discrete latents | Ch. 3 |
| Concentration for error bars | Exercise B1's alternative to CLT bars | Ch. 4 |
| Gaussian identities: products, conditioning, KL | Prop. 13.44, Thm. 13.45, Worked Ex. 13.12 | Ch. 5 |
| Conditional expectation as $L^2$ projection | Prop. 13.41 (DSM), Prop. 13.10 (Rao–Blackwell) | Ch. 6 |
| Law of total variance | Prop. 13.10, Def. 13.47 | Ch. 6 |
| SGD convergence | SVI (§13.7.3), SGLD (§13.5.2) | Ch. 6 |
| CLT and Slutsky | Prop. 13.3, Thm. 13.19's CLT | Ch. 7 |
| EM | Thm. 13.31(c) | Ch. 8 |
| Rao–Blackwellization | §13.2.4, analytic KL in the ELBO | Ch. 8 |
| Coverage as a property of a procedure | §13.10.3, Common Confusion 13.8 | Ch. 9 |
| Posteriors, predictive distributions, Laplace | §13.6's cross-checks, §13.10.1 | Ch. 10 |
| Decision theory | Common Confusion 13.8 | Ch. 10 |
| KL asymmetry, mode-seeking | Prop. 13.34, §13.8.2 | Ch. 11 |
| Mutual information, rate–distortion | Prop. 13.37, blurry samples | Ch. 11 |
| Jensen–Shannon on disjoint supports | §13.9.3 | Ch. 11 |
| High-dimensional geometry; the caution about over-reading it | Prop. 13.13, Common Confusion 13.5 | Ch. 12 |

And here, honestly, is what remains open. **Convergence rates for MCMC on realistic posteriors** are essentially unknown; every practical guarantee is asymptotic or relies on unverifiable mixing conditions, which is why §13.6 is a checklist rather than a theorem. **The variational gap is hard to measure** in the models where it matters. This deserves a caveat, because the flat claim that it is unmeasurable is too strong: annealed importance sampling and bidirectional Monte Carlo can *sandwich* $\log p(x)$ between stochastic lower and upper bounds, and IWAE with large $K$ gives an increasingly tight lower bound (Prop. 13.38). What is true is that these estimators are expensive, are themselves importance-sampling schemes subject to §13.2.2's weight degeneracy, and are almost never run — so in practice a reported ELBO comes with no measurement of its gap, and the amortization gap is rarely bounded at all. **Diffusion sampling theory** is young: we have convergence results for score estimation error under smoothness assumptions, but nothing that explains why 50-step samplers work as well as they do, and nothing that explains guidance. **Uncertainty under distribution shift** has no distribution-free solution and, by the impossibility result cited in Common Confusion 13.8, cannot have one without assumptions. **Whether deep-network posteriors are worth approximating at all** is genuinely contested: the modes are symmetric-equivalent in enormous numbers, and it is not clear that "the posterior" is the right object rather than an ensemble or a calibrated point predictor. Do not let the tidiness of the theorems in this chapter suggest that the practice is settled. It is not.

**What this chapter left out, and where to find it.** Being explicit about the omissions is part of being self-contained. *Sequential Monte Carlo* (particle filters) is the natural synthesis of §13.2 and §13.4 — importance sampling with resampling, propagated through time — and is the standard tool for state-space and time-series models; nothing here covers it, and Chopin & Papaspiliopoulos is the entry point. *Annealed importance sampling* and *bridge/path sampling* estimate normalizing constants and marginal likelihoods, which is exactly the quantity §13.7 could only bound. *Quasi-Monte Carlo* replaces randomness with low-discrepancy structure and can approach $O(n^{-1})$ for smooth low-effective-dimension integrands; Owen's book covers it properly. *Parallel and simulated tempering* got a paragraph in §13.4.4 and deserve a chapter. On the modelling side, *normalizing flows*, *autoregressive models*, and *GANs* appear here only as context (§13.9's preamble) or as a diagnosed failure (§13.9.3); *flow matching*, which now subsumes diffusion as one choice of probability path, is deferred to the closing “Where to go next” section. None of these omissions changes any statement in this chapter, but a reader who thinks Monte Carlo ends at MCMC and generative modelling ends at diffusion has been misled by the table of contents rather than by the text.

---

## 13.12 Results used but not proved

| Result | Used in | Where to find a proof |
|---|---|---|
| Strong law of large numbers | Prop. 13.2, Prop. 13.7, Prop. 13.38 | Ch. 7 of this tutorial; Durrett, *Probability: Theory and Examples*, §2.4 |
| Central limit theorem; Slutsky's theorem | Prop. 13.3 | Ch. 7; Durrett §3.4, van der Vaart §2.8 |
| Ergodic theorem for Harris-recurrent chains; TV convergence; CLT for geometrically ergodic chains | Thm. 13.19 and everything that averages along a chain | Meyn & Tweedie, *Markov Chains and Stochastic Stability*, Chs. 13, 17; Roberts & Rosenthal (2004) survey, Thms. 4–5, 24 |
| Existence/uniqueness of the Langevin diffusion and its convergence to $\pi$ (not merely stationarity) | §13.5.2 | Bakry–Gentil–Ledoux, *Analysis and Geometry of Markov Diffusion Operators*, Ch. 1; Pavliotis, *Stochastic Processes and Applications*, Ch. 4 |
| Anderson's time-reversal theorem for SDEs; equality of marginals for the probability-flow ODE | Prop. 13.46 | Anderson (1982), *Stochastic Processes and their Applications*; Song et al. (2021), App. D |
| Weak-convergence/SDE approximation of SGD | Common Confusion 13.5 | Li, Tai & E (2019), *JMLR* 20(40) |
| Every $\pi$-irreducible Metropolis–Hastings chain is Harris recurrent | §13.4.2 (checking Theorem 13.19's hypotheses) | Tierney (1994), *Ann. Statist.* 22, Corollary 2; Roberts & Rosenthal (2004), §20 |
| Optimal scaling results ($0.234$ for RWMH; $d^{-1/4}$ for HMC) | §13.4.3, §13.5.1 | Roberts, Gelman & Gilks (1997); Beskos et al. (2013) |
| Zhang–Stephens generalized-Pareto estimator and the $\hat k$ diagnostic's calibration | Def. 13.6, Worked Ex. 13.6 | Zhang & Stephens (2009); Vehtari et al. (2024), *JMLR* 25(72) |
| NUTS's validity (that the doubling scheme preserves detailed balance) | §13.5.1 | Hoffman & Gelman (2014), *JMLR* 15, Thm. 1 |
| Uniform integrability step in the $o(1/n)$ bias expansion | Prop. 13.7 | Owen, *Monte Carlo theory, methods and examples*, Ch. 9 |
| Convergence $\mathcal L_K \to \log p(x)$ for IWAE | Prop. 13.38 | Burda, Grosse & Salakhutdinov (2016), App. A |
| IWAE encoder-gradient SNR decay $O(K^{-1/2})$ | §13.8.2 | Rainforth et al. (2018), *ICML* |
| Impossibility of non-trivial distribution-free *conditional* coverage | Common Confusion 13.8 | Vovk (2012); Barber, Candès, Ramdas & Tibshirani (2021), *Information and Inference* |
| Weighted conformal prediction under covariate shift | §13.10.3 | Tibshirani, Barber, Candès & Ramdas (2019), *NeurIPS* |
| Existence of the divergence/entropy decomposition for general proper scoring rules | Prop. 13.51 (proved for Brier and log only) | Gneiting & Raftery (2007), *JASA* 102(477) |
| Non-homeomorphism of $\mathbb R$ and $S^1$ (used topologically) | Worked Ex. 13.21 | Munkres, *Topology*, §24 (connectedness argument) |
| Peskun–Tierney ordering: a pointwise-larger acceptance function gives smaller asymptotic variance for every $L^2(\pi)$ functional | Solution C1, Step 3 | Peskun (1973), *Biometrika* 60; Tierney (1998), *Ann. Appl. Prob.* 8 |
| Validity and consistency of Geyer's initial-positive-sequence estimator of $\tau$ | every ESS number in §§13.4–13.6 and Labs B2–B3 | Geyer (1992), *Statistical Science* 7; Geyer's chapter in *Handbook of MCMC* (2011) |
| Volume of the unit $\ell^2$ ball, $V_d=\pi^{d/2}/\Gamma(d/2+1)$, and its decay | Prop. 13.13's acceptance table | Ch. 12 of this tutorial; Ball, *An Elementary Introduction to Modern Convex Geometry*, Lecture 1 |
| Self-normalized limit theory for infinite-variance sums | Solution A2(c) | Logan, Mallows, Rice & Shepp (1973), *Ann. Prob.* 1 |
| Existence of exact samplers for the Gaussian and Gamma laws used throughout (Box–Muller, Marsaglia–Tsang) | every experiment | Devroye, *Non-Uniform Random Variate Generation*, Chs. 4, 9 |

---

## 13.13 Chapter summary

1. **Monte Carlo trades rate for dimension-independence.** The error is $\sigma/\sqrt n$ with a constant that does not know $d$, which is why it beats quadrature past $d\approx5$ and why it is the only option in modern ML. It buys nothing if you cannot draw the samples.
2. **Finite variance is a hypothesis with teeth, and importance sampling is where it bites.** With $\sigma^2=\infty$ the estimator still converges but the CLT interval silently under-covers forever: measured coverage $0.63$–$0.67$ for a nominal $0.95$, flat from $n=10^3$ to $n=10^6$ (Worked Ex. 13.4, Solution A2). Importance sampling needs $P\ll Q$ and has variance governed by $\mathbb{E}_P[f^2w]$, so light-tailed proposals for heavy-tailed targets are exactly the case where the second moment diverges. And even a well-located proposal degrades exponentially in dimension: a 20% per-coordinate scale mismatch inflates $\mathbb E[w^2]$ by $\approx 4$ at $d=100$ and $\approx 1.3\times10^6$ at $d=1000$, in closed form (Worked Ex. 13.7). Diagnose with ESS *and* $\hat k$: ESS estimates a quantity that may not exist, $\hat k$ estimates whether it exists.
3. **The four variance-reduction techniques are all one idea — use structure you know.** Measured on one integral with a fixed evaluation budget: antithetic $29.8\times$, one control variate $62.3\times$, two control variates $8718\times$, 40 strata $1470\times$ (Worked Ex. 13.10). Antithetic sampling *doubles* the variance for symmetric integrands.
4. **Rejection sampling is exact and unusable.** A proposal wrong by 10% per coordinate costs $1.1^{d}$; the unit ball in the cube has acceptance $10^{-28}$ at $d=50$. Hence MCMC.
5. **Detailed balance buys invariance in one line**, and Metropolis–Hastings is the unique-looking construction that makes $\pi(x)q(x,y)\alpha(x,y)$ symmetric: $\min\{\pi(x)q(x,y),\pi(y)q(y,x)\}$. Only ratios of $\pi$ appear, which is why unnormalized targets are fine.
6. **Invariance is not convergence.** A chain can have the right stationary distribution and never find it; tuning is about ESS per evaluation, and the famous $0.234$ acceptance rate is an asymptotic result for product targets, not a law (measured optimum on a 2-D banana: acceptance 4–17%).
7. **Gradients change the scaling.** HMC's leapfrog integrator is volume-preserving and reversible, so the MH correction only has to absorb $O(\epsilon^2)$ energy error. Measured from $d=2$ to $d=200$: RWMH efficiency per evaluation fell $50\times$, HMC's fell $4\times$ (Worked Ex. 13.14).
8. **Unadjusted samplers carry systematic bias that more samples never remove.** ULA on $N(0,1)$ with step $h$ targets $N(0,1/(1-h/4))$ exactly — at $h=1$ a 15% inflation of the reported sd — while MALA is exact for one extra density evaluation. Constant-step SGD is the same phenomenon without the theorem: for a quadratic with additive noise its stationary variance is provably $\eta s^2/(a(2-\eta a))$ (verified to three decimals), and beyond that case "SGD is a sampler" is an analogy. State which of the two you are invoking.
9. **No diagnostic proves convergence.** Split-$\hat R$, ESS, and divergence counts detect *inconsistency between things you observed*; all of them pass when every chain sits in the same wrong mode. Run at least four dispersed chains.
10. **The ELBO has three faces** — Jensen, the exact identity $\log p(x) = \mathcal L + D_{\mathrm{KL}}(q\|p(\cdot\mid x))$, and EM with an inexact E-step. The second is the one that tells you the error's size and sign; the first is the one that generalizes to IWAE.
11. **Mean-field VI reports conditional variances in place of marginal ones**, so it is systematically overconfident: exactly $1-\rho^2$ instead of $1$ for a correlated Gaussian pair (a $7\times$ too-narrow interval at $\rho=0.99$). Measured gap in a conjugate model: 8% too small on a $0.021$-nat KL gap.
12. **A VAE's KL term is an information bottleneck**, not a free regularizer: the average KL equals $I_q(X;Z)$ plus an aggregate-posterior mismatch, so driving it to zero forces the latent to be independent of the data. Deliberately induced at $\beta=100$: KL $=2\times10^{-4}$ nats and a decoder that reproduces the data mean (Worked Ex. 13.20).
13. **Denoising is score estimation.** Vincent's identity turns an intractable divergence into a regression, and — decisively — it needs no density on the data, because Gaussian convolution manufactures one. DDPM's objective follows from a variational bound plus a specific list of modelling choices, of which dropping the $t$-weighting is the one that breaks the bound.
14. **Diffusion circumvents Chapter 1's measure-zero obstruction by smoothing — it does not remove it.** At every $\sigma>0$ the target has full-dimensional support, so scores, KLs, and likelihoods exist; the difficulty returns only as $\sigma\to0$, where the score diverges like $1/\sigma^2$ (measured: $10.5$ at $\sigma=0.1$ and $1000.5$ at $\sigma=0.01$ at fixed distance $0.1$ from the manifold).
15. **Calibration is cheap and ECE is a biased metric.** One scalar temperature (fitted $2.4838$ against a true $2.5$) cut ECE from $0.184$ to $0.011$ without changing accuracy; a *perfectly* calibrated model scores ECE $0.111$ on 100 test points, so small ECE differences are noise.
16. **Split-conformal coverage is a real finite-sample guarantee that uses only exchangeability** — verified inside its $[1-\alpha, 1-\alpha+1/(n+1)]$ band at three levels — and it is *marginal*: the same predictor covered 100% in a quiet region and 62% in a noisy one, and dropped to 73% under covariate shift.
### Callbacks

This chapter discharged the promissory notes of the whole tutorial. Chapter 1's four failures are now fully closed: the missing-density problem (Failure 1) and the "neither discrete nor continuous" problem (Failure 2) reappear here as the Metropolis–Hastings kernel, which is genuinely a mixture of a density and an atom and which we could write down only because Chapter 2 gave us base measures; the grad/$\mathbb{E}$ swap (Failure 3) is the engine of the reparameterized VAE gradient (Proposition 13.36); and the GAN's constant loss on disjoint supports (Failure 4) is *circumvented* — not resolved — by §13.9.3's observation that Gaussian smoothing gives full-dimensional support at every noise level, so that the objective never compares two mutually singular measures. It is worth being exact about what did and did not change. The geometric fact is untouched: the data still live on a measure-zero manifold. What changed is that diffusion never needs a density for the data, and its own output law is never singular either — it is a thin full-dimensional blanket over the manifold, of thickness set by $\sigma_{\min}$. Worked Example 13.24 shows the obstruction reappearing in the $\sigma\to0$ limit as a score that diverges like $1/\sigma^2$, which is the price of the circumvention rather than evidence against it. Chapter 1's circle generator, introduced to show that pushforwards of low-dimensional latents have Lebesgue-null support, came back twice: as the data for a VAE whose 1-D latent provably cannot cover it (Worked Example 13.21) and as the data for a DDPM that fits its support to within 1.7% (Worked Example 13.23). Chapter 2's warning about infinite-variance importance sampling became Worked Example 13.4's measured coverage failure and §13.2.2's diagnostics. Chapter 8 promised that EM was a special case of something larger; Theorem 13.31(c) delivered it. Chapter 11 promised that KL's asymmetry had operational consequences; Proposition 13.34 turned it into a formula for how wrong your error bars are. Chapter 12 promised that high-dimensional geometry would explain algorithmic choices; Proposition 13.13 and Worked Example 13.14 are that explanation.

### Before you continue

1. Why does the Monte Carlo error rate not depend on dimension, and name two ways dimension nevertheless hurts.
2. Write the importance-sampling identity and state the hypothesis that makes it valid. What quantity controls the estimator's variance?
3. Derive the Metropolis–Hastings acceptance probability from the requirement that $\pi(x)q(x,y)\alpha(x,y)$ be symmetric in $(x,y)$.
4. Why does HMC accept nearly everything at small $\epsilon$, and what exactly does the acceptance probability measure?
5. State the ELBO identity and say what the gap equals. Which direction of KL is it, and what does that direction do to reported posterior variances?
6. Explain in two sentences why a powerful decoder encourages posterior collapse.
7. Write down Vincent's identity in words: what does a network trained to predict the added noise actually learn?
8. Why does a DDPM have a well-defined likelihood and score at every $t>0$ even when the data distribution has no density at all?
9. What exactly does split-conformal prediction guarantee, what does it assume, and what does it not promise?

### Looking ahead: reading papers

There is no Chapter 14. This was the capstone, and the honest thing to say is that the next chapter is a paper — yours or someone else's — read with the tools you now have. What changes is your reading: when a paper writes $\nabla_\theta\mathbb{E}_{q_\theta}[f]$ you will ask which estimator and what its variance is; when it reports an ELBO you will ask about the gap; when it reports ECE you will ask about $n$ and the bin count; when it reports a posterior you will ask for $\hat R$ and dispersed initialization; when it claims a coverage guarantee you will ask marginal over what. The final section below is a map of where to go to keep going.

---


# Exercises

Difficulty: ★ routine consolidation, ★★ requires an idea, ★★★ hard or open-ended. Every exercise has a full solution at the end of the chapter; the solutions explain the *thinking*, not just the answer, so read them only after a serious attempt.

## Part A — Theory

**A1 (★).** Let $X_1,\dots,X_n$ be i.i.d. with $\mathbb E[f(X)^2]<\infty$ and let $\hat\theta_n = n^{-1}\sum_i f(X_i)$, $s_n^2 = (n-1)^{-1}\sum_i (f(X_i)-\hat\theta_n)^2$.
(a) Prove $\mathbb E[\hat\theta_n]=\theta$ and $\operatorname{Var}(\hat\theta_n)=\sigma^2/n$.
(b) Prove $\mathbb E[s_n^2]=\sigma^2$, being explicit about where the $n-1$ comes from.
(c) A colleague computes $s_n^2$ with divisor $n$ and reports error bars $1.96 s_n/\sqrt n$. By what factor are the bars wrong at $n=4$ and at $n=4000$? Is this the error worth worrying about?

**A2 (★★).** Let $U\sim\mathrm{Unif}(0,1)$ and $f(u)=u^{-\gamma}$.
(a) For which $\gamma$ is $\mathbb E[f(U)]<\infty$? For which is $\operatorname{Var}(f(U))<\infty$?
(b) Take $\gamma=3/4$. Show $\mathbb E[f(U)]=4$ and $\operatorname{Var}=\infty$. Explain, using the SLLN and Proposition 13.3, exactly which conclusion survives and which fails.
(c) The measured coverage of the nominal-$95\%$ CLT interval in Worked Example 13.4 was $0.648$ at $n=1000$. Predict qualitatively what happens to this number as $n\to\infty$ and justify your prediction. (Hint: what is the limiting law of $n^{1/\gamma-1}(\hat\theta_n-\theta)$?)

**A3 (★★).** Consider importance sampling of $\theta=\mathbb E_P[f]$ with proposal density $q$.
(a) Derive $\operatorname{Var}(\hat\theta_n^{\mathrm{IS}}) = n^{-1}\big(\mathbb E_Q[f^2w^2]-\theta^2\big)$ where $w=p/q$.
(b) Minimize over $q$ subject to $\int q=1$ and show that the optimum is $q^\star \propto |f|p$.
(c) Show that if $f\ge0$ the optimal variance is exactly $0$, and explain why this is useless in practice rather than a triumph.
(d) Show that $q^\star$ for $f=\mathbf 1_A$ is $P$ restricted to $A$, and connect this to Worked Example 13.5's shifted Gaussian.

**A4 (★).** Let $h$ have known mean $\mu_h$ and define $\hat\theta_c = n^{-1}\sum_i\big(f(X_i)-c(h(X_i)-\mu_h)\big)$.
(a) Show $\hat\theta_c$ is unbiased for every $c$.
(b) Derive $c^\star=\operatorname{Cov}(f,h)/\operatorname{Var}(h)$ and show the variance ratio is $1-\rho^2$.
(c) Estimating $c^\star$ from the same samples introduces bias of order $1/n$. Explain why this is usually acceptable, and describe a splitting scheme that removes it exactly.

**A5 (★★).** Antithetic sampling of $\mathbb E[f(U)]$, $U\sim\mathrm{Unif}(0,1)$, averages $\tfrac12(f(U)+f(1-U))$.
(a) Show that per *evaluation*, antithetic sampling beats plain Monte Carlo if and only if $\operatorname{Cov}(f(U),f(1-U))<0$.
(b) Prove that this covariance is $\le0$ whenever $f$ is monotone. (Hint: for monotone $f$ and $g$ with $g$ decreasing, consider $\big(f(u)-f(v)\big)\big(f(1-u)-f(1-v)\big)$ and integrate over $(u,v)$.)
(c) Compute the covariance for $f(u)=(u-\tfrac12)^2$ and reconcile with the measured factor of two *loss* reported in §13.2.4.

**A6 (★★).** Partition $(0,1)$ into $K$ equal strata and draw one uniform point in each, giving $\hat\theta^{\mathrm{str}}=K^{-1}\sum_k f(U_k)$ with $U_k\sim\mathrm{Unif}((k-1)/K,k/K)$.
(a) Show $\hat\theta^{\mathrm{str}}$ is unbiased.
(b) Show its variance is $K^{-2}\sum_k \operatorname{Var}(f(U_k))$ and that this equals $K^{-1}\mathbb E[\text{within-stratum variance}]$, hence is never larger than the plain variance $\operatorname{Var}(f(U))/K$ at the same cost.
(c) Identify the exact quantity that stratification removes, and give an $f$ for which two strata give variance exactly zero.

**A7 (★★).** In the setting of Theorem 13.12, let $N$ be the number of proposals until the first acceptance and $Y$ the returned draw.
(a) Show $N\sim\mathrm{Geometric}(1/M)$ with $\mathbb E[N]=M$.
(b) Show that $Y$ and $N$ are independent, and explain why this is what licenses the naive `while` loop.
(c) Suppose you cap the loop at $N\le100$ and return the last proposal on failure. Show the output is biased and describe the bias's direction for a target with heavier tails than the proposal.

**A8 (★★).** Let $K$ be a Markov kernel on a countable state space.
(a) Prove that detailed balance with respect to $\pi$ implies $\pi K=\pi$.
(b) Exhibit a kernel on $\{1,2,3\}$ that is invariant for the uniform distribution but does *not* satisfy detailed balance. Is it irreducible? Aperiodic?
(c) Explain what practical advantage a non-reversible chain can have, referring to the ordered-scan Gibbs sampler of Proposition 13.22.

**A9 (★★).** Take an *independence* MH proposal $q(x,y)=q(y)$ not depending on the current state, and set $w(x)=\pi(x)/q(x)$.
(a) Show the acceptance probability is $\min\{1,w(y)/w(x)\}$.
(b) Show the chain is irreducible whenever $\{\pi>0\}\subseteq\{q>0\}$, and relate this to the hypothesis of Proposition 13.4.
(c) Both this sampler and importance sampling use $w$. Explain what the MH version buys and what it gives up, and say what happens to each when $w$ is unbounded.

**A10 (★★).** Let $\Phi_{\epsilon}$ be one leapfrog step of Definition 13.23 and $F(x,p)=(x,-p)$.
(a) Compute the Jacobian of each of the three sub-steps and conclude $|\det D\Phi_\epsilon|=1$.
(b) Show $(F\circ\Phi_\epsilon^L)\circ(F\circ\Phi_\epsilon^L)=\mathrm{id}$, i.e. the full proposal map is an involution.
(c) Explain precisely why (a) and (b) together let the MH ratio reduce to $\exp(H(x,p)-H(x',p'))$, and what would appear in the ratio if the integrator were merely accurate but not volume-preserving.

**A11 (★★).** Consider ULA on $\pi=N(0,\sigma^2)$: $x_{k+1}=x_k - \tfrac h2 x_k/\sigma^2 + \sqrt h\,\xi_k$.
(a) Find the stationary variance $v(h)$ exactly, and give the condition on $h$ for stability.
(b) Specialize to $\sigma^2=1$ and confirm Proposition 13.27's $1/(1-h/4)$.
(c) For a $d$-dimensional Gaussian with eigenvalues $\sigma_1^2\ge\cdots\ge\sigma_d^2$, find the largest stable $h$ and the resulting relative bias in the *largest* variance. Conclude how ULA's usable step size depends on the condition number.

**A12 (★).** Prove Theorem 13.31(b), i.e. $\log p(x)=\mathcal L(q)+D_{\mathrm{KL}}(q\|p(\cdot\mid x))$, directly from the definition of $\mathcal L$.
(a) Deduce that maximizing $\mathcal L$ over $q$ is equivalent to minimizing that KL.
(b) Show that enlarging $\mathcal Q$ can never decrease $\max_{q\in\mathcal Q}\mathcal L$, and determine whether the gap can increase when $\mathcal Q$ grows.
(c) Two models are compared by their ELBOs. Give a concrete situation in which the model with the larger ELBO has the *smaller* marginal likelihood.

**A13 (★★).** Let $p=N(0,\Sigma)$ on $\mathbb R^d$ with precision $\Lambda=\Sigma^{-1}$, and let $q=\prod_i N(m_i,s_i^2)$ minimize $D_{\mathrm{KL}}(q\|p)$.
(a) Show $m=0$ and $s_i^2=1/\Lambda_{ii}$.
(b) Interpret $1/\Lambda_{ii}$ probabilistically and explain in one sentence why mean-field VI is therefore systematically overconfident.
(c) Specialize to $d=2$ with correlation $\rho$ to recover $s^2=1-\rho^2$ and $D_{\mathrm{KL}}=-\tfrac12\log(1-\rho^2)$, and compute both at $\rho=0.99$.

**A14 (★★).** Let $q(z\mid x)$ be an encoder, $q(z)=\mathbb E_{x\sim\mathcal D}[q(z\mid x)]$ the aggregate posterior, and $p(z)$ the prior. Prove
$$\mathbb E_{x}\big[D_{\mathrm{KL}}(q(z\mid x)\,\|\,p(z))\big] = I_q(X;Z) + D_{\mathrm{KL}}(q(z)\,\|\,p(z)).$$
(a) Give the proof.
(b) Deduce that a per-datapoint KL of $0$ for ($\mathcal D$-almost) all $x$ forces $I_q(X;Z)=0$.
(c) Explain why this makes "the KL term is just a regularizer, so weight it however you like" a dangerous statement, and connect the two terms to Chapter 11's rate–distortion picture.

**A15 (★★★).** For $\mathcal L_K(x)=\mathbb E\big[\log\frac1K\sum_{k=1}^K w_k\big]$ with $w_k=p(x,z_k)/q(z_k\mid x)$ i.i.d.:
(a) Show $\mathcal L_K\le\log p(x)$ for every $K$.
(b) Show $\mathcal L_K$ is non-decreasing in $K$. (Hint: write $\frac1{K+1}\sum_{k=1}^{K+1}w_k$ as an average over the $K+1$ subsets of size $K$ of their means, then apply Jensen conditionally.)
(c) Explain what additional hypothesis is needed for $\mathcal L_K\to\log p(x)$, and why (b) does not by itself imply that more is always better in practice.

**A16 (★★).** Let $S_1,\dots,S_n,S_{n+1}$ be exchangeable real random variables, a.s. distinct, and let $\hat q$ be the $\lceil(n+1)(1-\alpha)\rceil$-th smallest of $S_1,\dots,S_n$.
(a) Prove $\mathbb P(S_{n+1}\le\hat q)\ge1-\alpha$.
(b) Prove the upper bound $\mathbb P(S_{n+1}\le\hat q)\le 1-\alpha+\frac1{n+1}$.
(c) Show by example that dropping distinctness can only help coverage, never hurt it, and explain what "randomized tie-breaking" is for.

## Part B — Computational labs

Each lab asks for numbers. State your seed, your sample size, and your budget convention (evaluations, gradients, or wall-clock) with every number you report; a variance-reduction factor quoted without a budget convention is meaningless.

A note on tooling before you start. The reference runs behind the solutions were written in NumPy with hand-derived gradients, because that is what the environment had; B5 and B6 in particular ask for a VAE and a DDPM "from scratch". If you have an automatic-differentiation library, **use it** — PyTorch or JAX will cut those two labs from a long weekend to an afternoon, and every pedagogical payload survives intact, since what the labs actually test is the ELBO decomposition, the forward-marginal check, the $\beta$-sweep and the sampling diagnostics, not your ability to differentiate a $\tanh$ by hand. Do keep one habit from the hard version: run a finite-difference check on any gradient you write yourself, and report the maximum relative error. If you use autodiff, substitute a different sanity check — verify that the analytic Gaussian KL matches a Monte Carlo estimate of it, which catches the same class of bug. Reference values from the chapter's own runs are given in the solutions so you can check yourself.

**B1 (★).** Estimate $\theta=\mathbb E[e^U]$, $U\sim\mathrm{Unif}(0,1)$, by plain Monte Carlo, antithetic sampling, one control variate ($h=u$), two control variates ($h=(u,u^2)$), and stratification with 40 strata. Hold the number of integrand evaluations fixed. Report the empirical variance of each estimator over at least 2000 independent replications and the variance-reduction factor relative to plain MC. Compare the control-variate factors to the theoretical $1/(1-\rho^2)$. Then break antithetic sampling: find an $f$ for which it is worse than plain MC and report by how much.

**B2 (★★).** Implement random-walk Metropolis on the banana target $\log\pi(x)=-x_1^2/200-\tfrac12\big(x_2-0.03(x_1^2-100)\big)^2$ with an isotropic Gaussian proposal. Sweep the step size over at least $\sigma\in\{0.2,1,2.4,4,6,8,10,20\}$. For each, report the acceptance rate and the effective sample size *per 1000 density evaluations* for both coordinates. Plot ESS per evaluation against acceptance rate. Is the optimum near $0.234$? Explain what you see. Verify your ESS code against an AR(1) chain with known $\tau=(1+\phi)/(1-\phi)$.

**B3 (★★).** Add HMC to the same target: leapfrog with $(\epsilon,L)$ swept over at least $\{(0.2,20),(0.5,10),(0.6,8),(0.8,5),(1.0,3)\}$. Report acceptance and ESS per 1000 *gradient* evaluations. Compare against your best RWMH result from B2 on a per-evaluation basis and say honestly which comparison convention you used and why it matters. Then repeat both on isotropic $N(0,I_d)$ for $d\in\{2,10,50,200\}$ with $\sigma=2.4/\sqrt d$ and $\epsilon=0.9\,d^{-1/4}$, and report how the two efficiencies scale.

**B4 (★★).** Implement CAVI for the Normal–Gamma model $x_i\mid\mu,\tau\sim N(\mu,\tau^{-1})$, $\mu\mid\tau\sim N(\mu_0,(\lambda_0\tau)^{-1})$, $\tau\sim\mathrm{Gamma}(a_0,b_0)$ with a mean-field family $q(\mu)q(\tau)$. Simulate $n=20$ points from $N(2,0.5^2)$. Report the ELBO at every iteration and verify it is non-decreasing to machine precision. Compute the exact log marginal likelihood analytically, report the converged gap in nats, and compare $\operatorname{Var}_q(\mu)$ to the exact posterior variance. Then deliberately break monotonicity by using a stale expectation in one update and show what the ELBO trace looks like.

**B5 (★★★).** Train a small VAE with a 1-D latent on 2-D data sampled from the unit circle plus small noise. Implement it from scratch (finite-difference check your gradients and report the maximum relative error). Report the ELBO decomposed into reconstruction and KL terms. Plot the decoder curve $\{g_\theta(z):z\in[-3,3]\}$ over the data and identify the seam. Then sweep the KL weight $\beta\in\{0,1,4,20,100\}$ and report reconstruction, KL, and ELBO for each; show that at large $\beta$ the KL goes to zero and the reconstruction error matches that of a constant decoder outputting the data mean. Explain which term of Proposition 13.37 you just drove to zero.

**B6 (★★★).** Implement DDPM from scratch on the same 2-D circle data: a linear $\beta$ schedule with $T=200$, a small MLP taking $(x_t,t)$ with sinusoidal time features, and the $\varepsilon$-prediction loss. First *verify the forward process*: for several $t$, compare the empirical mean and standard deviation of iterated single-step noising against the closed form $N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$. Report both. Then train and sample 5000 points; report the radius mean and standard deviation, and a $\chi^2$ test of uniformity of the angle in 8 bins. Comment honestly on whether the angular distribution passes and what that tells you about mode coverage versus support coverage.

**B7 (★★).** Build a synthetic $K$-class problem whose true conditional probabilities you know, then create an overconfident "trained" model by scaling the true logits by $2.5$. On held-out data, report accuracy, mean confidence, ECE with 15 equal-width bins, NLL, and Brier score. Print a reliability table. Fit a single temperature on a calibration split by minimizing NLL; report the fitted $T$, and all the metrics again after scaling. Confirm accuracy is unchanged. Finally, take a *perfectly calibrated* model and report its measured ECE at $n\in\{100,500,2000,10000\}$; use this to state how large an ECE difference has to be before you would believe it.

**B8 (★★).** Implement split-conformal regression on $x\sim\mathrm{Unif}(-3,3)$, $y=\sin x+\sigma(x)\xi$ with $\sigma(x)$ increasing on $x>0$. Using the absolute-residual score, report empirical marginal coverage over at least 200 repetitions for $\alpha\in\{0.20,0.10,0.05\}$ and $n_{\mathrm{cal}}\in\{100,1000\}$, and check every number against the theoretical band $[1-\alpha,\,1-\alpha+1/(n+1)]$. Then report coverage *conditional* on $x$ in six bins and comment. Fix it with a normalized score $|y-\hat\mu(x)|/\hat\sigma(x)$ and report the conditional coverages again. Finally, calibrate on $\mathrm{Unif}(-3,3)$ but test on $\mathrm{Unif}(1,3)$ and report the coverage you actually get.

## Part C — Paper-style derivations

These three are the ones to write out properly, in full sentences, as if for a methods section. Each is a complete short paper.

**C1 (★★).** *Derive the Metropolis–Hastings algorithm and prove it correct.* Begin from the requirement of detailed balance for a kernel of the form $K(x,\mathrm dy)=q(x,y)\alpha(x,y)\,\mathrm dy+r(x)\delta_x(\mathrm dy)$. Show that reversibility is equivalent to symmetry of $\pi(x)q(x,y)\alpha(x,y)$, exhibit the Metropolis solution, prove it is the largest admissible $\alpha$ pointwise, then prove that $\pi$ is invariant — treating the atom at $x$ carefully rather than waving at it. State exactly which measure-theoretic facts you used and where. Conclude with a paragraph on what the theorem does *not* give you.

**C2 (★★★).** *Derive the DDPM training objective from the variational bound.* Start from $-\log p_\theta(x_0)\le\mathbb E_q[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}]$ and reorganize it into $L_T+\sum_{t\ge2}L_{t-1}+L_0$. Compute $q(x_{t-1}\mid x_t,x_0)$ in closed form. Show that with a Gaussian reverse parameterization each $L_{t-1}$ is a weighted squared error, and that reparameterizing the mean in terms of $\varepsilon$ turns it into $\|\varepsilon-\varepsilon_\theta(x_t,t)\|^2$ with an explicit $t$-dependent weight. Then produce an explicit table separating what is *forced* (by the algebra) from what is *chosen* (by DDPM), and for each choice say what is lost. Finish by explaining why the objective is well defined when the data distribution has no density.

**C3 (★★).** *Prove split-conformal coverage and interrogate it.* State the exchangeability assumption precisely, prove the two-sided coverage bound, and then write the interrogation: (i) exactly which step uses exchangeability and which steps use nothing at all about the model $\hat\mu$; (ii) what the guarantee is marginal over, with a worked demonstration that conditional coverage can be arbitrarily bad; (iii) what happens under covariate shift, with the weighted-conformal fix stated and its own assumption named; (iv) a deployment recommendation of at most a paragraph. Support (ii) and (iii) with numbers from your own B8 run.

---

# Appendix A — Self-check answers

**Self-check 13.1.**
1. $\mathrm{se}=\sqrt{0.812\cdot0.188/2000}=\sqrt{7.633\times10^{-5}}=0.0087$, so a $95\%$ interval of roughly $0.812\pm0.017$. It is legitimate because accuracy *is* an integral: it is $\mathbb E[\mathbf 1\{\text{correct}\}]$ under the test distribution, and your test set is an i.i.d. sample from it. You did not choose to do Monte Carlo; you did it by evaluating on a finite sample. The variance is $p(1-p)$, automatically finite, so Proposition 13.3 applies. (What it does *not* cover is the extra variability from having trained on one particular training set.)
2. $s_n=0$, so the CLT interval is the single point $[0,0]$ — the normal approximation is worthless because the effective count is zero, not because the code is wrong; the CLT needs roughly $np\gtrsim10$ successes. Report an exact one-sided bound instead: with no observed events, $p\le 3/n=3\times10^{-6}$ at $95\%$ confidence (the "rule of three", i.e. Clopper–Pearson).
3. **False.** Consistency and unbiasedness say nothing about the second moment. Exercise A2's $f(u)=u^{-3/4}$ is unbiased and consistent with measured coverage $0.648$.
4. Width shrinks by $\sqrt2\approx1.414$. One extra decimal digit means a tenfold narrower interval, hence $100\times$ the samples.

**Self-check 13.2.**
1. $\hat k=1.2>1$ says the fitted generalized-Pareto tail has infinite *mean*, so certainly $\operatorname{Var}(w)=\infty$; the ESS number and the CLT error bars are both meaningless, and even the point estimate converges slowly. Do not "collect more samples": change the proposal (widen it, or use a defensive mixture $0.9q+0.1\,\text{heavy}$), or abandon importance sampling for MCMC.
2. Because $\mathbb E_{q_\theta}[\nabla_\theta\log q_\theta(z)]=0$ — the score has mean zero, which is Chapter 2's differentiation-under-the-integral result. Subtracting $b\,\nabla_\theta\log q_\theta(z)$ therefore subtracts a mean-zero quantity: it is a control variate with *known* mean, unbiased for every $b$, and only the variance changes.
3. With one point per stratum, the variance is the sum of within-stratum variances, so it is exactly zero iff $f$ is constant on each stratum: $f=\mathbf 1\{u>1/2\}$ with strata $(0,\tfrac12),(\tfrac12,1)$.
4. Absolute continuity $P\ll Q$ fails: the target puts mass where the proposal puts none. The estimator is **biased** — it simply never sees that action, so it converges to an integral over a smaller set. This is the one failure mode that more data cannot repair.

**Self-check 13.3.**
1. It is used to guarantee $p\le Mq$ everywhere, i.e. that the acceptance ratio $p/(Mq)$ is a genuine probability. If $p>Mq$ on a positive-measure set $B$, the acceptance probability is clipped to $1$ on $B$ and the algorithm returns draws from the normalized version of $\min\{p,Mq\}$, which undersamples $B$ — a bias, silently.
2. $\mathbb E[N]=M=500$ per accepted draw, so about $5\times10^5$ trials and hence $5\times10^5$ evaluations of $p$ (and of $q$) for 1000 samples.
3. Because the number of trials is random and data-dependent; if the returned value and the trial count were dependent, then conditioning on "we stopped now" would tilt the output law. Independence (Theorem 13.12) is exactly the statement that the waiting time carries no information about the draw, so the loop is correct and you may ignore the random run time when reasoning about the output.
4. Importance sampling replaces a *uniform bound* $\sup p/q<\infty$ with a *moment* condition $\mathbb E_Q[f^2w^2]<\infty$. That is weaker, and it is why importance sampling still runs when rejection sampling cannot start — but the price is that you get weighted, correlated-in-effect samples whose error bars can be infinite, and you must diagnose that (ESS, $\hat k$).

**Self-check 13.4.**
1. If $q(x,y)=q(y,x)$ the ratio $q(y,x)/q(x,y)=1$, leaving $\alpha=\min\{1,\pi(y)/\pi(x)\}$. Since $\pi$ appears only through the ratio $\pi(y)/\pi(x)$, replacing $\pi$ by $c\pi$ changes nothing: the normalizing constant cancels. This is the single most consequential line in the chapter.
2. Because with probability $r(x)>0$ the chain does not move, so $K(x,\cdot)$ has an atom $r(x)\delta_x$, and no atom is absolutely continuous with respect to Lebesgue measure. Chapter 2's Radon–Nikodym machinery lets you write the kernel honestly as a density part plus an atom — i.e. with respect to the base measure $\mathrm{Leb}+\delta_x$ — which is precisely Chapter 1's Failure 2 reappearing as a working tool.
3. The lag-one autocorrelation of Gibbs on the bivariate Gaussian is $\rho^2$, so $\tau=(1+\rho^2)/(1-\rho^2)\to\infty$ as $\rho\to1$: the sampler crawls along the ridge in axis-aligned steps. Blocked Gibbs updates $(x_1,x_2)$ jointly from the exact joint, giving independent draws and $\tau=1$. The lesson is that the *parameterization*, not the algorithm, was the problem.
4. The acceptance rate distinguishes them immediately: too-small steps give acceptance near $1$ with tiny moves; too-large steps give acceptance near $0$ with long flat stretches in the trace. The trace plots also look different — a slow random walk versus a step function.

**Self-check 13.5.**
1. To make the proposal map an involution, so that the forward and reverse proposal densities are equal and the $q$-ratio cancels. Without the flip, the reverse move is not the one the algorithm would ever propose, and detailed balance fails. It is free because the momentum is resampled next iteration anyway.
2. The acceptance probability $\min\{1,e^{-\Delta H}\}$ measures the *integrator's energy error*, nothing else — not the target's difficulty, not the distance travelled. That is why acceptance is a step-size diagnostic in HMC and a step-size *and* geometry diagnostic in RWMH.
3. As $h\to4$ the factor $1/(1-h/4)$ diverges: the stationary variance blows up and for $h\ge4$ the chain is unstable. For a target with a large Hessian eigenvalue (a narrow direction), stability requires $h<4\sigma_{\min}^2$, so the *narrowest* direction caps the step size while the widest direction then has negligible bias. Ill-conditioned targets therefore force tiny steps: preconditioning is not optional.
4. Any two of: the noise covariance is state-dependent and strongly anisotropic (roughly Fisher-shaped) rather than the isotropic $hI$ that ULA requires; the noise is neither Gaussian nor independent across steps under epoch shuffling; there is no Metropolis correction, so the bias is systematic and unmeasured; and the step size is chosen to minimize loss, not to set a temperature — so the implied "posterior" changes when you change the schedule.

**Self-check 13.6.**
1. Report the *tail* ESS, $90$, next to a credible interval, because the interval is a quantile functional and the relevant autocorrelation is that of the tail indicator, not of the identity function. They differ because $\tau$ depends on the function being averaged: rare events are visited rarely, so their indicators decorrelate slowly.
2. Because splitting compares the first half of each chain with its second half, which surfaces *within*-chain drift — a chain still travelling toward the mode. Comparing whole-chain means can average that trend away and report $\hat R\approx1$ for four chains that are all still moving.
3. A divergence means the leapfrog integrator's energy error exploded, i.e. the sampler *cannot enter* a region (typically a funnel neck) at the current step size. That is a bias: whole parts of the posterior are unexplored. An ordinary rejection is the algorithm working correctly and costs only efficiency.
4. Multimodality with all chains initialized in one mode — for instance label-switching modes in a mixture model, or one basin of a neural-network posterior. Each chain mixes beautifully within its mode, $\hat R\approx1.00$, ESS is large, there are no divergences, and the sample is missing almost all of the posterior mass.

**Self-check 13.7.**
1. It can only increase (weakly): a larger family contains the old optimum. Since $\log p(x)$ is fixed and the gap equals $\log p(x)-\max\mathcal L$, the gap can only *decrease*. It cannot increase.
2. Conjugacy is what makes $q_j^\star\propto\exp\big(\mathbb E_{q_{-j}}[\log p(z_j\mid z_{-j},x)]\big)$ land in a closed-form family with computable expectations. Without it the coordinate update has no analytic solution and you must switch to black-box/stochastic gradient VI with a reparameterized or score-function estimator.
3. That you have a bug. CAVI's ELBO is non-decreasing by construction (each coordinate update is an exact maximization), so a decrease means a stale expectation, a wrong update formula, or an error in the ELBO code — unless the decrease is at the $10^{-14}$ level, which is floating point.
4. (i) Mean-field discards posterior correlation, so it reports a *conditional* standard deviation where a marginal one belongs (Proposition 13.34); (ii) the direction $D_{\mathrm{KL}}(q\|p)$ is zero-forcing, so $q$ is penalized for putting mass where $p$ has little and hugs a single mode. Chapter 11's asymmetry result predicts the *sign*: too small, essentially always.

**Self-check 13.8.**
1. The variational gap $D_{\mathrm{KL}}(q(z\mid x)\|p_\theta(z\mid x))$ and the amortization gap between the encoder network's output and the best member of the variational family for that particular $x$. The second does not exist in non-amortized VI, where you optimize free variational parameters separately for every datapoint.
2. Because the identity writes the left-hand side as a sum of two non-negative terms, $I_q(X;Z)+D_{\mathrm{KL}}(q(z)\|p(z))$. A sum of non-negative terms is zero only if each is zero, so $I_q(X;Z)=0$: the latent carries no information about the data.
3. The encoder's gradient signal-to-noise ratio decays like $O(K^{-1/2})$ (Rainforth et al. 2018): the tighter bound depends less and less on $q$, so the learning signal for the encoder is swamped by noise, and a bad encoder produces degenerate weights (§13.2.2's problem, now inside the training loop).
4. $g_\theta(z)$ is the conditional *mean* of $p_\theta(\cdot\mid z)$, so displaying it hides the observation noise $\sigma$ that the likelihood claims is there: the model's actual samples are blurrier than the picture. Honest display: show real draws from $p_\theta(x\mid z)$ alongside the means, or plot the mean curve with a $\pm\sigma$ band and state $\sigma$.

**Self-check 13.9.**
1. Hyvärinen's identity is proved by integration by parts, which needs $p>0$ (so $\log p$ and $\nabla\log p$ exist everywhere), differentiability, and enough decay that the boundary term vanishes. Proposition 13.41 drops the existence of a data density altogether: convolving with $N(0,\sigma^2I)$ manufactures a smooth positive density from any distribution, however singular, and the regression target becomes the noise, which we know.
2. $L_T=D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))$ contains no learnable parameters once the $\beta$ schedule is fixed. Its being small requires $\bar\alpha_T\approx0$ — the forward process must actually destroy the signal (in our run $\bar\alpha_T=0.002167$).
3. Because $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$, and that factor diverges as $t\to0$. Predicting $\varepsilon$ keeps the regression target $O(1)$ at every noise level, so a single network with a single output scale can serve all $t$; predicting the score directly asks one network to produce outputs spanning orders of magnitude. The choice also silently changes the *implicit weighting* of the loss across $t$.
4. Because Jensen–Shannon between mutually singular measures equals $\log 2$ *regardless of how far apart the supports are*: the divergence is locally constant in the generator's parameters, so its gradient contains no information about which way to move. A DDPM's loss is not a divergence between singular measures at all — it is a squared-error regression at noise levels $\sigma_t>0$, where the noised data has a full-dimensional smooth density, and the loss varies smoothly with the parameters even when the clean data lives on a measure-zero set.

**Self-check 13.10.**
1. Because dividing all logits by $T>0$ is a strictly monotone transformation of each logit vector that preserves the argmax, so every prediction — and therefore the accuracy, and the whole confusion matrix — is unchanged. What changes is the *confidence values*, hence ECE, NLL, and Brier.
2. That $\mathrm{ECE}=0.015$ at $n=1000$ is inside the noise floor: in our runs a *perfectly calibrated* model measured $0.0491$ at $n=500$ and $0.0253$ at $n=2000$, so at $n=1000$ you should expect roughly $0.03$ from a flawless model. Their number is not evidence of near-perfect calibration; it is not even distinguishable from it. Ask for bootstrap error bars, sensitivity to the bin count, and a proper score such as NLL or Brier alongside.
3. Exchangeability is used exactly once, and only to conclude that the rank of $S_{n+1}$ among $S_1,\dots,S_{n+1}$ is uniform on $\{1,\dots,n+1\}$. Nothing else in the proof needs it. For time series exchangeability genuinely fails; replace it with a scheme that does not need it — online/adaptive conformal inference, which updates the quantile from realized coverage, or blocked and weighted variants.
4. Medical triage where risk is heteroscedastic: an interval procedure with $90\%$ marginal coverage might cover $99\%$ of the healthy majority and $60\%$ of the sickest tail — exactly the patients the interval was for. Our B8 run is a small version of this: $100\%$ on $x\in(-3,-2]$ and $62.4\%$ on $x\in(2,3]$. Compute group-conditional coverage for every subgroup you care about, and switch to a normalized score $|y-\hat\mu|/\hat\sigma$ (which restored $\approx0.90$ in every bin) or to Mondrian/group-conditional conformal.

---

# Solutions to Exercises

## Part A

**A1.** *(a)* Linearity gives $\mathbb E[\hat\theta_n]=\theta$; independence gives $\operatorname{Var}(\hat\theta_n)=n^{-2}\sum_i\operatorname{Var}(f(X_i))=\sigma^2/n$. Note what independence was used for: only the vanishing of cross-covariances, which is why the same formula holds for merely uncorrelated draws.

*(b)* Write $f_i=f(X_i)$ and expand $\sum_i(f_i-\hat\theta_n)^2=\sum_i(f_i-\theta)^2-n(\hat\theta_n-\theta)^2$. Taking expectations, the first term is $n\sigma^2$ and the second is $n\cdot\sigma^2/n=\sigma^2$, so $\mathbb E[\sum_i(f_i-\hat\theta_n)^2]=(n-1)\sigma^2$. The $n-1$ is *exactly* the one degree of freedom spent estimating the centre: subtracting $\hat\theta_n$ instead of $\theta$ removes on average one $\sigma^2$ worth of spread.

*(c)* The divisor-$n$ version underestimates $\sigma^2$ by the factor $(n-1)/n$, so the bars are too narrow by $\sqrt{(n-1)/n}$: at $n=4$ that is $0.866$ (a $13\%$ understatement), at $n=4000$ it is $0.99987$ ($0.01\%$). The thinking to take away: this is a real bias but it is $O(1/n)$ in the variance, and at any sample size where you are reporting CLT intervals at all it is dwarfed by the CLT approximation error itself. Spend your worry on whether $\sigma^2<\infty$, not on the divisor.

**A2.** *(a)* $\mathbb E[U^{-\gamma}]=\int_0^1u^{-\gamma}\,\mathrm du=1/(1-\gamma)$ for $\gamma<1$, infinite otherwise. The second moment is $\mathbb E[U^{-2\gamma}]=1/(1-2\gamma)$ for $\gamma<1/2$. So the mean exists iff $\gamma<1$ and the variance iff $\gamma<1/2$.

*(b)* At $\gamma=3/4$: $\mathbb E=1/(1-3/4)=4$ and $\mathbb E[f^2]=\mathbb E[U^{-3/2}]=\infty$ since $3/2>1$. The SLLN needs only $\mathbb E|f|<\infty$, so $\hat\theta_n\to4$ almost surely: the estimator is fine. Proposition 13.3 needs $\sigma^2<\infty$, which fails, so the $\pm1.96s_n/\sqrt n$ interval has no justification — and $s_n$ itself does not converge, it drifts upward as ever-larger draws arrive.

*(c)* This part is a trap, and it is worth walking into it before reading the answer. The tempting argument goes: the summands lie in the domain of attraction of a stable law of index $\alpha=1/\gamma=4/3\in(1,2)$, so the correct normalization is $n^{1-1/\alpha}=n^{1/4}$ and the true error decays like $n^{-1/4}$; the reported half-width decays like $n^{-1/2}$; therefore the interval shrinks faster than the error and coverage collapses to zero.

The first two clauses are right and the conclusion is wrong, because the argument forgets that $s_n$ diverges too. Since $f^2=U^{-3/2}$ has tail index $2/3<1$, $\sum_if(U_i)^2$ grows like $n^{3/2}$, so $s_n\sim n^{1/4}$ and the half-width $1.96s_n/\sqrt n$ decays like $n^{-1/4}$ — *the same rate as the error*. The studentized quantity $\sqrt n(\hat\theta_n-\theta)/s_n$ therefore has a non-degenerate limit law (the self-normalized stable limit theory of Logan, Mallows, Rice & Shepp, 1973), and coverage converges to a **constant strictly below the nominal level**, not to zero. Measured, seed $2024$, $400$–$4000$ replications per row:

| $n$ | coverage | mean width |
|---|---|---|
| $10^3$ | $0.628$ | $2.680$ |
| $10^4$ | $0.644$ | $1.685$ |
| $10^5$ | $0.647$ | $0.797$ |
| $10^6$ | $0.667$ | $0.842$ |

Flat at roughly $0.65$ across three decades of $n$. Two lessons, and the second is the important one. First, when checking an asymptotic claim about an interval, you must track the behaviour of the *width estimate* as well as the estimator — self-normalization changes the answer. Second, this failure mode is nastier than a visible collapse: the interval keeps shrinking, the point estimate keeps converging, every plot looks healthy, and the procedure under-covers by thirty percentage points forever. A diagnostic that *does* fire is the width itself, which decays like $n^{-1/4}$ instead of $n^{-1/2}$ — if you halve your error bars by quadrupling $n$ and they stubbornly refuse to halve, suspect an infinite second moment.

**A3.** *(a)* $\hat\theta^{\mathrm{IS}}_n=n^{-1}\sum_i f(Y_i)w(Y_i)$ with $Y_i\sim Q$ is an average of i.i.d. terms with mean $\mathbb E_Q[fw]=\int fp=\theta$, so the variance is $n^{-1}(\mathbb E_Q[f^2w^2]-\theta^2)$.

*(b)* Only $\mathbb E_Q[f^2w^2]=\int f^2p^2/q$ depends on $q$. By Cauchy–Schwarz,
$$\Big(\int|f|p\Big)^2=\Big(\int\frac{|f|p}{\sqrt q}\sqrt q\Big)^2\le\int\frac{f^2p^2}{q}\int q=\int\frac{f^2p^2}{q},$$
with equality iff $|f|p/\sqrt q\propto\sqrt q$, i.e. $q\propto|f|p$. So $q^\star\propto|f|p$ and the minimum is $(\mathbb E_P|f|)^2$.

*(c)* If $f\ge0$ then $\mathbb E_P|f|=\theta$ and the minimal variance is $\theta^2-\theta^2=0$. It is useless because sampling from $q^\star\propto fp$ requires its normalizing constant, which *is* $\theta$: the zero-variance proposal presupposes the answer. What it is good for is guidance — it says put proposal mass where $|f|p$ is large, which is the design principle behind every practical importance sampler.

*(d)* For $f=\mathbf 1_A$, $q^\star\propto p\mathbf 1_A$, i.e. $P$ conditioned on $A$. In Worked Example 13.5, $A=\{z>4\}$ and $P|_A$ is a truncated normal we could sample but chose not to; $N(4,1)$ is a cheap surrogate that puts about half its mass in $A$ and has the right tail decay. That approximation of the ideal proposal took the relative standard error from $0.71$ to $0.0067$.

**A4.** *(a)* $\mathbb E[h(X)-\mu_h]=0$, so $\mathbb E[\hat\theta_c]=\theta$ for every fixed $c$ — unbiasedness is free, which is the whole appeal.

*(b)* $\operatorname{Var}(f-ch)=\sigma_f^2-2c\operatorname{Cov}(f,h)+c^2\sigma_h^2$ is a quadratic in $c$ minimized at $c^\star=\operatorname{Cov}(f,h)/\sigma_h^2$, where its value is $\sigma_f^2(1-\rho^2)$ with $\rho=\operatorname{Corr}(f,h)$.

*(c)* $\hat c$ is a ratio of sample moments correlated with the residuals, so $\mathbb E[\hat\theta_{\hat c}]=\theta+O(1/n)$ while the standard error is $O(n^{-1/2})$: the bias is asymptotically negligible relative to the noise it buys you. To remove it exactly, split the sample: estimate $\hat c$ on one half, apply it to the other (or use a leave-one-out $\hat c_{-i}$ for observation $i$). You lose a little efficiency and gain exact unbiasedness — worth it only when the bias is being audited.

**A5.** *(a)* Fix a budget of $2n$ evaluations. Plain MC has variance $\sigma^2/(2n)$. The antithetic estimator averages $n$ i.i.d. pairs $Z_i=\tfrac12(f(U_i)+f(1-U_i))$ with $\operatorname{Var}(Z)=\tfrac14(2\sigma^2+2C)=\tfrac12(\sigma^2+C)$ where $C=\operatorname{Cov}(f(U),f(1-U))$, so the estimator's variance is $(\sigma^2+C)/(2n)$. This beats $\sigma^2/(2n)$ iff $C<0$. The budget convention is essential: compared per *pair* rather than per evaluation, antithetic sampling would look better than it is.

*(b)* Let $g(u)=f(1-u)$, decreasing when $f$ is increasing. For independent $U,V\sim\mathrm{Unif}(0,1)$,
$$\mathbb E\big[(f(U)-f(V))(g(U)-g(V))\big]=2\big(\mathbb E[fg]-\mathbb E[f]\mathbb E[g]\big)=2\operatorname{Cov}(f(U),g(U)).$$
But pointwise $(f(u)-f(v))(g(u)-g(v))\le0$, because $f$ and $g$ move in opposite directions. Hence $C\le0$. (This is the one-dimensional case of Hoeffding's covariance identity; the same trick proves the FKG-type inequalities.)

*(c)* $f(u)=(u-\tfrac12)^2$ satisfies $f(1-u)=f(u)$ identically, so $C=\operatorname{Var}(f)=\sigma^2$ and $\rho=1$. The antithetic variance is then $(\sigma^2+\sigma^2)/(2n)=\sigma^2/n$, exactly twice the plain variance at the same $2n$ evaluations. The measurement in §13.2.4 found $\rho=1.0000$ and a factor of two loss, as predicted. The moral: antithetic sampling exploits *monotonicity*, and a symmetric integrand is the exact opposite of monotone.

**A6.** *(a)* $\mathbb E[\hat\theta^{\mathrm{str}}]=K^{-1}\sum_k\mathbb E[f(U_k)]=K^{-1}\sum_k K\int_{I_k}f=\int_0^1f=\theta$.

*(b)* Independence across strata gives $\operatorname{Var}=K^{-2}\sum_k\operatorname{Var}(f(U_k))$. Writing $S$ for the (uniform) stratum label of a plain draw $U$, the law of total variance (Ch. 6) says
$$\operatorname{Var}(f(U))=\underbrace{\mathbb E[\operatorname{Var}(f(U)\mid S)]}_{=K^{-1}\sum_k\operatorname{Var}(f(U_k))}+\underbrace{\operatorname{Var}(\mathbb E[f(U)\mid S])}_{\ge0}.$$
Hence $K^{-2}\sum_k\operatorname{Var}(f(U_k))=K^{-1}\mathbb E[\operatorname{Var}(f\mid S)]\le K^{-1}\operatorname{Var}(f(U))$, which is the plain variance at the same cost $K$. Stratification is therefore *never* worse — unlike antithetic sampling.

*(c)* It removes exactly the between-stratum variance $\operatorname{Var}(\mathbb E[f\mid S])$, i.e. the part of the variability explained by knowing which stratum you are in. Two strata give zero for any $f$ constant on each half, e.g. $f=\mathbf 1\{u>1/2\}$.

**A7.** *(a)* The acceptance probability of a single trial is $\int q(y)\frac{p(y)}{Mq(y)}\mathrm dy=\frac1M\int p=\frac1M$. Trials are i.i.d., so $N$ is geometric with success probability $1/M$ and $\mathbb E[N]=M$.

*(b)* For any measurable $A$,
$$\mathbb P(Y\in A,\,N=n)=\Big(1-\tfrac1M\Big)^{n-1}\int_Aq(y)\frac{p(y)}{Mq(y)}\mathrm dy=\Big(1-\tfrac1M\Big)^{n-1}\tfrac1M\,P(A),$$
which factorizes as $\mathbb P(N=n)\,P(A)$. So $Y\sim P$ exactly and $Y\perp N$. This is what licenses the `while` loop: the code returns the draw at a *random, data-dependent* stopping time, and only independence guarantees that conditioning on "we stopped here" does not tilt the output law.

*(c)* With a cap, the output is a two-component mixture. With probability $1-\delta$, where $\delta=(1-1/M)^{100}$, the loop accepts and returns an exact $P$-draw; with probability $\delta$ it returns the $100$th proposal, and it is worth being careful about that component's law. That proposal was *rejected*, so conditionally its density is proportional to $q(y)\big(1-\frac{p(y)}{Mq(y)}\big)=q(y)-p(y)/M$, normalized by $1-1/M$ — it is $Q$ tilted *away* from the region where $p/q$ is large, not $Q$ itself. So the output law is
$$(1-\delta)\,P+\delta\cdot\frac{Q-P/M}{1-1/M},$$
biased for any $\delta>0$. Since $p/q$ is largest in the tails when $P$ is heavier-tailed than $Q$, the contaminating component is doubly deficient there — once because $Q$ is already too light, and again because conditioning on rejection removes exactly the draws that would have landed in the tail. The output is systematically **too light in the tails**, and the bias hits precisely the region you were worried about. (This is the standard bug in "safe" rejection samplers with a retry limit; if you must cap, raise an error rather than returning anything.)

**A8.** *(a)* Summing detailed balance $\pi(x)K(x,y)=\pi(y)K(y,x)$ over $x$:
$$(\pi K)(y)=\sum_x\pi(x)K(x,y)=\sum_x\pi(y)K(y,x)=\pi(y)\sum_xK(y,x)=\pi(y).$$
The only facts used are the symmetry and that $K(y,\cdot)$ is a probability measure. This is the cheapest theorem in the chapter and the reason MCMC exists.

*(b)* Take the deterministic cycle $K(i,i+1\bmod 3)=1$. Then $\pi K=\pi$ for uniform $\pi$, but $\pi(1)K(1,2)=1/3\ne0=\pi(2)K(2,1)$, so detailed balance fails. It is irreducible but has period $3$, hence not aperiodic. Fixing that: $K(i,i+1)=0.9$, $K(i,i)=0.1$ is irreducible, aperiodic, uniform-invariant, and still not reversible.

*(c)* Non-reversible chains can move *ballistically* instead of diffusively: a reversible chain retraces its steps and covers distance $\sqrt t$, while a chain with persistent direction can cover distance $\propto t$, which is the idea behind lifted and piecewise-deterministic samplers. Proposition 13.22's ordered (systematic) scan Gibbs is a familiar example — it is $\pi$-invariant but not reversible, whereas random-scan Gibbs is reversible; the ordered scan is the one everybody actually runs, and it typically mixes at least as fast.

**A9.** *(a)* With $q(x,y)=q(y)$,
$$\alpha=\min\Big\{1,\frac{\pi(y)q(x)}{\pi(x)q(y)}\Big\}=\min\Big\{1,\frac{w(y)}{w(x)}\Big\}.$$
*(b)* If $\{\pi>0\}\subseteq\{q>0\}$ then from any $x$ with $\pi(x)>0$ the proposal reaches any $A$ with $\pi(A)>0$ with probability $\int_Aq>0$, and the acceptance probability is strictly positive there, so the chain is irreducible. This is *the same* absolute-continuity hypothesis as Proposition 13.4's $P\ll Q$: the proposal must be able to reach everywhere the target lives.

*(c)* MH returns unweighted draws and never needs $\mathbb E[w]$, so there is no weight-degeneracy in the estimator and no normalizing constants; the price is dependence, and the loss of the clean i.i.d. error theory. When $w$ is unbounded the two fail *differently*: importance sampling's variance can be infinite (silently, with cheerful error bars), while independence MH gets stuck — landing once on a state with enormous $w(x)$ makes every subsequent proposal be rejected for a geometrically long time with huge mean, which at least *looks* broken in a trace plot. Failing visibly is a real advantage.

**A10.** *(a)* The three sub-steps are $p\mapsto p-\tfrac\epsilon2\nabla U(x)$ (with $x$ fixed), $x\mapsto x+\epsilon p$ (with $p$ fixed), and another half-kick. Each is a shear: the Jacobian is block-triangular with identity blocks on the diagonal, hence determinant $1$. The composition has determinant $1$, so $|\det D\Phi_\epsilon|=1$ exactly — not to $O(\epsilon^2)$, exactly.

*(b)* Let $\Psi=F\circ\Phi_\epsilon^L$. One checks directly that $\Phi_\epsilon\circ F\circ\Phi_\epsilon\circ F=\mathrm{id}$ for a single step (flip, step, flip, step retraces the leapfrog exactly because the kick–drift–kick pattern is palindromic), and induction on $L$ gives $\Psi\circ\Psi=\mathrm{id}$.

*(c)* The MH ratio for a deterministic proposal $\Psi$ on the augmented target $\tilde\pi(x,p)\propto e^{-H(x,p)}$ is
$$\frac{\tilde\pi(\Psi(x,p))}{\tilde\pi(x,p)}\big|\det D\Psi(x,p)\big|.$$
By (b) the proposal is its own inverse, so the "reverse proposal" is the move the algorithm actually makes — this is what makes a deterministic proposal admissible at all — and by (a) the Jacobian factor is $1$. What survives is $\exp\big(H(x,p)-H(x',p')\big)$. Without volume preservation the Jacobian determinant would remain in the ratio, exactly as in a normalizing flow (Ch. 3): computable in principle, $O(d^3)$ in general, and it would have to be recomputed along the whole trajectory. Symplectic integrators are not chosen for accuracy; they are chosen because the correction term is *free*.

**A11.** *(a)* The recursion is $x_{k+1}=ax_k+\sqrt h\,\xi_k$ with $a=1-h/(2\sigma^2)$. A stationary variance $v$ must satisfy $v=a^2v+h$, so $v=h/(1-a^2)$. Since $1-a^2=(1-a)(1+a)=\frac{h}{2\sigma^2}\big(2-\frac{h}{2\sigma^2}\big)=\frac h{\sigma^2}-\frac{h^2}{4\sigma^4}$,
$$v(h)=\frac{\sigma^2}{1-h/(4\sigma^2)},\qquad\text{stable iff }|a|<1\iff 0<h<4\sigma^2.$$
*(b)* At $\sigma^2=1$, $v(h)=1/(1-h/4)$, matching Proposition 13.27 and the measured $1.0242/1.1445/1.3314$ at $h=0.1/0.5/1.0$.

*(c)* For $\Sigma=\mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)$ the coordinates evolve independently, so stability requires $h<4\sigma_d^2$ — the *narrowest* direction caps the step. Taking $h=c\cdot4\sigma_d^2$ with $c<1$, the relative bias in the largest variance is
$$\frac{1}{1-h/(4\sigma_1^2)}-1\approx\frac{h}{4\sigma_1^2}=c\,\frac{\sigma_d^2}{\sigma_1^2}=\frac{c}{\kappa},$$
with $\kappa$ the condition number. So the bias in the wide direction is *negligible* — and that is the bad news, not the good news: it is negligible because the sampler is barely moving in that direction. The wide direction has relaxation time $\sim\sigma_1^2/h\sim\kappa$ steps. ULA's cost is linear in the condition number, which is why preconditioning (or HMC with a mass matrix) is the real fix.

**A12.** *(a)* Substitute $p(x,z)=p(z\mid x)p(x)$:
$$\mathcal L(q)=\mathbb E_q\big[\log p(z\mid x)+\log p(x)-\log q(z)\big]=\log p(x)-D_{\mathrm{KL}}(q\|p(\cdot\mid x)).$$
Since $\log p(x)$ does not depend on $q$, maximizing $\mathcal L$ is exactly minimizing that KL. Note that no inequality was used: (b) of Theorem 13.31 is an *identity*, and Jensen's version is a corollary via $D_{\mathrm{KL}}\ge0$.

*(b)* If $\mathcal Q\subseteq\mathcal Q'$ then $\max_{\mathcal Q'}\mathcal L\ge\max_{\mathcal Q}\mathcal L$. The gap is $\log p(x)-\max\mathcal L$ with $\log p(x)$ fixed, so the gap weakly **decreases**; it cannot increase. (For amortized VI this reasoning applies to the variational gap only — enlarging the encoder's output family can still make optimization worse, which is an amortization effect, not a violation.)

*(c)* Let model A have $\log p_A(x)=-100$ with a tight family (gap $0.1$, ELBO $-100.1$) and model B have $\log p_B(x)=-99$ with a poor family (gap $5$, ELBO $-104$). Then A has the larger ELBO and the smaller marginal likelihood. Comparing models by ELBO confounds *fit* with *tightness of the approximating family*, and there is no general fix — which is why ELBO-based model comparison needs either a shared variational family or an independent estimate (IWAE with large $K$, or annealed importance sampling).

**A13.** *(a)* With $q=N(m,S)$, $S=\mathrm{diag}(s_1^2,\dots,s_d^2)$, and $p=N(0,\Sigma)$, $\Lambda=\Sigma^{-1}$:
$$2D_{\mathrm{KL}}(q\|p)=\operatorname{tr}(\Lambda S)+m^\top\Lambda m-d+\log\frac{\det\Sigma}{\det S}.$$
The $m$-terms are $m^\top\Lambda m\ge0$, minimized at $m=0$. For $s_i^2$, the relevant terms are $\Lambda_{ii}s_i^2-\log s_i^2$; differentiating, $\Lambda_{ii}-1/s_i^2=0$, so $s_i^2=1/\Lambda_{ii}$. (The off-diagonal entries of $\Lambda$ enter only through $\operatorname{tr}(\Lambda S)$, and $S$ diagonal kills them — which is precisely the information mean-field throws away.)

*(b)* $1/\Lambda_{ii}=\operatorname{Var}(z_i\mid z_{-i})$, the **conditional** variance given all other coordinates. And $\Sigma_{ii}\ge1/\Lambda_{ii}$ always (Schur complement / "conditioning reduces variance", Ch. 6). So mean-field VI reports a conditional standard deviation where the user reads a marginal one: overconfident, always in the same direction, with the discrepancy growing with posterior correlation.

*(c)* Unit variances and correlation $\rho$ give $\Lambda=\frac1{1-\rho^2}\begin{pmatrix}1&-\rho\\-\rho&1\end{pmatrix}$, so $s^2=1/\Lambda_{11}=1-\rho^2$. Substituting $S=(1-\rho^2)I$: $\operatorname{tr}(\Lambda S)=2$, $\det S=(1-\rho^2)^2$, $\det\Sigma=1-\rho^2$, so $2D_{\mathrm{KL}}=2-2+\log\frac{1-\rho^2}{(1-\rho^2)^2}=-\log(1-\rho^2)$. At $\rho=0.99$: $s^2=0.0199$, so $s=0.141$ against a true marginal sd of $1$ — intervals $7.1\times$ too narrow — and $D_{\mathrm{KL}}=1.9585$ nats.

**A14.** *(a)* Under the joint $q(x,z)=p_{\mathcal D}(x)q(z\mid x)$ with marginal $q(z)=\mathbb E_x[q(z\mid x)]$,
$$\mathbb E_x\big[D_{\mathrm{KL}}(q(z\mid x)\|p(z))\big]=\mathbb E_{q(x,z)}\Big[\log\frac{q(z\mid x)}{q(z)}\Big]+\mathbb E_{q(x,z)}\Big[\log\frac{q(z)}{p(z)}\Big]=I_q(X;Z)+D_{\mathrm{KL}}(q(z)\|p(z)).$$
The trick is the single insertion of $q(z)/q(z)$; the first term is by definition the mutual information, the second does not depend on $x$.

*(b)* Both terms are non-negative, and a sum of non-negative terms vanishes only if each vanishes. So a per-datapoint KL of zero forces $I_q(X;Z)=0$ *and* $q(z)=p(z)$: the encoder ignores its input.

*(c)* Because the KL term is a **rate**, not a penalty on wiggliness. Setting $\beta$ large does not "smooth" the latent, it *reduces the channel capacity between data and code*, and at rate zero the decoder can only output a constant — measured in Worked Example 13.20 as reconstruction $47.31$ against a constant-decoder baseline of $47.19$. In Chapter 11's language the $\beta$-sweep traverses a rate–distortion curve: reconstruction is distortion, KL is rate, and $\beta$ is the slope at which you choose to sit. There is no value of $\beta$ that is "just regularization".

**A15.** *(a)* $\mathcal L_K=\mathbb E[\log(\frac1K\sum_kw_k)]\le\log\mathbb E[\frac1K\sum_kw_k]=\log\mathbb E[w]=\log p(x)$, by Jensen and $\mathbb E_{q}[p(x,z)/q(z\mid x)]=p(x)$ — the identity that, per Theorem 13.31(a), requires $p(\cdot\mid x)\ll q(\cdot\mid x)$, target $\ll$ proposal.

*(b)* Let $w_1,\dots,w_{K+1}$ be i.i.d. and let $I$ be uniform on the $K+1$ subsets of size $K$. Then
$$\frac1{K+1}\sum_{k=1}^{K+1}w_k=\mathbb E_I\Big[\frac1K\sum_{k\in I}w_k\Big],$$
because each $w_k$ appears in exactly $K$ of the $K+1$ subsets. Applying Jensen to the concave $\log$ *conditionally on the $w$'s*,
$$\log\Big(\frac1{K+1}\sum_{k}w_k\Big)\ge\mathbb E_I\Big[\log\frac1K\sum_{k\in I}w_k\Big].$$
Take expectations over the $w$'s; by exchangeability every subset contributes the same value $\mathcal L_K$, giving $\mathcal L_{K+1}\ge\mathcal L_K$. The whole content is the observation that an average of $K+1$ things is an average of averages of $K$ things.

*(c)* You need enough integrability to pass to the limit: $\frac1K\sum w_k\to p(x)$ a.s. by the SLLN, and a domination/uniform-integrability argument then gives $\mathcal L_K\to\log p(x)$ (Burda et al., App. A). It does not follow that larger $K$ is better in practice, for two reasons: the cost is linear in $K$, and — more subtly — the *encoder's* gradient signal-to-noise ratio decays like $O(K^{-1/2})$, so a tighter bound comes with a weaker training signal for $q$. A better bound optimized worse can yield a worse model. Tightness of a bound and quality of the resulting learning algorithm are different things, and this is the cleanest example in the chapter.

**A16.** *(a)* By exchangeability, the rank $R$ of $S_{n+1}$ among $S_1,\dots,S_{n+1}$ is uniform on $\{1,\dots,n+1\}$: all $(n+1)!$ orderings are equally likely, so all $n+1$ positions for $S_{n+1}$ are. Now $S_{n+1}\le\hat q$, the $k$-th smallest of the first $n$ with $k=\lceil(n+1)(1-\alpha)\rceil$, holds if and only if $R\le k$. Hence $\mathbb P=k/(n+1)\ge(1-\alpha)$ by definition of the ceiling.

*(b)* $\lceil t\rceil<t+1$ gives $k<(n+1)(1-\alpha)+1$, so $k/(n+1)<1-\alpha+\frac1{n+1}$.

*(c)* With ties, $\{S_{n+1}\le\hat q\}$ contains $\{R\le k\}$ (a tied score still satisfies $\le$), so the coverage can only go up: the *lower* bound survives and only the upper bound is at risk. Randomized tie-breaking — adding an infinitesimal independent noise to every score — restores a.s. distinctness and hence the sharp two-sided statement. You want it when you need coverage $\approx1-\alpha$ rather than $\ge1-\alpha$; discrete scores (e.g. classification with few classes) are where this bites, and it is the reason conformal classification intervals are often noticeably conservative.

## Part B

Reference values below are from the chapter's own runs; your numbers should agree to within the reported Monte Carlo noise, not exactly, and if they do not the *first* thing to check is your budget convention.

**B1.** Reference run: $4000$ replications, $N=1200$ integrand evaluations each, seed $99$, $\theta=e-1=1.718282$, $\operatorname{Var}(e^U)=0.242036$.

| Estimator | Empirical variance | Factor vs plain | Theory |
|---|---|---|---|
| Plain MC | $1.981\times10^{-4}$ | $1$ | $\sigma^2/N=2.017\times10^{-4}$ |
| Antithetic ($600$ pairs) | $6.764\times10^{-6}$ | $29.8$ | $30.9$ |
| Control variate, $h=u$ | $3.237\times10^{-6}$ | $62.3$ | $61.4=1/(1-\rho^2)$ |
| Control variates, $h=(u,u^2)$ | $2.313\times10^{-8}$ | $8718$ | — |
| Stratified, $40$ strata | $1.372\times10^{-7}$ | $1470$ | — |

The single control variate has $c^\star=\operatorname{Cov}(e^U,U)/\operatorname{Var}(U)=0.140859/(1/12)=1.690309$ and $\rho^2=0.983720$, so A4 predicts a factor $1/(1-\rho^2)=61.4$ against a measured $62.3$ — agreement to within replication noise, which is the point of the exercise: the theory is not approximate, only your estimate of it is. Two control variates work because $e^u$ is nearly quadratic on $[0,1]$; this is Taylor expansion wearing a Monte Carlo costume. Stratification's factor is roughly $K^2$ for smooth $f$ (each stratum's within-variance is $O(K^{-2})$), which is why $40$ strata gave $\sim10^3$ and not $\sim40$.

Breaking antithetic sampling: $f(u)=(u-\tfrac12)^2$ has $f(1-u)=f(u)$, so $\rho=1.0000$ and the antithetic estimator is exactly **twice as bad** per evaluation (plain per-evaluation variance $5.552\times10^{-3}$). Report it as a factor of $0.5$, not as a bug.

**B2.** Reference run: $N=20000$ iterations, seed $101$, isotropic Gaussian proposal, ESS via the initial-positive-sequence estimator. Budget convention: RWMH costs one $\log\pi$ evaluation per iteration, so ESS per $1000$ evaluations is $\mathrm{ESS}/20$.

| $\sigma$ | acceptance | ESS $x_1$ | ESS $x_2$ | ESS$/10^3$ evals ($x_1$) |
|---|---|---|---|---|
| $0.2$ | $0.930$ | $4.3$ | $6.1$ | $0.22$ |
| $1.0$ | $0.656$ | $17.8$ | $17.7$ | $0.89$ |
| $2.4$ | $0.404$ | $118.7$ | $82.2$ | $5.9$ |
| $4$ | $0.257$ | $183.8$ | $197.0$ | $9.2$ |
| $6$ | $0.168$ | $249.1$ | $367.0$ | $12.5$ |
| $8$ | $0.130$ | $314.4$ | $289.3$ | $15.7$ |
| $10$ | $0.097$ | $296.1$ | $332.5$ | $14.8$ |
| $20$ | $0.037$ | $301.0$ | $266.5$ | $15.1$ |

The optimum is a broad plateau around $\sigma\in[6,20]$ with acceptance between $0.04$ and $0.17$ — nowhere near $0.234$. Two reasons, and you should be able to say both: the $0.234$ result is asymptotic in $d$ for *product* targets with a scaled random-walk proposal, and $d=2$ is not large; more importantly the banana is strongly non-product, with a marginal scale of $10$ in $x_1$ and $\approx1$ in $x_2$, so an isotropic proposal cannot be simultaneously well scaled in both directions. The best compromise deliberately over-shoots in $x_2$ (paying acceptance) to make progress in $x_1$. The lesson is that $0.234$ is a useful default for a *well-preconditioned* sampler and a distraction otherwise.

Validating the ESS code on AR(1) with $x_{k+1}=\phi x_k+\sqrt{1-\phi^2}\xi_k$: the exact integrated autocorrelation time is $\tau=(1+\phi)/(1-\phi)$, so your estimator should recover $\tau\approx19$ at $\phi=0.9$ and $\approx199$ at $\phi=0.99$ from a long chain, with the usual downward bias when the chain length is not $\gg\tau$. Do this before trusting any ESS number in B2 or B3; a truncation bug shows up here and nowhere else.

**B3.** Reference run: seed $202$, $N=20000$ iterations. Budget convention: HMC with $L$ leapfrog steps costs $L$ gradient evaluations per iteration, so ESS per $1000$ gradients is $\mathrm{ESS}\cdot1000/(NL)$.

| $(\epsilon,L)$ | acceptance | ESS $x_1$ | ESS $x_2$ | ESS$/10^3$ grads ($x_1$, $x_2$) |
|---|---|---|---|---|
| $(0.2,20)$ | $0.997$ | $723.7$ | $1058.2$ | $1.72$, $2.52$ |
| $(0.5,10)$ | $0.981$ | $1040$ | $1341$ | $4.73$, $6.10$ |
| $(0.6,8)$ | $0.970$ | $940.6$ | $1231.1$ | $5.23$, $6.84$ |
| $(0.8,5)$ | $0.938$ | $642.5$ | $881.9$ | $5.35$, $7.35$ |
| $(1.0,3)$ | $0.938$ | — | — | $5.15$, $8.35$ |

HMC's best per-gradient efficiency ($\approx5.3$ per $1000$) is *comparable to* RWMH's best per-evaluation efficiency ($\approx15.7$ per $1000$) on this two-dimensional problem — and by that convention RWMH wins. This is the honest comparison and it is worth sitting with: a gradient is not a free density evaluation, in $d=2$ the random walk is not badly handicapped, and HMC's overhead is real. Note also $(0.2,20)$: acceptance $0.997$, the best raw ESS per *iteration*, and the worst per gradient. Tuning to acceptance instead of to cost gets this exactly backwards.

The dimension sweep is where the picture inverts. With $\sigma=2.4/\sqrt d$ and $\epsilon=0.9d^{-1/4}$, $L=\lceil1/\epsilon\rceil$, seeds $1000+d$, $N=20000$ on $N(0,I_d)$:

| $d$ | RWMH acc | ESS$/10^3$ evals | HMC acc | HMC $(\epsilon,L)$ | ESS$/10^3$ grads |
|---|---|---|---|---|---|
| $2$ | $0.351$ | $132.59$ | $0.921$ | $(0.757,2)$ | $292.24$ |
| $10$ | $0.259$ | $28.67$ | $0.930$ | $(0.506,2)$ | $106.67$ |
| $50$ | $0.233$ | $6.15$ | $0.932$ | $(0.338,3)$ | $72.06$ |
| $200$ | $0.234$ | $2.65$ | $0.926$ | $(0.239,5)$ | $71.53$ |

RWMH's acceptance converges to $0.234$ to three decimals — the asymptotic theory is exactly right for this product target — while its efficiency falls by a factor of $50$ from $d=2$ to $d=200$. HMC's falls by a factor of $4$ and is essentially flat from $d=50$ onward. That gap is the whole reason gradient-based samplers took over, and you cannot see it in $d=2$.

**B4.** Reference run: $n=20$ points from $N(2,0.5^2)$, seed $5$, giving $\bar x=1.9031$ and $\sum(x_i-\bar x)^2=4.3629$; priors $\mu_0=0$, $\lambda_0=0.1$, $a_0=b_0=2$. CAVI ELBO trace:
$$-19.96383110\ \to\ -19.83677331\ \to\ -19.83665127\ \to\ -19.83665108\ \to\ \text{(converged)}.$$
Monotone at every step, as A12/Proposition 13.33 require. The exact log marginal likelihood is $-19.81596249$ (Normal–Gamma is conjugate: $\lambda_n=20.100$, $a_n=12.000$, $b_n=4.3616$), so the converged gap is
$$D_{\mathrm{KL}}(q\|p(\mu,\tau\mid x))=0.02068859\ \text{nats}.$$
Two decimal places of a nat sounds negligible, and the point of the exercise is that it is not the right thing to look at: $\operatorname{Var}_q(\mu)=0.018083$ against the exact posterior variance $0.019727$, a ratio of $0.9167$. An $8\%$ understatement of the variance is a $4\%$ understatement of every interval width, produced by a KL gap of two hundredths of a nat. Small divergences do not imply small errors in the functionals you report — and here the direction is the one A13 predicts.

To break monotonicity, cache $\mathbb E_q[\tau]$ before updating $q(\tau)$ and reuse the stale value in the $q(\mu)$ update. The ELBO trace then wanders — it will typically still increase on most iterations, but you will see decreases, and if you make the staleness worse (freeze it for several sweeps) you can get sustained oscillation. This is worth doing once, because it is the failure mode of nearly every hand-written CAVI: the algorithm is only monotone if each update is an exact maximization *given the current* values of the other factors, and a stale expectation silently breaks that hypothesis. Monotonicity is your unit test.

**B5.** Reference implementation: NumPy, $1024$ points on the unit circle plus $N(0,0.05^2)$ noise, one latent dimension, $32$ hidden units, fixed decoder noise $\sigma=0.1$, Adam at $3\times10^{-3}$, $8000$ steps, batch $128$, seed $0$. Finite-difference gradient check: maximum relative error $1.29\times10^{-8}$ — do this first, and report it, because a hand-written VAE with a subtly wrong KL gradient still trains and still produces plausible pictures.

| $\beta$ | reconstruction term | KL (nats) | ELBO |
|---|---|---|---|
| $0$ | $-0.5512$ | $7.4096$ | $-6.8584$ |
| $1$ | $-0.1836$ | $2.9354$ | $-2.7518$ |
| $4$ | $1.7610$ | $2.1592$ | $-3.9202$ |
| $20$ | $11.5260$ | $1.1040$ | $-12.6300$ |
| $100$ | $47.3141$ | $0.000208$ | $-47.3143$ |

At $\beta=100$ the KL is $2\times10^{-4}$ nats and the reconstruction cost is $47.31$, against $47.19$ for a decoder that ignores $z$ entirely and outputs the data mean. The latent has been switched off. By A14 the quantity you drove to zero is $I_q(X;Z)+D_{\mathrm{KL}}(q(z)\|p(z))$, hence the mutual information: not "too much smoothing", but zero channel capacity.

The decoder curve at $\beta=1$ has radii $0.905,\,0.983,\,0.977,\,0.607,\,0.995,\,0.952,\,0.910$ at $z=-3,-2,-1,0,1,2,3$. The dip to $0.607$ at $z=0$ is the **seam**: a continuous map $\mathbb R\to\mathbb R^2$ cannot cover a circle, so the model must cut the circle somewhere and pass through the interior, and it does so near the middle of the prior where it costs the least probability mass. This is Chapter 1's circle example and Worked Example 13.21's topological obstruction, visible as a number. Do not "fix" it by training longer; fix it with a 2-D latent, or accept the artifact.

**B6.** Reference implementation: $T=200$, linear $\beta$ from $10^{-4}$ to $0.06$ (so $\bar\alpha_1=0.99990$, $\bar\alpha_T=0.002167$), MLP $9\to128\to128\to2$ with $\tanh$ and sinusoidal time features, Adam $10^{-3}$, batch $256$, $60000$ steps, seeds $3$/$11$. Final loss $0.4447$ against $2.0$ for the trivial predictor $\varepsilon_\theta\equiv0$ — quote that baseline, or the loss number means nothing.

Forward-process verification, $20000$ iterates of the single-step recursion versus the closed form of Proposition 13.43 (seeds $1000+t$):

| $t$ | empirical mean | closed-form $\sqrt{\bar\alpha_t}$ | empirical sd | closed-form $\sqrt{1-\bar\alpha_t}$ |
|---|---|---|---|---|
| $10$ | $(0.9933,-0.0004)$ | $0.9927$ | $(0.1206,0.1196)$ | $0.1202$ |
| $50$ | $(0.8304,-0.0009)$ | $0.8288$ | $(0.5576,0.5620)$ | $0.5596$ |
| $100$ | $(0.4688,-0.0078)$ | $0.4688$ | $(0.8813,0.8838)$ | $0.8833$ |
| $200$ | $(0.0415,-0.0017)$ | $0.0466$ | $(1.0021,1.0054)$ | $0.9989$ |

Do this before training. It is a three-line check that catches every off-by-one in the $\bar\alpha$ indexing, and an off-by-one there produces a model that trains happily and samples badly.

Sampling $5000$ points (seed $77$): radius mean $0.9984$, sd $0.0167$. The support is recovered to better than $2\%$ — the thing a GAN cannot do stably and the thing that made §13.9.3's argument concrete. The angle histogram in $8$ bins is $[658,670,585,785,525,478,634,665]$ against an expected $625$, giving $\chi^2=101.8$ on $7$ degrees of freedom. That is a **decisive rejection** of uniformity ($p<10^{-17}$), and the honest report is: the model has learned the manifold and has *not* learned the uniform distribution on it. Support coverage and density accuracy are different properties, and a small network trained for $60000$ steps on a rotationally symmetric target has residual angular error of order $\pm20\%$ per bin. Reporting only "radius $0.9984\pm0.0167$" would have been a misleading success story.

**B7.** Reference run: $K=5$ classes, $d=10$ features, $n=20000$ points, seed $42$; the "trained" model uses the true logits multiplied by $2.5$; $5000$ points for calibration, $15000$ for test; $15$ equal-width bins.

| | accuracy | mean confidence | ECE | NLL | Brier |
|---|---|---|---|---|---|
| As trained | $0.6664$ | $0.8503$ | $0.1840$ | $1.1303$ | $0.5048$ |
| After temperature scaling | $0.6664$ | $0.6714$ | $0.0111$ | $0.8230$ | $0.4424$ |
| True model | $0.6664$ | — | $0.0097$ | $0.8230$ | — |

The fitted temperature is $T=2.4838$ against the true $2.5$: one scalar, fitted by minimizing NLL on the calibration split, recovers the corruption almost exactly. Accuracy is identical to four decimals because dividing logits by a positive constant cannot change an argmax. Reliability rows, as trained:

| bin | $n$ | mean confidence | accuracy |
|---|---|---|---|
| $(0.47,0.53]$ | $689$ | $0.504$ | $0.388$ |
| $(0.60,0.67]$ | $836$ | $0.634$ | $0.458$ |
| $(0.73,0.80]$ | $999$ | $0.767$ | $0.481$ |
| $(0.87,0.93]$ | $1711$ | $0.902$ | $0.606$ |
| $(0.93,1.00]$ | $7413$ | $0.984$ | $0.834$ |

Every row is above the diagonal, which is the signature of uniform overconfidence — and the reason a *single* parameter suffices. A model miscalibrated in one direction on some classes and the other direction elsewhere would need vector scaling or Dirichlet calibration, and you would see it here as rows straddling the diagonal.

The null-bias experiment, on a *perfectly calibrated* model ($200$ replications, $15$ bins):

| $n$ | measured ECE |
|---|---|
| $100$ | $0.1111$ |
| $500$ | $0.0491$ |
| $2000$ | $0.0253$ |
| $10000$ | $0.0149$ |

The estimator decays like roughly $n^{-1/2}$ toward zero and is **always positive**, because within-bin sampling noise produces $|\hat p-\bar c|>0$ regardless of the truth (Proposition 13.52). Conclusion to state: an ECE difference is believable only if it exceeds the null level at your $n$ and bin count — so at $n=2000$, differences below about $0.025$ are noise, and the post-scaling $0.0111$ at $n=15000$ is at the noise floor and therefore indistinguishable from the true model's $0.0097$. Report ECE with the null level next to it, or report NLL and Brier instead.

**B8.** Reference run: $x\sim\mathrm{Unif}(-3,3)$, $y=\sin x+\sigma(x)\xi$ with $\sigma(x)=0.15+1.5\cdot\mathbf 1\{x>0\}\cdot x/3$; score $s=|y-\sin x|$ (using the true mean function, so that all the behaviour on display is conformal's and not the regressor's); $200$ repetitions, $2000$ test points each.

| $\alpha$ | $n_{\mathrm{cal}}$ | $\hat q$ | coverage (sd) | theoretical band |
|---|---|---|---|---|
| $0.20$ | $100$ | $0.689$ | $0.8023$ $(0.0398)$ | $[0.800,0.810]$ |
| $0.20$ | $1000$ | $0.666$ | $0.8006$ $(0.0155)$ | $[0.800,0.801]$ |
| $0.10$ | $100$ | $1.214$ | $0.9009$ $(0.0292)$ | $[0.900,0.910]$ |
| $0.10$ | $1000$ | $1.178$ | $0.8993$ $(0.0119)$ | $[0.900,0.901]$ |
| $0.05$ | $100$ | $1.751$ | $0.9514$ $(0.0206)$ | $[0.950,0.960]$ |
| $0.05$ | $1000$ | $1.677$ | $0.9500$ $(0.0087)$ | $[0.950,0.951]$ |

Every average sits inside its band (the $0.8993$ is inside once you account for the reported standard error over $200$ repetitions). This is a finite-sample guarantee being met exactly, with no assumption on $\sigma(x)$, no assumption of Gaussianity, and no asymptotics — the only such guarantee in the chapter.

Conditional coverage at $\alpha=0.10$, $n_{\mathrm{cal}}=2000$, $\hat q=1.231$, $200{,}000$ test points:

| $x$ bin | $(-3,-2]$ | $(-2,-1]$ | $(-1,0]$ | $(0,1]$ | $(1,2]$ | $(2,3]$ | overall |
|---|---|---|---|---|---|---|---|
| coverage | $1.0000$ | $1.0000$ | $1.0000$ | $0.9888$ | $0.8263$ | $0.6244$ | $0.9065$ |

The marginal $0.9065$ is the average of $100\%$ where nothing happens and $62\%$ where the noise is largest. Nothing is broken: this is what "marginal" means, and the theorem never promised otherwise. With the normalized score $|y-\hat\mu(x)|/\hat\sigma(x)$ ($\hat q=1.678$) the same experiment gives $0.9058,\,0.9058,\,0.9035,\,0.9068,\,0.9037,\,0.9075$, overall $0.9055$ — flat across every bin. The fix was a better score function, not a better theorem, and it buys *approximate* conditional coverage with no loss of the exact marginal guarantee.

Finally, calibrating on $\mathrm{Unif}(-3,3)$ and testing on $\mathrm{Unif}(1,3)$ ($100{,}000$ points) gives coverage $0.7348$ against a nominal $0.90$. Exchangeability was the only hypothesis, and shifting the covariate distribution destroys it. Report this number in any deployment memo you write.

## Part C

**C1. The Metropolis–Hastings algorithm: derivation and proof.**

*Setup.* Let $\pi$ be a probability measure on $(\mathbb R^d,\mathcal B)$ with density $\pi(x)$ with respect to Lebesgue measure, known only up to a constant. Let $q(x,y)$ be a proposal density in $y$ for each $x$. We look for a kernel of the form
$$K(x,\mathrm dy)=q(x,y)\alpha(x,y)\,\mathrm dy+r(x)\,\delta_x(\mathrm dy),\qquad r(x)=1-\int q(x,y)\alpha(x,y)\,\mathrm dy,$$
with $\alpha:\mathbb R^d\times\mathbb R^d\to[0,1]$ to be determined. Note immediately that $K(x,\cdot)$ is *not* absolutely continuous with respect to Lebesgue measure: it is the sum of an absolutely continuous part and an atom, so it has a density only with respect to the base measure $\mathrm{Leb}+\delta_x$. This is Chapter 1's Failure 2 in load-bearing form, and Chapter 2's Radon–Nikodym theorem is what licenses writing it down.

*Step 1: reversibility reduces to a symmetry condition.* Reversibility with respect to $\pi$ means that for all $A,B\in\mathcal B$,
$$\int_A\pi(x)K(x,B)\,\mathrm dx=\int_B\pi(y)K(y,A)\,\mathrm dy.$$
Split each side into its continuous and atomic parts. The atomic parts contribute $\int_{A\cap B}\pi(x)r(x)\,\mathrm dx$ to *both* sides and therefore cancel identically — the atom is always reversible, whatever $r$ is. What remains is
$$\int_A\!\!\int_B\pi(x)q(x,y)\alpha(x,y)\,\mathrm dy\,\mathrm dx=\int_B\!\!\int_A\pi(y)q(y,x)\alpha(y,x)\,\mathrm dx\,\mathrm dy,$$
and since $A,B$ are arbitrary this holds iff, for Lebesgue-a.e. $(x,y)$,
$$\pi(x)q(x,y)\alpha(x,y)=\pi(y)q(y,x)\alpha(y,x).\tag{$\ast$}$$
So the entire design problem is: make the function $(x,y)\mapsto\pi(x)q(x,y)\alpha(x,y)$ symmetric.

*Step 2: the Metropolis solution.* Write $u(x,y)=\pi(x)q(x,y)$. We need $u(x,y)\alpha(x,y)$ symmetric. The choice
$$\alpha(x,y)=\min\Big\{1,\frac{u(y,x)}{u(x,y)}\Big\}=\min\Big\{1,\frac{\pi(y)q(y,x)}{\pi(x)q(x,y)}\Big\}$$
gives $u(x,y)\alpha(x,y)=\min\{u(x,y),u(y,x)\}$, which is manifestly symmetric. (Where $u(x,y)=0$ set $\alpha=1$; the corresponding set of $(x,y)$ has $\pi\otimes q$-measure zero and contributes nothing to either side of $(\ast)$.) Observe that $\pi$ enters only through the ratio $\pi(y)/\pi(x)$: replacing $\pi$ by $c\pi$ leaves $\alpha$ unchanged, which is the single fact that makes MCMC applicable to unnormalized posteriors.

*Step 3: this $\alpha$ is pointwise maximal.* Suppose $\alpha'$ satisfies $(\ast)$ and $\alpha'\le1$. Fix $(x,y)$ and assume without loss of generality $u(y,x)\le u(x,y)$. Then $(\ast)$ gives $u(x,y)\alpha'(x,y)=u(y,x)\alpha'(y,x)\le u(y,x)$, so $\alpha'(x,y)\le u(y,x)/u(x,y)=\alpha(x,y)$; and in the other direction $\alpha'(y,x)\le1=\alpha(y,x)$. So no admissible acceptance function is ever larger. Larger $\alpha$ plainly means fewer rejections. That it also means a smaller asymptotic variance for *every* $L^2(\pi)$ functional is a genuine theorem — Peskun (1973) for discrete state spaces, extended by Tierney (1998) to general ones — whose hypotheses are that both kernels be reversible with respect to $\pi$ and that the comparison be of the off-diagonal parts, both of which hold here. We use it without proof (§13.12). With it, Metropolis's choice is optimal within this family: a real optimality statement, not folklore. It does not say MH is the best possible sampler; the family of candidate kernels was fixed in advance.

*Step 4: invariance, with the atom handled honestly.* Let $\mu(\mathrm dx)=\pi(x)\mathrm dx$. For $B\in\mathcal B$,
$$(\mu K)(B)=\int\pi(x)K(x,B)\,\mathrm dx=\underbrace{\int\!\!\int_B\pi(x)q(x,y)\alpha(x,y)\,\mathrm dy\,\mathrm dx}_{(\mathrm I)}+\underbrace{\int_B\pi(x)r(x)\,\mathrm dx}_{(\mathrm{II})}.$$
For $(\mathrm I)$: the integrand is non-negative and measurable, so Tonelli's theorem (Ch. 2) permits exchanging the order of integration; by $(\ast)$,
$$(\mathrm I)=\int_B\int\pi(y)q(y,x)\alpha(y,x)\,\mathrm dx\,\mathrm dy=\int_B\pi(y)\big(1-r(y)\big)\mathrm dy,$$
using the definition of $r$. Adding $(\mathrm{II})=\int_B\pi(y)r(y)\mathrm dy$ gives $(\mu K)(B)=\int_B\pi(y)\,\mathrm dy=\mu(B)$. Notice that the rejection probability $r$ appears twice with opposite signs and cancels exactly: the atom is not a nuisance to be bounded, it is the term that makes the accounting close.

*Measure-theoretic facts used.* (i) The decomposition of $K(x,\cdot)$ with respect to $\mathrm{Leb}+\delta_x$, i.e. Radon–Nikodym. (ii) Tonelli, to swap the order of integration in $(\mathrm I)$ — legitimate because the integrand is non-negative, so no integrability hypothesis is needed. (iii) A $\pi$-null-set argument to dispose of $\{u=0\}$. (iv) The fact that reversibility of a kernel is a statement about *measures*, tested against arbitrary $A,B$, not about densities pointwise.

*What the theorem does not give.* It gives invariance, and nothing else. It does not give irreducibility (a proposal supported on a ball of radius $\tfrac12$ cannot cross a gap of width $2$; the chain is still $\pi$-invariant and still wrong); it does not give aperiodicity, ergodicity, a rate, or a CLT — for those you need Harris recurrence and the results cited in §13.12, and geometric ergodicity for a CLT. It says nothing about how long you must wait, which is why §13.6 exists and why "my sampler is provably correct" is a claim about the limit and not about your run. Every diagnostic in this chapter is in the gap between Step 4 and reality.

**C2. The DDPM training objective from the variational bound.**

*Forward process.* Fix $\beta_1,\dots,\beta_T\in(0,1)$, set $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s\le t}\alpha_s$, and define $q(x_t\mid x_{t-1})=N(\sqrt{\alpha_t}\,x_{t-1},\beta_tI)$. Composition of Gaussians (Ch. 5) gives the closed form
$$q(x_t\mid x_0)=N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),$$
so $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$ with $\varepsilon\sim N(0,I)$: any noise level is reachable in one step, which is what makes training $O(1)$ per example rather than $O(T)$.

*The bound.* By Jensen (or Theorem 13.31 applied to the latent sequence $x_{1:T}$),
$$-\log p_\theta(x_0)\le\mathbb E_q\Big[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]=:L.$$
*Reorganization.* Expand and apply Bayes' rule (Prop. 1.12) to the forward transition, conditioning throughout on $x_0$:
$$q(x_t\mid x_{t-1})=q(x_t\mid x_{t-1},x_0)=\frac{q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.$$
The first equality is the Markov property. Substituting and telescoping the $q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$ factors leaves
$$L=\underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T}+\sum_{t=2}^T\underbrace{\mathbb E_q\,D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}}\underbrace{-\,\mathbb E_q\log p_\theta(x_0\mid x_1)}_{L_0}.$$
Everything so far is forced.

*The forward posterior.* Both $q(x_t\mid x_{t-1})$ and $q(x_{t-1}\mid x_0)$ are Gaussian in $x_{t-1}$, so Gaussian conjugacy (Ch. 5, Prop. 13.44) gives $q(x_{t-1}\mid x_t,x_0)=N(\tilde\mu_t,\tilde\beta_tI)$ with
$$\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\qquad \tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$
This is a precision-weighted average of the two sources of information about $x_{t-1}$, and it is *tractable* — the crucial asymmetry with the true reverse $q(x_{t-1}\mid x_t)$, which is not.

*Each term becomes a squared error.* Choose $p_\theta(x_{t-1}\mid x_t)=N(\mu_\theta(x_t,t),\sigma_t^2I)$ with $\sigma_t^2$ fixed. The KL between two Gaussians with equal covariance is a scaled squared distance between means:
$$L_{t-1}=\mathbb E_q\Big[\frac{1}{2\sigma_t^2}\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\|^2\Big]+\text{const}.$$
Now substitute $x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon)/\sqrt{\bar\alpha_t}$ into $\tilde\mu_t$; the algebra collapses to
$$\tilde\mu_t=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon\Big).$$
Parameterizing the model's mean in the *same* form with $\varepsilon_\theta$ in place of $\varepsilon$ makes the difference of means proportional to $\varepsilon-\varepsilon_\theta$, and
$$L_{t-1}=\mathbb E_{x_0,\varepsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\|\varepsilon-\varepsilon_\theta(x_t,t)\big\|^2\Big],\qquad x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon.$$
So the variational bound *is* a weighted denoising regression, with the weight $w_t=\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ determined by the schedule. DDPM then trains $\mathbb E_{t\sim\mathrm{Unif}\{1..T\}}\mathbb E_{x_0,\varepsilon}\|\varepsilon-\varepsilon_\theta(x_t,t)\|^2$, i.e. with $w_t\equiv1$.

*Forced versus chosen.*

| Ingredient | Forced or chosen | If chosen: what is lost |
|---|---|---|
| Decomposition $L_T+\sum L_{t-1}+L_0$ | forced (Bayes + telescoping) | — |
| $q(x_t\mid x_0)$ closed form | forced (Gaussian composition) | — |
| $q(x_{t-1}\mid x_t,x_0)$ Gaussian with $\tilde\mu_t,\tilde\beta_t$ | forced (conjugacy) | — |
| Each $L_{t-1}$ = scaled squared error in the means | forced *given* a Gaussian reverse with fixed covariance | — |
| Weight $w_t=\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ | forced *given* $\varepsilon$-parameterization | — |
| Fixed (non-learned) forward $\beta$ schedule | **chosen** | $L_T$ becomes parameter-free but the schedule is now a hyperparameter you must tune; a learned schedule could adapt noise levels to the data |
| Gaussian reverse with *fixed isotropic* $\sigma_t^2\in\{\beta_t,\tilde\beta_t\}$ | **chosen** | reverse variance is not modelled; costs likelihood, and learning it (Nichol–Dhariwal) improves NLL substantially |
| $\varepsilon$-parameterization rather than predicting $x_0$ or $\mu$ | **chosen** | changes the effective loss weighting across $t$ and the numerical conditioning; $x_0$-prediction is better at high noise, $\varepsilon$ at low |
| Dropping $w_t$ (the "simple" loss) | **chosen** | **the upper-bound property**. $L_{\mathrm{simple}}$ is no longer a bound on $-\log p_\theta(x_0)$. It up-weights large $t$ relative to the bound, which empirically improves sample quality and worsens likelihood |
| Uniform sampling of $t$ | **chosen** | a variance-reduction choice; importance-sampling $t$ reduces gradient variance |
| $L_0$ via a discretized Gaussian (or folded into the same regression) | **chosen** | affects reported likelihoods on discrete data only |

The row that matters is the fourth from the bottom: after dropping $w_t$ you are no longer doing variational inference. You are doing multi-scale denoising regression that *happens* to have a variational interpretation for a different weighting. Papers that report "the ELBO" for a model trained on $L_{\mathrm{simple}}$ are reporting a quantity the training objective did not optimize.

*Why this is well defined without a data density.* Every quantity above is an expectation over $x_0\sim p_{\mathrm{data}}$, $t\sim\mathrm{Unif}$, $\varepsilon\sim N(0,I)$ of a bounded-below continuous function of $(x_0,t,\varepsilon)$. That is an integral against the data *measure*; it never divides by $p_{\mathrm{data}}$, never evaluates $\log p_{\mathrm{data}}$, and never requires $p_{\mathrm{data}}\ll\mathrm{Leb}$. The densities that do appear — $q(x_t\mid x_0)$, $p_\theta(x_{t-1}\mid x_t)$ — are Gaussians we wrote down ourselves, and they are positive everywhere. Contrast §13.9.3: a GAN's Jensen–Shannon objective compares two measures that may be mutually singular, and singularity makes it constant. Diffusion's objective compares a *sample* to a *known noise vector*. That is the whole trick, and it is why the method works on data supported on a manifold of measure zero — as verified in B6, where a $2$-D model recovered a $1$-D circle to a radius sd of $0.0167$.

**C3. Split-conformal coverage: proof and interrogation.**

*Assumption (stated precisely).* Let $Z_i=(X_i,Y_i)$ for $i=1,\dots,n$ be the calibration data and $Z_{n+1}=(X_{n+1},Y_{n+1})$ the test point. Assume the $n+1$ tuples are **exchangeable**: for every permutation $\sigma$ of $\{1,\dots,n+1\}$, $(Z_{\sigma(1)},\dots,Z_{\sigma(n+1)})$ has the same joint law as $(Z_1,\dots,Z_{n+1})$. I.i.d. sampling implies this and is strictly stronger. Crucially, the fitted model $\hat\mu$ is trained on a **separate** split and is treated as a fixed measurable function; the score $s(x,y)=|y-\hat\mu(x)|$ is then a fixed function, so $S_i=s(Z_i)$ inherits exchangeability from $Z_i$. Assume additionally that the $S_i$ are a.s. distinct.

*Proof.* Let $k=\lceil(n+1)(1-\alpha)\rceil$ and $\hat q=S_{(k)}$, the $k$-th smallest of $S_1,\dots,S_n$, and set $C(X_{n+1})=\{y:s(X_{n+1},y)\le\hat q\}$. Because $(S_1,\dots,S_{n+1})$ is exchangeable and a.s. tie-free, the rank $R$ of $S_{n+1}$ in the full sample of $n+1$ is uniform on $\{1,\dots,n+1\}$. And $\{S_{n+1}\le S_{(k)}\}=\{R\le k\}$. Therefore
$$\mathbb P\big(Y_{n+1}\in C(X_{n+1})\big)=\mathbb P(R\le k)=\frac{k}{n+1}\in\Big[1-\alpha,\ 1-\alpha+\frac1{n+1}\Big),$$
the lower bound by $k\ge(n+1)(1-\alpha)$ and the upper by $k<(n+1)(1-\alpha)+1$. $\square$

*(i) Where each hypothesis is used.* Exchangeability is used **once**, in the single sentence "$R$ is uniform on $\{1,\dots,n+1\}$". Nothing else in the proof uses it. Distinctness is used to turn $\{S_{n+1}\le S_{(k)}\}$ into $\{R\le k\}$ exactly, and only affects the *upper* bound; with ties, coverage can only rise. About the model $\hat\mu$ the proof assumes **nothing at all**: not consistency, not correct specification, not even measurability of anything beyond $s$ being a fixed function. $\hat\mu$ may be a neural network, a constant, or deliberately adversarial. What it does require is that $\hat\mu$ not have seen the calibration data — that is what "split" buys, and reusing training data for calibration is the one mistake that voids the theorem silently, because the scores are then stochastically smaller than the test score and coverage falls below nominal.

*(ii) What the guarantee is marginal over.* Over the joint law of everything random: the calibration draw *and* the test point, including $X_{n+1}$. It is a statement about the long-run behaviour of the *procedure*, exactly in the sense of Chapter 9's coverage, not a statement about the interval you are holding. From the B8 run at $\alpha=0.10$ with $n_{\mathrm{cal}}=2000$ ($\hat q=1.231$, $200{,}000$ test points), coverage by covariate bin was $1.0000$, $1.0000$, $1.0000$, $0.9888$, $0.8263$, $0.6244$ for $x$ in $(-3,-2],\dots,(2,3]$, with overall $0.9065$. The nominal $90\%$ is achieved by over-covering the quiet region and under-covering the noisy one by $28$ points. Nothing has gone wrong; the theorem's promise is the average and the average is met. Since $\sigma(x)$ can be made to grow arbitrarily fast on a region of arbitrarily small probability, the conditional coverage on that region can be pushed arbitrarily close to $0$ while marginal coverage stays at $1-\alpha$. Distribution-free *conditional* coverage is in fact impossible for non-trivial intervals (Vovk 2012; Barber et al. 2021): any procedure achieving it must produce infinite-length intervals for almost every $x$. So the fix is not a better theorem but a better **score**: with $s=|y-\hat\mu(x)|/\hat\sigma(x)$ the same experiment gave $0.9058$, $0.9058$, $0.9035$, $0.9068$, $0.9037$, $0.9075$ — approximately conditional, exactly marginal, and the exact guarantee is untouched because the proof never looked at $s$.

*(iii) Under covariate shift.* Exchangeability between calibration and test points is precisely what shift destroys. Calibrating on $\mathrm{Unif}(-3,3)$ and testing on $\mathrm{Unif}(1,3)$ gave coverage $0.7348$ against a nominal $0.90$: a $17$-point shortfall from a shift that a monitoring dashboard tracking only input means would barely register. The available fix is *weighted* conformal prediction (Tibshirani et al. 2019): if the shift is pure covariate shift, $\mathrm dP_{\mathrm{test}}(x,y)=w(x)\,\mathrm dP_{\mathrm{cal}}(x,y)$ with the same $Y\mid X$, then replacing the empirical quantile by the $w(X_i)$-weighted quantile restores exact coverage. Its own assumption is named in that sentence and is strong: the likelihood ratio $w$ must be known or well estimated, and the conditional law of $Y$ given $X$ must be unchanged. Label shift, concept drift, and feedback loops are not covered.

*(iv) Deployment recommendation.* Use split conformal, because a finite-sample distribution-free marginal guarantee is more than any other method in this chapter offers, and it costs one held-out split and a sort. But: hold out a calibration set that the model has never touched, in any form, including through hyperparameter selection; use a *normalized or quantile-based* score so that the intervals adapt to local difficulty, since the exact guarantee is free of charge and the approximate conditional coverage is what your users actually experience; report group-conditional coverage for every subgroup you would be embarrassed to under-cover, and treat those numbers, not the marginal one, as the deployment criterion; monitor for covariate shift and recalibrate on fresh exchangeable data at a cadence set by how fast your inputs drift, since our own measurement showed coverage falling from $0.90$ to $0.73$ under a shift far milder than most production drift; and never describe the output as "a $90\%$ chance that $y$ is in this interval", which is a statement about this interval that the theorem does not license.

---

### References and further reading

**Textbooks and monographs.**

- C. P. Robert & G. Casella, *Monte Carlo Statistical Methods*, 2nd ed., Springer, 2004. The standard statistical treatment of §§13.1–13.4.
- A. B. Owen, *Monte Carlo Theory, Methods and Examples*, 2013–, available online. The best single reference for importance sampling, control variates, stratification, and quasi-Monte Carlo; Ch. 9 contains the bias expansion for self-normalized estimators used in Proposition 13.7.
- S. Meyn & R. L. Tweedie, *Markov Chains and Stochastic Stability*, 2nd ed., Cambridge, 2009. The source for Theorem 13.19 and for the drift/minorization conditions behind geometric ergodicity.
- G. O. Roberts & J. S. Rosenthal, "General state space Markov chains and MCMC algorithms", *Probability Surveys* 1 (2004), 20–71. A readable route into the same material.
- A. Gelman, J. Carlin, H. Stern, D. Dunson, A. Vehtari & D. Rubin, *Bayesian Data Analysis*, 3rd ed., CRC, 2013. Practice, priors, and diagnostics.
- S. Brooks, A. Gelman, G. Jones & X.-L. Meng (eds.), *Handbook of Markov Chain Monte Carlo*, CRC, 2011. Contains Neal's HMC chapter and Geyer's chapter on autocorrelation-time estimation (the initial-positive-sequence estimator used for every ESS in this chapter).
- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006, Ch. 10. Still the clearest textbook derivation of CAVI and mean-field VI, including the Normal–Gamma example of §13.7.2.
- D. Bakry, I. Gentil & M. Ledoux, *Analysis and Geometry of Markov Diffusion Operators*, Springer, 2014. The rigorous theory behind Proposition 13.26.
- V. Vovk, A. Gammerman & G. Shafer, *Algorithmic Learning in a Random World*, 2nd ed., Springer, 2022. The full conformal theory.
- A. N. Angelopoulos & S. Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification", arXiv:2107.07511, 2021. Start here for §13.10.3.
- T. Gneiting & A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation", *JASA* 102 (2007), 359–378. Definition 13.50 and Proposition 13.51.
- A. Vehtari, A. Gelman, D. Simpson, B. Carpenter & P.-C. Bürkner, "Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC", *Bayesian Analysis* 16 (2021), 667–718.
- A. Vehtari, D. Simpson, A. Gelman, Y. Yao & J. Gabry, "Pareto smoothed importance sampling", *JMLR* 25 (2024), no. 72. The $\hat k$ diagnostic of Definition 13.6.
- N. Chopin & O. Papaspiliopoulos, *An Introduction to Sequential Monte Carlo*, Springer, 2020. The main topic this chapter omits (§13.11); the natural continuation of §§13.2 and 13.4.
- L. Devroye, *Non-Uniform Random Variate Generation*, Springer, 1986. Where the exact samplers assumed throughout come from.
- K. Ball, *An Elementary Introduction to Modern Convex Geometry*, MSRI, 1997. Lecture 1 has the ball-volume asymptotics used in Proposition 13.13.

**Machine-learning papers.**

- N. Metropolis, A. Rosenbluth, M. Rosenbluth, A. Teller & E. Teller, "Equation of state calculations by fast computing machines", *J. Chem. Phys.* 21 (1953), 1087–1092.
- W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications", *Biometrika* 57 (1970), 97–109.
- G. O. Roberts, A. Gelman & W. R. Gilks, "Weak convergence and optimal scaling of random walk Metropolis algorithms", *Ann. Appl. Prob.* 7 (1997), 110–120. The $0.234$ result.
- S. Duane, A. D. Kennedy, B. J. Pendleton & D. Roweth, "Hybrid Monte Carlo", *Phys. Lett. B* 195 (1987), 216–222.
- R. M. Neal, "MCMC using Hamiltonian dynamics", in *Handbook of MCMC*, 2011. Definition 13.23 and Propositions 13.24–13.25.
- M. D. Hoffman & A. Gelman, "The No-U-Turn Sampler", *JMLR* 15 (2014), 1593–1623.
- A. Beskos, N. Pillai, G. Roberts, J.-M. Sanz-Serna & A. Stuart, "Optimal tuning of the hybrid Monte Carlo algorithm", *Bernoulli* 19 (2013), 1501–1534. The $d^{1/4}$ step-size scaling behind Worked Example 13.14.
- M. Betancourt, "A conceptual introduction to Hamiltonian Monte Carlo", arXiv:1701.02434, 2017.
- M. Welling & Y. W. Teh, "Bayesian learning via stochastic gradient Langevin dynamics", *ICML*, 2011.
- Q. Li, C. Tai & W. E, "Stochastic modified equations and dynamics of stochastic gradient algorithms", *JMLR* 20 (2019), no. 40. The honest version of the SGD-as-sampler analogy (Common Confusion 13.5).
- D. P. Kingma & M. Welling, "Auto-encoding variational Bayes", *ICLR*, 2014.
- D. J. Rezende, S. Mohamed & D. Wierstra, "Stochastic backpropagation and approximate inference in deep generative models", *ICML*, 2014.
- Y. Burda, R. Grosse & R. Salakhutdinov, "Importance weighted autoencoders", *ICLR*, 2016. Proposition 13.38.
- T. Rainforth, A. R. Kosiorek, T. A. Le, C. J. Maddison, M. Igl, F. Wood & Y. W. Teh, "Tighter variational bounds are not necessarily better", *ICML*, 2018.
- I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed & A. Lerchner, "$\beta$-VAE: learning basic visual concepts with a constrained variational framework", *ICLR*, 2017.
- A. Alemi, B. Poole, I. Fischer, J. Dillon, R. A. Saurous & K. Murphy, "Fixing a broken ELBO", *ICML*, 2018. The rate–distortion reading of Proposition 13.37.
- S. R. Bowman, L. Vilnis, O. Vinyals, A. Dai, R. Jozefowicz & S. Bengio, "Generating sentences from a continuous space", *CoNLL*, 2016. Posterior collapse in the wild.
- D. M. Blei, A. Kucukelbir & J. D. McAuliffe, "Variational inference: a review for statisticians", *JASA* 112 (2017), 859–877.
- M. D. Hoffman, D. M. Blei, C. Wang & J. Paisley, "Stochastic variational inference", *JMLR* 14 (2013), 1303–1347.
- A. Hyvärinen, "Estimation of non-normalized statistical models by score matching", *JMLR* 6 (2005), 695–709. Proposition 13.40.
- P. Vincent, "A connection between score matching and denoising autoencoders", *Neural Computation* 23 (2011), 1661–1674. Proposition 13.41 — arguably the single most consequential identity in the second half of this chapter.
- Y. Song & S. Ermon, "Generative modeling by estimating gradients of the data distribution", *NeurIPS*, 2019.
- J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan & S. Ganguli, "Deep unsupervised learning using nonequilibrium thermodynamics", *ICML*, 2015. The original diffusion construction.
- J. Ho, A. Jain & P. Abbeel, "Denoising diffusion probabilistic models", *NeurIPS*, 2020. Theorem 13.45.
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon & B. Poole, "Score-based generative modeling through stochastic differential equations", *ICLR*, 2021. Proposition 13.46.
- B. D. O. Anderson, "Reverse-time diffusion equation models", *Stochastic Processes and their Applications* 12 (1982), 313–326.
- A. Nichol & P. Dhariwal, "Improved denoising diffusion probabilistic models", *ICML*, 2021. Learning the reverse variance.
- J. Ho & T. Salimans, "Classifier-free diffusion guidance", *NeurIPS Workshop*, 2021.
- T. Karras, M. Aittala, T. Aila & S. Laine, "Elucidating the design space of diffusion-based generative models", *NeurIPS*, 2022.
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel & M. Le, "Flow matching for generative modeling", *ICLR*, 2023.
- I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville & Y. Bengio, "Generative adversarial nets", *NeurIPS*, 2014. And M. Arjovsky, S. Chintala & L. Bottou, "Wasserstein GAN", *ICML*, 2017, for the diagnosis of §13.9.3's obstruction.
- C. Guo, G. Pleiss, Y. Sun & K. Q. Weinberger, "On calibration of modern neural networks", *ICML*, 2017. Temperature scaling.
- A. Kumar, P. Liang & T. Ma, "Verified uncertainty calibration", *NeurIPS*, 2019. Why binned ECE is biased and what to do instead (Proposition 13.52).
- B. Lakshminarayanan, A. Pritzel & C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles", *NeurIPS*, 2017.
- V. Vovk, "Conditional validity of inductive conformal predictors", *ACML*, 2012; and R. F. Barber, E. J. Candès, A. Ramdas & R. J. Tibshirani, "The limits of distribution-free conditional predictive inference", *Information and Inference* 10 (2021), 455–482.
- R. J. Tibshirani, R. F. Barber, E. J. Candès & A. Ramdas, "Conformal prediction under covariate shift", *NeurIPS*, 2019.

**Classical results cited without proof.**

- P. H. Peskun, "Optimum Monte-Carlo sampling using Markov chains", *Biometrika* 60 (1973), 607–612; and L. Tierney, "A note on Metropolis–Hastings kernels for general state spaces", *Ann. Appl. Prob.* 8 (1998), 1–9. The variance ordering used in Solution C1.
- C. J. Geyer, "Practical Markov chain Monte Carlo", *Statistical Science* 7 (1992), 473–483. The initial-positive-sequence estimator behind every ESS in this chapter.
- B. F. Logan, C. L. Mallows, S. O. Rice & L. A. Shepp, "Limit distributions of self-normalized sums", *Ann. Prob.* 1 (1973), 788–809. Why the coverage in Solution A2(c) settles at a constant instead of collapsing.
- R. M. Neal, "Annealed importance sampling", *Statistics and Computing* 11 (2001), 125–139; and R. B. Grosse, Z. Ghahramani & R. P. Adams, "Sandwiching the marginal likelihood using bidirectional Monte Carlo", arXiv:1511.02543, 2015. How the variational gap can in fact be bounded on both sides (§13.11).
- D. J. Earl & M. W. Deem, "Parallel tempering: theory, applications, and new perspectives", *Phys. Chem. Chem. Phys.* 7 (2005), 3910–3916. The standard remedy discussed in §13.4.4.
- A. Doucet, N. de Freitas & N. Gordon (eds.), *Sequential Monte Carlo Methods in Practice*, Springer, 2001.

**Other generative-model families (context for §13.9).**

- G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed & B. Lakshminarayanan, "Normalizing flows for probabilistic modeling and inference", *JMLR* 22 (2021), no. 57.
- A. van den Oord, N. Kalchbrenner & K. Kavukcuoglu, "Pixel recurrent neural networks", *ICML*, 2016. The autoregressive strategy.
- E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer & P. Hennig, "Laplace redux: effortless Bayesian deep learning", *NeurIPS*, 2021; and W. Maddox, P. Izmailov, T. Garipov, D. Vetrov & A. G. Wilson, "A simple baseline for Bayesian uncertainty in deep learning", *NeurIPS*, 2019 (SWAG). The practical epistemic-uncertainty methods surveyed in §13.10.1.
## Where to go next

**If you want to go deeper on Monte Carlo and variance reduction.** Art Owen's *Monte Carlo theory, methods and examples* (free online, continuously updated) is the reference: rigorous, practical, and complete on control variates, stratification, quasi-Monte Carlo, and importance sampling diagnostics. Then Robert & Casella, *Monte Carlo Statistical Methods*, for the classical statistical framing. Randomized quasi-Monte Carlo is the main topic this chapter skipped entirely; it can achieve close to $O(n^{-1})$ for smooth low-effective-dimension integrands and is underused in ML.

**If you want to go deeper on MCMC.** Start with Roberts & Rosenthal's survey "General state space Markov chains and MCMC algorithms" (*Probability Surveys*, 2004) for the theory this chapter borrowed, then Meyn & Tweedie for the full treatment. For HMC, Betancourt's "A Conceptual Introduction to Hamiltonian Monte Carlo" (arXiv:1701.02434) is the best geometric exposition; Neal's "MCMC using Hamiltonian dynamics" (*Handbook of MCMC*, 2011) is the best practical one. For diagnostics, Vehtari, Gelman, Simpson, Carpenter & Bürkner, "Rank-normalization, folding, and localization: an improved $\hat R$" (*Bayesian Analysis*, 2021). Active research directions: non-reversible and lifted samplers, piecewise-deterministic Markov processes (bouncy particle, zig-zag), stochastic-gradient MCMC with provable bias control, and MCMC in discrete spaces (a genuinely open area with immediate applications to combinatorial generative modelling).

**If you want to go deeper on variational inference.** Blei, Kucukelbir & McAuliffe, "Variational Inference: A Review for Statisticians" (*JASA*, 2017) is the map; Hoffman, Blei, Wang & Paisley, "Stochastic Variational Inference" (*JMLR*, 2013) is the scaling result; Khan & Lin's work on natural-gradient/Bayesian-learning-rule formulations unifies VI, EM, Laplace, and Newton in one framework and is the most conceptually satisfying recent development. Open directions: tighter and lower-variance bounds, structured and normalizing-flow variational families, and diagnostics for the variational gap (currently the weakest link).

**If you want to go deeper on generative models.** For VAEs: Kingma & Welling's monograph *An Introduction to Variational Autoencoders* (2019), then the posterior-collapse literature (Bowman et al. 2016; Alemi et al., "Fixing a Broken ELBO", 2018 — which is the rate–distortion reading of §13.8.2). For diffusion: Ho, Jain & Abbeel (2020) for DDPM, Song et al. (2021) "Score-Based Generative Modeling through SDEs" for the unifying continuous view, Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based Generative Models" for the engineering, and Lipman et al. (2023) on flow matching, which is where the field has largely moved: it trains a vector field by regression against a conditional target and subsumes diffusion as one choice of path. Open directions: few-step samplers with guarantees, discrete diffusion for text, and a theory of guidance.

**If you want to go deeper on uncertainty.** Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (2021), then Vovk, Gammerman & Shafer's book for the full theory. For calibration: Guo et al. (2017) for temperature scaling, Kumar, Liang & Ma (2019) on why ECE estimators are biased and what to do instead, Gneiting & Raftery (2007) on proper scoring rules. For epistemic uncertainty in deep models: Lakshminarayanan et al. (2017) on deep ensembles, and the "loss landscape" literature on why ensembles beat single-mode posteriors. Open directions: conditional coverage under assumptions weak enough to be believable, uncertainty under shift, and calibration for generative and sequence models, where "the predicted distribution" is over an enormous structured space and even *defining* calibration is unsettled.

**A last piece of advice.** The most valuable habit this tutorial can leave you with is not a theorem but a reflex: whenever you meet an expectation, ask *what measure, what estimator, what variance, and what happens when the hypothesis fails*. Every failure catalogued in this chapter — lying error bars, stuck chains, collapsed latents, overconfident classifiers, evaporating coverage — was a case of someone not asking one of those four questions.
