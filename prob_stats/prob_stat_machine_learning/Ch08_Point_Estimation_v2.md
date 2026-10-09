# Chapter 8 — Point Estimation

## Chapter 8 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 8.1 Estimators and their evaluation | An estimator is a measurable function of the data, hence a random variable; judge it by bias, variance, MSE, consistency, and asymptotic variance | Every reported metric, every learned weight, is an estimator; knowing that "unbiased" is not a synonym for "good" is what licenses regularization |
| 8.2 Method of moments | Match sample moments to population moments and solve | The cheap, likelihood-free initializer; the ancestor of GMM in econometrics and of moment-matching objectives (MMD, feature matching in GANs) |
| 8.3 Maximum likelihood | Maximize the density of the observed data with respect to a *stated* base measure | Cross-entropy loss **is** the negative log-likelihood; the failure modes of MLE are the failure modes of your training loop |
| 8.4 Score and Fisher information | The gradient of the log-likelihood has mean zero; its variance equals the expected curvature | Natural gradient, K-FAC, Laplace approximations, and "sharp vs flat minima" all live here |
| 8.5 Cramér–Rao bound | No unbiased estimator has variance below $1/(nI(\theta))$; when regularity fails, Chapman–Robbins replaces it | The reference point against which every estimator, including a trained network, is measured |
| 8.6 Asymptotics of the MLE | $\sqrt{n}(\hat\theta-\theta)\to_d\mathcal N(0,I(\theta)^{-1})$; the delta method for transformed parameters; profile likelihood for nuisance parameters; the sandwich under misspecification | Every standard error you report, and why model-based error bars are too small when the model is wrong — which, for us, is always |
| 8.7 Sufficiency and Rao–Blackwell | Conditioning on a sufficient statistic never hurts and usually helps | Why averaging over nuisance randomness (dropout masks, data order, augmentations) reduces variance for free |
| 8.8 MAP, regularization, shrinkage | Priors are penalties; the optimal shrinkage factor beats the MLE at every $\theta$, and from $d\ge3$ you can estimate it | Weight decay $=$ Gaussian prior; lasso $=$ Laplace prior; James–Stein is the first proof that high dimensions change the rules |
| 8.9 The EM algorithm | Alternately tighten and maximize the ELBO | EM and variational inference are one construction; the E-step is exact in one and approximate in the other |
| 8.10 Bias–variance when $p \gg n$ | The classical U-curve is a special case, not a law | Double descent, benign overfitting, and why "more parameters $\Rightarrow$ more variance" is false |

---

## 8.0 What this chapter is for

You have spent seven chapters building machinery. This is the chapter where the machinery starts paying rent.

Here is the situation. Nature has a distribution $P$. You do not know it. You observe $X_1,\dots,X_n$ drawn i.i.d. from $P$, and you want to produce a number — or a vector, or a matrix of a hundred billion weights — that summarizes something about $P$. That production rule is an *estimator*, and the entire content of this chapter is: how do we build estimators, and how do we tell a good one from a bad one?

The reason this deserves a measure-theoretic treatment rather than a recipe book is that the two most important estimators in machine learning, maximum likelihood and its Bayesian cousin MAP, are defined by an optimization over densities, and Chapter 2 taught us that a density is not a thing a distribution *has* — it is a thing a distribution has *relative to a base measure*. Get sloppy about the base measure and "maximum likelihood" stops being well defined. Get careful about it and suddenly MLE works perfectly well for the distribution $A = \max(0,Z)$ from Chapter 1, which is neither discrete nor continuous, and which defeated every formula in a first course. That is not a technicality; it is the difference between a method that covers censored survival data, zero-inflated counts, and quantized activations, and a method that covers textbook exercises.

What you are assumed to bring. From Chapter 2: differentiating under the integral sign, with the Dominated Convergence Theorem as its engine — the "regularity conditions" that litter this chapter are exactly DCT hypotheses in disguise, and I will point that out every single time; Radon–Nikodym densities relative to a stated base measure; and the ELBO identity, $\log p_\theta(x) = \mathcal L(q,\theta) + D_{\mathrm{KL}}(q \,\|\, p_\theta(\cdot\mid x))$, which will reappear in §8.9 as the whole of EM. From Chapter 3: exponential families, their log-partition function $A(\eta)$ with $\nabla A(\eta)=\mathbb E[T(X)]$ and $\nabla^2 A(\eta)=\operatorname{Var}(T(X))$, sufficiency, and the Fisher–Neyman factorization. From Chapter 4: Jensen's inequality, in the form that makes $\mathbb E_P[\log (q/p)] \le 0$. From Chapter 5: the Gaussian identities. From Chapter 6: conditional expectation as an $L^2$ projection, the tower property, the law of total variance, and the bias–variance decomposition — Rao–Blackwell in §8.7 is the law of total variance wearing a hat. From Chapter 7: the SLLN, the CLT, Slutsky's theorem, and the delta method; these four supply essentially the entire asymptotic story. And from Chapter 1, cited by number: Definition 1.16 (law as pushforward), Definition 1.23 (independence), and Proposition 1.12 (Bayes' rule), which is all MAP is.

The punchline, stated in advance, has four parts. **First**: maximum likelihood is not a principle handed down from on high; it is the empirical minimizer of KL divergence to the truth, and everything good about it (consistency, asymptotic efficiency, reparameterization invariance) and everything bad about it (unbounded likelihoods, spurious spikes, overconfidence under misspecification) follows from that one fact. **Second**: the Fisher information is simultaneously the variance of the score and the expected curvature of the log-likelihood, and this coincidence — the *information equality* — is what makes second-order optimization and uncertainty quantification the same subject. When the model is wrong, the coincidence breaks, the two quantities separate, and the gap between them is exactly the correction the sandwich estimator applies. **Third**: unbiasedness is a red herring. It is not free, it is not always achievable, it is sometimes achievable only by absurd estimators, and in dimension $\ge 3$ the unbiased estimator of a Gaussian mean is *inadmissible* — beaten everywhere by a shrunken version of itself. Regularization is not a hack you apply because the theory ran out; the theory says to do it. **Fourth**: the classical bias–variance U-curve is a theorem about a specific regime, not a law of nature, and modern overparameterized models live outside that regime. I will be honest about how much of that last story is settled (some) and how much is active research (most).

A word on how to read this. The sections are ordered so that each one is a tool the next one needs: you cannot state the Cramér–Rao bound without Fisher information, and you cannot get the asymptotic distribution of the MLE without both. If you are short on time, §8.3, §8.4, and §8.6 are the spine. But §8.8 and §8.10 are where the surprises are, and §8.9 is where the chapter cashes in Chapter 2's ELBO for an algorithm you will implement.

---

## 8.1 Estimators and their evaluation

### 8.1.1 What an estimator is

Start with the thing itself. We have a family of candidate distributions $\{P_\theta : \theta \in \Theta\}$ on a measurable space $(\mathcal X, \mathcal A)$, indexed by a parameter $\theta$ living in $\Theta \subseteq \mathbb R^d$. We observe $n$ i.i.d. draws, which means the actual probability space carries the product measure $P_\theta^{\otimes n}$ on $(\mathcal X^n, \mathcal A^{\otimes n})$. An estimator is a rule that turns the sample into a guess.

The important word in the definition below is *measurable*, and the important consequence is that an estimator is a **random variable** — everything Chapter 1 said about random variables applies to it. In particular, an estimator has a distribution (its *sampling distribution*), and the whole of this chapter is the study of that distribution. The randomness is not in $\theta$; $\theta$ is a fixed unknown constant. The randomness is in which sample you happened to draw, which is to say, in $\omega$ — the random seed.

> **Definition 8.1 (Statistic, estimator).**
> A **statistic** is a measurable function $T : (\mathcal X^n, \mathcal A^{\otimes n}) \to (\mathcal Y, \mathcal B)$ that does not depend on $\theta$. An **estimator** of a quantity $g(\theta) \in \mathbb R^k$ is a statistic $\hat g_n : \mathcal X^n \to \mathbb R^k$ (with $\mathbb R^k$ carrying $\mathcal B(\mathbb R^k)$, per Definition 1.6). We write $\hat g_n = \hat g_n(X_1,\dots,X_n)$ and suppress the arguments when they are clear. An **estimate** is the realized value $\hat g_n(x_1,\dots,x_n)$ at an observed sample.

Two remarks that are worth more than they look.

*"Does not depend on $\theta$" is a real constraint.* The function $(x_1,\dots,x_n)\mapsto \theta$ is measurable and would make a fantastic estimator, but you cannot compute it, because you do not know $\theta$. This is the whole difficulty of statistics compressed into one clause.

*Measurability is not automatic when the estimator is defined by an optimization.* If $\hat\theta_n := \arg\max_{\theta\in\Theta} \ell_n(\theta)$ and $\Theta$ is an uncountable set, then $\hat\theta_n$ is a supremum over an uncountable family, and Chapter 1 warned us in bold letters that **countability is load-bearing**: a supremum over an uncountable index set need not be measurable. What rescues us in practice is the same thing that rescued the uniform law of large numbers: if $\Theta$ is separable and $\theta \mapsto \ell_n(\theta)$ is continuous, then $\sup_{\theta \in \Theta} \ell_n(\theta) = \sup_{\theta \in \Theta_0} \ell_n(\theta)$ for a countable dense $\Theta_0$, and measurability follows from Proposition 1.15. Papers never mention this. I mention it once, here, and then join them in silence — but when you meet an estimator defined by maximizing a *discontinuous* criterion (0–1 loss, say, or a rank statistic), the issue is real and the standard fix is a measurable-selection theorem.

The distinction between "estimator" and "estimate" is the distinction between $X$ and $x$ in the canonical notation, and confusing them is the single most common source of muddle in this subject. "The variance of the estimator" is a meaningful phrase; "the variance of the estimate" is not — an estimate is a number, and numbers do not have variances.

### 8.1.2 Bias, variance, mean squared error

> **Definition 8.2 (Bias, variance, MSE).**
> Let $\hat g_n$ estimate $g(\theta) \in \mathbb R$, and suppose $\mathbb E_\theta[\hat g_n^2] < \infty$ for all $\theta$ (all expectations are under $P_\theta^{\otimes n}$; the subscript $\theta$ records that). Define
> $$\operatorname{Bias}_\theta(\hat g_n) := \mathbb E_\theta[\hat g_n] - g(\theta), \qquad \operatorname{Var}_\theta(\hat g_n) := \mathbb E_\theta\big[(\hat g_n - \mathbb E_\theta \hat g_n)^2\big],$$
> $$\operatorname{MSE}_\theta(\hat g_n) := \mathbb E_\theta\big[(\hat g_n - g(\theta))^2\big].$$
> The estimator is **unbiased** if $\operatorname{Bias}_\theta(\hat g_n) = 0$ for every $\theta \in \Theta$.

Note the quantifier in "unbiased": *for every $\theta$*. An estimator that happens to be unbiased at one particular $\theta$ is not unbiased; it is lucky.

> **Proposition 8.3 (Bias–variance decomposition).**
> Under the conditions of Definition 8.2, for every $\theta$,
> $$\operatorname{MSE}_\theta(\hat g_n) = \operatorname{Bias}_\theta(\hat g_n)^2 + \operatorname{Var}_\theta(\hat g_n).$$
>
> *Proof.* Write $b := \mathbb E_\theta[\hat g_n] - g(\theta)$ and $D := \hat g_n - \mathbb E_\theta[\hat g_n]$, so that $\hat g_n - g(\theta) = D + b$ with $b$ a constant and $\mathbb E_\theta[D]=0$. Then $\mathbb E_\theta[(D+b)^2] = \mathbb E_\theta[D^2] + 2b\,\mathbb E_\theta[D] + b^2 = \operatorname{Var}_\theta(\hat g_n) + b^2$. All terms are finite by the square-integrability assumption. $\blacksquare$

This is the same computation as the bias–variance decomposition of Chapter 6, and for the same reason: both are Pythagoras in $L^2(P_\theta^{\otimes n})$, with the constant $b$ orthogonal to the mean-zero fluctuation $D$. The vector-valued version replaces MSE by $\mathbb E_\theta\|\hat g_n - g(\theta)\|^2$ and variance by the trace of the covariance matrix; the proof is unchanged.

Now for the first genuinely useful thing in the chapter, which is that Proposition 8.3 has *two* terms and unbiasedness only zeroes one of them.

**Worked Example 8.1 (Three estimators of a Gaussian variance).**
Let $X_1,\dots,X_n \stackrel{iid}\sim \mathcal N(\mu,\sigma^2)$ with both parameters unknown. Consider the family
$$S_c := \frac 1c \sum_{i=1}^n (X_i - \bar X_n)^2, \qquad \bar X_n := \frac 1n \sum_i X_i,$$
indexed by the divisor $c>0$. Three choices are traditional or interesting: $c=n$ (the MLE, as we will prove in §8.3), $c = n-1$ (the unbiased sample variance $S^2$), and $c = n+1$ (which will turn out to minimize MSE).

From Chapter 5 we know the sampling distribution: $\sum_i (X_i-\bar X_n)^2 = \sigma^2 W$ with $W \sim \chi^2_{n-1}$, independent of $\bar X_n$. Since $\mathbb E[W]=n-1$ and $\operatorname{Var}(W) = 2(n-1)$,
$$\mathbb E[S_c] = \frac{(n-1)\sigma^2}{c}, \qquad \operatorname{Bias}(S_c) = \sigma^2\Big(\frac{n-1}{c}-1\Big), \qquad \operatorname{Var}(S_c) = \frac{2(n-1)\sigma^4}{c^2},$$
and therefore, by Proposition 8.3,
$$\operatorname{MSE}(S_c) = \frac{\sigma^4}{c^2}\Big[2(n-1) + (c-n+1)^2\Big].$$
Differentiate the bracket-over-$c^2$ expression: writing $f(c) = [2(n-1)+(c-n+1)^2]/c^2$,
$$f'(c) = \frac{2(n-1)}{c^3}\,(c - n - 1),$$
so $f$ decreases on $(0,n+1)$ and increases after: the MSE-optimal divisor is $c^\star = n+1$, giving $\operatorname{MSE}(S_{n+1}) = 2\sigma^4/(n+1)$.

Here are the exact numbers, with $\sigma^2 = 1$ (all entries computed from the closed forms above; the simulation check follows):

| $n$ | divisor | bias | variance | MSE |
|---|---|---|---|---|
| 5 | $n$ | $-0.20000$ | $0.32000$ | $0.36000$ |
| 5 | $n-1$ | $0.00000$ | $0.50000$ | $0.50000$ |
| 5 | $n+1$ | $-0.33333$ | $0.22222$ | $\mathbf{0.33333}$ |
| 10 | $n$ | $-0.10000$ | $0.18000$ | $0.19000$ |
| 10 | $n-1$ | $0.00000$ | $0.22222$ | $0.22222$ |
| 10 | $n+1$ | $-0.18182$ | $0.14876$ | $\mathbf{0.18182}$ |
| 20 | $n$ | $-0.05000$ | $0.09500$ | $0.09750$ |
| 20 | $n-1$ | $0.00000$ | $0.10526$ | $0.10526$ |
| 20 | $n+1$ | $-0.09524$ | $0.08617$ | $\mathbf{0.09524}$ |
| 50 | $n$ | $-0.02000$ | $0.03920$ | $0.03960$ |
| 50 | $n-1$ | $0.00000$ | $0.04082$ | $0.04082$ |
| 50 | $n+1$ | $-0.03922$ | $0.03768$ | $\mathbf{0.03922}$ |

Monte Carlo check at $n=10$ with $N = 400{,}000$ replicates (NumPy `default_rng`, seed 20240801): $c=n$ gave mean $0.90081$, variance $0.18060$, MSE $0.19044$; $c=n-1$ gave mean $1.00091$, variance $0.22297$, MSE $0.22297$; $c=n+1$ gave mean $0.81892$, variance $0.14926$, MSE $0.18205$. These agree with the table to within Monte Carlo error.

Read the table. The unbiased estimator has the **largest** MSE of the three, at every sample size. Its bias is zero and it pays for that with variance. The MSE-optimal estimator has the largest bias of the three. And the gap does not vanish: at $n=50$ the unbiased estimator still carries about 4% more MSE than the optimal one. If your loss is squared error — and in machine learning it usually is — insisting on unbiasedness is choosing to lose.

One concrete sample, so the abstraction has a body. Ten draws from $\mathcal N(0,1)$ (seed 7, rounded to three decimals): $0.001, 0.299, -0.274, -0.891, -0.455, -0.992, 0.060, 1.340, -0.492, -0.620$. Here $\bar x = -0.2024$ and $\sum(x_i-\bar x)^2 = 4.1655$, so the three estimates are $0.4165$ (divisor 10), $0.4628$ (divisor 9), and $0.3787$ (divisor 11). The truth is $1$. All three are badly off, which is what $n=10$ buys you; the point of the table is that *on average over samples* one of them is off by less.

> **Common Confusion 8.1.**
> *"But surely unbiased is better? If my estimator is biased, I'm systematically wrong. That sounds bad."*
>
> Systematically wrong by a little is often better than unsystematically wrong by a lot. Unbiasedness says: if you repeated the experiment infinitely many times and averaged your estimates, you would hit the target. It says nothing about how far any single estimate is from the target, and a single estimate is what you actually have. Worse, unbiasedness is not preserved by nonlinear transformation: $S^2$ is unbiased for $\sigma^2$, but $\sqrt{S^2}$ is *not* unbiased for $\sigma$ (by Jensen, Ch. 4, it is biased low, since $\sqrt\cdot$ is concave). So "unbiased" is a property of a parameterization, not of an estimator's quality. And in §8.8 we will see that in dimension $\ge 3$ the natural unbiased estimator of a Gaussian mean is *dominated* — there is another estimator with strictly smaller MSE at **every** $\theta$. Unbiasedness is a mathematically convenient constraint that makes theorems provable (§8.5, §8.7). It is not a goal.

There are also settings where unbiasedness is exactly what you need, and it is worth being fair. In stochastic gradient descent the assumption $\mathbb E[g_t \mid \mathcal F_t] = \nabla F(\theta_t)$ — a conditional-expectation statement in the sense of Chapter 6, and Chapter 1's headline example of a filtration doing work — is what makes SGD converge to a stationary point of $F$ rather than of something else. There the estimand is a gradient that gets *averaged over many steps*, so the "repeat infinitely and average" story is literally what happens, and bias accumulates rather than cancels. Unbiasedness matters when you are averaging; it matters much less when you get one shot.

### 8.1.3 Large-sample criteria

Finite-sample MSE comparisons are the gold standard but are rarely computable outside toy models. The workaround is asymptotics: ask what happens as $n\to\infty$, using the machinery of Chapter 7.

> **Definition 8.4 (Consistency, asymptotic normality, efficiency).**
> Let $\hat\theta_n$ be a sequence of estimators of $\theta \in \Theta \subseteq \mathbb R^d$.
> 1. $\hat\theta_n$ is **(weakly) consistent** if $\hat\theta_n \to_p \theta$ under $P_\theta$ for every $\theta \in \Theta$; **strongly consistent** if $\hat\theta_n \to_{a.s.} \theta$.
> 2. $\hat\theta_n$ is **asymptotically normal** with asymptotic covariance $\Sigma(\theta)$ if $\sqrt n\,(\hat\theta_n - \theta) \to_d \mathcal N(0, \Sigma(\theta))$ under $P_\theta$, for every $\theta$.
> 3. Given two asymptotically normal estimators with asymptotic variances $\Sigma_1(\theta), \Sigma_2(\theta)$ (scalar case), the **asymptotic relative efficiency** of $\hat\theta^{(1)}$ to $\hat\theta^{(2)}$ is $\mathrm{ARE} = \Sigma_2(\theta)/\Sigma_1(\theta)$: the factor by which sample size can be reduced by using estimator 1 instead of 2 while achieving the same asymptotic variance.
> 4. $\hat\theta_n$ is **asymptotically efficient** if it is asymptotically normal with $\Sigma(\theta) = I(\theta)^{-1}$, the inverse Fisher information of §8.4.

Consistency is a minimal sanity requirement: with unlimited data you should find the truth. It is remarkably weak — it constrains nothing at any finite $n$ — and remarkably easy to fail in interesting ways, as §8.6 will show with the Neyman–Scott example. Asymptotic normality is the workhorse: it turns an estimator into a confidence interval and a hypothesis test (Chapter 9).

> **Proposition 8.5 ($L^2$ consistency implies consistency).**
> If $\operatorname{MSE}_\theta(\hat\theta_n)\to 0$ as $n\to\infty$, then $\hat\theta_n \to_p \theta$.
>
> *Proof.* For $\varepsilon > 0$, Markov's inequality (Ch. 4) applied to the nonnegative random variable $(\hat\theta_n-\theta)^2$ gives $P_\theta(|\hat\theta_n-\theta|\ge \varepsilon) \le \operatorname{MSE}_\theta(\hat\theta_n)/\varepsilon^2 \to 0$. $\blacksquare$

Equivalently, by Proposition 8.3, *bias $\to 0$ and variance $\to 0$* implies consistency. Both halves are needed: $\hat\theta_n = X_1$ has zero bias for a mean but constant variance and is not consistent; $\hat\theta_n = \bar X_n + 1$ has vanishing variance but constant bias and is not consistent either. The converse of Proposition 8.5 is false — an estimator can be consistent with infinite MSE at every $n$ (put a vanishing probability $1/n$ on a value of size $e^n$).

> **Self-check 8.1.**
> 1. Is $\hat\theta_n(X_1,\dots,X_n) := 7$, regardless of the data, an estimator in the sense of Definition 8.1? Is it unbiased? Is it consistent? What is its MSE?
> 2. In Worked Example 8.1, why did the optimal divisor come out as $n+1$ and not, say, $n+5$? Give the one-line reason in terms of the two competing terms.
> 3. $S^2$ is unbiased for $\sigma^2$. Is $S$ unbiased for $\sigma$? Which direction is the bias, and which chapter's inequality tells you?
> 4. Give an estimator that is unbiased at every $n$ but not consistent.

---

## 8.2 The method of moments

### 8.2.1 The idea in one sentence

The population has moments; the sample has moments; the law of large numbers says the second converge to the first; so set them equal and solve for $\theta$. That is the whole method, and it was Karl Pearson's, from the 1890s. Its virtue is that it needs no likelihood — only that some moments exist and that the map from parameters to moments is invertible. Its vice is that it throws away almost everything the data know.

> **Definition 8.6 (Method of moments estimator).**
> Let $\{P_\theta:\theta\in\Theta\subseteq\mathbb R^d\}$ be a family for which the first $d$ moments $m_k(\theta) := \mathbb E_\theta[X^k]$, $k=1,\dots,d$, exist and are finite. Let $\hat m_k := \frac1n\sum_{i=1}^n X_i^k$ be the sample moments. A **method-of-moments (MoM) estimator** is any $\hat\theta_n^{\mathrm{MoM}} \in \Theta$ solving the system
> $$m_k(\hat\theta_n^{\mathrm{MoM}}) = \hat m_k, \qquad k = 1,\dots,d,$$
> when a solution exists. More generally, for a vector of $d$ measurable functions $h_1,\dots,h_d$ with finite means, solve $\mathbb E_\theta[h_k(X)] = \frac1n\sum_i h_k(X_i)$.

Note that $\hat m_k$ is exactly the expectation of $x^k$ under the empirical distribution $\hat P_n = \frac1n\sum_i \delta_{X_i}$ from Chapter 1: MoM matches moments of $P_\theta$ to moments of $\hat P_n$.

> **Proposition 8.7 (Consistency and asymptotic normality of MoM).**
> Suppose $m_{2d}(\theta) < \infty$ (so the sample moments have finite variance), the map $M:\theta\mapsto(m_1(\theta),\dots,m_d(\theta))$ is injective on $\Theta$, continuously differentiable, and its inverse $M^{-1}$ is continuously differentiable at $M(\theta_0)$ with Jacobian $J = (\nabla M(\theta_0))^{-1}$. Then $\hat\theta^{\mathrm{MoM}}_n = M^{-1}(\hat m_1,\dots,\hat m_d)$ satisfies
> $$\hat\theta_n^{\mathrm{MoM}} \to_{a.s.} \theta_0, \qquad \sqrt n(\hat\theta_n^{\mathrm{MoM}} - \theta_0) \to_d \mathcal N\big(0,\; J\,\Sigma_m\,J^\top\big),$$
> where $\Sigma_m$ is the covariance matrix of the vector $(X, X^2,\dots,X^d)$ under $P_{\theta_0}$, i.e. $(\Sigma_m)_{jk} = m_{j+k}(\theta_0) - m_j(\theta_0)m_k(\theta_0)$.
>
> *Proof.* The SLLN (Ch. 7) applied coordinatewise gives $(\hat m_1,\dots,\hat m_d)\to_{a.s.} M(\theta_0)$; continuity of $M^{-1}$ and the continuous mapping theorem (Ch. 7) give almost sure convergence of $\hat\theta_n$. The multivariate CLT (Ch. 7) gives $\sqrt n\big((\hat m_k)_k - M(\theta_0)\big) \to_d \mathcal N(0,\Sigma_m)$, finite because $m_{2d}<\infty$. The delta method (Ch. 7) applied to $M^{-1}$ transfers this through, with Jacobian $\nabla M^{-1}(M(\theta_0)) = (\nabla M(\theta_0))^{-1} = J$ by the inverse function theorem. $\blacksquare$

Everything in that proof is a citation. That is the point of Chapter 7: once you have SLLN + CLT + delta method + Slutsky, a large class of estimators is handled mechanically.

### 8.2.2 Two examples and three failures

**Worked Example 8.2 (Method of moments for the Gamma and the Beta, and three ways it breaks).**

*Gamma.* Let $X\sim\mathrm{Gamma}(\alpha, s)$ with shape $\alpha$ and scale $s$, density $p(x) = x^{\alpha-1}e^{-x/s}/(\Gamma(\alpha)s^\alpha)$ on $(0,\infty)$ w.r.t. Lebesgue measure. Then $m_1 = \alpha s$ and $\operatorname{Var} = \alpha s^2$, so with $\hat v := \hat m_2 - \hat m_1^2$ (the sample variance with divisor $n$),
$$\hat\alpha^{\mathrm{MoM}} = \frac{\hat m_1^2}{\hat v}, \qquad \hat s^{\mathrm{MoM}} = \frac{\hat v}{\hat m_1}.$$
Closed form, no iteration, no special functions. Contrast the MLE, which requires solving $\log\hat\alpha - \psi(\hat\alpha) = \log \bar X - \overline{\log X}$ numerically, where $\psi$ is the digamma function. Numbers: $n=200$ draws from $\mathrm{Gamma}(\alpha=2, s=3)$ (seed 11) gave $\hat m_1 = 6.1981$, $\hat m_2 = 52.2511$, hence $\hat v = 13.8351$ and
$$\hat\alpha^{\mathrm{MoM}} = 2.7767,\quad \hat s^{\mathrm{MoM}} = 2.2322; \qquad \hat\alpha^{\mathrm{MLE}} = 2.3612,\quad \hat s^{\mathrm{MLE}} = 2.6250.$$
Both are in the neighborhood; the MLE is closer on this draw, and Lab B2 will show it is closer on average by a factor of about $1.74$ in MSE once $n\ge 50$.

*Beta.* For $X\sim\mathrm{Beta}(a,b)$, $\mu = a/(a+b)$ and $v = \mu(1-\mu)/(a+b+1)$. Solving, with $\kappa := \hat\mu(1-\hat\mu)/\hat v - 1$,
$$\hat a^{\mathrm{MoM}} = \kappa\hat\mu, \qquad \hat b^{\mathrm{MoM}} = \kappa(1-\hat\mu).$$
On the five numbers $0.12, 0.25, 0.31, 0.09, 0.43$: $\hat\mu = 0.2400$, $\hat v = 0.015600$, $\kappa = 10.6923$, so $\hat a = 2.5662$, $\hat b = 8.1262$. On $n=100$ draws from $\mathrm{Beta}(2,5)$ (seed 12) MoM gives $(3.0415, 7.5817)$ against the MLE's $(2.6081, 6.5929)$.

A pleasant surprise here, worth pausing on because the obvious worry turns out to be groundless: $\hat a$ and $\hat b$ are *always* positive, so the Beta MoM estimate can never fall outside the parameter space. Reason: for $x_i\in(0,1)$ we have $x_i^2 < x_i$, hence
$$\hat v = \tfrac1n\textstyle\sum_i x_i^2 - \hat\mu^2 \;<\; \hat\mu - \hat\mu^2 = \hat\mu(1-\hat\mu),$$
so $\kappa > 0$ identically — with one edge case worth naming, in the spirit of Chapter 1's "probability zero is not impossible": if every observation is *equal*, then $\hat v = 0$ and $\kappa$ is undefined (division by zero), so the estimator fails to exist rather than turning negative. Under the Beta model this event has probability zero; on real data rounded to two decimals it does not, and code should check for it. (A numerical stress test agrees with the strict inequality: over $200{,}000$ samples of size $2$–$7$ from the extremely U-shaped $\mathrm{Beta}(0.1,0.1)$, seed 0, the quantity $\hat\mu(1-\hat\mu)-\hat v$ was never negative.) This is the kind of claim that is easy to assert and easy to check, and checking is cheap.

Now the failures.

*Failure 1: the moments may not exist.* For a Cauchy location family, $\mathbb E|X| = \infty$, so $\hat m_1$ does not converge to anything (it is itself Cauchy, by the stability property from Chapter 7). MoM is not merely inefficient here; it does not exist. Any estimator built on sample means inherits this.

*Failure 2: the solution may fall outside $\Theta$.* The Beta was lucky; other families are not. Take the negative binomial with mean $\mu$ and variance $\mu + \mu^2/r$, a standard model for overdispersed counts. Matching the first two moments gives
$$\hat\mu = \bar X, \qquad \hat r = \frac{\bar X^2}{\hat v - \bar X},$$
which is negative — meaningless — whenever the sample happens to be *under*dispersed, $\hat v \le \bar X$. That is not rare. For $r=8$, $\mu=3$ (so the true variance is $4.1250$), simulating $400{,}000$ samples (seed 7):
$$P(\hat v \le \bar X) = 0.4167 \ (n=10), \quad 0.2189\ (n=25), \quad 0.1012\ (n=50).$$
At $n=10$ the method returns nonsense two times in five. A concrete instance (seed 101): the sample $0,3,4,4,5,5,2,3,3,0$ has $\bar x = 2.9000$ and $\hat v = 2.8900$, giving $\hat r = -841.0$. The MLE, which maximizes a likelihood that is $-\infty$ outside $\Theta$, cannot do this.

*Failure 3: the solution may be absurd even when it is in $\Theta$.* Take $X_i \sim \mathrm{Unif}[0,\theta]$. Since $\mathbb E X = \theta/2$, MoM gives $\hat\theta^{\mathrm{MoM}} = 2\bar X_n$. But $\bar X_n$ can easily be small enough that $2\bar X_n < \max_i X_i$, and then the estimate assigns probability zero to data you actually observed. Concretely, six draws from $\mathrm{Unif}[0,1]$ (seed 3): $0.0856, 0.2368, 0.8013, 0.5822, 0.0941, 0.4331$, giving $2\bar x = 0.7444 < 0.8013 = \max_i x_i$. This is not a freak event: simulating $N=200{,}000$ samples of size $6$ (seed 99), $P(2\bar X_n < \max_i X_i) = 0.2254$. Roughly one time in four, MoM returns a parameter under which the observed sample was impossible. Maximum likelihood, by construction, can never do this.

> **Common Confusion 8.2.**
> *"MoM and MLE both use the data, so they should give nearly the same answer — the difference is just numerical."*
>
> They use *different functionals of the data*. MoM uses $\bar X$ and $\overline{X^2}$; the Gamma MLE uses $\bar X$ and $\overline{\log X}$. Those are different statistics with different variances, and for the Gamma, $\overline{\log X}$ is (jointly with $\bar X$) sufficient while $\overline{X^2}$ is not — sufficiency in the sense of Chapter 3's factorization theorem. Discarding a sufficient statistic in favor of a non-sufficient one loses information, permanently, and §8.7 will quantify exactly how much. For the Gamma shape at $\alpha = 2$ the asymptotic variances are computable in closed form: $n\operatorname{Var}(\hat\alpha^{\mathrm{MLE}})\to \alpha/(\alpha\psi'(\alpha)-1) = 6.8997$ (with $\psi'(2) = \pi^2/6 - 1$) and, by the delta method, $n\operatorname{Var}(\hat\alpha^{\mathrm{MoM}})\to 12.0000$ exactly. The ARE of MoM to MLE is therefore $0.5750$: you need about $74\%$ more data to do as well with MoM. Lab B2 reproduces the ratio $1.74$ by simulation.

### 8.2.3 Generalized method of moments, in one paragraph

If you have *more* moment conditions than parameters — $q > d$ functions $h_k$ with $\mathbb E_\theta[h_k(X)]=0$ at the true $\theta$ — the system is overdetermined and generally has no exact solution. The generalized method of moments (GMM) replaces "solve" with "minimize": $\hat\theta = \arg\min_\theta \bar h(\theta)^\top W \bar h(\theta)$, where $\bar h(\theta) := \frac1n\sum_i h(X_i,\theta)$ and $W$ is a positive definite weight matrix, optimally $W = \operatorname{Var}(h(X,\theta))^{-1}$. GMM is the backbone of empirical economics and the direct ancestor of several machine learning objectives: maximum mean discrepancy matches an infinite family of moments indexed by an RKHS; feature matching in GANs matches the mean of a discriminator's intermediate activations; and the "moment matching" family of variational objectives (expectation propagation) is GMM with a specific choice of $h$. The asymptotic theory is Proposition 8.7 with an extra projection; Hansen (1982) is the reference.

> **Self-check 8.2.**
> 1. Why does MoM for $\mathrm{Unif}[0,\theta]$ sometimes produce an estimate under which the data have probability zero, while the MLE cannot?
> 2. The MoM estimator of the Gamma shape is $\bar X^2/\hat v$. Which theorem from Chapter 7 tells you it is asymptotically normal, and what is the role of the delta method?
> 3. Give a one-parameter family where MoM using $\mathbb E[X]$ fails but MoM using $\mathbb E[|X|]$ or a median-type condition works.
> 4. If MoM and MLE are both consistent, why prefer MLE?

---

## 8.3 Maximum likelihood

### 8.3.1 Why the base measure is not a technicality

Everyone remembers the slogan: choose the parameter that makes the observed data most probable. For discrete data that is literally correct. For continuous data it is literally false, because every particular observed value has probability zero (Chapter 1, Failure 1). The repair every textbook makes is "use the density instead of the probability", and the repair is right, but it hides the question Chapter 2 forced us to confront: *density with respect to what?*

Here is the honest definition.

> **Definition 8.8 (Likelihood, log-likelihood, MLE).**
> Let $\{P_\theta : \theta\in\Theta\}$ be a family of probability measures on $(\mathcal X,\mathcal A)$, and suppose there is a single $\sigma$-finite measure $\mu$ on $(\mathcal X,\mathcal A)$ with $P_\theta \ll \mu$ for all $\theta$ (the family is **dominated** by $\mu$). By Radon–Nikodym (Ch. 2) each $P_\theta$ has a density $p_\theta := \frac{dP_\theta}{d\mu}$, determined $\mu$-a.e.; fix one version for each $\theta$. Given data $x = (x_1,\dots,x_n)$, the **likelihood function** is
> $$L_n(\theta) := \prod_{i=1}^n p_\theta(x_i), \qquad \ell_n(\theta) := \log L_n(\theta) = \sum_{i=1}^n \log p_\theta(x_i),$$
> the density of $P_\theta^{\otimes n}$ w.r.t. $\mu^{\otimes n}$ evaluated at the data. A **maximum likelihood estimator** is any measurable $\hat\theta_n$ with $\ell_n(\hat\theta_n) = \sup_{\theta\in\Theta}\ell_n(\theta)$.

Three consequences of stating it this way.

**(a) The choice of $\mu$ does not matter, as long as it is the same $\mu$ for all $\theta$.** If $\nu$ is another $\sigma$-finite measure with $P_\theta\ll\mu\ll\nu$, then (by the chain rule for Radon–Nikodym derivatives, Ch. 2) $\frac{dP_\theta}{d\nu} = \frac{dP_\theta}{d\mu}\cdot\frac{d\mu}{d\nu}$, and the extra factor $\frac{d\mu}{d\nu}(x_i)$ does not depend on $\theta$. (If neither of $\mu,\nu$ dominates the other, compare both to $\mu+\nu$, which dominates each.) It multiplies $L_n$ by a positive constant and shifts $\ell_n$ by an additive constant, changing neither the argmax nor likelihood *ratios*. The likelihood function is only ever defined up to a $\theta$-free positive factor, and every legitimate use of it respects that.

**(b) But the $\mu$ must be shared.** If you were allowed to pick a different base measure for each $\theta$, you could make any $\theta$ win. This is exactly the sin committed by the "spike" that blows up mixture likelihoods (§8.3.4), viewed the right way, and it is why $\theta$-dependent supports are so dangerous.

**(c) Mixed distributions are handled, with no new ideas.** This closes Failure 2 from Chapter 1 in the estimation setting.

**Worked Example 8.3 (MLE for the Bernoulli — the discrete warm-up).**
$\mathcal X = \{0,1\}$, $\mu = \#$ counting measure, $p_\theta(x) = \theta^x(1-\theta)^{1-x}$, $\theta\in[0,1]$. With $k := \sum_i x_i$,
$$\ell_n(\theta) = k\log\theta + (n-k)\log(1-\theta), \qquad \ell_n'(\theta) = \frac k\theta - \frac{n-k}{1-\theta}.$$
Setting this to zero gives $\hat\theta = k/n$. Check it is a maximum: $\ell_n''(\theta) = -k/\theta^2 - (n-k)/(1-\theta)^2 < 0$ for $0<k<n$, so $\ell_n$ is strictly concave and the stationary point is the global max. If $k=0$ the log-likelihood is $n\log(1-\theta)$, decreasing, so $\hat\theta = 0$ — a boundary solution where the derivative never vanishes. Likewise $k=n$ gives $\hat\theta=1$. Already in the simplest possible model, "set the derivative to zero" is not the definition of the MLE; "maximize" is.

This is also the first place to say what all of machine learning says in a different accent: $-\ell_n(\theta)/n$ for the Bernoulli model **is** the binary cross-entropy loss. Training a classifier with cross-entropy is maximum likelihood under a conditional Bernoulli (or categorical) model. Nothing else. Every property in this chapter is therefore a property of your training objective.

**Worked Example 8.4 (MLE for the Gaussian, both parameters).**
$\mu = \lambda$ Lebesgue on $\mathbb R$, $\theta = (m, v)$ with $v = \sigma^2 > 0$:
$$\ell_n(m,v) = -\frac n2\log(2\pi) - \frac n2 \log v - \frac{1}{2v}\sum_{i=1}^n (x_i-m)^2.$$
$\partial_m \ell_n = \frac1v\sum_i (x_i - m) = 0 \Rightarrow \hat m = \bar x$, for any $v>0$. Substituting and differentiating in $v$: $\partial_v \ell_n = -\frac{n}{2v} + \frac{1}{2v^2}\sum_i (x_i-\bar x)^2 = 0 \Rightarrow \hat v = \frac1n\sum_i (x_i-\bar x)^2$. The second derivative in $v$ at $\hat v$ is $-n/(2\hat v^2) < 0$; as $v\to0^+$ or $v\to\infty$, $\ell_n\to-\infty$ (provided not all $x_i$ are equal), so this is the global maximum.

So the Gaussian MLE for the variance is the divisor-$n$ estimator, which Worked Example 8.1 showed is biased low by a factor $(n-1)/n$. **MLE is not unbiased**, and it is not trying to be. On the eight numbers $2.1, -0.4, 1.3, 0.7, -1.2, 0.9, 2.4, 0.1$: $\hat m = 0.7375$, $\hat v = 1.3023$ (the unbiased $s^2$ is $1.4884$), and the attained maximum log-likelihood is $\ell_n(\hat\theta) = -12.4082$.

**Worked Example 8.5 (Uniform on $[0,\theta]$: the boundary case).**
$p_\theta(x) = \theta^{-1}\mathbf 1_{[0,\theta]}(x)$ w.r.t. Lebesgue. Then
$$L_n(\theta) = \theta^{-n}\prod_i \mathbf 1_{\{x_i\le\theta\}} = \theta^{-n}\mathbf 1_{\{\theta \ge x_{(n)}\}}, \qquad x_{(n)} := \max_i x_i.$$
This is zero for $\theta < x_{(n)}$ and strictly decreasing thereafter, so $\hat\theta^{\mathrm{MLE}} = x_{(n)}$. The log-likelihood is *not differentiable* at the maximum — indeed it jumps from $-\infty$ to its maximum there — so no score equation is available, and every asymptotic result in §8.4–§8.6 fails for this model. Chapter 1 flagged this family; here is why it is the standard counterexample: the support of $P_\theta$ depends on $\theta$, so the family is dominated by Lebesgue measure but the densities' supports move with $\theta$, and interchanging $\partial_\theta$ with $\int$ is illegal.

Finite-sample facts, all elementary from $P(X_{(n)}\le t) = (t/\theta)^n$: $\mathbb E[X_{(n)}] = \frac{n}{n+1}\theta$ (bias $-\theta/(n+1)$), $\operatorname{Var}(X_{(n)}) = \frac{n\theta^2}{(n+1)^2(n+2)}$, and $\operatorname{MSE}(X_{(n)}) = \frac{2\theta^2}{(n+1)(n+2)}$. The bias-corrected $\frac{n+1}{n}X_{(n)}$ is unbiased with MSE $\frac{\theta^2}{n(n+2)}$ — smaller for all $n\ge1$, so unlike Worked Example 8.1 removing the bias *does* help here. But it does not help as much as possible: minimizing $\operatorname{MSE}(cX_{(n)}) = \theta^2\big(\frac{n}{n+2}c^2 - \frac{2n}{n+1}c+1\big)$ over $c$ (Exercise A2) gives
$$c^\star = \frac{n+2}{n+1}, \qquad \operatorname{MSE}(c^\star X_{(n)}) = \frac{\theta^2}{(n+1)^2},$$
and since $\frac{n+2}{n+1} < \frac{n+1}{n}$, the optimum sits strictly *between* the MLE and the unbiased estimator — still biased low. At $n=5$ the three MSE coefficients are $0.047619$ (MLE), $0.028571$ (unbiased) and $0.027778$ (optimal). So the chapter's thesis survives intact: the best estimator is biased in both of our examples; only the direction of the correction changed. On the data $0.31, 0.88, 0.42, 0.71, 0.55$ ($n=5$): $\hat\theta^{\mathrm{MLE}} = 0.88$, the unbiased version is $1.0560$; MSE coefficients are $0.047619\,\theta^2$ and $0.028571\,\theta^2$ respectively. Note also the rate: the error is $O_P(1/n)$, not $O_P(1/\sqrt n)$, and $n(\theta - X_{(n)})/\theta \to_d \mathrm{Exp}(1)$ — not a normal limit. Everything about this model is non-standard, and that is precisely its pedagogical value.

**Worked Example 8.6 (Laplace: the MLE is the median, and it can be non-unique).**
$p_{m}(x) = \frac{1}{2b}e^{-|x-m|/b}$. For fixed $b$, maximizing $\ell_n$ means minimizing $\sum_i |x_i - m|$. The function $m\mapsto\sum_i|x_i-m|$ is convex and piecewise linear with slope $\#\{i: x_i<m\} - \#\{i: x_i>m\}$, which changes sign at the median. For odd $n$ the minimizer is the unique middle order statistic; for even $n$ **every point in $[x_{(n/2)}, x_{(n/2+1)}]$ is a maximizer**, so the MLE is a set, not a point. Definition 8.8 said "any measurable $\hat\theta_n$" for exactly this reason.

On the eight Gaussian numbers above, sorted: $-1.2, -0.4, 0.1, 0.7, 0.9, 1.3, 2.1, 2.4$. Every $m\in[0.7,0.9]$ maximizes; a grid search over $4001$ points in $[-1.5,2.5]$ returns the leftmost, $0.7000$. Taking $\hat m = 0.8$ (the conventional midpoint) gives $\hat b = \frac1n\sum_i|x_i-\hat m| = 0.9375$ and $\ell_n = -13.0289$, below the Gaussian model's $-12.4082$ — as it should be, since the data were generated from a Gaussian. Comparing maximized log-likelihoods across *different* models is the seed of model selection (AIC, BIC) and of the likelihood-ratio test in Chapter 9.

The ML content: least-squares regression is Gaussian MLE, and least-absolute-deviation regression is Laplace MLE. "Use L1 loss for robustness" is the same statement as "assume heavier-tailed noise".

**Worked Example 8.7 (A mixed distribution: MLE with a two-piece base measure).**
Chapter 1's Failure 2 was $A = \max(0,Z)$, which has neither a pmf nor a pdf. Estimation in such models is routine once the base measure is right. Model: $A = \max(0, Z + m)$ with $Z\sim\mathcal N(0,1)$ and unknown $m$ — a *censored* or *Tobit* model, the exact structure of "activation after a ReLU", of survival data with a study end date, and of any sensor with a floor.

Take $\mu := \delta_0 + \lambda|_{(0,\infty)}$ on $[0,\infty)$: counting measure on the single atom $\{0\}$ plus Lebesgue measure on the open half-line. This $\mu$ is $\sigma$-finite and dominates $P_m$ for every $m$. The density is
$$p_m(a) = \begin{cases} \Phi(-m), & a = 0,\\[2pt] \phi(a-m), & a>0,\end{cases}$$
where the top line is a genuine probability (the mass of the atom) and the bottom is a genuine Lebesgue density, glued into one function by the two-piece $\mu$. Verify: $\int p_m\,d\mu = \Phi(-m)\cdot 1 + \int_0^\infty \phi(a-m)\,da = \Phi(-m) + \Phi(m) = 1$. Good. Then, with $Z_0 := \{i : a_i = 0\}$,
$$\ell_n(m) = |Z_0|\log\Phi(-m) + \sum_{i\notin Z_0} \log\phi(a_i - m),$$
which is smooth in $m$ and can be maximized by any optimizer. With $n=200$ draws at true $m = 0.5$ (seed 2024) there were $61$ zeros, and numerical maximization gave $\hat m = 0.4765$. The two naive alternatives are badly biased: the mean of all the $A_i$ is $0.6740$ (biased *up*, because censoring at zero truncates the left tail), and the mean of the positive values only is $0.9697$ (much worse — selecting on $A>0$ selects on $Z > -m$).

That is Failure 2 closed. A single object — the Radon–Nikodym derivative with respect to a stated $\mu$ — is valid in the discrete case, the continuous case, and the mixed case, and MLE is defined identically in all three.

### 8.3.2 Invariance, and the exponential-family shortcut

> **Proposition 8.9 (Equivariance of the MLE under reparameterization).**
> Let $\psi = g(\theta)$ for a function $g:\Theta\to\Psi$. Define the induced likelihood $L^*_n(\psi) := \sup\{L_n(\theta) : g(\theta) = \psi\}$ (the *profile* likelihood; if $g$ is a bijection this is just $L_n(g^{-1}(\psi))$). If $\hat\theta_n$ maximizes $L_n$, then $\hat\psi_n := g(\hat\theta_n)$ maximizes $L^*_n$.
>
> *Proof.* For any $\psi\in\Psi$, $L^*_n(\psi) = \sup_{\theta\in g^{-1}(\psi)} L_n(\theta) \le \sup_{\theta\in\Theta}L_n(\theta) = L_n(\hat\theta_n)$. Also $L^*_n(g(\hat\theta_n)) = \sup_{\theta \in g^{-1}(g(\hat\theta_n))} L_n(\theta) \ge L_n(\hat\theta_n)$. Combining, $L^*_n(g(\hat\theta_n)) = L_n(\hat\theta_n) \ge L^*_n(\psi)$ for all $\psi$. $\blacksquare$

In words: **the MLE of a function is the function of the MLE.** If $\hat\sigma^2$ is the MLE of the variance then $\hat\sigma = \sqrt{\hat\sigma^2}$ is the MLE of the standard deviation; if $\hat p$ is the MLE of a Bernoulli parameter then $\log(\hat p/(1-\hat p))$ is the MLE of the log-odds. This is a genuinely valuable property. It means MLE does not care whether you parameterize a network's output layer by a probability, a logit, or a temperature-scaled logit; the fitted predictive distribution is the same object. In §8.8 we will find that MAP does **not** have this property, and the asymmetry is worth taking seriously.

Contrast unbiasedness, which is destroyed by nonlinear $g$ (Common Confusion 8.1). Equivariance and unbiasedness are pulling in opposite directions, and MLE picks equivariance.

> **Proposition 8.10 (MLE in an exponential family is moment matching).**
> Let $p_\eta(x) = h(x)\exp(\eta^\top T(x) - A(\eta))$ w.r.t. $\mu$, a full-rank exponential family in natural parameterization with natural parameter space $\mathcal E$ open (Ch. 3). Then $\ell_n(\eta) = \sum_i \eta^\top T(x_i) - nA(\eta) + \text{const}$ is strictly concave on $\mathcal E$, and $\hat\eta$ solves
> $$\nabla A(\hat\eta) = \frac1n\sum_{i=1}^n T(x_i), \quad\text{i.e.}\quad \mathbb E_{\hat\eta}[T(X)] = \overline{T}_n.$$
> If a solution exists in $\mathcal E$ it is the unique MLE.
>
> *Proof.* From Chapter 3, $\nabla A(\eta) = \mathbb E_\eta[T(X)]$ and $\nabla^2 A(\eta) = \operatorname{Var}_\eta(T(X)) \succ 0$ by full rank. Hence $\nabla^2 \ell_n(\eta) = -n\nabla^2A(\eta) \prec 0$: $\ell_n$ is strictly concave, so a stationary point is the unique global maximum, and $\nabla\ell_n(\eta) = \sum_i T(x_i) - n\nabla A(\eta) = 0$ is the stated equation. $\blacksquare$

So for exponential families, MLE *is* method of moments — for the sufficient statistics specifically, not for arbitrary moments. Bernoulli: $T(x)=x$, match $\bar x$. Gaussian: $T(x) = (x,x^2)$, match the first two moments. Poisson: $T(x)=x$, $\hat\lambda = \bar x$. This is why the two methods agree for the Gaussian and disagree for the Gamma-with-unknown-shape (whose sufficient statistic is $(\sum x_i, \sum\log x_i)$, and $\overline{\log x}$ is not a power moment).

> **Common Confusion 8.3.**
> *"The likelihood is the probability of $\theta$ given the data — that's why we maximize it."*
>
> No. $L_n(\theta) = p_\theta(x)$ is a function of $\theta$ obtained by freezing $x$ in a *density over $x$*. It is not a density over $\theta$: $\int_\Theta L_n(\theta)\,d\theta$ is typically not $1$ and can be $\infty$. Nothing in Definition 8.8 puts a measure on $\Theta$. The object that *is* a distribution over $\theta$ is the posterior $p(\theta\mid x)\propto L_n(\theta)\pi(\theta)$ from Proposition 1.12, and getting it requires a prior. Relatedly: likelihood *values* are not probabilities and can exceed 1 whenever the base measure is continuous. Chapter 1's Failure 1 again — a density of $10^{6}$ at a point is perfectly ordinary if the distribution is concentrated in a region of width $10^{-6}$. When you see a reported "likelihood" of $e^{4000}$ in a paper on normalizing flows, nothing is wrong; the units are nats per unit volume, and the volume element came from the base measure.

### 8.3.3 Why maximum likelihood? The KL answer

Maximum likelihood is often presented as a plausible heuristic. It is better than that. Divide the log-likelihood by $n$:
$$\frac1n \ell_n(\theta) = \frac 1n \sum_{i=1}^n \log p_\theta(X_i) = \mathbb E_{\hat P_n}[\log p_\theta(X)],$$
an expectation under the empirical distribution. By the SLLN (Ch. 7), for each fixed $\theta$ this converges a.s. to $\mathbb E_{P^\star}[\log p_\theta(X)]$, where $P^\star$ is the true data distribution (with density $p^\star$ w.r.t. $\mu$, assuming $P^\star\ll\mu$). And
$$\mathbb E_{P^\star}[\log p_\theta(X)] = \mathbb E_{P^\star}[\log p^\star(X)] - D_{\mathrm{KL}}(P^\star \,\|\, P_\theta),$$
where the first term does not involve $\theta$. Therefore:

> **Maximizing the likelihood is minimizing an empirical estimate of $D_{\mathrm{KL}}(P^\star\|P_\theta)$.**

Three things follow immediately, and they organize the rest of the chapter.

*If the model is correct* — $P^\star = P_{\theta_0}$ for some $\theta_0\in\Theta$ — the KL is minimized (at zero) exactly at $\theta_0$, uniquely if the model is identifiable, since $D_{\mathrm{KL}}(P\|Q)=0 \iff P=Q$ (Ch. 4, via Jensen). That is the seed of the consistency proof in §8.6.

*If the model is wrong*, the MLE still converges — to the $\theta^\star$ minimizing $D_{\mathrm{KL}}(P^\star\|P_\theta)$ over $\Theta$, the **KL projection** of the truth onto the model. This is a completely explicit description of what a misspecified model learns, and it is the honest answer to "what is my language model estimating?" (Not the truth. The reverse-information-projection of the truth onto its own hypothesis class.)

*The asymmetry of KL matters.* MLE minimizes $D_{\mathrm{KL}}(P^\star\|P_\theta)$ — the *forward* KL, expectation under the truth. Forward KL is mode-covering: it punishes $p_\theta(x)\approx 0$ where $p^\star(x)>0$ infinitely hard, and barely punishes $p_\theta(x)>0$ where $p^\star(x)\approx0$. That is why maximum-likelihood generative models blur and hallucinate rather than dropping modes, and why variational inference (Chapter 13), which minimizes the *reverse* KL $D_{\mathrm{KL}}(q\|p)$, does the opposite and collapses onto one mode. Same divergence, opposite argument order, opposite pathology.

**Worked Example 8.8 (What a misspecified Gaussian fit converges to).**
Fit $\mathcal N(m,v)$ to data from $\mathrm{Gamma}(2,3)$. Minimizing $D_{\mathrm{KL}}(P^\star\|\mathcal N(m,v))$ over $(m,v)$: the objective is $-\mathbb E_{P^\star}[\log p_{m,v}] + \text{const} = \frac12\log v + \frac{1}{2v}\mathbb E_{P^\star}[(X-m)^2] + \text{const}$, minimized at $m^\star = \mathbb E_{P^\star}[X] = 6$ and $v^\star = \operatorname{Var}_{P^\star}(X)=18$. Simulation (seed 66): $n=10^2$ gave $(5.8409,\,18.9039)$; $n=10^4$ gave $(6.0118,\,18.2671)$; $n=10^6$ gave $(6.0076,\,18.0599)$. The residual divergence at the optimum, by Monte Carlo with $N = 4{,}000{,}000$ (seed 77), is $D_{\mathrm{KL}} = 0.1888$ nats — the irreducible cost of using a symmetric model for a skewed truth. No amount of data reduces it.

### 8.3.4 Where maximum likelihood breaks

MLE is not a universal solvent, and the ways it fails are the ways your training runs fail.

**Non-existence.** The supremum may not be attained. For $\mathrm{Unif}[0,\theta]$ with the *open* interval $(0,\theta)$ and $\Theta = (0,\infty)$, $L_n(\theta) = \theta^{-n}\mathbf 1_{\{\theta > x_{(n)}\}}$ has no maximizer — the supremum is approached as $\theta\downarrow x_{(n)}$ but never attained. Half-open conventions save this example; they do not save the general phenomenon (e.g. a two-parameter family whose supremum sits at an infinite boundary).

**Non-uniqueness.** Worked Example 8.6 gave a continuum of MLEs for the Laplace median. Mixture models give a *combinatorial* non-uniqueness: relabeling the components of a $K$-mixture leaves the likelihood exactly unchanged, so every MLE comes in a family of $K!$ copies — **label switching**. This is a failure of *identifiability*: the map $\theta\mapsto P_\theta$ is not injective. Since MLE identifies $P_\theta$, not $\theta$, this is harmless for prediction and fatal for interpretation. Neural networks are wildly non-identifiable in exactly this way (permute hidden units, rescale a ReLU layer up and the next down), which is why "the weights mean something" is a claim needing evidence.

Mixtures also have *genuinely distinct* local optima, not just relabelings. Fitting a two-component Gaussian mixture by EM to $n=120$ points drawn from three well-separated clusters at $-3, 0, 3$ (sd $0.6$, seed 24601) converges to $\ell_n = -264.4678$ from one starting point and to $\ell_n = -262.4326$ from another — a gap of $2.04$ nats between two solutions that group the three clusters differently (left alone versus right alone). Neither is a relabeling of the other, and only the second is the MLE.

And non-uniqueness is not the worst of it: the likelihood of a perfectly ordinary, identifiable, one-parameter family can have several local maxima, with the "obvious" estimator sitting on the wrong one.

**Worked Example 8.9 (A multimodal likelihood in a regular one-parameter family).**
Let $X_1,\dots,X_n\stackrel{iid}\sim\mathrm{Cauchy}(\theta,1)$, density $p_\theta(x)=\big(\pi(1+(x-\theta)^2)\big)^{-1}$ w.r.t. Lebesgue. This family is as well behaved as it gets: one parameter, $\theta$-free support, densities smooth and bounded in every derivative, fully identifiable, and (as §8.4 will confirm) $I(\theta)=1/2$ exactly. Yet
$$\ell_n(\theta) = -\sum_{i=1}^n \log\big(1+(x_i-\theta)^2\big) - n\log\pi$$
is a sum of bumps, one centred on each observation, and when the data are spread out the bumps do not merge into a single peak.

Take the five observations $-6, -5, 0, 5, 6$. Numerical maximization on each of three brackets gives

| stationary point | $\theta$ | $\ell_5(\theta)$ | type |
|---|---|---|---|
| left | $-5.1225$ | $-19.0794$ | local max (global, tied) |
| centre | $0.0000$ | $-19.4617$ | local max |
| right | $+5.1225$ | $-19.0794$ | local max (global, tied) |
| between left and centre | $-2.4417$ | $-20.6120$ | local min |
| between centre and right | $+2.4417$ | $-20.6120$ | local min |

Three local maxima and two local minima, from five data points and one parameter. Read off the lessons. (i) The sample median and the sample mean are both $0$, and $\theta=0$ is a local maximum — but it is *not* the MLE; it is worse than $\pm5.1225$ by $0.38$ nats. Any Newton or gradient method started near the median converges to the wrong stationary point and reports it, with a perfectly respectable-looking negative second derivative. (ii) The global maximum is attained at *two* points here, because these data happen to be symmetric; a tiny perturbation breaks the tie but leaves both local maxima in place. (iii) None of this is caused by mixtures, latent variables, non-identifiability, or overparameterization. It is caused by heavy tails: an observation far from $\theta$ contributes almost nothing to $\ell_n$ (the log of a quadratic grows slowly), so the likelihood cannot decide between "the truth is near the left cluster and the right points are outliers" and the mirror-image story.

The practical rule this justifies: for any non-concave likelihood, *multi-start*. It is known that the Cauchy log-likelihood can have as many as $n$ local maxima, and the number does not decrease with better software. Figure 8.1(b) plots this exact curve.

**Unbounded likelihood.** This one is serious, and it is where Chapter 2's picture of mass escaping vertically becomes an algorithm's problem.

> **Proposition 8.11 (The likelihood of a Gaussian mixture is unbounded).**
> Let $\Theta = \{(w,m_1,v_1,m_2,v_2) : w\in(0,1), m_j\in\mathbb R, v_j > 0\}$ with
> $$p_\theta(x) = w\,\phi_{m_1,v_1}(x) + (1-w)\phi_{m_2,v_2}(x)$$
> w.r.t. Lebesgue measure, $\phi_{m,v}$ the $\mathcal N(m,v)$ density. Then for any sample $x_1,\dots,x_n$ with $n\ge2$, $\sup_{\theta\in\Theta} \ell_n(\theta) = +\infty$, so no MLE exists.
>
> *Proof.* Fix $m_1 := x_1$, $w := 1/2$, and any $(m_2,v_2)$ with $v_2>0$; let $v_1 = \varepsilon^2 \downarrow 0$. For $i\ge2$,
> $p_\theta(x_i) \ge \frac12 \phi_{m_2,v_2}(x_i) > 0$, a bound independent of $\varepsilon$. For $i=1$,
> $p_\theta(x_1) \ge \frac12\phi_{x_1,\varepsilon^2}(x_1) = \frac{1}{2\varepsilon\sqrt{2\pi}} \to\infty.$
> Hence $\ell_n(\theta) \ge -\log(2\varepsilon\sqrt{2\pi}) + \sum_{i\ge2}\log\big(\tfrac12\phi_{m_2,v_2}(x_i)\big) \to +\infty$ as $\varepsilon\downarrow0$. $\blacksquare$

Numerically, with data $-1.2, 0.3, 0.8, 1.5, 2.2$, $m_1 = x_1 = -1.2$, $w = 1/2$, $(m_2,v_2) = (1.2, 1)$:

| $\sigma_1$ | $1.0$ | $0.5$ | $0.1$ | $0.01$ | $0.001$ |
|---|---|---|---|---|---|
| $\ell_5(\theta)$ | $-8.4704$ | $-8.3361$ | $-6.7822$ | $-4.4847$ | $-2.1826$ |

Every factor of $10$ in $\sigma_1$ buys about $2.30 = \log 10$ nats, forever. The "optimum" is a Dirac spike on one data point: a degenerate distribution that explains one observation perfectly and everything else by the other component. In Chapter 2's language, the density is escaping vertically while its mass concentrates on a Lebesgue-null set; the limit is not a probability density at all, but $\delta_{x_1}$, which has no Lebesgue density. The likelihood is a supremum over a non-compact set and the sup lives on the boundary, outside the model.

What people do about it: constrain $v_j \ge v_{\min}$; add a prior on $v_j$ (an inverse-Gamma, giving MAP instead of MLE — §8.8); tie the variances across components; or, in practice, run EM from several random starts and quietly discard runs that collapse. §8.9 shows a collapsed run with numbers.

**Inconsistency.** MLE can be consistent-looking and simply wrong when the number of parameters grows with $n$; §8.6 gives the Neyman–Scott example, where the MLE of a variance converges to *half* the truth. Since deep learning is the regime where parameters outnumber data, this is not an academic worry, and §8.10 returns to it.

Why does your mixture code ever return an answer, then? Because gradient ascent and EM find *local* maxima, and the escape to infinity has a small basin of attraction — you must start near a data point with a tiny variance to fall into it. Your code is not solving the stated problem; it is solving "find a good local optimum", and it usually succeeds. That is worth internalizing, because it is also the situation in neural network training: the objective you wrote down may be pathological, and what saves you is the optimizer's inductive bias, not the objective. Section 8.10 makes the same point from another direction.

<figure>
<svg viewBox="0 0 720 350" style="max-width:100%;height:auto" role="img" aria-label="Two computed log-likelihood curves: a concave Bernoulli log-likelihood with one interior maximum, and a Cauchy location log-likelihood with three local maxima">
  <line x1="60" y1="230" x2="330" y2="230" stroke="currentColor" stroke-width="1.5"/>
  <line x1="60" y1="230" x2="60" y2="45" stroke="currentColor" stroke-width="1.5"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="62.7,151.4 68.0,111.2 73.3,93.4 78.7,82.4 84.0,74.6 89.3,68.9 94.6,64.4 99.9,61.0 105.2,58.3 110.6,56.1 115.9,54.5 121.2,53.2 126.5,52.3 131.8,51.8 137.2,51.5 142.5,51.4 147.8,51.6 153.1,52.0 158.4,52.6 163.8,53.4 169.1,54.4 174.4,55.6 179.7,57.0 185.0,58.5 190.3,60.3 195.7,62.2 201.0,64.4 206.3,66.7 211.6,69.3 216.9,72.1 222.3,75.2 227.6,78.5 232.9,82.1 238.2,86.0 243.5,90.2 248.9,94.8 254.2,99.9 259.5,105.4 264.8,111.4 270.1,118.1 275.4,125.5 280.8,133.8 286.1,143.2 291.4,153.9 296.7,166.3 302.0,181.2 307.4,199.3 312.7,222.6 314.0,229.6"/>
  <line x1="142.5" y1="51.4" x2="142.5" y2="230" stroke="#3b82f6" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="142.5" y="246" text-anchor="middle" font-size="12.5" fill="currentColor">θ̂ = 0.3</text>
  <text x="150" y="72" font-size="12.5" fill="currentColor">unique interior maximum;</text>
  <text x="150" y="88" font-size="12.5" fill="currentColor">strictly concave (ℓ″ &lt; 0)</text>
  <text x="30" y="140" font-size="12.5" fill="currentColor" transform="rotate(-90 30 140)">ℓ(θ)</text>
  <text x="195" y="264" text-anchor="middle" font-size="12.5" fill="currentColor">(a) Bernoulli, n = 20, k = 6. Max ℓ = −12.2173 at θ = 0.3.</text>

  <line x1="400" y1="230" x2="690" y2="230" stroke="currentColor" stroke-width="1.5"/>
  <line x1="400" y1="230" x2="400" y2="45" stroke="currentColor" stroke-width="1.5"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2.5" points="424.2,230.0 426.2,226.9 428.2,215.6 430.2,203.9 432.2,192.0 434.2,180.0 436.2,167.8 438.3,155.6 440.3,143.7 442.3,132.2 444.3,121.3 446.3,111.3 448.3,102.3 450.3,94.6 452.4,88.1 454.4,82.8 456.4,78.6 458.4,75.6 460.4,73.8 462.4,73.2 464.4,73.8 466.5,75.5 468.5,78.3 470.5,82.0 472.5,86.3 474.5,90.9 476.5,95.8 478.5,100.6 480.6,105.3 482.6,109.7 484.6,113.8 486.6,117.6 488.6,121.0 490.6,124.0 492.6,126.7 494.7,128.9 496.7,130.8 498.7,132.3 500.7,133.4 502.7,134.1 504.7,134.4 506.7,134.4 508.8,134.0 510.8,133.3 512.8,132.2 514.8,130.8 516.8,128.9 518.8,126.8 520.8,124.3 522.8,121.5 524.9,118.3 526.9,114.9 528.9,111.2 530.9,107.4 532.9,103.5 534.9,99.7 536.9,96.2 539.0,93.0 541.0,90.6 543.0,89.0 545.0,88.5 547.0,89.0 549.0,90.6 551.0,93.0 553.1,96.2 555.1,99.7 557.1,103.5 559.1,107.4 561.1,111.2 563.1,114.9 565.1,118.3 567.2,121.5 569.2,124.3 571.2,126.8 573.2,128.9 575.2,130.8 577.2,132.2 579.2,133.3 581.2,134.0 583.3,134.4 585.3,134.4 587.3,134.1 589.3,133.4 591.3,132.3 593.3,130.8 595.3,128.9 597.4,126.7 599.4,124.0 601.4,121.0 603.4,117.6 605.4,113.8 607.4,109.7 609.4,105.3 611.5,100.6 613.5,95.8 615.5,90.9 617.5,86.3 619.5,82.0 621.5,78.3 623.5,75.5 625.6,73.8 627.6,73.2 629.6,73.8 631.6,75.6 633.6,78.6 635.6,82.8 637.6,88.1 639.7,94.6 641.7,102.3 643.7,111.3 645.7,121.3 647.7,132.2 649.7,143.7 651.7,155.6 653.8,167.8 655.8,180.0 657.8,192.0 659.8,203.9 661.8,215.6 663.8,226.9 665.8,230.0"/>
  <circle cx="462.5" cy="73.2" r="4" fill="#f59e0b"/>
  <circle cx="627.5" cy="73.2" r="4" fill="#f59e0b"/>
  <circle cx="545.0" cy="88.5" r="4" fill="none" stroke="#ef4444" stroke-width="2"/>
  <g fill="currentColor">
    <circle cx="448.3" cy="235" r="2.2"/><circle cx="464.4" cy="235" r="2.2"/><circle cx="545.0" cy="235" r="2.2"/><circle cx="625.6" cy="235" r="2.2"/><circle cx="641.7" cy="235" r="2.2"/>
  </g>
  <text x="545" y="250" text-anchor="middle" font-size="11.5" fill="currentColor">data: −6, −5, 0, 5, 6</text>
  <text x="470" y="60" font-size="12" fill="#f59e0b">global maxima, ℓ = −19.0794</text>
  <text x="558" y="86" font-size="12" fill="#ef4444">median: local max only, −19.4617</text>
  <text x="545" y="264" text-anchor="middle" font-size="12.5" fill="currentColor">(b) Cauchy location, n = 5. Three local maxima, two local minima.</text>

  <text x="360" y="292" text-anchor="middle" font-size="12.5" fill="currentColor">Figure 8.1 — Both curves are computed from the formulas, not sketched. Left: a full-rank exponential family gives a</text>
  <text x="360" y="309" text-anchor="middle" font-size="12.5" fill="currentColor">strictly concave log-likelihood, so setting ℓ′ = 0 finds the unique MLE. Right: an equally regular one-parameter family</text>
  <text x="360" y="326" text-anchor="middle" font-size="12.5" fill="currentColor">whose log-likelihood has three local maxima; the sample median is a local maximum but not the MLE. Concavity, not</text>
  <text x="360" y="343" text-anchor="middle" font-size="12.5" fill="currentColor">regularity, is what makes "set the gradient to zero" safe. Multi-start is not optional here.</text>
</svg>
</figure>

> **Self-check 8.3.**
> 1. Two people fit the same model, one using Lebesgue base measure and one using $\mu(dx) = e^{-x^2}\,dx$. Do they get the same MLE? Do they report the same maximized likelihood value?
> 2. Why can the MLE for $\mathrm{Unif}[0,\theta]$ not be found by solving $\ell_n'(\theta)=0$?
> 3. In Proposition 8.11, where exactly does the argument use that $m_1$ is set equal to an observed data point rather than an arbitrary constant? And in Worked Example 8.9, what property of the Cauchy density — not shared by the Gaussian — creates the extra local maxima?
> 4. State in one sentence what a misspecified maximum-likelihood fit converges to, and say which direction of KL is involved.

---

## 8.4 Score, Fisher information, and the information equality

### 8.4.1 The score

Fix a dominated family with densities $p_\theta$ and suppose $\theta\mapsto p_\theta(x)$ is differentiable. The log-likelihood of a single observation is $\log p_\theta(x)$; its gradient in $\theta$ tells you which direction in parameter space would make this particular observation more likely.

> **Definition 8.12 (Score).**
> The **score** of the model at $\theta$ is the random vector
> $$s(\theta; X) := \nabla_\theta \log p_\theta(X) = \frac{\nabla_\theta p_\theta(X)}{p_\theta(X)} \in\mathbb R^d,$$
> defined for $x$ with $p_\theta(x)>0$. For an i.i.d. sample, the sample score is $\nabla_\theta \ell_n(\theta) = \sum_{i=1}^n s(\theta;X_i)$.

The score is exactly the per-example gradient that your training loop computes. When you write `loss.backward()` on a cross-entropy loss, the gradient you get back is $-s(\theta;x_i)$ (negated, because you are minimizing $-\ell$). Everything in this section is therefore a statement about the geometry of the thing your optimizer is following.

Before any theorem, the regularity conditions. They are not decoration; §8.3 already showed a family that violates them, and every asymptotic result below is false for that family.

> **Definition 8.13 (Regularity conditions).**
> Say the family $\{p_\theta:\theta\in\Theta\}$, $\Theta\subseteq\mathbb R^d$ open, is **regular at $\theta$** if:
> **(R1) Common support.** The set $\{x : p_\theta(x)>0\}$ does not depend on $\theta$ (call it $\mathcal S$).
> **(R2) Smoothness.** For $\mu$-a.e. $x\in\mathcal S$, $\theta'\mapsto p_{\theta'}(x)$ is twice continuously differentiable in a neighborhood $U$ of $\theta$. (Results about the score alone need only once-differentiability and (R3); the second derivative is used from Theorem 8.15 onward. I state the stronger condition once so that later results can cite a single label.)
> **(R3) Domination for the first derivative.** There is $g_1 \in L^1(\mu)$ with $\|\nabla_{\theta'} p_{\theta'}(x)\| \le g_1(x)$ for all $\theta'\in U$ and $\mu$-a.e. $x$.
> **(R4) Domination for the second derivative.** There is $g_2\in L^1(\mu)$ with $\|\nabla^2_{\theta'}p_{\theta'}(x)\|\le g_2(x)$ for all $\theta'\in U$ and $\mu$-a.e. $x$.
> Additionally we assume $\mathbb E_\theta\|s(\theta;X)\|^2 < \infty$.

(R3) and (R4) are precisely the hypotheses of the differentiation-under-the-integral theorem from Chapter 2, whose proof was the Dominated Convergence Theorem applied to difference quotients. Nothing new is being assumed here; the conditions are being *named* so that we can point at them when they fail. (R1) is the one that fails for $\mathrm{Unif}[0,\theta]$, and (R1) failing is fatal in a way that (R3) failing usually is not.

> **Proposition 8.14 (The score has mean zero).**
> Under (R1)–(R3), $\mathbb E_\theta[s(\theta;X)] = 0$ for every $\theta$ in the interior of $\Theta$.
>
> *Proof.* Since $\int_{\mathcal S} p_{\theta'}(x)\,\mu(dx) = 1$ for all $\theta'\in U$, differentiate both sides at $\theta'=\theta$. The right side gives $0$. On the left, (R2)+(R3) license the interchange of $\nabla_{\theta'}$ and $\int$ (Ch. 2), giving
> $$0 = \nabla_{\theta'}\Big|_{\theta'=\theta}\int_{\mathcal S} p_{\theta'}\,d\mu = \int_{\mathcal S}\nabla_\theta p_\theta\,d\mu = \int_{\mathcal S} \frac{\nabla_\theta p_\theta}{p_\theta}\,p_\theta\,d\mu = \mathbb E_\theta[s(\theta;X)],$$
> where dividing and multiplying by $p_\theta$ is legitimate because $p_\theta>0$ on $\mathcal S$ by (R1), and $\mu(\mathcal S^c\cap\{p_\theta>0\}) = 0$. $\blacksquare$

This is the result Chapter 2 promised and it deserves a moment. It says: **at the true parameter, the expected gradient of the log-likelihood is exactly zero.** The average training gradient vanishes at the truth — not approximately, exactly. Estimation is therefore a root-finding problem for a function whose true root you are trying to locate from a noisy version of it, which is exactly the framing of *M-estimation* and *estimating equations*, and exactly what SGD is doing when it chases $\mathbb E[g_t\mid\mathcal F_t] = \nabla F(\theta_t)$.

Where it fails: for $\mathrm{Unif}[0,\theta]$, $\log p_\theta(x) = -\log\theta$ on the support, so $s(\theta;x) = -1/\theta \neq 0$, and $\mathbb E_\theta[s] = -1/\theta \neq 0$. The score is a nonzero constant. (R1) is violated — the support $[0,\theta]$ moves — and the conclusion is simply false. Every downstream theorem (information equality, Cramér–Rao, asymptotic normality at rate $\sqrt n$) fails for this family, and Worked Example 8.5 showed the actual rate is $n$, not $\sqrt n$.

### 8.4.2 Fisher information and the information equality

> **Theorem 8.15 (Information equality).**
> Under (R1)–(R4), define the **Fisher information matrix**
> $$I(\theta) := \operatorname{Var}_\theta\big(s(\theta;X)\big) = \mathbb E_\theta\big[s(\theta;X)s(\theta;X)^\top\big]$$
> (the second equality by Proposition 8.14). Then
> $$I(\theta) = -\,\mathbb E_\theta\big[\nabla^2_\theta \log p_\theta(X)\big].$$
>
> *Proof.* Write $s = \nabla p_\theta/p_\theta$ and differentiate once more, using the quotient rule pointwise on $\mathcal S$:
> $$\nabla^2 \log p_\theta = \nabla\Big(\frac{\nabla p_\theta}{p_\theta}\Big) = \frac{\nabla^2 p_\theta}{p_\theta} - \frac{\nabla p_\theta (\nabla p_\theta)^\top}{p_\theta^2} = \frac{\nabla^2p_\theta}{p_\theta} - s s^\top .$$
> Take expectations under $P_\theta$. The second term contributes $-I(\theta)$. For the first,
> $$\mathbb E_\theta\Big[\frac{\nabla^2 p_\theta(X)}{p_\theta(X)}\Big] = \int_{\mathcal S}\frac{\nabla^2p_\theta}{p_\theta}p_\theta\,d\mu = \int_{\mathcal S}\nabla^2 p_\theta\,d\mu = \nabla^2\int_{\mathcal S}p_\theta\,d\mu = \nabla^2 (1) = 0,$$
> where the third equality interchanges $\nabla^2$ and $\int$, licensed by (R2)+(R4) via Chapter 2 applied twice (once to get $\nabla\int = \int\nabla$ using $g_1$, once more using $g_2$). Rearranging gives the claim; all quantities are finite by the assumed square-integrability of the score. $\blacksquare$

Two objects that have no business being equal turn out to be equal:

- $\operatorname{Var}(s)$ measures **how noisy the gradient is** — how much the per-example gradient varies across examples.
- $-\mathbb E[\nabla^2\log p_\theta]$ measures **how curved the expected log-likelihood is** at $\theta$ — how sharply the objective peaks.

The information equality says: *a sharply peaked likelihood is exactly one whose per-example gradients disagree a lot.* Intuitively, if all examples agreed on the gradient direction, moving in that direction would improve all of them and the peak would be broad and flat; disagreement is what pins the optimum down. This single identity underlies (i) the Gauss–Newton and Fisher approximations to the Hessian used by K-FAC and natural gradient methods, which replace an expensive $\nabla^2$ with a cheap outer product of gradients; (ii) the Laplace approximation to a posterior, whose covariance is an inverse information; and (iii) the folklore that "flat minima generalize better" — flatness is low information is high posterior variance is, under some assumptions, low sensitivity to the data.

**Important caveat, flagged now and cashed in §8.6:** the equality holds *at the true $\theta$ under the true model*. If the model is misspecified, $\operatorname{Var}(s) \ne -\mathbb E[\nabla^2\log p]$ in general, and the gap is exactly what the sandwich estimator corrects.

> **Proposition 8.16 (Additivity over independent observations).**
> If $X_1,\dots,X_n$ are i.i.d. from $P_\theta$ and the model is regular, the Fisher information of the sample is $I_n(\theta) = n I(\theta)$.
>
> *Proof.* The sample score is $\sum_i s(\theta;X_i)$, a sum of i.i.d. mean-zero (Proposition 8.14) random vectors, so by independence (Definition 1.23) its covariance is the sum of covariances, $nI(\theta)$. $\blacksquare$

Information is additive in the number of independent observations. Hence "the information doubles when the data doubles", hence the $\sqrt n$ rate everywhere. For *dependent* observations, additivity fails, which is one reason why the effective sample size of a correlated time series is smaller than its length, and why naive error bars on autocorrelated RL returns are too small.

> **Proposition 8.17 (Exponential families).**
> For $p_\eta(x)=h(x)\exp(\eta^\top T(x)-A(\eta))$ in natural parameterization on an open natural parameter space, the regularity conditions hold and
> $$s(\eta;x) = T(x)-\nabla A(\eta), \qquad I(\eta) = \nabla^2 A(\eta) = \operatorname{Var}_\eta(T(X)).$$
>
> *Proof.* $\log p_\eta(x) = \log h(x) + \eta^\top T(x) - A(\eta)$, so $\nabla_\eta \log p_\eta = T(x)-\nabla A(\eta)$ and $\nabla^2_\eta \log p_\eta = -\nabla^2A(\eta)$, which is *deterministic*. Hence $-\mathbb E[\nabla^2\log p_\eta]=\nabla^2A(\eta)$, and Chapter 3's identities give $\nabla A = \mathbb E[T]$, $\nabla^2 A=\operatorname{Var}(T)$; the variance of the score is then $\operatorname{Var}(T)$ directly. $\blacksquare$

Notice something remarkable: in an exponential family, the Hessian of the log-likelihood does not depend on the data at all. Observed and expected information coincide *exactly*, at every sample and every parameter value, not just asymptotically. This is why logistic regression's Newton step (IRLS) is so well behaved and why exponential-family GLMs are the one corner of statistics where second-order optimization is uncontroversial.

> **Proposition 8.18 (Reparameterization).**
> Let $\psi = g(\theta)$ with $g$ a $C^1$ diffeomorphism, and write $I_\theta$ and $I_\psi$ for the informations in the two parameterizations. Then
> $$I_\theta(\theta) = J^\top I_\psi(g(\theta))\, J, \qquad J := \frac{\partial g}{\partial\theta}(\theta).$$
>
> *Proof.* By the chain rule, $\nabla_\theta \log p(x) = J^\top \nabla_\psi \log p(x)$; take the covariance of both sides. $\blacksquare$

Fisher information is not a number attached to a distribution; it is a *quadratic form on the tangent space of the parameter manifold*, and it transforms like a metric tensor. That is the content of information geometry: $I(\theta)$ defines the **Fisher–Rao metric**, under which the distance between $P_\theta$ and $P_{\theta+d\theta}$ is $\sqrt{d\theta^\top I(\theta)\,d\theta}$ — a distance between *distributions*, invariant to how you happened to coordinatize them. The **natural gradient** is $\tilde\nabla \ell = I(\theta)^{-1}\nabla\ell$: steepest ascent measured in that metric rather than in Euclidean parameter space. Its selling point is exactly Proposition 8.18 — natural gradient descent is invariant to smooth reparameterization, so it does not care whether your network parameterizes a variance, a log-variance, or a softplus of something. Practical implementations (natural gradient, K-FAC, Adam viewed loosely as a diagonal approximation) all trade exactness for tractability; the optimization literature is where this thread continues.

**Worked Example 8.10 (Fisher information three ways, plus a reparameterization check).**
Take $X\sim\mathcal N(0,\sigma^2)$ parameterized by $\tau := \log\sigma$ — the parameterization deep learning actually uses, since it removes the positivity constraint. Then
$$\log p_\tau(x) = -\tau-\tfrac12\log(2\pi)-\tfrac12 x^2 e^{-2\tau},\quad s(\tau;x)=-1+x^2e^{-2\tau},\quad \partial_\tau^2\log p_\tau = -2x^2e^{-2\tau}.$$
Analytically: $\mathbb E[s] = -1+\mathbb E[X^2]/\sigma^2 = 0$ ✓; $\operatorname{Var}(s)=\operatorname{Var}(X^2)/\sigma^4 = 2$; $-\mathbb E[\partial_\tau^2\log p_\tau] = 2\mathbb E[X^2]/\sigma^2 = 2$. So $I(\tau)=2$, a constant — the information about $\log\sigma$ does not depend on $\sigma$.

Numerically, at $\sigma=2$ (so $\tau = 0.693147$), $M = 2{,}000{,}000$ Monte Carlo draws, seed 5150:

| route | value |
|---|---|
| (0) MC mean of the score (should be $0$) | $-0.001012$ |
| (1) MC variance of the score | $1.989877$ |
| (2) $-$MC mean of $\partial_\tau^2\log p_\tau$ | $1.997976$ |
| (2′) numerical second difference of the MC expected log-likelihood, $h=10^{-4}$ | $1.992028$ |
| (3) observed information $/n$ at the MLE, $n=500$ (seed 606) | $2.000000$ |

Route (3) is exactly $2.000000$, and not by luck: the observed information here is $2\sum_i x_i^2 e^{-2\hat\tau}/n$, and the MLE satisfies $e^{2\hat\tau} = \frac1n\sum_i x_i^2$, so the ratio is identically $2$. Here $\partial_\tau^2\log p_\tau$ is genuinely random, so observed and expected information differ at a generic $\tau$ — but the MLE equation is precisely the statement that makes their ratio deterministic, and the randomness cancels *at the maximizer*. (In the natural parameterization $\eta = -1/(2\sigma^2)$ the cancellation is stronger still and holds at every $\eta$, by Proposition 8.17.) In a model where this does not happen, observed and expected information genuinely differ: for the Cauchy location family, $I(\theta) = 1/2$ per observation, while the observed information per observation at the MLE has mean $0.5389$ (sd $0.1738$) at $n=20$, $0.5109$ (sd $0.1119$) at $n=50$, and $0.5038$ (sd $0.0558$) at $n=200$ over $4000$ replicates (seed 31415). It converges, but at any finite $n$ it is a random variable.

*Reparameterization check.* For the Bernoulli, $I(p)=1/(p(1-p))$ and in the natural (logit) parameterization $\eta=\log\frac{p}{1-p}$, $A(\eta)=\log(1+e^\eta)$, $I(\eta)=A''(\eta)=p(1-p)$. Proposition 8.18: $I(p) = (d\eta/dp)^2 I(\eta) = \big(\tfrac{1}{p(1-p)}\big)^2 p(1-p) = \tfrac1{p(1-p)}$ ✓. At $p=0.1$: $I(\eta)=0.090000$, $I(p)=11.111111$. The same statistical situation, two wildly different numbers, because they are measured in different units. **Fisher information is not comparable across parameterizations** — a fact routinely ignored when people compare "curvature" between architectures with different parameterizations of the same function class.

<figure>
<svg viewBox="0 0 720 340" style="max-width:100%;height:auto" role="img" aria-label="Computed expected log-likelihood curves for two Gaussian models of differing information, and the corresponding sampling densities of the MLE">
  <line x1="55" y1="185" x2="340" y2="185" stroke="currentColor" stroke-width="1.5"/>
  <line x1="197.5" y1="195" x2="197.5" y2="45" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="55.0,185.0 88.3,185.0 90.6,181.4 95.4,172.5 100.1,164.0 104.9,156.0 109.6,148.3 114.4,141.1 119.1,134.2 123.9,127.8 128.6,121.7 133.4,116.1 138.1,110.8 142.9,106.0 147.6,101.5 152.4,97.5 157.1,93.9 161.9,90.7 166.6,87.8 171.4,85.4 176.1,83.4 180.9,81.8 185.6,80.6 190.4,79.8 195.1,79.4 199.9,79.4 204.6,79.8 209.4,80.6 214.1,81.8 218.9,83.4 223.6,85.4 228.4,87.8 233.1,90.7 237.9,93.9 242.6,97.5 247.4,101.5 252.1,106.0 256.9,110.8 261.6,116.1 266.4,121.7 271.1,127.8 275.9,134.2 280.6,141.1 285.4,148.3 290.1,156.0 294.9,164.0 299.6,172.5 304.4,181.4 306.8,185.0 340.0,185.0"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="55.0,132.9 64.5,129.5 74.0,126.2 83.5,123.3 93.0,120.5 102.5,118.0 112.0,115.7 121.5,113.7 131.0,111.9 140.5,110.4 150.0,109.1 159.5,108.0 169.0,107.2 178.5,106.6 188.0,106.2 197.5,106.1 207.0,106.2 216.5,106.6 226.0,107.2 235.5,108.0 245.0,109.1 254.5,110.4 264.0,111.9 273.5,113.7 283.0,115.7 292.5,118.0 302.0,120.5 311.5,123.3 321.0,126.2 330.5,129.5 340.0,132.9"/>
  <text x="197.5" y="208" text-anchor="middle" font-size="12.5" fill="currentColor">θ₀</text>
  <text x="205" y="72" font-size="12" fill="#3b82f6">σ = 0.5: I(θ) = 4.00, sharp</text>
  <text x="205" y="100" font-size="12" fill="#10b981">σ = 1.3: I(θ) = 0.59, flat</text>
  <text x="28" y="128" font-size="12.5" fill="currentColor" transform="rotate(-90 28 128)">E[log p_θ(X)]</text>
  <text x="197" y="232" text-anchor="middle" font-size="12.5" fill="currentColor">expected log-likelihood of one observation</text>

  <line x1="400" y1="185" x2="690" y2="185" stroke="currentColor" stroke-width="1.5"/>
  <line x1="545" y1="195" x2="545" y2="45" stroke="currentColor" stroke-width="1" stroke-dasharray="4 3"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="400.0,185.0 494.2,185.0 499.1,184.8 503.9,184.3 508.8,182.6 513.6,178.5 518.4,169.6 523.2,153.4 528.1,128.8 532.9,98.4 537.8,69.5 542.6,51.6 545.0,49.1 547.4,51.6 552.2,69.5 557.1,98.4 561.9,128.8 566.8,153.4 571.6,169.6 576.4,178.5 581.2,182.6 586.1,184.3 590.9,184.8 595.8,185.0 690.0,185.0"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="400.0,185.0 419.3,185.0 429.0,184.9 438.7,184.7 448.3,184.3 458.0,183.3 467.7,181.6 477.3,178.5 487.0,173.7 496.7,167.0 506.3,158.6 516.0,149.4 525.7,140.9 535.3,134.9 545.0,132.7 554.7,134.9 564.3,140.9 574.0,149.4 583.7,158.6 593.3,167.0 603.0,173.7 612.7,178.5 622.3,181.6 632.0,183.3 641.7,184.3 651.3,184.7 661.0,184.9 670.7,185.0 690.0,185.0"/>
  <text x="545" y="208" text-anchor="middle" font-size="12.5" fill="currentColor">θ₀</text>
  <text x="556" y="60" font-size="12" fill="#3b82f6">sd = 0.158</text>
  <text x="620" y="145" font-size="12" fill="#10b981">sd = 0.411</text>
  <text x="545" y="232" text-anchor="middle" font-size="12.5" fill="currentColor">density of θ̂ = X̄ₙ at n = 10, i.e. N(θ₀, σ²/n)</text>

  <text x="360" y="270" text-anchor="middle" font-size="12.5" fill="currentColor">Figure 8.2 — Both panels are computed, on a common θ-axis. Left: the expected log-likelihood for N(θ, σ²) has curvature</text>
  <text x="360" y="287" text-anchor="middle" font-size="12.5" fill="currentColor">exactly I(θ) = 1/σ² at the truth — 4.00 for σ = 0.5, 0.59 for σ = 1.3. Right: the sampling density of the MLE, whose</text>
  <text x="360" y="304" text-anchor="middle" font-size="12.5" fill="currentColor">standard deviation is 1/√(n I(θ)) — 0.158 and 0.411. Sharper curvature, narrower sampling distribution, and by</text>
  <text x="360" y="321" text-anchor="middle" font-size="12.5" fill="currentColor">Theorem 8.15 the same number also equals the variance of the per-observation score. One quantity, three readings.</text>
</svg>
</figure>

> **Common Confusion 8.4.**
> *"Fisher information is the information in the data about $\theta$. So a model with more parameters has more information, right?"*
>
> Backwards on two counts. First, $I(\theta)$ is a property of the *model* at a point, not of a dataset; the data contribute only through the factor $n$ (Proposition 8.16). Second, more parameters usually means *less* information per parameter, because the information matrix gets larger and its inverse — which is what governs estimation error — gets bigger in the directions that are poorly determined. In the extreme, an overparameterized model has a **singular** $I(\theta)$: entire directions in parameter space leave the distribution unchanged (permute two hidden units; scale a layer up and the next down). Along those directions the information is exactly zero, $I(\theta)^{-1}$ does not exist, and every asymptotic result in §8.6 is inapplicable as stated. This is not a corner case for neural networks; it is the generic situation, and it is why the literature on "singular learning theory" exists.

> **Self-check 8.4.**
> 1. Which regularity condition does $\mathrm{Unif}[0,\theta]$ violate, and what goes wrong with Proposition 8.14 as a result?
> 2. State the information equality and say, in one sentence each, what the left and right sides measure operationally.
> 3. Why is Fisher information additive over i.i.d. observations but not over dependent ones?
> 4. If you reparameterize $\sigma\mapsto\log\sigma$, does the Fisher information change? Does the Fisher–Rao distance between two fixed distributions change?

---

## 8.5 The Cramér–Rao lower bound

### 8.5.1 Statement and proof

We now have a way to say how much information a model carries about $\theta$. The Cramér–Rao bound turns that into a hard limit: no unbiased estimator can have variance below the reciprocal of the information. It is the "you cannot go faster than light" of estimation theory — with the important footnote that, unlike light speed, you *can* beat it by being biased (§8.8).

> **Theorem 8.19 (Cramér–Rao, scalar case).**
> Let $\{p_\theta:\theta\in\Theta\subseteq\mathbb R\}$ be regular in the sense of Definition 8.13 with $0 < I(\theta)<\infty$, and let $\hat g_n = \hat g_n(X_1,\dots,X_n)$ be an estimator with $\mathbb E_\theta[\hat g_n^2]<\infty$ and $\mathbb E_\theta[\hat g_n] = g(\theta)$ for all $\theta$, where $g$ is differentiable and the interchange $\partial_\theta \int \hat g_n\,p_\theta^{\otimes n}d\mu^{\otimes n} = \int \hat g_n \,\partial_\theta p_\theta^{\otimes n}\,d\mu^{\otimes n}$ is valid. Then for all $\theta$,
> $$\operatorname{Var}_\theta(\hat g_n) \;\ge\; \frac{g'(\theta)^2}{n\,I(\theta)}.$$
> In particular, for an unbiased estimator of $\theta$ itself ($g(\theta)=\theta$), $\operatorname{Var}_\theta(\hat\theta_n)\ge 1/(nI(\theta))$.
>
> *Proof.* Write $S_n := \sum_{i=1}^n s(\theta;X_i)$ for the sample score, so $\mathbb E_\theta[S_n]=0$ (Proposition 8.14) and $\operatorname{Var}_\theta(S_n)=nI(\theta)$ (Proposition 8.16). Compute the covariance of $\hat g_n$ with $S_n$:
> $$\operatorname{Cov}_\theta(\hat g_n, S_n) = \mathbb E_\theta[\hat g_n S_n] = \int \hat g_n \frac{\partial_\theta \big(\prod_i p_\theta(x_i)\big)}{\prod_i p_\theta(x_i)}\prod_i p_\theta(x_i)\,\mu^{\otimes n}(dx) = \partial_\theta\int \hat g_n \prod_i p_\theta(x_i)\,\mu^{\otimes n}(dx) = g'(\theta),$$
> using the assumed interchange in the third step and unbiasedness in the last. Now apply Cauchy–Schwarz (Ch. 4) to the two mean-adjusted random variables $\hat g_n - g(\theta)$ and $S_n$:
> $$g'(\theta)^2 = \operatorname{Cov}_\theta(\hat g_n,S_n)^2 \le \operatorname{Var}_\theta(\hat g_n)\operatorname{Var}_\theta(S_n) = \operatorname{Var}_\theta(\hat g_n)\cdot n I(\theta).$$
> Divide. $\blacksquare$

The proof is three lines and one inequality, and the inequality is Cauchy–Schwarz. It is worth seeing what that means geometrically. In $L^2(P_\theta^{\otimes n})$, the score $S_n$ is a fixed direction. The covariance $\operatorname{Cov}(\hat g_n, S_n)=g'(\theta)$ is forced by unbiasedness: *every* unbiased estimator must have the same inner product with the score. Among all vectors with a prescribed inner product against a fixed direction, the one of smallest norm is the one parallel to that direction. So the bound is attained exactly when $\hat g_n - g(\theta)$ is proportional to $S_n$.

> **Corollary 8.20 (Multivariate Cramér–Rao).**
> Under the analogous conditions with $\theta\in\mathbb R^d$ and $\hat\theta_n$ unbiased for $\theta$,
> $$\operatorname{Cov}_\theta(\hat\theta_n) \;\succeq\; \frac{1}{n} I(\theta)^{-1},$$
> meaning the difference is positive semidefinite. Consequently $\operatorname{Var}_\theta(\hat\theta_{n,j}) \ge \frac1n (I(\theta)^{-1})_{jj}$ for each coordinate $j$.
>
> *Proof sketch.* Fix $a\in\mathbb R^d$ and apply Theorem 8.19 to the scalar estimator $a^\top\hat\theta_n$, unbiased for $a^\top\theta$. The Cauchy–Schwarz step becomes: for the $2d$-vector formed by stacking $\hat\theta_n - \theta$ and $S_n$, the covariance matrix $\begin{pmatrix}\Sigma & \mathrm{Id}\\ \mathrm{Id} & nI\end{pmatrix}$ is PSD, and a Schur-complement argument gives $\Sigma \succeq (nI)^{-1}$. $\blacksquare$

The coordinatewise statement is the practically important one and contains a warning: the bound for coordinate $j$ is $(I^{-1})_{jj}$, **not** $1/I_{jj}$. Since $(I^{-1})_{jj}\ge 1/I_{jj}$ always (with equality iff $I$ is block-diagonal separating $j$), estimating one parameter when others are unknown is strictly harder than estimating it when they are known. That gap is the cost of nuisance parameters, and it is why "we froze the other layers, so the analysis is fine" is usually not fine.

> **Definition 8.21 (Efficiency).**
> An unbiased estimator attaining the Cramér–Rao bound at every $\theta$ is called **efficient**. The **efficiency** of an unbiased $\hat\theta_n$ is the ratio $\frac{1}{nI(\theta)\operatorname{Var}_\theta(\hat\theta_n)}\in(0,1]$.

One direction of an implication is worth fixing in your head now, because the terms sound interchangeable and are not. An efficient estimator is automatically a *uniformly minimum variance unbiased* estimator (UMVUE, §8.7) — nothing unbiased can have smaller variance than a bound-attainer. The converse is **false**: $S^2$ is the UMVUE of a Gaussian variance (Theorem 8.32 will prove this) and yet sits strictly above the bound at every finite $n$, as Worked Example 8.11 is about to show. "Best among unbiased" and "attains the bound" are different claims; the bound is not always attainable, and when it is not, the UMVUE is simply the best of a class none of whose members reach it.

> **Proposition 8.22 (When the bound is attained).**
> Equality holds in Theorem 8.19 at $\theta$ if and only if there is a constant $c(\theta)$ (not depending on the data) with
> $$\hat g_n(x) - g(\theta) = c(\theta)\,S_n(\theta;x) \quad \text{for } P_\theta\text{-a.e. } x.$$
> If this holds for all $\theta$ in an open set, then the sample density $\prod_i p_\theta(x_i)$ is a one-parameter exponential family with natural sufficient statistic $\hat g_n$; by the i.i.d. product structure the single-observation family is then itself exponential, with $\hat g_n$ an affine function of the sum of its natural statistics. (Exercise A7 carries out the integration for $n=1$.)
>
> *Proof of the first claim.* Cauchy–Schwarz is an equality iff the two random variables are a.s. proportional; the mean-zero versions here are $\hat g_n - g(\theta)$ and $S_n$. $\blacksquare$

This is a strong restriction. Efficiency in the exact, finite-$n$ sense is essentially available only inside exponential families, for the natural sufficient statistic, in the mean parameterization. Everywhere else the bound is a target, approached asymptotically at best.

### 8.5.2 The bound in action, and where it does not apply

**Worked Example 8.11 (Gaussian mean: attained; Gaussian variance: not).**
*Mean, variance known.* $X_i\sim\mathcal N(\mu,\sigma^2)$, $\sigma^2$ known. $s(\mu;x)=(x-\mu)/\sigma^2$, so $I(\mu)=1/\sigma^2$ and the bound is $\sigma^2/n$. The estimator $\bar X_n$ is unbiased with variance exactly $\sigma^2/n$: **attained**. Indeed $\bar X_n - \mu = \frac{\sigma^2}{n}S_n$, the exact proportionality of Proposition 8.22. Simulation with $n=8$, $N=500{,}000$ (seed 1234) gives $\operatorname{Var}(\bar X)=0.124663$ against the bound $1/8 = 0.125$.

*Variance, mean unknown.* $\theta = (\mu,\sigma^2)$. From $\log p = -\frac12\log(2\pi\sigma^2) - (x-\mu)^2/(2\sigma^2)$,
$$s_\mu = \frac{x-\mu}{\sigma^2},\quad s_{\sigma^2} = -\frac{1}{2\sigma^2}+\frac{(x-\mu)^2}{2\sigma^4},\qquad I(\theta)=\begin{pmatrix}1/\sigma^2 & 0\\ 0 & 1/(2\sigma^4)\end{pmatrix}.$$
(The off-diagonal vanishes because $\mathbb E[(X-\mu)^3]=0$.) Monte Carlo confirmation with $\mu=1,\sigma^2=4$, $N=4\times10^6$ (seed 24680): the sample covariance of the score is $\begin{pmatrix}0.249768 & 0.000065\\ 0.000065 & 0.031192\end{pmatrix}$ against the analytic $\operatorname{diag}(0.250000, 0.031250)$.

So the bound for $\sigma^2$ is $2\sigma^4/n$. But the unbiased $S^2$ has variance $2\sigma^4/(n-1) > 2\sigma^4/n$ — strictly above the bound at every finite $n$, approaching it only as $n\to\infty$. With $\sigma^2=1$:

| $n$ | CRLB $=2/n$ | $\operatorname{Var}(\hat\sigma^2_{\mathrm{MLE}}) = 2(n-1)/n^2$ | $\operatorname{Var}(S^2)=2/(n-1)$ | ratio MLE/CRLB |
|---|---|---|---|---|
| 5 | $0.400000$ | $0.320000$ | $0.500000$ | $0.8000$ |
| 10 | $0.200000$ | $0.180000$ | $0.222222$ | $0.9000$ |
| 20 | $0.100000$ | $0.095000$ | $0.105263$ | $0.9500$ |
| 50 | $0.040000$ | $0.039200$ | $0.040816$ | $0.9800$ |
| 200 | $0.010000$ | $0.009950$ | $0.010050$ | $0.9950$ |

Look at the third column: the MLE's variance is *below* the Cramér–Rao bound at every $n$. There is no contradiction — the MLE is biased, and Theorem 8.19 constrains unbiased estimators only. This is the single most common misreading of the bound, and the table exists to make it unmissable.

<figure>
<svg viewBox="0 0 700 350" style="max-width:100%;height:auto" role="img" aria-label="Plot of Cramer-Rao bound, variance of the unbiased sample variance, and variance of the MLE against sample size">
  <line x1="80" y1="300" x2="640" y2="300" stroke="currentColor" stroke-width="1.5"/>
  <line x1="80" y1="300" x2="80" y2="35" stroke="currentColor" stroke-width="1.5"/>
  <text x="360" y="322" text-anchor="middle" font-size="13" fill="currentColor">sample size n  (5 → 50)</text>
  <text x="34" y="170" font-size="13" fill="currentColor" transform="rotate(-90 34 170)">variance (σ² = 1)</text>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="80.0,110.9 104.0,164.9 128.0,194.9 152.0,214.0 176.0,227.3 200.0,237.0 224.0,244.4 248.0,250.2 272.0,255.0 296.0,258.9 320.0,262.2 344.0,265.0 368.0,267.4 392.0,269.5 416.0,271.3 440.0,273.0 464.0,274.4 488.0,275.8 512.0,276.9 536.0,278.0 560.0,279.0 584.0,279.9 608.0,280.7 620.0,281.1"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2.5" points="80.0,63.6 104.0,142.4 128.0,181.8 152.0,205.5 176.0,221.2 200.0,232.5 224.0,240.9 248.0,247.5 272.0,252.7 296.0,257.0 320.0,260.6 344.0,263.6 368.0,266.2 392.0,268.5 416.0,270.5 440.0,272.2 464.0,273.7 488.0,275.1 512.0,276.4 536.0,277.5 560.0,278.5 584.0,279.4 608.0,280.3 620.0,280.7"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="6 4" points="80.0,148.7 104.0,184.2 128.0,206.6 152.0,221.9 176.0,232.9 200.0,241.2 224.0,247.7 248.0,252.9 272.0,257.1 296.0,260.7 320.0,263.7 344.0,266.3 368.0,268.5 392.0,270.5 416.0,272.2 440.0,273.8 464.0,275.1 488.0,276.4 512.0,277.5 536.0,278.5 560.0,279.5 584.0,280.3 608.0,281.1 620.0,281.5"/>
  <text x="150" y="58" font-size="13" fill="#ef4444">Var(S²) = 2/(n−1) — unbiased, above the bound</text>
  <text x="150" y="103" font-size="13" fill="#3b82f6">CRLB = 2/n</text>
  <text x="150" y="140" font-size="13" fill="#10b981">Var(MLE) = 2(n−1)/n² — biased, below the bound</text>
  <text x="350" y="342" text-anchor="middle" font-size="12.5" fill="currentColor">Figure 8.3 — The Cramér–Rao bound (blue) constrains unbiased estimators only. S² (red) stays strictly above it and approaches it as n grows; the biased MLE (green, dashed) sits legally below it. All three curves merge at rate 2/n.</text>
</svg>
</figure>

**Where the bound does not apply at all.** If (R1) fails the bound is meaningless, and the uniform family shows how badly. For $\mathrm{Unif}[0,\theta]$ the "score" is the constant $-1/\theta$, so a naive computation of $\mathbb E_\theta[s^2]$ gives $I(\theta)=1/\theta^2$ and a naive bound of $\theta^2/n$. (That the same family gives $\operatorname{Var}(s)=0$ and $-\mathbb E[\partial^2_\theta\log p_\theta]=-1/\theta^2$ — a *negative* number, which no variance can be — is itself proof that something has gone wrong.) But the unbiased estimator $\frac{n+1}{n}X_{(n)}$ has variance $\frac{\theta^2}{n(n+2)}$, which is smaller than $\theta^2/n$ by a factor $n+2$. The bound is violated by an arbitrarily large factor, because it was never valid: the interchange in Proposition 8.14 and in Theorem 8.19's proof is illegal when the support moves with $\theta$.

That is not the end of the story, though, and it would be misleading to leave it there. There *is* a variance bound for irregular models; it just does not use derivatives.

> **Proposition 8.23 (Chapman–Robbins bound).**
> Let $\hat g$ be unbiased for $g(\theta)$ with finite variance, and suppose $h\ne0$ is such that $P_{\theta+h}^{\otimes n}\ll P_\theta^{\otimes n}$. Then
> $$\operatorname{Var}_\theta(\hat g) \;\ge\; \sup_{h}\; \frac{\big(g(\theta+h)-g(\theta)\big)^2}{\chi^2_n(\theta+h,\theta)}, \qquad \chi^2_n(\theta',\theta) := \mathbb E_\theta\Big[\Big(\tfrac{dP_{\theta'}^{\otimes n}}{dP_\theta^{\otimes n}}-1\Big)^2\Big].$$
> No differentiability, no common support, no interchange of $\partial_\theta$ and $\int$ is required.
>
> *Proof.* Fix such an $h$ and write $L := \frac{dP_{\theta'}^{\otimes n}}{dP_\theta^{\otimes n}}$ with $\theta' := \theta+h$, which exists by absolute continuity (Radon–Nikodym, Ch. 2). Then $\mathbb E_\theta[L-1]=0$ and
> $$\operatorname{Cov}_\theta(\hat g, L-1) = \mathbb E_\theta[\hat g\,L] - \mathbb E_\theta[\hat g] = \mathbb E_{\theta'}[\hat g] - \mathbb E_\theta[\hat g] = g(\theta')-g(\theta),$$
> using unbiasedness at *both* parameter values. Cauchy–Schwarz (Ch. 4) gives $\big(g(\theta')-g(\theta)\big)^2 \le \operatorname{Var}_\theta(\hat g)\cdot\chi^2_n(\theta',\theta)$. Take the supremum over admissible $h$. $\blacksquare$

Compare the two proofs: Cramér–Rao takes the covariance of $\hat g$ with the *score* (an infinitesimal likelihood ratio) and needs differentiability to make sense of it; Chapman–Robbins takes the covariance with a *finite* likelihood ratio and needs nothing. Cramér–Rao is the $h\to0$ limit when that limit exists, since $\chi^2_n(\theta+h,\theta)/h^2 \to nI(\theta)$ under regularity.

**The uniform family, done correctly.** For $\mathrm{Unif}[0,\theta]$ take $h<0$ so that $[0,\theta+h]\subset[0,\theta]$ and absolute continuity holds (for $h>0$ it fails, and $\chi^2=\infty$ — which is exactly the asymmetry the moving support creates). Writing $t := (\theta+h)/\theta\in(0,1)$, one observation has $\mathbb E_\theta[L_1^2] = t\cdot t^{-2} = 1/t$, so by independence $\chi^2_n = t^{-n}-1$, and with $g(\theta)=\theta$,
$$\operatorname{Var}_\theta(\hat\theta) \;\ge\; \theta^2\sup_{t\in(0,1)}\frac{(1-t)^2\,t^{n}}{1-t^{n}}.$$
Evaluating the supremum numerically (grid of $4\times10^6$ points):

| $n$ | Chapman–Robbins bound $/\theta^2$ | attained at $t$ | $\operatorname{Var}\big(\tfrac{n+1}{n}X_{(n)}\big)/\theta^2$ | ratio |
|---|---|---|---|---|
| 1 | $0.250000$ | $0.5000$ | $0.333333$ | $1.333$ |
| 2 | $0.090170$ | $0.6180$ | $0.125000$ | $1.386$ |
| 5 | $0.019648$ | $0.7773$ | $0.028571$ | $1.454$ |
| 10 | $0.005586$ | $0.8688$ | $0.008333$ | $1.492$ |
| 50 | $0.000251$ | $0.9694$ | $0.000385$ | $1.532$ |

Now everything is consistent. The bound is *valid* (always below the achieved variance), *non-vacuous* (within a factor of about $1.5$), and *correctly scaled*: it decays like $n^{-2}$, matching the $O_P(1/n)$ rate of Worked Example 8.5, whereas the illegitimate $\theta^2/n$ decays like $n^{-1}$ and is above the true variance from $n=1$ onward. When a model is irregular, the right response is a different bound, not no bound.

> **Common Confusion 8.5.**
> *"The Cramér–Rao bound says no estimator can do better than $1/(nI(\theta))$. So the MLE is optimal and I should stop looking."*
>
> Three separate errors. (i) The bound applies to *unbiased* estimators; biased ones routinely beat it, as the table above shows and as James–Stein will show spectacularly in §8.8. (ii) The bound is about variance at a *fixed* $\theta$; an estimator can have tiny variance at one $\theta$ (e.g. the constant estimator $\hat\theta\equiv 7$ has zero variance everywhere) while being useless. The bound rules out uniformly-good-and-unbiased, not pointwise-good. (iii) Attaining the bound requires exact proportionality to the score (Proposition 8.22), which is rare; typically the best you get is asymptotic attainment (§8.6), and "asymptotically efficient" is a much weaker claim than most readers assume — it says nothing at $n=100$.

> **Self-check 8.5.**
> 1. In the proof of Theorem 8.19, exactly which step uses unbiasedness, and which uses Cauchy–Schwarz?
> 2. Why is $\operatorname{Var}(\hat\sigma^2_{\mathrm{MLE}}) < 2\sigma^4/n$ not a counterexample to the bound?
> 3. Explain in one sentence why $(I^{-1})_{jj} \ge 1/I_{jj}$ is the "price of nuisance parameters".
> 4. For $\mathrm{Unif}[0,\theta]$, an unbiased estimator has variance $\theta^2/(n(n+2))$, far below $\theta^2/n$. Which hypothesis of Theorem 8.19 fails, and which bound *does* apply — with what $n$-dependence?

---

## 8.6 Asymptotic theory of the maximum likelihood estimator

This is the section that justifies almost every standard error you will ever compute. The headline is one line:
$$\sqrt n\,(\hat\theta_n - \theta_0) \to_d \mathcal N\big(0, I(\theta_0)^{-1}\big).$$
The MLE is consistent, asymptotically normal, and asymptotically attains the Cramér–Rao bound. It is worth understanding both why this is true and — more importantly for practice — under exactly which hypotheses, because in machine learning most of them fail.

### 8.6.1 Consistency

> **Theorem 8.24 (Consistency of the MLE).**
> Let $X_1,\dots,X_n\stackrel{iid}\sim P_{\theta_0}$ with $\theta_0$ in the interior of a compact $\Theta\subseteq\mathbb R^d$. Assume:
> **(C1) Identifiability.** $\theta\ne\theta_0 \Rightarrow P_\theta \ne P_{\theta_0}$.
> **(C2) Continuity.** $\theta\mapsto \log p_\theta(x)$ is continuous for $\mu$-a.e. $x$.
> **(C3) Uniform integrable envelope.** $\mathbb E_{\theta_0}\big[\sup_{\theta\in\Theta}|\log p_\theta(X)|\big]<\infty$.
> Then $\hat\theta_n\to_{a.s.}\theta_0$.
>
> *Proof.* Define $M(\theta) := \mathbb E_{\theta_0}[\log p_\theta(X)]$ and $M_n(\theta) := \frac1n\ell_n(\theta)$. Note $\hat\theta_n$ maximizes $M_n$.
> *Step 1: $M$ is uniquely maximized at $\theta_0$.* For any $\theta$, $M(\theta_0)-M(\theta) = \mathbb E_{\theta_0}\big[\log\frac{p_{\theta_0}(X)}{p_\theta(X)}\big] = D_{\mathrm{KL}}(P_{\theta_0}\|P_\theta)\ge0$, with equality iff $P_\theta = P_{\theta_0}$ (Ch. 4, from Jensen applied to the strictly convex $-\log$), i.e. iff $\theta=\theta_0$ by (C1).
> *Step 2: uniform SLLN.* By (C2), (C3) and compactness of $\Theta$, the classical uniform law of large numbers gives $\sup_{\theta\in\Theta}|M_n(\theta)-M(\theta)|\to_{a.s.}0$. (Sketch: cover $\Theta$ by finitely many balls; on each, control the oscillation of $\log p_\theta(x)$ by a bracketing function whose expectation is small by (C3) and dominated convergence, then apply the SLLN of Chapter 7 to the finitely many bracket endpoints and take a union bound — Proposition 1.8(2). Full details in van der Vaart, Thm 5.7.)
> *Step 3: argmax convergence.* Fix $\varepsilon>0$ and let $\eta := M(\theta_0)-\sup_{\|\theta-\theta_0\|\ge\varepsilon} M(\theta)$, which is $>0$ by Step 1 plus compactness and continuity of $M$. On the a.s. event where $\sup_\theta|M_n-M|<\eta/2$ eventually, any $\theta$ with $\|\theta-\theta_0\|\ge\varepsilon$ has $M_n(\theta) < M(\theta)+\eta/2 \le M(\theta_0)-\eta/2 < M_n(\theta_0)$, so the maximizer $\hat\theta_n$ cannot be there. Hence $\|\hat\theta_n-\theta_0\|<\varepsilon$ eventually, a.s. $\blacksquare$

One hypothesis deserves an honest flag before we congratulate ourselves, because a careful reader will notice it immediately: Theorem 8.24 assumes $\Theta$ is **compact**, and almost none of our examples satisfy that. The Gaussian has $\Theta = \mathbb R\times(0,\infty)$, the exponential has $(0,\infty)$, the Bernoulli's $[0,1]$ is the rare exception. Compactness is doing one job in the proof — guaranteeing in Step 3 that the population objective drops by a definite amount $\eta>0$ outside every $\varepsilon$-ball, which an objective on a non-compact set can fail to do (its supremum "at infinity" can creep back up to $M(\theta_0)$). The standard repairs, due to Wald: either show directly that $M(\theta)\to-\infty$ as $\theta$ approaches the boundary or infinity (true for the Gaussian: $v\to0$ or $v\to\infty$ both send $M\to-\infty$, so the maximizer is eventually trapped in a compact set and the theorem applies there), or compactify $\Theta$ by adding its boundary points and verify (C2)–(C3) extend by upper semicontinuity. Both are bookkeeping, but they are *your* bookkeeping when you use this theorem on a non-compact family; papers that cite "Wald consistency" are citing exactly this repair. The envelope condition (C3) is the other hypothesis that quietly fails in interesting models — for the Gaussian mixture of Proposition 8.11, $\sup_\theta \log p_\theta(x) = +\infty$ at every $x$, so (C3) is violated and consistency of "the MLE" is not even a well-posed question, which is the consistency-side view of the unbounded-likelihood pathology.

Read Step 1 again: **the MLE is consistent because KL divergence is minimized at the truth.** Everything else is bookkeeping to get from "the limiting objective has the right maximizer" to "the maximizers of the finite-$n$ objectives converge", and that bookkeeping needs *uniform* convergence, not pointwise. Pointwise convergence is not enough — an objective can converge pointwise everywhere while its argmax runs off to infinity. This is exactly the same distinction Chapter 1 flagged around $\sup_{h\in\mathcal H}|\hat R_n(h)-R(h)|$: the machine learning version of Step 2 is the uniform convergence at the heart of generalization theory, and Chapter 12 develops it properly.

**Worked Example 8.12 (Neyman–Scott: consistency can fail when $d$ grows with $n$).**
Observe pairs $X_{i1},X_{i2}\stackrel{ind}\sim\mathcal N(m_i,\sigma^2)$ for $i=1,\dots,n$: each pair has its *own* mean, and $\sigma^2$ is common. The parameter is $(m_1,\dots,m_n,\sigma^2)$, so the parameter count grows with the sample size. Maximizing,
$$\hat m_i = \tfrac12(X_{i1}+X_{i2}), \qquad \hat\sigma^2 = \frac{1}{2n}\sum_{i=1}^n\sum_{j=1}^2 (X_{ij}-\hat m_i)^2 = \frac{1}{4n}\sum_{i=1}^n (X_{i1}-X_{i2})^2.$$
Since $X_{i1}-X_{i2}\sim\mathcal N(0,2\sigma^2)$, each summand has mean $2\sigma^2$, so
$$\mathbb E[\hat\sigma^2]=\frac{\sigma^2}{2} \quad\text{exactly, for every } n, \qquad \operatorname{Var}(\hat\sigma^2) = \frac{\sigma^4}{2n},$$
and by the SLLN $\hat\sigma^2\to_{a.s.}\sigma^2/2$. **The MLE converges to half the truth.** Note carefully what is and is not happening as $n$ grows: the *bias does not shrink at all* — it is $-\sigma^2/2$ at every sample size — while the variance goes to zero. So the estimator becomes ever more precisely centred on the wrong number. A simulation with $\sigma^2=1$ and $200$ replicates (seed 1001) returned mean $\hat\sigma^2$ of $0.4880$, $0.4939$, $0.4999$, $0.5000$ at $n=10,100,1000,20{,}000$ pairs; since the exact mean is $0.5$ regardless of $n$, this is purely a check on the algebra, and the apparent drift is Monte Carlo noise (the standard error at $n=10$ is $\sqrt{1/(2\cdot10)/200}=0.0158$, so $0.4880$ is well within one standard error). More data buys confidence, not accuracy.

Which hypothesis broke? (C1) and (C3) are fine; the model is identifiable. What broke is the unstated assumption behind the whole setup: that $\Theta$ is *fixed* while $n\to\infty$. Here the dimension of $\Theta$ is $n+1$. Each pair contributes two observations and one new nuisance parameter, so the information about $\sigma^2$ per pair never accumulates cleanly. This is the incidental-parameters problem, and it is the closest classical statistics comes to the overparameterized regime of §8.10. The standard fixes — profile out the nuisance parameters, use a conditional or marginal likelihood, or put a prior on them and integrate (which is what a hierarchical Bayesian model does) — are all in the modern practitioner's toolkit under different names.

### 8.6.2 Asymptotic normality

> **Theorem 8.25 (Asymptotic normality of the MLE).**
> Assume (i) $\hat\theta_n\to_p\theta_0$ with $\theta_0$ interior to $\Theta$; (ii) regularity (R1)–(R4) in a neighborhood of $\theta_0$, with $I(\theta_0)$ finite and nonsingular; (iii) $\hat\theta_n$ eventually solves $\nabla\ell_n(\hat\theta_n)=0$; (iv) the third derivatives of $\log p_\theta(x)$ are dominated near $\theta_0$ by an integrable $g_3(x)$ (or, more weakly, $\theta\mapsto\nabla^2\log p_\theta$ is continuous with a locally integrable envelope). Then
> $$\sqrt n(\hat\theta_n-\theta_0) \to_d \mathcal N\big(0, I(\theta_0)^{-1}\big).$$
>
> *Proof.* Expand the sample score around $\theta_0$. Since $\nabla\ell_n(\hat\theta_n)=0$, Taylor's theorem gives
> $$0 = \frac{1}{\sqrt n}\nabla\ell_n(\theta_0) + \bar H_n\,\sqrt n(\hat\theta_n-\theta_0), \qquad \bar H_n := \frac1n\int_0^1 \nabla^2\ell_n\big(\theta_0 + t(\hat\theta_n-\theta_0)\big)\,dt,$$
> the **integral form of the remainder**, which is exact in any dimension with no intermediate point needed (it is the fundamental theorem of calculus applied to $t\mapsto\nabla\ell_n(\theta_0+t(\hat\theta_n-\theta_0))$, coordinate by coordinate). A remark on the version you will see in most textbooks: the scalar mean-value theorem applies to each *coordinate* of the score separately, producing a possibly *different* intermediate point $\tilde\theta_{n,j}$ on the segment for each row $j$ of the Hessian; there is in general **no single** $\tilde\theta_n$ that works for all rows simultaneously, and proofs that write one are cutting a corner. Both fixes lead to the same place, because every point on the segment converges to $\theta_0$ when $\hat\theta_n$ does, and the uniform argument below controls the Hessian at *all* of them at once. We write the expansion with $\bar H_n$ and, for readability, phrase the convergence step below in terms of the segment; to translate, read "$\frac1n\nabla^2\ell_n(\tilde\theta_n)$" as "$\bar H_n$", justified because $\|\bar H_n + I(\theta_0)\| \le \sup_{\theta\in B(\theta_0,\|\hat\theta_n-\theta_0\|)}\|\frac1n\nabla^2\ell_n(\theta)+I(\theta_0)\|$, the same supremum the uniform law controls. Displaying the segment form:
> $$0 = \frac{1}{\sqrt n}\nabla\ell_n(\theta_0) + \Big(\frac1n\nabla^2\ell_n(\tilde\theta_n)\Big)\sqrt n(\hat\theta_n-\theta_0).$$
> **Term 1.** $\frac1{\sqrt n}\nabla\ell_n(\theta_0) = \frac{1}{\sqrt n}\sum_i s(\theta_0;X_i)$ is $\sqrt n$ times the average of i.i.d. mean-zero (Proposition 8.14) vectors with covariance $I(\theta_0)$ (Theorem 8.15), finite by hypothesis. The multivariate CLT (Ch. 7) gives
> $$\frac1{\sqrt n}\nabla\ell_n(\theta_0)\to_d \mathcal N(0, I(\theta_0)).$$
> **Term 2.** $\frac1n \nabla^2\ell_n(\theta) = \frac1n\sum_i \nabla^2\log p_\theta(X_i)$. By the SLLN, at the fixed point $\theta_0$ this converges a.s. to $\mathbb E_{\theta_0}[\nabla^2\log p_{\theta_0}(X)] = -I(\theta_0)$ (Theorem 8.15). Because $\tilde\theta_n$ lies between $\hat\theta_n$ and $\theta_0$ and $\hat\theta_n\to_p\theta_0$, we have $\tilde\theta_n\to_p\theta_0$; hypothesis (iv) plus a uniform LLN over a shrinking neighborhood upgrades this to
> $$\frac1n\nabla^2\ell_n(\tilde\theta_n)\to_p -I(\theta_0).$$
> **Combine.** Solving, $\sqrt n(\hat\theta_n-\theta_0) = \big(-\frac1n\nabla^2\ell_n(\tilde\theta_n)\big)^{-1}\frac1{\sqrt n}\nabla\ell_n(\theta_0)$. The matrix converges in probability to $I(\theta_0)$, which is invertible, so by the continuous mapping theorem its inverse converges in probability to $I(\theta_0)^{-1}$. Slutsky's theorem (Ch. 7) then gives
> $$\sqrt n(\hat\theta_n-\theta_0)\to_d I(\theta_0)^{-1}\mathcal N(0,I(\theta_0)) = \mathcal N\big(0, I(\theta_0)^{-1}I(\theta_0)I(\theta_0)^{-1}\big) = \mathcal N(0,I(\theta_0)^{-1}). \qquad\blacksquare$$

Three things to notice, because they are what you actually need to remember.

*Where consistency is used.* Twice, and both times essentially: to know $\hat\theta_n$ is eventually interior so that $\nabla\ell_n(\hat\theta_n)=0$ holds at all, and to know $\tilde\theta_n\to_p\theta_0$ so the Hessian is evaluated near the right place. Theorem 8.25 assumes consistency; it does not prove it. That division of labor is why the two theorems are separate.

*Where uniform convergence is used.* In Term 2, to pass from "the average Hessian converges at the fixed point $\theta_0$" to "it converges at the random, data-dependent point $\tilde\theta_n$". Pointwise LLN does not give this. Papers say "by a standard argument"; the standard argument is a uniform LLN on a shrinking ball.

*Why the answer is the inverse information and not something uglier.* The middle of the sandwich (the covariance of the score, $I$) and the bread (the inverse Hessian, $I^{-1}$) are made of the same matrix — by Theorem 8.15. The two $I$'s cancel one $I^{-1}$. **When the model is misspecified they are different matrices and nothing cancels.** That is the entire content of §8.6.4.

Theorem 8.25 gives error bars for $\theta$. Usually you want error bars for something else — a probability rather than a logit, a standard deviation rather than a log-variance, an odds ratio, a predicted value. Proposition 8.9 already told you the point estimate: apply $g$. The variance follows from Chapter 7 with no extra work.

> **Corollary 8.26 (Standard errors for a transformed MLE).**
> Under the hypotheses of Theorem 8.25, let $g:\Theta\to\mathbb R^k$ be continuously differentiable at $\theta_0$ with Jacobian $G := \nabla g(\theta_0)$. Then
> $$\sqrt n\big(g(\hat\theta_n)-g(\theta_0)\big)\to_d \mathcal N\big(0,\; G\,I(\theta_0)^{-1}G^\top\big).$$
> In the scalar case the practical recipe is $\widehat{\operatorname{SE}}\big(g(\hat\theta_n)\big) = |g'(\hat\theta_n)|\cdot\widehat{\operatorname{SE}}(\hat\theta_n)$.
>
> *Proof.* Theorem 8.25 plus the delta method (Ch. 7). $\blacksquare$

**A worked interval for a log-odds.** Bernoulli data, $n=100$, $\hat p = 0.30$. Then $\widehat{\operatorname{SE}}(\hat p) = \sqrt{\hat p(1-\hat p)/n} = 0.0458$. For the log-odds $g(p)=\log\frac{p}{1-p}$ we have $g'(p) = \frac{1}{p(1-p)} = 4.7619$, so $\widehat{\operatorname{SE}}(\hat\eta) = 4.7619\times0.0458 = 0.2182$ — which agrees exactly with the direct computation $\sqrt{1/(n\hat p(1-\hat p))} = 0.2182$, as Proposition 8.18 says it must. The $95\%$ Wald interval on the logit scale is $[-1.2750,\,-0.4196]$; pushing its endpoints back through the inverse logit gives $[0.2184,\,0.3966]$ for $p$, against the direct Wald interval $[0.2102,\,0.3898]$.

Note that the two intervals for $p$ are *different*, and neither is wrong. Wald intervals are not equivariant, because a symmetric interval on one scale maps to an asymmetric interval on another. The logit-scale version is usually better behaved (it cannot leave $[0,1]$, and its coverage is closer to nominal for $\hat p$ near the boundary), which is why logistic-regression software reports intervals on the coefficient scale and transforms afterwards. Chapter 9 returns to this when comparing Wald, score, and likelihood-ratio intervals.

> **Definition 8.27 (Observed and expected information).**
> The **observed information** at $\theta$ is $J_n(\theta) := -\nabla^2\ell_n(\theta)$, a random matrix computed from the data. The **expected information** for $n$ observations is $nI(\theta) = \mathbb E_\theta[J_n(\theta)]$.

In practice you almost always use $J_n(\hat\theta_n)$ in place of $nI(\hat\theta_n)$, because it requires no expectation over a model you are not sure about, and it is what your autodiff library will hand you. Both are consistent estimators of $nI(\theta_0)$ when the model is correct; Efron and Hinkley (1978) argue the observed version is preferable even then, because it conditions on an approximate ancillary. The practical Wald interval is
$$\hat\theta_{n,j} \pm z_{1-\alpha/2}\sqrt{\big(J_n(\hat\theta_n)^{-1}\big)_{jj}}.$$

**Worked Example 8.13 (Asymptotic normality and Wald coverage for an exponential rate).**
(*A note on notation, once and for all: the canonical notation of this tutorial reserves $\lambda$ for Lebesgue measure, but statistics has irreversibly attached the same letter to the Exponential and Poisson rate and to regularization strength, and this chapter uses all three. Context always disambiguates — Lebesgue $\lambda$ appears only next to "base measure" or as $\lambda|_{(0,\infty)}$, a rate appears inside $\mathrm{Exp}(\lambda)$ or $\mathrm{Poisson}(\lambda)$, and a penalty weight appears next to a norm, as in $\lambda\|\beta\|^2$ in §8.8 — but you should notice the collision consciously rather than trip over it.*)

$X_i\stackrel{iid}\sim\mathrm{Exp}(\lambda)$, density $\lambda e^{-\lambda x}$. Then $\log p = \log\lambda - \lambda x$, $s(\lambda;x) = 1/\lambda - x$, $\partial^2_\lambda\log p = -1/\lambda^2$, so $I(\lambda)=1/\lambda^2$ and $\hat\lambda_n = 1/\bar X_n$. Theorem 8.25 predicts $\sqrt n(\hat\lambda_n - \lambda)\to_d\mathcal N(0,\lambda^2)$.

Simulation with $\lambda = 2$, $N = 200{,}000$ replicates (seed 808), using the exact fact that $\sum_i X_i\sim\mathrm{Gamma}(n,1/\lambda)$:

| $n$ | $\mathbb E[\hat\lambda_n]$ | $\operatorname{sd}\!\big(\sqrt n(\hat\lambda_n-\lambda)\big)$ | Wald 95% coverage |
|---|---|---|---|
| 5 | $2.5014$ | $3.2398$ | $0.9556$ |
| 20 | $2.1045$ | $2.2069$ | $0.9534$ |
| 100 | $2.0205$ | $2.0378$ | $0.9513$ |
| 1000 | $2.0018$ | $2.0031$ | $0.9498$ |

The target sd is $\lambda = 2.0000$; at $n=5$ the finite-sample sd is $62\%$ too large (indeed $\mathbb E[\hat\lambda_n]=\frac{n}{n-1}\lambda$ exactly, and $\operatorname{Var}(\hat\lambda_n)$ is infinite for $n\le2$). Coverage is close to nominal even at small $n$ here, but that is a happy accident of this model, and it is worth seeing exactly which two errors are cancelling. Error one: the plug-in standard error is $\hat\lambda_n/\sqrt n$, and $\hat\lambda_n$ is biased *upward* by the factor $n/(n-1)$, so the intervals are systematically too *wide* — which by itself pushes coverage above $95\%$. Error two: the sampling distribution of $\hat\lambda_n = 1/\bar X_n$ (an inverse-Gamma) is right-skewed at small $n$, so a symmetric interval centred at $\hat\lambda_n$ misses the true $\lambda$ on the left tail more often than the normal approximation predicts — which by itself pushes coverage below $95\%$. At $n=5$ these happen to net out near $0.9556$. Neither error is small; they are large and opposite. In a model where they pointed the same way — a downward-biased estimator with the same skew, say — coverage would be badly off, so do not generalize the happy total.

### 8.6.3 Asymptotic efficiency, and the caveats nobody prints

Combining Theorem 8.25 with Theorem 8.19: the MLE is asymptotically unbiased with asymptotic variance $I(\theta_0)^{-1}/n$, which is the Cramér–Rao bound. So the MLE is *asymptotically efficient*. This is the standard justification for using it.

The caveats are real and are routinely omitted.

**(1) "Asymptotically efficient" is a limit statement and carries no finite-$n$ guarantee.** There exist *superefficient* estimators (Hodges' example: take the MLE, but shrink it to $0$ whenever $|\hat\theta_n| < n^{-1/4}$) that beat $I^{-1}$ at a single point $\theta=0$ while matching it elsewhere. The price is bad behavior in a shrinking neighborhood of that point, and the Le Cam–Hájek convolution theorem shows that in a suitable local-uniform sense $I^{-1}$ really is optimal. But the crude statement "nothing beats the MLE asymptotically" is false as stated.

**(2) The asymptotics are in $n$ with $d$ fixed.** When $d$ grows with $n$ — always, in modern ML — Theorem 8.25 says nothing. Worked Example 8.12 shows how badly it can fail.

**(3) $I(\theta_0)$ must be nonsingular.** Neural networks violate this by construction (Common Confusion 8.4).

**(4) The model must be correct.** Which brings us to the last part of this section.

### 8.6.4 Misspecification: the KL projection and the sandwich

Suppose the data actually come from $P^\star$, which is *not* in the model $\{P_\theta\}$. Section 8.3.3 already told us where the MLE goes: to
$$\theta^\star := \arg\min_{\theta\in\Theta} D_{\mathrm{KL}}(P^\star\,\|\,P_\theta) = \arg\max_{\theta\in\Theta}\mathbb E_{P^\star}[\log p_\theta(X)],$$
the **KL projection** of the truth onto the model, assumed to exist and be unique and interior. Consistency for $\theta^\star$ follows from Theorem 8.24 with $M(\theta) := \mathbb E_{P^\star}[\log p_\theta(X)]$ — nothing in Steps 2 and 3 used correctness, only Step 1, which is replaced by "the population objective has a unique maximizer at $\theta^\star$" by assumption.

> **Theorem 8.28 (Sandwich / White's theorem).**
> Under regularity conditions analogous to Theorem 8.25 but with expectations taken under $P^\star$, and with $\hat\theta_n\to_p\theta^\star$,
> $$\sqrt n(\hat\theta_n - \theta^\star)\to_d\mathcal N\big(0,\; H^{-1} G\, H^{-1}\big),$$
> where
> $$G := \mathbb E_{P^\star}\big[s(\theta^\star;X)s(\theta^\star;X)^\top\big] \quad\text{(the "meat")},\qquad H := -\mathbb E_{P^\star}\big[\nabla^2\log p_{\theta^\star}(X)\big] \quad\text{(the "bread")}.$$
> If the model is correctly specified, $G = H = I(\theta_0)$ by Theorem 8.15 and the sandwich collapses to $I(\theta_0)^{-1}$.
>
> *Proof.* Identical to Theorem 8.25 up to the last line. The score at $\theta^\star$ has mean zero under $P^\star$ — not by Proposition 8.14, which needs correctness, but by the first-order condition defining $\theta^\star$: $\nabla_\theta \mathbb E_{P^\star}[\log p_\theta(X)]\big|_{\theta^\star}=0$, with the interchange licensed by domination. Its covariance is $G$. The average Hessian converges to $-H$. Slutsky gives $\sqrt n(\hat\theta_n-\theta^\star)\to_d H^{-1}\mathcal N(0,G) = \mathcal N(0,H^{-1}GH^{-1})$. Without Theorem 8.15 there is no cancellation. $\blacksquare$

The plug-in estimator is
$$\widehat{\operatorname{Cov}}(\hat\theta_n) = \hat H^{-1}\hat G\hat H^{-1}/n, \quad \hat H := -\tfrac1n\nabla^2\ell_n(\hat\theta_n),\quad \hat G := \tfrac1n\sum_i s(\hat\theta_n;X_i)s(\hat\theta_n;X_i)^\top,$$
known as the **robust**, **Huber–White**, or **heteroskedasticity-consistent** covariance estimator. Both ingredients are things autodiff gives you: a Hessian and a sum of outer products of per-example gradients.

**Worked Example 8.14 (Naive vs sandwich standard errors under heteroskedasticity).**
Generate $x_i\sim\mathcal N(0,1)$ and $y_i = 1 + 2x_i + \varepsilon_i$ with $\varepsilon_i\sim\mathcal N\big(0,(0.5+2|x_i|)^2\big)$ — the noise grows with $|x|$. Fit ordinary least squares, which is the MLE under the (wrong) homoskedastic Gaussian model. The slope estimate is still consistent for $2$, because the conditional mean is correctly specified; what is wrong is the *uncertainty*.

With $n = 100$ and $R = 20{,}000$ replications (seed 2718):

| quantity | value |
|---|---|
| empirical sd of $\hat\beta_1$ across replications | $0.3902$ |
| mean naive (model-based) standard error | $0.2401$ |
| coverage of the naive 95% interval | $0.7716$ |
| mean sandwich standard error | $0.3684$ |
| coverage of the sandwich 95% interval | $0.9268$ |

The naive standard error understates the true variability by $38\%$, and a nominal 95% interval covers $77\%$ of the time. The sandwich is not perfect at $n=100$ (it is a plug-in estimate of a fourth moment and is itself noisy and biased low), but it recovers most of the gap.

Why this matters for machine learning: **every model you fit is misspecified.** A language model is not the true distribution over text. A ResNet's softmax is not the true conditional label distribution. Therefore every "model-based" uncertainty — the posterior variance from a Laplace approximation, the predictive variance of a Bayesian neural network fitted with a Gaussian likelihood, the confidence implied by a calibrated softmax — is computed with the bread only and is systematically too small in exactly the way the table above shows. When people observe that deep networks are overconfident, part of that is optimization and part of it is this: the information equality is being invoked in a setting where it does not hold.

> **Common Confusion 8.6.**
> *"If the model is misspecified, the MLE is just wrong, so who cares what it converges to?"*
>
> It converges to a specific, meaningful, often useful object: the closest point in your model class to the truth in the forward-KL sense. That is frequently what you wanted. A linear model fit to nonlinear data recovers the best linear predictor; a Gaussian fit to a skewed distribution recovers the true mean and variance (Worked Example 8.8); a next-token model with limited context recovers the best limited-context approximation. The problem is not the point estimate; the problem is the *error bars*, which are computed under the assumption that the model is right. Misspecification breaks the variance, not the target.

### 8.6.5 Nuisance parameters and the profile likelihood

Corollary 8.20 warned that estimating $\theta_1$ while $\theta_2$ is unknown is strictly harder than estimating it when $\theta_2$ is known, the penalty being $(I^{-1})_{11} \ge 1/I_{11}$. The standard tool for working with one parameter of interest in the presence of nuisance parameters is already in front of us: it appeared in Proposition 8.9 as a device for proving equivariance.

> **Definition 8.29 (Profile likelihood).**
> Partition $\theta = (\psi,\lambda)$ into a parameter of interest $\psi\in\mathbb R^{d_1}$ and a nuisance parameter $\lambda$. The **profile log-likelihood** for $\psi$ is
> $$\ell_n^{\mathrm{P}}(\psi) := \sup_{\lambda} \ell_n(\psi,\lambda) = \ell_n\big(\psi, \hat\lambda(\psi)\big),$$
> where $\hat\lambda(\psi)$ is the MLE of $\lambda$ with $\psi$ held fixed.

By construction $\ell_n^{\mathrm{P}}$ is maximized at $\hat\psi$, the corresponding block of the full MLE — that is Proposition 8.9 again. What is less obvious, and genuinely useful, is that the profile likelihood also carries the *right* curvature. Differentiating the identity $\nabla_\lambda \ell_n(\psi,\hat\lambda(\psi)) = 0$ implicitly in $\psi$ and substituting gives, at the MLE and after dividing by $n$,
$$-\tfrac1n\nabla^2_\psi \ell_n^{\mathrm{P}}(\hat\psi) \;\longrightarrow\; I_{\psi\psi} - I_{\psi\lambda}I_{\lambda\lambda}^{-1}I_{\lambda\psi} \;=\; \big[(I^{-1})_{\psi\psi}\big]^{-1},$$
the Schur complement — which is exactly the quantity Corollary 8.20 said governs the achievable variance, *not* the naive $I_{\psi\psi}$. So the profile likelihood automatically pays the nuisance-parameter price, and Wald intervals built from its curvature are the correct ones.

A concrete check. For the Gamma$(\alpha,s)$ family, $I(\alpha,s) = \begin{pmatrix}\psi'(\alpha) & 1/s\\ 1/s & \alpha/s^2\end{pmatrix}$. At $\alpha=2$, $s=3$: $I_{\alpha\alpha}=\psi'(2)=0.644934$, but the Schur complement is $\psi'(\alpha)-1/\alpha = 0.144934$, and $1/(I^{-1})_{\alpha\alpha} = 0.144934$ as well. Using $I_{\alpha\alpha}$ instead of the Schur complement would understate the standard error of $\hat\alpha$ by a factor $\sqrt{0.644934/0.144934} = 2.11$ — more than a doubling. Reporting $1/\sqrt{n I_{jj}}$ when other parameters are also being estimated is a common and consequential error.

Two forward pointers. First, $2\big(\ell_n^{\mathrm{P}}(\hat\psi) - \ell_n^{\mathrm{P}}(\psi_0)\big)$ has a $\chi^2_{d_1}$ limit — Wilks' theorem, which is Theorem 8.25 in disguise and is the subject of Chapter 9. Second, the profile likelihood is *not* a likelihood: it is not the density of anything, it does not account for the uncertainty in $\hat\lambda(\psi)$, and when $\dim\lambda$ grows with $n$ it inherits exactly the Neyman–Scott pathology of Worked Example 8.12 — the profile MLE of $\sigma^2$ there *is* the inconsistent one. Modified profile likelihoods exist for this reason.

> **Self-check 8.6.**
> 1. In the proof of Theorem 8.24, which step uses that KL divergence is minimized at the truth, and which uses uniform convergence?
> 2. In the proof of Theorem 8.25, where precisely is consistency needed?
> 3. In Neyman–Scott, the MLE converges to $\sigma^2/2$. Which assumption of Theorem 8.24 is being violated — and is it one of (C1)–(C3)?
> 4. Under correct specification the sandwich collapses. Which theorem makes it collapse, and what does the collapse mean about the relationship between gradient noise and curvature? Separately: if you report $1/\sqrt{nI_{jj}}$ as the standard error of $\hat\theta_j$ while other parameters are also estimated, is your interval too wide or too narrow?

---

## 8.7 Sufficiency and Rao–Blackwell

### 8.7.1 Conditioning on a sufficient statistic never hurts

Chapter 3 defined sufficiency and proved the Fisher–Neyman factorization: $T$ is sufficient for $\theta$ iff $p_\theta(x) = g_\theta(T(x))h(x)$, and the operational meaning is that the conditional distribution of $X$ given $T(X)=t$ does not depend on $\theta$. So $T$ retains all the $\theta$-relevant information; the rest of $X$ is noise.

That last sentence suggests an algorithm: take any estimator, average out the noise. The averaging is done by conditional expectation — Chapter 6's object — and the improvement is guaranteed by the law of total variance.

> **Theorem 8.30 (Rao–Blackwell).**
> Let $\delta = \delta(X_1,\dots,X_n)$ be an estimator of $g(\theta)$ with $\mathbb E_\theta[\delta^2]<\infty$ for all $\theta$, and let $T$ be sufficient for $\theta$. Define
> $$\delta^{\mathrm{RB}} := \mathbb E[\delta \mid T].$$
> Then (i) $\delta^{\mathrm{RB}}$ is a statistic (it does not depend on $\theta$); (ii) $\mathbb E_\theta[\delta^{\mathrm{RB}}]=\mathbb E_\theta[\delta]$, so $\delta^{\mathrm{RB}}$ has the same bias; and (iii) for every $\theta$,
> $$\operatorname{MSE}_\theta(\delta^{\mathrm{RB}}) \le \operatorname{MSE}_\theta(\delta),$$
> with equality iff $\delta = \delta^{\mathrm{RB}}$ $P_\theta$-a.s.
>
> *Proof.* (i) The conditional distribution of the data given $T$ is free of $\theta$ by sufficiency, so the map $t\mapsto\mathbb E[\delta\mid T=t]$ is a fixed measurable function computable without knowing $\theta$. This is exactly where sufficiency is needed and it is the only place. (ii) Tower property (Ch. 6): $\mathbb E_\theta[\mathbb E[\delta\mid T]]=\mathbb E_\theta[\delta]$.
> (iii) Apply the law of total variance (Ch. 6) to $\delta$ conditioning on $T$:
> $$\operatorname{Var}_\theta(\delta) = \mathbb E_\theta\big[\operatorname{Var}(\delta\mid T)\big] + \operatorname{Var}_\theta\big(\mathbb E[\delta\mid T]\big) \;\ge\; \operatorname{Var}_\theta(\delta^{\mathrm{RB}}),$$
> since the first term is nonnegative. By (ii) the biases agree, so Proposition 8.3 turns the variance inequality into an MSE inequality. Equality forces $\mathbb E_\theta[\operatorname{Var}(\delta\mid T)]=0$, i.e. $\operatorname{Var}(\delta\mid T)=0$ a.s., i.e. $\delta$ is a.s. a function of $T$, i.e. $\delta=\delta^{\mathrm{RB}}$ a.s. $\blacksquare$

One warning before the example. The variance inequality in part (iii) used no sufficiency at all — it is just the law of total variance, valid for conditioning on *any* $T$. Sufficiency was needed only for part (i): without it, $\mathbb E[\delta\mid T]$ depends on $\theta$ and so is not an estimator, because you could not compute it without already knowing the answer. Sufficiency is exactly the condition that makes the conditional expectation computable. The machine learning analogue: averaging over dropout masks is legitimate because the mask distribution is known and does not involve the parameters; "averaging over the label noise" is not, because the noise distribution is part of what you are estimating.

**Rao–Blackwell is the law of total variance.** That is the whole theorem. Chapter 6's decomposition said the variance of anything splits into "variance explained by $T$" plus "variance left over given $T$"; conditioning on $T$ deletes the leftover term. The generalization is immediate: for any convex loss $L$, Jensen's inequality for conditional expectations (Ch. 4 and Ch. 6) gives $\mathbb E_\theta[L(\delta^{\mathrm{RB}}, g(\theta))] \le \mathbb E_\theta[L(\delta,g(\theta))]$, so the theorem is not special to squared error.

The ML version of this idea, which you have probably used without naming it: whenever your estimator depends on auxiliary randomness that is not informative about the target — a dropout mask, a data-augmentation draw, a minibatch permutation, a sampled latent — averaging over that randomness (analytically if you can, by Monte Carlo if you must) can only reduce variance and never changes the mean. That is why "sample $K$ times and average" reduces the variance of a REINFORCE gradient estimator, why the *reparameterization trick* beats the score-function estimator (it Rao–Blackwellizes over the noise), why Rao–Blackwellized particle filters exist, and why test-time augmentation helps. The theoretical guarantee is Theorem 8.30.

**Worked Example 8.15 (Improving a crude estimator by conditioning, with numbers).**
Let $X_1,\dots,X_n\stackrel{iid}\sim\mathrm{Poisson}(\lambda)$ and suppose we want $g(\lambda) := P_\lambda(X=0) = e^{-\lambda}$ — the probability that a count is zero, a quantity of real interest in zero-inflation modeling.

*Crude estimator.* $\delta := \mathbf 1\{X_1 = 0\}$. It is unbiased: $\mathbb E[\delta] = P(X_1=0)=e^{-\lambda}$. It is also idiotic — it throws away $X_2,\dots,X_n$ and returns only $0$ or $1$. Its variance is $\theta(1-\theta)$ with $\theta = e^{-\lambda}$.

*The sufficient statistic.* $T = \sum_i X_i \sim \mathrm{Poisson}(n\lambda)$; sufficiency is immediate from the factorization $p_\lambda(x)=e^{-n\lambda}\lambda^{\sum x_i}/\prod x_i!$.

*Rao–Blackwellize.* We need $\mathbb E[\mathbf 1\{X_1=0\}\mid T=t] = P(X_1 = 0\mid T=t)$. Given $\sum_i X_i = t$, the vector $(X_1,\dots,X_n)$ is Multinomial$(t; 1/n,\dots,1/n)$ — a standard Poisson fact, and note it does not involve $\lambda$, as sufficiency promised. Hence
$$\delta^{\mathrm{RB}} = P(X_1=0\mid T=t) = \Big(1-\frac1n\Big)^{t} = \Big(\frac{n-1}{n}\Big)^{T}.$$
Unbiased, uses all the data, and takes values in $(0,1]$.

*The numbers.* Take $\lambda=1$, $n=10$, so $\theta = e^{-1}=0.367879$. Since $T\sim\mathrm{Poisson}(10)$ and $\mathbb E[a^T]=e^{n\lambda(a-1)}$,
$$\operatorname{Var}(\delta^{\mathrm{RB}}) = e^{n\lambda(a-1)} - \theta^2, \quad a := \big(\tfrac{9}{10}\big)^2 = 0.8100,$$
giving $0.149569 - 0.135335 = 0.014233$, against $\operatorname{Var}(\delta) = 0.232544$. **A variance reduction of $16.34\times$, for free.** Simulation with $N=2{,}000{,}000$ (seed 4242): crude mean $0.367946$, variance $0.232562$; RB mean $0.367858$, variance $0.014221$. For comparison, the MLE plug-in $e^{-\bar X}$ (which is *not* unbiased, by Jensen) had mean $0.386092$ and MSE $0.014450$ — slightly worse in MSE than the unbiased RB estimator's $0.014221$, which is unusual and worth noticing, since plug-in MLEs usually win.

### 8.7.2 Completeness, Lehmann–Scheffé, and the limits of the program

Rao–Blackwell improves an estimator but does not tell you when to stop. Completeness does.

> **Definition 8.31 (Complete statistic).**
> A statistic $T$ is **complete** for $\{P_\theta\}$ if, for every measurable $f$ with $\mathbb E_\theta[|f(T)|]<\infty$,
> $$\mathbb E_\theta[f(T)] = 0 \text{ for all }\theta \implies f(T)=0 \ P_\theta\text{-a.s. for all }\theta.$$

In words: the only unbiased estimator of zero that is a function of $T$ is zero itself. Full-rank exponential families have complete sufficient statistics (their natural sufficient statistic $T$ is complete when the natural parameter space contains an open set — a fact from Chapter 3, following from uniqueness of Laplace transforms).

> **Theorem 8.32 (Lehmann–Scheffé).**
> Let $T$ be complete and sufficient, and let $\delta$ be any unbiased estimator of $g(\theta)$ with finite variance. Then $\delta^{\mathrm{RB}}=\mathbb E[\delta\mid T]$ is the **unique** uniformly minimum variance unbiased estimator (UMVUE) of $g(\theta)$.
>
> *Proof.* $\delta^{\mathrm{RB}}$ is unbiased (Theorem 8.30(ii)) and a function of $T$. Let $\delta'$ be any other unbiased estimator; then $\delta'^{\mathrm{RB}}:=\mathbb E[\delta'\mid T]$ is also unbiased and a function of $T$. Their difference $f(T) := \delta^{\mathrm{RB}}-\delta'^{\mathrm{RB}}$ satisfies $\mathbb E_\theta[f(T)]=0$ for all $\theta$, so by completeness $f(T)=0$ a.s.: all Rao–Blackwellized unbiased estimators coincide. By Theorem 8.30(iii), $\operatorname{Var}_\theta(\delta^{\mathrm{RB}}) = \operatorname{Var}_\theta(\delta'^{\mathrm{RB}}) \le \operatorname{Var}_\theta(\delta')$ for every $\theta$ and every unbiased $\delta'$. Uniqueness follows from the equality condition. $\blacksquare$

So $\big(\frac{n-1}{n}\big)^T$ is *the* UMVUE of $e^{-\lambda}$: nothing unbiased does better, anywhere.

Now the honest assessment. This is a beautiful theory with a narrow domain. It requires (a) restricting to unbiased estimators — which §8.1 and §8.8 argue is the wrong restriction; (b) a complete sufficient statistic, which essentially means an exponential family of fixed dimension; and (c) an initial unbiased estimator to improve, which may not exist (there is no unbiased estimator of $1/\lambda$ for a Poisson, for instance). In modern machine learning, condition (b) never holds. The *idea* — average out uninformative randomness — transfers completely and is used constantly. The *theorem* about UMVUEs does not transfer and should not be over-sold.

> **Self-check 8.7.**
> 1. Which chapter-6 identity is the engine of Theorem 8.30, and which term does conditioning kill?
> 2. Why does the theorem require $T$ *sufficient*, given that the variance inequality holds for any $T$?
> 3. In Worked Example 8.15, why does the conditional distribution of $(X_1,\dots,X_n)$ given $T=t$ not involve $\lambda$?
> 4. Give one machine learning technique that is Rao–Blackwellization in disguise, and say what plays the role of $T$.

---

## 8.8 MAP, regularization, and shrinkage

### 8.8.1 A prior is a penalty

Chapter 1's Proposition 1.12 gave Bayes' rule with the terminology fixed: posterior $\propto$ likelihood $\times$ prior. Put a probability measure $\Pi$ on $\Theta$ with density $\pi$ (w.r.t. Lebesgue on $\Theta$, say), and the posterior density given data $x$ is
$$\pi(\theta\mid x) = \frac{L_n(\theta)\pi(\theta)}{\int_\Theta L_n(\theta')\pi(\theta')\,d\theta'}.$$
The denominator (the evidence) does not depend on $\theta$.

> **Definition 8.33 (Maximum a posteriori estimator).**
> A **MAP estimator** is any maximizer
> $$\hat\theta^{\mathrm{MAP}} \in \arg\max_{\theta\in\Theta} \big\{\ell_n(\theta) + \log\pi(\theta)\big\}.$$

That is the entire definition, and read from right to left it says: **MAP is penalized maximum likelihood, with penalty $-\log\pi(\theta)$.** Every regularizer you have ever added to a loss is a negative log prior, whether or not anyone intended it that way.

> **Proposition 8.34 (Gaussian prior gives ridge; Laplace prior gives lasso).**
> Consider the Gaussian linear model $y_i \mid x_i \sim \mathcal N(x_i^\top\beta, \sigma^2)$ with $\sigma^2$ known, $\beta \in \mathbb R^d$.
> (i) With prior $\beta \sim \mathcal N(0, \tau^2 I_d)$,
> $$\hat\beta^{\mathrm{MAP}} = \arg\min_\beta \Big\{\tfrac{1}{2\sigma^2}\|y - X\beta\|_2^2 + \tfrac{1}{2\tau^2}\|\beta\|_2^2\Big\} = (X^\top X + \tfrac{\sigma^2}{\tau^2}I)^{-1}X^\top y,$$
> which is **ridge regression** with $\lambda = \sigma^2/\tau^2$ (equivalently, $L_2$ weight decay).
> (ii) With prior $\beta_j\stackrel{iid}\sim\mathrm{Laplace}(0,b)$, i.e. $\pi(\beta)\propto e^{-\|\beta\|_1/b}$,
> $$\hat\beta^{\mathrm{MAP}} = \arg\min_\beta\Big\{\tfrac{1}{2\sigma^2}\|y-X\beta\|_2^2 + \tfrac1b\|\beta\|_1\Big\},$$
> which is the **lasso** with $\lambda = \sigma^2/b$.
>
> *Proof.* (i) $-\log\pi(\beta) = \frac{1}{2\tau^2}\|\beta\|^2 + $ const, and $-\ell_n(\beta) = \frac{1}{2\sigma^2}\|y-X\beta\|^2+$ const. The objective is strictly convex (its Hessian is $\frac{1}{\sigma^2}X^\top X + \frac{1}{\tau^2}I \succ 0$), so the unique minimizer solves $\frac{1}{\sigma^2}X^\top(X\beta - y) + \frac{1}{\tau^2}\beta = 0$, which rearranges to the stated formula. (ii) $-\log\pi(\beta)=\frac1b\|\beta\|_1+$const; the objective is convex but non-differentiable at $\beta_j=0$, so first-order conditions are stated with subgradients. $\blacksquare$

Two consequences worth stating explicitly.

**Weight decay is a Gaussian prior with variance $\sigma^2/\lambda$.** When you set weight decay to $10^{-4}$ in an optimizer, you are asserting a prior belief that each weight is roughly $\mathcal N(0, \sigma^2/10^{-4})$. Whether you believe that is a separate question; the point is that you have asserted it.

**Sparsity comes from the shape of the prior at zero, not from its tails.** In the orthogonal design $X^\top X = I$ (and taking $\sigma^2=1$, so that $\lambda$ is the penalty weight as written), the lasso solution is coordinatewise soft-thresholding, $\hat\beta_j = \operatorname{sign}(z_j)(|z_j|-\lambda)_+$ with $z = X^\top y$, while ridge is uniform scaling $\hat\beta_j = z_j/(1+\lambda)$. With $z=(2.5,-1.1,0.4,-0.2,3.0)$ and $\lambda = 0.8$: lasso gives $(1.7,\,-0.3,\,0.0,\,-0.0,\,2.2)$ — two exact zeros — while ridge gives $(1.389,\,-0.611,\,0.222,\,-0.111,\,1.667)$ — no zeros, everything shrunk. The Laplace density has a *kink* at $0$ (its derivative jumps), which makes $0$ a stationary point of the penalized objective over a whole interval of $z$; the Gaussian density is smooth at $0$ and produces no such interval. This is why "$L_1$ gives sparsity" is a statement about non-differentiability, not about heavy tails.

### 8.8.2 MAP is not reparameterization invariant, and MLE is

Proposition 8.9 said the MLE of $g(\theta)$ is $g$ of the MLE. MAP has no such property, and the reason is Chapter 3's change-of-variables formula.

> **Proposition 8.35 (MAP is not equivariant).**
> Let $\psi = g(\theta)$ with $g$ a $C^1$ diffeomorphism. If $\pi_\theta$ is the prior density in the $\theta$-parameterization, the induced prior density in $\psi$ is $\pi_\psi(\psi) = \pi_\theta(g^{-1}(\psi))\,|\det J_{g^{-1}}(\psi)|$. Consequently
> $$\arg\max_\psi\{\ell_n(g^{-1}(\psi)) + \log\pi_\psi(\psi)\} \;\ne\; g\big(\arg\max_\theta\{\ell_n(\theta)+\log\pi_\theta(\theta)\}\big)$$
> in general: the two differ by the additional term $\log|\det J_{g^{-1}}(\psi)|$, which depends on $\psi$ and so shifts the maximizer.
>
> *Proof.* The likelihood term is equivariant (it is a function value, not a density in $\theta$), but the prior term is a *density* and therefore picks up a Jacobian under change of variables. Maximizing $f(\psi)$ and maximizing $f(\psi)+J(\psi)$ give different answers unless $J$ is constant, i.e. unless $g$ is affine. $\blacksquare$

**Worked Example 8.16 (The same posterior, two different MAPs).**
Binomial data: $n=10$ trials, $y=3$ successes, prior $p\sim\mathrm{Beta}(2,2)$. The posterior is $\mathrm{Beta}(y+2, n-y+2)=\mathrm{Beta}(5,9)$, whose density is $\propto p^{4}(1-p)^{8}$.

*MAP in $p$*: maximize $4\log p + 8\log(1-p)$, giving $\hat p = 4/12 = 1/3 = 0.33333$. (Grid search over $2{,}000{,}001$ points confirms $0.33333$.)

*MAP in $\eta = \log\frac{p}{1-p}$*: the Jacobian is $dp/d\eta = p(1-p)$, so the posterior density in $\eta$ is $\propto p^{5}(1-p)^{9}$, maximized at $p = 5/14 = 0.35714$. (Grid search confirms $0.35714$.)

Two "most probable values" for the same posterior distribution: $0.33333$ and $0.35714$. Both are correct answers to well-posed questions; they are just different questions. The MLE, by contrast, is $0.30000$ in both parameterizations. And note the coincidence worth remembering: for a $\mathrm{Beta}(\alpha,\beta)$ posterior the logit-scale mode is exactly $\alpha/(\alpha+\beta) = 5/14$, the posterior **mean**. Posterior means and posterior medians are equivariant in the way one might want (means under affine maps, medians under all monotone maps); posterior *modes* are not equivariant at all.

The practical lesson is uncomfortable and mostly ignored. A MAP estimate of a variance parameter with an inverse-Gamma prior gives a different answer than a MAP estimate of the log-variance with the corresponding transformed prior, and deep learning code parameterizes scales as log-scales all the time. If you are doing MAP, the parameterization is part of the model. If you want something parameterization-free, use the full posterior, or a posterior mean, or the MLE.

### 8.8.3 Shrinkage and the James–Stein shock

Here is the setup that changed how statisticians think about high dimensions. Observe a single vector $X\sim\mathcal N(\theta, I_d)$ — think of $d$ separate, independent, unit-variance measurements of $d$ unrelated quantities. Estimate $\theta$ under total squared error $\mathbb E\|\hat\theta-\theta\|^2$.

The obvious estimator is $\hat\theta^{\mathrm{MLE}} = X$. It is unbiased, it is the MLE, it is the UMVUE, it attains the Cramér–Rao bound, and its risk is $\mathbb E\|X-\theta\|^2 = d$ for every $\theta$. It is, by every criterion in §8.5 and §8.7, optimal.

> **Definition 8.36 (Domination and admissibility).**
> Estimator $\hat\theta_1$ **dominates** $\hat\theta_2$ under risk $R(\theta,\cdot)$ if $R(\theta,\hat\theta_1)\le R(\theta,\hat\theta_2)$ for all $\theta$, with strict inequality for at least one $\theta$. An estimator is **admissible** if no estimator dominates it, and **inadmissible** otherwise.

Before the theorem, the two-line computation that makes it inevitable. It is the cleanest bias–variance trade in the subject, and it is worth doing by hand.

> **Proposition 8.37 (Optimal linear shrinkage).**
> Let $X\sim\mathcal N(\theta,I_d)$ and consider $\hat\theta_c := cX$ for a constant $c\in[0,1]$. Then
> $$\mathbb E_\theta\|cX-\theta\|^2 = \underbrace{c^2 d}_{\text{variance}} + \underbrace{(1-c)^2\|\theta\|^2}_{\text{squared bias}},$$
> minimized at
> $$c^\star = \frac{\|\theta\|^2}{\|\theta\|^2+d}, \qquad \text{with } \ \mathbb E_\theta\|c^\star X-\theta\|^2 = \frac{d\,\|\theta\|^2}{\|\theta\|^2+d} \;<\; d \ \text{ for every } \theta.$$
>
> *Proof.* Write $cX-\theta = c(X-\theta) + (c-1)\theta$. The first term has mean zero and covariance $c^2I_d$; the second is a constant. By Proposition 8.3 (in its vector form) the cross term vanishes and $\mathbb E\|cX-\theta\|^2 = c^2 d + (1-c)^2\|\theta\|^2$. This is a strictly convex quadratic in $c$; setting the derivative $2cd - 2(1-c)\|\theta\|^2$ to zero gives $c^\star$, and substituting gives the stated risk, which is $d\cdot\frac{\|\theta\|^2}{\|\theta\|^2+d} < d$. $\blacksquare$

Read what this says. **Some amount of shrinkage strictly beats no shrinkage at every $\theta$, in every dimension, including $d=1$.** There is no mystery and no high-dimensional magic; it is one quadratic minimization. The catch, and it is the whole catch, is that $c^\star$ depends on $\|\theta\|$, which you do not know. So $c^\star X$ is not an estimator — it is an *oracle*, a benchmark. The question James and Stein answered is whether you can estimate $c^\star$ from the data well enough that the gain survives. In $d=1$ or $2$ you cannot. From $d=3$ you can.

That is also the answer to "where does the shrinkage factor $1-\frac{d-2}{\|X\|^2}$ come from?" Since $\mathbb E\|X\|^2 = \|\theta\|^2+d$, the plug-in estimate of $c^\star = 1 - \frac{d}{\|\theta\|^2+d}$ is $1-\frac{d}{\|X\|^2}$; replacing $d$ by $d-2$ is the correction that makes the resulting risk formula exact and, as it turns out, uniformly favourable.

> **Theorem 8.38 (James–Stein).**
> Let $X\sim\mathcal N(\theta,I_d)$ with $d\ge3$ and define
> $$\hat\theta^{\mathrm{JS}} := \Big(1 - \frac{d-2}{\|X\|^2}\Big)X.$$
> Then for every $\theta\in\mathbb R^d$,
> $$\mathbb E_\theta\|\hat\theta^{\mathrm{JS}}-\theta\|^2 = d - (d-2)^2\,\mathbb E_\theta\Big[\frac{1}{\|X\|^2}\Big] \;<\; d = \mathbb E_\theta\|X-\theta\|^2 .$$
> Hence $X$ is inadmissible for $d\ge3$.
>
> *Proof (via Stein's identity).* Stein's lemma states that if $Z\sim\mathcal N(0,1)$ and $h$ is absolutely continuous with $\mathbb E|h'(Z)|<\infty$, then $\mathbb E[Z h(Z)] = \mathbb E[h'(Z)]$; the multivariate form gives $\mathbb E[(X_j-\theta_j)g_j(X)] = \mathbb E[\partial_j g_j(X)]$ for suitably regular $g$. Write $\hat\theta^{\mathrm{JS}} = X + g(X)$ with $g(x) = -(d-2)x/\|x\|^2$. Then
> $$\mathbb E\|\hat\theta^{\mathrm{JS}}-\theta\|^2 = \mathbb E\|X-\theta\|^2 + 2\mathbb E[(X-\theta)^\top g(X)] + \mathbb E\|g(X)\|^2 = d + 2\sum_j\mathbb E[\partial_j g_j(X)] + \mathbb E\|g(X)\|^2 .$$
> Now $\partial_j g_j(x) = -(d-2)\big(\frac{1}{\|x\|^2} - \frac{2x_j^2}{\|x\|^4}\big)$, so $\sum_j \partial_j g_j(x) = -(d-2)\big(\frac{d}{\|x\|^2}-\frac{2}{\|x\|^2}\big) = -\frac{(d-2)^2}{\|x\|^2}$, and $\|g(x)\|^2 = \frac{(d-2)^2}{\|x\|^2}$. Therefore
> $$\mathbb E\|\hat\theta^{\mathrm{JS}}-\theta\|^2 = d - 2(d-2)^2\mathbb E\Big[\frac1{\|X\|^2}\Big] + (d-2)^2\mathbb E\Big[\frac1{\|X\|^2}\Big] = d - (d-2)^2\mathbb E\Big[\frac{1}{\|X\|^2}\Big].$$
> For $d\ge3$, $\mathbb E[\|X\|^{-2}]$ is finite and strictly positive (the integrability at the origin needs $d\ge3$, which is exactly why the theorem fails for $d\le2$), so the risk is strictly less than $d$ everywhere. $\blacksquare$

Take a moment with this. The three coordinates might be the mass of an electron, the batting average of a baseball player, and the annual rainfall in Ulaanbaatar. They are independent. And yet: using the rainfall measurement to adjust your estimate of the electron mass *strictly reduces total squared error*, at every possible value of the parameters. Nothing in the setup connects them. The connection is imposed by the *loss*, which sums errors across coordinates and therefore lets a small gain in many coordinates pay for a small loss in a few.

Where does the gain come from? $\|X\|^2$ has expectation $\|\theta\|^2 + d$: the observed vector is systematically *longer* than the true one, because noise adds energy in every direction. Shrinking toward the origin corrects a real, systematic overestimate of length. In $d=1$ there is no such reliable inflation to exploit (and the shrinkage factor can be wildly wrong on a single coordinate); by $d=3$ the concentration of $\|X\|^2$ around $\|\theta\|^2+d$ is strong enough to make the correction pay.

**Worked Example 8.17 (James–Stein, simulated).**
$d=10$, $N=200{,}000$ replicates (seed 1618), $\theta = (\|\theta\|,0,\dots,0)$:

| $\|\theta\|$ | MSE(MLE) | MSE(JS) | MSE(JS$^+$) | closed form $d-(d-2)^2\mathbb E[1/\|X\|^2]$ | oracle $c^\star$ risk |
|---|---|---|---|---|---|
| $0$ | $10.0080$ | $2.0064$ | $1.2628$ | $2.0016$ | $0.0000$ |
| $1$ | $9.9909$ | $2.7308$ | $2.0764$ | $2.7436$ | $0.9091$ |
| $3$ | $9.9928$ | $5.9817$ | $5.8561$ | $5.9868$ | $4.7368$ |
| $10$ | $9.9866$ | $9.3869$ | $9.3869$ | $9.3967$ | $9.0909$ |

Here $\mathrm{JS}^+ := \big(1-\frac{d-2}{\|X\|^2}\big)_+X$ is the *positive-part* James–Stein estimator, which never flips the sign of $X$; it dominates plain JS (so plain JS is itself inadmissible!). The last column is the unattainable oracle risk $d\|\theta\|^2/(\|\theta\|^2+d)$ of Proposition 8.37. Three things to read off. The simulated risks match the closed form to Monte Carlo accuracy. The gain over the MLE is enormous near $\theta=0$ — a factor of five — and shrinks toward nothing as $\|\theta\|$ grows, because the shrinkage factor tends to $1$: **shrinkage helps most when the truth is near where you shrink toward, and never hurts.** And James–Stein tracks the oracle remarkably closely once $\|\theta\|$ is moderate ($5.86$ against $4.74$ at $\|\theta\|=3$; $9.39$ against $9.09$ at $\|\theta\|=10$), which is the precise sense in which estimating $c^\star$ from the data costs little in dimension $10$. At $\theta=0$ the oracle achieves risk $0$ and no estimator can match it, since the oracle is allowed to know the answer.

<figure>
<svg viewBox="0 0 700 320" style="max-width:100%;height:auto" role="img" aria-label="Scatter of noisy estimates being pulled toward a common prior mean, with risk curves">
  <circle cx="200" cy="150" r="95" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4"/>
  <circle cx="200" cy="150" r="46" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4"/>
  <circle cx="200" cy="150" r="3.5" fill="currentColor"/>
  <text x="200" y="140" text-anchor="middle" font-size="12.5" fill="currentColor">prior mean</text>
  <g stroke="#3b82f6" stroke-width="1.8" fill="#3b82f6">
    <circle cx="288" cy="96" r="4"/><circle cx="120" cy="88" r="4"/><circle cx="272" cy="216" r="4"/>
    <circle cx="112" cy="204" r="4"/><circle cx="200" cy="52" r="4"/><circle cx="296" cy="160" r="4"/>
  </g>
  <g stroke="#10b981" stroke-width="1.6" fill="#10b981">
    <circle cx="244" cy="123" r="3.6"/><circle cx="160" cy="119" r="3.6"/><circle cx="236" cy="183" r="3.6"/>
    <circle cx="156" cy="177" r="3.6"/><circle cx="200" cy="101" r="3.6"/><circle cx="248" cy="155" r="3.6"/>
  </g>
  <g stroke="#f59e0b" stroke-width="1.3" marker-end="url(#ah)">
    <line x1="283" y1="99" x2="250" y2="120"/><line x1="125" y1="92" x2="156" y2="115"/>
    <line x1="268" y1="212" x2="240" y2="187"/><line x1="117" y1="200" x2="152" y2="181"/>
    <line x1="200" y1="57" x2="200" y2="96"/><line x1="291" y1="159" x2="253" y2="156"/>
  </g>
  <defs><marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#f59e0b"/></marker></defs>
  <text x="330" y="70" font-size="13" fill="#3b82f6">raw MLEs (unbiased, spread out)</text>
  <text x="330" y="90" font-size="13" fill="#10b981">shrunken estimates (biased, closer)</text>
  <text x="330" y="110" font-size="13" fill="#f59e0b">shrinkage factor 1 − (d−2)/‖X‖²</text>
  <text x="330" y="140" font-size="13" fill="currentColor">‖X‖² ≈ ‖θ‖² + d: the observed</text>
  <text x="330" y="158" font-size="13" fill="currentColor">vector is systematically too long,</text>
  <text x="330" y="176" font-size="13" fill="currentColor">so pulling in corrects a real bias.</text>
  <text x="330" y="206" font-size="13" fill="currentColor">Total squared error falls for every θ</text>
  <text x="330" y="224" font-size="13" fill="currentColor">once d ≥ 3 (Theorem 8.38).</text>
  <text x="350" y="272" text-anchor="middle" font-size="12.5" fill="currentColor">Figure 8.4 — Shrinkage as pulling every estimate toward a common point. Each individual estimate becomes biased;</text>
  <text x="350" y="289" text-anchor="middle" font-size="12.5" fill="currentColor">the sum of squared errors across coordinates goes down. Weight decay, ridge, and empirical-Bayes hierarchical</text>
  <text x="350" y="306" text-anchor="middle" font-size="12.5" fill="currentColor">models are all instances; the prior mean is wherever your penalty says "cheap".</text>
</svg>
</figure>

**The bridge to regularization.** Suppose you put a prior $\theta\sim\mathcal N(0,\tau^2 I_d)$ on the mean vector. The posterior mean is $\frac{\tau^2}{\tau^2+1}X$ — linear shrinkage toward zero. James–Stein is what you get if you *estimate* $\tau^2$ from the data itself: $\mathbb E\|X\|^2 = d(\tau^2+1)$ suggests $\widehat{\frac{\tau^2}{\tau^2+1}} = 1 - \frac{d}{\|X\|^2}$, and the $d-2$ in Theorem 8.38 is the exactly-tuned version. So James–Stein is **empirical Bayes**: a prior whose hyperparameter is estimated from the data. And this is precisely what tuning weight decay by cross-validation is. The theory says shrinkage must help in high dimensions; the practice says pick $\lambda$ by validation. They are the same act.

Forward pointer: Chapter 12 develops high-dimensional phenomena systematically — concentration of measure, why $\|X\|^2$ is so predictable, and why nearly all of the volume of a high-dimensional ball is near its surface. James–Stein is the first sighting of the general principle that **high dimensions are not low dimensions with more coordinates**; they have their own geometry, and estimators that are optimal in $d=1$ can be beaten everywhere in $d=3$.

> **Common Confusion 8.7.**
> *"James–Stein must be cheating — it's using information from unrelated coordinates. Surely if I only care about coordinate 1, the MLE is still best?"*
>
> Correct, and important. James–Stein dominates under *total* squared error $\sum_j(\hat\theta_j-\theta_j)^2$. For the single coordinate $\theta_1$ alone, JS can be strictly worse than $X_1$ at some $\theta$ — you buy a large gain on nine coordinates by accepting a small loss on one, and if the one you care about is the loser, you lose. This is exactly the situation with multi-task learning and with regularizing a shared representation: total performance improves, individual tasks can degrade. It is also why "the model got better on average" is not an answer to "did it get worse on my subgroup". Admissibility is a statement about an aggregate risk, and aggregates hide redistribution.

> **Self-check 8.8.**
> 1. Write the penalty corresponding to a $\mathcal N(0,\tau^2)$ prior and to a $\mathrm{Laplace}(0,b)$ prior. Which produces exact zeros, and what feature of the density is responsible?
> 2. Why is the MLE equivariant under reparameterization but MAP not? Name the culprit term.
> 3. State the risk of the James–Stein estimator and identify where $d\ge3$ is used in the proof.
> 4. Your colleague reports that shrinkage improved total MSE across 50 tasks. What can you *not* conclude about task 17?

---

## 8.9 The EM algorithm

### 8.9.1 EM is the ELBO, twice

Chapter 2 proved an identity that looked at the time like a curiosity. Let $x$ be observed, $z$ a latent variable, $p_\theta(x,z)$ a joint density w.r.t. a product base measure, and $q$ any density over $z$ with $q \ll p_\theta(\cdot\mid x)$. Then
$$\log p_\theta(x) \;=\; \underbrace{\mathbb E_{q}\Big[\log\frac{p_\theta(x,Z)}{q(Z)}\Big]}_{=:\ \mathcal L(q,\theta),\ \text{the ELBO}} \;+\; D_{\mathrm{KL}}\big(q \,\big\|\, p_\theta(\cdot\mid x)\big).$$
Since KL is nonnegative (Ch. 4), $\mathcal L(q,\theta) \le \log p_\theta(x)$ for every $q$ — a *lower bound* on the log-likelihood, tight exactly when $q = p_\theta(\cdot\mid x)$.

The identity has two free arguments. Maximize alternately in each. That is EM.

> **The EM algorithm.** Given $\theta^{(t)}$:
> **E-step.** $q^{(t+1)} := \arg\max_q \mathcal L(q,\theta^{(t)}) = p_{\theta^{(t)}}(\cdot\mid x)$ — the posterior over latents at the current parameters. The bound becomes tight: $\mathcal L(q^{(t+1)},\theta^{(t)}) = \log p_{\theta^{(t)}}(x)$.
> **M-step.** $\theta^{(t+1)} := \arg\max_\theta \mathcal L(q^{(t+1)},\theta) = \arg\max_\theta \mathbb E_{q^{(t+1)}}[\log p_\theta(x,Z)]$, since the $-\mathbb E_q[\log q]$ term does not involve $\theta$. The quantity being maximized is traditionally called $Q(\theta\mid\theta^{(t)})$.

> **Theorem 8.39 (Monotone likelihood improvement).**
> The EM iterates satisfy $\log p_{\theta^{(t+1)}}(x) \ge \log p_{\theta^{(t)}}(x)$ for every $t$, with equality only if $\theta^{(t+1)}$ fails to increase $Q(\cdot\mid\theta^{(t)})$.
>
> *Proof.* Chain the two steps through the identity:
> $$\log p_{\theta^{(t+1)}}(x) \;\overset{(a)}{\ge}\; \mathcal L\big(q^{(t+1)},\theta^{(t+1)}\big) \;\overset{(b)}{\ge}\; \mathcal L\big(q^{(t+1)},\theta^{(t)}\big) \;\overset{(c)}{=}\; \log p_{\theta^{(t)}}(x).$$
> Step (a) is the ELBO inequality at the new $\theta$ with the old $q$ (the bound holds for *any* $q$). Step (b) is the M-step: $\theta^{(t+1)}$ maximizes $\mathcal L(q^{(t+1)},\cdot)$, so it does at least as well as $\theta^{(t)}$. Step (c) is the E-step making the bound tight at $\theta^{(t)}$, i.e. $D_{\mathrm{KL}}(q^{(t+1)}\|p_{\theta^{(t)}}(\cdot\mid x))=0$. $\blacksquare$

Three lines, one inequality used twice, and the guarantee is unconditional: EM never decreases the observed-data likelihood, regardless of initialization, step size, or model. That is a stronger guarantee than gradient ascent gives you without a line search.

What it does *not* guarantee: convergence to a global maximum, convergence to any maximum at all (it can stall at a saddle), or convergence in parameter space (the likelihood values converge if bounded; the parameters need not, and in the mixture case they can run off to a spike). Read Theorem 8.39 for exactly what it says.

<figure>
<svg viewBox="0 0 700 330" style="max-width:100%;height:auto" role="img" aria-label="EM as alternating tightening and maximizing of a lower bound on the log-likelihood">
  <line x1="60" y1="250" x2="650" y2="250" stroke="currentColor" stroke-width="1.5"/>
  <text x="355" y="272" text-anchor="middle" font-size="13" fill="currentColor">θ</text>
  <text x="34" y="150" font-size="13" fill="currentColor" transform="rotate(-90 34 150)">log p(x | θ)</text>
  <path d="M60 215 Q 200 195 300 120 Q 420 35 650 60" fill="none" stroke="currentColor" stroke-width="2.5"/>
  <text x="560" y="40" font-size="13" fill="currentColor">log-likelihood</text>
  <path d="M110 245 Q 170 190 230 205 Q 260 215 300 245" fill="none" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5 3"/>
  <path d="M230 245 Q 300 135 380 105 Q 440 90 500 160" fill="none" stroke="#10b981" stroke-width="2" stroke-dasharray="5 3"/>
  <circle cx="170" cy="200" r="4.5" fill="#3b82f6"/>
  <circle cx="170" cy="200" r="8" fill="none" stroke="#3b82f6" stroke-width="1"/>
  <circle cx="230" cy="203" r="4.5" fill="#3b82f6"/>
  <circle cx="230" cy="203" r="8" fill="none" stroke="#10b981" stroke-width="1"/>
  <circle cx="360" cy="103" r="4.5" fill="#10b981"/>
  <line x1="170" y1="250" x2="170" y2="196" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3 3"/>
  <line x1="230" y1="250" x2="230" y2="199" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3 3"/>
  <line x1="360" y1="250" x2="360" y2="99" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3 3"/>
  <text x="163" y="266" font-size="12.5" fill="currentColor">θ⁽ᵗ⁾</text>
  <text x="220" y="266" font-size="12.5" fill="currentColor">θ⁽ᵗ⁺¹⁾</text>
  <text x="348" y="266" font-size="12.5" fill="currentColor">θ⁽ᵗ⁺²⁾</text>
  <text x="95" y="175" font-size="12.5" fill="#3b82f6">E-step: build bound</text>
  <text x="95" y="192" font-size="12.5" fill="#3b82f6">tight at θ⁽ᵗ⁾</text>
  <text x="395" y="140" font-size="12.5" fill="#10b981">M-step: maximize the bound;</text>
  <text x="395" y="157" font-size="12.5" fill="#10b981">new bound is tight at θ⁽ᵗ⁺¹⁾</text>
  <text x="350" y="298" text-anchor="middle" font-size="12.5" fill="currentColor">Figure 8.5 — EM alternates: the E-step constructs a lower bound touching the log-likelihood at the current θ (KL = 0),</text>
  <text x="350" y="315" text-anchor="middle" font-size="12.5" fill="currentColor">the M-step jumps to that bound's maximum. Since the bound touches from below, the likelihood cannot decrease</text>
  <text x="350" y="329" text-anchor="middle" font-size="12.5" fill="currentColor">(Theorem 8.39). Variational inference uses the same picture with a restricted q, so the bound never touches.</text>
</svg>
</figure>

### 8.9.2 EM for a Gaussian mixture

> **Proposition 8.40 (EM updates for a $K$-component Gaussian mixture).**
> Let $p_\theta(x) = \sum_{k=1}^K w_k\,\phi_{m_k,v_k}(x)$ with $\sum_k w_k=1$, $w_k>0$, and latent $Z_i\in\{1,\dots,K\}$ indicating the component of $x_i$. Given $\theta^{(t)}$, define the **responsibilities**
> $$r_{ik} := P_{\theta^{(t)}}(Z_i=k\mid x_i) = \frac{w_k^{(t)}\phi_{m_k^{(t)},v_k^{(t)}}(x_i)}{\sum_{j} w_j^{(t)}\phi_{m_j^{(t)},v_j^{(t)}}(x_i)}.$$
> Then the M-step has the closed form
> $$w_k^{(t+1)} = \frac{1}{n}\sum_{i=1}^n r_{ik},\qquad m_k^{(t+1)} = \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}},\qquad v_k^{(t+1)} = \frac{\sum_i r_{ik}\big(x_i-m_k^{(t+1)}\big)^2}{\sum_i r_{ik}}.$$
>
> *Proof.* The E-step is Bayes' rule (Proposition 1.12) with prior $w^{(t)}$ over components and likelihood $\phi$. For the M-step, the complete-data log-likelihood is $\sum_i\sum_k \mathbf 1\{Z_i=k\}\big(\log w_k + \log\phi_{m_k,v_k}(x_i)\big)$, so
> $$Q(\theta\mid\theta^{(t)}) = \sum_{i}\sum_k r_{ik}\Big(\log w_k -\tfrac12\log(2\pi v_k) - \tfrac{(x_i-m_k)^2}{2v_k}\Big).$$
> This separates: maximizing over $w$ subject to $\sum_k w_k=1$ is maximizing $\sum_k (\sum_i r_{ik})\log w_k$, whose solution by Lagrange multipliers (or by recognizing it as a KL minimization, Ch. 4) is $w_k\propto\sum_i r_{ik}$. Maximizing over $(m_k,v_k)$ is a weighted version of Worked Example 8.4, giving the weighted mean and weighted variance. $\blacksquare$

Compare with hard assignment: if you replace $r_{ik}$ by $\mathbf 1\{k = \arg\max_j r_{ij}\}$ and hold both the variances $v_k$ and the mixing weights $w_k$ fixed and equal across components (so that the arg max is just the nearest centroid), the updates become exactly **$k$-means**. $k$-means is EM with a zero-temperature E-step. That also explains $k$-means' failure modes (sensitivity to initialization, spherical clusters only) as inherited from the mixture model it approximates.

**Worked Example 8.18 (EM on real numbers, including a run that collapses).**
Generate $n=300$ points from $0.35\,\mathcal N(-2, 0.8^2) + 0.65\,\mathcal N(1.5, 1.2^2)$ (seed 90210). Run EM from $w=0.5$, $(m_1,\sigma_1)=(-1,1)$, $(m_2,\sigma_2)=(1,1)$.

Log-likelihood by iteration (first twelve values, starting from the initialization):
$$-694.7263,\ -615.5364,\ -614.9389,\ -614.5356,\ -614.1977,\ -613.8829,\ -613.5813,\ -613.2926,\ -613.0189,\ -612.7628,\ -612.5266,\ -612.3121$$
After 40 iterations, $-611.0872$. The increments are $+79.1899, +0.5976, +0.4033, +0.3379, +0.3148, +0.3016,\dots$ — monotone increase at every step, verified numerically for all 40 iterations, exactly as Theorem 8.39 promises, with a huge first step and then slow linear convergence (EM's convergence rate is linear, with a factor governed by the fraction of missing information; this is why practitioners switch to a quasi-Newton method near the optimum). Fitted parameters: $\hat w = 0.3001$, $(\hat m_1,\hat\sigma_1) = (-2.0386,\, 0.8034)$, $(\hat m_2,\hat\sigma_2)=(1.4401,\,1.3341)$. Against the truth $(0.35;\,-2,0.8;\,1.5,1.2)$ that is a good fit at $n=300$.

Now the degenerate run. Initialize with $m_1 = x_1 = 1.5902$ and $\sigma_1 = 0.05$ — a near-spike sitting on a data point, exactly the configuration of Proposition 8.11. Log-likelihoods: $-802.596, -629.290, -627.363, -626.979, -626.833, -626.748, -626.683, -626.629,\dots$, converging to $-626.3692$. Still monotone — Theorem 8.39 holds unconditionally — but converging to a *different, much worse* local optimum: $\hat w = 0.077228$, $(\hat m_1,\hat\sigma_1)=(1.6099,\,0.165492)$, $(\hat m_2,\hat\sigma_2)=(0.2947,\,2.0441)$. Component 1 has captured $7.7\%$ of the mass in a narrow spike of width $0.17$ around a cluster of nearby points; component 2 is a wide blob absorbing everything else. The log-likelihood gap is $15.3$ nats below the good run.

This is what "run EM from several random restarts" is protecting you against, and it is the same pathology as a neural network converging to a degenerate solution that memorizes a few points. Note that with a slightly more aggressive initialization ($\sigma_1 = 10^{-3}$, say) the run would march toward $\sigma_1\to0$ and the likelihood toward $+\infty$; the algorithm would report a triumphant, monotone increase all the way into nonsense.

### 8.9.3 EM and variational inference are the same construction

Look at the E-step again. It sets $q = p_\theta(\cdot\mid x)$ exactly, which requires that you can compute the posterior over latents in closed form. For a finite mixture you can — it is a $K$-way softmax. For a deep latent-variable model with $z\in\mathbb R^{50}$ and a neural decoder, you cannot.

**Variational inference restricts $q$ to a tractable family $\mathcal Q$** and does the E-step approximately:
$$q^{(t+1)} := \arg\max_{q\in\mathcal Q} \mathcal L(q,\theta^{(t)}) = \arg\min_{q\in\mathcal Q} D_{\mathrm{KL}}\big(q\,\|\,p_{\theta^{(t)}}(\cdot\mid x)\big).$$
Everything else is unchanged. If $p_\theta(\cdot\mid x)\in\mathcal Q$ this reduces exactly to EM. If not, the bound never becomes tight, step (c) in the proof of Theorem 8.39 becomes an inequality in the *wrong* direction, and the monotone-likelihood guarantee is lost. What survives is monotonicity of the *bound*: $\mathcal L(q^{(t+1)},\theta^{(t+1)}) \ge \mathcal L(q^{(t)},\theta^{(t)})$, since each step maximizes over one argument. You are guaranteed to improve a lower bound on the log-likelihood, not the log-likelihood.

That is the exact conceptual position of a variational autoencoder: the encoder is an amortized approximate E-step (one shared network computing $q_\phi(z\mid x)$ for all $x$, instead of a separate optimization per data point), the decoder is the M-step model, and the training objective is the ELBO. The gap $D_{\mathrm{KL}}(q_\phi(\cdot\mid x)\|p_\theta(\cdot\mid x))$ is invisible to the optimizer — it never appears in the loss — which is why reported ELBOs are not log-likelihoods and why comparing a VAE's ELBO to an autoregressive model's exact log-likelihood is comparing a lower bound to a value. Chapter 13 develops all of this properly.

> **Common Confusion 8.8.**
> *"EM converges to the MLE."*
>
> EM converges to a point where the likelihood stops increasing. Under regularity conditions (Wu, 1983) limit points of the EM sequence are stationary points of the likelihood — which includes local maxima, saddle points, and, in the mixture case, the boundary spikes of Proposition 8.11. In practice EM run on a $K$-component mixture with random initialization finds different local optima on different runs, and the spread can be large. The right mental model is "coordinate ascent on a non-concave surface with a guarantee of no backsliding", not "an algorithm that solves the problem".

> **Self-check 8.9.**
> 1. Write the three-step chain proving Theorem 8.39 and say which step uses the E-step and which uses the M-step.
> 2. Why does the $-\mathbb E_q[\log q]$ term drop out of the M-step?
> 3. What exactly is lost when the E-step is approximate, and what is retained?
> 4. In Worked Example 8.18, the degenerate run still had monotonically increasing likelihood. Why is that not a contradiction?

---

## 8.10 Bias, variance, and overparameterization — an honest section

### 8.10.1 The classical picture, and what it actually assumes

Here is the story every course tells. Index a family of models by complexity $p$ (polynomial degree, number of parameters, inverse regularization strength). As $p$ grows, bias falls — richer models can represent the truth better — and variance rises, because more parameters must be estimated from the same $n$ data points. By Proposition 8.3 the risk is bias$^2$ + variance, so it falls, bottoms out, and rises: a U-curve. Choose $p$ at the bottom. Anything to the right is *overfitting*.

That story is correct, and it is a theorem, in a specific regime. Take fixed-design linear regression with $p \le n$, Gaussian noise of variance $\sigma^2$, and the *ordinary least squares* fit. The average variance of the fitted values, $\frac1n\sum_i\operatorname{Var}(\hat y_i) = \frac{\sigma^2}{n}\operatorname{tr}\big(X(X^\top X)^{-1}X^\top\big)$, is exactly $\sigma^2 p/n$ — linear in $p$, with a slope set by the noise. It genuinely blows up as $p \to n$. Add a bias term that decreases in $p$, and you get the U.

The assumptions doing the work are worth naming, because modern practice violates every one of them:
1. $p \le n$, so the least-squares problem has a unique solution.
2. The fit is *the* least-squares solution, with no further selection among the many possible fits.
3. "Complexity" is measured by parameter count.
4. The noise is irreducible and must be fit or absorbed.

### 8.10.2 The interpolation regime and double descent

When $p > n$ the linear system is underdetermined: infinitely many parameter vectors fit the training data exactly. Assumption 2 is now a *choice*, and gradient descent from zero initialization makes a specific one — it converges to the minimum-$\ell_2$-norm interpolator. That choice is a regularizer, imposed by the algorithm rather than the objective: **implicit regularization**.

And as $p$ grows past $n$, the minimum-norm interpolator gets *better*, because a larger feature space contains interpolating solutions of smaller norm. The risk curve therefore falls again. Combined with the classical U to the left of $p=n$ and the blow-up at $p = n$ (where the system is exactly determined, the design matrix is near-singular, and the unique interpolator has enormous norm), you get **double descent**: down, up, spike, down again.

I ran this. Random-features regression: $n=100$ training points, $d=20$ input dimensions, target $y = x^\top\beta + \varepsilon$ with $\varepsilon\sim\mathcal N(0,0.5^2)$, features $\tanh(W^\top x)$ with $W$ random and $p$ columns, fit by minimum-norm least squares (`pinv`), test MSE on $5000$ fresh points, averaged over $20$ draws of $(W, \text{train}, \text{test})$, seed 112358:

| $p$ | 2 | 10 | 20 | 40 | 70 | 90 | 98 | **100** | 102 | 110 | 130 | 200 | 400 | 800 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ridgeless test MSE | $0.657$ | $0.537$ | $0.412$ | $0.488$ | $0.914$ | $3.171$ | $15.35$ | $\mathbf{2285.2}$ | $18.00$ | $4.074$ | $1.327$ | $0.716$ | $0.459$ | $0.406$ | $\mathbf{0.358}$ |
| with ridge $\lambda=1$ | $1.092$ | $0.739$ | $0.554$ | $0.433$ | $0.425$ | $0.436$ | $0.438$ | $0.440$ | $0.447$ | $0.437$ | $0.442$ | $0.448$ | $0.414$ | $0.377$ | $0.366$ |

Read the first row left to right. Classical U: risk falls to $0.412$ at $p=20$, then rises. Interpolation threshold: at $p = n = 100$ the test MSE is $2285$ — a catastrophe, three orders of magnitude worse than the classical optimum. Second descent: past the threshold the risk falls monotonically, and by $p = 2000$ it reaches $0.358$, **better than the classical minimum at $p=20$**. Twenty times more parameters than data points, zero training error, and the best test performance in the table.

The second row is the honest counterpoint. With a modest ridge penalty the peak disappears entirely and the curve is nearly flat from $p=40$ onward, bottoming at $0.366$ — essentially the same value the ridgeless model reaches at $p=2000$. **Double descent is largely a story about un-regularized interpolation.** With appropriate regularization, "more parameters" mostly just does not hurt; the dramatic peak is an artifact of insisting on interpolating with an ill-conditioned feature matrix. That is a fair summary of where the field has landed, and it deserves to be said more often than it is.

<figure>
<svg viewBox="0 0 700 350" style="max-width:100%;height:auto" role="img" aria-label="Double descent risk curve versus number of features, with and without ridge regularization">
  <line x1="90" y1="300" x2="650" y2="300" stroke="currentColor" stroke-width="1.5"/>
  <line x1="90" y1="300" x2="90" y2="35" stroke="currentColor" stroke-width="1.5"/>
  <line x1="384.5" y1="300" x2="384.5" y2="45" stroke="currentColor" stroke-width="1" stroke-dasharray="5 4"/>
  <text x="384.5" y="38" text-anchor="middle" font-size="12.5" fill="currentColor">p = n</text>
  <polyline fill="none" stroke="#ef4444" stroke-width="2.5" points="90.0,269.3 159.0,271.1 211.2,275.7 263.3,284.2 315.5,278.8 357.6,258.8 376.6,219.0 383.0,168.7 384.5,64.7 386.0,163.6 391.7,211.0 404.2,246.9 436.7,266.6 488.8,280.7 541.0,284.7 610.0,288.7"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="90.0,253.1 159.0,256.1 211.2,265.6 263.3,274.7 315.5,282.6 357.6,283.2 376.6,282.4 383.0,282.3 384.5,282.1 386.0,281.6 391.7,282.3 404.2,282.0 436.7,281.5 488.8,284.0 541.0,287.1 610.0,288.0"/>
  <text x="410" y="90" font-size="13" fill="#ef4444">ridgeless (min-norm interpolator)</text>
  <text x="410" y="255" font-size="13" fill="#10b981">ridge λ = 1</text>
  <text x="190" y="255" font-size="12.5" fill="currentColor">classical U</text>
  <text x="470" y="230" font-size="12.5" fill="currentColor">second descent</text>
  <text x="370" y="322" text-anchor="middle" font-size="13" fill="currentColor">number of random features p  (log scale, 2 → 2000)</text>
  <text x="40" y="180" font-size="13" fill="currentColor" transform="rotate(-90 40 180)">test MSE (log scale)</text>
  <text x="350" y="342" text-anchor="middle" font-size="12.5" fill="currentColor">Figure 8.6 — Measured double descent: n = 100, tanh random features, 20 replicates (seed 112358). The ridgeless curve peaks at p = n with test MSE 2285; past the threshold it descends below the classical optimum. Ridge removes the peak almost entirely.</text>
</svg>
</figure>

### 8.10.3 What is actually going on, and how much is settled

Some of this is understood. Some of it is not. Here is my reading of the line, with the caveat that this is a fast-moving area and you should check the literature rather than trust a textbook from any particular year.

**Settled: parameter count is the wrong complexity measure.** The variance of a fitted predictor is controlled by things like the norm of the solution, the effective rank of the feature covariance, and the margin — not by $p$. In ridge regression the relevant quantity is the *effective degrees of freedom* $\operatorname{tr}\big(X(X^\top X+\lambda I)^{-1}X^\top\big)$, which is $\le \min(n,p)$ and can be far smaller. A model with $10^{11}$ parameters and strong implicit regularization can have far fewer effective degrees of freedom than a small model fit without constraint. This has been true since ridge regression was invented; overparameterized networks made it impossible to ignore.

**Settled: interpolating noisy data need not be fatal.** The name is **benign overfitting** (Bartlett, Long, Lugosi, Tsigler, 2020). In linear regression with a suitable spectrum of the feature covariance — many small eigenvalues, so that noise can be absorbed into directions that barely affect prediction — the minimum-norm interpolator is consistent despite fitting the noise exactly. The precise spectral conditions are known for linear and kernel regression. Whether real networks satisfy analogues of them is not known.

**Fairly well understood: the peak at $p=n$.** It comes from the smallest singular value of the feature matrix approaching zero, making the interpolator's norm explode. This is conditioning, not statistics, and any regularizer — explicit ridge, early stopping, noise in SGD — smooths it away. Hastie, Montanari, Rosset and Tibshirani (2022) give exact asymptotic risk curves for linear and random-features regression in the proportional regime $p/n\to\gamma$, and the shapes match simulations like the one above closely.

**Active research: implicit regularization, and whether double descent is the right picture at all.** For linear models with gradient descent from zero the implicit bias is minimum $\ell_2$ norm; for logistic regression on separable data it is the maximum-margin direction; for some matrix factorizations it is nuclear norm. For a transformer trained with Adam, weight decay, dropout, augmentation, schedules and clipping, nobody can write the induced regularizer down, so "SGD implicitly regularizes" is established in specific settings and extrapolated elsewhere. Nakkiran et al. (2021) documented double descent in model size, data size *and* training time for realistic networks, which is striking — but the epoch-wise version interacts with optimization in complicated ways, the model-size version depends heavily on regularization and label noise, and the effect can often be tuned away. Treat published curves as evidence about the settings that produced them.

**Where theory is commonly misapplied.** Three specific things I would push back on if a paper said them.

*"Our model has more parameters than data, so classical statistics doesn't apply."* Classical statistics did not say more parameters is bad; it said more *effective* complexity relative to $n$ is bad, and it said it under assumptions that are checkable. The Neyman–Scott example (Worked Example 8.12) is a classical result about exactly this regime, and it says the MLE can be badly inconsistent when parameters grow with $n$. Sometimes overparameterization is benign; sometimes it is Neyman–Scott. The difference is structural, not a matter of counting.

*"Test loss went down when we added parameters, so we're in the second descent."* Maybe. Or you are in the first descent and have not reached the interpolation threshold. Or your regularization changed. The distinguishing feature of the second descent is that training error is (near) zero throughout, which is checkable and rarely checked.

*"Flat minima generalize better."* Flatness in *parameter space* is not reparameterization invariant — you can rescale a layer up and the next down, leaving the function unchanged while changing the Hessian's eigenvalues arbitrarily (Dinh et al., 2017). This is Proposition 8.18 again: curvature is a quadratic form that transforms with a Jacobian, and comparing raw curvatures across parameterizations is a units error. Fisher–Rao-based measures are better behaved for the same reason natural gradient is.

### 8.10.4 What to take away

Bias–variance is still exactly true — Proposition 8.3 is an algebraic identity and cannot fail. What fails is the *mapping* from "number of parameters" to "variance". Estimation error depends on how much freedom the fitting procedure actually exercises, and modern procedures exercise far less than their parameter count suggests, because of penalties, early stopping, architecture, and the geometry of the loss landscape.

The right instinct, and the one this chapter has been building toward: variance is what you pay for fitting the noise, and shrinkage — explicit or implicit — is what you use to pay less. James–Stein (§8.8) said that in $d\ge3$ shrinkage is not optional but mandatory for optimality. Double descent says something compatible: in very high dimensions, shrinkage supplied by the algorithm can be enough that the raw parameter count stops being informative. Neither says the trade-off went away.

> **Self-check 8.10.**
> 1. Which four assumptions underlie the classical U-shaped risk curve, and which does an overparameterized network violate?
> 2. In the measured table, what is the test MSE at $p=n=100$, and what causes the spike?
> 3. Why does adding a ridge penalty remove the peak almost entirely, and what does that suggest about how much of double descent is "statistical"?
> 4. Give the one-sentence reason why comparing Hessian eigenvalues ("flatness") across two parameterizations of the same function is a units error.

---

## 8.11 Results used but not proved

| Result | Used in | Where to find a proof |
|---|---|---|
| Differentiation under the integral sign (DCT-based) | Propositions 8.14, 8.17; Theorems 8.15, 8.19, 8.26 | Chapter 2 of this tutorial; Folland, *Real Analysis*, Thm 2.27 |
| Radon–Nikodym theorem and the chain rule for densities | Definition 8.8 and §8.3.1(a) | Chapter 2; Durrett, *Probability*, §4.1 |
| ELBO identity $\log p_\theta(x) = \mathcal L(q,\theta) + D_{\mathrm{KL}}(q\|p_\theta(\cdot|x))$ | §8.9 throughout | Chapter 2 |
| Exponential-family identities $\nabla A = \mathbb E[T]$, $\nabla^2 A = \operatorname{Var}(T)$ | Propositions 8.10, 8.17 | Chapter 3 |
| Fisher–Neyman factorization; completeness of full-rank exponential families | §8.7, Definition 8.31, Theorem 8.32 | Chapter 3; Lehmann & Casella, *Theory of Point Estimation*, Thms 1.6.5, 4.3.1 |
| Jensen's inequality; $D_{\mathrm{KL}}\ge0$ with equality iff $P=Q$ | Theorem 8.24 Step 1; §8.3.3; §8.9 | Chapter 4 |
| Cauchy–Schwarz in $L^2$ | Theorem 8.19 | Chapter 4 |
| Cochran-type Gaussian facts: $\sum_i(X_i-\bar X)^2 \sim \sigma^2\chi^2_{n-1}$, independent of $\bar X$ | Worked Examples 8.1, 8.11 | Chapter 5 |
| Tower property; law of total variance; conditional Jensen | Theorem 8.30 | Chapter 6 |
| SLLN, multivariate CLT, Slutsky, delta method, continuous mapping | Proposition 8.7; Theorems 8.24, 8.25, 8.28; Corollary 8.26; §8.3.3 | Chapter 7 |
| Markov's inequality | Proposition 8.5 | Chapter 4 |
| Uniform law of large numbers over a compact parameter set (bracketing/covering argument) | Theorem 8.24 Step 2; Theorem 8.25 Term 2 | van der Vaart, *Asymptotic Statistics*, Thm 5.7 and §19; Chapter 12 of this tutorial develops the tools |
| Taylor's theorem with mean-value / integral remainder for vector-valued maps | Theorem 8.25 | Any analysis text; Rudin, *Principles*, Thm 5.15 (scalar), or Hörmander for the multivariate form |
| Measurable selection: existence of a measurable maximizer of a random criterion | Definition 8.1 discussion; Definition 8.8 | Aliprantis & Border, *Infinite Dimensional Analysis*, Thm 18.19 |
| Stein's lemma $\mathbb E[Zh(Z)] = \mathbb E[h'(Z)]$ and its multivariate form | Theorem 8.38 | Lehmann & Casella, Lemma 5.2.1; original: Stein (1981) |
| Implicit-function/envelope differentiation of $\hat\lambda(\psi)$, giving the Schur-complement curvature of the profile likelihood | Definition 8.29 | Barndorff-Nielsen & Cox, *Inference and Asymptotics*, §3.4; Murphy & van der Vaart (2000), *JASA* 95 |
| The Cauchy location log-likelihood can have up to $n$ local maxima | Worked Example 8.9 | Reeds (1985), *Ann. Statist.* 13(2); Barnett (1966), *Biometrika* 53 |
| Uniqueness of power-series coefficients (used to rule out an unbiased estimator of $1/\lambda$) | Exercise A5 | Any complex-analysis text; Rudin, *Principles*, Thm 8.1 |
| Poisson conditioning: $(X_1,\dots,X_n)\mid\sum X_i = t \sim \mathrm{Multinomial}(t;1/n,\dots,1/n)$ | Worked Example 8.15 | Casella & Berger, *Statistical Inference*, Thm 4.6.7 (direct computation) |
| Convergence of EM limit points to stationary points of the likelihood | §8.9.2, Common Confusion 8.8 | Wu (1983), *Annals of Statistics* 11(1) |
| Le Cam–Hájek convolution / local asymptotic minimax theorems (superefficiency) | §8.6.3 caveat (1) | van der Vaart, *Asymptotic Statistics*, Ch. 8 |
| Exact asymptotic risk curves for random-features regression in the proportional regime | §8.10.3 | Hastie, Montanari, Rosset & Tibshirani (2022), *Annals of Statistics* 50(2) |
| Benign overfitting conditions for minimum-norm linear interpolators | §8.10.3 | Bartlett, Long, Lugosi & Tsigler (2020), *PNAS* 117(48) |
| Schur complement characterization of PSD block matrices | Corollary 8.20 | Horn & Johnson, *Matrix Analysis*, §7.7 |

---

## 8.12 Chapter summary

1. An estimator is a measurable function of the data, hence a random variable on $(\mathcal X^n,\mathcal A^{\otimes n},P_\theta^{\otimes n})$. Its sampling distribution is the object of study; "the variance of the estimate" is a category error.
2. MSE $=$ bias$^2$ $+$ variance (Proposition 8.3), a Pythagorean identity in $L^2$. Unbiasedness zeroes one term at the cost of the other and is not a goal: for a Gaussian variance the MSE-optimal divisor is $n+1$, and for $\mathrm{Unif}[0,\theta]$ the MSE-optimal multiple of $X_{(n)}$ is $\frac{n+2}{n+1}$ — biased in both cases, in opposite directions.
3. Method of moments is cheap and likelihood-free, and consistent and asymptotically normal via the SLLN and delta method (Proposition 8.7). It can fail to exist (Cauchy), leave the parameter space (negative binomial: $\hat r < 0$ with probability $0.4167$ at $n=10$), or return parameters under which the observed data were impossible (uniform, $22.5\%$ of the time at $n=6$). Its efficiency loss is real: for the Gamma shape you need $74\%$ more data.
4. Maximum likelihood must be defined relative to a *stated* dominating measure $\mu$ (Definition 8.8). The choice of $\mu$ is irrelevant provided it is shared by all $\theta$; making it explicit is what extends MLE to mixed distributions such as the censored Gaussian, closing Chapter 1's Failure 2.
5. The MLE is equivariant under reparameterization (Proposition 8.9) and, in an exponential family, is exactly moment matching for the sufficient statistic (Proposition 8.10). Unbiasedness is destroyed by nonlinear transformation; the two properties are in tension and MLE chooses equivariance.
6. Maximizing likelihood minimizes an empirical estimate of the *forward* KL divergence $D_{\mathrm{KL}}(P^\star\|P_\theta)$. Under misspecification the MLE converges to the KL projection of the truth onto the model — a well-defined and often useful target. The direction of KL is why maximum-likelihood generative models blur rather than mode-collapse.
7. The MLE's failure modes are the failure modes of a training run: non-existence; non-uniqueness (Laplace medians, label switching); *multimodality even in regular one-parameter families* (a five-point Cauchy sample has three local maxima and the sample median is not the MLE); unbounded likelihood (a Gaussian-mixture spike on one datum drives $\ell_n\to+\infty$); and inconsistency when the parameter count grows with $n$ (Neyman–Scott: $\hat\sigma^2\to\sigma^2/2$, with bias that never shrinks and variance that does).
8. The score has mean zero at the truth (Proposition 8.14) — Chapter 2's differentiation under the integral — and this fails exactly when the support depends on $\theta$.
9. The information equality (Theorem 8.15) says $\operatorname{Var}(s) = -\mathbb E[\nabla^2\log p_\theta]$: gradient noise equals expected curvature. It underlies natural gradient, K-FAC, and Laplace approximations, and it holds only under correct specification. Fisher information is additive over independent observations, equals $\nabla^2A(\eta)$ in an exponential family, and transforms as a metric tensor (Proposition 8.18) — so it is not comparable across parameterizations.
10. The Cramér–Rao bound (Theorem 8.19) is Cauchy–Schwarz applied to the covariance of an estimator with the score. It binds *unbiased* estimators only — the biased Gaussian-variance MLE sits legally below it — and equality demands exact proportionality to the score, confining finite-sample efficiency to exponential families. When regularity fails, the correct replacement is the derivative-free Chapman–Robbins bound (Proposition 8.23), which for $\mathrm{Unif}[0,\theta]$ is valid, non-vacuous (within $1.5\times$), and scales like $n^{-2}$ as it should.
11. The MLE is consistent because KL is minimized at the truth, with a *uniform* law of large numbers transferring that to argmaxes (Theorem 8.24); it is asymptotically normal with covariance $I(\theta_0)^{-1}/n$ (Theorem 8.25), via a Taylor expansion of the score plus the CLT and Slutsky. The delta method converts this into standard errors for any smooth function of $\hat\theta$ (Corollary 8.26).
12. With nuisance parameters, the achievable variance is governed by $(I^{-1})_{jj}$, not $1/I_{jj}$ (Corollary 8.20), and the profile likelihood (Definition 8.29) automatically supplies the right curvature — the Schur complement. Using $I_{jj}$ instead understates the Gamma shape's standard error by a factor $2.11$.
13. Under misspecification the asymptotic covariance is the sandwich $H^{-1}GH^{-1}$ (Theorem 8.28); the two matrices coincide only when the information equality holds. Model-based standard errors are systematically too small: nominal $95\%$ intervals covered $77.2\%$ in a simulated heteroskedastic regression, and the sandwich restored them to $92.7\%$.
14. Rao–Blackwell (Theorem 8.30) is the law of total variance: conditioning preserves bias and cannot increase MSE, and sufficiency is needed only to make the conditional expectation computable. The Poisson example gave a $16.3\times$ variance reduction; the ML analogues are averaging over dropout masks, augmentations, and sampled latents.
15. MAP is penalized maximum likelihood with penalty $-\log\pi(\theta)$: Gaussian prior $\Rightarrow$ ridge/weight decay, Laplace $\Rightarrow$ lasso, with sparsity coming from the kink at zero rather than the tails. MAP is *not* reparameterization equivariant — the same Beta posterior has modes $0.33333$ in $p$ and $0.35714$ in log-odds. Shrinkage helps for a reason available in two lines: $\mathbb E\|cX-\theta\|^2 = c^2d+(1-c)^2\|\theta\|^2$ beats $d$ for some $c$ at every $\theta$ (Proposition 8.37); James–Stein (Theorem 8.38) shows the gain survives *estimating* the optimal $c$ once $d\ge3$, making the MLE of a Gaussian mean inadmissible.
16. EM is alternating maximization of the ELBO in $q$ and in $\theta$; monotone likelihood improvement is three lines (Theorem 8.39). It converges to stationary points, not to the MLE. Restricting $q$ to a tractable family turns EM into variational inference and forfeits the likelihood guarantee, retaining only monotonicity of the bound. Finally, the classical bias–variance U-curve is a theorem about $p\le n$ with no implicit regularization: measured double descent (peak test MSE $2285$ at $p=n=100$, falling to $0.358$ at $p=2000$, and a peak that reappears at $n\approx p$) shows parameter count is the wrong complexity measure — while a modest ridge penalty removes the peak entirely, which is the part of the story most often left out.

### Callbacks

This chapter discharged the promises Chapter 1 made when it opened with the Four Failures, and it did so using the tools Chapter 2 built. Failure 1 (densities exceeding 1, $P(X=x)=0$ for continuous $X$) was settled by Chapter 2's Radon–Nikodym derivative and revisited here in Common Confusion 8.3, where likelihood values are shown to be densities with respect to a base measure and therefore unbounded above with no contradiction. Failure 2 ($A=\max(0,Z)$ being neither discrete nor continuous) is now fully closed: Worked Example 8.7 estimated a censored-Gaussian parameter by maximum likelihood with respect to the two-piece base measure $\delta_0 + \lambda|_{(0,\infty)}$, one object valid in all three cases. Failure 3 (swapping $\nabla$ and $\mathbb E$) was Chapter 2's Dominated Convergence Theorem, and this chapter cashed it repeatedly — Definition 8.13's conditions (R3) and (R4) *are* the DCT hypotheses, Proposition 8.14 and Theorem 8.15 are the payoffs, and $\mathrm{Unif}[0,\theta]$ is the counterexample showing what happens without them. Failure 4 (constant GAN loss on disjoint supports) was diagnosed in Chapter 1 and resolved in Chapter 2; §8.3.3 completed the picture from the estimation side by showing that maximum likelihood minimizes the *forward* KL, which is infinite on disjoint supports — which is exactly why likelihood-based generative training is impossible for a pushforward with a lower-dimensional support, and why practitioners either add noise, use a different divergence, or use an invertible map. Chapter 1's warning about countability and measurability of suprema was honored in §8.1.1 rather than swept aside; its running examples (three coin flips, $A = \max(0,Z)$, the disease test's base rates) surfaced again here, and its Proposition 1.12 became MAP.

### Before you continue

1. Why is "the variance of the estimate" a category error, and what is the correct phrase?
2. State the bias–variance decomposition and explain why the MSE-optimal divisor for a Gaussian variance is $n+1$ rather than $n-1$.
3. Maximum likelihood requires a dominating measure. Explain why the *choice* of dominating measure does not affect the MLE, and why it must nonetheless be the same for every $\theta$.
4. State the information equality and name one machine learning algorithm that exploits it.
5. The Cramér–Rao bound is $1/(nI(\theta))$. Give two distinct reasons an estimator might legitimately have smaller variance than this.
6. Write the sandwich covariance and say what makes it collapse to $I^{-1}$.
7. Rao–Blackwell requires a *sufficient* statistic even though the variance inequality holds for any conditioning variable. Why?
8. Give the three-line proof that EM never decreases the likelihood, and say which line breaks under an approximate E-step.
9. Name two distinct properties that MLE has and MAP lacks, and one that MAP has and MLE lacks.

### Looking ahead to Chapter 9

Chapter 8 produced a number; Chapter 9 asks whether you should believe it. We turn to hypothesis testing and confidence sets: the Neyman–Pearson lemma and why the likelihood ratio is the optimal test statistic (a result that will feel inevitable after §8.3), generalized likelihood ratio tests and Wilks' theorem — whose $\chi^2$ limit is Theorem 8.25 seen from another angle — and the duality between confidence sets and tests. Then the uncomfortable part: what a $p$-value is and, at length, what it is not; why "$p < 0.05$" is not the probability the null is true; the multiple-comparisons problem and its corrections (Bonferroni, which is Proposition 1.8's union bound wearing a suit, and Benjamini–Hochberg's false discovery rate); and the bootstrap, which replaces analytic sampling distributions with resampling from $\hat P_n$. The chapter closes on the question you will actually face: how to compare two machine learning models honestly, given that test sets are reused, hyperparameters are tuned on them, benchmarks are selected after the fact, and the number of implicit comparisons is far larger than the number reported.

---

# Exercises

## Part A — Theory

**A1 (★).** Let $X_1,\dots,X_n\stackrel{iid}\sim\mathrm{Poisson}(\lambda)$. (a) Find the MLE of $\lambda$ and verify it is the unique maximizer. (b) Compute $I(\lambda)$ two ways (variance of the score and negative expected second derivative) and check they agree. (c) Is $\bar X_n$ efficient in the sense of Definition 8.21? (d) Using Proposition 8.9, write down the MLE of $\theta := P_\lambda(X=0)$ and say whether it is unbiased.

**A2 (★).** Let $X_1,\dots,X_n\stackrel{iid}\sim\mathrm{Unif}[0,\theta]$. Derive $\mathbb E[X_{(n)}]$, $\operatorname{Var}(X_{(n)})$ and $\operatorname{MSE}(X_{(n)})$ from the CDF $P(X_{(n)}\le t)=(t/\theta)^n$, and find the constant $c$ minimizing $\operatorname{MSE}(cX_{(n)})$. Compare to the MLE and to the unbiased estimator.

**A3 (★★).** Prove that for any positive definite $I$, $(I^{-1})_{jj}\ge 1/I_{jj}$, with equality iff the $j$-th row of $I$ has zeros off the diagonal. Interpret in terms of nuisance parameters.

**A4 (★★).** Let $X\sim\mathcal N(\mu,\sigma^2)$ with both unknown. (a) Show the Fisher information matrix is diagonal. (b) Explain why "the mean and the variance can be estimated independently" is a fair summary here but is *not* implied by diagonality of $I$ in general. (c) Reparameterize to $(\mu,\tau)$ with $\tau=\log\sigma$ and recompute $I$ using Proposition 8.18.

**A5 (★★).** (Non-existence of an unbiased estimator.) Let $X\sim\mathrm{Poisson}(\lambda)$, $n=1$. Show there is no unbiased estimator of $g(\lambda)=1/\lambda$. *Hint:* an unbiased estimator would give $\sum_{k\ge0}\delta(k)\lambda^k e^{-\lambda}/k! = 1/\lambda$ for all $\lambda>0$; multiply by $\lambda e^{\lambda}$ and compare power series.

**A6 (★★).** Let $\delta$ be unbiased for $g(\theta)$ and let $T$ be sufficient. Show directly (without invoking Theorem 8.30) that $\operatorname{Cov}(\delta - \delta^{\mathrm{RB}}, \delta^{\mathrm{RB}}) = 0$, and deduce the variance inequality. Which chapter-6 fact are you using?

**A7 (★★★).** Prove that if equality holds in the Cramér–Rao bound for all $\theta$ in an open interval, with $\hat g_n$ a fixed statistic, then the model is a one-parameter exponential family in which $\hat g_n$ is an affine function of the natural sufficient statistic. State carefully the regularity you need.

**A8 (★★★).** (Superefficiency.) Define Hodges' estimator $\tilde\theta_n := \hat\theta_n\mathbf 1\{|\hat\theta_n| \ge n^{-1/4}\}$ where $\hat\theta_n = \bar X_n$ for $X_i\sim\mathcal N(\theta,1)$. Show that $\sqrt n(\tilde\theta_n-\theta)\to_d\mathcal N(0,1)$ for $\theta\ne0$ but $\sqrt n\,\tilde\theta_n\to_p 0$ at $\theta = 0$. Then show that for $\theta_n := n^{-1/4}/2$ the risk $n\mathbb E_{\theta_n}(\tilde\theta_n-\theta_n)^2$ diverges. Explain what this says about the phrase "asymptotically efficient".

**A9 (★★).** (Bimodality of the Cauchy likelihood.) Let $n=2$ with observations $x_1=-a$, $x_2=+a$, $a>0$, from $\mathrm{Cauchy}(\theta,1)$. (a) Show that $\theta=0$ is always a stationary point. (b) Show that it is a local *maximum* iff $a\le1$ and a local *minimum* iff $a>1$, and that in the latter case there are exactly two symmetric global maxima. (c) Locate them in closed form. (d) Explain in one sentence why no analogue of this happens for the Gaussian location model.

**A10 (★★).** (Chapman–Robbins in a second irregular model.) For the shifted exponential $p_\theta(x)=e^{-(x-\theta)}\mathbf 1_{\{x>\theta\}}$ with $n$ i.i.d. observations, compute $\chi^2_n(\theta+h,\theta)$ for the admissible sign of $h$, evaluate the Chapman–Robbins bound, and compare it with the exact variance of the unbiased estimator $X_{(1)}-1/n$. Comment on the $n$-dependence.

## Part B — Computational labs

Report seeds, sample sizes, and results to four decimals. Where a closed form exists, check against it.

**B1 (★).** *Fisher information three ways.* For $X\sim\mathcal N(0,e^{2\tau})$ with $\tau = \log 2$: (a) estimate $\operatorname{Var}(s(\tau;X))$ by Monte Carlo; (b) estimate $-\mathbb E[\partial_\tau^2\log p_\tau]$ by Monte Carlo; (c) compute the observed information per observation at the MLE from a finite sample. Confirm all three approach $2$. Then repeat (c) for the Cauchy location model, where $I(\theta)=1/2$, and report the spread of the observed information across replications at $n\in\{20,50,200\}$. Explain the qualitative difference between the two models.

**B2 (★★).** *MoM vs MLE for the Gamma.* For true shape $\alpha=2$, scale $s=3$ and $n\in\{10,25,50,100,400\}$, simulate $4000$ replicates and tabulate bias and MSE of $\hat\alpha^{\mathrm{MoM}}$ and $\hat\alpha^{\mathrm{MLE}}$. Compare the MSE ratio with the theoretical ARE computed from $\alpha/(\alpha\psi'(\alpha)-1)$ and the delta method.

**B3 (★★).** *CRLB attained and not attained.* For $\mathcal N(\mu,\sigma^2)$: (a) verify by simulation that $\operatorname{Var}(\bar X_n)$ equals $\sigma^2/n$ exactly; (b) tabulate $\operatorname{Var}(S^2)$, $\operatorname{Var}(\hat\sigma^2_{\mathrm{MLE}})$ and the CRLB $2\sigma^4/n$ for $n\in\{5,10,20,50,200\}$; (c) explain why one of your columns falls below the bound without contradicting Theorem 8.19.

**B4 (★★★).** *EM for a two-component Gaussian mixture, from scratch.* Implement the updates of Proposition 8.40. (a) On $n=300$ points from $0.35\,\mathcal N(-2,0.8^2)+0.65\,\mathcal N(1.5,1.2^2)$, confirm the log-likelihood is non-decreasing at every iteration and report its first six increments. (b) Produce a run that converges to a spurious narrow component by initializing $m_1$ at a data point with a small $\sigma_1$; report the final parameters and the log-likelihood gap. (c) Modify the M-step to enforce $\sigma_k\ge\sigma_{\min}$ and show this is MAP under a specific (improper) prior — or explain why it is not exactly MAP under any prior.

**B5 (★★★).** *Double descent for random features.* With $n=100$, $d=20$, targets $y=x^\top\beta+0.5\varepsilon$, features $\tanh(W^\top x)$, fit by minimum-norm least squares for $p$ ranging over $[2, 2000]$ and plot test MSE on a log–log scale, averaging over at least 20 draws. Locate the interpolation peak. Repeat with ridge $\lambda=1$ and describe the difference. Then: hold $p=2000$ fixed and vary $n$; does the peak reappear?

## Part C — Paper-style derivations

**C1 (★★★).** *The information equality, with its hypotheses and a counterexample.* State conditions (R1)–(R4) precisely. Prove $\mathbb E_\theta[s(\theta;X)]=0$ and $I(\theta)=\operatorname{Var}(s)=-\mathbb E[\nabla^2\log p_\theta]$, indicating at each interchange of derivative and integral exactly which condition licenses it and which theorem of Chapter 2 is being invoked. Then exhibit a family violating (R1) for which both conclusions fail, computing the "score" explicitly and showing its mean is nonzero. Finally, discuss whether (R1) can be weakened to "the support is $\theta$-dependent but the boundary contributes no mass".

**C2 (★★★).** *Asymptotic normality of the MLE.* Write a complete proof of Theorem 8.25. Be explicit about: (i) where consistency is used and why the theorem cannot prove it; (ii) where a uniform (rather than pointwise) law of large numbers is required and what would go wrong with only the pointwise version; (iii) which step uses the CLT and which uses Slutsky; (iv) exactly which step fails under misspecification, and what the conclusion becomes.

**C3 (★★★).** *EM from the ELBO.* Derive EM as alternating maximization of $\mathcal L(q,\theta)$, prove monotone improvement of the observed-data log-likelihood, and then analyze the approximate-E-step case: show that if $q^{(t+1)}$ is restricted to a family $\mathcal Q$ not containing the true posterior, the chain of inequalities breaks at a specific step; identify the residual gap; prove that the ELBO is still monotone; and give a concrete two-point example where the observed-data likelihood actually *decreases* across a variational EM step.

---

# Appendix A — Self-check answers

**8.1.** (1) Yes, it is measurable (constant functions are), so it is a legitimate estimator; it is unbiased only if $\theta=7$, so not unbiased in the sense of Definition 8.2 (which quantifies over all $\theta$); it is not consistent (it converges to 7, not $\theta$); its MSE is $(7-\theta)^2$, which is *zero* at $\theta=7$ — smaller than any sensible estimator can achieve there. This is why "small variance at one point" is worthless as a criterion. (2) Increasing $c$ beyond $n-1$ trades a growing squared bias against a falling variance; the variance falls like $1/c^2$ while the squared bias grows like $(c-n+1)^2/c^2$, and the derivative $f'(c)=2(n-1)(c-n-1)/c^3$ vanishes at exactly $n+1$. (3) No. By Jensen (Ch. 4) applied to the strictly concave $\sqrt\cdot$, $\mathbb E[\sqrt{S^2}] < \sqrt{\mathbb E[S^2]} = \sigma$: biased low. (4) $\hat\theta_n := X_1$ for a mean parameter — unbiased at every $n$, variance constant, no convergence.

**8.2.** (1) Because MoM only constrains the mean, and a sample can have a small mean while containing one large observation; the MLE maximizes $\prod p_\theta(x_i)$, which is zero for any $\theta$ excluding an observed point, so such $\theta$ can never be the argmax. (2) The multivariate CLT gives asymptotic normality of the sample moment vector; the delta method transfers it through the smooth map $(m_1,m_2)\mapsto m_1^2/(m_2-m_1^2)$, with the gradient of that map appearing in the asymptotic variance. (3) Any Cauchy location family: $\mathbb E[X]$ does not exist, but $\mathbb E[\arctan(X-\theta)] = 0$ or the population median works and gives a consistent estimator. (4) Because consistency is a statement about the limit; at finite $n$ the MLE typically has smaller variance, and asymptotically it attains the Cramér–Rao bound while MoM generally does not — for the Gamma shape the MoM needs $74\%$ more data.

**8.3.** (1) Yes to the same MLE — the base measures differ by a $\theta$-free factor which shifts $\ell_n$ by a constant. No to the same value — the reported maximized likelihood differs by $\prod_i\frac{d\mu}{d\nu}(x_i)$. This is why likelihood *values* are only meaningful relative to a stated base measure, while likelihood *ratios* are absolute. (2) Because the likelihood is discontinuous at $\theta = x_{(n)}$ (jumping from $0$) and strictly decreasing after; the maximum occurs at a non-differentiable point. (3) In the bound $p_\theta(x_1)\ge \frac12\phi_{x_1,\varepsilon^2}(x_1) = \frac{1}{2\varepsilon\sqrt{2\pi}}$: this diverges only because the spike is centered exactly on $x_1$. For $m_1$ a constant $\ne x_1$, $\phi_{m_1,\varepsilon^2}(x_1)\to0$. For the Cauchy: heavy tails. A distant observation contributes $-\log(1+(x_i-\theta)^2)$, which grows only logarithmically, so remote points barely penalize $\theta$ and each cluster of data can support its own local maximum. Under a Gaussian the penalty is quadratic, remote points dominate, and the log-likelihood is concave in the location parameter, so no second maximum can form. (4) It converges to the KL projection $\arg\min_\theta D_{\mathrm{KL}}(P^\star\|P_\theta)$ — the forward KL, expectation taken under the truth.

**8.4.** (1) (R1), common support: the support $[0,\theta]$ moves with $\theta$. As a result the interchange $\partial_\theta\int p_\theta d\mu = \int\partial_\theta p_\theta \,d\mu$ is invalid, and indeed the score is the constant $-1/\theta$ with mean $-1/\theta\ne0$. (2) $\operatorname{Var}(\nabla\log p_\theta) = -\mathbb E[\nabla^2\log p_\theta]$. Left: how much per-example gradients disagree. Right: how sharply the expected log-likelihood peaks. (3) Because the sample score is a *sum* of per-observation scores, and the variance of a sum of independent mean-zero terms is the sum of variances (Definition 1.23). With dependence, cross-covariances survive. (4) Yes, the information changes (by the Jacobian sandwich of Proposition 8.18): $I(\tau)=\sigma^2 \cdot I(\sigma)\cdot\sigma$ appropriately. No, the Fisher–Rao distance between two *fixed distributions* is a property of the distributions and is parameterization-independent.

**8.5.** (1) Unbiasedness is used in $\operatorname{Cov}(\hat g_n,S_n)=g'(\theta)$ — specifically in identifying the derivative of $\mathbb E_\theta[\hat g_n]$ as $g'(\theta)$. Cauchy–Schwarz is used in bounding that covariance by the product of standard deviations. (2) Because the MLE is biased and Theorem 8.19 constrains only unbiased estimators. (3) Estimating $\theta_j$ when the other coordinates are unknown costs you the part of the score for $\theta_j$ that is explained by the other coordinates' scores; only when that correlation is zero do the two agree. (4) (R1) — the common-support condition — which invalidates the interchange used to compute $\operatorname{Cov}(\hat g_n, S_n)$; the whole bound is void, not merely loose. The Chapman–Robbins bound (Proposition 8.23) applies instead: it replaces the score by a finite likelihood ratio and needs no differentiability. It decays like $n^{-2}$ here (e.g. $0.019648\,\theta^2$ at $n=5$, against an achieved $0.028571\,\theta^2$), matching the true $O_P(1/n)$ error rate, whereas the illegitimate $\theta^2/n$ decays only like $n^{-1}$.

**8.6.** (1) Step 1 uses the KL characterization to show the population objective is uniquely maximized at $\theta_0$; Step 2 uses uniform convergence to transfer that from the population objective to the sample objective, and Step 3 turns it into convergence of argmaxes. (2) Twice: to guarantee $\hat\theta_n$ is eventually interior so that $\nabla\ell_n(\hat\theta_n)=0$, and to guarantee the intermediate point $\tilde\theta_n$ in the Taylor expansion converges to $\theta_0$. (3) None of (C1)–(C3). What is violated is the standing assumption that $\Theta$ is fixed as $n\to\infty$; here $\dim\Theta = n+1$. (4) Theorem 8.15, the information equality, which forces $G = H$. Its collapse says gradient noise and curvature are the same matrix — the property that makes $I^{-1}$ both the optimal preconditioner and the asymptotic covariance. Too narrow: $(I^{-1})_{jj}\ge 1/I_{jj}$, so $1/\sqrt{nI_{jj}}$ underestimates the standard error (by a factor $2.11$ for the Gamma shape at $\alpha=2$, $s=3$). The profile likelihood's curvature gives the correct value automatically.

**8.7.** (1) The law of total variance; conditioning removes $\mathbb E[\operatorname{Var}(\delta\mid T)]$, the within-$T$ variance. (2) Because without sufficiency, $\mathbb E[\delta\mid T]$ depends on $\theta$ and is therefore not a statistic — you could not compute it without knowing the answer. (3) Because $(X_1,\dots,X_n)\mid \sum_i X_i=t$ is Multinomial$(t;1/n,\dots,1/n)$, and $\lambda$ has cancelled out; that cancellation *is* sufficiency of $T$. (4) Averaging a policy-gradient estimator over several sampled actions, or averaging predictions over dropout masks / augmentations at test time; the role of $T$ is played by the observed data (or the observed data plus whatever the estimator legitimately depends on), and the averaged-out randomness is the auxiliary noise.

**8.8.** (1) $\frac{1}{2\tau^2}\|\beta\|_2^2$ and $\frac1b\|\beta\|_1$. The Laplace prior produces exact zeros, because its density has a kink (non-differentiable point) at $0$, making $\beta_j=0$ a subgradient-stationary point for a whole interval of data values. (2) Because the likelihood is a function *value* and transforms trivially, while the prior is a *density* and picks up a Jacobian factor $|\det J|$ under change of variables; that $\psi$-dependent term shifts the maximizer. (3) $\mathbb E_\theta\|\hat\theta^{\mathrm{JS}}-\theta\|^2 = d-(d-2)^2\mathbb E[1/\|X\|^2]$. The condition $d\ge3$ is used to ensure $\mathbb E[\|X\|^{-2}]<\infty$ (the integrand $\|x\|^{-2}$ is integrable near the origin in dimension $\ge3$) and $(d-2)^2>0$. (4) Nothing. Domination is a statement about the *summed* risk; individual coordinates can and often do get worse.

**8.9.** (1) $\log p_{\theta^{(t+1)}}(x)\ge\mathcal L(q^{(t+1)},\theta^{(t+1)})\ge\mathcal L(q^{(t+1)},\theta^{(t)})=\log p_{\theta^{(t)}}(x)$. The second inequality is the M-step; the final equality is the E-step (KL $=0$). (2) Because $\mathcal L(q,\theta)=\mathbb E_q[\log p_\theta(x,Z)] - \mathbb E_q[\log q(Z)]$ and the second term has no $\theta$; it is an additive constant in the M-step optimization. (3) Lost: the final equality, hence the guarantee that the observed-data likelihood increases. Retained: monotonicity of the ELBO itself, since each step still maximizes over one argument. (4) Theorem 8.39 guarantees no decrease, not convergence to the global maximum; a monotone sequence can converge to any local maximum or stationary point, and the degenerate run found a poor one.

**8.10.** (1) $p\le n$; the estimator is *the* least-squares fit with no further selection; complexity is measured by parameter count; noise must be fit or absorbed. An overparameterized network violates the first three (it has $p\gg n$, the fit is selected implicitly by the optimizer, and its effective complexity is far below $p$). (2) $2285.2$; the design matrix is square and nearly singular, so the smallest singular value is near zero and the interpolating solution has enormous norm — a conditioning failure. (3) Ridge bounds the solution norm and floors the effective smallest eigenvalue at $\lambda$, so the norm cannot explode; this suggests the peak is largely a numerical/conditioning artifact of insisting on exact interpolation, not a fundamental statistical barrier. (4) Because curvature is a quadratic form that transforms as $J^\top H J$ under reparameterization (Proposition 8.18), so its eigenvalues can be changed arbitrarily by a rescaling that leaves the function — and hence the generalization behavior — untouched.

---

# Solutions to Exercises

## Part A

**A1.** *(a)* With $\mu = \#$ on $\{0,1,2,\dots\}$, $p_\lambda(k)=e^{-\lambda}\lambda^k/k!$, so $\ell_n(\lambda) = -n\lambda + \big(\sum_i x_i\big)\log\lambda - \sum_i\log(x_i!)$. Then $\ell_n'(\lambda) = -n + \frac{\sum_i x_i}{\lambda}$, vanishing at $\hat\lambda = \bar x$, and $\ell_n''(\lambda) = -\sum_i x_i/\lambda^2 < 0$ whenever $\sum_i x_i>0$, so $\ell_n$ is strictly concave and $\bar x$ is the unique maximizer. (If every $x_i=0$, $\ell_n(\lambda)=-n\lambda$ is decreasing and $\hat\lambda=0$, a boundary solution — the same phenomenon as $k=0$ in Worked Example 8.3. The thinking to internalize: always check the boundary separately, because the stationarity equation cannot see it.)

*(b)* Score: $s(\lambda;x) = \partial_\lambda(-\lambda + x\log\lambda) = x/\lambda - 1$. Route one: $\operatorname{Var}(s) = \operatorname{Var}(X)/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$. Route two: $\partial_\lambda^2\log p_\lambda = -x/\lambda^2$, so $-\mathbb E[\cdot] = \lambda/\lambda^2 = 1/\lambda$. They agree, as Theorem 8.15 requires. (Sanity check by Proposition 8.17: the Poisson is an exponential family with $\eta = \log\lambda$, $T(x)=x$, $A(\eta)=e^\eta$, so $I(\eta)=A''(\eta)=e^\eta=\lambda$; Proposition 8.18 with $d\eta/d\lambda = 1/\lambda$ gives $I(\lambda)=(1/\lambda)^2\lambda = 1/\lambda$ ✓.)

*(c)* Yes. $\bar X_n$ is unbiased for $\lambda$ with variance $\lambda/n$, which equals $1/(nI(\lambda))$. Note also $\bar X_n - \lambda = \frac\lambda n S_n$, the exact proportionality demanded by Proposition 8.22 — as it must be, since the Poisson is an exponential family and $\bar X$ is its natural sufficient statistic in mean parameterization.

*(d)* By equivariance, $\hat\theta = e^{-\bar X_n}$. It is *not* unbiased: $\mathbb E[e^{-\bar X}] > e^{-\mathbb E[\bar X]} = e^{-\lambda}$ by strict convexity of $t\mapsto e^{-t}$ and Jensen (Ch. 4). Worked Example 8.15 measured the bias: $0.386092$ against the truth $0.367879$ at $\lambda=1,n=10$. The UMVUE is $\big(\frac{n-1}{n}\big)^{T}$.

**A2.** From $F(t)=(t/\theta)^n$ on $[0,\theta]$, the density of $X_{(n)}$ is $nt^{n-1}/\theta^n$. Hence
$$\mathbb E[X_{(n)}] = \int_0^\theta t\cdot\frac{nt^{n-1}}{\theta^n}dt = \frac{n}{n+1}\theta, \qquad \mathbb E[X_{(n)}^2] = \frac{n}{n+2}\theta^2,$$
so $\operatorname{Var}(X_{(n)}) = \theta^2\big(\frac{n}{n+2}-\frac{n^2}{(n+1)^2}\big) = \frac{n\theta^2}{(n+1)^2(n+2)}$ and, by Proposition 8.3 with bias $-\theta/(n+1)$,
$$\operatorname{MSE}(X_{(n)}) = \frac{n\theta^2}{(n+1)^2(n+2)} + \frac{\theta^2}{(n+1)^2} = \frac{2\theta^2}{(n+1)(n+2)}.$$
For general $c$, expanding $\mathbb E(cX_{(n)}-\theta)^2 = c^2\mathbb E[X_{(n)}^2] - 2c\theta\,\mathbb E[X_{(n)}] + \theta^2$ and substituting the two moments gives
$$\operatorname{MSE}(cX_{(n)}) = \theta^2\Big(\frac{n}{n+2}c^2 - \frac{2n}{n+1}c + 1\Big),$$
a quadratic in $c$ with positive leading coefficient, minimized where the derivative $\frac{2n}{n+2}c - \frac{2n}{n+1}$ vanishes:
$$c^\star = \frac{n/(n+1)}{n/(n+2)} = \frac{n+2}{n+1}, \qquad \operatorname{MSE}(c^\star X_{(n)}) = \theta^2\Big(1 - \frac{n(n+2)}{(n+1)^2}\Big) = \frac{\theta^2}{(n+1)^2},$$
the last step because $(n+1)^2 - n(n+2) = 1$. So the ordering at every $n$ is
$$\underbrace{\frac{\theta^2}{(n+1)^2}}_{c^\star=\frac{n+2}{n+1}} < \underbrace{\frac{\theta^2}{n(n+2)}}_{\text{unbiased}, \ c=\frac{n+1}{n}} < \underbrace{\frac{2\theta^2}{(n+1)(n+2)}}_{\text{MLE},\ c=1}.$$
(Check at $n=5$: $1/36 = 0.02778 < 1/35 = 0.02857 < 2/42 = 0.04762$ ✓.) The thinking: the MLE systematically *undershoots* (it can never exceed $\theta$), so inflating helps; but the MSE-optimal inflation overshoots the unbiased one, because the extra bias buys a variance reduction. Same lesson as Worked Example 8.1, opposite direction — here inflation, there deflation, both away from unbiasedness.

**A3.** Let $u := e_j$. Since $I\succ0$, define the inner product $\langle a,b\rangle := a^\top I b$. Then
$$1 = (e_j^\top e_j)^2 = \big(e_j^\top I\,(I^{-1}e_j)\big)^2 = \langle e_j,\, I^{-1}e_j\rangle^2 \le \langle e_j,e_j\rangle\,\langle I^{-1}e_j, I^{-1}e_j\rangle = I_{jj}\cdot (e_j^\top I^{-1}e_j) = I_{jj}(I^{-1})_{jj},$$
using Cauchy–Schwarz in $\langle\cdot,\cdot\rangle$. Divide by $I_{jj}>0$. Equality holds iff $I^{-1}e_j$ is proportional to $e_j$, i.e. $e_j$ is an eigenvector of $I^{-1}$, equivalently of $I$, equivalently $I_{kj}=0$ for all $k\ne j$. Interpretation: $1/I_{jj}$ is the bound you would face if all other parameters were *known*; $(I^{-1})_{jj}$ is the bound with them unknown. The gap is the price of nuisance parameters, and it vanishes exactly when the score for $\theta_j$ is uncorrelated with all the other scores — i.e. when learning about the others tells you nothing about $\theta_j$.

**A4.** *(a)* $\partial_\mu\log p = (x-\mu)/\sigma^2$ and $\partial_{\sigma^2}\log p = -\frac{1}{2\sigma^2}+\frac{(x-\mu)^2}{2\sigma^4}$. Their product has expectation $\frac{1}{\sigma^2}\mathbb E\big[(X-\mu)\big(\frac{(X-\mu)^2}{2\sigma^4}-\frac1{2\sigma^2}\big)\big]$, which involves only the first and third central moments of a Gaussian, both zero. So $I_{\mu\sigma^2}=0$ and $I=\operatorname{diag}(1/\sigma^2, 1/(2\sigma^4))$ (Monte Carlo confirmation in Worked Example 8.11).

*(b)* Diagonality of $I$ says the *scores* are uncorrelated, hence (by Theorem 8.25 plus Corollary 8.20) the MLEs are asymptotically uncorrelated and the nuisance penalty of A3 vanishes. Here the summary is fair in a stronger, exact sense, but only because of a separate fact — Cochran's theorem (Ch. 5) makes $\bar X$ and $\sum(X_i-\bar X)^2$ genuinely *independent* at every $n$. Diagonality of $I$ alone gives neither exact independence nor even exact zero correlation of the estimators at finite $n$; it is a statement about the model's local geometry at one point, not about the joint law of two statistics.

*(c)* With $\tau = \log\sigma$: $\log p = -\tau - \frac12\log2\pi - \frac{(x-\mu)^2}{2}e^{-2\tau}$, so $\partial_\mu\log p = (x-\mu)e^{-2\tau}$ with variance $e^{-2\tau}=1/\sigma^2$, and $\partial_\tau\log p = -1+(x-\mu)^2e^{-2\tau}$ with variance $2$ (as in Worked Example 8.10). The cross term again vanishes by oddness. So $I(\mu,\tau)=\operatorname{diag}(1/\sigma^2,\,2)$. Cross-check with Proposition 8.18 on the second coordinate: $\tau = \frac12\log\sigma^2$, $d\tau/d\sigma^2 = 1/(2\sigma^2)$, and $I_{\sigma^2\sigma^2} = (d\tau/d\sigma^2)^2 I_{\tau\tau}$ gives $\frac{1}{2\sigma^4} = \frac{1}{4\sigma^4}\cdot 2$ ✓. Note the information about $\tau$ is a constant: the log-scale parameterization is the one in which the model's geometry is flat, which is exactly why deep learning parameterizes scales logarithmically.

**A5.** Suppose $\delta$ is unbiased: $\sum_{k\ge0}\delta(k)\frac{\lambda^k e^{-\lambda}}{k!} = \frac1\lambda$ for all $\lambda>0$. Multiply both sides by $\lambda e^{\lambda}$:
$$\sum_{k\ge0}\frac{\delta(k)}{k!}\lambda^{k+1} = e^{\lambda} = \sum_{j\ge0}\frac{\lambda^j}{j!}.$$
Both sides are power series in $\lambda$ convergent on $(0,\infty)$, hence (by uniqueness of power-series coefficients) their coefficients agree. The left side has zero coefficient on $\lambda^0$; the right side has coefficient $1$. Contradiction. So no unbiased estimator of $1/\lambda$ exists — for any $n$, in fact, by the same argument applied to $T=\sum_i X_i\sim\mathrm{Poisson}(n\lambda)$ and completeness. The lesson: the unbiasedness constraint of §8.5 and §8.7 is not merely restrictive, it is sometimes *empty*, and any theory built only on it has nothing to say about $1/\lambda$.

**A6.** Write $D := \delta - \delta^{\mathrm{RB}}$. By the tower property (Ch. 6), $\mathbb E[D\mid T] = \mathbb E[\delta\mid T] - \delta^{\mathrm{RB}} = 0$ a.s. Since $\delta^{\mathrm{RB}}$ is $\sigma(T)$-measurable (indeed $\delta^{\mathrm{RB}}=h(T)$ for some measurable $h$ by Doob–Dynkin, Lemma 1.22),
$$\operatorname{Cov}(D,\delta^{\mathrm{RB}}) = \mathbb E[D\,h(T)] - \mathbb E[D]\mathbb E[h(T)] = \mathbb E\big[h(T)\,\mathbb E[D\mid T]\big] - 0 = 0.$$
Therefore $\operatorname{Var}(\delta) = \operatorname{Var}(D + \delta^{\mathrm{RB}}) = \operatorname{Var}(D)+\operatorname{Var}(\delta^{\mathrm{RB}}) \ge \operatorname{Var}(\delta^{\mathrm{RB}})$. The chapter-6 fact is that conditional expectation is the $L^2$ projection onto $\sigma(T)$-measurable functions, of which "the residual is orthogonal to the projection" is the defining property; the law of total variance in Theorem 8.30 is the same statement in different clothing.

**A7.** Suppose equality holds for all $\theta$ in an open interval. By Proposition 8.22 there is $c(\theta)$ with, for a single observation (take $n=1$ for clarity; the general case is identical with $\hat g_n$ in place of $\hat g$),
$$\partial_\theta\log p_\theta(x) = \frac{1}{c(\theta)}\big(\hat g(x) - g(\theta)\big) = a(\theta)\hat g(x) + b(\theta),$$
where $a := 1/c$, $b := -g/c$, and the identity holds for $P_\theta$-a.e. $x$; assume (regularity) it holds for $\mu$-a.e. $x$ in the common support $\mathcal S$ and that $a,b$ are continuous. Fix $\theta_0$ in the interval and integrate in $\theta$ from $\theta_0$ to $\theta$:
$$\log p_\theta(x) - \log p_{\theta_0}(x) = A(\theta)\hat g(x) + B(\theta), \qquad A(\theta) := \int_{\theta_0}^\theta a, \quad B(\theta):=\int_{\theta_0}^\theta b.$$
Exponentiating, $p_\theta(x) = h(x)\exp\{A(\theta)\hat g(x) + B(\theta)\}$ with $h := p_{\theta_0}$: a one-parameter exponential family with natural parameter $A(\theta)$, sufficient statistic $T = \hat g$, and $-B(\theta) = A^*(A(\theta))$ the log-partition function. So $\hat g$ is (exactly, not merely affinely) the natural sufficient statistic. Regularity needed: differentiability of $\theta\mapsto\log p_\theta(x)$ on the interval, a $\theta$-free support (R1) so that "$\mu$-a.e. $x$" makes sense uniformly, continuity of $a$ and $b$ to justify the fundamental theorem of calculus, and Fubini/monotone convergence to conclude that the resulting family is normalizable. If one only assumes equality at a *single* $\theta$, no such conclusion follows: attainment at one point is a much weaker condition.

**A8.** Let $\hat\theta_n = \bar X_n\sim\mathcal N(\theta,1/n)$ and $\tilde\theta_n := \hat\theta_n\mathbf 1\{|\hat\theta_n|\ge n^{-1/4}\}$.

*Case $\theta\ne0$.* $P(|\hat\theta_n| < n^{-1/4}) \le P(|\hat\theta_n - \theta| > |\theta| - n^{-1/4}) \to 0$ exponentially fast (for $n$ large enough that $n^{-1/4} < |\theta|/2$, the deviation required is $\ge|\theta|/2$, a fixed amount, and $\hat\theta_n$ has sd $n^{-1/2}$). Hence $P(\tilde\theta_n = \hat\theta_n)\to1$, so $\sqrt n(\tilde\theta_n-\theta)$ and $\sqrt n(\hat\theta_n-\theta)$ differ on an event of vanishing probability and have the same limit, $\mathcal N(0,1)$.

*Case $\theta=0$.* Now $\hat\theta_n\sim\mathcal N(0,1/n)$, so $P(|\hat\theta_n|\ge n^{-1/4}) = 2\Phi(-n^{1/4})\to0$ (the threshold is $n^{1/4}$ standard deviations away). Hence $\tilde\theta_n = 0$ with probability $\to1$, so $\sqrt n\,\tilde\theta_n\to_p 0$: the asymptotic variance at $\theta=0$ is $0$, strictly below the Cramér–Rao bound $1$. That is superefficiency.

*The price.* Take $\theta_n := \tfrac12 n^{-1/4}$. The threshold $n^{-1/4}$ is $(n^{-1/4}-\theta_n)/n^{-1/2} = \tfrac12 n^{1/4}$ standard deviations above the mean, so $P(\tilde\theta_n = 0) = P(|\hat\theta_n|<n^{-1/4}) \to 1$. On that event the error is exactly $\theta_n$, so
$$n\,\mathbb E_{\theta_n}\big(\tilde\theta_n - \theta_n\big)^2 \ \ge\ n\,\theta_n^2\,P(\tilde\theta_n = 0) = \frac{\sqrt n}{4}\,P(\tilde\theta_n=0)\ \to\ \infty,$$
while the same quantity for $\bar X_n$ is identically $1$. So the local risk is unbounded: the improvement at the single point $\theta=0$ is paid for by arbitrarily bad behavior in a shrinking neighborhood of it. This is why "asymptotically efficient" must be read as a *local uniform* statement (the Le Cam–Hájek framework), and why the informal claim "no estimator beats the MLE asymptotically" is false at the level of pointwise asymptotic variance and true only under a local-uniformity qualifier.

**A9.** *(a)* $\ell(\theta) = -\log(1+(\theta+a)^2)-\log(1+(\theta-a)^2) - 2\log\pi$ is an even function of $\theta$, so $\ell'(0)=0$; explicitly $\ell'(\theta) = \frac{-2(\theta+a)}{1+(\theta+a)^2}+\frac{-2(\theta-a)}{1+(\theta-a)^2}$, which vanishes at $\theta=0$.

*(b)* Differentiate again. With $u := \theta+a$, $v := \theta-a$, $\ell''(\theta) = \frac{2(u^2-1)}{(1+u^2)^2}+\frac{2(v^2-1)}{(1+v^2)^2}$. At $\theta=0$, $u=a$, $v=-a$, so
$$\ell''(0) = \frac{4(a^2-1)}{(1+a^2)^2},$$
negative iff $a<1$ and positive iff $a>1$. So the "obvious" estimate $\theta=0$ — the mean, the median, the midpoint — is a local *minimum* of the likelihood as soon as the two points are more than $2$ apart. Since $\ell(\theta)\to-\infty$ as $|\theta|\to\infty$ and $\ell$ is smooth and even, there must then be at least one maximum on each side, and by symmetry they are mirror images with equal likelihood; hence two global maxima.

*(c)* Set $\ell'(\theta)=0$ with $\theta\ne0$. Clearing denominators,
$$(\theta+a)\big(1+(\theta-a)^2\big) + (\theta-a)\big(1+(\theta+a)^2\big)=0 \;\Longrightarrow\; 2\theta\big(1 + \theta^2 - a^2\big)=0,$$
(expand: the $a$-odd terms cancel), so the non-zero stationary points are $\theta = \pm\sqrt{a^2-1}$, real exactly when $a>1$. Sanity check against Worked Example 8.9's five-point sample: the same mechanism, with two clusters instead of two points, put the outer maxima at $\pm5.1225$ — outside the data's centre, pulled toward a cluster.

*(d)* For the Gaussian location model $\ell(\theta)=-\frac{1}{2}\sum_i(x_i-\theta)^2+$const is a concave quadratic in $\theta$ for every dataset, so it has exactly one maximum, always. Concavity, not regularity, is what rules out multimodality.

**A10.** The support is $(\theta,\infty)$, so $P_{\theta+h}\ll P_\theta$ requires $h>0$ (shifting *right* shrinks the support). For one observation and $h>0$, the likelihood ratio is $L_1(x) = e^{h}\mathbf 1_{\{x>\theta+h\}}$, so
$$\mathbb E_\theta[L_1^2] = e^{2h}\,P_\theta(X>\theta+h) = e^{2h}e^{-h}=e^{h},$$
and by independence $\chi^2_n = e^{nh}-1$. With $g(\theta)=\theta$ the bound is
$$\operatorname{Var}_\theta(\hat\theta)\;\ge\;\sup_{h>0}\frac{h^2}{e^{nh}-1}.$$
Substituting $h = u/n$ gives $\frac{1}{n^2}\sup_{u>0}\frac{u^2}{e^u-1}$, and the numerical maximum of $u^2/(e^u-1)$ is $0.6476$ at $u=1.5936$. So the bound is $0.6476/n^2$. The exact answer: $X_{(1)}-\theta\sim\mathrm{Exp}(n)$, so $X_{(1)}-1/n$ is unbiased with variance exactly $1/n^2$. The bound is valid, and tight to within a factor $1.544$ — uniformly in $n$, since both sides scale as $n^{-2}$. Comment: this is the signature of an irregular model. Cramér–Rao would predict $1/n$ (the naive $I(\theta)=1$), off by a whole factor of $n$; Chapman–Robbins gets the *rate* right because it never assumed the likelihood was differentiable in $\theta$.

## Part B

**B1.** All numbers below are from the runs reported in Worked Example 8.10; reproduce with the stated seeds.

*(a)–(b)* $M = 2{,}000{,}000$ draws from $\mathcal N(0,4)$, seed 5150. Sample mean of the score: $-0.001012$ (target $0$, consistent with a Monte Carlo standard error of $\sqrt{2/M}\approx0.001$). Sample variance of the score: $1.989877$. Negative sample mean of $\partial_\tau^2\log p_\tau$: $1.997976$. A numerical second difference of the Monte Carlo expected log-likelihood ($h=10^{-4}$) gives $1.992028$. All four are consistent with $I(\tau)=2$; the residual discrepancies are Monte Carlo noise, and note that the score's variance is estimated from a fourth moment of a Gaussian, so it converges more slowly than the Hessian route — a practical reason to prefer route (b) when both are available.

*(c)* $n=500$, seed 606: $\hat\tau = 0.669039$ ($\hat\sigma = 1.9524$), observed information per observation $= 2.000000$ exactly. The exactness is structural: $J_n(\hat\tau)/n = 2\overline{x^2}e^{-2\hat\tau}$ and the MLE equation is $e^{2\hat\tau} = \overline{x^2}$.

*Cauchy contrast.* Observed information per observation at the MLE, $4000$ replicates, seed 31415: mean $0.5389$ (sd $0.1738$) at $n=20$; $0.5109$ (sd $0.1119$) at $n=50$; $0.5038$ (sd $0.0558$) at $n=200$. The qualitative difference: in the Gaussian-scale model the second derivative of $\log p$ is a fixed multiple of $x^2$ and the MLE equation cancels the randomness exactly, so observed $=$ expected at the maximizer; in the Cauchy model $\partial_\theta^2\log p$ is a bounded non-monotone function of $x-\theta$ with no such cancellation, so the observed information is genuinely random, with standard deviation shrinking like $n^{-1/2}$. Practical upshot: quoting an observed-information standard error at $n=20$ in a heavy-tailed model carries roughly $35\%$ relative uncertainty in the error bar itself.

**B2.** $R = 4000$ replicates, seed 31337, true $(\alpha,s)=(2,3)$, estimating $\alpha$:

| $n$ | bias MoM | MSE MoM | bias MLE | MSE MLE | MSE ratio |
|---|---|---|---|---|---|
| 10 | $0.9910$ | $3.9262$ | $0.7455$ | $3.0437$ | $1.2899$ |
| 25 | $0.3769$ | $0.7682$ | $0.2541$ | $0.4996$ | $1.5374$ |
| 50 | $0.1671$ | $0.2892$ | $0.1097$ | $0.1784$ | $1.6213$ |
| 100 | $0.0898$ | $0.1328$ | $0.0516$ | $0.0764$ | $1.7387$ |
| 400 | $0.0235$ | $0.0310$ | $0.0144$ | $0.0179$ | $1.7353$ |

Theory: $n\operatorname{Var}(\hat\alpha^{\mathrm{MLE}})\to \alpha/(\alpha\psi'(\alpha)-1) = 2/0.289868 = 6.8997$; the delta method applied to $(m_1,m_2)\mapsto m_1^2/(m_2-m_1^2)$ with Gamma moments gives $n\operatorname{Var}(\hat\alpha^{\mathrm{MoM}})\to 12.0000$ exactly. Ratio $1.7392$ — matched by the simulation at $n=100,400$ to two decimals. At $n=10$ the ratio is only $1.29$ because both estimators are dominated by bias and heavy right tails at that sample size, so the asymptotic variance comparison is not yet the whole story. The thinking to take away: verify asymptotic efficiency claims *at the sample size you actually have*; ARE is a statement about the limit and can understate or overstate the finite-$n$ advantage.

**B3.** *(a)* $n=8$, $N=500{,}000$, seed 1234: $\operatorname{Var}(\bar X) = 0.124663$ against $\sigma^2/n = 0.125$ — agreement to Monte Carlo error, and exact in theory, since $\bar X-\mu = \frac{\sigma^2}{n}S_n$ (Proposition 8.22).
*(b)* See the table in Worked Example 8.11 (CRLB $2/n$; $\operatorname{Var}(\hat\sigma^2_{\mathrm{MLE}})=2(n-1)/n^2$; $\operatorname{Var}(S^2)=2/(n-1)$; ratios $0.80,0.90,0.95,0.98,0.995$ for $n=5,10,20,50,200$).
*(c)* The MLE column falls below the bound because the MLE is biased ($\mathbb E[\hat\sigma^2_{\mathrm{MLE}}]=\frac{n-1}{n}\sigma^2$), and Theorem 8.19 constrains unbiased estimators only. Among unbiased estimators, $S^2$ is the UMVUE (Lehmann–Scheffé, Theorem 8.32, since $(\bar X,\sum(X_i-\bar X)^2)$ is complete sufficient), and it lies strictly above the bound at every finite $n$: the bound is *not attained* for the variance, only approached. Both facts coexist because "smallest variance among unbiased" and "smallest variance overall" are different questions.

**B4.** *(a)* $n=300$, seed 90210, init $w=0.5$, $(-1,1),(1,1)$. Log-likelihoods $-694.7263, -615.5364, -614.9389, -614.5356, -614.1977, -613.8829, -613.5813,\dots \to -611.0872$ at iteration 40. First six increments: $+79.1899, +0.5976, +0.4033, +0.3379, +0.3148, +0.3016$. Monotone at all 40 steps (checked programmatically). Fitted $\hat w = 0.3001$, $(-2.0386, 0.8034)$, $(1.4401, 1.3341)$. Note the shape of the increments: one enormous step (the initialization was bad and the first M-step fixes most of it) then slow linear decay. EM's asymptotic rate is linear with factor equal to the fraction of missing information; if you need many digits, switch to Newton or quasi-Newton once you are close.

*(b)* Init $m_1 = x_1 = 1.5902$, $\sigma_1 = 0.05$: log-likelihoods $-802.596, -629.290, -627.363, -626.979, -626.833, \dots \to -626.3692$, still monotone, converging to $\hat w=0.077228$, $(\hat m_1,\hat\sigma_1)=(1.6099, 0.165492)$, $(\hat m_2,\hat\sigma_2)=(0.2947, 2.0441)$. The gap to the good run is $15.28$ nats. With $\sigma_1 = 10^{-3}$ the run instead marches toward $\sigma_1\to0$ and $\ell_n\to+\infty$ (Proposition 8.11).

*(c)* Clipping $\sigma_k \leftarrow \max(\sigma_{\min}, \hat\sigma_k^{\text{weighted}})$ *is* exactly MAP under the improper prior $\pi(\theta)\propto \prod_k \mathbf 1\{\sigma_k\ge\sigma_{\min}\}$ (flat on the feasible set, zero outside): the penalized objective is $\ell_n$ restricted to the feasible set, and the M-step objective $Q(\cdot\mid\theta^{(t)})$ is, as a function of $\sigma_k$ alone, unimodal with its unconstrained maximizer at the weighted standard deviation, so its constrained maximizer is the clip. Because the M-step still exactly maximizes the (now constrained) $Q$, Theorem 8.39's proof goes through verbatim and the *penalized* log-likelihood is still monotone. Two caveats worth stating: the prior is improper (it does not integrate), so this is penalized ML rather than genuine Bayes; and if you instead clipped *after* an unconstrained update in a way that did not maximize the constrained $Q$ — for instance by clipping $v_k$ but not recomputing $m_k$ — the monotonicity guarantee would be lost, since step (b) of the proof would fail.

**B5.** The measured ridgeless and ridge curves are the table and Figure 8.6 in §8.10.2: classical minimum $0.4119$ at $p=20$, peak $2285.2228$ at $p=n=100$, second descent to $0.3577$ at $p=2000$; with $\lambda=1$ the curve is nearly flat from $p=40$ ($0.4329$) through $p=2000$ ($0.3656$) with no peak. To locate the peak, sample $p$ densely near $n$ — at $p=98$ and $p=102$ the risk is already down to $15.35$ and $18.00$, two orders of magnitude below the value at $p=100$, so a coarse grid will miss it entirely. That is a general lesson about reading published double-descent curves: the height of the peak is a property of the grid as much as of the model.

*Does the peak reappear when $n$ varies with $p$ fixed?* Yes, at $n\approx p$. With $p=300$ fixed, $d=20$, $\sigma=0.5$, $10$ replicates, seed 2468:

| $n$ | 50 | 100 | 200 | 280 | **300** | 320 | 400 | 800 | 2000 |
|---|---|---|---|---|---|---|---|---|---|
| test MSE | $0.5526$ | $0.5097$ | $1.0725$ | $5.3583$ | $\mathbf{874.20}$ | $5.1813$ | $1.1605$ | $0.4228$ | $0.3045$ |

So the peak is a function of the *ratio* $p/n$, not of $p$ alone — it sits at $p/n = 1$, where the feature matrix is square and worst-conditioned. Read the row right to left and you get the alarming version: **adding data can make test error worse**, dramatically so, if it pushes you toward the interpolation threshold from above. This "sample-wise double descent" is documented for real networks (Nakkiran et al., 2021) and is a good antidote to the belief that more data is monotonically better for a fixed un-regularized model.

## Part C

**C1.** *Conditions.* State (R1)–(R4) as in Definition 8.13. (R1): the support $\mathcal S = \{p_\theta>0\}$ is $\theta$-free. (R2): $\theta\mapsto p_\theta(x)$ is $C^2$ on a neighborhood $U$ of $\theta$ for $\mu$-a.e. $x\in\mathcal S$. (R3): $\|\nabla_\theta p_\theta(x)\|\le g_1(x)\in L^1(\mu)$ uniformly on $U$. (R4): $\|\nabla^2_\theta p_\theta(x)\|\le g_2(x)\in L^1(\mu)$ uniformly on $U$. Plus $\mathbb E_\theta\|s\|^2<\infty$.

*Mean-zero score.* As in Proposition 8.14. The single interchange is $\nabla_\theta\int p_\theta\,d\mu = \int\nabla_\theta p_\theta\,d\mu$, licensed by (R2)+(R3) through Chapter 2's differentiation-under-the-integral theorem, whose proof is: write the partial derivative as a limit of difference quotients, bound each quotient by $g_1$ using the mean value theorem, and apply Dominated Convergence. (R1) is used separately, to guarantee that dividing by $p_\theta$ is legitimate on a $\theta$-free set of full measure, so that $\int\nabla p_\theta\,d\mu = \int (\nabla p_\theta/p_\theta)p_\theta\,d\mu$ has no boundary contribution.

*Information equality.* As in Theorem 8.15. Two interchanges: $\nabla\int p_\theta = \int\nabla p_\theta$ (as above, via $g_1$) and $\nabla^2\int p_\theta = \int\nabla^2 p_\theta$ (a second application, now to the function $\theta\mapsto\int\nabla p_\theta\,d\mu$, with envelope $g_2$). Both are DCT. The algebraic step $\nabla^2\log p = \frac{\nabla^2p}{p} - ss^\top$ is pointwise calculus and needs only (R1)+(R2).

*Counterexample.* $p_\theta(x) = \theta^{-1}\mathbf 1_{[0,\theta]}(x)$, $\Theta = (0,\infty)$, $\mu = \lambda$. (R1) fails. On the support, $\log p_\theta(x) = -\log\theta$, so $s(\theta;x) = -1/\theta$ and $\mathbb E_\theta[s] = -1/\theta \ne 0$: the first conclusion fails. Also $\partial^2_\theta\log p_\theta = 1/\theta^2$, so $-\mathbb E[\partial^2\log p] = -1/\theta^2 < 0$, while $\operatorname{Var}(s) = 0$: the information equality fails, and indeed the "negative expected Hessian" is *negative*, which no variance can be. The mechanism: $\int_0^\theta \theta^{-1}dx = 1$ identically, and differentiating the identity in $\theta$ produces two terms — an interior term $\int_0^\theta \partial_\theta\theta^{-1}dx = -1/\theta$ and a boundary term $\theta^{-1}\cdot\frac{d\theta}{d\theta} = +1/\theta$ — which cancel. The score computation keeps only the interior term. (R1) is precisely the condition that kills the boundary term.

*Can (R1) be weakened?* Yes, but carefully. What is really needed is that the boundary contributes nothing to the derivative of $\int p_\theta d\mu$. Sufficient replacements: the support may depend on $\theta$ provided $p_\theta$ vanishes continuously at the moving boundary at a sufficient rate (so the Leibniz boundary term $p_\theta(\text{boundary})\cdot\partial_\theta(\text{boundary})$ is zero), or provided the family is dominated by a fixed measure and the *union* of supports differs from each individual support by a $\mu$-null set. The uniform family fails both: $p_\theta$ jumps from $1/\theta$ to $0$ at the boundary, and the supports differ by sets of positive Lebesgue measure. A useful contrast: the exponential family $p_\theta(x) = \theta e^{-\theta x}$ on $(0,\infty)$ has $\theta$-free support and is perfectly regular; the shifted exponential $p_\theta(x)=e^{-(x-\theta)}\mathbf 1_{x>\theta}$ has moving support and behaves like the uniform (rate $n$, non-normal limit).

**C2.** The proof is Theorem 8.25; here are the four requested points, which are what a referee would check. (A fifth point a referee would also check, handled in the theorem's proof and worth repeating in your own write-up: in dimension $d>1$ there is no single mean-value intermediate point valid for the whole score vector, so either use the integral form of the Taylor remainder, $\bar H_n = \frac1n\int_0^1\nabla^2\ell_n(\theta_0+t(\hat\theta_n-\theta_0))\,dt$, or apply the scalar mean-value theorem row by row with a different $\tilde\theta_{n,j}$ per row; both are controlled by the same uniform supremum in (ii) below.)

*(i) Where consistency is used, and why it cannot be proved here.* Twice. First, to conclude that $\hat\theta_n$ eventually lies in the interior of $\Theta$, which is what makes $\nabla\ell_n(\hat\theta_n)=0$ available — for a boundary maximizer the score need not vanish (Worked Example 8.3 with $k=0$; Worked Example 8.5 always). Second, to conclude $\tilde\theta_n\to_p\theta_0$ for the intermediate point in the Taylor expansion, so the Hessian is evaluated where the LLN applies. The theorem cannot prove consistency because its entire argument is *local*: it studies $\ell_n$ in a shrinking neighborhood of $\theta_0$ and says nothing about whether the global maximizer is in that neighborhood. Consistency is a global statement and needs the global argument of Theorem 8.24 (KL identification plus uniform convergence on all of $\Theta$).

*(ii) Where uniformity is required.* In showing $\frac1n\nabla^2\ell_n(\tilde\theta_n)\to_p -I(\theta_0)$. The pointwise SLLN gives $\frac1n\nabla^2\ell_n(\theta)\to_{a.s.}-I(\theta)$ for each *fixed* $\theta$. But $\tilde\theta_n$ is random and data-dependent, so no fixed-$\theta$ statement applies. With only pointwise convergence one can construct sequences $\theta_n\to\theta_0$ along which $\frac1n\nabla^2\ell_n(\theta_n)$ fails to converge: the null sets on which the SLLN fails vary with $\theta$, and their uncountable union can be everything. The fix is a uniform LLN on a closed ball $B(\theta_0,\varepsilon)$ — $\sup_{\theta\in B}\|\frac1n\nabla^2\ell_n(\theta)+I(\theta)\|\to_p0$ — which needs an integrable envelope for the second derivatives on $B$ (hypothesis (iv)) plus continuity of $\theta\mapsto I(\theta)$. Then $\|\frac1n\nabla^2\ell_n(\tilde\theta_n)+I(\tilde\theta_n)\|\le\sup_B\to_p0$ and $I(\tilde\theta_n)\to_p I(\theta_0)$ by continuity and the continuous mapping theorem.

*(iii) CLT and Slutsky.* The CLT is applied to $\frac{1}{\sqrt n}\sum_i s(\theta_0;X_i)$, a normalized sum of i.i.d. mean-zero vectors with covariance $I(\theta_0)$ — the mean-zero property is Proposition 8.14 and the covariance identification is Theorem 8.15. Slutsky is applied to the product $\big(-\frac1n\nabla^2\ell_n(\tilde\theta_n)\big)^{-1}\cdot\frac{1}{\sqrt n}\nabla\ell_n(\theta_0)$: a random matrix converging in probability to a constant times a sequence converging in distribution. Without Slutsky one would only know the two factors' marginal limits, not the limit of the product.

*(iv) What fails under misspecification, and what replaces it.* Nothing fails in the expansion itself. What fails is the *identification of two quantities as the same matrix*. The score at the pseudo-true $\theta^\star$ still has mean zero, but for a different reason — the first-order condition defining $\theta^\star$ as the KL projection, not Proposition 8.14 — and its covariance is $G := \mathbb E_{P^\star}[ss^\top]$. The limiting average Hessian is $-H := \mathbb E_{P^\star}[\nabla^2\log p_{\theta^\star}]$. Under correct specification Theorem 8.15 gives $G=H=I$ and the cancellation $H^{-1}GH^{-1}=I^{-1}$ occurs. Under misspecification there is no such theorem and the conclusion becomes $\sqrt n(\hat\theta_n-\theta^\star)\to_d\mathcal N(0,H^{-1}GH^{-1})$ (Theorem 8.28). Note which parts of the standard toolkit remain valid: the point estimate is still consistent for a well-defined target, and the limit is still normal at rate $\sqrt n$; only the variance formula changes. That is why the fix is a different covariance estimator, not a different estimator.

**C3.** *Derivation.* For any $q\ll p_\theta(\cdot\mid x)$,
$$\log p_\theta(x) = \mathbb E_q\Big[\log\frac{p_\theta(x,Z)}{q(Z)}\Big] + D_{\mathrm{KL}}\big(q\|p_\theta(\cdot\mid x)\big) = \mathcal L(q,\theta) + D_{\mathrm{KL}}\big(q\|p_\theta(\cdot\mid x)\big),$$
verified by writing $p_\theta(x,z)=p_\theta(x)p_\theta(z\mid x)$ inside the logarithm and splitting. Since KL $\ge0$ (Ch. 4), $\mathcal L$ is a lower bound, tight iff $q = p_\theta(\cdot\mid x)$ a.e. EM alternates: E-step maximizes $\mathcal L$ in $q$ (answer: the posterior), M-step in $\theta$ (answer: maximize $Q(\theta\mid\theta^{(t)}) = \mathbb E_{q^{(t+1)}}[\log p_\theta(x,Z)]$, the entropy term being $\theta$-free).

*Monotonicity.* As in Theorem 8.39: $\log p_{\theta^{(t+1)}}(x)\ \ge\ \mathcal L(q^{(t+1)},\theta^{(t+1)})\ \ge\ \mathcal L(q^{(t+1)},\theta^{(t)})\ =\ \log p_{\theta^{(t)}}(x)$, using the ELBO bound, the M-step, and E-step tightness respectively.

*Approximate E-step.* Restrict to $q\in\mathcal Q$ with $p_\theta(\cdot\mid x)\notin\mathcal Q$. Then step (c) — the final equality — becomes a strict inequality in the *unhelpful* direction: $\mathcal L(q^{(t+1)},\theta^{(t)}) = \log p_{\theta^{(t)}}(x) - \varepsilon_t$ with $\varepsilon_t := D_{\mathrm{KL}}(q^{(t+1)}\|p_{\theta^{(t)}}(\cdot\mid x)) > 0$. Carrying the gaps through,
$$\log p_{\theta^{(t+1)}}(x) - \log p_{\theta^{(t)}}(x) = \underbrace{\big[\mathcal L(q^{(t+1)},\theta^{(t+1)})-\mathcal L(q^{(t+1)},\theta^{(t)})\big]}_{\ge0 \text{ (M-step)}} + \underbrace{\big[\varepsilon_{t+1}' - \varepsilon_t\big]}_{\text{sign unknown}},$$
where $\varepsilon'_{t+1} := D_{\mathrm{KL}}(q^{(t+1)}\|p_{\theta^{(t+1)}}(\cdot\mid x))$. The likelihood decreases whenever the approximation gap shrinks by more than the ELBO gains — i.e. whenever the M-step moves $\theta$ toward a region where the posterior happens to be *better approximable* by $\mathcal Q$, at the cost of fit. Variational EM therefore has a systematic bias toward parameter values whose posteriors are close to the variational family (posteriors that are more factorized, more Gaussian, less multimodal), a well-documented pathology.

*What survives.* Monotonicity of the bound: $\mathcal L(q^{(t+1)},\theta^{(t+1)})\ge\mathcal L(q^{(t+1)},\theta^{(t)})\ge\mathcal L(q^{(t)},\theta^{(t)})$, the second inequality because $q^{(t+1)}$ maximizes $\mathcal L(\cdot,\theta^{(t)})$ over $\mathcal Q\ni q^{(t)}$. So variational EM is coordinate ascent on $\mathcal L$ and $\mathcal L$ never decreases — you are guaranteed to improve a lower bound, which is not the same as improving what it bounds.

*A two-point counterexample.* Let $Z\in\{1,2\}$, $x$ fixed, and let $\mathcal Q = \{q_0\}$ with $q_0 = (\tfrac12,\tfrac12)$ (a maximally rigid variational family, so the E-step is forced). Consider two candidate parameters:
$$\theta = A: \ \big(p_A(x,1), p_A(x,2)\big) = (0.79,\ 0.01); \qquad \theta = B:\ \big(p_B(x,1),p_B(x,2)\big) = (0.25,\ 0.25).$$
Then (all logs natural, computed numerically):

| $\theta$ | $p_\theta(x)$ | $\log p_\theta(x)$ | posterior $p_\theta(\cdot\mid x)$ | $D_{\mathrm{KL}}(q_0\|p_\theta(\cdot\mid x))$ | $\mathcal L(q_0,\theta)$ |
|---|---|---|---|---|---|
| $A$ | $0.80$ | $-0.2231$ | $(0.9875, 0.0125)$ | $1.5042$ | $-1.7273$ |
| $B$ | $0.50$ | $-0.6931$ | $(0.5,0.5)$ | $0.0000$ | $-0.6931$ |

Start at $\theta^{(t)}=A$. The E-step returns $q_0$ (no choice). The M-step maximizes $\mathcal L(q_0,\cdot)$ over $\{A,B\}$ and selects $B$, since $-0.6931 > -1.7273$. But the observed-data log-likelihood has *decreased*, from $-0.2231$ to $-0.6931$: a loss of $0.4700$ nats. The ELBO rose by $1.0342$ nats while the approximation gap fell by $1.5042$; the difference, $-0.4700$, is the drop in the likelihood, exactly as the identity above predicts. What the algorithm did is perfectly rational given its objective — it moved to the parameter whose posterior its variational family can represent — and perfectly wrong given the objective we claimed to care about. Every reported VAE ELBO carries this hazard, which is why ELBO improvements should not be quoted as likelihood improvements without an independent estimate of the gap (importance-weighted bounds, annealed importance sampling).

---

### References and further reading

**Textbooks and monographs.**

- E. L. Lehmann and G. Casella, *Theory of Point Estimation*, 2nd ed., Springer, 1998. Canonical for §8.5, §8.7, §8.8: Cramér–Rao (§2.5), Rao–Blackwell and Lehmann–Scheffé (§2.1, §5.1), Stein's lemma and James–Stein (§5.2, §5.5), admissibility (§5.7), Chapman–Robbins (§2.5, Problem 2.5.11).
- A. W. van der Vaart, *Asymptotic Statistics*, Cambridge, 1998. The reference for §8.6: consistency of M-estimators and asymptotic normality (Ch. 5), misspecification (§5.6), Le Cam–Hájek and superefficiency (Ch. 8–9).
- G. Casella and R. L. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002. Gentler, excellent for §8.1–§8.3.
- P. Bickel and K. Doksum, *Mathematical Statistics*, Vol. 1, 2nd ed., 2015. Unusually careful about regularity conditions.
- B. Efron and T. Hastie, *Computer Age Statistical Inference*, Cambridge, 2016. Ch. 7 on James–Stein and empirical Bayes is the best short treatment I know.
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009. §3.4 (ridge/lasso as MAP), §7 (bias–variance, effective degrees of freedom).
- O. Barndorff-Nielsen and D. R. Cox, *Inference and Asymptotics*, Chapman & Hall, 1994. Profile and modified profile likelihood, done properly.
- S. Amari, *Information Geometry and Its Applications*, Springer, 2016. Fisher–Rao and natural gradient.
- C. Robert, *The Bayesian Choice*, 2nd ed., Springer, 2007, Ch. 2, 8: MAP, admissibility, decision-theoretic shrinkage.

**Papers.**

- C. Stein, "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution," *Proc. Third Berkeley Symp.*, 1956; W. James and C. Stein, "Estimation with quadratic loss," *Proc. Fourth Berkeley Symp.*, 1961; C. Stein, "Estimation of the mean of a multivariate normal distribution," *Ann. Statist.* 9(6), 1981 (the identity used in Theorem 8.38).
- H. White, "Maximum likelihood estimation of misspecified models," *Econometrica* 50(1), 1982 — Theorem 8.28.
- P. J. Huber, "The behavior of maximum likelihood estimates under nonstandard conditions," *Proc. Fifth Berkeley Symp.*, 1967 — the other half of the sandwich's parentage.
- B. Efron and D. V. Hinkley, "Assessing the accuracy of the maximum likelihood estimator: observed versus expected Fisher information," *Biometrika* 65(3), 1978.
- A. P. Dempster, N. M. Laird and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," *JRSS-B* 39(1), 1977; C. F. J. Wu, "On the convergence properties of the EM algorithm," *Ann. Statist.* 11(1), 1983.
- R. Neal and G. Hinton, "A view of the EM algorithm that justifies incremental, sparse, and other variants," in *Learning in Graphical Models*, 1998 — the ELBO view of §8.9 in its original ML form.
- L. Hansen, "Large sample properties of generalized method of moments estimators," *Econometrica* 50(4), 1982 — the GMM paragraph of §8.2.3.
- S. Amari, "Natural gradient works efficiently in learning," *Neural Computation* 10(2), 1998; J. Martens and R. Grosse, "Optimizing neural networks with Kronecker-factored approximate curvature," *ICML* 2015 — the information equality put to work.
- D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," *ICLR* 2014; Y. Burda, R. Grosse and R. Salakhutdinov, "Importance weighted autoencoders," *ICLR* 2016 — approximate E-steps, and how to measure the gap that C3 exhibits.
- C. Zhang, S. Bengio, M. Hardt, B. Recht and O. Vinyals, "Understanding deep learning requires rethinking generalization," *ICLR* 2017 — the empirical result that forced §8.10 to be written.
- M. Belkin, D. Hsu, S. Ma and S. Mandal, "Reconciling modern machine learning practice and the classical bias–variance trade-off," *PNAS* 116(32), 2019 — double descent named.
- P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak and I. Sutskever, "Deep double descent: where bigger models and more data hurt," *JSTAT* 2021 — including the sample-wise version reproduced in Lab B5.
- P. L. Bartlett, P. M. Long, G. Lugosi and A. Tsigler, "Benign overfitting in linear regression," *PNAS* 117(48), 2020; T. Hastie, A. Montanari, S. Rosset and R. J. Tibshirani, "Surprises in high-dimensional ridgeless least squares interpolation," *Ann. Statist.* 50(2), 2022 — the theory behind Figure 8.6.
- L. Dinh, R. Pascanu, S. Bengio and Y. Bengio, "Sharp minima can generalize for deep nets," *ICML* 2017 — why raw curvature comparisons are a units error.
- S. Watanabe, "Algebraic geometry and statistical learning theory," Cambridge, 2009 — what to do when $I(\theta)$ is singular, as it is for every neural network.
