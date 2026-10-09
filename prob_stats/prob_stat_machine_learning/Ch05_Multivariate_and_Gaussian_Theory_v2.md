# Chapter 5 — Multivariate and Gaussian Theory

## Chapter 5 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 5.1 | The covariance matrix is a bilinear form; $\operatorname{Var}(a^\top X) = a^\top \Sigma a$ forces $\Sigma \succeq 0$; and what happens when you estimate it from $n$ samples | Why "uncorrelated" is a weaker claim than you want; why $\hat\Sigma_n$ is singular whenever $n\le d$, which silently breaks Mahalanobis scoring and whitening in high dimensions |
| 5.2 | Eigendecomposition, whitening, PCA, Mahalanobis geometry, and $\log\det\Sigma$ as entropy | PCA, ZCA whitening, Mahalanobis outlier scores, the origin of every "volume penalty" in a variational objective, and the exact sense in which "decorrelated" $\ne$ "disentangled" |
| 5.3 | Three equivalent definitions of the multivariate Gaussian; only one survives singular $\Sigma$; and the high-dimensional "soap bubble" | Understanding degenerate latent-variable models, why VAE posteriors need $\sigma > 0$, why "the Gaussian density" is the wrong starting definition, and why the latent-space mode is nowhere near a typical sample |
| 5.4 | Marginals, conditionals, Schur complements, precision matrices | The conditional mean is *linear* and the conditional covariance is *constant* — the single fact that makes Kalman filters, GPs, and Gaussian graphical models tractable |
| 5.5 | A reusable toolkit: sums, products, the linear-Gaussian lemma, and closed-form KL | The algebra behind every variational-Gaussian derivation in Chapters 10 and 13 — and the exact line of code that is the VAE's KL term, plus why it rewards posterior collapse |
| 5.6 | Linear-Gaussian models: Bayesian regression, Kalman filtering, factor analysis, PPCA | Closed-form Bayesian inference; state-space models; why PPCA and PCA agree in the noise-free limit |
| 5.7 | Variance propagation through a deep network | A first-principles derivation of He ($2/n_{\text{in}}$) and Xavier ($2/(n_{\text{in}}+n_{\text{out}})$) initialization, including where the factor of 2 comes from |
| 5.8 | Gaussian processes as consistent families of Gaussian marginals | GP regression *is* the conditioning formula of §5.4; kernels as assumptions; the $O(n^3)$ wall; NNGP/NTK |

---

## 5.0 Orientation: the one distribution you can compute with

Almost every closed-form calculation in machine learning is a Gaussian calculation. When a paper writes down a posterior and it has a formula rather than a sampler, the model was linear-Gaussian. When a variational bound has an analytic KL term, both distributions were Gaussian. When someone initializes a network with `std = sqrt(2/fan_in)`, they are quoting the conclusion of a variance-propagation argument that takes about two pages and that we will do in full. When a Bayesian optimization loop produces an uncertainty band, that band is a Gaussian conditional variance. The Gaussian is not merely one distribution among many; it is the distribution whose algebra is closed enough that you can do inference with a pen.

This chapter builds that algebra from the ground up, and it does so in a specific order that I want to flag in advance. We start with covariance *before* we start with Gaussians, because the covariance matrix is a purely second-moment object that makes sense for any square-integrable random vector, Gaussian or not, and because half of the confusion in practice comes from importing Gaussian intuitions into non-Gaussian settings. Then we do the geometry of $\Sigma$: eigendecomposition, whitening, PCA, ellipsoids. Only then do we define the Gaussian, and we define it as an *affine image of independent standard normals* rather than by its density. That choice will look pedantic for about three pages and then repay itself permanently, because it is the only definition that survives when $\Sigma$ is singular — which is exactly the regime that deep generative models live in.

Here is what I assume you bring. From Chapter 1, by number: **Definition 1.13** (measurable function, random variable, random vector), **Definition 1.16** (the law of $X$ as the pushforward $P_X = X_\# P$), **Definition 1.23** (independence of random variables), **Definition 1.24** (conditional independence), and the fact that a map into $\mathbb{R}^d$ is measurable if and only if each coordinate is. From Chapter 2, by name: densities with respect to Lebesgue measure $\lambda_d$ and the Radon–Nikodym derivative; Fubini–Tonelli for interchanging iterated integrals; dominated convergence for interchanging limits and expectations. From Chapter 3: the change-of-variables formula with Jacobians, and the exponential family. From Chapter 4: sub-Gaussianity and the concentration machinery built on moment generating functions. I will name these when I use them; I will not re-derive them.

The punchline, stated in advance, is a single sentence with three clauses. **For a jointly Gaussian vector, conditioning on part of the vector leaves you with a Gaussian whose mean is an affine function of the conditioning value and whose covariance does not depend on the conditioning value at all.** Every clause is doing work. "Gaussian" means the family is closed under conditioning, so you can condition repeatedly and never leave the family — that is the Kalman filter. "Mean is affine" means Bayesian prediction reduces to a matrix multiply — that is Gaussian process regression and the linear-Gaussian E-step. "Covariance does not depend on the conditioning value" means uncertainty can be precomputed before you see data — that is why GP error bars can be plotted before the $y$-values arrive, and it is also the single most misleading feature of the Gaussian, because in almost every real problem uncertainty *should* depend on where you are.

A word about the two payoffs I have promised. Section 5.7 derives He and Xavier initialization completely: forward pass, backward pass, the ReLU second moment $\mathbb{E}[\mathrm{ReLU}(Z)^2] = 1/2$ that produces the famous factor of $2$, and a numerical experiment on a 20-layer network confirming that $\operatorname{Var}(w) = 2/n_{\text{in}}$ keeps activation variance flat while $1/n_{\text{in}}$ and $4/n_{\text{in}}$ produce vanishing and exploding activations at precisely the predicted rates. Section 5.8 then shows that Gaussian process regression contains no new mathematics whatsoever: it is the conditioning formula of §5.4 applied to a vector whose index set happens to be infinite. If, after §5.8, GP regression still feels like a separate subject, I have failed somewhere in §5.4 and you should go back.

Finally, a promise about difficulty. Section 5.4 contains the hardest algebra in the chapter — block matrix inversion and completing the square in a quadratic form of a vector variable. I am not going to pretend it is easy. I am going to do it three times: once by brute-force completing the square, once by an "invent the answer and verify" route, and once by a slick argument that constructs an explicit independent decomposition. If one of the three does not land, another will.

---

## 5.1 Random vectors, mean, and covariance

### 5.1.1 What a covariance matrix actually is

Start with intuition. A random vector $X = (X_1,\dots,X_d)^\top$ has $d$ coordinates, each of which wobbles around its mean. The variance of coordinate $i$ tells you how much $X_i$ wobbles. But there is more information available: coordinates can wobble *together*. If $X_1$ is above its mean, is $X_2$ typically above or below its? The covariance $\operatorname{Cov}(X_1,X_2)$ is the average product of the two deviations, and its sign answers that question.

The covariance *matrix* collects all $d^2$ such numbers. But the right way to think about it is not "a table of pairwise numbers." The right way is: **the covariance matrix is a machine that eats a direction and reports how much the cloud of data spreads in that direction.** Feed it the direction $a$; it returns $a^\top \Sigma a$, which is exactly the variance of the scalar random variable $a^\top X$, the projection of $X$ onto $a$. Everything in §5.1 and §5.2 follows from taking that sentence seriously.

**Definition 5.1 (mean vector, covariance matrix, cross-covariance).**
Let $X$ be a random vector in $\mathbb{R}^d$ on a probability space $(\Omega,\mathcal{F},P)$, and suppose $\mathbb{E}[\|X\|^2] < \infty$ (equivalently, every $X_i \in L^2(P)$; the equivalence is immediate since $\|X\|^2 = \sum_i X_i^2$). Define the **mean vector** $\mu = \mathbb{E}[X] \in \mathbb{R}^d$ coordinatewise, $\mu_i := \mathbb{E}[X_i]$, and the **covariance matrix**
$$
\Sigma \;=\; \operatorname{Cov}(X) \;:=\; \mathbb{E}\big[(X-\mu)(X-\mu)^\top\big] \in \mathbb{R}^{d\times d},
\qquad \Sigma_{ij} = \mathbb{E}\big[(X_i-\mu_i)(X_j-\mu_j)\big].
$$
For a second random vector $Y$ in $\mathbb{R}^m$ with $\mathbb{E}[\|Y\|^2]<\infty$ and mean $\nu$, the **cross-covariance** is $\operatorname{Cov}(X,Y) := \mathbb{E}[(X-\mu)(Y-\nu)^\top] \in \mathbb{R}^{d\times m}$. Note $\operatorname{Cov}(X) = \operatorname{Cov}(X,X)$ and $\operatorname{Cov}(Y,X) = \operatorname{Cov}(X,Y)^\top$.

Two remarks on hypotheses. First, the square-integrability assumption is not decorative: a Cauchy random vector has no mean and no covariance, and a $t_2$-distributed vector has a mean but infinite covariance. In ML this matters more than people expect — heavy-tailed gradient noise is documented in language-model training, and a covariance-based analysis is simply inapplicable there. Second, existence of $\Sigma$ requires only *second* moments, not a density, not independence, not anything else. The covariance of the discrete vector of coin-flip indicators is perfectly well defined.

Expanding the product gives the computational form $\Sigma = \mathbb{E}[XX^\top] - \mu\mu^\top$, valid because expectation is linear on matrices entrywise and $\mathbb{E}[X\mu^\top] = \mu\mu^\top$. The scalar case $\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}X)^2$ is the $d=1$ instance.

> **Worked Example 5.1 (two dice — the running example from Chapter 1).**
> Roll two fair dice. Let $X$ be the first die and $Y$ the sum. Chapter 1 used this pair to make the point that $P(X=1, Y=7) = P(X=1)P(Y=7)$ exactly, even though $X$ and $Y$ are dependent — verifying independence on one pair of events proves nothing. Let us now compute the second-moment structure.
>
> Write $Y = X + Z$ with $Z$ the second die, $X \perp Z$. Then $\mathbb{E}[X] = 3.5$, $\mathbb{E}[Y] = 7$. For the variance of a single fair die,
> $$\operatorname{Var}(X) = \tfrac{1}{6}\sum_{k=1}^{6}k^2 - 3.5^2 = \tfrac{91}{6} - \tfrac{49}{4} = \tfrac{182-147}{12} = \tfrac{35}{12} \approx 2.9167.$$
> Since $X$ and $Z$ are independent, $\operatorname{Var}(Y) = \operatorname{Var}(X)+\operatorname{Var}(Z) = \tfrac{70}{12} \approx 5.8333$. And
> $$\operatorname{Cov}(X,Y) = \operatorname{Cov}(X, X+Z) = \operatorname{Var}(X) + \operatorname{Cov}(X,Z) = \tfrac{35}{12} + 0 = \tfrac{35}{12}.$$
> So
> $$\Sigma = \begin{pmatrix} 35/12 & 35/12 \\ 35/12 & 70/12\end{pmatrix} \approx \begin{pmatrix} 2.9167 & 2.9167 \\ 2.9167 & 5.8333\end{pmatrix},\qquad \rho_{XY} = \frac{35/12}{\sqrt{(35/12)(70/12)}} = \frac{1}{\sqrt 2} \approx 0.7071.$$
> (Exhaustive enumeration over all 36 outcomes confirms every entry to machine precision.) Notice what the covariance matrix does and does not know. It knows $X$ and $Y$ are strongly positively associated. It does *not* know the Chapter 1 fact that one particular pair of events factorizes. Second moments are a lossy summary — a deliberate, useful, lossy summary.

> **Worked Example 5.2 (three coin flips — the other Chapter 1 running example).**
> $\Omega = \{H,T\}^3$ with the uniform measure, $I_k = \mathbf{1}\{k\text{-th flip is }H\}$, and $S = I_1+I_2+I_3$ the number of heads. Each $I_k$ is Bernoulli$(1/2)$, so $\mathbb{E}[I_k] = 1/2$ and $\operatorname{Var}(I_k) = 1/4$. The flips are independent, so
> $$\operatorname{Cov}(I) = \tfrac14 I_3 = \begin{pmatrix}0.25&0&0\\0&0.25&0\\0&0&0.25\end{pmatrix}.$$
> Now take the direction $a = (1,1,1)^\top$. The projection $a^\top I$ is exactly $S$, so the "machine" reading of the covariance matrix predicts
> $$\operatorname{Var}(S) = a^\top \Sigma a = \tfrac14\|a\|^2 = \tfrac34 = 0.75,$$
> which matches the direct computation $\operatorname{Var}(S) = 3\cdot\frac14$. Enumerating all eight outcomes gives $\operatorname{Var}(S) = 0.75$ exactly. The point is not that we needed the matrix — it is that the matrix contains the variance of *every* linear statistic of the flips at once, and we extracted one of them by plugging in a direction.

### 5.1.2 Positive semidefiniteness, derived rather than asserted

Textbooks often assert "$\Sigma$ is symmetric positive semidefinite" as though it were part of the definition. It is not. It is a theorem, and the proof is one line, and that line is the entire conceptual content of the covariance matrix.

**Proposition 5.2 (the covariance matrix as a quadratic form).**
Let $X$ be a random vector in $\mathbb{R}^d$ with $\mathbb{E}\|X\|^2 < \infty$, mean $\mu$, covariance $\Sigma$. Then for every $a \in \mathbb{R}^d$:

1. $\mathbb{E}[a^\top X] = a^\top \mu$ and $\operatorname{Var}(a^\top X) = a^\top \Sigma a$.
2. $\Sigma$ is symmetric and positive semidefinite, written $\Sigma \succeq 0$.
3. $a^\top \Sigma a = 0$ if and only if $a^\top X = a^\top \mu$ almost surely. Consequently, if $\Sigma$ is singular with $\Sigma a = 0$ for some $a \ne 0$, then $X$ lies almost surely in the affine hyperplane $H_a := \{x : a^\top x = a^\top\mu\}$, which has $\lambda_d$-measure zero; hence $P_X$ has **no** density with respect to $\lambda_d$.

*Proof.* (1) Linearity of expectation gives the mean. For the variance, the scalar $a^\top X - a^\top\mu = a^\top(X-\mu)$ satisfies
$$
\operatorname{Var}(a^\top X) = \mathbb{E}\big[\big(a^\top (X-\mu)\big)^2\big] = \mathbb{E}\big[a^\top (X-\mu)(X-\mu)^\top a\big] = a^\top \mathbb{E}\big[(X-\mu)(X-\mu)^\top\big] a = a^\top \Sigma a,
$$
where we used that a scalar equals its own transpose to write $s^2 = s\cdot s^\top$ with $s = a^\top(X-\mu)$, and pulled the constant $a$ out of the expectation (legitimate entrywise by linearity, and every entry is integrable because $|(X_i-\mu_i)(X_j-\mu_j)| \le \frac12((X_i-\mu_i)^2+(X_j-\mu_j)^2)$ is integrable by assumption).

(2) Symmetry is immediate from $\Sigma_{ij} = \mathbb{E}[(X_i-\mu_i)(X_j-\mu_j)] = \Sigma_{ji}$. For semidefiniteness: $a^\top \Sigma a = \operatorname{Var}(a^\top X) \ge 0$ because a variance is the expectation of a nonnegative random variable. That is the whole argument. $\Sigma \succeq 0$ *because variances are nonnegative*.

(3) $\operatorname{Var}(W) = 0$ for a scalar $W$ implies $W = \mathbb{E}[W]$ a.s.: indeed $\mathbb{E}[(W-\mathbb{E}W)^2] = 0$ and a nonnegative random variable with zero expectation is zero a.s. (a standard consequence of Markov's inequality, or of monotone convergence applied to $\{|W - \mathbb{E}W| > 1/n\}$). Conversely if $a^\top X$ is a.s. constant its variance is $0$. If $\Sigma a = 0$ with $a \ne 0$ then in particular $a^\top \Sigma a = 0$, so $X \in H_a$ a.s. The hyperplane $H_a$ is a translate of a $(d-1)$-dimensional subspace and therefore $\lambda_d(H_a) = 0$ (Chapter 2). If $P_X$ had a density $p$ w.r.t. $\lambda_d$ we would get $1 = P_X(H_a) = \int_{H_a} p \,d\lambda_d = 0$, a contradiction. $\square$

Part (3) is the bridge to Chapter 1's central warning. Chapter 1 showed that the circle generator $G(t) = (\cos t, \sin t)$ pushes a one-dimensional latent onto a set of Lebesgue measure zero in $\mathbb{R}^2$, so $P_\theta = G_\#\,\mathrm{Unif}$ has no density. Proposition 5.2(3) exhibits a *second*, purely linear-algebraic route to the same pathology: a rank-deficient covariance forces the whole distribution into a hyperplane. In a deep generative model $P_\theta = (G_\theta)_\#\mathcal N(0,I_k)$ with $k < d$, *both* mechanisms are present. If $G_\theta$ happens to be affine, the covariance is $A A^\top$ with $A \in \mathbb{R}^{d\times k}$, which has rank at most $k < d$ and is therefore singular; Proposition 5.2(3) applies directly. If $G_\theta$ is a genuinely nonlinear network, the covariance can be full rank while the support is still measure-zero, and only the Chapter 1 argument applies. Which brings us to the first confusion.

> **Common Confusion 5.1.**
> *"If $\Sigma$ is invertible, then $X$ spreads out in all $d$ directions, so surely it has a density?"*
>
> No, and the counterexample is already in your hands. Take $T \sim \mathrm{Unif}[0,2\pi)$ and $X = (\cos T, \sin T)$ — Chapter 1's circle. Then $\mathbb{E}[\cos T] = \mathbb{E}[\sin T] = 0$, $\mathbb{E}[\cos^2 T] = \mathbb{E}[\sin^2 T] = 1/2$, and $\mathbb{E}[\cos T\sin T] = \frac{1}{2}\mathbb{E}[\sin 2T] = 0$. So
> $$\Sigma = \begin{pmatrix} 1/2 & 0 \\ 0 & 1/2\end{pmatrix},$$
> which is invertible, with $\det\Sigma = 1/4 > 0$. (Monte Carlo with $n = 2\times10^6$ samples, seed 0, gives empirical covariance $\begin{pmatrix}0.50005 & -0.00002\\ -0.00002 & 0.49995\end{pmatrix}$.) And yet $X$ lives on the unit circle, a set of $\lambda_2$-measure zero, so $P_X$ has no Lebesgue density.
>
> The logic of Proposition 5.2(3) runs one way only. Singular $\Sigma$ $\Rightarrow$ no density. Nonsingular $\Sigma$ $\Rightarrow$ nothing. What nonsingularity buys you is that $X$ is not confined to a *hyperplane*; it says nothing about curved measure-zero sets. This is the exact reason a GAN's discriminator can achieve perfect separation even when the generator's outputs have a healthy, full-rank empirical covariance — full-rank covariance is not evidence of a density, and diagnosing mode collapse by looking at $\det\hat\Sigma$ is diagnosing the wrong thing.

**Proposition 5.3 (affine transformation).**
Let $X$ be a random vector in $\mathbb{R}^d$ with mean $\mu$, covariance $\Sigma$; let $A \in \mathbb{R}^{m\times d}$, $b \in \mathbb{R}^m$, and $Y = AX+b$. Then $\mathbb{E}[Y] = A\mu + b$ and
$$
\operatorname{Cov}(Y) = A\Sigma A^\top.
$$
More generally, for random vectors $X, Z$ and matrices $A, B$ of compatible shapes, $\operatorname{Cov}(AX + b, BZ + c) = A\operatorname{Cov}(X,Z)B^\top$; and if $X, Z$ take values in the same $\mathbb{R}^d$,
$$
\operatorname{Cov}(X+Z) = \operatorname{Cov}(X) + \operatorname{Cov}(X,Z) + \operatorname{Cov}(Z,X) + \operatorname{Cov}(Z).
$$

*Proof.* $Y - \mathbb{E}Y = A(X-\mu)$, so $\mathbb{E}[(Y-\mathbb{E}Y)(Y-\mathbb{E}Y)^\top] = \mathbb{E}[A(X-\mu)(X-\mu)^\top A^\top] = A\Sigma A^\top$, moving the constant matrices outside the (entrywise) expectation. The bilinear statements are the same computation with two different centered vectors, and the sum formula is bilinearity expanded. $\square$

Observe that Proposition 5.3 *contains* Proposition 5.2(1) as the case $A = a^\top$, $m=1$. It also gives an alternative, coordinate-free proof of PSD-ness: $\Sigma = \operatorname{Cov}(X)$ satisfies $a^\top\Sigma a = \operatorname{Cov}(a^\top X) \ge 0$. And it is the workhorse of the entire chapter: whitening, PCA, the Gaussian definition, the Kalman filter, and the initialization derivation are all instances of "push a covariance through a linear map."

**Definition 5.4 (correlation).** If $\Sigma_{ii} > 0$ for all $i$, write $D = \operatorname{diag}(\sqrt{\Sigma_{11}},\dots,\sqrt{\Sigma_{dd}})$ and define the **correlation matrix** $R := D^{-1}\Sigma D^{-1}$, with entries $\rho_{ij} = \Sigma_{ij}/\sqrt{\Sigma_{ii}\Sigma_{jj}}$. Note $R$ is the covariance matrix of the standardized vector $D^{-1}(X-\mu)$, hence $R \succeq 0$ with unit diagonal.

**Proposition 5.5 (Cauchy–Schwarz for covariance).** $|\operatorname{Cov}(X_i,X_j)| \le \sqrt{\operatorname{Var}(X_i)\operatorname{Var}(X_j)}$, so $|\rho_{ij}| \le 1$; equality holds iff $X_i - \mu_i = c(X_j - \mu_j)$ a.s. for some constant $c$ (with $c>0$ for $\rho=+1$, $c<0$ for $\rho=-1$).

*Proof.* This is the Cauchy–Schwarz inequality in $L^2(P)$ applied to the centered variables $U = X_i-\mu_i$, $V = X_j - \mu_j$: $|\mathbb{E}[UV]| \le \|U\|_2\|V\|_2$, with equality iff $U,V$ are linearly dependent in $L^2$, i.e. one is an a.s. scalar multiple of the other. Alternatively and more in the spirit of this chapter: the $2\times2$ submatrix $\Sigma_{\{i,j\}}$ is PSD by Proposition 5.2 (restrict $a$ to have support on $\{i,j\}$), and a symmetric $2\times2$ PSD matrix has nonnegative determinant, i.e. $\Sigma_{ii}\Sigma_{jj} - \Sigma_{ij}^2 \ge 0$. Equality means the submatrix is singular, and Proposition 5.2(3) turns singularity into an a.s. linear relation. $\square$

### 5.1.3 Uncorrelated is not independent

**Proposition 5.6.** If $X_i \perp X_j$ and both are in $L^2$, then $\operatorname{Cov}(X_i,X_j) = 0$. The converse is false.

*Proof.* Independence implies $\mathbb{E}[X_iX_j] = \mathbb{E}[X_i]\mathbb{E}[X_j]$: the joint law is the product measure (Definition 1.23), so this is Fubini's theorem (Ch. 2) applied to the product-measurable function $(u,v)\mapsto uv$, whose integrability follows from $\mathbb{E}|X_iX_j| \le \|X_i\|_2\|X_j\|_2 < \infty$. Then $\operatorname{Cov} = \mathbb{E}[X_iX_j] - \mathbb{E}[X_i]\mathbb{E}[X_j] = 0$. For the converse, see Worked Example 5.3. $\square$

> **Worked Example 5.3 (zero covariance, maximal dependence — the circle again).**
> $T \sim \mathrm{Unif}[0,2\pi)$, $X = \cos T$, $Y = \sin T$. We computed $\operatorname{Cov}(X,Y) = 0$ in Common Confusion 5.1, so $\rho_{XY} = 0$. Yet $X^2 + Y^2 = 1$ identically: knowing $X$ pins $|Y|$ exactly. The dependence could hardly be stronger, and correlation registers none of it.
>
> Why does correlation miss it? Because correlation measures *linear* association, and the map $x \mapsto \mathbb{E}[Y \mid X = x]$ here is $0$ for every $x$ by symmetry ($\pm\sqrt{1-x^2}$ are equally likely). The best linear predictor of $Y$ from $X$ is the constant $0$, and correlation is precisely a rescaled measure of how much better than constant the best linear predictor does. A second classic instance: $Z\sim\mathcal N(0,1)$, $Y = Z^2$. Then $\operatorname{Cov}(Z,Z^2) = \mathbb{E}[Z^3] = 0$ by symmetry, but $Y$ is a deterministic function of $Z$.
>
> The ML consequence is concrete and frequently ignored. Correlation-based feature screening — dropping features whose correlation with the target is below a threshold — will delete a feature that determines the target through any even or otherwise symmetric relationship. And "the residuals are uncorrelated with the fitted values" is a much weaker statement than "the model has extracted all the signal."

There is a standing temptation to reply: *but for Gaussians, uncorrelated does mean independent, and everything in deep learning is roughly Gaussian.* Two errors are compounded there. First, the Gaussian implication requires **joint** Gaussianity, not marginal Gaussianity of each coordinate — Corollary 5.23 states the true version, and Common Confusion 5.4 constructs two marginally normal, uncorrelated, wildly dependent variables in three lines. Second, "everything is roughly Gaussian" is doing enormous unexamined work. Post-ReLU activations are *never* Gaussian: they are exactly half a point mass at $0$ plus half a Gaussian tail, which is Chapter 1's running example $A = \max(0,Z)$, a distribution with neither a pmf nor a pdf. Attention logits are heavy-tailed. Gradient noise in large-batch training has repeatedly been measured to be non-Gaussian. Whenever you invoke "uncorrelated hence independent," you are invoking joint Gaussianity, and you should be able to say why it holds.

> **Self-check 5.1.**
> 1. $X$ has covariance $\Sigma = \begin{pmatrix} 3 & -1 \\ -1 & 2\end{pmatrix}$. What is $\operatorname{Var}(X_1 - 2X_2)$?
> 2. Can a covariance matrix have a negative diagonal entry? Can it have a negative eigenvalue? Can $\Sigma_{12} = 5$ while $\Sigma_{11} = \Sigma_{22} = 1$?
> 3. $Y = AX$ with $A$ of shape $2\times 5$. What is the largest possible rank of $\operatorname{Cov}(Y)$, and what does it mean if the rank is 1?
> 4. True or false: if $\operatorname{Cov}(X) = I_d$ then the coordinates of $X$ are independent.
> 5. You have $n = 20$ samples in $d = 50$ dimensions. Can you compute $\hat\Sigma_n^{-1}$?

### 5.1.4 Estimating $\mu$ and $\Sigma$ from data

Everything so far has treated $\mu$ and $\Sigma$ as known. In practice you have samples, and the estimators you form from them are themselves random. Two facts about those estimators cause real trouble in practice, and both follow from what we have already proved.

**Proposition 5.7 (sample mean and sample covariance).**
Let $X_1,\dots,X_n$ be i.i.d. copies of $X$ with mean $\mu$ and covariance $\Sigma$, and set
$$
\bar X_n := \frac1n\sum_{i=1}^n X_i,\qquad
\hat\Sigma_n := \frac1n\sum_{i=1}^n (X_i-\bar X_n)(X_i-\bar X_n)^\top .
$$
Then:
1. $\mathbb{E}[\bar X_n] = \mu$ and $\operatorname{Cov}(\bar X_n) = \Sigma/n$.
2. $\mathbb{E}[\hat\Sigma_n] = \frac{n-1}{n}\Sigma$. The rescaled estimator $S_n := \frac{n}{n-1}\hat\Sigma_n$ is unbiased.
3. $\operatorname{rank}(\hat\Sigma_n) \le \min(n-1,\ d)$. In particular $\hat\Sigma_n$ is **singular whenever $n \le d$**, so $\hat\Sigma_n^{-1}$ does not exist.
4. If $X\sim\mathcal N(\mu,\Sigma)$ with $\Sigma\succ0$ and $n>d$, the maximum-likelihood estimator of $(\mu,\Sigma)$ is $(\bar X_n,\ \hat\Sigma_n)$ — the *biased* version.

*Proof.* (1) Linearity gives the mean; Proposition 5.3 with $A = \frac1n(I\ \cdots\ I)$ applied to the stacked vector, whose covariance is block-diagonal by independence, gives $\operatorname{Cov}(\bar X_n) = \frac{1}{n^2}\sum_i\Sigma = \Sigma/n$.

(2) Write $Y_i := X_i-\mu$, so $\mathbb{E}[Y_iY_j^\top] = \delta_{ij}\Sigma$ and $X_i-\bar X_n = Y_i-\bar Y_n$. Then
$$
\mathbb{E}\big[(Y_i-\bar Y_n)(Y_i-\bar Y_n)^\top\big] = \Sigma - 2\,\mathbb{E}[Y_i\bar Y_n^\top] + \mathbb{E}[\bar Y_n\bar Y_n^\top] = \Sigma - \tfrac2n\Sigma + \tfrac1n\Sigma = \tfrac{n-1}{n}\Sigma,
$$
using $\mathbb{E}[Y_i\bar Y_n^\top] = \frac1n\Sigma$ (only the $j=i$ term survives) and part (1). Averaging over $i$ gives the claim.

(3) $\hat\Sigma_n = \frac1n C^\top C$ where $C$ is the $n\times d$ matrix with rows $(X_i-\bar X_n)^\top$. Those rows sum to $0$ by construction, so they span a subspace of dimension at most $n-1$; hence $\operatorname{rank}(C)\le\min(n-1,d)$, and $\operatorname{rank}(C^\top C) = \operatorname{rank}(C)$.

(4) This is a computation with matrix derivatives; it is recorded in §5.9 rather than reproduced here. $\square$

Part (2) is the origin of the $n$-versus-$n-1$ question that everyone meets once and then files away. The honest summary: the two estimators differ by a factor $n/(n-1)$, which matters at $n=5$ (a $20\%$ underestimate of every variance) and is irrelevant at $n=5000$. Simulating $20{,}000$ replicates at $n=5$ from $\Sigma = \begin{pmatrix}2&0.6&-0.4\\0.6&1&0.2\\-0.4&0.2&0.75\end{pmatrix}$ gives an average $\hat\Sigma_n$ of $\begin{pmatrix}1.605&0.478&-0.323\\0.478&0.797&0.161\\-0.323&0.161&0.604\end{pmatrix}$, against a prediction of $\frac45\Sigma = \begin{pmatrix}1.60&0.48&-0.32\\0.48&0.80&0.16\\-0.32&0.16&0.60\end{pmatrix}$. At $n=50$ the same experiment gives $1.9607$ in the $(1,1)$ entry against a predicted $1.96$.

This is not merely pedantry, and PyTorch's `BatchNorm` illustrates why. It normalizes each batch using the **biased** variance — because the goal there is to make *this batch* have unit variance, and the biased estimate is the one that does that exactly — while accumulating running statistics using the **unbiased** variance, because those are meant to estimate the population quantity used at inference time. Two different jobs, two different estimators, in the same twenty lines of code.

Part (3) is the one that breaks things. Any method that inverts an estimated covariance — Mahalanobis-based outlier and out-of-distribution detection (§5.2.3), Gaussian discriminant analysis, whitening a feature matrix, estimating a Gaussian graphical model (§5.4.4) — is silently assuming $n \gg d$. With $2048$-dimensional penultimate features and $500$ in-distribution samples per class, $\hat\Sigma_n$ has rank at most $499$ and is not invertible at all; with $3000$ samples it is invertible but its smallest eigenvalues are severely underestimated, and inverting amplifies exactly those. The standard repairs are shrinkage, $\hat\Sigma_n + \varepsilon I$ or Ledoit–Wolf, and sparsity penalties on the precision matrix. Exercise B5 makes the failure visible.

---

## 5.2 Whitening, PCA, and the geometry of $\Sigma$

### 5.2.1 The spectral picture

Proposition 5.2 said $\Sigma$ is symmetric PSD. Linear algebra tells us exactly what such matrices look like.

**Theorem 5.8 (spectral theorem for real symmetric matrices; proof cited).**
Every symmetric $\Sigma \in \mathbb{R}^{d\times d}$ admits a decomposition $\Sigma = UDU^\top$ where $U$ is orthogonal ($U^\top U = I$), its columns $u_1,\dots,u_d$ are eigenvectors, and $D = \operatorname{diag}(\lambda_1,\dots,\lambda_d)$ with $\lambda_1 \ge \dots \ge \lambda_d$ real. If moreover $\Sigma \succeq 0$ then all $\lambda_i \ge 0$; and $\Sigma \succ 0$ (positive definite) iff all $\lambda_i > 0$.

*A notation warning, worth thirty seconds now.* Most textbooks write $\Sigma = U\Lambda U^\top$, using $\Lambda$ for the eigenvalue diagonal. This chapter deliberately writes $D$ instead, because from §5.3.3 onward the symbol $\Lambda$ is reserved for a completely different and far more important object: the **precision matrix** $\Lambda = \Sigma^{-1}$, which is *not* diagonal in general and is not the eigenvalue matrix of anything we care about here. When you read papers, expect both conventions and let context disambiguate. A second, smaller clash to be aware of: the scalar $\lambda_i$ always denotes an eigenvalue, while $\lambda_d$ *applied to a set*, as in $\lambda_d(A)$, is Lebesgue measure on $\mathbb{R}^d$ (Chapter 2's convention) — an eigenvalue never takes an argument, so the two are always distinguishable, but Proposition 5.13 below uses both in one display and you should not be startled.

I do not prove the spectral theorem here; it belongs to linear algebra, and it is recorded in §5.9. But I do want to translate it. Writing $\Sigma = \sum_i \lambda_i u_iu_i^\top$ and applying Proposition 5.2(1) to a unit vector $a$:
$$
\operatorname{Var}(a^\top X) = a^\top \Sigma a = \sum_{i=1}^d \lambda_i (u_i^\top a)^2 .
$$
So the variance in direction $a$ is a *convex combination* of the eigenvalues, weighted by how much $a$ aligns with each eigenvector (the weights $(u_i^\top a)^2$ sum to $\|a\|^2 = 1$). Immediately: the maximum possible variance over unit directions is $\lambda_1$, attained at $a = u_1$; the minimum is $\lambda_d$, attained at $u_d$. That is the entire content of PCA, and we have just proved it — I will restate it formally in a moment.

**Definition 5.9 (matrix square root, Cholesky factor, whitening).**
For $\Sigma \succeq 0$ with spectral decomposition $UDU^\top$, the **symmetric square root** is $\Sigma^{1/2} := UD^{1/2}U^\top$ where $D^{1/2} = \operatorname{diag}(\sqrt{\lambda_i})$; it is the unique symmetric PSD matrix whose square is $\Sigma$. If $\Sigma \succ 0$, the **Cholesky factor** is the unique lower-triangular $L$ with positive diagonal such that $\Sigma = LL^\top$. Any matrix $B$ with $BB^\top = \Sigma$ is called a **factor** of $\Sigma$. Given $\Sigma \succ 0$, a **whitening transform** is any map $x \mapsto W(x-\mu)$ with $W\Sigma W^\top = I_d$; canonical choices are $W = D^{-1/2}U^\top$ (PCA whitening), $W = \Sigma^{-1/2}$ (ZCA whitening), and $W = L^{-1}$ (Cholesky whitening).

**Proposition 5.10 (whitening).** Let $\Sigma \succ 0$ and $W$ satisfy $W\Sigma W^\top = I$. Set $\tilde X := W(X-\mu)$. Then $\mathbb{E}\tilde X = 0$ and $\operatorname{Cov}(\tilde X) = I_d$. Moreover, $W$ is a whitening matrix iff $W = Q\Sigma^{-1/2}$ for some orthogonal $Q$; so whitening is unique only up to an arbitrary rotation.

*Proof.* The first claim is Proposition 5.3. For the characterization: if $W\Sigma W^\top = I$, put $Q := W\Sigma^{1/2}$; then $QQ^\top = W\Sigma W^\top = I$, so $Q$ is orthogonal and $W = Q\Sigma^{-1/2}$. Conversely $(Q\Sigma^{-1/2})\Sigma(Q\Sigma^{-1/2})^\top = Q\Sigma^{-1/2}\Sigma\Sigma^{-1/2}Q^\top = QQ^\top = I$. $\square$

The non-uniqueness is worth pausing on. ZCA whitening ($W = \Sigma^{-1/2}$) is the choice that is "closest to the identity" in the sense that it minimizes $\mathbb{E}\|\tilde X - (X-\mu)\|^2$ among whitening transforms — a genuine theorem, not a slogan; it is recorded in §5.9 with a reference, and the proof is a short exercise in the polar decomposition. This optimality is why ZCA preserves the visual appearance of images while PCA whitening scrambles them. The freedom $Q$ is precisely the reason that "whitening your features" does not identify a canonical basis, and it foreshadows the non-identifiability of factor analysis in §5.6.3.

> **Worked Example 5.4 (eigendecomposition and three whiteners, computed).**
> Let $\Sigma = \begin{pmatrix}4 & 2\\ 2& 3\end{pmatrix}$. Trace $=7$, determinant $= 12-4 = 8$, so eigenvalues solve $\lambda^2 - 7\lambda + 8 = 0$:
> $$\lambda_{1,2} = \frac{7\pm\sqrt{49-32}}{2} = \frac{7\pm\sqrt{17}}{2},\qquad \lambda_1 = 5.5616,\quad \lambda_2 = 1.4384 .$$
> Eigenvector for $\lambda_1$: solve $(4-\lambda_1)x + 2y = 0$, i.e. $y = (\lambda_1-4)x/2 = 0.7808x$, normalize: $u_1 = (0.7882, 0.6154)^\top$, at angle $\arctan(0.6154/0.7882) = 37.98^\circ$ from the $x$-axis. Then $u_2 = (-0.6154, 0.7882)^\top$. (Numerically verified: `numpy.linalg.eigh` returns eigenvalues $5.561553, 1.438447$ and the same eigenvectors up to sign.)
>
> Three factorizations of the same $\Sigma$:
> $$\Sigma^{1/2} = \begin{pmatrix}1.9194 & 0.5622\\ 0.5622 & 1.6383\end{pmatrix},\qquad L = \begin{pmatrix}2 & 0\\ 1 & \sqrt2\end{pmatrix},\qquad UD^{1/2} = \text{(rotate then scale)} .$$
> The Cholesky factor is exact and trivially checkable: $LL^\top = \begin{pmatrix}4 & 2\\2 & 1+2\end{pmatrix} = \Sigma$. The corresponding whiteners are $\Sigma^{-1/2} = \begin{pmatrix}0.5792 & -0.1988\\ -0.1988 & 0.6786\end{pmatrix}$ and $L^{-1} = \begin{pmatrix}0.5 & 0\\ -0.3536 & 0.7071\end{pmatrix}$; both satisfy $W\Sigma W^\top = I$ to machine precision. They differ by a rotation, exactly as Proposition 5.10 predicts.
>
> Practical note: for sampling and for solving linear systems you should use Cholesky, which costs $d^3/3$ flops and is numerically stable, not the eigendecomposition ($\approx 9d^3$) and never an explicit inverse. Every serious GP or Kalman implementation is a Cholesky implementation.

### 5.2.2 PCA in one theorem

**Theorem 5.11 (variance-maximizing directions).**
Let $\Sigma \succeq 0$ have eigenvalues $\lambda_1\ge\dots\ge\lambda_d$ with orthonormal eigenvectors $u_1,\dots,u_d$. Then
$$
\max_{\|a\|=1} a^\top \Sigma a = \lambda_1, \text{ attained at } a = u_1,
$$
and, recursively, $\max\{a^\top\Sigma a : \|a\|=1,\ a\perp u_1,\dots,u_{k-1}\} = \lambda_k$, attained at $u_k$. Moreover, among all $k$-dimensional subspaces $V \subseteq \mathbb{R}^d$, the total variance captured, $\operatorname{tr}(\Pi_V \Sigma \Pi_V)$ with $\Pi_V$ the orthogonal projector, is maximized by $V = \operatorname{span}(u_1,\dots,u_k)$, with maximum $\lambda_1+\dots+\lambda_k$.

*Proof.* Expand $a$ in the eigenbasis: $a = \sum_i c_iu_i$ with $\sum_i c_i^2 = \|a\|^2 = 1$. Then $a^\top\Sigma a = \sum_i \lambda_i c_i^2$, a convex combination of the $\lambda_i$, hence at most $\max_i\lambda_i = \lambda_1$, with equality iff all the weight sits on eigenvectors achieving $\lambda_1$ (in particular $a = u_1$ works). Restricting to $a \perp u_1,\dots,u_{k-1}$ forces $c_1=\dots=c_{k-1}=0$, so the combination is over $\lambda_k,\dots,\lambda_d$ and the same argument gives $\lambda_k$.

For the subspace statement, write $\Pi_V = \sum_{j=1}^k v_jv_j^\top$ for an orthonormal basis $v_1,\dots,v_k$ of $V$. Then $\operatorname{tr}(\Pi_V\Sigma\Pi_V) = \sum_j v_j^\top \Sigma v_j = \sum_j\sum_i \lambda_i (u_i^\top v_j)^2 = \sum_i \lambda_i w_i$ where $w_i := \sum_j (u_i^\top v_j)^2 = \|\Pi_V u_i\|^2 \in [0,1]$ and $\sum_i w_i = \operatorname{tr}(\Pi_V) = k$. Maximizing $\sum_i\lambda_iw_i$ subject to $0\le w_i\le 1$, $\sum_i w_i = k$ is a linear program whose solution puts $w_i=1$ on the $k$ largest $\lambda_i$; that assignment is realized by $V = \operatorname{span}(u_1,\dots,u_k)$. $\square$

That is PCA. Note what the theorem is and is not. It *is* a statement about a population covariance matrix. It is *not* a statement about data until you replace $\Sigma$ by the sample covariance $\hat\Sigma_n = \frac{1}{n}\sum_i (x_i-\bar x)(x_i-\bar x)^\top$, at which point everything above still holds verbatim (because $\hat\Sigma_n$ is itself a covariance matrix — of the empirical distribution $\hat P_n$ from Chapter 1) but the *interpretation* changes: you are now maximizing variance of a finite sample, and whether that generalizes is a separate question governed by the concentration results of Chapter 4.

> **Worked Example 5.5 (a three-feature PCA, fully computed).**
> $$\Sigma = \begin{pmatrix} 2.0 & 0.8 & 0.4\\ 0.8 & 1.0 & 0.3 \\ 0.4 & 0.3 & 0.5\end{pmatrix}.$$
> Eigenvalues (computed): $\lambda = (2.562473,\ 0.578888,\ 0.358639)$, summing to $3.5 = \operatorname{tr}\Sigma$ as they must (the trace is the total variance $\sum_i\operatorname{Var}(X_i)$, which is basis-independent). Top eigenvector $u_1 = (0.8465, 0.4783, 0.2337)^\top$ (sign chosen positive).
>
> Variance ratios: $\lambda_i/\operatorname{tr}\Sigma = (0.7321,\ 0.1654,\ 0.1025)$; cumulative $(0.7321,\ 0.8975,\ 1.0000)$. So one component captures 73.2% of total variance, two capture 89.8%.
>
> Reading $u_1$: the loading on feature 1 is largest, so the dominant direction of variation is mostly "feature 1 moves, and features 2 and 3 move along with it in the same direction." Note also that $u_1$ is *not* $(1,0,0)$ even though feature 1 has the largest individual variance — correlation drags the principal direction toward the other features. And the "73%" figure is entirely scale-dependent: multiply feature 1 by 10 and it will capture 99%. This is why PCA on unstandardized features with mixed units is almost always a mistake, and why practitioners usually run it on the correlation matrix instead.

### 5.2.3 Mahalanobis distance and the ellipsoid picture

**Definition 5.12 (Mahalanobis distance).** For $\Sigma\succ0$, the Mahalanobis distance from $x$ to $\mu$ is
$$
d_\Sigma(x,\mu) := \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)} = \|\Sigma^{-1/2}(x-\mu)\|_2 .
$$
Equivalently: whiten, then measure ordinary Euclidean distance. The level sets $\{x : d_\Sigma(x,\mu) = r\}$ are ellipsoids centered at $\mu$.

**Proposition 5.13 (ellipsoid geometry).** For $\Sigma = UDU^\top \succ 0$, the set $E_r := \{x : (x-\mu)^\top\Sigma^{-1}(x-\mu)\le r^2\}$ is an ellipsoid centered at $\mu$ whose principal axes point along the eigenvectors $u_i$ with semi-axis lengths $r\sqrt{\lambda_i}$, and whose volume is
$$
\lambda_d(E_r) = V_d\, r^d \sqrt{\det\Sigma},\qquad V_d := \lambda_d(\text{unit ball in }\mathbb{R}^d) = \frac{\pi^{d/2}}{\Gamma(d/2+1)} .
$$

*Proof.* Substituting $y = \Sigma^{-1/2}(x-\mu)$ turns $E_r$ into the ball $\{\|y\|\le r\}$. The map $x = \mu + \Sigma^{1/2}y$ is affine with Jacobian determinant $\det\Sigma^{1/2} = \sqrt{\det\Sigma}$, so by the change-of-variables formula (Ch. 3) the volume scales by that factor: $\lambda_d(E_r) = \sqrt{\det\Sigma}\cdot\lambda_d(\{\|y\|\le r\}) = \sqrt{\det\Sigma}\,V_dr^d$. For the axes, note $x - \mu = \Sigma^{1/2}y = UD^{1/2}U^\top y$; taking $y = r\,U^\top$-image of a standard basis direction shows the extreme points along $u_i$ occur at distance $r\sqrt{\lambda_i}$. $\square$

So $\sqrt{\det\Sigma}$ is a *generalized standard deviation*: it measures the volume of the uncertainty region. This is why $\log\det\Sigma$ appears in every Gaussian log-likelihood (§5.3) and why $-\frac12\log\det$ terms show up as "volume penalties" in variational objectives.

That last remark can be made exact, and it is worth doing now because it explains where those penalties come from. The statement anticipates §5.3, which defines $\mathcal N(\mu,\Sigma)$ properly and derives its density; nothing in §5.3 uses this proposition, so there is no circularity, and a reader working strictly in order may skip the next page and return after Theorem 5.17.

**Proposition 5.14 (differential entropy of a Gaussian).** If $X\sim\mathcal N(\mu,\Sigma)$ with $\Sigma\succ0$ and $p$ is its Lebesgue density, then
$$
h(X) := -\int_{\mathbb{R}^d} p(x)\log p(x)\,dx = \tfrac12\log\big((2\pi e)^d\det\Sigma\big) = \tfrac d2\big(1+\log 2\pi\big) + \tfrac12\log\det\Sigma .
$$
Moreover, among all distributions on $\mathbb{R}^d$ with covariance $\Sigma$, the Gaussian is the unique maximizer of $h$.

*Proof of the formula.* Using the density derived in §5.3 (Theorem 5.17(c)),
$$
-\log p(X) = \tfrac12\log\big((2\pi)^d\det\Sigma\big) + \tfrac12 (X-\mu)^\top\Sigma^{-1}(X-\mu).
$$
Take expectations. The quadratic term is a scalar, so it equals its own trace, and by linearity and cyclicity of the trace ($\operatorname{tr}(AB)=\operatorname{tr}(BA)$, which lets the expectation slide inside)
$$
\mathbb{E}\big[(X-\mu)^\top\Sigma^{-1}(X-\mu)\big] = \mathbb{E}\operatorname{tr}\big(\Sigma^{-1}(X-\mu)(X-\mu)^\top\big) = \operatorname{tr}(\Sigma^{-1}\Sigma) = d .
$$
Hence $h(X) = \frac12\log((2\pi)^d\det\Sigma) + \frac d2 = \frac12\log((2\pi e)^d\det\Sigma)$. The maximum-entropy claim is standard and is recorded in §5.9. $\square$

Entropy is $\log$-volume up to an additive constant, exactly as Proposition 5.13 suggests: doubling every semi-axis multiplies the volume by $2^d$ and adds $d\log 2$ to $h$. For the $3\times3$ covariance of §5.1.4, $\det\Sigma = 0.894$, giving $h = 4.2008$ nats against $4.2568$ for $\mathcal N(0,I_3)$ — slightly less spread out, because $\det\Sigma<1$. Note that $h$ can be *negative* (take $\Sigma = 0.01\,I_2$), which is the entropy analogue of densities exceeding $1$; only differences of differential entropies, such as a KL divergence, are base-measure-free.

The maximum-entropy property is the honest justification for the Gaussian's ubiquity in modeling — more honest, in most cases, than an appeal to the central limit theorem. If all you are willing to commit to is a mean and a covariance, the Gaussian is the distribution that commits to nothing further.

<figure>
<svg viewBox="0 0 760 350" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three covariance ellipses with eigenvector axes">
  <g fill="none" stroke="currentColor" stroke-width="1" opacity="0.35">
    <line x1="60" y1="170" x2="200" y2="170"/><line x1="130" y1="100" x2="130" y2="240"/>
    <line x1="300" y1="170" x2="480" y2="170"/><line x1="390" y1="105" x2="390" y2="235"/>
    <line x1="560" y1="170" x2="720" y2="170"/><line x1="640" y1="100" x2="640" y2="240"/>
  </g>
  <polyline points="161.8,170.0 161.7,166.8 161.2,163.7 160.4,160.6 159.3,157.6 157.9,154.8 156.3,152.1 154.4,149.5 152.2,147.2 149.8,145.1 147.3,143.3 144.5,141.7 141.6,140.4 138.6,139.4 135.5,138.7 132.4,138.3 129.2,138.2 126.0,138.4 122.9,139.0 119.9,139.8 116.9,141.0 114.1,142.4 111.4,144.2 108.9,146.1 106.7,148.4 104.6,150.8 102.8,153.4 101.3,156.2 100.1,159.1 99.2,162.1 98.5,165.3 98.2,168.4 98.2,171.6 98.5,174.7 99.2,177.9 100.1,180.9 101.3,183.8 102.8,186.6 104.6,189.2 106.7,191.6 108.9,193.9 111.4,195.8 114.1,197.6 116.9,199.0 119.9,200.2 122.9,201.0 126.0,201.6 129.2,201.8 132.4,201.7 135.5,201.3 138.6,200.6 141.6,199.6 144.5,198.3 147.3,196.7 149.8,194.9 152.2,192.8 154.4,190.5 156.3,187.9 157.9,185.2 159.3,182.4 160.4,179.4 161.2,176.3 161.7,173.2 161.8,170.0" fill="none" stroke="#3b82f6" stroke-width="2.2"/>
  <polyline points="390.0,147.5 402.6,147.9 414.7,149.3 425.9,151.4 435.6,154.3 443.5,157.8 449.2,161.8 452.7,166.1 453.6,170.6 452.0,175.0 448.0,179.3 441.7,183.1 433.3,186.5 423.2,189.2 411.8,191.1 399.5,192.2 386.8,192.5 374.3,191.8 362.4,190.3 351.6,187.9 342.3,184.9 334.9,181.3 329.7,177.2 326.9,172.8 326.5,168.3 328.7,163.9 333.4,159.7 340.2,156.0 349.1,152.8 359.6,150.2 371.2,148.5 383.7,147.6 390.0,147.5" fill="none" stroke="#10b981" stroke-width="2.2"/>
  <polyline points="663.5,200.1 651.3,208.6 638.7,215.7 626.1,220.9 614.1,224.1 603.1,225.1 593.5,224.0 585.8,220.7 580.3,215.4 577.1,208.3 576.4,199.7 578.3,189.9 582.6,179.4 589.1,168.4 597.7,157.5 608.0,147.1 619.5,137.7 631.8,129.5 644.5,122.9 657.0,118.1 668.8,115.5 679.5,115.0 688.6,116.6 695.8,120.4 700.7,126.2 703.3,133.7 703.6,138.0 702.4,147.5 698.7,157.9 692.7,168.8 684.6,179.8 674.7,190.3 663.5,200.1" fill="none" stroke="#f59e0b" stroke-width="2.2"/>
  <g stroke="currentColor" stroke-width="2">
    <line x1="130" y1="170" x2="143" y2="170"/><line x1="130" y1="170" x2="130" y2="157"/>
    <line x1="390" y1="170" x2="416" y2="170"/><line x1="390" y1="170" x2="390" y2="161"/>
    <line x1="640" y1="170" x2="664.2" y2="151.1"/><line x1="640" y1="170" x2="630.4" y2="182.3"/>
  </g>
  <g fill="currentColor" font-size="13" text-anchor="middle" font-family="sans-serif">
    <text x="130" y="255">Σ = I</text>
    <text x="390" y="255">Σ = diag(4, 0.5)</text>
    <text x="640" y="255">Σ = [[4,2],[2,3]]</text>
    <text x="130" y="273" font-size="11">λ = (1, 1); circle</text>
    <text x="390" y="273" font-size="11">λ = (4, 0.5); axis-aligned</text>
    <text x="640" y="273" font-size="11">λ = (5.56, 1.44); tilted 37.98°</text>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="20" y="305">Figure 5.1 — 95% Mahalanobis ellipses {x : xᵀΣ⁻¹x ≤ 5.9915} for three covariance matrices, drawn to a common scale.</text>
    <text x="20" y="323">Short black segments are the eigenvectors uᵢ scaled by √λᵢ. Axis half-lengths are √(5.9915·λᵢ): correlation tilts</text>
    <text x="20" y="341">the ellipse off the coordinate axes, and unequal eigenvalues stretch it. Volume ∝ √det Σ = 1.00, 1.41, 2.83.</text>
  </g>
</svg>
</figure>

The constant $5.9915$ is $\chi^2_2$'s $95$th percentile, computed as `scipy.stats.chi2.ppf(0.95, 2) = 5.991465`. The reason a chi-square quantile appears will be clear after §5.3: if $X\sim\mathcal N(\mu,\Sigma)$ with $\Sigma\succ0$ then $d_\Sigma(X,\mu)^2 = \|\Sigma^{-1/2}(X-\mu)\|^2 \sim \chi^2_d$, being the squared norm of a standard Gaussian vector. Figure 5.1 is drawn for general $\Sigma$, but the *probability* interpretation of the level $5.9915$ is Gaussian-specific.

> **Worked Example 5.6 (which point is the outlier?).**
> Take $\Sigma = \begin{pmatrix}4&2\\2&3\end{pmatrix}$ (as in Worked Example 5.4), $\mu = 0$, and two candidate points $p_1 = (3,0)^\top$ and $p_2 = (0,2)^\top$. Euclidean distances: $\|p_1\| = 3 > \|p_2\| = 2$, so naive distance says $p_1$ is more anomalous.
>
> Mahalanobis: $\Sigma^{-1} = \frac{1}{8}\begin{pmatrix}3&-2\\-2&4\end{pmatrix} = \begin{pmatrix}0.375&-0.25\\-0.25&0.5\end{pmatrix}$. Then
> $$d^2_\Sigma(p_1) = 9\cdot 0.375 = 3.375,\qquad d^2_\Sigma(p_2) = 4\cdot 0.5 = 2.0 .$$
> So $d_\Sigma(p_1) = 1.8371$, $d_\Sigma(p_2) = 1.4142$: $p_1$ is still the more anomalous, but the gap has narrowed from a ratio of $1.50$ to a ratio of $1.30$ — the cloud spreads more in the $x$-direction ($\Sigma_{11}=4$) than in $y$ ($\Sigma_{22}=3$), so a displacement in $x$ is less surprising.
>
> Now change the comparison to $q_1 = (3,3)^\top$ and $q_2 = (3,-1)^\top$. Euclidean says $q_1$ is the outlier: $\|q_1\| = 4.2426 > \|q_2\| = 3.1623$. Mahalanobis reverses the verdict:
> $$d^2_\Sigma(q_1) = 0.375(9) - 2(0.25)(9) + 0.5(9) = 3.375,\qquad d^2_\Sigma(q_2) = 0.375(9) - 2(0.25)(-3) + 0.5(1) = 5.375 .$$
> So $d_\Sigma(q_1) = 1.8371 < d_\Sigma(q_2) = 2.3184$. The point $(3,3)$ moves *with* the positive correlation and is unremarkable; the point $(3,-1)$ moves *against* it and is the genuine anomaly, despite being closer to the origin. Figure 5.1's tilted amber ellipse is that effect drawn.
>
> This is the whole idea behind Mahalanobis-based out-of-distribution detection for neural networks: fit a Gaussian to penultimate-layer features per class, score a test point by $\min_c d_{\Sigma_c}(z, \mu_c)$. It works better than Euclidean distance precisely because learned feature dimensions are correlated and unequally scaled.

> **Common Confusion 5.2.**
> *"I whitened my representation, so the dimensions are now independent — I've disentangled the factors of variation."*
>
> Whitening enforces $\operatorname{Cov}(\tilde X) = I$. That is a statement about **second moments only**. Worked Example 5.3 gave a distribution on the unit circle whose covariance is $\frac12 I$; rescale by $\sqrt2$ and its covariance is exactly $I$, yet the coordinates satisfy $\tilde X_1^2 + \tilde X_2^2 = 2$ almost surely. Perfectly white, perfectly dependent.
>
> Worse, Proposition 5.10 says whitening is only defined up to an arbitrary orthogonal $Q$. So even if whitening *did* produce independence, it would not produce a canonical set of axes — you could rotate and still be white. Any claim that a method "disentangles" must therefore rest on something beyond second-order statistics: a supervision signal, an architectural asymmetry, a non-Gaussian objective (this is exactly why ICA maximizes non-Gaussianity — it uses fourth moments to break the rotational tie that second moments cannot). Decorrelation is a preprocessing convenience. It is not an identification result, and papers that treat it as one are overclaiming.

> **Self-check 5.2.**
> 1. If $\Sigma$ has eigenvalues $(9,4,1)$, what is $\max_{\|a\|=1}\operatorname{Var}(a^\top X)$, and what fraction of total variance do the top two components capture?
> 2. Why is $\sqrt{\det\Sigma}$, not $\det\Sigma$, the natural "volume scale"?
> 3. Two whitening matrices $W_1, W_2$ for the same $\Sigma$. What is $W_1W_2^{-1}$?
> 4. Give a one-line reason why PCA applied to features measured in metres versus millimetres gives different answers.

---

## 5.3 The multivariate Gaussian: three definitions, one of which is right

### 5.3.1 Why not start with the density?

Most courses define the multivariate Gaussian by writing down
$$
p(x) = \frac{1}{(2\pi)^{d/2}(\det\Sigma)^{1/2}}\exp\Big(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\Big)
$$
and moving on. That formula is correct, useful, and *not a definition*, because it requires $\Sigma \succ 0$. The moment $\Sigma$ is singular the formula is meaningless — $\det\Sigma = 0$, $\Sigma^{-1}$ does not exist — and yet there is obviously still a sensible distribution there. If $Z\sim\mathcal N(0,1)$ and I ask about the law of $(Z,Z)^\top$ in $\mathbb{R}^2$, nothing has gone wrong with the world; a perfectly definite random vector is sitting on the diagonal line. It just has no Lebesgue density (Proposition 5.2(3)).

This is not a corner case in ML. It is the *typical* case. Every latent-variable generative model with a linear decoder produces exactly this situation: $X = Az + b$ with $z \sim \mathcal N(0,I_k)$ and $A \in \mathbb{R}^{d\times k}$, $k \ll d$. That is a Gaussian on a $k$-dimensional affine subspace of $\mathbb{R}^d$. Probabilistic PCA in the zero-noise limit, the prior predictive of a linear-Gaussian model with a low-rank loading matrix, and the pushforward of a latent through a linearized decoder are all singular Gaussians. If your definition cannot name them, your definition is wrong.

So we define the Gaussian *constructively*: it is whatever you get by pushing standard normal noise through an affine map. Definition-as-recipe. Then we prove that when $\Sigma\succ0$ this recipe reproduces the familiar density.

**Definition 5.15 (multivariate Gaussian).**
A random vector $X$ in $\mathbb{R}^d$ is **Gaussian** (or *normal*), written $X\sim\mathcal N(\mu,\Sigma)$, if there exist $\mu\in\mathbb{R}^d$, a matrix $A\in\mathbb{R}^{d\times k}$ for some $k\ge0$, and $Z\sim\mathcal N(0,I_k)$ (a vector of $k$ i.i.d. standard normal scalars) such that
$$
X \stackrel{d}{=} \mu + AZ .
$$
By Proposition 5.3 this forces $\mathbb{E}X = \mu$ and $\operatorname{Cov}(X) = AA^\top =: \Sigma$. The case $k=0$ (no noise) gives the degenerate point mass $\delta_\mu$, which we accept as the Gaussian $\mathcal N(\mu,0)$.

Two immediate sanity checks. Every $\Sigma\succeq0$ *is* realizable, because $\Sigma = \Sigma^{1/2}(\Sigma^{1/2})^\top$; so for each $(\mu,\Sigma)$ with $\Sigma\succeq0$ at least one Gaussian exists. And the law depends only on $(\mu,\Sigma)$, not on the particular factor $A$ chosen — this needs proof, and it is what Lemma 5.16 delivers.

### 5.3.2 The characteristic function, and why it settles everything

The tool that makes everything in this section a five-line proof is the characteristic function, and since not every first probability course covers it — and this tutorial has so far integrated only real-valued functions — let us build it honestly before using it. It takes half a page.

> **A primer on characteristic functions (everything this chapter needs).**
> For a complex-valued function $W = U + iV$ of a random outcome, with $U, V$ real random variables, *define* $\mathbb{E}[W] := \mathbb{E}[U] + i\,\mathbb{E}[V]$, whenever both real expectations exist. Nothing new is being integrated; a complex expectation is two real expectations in a trench coat, and every property we need reduces to Chapter 2 facts applied to the real and imaginary parts separately:
>
> 1. **Existence is automatic.** The **characteristic function** of a random vector $X$ in $\mathbb{R}^d$ is $\varphi_X(t) := \mathbb{E}[e^{it^\top X}] = \mathbb{E}[\cos(t^\top X)] + i\,\mathbb{E}[\sin(t^\top X)]$ for $t \in \mathbb{R}^d$. Both $\cos(t^\top X)$ and $\sin(t^\top X)$ are bounded measurable functions (compositions of continuous functions with a random vector, Proposition 1.15), so $\varphi_X(t)$ exists for *every* $t$ and *every* distribution — no moment conditions, unlike the moment generating function of Chapter 4, which can be infinite. That is the whole reason probabilists prefer $e^{itX}$ to $e^{tX}$.
> 2. **Linearity** holds because it holds for the real and imaginary parts separately.
> 3. **The modulus inequality** $|\mathbb{E}[W]| \le \mathbb{E}|W|$ holds: write $\mathbb{E}[W] = re^{i\theta}$ in polar form; then $|\mathbb{E}[W]| = r = \mathbb{E}[e^{-i\theta}W] = \mathbb{E}[\operatorname{Re}(e^{-i\theta}W)] \le \mathbb{E}|W|$, where the middle equality holds because $r$ is real so the imaginary part of the expectation must vanish, and the inequality is monotonicity of the (real) expectation applied to $\operatorname{Re}(z) \le |z|$. In particular $|\varphi_X(t)| \le \mathbb{E}|e^{it^\top X}| = 1$: characteristic functions never exceed $1$ in modulus.
> 4. **Independence factorizes:** if $X \perp Y$ then $\mathbb{E}[e^{it^\top X}e^{is^\top Y}] = \varphi_X(t)\varphi_Y(s)$ — expand both exponentials into cosine and sine, multiply out the four real products, and apply Fubini (Ch. 2) to each; every factor is bounded, so integrability is free.
> 5. **The one Gaussian integral.** For scalar $Z\sim\mathcal N(0,1)$: $\varphi_Z(t) = e^{-t^2/2}$. Sketch of the standard proof, so nothing is on faith: $\varphi_Z(t) = \int \cos(tz)\,\phi(z)\,dz$ (the sine integral vanishes by symmetry); differentiating under the integral in $t$ — legal by dominated convergence (Ch. 2), with dominating function $|z|\phi(z)$ — and integrating by parts gives $\varphi_Z'(t) = -t\,\varphi_Z(t)$, a first-order ODE with $\varphi_Z(0) = 1$, whose unique solution is $e^{-t^2/2}$ (uniqueness is elementary: $\frac{d}{dt}\big[\varphi_Z(t)e^{t^2/2}\big] = 0$, so the bracket is the constant $\varphi_Z(0) = 1$). Compare Chapter 4's real analogue $\mathbb{E}[e^{sZ}] = e^{s^2/2}$: same shape, opposite sign in the exponent, and the $i$ is what flips it.
> 6. **The deep fact, cited not proved:** two random vectors with the same characteristic function have the same law (uniqueness theorem, §5.9). This is the only ingredient we take from outside; everything else above you have now verified.

With the toolkit assembled, here is the computation that settles which parameters determine a Gaussian.

**Lemma 5.16 (characteristic function of a Gaussian).**
If $X\sim\mathcal N(\mu,\Sigma)$ in the sense of Definition 5.15, then for all $t\in\mathbb{R}^d$
$$
\varphi_X(t) := \mathbb{E}\big[e^{i t^\top X}\big] = \exp\Big(i t^\top\mu - \tfrac12 t^\top \Sigma t\Big).
$$
Consequently the law of $X$ depends only on $(\mu,\Sigma)$, so the notation $\mathcal N(\mu,\Sigma)$ is well defined, and any two factors $A, B$ with $AA^\top = BB^\top$ give the same law.

*Proof.* Write $X = \mu + AZ$ with $Z = (Z_1,\dots,Z_k)$ i.i.d. $\mathcal N(0,1)$. Put $s := A^\top t\in\mathbb{R}^k$. Then $t^\top X = t^\top\mu + s^\top Z = t^\top\mu + \sum_{j} s_jZ_j$, so
$$
\mathbb{E}[e^{it^\top X}] = e^{it^\top\mu}\,\mathbb{E}\Big[\prod_{j=1}^k e^{is_jZ_j}\Big] = e^{it^\top\mu}\prod_{j=1}^k \mathbb{E}[e^{is_jZ_j}] = e^{it^\top\mu}\prod_{j=1}^k e^{-s_j^2/2} = e^{it^\top\mu}e^{-\|s\|^2/2},
$$
where the product came out of the expectation by independence — item 4 of the primer above (Definition 1.23 plus Fubini (Ch. 2), applied separately to real and imaginary parts; everything is bounded by $1$ so integrability is free). Finally $\|s\|^2 = s^\top s = t^\top AA^\top t = t^\top\Sigma t$. Since $\varphi_X$ depends on $A$ only through $AA^\top = \Sigma$, and characteristic functions determine laws, the law depends only on $(\mu,\Sigma)$. $\square$

Now the main structural theorem.

**Theorem 5.17 (three equivalent definitions).**
Let $X$ be a random vector in $\mathbb{R}^d$ with $\mathbb{E}\|X\|^2<\infty$, mean $\mu$, covariance $\Sigma$. The following are equivalent.

**(a)** $X \stackrel{d}{=} \mu + AZ$ for some $A$ and $Z\sim\mathcal N(0,I_k)$ (Definition 5.15).
**(b)** For every $a\in\mathbb{R}^d$, the scalar $a^\top X$ has a univariate normal distribution $\mathcal N(a^\top\mu,\, a^\top\Sigma a)$, where the degenerate case $a^\top\Sigma a = 0$ is read as the point mass at $a^\top\mu$.
**(c)** *(only when $\Sigma\succ0$)* $P_X \ll \lambda_d$ with density
$$
p(x) = (2\pi)^{-d/2}(\det\Sigma)^{-1/2}\exp\big(-\tfrac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\big).
$$

*Proof.*

**(a) $\Rightarrow$ (b).** Given $X = \mu+AZ$, we have $a^\top X = a^\top\mu + (A^\top a)^\top Z = a^\top\mu + \sum_j s_jZ_j$ with $s = A^\top a$. A finite linear combination of independent standard normals is normal (this is the scalar case of Lemma 5.16, or the classical convolution identity), with mean $0$ and variance $\|s\|^2 = a^\top\Sigma a$. So $a^\top X\sim\mathcal N(a^\top\mu, a^\top\Sigma a)$.

**(b) $\Rightarrow$ (a).** Assume every projection is normal. Compute the characteristic function of $X$ directly: for $t\in\mathbb{R}^d$, $\varphi_X(t) = \mathbb{E}[e^{it^\top X}]$ is the characteristic function of the scalar random variable $t^\top X$, evaluated at $1$. By hypothesis $t^\top X\sim\mathcal N(t^\top\mu,\,t^\top\Sigma t)$, whose characteristic function at $1$ is $\exp(it^\top\mu - \frac12 t^\top\Sigma t)$. So $\varphi_X(t) = \exp(it^\top\mu-\frac12t^\top\Sigma t)$, which by Lemma 5.16 is the characteristic function of $\mu + \Sigma^{1/2}Z$ with $Z\sim\mathcal N(0,I_d)$. Uniqueness of characteristic functions gives $X\stackrel{d}{=}\mu+\Sigma^{1/2}Z$, which is form (a). (This step is the **Cramér–Wold device**: the collection of one-dimensional projections determines the joint law.)

**(a) $\Rightarrow$ (c) when $\Sigma\succ0$.** Take $A = \Sigma^{1/2}$, which is now invertible, and $k=d$. The density of $Z\sim\mathcal N(0,I_d)$ is $(2\pi)^{-d/2}e^{-\|z\|^2/2}$, a product over coordinates by Fubini (Ch. 2). The map $T(z) = \mu+\Sigma^{1/2}z$ is an affine bijection with $T^{-1}(x) = \Sigma^{-1/2}(x-\mu)$ and constant Jacobian determinant $(\det\Sigma)^{-1/2}$. The change-of-variables formula (Ch. 3) says the pushforward $T_\#P_Z$ has density
$$
p(x) = p_Z(T^{-1}(x))\,\big|\det DT^{-1}(x)\big| = (2\pi)^{-d/2}e^{-\|\Sigma^{-1/2}(x-\mu)\|^2/2}(\det\Sigma)^{-1/2},
$$
and $\|\Sigma^{-1/2}(x-\mu)\|^2 = (x-\mu)^\top\Sigma^{-1}(x-\mu)$ since $\Sigma^{-1/2}$ is symmetric. This is (c).

**(c) $\Rightarrow$ (a).** Reverse the change of variables: if $X$ has that density, then $Z := \Sigma^{-1/2}(X-\mu)$ has density $(2\pi)^{-d/2}e^{-\|z\|^2/2}$, which factorizes into $d$ standard normal densities, so by the product-density criterion for independence (Ch. 2 / Definition 1.23) the coordinates of $Z$ are i.i.d. $\mathcal N(0,1)$, and $X = \mu+\Sigma^{1/2}Z$. $\square$

Let me say plainly why **(a) is the right definition**. (c) breaks when $\Sigma$ is singular — it cannot even be stated. (b) is beautiful and is the workhorse for proofs (it turns $d$-dimensional questions into $1$-dimensional ones), but it is not constructive: it tells you how to *check* Gaussianity, not how to *build* a Gaussian, and it presupposes you already know what a univariate normal is. (a) is constructive, handles singular $\Sigma$ without a murmur, is exactly how you sample on a computer (draw $Z$, multiply by the Cholesky factor, add $\mu$), and is exactly the "pushforward of a simple latent" picture that Chapter 1 established as the organizing idea for generative models. A Gaussian is $\mathcal N(0,I_k)$ viewed through an affine lens.

One technical remark, so you know which hypotheses are load-bearing. The theorem assumed $\mathbb{E}\|X\|^2 < \infty$ up front, which lets us speak of $\mu$ and $\Sigma$ throughout, but for (b) that assumption is a convenience, not a necessity: if every projection $a^\top X$ is normal, then in particular every coordinate $e_i^\top X = X_i$ is normal and hence has finite variance, so $\mathbb{E}\|X\|^2 = \sum_i \mathbb{E}[X_i^2] < \infty$ comes *for free*. Definition (b) is therefore self-sufficient as a definition; we assumed square-integrability only to state all three conditions on a common footing.

**Corollary 5.18 (Gaussians are closed under affine maps).** If $X\sim\mathcal N(\mu,\Sigma)$ in $\mathbb{R}^d$, $B\in\mathbb{R}^{m\times d}$, $c\in\mathbb{R}^m$, then $BX+c \sim \mathcal N(B\mu+c,\ B\Sigma B^\top)$.

*Proof.* Write $X\stackrel d= \mu+AZ$; then $BX+c \stackrel d= (B\mu+c) + (BA)Z$, which is form (a) with mean $B\mu+c$ and covariance $BA(BA)^\top = B\Sigma B^\top$. $\square$

This one-line corollary is doing an enormous amount of work in the rest of the chapter. Marginals (§5.4) are the case where $B$ selects coordinates. Sums of independent Gaussians (§5.5) are the case where $B = (I\ \ I)$ applied to a stacked vector. The forward pass of a linear layer (§5.7) is the case where $B$ is a weight matrix.

> **Worked Example 5.7 (a singular Gaussian, and what "no density" looks like).**
> Let $Z\sim\mathcal N(0,1)$ and $X := (Z, Z)^\top = AZ$ with $A = (1,1)^\top \in\mathbb{R}^{2\times1}$. By Definition 5.15, $X$ is Gaussian with $\mu=0$ and
> $$\Sigma = AA^\top = \begin{pmatrix}1&1\\1&1\end{pmatrix},\qquad \operatorname{rank}\Sigma = 1,\quad \det\Sigma = 0 .$$
> Definition (c) is unusable. Definition (b) works fine: for any $a = (a_1,a_2)$, $a^\top X = (a_1+a_2)Z\sim\mathcal N(0,(a_1+a_2)^2)$, and indeed $a^\top\Sigma a = (a_1+a_2)^2$. Note the direction $a = (1,-1)$ gives variance $0$, and correspondingly $X_1 - X_2 = 0$ a.s. — Proposition 5.2(3) in action, with the hyperplane $H = \{x_1 = x_2\}$.
>
> $P_X$ assigns probability $1$ to the line $H$, and $\lambda_2(H) = 0$. So $P_X$ is a Gaussian with no Lebesgue density. It does, however, have a perfectly good density with respect to *one-dimensional Hausdorff measure on the line $H$*, or equivalently it is the pushforward of the standard $\mathcal N(0,1)$ on $\mathbb{R}$ under $z\mapsto(z,z)$. The lesson from Chapter 2 recurs: "has a density" is always "has a density with respect to a stated base measure," never an absolute property.
>
> **ML translation.** A VAE encoder outputs $q_\phi(z\mid x) = \mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi(x)^2))$. If $\sigma_\phi(x)\to0$ for some coordinate — "posterior collapse to a point" — that coordinate's Gaussian becomes singular, the density blows up to $+\infty$ pointwise, and the ELBO's $\log q_\phi$ term diverges to $+\infty$ while the KL term diverges too. Implementations that parameterize $\log\sigma$ and clamp it below (e.g. at $-7$) are not being superstitious; they are enforcing $\Sigma\succ0$ so that definition (c) remains applicable.

> **Common Confusion 5.3.**
> *"$\mathcal N(\mu,\Sigma)$ needs $\Sigma$ invertible, otherwise the distribution doesn't exist."*
>
> It exists; it just has no Lebesgue density. Definition 5.15 constructs it for any $\Sigma\succeq0$: factor $\Sigma = AA^\top$ (always possible via $\Sigma^{1/2}$), draw $Z$, output $\mu+AZ$. What fails when $\Sigma$ is singular is not existence but a *representation*. And what fails downstream is anything that needs $\Sigma^{-1}$ or $\log\det\Sigma$: KL divergences between Gaussians, log-likelihood evaluation, Mahalanobis distances. This is precisely why numerical code adds jitter — `Sigma + 1e-6*I` — before a Cholesky factorization. The jitter is not a hack to fix "numerical error"; it is a deliberate replacement of a singular Gaussian by a nearby nonsingular one so that density-based formulas become legal. If you find yourself needing jitter of order $10^{-2}$, your model is telling you something real about rank deficiency, not complaining about floating point.

Figure 5.2 draws both halves of that story. On the left, the law of $(Z,Z)$ puts all its mass on the line $x_2 = x_1$, a set of Lebesgue measure zero in $\mathbb{R}^2$; there is no function $p$ with $P(A) = \int_Ap\,d\lambda_2$, because every such integral over the line would be $0$. On the right, adding $\varepsilon I$ to $\Sigma$ replaces the line by a very thin ellipse: a density now exists, but it is concentrated in a sliver of area proportional to $\sqrt{\varepsilon}$, and its peak value $\big(2\pi\sqrt{\det(\Sigma+\varepsilon I)}\big)^{-1}$ grows without bound as $\varepsilon\to0$ — numerically $0.347$, $1.123$, $3.558$ and $112.5$ for $\varepsilon = 10^{-1}, 10^{-2}, 10^{-3}, 10^{-6}$. Nothing pathological is happening to the *probabilities*; it is the density, a ratio against a base measure that is about to become inappropriate, that misbehaves.

<figure>
<svg viewBox="0 0 720 380" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A singular Gaussian supported on a line, and its regularized version">
  <g fill="none" stroke="currentColor" stroke-width="1" opacity="0.35">
    <line x1="60" y1="170" x2="285" y2="170"/><line x1="170" y1="62" x2="170" y2="278"/>
    <line x1="400" y1="170" x2="655" y2="170"/><line x1="520" y1="62" x2="520" y2="278"/>
  </g>
  <g fill="currentColor" font-size="11" opacity="0.7" font-family="sans-serif">
    <text x="289" y="174">x₁</text><text x="163" y="56">x₂</text>
    <text x="659" y="174">x₁</text><text x="513" y="56">x₂</text>
  </g>
  <line x1="67.4" y1="272.6" x2="272.6" y2="67.4" stroke="#3b82f6" stroke-width="3"/>
  <g fill="#3b82f6">
    <circle cx="92.8" cy="247.2" r="3.4"/><circle cx="107.1" cy="232.9" r="3.4"/><circle cx="137.2" cy="202.8" r="3.4"/>
    <circle cx="139.5" cy="200.5" r="3.4"/><circle cx="152.8" cy="187.2" r="3.4"/><circle cx="155.0" cy="185.0" r="3.4"/>
    <circle cx="168.2" cy="171.8" r="3.4"/><circle cx="179.2" cy="160.8" r="3.4"/><circle cx="179.5" cy="160.5" r="3.4"/>
    <circle cx="186.4" cy="153.6" r="3.4"/><circle cx="194.9" cy="145.1" r="3.4"/><circle cx="213.5" cy="126.5" r="3.4"/>
    <circle cx="223.6" cy="116.4" r="3.4"/><circle cx="265.8" cy="74.2" r="3.4"/>
  </g>
  <circle cx="215" cy="215" r="26" fill="none" stroke="#ef4444" stroke-width="1.8" stroke-dasharray="4 3"/>
  <text x="215" y="219" fill="#ef4444" font-size="11" text-anchor="middle" font-family="sans-serif">P = 0</text>
  <ellipse cx="520" cy="170" rx="134.8" ry="29.4" transform="rotate(-45 520 170)" fill="none" stroke="currentColor" stroke-width="1.6" opacity="0.45"/>
  <ellipse cx="520" cy="170" rx="131.9" ry="9.3" transform="rotate(-45 520 170)" fill="none" stroke="#10b981" stroke-width="2.4"/>
  <g fill="currentColor" font-size="12.5" text-anchor="middle" font-family="sans-serif">
    <text x="170" y="300">Σ = [[1,1],[1,1]],  det Σ = 0</text>
    <text x="170" y="317" font-size="11">rank 1: all mass on x₂ = x₁; no Lebesgue density</text>
    <text x="520" y="300">Σ + εI,  det = 0.21 and 0.0201</text>
    <text x="520" y="317" font-size="11">rank 2: density exists; peak 0.347 (ε=0.1), 1.123 (ε=0.01)</text>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="20" y="345">Figure 5.2 — Left: the singular Gaussian (Z, Z). Every open ball missing the line, such as the dashed red one, has</text>
    <text x="20" y="363">probability zero, and so does every subset of the line with zero length — so no density w.r.t. λ₂ can exist. Right:</text>
    <text x="20" y="378">jittering by εI (faint, ε = 0.1; green, ε = 0.01) restores a density by fattening the line into a thin ellipse.</text>
  </g>
</svg>
</figure>

> **Worked Example 5.8 (evaluating a 2-D Gaussian density).**
> $\mu = (1,-1)^\top$, $\Sigma = \begin{pmatrix}4&2\\2&3\end{pmatrix}$, evaluate $p$ at $x = (2,0)^\top$.
> Step 1: $x-\mu = (1,1)^\top$.
> Step 2: quadratic form $(x-\mu)^\top\Sigma^{-1}(x-\mu) = (1,1)\begin{pmatrix}0.375&-0.25\\-0.25&0.5\end{pmatrix}\binom{1}{1} = 0.375-0.25-0.25+0.5 = 0.375$.
> Step 3: normalizer $(2\pi)^{-1}(\det\Sigma)^{-1/2} = \frac{1}{2\pi\sqrt8} = 0.0562698$.
> Step 4: $p(x) = 0.0562698 \times e^{-0.1875} = 0.0466493$.
> A numerical integration of $p$ over $[-15,17]^2$ on a $2001\times2001$ grid returns $1.0000000$, confirming normalization to seven decimals.
>
> Notice that $p(x)$ here is less than $1$, but nothing forces that: at $x=\mu$ the density is $0.05627$, and shrinking $\Sigma$ to $0.01\,I_2$ would give $p(\mu) = 1/(2\pi\cdot 0.01) = 15.9$. Densities exceed $1$ routinely; they are Radon–Nikodym derivatives, not probabilities. This was the **first of the Four Failures** flagged in Chapter 1 and closed in Chapter 2; the multivariate case adds nothing new to the resolution.

### 5.3.3 The Gaussian as an exponential family

**Proposition 5.19 (Gaussian in natural form).** For $\Sigma\succ0$, write $\Lambda := \Sigma^{-1}$ (the precision) and $\eta := \Lambda\mu$. Then
$$
p(x) = \exp\Big(\eta^\top x - \tfrac12 x^\top\Lambda x - A(\eta,\Lambda)\Big),\qquad A(\eta,\Lambda) = \tfrac12\eta^\top\Lambda^{-1}\eta + \tfrac12\log\det(2\pi\Lambda^{-1}),
$$
so the Gaussian family with $\Sigma\succ0$ is an exponential family (Ch. 3) with natural parameters $(\eta, -\tfrac12\Lambda)$ and sufficient statistics $(x, xx^\top)$.

*Proof.* Expand the exponent of Theorem 5.17(c): $-\frac12(x-\mu)^\top\Lambda(x-\mu) = -\frac12x^\top\Lambda x + \mu^\top\Lambda x - \frac12\mu^\top\Lambda\mu$. Set $\eta = \Lambda\mu$ and absorb $-\frac12\mu^\top\Lambda\mu = -\frac12\eta^\top\Lambda^{-1}\eta$ together with $\log$ of the normalizer into $-A$. $\square$

The exponential-family view (Ch. 3) explains several facts you may have taken on faith. The sufficient statistics are the first and second empirical moments, so maximum likelihood estimation of a Gaussian depends on the data only through $\bar x$ and $\frac1n\sum x_ix_i^\top$ — that is why fitting a Gaussian is a one-pass computation. The log-partition $A$ is convex in the natural parameters, so Gaussian MLE is a convex problem in $(\eta,\Lambda)$ (though not in $(\mu,\Sigma)$). And the "$\eta$ plus $\Lambda$" pair is exactly the *information form* used in information filters and in Gaussian belief propagation, where multiplying densities is addition of natural parameters — a fact we will exploit in §5.5.

> **Common Confusion 5.4.**
> *"Each of my variables is Gaussian, so the vector is Gaussian."*
>
> No — that is the difference between marginal and joint normality, and Theorem 5.17(b) is precise about it: joint Gaussianity requires **every** linear combination to be normal, not just the coordinate projections $e_i^\top X$.
>
> Counterexample in three lines. Let $Z\sim\mathcal N(0,1)$ and independently $S = \pm1$ with probability $1/2$ each. Put $X = Z$ and $Y = SZ$. Then $Y\sim\mathcal N(0,1)$ (condition on $S$; both branches give $\mathcal N(0,1)$), so both marginals are standard normal. Also $\operatorname{Cov}(X,Y) = \mathbb{E}[SZ^2] = \mathbb{E}[S]\mathbb{E}[Z^2] = 0$, so they are uncorrelated. But $X+Y = (1+S)Z$ equals $2Z$ with probability $1/2$ and $0$ with probability $1/2$ — a mixture with an atom at $0$, definitely not normal. So $(X,Y)$ is not jointly Gaussian; and $|X| = |Y|$ always, so $X$ and $Y$ are wildly dependent despite being uncorrelated and marginally normal.
>
> Where this bites in practice: papers that fit a Gaussian to each coordinate of a learned representation and conclude the representation "is Gaussian." Coordinatewise normality is a necessary condition and a very weak one.

**The geometry of a high-dimensional Gaussian: the mode is not typical.** Before leaving §5.3, one fact about $\mathcal N(0, I_d)$ in large $d$ deserves a page, because it overturns an intuition everyone imports from the bell curve in $d=1$ and because it matters directly for latent-variable models. In one dimension, "near the mode" and "typical" are the same thing: a standard normal sample is usually close to $0$. In high dimension they come apart completely. Write $\|Z\|^2 = \sum_{i=1}^d Z_i^2$, a sum of $d$ i.i.d. terms with mean $1$ and variance $2$ (the variance of $Z_1^2$ is $\mathbb{E}[Z^4]-1 = 3-1 = 2$, using the Gaussian fourth moment $\mathbb{E}[Z^4]=3$, obtainable by differentiating Chapter 4's MGF $e^{s^2/2}$ four times at $s=0$). So $\mathbb{E}\|Z\|^2 = d$, $\operatorname{Var}(\|Z\|^2) = 2d$, and already Chebyshev — no machinery beyond Chapter 1 — says $\|Z\|^2/d \to 1$ in probability. Chapter 4's sub-exponential concentration sharpens this dramatically: $\|Z\|$ itself concentrates around $\sqrt d$ with fluctuations of *constant* order — the standard deviation of $\|Z\|$ tends to $1/\sqrt 2 \approx 0.71$ and does not grow with $d$ at all (the sharp statement is recorded in §5.9). The picture to hold in mind: essentially all the mass of a high-dimensional standard Gaussian sits in a thin spherical shell of radius $\sqrt d$ and thickness $O(1)$ — the "soap bubble."

The numbers make it vivid. For $d = 100$: the exact chi-square computation gives $P(9 \le \|Z\| \le 11) = 0.8428$, while $P(\|Z\| \le 5)$ — being within *half* the typical radius of the mode — is $1.1\times10^{-15}$, and $P(\|Z\|\le 1)$ is $1.8\times10^{-80}$. (Monte Carlo with $4\times10^5$ samples, seed 42, gives $0.8422$ for the shell probability and, unsurprisingly, zero hits inside radius $5$.) Yet the *density* is maximal at the origin: $p(0)/p(z_{\text{typ}}) = e^{d/2} \approx 5.2\times10^{21}$ for any typical point $z_{\text{typ}}$ with $\|z_{\text{typ}}\| = \sqrt d$. There is no contradiction — this is Chapter 2's lesson that a density is a *ratio against volume*, and the volume of the region near the origin is so catastrophically small that an astronomical density times a negligible volume is still negligible probability. Densities exceed $1$; modes can be nowhere.

Two consequences for practice. First, **linear interpolation in a Gaussian latent space leaves the typical set**: if $z_1, z_2$ are independent draws from $\mathcal N(0,I_d)$, their midpoint $(z_1+z_2)/2 \sim \mathcal N(0, \tfrac12 I_d)$ has norm concentrating at $\sqrt{d/2} \approx 0.71\sqrt d$ (measured: $7.05$ against $\sqrt{50} = 7.07$ at $d=100$) — well inside the shell where the prior, and hence the trained decoder, has essentially never seen an input. This is why practitioners interpolate along the sphere ("slerp") rather than along the chord when walking through GAN or VAE latent spaces. Second, **"the most likely point" is a trap**: the latent $z = 0$ maximizes the prior density, but a decoder evaluated there is being asked about a point the prior visits with probability zero for all practical purposes, which is why "generate from the mode" produces blurry, over-smoothed outputs in many latent-variable models. Whenever a paper reasons about a high-dimensional Gaussian through its density peak, run this section through your head first.

> **Self-check 5.3.**
> 1. Which of the three definitions still makes sense when $\Sigma = 0$? What distribution do you get?
> 2. $X\sim\mathcal N(\mu,\Sigma)$ in $\mathbb{R}^3$ and $B = (1,1,1)$. What is the law of $BX$?
> 3. Why does the proof of (b) $\Rightarrow$ (a) need a uniqueness theorem for characteristic functions?
> 4. If $\Sigma\succ0$, how large can $p(x)$ be? Does this contradict $\int p\,d\lambda_d = 1$?

---

## 5.4 Marginals and conditionals: the heart of the chapter

Partition the vector. Write
$$
X = \begin{pmatrix}X_1\\X_2\end{pmatrix},\qquad
\mu = \begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},\qquad
\Sigma = \begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\ \Sigma_{21}&\Sigma_{22}\end{pmatrix},
$$
with $X_1\in\mathbb{R}^{d_1}$, $X_2\in\mathbb{R}^{d_2}$, $d_1+d_2 = d$, and $\Sigma_{21} = \Sigma_{12}^\top$. Throughout §5.4, $X\sim\mathcal N(\mu,\Sigma)$.

### 5.4.1 Marginals are free

**Proposition 5.20 (marginals).** $X_1\sim\mathcal N(\mu_1,\Sigma_{11})$.

*Proof.* $X_1 = BX$ with $B = (I_{d_1}\ \ 0)$; apply Corollary 5.18, and note $B\mu = \mu_1$, $B\Sigma B^\top = \Sigma_{11}$. $\square$

That is the entire proof, and it is worth appreciating how much labor Definition 5.15 just saved. If you had defined the Gaussian by its density, computing a marginal would require integrating out $x_2$ from a $d$-dimensional exponential of a quadratic form — a genuinely fiddly computation involving completing the square. With definition (a), the marginal is a coordinate selection, which is an affine map, and affine maps preserve Gaussianity by construction. **Read off the sub-block; you are done.**

There is also a useful negative lesson here: the marginal covariance $\Sigma_{11}$ *ignores* $\Sigma_{12}$ entirely. Whatever relationship $X_1$ has with $X_2$ leaves no trace in the marginal law of $X_1$. Marginalization is genuinely destructive.

### 5.4.2 Block inversion and the Schur complement

Conditioning, unlike marginalization, requires real algebra. The reason is that conditioning acts naturally on the *precision* matrix $\Lambda = \Sigma^{-1}$, while marginalization acts naturally on $\Sigma$. To move between them we need block inversion.

**Lemma 5.21 (block inversion; Schur complement).** Suppose $\Sigma\succ0$ with the block structure above (so $\Sigma_{22}\succ0$ as a principal submatrix of a positive definite matrix, and $\Sigma_{11}\succ0$ likewise). Define the **Schur complement of $\Sigma_{22}$ in $\Sigma$**:
$$
S \;:=\; \Sigma/\Sigma_{22} \;:=\; \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.
$$
Then $S\succ0$ and
$$
\Sigma^{-1} = \begin{pmatrix}\Lambda_{11}&\Lambda_{12}\\ \Lambda_{21}&\Lambda_{22}\end{pmatrix}
=\begin{pmatrix} S^{-1} & -S^{-1}\Sigma_{12}\Sigma_{22}^{-1}\\[2pt] -\Sigma_{22}^{-1}\Sigma_{21}S^{-1} & \Sigma_{22}^{-1}+\Sigma_{22}^{-1}\Sigma_{21}S^{-1}\Sigma_{12}\Sigma_{22}^{-1}\end{pmatrix}.
$$
In particular $\Lambda_{11} = S^{-1}$, i.e. $S = \Lambda_{11}^{-1}$, and $\Lambda_{12} = -\Lambda_{11}\Sigma_{12}\Sigma_{22}^{-1}$. Moreover $\det\Sigma = \det(\Sigma_{22})\det(S)$.

*Proof.* The clean route is a block $LDU$ factorization. Set $K := \Sigma_{12}\Sigma_{22}^{-1}$. Then one verifies directly by block multiplication that
$$
\Sigma = \begin{pmatrix}I & K\\ 0& I\end{pmatrix}
\begin{pmatrix}S & 0\\ 0 & \Sigma_{22}\end{pmatrix}
\begin{pmatrix}I & 0\\ K^\top& I\end{pmatrix}.
$$
Check the $(1,1)$ block of the right side: $S + K\Sigma_{22}K^\top = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} + \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22}\Sigma_{22}^{-1}\Sigma_{21} = \Sigma_{11}$. ✓ The $(1,2)$ block: $K\Sigma_{22} = \Sigma_{12}$. ✓ The $(2,2)$ block: $\Sigma_{22}$. ✓

Because the outer factors are unit-triangular (hence invertible with determinant $1$, and with explicit inverses obtained by negating $K$), the middle factor is congruent to $\Sigma$; congruence preserves positive definiteness, so $\operatorname{diag}(S,\Sigma_{22})\succ0$, giving $S\succ0$. Taking determinants gives $\det\Sigma = \det S\cdot\det\Sigma_{22}$. Inverting the product in reverse order,
$$
\Sigma^{-1} = \begin{pmatrix}I&0\\-K^\top&I\end{pmatrix}\begin{pmatrix}S^{-1}&0\\0&\Sigma_{22}^{-1}\end{pmatrix}\begin{pmatrix}I&-K\\0&I\end{pmatrix},
$$
and multiplying out yields exactly the displayed blocks. $\square$

The Schur complement will keep reappearing, so let me give it a name in words: **$S = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ is the covariance of $X_1$ *after* the part of it that is linearly predictable from $X_2$ has been removed.** The subtracted term $\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ is the covariance of the best linear predictor $KX_2$. Since $S\succ0$ we get $S \preceq \Sigma_{11}$: conditioning never increases (the linear-prediction part of) uncertainty.

### 5.4.3 The conditioning theorem, proved three ways

**Theorem 5.22 (Gaussian conditioning).** Let $X = (X_1,X_2)\sim\mathcal N(\mu,\Sigma)$ with $\Sigma\succ0$. Then for every $x_2\in\mathbb{R}^{d_2}$, the conditional distribution of $X_1$ given $X_2 = x_2$ is Gaussian:
$$
X_1 \mid X_2 = x_2 \ \sim\ \mathcal N\big(\mu_{1|2}(x_2),\ \Sigma_{1|2}\big),
$$
$$
\boxed{\ \mu_{1|2}(x_2) = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\qquad \Sigma_{1|2} = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = \Sigma/\Sigma_{22} = \Lambda_{11}^{-1}.\ }
$$
The conditional mean is an **affine function of $x_2$**; the conditional covariance **does not depend on $x_2$ at all**.

(Since $\Sigma\succ0$ implies $\Sigma_{22}\succ0$, the conditional density $p(x_1\mid x_2) := p(x_1,x_2)/p_2(x_2)$ is well defined for every $x_2$, no null-set caveats needed. Chapter 6 will define conditional expectation without assuming densities; here the elementary definition suffices.)

**Proof 1: completing the square.** By Theorem 5.17(c) and Proposition 5.20,
$$
p(x_1\mid x_2) = \frac{p(x_1,x_2)}{p_2(x_2)} \propto \exp\Big(-\tfrac12 (x-\mu)^\top\Lambda(x-\mu)\Big)
$$
as a function of $x_1$, with $x_2$ held fixed — the denominator and all $x_1$-free factors are absorbed into the proportionality. Write $u := x_1-\mu_1$, $v := x_2-\mu_2$ and expand using the block form of $\Lambda$:
$$
(x-\mu)^\top\Lambda(x-\mu) = u^\top\Lambda_{11}u + 2u^\top\Lambda_{12}v + v^\top\Lambda_{22}v .
$$
Only the first two terms involve $u$. Complete the square in $u$ (this is the multivariate version of $au^2+2bu = a(u+b/a)^2 - b^2/a$, valid because $\Lambda_{11}\succ0$ is invertible):
$$
u^\top\Lambda_{11}u + 2u^\top\Lambda_{12}v = \big(u + \Lambda_{11}^{-1}\Lambda_{12}v\big)^\top\Lambda_{11}\big(u+\Lambda_{11}^{-1}\Lambda_{12}v\big) - v^\top\Lambda_{21}\Lambda_{11}^{-1}\Lambda_{12}v .
$$
Verify by expanding the right-hand side: the cross terms give $2u^\top\Lambda_{11}\Lambda_{11}^{-1}\Lambda_{12}v = 2u^\top\Lambda_{12}v$ (using $\Lambda_{11}$ symmetric), ✓; the pure-$v$ terms cancel, ✓. Therefore, as a function of $x_1$,
$$
p(x_1\mid x_2)\ \propto\ \exp\Big(-\tfrac12\big(x_1 - m\big)^\top\Lambda_{11}\big(x_1-m\big)\Big),\qquad m := \mu_1 - \Lambda_{11}^{-1}\Lambda_{12}(x_2-\mu_2).
$$
An unnormalized density of the form $\exp(-\frac12(x_1-m)^\top P(x_1-m))$ with $P\succ0$ *must* be the $\mathcal N(m,P^{-1})$ density once normalized, because the normalizing constant is determined and Theorem 5.17(c) identifies the shape. Hence the conditional law is $\mathcal N(m,\Lambda_{11}^{-1})$. Finally translate to $\Sigma$-blocks using Lemma 5.21: $\Lambda_{11}^{-1} = S = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$, and $-\Lambda_{11}^{-1}\Lambda_{12} = -S\cdot(-S^{-1}\Sigma_{12}\Sigma_{22}^{-1}) = \Sigma_{12}\Sigma_{22}^{-1}$, so $m = \mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)$. $\square$

**Proof 2: construct an independent decomposition (no densities needed).** Define
$$
W := X_1 - \Sigma_{12}\Sigma_{22}^{-1}(X_2-\mu_2).
$$
Then $(W, X_2)$ is an affine image of $X$, hence jointly Gaussian by Corollary 5.18. Compute the cross-covariance:
$$
\operatorname{Cov}(W,X_2) = \operatorname{Cov}(X_1,X_2) - \Sigma_{12}\Sigma_{22}^{-1}\operatorname{Cov}(X_2,X_2) = \Sigma_{12}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22} = 0 .
$$
Jointly Gaussian and uncorrelated implies independent (Corollary 5.23, proved just below and not circularly — its proof uses only Lemma 5.16). Also $\mathbb{E}[W] = \mu_1$ and
$$
\operatorname{Cov}(W) = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} + \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22}\Sigma_{22}^{-1}\Sigma_{21} = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = S .
$$
Now $X_1 = W + \Sigma_{12}\Sigma_{22}^{-1}(X_2-\mu_2)$, a sum of something independent of $X_2$ and something determined by $X_2$. Conditioning on $X_2 = x_2$ freezes the second term to the constant $\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)$ and leaves $W$'s law untouched (independence). Hence $X_1\mid X_2=x_2 \sim \mathcal N(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\ S)$.

To hold this proof to the chapter's own standard, let us say exactly what the last step means under our working definition of conditioning, $p(x_1\mid x_2) := p(x_1,x_2)/p_2(x_2)$. The claim is that the density of $\mathcal N(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2), S)$, multiplied by the marginal density $p_2(x_2)$, recovers the joint density $p(x_1,x_2)$ — and that identity is precisely what the independence of $W$ and $X_2$ delivers: the joint density of $(W, X_2)$ factorizes (product-density criterion, Ch. 2), and the affine change of variables $(w, x_2) \mapsto (x_1, x_2) = (w + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\, x_2)$ has Jacobian determinant $1$ (Ch. 3), turning that factorization into exactly the required product. Exercise A5 asks you to verify the same identity by direct multiplication, which is the third proof. When Chapter 6 defines conditional distributions without densities, this argument survives verbatim while Proof 1 does not — one more reason to prefer it. $\square$

I like Proof 2 best. It explains *why* the conditional covariance is $x_2$-free: the residual $W$ is a fixed random vector, statistically unrelated to $X_2$, and conditioning cannot change the law of something independent of what you conditioned on. It also explains *why* the conditional mean is affine: because the decomposition $X_1 = (\text{independent part}) + (\text{linear function of }X_2)$ exists at all, which is a special property of the Gaussian. And it works verbatim when $\Sigma_{11}$ is singular, since no inverse of $\Sigma_{11}$ was ever taken.

**Proof 3 (sketch): guess and verify.** Any candidate conditional must satisfy $p(x_1\mid x_2)p_2(x_2) = p(x_1,x_2)$. Multiply the claimed $\mathcal N(m(x_2),S)$ density by the $\mathcal N(\mu_2,\Sigma_{22})$ density, use $\det\Sigma = \det S\det\Sigma_{22}$ (Lemma 5.21) to see the normalizers match, and expand the exponents to see the quadratic forms match. This is Exercise A5.

**Corollary 5.23 (uncorrelated blocks of a jointly Gaussian vector are independent).** If $X = (X_1,X_2)$ is jointly Gaussian and $\Sigma_{12} = 0$, then $X_1 \perp X_2$.

*Proof.* By Lemma 5.16, $\varphi_X(t) = \exp(it^\top\mu - \frac12t^\top\Sigma t)$. Write $t = (t_1,t_2)$. With $\Sigma_{12}=0$, $t^\top\Sigma t = t_1^\top\Sigma_{11}t_1 + t_2^\top\Sigma_{22}t_2$ and $t^\top\mu = t_1^\top\mu_1+t_2^\top\mu_2$, so $\varphi_X(t) = \varphi_{X_1}(t_1)\varphi_{X_2}(t_2)$. A joint characteristic function that factorizes implies the joint law equals the product of the marginals (again by uniqueness of characteristic functions, applied on $\mathbb{R}^{d}$ to the two measures $P_X$ and $P_{X_1}\otimes P_{X_2}$), which is Definition 1.23. $\square$

This corollary is exactly what the warning at the end of §5.1.3 told you not to over-apply: it needs *joint* Gaussianity. Note also that it is a statement about blocks, not just scalars, and that no density is required — it holds for singular $\Sigma$ too.

**What if $\Sigma_{22}$ itself is singular?** Theorem 5.22 assumed $\Sigma \succ 0$, and this chapter has insisted that singular Gaussians are the *typical* case in ML — so the omission would be hypocritical if left unaddressed. The answer is that the conditioning formula survives with the Moore–Penrose pseudoinverse in place of the inverse: for any jointly Gaussian $(X_1, X_2)$, and any $x_2$ in the support $\mu_2 + \operatorname{range}(\Sigma_{22})$,
$$
X_1 \mid X_2 = x_2 \ \sim\ \mathcal N\big(\mu_1 + \Sigma_{12}\Sigma_{22}^{+}(x_2-\mu_2),\ \ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{+}\Sigma_{21}\big),
$$
and the same Proof-2 strategy establishes it, with one extra lemma doing the work: **positive semidefiniteness of $\Sigma$ forces $\operatorname{range}(\Sigma_{21}) \subseteq \operatorname{range}(\Sigma_{22})$** — a degenerate direction of $X_2$ (a direction $a$ with $\operatorname{Var}(a^\top X_2) = 0$) cannot covary with anything, by Cauchy–Schwarz (Proposition 5.5), so the cross-covariance never "points outside" what $\Sigma_{22}^{+}$ can invert. Conditioning on values *outside* the support is genuinely meaningless (you would be conditioning on an event no version of the conditional can see; Chapter 6 makes the null-set issues precise). Exercise A10 walks you through the full proof; Exercise A8 is its companion for densities. If you only remember the slogan: **replace $\Sigma_{22}^{-1}$ by $\Sigma_{22}^{+}$, restrict to the support, and everything in §5.4 goes through.**

> **Worked Example 5.9 (2-D conditioning, numerically verified).**
> $\mu = 0$, $\Sigma = \begin{pmatrix}1&0.8\\0.8&1\end{pmatrix}$. Condition on $X_2 = 1.5$:
> $$\mathbb{E}[X_1\mid X_2=1.5] = 0 + \frac{0.8}{1}(1.5-0) = 1.2,\qquad \operatorname{Var}(X_1\mid X_2) = 1 - \frac{0.8^2}{1} = 0.36,\ \text{sd} = 0.6 .$$
> **Verification by rejection-conditioning.** Draw $4\times10^6$ samples from $\mathcal N(0,\Sigma)$ via Cholesky (seed 7), keep those with $|X_2 - 1.5| < 0.02$. Result: $20{,}688$ samples retained, empirical mean $1.2051$, empirical variance $0.3617$. Against the theoretical $1.2$ and $0.36$, with a Monte Carlo standard error of about $0.6/\sqrt{20688} = 0.0042$ on the mean, that is agreement to well within noise.
>
> Three things to notice. (i) The conditional mean $1.2$ is **shrunk** toward $0$ relative to $x_2 = 1.5$; the shrinkage factor $\rho = 0.8$ is regression to the mean, and it is a theorem, not a psychological curiosity. (ii) The conditional variance $0.36 = 1-\rho^2$ is smaller than the marginal variance $1$ — observing $X_2$ told us something. (iii) Condition instead on $X_2 = 5.0$: the conditional variance is *still* $0.36$. Even at an absurdly improbable value of $X_2$, the Gaussian claims exactly the same residual uncertainty. That is Theorem 5.22's "$\Sigma_{1|2}$ does not depend on $x_2$," and it is the single most consequential *idealization* in Gaussian modeling.

<figure>
<svg viewBox="0 0 700 440" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Conditioning a bivariate Gaussian by slicing and renormalizing">
  <g stroke="currentColor" stroke-width="1" opacity="0.4">
    <line x1="150" y1="200" x2="470" y2="200"/><line x1="300" y1="60" x2="300" y2="345"/>
  </g>
  <g fill="currentColor" font-size="11" font-family="sans-serif">
    <text x="466" y="216">x₁</text><text x="306" y="70">x₂</text>
    <text x="184" y="215">−2</text><text x="404" y="215">2</text>
    <text x="284" y="314">−2</text><text x="288" y="94">2</text>
  </g>
  <polyline points="282.6,182.6 294.0,172.0 305.7,162.6 317.2,154.9 327.8,149.3 337.2,145.9 344.9,145.0 350.6,146.6 354.0,150.6 355.0,156.8 353.5,165.0 349.5,174.8 343.3,185.6 335.2,197.2 325.5,208.9 314.6,220.1 303.1,230.5 291.4,239.5 280.1,246.7 269.7,251.8 260.7,254.5 253.4,254.8 248.3,252.6 245.5,248.1 245.2,241.3 247.3,232.7 251.8,222.6 258.5,211.5 267.1,199.9 277.2,188.3 282.6,182.6" fill="none" stroke="#3b82f6" stroke-width="1.8" opacity="0.9"/>
  <polyline points="272.5,172.5 290.6,155.7 309.0,140.9 327.1,128.7 344.0,119.8 358.8,114.5 371.0,113.1 380.0,115.6 385.4,121.9 386.9,131.7 384.5,144.6 378.3,160.1 368.5,177.3 355.7,195.6 340.3,214.0 323.1,231.9 304.8,248.2 286.4,262.4 268.5,273.8 252.1,281.9 237.8,286.2 226.4,286.7 218.3,283.2 213.8,276.0 213.3,265.3 216.7,251.7 223.8,235.8 234.4,218.2 248.0,199.8 263.9,181.4 272.5,172.5" fill="none" stroke="#3b82f6" stroke-width="1.6" opacity="0.6"/>
  <polyline points="257.4,157.4 285.4,131.4 314.0,108.5 342.0,89.7 368.1,75.8 391.1,67.6 410.0,65.4 423.9,69.3 432.2,79.1 434.6,94.3 430.9,114.3 421.2,138.2 406.1,164.9 386.2,193.1 362.3,221.7 335.7,249.3 307.5,274.7 278.9,296.7 251.3,314.3 225.8,326.7 203.8,333.5 186.0,334.2 173.5,328.8 166.6,317.6 165.8,301.1 171.0,280.0 182.1,255.4 198.5,228.2 219.5,199.7 244.1,171.2 257.4,157.4" fill="none" stroke="#3b82f6" stroke-width="1.4" opacity="0.4"/>
  <line x1="185.6" y1="343.0" x2="414.4" y2="57.0" stroke="#10b981" stroke-width="2" stroke-dasharray="6 4"/>
  <line x1="150" y1="117.5" x2="500" y2="117.5" stroke="#ef4444" stroke-width="2"/>
  <polyline points="179.0,117.5 195.2,117.5 211.4,117.5 227.7,117.5 243.9,117.4 260.1,117.2 276.3,116.0 292.5,112.5 308.8,104.2 325.0,89.8 341.2,72.3 357.4,59.5 373.6,59.1 389.9,71.3 406.1,88.8 422.3,103.5 438.5,112.1 454.7,115.9 471.0,117.1 487.2,117.4 498.0,117.5" fill="none" stroke="#ef4444" stroke-width="2.2"/>
  <line x1="366" y1="117.5" x2="366" y2="57.6" stroke="#ef4444" stroke-width="1" stroke-dasharray="3 3"/>
  <circle cx="366" cy="117.5" r="3.5" fill="#ef4444"/>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="505" y="121">slice x₂ = 1.5</text>
    <text x="418" y="52">E[X₁|x₂] = 0.8·x₂</text>
    <text x="330" y="50" font-size="11">mean 1.2</text>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="20" y="380">Figure 5.3 — Conditioning a bivariate Gaussian (ρ = 0.8) is slicing and renormalizing. Blue: joint density contours at</text>
    <text x="20" y="398">Mahalanobis levels 1.0, 2.5 and 5.99. Red: the slice x₂ = 1.5 and the renormalized conditional density along it, a</text>
    <text x="20" y="416">Gaussian with mean 1.2 and sd 0.6. Green dashed: the conditional-mean line E[X₁|X₂=x₂] = 0.8·x₂ — the slice's peak</text>
    <text x="20" y="434">tracks this line as x₂ moves, while the slice's WIDTH (sd 0.6) is the same at every height. That constancy is Theorem 5.22.</text>
  </g>
</svg>
</figure>

> **Common Confusion 5.5.**
> *"Surely the conditional variance should depend on where I condition — I know less about $X_1$ out in the tail of $X_2$ than near the mean."*
>
> Your intuition is excellent statistics and bad Gaussian algebra. For a genuine Gaussian, the answer really is that $\operatorname{Var}(X_1\mid X_2 = x_2) = \Sigma_{1|2}$ for every $x_2$, however extreme. Proof 2 shows why: $X_1$ splits into a linear function of $X_2$ plus an independent residual $W$, and conditioning cannot alter the law of $W$.
>
> But now notice what this *costs* you. Your intuition is right about the world, and the Gaussian is wrong about it. Real predictive uncertainty usually grows as you leave the data. Three consequences worth knowing by name:
> - A Gaussian process with a stationary kernel has posterior variance that grows away from data (§5.8) — but that is a different mechanism: the *index* moves away from observed indices, not the conditioning value moving into a tail. Fix the test point, and doubling the observed $y$-value changes the GP posterior mean while leaving the posterior variance *identical*. GP error bars are entirely data-value-independent. That surprises people, and it is worth checking in code once.
> - Heteroscedastic regression exists precisely because the Gaussian's constant conditional variance is inadequate; you must model $\sigma(x)$ explicitly, which takes you outside the joint-Gaussian family.
> - Bayesian linear regression (§5.6.1) inherits this: the predictive variance $x_*^\top\Sigma_{\text{post}}x_* + \sigma^2$ depends on $x_*$ but not on the observed $y$'s. Two datasets with identical $X$ but wildly different $y$ produce identical error bars. If that sounds wrong, it is — it is a real limitation of the model, not a subtlety of the math.

### 5.4.4 The precision matrix and the Gaussian graphical model

Theorem 5.22 gave us $\Lambda_{11} = \Sigma_{1|2}^{-1}$: the top-left block of the *precision* matrix is the inverse conditional covariance. Push that observation to its limit by taking $X_1$ to be a single pair of coordinates.

**Definition 5.24 (precision matrix).** For $\Sigma\succ0$, the **precision matrix** (or *concentration matrix*, or *information matrix*) is $\Lambda := \Sigma^{-1}$.

**Theorem 5.25 (precision zeros = conditional independence).** Let $X\sim\mathcal N(\mu,\Sigma)$ with $\Sigma\succ0$ and $\Lambda = \Sigma^{-1}$. Fix $i\ne j$ and let $R := \{1,\dots,d\}\setminus\{i,j\}$ denote all the other coordinates. Then
$$
\Lambda_{ij} = 0 \iff X_i \perp X_j \mid X_R .
$$
Moreover the **partial correlation** is $\rho_{ij\cdot R} = -\Lambda_{ij}/\sqrt{\Lambda_{ii}\Lambda_{jj}}$.

*Proof.* Apply Theorem 5.22 with the "$1$" block equal to $\{i,j\}$ and the "$2$" block equal to $R$. The conditional law of $(X_i,X_j)$ given $X_R = x_R$ is Gaussian with covariance $\Sigma_{\{i,j\}|R} = (\Lambda_{\{i,j\}\{i,j\}})^{-1}$, where $\Lambda_{\{i,j\}\{i,j\}} = \begin{pmatrix}\Lambda_{ii}&\Lambda_{ij}\\\Lambda_{ij}&\Lambda_{jj}\end{pmatrix}$ is the $2\times2$ submatrix of $\Lambda$ (this is Lemma 5.21's identity $\Lambda_{11} = S^{-1}$ read backwards). Inverting a $2\times2$ matrix,
$$
\Sigma_{\{i,j\}|R} = \frac{1}{\Lambda_{ii}\Lambda_{jj}-\Lambda_{ij}^2}\begin{pmatrix}\Lambda_{jj} & -\Lambda_{ij}\\ -\Lambda_{ij}&\Lambda_{ii}\end{pmatrix}.
$$
Its off-diagonal entry vanishes iff $\Lambda_{ij}=0$. By Corollary 5.23 applied to the (jointly Gaussian) conditional law, a vanishing off-diagonal entry is equivalent to conditional independence of $X_i$ and $X_j$ given $X_R = x_R$; since this holds for every $x_R$, it is conditional independence in the sense of Definition 1.24. The partial correlation formula is the correlation read off the displayed matrix: $-\Lambda_{ij}/\sqrt{\Lambda_{jj}\Lambda_{ii}}$. $\square$

This theorem is the foundation of **Gaussian graphical models**: build a graph $G$ on $d$ nodes with an edge $\{i,j\}$ whenever $\Lambda_{ij}\ne0$; then the graph's missing edges are exactly the conditional independencies. Estimating a sparse $\Lambda$ from data is the graphical lasso, which maximizes $\log\det\Lambda - \operatorname{tr}(\hat\Sigma\Lambda) - \alpha\|\Lambda\|_1$ — convex, thanks to Proposition 5.19's exponential-family structure.

> **Worked Example 5.10 (a three-node chain: $\Sigma$ is dense, $\Lambda$ is sparse).**
> Take the AR(1)-style covariance with $\rho = 0.5$:
> $$\Sigma = \begin{pmatrix}1&0.5&0.25\\0.5&1&0.5\\0.25&0.5&1\end{pmatrix},\qquad \Lambda = \Sigma^{-1} = \begin{pmatrix}4/3 & -2/3 & 0\\ -2/3 & 5/3 & -2/3\\ 0 & -2/3 & 4/3\end{pmatrix} \approx \begin{pmatrix}1.3333&-0.6667&0\\-0.6667&1.6667&-0.6667\\0&-0.6667&1.3333\end{pmatrix}.$$
> Every entry of $\Sigma$ is nonzero: $X_1$ and $X_3$ have correlation $0.25$, so they are marginally dependent. But $\Lambda_{13} = 0$ **exactly**, so $X_1\perp X_3\mid X_2$: once you know the middle variable, the two ends carry no further information about each other. The graph is the chain $1 - 2 - 3$.
>
> Direct check via Theorem 5.22 with block $1 = \{1,3\}$, block $2 = \{2\}$:
> $$\Sigma_{\{1,3\}|2} = \begin{pmatrix}1&0.25\\0.25&1\end{pmatrix} - \begin{pmatrix}0.5\\0.5\end{pmatrix}(1)^{-1}\begin{pmatrix}0.5&0.5\end{pmatrix} = \begin{pmatrix}0.75&0\\0&0.75\end{pmatrix},$$
> off-diagonal exactly $0$. ✓ (A second numeric instance: $\Lambda = \begin{pmatrix}1&-0.5&0\\-0.5&1.5&-0.5\\0&-0.5&1\end{pmatrix}$ inverts to $\Sigma = \begin{pmatrix}1.25&0.5&0.25\\0.5&1&0.5\\0.25&0.5&1.25\end{pmatrix}$, with $\operatorname{corr}(X_1,X_3) = 0.25/1.25 = 0.2 \ne 0$ but conditional covariance $\begin{pmatrix}1&0\\0&1\end{pmatrix}$ given $X_2$.)
>
> **ML reading.** This is exactly the structure of a linear-Gaussian state space model — the Kalman filter's chain — and of a Gaussian Markov random field. It also explains why sparsity is a modeling assumption about $\Lambda$, not $\Sigma$: real dependence structures are usually locally sparse and globally dense.

<figure>
<svg viewBox="0 0 700 320" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Precision matrix zeros correspond to missing edges in a graph">
  <g font-family="monospace" font-size="13" fill="currentColor">
    <text x="30" y="45">Λ = Σ⁻¹</text>
    <text x="40" y="80">1.33  −0.67   0</text>
    <text x="40" y="105">−0.67  1.67  −0.67</text>
    <text x="40" y="130">0    −0.67   1.33</text>
  </g>
  <rect x="152" y="66" width="34" height="20" fill="none" stroke="#ef4444" stroke-width="2"/>
  <rect x="40" y="116" width="18" height="20" fill="none" stroke="#ef4444" stroke-width="2"/>
  <g font-family="sans-serif" font-size="11" fill="#ef4444">
    <text x="196" y="81">Λ₁₃ = Λ₃₁ = 0</text>
  </g>
  <g stroke="currentColor" stroke-width="1.6" fill="none">
    <line x1="430" y1="100" x2="510" y2="100"/><line x1="510" y1="100" x2="590" y2="100"/>
  </g>
  <line x1="430" y1="100" x2="590" y2="100" stroke="#ef4444" stroke-width="1.6" stroke-dasharray="5 5" opacity="0"/>
  <path d="M 430 100 Q 510 40 590 100" fill="none" stroke="#ef4444" stroke-width="1.6" stroke-dasharray="6 5"/>
  <line x1="500" y1="52" x2="520" y2="72" stroke="#ef4444" stroke-width="2.4"/>
  <line x1="520" y1="52" x2="500" y2="72" stroke="#ef4444" stroke-width="2.4"/>
  <g fill="currentColor">
    <circle cx="430" cy="100" r="18" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <circle cx="510" cy="100" r="18" fill="none" stroke="currentColor" stroke-width="1.8"/>
    <circle cx="590" cy="100" r="18" fill="none" stroke="currentColor" stroke-width="1.8"/>
  </g>
  <g font-family="sans-serif" font-size="13" fill="currentColor" text-anchor="middle">
    <text x="430" y="105">X₁</text><text x="510" y="105">X₂</text><text x="590" y="105">X₃</text>
    <text x="510" y="150">graph of Λ: edge ⟺ Λᵢⱼ ≠ 0</text>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="20" y="200">Figure 5.4 — Precision zeros are missing edges. Here Σ is dense (corr(X₁,X₃) = 0.25 ≠ 0, so X₁ and X₃ ARE marginally</text>
    <text x="20" y="218">dependent), yet Λ₁₃ = 0 exactly, so X₁ ⟂ X₃ | X₂. Knowing the middle variable screens off the two ends. Zeros in Σ mean</text>
    <text x="20" y="236">MARGINAL independence; zeros in Λ mean CONDITIONAL independence given everything else. These are different claims,</text>
    <text x="20" y="254">and neither implies the other. The graph shown is the conditional-independence graph, not a picture of Σ.</text>
  </g>
</svg>
</figure>

> **Common Confusion 5.6.**
> *"$\Sigma_{ij} = 0$ means $X_i\perp X_j$ (for Gaussians), so $\Lambda_{ij} = 0$ must mean the same thing."*
>
> They are different, and Worked Example 5.10 is the counterexample sitting right there: $\Sigma_{13} = 0.25 \ne 0$ while $\Lambda_{13} = 0$. Marginal independence and conditional independence are logically independent conditions (Definition 1.24 already warned you: neither implies the other).
>
> A vivid case in the other direction. Let $X_1, X_2$ be independent $\mathcal N(0,1)$ and $X_3 = X_1+X_2$. Then $\Sigma_{12} = 0$ — marginally independent. But given $X_3 = 4$, knowing $X_1 = 3$ forces $X_2 = 1$: conditionally, they are perfectly (negatively) dependent. This is "explaining away," and in the precision matrix it shows up as $\Lambda_{12}\ne0$. (Strictly, this $3$-vector has singular covariance, so compute with $X_3 = X_1+X_2+\varepsilon$ for small independent $\varepsilon$; the conclusion is the same.)
>
> The practical version: a feature-correlation heatmap (a picture of $\Sigma$) and a conditional-dependence graph (a picture of $\Lambda$) answer different questions, and the second is usually the one you meant to ask.

> **Self-check 5.4.**
> 1. In Theorem 5.22, which is bigger in the PSD order: $\Sigma_{11}$ or $\Sigma_{1|2}$? Why?
> 2. If $\Sigma_{12}=0$, what does the conditioning formula reduce to, and does that match intuition?
> 3. Give the conditional mean and variance for $\Sigma = \begin{pmatrix}4&2\\2&3\end{pmatrix}$, $\mu=0$, conditioning on $X_2 = 3$.
> 4. Why is $\Lambda_{11}^{-1}$, and not $(\Sigma^{-1})_{11}$ inverted entrywise, the conditional covariance?

---

## 5.5 The Gaussian identities toolkit

This section is a toolbox. Everything in it follows from §5.3 and §5.4, but the specific forms below are the ones you will meet, verbatim, in derivations of Kalman filters, variational autoencoders, Bayesian regression, and Gaussian process regression. There are five: sums, products, the linear-Gaussian lemma, the completing-the-square procedure, and the closed-form KL divergence. I recommend learning the *pattern* — completing the square — rather than memorizing the formulas, and I give the pattern explicitly in §5.5.4; the KL formula of §5.5.5 is the one exception worth committing to memory, because you will type it.

### 5.5.1 Sums

**Proposition 5.26 (independent sums).** If $X\sim\mathcal N(\mu_X,\Sigma_X)$ and $Y\sim\mathcal N(\mu_Y,\Sigma_Y)$ are independent random vectors in $\mathbb{R}^d$, then $X+Y\sim\mathcal N(\mu_X+\mu_Y,\ \Sigma_X+\Sigma_Y)$. More generally, for $A,B$ of compatible shape, $AX+BY \sim \mathcal N(A\mu_X+B\mu_Y,\ A\Sigma_XA^\top + B\Sigma_YB^\top)$.

*Proof.* Independence makes the stacked vector $(X,Y)$ jointly Gaussian with block-diagonal covariance $\operatorname{diag}(\Sigma_X,\Sigma_Y)$ — jointly Gaussian because $X = \mu_X+A_XZ_1$, $Y = \mu_Y+A_YZ_2$ with $Z_1\perp Z_2$ standard normal, so $(X,Y)$ is an affine image of the single standard normal vector $(Z_1,Z_2)$. Then $AX+BY = (A\ \ B)(X,Y)^\top$ is an affine image; apply Corollary 5.18 and multiply out the blocks. $\square$

The independence hypothesis is essential for the covariance, though not for Gaussianity: if $X,Y$ are jointly (not merely marginally) Gaussian but dependent, $X+Y$ is still Gaussian with covariance $\Sigma_X+\Sigma_{XY}+\Sigma_{YX}+\Sigma_Y$. If they are only *marginally* Gaussian, $X+Y$ need not be Gaussian at all — that is Common Confusion 5.4's counterexample.

> **Worked Example 5.11 (a two-line computation you will do constantly).**
> $X\sim\mathcal N(1,4)$, $Y\sim\mathcal N(-2,9)$, independent scalars. Then $X+Y\sim\mathcal N(-1,13)$ and $2X-Y\sim\mathcal N(2\cdot1-(-2),\ 4\cdot4+1\cdot9) = \mathcal N(4,25)$. Note the coefficient enters the variance *squared*: $\operatorname{Var}(2X) = 4\operatorname{Var}(X) = 16$, and the minus sign disappears because $(-1)^2 = 1$. In the residual-stream arithmetic of a transformer, this is why adding $L$ independent sub-layer outputs of variance $v$ each gives variance $Lv$ and hence standard deviation $\sqrt{L}\,\sqrt v$ — the growth of residual-stream norm with depth is Proposition 5.26 and nothing more.

### 5.5.2 Products of Gaussian densities

Multiplying two Gaussian densities in the *same* variable is not a Gaussian density (it does not integrate to $1$), but it is a Gaussian *shape* times a constant. This is what happens whenever you multiply a prior by a likelihood.

**Lemma 5.27 (product of Gaussian densities).** Let $\mathcal N(x; m,P)$ denote the density of $\mathcal N(m,P)$ evaluated at $x$, with $P\succ0$. Then for $P_1,P_2\succ0$,
$$
\mathcal N(x; m_1,P_1)\,\mathcal N(x;m_2,P_2) = c\cdot\mathcal N(x;m_*,P_*),
$$
where, writing $\Lambda_i = P_i^{-1}$,
$$
\Lambda_* = \Lambda_1+\Lambda_2,\qquad P_* = \Lambda_*^{-1},\qquad m_* = P_*(\Lambda_1m_1+\Lambda_2m_2),\qquad c = \mathcal N(m_1;\,m_2,\ P_1+P_2).
$$

*Proof.* Collect the exponents. Up to $x$-free terms,
$$
-\tfrac12(x-m_1)^\top\Lambda_1(x-m_1) - \tfrac12(x-m_2)^\top\Lambda_2(x-m_2)
= -\tfrac12 x^\top(\Lambda_1+\Lambda_2)x + x^\top(\Lambda_1m_1+\Lambda_2m_2) + \text{const}.
$$
Complete the square: $-\frac12x^\top\Lambda_*x + x^\top b = -\frac12(x-\Lambda_*^{-1}b)^\top\Lambda_*(x-\Lambda_*^{-1}b) + \frac12 b^\top\Lambda_*^{-1}b$, with $b = \Lambda_1m_1+\Lambda_2m_2$. So the product is proportional to $\mathcal N(x;m_*,P_*)$ with the stated $m_*, P_*$. To identify the constant $c$, integrate both sides over $x$: the left integrates to $\int \mathcal N(x;m_1,P_1)\mathcal N(x;m_2,P_2)\,dx$, which is the density at $0$ of the difference of two independent Gaussians $\mathcal N(m_1,P_1)-\mathcal N(m_2,P_2)\sim\mathcal N(m_1-m_2, P_1+P_2)$ (Proposition 5.26), i.e. $\mathcal N(m_1;m_2,P_1+P_2)$; the right integrates to $c$. (The interchange of the product and the integral is just Fubini (Ch. 2) applied to a nonnegative integrand.) $\square$

The precision form is the memorable one: **precisions add, and the new mean is a precision-weighted average.** If you have seen "information filter" or "natural-parameter addition," this is it — and Proposition 5.19 explains why, since multiplying exponential-family densities adds natural parameters.

> **Worked Example 5.12 (fusing two noisy sensors).**
> Sensor A reports $m_1 = 2.0$ with $\sigma_1 = 1.0$; sensor B reports $m_2 = 3.0$ with $\sigma_2 = 0.5$. Treating both as Gaussian likelihoods for the same quantity and multiplying:
> $$\Lambda_* = \frac{1}{1^2}+\frac{1}{0.5^2} = 1 + 4 = 5,\qquad \sigma_*^2 = 0.2,\ \sigma_* = 0.4472,$$
> $$m_* = 0.2\,(1\cdot 2.0 + 4\cdot 3.0) = 0.2\times 14 = 2.8 .$$
> The fused estimate sits at $2.8$, four-fifths of the way toward the more precise sensor, and the fused standard deviation $0.4472$ is *smaller than either sensor's*. That is not sleight of hand: two independent measurements genuinely carry more information than one.
> The normalizer is $c = \mathcal N(2.0;\,3.0,\ 1.0+0.25) = \frac{1}{\sqrt{2\pi\cdot1.25}}e^{-1/(2\cdot1.25)} = 0.2391868$. Numerical integration of the product density over a fine grid returns $0.23918683$, matching to eight digits. That constant $c$ is not decorative — it is exactly the *marginal likelihood* (evidence) of the second reading under the first, and low values of $c$ flag sensor disagreement.

### 5.5.3 The linear-Gaussian lemma

This is the single most-used identity in the book.

**Theorem 5.28 (marginalization and conditioning for linear-Gaussian models).**
Let
$$
X\sim\mathcal N(\mu,\ \Lambda_x^{-1}),\qquad Y\mid X = x \ \sim\ \mathcal N(Ax+b,\ \Lambda_y^{-1}),
$$
with $\Lambda_x\succ0$, $\Lambda_y\succ0$, $A\in\mathbb{R}^{m\times d}$. (We write $\Lambda_y$, not $L$, for the observation precision: $L$ is already taken by the Cholesky factor of Definition 5.9, and this lemma is used in code where both appear at once.) Then the joint law of $(X,Y)$ is Gaussian, and

1. **(Marginal / prior predictive)** $\displaystyle Y\sim\mathcal N\big(A\mu+b,\ \Lambda_y^{-1}+A\Lambda_x^{-1}A^\top\big)$.
2. **(Posterior)** $\displaystyle X\mid Y=y\ \sim\ \mathcal N\big(\Sigma_{\!*}\{A^\top \Lambda_y(y-b)+\Lambda_x\mu\},\ \Sigma_{\!*}\big)$ with $\Sigma_{\!*} = (\Lambda_x + A^\top \Lambda_y A)^{-1}$.
3. **(Cross-covariance)** $\operatorname{Cov}(X,Y) = \Lambda_x^{-1}A^\top$.

*Proof.* Realize the model constructively, which immediately gives joint Gaussianity. Write $X = \mu + \Lambda_x^{-1/2}Z_1$ and $Y = AX+b+\varepsilon$ where $\varepsilon\sim\mathcal N(0,\Lambda_y^{-1})$ is independent of $X$ (this reproduces the stated conditional law: given $X=x$, $Y = Ax+b+\varepsilon$ has law $\mathcal N(Ax+b,\Lambda_y^{-1})$). Then
$$
\begin{pmatrix}X\\Y\end{pmatrix} = \begin{pmatrix}\mu\\A\mu+b\end{pmatrix} + \begin{pmatrix}I&0\\A&I\end{pmatrix}\begin{pmatrix}\Lambda_x^{-1/2}Z_1\\ L^{-1/2}Z_2\end{pmatrix},
$$
an affine image of a standard normal vector, hence jointly Gaussian by Definition 5.15. Its covariance is
$$
\begin{pmatrix}I&0\\A&I\end{pmatrix}\begin{pmatrix}\Lambda_x^{-1}&0\\0&\Lambda_y^{-1}\end{pmatrix}\begin{pmatrix}I&A^\top\\0&I\end{pmatrix}
=\begin{pmatrix}\Lambda_x^{-1} & \Lambda_x^{-1}A^\top\\ A\Lambda_x^{-1} & A\Lambda_x^{-1}A^\top + \Lambda_y^{-1}\end{pmatrix}.
$$
Claims (1) and (3) can be read straight off this matrix, using Proposition 5.20 for the marginal.

For (2), apply Theorem 5.22 with the roles $1 \leftrightarrow X$, $2\leftrightarrow Y$:
$$
\operatorname{Cov}(X\mid Y=y) = \Lambda_x^{-1} - \Lambda_x^{-1}A^\top\big(A\Lambda_x^{-1}A^\top+\Lambda_y^{-1}\big)^{-1}A\Lambda_x^{-1}.
$$
This equals $(\Lambda_x+A^\top \Lambda_y A)^{-1}$ by the Woodbury matrix identity, $(\Lambda_x + A^\top \Lambda_y A)^{-1} = \Lambda_x^{-1} - \Lambda_x^{-1}A^\top(\Lambda_y^{-1}+A\Lambda_x^{-1}A^\top)^{-1}A\Lambda_x^{-1}$ (stated in §5.9). The mean follows similarly, or more transparently by completing the square directly in the log-posterior:
$$
\log p(x\mid y) = -\tfrac12(y-Ax-b)^\top \Lambda_y(y-Ax-b) - \tfrac12(x-\mu)^\top\Lambda_x(x-\mu) + \text{const},
$$
whose $x$-quadratic coefficient is $-\frac12 x^\top(A^\top \Lambda_y A + \Lambda_x)x$ and whose $x$-linear coefficient is $x^\top(A^\top \Lambda_y(y-b) + \Lambda_x\mu)$. By the completing-the-square pattern below, the posterior is $\mathcal N(\Sigma_*b_*,\Sigma_*)$ with $\Sigma_* = (A^\top \Lambda_y A+\Lambda_x)^{-1}$ and $b_* = A^\top \Lambda_y(y-b)+\Lambda_x\mu$. $\square$

Notice the two faces of the posterior covariance. The form $(\Lambda_x + A^\top \Lambda_y A)^{-1}$ is a $d\times d$ inverse: prior precision plus data precision. The form $\Lambda_x^{-1} - \Lambda_x^{-1}A^\top(\cdot)^{-1}A\Lambda_x^{-1}$ requires an $m\times m$ inverse. **Use whichever dimension is smaller.** With $n$ data points and $d$ features, that choice is the difference between $O(d^3)$ (the "weight space" view of Bayesian linear regression) and $O(n^3)$ (the "function space" view, which is Gaussian process regression). They are the *same computation*, and §5.8 will make that identity explicit.

### 5.5.4 Completing the square, as a procedure

Here is the reusable recipe. Whenever you have a log-density in $x$ that is quadratic,
$$
\log f(x) = -\tfrac12 x^\top P x + x^\top b + \text{const},\qquad P \succ 0,
$$
you may immediately conclude
$$
f \propto \mathcal N\big(x;\ P^{-1}b,\ P^{-1}\big),
$$
and the missing constant is $\exp(\frac12 b^\top P^{-1}b)\cdot(2\pi)^{-d/2}(\det P)^{1/2}$ if you need $f$ itself normalized. The procedure in practice:

1. Write down every term of the log-density involving $x$; discard everything else into "const."
2. Collect all terms quadratic in $x$ into $-\frac12 x^\top P x$; **$P$ is the posterior precision.**
3. Collect all terms linear in $x$ into $x^\top b$.
4. The mean is $P^{-1}b$; the covariance is $P^{-1}$. Done.

That is it. Almost every "and after some algebra, the posterior is Gaussian with..." in a machine learning paper is steps 1–4. Practise it once on Theorem 5.28 with pen and paper and you will never again be at the mercy of an unexplained derivation.

A warning before you use any of this. Three different objects are routinely confused, and each corresponds to a different modelling operation. The **product of two Gaussian densities** (Lemma 5.27) is proportional to a Gaussian density; it is what you compute when you combine two pieces of evidence about the *same* quantity. The **convolution of two Gaussian densities** (Proposition 5.26) is the density of the *sum*, with variances adding; it is what you compute when independent noise accumulates. The **product of two Gaussian random variables** is neither: if $U,V\sim\mathcal N(0,1)$ are independent, $UV$ has a density involving a Bessel function $K_0$ that diverges logarithmically at the origin, and it is emphatically not Gaussian. The classic derivation bug is to fuse two sensors by adding variances (which makes you *less* certain) rather than adding precisions (which makes you more certain). If a fusion step ever leaves you more uncertain than either input, you have made this mistake.

### 5.5.5 Kullback–Leibler divergence between Gaussians

One identity is missing, and it is the one you will type most often. Every variational method — VAEs, variational GPs, variational Bayes, mean-field approximations, and the KL penalties inside PPO-style RL objectives — needs $D_{\mathrm{KL}}(q\,\|\,p)$ for Gaussian $q$ and $p$, and the Gaussian case has a closed form.

**Proposition 5.29 (KL divergence between Gaussians).** Let $P_1 = \mathcal N(\mu_1,\Sigma_1)$ and $P_2 = \mathcal N(\mu_2,\Sigma_2)$ on $\mathbb{R}^d$ with $\Sigma_1,\Sigma_2\succ0$. Then $P_1\ll P_2$ and
$$
D_{\mathrm{KL}}(P_1\,\|\,P_2) = \tfrac12\Big[\operatorname{tr}\big(\Sigma_2^{-1}\Sigma_1\big) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - d + \log\tfrac{\det\Sigma_2}{\det\Sigma_1}\Big].
$$

*Proof.* Both laws have positive densities w.r.t. $\lambda_d$, so $D_{\mathrm{KL}} = \mathbb{E}_{X\sim P_1}[\log p_1(X) - \log p_2(X)]$. The first term is $-h(P_1)$, computed in Proposition 5.14: $\mathbb{E}[\log p_1(X)] = -\frac12\log((2\pi)^d\det\Sigma_1) - \frac d2$. For the second,
$$
-\mathbb{E}[\log p_2(X)] = \tfrac12\log\big((2\pi)^d\det\Sigma_2\big) + \tfrac12\mathbb{E}\big[(X-\mu_2)^\top\Sigma_2^{-1}(X-\mu_2)\big].
$$
Write $X-\mu_2 = (X-\mu_1) + \delta$ with $\delta := \mu_1-\mu_2$ deterministic. Expanding, the cross term has expectation $0$ because $\mathbb{E}[X-\mu_1] = 0$, so by the same trace argument as before
$$
\mathbb{E}\big[(X-\mu_2)^\top\Sigma_2^{-1}(X-\mu_2)\big] = \operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + \delta^\top\Sigma_2^{-1}\delta .
$$
Adding the two pieces, the $(2\pi)^d$ factors cancel and the stated formula drops out. $\square$

Three special cases carry almost all the practical weight.

*Equal covariances.* If $\Sigma_1 = \Sigma_2 = \Sigma$, the trace term is $d$ and the log-det term is $0$, leaving $D_{\mathrm{KL}} = \frac12 d_\Sigma(\mu_1,\mu_2)^2$: **half the squared Mahalanobis distance** of Definition 5.12. Gaussian KL and Mahalanobis distance are the same object seen twice.

*Equal means, one-dimensional.* $D_{\mathrm{KL}} = \frac12(\sigma_1^2/\sigma_2^2 - 1 - \log(\sigma_1^2/\sigma_2^2))$, a function of the variance ratio alone, minimized at $0$ when the ratio is $1$.

*The VAE term.* With $q = \mathcal N(\mu,\operatorname{diag}(\sigma_1^2,\dots,\sigma_d^2))$ and $p = \mathcal N(0,I_d)$, everything is diagonal and the formula separates:
$$
D_{\mathrm{KL}}(q\,\|\,p) = \tfrac12\sum_{j=1}^d\big(\sigma_j^2 + \mu_j^2 - 1 - 2\log\sigma_j\big).
$$
This is line-for-line the `kld` term in every VAE implementation. Take $\mu = (0.5,-1.2,0,2)$ and $\sigma = (0.8,1.5,1,0.3)$: the per-coordinate contributions are $0.1681$, $0.9395$, $0$ and $2.7490$, totalling $3.8567$ nats. Read the third coordinate carefully. It has $\mu_3 = 0$, $\sigma_3 = 1$, meaning the encoder has learned to ignore the input entirely for that latent dimension — and its KL contribution is exactly $0$. **The KL term does not merely tolerate posterior collapse; it actively rewards it.** That is the whole mechanism behind the phenomenon Worked Example 5.7 described, and it is why KL annealing, free-bits floors and $\beta<1$ weighting exist. Note also the fourth coordinate: $\sigma_4 = 0.3$ is confidently informative and pays $2.749$ nats for the privilege.

Two cautions. First, KL is **not symmetric**: for $P_1 = \mathcal N\big((1,-1),\begin{pmatrix}4&2\\2&3\end{pmatrix}\big)$ and $P_2 = \mathcal N(0,I_2)$ we get $D_{\mathrm{KL}}(P_1\|P_2) = 2.4603$ while $D_{\mathrm{KL}}(P_2\|P_1) = 1.1647$ (a Monte Carlo estimate of the first with $4\times10^6$ samples, seed 23, returns $2.4578$). This asymmetry is not a defect to be tidied away — minimizing $D_{\mathrm{KL}}(q\|p)$ over $q$ is *mode-seeking*, since $q$ is penalized wherever $p$ is small, and produces the characteristically over-confident variational posteriors; minimizing $D_{\mathrm{KL}}(p\|q)$ is *mass-covering*. Which direction you write down is a modelling decision.

Second, the formula requires $\Sigma_1,\Sigma_2\succ0$ and blows up when either degenerates: $\log\det\Sigma_1\to-\infty$ sends $D_{\mathrm{KL}}(P_1\|P_2)\to+\infty$. That is Common Confusion 5.3 arriving with a bill. If your variational objective is diverging, check whether some $\log\sigma_j$ is being driven to $-\infty$ before you look anywhere else.

> **Self-check 5.5.**
> 1. In Lemma 5.27, what is the constant $c$ the density of, and evaluated at what point? Can $c$ exceed $1$?
> 2. Two independent estimates with variances $1$ and $4$. What is the fused variance, and how much weight does each estimate get?
> 3. In Theorem 5.28, what happens to the posterior as $\Lambda_y\to0$ (infinitely noisy observations)?
> 4. You have $\log f(x) = -3x^2 + 12x + 7$ for a scalar $x$. What Gaussian is this?
> 5. In the VAE formula, which is penalized more: a latent coordinate with $\sigma_j = 0.1$, or one with $\sigma_j = 10$? Why is the asymmetry there?

---

## 5.6 Linear-Gaussian models

Everything in §5.6 is Theorem 5.28 applied in a different costume.

### 5.6.1 Bayesian linear regression (previewed; full treatment in Chapter 10)

Model: $y = X\theta + \varepsilon$ with design matrix $X\in\mathbb{R}^{n\times p}$ (fixed), noise $\varepsilon\sim\mathcal N(0,\sigma^2I_n)$, and prior $\theta\sim\mathcal N(0,\alpha I_p)$.

**Proposition 5.30 (posterior and predictive).** With the above,
$$
\theta\mid y \ \sim\ \mathcal N\big(\Sigma_\theta X^\top y/\sigma^2,\ \Sigma_\theta\big),\qquad \Sigma_\theta = \Big(\tfrac{1}{\sigma^2}X^\top X + \tfrac1\alpha I_p\Big)^{-1},
$$
and for a new input $x_*$, the posterior predictive is
$$
y_*\mid y\ \sim\ \mathcal N\big(x_*^\top\mu_\theta,\ x_*^\top\Sigma_\theta x_* + \sigma^2\big).
$$

*Proof.* Theorem 5.28 with $A = X$, $b=0$, $\Lambda_x = \alpha^{-1}I_p$, $\Lambda_y = \sigma^{-2}I_n$, $\mu=0$. The predictive follows because $y_* = x_*^\top\theta+\varepsilon_*$ is an affine function of $\theta$ plus independent noise; apply Corollary 5.18 and Proposition 5.26. $\square$

The posterior mean $\Sigma_\theta X^\top y/\sigma^2 = (X^\top X + \frac{\sigma^2}{\alpha}I)^{-1}X^\top y$ is **exactly ridge regression** with $\lambda = \sigma^2/\alpha$. That correspondence — Gaussian prior $\leftrightarrow$ $\ell_2$ penalty — is one of the most reused facts in the field, and it drops out of the completing-the-square pattern in two lines.

> **Worked Example 5.13 (four data points, computed).**
> Fit an intercept and slope to $x = (0,1,2,3)$, $y = (0.2,\ 0.9,\ 2.1,\ 2.8)$, with $\sigma^2 = 0.25$ and prior $\theta\sim\mathcal N(0, 1\cdot I_2)$.
> $$X^\top X = \begin{pmatrix}4 & 6\\ 6& 14\end{pmatrix},\qquad \Sigma_\theta = \Big(\tfrac{1}{0.25}\begin{pmatrix}4&6\\6&14\end{pmatrix}+I_2\Big)^{-1} = \begin{pmatrix}17&24\\24&57\end{pmatrix}^{-1} = \begin{pmatrix}0.145038&-0.061069\\-0.061069&0.043257\end{pmatrix}.$$
> Posterior mean: $\mu_\theta = \Sigma_\theta X^\top y/\sigma^2 = (0.183206,\ 0.870229)^\top$. Compare the OLS estimate $(0.15,\ 0.90)^\top$: the prior has pulled the slope in from $0.90$ to $0.87$ and pushed the intercept out slightly, exactly the ridge shrinkage the correspondence predicts.
> Predicting at $x_* = (1,4)^\top$ (i.e. input value $4$): mean $= 0.183206 + 4(0.870229) = 3.664122$, variance $= x_*^\top\Sigma_\theta x_* + 0.25 = 0.348601 + 0.25 = 0.598601$, sd $= 0.7737$. The split is instructive: $0.348601$ is *epistemic* (uncertainty about $\theta$) and $0.25$ is *aleatoric* (irreducible observation noise). Evaluate instead at the centre of the data, $x_* = (1,1.5)^\top$, and the epistemic part collapses to $0.059160$ while the aleatoric part is unchanged at $0.25$. Extrapolation to $x=4$ costs you a sixfold increase in epistemic variance; no amount of extra data at $x\in[0,3]$ would ever reduce the $0.25$.

It is worth flagging honestly where this gets misapplied. Bayesian linear regression is often sold as giving uncertainty that will "tell you when you are out of distribution." Only partly. The predictive variance $x_*^\top\Sigma_\theta x_* + \sigma^2$ genuinely grows as $x_*$ moves into directions poorly covered by the design matrix — Worked Example 5.13 shows exactly that, sixfold from the data centre to $x=4$. But it grows quadratically along the *linear* directions the model already has, and it never notices nonlinear novelty at all: feed it an input from a completely different data manifold that happens to lie inside the span of the training inputs and you will get confident nonsense with tight error bars. Furthermore, as Common Confusion 5.5 pointed out, $\Sigma_\theta$ does not depend on $y$ at all, so the error bars are fixed by the design matrix before a single label is observed. Two datasets with identical inputs and wildly different outputs get identical uncertainty. That is a property of the linear-Gaussian assumption, not a bug in the derivation — but it does mean "Bayesian uncertainty" and "OOD detection" are different things, despite being conflated constantly.

### 5.6.2 The Kalman filter as repeated conditioning

**Theorem 5.31 (Kalman filter).** Consider the linear-Gaussian state-space model
$$
x_t = F x_{t-1} + w_t,\quad w_t\sim\mathcal N(0,Q);\qquad y_t = Hx_t + v_t,\quad v_t\sim\mathcal N(0,R),
$$
with $x_0\sim\mathcal N(m_0,P_0)$ and all noises independent of each other and of $x_0$. Write $m_{t|s}, P_{t|s}$ for the mean and covariance of $x_t$ given $y_{1:s}$. Then these are Gaussian for all $t$, and satisfy

*Predict:*
$$m_{t|t-1} = Fm_{t-1|t-1},\qquad P_{t|t-1} = FP_{t-1|t-1}F^\top + Q .$$
*Update:*
$$S_t = HP_{t|t-1}H^\top + R,\quad K_t = P_{t|t-1}H^\top S_t^{-1},$$
$$m_{t|t} = m_{t|t-1} + K_t(y_t - Hm_{t|t-1}),\qquad P_{t|t} = (I-K_tH)P_{t|t-1}.$$

*Proof (one step; the rest is induction).* Suppose inductively $x_{t-1}\mid y_{1:t-1}\sim\mathcal N(m_{t-1|t-1},P_{t-1|t-1})$. The *predict* step is Corollary 5.18 plus Proposition 5.26 applied to $x_t = Fx_{t-1}+w_t$ with $w_t$ independent of everything so far. For the *update* step, condition on $y_{1:t-1}$ throughout (legitimate because Proposition 1.11 says a conditional probability is itself a probability measure, so every Gaussian result transfers verbatim to the conditioned world). In that world the model reads
$$
x_t\sim\mathcal N(m_{t|t-1},P_{t|t-1}),\qquad y_t\mid x_t\sim\mathcal N(Hx_t,R),
$$
which is precisely Theorem 5.28 with $A = H$, $b = 0$, $\Lambda_x = P_{t|t-1}^{-1}$, $\Lambda_y = R^{-1}$. Using the *covariance-form* of Theorem 5.22 rather than the precision form: the joint covariance of $(x_t,y_t)$ is
$$
\begin{pmatrix}P_{t|t-1} & P_{t|t-1}H^\top\\ HP_{t|t-1} & HP_{t|t-1}H^\top+R\end{pmatrix},
$$
so conditioning on $y_t$ gives mean $m_{t|t-1} + P_{t|t-1}H^\top S_t^{-1}(y_t - Hm_{t|t-1})$ and covariance $P_{t|t-1}-P_{t|t-1}H^\top S_t^{-1}HP_{t|t-1}$. Substituting $K_t := P_{t|t-1}H^\top S_t^{-1}$ gives exactly the stated formulas. $\square$

Read the update in words. $y_t - Hm_{t|t-1}$ is the **innovation**: how surprised you are. $S_t$ is the innovation covariance: how surprised you *expected* to be. $K_t$ is the **Kalman gain**: the fraction of your surprise you should believe. When $R$ is large (noisy sensor), $K_t$ is small and you barely move. When $P_{t|t-1}$ is large (uncertain state), $K_t$ is large and you move a lot. And $P_{t|t}$ decreases regardless of the observed value — Common Confusion 5.5 again.

> **Worked Example 5.14 (two Kalman steps, all numbers computed).**
> Scalar model: $F=1$, $Q=0.1$, $H=1$, $R=0.5$, $m_0 = 0$, $P_0 = 1$. Observations $y_1 = 0.9$, $y_2 = 1.4$.
>
> **Step 1.** Predict: $m_{1|0} = 0$, $P_{1|0} = 1 + 0.1 = 1.1$. Innovation covariance $S_1 = 1.1+0.5 = 1.6$; gain $K_1 = 1.1/1.6 = 0.6875$. Update: $m_{1|1} = 0 + 0.6875(0.9-0) = 0.618750$; $P_{1|1} = (1-0.6875)(1.1) = 0.343750$.
>
> **Step 2.** Predict: $m_{2|1} = 0.618750$, $P_{2|1} = 0.343750+0.1 = 0.443750$. $S_2 = 0.943750$, $K_2 = 0.443750/0.943750 = 0.470199$. Update: $m_{2|2} = 0.618750+0.470199(1.4-0.618750) = 0.986093$; $P_{2|2} = (1-0.470199)(0.443750) = 0.235099$.
>
> Watch the gain fall from $0.6875$ to $0.4702$: as the filter accumulates information, $P$ shrinks, so each new observation moves the estimate less. Watch $P$ fall from $1$ to $0.3438$ to $0.2351$ — but not to zero, because process noise $Q = 0.1$ is re-injected at every predict step. In the long run $P_{t|t}$ converges to the fixed point of the scalar Riccati recursion $P = (1-K)(P+Q)$; iterating that map from $P=1$ for 500 steps gives the limit $0.179129$, with steady-state gain $0.358258$. Nothing in these recursions depended on the *values* $0.9$ and $1.4$: the covariance sequence could have been computed offline before any data arrived.

### 5.6.3 Factor analysis and probabilistic PCA

**Proposition 5.32 (factor analysis marginal).** Let $z\sim\mathcal N(0,I_k)$ and $x = Wz+\mu+\varepsilon$ with $\varepsilon\sim\mathcal N(0,\Psi)$ independent of $z$, $W\in\mathbb{R}^{d\times k}$, $\Psi$ diagonal positive. Then
$$
x\sim\mathcal N(\mu,\ WW^\top+\Psi),\qquad z\mid x \sim \mathcal N\big(M^{-1}W^\top\Psi^{-1}(x-\mu),\ M^{-1}\big),\ \ M := I_k + W^\top\Psi^{-1}W .
$$
**Probabilistic PCA** is the special case $\Psi = \sigma^2I_d$, giving marginal covariance $WW^\top + \sigma^2I_d$.

*Proof.* Theorem 5.28 with $A = W$, $b=\mu$, prior precision $I_k$, noise precision $\Psi^{-1}$. $\square$

Two consequences worth stating. First, **non-identifiability**: replacing $W$ by $WQ$ for any orthogonal $Q\in\mathbb{R}^{k\times k}$ leaves $WW^\top$ unchanged, hence leaves the marginal law of $x$ unchanged. The likelihood cannot distinguish them. This is Proposition 5.10's rotational ambiguity resurfacing in a different costume, and it is the mathematical reason "the factors are the interpretable causes" is an unjustified claim without additional structure (varimax rotation, sparsity, non-Gaussianity, or supervision). Second, the covariance $WW^\top+\sigma^2I$ has $k$ eigenvalues equal to (the eigenvalues of $WW^\top$) $+\ \sigma^2$ and $d-k$ eigenvalues equal to $\sigma^2$; the MLE for $W$ recovers the top-$k$ eigenvectors of the sample covariance, and as $\sigma^2\to0$ the PPCA posterior mean converges to the classical PCA projection. So PPCA is PCA with a noise model bolted on — and the noise model is what buys you a likelihood, hence model selection and missing-data handling.

> **Worked Example 5.15 (PPCA spectrum).**
> $k=1$, $W = (1.0,\ 0.8,\ 0.6)^\top$, $\sigma^2 = 0.2$, $d=3$. Then $W^\top W = 2.0$ and
> $$C = WW^\top+\sigma^2I = \begin{pmatrix}1.20&0.80&0.60\\0.80&0.84&0.48\\0.60&0.48&0.56\end{pmatrix}.$$
> Eigenvalues of $C$ (computed): $(2.2,\ 0.2,\ 0.2)$. Exactly as predicted: one "signal" eigenvalue equal to $\|W\|^2 + \sigma^2 = 2.0+0.2$, and $d-k = 2$ "noise" eigenvalues equal to $\sigma^2 = 0.2$. The flat tail of equal eigenvalues is the signature of isotropic noise, and looking for the "elbow" in a scree plot is, under this model, looking for where the spectrum flattens out to $\sigma^2$.

> **Self-check 5.6.**
> 1. In Bayesian linear regression, what happens to the posterior as $\alpha\to\infty$ (flat prior)?
> 2. In the Kalman filter, what is $K_t$ when $R = 0$ (perfect sensor)? Interpret.
> 3. Why is $P_{t|t}$ computable without seeing any data?
> 4. Give two matrices $W$ with the same PPCA marginal law and explain what that means for interpretability.

---

## 5.7 Payoff: deriving He and Xavier initialization from scratch

Here is the practical question. You are about to train a 20-layer fully connected ReLU network. You must choose the scale of the random initial weights. Choose too small and the signal dies as it propagates forward: by layer 20 every activation is $10^{-6}$ times its input scale, the outputs are effectively constant, and the gradients that flow back are equally tiny — the network does not train. Choose too large and activations blow up by a factor of $10^6$, the loss overflows, and you get `NaN` before the first epoch ends. There is a knife's edge, and this section computes exactly where it is.

The answer, famously, is $\operatorname{Var}(w) = 2/n_{\text{in}}$ for ReLU networks. The interesting part is the $2$. It comes from a single expectation — $\mathbb{E}[\mathrm{ReLU}(Z)^2] = 1/2$ for $Z\sim\mathcal N(0,1)$ — and that expectation is a two-line calculation with Chapter 1's running example $A = \max(0,Z)$.

### 5.7.1 The ReLU moments

**Lemma 5.33 (moments of $\mathrm{ReLU}(Z)$).** Let $Z\sim\mathcal N(0,1)$ and $A := \mathrm{ReLU}(Z) = \max(0,Z)$. Then
$$
\mathbb{E}[A] = \frac{1}{\sqrt{2\pi}} \approx 0.3989423,\qquad \mathbb{E}[A^2] = \frac12,\qquad \operatorname{Var}(A) = \frac12 - \frac{1}{2\pi} \approx 0.3408451 .
$$
More generally for $Z\sim\mathcal N(0,\sigma^2)$: $\mathbb{E}[\mathrm{ReLU}(Z)] = \sigma/\sqrt{2\pi}$ and $\mathbb{E}[\mathrm{ReLU}(Z)^2] = \sigma^2/2$.

*Proof.* Let $\phi(z) = (2\pi)^{-1/2}e^{-z^2/2}$. Then
$$
\mathbb{E}[A] = \int_0^\infty z\,\phi(z)\,dz = \frac{1}{\sqrt{2\pi}}\int_0^\infty z e^{-z^2/2}dz = \frac{1}{\sqrt{2\pi}}\Big[-e^{-z^2/2}\Big]_0^\infty = \frac{1}{\sqrt{2\pi}} .
$$
For the second moment, use the symmetry of $\phi$: $A^2 = Z^2\mathbf 1\{Z>0\}$, and $\mathbb{E}[Z^2\mathbf 1\{Z>0\}] = \mathbb{E}[Z^2\mathbf 1\{Z<0\}]$ because $z\mapsto z^2\phi(z)$ is even. The two halves sum to $\mathbb{E}[Z^2] = 1$, so each is $1/2$. The variance follows, and the $\sigma$-scaling from $\mathrm{ReLU}(\sigma Z) = \sigma\,\mathrm{ReLU}(Z)$ (positive homogeneity). $\square$

Monte Carlo check with $n = 2\times10^7$ samples (seed 11): $\hat{\mathbb{E}}[A] = 0.399047$ against $0.398942$; $\hat{\mathbb{E}}[A^2] = 0.500273$ against $0.5$. Both agree to within Monte Carlo error.

**That factor of $1/2$ is the entire origin of He's factor of $2$.** ReLU throws away half the second moment, so the weights must supply twice as much to compensate. If you remember nothing else from this section, remember that sentence.

It is also worth pausing on the object $A = \max(0,Z)$ itself. Chapter 1 introduced it as the canonical distribution that is **neither discrete nor continuous**: $P(A=0) = P(Z\le0) = 1/2$ is an atom, while on $(0,\infty)$ it has a density. It has no pmf (it is not supported on a countable set) and no pdf (an atom is incompatible with a Lebesgue density). Chapter 2's Radon–Nikodym machinery handled it by writing $P_A = \frac12\delta_0 + \frac12(\text{half-Gaussian})$, a mixture over two different base measures. Every post-ReLU activation in every network you will ever train has exactly this structure. Whenever someone says "the activations are approximately Gaussian," they are wrong at the most important point of the distribution — the atom at zero holds half the mass, and it is the reason dead ReLU units exist at all.

### 5.7.2 Forward propagation of variance

Set up the network. Layer $l$ has $n_l$ units. Write
$$
z^{(l+1)} = W^{(l)}a^{(l)},\qquad a^{(l+1)} = \phi\big(z^{(l+1)}\big),
$$
with $W^{(l)}\in\mathbb{R}^{n_{l+1}\times n_l}$, and take biases zero at initialization (the standard choice). Assume at initialization:

**(A1)** the entries $W^{(l)}_{ij}$ are i.i.d. with mean $0$ and variance $v_l$, independent across layers;
**(A2)** the weights $W^{(l)}$ are independent of the activations $a^{(l)}$;
**(A3)** the activations $a^{(l)}_j$, $j = 1,\dots,n_l$, are i.i.d. across $j$ with common second moment $\mathbb{E}[(a^{(l)})^2]$;
**(A4)** $z^{(l)}$ is approximately Gaussian and symmetric about $0$, which is what makes Lemma 5.33 applicable.

**Theorem 5.34 (forward variance recursion and the He rule).** Under (A1)–(A4), for each unit $i$,
$$
\mathbb{E}\big[(z_i^{(l+1)})^2\big] = n_l\, v_l\, \mathbb{E}\big[(a^{(l)})^2\big].
$$
If $\phi = \mathrm{ReLU}$ then $\mathbb{E}[(a^{(l)})^2] = \frac12\mathbb{E}[(z^{(l)})^2]$, so
$$
\mathbb{E}\big[(z^{(l+1)})^2\big] = \frac{n_lv_l}{2}\,\mathbb{E}\big[(z^{(l)})^2\big],
$$
and the second moment is preserved across depth **iff**
$$
\boxed{\ v_l = \operatorname{Var}(w) = \frac{2}{n_l} = \frac{2}{n_{\text{in}}}\ }\qquad\text{(He initialization).}
$$
If instead $\phi$ is the identity (or any activation with $\mathbb{E}[\phi(z)^2] = \mathbb{E}[z^2]$), preservation requires $v_l = 1/n_l$.

*Proof.* Write $z^{(l+1)}_i = \sum_{j=1}^{n_l}W^{(l)}_{ij}a^{(l)}_j$. Squaring and taking expectations,
$$
\mathbb{E}\big[(z^{(l+1)}_i)^2\big] = \sum_{j}\sum_{k}\mathbb{E}\big[W_{ij}W_{ik}a_ja_k\big] = \sum_j\sum_k \mathbb{E}[W_{ij}W_{ik}]\,\mathbb{E}[a_ja_k],
$$
using (A2) to factor the weights from the activations. By (A1) the weights are i.i.d. with mean $0$, so $\mathbb{E}[W_{ij}W_{ik}] = v_l\delta_{jk}$; only diagonal terms survive:
$$
\mathbb{E}\big[(z_i^{(l+1)})^2\big] = v_l\sum_{j=1}^{n_l}\mathbb{E}[a_j^2] = n_lv_l\,\mathbb{E}\big[(a^{(l)})^2\big],
$$
using (A3). Note we did *not* need the activations to be independent of each other — only that $\mathbb{E}[W_{ij}W_{ik}] = 0$ for $j\ne k$, which kills all cross terms regardless. For the ReLU step, (A4) says $z^{(l)}\approx\mathcal N(0,s^2)$ with $s^2 = \mathbb{E}[(z^{(l)})^2]$, so Lemma 5.33 gives $\mathbb{E}[\mathrm{ReLU}(z^{(l)})^2] = s^2/2$. Combining, $\mathbb{E}[(z^{(l+1)})^2] = \frac{n_lv_l}{2}\mathbb{E}[(z^{(l)})^2]$; the multiplier is $1$ iff $v_l = 2/n_l$. $\square$

Unrolling the recursion over $L$ layers gives the amplification factor
$$
\frac{\mathbb{E}[(z^{(L)})^2]}{\mathbb{E}[(z^{(0)})^2]} = \prod_{l=0}^{L-1}\frac{n_lv_l}{2} = \Big(\frac{g}{2}\Big)^{L}\ \text{ if } v_l = g/n_l\ \text{ for all }l.
$$
This is a geometric progression in $L$, which is why the failure is so violent: at $L=20$, $g=1$ gives $2^{-20} \approx 9.54\times10^{-7}$ and $g=4$ gives $2^{20} \approx 1.05\times10^6$. There is no "close enough."

> **Worked Example 5.16 (a 20-layer network at three initialization scales — the lab, run).**
> Network: 20 layers, width $n = 512$ throughout, ReLU activations, zero biases, weights $W_{ij}\sim\mathcal N(0, g/n)$. Input $a^{(0)}$ has i.i.d. standard-normal entries, batch $1024$. To measure an *expectation* rather than one draw's fluctuation, average $\mathbb{E}[(z^{(l)})^2]$ over $30$ independently sampled networks (seeds $1000$–$1029$).
>
> | layer $l$ | $g=1$ | $g=2$ (He) | $g=4$ |
> |---|---|---|---|
> | 0 (input) | 0.9997 | 0.9997 | 0.9997 |
> | 1 | 1.0000 | 1.9999 | 3.9999 |
> | 5 | 0.0633 | 2.0251 | 64.80 |
> | 10 | 0.001979 | 2.026 | 2075 |
> | 15 | 0.0000619 | 2.027 | 66434 |
> | 20 | 0.00000186 | 1.950 | 2044600 |
>
> Ratios of layer 20 to layer 0: $1.86\times10^{-6}$ for $g=1$ (theory $2^{-20} = 9.54\times10^{-7}$), $1.95$ for $g=2$ (theory $1$, but note the layer-1 value is already $2.0$ because $z^{(1)}$ has not yet been through a ReLU — comparing layer 20 to *layer 1* gives $0.975$, essentially exactly $1$), and $2.045\times10^6$ for $g=4$ (theory $2^{20} = 1.05\times10^6$).
>
> The $g=1$ and $g=4$ ratios each overshoot the theory by about a factor of $2$, and that factor is not noise — it is the same layer-0-versus-layer-1 offset: the input has not passed through a ReLU, so the very first multiplication contributes $g$ rather than $g/2$. Correcting for it, $g=1$ predicts $1\cdot 2^{-19} = 1.91\times10^{-6}$ (observed $1.86\times10^{-6}$) and $g=4$ predicts $4\cdot 2^{19} = 2.10\times10^6$ (observed $2.04\times10^6$). Agreement to within a few percent across six orders of magnitude in each direction.
>
> One more observation from the raw run. A **single** $g=2$ network (rather than an average of 30) drifts: one draw with seed 2024, batch 4096, gave $\mathbb{E}[(z^{(20)})^2] = 3.54$ instead of $2.0$. The recursion is exact in expectation but the realized product $\prod_l(\text{empirical ratio}_l)$ performs a multiplicative random walk whose log-variance grows like $L/n$. That is why very narrow networks are hard to initialize well even with the correct rule, and why width helps stability independently of capacity.

<figure>
<svg viewBox="0 0 740 400" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Activation variance versus depth for three initialization scales">
  <g stroke="currentColor" stroke-width="1" opacity="0.5">
    <line x1="90" y1="60" x2="90" y2="300"/><line x1="90" y1="300" x2="660" y2="300"/>
  </g>
  <g stroke="currentColor" stroke-width="0.5" opacity="0.2">
    <line x1="90" y1="242.4" x2="660" y2="242.4"/><line x1="90" y1="184.8" x2="660" y2="184.8"/>
    <line x1="90" y1="127.2" x2="660" y2="127.2"/><line x1="90" y1="69.6" x2="660" y2="69.6"/>
  </g>
  <g fill="currentColor" font-size="11" font-family="sans-serif" text-anchor="end">
    <text x="84" y="304">10⁻⁶</text><text x="84" y="246">10⁻³</text><text x="84" y="189">10⁰</text>
    <text x="84" y="131">10³</text><text x="84" y="73">10⁶</text>
  </g>
  <g fill="currentColor" font-size="11" font-family="sans-serif" text-anchor="middle">
    <text x="90" y="318">0</text><text x="230" y="318">5</text><text x="370" y="318">10</text><text x="510" y="318">15</text><text x="650" y="318">20</text>
    <text x="375" y="338">layer index l</text>
  </g>
  <text x="30" y="45" fill="currentColor" font-size="11" font-family="sans-serif">E[(z⁽ˡ⁾)²]</text>
  <polyline points="90.0,184.8 118.0,184.8 146.0,190.6 174.0,196.4 202.0,202.1 230.0,207.8 258.0,213.6 286.0,219.4 314.0,225.2 342.0,231.0 370.0,236.7 398.0,242.5 426.0,248.6 454.0,254.1 482.0,259.9 510.0,265.6 538.0,271.6 566.0,277.3 594.0,283.2 622.0,288.9 650.0,294.8" fill="none" stroke="#3b82f6" stroke-width="2.4"/>
  <polyline points="90.0,184.8 118.0,179.0 146.0,179.0 174.0,179.0 202.0,179.0 230.0,178.9 258.0,178.9 286.0,178.9 314.0,178.9 342.0,179.0 370.0,178.9 398.0,179.0 426.0,179.3 454.0,179.0 482.0,178.9 510.0,178.9 538.0,179.1 566.0,179.0 594.0,179.2 622.0,179.1 650.0,179.2" fill="none" stroke="#10b981" stroke-width="3"/>
  <polyline points="90.0,184.8 118.0,173.2 146.0,167.5 174.0,161.7 202.0,155.8 230.0,150.0 258.0,144.2 286.0,138.5 314.0,132.7 342.0,127.0 370.0,121.1 398.0,115.4 426.0,109.9 454.0,103.8 482.0,98.0 510.0,92.2 538.0,86.6 566.0,80.8 594.0,75.2 622.0,69.3 650.0,63.6" fill="none" stroke="#ef4444" stroke-width="2.4"/>
  <g font-size="12" font-family="sans-serif">
    <text x="664" y="298" fill="#3b82f6">g = 1</text>
    <text x="664" y="183" fill="#10b981">g = 2</text>
    <text x="664" y="67" fill="#ef4444">g = 4</text>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="20" y="362">Figure 5.5 — Activation second moment vs. depth, 20-layer ReLU net of width 512, weights ~ N(0, g/n), averaged over</text>
    <text x="20" y="380">30 networks (seeds 1000–1029). Vertical axis is log scale spanning 12 decades. g = 2 (He) is flat at ≈ 2.0; g = 1 decays</text>
    <text x="20" y="398">by a factor 2 per layer to 1.9e−6; g = 4 grows by 2 per layer to 2.0e+6. The knife's edge is exactly Var(w) = 2/n_in.</text>
  </g>
</svg>
</figure>

### 5.7.3 The backward pass and Xavier's compromise

Forward stability is only half the requirement. Gradients also propagate, and they propagate through the *transpose* of the weight matrix, which changes which dimension appears in the variance.

**Proposition 5.35 (backward recursion; Xavier/Glorot).** Let $\delta^{(l)} := \partial\mathcal L/\partial z^{(l)}$. For a ReLU network, backpropagation gives
$$
\delta^{(l)} = \big(W^{(l)}\big)^\top\delta^{(l+1)}\odot\phi'\big(z^{(l)}\big),
$$
and under assumptions analogous to (A1)–(A4) (in particular, treating the backward weights as independent of the forward ones — see the caveat below),
$$
\mathbb{E}\big[(\delta^{(l)})^2\big] = \frac{n_{l+1}v_l}{2}\,\mathbb{E}\big[(\delta^{(l+1)})^2\big],
$$
so gradient magnitudes are preserved iff $v_l = 2/n_{l+1} = 2/n_{\text{out}}$. Since $2/n_{\text{in}}$ and $2/n_{\text{out}}$ cannot both hold when $n_{\text{in}}\ne n_{\text{out}}$, the **Xavier/Glorot compromise** takes the harmonic-style average
$$
\boxed{\ \operatorname{Var}(w) = \frac{2}{n_{\text{in}}+n_{\text{out}}}\ }
$$
Be careful about what this compromise is compromising *between*. The value $2/(n_{\text{in}}+n_{\text{out}})$ is exactly the **harmonic mean of $1/n_{\text{in}}$ and $1/n_{\text{out}}$** — the forward and backward targets for a *linear* network. Glorot and Bengio derived it for tanh units operating near their linear regime, which is why the classical Xavier constant contains no factor of 2. The ReLU analogue, harmonic-averaging $2/n_{\text{in}}$ and $2/n_{\text{out}}$, is $4/(n_{\text{in}}+n_{\text{out}})$ — twice Xavier. Deep learning libraries reflect exactly this: `xavier_normal_` uses $2/(n_{\text{in}}+n_{\text{out}})$, while `kaiming_normal_` uses $2/n_{\text{in}}$ or $2/n_{\text{out}}$ according to a `mode` flag, with an extra `gain` for non-ReLU activations. Choosing Xavier for a ReLU network therefore starts you a factor of $\sqrt2$ too small in every layer — survivable at depth 10, fatal at depth 100.

*Proof.* Each coordinate is $\delta^{(l)}_j = \phi'(z_j^{(l)})\sum_{i=1}^{n_{l+1}}W^{(l)}_{ij}\delta^{(l+1)}_i$. Squaring and taking expectations, the same orthogonality argument as in Theorem 5.34 kills cross terms, leaving $n_{l+1}v_l\,\mathbb{E}[(\delta^{(l+1)})^2]$ times $\mathbb{E}[\phi'(z^{(l)})^2]$. For ReLU, $\phi'(z) = \mathbf 1\{z>0\}$, so $\phi'(z)^2 = \mathbf 1\{z>0\}$ and, for $z$ symmetric about $0$, $\mathbb{E}[\phi'(z)^2] = P(z>0) = 1/2$. That is the second appearance of the same factor $1/2$, now from the *derivative* rather than the value. Hence the recursion, and the condition $n_{l+1}v_l/2 = 1$. $\square$

Look at where the $1/2$ came from each time. Forward: $\mathbb{E}[\mathrm{ReLU}(Z)^2] = \frac12\mathbb{E}[Z^2]$ because ReLU zeroes half the mass. Backward: $\mathbb{E}[\mathbf 1\{Z>0\}] = \frac12$ because the gradient is blocked on half the units. Same constant, two different mechanisms, and both point at $\operatorname{Var}(w) = 2/n$.

> **Worked Example 5.17 (why tanh and sigmoid need different constants — and what still breaks).**
> Run the same 20-layer, width-512 experiment with $\phi = \tanh$ (seed 5, batch 4096, single draw):
>
> | layer | $\tanh$, $g=1$ | $\tanh$, $g=2$ | sigmoid, $g=1$ |
> |---|---|---|---|
> | 1 | 0.9992 | 1.9985 | 0.9992 |
> | 2 | 0.3938 | 1.0388 | 0.3206 |
> | 5 | 0.1273 | 0.6638 | 0.2675 |
> | 10 | 0.0588 | 0.6248 | 0.2651 |
> | 20 | 0.0277 | 0.6185 | 0.2306 |
>
> Three lessons. **(i)** Xavier's $g=1$ rule, derived for the linear regime, still lets $\tanh$ activations decay — by layer 20 the second moment is $0.0277$, a $36\times$ shrinkage. The reason is that $\tanh$ is contractive ($|\tanh(z)|<|z|$), so the linearization is optimistic; the correct constant for $\tanh$ in the usual convention is a "gain" of about $5/3$, and $g=2$ overshoots slightly, settling at $0.62$ rather than $1$.
> **(ii)** Sigmoid is worse in a way the variance table hides. It stabilizes at $\approx0.23$, which looks harmless — but sigmoid outputs are centred at $0.5$, not $0$, so assumption (A4)'s symmetry fails immediately and the *mean* propagates as a systematic bias through every layer. Worse still, $\sigma'(z)\le1/4$, so the backward recursion picks up a factor of at most $1/16$ per layer in the saturated regime; that is the classical vanishing-gradient problem, and no choice of $\operatorname{Var}(w)$ fixes it, because the problem is the derivative's ceiling, not the weight scale.
> **(iii)** These runs are single draws, so the layer-to-layer wiggles are finite-width noise, not structure.

> **Common Confusion 5.7.**
> *"He initialization is proved, so activation variance stays constant throughout training."*
>
> Every assumption in Theorem 5.34 is an *initialization-time* assumption, and training breaks all of them. (A1) fails because trained weights are neither i.i.d. nor mean-zero. (A2) fails immediately and catastrophically after the first update: the weights become a function of the data that flowed through them, so weights and activations are correlated. (A3) fails because features specialize. (A4) fails because the pre-activation distribution acquires skew and heavy tails.
>
> What the theorem actually delivers is a good *starting point* — enough that the first few gradient steps are well-scaled. Keeping the scale healthy thereafter is what normalization layers are for: batch norm, layer norm and RMS norm all *enforce* by construction what He initialization only arranges at step 0. This is a large part of why residual networks with normalization can be trained at depths where initialization alone is hopeless, and it explains a fact practitioners notice: with layer norm everywhere, the exact initialization constant matters much less. Papers that report initialization ablations on normalized architectures often find nearly flat results, and this is why. Conversely, in architectures without normalization — some implicit models, some very deep RNNs — the constant matters enormously and people rediscover that fact painfully.
>
> One further honest caveat about the derivation itself: the backward pass analysis treats $W^\top$ appearing in the gradient as independent of the $W$ used in the forward pass, which is false — it is the same matrix. The approximation is standard, is exact in the infinite-width limit, and is empirically excellent at width 512, but it is an approximation, and the original papers are explicit about this even though summaries of them usually are not.

> **Self-check 5.7.**
> 1. Where exactly does the $2$ in He initialization come from? Name the expectation.
> 2. A layer has $n_{\text{in}} = 1024$, $n_{\text{out}} = 256$. Give the He and Xavier standard deviations.
> 3. For a purely linear network (no activation), what is the correct $\operatorname{Var}(w)$, and what does the network compute?
> 4. Why does the failure mode scale like $2^{\pm L}$ rather than linearly in $L$?

---

## 5.8 Gaussian processes: §5.4 with an infinite index set

### 5.8.1 What a GP is, and what it is not

The slogan is "a distribution over functions." That is evocative but slightly misleading, because the object we actually construct is not a measure on some exotic function space — it is a *consistent family of finite-dimensional Gaussians*, one for each finite set of input points. You never touch infinitely many coordinates at once. You only ever ask: "at these $n$ inputs, what is the joint distribution of the $n$ outputs?" And the answer is always a plain $n$-dimensional Gaussian, to which §5.4 applies without modification.

**Definition 5.36 (Gaussian process).** Let $\mathcal X$ be any index set. A **Gaussian process** on $\mathcal X$ with mean function $m:\mathcal X\to\mathbb{R}$ and covariance function (kernel) $k:\mathcal X\times\mathcal X\to\mathbb{R}$ is a collection of random variables $\{f(x)\}_{x\in\mathcal X}$ such that for every finite subset $\{x_1,\dots,x_n\}\subseteq\mathcal X$,
$$
\big(f(x_1),\dots,f(x_n)\big)^\top \ \sim\ \mathcal N\big(\mathbf m,\ K\big),\qquad \mathbf m_i = m(x_i),\ \ K_{ij} = k(x_i,x_j).
$$
We write $f\sim\mathcal{GP}(m,k)$.

For this to be coherent, two things must hold. Each $K$ must be a valid covariance matrix — so $k$ must be symmetric and **positive semidefinite**, meaning $\sum_{i,j}c_ic_jk(x_i,x_j)\ge0$ for all finite $\{x_i\}$ and coefficients $c$. And the finite-dimensional families must be *consistent*: the distribution assigned to $(f(x_1),f(x_2))$ must have as its first marginal the distribution assigned to $f(x_1)$. Consistency is automatic here, because Proposition 5.20 says Gaussian marginals are obtained by deleting rows and columns of $K$ — which is exactly what restricting the index set does. That consistency is what allows a genuine stochastic process to exist.

**Theorem 5.37 (Kolmogorov extension theorem; cited, not proved).** Given a family of finite-dimensional distributions on $\mathbb{R}^{S}$ for each finite $S\subseteq\mathcal X$ that is consistent under marginalization and permutation, there exists a probability space and a stochastic process $\{f(x)\}_{x\in\mathcal X}$ whose finite-dimensional distributions are exactly that family.

**Proposition 5.38 (existence of GPs).** For any $m:\mathcal X\to\mathbb{R}$ and any symmetric PSD kernel $k$, a Gaussian process $\mathcal{GP}(m,k)$ exists.

*Proof.* For each finite $S$, define the candidate law as $\mathcal N(\mathbf m_S, K_S)$, which exists by Definition 5.15 since $K_S\succeq0$. Permutation consistency is clear (permuting coordinates permutes $\mathbf m$ and $K$). Marginal consistency is Proposition 5.20. Apply Theorem 5.37. $\square$

A caution that is skipped in most tutorials. Kolmogorov's theorem gives you a process on the product $\sigma$-algebra, and the product $\sigma$-algebra on $\mathbb{R}^{\mathcal X}$ (for uncountable $\mathcal X$) contains *only* sets determined by countably many coordinates. So "$f$ is continuous" is **not** a measurable event in that space — this is Chapter 1's "countability is load-bearing" theme in its most consequential form. Statements about sample-path continuity or differentiability require an extra construction (choosing a continuous *modification*, guaranteed under mild conditions on $k$ by the Kolmogorov–Chentsov theorem). Papers say "samples from an RBF-kernel GP are smooth" without comment; there is real mathematics behind that sentence, and §5.9 records where to find it.

### 5.8.2 Kernels are assumptions

The kernel is the model. Everything the GP believes about the function before seeing data is encoded in $k$.

| Kernel | Formula | What it assumes |
|---|---|---|
| Squared exponential (RBF) | $\sigma_f^2\exp\big(-\|x-x'\|^2/(2\ell^2)\big)$ | Infinitely differentiable sample paths; extremely strong smoothness |
| Matérn $\nu$ | $\sigma_f^2\frac{2^{1-\nu}}{\Gamma(\nu)}\big(\frac{\sqrt{2\nu}r}{\ell}\big)^\nu K_\nu\big(\frac{\sqrt{2\nu}r}{\ell}\big)$, $r=\|x-x'\|$ | Sample paths are $\lceil\nu\rceil-1$ times differentiable; $\nu=1/2$ gives the (nowhere differentiable) Ornstein–Uhlenbeck process, $\nu\to\infty$ recovers RBF |
| Linear | $\sigma_0^2 + x^\top x'$ | The function *is* linear; GP regression then coincides exactly with Bayesian linear regression |
| Periodic | $\sigma_f^2\exp\big(-2\sin^2(\pi\|x-x'\|/p)/\ell^2\big)$ | Exact periodicity with period $p$ |

Two facts make kernel engineering practical: sums and products of PSD kernels are PSD (so you can build "smooth trend $\times$ periodic $+$ noise"), and $k(x,x') = \varphi(x)^\top\varphi(x')$ is PSD for any feature map $\varphi$. The last row of the table deserves emphasis: with the linear kernel, GP regression is *exactly* Bayesian linear regression (§5.6.1) — the kernel trick has bought nothing, because the feature map is the identity. Everything the GP adds over Proposition 5.30 comes from choosing a $\varphi$ you would not want to write down.

**Why $\nu$ controls smoothness.** The claim in the table is not magic, and knowing where it comes from is what lets you choose a kernel rather than copy one. For a stationary kernel $k(x,x') = k(r)$ with $r = \|x-x'\|$, the smoothness of the sample paths is governed entirely by the behaviour of $k$ *at the origin*. The reason is a one-line computation: for a zero-mean GP,
$$
\mathbb{E}\big[(f(x+h)-f(x))^2\big] = 2\big(k(0)-k(h)\big),
$$
so how fast $k(h)$ peels away from $k(0)$ is precisely how fast $f$ can wiggle. Expand each kernel near $r=0$. The RBF gives $\sigma_f^2(1 - r^2/2\ell^2 + r^4/8\ell^4 - \cdots)$: analytic, with every odd power absent, so the increments are $O(r^2)$ and $f$ is infinitely mean-square differentiable. The Matérn-$\nu$ expansion instead contains a term in $r^{2\nu}$ — for $\nu=1/2$ that is $|r|$ itself, a kink, giving increments of order $r$ and hence sample paths that are continuous but nowhere differentiable (the Ornstein–Uhlenbeck process). The general rule is that a stationary GP is $m$ times mean-square differentiable exactly when $k$ is $2m$ times differentiable at $0$, which for Matérn happens when $m<\nu$ — hence $\lceil\nu\rceil-1$.

The same fact is visible in the frequency domain, which is often the more useful view: Bochner's theorem says a stationary $k$ is the Fourier transform of a positive spectral density $S(\omega)$, and $S$ tells you which frequencies the prior admits. The RBF's spectral density is itself Gaussian, decaying like $e^{-\ell^2\omega^2/2}$ — it assigns essentially zero prior mass to high frequencies, which is a very strong claim. The Matérn's decays only polynomially, like $(1+\ell^2\omega^2)^{-(\nu+d/2)}$, leaving room for roughness.

So the much-repeated advice "the RBF kernel is a safe default" is dubious: it asserts infinite differentiability, which almost no real function has, and its super-exponentially thin spectral tail is a large part of why RBF-GPs are overconfident when extrapolating. Matérn-$5/2$ ($m=2$ derivatives) is a more honest default and is what most Bayesian-optimization libraries actually use.

### 5.8.3 GP regression is Gaussian conditioning. That is all.

**Theorem 5.39 (GP regression posterior).** Let $f\sim\mathcal{GP}(0,k)$ and observe $y_i = f(x_i)+\varepsilon_i$, $\varepsilon_i\stackrel{iid}\sim\mathcal N(0,\sigma_n^2)$, at training inputs $X = (x_1,\dots,x_n)$. Let $X_*$ be test inputs, $K = k(X,X)$, $K_* = k(X,X_*)$, $K_{**} = k(X_*,X_*)$. Then
$$
f(X_*)\mid y \ \sim\ \mathcal N\big(\bar f_*,\ \operatorname{Cov}(f_*)\big),
$$
$$
\bar f_* = K_*^\top\big(K+\sigma_n^2I\big)^{-1}y,\qquad
\operatorname{Cov}(f_*) = K_{**} - K_*^\top\big(K+\sigma_n^2I\big)^{-1}K_* .
$$

*Proof.* The vector $(y, f(X_*))$ is jointly Gaussian: $f(X)$ and $f(X_*)$ are jointly Gaussian by Definition 5.36 applied to the union of the index sets, and $y = f(X)+\varepsilon$ adds independent Gaussian noise, so the stack is an affine image of a Gaussian (Corollary 5.18, Proposition 5.26). Its covariance blocks are $\operatorname{Cov}(y) = K+\sigma_n^2I$, $\operatorname{Cov}(f_*) = K_{**}$, $\operatorname{Cov}(f_*,y) = K_*^\top$. Now apply Theorem 5.22 with block $1 = f(X_*)$ and block $2 = y$. $\square$

That proof is four lines, and it is the *entire* content of GP regression. There is no separate GP theory. Compare the formula to Theorem 5.22's $\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)$ and $\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ — it is a literal substitution. Every property inherits accordingly: the posterior mean is *linear in $y$*; the posterior covariance *does not depend on $y$*; and $\operatorname{Cov}(f_*)\preceq K_{**}$, so observing data never increases uncertainty (Lemma 5.21's $S\preceq\Sigma_{11}$).

The posterior mean can be rewritten $\bar f_* = \sum_{i=1}^n\alpha_i k(x_i,x_*)$ with $\alpha = (K+\sigma_n^2I)^{-1}y$: **the posterior mean is a linear combination of kernel functions centred at the training points.** This is the representer theorem in Gaussian clothing (the equivalence is recorded in §5.9), and it is why GP regression and kernel ridge regression produce identical point predictions. GPs add the variance; kernel ridge does not.

**Proposition 5.40 (log marginal likelihood).** With the setup above and $K_y := K+\sigma_n^2I$,
$$
\log p(y\mid X,\theta) = -\tfrac12 y^\top K_y^{-1}y \;-\; \tfrac12\log\det K_y \;-\; \tfrac n2\log2\pi .
$$

*Proof.* By Theorem 5.28(1) / Proposition 5.20, $y\sim\mathcal N(0,K_y)$; write down its log-density from Theorem 5.17(c). $\square$

The three terms have names worth knowing: $-\frac12y^\top K_y^{-1}y$ is *data fit*, $-\frac12\log\det K_y$ is a *complexity penalty* (via Proposition 5.13, $\sqrt{\det K_y}$ is the volume of the prior's uncertainty ellipsoid — a model that spreads its prior mass widely is penalized), and the last term is a constant. Maximizing this over kernel hyperparameters is "empirical Bayes" or "type-II maximum likelihood," and the automatic fit-versus-complexity tradeoff is the standard argument for why GPs resist overfitting their hyperparameters. It is a good argument and it is not a guarantee — the marginal likelihood is non-convex, has local optima, and can be badly misled under model misspecification.

> **Worked Example 5.18 (GP regression from scratch, with the length-scale swept).**
> Target function $f(x) = \sin x + 0.3x$ on $[-1,10]$. Six training inputs $x = (0.5, 1.5, 3.0, 4.5, 6.0, 7.5)$, observations $y = f(x)+0.1\epsilon$ with $\epsilon$ standard normal (seed 3), giving
> $$y = (0.834,\ 1.192,\ 1.083,\ 0.316,\ 1.475,\ 3.166).$$
> RBF kernel with $\sigma_f = 1$, noise $\sigma_n = 0.1$. Computing via Cholesky of $K+\sigma_n^2I$:
>
> | length-scale $\ell$ | log marginal likelihood | posterior sd at $x=2.25$ (between data) | posterior sd at $x=9.0$ (extrapolating) |
> |---|---|---|---|
> | 0.3 | −13.2579 | 0.9975 | 1.0000 |
> | 1.0 | −11.1702 | 0.3099 | 0.9407 |
> | 3.0 | −19.5192 | 0.0791 | 0.3178 |
>
> Read the table carefully; three separate phenomena are visible.
> **Too short ($\ell=0.3$).** Points $1.5$ units apart have prior correlation $e^{-1.5^2/(2\cdot0.09)} = e^{-12.5} \approx 4\times10^{-6}$: effectively zero. Each observation tells you about its own immediate neighbourhood and nothing else, so at $x=2.25$ the posterior sd is $0.9975$ — barely below the prior sd of $1$. The GP has memorized six points and learned no function.
> **Too long ($\ell=3$).** The sd at $x=2.25$ collapses to $0.0791$ and even at $x=9$, well beyond the data, it is only $0.3178$. The model is confident everywhere, including where it has no right to be. Its log marginal likelihood ($-19.52$) is the worst of the three: it cannot simultaneously fit the wiggle at $x\in[3,6]$ and stay smooth, so data fit suffers.
> **About right ($\ell=1$).** Best log marginal likelihood ($-11.17$), sd $0.31$ between data points and $0.94$ when extrapolating — uncertainty that grows honestly away from the data. A finer sweep of $\ell$ over $[0.2,5]$ on a 97-point grid maximizes the log marginal likelihood at $\ell = 1.75$ with value $-9.5328$.
>
> **A hand-checkable miniature.** Two training points at $x = 0$ and $x = 2$ with $y = (0,1)$, $\ell = 1$, $\sigma_n = 0.1$. Then $k(0,2) = e^{-2} = 0.135335$, so
> $$K_y = \begin{pmatrix}1.01 & 0.135335\\ 0.135335 & 1.01\end{pmatrix},\qquad k_* = \big(k(0,1), k(2,1)\big)^\top = (0.606531,\ 0.606531)^\top .$$
> Predicting at $x_*=1$: $\alpha = K_y^{-1}y = (-0.135094,\ 1.008201)^\top$, so
> $$\bar f_* = k_*^\top\alpha = 0.606531(-0.135094+1.008201) = 0.529566,$$
> $$\operatorname{Var}(f_*) = 1 - k_*^\top K_y^{-1}k_* = 0.357604,\quad \text{sd} = 0.598 .$$
> The prediction $0.5296$ sits just above the midpoint of $0$ and $1$ (it is pulled toward $0$ by the zero prior mean, and pushed by the correlation structure), and the sd $0.598$ is well below the prior sd of $1$ but far from zero — the test point is one length-scale from each observation.

<figure>
<svg viewBox="0 0 740 400" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gaussian process prior samples and posterior with uncertainty band">
  <g stroke="currentColor" stroke-width="1" opacity="0.45">
    <line x1="60" y1="204" x2="345" y2="204"/><line x1="85.5" y1="60" x2="85.5" y2="300"/>
    <line x1="400" y1="204" x2="685" y2="204"/><line x1="425.5" y1="60" x2="425.5" y2="300"/>
  </g>
  <polyline points="60.0,229.7 70.2,227.7 80.4,231.0 90.5,230.2 100.7,220.1 110.9,205.7 121.1,197.3 131.3,200.5 141.5,208.6 151.6,207.4 161.8,191.6 172.0,170.9 182.2,158.1 192.4,157.5 202.5,164.5 212.7,172.7 222.9,180.2 233.1,190.0 243.3,206.4 253.5,229.2 263.6,249.8 273.8,256.7 284.0,246.7 294.2,228.5 304.4,215.1 314.5,213.1 324.7,219.0 334.9,226.0 340.0,228.7" fill="none" stroke="#3b82f6" stroke-width="1.7" opacity="0.85"/>
  <polyline points="60.0,172.9 70.2,187.3 80.4,203.0 90.5,214.9 100.7,219.2 110.9,215.9 121.1,208.1 131.3,201.5 141.5,200.6 151.6,204.2 161.8,209.9 172.0,218.3 182.2,229.7 192.4,239.2 202.5,240.3 212.7,231.4 222.9,216.8 233.1,202.6 243.3,193.0 253.5,187.4 263.6,180.0 273.8,166.6 284.0,151.3 294.2,142.4 304.4,141.9 314.5,145.9 324.7,150.4 334.9,153.4 340.0,154.3" fill="none" stroke="#10b981" stroke-width="1.7" opacity="0.85"/>
  <polyline points="60.0,174.0 70.2,192.7 80.4,204.5 90.5,213.5 100.7,221.3 110.9,227.2 121.1,229.2 131.3,227.9 141.5,227.2 151.6,229.2 161.8,229.5 172.0,222.8 182.2,208.8 192.4,190.8 202.5,173.8 212.7,164.7 222.9,167.5 233.1,178.7 243.3,191.6 253.5,201.7 263.6,205.8 273.8,205.0 284.0,199.6 294.2,193.1 304.4,186.2 314.5,176.5 324.7,166.6 334.9,171.3 340.0,175.7" fill="none" stroke="#f59e0b" stroke-width="1.7" opacity="0.85"/>
  <polygon points="400.0,139.8 410.2,144.3 420.4,154.0 430.5,166.7 440.7,168.9 450.9,159.2 461.1,159.9 471.3,152.4 481.5,144.3 491.6,149.6 501.8,163.3 512.0,160.0 522.2,163.2 532.4,177.7 542.5,186.2 552.7,171.3 562.9,160.0 573.1,155.4 583.3,136.2 593.5,104.6 603.6,89.0 613.8,95.1 624.0,91.9 634.2,89.9 644.4,99.0 654.5,112.9 664.7,125.3 674.9,133.4 680.0,135.9 680.0,263.8 674.9,261.1 664.7,251.4 654.5,233.3 644.4,204.5 634.2,166.8 624.0,128.2 613.8,111.3 603.6,130.7 593.5,148.4 583.3,157.8 573.1,176.9 562.9,202.1 552.7,209.8 542.5,201.6 532.4,205.3 522.2,206.3 512.0,193.1 501.8,176.0 491.6,181.4 481.5,183.8 471.3,176.9 461.1,173.6 450.9,181.8 440.7,182.8 430.5,198.1 420.4,224.1 410.2,245.3 400.0,258.2" fill="#3b82f6" opacity="0.18" stroke="none"/>
  <polyline points="400.0,240.5 420.4,212.3 440.7,180.2 461.1,159.0 481.5,157.0 501.8,170.7 522.2,187.1 542.5,191.6 562.9,176.9 583.3,147.1 603.6,115.8 624.0,97.2 644.4,97.9 664.7,113.0 680.0,125.4" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="5 4" opacity="0.8"/>
  <polyline points="400.0,199.0 410.2,194.8 420.4,189.1 430.5,182.4 440.7,175.9 450.9,170.5 461.1,166.7 471.3,164.6 481.5,164.1 491.6,165.5 501.8,169.6 512.0,176.6 522.2,184.8 532.4,191.5 542.5,193.9 552.7,190.5 562.9,181.1 573.1,166.2 583.3,147.0 593.5,126.5 603.6,109.8 613.8,103.2 624.0,110.0 634.2,128.4 644.4,151.7 654.5,173.1 664.7,188.4 674.9,197.3 680.0,199.8" fill="none" stroke="#3b82f6" stroke-width="2.6"/>
  <g fill="#ef4444">
    <circle cx="438.2" cy="177.3" r="4"/><circle cx="463.6" cy="165.9" r="4"/><circle cx="501.8" cy="169.3" r="4"/>
    <circle cx="540.0" cy="193.9" r="4"/><circle cx="578.2" cy="156.8" r="4"/><circle cx="616.4" cy="102.7" r="4"/>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif" text-anchor="middle">
    <text x="200" y="326">Prior: 3 samples from GP(0, RBF, ℓ = 1)</text>
    <text x="540" y="326">Posterior given 6 noisy observations</text>
  </g>
  <g fill="currentColor" font-size="12" font-family="sans-serif">
    <text x="20" y="362">Figure 5.6 — Left: three functions drawn from the GP prior; each is a sample of a 56-dimensional Gaussian whose</text>
    <text x="20" y="380">covariance is the RBF kernel. Right: posterior mean (blue) with ±2 sd band (shaded), true function f(x) = sin x + 0.3x</text>
    <text x="20" y="398">(dashed), and the six observations (red). The band pinches at data and widens between and beyond it — that is Theorem 5.22.</text>
  </g>
</svg>
</figure>

### 5.8.4 The cubic wall, and two asymptotic bridges

Every formula in Theorem 5.39 contains $(K+\sigma_n^2I)^{-1}$, an $n\times n$ solve. In practice one computes the Cholesky factor $L$ of $K_y$ once ($n^3/3$ flops, $n^2$ memory) and reuses it for the mean, the variance, and the log marginal likelihood ($\log\det K_y = 2\sum_i\log L_{ii}$). That is $O(n^3)$ time and $O(n^2)$ memory, and it is the binding constraint on GP practice: $n = 10^4$ is routine, $n=10^5$ needs care, $n=10^6$ needs approximation (inducing-point/sparse variational methods reducing to $O(nm^2)$, structured kernels exploiting Kronecker or Toeplitz structure, or conjugate-gradient-based methods that never form the inverse). It is also why Bayesian optimization — which needs perhaps a few hundred function evaluations — is the killer application for exact GPs, while GP regression on ImageNet-scale data is not.

Two results connect GPs to neural networks, both of which I state as cited facts (§5.9).

**NNGP.** Take a fully connected network with i.i.d. random weights initialized as in §5.7 and let every hidden width go to infinity. Each pre-activation at any finite collection of inputs is then a sum of infinitely many i.i.d. contributions, and a central limit theorem (Ch. 4) gives a Gaussian limit; by Theorem 5.17(b) the collection is jointly Gaussian. So an infinitely wide, randomly initialized network *is* a GP, with a kernel computed by a layerwise recursion whose ReLU case involves exactly the arccosine formula descended from Lemma 5.33. Notice the pleasing circularity: §5.7's variance recursion is the diagonal of the NNGP kernel recursion.

**NTK.** Under gradient descent with infinitesimal step size in the infinite-width limit, the network's function evolves as a *linear* model in its features, with a fixed kernel — the neural tangent kernel — so training is kernel regression and the resulting predictor has a closed form. The honest caveats matter: the NTK regime requires a particular parameterization and scaling, it is a *lazy* regime in which features do not move, and it demonstrably fails to capture feature learning, which is what makes finite-width networks work well. NTK is a rigorous and illuminating limit, not a theory of deep learning.

> **Common Confusion 5.8.**
> *"A GP is nonparametric, so it can learn any function — the prior is uninformative."*
>
> The prior is enormously informative; it is just infinite-dimensional. Three concrete ways the kernel constrains you:
> - **Smoothness is fixed in advance.** With an RBF kernel, every posterior sample is infinitely differentiable, always. No amount of data will produce a kink. If the truth has a kink, the GP will oscillate around it (Gibbs-like ringing) and report inappropriate confidence.
> - **The posterior mean lives in a fixed finite-dimensional space.** By the representer form $\bar f_* = \sum_i\alpha_ik(x_i,x_*)$, the posterior mean is confined to the $n$-dimensional span of kernel slices at the training points, regardless of how you tune hyperparameters.
> - **Extrapolation reverts to the prior mean.** With a stationary kernel, $k(x,x_*)\to0$ as $x_*$ moves far from all data, so $\bar f_*\to m(x_*)$ (usually $0$) and $\operatorname{Var}\to k(x_*,x_*)$. A GP does not extrapolate trends; it forgets them. Figure 5.6's right edge shows this beginning to happen.
>
> "Nonparametric" means the number of effective parameters grows with $n$. It does not mean "assumption-free," and treating the kernel as a tuning knob rather than a modeling commitment is one of the most common ways GP results are oversold.

> **Self-check 5.8.**
> 1. Why does GP regression need no theory beyond Theorem 5.22?
> 2. What changes in the posterior if you double every observed $y_i$? What does not change?
> 3. Why is the RBF kernel with a large $\ell$ a *stronger* assumption, not a weaker one?
> 4. You have $n = 50{,}000$ training points. Estimate the memory needed to store $K$ in float64, and say what to do instead.

---

## 5.9 Results used but not proved

Everything in this table is used somewhere in this chapter — including in the exercise solutions — without being proved here. The list is meant to be exhaustive; if you find something else asserted without proof or citation, that is an error, not a deliberate omission.

| Result | Used in | Where to find a proof |
|---|---|---|
| Spectral theorem for real symmetric matrices (Theorem 5.8) | §5.2 throughout; Theorem 5.11; Proposition 5.13 | Horn & Johnson, *Matrix Analysis*, Thm 4.1.5; Axler, *Linear Algebra Done Right*, Ch. 7 |
| Existence/uniqueness of the PSD square root $\Sigma^{1/2}$ and of the Cholesky factor | Definition 5.9; Worked Example 5.4 | Horn & Johnson, Thm 7.2.6 (square root), Cor. 7.2.9 (Cholesky) |
| Uniqueness theorem for characteristic functions (equal $\varphi$ $\Rightarrow$ equal law); Cramér–Wold device | Lemma 5.16; Theorem 5.17(b)$\Rightarrow$(a); Corollary 5.23 | Billingsley, *Probability and Measure*, §26 (Thm 26.2) and §29; Durrett, *PTE*, Thm 3.3.6 and Thm 3.10.5 |
| A linear combination of independent univariate normals is normal | Theorem 5.17(a)$\Rightarrow$(b) | Immediate from Lemma 5.16 in $d=1$; or Durrett, Exercise 3.3.x by convolution |
| Change-of-variables formula with Jacobian for pushforwards under $C^1$ diffeomorphisms | Theorem 5.17(a)$\Rightarrow$(c); Proposition 5.13 | Chapter 3 of this tutorial |
| Fubini–Tonelli; product-measure characterization of independence; densities w.r.t. $\lambda_d$; Radon–Nikodym | Propositions 5.6, 5.24; Lemma 5.16; Lemma 5.27 | Chapter 2 of this tutorial |
| Exponential-family structure and convexity of the log-partition function | Proposition 5.19 | Chapter 3 of this tutorial; Wainwright & Jordan, *Graphical Models, Exponential Families, and Variational Inference*, §3 |
| Woodbury / Sherman–Morrison matrix identity | Theorem 5.28(2) | Golub & Van Loan, *Matrix Computations*, §2.1.4; direct verification by multiplication |
| Cauchy–Schwarz inequality in $L^2(P)$ | Proposition 5.5 | Any measure theory text; e.g. Folland, *Real Analysis*, §6.1 |
| A nonnegative random variable with zero expectation is $0$ a.s. | Proposition 5.2(3) | Chapter 2 of this tutorial (Markov's inequality) |
| $\|\Sigma^{-1/2}(X-\mu)\|^2\sim\chi^2_d$ for $X\sim\mathcal N(\mu,\Sigma)$, $\Sigma\succ0$ | The constant $5.9915$ in Figure 5.1 and §5.2.3 | Definition of $\chi^2_d$ as a sum of $d$ squared standard normals, plus Theorem 5.17; Casella & Berger, *Statistical Inference*, §5.3 |
| Concentration of $\|Z\|$ around $\sqrt d$ for $Z\sim\mathcal N(0,I_d)$ with $O(1)$ fluctuations (the sharp sub-exponential form; the crude Chebyshev form is proved in §5.3) | The soap-bubble discussion at the end of §5.3 | Chapter 4 of this tutorial; Vershynin, *High-Dimensional Probability*, Thm 3.1.1 |
| ZCA optimality: $W = \Sigma^{-1/2}$ uniquely minimizes $\mathbb{E}\|\tilde X-(X-\mu)\|^2$ over whitening transforms | §5.2.1, after Proposition 5.10 | Kessy, Lewin & Strimmer, "Optimal whitening and decorrelation," *The American Statistician* 2018, §3.1; short proof via the polar decomposition |
| GP posterior mean $=$ kernel ridge regression point prediction (representer theorem) | §5.8.3, after Theorem 5.39 | Rasmussen & Williams, §6.2; Schölkopf, Herbrich & Smola, "A generalized representer theorem," COLT 2001 |
| Moore–Penrose pseudoinverse: existence via the spectral decomposition, and the property that $\Sigma^{+}\Sigma$ is the orthogonal projector onto $\operatorname{range}(\Sigma)$ for symmetric $\Sigma\succeq0$ | The singular-$\Sigma_{22}$ remark after Corollary 5.23; solutions to Exercises A8 and A10 | Golub & Van Loan, *Matrix Computations*, §5.5.2; immediate from writing $\Sigma = UDU^\top$ and $\Sigma^{+} = UD^{+}U^\top$ |
| Kolmogorov extension theorem (Theorem 5.37) | Proposition 5.38 | Kallenberg, *Foundations of Modern Probability*, Thm 6.16; Billingsley §36 |
| Kolmogorov–Chentsov continuity criterion (existence of a continuous modification of a GP) | The caution after Proposition 5.38 | Karatzas & Shreve, *Brownian Motion and Stochastic Calculus*, Thm 2.2.8 |
| Positive semidefiniteness of RBF, Matérn, linear and periodic kernels; closure of PSD kernels under sums and products | §5.8.2 | Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Ch. 4; Schoenberg's theorem |
| Central limit theorem for triangular arrays (used informally for the NNGP limit) | §5.8.4 | Chapter 4 of this tutorial; Durrett, §3.4 |
| NNGP correspondence (infinite-width networks are GPs) and the NTK theorem | §5.8.4 | Neal (1996); Lee et al. (2018); Matthews et al. (2018); Jacot, Gabriel & Hongler (2018) |
| Consistency of the PPCA maximum-likelihood solution with the top-$k$ eigenvectors | §5.6.3 | Tipping & Bishop (1999), Thm 2 |
| Maximum-likelihood estimate of $(\mu,\Sigma)$ for a Gaussian is $(\bar X_n,\hat\Sigma_n)$ | Proposition 5.7(4) | Bishop, *PRML*, §2.3.4; Anderson, *An Introduction to Multivariate Statistical Analysis*, §3.2 |
| The Gaussian uniquely maximizes differential entropy at fixed covariance | Proposition 5.14 | Cover & Thomas, *Elements of Information Theory*, Thm 8.6.5 |
| Mean-square differentiability of a stationary GP is governed by the smoothness of $k$ at $r=0$; Bochner's theorem and the Matérn spectral density | §5.8.2 | Rasmussen & Williams, §4.1.1 and §4.2.1; Stein, *Interpolation of Spatial Data*, §2.6 |
| GP regression with the linear kernel coincides with Bayesian linear regression | §5.8.2 | Rasmussen & Williams, §2.1–2.2 |
| Convergence of the Riccati recursion to a unique PSD fixed point (requires detectability and stabilizability, which hold in Worked Example 5.14) | §5.6.2 | Särkkä & Svensson, §6.4; Anderson & Moore, *Optimal Filtering*, Ch. 4 |
| Tower property of conditional expectation, $\mathbb{E}[\mathbb{E}[X\mid Y]] = \mathbb{E}[X]$, and its projection consequence | Solution to Exercise A6 | Chapter 6 of this tutorial |

---

## 5.10 Chapter summary

1. **The covariance matrix is a quadratic form, not a table.** $\operatorname{Var}(a^\top X) = a^\top\Sigma a$ is the definition-in-spirit; positive semidefiniteness is a one-line *consequence* of "variances are nonnegative" (Proposition 5.2), and affine maps act by $\Sigma\mapsto A\Sigma A^\top$ (Proposition 5.3).
2. **Singular $\Sigma$ means the distribution lives in a hyperplane** and therefore has no Lebesgue density (Proposition 5.2(3), drawn in Figure 5.2). The converse fails: the unit circle has covariance $\frac12I$, invertible, and still no density. Full-rank covariance is not evidence of a density. In the other direction, $\sqrt{\det\Sigma}$ *is* the volume of the uncertainty region (Proposition 5.13), and $\frac12\log\det\Sigma$ is its entropy up to an additive constant (Proposition 5.14) — which is where every "volume penalty" term in a variational objective comes from.
3. **Uncorrelated is not independent**, and whitening only equalizes second moments. "Decorrelated" is never "disentangled," and whitening is only defined up to an arbitrary rotation (Proposition 5.10) — so it cannot identify a canonical basis even in principle.
4. **PCA is a two-line theorem** once you have the spectral decomposition: the variance in direction $a$ is a convex combination of eigenvalues weighted by alignment, so the maximum is $\lambda_1$ at $u_1$ (Theorem 5.11). Everything else about PCA is bookkeeping — but it is bookkeeping in the units you chose, since rescaling one feature by $50$ can hand it $99.9\%$ of the explained variance (Exercise B6).
5. **Define the Gaussian as an affine image of i.i.d. standard normals** (Definition 5.15). This handles singular $\Sigma$, matches how you sample on a computer, and is Chapter 1's "distribution as pushforward" idea in the linear case. The density is a *theorem* valid only when $\Sigma\succ0$ (Theorem 5.17).
6. **Every linear functional of a Gaussian is normal, and conversely** (Cramér–Wold). Marginal normality of coordinates is far weaker than joint normality — the $(Z, SZ)$ counterexample takes three lines.
7. **Marginals are free; conditionals require algebra.** Marginalizing is deleting rows and columns of $\Sigma$ (Proposition 5.20). Conditioning is the Schur complement (Lemma 5.21, Theorem 5.22).
8. **The two Gaussian miracles**: the conditional mean $\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)$ is *affine* in the conditioning value, and the conditional covariance $\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ *does not depend on it at all*. Both are special to the Gaussian. The second one is also the Gaussian's most misleading idealization: real uncertainty should depend on the data you saw.
9. **$\Sigma$ answers marginal questions; $\Lambda = \Sigma^{-1}$ answers conditional ones.** $\Lambda_{ij}=0$ iff $X_i\perp X_j\mid X_{\text{rest}}$ (Theorem 5.25). A dense $\Sigma$ is entirely compatible with a sparse $\Lambda$, which is the whole point of Gaussian graphical models.
10. **Completing the square is a procedure, not an inspiration**, and its content is that precisions add. Collect the $x$-quadratic terms into $-\frac12x^\top Px$ and the $x$-linear terms into $x^\top b$; the answer is $\mathcal N(P^{-1}b, P^{-1})$. Applied to prior-times-likelihood (Lemma 5.27) this says precisions add and means are precision-weighted averages, which is why Bayesian updating in the Gaussian world is arithmetic. Almost every "after some algebra" in the literature is these four steps.
11. **The KL divergence between Gaussians is closed form** (Proposition 5.29), and its three pieces are a trace, a Mahalanobis distance, and a log-determinant — i.e. a shape mismatch, a location mismatch, and a volume mismatch. With equal covariances it is exactly half a squared Mahalanobis distance; against $\mathcal N(0,I)$ with diagonal $q$ it is the VAE's KL term, which contributes exactly $0$ for a collapsed latent coordinate and therefore *rewards* posterior collapse.
12. **Linear-Gaussian models are one theorem in four costumes** (Theorem 5.28): Bayesian linear regression, the Kalman filter, factor analysis, and probabilistic PCA are all "prior Gaussian, likelihood linear-Gaussian, apply the lemma," and the Kalman filter is repeated conditioning and nothing more. The theorem gives two algebraically identical forms for the posterior covariance, and you choose between them by dimension: $(\Lambda_x+A^\top \Lambda_y A)^{-1}$ costs $O(d^3)$ while the Woodbury form costs $O(m^3)$. That single choice is the entire difference between weight-space Bayesian regression and function-space GP regression.
13. **Estimated covariances are not covariances you can invert.** $\hat\Sigma_n$ has rank at most $\min(n-1,d)$ (Proposition 5.7), so it is singular whenever $n\le d$ and badly conditioned for some way beyond that. Every method that inverts an estimated covariance — Mahalanobis OOD scoring, whitening, Gaussian discriminant analysis, graphical-model estimation — is silently assuming $n\gg d$ and needs shrinkage when that fails. Relatedly, $\hat\Sigma_n$ is the MLE but is biased by $\frac{n-1}{n}$, which is why `BatchNorm` uses the biased variance to normalize a batch and the unbiased one for its running statistics.
14. **He initialization comes from $\mathbb{E}[\mathrm{ReLU}(Z)^2] = \frac12$.** ReLU discards half the second moment on the forward pass and blocks half the gradient on the backward pass, so weights must supply a factor of $2$: $\operatorname{Var}(w) = 2/n_{\text{in}}$ (Theorem 5.34, Proposition 5.35). Xavier's $2/(n_{\text{in}}+n_{\text{out}})$ is the harmonic mean of the *linear*-network targets and is a factor $\sqrt2$ too small for ReLU.
15. **The failure is geometric in depth.** A gain of $g$ multiplies the second moment by $(g/2)^L$; at $L=20$ that is $10^{\pm6}$. Measured on a 20-layer width-512 network averaged over 30 seeds: $1.9\times10^{-6}$, $1.95$, $2.0\times10^{6}$ for $g = 1, 2, 4$. And every assumption in the derivation holds only at initialization — normalization layers, not initialization, are what keep scales healthy during training.
16. **Gaussian process regression is Theorem 5.22 with an infinite index set.** The kernel is a strong modeling assumption (smoothness class, extrapolation behaviour), the posterior mean is linear in $y$, the posterior variance does not depend on $y$, and the whole thing costs $O(n^3)$ because of one Cholesky factorization.

### Callbacks

This chapter discharged three promises. Chapter 1's insistence that **a distribution is a pushforward** became the *definition* of the Gaussian (Definition 5.15), and the circle generator $G(t)=(\cos t,\sin t)$ returned twice: once to show that zero correlation coexists with total dependence (Worked Example 5.3), and once to show that a full-rank covariance does not imply a density (Common Confusion 5.1) — a strictly stronger warning than Chapter 1's, since Chapter 1's example could have been dismissed as "obviously degenerate." Chapter 1's running example $A=\max(0,Z)$, the distribution with neither pmf nor pdf, turned out to be the engine of the entire initialization literature: $\mathbb{E}[A^2]=1/2$ *is* He's factor of two (Lemma 5.33). Chapter 1's warning that **countability is load-bearing** reappeared at the sharpest possible point, in the observation that sample-path continuity of a Gaussian process is not even a measurable event in the Kolmogorov construction. Chapter 2's Radon–Nikodym theorem underwrote every statement of the form "has a density with respect to $\lambda_d$" — Figure 5.2 is that lesson drawn, with a Gaussian whose mass sits on a line and whose "density" exists only after you fatten the line — and Chapter 3's change-of-variables formula did the real work in deriving the Gaussian density (Theorem 5.17) and in identifying $\sqrt{\det\Sigma}$ as a volume, hence $\frac12\log\det\Sigma$ as an entropy (Proposition 5.14). Chapter 4's concentration machinery paid an unadvertised dividend: the soap-bubble geometry of §5.3 — a high-dimensional Gaussian's mass living in a thin shell of radius $\sqrt d$ — is chi-square concentration wearing geometric clothing, and it is the reason latent-space interpolation is done on spheres rather than chords. Finally, the promise made in §5.0 — that GP regression contains no mathematics beyond Gaussian conditioning — was discharged in the four-line proof of Theorem 5.39.

### Before you continue

1. Why is $\Sigma\succeq0$ a theorem rather than part of the definition of a covariance matrix, and what is the one-line argument?
2. Give an example of a distribution with invertible covariance and no Lebesgue density. Give an example with singular covariance.
3. State the three equivalent definitions of the multivariate Gaussian and say precisely which one fails when $\Sigma$ is singular, and why.
4. Without looking: write down $\mathbb{E}[X_1\mid X_2=x_2]$ and $\operatorname{Cov}(X_1\mid X_2=x_2)$ for a jointly Gaussian vector. Which one depends on $x_2$?
5. What is the difference between a zero in $\Sigma$ and a zero in $\Sigma^{-1}$?
6. You are given $\log f(x) = -\frac12x^\top Px + x^\top b + c$ with $P\succ0$. What distribution is $f$ proportional to? And if $P_1,P_2$ are Gaussians with the same covariance, what is $D_{\mathrm{KL}}(P_1\|P_2)$ in one symbol?
7. Where does the factor of $2$ in He initialization come from? State the expectation and the two places it appears.
8. Why does GP regression cost $O(n^3)$, and what exactly is the $n\times n$ matrix being factorized?
9. Name two properties of the Gaussian conditional that you would *not* want a realistic predictive model to have.

### Looking ahead to Chapter 6

Everything in §5.4 was conditioning in the easy case: densities existed, $\Sigma_{22}$ was invertible, and $P(X_2 = x_2)$ could be sidestepped by taking a ratio of densities. Chapter 6 removes those crutches and defines **conditional expectation** properly — first as an orthogonal projection in $L^2$, which makes $\mathbb{E}[Y\mid\mathcal G]$ the best $\mathcal G$-measurable predictor of $Y$ and explains the "regression to the mean" shrinkage you saw in Worked Example 5.9 as a projection, and then in full generality via the Radon–Nikodym theorem, which defines $\mathbb{E}[Y\mid\mathcal G]$ for any integrable $Y$ and any sub-$\sigma$-algebra $\mathcal G$ with no densities anywhere in sight. With that in hand, Chapter 6 builds **martingales** — Chapter 1's filtrations finally doing real work — proves the **optional stopping theorem** and **Azuma–Hoeffding's** inequality for bounded-difference martingales, and closes with a complete convergence proof for stochastic gradient descent, in which the assumption $\mathbb{E}[g_t\mid\mathcal F_t] = \nabla F(\theta_t)$ that Chapter 1 introduced as a slogan becomes a precise and usable hypothesis.

---

# Exercises

Difficulty: ★ routine, ★★ requires an idea, ★★★ substantial.

## Part A — Theory

**A1 (★).** Let $X$ have covariance $\Sigma$. Show that $\operatorname{Cov}(X)$ is unchanged if you replace $X$ by $X+c$ for a constant vector $c$, and that $\operatorname{Cov}(cX) = c^2\Sigma$ for a scalar $c$. Then show $\operatorname{tr}\Sigma = \sum_i\operatorname{Var}(X_i)$ is invariant under orthogonal transformations of $X$.

**A2 (★).** Let $X_1,\dots,X_n$ be i.i.d. with mean $\mu$ and covariance $\Sigma$. Compute $\operatorname{Cov}(\bar X_n)$ where $\bar X_n = \frac1n\sum_iX_i$. Then compute $\operatorname{Cov}(\bar X_n)$ when instead $\operatorname{Cov}(X_i,X_j) = \rho\Sigma$ for $i\ne j$, and explain what happens as $n\to\infty$.

**A3 (★★).** Prove that if $\Sigma\succeq0$ and $\Sigma_{ii}=0$ for some $i$, then $\Sigma_{ij}=0$ for all $j$. Interpret probabilistically.

**A4 (★★).** Let $X\sim\mathcal N(0,\Sigma)$ in $\mathbb{R}^2$ with $\Sigma = \begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}$, $|\rho|<1$. Compute $P(X_1>0, X_2>0)$. (Hint: this is the *orthant probability*; use the representation $X_2 = \rho X_1 + \sqrt{1-\rho^2}\,W$ and a rotation argument, or compute the angle subtended.)

**A5 (★★).** Complete "Proof 3" of Theorem 5.22: verify by direct multiplication that $\mathcal N(x_1;\mu_{1|2}(x_2),S)\cdot\mathcal N(x_2;\mu_2,\Sigma_{22}) = \mathcal N(x;\mu,\Sigma)$, using $\det\Sigma = \det S\det\Sigma_{22}$ and the block form of $\Lambda$.

**A6 (★★).** Show that for jointly Gaussian $(X,Y)$ with $\operatorname{Var}(Y)>0$, the best predictor $\mathbb{E}[X\mid Y]$ coincides with the best *linear* predictor $\arg\min_{a,b}\mathbb{E}[(X-aY-b)^2]$. Then give a non-Gaussian example where they differ.

**A7 (★★★).** Prove that if $X\sim\mathcal N(0,\Sigma)$ with $\Sigma\succ0$ then $X^\top\Sigma^{-1}X\sim\chi^2_d$, and deduce that the ellipsoid $\{x: x^\top\Sigma^{-1}x\le \chi^2_d(1-\alpha)\}$ has probability exactly $1-\alpha$. Then show that for $d=2$, $\alpha=0.05$, the constant is $5.9915$, and explain why the *area* of this region depends on $\Sigma$ but its *probability* does not.

**A8 (★★★).** Let $X\sim\mathcal N(\mu,\Sigma)$ with $\Sigma\succeq0$ possibly singular of rank $r<d$. Define the Gaussian's density with respect to the $r$-dimensional Lebesgue measure on the affine subspace $\mu+\operatorname{range}(\Sigma)$, and show that it is $(2\pi)^{-r/2}(\prod_{i\le r}\lambda_i)^{-1/2}\exp(-\frac12(x-\mu)^\top\Sigma^+(x-\mu))$ where $\Sigma^+$ is the Moore–Penrose pseudoinverse. Explain in one sentence why this is the right generalization.

**A9 (★★).** Derive the KL formula of Proposition 5.29 in the scalar case from scratch, then use the general formula to establish three things: (i) when $\Sigma_1=\Sigma_2$, $D_{\mathrm{KL}}$ is half a squared Mahalanobis distance; (ii) $D_{\mathrm{KL}}(P_1\|P_2)\ge0$ with equality iff $P_1=P_2$, using only the scalar inequality $\log t \le t-1$; (iii) $D_{\mathrm{KL}}(P_1\|P_2) = h(P_2) - h(P_1)$ whenever $\mu_1=\mu_2$ **and** $\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) = d$, and give a nontrivial example where that side condition holds.

**A10 (★★★).** Conditioning with a singular $\Sigma_{22}$, completing the remark after Corollary 5.23. Let $(X_1,X_2)$ be jointly Gaussian with the usual block structure, $\Sigma_{22}$ possibly singular, and let $\Sigma_{22}^{+}$ denote the Moore–Penrose pseudoinverse. (i) Prove the range lemma: $\Sigma\succeq0$ implies $\operatorname{range}(\Sigma_{21})\subseteq\operatorname{range}(\Sigma_{22})$. (ii) Set $W := X_1 - \Sigma_{12}\Sigma_{22}^{+}(X_2-\mu_2)$ and show $\operatorname{Cov}(W,X_2) = 0$, using (i). (iii) Conclude that for every $x_2\in\mu_2+\operatorname{range}(\Sigma_{22})$, $X_1\mid X_2=x_2\sim\mathcal N\big(\mu_1+\Sigma_{12}\Sigma_{22}^{+}(x_2-\mu_2),\ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{+}\Sigma_{21}\big)$, and explain why no meaning can be attached to conditioning on an $x_2$ outside that affine subspace. (iv) Verify your formula numerically on $X_2 = (Z,Z)^\top$, $X_1 = Z+\varepsilon$ with $Z,\varepsilon$ independent standard normals.

## Part B — Computational labs

**B1 (★).** Sample $\mathcal N(\mu,\Sigma)$ via Cholesky for $\mu = (1,-2,0.5)^\top$ and $\Sigma = \begin{pmatrix}2&0.6&-0.4\\0.6&1&0.2\\-0.4&0.2&0.75\end{pmatrix}$. For $n\in\{10^3,10^5,10^7\}$ report the maximum absolute error in the empirical mean and covariance. Confirm the errors shrink like $n^{-1/2}$. Then verify that using $\Sigma^{1/2}$ instead of $L$ gives a *different sample* for the same random seed but the same empirical covariance.

**B2 (★★).** Verify Theorem 5.22 empirically without using the formula. Sample $10^6$ points from $\mathcal N(0,\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix})$, retain those with $|X_2-1.5|<\epsilon$ for $\epsilon\in\{0.2, 0.05, 0.01\}$, and report the conditional mean and variance for each. Explain the bias–variance tradeoff in the choice of $\epsilon$.

**B3 (★★★).** Run a 20-layer fully connected ReLU network at width 512 with weights $\mathcal N(0,g/n)$ for $g\in\{1,2,4\}$ and plot $\mathbb{E}[(z^{(l)})^2]$ against $l$ on a log scale. Confirm $2/n_{\text{in}}$ keeps it flat. Then repeat with a *single* network draw rather than an average over 30, and explain the drift you observe. Finally, repeat with width 32 and comment.

**B4 (★★).** Implement GP regression from scratch (Cholesky only; no matrix inverses) on $f(x)=\sin x + 0.3x$ with six noisy observations. Plot the posterior mean and $\pm2$ sd band for $\ell\in\{0.3,1,3\}$, and plot the log marginal likelihood as a function of $\ell$ over $[0.2,5]$. Report the maximizing $\ell$.

**B5 (★★★).** Estimate a sparse precision matrix. Simulate $n=2000$ samples from a 10-node Gaussian chain (i.e. $\Lambda$ tridiagonal). Compute $\hat\Sigma$, invert it, and compare the recovered $\hat\Lambda$ to the truth. Then repeat with $n=15$ and explain what goes wrong; connect your answer to Proposition 5.2(3).

**B6 (★★).** Generate $n=20{,}000$ samples in $d=5$ from a Gaussian whose fifth feature has been rescaled by a factor of $50$ (simulating a feature recorded in different units). (i) Run PCA on the raw covariance and report the variance-explained ratios and the loadings of the top component. (ii) Standardize each feature to unit variance and repeat. (iii) Whiten the data with both $L^{-1}$ and $\Sigma^{-1/2}$ and verify numerically that the empirical covariance is $I_5$ in both cases while the individual whitened samples differ. Explain (i) versus (ii) in terms of Theorem 5.11, and (iii) in terms of Proposition 5.10.

## Part C — Paper-style derivations

**C1 (★★★).** Derive the Gaussian conditional mean and covariance by completing the square, starting from the joint density and the block form of $\Lambda$. Identify the conditional covariance as the Schur complement $\Sigma/\Sigma_{22}$, and state precisely where positive definiteness is needed and what fails without it. Then give the alternative derivation via the independent decomposition $W = X_1-\Sigma_{12}\Sigma_{22}^{-1}(X_2-\mu_2)$ and explain which hypotheses each proof requires.

**C2 (★★★).** Prove $\Lambda_{ij}=0\iff X_i\perp X_j\mid X_{\text{rest}}$ for a nondegenerate Gaussian. Then construct an explicit $4\times4$ example in which $\Sigma$ has no zero entries but $\Lambda$ has four, draw the corresponding graph, and verify one of the conditional independencies numerically by simulation.

**C3 (★★★).** Derive He initialization from first principles, including the backward pass. State all four assumptions explicitly, prove the forward and backward recursions, and compute the two places where the factor $1/2$ enters. Then explain what breaks for sigmoid and tanh: give the correct gain for $\tanh$ under the linearization, explain why the linearization is optimistic, and explain why no choice of weight variance repairs sigmoid.

---

# Appendix A — Self-check answers

**Self-check 5.1.**
1. Use Proposition 5.2 with $a = (1,-2)^\top$: $\Sigma a = (3\cdot1 + (-1)(-2),\ (-1)(1)+2(-2))^\top = (5,-5)^\top$, so $a^\top\Sigma a = 1(5) + (-2)(-5) = 15$.
2. No — diagonal entries are variances, hence $\ge0$. No — $\Sigma\succeq0$ means all eigenvalues are $\ge0$ (Proposition 5.2). No — Cauchy–Schwarz (Proposition 5.5) forces $|\Sigma_{12}|\le\sqrt{1\cdot1}=1$.
3. At most $2$, since $\operatorname{Cov}(Y) = A\Sigma A^\top$ has rank at most $\operatorname{rank}(A)\le2$. Rank $1$ means some $a\ne0$ has $a^\top Y$ a.s. constant, i.e. $Y$ lies on a line in $\mathbb{R}^2$.
4. False. $\operatorname{Cov}(X)=I$ says the coordinates are uncorrelated with unit variance; the rescaled circle example (Common Confusion 5.2) has $\operatorname{Cov} = I$ and deterministic dependence.
5. No. By Proposition 5.7(3), $\operatorname{rank}(\hat\Sigma_n)\le\min(n-1,d) = 19 < 50$, so $\hat\Sigma_n$ is singular and has no inverse. You need shrinkage ($\hat\Sigma_n+\varepsilon I$, Ledoit–Wolf) or a structural assumption such as diagonality or sparsity of $\Lambda$.

**Self-check 5.2.**
1. $\max = \lambda_1 = 9$ (Theorem 5.11). Top two capture $(9+4)/14 = 13/14 \approx 0.9286$.
2. Because a linear map $x\mapsto Ax$ scales $\lambda_d$ by $|\det A|$, and a covariance is $\Sigma = AA^\top$, so $|\det A| = \sqrt{\det\Sigma}$. The determinant of $\Sigma$ has units of (length)$^{2d}$; its square root has units of volume.
3. Orthogonal. By Proposition 5.10 both equal $Q_i\Sigma^{-1/2}$, so $W_1W_2^{-1} = Q_1Q_2^{-1}$.
4. Rescaling a feature by $1000$ multiplies its variance by $10^6$; since PCA maximizes raw variance, that feature will dominate the leading component regardless of its actual importance.

**Self-check 5.3.**
1. Definitions (a) and (b). Definition (c) is unstatable since $\det\Sigma = 0$. You get the point mass $\delta_\mu$ (take $k=0$, or $A=0$).
2. $BX\sim\mathcal N\big(\sum_i\mu_i,\ \sum_{i,j}\Sigma_{ij}\big)$ by Corollary 5.18.
3. Because that proof only computes $\varphi_X$ and observes it matches a known Gaussian's characteristic function. Concluding that the *laws* agree — not merely their transforms — requires the uniqueness theorem. Without it, we would only know two distributions share a transform.
4. Arbitrarily large: $p(\mu) = (2\pi)^{-d/2}(\det\Sigma)^{-1/2}\to\infty$ as $\det\Sigma\to0$. No contradiction, because the region where $p$ is large shrinks in volume at exactly the compensating rate (Proposition 5.13). Densities are Radon–Nikodym derivatives, not probabilities.

**Self-check 5.4.**
1. $\Sigma_{11}\succeq\Sigma_{1|2}$, because the difference is $\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = (\Sigma_{22}^{-1/2}\Sigma_{21})^\top(\Sigma_{22}^{-1/2}\Sigma_{21})\succeq0$. Conditioning cannot increase uncertainty in the PSD order.
2. It reduces to $\mu_{1|2}=\mu_1$ and $\Sigma_{1|2}=\Sigma_{11}$ — the conditional law equals the marginal. That is exactly Corollary 5.23: uncorrelated jointly Gaussian blocks are independent, so conditioning tells you nothing.
3. Mean $= \frac{2}{3}(3) = 2$; variance $= 4 - \frac{2\cdot2}{3} = 4-\frac43 = \frac83\approx2.6667$.
4. Because inverting a submatrix is not the same as taking a submatrix of the inverse. Lemma 5.21 shows $\Lambda_{11} = S^{-1}$ where $S$ is the Schur complement, so $\Lambda_{11}^{-1} = S$ — the *block* inverse of the *block* of $\Lambda$, never anything entrywise.

**Self-check 5.5.**
1. $c = \mathcal N(m_1;\ m_2,\ P_1+P_2)$: the density of a $\mathcal N(m_2,P_1+P_2)$ evaluated at the point $m_1$. Yes, $c$ can exceed $1$ — it is a density value, and if $P_1+P_2$ is small it is large (in the scalar case $c\le(2\pi(\sigma_1^2+\sigma_2^2))^{-1/2}$, which exceeds $1$ once $\sigma_1^2+\sigma_2^2 < 1/(2\pi)$).
2. Fused precision $1 + \frac14 = 1.25$, so fused variance $0.8$. Weights are proportional to precisions: $\frac{1}{1.25}=0.8$ on the first estimate, $\frac{0.25}{1.25}=0.2$ on the second.
3. The posterior converges to the prior: $\Sigma_* \to \Lambda_x^{-1}$ and the mean $\to\mu$. Infinitely noisy data carries no information.
4. $P = 6$, $b = 12$, so $\mathcal N(P^{-1}b, P^{-1}) = \mathcal N(2,\ 1/6)$. (Check: $-3x^2+12x = -3(x-2)^2+12$, and $-3(x-2)^2 = -\frac12\cdot6\cdot(x-2)^2$.)
5. $\sigma_j=10$ costs far more: $\frac12(100-1-2\log 10) = 47.20$ nats versus $\frac12(0.01-1-2\log0.1) = 1.81$ for $\sigma_j=0.1$. The asymmetry is structural — the $\sigma_j^2$ term grows *linearly* in the variance while $-2\log\sigma_j$ grows only *logarithmically* as $\sigma_j\to0$. So the KL term punishes an over-dispersed posterior harshly and an over-confident one gently, which is the same mode-seeking bias that makes variational posteriors too narrow.

**Self-check 5.6.**
1. $\Sigma_\theta\to\sigma^2(X^\top X)^{-1}$ and the posterior mean $\to(X^\top X)^{-1}X^\top y$, the OLS estimate. The prior stops shrinking, which also means the posterior is undefined when $X^\top X$ is singular.
2. $K_t = P_{t|t-1}H^\top(HP_{t|t-1}H^\top)^{-1}$; in the scalar case with $H=1$ this is $1$, so $m_{t|t}=y_t$ and $P_{t|t}=0$. A perfect sensor makes the prior irrelevant: trust the measurement completely.
3. Because the recursions for $P$ involve only $F,Q,H,R$ and $P_0$ — the observations $y_t$ enter only the mean recursion. This is Theorem 5.22's "$\Sigma_{1|2}$ does not depend on $x_2$" propagated through time.
4. $W$ and $WQ$ for any orthogonal $Q\in\mathbb{R}^{k\times k}$: $WQ(WQ)^\top = WW^\top$. The likelihood is identical, so no amount of data distinguishes them; "the $j$-th factor means such-and-such" is therefore not a claim the model can support without extra assumptions.

**Self-check 5.7.**
1. From $\mathbb{E}[\mathrm{ReLU}(Z)^2] = \frac12\mathbb{E}[Z^2]$ (Lemma 5.33) on the forward pass, and from $\mathbb{E}[\mathbf 1\{Z>0\}] = \frac12$ on the backward pass. Same constant, two mechanisms.
2. He (fan-in): $\sqrt{2/1024} = 0.044194$. Xavier: $\sqrt{2/(1024+256)} = \sqrt{2/1280} = 0.039528$. (He fan-out would be $\sqrt{2/256} = 0.088388$.)
3. $\operatorname{Var}(w) = 1/n_{\text{in}}$, since $\mathbb{E}[\phi(z)^2]=\mathbb{E}[z^2]$ for the identity. The network computes $W^{(L-1)}\cdots W^{(0)}x$, a single linear map: depth adds no expressive power (though it does change the optimization geometry, which is a separate and interesting story).
4. Because each layer *multiplies* the second moment by a fixed factor $g/2$; composing $L$ layers multiplies $L$ factors, giving $(g/2)^L$. Exponential, not linear — which is why "roughly the right scale" is not good enough at depth.

**Self-check 5.8.**
1. Because a GP is *defined* by its finite-dimensional marginals, and regression only ever asks about finitely many points (the $n$ training inputs and the test inputs). Theorem 5.39 is Theorem 5.22 with relabeled blocks.
2. The posterior mean doubles, since $\bar f_* = K_*^\top K_y^{-1}y$ is linear in $y$. The posterior covariance is completely unchanged — it does not involve $y$. So do the error bars: identical.
3. Because a large $\ell$ asserts that the function varies only over long distances, i.e. it forbids rapid variation. Small $\ell$ permits more functions, so it is the weaker assumption — but a GP that permits everything also learns almost nothing from a finite sample ($\ell=0.3$ in Worked Example 5.18).
4. $50000^2\times8$ bytes $= 2\times10^{10}$ bytes $= 20$ GB for $K$ alone, plus another 20 GB for its Cholesky factor. Instead: sparse/inducing-point variational GPs ($O(nm^2)$ with $m\sim10^3$), structured kernel interpolation, or conjugate-gradient methods that only need matrix–vector products.

---

# Solutions to Exercises

## Part A

**A1.** *Thinking:* the covariance is built from *centered* variables, so anything that shifts the mean must cancel.
Since $\mathbb{E}[X+c] = \mu+c$, the centered vector $(X+c)-(\mu+c) = X-\mu$ is unchanged, hence so is $\Sigma$. For scaling, $\operatorname{Cov}(cX) = c\Sigma c^\top = c^2\Sigma$ by Proposition 5.3 with $A = cI$. For the trace: if $Y = UX$ with $U$ orthogonal, then $\operatorname{Cov}(Y) = U\Sigma U^\top$, and $\operatorname{tr}(U\Sigma U^\top) = \operatorname{tr}(\Sigma U^\top U) = \operatorname{tr}\Sigma$ by cyclicity of the trace. Total variance is a rotation invariant; this is why "percentage of variance explained" in PCA is well posed but *scale*-dependent.

**A2.** $\operatorname{Cov}(\bar X_n) = \frac{1}{n^2}\operatorname{Cov}\big(\sum_iX_i\big) = \frac{1}{n^2}\sum_i\operatorname{Cov}(X_i) = \frac{\Sigma}{n}$, using independence to kill $n(n-1)$ cross terms.
With equicorrelation $\operatorname{Cov}(X_i,X_j) = \rho\Sigma$ for $i\ne j$, there are $n(n-1)$ cross terms:
$$
\operatorname{Cov}(\bar X_n) = \frac{1}{n^2}\big(n\Sigma + n(n-1)\rho\Sigma\big) = \frac{\Sigma}{n}\big(1+(n-1)\rho\big)\ \xrightarrow[n\to\infty]{}\ \rho\Sigma .
$$
*Thinking:* the limit is not zero. Averaging correlated samples does **not** drive the variance to zero; it plateaus at $\rho\Sigma$. Numerically with $\rho=0.5$: the factor $(1+(n-1)\rho)/n$ is $0.55$ at $n=10$, $0.505$ at $n=100$, $0.5005$ at $n=1000$ — you gain almost nothing after the first few samples. This is why correlated Monte Carlo (e.g. consecutive MCMC draws, or augmentations of the same image) has an *effective sample size* far below $n$, and why "just collect more data" fails when the data are not independent.

**A3.** *Thinking:* use the quadratic form, not the entries.
Suppose $\Sigma_{ii}=0$, i.e. $\operatorname{Var}(X_i)=0$, so $X_i = \mu_i$ a.s. by Proposition 5.2(3). Then $\Sigma_{ij} = \mathbb{E}[(X_i-\mu_i)(X_j-\mu_j)] = \mathbb{E}[0\cdot(X_j-\mu_j)] = 0$. Alternatively, purely algebraically: the $2\times2$ submatrix $\begin{pmatrix}0&\Sigma_{ij}\\\Sigma_{ij}&\Sigma_{jj}\end{pmatrix}$ must be PSD, and its determinant is $-\Sigma_{ij}^2\ge0$, forcing $\Sigma_{ij}=0$. Probabilistically: a constant coordinate carries no information and covaries with nothing.

**A4.** *Thinking:* rotate into independent coordinates and count angles.
Write $X_1 = Z_1$, $X_2 = \rho Z_1 + \sqrt{1-\rho^2}Z_2$ with $Z\sim\mathcal N(0,I_2)$ (this reproduces the right covariance). Then $\{X_1>0,X_2>0\}$ is the event that $Z$ lies in a wedge with vertex at the origin bounded by the lines $z_1=0$ and $\rho z_1+\sqrt{1-\rho^2}z_2 = 0$. Because $\mathcal N(0,I_2)$ is rotationally invariant, the probability of a wedge of angle $\theta$ is $\theta/(2\pi)$. The two boundary lines make an angle of $\pi/2 + \arcsin\rho$, giving
$$
P(X_1>0,X_2>0) = \frac{\pi/2+\arcsin\rho}{2\pi} = \frac14 + \frac{\arcsin\rho}{2\pi}.
$$
Checks: $\rho=0$ gives $1/4$ (independence). $\rho=1$ gives $1/2$. $\rho=-1$ gives $0$. For $\rho=0.8$: $\frac14+\frac{\arcsin0.8}{2\pi} = 0.397584$; Monte Carlo with $5\times10^6$ samples (seed 1) gives $0.397630$. For $\rho=-0.5$: $1/6 = 0.166667$.

**A5.** *Thinking:* two things must match — the normalizing constants and the exponents.
Constants: $\mathcal N(x_1;\mu_{1|2},S)$ contributes $(2\pi)^{-d_1/2}(\det S)^{-1/2}$ and $\mathcal N(x_2;\mu_2,\Sigma_{22})$ contributes $(2\pi)^{-d_2/2}(\det\Sigma_{22})^{-1/2}$. Their product is $(2\pi)^{-d/2}(\det S\det\Sigma_{22})^{-1/2} = (2\pi)^{-d/2}(\det\Sigma)^{-1/2}$ by Lemma 5.21. ✓
Exponents: with $u = x_1-\mu_1$, $v = x_2-\mu_2$, $K = \Sigma_{12}\Sigma_{22}^{-1}$, the sum of the two exponents is
$$
-\tfrac12(u-Kv)^\top S^{-1}(u-Kv) - \tfrac12 v^\top\Sigma_{22}^{-1}v .
$$
Expand: $-\frac12\big[u^\top S^{-1}u - 2u^\top S^{-1}Kv + v^\top(K^\top S^{-1}K + \Sigma_{22}^{-1})v\big]$. Now compare with Lemma 5.21's blocks: $\Lambda_{11} = S^{-1}$ ✓; $\Lambda_{12} = -S^{-1}\Sigma_{12}\Sigma_{22}^{-1} = -S^{-1}K$, matching the cross term $2u^\top\Lambda_{12}v$ ✓; and $\Lambda_{22} = \Sigma_{22}^{-1}+\Sigma_{22}^{-1}\Sigma_{21}S^{-1}\Sigma_{12}\Sigma_{22}^{-1} = \Sigma_{22}^{-1}+K^\top S^{-1}K$ ✓. So the exponent equals $-\frac12(x-\mu)^\top\Lambda(x-\mu)$, and the product is the joint density.

**A6.** *Thinking:* the general best predictor minimizes MSE over all measurable functions; the Gaussian's conditional mean happens to be affine, so the two optimizations coincide.
By Theorem 5.22, $\mathbb{E}[X\mid Y=y] = \mu_X + \frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(Y)}(y-\mu_Y)$, which is affine in $y$, so it lies in the feasible set of the linear problem. And the conditional mean is the global minimizer of $\mathbb{E}[(X-g(Y))^2]$ over all measurable $g$ (this is the projection property Chapter 6 develops; here it follows from expanding $\mathbb{E}[(X-g(Y))^2] = \mathbb{E}[(X-\mathbb{E}[X|Y])^2] + \mathbb{E}[(\mathbb{E}[X|Y]-g(Y))^2]$, since the cross term vanishes by conditioning). A minimizer over a larger set that happens to lie in the smaller set is also a minimizer over the smaller set. Hence they coincide.
Non-Gaussian counterexample: $Y\sim\mathrm{Unif}[-1,1]$, $X = Y^2$. Then $\operatorname{Cov}(X,Y) = \mathbb{E}[Y^3] = 0$, so the best linear predictor is the constant $\mathbb{E}[X] = 1/3$, with MSE $\operatorname{Var}(Y^2) = 4/45$. But $\mathbb{E}[X\mid Y] = Y^2$ exactly, with MSE $0$.

**A7.** *Thinking:* whiten first; everything follows.
Let $Z := \Sigma^{-1/2}X$. By Corollary 5.18, $Z\sim\mathcal N(0,\Sigma^{-1/2}\Sigma\Sigma^{-1/2}) = \mathcal N(0,I_d)$, so its coordinates are i.i.d. standard normal (Theorem 5.17(c) $\Rightarrow$ the density factorizes). Then $X^\top\Sigma^{-1}X = \|\Sigma^{-1/2}X\|^2 = \sum_{i=1}^dZ_i^2$, which is $\chi^2_d$ by definition. Consequently $P(X^\top\Sigma^{-1}X\le q) = F_{\chi^2_d}(q)$, so choosing $q = \chi^2_d(1-\alpha)$ gives probability exactly $1-\alpha$.
For $d=2$, $\chi^2_2$ is $\mathrm{Exp}(1/2)$, so $P(Q\le q) = 1-e^{-q/2} = 0.95$ gives $q = -2\log(0.05) = 5.991465$. ✓
Area versus probability: by Proposition 5.13 the region has area $\pi q\sqrt{\det\Sigma}$, which depends on $\Sigma$ — but the *probability* is computed after whitening, and whitening maps the region to a fixed disc of radius $\sqrt q$ while mapping $P_X$ to $\mathcal N(0,I)$. Both the region and the measure transform together, so the probability is $\Sigma$-free. This is the same "the map takes everything with it" argument as in the change-of-variables formula.

**A8.** *Thinking:* the right base measure is the one that lives on the support.
Write $\Sigma = UDU^\top$ with $D = \operatorname{diag}(\lambda_1,\dots,\lambda_r,0,\dots,0)$, $\lambda_i>0$ for $i\le r$, and let $D_r := \operatorname{diag}(\lambda_1,\dots,\lambda_r)$. Let $U_r$ be the first $r$ columns. By Definition 5.15, $X = \mu + U_rD_r^{1/2}Z_r$ with $Z_r\sim\mathcal N(0,I_r)$, so $X$ lives on $M := \mu+\operatorname{range}(U_r)$, an $r$-dimensional affine subspace. Parameterize $M$ by $x = \mu+U_r t$, $t\in\mathbb{R}^r$; since $U_r$ has orthonormal columns, this parameterization is an isometry, so $r$-dimensional Lebesgue measure on $M$ pulls back to Lebesgue measure on the $t$-coordinates. There, $t = D_r^{1/2}Z_r\sim\mathcal N(0,D_r)$, whose density is $(2\pi)^{-r/2}(\prod_{i\le r}\lambda_i)^{-1/2}\exp(-\frac12 t^\top D_r^{-1}t)$. Finally, for $x\in M$ we have $t = U_r^\top(x-\mu)$ and
$$
t^\top D_r^{-1}t = (x-\mu)^\top U_rD_r^{-1}U_r^\top(x-\mu) = (x-\mu)^\top\Sigma^+(x-\mu),
$$
since $\Sigma^+ = U_rD_r^{-1}U_r^\top$ is precisely the Moore–Penrose pseudoinverse. This is the right generalization because it replaces $\det\Sigma$ (which is $0$) by the product of the *nonzero* eigenvalues, and $\Sigma^{-1}$ by $\Sigma^+$ — exactly the substitutions that restrict the whole computation to the subspace where the mass actually is.

**A9.** *Thinking:* the whole derivation is one expectation of a quadratic form; the inequality at the end is Jensen in disguise.
**Scalar case.** With $p_i = \mathcal N(\mu_i,\sigma_i^2)$,
$$
\log\frac{p_1(x)}{p_2(x)} = \log\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}.
$$
Take $\mathbb{E}_{X\sim P_1}$. The middle term gives $-\frac12$. For the last, write $X-\mu_2 = (X-\mu_1)+(\mu_1-\mu_2)$; the cross term vanishes and we get $\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}$. Hence
$$
D_{\mathrm{KL}} = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12,
$$
which is Proposition 5.29 at $d=1$.
**(i)** $\Sigma_1=\Sigma_2=\Sigma$ makes $\operatorname{tr}(\Sigma^{-1}\Sigma)=d$ and $\log(\det\Sigma/\det\Sigma)=0$, so only the middle term survives: $D_{\mathrm{KL}} = \frac12(\mu_2-\mu_1)^\top\Sigma^{-1}(\mu_2-\mu_1) = \frac12 d_\Sigma(\mu_1,\mu_2)^2$.
**(ii)** Let $t_1,\dots,t_d>0$ be the eigenvalues of $\Sigma_2^{-1}\Sigma_1$ (real and positive because $\Sigma_2^{-1}\Sigma_1$ is similar to the symmetric positive-definite $\Sigma_2^{-1/2}\Sigma_1\Sigma_2^{-1/2}$). Then $\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) = \sum_it_i$ and $\log\frac{\det\Sigma_2}{\det\Sigma_1} = -\sum_i\log t_i$, so
$$
D_{\mathrm{KL}} = \tfrac12\sum_{i=1}^d\big(t_i - 1 - \log t_i\big) + \tfrac12(\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1).
$$
Both pieces are $\ge0$: the second because $\Sigma_2^{-1}\succ0$, the first because $\log t\le t-1$ for all $t>0$ with equality only at $t=1$. Equality throughout forces $\mu_1=\mu_2$ and every $t_i=1$, i.e. $\Sigma_2^{-1/2}\Sigma_1\Sigma_2^{-1/2}=I$, i.e. $\Sigma_1=\Sigma_2$. *This decomposition is worth remembering:* KL between Gaussians is a sum of one scalar penalty $t-1-\log t$ per eigenvalue of the covariance mismatch, plus a Mahalanobis term for the mean mismatch.
**(iii)** If $\mu_1=\mu_2$ and $\operatorname{tr}(\Sigma_2^{-1}\Sigma_1)=d$, the formula collapses to $\frac12\log\frac{\det\Sigma_2}{\det\Sigma_1}$, and by Proposition 5.14 that is exactly $h(P_2)-h(P_1)$ since the $(2\pi e)^d$ factors cancel. A nontrivial example: $d=2$, $\Sigma_2 = I_2$, $\Sigma_1 = \operatorname{diag}(a, 2-a)$ for any $a\in(0,2)$, since $\operatorname{tr}(\Sigma_1) = 2 = d$. At $a=0.5$: $\det\Sigma_1 = 0.75$ and $D_{\mathrm{KL}} = \frac12\log\frac43 = 0.14384$. Note this is *not* a general identity — KL is not a difference of entropies, and the side condition is precisely what makes the trace term cancel the $-d$.

**A10.** *Thinking:* Proof 2 of Theorem 5.22 never inverted anything except $\Sigma_{22}$; replace that one inverse by a pseudoinverse and check that the single place invertibility was used still goes through — that place is $\operatorname{Cov}(W,X_2) = 0$.
**(i) Range lemma.** Suppose $a \in \ker(\Sigma_{22})$, i.e. $\Sigma_{22}a = 0$. Then $\operatorname{Var}(a^\top X_2) = a^\top\Sigma_{22}a = 0$, so by Proposition 5.5 (Cauchy–Schwarz in $L^2$), for every coordinate $X_{1,i}$ of $X_1$: $|\operatorname{Cov}(X_{1,i},\,a^\top X_2)| \le \sqrt{\operatorname{Var}(X_{1,i})\cdot 0} = 0$. Stacking coordinates, $\Sigma_{12}a = 0$, i.e. $\ker(\Sigma_{22}) \subseteq \ker(\Sigma_{12})$. Taking orthogonal complements (both matrices are built from the symmetric $\Sigma$, and $\operatorname{range}(M^\top) = \ker(M)^\perp$) gives $\operatorname{range}(\Sigma_{21}) \subseteq \operatorname{range}(\Sigma_{22})$. A degenerate direction covaries with nothing — the probabilistic content of the lemma.
**(ii)** $\operatorname{Cov}(W,X_2) = \Sigma_{12} - \Sigma_{12}\Sigma_{22}^{+}\Sigma_{22}$. Now $\Sigma_{22}^{+}\Sigma_{22}$ is the orthogonal projector onto $\operatorname{range}(\Sigma_{22})$ (a defining property of the pseudoinverse for symmetric matrices), so $\Sigma_{12}\Sigma_{22}^{+}\Sigma_{22} = \Sigma_{12}$ *precisely because* part (i) put every row of $\Sigma_{12}$ (column of $\Sigma_{21}$) inside that range — the projector acts as the identity on it. Hence $\operatorname{Cov}(W,X_2) = 0$. This is the step that silently fails if you skip (i): for an arbitrary matrix pair the projector would chop something off.
**(iii)** $(W,X_2)$ is jointly Gaussian (affine image of $(X_1,X_2)$, Corollary 5.18) and uncorrelated, hence independent by Corollary 5.23 — which, importantly, was proved for singular covariances too. Then $X_1 = W + \Sigma_{12}\Sigma_{22}^{+}(X_2-\mu_2)$ decomposes $X_1$ into an $X_2$-independent part plus a function of $X_2$, and conditioning on $X_2 = x_2$ gives the stated Gaussian with $\mathbb{E}[W] = \mu_1$ and $\operatorname{Cov}(W) = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{+}\Sigma_{21}$ (expand as in Proof 2, using $\Sigma_{22}^{+}\Sigma_{22}\Sigma_{22}^{+} = \Sigma_{22}^{+}$). For $x_2\notin\mu_2+\operatorname{range}(\Sigma_{22})$: the event $\{X_2 \text{ near } x_2\}$ has probability zero for *every* neighbourhood, not merely for the point — $X_2$ lives a.s. in the affine subspace $\mu_2+\operatorname{range}(\Sigma_{22})$ (Proposition 5.2(3) applied in each degenerate direction) — so there is no sequence of genuine conditional probabilities to take a limit of, and any "conditional given $X_2 = x_2$" would be a definition about nothing. Chapter 6's disintegration machinery makes the same verdict precise: conditional laws are defined only up to $P_{X_2}$-null sets, and the whole complement of the support is one such set.
**(iv)** $X_2 = (Z,Z)^\top$, $X_1 = Z+\varepsilon$: here $\Sigma_{22} = \begin{pmatrix}1&1\\1&1\end{pmatrix}$ (rank 1), $\Sigma_{12} = (1,1)$, $\Sigma_{11} = 2$, and $\Sigma_{22}^{+} = \frac14\begin{pmatrix}1&1\\1&1\end{pmatrix}$. The formula gives conditional mean $\Sigma_{12}\Sigma_{22}^{+}x_2 = \frac{x_{2,1}+x_{2,2}}{2}$ — on the support $x_2 = (z,z)$ this is $z$ — and conditional variance $2 - (1,1)\Sigma_{22}^{+}(1,1)^\top = 2-1 = 1$. Sanity: given $Z = z$, $X_1 = z+\varepsilon\sim\mathcal N(z,1)$ directly. ✓ Rejection check: $4\times10^6$ draws (seed 10), keeping $|Z-1|<0.01$, retains $19{,}334$ samples with empirical mean $0.9962$ and variance $1.0009$ against the predicted $1$ and $1$ — agreement within the Monte Carlo standard error $1/\sqrt{19334} = 0.0072$.

## Part B

**B1.** *Thinking:* the point is that the empirical covariance converges at the $n^{-1/2}$ rate, and that the choice of factor is invisible in the limit. But a rate cannot be read off one run.
A first attempt with a single seed gave maximum absolute deviations of $0.0585$, $0.0147$, $0.0018$ at $n = 10^3,10^5,10^7$ — a total shrinkage of $33\times$ where $\sqrt{10^4} = 100$ was predicted. Rerunning across seeds shows why: the $n=10^3$ error alone ranges from $0.039$ to $0.168$ over 20 seeds, a factor of four. **A single realization cannot demonstrate a rate.** Taking medians over 20 seeds instead:

| $n$ | Cholesky factor (median, [min, max]) | symmetric square root (median) |
|---|---|---|
| $10^3$ | 0.07335 &nbsp;[0.0393, 0.1676] | 0.06585 |
| $10^5$ | 0.00771 &nbsp;[0.0036, 0.0194] | 0.00855 |
| $10^7$ | 0.00077 &nbsp;[0.0002, 0.0020] | 0.00066 |

Now the rate is clean: $0.07335/0.00771 = 9.5$ and $0.00771/0.00077 = 10.0$ per two decades of $n$, against a predicted $\sqrt{100} = 10$; end to end, $95\times$ and $100\times$ against a predicted $100$. This is the same lesson Exercise B3 teaches about network initialization, and it is worth internalizing as a habit: **whenever you want to verify a rate, average over seeds; whenever you want to see the spread, report the seeds.**
For reference, one $n=10^6$ run gave
$$
\begin{pmatrix}1.997027&0.598935&-0.399803\\0.598935&0.999450&0.199381\\-0.399803&0.199381&0.749793\end{pmatrix}
$$
against a truth of $\begin{pmatrix}2&0.6&-0.4\\0.6&1&0.2\\-0.4&0.2&0.75\end{pmatrix}$.
Using $\Sigma^{1/2}$ instead of $L$ with the *same* $Z$ produces different individual samples, because $L$ and $\Sigma^{1/2}$ differ by an orthogonal rotation of the latent (Proposition 5.10). But both have covariance $\Sigma$, so the two columns above are statistically indistinguishable — neither is systematically better. **The factor is not identifiable from the data; only $\Sigma$ is.** That is Proposition 5.10 and the factor-analysis non-identifiability of §5.6.3, met in the simplest possible setting.

**B2.** With $10^6$ samples (seed 7) from $\mathcal N(0,\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix})$:

| $\epsilon$ | retained $n$ | conditional mean | conditional variance |
|---|---|---|---|
| 0.20 | 52312 | 1.1868 | 0.3726 |
| 0.05 | 12986 | 1.2013 | 0.3707 |
| 0.01 | 2506 | 1.1868 | 0.3571 |

Truth: mean $1.2$, variance $0.36$.
*Thinking about the tradeoff.* A wide slab keeps many samples (low Monte Carlo variance) but conditions on $X_2\in[1.5-\epsilon,1.5+\epsilon]$ rather than $X_2 = 1.5$, mixing in conditional distributions with different means — a bias of order $\epsilon^2$ in the variance estimate and order $\epsilon$ in the mean (partly cancelling by symmetry, though the density's slope over the slab tilts the weighting). A narrow slab has small bias but few samples, so the estimate is noisy: at $\epsilon=0.01$ the standard error on the mean is $0.6/\sqrt{2506}=0.012$. Both columns show it: the variance estimate drifts *down* toward $0.36$ as $\epsilon$ shrinks (bias reduction) while the numbers get visibly noisier. This is the same bias–variance structure as kernel density estimation bandwidth selection, and it is also why rejection-conditioning is a diagnostic tool rather than an algorithm — in $d$ dimensions the retention rate falls like $\epsilon^d$.

**B3.** See Worked Example 5.16 and Figure 5.5 for the averaged result: with $g=2$ the second moment sits at $\approx2.0$ from layer 1 to layer 20; with $g=1$ it falls to $1.86\times10^{-6}$; with $g=4$ it rises to $2.04\times10^6$.
*Single draw.* One network (seed 2024, width 512, batch 4096) at $g=2$ gave layer-20 second moment $3.54$ rather than $2.0$ — a $77\%$ overshoot. *Thinking:* the recursion $\mathbb{E}[(z^{(l+1)})^2] = \frac{n v}{2}\mathbb{E}[(z^{(l)})^2]$ is exact **in expectation over the weights**. For a single realization, each layer contributes a multiplicative factor that fluctuates around $1$ with relative standard deviation of order $n^{-1/2}$; the log of the product is then a random walk with $L$ steps, so $\log$-variance grows like $L/n$ and the realized ratio is log-normally spread. At width 32 the effect is dramatic: $L/n = 20/32$ gives log-standard-deviation of order $0.8$, i.e. realized ratios spanning roughly an order of magnitude across seeds. Practical conclusion: correct initialization is *necessary but not sufficient* at small width; width itself buys concentration, which is an independent reason wide networks are easier to train.

**B4.** With $f(x)=\sin x+0.3x$, six inputs $(0.5,1.5,3,4.5,6,7.5)$, noise $\sigma_n = 0.1$, seed 3 giving $y = (0.834, 1.192, 1.083, 0.316, 1.475, 3.166)$:

| $\ell$ | log marginal likelihood | sd at $x=2.25$ | sd at $x=9$ |
|---|---|---|---|
| 0.3 | −13.2579 | 0.9975 | 1.0000 |
| 1.0 | −11.1702 | 0.3099 | 0.9407 |
| 3.0 | −19.5192 | 0.0791 | 0.3178 |

Sweeping $\ell$ over a 97-point grid on $[0.2,5]$ maximizes the log marginal likelihood at $\ell = 1.75$, value $-9.5328$.
*Implementation notes that matter.* Compute $L = \mathrm{chol}(K+\sigma_n^2I)$ once; then $\alpha = L^{-\top}(L^{-1}y)$ by two triangular solves, $\bar f_* = K_*^\top\alpha$, $v = L^{-1}K_*$ and $\operatorname{Var}(f_*) = \operatorname{diag}(K_{**}) - \sum v^2$ columnwise, and $\log\det K_y = 2\sum_i\log L_{ii}$. Never form $(K+\sigma_n^2I)^{-1}$: it costs three times more and is numerically worse. If the Cholesky fails, add jitter — and re-read Common Confusion 5.3 about what that jitter means.

**B5.** With a 10-node chain $\Lambda = 2I - 0.7(\text{off-diagonals})$ and $n=2000$ (seed 5), inverting the sample covariance recovers $\Lambda$ with maximum entrywise error $0.1584$ — the tridiagonal structure is clearly visible, though the off-chain entries are not exactly zero.
With $n=15$, the maximum error explodes to $65.9067$. *Thinking about why.* The sample covariance $\hat\Sigma_n = \frac1n\sum(x_i-\bar x)(x_i-\bar x)^\top$ is itself a covariance matrix — of the empirical distribution $\hat P_n$ — so Proposition 5.2 applies to it. Its rank is at most $\min(n-1,d)$, the $-1$ coming from centering. So for $n\le d$ it is *singular*: by Proposition 5.2(3) the empirical distribution lives in a proper subspace, and $\hat\Sigma_n^{-1}$ does not exist at all. For $n$ just above $d$ it is invertible but nearly singular, and inversion amplifies the small eigenvalues catastrophically — the smallest eigenvalue of $\hat\Sigma_n$ is badly underestimated, and $1/\hat\lambda_{\min}$ is enormous. The fix is exactly the graphical lasso: penalize $\|\Lambda\|_1$, which both regularizes the inverse and imposes the sparsity you believe in. Proposition 5.7(3) is the general statement; here $d=10$ and $n=15$ puts you just barely on the invertible side of it, which is the worst place to be — invertible enough that the code runs, singular enough that the answer is meaningless.

**B6.** *Thinking:* PCA maximizes raw variance, and raw variance has units.
With $n=20{,}000$, $d=5$, and the fifth feature multiplied by $50$:

| | eigenvalues | variance explained |
|---|---|---|
| raw covariance | $8341.1,\ 6.56,\ 1.09,\ 0.339,\ 0.305$ | $0.9990,\ 0.0008,\ 0.0001,\ \approx0,\ \approx0$ |
| after standardizing | $2.814,\ 1.750,\ 0.249,\ 0.112,\ 0.076$ | $0.5627,\ 0.3499,\ 0.0498,\ 0.0224,\ 0.0152$ |

The raw top component has loadings $(0.010,\ 0.008,\ 0.021,\ 0.008,\ \mathbf{0.9997})$ — it *is* the fifth feature, to four decimals. PC1 "explains $99.9\%$ of the variance," and the statement is completely vacuous: rescaling that feature by $50$ multiplied its variance by $2500$, and Theorem 5.11 dutifully reports that the largest-variance direction is the one you inflated. After standardizing, the top loading is spread across all five features $(0.179, 0.492, 0.527, 0.359, 0.565)$ and the spectrum reveals the genuine three-factor structure the data was generated with. **A "percentage of variance explained" is meaningful only relative to a choice of units**, which is why `sklearn` pipelines put a `StandardScaler` before `PCA` and why doing PCA on raw pixel intensities and on $z$-scored pixel intensities are different analyses.
For (iii): whitening with $L^{-1}$ gives an empirical covariance within $3.3\times10^{-15}$ of $I_5$, and whitening with $\Sigma^{-1/2}$ within $6.5\times10^{-14}$ — both exact to floating point. Yet the individual whitened vectors differ (the first three samples differ by up to $1.52$ in absolute value). Proposition 5.10 says exactly this: the two whiteners differ by an orthogonal $Q$, so they produce the same second moments and different coordinates. If your downstream method's answer changes when you swap $L^{-1}$ for $\Sigma^{-1/2}$, it is depending on something second moments do not determine.

## Part C

**C1.** *Thinking:* the two proofs differ in what they need, and that difference is the interesting part.
**Completing the square.** Start from Theorem 5.17(c) with $\Sigma\succ0$. Fix $x_2$; as a function of $x_1$,
$$
-2\log p(x_1\mid x_2) = (x-\mu)^\top\Lambda(x-\mu) + \text{const} = u^\top\Lambda_{11}u + 2u^\top\Lambda_{12}v + \text{const},
$$
with $u = x_1-\mu_1$, $v=x_2-\mu_2$. Because $\Lambda_{11}\succ0$ it is invertible and we may write
$$
u^\top\Lambda_{11}u+2u^\top\Lambda_{12}v = (u+\Lambda_{11}^{-1}\Lambda_{12}v)^\top\Lambda_{11}(u+\Lambda_{11}^{-1}\Lambda_{12}v) - v^\top\Lambda_{21}\Lambda_{11}^{-1}\Lambda_{12}v .
$$
Applying the completing-the-square pattern of §5.5.4, the conditional is $\mathcal N(\mu_1 - \Lambda_{11}^{-1}\Lambda_{12}v,\ \Lambda_{11}^{-1})$. Lemma 5.21 converts: $\Lambda_{11}^{-1} = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = \Sigma/\Sigma_{22}$, the **Schur complement of $\Sigma_{22}$**, and $-\Lambda_{11}^{-1}\Lambda_{12} = \Sigma_{12}\Sigma_{22}^{-1}$.
**Where positive definiteness is needed.** Three places, and they are different. (i) $\Sigma\succ0$ is needed to have a density at all — Theorem 5.17(c) is inapplicable otherwise. (ii) $\Sigma_{22}\succ0$ is needed for $\Sigma_{22}^{-1}$ in the statement; it follows from $\Sigma\succ0$ but is the weaker condition actually used, and it is what makes $p_2(x_2)>0$ so the ratio defining the conditional density is legal for *every* $x_2$. (iii) $\Lambda_{11}\succ0$ is needed to complete the square; by Lemma 5.21 it equals $S^{-1}$, and $S\succ0$ came from the congruence argument. Without (i), the whole density-ratio approach collapses; without (ii), conditioning on $x_2$ is only defined up to null sets and requires Chapter 6's machinery.
**Independent-decomposition proof.** Set $W := X_1-\Sigma_{12}\Sigma_{22}^{-1}(X_2-\mu_2)$. Then $(W,X_2)$ is jointly Gaussian (affine image), $\operatorname{Cov}(W,X_2) = \Sigma_{12}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22}=0$, hence $W\perp X_2$ by Corollary 5.23; and $\operatorname{Cov}(W) = S$. Since $X_1 = W + \Sigma_{12}\Sigma_{22}^{-1}(X_2-\mu_2)$, conditioning on $X_2=x_2$ shifts by a constant and leaves $W$'s law alone.
**Comparison of hypotheses.** The completing-the-square proof needs $\Sigma\succ0$ (a density) and gives the precision form for free. The decomposition proof needs only $\Sigma_{22}\succ0$ — it never inverts $\Sigma_{11}$ or $\Sigma$ — so it covers *singular* $\Sigma_{11}$, i.e. a degenerate $X_1$ block. It also explains the two miracles structurally rather than algebraically: the mean is affine because such a decomposition exists, and the covariance is $x_2$-free because $W$ is independent of $X_2$.

**C2.** *Thinking:* the theorem is really "condition on the rest, then read off a $2\times2$."
Let $\Sigma\succ0$, $\Lambda=\Sigma^{-1}$, $i\ne j$, $R$ the remaining indices. Apply Theorem 5.22 with block $1 = \{i,j\}$, block $2 = R$: the conditional law of $(X_i,X_j)$ given $X_R=x_R$ is Gaussian with covariance equal to the Schur complement $\Sigma/\Sigma_{RR}$, which by Lemma 5.21 equals $(\Lambda_{\{i,j\}\{i,j\}})^{-1}$. Inverting the $2\times2$ matrix $\begin{pmatrix}\Lambda_{ii}&\Lambda_{ij}\\\Lambda_{ij}&\Lambda_{jj}\end{pmatrix}$ gives off-diagonal entry $-\Lambda_{ij}/(\Lambda_{ii}\Lambda_{jj}-\Lambda_{ij}^2)$, which vanishes iff $\Lambda_{ij}=0$. Since the conditional law is jointly Gaussian, Corollary 5.23 turns "zero conditional covariance" into "conditional independence," and this holds for every $x_R$, giving Definition 1.24's conditional independence. Reading the correlation off the same $2\times2$ gives $\rho_{ij\cdot R} = -\Lambda_{ij}/\sqrt{\Lambda_{ii}\Lambda_{jj}}$.
**Explicit $4\times4$.** Take the chain precision
$$
\Lambda = \begin{pmatrix}2&-0.8&0&0\\-0.8&2&-0.8&0\\0&-0.8&2&-0.8\\0&0&-0.8&2\end{pmatrix},\qquad
\Sigma = \Lambda^{-1} = \begin{pmatrix}0.6232&0.3079&0.1466&0.0587\\0.3079&0.7698&0.3666&0.1466\\0.1466&0.3666&0.7698&0.3079\\0.0587&0.1466&0.3079&0.6232\end{pmatrix}.
$$
Every entry of $\Sigma$ is nonzero (the smallest in absolute value is $0.058651$), while $\Lambda$ has four zeros: $\Lambda_{13}=\Lambda_{14}=\Lambda_{24}=0$ and their symmetric partners. The graph is the path $1-2-3-4$.
**Numerical check** of $X_1\perp X_3\mid (X_2,X_4)$: the Schur complement is
$$
\Sigma_{\{1,3\}} - \Sigma_{\{1,3\},\{2,4\}}\Sigma_{\{2,4\}}^{-1}\Sigma_{\{2,4\},\{1,3\}} = \begin{pmatrix}0.5&0\\0&0.5\end{pmatrix},
$$
off-diagonal exactly $0$, as predicted — while the marginal correlation of $X_1$ and $X_3$ is $0.1466/\sqrt{0.6232\times0.7698} = 0.2117 \ne 0$.

**C3.** *Thinking:* be explicit about what is assumed, because that is where the derivation is fragile.
**Assumptions.** (A1) weights i.i.d., mean $0$, variance $v_l$, independent across layers; (A2) weights independent of the incoming activations; (A3) activations identically distributed across units within a layer; (A4) pre-activations approximately Gaussian and symmetric about $0$. All four are initialization-time statements.
**Forward.** $z^{(l+1)}_i = \sum_j W_{ij}a_j$. Then $\mathbb{E}[(z_i^{(l+1)})^2] = \sum_{j,k}\mathbb{E}[W_{ij}W_{ik}]\mathbb{E}[a_ja_k] = v_l\sum_j\mathbb{E}[a_j^2] = n_lv_l\mathbb{E}[(a^{(l)})^2]$, since $\mathbb{E}[W_{ij}W_{ik}] = v_l\delta_{jk}$. (Cross terms die from the *weights*, so no independence among activations is needed.)
**First $1/2$.** With $z^{(l)}\approx\mathcal N(0,s^2)$: $\mathbb{E}[\mathrm{ReLU}(z)^2] = \mathbb{E}[z^2\mathbf 1\{z>0\}] = \frac12\mathbb{E}[z^2] = s^2/2$, because $z\mapsto z^2\phi(z)$ is even. Hence $\mathbb{E}[(z^{(l+1)})^2] = \frac{n_lv_l}{2}\mathbb{E}[(z^{(l)})^2]$, preserved iff $v_l = 2/n_l$.
**Backward.** $\delta^{(l)}_j = \phi'(z^{(l)}_j)\sum_iW_{ij}\delta_i^{(l+1)}$, so the same orthogonality gives $\mathbb{E}[(\delta^{(l)})^2] = \mathbb{E}[\phi'(z^{(l)})^2]\,n_{l+1}v_l\,\mathbb{E}[(\delta^{(l+1)})^2]$.
**Second $1/2$.** $\phi'(z)=\mathbf 1\{z>0\}$, so $\phi'(z)^2 = \mathbf 1\{z>0\}$ and $\mathbb{E}[\phi'(z)^2] = P(z>0) = 1/2$ by symmetry. Preservation requires $v_l = 2/n_{l+1}$.
**Compromise.** Forward wants $2/n_{\text{in}}$, backward wants $2/n_{\text{out}}$; Xavier takes the harmonic mean of the *linear*-network targets $1/n_{\text{in}}, 1/n_{\text{out}}$, giving $2/(n_{\text{in}}+n_{\text{out}})$.
**What breaks for tanh.** Near $0$, $\tanh(z)\approx z$, so the linearized target is $\mathbb{E}[\phi(z)^2]=\mathbb{E}[z^2]$, giving $v_l = 1/n_l$. But $|\tanh z| < |z|$ strictly, so the linearization *overestimates* the transmitted second moment and the true recursion contracts. The 20-layer experiment confirms it: with $g=1$ the second moment decays from $0.9992$ to $0.0277$ over 20 layers. The standard fix is a gain of $5/3$ (PyTorch's `calculate_gain('tanh')`), i.e. $v_l = (5/3)^2/n_l$; empirically $g=2$ already stabilizes at $\approx0.62$ rather than collapsing.
**What breaks for sigmoid, and why no $v_l$ fixes it.** Two independent failures. First, $\sigma(z)\in(0,1)$ has mean $\approx0.5$, so assumption (A4)'s zero-symmetry fails at layer 1 and a systematic positive mean propagates, correlating all units and making the "sum of independent zero-mean terms" picture wrong. Second, and fatally, $\sigma'(z) = \sigma(z)(1-\sigma(z))\le1/4$ *always*. So the backward recursion carries a factor $\mathbb{E}[\sigma'(z)^2]\le1/16$ per layer no matter what $v_l$ is — increasing $v_l$ to compensate pushes $z$ into the saturated region where $\sigma'$ is even smaller, making things worse. The weight scale can trade forward against backward, but it cannot raise a ceiling imposed by the activation's derivative. This is the classical vanishing-gradient argument, and it is why the field moved to ReLU (derivative exactly $1$ on half the line, exactly $0$ on the other half) rather than trying harder to initialize sigmoids.

---

### References and further reading

**Textbooks and monographs.**

- P. Billingsley, *Probability and Measure*, 3rd ed., Wiley, 1995. §26 for characteristic functions and the uniqueness theorem; §29 for Cramér–Wold; §36 for the Kolmogorov extension theorem.
- R. Durrett, *Probability: Theory and Examples*, 5th ed., Cambridge, 2019. §3.3 and §3.10 for characteristic functions and multivariate normals; §3.4 for the CLT used in the NNGP discussion.
- R. A. Horn and C. R. Johnson, *Matrix Analysis*, 2nd ed., Cambridge, 2013. Thm 4.1.5 (spectral theorem), Thm 7.2.6 (PSD square root), §0.8.5 (Schur complements and block inversion).
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins, 2013. §4.2 for Cholesky, §2.1.4 for Woodbury. The right reference for *how* to compute anything in this chapter.
- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006. Chapter 2.3 is the standard ML presentation of the Gaussian; §2.3.3 is the linear-Gaussian lemma in the form used here. Chapter 12 covers PPCA and factor analysis.
- K. P. Murphy, *Probabilistic Machine Learning: Advanced Topics*, MIT Press, 2023. Kalman filtering, Gaussian graphical models, and structured Gaussian inference in a modern idiom.
- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006. Chapter 2 is Theorem 5.39 with more discussion; Chapter 4 is the kernel catalogue; Chapter 5 is marginal-likelihood hyperparameter selection. Freely available online.
- S. Särkkä and L. Svensson, *Bayesian Filtering and Smoothing*, 2nd ed., Cambridge, 2023. The Kalman filter derived exactly as in §5.6.2, plus smoothers and nonlinear extensions.
- M. J. Wainwright and M. I. Jordan, *Graphical Models, Exponential Families, and Variational Inference*, FnT ML, 2008. §3 for the exponential-family view of Proposition 5.19; the definitive treatment of Gaussian graphical models.
- I. Karatzas and S. E. Shreve, *Brownian Motion and Stochastic Calculus*, 2nd ed., Springer, 1991. Thm 2.2.8 for the continuous-modification issue flagged after Proposition 5.38.
- T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006. §8.4–8.6 for differential entropy, Thm 8.6.5 for the Gaussian's maximum-entropy property, and §8.5 for the properties of KL used in §5.5.5.
- T. W. Anderson, *An Introduction to Multivariate Statistical Analysis*, 3rd ed., Wiley, 2003. §3.2 for the Gaussian MLE of Proposition 5.7(4), and the classical sampling theory of $\hat\Sigma_n$ (Wishart distributions) that this chapter only gestures at.
- M. L. Stein, *Interpolation of Spatial Data: Some Theory for Kriging*, Springer, 1999. §2.6 for the spectral argument behind §5.8.2's smoothness claims, and the sharpest available case against the RBF kernel.
- B. D. O. Anderson and J. B. Moore, *Optimal Filtering*, Prentice-Hall, 1979 (Dover reprint 2005). Ch. 4 for convergence of the Riccati recursion and the conditions it needs.
- R. Vershynin, *High-Dimensional Probability: An Introduction with Applications in Data Science*, Cambridge, 2018. §3.1 for the sharp concentration of $\|Z\|$ behind §5.3's soap-bubble picture, and Chapter 3 generally for the geometry of high-dimensional Gaussians. Freely available online.

**Machine learning papers.**

- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS 2010. The origin of $2/(n_{\text{in}}+n_{\text{out}})$; note that the derivation is for the *linear* regime, which §5.7.3 makes explicit.
- K. He, X. Zhang, S. Ren and J. Sun, "Delving deep into rectifiers: surpassing human-level performance on ImageNet classification," ICCV 2015. The ReLU correction $2/n_{\text{in}}$ and the backward-pass argument reproduced in Proposition 5.35.
- S. Ioffe and C. Szegedy, "Batch normalization: accelerating deep network training by reducing internal covariate shift," ICML 2015; and J. L. Ba, J. R. Kiros and G. E. Hinton, "Layer normalization," 2016. The architectural answer to Common Confusion 5.7's point that initialization-time guarantees do not survive training.
- M. E. Tipping and C. M. Bishop, "Probabilistic principal component analysis," JRSS-B 61(3), 1999. Proposition 5.32's model, the PPCA/PCA limit, and the rotational non-identifiability.
- R. M. Neal, *Bayesian Learning for Neural Networks*, Springer, 1996, Ch. 2; J. Lee et al., "Deep neural networks as Gaussian processes," ICLR 2018; A. G. de G. Matthews et al., "Gaussian process behaviour in wide deep neural networks," ICLR 2018. The NNGP correspondence.
- A. Jacot, F. Gabriel and C. Hongler, "Neural tangent kernel: convergence and generalization in neural networks," NeurIPS 2018; and G. Yang and E. J. Hu, "Feature learning in infinite-width neural networks," ICML 2021, for the honest limitations of the lazy regime.
- J. Hensman, N. Fusi and N. D. Lawrence, "Gaussian processes for big data," UAI 2013; and J. R. Gardner et al., "GPyTorch: blackbox matrix-matrix Gaussian process inference with GPU acceleration," NeurIPS 2018. Two different escapes from the $O(n^3)$ wall of §5.8.4.
- J. Friedman, T. Hastie and R. Tibshirani, "Sparse inverse covariance estimation with the graphical lasso," Biostatistics 9(3), 2008. Theorem 5.25 turned into an estimator, and the answer to Exercise B5.
- K. Lee, K. Lee, H. Lee and J. Shin, "A simple unified framework for detecting out-of-distribution samples and adversarial attacks," NeurIPS 2018. Mahalanobis distance (Definition 5.12) as an OOD score, as discussed in Worked Example 5.6.
- O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," *J. Multivariate Analysis* 88(2), 2004. The standard shrinkage fix for Proposition 5.7(3), and what to reach for when $n$ is not comfortably larger than $d$.
- I. Higgins et al., "beta-VAE: learning basic visual concepts with a constrained variational framework," ICLR 2017; and S. Bowman et al., "Generating sentences from a continuous space," CoNLL 2016. The $\beta$ weighting and KL annealing referred to in §5.5.5, both responses to the collapse-rewarding structure of the Gaussian KL term.
- A. Kessy, A. Lewin and K. Strimmer, "Optimal whitening and decorrelation," *The American Statistician* 72(4), 2018. The ZCA optimality claim of §5.2.1, plus a systematic comparison of the whitening transforms of Definition 5.9.
- F. Locatello et al., "Challenging common assumptions in the unsupervised learning of disentangled representations," ICML 2019. A rigorous version of Common Confusion 5.2's point: without inductive bias or supervision, disentanglement is not identifiable.
