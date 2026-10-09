# Chapter 3 — Transformations and Distribution Families

## Chapter 3 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 3.1 | The CDF method and the monotone change-of-variables formula in one dimension | The reflex you use every time a network layer, link function, or preprocessing step transforms a random quantity |
| 3.2 | Non-injective maps: sum over branches; atoms and flat regions | The $\chi^2$ distribution; the exact law of ReLU activations, derived rather than memorized |
| 3.3 | Sums of independent variables: convolution | Why adding noise smooths distributions (instance noise for GANs); the algebra behind "sum of Gaussians is Gaussian" |
| 3.4 | Order statistics and extremes | Why maxima concentrate at the edge of the support; where the Gumbel distribution comes from |
| 3.5 | The multivariate change of variables and the Jacobian determinant | The single formula underneath every normalizing flow; where $(\det\Sigma)^{-1/2}$ in the Gaussian density comes from; Box–Muller |
| 3.6 | Inverse transform sampling and the probability integral transform | How samplers actually work; quantile preprocessing; the converse of Proposition 1.19, finally proved |
| 3.7 | Normalizing flows | Exact log-likelihood generative models; coupling layers, autoregressive flows, and the engineering of tractable log-determinants |
| 3.8 | The exponential family | Softmax, cross-entropy, and logistic regression as one theorem; MLE = moment matching, and when it fails (separable data) |
| 3.9 | Sufficiency and the Fisher–Neyman factorization | When a summary statistic loses nothing; why heavy-tailed models resist summarization |
| 3.10 | The distribution zoo, organized by use | Knowing which distribution to reach for, and why, instead of defaulting to Gaussian |
| 3.11 | The Gumbel-max trick and Gumbel-Softmax | Reparameterizing discrete random variables; closing Chapter 2's REINFORCE-vs-reparameterization gap |

## 3.0 From one distribution, all the others

Chapters 1 and 2 built the machinery: probability spaces, random variables as deterministic functions of a random seed $\omega$, laws as pushforwards (Definition 1.16), the Lebesgue integral and expectation (Ch. 2), and densities as Radon–Nikodym derivatives with respect to a stated base measure (Ch. 2). What we have not yet done is *compute* with that machinery at scale. This chapter is where the machinery earns its keep. The organizing question is simple: **if I know the distribution of $X$, what is the distribution of $g(X)$?**

That question is the whole of generative modeling in miniature. Chapter 1 established that every deep generative model is a pushforward $P_\theta = (G_\theta)_\# \mathcal{N}(0, I_k)$ of a simple latent distribution through a network. To train such a model by maximum likelihood you need the *density* of the pushforward, not just its abstract existence — and Chapter 1 also proved the discouraging fact that when $k < d$ the pushforward has no Lebesgue density at all, because its support is Lebesgue-null. Sections 3.1–3.5 develop, with full proofs where our toolkit allows, exactly when and how the density of $g(X)$ can be written down. Section 3.7 then assembles those formulas into **normalizing flows**: generative models that keep $k = d$ and keep $g$ invertible *precisely so that* the change-of-variables formula applies and the log-likelihood is exact. The measure-zero trap of Chapter 1 is not a curiosity; it is the design constraint that shaped an entire model family.

The second half of the chapter changes direction. Instead of asking what happens to one distribution under a map, it asks how distributions organize into *families* — parameterized sets $\{P_\theta\}$ — and what structure makes a family pleasant to estimate. The answer, to a remarkable degree, is one structure: the **exponential family** (Section 3.8). Softmax classification, logistic regression, Poisson regression, and Gaussian maximum likelihood are all the same computation wearing different clothes, and the log-partition function $A$ is the object that unifies them: its gradient gives means, its Hessian gives covariances, its convexity gives well-posed optimization, and its Legendre transform gives the duality between natural and mean parameters. Section 3.9 adds the concept of **sufficiency** — when does a statistic $T(x)$ capture everything the data says about $\theta$? — and Section 3.10 tours the standard distributions with a persistent, practical question attached to each: *when would I actually reach for this?* Section 3.11 closes with the Gumbel-max trick, which repays a debt from Chapter 2: it supplies the reparameterization for *discrete* random variables that the REINFORCE-versus-reparameterization comparison left wanting.

What you are assumed to bring: from Chapter 1, the pushforward definition of a law (Definition 1.16), supports (Definition 1.17), CDFs and their properties (Definition 1.18, Proposition 1.19), the fact that a CDF determines a law (Theorem 1.20), and Doob–Dynkin (Lemma 1.22). From Chapter 2, cited by name throughout: the Lebesgue integral and its convergence theorems (Monotone and Dominated Convergence), differentiation under the integral sign, Fubini–Tonelli, Hölder's inequality, absolute continuity and the Radon–Nikodym theorem, and the KL divergence. We will not re-derive any of these.

Two honesty flags, so you know the shape of the terrain in advance. First: the multivariate change-of-variables theorem for general smooth maps is a genuinely hard theorem of real analysis. We will *prove* it completely for linear and affine maps — which already explains where the Jacobian determinant comes from and settles every Gaussian computation — and we will explain the general case by linearization, with the full theorem stated precisely and logged in the "Results used but not proved" table. Second: the fully general theory of sufficiency requires conditional distributions, which a later chapter constructs. We will define and prove everything completely in the discrete case, adopt the factorization criterion as the working definition in general, and log the equivalence. Nothing will be hand-waved; where we borrow, we will say so out loud.

The punchline, stated in advance: **densities transform by the inverse Jacobian determinant; log-likelihoods of flows are exact because of it; exponential families make maximum likelihood into moment matching; and one Gumbel identity turns discrete sampling into a differentiable computation.** Everything else in the chapter is the careful working-out of those four sentences.

## 3.1 Transformations in one dimension

### 3.1.1 The CDF method: the reflex that always works

Let $X$ be a real random variable and $g:\mathbb{R}\to\mathbb{R}$ a Borel-measurable function (Proposition 1.15 guarantees $Y = g(X)$ is again a random variable). The law of $Y$ is, by Definition 1.16, the pushforward $P_Y = g_\# P_X$: for every Borel set $B$,

$$P_Y(B) = P_X(g^{-1}(B)).$$

This is always true and requires nothing of $g$ beyond measurability. In one dimension, Theorem 1.20 tells us the CDF determines the law, so it suffices to compute

$$F_Y(y) = P(g(X) \le y) = P_X\big(\{x : g(x) \le y\}\big),$$

which is a probability that $X$ lands in some set — a computation about $X$, which we understand. This is the **CDF method**, and it should be your default reflex: it works for non-monotone $g$, for discontinuous $g$, for $X$ with atoms, for everything. Density formulas, which we derive next, are shortcuts valid under smoothness hypotheses; the CDF method is the ground truth they are shortcuts *to*. When in doubt, or when a formula's hypotheses are unclear, fall back to the CDF.

**Worked Example 3.1 (affine maps).** Let $X$ have CDF $F_X$ and let $Y = aX + b$ with $a > 0$. Then

$$F_Y(y) = P(aX + b \le y) = P\!\left(X \le \tfrac{y-b}{a}\right) = F_X\!\left(\tfrac{y-b}{a}\right).$$

If $X$ has a density $p_X$ (with respect to Lebesgue measure — we will always state the base measure, per Chapter 2's discipline), differentiating gives

$$p_Y(y) = \frac{1}{a}\,p_X\!\left(\frac{y-b}{a}\right).$$

For $a<0$ the inequality flips: $F_Y(y) = P(X \ge (y-b)/a) = 1 - F_X\big((\tfrac{y-b}{a})^-\big)$, and differentiating (where $p_X$ is continuous) gives $p_Y(y) = \frac{1}{|a|}p_X(\frac{y-b}{a})$. Note the two ingredients that will recur all chapter: the argument is pushed through the *inverse* map, and the density is rescaled by a *derivative* factor — here $1/|a|$ — that keeps total mass equal to $1$. Stretching an interval by $a$ dilutes the probability lying over it by $1/|a|$. $\square$

**Worked Example 3.2 (the log-normal).** Let $Z \sim \mathcal{N}(0,1)$ and $Y = e^{\sigma Z + \mu}$, the *log-normal* distribution — the default model for quantities that are products of many small independent multiplicative effects (asset prices, cell sizes, training-loss spikes). For $y > 0$:

$$F_Y(y) = P(e^{\sigma Z + \mu} \le y) = P\!\left(Z \le \frac{\log y - \mu}{\sigma}\right) = \Phi\!\left(\frac{\log y - \mu}{\sigma}\right),$$

and differentiating with the chain rule,

$$p_Y(y) = \frac{1}{y\sigma}\,\varphi\!\left(\frac{\log y - \mu}{\sigma}\right) = \frac{1}{y\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\log y - \mu)^2}{2\sigma^2}\right), \qquad y > 0.$$

The factor $1/y$ is the derivative of $\log y$ — the now-familiar rescaling. Notice what it does to the right tail: $p_Y$ decays like $e^{-(\log y)^2/2\sigma^2}/y$, which is slower than *any* exponential $e^{-cy}$. The log-normal is **heavy-tailed** relative to the Gaussian even though it is built from one. Exponentiating a light-tailed variable manufactures heavy tails; keep this in mind whenever a model parameterizes a positive quantity as $e^{\text{network output}}$, as flows and variational autoencoders routinely do for scale parameters. $\square$

### 3.1.2 The monotone density formula

The pattern in both examples — invert the map, rescale by a derivative — is a theorem.

**Theorem 3.1 (change of variables, one dimension, monotone case).** Let $I \subseteq \mathbb{R}$ be an open interval, let $X$ take values in $I$ with density $p_X$ with respect to Lebesgue measure, and let $g : I \to J$ be continuously differentiable and strictly monotone with $g'(x) \ne 0$ for all $x \in I$. Then $Y = g(X)$ has a density on $J = g(I)$ given by

$$p_Y(y) \;=\; p_X\big(g^{-1}(y)\big)\,\big|(g^{-1})'(y)\big| \;=\; \frac{p_X\big(g^{-1}(y)\big)}{\big|g'\big(g^{-1}(y)\big)\big|}, \qquad y \in J.$$

*Proof.* Suppose first $g$ is strictly increasing. By the inverse function theorem (calculus), $g^{-1}: J \to I$ exists, is continuously differentiable, and $(g^{-1})'(y) = 1/g'(g^{-1}(y)) > 0$. For $y \in J$,

$$F_Y(y) = P(g(X)\le y) = P\big(X \le g^{-1}(y)\big) = F_X\big(g^{-1}(y)\big),$$

where the middle equality uses that $g$ is increasing, so $g(x) \le y \iff x \le g^{-1}(y)$. Now we must differentiate. At any $y$ where $p_X$ is continuous at $g^{-1}(y)$, $F_X$ is differentiable there with derivative $p_X$ (fundamental theorem of calculus, since $F_X(t) = \int_{-\infty}^t p_X\,d\lambda$ and the integrand is continuous at the point), and the chain rule gives

$$F_Y'(y) = p_X\big(g^{-1}(y)\big)\,(g^{-1})'(y).$$

If $p_X$ is continuous on $I$, this identifies $F_Y'$ everywhere on $J$, and since $F_Y(y) = \int_{\inf J}^{y} F_Y'\,d\lambda$ (again the fundamental theorem, as $F_Y'$ is continuous), the function $F_Y'$ is a density for $Y$. The decreasing case is identical with $F_Y(y) = 1 - F_X(g^{-1}(y)^-) = 1 - F_X(g^{-1}(y))$ (no atom, since $X$ has a density), producing $-(g^{-1})' = |(g^{-1})'|$. $\blacksquare$

Two remarks on rigor, because a self-studying reader deserves to know exactly what was and wasn't proved. First, the proof as written assumes $p_X$ continuous; for a merely measurable density the same formula holds for Lebesgue-almost-every $y$, which is all a density is ever required to satisfy. That stronger statement is the $d=1$ case of the general change-of-variables theorem (Theorem 3.8 below), which we state but do not prove; the table in Section 3.12 records the debt. Second, memorize the formula in its *inverse-derivative* form, $p_X(g^{-1}(y))|(g^{-1})'(y)|$: it generalizes verbatim to higher dimensions with $|(g^{-1})'|$ replaced by $|\det J_{g^{-1}}|$, whereas the "divide by $g'$" form invites sign and direction errors.

> **Common Confusion 3.1.** *"To get the density of $Y = g(X)$, I just substitute: $p_Y(y) = p_X(g^{-1}(y))$. The derivative factor seems like bookkeeping."*
>
> Substitution alone is wrong, and not by a little. Try it on Worked Example 3.1 with $Y = X/10$, $X \sim \mathrm{Unif}[0,1]$: substitution predicts $p_Y \equiv 1$ on $[0, 0.1]$, which integrates to $0.1$, not $1$. The correct density is $10$ on $[0,0.1]$. The derivative factor is not bookkeeping — it is the entire content of the theorem. A density is a *rate* (probability per unit length, Ch. 2), and when $g$ compresses lengths, the rate must go up to conserve total probability. Forgetting the Jacobian factor is the single most common error in derivations of flow likelihoods, and it silently corrupts the loss rather than crashing, which is what makes it dangerous.

**Worked Example 3.3 (the square on $[0,1]$, and a density that blows up).** Chapter 1 introduced the probability space $\Omega = [-1,1]$ with normalized Lebesgue measure and the random variable $X(\omega) = \omega^2$, promising that its density would eventually fall out of a formula. We can now keep half of that promise — and the restriction we impose is instructive. On $\Omega = [0,1]$ with $P = \lambda|_{[0,1]}$ (so $\omega \sim \mathrm{Unif}[0,1]$, density $p(\omega) = 1$), the map $g(\omega) = \omega^2$ is strictly increasing and $C^1$ on the open interval $(0,1)$ with $g'(\omega) = 2\omega \neq 0$ there, so Theorem 3.1 applies with $g^{-1}(b) = \sqrt{b}$:

$$p_X(b) = p\big(\sqrt{b}\big) \cdot \left|\frac{d}{db}\sqrt{b}\right| = \frac{1}{2\sqrt{b}}, \qquad b \in (0,1).$$

Sanity check: $\int_0^1 \frac{1}{2\sqrt b}\,db = \big[\sqrt b\big]_0^1 = 1$. ✓ The density is **unbounded** near $0$ — it blows up like $b^{-1/2}$ — and yet perfectly integrable. Where does the blow-up come from? From $|g'(\omega)| = 2\omega \to 0$ as $\omega \to 0$: the map $\omega \mapsto \omega^2$ *compresses* the neighborhood of $0$ violently (the whole interval $[0, 0.1]$ of $\omega$-values, carrying probability $0.1$, is squeezed into $[0, 0.01]$ of $b$-values), and probability per unit length must spike to compensate. This is Chapter 1's Failure 1 — "densities can exceed 1" — in its sharpest form: a density can exceed *every* bound and still be a legitimate Radon–Nikodym derivative (Ch. 2). Whenever a transformation has a critical point ($g' = 0$) inside the support, expect an integrable singularity in the output density at the image of that point. $\square$

But the original example lived on $[-1,1]$, where $g(\omega) = \omega^2$ is *not* monotone: both $\omega$ and $-\omega$ map to the same $b$. Theorem 3.1's hypotheses genuinely fail, and applying its formula anyway gives an answer that integrates to $\tfrac12$ — off by exactly the missed second branch. Handling that correctly is the business of the next section.

<svg viewBox="0 0 720 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img" aria-label="A monotone map stretching and compressing a density">
  <style>text{font-family:sans-serif;font-size:13px;fill:currentColor}.cap{font-size:12px}</style>
  <!-- axes left panel: x-space density -->
  <line x1="50" y1="230" x2="310" y2="230" stroke="currentColor" stroke-width="1.5"/>
  <line x1="50" y1="230" x2="50" y2="60" stroke="currentColor" stroke-width="1.5"/>
  <text x="300" y="248">x</text>
  <text x="38" y="58">p_X</text>
  <!-- uniform density on [0,1] mapped to [60,300] -->
  <rect x="60" y="140" width="240" height="90" fill="#3b82f6" opacity="0.25"/>
  <line x1="60" y1="140" x2="300" y2="140" stroke="#3b82f6" stroke-width="2.5"/>
  <text x="130" y="130" fill="#3b82f6">p_X = 1 on [0,1]</text>
  <!-- highlight two equal-width x-intervals -->
  <rect x="80" y="140" width="30" height="90" fill="#10b981" opacity="0.5"/>
  <rect x="240" y="140" width="30" height="90" fill="#ef4444" opacity="0.5"/>
  <!-- right panel: y-space density 1/(2 sqrt b) -->
  <line x1="410" y1="230" x2="690" y2="230" stroke="currentColor" stroke-width="1.5"/>
  <line x1="410" y1="230" x2="410" y2="40" stroke="currentColor" stroke-width="1.5"/>
  <text x="680" y="248">b</text>
  <text x="398" y="38">p_B</text>
  <path d="M 418 42 Q 430 110 460 150 Q 510 195 660 212" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
  <text x="470" y="105" fill="#3b82f6">p_B(b) = 1/(2√b)</text>
  <!-- images of intervals: near 0 compressed (green -> tall thin), near 1 stretched (red -> wide short) -->
  <rect x="418" y="60" width="14" height="170" fill="#10b981" opacity="0.5"/>
  <rect x="580" y="196" width="70" height="34" fill="#ef4444" opacity="0.5"/>
  <!-- arrows -->
  <path d="M 315 150 C 350 120 380 120 408 130" fill="none" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr)"/>
  <path d="M 315 200 C 360 230 400 235 575 215" fill="none" stroke="#ef4444" stroke-width="1.5" marker-end="url(#arr)"/>
  <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="currentColor"/></marker></defs>
  <text x="330" y="112" fill="#10b981">g compresses near 0: density spikes</text>
  <text x="430" y="262" fill="#ef4444">g stretches near 1: density dilutes</text>
  <text x="50" y="292" class="cap">Figure 3.1 — The map g(ω)=ω² sends equal-width intervals of a uniform density (left) to unequal-width</text>
  <text x="50" y="308" class="cap">images (right). Compressed regions must carry the same probability over less length, so the density 1/(2√b) blows up at 0.</text>
</svg>

> **Self-check 3.1.** (a) Why is the CDF method more general than the density formula of Theorem 3.1 — name two situations the CDF method handles that the formula does not. (b) For $Y = e^X$ with $X$ having density $p_X$, what is $p_Y(y)$, and what is the derivative factor? (c) A colleague computes the density of $Y = g(X)$ and finds it integrates to $0.5$. Assuming their algebra within each step is right, what structural mistake should you suspect? *(Answers in Appendix A.)*

## 3.2 Non-injective maps, atoms, and flat regions

### 3.2.1 Summing over branches

When $g$ is not injective, several $x$-values fold onto each $y$, and each contributes probability. The fix is exactly what intuition suggests: apply the monotone formula on each piece where $g$ *is* monotone, and add.

**Theorem 3.2 (change of variables with branches).** Let $X$ have density $p_X$ with respect to Lebesgue measure on $\mathbb{R}$. Suppose there are countably many disjoint open intervals $I_1, I_2, \dots$ with $P(X \in \bigcup_k I_k) = 1$, such that on each $I_k$ the map $g$ is $C^1$, strictly monotone, and $g' \ne 0$. Write $g_k := g|_{I_k}$. Then $Y = g(X)$ has density

$$p_Y(y) \;=\; \sum_{k}\; \frac{p_X\big(g_k^{-1}(y)\big)}{\big|g'\big(g_k^{-1}(y)\big)\big|}\;\mathbf{1}\{y \in g(I_k)\} \;=\; \sum_{x\,\in\, g^{-1}(y)} \frac{p_X(x)}{|g'(x)|}.$$

*Proof.* Fix a Borel set $B$. The events $\{X \in I_k \cap g^{-1}(B)\}$ are disjoint and their union carries all the probability of $\{Y \in B\}$, so by countable additivity (Definition 1.7),

$$P(Y \in B) = \sum_k P\big(X \in I_k,\; g(X) \in B\big).$$

The $k$-th term is $P(X_k' \in B)\cdot$(mass of $I_k$) in disguise: define the sub-probability density $p_X \mathbf{1}_{I_k}$; Theorem 3.1 applies verbatim to $g_k$ on $I_k$ (its proof never used that the density integrates to exactly $1$), giving

$$P\big(X \in I_k,\; g(X)\in B\big) = \int_{B \cap g(I_k)} \frac{p_X\big(g_k^{-1}(y)\big)}{\big|g'(g_k^{-1}(y))\big|}\,dy.$$

Summing over $k$ and interchanging the sum with the integral — legitimate by Tonelli's theorem (Ch. 2), since every term is nonnegative — yields $P(Y \in B) = \int_B p_Y\,d\lambda$ with $p_Y$ as displayed. $\blacksquare$

Note where countability earned its keep, continuing a Chapter 1 theme: countable additivity justified the decomposition, and Tonelli justified the swap. With uncountably many branches neither step is available.

**Worked Example 3.4 (the square on $[-1,1]$, completed).** Return to $\Omega = [-1,1]$, $P = \tfrac{1}{2}\lambda|_{[-1,1]}$ (density $p(\omega) = \tfrac12$), and $X(\omega) = \omega^2$. Take $I_1 = (-1, 0)$ and $I_2 = (0,1)$; the point $\{0\}$ and endpoints are Lebesgue-null, so the hypothesis $P(\omega \in I_1 \cup I_2)=1$ holds. For $b \in (0,1)$ the preimages are $-\sqrt b \in I_1$ and $\sqrt b \in I_2$, and $|g'(\pm\sqrt b)| = 2\sqrt b$ at both. Theorem 3.2 gives

$$p_X(b) = \frac{1/2}{2\sqrt b} + \frac{1/2}{2\sqrt b} = \frac{1}{2\sqrt b}, \qquad b \in (0,1)$$

— the same $\tfrac{1}{2\sqrt b}$ as in Worked Example 3.3, and now we can see why: the halved density (mass spread over an interval twice as long) is exactly compensated by the doubled branch count, by the symmetry $\omega \mapsto -\omega$. Chapter 1's promise about this example is now fully discharged: the density comes from a formula, and its blow-up at $0$ has a name and a cause ($|g'| \to 0$ at the fold point). $\square$

> **Common Confusion 3.2.** *"Theorem 3.1's formula, applied blindly to a two-to-one map, gives an answer that looks like a density. How would I even notice the mistake?"*
>
> You notice because it fails the only unconditional test a density must pass: integrating to $1$. Apply the monotone formula to $\omega^2$ on all of $[-1,1]$ using only the branch $g^{-1}(b) = \sqrt b$: you get $\tfrac{1/2}{2\sqrt b} = \tfrac{1}{4\sqrt b}$, which integrates to $\tfrac12$ — you have silently discarded the negative branch and half the probability with it. *Always integrate your candidate density* (analytically or numerically) before trusting it. In flow implementations the same bug appears as a model that assigns half (or $1/|{\text{branches}}|$) of the correct likelihood, trains anyway, and quietly reports wrong log-likelihoods in the paper's table. Invertibility of every layer is what licenses flows to skip the branch sum; break invertibility and the reported likelihood is simply wrong.

**Worked Example 3.5 (the $\chi^2_1$ distribution, computed twice).** Let $Z \sim \mathcal{N}(0,1)$ with density $\varphi(z) = \tfrac{1}{\sqrt{2\pi}}e^{-z^2/2}$, and let $Y = Z^2$. This is the same two-branch geometry as the previous example, with branches $\pm\sqrt y$ and $|g'(\pm\sqrt y)| = 2\sqrt y$.

*Via Theorem 3.2:*
$$p_Y(y) = \frac{\varphi(\sqrt y)}{2\sqrt y} + \frac{\varphi(-\sqrt y)}{2\sqrt y} = \frac{2\varphi(\sqrt y)}{2\sqrt y} = \frac{1}{\sqrt{2\pi y}}\,e^{-y/2}, \qquad y > 0,$$
using the symmetry $\varphi(-z) = \varphi(z)$.

*Via the CDF method, as a check:* for $y > 0$,
$$F_Y(y) = P(-\sqrt y \le Z \le \sqrt y) = \Phi(\sqrt y) - \Phi(-\sqrt y) = 2\Phi(\sqrt y) - 1,$$
$$p_Y(y) = F_Y'(y) = 2\varphi(\sqrt y)\cdot \frac{1}{2\sqrt y} = \frac{1}{\sqrt{2\pi y}}e^{-y/2}.\ \checkmark$$

The two routes agree, as they must — and running both is cheap insurance you should buy whenever the stakes are high. The result is the **chi-squared distribution with one degree of freedom**, $\chi^2_1$. Writing $\tfrac{1}{\sqrt{2\pi y}}e^{-y/2} = \tfrac{(1/2)^{1/2}}{\Gamma(1/2)}\,y^{1/2 - 1}e^{-y/2}$ (using $\Gamma(\tfrac12) = \sqrt\pi$) identifies it as the $\mathrm{Gamma}(\tfrac12, \tfrac12)$ distribution — an identification the zoo of Section 3.10 will use. Like Worked Example 3.4, the density blows up at $0$: the parabola is flat at its vertex, so $|g'| \to 0$ there. Numerically, $P(Z^2 \le 1) = 2\Phi(1) - 1 = 0.6827$, matching the $\chi^2_1$ CDF at $1$ to four decimals — the familiar "68% within one sigma" fact, transformed. $\square$

<svg viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img" aria-label="Two branches of y equals x squared">
  <style>text{font-family:sans-serif;font-size:13px;fill:currentColor}.cap{font-size:12px}</style>
  <!-- axes -->
  <line x1="360" y1="270" x2="360" y2="30" stroke="currentColor" stroke-width="1.5"/>
  <line x1="80" y1="240" x2="640" y2="240" stroke="currentColor" stroke-width="1.5"/>
  <text x="630" y="258">x</text>
  <text x="368" y="40">y = x²</text>
  <!-- parabola: x from -240..240 px maps, y = x^2 scaled -->
  <path d="M 120 40 Q 360 440 600 40" fill="none" stroke="currentColor" stroke-width="2"/>
  <!-- horizontal slice at some y -->
  <line x1="120" y1="120" x2="640" y2="120" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="6 4"/>
  <text x="646" y="124" fill="#f59e0b">y</text>
  <!-- intersection points approx: solve on drawn parabola; place markers symmetric -->
  <circle cx="215" cy="120" r="5" fill="#3b82f6"/>
  <circle cx="505" cy="120" r="5" fill="#10b981"/>
  <text x="150" y="106" fill="#3b82f6">x = −√y</text>
  <text x="515" y="106" fill="#10b981">x = +√y</text>
  <!-- vertical drop lines to x axis -->
  <line x1="215" y1="120" x2="215" y2="240" stroke="#3b82f6" stroke-width="1.2" stroke-dasharray="3 3"/>
  <line x1="505" y1="120" x2="505" y2="240" stroke="#10b981" stroke-width="1.2" stroke-dasharray="3 3"/>
  <!-- slope annotations -->
  <text x="90" y="170" fill="#ef4444">|g'| small near vertex ⇒</text>
  <text x="90" y="188" fill="#ef4444">density of y blows up at 0</text>
  <circle cx="360" cy="240" r="4" fill="#ef4444"/>
  <text x="60" y="300" class="cap">Figure 3.2 — Each level y &gt; 0 of the map g(x)=x² has two preimages, ±√y; both branches contribute p_X(x)/|g'(x)|</text>
  <text x="60" y="316" class="cap">to the density of Y. The flat vertex (|g'|→0, red) is where the output density develops its integrable singularity.</text>
</svg>

### 3.2.2 The general recipe: branches, atoms, and flat regions together

Real maps in ML mix all three behaviors. ReLU, $g(z) = \max(0, z)$, is the canonical case: strictly increasing on $(0,\infty)$, *constant* on $(-\infty, 0)$. A flat region is a place where $g$ crushes an interval — a set of positive probability — onto a single point, and a single point of positive probability is by definition an **atom** (Proposition 1.19 connected atoms to CDF jumps). No Lebesgue density can represent an atom (Ch. 2: an atom is singular with respect to $\lambda$). The general recipe:

1. **Partition the domain** into (countably many) pieces where $g$ is strictly monotone $C^1$, pieces where $g$ is constant, and a Lebesgue-null leftover.
2. **Monotone pieces** contribute an absolutely continuous part via Theorem 3.2.
3. **Each constant piece** $g \equiv c$ on $C$ contributes an atom $P(X \in C)\,\delta_c$.
4. **Report the law as a mixture**, and its density with respect to a base measure that can see both parts — typically $\lambda + \sum_j \delta_{c_j}$ (Ch. 2 established that densities are always relative to a stated base measure, and that this is precisely the situation Radon–Nikodym was built for).

**Worked Example 3.6 (ReLU's mixed law, now from the recipe).** Let $Z \sim \mathcal{N}(0,1)$ and $A = \max(0, Z)$ — Chapter 1's running example (ii), the variable with "neither a pmf nor a pdf," whose resolution Chapter 2 promised via base measures. The recipe: on $(0,\infty)$, $g$ is the identity (monotone, $g' = 1$), contributing density $\varphi(a)$ for $a > 0$; on $(-\infty, 0)$, $g \equiv 0$, contributing the atom $P(Z < 0)\,\delta_0 = \tfrac12 \delta_0$. So

$$P_A = \tfrac{1}{2}\,\delta_0 \;+\; \varphi\,\mathbf{1}_{(0,\infty)}\cdot\lambda,$$

and with respect to the base measure $\mu := \delta_0 + \lambda$, the Radon–Nikodym derivative (Ch. 2) is

$$\frac{dP_A}{d\mu}(a) = \begin{cases} \tfrac12 & a = 0,\\[2pt] \varphi(a) & a > 0,\\[2pt] 0 & a<0.\end{cases}$$

Every expectation now runs through one integral: $\mathbb{E}[A] = \int a \,\tfrac{dP_A}{d\mu}\,d\mu = \tfrac12\cdot 0 + \int_0^\infty a\varphi(a)\,da = \tfrac{1}{\sqrt{2\pi}} \approx 0.3989$, recovering Chapter 1's value with no case analysis. This is worth internalizing as an ML fact, not just an exercise: **the post-activation distribution of a ReLU unit with roughly centered pre-activations is half point-mass, half half-Gaussian.** That atom at zero is the "dying ReLU" phenomenon in embryo, it is why ReLU networks' activation statistics are not Gaussian no matter what the pre-activations are, and it is why any "density" analysis of ReLU networks that assumes Lebesgue densities throughout is broken from the first line. $\square$

> **Self-check 3.2.** (a) For $Y = |X|$ with $X$ having density $p_X$, write the branch formula. (b) In Worked Example 3.4, what geometric feature of $g$ made the two branch contributions equal, and when would they differ? (c) What base measure would you use for the law of $g(Z) = \min(1, \max(0, Z))$, $Z \sim \mathcal{N}(0,1)$ (a "hard sigmoid"), and what does the density look like relative to it? *(Answers in Appendix A.)*

## 3.3 Sums of independent random variables: convolution

Sums are the transformation you will use most, yet they do not fit the previous sections' template: $(x, y) \mapsto x + y$ maps two dimensions to one, so it has no inverse and no branch decomposition. What it does have is Fubini.

**Definition 3.3 (convolution).** For independent random variables $X \perp Y$ with laws $P_X, P_Y$, the **convolution** $P_X * P_Y$ is the law of $X + Y$: the pushforward of the product measure $P_X \otimes P_Y$ (Ch. 2, Fubini–Tonelli's setting) under the addition map $s(x,y) = x + y$. Independence is what licenses the product-measure description of the pair $(X, Y)$ — without it, "the law of $X+Y$" still exists but is not determined by the two marginals at all.

**Theorem 3.4 (density of a sum).** Let $X \perp Y$. If $X$ has density $p_X$ with respect to Lebesgue measure, then $X + Y$ has a Lebesgue density regardless of what $P_Y$ is:

$$p_{X+Y}(s) \;=\; \int p_X(s - y)\; dP_Y(y).$$

In particular, if $Y$ also has a Lebesgue density $p_Y$, then $p_{X+Y}(s) = \int p_X(s-y)\,p_Y(y)\,dy =: (p_X * p_Y)(s)$; and if both are integer-valued with pmfs (densities w.r.t. counting measure), $p_{X+Y}(s) = \sum_{k} p_X(s-k)\,p_Y(k)$.

*Proof.* For Borel $B \subseteq \mathbb{R}$, using the product-measure description and Tonelli (Ch. 2; everything in sight is nonnegative, so no integrability check is needed):

$$P(X + Y \in B) = \iint \mathbf{1}_B(x+y)\,p_X(x)\,dx\;dP_Y(y).$$

Fix $y$ and substitute $s = x + y$ in the inner integral — a shift, which preserves Lebesgue measure (translation invariance, provable directly from the definition of $\lambda$ on intervals and Theorem 1.20's uniqueness; it is also the $T = \mathrm{Id}$, $b = y$ case of Theorem 3.7 below):

$$\int \mathbf{1}_B(x + y)\,p_X(x)\,dx = \int_B p_X(s - y)\,ds.$$

Then Tonelli again to swap the remaining two integrals:

$$P(X+Y \in B) = \int\!\!\int_B p_X(s-y)\,ds\,dP_Y(y) = \int_B \left[\int p_X(s-y)\,dP_Y(y)\right] ds. \qquad \blacksquare$$

Read the hypotheses again: only **one** summand needed a density. This is the theorem behind the slogan **convolution smooths**. Adding *any* independent noise with a density to *any* random variable — even a discrete one, even one supported on a measure-zero manifold's coordinate — produces something with a density. The sum is always at least as regular as its smoothest ingredient.

That observation has a direct ML payoff, and it closes a loop from Chapter 1. Failure 4 diagnosed why GAN training stalls: generator and data distributions with disjoint (Lebesgue-null) supports make the Jensen–Shannon divergence lock at $\log 2$, so the loss carries no gradient signal. Theorem 3.4 is the mathematics behind one published fix, **instance noise**: add independent Gaussian noise to both real and generated samples. Both blurred distributions instantly acquire full-support Lebesgue densities, the supports overlap everywhere, and the divergence between them becomes informative. The noise scale is then annealed as training proceeds. The remedy is not a heuristic — it is Theorem 3.4 applied twice.

**Worked Example 3.7 (Uniform + Uniform = triangular).** Let $X, Y \overset{\text{i.i.d.}}{\sim} \mathrm{Unif}[0,1]$. Then $p_{X+Y}(s) = \int_0^1 \mathbf{1}\{0 \le s - y \le 1\}\,dy = \lambda\big([s-1, s] \cap [0,1]\big)$, an overlap length:

$$p_{X+Y}(s) = \begin{cases} s & 0 \le s \le 1,\\ 2 - s & 1 \le s \le 2,\\ 0 & \text{else,}\end{cases}$$

the **triangular density**. A flat density in, a tent out: one convolution already manufactures a peak. (Convolve again and you get a piecewise-cubic bump that is visibly bell-shaped — a first shadow of the Central Limit Theorem, which Chapter 5 proves.) *Verification (Lab B.5, seed 0):* $10^6$ simulated sums, binned into 20 equal bins on $[0,2]$, matched the exact bin probabilities with maximum absolute error $0.00065$; e.g., the first five empirical bin probabilities were $(0.0050, 0.0150, 0.0251, 0.0349, 0.0450)$ against exact $(0.0050, 0.0150, 0.0250, 0.0350, 0.0450)$. $\square$

**Worked Example 3.8 (sum of independent Gaussians).** Let $X \sim \mathcal{N}(\mu_1, \sigma_1^2) \perp Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$. Claim: $X + Y \sim \mathcal{N}(\mu_1 + \mu_2,\; \sigma_1^2 + \sigma_2^2)$. By Worked Example 3.1 it suffices to treat $\mu_1 = \mu_2 = 0$ (shift afterwards). Write $\tau^2 := \sigma_1^2 + \sigma_2^2$. Then

$$p_{X+Y}(s) = \int \frac{1}{2\pi\sigma_1\sigma_2}\exp\!\left(-\frac{(s-y)^2}{2\sigma_1^2} - \frac{y^2}{2\sigma_2^2}\right) dy.$$

Complete the square in $y$. The exponent is $-\tfrac{1}{2}\big[\tfrac{\tau^2}{\sigma_1^2\sigma_2^2}\big(y - \tfrac{\sigma_2^2}{\tau^2}s\big)^2 + \tfrac{s^2}{\tau^2}\big]$ — expand to verify: the $y^2$ coefficient is $\tfrac{\tau^2}{2\sigma_1^2\sigma_2^2} = \tfrac{1}{2\sigma_1^2}+\tfrac{1}{2\sigma_2^2}$ ✓; the $y$ coefficient is $-\tfrac{\tau^2}{\sigma_1^2\sigma_2^2}\cdot\tfrac{\sigma_2^2}{\tau^2}s = -\tfrac{s}{\sigma_1^2}$ ✓; the constants match ✓. The $y$-integral is a Gaussian integral equal to $\sqrt{2\pi}\,\tfrac{\sigma_1\sigma_2}{\tau}$, leaving

$$p_{X+Y}(s) = \frac{1}{\sqrt{2\pi}\,\tau}\,e^{-s^2/2\tau^2}. \qquad \checkmark$$

Variances add; standard deviations do not ($\tau = \sqrt{\sigma_1^2+\sigma_2^2}$, the Pythagorean combination — independent noise sources add in quadrature). This computation is honest but laborious; Chapter 5's characteristic functions will reduce it to one line, which is a fair preview of why that tool exists. $\square$

**Worked Example 3.9 (two inductions for the zoo: Exponential $\to$ Gamma, and Poisson additivity).** These two computations *license* edges of the distribution map in Section 3.10 — they are the proofs behind the arrows.

*(a)* Let $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Exp}(\beta)$, density $\beta e^{-\beta x}\mathbf{1}_{x>0}$. Claim: $S_n := \sum_i X_i \sim \mathrm{Gamma}(n, \beta)$, density $\tfrac{\beta^n}{(n-1)!}s^{n-1}e^{-\beta s}\mathbf{1}_{s>0}$. Induction on $n$; the base case is the claim for $n=1$. Step, via Theorem 3.4:

$$p_{S_{n+1}}(s) = \int_0^s \frac{\beta^n}{(n-1)!}x^{n-1}e^{-\beta x}\cdot \beta e^{-\beta(s-x)}\,dx = \frac{\beta^{n+1}e^{-\beta s}}{(n-1)!}\int_0^s x^{n-1}dx = \frac{\beta^{n+1}}{n!}\,s^{n}e^{-\beta s}.\ \checkmark$$

The interpretation is the useful part: if events arrive with exponential inter-arrival times (memoryless waiting), the time until the $n$-th arrival is Gamma. The same convolution argument, run with general shape parameters, shows $\mathrm{Gamma}(\alpha_1,\beta) * \mathrm{Gamma}(\alpha_2,\beta) = \mathrm{Gamma}(\alpha_1+\alpha_2,\beta)$ (same rate required!); combined with $\chi^2_1 = \mathrm{Gamma}(\tfrac12,\tfrac12)$ from Worked Example 3.5, it follows that a sum of $k$ independent squared standard Gaussians is $\mathrm{Gamma}(\tfrac k2, \tfrac12) = \chi^2_k$ — the general chi-squared distribution, obtained without ever integrating over a sphere.

*(b)* Let $X \sim \mathrm{Poisson}(\lambda_1) \perp Y \sim \mathrm{Poisson}(\lambda_2)$. Discrete convolution:

$$P(X+Y = s) = \sum_{k=0}^{s} \frac{e^{-\lambda_1}\lambda_1^k}{k!}\cdot\frac{e^{-\lambda_2}\lambda_2^{s-k}}{(s-k)!} = \frac{e^{-(\lambda_1+\lambda_2)}}{s!}\sum_{k=0}^s \binom{s}{k}\lambda_1^k\lambda_2^{s-k} = \frac{e^{-(\lambda_1+\lambda_2)}(\lambda_1+\lambda_2)^s}{s!},$$

by the binomial theorem: $X + Y \sim \mathrm{Poisson}(\lambda_1 + \lambda_2)$. Merging two independent streams of rare events gives a stream of rare events with summed rate — the reason Poisson models are closed under aggregating (e.g., pooling error counts across independent shards). The identical computation with Bernoulli/Binomial pmfs and Vandermonde's identity gives $\mathrm{Bin}(n,p) * \mathrm{Bin}(m,p) = \mathrm{Bin}(n+m,p)$, so in particular a Binomial is a sum of i.i.d. Bernoullis — the definition most useful in practice. $\square$

> **Common Confusion 3.3.** *"A sum of independent Gaussians is Gaussian. So a mixture of Gaussians is Gaussian too, right? Both are 'combining Gaussians.'"*
>
> They are entirely different operations, and conflating them is a classic exam and interview trap. A **sum** adds *values*: sample from both, output $x + y$; the densities convolve, and Gaussianity is preserved (Worked Example 3.8). A **mixture** flips a coin to decide *which* distribution to sample from: the densities *average*, $p(x) = \pi p_1(x) + (1-\pi)p_2(x)$, and a mixture of two Gaussians with separated means is bimodal — about as non-Gaussian as a unimodal family's combination can be. Convolution smooths and unifies; mixing superposes and can create structure. A Gaussian mixture model is emphatically not a model of sums.

## 3.4 Order statistics and extremes

Sorting is a transformation too. Given i.i.d. $X_1, \dots, X_n$, write $X_{(1)} \le X_{(2)} \le \cdots \le X_{(n)}$ for the sorted values — the **order statistics**. The extremes $X_{(1)} = \min$ and $X_{(n)} = \max$ have laws computable in two lines, because "the max is small" and "the min is large" are *intersections of independent events*.

**Proposition 3.5 (order statistics).** Let $X_1, \dots, X_n$ be i.i.d. with CDF $F$. Then

$$F_{X_{(n)}}(x) = F(x)^n, \qquad F_{X_{(1)}}(x) = 1 - \big(1 - F(x)\big)^n,$$

and if $F$ has density $f$, then $X_{(n)}$ and $X_{(1)}$ have densities $n F(x)^{n-1}f(x)$ and $n(1-F(x))^{n-1}f(x)$. More generally, for $1 \le k \le n$,

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!\,(n-k)!}\;F(x)^{k-1}\,\big(1-F(x)\big)^{n-k}\,f(x).$$

*Proof.* $\{X_{(n)} \le x\} = \bigcap_{i=1}^n \{X_i \le x\}$, and by independence (Definition 1.23) the probability of the intersection is the product $F(x)^n$. Likewise $\{X_{(1)} > x\} = \bigcap_i\{X_i > x\}$ gives the min. Densities follow by the chain rule wherever $f$ is continuous (with the same a.e. caveat as Theorem 3.1). For general $k$: $\{X_{(k)} \le x\}$ means *at least $k$* of the $n$ i.i.d. draws land in $(-\infty, x]$, each independently with probability $F(x)$, so $F_{X_{(k)}}(x) = \sum_{j=k}^{n}\binom{n}{j}F(x)^j(1-F(x))^{n-j}$ — a Binomial tail. Differentiating this *finite* sum term by term (no convergence issues) and writing $F' = f$:

$$f_{X_{(k)}} = f\sum_{j=k}^{n}\binom{n}{j}\Big[\,jF^{j-1}(1-F)^{n-j} - (n-j)F^{j}(1-F)^{n-j-1}\Big].$$

Using $j\binom nj = n\binom{n-1}{j-1}$ and $(n-j)\binom nj = n\binom{n-1}{j}$, the sum telescopes: the second term of index $j$ cancels the first term of index $j+1$, leaving only the first term at $j = k$, namely $n\binom{n-1}{k-1}F^{k-1}(1-F)^{n-k}f$, which equals the displayed constant. $\blacksquare$

The formula has a pleasant reading: to have $X_{(k)}$ at $x$, choose which $k-1$ observations fall below ($F^{k-1}$), which $n-k$ fall above ($(1-F)^{n-k}$), and which one sits at $x$ ($f$); the multinomial coefficient counts the assignments.

**Worked Example 3.10 (maximum of uniforms — a statistic we will meet again).** Let $X_1,\dots,X_n \overset{\text{i.i.d.}}{\sim} \mathrm{Unif}[0,\theta]$, so $F(x) = x/\theta$ on $[0,\theta]$. Then $F_{X_{(n)}}(x) = (x/\theta)^n$ and $f_{X_{(n)}}(x) = \tfrac{n x^{n-1}}{\theta^n}$ on $[0,\theta]$, giving

$$\mathbb{E}\big[X_{(n)}\big] = \int_0^\theta x\cdot\frac{nx^{n-1}}{\theta^n}\,dx = \frac{n}{n+1}\,\theta.$$

The max concentrates at the *edge of the support*, at distance $O(\theta/n)$ from it — much faster than the $O(1/\sqrt n)$ rate at which sample means concentrate (Chapter 4 makes that comparison precise). *Verification (seed 3, $10^6$ replications, $n = 10$, $\theta = 1$):* empirical $\mathbb{E}[X_{(10)}] = 0.9090$ vs. theory $10/11 = 0.9091$; empirical $P(X_{(10)} \le 0.9) = 0.3491$ vs. theory $0.9^{10} = 0.3487$. Hold onto this statistic: Section 3.9 will show $X_{(n)}$ is *sufficient* for $\theta$ — those $n$ observations compress to one number with zero loss of information about $\theta$. $\square$

One forward-looking paragraph on maxima, because it explains a guest who arrives unannounced in Section 3.11. As $n \to \infty$, $F(x)^n \to 0$ for any fixed $x$ with $F(x)<1$: maxima escape to the edge of the support, so to get a nondegenerate limit one must rescale, just as the CLT rescales sums. The **Fisher–Tippett–Gnedenko theorem** (stated, not proved — Section 3.12 table) says that if $a_n(X_{(n)} - b_n)$ converges in distribution to something nondegenerate, that something belongs to one of exactly three families, and for light-tailed $X_i$ (Gaussian, Exponential, Logistic, ...) the limit is the **Gumbel distribution**, with CDF $\exp(-e^{-x})$. The Gumbel is to maxima what the Gaussian is to sums. That, and its exact algebraic properties under maximization (Proposition 3.29), is why it — and not some other bump-shaped law — is the distribution that makes the argmax trick of Section 3.11 work.

> **Self-check 3.4** *(covers Sections 3.3–3.4)*. (a) In Theorem 3.4, which summand needs a density, and what does the answer imply about adding Gaussian noise to a quantized (discrete) signal? (b) Why does $\mathrm{Gamma}(\alpha_1,\beta_1) * \mathrm{Gamma}(\alpha_2,\beta_2)$ fail to be Gamma when $\beta_1 \neq \beta_2$ — where does the convolution computation break? (c) The min of $n$ i.i.d. $\mathrm{Exp}(\beta)$ variables is $\mathrm{Exp}(n\beta)$ — verify in one line from Proposition 3.5. (d) Why must maxima be recentered/rescaled to have a distributional limit, when a sample mean already converges without rescaling? *(Answers in Appendix A.)*

## 3.5 The multivariate change of variables

### 3.5.1 What linear maps do to volume — proved, not asserted

Everything in this section flows from one geometric fact: **a linear map rescales all volumes by one number, $|\det T|$**. We prove it, because it is the actual content of the Jacobian formula; the rest is bookkeeping.

**Lemma 3.6 (Lebesgue measure under linear maps).** Let $T : \mathbb{R}^d \to \mathbb{R}^d$ be linear and invertible. Then for every Borel set $A$, $\lambda_d(TA) = |\det T|\,\lambda_d(A)$.

*Proof.* Define $\mu_T(A) := \lambda_d(TA)$; this is a Borel measure (countable additivity is inherited because $T$ is a bijection, so it maps disjoint unions to disjoint unions), it is finite on bounded sets, and it is translation invariant: $\mu_T(A + x) = \lambda_d(TA + Tx) = \lambda_d(TA)$, using the translation invariance of $\lambda_d$ itself (provable on boxes from the definition, then extended to all Borel sets by Theorem 1.20's uniqueness, since boxes form a generating $\pi$-system).

*Step 1: a translation-invariant Borel measure $\mu$ on $\mathbb{R}^d$, finite on the unit cube, is $c\,\lambda_d$ with $c = \mu([0,1)^d)$.* Partition the half-open unit cube into $2^{kd}$ translated copies of the dyadic cube $[0, 2^{-k})^d$. Translation invariance forces every copy to have equal $\mu$-measure, so $\mu([0,2^{-k})^d) = 2^{-kd}c = c\,\lambda_d([0,2^{-k})^d)$. Every half-open box with dyadic-rational corners is a finite disjoint union of such dyadic cubes, so $\mu = c\lambda_d$ on all of them; dyadic boxes form a $\pi$-system generating $\mathcal{B}(\mathbb{R}^d)$ (Definition 1.6), and both measures are $\sigma$-finite over them, so Theorem 1.20 gives $\mu = c\,\lambda_d$ on all Borel sets.

Applying Step 1 to $\mu_T$: there is a constant $c(T) \ge 0$ with $\lambda_d(TA) = c(T)\lambda_d(A)$ for all Borel $A$, and $c(T) > 0$ since $T$ is invertible ($T[0,1)^d$ contains an open set). The map $c$ is multiplicative: $c(ST)\lambda_d(A) = \lambda_d(STA) = c(S)\lambda_d(TA) = c(S)c(T)\lambda_d(A)$.

*Step 2: evaluate $c$ on two special classes.* If $D = \mathrm{diag}(d_1,\dots,d_d)$ with all $d_i \neq 0$, then $D[0,1)^d$ is a box with side lengths $|d_i|$, so $c(D) = \prod_i |d_i| = |\det D|$, directly from the definition of $\lambda_d$ on boxes. If $O$ is orthogonal, then $O$ maps the closed unit ball $B$ to itself, so $c(O)\lambda_d(B) = \lambda_d(OB) = \lambda_d(B)$, and since $0 < \lambda_d(B) < \infty$, $c(O) = 1 = |\det O|$.

*Step 3: assemble via the singular value decomposition (linear algebra).* Any invertible $T$ factors as $T = U\Sigma V^{\top}$ with $U, V$ orthogonal and $\Sigma$ diagonal with positive entries (the singular values). By multiplicativity and Step 2, $c(T) = c(U)c(\Sigma)c(V^\top) = 1\cdot \det\Sigma\cdot 1 = \det \Sigma = |\det T|$, the last equality because $|\det T| = |\det U||\det\Sigma||\det V^\top| = \det\Sigma$. $\blacksquare$

Pause on what the SVD step *says*: every linear map is a rotation, then independent axis scalings, then another rotation. Rotations move volume around without changing it; the scalings multiply it by $\prod \sigma_i = |\det T|$. The determinant is not a formula that happens to appear — it *is* the volume-rescaling factor, and Lemma 3.6 is the theorem that makes "det = volume change" a fact rather than a slogan.

**Theorem 3.7 (affine change of variables).** Let $X$ have density $p_X$ on $\mathbb{R}^d$ (w.r.t. $\lambda_d$), and let $Y = TX + b$ with $T$ invertible. Then

$$p_Y(y) = \frac{1}{|\det T|}\;p_X\big(T^{-1}(y - b)\big).$$

*Proof.* First, a change-of-variables identity for integrals: for every measurable $f \ge 0$,

$$\int f(Tx + b)\,dx = \frac{1}{|\det T|}\int f(y)\,dy. \tag{3.5.1}$$

For $f = \mathbf{1}_A$ this *is* Lemma 3.6 plus translation invariance: $\int \mathbf{1}_A(Tx+b)\,dx = \lambda_d(T^{-1}(A - b)) = |\det T^{-1}|\lambda_d(A) = \lambda_d(A)/|\det T|$. Extend by linearity to nonnegative simple functions, then to all measurable $f \ge 0$ by Monotone Convergence (Ch. 2), using that every such $f$ is an increasing limit of simple functions (Ch. 2's "standard machine"). Now compute, for Borel $B$:

$$P(Y \in B) = P\big(X \in T^{-1}(B - b)\big) = \int \mathbf{1}_B(Tx + b)\,p_X(x)\,dx.$$

Apply (3.5.1) to $f(y) = \mathbf{1}_B(y)\,p_X(T^{-1}(y - b))$ — check: $f(Tx+b) = \mathbf{1}_B(Tx+b)\,p_X(x)$ ✓ — to get $P(Y\in B) = \tfrac{1}{|\det T|}\int_B p_X(T^{-1}(y-b))\,dy$. $\blacksquare$

**Worked Example 3.11 (the multivariate Gaussian density, derived).** Define $Z \sim \mathcal{N}(0, I_d)$ by its density $p_Z(z) = (2\pi)^{-d/2}e^{-\|z\|^2/2}$ (a product of $d$ standard normals; that the product of densities is the density of the independent vector is Fubini–Tonelli, Ch. 2). For a symmetric positive-definite $\Sigma$, let $\Sigma^{1/2}$ be its symmetric square root (spectral theorem) and set $X = \mu + \Sigma^{1/2}Z$ — this is how one *defines* and *samples* $\mathcal{N}(\mu, \Sigma)$. Theorem 3.7 with $T = \Sigma^{1/2}$:

$$p_X(x) = \frac{1}{|\det \Sigma^{1/2}|}\;p_Z\big(\Sigma^{-1/2}(x-\mu)\big) = \frac{1}{(2\pi)^{d/2}(\det\Sigma)^{1/2}}\exp\!\Big(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\Big),$$

using $\|\Sigma^{-1/2}(x-\mu)\|^2 = (x-\mu)^\top\Sigma^{-1}(x-\mu)$ and $\det \Sigma^{1/2} = (\det\Sigma)^{1/2}$. So the mysterious normalizer $(\det \Sigma)^{-1/2}$ is not a formula to memorize: it is Lemma 3.6's volume factor for the map that manufactures correlated Gaussians from white noise. (Any $T$ with $TT^\top = \Sigma$ — e.g., the Cholesky factor, which is what library samplers actually use — gives the same law: the density above depends on $T$ only through $\Sigma$.) $\square$

### 3.5.2 The nonlinear case: linearize, then be honest

A differentiable map $g$ is, near each point $x$, its own best affine approximation: $g(x + h) \approx g(x) + J_g(x)\,h$, where $J_g(x)$ is the $d\times d$ **Jacobian matrix** $[\partial g_i/\partial x_j]$. (Notation: we write $J_g$ or $Dg$, never "$\nabla g$", reserving $\nabla$ for gradients of scalar functions — a convention that will prevent a sign disaster in Section 3.7.) A tiny cube of volume $\varepsilon^d$ at $x$ is mapped, to first order, to a parallelepiped of volume $|\det J_g(x)|\,\varepsilon^d$ (Lemma 3.6 applied to the linearization). Summing over a fine partition, the density must transform locally exactly as in Theorem 3.7, with $|\det T|$ replaced by the *local* factor $|\det J_g(x)|$. That is the intuition; here is the precise statement.

**Theorem 3.8 (change of variables, $C^1$ diffeomorphisms — stated; proof deferred).** Let $U, V \subseteq \mathbb{R}^d$ be open and $g : U \to V$ a bijection, continuously differentiable with $\det J_g(x) \ne 0$ for all $x \in U$ (so $g^{-1}$ is also $C^1$, by the inverse function theorem). Then for every measurable $f \ge 0$ on $V$,

$$\int_V f(y)\,dy = \int_U f\big(g(x)\big)\,\big|\det J_g(x)\big|\,dx.$$

Consequently, if $X$ has density $p_X$ supported in $U$, then $Y = g(X)$ has density on $V$

$$p_Y(y) = p_X\big(g^{-1}(y)\big)\,\big|\det J_{g^{-1}}(y)\big| = \frac{p_X\big(g^{-1}(y)\big)}{\big|\det J_g\big(g^{-1}(y)\big)\big|}.$$

This is honesty flag (a) from the orientation: the full proof requires a genuinely delicate approximation argument (covering $U$ by small cubes, controlling the error of the linearization uniformly, and a Vitali-type covering step) that belongs to a real-analysis course, not this book. We have proved the affine case completely (Theorem 3.7), which is both the engine of the general proof and — as the next example and all of Section 3.7 show — most of what ML ever uses. The general statement is logged in Section 3.12 with references. Note also that Theorem 3.1 is the $d = 1$ case, and the equality of the two displayed forms is the inverse function theorem again: $J_{g^{-1}}(y) = [J_g(g^{-1}(y))]^{-1}$, and $\det$ of an inverse is the reciprocal.

<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img" aria-label="A small square mapped to a parallelogram by the Jacobian">
  <style>text{font-family:sans-serif;font-size:13px;fill:currentColor}.cap{font-size:12px}</style>
  <!-- left: grid + small square -->
  <g stroke="currentColor" stroke-width="0.5" opacity="0.35">
    <line x1="60" y1="60" x2="60" y2="220"/><line x1="110" y1="60" x2="110" y2="220"/><line x1="160" y1="60" x2="160" y2="220"/><line x1="210" y1="60" x2="210" y2="220"/><line x1="260" y1="60" x2="260" y2="220"/>
    <line x1="60" y1="60" x2="260" y2="60"/><line x1="60" y1="110" x2="260" y2="110"/><line x1="60" y1="160" x2="260" y2="160"/><line x1="60" y1="220" x2="260" y2="220"/>
  </g>
  <rect x="110" y="110" width="50" height="50" fill="#3b82f6" opacity="0.45" stroke="#3b82f6"/>
  <text x="98" y="98" fill="#3b82f6">volume ε^d at x</text>
  <!-- arrow -->
  <path d="M 285 140 C 330 120 350 120 395 138" fill="none" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr2)"/>
  <defs><marker id="arr2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="currentColor"/></marker></defs>
  <text x="300" y="112">g(x+h) ≈ g(x) + J_g(x) h</text>
  <!-- right: curved grid + parallelogram -->
  <g stroke="currentColor" stroke-width="0.5" opacity="0.35" fill="none">
    <path d="M 430 70 Q 500 60 560 80 Q 620 100 680 90"/>
    <path d="M 425 120 Q 500 110 565 130 Q 625 150 685 140"/>
    <path d="M 420 175 Q 500 165 570 185 Q 630 205 690 195"/>
    <path d="M 430 70 Q 425 120 420 175"/>
    <path d="M 495 64 Q 495 112 493 168"/>
    <path d="M 560 80 Q 565 130 570 185"/>
    <path d="M 625 92 Q 628 143 632 198"/>
  </g>
  <polygon points="495,112 560,128 566,183 500,166" fill="#10b981" opacity="0.5" stroke="#10b981"/>
  <text x="470" y="222" fill="#10b981">volume ≈ |det J_g(x)| ε^d</text>
  <text x="60" y="262" class="cap">Figure 3.3 — Locally, a smooth map acts like its Jacobian: an infinitesimal square (blue) is carried to a parallelogram (green)</text>
  <text x="60" y="278" class="cap">whose volume is |det J_g| times larger. Densities must shrink by the same factor so that probability over the patch is conserved.</text>
</svg>

**Worked Example 3.12 (polar coordinates and Box–Muller).** How do you turn the uniform random numbers your computer generates into exact Gaussians? Take $U_1, U_2 \overset{\text{i.i.d.}}{\sim} \mathrm{Unif}(0,1)$ and set

$$Z_1 = \sqrt{-2\log U_1}\,\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\log U_1}\,\sin(2\pi U_2).$$

*Claim: $(Z_1, Z_2) \sim \mathcal{N}(0, I_2)$, exactly.* The map $g : (0,1)^2 \to \mathbb{R}^2\setminus(\text{a null ray})$ is a $C^1$ bijection, and its inverse is explicit: $r^2 := z_1^2 + z_2^2$ gives $U_1 = e^{-r^2/2}$ and $U_2 = \tfrac{1}{2\pi}\operatorname{atan2}(z_2, z_1)$. Rather than differentiating square roots, apply Theorem 3.8 in the direction with the easy Jacobian, computing $p_Z(z) = p_U(g^{-1}(z))\,|\det J_{g^{-1}}(z)|$:

$$J_{g^{-1}}(z) = \begin{pmatrix} \partial_{z_1}e^{-r^2/2} & \partial_{z_2}e^{-r^2/2}\\[2pt] \partial_{z_1}\tfrac{\phi}{2\pi} & \partial_{z_2}\tfrac{\phi}{2\pi}\end{pmatrix} = \begin{pmatrix} -z_1 e^{-r^2/2} & -z_2 e^{-r^2/2}\\[2pt] \tfrac{-z_2}{2\pi r^2} & \tfrac{z_1}{2\pi r^2}\end{pmatrix},$$

using $\partial_{z_1}\phi = -z_2/r^2$, $\partial_{z_2}\phi = z_1/r^2$ for the angle $\phi$. The determinant is $-\tfrac{z_1^2}{2\pi r^2}e^{-r^2/2} - \tfrac{z_2^2}{2\pi r^2}e^{-r^2/2} = -\tfrac{1}{2\pi}e^{-r^2/2}$. Since $p_U \equiv 1$ on $(0,1)^2$,

$$p_Z(z_1,z_2) = 1 \cdot \left|-\tfrac{1}{2\pi}e^{-r^2/2}\right| = \frac{1}{2\pi}e^{-(z_1^2+z_2^2)/2} = \varphi(z_1)\,\varphi(z_2). \qquad \checkmark$$

Two i.i.d. standard Gaussians, from two uniforms, with no approximation. The geometry: $-2\log U_1 \sim \mathrm{Exp}(\tfrac12)$ supplies the squared radius (which for a 2-D Gaussian is exactly $\chi^2_2 = \mathrm{Exp}(\tfrac12)$ — Worked Example 3.9's identification), while $2\pi U_2$ supplies a uniform angle; a Gaussian in 2-D is precisely "exponential energy, uniform direction." *Verification (seed 0, $n = 10^6$):* sample means $-0.0008, +0.0014$; variances $1.0011, 1.0011$; correlation $+0.00043$; $P(|Z_1|\le 1) = 0.6826$ vs. $0.6827$; skewness $-0.0029$ and excess kurtosis $+0.0022$ vs. $0$ and $0$. $\square$

### 3.5.3 The log-determinant as a volume-change rate — and a sign table

Take logarithms of Theorem 3.8's density formula:

$$\log p_Y(y) = \log p_X\big(g^{-1}(y)\big) - \log\big|\det J_g\big(g^{-1}(y)\big)\big|.$$

The log-det is the *local volume expansion rate in nats*: where $g$ expands volume ($|\det J_g| > 1$), log-density drops by exactly the expansion; where $g$ contracts, log-density rises. Because Section 3.7 lives and dies by the sign of this term, and because the literature genuinely disagrees on which direction to call "forward," here is the table to consult whenever in doubt. Let $f : \mathcal{X} \to \mathcal{Z}$ (data to latent) be invertible with $g := f^{-1} : \mathcal{Z} \to \mathcal{X}$ (latent to data), and recall $\det J_{g}(z) = 1/\det J_f(g(z))$.

| You know the density of... | You want the density of... | Formula |
|---|---|---|
| latent $Z$, i.e. $p_Z$ | data $X = g(Z)$ | $\log p_X(x) = \log p_Z(f(x)) + \log\lvert\det J_f(x)\rvert$ |
| latent $Z$, i.e. $p_Z$ | same, written with $g$ | $\log p_X(x) = \log p_Z(z) - \log\lvert\det J_g(z)\rvert$ at $z = f(x)$ |
| data $X$, i.e. $p_X$ | latent $Z = f(X)$ | $\log p_Z(z) = \log p_X(g(z)) + \log\lvert\det J_g(z)\rvert$ |

Mnemonic that survives panic: **the Jacobian in the formula belongs to the map you evaluate the *other* density through** — equivalently, "add the log-det of the direction you are mapping *into* the known density." If the signs in a paper or codebase seem off, translate them into this table before concluding anything; more often than not the paper simply fixed the opposite convention.

> **Self-check 3.5.** (a) In Lemma 3.6's proof, exactly where was Theorem 1.20 (uniqueness on a generating $\pi$-system) used, and why were dyadic boxes a legitimate choice? (b) A flow layer doubles the first coordinate and halves the second. What is $|\det J|$, and what happens to the log-density of a point pushed through it? (c) Derive in one line what Theorem 3.7 says for orthogonal $T$: how do Gaussian densities behave under rotation, and why does that make $\mathcal{N}(0, I)$ the natural "directionless" latent? *(Answers in Appendix A.)*

## 3.6 Inverse transform sampling and the probability integral transform

This section runs the chapter's question in reverse. Instead of "given $X$, what is the law of $g(X)$?", we ask: **given a target law, which $g$ applied to a uniform random number produces it?** The answer settles a debt outstanding since Chapter 1, and it is the mathematical foundation of random number generation.

### 3.6.1 The generalized inverse

If $F$ is a continuous, strictly increasing CDF, the natural candidate is the ordinary inverse $F^{-1}$. But CDFs in the wild have **jumps** (atoms) and **flats** (gaps in the support) — the ReLU variable's CDF has a jump of height $\tfrac12$ at $0$ — and an honest theory must handle them. The right object:

**Definition 3.9 (generalized inverse / quantile function).** For a CDF $F$, define $F^{-1} : (0,1) \to \mathbb{R}$ by

$$F^{-1}(u) := \inf\{x \in \mathbb{R} : F(x) \ge u\}.$$

(The set is nonempty since $F \to 1$, and bounded below since $F \to 0$, by Proposition 1.19; so the infimum is finite.) When $F$ is continuous and strictly increasing this is the ordinary inverse. At a jump of $F$ (an atom of the law), $F$ skips over an interval of $u$-values, and $F^{-1}$ maps that whole interval to the jump location — the atom gets a $u$-interval of positive length, exactly its probability. Across a flat of $F$ (a gap in the support), $F^{-1}$ jumps, skipping the gap. Jumps and flats trade places under inversion.

Everything one ever needs about $F^{-1}$ follows from a single inequality — it says $F^{-1}$ and $F$ interlock like a Galois connection:

**Lemma 3.10 (the switching relation).** For all $u \in (0,1)$ and $x \in \mathbb{R}$:

$$F^{-1}(u) \le x \iff u \le F(x).$$

*Proof.* ($\Leftarrow$) If $u \le F(x)$, then $x$ belongs to $\{x' : F(x') \ge u\}$, so the infimum $F^{-1}(u)$ is $\le x$. ($\Rightarrow$) First observe $F(F^{-1}(u)) \ge u$: take $x_n \downarrow F^{-1}(u)$ with $F(x_n) \ge u$ (possible by the definition of infimum — either the inf is attained, or points of the set approach it from the right); right-continuity of $F$ (Proposition 1.19) gives $F(F^{-1}(u)) = \lim F(x_n) \ge u$. Now if $F^{-1}(u) \le x$, monotonicity of $F$ gives $F(x) \ge F(F^{-1}(u)) \ge u$. $\blacksquare$

Note which unglamorous property did the work: **right-continuity** — the property Proposition 1.19 established and which might have seemed like pedantry at the time. Had CDFs been defined with $P(X < a)$ (left-continuous), the lemma would need reworking. Small conventions, chosen once, pay compound interest.

A free consequence: $F^{-1}$ is measurable, with no extra argument. By Lemma 3.10, $\{u \in (0,1) : F^{-1}(u) \le x\} = (0, F(x)] \cap (0,1)$, an interval, hence Borel; and half-lines generate $\mathcal{B}(\mathbb{R})$ (Lemma 1.14). (It is also nondecreasing, but we didn't even need that.)

### 3.6.2 Sampling anything from Uniform(0,1)

**Theorem 3.11 (inverse transform sampling).** Let $F$ be any CDF and $U \sim \mathrm{Unif}(0,1)$. Then $X := F^{-1}(U)$ has CDF exactly $F$.

*Proof.* For any $x$: $P(X \le x) = P\big(F^{-1}(U) \le x\big) = P\big(U \le F(x)\big) = F(x)$, the middle step being Lemma 3.10 applied pointwise to the event, and the last step being the uniform CDF. $\blacksquare$

One line, total generality: jumps, flats, heavy tails, anything. And it immediately pays a debt:

**Corollary 3.12 (the converse of Proposition 1.19, at last).** Every function $F : \mathbb{R} \to [0,1]$ that is nondecreasing, right-continuous, with $\lim_{x\to-\infty}F = 0$ and $\lim_{x\to+\infty}F = 1$, is the CDF of some random variable on some probability space.

*Proof.* Take the space $((0,1), \mathcal{B}, \lambda)$ — Chapter 1's uniform space — and the random variable $X(u) = F^{-1}(u)$, measurable by the observation above. Theorem 3.11's computation shows $P(X \le x) = F(x)$. $\blacksquare$

Chapter 1 stated this converse "on faith"; the faith is hereby discharged. Notice the philosophical bonus: the proof shows a single probability space, the uniform unit interval, is rich enough to host a random variable with *any* prescribed one-dimensional law. In a precise sense, $\mathrm{Unif}(0,1)$ is the universal source of randomness, and every scalar sampler ever written is a measurable function applied to it — the "randomness as a seed $\omega$" picture of Chapter 1, made literal.

**Worked Example 3.13 (two samplers, verified).** *(a) Exponential.* For $F(x) = 1 - e^{-\beta x}$, solving $F(x) = u$ gives $F^{-1}(u) = -\log(1-u)/\beta$; since $1 - U \sim \mathrm{Unif}(0,1)$ too, the sampler $X = -\log(U)/\beta$ is standard. *Verification (Lab B.1, seed 0, $n = 10^6$, $\beta = 2$):* sample mean $0.4999$ (target $0.5$), sample variance $0.2502$ (target $0.25$), Kolmogorov–Smirnov statistic $0.00096$ with $p$-value $0.314$ — fully consistent with the target law. *(b) A discrete target, exercising the jumps.* Target: $P(X = k) = \pi_k$ for $k \in \{0,1,2\}$, $\pi = (0.2, 0.5, 0.3)$. The CDF is a staircase; $F^{-1}$ maps $(0, 0.2] \mapsto 0$, $(0.2, 0.7] \mapsto 1$, $(0.7, 1) \mapsto 2$. *Verification (same lab, $n = 10^6$):* empirical frequencies $(0.2000, 0.5002, 0.2998)$. The generalized inverse handled the atoms with no special-casing — the $u$-interval lengths *are* the probabilities. $\square$

### 3.6.3 The probability integral transform

Now the other direction: push $X$ through its *own* CDF.

**Theorem 3.13 (probability integral transform).** If $X$ has a **continuous** CDF $F$, then $F(X) \sim \mathrm{Unif}(0,1)$.

*Proof.* Fix $u \in (0,1)$ and let $q := F^{-1}(u)$. Lemma 3.10 says $u \le F(x) \iff q \le x$ for every real $x$; applying this to each realization of $X$ identifies the events $\{F(X) \ge u\} = \{X \ge q\}$. Therefore

$$P\big(F(X) \ge u\big) = P(X \ge q) = 1 - P(X < q) = 1 - F(q^-) = 1 - F(q),$$

the last step by continuity of $F$ (no atom at $q$, so the left limit equals the value). It remains to show $F(q) = u$. We already know $F(q) \ge u$ (established inside Lemma 3.10's proof, from right-continuity). Suppose $F(q) > u$. Since $F$ is continuous with $\lim_{x \to -\infty} F(x) = 0 < u$, the intermediate value theorem yields a point $x_0 < q$ with $F(x_0) = u$; but then $x_0$ lies in the set $\{x : F(x) \ge u\}$ whose infimum is $q$, contradicting $x_0 < q$. Hence $F(q) = u$, so $P(F(X) \ge u) = 1 - u$ for every $u \in (0,1)$: the CDF of $F(X)$ is that of $\mathrm{Unif}(0,1)$, and Theorem 1.20 finishes the identification. $\blacksquare$

> **Common Confusion 3.4.** *"$F(X) \sim \mathrm{Unif}(0,1)$ always, right? It's just plugging a variable into its own CDF."*
>
> Continuity is essential, and the failure is total, not marginal. Let $X \sim \mathrm{Bernoulli}(\tfrac12)$. Then $F(X)$ takes the value $F(0) = \tfrac12$ or $F(1) = 1$, each with probability $\tfrac12$ — a two-point distribution, nowhere near uniform. The mechanism: $F(X)$ can only take values in the *range* of $F$, and a CDF with jumps skips most of $(0,1)$. The practical bite: p-values are built on the PIT (under a continuous null, the p-value is uniform), so **p-values of discrete test statistics are not uniform under the null** — they are stochastically larger, making such tests conservative. If you have ever seen a p-value histogram with bands and spikes instead of a flat profile, you have seen this confusion in the wild.

Theorems 3.11 and 3.13 together say: for continuous laws, the CDF and its inverse form a lossless round trip through $\mathrm{Unif}(0,1)$. Any continuous law can be mapped onto any other:

$$X \sim P \;\xrightarrow{\;F_P\;}\; U \sim \mathrm{Unif}(0,1) \;\xrightarrow{\;F_Q^{-1}\;}\; Y \sim Q.$$

Two applications you will meet in practice. **Quantile preprocessing:** scikit-learn's `QuantileTransformer(output_distribution="normal")` is literally $\Phi^{-1} \circ \hat F_n$ — the empirical CDF into the Gaussian quantile function — a nonparametric "Gaussianizer" for skewed features, and now you know the theorem it rides on. **Copulas:** applying the PIT to each coordinate of a random vector strips away the marginals, leaving a distribution on $[0,1]^d$ — the *copula* — that carries exactly the dependence structure. **Sklar's theorem** (stated by name, not proved; Section 3.12) says every joint law factors into its marginals plus a copula. Quantitative finance runs on this decomposition; ML uses it for dependence modeling and for making features comparable.

> **Self-check 3.6.** (a) Compute $F^{-1}(0.5)$ for the ReLU variable $A = \max(0,Z)$ of Worked Example 3.6, and explain which feature of $F^{-1}$ (jump-handling or flat-handling) it exercises. (b) Why is Theorem 3.11's proof one line while Theorem 3.13 needed continuity and a compactness-flavored argument — what is the asymmetry? (c) Your test statistic is discrete and you compute p-values from it. Will a correctly-null p-value histogram look flat, and if not, in which direction does it deviate? *(Answers in Appendix A.)*

## 3.7 Normalizing flows: change of variables as a generative model

### 3.7.1 The convention box — read this before anything else

> **Convention (fixed for this book).** A normalizing flow is an invertible, differentiable map with a fixed direction convention:
>
> $$f_\theta : \underbrace{\mathcal{X}}_{\text{data}} \to \underbrace{\mathcal{Z}}_{\text{latent}} \quad (\textbf{normalizing direction: } x \mapsto z), \qquad g_\theta := f_\theta^{-1} : \mathcal{Z} \to \mathcal{X} \quad (\textbf{generative direction: } z \mapsto x).$$
>
> "Normalizing" because $f_\theta$ carries messy data to a simple ("normal") base law. **The literature is genuinely inconsistent**: some papers call the generative direction forward, others the normalizing one, and the sign of the log-det term flips between conventions. When reading any flow paper or codebase, your first act should be to determine its direction convention; when the loss seems to have the wrong sign, consult the table in Section 3.5.3. This is the most common implementation error in the entire topic, and it produces a model that *trains* (the optimizer happily maximizes volume expansion) while computing garbage likelihoods.

### 3.7.2 Exact log-likelihood

**Proposition 3.14 (flow log-likelihood and composition).** Let $f_\theta : \mathbb{R}^d \to \mathbb{R}^d$ be a $C^1$ diffeomorphism and fix a base density $p_Z$ (typically $\mathcal{N}(0, I_d)$). Define the model $X = g_\theta(Z)$, $Z \sim p_Z$. Then the model density exists and is exactly

$$\log p_\theta(x) = \log p_Z\big(f_\theta(x)\big) + \log\big|\det J_{f_\theta}(x)\big|,$$

and if $f_\theta = f_L \circ \cdots \circ f_1$ is a composition of diffeomorphisms with intermediate values $x_0 := x$, $x_\ell := f_\ell(x_{\ell-1})$, then

$$\log\big|\det J_{f_\theta}(x)\big| = \sum_{\ell=1}^{L}\log\big|\det J_{f_\ell}(x_{\ell-1})\big|.$$

*Proof.* The density formula is Theorem 3.8 applied to the generative map $g_\theta$ (whose inverse Jacobian at $x$ is $J_{f_\theta}(x)$; see the table in 3.5.3, first row). For the composition: the chain rule gives $J_{f_\theta}(x) = J_{f_L}(x_{L-1})\cdots J_{f_1}(x_0)$, determinants multiply, and logs of absolute values add. $\blacksquare$

Pause on what this buys. GANs (Failure 4) have no density; VAEs (Ch. 2's variational bound, revisited later) optimize a lower bound on $\log p_\theta$; flows compute $\log p_\theta(x)$ **exactly**, so training is plain maximum likelihood, evaluation is honest test log-likelihood in nats, and model comparison is apples to apples. The price, from Chapter 1's measure-zero theorem: the escape from "no density" required $k = d$ and invertibility. **No compression.** The latent space of a flow on $256\times256\times3$ images has $196{,}608$ dimensions, every one preserved through every layer. VAEs pay instead with a bound and an approximate posterior; diffusion models pay with slow iterative sampling; flows pay with dimension and architectural constraints (invertibility). Every deep generative family is a different answer to the same measure-theoretic constraint — that is the honest one-sentence summary of the field.

The composition rule turns design into a modular question: **find cheap invertible layers whose log-dets are cheap**, then stack. A dense $d \times d$ determinant costs $O(d^3)$ — hopeless inside a training loop at $d \sim 10^5$. The field's solution is structural sparsity of the Jacobian:

### 3.7.3 Coupling layers (RealNVP)

Split coordinates into two blocks, $x = (x_A, x_B)$. A **coupling layer** computes, in the normalizing direction,

$$z_A = x_A, \qquad z_B = x_B \odot e^{s_\theta(x_A)} + t_\theta(x_A),$$

where $s_\theta, t_\theta$ are arbitrary neural networks (no invertibility requirement on them!) and $\odot$ is elementwise product. Inversion is closed-form — $x_B = (z_B - t_\theta(z_A))\odot e^{-s_\theta(z_A)}$, using $x_A = z_A$ — so both directions cost one network evaluation. The Jacobian is block-triangular:

$$J_f = \begin{pmatrix} I & 0\\ \partial z_B/\partial x_A & \mathrm{diag}\big(e^{s_\theta(x_A)}\big)\end{pmatrix} \implies \log|\det J_f| = \sum_j s_\theta(x_A)_j,$$

because the determinant of a block-triangular matrix is the product of the diagonal blocks' determinants (linear algebra), and the messy block $\partial z_B/\partial x_A$ — the only place the deep networks' derivatives appear — **multiplies into nothing**. The model gets full neural expressiveness in $s_\theta, t_\theta$ at *zero* log-det cost: reading the predicted log-scales and summing is $O(d)$. Since $x_A$ passes through unchanged, layers alternate which block is which (or, in Glow, mix coordinates between couplings with learned invertible $1\times1$ convolutions — small $c \times c$ determinants over channels, cheap because $c \ll d$), so that after a few layers every coordinate has been transformed conditioned on every other.

<svg viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img" aria-label="Coupling layer and its block triangular Jacobian">
  <style>text{font-family:sans-serif;font-size:13px;fill:currentColor}.cap{font-size:12px}.s{font-size:11px}</style>
  <!-- left: dataflow -->
  <rect x="40" y="60" width="70" height="40" fill="#3b82f6" opacity="0.35" stroke="#3b82f6"/><text x="58" y="85">x_A</text>
  <rect x="40" y="170" width="70" height="40" fill="#f59e0b" opacity="0.35" stroke="#f59e0b"/><text x="58" y="195">x_B</text>
  <rect x="250" y="60" width="70" height="40" fill="#3b82f6" opacity="0.35" stroke="#3b82f6"/><text x="258" y="85">z_A = x_A</text>
  <rect x="250" y="170" width="70" height="40" fill="#10b981" opacity="0.35" stroke="#10b981"/><text x="262" y="195">z_B</text>
  <line x1="110" y1="80" x2="250" y2="80" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr3)"/>
  <rect x="150" y="110" width="80" height="36" rx="6" fill="none" stroke="currentColor"/><text x="158" y="133">s(·), t(·)</text>
  <path d="M 145 82 C 160 95 165 100 175 110" fill="none" stroke="currentColor" stroke-width="1.3" marker-end="url(#arr3)"/>
  <path d="M 195 146 C 205 160 215 172 240 184" fill="none" stroke="currentColor" stroke-width="1.3" marker-end="url(#arr3)"/>
  <line x1="110" y1="190" x2="245" y2="190" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr3)"/>
  <text x="118" y="215" class="s">z_B = x_B ⊙ exp(s(x_A)) + t(x_A)</text>
  <defs><marker id="arr3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="currentColor"/></marker></defs>
  <!-- right: Jacobian sparsity -->
  <text x="430" y="52">Jacobian ∂z/∂x</text>
  <g>
    <rect x="420" y="60" width="110" height="110" fill="none" stroke="currentColor"/>
    <rect x="530" y="60" width="110" height="110" fill="none" stroke="currentColor"/>
    <rect x="420" y="170" width="110" height="110" fill="none" stroke="currentColor"/>
    <rect x="530" y="170" width="110" height="110" fill="none" stroke="currentColor"/>
    <!-- identity block -->
    <line x1="424" y1="64" x2="526" y2="166" stroke="#3b82f6" stroke-width="4"/>
    <text x="440" y="120" fill="#3b82f6">I</text>
    <!-- zero block top right -->
    <text x="578" y="120">0</text>
    <!-- dense block bottom left -->
    <g fill="#f59e0b" opacity="0.7">
      <circle cx="440" cy="190" r="4"/><circle cx="470" cy="200" r="4"/><circle cx="500" cy="188" r="4"/><circle cx="452" cy="222" r="4"/><circle cx="488" cy="235" r="4"/><circle cx="515" cy="215" r="4"/><circle cx="435" cy="255" r="4"/><circle cx="472" cy="262" r="4"/><circle cx="508" cy="252" r="4"/>
    </g>
    <text x="430" y="300" fill="#f59e0b" class="s">dense (network grads) — never enters det</text>
    <!-- diagonal block bottom right -->
    <line x1="534" y1="174" x2="636" y2="276" stroke="#10b981" stroke-width="4"/>
    <text x="600" y="205" fill="#10b981" class="s">diag(e^s)</text>
  </g>
  <text x="40" y="252" class="s">both directions: one network pass;</text>
  <text x="40" y="268" class="s">log|det J_f| = Σ_j s(x_A)_j  — O(d)</text>
  <text x="40" y="318" class="cap">Figure 3.4 — A coupling layer (left) copies x_A and affinely transforms x_B using scales/shifts predicted from x_A. Its Jacobian</text>
  <text x="40" y="330" class="cap" style="font-size:12px">(right) is block-triangular: the determinant sees only the green diagonal, so arbitrary networks s,t contribute zero log-det cost.</text>
</svg>

**Lab B.2 (verified).** A 2-coupling-layer RealNVP (hidden width 64, tanh, log-scales bounded by $2\tanh(\cdot)$ for stability), trained by full-batch Adam (learning rate $10^{-3}$, 4000 iterations) on 2000 points of a standardized two-moons dataset (data seed 0, init seeds 42/43). Baseline: the standard normal assigns the data $2.8379$ nats/point of negative log-likelihood (NLL); the untrained flow starts at $2.8297$. Training NLL: $2.1389$ (iter 500) $\to$ $1.8820$ (1000) $\to$ $1.8359$ (2000) $\to$ $1.8175$ (4000); held-out NLL on 2000 fresh points: $1.8086$ nats/point — about one full nat per point better than the Gaussian, from two coupling layers, with the likelihood *exact*, not bounded. And the formula itself was audited: at three data points, the analytic $\log|\det J_f(x)|$ (sum of predicted log-scales) was checked against a finite-difference Jacobian ($h = 10^{-5}$) — values $+0.998602$, $+0.485080$, $+0.032723$, agreeing to within $6.7\times10^{-10}$. One implementation note in the spirit of the convention box: the lab code parameterized the *normalizing* direction directly, $z_B = (x_B - t)\,e^{-s}$ with $\log|\det J_f| = -\sum_j s_j$ — the same model with the roles of the two parameterizations swapped, and a live demonstration that the sign convention must be read off the code, not assumed.

### 3.7.4 Autoregressive flows, and who pays the sequential cost

A triangular Jacobian need not come in two blocks; it can be fully triangular. An **autoregressive flow** sets $z_i = \tau\big(x_i;\, c_i(x_{<i})\big)$, each coordinate transformed by an invertible scalar map $\tau$ whose parameters depend only on *earlier* coordinates. Then $\partial z_i/\partial x_j = 0$ for $j > i$: triangular Jacobian, log-det $= \sum_i \log|\partial z_i/\partial x_i|$, again $O(d)$. The two named families differ only in which direction is parallel — and this asymmetry decides what each is *for*:

- **MAF** (masked autoregressive flow) parameterizes the normalizing direction $x \to z$ autoregressively: **density evaluation is one parallel pass** (all conditioners $c_i(x_{<i})$ computed at once from the observed $x$, à la masked networks), but **sampling is sequential** — $x_1$ must be produced before $x_2$'s conditioner can run, so generating one point costs $d$ passes. Built for density estimation.
- **IAF** (inverse autoregressive flow) parameterizes the generative direction $z \to x$ autoregressively in $z$: **sampling is one parallel pass** (all of $z$ is available at once), but **evaluating the density of an external data point is sequential**. Built for fast sampling — its historical role is as a flexible posterior inside variational autoencoders, where you only ever need densities of your *own* samples, which come with their $z$'s attached.

Same math, transposed costs. If you remember one sentence: *MAF is fast at reading, IAF is fast at writing.* Coupling layers (RealNVP, Glow) are the compromise that is one-pass in both directions, at some cost in per-layer expressiveness.

One paragraph on the continuous limit, as a forward pointer. Let the flow be an ODE, $\tfrac{d}{dt}z_t = v_\theta(z_t, t)$: an infinitesimal step $z_{t+\varepsilon} = z_t + \varepsilon v_\theta(z_t,t)$ has Jacobian $I + \varepsilon J_{v_\theta}$, and $\log|\det(I + \varepsilon J)| = \varepsilon\,\mathrm{tr}(J) + O(\varepsilon^2)$, so the log-density evolves by $\tfrac{d}{dt}\log p(z_t) = -\mathrm{tr}\,J_{v_\theta}(z_t, t)$ — a *trace* replaces the determinant, and traces admit cheap unbiased estimates. This is the instantaneous change of variables behind continuous normalizing flows (neural ODEs, FFJORD), and it is the bridge to the flow-matching and diffusion viewpoint that Chapter 13 develops; we plant the flag here and move on.

> **Common Confusion 3.5.** *"My flow trains fine — loss goes down smoothly — so the log-det sign must be right."*
>
> The optimizer cannot tell your mistake from a model. Flip the sign of the log-det and the "likelihood" rewards volume *expansion* at the data instead of contraction; gradient descent will cheerfully inflate the scales, the loss will decrease forever (it is now unbounded below), and samples will be garbage while the loss looks great. Diagnostics that actually catch it: (i) compare model NLL against the standard-normal baseline on the same data — a correct model beats it, a sign-flipped one "beats" it implausibly fast and keeps going; (ii) finite-difference the Jacobian at a few points, as Lab B.2 does — a $10$-line check that would have caught every sign bug the authors of this book have personally written; (iii) sample from the model early and often. Training curves are not correctness proofs.

> **Self-check 3.7.** (a) Why do the networks $s_\theta, t_\theta$ inside a coupling layer need no invertibility or Jacobian computation of their own? (b) A flow on $d = 100$ uses fully dense linear layers as its invertible maps. What does each layer's log-det cost, and which two structural fixes from this section repair it? (c) You need a model that will score millions of externally-provided points per second, and you sample rarely. MAF or IAF — and why? *(Answers in Appendix A.)*

## 3.8 The exponential family

If this chapter has a center of mass, it is here. One algebraic form underlies most of the distributions you use and most of the losses you minimize; learning its calculus once replaces a dozen distribution-specific derivations.

### 3.8.1 Definition and first examples

**Definition 3.15 (exponential family).** Fix a $\sigma$-finite base measure $\mu$ on a space $\mathcal{X}$ (Lebesgue, counting, or other — always stated, per Chapter 2's discipline), a measurable **sufficient statistic** $T : \mathcal{X} \to \mathbb{R}^m$ (the name is justified in Section 3.9), and a measurable $h : \mathcal{X} \to [0, \infty)$ (the **carrier**). For $\eta \in \mathbb{R}^m$ define the **log-partition function**

$$A(\eta) := \log \int_{\mathcal{X}} h(x)\,e^{\langle \eta, T(x)\rangle}\,d\mu(x) \in (-\infty, +\infty],$$

and the **natural parameter space** $\Xi := \{\eta \in \mathbb{R}^m : A(\eta) < \infty\}$. The associated **exponential family** is the set of probability densities, with respect to $\mu$,

$$p_\eta(x) = h(x)\,\exp\big(\langle \eta, T(x)\rangle - A(\eta)\big), \qquad \eta \in \Xi.$$

$A(\eta)$ is precisely the log of the normalizing constant — it is *defined* to make $\int p_\eta \,d\mu = 1$. Families indexed by a conventional parameter $\theta$ (a probability $p$, a rate $\lambda$, a mean-and-variance pair) fit the template through a reparameterization $\eta = \eta(\theta)$; when we write "the Bernoulli family is exponential" we mean after that change of coordinates.

**Worked Example 3.14 (Bernoulli — and softplus/sigmoid demystified).** $\mathcal{X} = \{0, 1\}$, $\mu = $ counting measure, $T(x) = x$, $h \equiv 1$. Start from $P(X = x) = p^x(1-p)^{1-x}$ and exponentiate the logarithm:

$$p^x(1-p)^{1-x} = \exp\Big(x\log\tfrac{p}{1-p} + \log(1-p)\Big),$$

so the natural parameter is the **log-odds (logit)** $\eta = \log\tfrac{p}{1-p}$, and matching the template, $A(\eta) = -\log(1-p) = \log(1 + e^\eta)$ — the **softplus** function. The natural parameter space is all of $\mathbb{R}$ (softplus is finite everywhere). Differentiate:

$$A'(\eta) = \frac{e^\eta}{1+e^\eta} = \sigma(\eta) \quad (\text{the sigmoid}), \qquad A''(\eta) = \sigma(\eta)\big(1 - \sigma(\eta)\big).$$

But $\sigma(\eta) = p = \mathbb{E}[X]$, and $\sigma(1-\sigma) = p(1-p) = \mathrm{Var}(X)$. The first two derivatives of the log-partition function produced the mean and the variance. Theorem 3.17 will show this is not a Bernoulli coincidence but the general law of the family — and with it, the trio (logit, softplus, sigmoid) that saturates deep learning is revealed as (natural parameter, log-partition, mean map) of the simplest exponential family. $\square$

**Worked Example 3.15 (Gaussian, both parameters unknown).** $\mathcal{X} = \mathbb{R}$, $\mu = \lambda$, density $\tfrac{1}{\sqrt{2\pi}\sigma}\exp(-\tfrac{(x-\mu_0)^2}{2\sigma^2})$ (writing $\mu_0$ for the mean to spare the base-measure symbol). Expand the square:

$$\log p = \frac{\mu_0}{\sigma^2}\,x \;-\; \frac{1}{2\sigma^2}\,x^2 \;-\; \Big(\frac{\mu_0^2}{2\sigma^2} + \log\sigma + \tfrac12\log 2\pi\Big),$$

so $T(x) = (x, x^2)$, $\eta = (\eta_1, \eta_2) = \big(\tfrac{\mu_0}{\sigma^2},\, -\tfrac{1}{2\sigma^2}\big)$, $h(x) = (2\pi)^{-1/2}$, and, solving back ($\sigma^2 = -\tfrac{1}{2\eta_2}$, $\mu_0 = -\tfrac{\eta_1}{2\eta_2}$),

$$A(\eta) = \frac{\mu_0^2}{2\sigma^2} + \log \sigma = -\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log(-2\eta_2), \qquad \Xi = \mathbb{R} \times (-\infty, 0).$$

Check the moment machinery in advance of the theorem: $\partial A/\partial\eta_1 = -\tfrac{\eta_1}{2\eta_2} = \mu_0 = \mathbb{E}[X]$ ✓, and $\partial A/\partial\eta_2 = \tfrac{\eta_1^2}{4\eta_2^2} - \tfrac{1}{2\eta_2} = \mu_0^2 + \sigma^2 = \mathbb{E}[X^2]$ ✓ — each derivative delivers the expectation of the corresponding component of $T$. Note $\Xi$ is an open half-plane: openness is not decorative, it is exactly what Theorem 3.17's differentiation argument will need. $\square$

**Worked Example 3.16 (Poisson and Gamma, briefly).** *Poisson:* $\mathcal{X} = \{0,1,2,\dots\}$, $\mu = $ counting, $\tfrac{e^{-\lambda}\lambda^x}{x!} = \tfrac{1}{x!}\exp(x\log\lambda - \lambda)$: so $T(x) = x$, $h(x) = 1/x!$, $\eta = \log\lambda$, $A(\eta) = \lambda = e^\eta$, $\Xi = \mathbb{R}$. (Instant dividend: $A'(\eta) = e^\eta = \lambda = \mathbb{E}[X]$ and $A''(\eta) = \lambda = \mathrm{Var}(X)$ — the equal-mean-and-variance signature of the Poisson, in one differentiation.) *Gamma:* density $\tfrac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha - 1}e^{-\beta x}$ on $(0,\infty)$: $T(x) = (\log x,\, x)$, $\eta = (\alpha - 1, -\beta)$, $h \equiv 1$, $A(\eta) = \log\Gamma(\eta_1 + 1) - (\eta_1 + 1)\log(-\eta_2)$, $\Xi = (-1,\infty)\times(-\infty,0)$. The statistic $(\log x, x)$ explains why Lab B.4 fits a Gamma by matching the sample means of $x$ *and of* $\log x$ — those two averages are the family's whole memory of the data. $\square$

### 3.8.2 The log-partition function is convex, and its derivatives are moments

**Proposition 3.16 (convexity).** $\Xi$ is a convex subset of $\mathbb{R}^m$, and $A$ is convex on $\Xi$.

*Proof.* Let $\eta_0, \eta_1 \in \Xi$, $t \in (0,1)$, and apply Hölder's inequality (Ch. 2) with conjugate exponents $\tfrac1t$ and $\tfrac{1}{1-t}$ to the functions $\big(he^{\langle\eta_0,T\rangle}\big)^{t}$ and $\big(he^{\langle\eta_1,T\rangle}\big)^{1-t}$:

$$\int h\,e^{\langle t\eta_0 + (1-t)\eta_1,\,T\rangle}d\mu = \int \big(he^{\langle \eta_0,T\rangle}\big)^{t}\big(he^{\langle \eta_1,T\rangle}\big)^{1-t}d\mu \le \left(\int he^{\langle\eta_0,T\rangle}d\mu\right)^{t}\left(\int he^{\langle\eta_1,T\rangle}d\mu\right)^{1-t}.$$

The right side is finite, so $t\eta_0 + (1-t)\eta_1 \in \Xi$ (convexity of $\Xi$); taking logs, $A(t\eta_0 + (1-t)\eta_1) \le tA(\eta_0) + (1-t)A(\eta_1)$ (convexity of $A$). $\blacksquare$

**Theorem 3.17 (derivatives of $A$ are moments of $T$).** Let $\eta$ lie in the interior of $\Xi$. Then $A$ is differentiable at $\eta$ (indeed infinitely differentiable on the interior), and

$$\nabla A(\eta) = \mathbb{E}_{\eta}\big[T(X)\big], \qquad \nabla^2 A(\eta) = \mathrm{Cov}_{\eta}\big(T(X)\big).$$

*Proof.* Write $Z(\eta) = e^{A(\eta)} = \int h\,e^{\langle\eta,T\rangle}d\mu$. We differentiate under the integral sign using the DCT-based lemma from Chapter 2, which requires a dominating integrable function valid in a neighborhood. Since $\eta \in \mathrm{int}\,\Xi$, choose $\delta > 0$ with $\eta \pm 2\delta e_i \in \Xi$ for each coordinate direction $e_i$. The elementary bound $|u| \le \tfrac{1}{\delta}\big(e^{\delta u} + e^{-\delta u}\big)$ for all real $u$ (each summand alone dominates $\delta|u|$ on one half-line, since $e^{v} \ge v$) gives, for every $\eta'$ in the ball of radius $\delta$ around $\eta$ (so that $\eta' \pm \delta e_i$ stays in $\Xi$ by convexity),

$$\big|T_i(x)\big|\,h(x)e^{\langle\eta',T(x)\rangle} \;\le\; \frac{1}{\delta}\Big(h(x)e^{\langle\eta' + \delta e_i,\,T(x)\rangle} + h(x)e^{\langle\eta' - \delta e_i,\,T(x)\rangle}\Big),$$

and the right side is $\mu$-integrable with integral $\tfrac{1}{\delta}(Z(\eta' + \delta e_i) + Z(\eta' - \delta e_i)) < \infty$. This is the domination hypothesis; the differentiation lemma (Ch. 2) then yields

$$\frac{\partial Z}{\partial \eta_i}(\eta) = \int T_i(x)\,h(x)\,e^{\langle\eta,T(x)\rangle}\,d\mu(x).$$

Divide by $Z(\eta)$: $\partial_i A = \partial_i \log Z = \tfrac{\partial_i Z}{Z} = \int T_i \,p_\eta\, d\mu = \mathbb{E}_\eta[T_i]$. For second derivatives, dominate $|T_iT_j|e^{\langle\eta',T\rangle} \le \tfrac12(T_i^2 + T_j^2)e^{\langle\eta',T\rangle}$ and each $T_i^2 e^{\langle \eta', T\rangle}$ by $\tfrac{2}{\delta^2}\big(e^{\langle\eta'+\delta e_i,T\rangle} + e^{\langle\eta'-\delta e_i,T\rangle}\big)$ (squaring the previous bound and using $(a+b)^2 \le 2a^2 + 2b^2$); differentiating again,

$$\partial_j\partial_i A = \partial_j\!\left(\frac{\partial_i Z}{Z}\right) = \frac{\partial_j\partial_i Z}{Z} - \frac{\partial_i Z\,\partial_j Z}{Z^2} = \mathbb{E}_\eta[T_iT_j] - \mathbb{E}_\eta[T_i]\,\mathbb{E}_\eta[T_j] = \mathrm{Cov}_\eta(T_i, T_j).$$

Higher derivatives iterate the same domination pattern. $\blacksquare$

Where interiority entered: the dominating function had to be integrable at parameters *slightly beyond* $\eta'$ in every direction, i.e. we spent a $\delta$-margin inside $\Xi$. At boundary points of $\Xi$ there is no margin to spend, and the conclusion can genuinely fail. This is a theorem with its hypotheses load-bearing, not decorative.

Since covariance matrices are positive semidefinite, $\nabla^2 A \succeq 0$ — a second proof of convexity on the interior, and a hint: $A$ fails to be *strictly* convex exactly when some linear combination $\langle a, T(X)\rangle$ has zero variance, i.e. is a.s. constant. That degeneracy has a name:

**Definition 3.18 (minimal representation; mean parameter space).** The representation $(T, h, \mu)$ is **minimal** if no nonzero $a \in \mathbb{R}^m$ and constant $c$ satisfy $\langle a, T(x)\rangle = c$ for $\mu$-almost-every $x$ with $h(x) > 0$. The **mean parameter space** is $\mathcal{M} := \nabla A(\mathrm{int}\,\Xi) = \{\mathbb{E}_\eta[T] : \eta \in \mathrm{int}\,\Xi\}$.

Under a minimal representation, $\mathrm{Cov}_\eta(T) \succ 0$ (a zero-variance direction would violate minimality), so $A$ is strictly convex on $\mathrm{int}\,\Xi$, and consequently $\nabla A : \mathrm{int}\,\Xi \to \mathcal{M}$ is **injective** (Exercise A7 proves this cleanly): each achievable mean of $T$ corresponds to exactly one natural parameter. The pair of coordinate systems $(\eta, \nabla A(\eta))$ — **natural** and **mean** — are two charts on the same family, exchanged by the gradient of $A$ in one direction and, by Legendre duality (the gradient of the convex conjugate $A^*$), in the other. The picture to keep:

<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img" aria-label="Natural and mean parameter duality">
  <style>text{font-family:sans-serif;font-size:13px;fill:currentColor}.cap{font-size:12px}.s{font-size:11px}</style>
  <ellipse cx="170" cy="130" rx="120" ry="75" fill="#3b82f6" opacity="0.15" stroke="#3b82f6"/>
  <text x="105" y="70" fill="#3b82f6">natural space  int Ξ</text>
  <text x="145" y="135">η</text><circle cx="135" cy="130" r="4" fill="#3b82f6"/>
  <ellipse cx="550" cy="130" rx="120" ry="75" fill="#10b981" opacity="0.15" stroke="#10b981"/>
  <text x="490" y="70" fill="#10b981">mean space  M</text>
  <text x="562" y="135">m = E_η[T]</text><circle cx="552" cy="130" r="4" fill="#10b981"/>
  <path d="M 260 105 C 340 70 400 70 470 105" fill="none" stroke="currentColor" stroke-width="1.8" marker-end="url(#arr4)"/>
  <text x="330" y="72">∇A  (moment map)</text>
  <path d="M 470 160 C 400 195 340 195 260 160" fill="none" stroke="currentColor" stroke-width="1.8" marker-end="url(#arr4)"/>
  <text x="300" y="212">∇A*  (Legendre dual / MLE map)</text>
  <defs><marker id="arr4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="currentColor"/></marker></defs>
  <text x="60" y="252" class="s">Under a minimal representation, ∇A is a bijection between the two spaces: Cov(T) ≻ 0 ⇒ strict convexity ⇒ injectivity.</text>
  <text x="60" y="270" class="cap">Figure 3.5 — One family, two coordinate systems. Optimization is convex in η; data lives in M (the empirical mean of T);</text>
  <text x="60" y="286" class="cap">maximum likelihood is exactly the trip from the data's mean back to a natural parameter, ∇A⁻¹ = ∇A*.</text>
</svg>

**Worked Example 3.17 (Categorical, softmax, and what the logits don't determine).** $\mathcal{X} = \{1, \dots, K\}$, counting measure, $T(x) = e_x \in \mathbb{R}^K$ (one-hot), $h \equiv 1$, $\eta \in \mathbb{R}^K$ the **logits**:

$$p_\eta(x) = \exp\big(\eta_x - A(\eta)\big), \qquad A(\eta) = \log\sum_{k=1}^K e^{\eta_k} \ (\textbf{logsumexp}), \qquad \nabla A(\eta) = \mathrm{softmax}(\eta).$$

So softmax is not an arbitrary squashing choice: it is the mean map $\nabla A$ of the categorical exponential family, and logsumexp is its log-partition function — which is why $\nabla \mathrm{logsumexp} = \mathrm{softmax}$, an identity you have differentiated a thousand times without necessarily naming. But this representation is **not minimal**: $\langle \mathbf{1}, T(x)\rangle = 1$ for every $x$ (one-hots sum to one), so $a = \mathbf{1}$ violates Definition 3.18, and indeed $p_{\eta + c\mathbf{1}} = p_\eta$ for every constant $c$ — **logits are only identified up to an additive shift**. Consequences, all of which practitioners meet: softmax classifiers have a flat direction in parameter space (harmless for prediction, fatal for interpreting individual logit values, and the reason weight-space comparisons between runs are ill-posed along $\mathbf{1}$); frameworks stabilize logsumexp by subtracting $\max_k \eta_k$, which is legitimate *because* of this invariance; and the minimal representation is obtained by pinning one class, $\eta_K := 0$, with $T(x) = (\mathbf{1}\{x=1\}, \dots, \mathbf{1}\{x = K-1\})$ — for $K = 2$ this is exactly the Bernoulli logit of Worked Example 3.14. Finally, **temperature**: $p(x) \propto e^{\eta_x/\tau}$ is not a new family, just the natural parameter rescaled to $\eta/\tau$ — sliding the distribution along a ray in $\Xi$, toward uniform as $\tau \to \infty$ and toward the argmax vertex as $\tau \to 0$. $\square$

### 3.8.3 Maximum likelihood is moment matching

**Theorem 3.19 (MLE in an exponential family).** Let $x_1, \dots, x_n$ be i.i.d. observations, and write $\bar T := \tfrac1n\sum_{i=1}^n T(x_i)$. The average log-likelihood

$$\ell(\eta) := \frac1n \sum_{i=1}^n \log p_\eta(x_i) = \frac1n\sum_i \log h(x_i) + \langle \eta, \bar T\rangle - A(\eta)$$

is concave on $\Xi$, and $\nabla \ell(\eta) = \bar T - \nabla A(\eta)$ on $\mathrm{int}\,\Xi$. Consequently:

1. **(Stationarity = moment matching.)** $\hat\eta \in \mathrm{int}\,\Xi$ is a maximizer of $\ell$ **iff** $\nabla A(\hat\eta) = \bar T$, i.e. iff $\mathbb{E}_{\hat\eta}[T(X)] = \bar T$: *the fitted model reproduces the empirical mean of the sufficient statistic, exactly.*
2. **(Existence and uniqueness.)** If the representation is minimal and $\bar T \in \mathcal{M} = \nabla A(\mathrm{int}\,\Xi)$, the MLE exists, is unique, and equals $\hat\eta = (\nabla A)^{-1}(\bar T)$.

*Proof.* Concavity: $\ell$ is a linear function of $\eta$ minus the convex $A$ (Proposition 3.16). The gradient formula is Theorem 3.17. (1): for a differentiable concave function on a convex set, an interior point is a global maximizer iff its gradient vanishes (Ch. 2's convexity toolkit; one line: concavity gives $\ell(\eta') \le \ell(\eta) + \langle\nabla\ell(\eta), \eta' - \eta\rangle$ for all $\eta'$). (2): if $\bar T \in \mathcal{M}$, some $\hat\eta \in \mathrm{int}\,\Xi$ has $\nabla A(\hat\eta) = \bar T$, which by (1) is a maximizer; minimality makes $\nabla A$ injective (strict convexity, Exercise A7), so it is the only stationary point, and strict concavity of $\ell$ on $\mathrm{int}\,\Xi$ rules out any other maximizer. $\blacksquare$

Read statement (1) again; it is one of the great simplifications in statistics. Maximum likelihood — an optimization over a possibly huge family — collapses to solving one *moment equation*: make the model's mean of $T$ equal the data's mean of $T$. For the Gaussian, $T = (x, x^2)$, so the MLE matches the sample mean and the sample second moment — the familiar $\hat\mu = \bar x$, $\hat\sigma^2 = \overline{x^2} - \bar x^2$, no calculus per family required, ever again. *Verification that the shortcut is the whole answer (Lab B.4, seed 0):* $5000$ draws from $\mathrm{Gamma}(\alpha = 2.5, \beta = 1.2)$; solving the two moment equations $\mathbb{E}[X] = \bar x$, $\mathbb{E}[\log X] = \overline{\log x}$ gave $(\hat\alpha, \hat\beta) = (2.512094,\, 1.206314)$; independently maximizing the log-likelihood numerically gave $(2.512094,\, 1.206314)$ — agreement to $8\times10^{-8}$, i.e. to solver tolerance.

And when does the MLE **fail to exist**? Exactly when $\bar T$ falls outside the open set $\mathcal{M}$ — on its boundary, typically. This is not a pathology reserved for theorists; you have trained models where it happens.

- **All heads.** Bernoulli, $n$ tosses, all ones: $\bar T = 1$. The mean space is $\mathcal{M} = \sigma(\mathbb{R}) = (0,1)$ — open! — and $1 \notin \mathcal{M}$. The likelihood $\ell(\eta) = \eta - \log(1+e^\eta)$ is strictly increasing in $\eta$ with supremum $0$ approached as $\eta \to +\infty$: no maximizer. The "MLE" $\hat p = 1$ exists only in the closed mean parameterization, and its natural parameter is at infinity. Any downstream computation that needs $\hat\eta$ (a logit, a log-odds-ratio, a standard error) diverges.

- **Separable data.** Logistic regression is the conditional version of the Bernoulli story: $\eta_i = w^\top x_i$ ties the natural parameter to weights. If some $w_\star$ classifies every training point correctly with margin ($y_i$-signed $w_\star^\top x_i > 0$ for all $i$), then scaling $c\,w_\star$, $c \to \infty$, drives every likelihood term to $1$: the loss infimum is $0$, approached but never attained; **there is no MLE, and gradient descent chases infinity**. This is Theorem 3.19's interiority condition failing in conditional form — the empirical statistic sits on the boundary of what the model can match. *Verification (Lab B.4b, data seed 1):* $100$ points from two well-separated Gaussian blobs (means $(\pm2, 0)$, s.d. $0.5$); full-batch gradient descent, learning rate $0.5$. The weight norm grows without bound — $\|w\| = 2.81$ at iteration $10^2$, $4.30$ at $10^3$, $6.05$ at $10^4$, $8.06$ at $10^5$ (the $\sim\log t$ growth is the known asymptotic rate) — while the mean NLL decays $0.008456 \to 0.000019$, vanishing but never zero. In practice this is why regularization (weight decay restores a finite minimizer), early stopping, or margin-based analyses are not optional garnish on logistic-type losses: on separable data the unregularized problem literally has no solution. (The direction $w/\|w\|$ does converge — to the max-margin separator, a celebrated implicit-bias result; reference in the further-reading list.)

> **Common Confusion 3.6.** *"My two training runs learned visibly different logit vectors for the same inputs, so the runs disagree."*
>
> Check the difference first: if the logit vectors differ by (approximately) a constant shift per input, the two runs define the *same* conditional distributions — the softmax non-identifiability of Worked Example 3.17 in action. Only shift-invariant functionals of logits (differences $\eta_j - \eta_k$, the softmax itself, margins) are meaningful model outputs; raw logit magnitudes are coordinates in a direction the likelihood cannot see. The same caution applies to interpreting "the logit for class $k$ went up": up relative to what?

### 3.8.4 KL divergence inside the family is a Bregman divergence

**Proposition 3.20 (KL = Bregman divergence of $A$).** For $\eta_1, \eta_2 \in \mathrm{int}\,\Xi$ in a common exponential family,

$$D_{\mathrm{KL}}\big(p_{\eta_1}\,\|\,p_{\eta_2}\big) = A(\eta_2) - A(\eta_1) - \big\langle \eta_2 - \eta_1,\; \nabla A(\eta_1)\big\rangle \;=:\; B_A(\eta_2, \eta_1).$$

*Proof.* Three lines from the definitions:

$$D_{\mathrm{KL}}(p_{\eta_1}\|p_{\eta_2}) = \mathbb{E}_{\eta_1}\!\left[\log\frac{p_{\eta_1}(X)}{p_{\eta_2}(X)}\right] = \mathbb{E}_{\eta_1}\big[\langle \eta_1 - \eta_2, T(X)\rangle\big] - A(\eta_1) + A(\eta_2)$$

(the carriers $h$ cancel in the ratio — same family, same $h$, same $\mu$), and $\mathbb{E}_{\eta_1}[T] = \nabla A(\eta_1)$ by Theorem 3.17. Rearranged, this is the displayed expression: the gap between $A$ at $\eta_2$ and its first-order Taylor expansion around $\eta_1$ — the **Bregman divergence** of $A$. $\blacksquare$

Convexity of $A$ makes $B_A \ge 0$ (Taylor underestimation), re-deriving Gibbs' inequality (Ch. 2) inside the family with equality iff $\eta_1 = \eta_2$ (minimal case). Two specializations you use weekly, derived in Part C.5: for Gaussians with common covariance $\Sigma$, $A(\eta) = \tfrac12\eta^\top\Sigma\eta$ is quadratic and its Bregman divergence is the squared Mahalanobis distance $\tfrac12(\mu_1-\mu_2)^\top\Sigma^{-1}(\mu_1-\mu_2)$; for Categoricals, $B_A$ unwinds to $\sum_k \pi_{1,k}\log\tfrac{\pi_{1,k}}{\pi_{2,k}}$, i.e. **cross-entropy minus entropy** — so minimizing cross-entropy of a softmax classifier against labels *is* minimizing a KL inside the categorical family, *is* exponential-family maximum likelihood. Three descriptions of your classification loss; one object. This identity is also the seed from which **natural gradient** descent (precondition by $\nabla^2 A = \mathrm{Cov}(T)$, the Fisher information) and **mirror descent** (gradient steps in the dual/mean coordinates) grow — named here, developed in the optimization chapter.

One paragraph on **generalized linear models**, because it packages this section for daily use. A GLM assumes the response follows an exponential family whose natural parameter is linear in features: $\eta_i = w^\top x_i$ (the **canonical link**). Then: Gaussian response $\Rightarrow$ linear regression (squared loss); Bernoulli $\Rightarrow$ logistic regression (log loss); Poisson $\Rightarrow$ Poisson regression (log link, count data); Categorical $\Rightarrow$ softmax regression. In every case the average negative log-likelihood is $A(w^\top x_i) - y_i\,w^\top x_i$ averaged over data — convex in $w$ by Proposition 3.16 and the affine composition rule — with gradient $\tfrac1n\sum_i (\nabla A(w^\top x_i) - y_i)\,x_i = \tfrac1n\sum_i(\hat y_i - y_i)x_i$: **prediction error times features**, the formula every one of these regressions shares, now explained once. A neural network classifier is precisely a GLM whose features are learned: the last linear layer plus softmax cross-entropy is the categorical GLM sitting on top of $\phi_\theta(x)$.

> **Self-check 3.8.** (a) For the Poisson family, verify $\mathcal{M} = (0,\infty)$ and describe a dataset whose $\bar T$ lies outside $\mathcal{M}$ — what happens to the MLE? (b) Why does the same symbol $T$ deserve the name "sufficient statistic" already in Definition 3.15 — what does Theorem 3.19(1) say the fitted model remembers about the data? (c) Cross-entropy loss is convex in the logits. Which two results of this section combine to prove that in one line? (d) Is the exponential family form preserved under temperature scaling, and what does that imply about "temperature" as a hyperparameter? *(Answers in Appendix A.)*

## 3.9 Sufficiency: when a statistic loses nothing

Honesty flag (b) governs this section, so we state it precisely before anything else. The fully general definition of sufficiency says: *the conditional distribution of the data given $T$ does not depend on $\theta$.* Making "conditional distribution of $X$ given $T = t$" rigorous for arbitrary spaces requires **regular conditional distributions**, machinery a later chapter constructs on top of conditional expectation. So this section proceeds in two honest steps: in the **discrete case**, conditional probabilities are elementary (Definition 1.10), and we define sufficiency in full generality *there* and prove the factorization theorem completely, in both directions. In the **dominated case** — every $P_\theta$ has a density with respect to a common $\sigma$-finite $\mu$, which is the only setting ML ever occupies — we *adopt* the factorization property as the working definition, and record in the not-proved table both the general equivalence (the Halmos–Savage theorem) and the promise that the conditional-distribution formulation returns, with proofs, once conditioning is built.

### 3.9.1 The idea, and the discrete case done right

You have $n$ observations and a model $\{P_\theta\}$. A statistic $T$ (any measurable function of the data) is *sufficient* if, once you know $T$, the remaining detail of the data is — as far as $\theta$ is concerned — pure noise: something whose distribution is the same for every $\theta$, hence carries no evidence. The compressed summary $T(x)$ is then a lossless compression *for inference about this model*.

**Definition 3.21 (sufficiency, discrete case).** Let $\mathcal{X}$ be countable and $\{P_\theta\}_{\theta \in \Theta}$ a family of laws on it, with pmfs $p_\theta$. A statistic $T$ is **sufficient for $\{P_\theta\}$** if for every $t$ with $P_\theta(T(X) = t) > 0$ and every $x$, the conditional probability

$$P_\theta\big(X = x \,\big|\, T(X) = t\big)$$

is the same for all $\theta$ (for which it is defined).

**Theorem 3.22 (Fisher–Neyman factorization, discrete case).** In the setting above, $T$ is sufficient **iff** there exist functions $g_\theta \ge 0$ (depending on $\theta$ and on the value of $T$ only) and $h \ge 0$ (depending on $x$ only) such that

$$p_\theta(x) = g_\theta\big(T(x)\big)\,h(x) \qquad \text{for all } x, \theta.$$

*Proof.* ($\Leftarrow$, factorization implies sufficiency.) Fix $t$ with $P_\theta(T = t) > 0$ and $x$ with $T(x) = t$ (for $T(x) \ne t$ the conditional probability is $0$ for every $\theta$ — already $\theta$-free). By Definition 1.10,

$$P_\theta\big(X = x \mid T = t\big) = \frac{P_\theta(X = x)}{P_\theta(T = t)} = \frac{g_\theta(t)\,h(x)}{\sum_{x': T(x') = t} g_\theta(t)\,h(x')} = \frac{h(x)}{\sum_{x' : T(x') = t} h(x')},$$

where the cancellation of $g_\theta(t)$ is legitimate because $P_\theta(T = t) = g_\theta(t)\sum_{x':T(x')=t}h(x') > 0$ forces $g_\theta(t) > 0$. The result is manifestly $\theta$-free.

($\Rightarrow$, sufficiency implies factorization.) By sufficiency, the conditional probability $P_\theta\big(X = x \mid T = T(x)\big)$ is the same for every $\theta$ for which it is defined; call this common, $\theta$-free value $q(x)$. Define $g_\theta(t) := P_\theta(T(X) = t)$. Then for any $x$, with $t = T(x)$:

$$p_\theta(x) = P_\theta(X = x) = P_\theta\big(X = x,\, T(X) = t\big) = P_\theta(T = t)\;P_\theta\big(X = x \mid T = t\big) = g_\theta(t)\,q(x)$$

whenever $P_\theta(T = t) > 0$; and when $P_\theta(T = t) = 0$, both sides are $0$. This is the factorization with $h = q$. $\blacksquare$

The proof is almost embarrassingly short once conditional probability is available — which is exactly the point of the honesty flag: in general spaces the *statement* "conditional law of $X$ given $T$" is the hard part, not the argument.

**Definition 3.23 (sufficiency, dominated case — working definition).** Let $\{P_\theta\}$ all be absolutely continuous with respect to a $\sigma$-finite $\mu$, with densities $p_\theta = dP_\theta/d\mu$ (Radon–Nikodym, Ch. 2). A statistic $T$ is **sufficient** if there exist measurable $g_\theta \ge 0$ and $h \ge 0$, not depending on $\theta$, with

$$p_\theta(x) = g_\theta\big(T(x)\big)\,h(x) \qquad \text{for } \mu\text{-almost every } x, \text{ for all } \theta.$$

By Theorem 3.22 this agrees with the conditional definition whenever both make elementary sense; the general equivalence (and the theorem that in the dominated case the two notions coincide) is Halmos–Savage, table entry, later chapter. Two immediate sanity checks the definition passes: $T(x) = x$ (the whole data) is always sufficient ($g_\theta = p_\theta$, $h = 1$); and any *invertible* reshaping of a sufficient statistic is sufficient (compose $g_\theta$ with the inverse — by Lemma 1.22, information is what matters, not packaging).

**Definition 3.24 (minimal sufficiency).** A sufficient statistic $T$ is **minimal** if for every sufficient statistic $S$ there is a measurable $\varphi$ with $T = \varphi(S)$ ($P_\theta$-a.s. for all $\theta$): $T$ is a function of every other sufficient statistic — the coarsest lossless summary, the true dimension of what the data says about the model.

### 3.9.2 Sufficiency in action

**Proposition 3.25 (exponential families compress; minimally so when minimal).** Let $x_1,\dots,x_n$ be i.i.d. from an exponential family (Definition 3.15). Then the statistic $S_n(x_{1:n}) = \sum_{i=1}^n T(x_i)$ — of fixed dimension $m$, no matter how large $n$ — is sufficient for the family of joint distributions. If moreover the representation is minimal, $S_n$ is minimally sufficient (stated; proof via the likelihood-ratio characterization of minimality, table entry).

*Proof (of sufficiency).* The joint density w.r.t. $\mu^{\otimes n}$ (product base measure, Fubini–Tonelli, Ch. 2) is

$$p_\eta(x_{1:n}) = \prod_{i=1}^n h(x_i)e^{\langle\eta, T(x_i)\rangle - A(\eta)} = \underbrace{\Big(\prod_i h(x_i)\Big)}_{=:h(x_{1:n})}\;\underbrace{\exp\big(\langle \eta, S_n(x_{1:n})\rangle - nA(\eta)\big)}_{=:g_\eta(S_n(x_{1:n}))},$$

which is a factorization in the form of Definition 3.23. $\blacksquare$

This is why Theorem 3.19 could reduce MLE to $\bar T = S_n/n$: the likelihood *only sees* the data through $S_n$. A million Gaussian observations and the pair $(\sum x_i, \sum x_i^2)$ yield identical inferences — identical likelihood functions, identical MLEs, identical posteriors for any prior. That is the rigorous meaning of "two numbers summarize the data" — and, spelled out for the deep-learning reader: **the entire training set enters an exponential-family fit only through one $m$-dimensional average**, which is why classical statistics could be done by hand and why streaming/distributed fitting of such models is trivial (means of $T$ merge across shards).

**Worked Example 3.18 (three models, three compression behaviors).** *(a) Gaussian:* by Proposition 3.25 with $T = (x, x^2)$ (Worked Example 3.15): $n$ real observations compress to $2$ numbers — sufficiency as radical compression. *(b) Uniform$[0,\theta]$:* the joint density w.r.t. $\lambda_n$ is

$$p_\theta(x_{1:n}) = \prod_{i=1}^n \tfrac1\theta \mathbf{1}\{0 \le x_i \le \theta\} = \underbrace{\theta^{-n}\,\mathbf{1}\{x_{(n)} \le \theta\}}_{g_\theta(x_{(n)})}\;\cdot\;\underbrace{\mathbf{1}\{x_{(1)} \ge 0\}}_{h(x_{1:n})},$$

a factorization through the **maximum** $x_{(n)}$ alone — one number, and Section 3.4's order-statistics analysis (Worked Example 3.10) now reads as the sampling theory of a sufficient statistic. Note this family is *not* exponential (its support moves with $\theta$; Exercise A6), yet sufficiency compresses beautifully anyway: exponential structure is sufficient for compression, not necessary. *(c) Cauchy:* location family $p_\theta(x) = \tfrac{1}{\pi(1 + (x-\theta)^2)}$. Here compression **fails**: no sufficient statistic of fixed finite dimension exists for i.i.d. samples of growing size $n$ — the order statistics (the whole sorted sample, dimension $n$) are minimally sufficient. This nonexistence is a theorem we cite rather than prove (it follows from the Pitman–Koopman–Darmois circle of results; table entry). The prose reason is worth absorbing even without the proof: the Cauchy likelihood $\prod_i \tfrac{1}{1 + (x_i - \theta)^2}$ is a product of rational functions whose shape in $\theta$ genuinely depends on every observation separately — it can be multimodal, and a far-out observation reshapes it in a way no running average anticipates. Heavy tails mean any single point might be a wild excursion *or* the most informative point in the sample, and no fixed-dimensional summary can hold that distinction open. This is the statistical root of a practical habit: with heavy-tailed data (finance, network traffic, LLM loss spikes), practitioners keep raw data and use order-statistics-based methods (medians, quantiles, trimming) — the theory just told you the shortcut you're tempted to take (running mean and variance) does not exist. $\square$

> **Common Confusion 3.7.** *"The sample mean is a sufficient statistic. So I can always keep just the mean."*
>
> Sufficient *for what model*? Sufficiency is a three-way relation between a statistic, the data, and a **family** $\{P_\theta\}$; drop the family and the word is meaningless. The mean is sufficient for the Gaussian-with-known-variance family. For the Laplace location family, it is not — there the likelihood factors through no fixed small statistic and the *median* is the MLE, so a mean-only summary throws away exactly the evidence that model treats as primary. For the Uniform$[0,\theta]$ family, keeping the mean and discarding the max is discarding the *entire* content of the sufficient statistic. And if your model family is "all distributions" (the nonparametric setting of much of ML), nothing short of the full empirical distribution $\hat P_n$ is sufficient. Compression guarantees are always purchased with modeling assumptions; when someone says "sufficient," ask "for which family?"

> **Self-check 3.9.** (a) Factor the i.i.d. Bernoulli joint pmf to exhibit $\sum_i x_i$ as sufficient. (b) In the Uniform$[0,\theta]$ factorization, why must the indicator $\mathbf{1}\{x_{(n)} \le \theta\}$ live in $g_\theta$ and not in $h$? (c) True or false, with reason: a sufficient statistic for a family remains sufficient for any subfamily. (d) What is the minimal sufficient statistic when the model family is a *single* distribution $\{P_0\}$? *(Answers in Appendix A.)*

## 3.10 The distribution zoo, organized by use

You now own every tool the zoo requires: transformations (§3.1–3.2, 3.5), convolutions (§3.3), order statistics (§3.4), mixtures (below), and the exponential-family lens (§3.8). This section is a working tour, not a museum: each entry answers *when would I actually reach for this?*, and every arrow in the relationship map (Figure 3.6) is **licensed** — proved in this chapter, proved in an exercise with full solution, or explicitly logged in the not-proved table. No folklore edges.

### 3.10.1 Location–scale families: half the zoo in one corollary

**Definition 3.26 (location–scale family).** Fix a "standard" random variable $Z$ with density $p_Z$ (w.r.t. $\lambda$). The **location–scale family** generated by $Z$ is $\{X = \mu + \sigma Z : \mu \in \mathbb{R}, \sigma > 0\}$.

**Corollary 3.27 (location–scale densities).** $X = \mu + \sigma Z$ has density

$$p_X(x) = \frac{1}{\sigma}\,p_Z\!\left(\frac{x - \mu}{\sigma}\right).$$

*Proof.* Theorem 3.1 (equivalently Worked Example 3.1) with the affine map $g(z) = \mu + \sigma z$, whose inverse is $z = (x-\mu)/\sigma$ with derivative $1/\sigma$. $\blacksquare$

Gaussian, Laplace, Cauchy, Uniform, Gumbel, and Student-$t$ (at fixed $\nu$) are all location–scale families: learn each one's *standard* member and Corollary 3.27 generates the rest. This is why "standardization" $z = (x - \mu)/\sigma$ works, why one table of $\Phi$ sufficed for a century of statistics, why `scipy.stats` gives every continuous distribution `loc` and `scale` arguments, and why a flow's final layer often needs only an affine map to fix means and scales — the shape is the hard part; location and scale are a corollary.

### 3.10.2 The reference table

Densities are w.r.t. Lebesgue measure for continuous entries, counting measure for discrete ones. "Reach for it when" is the column to actually read.

| Distribution | Support | pmf / pdf | Mean | Variance | Reach for it when |
|---|---|---|---|---|---|
| Bernoulli$(p)$ | $\{0,1\}$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | Binary outcomes; the atom of classification (WE 3.14) |
| Categorical$(\pi)$ | $\{1..K\}$ | $\pi_x$ | — | — | One of $K$ classes; softmax's target (WE 3.17) |
| Binomial$(n,p)$ | $\{0..n\}$ | $\binom nx p^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | Counts of successes in $n$ known, independent trials |
| Poisson$(\lambda)$ | $\{0,1,\dots\}$ | $e^{-\lambda}\lambda^x/x!$ | $\lambda$ | $\lambda$ | Counts of rare events, no fixed trial count; mean $\approx$ variance in the data |
| Geometric$(p)$ | $\{1,2,\dots\}$ | $(1-p)^{x-1}p$ | $1/p$ | $\tfrac{1-p}{p^2}$ | Trials until first success; memoryless in discrete time |
| Neg. Binomial$(r,p)$ | $\{0,1,\dots\}$ | $\binom{x + r - 1}{x}(1-p)^{x}p^{r}$ | $r\tfrac{1-p}{p}$ | $r\tfrac{1-p}{p^2}$ | **Overdispersed** counts, variance $>$ mean (WE 3.19: it is a Gamma-mixed Poisson) |
| Uniform$[a,b]$ | $[a,b]$ | $\tfrac{1}{b-a}$ | $\tfrac{a+b}2$ | $\tfrac{(b-a)^2}{12}$ | Maximal ignorance on a known range; the source of all sampling (§3.6) |
| Exponential$(\beta)$ | $(0,\infty)$ | $\beta e^{-\beta x}$ | $1/\beta$ | $1/\beta^2$ | Memoryless waiting times; inter-arrival gaps |
| Gamma$(\alpha,\beta)$ | $(0,\infty)$ | $\tfrac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Positive, right-skewed quantities; waiting time to $\alpha$-th event (WE 3.9); rate's conjugate prior |
| Beta$(a,b)$ | $(0,1)$ | $\tfrac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a-1}(1-x)^{b-1}$ | $\tfrac{a}{a+b}$ | $\tfrac{ab}{(a+b)^2(a+b+1)}$ | Random probabilities; order statistics of uniforms (Ex. A5); Bernoulli's conjugate prior |
| Dirichlet$(\alpha_{1:K})$ | simplex $\Delta^{K-1}$ | $\propto \prod_k x_k^{\alpha_k - 1}$ | $\tfrac{\alpha_k}{\sum\alpha}$ | see §3.10.3 | Random probability *vectors*; Categorical's conjugate prior |
| Gaussian$(\mu,\sigma^2)$ | $\mathbb{R}$ | WE 3.15 | $\mu$ | $\sigma^2$ | Sums of many small effects (CLT, Ch. 5); max entropy given mean & variance; default noise |
| Student-$t_\nu$ | $\mathbb{R}$ | $\propto (1 + x^2/\nu)^{-\frac{\nu+1}{2}}$ | $0$ ($\nu>1$) | $\tfrac{\nu}{\nu-2}$ ($\nu>2$) | Gaussian-like center, heavy tails; robust regression; small-sample inference |
| Laplace$(\mu,b)$ | $\mathbb{R}$ | $\tfrac1{2b}e^{-\lvert x-\mu\rvert/b}$ | $\mu$ | $2b^2$ | Sparse noise; MAP with Laplace prior $=$ lasso (below) |
| Cauchy$(\theta)$ | $\mathbb{R}$ | $\tfrac{1}{\pi(1+(x-\theta)^2)}$ | **undefined** | undefined | Stress-testing; ratios of Gaussians (Ex. A2); tails so heavy means fail (Ch. 2) |
| $\chi^2_k$ | $(0,\infty)$ | $= \mathrm{Gamma}(\tfrac k2,\tfrac12)$ | $k$ | $2k$ | Sums of squared Gaussians (WE 3.5, 3.9); norms of Gaussian vectors; test statistics |
| Gumbel | $\mathbb{R}$ | $e^{-x}e^{-e^{-x}}$ | $\gamma \approx 0.5772$ | $\pi^2/6$ | Maxima of light-tailed samples (§3.4); the argmax trick (§3.11) |
| von Mises$(\mu,\kappa)$ | circle | $\propto e^{\kappa\cos(x-\mu)}$ | — | — | Angles and phases; "Gaussian on the circle" |

Two one-liners promised above. **Laplace $\Rightarrow$ lasso:** with likelihood $y_i \sim \mathcal{N}(w^\top x_i, \sigma^2)$ and i.i.d. Laplace prior $w_j \sim \mathrm{Laplace}(0, b)$, the negative log-posterior is, up to constants,

$$\frac{1}{2\sigma^2}\sum_i (y_i - w^\top x_i)^2 + \frac1b\|w\|_1,$$

so **MAP estimation with a Laplace prior is exactly the lasso** — $\ell_1$ regularization is a distributional assumption wearing an optimization costume (and Gaussian priors, by the same one line, are ridge/weight decay). **Cauchy has no mean:** $\int |x|\,p(x)\,dx = \infty$ (Ch. 2's integrability failure), so sample means of Cauchy data do not converge — they *are* Cauchy at every $n$ (a stability fact provable with Chapter 5's tools) — and any pipeline that monitors a running mean of heavy-tailed quantities is monitoring noise.

### 3.10.3 Four deep dives

**Gaussian.** Everything about it is a reprise: its density was *derived* (WE 3.11) from white noise plus one affine map, its normalizer is Lemma 3.6's volume factor, it is closed under affine maps (Thm 3.7), under convolution (WE 3.8), and under conditioning and marginalization (both affine operations on the $(\mu, \Sigma)$ parameterization — worked out when Gaussian processes need them). It is the exponential family with $T = (x, x^2)$ (WE 3.15), and — a fact proved in the information-theory chapter — the **maximum-entropy** distribution given a mean and variance: assuming Gaussianity is assuming *nothing else*. That is its real license in ML: not that data is Gaussian (it rarely is), but that the Gaussian adds the fewest assumptions beyond second moments, and every formula stays closed-form. Its one systematic sin: tails so light ($e^{-x^2/2}$) that it is chronically surprised by outliers — which is the cue for the Student-$t$.

**Gamma.** The distribution of positive, right-skewed quantities, with three independent claims to relevance. (i) *Waiting times:* the time to the $\alpha$-th arrival of a memoryless process (WE 3.9's induction). (ii) *Exponential-family structure:* $T = (\log x, x)$ (WE 3.16), so fitting matches $\overline{x}$ and $\overline{\log x}$ (Lab B.4 verified moment matching $=$ MLE to $8\times10^{-8}$). (iii) *The conjugate prior for rates:* a Gamma prior on a Poisson rate (or Exponential rate) yields a Gamma posterior — the mechanism is visible without any Bayesian machinery: the Poisson likelihood in $\lambda$ is $\propto \lambda^{\sum x_i}e^{-n\lambda}$, which is *shaped like* a Gamma density in $\lambda$, so multiplying by a Gamma prior stays Gamma (conjugacy in one sentence; full development in the Bayesian chapter). Its special cases: Exponential ($\alpha = 1$), $\chi^2_k$ ($\alpha = k/2, \beta = 1/2$).

**Student-$t$.** Take $Z \sim \mathcal{N}(0,1)$ and, independently, $V \sim \chi^2_\nu$; then

$$X = \frac{Z}{\sqrt{V/\nu}} \;\sim\; t_\nu.$$

Read the construction, not just the formula: $X$ is a Gaussian whose **scale is itself random** — $1/\sqrt{V/\nu}$ is large exactly when the $\chi^2$ draw is small — i.e. the $t$ is a *scale mixture of Gaussians* (equivalently: Gaussian with a Gamma-distributed random precision; the density derivation is an integral in the pattern of WE 3.19 below and is logged in the table with references). Heavy tails, from this viewpoint, are not exotic — they are ordinary Gaussians with *uncertainty about the scale*, and that is exactly how they arise in practice: pooled data whose subgroups have different variances is tail-heavy even if every subgroup is Gaussian. As $\nu \to \infty$, $V/\nu \to 1$ (concentration, Ch. 4) and $t_\nu \to \mathcal{N}(0,1)$; at $\nu = 1$ it *is* the Cauchy. Reach for it when residuals show outliers a Gaussian can't forgive: swapping Gaussian for $t_\nu$ likelihoods turns squared loss into a loss that *saturates* for large residuals, which is robust regression by another name.

**Dirichlet.** The distribution on the probability simplex $\Delta^{K-1} = \{x \ge 0 : \sum_k x_k = 1\}$, density $\propto \prod_k x_k^{\alpha_k - 1}$ with respect to Lebesgue measure *on the simplex* (a $(K-1)$-dimensional set — Chapter 1's measure-zero lesson applies: it has no density in $\mathbb{R}^K$!). It is what Beta becomes for probability *vectors* ($K = 2$ recovers Beta), and it is the Categorical/Multinomial's conjugate prior by the same one-sentence mechanism as Gamma–Poisson: the multinomial likelihood $\propto \prod_k \pi_k^{n_k}$ has Dirichlet shape in $\pi$. Its geometry is controlled by $\alpha$: all $\alpha_k > 1$ concentrates mass in the interior (dense mixtures); all $\alpha_k < 1$ pushes mass to the *corners and edges* of the simplex (sparse mixtures — most coordinates near zero), which is why topic models and mixture priors use $\alpha < 1$ when they want each document/cluster to use few topics/components; $\alpha_k \equiv 1$ is uniform on the simplex. Mean $\mathbb{E}[x_k] = \alpha_k/\sum_j \alpha_j$; the total $\sum_j\alpha_j$ acts as a concentration (pseudo-count) knob, with variance $\mathrm{Var}(x_k) = \tfrac{\mathbb{E}[x_k](1-\mathbb{E}[x_k])}{1 + \sum_j \alpha_j}$ shrinking as it grows.

### 3.10.4 A mixture worth deriving: Negative Binomial = Gamma-mixed Poisson

**Worked Example 3.19 (overdispersion explained, then verified).** Count data routinely violates the Poisson's signature $\mathrm{Var} = \mathbb{E}$: single-cell RNA-seq gene counts, word frequencies, goals per match, packets per interval all show $\mathrm{Var} > \mathbb{E}$ (**overdispersion**). The generative explanation: the rate itself varies across units. Model the two stages: $\Lambda \sim \mathrm{Gamma}(\alpha, \beta)$, then $K \mid \Lambda \sim \mathrm{Poisson}(\Lambda)$. The marginal pmf integrates the rate out (the integral is finite-everything, Tonelli, Ch. 2):

$$P(K = k) = \int_0^\infty \frac{e^{-\lambda}\lambda^k}{k!}\cdot\frac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha - 1}e^{-\beta\lambda}\,d\lambda = \frac{\beta^\alpha}{k!\,\Gamma(\alpha)}\int_0^\infty \lambda^{k + \alpha - 1}e^{-(1+\beta)\lambda}\,d\lambda = \frac{\Gamma(k+\alpha)}{k!\,\Gamma(\alpha)}\cdot\frac{\beta^\alpha}{(1+\beta)^{k+\alpha}},$$

recognizing the Gamma-function integral $\int_0^\infty \lambda^{s-1}e^{-c\lambda}d\lambda = \Gamma(s)/c^s$. Setting $r = \alpha$ and $p = \tfrac{\beta}{1+\beta}$, this is exactly the Negative Binomial pmf $\binom{k + r - 1}{k}(1-p)^k p^r$ (with the binomial coefficient read through Gamma functions for non-integer $r$). Moments, by the two-stage structure (law of total expectation/variance — used here in the elementary form provable by summing over $\lambda$-slices; the general tower property is the conditioning chapter's): $\mathbb{E}[K] = \mathbb{E}[\Lambda] = \tfrac\alpha\beta$ and

$$\mathrm{Var}(K) = \underbrace{\mathbb{E}[\Lambda]}_{\text{Poisson part}} + \underbrace{\mathrm{Var}(\Lambda)}_{\text{rate spread}} = \frac{\alpha}{\beta} + \frac{\alpha}{\beta^2} = \frac{\alpha}{\beta}\Big(1 + \frac1\beta\Big) \;>\; \mathbb{E}[K]:$$

overdispersion is not a defect to be patched but the *arithmetic signature of a random rate*, with the excess $\mathrm{Var}(\Lambda)$ measuring rate heterogeneity. *Verification (seed 0, $n = 10^6$, $\alpha = 3$, $\beta = 1.5$):* empirical mean $2.0019$ vs. theory $2.0000$; empirical variance $3.3343$ vs. theory $3.3333$; and the empirical pmf at $k = 0,\dots,8$, $(0.2155, 0.2595, 0.2068, 0.1384, 0.0833, 0.0466, 0.0247, 0.0128, 0.0064)$, matches $\mathrm{NB}(r = 3, p = 0.6)$'s $(0.2160, 0.2592, 0.2074, 0.1382, 0.0829, 0.0464, 0.0248, 0.0127, 0.0064)$ to within simulation error. This is why NB regression, not Poisson regression, is the default for real count data — and the derivation pattern (integrate a conjugate prior against a likelihood) is one you will reuse constantly: it is the same integral that computes Bayesian marginal likelihoods. $\square$

<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img" aria-label="Relationship map of distributions with licensed edges">
  <style>text{font-family:sans-serif;font-size:13px;fill:currentColor}.s{font-size:10.5px}.cap{font-size:12px}
  .node{fill:none;stroke:currentColor;stroke-width:1.3;rx:8}</style>
  <defs><marker id="arr5" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="currentColor"/></marker></defs>
  <!-- nodes -->
  <rect class="node" x="30" y="30" width="90" height="30" rx="8"/><text x="42" y="50">Bernoulli</text>
  <rect class="node" x="180" y="30" width="90" height="30" rx="8"/><text x="192" y="50">Binomial</text>
  <rect class="node" x="330" y="30" width="90" height="30" rx="8"/><text x="345" y="50">Poisson</text>
  <rect class="node" x="500" y="30" width="110" height="30" rx="8"/><text x="508" y="50">Neg. Binomial</text>
  <rect class="node" x="330" y="130" width="90" height="30" rx="8"/><text x="345" y="150">Gamma</text>
  <rect class="node" x="180" y="130" width="110" height="30" rx="8"/><text x="188" y="150">Exponential</text>
  <rect class="node" x="30" y="130" width="90" height="30" rx="8"/><text x="45" y="150">Uniform</text>
  <rect class="node" x="30" y="230" width="90" height="30" rx="8"/><text x="55" y="250">Beta</text>
  <rect class="node" x="180" y="230" width="90" height="30" rx="8"/><text x="192" y="250">Dirichlet</text>
  <rect class="node" x="500" y="130" width="70" height="30" rx="8"/><text x="515" y="150">χ²_k</text>
  <rect class="node" x="480" y="230" width="90" height="30" rx="8"/><text x="492" y="250">Gaussian</text>
  <rect class="node" x="630" y="230" width="100" height="30" rx="8"/><text x="642" y="250">Student-t</text>
  <rect class="node" x="630" y="330" width="90" height="30" rx="8"/><text x="645" y="350">Cauchy</text>
  <rect class="node" x="330" y="330" width="90" height="30" rx="8"/><text x="345" y="350">Gumbel</text>
  <rect class="node" x="480" y="330" width="90" height="30" rx="8"/><text x="500" y="350">t → N</text>
  <!-- edges -->
  <line x1="120" y1="45" x2="176" y2="45" stroke="#3b82f6" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="122" y="38" fill="#3b82f6">sum of n (WE 3.9b)</text>
  <line x1="270" y1="45" x2="326" y2="45" stroke="#10b981" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="272" y="38" fill="#10b981">limit (Ex. A4)</text>
  <line x1="420" y1="45" x2="496" y2="45" stroke="#f59e0b" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="408" y="26" fill="#f59e0b">Gamma-mixed (WE 3.19)</text>
  <path d="M 375 130 C 375 100 375 90 375 64" fill="none" stroke="#f59e0b" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="382" y="100" fill="#f59e0b">rate prior</text>
  <line x1="290" y1="145" x2="326" y2="145" stroke="#3b82f6" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="262" y="122" fill="#3b82f6">sum of n (WE 3.9a)</text>
  <path d="M 120 145 C 150 145 150 145 176 145" fill="none" stroke="#3b82f6" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="118" y="168" fill="#3b82f6">−log U (WE 3.13)</text>
  <line x1="420" y1="145" x2="496" y2="145" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="424" y="138">= Gamma(k/2,1/2) (WE 3.5+3.9)</text>
  <path d="M 535 160 C 530 190 528 200 527 226" fill="none" stroke="#3b82f6" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="538" y="196" fill="#3b82f6">Z² (WE 3.5)</text>
  <path d="M 570 250 C 600 250 605 250 626 250" fill="none" stroke="#f59e0b" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="574" y="242" fill="#f59e0b">scale mix (§3.10.3, table)</text>
  <path d="M 680 260 C 678 290 677 300 676 326" fill="none" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="686" y="296">ν = 1</text>
  <path d="M 65 160 C 65 190 65 200 65 226" fill="none" stroke="#10b981" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="70" y="196" fill="#10b981">order stats (Ex. A5)</text>
  <line x1="120" y1="245" x2="176" y2="245" stroke="currentColor" stroke-width="1.6" marker-end="url(#arr5)"/><text class="s" x="124" y="238">K=2 case (§3.10.3)</text>
  <path d="M 90 130 C 200 90 300 320 326 340" fill="none" stroke="#10b981" stroke-width="1.3" marker-end="url(#arr5)"/><text class="s" x="150" y="300" fill="#10b981">max of n, rescaled (§3.4, table: FTG)</text>
  <line x1="570" y1="345" x2="576" y2="345" stroke="none"/>
  <path d="M 626 345 C 600 345 596 345 572 345" fill="none" stroke="currentColor" stroke-width="1.3" marker-end="url(#arr5)"/><text class="s" x="470" y="372">ν → ∞ (§3.10.3, Ch.4 concentration)</text>
  <text x="30" y="398" class="cap">Figure 3.6 — The zoo's licensed relationship map. Blue: transformation/convolution edges proved in this chapter. Green: limits and</text>
  <text x="30" y="414" class="cap">order-statistics edges (proved here or in a solved exercise). Amber: mixture edges. Unlabeled-source claims do not exist: every edge cites its proof or its table entry.</text>
</svg>

> **Self-check 3.10** *(covers §3.10)*. (a) Your count data has sample mean 4.1 and sample variance 11.7. Which two zoo members are candidates, and which does the evidence favor? (b) Why does the Dirichlet have no density with respect to $\lambda_K$ on $\mathbb{R}^K$, and with respect to what does it have one? (c) Which one-line calculation identifies $\ell_1$-regularized regression as MAP inference, and under what prior? (d) Name the construction that manufactures heavy tails from Gaussians, and the practical situation it models. *(Answers in Appendix A.)*

## 3.11 The Gumbel-max trick: sampling discrete variables differentiably

Chapter 2 compared two ways to differentiate through randomness: **REINFORCE** (score-function gradients — unbiased, high variance, needs only the ability to sample) and the **reparameterization trick** (pathwise gradients — typically far lower variance, but needs the sample to be a differentiable function of the parameters, as in $x = \mu + \sigma z$). For *discrete* variables, that comparison ended on a cliffhanger: how do you reparameterize a Categorical? A sample from $\mathrm{Cat}(\pi)$ is an index — what would "differentiate the sample with respect to the logits" even mean? This section supplies the missing construction, in two moves: an exact identity (Gumbel-max), then a differentiable relaxation of it (Gumbel-Softmax).

### 3.11.1 The exact identity

**Theorem 3.28 (Gumbel-max trick).** Let $\pi \in \Delta^{K-1}$ with all $\pi_i > 0$, and let $G_1, \dots, G_K$ be i.i.d. standard Gumbel (CDF $e^{-e^{-g}}$, density $e^{-g}e^{-e^{-g}}$; both from §3.4/§3.10). Then

$$\arg\max_{i \in \{1..K\}}\big(\log \pi_i + G_i\big) \;\sim\; \mathrm{Categorical}(\pi).$$

*Proof.* Let $A_i := \log\pi_i + G_i$. By Worked Example 3.1 (a location shift), $A_i$ has CDF $F_i(a) = \exp(-e^{-(a - \log\pi_i)}) = \exp(-\pi_i e^{-a})$ and density $p_i(a) = \pi_i e^{-a}\exp(-\pi_i e^{-a})$. The $A_i$ are independent (functions of independent variables, Definition 1.23), and ties have probability zero (continuous laws), so

$$P\big(\text{index } i \text{ wins}\big) = P\big(A_j \le A_i\ \forall j \ne i\big) = \int_{-\infty}^{\infty} p_i(a)\prod_{j \ne i}F_j(a)\,da,$$

conditioning on the value of $A_i$ — rigorously, this is Fubini–Tonelli (Ch. 2) applied to $P(A_i \in da,\, A_j \le a\ \forall j\ne i)$ over the product law. Multiply the CDFs: $\prod_{j\ne i}\exp(-\pi_j e^{-a}) = \exp\big(-(1 - \pi_i)e^{-a}\big)$, using $\sum_j \pi_j = 1$. So the integrand is

$$\pi_i\,e^{-a}\exp(-\pi_i e^{-a})\cdot \exp(-(1-\pi_i)e^{-a}) = \pi_i\,e^{-a}\,e^{-e^{-a}}.$$

Substitute $u = e^{-a}$, $du = -e^{-a}da$ (Theorem 3.1's territory): the integral becomes $\pi_i \int_0^\infty e^{-u}\,du = \pi_i$. $\blacksquare$

Look at what canceled: the winner-dependent factor collapsed to the *same* standard-Gumbel-shaped integrand for every $i$, scaled by $\pi_i$. That is not luck; it is the Gumbel's characteristic algebra, worth stating on its own:

**Proposition 3.29 (max-stability, and where logsumexp comes from).** For independent $G_i \sim$ standard Gumbel and any $\eta \in \mathbb{R}^K$:

$$\max_i\,(\eta_i + G_i) \;\sim\; \mathrm{Gumbel}\Big(\mathrm{location} = \mathrm{logsumexp}(\eta) = \log\sum_i e^{\eta_i}\Big),$$

and consequently $\mathbb{E}\big[\max_i(\eta_i + G_i)\big] = \mathrm{logsumexp}(\eta) + \gamma$, where $\gamma$ is the standard Gumbel's mean (Euler–Mascheroni, $\approx 0.5772$).

*Proof.* $P(\max_i(\eta_i + G_i) \le a) = \prod_i \exp(-e^{\eta_i}e^{-a}) = \exp\big(-e^{-a}\sum_i e^{\eta_i}\big) = \exp\big(-e^{-(a - \mathrm{logsumexp}(\eta))}\big)$, a Gumbel CDF shifted by $\mathrm{logsumexp}(\eta)$; the expectation is the shift plus the standard mean (Worked Example 3.1 again). $\blacksquare$

The family is closed under shifted maxima — **max-stable** — exactly as the Gaussian is closed under sums; this is the structural reason the Gumbel, and no other bump, makes Theorem 3.28's integral collapse (and it echoes §3.4: the Gumbel is the *limit* of maxima because it is the *fixed point* of maxima). As a corollary, deep learning's two smoothed maxima are revealed as the same object seen twice: $\mathrm{logsumexp}(\eta)$ is the *expected max under Gumbel perturbation* (up to $+\gamma$), and its gradient $\nabla\,\mathrm{logsumexp} = \mathrm{softmax}(\eta)$ (Worked Example 3.17) is the vector of *win probabilities* from Theorem 3.28. Softmax does not merely resemble a soft argmax; under Gumbel noise it is exactly the distribution of the argmax.

*Verification (Lab B.3a, seed 0, $n = 10^6$, $\pi = (0.5, 0.3, 0.15, 0.05)$):* empirical win frequencies $(0.4999, 0.3003, 0.1500, 0.0498)$ — the identity holds to four decimals, as an exact theorem should.

### 3.11.2 The relaxation: Gumbel-Softmax / Concrete

Theorem 3.28 rewrites discrete sampling as *deterministic function of logits* $\circ$ *parameter-free noise*: with $\eta = \log \pi$,

$$x = \arg\max_i\,(\eta_i + G_i), \qquad G_i \sim \text{i.i.d. Gumbel}.$$

That is precisely reparameterization's shape — noise in, differentiable map out — except the map is $\arg\max$, whose derivative is zero almost everywhere (it is piecewise constant: tiny logit changes almost never flip the winner). The gradient exists and is useless. The **Gumbel-Softmax** (a.k.a. **Concrete**) relaxation replaces the hard argmax with its temperature-$\tau$ softening:

$$y_\tau \;=\; \mathrm{softmax}\!\Big(\frac{\eta + G}{\tau}\Big) \;\in\; \Delta^{K-1}, \qquad \tau > 0.$$

As $\tau \to 0$, $y_\tau \to$ the one-hot vector of the true Gumbel-max sample (almost surely — the max coordinate's softmax weight goes to 1), so the relaxation *contains* the exact trick as its zero-temperature limit; for $\tau > 0$, $y_\tau$ is a smooth function of $\eta$, and gradients flow. The price is the fundamental trade of the section:

- **Bias:** for $\tau > 0$, $y_\tau$ is not one-hot and $\mathbb{E}[f(y_\tau)]$ is not $\mathbb{E}[f(\text{one-hot sample})]$; gradients of the relaxed objective are *biased* for the true discrete objective, with bias growing in $\tau$.
- **Variance:** as $\tau \to 0$, the softmax sharpens and its Jacobian ($\propto 1/\tau$) blows up along category boundaries; pathwise gradients become *high variance*. (In deep learning terms: at low temperature, the relaxation is a nearly-discontinuous function, and pathwise gradients through near-discontinuities are noisy.)

*Verification (Lab B.3b, seed 7, $N = 2\times10^5$ gradient samples).* Toy objective $f(y) = y^\top v$, $v = (1.0, 0.5, 0.1)$, logits $\theta = (0.2, -0.1, 0.05)$, so $\pi = (0.3844, 0.2848, 0.3308)$ and the true gradient of $\mathbb{E}[f(\text{one-hot})] = \pi^\top v$ is computable exactly: $(+0.16919, -0.01704, -0.15214)$.

| Estimator | Bias (norm) | Total variance |
|---|---|---|
| REINFORCE | $0$ (unbiased; empirical mean matched to $4\times10^{-4}$) | $0.2236$ |
| REINFORCE + mean baseline | $0$ (unbiased) | $0.0385$ |
| Gumbel-Softmax, $\tau = 0.1$ | $0.0004$ | $0.4196$ |
| Gumbel-Softmax, $\tau = 0.5$ | $0.0296$ | $0.0399$ |
| Gumbel-Softmax, $\tau = 1.0$ | $0.0751$ | $0.0077$ |

The bias–variance dial is exactly as advertised: two orders of magnitude of variance reduction for a bias that grows with $\tau$. And one honest reading the table forces on us: **the relaxation is not uniformly better** — on this linear $f$, REINFORCE with a simple baseline beats GS at $\tau = 0.1$ on both counts. Gumbel-Softmax earns its keep when $f$ is a deep network, where pathwise gradients exploit $\nabla f$'s structure while score-function variance grows with the network's roughness, and where a modest, *annealed* $\tau$ (start warm, cool during training) buys stable optimization. It is a tool with a regime, not a free lunch.

> **Common Confusion 3.8.** *"With Gumbel-Softmax my model receives samples from the Categorical, just differentiably."*
>
> For $\tau > 0$ it receives no such thing: $y_\tau$ lives in the **interior of the simplex** — every coordinate strictly positive — so the downstream network is trained on soft mixtures that the hard, deployed system (which takes genuine one-hot decisions) will never produce. That train/deploy mismatch is a second, distinct source of bias beyond the gradient bias, and it is why the **straight-through (ST) Gumbel estimator** exists: in the forward pass, use the *hard* one-hot $\mathrm{onehot}(\arg\max_i(\eta_i + G_i))$ (so the network downstream always sees legal discrete inputs); in the backward pass, pretend the derivative is that of $y_\tau$ (implementation idiom: $y_{\text{hard}} + y_\tau - \mathrm{stopgrad}(y_\tau)$). ST is a knowingly inconsistent gradient — biased even as $\tau \to 0$ — but it removes the mismatch, and it is the practice-standard default in discrete-latent models (e.g., vector-quantized autoencoders use its close cousin). Know which bias you are trading for which.

> **Self-check 3.11** *(covers §3.11)*. (a) In Theorem 3.28's proof, which single algebraic property of the Gumbel CDF makes the product of the losers' CDFs merge with the winner's density? (b) What breaks if you run the Gumbel-max trick with Gaussian noise instead? (c) State the two *distinct* biases discussed for Gumbel-Softmax pipelines, and which one straight-through addresses. (d) Why is $\mathbb{E}[\max_i(\eta_i + G_i)] - \gamma$ a smooth function of $\eta$ even though $\max$ is not smooth? *(Answers in Appendix A.)*

## 3.12 Results used but not proved

Complete list of borrowings. Everything else in the chapter was proved in the text or in a solved exercise.

| Result | Used in | Where to find a proof |
|---|---|---|
| General change of variables for $C^1$ diffeomorphisms (Theorem 3.8) | §3.5.2, Box–Muller (WE 3.12), all of §3.7; also the a.e. version of Theorems 3.1/3.2 for discontinuous densities | Folland, *Real Analysis*, Thm 2.47; Rudin, *Real and Complex Analysis*, Thm 7.26 |
| Halmos–Savage theorem (factorization $\iff$ conditional-law sufficiency in general dominated families) | §3.9 (honesty flag (b); Definitions 3.21/3.23 reconciliation) | Halmos & Savage (1949); Lehmann & Casella, *Theory of Point Estimation*, §1.6 & Thm 6.5; conditional-distribution formulation revisited in this book's conditioning chapter |
| Minimality of $\sum_i T(x_i)$ under a minimal representation (Proposition 3.25, second clause) | §3.9.2 | Lehmann & Casella, *TPE*, Thm 1.6.22 / §6.2 (likelihood-ratio criterion) |
| Nonexistence of a finite-dimensional sufficient statistic for the Cauchy location family | WE 3.18(c) | Via the Pitman–Koopman–Darmois theorem: Koopman (1936), Darmois (1935), Pitman (1936); modern treatment in Lehmann & Casella §1.6 notes |
| Fisher–Tippett–Gnedenko extreme value theorem (three limit families; Gumbel for light tails) | §3.4 closing paragraph; Gumbel's zoo entry; Figure 3.6 edge | de Haan & Ferreira, *Extreme Value Theory*, Ch. 1; Resnick, *Extreme Values, Regular Variation and Point Processes* |
| Sklar's theorem (joint law $=$ marginals $+$ copula) | §3.6.3 applications paragraph | Nelsen, *An Introduction to Copulas*, Thm 2.3.3 |
| Student-$t$ density from the Gaussian scale-mixture representation (the mixing integral) | §3.10.3 deep dive; Figure 3.6 edge | Casella & Berger, *Statistical Inference*, §5.3 (derivation of the $t$); or direct Gamma-integral computation in the pattern of WE 3.19 |
| Convergence of the normalized logistic-regression direction to the max-margin separator on separable data (mentioned, not used) | §3.8.3 separable-data discussion | Soudry et al. (2018), "The Implicit Bias of Gradient Descent on Separable Data" |
| Law of total expectation/variance in general form (used only in the elementary discrete-slice form in WE 3.19) | WE 3.19 moments | This book's conditioning chapter; Casella & Berger §4.4 |

## 3.13 Chapter summary

1. The **CDF method** — compute $P(g(X) \le y)$ directly — is the universally valid way to find the law of a transformed variable; density formulas are its smooth-case shortcuts (§3.1).
2. Under a strictly monotone $C^1$ map, densities transform by the **inverse-derivative factor**: $p_Y(y) = p_X(g^{-1}(y))|(g^{-1})'(y)|$ (Theorem 3.1). Forgetting the factor is the field's most common silent bug (CC 3.1).
3. Non-injective maps: **sum over branches** (Theorem 3.2); flat regions create **atoms**, and mixed laws get densities with respect to composite base measures like $\delta_0 + \lambda$ — ReLU's law, derived, is $\tfrac12\delta_0 + \varphi\mathbf{1}_{(0,\infty)}\lambda$ (WE 3.6).
4. Densities blow up (integrably) where $|g'| \to 0$: critical points of the map are singularities of the pushforward density (WE 3.3–3.5).
5. Sums of independent variables **convolve**; only one summand needs a density for the sum to have one — convolution smooths, which is why noise injection regularizes and reconnects disjoint supports (Theorem 3.4, Failure 4 payoff).
6. **Order statistics** have two-line CDFs; maxima concentrate at the support's edge at rate $O(1/n)$; the Gumbel is the limit (and fixed point) of maxima (§3.4, Prop. 3.29).
7. In $d$ dimensions, invertible linear maps scale volume by $|\det T|$ — *proved* via dyadic cubes + SVD (Lemma 3.6) — giving the affine change of variables (Theorem 3.7), the multivariate Gaussian density with its $(\det\Sigma)^{-1/2}$ explained (WE 3.11), and, by linearization, the general Jacobian formula (Theorem 3.8, stated; table).
8. The **generalized inverse** $F^{-1}(u) = \inf\{x: F(x)\ge u\}$ satisfies $F^{-1}(u)\le x \iff u \le F(x)$; hence $F^{-1}(U) \sim F$ for uniform $U$ (inverse transform sampling, any law), and Chapter 1's converse-of-Proposition-1.19 debt is paid (Corollary 3.12). $F(X)$ is uniform **iff**-style only for continuous $F$ (PIT, Theorem 3.13; discrete counterexample CC 3.4).
9. **Normalizing flows** = change of variables as a model: exact log-likelihood $\log p_Z(f(x)) + \log|\det J_f(x)|$, engineered around cheap log-dets (coupling layers' block-triangular Jacobians; autoregressive triangularity with the MAF/IAF read-write asymmetry). The design exists because $k = d$ + invertibility is the escape from Chapter 1's measure-zero trap — and its cost is zero compression (§3.7).
10. The **exponential family** $p_\eta = h\,e^{\langle\eta,T\rangle - A(\eta)}$: $A$ is convex (Hölder), $\nabla A = \mathbb{E}[T]$, $\nabla^2 A = \mathrm{Cov}(T)$ (differentiation under the integral, with interiority load-bearing), and softmax/softplus/sigmoid/logsumexp are its categorical and Bernoulli instances (§3.8).
11. **MLE = moment matching**: maximize likelihood $\iff$ set $\mathbb{E}_\eta[T] = \bar T$; exists uniquely iff (minimal representation and) $\bar T$ interior to the mean space. Boundary cases are real: all-heads Bernoulli and separable logistic regression have **no MLE**, and gradient descent diverges at rate $\log t$ — regularization is what restores existence (Theorem 3.19, Lab B.4).
12. **KL within an exponential family is the Bregman divergence of $A$** (Prop. 3.20): cross-entropy training is exponential-family MLE; equal-covariance Gaussian KL is Mahalanobis; natural gradient and mirror descent germinate here.
13. **Sufficiency**: $T$ is sufficient when the likelihood factors as $g_\theta(T(x))h(x)$ (Fisher–Neyman: proved both ways in the discrete case; factorization adopted as the dominated-case definition). Exponential families compress $n$ observations to a fixed-dimensional $\sum_i T(x_i)$; Uniform$[0,\theta]$ compresses to the max; **Cauchy does not compress at all** — heavy tails resist summarization (§3.9).
14. The zoo is organized by *use*: location–scale families standardize; NB $=$ Gamma-mixed Poisson explains overdispersion (Var $>$ mean $\iff$ random rate, verified); Laplace prior MAP $=$ lasso; Student-$t$ $=$ Gaussian with random scale $=$ built-in robustness; every relationship-map edge carries its license (§3.10).
15. **Gumbel-max**: $\arg\max_i(\log\pi_i + G_i) \sim \mathrm{Cat}(\pi)$ exactly (Theorem 3.28); max-stability makes logsumexp the expected perturbed max and softmax the argmax's distribution (Prop. 3.29). **Gumbel-Softmax** relaxes argmax at temperature $\tau$, trading bias (grows with $\tau$) for variance (shrinks with $\tau$) — verified numerically, including the honest finding that REINFORCE-with-baseline can win on easy objectives; straight-through fixes the train/deploy mismatch, not the gradient bias (§3.11).
16. Recurring meta-lesson: every formula's hypotheses did real work somewhere in this chapter — monotonicity (branches), continuity (PIT), interiority (MLE existence), minimality (identifiability), invertibility (flows). When an ML method misbehaves, the first question is which hypothesis your data just violated.

### Callbacks

This chapter discharged five standing promises. (1) The converse of **Proposition 1.19**, taken on faith in Chapter 1, is now Corollary 3.12, proved via the generalized inverse on the uniform space. (2) Chapter 1's running example $X = \omega^2$ on $[-1,1]$ is finished: its density $\tfrac{1}{2\sqrt b}$ was derived on the monotone piece (WE 3.3) and re-derived by the branch formula on the full domain (WE 3.4), with its blow-up located and explained. (3) The ReLU variable $A = \max(0, Z)$ — Chapter 1's "neither pmf nor pdf" specimen, given a base-measure home by Chapter 2 — now has its mixed law *derived* from the general branches-atoms-flats recipe, with $dP_A/d(\delta_0 + \lambda)$ written explicitly (WE 3.6). (4) Chapter 1's measure-zero-support theorem, which diagnosed GAN Failure 4, now has its constructive escape route: flows keep $k = d$ and invertibility precisely so Theorem 3.8 applies — with the cost (no compression) stated, and convolution (Theorem 3.4) shown to be the mathematics behind the instance-noise remedy for the original failure. (5) Chapter 2's REINFORCE-versus-reparameterization gap for discrete variables is closed by Theorem 3.28 + Gumbel-Softmax, with the bias–variance trade quantified rather than asserted (§3.11).

### Before you continue

Answer without looking back:

1. A colleague's flow reports better-than-baseline likelihoods but generates noise. What sign error do you suspect, and which 10-line numerical check would confirm it?
2. Derive, in two lines, the density of $Y = \sigma Z + \mu$ from that of $Z$. Which theorem did you use and what is its multivariate generalization's extra ingredient?
3. Why does $F(X) \sim \mathrm{Unif}(0,1)$ fail for discrete $X$, and what does that imply about p-values from discrete test statistics?
4. Your Bernoulli MLE pipeline crashes with infinite logits on a rare all-positive batch. State the theorem clause whose hypothesis failed and two standard fixes.
5. What is the sufficient statistic for an i.i.d. Gaussian sample, what dimension is it, and for which famous heavy-tailed family does *no* fixed-dimensional sufficient statistic exist?
6. Count data shows variance triple the mean. Name the two-stage generative story that produces exactly this and the resulting marginal distribution.
7. State the Gumbel-max identity and the two costs (one per direction of the temperature dial) of relaxing its argmax.
8. Where exactly does the $(\det\Sigma)^{-1/2}$ in the Gaussian density come from — which lemma, applied to which map?

### Looking ahead to Chapter 4

The next chapter is about **inequalities and concentration**: Jensen's inequality (convexity meets expectation — the exponential family's $A$ will reappear immediately), Markov and Chebyshev (the crude but universal tail bounds), and then the Chernoff method — the moment-generating-function bound whose optimization is, not coincidentally, a Legendre transform of the log-partition objects you met in Section 3.8. From there: sub-Gaussian random variables, Hoeffding's and Bernstein's inequalities for sums, and McDiarmid's inequality for functions of independent variables — the toolkit that turns "the empirical mean is close to the true mean" into quantitative statements with explicit failure probabilities. The destination is the mathematics of generalization: union bounds over hypothesis classes, Rademacher complexity, and VC dimension — the chain of arguments by which "my model did well on the training set" becomes evidence, with stated confidence, that it will do well on the distribution.

# Exercises

**Part A — Theory.**

**A1 (★).** Let $Z \sim \mathcal{N}(0,1)$ and $Y = |Z|$ (the *half-normal* distribution). Derive the density of $Y$ two ways: by the CDF method and by the branch formula (Theorem 3.2). Then compute $\mathbb{E}[Y]$ and reconcile it with the value of $\mathbb{E}[\max(0, Z)]$ from Worked Example 3.6.

**A2 (★).** Let $U \sim \mathrm{Unif}(-\tfrac\pi2, \tfrac\pi2)$ and $Y = \tan(U)$. Show $Y \sim \mathrm{Cauchy}(0)$, i.e. $p_Y(y) = \tfrac{1}{\pi(1+y^2)}$. Explain in one sentence why this gives an exact Cauchy sampler via §3.6.

**A3 (★★).** Let $X_1, \dots, X_n$ be independent with $X_i \sim \mathrm{Exp}(\beta_i)$. Show $\min_i X_i \sim \mathrm{Exp}(\sum_i \beta_i)$, and show that $P(\text{the minimum is } X_j) = \beta_j / \sum_i \beta_i$. (This "first arrival wins, proportional to rate" identity is the continuous-time cousin of the Gumbel-max trick.)

**A4 (★★).** (Licenses the Binomial $\to$ Poisson edge of Figure 3.6.) Fix $\lambda > 0$ and let $X_n \sim \mathrm{Bin}(n, \lambda/n)$. Show that for every fixed $k$, $P(X_n = k) \to e^{-\lambda}\lambda^k/k!$ as $n \to \infty$. State in one sentence the modeling moral ("law of rare events").

**A5 (★★).** (Licenses the Uniform $\to$ Beta edge.) Let $U_1, \dots, U_n$ be i.i.d. $\mathrm{Unif}(0,1)$. Using Proposition 3.5, show $U_{(k)} \sim \mathrm{Beta}(k, n - k + 1)$, and verify the density integrates to 1 by identifying the normalizing constant with the Beta function value $B(k, n-k+1) = \tfrac{(k-1)!(n-k)!}{n!}$.

**A6 (★★).** Show that the family $\{\mathrm{Unif}[0,\theta] : \theta > 0\}$ is **not** an exponential family (Definition 3.15) with respect to any choice of $h$, $T$, $\mu$: use the fact that all members of an exponential family are mutually absolutely continuous (they share the support $\{h > 0\}$ up to $\mu$-null sets), and exhibit two Uniform members violating that.

**A7 (★★★).** Prove the injectivity claim used in Theorem 3.19: if the representation is minimal, then $\nabla A$ is injective on $\mathrm{int}\,\Xi$. (Route: for $\eta_1 \ne \eta_2$, consider $\phi(t) = A(\eta_1 + t(\eta_2 - \eta_1))$ on $[0,1]$; compute $\phi''(t)$ via Theorem 3.17 and show it is strictly positive under minimality; conclude $\langle \nabla A(\eta_2) - \nabla A(\eta_1), \eta_2 - \eta_1\rangle > 0$.)

**Part B — Computational labs.** (All numbers quoted in the chapter came from these; seeds and sizes as stated. Reproduce and extend.)

**B1.** Inverse transform sampling. (a) Sample $10^6$ points as $-\log(U)/2$ and verify mean, variance, and a KS test against $\mathrm{Exp}(2)$. (b) Sample a discrete law $\pi = (0.2, 0.5, 0.3)$ via the generalized inverse (`searchsorted` on the CDF) and report frequencies to 4 decimals. Expected results: §3.6.2's Worked Example 3.13.

**B2.** RealNVP on two moons. Implement two affine coupling layers (hidden width 64), train by Adam on 2000 points, report train and held-out NLL in nats/point against the standard-normal baseline, and verify $\log|\det J_f|$ at 3 points against a central finite difference with $h = 10^{-5}$. Expected results: §3.7.3's lab box. Then, deliberately flip the log-det sign and document what the loss curve and samples do (CC 3.5's autopsy).

**B3.** Gumbel experiments. (a) Verify Theorem 3.28 over $10^6$ draws with $\pi = (0.5, 0.3, 0.15, 0.05)$. (b) Reproduce the bias–variance table of §3.11.2 for REINFORCE (with and without baseline) vs. Gumbel-Softmax at $\tau \in \{0.1, 0.5, 1.0\}$ on $f(y) = y^\top v$; the true gradient is computable in closed form — derive it first.

**B4.** Exponential-family fitting. (a) Fit $\mathrm{Gamma}(\alpha, \beta)$ to 5000 draws by solving the two moment equations ($\mathbb{E}X = \bar x$, $\mathbb{E}\log X = \overline{\log x}$) and separately by direct likelihood maximization; confirm agreement to solver tolerance. (b) Run full-batch gradient descent on logistic regression over two separated Gaussian blobs; log $\|w\|$ at iterations $10^2, 10^3, 10^4, 10^5$ and confirm unbounded ($\sim \log t$) growth with NLL $\to 0$. Then add weight decay $10^{-2}$ and confirm convergence.

**B5.** Convolution. Sum $10^6$ pairs of $\mathrm{Unif}[0,1]$ draws, histogram into 20 bins, and compare to exact triangular bin probabilities; report the maximum absolute deviation. Expected: $\le 10^{-3}$ (WE 3.7). Extend: convolve three uniforms and overlay a Gaussian with matching mean/variance — the CLT preview.

**Part C — Paper-style derivations.**

**C1.** Derive $\nabla A = \mathbb{E}_\eta[T]$ and $\nabla^2 A = \mathrm{Cov}_\eta(T)$, stating precisely the domination used and where interiority of $\eta$ enters; then derive MLE $=$ moment matching including the existence/uniqueness conditions, and analyze the all-heads Bernoulli and separable logistic regression failures.

**C2.** Prove the Fisher–Neyman factorization theorem in the discrete case, both directions, and explain (one paragraph) why the dominated case adopts factorization as a definition in this book, what the Halmos–Savage theorem adds, and which later-chapter machinery the conditional-law formulation awaits.

**C3.** Derive the Gumbel-max identity and the Gumbel-Softmax estimator; prove max-stability (Proposition 3.29); and discuss the $\tau \to 0$ bias–variance trade, including why the pathwise Jacobian scales like $1/\tau$.

**C4.** Derive the Negative Binomial as a Gamma-mixed Poisson; compute its mean and variance from the two-stage structure; and exhibit overdispersion $\mathrm{Var} > \mathbb{E}$ explicitly, identifying the excess with $\mathrm{Var}(\Lambda)$.

**C5.** Derive $D_{\mathrm{KL}}(p_{\eta_1}\|p_{\eta_2}) = B_A(\eta_2, \eta_1)$ and specialize: (a) Gaussians with common covariance $\Sigma$ (obtain $\tfrac12(\mu_1 - \mu_2)^\top\Sigma^{-1}(\mu_1-\mu_2)$); (b) Categoricals (obtain cross-entropy minus entropy, and conclude that minimizing cross-entropy in a softmax classifier is exponential-family MLE).

# Appendix A — Answers to self-checks

**Self-check 3.1.** (a) The CDF method needs only measurability of $g$; the density formula needs $g$ strictly monotone and $C^1$ *and* needs $X$ to have a density. Two situations only the CDF method handles: non-injective maps (like $\omega^2$ on $[-1,1]$, before Theorem 3.2), and input variables with atoms (like the ReLU output — no Lebesgue density exists to plug into the formula). (b) $g^{-1}(y) = \log y$, so $p_Y(y) = p_X(\log y)\cdot\tfrac1y$ for $y > 0$; the factor $\tfrac1y = \tfrac{d}{dy}\log y$ is the inverse-derivative rescaling. (c) A candidate density integrating to $0.5$ with locally correct algebra almost always means a branch of a non-injective map (or a piece of the domain) was silently dropped — half the probability went with it. The integrate-to-one test is the unconditional sanity check; run it every time.

**Self-check 3.2.** (a) $|X| \le y \iff -y \le X \le y$, so by the branch formula with branches $\pm y$ (both with $|g'| = 1$): $p_{|X|}(y) = p_X(y) + p_X(-y)$ for $y > 0$. (b) The two contributions in WE 3.4 were equal because the input density was symmetric about the fold point $0$ and $|g'(\pm\sqrt b)|$ coincide; with an asymmetric input density (e.g. $\omega$ non-uniform on $[-1,1]$), the branches contribute unequally and neither may be dropped. (c) The hard sigmoid $\min(1,\max(0,Z))$ has *two* flat regions, hence two atoms: use $\mu = \delta_0 + \delta_1 + \lambda$. Density w.r.t. $\mu$: value $\Phi(0) = \tfrac12$ at $0$; value $1 - \Phi(1) \approx 0.1587$ at $1$; $\varphi(a)$ for $a \in (0,1)$; zero elsewhere.

**Self-check 3.4.** (a) Only *one* summand needs a density (Theorem 3.4). Consequence: adding independent Gaussian (or uniform) noise to a quantized, discrete-valued signal produces a variable with a genuine Lebesgue density — this is exactly the "dequantization" step flow models apply to 8-bit image data, and it is Theorem 3.4, not a hack. (b) With unequal rates the integrand $\beta_1e^{-\beta_1x}\beta_2e^{-\beta_2(s-x)}$ keeps an $x$-dependent exponential $e^{-(\beta_1 - \beta_2)x}$ inside the integral; the result is a weighted *difference* of two exponentials, not $s^{\text{power}}e^{-\beta s}$ — the Gamma pattern needs the rates to merge, which requires $\beta_1 = \beta_2$. (c) $P(\min > x) = \prod_{i}e^{-\beta x} = e^{-n\beta x}$, so $\min \sim \mathrm{Exp}(n\beta)$ by Proposition 3.5's min formula. (d) $F(x)^n \to \mathbf{1}\{F(x) = 1\}$: without recentering/rescaling the max's distribution collapses to a point at the top of the support (a degenerate limit), whereas a sample mean's fluctuations already shrink on their own; extremes need an $(a_n, b_n)$ zoom to reveal a nondegenerate limit shape.

**Self-check 3.5.** (a) Theorem 1.20 was used to conclude $\mu = c\lambda_d$ on *all* Borel sets from agreement on dyadic boxes: dyadic boxes are closed under intersection (a $\pi$-system), they generate $\mathcal{B}(\mathbb{R}^d)$, and both measures are $\sigma$-finite along an exhausting sequence of such boxes — precisely Theorem 1.20's hypotheses. (b) $|\det J| = 2 \cdot \tfrac12 = 1$: the layer is volume-preserving, and the log-density of a transported point is unchanged (log-det term $= 0$). (c) For orthogonal $T = O$: $p_Y(y) = p_X(O^{-1}y)/|\det O| = p_X(O^\top y)$. For $X \sim \mathcal{N}(0, I)$, $p_X(O^\top y) \propto e^{-\|O^\top y\|^2/2} = e^{-\|y\|^2/2}$: the standard Gaussian is rotation-invariant, which is why it is the natural "directionless" base distribution for latents.

**Self-check 3.6.** (a) The ReLU CDF is $F(a) = 0$ for $a<0$, jumps to $\tfrac12$ at $a = 0$, then follows $\Phi(a)$. $F^{-1}(0.5) = \inf\{a : F(a) \ge 0.5\} = 0$: the value $u = 0.5$ lands exactly on the jump, and the generalized inverse's jump-handling assigns it to the atom's location. (b) The asymmetry: $F^{-1}(U)$ only needs the switching relation, which holds for arbitrary CDFs — the generalized inverse *manufactures* atoms and skips gaps correctly by construction. But $F(X)$ can only ever take values in the range of $F$; if that range misses chunks of $(0,1)$ (jumps), uniformity is impossible, so the PIT needs continuity to guarantee the range fills $(0,1)$. (c) Not flat: the p-value of a discrete statistic is stochastically *larger* than uniform under the null (its CDF sits at or below the diagonal), producing spikes at the achievable values and making the test conservative — true type-I error below nominal.

**Self-check 3.7.** (a) $s_\theta, t_\theta$ are only ever *evaluated*, never inverted: inversion of the layer solves for $x_B$ algebraically because $x_A = z_A$ passes through and re-supplies the same $s, t$ values; and their derivatives sit in the Jacobian's lower-left block, which block-triangularity excludes from the determinant. (b) A dense layer costs $O(d^3)$ per log-det evaluation ($10^6$ flops at $d = 100$, at every training step, per layer) plus the same again for inversion; the fixes are coupling layers (block-triangular Jacobian, $O(d)$ log-det, closed-form inverse) and autoregressive layers (triangular Jacobian, $O(d)$ log-det) — with Glow's $1\times1$ convolutions as the compromise that keeps small dense determinants only over channels. (c) MAF: density evaluation of external points is its parallel direction; IAF would pay a $d$-pass sequential loop per scored point.

**Self-check 3.8.** (a) $\mathcal{M} = A'(\mathbb{R}) = e^{\mathbb{R}} = (0,\infty)$. An all-zeros count dataset has $\bar T = 0 \notin \mathcal{M}$: the likelihood in $\eta = \log\lambda$ increases as $\eta \to -\infty$, so no MLE exists in the natural parameterization ($\hat\lambda = 0$ is a boundary point of the mean parameterization, not a member of the family). (b) Theorem 3.19(1): the fitted model reproduces $\bar T$ and *nothing else* — the likelihood sees the data only through $\bar T$, so $T$ already carries everything the fit will ever use; Section 3.9 upgrades this to the formal sufficiency statement. (c) Proposition 3.16 ($A$ convex) plus the fact that convexity is preserved under composition with affine maps and addition of linear functions: NLL$(\eta) = A(\eta) - \eta_{y}$ is convex in the logits, and with $\eta = Wx$ convex in $W$. (d) Yes: $p \propto e^{\eta/\tau\cdot T}$ is the member at natural parameter $\eta/\tau$ — temperature slides along a ray in $\Xi$, so it changes *which member* you use, never the family; that is why temperature scaling is a legitimate one-parameter post-hoc calibration knob that preserves the argmax.

**Self-check 3.9.** (a) $p_\theta(x_{1:n}) = \prod_i \theta^{x_i}(1-\theta)^{1 - x_i} = \theta^{\sum x_i}(1-\theta)^{n - \sum x_i} = g_\theta(\textstyle\sum_i x_i)\cdot 1$: the factorization is already complete with $h \equiv 1$. (b) The factorization requires $h$ to be $\theta$-free and $g_\theta$ to touch the data only through $T$; the indicator $\mathbf{1}\{x_{(n)} \le \theta\}$ involves both $\theta$ and the data, so the only legal home for it is $g_\theta(T)$ — which forces $T$ to include $x_{(n)}$. That is how the factorization *identifies* the sufficient statistic. (c) True: the same functions $g_\theta, h$ restricted to the subfamily's $\theta$'s still factor every density. (Minimality, by contrast, is *not* inherited.) (d) For a single-member family, every statistic is sufficient (there is no $\theta$ for anything to depend on), so a constant statistic — remembering nothing — is minimal: with no parameter uncertainty, the data carries no information to preserve.

**Self-check 3.10.** (a) Poisson and Negative Binomial; variance $\approx 2.9\times$ the mean is strong overdispersion, favoring the NB (a Poisson fit would be rejected by exactly this diagnostic). (b) The simplex is a $(K-1)$-dimensional subset of $\mathbb{R}^K$, hence $\lambda_K$-null — Chapter 1's measure-zero support theorem again; the Dirichlet's density is with respect to $(K-1)$-dimensional Lebesgue measure on the simplex (e.g., in the coordinates $(x_1, \dots, x_{K-1})$). (c) Write the negative log-posterior of a Gaussian likelihood with i.i.d. Laplace$(0,b)$ priors on the weights: squared loss plus $\tfrac1b\|w\|_1$ — lasso. (d) The scale mixture: a Gaussian whose variance (or precision) is itself random, yielding the Student-$t$; it models pooled data whose subgroups have heterogeneous variances — Gaussian pieces, heavy-tailed whole.

**Self-check 3.11.** (a) The exponential-of-exponential form: shifted Gumbel CDFs are $\exp(-c\,e^{-a})$, and *products* of such functions are again of the same form with the $c$'s summed — so the $K-1$ losers' CDFs and the winner's density merge into one standard-Gumbel integrand. (b) With Gaussian noise the product of losers' CDFs, $\prod_{j\ne i}\Phi(a - \eta_j)$, is not a shifted Gaussian CDF (the Gaussian family is sum-stable, not max-stable), the integral does not collapse, and the resulting argmax distribution is *not* $\mathrm{softmax}(\eta)$ — the trick's exactness is a Gumbel-specific theorem. (c) Bias 1: the pathwise gradient of the relaxed objective is biased for the gradient of the true discrete objective (grows with $\tau$). Bias 2: the downstream network trains on interior-of-simplex inputs it will never see at deployment. Straight-through addresses only the second, by feeding hard one-hots forward, at the price of a knowingly inconsistent backward pass. (d) Because expectation integrates the kinks away: $\mathbb{E}[\max_i(\eta_i + G_i)] = \mathrm{logsumexp}(\eta) + \gamma$ by Proposition 3.29, and logsumexp is $C^\infty$ (it is the categorical family's log-partition function $A$, smooth on the interior by Theorem 3.17). Averaging over noise is a smoother — the same reason expected losses of discrete decisions can have useful gradients even when each realized loss is piecewise constant.

# Solutions to exercises

## Part A

**A1.** *CDF method.* For $y > 0$: $F_Y(y) = P(|Z| \le y) = P(-y \le Z \le y) = \Phi(y) - \Phi(-y) = 2\Phi(y) - 1$; differentiate: $p_Y(y) = 2\varphi(y)$, $y > 0$. *Branch formula.* $g(z) = |z|$ is monotone on $(-\infty,0)$ and $(0,\infty)$ with $|g'| = 1$; the preimages of $y > 0$ are $\pm y$, so $p_Y(y) = \varphi(y) + \varphi(-y) = 2\varphi(y)$. ✓ The two routes agree — the general lesson being that when they don't, the branch bookkeeping is the first suspect. *Mean:*

$$\mathbb{E}|Z| = \int_0^\infty y\cdot 2\varphi(y)\,dy = \frac{2}{\sqrt{2\pi}}\int_0^\infty y e^{-y^2/2}dy = \frac{2}{\sqrt{2\pi}}\Big[-e^{-y^2/2}\Big]_0^\infty = \sqrt{\frac{2}{\pi}} \approx 0.7979.$$

*Reconciliation.* The identity $\max(0, z) = \tfrac{z + |z|}{2}$ holds for every real $z$; taking expectations (linearity, Ch. 2) with $\mathbb{E}[Z] = 0$ gives $\mathbb{E}[\max(0,Z)] = \tfrac12\mathbb{E}|Z| = \tfrac{1}{\sqrt{2\pi}} \approx 0.3989$ — exactly Worked Example 3.6's value, computed there from the mixed density. Two different decompositions of the same expectation; agreement is a consistency check on both. The thinking to retain: before integrating anything, look for an algebraic identity relating your variable to one you already understand.

**A2.** $\tan$ is strictly increasing and $C^1$ on $(-\tfrac\pi2, \tfrac\pi2)$, mapping onto $\mathbb{R}$, so Theorem 3.1 applies with $g^{-1} = \arctan$ and $(g^{-1})'(y) = \tfrac{1}{1+y^2}$. With $p_U \equiv \tfrac1\pi$ on the domain:

$$p_Y(y) = \frac{1}{\pi}\cdot\frac{1}{1 + y^2}, \qquad y \in \mathbb{R}. \qquad \blacksquare$$

Sampler: $U = \pi(V - \tfrac12)$ for $V \sim \mathrm{Unif}(0,1)$ is uniform on the right interval, so $Y = \tan(\pi(V - \tfrac12))$ is an exact Cauchy draw — this is inverse transform sampling (Theorem 3.11) in disguise, since $\arctan$-based formulas are exactly the Cauchy CDF inverted. Geometric bonus worth knowing: a uniformly random *direction* hitting a wall produces Cauchy-distributed positions, which is why Cauchy laws appear in resonance physics and in ratios of centered Gaussians.

**A3.** *Min.* Using independence (Definition 1.23), $P(\min_i X_i > x) = \prod_i P(X_i > x) = \prod_i e^{-\beta_i x} = e^{-(\sum_i \beta_i)x}$ for $x > 0$: the min is $\mathrm{Exp}(\sum_i\beta_i)$. (Rates add for minima — "hazards add".) *Which one wins.* Condition on the value of $X_j$ and require all others to exceed it (Fubini–Tonelli over the product law, as in Theorem 3.28's proof):

$$P(X_j = \min) = \int_0^\infty \beta_j e^{-\beta_j x}\prod_{i \ne j} e^{-\beta_i x}\,dx = \beta_j\int_0^\infty e^{-(\sum_i \beta_i)x}dx = \frac{\beta_j}{\sum_i \beta_i}. \qquad \blacksquare$$

The structural parallel to note: "winner's density times losers' survival functions, integrated" is *exactly* the computation in Theorem 3.28, with exponentials in place of Gumbels and min in place of max — one integral template, two celebrated identities. (Under $x \mapsto -\log x$ transformations the two are in fact the same identity; a pleasant extra exercise.)

**A4.** Write out the pmf and split it into pieces with known limits:

$$P(X_n = k) = \binom nk \Big(\frac\lambda n\Big)^k\Big(1 - \frac\lambda n\Big)^{n-k} = \underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to\,1}\cdot\frac{\lambda^k}{k!}\cdot\underbrace{\Big(1 - \frac\lambda n\Big)^{n}}_{\to\, e^{-\lambda}}\cdot\underbrace{\Big(1-\frac\lambda n\Big)^{-k}}_{\to\,1}.$$

The first factor is a product of $k$ terms each $\to 1$ ($k$ fixed); the third is the classical exponential limit; the fourth $\to 1$ since $k$ is fixed. Hence $P(X_n = k) \to e^{-\lambda}\lambda^k/k!$. *Moral:* many independent opportunities, each individually unlikely, with a stable expected count $np = \lambda$, produce Poisson counts — the "law of rare events," and the reason Poisson is the default for error counts, mutation counts, arrival counts.

**A5.** Proposition 3.5 with $F(u) = u$, $f(u) = 1$ on $(0,1)$:

$$f_{U_{(k)}}(u) = \frac{n!}{(k-1)!\,(n-k)!}\,u^{k-1}(1-u)^{n-k}, \qquad u \in (0,1).$$

This *is* the $\mathrm{Beta}(a,b)$ density with $a = k$, $b = n - k + 1$, provided the constants match: the Beta density is $u^{a-1}(1-u)^{b-1}/B(a,b)$, and

$$B(k, n-k+1) = \frac{\Gamma(k)\Gamma(n-k+1)}{\Gamma(n+1)} = \frac{(k-1)!\,(n-k)!}{n!},$$

whose reciprocal is exactly Proposition 3.5's constant. ✓ (And since Proposition 3.5's expression is a density of an actual random variable, it integrates to 1 automatically — which, read backwards, is a probabilistic *proof* of the Beta integral identity $\int_0^1 u^{k-1}(1-u)^{n-k}du = B(k, n-k+1)$ for integer arguments. Getting integral identities for free from "this is a density" is a trick worth stealing.) Interpretation: the $k$-th smallest of $n$ uniforms is Beta — e.g., the median of $2m+1$ uniforms is $\mathrm{Beta}(m+1, m+1)$, concentrating around $\tfrac12$ like $1/\sqrt n$.

**A6.** First the general fact. If $p_\eta = h\,e^{\langle\eta,T\rangle - A(\eta)}$, then for every $\eta$, $\{x : p_\eta(x) > 0\} = \{x : h(x) > 0\}$ up to $\mu$-null sets — the exponential factor is strictly positive everywhere. So all members share the same $\mu$-null-modulo support; in particular, for any two members $P_{\eta_1}, P_{\eta_2}$ and any event $B$: $P_{\eta_1}(B) = 0 \iff \int_B h e^{\langle\eta_1,T\rangle}d\mu = 0 \iff h\mathbf{1}_B = 0$ $\mu$-a.e. $\iff P_{\eta_2}(B) = 0$. Members are mutually absolutely continuous. Now the counterexample: take $B = (1, 2]$. Then $\mathrm{Unif}[0,2]$ gives $P(B) = \tfrac12 > 0$ while $\mathrm{Unif}[0,1]$ gives $P(B) = 0$. Two members of the would-be family disagree about a null set, contradicting mutual absolute continuity. Hence no representation $(h, T, \mu)$ can realize $\{\mathrm{Unif}[0,\theta]\}_{\theta>0}$ as an exponential family. $\blacksquare$ The moral in one sentence: **exponential families cannot move their support**, so any family whose support depends on the parameter (Uniform$[0,\theta]$, shifted exponentials, ...) is automatically outside the framework — which is also why such families produce non-standard estimation phenomena (WE 3.10's edge-concentrating max).

**A7.** Let $\eta_1 \ne \eta_2 \in \mathrm{int}\,\Xi$, set $\Delta := \eta_2 - \eta_1 \ne 0$, and define $\phi(t) := A(\eta_1 + t\Delta)$ for $t \in [0,1]$; the segment lies in $\mathrm{int}\,\Xi$ by convexity of $\Xi$ (Proposition 3.16) and openness of the interior. By Theorem 3.17 and the chain rule, $\phi$ is twice differentiable with

$$\phi''(t) = \Delta^\top \nabla^2 A(\eta_1 + t\Delta)\,\Delta = \Delta^\top \mathrm{Cov}_{\eta_t}(T)\,\Delta = \mathrm{Var}_{\eta_t}\big(\langle \Delta, T(X)\rangle\big) \ \ge\ 0.$$

Suppose $\phi''(t_0) = 0$ for some $t_0$. Then $\langle\Delta, T(X)\rangle$ is $P_{\eta_{t_0}}$-a.s. constant. But $P_{\eta_{t_0}}$ has density $h e^{\langle\eta_{t_0},T\rangle - A}$, strictly positive exactly on $\{h > 0\}$ (up to $\mu$-null sets, as in A6), so "$P_{\eta_{t_0}}$-a.s." is the same as "$\mu$-a.e. on $\{h > 0\}$" — and a nonzero $a = \Delta$ with $\langle a, T\rangle$ constant $\mu$-a.e. on $\{h>0\}$ is precisely what minimality (Definition 3.18) forbids. Hence $\phi'' > 0$ on $[0,1]$, so $\phi'$ is strictly increasing, so $\phi'(1) > \phi'(0)$; unpacking $\phi'(t) = \langle \nabla A(\eta_1 + t\Delta), \Delta\rangle$:

$$\big\langle \nabla A(\eta_2) - \nabla A(\eta_1),\ \eta_2 - \eta_1 \big\rangle > 0.$$

In particular $\nabla A(\eta_2) \ne \nabla A(\eta_1)$: the gradient map is injective. $\blacksquare$ (The displayed inequality says $\nabla A$ is *strictly monotone* as a vector field — the multivariate expression of strict convexity, and the property that makes the natural$\leftrightarrow$mean bijection of Figure 3.5 well-defined. Note where each hypothesis worked: interiority let us differentiate twice; minimality killed the degenerate direction; the shared support from A6's argument converted "a.s. under one member" into "a.s. under all.")

## Part B (lab reports: procedure, verified results, and what to notice)

**B1.** *Procedure:* generate $U \sim \mathrm{Unif}(0,1)^{\otimes 10^6}$ with a seeded generator (seed 0); transform $X = -\log(U)/2$; compare moments and run `scipy.stats.kstest` against $\mathrm{Exp}(\text{scale}=0.5)$; for the discrete part, `np.searchsorted(np.cumsum(pi), U)` implements $F^{-1}$ literally per Definition 3.9. *Verified results:* mean $0.4999$ (target $0.5$), variance $0.2502$ (target $0.25$), KS statistic $0.00096$, $p = 0.314$ — no evidence against the target law, at a sample size where even small systematic errors would be flagged; discrete frequencies $(0.2000, 0.5002, 0.2998)$ against $(0.2, 0.5, 0.3)$. *Notice:* $-\log U$ rather than $-\log(1-U)$ is legitimate because $1 - U$ is also uniform — a tiny transformation argument of exactly this chapter's kind.

**B2.** *Procedure:* two coupling layers with alternating pass-through coordinates; each conditioner a 1–64–64–2 tanh MLP emitting $(s, t)$ with $s \mapsto 2\tanh(s)$ bounding the log-scale (unbounded log-scales are the classic flow instability); exact NLL $-\tfrac1n\sum \log p_\theta(x_i)$ with $\log p_\theta(x) = \log \mathcal N(f(x); 0, I) + \log|\det J_f(x)|$; full-batch Adam at $10^{-3}$. *Verified results (data seed 0, init seeds 42/43):* standard-normal baseline $2.8379$ nats/point; NLL $2.8297 \to 2.1389 \to 1.8820 \to 1.8359 \to 1.8175$ at iterations $0/500/1000/2000/4000$; held-out $1.8086$. Finite-difference audit at three points: analytic log-dets $+0.998602, +0.485080, +0.032723$; central differences ($h = 10^{-5}$) agree to $\le 6.7\times 10^{-10}$ — the implementation computes the mathematics it claims to. *Sign-flip extension (predicted autopsy — run it and watch):* with the log-det sign flipped, Proposition 3.14 says the objective now *rewards* volume expansion at the data; theory predicts a loss that decreases without bound (the objective is no longer bounded below), predicted log-scales growing toward their $\tanh$ bound, "likelihoods" racing implausibly past the baseline, and samples that are visibly wrong — the exact syndrome of Common Confusion 3.5. The point of running it is to *see* that the training curve alone looks healthy.

**B3.** *(a)* $10^6$ draws of $\arg\max_i(\log\pi_i + G_i)$, seed 0: frequencies $(0.4999, 0.3003, 0.1500, 0.0498)$ vs. $\pi = (0.5, 0.3, 0.15, 0.05)$. *(b) The derivation asked for:* $\mathbb{E}[f(\text{one-hot})] = \sum_i \pi_i v_i$ with $\pi = \mathrm{softmax}(\theta)$, so the true gradient is $J_{\mathrm{softmax}}^\top v$, where $J_{\mathrm{softmax}} = \mathrm{diag}(\pi) - \pi\pi^\top$ (differentiate WE 3.17's mean map): true gradient $(+0.16919, -0.01704, -0.15214)$ at $\theta = (0.2, -0.1, 0.05)$. REINFORCE estimator: $v_c(e_c - \pi)$ for sampled class $c$ (score function of the categorical); baseline version subtracts $b = \pi^\top v$ from $v_c$. Gumbel-Softmax pathwise gradient of $f(y_\tau) = y_\tau^\top v$: $\tfrac{y_j}{\tau}(v_j - y^\top v)$ in coordinate $j$ (chain rule through the tempered softmax — note the advertised $1/\tau$ factor, the analytic source of the low-$\tau$ variance blowup). *Verified results (seed 7, $N = 2\times 10^5$):* the table of §3.11.2 — REINFORCE variance $0.2236$ (unbiased), with baseline $0.0385$; GS bias/variance $(0.0004, 0.4196)$, $(0.0296, 0.0399)$, $(0.0751, 0.0077)$ at $\tau = 0.1, 0.5, 1.0$. *Notice:* the honest conclusion stated in the text — for this linear $f$, a good baseline beats the relaxation; the relaxation's advantage is structural (using $\nabla f$) and appears when $f$ is a network.

**B4.** *(a) Verified (seed 0, $n = 5000$, truth $\alpha = 2.5, \beta = 1.2$):* moment matching (solving $\alpha/\beta = \bar x$, $\psi(\alpha) - \log\beta = \overline{\log x}$, with $\psi$ the digamma) gives $(2.512094, 1.206314)$; Nelder–Mead maximization of the exact log-likelihood gives $(2.512094, 1.206314)$; maximum coordinate difference $8.1\times10^{-8}$. Theorem 3.19 in silico: two very different-looking computations, one answer, because the likelihood never sees anything but $(\bar x, \overline{\log x})$. *(b) Verified (data seed 1):* $\|w\| = 2.8066, 4.3021, 6.0474, 8.0609$ at iterations $10^2, 10^3, 10^4, 10^5$ — approximately equal increments per decade of iterations, i.e. $\|w\| \sim \log t$ — with mean NLL $0.008456 \to 0.000019$. *Weight-decay extension (predicted — run to confirm):* adding $\tfrac{10^{-2}}{2}\|w\|^2$ makes the objective strongly convex and coercive, so a unique finite minimizer exists and $\|w\|$ must plateau; the cross-entropy term no longer goes to zero. The comparison of the two runs is the cleanest possible demonstration that regularization changes *existence*, not just numerics.

**B5.** *Verified (seed 0, $10^6$ sums, 20 bins):* maximum absolute deviation between empirical and exact triangular bin probabilities $0.00065$; first five bins $(0.0050, 0.0150, 0.0251, 0.0349, 0.0450)$ vs. exact $(0.0050, 0.0150, 0.0250, 0.0350, 0.0450)$. *Extension (predicted):* the three-fold convolution is the piecewise-quadratic Irwin–Hall density on $[0,3]$; overlaying $\mathcal{N}(1.5, \tfrac{3}{12})$ shows a visually close bell after only three convolutions — the CLT arriving early for bounded, symmetric summands, a preview Chapter 5 makes quantitative.

## Part C

**C1.** *Setup and moments.* With $Z(\eta) = \int h e^{\langle\eta,T\rangle}d\mu$ and $\eta \in \mathrm{int}\,\Xi$, fix $\delta > 0$ with the $2\delta$-ball around $\eta$ inside $\Xi$. The pointwise bound $|T_i| \le \tfrac1\delta(e^{\delta T_i} + e^{-\delta T_i})$ dominates the differentiated integrand uniformly over $\eta'$ in the $\delta$-ball by $\tfrac1\delta(he^{\langle\eta' + \delta e_i, T\rangle} + he^{\langle \eta' - \delta e_i, T\rangle})$, which is $\mu$-integrable because $\eta' \pm \delta e_i \in \Xi$. **Interiority enters here and only here**: the domination must hold at parameters strictly beyond the point of differentiation, and boundary points of $\Xi$ have no margin. Chapter 2's differentiation lemma (a DCT corollary) then gives $\partial_i Z = \int T_i h e^{\langle\eta,T\rangle}d\mu$, so $\partial_i A = \partial_i Z/Z = \mathbb{E}_\eta[T_i]$. Repeating with the squared bound dominates second derivatives, and the quotient rule yields $\partial_j\partial_i A = \mathbb{E}[T_iT_j] - \mathbb{E}[T_i]\mathbb{E}[T_j] = \mathrm{Cov}(T_i, T_j)$. *MLE.* $\ell(\eta) = \mathrm{const} + \langle\eta, \bar T\rangle - A(\eta)$ is concave ($A$ convex, by Hölder as in Proposition 3.16 or by $\nabla^2 A \succeq 0$); $\nabla\ell = \bar T - \nabla A(\eta)$; for concave differentiable functions, interior stationary points are exactly the global maximizers, so MLEs in $\mathrm{int}\,\Xi$ are exactly solutions of $\mathbb{E}_\eta[T] = \bar T$. If the representation is minimal and $\bar T \in \nabla A(\mathrm{int}\,\Xi)$: a solution exists by assumption and is unique because $\nabla A$ is injective (A7). *Failures.* Bernoulli all-heads: $\bar T = 1$, $\nabla A = \sigma$ has range $(0,1) \not\ni 1$; $\ell(\eta) = \eta - \mathrm{softplus}(\eta)$ is strictly increasing (derivative $1 - \sigma(\eta) > 0$), supremum $0$ at $\eta = +\infty$, never attained. Separable logistic regression: with $y_i \in \{\pm1\}$ notation, $\ell(w) = -\tfrac1n\sum\log(1 + e^{-y_iw^\top x_i})$; a separating $w_\star$ has all $y_iw_\star^\top x_i > 0$, so $\ell(cw_\star) \uparrow 0$ as $c \to \infty$ while $\ell < 0$ everywhere: supremum not attained, gradient flow diverges (empirically at rate $\|w\| \sim \log t$, Lab B.4b), and the practical remedies (weight decay, early stopping) work precisely by restoring a finite maximizer or refusing to chase the infinite one.

**C2.** *Both directions, discrete case.* ($\Leftarrow$) Given $p_\theta = g_\theta(T)h$: for $T(x) = t$ with $P_\theta(T = t) > 0$,
$P_\theta(X = x \mid T = t) = \tfrac{g_\theta(t)h(x)}{g_\theta(t)\sum_{x':T(x')=t}h(x')} = \tfrac{h(x)}{\sum_{x':T(x')=t}h(x')}$, which is $\theta$-free ($g_\theta(t) > 0$ since the conditioning event has positive probability); for $T(x) \ne t$ the conditional probability is identically 0. ($\Rightarrow$) Given sufficiency, define $q(x)$ as the common value of $P_\theta(X = x \mid T = T(x))$ and $g_\theta(t) = P_\theta(T = t)$; the chain $p_\theta(x) = P_\theta(T = T(x))\,P_\theta(X = x \mid T = T(x)) = g_\theta(T(x))\,q(x)$ is the factorization (both sides $0$ when $P_\theta(T = T(x)) = 0$). *The paragraph.* In general spaces, "the conditional distribution of $X$ given $T = t$" is not an elementary object: conditioning on probability-zero events requires regular conditional distributions, built from conditional expectation via a measure-theoretic construction this book undertakes in the conditioning chapter. Since every model in this book's scope is dominated (densities w.r.t. a common $\sigma$-finite $\mu$), we adopt the factorization — the criterion one actually *checks* — as the definition there. The Halmos–Savage theorem is what makes this honest rather than convenient: it proves that in dominated families the factorization criterion is equivalent to the conditional-law definition, so nothing is lost; the awaited machinery is precisely regular conditional probability, and the equivalence proof will be within reach once it exists.

**C3.** *Gumbel-max.* With $A_i = \log\pi_i + G_i$: shifted-Gumbel CDF $F_i(a) = e^{-\pi_ie^{-a}}$, density $\pi_ie^{-a}e^{-\pi_ie^{-a}}$ (location shift, Theorem 3.1/WE 3.1). $P(i \text{ wins}) = \int p_i(a)\prod_{j\ne i}F_j(a)\,da$; the product is $e^{-(1-\pi_i)e^{-a}}$, merging with the winner's density into $\pi_i e^{-a}e^{-e^{-a}}$, and $u = e^{-a}$ turns the integral into $\pi_i\int_0^\infty e^{-u}du = \pi_i$. *Max-stability.* $P(\max_i(\eta_i + G_i)\le a) = \prod e^{-e^{\eta_i - a}} = e^{-e^{-(a - \mathrm{logsumexp}(\eta))}}$: Gumbel with location $\mathrm{logsumexp}(\eta)$; hence $\mathbb{E}[\max] = \mathrm{logsumexp}(\eta) + \gamma$, and differentiating the expected-max identity in $\eta$ recovers softmax as the win-probability vector — the smoothed-max/soft-argmax pair are two faces of the same Gumbel calculation. *The estimator and the trade.* GS sample $y_\tau = \mathrm{softmax}((\eta + G)/\tau)$; as $\tau \to 0$, $y_\tau \to \mathrm{onehot}(\arg\max(\eta + G))$ a.s. (the gap between the top perturbed logit and the rest is a.s. positive, and the softmax of a fixed gap divided by $\tau$ collapses). Bias: for $\tau > 0$ the relaxed objective $\mathbb{E}[f(y_\tau)]$ differs from the discrete $\mathbb{E}[f(\mathrm{onehot})]$ (unless $f$ is affine *and* evaluated exactly — even for our linear $f$ the *gradient through the softmax* is biased, as the verified table shows, because $y_\tau$'s law is not the categorical's). Variance: the pathwise gradient carries the softmax Jacobian, whose entries scale as $1/\tau$ (in coordinates, $\partial y_j/\partial\eta_j = \tfrac1\tau y_j(1 - y_j)$); near category boundaries (two competing coordinates) $y_j(1-y_j)$ is order 1 while $1/\tau \to \infty$, so gradient second moments blow up — measured: variance $0.0077 \to 0.4196$ as $\tau$ drops $1.0 \to 0.1$.

**C4.** Two stages: $\Lambda \sim \mathrm{Gamma}(\alpha,\beta)$, $K\mid\Lambda \sim \mathrm{Poisson}(\Lambda)$. Marginal:

$$P(K = k) = \int_0^\infty \frac{e^{-\lambda}\lambda^k}{k!}\frac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha-1}e^{-\beta\lambda}d\lambda = \frac{\beta^\alpha}{k!\Gamma(\alpha)}\cdot\frac{\Gamma(k+\alpha)}{(1+\beta)^{k+\alpha}} = \frac{\Gamma(k+\alpha)}{k!\,\Gamma(\alpha)}\Big(\frac{\beta}{1+\beta}\Big)^{\!\alpha}\Big(\frac{1}{1+\beta}\Big)^{\!k},$$

the $\mathrm{NB}(r = \alpha,\ p = \tfrac{\beta}{1+\beta})$ pmf (Gamma-integral $\int\lambda^{s-1}e^{-c\lambda}d\lambda = \Gamma(s)c^{-s}$; Tonelli justifies all rearrangements). Moments from the structure, using the elementary slice-decomposition form of total expectation/variance: $\mathbb{E}[K] = \mathbb{E}[\mathbb{E}[K\mid\Lambda]] = \mathbb{E}[\Lambda] = \alpha/\beta$;

$$\mathrm{Var}(K) = \mathbb{E}[\mathrm{Var}(K\mid\Lambda)] + \mathrm{Var}(\mathbb{E}[K\mid\Lambda]) = \mathbb{E}[\Lambda] + \mathrm{Var}(\Lambda) = \frac{\alpha}{\beta} + \frac{\alpha}{\beta^2},$$

so $\mathrm{Var}(K) - \mathbb{E}[K] = \mathrm{Var}(\Lambda) > 0$: the overdispersion *equals* the rate's variance — an interpretable decomposition (within-unit Poisson noise plus between-unit rate spread) that also tells you what fitting an NB is implicitly estimating. Verified numerically in WE 3.19 to four decimals.

**C5.** *General identity.* Inside one family, $\log\tfrac{p_{\eta_1}}{p_{\eta_2}} = \langle \eta_1 - \eta_2, T(x)\rangle - A(\eta_1) + A(\eta_2)$ ($h$ cancels). Take $\mathbb{E}_{\eta_1}$ and apply Theorem 3.17:

$$D_{\mathrm{KL}}(p_{\eta_1}\|p_{\eta_2}) = \langle\eta_1 - \eta_2, \nabla A(\eta_1)\rangle - A(\eta_1) + A(\eta_2) = B_A(\eta_2, \eta_1).$$

*(a) Fixed-covariance Gaussians.* $\mathcal{N}(\mu, \Sigma)$ with $\Sigma$ known: expanding the quadratic form, $T(x) = x$, $\eta = \Sigma^{-1}\mu$, $h(x) = (2\pi)^{-d/2}(\det\Sigma)^{-1/2}e^{-\frac12 x^\top\Sigma^{-1}x}$, $A(\eta) = \tfrac12\mu^\top\Sigma^{-1}\mu = \tfrac12\eta^\top\Sigma\,\eta$. For quadratic $A$, the Bregman divergence is exactly the second-order term: $B_A(\eta_2,\eta_1) = \tfrac12(\eta_2 - \eta_1)^\top\Sigma(\eta_2-\eta_1) = \tfrac12(\mu_2 - \mu_1)^\top\Sigma^{-1}(\mu_2-\mu_1)$ (substituting $\eta_i = \Sigma^{-1}\mu_i$): KL between equal-covariance Gaussians is half the squared Mahalanobis distance — symmetric here, a special quadratic privilege. *(b) Categoricals.* Use WE 3.17's representation and the convenient gauge $\eta_{i,k} = \log\pi_{i,k}$ (so $A(\eta_i) = \log\sum_k \pi_{i,k} = 0$) with $\nabla A(\eta_1) = \pi_1$:

$$B_A(\eta_2, \eta_1) = 0 - 0 - \sum_k(\log\pi_{2,k} - \log\pi_{1,k})\,\pi_{1,k} = \sum_k \pi_{1,k}\log\frac{\pi_{1,k}}{\pi_{2,k}} = \underbrace{\Big(-\sum_k \pi_{1,k}\log\pi_{2,k}\Big)}_{\text{cross-entropy }H(\pi_1,\pi_2)} - \underbrace{\Big(-\sum_k\pi_{1,k}\log\pi_{1,k}\Big)}_{\text{entropy }H(\pi_1)}.$$

With $\pi_1$ the (fixed) label distribution and $\pi_2 = \mathrm{softmax}(\text{model logits})$, the entropy term is a constant of the optimization: **minimizing cross-entropy $=$ minimizing $D_{\mathrm{KL}}(\text{labels}\,\|\,\text{model})$ $=$ maximizing the categorical exponential-family likelihood** — the promised triple identity, now derived rather than asserted. $\blacksquare$

### References and further reading

**Textbooks (proofs we deferred, and deeper treatments).**
- G. Casella & R. Berger, *Statistical Inference*, 2nd ed. — transformations, order statistics, exponential families, sufficiency at exactly this chapter's level; the $t$-distribution derivation.
- P. Billingsley, *Probability and Measure*; R. Durrett, *Probability: Theory and Examples* — the measure-theoretic backbone, including product measures and the results cited from Chapters 1–2.
- G. Folland, *Real Analysis: Modern Techniques and Their Applications* (Thm 2.47); W. Rudin, *Real and Complex Analysis* (Thm 7.26) — full proofs of the general change-of-variables theorem (our Theorem 3.8).
- E. Lehmann & G. Casella, *Theory of Point Estimation*, 2nd ed. — sufficiency, minimal sufficiency, Halmos–Savage, completeness; the professional-grade sequel to Section 3.9.
- L. D. Brown, *Fundamentals of Statistical Exponential Families* — everything about $A(\eta)$, including the boundary behavior we tiptoed around.
- M. Wainwright & M. Jordan, *Graphical Models, Exponential Families, and Variational Inference* — the natural/mean duality of Figure 3.5 as the engine of variational inference.
- R. Nelsen, *An Introduction to Copulas* — Sklar's theorem. L. de Haan & A. Ferreira, *Extreme Value Theory: An Introduction* — Fisher–Tippett–Gnedenko.

**ML papers (the ideas of §§3.7 and 3.11 at their sources).**
- L. Dinh, D. Krueger, Y. Bengio, "NICE: Non-linear Independent Components Estimation" (2015); L. Dinh, J. Sohl-Dickstein, S. Bengio, "Density Estimation using Real NVP" (ICLR 2017) — additive and affine coupling layers.
- D. Kingma et al., "Improved Variational Inference with Inverse Autoregressive Flow" (NeurIPS 2016); G. Papamakarios, T. Pavlakou, I. Murray, "Masked Autoregressive Flow for Density Estimation" (NeurIPS 2017) — the IAF/MAF read–write asymmetry of §3.7.4.
- D. Kingma & P. Dhariwal, "Glow: Generative Flow with Invertible 1×1 Convolutions" (NeurIPS 2018).
- R.T.Q. Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018); W. Grathwohl et al., "FFJORD" (ICLR 2019) — the trace-based continuous limit.
- G. Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and Inference" (JMLR 2021) — the survey to read after this chapter.
- E. Jang, S. Gu, B. Poole, "Categorical Reparameterization with Gumbel-Softmax" (ICLR 2017); C. Maddison, A. Mnih, Y.W. Teh, "The Concrete Distribution" (ICLR 2017) — §3.11's relaxation, twice independently.
- Y. Bengio, N. Léonard, A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons" (2013) — the straight-through estimator.
- D. Soudry et al., "The Implicit Bias of Gradient Descent on Separable Data" (JMLR 2018) — the max-margin direction and the $\log t$ norm growth verified in Lab B.4b.
- C.K. Sønderby et al., "Amortised MAP Inference for Image Super-resolution" (ICLR 2017); M. Arjovsky & L. Bottou, "Towards Principled Methods for Training GANs" (ICLR 2017) — instance noise: Theorem 3.4 deployed against Failure 4.
