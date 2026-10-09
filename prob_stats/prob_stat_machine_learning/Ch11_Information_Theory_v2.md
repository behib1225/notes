# Chapter 11 — Information Theory

## Chapter 11 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 11.1 Entropy | $H(P)=-\sum p\log p$ is the unique (up to units) measure of uncertainty satisfying three natural axioms; it is also the shortest achievable average code length | Explains why cross-entropy is *the* classification loss; maximum entropy under moment constraints **is** the exponential family, so softmax and Gibbs distributions are not arbitrary choices |
| 11.2 Differential entropy | The continuous analogue is **not** a limit of $H$, can be negative, and shifts by $\mathbb{E}[\log\lvert\det J\rvert]$ under reparameterisation | Normalising-flow log-likelihoods, bits/dim reporting, why "entropy of a Gaussian" is meaningless without a base measure |
| 11.3 KL divergence | $D_{\mathrm{KL}}\ge 0$ with equality a.e.; cross-entropy $=$ entropy $+$ KL; MLE $=$ minimising $D_{\mathrm{KL}}(\hat P_n\Vert P_\theta)$; and near a parameter, $D_{\mathrm{KL}}(P_\theta\Vert P_{\theta+\delta}) = \tfrac12\delta^\top I(\theta)\delta + o(\lVert\delta\rVert^2)$ | Cross-entropy loss, label smoothing, distillation, the forward/reverse asymmetry that decides whether a fitted model covers mass or chases modes, and the Fisher geometry behind natural gradient and trust-region (TRPO/PPO) updates |
| 11.4 Mutual information | $I(X;Y)=D_{\mathrm{KL}}(P_{XY}\Vert P_X\otimes P_Y)$: reparameterisation-invariant, symmetric, obeys the data-processing inequality | Representation learning, InfoNCE, information bottleneck, feature selection; and the reason no layer can create information |
| 11.5 $f$-divergences | One convex generator $f$ produces KL, reverse KL, JS, $\chi^2$, TV, Hellinger; all obey DPI; all have variational (critic) forms | GAN discriminators, $f$-GAN, MINE, InfoNCE are all the same variational trick — plus the honest failure modes of these estimators |
| 11.6 TV, Pinsker, coupling | TV is the maximal probability gap, the $L^1$ distance, and the minimal disagreement over couplings; $\mathrm{TV}\le\sqrt{D_{\mathrm{KL}}/2}$ | Converts likelihood-ratio bounds into testing/error bounds; the currency in which distribution shift and privacy guarantees are stated |
| 11.7 Optimal transport | $W_p$ uses a ground metric, not density ratios; 1-D closed form; $W_1$ has a Lipschitz dual | Closes Chapter 2's Failure 4: why the GAN loss is flat on disjoint supports and why WGAN's critic is not |
| 11.8 Coding and compression | Kraft, source coding, the asymptotic equipartition property, and "redundancy of a wrong code $=$ KL"; bits/dim done properly | Lets you audit a reported likelihood, convert between $[0,1]$ and $[0,255]$ scales, and read MDL arguments |
| 11.9 Fano | Small mutual information $\Rightarrow$ large error, for *every* estimator or classifier | Floors on achievable accuracy from a representation; how impossibility results are proved; matching lower bounds for the upper bounds of Chapter 4 |
| 11.10 ML payoffs | Consolidated: cross-entropy, label smoothing, information bottleneck, contrastive learning, rate-distortion view of VAEs | A checklist of where the theory is used well and where it is misapplied |

## 11.0 Orientation: what information theory buys a machine learner

Almost every loss function you will meet in modern machine learning is an information-theoretic object wearing a disguise. The cross-entropy loss for classification is a KL divergence to an empirical distribution. The ELBO is a rate-distortion trade-off. The GAN objective is a variational estimate of a Jensen–Shannon divergence. The contrastive loss used to pretrain vision and language encoders is a lower bound on mutual information. Distillation, label smoothing, and $\beta$-VAE annealing are all statements about which divergence you are willing to pay for. This chapter gives you the objects themselves, so that when a paper writes "we minimise $\mathcal{L} = -\mathbb{E}_{q}[\log p(x\mid z)] + \beta\, D_{\mathrm{KL}}(q(z\mid x)\Vert p(z))$" you see the structure, not just the symbols.

You already have most of the machinery. From Chapter 2 you have the Radon–Nikodym theorem, absolute continuity, the definition of KL divergence with respect to a base measure, the Dominated Convergence Theorem, the ELBO, and the result that the Jensen–Shannon divergence between mutually singular measures is exactly $\log 2$ — the diagnosis of Failure 4. From Chapter 3 you have the change-of-variables formula with its Jacobian determinant, and the exponential family. From Chapter 4 you have Jensen's inequality and concentration upper bounds. From Chapter 6 you have conditional expectation as a projection. From Chapter 8 you have maximum likelihood and Fisher information. From Chapter 1, by number, you have Definition 1.17 (support), Definition 1.23 (independence, which is literally the statement $P_{XY}=P_X\otimes P_Y$), Definition 1.24 (conditional independence), and Proposition 1.12 (Bayes). I will not re-derive any of these; I will name them where they are used.

The punchline, stated in advance, has three parts.

**First: entropy is fragile, KL is robust.** Discrete entropy is a genuine intrinsic quantity: it does not depend on how you label the outcomes. Differential entropy is *not* intrinsic — it changes when you change coordinates, and it can be negative, and it is not the limit of discrete entropies. But KL divergence and mutual information, which look like they are built from entropies, *are* intrinsic: they are invariant under any invertible reparameterisation. So the quantities that survive as objective functions in machine learning are the divergences, not the entropies. Every time you see a paper report an "entropy in nats" for a continuous variable, you should ask which base measure, and every time you see a mutual information you should not.

**Second: all the standard divergences except the transport distances are blind in the same place.** KL, reverse KL, JS, $\chi^2$, Hellinger, and total variation are all instances of a single formula $D_f(P\Vert Q)=\int f(\mathrm{d}P/\mathrm{d}Q)\,\mathrm{d}Q$ with different convex $f$. Because they depend on $P$ and $Q$ only through the density ratio, they cannot see *how far apart* two mutually singular measures are — they saturate. Wasserstein distance is built from a ground metric instead, so it can. This is not a philosophical remark: it is the reason a GAN generator whose support is a low-dimensional manifold (Chapter 1 proved it must be, when the latent dimension is smaller than the data dimension) receives no gradient from a perfectly trained discriminator.

**Third: divergences are the currency of both upper and lower bounds.** Chapter 4 used concentration to prove statements of the form "with this many samples, the error is at most this much." Fano's inequality goes the other way: if the mutual information between the data and the parameter is small, then *no* algorithm can succeed. Upper and lower bounds together pin down a rate. Learning to read a Fano argument is learning to tell "we haven't found the right method yet" apart from "no method exists."

A word about honesty. Information theory has a bad habit in applied machine learning: because its quantities have compelling names, they get used as explanations when they are only descriptions. "The network learns to maximise mutual information with the label" is often unfalsifiable in practice, because mutual information in high dimensions is essentially unestimable from realistic sample sizes — a fact we will demonstrate numerically, not merely assert. I will flag these places as they arise rather than at the end. The theory is beautiful and load-bearing; its estimators are not.

Units: I will use **nats** (natural logarithm) by default because derivatives are cleaner, and **bits** ($\log_2$) whenever the answer is more interpretable that way. I will always say which. Conversion is one multiplication: $1$ nat $=1/\log 2 \approx 1.442695$ bits.

**Notation used throughout.** Everything below is standard in this book; the table is here so you can read any section without hunting back through earlier chapters.

| Symbol | Meaning |
|---|---|
| $P_X$, $P_{XY}$, $P_{Y\mid X}$ | the law of $X$; the joint law; the conditional law (a Markov kernel) |
| $P_X\otimes P_Y$ | product measure; $P_{XY} = P_X\otimes P_Y$ *is* the definition of independence (Def. 1.23) |
| $\mu$, $\lambda_d$, $\delta_x$ | a generic base ($\sigma$-finite) measure; Lebesgue measure on $\mathbb{R}^d$; the point mass at $x$ |
| $\nu\ll\mu$ | $\nu$ is absolutely continuous w.r.t. $\mu$; $\mathrm{d}\nu/\mathrm{d}\mu$ is the Radon–Nikodym derivative |
| $P\perp Q$ | $P$ and $Q$ are mutually singular (each gives full mass to a set the other calls null) |
| $H(X)$, $H(X\mid Y)$, $H_\mu(X)$ | Shannon entropy; conditional entropy; entropy relative to a base measure $\mu$ (Def. 11.10) |
| $h(X)$ | differential entropy, i.e. $H_{\lambda_d}(X)$ — a *different animal* from $H$ (see §11.2) |
| $D_{\mathrm{KL}}(P\Vert Q)$, $D_f(P\Vert Q)$ | Kullback–Leibler divergence; the $f$-divergence with generator $f$ |
| $I(X;Y)$, $I(X;Y\mid Z)$ | mutual information; conditional mutual information |
| $\mathrm{TV}$, $\mathrm{JS}$, $\mathrm{Hel}$, $\chi^2$ | total variation, Jensen–Shannon, Hellinger, chi-squared divergences |
| $W_p$, $\Pi(P,Q)$ | $p$-Wasserstein distance; the set of couplings of $P$ and $Q$ |
| $\hat P_n$ | the empirical measure $\frac1n\sum_i\delta_{X_i}$ |
| $H_2(t)$ | binary entropy $-t\log t-(1-t)\log(1-t)$ |
| $f'(\infty)$ | $\lim_{t\to\infty}f(t)/t$, the slope that prices mass where the denominator vanishes |
| $PK$ | the law of the output of channel $K$ when its input has law $P$ |

**How to read this chapter.** The sections are ordered so that each one uses only what came before, and the self-checks at the end of every section are the intended checkpoint: if you cannot answer them, reread before continuing rather than pressing on. Answers are in Appendix A. Proofs are complete unless the statement says otherwise, and §11.11 lists every exception with a reference. The exercises are not decoration — Parts B and C carry material that the exposition only gestures at, and their solutions are written as worked arguments rather than answer keys.

## 11.1 Entropy

### 11.1.1 What we want a measure of uncertainty to do

Start with a finite set of outcomes and a distribution over them. Before any formula, ask what a number measuring "how uncertain am I?" should satisfy.

If one outcome has probability one, you are not uncertain at all, so the number should be zero. If all $k$ outcomes are equally likely you are maximally uncertain, and the number should increase with $k$ — more equally likely alternatives, more uncertainty. And there is a third, less obvious requirement, which is the one that actually pins the formula down. Suppose you reveal the outcome in two stages: first you say which of two groups it fell into, then you say which element of that group. The total uncertainty resolved should not depend on how you chose to stage the revelation. Uncertainty should be *additive over a hierarchical decomposition*, weighted by how often you visit each branch.

That third requirement is very strong. It forces the logarithm.

**Definition 11.1 (Shannon entropy).** Let $X$ be a random variable taking values in a countable set $\mathcal{X}$ with $P_X(\{x\}) = p(x)$, where $p$ is the density of $P_X$ with respect to counting measure $\#$ on $\mathcal{X}$. The **entropy** of $X$ (equivalently, of the law $P_X$) is
$$H(X) := -\sum_{x\in\mathcal{X}} p(x)\log p(x) = \mathbb{E}\!\left[-\log p(X)\right],$$
with the convention $0\log 0 := 0$, justified because $t\log t \to 0$ as $t\downarrow 0$. In bits, replace $\log$ by $\log_2$. If the sum diverges we write $H(X)=+\infty$.

Two remarks that matter later. First, $H(X)$ depends only on the *multiset of probabilities*, not on the values $x$ themselves: relabelling outcomes does not change entropy. This is why entropy is intrinsic in the discrete case, and it will fail spectacularly in the continuous case. Second, $H(X)$ is a functional of the law $P_X$, so writing $H(P)$ is equally correct; I use both. The quantity $-\log p(x)$ is called the **surprisal** of the outcome $x$, and entropy is its expectation.

**Proposition 11.2 (range and extremes).** Let $\mathcal{X}$ be finite with $|\mathcal{X}| = k$. Then $0 \le H(X) \le \log k$. The left equality holds iff $p$ is a point mass; the right equality holds iff $p$ is uniform.

*Proof.* Each term $-p(x)\log p(x)$ is $\ge 0$ because $p(x)\in[0,1]$, so $H\ge 0$, with equality iff every term vanishes, i.e. every $p(x)\in\{0,1\}$, i.e. $p$ is a point mass. For the upper bound, let $u(x)=1/k$ be the uniform density and compute
$$\log k - H(X) = \sum_x p(x)\log k + \sum_x p(x)\log p(x) = \sum_x p(x)\log\frac{p(x)}{u(x)} = D_{\mathrm{KL}}(P\Vert U)\ \ge 0,$$
where non-negativity of KL is Theorem 11.17 below (its proof does not use this proposition, so there is no circularity). Equality in Theorem 11.17 holds iff $P=U$. $\blacksquare$

That proof is worth pausing on, because it is the template for the whole chapter: *a statement about entropy became a statement about a KL divergence to a reference distribution.* The reference distribution here is the uniform one, i.e. counting measure normalised. Entropy is a divergence in disguise, and the disguise is exactly the choice of base measure. Hold that thought until §11.2, where the disguise slips.

### 11.1.2 The axiomatic characterisation

Now let us make good on the promise that the three requirements force the logarithm. Write $A(n) := H(\text{uniform on } n \text{ points})$.

**Lemma 11.3.** Let $A:\mathbb{N}_{\ge 1}\to[0,\infty)$ be non-decreasing with $A(1)=0$ and $A(mn)=A(m)+A(n)$ for all $m,n\ge 1$. Then $A(n) = c\log n$ for some constant $c\ge 0$.

*Proof.* If $A(2)=0$ then by additivity $A(2^k)=0$ for all $k$, and monotonicity squeezes $A(n)=0$ for all $n$ (since every $n$ lies between two powers of $2$), giving $c=0$. So assume $A(2)>0$ and set $c := A(2)/\log 2>0$.

Fix $n\ge 2$ and a large integer $r$. Choose the integer $k=k(r)$ with $2^{k} \le n^{r} < 2^{k+1}$. Taking $\log$: $k\log 2 \le r\log n < (k+1)\log 2$, hence
$$\frac{k}{r} \le \frac{\log n}{\log 2} < \frac{k+1}{r}. \tag{11.1}$$
Monotonicity and additivity give $A(2^k) \le A(n^r) < A(2^{k+1})$, i.e. $k\,A(2) \le r\,A(n) < (k+1)A(2)$, hence
$$\frac{k}{r} \le \frac{A(n)}{A(2)} < \frac{k+1}{r}. \tag{11.2}$$
Both $\log n/\log 2$ and $A(n)/A(2)$ lie in the interval $[k/r,(k+1)/r)$ of length $1/r$, so they differ by less than $1/r$. Letting $r\to\infty$ gives $A(n)/A(2) = \log n/\log 2$, i.e. $A(n) = c\log n$. $\blacksquare$

**Theorem 11.4 (Shannon's characterisation).** Let $H_k(p_1,\dots,p_k)$ be defined for every $k\ge 1$ and every probability vector, and suppose:

1. **(Continuity)** $H_2(p,1-p)$ is continuous in $p$.
2. **(Monotone uniform)** $A(k):=H_k(1/k,\dots,1/k)$ is non-decreasing in $k$.
3. **(Grouping)** For any partition of $\{1,\dots,k\}$ into blocks $B_1,\dots,B_m$ with weights $w_j := \sum_{i\in B_j} p_i > 0$,
$$H_k(p_1,\dots,p_k) = H_m(w_1,\dots,w_m) + \sum_{j=1}^m w_j\, H_{|B_j|}\!\left(\frac{p_i}{w_j}: i \in B_j\right).$$

Then $H_k(p) = -c\sum_i p_i \log p_i$ for a constant $c>0$ (or $c = 0$, the trivial case).

*Proof.* First, grouping with $m=k$ and singleton blocks gives $H_k(p) = H_k(p) + \sum_i p_i H_1(1)$, so $H_1(1)=0$, i.e. $A(1)=0$.

Next, take $k = mn$ points arranged in $m$ blocks of $n$, all outcomes equally likely. Then $w_j = 1/m$ and each conditional distribution is uniform on $n$ points, so grouping gives $A(mn) = A(m) + \sum_j (1/m) A(n) = A(m)+A(n)$. By Lemma 11.3, $A(n) = c\log n$ with $c \ge 0$; assume $c>0$.

Now let $p$ be a *rational* probability vector, $p_i = n_i/N$ with $\sum_i n_i = N$. Consider the uniform distribution on $N$ points, partitioned into $k$ blocks of sizes $n_1,\dots,n_k$. The block weights are exactly $w_i = n_i/N = p_i$, and each conditional distribution is uniform on $n_i$ points. Grouping gives
$$A(N) = H_k(p) + \sum_i p_i A(n_i) \quad\Longrightarrow\quad c\log N = H_k(p) + c\sum_i p_i \log n_i,$$
so
$$H_k(p) = c\sum_i p_i\bigl(\log N - \log n_i\bigr) = -c\sum_i p_i \log \frac{n_i}{N} = -c\sum_i p_i\log p_i .$$
Finally extend from rationals to reals. For $k=2$ this is exactly hypothesis 1, since the rationals are dense in $[0,1]$ and both sides are continuous. For general $k$, apply grouping repeatedly to split a $k$-vector into a chain of binary splits: $H_k(p_1,\dots,p_k) = H_2(p_1, 1-p_1) + (1-p_1)H_{k-1}\!\left(\frac{p_2}{1-p_1},\dots\right)$, and induct on $k$; each stage is continuous in the $p_i$ because $H_2$ is, so $H_k$ is continuous and agrees with $-c\sum p_i\log p_i$ on a dense set. $\blacksquare$

The constant $c$ is the choice of units: $c=1$ gives nats, $c = 1/\log 2$ gives bits. Nothing else is free. This is the sense in which entropy is not a modelling choice.

> **Common Confusion 11.1.** *"Entropy is the expected number of yes/no questions needed to identify the outcome. But you can't ask 1.81 questions. So isn't the formula wrong, or at least only approximate?"*
>
> The resolution is that entropy is the expected number of questions **per outcome when you get to batch outcomes together**, and batching is what makes fractional answers meaningful. For a single outcome you are stuck with integers, and the best integer strategy generally costs strictly more than $H$. Worked Example 11.1 makes this quantitative: for the three-coin-flip variable $X$, $H(X) = 1.811278$ bits but the best single-symbol code costs $1.875$ bits per symbol. Encode two independent copies at once and the cost per symbol drops to $1.828125$; three copies, $1.820313$. The gap is closing to zero at rate $O(1/n)$, exactly as Theorem 11.46 will predict. Entropy is the limit of an averaged quantity, and the limit need not be attained at $n=1$ — which is not a defect but the entire content of the source coding theorem.

<svg viewBox="0 0 680 460" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<text x="340.0" y="30" fill="currentColor" font-size="15" text-anchor="middle">Figure 11.1 — Binary entropy H(p) in bits</text>
<line x1="100" y1="340" x2="600" y2="340" stroke="currentColor" stroke-width="1.2"/>
<line x1="100" y1="340" x2="100" y2="60" stroke="currentColor" stroke-width="1.2"/>
<line x1="100.0" y1="340" x2="100.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="100.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0</text>
<line x1="225.0" y1="340" x2="225.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="225.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.25</text>
<line x1="350.0" y1="340" x2="350.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="350.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.5</text>
<line x1="475.0" y1="340" x2="475.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="475.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.75</text>
<line x1="600.0" y1="340" x2="600.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="600.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">1</text>
<line x1="94" y1="340.0" x2="100" y2="340.0" stroke="currentColor" stroke-width="1"/>
<text x="90" y="344.0" fill="currentColor" font-size="13" text-anchor="end">0</text>
<line x1="94" y1="206.7" x2="100" y2="206.7" stroke="currentColor" stroke-width="1"/>
<text x="90" y="210.7" fill="currentColor" font-size="13" text-anchor="end">0.5</text>
<line x1="94" y1="73.3" x2="100" y2="73.3" stroke="currentColor" stroke-width="1"/>
<text x="90" y="77.3" fill="currentColor" font-size="13" text-anchor="end">1</text>
<text x="350.0" y="385" fill="currentColor" font-size="14" text-anchor="middle">p = P(X = 1)</text>
<text x="58" y="200.0" fill="currentColor" font-size="14" text-anchor="middle" transform="rotate(-90 58 200.0)">H(p)  (bits)</text>
<polyline points="100.0,340.0 100.0,339.6 103.2,325.2 106.3,314.0 109.4,304.0 112.5,294.9 115.7,286.4 118.8,278.4 121.9,270.8 125.0,263.5 128.2,256.6 131.3,250.0 134.4,243.6 137.5,237.4 140.7,231.5 143.8,225.8 146.9,220.2 150.0,214.9 153.2,209.7 156.3,204.6 159.4,199.7 162.5,195.0 165.7,190.4 168.8,185.9 171.9,181.6 175.0,177.3 178.2,173.2 181.3,169.2 184.4,165.3 187.5,161.6 190.7,157.9 193.8,154.3 196.9,150.8 200.0,147.5 203.2,144.2 206.3,141.0 209.4,137.9 212.5,134.9 215.7,131.9 218.8,129.1 221.9,126.3 225.0,123.6 228.1,121.0 231.3,118.5 234.4,116.1 237.5,113.7 240.6,111.4 243.8,109.2 246.9,107.0 250.0,105.0 253.1,103.0 256.3,101.0 259.4,99.2 262.5,97.4 265.6,95.7 268.8,94.0 271.9,92.4 275.0,90.9 278.1,89.5 281.3,88.1 284.4,86.7 287.5,85.5 290.6,84.3 293.8,83.2 296.9,82.1 300.0,81.1 303.1,80.1 306.3,79.3 309.4,78.4 312.5,77.7 315.6,77.0 318.8,76.3 321.9,75.8 325.0,75.3 328.1,74.8 331.3,74.4 334.4,74.1 337.5,73.8 340.6,73.6 343.8,73.5 346.9,73.4 350.0,73.3 353.1,73.4 356.2,73.5 359.4,73.6 362.5,73.8 365.6,74.1 368.7,74.4 371.9,74.8 375.0,75.3 378.1,75.8 381.2,76.3 384.4,77.0 387.5,77.7 390.6,78.4 393.7,79.3 396.9,80.1 400.0,81.1 403.1,82.1 406.2,83.2 409.4,84.3 412.5,85.5 415.6,86.7 418.7,88.1 421.9,89.5 425.0,90.9 428.1,92.4 431.2,94.0 434.4,95.7 437.5,97.4 440.6,99.2 443.7,101.0 446.9,103.0 450.0,105.0 453.1,107.0 456.2,109.2 459.4,111.4 462.5,113.7 465.6,116.1 468.7,118.5 471.9,121.0 475.0,123.6 478.1,126.3 481.2,129.1 484.3,131.9 487.5,134.9 490.6,137.9 493.7,141.0 496.8,144.2 500.0,147.5 503.1,150.8 506.2,154.3 509.3,157.9 512.5,161.6 515.6,165.3 518.7,169.2 521.8,173.2 525.0,177.3 528.1,181.6 531.2,185.9 534.3,190.4 537.5,195.0 540.6,199.7 543.7,204.6 546.8,209.7 550.0,214.9 553.1,220.2 556.2,225.8 559.3,231.5 562.5,237.4 565.6,243.6 568.7,250.0 571.8,256.6 575.0,263.5 578.1,270.8 581.2,278.4 584.3,286.4 587.5,294.9 590.6,304.0 593.7,314.0 596.8,325.2 600.0,339.6 600.0,340.0" fill="none" stroke="#3b82f6" stroke-width="2.4"/>
<line x1="350.0" y1="340" x2="350.0" y2="73.3" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4"/>
<line x1="100" y1="73.3" x2="350.0" y2="73.3" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4"/>
<circle cx="350.0" cy="73.3" r="4" fill="#3b82f6"/>
<text x="360.0" y="63.3" fill="currentColor" font-size="13">maximum: 1 bit at p = 1/2</text>
<circle cx="155.0" cy="206.7" r="4" fill="#f59e0b"/>
<text x="163.0" y="198.7" fill="currentColor" font-size="13">p = 0.11: H = 0.500 bits</text>
<circle cx="102.5" cy="327.9" r="4" fill="#ef4444"/>
<text x="110.5" y="319.9" fill="currentColor" font-size="13">prevalence 0.5%: H = 0.0454 bits</text>
<text x="340.0" y="400" fill="currentColor" font-size="13.5" text-anchor="middle">H is concave, zero at the deterministic ends, maximal at p = 1/2, and flat near the top:</text>
<text x="340.0" y="419" fill="currentColor" font-size="13.5" text-anchor="middle">moving p from 0.5 to 0.4 costs only 0.029 bits. Near p = 0 it falls steeply, so rare</text>
<text x="340.0" y="438" fill="currentColor" font-size="13.5" text-anchor="middle">events carry little entropy even though each occurrence is very surprising.</text>
</svg>

**Worked Example 11.1 (three coin flips, and what a code costs).** Recall Chapter 1's running example: $\Omega = \{H,T\}^3$ with the uniform measure, and $X(\omega) = $ the number of heads. Then $P_X = (1/8, 3/8, 3/8, 1/8)$ on $\{0,1,2,3\}$.

Entropy in bits:
$$H(X) = -2\cdot\tfrac18\log_2\tfrac18 - 2\cdot\tfrac38\log_2\tfrac38 = \tfrac{6}{8} + \tfrac34\log_2\tfrac83 = 1.811278\ \text{bits}.$$
Meanwhile $H(\omega) = \log_2 8 = 3$ bits, because $\omega$ is uniform on $8$ outcomes. So the map $\omega\mapsto X(\omega)$ destroys $3 - 1.811278 = 1.188722$ bits: it is not injective, and Proposition 11.7 below says a function can only lose entropy. This is the entropy-flavoured version of a fact you already know from Chapter 1: $\sigma(X) \subsetneq \mathcal{F}$, so $X$ carries strictly less information than $\omega$.

Now build an actual code. Huffman's algorithm on $(1/8,3/8,3/8,1/8)$ yields codeword lengths $(3,2,1,3)$ for outcomes $(0,1,2,3)$, and expected length
$$\mathbb{E}[L] = \tfrac18\cdot 3 + \tfrac38\cdot 2 + \tfrac38\cdot 1 + \tfrac18\cdot 3 = 1.875\ \text{bits},$$
a redundancy of $0.063722$ bits over $H(X)$. Applying Huffman to blocks of two i.i.d. copies (a $16$-symbol alphabet) gives $1.828125$ bits per symbol; blocks of three, $1.820313$. All four numbers were computed by running Huffman's algorithm, not estimated.

**Worked Example 11.2 (chain rule and shared information, same example).** Keep $X = $ number of heads and let $Y = \mathbf{1}\{\text{first flip is } H\}$. The joint law, from enumerating all eight $\omega$:

| $P(X=i,Y=j)$ | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | $1/8$ | $0$ |
| $X=1$ | $2/8$ | $1/8$ |
| $X=2$ | $1/8$ | $2/8$ |
| $X=3$ | $0$ | $1/8$ |

Computed values (bits): $H(X)=1.811278$, $H(Y)=1$, $H(X,Y)=2.5$. Hence
$$H(X\mid Y) = H(X,Y)-H(Y) = 1.5,\qquad I(X;Y) = H(X)+H(Y)-H(X,Y) = 0.311278 .$$
Check the conditional entropies directly: given $Y=0$ the conditional law of $X$ is $(1/4,1/2,1/4,0)$ with entropy $1.5$ bits; given $Y=1$ it is $(0,1/4,1/2,1/4)$, also $1.5$ bits. Their average is $1.5$, matching. Notice $H(X\mid Y) = 1.5 < 1.811278 = H(X)$: knowing the first flip does reduce uncertainty about the total, by $0.311278$ bits. Notice also that this is a *strict* decrease, unlike Chapter 1's example (iv), where $P(X=1,Y=7) = P(X=1)P(Y=7)$ held exactly for one pair of values without independence. Entropy inequalities are about the whole joint law, so they cannot be fooled by a single coincidence — that is one concrete advantage of information measures over spot checks.

### 11.1.3 Joint, conditional, chain rule

**Definition 11.5 (joint and conditional entropy).** For discrete $X, Y$ with joint density $p(x,y)$ w.r.t. counting measure,
$$H(X,Y) := -\sum_{x,y}p(x,y)\log p(x,y), \qquad H(Y\mid X) := \sum_x p(x)\, H(P_{Y\mid X=x}) = -\sum_{x,y}p(x,y)\log p(y\mid x),$$
with the convention that terms with $p(x,y)=0$ are dropped. Note $H(Y\mid X)$ is a *number*, not a random variable: it is the expectation over $X$ of the entropy of the conditional law. (Contrast conditional expectation from Chapter 6, which *is* a random variable. The entropy of a conditional law, $H(P_{Y\mid X})$, is a function of $X$; $H(Y\mid X)$ is its mean.)

**Theorem 11.6 (chain rule).** $H(X,Y) = H(X) + H(Y\mid X)$, and more generally
$$H(X_1,\dots,X_n) = \sum_{i=1}^n H(X_i \mid X_1,\dots,X_{i-1}).$$

*Proof.* Factor $p(x,y) = p(x)p(y\mid x)$ and split the logarithm:
$$H(X,Y) = -\sum_{x,y}p(x,y)\bigl[\log p(x) + \log p(y\mid x)\bigr] = -\sum_x p(x)\log p(x) - \sum_{x,y}p(x,y)\log p(y\mid x),$$
which is $H(X)+H(Y\mid X)$. The general case follows by induction, applying the two-variable rule to the pair $\bigl((X_1,\dots,X_{i-1}), X_i\bigr)$. $\blacksquare$

The chain rule is the grouping axiom, now a theorem. It is also the formal content of the picture that autoregressive models are built on: a language model factorises $\log p(x_1,\dots,x_T) = \sum_t \log p(x_t\mid x_{<t})$, and its training loss is an empirical estimate of $\sum_t H(X_t \mid X_{<t})$ plus a KL penalty that we will identify precisely in §11.3. The per-token cross-entropy that everyone reports is an estimate of the *conditional* entropy rate, not of $H(X_1)$.

**Proposition 11.7 (conditioning reduces entropy; functions lose entropy).** For discrete variables:

1. $0 \le H(Y\mid X) \le H(Y)$, with the right equality iff $X \perp Y$ (Definition 1.23).
2. $H(X,Y) \le H(X)+H(Y)$ (subadditivity), with equality iff $X\perp Y$.
3. For any measurable $g$, $H(g(X)) \le H(X)$, with equality iff $g$ is injective on $\{x: p(x)>0\}$.

*Proof.* (1) Non-negativity is Proposition 11.2 applied to each conditional law. For the upper bound,
$$H(Y) - H(Y\mid X) = \sum_{x,y}p(x,y)\log\frac{p(y\mid x)}{p(y)} = \sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = D_{\mathrm{KL}}(P_{XY}\Vert P_X\otimes P_Y) \ge 0$$
by Theorem 11.17, with equality iff $P_{XY} = P_X\otimes P_Y$, which is exactly independence.
(2) Immediate from (1) and the chain rule.
(3) Let $Z=g(X)$. Since $Z$ is a function of $X$, the conditional law of $Z$ given $X=x$ is a point mass, so $H(Z\mid X)=0$; the chain rule applied twice gives $H(X) = H(X,Z) = H(Z) + H(X\mid Z) \ge H(Z)$. Equality iff $H(X\mid Z)=0$, i.e. $X$ is determined by $Z$ almost surely, i.e. $g$ is injective on the support. $\blacksquare$

Part (3) deserves a name it rarely gets: it is the entropy version of the data-processing inequality, and it is the reason no amount of clever post-processing of a network's output can recover information that an earlier layer discarded. Theorem 11.28 upgrades it to mutual information, which is the form used in representation-learning papers.

A caution on part (1). "Conditioning reduces entropy" is a statement about the *average* $H(Y\mid X)$. It is **false** for individual conditioning events: $H(P_{Y\mid X=x})$ can exceed $H(Y)$ for particular $x$. Here is an example that actually exhibits the increase. Let $Y\sim\mathrm{Ber}(0.9)$ — a heavily loaded coin, $H(Y) = H_2(0.9) = 0.468996$ bits — and let $X$ be a noisy reading of $Y$ that is flipped with probability $0.2$. The surprising reading $X = 0$ occurs with probability $0.9(0.2)+0.1(0.8) = 0.26$ and yields $P(Y=1\mid X=0) = 0.18/0.26 = 9/13 = 0.692308$, so
$$H(P_{Y\mid X=0}) = H_2(9/13) = 0.890492\ \text{bits} \;>\; 0.468996\ \text{bits} = H(Y):$$
the surprising observation nearly *doubled* the uncertainty, by destroying the prior's confidence without replacing it with confidence of its own. The unsurprising reading $X=1$ gives $P(Y=1\mid X=1) = 0.972973$ and entropy $0.179256$ bits, and the average is $0.26\times0.890492 + 0.74\times0.179256 = 0.364177 < 0.468996$ — the *average* decreases, exactly as part (1) demands. So an individual observation can leave you more uncertain than you started, but observations cannot do so on average; papers that say "observing $x$ increased my uncertainty" are describing a real phenomenon, not a computational error.

**Proposition 11.8 (concavity).** On the simplex of distributions over a fixed finite $\mathcal{X}$, the map $P \mapsto H(P)$ is concave: for $\lambda\in[0,1]$, $H(\lambda P + (1-\lambda)Q) \ge \lambda H(P) + (1-\lambda)H(Q)$.

*Proof.* The scalar function $\phi(t) = -t\log t$ is concave on $[0,1]$ since $\phi''(t) = -1/t < 0$. Then $H(P) = \sum_x \phi(p(x))$ is a sum of concave functions of the coordinates, hence concave. $\blacksquare$

Concavity has an interpretation you should keep: mixing distributions never decreases uncertainty. It is also the reason mutual information is concave in the input distribution for a fixed channel, which is why channel capacity is a well-posed convex program — and, in the ML world, why the "optimal input distribution" computations in information-bottleneck papers are tractable at all.

### 11.1.4 Maximum entropy and why the exponential family is not arbitrary

Here is the result that links this chapter to Chapter 3. Suppose all you know about a distribution is the value of a few expectations — a mean, a second moment, a set of feature averages. Which distribution should you use? The maximum entropy answer: the one that commits to nothing beyond the constraints. That answer is always an exponential family member.

**Theorem 11.9 (maximum entropy under moment constraints).** Let $\mu$ be a $\sigma$-finite base measure on $\mathcal{X}$, let $T:\mathcal{X}\to\mathbb{R}^m$ be measurable, and let $\tau\in\mathbb{R}^m$. Consider
$$\mathcal{C} := \{P \ll \mu:\ \mathbb{E}_P[T(X)] = \tau\}.$$
Define entropy relative to $\mu$ as $H_\mu(P) := -\int \log\frac{\mathrm{d}P}{\mathrm{d}\mu}\,\mathrm{d}P$. Suppose there exists $\lambda\in\mathbb{R}^m$ such that
$$p_\lambda(x) := \frac{\mathrm{d}P_\lambda}{\mathrm{d}\mu}(x) = \exp\bigl(\langle\lambda, T(x)\rangle - A(\lambda)\bigr), \qquad A(\lambda) := \log\int e^{\langle \lambda,T\rangle}\mathrm{d}\mu < \infty,$$
satisfies $\mathbb{E}_{P_\lambda}[T] = \tau$ (so $P_\lambda\in\mathcal{C}$). Then for every $Q \in \mathcal{C}$ with $H_\mu(Q) > -\infty$,
$$H_\mu(Q) = H_\mu(P_\lambda) - D_{\mathrm{KL}}(Q\Vert P_\lambda) \le H_\mu(P_\lambda),$$
with equality iff $Q = P_\lambda$. Thus $P_\lambda$ is the unique maximum-entropy element of $\mathcal{C}$.

*Proof.* Let $q = \mathrm{d}Q/\mathrm{d}\mu$ (which exists: if $Q\not\ll\mu$ then $H_\mu(Q)$ is undefined, so we may assume it does, using Radon–Nikodym from Chapter 2). Compute
$$D_{\mathrm{KL}}(Q\Vert P_\lambda) = \int q \log\frac{q}{p_\lambda}\,\mathrm{d}\mu = -H_\mu(Q) - \int q\log p_\lambda\,\mathrm{d}\mu = -H_\mu(Q) - \bigl(\langle\lambda,\mathbb{E}_Q[T]\rangle - A(\lambda)\bigr).$$
Because $Q\in\mathcal{C}$, $\mathbb{E}_Q[T]=\tau$, so the bracket equals $\langle\lambda,\tau\rangle - A(\lambda)$, which does **not** depend on $Q$. Substituting $Q=P_\lambda$ (also in $\mathcal{C}$, with $D_{\mathrm{KL}}(P_\lambda\Vert P_\lambda)=0$) identifies that constant as $-H_\mu(P_\lambda)$. Therefore $D_{\mathrm{KL}}(Q\Vert P_\lambda) = H_\mu(P_\lambda) - H_\mu(Q)$. Non-negativity and the equality case are Theorem 11.17. $\blacksquare$

Two comments. First, the identity $H_\mu(P_\lambda) - H_\mu(Q) = D_{\mathrm{KL}}(Q\Vert P_\lambda)$ is an exact accounting, not just an inequality: the entropy you give up by choosing a different distribution with the same moments is *precisely* the KL divergence to the exponential-family member. It is a Pythagorean identity, and Chapter 3's geometry of exponential families is where it comes from. Second, the hypothesis "suppose there exists $\lambda$ with matching moments" is doing real work; it can fail (e.g. constraints that are infeasible, or feasible only in a limit). I have stated it as a hypothesis rather than pretend it is automatic.

Specialisations you should recognise instantly, all immediate from Theorem 11.9 with the right $\mu$ and $T$:

| Base measure $\mu$ | Constraint $T$ | Max-entropy law |
|---|---|---|
| counting on finite $\mathcal{X}$ | none | uniform |
| counting on $\{0,1,2,\dots\}$ | $\mathbb{E}[X]=\tau$ | geometric |
| counting on $\{1,\dots,6\}$ | $\mathbb{E}[X]=\tau$ | tilted die (Worked Example 11.4) |
| Lebesgue on $[a,b]$ | none | uniform |
| Lebesgue on $\mathbb{R}$ | $\mathbb{E}[X],\ \mathbb{E}[X^2]$ | Gaussian (Theorem 11.14) |
| Lebesgue on $[0,\infty)$ | $\mathbb{E}[X]$ | exponential |
| Lebesgue on $\mathbb{R}^d$ | $\mathbb{E}[XX^\top]=\Sigma$ | $N(0,\Sigma)$ |

The last two rows are exactly why Gaussians appear as default priors, and the first row is why softmax appears as a default output layer — but be careful to say it the right way round. Take counting measure on the $K$ classes and constrain the *expected values of the features*, $\mathbb{E}[T(Y)] = \tau$, where $T$ maps each class to its feature vector. Theorem 11.9 then says the least-committal distribution consistent with those feature averages is $p(k) \propto e^{\langle\lambda, T(k)\rangle}$: log-probabilities *come out* linear in the features. That is a softmax. The exponential-of-linear form is the conclusion of the maximisation, not an assumption fed into it — constraining the log-probabilities to be linear would be circular, since that constraint already asserts the answer. Nobody chose softmax for convenience; it is what "assume nothing beyond the feature averages" forces.

**Worked Example 11.3 (how much does a diagnostic test tell you?).** Chapter 1's example (v): prevalence $P(D) = 0.005$, sensitivity $P(+\mid D) = 0.99$, specificity $P(-\mid D^c) = 0.95$. From Proposition 1.12, $P(+) = 0.005(0.99) + 0.995(0.05) = 0.0547$ and $P(D\mid +) = 0.090494$; also $P(D\mid -) = 0.005(0.01)/0.9453 = 5.289\times 10^{-5}$.

Entropies in bits, all computed numerically:
$$H(D) = H_2(0.005) = 0.045415,\qquad H(D\mid T) = 0.0547\,H_2(0.090494) + 0.9453\, H_2(5.289\times 10^{-5}) = 0.024747,$$
$$I(D;T) = 0.045415 - 0.024747 = 0.020667\ \text{bits}.$$
So the test removes $45.5\%$ of the uncertainty about disease status — and the *absolute* amount is about $0.0207$ bits, roughly one fiftieth of a single yes/no answer. Both readings are correct and they teach opposite lessons. The relative reading says the test is informative. The absolute reading says that with prevalence this low there was very little uncertainty to remove in the first place: $H(D)=0.0454$ bits, because $H$ is tiny near $p=0$ (Figure 11.1). Meanwhile $H(T) = H_2(0.0547) = 0.306036$ bits, so the test result itself is far more "uncertain" than the disease status it is diagnosing. A large fraction of the test's own entropy is noise, which is exactly the base-rate story from Chapter 1 restated in information units: most positives are false positives, and mutual information does not fix that, it quantifies it.

**Worked Example 11.4 (the maximum-entropy die).** Someone tells you a six-sided die has mean $4.5$ instead of $3.5$. What is the least-committal distribution consistent with that? Theorem 11.9 with $\mu = \#$ on $\{1,\dots,6\}$ and $T(x)=x$ gives $p_\lambda(k)\propto e^{\lambda k}$. Solving $\mathbb{E}_{P_\lambda}[X]=4.5$ numerically (Brent's method) gives $\lambda = 0.371049$ and
$$p_\lambda = (0.054353,\ 0.078772,\ 0.114160,\ 0.165447,\ 0.239774,\ 0.347494),$$
with $H(p_\lambda) = 2.327905$ bits, against the unconstrained maximum $\log_2 6 = 2.584963$ bits. The constraint cost $0.257$ bits.

Now compare an *ad hoc* guess with the same mean: $q = (0,0,0,0.5,0.5,0)$, i.e. "it's a 4 or a 5, equally likely." Then $H(q) = 1$ bit and
$$D_{\mathrm{KL}}(q\Vert p_\lambda) = 1.327905\ \text{bits} = 2.327905 - 1 = H(p_\lambda) - H(q),$$
verifying the Pythagorean identity of Theorem 11.9 to six decimals. The interpretation is sharp: by choosing $q$ over the maximum-entropy law you have smuggled in $1.33$ bits of assumptions you were never given.

> **Self-check 11.1.**
> 1. $X$ is uniform on $\{1,\dots,8\}$ and $Y = X \bmod 2$. Compute $H(X)$, $H(Y)$, $H(X\mid Y)$, and $H(Y\mid X)$ in bits, without a calculator.
> 2. True or false: if $H(Y\mid X) = 0$ then $Y = g(X)$ for some function $g$. Which Chapter 1 result is this?
> 3. You are told $\mathbb{E}[X] = 0$ and $\mathbb{E}[X^2] = 1$ for a real-valued $X$ with a density. Which distribution maximises differential entropy, and how many bits do you lose by using a Laplace distribution with the same variance instead? (You may quote §11.2's numbers.)
> 4. Why does Proposition 11.2's proof not depend circularly on Theorem 11.17?

## 11.2 Differential entropy, carefully

### 11.2.1 The definition, and a warning label

The obvious way to extend entropy to a real-valued random variable with a density is to replace the sum by an integral. The obvious thing is what everybody does, and it is fine as long as you never forget what it does *not* mean.

**Definition 11.10 (differential entropy).** Let $P$ be a probability measure on $\mathbb{R}^d$ with $P \ll \lambda_d$ and density $p = \mathrm{d}P/\mathrm{d}\lambda_d$. The **differential entropy** is
$$h(P) = h(X) := -\int_{\mathbb{R}^d} p(x)\log p(x)\,\mathrm{d}\lambda_d(x) = -\mathbb{E}[\log p(X)],$$
whenever the integral exists. More generally, for a $\sigma$-finite base measure $\mu$ and $P\ll\mu$ with density $p_\mu$, define the **entropy relative to $\mu$** by $H_\mu(P) := -\int \log p_\mu\,\mathrm{d}P$. Discrete entropy is $H_\#$ and differential entropy is $H_{\lambda_d}$: they are the *same* definition with different base measures.

Everything that goes wrong with differential entropy, and everything that stays right, follows from that last sentence. The base measure is a choice. Counting measure is canonical on a countable set — there is only one sensible way to count points — so $H_\#$ is intrinsic. Lebesgue measure is *not* canonical on $\mathbb{R}^d$, because it depends on the units in which you measure $x$. Change from metres to centimetres and Lebesgue measure changes by a factor of $100^d$, so the density changes, so $h$ changes. Nothing is broken; you were just measuring a quantity that was never coordinate-free.

Three consequences, which we now prove:

- $h$ can be negative (Proposition 11.11), so it cannot be "expected number of bits."
- $h$ is not the limit of discrete entropies; the discretised entropy diverges like $-d\log\Delta$ (Proposition 11.13).
- $h$ shifts under a change of variables by exactly $\mathbb{E}[\log\lvert\det J\rvert]$ (Theorem 11.12), whereas KL and mutual information do not shift at all (Proposition 11.15).

**Proposition 11.11 (standard differential entropies).** In nats:
$$h(\mathrm{Unif}[0,a]) = \log a, \qquad h(N(\mu,\sigma^2)) = \tfrac12\log(2\pi e\sigma^2), \qquad h\bigl(N(m,\Sigma)\bigr) = \tfrac12\log\bigl((2\pi e)^d\det\Sigma\bigr),$$
$$h(\mathrm{Exp}(\text{rate }\beta)) = 1 - \log\beta, \qquad h(\mathrm{Laplace}(0,b)) = 1+\log(2b).$$

*Proof.* For the uniform, $p \equiv 1/a$ on $[0,a]$, so $-\int p\log p = -\log(1/a) = \log a$. For the Gaussian, write $\log p(x) = -\tfrac12\log(2\pi\det\Sigma) - \tfrac12 (x-m)^\top\Sigma^{-1}(x-m)$ in the multivariate case, so
$$h = \tfrac12\log\bigl((2\pi)^d\det\Sigma\bigr) + \tfrac12\mathbb{E}\bigl[(X-m)^\top\Sigma^{-1}(X-m)\bigr] = \tfrac12\log\bigl((2\pi)^d\det\Sigma\bigr) + \tfrac{d}{2},$$
using $\mathbb{E}[(X-m)^\top\Sigma^{-1}(X-m)] = \operatorname{tr}(\Sigma^{-1}\Sigma) = d$; absorb $d/2 = \tfrac12\log e^d$. The exponential and Laplace cases are the same computation. $\blacksquare$

**Worked Example 11.5 (negative entropy is normal, not pathological).** Take $a = 1/2$: $h(\mathrm{Unif}[0,\tfrac12]) = \log(1/2) = -0.693147$ nats $= -1$ bit. Take a Gaussian with $\sigma = 0.1$: $h = \tfrac12\log(2\pi e \cdot 0.01) = -0.883647$ nats. Take $\sigma=1$: $h = 1.418939$ nats $= 2.047096$ bits.

What is going on? The density of $\mathrm{Unif}[0,\tfrac12]$ equals $2$ everywhere on its support, so $-\log p = -\log 2 < 0$ pointwise. Densities are not probabilities and may exceed $1$ — that was Chapter 1's Failure 1, resolved in Chapter 2 by Radon–Nikodym, and here is its shadow: a density greater than $1$ makes surprisal negative. A negative differential entropy says "this distribution is more concentrated than the unit-scale reference," nothing more. If you rescale to $\mathrm{Unif}[0,2]$ you get $h = +0.693$, describing the same *shape* of distribution. The number moved; the distribution did not.

For a two-dimensional example with correlation, take $\Sigma = \begin{pmatrix}2 & 0.9\\ 0.9 & 1\end{pmatrix}$, $\det\Sigma = 1.19$. Then $h(N(0,\Sigma)) = \tfrac12\log\bigl((2\pi e)^2 \cdot 1.19\bigr) = 2.924854$ nats. Whitening — i.e. setting $Z = L^{-1}X$ where $LL^\top = \Sigma$ — changes $h$ by $-\log\lvert\det L\rvert = -\tfrac12\log\det\Sigma = -0.086977$, landing at $h(N(0,I_2)) = 2.837877$ nats, as Theorem 11.12 requires. Whitening is a bijection: it destroys no information whatsoever, yet it changes $h$. If you needed one sentence to convince yourself that $h$ is not "information content," that is the sentence.

### 11.2.2 Change of variables: where the Jacobian goes

**Theorem 11.12 (differential entropy under a diffeomorphism).** Let $X$ have density $p_X$ on an open set $U\subseteq\mathbb{R}^d$, and let $T:U\to V$ be a $C^1$ diffeomorphism with Jacobian matrix $J_T(x)$. Put $Y = T(X)$. If $h(X)$ and $\mathbb{E}[\log\lvert\det J_T(X)\rvert]$ are both finite, then
$$h(Y) = h(X) + \mathbb{E}\bigl[\log\lvert\det J_T(X)\rvert\bigr].$$

*Proof.* By the change-of-variables formula (Ch. 3), $Y$ has density
$$p_Y(y) = p_X\bigl(T^{-1}(y)\bigr)\,\bigl\lvert\det J_{T^{-1}}(y)\bigr\rvert = \frac{p_X(x)}{\lvert\det J_T(x)\rvert}\Big|_{x = T^{-1}(y)} ,$$
where the second form uses $J_{T^{-1}}(T(x)) = J_T(x)^{-1}$ and $\det(A^{-1}) = 1/\det A$. Therefore
$$h(Y) = -\mathbb{E}\bigl[\log p_Y(Y)\bigr] = -\mathbb{E}\!\left[\log p_X(X) - \log\lvert\det J_T(X)\rvert\right] = h(X) + \mathbb{E}\bigl[\log\lvert\det J_T(X)\rvert\bigr],$$
where we used that $Y = T(X)$ so $T^{-1}(Y) = X$ almost surely, and linearity of expectation (legitimate because both terms are integrable by hypothesis). $\blacksquare$

This formula is the engine of normalising flows, and it is worth seeing the identification explicitly. A flow model defines $x = G_\theta(z)$ with $z\sim N(0,I_d)$ and $G_\theta$ invertible, and reports
$$\log p_\theta(x) = \log \varphi\bigl(G_\theta^{-1}(x)\bigr) - \log\bigl\lvert\det J_{G_\theta}\bigl(G_\theta^{-1}(x)\bigr)\bigr\rvert .$$
Taking expectations under $p_\theta$ and comparing with Theorem 11.12: $h(X) = h(Z) + \mathbb{E}[\log\lvert\det J_{G_\theta}(Z)\rvert]$. The log-determinant term that flow architectures work so hard to make cheap (coupling layers, autoregressive triangular Jacobians, $1\times1$ convolutions) is *exactly* the entropy bookkeeping. The architecture is a device for tracking how much a bijection stretches volume.

**Worked Example 11.6 (three changes of variables).**

*(a) Affine.* $X\sim N(0,1)$, $Y = 2X+3$. Then $\log\lvert\det J\rvert = \log 2$ everywhere, so $h(Y) = 1.418939 + 0.693147 = 2.112086$ nats. Direct check: $Y\sim N(3,4)$ and $\tfrac12\log(2\pi e\cdot 4) = 2.112086$. Note the shift by $3$ contributed nothing: translation preserves Lebesgue measure.

*(b) Squashing.* $X\sim N(0,1)$, $Y = \sigma(X) = 1/(1+e^{-X}) \in (0,1)$. Here $\sigma'(x) = \sigma(x)(1-\sigma(x))$, and numerical integration (Gauss–Kronrod on $[-40,40]$) gives $\mathbb{E}[\log\sigma'(X)] = -1.612118$ nats, so
$$h(Y) = 1.418939 - 1.612118 = -0.193180\ \text{nats}.$$
This *must* be negative: $Y$ lives in $(0,1)$, where by Theorem 11.14's cousin the maximum differential entropy over that interval is $\log 1 = 0$, attained by the uniform. So a sigmoid output layer necessarily has negative differential entropy, always, for every input distribution. Any paper that treats "negative entropy" of a bounded activation as a red flag has confused a unit convention for a bug.

*(c) Whitening.* Done in Worked Example 11.5: $h$ shifts by $-\tfrac12\log\det\Sigma$.

**Worked Example 11.7 (8-bit images and the $256^d$ elephant).** Let $x\in[0,1]^d$ be an image scaled to the unit cube, and $\tilde x = 255\,x$ the same image on the standard $[0,255]$ scale. The map is linear with $\det J = 255^d$, so
$$h(\tilde X) = h(X) + d\log 255 .$$
For CIFAR-10-sized data, $d = 32\times32\times3 = 3072$, and using the more common convention of dividing by $256$ (so that a pixel bin has width $1/256$), the shift is $d\log 256 = 17034.79$ nats $= 24576$ bits $= 8d$ bits. That is an enormous number compared with anything a model's log-likelihood will differ by. **Two papers reporting log-likelihoods on different pixel scales are not comparable, and the discrepancy is bigger than every modelling improvement in the literature combined.** §11.8 turns this into an auditing procedure.

### 11.2.3 Why $h$ is not the limit of $H$

**Proposition 11.13 (quantisation).** Let $X$ have a density $p$ on $\mathbb{R}^d$ that is bounded, compactly supported, and continuous outside a Lebesgue-null set. For $\Delta>0$ let $X^\Delta$ be $X$ rounded to the grid $\Delta\mathbb{Z}^d$ (say, to the lower-left corner of its cell). Then
$$\lim_{\Delta\downarrow 0}\ \Bigl[H(X^\Delta) + d\log\Delta\Bigr] = h(X).$$
Equivalently, $H(X^\Delta) = h(X) - d\log\Delta + o(1)$, so $H(X^\Delta)\to+\infty$.

(The hypotheses are stated this way deliberately: a *continuous* density would exclude $\mathrm{Unif}[0,1]$, which is discontinuous at its endpoints and is the very example we have been using. Almost-everywhere continuity plus boundedness covers uniforms, truncated Gaussians, mixtures with jumps, and everything else you are likely to meet.)

*Proof.* Index the grid cells by $i$, write $B_i$ for cell $i$ (volume $\Delta^d$), and set
$$p_i := \mathbb{P}(X\in B_i) = \int_{B_i} p\,\mathrm{d}\lambda, \qquad \bar p_i := \frac{p_i}{\Delta^d} = \text{the average of } p \text{ over } B_i .$$
Only finitely many cells matter, since $p$ has compact support. Substituting $p_i = \Delta^d\bar p_i$,
$$H(X^\Delta) = -\sum_i p_i\log p_i = -\sum_i \Delta^d\,\bar p_i\bigl[\log\bar p_i + d\log\Delta\bigr] = \underbrace{-\sum_i \Delta^d\,\bar p_i\log\bar p_i}_{(\ast)} \;-\; d\log\Delta\underbrace{\sum_i \Delta^d\bar p_i}_{=\,1},$$
so $H(X^\Delta)+d\log\Delta = (\ast)$, and the whole claim is that $(\ast)\to-\int p\log p = h(X)$.

Now $(\ast)$ is the integral of the **step function** $p^\Delta := \sum_i \bar p_i\mathbf{1}_{B_i}$ against $-\log p^\Delta$, i.e. $(\ast) = -\int p^\Delta\log p^\Delta\,\mathrm{d}\lambda$. At every point of continuity of $p$ we have $p^\Delta(x)\to p(x)$ as $\Delta\downarrow0$, because $\bar p_i$ is an average of $p$ over a shrinking neighbourhood of $x$; by hypothesis that is $\lambda$-almost every $x$. Also $0\le p^\Delta\le \lVert p\rVert_\infty$ uniformly, all the $p^\Delta$ vanish outside a fixed compact set (a one-cell enlargement of $\mathrm{supp}\,p$), and $t\mapsto -t\log t$ is continuous and bounded on $[0,\lVert p\rVert_\infty]$. So $-p^\Delta\log p^\Delta \to -p\log p$ almost everywhere, dominated by a constant times the indicator of that compact set, and the Dominated Convergence Theorem (Ch. 2) gives $(\ast)\to-\int p\log p$. $\blacksquare$

Read the formula: a continuous variable observed at resolution $\Delta$ carries about $h(X) + d\log(1/\Delta)$ nats. The $d\log(1/\Delta)$ part is the cost of specifying *where in the cell* you are, and it blows up as you refine. Differential entropy is what is left after you subtract that infinity — a **relative** quantity, the amount by which $X$ is more or less spread out than a unit-volume box. That is a perfectly good thing to measure. It is not a count of bits.

> **Common Confusion 11.2.** *"Differential entropy is just the continuous version of entropy. So $h(N(0,1)) = 1.42$ nats should mean a standard normal 'contains' about $2.05$ bits, and a very concentrated Gaussian should contain fewer bits — but you're telling me it can be negative? Negative bits?"*
>
> There are three separate errors bundled together here, and it is worth separating them.
>
> *Error 1: $h$ is not a limit of $H$.* By Proposition 11.13, $H(X^\Delta) \approx h(X) + d\log(1/\Delta)$. The true "number of bits in a real number" is infinite, for every non-degenerate continuous distribution. Differential entropy is the finite part after removing a divergence, and where you put the cut ($\Delta = 1$? $\Delta = 1/256$?) is your choice.
>
> *Error 2: negative $h$ is not a contradiction because $h$ is not a count.* $-\log p(x)$ is negative wherever $p(x)>1$, and densities routinely exceed $1$ (Chapter 1's Failure 1). $h(\mathrm{Unif}[0,\tfrac12]) = -1$ bit exactly.
>
> *Error 3 (the important one): $h$ is not a property of the random variable, only of the variable-and-coordinate-system.* Whitening a correlated Gaussian is a bijection, and it changed $h$ from $2.924854$ to $2.837877$ nats (Worked Example 11.5). No information was lost or gained — a bijection cannot do that. Therefore $h$ is not measuring information. What *is* invariant is $D_{\mathrm{KL}}$ and $I$ (Proposition 11.15). The practical rule: never compare differential entropies across parameterisations, never report one without a base measure, and if what you actually care about is invariant, write it as a divergence and the problem disappears.

**Theorem 11.14 (the Gaussian maximises $h$ at fixed covariance).** Let $P$ on $\mathbb{R}^d$ have density $q$, mean $m$, and covariance $\Sigma \succ 0$. Then
$$h(P) \le \tfrac12\log\bigl((2\pi e)^d\det\Sigma\bigr) = h\bigl(N(m,\Sigma)\bigr),$$
with equality iff $P = N(m,\Sigma)$.

*Proof.* Let $\varphi$ be the $N(m,\Sigma)$ density. By Theorem 11.17,
$$0 \le D_{\mathrm{KL}}(P\Vert N(m,\Sigma)) = \int q\log q - \int q \log\varphi = -h(P) - \int q\log\varphi.$$
Now $\log\varphi(x) = -\tfrac12\log\bigl((2\pi)^d\det\Sigma\bigr) - \tfrac12 (x-m)^\top\Sigma^{-1}(x-m)$, and since $P$ has mean $m$ and covariance $\Sigma$,
$$\int q\log \varphi = -\tfrac12\log\bigl((2\pi)^d\det\Sigma\bigr) - \tfrac12\operatorname{tr}\bigl(\Sigma^{-1}\Sigma\bigr) = -\tfrac12\log\bigl((2\pi)^d\det\Sigma\bigr) - \tfrac{d}{2}.$$
Crucially this depends on $P$ only through its first two moments, which are fixed. Rearranging gives the bound, and equality in Theorem 11.17 forces $P = N(m,\Sigma)$. $\blacksquare$

The trick — *replace the unknown density's log by the reference's log, which only involves the constrained moments* — is Theorem 11.9 again, with $\mu = \lambda_d$ and $T(x) = (x, xx^\top)$. Numerically, at variance $\sigma^2 = 1.69$: the Gaussian achieves $h = 1.681303$ nats, the uniform with the same variance (width $\sigma\sqrt{12} = 4.503332$) achieves $\log(4.503332) = 1.504818$, and the Laplace with the same variance (scale $b = \sigma/\sqrt2 = 0.919239$) achieves $1 + \log(2b) = 1.608938$. Gaussian wins, as promised, and the gaps are the respective KL divergences to the Gaussian.

### 11.2.4 What survives: KL and mutual information are invariant

**Proposition 11.15 (reparameterisation invariance).** Let $T:\mathcal{X}\to\mathcal{Y}$ be a bimeasurable bijection (measurable with measurable inverse) between measurable spaces. Then for all probability measures $P,Q$ on $\mathcal{X}$,
$$D_{\mathrm{KL}}\bigl(T_\#P \,\Vert\, T_\#Q\bigr) = D_{\mathrm{KL}}(P\Vert Q).$$
Consequently, if $S,T$ are bimeasurable bijections then $I\bigl(S(X); T(Y)\bigr) = I(X;Y)$.

*Proof.* If $P\not\ll Q$ then there is $A$ with $Q(A) = 0 < P(A)$; then $T(A)$ is measurable (as $T^{-1}$ is measurable, $T(A) = (T^{-1})^{-1}(A)$) and $T_\#Q(T(A)) = 0 < T_\#P(T(A))$, so both sides are $+\infty$. So assume $P\ll Q$ with $r := \mathrm{d}P/\mathrm{d}Q$. I claim $\mathrm{d}(T_\#P)/\mathrm{d}(T_\#Q) = r\circ T^{-1}$. Indeed for measurable $B\subseteq\mathcal{Y}$,
$$\int_B (r\circ T^{-1})\,\mathrm{d}(T_\#Q) = \int_{T^{-1}(B)} r \,\mathrm{d}Q = P\bigl(T^{-1}(B)\bigr) = T_\#P(B),$$
where the first equality is the pushforward change-of-variables identity $\int g\,\mathrm{d}(T_\#Q) = \int g\circ T\,\mathrm{d}Q$ (Ch. 2) applied to $g = \mathbf{1}_B\cdot(r\circ T^{-1})$. Hence
$$D_{\mathrm{KL}}(T_\#P\Vert T_\#Q) = \int \log\bigl(r\circ T^{-1}\bigr)\,\mathrm{d}(T_\#P) = \int \log r\,\mathrm{d}P = D_{\mathrm{KL}}(P\Vert Q).$$
For the mutual information claim, apply the above with the bijection $(x,y)\mapsto (S(x),T(y))$ on the product space, noting that it maps $P_{XY}\mapsto P_{S(X)T(Y)}$ and $P_X\otimes P_Y \mapsto P_{S(X)}\otimes P_{T(Y)}$, the latter because pushforwards of product measures under product maps are products. $\blacksquare$

Notice what the proof used: the *density ratio* transforms by composition with $T^{-1}$ and no Jacobian factor, because the Jacobian appears in numerator and denominator and cancels. That cancellation is the whole story. Differential entropy has only a numerator, so nothing cancels, so it moves. Every $f$-divergence in §11.5 inherits the invariance for the same reason.

Practical upshots, worth stating concretely:

- The KL term in a VAE loss, $D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))$, is unchanged if you reparameterise the latent space by any bijection — as long as you transform *both* $q$ and $p$. Papers that reparameterise only the posterior and then compare KL numbers are comparing different objectives.
- A mutual information between a representation and a label is invariant to whether you store the representation as raw floats, whitened floats, or PCA coefficients (bijections all). Correlation is not: it can be destroyed by a nonlinear bijection. That is the concrete advantage over correlation-based diagnostics, and it will come up again in §11.4.
- Reported "entropy of the latent" numbers in flow and VAE papers are only meaningful relative to a stated base measure. Reported KLs are meaningful as they stand.

**Worked Example 11.8 (the variable that has no entropy at all: $A = \max(0,Z)$).** Chapter 1's example (ii): $Z\sim N(0,1)$ and $A = \max(0,Z)$, with $P(A = 0) = 1/2$ and $F_A(a) = \Phi(a)$ for $a\ge 0$. Chapter 1 established that $P_A$ has neither a pmf (it has a continuous part) nor a Lebesgue density (it has an atom). Status of Chapter 2's Failure 2: half-resolved by describing $P_A$ through its CDF. Here is the entropy version of the full resolution.

$H_\#(P_A)$ is not defined in any useful way: $P_A$ is not absolutely continuous with respect to counting measure on $\mathbb{R}$ (a countable set can carry at most the atom's mass $1/2$, never all of it). $h(P_A) = H_{\lambda}(P_A)$ is also not defined: $P_A \not\ll\lambda$, because $\lambda(\{0\}) = 0 < 1/2 = P_A(\{0\})$. So neither of the two standard entropies exists. But now use Definition 11.10's general form with the base measure Chapter 2 told us to use, $\mu := \delta_0 + \lambda$. Then $P_A\ll\mu$ with
$$\frac{\mathrm{d}P_A}{\mathrm{d}\mu}(a) = \begin{cases} 1/2, & a = 0,\\ \varphi(a), & a>0,\\ 0, & a<0,\end{cases}$$
and therefore
$$H_\mu(P_A) = -\tfrac12\log\tfrac12 - \int_0^\infty \varphi(a)\log\varphi(a)\,\mathrm{d}a = \tfrac12\log 2 + \tfrac14\log(2\pi) + \tfrac14 = 1.056043\ \text{nats}.$$
The closed form comes from $\int_0^\infty\varphi\log\varphi = -\tfrac14\log(2\pi) - \tfrac12\mathbb{E}[Z^2\mathbf{1}\{Z>0\}] = -\tfrac14\log(2\pi) - \tfrac14$, using $\mathbb{E}[Z^2\mathbf{1}\{Z>0\}] = 1/2$ by symmetry; numerical quadrature agrees to eight decimals ($0.70946927$ for $-\int_0^\infty\varphi\log\varphi$). **One object, one formula, valid for discrete, continuous, and mixed laws — provided you say which $\mu$.** That is Failure 2 closed, and it is closed by the same device that closed Failure 1: name the base measure.

> **Self-check 11.2.**
> 1. $X\sim\mathrm{Unif}[0,10^{-3}]$. Compute $h(X)$ in bits. Is $X$ "less random" than a fair coin flip?
> 2. $Y = AX$ for an invertible matrix $A$. By how much does $h$ change? What happens if $A$ is a rotation?
> 3. A student reports "the differential entropy of my model's latent is $-4.1$ nats, which is impossible." What do you say?
> 4. Why is $I(X;Y)$ invariant under $X\mapsto X^3$ on $\mathbb{R}$ while $\mathrm{Corr}(X,Y)$ is not?

## 11.3 KL divergence revisited

### 11.3.1 The definition and Gibbs' inequality

**Definition 11.16 (Kullback–Leibler divergence; recalled from Ch. 2).** For probability measures $P,Q$ on a common measurable space,
$$D_{\mathrm{KL}}(P\Vert Q) := \begin{cases}\displaystyle \int \log\frac{\mathrm{d}P}{\mathrm{d}Q}\,\mathrm{d}P = \int \frac{\mathrm{d}P}{\mathrm{d}Q}\log\frac{\mathrm{d}P}{\mathrm{d}Q}\,\mathrm{d}Q, & P\ll Q,\\[4pt] +\infty, & \text{otherwise.}\end{cases}$$
If both $P,Q\ll\mu$ with densities $p,q$, then $D_{\mathrm{KL}}(P\Vert Q) = \int p\log(p/q)\,\mathrm{d}\mu$, and this value does not depend on which common $\mu$ you chose — a fact you should now see as an instance of Proposition 11.15's cancellation.

**Theorem 11.17 (Gibbs' inequality).** $D_{\mathrm{KL}}(P\Vert Q)\ge 0$, with equality if and only if $P = Q$.

*Proof.* If $P\not\ll Q$ the claim is trivial. Otherwise let $r := \mathrm{d}P/\mathrm{d}Q$ and $\phi(t) := t\log t$, extended by $\phi(0)=0$. Then $\phi$ is convex on $[0,\infty)$ and *strictly* convex on $(0,\infty)$, since $\phi''(t) = 1/t > 0$. By Jensen's inequality (Ch. 4) applied under $Q$,
$$D_{\mathrm{KL}}(P\Vert Q) = \mathbb{E}_Q[\phi(r)] \ \ge\ \phi\bigl(\mathbb{E}_Q[r]\bigr) = \phi(1) = 0,$$
because $\mathbb{E}_Q[r] = \int \mathrm{d}P/\mathrm{d}Q\,\mathrm{d}Q = P(\mathcal{X}) = 1$. For the equality case: Jensen's inequality with a strictly convex $\phi$ is an equality iff $r$ is $Q$-almost surely constant. Since $\mathbb{E}_Q[r]=1$ that constant is $1$, so $\mathrm{d}P/\mathrm{d}Q = 1$ $Q$-a.s., which gives $P(A) = \int_A r\,\mathrm{d}Q = Q(A)$ for every measurable $A$, i.e. $P = Q$. Conversely $P=Q$ gives $r\equiv1$ and $D_{\mathrm{KL}}=0$. $\blacksquare$

Everything in §11.1 that I deferred is now paid for: Proposition 11.2 (uniform maximises entropy), Proposition 11.7 (conditioning reduces entropy), Theorem 11.9 (max-entropy is exponential-family), and Theorem 11.14 (Gaussian maximises $h$) all reduced to Theorem 11.17, and none of their proofs used the results now being justified.

**Proposition 11.18 (KL is not a metric).** In general $D_{\mathrm{KL}}(P\Vert Q)\ne D_{\mathrm{KL}}(Q\Vert P)$, and the triangle inequality fails.

*Proof.* Both by counterexample, computed exactly. With $\mathrm{Ber}(p)$ denoting the Bernoulli law:
$$D_{\mathrm{KL}}\bigl(\mathrm{Ber}(0.1)\Vert\mathrm{Ber}(0.5)\bigr) = 0.368064,\qquad D_{\mathrm{KL}}\bigl(\mathrm{Ber}(0.5)\Vert\mathrm{Ber}(0.1)\bigr) = 0.510826$$
nats, so it is not symmetric. For the triangle inequality, take $P = \mathrm{Ber}(0.1)$, $R = \mathrm{Ber}(0.11)$, $S = \mathrm{Ber}(0.9)$:
$$D_{\mathrm{KL}}(P\Vert S) = 1.757780 \quad>\quad D_{\mathrm{KL}}(P\Vert R) + D_{\mathrm{KL}}(R\Vert S) = 0.000525 + 1.714375 = 1.714900 . \ \blacksquare$$

Both facts have teeth. The asymmetry is the difference between "fit the model to the data" and "fit an approximation to the model," and it produces qualitatively different solutions — that is the content of Figure 11.2 and the lab of §11.3.3. The failure of the triangle inequality means you cannot chain approximations: "my model is close to a good model, which is close to the truth" does not imply "my model is close to the truth" in KL. Distillation pipelines that compose several KL steps have no telescoping guarantee. (Total variation, being a genuine metric, does telescope — §11.6.)

> **Common Confusion 11.3.** *"KL is a distance between distributions — that's why we minimise it. Isn't $D_{\mathrm{KL}}(P\Vert Q)$ small exactly when $P$ and $Q$ are close?"*
>
> It is a *divergence*: non-negative, zero only on the diagonal, and that is all. It is not symmetric, it does not satisfy the triangle inequality (Proposition 11.18), and it is not even bounded — $D_{\mathrm{KL}}(P\Vert Q) = \infty$ as soon as $P$ puts mass where $Q$ puts none, no matter how tiny that mass is. Two distributions can be visually indistinguishable and have infinite KL: $\mathrm{Unif}[0,1]$ and $\mathrm{Unif}[0,1{-}\varepsilon]$ have $D_{\mathrm{KL}} = \infty$ for every $\varepsilon>0$ in one direction and $-\log(1-\varepsilon)$ in the other. What KL *does* control is likelihood ratios and hypothesis testing, which is exactly what Pinsker's inequality (Theorem 11.38) and Fano's inequality (Theorem 11.51) exploit. The right mental model is not "distance" but "expected log-likelihood ratio in favour of $P$ when $P$ is true," i.e. the exponential rate at which evidence accumulates.

### 11.3.2 Cross-entropy is entropy plus KL, and that is why classifiers use it

**Definition 11.19 (cross-entropy).** For $P\ll\mu$, $Q\ll\mu$ with densities $p,q$, the **cross-entropy** of $Q$ relative to $P$ is
$$H_\mu(P,Q) := -\int p\log q\,\mathrm{d}\mu = -\mathbb{E}_{X\sim P}\bigl[\log q(X)\bigr].$$
(For discrete $\mathcal{X}$ with $\mu = \#$ I will write $H(P,Q)$.)

**Proposition 11.20 (the decomposition).** Whenever the terms are finite,
$$H_\mu(P,Q) = H_\mu(P) + D_{\mathrm{KL}}(P\Vert Q) .$$
Hence for fixed $P$, minimising cross-entropy over $Q$ is exactly minimising $D_{\mathrm{KL}}(P\Vert Q)$, and the minimum value is $H_\mu(P)$, attained iff $Q=P$.

*Proof.* $-\int p\log q = -\int p\log p + \int p\log(p/q)$, term by term. The minimum claim follows from Theorem 11.17. $\blacksquare$

This one line is why classification uses the loss it uses. Let $\hat P_n$ be the empirical distribution (Chapter 1's $\hat P_n$) of labels for a given input, and $Q_\theta$ the model's predicted distribution. The average cross-entropy loss on a dataset is
$$\mathcal{L}(\theta) = -\frac1n\sum_{i=1}^n \log q_\theta(y_i\mid x_i) = \mathbb{E}_{(x,y)\sim\hat P_n}\bigl[-\log q_\theta(y\mid x)\bigr],$$
which by Proposition 11.20 equals $H(\hat P_n) + D_{\mathrm{KL}}(\hat P_n\Vert Q_\theta)$ up to the conditioning bookkeeping. The first term does not involve $\theta$. **So training with cross-entropy loss is literally gradient descent on a KL divergence to the empirical distribution, and the entropy of the data is an additive constant that sets the floor of the loss curve.**

This explains a phenomenon that puzzles people the first time they see it: on a dataset with label noise, the training loss cannot go to zero, and the value it bottoms out at is not arbitrary — it is the conditional entropy of the labels.

**Worked Example 11.9 (the loss floor is an entropy).** Suppose a particular input $x$ appears many times in your dataset and its label is $1$ in $70\%$ of occurrences, $0$ in $30\%$ — genuine label noise, not a modelling failure. Then $\hat P = (0.3, 0.7)$ for that input, and the model's best possible prediction is $q = \hat P$. Cross-entropy in nats, computed for three candidate predictions:

| model prediction $q(1)$ | cross-entropy | $= H(\hat P)$ | $+\ D_{\mathrm{KL}}(\hat P\Vert Q)$ |
|---|---|---|---|
| $0.5$ | $0.693147$ | $0.610864$ | $0.082283$ |
| $0.7$ | $0.610864$ | $0.610864$ | $0.000000$ |
| $0.9$ | $0.764528$ | $0.610864$ | $0.153664$ |

The floor is $H(\hat P) = 0.610864$ nats $= 0.881291$ bits. A model reporting a training loss of $0.61$ nats here is *perfect*, not undertrained. Practitioners who chase loss $\to 0$ on noisy labels are asking the model to memorise noise, and the numeric value of the achievable floor is computable in advance from the label distribution. Conversely, a *lower* loss than the floor on training data is a diagnostic that the model is exploiting some identifier that lets it distinguish nominally identical inputs.

**Proposition 11.21 (MLE $=$ minimising KL to the empirical distribution).** Let $x_1,\dots,x_n$ be observations in a countable space (or in $\mathbb{R}^d$ with a fixed base measure $\mu$ and $P_\theta\ll\mu$), and $\hat P_n = \frac1n\sum_i \delta_{x_i}$ the empirical distribution. Then
$$\arg\max_\theta \frac1n\sum_{i=1}^n \log p_\theta(x_i) \;=\; \arg\min_\theta\, D_{\mathrm{KL}}\bigl(\hat P_n\,\Vert\, P_\theta\bigr)$$
whenever the right-hand side is finite for some $\theta$, and in the discrete case $D_{\mathrm{KL}}(\hat P_n\Vert P_\theta) = -\frac1n\sum_i\log p_\theta(x_i) - H(\hat P_n)$.

*Proof.* In the discrete case, $\hat P_n$ has density $\hat p_n(x) = n_x/n$ where $n_x = \#\{i: x_i = x\}$, so
$$D_{\mathrm{KL}}(\hat P_n\Vert P_\theta) = \sum_x \frac{n_x}{n}\log\frac{n_x/n}{p_\theta(x)} = -H(\hat P_n) - \frac1n\sum_x n_x\log p_\theta(x) = -H(\hat P_n) - \frac1n\sum_{i=1}^n \log p_\theta(x_i).$$
Only the last term depends on $\theta$, and it appears with a minus sign, so minimising the KL maximises the log-likelihood. In the continuous case $\hat P_n\not\ll\mu$, so the KL is literally $+\infty$ and the identity must be read through the cross-entropy: $H_\mu(\hat P_n, P_\theta) = -\frac1n\sum_i\log p_\theta(x_i)$ is finite and is what is minimised; the "entropy of $\hat P_n$" term is the infinite constant that Proposition 11.13 taught us to expect. $\blacksquare$

I want to flag that last sentence rather than bury it, because it is a place where the standard slogan is technically wrong and the wrongness is instructive. "Maximum likelihood minimises $D_{\mathrm{KL}}(\hat P_n\Vert P_\theta)$" is exactly true for discrete data. For continuous data the empirical measure is a sum of Dirac masses and is *mutually singular* with every model with a density, so that KL is $+\infty$ for every $\theta$ and cannot be minimised. What is true is that maximum likelihood minimises the *cross-entropy* $H_\mu(\hat P_n, P_\theta)$, which is finite, and that in the large-$n$ limit this converges (under conditions from Chapter 8) to minimising $D_{\mathrm{KL}}(P_{\text{data}}\Vert P_\theta)$ against the true law. The slogan is right about the population and wrong about the sample. Chapter 2's Failure 4 was the same phenomenon in a different costume: singular measures make KL blind.

### 11.3.3 Forward versus reverse: the most useful picture in the chapter

Both $D_{\mathrm{KL}}(P\Vert Q)$ and $D_{\mathrm{KL}}(Q\Vert P)$ are used constantly, with $P$ the target and $Q$ the tractable approximation. They behave completely differently, and you can read off why from the integrand.

- **Forward KL, $D_{\mathrm{KL}}(P\Vert Q) = \int p\log(p/q)$.** The integral is taken against $p$. Wherever $p>0$ and $q\to0$, the integrand $\to+\infty$. So forward KL *forbids the model from assigning near-zero probability anywhere the target has mass*: it is **mass-covering** (also called zero-avoiding). Where $p=0$ it does not care what $q$ does. This is the maximum-likelihood objective.
- **Reverse KL, $D_{\mathrm{KL}}(Q\Vert P) = \int q\log(q/p)$.** The integral is taken against $q$. Wherever $q>0$ and $p\to 0$, the integrand blows up. So reverse KL *forbids the model from putting mass where the target has none*: it is **mode-seeking** (zero-forcing). It is happy to ignore entire modes of $P$. This is the variational inference objective, since the ELBO's KL term is $D_{\mathrm{KL}}(q_\phi\Vert p)$.

Now the numbers. Take $p = 0.3\,N(-4,1) + 0.7\,N(4,1)$, a clearly bimodal target, and fit a single Gaussian $q = N(m,s^2)$ by minimising each direction. All values below were computed by exact numerical minimisation on a grid of step $10^{-4}$ over $[-20,20]$ (Nelder–Mead on $(m,\log s)$, tolerance $10^{-9}$).

| objective | minimiser | $D_{\mathrm{KL}}(p\Vert q)$ | $D_{\mathrm{KL}}(q\Vert p)$ | $q(x<0)$ |
|---|---|---|---|---|
| forward $D_{\mathrm{KL}}(p\Vert q)$ | $N(1.600,\,3.800^2)$ | $\mathbf{0.724222}$ | $2.095935$ | $0.336858$ |
| reverse $D_{\mathrm{KL}}(q\Vert p)$, global | $N(4.000,\,1.000^2)$ | $8.978460$ | $\mathbf{0.356615}$ | $3.2\times10^{-5}$ |
| reverse, other local min | $N(-4.000,\,1.001^2)$ | $21.732651$ | $1.203826$ | $\approx 1$ |

True $p(x<0) = 0.300013$. Read the table three ways.

First, the forward-KL minimiser is exactly the moment-matching Gaussian: mean $\sum_k w_k\mu_k = 0.3(-4)+0.7(4) = 1.6$ and variance $\sum_k w_k(\mu_k^2+\sigma_k^2) - 1.6^2 = 17 - 2.56 = 14.44 = 3.8^2$. That is not a coincidence — minimising forward KL over an exponential family is moment matching, a direct corollary of Theorem 11.9 (see Exercise A3). It covers both modes, at the price of putting substantial density in the valley between them where $p$ is nearly zero. In generative-model terms: **blurry samples**.

Second, the reverse-KL minimiser collapses onto the heavier mode and matches it almost perfectly ($s = 1.0005$ against the true component width $1$), assigning probability $3.2\times10^{-5}$ to the entire left half of the space, where the truth has $30\%$ of its mass. In generative-model terms: **sharp but non-diverse samples**, i.e. mode collapse. Crucially this is not a failure of the optimiser. It is the exact global optimum of the objective people actually write down in variational inference.

Third, reverse KL has *multiple local minima* — one per mode — and which one you find depends on initialisation. Starting the optimiser at $m_0\in\{-6,-4,-1\}$ converges to the left mode (objective $1.203826$); starting at $m_0\in\{0,1,4,6\}$ converges to the right (objective $0.356615$). Forward KL, by contrast, is a convex problem in the natural parameters of $q$ over an exponential family, so initialisation is irrelevant. The optimisation landscapes are as different as the solutions.

<svg viewBox="0 0 720 470" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<text x="360.0" y="30" fill="currentColor" font-size="15" text-anchor="middle">Figure 11.2 — Forward KL is mass-covering; reverse KL is mode-seeking</text>
<line x1="90" y1="330" x2="660" y2="330" stroke="currentColor" stroke-width="1.2"/>
<line x1="90" y1="330" x2="90" y2="60" stroke="currentColor" stroke-width="1.2"/>
<line x1="147.0" y1="330" x2="147.0" y2="336" stroke="currentColor" stroke-width="1"/>
<text x="147.0" y="352" fill="currentColor" font-size="13" text-anchor="middle">-8</text>
<line x1="261.0" y1="330" x2="261.0" y2="336" stroke="currentColor" stroke-width="1"/>
<text x="261.0" y="352" fill="currentColor" font-size="13" text-anchor="middle">-4</text>
<line x1="375.0" y1="330" x2="375.0" y2="336" stroke="currentColor" stroke-width="1"/>
<text x="375.0" y="352" fill="currentColor" font-size="13" text-anchor="middle">0</text>
<line x1="489.0" y1="330" x2="489.0" y2="336" stroke="currentColor" stroke-width="1"/>
<text x="489.0" y="352" fill="currentColor" font-size="13" text-anchor="middle">4</text>
<line x1="603.0" y1="330" x2="603.0" y2="336" stroke="currentColor" stroke-width="1"/>
<text x="603.0" y="352" fill="currentColor" font-size="13" text-anchor="middle">8</text>
<line x1="84" y1="330.0" x2="90" y2="330.0" stroke="currentColor" stroke-width="1"/>
<text x="80" y="334.0" fill="currentColor" font-size="13" text-anchor="end">0</text>
<line x1="84" y1="201.4" x2="90" y2="201.4" stroke="currentColor" stroke-width="1"/>
<text x="80" y="205.4" fill="currentColor" font-size="13" text-anchor="end">0.2</text>
<line x1="84" y1="72.9" x2="90" y2="72.9" stroke="currentColor" stroke-width="1"/>
<text x="80" y="76.9" fill="currentColor" font-size="13" text-anchor="end">0.4</text>
<text x="375.0" y="375" fill="currentColor" font-size="14" text-anchor="middle">x</text>
<text x="48" y="195.0" fill="currentColor" font-size="14" text-anchor="middle" transform="rotate(-90 48 195.0)">density</text>
<polyline points="90.0,330.0 93.6,330.0 97.1,330.0 100.7,330.0 104.2,330.0 107.8,330.0 111.4,330.0 114.9,330.0 118.5,330.0 122.1,330.0 125.6,330.0 129.2,330.0 132.8,330.0 136.3,330.0 139.9,330.0 143.4,330.0 147.0,330.0 150.6,330.0 154.1,329.9 157.7,329.9 161.2,329.8 164.8,329.7 168.4,329.6 171.9,329.4 175.5,329.1 179.1,328.8 182.6,328.2 186.2,327.5 189.8,326.6 193.3,325.4 196.9,323.9 200.4,322.0 204.0,319.6 207.6,316.7 211.1,313.4 214.7,309.5 218.2,305.0 221.8,300.1 225.4,294.8 228.9,289.1 232.5,283.3 236.1,277.5 239.6,271.9 243.2,266.7 246.8,262.1 250.3,258.3 253.9,255.4 257.4,253.7 261.0,253.1 264.6,253.7 268.1,255.4 271.7,258.3 275.2,262.1 278.8,266.7 282.4,271.9 285.9,277.5 289.5,283.3 293.1,289.1 296.6,294.8 300.2,300.1 303.8,305.0 307.3,309.5 310.9,313.4 314.4,316.7 318.0,319.6 321.6,322.0 325.1,323.9 328.7,325.4 332.2,326.6 335.8,327.5 339.4,328.2 342.9,328.8 346.5,329.1 350.1,329.4 353.6,329.6 357.2,329.7 360.8,329.8 364.3,329.9 367.9,329.9 371.4,329.9 375.0,329.9 378.6,329.9 382.1,329.8 385.7,329.7 389.2,329.6 392.8,329.4 396.4,329.1 399.9,328.6 403.5,328.0 407.1,327.1 410.6,325.9 414.2,324.3 417.8,322.1 421.3,319.3 424.9,315.7 428.4,311.2 432.0,305.7 435.6,299.0 439.1,291.2 442.7,282.1 446.2,271.7 449.8,260.2 453.4,247.8 456.9,234.7 460.5,221.1 464.1,207.6 467.6,194.5 471.2,182.3 474.8,171.6 478.3,162.7 481.9,156.0 485.4,151.9 489.0,150.5 492.6,151.9 496.1,156.0 499.7,162.7 503.2,171.6 506.8,182.3 510.4,194.5 513.9,207.6 517.5,221.1 521.1,234.7 524.6,247.8 528.2,260.2 531.8,271.7 535.3,282.1 538.9,291.2 542.4,299.0 546.0,305.7 549.6,311.2 553.1,315.7 556.7,319.3 560.2,322.1 563.8,324.3 567.4,325.9 570.9,327.1 574.5,328.0 578.1,328.6 581.6,329.1 585.2,329.4 588.8,329.6 592.3,329.7 595.9,329.8 599.4,329.9 603.0,329.9 606.6,330.0 610.1,330.0 613.7,330.0 617.2,330.0 620.8,330.0 624.4,330.0 627.9,330.0 631.5,330.0 635.1,330.0 638.6,330.0 642.2,330.0 645.8,330.0 649.3,330.0 652.9,330.0 656.4,330.0 660.0,330.0" fill="none" stroke="currentColor" stroke-width="2.6"/>
<polyline points="90.0,329.4 93.6,329.3 97.1,329.2 100.7,329.1 104.2,329.1 107.8,329.0 111.4,328.9 114.9,328.7 118.5,328.6 122.1,328.5 125.6,328.3 129.2,328.2 132.8,328.0 136.3,327.8 139.9,327.7 143.4,327.4 147.0,327.2 150.6,327.0 154.1,326.7 157.7,326.5 161.2,326.2 164.8,325.9 168.4,325.5 171.9,325.2 175.5,324.8 179.1,324.4 182.6,324.0 186.2,323.5 189.8,323.0 193.3,322.5 196.9,322.0 200.4,321.5 204.0,320.9 207.6,320.3 211.1,319.6 214.7,318.9 218.2,318.2 221.8,317.5 225.4,316.7 228.9,315.9 232.5,315.1 236.1,314.2 239.6,313.3 243.2,312.4 246.8,311.4 250.3,310.4 253.9,309.4 257.4,308.3 261.0,307.2 264.6,306.1 268.1,304.9 271.7,303.8 275.2,302.6 278.8,301.4 282.4,300.1 285.9,298.8 289.5,297.6 293.1,296.3 296.6,295.0 300.2,293.6 303.8,292.3 307.3,290.9 310.9,289.6 314.4,288.3 318.0,286.9 321.6,285.6 325.1,284.2 328.7,282.9 332.2,281.6 335.8,280.3 339.4,279.1 342.9,277.8 346.5,276.6 350.1,275.4 353.6,274.3 357.2,273.1 360.8,272.1 364.3,271.0 367.9,270.1 371.4,269.1 375.0,268.2 378.6,267.4 382.1,266.6 385.7,265.9 389.2,265.3 392.8,264.7 396.4,264.2 399.9,263.7 403.5,263.3 407.1,263.0 410.6,262.8 414.2,262.6 417.8,262.5 421.3,262.5 424.9,262.6 428.4,262.7 432.0,262.9 435.6,263.2 439.1,263.5 442.7,263.9 446.2,264.4 449.8,264.9 453.4,265.5 456.9,266.2 460.5,266.9 464.1,267.7 467.6,268.6 471.2,269.5 474.8,270.4 478.3,271.4 481.9,272.5 485.4,273.6 489.0,274.7 492.6,275.9 496.1,277.1 499.7,278.3 503.2,279.6 506.8,280.8 510.4,282.1 513.9,283.4 517.5,284.8 521.1,286.1 524.6,287.5 528.2,288.8 531.8,290.1 535.3,291.5 538.9,292.8 542.4,294.2 546.0,295.5 549.6,296.8 553.1,298.1 556.7,299.4 560.2,300.6 563.8,301.8 567.4,303.1 570.9,304.2 574.5,305.4 578.1,306.5 581.6,307.7 585.2,308.7 588.8,309.8 592.3,310.8 595.9,311.8 599.4,312.7 603.0,313.7 606.6,314.5 610.1,315.4 613.7,316.2 617.2,317.0 620.8,317.8 624.4,318.5 627.9,319.2 631.5,319.9 635.1,320.5 638.6,321.1 642.2,321.7 645.8,322.2 649.3,322.7 652.9,323.2 656.4,323.7 660.0,324.1" fill="none" stroke="#3b82f6" stroke-width="2.2"/>
<polyline points="90.0,330.0 93.6,330.0 97.1,330.0 100.7,330.0 104.2,330.0 107.8,330.0 111.4,330.0 114.9,330.0 118.5,330.0 122.1,330.0 125.6,330.0 129.2,330.0 132.8,330.0 136.3,330.0 139.9,330.0 143.4,330.0 147.0,330.0 150.6,330.0 154.1,330.0 157.7,330.0 161.2,330.0 164.8,330.0 168.4,330.0 171.9,330.0 175.5,330.0 179.1,330.0 182.6,330.0 186.2,330.0 189.8,330.0 193.3,330.0 196.9,330.0 200.4,330.0 204.0,330.0 207.6,330.0 211.1,330.0 214.7,330.0 218.2,330.0 221.8,330.0 225.4,330.0 228.9,330.0 232.5,330.0 236.1,330.0 239.6,330.0 243.2,330.0 246.8,330.0 250.3,330.0 253.9,330.0 257.4,330.0 261.0,330.0 264.6,330.0 268.1,330.0 271.7,330.0 275.2,330.0 278.8,330.0 282.4,330.0 285.9,330.0 289.5,330.0 293.1,330.0 296.6,330.0 300.2,330.0 303.8,330.0 307.3,330.0 310.9,330.0 314.4,330.0 318.0,330.0 321.6,330.0 325.1,330.0 328.7,330.0 332.2,330.0 335.8,330.0 339.4,330.0 342.9,330.0 346.5,330.0 350.1,330.0 353.6,330.0 357.2,330.0 360.8,330.0 364.3,330.0 367.9,330.0 371.4,329.9 375.0,329.9 378.6,329.9 382.1,329.8 385.7,329.6 389.2,329.4 392.8,329.1 396.4,328.7 399.9,328.1 403.5,327.2 407.1,325.9 410.6,324.2 414.2,321.8 417.8,318.7 421.3,314.7 424.9,309.6 428.4,303.2 432.0,295.3 435.6,285.8 439.1,274.5 442.7,261.5 446.2,246.7 449.8,230.3 453.4,212.6 456.9,193.8 460.5,174.4 464.1,155.1 467.6,136.4 471.2,119.0 474.8,103.7 478.3,91.0 481.9,81.4 485.4,75.5 489.0,73.5 492.6,75.5 496.1,81.4 499.7,91.0 503.2,103.7 506.8,119.0 510.4,136.4 513.9,155.1 517.5,174.4 521.1,193.8 524.6,212.6 528.2,230.3 531.8,246.7 535.3,261.5 538.9,274.5 542.4,285.8 546.0,295.3 549.6,303.2 553.1,309.6 556.7,314.7 560.2,318.7 563.8,321.8 567.4,324.2 570.9,325.9 574.5,327.2 578.1,328.1 581.6,328.7 585.2,329.1 588.8,329.4 592.3,329.6 595.9,329.8 599.4,329.9 603.0,329.9 606.6,329.9 610.1,330.0 613.7,330.0 617.2,330.0 620.8,330.0 624.4,330.0 627.9,330.0 631.5,330.0 635.1,330.0 638.6,330.0 642.2,330.0 645.8,330.0 649.3,330.0 652.9,330.0 656.4,330.0 660.0,330.0" fill="none" stroke="#10b981" stroke-width="2.2" stroke-dasharray="7 4"/>
<line x1="120" y1="80" x2="154" y2="80" stroke="currentColor" stroke-width="2.6"/><text x="162" y="85" fill="currentColor" font-size="13">target p: 0.3 N(-4,1) + 0.7 N(4,1)</text>
<line x1="120" y1="102" x2="154" y2="102" stroke="#3b82f6" stroke-width="2.2"/><text x="162" y="107" fill="currentColor" font-size="13">argmin D(p||q): N(1.600, 3.800^2)</text>
<line x1="120" y1="124" x2="154" y2="124" stroke="#10b981" stroke-width="2.2" stroke-dasharray="7 4"/><text x="162" y="129" fill="currentColor" font-size="13">argmin D(q||p): N(4.000, 1.000^2)</text>
<text x="349.3" y="313.3" fill="#3b82f6" font-size="12.5" text-anchor="middle">forward KL keeps mass in the valley</text>
<text x="255.3" y="223.9" fill="#10b981" font-size="12.5" text-anchor="middle">reverse KL abandons</text>
<text x="255.3" y="240.9" fill="#10b981" font-size="12.5" text-anchor="middle">this mode entirely</text>
<text x="360.0" y="400" fill="currentColor" font-size="13.5" text-anchor="middle">Both fits are exact minimisers over the Gaussian family (grid step 1e-4 on [-20,20]).</text>
<text x="360.0" y="419" fill="currentColor" font-size="13.5" text-anchor="middle">Forward KL matches moments: it puts 0.337 of its mass below 0 (truth 0.300) but peaks at</text>
<text x="360.0" y="438" fill="currentColor" font-size="13.5" text-anchor="middle">only 0.105. Reverse KL matches one mode and puts 3.2e-5 below 0. Same family, opposite failure.</text>
</svg>

The ML consequences are concrete and worth naming.

- **VAEs** minimise reverse KL in the latent space and forward KL (through the likelihood term) in the data space. The posterior collapse phenomenon — $q_\phi(z\mid x)$ ignoring $x$ and reverting to the prior — is a mode-seeking solution to the reverse-KL term, exactly the third row of the table.
- **Standard GAN** objectives approximate JS, which is a symmetrised compromise (§11.5), and are usually observed to behave more like reverse KL: sharper, less diverse.
- **Distillation** with the student in the second slot, $D_{\mathrm{KL}}(p_{\text{teacher}}\Vert p_{\text{student}})$, is mass-covering: the student is forced to keep the teacher's "dark knowledge" (small probabilities on wrong classes). Flip the arguments and the student is licensed to discard them. This is a design decision, and papers that flip it without comment are changing the objective, not the implementation.
- **RLHF/policy-regularisation** terms $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\text{ref}})$ are reverse-KL-shaped with respect to the reference: they penalise the new policy for going where the reference does not, which is precisely the intended "stay on-distribution" behaviour. Writing it the other way round would instead force the new policy to cover everything the reference does, a very different constraint.

### 11.3.4 Label smoothing, analysed properly

**Proposition 11.22 (label smoothing).** Fix $K\ge2$ classes, a true class $y$, and $\varepsilon\in[0,1)$. Label smoothing replaces the one-hot target $e_y$ by
$$\tilde p := (1-\varepsilon)e_y + \varepsilon u, \qquad u = \text{uniform on } K \text{ classes},$$
and minimises $H(\tilde p, q_\theta)$. Then:

1. The minimiser over all distributions $q$ is $q = \tilde p$, with minimum value $H(\tilde p)$, so the loss floor is raised from $0$ to $H(\tilde p) > 0$.
2. If $q$ is a softmax over logits $z$, the optimal logit gap between the true class and any other is finite and equal to $\log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$.
3. $H(\tilde p, q) = H(\tilde p) + D_{\mathrm{KL}}(\tilde p\Vert q)$, and $H(\tilde p, q) = (1-\varepsilon)H(e_y,q) + \varepsilon H(u,q)$, so smoothed cross-entropy is a convex combination of the hard loss and a uniform-target loss.

*Proof.* (1) and (3) are Proposition 11.20 plus linearity of $q\mapsto -\sum_k \tilde p_k\log q_k$ in $\tilde p$. For (2), by (1) the optimum has $q_k = \tilde p_k$, and for a softmax $q_k \propto e^{z_k}$, so $z_y - z_k = \log(\tilde p_y/\tilde p_k) = \log\frac{1-\varepsilon+\varepsilon/K}{\varepsilon/K}$. $\blacksquare$

**Worked Example 11.10 (what label smoothing actually costs and buys).** Take $K=1000$ (ImageNet) and the standard $\varepsilon = 0.1$. Then $\tilde p_y = 0.9001$ and $\tilde p_k = 10^{-4}$ for the other $999$ classes. Computed values:

- loss floor $H(\tilde p) = 1.014848$ nats $= 1.464116$ bits (with hard labels the floor is $0$);
- optimal logit gap $= \log(0.9001/0.0001) = 9.105091$ nats;
- for comparison, the loss of a uniform predictor is $\log 1000 = 6.907755$ nats.

Two things follow. First, **a smoothed model's training loss cannot go below $1.0148$ nats**, so comparing raw loss numbers between smoothed and unsmoothed runs is meaningless; you must subtract the respective floors (equivalently, compare the KL terms). Papers that report "our loss is lower" across a smoothing change are reporting an artefact.

Second, the mechanism becomes transparent. With hard targets, the cross-entropy minimiser wants $z_y - z_k\to+\infty$: there is no finite optimum, and the logits grow without bound, which is where the notorious overconfidence and poor calibration come from. Label smoothing replaces that divergent target with a finite one, $9.105$ nats. It is not a regulariser in the weight-decay sense — it changes the *target distribution*, hence changes which $q$ is optimal, hence changes the argmin, not just the path to it. Whether that is desirable depends on whether you want calibrated probabilities (usually yes) or maximum-margin logits for downstream retrieval (sometimes no — label smoothing is known to hurt when the penultimate-layer geometry is used for transfer, because it collapses within-class variation).

> **Common Confusion 11.4.** *"If minimising cross-entropy is the same as minimising KL, why does every framework implement cross-entropy instead of KL? Wouldn't KL be more principled?"*
>
> They differ by $H(P)$, which does not depend on the parameters, so their gradients are *identical* and their argmins coincide. Cross-entropy is implemented because it needs only $\log q_\theta$ evaluated at the observed labels — you never have to compute $H(\hat P_n)$, which for one-hot targets is $0$ anyway and for continuous data is infinite (Proposition 11.13). Choosing cross-entropy is a numerical convenience with zero conceptual cost.
>
> But the difference is not always zero, and here is where it bites: the *loss value* differs by $H(P)$, so loss values are only comparable when the target distributions have the same entropy. Hard labels: floor $0$. Smoothed labels with $\varepsilon = 0.1$, $K=1000$: floor $1.0148$ nats. Distillation with a temperature-$T$ teacher: floor $H(p_{\text{teacher}}^{(T)})$, which varies with $T$. Three papers reporting "cross-entropy loss" can have three different zeros. When comparing across such changes, report the KL, or report the floor alongside the loss.

### 11.3.5 Near a parameter, KL is a quadratic form: the Fisher connection

Everything in this section so far has treated $D_{\mathrm{KL}}$ globally — two arbitrary distributions, any distance apart. But in machine learning the two distributions being compared are very often *neighbours in the same parametric family*: the policy before and after one gradient step, the fine-tuned language model and the reference model it is regularised towards, $P_\theta$ and $P_{\theta+\delta}$ for a small update $\delta$. In that regime KL simplifies dramatically, and the simplification is where Chapter 8's Fisher information $I(\theta)$ — until now a quantity about *estimation* — re-enters as a quantity about *geometry*.

**Proposition 11.23 (KL is locally the Fisher quadratic form).** Let $\{P_\theta:\theta\in\Theta\}$, with $\Theta\subseteq\mathbb{R}^k$ open, have densities $p_\theta>0$ $\mu$-a.e. with respect to a common $\sigma$-finite base measure $\mu$. Fix $\theta\in\Theta$ and assume: (i) for $\mu$-a.e. $x$, the map $\theta'\mapsto\log p_{\theta'}(x)$ is twice continuously differentiable on a neighbourhood of $\theta$; (ii) first and second $\theta$-derivatives may be passed under the integral sign on that neighbourhood, because an integrable envelope exists (Dominated Convergence, Ch. 2); (iii) the Fisher information $I(\theta) := \mathbb{E}_\theta\bigl[\nabla\log p_\theta(X)\,\nabla\log p_\theta(X)^\top\bigr]$ is finite and the information equality $\mathbb{E}_\theta[\nabla^2\log p_\theta(X)] = -I(\theta)$ holds (Ch. 8). Then, as $\delta\to0$,
$$D_{\mathrm{KL}}\bigl(P_\theta\Vert P_{\theta+\delta}\bigr) = \tfrac12\,\delta^\top I(\theta)\,\delta + o(\lVert\delta\rVert^2)\qquad\text{and}\qquad D_{\mathrm{KL}}\bigl(P_{\theta+\delta}\Vert P_\theta\bigr) = \tfrac12\,\delta^\top I(\theta)\,\delta + o(\lVert\delta\rVert^2).$$
In particular the asymmetry exhibited in Proposition 11.18 is a *third-order* effect: forward and reverse KL agree through second order in $\delta$.

*Proof.* **Forward direction.** Write $\psi(\delta) := \int p_\theta\log p_{\theta+\delta}\,\mathrm{d}\mu$, so that $D_{\mathrm{KL}}(P_\theta\Vert P_{\theta+\delta}) = \psi(0)-\psi(\delta)$. By (i)–(ii),
$$\nabla\psi(0) = \int p_\theta\,\nabla\log p_\theta\,\mathrm{d}\mu = \int \nabla p_\theta\,\mathrm{d}\mu = \nabla_\theta\!\int p_\theta\,\mathrm{d}\mu = \nabla_\theta(1) = 0:$$
the score has mean zero, for the always-charming reason that total probability cannot depend on the parameter (Ch. 8). Differentiating once more under the integral, $\nabla^2\psi(0) = \int p_\theta\,\nabla^2\log p_\theta\,\mathrm{d}\mu = -I(\theta)$ by the information equality. Taylor's theorem with the Peano remainder — legitimate because $\psi$ is $C^2$ near $0$ by (i)–(ii) — gives $\psi(\delta) = \psi(0) - \tfrac12\delta^\top I(\theta)\delta + o(\lVert\delta\rVert^2)$, which is the first display.

**Reverse direction.** Write $\chi(\delta) := D_{\mathrm{KL}}(P_{\theta+\delta}\Vert P_\theta) = \int p_{\theta+\delta}\bigl(\log p_{\theta+\delta}-\log p_\theta\bigr)\mathrm{d}\mu$, and use $\nabla p = p\,\nabla\log p$ throughout. Then
$$\nabla\chi(\delta) = \int \nabla p_{\theta+\delta}\,\bigl(\log p_{\theta+\delta}-\log p_\theta\bigr)\,\mathrm{d}\mu \;+\; \underbrace{\int \nabla p_{\theta+\delta}\,\mathrm{d}\mu}_{=\ \nabla(1)\ =\ 0},$$
and at $\delta = 0$ the surviving integral also vanishes, because the log-ratio is identically zero there; so $\nabla\chi(0)=0$. Differentiating the surviving term once more and evaluating at $0$,
$$\nabla^2\chi(0) = \underbrace{\int \nabla^2 p_\theta\cdot 0\;\mathrm{d}\mu}_{=\,0} \;+\; \int \nabla p_\theta\,\nabla\log p_\theta^\top\,\mathrm{d}\mu = \int p_\theta\,\nabla\log p_\theta\,\nabla\log p_\theta^\top\,\mathrm{d}\mu = I(\theta),$$
and Taylor finishes as before. (Exercise A15 asks you to redo this direction with the hypothesis bookkeeping made explicit.) $\blacksquare$

Two sanity checks, one exact and one numerical. For the Gaussian location family $P_\theta = N(\theta,\sigma^2)$ the expansion is exact with no remainder at all: $D_{\mathrm{KL}}(N(0,\sigma^2)\Vert N(\delta,\sigma^2)) = \delta^2/2\sigma^2 = \tfrac12\delta^2 I(\theta)$, since $I(\theta)=1/\sigma^2$ — this is the family Worked Example 11.15 computes with, which is why its true KLs were quadratic in the shift. For the Bernoulli family at $p=0.3$, where $I(p) = 1/\bigl(p(1-p)\bigr) = 4.761905$, computed values:

| $\delta$ | $D_{\mathrm{KL}}(P_p\Vert P_{p+\delta})$ | $D_{\mathrm{KL}}(P_{p+\delta}\Vert P_p)$ | $\tfrac12\delta^2 I(p)$ | fwd$/$quad | rev$/$quad |
|---|---|---|---|---|---|
| $0.1$ | $0.021601$ | $0.022582$ | $0.023810$ | $0.9072$ | $0.9485$ |
| $0.05$ | $0.005630$ | $0.005783$ | $0.005952$ | $0.9459$ | $0.9715$ |
| $0.01$ | $0.000235169$ | $0.000236616$ | $0.000238095$ | $0.9877$ | $0.9938$ |
| $0.001$ | $2.37793\times10^{-6}$ | $2.37944\times10^{-6}$ | $2.38095\times10^{-6}$ | $0.9987$ | $0.9994$ |

Both directions converge to the same quadratic, and their *difference* divided by $\delta^3$ tends to a constant ($-1.5117$ for this family at $p=0.3$; at $\delta=0.001$ the finite-$\delta$ value is $-1.5051$, still converging). Exercise A15 derives the cubic coefficients of both directions exactly. The asymmetry that Proposition 11.18 exhibited globally is real, but it lives entirely in the third order and beyond.

Now the machine-learning payoff, which is large.

**Natural gradient.** Ordinary gradient descent answers the question “which update $\delta$ most decreases the loss *per unit Euclidean length of $\delta$*?” But the Euclidean norm on parameters is a choice of coordinates — precisely the thing §11.2 taught you to distrust. Reparameterise the model (probabilities to logits, weights to rescaled weights) and the “same” gradient step becomes a different step. The **natural gradient** asks instead “which $\delta$ most decreases the loss *per unit of KL moved*?”: maximise $-\nabla L^\top\delta$ subject to $\tfrac12\delta^\top I(\theta)\delta\le\varepsilon$, whose solution is $\delta \propto -I(\theta)^{-1}\nabla L$. Because KL is invariant under reparameterisation (Proposition 11.15) while the Euclidean norm is not, natural-gradient updates are, to first order, the *same* update however you parameterise; vanilla updates are not. A hand computation makes it concrete. Fit $\mathrm{Ber}(p)$ towards $\mathrm{Ber}(0.9)$ by minimising $L(p) = D_{\mathrm{KL}}(\mathrm{Ber}(0.9)\Vert\mathrm{Ber}(p))$, starting at $p=0.3$. The vanilla gradient is $\partial L/\partial p = -0.9/p + 0.1/(1-p) = -20/7 = -2.857143$ in the $p$-parameterisation, but $-0.6$ in the logit parameterisation $\theta = \log\frac{p}{1-p}$ (chain rule: $\times\,p(1-p) = \times\,0.21$) — and these are genuinely different updates once mapped to a common space, since the unit-rate logit step corresponds to $\Delta p \approx -0.6\times0.21 = -0.126$, not $-2.857$. The natural-gradient step, by contrast, is $I^{-1}\times$gradient $= 0.21\times(-20/7) = -0.6$ computed in $p$-space, and $(1/0.21)\times(-0.6) = -2.857143$ computed in logit space — which map to each other *exactly* under the change of variables ($\Delta p = \Delta\theta\cdot p(1-p) = -2.857143\times0.21 = -0.6$). One update, two coordinate systems.

**Trust regions: TRPO, PPO, and the RLHF penalty.** Trust-region policy optimisation constrains each policy update by an average KL to the previous policy, $\overline{D}_{\mathrm{KL}}\bigl(\pi_{\theta_{\mathrm{old}}}\Vert\pi_{\theta}\bigr)\le\varepsilon$. By Proposition 11.23, near $\theta_{\mathrm{old}}$ that feasible set is, to second order, the ellipsoid $\tfrac12\,\delta^\top \bar I(\theta_{\mathrm{old}})\,\delta\le\varepsilon$: *round in policy space, ellipsoidal in weight space*, elongated along directions the policy barely responds to and pinched along directions it is sensitive to. This is why the constraint is imposed on distributions rather than on weights: a microscopic weight change can be an enormous distribution change (a logit crossing zero on a common state) and a huge weight change can be a negligible one (rescaling a saturated unit), and $I(\theta)$ is the exchange rate between the two currencies. PPO's clipped objective is a cheap surrogate for the same trust region. The RLHF penalty $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$ of §11.3.3 gets the same local reading: for small deviations the penalty is $\tfrac12\delta^\top I\delta$ *regardless of which direction of KL you wrote down*, and the mode-seeking-versus-mass-covering distinction of §11.3.3 only reawakens once the policies drift far enough apart for third-order terms to matter.

One honest caveat. The expansion is local, and the regularity hypotheses fail exactly where they should make you nervous: at the boundary of the simplex, $I(p) = 1/\bigl(p(1-p)\bigr)\to\infty$ as $p\to0$ or $1$. The Fisher geometry is reporting that near-deterministic distributions are *far* from their neighbours in KL even when they are close in parameter distance — which is precisely why a policy that has collapsed to near-determinism is so hard to move with a KL-constrained update, and why entropy bonuses exist.

> **Self-check 11.3.**
> 1. Compute $D_{\mathrm{KL}}(\mathrm{Ber}(0)\Vert\mathrm{Ber}(0.5))$ and $D_{\mathrm{KL}}(\mathrm{Ber}(0.5)\Vert\mathrm{Ber}(0))$. Which is infinite and why?
> 2. Your classifier's training loss plateaus at $0.35$ nats. Under what circumstance is that evidence of a *perfect* fit?
> 3. You fit a unimodal $q$ to a bimodal $p$ and get a wide, flat $q$. Which direction of KL did you minimise?
> 4. Label smoothing with $\varepsilon = 0.1$ and $K=10$: what is the optimal logit gap, and how does it compare with the $K=1000$ case?

## 11.4 Mutual information

### 11.4.1 One definition, three faces

Chapter 1's Definition 1.23 says $X\perp Y$ means exactly $P_{XY} = P_X\otimes P_Y$. Independence is an *equation between measures*. So the natural way to measure dependence is to measure how far that equation is from holding, and we now have a tool for measuring how far two measures are apart.

**Definition 11.24 (mutual information).** For random elements $X,Y$ on a common probability space with joint law $P_{XY}$ and marginals $P_X, P_Y$,
$$I(X;Y) := D_{\mathrm{KL}}\bigl(P_{XY}\,\Vert\, P_X\otimes P_Y\bigr) \in [0,\infty].$$
For a third variable $Z$, the **conditional mutual information** is
$$I(X;Y\mid Z) := \mathbb{E}_{Z}\Bigl[D_{\mathrm{KL}}\bigl(P_{XY\mid Z}\,\Vert\, P_{X\mid Z}\otimes P_{Y\mid Z}\bigr)\Bigr],$$
the expectation of the KL between the conditional joint and the product of conditionals (well defined via the regular conditional distributions built in Ch. 6).

This definition needs no densities, no discreteness, and no finiteness: it is meaningful for any pair of random elements in any measurable spaces, including a discrete label and a $3072$-dimensional image. That generality is exactly why it, rather than entropy, is the quantity that appears in machine learning theory.

**Proposition 11.25 (three equivalent expressions).** 

1. Always: $I(X;Y) = \mathbb{E}_{X}\bigl[D_{\mathrm{KL}}(P_{Y\mid X}\Vert P_Y)\bigr]$, the average amount by which learning $X$ moves your belief about $Y$.
2. If $X,Y$ are discrete: $I(X;Y) = H(X) + H(Y) - H(X,Y) = H(Y)-H(Y\mid X) = H(X)-H(X\mid Y)$.
3. If $(X,Y)$ has a joint Lebesgue density: $I(X;Y) = h(X)+h(Y)-h(X,Y)$.

*Proof.* (1) Write $\mathrm{d}P_{XY} = \mathrm{d}P_X\,\mathrm{d}P_{Y\mid X}$ (disintegration, Ch. 6) and $\mathrm{d}(P_X\otimes P_Y) = \mathrm{d}P_X\,\mathrm{d}P_Y$. The Radon–Nikodym derivative of the joint w.r.t. the product is then $\mathrm{d}P_{Y\mid X=x}/\mathrm{d}P_Y$ evaluated at $y$, so
$$I(X;Y) = \int\!\!\int \log\frac{\mathrm{d}P_{Y\mid X=x}}{\mathrm{d}P_Y}(y)\,\mathrm{d}P_{Y\mid X=x}(y)\,\mathrm{d}P_X(x) = \int D_{\mathrm{KL}}\bigl(P_{Y\mid X=x}\Vert P_Y\bigr)\mathrm{d}P_X(x).$$
(2) With $\mu = \#\otimes\#$ as base measure, $\mathrm{d}P_{XY}/\mathrm{d}\mu = p(x,y)$ and $\mathrm{d}(P_X\otimes P_Y)/\mathrm{d}\mu = p(x)p(y)$, so
$$I(X;Y) = \sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = \sum p(x,y)\bigl[\log p(x,y) - \log p(x)-\log p(y)\bigr],$$
which is $-H(X,Y)+H(X)+H(Y)$. The other two forms follow from the chain rule (Theorem 11.6).
(3) Identical computation with $\mu = \lambda_{d_1}\otimes\lambda_{d_2}$; note that the *differential* entropies individually depend on coordinates but their combination does not, exactly as Proposition 11.15 predicts, because the Jacobian factors cancel: under $X\mapsto S(X)$, $h(X)$ and $h(X,Y)$ each shift by $\mathbb{E}[\log|\det J_S|]$ and the shifts cancel in the difference. $\blacksquare$

Form (2) is the one drawn as a Venn diagram, and it is worth having the picture — with a health warning attached.

<svg viewBox="0 0 680 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<text x="340.0" y="30" fill="currentColor" font-size="15" text-anchor="middle">Figure 11.3 — The information diagram for two variables</text>
<circle cx="270" cy="185" r="112" fill="#3b82f6" fill-opacity="0.14" stroke="#3b82f6" stroke-width="2"/>
<circle cx="410" cy="185" r="112" fill="#10b981" fill-opacity="0.14" stroke="#10b981" stroke-width="2"/>
<text x="200" y="190" fill="currentColor" font-size="14" text-anchor="middle">H(X|Y)</text>
<text x="480" y="190" fill="currentColor" font-size="14" text-anchor="middle">H(Y|X)</text>
<text x="340.0" y="190" fill="currentColor" font-size="14" text-anchor="middle">I(X;Y)</text>
<text x="190" y="61" fill="#3b82f6" font-size="14" text-anchor="middle">H(X)</text>
<text x="490" y="61" fill="#10b981" font-size="14" text-anchor="middle">H(Y)</text>
<text x="340.0" y="337" fill="currentColor" font-size="13.5" text-anchor="middle">H(X,Y) = whole shaded region = H(X) + H(Y) - I(X;Y)</text>
<text x="340.0" y="405" fill="currentColor" font-size="13.5" text-anchor="middle">Areas add exactly like this for discrete entropies. It is a mnemonic, not a proof: it breaks</text>
<text x="340.0" y="424" fill="currentColor" font-size="13.5" text-anchor="middle">for three or more variables, where the triple intersection can be negative, and for differential</text>
<text x="340.0" y="443" fill="currentColor" font-size="13.5" text-anchor="middle">entropies, where h can be negative while I(X;Y) is still non-negative.</text>
</svg>

**Theorem 11.26 (properties).** For any random elements:

1. **Non-negativity with a sharp equality case:** $I(X;Y)\ge 0$, with equality iff $X\perp Y$.
2. **Symmetry:** $I(X;Y) = I(Y;X)$.
3. **Invariance:** for bimeasurable bijections $S,T$, $I(S(X);T(Y)) = I(X;Y)$.
4. **Chain rule:** $I(X; Y_1,\dots,Y_n) = \sum_{i=1}^n I(X;Y_i \mid Y_1,\dots,Y_{i-1})$.
5. $I(X;X) = H(X)$ for discrete $X$ (and $=+\infty$ for $X$ with a density).

*Proof.* (1) is Theorem 11.17 applied to $P = P_{XY}$, $Q = P_X\otimes P_Y$, plus Definition 1.23. (2) is symmetry of the defining expression under swapping coordinates (the map $(x,y)\mapsto(y,x)$ is a bijection carrying $P_{XY}\mapsto P_{YX}$ and $P_X\otimes P_Y\mapsto P_Y\otimes P_X$; apply Proposition 11.15). (3) is Proposition 11.15. (4) Expand each conditional MI using Proposition 11.25(1) and telescope, or, in the discrete case, use $I(X;Y_{1:n}) = H(Y_{1:n}) - H(Y_{1:n}\mid X)$ and apply Theorem 11.6 to both terms. (5) For discrete $X$, $I(X;X) = H(X)-H(X\mid X) = H(X)$ since $H(X\mid X)=0$. For $X$ with a density, take $Y=X$ in Definition 11.24: $P_{XX}$ is supported on the diagonal, which is $(P_X\otimes P_X)$-null, so $P_{XX}\not\ll P_X\otimes P_X$ and $I = +\infty$. $\blacksquare$

Property (5) is quietly important for deep learning. A deterministic encoder $Z = f(X)$ applied to continuous data has $I(X;Z) = +\infty$ whenever $f$ is injective, and $+\infty$ for a great many non-injective $f$ as well. So the objective "maximise $I(X;Z)$" is vacuous for deterministic encoders on continuous data: it is already maximal, at infinity, before you train anything. This is not a technicality; it is the core of the well-known critique of the information bottleneck story for deterministic networks, and we return to it in §11.10.

> **Common Confusion 11.5.** *"Mutual information zero means uncorrelated, right? So I can use correlation as a cheap proxy."*
>
> The implication runs one way only, and both directions are instructive.
>
> $I(X;Y)=0$ implies independence (Theorem 11.26(1)), which implies zero correlation. So $I=0 \Rightarrow$ correlation $=0$. Good.
>
> The converse fails badly. Let $X\sim N(0,1)$ and $Y = X^2$. Then $\mathrm{Cov}(X,Y) = \mathbb{E}[X^3] = 0$, so the correlation is exactly zero, yet $I(X;Y) = +\infty$. Here is why: $P_{XY}$ is supported on the parabola $\{(x,x^2): x\in\mathbb{R}\}$, which is a Lebesgue-null subset of $\mathbb{R}^2$ and therefore also $(P_X\otimes P_Y)$-null, since $P_X\otimes P_Y\ll\lambda_2$. So $P_{XY}\not\ll P_X\otimes P_Y$, and Definition 11.16 gives $+\infty$. Zero correlation, infinite mutual information, from a relationship a five-year-old could draw.
>
> The reverse trap is subtler and is the one that catches practitioners: **a large mutual information does not mean a large correlation, and a large correlation does not mean the relationship is usable.** For a bivariate Gaussian the two are in bijection, $I = -\tfrac12\log(1-\rho^2)$, and this special case is where the folklore comes from. Outside jointly Gaussian data, $I$ and $\rho$ measure different things: $I$ is invariant under any monotone (indeed any bijective) transformation of either variable, while $\rho$ — the correlation coefficient of Chapter 5 — is only invariant under affine maps. Which you want depends on the question. If you are asking "is there any dependence at all", use $I$ (and read §11.5 on how hard it is to estimate). If you are asking "does a linear predictor work", $\rho$ is the right answer and $I$ is not.

### 11.4.2 Gaussian and channel computations

**Proposition 11.27 (Gaussian mutual information).** Let $(X,Y)$ be jointly Gaussian in $\mathbb{R}^{d_1}\times\mathbb{R}^{d_2}$ with covariance blocks $\Sigma_{XX},\Sigma_{YY},\Sigma_{XY}$ and joint covariance $\Sigma$, all non-singular. Then
$$I(X;Y) = \tfrac12\log\frac{\det\Sigma_{XX}\,\det\Sigma_{YY}}{\det\Sigma}.$$
In the scalar case with correlation $\rho$, $I(X;Y) = -\tfrac12\log(1-\rho^2)$.

*Proof.* Apply Proposition 11.25(3) with Proposition 11.11:
$$I = \tfrac12\log\bigl((2\pi e)^{d_1}\det\Sigma_{XX}\bigr) + \tfrac12\log\bigl((2\pi e)^{d_2}\det\Sigma_{YY}\bigr) - \tfrac12\log\bigl((2\pi e)^{d_1+d_2}\det\Sigma\bigr),$$
and the $(2\pi e)$ powers cancel. Scalar case: $\det\Sigma = \sigma_X^2\sigma_Y^2(1-\rho^2)$. $\blacksquare$

Note the cancellation of $(2\pi e)^{d}$: those are precisely the coordinate-dependent parts, and they must cancel by Proposition 11.15. Watching them cancel is a good way to internalise which quantities are physical.

**Worked Example 11.11 (two channels, computed).** *(a) Bivariate Gaussian.* Values of $I(X;Y) = -\tfrac12\log(1-\rho^2)$:

| $\rho$ | $I$ (nats) | $I$ (bits) |
|---|---|---|
| $0.3$ | $0.047155$ | $0.068031$ |
| $0.5$ | $0.143841$ | $0.207519$ |
| $0.9$ | $0.830366$ | $1.197964$ |
| $0.99$ | $1.958518$ | $2.825544$ |

Read the middle row: a correlation of $0.5$, which most people would call a substantial relationship, carries barely one fifth of a bit. Mutual information is brutally unimpressed by moderate correlation, and this recalibrates the intuition needed to read papers reporting MI in the range of a few bits. Note also that $I\to\infty$ as $\rho\to1$: perfect dependence between continuous variables is infinite information, consistent with Theorem 11.26(5).

*(b) Binary symmetric channel.* $X\in\{0,1\}$ is sent, and $Y = X\oplus B$ where $B\sim\mathrm{Ber}(\epsilon)$ independently. With $X$ uniform,
$$I(X;Y) = H(Y)-H(Y\mid X) = 1 - H_2(\epsilon)\ \text{bits}.$$
Computed: $\epsilon = 0.01$ gives $0.919207$; $\epsilon = 0.1$ gives $0.531004$; $\epsilon=0.5$ gives exactly $0$ — a channel that flips fairly transmits nothing, which matches the intuition that its output is independent of its input. With a non-uniform input ($\epsilon = 0.1$ fixed): $P(X=1)=0.5$ gives $0.531004$ bits, $P(X=1)=0.3$ gives $0.455823$, $P(X=1)=0.1$ gives $0.211081$. The uniform input is optimal here, illustrating both the concavity of $I$ in the input distribution (Proposition 11.8's consequence) and the symmetry of the channel.

### 11.4.3 The data processing inequality

**Theorem 11.28 (data processing inequality for mutual information).** Suppose $X \to Y \to Z$ is a Markov chain, i.e. $X \perp Z \mid Y$ (Definition 1.24). Then
$$I(X;Z) \le I(X;Y),$$
with equality iff $X\perp Y\mid Z$. In particular, for any measurable $g$ and $Z = g(Y)$ (possibly randomised, i.e. $Z$ generated from $Y$ by a kernel independent of $X$), no processing of $Y$ can increase its information about $X$.

*Proof.* Apply the chain rule (Theorem 11.26(4)) to $I(X;(Y,Z))$ in the two possible orders:
$$I(X;Y,Z) = I(X;Y) + I(X;Z\mid Y) = I(X;Z) + I(X;Y\mid Z).$$
The Markov assumption $X\perp Z\mid Y$ says exactly $I(X;Z\mid Y) = 0$ (Definition 11.24's conditional form is a KL, and it is zero iff the conditional joint factorises, which is conditional independence). Hence
$$I(X;Y) = I(X;Z) + I(X;Y\mid Z) \ \ge\ I(X;Z),$$
since conditional mutual information is an average of KL divergences, hence non-negative. Equality holds iff $I(X;Y\mid Z)=0$, i.e. $X\perp Y\mid Z$. $\blacksquare$

The DPI is the most quoted inequality in representation-learning papers and the most misquoted. What it says: information about $X$ cannot be created by processing something downstream of $X$. What it does not say: that deep networks are useless, or that layers must lose information. Take a moment on this, because it is a genuine source of confusion.

For a chain $X\to Z_1\to Z_2\to\cdots\to Z_L$ (each layer a function of the previous), the DPI gives $I(X;Y)\ge I(Z_1;Y)\ge\cdots\ge I(Z_L;Y)$ for a label $Y$ *only if the chain is $Y\to X\to Z_1\to\cdots$*, which it is: the label generates the data, and layers process the data. So the information about the label is non-increasing with depth. Layers therefore cannot increase the *amount* of label information; what they can do is make that information *accessible to a simple decoder*. Depth buys computational accessibility, not information. Once you say it that way, the DPI stops sounding like a paradox and starts sounding like a definition of what representation learning is for.

**Worked Example 11.12 (binning destroys information, quantitatively).** Take $(X,Y)$ bivariate Gaussian with $\rho = 0.9$, so $I(X;Y) = 1.197964$ bits exactly. Now discretise both coordinates into $m$ equal-probability bins — a deterministic function of each variable, so the DPI applies and the binned MI must be no larger. Estimated from $n = 4{,}000{,}000$ samples (seed $0$), so that estimation error is negligible relative to the effect:

| bins $m$ per axis | $I(\text{binned})$ (bits) | fraction of the truth |
|---|---|---|
| $2$ | $0.4061$ | $34\%$ |
| $4$ | $0.7786$ | $65\%$ |
| $8$ | $1.0057$ | $84\%$ |
| $16$ | $1.1135$ | $93\%$ |
| $32$ | $1.1613$ | $97\%$ |

For $m = 2$ the value is available in closed form: $P(X>0,Y>0) = \tfrac14 + \tfrac{\arcsin\rho}{2\pi} = 0.428217$, giving $I = 0.406497$ bits, matching the simulation to three decimals. Two lessons. First, the DPI is not a loose bound here — binning at coarse resolution throws away most of the information, and thresholding at the median (the $m=2$ row) keeps only a third. Second, the convergence to the truth is slow, $1-O(1/m^2)$-ish, which is precisely why histogram-based mutual information estimators need many bins, and many bins need enormous samples. §11.5 quantifies that trade-off with real estimator runs.

**Worked Example 11.13 (Chapter 1's two dice: what a sum forgets).** Roll two fair dice, $D_1,D_2$ independent and uniform on $\{1,\dots,6\}$, and let $S := D_1+D_2$ — Chapter 1's running example. Everything below is exact arithmetic on the pmf $P_S = (1,2,3,4,5,6,5,4,3,2,1)/36$.

$$H(D_1) = \log_2 6 = 2.584963,\quad H(D_1,D_2) = 2\log_2 6 = 5.169925,\quad H(S) = 3.274402\ \text{bits}.$$

Four readings, each of a result from this section.

*A function loses entropy.* $S = g(D_1,D_2)$ is not injective, and indeed $H(S) = 3.274402 < 5.169925 = H(D_1,D_2)$, as Proposition 11.7(3) requires. The gap $H(D_1,D_2\mid S) = 1.895523$ bits is exactly what the sum forgets: told that $S=7$, you still do not know which of six ordered pairs occurred.

*Mutual information between a part and the whole.* Given $D_1$, the map $D_2\mapsto S$ is a bijection, so $H(S\mid D_1) = H(D_2) = \log_2 6$, and
$$I(D_1;S) = H(S)-H(S\mid D_1) = 3.274402-2.584963 = 0.689439\ \text{bits}.$$
So one die tells you $0.689$ bits about the sum — about $21\%$ of the sum's $3.274$ bits.

*The DPI, with numbers.* The chain $D_1\to(D_1,D_2)\to S$ is Markov, and $I(D_1;(D_1,D_2)) = H(D_1) = 2.584963$ bits. Theorem 11.28 demands $I(D_1;S)\le2.584963$, and $0.689439$ obeys it with a lot of room to spare. The lost $1.895524$ bits are exactly $I(D_1;D_2\mid S)$ — the deficit identified in the chain-rule proof.

*Conditioning can create dependence.* $D_1$ and $D_2$ are independent, so $I(D_1;D_2) = 0$. But conditionally on $S$, knowing $D_2$ determines $D_1$, so $H(D_1\mid D_2,S) = 0$ and
$$I(D_1;D_2\mid S) = H(D_1\mid S) = 1.895523\ \text{bits}\ >\ 0 = I(D_1;D_2).$$
Conditioning on a common consequence made two independent variables dependent. This is the same phenomenon as the XOR example in Exercise A7, in a setting you have rolled by hand — and it is worth pausing on, because a great deal of informal reasoning about "controlling for" a variable assumes the opposite.

**A practical caveat on estimating any of these.** Every number above came from the exact pmf. If instead you estimate a discrete entropy by plugging in observed frequencies, $\hat H = -\sum_x \hat p(x)\log\hat p(x)$, the estimate is biased *downward*, by approximately $(K-1)/(2n)$ nats where $K$ is the number of outcomes with positive probability and $n$ the sample size. Checked by simulation on a fair die ($K = 6$, $20{,}000$ replicates): at $n = 100$ the mean plug-in entropy is $1.766504$ against the true $\log 6 = 1.791759$ nats, a bias of $-0.025256$ against the predicted $-0.025$; at $n = 1000$, $-0.002496$ against $-0.0025$. The approximation degrades when $K$ is comparable to $n$: for $K = 256$ uniform outcomes at $n = 1000$ the measured bias is $-0.135899$ against a predicted $-0.1275$. The consequence for mutual information is worse rather than better, because $\hat I = \hat H(X)+\hat H(Y)-\hat H(X,Y)$ combines three biased terms and the joint alphabet is the largest, so the biases do not cancel — they add up in the direction of *over*-stating dependence. Bias-corrected estimators (Miller–Madow and its descendants) subtract the leading term; §11.5 takes up the harder continuous case.

> **Self-check 11.4.**
> 1. Show $I(X;Y)\le \min\{H(X),H(Y)\}$ for discrete variables. Where does the argument fail for continuous variables?
> 2. $Z = f(X)$ deterministic, $X$ continuous with a density, $f$ injective. What is $I(X;Z)$? Why is "maximise $I(X;Z)$" a bad objective here?
> 3. A network computes $Z_1 \to Z_2$. Someone claims layer 2 "extracts more information about the label." What is the charitable reading, and what is the DPI-based objection?
> 4. Compute $I(X;Y)$ in bits for a bivariate Gaussian with $\rho = 0.7$.

## 11.5 $f$-divergences and variational representations

### 11.5.1 One formula, six familiar divergences

Every divergence used in generative modelling except the transport distances is an instance of a single construction.

**Definition 11.29 ($f$-divergence).** Let $f:(0,\infty)\to\mathbb{R}$ be convex with $f(1)=0$, extended by $f(0) := \lim_{t\downarrow0}f(t)$ and with $f'(\infty) := \lim_{t\to\infty}f(t)/t$. For probability measures $P,Q$ on a common space, choose any $\mu$ with $P,Q\ll\mu$ (e.g. $\mu = P+Q$), write $p,q$ for the densities, and define
$$D_f(P\Vert Q) := \int_{\{q>0\}} q\, f\!\left(\frac{p}{q}\right)\mathrm{d}\mu \;+\; f'(\infty)\,P(\{q = 0\}).$$
The value does not depend on the choice of $\mu$.

The final term handles the case where $P$ escapes the support of $Q$: it charges $f'(\infty)$ per unit of escaped mass, which is $+\infty$ for KL ($f'(\infty)=\infty$) and finite for TV ($f'(\infty)=1/2$). Getting that term right is what makes statements like "TV is bounded by $1$ even for mutually singular measures" true rather than a special case.

Here is the family, with generators normalised so that $f \ge 0$ (which one may always do by replacing $f(t)$ with $f(t) - c(t-1)$; this changes nothing, since $\int q\,(p/q - 1)\,\mathrm{d}\mu = 0$).

| Divergence | Generator $f(t)$ | $f'(\infty)$ | Range | Symmetric? |
|---|---|---|---|---|
| KL $D_{\mathrm{KL}}(P\Vert Q)$ | $t\log t - t + 1$ | $\infty$ | $[0,\infty]$ | no |
| reverse KL $D_{\mathrm{KL}}(Q\Vert P)$ | $-\log t + t - 1$ | $1$ | $[0,\infty]$ | no |
| Jensen–Shannon | $\tfrac12\bigl[t\log\frac{2t}{t+1} + \log\frac{2}{t+1}\bigr]$ | $\tfrac12\log 2$ | $[0,\log 2]$ | yes |
| $\chi^2$ | $(t-1)^2$ | $\infty$ | $[0,\infty]$ | no |
| total variation | $\tfrac12\lvert t-1\rvert$ | $\tfrac12$ | $[0,1]$ | yes |
| squared Hellinger | $(\sqrt t - 1)^2$ | $1$ | $[0,2]$ | yes |

<svg viewBox="0 0 720 470" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<text x="360.0" y="30" fill="currentColor" font-size="15" text-anchor="middle">Figure 11.4 — Generators f of the standard f-divergences</text>
<line x1="100" y1="340" x2="600" y2="340" stroke="currentColor" stroke-width="1.2"/>
<line x1="100" y1="340" x2="100" y2="60" stroke="currentColor" stroke-width="1.2"/>
<line x1="100.0" y1="340" x2="100.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="100.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0</text>
<line x1="258.7" y1="340" x2="258.7" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="258.7" y="362" fill="currentColor" font-size="13" text-anchor="middle">1</text>
<line x1="417.5" y1="340" x2="417.5" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="417.5" y="362" fill="currentColor" font-size="13" text-anchor="middle">2</text>
<line x1="576.2" y1="340" x2="576.2" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="576.2" y="362" fill="currentColor" font-size="13" text-anchor="middle">3</text>
<line x1="94" y1="340.0" x2="100" y2="340.0" stroke="currentColor" stroke-width="1"/>
<text x="90" y="344.0" fill="currentColor" font-size="13" text-anchor="end">0</text>
<line x1="94" y1="232.3" x2="100" y2="232.3" stroke="currentColor" stroke-width="1"/>
<text x="90" y="236.3" fill="currentColor" font-size="13" text-anchor="end">1</text>
<line x1="94" y1="124.6" x2="100" y2="124.6" stroke="currentColor" stroke-width="1"/>
<text x="90" y="128.6" fill="currentColor" font-size="13" text-anchor="end">2</text>
<text x="350.0" y="385" fill="currentColor" font-size="14" text-anchor="middle">t = p(x)/q(x)   (density ratio)</text>
<text x="58" y="200.0" fill="currentColor" font-size="14" text-anchor="middle" transform="rotate(-90 58 200.0)">f(t)</text>
<polyline points="104.8,246.9 108.9,255.7 113.0,263.1 117.0,269.7 121.1,275.6 125.2,280.9 129.3,285.8 133.4,290.3 137.5,294.5 141.6,298.3 145.7,301.9 149.8,305.3 153.9,308.4 158.0,311.3 162.1,314.0 166.2,316.5 170.3,318.8 174.4,321.0 178.5,323.1 182.6,324.9 186.7,326.7 190.8,328.3 194.9,329.8 198.9,331.2 203.0,332.4 207.1,333.6 211.2,334.6 215.3,335.5 219.4,336.4 223.5,337.1 227.6,337.8 231.7,338.3 235.8,338.8 239.9,339.2 244.0,339.5 248.1,339.8 252.2,339.9 256.3,340.0 260.4,340.0 264.5,339.9 268.6,339.8 272.7,339.6 276.8,339.3 280.8,339.0 284.9,338.6 289.0,338.2 293.1,337.6 297.2,337.1 301.3,336.4 305.4,335.7 309.5,335.0 313.6,334.2 317.7,333.3 321.8,332.4 325.9,331.5 330.0,330.5 334.1,329.4 338.2,328.3 342.3,327.2 346.4,326.0 350.5,324.7 354.6,323.4 358.7,322.1 362.7,320.7 366.8,319.3 370.9,317.8 375.0,316.3 379.1,314.8 383.2,313.2 387.3,311.6 391.4,309.9 395.5,308.2 399.6,306.5 403.7,304.7 407.8,302.8 411.9,301.0 416.0,299.1 420.1,297.2 424.2,295.2 428.3,293.2 432.4,291.2 436.5,289.1 440.5,287.0 444.6,284.8 448.7,282.7 452.8,280.5 456.9,278.2 461.0,276.0 465.1,273.7 469.2,271.3 473.3,269.0 477.4,266.6 481.5,264.2 485.6,261.7 489.7,259.2 493.8,256.7 497.9,254.2 502.0,251.6 506.1,249.0 510.2,246.4 514.3,243.8 518.4,241.1 522.4,238.4 526.5,235.6 530.6,232.9 534.7,230.1 538.8,227.3 542.9,224.4 547.0,221.6 551.1,218.7 555.2,215.8 559.3,212.8 563.4,209.9 567.5,206.9 571.6,203.9 575.7,200.8 579.8,197.8 583.9,194.7 588.0,191.6 592.1,188.4" fill="none" stroke="#3b82f6" stroke-width="2.2" stroke-dasharray="none"/>
<polyline points="104.8,66.8 108.9,130.9 113.0,169.0 117.0,195.8 121.1,216.2 125.2,232.5 129.3,245.9 133.4,257.2 137.5,266.9 141.6,275.3 145.7,282.6 149.8,289.1 153.9,294.8 158.0,299.9 162.1,304.5 166.2,308.6 170.3,312.3 174.4,315.6 178.5,318.6 182.6,321.3 186.7,323.7 190.8,325.9 194.9,327.9 198.9,329.7 203.0,331.3 207.1,332.7 211.2,333.9 215.3,335.0 219.4,336.0 223.5,336.9 227.6,337.6 231.7,338.2 235.8,338.8 239.9,339.2 244.0,339.5 248.1,339.7 252.2,339.9 256.3,340.0 260.4,340.0 264.5,339.9 268.6,339.8 272.7,339.6 276.8,339.4 280.8,339.0 284.9,338.7 289.0,338.3 293.1,337.8 297.2,337.3 301.3,336.7 305.4,336.1 309.5,335.4 313.6,334.7 317.7,334.0 321.8,333.2 325.9,332.4 330.0,331.6 334.1,330.7 338.2,329.8 342.3,328.9 346.4,327.9 350.5,326.9 354.6,325.9 358.7,324.8 362.7,323.7 366.8,322.6 370.9,321.5 375.0,320.3 379.1,319.1 383.2,317.9 387.3,316.7 391.4,315.4 395.5,314.1 399.6,312.8 403.7,311.5 407.8,310.2 411.9,308.8 416.0,307.5 420.1,306.1 424.2,304.7 428.3,303.2 432.4,301.8 436.5,300.3 440.5,298.8 444.6,297.4 448.7,295.9 452.8,294.3 456.9,292.8 461.0,291.2 465.1,289.7 469.2,288.1 473.3,286.5 477.4,284.9 481.5,283.3 485.6,281.7 489.7,280.0 493.8,278.4 497.9,276.7 502.0,275.0 506.1,273.3 510.2,271.6 514.3,269.9 518.4,268.2 522.4,266.5 526.5,264.8 530.6,263.0 534.7,261.2 538.8,259.5 542.9,257.7 547.0,255.9 551.1,254.1 555.2,252.3 559.3,250.5 563.4,248.7 567.5,246.8 571.6,245.0 575.7,243.2 579.8,241.3 583.9,239.4 588.0,237.6 592.1,235.7" fill="none" stroke="#10b981" stroke-width="2.2" stroke-dasharray="none"/>
<polyline points="104.8,238.7 108.9,244.0 113.0,249.2 117.0,254.2 121.1,259.1 125.2,263.8 129.3,268.4 133.4,272.9 137.5,277.2 141.6,281.4 145.7,285.4 149.8,289.3 153.9,293.0 158.0,296.6 162.1,300.1 166.2,303.4 170.3,306.6 174.4,309.6 178.5,312.5 182.6,315.2 186.7,317.8 190.8,320.3 194.9,322.6 198.9,324.7 203.0,326.7 207.1,328.6 211.2,330.4 215.3,331.9 219.4,333.4 223.5,334.7 227.6,335.9 231.7,336.9 235.8,337.8 239.9,338.5 244.0,339.1 248.1,339.5 252.2,339.8 256.3,340.0 260.4,340.0 264.5,339.9 268.6,339.6 272.7,339.2 276.8,338.6 280.8,337.9 284.9,337.1 289.0,336.1 293.1,334.9 297.2,333.7 301.3,332.2 305.4,330.7 309.5,329.0 313.6,327.1 317.7,325.1 321.8,323.0 325.9,320.7 330.0,318.3 334.1,315.7 338.2,313.0 342.3,310.2 346.4,307.2 350.5,304.0 354.6,300.8 358.7,297.3 362.7,293.8 366.8,290.0 370.9,286.2 375.0,282.2 379.1,278.0 383.2,273.8 387.3,269.3 391.4,264.8 395.5,260.0 399.6,255.2 403.7,250.2 407.8,245.0 411.9,239.7 416.0,234.3 420.1,228.7 424.2,223.0 428.3,217.1 432.4,211.1 436.5,205.0 440.5,198.7 444.6,192.3 448.7,185.7 452.8,179.0 456.9,172.1 461.0,165.1 465.1,157.9 469.2,150.6 473.3,143.2 477.4,135.6 481.5,127.9 485.6,120.0 489.7,112.0 493.8,103.8 497.9,95.5 502.0,87.1 506.1,78.5 510.2,69.8" fill="none" stroke="#ef4444" stroke-width="2.2" stroke-dasharray="none"/>
<polyline points="104.8,266.4 108.9,277.2 113.0,285.0 117.0,291.3 121.1,296.6 125.2,301.1 129.3,305.0 133.4,308.5 137.5,311.6 141.6,314.4 145.7,316.9 149.8,319.2 153.9,321.2 158.0,323.2 162.1,324.9 166.2,326.5 170.3,327.9 174.4,329.3 178.5,330.5 182.6,331.6 186.7,332.7 190.8,333.6 194.9,334.5 198.9,335.2 203.0,335.9 207.1,336.6 211.2,337.1 215.3,337.7 219.4,338.1 223.5,338.5 227.6,338.8 231.7,339.1 235.8,339.4 239.9,339.6 244.0,339.8 248.1,339.9 252.2,340.0 256.3,340.0 260.4,340.0 264.5,340.0 268.6,339.9 272.7,339.8 276.8,339.7 280.8,339.5 284.9,339.3 289.0,339.1 293.1,338.9 297.2,338.6 301.3,338.3 305.4,338.0 309.5,337.6 313.6,337.2 317.7,336.8 321.8,336.4 325.9,336.0 330.0,335.5 334.1,335.1 338.2,334.6 342.3,334.0 346.4,333.5 350.5,332.9 354.6,332.4 358.7,331.8 362.7,331.2 366.8,330.5 370.9,329.9 375.0,329.2 379.1,328.5 383.2,327.9 387.3,327.2 391.4,326.4 395.5,325.7 399.6,324.9 403.7,324.2 407.8,323.4 411.9,322.6 416.0,321.8 420.1,321.0 424.2,320.2 428.3,319.3 432.4,318.5 436.5,317.6 440.5,316.7 444.6,315.9 448.7,315.0 452.8,314.0 456.9,313.1 461.0,312.2 465.1,311.3 469.2,310.3 473.3,309.3 477.4,308.4 481.5,307.4 485.6,306.4 489.7,305.4 493.8,304.4 497.9,303.4 502.0,302.3 506.1,301.3 510.2,300.3 514.3,299.2 518.4,298.1 522.4,297.1 526.5,296.0 530.6,294.9 534.7,293.8 538.8,292.7 542.9,291.6 547.0,290.5 551.1,289.3 555.2,288.2 559.3,287.1 563.4,285.9 567.5,284.8 571.6,283.6 575.7,282.4 579.8,281.3 583.9,280.1 588.0,278.9 592.1,277.7" fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="none"/>
<polyline points="104.8,277.7 108.9,284.7 113.0,290.4 117.0,295.3 121.1,299.6 125.2,303.4 129.3,306.8 133.4,309.9 137.5,312.7 141.6,315.2 145.7,317.6 149.8,319.7 153.9,321.7 158.0,323.5 162.1,325.2 166.2,326.7 170.3,328.1 174.4,329.4 178.5,330.6 182.6,331.7 186.7,332.7 190.8,333.6 194.9,334.5 198.9,335.3 203.0,336.0 207.1,336.6 211.2,337.2 215.3,337.7 219.4,338.1 223.5,338.5 227.6,338.9 231.7,339.1 235.8,339.4 239.9,339.6 244.0,339.8 248.1,339.9 252.2,340.0 256.3,340.0 260.4,340.0 264.5,340.0 268.6,339.9 272.7,339.8 276.8,339.7 280.8,339.5 284.9,339.3 289.0,339.1 293.1,338.9 297.2,338.6 301.3,338.3 305.4,338.0 309.5,337.6 313.6,337.2 317.7,336.9 321.8,336.4 325.9,336.0 330.0,335.5 334.1,335.1 338.2,334.6 342.3,334.1 346.4,333.5 350.5,333.0 354.6,332.4 358.7,331.8 362.7,331.2 366.8,330.6 370.9,329.9 375.0,329.3 379.1,328.6 383.2,327.9 387.3,327.2 391.4,326.5 395.5,325.8 399.6,325.1 403.7,324.3 407.8,323.6 411.9,322.8 416.0,322.0 420.1,321.2 424.2,320.4 428.3,319.6 432.4,318.7 436.5,317.9 440.5,317.0 444.6,316.1 448.7,315.3 452.8,314.4 456.9,313.5 461.0,312.6 465.1,311.7 469.2,310.7 473.3,309.8 477.4,308.8 481.5,307.9 485.6,306.9 489.7,306.0 493.8,305.0 497.9,304.0 502.0,303.0 506.1,302.0 510.2,301.0 514.3,299.9 518.4,298.9 522.4,297.9 526.5,296.8 530.6,295.8 534.7,294.7 538.8,293.7 542.9,292.6 547.0,291.5 551.1,290.4 555.2,289.3 559.3,288.2 563.4,287.1 567.5,286.0 571.6,284.9 575.7,283.8 579.8,282.7 583.9,281.5 588.0,280.4 592.1,279.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="1 0"/>
<polyline points="104.8,287.8 108.9,289.2 113.0,290.5 117.0,291.9 121.1,293.3 125.2,294.7 129.3,296.1 133.4,297.5 137.5,298.9 141.6,300.3 145.7,301.7 149.8,303.0 153.9,304.4 158.0,305.8 162.1,307.2 166.2,308.6 170.3,310.0 174.4,311.4 178.5,312.8 182.6,314.2 186.7,315.6 190.8,316.9 194.9,318.3 198.9,319.7 203.0,321.1 207.1,322.5 211.2,323.9 215.3,325.3 219.4,326.7 223.5,328.1 227.6,329.4 231.7,330.8 235.8,332.2 239.9,333.6 244.0,335.0 248.1,336.4 252.2,337.8 256.3,339.2 260.4,339.4 264.5,338.1 268.6,336.7 272.7,335.3 276.8,333.9 280.8,332.5 284.9,331.1 289.0,329.7 293.1,328.3 297.2,326.9 301.3,325.6 305.4,324.2 309.5,322.8 313.6,321.4 317.7,320.0 321.8,318.6 325.9,317.2 330.0,315.8 334.1,314.4 338.2,313.0 342.3,311.7 346.4,310.3 350.5,308.9 354.6,307.5 358.7,306.1 362.7,304.7 366.8,303.3 370.9,301.9 375.0,300.5 379.1,299.2 383.2,297.8 387.3,296.4 391.4,295.0 395.5,293.6 399.6,292.2 403.7,290.8 407.8,289.4 411.9,288.0 416.0,286.7 420.1,285.3 424.2,283.9 428.3,282.5 432.4,281.1 436.5,279.7 440.5,278.3 444.6,276.9 448.7,275.5 452.8,274.2 456.9,272.8 461.0,271.4 465.1,270.0 469.2,268.6 473.3,267.2 477.4,265.8 481.5,264.4 485.6,263.0 489.7,261.7 493.8,260.3 497.9,258.9 502.0,257.5 506.1,256.1 510.2,254.7 514.3,253.3 518.4,251.9 522.4,250.5 526.5,249.1 530.6,247.8 534.7,246.4 538.8,245.0 542.9,243.6 547.0,242.2 551.1,240.8 555.2,239.4 559.3,238.0 563.4,236.6 567.5,235.3 571.6,233.9 575.7,232.5 579.8,231.1 583.9,229.7 588.0,228.3 592.1,226.9" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="5 4"/>
<line x1="130" y1="82" x2="160" y2="82" stroke="#3b82f6" stroke-width="2.2" stroke-dasharray="none"/><text x="168" y="86" fill="currentColor" font-size="12.5">KL</text>
<line x1="130" y1="102" x2="160" y2="102" stroke="#10b981" stroke-width="2.2" stroke-dasharray="none"/><text x="168" y="106" fill="currentColor" font-size="12.5">reverse KL</text>
<line x1="130" y1="122" x2="160" y2="122" stroke="#ef4444" stroke-width="2.2" stroke-dasharray="none"/><text x="168" y="126" fill="currentColor" font-size="12.5">chi-squared</text>
<line x1="130" y1="142" x2="160" y2="142" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="none"/><text x="168" y="146" fill="currentColor" font-size="12.5">Hellinger^2</text>
<line x1="130" y1="162" x2="160" y2="162" stroke="currentColor" stroke-width="2.2" stroke-dasharray="1 0"/><text x="168" y="166" fill="currentColor" font-size="12.5">2 x JS</text>
<line x1="130" y1="182" x2="160" y2="182" stroke="currentColor" stroke-width="2.2" stroke-dasharray="5 4"/><text x="168" y="186" fill="currentColor" font-size="12.5">TV</text>
<circle cx="258.7" cy="340.0" r="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="258.7" y="374.0" fill="currentColor" font-size="12.5" text-anchor="middle">f(1) = 0</text>
<text x="360.0" y="400" fill="currentColor" font-size="13.5" text-anchor="middle">Each f is convex and normalised so f(1) = 0, with zero attained only at t = 1: that strictness is</text>
<text x="360.0" y="419" fill="currentColor" font-size="13.5" text-anchor="middle">what forces P = Q when D_f = 0. Behaviour as t -> 0 decides the blow-up: reverse KL diverges</text>
<text x="360.0" y="438" fill="currentColor" font-size="13.5" text-anchor="middle">there, while KL, TV and Hellinger stay finite. Growth as t -> infinity governs the other tail.</text>
</svg>

**Proposition 11.30 (non-negativity and identifiability).** $D_f(P\Vert Q)\ge 0$ always. If $f$ is strictly convex at $t=1$ (i.e. $f(t) > f(1) + s(t-1)$ for $t\ne1$ and any subgradient $s$ at $1$), then $D_f(P\Vert Q) = 0$ iff $P = Q$.

*Proof.* Assume first $P\ll Q$, so the escape term vanishes. With $r = \mathrm{d}P/\mathrm{d}Q$, Jensen under $Q$ gives $\mathbb{E}_Q[f(r)]\ge f(\mathbb{E}_Q[r]) = f(1) = 0$. If $f$ is strictly convex at $1$, equality forces $r = 1$ $Q$-a.s., hence $P=Q$ (same argument as Theorem 11.17). If $P\not\ll Q$ then $P(\{q=0\})>0$; strict convexity at $1$ plus $f(1)=0$ forces $f'(\infty)>0$, so the escape term is strictly positive, and the first term is $\ge0$. $\blacksquare$

**Theorem 11.31 (joint convexity).** $(P,Q)\mapsto D_f(P\Vert Q)$ is jointly convex: for $\lambda\in[0,1]$,
$$D_f\bigl(\lambda P_1 + (1-\lambda)P_2 \,\Vert\, \lambda Q_1 + (1-\lambda)Q_2\bigr) \le \lambda D_f(P_1\Vert Q_1) + (1-\lambda)D_f(P_2\Vert Q_2).$$

*Proof.* Work with densities w.r.t. a common $\mu$ dominating all four measures. Fix $x$ and write $p_i = p_i(x)$, $q_i = q_i(x)$, $\bar p = \lambda p_1 + (1-\lambda)p_2$, $\bar q = \lambda q_1 + (1-\lambda)q_2$. Define the **perspective function** $g(p,q) := q f(p/q)$ for $q>0$, extended by $g(p,0) := f'(\infty)p$. A standard fact (proved below) is that $g$ is jointly convex on $[0,\infty)^2$. Given that,
$$g(\bar p,\bar q) \le \lambda g(p_1,q_1) + (1-\lambda)g(p_2,q_2)$$
pointwise in $x$; integrating $\mathrm{d}\mu$ gives the claim.

*Convexity of the perspective.* Let $q_1,q_2>0$, $\alpha\in[0,1]$, and set $\bar q = \alpha q_1 + (1-\alpha)q_2$, $\bar p = \alpha p_1 + (1-\alpha)p_2$. Write $\theta := \alpha q_1/\bar q \in[0,1]$, so $1-\theta = (1-\alpha)q_2/\bar q$. Then
$$\frac{\bar p}{\bar q} = \frac{\alpha q_1}{\bar q}\cdot\frac{p_1}{q_1} + \frac{(1-\alpha)q_2}{\bar q}\cdot\frac{p_2}{q_2} = \theta\frac{p_1}{q_1} + (1-\theta)\frac{p_2}{q_2},$$
so convexity of $f$ gives $f(\bar p/\bar q)\le \theta f(p_1/q_1) + (1-\theta)f(p_2/q_2)$. Multiply by $\bar q$:
$$\bar q f(\bar p/\bar q) \le \alpha q_1 f(p_1/q_1) + (1-\alpha)q_2 f(p_2/q_2),$$
which is the assertion. The boundary cases $q_i = 0$ follow by taking limits, using the definition of $f'(\infty)$. $\blacksquare$

**Theorem 11.32 (data processing inequality for $f$-divergences).** Let $K$ be a Markov kernel (a "channel") from $\mathcal{X}$ to $\mathcal{Y}$, and write $PK$ for the law of the output when the input is drawn from $P$. Then
$$D_f(PK\Vert QK) \le D_f(P\Vert Q),$$
with no assumption of absolute continuity: the inequality holds even when $P$ and $Q$ are mutually singular.

*Proof.* The proof that works in full generality goes through the **perspective function** of $f$,
$$g(u,v) := \begin{cases} v\,f(u/v), & v>0,\\ u\,f'(\infty), & v=0,\ u>0,\\ 0,& u=v=0,\end{cases}$$
which has two properties and needs no others. It is **jointly convex** on $[0,\infty)^2$ — this is exactly the fact established inside the proof of Theorem 11.31, so we may reuse it — and it is **positively homogeneous**, $g(tu,tv) = t\,g(u,v)$ for $t>0$. In this notation Definition 11.29 reads
$$D_f(P\Vert Q) = \int_{\mathcal{X}} g\bigl(p(x),q(x)\bigr)\,\mathrm{d}\mu(x)$$
for any $\mu$ dominating both, and homogeneity is precisely what makes the value independent of the choice of $\mu$ — so the boundary convention $g(u,0) = uf'(\infty)$ is not a patch but a requirement.

Suppose $K(\cdot\mid x)$ has density $k(y\mid x)$ with respect to a measure $\nu$ on $\mathcal{Y}$, so $(PK)(y) = \int k(y\mid x)p(x)\,\mathrm{d}\mu(x)$ and likewise for $Q$. Fix $y$, put $m(y):=\int k(y\mid x)\,\mathrm{d}\mu(x)$, and let $\tilde\nu_y := k(y\mid\cdot)\,\mathrm{d}\mu/m(y)$, a probability measure on $\mathcal{X}$. Then, using homogeneity and then Jensen's inequality for the jointly convex $g$ (the two-dimensional form: $g$ of an average is at most the average of $g$),
$$g\bigl((PK)(y),(QK)(y)\bigr) = m(y)\,g\!\left(\int p\,\mathrm{d}\tilde\nu_y,\ \int q\,\mathrm{d}\tilde\nu_y\right)\ \le\ m(y)\int g(p,q)\,\mathrm{d}\tilde\nu_y = \int k(y\mid x)\,g\bigl(p(x),q(x)\bigr)\,\mathrm{d}\mu(x).$$
Integrate over $y$ against $\nu$ and exchange the order of integration (Tonelli; replacing $f(t)$ by $f(t)-f'(1)(t-1)$ makes the integrand non-negative without changing $D_f$, since $\int(p-q)\,\mathrm{d}\mu = 0$):
$$D_f(PK\Vert QK) \le \int_{\mathcal{X}} g\bigl(p,q\bigr)\left[\int_{\mathcal{Y}} k(y\mid x)\,\mathrm{d}\nu(y)\right]\mathrm{d}\mu = \int g(p,q)\,\mathrm{d}\mu = D_f(P\Vert Q),$$
because $\int k(y\mid x)\,\mathrm{d}\nu(y) = 1$ for every $x$. For a kernel without densities, apply the same two properties to the joint laws $P\otimes K$ and $Q\otimes K$ on $\mathcal{X}\times\mathcal{Y}$ and then marginalise; only joint convexity and homogeneity are ever used. $\blacksquare$

Two remarks. First, where $q(x) = 0 < p(x)$ the integrand contributes $p(x)f'(\infty)$, which is $+\infty$ for KL and $\chi^2$; the inequality is then either trivially true or genuinely informative, but it never breaks, and this matters because the mutually singular case is exactly the one Proposition 11.44 will need. Second, if $P\ll Q$ there is a shorter route worth knowing, and Exercise C1 asks you to supply it: writing $r = \mathrm{d}P/\mathrm{d}Q$, the output likelihood ratio is $\mathbb{E}[r(X)\mid Y=y]$, and the result is one application of conditional Jensen.

The proof is worth remembering because it explains *why* processing cannot increase divergence: the output likelihood ratio at $y$ is a weighted average of input likelihood ratios over all $x$ that could have produced $y$, and convexity says averaging can only pull $f$ down. All the DPIs in this chapter — Proposition 11.7(3), Theorem 11.28, Theorem 11.32 — are the same fact: processing replaces the likelihood ratio by an average of likelihood ratios, and a convex function of an average is at most the average of the convex function.

**Worked Example 11.14 (the whole family on one pair of coins).** Computed values, nats where applicable:

| | $P=\mathrm{Ber}(0.9), Q=\mathrm{Ber}(0.5)$ | $P=\mathrm{Ber}(0.9), Q=\mathrm{Ber}(0.1)$ |
|---|---|---|
| $D_{\mathrm{KL}}(P\Vert Q)$ | $0.368064$ | $1.757780$ |
| $D_{\mathrm{KL}}(Q\Vert P)$ | $0.510826$ | $1.757780$ |
| JS | $0.101749$ | $0.368064$ |
| $\chi^2$ | $0.640000$ | $7.111111$ |
| TV | $0.400000$ | $0.800000$ |
| squared Hellinger | $0.211146$ | $0.800000$ |

Observations you should be able to explain: the second column is symmetric in KL because $P$ and $Q$ are mirror images; JS is bounded by $\log 2 = 0.693147$ and never approaches it here because the supports overlap; TV and Hellinger stay in their advertised ranges; and $\chi^2$ blows up fastest, which is why it appears in variance calculations (Chapter 8's Cramér–Rao neighbourhood) but rarely as a training objective.

### 11.5.2 Variational representations: how a divergence becomes an optimisation over critics

We can compute divergences when we have densities. In machine learning we almost never do — we have *samples*. The variational representations turn a divergence into a supremum over functions, which is something a neural network can chase.

**Theorem 11.33 (Donsker–Varadhan).** Let $P,Q$ be probability measures on $\mathcal{X}$. Then
$$D_{\mathrm{KL}}(P\Vert Q) = \sup_{T}\ \Bigl\{ \mathbb{E}_P[T] - \log\mathbb{E}_Q\bigl[e^{T}\bigr]\Bigr\},$$
where the supremum is over measurable $T:\mathcal{X}\to\mathbb{R}$ with $\mathbb{E}_Q[e^T]<\infty$. If $P\ll Q$ the supremum is attained by $T^\star = \log\frac{\mathrm{d}P}{\mathrm{d}Q} + c$ for any constant $c$.

*Proof.* **Upper bound on the right side.** Fix $T$ with $Z := \mathbb{E}_Q[e^T]<\infty$ and define the **tilted measure** $\mathrm{d}Q_T := e^{T}\mathrm{d}Q/Z$, a probability measure. If $P\not\ll Q$ then $D_{\mathrm{KL}}(P\Vert Q)=\infty$ and there is nothing to prove for the inequality $\ge$; assume $P\ll Q$. Then $P\ll Q_T$ as well (they have the same null sets, since $e^T>0$), and
$$D_{\mathrm{KL}}(P\Vert Q_T) = \int\log\frac{\mathrm{d}P}{\mathrm{d}Q}\frac{Z}{e^{T}}\,\mathrm{d}P = D_{\mathrm{KL}}(P\Vert Q) - \mathbb{E}_P[T] + \log Z .$$
Since $D_{\mathrm{KL}}(P\Vert Q_T)\ge 0$ by Theorem 11.17, we get
$$\mathbb{E}_P[T] - \log\mathbb{E}_Q[e^T] \ \le\ D_{\mathrm{KL}}(P\Vert Q) \tag{11.3}$$
for every admissible $T$, so the supremum is at most the KL.

**Attainment.** Take $T^\star = \log(\mathrm{d}P/\mathrm{d}Q)$. Then $\mathbb{E}_Q[e^{T^\star}] = \int \mathrm{d}P/\mathrm{d}Q\,\mathrm{d}Q = 1$, so $\log Z = 0$, and $\mathbb{E}_P[T^\star] = D_{\mathrm{KL}}(P\Vert Q)$. So the bound (11.3) is an equality at $T^\star$, and adding a constant $c$ to $T$ changes both terms by $c$, leaving the objective unchanged.

**The case $P\not\ll Q$.** Then there is $A$ with $Q(A) = 0 < P(A)$. Take $T_n := n\mathbf{1}_A$. Then $\mathbb{E}_Q[e^{T_n}] = 1$ (the set where $T_n\ne0$ is $Q$-null) while $\mathbb{E}_P[T_n] = nP(A)\to\infty$, so the supremum is $+\infty = D_{\mathrm{KL}}(P\Vert Q)$. $\blacksquare$

I have written that proof out in full because it is the mathematical content of a large family of algorithms, and the structure — *tilt the reference measure by $e^T$, then use non-negativity of KL* — is the whole trick. Notice three features that will matter for practice:

1. The bound (11.3) holds for **every** $T$. So any critic gives a valid **lower** bound on the KL. Training a critic is climbing toward the true value from below.
2. The optimal critic is the log density ratio. So a well-trained discriminator *is* a density-ratio estimator; this is the precise sense in which GAN discriminators "know" the ratio.
3. The objective involves $\log\mathbb{E}_Q[e^T]$. Expectations of exponentials are dominated by rare large values of $T$, so estimating this term from a finite sample is exactly the situation in which sample averages behave badly. That is the seed of every practical failure mode in §11.5.3.

**Theorem 11.34 (Fenchel / $f$-GAN variational lower bound).** Let $f$ be as in Definition 11.29 with convex conjugate $f^*(s) := \sup_{t>0}\{st - f(t)\}$. Then for all $P,Q$,
$$D_f(P\Vert Q) \ \ge\ \sup_{T}\ \Bigl\{\mathbb{E}_P[T] - \mathbb{E}_Q\bigl[f^*(T)\bigr]\Bigr\},$$
with equality when the supremum ranges over all measurable $T$ and $P\ll Q$, attained at $T^\star = f'\bigl(\mathrm{d}P/\mathrm{d}Q\bigr)$.

*Proof.* By Fenchel duality, $f(t) = \sup_s\{st - f^*(s)\}$ for convex lower-semicontinuous $f$. Substituting $t = p/q$ pointwise,
$$D_f(P\Vert Q) = \int q\,\sup_{s}\Bigl\{s\frac pq - f^*(s)\Bigr\}\mathrm{d}\mu = \int \sup_s\bigl\{s\,p - f^*(s)q\bigr\}\mathrm{d}\mu \ \ge\ \int \bigl\{T p - f^*(T)q\bigr\}\mathrm{d}\mu$$
for any measurable $T$, which is the stated bound. For equality, take $T^\star(x) = f'(p(x)/q(x))$, the maximiser of the inner supremum (by the first-order condition $t = (f^*)'(s)\iff s = f'(t)$ for differentiable convex $f$); then the pointwise supremum is attained everywhere, and the inequality is an equality by construction. Measurability of $T^\star$ holds because $f'$ is monotone hence Borel and $p/q$ is measurable. $\blacksquare$

Now read off the algorithms. Take the JS generator and substitute $T = \log\frac{2D}{1}$-style reparameterisations; the cleanest route is direct. Define the GAN value function
$$V(D) := \mathbb{E}_{P}\bigl[\log D(X)\bigr] + \mathbb{E}_{Q}\bigl[\log(1-D(X))\bigr].$$
Pointwise maximisation of $p\log D + q\log(1-D)$ over $D\in(0,1)$ gives $D^\star = p/(p+q)$, and substituting back,
$$\sup_D V(D) = \int p\log\frac{p}{p+q} + q\log\frac{q}{p+q}\,\mathrm{d}\mu = 2\,\mathrm{JS}(P,Q) - \log 4,$$
so $\mathrm{JS}(P,Q) = \tfrac12\bigl[\sup_D V(D) + \log 4\bigr]$. This is *the* original GAN discriminator objective, and the optimal discriminator is a monotone function of the density ratio $p/q$, exactly as Theorem 11.34 requires: a trained discriminator is a density-ratio estimator wearing a sigmoid. With $f$ the KL generator you get MINE (mutual information neural estimation), applying Theorem 11.33 with $P = P_{XY}$ and $Q = P_X\otimes P_Y$, where samples from $Q$ are obtained by shuffling one coordinate within a minibatch. With a particular restricted critic family and a softmax normalisation you get InfoNCE. **These are not analogies. They are the same theorem with different $f$ and different critic classes.**

### 11.5.3 The honest part: these estimators are bad in the regime we care about

Every variational bound above is valid. None of them says the *estimate you compute from a finite sample* is close to the truth. Two separate problems compound.

**Problem 1: the estimator is biased, and the bias direction is not obvious.** The Donsker–Varadhan objective contains $\log\mathbb{E}_Q[e^T]$, replaced in practice by $\log\bigl(\frac1n\sum_j e^{T(x_j)}\bigr)$. Since $\log$ is concave, Jensen gives $\mathbb{E}\bigl[\log \widehat{Z}\bigr]\le \log \mathbb{E}[\widehat Z]$, so $-\log\widehat Z$ is biased **upward**: the plug-in DV estimate systematically *overstates* the KL, even with the exactly optimal critic.

**Problem 2: the critic is fitted on the same samples**, so it overfits the minibatch, inflating $\mathbb{E}_P[T]$ on training data and collapsing out of sample.

**Worked Example 11.15 (both failure modes, measured).** Let $P = N(0,1)$ and $Q = N(\Delta,1)$, so $D_{\mathrm{KL}}(P\Vert Q) = \Delta^2/2$ exactly and the optimal critic is $T^\star(x) = \log\frac{p(x)}{q(x)} = -\Delta x + \Delta^2/2$, a linear function.

*(a) Plug-in bias with the optimal critic.* Using the exact $T^\star$ (no fitting at all), $500$ independent trials per cell, seed $7$, mean estimate $\pm$ standard deviation:

| true KL (nats) | batch $64$ | batch $512$ | batch $4096$ |
|---|---|---|---|
| $1$ | $1.038 \pm 0.333$ | $1.005\pm0.123$ | $0.999\pm0.045$ |
| $4$ | $4.656\pm0.993$ | $4.222\pm0.578$ | $4.069\pm0.328$ |
| $8$ | $10.196\pm1.567$ | $9.181\pm1.080$ | $8.597\pm0.753$ |
| $18$ | $25.658\pm2.711$ | $23.354\pm1.931$ | $21.809\pm1.692$ |

At $\mathrm{KL} = 18$ nats with a batch of $64$, the estimator returns $25.7$ on average — a $43\%$ overestimate, with the *correct* critic and no optimisation error whatsoever. The problem is entirely the $\log\frac1n\sum e^{T}$ term: the relevant expectation is dominated by samples in the far tail of $Q$ that a batch of $64$ never contains.

*(b) A trained critic (MINE) on bivariate Gaussians.* Now estimate $I(X;Y)$ for correlated Gaussians with $n = 20{,}000$ samples (seed $0$), by three methods: equal-mass binning with $16$ bins per axis; the KSG $k$-nearest-neighbour estimator with $k = 3$ (max-norm, digamma corrections); and a MINE critic (a $2\to128\to128\to1$ ReLU network, Adam with learning rate $5\times10^{-4}$, $4000$ steps, batch $256$, then evaluated on twenty fresh batches of $4096$).

| true $I$ (nats) | $\rho$ | binning, $16$ bins | KSG, $k{=}3$ | MINE (held-out) | MINE (final training batches) |
|---|---|---|---|---|---|
| $0.2$ | $0.5742$ | $0.194$ | $0.197$ | $0.192$ | $0.193$ |
| $0.5$ | $0.7951$ | $0.472$ | $0.497$ | $0.496$ | $0.492$ |
| $1.0$ | $0.9299$ | $0.921$ | $1.001$ | $1.002$ | $0.997$ |
| $2.0$ | $0.9908$ | $1.702$ | $2.003$ | $1.957$ | $1.987$ |
| $4.0$ | $0.99983$ | $2.492$ | $4.020$ | $3.539$ | $4.002$ |
| $6.0$ | $1.00000$ | $2.717$ | $6.011$ | $3.127$ | $6.168$ |

Three things to take from this table. Binning saturates: with $16$ bins per axis the maximum representable MI is $\log 16 = 2.77$ nats, and the estimate flattens at $2.7$ no matter how dependent the variables are — the DPI of Worked Example 11.12 in action, now as an estimator ceiling. KSG does very well here, but note this is the easiest possible case (two dimensions, smooth Gaussian, huge sample); its performance degrades sharply with dimension. And MINE tracks the truth up to about $2$ nats and then breaks: at $I = 6$ nats it reports $6.17$ on its own training batches (an *over*estimate, per problem 1) and $3.13$ on fresh batches (a $48\%$ *under*estimate, per problem 2). The gap between those two columns is the overfitting, and it appears exactly in the high-MI regime where representation-learning papers operate.

There is a theoretical result behind this, which I will state and not prove: any estimator of mutual information that is a *high-confidence lower bound* valid for all distributions, computed from $n$ samples, is bounded by $O(\log n)$. With a batch of $256$ that is at most about $5.5$ nats, whatever your architecture. InfoNCE has the same ceiling in a cleaner form: an InfoNCE objective with $K$ negatives is bounded above by $\log K$, so a batch of $65536$ can certify at most $11.09$ nats $=16$ bits. Papers reporting "our method achieves high mutual information" with batch sizes of a few hundred are reporting a number that cannot exceed a handful of nats by construction.

> **Common Confusion 11.6.** *"My MINE estimator converged to $6.2$ nats, so my encoder captures $6.2$ nats of information about the input."*
>
> Three separate objections, in increasing severity.
>
> First, $6.2$ was the value on the batches the critic was trained on; on held-out batches from the same distribution it was $3.1$ (Worked Example 11.15). The optimisation found a critic that exploits the specific samples, and the DV objective rewards that.
>
> Second, even with a perfect critic the plug-in estimate is biased upward by the $\log\frac1n\sum e^T$ term, and the bias grows with the true value — $43\%$ at $18$ nats with batch $64$.
>
> Third and most fundamental: if the encoder is deterministic and the input is continuous, the true $I(X;Z)$ is $+\infty$ (Theorem 11.26(5)), so *any* finite number your estimator prints is infinitely wrong. The quantity people actually want in that case is something else — an accuracy under a decoder class, a rate under a quantisation, an $I$ with noise added — and it is worth saying which one you mean. The estimator is not measuring what its name suggests; it is a training signal that happens to work, which is a perfectly respectable thing to be, but it should be reported as such.

> **Self-check 11.5.**
> 1. Verify that the TV generator $f(t)=\tfrac12|t-1|$ is convex with $f(1)=0$, and compute $f'(\infty)$.
> 2. Why does every $f$-divergence satisfy $D_f(P\Vert Q) = D_{f^\dagger}(Q\Vert P)$ where $f^\dagger(t) := t f(1/t)$? Which generators are self-conjugate in this sense?
> 3. In the Donsker–Varadhan formula, what happens to the objective if you add a constant to $T$? What does that tell you about identifiability of the learned critic?
> 4. An InfoNCE loss is trained with batch size $128$. What is the largest mutual information it can certify, in bits?

## 11.6 Total variation, Pinsker, and coupling

Total variation deserves its own section because it is the divergence that answers the question *"can anybody tell these two distributions apart?"* — and because, unlike KL, it is a genuine metric, bounded, and interpretable without any log-scale gymnastics.

**Definition 11.35 (total variation distance).** For probability measures $P,Q$ on $(\mathcal{X},\mathcal{G})$,
$$\mathrm{TV}(P,Q) := \sup_{A\in\mathcal{G}}\ \bigl\lvert P(A)-Q(A)\bigr\rvert .$$

**Proposition 11.36 (three faces of TV).** Let $\mu$ dominate both $P$ and $Q$, with densities $p,q$. Then
$$\mathrm{TV}(P,Q) = \tfrac12\int \lvert p - q\rvert\,\mathrm{d}\mu = 1 - \int \min(p,q)\,\mathrm{d}\mu = D_f(P\Vert Q) \text{ with } f(t) = \tfrac12\lvert t-1\rvert,$$
and the supremum in Definition 11.35 is attained at $A^\star = \{p > q\}$.

*Proof.* Let $A^\star := \{p>q\}$. For any $A$,
$$P(A)-Q(A) = \int_A (p-q)\,\mathrm{d}\mu \le \int_{A^\star}(p-q)\,\mathrm{d}\mu,$$
since including any point where $p<q$ only decreases the integral; so the sup over $A$ of $P(A)-Q(A)$ is attained at $A^\star$, and by symmetry the sup of $|P(A)-Q(A)|$ is too. Next, because $\int(p-q)\mathrm{d}\mu = 0$, the positive and negative parts of $p-q$ have equal mass:
$$\int_{A^\star}(p-q)\,\mathrm{d}\mu = \int_{(A^\star)^c}(q-p)\,\mathrm{d}\mu = \tfrac12\int\lvert p-q\rvert\,\mathrm{d}\mu .$$
For the third expression, use $\min(p,q) = \tfrac12\bigl(p+q-\lvert p-q\rvert\bigr)$ and integrate: $\int\min(p,q)\,\mathrm{d}\mu = 1 - \tfrac12\int|p-q|\,\mathrm{d}\mu$. The $f$-divergence identification is immediate from Definition 11.29. $\blacksquare$

The middle expression, $1-\int\min(p,q)$, is the one to remember: $\int\min(p,q)\,\mathrm{d}\mu$ is the **overlap** of the two distributions, and TV is one minus the overlap. Two distributions with $80\%$ overlap are at TV distance $0.2$. That is a sentence you can say to a non-specialist.

**Proposition 11.37 (TV is exactly the testing advantage).** Consider deciding between $H_0: X\sim P$ and $H_1: X\sim Q$ from a single observation, using a (possibly randomised) test. Then
$$\inf_{\text{tests}}\ \bigl\{ \mathbb{P}(\text{type I error}) + \mathbb{P}(\text{type II error})\bigr\} = 1 - \mathrm{TV}(P,Q),$$
attained by the likelihood-ratio test that rejects $H_0$ on $\{q>p\}$.

*Proof.* A deterministic test is a set $A$ ("declare $H_1$ if $x\in A$"), with type I error $P(A)$ and type II error $Q(A^c)$. Then
$$P(A)+Q(A^c) = P(A) + 1 - Q(A) = 1 - \bigl(Q(A)-P(A)\bigr)\ \ge\ 1 - \mathrm{TV}(P,Q),$$
by Definition 11.35, with equality iff $A$ attains the supremum, i.e. $A = \{q>p\}$ up to null sets (Proposition 11.36). Randomised tests are mixtures of deterministic ones and the objective is affine in the test, so they cannot beat the best deterministic test. $\blacksquare$

This is why TV is *the* right notion of distinguishability, and it dovetails with Chapter 9's Neyman–Pearson lemma: the optimal test is a likelihood-ratio threshold, and TV is the value of the game when you weight the two errors equally. In ML terms: if a real and a generated distribution are at TV distance $0.1$, then the very best possible discriminator — not a neural network, the Bayes-optimal one — achieves error $0.45$ on a balanced two-sample test. Conversely, an adversary who can classify real vs. fake at $95\%$ accuracy proves $\mathrm{TV}\ge0.9$. TV converts "can a model be caught?" into a number, and no amount of architecture search moves it.

**Theorem 11.38 (Pinsker's inequality).** For all $P,Q$: $\quad\mathrm{TV}(P,Q) \le \sqrt{\tfrac12 D_{\mathrm{KL}}(P\Vert Q)}$.

*Proof.* **Step 1: the two-point case.** For $a,b\in(0,1)$ write $d(a\Vert b) := a\log\frac ab + (1-a)\log\frac{1-a}{1-b}$, the KL between $\mathrm{Ber}(a)$ and $\mathrm{Ber}(b)$. Claim: $d(a\Vert b)\ge 2(a-b)^2$. Fix $b$ and define $g(a) := d(a\Vert b) - 2(a-b)^2$ on $(0,1)$. Then $g(b) = 0$ and
$$g'(a) = \log\frac{a}{1-a} - \log\frac{b}{1-b} - 4(a-b) = \int_b^a\left(\frac{1}{t(1-t)} - 4\right)\mathrm{d}t \ \ge\ 0 \quad\text{for } a \ge b,$$
because $t(1-t)\le 1/4$ on $[0,1]$ gives $1/(t(1-t))\ge 4$. So $g$ is non-decreasing on $[b,1)$ and $g(b)=0$, hence $g\ge0$ there. For $a<b$ the same computation shows $g'\le0$ on $(0,b]$, so $g$ is non-increasing up to $b$ where it is $0$, hence again $g\ge0$. This proves the claim.

**Step 2: reduce the general case by data processing.** Let $A^\star = \{p>q\}$ and let $\Phi(x) := \mathbf{1}_{A^\star}(x)$, a deterministic map to $\{0,1\}$, i.e. a Markov kernel. By Theorem 11.32 with the KL generator,
$$D_{\mathrm{KL}}(P\Vert Q)\ \ge\ D_{\mathrm{KL}}\bigl(\Phi_\#P\,\Vert\,\Phi_\#Q\bigr) = d\bigl(P(A^\star)\,\Vert\,Q(A^\star)\bigr) \ \ge\ 2\bigl(P(A^\star)-Q(A^\star)\bigr)^2 = 2\,\mathrm{TV}(P,Q)^2,$$
using Step 1 and Proposition 11.36. Rearranging gives the result. (Edge cases $P(A^\star)\in\{0,1\}$ or $Q(A^\star)\in\{0,1\}$ are handled by continuity, or directly: if $Q(A^\star)=0<P(A^\star)$ then $D_{\mathrm{KL}} = \infty$.) $\blacksquare$

That proof is a template worth internalising: *prove the inequality for two points, then push a general problem down to two points using the data-processing inequality.* Many statistical lower bounds have exactly this shape, including the ones in §11.9.

**Worked Example 11.16 (how tight is Pinsker?).** Computed values for Bernoulli pairs:

| $P$ | $Q$ | $\mathrm{TV}$ | $\sqrt{D_{\mathrm{KL}}/2}$ | ratio |
|---|---|---|---|---|
| $\mathrm{Ber}(0.50)$ | $\mathrm{Ber}(0.51)$ | $0.010000$ | $0.010001$ | $0.9999$ |
| $\mathrm{Ber}(0.50)$ | $\mathrm{Ber}(0.60)$ | $0.100000$ | $0.101022$ | $0.9899$ |
| $\mathrm{Ber}(0.50)$ | $\mathrm{Ber}(0.90)$ | $0.400000$ | $0.505384$ | $0.7915$ |
| $\mathrm{Ber}(0.50)$ | $\mathrm{Ber}(0.99)$ | $0.490000$ | $0.898461$ | $0.5454$ |
| $\mathrm{Ber}(0.10)$ | $\mathrm{Ber}(0.11)$ | $0.010000$ | $0.016201$ | $0.6172$ |
| $\mathrm{Ber}(0.01)$ | $\mathrm{Ber}(0.02)$ | $0.010000$ | $0.039493$ | $0.2532$ |

Pinsker is essentially exact for nearby distributions centred at $p = 1/2$ (first two rows), which is the regime where it is used in concentration arguments. It degrades in two ways. It becomes weak for *far apart* distributions, and indeed becomes vacuous once $D_{\mathrm{KL}}\ge 2$, since then the bound exceeds $1$ while TV never can: $\mathrm{Ber}(0.5)$ vs $\mathrm{Ber}(0.99)$ has bound $0.898$ against truth $0.49$. And it becomes weak for distributions concentrated near the boundary (last two rows), because there the KL is dominated by the ratio of small probabilities while TV only sees their difference. In that regime sharper bounds are needed; Pinsker alone will not do. The most useful replacement is the **Bretagnolle–Huber inequality**,
$$\mathrm{TV}(P,Q)\ \le\ \sqrt{1 - e^{-D_{\mathrm{KL}}(P\Vert Q)}},$$
which — unlike Pinsker — stays below $1$ for *every* finite KL and so never goes vacuous (quoted, not proved; see §11.11). Reverse-Pinsker bounds, which control KL by TV under a bounded density ratio, are the standard patch in the other direction.

For Gaussians the comparison is clean: $P = N(0,1)$, $Q = N(\Delta,1)$ gives $D_{\mathrm{KL}} = \Delta^2/2$, so the Pinsker bound is exactly $\Delta/2$, while $\mathrm{TV} = 2\Phi(\Delta/2)-1$. At $\Delta = 0.1$: TV $=0.039878$ versus bound $0.05$. At $\Delta = 4$: TV $= 0.954500$ versus bound $2$ — vacuous — while Bretagnolle–Huber gives $\sqrt{1-e^{-8}} = 0.999832$: loose, but still a statement. Pinsker is a small-$\Delta$ tool; Bretagnolle–Huber is the all-weather one.

**Theorem 11.39 (coupling characterisation; maximal coupling).** Let $\Pi(P,Q)$ denote the set of **couplings** of $P$ and $Q$: joint laws $\pi$ on $\mathcal{X}\times\mathcal{X}$ with marginals $P$ and $Q$. Then
$$\mathrm{TV}(P,Q) = \min_{\pi\in\Pi(P,Q)}\ \pi\bigl(\{(x,y): x\ne y\}\bigr),$$
and the minimum is attained.

*Proof.* **Lower bound.** For any coupling $\pi$ and any measurable $A$,
$$P(A)-Q(A) = \pi(X\in A) - \pi(Y\in A) = \pi(X\in A, Y\notin A) - \pi(X\notin A, Y\in A) \le \pi(X\ne Y),$$
so taking the sup over $A$ gives $\mathrm{TV}(P,Q)\le\pi(X\ne Y)$ for every coupling.

**Attainment (the maximal coupling).** Let $\mu$ dominate $P,Q$ with densities $p,q$, and define the overlap measure $\nu$ by $\mathrm{d}\nu := \min(p,q)\,\mathrm{d}\mu$, with total mass $m := \nu(\mathcal{X}) = 1-\mathrm{TV}(P,Q)$ (Proposition 11.36). If $m = 0$ the measures are mutually singular, any product coupling has $\pi(X\ne Y)=1=\mathrm{TV}$, and we are done. Otherwise set $\delta := \mathrm{TV}(P,Q) = 1-m$ and define
$$\mathrm{d}P^- := \frac{(p - \min(p,q))\,\mathrm{d}\mu}{\delta}, \qquad \mathrm{d}Q^- := \frac{(q-\min(p,q))\,\mathrm{d}\mu}{\delta},$$
both probability measures (each numerator has mass $1-m = \delta$). Now sample as follows: with probability $m$, draw $X = Y \sim \nu/m$; with probability $1-m = \delta$, draw $X\sim P^-$ and $Y\sim Q^-$ independently. The resulting joint law $\pi$ has first marginal $m\cdot\frac{\nu}{m} + \delta\cdot P^- = \nu + (P - \nu) = P$, and similarly the second is $Q$. Finally, $P^-$ is supported on $\{p>q\}$ and $Q^-$ on $\{q>p\}$, which are disjoint, so in the second branch $X\ne Y$ almost surely, giving $\pi(X\ne Y) = \delta = \mathrm{TV}(P,Q)$. $\blacksquare$

**Worked Example 11.17 (a maximal coupling you can write down).** Take $P = \mathrm{Ber}(0.3)$ and $Q = \mathrm{Ber}(0.5)$, so $\mathrm{TV} = \tfrac12(|0.7-0.5| + |0.3-0.5|) = 0.2$ and the overlap is $\min(0.7,0.5)+\min(0.3,0.5) = 0.5+0.3 = 0.8 = 1 - \mathrm{TV}$. The maximal coupling is the joint table

| $\pi(X{=}i,Y{=}j)$ | $Y=0$ | $Y=1$ | row sum |
|---|---|---|---|
| $X=0$ | $0.5$ | $0.2$ | $0.7$ |
| $X=1$ | $0.0$ | $0.3$ | $0.3$ |
| col sum | $0.5$ | $0.5$ | $1$ |

Marginals are correct, and $\pi(X\ne Y) = 0.2 + 0.0 = 0.2 = \mathrm{TV}$. Note the coupling is *not* the product: it deliberately correlates $X$ and $Y$ to make them agree as often as possible. This is the pattern behind every coupling argument: to bound how different two processes are, run them on shared randomness and count disagreements. It is also the pattern behind the transport distances of §11.7 — there, instead of counting disagreements we charge for how far apart the coupled points are.

> **Self-check 11.6.**
> 1. Two distributions have overlap $0.65$. What is their TV distance, and what is the best possible accuracy of a balanced two-sample classifier?
> 2. Is $\mathrm{TV}(P,Q)\le\sqrt{D_{\mathrm{KL}}(Q\Vert P)/2}$ also true? (Careful: which direction did we prove?)
> 3. Give two distributions with $\mathrm{TV} = 1$ but which "look close" in a scatter plot.
> 4. Why is Pinsker useless when $D_{\mathrm{KL}} = 5$ nats, and what does that tell you about using KL to certify indistinguishability?

## 11.7 Optimal transport

### 11.7.1 Couplings with a price tag

The maximal coupling of Theorem 11.39 asked: how often can we make $X$ and $Y$ agree? Optimal transport asks a refined question: when they disagree, *how far apart* are they? To ask that, you need a metric on the space — and that single extra ingredient is what makes transport distances see things that $f$-divergences cannot.

**Definition 11.40 (Wasserstein distance).** Let $(\mathcal{X},d)$ be a Polish metric space, $p\ge1$, and let $P,Q$ be probability measures with finite $p$-th moments. With $\Pi(P,Q)$ the set of couplings as in Theorem 11.39,
$$W_p(P,Q) := \left(\inf_{\pi\in\Pi(P,Q)}\ \int_{\mathcal{X}\times\mathcal{X}} d(x,y)^p\,\mathrm{d}\pi(x,y)\right)^{1/p}.$$
The infimum is the **Kantorovich problem**; a minimising $\pi$ is an **optimal plan**.

Read $\pi(\mathrm{d}x,\mathrm{d}y)$ as a plan for moving a pile of earth shaped like $P$ into a pile shaped like $Q$: $\pi(\mathrm{d}x,\mathrm{d}y)$ is how much mass goes from $x$ to $y$, the marginal constraints say you must move all of $P$ and end up with exactly $Q$, and $d(x,y)^p$ is the price per unit. $W_p$ is the cost of the cheapest plan. Note the contrast with Definition 11.29 in the sharpest terms: an $f$-divergence looks only at the *ratio* $p(x)/q(x)$ at each single point $x$, so it never compares different points; $W_p$ is built entirely from comparisons between different points. That is the whole difference, and everything below is a consequence.

Basic facts (proofs in the references; $W_p$ being a metric requires the gluing lemma):

**Theorem 11.41 (Kantorovich duality, stated).** With $\mathcal{X}$ Polish and $c(x,y)$ lower semicontinuous and non-negative,
$$\inf_{\pi\in\Pi(P,Q)}\int c\,\mathrm{d}\pi = \sup\left\{\int \varphi\,\mathrm{d}P + \int\psi\,\mathrm{d}Q \ :\ \varphi(x)+\psi(y)\le c(x,y)\ \forall x,y\right\},$$
both extrema being attained under mild conditions. In the special case $c = d$ (i.e. $p=1$), this reduces to the **Lipschitz dual**
$$W_1(P,Q) = \sup_{\lVert \varphi\rVert_{\mathrm{Lip}}\le 1}\ \left\{\mathbb{E}_P[\varphi] - \mathbb{E}_Q[\varphi]\right\}. \tag{11.4}$$

The reduction from the general dual to (11.4) is worth one sentence, because it explains where the Lipschitz constraint in WGAN comes from: given any admissible pair $(\varphi,\psi)$ for $c=d$, one may replace $\varphi$ by its $c$-transform and use the triangle inequality to show the optimum can be taken with $\psi = -\varphi$ and $\varphi$ 1-Lipschitz. So the "critic" in a Wasserstein GAN is a Kantorovich potential, and the weight clipping or gradient penalty is an attempt to enforce $\lVert\varphi\rVert_{\mathrm{Lip}}\le1$. If the constraint is not enforced, the supremum in (11.4) is $+\infty$ and the objective is meaningless — which is why the constraint is not an optional regulariser.

### 11.7.2 The one-dimensional closed form

**Theorem 11.42 (transport on the line).** Let $P,Q$ be probability measures on $\mathbb{R}$ with CDFs $F,G$ and quantile functions $F^{-1},G^{-1}$ (generalised inverses). Then for $p\ge1$,
$$W_p(P,Q)^p = \int_0^1 \bigl\lvert F^{-1}(u) - G^{-1}(u)\bigr\rvert^p\,\mathrm{d}u,$$
and the **monotone** coupling $X = F^{-1}(U)$, $Y = G^{-1}(U)$ with $U\sim\mathrm{Unif}[0,1]$ is optimal. For $p>1$ the cost is *strictly* convex and the monotone coupling is the unique optimum; for $p=1$ it is optimal but generally **not** unique, a fact with practical consequences explored in Exercise B7. In particular
$$W_1(P,Q) = \int_{\mathbb{R}}\bigl\lvert F(t)-G(t)\bigr\rvert\,\mathrm{d}t. \tag{11.5}$$

*Proof of (11.5) and of optimality for $p=1$.* For real numbers $x,y$,
$$\lvert x - y\rvert = \int_{\mathbb{R}}\bigl\lvert \mathbf{1}\{x\le t\} - \mathbf{1}\{y\le t\}\bigr\rvert\,\mathrm{d}t,$$
because the integrand equals $1$ exactly for $t$ between $x$ and $y$ and $0$ elsewhere. Let $\pi$ be any coupling and take expectations (Tonelli, everything non-negative):
$$\mathbb{E}_\pi\lvert X-Y\rvert = \int_{\mathbb{R}}\mathbb{E}_\pi\bigl\lvert \mathbf{1}\{X\le t\}-\mathbf{1}\{Y\le t\}\bigr\rvert\,\mathrm{d}t \ \ge\ \int_{\mathbb{R}}\bigl\lvert \mathbb{E}_\pi\mathbf{1}\{X\le t\} - \mathbb{E}_\pi\mathbf{1}\{Y\le t\}\bigr\rvert\,\mathrm{d}t = \int\lvert F-G\rvert,$$
using $\mathbb{E}|Z|\ge|\mathbb{E}Z|$ pointwise in $t$. So (11.5) is a lower bound for every coupling. Now take the monotone coupling: $\mathbf{1}\{F^{-1}(U)\le t\} = \mathbf{1}\{U\le F(t)\}$ almost surely, so
$$\mathbf{1}\{X\le t\}-\mathbf{1}\{Y\le t\} = \mathbf{1}\{U\le F(t)\} - \mathbf{1}\{U\le G(t)\},$$
whose sign is that of $F(t)-G(t)$ for *every* $u$, so the absolute value passes outside the expectation without loss: $\mathbb{E}\lvert\cdot\rvert = \lvert F(t)-G(t)\rvert$. Integrating gives equality, so the monotone coupling attains the infimum and $W_1 = \int|F-G|$.

Notice what the argument does *not* show: it shows the monotone coupling is *a* minimiser, and the chain of inequalities is an equality for it, but nothing forces every minimiser to be monotone. For $p=1$ there are typically many, because the bound $\mathbb{E}|Z|\ge|\mathbb{E}Z|$ is an equality whenever $Z$ has a constant sign — not only for the monotone coupling. The general-$p$ statement, including uniqueness under strict convexity, is quoted in §11.11. $\blacksquare$

Formula (11.5) is why one-dimensional transport is trivial to compute: sort both samples and average absolute differences of matched order statistics. It is also why sliced Wasserstein distances exist — project onto random directions, use the 1-D formula, average — and why they are cheap.

### 11.7.3 Why $f$-divergences go blind and $W_1$ does not

**Proposition 11.43 (every $f$-divergence is constant on mutually singular pairs).** Suppose $P$ and $Q$ are mutually singular. Then
$$D_f(P\Vert Q) = f(0) + f'(\infty),$$
a constant depending only on $f$, not on $P$ or $Q$.

*Proof.* Take $\mu := P+Q$, with densities $p,q$. Mutual singularity means there is a measurable $S$ with $P(S^c)=0=Q(S)$; hence $q = 0$ $\mu$-a.e. on $S$ and $p=0$ $\mu$-a.e. on $S^c$. By Definition 11.29,
$$D_f(P\Vert Q) = \int_{\{q>0\}} q\,f(p/q)\,\mathrm{d}\mu + f'(\infty)P(\{q=0\}) = \int_{S^c} q f(0)\,\mathrm{d}\mu + f'(\infty)\cdot 1 = f(0)\,Q(S^c) + f'(\infty) = f(0)+f'(\infty). \ \blacksquare$$

Substituting from the table in §11.5.1: JS $= \tfrac12\log2 + \tfrac12\log 2 = \log 2$ (Chapter 2's result, now a one-line corollary of a general structural fact); TV $= \tfrac12+\tfrac12 = 1$; squared Hellinger $=1+1 = 2$; KL and $\chi^2$ $=+\infty$. **Every one of them is pinned at a constant.** A constant has zero gradient. That is Failure 4, and Proposition 11.43 shows it was never about the JS divergence in particular; it is a structural property of anything built from density ratios.

**Proposition 11.44 (transport sees the gap: the two parallel segments).** In $\mathbb{R}^2$ with the Euclidean metric, let
$$P := \text{law of }(0,U),\qquad Q_\alpha := \text{law of }(\alpha,U),\qquad U\sim\mathrm{Unif}[0,1].$$
Then $P\perp Q_\alpha$ for every $\alpha\ne0$, so $\mathrm{JS}(P,Q_\alpha)=\log 2$, $\mathrm{TV} = 1$, and $D_{\mathrm{KL}} = \infty$ for all $\alpha \ne 0$; while
$$W_1(P,Q_\alpha) = \lvert\alpha\rvert .$$

*Proof.* Mutual singularity: the two supports are the disjoint closed segments $\{0\}\times[0,1]$ and $\{\alpha\}\times[0,1]$, each carrying full mass of its own measure and zero of the other's. Apply Proposition 11.43 with the generators listed above.

For the transport distance, upper bound first: the coupling $\pi$ that sends $(0,u)\mapsto(\alpha,u)$ (formally, the law of $\bigl((0,U),(\alpha,U)\bigr)$) is a valid coupling with cost $\int \lVert(0,u)-(\alpha,u)\rVert\,\mathrm{d}u = |\alpha|$, so $W_1\le|\alpha|$. Lower bound by duality (11.4): take $\varphi(x_1,x_2) := \mathrm{sign}(\alpha)\,x_1$, which is $1$-Lipschitz. Then $\mathbb{E}_{Q_\alpha}[\varphi] - \mathbb{E}_P[\varphi] = \mathrm{sign}(\alpha)\alpha - 0 = |\alpha|$, and since $W_1$ is symmetric, $W_1\ge|\alpha|$. $\blacksquare$

Chapter 1 proved that a generator $G_\theta:\mathbb{R}^k\to\mathbb{R}^d$ with $k<d$ has $\lambda_d(\mathrm{supp}\,P_\theta) = 0$, using the circle example $G(t)=(\cos t,\sin t)$. Chapter 2 proved that mutual singularity pins JS at $\log 2$. Proposition 11.43 and Proposition 11.44 finish the story: the pinning is not special to JS, it is what "depends only on the density ratio" *means*, and the cure is not a better $f$ but a different kind of object — one that uses the geometry of the space. **This is the complete answer to Failure 4**, and I want to be precise about what it does and does not claim. It says the population objective has zero gradient in $\alpha$ for $f$-divergences and slope $1$ for $W_1$. It does not say WGAN training is therefore stable in practice: the dual (11.4) has to be estimated with a constrained neural critic, and the estimation problem has its own pathologies, discussed in the confusion box below.

<svg viewBox="0 0 720 470" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<text x="360.0" y="30" fill="currentColor" font-size="15" text-anchor="middle">Figure 11.5 — Divergences between Unif[0,1] and Unif[a, 1+a] as a grows</text>
<line x1="100" y1="340" x2="600" y2="340" stroke="currentColor" stroke-width="1.2"/>
<line x1="100" y1="340" x2="100" y2="60" stroke="currentColor" stroke-width="1.2"/>
<line x1="100.0" y1="340" x2="100.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="100.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0</text>
<line x1="219.0" y1="340" x2="219.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="219.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.5</text>
<line x1="338.1" y1="340" x2="338.1" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="338.1" y="362" fill="currentColor" font-size="13" text-anchor="middle">1</text>
<line x1="457.1" y1="340" x2="457.1" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="457.1" y="362" fill="currentColor" font-size="13" text-anchor="middle">1.5</text>
<line x1="576.2" y1="340" x2="576.2" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="576.2" y="362" fill="currentColor" font-size="13" text-anchor="middle">2</text>
<line x1="94" y1="340.0" x2="100" y2="340.0" stroke="currentColor" stroke-width="1"/>
<text x="90" y="344.0" fill="currentColor" font-size="13" text-anchor="end">0</text>
<line x1="94" y1="274.9" x2="100" y2="274.9" stroke="currentColor" stroke-width="1"/>
<text x="90" y="278.9" fill="currentColor" font-size="13" text-anchor="end">0.5</text>
<line x1="94" y1="209.8" x2="100" y2="209.8" stroke="currentColor" stroke-width="1"/>
<text x="90" y="213.8" fill="currentColor" font-size="13" text-anchor="end">1</text>
<line x1="94" y1="79.5" x2="100" y2="79.5" stroke="currentColor" stroke-width="1"/>
<text x="90" y="83.5" fill="currentColor" font-size="13" text-anchor="end">2</text>
<text x="350.0" y="385" fill="currentColor" font-size="14" text-anchor="middle">displacement a</text>
<text x="58" y="200.0" fill="currentColor" font-size="14" text-anchor="middle" transform="rotate(-90 58 200.0)">divergence value</text>
<polyline points="100.0,340.0 106.2,336.6 112.4,333.2 118.5,329.9 124.7,326.5 130.9,323.1 137.1,319.7 143.2,316.3 149.4,313.0 155.6,309.6 161.8,306.2 168.0,302.8 174.1,299.4 180.3,296.1 186.5,292.7 192.7,289.3 198.9,285.9 205.0,282.5 211.2,279.2 217.4,275.8 223.6,272.4 229.7,269.0 235.9,265.7 242.1,262.3 248.3,258.9 254.5,255.5 260.6,252.1 266.8,248.8 273.0,245.4 279.2,242.0 285.4,238.6 291.5,235.2 297.7,231.9 303.9,228.5 310.1,225.1 316.2,221.7 322.4,218.3 328.6,215.0 334.8,211.6 341.0,208.2 347.1,204.8 353.3,201.4 359.5,198.1 365.7,194.7 371.9,191.3 378.0,187.9 384.2,184.5 390.4,181.2 396.6,177.8 402.7,174.4 408.9,171.0 415.1,167.6 421.3,164.3 427.5,160.9 433.6,157.5 439.8,154.1 446.0,150.8 452.2,147.4 458.3,144.0 464.5,140.6 470.7,137.2 476.9,133.9 483.1,130.5 489.2,127.1 495.4,123.7 501.6,120.3 507.8,117.0 514.0,113.6 520.1,110.2 526.3,106.8 532.5,103.4 538.7,100.1 544.8,96.7 551.0,93.3 557.2,89.9 563.4,86.5 569.6,83.2 575.7,79.8 581.9,76.4 588.1,73.0" fill="none" stroke="#3b82f6" stroke-width="2.4"/>
<polyline points="100.0,340.0 106.2,336.6 112.4,333.2 118.5,329.9 124.7,326.5 130.9,323.1 137.1,319.7 143.2,316.3 149.4,313.0 155.6,309.6 161.8,306.2 168.0,302.8 174.1,299.4 180.3,296.1 186.5,292.7 192.7,289.3 198.9,285.9 205.0,282.5 211.2,279.2 217.4,275.8 223.6,272.4 229.7,269.0 235.9,265.7 242.1,262.3 248.3,258.9 254.5,255.5 260.6,252.1 266.8,248.8 273.0,245.4 279.2,242.0 285.4,238.6 291.5,235.2 297.7,231.9 303.9,228.5 310.1,225.1 316.2,221.7 322.4,218.3 328.6,215.0 334.8,211.6 341.0,209.8 347.1,209.8 353.3,209.8 359.5,209.8 365.7,209.8 371.9,209.8 378.0,209.8 384.2,209.8 390.4,209.8 396.6,209.8 402.7,209.8 408.9,209.8 415.1,209.8 421.3,209.8 427.5,209.8 433.6,209.8 439.8,209.8 446.0,209.8 452.2,209.8 458.3,209.8 464.5,209.8 470.7,209.8 476.9,209.8 483.1,209.8 489.2,209.8 495.4,209.8 501.6,209.8 507.8,209.8 514.0,209.8 520.1,209.8 526.3,209.8 532.5,209.8 538.7,209.8 544.8,209.8 551.0,209.8 557.2,209.8 563.4,209.8 569.6,209.8 575.7,209.8 581.9,209.8 588.1,209.8" fill="none" stroke="#10b981" stroke-width="2.4"/>
<polyline points="100.0,340.0 106.2,336.7 112.4,333.4 118.5,330.1 124.7,326.8 130.9,323.4 137.1,320.1 143.2,316.8 149.4,313.5 155.6,310.2 161.8,306.9 168.0,303.6 174.1,300.3 180.3,296.9 186.5,293.6 192.7,290.3 198.9,287.0 205.0,283.7 211.2,280.4 217.4,277.1 223.6,273.8 229.7,270.5 235.9,267.1 242.1,263.8 248.3,260.5 254.5,257.2 260.6,253.9 266.8,250.6 273.0,247.3 279.2,244.0 285.4,240.6 291.5,237.3 297.7,234.0 303.9,230.7 310.1,227.4 316.2,224.1 322.4,220.8 328.6,217.5 334.8,214.1 341.0,212.4 347.1,212.4 353.3,212.4 359.5,212.4 365.7,212.4 371.9,212.4 378.0,212.4 384.2,212.4 390.4,212.4 396.6,212.4 402.7,212.4 408.9,212.4 415.1,212.4 421.3,212.4 427.5,212.4 433.6,212.4 439.8,212.4 446.0,212.4 452.2,212.4 458.3,212.4 464.5,212.4 470.7,212.4 476.9,212.4 483.1,212.4 489.2,212.4 495.4,212.4 501.6,212.4 507.8,212.4 514.0,212.4 520.1,212.4 526.3,212.4 532.5,212.4 538.7,212.4 544.8,212.4 551.0,212.4 557.2,212.4 563.4,212.4 569.6,212.4 575.7,212.4 581.9,212.4 588.1,212.4" fill="none" stroke="#f59e0b" stroke-width="2.4" stroke-dasharray="7 4"/>
<line x1="114.3" y1="340" x2="114.3" y2="66" stroke="#ef4444" stroke-width="2"/>
<polygon points="109.3,66 119.3,66 114.3,54" fill="#ef4444"/>
<text x="126.3" y="64" fill="#ef4444" font-size="13">KL = infinity for every a &gt; 0</text>
<text x="568.2" y="81.5" fill="#3b82f6" font-size="13" text-anchor="end">W_1 = a</text>
<text x="481.0" y="199.8" fill="#10b981" font-size="13">TV = min(a,1)</text>
<text x="481.0" y="231.8" fill="#f59e0b" font-size="13">JS/log 2 = min(a,1)</text>
<text x="360.0" y="400" fill="currentColor" font-size="13.5" text-anchor="middle">Closed forms checked numerically (JS at a = 0.3: 0.207944 nats = 0.3 log 2). Once the supports</text>
<text x="360.0" y="419" fill="currentColor" font-size="13.5" text-anchor="middle">separate at a = 1 every f-divergence is pinned at its ceiling, so its slope in a is zero,</text>
<text x="360.0" y="438" fill="currentColor" font-size="13.5" text-anchor="middle">while W_1 keeps growing linearly. This picture is the whole argument for Wasserstein GANs.</text>
</svg>

**Worked Example 11.18 (a full table as overlap vanishes).** Let $P = \mathrm{Unif}[0,1]$ and $Q_a = \mathrm{Unif}[a,1+a]$ on $\mathbb{R}$, $a\ge0$. Closed forms, all verified numerically:
$$\mathrm{TV} = \min(a,1),\qquad \mathrm{JS} = \min(a,1)\log 2,\qquad D_{\mathrm{KL}}(P\Vert Q_a) = \begin{cases}0,& a=0\\ +\infty, & a>0\end{cases},\qquad W_1 = a.$$

| $a$ | TV | JS (nats) | JS$/\log2$ | $D_{\mathrm{KL}}$ | $W_1$ |
|---|---|---|---|---|---|
| $0$ | $0$ | $0$ | $0$ | $0$ | $0$ |
| $0.25$ | $0.25$ | $0.173287$ | $0.25$ | $\infty$ | $0.25$ |
| $0.50$ | $0.50$ | $0.346574$ | $0.50$ | $\infty$ | $0.50$ |
| $1.00$ | $1$ | $0.693147$ | $1$ | $\infty$ | $1.00$ |
| $1.50$ | $1$ | $0.693147$ | $1$ | $\infty$ | $1.50$ |
| $2.00$ | $1$ | $0.693147$ | $1$ | $\infty$ | $2.00$ |

Numerical check of the JS closed form at $a=0.3$: quadrature gives $0.207944$ nats, and $0.3\log 2 = 0.207944$. Numerical check of $W_1$ via the quantile formula (Theorem 11.42 with $201{,}001$ grid points): $0.299999$ at $a = 0.3$ and $1.499997$ at $a = 1.5$.

Note the three distinct behaviours. $D_{\mathrm{KL}}$ is useless immediately: infinite for every $a>0$, so it cannot even rank $a = 0.01$ against $a=10$. TV and JS are informative while the supports overlap and then die at $a=1$ — they measure overlap, and once overlap is zero there is nothing left to measure. $W_1$ keeps increasing forever, because it measures distance. If your generator's samples are currently $5$ units away from the data, $W_1$ tells you that and points downhill; JS says "$\log 2$", which is what it also says when you are $500$ units away.

> **Common Confusion 11.7.** *"So Wasserstein is simply the better divergence. Why would anyone use KL?"*
>
> Three reasons, all practical.
>
> **Estimation.** $W_1$ in $d$ dimensions suffers a curse of dimensionality: the empirical Wasserstein distance between $n$ samples and the truth converges at rate $n^{-1/d}$ in general, so in $d = 100$ you need astronomically many samples for the empirical $W_1$ to mean anything. KL-based objectives, when densities are available, do not have that problem. The WGAN literature works because the critic class is restricted, not because $W_1$ is easy to estimate.
>
> **Availability of densities.** If you have a model density (flows, autoregressive models, VAEs' ELBO), the KL-type objectives are directly computable and give unbiased stochastic gradients. Transport distances require solving an optimisation problem *inside* every gradient step.
>
> **The right notion of "close" is task-dependent.** Transport distances say two distributions are close if you can move a little mass a little way. That is exactly right for images (a one-pixel shift should be a small change) and exactly wrong for, say, molecular validity (a small perturbation of atom positions can make an invalid molecule). Meanwhile TV and KL are invariant to the geometry, which is right when the label space has no meaningful metric. There is no universally better divergence; there is a choice about what you want to be insensitive to.
>
> Also: the Lipschitz constraint in (11.4) is load-bearing, and weight clipping does not implement it faithfully. That is a known failure mode of the original WGAN, addressed (not solved) by gradient penalties.

### 11.7.4 Entropic regularisation in one paragraph

Solving the Kantorovich problem exactly costs $O(n^3\log n)$ for $n$ support points. Adding an entropy term makes it strictly convex and solvable by alternating projections: minimise $\int c\,\mathrm{d}\pi + \varepsilon D_{\mathrm{KL}}(\pi\Vert P\otimes Q)$ over $\pi\in\Pi(P,Q)$. The optimum has the form $\mathrm{d}\pi^\star = u(x)e^{-c(x,y)/\varepsilon}v(y)\,\mathrm{d}(P\otimes Q)$, and $u,v$ are found by Sinkhorn's algorithm — alternately rescaling rows and columns of the matrix $K_{ij} = e^{-c_{ij}/\varepsilon}$ so that the marginals come out right. Each iteration is a matrix–vector product, so it runs on a GPU, which is the entire reason entropic OT is used in machine learning. The price is bias: the regularised optimum overestimates the transport cost, by $O(\varepsilon\log(1/\varepsilon))$-ish, and the "Sinkhorn divergence" corrections exist to debias it.

**Worked Example 11.19 (Sinkhorn's bias, measured).** Take $P$ uniform on $\{0,1,2,3,4\}$ and $Q$ uniform on $\{1,2,3,4,5\}$, each atom of mass $0.2$, with $c(x,y)=|x-y|$. The exact answer is $W_1 = 1$ (shift everything right by one; equivalently, Theorem 11.42 with $F^{-1}(u) = G^{-1}(u)-1$). Running Sinkhorn to convergence ($20{,}000$ iterations, so the residual marginal error is negligible):

| $\varepsilon$ | $\langle \pi^\star, c\rangle$ |
|---|---|
| $1.00$ | $1.155913$ |
| $0.30$ | $1.001527$ |
| $0.10$ | $1.000000$ |
| $0.03$ | $1.000000$ |

At $\varepsilon = 1$ the entropic plan spreads mass over many pairs and overpays by $16\%$; by $\varepsilon = 0.1$ it has effectively recovered the exact plan on this small problem. In higher dimensions one cannot take $\varepsilon$ that small without numerical underflow in $e^{-c/\varepsilon}$ (hence log-domain Sinkhorn implementations), and the bias is the thing that has to be managed rather than eliminated.

> **Self-check 11.7.**
> 1. Compute $W_1(\delta_0,\delta_a)$ and $\mathrm{TV}(\delta_0,\delta_a)$ for $a\ne0$. Which one is informative about $a$?
> 2. Two empirical distributions of $n$ points each on the line. Describe the algorithm that computes $W_1$ between them exactly in $O(n\log n)$.
> 3. Why must the critic in (11.4) be constrained? What is the supremum without a constraint?
> 4. Give a task where transport geometry is the *wrong* notion of closeness.

## 11.8 Coding and compression

### 11.8.1 Kraft's inequality and the source coding theorem

Entropy was introduced axiomatically in §11.1. Its operational meaning — the one that licenses phrases like "this model compresses text to $0.9$ bits per character" — comes from coding theory, and it is worth having properly, because the "bits per dimension" numbers reported in generative modelling papers are exactly this quantity and are routinely misread.

Fix a finite source alphabet $\mathcal{X}$ and a code alphabet of size $D$ (usually $D=2$). A **code** assigns to each $x$ a finite string $C(x)$ of code symbols, with length $\ell(x)$. The code is **prefix-free** if no codeword is a prefix of another; prefix-free codes are exactly the ones you can decode as you read, with no lookahead.

**Theorem 11.45 (Kraft's inequality).** (a) Every prefix-free code satisfies $\sum_{x\in\mathcal{X}} D^{-\ell(x)}\le 1$. (b) Conversely, given any lengths $\{\ell(x)\}$ of positive integers with $\sum_x D^{-\ell(x)}\le1$, there exists a prefix-free code with those lengths.

*Proof.* (a) Let $L := \max_x \ell(x)$. Think of the $D$-ary tree of depth $L$; it has $D^L$ leaves. A codeword of length $\ell$ corresponds to a node at depth $\ell$, and the prefix-free condition says no codeword node is an ancestor of another. The subtree below a node at depth $\ell$ contains $D^{L-\ell}$ leaves, and the prefix condition makes these leaf sets pairwise disjoint. Counting leaves:
$$\sum_x D^{L-\ell(x)} \le D^L \quad\Longrightarrow\quad \sum_x D^{-\ell(x)}\le 1 .$$
(b) Sort the lengths increasingly, $\ell_1\le\ell_2\le\cdots\le\ell_m$. Assign codewords greedily: having assigned the first $i-1$, let $c_i := \sum_{j<i}D^{\ell_i - \ell_j}$, which counts the depth-$\ell_i$ nodes excluded by the previous choices. Kraft's sum bounds $\sum_{j\le i}D^{-\ell_j}\le1$, and the $i$-th term is strictly positive, so $\sum_{j<i}D^{-\ell_j}\le 1-D^{-\ell_i}$; multiplying by $D^{\ell_i}$ gives $c_i \le D^{\ell_i}-1 < D^{\ell_i}$, so at least one depth-$\ell_i$ node remains available; take it. By construction no assigned node is a descendant of an earlier one, so the code is prefix-free. $\blacksquare$

**Theorem 11.46 (source coding, single-symbol version).** Let $X\sim p$ on a finite $\mathcal{X}$ and let $L^\star := \min\{\mathbb{E}[\ell(X)] : \text{prefix-free codes}\}$. Then, with $H_D(X) := H(X)/\log D$ the entropy in base-$D$ units,
$$H_D(X) \ \le\ L^\star \ <\ H_D(X) + 1 .$$
Moreover, coding blocks of $n$ i.i.d. symbols gives $L_n^\star/n \to H_D(X)$.

*Proof.* **Lower bound.** Let $\ell$ be the lengths of any prefix-free code, put $c := \sum_x D^{-\ell(x)}\le1$ (Theorem 11.45a) and $q(x) := D^{-\ell(x)}/c$, a probability density on $\mathcal{X}$. Then
$$\mathbb{E}[\ell(X)] - H_D(X) = \sum_x p(x)\Bigl[\ell(x) + \log_D p(x)\Bigr] = \sum_x p(x)\log_D\frac{p(x)}{D^{-\ell(x)}} = \sum_x p(x)\log_D\frac{p(x)}{c\,q(x)},$$
which equals $\frac{1}{\log D}D_{\mathrm{KL}}(p\Vert q) + \log_D(1/c) \ge 0$ by Theorem 11.17 and $c\le1$. So $\mathbb{E}[\ell]\ge H_D(X)$.

**Upper bound.** Take the **Shannon lengths** $\ell(x) := \lceil -\log_D p(x)\rceil$. These satisfy Kraft, since $D^{-\ell(x)}\le p(x)$ and $\sum p = 1$, so by Theorem 11.45b a prefix code with these lengths exists, and
$$\mathbb{E}[\ell(X)] < \sum_x p(x)\bigl(-\log_D p(x) + 1\bigr) = H_D(X)+1 .$$

**Blocks.** Apply the two bounds to $X^n = (X_1,\dots,X_n)$, whose entropy is $nH_D(X)$ by the chain rule and independence. Then $nH_D(X)\le L^\star_n < nH_D(X)+1$, so $L^\star_n/n \to H_D(X)$ with error at most $1/n$. $\blacksquare$

The block statement is the resolution of Common Confusion 11.1, now with a proof: the overhead is at most one code symbol *per block*, hence at most $1/n$ per symbol. Worked Example 11.1's numbers, $1.875 \to 1.828125 \to 1.820313$ against $H = 1.811278$, are this $O(1/n)$ decay in action.

**A note on scope.** Classical information theory has a second half that this chapter deliberately omits: *channel coding* — the capacity $C = \max_{P_X} I(X;Y)$ of a noisy channel, and Shannon's noisy-channel coding theorem, which says that reliable communication is possible at every rate below $C$ and impossible above it. (The binary symmetric channel of Worked Example 11.11 and the concavity remark after Proposition 11.8 are the only appearances of a channel in these pages.) The omission is a judgement about payoff, not importance: the information theory that modern machine learning actually consumes runs almost entirely through source coding, divergences, and lower bounds, and Fano's inequality — historically a lemma in the converse half of the channel coding theorem — is developed in §11.9 for its statistical use directly. If you cross-reference Cover & Thomas you will meet capacity in their Chapter 7; nothing in this book depends on it.

### 11.8.2 Why blocks work: the asymptotic equipartition property

Theorem 11.46 says the per-symbol cost of block coding tends to $H$, but its proof is an accounting argument and does not explain *what* makes long blocks compressible. The reason is a concentration phenomenon, and stating it here does double duty: it makes source coding intuitive, and it is the first genuine appearance of the idea that Chapter 12 is built on.

**Proposition 11.47 (asymptotic equipartition property).** Let $X_1,X_2,\dots$ be i.i.d. with pmf $p$ on a finite alphabet, and write $p(x^n) := \prod_{i=1}^n p(x_i)$. Then
$$-\frac1n\log p(X_1,\dots,X_n)\ \xrightarrow{\ \mathbb{P}\ }\ H(X).$$
Consequently, for every $\varepsilon>0$ the **typical set** $A_n^\varepsilon := \bigl\{x^n : \bigl|-\tfrac1n\log p(x^n)-H(X)\bigr|\le\varepsilon\bigr\}$ satisfies, for all $n$ large enough,
$$\mathbb{P}\bigl(X^n\in A_n^\varepsilon\bigr)\ \ge\ 1-\varepsilon, \qquad (1-\varepsilon)e^{n(H-\varepsilon)}\ \le\ \bigl|A_n^\varepsilon\bigr|\ \le\ e^{n(H+\varepsilon)} .$$

*Proof.* The random variables $-\log p(X_i)$ are i.i.d. with mean $\mathbb{E}[-\log p(X)] = H(X)$, finite because the alphabet is finite, and $-\frac1n\log p(X^n) = \frac1n\sum_i\bigl(-\log p(X_i)\bigr)$ is their average. The weak law of large numbers (Ch. 4) gives convergence in probability to $H(X)$, which is the first display, and the probability bound follows by taking $n$ large enough that the deviation probability is below $\varepsilon$.

For the size bounds, note that $x^n\in A_n^\varepsilon$ means precisely $e^{-n(H+\varepsilon)}\le p(x^n)\le e^{-n(H-\varepsilon)}$. Upper bound:
$$1\ \ge\ \sum_{x^n\in A_n^\varepsilon}p(x^n)\ \ge\ \bigl|A_n^\varepsilon\bigr|\,e^{-n(H+\varepsilon)} \quad\Longrightarrow\quad \bigl|A_n^\varepsilon\bigr|\le e^{n(H+\varepsilon)} .$$
Lower bound, for $n$ large enough that $\mathbb{P}(A_n^\varepsilon)\ge1-\varepsilon$:
$$1-\varepsilon\ \le\ \sum_{x^n\in A_n^\varepsilon}p(x^n)\ \le\ \bigl|A_n^\varepsilon\bigr|\,e^{-n(H-\varepsilon)} \quad\Longrightarrow\quad \bigl|A_n^\varepsilon\bigr|\ge(1-\varepsilon)e^{n(H-\varepsilon)}. \ \blacksquare$$

Here is the picture the proposition licenses, and it is the right mental model for compression. The alphabet has $|\mathcal{X}|^n$ possible sequences, but almost all of the probability sits on a set of size about $e^{nH}$ — an exponentially *smaller* fraction, since $H<\log|\mathcal{X}|$ unless $p$ is uniform — and within that set the sequences are all roughly equally likely, which is where the name comes from. Compression is then obvious: index the typical set, which needs about $nH$ nats, and handle the rest with a flag and a wasteful fallback code whose contribution to the average vanishes.

For the three-flip variable of Worked Example 11.1, $H = 1.255482$ nats. With $n = 1000$ and $\varepsilon = 0.1$, the typical set contains between $2^{1667}$ and $2^{1956}$ sequences out of $4^{1000} = 2^{2000}$ — so a fraction of at most $2^{-44}$ of all sequences carries at least $90\%$ of the probability. The source coding theorem's $H$ is the exponent of that set's size.

The reason to flag this now is the shape of the argument, not the conclusion. We took a sum of $n$ independent terms and concluded it concentrates near its mean, then converted a probabilistic statement into a *counting* statement about a set in a high-dimensional space. That two-step move — concentrate, then count — is the entire method of Chapter 12, where the sums are Lipschitz functions rather than averages and the sets live on spheres rather than in sequence space.

### 11.8.3 Codes, wrong codes, and bits per dimension

**Proposition 11.48 (the redundancy of a wrong code is exactly a KL divergence).** Suppose you build an (idealised, fractional-length) code for the distribution $q$, assigning length $-\log_D q(x)$, but the data is actually distributed as $p$. Then
$$\underbrace{\mathbb{E}_{p}\bigl[-\log_D q(X)\bigr]}_{\text{bits you actually pay}} - \underbrace{H_D(p)}_{\text{bits you could have paid}} = \frac{D_{\mathrm{KL}}(p\Vert q)}{\log D}.$$

*Proof.* This is Proposition 11.20 divided by $\log D$: cross-entropy minus entropy is KL. $\blacksquare$

That identity is the cleanest possible answer to "what does KL *mean*": it is the number of extra bits per symbol you pay for believing $q$ when the truth is $p$. It also makes the whole of §11.3 operational — training a language model to minimise cross-entropy is training it to minimise the compressed size of the training corpus, and the reported loss *is* a compression rate.

**Worked Example 11.20 (paying for the wrong model).** Let $p = (0.4,0.3,0.2,0.1)$ on four symbols, so $H(p) = 1.846439$ bits. Suppose you wrongly believe the distribution is uniform, $q = (0.25,0.25,0.25,0.25)$, and use the corresponding code (all lengths $2$). You pay $\mathbb{E}_p[\ell] = 2$ bits per symbol. The excess is $2 - 1.846439 = 0.153561$ bits, and indeed
$$D_{\mathrm{KL}}(p\Vert q) = 0.153561\ \text{bits},$$
matching to six decimals. For comparison, the actual Huffman code for $p$ has lengths $(1,2,3,3)$ and expected length $1.9$ bits — worse than the ideal $1.846$ by $0.054$ bits (integer-length overhead), better than the wrong-model code by $0.1$ bits. Two distinct sources of loss, cleanly separated: **integer rounding** (Theorem 11.46's "$+1$") and **model error** (Proposition 11.48's KL). Shannon lengths for $p$, incidentally, give $(2,2,3,4)$ and expected length $2.4$ bits, worse than Huffman — Shannon's construction is used because it is easy to analyse, not because it is optimal.

#### Perplexity: the same number, exponentiated

Language-model papers rarely report the per-token cross-entropy directly; they report its exponential. For a model $q$ evaluated on a token sequence $w_1,\dots,w_n$, define the per-token cross-entropy $\mathrm{CE} := -\tfrac1n\sum_{t=1}^n \log q(w_t\mid w_{<t})$ (an empirical estimate of the conditional entropy rate plus a KL, by Proposition 11.20 and the chain-rule remark after Theorem 11.6) and the **perplexity**
$$\mathrm{PPL} := e^{\mathrm{CE}\ (\text{nats})} = 2^{\mathrm{CE}\ (\text{bits})}.$$
The interpretation that makes it stick: perplexity is the **effective branching factor**. A uniform choice among $k$ alternatives has cross-entropy $\log k$ and hence perplexity exactly $k$, so “perplexity $20$” means “per token, exactly as uncertain as a fair choice among $20$ options.” Computed anchors: $\mathrm{CE} = 3.0$ nats/token is $\mathrm{PPL} = 20.0855$; $\mathrm{CE} = 3.2$ is $24.5325$; a no-model uniform baseline over a $50{,}257$-token vocabulary (GPT-2's) has $\mathrm{CE} = \log 50257 = 10.8249$ nats and $\mathrm{PPL} = 50257$ exactly — the analogue of the $8$-bpd baseline below; the noisy-label loss floor of Worked Example 11.9 ($0.610864$ nats) is a perplexity *floor* of $1.842022$, above $1$ for the same reason the loss floor is above $0$; and §11.8.1's “$0.9$ bits per character” is a per-character perplexity of $2^{0.9} = 1.8661$.

Two caveats that this chapter equips you to state precisely. First, perplexity inherits every convention issue in the checklist of Common Confusion 11.8 below: per *what* (token, word, and character perplexities are not interchangeable), and computed from nats or bits (the CE changes, the PPL does not, provided you exponentiate in the matching base). Second and more important: **perplexities are comparable only under the same tokenizer.** Changing the tokenizer changes the sequence of symbols being predicted — it changes the base measure, §11.2's lesson in a new costume — and a model with a larger vocabulary faces fewer, harder prediction events per unit of text. The convention-free comparison is bits per *byte* (or per character) of raw text: total code length divided by a tokenizer-independent denominator, which is exactly the bits-per-dimension move of Definition 11.49 applied to text. When two papers report perplexities over different vocabularies, convert both to bits per byte before concluding anything.

#### Bits per dimension, done properly

Now we can discharge the promise from Chapter 2's discussion of dequantisation, and give you a procedure for auditing a reported likelihood.

Image data is *discrete*: an 8-bit pixel takes one of $256$ values. Generative models are usually *continuous* densities. Comparing them requires care, and here is the arithmetic that makes it work.

**Definition 11.49 (bits per dimension).** Let $\tilde x\in\{0,1,\dots,255\}^d$ be discrete data and $\tilde P$ a probability mass function on that set. The **bits per dimension** of $\tilde x$ under $\tilde P$ is
$$\mathrm{bpd}(\tilde x) := \frac{-\log_2 \tilde P(\tilde x)}{d}.$$
By Theorem 11.46 and Proposition 11.48, $\mathbb{E}[\mathrm{bpd}]$ is (up to $\le 1/d$ bits of integer overhead) the average number of bits per pixel needed to store the data with a code matched to $\tilde P$, and it decomposes as $\frac1d\bigl[H_2(\text{data}) + D_{\mathrm{KL}}\bigr]$.

**Proposition 11.50 (scale conversion, and why reported bpd is an upper bound).** Let $p$ be a density on $\mathbb{R}^d$ modelling images rescaled to $x = \tilde x/s$ for a scale factor $s>0$ (typically $s=256$), and let
$$\tilde P(\tilde x) := \int_{[0,1)^d} p_{\tilde X}(\tilde x+u)\,\mathrm{d}u, \qquad p_{\tilde X}(\tilde x) := s^{-d}p(\tilde x/s),$$
be the probability mass the model assigns to the grid cell at $\tilde x$. Then:

1. **(Scale)** $\log p_{\tilde X}(\tilde x) = \log p(x) - d\log s$.
2. **(Dequantisation is a bound in the safe direction)** For each $\tilde x$,
$$\mathbb{E}_u\bigl[\log p_{\tilde X}(\tilde x+u)\bigr]\ \le\ \log\tilde P(\tilde x),$$
with equality if and only if $p_{\tilde X}$ is constant on the cell.
3. **(The reported number)** Define the **reported** bits per dimension of a model that quotes $\log p(x)$ for $x\in[0,1]^d$, with $s=256$, by
$$\mathrm{bpd}_{\mathrm{rep}} := -\frac{\log p(x)}{d\log 2} + 8 .$$
Then $\mathrm{bpd}_{\mathrm{rep}} \ge \mathrm{bpd}_{\mathrm{true}} := -\log_2\tilde P(\tilde x)/d$ whenever the quoted $\log p(x)$ is the dequantised average $\mathbb{E}_u[\log p_{\tilde X}(\tilde x+u)] + d\log s$; and $\mathrm{bpd}_{\mathrm{rep}} = \mathrm{bpd}_{\mathrm{true}}$ when $p$ is constant on cells.

*Proof.* (1) is the change-of-variables formula (Ch. 3) for $x\mapsto sx$, with $|\det J| = s^d$; equivalently Theorem 11.12 in log-density form.

(2) is Jensen's inequality (Ch. 4) for the concave logarithm: $\mathbb{E}_u[\log p_{\tilde X}(\tilde x+u)] \le \log\mathbb{E}_u[p_{\tilde X}(\tilde x+u)] = \log\tilde P(\tilde x)$. Jensen is strict for a strictly concave function unless the random variable is almost surely constant, which here means $p_{\tilde X}$ constant on the cell.

(3) Combine: with $\log p(x) = \mathbb{E}_u[\log p_{\tilde X}(\tilde x+u)]+d\log s$, part (2) gives $\log p(x) - d\log s \le \log\tilde P(\tilde x)$, so
$$\mathrm{bpd}_{\mathrm{rep}} = -\frac{\log p(x) - d\log s}{d\log 2} \ \ge\ -\frac{\log \tilde P(\tilde x)}{d\log2} = \mathrm{bpd}_{\mathrm{true}},$$
and $\log_2 256 = 8$ gives the stated form. Equality is the equality case of (2). $\blacksquare$

The direction in part (3) is the whole point, and it is easy to get backwards, so let me say it twice. The reported number is an **over**-estimate of the code length your model could actually achieve. That is the honest direction: you are claiming a compression rate you can definitely reach, not one you might reach. If instead a paper evaluates a continuous density *at* the integer grid points without dequantising, the reported number is not an upper bound, not a lower bound, and not a probability — it is a density value, and densities are unbounded above, so the number can be made arbitrarily good by concentrating spikes on the data.

Part (2) is the theoretical justification for **uniform dequantisation**: train the continuous model on $\tilde x + u$ with $u\sim\mathrm{Unif}[0,1)^d$, and the resulting average log-density is a *lower bound* on the log-likelihood of the discrete model, hence the reported bpd is an *upper* bound on the true achievable code length. That is the right direction for an honest claim — you are reporting something you can actually achieve — and it is why the dequantisation convention matters more than it looks.

**Worked Example 11.21 (auditing a reported number).** Take CIFAR-10, $d = 32\times32\times3 = 3072$.

*(a) From nats to bpd.* A flow model reports an average $\log p(x) = 6000$ nats with $x\in[0,1]^d$. Then
$$\mathrm{bpd} = -\frac{6000}{3072\times0.693147} + 8 = -2.8178 + 8 = 5.1822 .$$
A different model reports $8000$ nats: $\mathrm{bpd} = 4.2430$. So the second is better by $0.94$ bits per pixel.

*(b) Sanity limits.* Uniform noise on the 8-bit grid has $\tilde P(\tilde x) = 256^{-d}$, i.e. exactly $8$ bpd — that is the "no model at all" baseline, and it corresponds to $\log p(x)=0$ on the unit-cube scale (density $1$ everywhere on $[0,1]^d$). Any bpd above $8$ means your model is worse than storing raw pixels. Current strong CIFAR-10 results sit near $2.9$ bpd, which corresponds to
$$\log p(x) = (8-2.9)\times 3072\times\log 2 = 10859.7\ \text{nats}$$
on the $[0,1]^d$ scale. If a paper reports a log-likelihood far outside the range $[0,\ 17000]$ nats for CIFAR-10-sized data, the scale convention is almost certainly different from what you assume.

*(c) The trap.* Suppose someone reports $\log p(\tilde x) = -10000$ nats on the $[0,255]$ scale, and you compare it with $6000$ nats on the $[0,1]$ scale. The two differ by the constant $d\log 256 = 17034.79$ nats (Worked Example 11.7). Converting: $-10000$ nats on the $[0,255]$ scale corresponds to $-10000 + 17034.79 = 7034.79$ nats on the unit scale, i.e. $\mathrm{bpd} = 4.696$ — *better* than the $6000$-nat model, not $16000$ nats worse. The sign and the magnitude of the naive comparison are both wrong.

> **Common Confusion 11.8.** *"Model A reports log-likelihood $-3200$ and model B reports $-3100$, so B is better."*
>
> Only if the two numbers are computed with respect to the same base measure, on the same data scale, with the same dequantisation convention, and per the same unit (per image? per dimension? nats or bits?). All four have been silently varied in the published literature. The checklist:
>
> 1. **Base measure and scale.** $[0,1]^d$ or $[0,255]^d$? The gap is $d\log 256$, which is $17034.79$ nats for CIFAR-10 — orders of magnitude larger than any real modelling difference.
> 2. **Discrete or continuous.** A continuous density on discrete data can be made arbitrarily large by concentrating spikes on the grid points (there is no upper bound on a density). Reported "likelihoods" for continuous models on un-dequantised discrete data are meaningless; Proposition 11.50(2) is the fix.
> 3. **Per-dimension normalisation.** bpd divides by $d$ and converts to bits. A factor of $3072\log 2 = 2129$ separates the conventions.
> 4. **Importance-weighted or single-sample bound.** For VAEs the reported number is usually an ELBO — a *lower* bound — and tightening the bound (more importance samples) improves the number without changing the model.
>
> Bits per dimension exists precisely to make the comparison well posed. Use it, and state the dequantisation.

**MDL in one paragraph.** Minimum description length turns Proposition 11.48 into a model-selection principle: choose the model that minimises the total code length of *the model plus the data given the model*, $\ell(\text{model}) + \ell(\text{data}\mid\text{model})$. Since $\ell(\text{data}\mid\text{model}) \approx -\log_2 P_\theta(\text{data})$ and $\ell(\text{model})\approx$ the bits to specify $\theta$ at appropriate precision, minimising the sum reproduces, to leading order, the Bayesian Information Criterion $-\log P_{\hat\theta}(\text{data}) + \tfrac{k}{2}\log n$ for a $k$-parameter model — the $\tfrac{k}{2}\log n$ being the cost of specifying each parameter to precision $O(1/\sqrt n)$, which is the resolution at which the likelihood can distinguish parameter values (Chapter 8's Fisher information sets exactly that scale). MDL is thus not a separate philosophy but a coding restatement of penalised likelihood; its practical value is that it makes "model complexity" a number of bits rather than a vague quantity, and its practical weakness is that "the bits to specify a neural network" is not a well-defined quantity without a coding scheme, which is why MDL arguments about deep networks are suggestive rather than decisive.

> **Self-check 11.8.**
> 1. A source has $H(X) = 2.3$ bits. What is the smallest possible expected length of a binary prefix code for one symbol? For blocks of $10$ i.i.d. symbols, per symbol?
> 2. You use a code optimal for $q$ but the data is $p$, and measure an excess of $0.4$ bits per symbol. What is $D_{\mathrm{KL}}(p\Vert q)$?
> 3. A model reports $2.9$ bpd on $[0,1]^d$-scaled CIFAR-10. What is $-\log p(x)$ in nats?
> 4. Why can a continuous density give an unboundedly large "likelihood" on discrete data, and what does dequantisation do about it?

## 11.9 Fano's inequality and lower bounds on learning

### 11.9.1 You cannot decode what was never transmitted

Every result so far has been about *describing* distributions. This section is about *impossibility*, and it is the reason information theory appears in statistics papers that never mention compression. The logic is one sentence long: if a learning problem requires distinguishing $M$ possibilities, and the data carries less than $\log M$ nats of information about which one is true, then no procedure — however clever, however overparameterised — can succeed. Fano's inequality makes that sentence quantitative.

**Theorem 11.51 (Fano's inequality).** Let $X$ take values in a finite set $\mathcal{X}$ with $|\mathcal{X}| = M \ge 2$, let $X \to Y \to \hat X$ be a Markov chain (so $\hat X$ is any estimator computed from $Y$ alone, possibly randomised), and let $P_{\mathrm{e}} := \mathbb{P}(\hat X \ne X)$. Then
$$H(X\mid Y) \ \le\ H_2(P_{\mathrm{e}}) + P_{\mathrm{e}}\log(M-1), \tag{11.6}$$
where $H_2(t) = -t\log t - (1-t)\log(1-t)$. If moreover $X$ is uniform on $\mathcal{X}$,
$$P_{\mathrm{e}} \ \ge\ 1 - \frac{I(X;Y) + \log 2}{\log M}. \tag{11.7}$$

*Proof.* Let $E := \mathbf{1}\{\hat X \ne X\}$. Expand $H(E,X\mid \hat X)$ two ways with the chain rule (Theorem 11.6, conditional version). First,
$$H(E,X\mid\hat X) = H(X\mid \hat X) + H(E\mid X,\hat X) = H(X\mid\hat X),$$
because $E$ is a deterministic function of the pair $(X,\hat X)$, so its conditional entropy given both is zero. Second,
$$H(E,X\mid\hat X) = H(E\mid\hat X) + H(X\mid E,\hat X) \le H_2(P_{\mathrm{e}}) + H(X\mid E,\hat X),$$
using $H(E\mid \hat X)\le H(E) = H_2(P_{\mathrm{e}})$ (Proposition 11.7, conditioning reduces entropy). For the last term, split on $E$:
$$H(X\mid E,\hat X) = (1-P_{\mathrm{e}})\,H(X\mid E=0,\hat X) + P_{\mathrm{e}}\,H(X\mid E=1,\hat X) \le (1-P_{\mathrm{e}})\cdot 0 + P_{\mathrm{e}}\log(M-1),$$
since on $\{E=0\}$ we have $X = \hat X$ (no uncertainty), and on $\{E=1\}$ the variable $X$ is confined to $\mathcal{X}\setminus\{\hat X\}$, a set of $M-1$ elements, so Proposition 11.2 caps its entropy at $\log(M-1)$. Combining the two expansions gives $H(X\mid\hat X) \le H_2(P_{\mathrm{e}}) + P_{\mathrm{e}}\log(M-1)$.

Finally, the Markov chain $X\to Y\to\hat X$ and the data processing inequality (Theorem 11.28) give $I(X;\hat X)\le I(X;Y)$, i.e. $H(X\mid \hat X)\ge H(X\mid Y)$, which yields (11.6).

For (11.7): with $X$ uniform, $H(X) = \log M$, so $H(X\mid Y) = \log M - I(X;Y)$. Bound $H_2(P_{\mathrm{e}})\le\log 2$ and $\log(M-1)\le \log M$ in (11.6):
$$\log M - I(X;Y) \le \log 2 + P_{\mathrm{e}}\log M \quad\Longrightarrow\quad P_{\mathrm{e}} \ge \frac{\log M - I(X;Y) - \log2}{\log M}. \ \blacksquare$$

Read (11.7) as a budget. You need $\log M$ nats to pin down the answer; the channel supplies $I(X;Y)$; the shortfall, divided by $\log M$, is a floor on your error probability. The $\log 2$ is the price of the binary "did I get it right?" variable and is negligible when $M$ is large.

**Worked Example 11.22 (the budget, numerically).** All entries from (11.7).

| $M$ | $\log M$ (nats) | $I(X;Y)$ (nats) | error floor $P_{\mathrm{e}}\ge$ |
|---|---|---|---|
| $8$ | $2.079$ | $0.1$ | $0.6186$ |
| $10$ | $2.303$ | $0.5$ | $0.4818$ |
| $10$ | $2.303$ | $1.0$ | $0.2647$ |
| $1000$ | $6.908$ | $0.5$ | $0.8273$ |
| $1000$ | $6.908$ | $1.0$ | $0.7549$ |
| $1000$ | $6.908$ | $3.0$ | $0.4654$ |

<svg viewBox="0 0 760 460" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<g stroke="currentColor" fill="none" stroke-width="1.2">
<path d="M 90 330 L 710 330"/><path d="M 90 330 L 90 55"/>
</g>
<path d="M 86 330.0 L 90 330.0" stroke="currentColor" stroke-width="1.2"/>
<text x="80" y="334.0" font-size="12" fill="currentColor" text-anchor="end">0</text>
<path d="M 86 262.5 L 90 262.5" stroke="currentColor" stroke-width="1.2"/>
<text x="80" y="266.5" font-size="12" fill="currentColor" text-anchor="end">0.25</text>
<path d="M 90 262.5 L 710 262.5" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 4" opacity="0.3"/>
<path d="M 86 195.0 L 90 195.0" stroke="currentColor" stroke-width="1.2"/>
<text x="80" y="199.0" font-size="12" fill="currentColor" text-anchor="end">0.50</text>
<path d="M 90 195.0 L 710 195.0" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 4" opacity="0.3"/>
<path d="M 86 127.5 L 90 127.5" stroke="currentColor" stroke-width="1.2"/>
<text x="80" y="131.5" font-size="12" fill="currentColor" text-anchor="end">0.75</text>
<path d="M 90 127.5 L 710 127.5" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 4" opacity="0.3"/>
<path d="M 86 60.0 L 90 60.0" stroke="currentColor" stroke-width="1.2"/>
<text x="80" y="64.0" font-size="12" fill="currentColor" text-anchor="end">1.00</text>
<path d="M 90 60.0 L 710 60.0" stroke="currentColor" stroke-width="0.5" stroke-dasharray="3 4" opacity="0.3"/>
<path d="M 90.0 330 L 90.0 334" stroke="currentColor" stroke-width="1.2"/>
<text x="90.0" y="350" font-size="12" fill="currentColor" text-anchor="middle">0</text>
<path d="M 245.0 330 L 245.0 334" stroke="currentColor" stroke-width="1.2"/>
<text x="245.0" y="350" font-size="12" fill="currentColor" text-anchor="middle">0.25</text>
<path d="M 400.0 330 L 400.0 334" stroke="currentColor" stroke-width="1.2"/>
<text x="400.0" y="350" font-size="12" fill="currentColor" text-anchor="middle">0.50</text>
<path d="M 555.0 330 L 555.0 334" stroke="currentColor" stroke-width="1.2"/>
<text x="555.0" y="350" font-size="12" fill="currentColor" text-anchor="middle">0.75</text>
<path d="M 710.0 330 L 710.0 334" stroke="currentColor" stroke-width="1.2"/>
<text x="710.0" y="350" font-size="12" fill="currentColor" text-anchor="middle">1.00</text>
<text x="400" y="372" font-size="13" fill="currentColor" text-anchor="middle">information supplied, as a fraction of what is needed:  I(X;Y) / log M</text>
<text x="26" y="195" font-size="13" fill="currentColor" text-anchor="middle" transform="rotate(-90 26 195)">error-probability floor</text>
<polyline points="90.0,141.3 91.5,142.0 93.1,142.6 94.7,143.3 96.2,144.0 97.8,144.7 99.3,145.3 100.8,146.0 102.4,146.7 104.0,147.4 105.5,148.0 107.0,148.7 108.6,149.4 110.2,150.1 111.7,150.7 113.2,151.4 114.8,152.1 116.3,152.8 117.9,153.4 119.5,154.1 121.0,154.8 122.5,155.5 124.1,156.1 125.7,156.8 127.2,157.5 128.8,158.2 130.3,158.8 131.8,159.5 133.4,160.2 134.9,160.9 136.5,161.5 138.1,162.2 139.6,162.9 141.2,163.6 142.7,164.2 144.2,164.9 145.8,165.6 147.3,166.3 148.9,166.9 150.4,167.6 152.0,168.3 153.6,169.0 155.1,169.6 156.7,170.3 158.2,171.0 159.8,171.7 161.3,172.3 162.9,173.0 164.4,173.7 165.9,174.4 167.5,175.0 169.1,175.7 170.6,176.4 172.2,177.1 173.7,177.7 175.2,178.4 176.8,179.1 178.4,179.8 179.9,180.4 181.4,181.1 183.0,181.8 184.6,182.5 186.1,183.1 187.7,183.8 189.2,184.5 190.8,185.2 192.3,185.8 193.9,186.5 195.4,187.2 196.9,187.9 198.5,188.5 200.1,189.2 201.6,189.9 203.1,190.6 204.7,191.2 206.2,191.9 207.8,192.6 209.4,193.3 210.9,193.9 212.4,194.6 214.0,195.3 215.6,196.0 217.1,196.6 218.7,197.3 220.2,198.0 221.8,198.7 223.3,199.3 224.8,200.0 226.4,200.7 227.9,201.4 229.5,202.0 231.1,202.7 232.6,203.4 234.2,204.1 235.7,204.7 237.2,205.4 238.8,206.1 240.3,206.8 241.9,207.4 243.4,208.1 245.0,208.8 246.6,209.5 248.1,210.1 249.7,210.8 251.2,211.5 252.8,212.2 254.3,212.8 255.9,213.5 257.4,214.2 259.0,214.9 260.5,215.5 262.1,216.2 263.6,216.9 265.1,217.6 266.7,218.2 268.2,218.9 269.8,219.6 271.4,220.3 272.9,220.9 274.4,221.6 276.0,222.3 277.5,223.0 279.1,223.6 280.6,224.3 282.2,225.0 283.8,225.7 285.3,226.3 286.9,227.0 288.4,227.7 290.0,228.4 291.5,229.0 293.1,229.7 294.6,230.4 296.1,231.1 297.7,231.7 299.2,232.4 300.8,233.1 302.4,233.8 303.9,234.4 305.5,235.1 307.0,235.8 308.5,236.5 310.1,237.1 311.6,237.8 313.2,238.5 314.8,239.2 316.3,239.8 317.9,240.5 319.4,241.2 320.9,241.9 322.5,242.5 324.1,243.2 325.6,243.9 327.1,244.6 328.7,245.2 330.2,245.9 331.8,246.6 333.4,247.3 334.9,247.9 336.5,248.6 338.0,249.3 339.6,250.0 341.1,250.6 342.6,251.3 344.2,252.0 345.8,252.7 347.3,253.3 348.8,254.0 350.4,254.7 351.9,255.4 353.5,256.0 355.1,256.7 356.6,257.4 358.1,258.1 359.7,258.7 361.2,259.4 362.8,260.1 364.4,260.8 365.9,261.4 367.4,262.1 369.0,262.8 370.6,263.5 372.1,264.1 373.7,264.8 375.2,265.5 376.8,266.2 378.3,266.8 379.9,267.5 381.4,268.2 383.0,268.9 384.5,269.5 386.1,270.2 387.6,270.9 389.1,271.6 390.7,272.2 392.2,272.9 393.8,273.6 395.4,274.3 396.9,274.9 398.4,275.6 400.0,276.3 401.6,277.0 403.1,277.6 404.7,278.3 406.2,279.0 407.8,279.7 409.3,280.3 410.8,281.0 412.4,281.7 413.9,282.4 415.5,283.0 417.0,283.7 418.6,284.4 420.1,285.1 421.7,285.7 423.2,286.4 424.8,287.1 426.3,287.8 427.9,288.4 429.4,289.1 431.0,289.8 432.6,290.5 434.1,291.1 435.6,291.8 437.2,292.5 438.8,293.2 440.3,293.8 441.9,294.5 443.4,295.2 444.9,295.9 446.5,296.5 448.1,297.2 449.6,297.9 451.2,298.6 452.7,299.2 454.2,299.9 455.8,300.6 457.4,301.3 458.9,301.9 460.5,302.6 462.0,303.3 463.6,304.0 465.1,304.6 466.7,305.3 468.2,306.0 469.8,306.7 471.3,307.3 472.9,308.0 474.4,308.7 476.0,309.4 477.5,310.0 479.1,310.7 480.6,311.4 482.2,312.1 483.7,312.7 485.3,313.4 486.8,314.1 488.3,314.8 489.9,315.4 491.4,316.1 493.0,316.8 494.5,317.5 496.1,318.1 497.6,318.8 499.2,319.5 500.8,320.2 502.3,320.8 503.8,321.5 505.4,322.2 506.9,322.9 508.5,323.5 510.1,324.2 511.6,324.9 513.1,325.6 514.7,326.2 516.2,326.9 517.8,327.6 519.4,328.3 520.9,328.9 522.5,329.6 524.0,330.0 525.5,330.0 527.1,330.0 528.7,330.0 530.2,330.0 531.8,330.0 533.3,330.0 534.9,330.0 536.4,330.0 538.0,330.0 539.5,330.0 541.0,330.0 542.6,330.0 544.2,330.0 545.7,330.0 547.2,330.0 548.8,330.0 550.4,330.0 551.9,330.0 553.5,330.0 555.0,330.0 556.5,330.0 558.1,330.0 559.7,330.0 561.2,330.0 562.8,330.0 564.3,330.0 565.9,330.0 567.4,330.0 569.0,330.0 570.5,330.0 572.0,330.0 573.6,330.0 575.1,330.0 576.7,330.0 578.2,330.0 579.8,330.0 581.3,330.0 582.9,330.0 584.5,330.0 586.0,330.0 587.5,330.0 589.1,330.0 590.6,330.0 592.2,330.0 593.8,330.0 595.3,330.0 596.9,330.0 598.4,330.0 600.0,330.0 601.5,330.0 603.0,330.0 604.6,330.0 606.1,330.0 607.7,330.0 609.2,330.0 610.8,330.0 612.4,330.0 613.9,330.0 615.5,330.0 617.0,330.0 618.6,330.0 620.1,330.0 621.6,330.0 623.2,330.0 624.8,330.0 626.3,330.0 627.9,330.0 629.4,330.0 631.0,330.0 632.5,330.0 634.1,330.0 635.6,330.0 637.2,330.0 638.7,330.0 640.2,330.0 641.8,330.0 643.4,330.0 644.9,330.0 646.4,330.0 648.0,330.0 649.5,330.0 651.1,330.0 652.6,330.0 654.2,330.0 655.8,330.0 657.3,330.0 658.9,330.0 660.4,330.0 662.0,330.0 663.5,330.0 665.0,330.0 666.6,330.0 668.1,330.0 669.7,330.0 671.2,330.0 672.8,330.0 674.4,330.0 675.9,330.0 677.5,330.0 679.0,330.0 680.5,330.0 682.1,330.0 683.6,330.0 685.2,330.0 686.8,330.0 688.3,330.0 689.9,330.0 691.4,330.0 693.0,330.0 694.5,330.0 696.1,330.0 697.6,330.0 699.1,330.0 700.7,330.0 702.2,330.0 703.8,330.0 705.4,330.0 706.9,330.0 708.5,330.0 710.0,330.0" fill="none" stroke="#3b82f6" stroke-width="2.4"/>
<text x="98.0" y="133.3" font-size="12" fill="#3b82f6">M = 10</text>
<circle cx="523.4" cy="330.0" r="3.5" fill="#3b82f6"/>
<text x="525.4" y="318" font-size="11" fill="#3b82f6" text-anchor="middle">0.70</text>
<polyline points="90.0,100.6 91.5,101.3 93.1,102.0 94.7,102.7 96.2,103.3 97.8,104.0 99.3,104.7 100.8,105.4 102.4,106.0 104.0,106.7 105.5,107.4 107.0,108.1 108.6,108.7 110.2,109.4 111.7,110.1 113.2,110.8 114.8,111.4 116.3,112.1 117.9,112.8 119.5,113.5 121.0,114.1 122.5,114.8 124.1,115.5 125.7,116.2 127.2,116.8 128.8,117.5 130.3,118.2 131.8,118.9 133.4,119.5 134.9,120.2 136.5,120.9 138.1,121.6 139.6,122.2 141.2,122.9 142.7,123.6 144.2,124.3 145.8,124.9 147.3,125.6 148.9,126.3 150.4,127.0 152.0,127.6 153.6,128.3 155.1,129.0 156.7,129.7 158.2,130.3 159.8,131.0 161.3,131.7 162.9,132.4 164.4,133.0 165.9,133.7 167.5,134.4 169.1,135.1 170.6,135.7 172.2,136.4 173.7,137.1 175.2,137.8 176.8,138.4 178.4,139.1 179.9,139.8 181.4,140.5 183.0,141.1 184.6,141.8 186.1,142.5 187.7,143.2 189.2,143.8 190.8,144.5 192.3,145.2 193.9,145.9 195.4,146.5 196.9,147.2 198.5,147.9 200.1,148.6 201.6,149.2 203.1,149.9 204.7,150.6 206.2,151.3 207.8,151.9 209.4,152.6 210.9,153.3 212.4,154.0 214.0,154.6 215.6,155.3 217.1,156.0 218.7,156.7 220.2,157.3 221.8,158.0 223.3,158.7 224.8,159.4 226.4,160.0 227.9,160.7 229.5,161.4 231.1,162.1 232.6,162.7 234.2,163.4 235.7,164.1 237.2,164.8 238.8,165.4 240.3,166.1 241.9,166.8 243.4,167.5 245.0,168.1 246.6,168.8 248.1,169.5 249.7,170.2 251.2,170.8 252.8,171.5 254.3,172.2 255.9,172.9 257.4,173.5 259.0,174.2 260.5,174.9 262.1,175.6 263.6,176.2 265.1,176.9 266.7,177.6 268.2,178.3 269.8,178.9 271.4,179.6 272.9,180.3 274.4,181.0 276.0,181.6 277.5,182.3 279.1,183.0 280.6,183.7 282.2,184.3 283.8,185.0 285.3,185.7 286.9,186.4 288.4,187.0 290.0,187.7 291.5,188.4 293.1,189.1 294.6,189.7 296.1,190.4 297.7,191.1 299.2,191.8 300.8,192.4 302.4,193.1 303.9,193.8 305.5,194.5 307.0,195.1 308.5,195.8 310.1,196.5 311.6,197.2 313.2,197.8 314.8,198.5 316.3,199.2 317.9,199.9 319.4,200.5 320.9,201.2 322.5,201.9 324.1,202.6 325.6,203.2 327.1,203.9 328.7,204.6 330.2,205.3 331.8,205.9 333.4,206.6 334.9,207.3 336.5,208.0 338.0,208.6 339.6,209.3 341.1,210.0 342.6,210.7 344.2,211.3 345.8,212.0 347.3,212.7 348.8,213.4 350.4,214.0 351.9,214.7 353.5,215.4 355.1,216.1 356.6,216.7 358.1,217.4 359.7,218.1 361.2,218.8 362.8,219.4 364.4,220.1 365.9,220.8 367.4,221.5 369.0,222.1 370.6,222.8 372.1,223.5 373.7,224.2 375.2,224.8 376.8,225.5 378.3,226.2 379.9,226.9 381.4,227.5 383.0,228.2 384.5,228.9 386.1,229.6 387.6,230.2 389.1,230.9 390.7,231.6 392.2,232.3 393.8,232.9 395.4,233.6 396.9,234.3 398.4,235.0 400.0,235.6 401.6,236.3 403.1,237.0 404.7,237.7 406.2,238.3 407.8,239.0 409.3,239.7 410.8,240.4 412.4,241.0 413.9,241.7 415.5,242.4 417.0,243.1 418.6,243.7 420.1,244.4 421.7,245.1 423.2,245.8 424.8,246.4 426.3,247.1 427.9,247.8 429.4,248.5 431.0,249.1 432.6,249.8 434.1,250.5 435.6,251.2 437.2,251.8 438.8,252.5 440.3,253.2 441.9,253.9 443.4,254.5 444.9,255.2 446.5,255.9 448.1,256.6 449.6,257.2 451.2,257.9 452.7,258.6 454.2,259.3 455.8,259.9 457.4,260.6 458.9,261.3 460.5,262.0 462.0,262.6 463.6,263.3 465.1,264.0 466.7,264.7 468.2,265.3 469.8,266.0 471.3,266.7 472.9,267.4 474.4,268.0 476.0,268.7 477.5,269.4 479.1,270.1 480.6,270.7 482.2,271.4 483.7,272.1 485.3,272.8 486.8,273.4 488.3,274.1 489.9,274.8 491.4,275.5 493.0,276.1 494.5,276.8 496.1,277.5 497.6,278.2 499.2,278.8 500.8,279.5 502.3,280.2 503.8,280.9 505.4,281.5 506.9,282.2 508.5,282.9 510.1,283.6 511.6,284.2 513.1,284.9 514.7,285.6 516.2,286.3 517.8,286.9 519.4,287.6 520.9,288.3 522.5,289.0 524.0,289.6 525.5,290.3 527.1,291.0 528.7,291.7 530.2,292.3 531.8,293.0 533.3,293.7 534.9,294.4 536.4,295.0 538.0,295.7 539.5,296.4 541.0,297.1 542.6,297.7 544.2,298.4 545.7,299.1 547.2,299.8 548.8,300.4 550.4,301.1 551.9,301.8 553.5,302.5 555.0,303.1 556.5,303.8 558.1,304.5 559.7,305.2 561.2,305.8 562.8,306.5 564.3,307.2 565.9,307.9 567.4,308.5 569.0,309.2 570.5,309.9 572.0,310.6 573.6,311.2 575.1,311.9 576.7,312.6 578.2,313.3 579.8,313.9 581.3,314.6 582.9,315.3 584.5,316.0 586.0,316.6 587.5,317.3 589.1,318.0 590.6,318.7 592.2,319.3 593.8,320.0 595.3,320.7 596.9,321.4 598.4,322.0 600.0,322.7 601.5,323.4 603.0,324.1 604.6,324.7 606.1,325.4 607.7,326.1 609.2,326.8 610.8,327.4 612.4,328.1 613.9,328.8 615.5,329.5 617.0,330.0 618.6,330.0 620.1,330.0 621.6,330.0 623.2,330.0 624.8,330.0 626.3,330.0 627.9,330.0 629.4,330.0 631.0,330.0 632.5,330.0 634.1,330.0 635.6,330.0 637.2,330.0 638.7,330.0 640.2,330.0 641.8,330.0 643.4,330.0 644.9,330.0 646.4,330.0 648.0,330.0 649.5,330.0 651.1,330.0 652.6,330.0 654.2,330.0 655.8,330.0 657.3,330.0 658.9,330.0 660.4,330.0 662.0,330.0 663.5,330.0 665.0,330.0 666.6,330.0 668.1,330.0 669.7,330.0 671.2,330.0 672.8,330.0 674.4,330.0 675.9,330.0 677.5,330.0 679.0,330.0 680.5,330.0 682.1,330.0 683.6,330.0 685.2,330.0 686.8,330.0 688.3,330.0 689.9,330.0 691.4,330.0 693.0,330.0 694.5,330.0 696.1,330.0 697.6,330.0 699.1,330.0 700.7,330.0 702.2,330.0 703.8,330.0 705.4,330.0 706.9,330.0 708.5,330.0 710.0,330.0" fill="none" stroke="#10b981" stroke-width="2.4"/>
<text x="98.0" y="92.6" font-size="12" fill="#10b981">M = 100</text>
<circle cx="616.7" cy="330.0" r="3.5" fill="#10b981"/>
<text x="618.7" y="318" font-size="11" fill="#10b981" text-anchor="middle">0.85</text>
<polyline points="90.0,87.1 91.5,87.8 93.1,88.4 94.7,89.1 96.2,89.8 97.8,90.5 99.3,91.1 100.8,91.8 102.4,92.5 104.0,93.2 105.5,93.8 107.0,94.5 108.6,95.2 110.2,95.9 111.7,96.5 113.2,97.2 114.8,97.9 116.3,98.6 117.9,99.2 119.5,99.9 121.0,100.6 122.5,101.3 124.1,101.9 125.7,102.6 127.2,103.3 128.8,104.0 130.3,104.6 131.8,105.3 133.4,106.0 134.9,106.7 136.5,107.3 138.1,108.0 139.6,108.7 141.2,109.4 142.7,110.0 144.2,110.7 145.8,111.4 147.3,112.1 148.9,112.7 150.4,113.4 152.0,114.1 153.6,114.8 155.1,115.4 156.7,116.1 158.2,116.8 159.8,117.5 161.3,118.1 162.9,118.8 164.4,119.5 165.9,120.2 167.5,120.8 169.1,121.5 170.6,122.2 172.2,122.9 173.7,123.5 175.2,124.2 176.8,124.9 178.4,125.6 179.9,126.2 181.4,126.9 183.0,127.6 184.6,128.3 186.1,128.9 187.7,129.6 189.2,130.3 190.8,131.0 192.3,131.6 193.9,132.3 195.4,133.0 196.9,133.7 198.5,134.3 200.1,135.0 201.6,135.7 203.1,136.4 204.7,137.0 206.2,137.7 207.8,138.4 209.4,139.1 210.9,139.7 212.4,140.4 214.0,141.1 215.6,141.8 217.1,142.4 218.7,143.1 220.2,143.8 221.8,144.5 223.3,145.1 224.8,145.8 226.4,146.5 227.9,147.2 229.5,147.8 231.1,148.5 232.6,149.2 234.2,149.9 235.7,150.5 237.2,151.2 238.8,151.9 240.3,152.6 241.9,153.2 243.4,153.9 245.0,154.6 246.6,155.3 248.1,155.9 249.7,156.6 251.2,157.3 252.8,158.0 254.3,158.6 255.9,159.3 257.4,160.0 259.0,160.7 260.5,161.3 262.1,162.0 263.6,162.7 265.1,163.4 266.7,164.0 268.2,164.7 269.8,165.4 271.4,166.1 272.9,166.7 274.4,167.4 276.0,168.1 277.5,168.8 279.1,169.4 280.6,170.1 282.2,170.8 283.8,171.5 285.3,172.1 286.9,172.8 288.4,173.5 290.0,174.2 291.5,174.8 293.1,175.5 294.6,176.2 296.1,176.9 297.7,177.5 299.2,178.2 300.8,178.9 302.4,179.6 303.9,180.2 305.5,180.9 307.0,181.6 308.5,182.3 310.1,182.9 311.6,183.6 313.2,184.3 314.8,185.0 316.3,185.6 317.9,186.3 319.4,187.0 320.9,187.7 322.5,188.3 324.1,189.0 325.6,189.7 327.1,190.4 328.7,191.0 330.2,191.7 331.8,192.4 333.4,193.1 334.9,193.7 336.5,194.4 338.0,195.1 339.6,195.8 341.1,196.4 342.6,197.1 344.2,197.8 345.8,198.5 347.3,199.1 348.8,199.8 350.4,200.5 351.9,201.2 353.5,201.8 355.1,202.5 356.6,203.2 358.1,203.9 359.7,204.5 361.2,205.2 362.8,205.9 364.4,206.6 365.9,207.2 367.4,207.9 369.0,208.6 370.6,209.3 372.1,209.9 373.7,210.6 375.2,211.3 376.8,212.0 378.3,212.6 379.9,213.3 381.4,214.0 383.0,214.7 384.5,215.3 386.1,216.0 387.6,216.7 389.1,217.4 390.7,218.0 392.2,218.7 393.8,219.4 395.4,220.1 396.9,220.7 398.4,221.4 400.0,222.1 401.6,222.8 403.1,223.4 404.7,224.1 406.2,224.8 407.8,225.5 409.3,226.1 410.8,226.8 412.4,227.5 413.9,228.2 415.5,228.8 417.0,229.5 418.6,230.2 420.1,230.9 421.7,231.5 423.2,232.2 424.8,232.9 426.3,233.6 427.9,234.2 429.4,234.9 431.0,235.6 432.6,236.3 434.1,236.9 435.6,237.6 437.2,238.3 438.8,239.0 440.3,239.6 441.9,240.3 443.4,241.0 444.9,241.7 446.5,242.3 448.1,243.0 449.6,243.7 451.2,244.4 452.7,245.0 454.2,245.7 455.8,246.4 457.4,247.1 458.9,247.7 460.5,248.4 462.0,249.1 463.6,249.8 465.1,250.4 466.7,251.1 468.2,251.8 469.8,252.5 471.3,253.1 472.9,253.8 474.4,254.5 476.0,255.2 477.5,255.8 479.1,256.5 480.6,257.2 482.2,257.9 483.7,258.5 485.3,259.2 486.8,259.9 488.3,260.6 489.9,261.2 491.4,261.9 493.0,262.6 494.5,263.3 496.1,263.9 497.6,264.6 499.2,265.3 500.8,266.0 502.3,266.6 503.8,267.3 505.4,268.0 506.9,268.7 508.5,269.3 510.1,270.0 511.6,270.7 513.1,271.4 514.7,272.0 516.2,272.7 517.8,273.4 519.4,274.1 520.9,274.7 522.5,275.4 524.0,276.1 525.5,276.8 527.1,277.4 528.7,278.1 530.2,278.8 531.8,279.5 533.3,280.1 534.9,280.8 536.4,281.5 538.0,282.2 539.5,282.8 541.0,283.5 542.6,284.2 544.2,284.9 545.7,285.5 547.2,286.2 548.8,286.9 550.4,287.6 551.9,288.2 553.5,288.9 555.0,289.6 556.5,290.3 558.1,290.9 559.7,291.6 561.2,292.3 562.8,293.0 564.3,293.6 565.9,294.3 567.4,295.0 569.0,295.7 570.5,296.3 572.0,297.0 573.6,297.7 575.1,298.4 576.7,299.0 578.2,299.7 579.8,300.4 581.3,301.1 582.9,301.7 584.5,302.4 586.0,303.1 587.5,303.8 589.1,304.4 590.6,305.1 592.2,305.8 593.8,306.5 595.3,307.1 596.9,307.8 598.4,308.5 600.0,309.2 601.5,309.8 603.0,310.5 604.6,311.2 606.1,311.9 607.7,312.5 609.2,313.2 610.8,313.9 612.4,314.6 613.9,315.2 615.5,315.9 617.0,316.6 618.6,317.3 620.1,317.9 621.6,318.6 623.2,319.3 624.8,320.0 626.3,320.6 627.9,321.3 629.4,322.0 631.0,322.7 632.5,323.3 634.1,324.0 635.6,324.7 637.2,325.4 638.7,326.0 640.2,326.7 641.8,327.4 643.4,328.1 644.9,328.7 646.4,329.4 648.0,330.0 649.5,330.0 651.1,330.0 652.6,330.0 654.2,330.0 655.8,330.0 657.3,330.0 658.9,330.0 660.4,330.0 662.0,330.0 663.5,330.0 665.0,330.0 666.6,330.0 668.1,330.0 669.7,330.0 671.2,330.0 672.8,330.0 674.4,330.0 675.9,330.0 677.5,330.0 679.0,330.0 680.5,330.0 682.1,330.0 683.6,330.0 685.2,330.0 686.8,330.0 688.3,330.0 689.9,330.0 691.4,330.0 693.0,330.0 694.5,330.0 696.1,330.0 697.6,330.0 699.1,330.0 700.7,330.0 702.2,330.0 703.8,330.0 705.4,330.0 706.9,330.0 708.5,330.0 710.0,330.0" fill="none" stroke="#f59e0b" stroke-width="2.4"/>
<text x="98.0" y="79.1" font-size="12" fill="#f59e0b">M = 1000</text>
<circle cx="647.8" cy="330.0" r="3.5" fill="#f59e0b"/>
<text x="649.8" y="318" font-size="11" fill="#f59e0b" text-anchor="middle">0.90</text>
<text x="474.4" y="162.6" font-size="12" fill="currentColor">Fano forbids nothing</text>
<text x="474.4" y="178.6" font-size="12" fill="currentColor">to the right of the dots</text>
<text x="90" y="392" font-size="11.5" fill="currentColor" opacity="0.85">Figure 11.6  Fano's error floor as an information budget. Each curve is the bound</text>
<text x="90" y="407" font-size="11.5" fill="currentColor" opacity="0.85">Pe >= 1 - (I + log 2)/log M for a uniform target with M values, plotted against the fraction</text>
<text x="90" y="422" font-size="11.5" fill="currentColor" opacity="0.85">of the required log M nats that the data supplies. The dot marks where the bound hits zero,</text>
<text x="90" y="437" font-size="11.5" fill="currentColor" opacity="0.85">at I/log M = 1 - log 2/log M: past it Fano is vacuous, and the gap shrinks as M grows.</text>
</svg>

Two lessons. First, the floor is *non-trivial only when $I$ is small relative to $\log M$*: with $M = 1000$ you need roughly $\log 1000 = 6.9$ nats before Fano stops forbidding success, and $3$ nats still forces a $47\%$ error rate. Second, note how brutal the many-class case is — this is why Fano-style arguments are the standard tool for high-dimensional lower bounds, where $M$ grows exponentially in the dimension and $I$ grows only linearly in the sample size.

Before turning to statistics, here is the version of Fano that answers the question practitioners actually ask, which is not about minimax rates but about whether a given feature set can possibly work.

**Corollary 11.52 (information floor on classification accuracy).** Let $Y$ take $M\ge2$ values with *any* distribution, let $Z$ be any representation such that $Y\to X\to Z$ is a Markov chain, and let $\hat Y$ be any (possibly randomised) classifier computed from $Z$. Write $P_{\mathrm{e}} := \mathbb{P}(\hat Y\ne Y)$. Then

1. for $M\ge3$,
$$P_{\mathrm{e}}\ \ge\ \frac{H(Y)-I(Y;Z)-\log 2}{\log (M-1)};$$
2. for $M = 2$, $\ H_2(P_{\mathrm{e}})\ \ge\ H(Y)-I(Y;Z)$, so $P_{\mathrm{e}}\ \ge\ H_2^{-1}\bigl(H(Y\mid Z)\bigr)$, taking the root in $[0,\tfrac12]$.

*Proof.* Theorem 11.51's inequality (11.6) applies verbatim with $(X,Y,\hat X)$ replaced by $(Y,Z,\hat Y)$, since $Y\to Z\to\hat Y$ is a Markov chain: $H(Y\mid Z)\le H_2(P_{\mathrm{e}})+P_{\mathrm{e}}\log(M-1)$. Substituting $H(Y\mid Z) = H(Y)-I(Y;Z)$ and bounding $H_2(P_{\mathrm{e}})\le\log2$ gives (1) after dividing by $\log(M-1)>0$. For $M=2$ the term $\log(M-1)$ vanishes, so the inequality reads $H(Y\mid Z)\le H_2(P_{\mathrm{e}})$ directly; $H_2$ is increasing on $[0,\tfrac12]$, so it may be inverted there. $\blacksquare$

Note that (1) drops the uniform-prior assumption of (11.7) — it is $H(Y)$, not $\log M$, that measures how much there is to learn — and note that the binary case must be handled separately, because $\log(M-1) = 0$ makes (1) meaningless at $M=2$. That degeneracy is not a technicality: it is the statement that with two classes an error can never be "partially" correct, so the crude step $H_2(P_{\mathrm{e}})\le\log2$ throws away everything.

**Worked Example 11.23 (what the disease test can and cannot achieve).** Return to Chapter 1's screening example: prevalence $0.5\%$, sensitivity $99\%$, specificity $95\%$. Section 11.4 computed $H(D) = 0.045415$ bits, $I(D;T) = 0.020667$ bits, and therefore $H(D\mid T) = 0.024747$ bits. With $M = 2$, part (2) gives
$$P_{\mathrm{e}}\ \ge\ H_2^{-1}(0.024747\ \text{bits}) = 0.002446 .$$
So no rule whatsoever — no threshold, no model, no ensemble — can predict disease status from this single test with error below $0.2446\%$.

Is that a strong statement? It is worth being careful, because the naive reaction is "so the test is nearly perfect", and that is wrong. Compare three numbers. The trivial classifier "always predict healthy" has error $0.5\%$, the prevalence. The Fano floor is $0.2446\%$. And the *actual* Bayes-optimal rule for these parameters also predicts healthy for every outcome — because even after a positive test, $\mathbb{P}(D\mid+) = 9.05\% < 50\%$ — so its error is exactly $0.5\%$. The floor of $0.2446\%$ is therefore *not attained*: Fano tells us no rule beats $0.2446\%$, and separately we can see that no rule beats $0.5\%$ either. Fano gives a valid but loose obstruction here, and the looseness is precisely the $H_2(P_{\mathrm{e}})\le\log2$ step being applied when $P_{\mathrm{e}}$ is tiny rather than near $\tfrac12$.

Two lessons, and the second is the one to keep. First, error-rate floors from mutual information are one-sided: they say what is impossible, never what is achievable, so a comfortable gap between the floor and your model's error is not evidence that improvement is available. Second, in a severely imbalanced problem, accuracy is the wrong currency — the Bayes rule here achieves $99.5\%$ accuracy by never diagnosing anyone, which is why $I(D;T) = 45.5\%$ of $H(D)$ is the more honest summary of what the test is worth than any accuracy number.

### 11.9.2 From Fano to minimax lower bounds

Here is the translation into statistics. A minimax risk is
$$\mathfrak{M}_n(\Theta) := \inf_{\hat\theta}\ \sup_{\theta\in\Theta}\ \mathbb{E}_{\theta}\bigl[d(\hat\theta(X_1,\dots,X_n),\theta)\bigr],$$
the best worst-case error achievable from $n$ observations. Upper bounds come from exhibiting an estimator and controlling it (Chapter 4's law of large numbers and concentration tools). Lower bounds come from information theory, via the following reduction: *make the estimation problem contain a decoding problem*.

**Corollary 11.53 (Fano's method).** Let $\theta_1,\dots,\theta_M\in\Theta$ be $2s$-separated, i.e. $d(\theta_j,\theta_k)\ge 2s$ for $j\ne k$, and write $P_j^{(n)}$ for the law of the $n$ observations under $\theta_j$. Then
$$\mathfrak{M}_n(\Theta)\ \ge\ s\left(1 - \frac{\bar I + \log 2}{\log M}\right),\qquad \bar I \le \frac{1}{M^2}\sum_{j,k} D_{\mathrm{KL}}\bigl(P_j^{(n)}\Vert P_k^{(n)}\bigr) ,$$
where $\bar I := I(J; X_1,\dots,X_n)$ for $J$ uniform on $\{1,\dots,M\}$.

*Proof.* Draw $J$ uniformly and generate data from $P_J^{(n)}$. Given any estimator $\hat\theta$, define $\hat J := \arg\min_j d(\hat\theta,\theta_j)$ (ties broken arbitrarily). If $\hat J \ne J$ then, by $2s$-separation and the triangle inequality, $d(\hat\theta,\theta_J)\ge s$: otherwise $d(\hat\theta,\theta_J)<s$ would force $d(\theta_J,\theta_{\hat J}) \le d(\theta_J,\hat\theta)+d(\hat\theta,\theta_{\hat J}) < 2s$. Hence
$$\sup_\theta \mathbb{E}_\theta[d(\hat\theta,\theta)] \ \ge\ \max_j \mathbb{E}_{\theta_j}[d(\hat\theta,\theta_j)] \ \ge\ \mathbb{E}\bigl[d(\hat\theta,\theta_J)\bigr] \ \ge\ s\,\mathbb{P}(\hat J\ne J),$$
where the middle step uses max $\ge$ average over the uniform $J$. Now $J\to X^{(n)}\to\hat J$ is a Markov chain, so Theorem 11.51 applies to bound $\mathbb{P}(\hat J \ne J)$ from below by $1 - (\bar I+\log2)/\log M$. The stated bound on $\bar I$ is the standard variational fact that $I(J;X) = \frac1M\sum_j D_{\mathrm{KL}}(P_j\Vert \bar P)$ with $\bar P = \frac1M\sum_k P_k$, together with $D_{\mathrm{KL}}(P_j\Vert\bar P) \le \frac1M\sum_k D_{\mathrm{KL}}(P_j\Vert P_k)$ by convexity of KL in its second argument (Theorem 11.31). $\blacksquare$

The whole art of using Corollary 11.53 lies in constructing the $M$ points: they must be *far apart* in the loss metric (large $s$) yet *statistically close* (small pairwise KL), and $M$ must be large. In high dimensions one builds such packings from binary codes with large minimum Hamming distance; the existence of exponentially many such codewords is the Varshamov–Gilbert bound, which we quote in §11.11 rather than prove.

For the simplest case $M=2$ the Fano machinery is overkill and there is a cleaner statement, which is also where Pinsker earns its keep.

**Proposition 11.54 (Le Cam's two-point method).** Let $\theta_0,\theta_1\in\Theta$ with $d(\theta_0,\theta_1)\ge 2s$, and let $P_0,P_1$ be the corresponding laws of the data. Then
$$\mathfrak{M}_n(\Theta)\ \ge\ \frac{s}{2}\Bigl(1 - \mathrm{TV}(P_0,P_1)\Bigr)\ \ge\ \frac{s}{2}\left(1 - \sqrt{\tfrac12 D_{\mathrm{KL}}(P_0\Vert P_1)}\right).$$

*Proof.* As in Corollary 11.53, convert $\hat\theta$ into a test $\psi := \arg\min_{j\in\{0,1\}} d(\hat\theta,\theta_j)$, so that $\psi\ne j$ implies $d(\hat\theta,\theta_j)\ge s$. Then
$$\max_{j}\mathbb{E}_{j}[d(\hat\theta,\theta_j)] \ \ge\ s\max_j \mathbb{P}_j(\psi\ne j)\ \ge\ \frac{s}{2}\Bigl[\mathbb{P}_0(\psi\ne0)+\mathbb{P}_1(\psi\ne1)\Bigr]\ \ge\ \frac{s}{2}\bigl(1-\mathrm{TV}(P_0,P_1)\bigr),$$
the last step being exactly Proposition 11.37 — the sum of the two error probabilities of *any* test is at least $1 - \mathrm{TV}$, with equality for the likelihood-ratio test (Neyman–Pearson, Ch. 9). The second inequality is Pinsker (Theorem 11.38). $\blacksquare$

Notice the division of labour. Total variation is the *right* quantity — it is exactly the testing difficulty. But TV between two $n$-fold product measures is awkward to compute, whereas KL is additive over independent coordinates: $D_{\mathrm{KL}}(P_0^{\otimes n}\Vert P_1^{\otimes n}) = n\,D_{\mathrm{KL}}(P_0\Vert P_1)$. So the strategy is: work with KL because it tensorises, convert to TV with Pinsker because TV is what controls testing. That is the whole reason Pinsker's inequality is in every lower-bound proof you will ever read.

**Worked Example 11.24 (the $\sigma/\sqrt n$ rate is not an artefact of your estimator).** Let $X_1,\dots,X_n\stackrel{\text{iid}}{\sim}N(\theta,\sigma^2)$ with $\theta\in\mathbb{R}$ unknown, $\sigma$ known, and loss $d(\hat\theta,\theta)=|\hat\theta-\theta|$. Take $\theta_0 = 0$ and $\theta_1 = \delta$, so $s = \delta/2$. Then
$$D_{\mathrm{KL}}\bigl(P_0^{\otimes n}\Vert P_1^{\otimes n}\bigr) = n\cdot\frac{\delta^2}{2\sigma^2},\qquad\text{so}\qquad \mathrm{TV} \le \sqrt{\tfrac12\cdot\tfrac{n\delta^2}{2\sigma^2}} = \frac{\delta\sqrt n}{2\sigma}.$$
Choose $\delta := \sigma/\sqrt n$, which makes the Pinsker bound $\mathrm{TV}\le\tfrac12$. With $n = 100$, $\sigma=1$, so $\delta = 0.1$: the bound gives $\mathrm{TV}\le0.5$, while the exact value (Gaussian location families have a closed form, $\mathrm{TV} = 2\Phi(\delta\sqrt n/2\sigma)-1$) is $0.382925$. So the best possible sum of the two testing errors is $1 - 0.382925 = 0.617075$, and Proposition 11.54 gives
$$\mathfrak{M}_{100} \ \ge\ \frac{0.1}{4}\times 0.617075 = 0.01543 .$$
Halving the separation to $\delta = 0.05$ gives a Pinsker bound of $0.25$, exact $\mathrm{TV} = 0.197413$, hence a floor of $\frac{0.05}{4}\times0.802587 = 0.01003$ — a weaker bound, since we made the two hypotheses easier to tell apart than necessary. The tuning $\delta \asymp \sigma/\sqrt n$ is what maximises the product.

In general $\delta = c\sigma/\sqrt n$ gives $\mathfrak{M}_n \ge \frac{c\sigma}{4\sqrt n}\bigl(1-2\Phi(c/2)+1\bigr)$, i.e.
$$\mathfrak{M}_n \ \ge\ \frac{C\sigma}{\sqrt n}\quad\text{for an absolute constant }C>0.$$
Now pair this with the upper bound: the sample mean satisfies $\mathbb{E}|\bar X_n - \theta| = \sigma\sqrt{2/(\pi n)} \le \sigma/\sqrt n$ for every $\theta$, so $\mathfrak{M}_n \le \sigma\sqrt{2/(\pi n)}$. **Upper and lower bounds match in order**, and $n^{-1/2}$ is therefore the exact rate for this problem — not a limitation of the sample mean, but a property of the problem. For the $d$-dimensional mean with squared-error loss the same argument (applied coordinatewise, or via Fano with a packing of the cube) gives $\mathfrak{M}_n \asymp d\sigma^2/n$: with $d = 100$, $\sigma = 1$, that is $1.0$ at $n = 100$, $0.1$ at $n = 1000$, and $0.01$ at $n = 10{,}000$.

This is the pattern to internalise. A rate is *pinned* only when you have both halves: an estimator whose risk you can bound above (concentration, Chapter 4) and an information-theoretic argument forbidding anything better (Fano or Le Cam, this section). Papers that report only the first half have proved their method works; papers that report both have proved nothing else could work substantially better.

> **Self-check 11.9.**
> 1. In (11.7), what happens to the bound as $I(X;Y)\to\log M$? Is the bound still useful?
> 2. Why does the proof of Theorem 11.51 need the Markov chain assumption $X\to Y\to\hat X$? What would go wrong if $\hat X$ could see $X$?
> 3. In Corollary 11.53, why do we want the $\theta_j$ to be far apart *and* the $P_j$ to be close? What happens to the bound if the $P_j$ are far apart?
> 4. Explain in one sentence why the lower-bound argument uses KL for the tensorisation step but TV for the testing step.

## 11.10 Consolidated machine learning payoffs

The individual payoffs have appeared as they became available. This section does two things the earlier ones could not: it derives the rate–distortion decomposition that unifies the VAE family, and it applies the chapter's results critically to three popular claims — the information bottleneck, contrastive learning as MI maximisation, and MI-based feature selection. In each case the honest conclusion is more interesting than the slogan.

### 11.10.1 The ELBO is a rate–distortion trade-off

**Proposition 11.55 (rate, distortion, and the sandwich on mutual information).** Let $P_X$ be the data distribution with (Shannon or differential) entropy $H(X)$, let $q(z\mid x)$ be an encoder, $p(x\mid z)$ a decoder, and $p(z)$ a prior. Define
$$\underbrace{D := \mathbb{E}_{P_X}\mathbb{E}_{q(z\mid x)}\bigl[-\log p(x\mid z)\bigr]}_{\text{distortion}},\qquad \underbrace{R := \mathbb{E}_{P_X}\bigl[D_{\mathrm{KL}}(q(\cdot\mid x)\Vert p(\cdot))\bigr]}_{\text{rate}}.$$
Write $q(z) := \mathbb{E}_{P_X}[q(z\mid x)]$ for the aggregate posterior and $I_q(X;Z)$ for the mutual information under the joint $P_X(\mathrm{d}x)q(\mathrm{d}z\mid x)$. Then:

1. $-\mathbb{E}_{P_X}[\mathrm{ELBO}(x)] = D + R$.
2. $R = I_q(X;Z) + D_{\mathrm{KL}}(q(z)\Vert p(z)) \ \ge\ I_q(X;Z)$.
3. $D \ge H(X\mid Z) = H(X) - I_q(X;Z)$, hence $H(X) - D \le I_q(X;Z) \le R$.

*Proof.* (1) The ELBO from Chapter 2 is $\mathrm{ELBO}(x) = \mathbb{E}_{q(z\mid x)}[\log p(x\mid z)] - D_{\mathrm{KL}}(q(\cdot\mid x)\Vert p)$; average over $P_X$ and negate.

(2) Insert $q(z)$ and split the logarithm:
$$R = \mathbb{E}\Bigl[\log\frac{q(Z\mid X)}{p(Z)}\Bigr] = \mathbb{E}\Bigl[\log\frac{q(Z\mid X)}{q(Z)}\Bigr] + \mathbb{E}\Bigl[\log\frac{q(Z)}{p(Z)}\Bigr] = I_q(X;Z) + D_{\mathrm{KL}}(q\Vert p),$$
the first term being Definition 11.24's MI written as a KL between joint and product of marginals, the second an expectation over $Z\sim q(z)$. Non-negativity of KL (Theorem 11.17) gives the inequality, with equality exactly when the aggregate posterior matches the prior.

(3) Write $q(x\mid z)$ for the true conditional of the joint $P_X q(z\mid x)$. Then
$$D = \mathbb{E}\bigl[-\log p(X\mid Z)\bigr] = \mathbb{E}\bigl[-\log q(X\mid Z)\bigr] + \mathbb{E}\Bigl[\log\frac{q(X\mid Z)}{p(X\mid Z)}\Bigr] = H(X\mid Z) + \mathbb{E}_{q(z)}\bigl[D_{\mathrm{KL}}(q(\cdot\mid z)\Vert p(\cdot\mid z))\bigr],$$
and the second term is $\ge 0$. Substituting $H(X\mid Z) = H(X)-I_q(X;Z)$ (Proposition 11.25) and rearranging gives the sandwich. $\blacksquare$

<svg viewBox="0 0 720 470" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="ui-sans-serif, system-ui, sans-serif">
<text x="360.0" y="30" fill="currentColor" font-size="15" text-anchor="middle">Figure 11.7 — Rate-distortion curve of a N(0,1) source and the beta-VAE trade-off</text>
<line x1="100" y1="340" x2="600" y2="340" stroke="currentColor" stroke-width="1.2"/>
<line x1="100" y1="340" x2="100" y2="60" stroke="currentColor" stroke-width="1.2"/>
<line x1="100.0" y1="340" x2="100.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="100.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">0</text>
<line x1="215.7" y1="340" x2="215.7" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="215.7" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.25</text>
<line x1="331.5" y1="340" x2="331.5" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="331.5" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.5</text>
<line x1="447.2" y1="340" x2="447.2" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="447.2" y="362" fill="currentColor" font-size="13" text-anchor="middle">0.75</text>
<line x1="563.0" y1="340" x2="563.0" y2="346" stroke="currentColor" stroke-width="1"/>
<text x="563.0" y="362" fill="currentColor" font-size="13" text-anchor="middle">1</text>
<line x1="94" y1="340.0" x2="100" y2="340.0" stroke="currentColor" stroke-width="1"/>
<text x="90" y="344.0" fill="currentColor" font-size="13" text-anchor="end">0</text>
<line x1="94" y1="236.3" x2="100" y2="236.3" stroke="currentColor" stroke-width="1"/>
<text x="90" y="240.3" fill="currentColor" font-size="13" text-anchor="end">1</text>
<line x1="94" y1="132.6" x2="100" y2="132.6" stroke="currentColor" stroke-width="1"/>
<text x="90" y="136.6" fill="currentColor" font-size="13" text-anchor="end">2</text>
<text x="350.0" y="385" fill="currentColor" font-size="14" text-anchor="middle">distortion D (mean squared error)</text>
<text x="58" y="200.0" fill="currentColor" font-size="14" text-anchor="middle" transform="rotate(-90 58 200.0)">rate R(D)  (bits)</text>
<polyline points="113.0,72.5 116.2,89.2 119.4,102.8 122.7,114.4 125.9,124.3 129.2,133.1 132.4,141.0 135.6,148.2 138.9,154.7 142.1,160.6 145.3,166.2 148.6,171.3 151.8,176.2 155.0,180.7 158.3,185.0 161.5,189.0 164.8,192.9 168.0,196.5 171.2,200.0 174.5,203.3 177.7,206.5 180.9,209.6 184.2,212.5 187.4,215.3 190.7,218.0 193.9,220.7 197.1,223.2 200.4,225.6 203.6,228.0 206.8,230.3 210.1,232.5 213.3,234.7 216.6,236.8 219.8,238.9 223.0,240.9 226.3,242.8 229.5,244.7 232.7,246.6 236.0,248.4 239.2,250.1 242.5,251.8 245.7,253.5 248.9,255.2 252.2,256.8 255.4,258.3 258.6,259.9 261.9,261.4 265.1,262.9 268.4,264.3 271.6,265.8 274.8,267.2 278.1,268.5 281.3,269.9 284.5,271.2 287.8,272.5 291.0,273.8 294.3,275.0 297.5,276.3 300.7,277.5 304.0,278.7 307.2,279.9 310.4,281.0 313.7,282.2 316.9,283.3 320.2,284.4 323.4,285.5 326.6,286.6 329.9,287.6 333.1,288.7 336.3,289.7 339.6,290.7 342.8,291.7 346.1,292.7 349.3,293.7 352.5,294.7 355.8,295.6 359.0,296.6 362.2,297.5 365.5,298.4 368.7,299.3 372.0,300.2 375.2,301.1 378.4,302.0 381.7,302.8 384.9,303.7 388.1,304.5 391.4,305.4 394.6,306.2 397.9,307.0 401.1,307.8 404.3,308.6 407.6,309.4 410.8,310.2 414.0,311.0 417.3,311.7 420.5,312.5 423.8,313.2 427.0,314.0 430.2,314.7 433.5,315.5 436.7,316.2 439.9,316.9 443.2,317.6 446.4,318.3 449.7,319.0 452.9,319.7 456.1,320.4 459.4,321.1 462.6,321.7 465.8,322.4 469.1,323.0 472.3,323.7 475.6,324.3 478.8,325.0 482.0,325.6 485.3,326.3 488.5,326.9 491.7,327.5 495.0,328.1 498.2,328.7 501.5,329.3 504.7,329.9 507.9,330.5 511.2,331.1 514.4,331.7 517.6,332.3 520.9,332.9 524.1,333.4 527.4,334.0 530.6,334.6 533.8,335.1 537.1,335.7 540.3,336.2 543.5,336.8 546.8,337.3 550.0,337.9 553.3,338.4 556.5,338.9 559.7,339.5 563.0,340.0" fill="none" stroke="#3b82f6" stroke-width="2.4"/>
<circle cx="434.0" cy="315.6" r="5" fill="#10b981"/>
<polyline points="304.3,286.5 563.6,344.6" fill="none" stroke="#10b981" stroke-width="1.4" stroke-dasharray="5 4"/>
<text x="444.0" y="305.6" fill="#10b981" font-size="13">beta = 1: D = 0.721, R = 0.236</text>
<circle cx="267.0" cy="263.7" r="5" fill="#f59e0b"/>
<polyline points="137.3,205.6 396.6,321.8" fill="none" stroke="#f59e0b" stroke-width="1.4" stroke-dasharray="5 4"/>
<text x="277.0" y="253.7" fill="#f59e0b" font-size="13">beta = 2: D = 0.361, R = 0.736</text>
<circle cx="183.5" cy="211.9" r="5" fill="#ef4444"/>
<polyline points="113.9,149.5 313.1,328.0" fill="none" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="5 4"/>
<text x="193.5" y="201.9" fill="#ef4444" font-size="13">beta = 4: D = 0.180, R = 1.236</text>
<text x="557.0" y="328.0" fill="currentColor" font-size="12.5" text-anchor="end">R = 0: output the mean, D = Var = 1</text>
<text x="360.0" y="400" fill="currentColor" font-size="13.5" text-anchor="middle">R(D) = (1/2) log2(1/D) for D at most 1. A beta-VAE with KL weight beta lands where the curve</text>
<text x="360.0" y="419" fill="currentColor" font-size="13.5" text-anchor="middle">has slope -beta (in nats), so beta is a price paid for bits: raising it buys a lower rate at</text>
<text x="360.0" y="438" fill="currentColor" font-size="13.5" text-anchor="middle">strictly higher distortion. The three marked points are computed exactly, not eyeballed.</text>
</svg>

Three consequences worth stating explicitly.

**The $\beta$-VAE is a Lagrangian.** Training the $\beta$-VAE objective $D + \beta R$ is solving $\min_q D$ subject to $R \le R_0$, with $\beta$ the multiplier; sweeping $\beta$ traces the feasible frontier in the $(R,D)$ plane. Concretely, for the illustrative frontier in Figure 11.7: $\beta = 1$ lands at $(R,D) = (0.236, 0.721)$ bits, $\beta=2$ at $(0.736, 0.361)$, and $\beta = 4$ at $(1.236, 0.180)$. Higher $\beta$ buys a lower rate — fewer bits in the latent, more disentanglement pressure — at the cost of worse reconstruction. There is no single "best" point; there is a curve, and $\beta$ selects a point on it.

**The ELBO cannot distinguish models on the frontier.** Since $-\mathrm{ELBO} = D + R$, all points on the line $D + R = c$ have identical ELBO. The extreme case is $R = 0$: the encoder ignores $x$, the aggregate posterior equals the prior, $I_q(X;Z) = 0$ by part (2), and yet if the decoder is powerful enough (an autoregressive $p(x\mid z)$ that ignores $z$) the distortion can equal $H(X)$, giving $-\mathrm{ELBO} = H(X)$ — the same value achieved by a good lossless model with a genuinely informative latent. This is **posterior collapse**, and Proposition 11.55 shows it is not a bug in the optimiser but a point on the objective's own level set. If you care about the latent being informative, the ELBO alone does not express that preference; the rate does.

**Part (3) is a usable diagnostic.** Report $R$ and $D$ separately, not just their sum. If $R$ is near zero, the latent carries no information regardless of what the reconstructions look like. If $H(X) - D$ is far below $R$, the decoder is failing to exploit the information the encoder is transmitting.

### 11.10.2 The information bottleneck: what survives scrutiny

The information bottleneck proposes that a good representation $Z$ of input $X$ for predicting $Y$ solves
$$\min_{q(z\mid x)}\ I(X;Z) - \beta\, I(Z;Y),$$
compressing away input detail while keeping label-relevant detail. As an *objective* it is well posed and its stochastic variants (variational IB) are trainable. The stronger claim — that ordinary deep network training implicitly performs IB, passing through a "fitting" phase and then a "compression" phase visible in the $(I(X;Z), I(Z;Y))$ plane — needs this chapter's results to evaluate, and does not survive them intact.

> **Common Confusion 11.9.** *"The information bottleneck explains why deep learning generalises: the layers compress away irrelevant information, and you can see it in the information plane."*
>
> Take the claim apart with three results from this chapter.
>
> **What is $I(X;Z)$ for a deterministic network?** If $Z = f(X)$ with $f$ deterministic and continuous and $X$ has a continuous distribution, then $I(X;Z) = +\infty$ whenever $f$ is injective, by Theorem 11.26 — the same pathology as $I(X;X)=\infty$. If $X$ is discrete, $I(X;Z) = H(Z) \le H(X)$, and $I(X;Z) = H(X)$ for any injective $f$: a *constant*, independent of the weights, so it cannot decrease during training. Either way, the quantity plotted in an information plane for a deterministic network is not $I(X;Z)$.
>
> **What is being plotted, then?** An estimate obtained by binning the activations (or by adding noise). Worked Example 11.12 measured what binning does: with $\rho = 0.9$ Gaussians and four million samples, equal-mass binning reported $0.406$, $0.779$, $1.006$, $1.113$, $1.161$ bits for $m = 2,4,8,16,32$ bins against a truth of $1.198$ bits. The estimate is a monotone function of bin resolution relative to the activation *scale*. Weight decay shrinks activations; saturating nonlinearities compress them into a bounded range. Both reduce the number of occupied bins and therefore reduce the reported "information" with no change in any invariant quantity. The observed compression phase is, in substantial part, a measurement of activation scale.
>
> **Is the phenomenon universal?** Saxe and coauthors reproduced the experiments and found the two-phase picture depends on the nonlinearity: prominent with saturating $\tanh$ units, largely absent with ReLU units, and compression was neither necessary nor sufficient for good generalisation in their experiments.
>
> What survives: IB is a legitimate and useful *training objective* when the encoder is genuinely stochastic, in which case $I(X;Z)$ is finite and meaningful, and it is then essentially the rate term of Proposition 11.55. What does not survive: the claim that information-plane dynamics of deterministic networks explain generalisation.

### 11.10.3 Contrastive learning and the $\log K$ ceiling

InfoNCE, the objective behind much of self-supervised learning, maximises a lower bound on $I(X;Y)$ for positive pairs $(x,y)$ against $K-1$ negatives. Two facts from §11.5 govern how to read it.

First, the bound is capped: the InfoNCE objective cannot exceed $\log K$ nats regardless of the true MI. With the batch sizes in common use, the ceilings are $\log 64 = 4.1589$ nats ($6$ bits), $\log 512 = 6.2383$ ($9$ bits), $\log 4096 = 8.3178$ ($12$ bits), $\log 65536 = 11.0904$ ($16$ bits). If the true mutual information between two views of an image is plausibly hundreds of bits, then a batch of $4096$ is estimating at most $12$ bits of it, and *the bound is essentially saturated throughout training*. The reported objective value is then telling you about $K$, not about MI.

Second, Worked Example 11.15 quantified the general problem: to certify $I$ nats from a lower bound of this type you need sample size exponential in $I$, and the McAllester–Stratos result quoted in §11.11 makes this a theorem rather than an observation about a particular estimator — any high-confidence lower bound on MI from $n$ samples is $O(\log n)$.

The reasonable inference, supported by the empirical literature (Tschannen and coauthors), is that InfoNCE works for reasons only loosely connected to mutual information: the specific critic architecture, the choice of augmentations defining what counts as a positive pair, and a metric-learning effect on the embedding geometry. Two encoders achieving the same InfoNCE value can produce representations of very different downstream utility, and encoders achieving *higher* estimated MI are sometimes worse. Treat InfoNCE as a well-tested contrastive loss, not as a mutual information maximiser.

### 11.10.4 Why "select the features with highest mutual information" underdelivers

MI-based feature selection scores each candidate feature $X_j$ by $\hat I(X_j;Y)$ and keeps the top $k$. Three results from this chapter predict its failure modes.

**Invariance means MI cannot see difficulty.** By Proposition 11.15, MI is unchanged under any invertible reparameterisation. So $X_j$ and $\exp(X_j^3)$ have identical MI with $Y$, though one may be linearly separable and the other useless to a linear model. MI measures how much information is present *for an unconstrained decoder*; your downstream model is constrained. A feature with high MI that no member of your hypothesis class can exploit is not a good feature.

**DPI means "more MI" is trivially achieved by doing nothing.** By Theorem 11.28, $I(g(X);Y)\le I(X;Y)$ for every $g$. So the MI-maximal "representation" is always the raw input, and any objective of the pure form "maximise $I(Z;Y)$ over representations $Z = g(X)$" is maximised by $g = \mathrm{id}$. Whatever makes a representation good is therefore in the *constraint* — dimension, smoothness, rate, invariance to nuisances — not in the MI term. The IB objective is coherent for exactly this reason: it has a competing $I(X;Z)$ term.

**Marginal scores ignore redundancy and interaction.** $I(X_j;Y)$ high for each of two features says nothing about $I(X_1,X_2;Y)$: the features may be duplicates (jointly no better than one) or, conversely, individually uninformative and jointly decisive. Common Confusion 11.5's example is the extreme case — with $Y = \mathrm{sign}(X_1X_2)$ and independent symmetric $X_i$, each marginal MI is zero while the joint MI is $\log 2$. A marginal screen discards both features.

**And the estimates are biased downwards at exactly the wrong place.** The MI lab in §11.4 and §11.5 showed binning estimators saturating near $\log(\text{bins})$ and neural estimators under-reporting large MI on held-out data ($3.13$ nats against a true $6$). So the features you most want to keep — the highly informative ones — are the ones whose scores are most compressed toward each other.

None of this makes MI useless for feature selection; conditional-MI greedy schemes address redundancy, and for low-dimensional discrete features the estimates are trustworthy. It does mean the method should be justified by cross-validated downstream performance rather than by the information-theoretic story.

### 11.10.5 A payoff table

| Practice | Information-theoretic content | Where in this chapter | Caveat to remember |
|---|---|---|---|
| Cross-entropy training | $\mathbb{E}[\mathrm{CE}] = H + D_{\mathrm{KL}}$; MLE $=$ min forward KL | Prop 11.20, 11.21 | The floor is $H$, not $0$ (WE 11.9) |
| Label smoothing | Optimum is smoothed target; finite logit gap | Prop 11.22 | Raises the loss floor by design |
| Bits/dim and perplexity reporting | Source coding; redundancy $=$ KL | Thm 11.46, Prop 11.48, Def 11.49 | State scale, dequantisation, and tokenizer (CC 11.8) |
| VAE / $\beta$-VAE | $-\mathrm{ELBO} = D + R$; $R \ge I_q(X;Z)$ | Prop 11.55 | ELBO is blind along $D+R=c$ |
| GAN (original) | $\min_G \mathrm{JS}$ via Fenchel duality | Thm 11.34 | Pinned at $\log2$ on singular supports (Prop 11.43) |
| WGAN | Kantorovich dual with 1-Lipschitz critic | Thm 11.41, Prop 11.44 | Constraint is load-bearing; $n^{-1/d}$ rates (CC 11.7) |
| MINE / DV estimators | $D_{\mathrm{KL}} = \sup$ of a DV functional | Thm 11.33 | Biased; variance grows with the value (WE 11.15) |
| InfoNCE | Lower bound on MI, capped at $\log K$ | §11.10.3 | Saturated in practice; not really MI |
| Distillation | Minimises $D_{\mathrm{KL}}(\text{teacher}\Vert\text{student})$ | §11.3.3 | Direction of KL determines mode behaviour |
| RLHF KL penalty | Reverse KL to the reference policy; locally $\tfrac12\delta^\top I\delta$ either way | §11.3.3, §11.3.5 | Mode-seeking: expect narrowing |
| Natural gradient; TRPO/PPO trust region | Steepest descent when step size is measured in KL; the feasible set is the Fisher ellipsoid | Prop 11.23 | The trust region lives in policy space, not weight space |
| Minimax rate claims | Fano / Le Cam lower bounds | Cor 11.53, Prop 11.54 | A rate needs both halves (WE 11.24) |
| Representation learning | DPI: processing cannot add information | Thm 11.28, §11.10.4 | Constraints, not MI, make a representation good |

> **Self-check 11.10.**
> 1. A VAE reports $-\mathrm{ELBO} = 3.2$ nats/dim. What extra number do you need to know whether the latent is doing anything?
> 2. Why is $I(X;Z)$ either infinite or constant for a deterministic injective encoder, and what does that imply about information-plane plots?
> 3. Your InfoNCE objective sits at $8.2$ nats with batch size $4096$. What can you conclude about $I(X;Y)$?
> 4. Give the one-line DPI argument for why "maximise $I(Z;Y)$ over $Z=g(X)$" cannot by itself define a good representation.

## 11.11 Results used but not proved

Everything else in this chapter has a complete proof in the text. The following results were used or quoted without proof, either because they belong to an earlier chapter or because their proofs would require machinery beyond our scope. Each entry says where the result was needed and where to find a proof.

| Result | Used in | Where to find a proof |
|---|---|---|
| Radon–Nikodym theorem; existence and a.e.-uniqueness of $\mathrm{d}\nu/\mathrm{d}\mu$ | Definitions 11.10, 11.16, 11.29 — every divergence in the chapter | Ch. 2 |
| Dominated Convergence Theorem | Interchanging limits and integrals in Theorem 11.14 and Proposition 11.13; differentiation under the integral sign in Proposition 11.23 | Ch. 2 |
| Jensen's inequality (measure-theoretic form) | Theorem 11.17 (Gibbs), Proposition 11.50(2), Theorems 11.31 and 11.32 | Ch. 4 |
| Weak law of large numbers | Proposition 11.47 (asymptotic equipartition) | Ch. 4 |
| Change of variables with Jacobian determinant | Theorem 11.12, Proposition 11.50(1) | Ch. 3 |
| Disintegration / existence of regular conditional distributions on Polish spaces | Definition 11.24 (conditional MI), Proposition 11.25, Theorem 11.28 | Ch. 6; Kallenberg, *Foundations of Modern Probability*, Thm 6.3 |
| Exponential family: identifiability and existence of the moment-matching parameter | Theorem 11.9 (the max-entropy solution) | Ch. 3; Brown, *Fundamentals of Statistical Exponential Families*, Ch. 2 |
| Neyman–Pearson lemma (likelihood-ratio tests are optimal) | Proposition 11.37, Proposition 11.54 | Ch. 9 |
| Bretagnolle–Huber inequality $\mathrm{TV}\le\sqrt{1-e^{-D_{\mathrm{KL}}}}$ | Worked Example 11.16 | Tsybakov, *Introduction to Nonparametric Estimation*, §2.4 (Lemma 2.6); Canonne, *A short note on an inequality between KL and TV* (2022) |
| Fisher information $I(\theta)$: the score has mean zero; the information equality $\mathbb{E}_\theta[\nabla^2\log p_\theta] = -I(\theta)$; $I$ as the local resolution scale $O(n^{-1/2})$ | Proposition 11.23; the MDL/BIC paragraph in §11.8.3 | Ch. 8 |
| Fenchel–Moreau biconjugation: $f = f^{**}$ for convex, lower-semicontinuous $f$ | Theorem 11.34 ($f$-GAN bound) | Rockafellar, *Convex Analysis*, Thm 12.2; Boyd & Vandenberghe §3.3 |
| Optimality of Huffman's algorithm among prefix-free codes | Worked Examples 11.1, 11.20; Exercise B6 | Cover & Thomas §5.8 |
| Kantorovich duality, including the reduction to the 1-Lipschitz dual for $c=d$ | Theorem 11.41, Proposition 11.44 | Villani, *Optimal Transport: Old and New*, Ch. 5; Santambrogio §1.6 |
| Optimality of the monotone coupling for general convex costs on $\mathbb{R}$, and its *uniqueness* when the cost is strictly convex ($p>1$) | Theorem 11.42, general-$p$ statement | Santambrogio §2.2; Villani, *Topics in Optimal Transportation*, Thm 2.18 |
| $W_p$ satisfies the triangle inequality (requires the gluing lemma) | Definition 11.40 calling $W_p$ a metric | Villani, *Optimal Transport: Old and New*, Ch. 6 |
| Convergence of Sinkhorn's algorithm; $O(\varepsilon\log(1/\varepsilon))$ entropic bias | §11.7.4, Worked Example 11.19 | Peyré & Cuturi, *Computational Optimal Transport*, Ch. 4 |
| Varshamov–Gilbert bound: existence of $2^{cd}$ binary codewords with pairwise Hamming distance $\ge d/4$ | The packing construction discussed after Corollary 11.53 | Tsybakov, *Introduction to Nonparametric Estimation*, Lemma 2.9 |
| Consistency of the KSG $k$-nearest-neighbour mutual information estimator | Worked Example 11.15(b) | Kraskov, Stögbauer & Grassberger (2004); Gao, Oh & Viswanath (2018) for rates |
| Any high-confidence distribution-free lower bound on MI computable from $n$ samples is $O(\log n)$ | §11.5.3, §11.10.3 | McAllester & Stratos (2020), *Formal Limitations on the Measurement of Mutual Information* |
| Empirical $W_1$ converges at rate $n^{-1/d}$ in general | Common Confusion 11.7 | Fournier & Guillin (2015); Weed & Bach (2019) |
| BIC as the leading-order asymptotic of two-part code length | The MDL paragraph in §11.8.3 | Grünwald, *The Minimum Description Length Principle*, Ch. 7 |

## 11.12 Chapter summary

1. **Entropy is forced, not chosen.** Continuity, monotonicity in the uniform case, and the grouping property pin $H$ down to $-c\sum p_i\log p_i$, with $c$ the choice of units (Theorem 11.4). Every alternative "uncertainty measure" violates one of the three.
2. **Maximising entropy subject to moment constraints produces exactly an exponential family** with the constraint functions as sufficient statistics (Theorem 11.9), and the shortfall of any other feasible distribution is a KL divergence to the max-entropy solution — the Pythagorean identity, verified numerically at $H(p_\lambda)-H(q) = D_{\mathrm{KL}}(q\Vert p_\lambda) = 1.327905$ bits for the constrained die.
3. **Differential entropy is not entropy.** It can be negative, it shifts by $\mathbb{E}[\log|\det J|]$ under reparameterisation (Theorem 11.12), and quantising at scale $\Delta$ gives $H(X^\Delta) = h(X) - d\log\Delta + o(1)$, which diverges (Proposition 11.13). Any claim about "the information content" of a continuous variable that is not invariant under smooth invertible maps is a claim about your coordinates.
4. **The general definition is relative entropy against a named base measure.** With $\mu = \delta_0 + \lambda$, the mixed variable $A = \max(0,Z)$ from Chapter 1 has a perfectly well-defined $H_\mu(A) = 1.056043$ nats. One formula covers discrete, continuous, and mixed laws — provided you say which $\mu$ (Worked Example 11.8).
5. **KL divergence is not a metric** — asymmetric and violating the triangle inequality by construction (Proposition 11.18, with the explicit Bernoulli counterexample $1.757780 > 1.714900$) — but it is the unique quantity with the operational meaning of *excess code length*: believing $q$ when the truth is $p$ costs $D_{\mathrm{KL}}(p\Vert q)$ extra bits per symbol (Proposition 11.48). Locally, though, KL forgets its asymmetry: $D_{\mathrm{KL}}(P_\theta\Vert P_{\theta+\delta}) = \tfrac12\delta^\top I(\theta)\delta + o(\lVert\delta\rVert^2)$ in *both* directions (Proposition 11.23), the Fisher quadratic form that makes natural gradient and trust-region KL constraints coherent.
6. **Cross-entropy $=$ entropy $+$ KL.** The training loss you minimise has an irreducible floor equal to the data's own entropy (Proposition 11.20). A cross-entropy of $0.61$ nats on a source with $H = 0.61$ nats is a *perfect* model, not a mediocre one, and label smoothing raises that floor deliberately (Proposition 11.22).
7. **The direction of the KL determines what your model does when it cannot fit.** Forward KL is mass-covering, reverse KL is mode-seeking; for the bimodal target of §11.3 the two optima are $N(1.600, 3.800^2)$ and $N(4.000,1.000^2)$ respectively, and the reverse-KL optimum assigns probability $3.2\times10^{-5}$ to a region carrying $30\%$ of the true mass.
8. **Mutual information is a KL divergence between the joint and the product of marginals**, hence zero exactly under independence (Definition 1.23), invariant under invertible reparameterisation, and equal to $+\infty$ whenever $Y$ is a deterministic injective function of a continuous $X$ (Theorem 11.26).
9. **The data processing inequality is the structural fact behind both good and bad arguments.** Processing cannot create information (Theorem 11.28), which is why binning estimators saturate ($0.406\to1.161$ bits with $2\to32$ bins against a truth of $1.198$) and why "maximise $I(Z;Y)$ over representations" is trivially solved by the identity map. The content of a representation-learning objective is in its constraints.
10. **$f$-divergences are exactly the quantities built from pointwise density ratios**, they all obey DPI and joint convexity (Theorems 11.31, 11.32), and *all of them are pinned at the constant $f(0)+f'(\infty)$ between mutually singular measures* (Proposition 11.43) — $\log 2$ for JS, $1$ for TV, $\infty$ for KL. Failure 4 was never about JS.
11. **Variational estimators of divergences are lower bounds with bias that grows with the value being estimated.** With the exactly optimal critic, the Donsker–Varadhan plug-in at batch $64$ reported $25.66\pm2.71$ nats for a true $18$; InfoNCE cannot exceed $\log K$; and no distribution-free high-confidence lower bound can exceed $O(\log n)$. Reported MI values in the tens of nats are reporting the batch size.
12. **Total variation is the testing distance**: the best achievable sum of Type I and Type II errors is $1-\mathrm{TV}$, and Pinsker's inequality $\mathrm{TV}\le\sqrt{D_{\mathrm{KL}}/2}$ is the bridge that lets you compute with KL (which tensorises) while reasoning about testing (which is TV).
13. **Transport distances see geometry that no $f$-divergence can.** For the two parallel segments, every $f$-divergence is constant in the displacement while $W_1 = |\alpha|$ (Proposition 11.44), and $W_1$ has the closed form $\int|F-G|$ on the line (Theorem 11.42). The price is estimation: $n^{-1/d}$ sample complexity, and a Lipschitz constraint that is load-bearing rather than decorative.
14. **Source coding makes entropy operational**: $H_D \le L^\star < H_D+1$ per symbol, with the overhead decaying as $O(1/n)$ over blocks (Theorem 11.46). The reason blocks work is a concentration statement — the asymptotic equipartition property (Proposition 11.47) — which says almost all probability sits on about $e^{nH}$ roughly-equiprobable sequences out of $|\mathcal{X}|^n$; compression is indexing that set. Bits per dimension is this quantity, and comparing likelihoods across papers requires matching four conventions — scale, discreteness, normalisation, and bound type — with the scale factor alone worth $d\log 256 = 17034.79$ nats on CIFAR-10.
15. **Fano's inequality converts an information budget into an error floor.** Directly, it bounds achievable classification accuracy from $I(Y;Z)$ (Corollary 11.52) — one-sidedly, so a gap between the floor and your error is not evidence that improvement exists, as the disease-test example shows. Via the testing reduction it produces minimax lower bounds (Corollary 11.53, Proposition 11.54). A rate is established only when a matching upper bound exists: for the Gaussian mean, $\mathfrak{M}_n \asymp \sigma/\sqrt n$, with the lower half from Le Cam and the upper half from Chapter 4.
16. **The ELBO is a rate–distortion trade-off**, $-\mathrm{ELBO} = D+R$ with $R \ge I_q(X;Z)$ (Proposition 11.55). It is therefore blind along every line $D+R=c$, which is why posterior collapse is a point on the objective's own level set rather than an optimiser failure, and why you should report $R$ and $D$ separately.

### Callbacks

This chapter discharged four promises made earlier. Chapter 1's **Failure 1** — densities exceed $1$, so "probability" and "density" are different objects — reappeared as its own shadow: negative differential entropy, resolved by recognising $h$ as a comparison against a unit-scale reference rather than a count of anything (§11.2.1). Chapter 1's **Failure 2** — the variable $A = \max(0,Z)$ has neither a pmf nor a Lebesgue density — was closed completely in Worked Example 11.8 by the same device that closed Failure 1, namely naming a base measure: $H_\mu(A)$ with $\mu = \delta_0+\lambda$ is a single finite number computed by a single formula. Chapter 2's **Failure 4** — the JS divergence is pinned at $\log 2$ between mutually singular measures, so the GAN objective has no gradient when the generator's support is a lower-dimensional manifold — was closed twice over: Proposition 11.43 showed the pinning is a structural feature of *every* $f$-divergence and not a defect of JS, and Proposition 11.44 exhibited the cure by computing $W_1 = |\alpha|$ for the same pair of measures where every $f$-divergence is constant. Chapter 2's **ELBO** was revealed in Proposition 11.55 to be a rate–distortion Lagrangian, which is what makes $\beta$-VAEs and posterior collapse a single phenomenon. Chapter 1's other running examples were put back to work rather than merely cited: the three coin flips carry the source-coding story from Worked Example 11.1 to the block-coding numbers of Theorem 11.46; the two dice give the chain rule, the data processing inequality, and the creation of dependence by conditioning in a single hand-computable table (Worked Example 11.13); and the screening test reappears twice, first as a mutual information ($45.5\%$ of $H(D)$) and then as an accuracy floor that is valid but not attained (Worked Example 11.23). Along the way, Chapter 3's exponential family arrived as the answer to a maximisation problem rather than as a definition (Theorem 11.9), Chapter 8's Fisher information appeared twice — as the quadratic form that KL becomes near a parameter (Proposition 11.23), and as the resolution scale that turns MDL into BIC, and Chapter 9's Neyman–Pearson lemma turned out to be the same statement as "total variation is the testing distance" (Proposition 11.37).

### Before you continue

1. Without looking, state the three hypotheses of Shannon's characterisation and say which one fails for the "variance" of a distribution as a candidate uncertainty measure.
2. Why is $h(X)$ coordinate-dependent while $I(X;Y)$ is not? Give the one-line reason in terms of Jacobians.
3. A colleague reports a cross-entropy loss of $0.35$ nats and calls it "not converged yet". What single additional quantity would settle whether that is true?
4. You are fitting a unimodal model to a bimodal target and you want the model to cover both modes. Which direction of KL do you minimise, and what is the cost of that choice?
5. Explain, in terms of Proposition 11.43, why replacing JS with reverse KL in a GAN would not fix the vanishing-gradient problem.
6. Your MI estimator reports $9.8$ nats with batch size $4096$. What is the first thing you check?
7. State Pinsker's inequality and explain why a minimax lower-bound proof needs both KL and TV rather than just one of them.
8. A generative model reports $-\log p(x) = 12000$ nats on CIFAR-10. Is that good? What do you need to know first?
9. Your VAE has $-\mathrm{ELBO} = 3.1$ nats/dim and beautiful reconstructions. Can you conclude the latent is informative?

### Looking ahead to Chapter 12

Chapter 12 turns from information to geometry, and asks what happens to probability in high dimensions — where intuitions trained in two and three dimensions fail systematically rather than occasionally. We will prove concentration of measure: that Lipschitz functions of many independent variables are nearly constant, so a random point on a high-dimensional sphere is, with overwhelming probability, almost orthogonal to any fixed direction and almost exactly at the mean distance from any fixed point. We will develop the Chernoff, Hoeffding, Bernstein, and McDiarmid inequalities as tools rather than curiosities, and use them to prove the Johnson–Lindenstrauss lemma: $n$ points in any dimension can be linearly embedded into $O(\varepsilon^{-2}\log n)$ dimensions with all pairwise distances preserved to within a factor $1\pm\varepsilon$ — a statement about *dimension reduction being nearly free* that underwrites random projections, locality-sensitive hashing, and much of modern approximate nearest-neighbour search. We will then look at random matrices: the spectrum of a random Gaussian matrix, the Marchenko–Pastur law, and why the singular values of a randomly initialised layer are what they are. The through-line is that the curse of dimensionality has a twin, a *blessing* of dimensionality, and the blessing is exactly what makes deep learning work: in high dimensions, random directions are nearly orthogonal, distances concentrate, and optimisation landscapes that look hopeless from a low-dimensional picture turn out to be navigable. Chapter 11's $n^{-1/d}$ transport rates and $\log K$ estimation ceilings were the curse; Chapter 12 is the blessing.

# Exercises

Difficulty is marked ★ (routine, applies a definition or a stated result), ★★ (requires assembling two or three ideas), ★★★ (a small research-level derivation or a lab with real pitfalls). Full solutions for every exercise appear after Appendix A; the point of the solutions is the *reasoning*, so read them as worked arguments rather than answer keys.

## Part A — Theory

**A1 ★.** Let $p = (\tfrac12,\tfrac14,\tfrac18,\tfrac18)$ on $\{a,b,c,d\}$. (a) Compute $H(p)$ in bits by hand. (b) Exhibit a binary prefix-free code whose expected length equals $H(p)$ exactly, and explain why this is possible here but not for the distribution of Worked Example 11.1. (c) Verify Kraft's inequality for your code with equality.

**A2 ★.** Prove that for a discrete random variable, $H(X) = 0$ if and only if $X$ is almost surely constant. Where in the argument do you use that $-t\log t \ge 0$ on $[0,1]$ with equality only at $t\in\{0,1\}$?

**A3 ★★.** Prove subadditivity, $H(X,Y)\le H(X)+H(Y)$, with equality if and only if $X$ and $Y$ are independent (Definition 1.23). Do it twice: once from the chain rule plus Proposition 11.7, and once by identifying the gap as a mutual information.

**A4 ★.** (a) For $X\sim\mathrm{Unif}[0,a]$, find the value of $a$ making $h(X) = 0$ and the value making $h(X) = -3$ bits. (b) A student concludes that the second distribution "has negative uncertainty". Write the two-sentence correction you would give.

**A5 ★★.** Let $X$ have a density on $\mathbb{R}^d$, let $A$ be invertible and $b\in\mathbb{R}^d$, and set $Y = AX + b$. Prove $h(Y) = h(X) + \log|\det A|$ directly from the change-of-variables formula (Ch. 3), without quoting Theorem 11.12. Deduce that $h$ is invariant under rotations and translations but not under scaling, and state what this implies about using $h$ to compare two models expressed in different units.

**A6 ★★.** Give an explicit pair $P,Q$ on $\mathbb{R}$ with $D_{\mathrm{KL}}(P\Vert Q)<\infty$ and $D_{\mathrm{KL}}(Q\Vert P) = +\infty$, and compute the finite value. Then explain, in terms of Definition 11.16, exactly which hypothesis fails in the infinite direction, and connect this to why "the KL divergence between my model and the data" can be infinite in one direction throughout training.

**A7 ★★.** (a) Prove $I(X;Y\mid Z)\ge0$. (b) Let $X,Y$ be independent fair bits and $Z := X\oplus Y$ (XOR). Compute $I(X;Y)$ and $I(X;Y\mid Z)$. (c) Conclude that conditioning can *increase* mutual information, and reconcile this with Proposition 11.7 ("conditioning reduces entropy"), which might seem to say the opposite.

**A8 ★★.** Prove the chain rule for mutual information, $I(X;Y_1,Y_2) = I(X;Y_1) + I(X;Y_2\mid Y_1)$, and use it to give a second proof of the data processing inequality (Theorem 11.28) for the chain $X\to Y\to Z$.

**A9 ★★.** Define the **Hellinger distance** by $\mathrm{Hel}(P,Q)^2 := \int(\sqrt p-\sqrt q)^2\,\mathrm{d}\mu$ (this is the $f$-divergence of §11.5 with $f(t) = (\sqrt t-1)^2$; the value does not depend on the dominating $\mu$). Prove the two-sided comparison with total variation,
$$\tfrac12\,\mathrm{Hel}(P,Q)^2\ \le\ \mathrm{TV}(P,Q)\ \le\ \mathrm{Hel}(P,Q)\sqrt{1-\tfrac14\mathrm{Hel}(P,Q)^2}\,,$$
using $\sqrt{pq}\ge\min(p,q)$ for the left inequality and Cauchy–Schwarz on the factorisation $|p-q| = |\sqrt p-\sqrt q|(\sqrt p+\sqrt q)$ for the right. Then show that Hellinger *tensorises*, $1-\tfrac12\mathrm{Hel}(P^{\otimes n},Q^{\otimes n})^2 = \bigl(1-\tfrac12\mathrm{Hel}(P,Q)^2\bigr)^n$, and explain in three sentences why this makes Hellinger, rather than KL, the natural tool for a lower-bound argument when $D_{\mathrm{KL}}(P\Vert Q) = \infty$.

**A10 ★★.** Show that no inequality of the form $D_{\mathrm{KL}}(P\Vert Q)\le C\cdot\mathrm{TV}(P,Q)^2$ can hold with a universal constant $C$, by exhibiting a family of pairs where the left side is infinite and the right side is arbitrarily small. Then state precisely what Pinsker's inequality does and does not let you conclude about a pair of distributions with small TV.

**A11 ★★.** Among all densities on $(0,\infty)$ with mean $m$ fixed, prove that the exponential density with rate $1/m$ uniquely maximises differential entropy, and compute the maximal value. Do it by the KL trick used in Theorem 11.14 rather than by Lagrange multipliers, and say which hypothesis of Theorem 11.9 you are instantiating.

**A12 ★★.** (a) Show that the Fano bound (11.7) is vacuous (i.e. gives $P_{\mathrm{e}}\ge$ a non-positive number) exactly when $I(X;Y)\ge \log M - \log 2$, and interpret that threshold. (b) Show directly from (11.6) that $P_{\mathrm{e}} = 0$ forces $H(X\mid Y) = 0$, and explain why this is the "no free lunch" statement in its purest form.

**A13 ★★★.** Let $\mathcal{L}$ be a set of probability measures defined by linear constraints, $\mathcal{L} := \{P : \mathbb{E}_P[T_j] = \tau_j,\ j=1,\dots,k\}$, let $R$ be a fixed reference measure, and suppose $P^\star := \arg\min_{P\in\mathcal{L}} D_{\mathrm{KL}}(P\Vert R)$ exists with $\mathrm{d}P^\star/\mathrm{d}R \propto \exp(\sum_j\lambda_j T_j)$. Prove the **Pythagorean identity**: for every $Q\in\mathcal{L}$,
$$D_{\mathrm{KL}}(Q\Vert R) = D_{\mathrm{KL}}(Q\Vert P^\star) + D_{\mathrm{KL}}(P^\star\Vert R).$$
Then deduce Theorem 11.9's version (with $R$ counting measure or Lebesgue measure, so that $D_{\mathrm{KL}}(\cdot\Vert R) = -H(\cdot)$) as a special case, and explain in one paragraph why this makes maximum-entropy modelling a *projection*.

**A14 ★★★.** Prove Le Cam's two-point bound for squared-error loss: if $\theta_0,\theta_1$ satisfy $|\theta_0-\theta_1| \ge 2s$ then
$$\inf_{\hat\theta}\max_{j\in\{0,1\}}\mathbb{E}_j\bigl[(\hat\theta-\theta_j)^2\bigr] \ \ge\ \frac{s^2}{2}\bigl(1-\mathrm{TV}(P_0,P_1)\bigr).$$
Be careful about the step where you convert an estimator into a test; the naive argument loses a factor you should identify.

**A15 ★★.** (The local geometry of KL.) Let $\{P_\theta\}$ satisfy the hypotheses of Proposition 11.23.

(a) The proof of Proposition 11.23 gave the forward direction in full and the reverse direction more briskly. Write out the reverse-direction argument completely, stating at each differentiation which hypothesis licenses passing the derivative under the integral, and identifying exactly where positivity of $p_\theta$ is used.
(b) Define the **Jeffreys divergence** $J(P,Q) := D_{\mathrm{KL}}(P\Vert Q)+D_{\mathrm{KL}}(Q\Vert P)$. Show that $J(P_\theta,P_{\theta+\delta}) = \delta^\top I(\theta)\delta + o(\lVert\delta\rVert^2)$, and explain in one sentence why symmetrising KL changes the local geometry only by a constant factor.
(c) For the Bernoulli family, expand $D_{\mathrm{KL}}(\mathrm{Ber}(p)\Vert\mathrm{Ber}(p+\delta))$ and $D_{\mathrm{KL}}(\mathrm{Ber}(p+\delta)\Vert\mathrm{Ber}(p))$ in powers of $\delta$ through the cubic term, obtaining the coefficients exactly. Verify the quadratic terms agree with $\tfrac12\delta^2 I(p)$, compute the difference of the cubic coefficients at $p = 0.3$, and reconcile your answer with the value $-1.5051$ measured at $\delta = 0.001$ in §11.3.5.
(d) TRPO constrains policy updates by an average KL to the previous policy. Using Proposition 11.23, describe the shape of the feasible set in weight space, explain why it is not a Euclidean ball, and give one concrete situation in which a Euclidean-ball trust region on the weights would permit a catastrophic policy change that the KL trust region forbids.

## Part B — Computational labs

Every lab asks you to report numbers. State your seed, sample sizes, and grid resolutions, and where a closed form exists, report the closed form alongside the estimate and the signed error. Reference values from the chapter are given so you can check yourself.

**B1 ★★ (forward versus reverse KL).** Let $p = 0.3\,N(-4,1) + 0.7\,N(4,1)$ and let $q_\theta = N(\mu,\sigma^2)$.

(a) On a grid over $[-20,20]$ with step $10^{-4}$, numerically minimise $D_{\mathrm{KL}}(p\Vert q_\theta)$ over $(\mu,\sigma)$. Report the optimum and verify it equals the moment-matching Gaussian (same mean and variance as $p$) to three decimals.
(b) Minimise $D_{\mathrm{KL}}(q_\theta\Vert p)$ from initialisations $\mu_0 \in\{-6,-4,-1,0,1,4,6\}$ with $\sigma_0=1$. Report which optimum each initialisation reaches, and both KL values at each.
(c) For each fitted $q$, report $q(X<0)$ and compare with $p(X<0)$.
(d) Write two sentences on which fit you would prefer for (i) a proposal distribution in importance sampling, (ii) a generative model whose samples will be shown to users.

Reference: forward optimum $N(1.600, 3.800^2)$ with $D_{\mathrm{KL}}(p\Vert q) = 0.724222$, $D_{\mathrm{KL}}(q\Vert p) = 2.095935$, $q(X<0)=0.336858$; reverse global optimum $N(4.000,1.000^2)$ with $D_{\mathrm{KL}}(q\Vert p)=0.356615$ and $q(X<0)=3.2\times10^{-5}$; reverse local optimum $N(-4.000,1.001^2)$ with $D_{\mathrm{KL}}(q\Vert p) = 1.203826$; $p(X<0) = 0.300013$.

**B2 ★★★ (three mutual information estimators and where each breaks).** Let $(X,Y)$ be standard bivariate Gaussian with correlation $\rho$, so $I = -\tfrac12\log(1-\rho^2)$.

(a) For target values $I\in\{0.2,0.5,1,2,4,6\}$ nats, solve for $\rho$ and draw $n = 20{,}000$ samples (seed $0$).
(b) Estimate $I$ three ways: equal-mass binning with $16$ bins per axis; the KSG estimator with $k=3$; and a MINE critic ($2\to128\to128\to1$, ReLU, Adam at $5\times10^{-4}$, $4000$ steps, batch $256$). For MINE report *both* the value on the final training batches and the value on twenty fresh batches of $4096$.
(c) Plot all three against the truth. Identify the binning ceiling analytically and confirm it numerically.
(d) Fix $I = 4$ nats and, using the *exactly optimal* Donsker–Varadhan critic $T^\star(x,y) = \log\frac{p(x,y)}{p(x)p(y)}$, compute the plug-in DV estimate over $500$ independent batches of size $64$, $512$, and $4096$. Report mean and standard deviation. Explain why the estimator is biased *upwards* even though the critic is optimal, and why the bias shrinks with batch size but grows with $I$.

Reference for (b) at $n=20{,}000$: at $I=1$, binning $0.921$, KSG $1.001$, MINE held-out $1.002$; at $I = 6$, binning $2.717$, KSG $6.011$, MINE held-out $3.127$ but $6.168$ on training batches. Reference for (d) at $I=4$: $4.656\pm0.993$ (batch $64$), $4.222\pm0.578$ ($512$), $4.069\pm0.328$ ($4096$).

**B3 ★★ (what happens as overlap vanishes).** Let $P = \mathrm{Unif}[0,1]$ and $Q_a = \mathrm{Unif}[a,1+a]$.

(a) Derive closed forms for $\mathrm{TV}$, $\mathrm{JS}$, $D_{\mathrm{KL}}(P\Vert Q_a)$, and $W_1$ as functions of $a\ge0$.
(b) Compute all four numerically on $a\in\{0,0.25,0.5,1,1.5,2\}$ and tabulate. Use the quantile formula (Theorem 11.42) for $W_1$ and quadrature for JS.
(c) Repeat with $P = N(0,1)$, $Q_\Delta = N(\Delta,1)$ for $\Delta\in\{0.5,1,2,4,8\}$, where all four have closed forms. Reference: at $\Delta = 4$, $D_{\mathrm{KL}} = 8$, $\mathrm{JS} = 0.632720$ nats, $\mathrm{TV} = 0.954500$, $W_1 = 4$.
(d) Explain, using your tables, why a generator trained on the JS objective receives no useful gradient in the uniform case for $a>1$ but does in the Gaussian case for every $\Delta$. What structural difference is responsible?

**B4 ★ (how tight is Pinsker?).** For Bernoulli pairs $(p,q)$ in $\{(0.5,0.51),(0.5,0.55),(0.5,0.6),(0.5,0.75),(0.5,0.9),(0.5,0.99),(0.1,0.11),(0.01,0.02)\}$, compute $\mathrm{TV}$, the Pinsker bound $\sqrt{D_{\mathrm{KL}}(P\Vert Q)/2}$, and their ratio. Then do the same for $N(0,1)$ versus $N(\Delta,1)$ at $\Delta\in\{0.1,0.5,1,2,4\}$ using $\mathrm{TV} = 2\Phi(\Delta/2)-1$. In one paragraph: in which regime is Pinsker nearly an equality, and in which is it badly loose? Reference: $(0.5,0.51)$ gives ratio $0.9999$; $(0.01,0.02)$ gives ratio $0.2532$; $\Delta=4$ gives $\mathrm{TV}=0.954500$ against a bound of $2$.

**B5 ★ (bits per dimension audit).** Take $d = 3072$.

(a) A model reports mean $\log p(x) = 6000$ nats with data scaled to $[0,1]^d$. Convert to bits per dimension. Repeat for $8000$ nats.
(b) Compute the bpd of the "uniform noise" baseline, and the $\log p$ (unit scale, nats) corresponding to $2.9$ bpd.
(c) A second paper reports $\log p(\tilde x) = -10000$ nats on the $[0,255]^d$ scale. Convert it to the unit scale and to bpd. Which of the models in (a) is it better than?
(d) Write the four-item checklist you would apply before believing any cross-paper likelihood comparison.

Reference: (a) $5.1822$ and $4.2430$ bpd; (b) $8$ bpd and $10859.7$ nats; (c) the scale gap is $d\log 256 = 17034.79$ nats, giving $4.696$ bpd.

**B6 ★★ (the $O(1/n)$ of the source coding theorem).** For $X$ the number of heads in three fair flips ($P_X = (\tfrac18,\tfrac38,\tfrac38,\tfrac18)$), implement Huffman's algorithm.

(a) Report codeword lengths and $\mathbb{E}[L]$ for single symbols; compare with $H(X) = 1.811278$ bits.
(b) Build the product distribution for blocks of $n = 2$ and $n = 3$ symbols, run Huffman on the $4^n$-symbol alphabet, and report bits per symbol. Reference: $1.875$, $1.828125$, $1.820313$.
(c) Plot the redundancy against $n$ and confirm it is $O(1/n)$ rather than $O(1/\sqrt n)$ or constant.
(d) Now use the code built for $q = (\tfrac14,\tfrac14,\tfrac14,\tfrac14)$ on data from $p = (0.4,0.3,0.2,0.1)$ and confirm numerically that the excess expected length equals $D_{\mathrm{KL}}(p\Vert q)$ in bits. Reference: excess $= 0.153561$ bits.

**B7 ★★ (entropic bias of Sinkhorn).** Let $P$ be uniform on $\{0,1,2,3,4\}$ and $Q$ uniform on $\{1,2,3,4,5\}$, with $c(x,y) = |x-y|$, so $W_1 = 1$ exactly.

(a) Implement Sinkhorn in the log domain and compute $\langle\pi^\star,c\rangle$ for $\varepsilon\in\{1,0.3,0.1,0.03\}$, iterating until the marginal violations are below $10^{-12}$. Reference: $1.155913$, $1.001527$, $1.000000$, $1.000000$.
(b) Report the entropy of $\pi^\star$ at each $\varepsilon$ and explain the trend in one sentence.
(c) Try $\varepsilon = 0.001$ with a naive (non-log-domain) implementation and describe what goes wrong and why.

## Part C — Paper-style derivations

These are the exercises to do slowly and write up properly. Each asks for a proof plus a paragraph of interpretation; the interpretation is half the marks.

**C1 ★★★ (the data processing inequality for a general $f$-divergence, and what it means for representation learning).**

(a) Theorem 11.32 proves this in full generality using the perspective function. Supply the *other* proof, the one that is shorter but assumes $P\ll Q$: with $r := \mathrm{d}P/\mathrm{d}Q$, show that the output likelihood ratio is a conditional expectation,
$$\frac{\mathrm{d}(PK)}{\mathrm{d}(QK)}(y) = \mathbb{E}\bigl[r(X)\,\big|\,Y=y\bigr],\qquad X\sim Q,\ Y\mid X\sim K(\cdot\mid X),$$
verify it against the definition of a Radon–Nikodym derivative, and conclude in one line by conditional Jensen (Ch. 6). Then say exactly where this argument breaks when $P\not\ll Q$, and how the convention $g(u,0) = uf'(\infty)$ in Theorem 11.32's proof repairs it.

(b) Specialise to $f(t) = t\log t$ to recover the DPI for KL, and then to the Markov chain $X\to Y\to Z$ to recover Theorem 11.28.
(c) Now the interpretation, in two or three paragraphs. Suppose someone proposes to learn a representation $Z = g_\phi(X)$ by maximising $I(Z;Y)$ for a label $Y$. Use (b) to show the objective is maximised by $g = \mathrm{id}$, so the objective as stated is empty. Then explain what actually makes a representation good in these terms, and use your answer to say precisely what the information bottleneck objective adds and why it is not vacuous. Finally, comment on the claim "layer $\ell+1$ contains more label information than layer $\ell$": give the charitable reading under which it can be true and the DPI-based sense in which it cannot.

**C2 ★★★ (Donsker–Varadhan, $f$-GAN, MINE, and why the estimator is biased).**

(a) Prove the Donsker–Varadhan representation: for $P\ll Q$,
$$D_{\mathrm{KL}}(P\Vert Q) = \sup_{T}\ \Bigl\{\mathbb{E}_P[T] - \log\mathbb{E}_Q\bigl[e^{T}\bigr]\Bigr\},$$
the supremum over measurable $T$ with $\mathbb{E}_Q[e^T]<\infty$, and identify the maximiser. Show explicitly that the objective is invariant under $T\mapsto T+c$ and explain what that implies about what a trained critic can and cannot be interpreted as.
(b) Derive the $f$-GAN variational bound (Theorem 11.34) from Fenchel duality, and show that the original GAN objective is the case $f(t) = t\log t - (t+1)\log\frac{t+1}{2}$ up to an additive constant, with optimal discriminator $D^\star = p/(p+q)$. State the resulting identity between the GAN value and $\mathrm{JS}$.
(c) MINE replaces the two expectations in (a) by empirical averages over a batch of size $b$ and $T$ by a neural network. Prove that the plug-in estimator with a *fixed* critic is biased, determine the sign of the bias, and show the leading-order size of the bias is $O(\mathrm{Var}_Q(e^T)/b)$ in the appropriate sense. Explain why $\mathrm{Var}_Q(e^{T^\star})$ grows with $D_{\mathrm{KL}}(P\Vert Q)$ — this is the crux — and hence why the estimator degrades exactly in the high-information regime that representation-learning papers care about.
(d) Separately, explain the *second* failure mode: when $T$ is trained on the same batch used for evaluation, the estimate can exceed the true KL. Relate both failure modes to the $O(\log n)$ limitation quoted in §11.11, and state the experimental protocol you would insist on before believing a reported MI value.

**C3 ★★★ (Pinsker, Fano, and pinning a rate from both sides).**

(a) Prove Pinsker's inequality $\mathrm{TV}(P,Q)\le\sqrt{\tfrac12 D_{\mathrm{KL}}(P\Vert Q)}$. Use the two-point reduction: first prove it for Bernoulli pairs, then reduce the general case to the Bernoulli case with the data processing inequality applied to the indicator of the set $A^\star = \{p>q\}$. Say why the reduction is legitimate and where you need $\mathrm{TV}$ to be attained at $A^\star$.
(b) Consider estimating $\theta$ from $X_1,\dots,X_n\stackrel{\text{iid}}{\sim}N(\theta,\sigma^2)$ with $\sigma$ known, under absolute-error loss. Using a two-point argument, prove that $\inf_{\hat\theta}\sup_\theta\mathbb{E}_\theta|\hat\theta-\theta| \ge C\sigma/\sqrt n$ for an explicit absolute constant $C>0$. Optimise your choice of separation $\delta$ and report the best constant your argument gives.
(c) Give the matching upper bound for the sample mean, with the exact constant.
(d) Write a paragraph explaining what it means to say $n^{-1/2}$ is "the rate" for this problem, why an upper bound alone is insufficient to support such a claim, and why a lower bound alone is equally insufficient. Then state what would have to change in the argument to obtain the $d$-dimensional result $\mathfrak{M}_n\asymp d\sigma^2/n$ for squared-error loss, and identify which ingredient from §11.11 the many-point version needs.

# Appendix A — Self-check answers

**Self-check 11.1.**
1. $H(X) = \log_2 8 = 3$ bits. $Y = X\bmod 2$ is a fair bit, so $H(Y) = 1$ bit. Given $Y$, $X$ is uniform on $4$ values, so $H(X\mid Y) = 2$ bits — consistent with the chain rule, $H(X,Y) = H(X) = 3 = H(Y)+H(X\mid Y)$. And $H(Y\mid X) = 0$ because $Y$ is a function of $X$.
2. True. $H(Y\mid X) = 0$ forces $Y$ to be almost surely determined by $X$, i.e. $Y$ is $\sigma(X)$-measurable, and Lemma 1.22 (Doob–Dynkin) then supplies a measurable $g$ with $Y = g(X)$ a.s.
3. The standard Gaussian, by Theorem 11.14. A Laplace distribution with the same variance $v$ has $b = \sqrt{v/2}$ and $h = 1+\log(2b)$, against the Gaussian's $\tfrac12\log(2\pi ev)$. The gap is $0.072365$ nats $= 0.104401$ bits, and — worth noticing — it is *independent of $v$*, since both entropies shift by $\tfrac12\log v$ under scaling (Theorem 11.12). You lose about a tenth of a bit.
4. Because Proposition 11.2's upper bound was proved by applying Jensen's inequality to the concave $\log$ directly, not by invoking non-negativity of KL. Theorem 11.17 also rests on Jensen, so both results descend from Chapter 4 independently; neither is used in the other's proof.

**Self-check 11.2.**
1. $h = \log_2(10^{-3}) = -9.965784$ bits. And no — the question is malformed, which is the point. A fair coin flip has $H = 1$ bit, a Shannon entropy of a discrete variable; $h$ is not a Shannon entropy and the two are not on a common scale (Proposition 11.13: the discretised entropy of $X$ diverges as the grid refines). The correct statement is that $X$ is concentrated on a very short interval, so its density is large, so $-\log p$ is very negative.
2. $h(Y) = h(X) + \log|\det A|$. If $A$ is a rotation, $|\det A| = 1$ and $h$ is unchanged.
3. Nothing is impossible: $h<0$ whenever the density exceeds $1$ on a set of substantial mass, which is routine (Chapter 1's Failure 1). Negative $h$ says the distribution is more concentrated than the unit-scale reference, and the number would change if the latent were rescaled — so it is a statement about coordinates, not about the model's quality.
4. $x\mapsto x^3$ is a smooth bijection of $\mathbb{R}$, and mutual information is invariant under such maps (Proposition 11.15), because it is a KL divergence between two measures that both get pushed forward by the same map with the same Jacobian factor, which cancels. Correlation is defined through first and second moments, which are not preserved by nonlinear maps; $\mathrm{Corr}$ is a property of a particular parameterisation, $I$ is a property of the dependence structure.

**Self-check 11.3.**
1. $D_{\mathrm{KL}}(\mathrm{Ber}(0)\Vert\mathrm{Ber}(0.5)) = 1\cdot\log\frac{1}{0.5} = \log 2 = 0.693147$ nats, finite. The reverse is $+\infty$: $\mathrm{Ber}(0.5)$ puts mass $\tfrac12$ on the outcome $1$, to which $\mathrm{Ber}(0)$ assigns zero, so absolute continuity fails and Definition 11.16 gives $+\infty$. Asymmetry is not a blemish; it encodes "which distribution is allowed to be surprised".
2. If the conditional entropy of the labels given the inputs is itself $0.35$ nats. Then $\mathbb{E}[\mathrm{CE}] = H + D_{\mathrm{KL}}$ with $D_{\mathrm{KL}} = 0$: the model is exactly right and the remaining loss is irreducible label noise (Proposition 11.20 and Worked Example 11.9).
3. Forward, $D_{\mathrm{KL}}(p\Vert q)$: it penalises $q$ for being small where $p$ is large, so $q$ must cover all of $p$'s mass, which for a unimodal family means spreading out.
4. With $K = 10$, $\varepsilon = 0.1$: targets $0.91$ and $0.01$, optimal logit gap $\log 91 = 4.510860$, loss floor $0.500288$ nats. With $K = 1000$: targets $0.9001$ and $10^{-4}$, gap $\log(9001) = 9.105091$, floor $1.014848$ nats. The gap *grows* with $K$ at fixed $\varepsilon$ because the smoothing mass $\varepsilon$ is divided among more classes, making each non-target smaller.

**Self-check 11.4.**
1. $I(X;Y) = H(X)-H(X\mid Y)\le H(X)$ since conditional entropy is non-negative for discrete variables, and by symmetry $I\le H(Y)$. For continuous variables $h(X\mid Y)$ can be negative, so the argument collapses — and indeed $I(X;X) = +\infty$ while $h(X)$ is finite.
2. $I(X;Z) = +\infty$ (Theorem 11.26): $Z$ determines $X$, so the joint law is concentrated on a graph, which is singular with respect to the product of the marginals. Maximising $I(X;Z)$ is therefore an objective whose optimum is "change nothing", which expresses no preference among representations at all.
3. Charitable reading: layer $2$'s features are more *usable* — more linearly decodable, or decodable by a smaller hypothesis class — which is a claim about a restricted decoder, not about information. DPI objection: since $Y\to Z_1\to Z_2$ is a Markov chain given the network's determinism, $I(Z_2;Y)\le I(Z_1;Y)$, so no layer can contain more label information than the one before it.
4. $I = -\tfrac12\log(1-0.49) = 0.336672$ nats $= 0.485715$ bits.

**Self-check 11.5.**
1. $f(t) = \tfrac12|t-1|$ is a scaled absolute value of an affine function, hence convex, and $f(1)=0$. $f'(\infty) = \lim_{t\to\infty}f(t)/t = \tfrac12$.
2. Substitute $p = q\cdot(p/q)$ in Definition 11.29: $\int q f(p/q)\,\mathrm{d}\mu = \int p\,\frac{q}{p}f\!\left(\frac{p}{q}\right)\mathrm{d}\mu = \int p\, f^\dagger(q/p)\,\mathrm{d}\mu$ with $f^\dagger(t) = tf(1/t)$, which is again convex with $f^\dagger(1)=0$. The self-conjugate generators are exactly the symmetric divergences: total variation, squared Hellinger, and Jensen–Shannon. KL's conjugate is reverse KL, and $\chi^2$'s conjugate is the reverse $\chi^2$.
3. Nothing changes: $\mathbb{E}_P[T+c]-\log\mathbb{E}_Q[e^{T+c}] = \mathbb{E}_P[T]+c-\log\mathbb{E}_Q[e^T]-c$. So the critic is identified only up to an additive constant, and it recovers $\log\frac{\mathrm{d}P}{\mathrm{d}Q}$ *plus an unknown offset*. You cannot read density-ratio values off a trained critic without separately fixing the constant, which is why critic outputs are not calibrated likelihood ratios.
4. $\log 128 = 4.852030$ nats $= 7$ bits exactly.

**Self-check 11.6.**
1. Overlap $\int\min(p,q) = 0.65$, so $\mathrm{TV} = 1-0.65 = 0.35$ (Proposition 11.36). The best balanced classifier accuracy is $\tfrac12(1+\mathrm{TV}) = 0.675$.
2. Yes, and for a slightly sneaky reason: $\mathrm{TV}$ is symmetric, so applying Theorem 11.38 to the pair in the other order gives $\mathrm{TV}(P,Q)=\mathrm{TV}(Q,P)\le\sqrt{D_{\mathrm{KL}}(Q\Vert P)/2}$. Both bounds hold and you should use whichever KL is smaller.
3. Any pair of mutually singular measures that are geometrically adjacent: the two parallel segments of Proposition 11.44 with $\alpha = 10^{-6}$; or $\mathrm{Unif}[0,1]$ against the uniform distribution on the grid $\{0,\tfrac1{10^6},\dots,1\}$. Scatter plots of samples would be indistinguishable to the eye; $\mathrm{TV} = 1$.
4. Pinsker gives $\mathrm{TV}\le\sqrt{2.5} = 1.58 > 1$, which is weaker than the trivial bound. In general Pinsker is informative only for $D_{\mathrm{KL}} < 2$. The lesson: KL bounds can *certify* indistinguishability only when they are small; and the converse direction fails entirely, since small TV is compatible with infinite KL (Exercise A10).

**Self-check 11.7.**
1. $W_1(\delta_0,\delta_a) = |a|$ (there is only one coupling). $\mathrm{TV}(\delta_0,\delta_a) = 1$ for every $a\ne0$. Only $W_1$ is informative about $a$; TV cannot distinguish $a = 10^{-9}$ from $a = 10^9$.
2. Sort both samples and average the absolute differences of matched order statistics: $W_1 = \frac1n\sum_i|x_{(i)}-y_{(i)}|$. This is Theorem 11.42 applied to two empirical measures whose quantile functions are step functions with equal-length steps. Cost is the two sorts, $O(n\log n)$.
3. Because $\varphi\mapsto t\varphi$ scales the objective linearly: if $\mathbb{E}_P[\varphi]\ne\mathbb{E}_Q[\varphi]$ for some $\varphi$, then $t\varphi$ drives the objective to $+\infty$ as $t\to\infty$. The unconstrained supremum is $+\infty$ whenever $P\ne Q$, so the Lipschitz constraint is what makes (11.4) a finite quantity equal to $W_1$.
4. Any task where the ground metric does not reflect semantics: classification over unordered categories (there is no meaningful distance between "cat" and "truck"); discrete token sequences where a one-token edit can change meaning entirely; or molecular geometry, where a small coordinate perturbation can produce a chemically invalid structure. In such cases the density-ratio view (TV, KL) is the appropriate one.

**Self-check 11.8.**
1. Theorem 11.46 gives $2.3\le L^\star<3.3$ bits for a single symbol; the exact value depends on the pmf, not just on $H$. For blocks of $10$, the per-symbol cost lies in $[2.3,\ 2.4)$.
2. $0.4$ bits, exactly (Proposition 11.48). The excess expected code length *is* the KL divergence, in whatever base the lengths are measured.
3. $-\log p(x) = 2.9\times3072\times\log2$ subtracted from the uniform baseline: $(8-2.9)\times3072\times\log2 = 10859.7$ nats, so $\log p(x) = +10859.7$ nats on the $[0,1]^d$ scale. Positive log-densities are normal here, since the density is concentrated far above $1$.
4. Because a density is not bounded above: put a narrow spike of height $M$ at each of the finitely many observed grid points and $\log p$ grows without limit while the model says nothing useful. Dequantisation replaces the point evaluation $p(\tilde x)$ with the cell integral $\tilde P(\tilde x) = \int_{[0,1)^d}p(\tilde x+u)\,\mathrm{d}u$, which is a genuine probability and hence at most $1$; Jensen (Proposition 11.50(2)) makes the average dequantised log-density a *lower* bound on $\log\tilde P$, so reported bpd is an honest upper bound on achievable code length.

**Self-check 11.9.**
1. The bound tends to $1-\frac{\log M+\log2}{\log M} = -\frac{\log 2}{\log M}$, which is negative — for $M = 10$, $-0.301$ — hence vacuous. Fano is a one-sided instrument: it forbids success when information is scarce and says nothing when information is plentiful.
2. The chain assumption is what licenses $I(X;\hat X)\le I(X;Y)$ via Theorem 11.28. Without it, take $\hat X := X$: then $P_{\mathrm{e}} = 0$ while $H(X\mid Y)$ can be as large as $\log M$, and (11.6) would read $\log M \le 0$. The inequality is about estimators that see only the data.
3. Large separation makes the multiplier $s$ large; small pairwise KL makes $\bar I$ small, which pushes the Fano error floor towards $1$. If the $P_j$ are far apart, $\bar I$ is large, $\log M - \bar I$ becomes negative, and the bound degenerates to the trivial statement $\mathfrak{M}_n\ge$ a negative number — the hypotheses are then statistically distinguishable and no lower bound is available from this construction.
4. Because KL is additive over independent coordinates, so $n$ samples multiply it by exactly $n$ and the dependence on sample size is transparent, whereas the quantity that actually controls testing error is TV; Pinsker is the conversion between the computable one and the meaningful one.

**Self-check 11.10.**
1. The rate $R = \mathbb{E}_{P_X}[D_{\mathrm{KL}}(q(\cdot\mid x)\Vert p)]$, reported separately from the distortion. By Proposition 11.55(2), $R \ge I_q(X;Z)$, so $R\approx0$ certifies that the latent is uninformative no matter what $-\mathrm{ELBO}$ says.
2. For a deterministic injective encoder with continuous $X$, $I(X;Z) = +\infty$; for discrete $X$, $I(X;Z) = H(X)$, a constant that does not depend on the weights at all. Either way the quantity cannot decrease during training, so a decreasing curve in an information plane is measuring the estimator's binning or noise procedure — in effect the scale of the activations — rather than $I(X;Z)$.
3. Only that $I(X;Y)\ge 8.2$ nats. The InfoNCE bound cannot exceed $\log 4096 = 8.317766$ nats, and $8.2$ is $98.6\%$ of that ceiling, so the bound is saturated and carries essentially no information about the true value, which could be $9$ nats or $900$.
4. By the data processing inequality, $I(g(X);Y)\le I(X;Y)$ for every measurable $g$, with equality at $g = \mathrm{id}$; so the identity map is always a maximiser and the objective cannot prefer any nontrivial representation to doing nothing.

# Solutions to Exercises

## Part A

**A1.** *(a)* All four probabilities are powers of two, so the surprisals are integers and the sum is easy:
$$H(p) = \tfrac12\cdot 1 + \tfrac14\cdot 2 + \tfrac18\cdot3+\tfrac18\cdot3 = 0.5+0.5+0.375+0.375 = 1.75\ \text{bits}.$$

*(b)* Take $C(a) = 0$, $C(b) = 10$, $C(c) = 110$, $C(d) = 111$, with lengths $(1,2,3,3)$. Then $\mathbb{E}[L] = 1.75$ bits exactly. The reason this is achievable is that $\ell(x) = -\log_2 p(x)$ came out an integer for every $x$; a distribution with this property is called **dyadic**. Worked Example 11.1's distribution $(\tfrac18,\tfrac38,\tfrac38,\tfrac18)$ is not dyadic — $-\log_2\tfrac38 = 1.415$ — so the ideal lengths are unattainable and Theorem 11.46's lower bound is strict, which is exactly why $\mathbb{E}[L] = 1.875 > 1.811278 = H$ there.

*(c)* $2^{-1}+2^{-2}+2^{-3}+2^{-3} = \tfrac12+\tfrac14+\tfrac18+\tfrac18 = 1$. Equality in Kraft means the code tree has no unused leaves — the code is *complete*, wasting nothing, which is the structural counterpart of attaining the entropy bound.

**A2.** Write $H(X) = \sum_x \eta(p(x))$ with $\eta(t) := -t\log t$ extended by $\eta(0)=0$. On $[0,1]$ we have $\eta\ge0$, with $\eta(t) = 0$ exactly at $t\in\{0,1\}$. A sum of non-negative terms vanishes if and only if every term vanishes, so $H(X)=0$ forces each $p(x)\in\{0,1\}$. Since the $p(x)$ sum to $1$, exactly one of them equals $1$, i.e. $X$ is almost surely constant. Conversely a point mass has $H=0$ by direct substitution.

The place where the characterisation of the zero set is essential is the forward direction: without knowing that $\eta(t)>0$ for $t\in(0,1)$, a vanishing sum could in principle come from cancellation, and the conclusion would not follow. Non-negativity *and* the identification of the zero set are both doing work.

**A3.** *First proof.* The chain rule (Theorem 11.6) gives $H(X,Y) = H(X)+H(Y\mid X)$, and Proposition 11.7 gives $H(Y\mid X)\le H(Y)$. Combining yields subadditivity. Equality requires $H(Y\mid X) = H(Y)$.

*Second proof.* The gap is exactly a mutual information:
$$H(X)+H(Y)-H(X,Y) = I(X;Y) = D_{\mathrm{KL}}\bigl(P_{XY}\,\Vert\,P_X\otimes P_Y\bigr)\ \ge\ 0$$
by Proposition 11.25 and Theorem 11.17. Equality in Gibbs' inequality holds if and only if the two measures coincide, i.e. $P_{XY} = P_X\otimes P_Y$ — which is *literally* Definition 1.23, the definition of independence. This second proof is the better one to remember, because it converts a statement about entropies into a statement about measures and gives the equality case for free rather than as a separate argument.

**A4.** *(a)* $h(\mathrm{Unif}[0,a]) = \log a$, so $h = 0$ at $a = 1$ and $h = -3$ bits at $\log_2 a = -3$, i.e. $a = 1/8$.

*(b)* "Differential entropy is not a count of anything, so it has no reason to be non-negative: it is $\mathbb{E}[-\log p(X)]$, and $p$ exceeds $1$ whenever the distribution is concentrated on a set of volume less than $1$. The number $-3$ bits says your distribution is $2^3$ times more concentrated than the unit-scale reference, and it would become $0$ if you rescaled the axis — the distribution has not changed, only the units."

**A5.** The map $T(x) = Ax+b$ is a diffeomorphism with constant Jacobian $A$, so by the change-of-variables formula the density of $Y$ is $p_Y(y) = p_X\bigl(A^{-1}(y-b)\bigr)\,|\det A|^{-1}$. Substitute $y = Ax+b$, $\mathrm{d}y = |\det A|\,\mathrm{d}x$:
$$h(Y) = -\int p_Y(y)\log p_Y(y)\,\mathrm{d}y = -\int \frac{p_X(x)}{|\det A|}\Bigl[\log p_X(x) - \log|\det A|\Bigr]\,|\det A|\,\mathrm{d}x = h(X)+\log|\det A|.$$
The $|\det A|$ factors cancel in the measure but *not* inside the logarithm, and that asymmetry is the whole phenomenon.

Rotations and reflections have $|\det A| = 1$, and translations do not appear in the formula at all, so $h$ is invariant under rigid motions. Scaling by $c>0$ in all coordinates gives $|\det A| = c^d$ and shifts $h$ by $d\log c$. Consequence: comparing $h$ (or any quantity built from it, such as a reported "entropy of the latent") between two models is meaningless unless both use the same parameterisation and the same units. A model expressed in millimetres has differential entropy $d\log 1000$ nats higher than the identical model in metres.

**A6.** Take $P = \mathrm{Unif}[0,1]$ and $Q = \mathrm{Unif}[0,2]$, with densities $1$ and $\tfrac12$ on their supports. Then $P\ll Q$ and
$$D_{\mathrm{KL}}(P\Vert Q) = \int_0^1 1\cdot\log\frac{1}{1/2}\,\mathrm{d}x = \log 2 = 0.693147\ \text{nats}.$$
In the other direction, $Q\bigl((1,2]\bigr) = \tfrac12 > 0$ while $P\bigl((1,2]\bigr) = 0$, so $Q\not\ll P$ and Definition 11.16 assigns $D_{\mathrm{KL}}(Q\Vert P) = +\infty$.

The hypothesis that fails is absolute continuity of the *first* argument with respect to the second. Read the definition as "$P$ is entitled to be surprised by $Q$, but not the reverse": $D_{\mathrm{KL}}(P\Vert Q)$ is finite iff $Q$ assigns positive probability everywhere $P$ does. In training, this is the everyday situation where a model $Q_\theta$ with bounded or lower-dimensional support is fitted to data $P$ with wider support: $D_{\mathrm{KL}}(P\Vert Q_\theta) = \infty$ for the whole trajectory, so the forward KL cannot even rank parameter values, while the reverse KL stays finite and happily concentrates the model on a subset of the data. This is the same mechanism as Failure 4 and as the maximum-likelihood caveat in Proposition 11.21.

**A7.** *(a)* By Definition 11.24, $I(X;Y\mid Z) = \mathbb{E}_{Z}\bigl[D_{\mathrm{KL}}\bigl(P_{XY\mid Z}\Vert P_{X\mid Z}\otimes P_{Y\mid Z}\bigr)\bigr]$, an expectation of non-negative quantities (Theorem 11.17), hence non-negative.

*(b)* $X,Y$ independent fair bits: $I(X;Y) = 0$. Now condition on $Z = X\oplus Y$. Given $Z = z$, we have $Y = X\oplus z$, so $Y$ is a deterministic function of $X$: $H(Y\mid X,Z) = 0$. And $Z$ alone is a fair bit independent of $Y$, so $H(Y\mid Z) = 1$ bit. Hence
$$I(X;Y\mid Z) = H(Y\mid Z)-H(Y\mid X,Z) = 1 - 0 = 1\ \text{bit} = \log 2\ \text{nats}.$$

*(c)* So conditioning took the mutual information from $0$ to $1$ bit. There is no contradiction with Proposition 11.7, and the reason is worth stating carefully. Proposition 11.7 says $H(Y\mid Z)\le H(Y)$: conditioning reduces *entropy*. Mutual information is a *difference* of entropies, $I(X;Y\mid Z) = H(Y\mid Z)-H(Y\mid X,Z)$, and conditioning on $Z$ reduces both terms. Whether the difference goes up or down depends on which term drops more. Here $H(Y)=H(Y\mid Z)=1$ bit (no reduction, because $Z$ alone is uninformative about $Y$) while $H(Y\mid X)=1$ collapses to $H(Y\mid X,Z)=0$. There is no general inequality between $I(X;Y\mid Z)$ and $I(X;Y)$ in either direction — a fact that quietly invalidates a good deal of informal reasoning about "controlling for a variable".

**A8.** Both sides telescope. Using $I(A;B) = H(A)-H(A\mid B)$ twice,
$$I(X;Y_1)+I(X;Y_2\mid Y_1) = \bigl[H(X)-H(X\mid Y_1)\bigr] + \bigl[H(X\mid Y_1)-H(X\mid Y_1,Y_2)\bigr] = H(X)-H(X\mid Y_1,Y_2) = I(X;Y_1,Y_2).$$
(For continuous variables run the same computation with the KL forms, or note that the identity is a statement about the four measures involved and holds by the chain rule for Radon–Nikodym derivatives.)

*DPI from the chain rule.* Expand $I(X;Y,Z)$ in two orders:
$$I(X;Y,Z) = I(X;Y)+I(X;Z\mid Y) = I(X;Z)+I(X;Y\mid Z).$$
The Markov property $X\to Y\to Z$ says $X$ and $Z$ are conditionally independent given $Y$ (Definition 1.24), hence $I(X;Z\mid Y)=0$. Therefore
$$I(X;Y) = I(X;Z)+I(X;Y\mid Z)\ \ge\ I(X;Z),$$
using part (a) of A7 for the non-negativity of the last term. This proof also hands you the *deficit*: the information lost by passing from $Y$ to $Z$ is exactly $I(X;Y\mid Z)$, the part of $Y$'s information about $X$ that is not already carried by $Z$.

**A9.** Write $\mathrm{Hel}^2 := \int(\sqrt p-\sqrt q)^2\mathrm{d}\mu = 2-2\int\sqrt{pq}\,\mathrm{d}\mu$, using $\int p = \int q = 1$. It is convenient to name the **affinity** $\rho := \int\sqrt{pq}\,\mathrm{d}\mu\in[0,1]$, so $\mathrm{Hel}^2 = 2(1-\rho)$.

*Left inequality.* For non-negative $a,b$ we have $\sqrt{ab}\ge\min(a,b)$ (if $a\le b$ then $\sqrt{ab}\ge\sqrt{a\cdot a}=a$). Hence $\rho\ge\int\min(p,q)\,\mathrm{d}\mu = 1-\mathrm{TV}$ by Proposition 11.36, and therefore
$$\tfrac12\mathrm{Hel}^2 = 1-\rho\ \le\ 1-\bigl(1-\mathrm{TV}\bigr) = \mathrm{TV}.$$

*Right inequality.* Factor the integrand and apply Cauchy–Schwarz:
$$2\,\mathrm{TV} = \int|p-q|\,\mathrm{d}\mu = \int\bigl|\sqrt p-\sqrt q\bigr|\bigl(\sqrt p+\sqrt q\bigr)\mathrm{d}\mu \le \left(\int(\sqrt p-\sqrt q)^2\right)^{1/2}\left(\int(\sqrt p+\sqrt q)^2\right)^{1/2}.$$
The first factor is $\mathrm{Hel}$. The second is $\bigl(2+2\rho\bigr)^{1/2} = \bigl(4-\mathrm{Hel}^2\bigr)^{1/2}$, since $2+2\rho = 4-2(1-\rho) = 4-\mathrm{Hel}^2$. So
$$\mathrm{TV}\le\tfrac12\mathrm{Hel}\sqrt{4-\mathrm{Hel}^2} = \mathrm{Hel}\sqrt{1-\tfrac14\mathrm{Hel}^2}.$$
Sanity check both ends: if $P=Q$ then $\mathrm{Hel}=\mathrm{TV}=0$; if $P\perp Q$ then $\rho = 0$, $\mathrm{Hel}^2 = 2$, and the bounds read $1\le\mathrm{TV}\le\sqrt2\sqrt{1/2} = 1$, pinning $\mathrm{TV}=1$ — consistent with Proposition 11.44.

*Tensorisation.* The affinity factorises over independent coordinates because the square root of a product is the product of square roots:
$$\rho_n := \int\sqrt{p^{\otimes n}q^{\otimes n}} = \prod_{i=1}^n\int\sqrt{p q} = \rho^n,$$
so $1-\tfrac12\mathrm{Hel}(P^{\otimes n},Q^{\otimes n})^2 = \rho_n = \rho^n = \bigl(1-\tfrac12\mathrm{Hel}(P,Q)^2\bigr)^n$.

*Why this matters for lower bounds.* Le Cam's method (Proposition 11.54) needs an upper bound on $\mathrm{TV}$ between two $n$-fold product measures, and the useful route is a divergence that behaves predictably under products. KL tensorises additively and is the usual choice — but it is $+\infty$ the moment the two hypotheses have different supports, and then Pinsker gives nothing at all. Hellinger is bounded by $\sqrt2$ always, tensorises multiplicatively as just shown, and controls $\mathrm{TV}$ from *both* sides, so it survives exactly the singular situations where KL fails; combining tensorisation with the right-hand inequality above gives $\mathrm{TV}(P^{\otimes n},Q^{\otimes n})\le\sqrt2\,\rho^{n/2}\sqrt{1-\rho^n}$-type bounds with no absolute-continuity hypothesis whatsoever.

**A10.** Take $P := \mathrm{Ber}(\varepsilon)$ and $Q := \mathrm{Ber}(0) = \delta_0$. Then $\mathrm{TV}(P,Q) = \varepsilon$, so the right-hand side is $C\varepsilon^2$, which tends to $0$. But $P(\{1\}) = \varepsilon>0 = Q(\{1\})$, so $P\not\ll Q$ and $D_{\mathrm{KL}}(P\Vert Q) = +\infty$ for every $\varepsilon>0$. No finite $C$ can work.

What Pinsker gives is a *one-way* control: small KL implies small TV, quantitatively, $\mathrm{TV}\le\sqrt{D_{\mathrm{KL}}/2}$. So KL is the stronger notion of closeness, and a KL bound is a licence to conclude statistical indistinguishability. What Pinsker does *not* give — and what the example above forbids in general — is any inference from small TV to small KL. Two distributions can be indistinguishable to any test at the $\varepsilon$ level while one assigns probability zero to an event the other allows, which is precisely the configuration that makes likelihood-based objectives blow up. This is why lower-bound proofs are stated in TV and computed in KL, never the other way round.

**A11.** Let $q(x) := \frac1m e^{-x/m}$ on $(0,\infty)$, which has mean $m$ and, by Proposition 11.11, $h(q) = 1+\log m$. Let $p$ be any density on $(0,\infty)$ with $\int xp(x)\,\mathrm{d}x = m$. Then, using $\log q(x) = -\log m - x/m$,
$$0 \ \le\ D_{\mathrm{KL}}(p\Vert q) = \int p\log p - \int p\log q = -h(p) + \log m + \frac{\mathbb{E}_p[X]}{m} = -h(p)+\log m + 1 .$$
Hence $h(p)\le 1+\log m = h(q)$, with equality if and only if $p = q$ almost everywhere (the equality case of Theorem 11.17). The maximal value is $1+\log m$ nats — for $m=1$, exactly $1$ nat.

The trick is worth naming: *the constraint enters through $\log q$ being affine in the constraint function*. That is exactly the hypothesis of Theorem 11.9 with $T(x) = x$ and support $(0,\infty)$: the max-entropy distribution subject to a fixed mean on the half-line is the one-parameter exponential family generated by $T(x)=x$, and the exponential distribution is that family. The Lagrange-multiplier route reaches the same answer but requires separate verification that the stationary point is a maximum; the KL route delivers the inequality, the equality case, and the value in one line.

**A12.** *(a)* The bound $1-\frac{I+\log2}{\log M}$ is non-positive exactly when $I+\log2\ge\log M$, i.e. $I\ge\log M-\log2$. Interpretation: the problem needs $\log M$ nats; Fano stops forbidding success as soon as the channel supplies all but $\log 2$ of them. The $\log2$ slack is not fundamental — it is the crude step $H_2(P_{\mathrm{e}})\le\log2$ in the proof, which is loose whenever $P_{\mathrm{e}}$ is far from $\tfrac12$. Sharper versions of Fano keep $H_2(P_{\mathrm{e}})$ implicit and invert numerically.

*(b)* Put $P_{\mathrm{e}} = 0$ into (11.6): $H_2(0) = 0$ and $0\cdot\log(M-1) = 0$, so $H(X\mid Y)\le 0$, and since conditional entropy is non-negative for discrete variables, $H(X\mid Y) = 0$. So error-free decoding is possible only if the observation already determines the answer almost surely. This is "no free lunch" stripped of all decoration: no algorithm, no architecture, no amount of computation can supply conditional entropy that the data does not contain. Everything an estimator does is a deterministic or randomised function of $Y$, and Fano's proof used only that.

**A13.** Write $\frac{\mathrm{d}P^\star}{\mathrm{d}R}(x) = \exp\bigl(\sum_j\lambda_j T_j(x)-\Lambda\bigr)$ with $\Lambda := \log\int\exp(\sum_j\lambda_jT_j)\,\mathrm{d}R$ the normaliser. Let $Q\in\mathcal{L}$ with $Q\ll R$. Split the logarithm through $P^\star$:
$$D_{\mathrm{KL}}(Q\Vert R) = \int q\log\frac{q}{r} = \int q\log\frac{q}{p^\star} + \int q\log\frac{p^\star}{r} = D_{\mathrm{KL}}(Q\Vert P^\star) + \mathbb{E}_Q\Bigl[\sum_j\lambda_jT_j - \Lambda\Bigr].$$
Now the key step, and it is the only step: because $Q\in\mathcal{L}$, we have $\mathbb{E}_Q[T_j] = \tau_j$, so the last expectation equals $\sum_j\lambda_j\tau_j-\Lambda$ — *a number that does not depend on which $Q\in\mathcal{L}$ we chose*. Applying the same identity with $Q := P^\star$ (which is in $\mathcal{L}$, by construction) gives $D_{\mathrm{KL}}(P^\star\Vert R) = 0 + \sum_j\lambda_j\tau_j-\Lambda$. Substituting back,
$$D_{\mathrm{KL}}(Q\Vert R) = D_{\mathrm{KL}}(Q\Vert P^\star)+D_{\mathrm{KL}}(P^\star\Vert R).$$

*Specialisation.* Take $R$ to be counting measure on a finite set (or Lebesgue measure), so that $D_{\mathrm{KL}}(Q\Vert R) = \int q\log q = -H(Q)$. The identity becomes $-H(Q) = D_{\mathrm{KL}}(Q\Vert P^\star)-H(P^\star)$, i.e.
$$H(P^\star)-H(Q) = D_{\mathrm{KL}}(Q\Vert P^\star),$$
which is Theorem 11.9's identity, and it re-derives the numerical check from §11.1: for the die constrained to mean $4.5$, $H(p_\lambda)-H(q) = 2.327905-1 = 1.327905$ bits $= D_{\mathrm{KL}}(q\Vert p_\lambda)$.

*Why this makes max-entropy a projection.* The identity is the Pythagorean theorem with $D_{\mathrm{KL}}(\cdot\Vert\cdot)$ in the role of squared distance, $\mathcal{L}$ in the role of an affine subspace, and $P^\star$ in the role of the foot of the perpendicular from $R$ to $\mathcal{L}$. The "right angle" is not a metaphor: it is the statement that the residual $D_{\mathrm{KL}}(P^\star\Vert R)$ is the same for every $Q\in\mathcal{L}$, which is exactly what orthogonality of the residual to the subspace means in the Euclidean case. Two structural consequences follow immediately. First, uniqueness: if two maximisers existed, each would be the projection of the other, forcing their KL divergence to zero. Second, the exponential-family form of $P^\star$ is not an ansatz but forced — the "direction" of the residual must be a linear combination of the constraint functions, and exponentiating a linear combination of the $T_j$ is what that looks like on the multiplicative scale. This is the same geometry that makes conditional expectation an orthogonal projection in Chapter 6, transplanted from $L^2$ to the KL geometry.

**A14.** Given an estimator $\hat\theta$, define the test
$$\psi := \begin{cases}0,& |\hat\theta-\theta_0|\le|\hat\theta-\theta_1|,\\ 1,&\text{otherwise.}\end{cases}$$
Suppose $\psi\ne j$. Then $\hat\theta$ is at least as close to $\theta_{1-j}$ as to $\theta_j$, so by the triangle inequality $2|\hat\theta-\theta_j| \ge |\hat\theta-\theta_j|+|\hat\theta-\theta_{1-j}| \ge |\theta_0-\theta_1| \ge 2s$, giving $|\hat\theta-\theta_j|\ge s$ and hence $(\hat\theta-\theta_j)^2 \ge s^2\mathbf{1}\{\psi\ne j\}$. Taking expectations under $P_j$,
$$\mathbb{E}_j\bigl[(\hat\theta-\theta_j)^2\bigr]\ \ge\ s^2\,\mathbb{P}_j(\psi\ne j).$$
Now bound the maximum by the average over the two hypotheses and apply Proposition 11.37:
$$\max_{j}\mathbb{E}_j\bigl[(\hat\theta-\theta_j)^2\bigr]\ \ge\ \frac12\sum_{j=0}^1 s^2\,\mathbb{P}_j(\psi\ne j)\ =\ \frac{s^2}{2}\Bigl[\mathbb{P}_0(\psi\ne0)+\mathbb{P}_1(\psi\ne1)\Bigr]\ \ge\ \frac{s^2}{2}\bigl(1-\mathrm{TV}(P_0,P_1)\bigr).$$
Since $\hat\theta$ was arbitrary, the same bound holds for the infimum, and $\sup_{\theta\in\Theta}\ge\max_{j\in\{0,1\}}$ completes the argument.

*The factor you should identify.* It is the $\tfrac12$ from "max $\ge$ average". A tempting shortcut is to write $\max_j\mathbb{E}_j \ge s^2\max_j\mathbb{P}_j(\psi\ne j)$ and then hope to bound $\max_j\mathbb{P}_j(\psi\ne j)$ below by $1-\mathrm{TV}$; but Proposition 11.37 bounds the *sum* of the two error probabilities, not the maximum, and the maximum can only be bounded by half the sum. There is no way to recover the lost factor without extra structure — and in fact the factor is real: a test can have one error probability equal to zero, in which case only the sum is informative.

**A15 (the local geometry of KL).**

*(a)* Set $\chi(\delta) := \int p_{\theta+\delta}\bigl(\log p_{\theta+\delta}-\log p_\theta\bigr)\,\mathrm{d}\mu$, well defined near $\delta = 0$ since $p_{\theta'}>0$ $\mu$-a.e. (hypothesis of Proposition 11.23) makes both logarithms finite a.e.; positivity is used *here*, to write $\nabla p = p\,\nabla\log p$ and to divide by densities at all. First derivative: the integrand is $C^1$ in $\delta$ for a.e. $x$ by (i), and (ii) supplies an integrable envelope, so differentiation passes under the integral (Dominated Convergence, Ch. 2):
$$\nabla\chi(\delta) = \int \nabla p_{\theta+\delta}\bigl(\log p_{\theta+\delta} - \log p_\theta\bigr)\,\mathrm{d}\mu + \int p_{\theta+\delta}\,\nabla\log p_{\theta+\delta}\,\mathrm{d}\mu.$$
The second integral equals $\int \nabla p_{\theta+\delta}\,\mathrm{d}\mu = \nabla_{\delta}\int p_{\theta+\delta}\,\mathrm{d}\mu = \nabla(1) = 0$ for every $\delta$ in the neighbourhood — passing the gradient outside the integral is again (ii) — so only the first integral survives, and at $\delta=0$ it vanishes because the log-ratio is identically zero. Hence $\chi(0)=0$ and $\nabla\chi(0)=0$. Second derivative of the surviving term, again under the integral by (i)–(ii):
$$\nabla^2\chi(\delta) = \int \nabla^2 p_{\theta+\delta}\,\bigl(\log p_{\theta+\delta}-\log p_\theta\bigr)\,\mathrm{d}\mu + \int \nabla p_{\theta+\delta}\,\nabla\log p_{\theta+\delta}^{\top}\,\mathrm{d}\mu,$$
and at $\delta = 0$ the first term dies with the log-ratio while the second becomes $\int p_\theta\,\nabla\log p_\theta\,\nabla\log p_\theta^\top\,\mathrm{d}\mu = I(\theta)$, finite by (iii). Since $\chi$ is $C^2$ near $0$, Taylor with the Peano remainder gives $\chi(\delta) = \tfrac12\delta^\top I(\theta)\delta + o(\lVert\delta\rVert^2)$. Note what was *not* needed: the information equality. The reverse direction produces $I(\theta)$ as the outer product of scores directly; it is the forward direction that routes through $\mathbb{E}[\nabla^2\log p_\theta] = -I(\theta)$.

*(b)* Add the two displays of Proposition 11.23: $J(P_\theta,P_{\theta+\delta}) = 2\cdot\tfrac12\delta^\top I\delta + o(\lVert\delta\rVert^2) = \delta^\top I(\theta)\delta + o(\lVert\delta\rVert^2)$. Symmetrising cannot change the local geometry because the asymmetry it removes was never present at second order — forward and reverse KL already share the quadratic term, so their sum simply doubles it.

*(c)* Write both divergences and expand each logarithm with $\log(1+u) = u - \tfrac{u^2}{2} + \tfrac{u^3}{3} - \cdots$. Forward:
$$D_{\mathrm{KL}}\bigl(\mathrm{Ber}(p)\Vert\mathrm{Ber}(p+\delta)\bigr) = -p\log\Bigl(1+\tfrac{\delta}{p}\Bigr) - (1-p)\log\Bigl(1-\tfrac{\delta}{1-p}\Bigr) = \frac{\delta^2}{2}\Bigl(\frac1p+\frac1{1-p}\Bigr) + \frac{\delta^3}{3}\Bigl(\frac{1}{(1-p)^2}-\frac{1}{p^2}\Bigr) + O(\delta^4),$$
the linear terms $-\delta + \delta$ cancelling. Reverse: expanding $(p+\delta)\log\frac{p+\delta}{p} + (1-p-\delta)\log\frac{1-p-\delta}{1-p}$ term by term gives
$$D_{\mathrm{KL}}\bigl(\mathrm{Ber}(p+\delta)\Vert\mathrm{Ber}(p)\bigr) = \frac{\delta^2}{2}\Bigl(\frac1p+\frac1{1-p}\Bigr) + \frac{\delta^3}{6}\Bigl(\frac{1}{(1-p)^2}-\frac{1}{p^2}\Bigr) + O(\delta^4).$$
Both quadratic terms are $\tfrac12\delta^2 I(p)$ with $I(p) = \tfrac1p+\tfrac1{1-p} = \tfrac{1}{p(1-p)}$, as required. The cubic coefficients differ by
$$\Bigl(\tfrac13-\tfrac16\Bigr)\Bigl(\frac{1}{(1-p)^2}-\frac{1}{p^2}\Bigr) = \frac16\Bigl(\frac{1}{(1-p)^2}-\frac{1}{p^2}\Bigr),$$
which at $p = 0.3$ is $\tfrac16(1/0.49 - 1/0.09) = \tfrac16(2.040816 - 11.111111) = -1.511716$. The measured $(\text{fwd}-\text{rev})/\delta^3 = -1.5051$ at $\delta = 0.001$ sits within $0.5\%$ of this limit; the residual gap is the $O(\delta)$ quartic correction, and the sequence of measured values in §11.3.5 ($-0.9816$, $-1.2174$, $-1.4469$, $-1.5051$ at $\delta = 0.1, 0.05, 0.01, 0.001$) is converging to it monotonically. The thinking to retain: the asymmetry of KL is governed by the *skewness* of the family — here the factor $1/(1-p)^2 - 1/p^2$, which vanishes at $p = \tfrac12$ where the family is symmetric and blows up towards the boundary.

*(d)* Near $\theta_{\mathrm{old}}$ the constraint $\overline{D}_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}}\Vert\pi_{\theta_{\mathrm{old}}+\delta})\le\varepsilon$ reads, by Proposition 11.23 averaged over states, $\tfrac12\delta^\top\bar I\,\delta\le\varepsilon$ — an ellipsoid whose axes are the eigenvectors of the average Fisher matrix $\bar I$, long where the policy is insensitive and short where it is sensitive. It is not a ball because sensitivity is wildly anisotropic in weight space. Concrete situation: one logit is saturated at $+10$ on the common states (its action already chosen with near-certainty; $I \approx p(1-p) \approx 0$ along that direction), while another sits at $0$ on those same states (a genuine $50/50$ choice; $I$ maximal). A Euclidean ball prices a unit step in each direction identically. Calibrate its radius so that useful progress is possible along the saturated direction, and it simultaneously permits a unit step on the $50/50$ logit — moving that action probability from $0.500$ to $0.731$, a large distributional jump on frequently visited states, which is exactly the destabilising update TRPO exists to forbid. The KL ellipsoid charges each direction at its true policy-space price, so it permits the first step and blocks the second with a single scalar $\varepsilon$.

## Part B

A general remark before the individual solutions. Every one of these labs has a closed form available somewhere, and the discipline being taught is to *compute the closed form first and the estimate second*. An estimator that agrees with a known answer on a synthetic problem of the same dimension and sample size as your real problem has earned a little trust; one that has never been checked against a closed form has earned none. All reference numbers below were produced by running the code, with seeds and sizes stated.

**B1 (forward versus reverse KL).**

*(a)* Represent $p$ and $q_\theta$ on a grid over $[-20,20]$ with step $10^{-4}$ and evaluate $D_{\mathrm{KL}}(p\Vert q_\theta) = \int p\log(p/q_\theta)$ by the trapezoid rule; minimise over $(\mu,\log\sigma)$ with Nelder–Mead to tolerance $10^{-9}$. The optimum is $\mu = 1.600$, $\sigma = 3.800$, with $D_{\mathrm{KL}}(p\Vert q) = 0.724222$ nats.

Check against theory. $D_{\mathrm{KL}}(p\Vert q_\theta) = -h(p) + \mathbb{E}_p[-\log q_\theta]$, and only the second term depends on $\theta$; expanding the Gaussian log-density shows it depends on $p$ only through $\mathbb{E}_p[X]$ and $\mathbb{E}_p[X^2]$. So the forward-KL optimum is exactly the moment-matching Gaussian. Directly: $\mathbb{E}_p[X] = 0.3(-4)+0.7(4) = 1.6$, and $\mathrm{Var}_p(X) = \mathbb{E}[X^2]-1.6^2 = (0.3\cdot17+0.7\cdot17)-2.56 = 17-2.56 = 14.44$, so $\sigma = \sqrt{14.44} = 3.8$. The numerical optimiser agrees to three decimals, which is the check the exercise asks for. Note what this means: for forward KL you did not need an optimiser at all.

*(b)* Reverse KL, $D_{\mathrm{KL}}(q_\theta\Vert p)$, has two local minima:

| $\mu_0$ | optimum reached | $D_{\mathrm{KL}}(q\Vert p)$ | $D_{\mathrm{KL}}(p\Vert q)$ |
|---|---|---|---|
| $-6,\ -4,\ -1$ | $N(-4.000,\ 1.001^2)$ | $1.203826$ | $21.732651$ |
| $0,\ 1,\ 4,\ 6$ | $N(4.000,\ 1.000^2)$ | $0.356615$ | $8.978460$ |

The right mode is the global optimum, as it must be: it carries $70\%$ of the mass, and reverse KL rewards sitting where $p$ is largest. The left basin is a genuine local minimum, not an optimiser artefact — three different initialisations converge to it and it is stable to tightening the tolerance.

*(c)* $p(X<0) = 0.300013$ (the tiny excess over $0.3$ is the left tail of the right-hand Gaussian component). The forward fit gives $q(X<0) = 0.336858$, a relative error of $12\%$. The reverse global fit gives $q(X<0) = 3.2\times10^{-5}$, which understates the true probability by a factor of about $9000$.

*(d)* For an importance-sampling proposal, take the forward fit: importance weights $p/q$ blow up wherever $q$ is small and $p$ is not, so a proposal that misses $30\%$ of the target's mass produces weights with infinite or astronomically large variance, and the estimator silently fails. Mass-covering is a safety property. For a generative model whose samples are shown to users, the reverse fit is usually preferable: every sample it produces lands in a high-density region of $p$, so samples look plausible, at the cost of never producing anything from the left mode. Mode-seeking buys sample quality and pays in diversity — which is the trade-off behind essentially every complaint about mode collapse in generative models, and behind the observed narrowing of output distributions under reverse-KL-regularised fine-tuning.

**B2 (three MI estimators).**

*(a)* Invert $I = -\tfrac12\log(1-\rho^2)$ to get $\rho = \sqrt{1-e^{-2I}}$: $I = 0.2\Rightarrow\rho = 0.57418$; $0.5\Rightarrow0.79506$; $1\Rightarrow0.92987$; $2\Rightarrow0.99080$; $4\Rightarrow0.99983$; $6\Rightarrow1.00000$ (to five decimals — note that beyond about $4$ nats the correlation is numerically indistinguishable from $1$, which is itself a warning).

*(b)* The full table is Worked Example 11.15(b); do not read the solution before producing your own. The four columns at $n = 20{,}000$, seed $0$, are binning, KSG, MINE held-out, MINE on training batches.

*(c)* The binning ceiling is analytic: with $m$ equal-mass bins per axis, the discretised variables $X^{(m)},Y^{(m)}$ satisfy $I(X^{(m)};Y^{(m)})\le\min\{H(X^{(m)}),H(Y^{(m)})\} = \log m$, so the estimate cannot exceed $\log 16 = 2.7726$ nats. The table confirms it: $2.717$ at $I = 6$, within $2\%$ of the ceiling, and it would report $2.7$ for a true value of $600$. This is the data processing inequality functioning as an estimator ceiling — binning is a Markov kernel applied to both variables, so Theorem 11.28 caps the result.

KSG does well throughout, but note the conditions: two dimensions, a smooth Gaussian, $20{,}000$ samples. Its bias grows quickly with dimension, and it has no ceiling, which means it also has no protection against reporting a large number for the wrong reason.

MINE tracks the truth to about $2$ nats and then separates into two curves. At $I = 6$ it reports $6.168$ on its own training batches — an *over*estimate of a quantity it is supposed to lower-bound — and $3.127$ on fresh batches, a $48\%$ underestimate. The gap between the last two columns is the entire diagnostic: report both, or report nothing.

*(d)* With the exactly optimal critic for two Gaussians differing in mean by $\Delta$ (so $D_{\mathrm{KL}} = \Delta^2/2$), namely $T^\star(x) = -\Delta x+\Delta^2/2$, the plug-in DV estimate over $500$ independent batches gives, at $D_{\mathrm{KL}} = 4$: $4.656\pm0.993$ (batch $64$), $4.222\pm0.578$ ($512$), $4.069\pm0.328$ ($4096$). At $D_{\mathrm{KL}} = 8$ the same three batch sizes give $10.196\pm1.567$, $9.181\pm1.080$, $8.597\pm0.753$; at $D_{\mathrm{KL}} = 18$, $25.658\pm2.711$, $23.354\pm1.931$, $21.809\pm1.692$.

*Why upwards.* The estimator is $\hat I = \frac1b\sum_i T^\star(X_i) - \log\bigl(\frac1b\sum_j e^{T^\star(Y_j)}\bigr)$. The first term is unbiased. The second involves $-\log$ of an unbiased estimate $\hat S$ of $S := \mathbb{E}_Q[e^{T^\star}] = 1$, and $-\log$ is convex, so Jensen gives $\mathbb{E}[-\log\hat S]\ge-\log\mathbb{E}[\hat S] = 0$. The bias is therefore non-negative *for any fixed critic*, optimal or not. Note the direction is the opposite of the failure mode in (b): DV plug-in with a fixed critic over-reports because of the logarithm, while a trained critic evaluated out of sample under-reports because it has overfitted.

*Why it grows with $I$.* The delta method gives $\mathbb{E}[-\log\hat S]\approx \mathrm{Var}(\hat S)/(2S^2) = \mathrm{Var}_Q(e^{T^\star})/(2b)$, and with $e^{T^\star} = \mathrm{d}P/\mathrm{d}Q$ this variance is exactly $\chi^2(P\Vert Q) = \mathbb{E}_Q[(\mathrm{d}P/\mathrm{d}Q)^2]-1$. For Gaussians differing by $\Delta$, $\chi^2 = e^{\Delta^2}-1 = e^{2D_{\mathrm{KL}}}-1$. So the predicted bias is $(e^{2D_{\mathrm{KL}}}-1)/(2b)$ — **exponential in the quantity being estimated**. Comparing prediction with measurement:

| true $D_{\mathrm{KL}}$ | $\chi^2$ | predicted / observed bias, $b=64$ | $b=512$ | $b=4096$ |
|---|---|---|---|---|
| $1$ | $6.39$ | $0.050$ / $0.038$ | $0.0062$ / $0.005$ | $0.0008$ / $-0.001$ |
| $2$ | $53.6$ | $0.419$ / $0.154$ | $0.052$ / $0.041$ | $0.0065$ / $0.006$ |
| $4$ | $2980$ | $23.3$ / $0.656$ | $2.91$ / $0.222$ | $0.364$ / $0.069$ |
| $8$ | $8.9\times10^6$ | $6.9\times10^4$ / $2.196$ | — / $1.181$ | — / $0.597$ |

Read that table carefully, because it teaches two things. The delta-method prediction is quantitatively good exactly while $\chi^2/b\ll1$ (rows $1$ and $2$), and it becomes wildly pessimistic once $\chi^2/b\gg1$ (rows $4$ and $8$) — the expansion is an asymptotic statement in $b$, and at $D_{\mathrm{KL}} = 4$ with $b = 64$ we are nowhere near its regime. But the *qualitative* conclusion is confirmed by the measured column: the bias grows monotonically with $D_{\mathrm{KL}}$ and shrinks with $b$, and to bring it under control at $D_{\mathrm{KL}} = 18$ you would need $b$ of order $e^{36}$. The honest summary is that the DV plug-in requires batch size exponential in the value being estimated, and that no amount of architecture search changes this, because the calculation above never mentioned the critic class — it used the *optimal* critic.

**B3 (as overlap vanishes).**

*(a)* With $p = \mathbf{1}_{[0,1]}$ and $q_a = \mathbf{1}_{[a,1+a]}$, the overlap is the interval $[a,1]$ when $a\le1$, of length $1-a$, and empty otherwise. Hence $\int\min(p,q_a) = \max(0,1-a)$ and $\mathrm{TV} = 1-\max(0,1-a) = \min(a,1)$. For JS, the mixture $m = \tfrac12(p+q_a)$ equals $\tfrac12$ on $[0,a)\cup(1,1+a]$ and $1$ on $[a,1]$, so $D_{\mathrm{KL}}(P\Vert M) = \int_0^a\log 2 = a\log 2$ and likewise for $Q_a$; averaging gives $\mathrm{JS} = \min(a,1)\log2$. For $a>0$, $P$ charges $[0,a)$ where $q_a = 0$, so $D_{\mathrm{KL}}(P\Vert Q_a) = +\infty$. Finally $F^{-1}(u) = u$ and $G^{-1}(u) = u+a$, so Theorem 11.42 gives $W_1 = \int_0^1 a\,\mathrm{d}u = a$ for every $a\ge0$.

*(b)* The table is Worked Example 11.18. Numerical checks: quadrature for JS at $a = 0.3$ returns $0.207944$ against the closed form $0.3\log2 = 0.207944$; the quantile formula on a $201{,}001$-point grid returns $W_1 = 0.299999$ at $a=0.3$ and $1.499997$ at $a=1.5$.

*(c)* For $N(0,1)$ against $N(\Delta,1)$, all four have closed forms — $D_{\mathrm{KL}} = \Delta^2/2$, $\mathrm{TV} = 2\Phi(\Delta/2)-1$, $W_1 = W_2 = \Delta$, and JS by quadrature. The values at $\Delta = 0.5,1,2,4,8$ are JS $= 0.030311$, $0.111421$, $0.336831$, $0.632720$, $0.693054$ nats and TV $=0.197413$, $0.382925$, $0.682689$, $0.954500$, $0.999937$.

*(d)* The structural difference is *support*. Two Gaussians are mutually absolutely continuous for every $\Delta$, so Proposition 11.43 never applies and JS stays strictly below $\log2$; the two shifted uniforms become mutually singular the moment $a\ge1$, and then JS is pinned at $\log2 = 0.693147$ with derivative zero in $a$.

But now read the Gaussian column honestly, because the naive moral is too kind to JS. At $\Delta = 8$ the JS divergence is $0.693054$ against a ceiling of $0.693147$: it has used up $99.987\%$ of its range, and $\partial\mathrm{JS}/\partial\Delta$ is of order $e^{-\Delta^2/8}$. So the gradient exists but is exponentially small, and in floating point it is indistinguishable from the uniform case. Meanwhile $W_1 = \Delta$ has gradient $1$ at every separation. The lesson is that absolute continuity gives you a nonzero gradient in principle and transport geometry gives you a *useful* one; the uniform example is the clean version of a phenomenon that is present, in softened form, whenever two distributions are far apart relative to their scale.

**B4 (how tight is Pinsker?).** The Bernoulli table is Worked Example 11.16; the ratios run from $0.9999$ at $(0.5,0.51)$ down to $0.2532$ at $(0.01,0.02)$.

Gaussians $N(0,1)$ versus $N(\Delta,1)$: at $\Delta = 0.1$, $\mathrm{TV} = 0.039878$ against a bound of $0.05$; at $\Delta=0.5$, $0.197413$ against $0.25$; at $\Delta=1$, $0.382925$ against $0.5$; at $\Delta=2$, $0.682689$ against $1$; at $\Delta=4$, $0.954500$ against $2$ — vacuous.

Pinsker is nearly an equality when the two distributions are close *and* near the maximum-variance configuration $p\approx q\approx\tfrac12$. The reason is visible in the proof: the only inequality used is $p(1-p)\le\tfrac14$, which is tight exactly at $p = \tfrac12$. It is badly loose in two regimes. First, near the boundary of the simplex: $(0.01,0.02)$ has the same TV as $(0.5,0.51)$ but a bound almost four times larger, because $p(1-p) = 0.0099$ is far from $\tfrac14$ — small probabilities carry large log-ratios. Second, whenever $D_{\mathrm{KL}}>2$ the bound exceeds $1$ and says nothing at all, which is the practical ceiling on using KL to certify indistinguishability.

**B5 (bits per dimension audit).** $d = 3072$, $\log 2 = 0.693147$.

*(a)* $\mathrm{bpd} = -\log p(x)/(d\log2)+8$. At $6000$ nats: $-6000/2129.36+8 = -2.8178+8 = 5.1822$. At $8000$ nats: $-3.7570+8 = 4.2430$. The second model is better by $0.9392$ bits per pixel, or exactly $2000$ nats per image.

*(b)* Uniform noise on the 8-bit grid assigns every image probability $256^{-d}$, so $\mathrm{bpd} = 8$ exactly; equivalently $\log p(x) = 0$ on the unit-cube scale, since the density is $1$ on $[0,1]^d$. For $2.9$ bpd, $\log p(x) = (8-2.9)\times3072\times\log2 = 10859.7$ nats.

*(c)* The scale conversion (Proposition 11.50(1)) is $\log p_{[0,1]} = \log p_{[0,255]} + d\log 256 = -10000+17034.79 = 7034.79$ nats, i.e. $\mathrm{bpd} = -7034.79/2129.36+8 = 4.696$. So it beats the $6000$-nat model ($5.1822$) and loses to the $8000$-nat model ($4.2430$). Note that the naive comparison — "$-10000$ versus $6000$, so it is $16000$ nats worse" — gets both the ranking and the magnitude wrong, and the entire error is a single missing constant.

*(d)* The checklist: (i) which scale is the data on, $[0,1]^d$ or $[0,255]^d$, and hence what is $d\log s$; (ii) is the reported quantity a discrete log-probability or a continuous log-density, and if continuous, was the data dequantised (without which the number is unbounded and meaningless); (iii) is it per image, per dimension, in nats or in bits; (iv) is it an exact likelihood or a bound — for VAEs an ELBO is a lower bound on $\log p$, so a *worse* reported number can come from a looser bound rather than a worse model, and importance-weighted variants improve the number without changing the model.

**B6 (the $O(1/n)$ of source coding).**

*(a)* Huffman on $(\tfrac18,\tfrac38,\tfrac38,\tfrac18)$ gives lengths $(3,2,1,3)$ for outcomes $(0,1,2,3)$ and $\mathbb{E}[L] = 3(\tfrac18)+2(\tfrac38)+1(\tfrac38)+3(\tfrac18) = 1.875$ bits, against $H(X) = 1.811278$: redundancy $0.063722$ bits.

*(b)* Blocks of $2$ (a $16$-letter alphabet) give $3.65625$ bits per block, i.e. $1.828125$ per symbol; blocks of $3$ ($64$ letters) give $5.460938$ per block, i.e. $1.820313$ per symbol.

*(c)* Redundancies: $0.063722$, $0.016847$, $0.009035$ for $n = 1,2,3$. Theorem 11.46 guarantees redundancy $<1/n$, i.e. $<1$, $<0.5$, $<0.333$ — comfortably satisfied, and the observed decay is faster than $1/n$ here. What the data rules out is the two wrong hypotheses: a constant redundancy (the values fall by a factor of $7$ over three points) and $O(1/\sqrt n)$ decay (which would predict $0.045$ and $0.037$ at $n=2,3$, far above the observed values). One caution worth stating: Huffman redundancy for a fixed source is not monotone in the block length in general — it depends on the arithmetic of the block probabilities — so three points establish a trend, not a theorem. The theorem is Theorem 11.46.

*(d)* With $p = (0.4,0.3,0.2,0.1)$ and the uniform code (all lengths $2$), $\mathbb{E}_p[L] = 2$ bits while $H(p) = 1.846439$, so the excess is $0.153561$ bits; and $D_{\mathrm{KL}}(p\Vert q) = \sum_i p_i\log_2(p_i/0.25) = 0.153561$ bits. They agree to six decimals, as Proposition 11.48 requires — and note that here the identity is *exact* rather than approximate, because the uniform distribution is dyadic so its ideal code lengths are integers.

**B7 (entropic bias of Sinkhorn).**

*(a)* In the log domain, iterate $f \leftarrow \varepsilon\bigl[\log a - \operatorname{logsumexp}_j((g_j - c_{ij})/\varepsilon)\bigr]$ and symmetrically for $g$, then form $\pi_{ij} = \exp((f_i+g_j-c_{ij})/\varepsilon)$. With $20{,}000$ iterations the marginal violations are at machine precision ($\le1.4\times10^{-15}$):

| $\varepsilon$ | $\langle\pi^\star,c\rangle$ | $H(\pi^\star)$ (nats) |
|---|---|---|
| $1.00$ | $1.155913$ | $2.953949$ |
| $0.30$ | $1.001527$ | $2.724325$ |
| $0.10$ | $1.000000$ | $2.718473$ |
| $0.03$ | $1.000000$ | $2.718473$ |

*(b)* The entropy decreases as $\varepsilon$ falls, because the objective $\langle\pi,c\rangle+\varepsilon D_{\mathrm{KL}}(\pi\Vert a\otimes b)$ weights the entropic term less and the plan is free to concentrate. The upper end is bounded by the independent coupling's entropy, $2\log 5 = 3.218876$ nats, which $\varepsilon = 1$ approaches.

There is something genuinely instructive in the lower end, and it is easy to miss. The entropy stalls at $2.718473$, not at $\log 5 = 1.609438$ — the entropy of the deterministic "shift everything right by one" plan. The reason is that the $W_1$ optimal plan is **not unique** here: the cost $|x-y|$ is convex but not *strictly* convex, so any plan that never "crosses" achieves the same total cost. For instance, sending $x=1\mapsto y=1$ (cost $0$) and $x=0\mapsto y=2$ (cost $2$) with the rest shifted by one also totals $1.000000$. The $\varepsilon\downarrow0$ limit of Sinkhorn is the maximum-entropy element of the set of optimal plans, which is strictly more spread out than the monotone one. Practical consequence: if you use entropic OT to recover a *matching* rather than a cost, check whether your ground cost is strictly convex, because with $p=1$ the matching you get back is not the one you probably intended.

*(c)* At $\varepsilon = 0.001$ the naive implementation forms $K_{ij} = e^{-c_{ij}/\varepsilon}$ directly. With $c_{ij}$ up to $5$, this asks for $e^{-5000}$, which underflows to exactly $0$ in double precision; the kernel matrix becomes structurally zero away from the diagonal band, the row and column rescalings divide by zero, and the iteration returns `nan`. The log-domain formulation avoids this by never exponentiating anything larger in magnitude than the shifted arguments inside `logsumexp`, which is the reason every production implementation is written that way.

## Part C

**C1 (DPI for a general $f$-divergence).**

*(a)* Fix $\mu := Q$ and $r := \mathrm{d}P/\mathrm{d}Q$. Let $X\sim Q$ and $Y\mid X\sim K(\cdot\mid X)$, so that $Y\sim QK$. The claim is that $\mathbb{E}[r(X)\mid Y]$ is a version of $\mathrm{d}(PK)/\mathrm{d}(QK)$, and verifying it is just the definition: for measurable $B$,
$$\int_B \mathbb{E}\bigl[r(X)\mid Y\bigr]\,\mathrm{d}(QK) = \mathbb{E}\bigl[r(X)\mathbf{1}_B(Y)\bigr] = \int_{\mathcal{X}}\!\int_{\mathcal{Y}}\mathbf{1}_B(y)\,K(\mathrm{d}y\mid x)\,r(x)\,\mathrm{d}Q(x) = \int_{\mathcal{X}} K(B\mid x)\,\mathrm{d}P(x) = (PK)(B),$$
where the first equality is the tower property (Ch. 6) and the third uses $r\,\mathrm{d}Q = \mathrm{d}P$. Since this holds for every $B$, the conditional expectation is the required derivative. Now apply conditional Jensen (Ch. 6) to the convex $f$, then the tower property again:
$$D_f(PK\Vert QK) = \mathbb{E}\Bigl[f\bigl(\mathbb{E}[r(X)\mid Y]\bigr)\Bigr]\ \le\ \mathbb{E}\Bigl[\mathbb{E}\bigl[f(r(X))\mid Y\bigr]\Bigr] = \mathbb{E}\bigl[f(r(X))\bigr] = D_f(P\Vert Q).$$

*Where it breaks.* Every step used $r = \mathrm{d}P/\mathrm{d}Q$ as an honest finite-valued function, which requires $P\ll Q$. If $P$ charges a set $N$ with $Q(N) = 0$, no such $r$ exists, the conditional expectation is not defined on $N$, and the display above is vacuous. That is not a corner case to be waved away: it is precisely the mutually singular configuration of Proposition 11.44, which is the case the chapter cares most about.

*How the convention repairs it.* Theorem 11.32's proof never forms a ratio. It works with the pair of densities $(p,q)$ against a common dominating $\mu$ and the perspective function $g(u,v) = v f(u/v)$, extended to the boundary by $g(u,0) := u f'(\infty)$. On $\{q = 0, p>0\}$ the integrand is then $p\,f'(\infty)$ — the limiting slope of $f$, which is the only value consistent with the positive homogeneity that makes $D_f$ independent of $\mu$. Jointly convexity of $g$ plus that homogeneity is all the proof uses, so it goes through with no absolute-continuity hypothesis, and it specialises to the conditional-Jensen argument above whenever $P\ll Q$. The moral: divide by a density only after you have checked it is positive, and if you cannot, work with the pair instead of the quotient.

*(b)* With $f(t) = t\log t$, $g(u,v) = u\log(u/v)$ and $D_f = D_{\mathrm{KL}}$, so the above is the DPI for KL. For the Markov chain $X\to Y\to Z$, apply it with the kernel $\tilde K\bigl(\mathrm{d}(x',z)\mid(x,y)\bigr) := \delta_x(\mathrm{d}x')\,P_{Z\mid Y=y}(\mathrm{d}z)$, which acts as the identity on the first coordinate and as $Z$'s conditional law on the second. Then $P_{XY}\tilde K = P_{XZ}$ and $(P_X\otimes P_Y)\tilde K = P_X\otimes P_Z$ — the first because of the Markov property, the second by independence — so
$$I(X;Z) = D_{\mathrm{KL}}(P_{XZ}\Vert P_X\otimes P_Z)\ \le\ D_{\mathrm{KL}}(P_{XY}\Vert P_X\otimes P_Y) = I(X;Y),$$
which is Theorem 11.28. (Exercise A8 gives the alternative proof, with the deficit identified as $I(X;Y\mid Z)$.)

*(c) Interpretation.* Suppose we propose to learn $Z = g_\phi(X)$ by maximising $I(Z;Y)$. The chain $Y\to X\to Z$ is Markov, because $Z = g_\phi(X)$ depends on $Y$ only through $X$: conditionally on $X$, the representation $Z$ is independent of $Y$. Part (b) then gives $I(Z;Y)\le I(X;Y)$ with equality when $g$ is injective — in particular for $g = \mathrm{id}$. So the objective is maximised by the identity map, and more generally by any invertible encoder, of which there are many with wildly different properties. **An objective whose maximiser is "change nothing" expresses no preference at all.** Any apparent success of such a method is coming from the optimisation being incomplete, from the estimator's bias, or from the architecture — not from the stated objective.

What actually makes a representation good, in these terms, is a *constraint* that the identity map violates. Candidate constraints: a dimension bound ($Z\in\mathbb{R}^k$ with $k\ll d$); a rate bound ($I(X;Z)\le R_0$, or equivalently a KL penalty to a prior, which is the VAE); a smoothness or Lipschitz bound; invariance to a specified nuisance transformation ($I(Z;N) = 0$ for a nuisance $N$); or a restriction that a *simple* decoder — a linear probe — must be able to read $Y$ off $Z$. Every one of these makes the problem non-trivial, and each picks out a different notion of "good". This is why the information bottleneck objective $\min I(X;Z)-\beta I(Z;Y)$ is not vacuous: the first term is a competing cost that the identity map maximises, so the optimum is a genuine trade-off, and by Proposition 11.55 the term is precisely the rate of a stochastic encoder. Note also the necessary fine print: for the IB objective to be meaningful, the encoder must be genuinely stochastic, because for a deterministic injective encoder $I(X;Z)$ is $+\infty$ (continuous $X$) or a weight-independent constant (discrete $X$).

On "layer $\ell+1$ contains more label information than layer $\ell$": in the DPI sense this is false, and unavoidably so, since $Y\to X\to Z_\ell\to Z_{\ell+1}$ is a Markov chain and hence $I(Z_{\ell+1};Y)\le I(Z_\ell;Y)$. The charitable reading, which is usually what people mean and is often true, is about *accessibility*: the label may be linearly decodable from $Z_{\ell+1}$ and not from $Z_\ell$, so for the restricted decoder class actually in use, the later layer is more useful. That is a statement about a constrained decoder, and it should be stated that way — measured, for example, by the accuracy of a linear probe — rather than as a claim about information, where it is provably wrong.

**C2 (Donsker–Varadhan, $f$-GAN, MINE).**

*(a)* Fix $T$ with $Z_T := \mathbb{E}_Q[e^T]<\infty$ and define the **Gibbs tilt** $P_T$ by $\frac{\mathrm{d}P_T}{\mathrm{d}Q} := e^{T}/Z_T$, a probability measure. Then, for $P\ll Q$,
$$D_{\mathrm{KL}}(P\Vert Q) - \Bigl(\mathbb{E}_P[T]-\log Z_T\Bigr) = \mathbb{E}_P\left[\log\frac{\mathrm{d}P}{\mathrm{d}Q} - T + \log Z_T\right] = \mathbb{E}_P\left[\log\frac{\mathrm{d}P/\mathrm{d}Q}{e^{T}/Z_T}\right] = \mathbb{E}_P\left[\log\frac{\mathrm{d}P}{\mathrm{d}P_T}\right] = D_{\mathrm{KL}}(P\Vert P_T)\ \ge\ 0 .$$
So the DV objective is a lower bound for every admissible $T$, and the slack is *exactly* $D_{\mathrm{KL}}(P\Vert P_T)$ — a much more informative statement than "$\le$". Equality holds iff $P_T = P$, i.e. $T = \log\frac{\mathrm{d}P}{\mathrm{d}Q}+c$ for a constant $c$. Taking the supremum gives the representation. (If $P\not\ll Q$ the left side is $+\infty$ and one checks the supremum is also $+\infty$ by taking $T = t\mathbf{1}_A$ on a set with $P(A)>0=Q(A)$ and letting $t\to\infty$.)

Invariance: $\mathbb{E}_P[T+c]-\log\mathbb{E}_Q[e^{T+c}] = \mathbb{E}_P[T]+c-\log\mathbb{E}_Q[e^T]-c$, unchanged. Consequence: a trained critic identifies the log-density-ratio only up to an additive constant. So critic *differences* $T(x_1)-T(x_2)$ are meaningful estimates of $\log\frac{p(x_1)q(x_2)}{q(x_1)p(x_2)}$, while critic *values* are not calibrated log-likelihood-ratios and should not be exponentiated and used as importance weights without separately estimating the normaliser — a mistake that appears in practice more often than it should.

*(b)* Let $f$ be convex, lower semicontinuous, $f(1)=0$, with conjugate $f^*(s) = \sup_{t>0}\{st-f(t)\}$. Fenchel–Moreau gives $f(t) = \sup_s\{st-f^*(s)\}$. Substituting $t = p(x)/q(x)$ and choosing, for each $x$, the value $s = T(x)$ (any measurable $T$, not necessarily the maximiser):
$$D_f(P\Vert Q) = \int q\,f\!\left(\frac pq\right)\mathrm{d}\mu \ \ge\ \int q\left[T\frac pq - f^*(T)\right]\mathrm{d}\mu = \mathbb{E}_P[T]-\mathbb{E}_Q[f^*(T)],$$
with equality when $T = f'(p/q)$ pointwise. This is Theorem 11.34.

For the GAN, take $f_{\mathrm{JS}}(t) := \tfrac12\bigl[t\log t-(t+1)\log\tfrac{t+1}{2}\bigr]$; one checks $f_{\mathrm{JS}}(1) = \tfrac12[0-2\log1] = 0$ and that $D_{f_{\mathrm{JS}}} = \mathrm{JS}$ by direct substitution:
$$\int q\,f_{\mathrm{JS}}(p/q) = \int\left[\tfrac12 p\log\frac{2p}{p+q}+\tfrac12 q\log\frac{2q}{p+q}\right]\mathrm{d}\mu = \tfrac12 D_{\mathrm{KL}}(P\Vert M)+\tfrac12 D_{\mathrm{KL}}(Q\Vert M),\qquad M := \tfrac{P+Q}{2}.$$
The original GAN value function is $V(D) = \mathbb{E}_P[\log D]+\mathbb{E}_Q[\log(1-D)]$. Pointwise maximisation of $p\log d+q\log(1-d)$ over $d\in(0,1)$ gives $d^\star = \frac{p}{p+q}$, i.e. $D^\star = p/(p+q)$ — the discriminator estimates the posterior probability that a sample is real, which is why a well-trained discriminator's output is interpretable as a probability rather than as a density ratio. Substituting,
$$\sup_D V(D) = \mathbb{E}_P\!\left[\log\frac{2p}{p+q}\right]+\mathbb{E}_Q\!\left[\log\frac{2q}{p+q}\right]-2\log2 = 2\,\mathrm{JS}(P,Q)-2\log 2,$$
so the identity is
$$\mathrm{JS}(P,Q) = \tfrac12\Bigl[\sup_D V(D)+\log 4\Bigr].$$
Sanity checks: $P=Q$ gives $D^\star\equiv\tfrac12$, $V = -\log4$, JS $=0$; mutually singular $P,Q$ give $V = 0$ and JS $=\log2$, matching Proposition 11.43.

*(c)* Fix a critic $T$ and let $\hat I := \frac1b\sum_{i=1}^b T(X_i)-\log\bigl(\frac1b\sum_{j=1}^b e^{T(Y_j)}\bigr)$ with $X_i\sim P$, $Y_j\sim Q$ independent. The first average is unbiased for $\mathbb{E}_P[T]$. For the second, write $\hat S := \frac1b\sum_j e^{T(Y_j)}$, so $\mathbb{E}[\hat S] = S := \mathbb{E}_Q[e^T]$. Since $u\mapsto-\log u$ is strictly convex, Jensen gives
$$\mathbb{E}\bigl[-\log\hat S\bigr]\ >\ -\log\mathbb{E}[\hat S] = -\log S$$
(strictly, unless $\hat S$ is degenerate). Hence $\mathbb{E}[\hat I] > \mathbb{E}_P[T]-\log S = $ the population DV objective at $T$, so the plug-in **over**-estimates the DV objective, and if $T$ is optimal it over-estimates $D_{\mathrm{KL}}$ itself. The bias is positive for every fixed critic, however good or bad.

Size, to leading order: expand $-\log\hat S$ around $S$. With $\hat S = S(1+\epsilon)$, $\epsilon := (\hat S-S)/S$ of mean $0$ and variance $\mathrm{Var}_Q(e^T)/(bS^2)$,
$$\mathbb{E}[-\log\hat S] = -\log S + \mathbb{E}\bigl[-\log(1+\epsilon)\bigr] = -\log S+\tfrac12\mathbb{E}[\epsilon^2]+O(\mathbb{E}|\epsilon|^3) = -\log S+\frac{\mathrm{Var}_Q(e^T)}{2bS^2}+\cdots.$$
So the bias is $\mathrm{Var}_Q(e^T)/(2bS^2)$ to leading order — the $O(\mathrm{Var}_Q(e^T)/b)$ claimed.

Now the crux. At the optimum $T^\star = \log\frac{\mathrm{d}P}{\mathrm{d}Q}$ (choosing the constant so $S=1$), $e^{T^\star} = \mathrm{d}P/\mathrm{d}Q$ and
$$\mathrm{Var}_Q\bigl(e^{T^\star}\bigr) = \mathbb{E}_Q\left[\left(\frac{\mathrm{d}P}{\mathrm{d}Q}\right)^{2}\right]-1 = \chi^2(P\Vert Q).$$
The bias is therefore $\chi^2(P\Vert Q)/(2b)$, and the $\chi^2$ divergence grows *exponentially* in the KL divergence for typical families. For $P = N(\Delta,1)$, $Q = N(0,1)$: $D_{\mathrm{KL}} = \Delta^2/2$ and $\chi^2 = e^{\Delta^2}-1 = e^{2D_{\mathrm{KL}}}-1$. Requiring the bias to be a small fraction of $D_{\mathrm{KL}}$ therefore requires
$$b \ \gtrsim\ \frac{e^{2D_{\mathrm{KL}}}}{2D_{\mathrm{KL}}},$$
i.e. batch size exponential in the value being estimated. At $D_{\mathrm{KL}} = 4$ nats that is already $\sim400$; at $10$ nats, $\sim2\times10^{7}$; at $18$ nats, $\sim10^{14}$. Exercise B2(d) measured this and found the leading-order formula accurate while $\chi^2/b\ll1$ and pessimistic beyond, but the qualitative conclusion held: the observed bias at batch $64$ rose from $0.038$ at $D_{\mathrm{KL}}=1$ to $7.658$ at $D_{\mathrm{KL}} = 18$. Since the derivation used the *optimal* critic, no architecture, learning rate, or regulariser can repair it. The degradation is worst exactly in the high-information regime that representation-learning papers report.

*(d)* The second failure mode is the opposite sign and a different mechanism. If $T$ is trained by maximising the *empirical* DV objective on the same batch used to report the value, then what is reported is $\sup_{T\in\mathcal{T}}\hat{\mathcal{L}}_b(T)$, a supremum of a random function. Even if every individual $\mathbb{E}[\hat{\mathcal{L}}_b(T)]$ were equal to the population value, the supremum of the empirical objective exceeds the supremum of its expectation, and with a flexible critic class the excess can be large: a critic with enough capacity can separate the $b$ samples of $P$ from the $b$ samples of $Q$ almost perfectly, producing a reported value limited only by capacity and optimisation. Worked Example 11.15 measured both effects in the same table: at a true $6$ nats, MINE reported $6.168$ on its training batches (overfitting, upward) and $3.127$ held out (bias of the trained critic, downward).

Both failure modes are instances of the general limitation quoted in §11.11: any distribution-free high-confidence *lower* bound on mutual information computable from $n$ samples is $O(\log n)$. Overfitting explains how a reported number can exceed the truth; the exponential sample requirement explains why an honest, held-out estimate cannot reach it.

The protocol I would insist on before believing a reported MI value: (i) evaluate the critic on data held out from critic training, and report both the training and held-out values, since their gap is the diagnostic; (ii) report the batch size and the resulting $\log K$ or $\log n$ ceiling alongside the number, so the reader can see whether the bound is saturated; (iii) sweep the batch size and show that the estimate has plateaued rather than tracking $\log b$; (iv) calibrate on a synthetic problem of the same dimension and sample size with a known closed form, and report the signed error there; (v) state the units. Absent (i) and (iv), a reported mutual information carries no information.

**C3 (Pinsker, Fano, and pinning a rate).**

*(a)* **Bernoulli case.** Write $\mathrm{kl}(a\Vert b) := a\log\frac ab+(1-a)\log\frac{1-a}{1-b}$. Fix $b\in(0,1)$ and set $\varphi(a) := \mathrm{kl}(a\Vert b)-2(a-b)^2$ on $(0,1)$. Then $\varphi(b) = 0$,
$$\varphi'(a) = \log\frac{a}{b}-\log\frac{1-a}{1-b}-4(a-b),\qquad \varphi'(b) = 0,$$
$$\varphi''(a) = \frac1a+\frac1{1-a}-4 = \frac{1}{a(1-a)}-4\ \ge\ 0,$$
the last step because $a(1-a)\le\tfrac14$ on $[0,1]$. So $\varphi$ is convex with a stationary point at $a=b$, hence $\varphi\ge0$: $2(a-b)^2\le\mathrm{kl}(a\Vert b)$. Since $\mathrm{TV}(\mathrm{Ber}(a),\mathrm{Ber}(b)) = |a-b|$, this is exactly Pinsker for Bernoullis. Boundary cases $b\in\{0,1\}$ make the right side $+\infty$ unless $a=b$, so they hold trivially.

**Reduction.** Let $A^\star := \{p>q\}$ and let $\Phi(x) := \mathbf{1}_{A^\star}(x)$, a deterministic map, hence a Markov kernel. Pushing forward, $P\circ\Phi^{-1} = \mathrm{Ber}(P(A^\star))$ and $Q\circ\Phi^{-1} = \mathrm{Ber}(Q(A^\star))$. Apply the DPI for KL (Theorem 11.32 with $f(t)=t\log t$, or part (b) of C1):
$$\mathrm{kl}\bigl(P(A^\star)\,\Vert\,Q(A^\star)\bigr)\ \le\ D_{\mathrm{KL}}(P\Vert Q).$$
By Proposition 11.36 the supremum defining TV is attained at $A^\star$, so
$$\mathrm{TV}(P,Q)^2 = \bigl(P(A^\star)-Q(A^\star)\bigr)^2 \ \le\ \tfrac12\,\mathrm{kl}\bigl(P(A^\star)\Vert Q(A^\star)\bigr)\ \le\ \tfrac12 D_{\mathrm{KL}}(P\Vert Q).$$
Taking square roots finishes it. Two remarks on why the reduction is legitimate. First, DPI is what allows us to replace an arbitrary pair of measures on an arbitrary space by a pair of Bernoullis: information can only be lost by the map $\Phi$, so an inequality proved downstream transfers upstream in the safe direction. Second, we need TV to be *attained* at $A^\star$, not merely approached, because we are equating the left-hand side with a quantity computed from the two-point distribution; Proposition 11.36 supplies exactly this. If we had used an arbitrary set $A$ we would have obtained the weaker $|P(A)-Q(A)|\le\sqrt{D_{\mathrm{KL}}/2}$, true but not Pinsker.

*(b)* Take $\theta_0 = 0$, $\theta_1 = \delta>0$, so $|\theta_0-\theta_1| = \delta = 2s$ with $s = \delta/2$. The data are $n$ i.i.d. observations, so by additivity of KL over independent coordinates,
$$D_{\mathrm{KL}}\bigl(P_0^{\otimes n}\Vert P_1^{\otimes n}\bigr) = n\,D_{\mathrm{KL}}\bigl(N(0,\sigma^2)\Vert N(\delta,\sigma^2)\bigr) = \frac{n\delta^2}{2\sigma^2}.$$
Proposition 11.54 with absolute-error loss gives
$$\mathfrak{M}_n \ \ge\ \frac{s}{2}\bigl(1-\mathrm{TV}\bigr) = \frac{\delta}{4}\bigl(1-\mathrm{TV}(P_0^{\otimes n},P_1^{\otimes n})\bigr).$$
Now substitute $\delta = c\sigma/\sqrt n$ and optimise over $c>0$. With the Pinsker bound $\mathrm{TV}\le\delta\sqrt n/(2\sigma) = c/2$:
$$\mathfrak{M}_n\ \ge\ \frac{c\sigma}{4\sqrt n}\Bigl(1-\frac c2\Bigr),$$
maximised at $c=1$, giving $\mathfrak{M}_n\ge\frac{\sigma}{8\sqrt n}$, i.e. $C = 1/8 = 0.125$.

If instead we use the exact total variation for Gaussian shift families, $\mathrm{TV}(P_0^{\otimes n},P_1^{\otimes n}) = 2\Phi\bigl(\delta\sqrt n/(2\sigma)\bigr)-1 = 2\Phi(c/2)-1$, the bound becomes $\frac{c\sigma}{4\sqrt n}\bigl(2-2\Phi(c/2)\bigr) = \frac{c\sigma}{2\sqrt n}\bigl(1-\Phi(c/2)\bigr)$, and numerical maximisation gives $c^\star = 1.503583$ with
$$\mathfrak{M}_n\ \ge\ \frac{0.169971\,\sigma}{\sqrt n}.$$
So the exact-TV version is better by a factor of about $1.36$, and the optimal separation is roughly $1.5\sigma/\sqrt n$ rather than $\sigma/\sqrt n$. The instructive part is the shape of the trade-off: pushing $\delta$ up increases the multiplier $s$ but makes the hypotheses easier to distinguish, driving $1-\mathrm{TV}$ towards $0$; pushing $\delta$ down does the reverse. The optimum sits where the statistical resolution scale is, which is $\sigma/\sqrt n$ — the same scale the Fisher information of Chapter 8 identifies, and not a coincidence.

*(c)* Upper bound. For the sample mean, $\bar X_n-\theta\sim N(0,\sigma^2/n)$ for *every* $\theta$, and $\mathbb{E}|N(0,\tau^2)| = \tau\sqrt{2/\pi}$. Hence
$$\sup_\theta\mathbb{E}_\theta\bigl|\bar X_n-\theta\bigr| = \sqrt{\frac{2}{\pi}}\cdot\frac{\sigma}{\sqrt n} = \frac{0.797885\,\sigma}{\sqrt n},$$
so $\mathfrak{M}_n\le 0.797885\,\sigma/\sqrt n$. The two bounds bracket $\mathfrak{M}_n$ within a factor of $0.797885/0.169971 = 4.69$.

*(d)* To say "$n^{-1/2}$ is the rate" is to say that $\mathfrak{M}_n = \Theta(\sigma n^{-1/2})$: there exist constants $0<C_1\le C_2<\infty$, independent of $n$ and $\sigma$, with $C_1\sigma n^{-1/2}\le\mathfrak{M}_n\le C_2\sigma n^{-1/2}$. Here $C_1 = 0.17$ and $C_2 = 0.80$ from parts (b) and (c). Neither half is dispensable, and for different reasons. An upper bound alone establishes that *your estimator* achieves $n^{-1/2}$; it is silent on whether a cleverer estimator achieves $n^{-1}$, and the history of statistics contains many problems where the obvious estimator was rate-suboptimal, so the possibility is real rather than rhetorical. A lower bound alone establishes that *no estimator* beats $n^{-1/2}$; it is silent on whether any estimator attains it, and there are problems where the minimax rate is strictly worse than the lower bound obtained by a given construction. A rate claim is a statement about a *gap being closed*, and closing it requires an argument from each direction — which is precisely why concentration inequalities (Chapter 4, and Chapter 12 in earnest) and information-theoretic lower bounds (this chapter) are studied together rather than as separate subjects.

For the $d$-dimensional squared-error result $\mathfrak{M}_n\asymp d\sigma^2/n$, the two-point argument is no longer enough: it can only produce a bound of order $\sigma^2/n$, missing the factor $d$, because two points probe one direction. One needs $M$ hypotheses filling the space, and specifically a set of $\theta_j$ that are pairwise separated by $\asymp\sqrt{d}\cdot\delta$ in Euclidean norm while remaining within $O(\delta)$ of each other in the per-coordinate scale, so that the pairwise KL stays $O(n\delta^2/\sigma^2)$ while $\log M \asymp d$. Corollary 11.53 then gives the extra factor $d$. The construction is a packing of the hypercube $\{0,\delta\}^d$ by binary codewords with pairwise Hamming distance $\ge d/4$, and the ingredient from §11.11 that guarantees exponentially many such codewords exist is the **Varshamov–Gilbert bound**. That single combinatorial lemma is what converts a one-dimensional argument into a dimension-dependent rate, and it is the reason Fano's method rather than Le Cam's is the default tool in high-dimensional statistics.

### References and further reading

#### Textbooks and monographs

**Cover, T. M. and Thomas, J. A., *Elements of Information Theory*, 2nd ed., Wiley, 2006.** The standard reference for §§11.1, 11.4, 11.8, 11.9. Chapters 2–5 cover entropy, chain rules, source coding, and Huffman optimality (§5.8); Chapter 2 has the Fano treatment this chapter follows. Its perspective is engineering rather than measure-theoretic, so read our §11.2 alongside its Chapter 8.

**Polyanskiy, Y. and Wu, Y., *Information Theory: From Coding to Learning*, Cambridge University Press, 2024.** The modern successor and the book to read if you liked this chapter's emphasis: measure-theoretic throughout, divergences defined via Radon–Nikodym derivatives, $f$-divergences and their variational representations developed properly, and serious space for Fano's and Le Cam's methods.

**Csiszár, I. and Shields, P., *Information Theory and Statistics: A Tutorial*, Foundations and Trends in Communications and Information Theory, 2004.** The canonical source for the geometry behind Exercise A13: I-projections, the Pythagorean identity, maximum entropy as projection onto a linear family. Csiszár's papers are where $f$-divergences originate.

**Tsybakov, A. B., *Introduction to Nonparametric Estimation*, Springer, 2009.** Chapter 2 is the definitive treatment of minimax lower bounds: Le Cam, Assouad, Fano, and the Varshamov–Gilbert bound (Lemma 2.9) quoted in §11.11.

**Boucheron, S., Lugosi, G. and Massart, P., *Concentration Inequalities: A Nonasymptotic Theory of Independence*, Oxford University Press, 2013.** The other half of the story: the upper bounds that pin rates from above, with an entropy method linking information quantities to concentration. The natural companion to Chapter 12.

**Villani, C., *Optimal Transport: Old and New*, Springer, 2009.** The comprehensive reference for §11.7: Kantorovich duality (Ch. 5), metric properties of $W_p$ via the gluing lemma (Ch. 6). The earlier *Topics in Optimal Transportation* (AMS, 2003) is gentler and has the one-dimensional theory, including monotone-coupling optimality for convex costs (Thm 2.18).

**Santambrogio, F., *Optimal Transport for Applied Mathematicians*, Birkhäuser, 2015.** More approachable for a first pass, with a good treatment of the one-dimensional case (§2.2) and the Lipschitz dual.

**Peyré, G. and Cuturi, M., *Computational Optimal Transport*, Foundations and Trends in Machine Learning, 2019.** The reference for §11.7.4: entropic regularisation, Sinkhorn, log-domain implementations, debiasing, sliced Wasserstein. Practitioner-facing and freely available.

**Grünwald, P., *The Minimum Description Length Principle*, MIT Press, 2007.** The full development of the MDL paragraph in §11.8.3, including the link to BIC and the two-part-code difficulties that make MDL arguments about neural networks suggestive rather than decisive.

**Kallenberg, O., *Foundations of Modern Probability*, 3rd ed., Springer, 2021.** For the prerequisites quoted in §11.11, particularly regular conditional distributions on Polish spaces.

#### Machine learning papers

**Goodfellow, I. et al., "Generative Adversarial Networks", NeurIPS 2014.** The origin of the objective analysed in Theorem 11.34 and Exercise C2(b), with $D^\star = p/(p+q)$ and the identification of the optimum with JS.

**Nowozin, S., Cseke, B. and Tomioka, R., "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization", NeurIPS 2016.** The GAN objective generalised to arbitrary $f$-divergences via Fenchel duality, as in Theorem 11.34, with a table of generators and conjugates.

**Arjovsky, M., Chintala, S. and Bottou, L., "Wasserstein GAN", ICML 2017.** The paper that made §11.7 part of the machine learning curriculum; its Section 2 contains the parallel-segments example of Proposition 11.44. Read with Gulrajani et al., "Improved Training of Wasserstein GANs" (NeurIPS 2017), which replaces weight clipping by a gradient penalty and is honest about why clipping does not implement the Lipschitz constraint.

**Belghazi, M. I. et al., "MINE: Mutual Information Neural Estimation", ICML 2018.** The Donsker–Varadhan estimator as a trainable objective. Read the next two entries before using it.

**Poole, B. et al., "On Variational Bounds of Mutual Information", ICML 2019.** A systematic account of the variational MI bounds — DV, NWJ, InfoNCE, interpolations — and their bias–variance trade-offs. It organises §11.5.3 into a coherent picture.

**McAllester, D. and Stratos, K., "Formal Limitations on the Measurement of Mutual Information", AISTATS 2020.** The $O(\log n)$ result quoted in §11.11 and used in §11.10.3. Short, and it changes how you read reported MI values.

**van den Oord, A., Li, Y. and Vinyals, O., "Representation Learning with Contrastive Predictive Coding", arXiv 2018.** The InfoNCE objective as an MI lower bound; the $\log K$ ceiling of §11.10.3 is immediate from its loss.

**Tschannen, M. et al., "On Mutual Information Maximization in Representation Learning", ICLR 2020.** Evidence that InfoNCE's success is not explained by mutual information: encoders with higher estimated MI are sometimes worse downstream. The basis for §11.10.3's conclusion.

**Tishby, N. and Zaslavsky, N., "Deep Learning and the Information Bottleneck Principle", ITW 2015**, and **Shwartz-Ziv, R. and Tishby, N., "Opening the Black Box of Deep Neural Networks via Information", arXiv 2017.** The IB claim about deep learning and the information-plane experiments discussed in Common Confusion 11.9. The original framework is Tishby, Pereira and Bialek (1999).

**Saxe, A. M. et al., "On the Information Bottleneck Theory of Deep Learning", ICLR 2018.** The critique: the compression phase depends on the nonlinearity, is largely absent with ReLU units, and is neither necessary nor sufficient for generalisation. The pair is a case study in evaluating a claim built on an estimated quantity.

**Higgins, I. et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017.** The objective analysed in Proposition 11.55 and Figure 11.7.

**Alemi, A. et al., "Fixing a Broken ELBO", ICML 2018.** The rate–distortion reading of the ELBO, the $(R,D)$ plane, and the ELBO's blindness along $D+R=c$ — the source of §11.10.1's framing. Its companion, "Deep Variational Information Bottleneck" (ICLR 2017), is the stochastic-encoder IB that survives Common Confusion 11.9's objections.

**Theis, L., van den Oord, A. and Bethge, M., "A Note on the Evaluation of Generative Models", ICLR 2016.** Required reading before reporting or comparing a likelihood: the dequantisation and scale issues of §11.8.3 and Common Confusion 11.8, plus a demonstration that log-likelihood, sample quality, and downstream utility vary independently.

**Kraskov, A., Stögbauer, H. and Grassberger, P., "Estimating Mutual Information", Physical Review E 69, 2004.** The KSG estimator of Worked Example 11.15 and Exercise B2. For rates and high-dimensional limits see Gao, Oh and Viswanath (IEEE Trans. Inf. Theory, 2018).

**Fournier, N. and Guillin, A., "On the Rate of Convergence in Wasserstein Distance of the Empirical Measure", Probability Theory and Related Fields, 2015**, and **Weed, J. and Bach, F., "Sharp Asymptotic and Finite-Sample Rates of Convergence of Empirical Measures in Wasserstein Distance", Bernoulli, 2019.** The $n^{-1/d}$ rates quoted in Common Confusion 11.7, and refinements showing the exponent improves when the measure sits on a lower-dimensional set — which is the generative-modelling situation.

**Müller, R., Kornblith, S. and Hinton, G., "When Does Label Smoothing Help?", NeurIPS 2019.** The empirical companion to Proposition 11.22, including effects on calibration and distillation.
