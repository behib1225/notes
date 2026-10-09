# Chapter 12 — High-Dimensional Probability

## Chapter 12 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 12.1 Four ways intuition fails | Volume flees to the boundary; the unit ball's volume collapses; random directions are nearly orthogonal; $\|Z\|$ pins to $\sqrt d$ | Explains why the mode of a Gaussian prior is not where samples live, why latent-space interpolation should follow arcs not chords, and why "sample near the mean" is bad advice |
| 12.2 Concentration of measure | Any Lipschitz function of a Gaussian vector has fluctuations $O(L)$, *independent of $d$* | The foundation of every generalization bound with teeth; certified adversarial robustness via randomized smoothing; why minibatch gradient norms are so stable |
| 12.3 Distance concentration | For isotropic data, $D_{\max}/D_{\min}\to 1$; but real data has low intrinsic dimension | Why $k$-NN, kernel methods, and vector retrieval work at all — and exactly when they stop working |
| 12.4 Johnson–Lindenstrauss | Random projection to $k=O(\varepsilon^{-2}\log n)$ preserves all pairwise distances | Random features, LSH, sketching, product quantization, embedding compression in vector databases |
| 12.5 Random matrices | Marchenko–Pastur, Bai–Yin, semicircle, $\|A\|\lesssim\sqrt n+\sqrt d$ via $\varepsilon$-nets | Initialization scaling, spectral norm regularization, why square random matrices are nearly singular |
| 12.6 Consequences for deep learning | The whole spectrum at init, not just the variance; dynamical isometry; Hessian structure | Orthogonal vs. Gaussian init; what "most critical points are saddles" does and does not license |
| 12.7 Covariance in high dimensions | $\hat\Sigma$ is spectrally wrong when $d\sim n$, even though it is unbiased; a true signal is visible to PCA only above the BBP threshold $1+\sqrt\gamma$ | Whitening, GDA/LDA, natural gradient, K-FAC, Shampoo — all silently depend on this; when PCA and linear probes can (and provably cannot) find a direction |
| 12.8 The double-edged sword | Concentration makes averages reliable *and* distances useless | A single frame for the whole chapter |

## 12.0 Orientation: your geometric intuition was trained on $d\le 3$

Every picture you have ever drawn of a probability distribution lives in one, two, or three dimensions. A bell curve. A contour plot with elliptical level sets. A blob of points with a denser middle. These pictures are not merely imprecise in high dimension — they are *actively misleading*, in the specific sense that they will cause you to predict the wrong sign of an effect.

Here is the punchline, stated in advance. In high dimension, probability mass separates from probability density. The place where the density is largest can have negligible probability, because there is almost no volume there. Almost all the volume of any reasonable body sits in a thin shell at its boundary, and almost all the mass of a standard Gaussian sits in a thin annulus at radius $\sqrt d$. Two consequences pull in opposite directions, and holding both at once is the whole skill this chapter teaches. On the one hand, *smooth functions of many independent variables barely fluctuate* — this is concentration of measure, and it is the reason machine learning is possible at all: empirical averages are reliable, so minimizing training loss says something about test loss. On the other hand, *distance stops discriminating* — for genuinely isotropic data the nearest and farthest neighbours of a query point are almost the same distance away, so nearest-neighbour search degenerates into a coin flip. Both facts come from the same theorem. That is the double-edged sword of the title of §12.8.

The resolution of the apparent paradox — how can retrieval, $k$-NN, kernel regression, and embedding search all work in production at $d = 1536$? — is the single most important idea in the chapter, and it is developed in §12.3. Real data is not isotropic. A $224\times 224\times 3$ image lives in $\mathbb{R}^{150528}$, but the set of images that look like photographs is a vanishingly thin subset of that space. Distances degrade at a rate governed by *intrinsic* dimension, not ambient dimension, and we will measure this contrast numerically: isotropic Gaussian data in $\mathbb{R}^{10^4}$ has essentially no distance contrast, while data on a $5$-dimensional manifold embedded in the *same* $\mathbb{R}^{10^4}$ has contrast that does not degrade at all.

What you are assumed to bring. From Chapter 4: sub-Gaussian random variables and their moment generating function characterization, the Chernoff method, Hoeffding's inequality, Bernstein's inequality, covering and packing numbers of subsets of $\mathbb{R}^d$, and Rademacher complexity. From Chapter 5: the multivariate Gaussian $\mathcal{N}(\mu,\Sigma)$, its rotational invariance when $\Sigma = \sigma^2 I$, covariance spectra, and whitening. From Chapter 7: the central limit theorem, and its honest finite-$n$ limitations — that $\sqrt n$-scaling describes the centre of the distribution and says nothing reliable about the tails. From Chapter 8: the James–Stein estimator, our first encounter with the fact that high dimension changes what "reasonable estimator" means. From Chapter 11: mutual information and total variation distance. From Chapter 1, cited by number: Definition 1.17 (support: the smallest closed set of full measure) and Proposition 1.8(2) (the union bound). We will not re-derive any of these.

What is genuinely new here. Three things. First, *dimension-free* concentration: Theorem 12.7 says that for $L$-Lipschitz $f$ and $Z\sim\mathcal{N}(0,I_d)$, the fluctuations of $f(Z)$ are controlled by $L$ alone, with no $d$ anywhere. When you first meet this it should feel wrong. Second, the *spectral* view of randomness: a random matrix is not a random collection of numbers, it is an object with a deterministic limiting singular value distribution (Marchenko–Pastur), and this determines whether signal propagates through a deep network. Third, the systematic *use* of concentration as a design tool: Johnson–Lindenstrauss is not a curiosity, it is the theoretical licence behind random features, locality-sensitive hashing, and every sketching algorithm in a modern retrieval stack.

Two notational points, since this chapter adds to the book's canonical list. First, $\lambda_d$ continues to mean $d$-dimensional Lebesgue measure (as in Proposition 12.1), but from §12.5 onward $\lambda_i(M)$ with a matrix argument means the $i$-th eigenvalue of $M$, and $s_i(M)$ means the $i$-th singular value; the argument always disambiguates, and I flag it here rather than let you trip over it. Second, $\gamma:=d/n$ is reserved throughout §12.5–12.7 for the aspect ratio of a data matrix, and $\gamma_d$ for the standard Gaussian measure on $\mathbb{R}^d$ in §12.2.3 only.

A warning about honesty. Section 12.6 discusses claims about deep learning loss landscapes that circulate in the literature with much more confidence than the mathematics supports. I will mark, explicitly and repeatedly, which statements are theorems, which are theorems about models nobody trains, which are empirical observations on specific architectures, and which are conjecture. This distinction is routinely blurred, and learning to see the seam is part of learning to read papers.

Every numerical value in this chapter was computed by running code. Seeds and sample sizes are stated. Where a bound is loose I say by how much, because knowing that a bound is conservative by a factor of five is part of knowing the bound.

How to read this chapter if you are working alone. Section 12.1 is meant to be read straight through in one sitting; the surprise is the point, and skipping to the explanations spoils it. Sections 12.2, 12.4 and 12.5 contain the proofs and can be read in any order after 12.1. Sections 12.6 and 12.7 are applications and are safe to skim on a first pass, then return to. If you only have time for one thing, do Lab B5 — the isotropic-versus-manifold contrast — because it is the result that changes how you read every paper that uses a distance.

## 12.1 Four ways intuition fails

We begin with facts, not theory. Each of the four subsections below states something you can verify by arithmetic, and each contradicts a picture you probably hold. The explanations come in §12.2. Resist the urge to skip ahead; the surprise is pedagogically load-bearing.

### 12.1.1 Volume flees to the boundary

Let $B_d(r) = \{x\in\mathbb{R}^d : \|x\|_2 \le r\}$ be the Euclidean ball of radius $r$ and let $V_d(r) = \lambda_d(B_d(r))$ be its Lebesgue volume.

**Proposition 12.1.** For every $d\ge 1$ and $r>0$,
$$V_d(r) = \frac{\pi^{d/2}}{\Gamma\!\big(\tfrac d2+1\big)}\, r^d .$$
Consequently:

1. *(Shell)* For $\varepsilon\in(0,1)$, the fraction of the volume of $B_d(1)$ lying outside $B_d(1-\varepsilon)$ is exactly $1-(1-\varepsilon)^d$, which tends to $1$ as $d\to\infty$ for each fixed $\varepsilon$.
2. *(Half-volume shell)* Half the volume of $B_d(1)$ lies within distance $\varepsilon_d := 1-2^{-1/d}$ of the surface, and $\varepsilon_d \sim (\log 2)/d \to 0$.
3. *(Collapse)* $V_d(1)\to 0$ as $d\to\infty$; in fact $\log V_d(1) = -\tfrac d2\log d + O(d)$.
4. *(Ball in cube)* The ball inscribed in the cube $[0,1]^d$ has volume $V_d(1/2) = V_d(1)2^{-d}\to 0$, so the inscribed ball occupies a vanishing fraction of the cube.

*Proof.* For the volume formula, compute the integral $I:=\int_{\mathbb{R}^d}e^{-\|x\|^2}dx$ two ways.

*Cartesian.* The integrand factorizes, $e^{-\|x\|^2}=\prod_{i=1}^de^{-x_i^2}$, so by Tonelli $I=\big(\int_{\mathbb{R}}e^{-t^2}dt\big)^d=\pi^{d/2}$.

*Polar.* By homogeneity $V_d(r)=V_d(1)r^d$, so the measure of the sphere of radius $r$ in the polar decomposition is $dV_d(r)=V_d(1)\,d\,r^{d-1}dr$ (this is Chapter 3's change-of-variables formula with the polar Jacobian $r^{d-1}$). Since $e^{-\|x\|^2}$ depends on $x$ only through $r=\|x\|$,
$$I=\int_0^\infty e^{-r^2}\,dV_d(r)=V_d(1)\,d\int_0^\infty e^{-r^2}r^{d-1}\,dr .$$
Substituting $u=r^2$ ($du=2r\,dr$) turns the integral into $\tfrac12\int_0^\infty e^{-u}u^{d/2-1}du=\tfrac12\Gamma(d/2)$, so
$$I=V_d(1)\cdot d\cdot\tfrac12\Gamma(d/2)=V_d(1)\cdot\tfrac d2\Gamma(d/2)=V_d(1)\,\Gamma\big(\tfrac d2+1\big),$$
the last step by the functional equation $z\Gamma(z)=\Gamma(z+1)$ with $z=d/2$. Equating the two evaluations gives $V_d(1)=\pi^{d/2}/\Gamma(d/2+1)$, and the general radius follows by homogeneity.

For (1): $V_d$ scales as the $d$-th power of the radius, so $V_d(1-\varepsilon)/V_d(1) = (1-\varepsilon)^d$, and $(1-\varepsilon)^d = e^{d\log(1-\varepsilon)}\to 0$ since $\log(1-\varepsilon)<0$. For (2): solve $(1-\varepsilon)^d = 1/2$, giving $\varepsilon = 1-2^{-1/d}$, and expand $2^{-1/d} = e^{-(\log 2)/d} = 1-(\log 2)/d + O(d^{-2})$. For (3): by Stirling's formula for the Gamma function (Results used but not proved), $\log\Gamma(d/2+1) = \tfrac d2\log\tfrac d2-\tfrac d2+O(\log d)$, so $\log V_d(1) = \tfrac d2\log\pi - \tfrac d2\log\tfrac d2+\tfrac d2+O(\log d) = -\tfrac d2\log d + O(d)$, which $\to-\infty$. For (4): immediate from the scaling. $\square$

The metaphor is an orange, and it is attached to a computation rather than floating free. In $d=3$, if you peel an orange of radius $1$ removing a rind of thickness $0.1$, you discard $1-0.9^3 = 27.1\%$ of the fruit. Same orange, same peeler, in $d=100$: you discard $1-0.9^{100} = 99.997\%$. The fruit *is* the peel.

> **Worked Example 12.1 (the orange peel, quantitatively).**
> Fix $\varepsilon = 0.1$ and compute $1-(1-\varepsilon)^d$:
>
> | $d$ | $1-0.9^d$ | thickness $\varepsilon_d$ holding half the volume |
> |---|---|---|
> | 2 | 0.190000 | 0.292893 |
> | 10 | 0.651322 | 0.066967 |
> | 100 | 0.999973 | 0.006908 |
> | 1000 | 1.000000 | 0.000693 |
> | 10000 | 1.000000 | 0.000069 |
>
> Read the last column as follows: in $\mathbb{R}^{10000}$, a shell of thickness $6.9\times 10^{-5}$ — seven parts in a hundred thousand of the radius — contains half of the ball's volume. Any algorithm whose behaviour depends on "being in the interior" of a high-dimensional body is operating on a set of measure zero for practical purposes.

> **Worked Example 12.2 (the unit ball's volume rises, then collapses).**
> Compute $V_d(1) = \pi^{d/2}/\Gamma(d/2+1)$ exactly:
>
> | $d$ | $V_d(1)$ | $V_d(1/2)$ = inscribed ball / cube $[0,1]^d$ |
> |---|---|---|
> | 1 | 2 | 1 |
> | 2 | 3.141593 | 0.785398 |
> | 3 | 4.188790 | 0.523599 |
> | 5 | 5.263789 | 0.164493 |
> | 10 | 2.550164 | $2.4904\times10^{-3}$ |
> | 20 | $2.5807\times10^{-2}$ | $2.4611\times10^{-8}$ |
> | 100 | $2.3682\times10^{-40}$ | $1.8682\times10^{-70}$ |
>
> The volume of the unit ball is *maximized at $d=5$* (value $5.2638$; compare $d=4$: $4.9348$, $d=6$: $5.1677$) and then falls off a cliff. This is worth pausing on, because "the unit ball has small volume" sounds like a units error — surely we can rescale? We can, but not usefully: to keep the volume at $1$ you need radius $r_d = V_d(1)^{-1/d}$, which grows like $\sqrt{d/(2\pi e)}$. A unit-volume ball in $\mathbb{R}^{100}$ has radius $\approx 2.5$, and $99.99\%$ of its volume sits in the outer $0.7\%$ of that radius.
>
> The last column is the fact that kills grid search and axis-aligned partitioning. A $k$-d tree splits space into boxes; a nearest-neighbour query needs a *ball*. In $\mathbb{R}^{100}$ the ball inscribed in the cell occupies $1.9\times10^{-70}$ of the cell's volume, so "check the cell containing the query, then a few neighbours" is not an approximation to a ball query — it is unrelated to one.

> **Worked Example 12.3 (the cube is a spiky object).**
> The cube $[0,1]^d$ has inradius $1/2$ (the distance from its centre to the nearest face) and circumradius $\sqrt d/2$ (the distance from its centre to a corner). For $d=2$ these are $0.5$ and $0.7071$; for $d=10000$ they are $0.5$ and $50.0$. The corner-to-face ratio is $\sqrt d$.
>
> So a high-dimensional cube is not box-shaped in any intuitive sense; it is a small round core with $2^d$ enormous spikes. Where is its volume? Not in the core: the fraction of the cube inside the inscribed ball is $1.87\times 10^{-70}$ at $d=100$. And not in any *particular* spike: the fraction with all coordinates in $[0.1,0.9]$ is $0.8^d$, which is $0.107$ at $d=10$ and $2\times 10^{-10}$ at $d=100$. The volume is spread across the $2^d$ spikes, in the sense that a uniform sample has $\|x-c\|\approx\sqrt{d/12}$ — much closer to the corners' distance $\sqrt d/2$ than to the faces' distance $1/2$.
>
> This is a concrete reason why uniform initialization on a box and Gaussian initialization behave similarly in deep networks: in high dimension a uniform sample from a cube is, up to a constant, at the same radius as a Gaussian sample with matched variance, and the direction is nearly uniform in both cases.

### 12.1.2 Random directions are nearly orthogonal

**Proposition 12.2.** Let $U, V$ be independent and uniform on the unit sphere $S^{d-1}\subset\mathbb{R}^d$, and let $T = \langle U,V\rangle = \cos\Theta$ where $\Theta\in[0,\pi]$ is the angle between them. Then for $d\ge 2$:

1. $T$ has density $c_d(1-t^2)^{(d-3)/2}$ on $[-1,1]$, with $c_d = \Gamma(d/2)/\big(\sqrt\pi\,\Gamma((d-1)/2)\big)$;
2. $T^2\sim\mathrm{Beta}\big(\tfrac12,\tfrac{d-1}2\big)$;
3. $\mathbb{E}[T]=0$, $\operatorname{Var}(T)=1/d$, and $\mathbb{E}|T| \sim \sqrt{2/(\pi d)}$;
4. $\Pr(|T|\ge s)\le 3e^{-(d-1)s^2/8}$ for all $s\in(0,1)$.

*Proof.* Fix $V$; by rotational invariance of the uniform measure on the sphere (which follows from the rotational invariance of $\mathcal{N}(0,I_d)$, Ch. 5, together with the representation $U = Z/\|Z\|$ for $Z\sim\mathcal{N}(0,I_d)$), we may take $V=e_1$, so $T = U_1$. Write $U = Z/\|Z\|$; then $T^2 = Z_1^2/\|Z\|^2 = Z_1^2/(Z_1^2 + W)$ with $W=\sum_{j\ge2}Z_j^2\sim\chi^2_{d-1}$ independent of $Z_1^2\sim\chi^2_1$. The ratio $\chi^2_a/(\chi^2_a+\chi^2_b)$ with independent numerator components is $\mathrm{Beta}(a/2,b/2)$ — a standard computation with the Gamma density and Chapter 3's change of variables — giving (2) with $a=1$, $b=d-1$. Transforming the Beta density back to $t$ gives (1), and $\mathbb{E} T = 0$ by the symmetry $U\mapsto -U$. For the variance, $\mathbb{E}[T^2] = \mathbb{E}[U_1^2] = \tfrac1d\sum_{j=1}^d\mathbb{E}[U_j^2] = \tfrac1d\mathbb{E}\|U\|^2 = 1/d$. The asymptotics of $\mathbb{E}|T|$ follow from $T\approx Z_1/\sqrt d$ for large $d$ and $\mathbb{E}|Z_1|=\sqrt{2/\pi}$.

For (4) we use the Gaussian representation and a Chernoff argument, which avoids any need to estimate the normalizing constant $c_d$. With $Z_1^2\sim\chi^2_1$ independent of $W:=\sum_{j\ge2}Z_j^2\sim\chi^2_{d-1}$,
$$T^2\ge s^2\iff Z_1^2\ge s^2(Z_1^2+W)\iff (1-s^2)Z_1^2\ge s^2W\iff Z_1^2-aW\ge0,\quad a:=\frac{s^2}{1-s^2}.$$
For any $\theta>0$ with $2\theta<1$, Markov's inequality applied to $e^{\theta(Z_1^2-aW)}$ and independence give
$$\Pr(T^2\ge s^2)\ \le\ \mathbb{E}\big[e^{\theta Z_1^2}\big]\,\mathbb{E}\big[e^{-\theta a W}\big] = (1-2\theta)^{-1/2}(1+2\theta a)^{-(d-1)/2},$$
using the $\chi^2$ moment generating function of Lemma 12.4 at $+\theta$ and at $-\theta a$. Choose $\theta=\tfrac14$, so $2\theta=\tfrac12$, and use $a\ge s^2$:
$$\Pr(T^2\ge s^2)\ \le\ \sqrt2\,\Big(1+\tfrac{s^2}2\Big)^{-(d-1)/2}=\sqrt2\,\exp\Big(-\tfrac{d-1}2\log\big(1+\tfrac{s^2}2\big)\Big).$$
Finally $\log(1+x)\ge x-\tfrac{x^2}2=x\big(1-\tfrac x2\big)$ for $x\ge0$, and with $x=s^2/2\le\tfrac12$ this gives $\log(1+\tfrac{s^2}2)\ge\tfrac{s^2}2\cdot\tfrac34=\tfrac{3s^2}8$. Hence $\Pr(T^2\ge s^2)\le\sqrt2\,e^{-3(d-1)s^2/16}$, and since $\{|T|\ge s\}=\{T^2\ge s^2\}$ we may also weaken $\tfrac3{16}$ to $\tfrac18$ and $\sqrt2$ to $3$ to obtain the stated form. ($d=2$ is covered directly: $T=\cos\Theta$ with $\Theta$ uniform, so $\Pr(|T|\ge s)=1-\tfrac2\pi\arcsin s\le 1\le 3e^{-s^2/8}$.) $\square$

The constant $\tfrac18$ is not sharp — the truth is $\tfrac12$, i.e. $\Pr(|T|\ge s)\le2e^{-ds^2/2}$, which I verified numerically against the exact Beta survival function on a grid of $200$ values of $s$ for $d\in\{2,3,5,10,10^2,10^3,10^4\}$ with no violations. We proved the weaker constant because the proof above is six lines and uses only Lemma 12.4. All *numerical* statements about angles in this chapter come from the exact Beta distribution of part (2), never from part (4).

> **Worked Example 12.4 (how orthogonal is "nearly orthogonal"?).**
> Using part (2) of Proposition 12.2, the probability that the angle between two independent random directions lies within $5^\circ$ of a right angle is $\Pr(T^2 \le \sin^2 5^\circ)$, computed from the Beta CDF. Monte Carlo columns use `numpy` default_rng(seed=12); sample sizes as marked.
>
> | $d$ | $\mathbb{E}\vert T\vert$ | $\sqrt{2/(\pi d)}$ | $\Pr(\Theta\in[85^\circ,95^\circ])$ | Monte Carlo |
> |---|---|---|---|---|
> | 2 | 0.636620 | 0.564190 | 0.055556 | 0.055585 ($n=2\times10^5$) |
> | 10 | 0.258690 | 0.252313 | 0.201129 | 0.201565 ($n=2\times10^5$) |
> | 100 | 0.079988 | 0.079788 | 0.613868 | 0.613395 ($n=2\times10^5$) |
> | 1000 | 0.025238 | 0.025231 | 0.994207 | 0.994260 ($n=5\times10^4$) |
> | 10000 | 0.007979 | 0.007979 | $1 - 2.5\times10^{-18}$ | 1.000000 ($n=5000$) |
>
> In $\mathbb{R}^{10000}$, two random directions are within five degrees of perpendicular with probability $1-2.5\times10^{-18}$. Figure 12.3 below shows the whole angle density collapsing onto $90^\circ$ as $d$ grows. This is why you can pack an enormous number of *nearly* orthogonal directions into $\mathbb{R}^d$ — vastly more than $d$ — and it is the geometric fact underlying superposition in mechanistic interpretability: a transformer with residual width $d$ can represent many more than $d$ features if each is assigned a random direction and only approximate orthogonality is required. It is also why random initialization gives you a nearly orthogonal basis for free, and why the "similarity" of two random embeddings in a $1536$-dimensional space is essentially zero, so any cosine similarity above $0.1$ is already a strong signal.

### 12.1.3 The Gaussian is a soap bubble

Now the fact that most directly changes how you think about generative models.

**Proposition 12.3.** Let $Z\sim\mathcal{N}(0,I_d)$. Then $\|Z\|^2\sim\chi^2_d$ and
$$\mathbb{E}\|Z\|^2 = d,\qquad \operatorname{Var}(\|Z\|^2)=2d,\qquad \mathbb{E}\|Z\| = \sqrt2\,\frac{\Gamma\big(\tfrac{d+1}2\big)}{\Gamma\big(\tfrac d2\big)},\qquad \operatorname{Var}(\|Z\|) = d - (\mathbb{E}\|Z\|)^2 \le 1 .$$
Moreover $\mathbb{E}\|Z\| = \sqrt{d-\tfrac12}\,(1+O(d^{-2}))$ and $\operatorname{Var}(\|Z\|)\to\tfrac12$.

*Proof.* $\|Z\|^2=\sum_i Z_i^2$ is a sum of $d$ i.i.d. $\chi^2_1$ variables, each with mean $1$ and variance $2$; the first two displays follow by linearity and independence. The formula for $\mathbb{E}\|Z\| = \mathbb{E}[(\chi^2_d)^{1/2}]$ is the $\tfrac12$-moment of a $\chi^2_d$, obtained from $\mathbb{E}[(\chi^2_d)^{s}] = 2^s\Gamma(d/2+s)/\Gamma(d/2)$ with $s=\tfrac12$. Then $\operatorname{Var}(\|Z\|)=\mathbb{E}\|Z\|^2-(\mathbb{E}\|Z\|)^2 = d-(\mathbb{E}\|Z\|)^2$. The bound $\operatorname{Var}(\|Z\|)\le 1$ follows from Theorem 12.7 below (the map $z\mapsto\|z\|$ is $1$-Lipschitz), or directly from the Gaussian Poincaré inequality. The asymptotics follow from $\log\Gamma$ expansions; numerically the agreement is striking, e.g. at $d=100$, $\mathbb{E}\|Z\| = 9.975032$ versus $\sqrt{99.5}=9.974969$. $\square$

The interpretation: the *squared* norm has mean $d$ and standard deviation $\sqrt{2d}$, so its relative fluctuation is $\sqrt{2/d}\to0$. Taking square roots, $\|Z\|$ has mean $\approx\sqrt d$ and standard deviation $\approx 1/\sqrt2 = 0.7071$ — a constant, not growing with $d$. The mass sits in an annulus of *fixed* width around a radius that grows like $\sqrt d$. Relative to the radius, the annulus is infinitely thin. That is the soap bubble.

> **Worked Example 12.5 (the mode is in a desert).**
> Exact values, computed from the $\chi^2_d$ CDF (and, for the last column, from the regularized incomplete Gamma function at 40-digit precision):
>
> | $d$ | $\mathbb{E}\|Z\|$ | $\mathrm{sd}(\|Z\|)$ | $\sqrt d$ | $\Pr(\big|\|Z\|-\sqrt d\big|\le1)$ | $\log_{10}\Pr(\|Z\|\le1)$ |
> |---|---|---|---|---|---|
> | 2 | 1.2533 | 0.6551 | 1.4142 | 0.863544 | $-0.4$ |
> | 10 | 3.0843 | 0.6978 | 3.1623 | 0.844294 | $-3.8$ |
> | 100 | 9.9750 | 0.7062 | 10.0000 | 0.842820 | $-79.7$ |
> | 1000 | 31.6149 | 0.7070 | 31.6228 | 0.842712 | $-1284.8$ |
> | 10000 | 99.9975 | 0.7071 | 100.0000 | 0.842702 | $-17831.0$ |
>
> Two things to extract. First, the annulus $\sqrt d\pm1$ captures $84.3\%$ of the mass at essentially every dimension (the table shows it settling by $d=10$; at $d=2$ it is $86.4\%$) — the width does not grow. Second, the unit ball around the origin, which is where the density is largest, has probability $10^{-1285}$ at $d=1000$. The density at the origin exceeds the density on the shell by a factor $e^{d/2}$, which at $d=1000$ is $10^{217}$ — and yet the shell wins, because it has $10^{1502}$ times more volume. Density times volume is what counts, and in high dimension volume wins by a landslide.
>
> Practical consequence, stated concretely. When you sample a VAE or a diffusion model's latent $z\sim\mathcal{N}(0,I_k)$ with $k=512$, the typical draw has $\|z\|\approx\sqrt{512}=22.63$, and draws with $\|z\|<20$ are rare but *not* impossible: $\Pr(\|z\|<20)=8.08\times10^{-5}$, so you would expect about $81$ of them per million samples. Push a little further and rarity becomes practical impossibility: $\Pr(\|z\|<18)=5.9\times10^{-12}$. (Chapter 1 insisted that "probability zero" is not "impossible"; the converse discipline matters just as much — do not write "never" when you mean $10^{-5}$.) So if you evaluate the decoder at $z=0$ — "the most likely latent" — you are evaluating it at a point the decoder effectively never saw during training. Practitioners rediscover this constantly: the "truncation trick" in BigGAN, which resamples latents whose norm exceeds a threshold, works precisely because it moves samples *off* the typical shell toward the mode, trading diversity for fidelity. And linear interpolation between two latents $z_0,z_1$ passes through the low-norm desert at the midpoint ($\|z_0+z_1\|/2 \approx \sqrt{k/2}$, a factor $1/\sqrt2$ too small), which is why spherical interpolation (`slerp`) produces visibly better intermediate samples. That is not a heuristic; it is Proposition 12.3.

<svg viewBox="0 0 760 300" style="max-width:100%;height:auto" role="img" aria-label="Two panels: shell volume fraction rising to one, and log volume of the unit ball collapsing.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.1 — Volume abandons the interior</text>
  <!-- Panel A -->
  <text x="200" y="40" text-anchor="middle" font-size="12" fill="currentColor">(a) fraction of ball volume outside radius 1-eps</text>
  <line x1="70" y1="205" x2="340" y2="205" stroke="currentColor" stroke-width="1"/>
  <line x1="70" y1="205" x2="70" y2="45" stroke="currentColor" stroke-width="1"/>
  <text x="66" y="49" text-anchor="end" font-size="10" fill="currentColor">1.0</text>
  <text x="66" y="128" text-anchor="end" font-size="10" fill="currentColor">0.5</text>
  <text x="66" y="209" text-anchor="end" font-size="10" fill="currentColor">0</text>
  <text x="70" y="220" text-anchor="middle" font-size="10" fill="currentColor">1</text>
  <text x="200" y="220" text-anchor="middle" font-size="10" fill="currentColor">30</text>
  <text x="330" y="220" text-anchor="middle" font-size="10" fill="currentColor">60</text>
  <text x="205" y="234" text-anchor="middle" font-size="11" fill="currentColor">dimension d</text>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.2" points="70,189 79,162 88,139 96,122 105,107 114,95 123,86 132,78 141,72 149,67 158,63 167,59 176,56 185,54 193,53 202,51 211,50 220,49 229,48 237,48 246,47 255,47 264,46 273,46 282,46 290,46 299,46 308,45 317,45 326,45"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="5,3" points="70,203 79,200 88,197 96,194 105,191 114,188 123,185 132,183 141,180 149,177 158,175 167,172 176,169 185,167 193,165 202,162 211,160 220,158 229,155 237,153 246,151 255,149 264,147 273,145 282,143 290,141 299,139 308,137 317,135 326,133"/>
  <text x="240" y="70" font-size="10" fill="#3b82f6">eps = 0.1</text>
  <text x="240" y="150" font-size="10" fill="#f59e0b">eps = 0.01</text>
  <!-- Panel B -->
  <text x="550" y="40" text-anchor="middle" font-size="12" fill="currentColor">(b) log10 volume, unit ball and ball-in-cube</text>
  <line x1="420" y1="205" x2="690" y2="205" stroke="currentColor" stroke-width="1"/>
  <line x1="420" y1="205" x2="420" y2="45" stroke="currentColor" stroke-width="1"/>
  <line x1="420" y1="55" x2="690" y2="55" stroke="currentColor" stroke-width="0.7" stroke-dasharray="3,3"/>
  <text x="416" y="59" text-anchor="end" font-size="10" fill="currentColor">0</text>
  <text x="416" y="109" text-anchor="end" font-size="10" fill="currentColor">-20</text>
  <text x="416" y="159" text-anchor="end" font-size="10" fill="currentColor">-40</text>
  <text x="416" y="205" text-anchor="end" font-size="10" fill="currentColor">-60</text>
  <text x="420" y="220" text-anchor="middle" font-size="10" fill="currentColor">1</text>
  <text x="550" y="220" text-anchor="middle" font-size="10" fill="currentColor">30</text>
  <text x="680" y="220" text-anchor="middle" font-size="10" fill="currentColor">60</text>
  <text x="555" y="234" text-anchor="middle" font-size="11" fill="currentColor">dimension d</text>
  <polyline fill="none" stroke="#10b981" stroke-width="2.2" points="420,54 429,53 438,53 446,53 455,54 464,54 473,55 482,56 491,57 499,58 508,60 517,61 526,63 535,64 543,66 552,68 561,69 570,71 579,73 587,75 596,77 605,79 614,81 623,84 632,86 640,88 649,90 658,93 667,95 676,98"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2.2" points="420,55 429,56 438,57 446,59 455,60 464,63 473,65 482,67 491,70 499,73 508,75 517,78 526,81 535,84 543,88 552,91 561,94 570,98 579,101 587,104 596,108 605,112 614,115 623,119 632,123 640,126 649,130 658,134 667,138 676,142"/>
  <text x="600" y="72" font-size="10" fill="#10b981">V_d(1)</text>
  <text x="560" y="112" font-size="10" fill="#ef4444">V_d(1) / 2^d  (ball in cube)</text>
  <text x="380" y="258" text-anchor="middle" font-size="11" fill="currentColor">Left: for any fixed shell thickness eps, essentially all volume of the d-ball lies in the shell once d is large.</text>
  <text x="380" y="273" text-anchor="middle" font-size="11" fill="currentColor">Right: the unit ball's volume peaks at d = 5, then decays like exp(-(d/2) log d); the inscribed ball occupies</text>
  <text x="380" y="288" text-anchor="middle" font-size="11" fill="currentColor">a fraction 2^-d of that, so box-based space partitioning cannot approximate ball queries in high dimension.</text>
</svg>

The prose statement of Figure 12.1, in case it does not render: panel (a) plots $1-(1-\varepsilon)^d$ against $d$ for $\varepsilon=0.1$ (rising to $0.9999$ by $d=100$) and $\varepsilon=0.01$ (rising more slowly, $0.634$ at $d=100$); panel (b) plots $\log_{10}V_d(1)$, which rises slightly to a maximum near $d=5$ and then falls without bound, together with $\log_{10}(V_d(1)2^{-d})$, which falls faster still.

<svg viewBox="0 0 760 300" style="max-width:100%;height:auto" role="img" aria-label="Densities of the norm of a standard Gaussian for several dimensions, each peaking near the square root of d.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.2 — The Gaussian norm concentrates at sqrt(d) with fixed width</text>
  <line x1="80" y1="235" x2="710" y2="235" stroke="currentColor" stroke-width="1"/>
  <line x1="80" y1="235" x2="80" y2="50" stroke="currentColor" stroke-width="1"/>
  <text x="80" y="249" text-anchor="middle" font-size="10" fill="currentColor">0</text>
  <text x="169" y="249" text-anchor="middle" font-size="10" fill="currentColor">2</text>
  <text x="257" y="249" text-anchor="middle" font-size="10" fill="currentColor">4</text>
  <text x="346" y="249" text-anchor="middle" font-size="10" fill="currentColor">6</text>
  <text x="434" y="249" text-anchor="middle" font-size="10" fill="currentColor">8</text>
  <text x="523" y="249" text-anchor="middle" font-size="10" fill="currentColor">10</text>
  <text x="611" y="249" text-anchor="middle" font-size="10" fill="currentColor">12</text>
  <text x="700" y="249" text-anchor="middle" font-size="10" fill="currentColor">14</text>
  <text x="395" y="264" text-anchor="middle" font-size="11" fill="currentColor">r = ||Z||</text>
  <text x="72" y="54" text-anchor="end" font-size="10" fill="currentColor">density</text>
  <polyline fill="none" stroke="currentColor" stroke-width="1.6" points="80,55 89,59 98,70 108,87 117,108 126,130 135,152 144,172 153,189 162,203 172,214 181,222 190,227 199,230 208,232 217,234 227,234 236,235 245,235 254,235 263,235 272,235 282,235 291,235 300,235 309,235 318,235 327,235 337,235 346,235"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="1.8" points="80,235 91,233 101,224 112,202 123,171 133,139 144,115 155,105 166,109 176,126 187,149 198,173 208,194 219,210 230,221 240,227 251,231 262,233 272,234 283,235 294,235 304,235 315,235 326,235 337,235 347,235 358,235 369,235 379,235 390,235"/>
  <polyline fill="none" stroke="#10b981" stroke-width="1.8" points="80,235 94,235 108,235 121,235 135,235 149,235 162,234 176,231 190,222 204,200 217,168 231,132 245,110 259,111 272,132 286,163 300,193 314,214 327,226 341,231 355,234 369,235 382,235 396,235 410,235 424,235 437,235 451,235 465,235 479,235"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.8" points="213,235 228,235 243,235 259,235 274,235 289,235 304,235 320,235 335,234 350,232 366,223 381,203 396,169 411,132 427,109 442,115 457,143 472,179 488,208 503,224 518,231 534,234 549,235 564,235 579,235 595,235 610,235 625,235 640,235 656,235"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="1.8" points="301,235 315,235 329,235 342,235 356,235 370,235 383,235 397,235 411,235 424,234 438,232 452,225 465,210 479,184 493,150 506,120 520,108 534,119 547,146 561,179 575,205 588,221 602,230 616,233 629,235 643,235 657,235 670,235 684,235 698,235"/>
  <text x="100" y="48" font-size="10" fill="currentColor">d=1</text>
  <text x="150" y="98" font-size="10" fill="#f59e0b">d=4</text>
  <text x="238" y="103" font-size="10" fill="#10b981">d=16</text>
  <text x="420" y="102" font-size="10" fill="#3b82f6">d=64</text>
  <text x="513" y="101" font-size="10" fill="#ef4444">d=100</text>
  <circle cx="80" cy="235" r="3" fill="#ef4444"/>
  <text x="88" y="228" font-size="10" fill="#ef4444">mode of the DENSITY p(z) is here (r=0)</text>
  <text x="380" y="282" text-anchor="middle" font-size="11" fill="currentColor">Each curve is the density of ||Z|| for Z ~ N(0, I_d). The peak sits at about sqrt(d) and the width stays near 0.71</text>
  <text x="380" y="296" text-anchor="middle" font-size="11" fill="currentColor">for every d. The density of Z is maximal at r = 0, but P(||Z|| &lt;= 1) is 1e-80 at d=100: mass, not density, decides.</text>
</svg>

If Figure 12.2 does not render: it shows five unimodal densities for $\|Z\|$, at $d=1,4,16,64,100$, whose peaks march right along the axis at $r\approx\sqrt d$ ($1$, $2$, $4$, $8$, $10$) while their widths stay essentially constant at about $0.7$; a marker at $r=0$ notes that this is where the density of $Z$ itself is maximal, and that this location carries negligible probability.

<svg viewBox="0 0 760 300" style="max-width:100%;height:auto" role="img" aria-label="Densities of the angle between two independent random directions, sharpening onto ninety degrees as dimension grows.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.3 — Two random directions become perpendicular</text>
  <line x1="70" y1="230" x2="690" y2="230" stroke="currentColor" stroke-width="1"/>
  <line x1="70" y1="230" x2="70" y2="50" stroke="currentColor" stroke-width="1"/>
  <text x="70" y="245" text-anchor="middle" font-size="10" fill="currentColor">0</text>
  <text x="173" y="245" text-anchor="middle" font-size="10" fill="currentColor">30</text>
  <text x="277" y="245" text-anchor="middle" font-size="10" fill="currentColor">60</text>
  <text x="380" y="245" text-anchor="middle" font-size="10" fill="currentColor">90</text>
  <text x="483" y="245" text-anchor="middle" font-size="10" fill="currentColor">120</text>
  <text x="587" y="245" text-anchor="middle" font-size="10" fill="currentColor">150</text>
  <text x="690" y="245" text-anchor="middle" font-size="10" fill="currentColor">180</text>
  <text x="380" y="260" text-anchor="middle" font-size="11" fill="currentColor">angle between U and V (degrees)</text>
  <text x="62" y="54" text-anchor="end" font-size="10" fill="currentColor">density</text>
  <line x1="380" y1="230" x2="380" y2="50" stroke="currentColor" stroke-width="0.8" stroke-dasharray="4,3"/>
  <text x="384" y="60" font-size="10" fill="currentColor">90 degrees</text>
  <polyline fill="none" stroke="currentColor" stroke-width="1.8" points="70,226 88,226 105,226 123,226 141,226 158,226 176,226 194,226 212,226 229,226 247,226 265,226 282,226 300,226 318,226 336,226 353,226 371,226 389,226 406,226 424,226 442,226 460,226 477,226 495,226 513,226 530,226 548,226 566,226 584,226 601,226 619,226 637,226 654,226 672,226 690,226"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="1.8" points="188,230 199,230 210,230 221,229 232,229 243,228 254,227 265,227 276,225 286,224 298,223 308,221 320,220 330,219 342,217 352,216 363,216 374,215 385,215 396,216 407,216 418,217 429,218 440,220 451,221 462,223 473,224 484,225 495,227 506,227 517,228 528,229 539,229 550,230 561,230 572,230"/>
  <polyline fill="none" stroke="#10b981" stroke-width="1.8" points="313,230 316,230 320,229 324,229 328,228 332,227 336,226 339,224 343,221 347,217 351,212 355,207 359,202 363,196 366,190 370,185 374,182 378,180 382,180 386,182 390,185 393,190 397,195 401,201 405,207 409,212 413,217 416,220 420,224 424,226 428,227 432,228 435,229 440,229 443,230 447,230"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.8" points="357,230 358,230 359,229 360,229 362,227 363,226 365,223 366,217 367,210 369,200 370,188 371,169 373,151 374,132 375,108 377,92 378,79 379,71 381,70 382,77 383,88 385,108 386,127 387,146 388,165 390,185 391,197 392,208 394,217 395,222 396,225 398,227 399,229 400,229 402,230 403,230"/>
  <text x="600" y="220" font-size="10" fill="currentColor">d = 2 (uniform)</text>
  <text x="577" y="214" font-size="10" fill="#f59e0b">d = 10</text>
  <text x="452" y="184" font-size="10" fill="#10b981">d = 100</text>
  <text x="408" y="76" font-size="10" fill="#3b82f6">d = 1000</text>
  <text x="380" y="278" text-anchor="middle" font-size="11" fill="currentColor">Exact density of the angle, proportional to sin^(d-2)(theta). At d = 2 it is flat: any angle is equally likely.</text>
  <text x="380" y="292" text-anchor="middle" font-size="11" fill="currentColor">By d = 1000 essentially all mass is within a few degrees of 90. The peak height grows like sqrt(d).</text>
</svg>

Without the figure: the exact density of the angle $\Theta$ is proportional to $\sin^{d-2}\theta$ on $[0,\pi]$. At $d=2$ it is flat (peak height $1/180$ per degree $=0.0056$) — every angle equally likely. At $d=10$ it is a broad hump peaking at $0.0203$; at $d=100$ a clear bell of peak $0.069$ and visible width around $\pm20^\circ$; at $d=1000$ a narrow spike of peak $0.220$ confined to roughly $85^\circ$–$95^\circ$. The peak height grows like $\sqrt d$ and the width shrinks like $1/\sqrt d$, which is Proposition 12.2(3) restated: $\operatorname{Var}(\cos\Theta)=1/d$.

### 12.1.4 The four failures side by side

| Fact | $d=2$ | $d=10$ | $d=100$ | $d=1000$ | $d=10000$ |
|---|---|---|---|---|---|
| (a) volume fraction outside $r=0.9$ | 0.190000 | 0.651322 | 0.999973 | 1.000000 | 1.000000 |
| (b) $\log_{10} V_d(1)$ | $+0.50$ | $+0.41$ | $-39.63$ | $-885.51$ | $-13839.88$ |
| (b′) $\log_{10}$ (inscribed ball / unit cube) | $-0.10$ | $-2.60$ | $-69.73$ | $-1186.54$ | $-16850.18$ |
| (c) $\Pr(\text{angle within }5^\circ\text{ of }90^\circ)$ | 0.055556 | 0.201129 | 0.613868 | 0.994207 | $1-2.5\!\times\!10^{-18}$ |
| (d) $\mathbb{E}\|Z\|$ vs. $\sqrt d$ | 1.2533 / 1.4142 | 3.0843 / 3.1623 | 9.9750 / 10.0000 | 31.6149 / 31.6228 | 99.9975 / 100.0000 |
| (d′) $\mathrm{sd}(\|Z\|)$ | 0.6551 | 0.6978 | 0.7062 | 0.7070 | 0.7071 |
| (d″) $\log_{10}\Pr(\|Z\|\le1)$ | $-0.4$ | $-3.8$ | $-79.7$ | $-1284.8$ | $-17831.0$ |

All entries are exact (Gamma and $\chi^2$ special functions, plus 40-digit arithmetic for the last row), not simulated.

> **Common Confusion 12.1.**
> *"But isn't the mode the most likely value? If I want a typical sample from $\mathcal{N}(0,I_d)$, shouldn't I look near the origin, where $p(z)$ is biggest?"*
>
> The mode maximizes the *density*, which is a ratio: probability per unit volume. It says nothing about probability until you multiply by volume. In one dimension the two are nearly the same because the volume element barely varies, so the habit is harmless. In $d$ dimensions the volume element at radius $r$ scales like $r^{d-1}$, and the competition is between $e^{-r^2/2}$ (shrinking) and $r^{d-1}$ (growing). The product $r^{d-1}e^{-r^2/2}$ is maximized at $r=\sqrt{d-1}$, not at $r=0$. At $d=1000$ the density at the origin is $10^{217}$ times the density on the shell, and the shell still wins because it has $10^{1502}$ times as much room.
>
> The technically correct statement is that $\mathcal{N}(0,I_d)$ has a *typical set*: a region of near-constant density carrying almost all the probability, whose density is exponentially smaller than the maximum. This is exactly the asymptotic equipartition property from information theory (Ch. 11), read geometrically. It is also why MAP estimation and posterior sampling can give qualitatively different answers in high dimension, and why "the posterior mode" is a poor summary of a high-dimensional posterior. Chapter 13's treatment of MCMC leans on this: a sampler tuned to find the mode will be a bad sampler.

> **Self-check 12.1.**
> 1. In $\mathbb{R}^{50}$, what fraction of the unit ball's volume lies at radius greater than $0.95$?
> 2. True or false: because $V_d(1)\to0$, a high-dimensional ball is "small". Explain what is actually shrinking.
> 3. You draw $z\sim\mathcal{N}(0,I_{256})$. Give a one-standard-deviation interval for $\|z\|$.
> 4. Two independent vectors are drawn uniformly on $S^{999}$. Roughly what is $\mathbb{E}|\cos\Theta|$, and why is this relevant to storing many features in a residual stream of width $1000$?

## 12.2 Concentration of measure

The four failures of §12.1 all have one cause. A high-dimensional random vector is a sum of many independent contributions, and any sufficiently smooth function of it is therefore an average-like quantity — and averages concentrate. This section makes that precise, first for the norm (where we can prove everything), then in the general Lipschitz form (where we must import one deep result).

### 12.2.1 Norm concentration for Gaussians

**Lemma 12.4 (chi-square moment generating function and tails).** Let $W\sim\chi^2_d$. Then $\mathbb{E}[e^{sW}] = (1-2s)^{-d/2}$ for $s<\tfrac12$, and for every $t>0$
$$\Pr\big(W \ge d + 2\sqrt{dt}+2t\big)\le e^{-t},\qquad \Pr\big(W\le d-2\sqrt{dt}\big)\le e^{-t}. \tag{12.1}$$

*Proof.* For $W_1\sim\chi^2_1$, i.e. $W_1=Z_1^2$ with $Z_1\sim\mathcal{N}(0,1)$,
$$\mathbb{E}[e^{sZ_1^2}] = \frac1{\sqrt{2\pi}}\int_{\mathbb{R}}e^{sz^2-z^2/2}\,dz = \frac1{\sqrt{2\pi}}\int_{\mathbb{R}}e^{-(1-2s)z^2/2}\,dz = (1-2s)^{-1/2},$$
finite exactly when $s<\tfrac12$. By independence, $\mathbb{E}[e^{sW}] = (1-2s)^{-d/2}$.

Now Chernoff (Ch. 4): for $s\in(0,\tfrac12)$ and $u>0$,
$$\Pr(W \ge d+u)\le e^{-s(d+u)}(1-2s)^{-d/2} = \exp\Big(-s(d+u)-\tfrac d2\log(1-2s)\Big).$$
Write $\psi(s) = -s(d+u)-\tfrac d2\log(1-2s)$. Then $\psi'(s) = -(d+u)+\dfrac{d}{1-2s}$, which vanishes when $1-2s = \dfrac{d}{d+u}$, i.e. at $s^\star = \dfrac{u}{2(d+u)}\in(0,\tfrac12)$. At that point $-s^\star(d+u) = -u/2$ and $-\tfrac d2\log(1-2s^\star) = -\tfrac d2\log\dfrac{d}{d+u} = \tfrac d2\log\big(1+\tfrac ud\big)$, so
$$\psi(s^\star) = -\frac u2 + \frac d2\log\Big(1+\frac ud\Big) = -\frac d2\Big[\frac ud-\log\Big(1+\frac ud\Big)\Big] =: -\frac d2\,h\Big(\frac ud\Big),$$
with $h(x)=x-\log(1+x)\ge0$ for $x\ge 0$. Now substitute $u = 2\sqrt{dt}+2t$ and set $a := \sqrt{t/d}$, so that $u/d = 2a+2a^2$ and
$$\frac d2 h\Big(\frac ud\Big) = \frac d2\Big(2a+2a^2-\log(1+2a+2a^2)\Big).$$
Since $e^{2a} = 1+2a+2a^2+\tfrac43a^3+\cdots \ge 1+2a+2a^2$ for $a\ge0$, we have $\log(1+2a+2a^2)\le 2a$, hence $2a+2a^2-\log(1+2a+2a^2)\ge 2a^2$ and therefore $\tfrac d2 h(u/d)\ge \tfrac d2\cdot 2a^2 = t$, giving the first bound in (12.1).

The lower tail is analogous with $s<0$: $\Pr(W\le d-u)\le \exp(-\tfrac d2[\,-\tfrac ud-\log(1-\tfrac ud)\,])$ and $-x-\log(1-x)\ge x^2/2$ for $x\in[0,1)$, so taking $u=2\sqrt{dt}$ gives exponent $\ge \tfrac d2\cdot\tfrac{4t/d}{2} = t$. $\square$

Inequality (12.1) is the Laurent–Massart form of the $\chi^2$ tail. Its shape encodes exactly the two regimes you should expect: for small $t$ the deviation is $\approx2\sqrt{dt}$ (Gaussian, sub-Gaussian scaling, matching the CLT of Ch. 7), and for large $t$ it is $\approx 2t$ (exponential, sub-exponential scaling, because $\chi^2_1$ has heavier-than-Gaussian tails). This is precisely Bernstein's dichotomy from Chapter 4.

> **Worked Example 12.6 (how tight is (12.1)?).**
> Exact tail probabilities from the $\chi^2_d$ survival function at $x = d+2\sqrt{dt}+2t$:
>
> | $d$ | $t=1$ | $t=3$ | $t=5$ | $t=10$ |
> |---|---|---|---|---|
> | 10 | $4.97\times10^{-2}$ | $2.65\times10^{-3}$ | $1.75\times10^{-4}$ | $2.67\times10^{-7}$ |
> | 100 | $6.67\times10^{-2}$ | $4.63\times10^{-3}$ | $3.71\times10^{-4}$ | $7.67\times10^{-7}$ |
> | 1000 | $7.45\times10^{-2}$ | $6.12\times10^{-3}$ | $5.91\times10^{-4}$ | $1.98\times10^{-6}$ |
> | bound $e^{-t}$ | $3.68\times10^{-1}$ | $4.98\times10^{-2}$ | $6.74\times10^{-3}$ | $4.54\times10^{-5}$ |
>
> The bound is loose by a factor between $6$ and $23$ over this range, and — importantly — the looseness does not grow with $d$. That is the useful property: the bound is dimension-uniform. A bound that is off by one order of magnitude but valid for all $d$ is far more useful in a proof than an asymptotically exact expression that you cannot control at finite $d$.

**Theorem 12.5 (Gaussian annulus theorem).** Let $Z\sim\mathcal{N}(0,I_d)$. For every $t\ge0$,
$$\Pr\big(\big|\,\|Z\|-\sqrt d\,\big|\ge t\big)\le 2e^{-t^2/8}. \tag{12.2}$$

*Proof.* Both tails follow from (12.1) with the choice $s := t^2/4$.

*Upper tail.* The event $\{\|Z\|\ge\sqrt d+t\}$ equals $\{\|Z\|^2\ge d+2t\sqrt d+t^2\}$. To apply the first bound of (12.1) we need $d+2\sqrt{ds}+2s\le d+2t\sqrt d+t^2$, i.e. $t\sqrt d+\tfrac{t^2}2\le 2t\sqrt d+t^2$, which holds for every $t\ge0$. Hence $\Pr(\|Z\|\ge\sqrt d+t)\le e^{-t^2/4}\le e^{-t^2/8}$.

*Lower tail.* If $t>\sqrt d$ the event $\{\|Z\|\le\sqrt d-t\}$ is empty and there is nothing to prove. For $0\le t\le\sqrt d$ the event equals $\{\|Z\|^2\le d-2t\sqrt d+t^2\}$, and to apply the second bound of (12.1) we need $d-2\sqrt{ds}\ge d-2t\sqrt d+t^2$, i.e. $t\sqrt d\le 2t\sqrt d-t^2$, i.e. $t\le\sqrt d$ — exactly our range. Hence $\Pr(\|Z\|\le\sqrt d-t)\le e^{-t^2/4}\le e^{-t^2/8}$.

Summing the two tails gives (12.2). $\square$

Two remarks. First, the constant $\tfrac18$ is not sharp: the correct exponent is $t^2/2$, which we will get for free from Theorem 12.7 below (with $\mathbb{E}\|Z\|$ in place of $\sqrt d$, a difference of at most $1/(2\sqrt d)$). We proved the weaker version because the proof is three lines and uses nothing we have not established. Second, and more important than any constant: **the bound has no $d$ in it.** The annulus of half-width $t$ around radius $\sqrt d$ captures all but $2e^{-t^2/8}$ of the mass, whatever the dimension. Set $t=6$: at least $1-2e^{-4.5}=97.8\%$ of the mass of $\mathcal{N}(0,I_{10^6})$ lies within $6$ of the sphere of radius $1000$.

**Theorem 12.6 (norm concentration for sub-Gaussian vectors).** Let $X=(X_1,\dots,X_d)$ have independent coordinates with $\mathbb{E}X_i=0$, $\mathbb{E}X_i^2=1$, and sub-Gaussian norm $\|X_i\|_{\psi_2}\le K$ (Ch. 4). Then there is an absolute constant $c>0$ with
$$\Pr\big(\big|\,\|X\|-\sqrt d\,\big|\ge t\big)\le 2\exp\big(-ct^2/K^4\big)\qquad\text{for all }t\ge0.$$

*Proof sketch (the mechanism, which is what matters).* The variables $Y_i := X_i^2-1$ are independent, mean zero, and sub-exponential with $\|Y_i\|_{\psi_1}\lesssim K^2$ (squaring a sub-Gaussian variable gives a sub-exponential one — Ch. 4). Bernstein's inequality for sub-exponential summands (Ch. 4) gives, for all $u\ge0$,
$$\Pr\Big(\Big|\tfrac1d\|X\|^2-1\Big|\ge u\Big)\le 2\exp\Big(-c\,d\min\big(u^2/K^4,\,u/K^2\big)\Big).$$
Now convert from squared norm to norm using the elementary inequality $|z-1|\ge\delta\Rightarrow|z^2-1|\ge\max(\delta,\delta^2)$, valid for $z\ge0$: apply it with $z = \|X\|/\sqrt d$ and $\delta=t/\sqrt d$. If $\delta\le1$ then $|z^2-1|\ge\delta$ and we take $u=\delta = t/\sqrt d$, so $d\min(u^2/K^4,u/K^2) \ge c' t^2/K^4$ when $t\le\sqrt d K^2$; if $\delta>1$ then $|z^2-1|\ge\delta^2$ and $u=\delta^2 = t^2/d$ gives exponent $\gtrsim t^2/K^2$. Either way the exponent is $\gtrsim t^2/K^4$. $\square$

The take-away is the *shape*: fluctuations of $\|X\|$ around $\sqrt d$ are $O(1)$, not $O(\sqrt d)$. This is what makes the phrase "the data lies on a sphere of radius $\sqrt d$" a legitimate approximation rather than sloppiness. It holds for Rademacher coordinates, for uniform coordinates, for any bounded coordinates — not just Gaussian.

### 12.2.2 Lipschitz concentration: the surprising theorem

Everything so far concerned one particular function, $x\mapsto\|x\|$. The following says the phenomenon has nothing to do with the norm.

**Theorem 12.7 (Gaussian Lipschitz concentration; Borell, Tsirelson–Ibragimov–Sudakov).** Let $Z\sim\mathcal{N}(0,I_d)$ and let $f:\mathbb{R}^d\to\mathbb{R}$ be $L$-Lipschitz with respect to the Euclidean norm, i.e. $|f(x)-f(y)|\le L\|x-y\|$ for all $x,y$. Then $f(Z)$ is integrable and for every $t\ge 0$
$$\Pr\big(|f(Z)-\mathbb{E}f(Z)|\ge t\big)\le 2\exp\Big(-\frac{t^2}{2L^2}\Big). \tag{12.3}$$
Equivalently, $f(Z)-\mathbb{E}f(Z)$ is sub-Gaussian with $\|f(Z)-\mathbb{E}f(Z)\|_{\psi_2}\lesssim L$. In particular $\operatorname{Var}(f(Z))\le L^2$.

We do not prove this; see Results used but not proved. But look hard at what it says. **The dimension $d$ does not appear.** A $1$-Lipschitz function of a million independent standard Gaussians has standard deviation at most $1$. Not $\sqrt{10^6}$. Not $\log 10^6$. At most $1$, the same as for a single Gaussian.

Why should you find this shocking, and why is it nonetheless plausible? Shocking, because a Lipschitz function can be a very complicated object; a $1$-Lipschitz function on $\mathbb{R}^{10^6}$ has an enormous amount of freedom, and you might expect it to be able to exploit the vastness of the space to spread its values out. Plausible, because the Lipschitz constraint caps how fast $f$ can change *per unit distance*, and — by Theorem 12.5 — the region where $Z$ actually lives has diameter $O(\sqrt d)$ in a way that is misleading: what matters is that in the directions that matter, the accessible range is $O(1)$. A cleaner heuristic: reveal the coordinates of $Z$ one at a time and write $f(Z)-\mathbb{E}f(Z)$ as the sum of the resulting increments $\mathbb{E}[f(Z)\mid Z_1,\dots,Z_i]-\mathbb{E}[f(Z)\mid Z_1,\dots,Z_{i-1}]$ — a telescoping decomposition into conditional-expectation differences. Each increment is controlled by the corresponding partial derivative, the increments are conditionally sub-Gaussian, and $\sum_i(\partial_if)^2 = \|\nabla f\|^2\le L^2$ — the Lipschitz constant *budgets the total sensitivity across all coordinates at once*, so adding dimensions does not add sensitivity, it only divides the existing budget more finely.

**Corollary 12.8.** Taking $f(x)=\|x\|$, which is $1$-Lipschitz by the triangle inequality: $\operatorname{Var}(\|Z\|)\le1$ and $\Pr(|\|Z\|-\mathbb{E}\|Z\||\ge t)\le2e^{-t^2/2}$, recovering Proposition 12.3's variance bound and sharpening Theorem 12.5's exponent from $t^2/8$ to $t^2/2$, at the cost of centring at $\mathbb{E}\|Z\|$ rather than $\sqrt d$. Since $|\mathbb{E}\|Z\|-\sqrt d|\le 1/(2\sqrt d)$ (from $\mathbb{E}\|Z\|=\sqrt{d-1/2}+O(d^{-3/2})$), the two centrings are interchangeable up to negligible error.

> **Worked Example 12.7 (Lipschitz concentration you can watch).**
> Take $f(x)=\max_{1\le i\le d}x_i$. This is $1$-Lipschitz: if $x,y$ differ by $\delta$ in Euclidean norm then they differ by at most $\delta$ in each coordinate, so the maxima differ by at most $\delta$. Theorem 12.7 predicts $\mathrm{sd}(f(Z))\le1$ *for every $d$*, even though $\mathbb{E}f(Z)$ grows.
>
> Monte Carlo, `default_rng(2024)`, $n=20{,}000$ draws per dimension:
>
> | $d$ | $\widehat{\mathbb{E}}[\max_i Z_i]$ | $\widehat{\mathrm{sd}}$ | $\sqrt{2\log d}$ |
> |---|---|---|---|
> | 10 | 1.5476 | 0.5867 | 2.1460 |
> | 100 | 2.5130 | 0.4314 | 3.0349 |
> | 1000 | 3.2455 | 0.3515 | 3.7169 |
> | 10000 | 3.8519 | 0.3031 | 4.2919 |
>
> The mean grows (like $\sqrt{2\log d}$, the classical extreme-value scale — note the estimate approaches it slowly from below, as expected since $\mathbb{E}\max = \sqrt{2\log d}(1-o(1))$ with a $\log\log d$ correction). The standard deviation *decreases*, and is comfortably under the bound $1$ at every $d$. So the maximum of ten thousand independent Gaussians is a nearly deterministic number, $3.85\pm0.30$. If you have ever wondered why the largest logit of an untrained network is so reproducible across seeds, this is why.

> **Worked Example 12.8 (Lipschitz concentration and certified robustness).**
> Let $g:\mathbb{R}^{d}\to\mathbb{R}$ be a network's margin function (logit of the true class minus the largest other logit). Randomized smoothing defines $\bar g(x) = \mathbb{E}_{\delta\sim\mathcal{N}(0,\sigma^2I_d)}[g(x+\delta)]$ and classifies by $\bar g$. Two facts, both instances of the material above.
>
> First, for a *fixed* $x$, the Monte Carlo estimate $\frac1m\sum_j g(x+\delta_j)$ concentrates at rate $O(1/\sqrt m)$ — that is just Hoeffding (Ch. 4) if $g$ is bounded.
>
> Second, and more interestingly: write $p_c(x):=\Pr_\delta(\text{class}=c\mid x+\delta)$ for the smoothed classifier's class-$c$ probability. Cohen–Rosenfeld–Kolter's certificate is exactly the statement that $x\mapsto\Phi^{-1}(p_c(x))$ is $1/\sigma$-Lipschitz, which is a corollary of the Gaussian isoperimetric inequality (Theorem 12.10 below). The certified radius is $\sigma\Phi^{-1}(\underline{p_c})$, and it does **not** degrade with $d$ — the same dimension-freeness as (12.3). This is a genuinely load-bearing use of high-dimensional probability in a practical defence: the certificate's guarantee at $d=150528$ (ImageNet) is the same functional form as at $d=784$ (MNIST). The *empirical* accuracy degrades, but the geometry does not.
>
> One caveat, and it is the whole content: the certificate is in $\ell_2$. Attempts to transfer randomized smoothing to $\ell_\infty$ certificates *do* incur a $\sqrt d$ loss, and papers claiming otherwise usually hide the dimension in a norm conversion. Knowing which norm the Lipschitz constant is measured in is not pedantry here; it is the whole content.

### 12.2.3 Isoperimetry, Lévy, Talagrand

Where does (12.3) come from? From a statement about *sets*, not functions.

**Definition 12.9 (concentration function).** Let $(\mathcal{X},\rho,\mu)$ be a metric probability space. For $A\subseteq\mathcal{X}$ and $t>0$ let $A_t=\{x:\rho(x,A)\le t\}$ be the $t$-enlargement. The concentration function is
$$\alpha_\mu(t) := \sup\big\{1-\mu(A_t)\ :\ A\text{ measurable},\ \mu(A)\ge\tfrac12\big\}.$$
The space "has the concentration of measure property" when $\alpha_\mu(t)$ decays rapidly in $t$.

Read this in words: take any set that holds at least half the mass; fatten it by $t$; how much can be left over? If $\alpha_\mu(t)$ is tiny, then every half-mass set, however weirdly shaped, nearly fills the space after a small fattening.

**Theorem 12.10 (Gaussian isoperimetric inequality).** Let $\gamma_d=\mathcal{N}(0,I_d)$ on $\mathbb{R}^d$. Among all measurable $A$ with $\gamma_d(A)=\Phi(a)$, the half-space $H=\{x:x_1\le a\}$ minimizes $\gamma_d(A_t)$ for every $t>0$. Consequently $\gamma_d(A_t)\ge\Phi\big(\Phi^{-1}(\gamma_d(A))+t\big)$, and $\alpha_{\gamma_d}(t)\le1-\Phi(t)\le\tfrac12e^{-t^2/2}$, *independent of $d$*.

Half-spaces are the extremal sets — that is the whole content, and it is the Gaussian analogue of "balls are the extremal sets for the classical isoperimetric problem." From Theorem 12.10 one derives Theorem 12.7 in a few lines by applying the enlargement inequality to the sublevel sets $A=\{f\le \mathrm{med}(f)\}$: since $f$ is $L$-Lipschitz, $A_t\subseteq\{f\le\mathrm{med}(f)+Lt\}$, so $\Pr(f>\mathrm{med}(f)+Lt)\le1-\gamma_d(A_t)\le1-\Phi(t)$.

**Corollary 12.11 (Lévy's lemma on the sphere).** Let $\mu$ be the uniform probability measure on $S^{d-1}$ and $f:S^{d-1}\to\mathbb{R}$ be $L$-Lipschitz in the geodesic (or Euclidean) metric. Then
$$\mu\big(|f-\mathrm{med}(f)|\ge t\big)\le 2\exp\Big(-\frac{(d-1)t^2}{2L^2}\Big).$$
Here the dimension appears — and helps: on the *unit* sphere, concentration gets stronger as $d$ grows, because the sphere's diameter stays $2$ while its "effective volume" spreads. The two statements are consistent: $S^{d-1}$ scaled to radius $\sqrt d$ has $L$-Lipschitz functions concentrating at scale $L$, matching (12.3).

> **Worked Example 12.9 (Lévy's lemma: the equatorial band).**
> Take $f(u)=u_1$ on $S^{d-1}$, which is $1$-Lipschitz. Its median is $0$. Lévy gives $\mu(|u_1|\ge t)\le2e^{-(d-1)t^2/2}$. Compare with the exact answer from Proposition 12.2(2), $\Pr(|T|\ge t) = \Pr(\mathrm{Beta}(\tfrac12,\tfrac{d-1}2)\ge t^2)$. At $d=1000$, $t=0.1$: bound $=2e^{-999\cdot0.005}=2e^{-4.995}=1.36\times10^{-2}$; exact $=1.36\times10^{-3}$ (from the Beta survival function). A factor of ten, uniformly good enough.
>
> The geometric reading: the band $\{|u_1|\le0.1\}$, which looks like a thin slice around the equator, contains $99.86\%$ of the sphere's surface at $d=1000$. Every "equator" of a high-dimensional sphere is almost the whole sphere. Combine this over many coordinates and you get Worked Example 12.4 again.

**Theorem 12.12 (Talagrand's convex distance inequality).** Let $X=(X_1,\dots,X_d)$ have independent coordinates with $X_i\in[0,1]$, and let $f:[0,1]^d\to\mathbb{R}$ be convex and $L$-Lipschitz. Then
$$\Pr\big(|f(X)-\mathrm{med}(f(X))|\ge t\big)\le 4\exp\Big(-\frac{t^2}{4L^2}\Big).$$
Again dimension-free, and again the hypotheses are doing real work. **Convexity cannot be dropped.** Here is a concrete counterexample. Let $X$ be uniform on $\{0,1\}^d$ (independent, bounded coordinates) with $d$ even, let $A=\{x\in\{0,1\}^d:\sum_ix_i=d/2\}$, and let $f(x)=\mathrm{dist}_2(x,A)$. A distance-to-a-set function is always $1$-Lipschitz, and here $f(x)=\sqrt{|\sum_ix_i-d/2|}$, because moving the coordinate sum by $m$ requires flipping $m$ coordinates, at Euclidean cost $\sqrt m$. But $A$ is not convex, and $f$ is not convex. Since $\sum_iX_i\sim\mathrm{Bin}(d,\tfrac12)$ has fluctuations of order $\sqrt d$, we get $f(X)\asymp d^{1/4}$ — mean *and* standard deviation both growing without bound. Measured, `default_rng(77)`, $2\times10^5$ draws:
>
| $d$ | $10^2$ | $10^3$ | $10^4$ | $10^5$ |
|---|---|---|---|---|
| $\widehat{\mathbb{E}}f$ | 1.807 | 3.264 | 5.815 | 10.323 |
| $\widehat{\mathrm{sd}}\,f$ | 0.847 | 1.409 | 2.472 | 4.391 |

Each tenfold increase in $d$ multiplies the standard deviation by about $1.77\approx10^{1/4}$. So a $1$-Lipschitz function of independent bounded variables can have unbounded fluctuations; Theorem 12.7's dimension-freeness is a *Gaussian* phenomenon, and Theorem 12.12 recovers it for bounded variables only at the price of convexity. Independence likewise cannot be dropped. Talagrand's inequality is the workhorse for random matrix norms with non-Gaussian entries and for empirical process bounds where Gaussianity is unavailable.

While we are keeping hypotheses straight, one more contrast worth internalizing, because students mix these up constantly: McDiarmid's bounded-differences inequality (Ch. 4) also gives dimension-free-looking concentration for functions of independent bounded variables — but its Lipschitz condition is with respect to the *Hamming*-like metric (change one coordinate, the function moves by at most $c_i$), and its deviation scale is $\sqrt{\sum_ic_i^2}$, which for a function that is $1$-Lipschitz per coordinate is $\sqrt d$, not $1$. So "Lipschitz" without naming the metric is not a hypothesis. Euclidean-Lipschitz plus Gaussian gives scale $L$ (Theorem 12.7); Euclidean-Lipschitz plus bounded plus *convex* gives scale $L$ (Theorem 12.12); per-coordinate-Lipschitz plus bounded gives scale $L\sqrt d$ (McDiarmid), and the counterexample above shows the $\sqrt d$ cannot in general be removed without convexity.

> **Common Confusion 12.2.**
> *"Concentration means the variance goes to zero as $d$ grows, right? So high-dimensional randomness is basically deterministic?"*
>
> No, and the distinction has bitten many people. For $Z\sim\mathcal{N}(0,I_d)$: $\operatorname{Var}(\|Z\|)\to\tfrac12$, which is a *constant*, not zero. What goes to zero is the *relative* fluctuation $\mathrm{sd}(\|Z\|)/\mathbb{E}\|Z\|\approx 1/\sqrt{2d}$.
>
> So concentration is a statement about scale, not about determinism. A $1$-Lipschitz function of $Z\sim\mathcal{N}(0,I_{10^6})$ still has standard deviation up to $1$ — a perfectly ordinary amount of randomness. What is remarkable is that it is not $1000$.
>
> Where this matters in practice: minibatch gradient *norms* concentrate tightly around their mean (relative fluctuation small), but minibatch gradient *directions* do not concentrate at all in the sense you might want — the cosine between two independent minibatch gradients can be small even when the norms are nearly identical, because Proposition 12.2 says random directions are near-orthogonal and the "signal" component may be a small fraction of the total. A stable gradient norm is not evidence of a stable gradient.

> **Common Confusion 12.3.**
> *"If Lipschitz functions of Gaussians concentrate independently of dimension, then a neural network's output must concentrate too — so networks can't be sensitive to their inputs, and adversarial examples shouldn't exist."*
>
> The Lipschitz constant is doing all the work, and for real networks it is enormous. A network's Lipschitz constant is bounded above by the product of its layers' spectral norms; for a trained ResNet this product is routinely $10^{10}$ or more. Equation (12.3) then says the output has standard deviation at most $10^{10}$ — a true statement with no content.
>
> Two conclusions. First, dimension-free bounds are only useful when $L$ is genuinely small, which is why spectral normalization, Lipschitz-constrained architectures, and gradient penalties are studied: they buy back the constant. Second, adversarial examples are entirely compatible with concentration — concentration says a *random* perturbation barely changes the output (which is true, and is why networks are robust to Gaussian noise), while an adversarial perturbation is chosen *after* seeing the function, i.e. it exploits the worst direction rather than a typical one. In $\mathbb{R}^d$ there are exponentially many nearly-orthogonal directions (Worked Example 12.4), so "typical direction" and "worst direction" are very different objects. The gap between them is roughly $\sqrt d$: a random unit perturbation aligns with the gradient by $\approx1/\sqrt d$, while the adversarial one aligns by $1$.

> **Self-check 12.2.**
> 1. State the Lipschitz constant of $x\mapsto\frac1d\sum_i x_i$ on $\mathbb{R}^d$ (Euclidean metric) and use (12.3) to bound the deviation of the coordinate mean of $Z\sim\mathcal{N}(0,I_d)$. Compare with what you know exactly.
> 2. Why does $d$ appear in Lévy's lemma (Corollary 12.11) but not in Theorem 12.7? Are these in conflict?
> 3. Give an example of a $1$-Lipschitz function of $Z\sim\mathcal{N}(0,I_d)$ whose *mean* grows with $d$ — so that concentration around the mean says nothing about concentration around a fixed value.
> 4. Talagrand's Theorem 12.12 requires convexity. Where in the argument would you expect convexity to matter?

## 12.3 Distance concentration and the failure of nearest neighbours

Now we cash in the concentration machinery for a result that sounds like bad news, then find out why it is not.

### 12.3.1 The theorem: contrast dies

Nearest-neighbour search, kernel methods, and vector retrieval all rest on a single premise: that the distance from a query to a relevant item is *meaningfully smaller* than the distance to an irrelevant one. Define, for a query $q$ and a reference set $x_1,\dots,x_n$, the **relative contrast**
$$C_n(q) := \frac{D_{\max}(q)-D_{\min}(q)}{D_{\min}(q)},\qquad D_{\max}(q)=\max_i\|q-x_i\|,\ D_{\min}(q)=\min_i\|q-x_i\|.$$
If $C_n(q)$ is large, "nearest" is a strong statement. If $C_n(q)\approx0$, the nearest neighbour is barely nearer than the farthest, and any measurement noise, quantization, or approximation in the index destroys the ranking.

**Theorem 12.13 (distance concentration for i.i.d. coordinates).** Let $X^{(1)},\dots,X^{(n)},Q$ be i.i.d. random vectors in $\mathbb{R}^d$ whose coordinates are i.i.d., mean $\mu$, variance $\sigma^2>0$, and sub-Gaussian with $\psi_2$-norm at most $K$. Then there is an absolute constant $c>0$ such that, with probability at least $1-\delta$,
$$\max_{1\le i\le n}\left|\frac{\|Q-X^{(i)}\|^2}{2\sigma^2 d}-1\right|\ \le\ c\,\frac{K^2}{\sigma^2}\sqrt{\frac{\log(n/\delta)}{d}} \ =:\ \eta . \tag{12.4}$$
Consequently, on that event,
$$C_n(Q)\ \le\ \frac{\sqrt{1+\eta}}{\sqrt{1-\eta}}-1\ =\ \eta+O(\eta^2),$$
so $C_n(Q)\to0$ in probability whenever $\log n/d\to0$.

*Proof.* Fix $i$ and write $Y := Q - X^{(i)}$. Its coordinates $Y_j = Q_j - X^{(i)}_j$ are independent (across $j$), mean $0$, variance $2\sigma^2$, and sub-Gaussian with $\|Y_j\|_{\psi_2}\le 2K$ (sub-Gaussianity is preserved under sums of independent sub-Gaussians, Ch. 4). Apply Theorem 12.6 to $Y/(\sigma\sqrt2)$, whose coordinates are standardized and sub-Gaussian with constant $\lesssim K/\sigma$: for each $u\ge0$,
$$\Pr\left(\left|\frac{\|Y\|}{\sigma\sqrt{2d}}-1\right|\ge \frac{u}{\sqrt d}\right)\le 2\exp\big(-c'u^2\sigma^4/K^4\big).$$
Set the right-hand side to $\delta/n$ and solve: $u = \sqrt{\log(2n/\delta)}\cdot K^2/(\sigma^2\sqrt{c'})$. By the union bound (Proposition 1.8(2)) over the $n$ indices, with probability $\ge1-\delta$ every $i$ satisfies $\big|\|Y^{(i)}\|/(\sigma\sqrt{2d})-1\big|\le u/\sqrt d$. Squaring and absorbing constants gives (12.4).

For the consequence: on that event $D_{\min}\ge\sigma\sqrt{2d}\sqrt{1-\eta}$ and $D_{\max}\le\sigma\sqrt{2d}\sqrt{1+\eta}$, so $D_{\max}/D_{\min}\le\sqrt{(1+\eta)/(1-\eta)}$. $\square$

Read (12.4) as a competition between $\log n$ and $d$, mediated by how much contrast you demand. Rearranging $\eta\le\varepsilon$ gives the usable form: **for isotropic data, contrast of size $\varepsilon$ survives only while $d\lesssim c^2\log(n/\delta)/\varepsilon^2$.** Two calibrations, using the constant $c\approx2.7$ measured in Worked Example 12.10. Demanding order-one contrast ($\varepsilon=1$) with $n=10^6$ allows $d$ up to about $7.3\cdot13.8\approx100$. Demanding only $\varepsilon=0.1$ allows $d$ up to about $10^4$. So the popular slogan "nearest neighbours are meaningless above a few dozen dimensions" corresponds to insisting on *strong* contrast; if you only need the nearest neighbour to be, say, $10\%$ closer than the farthest, isotropic data survives to surprisingly large $d$. The honest summary is the scaling $d\lesssim\log n/\varepsilon^2$, not a single threshold.

It is worth stating the converse, since otherwise the "competition" framing is only half-established. Theorem 12.13 gives an *upper* bound on the contrast, so it can only show contrast dying. That contrast genuinely *survives* in the complementary regime is immediate from a one-dimensional example: for $d=1$ and $X_i$ standard normal, $D_{\min}$ over $n$ points is of order $1/n$ while $D_{\max}$ is of order $\sqrt{\log n}$, so $C_n\to\infty$. More generally, for fixed $d$ and growing $n$ the minimum interpoint distance scales like $n^{-2/d}$ while the maximum stays bounded in probability, so $C_n\to\infty$ for every fixed $d$. The two statements together say that contrast is a function of $d/\log n$, and (12.4) locates the transition.

One piece of folklore to defuse before the worked examples, because you will meet it in blog posts and occasionally in papers: *"use $\ell_1$ or fractional $\ell_p$ distances to beat the curse of dimensionality."* The source is a real observation (Aggarwal, Hinneburg and Keim): for i.i.d. coordinates, the *ratio* $(D_{\max}-D_{\min})/D_{\min}$ decays more slowly for smaller $p$, essentially because $\|x\|_p^p$ is a sum whose mean grows like $d$ while its fluctuations grow like $\sqrt d$ regardless of $p$, and the $p$-th root maps this to relative contrast $\Theta(\sqrt{1/d})$ with a $p$-dependent constant. So the choice of $p$ moves the constant, not the rate: contrast still dies like $\sqrt{\log n/d}$ for isotropic data under every $\ell_p$. It cannot substitute for the genuine rescue of §12.3.2, which is that real data has low intrinsic dimension. Choose the metric to match the semantics of your features; do not choose it hoping to repeal Theorem 12.13.

> **Worked Example 12.10 (contrast for isotropic Gaussian data).**
> $n=1000$ reference points drawn i.i.d. $\mathcal{N}(0,I_d)$, `default_rng(7)`. Pairwise distances via `scipy.spatial.distance.pdist`; the last column compares the observed relative contrast to the $\sqrt{\log n/d}$ rate predicted by (12.4).
>
> | $d$ | $D_{\min}$ | $D_{\max}$ | $D_{\max}/D_{\min}$ | $(D_{\max}-D_{\min})/D_{\min}$ | $\sqrt{\log n/d}$ | ratio |
> |---|---|---|---|---|---|---|
> | 2 | 0.0034 | 6.2674 | 1865.47 | 1864.5 | — | — |
> | 10 | 0.8544 | 9.5134 | 11.13 | 10.13 | — | — |
> | 100 | 9.7966 | 18.9884 | 1.9383 | 0.9326 | 0.2628 | 3.55 |
> | 1000 | 40.2025 | 49.5355 | 1.2321 | 0.2321 | 0.0831 | 2.79 |
> | 10000 | 136.9701 | 145.7994 | 1.0645 | 0.0698 | 0.0263 | 2.66 |
>
> The predicted rate is right, with a constant settling near $2.7$. The interpretation of the $d=10000$ row: among a thousand points, the closest pair is $137.0$ apart and the most distant pair is $145.8$ apart — a spread of $6\%$. If your index quantizes distances to $8$ bits over a range that includes zero, you have thrown away the entire signal.
>
> A per-query version, `default_rng(7)`, $100$ queries against $n=1000$ references, mean of $D_{\max}/D_{\min}$: $d=2$: $109.70$; $d=10$: $4.19$; $d=100$: $1.492$; $d=1000$: $1.134$; $d=10000$: $1.041$.

### 12.3.2 The rescue: intrinsic dimension, not ambient dimension

If Theorem 12.13 were the last word, the entire retrieval industry would not work. Text embeddings live in $\mathbb{R}^{768}$ or $\mathbb{R}^{1536}$; image embeddings in $\mathbb{R}^{512}$ or larger; approximate nearest-neighbour libraries (FAISS, HNSW, ScaNN) return useful neighbours at those dimensions every day, at billion-scale. The premise of Theorem 12.13 is what fails: **coordinates of real embeddings are not independent, and the data does not fill its ambient space.**

Recall from Chapter 1 the two-part story about supports. Definition 1.17 says $\mathrm{supp}(P)$ is the smallest closed set of full measure, and Chapter 1's circle example — the generator $G(t)=(\cos t,\sin t)$, whose pushforward $G_\#P$ lives on a $1$-dimensional curve in $\mathbb{R}^2$ and hence has Lebesgue measure zero — is exactly the picture to hold. A deep generative model $P_\theta = (G_\theta)_\#\mathcal{N}(0,I_k)$ with $k<d$ has support of Lebesgue measure zero in $\mathbb{R}^d$, so it has no Lebesgue density at all.

The *manifold hypothesis* asserts that natural data behaves this way: images, audio, and text embeddings concentrate near a low-dimensional subset of their ambient space. Two versions must be distinguished, because they have different mathematical status and different consequences.

**Exactly low-dimensional support.** $P$ is supported on a $k$-dimensional embedded submanifold $M\subseteq\mathbb{R}^d$ with $k\ll d$, so $P(M)=1$ and $\lambda_d(M)=0$. This is the clean case, and it is what a generator with a $k$-dimensional latent produces by construction. It is almost certainly *false* for real data: pixel noise, JPEG artefacts, and sensor variation give full-dimensional support.

**Approximately low-dimensional mass.** There is a $k$-dimensional set $M$ and a small $\epsilon$ with $P(\{x:\mathrm{dist}(x,M)>\epsilon\})$ tiny. Here $\mathrm{supp}(P)$ may be all of $\mathbb{R}^d$ — remember Chapter 1's caution that a set of full measure need not contain the support, and that $\mathrm{supp}(\mathcal{N}(0,1))=\mathbb{R}$ despite the mass being concentrated near $0$. The support is about where mass *isn't*; here mass *is* everywhere, just extremely unevenly. This is the realistic case, and it is the one for which we need a quantitative notion.

**Definition 12.14 (intrinsic dimension via small-ball probability).** Let $P$ be a probability measure on $\mathbb{R}^d$. Say $P$ has *local intrinsic dimension $k$ at scale range $[r_0,r_1]$* if there are constants $0<c\le C<\infty$ with
$$c\,\big(r/r_1\big)^k\ \le\ P\big(B(x,r)\big)\ \le\ C\,\big(r/r_1\big)^k\qquad\text{for }P\text{-almost every }x\text{ and all }r\in[r_0,r_1].$$
Equivalently, $\log P(B(x,r))/\log r \approx k$ over the stated scales. This is the *correlation dimension* / doubling-dimension notion, and it is estimable from data: the two-nearest-neighbour (TwoNN) and MLE estimators of Levina–Bickel are built directly on this definition.

The point of Definition 12.14 is that it never mentions $d$. It measures how fast the mass in a ball grows with the radius, which is a property of the measure, not of the coordinate system you happened to write it in.

The definition is not decoration; it is what makes Theorem 12.13 quantitative for non-isotropic data. Here is the link, in one paragraph. Suppose $P$ has local intrinsic dimension $k$ at scales $[r_0,r_1]$. Fix a query $q$ and let $R_{\min}$ be the distance to the nearest of $n$ i.i.d. samples. Then $\Pr(R_{\min}>r)=(1-P(B(q,r)))^n\le\exp(-n\,c(r/r_1)^k)$, so $R_{\min}\lesssim r_1(\log n/(cn))^{1/k}$: the nearest-neighbour distance shrinks like $n^{-1/k}$, with the *intrinsic* $k$ in the exponent, while $D_{\max}$ stays of order the diameter. Hence the contrast $D_{\max}/D_{\min}$ *grows* like $n^{1/k}$ for any fixed $k$ — the opposite of Theorem 12.13's collapse. Isotropic Gaussian data in $\mathbb{R}^d$ has $k=d$, so $n^{1/k}=n^{1/d}\to1$ and we recover the collapse; data with $k=5$ has $n^{1/5}$, which for $n=1000$ is $3.98$ — the right order of magnitude for the plateau we are about to measure. This is the whole mechanism, and $d$ never appears in it.

**Proposition 12.15 (contrast is governed by intrinsic, not ambient, dimension).** Let $\Psi:\mathbb{R}^k\to\mathbb{R}^d$ be bi-Lipschitz with constants $0<\alpha\le\beta<\infty$, i.e. $\alpha\|z-z'\|\le\|\Psi(z)-\Psi(z')\|\le\beta\|z-z'\|$. Let $Z,Z^{(1)},\dots,Z^{(n)}$ be i.i.d. in $\mathbb{R}^k$ and set $X^{(i)}=\Psi(Z^{(i)})$, $Q=\Psi(Z)$. Then
$$\frac\alpha\beta\cdot\frac{\tilde D_{\max}}{\tilde D_{\min}}\ \le\ \frac{D_{\max}(Q)}{D_{\min}(Q)}\ \le\ \frac\beta\alpha\cdot\frac{\tilde D_{\max}}{\tilde D_{\min}},$$
where $\tilde D_{\max},\tilde D_{\min}$ are the extreme distances in the *latent* space $\mathbb{R}^k$. Hence the contrast in $\mathbb{R}^d$ is the latent contrast up to the fixed factor $\beta/\alpha$, and in particular it does not decay as $d$ grows with $k$ and $\Psi$'s distortion held fixed.

*Proof.* By the bi-Lipschitz property, $\alpha\tilde D_{\min}\le D_{\min}\le\beta\tilde D_{\min}$ and $\alpha\tilde D_{\max}\le D_{\max}\le\beta\tilde D_{\max}$. Divide. $\square$

This is nearly trivial as mathematics — three lines — and enormously important as an idea, so it is worth being precise about what the mathematics does and does not deliver.

*What Proposition 12.15 proves:* that ambient dimension per se is irrelevant. A bi-Lipschitz embedding into a huge ambient space raises $d$ without changing contrast beyond the fixed factor $\beta/\alpha$. Combined with the small-ball argument after Definition 12.14, this says contrast is governed by intrinsic dimension $k$ and distortion $\beta/\alpha$, and by nothing else. Theorem 12.13 therefore does not say "high dimension kills nearest neighbours"; it says "high *effective* dimension kills nearest neighbours," and the isotropic case is simply the special case $k=d$, $\beta/\alpha=1$.

*What it does not prove* — and this is the part usually elided in papers — is that real data actually has small $k$ and small $\beta/\alpha$. That is an **empirical** claim, the manifold hypothesis, and it is supported by intrinsic-dimension estimates (tens for ImageNet, per Pope et al.) rather than by any theorem. I will mark it as such, using the same labels as §12.6: *Proposition 12.15 is a theorem; "real data has low intrinsic dimension" is an empirical observation.* The experiments below therefore do two different jobs. Worked Example 12.11 verifies the *theorem* on synthetic data where $k$ and $\beta/\alpha$ are known and controlled. Whether your dataset resembles that synthetic construction is something you must measure, and Common Confusion 12.4 says how.

> **Worked Example 12.11 (the contrast that refuses to die).**
> Two datasets, both with $n=1000$ reference points, $100$ queries, `default_rng(11)`. Left column: isotropic $\mathcal{N}(0,I_d)$. Right column: a latent $Z\sim\mathcal{N}(0,I_5)$ mapped by $\Psi(z)=\sqrt2\cos(W^\top z+b)$ with $W\in\mathbb{R}^{5\times d}$ having i.i.d. $\mathcal{N}(0,1/5)$ entries and $b$ uniform on $[0,2\pi)^d$ — a smooth, injective-in-practice map into $\mathbb{R}^d$ whose image is a $5$-dimensional manifold. Reported: mean over queries of $D_{\max}/D_{\min}$.
>
> | $d$ | isotropic in $\mathbb{R}^d$ | $5$-dim manifold in $\mathbb{R}^d$ |
> |---|---|---|
> | 10 | 4.018 | 8.975 |
> | 30 | 2.136 | 5.688 |
> | 100 | 1.498 | 5.936 |
> | 300 | 1.258 | 5.391 |
> | 1000 | 1.136 | 5.150 |
> | 3000 | 1.076 | 5.188 |
> | 10000 | 1.041 | 5.132 |
>
> The isotropic column collapses toward $1$. The manifold column *plateaus near $5.1$* and stays there over three orders of magnitude of ambient dimension. Exactly Proposition 12.15: once $d$ exceeds the amount needed to embed the manifold with bounded distortion, adding dimensions changes nothing.
>
> **Where does $5.1$ come from?** A plateau at an unexplained number is not an explanation, so let us check it against Proposition 12.15 quantitatively. Two measurements, both with `default_rng(11)`.
>
> (i) *The latent contrast.* Running the same contrast measurement directly on $z\sim\mathcal{N}(0,I_5)$ in $\mathbb{R}^5$ — no embedding at all — gives $\tilde D_{\max}/\tilde D_{\min}=9.54,\,10.04,\,9.52$ over three replications, so $\tilde C\approx9.7$. (Reassuringly, this is close to the $n^{1/k}=1000^{1/5}=3.98$ scaling from the small-ball argument, up to the constant relating $D_{\max}$ to the diameter.)
>
> (ii) *The distortion.* Estimating $\beta/\alpha$ for $\Psi$ as the ratio of the largest to smallest value of $\|\Psi(z)-\Psi(z')\|/\|z-z'\|$ over all pairs from $600$ latent draws gives $3.68$ at $d=100$, $2.74$ at $d=1000$, and $2.70$ at $d=10^4$. Note that this *stops growing* — which is precisely the hypothesis Proposition 12.15 needs, and it is the reason the plateau exists.
>
> Proposition 12.15 then predicts the embedded contrast lies in $[\tilde C\alpha/\beta,\ \tilde C\beta/\alpha]=[9.7/2.70,\ 9.7\times2.70]=[3.6,\ 26.2]$. The observed $5.13$ sits comfortably inside, near the lower end. So the number is not magic: it is the five-dimensional contrast, degraded by the embedding's distortion.
>
> **One caveat.** $\Psi(z)=\sqrt2\cos(W^\top z+b)$ is not *proved* bi-Lipschitz — $\cos$ is not globally injective, so in principle two distant latents could collide. Measurement (ii) is what licenses the use of Proposition 12.15 here: it exhibits finite $\beta/\alpha$ on the sampled pairs, bounded uniformly in $d$. If you build your own example, measure the distortion; do not assume it.

> **Worked Example 12.12 (and the contrast translates into accuracy).**
> Contrast is a proxy; classification error is the thing we care about. Same construction, `default_rng(303)`, $n_{\text{train}}=2000$, $n_{\text{test}}=1000$, $1$-nearest-neighbour classifier with Euclidean distance. In *both* settings the label depends on exactly five informative variables: $y=\mathbb{1}\{\sin(2v_1)+v_2v_3-v_4+0.5v_5>0\}$, where $v$ is the first five ambient coordinates (isotropic case) or the five latent coordinates (manifold case).
>
> | $d$ | isotropic: $d$ i.i.d. coords, label from first $5$ | manifold: $k=5$ latent embedded in $\mathbb{R}^d$ |
> |---|---|---|
> | 5 | 0.1240 | 0.2710 |
> | 10 | 0.2360 | 0.1840 |
> | 100 | 0.4430 | 0.1190 |
> | 1000 | 0.4820 | 0.1270 |
> | 5000 | 0.4770 | 0.1180 |
>
> The isotropic classifier degrades to chance ($0.5$) — the $995$ irrelevant coordinates contribute pure noise to the distance and swamp the five that matter. The manifold classifier holds at $\approx0.12$ across three orders of magnitude of ambient dimension. (The $d=5$ manifold entry is worse because at $d=k$ the random cosine embedding is poorly conditioned — $\beta/\alpha$ is large — not because of dimension; by $d=100$ the embedding is well conditioned and the error settles.)
>
> **This contrast is the most important computed result in the chapter.** It is the reason a sentence-embedding retrieval system at $d=1536$ works, and simultaneously the reason that appending a thousand columns of measurement noise to your feature matrix will destroy a $k$-NN model. Both are the same theorem; which one you experience depends entirely on whether your added dimensions carry structure.

<svg viewBox="0 0 760 290" style="max-width:100%;height:auto" role="img" aria-label="Nearest-neighbour contrast versus ambient dimension: isotropic data collapses to one, manifold data plateaus.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.4 — Contrast dies for isotropic data, survives on a manifold</text>
  <line x1="90" y1="215" x2="680" y2="215" stroke="currentColor" stroke-width="1"/>
  <line x1="90" y1="215" x2="90" y2="55" stroke="currentColor" stroke-width="1"/>
  <text x="90" y="230" text-anchor="middle" font-size="10" fill="currentColor">10</text>
  <text x="273" y="230" text-anchor="middle" font-size="10" fill="currentColor">100</text>
  <text x="457" y="230" text-anchor="middle" font-size="10" fill="currentColor">1000</text>
  <text x="640" y="230" text-anchor="middle" font-size="10" fill="currentColor">10000</text>
  <text x="385" y="245" text-anchor="middle" font-size="11" fill="currentColor">ambient dimension d (log scale)</text>
  <text x="84" y="219" text-anchor="end" font-size="10" fill="currentColor">1</text>
  <text x="84" y="202" text-anchor="end" font-size="10" fill="currentColor">2</text>
  <text x="84" y="167" text-anchor="end" font-size="10" fill="currentColor">4</text>
  <text x="84" y="133" text-anchor="end" font-size="10" fill="currentColor">6</text>
  <text x="84" y="98" text-anchor="end" font-size="10" fill="currentColor">8</text>
  <text x="84" y="64" text-anchor="end" font-size="10" fill="currentColor">10</text>
  <text x="40" y="140" font-size="11" fill="currentColor" transform="rotate(-90 40 140)">Dmax / Dmin</text>
  <line x1="90" y1="215" x2="680" y2="215" stroke="currentColor" stroke-width="0.7" stroke-dasharray="3,3"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2.4" points="90,163 177,195 273,206 361,211 457,213 544,214 640,214"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2.4" points="90,78 177,134 273,130 361,139 457,144 544,143 640,144"/>
  <circle cx="90" cy="163" r="3" fill="#ef4444"/><circle cx="273" cy="206" r="3" fill="#ef4444"/><circle cx="457" cy="213" r="3" fill="#ef4444"/><circle cx="640" cy="214" r="3" fill="#ef4444"/>
  <circle cx="90" cy="78" r="3" fill="#10b981"/><circle cx="273" cy="130" r="3" fill="#10b981"/><circle cx="457" cy="144" r="3" fill="#10b981"/><circle cx="640" cy="144" r="3" fill="#10b981"/>
  <text x="480" y="205" font-size="11" fill="#ef4444">isotropic N(0, I_d): collapses to 1</text>
  <text x="470" y="136" font-size="11" fill="#10b981">5-dim manifold in R^d: plateaus at 5.1</text>
  <text x="380" y="264" text-anchor="middle" font-size="11" fill="currentColor">n = 1000 reference points, 100 queries, seed 11. Same ambient dimension on both curves. Nearest-neighbour</text>
  <text x="380" y="278" text-anchor="middle" font-size="11" fill="currentColor">search degrades with INTRINSIC dimension, not ambient dimension. This is why vector retrieval works.</text>
</svg>

Without the figure: the red curve (isotropic data) starts at contrast $4.0$ at $d=10$ and falls monotonically to $1.04$ at $d=10^4$; the green curve (five-dimensional manifold embedded in the same $\mathbb{R}^d$) starts at $9.0$, dips, and then stays flat at $\approx5.1$ from $d=100$ through $d=10^4$.

> **Common Confusion 12.4.**
> *"So the curse of dimensionality is a myth? Real data is low-dimensional, so I can ignore all this?"*
>
> Neither extreme is right, and the useful skill is diagnosing which regime you are in. Three concrete symptoms that you are in the *bad* regime, where dimension really is hurting you:
> (i) you appended features that are near-independent of the label — one-hot IDs, noisy sensor channels, random hashes — and $k$-NN or kernel accuracy dropped, matching the left column of Worked Example 12.12;
> (ii) your measured intrinsic dimension (TwoNN, MLE, or the participation ratio of the covariance spectrum) is close to $d$;
> (iii) the histogram of pairwise distances in your dataset is narrow and unimodal, with $(D_{\max}-D_{\min})/D_{\text{median}}$ under, say, $0.2$.
> Symptom (iii) takes two lines of code and is the single most useful diagnostic in this chapter. Run it before you tune your ANN index; if contrast is dead, no amount of index engineering will help, and you need a better representation instead.
>
> Conversely, "real data is low-dimensional" is an empirical claim about particular datasets, not a theorem. Estimated intrinsic dimensions for ImageNet run in the tens (Pope et al. report roughly $26$–$43$ depending on class and estimator), which is comfortably low relative to $150528$ but not $2$; and adversarial examples are precisely a demonstration that the *off-manifold* directions are not inert.

> **Self-check 12.3.**
> 1. Using (12.4), estimate how large $d$ can be before contrast falls below $10\%$ for $n=10^6$ isotropic points, taking the constant to be $2.7$ as measured.
> 2. A colleague normalizes all embeddings to unit norm and reports that distances now "concentrate less". What is actually happening to $D_{\max}/D_{\min}$, and why is cosine similarity the more informative quantity on the sphere?
> 3. Distinguish exactly-low-dimensional support from approximately-low-dimensional mass, and say which one Chapter 1's circle generator produces.
> 4. Why does Proposition 12.15 involve the ratio $\beta/\alpha$ rather than $\alpha$ and $\beta$ separately?

## 12.4 Johnson–Lindenstrauss: compression for free

Section 12.3 was concentration as an obstacle. This section is concentration as a tool, and it produces one of the most useful theorems in applied mathematics.

The question: given $n$ points in $\mathbb{R}^d$ with $d$ huge, can we map them into $\mathbb{R}^k$ with $k$ small while preserving all $\binom n2$ pairwise distances to within a factor $1\pm\varepsilon$? The answer is yes, with $k$ depending on $n$ and $\varepsilon$ *but not on $d$*, and the map is a random linear projection chosen without looking at the data.

### 12.4.1 One vector at a time

**Lemma 12.16 (distributional Johnson–Lindenstrauss).** Let $R\in\mathbb{R}^{k\times d}$ have i.i.d. $\mathcal{N}(0,1/k)$ entries, and fix $x\in\mathbb{R}^d$, $x\neq0$. Then for every $\varepsilon\in(0,1)$,
$$\Pr\Big(\big|\,\|Rx\|^2 - \|x\|^2\,\big| \ge \varepsilon\|x\|^2\Big)\ \le\ 2\exp\Big(-\frac{k\varepsilon^2}{8}\Big). \tag{12.5}$$

*Proof.* By homogeneity assume $\|x\|=1$. Row $j$ of $R$ is $r_j\sim\mathcal{N}(0,I_d/k)$, so $\langle r_j,x\rangle\sim\mathcal{N}(0,\|x\|^2/k)=\mathcal{N}(0,1/k)$ (this is where the Gaussian's stability under linear maps, Ch. 5, enters). The rows are independent, so
$$\|Rx\|^2=\sum_{j=1}^k\langle r_j,x\rangle^2 = \frac1k\sum_{j=1}^k G_j^2 = \frac{W}{k},\qquad G_j\stackrel{iid}\sim\mathcal{N}(0,1),\quad W\sim\chi^2_k .$$
So $\mathbb{E}\|Rx\|^2=1$ (the projection is unbiased — this is why we scaled by $1/\sqrt k$), and the event in (12.5) is $\{|W-k|\ge\varepsilon k\}$.

Apply Lemma 12.4 with $d\to k$. For the upper tail, take $t=k\varepsilon^2/8$; then $2\sqrt{kt}+2t = \tfrac{k\varepsilon}{\sqrt2}+\tfrac{k\varepsilon^2}{4}\le k\varepsilon\big(\tfrac1{\sqrt2}+\tfrac14\big) = 0.957\,k\varepsilon \le k\varepsilon$ for $\varepsilon\le1$, so $\{W\ge k+k\varepsilon\}\subseteq\{W\ge k+2\sqrt{kt}+2t\}$ and $\Pr(W\ge k(1+\varepsilon))\le e^{-k\varepsilon^2/8}$. For the lower tail with the same $t$: $2\sqrt{kt}=k\varepsilon/\sqrt2\le k\varepsilon$, so $\{W\le k-k\varepsilon\}\subseteq\{W\le k-2\sqrt{kt}\}$ and $\Pr(W\le k(1-\varepsilon))\le e^{-k\varepsilon^2/8}$. Adding gives (12.5). $\square$

Notice the two ingredients, because Part C asks you to name them: the *Gaussian rotational invariance* that turns $\|Rx\|^2$ into a scaled $\chi^2_k$ regardless of what $x$ is (this is why the bound has no $d$), and the *sub-Gaussian/sub-exponential concentration* of $\chi^2_k$ that makes the failure probability exponentially small in $k$.

### 12.4.2 All pairs at once

**Theorem 12.17 (Johnson–Lindenstrauss lemma).** Let $x_1,\dots,x_n\in\mathbb{R}^d$, let $\varepsilon\in(0,1)$ and $\delta\in(0,1)$, and let
$$k\ \ge\ \frac{8}{\varepsilon^{2}}\Big(2\log n+\log\tfrac1\delta\Big).$$
Let $R\in\mathbb{R}^{k\times d}$ have i.i.d. $\mathcal{N}(0,1/k)$ entries. Then with probability at least $1-\delta$,
$$(1-\varepsilon)\|x_i-x_j\|^2\ \le\ \|Rx_i-Rx_j\|^2\ \le\ (1+\varepsilon)\|x_i-x_j\|^2\qquad\text{for all }1\le i<j\le n. \tag{12.6}$$
In particular $k=O(\varepsilon^{-2}\log n)$ suffices for constant $\delta$, and $k$ does not depend on $d$.

*Proof.* There are $N=\binom n2 < n^2/2$ pairs. For each pair, apply Lemma 12.16 to the vector $x=x_i-x_j$ (if $x_i=x_j$ the claim is trivial), noting $R(x_i-x_j)=Rx_i-Rx_j$ by linearity. Each pair fails with probability at most $2e^{-k\varepsilon^2/8}$. By the union bound (Proposition 1.8(2)),
$$\Pr(\text{some pair fails})\ \le\ \binom n2\cdot 2e^{-k\varepsilon^2/8}\ \le\ n^2 e^{-k\varepsilon^2/8}.$$
This is at most $\delta$ as soon as $k\varepsilon^2/8\ge 2\log n+\log(1/\delta)$, i.e. $k\ge 8\varepsilon^{-2}(2\log n+\log(1/\delta)) = 16\varepsilon^{-2}\log(n/\sqrt\delta)$. $\square$

That is the whole proof. It is four lines and it is one of the most consequential four lines in the subject. Let us be precise about where each piece of the bound comes from, since this is exactly the kind of accounting Part C will ask for.

- The $\varepsilon^{-2}$ comes from **sub-Gaussian concentration**: the failure probability for one vector is $e^{-\Theta(k\varepsilon^2)}$, so to make it small you need $k\varepsilon^2$ large. The exponent $2$ on $\varepsilon$ is the signature of a second-moment (variance-type) concentration bound, exactly as in Hoeffding (Ch. 4).
- The $\log n$ comes from the **union bound** over $\binom n2$ pairs. This is the same mechanism that produces $\log|\mathcal{H}|$ in finite-hypothesis-class generalization bounds (Ch. 4) — one of Chapter 1's themes, that the union bound is the source of logarithmic dependence on the size of a collection.
- The absence of $d$ comes from **rotational invariance**: Lemma 12.16's bound depends on $x$ only through $\|x\|$.

**Corollary 12.18 (inner products are preserved too).** Under the hypotheses of Theorem 12.17 applied to the $2n$ vectors $\{x_i\}\cup\{x_i/\|x_i\|\}$ — or more simply, applied to all sums and differences $x_i\pm x_j$ — we get, with probability $\ge1-\delta$,
$$\big|\langle Rx_i,Rx_j\rangle - \langle x_i,x_j\rangle\big|\ \le\ \varepsilon\|x_i\|\|x_j\|\qquad\text{for all }i,j .$$
*Proof.* Polarization: $\langle u,v\rangle = \tfrac14(\|u+v\|^2-\|u-v\|^2)$. Applying (12.6) to $x_i+x_j$ and $x_i-x_j$ gives
$$\big|\langle Rx_i,Rx_j\rangle-\langle x_i,x_j\rangle\big|\le\tfrac\varepsilon4\big(\|x_i+x_j\|^2+\|x_i-x_j\|^2\big)=\tfrac\varepsilon2\big(\|x_i\|^2+\|x_j\|^2\big),$$
using the parallelogram law; rescaling to unit vectors gives the stated form. $\square$

Corollary 12.18 is why random projection is safe for *cosine similarity* retrieval, not merely for Euclidean distance — which matters, because most embedding search is cosine-based.

> **Worked Example 12.13 (what $k$ does the theorem actually demand?).**
> Using the cleaner constant $k=\lceil 8\varepsilon^{-2}\log n\rceil$ (i.e. dropping the $\delta$ term and one factor of $2$, which is the form usually quoted):
>
> | $n$ | $\varepsilon=0.5$ | $\varepsilon=0.2$ | $\varepsilon=0.1$ |
> |---|---|---|---|
> | $10^3$ | 222 | 1382 | 5527 |
> | $10^4$ | 295 | 1843 | 7369 |
> | $10^6$ | 443 | 2764 | 11053 |
> | $10^9$ | 664 | 4145 | 16579 |
>
> Two readings. The optimistic one: for a billion points, $664$ dimensions suffice to preserve every pairwise distance to within $50\%$, no matter whether the original data lived in $\mathbb{R}^{10^3}$ or $\mathbb{R}^{10^6}$. That is remarkable, and it is the theoretical licence for sketching at scale.
>
> The pessimistic one, which papers rarely state: for a realistic retrieval workload — $n=10^6$ embeddings of dimension $d=1536$, wanting $\varepsilon=0.1$ — the theorem demands $k=11053 > d$. **The bound is vacuous in exactly the regime where you wanted to use it.** In practice people project $1536\to256$ and it works fine. Why? Because JL is worst-case over point configurations and data-oblivious, whereas real embeddings have low intrinsic dimension, so PCA or a learned projection does far better. JL's real role is (a) as a guarantee when you cannot look at the data (streaming, adversarial, or one-pass settings), and (b) as an existence proof that dimension-independent compression is possible at all.

> **Worked Example 12.14 (measuring the distortion).**
> $n=500$ points i.i.d. $\mathcal{N}(0,I_{2000})$, `default_rng(1234)`. For each $k$, a fresh Gaussian $R\in\mathbb{R}^{2000\times k}$ with entries $\mathcal{N}(0,1/k)$; reported is $\max_{i<j}\big|\,\|Rx_i-Rx_j\|/\|x_i-x_j\|-1\,\big|$ over all $124{,}750$ pairs, plus the $95$th percentile and the mean.
>
> | $k$ | worst-case distortion | 95th pct | mean |
> |---|---|---|---|
> | 16 | 0.8240 | 0.3410 | 0.1400 |
> | 32 | 0.5897 | 0.2405 | 0.0986 |
> | 64 | 0.3708 | 0.1713 | 0.0701 |
> | 128 | 0.2682 | 0.1256 | 0.0514 |
> | 256 | 0.1999 | — | — |
> | 512 | 0.1423 | 0.0612 | 0.0250 |
> | 1024 | 0.1073 | 0.0433 | 0.0176 |
> | 1243 ($=\lceil 8\log n/0.2^2\rceil$) | 0.0890 | 0.0392 | 0.0160 |
>
> The theory prescribes $k=1243$ for $\varepsilon=0.2$; the observed worst-case distortion at that $k$ is $0.089$. Empirically $k=256$ already achieves $0.1999\le0.2$. So **the JL bound is conservative by a factor of about $4.9$ in $k$** for this configuration. That factor is roughly what you should expect: the union bound treats the $\binom n2$ pair events as if they were disjoint (Chapter 1's theme — the union bound is tight for disjoint events), whereas here they are strongly positively correlated, since all pairs share the same $R$.
>
> A second experiment confirms the $d$-independence directly: holding $k=200$ fixed and varying $d$ over $200, 1000, 5000, 20000$ gives worst-case distortions $0.2290, 0.2625, 0.2210, 0.2075$ — no trend in $d$.

<svg viewBox="0 0 760 300" style="max-width:100%;height:auto" role="img" aria-label="Histograms of the ratio of projected distance to original distance, for two projection dimensions.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.5 — Johnson–Lindenstrauss distortion, k = 64 vs k = 512</text>
  <line x1="80" y1="230" x2="710" y2="230" stroke="currentColor" stroke-width="1"/>
  <line x1="80" y1="230" x2="80" y2="55" stroke="currentColor" stroke-width="1"/>
  <text x="80" y="245" text-anchor="middle" font-size="10" fill="currentColor">0.5</text>
  <text x="235" y="245" text-anchor="middle" font-size="10" fill="currentColor">0.75</text>
  <text x="390" y="245" text-anchor="middle" font-size="10" fill="currentColor">1.0</text>
  <text x="545" y="245" text-anchor="middle" font-size="10" fill="currentColor">1.25</text>
  <text x="700" y="245" text-anchor="middle" font-size="10" fill="currentColor">1.5</text>
  <text x="390" y="260" text-anchor="middle" font-size="11" fill="currentColor">ratio  ||R x_i - R x_j|| / ||x_i - x_j||</text>
  <line x1="390" y1="230" x2="390" y2="55" stroke="currentColor" stroke-width="0.8" stroke-dasharray="4,3"/>
  <text x="394" y="66" font-size="10" fill="currentColor">ratio = 1 (no distortion)</text>
  <polyline fill="none" stroke="#f59e0b" stroke-width="1.8" points="80,230 220,230 220,229 235,229 235,228 250,228 250,227 266,227 266,223 282,223 282,218 297,218 297,210 312,210 312,201 328,201 328,191 344,191 344,182 359,182 359,177 375,177 375,174 390,174 390,178 405,178 405,184 421,184 421,193 437,193 437,202 452,202 452,210 468,210 468,216 483,216 483,221 498,221 498,225 514,225 514,227 530,227 530,228 545,228 545,229 560,229 560,230 700,230"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.8" points="80,230 328,230 328,226 344,226 344,206 359,206 359,149 375,149 375,84 390,84 390,93 405,93 405,160 421,160 421,211 437,211 437,227 452,227 452,230 700,230"/>
  <text x="150" y="180" font-size="11" fill="#f59e0b">k = 64: worst distortion 0.371</text>
  <text x="470" y="110" font-size="11" fill="#3b82f6">k = 512: worst distortion 0.142</text>
  <text x="380" y="278" text-anchor="middle" font-size="11" fill="currentColor">n = 500 points in R^2000, all 124,750 pairs, seed 1234. Increasing k by a factor of 8 narrows the</text>
  <text x="380" y="292" text-anchor="middle" font-size="11" fill="currentColor">distortion by about sqrt(8): the histogram width scales like 1/sqrt(k), exactly as Lemma 12.16 predicts.</text>
</svg>

Without the figure: two histograms of the distance ratio, both centred at $1$. The $k=64$ histogram has standard deviation $0.089$ and support roughly $[0.635,1.395]$; the $k=512$ histogram has standard deviation $0.031$ and support roughly $[0.858,1.145]$. The widths differ by a factor $\approx2.9\approx\sqrt8$, matching the $1/\sqrt k$ prediction.

### 12.4.3 What JL buys in practice

Four applications, each a direct instance of Theorem 12.17 or Corollary 12.18.

**Random features.** Rahimi and Recht's random Fourier features exploit Bochner's theorem, which writes any shift-invariant positive-definite kernel as $\kappa(x,y)=\mathbb{E}_{\omega\sim p}\big[\cos(\omega^\top(x-y))\big]$; drawing $\omega_1,\dots,\omega_m\sim p$ and $b_1,\dots,b_m$ uniform on $[0,2\pi)$ and setting $\phi(x)=\sqrt{2/m}\,\big(\cos(\omega_j^\top x+b_j)\big)_{j\le m}$ gives $\mathbb{E}\langle\phi(x),\phi(y)\rangle=\kappa(x,y)$. The uniform approximation guarantee is a concentration argument plus a covering-number union bound over the input domain — structurally the same as Theorem 12.17, with the covering number of the input set playing the role of $n$. This is the mechanism behind kernel approximation in large-scale SVMs, and (via the neural tangent kernel) part of the story of why wide random networks behave like kernel machines.

**Locality-sensitive hashing (SimHash).** Draw $r\sim\mathcal{N}(0,I_d)$ and hash $u\mapsto\mathrm{sign}(\langle r,u\rangle)$. Then $\Pr(\text{hash}(u)=\text{hash}(v)) = 1-\Theta(u,v)/\pi$ where $\Theta$ is the angle — a one-line consequence of the fact that the plane $\{x:\langle r,x\rangle=0\}$ has uniformly random orientation and separates $u$ from $v$ exactly when its normal falls in the wedge of angle $\Theta$.

> **Worked Example 12.15 (SimHash collision rates).**
> Verified with `default_rng(606)`, $d=256$, $2\times10^5$ random hyperplanes per row:
>
> | $\Theta$ | theory $1-\Theta/\pi$ | empirical |
> |---|---|---|
> | $10^\circ$ | 0.9444 | 0.9440 |
> | $30^\circ$ | 0.8333 | 0.8348 |
> | $60^\circ$ | 0.6667 | 0.6656 |
> | $90^\circ$ | 0.5000 | 0.4991 |
> | $120^\circ$ | 0.3333 | 0.3344 |
>
> An $L$-bit signature agrees fully with probability $(1-\Theta/\pi)^L$: for $\Theta=30^\circ$, that is $0.2326$ at $L=8$, $0.0541$ at $L=16$, $0.0029$ at $L=32$. This is why LSH uses *many short* signatures (the OR-of-ANDs amplification) rather than one long one: a single $32$-bit signature would miss $99.7\%$ of true $30^\circ$ neighbours.
>
> Note the connection back to §12.1: at $d=256$ two *random* vectors have $\Theta\approx90^\circ$ and hence collision probability $0.5$ per bit, so a $32$-bit signature collides with a random vector with probability $2^{-32}$. The separation between "true neighbour at $30^\circ$" and "random vector at $90^\circ$" is what makes hashing work, and it exists because §12.1.2 says random pairs really are near-orthogonal.

**Sketching and streaming.** Count-Sketch, AMS sketches, and the Frequent Directions matrix sketch all reduce to: apply a random linear map, then argue via a JL-type bound that the quantity of interest (a norm, an inner product, a frequency moment) is preserved. The $\varepsilon^{-2}$ in every sketch's space bound is the same $\varepsilon^{-2}$ as in Theorem 12.17.

**Vector databases.** Product quantization, OPQ, and the random-rotation preprocessing step in FAISS all use random orthogonal or Gaussian maps. The random rotation before coarse quantization is there specifically to equalize variance across coordinates so that a fixed bit budget per subspace is well spent — an application of Worked Example 12.4's near-orthogonality rather than of Theorem 12.17 directly. The caveat from Worked Example 12.13 applies: the *guarantees* these systems ship with are almost never the JL guarantee, because $k$ would exceed $d$. They ship with empirical recall curves.

> **Common Confusion 12.5.**
> *"JL says any dataset compresses to $O(\log n)$ dimensions, so PCA and learned embeddings are obsolete — just multiply by a random matrix."*
>
> Three separate errors. First, arithmetic: $O(\varepsilon^{-2}\log n)$ is not small. At $\varepsilon=0.1$ and $n=10^6$ it is over $11{,}000$, which exceeds most embedding dimensions (Worked Example 12.13). JL is only a *compression* result when $d$ is far larger than $\varepsilon^{-2}\log n$.
>
> Second, JL preserves *distances*, not *semantics*. If your downstream task depends on a direction that carries $0.1\%$ of the variance, JL will preserve that direction's contribution to distances only to within $\varepsilon$ — which can be far larger than the signal. PCA, by contrast, is chosen to keep the top-variance directions; a learned projection is chosen to keep the *task-relevant* ones. These are different objectives, and JL optimizes neither.
>
> Third, the lower bound. Larsen and Nelson proved that $k=\Omega(\varepsilon^{-2}\log n)$ is *necessary* for a linear map (and Alon–Klartag for the general case): JL is optimal among data-oblivious methods. So the only way to beat it is to look at the data — which is exactly what PCA does. Random projection is what you use when you *cannot* look.

> **Self-check 12.4.**
> 1. In the proof of Theorem 12.17, exactly which step introduces the $\log n$, and which introduces the $\varepsilon^{-2}$?
> 2. Why does the entry variance of $R$ have to be $1/k$ rather than $1$?
> 3. You have $n=10^4$ points in $\mathbb{R}^{50}$ and want $\varepsilon=0.1$. What does Theorem 12.17 tell you? Is it useful?
> 4. Explain in one sentence why the JL bound was conservative by a factor of about five in Worked Example 12.14.

## 12.5 Random matrices

A random matrix is not a bag of random numbers. Its *spectrum* — the collection of its singular values — converges to a deterministic limiting shape as the dimensions grow, and that shape governs everything from initialization scaling to the conditioning of covariance estimates. This section states the three limit laws you need to recognize in papers, then proves an operator-norm bound from scratch as an application of Chapter 4's covering numbers.

Throughout, $A\in\mathbb{R}^{n\times d}$ has i.i.d. entries; we normalize by $1/\sqrt n$ so that $\frac1n A^\top A$ estimates the identity. Set the **aspect ratio** $\gamma := d/n$.

### 12.5.1 The Marchenko–Pastur law

**Theorem 12.19 (Marchenko–Pastur).** Let $A\in\mathbb{R}^{n\times d}$ have i.i.d. entries with mean $0$ and variance $1/n$, and suppose $d/n\to\gamma\in(0,1]$ as $n\to\infty$. Then the empirical spectral distribution of $W=A^\top A$, namely $\mu_d := \frac1d\sum_{i=1}^d\delta_{\lambda_i(W)}$, converges weakly almost surely to the Marchenko–Pastur law with density
$$\rho_\gamma(x) = \frac{\sqrt{(b-x)(x-a)}}{2\pi\gamma x}\,\mathbb{1}\{a\le x\le b\},\qquad a=(1-\sqrt\gamma)^2,\ \ b=(1+\sqrt\gamma)^2 .$$
For $\gamma>1$ the same holds with an extra atom of mass $1-1/\gamma$ at $0$ (the matrix is rank-deficient).

This is a *universality* statement: the limit depends on the entry distribution only through its variance. Gaussian, Rademacher, uniform — same spectrum. That is the same universality as the CLT (Ch. 7), one level up: the CLT says sums of i.i.d. variables forget their distribution, MP says spectra of random matrices do too.

The single most important consequence, and the one people forget: **even when the true covariance is the identity, the sample covariance has eigenvalues spread over $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$.** At $\gamma=0.5$ (twice as many samples as dimensions) that interval is $[0.086,2.914]$: a condition number of $34$ where the truth is $1$.

**Theorem 12.20 (Bai–Yin edge behaviour).** Under the hypotheses of Theorem 12.19 with finite fourth moment, the extreme singular values of $A$ satisfy, almost surely,
$$s_{\max}(A)\to 1+\sqrt\gamma,\qquad s_{\min}(A)\to 1-\sqrt\gamma\quad(\gamma<1).$$
Equivalently $s_{\max}^2\to b$, $s_{\min}^2\to a$: there are no eigenvalue outliers beyond the MP support. (The fluctuations at the edge are of order $n^{-2/3}$ and follow the Tracy–Widom law — worth knowing the name, not needed here.)

**Theorem 12.21 (Wigner semicircle law).** Let $M\in\mathbb{R}^{d\times d}$ be symmetric with i.i.d. entries above the diagonal, mean $0$ and variance $1/d$. Then the empirical spectral distribution of $M$ converges weakly almost surely to the semicircle law with density $\rho_{sc}(x)=\frac1{2\pi}\sqrt{4-x^2}$ on $[-2,2]$.

The semicircle is the law for *symmetric* random matrices — Hessians of random functions, Gaussian orthogonal ensembles, graph adjacency matrices. Its most important feature for us: it is symmetric about zero, so exactly half its mass is negative. We use this in §12.6.

> **Worked Example 12.16 (MP and Bai–Yin numerically).**
> $A\in\mathbb{R}^{n\times d}$ with i.i.d. $\mathcal{N}(0,1/n)$ entries, $n=4000$, `default_rng(555)`:
>
> | $\gamma$ | $d$ | $s_{\min}$ observed | $1-\sqrt\gamma$ | $s_{\max}$ observed | $1+\sqrt\gamma$ |
> |---|---|---|---|---|---|
> | 0.10 | 400 | 0.6896 | 0.6838 | 1.3123 | 1.3162 |
> | 0.50 | 2000 | 0.2966 | 0.2929 | 1.7045 | 1.7071 |
> | 1.00 | 4000 | 0.0002 | 0.0000 | 1.9911 | 2.0000 |
>
> Agreement to three decimals at $n=4000$. Note the $\gamma=1$ row: a *square* random matrix has $s_{\min}\approx 2\times10^{-4}$, i.e. it is nearly singular. This is not an accident of the seed; Rudelson–Vershynin show $s_{\min}\asymp n^{-1/2}\cdot n^{-1/2}=1/n$ in the normalization used here, so square random matrices are badly conditioned with high probability. Anyone who has tried to invert a randomly initialized square weight matrix has met this.

<svg viewBox="0 0 760 300" style="max-width:100%;height:auto" role="img" aria-label="Empirical eigenvalue histograms of random Wishart matrices with the Marchenko-Pastur density overlaid, at two aspect ratios.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.6 — Marchenko–Pastur density vs empirical spectrum</text>
  <text x="220" y="40" text-anchor="middle" font-size="12" fill="currentColor">gamma = d/n = 0.2, support [0.306, 2.094]</text>
  <line x1="70" y1="230" x2="370" y2="230" stroke="currentColor" stroke-width="1"/>
  <line x1="70" y1="230" x2="70" y2="55" stroke="currentColor" stroke-width="1"/>
  <text x="70" y="245" text-anchor="middle" font-size="10" fill="currentColor">0</text>
  <text x="164" y="245" text-anchor="middle" font-size="10" fill="currentColor">1</text>
  <text x="257" y="245" text-anchor="middle" font-size="10" fill="currentColor">2</text>
  <text x="351" y="245" text-anchor="middle" font-size="10" fill="currentColor">3</text>
  <text x="220" y="260" text-anchor="middle" font-size="11" fill="currentColor">eigenvalue of (1/n) X^T X</text>
  <polyline fill="none" stroke="currentColor" stroke-width="1.3" opacity="0.75" points="70,230 78,230 78,230 87,230 87,230 95,230 95,200 103,200 103,148 112,148 112,134 120,134 120,138 128,138 128,140 137,140 137,140 145,140 145,148 153,148 153,150 162,150 162,160 170,160 170,164 178,164 178,164 187,164 187,172 195,172 195,176 203,176 203,180 212,180 212,186 220,186 220,188 228,188 228,198 237,198 237,198 245,198 245,208 253,208 253,214 262,214 262,224 270,224 270,230 278,230"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2.2" points="100,183 103,162 105,151 108,145 110,140 113,138 115,136 118,136 120,135 123,136 126,136 128,137 131,138 133,139 136,140 138,142 141,143 143,144 146,146 148,147 151,149 153,150 156,152 158,153 161,155 163,156 166,158 168,159 171,160 173,162 176,163 178,165 181,166 184,168 186,169 189,170 191,172 194,173 196,175 199,176 201,177 204,179 206,180 209,181 211,183 214,184 216,186 219,187 221,188 224,190 226,191 229,193 231,194 234,196 236,197 239,199 241,201 244,202 247,204 249,206 252,208 254,210 257,212 259,215 262,218 264,222"/>
  <text x="240" y="120" font-size="10" fill="#3b82f6">MP density</text>
  <text x="550" y="40" text-anchor="middle" font-size="12" fill="currentColor">gamma = 1.0, support [0, 4]</text>
  <line x1="420" y1="230" x2="720" y2="230" stroke="currentColor" stroke-width="1"/>
  <line x1="420" y1="230" x2="420" y2="55" stroke="currentColor" stroke-width="1"/>
  <text x="420" y="245" text-anchor="middle" font-size="10" fill="currentColor">0</text>
  <text x="514" y="245" text-anchor="middle" font-size="10" fill="currentColor">1</text>
  <text x="607" y="245" text-anchor="middle" font-size="10" fill="currentColor">2</text>
  <text x="701" y="245" text-anchor="middle" font-size="10" fill="currentColor">3</text>
  <text x="570" y="260" text-anchor="middle" font-size="11" fill="currentColor">eigenvalue of (1/n) X^T X</text>
  <polyline fill="none" stroke="currentColor" stroke-width="1.3" opacity="0.75" points="420,60 428,60 428,134 437,134 437,157 445,157 445,169 453,169 453,176 462,176 462,184 470,184 470,187 478,187 478,191 487,191 487,193 495,193 495,196 503,196 503,198 512,198 512,199 520,199 520,202 528,202 528,202 537,202 537,204 545,204 545,206 553,206 553,207 562,207 562,208 570,208 570,210 578,210 578,209 587,209 587,211 595,211 595,212 603,212 603,212 612,212 612,213 620,213 620,213 628,213 628,215 637,215 637,215 645,215 645,215 653,215 653,217 662,217 662,217 670,217 670,218 678,218 678,219 687,219 687,219 695,219 695,220 703,220 703,221 712,221 712,220 720,220"/>
  <polyline fill="none" stroke="#ef4444" stroke-width="2.2" points="423,60 425,86 428,113 430,129 433,140 435,148 438,154 440,159 443,163 445,167 448,170 450,173 453,175 455,178 458,180 460,181 463,183 465,184 468,186 470,187 473,188 476,189 478,190 481,191 483,192 486,193 488,194 491,195 493,196 496,196 498,197 501,198 503,198 506,199 508,200 511,200 513,201 516,201 518,202 521,202 523,203 526,203 528,203 531,204 534,204 536,205 539,205 541,206 544,206 546,206 549,207 551,207 554,207 556,208 559,208 561,208 564,209 566,209 569,209 571,209 574,210 576,210 579,210 581,211 584,211 586,211 589,211 591,212 594,212 597,212 599,212 602,213 604,213 607,213 609,213 612,213 614,214 617,214 619,214 622,214 624,215 627,215 629,215 632,215 634,215 637,216 639,216 642,216 644,216 647,216 649,217 652,217 654,217 657,217 660,217 662,217 665,218 667,218 670,218 672,218 675,218 677,219 680,219 682,219 685,219 687,219 690,219 692,220 695,220 697,220 700,220 702,220 705,220 707,221 710,221 712,221 715,221 717,221 720,222"/>
  <text x="600" y="150" font-size="10" fill="#ef4444">MP density (diverges at 0)</text>
  <text x="380" y="278" text-anchor="middle" font-size="11" fill="currentColor">n = 3000, seed 2718. TRUE covariance is the identity in both panels, yet the sample eigenvalues fill an</text>
  <text x="380" y="292" text-anchor="middle" font-size="11" fill="currentColor">interval — [0.31, 2.09] at gamma=0.2, and all of [0, 4] at gamma=1. This is the covariance problem of §12.7.</text>
</svg>

If the figure does not render: two overlaid histogram-and-curve plots. At $\gamma=0.2$ the eigenvalues of $\frac1nX^\top X$ fill $[0.307,2.105]$ (theory $[0.3056,2.0944]$), with the density peaking near the lower edge. At $\gamma=1$ they fill $[0.000,3.988]$ (theory $[0,4]$), with the density diverging like $x^{-1/2}$ at zero. In both panels the true covariance is $I$.

### 12.5.2 Operator norm via an $\varepsilon$-net: a full proof

The limit laws above are stated without proof. The following bound we prove completely, because the technique — discretize the sphere, union-bound over the net, pay for the discretization — is the single most reusable argument in high-dimensional probability, and Chapter 4 already gave us the covering numbers.

Recall from Chapter 4: an $\varepsilon$-net of a set $T$ in a metric space is a subset $\mathcal{N}\subseteq T$ such that every point of $T$ is within $\varepsilon$ of some point of $\mathcal{N}$, and the unit sphere $S^{d-1}$ admits an $\varepsilon$-net of cardinality at most $(1+2/\varepsilon)^d$.

**Lemma 12.22 (net-to-norm).** Let $A\in\mathbb{R}^{n\times d}$, let $\varepsilon\in(0,\tfrac12)$, and let $\mathcal{N}\subseteq S^{d-1}$, $\mathcal{M}\subseteq S^{n-1}$ be $\varepsilon$-nets. Then
$$\|A\|\ :=\ \sup_{v\in S^{d-1}}\|Av\| \ \le\ \frac1{1-2\varepsilon}\,\max_{u\in\mathcal{M},\,v\in\mathcal{N}}\ \langle u, Av\rangle .$$

*Proof.* Write $S:=\max_{u\in\mathcal{M},v\in\mathcal{N}}\langle u,Av\rangle$. Pick $u^\star\in S^{n-1}$, $v^\star\in S^{d-1}$ attaining $\|A\|=\langle u^\star,Av^\star\rangle$ (the sup is attained by compactness and continuity). Choose $u\in\mathcal{M}$ with $\|u-u^\star\|\le\varepsilon$ and $v\in\mathcal{N}$ with $\|v-v^\star\|\le\varepsilon$. Then
$$\|A\| = \langle u^\star,Av^\star\rangle = \langle u,Av\rangle + \langle u^\star-u,Av^\star\rangle + \langle u, A(v^\star-v)\rangle .$$
Bound the second term by $\|u^\star-u\|\,\|Av^\star\|\le\varepsilon\|A\|$ (Cauchy–Schwarz and $\|v^\star\|=1$), and the third by $\|u\|\,\|A\|\,\|v^\star-v\|\le\varepsilon\|A\|$. Hence $\|A\|\le S+2\varepsilon\|A\|$, and rearranging gives $\|A\|\le S/(1-2\varepsilon)$. $\square$

**Theorem 12.23 (operator norm of a Gaussian matrix).** Let $B\in\mathbb{R}^{n\times d}$ have i.i.d. $\mathcal{N}(0,1)$ entries. Fix any $\varepsilon\in(0,\tfrac12)$ and set
$$C_\varepsilon:=\frac{\sqrt{2\log(1+2/\varepsilon)}}{1-2\varepsilon},\qquad K_\varepsilon:=\frac1{1-2\varepsilon}.$$
Then for every $s\ge0$,
$$\Pr\Big(\|B\|\ \ge\ C_\varepsilon\big(\sqrt n+\sqrt d\big)\ +\ K_\varepsilon\,s\Big)\ \le\ e^{-s^2/2}.$$
The choice $\varepsilon^\star=0.0605$ minimizes $C_\varepsilon$, giving $C_{\varepsilon^\star}=3.0220$ and $K_{\varepsilon^\star}=1.1379$; so with probability at least $1-e^{-s^2/2}$,
$$\|B\|\ \le\ 3.023\big(\sqrt n+\sqrt d\big)\ +\ 1.138\,s .$$
In particular $\mathbb{E}\|B\|\lesssim\sqrt n+\sqrt d$ (integrate the tail).

*Proof.* Fix $\varepsilon\in(0,\tfrac12)$ and let $\mathcal{N},\mathcal{M}$ be $\varepsilon$-nets of $S^{d-1},S^{n-1}$ with $|\mathcal{N}|\le(1+2/\varepsilon)^d$, $|\mathcal{M}|\le(1+2/\varepsilon)^n$ (Ch. 4). Write $L:=\log(1+2/\varepsilon)$, so $|\mathcal{N}||\mathcal{M}|\le e^{L(n+d)}$.

*Step 1: one pair.* For fixed unit $u,v$, the scalar $\langle u,Bv\rangle = \sum_{i,j}u_iB_{ij}v_j$ is a linear combination of i.i.d. standard Gaussians with coefficient vector of Euclidean norm $\big(\sum_{i,j}u_i^2v_j^2\big)^{1/2}=\|u\|\|v\|=1$. Hence $\langle u,Bv\rangle\sim\mathcal{N}(0,1)$ exactly, and $\Pr(\langle u,Bv\rangle\ge r)\le e^{-r^2/2}$ by the standard Gaussian tail (Ch. 4).

*Step 2: union bound over the nets.* By Proposition 1.8(2),
$$\Pr\Big(\max_{u\in\mathcal{M},v\in\mathcal{N}}\langle u,Bv\rangle\ \ge\ r\Big)\ \le\ e^{L(n+d)}e^{-r^2/2}.$$

*Step 3: choose $r$.* Take $r := \sqrt{2L(n+d)}+s$. Since $(\alpha+\beta)^2\ge\alpha^2+\beta^2$ for $\alpha,\beta\ge0$, we get $r^2\ge2L(n+d)+s^2$, hence $L(n+d)-r^2/2\le-s^2/2$ and Step 2's bound becomes
$$\Pr\Big(\max_{u\in\mathcal{M},v\in\mathcal{N}}\langle u,Bv\rangle\ \ge\ \sqrt{2L(n+d)}+s\Big)\ \le\ e^{-s^2/2}.$$
(No case analysis is needed; the single elementary inequality does all the work.)

*Step 4: pay for the net.* By Lemma 12.22, $\|B\|\le(1-2\varepsilon)^{-1}\max_{u,v}\langle u,Bv\rangle$. Also $\sqrt{n+d}\le\sqrt n+\sqrt d$. Therefore with probability at least $1-e^{-s^2/2}$,
$$\|B\|\ \le\ \frac{\sqrt{2L}}{1-2\varepsilon}\big(\sqrt n+\sqrt d\big)\ +\ \frac{s}{1-2\varepsilon}\ =\ C_\varepsilon\big(\sqrt n+\sqrt d\big)+K_\varepsilon s,$$
which is exactly the claim. Minimizing $C_\varepsilon$ over $\varepsilon\in(0,\tfrac12)$ by bounded scalar optimization gives $\varepsilon^\star=0.0605$, $C_{\varepsilon^\star}=3.0220$, $K_{\varepsilon^\star}=1.1379$. $\square$

Note that we did *not* absorb $K_\varepsilon$ into the $s$ term. It is tempting to write the conclusion as "$\|B\|\le C(\sqrt n+\sqrt d)+s$ with probability $1-e^{-s^2/2}$", but rescaling $s\mapsto s/K_\varepsilon$ degrades the exponent to $-s^2/(2K_\varepsilon^2)$, and quietly dropping that is the kind of constant-shuffling error that makes a bound unusable when you later want a numerical value from it. Carrying $K_\varepsilon$ explicitly costs one symbol and keeps the statement true.

The trade-off in Step 4 is worth naming explicitly, because it is the generic tension in every net argument. A *fine* net ($\varepsilon$ small) makes the discretization penalty $(1-2\varepsilon)^{-1}$ close to $1$, but blows up the cardinality $(1+2/\varepsilon)^{n+d}$, which enters the bound as $\sqrt{2\log(1+2/\varepsilon)}$ — only logarithmically, which is why fine nets are affordable. A *coarse* net ($\varepsilon\to\tfrac12$) keeps the net small but the penalty $(1-2\varepsilon)^{-1}$ diverges. Because cardinality enters through a logarithm and the penalty enters linearly, the optimum sits at a fairly *fine* net: $\varepsilon^\star\approx0.06$, not $\varepsilon=1/4$ as is often chosen for convenience.

A forward pointer for when you meet the technique's descendants in papers. The net argument is a *one-scale* discretization: pick a single resolution, pay for it once. The multiscale refinement — approximate at a coarse scale, then control the corrections at successively finer scales — is *chaining* (Dudley's entropy integral, and Talagrand's generic chaining), and the quantity it produces for Gaussian processes, $\mathbb{E}\sup_{t\in T}\langle Z,t\rangle$, is called the *Gaussian width* of $T$. Gaussian width is the currency of modern signal-recovery and empirical-process bounds; when a paper's rate features an entropy integral or $w(T)$, it is running the argument of this section at all scales simultaneously. Vershynin's Chapters 7–8 are the place to learn it; nothing in this tutorial requires it.

> **Worked Example 12.17 (how good is $\sqrt n+\sqrt d$?).**
> `default_rng(555)`, $B$ with i.i.d. $\mathcal{N}(0,1)$ entries, exact spectral norm via SVD:
>
> | $n$ | $d$ | $\|B\|$ | $\sqrt n+\sqrt d$ | ratio |
> |---|---|---|---|---|
> | 100 | 100 | 19.258 | 20.000 | 0.9629 |
> | 1000 | 100 | 41.636 | 41.623 | 1.0003 |
> | 500 | 500 | 44.872 | 44.721 | 1.0034 |
> | 2000 | 200 | 58.612 | 58.863 | 0.9957 |
>
> So $\|B\|\approx\sqrt n+\sqrt d$ with a constant of essentially $1$. Compare the constants: Bai–Yin (Theorem 12.20) gives the sharp asymptotic constant $1$; our $\varepsilon$-net proof gives $C\le3.02$. The net argument is off by a factor of three — and that is a good trade, because it required no random matrix theory, it holds at every finite $n,d$ rather than asymptotically, it comes with an explicit tail, and (crucially) it generalizes verbatim to sub-Gaussian entries, structured matrices, and matrices with dependent rows, where Bai–Yin does not apply. This is the standard division of labour in the field: sharp constants from RMT, robust constants from nets.
>
> Practical payoff: $\|W\|\approx\sqrt{n_{\text{out}}}+\sqrt{n_{\text{in}}}$ for a Gaussian-initialized layer tells you the *worst-case* amplification of a single layer at initialization, which is what spectral normalization and Lipschitz-constrained architectures control. It also tells you the scaling: to keep $\|W\|=O(1)$ you must scale entries by $1/(\sqrt{n_{\text{out}}}+\sqrt{n_{\text{in}}})$, which for a square layer means variance $\asymp1/(4n)$ — noticeably smaller than the $2/n_{\text{in}}$ of He initialization, because He controls the *average* amplification (the Frobenius/variance scale), not the worst case.

**Theorem 12.24 (Matrix Bernstein).** Let $X_1,\dots,X_N$ be independent, mean-zero, symmetric $d\times d$ random matrices with $\|X_i\|\le K$ almost surely, and let $\sigma^2 := \big\|\sum_i\mathbb{E}X_i^2\big\|$. Then for all $t\ge0$,
$$\Pr\Big(\Big\|\sum_{i=1}^N X_i\Big\|\ \ge\ t\Big)\ \le\ 2d\,\exp\Big(\frac{-t^2/2}{\sigma^2+Kt/3}\Big).$$
Compare with scalar Bernstein (Ch. 4): the exponent is identical, and the only price for going from scalars to matrices is the prefactor $2d$ — i.e. an effective union bound over $d$ dimensions, which costs $\log d$ inside the deviation. Note the hypothesis: the summands must be *bounded* in operator norm. For genuinely heavy-tailed data one first truncates and controls the truncation error separately, or uses the sub-exponential extensions in Tropp's monograph; the bounded version stated here is the workhorse. It is the tool of choice for covariance estimation from bounded vectors, for randomized numerical linear algebra, and for analysing sampling-based approximations to large matrices (e.g. Nyström and column-sampling methods).

To see it earn its keep, here is the covariance application in full, because it complements Proposition 12.26 below: that proposition is Gaussian-specific, while this argument needs only boundedness. Let $X_1,\dots,X_n$ be i.i.d. mean-zero in $\mathbb{R}^d$ with covariance $\Sigma$ and $\|X_i\|^2\le B$ almost surely. Set $Y_i := \tfrac1n(X_iX_i^\top-\Sigma)$, so that $\hat\Sigma-\Sigma=\sum_iY_i$ with the $Y_i$ independent, mean-zero, and symmetric. Two quantities to compute. *Boundedness:* $\|X_iX_i^\top\|=\|X_i\|^2\le B$, so $\|Y_i\|\le\tfrac1n(B+\|\Sigma\|)\le\tfrac{2B}n=:K$, using $\|\Sigma\|=\|\mathbb{E}X X^\top\|\le\mathbb{E}\|X\|^2\le B$. *Variance:* $\mathbb{E}Y_i^2\preceq\tfrac1{n^2}\mathbb{E}\big[\|X_i\|^2X_iX_i^\top\big]\preceq\tfrac B{n^2}\Sigma$, so $\sigma^2=\|\sum_i\mathbb{E}Y_i^2\|\le\tfrac{B\|\Sigma\|}n$. Plugging both into Theorem 12.24 and inverting the tail: with probability at least $1-\delta$,
$$\|\hat\Sigma-\Sigma\|\ \lesssim\ \sqrt{\frac{B\,\|\Sigma\|\,\log(d/\delta)}{n}}\ +\ \frac{B\log(d/\delta)}{n}.$$
Read the shape against Proposition 12.26: the same two Bernstein regimes ($\sqrt{\cdot/n}$ then $1/n$), with the effective dimension $B/\|\Sigma\|$ — which is at most $d$ for isotropic data, but can be far smaller when the vectors are structured — replacing the raw $d$, at the price of an extra $\log d$ from the dimensional prefactor. Exercise A18 asks you to carry this computation through in detail and to compare the resulting sample-size requirement with the Gaussian answer.

> **Common Confusion 12.6.**
> *"Theorem 12.23 says the norm is about $\sqrt n+\sqrt d$, so a random matrix is well conditioned — all its singular values are of that order."*
>
> The theorem bounds the *largest* singular value only. The smallest is a completely different question, and the answer depends sharply on the aspect ratio. From Bai–Yin (in the $1/\sqrt n$ normalization): $s_{\min}\to1-\sqrt\gamma$, so
> - $\gamma\ll1$ (tall matrix, many more rows than columns): $s_{\min}\approx1$, condition number $\approx(1+\sqrt\gamma)/(1-\sqrt\gamma)\approx1$. Well conditioned.
> - $\gamma=0.5$: condition number $\to1.7071/0.2929=5.83$ (in singular values), i.e. $34.0$ for eigenvalues of $A^\top A$.
> - $\gamma\to1$: $s_{\min}\to0$. A square Gaussian matrix is nearly singular; Worked Example 12.16 measured $s_{\min}=2\times10^{-4}$ at $n=d=4000$.
>
> This matters concretely in deep learning. A square weight matrix at Gaussian initialization has condition number growing with width, which means that even though the *average* gain through the layer is correct (that is what He initialization ensures), the gain along the worst input direction is essentially zero. Repeated over many layers, the small singular values compound and the Jacobian's spectrum spreads over many orders of magnitude. §12.6 takes this up.

> **Self-check 12.5.**
> 1. What does Marchenko–Pastur predict for the spectrum of $\frac1nX^\top X$ when $d=n$ and the true covariance is $I_d$? Give the support.
> 2. In Lemma 12.22, why do we need nets on *both* spheres?
> 3. In Theorem 12.23's proof, where exactly does the union bound enter, and what is the cardinality being unioned over?
> 4. State the trade-off in choosing $\varepsilon$ for the net, and say why the optimum is closer to $0.06$ than to $0.4$.

## 12.6 Consequences for deep learning

This is the section where the mathematics meets practice, and it is also the section where the literature is least careful. I will therefore label every claim. **[T]** marks a theorem with hypotheses that hold for the object described. **[T*]** marks a theorem about a simplified model — deep linear networks, infinite width, random labels — that nobody actually trains. **[E]** marks an empirical observation, reproducible but not proved. **[C]** marks conjecture or folklore. Learning to make these distinctions yourself is more valuable than any single fact in this section.

### 12.6.1 Initialization: the whole spectrum, not just the variance

Chapter 5 derived He initialization by variance propagation: for a ReLU layer $h\mapsto\phi(Wh)$ with $W\in\mathbb{R}^{n_{\text{out}}\times n_{\text{in}}}$ having i.i.d. mean-zero entries of variance $\sigma^2$, requiring $\mathbb{E}\|\phi(Wh)\|^2=\|h\|^2$ forces $\sigma^2=2/n_{\text{in}}$. That derivation controls one number: the *average* gain.

**Proposition 12.25.** Let $W\in\mathbb{R}^{n_{\text{out}}\times n_{\text{in}}}$ have i.i.d. mean-zero entries with variance $\sigma^2$ and let $h\in\mathbb{R}^{n_{\text{in}}}$ be fixed. Then $\mathbb{E}\|Wh\|^2=n_{\text{out}}\sigma^2\|h\|^2$, and equivalently $\mathbb{E}\big[\tfrac1{n_{\text{in}}}\|W\|_F^2\big]=n_{\text{out}}\sigma^2$. Moreover, if $n_{\text{in}}/n_{\text{out}}\to\gamma'\in(0,1]$ (finite fourth moment), the singular values of $W/(\sigma\sqrt{n_{\text{out}}})$ have limiting distribution given by the Marchenko–Pastur law of Theorem 12.19 with aspect ratio $\gamma'$ (the "quarter-circle" law when $\gamma'=1$), and by Bai–Yin
$$s_{\max}(W)\to\sigma\big(\sqrt{n_{\text{in}}}+\sqrt{n_{\text{out}}}\big),\qquad s_{\min}(W)\to\sigma\,\big|\sqrt{n_{\text{in}}}-\sqrt{n_{\text{out}}}\big|.$$

*Proof of the first part.* $\mathbb{E}\|Wh\|^2 = \sum_{i=1}^{n_{\text{out}}}\mathbb{E}\big(\sum_j W_{ij}h_j\big)^2 = \sum_i\sum_j \sigma^2h_j^2 = n_{\text{out}}\sigma^2\|h\|^2$, using independence and mean zero to kill cross terms. For the spectral claim, view the $n_{\text{out}}$ rows of $W$ as "samples" in $\mathbb{R}^{n_{\text{in}}}$: the matrix $A:=W/(\sigma\sqrt{n_{\text{out}}})$ has i.i.d. entries of variance $1/n_{\text{out}}$, which is exactly the normalization of Theorem 12.19 with $n=n_{\text{out}}$ playing the sample role and $d=n_{\text{in}}$ the dimension role, hence aspect ratio $\gamma'=n_{\text{in}}/n_{\text{out}}$. Theorems 12.19 and 12.20 give $s_{\max}(A)\to1+\sqrt{\gamma'}$ and $s_{\min}(A)\to1-\sqrt{\gamma'}$; multiplying back by $\sigma\sqrt{n_{\text{out}}}$ gives $\sigma(\sqrt{n_{\text{out}}}+\sqrt{n_{\text{in}}})$ and $\sigma(\sqrt{n_{\text{out}}}-\sqrt{n_{\text{in}}})$ up to sign, i.e. the stated symmetric-in-$(n_{\text{in}},n_{\text{out}})$ form. $\square$ **[T]**

A caution worth a sentence, because the error is easy to make and invisible on square layers: it is tempting to write the edges as "$\sigma\sqrt{n_{\text{out}}}(1\pm\sqrt{\gamma})$ with $\gamma=n_{\text{out}}/n_{\text{in}}$", but that pairs the wrong aspect ratio with the wrong normalization and is off by the factor $\sqrt{n_{\text{out}}/n_{\text{in}}}$ whenever the layer is rectangular. The symmetric form $\sigma(\sqrt{n_{\text{in}}}+\sqrt{n_{\text{out}}})$ is the safe one to remember — it also matches Theorem 12.23's $\sqrt n+\sqrt d$, as it must, since both describe the same largest singular value. Checked numerically: $W\in\mathbb{R}^{400\times1600}$ with $\mathcal{N}(0,1)$ entries, `default_rng(808)`, gives $s_{\max}=59.891$ and $s_{\min}=20.412$ against the predictions $\sqrt{1600}+\sqrt{400}=60$ and $\sqrt{1600}-\sqrt{400}=20$; the mispaired formula would predict $30$ and $10$.

The mean gain being right does not mean the gain is right in every direction. For a square layer ($n_{\text{in}}=n_{\text{out}}=n$, so $\gamma'=1$) the spectrum runs all the way from $0$ to $2\sigma\sqrt{n}$: some input directions are amplified by twice the target factor, and others are annihilated.

> **Worked Example 12.18 (He initialization has a condition number of two thousand).**
> $W\in\mathbb{R}^{512\times512}$ with i.i.d. $\mathcal{N}(0,2/512)$ entries, `default_rng(808)`, exact SVD:
>
> | quantity | He (Gaussian) | scaled orthogonal ($\sqrt2\,Q$, $Q$ from QR) |
> |---|---|---|
> | $s_{\max}$ | 2.7847 | 1.4142 |
> | $s_{\min}$ | 0.001292 | 1.4142 |
> | condition number | 2155.8 | 1.0000 |
> | mean $s^2$ | 2.0129 | 2.0000 |
> | fraction with $s<0.1$ | 0.0469 | 0 |
>
> Both initializations get the *average* gain exactly right (mean $s^2\approx2$, which is what Chapter 5's derivation demanded, and $2\sigma\sqrt n=2\sqrt2=2.8284$ matches the observed $s_{\max}$). They differ completely in every other respect. Gaussian init sends $4.7\%$ of directions through with gain below $0.1$ — those directions carry essentially no signal — while a $\sqrt2$-scaled orthogonal matrix has gain exactly $\sqrt2$ in every direction.
>
> This is the content of **dynamical isometry**: the requirement that the input–output Jacobian's singular values all be close to $1$, not merely that their mean square be $1$. Saxe, McClelland and Ganguli **[T*]** proved for deep *linear* networks that orthogonal initialization gives depth-independent training time while Gaussian initialization gives training time growing with depth; Pennington, Schoenholz and Ganguli **[T*]** computed the full limiting Jacobian spectrum for random deep networks with various nonlinearities and showed that exact dynamical isometry is achievable with $\tanh$ plus orthogonal weights, but is *impossible* for ReLU (whose gating always spreads the spectrum). Empirically **[E]**, orthogonal initialization measurably helps very deep plain networks and RNNs, and matters much less once you have residual connections and normalization layers — which is why most practitioners never see the effect.
>
> A note on why the practical effect is smaller than the theory suggests. Forward-pass norm propagation is fine under He init even at depth: with width $512$, `default_rng(808)`, the hidden-activation norms at layers $1,2,5,10$ of a ReLU MLP starting from a unit-norm input are $1.0004$, $1.0204$, $1.1011$, $1.0802$ — no explosion or collapse. That is not a contradiction: the norm is a mean-square quantity and concentrates (§12.2), so He init gets it right; it is the *conditioning* that is bad, and conditioning costs you optimization speed rather than signal survival.

### 12.6.2 The Hessian spectrum and the saddle story

The claim you will meet, in some form, in a great many papers: *"in high dimension, almost all critical points of a random function are saddles rather than local minima, so gradient descent is not troubled by local minima."* Let us take this apart.

The kernel of truth is Theorem 12.21. If the Hessian at a critical point behaved like a Wigner matrix — i.i.d. symmetric entries — its spectrum would follow the semicircle law, symmetric about zero, so each eigenvalue would be negative with probability $\tfrac12$ and the probability that *all* $P$ eigenvalues are positive would be exponentially small in $P$. With $P=10^9$ parameters, a local minimum would be an astronomically rare kind of critical point. **[T]** for Wigner matrices.

> **Worked Example 12.19 (semicircle and the index).**
> $M=(G+G^\top)/\sqrt{2d}$ with $G$ having i.i.d. $\mathcal{N}(0,1)$ entries, $d=3000$, `default_rng(808)`: observed spectrum $[-2.0024,\,1.9944]$ (semicircle edges $\pm2$), fraction of negative eigenvalues $0.4997$. Shifting by $cI$ — which is what happens at a critical point of a function with a positive "mean curvature" $c$ — gives the fraction of negative eigenvalues (the *index*):
>
> | shift $c$ | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 |
> |---|---|---|---|---|---|---|
> | index (fraction of $\lambda<0$) | 0.4997 | 0.3420 | 0.1953 | 0.0720 | 0.0003 | 0.0000 |
>
> The index falls continuously to zero as the shift reaches the semicircle edge. So "index $\approx0$" (a local minimum) and "index $\approx\tfrac12$" (a generic saddle) are the two ends of a *continuum*, and which end you are at depends on the shift — which in the random-matrix models of loss landscapes is tied to the loss value. This is the real content of the Bray–Dean and Fyodorov–Williams analyses of Gaussian random fields, imported into ML by Dauphin et al.: **low-loss critical points have low index, high-loss critical points have index near $\tfrac12$.** **[T*]** — it is a theorem, but about isotropic Gaussian random fields, which a neural network loss is not.

Now the honest accounting of what is known.

**What is proved.** For deep *linear* networks with squared loss, every local minimum is a global minimum (Kawaguchi; earlier Baldi–Hornik for one hidden layer) **[T*]**. For matrix factorization and phase retrieval and a handful of other structured problems, the loss satisfies the *strict saddle property* — every critical point is either a local minimum or has a strictly negative Hessian eigenvalue — and perturbed gradient descent escapes all saddles in polynomial time (Ge–Huang–Jin–Yuan; Jin–Ge–Netrapalli–Kakade–Jordan) **[T]** for those problems. For sufficiently overparameterized networks in the NTK regime, gradient descent converges to global minimum training loss (Du et al., Allen-Zhu et al., Jacot et al.) **[T*]** — under width requirements that are polynomially large and far beyond anything trained in practice, and with a linearized dynamics that provably does *not* describe feature learning.

**What is proved in the other direction, and is usually not mentioned.** Two-layer ReLU networks *do* have spurious local minima, with positive probability over the data, even in very simple settings — Safran and Shamir constructed and counted them **[T]**. Yun, Sra and Jadbabaie characterized when spurious local minima exist for one-hidden-layer networks **[T]**. Deep networks with ReLU also have flat regions where whole units are dead. So "no bad local minima" is *false* as a general statement about the architectures we train; it is true for particular simplified models.

**What is empirical.** The Hessian of a trained network has a characteristic shape: a large bulk of near-zero eigenvalues, a small number of large positive outliers roughly matching the number of classes or of active features, and a small negative tail whose magnitude is tiny relative to the top eigenvalue (Sagun et al.; Ghorbani, Krishnan and Xiao; Papyan) **[E]**. Loss landscape visualizations show the paths found by SGD to be largely monotone and free of high barriers, and independently trained solutions are often connected by low-loss curves (Garipov et al.; Draxler et al.; Frankle et al.) **[E]**.

**What is conjecture.** That the *reason* SGD works is the absence of bad minima. This does not follow from any of the above and is probably not the right explanation: the empirical evidence points more toward the loss having a vast connected near-zero set which SGD's implicit bias navigates, with the interesting question being *which* of those minima it selects, not whether it can reach one **[C]**.

> **Worked Example 12.20 (measuring a Hessian spectrum).**
> A one-hidden-layer $\tanh$ network, input dimension $10$, width $20$, so $P=241$ parameters, trained by full-batch gradient descent on $n=200$ synthetic points generated by a teacher network plus noise. `default_rng(31415)`. The exact Hessian is computed by central differences on the analytic gradient (step $10^{-5}$) and symmetrized; eigenvalues by `eigvalsh`.
>
> | | at random init (loss $1.326$) | after $2\times10^5$ GD steps (loss $3.05\times10^{-6}$) |
> |---|---|---|
> | $\lambda_{\min}$ | $-1.3548$ | $-0.000595$ |
> | $\lambda_{\max}$ | $3.0737$ | $2.000571$ |
> | fraction $\lambda<0$ | 0.5104 | 0.1079 (26 of 241) |
> | $\vert\lambda_{\min}\vert/\lambda_{\max}$ | 0.441 | $2.97\times10^{-4}$ |
> | eigenvalues $>0.05$ | — | 20 (= number of hidden units) |
> | fraction $\vert\lambda\vert<0.01$ | — | 0.776 |
>
> Read the columns. At initialization the spectrum is roughly symmetric with index $0.51$ — exactly the semicircle prediction of Worked Example 12.19, and the reason the "critical points are saddles" heuristic has any force. After training, the picture is entirely different: $77.6\%$ of directions are flat to within $0.01$, there are exactly $20$ meaningful positive curvature directions (one per hidden unit — the outlier count tracks structure, matching the empirical literature), and although $26$ eigenvalues are technically negative, the most negative is $3\times10^{-4}$ of the largest. Numerically this is a minimum up to the resolution of the finite-difference Hessian; structurally it is a point in a very high-dimensional flat valley.
>
> This is one small model with one seed, and I am reporting it as **[E]**, not as evidence for a theorem. But it does illustrate why the two claims "most critical points are saddles" and "trained networks sit in flat minima" are both repeated: they describe *different points* in parameter space. Do not let a statement about random initializations be quoted at you as a statement about trained solutions.

<svg viewBox="0 0 760 300" style="max-width:100%;height:auto" role="img" aria-label="Sorted Hessian eigenvalues of a small network at initialization and after training.">
  <text x="380" y="20" text-anchor="middle" font-size="15" fill="currentColor" font-weight="bold">Figure 12.7 — Hessian spectrum: random init vs trained solution</text>
  <line x1="90" y1="220" x2="680" y2="220" stroke="currentColor" stroke-width="1"/>
  <line x1="90" y1="220" x2="90" y2="50" stroke="currentColor" stroke-width="1"/>
  <line x1="90" y1="165" x2="680" y2="165" stroke="currentColor" stroke-width="0.8" stroke-dasharray="4,3"/>
  <text x="684" y="169" font-size="10" fill="currentColor">0</text>
  <text x="84" y="203" text-anchor="end" font-size="10" fill="currentColor">-1</text>
  <text x="84" y="169" text-anchor="end" font-size="10" fill="currentColor">0</text>
  <text x="84" y="134" text-anchor="end" font-size="10" fill="currentColor">1</text>
  <text x="84" y="100" text-anchor="end" font-size="10" fill="currentColor">2</text>
  <text x="84" y="66" text-anchor="end" font-size="10" fill="currentColor">3</text>
  <text x="42" y="140" font-size="11" fill="currentColor" transform="rotate(-90 42 140)">eigenvalue</text>
  <text x="90" y="235" text-anchor="middle" font-size="10" fill="currentColor">1</text>
  <text x="385" y="235" text-anchor="middle" font-size="10" fill="currentColor">120</text>
  <text x="680" y="235" text-anchor="middle" font-size="10" fill="currentColor">241</text>
  <text x="385" y="249" text-anchor="middle" font-size="11" fill="currentColor">eigenvalue rank (ascending), P = 241 parameters</text>
  <polyline fill="none" stroke="#ef4444" stroke-width="2" points="90,212 100,204 110,201 120,198 129,194 139,183 149,176 159,174 169,173 178,172 188,171 198,170 208,169 218,169 228,168 238,168 247,167 257,167 267,167 277,166 287,166 296,166 306,166 316,166 326,166 336,165 346,165 356,165 365,165 375,165 385,165 395,165 405,165 414,165 424,165 434,165 444,165 454,165 464,165 474,165 483,165 493,165 503,164 513,164 523,164 532,164 542,164 552,164 562,164 572,164 582,163 592,163 601,162 611,149 621,145 631,139 641,131 650,127 660,118 670,95 680,59"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="90,165 150,165 250,165 350,165 450,165 550,165 572,164 592,164 611,164 631,163 650,162 660,140 670,107 680,96"/>
  <text x="150" y="196" font-size="11" fill="#ef4444">at init: 51% negative, spread [-1.35, 3.07]</text>
  <text x="200" y="152" font-size="11" fill="#3b82f6">trained: flat bulk, min eigenvalue -0.0006</text>
  <text x="380" y="272" text-anchor="middle" font-size="11" fill="currentColor">Small tanh network, seed 31415, exact Hessian by finite differences on the analytic gradient. At init the</text>
  <text x="380" y="286" text-anchor="middle" font-size="11" fill="currentColor">index is 0.51 (a generic saddle, as the semicircle law predicts); after training 78% of directions are flat.</text>
</svg>

If the figure does not render: two sorted-eigenvalue curves over rank $1$ to $241$. The initialization curve rises from $-1.35$, crosses zero near rank $123$, and climbs to $3.07$. The trained curve is indistinguishable from zero until about rank $220$, then rises through a handful of outliers to $2.00$; its minimum is $-0.0006$.

### 12.6.3 Why high-dimensional landscapes are more navigable

One more idea, and it is the constructive one. In one dimension, a function with two local minima traps a descent method in whichever basin it starts in — there is no way around a barrier, because there is only one direction. In $d$ dimensions, escaping requires only *one* descent direction out of $d$, and there are $d$ of them to choose from. Chapter 4's covering-number intuition and §12.1.2's near-orthogonality combine: a high-dimensional point has exponentially many nearly-orthogonal directions to try, so a "barrier" would have to block all of them simultaneously. Barriers are codimension-$1$ objects and they are cheap to walk around.

The precise version of this that is a theorem: at a strict saddle, gradient descent with random perturbation escapes in $O(\mathrm{poly}(1/\epsilon)\log d)$ iterations — the dimension enters only logarithmically (Jin et al.) **[T]**. The imprecise version, which is folklore but a good mental model **[C]**: overparameterization does not make the landscape convex, it makes the *level sets connected*, so that descent can route around obstacles rather than tunnel through them.

Two common misapplications to guard against in your own writing. First, people cite "high-dimensional landscapes have no bad minima" to justify not tuning initialization or learning rate — but the failures you actually hit in practice (divergence, dead ReLUs, loss plateaus, rank collapse in attention) are *conditioning* failures, governed by §12.5's spectra, not local-minimum failures. Second, people cite "most critical points are saddles" to explain why loss decreases; but for most of training you are nowhere near a critical point, and the relevant quantity is the local curvature spectrum (which sets the maximum stable learning rate at roughly $2/\lambda_{\max}$), not the census of critical points.

> **Common Confusion 12.7.**
> *"Deep learning works because high-dimensional loss landscapes have no bad local minima — that's a theorem, isn't it?"*
>
> No. Assembling the labels above: the "no bad local minima" theorems are about deep **linear** networks, or about the NTK regime at widths nobody uses, or about structured non-neural problems like matrix completion. Meanwhile there are *proofs* that two-layer ReLU networks have spurious local minima (Safran–Shamir). The semicircle-law argument about saddles is a theorem about isotropic Gaussian random fields, and a neural network loss is emphatically not one: it has enormous symmetry groups (permuting hidden units, rescaling ReLU layers), which produce whole manifolds of critical points and correlations between Hessian entries that Wigner's hypotheses exclude.
>
> What you can defensibly say: *empirically*, the Hessian at trained solutions has a flat bulk with few outliers and negligible negative curvature; *empirically*, independently trained solutions are often connected by low-loss paths; and *provably*, escaping strict saddles is easy in high dimension. If you write "it is known that deep networks have no bad local minima" in a paper, a careful reviewer should and will ask which theorem you mean.

> **Self-check 12.6.**
> 1. He initialization sets $\mathbb{E}\|Wh\|^2=\|h\|^2$. Which feature of the singular value spectrum does this pin down, and which does it leave free?
> 2. Why is exact dynamical isometry impossible for ReLU networks even with orthogonal weights?
> 3. At initialization the index of the Hessian was $0.51$; after training it was $0.108$ but with $|\lambda_{\min}|/\lambda_{\max}=3\times10^{-4}$. Why is the second column better described as "flat" than as "saddle"?
> 4. Give one claim from this section that is a theorem about a model nobody trains, and say what the gap to practice is.

## 12.7 Covariance estimation in high dimensions

The sample covariance matrix is the most-used estimator in statistics, and in the regime $d\sim n$ it is spectrally catastrophic. This section says how catastrophic, why, and what to do.

Let $X_1,\dots,X_n$ be i.i.d. mean-zero in $\mathbb{R}^d$ with covariance $\Sigma$, and let
$$\hat\Sigma := \frac1n\sum_{i=1}^n X_iX_i^\top .$$

Every entry of $\hat\Sigma$ is an unbiased estimator of the corresponding entry of $\Sigma$: $\mathbb{E}\hat\Sigma=\Sigma$ exactly. That is a true statement and it is nearly worthless, because everything downstream — whitening, Mahalanobis distance, discriminant analysis, natural gradient — depends on $\hat\Sigma^{-1}$ or on $\hat\Sigma$'s eigenvectors, and those are highly nonlinear in the entries.

**Proposition 12.26 (spectral error of the sample covariance).** Suppose $X_i\sim\mathcal{N}(0,\Sigma)$ and $\gamma=d/n$. Then:

1. *(Lower bound, from MP.)* If $\Sigma=I_d$ and $d/n\to\gamma\in(0,1]$, then almost surely
$$\|\hat\Sigma-I_d\|_{\text{op}}\ \longrightarrow\ \max\big\{(1+\sqrt\gamma)^2-1,\ 1-(1-\sqrt\gamma)^2\big\}\ =\ 2\sqrt\gamma+\gamma .$$
2. *(Upper bound, from Theorem 12.23.)* For every $s\ge0$, with probability at least $1-2e^{-s^2/2}$,
$$\|\hat\Sigma-\Sigma\|_{\text{op}}\ \le\ \|\Sigma\|_{\text{op}}\,C'\Big(\sqrt{\tfrac dn}+\tfrac dn+\tfrac{s}{\sqrt n}+\tfrac{s^2}{n}\Big)$$
for an absolute constant $C'$.

*Proof of (1).* Write $X_i$ as rows of $X\in\mathbb{R}^{n\times d}$ with i.i.d. $\mathcal{N}(0,1)$ entries, so $\hat\Sigma=\frac1nX^\top X$. The eigenvalues of $\hat\Sigma$ are the squared singular values of $X/\sqrt n$, which by Theorem 12.19 fill $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$, with the extremes attained in the limit by Theorem 12.20. Since $\hat\Sigma-I$ is symmetric, $\|\hat\Sigma-I\|_{\text{op}}=\max_i|\lambda_i(\hat\Sigma)-1|$, and the maximum over the MP support of $|x-1|$ is attained at one of the endpoints, giving $\max\{2\sqrt\gamma+\gamma,\,2\sqrt\gamma-\gamma\}=2\sqrt\gamma+\gamma$. $\square$

*Proof sketch of (2).* Reduce to $\Sigma=I$ by writing $X_i=\Sigma^{1/2}Z_i$ and using $\|\Sigma^{1/2}A\Sigma^{1/2}\|\le\|\Sigma\|\|A\|$. For $\Sigma=I$, $\hat\Sigma-I=\frac1n(B^\top B - nI)$ with $B$ Gaussian, and $\|\frac1nB^\top B-I\|\le\max\{|s_{\max}^2/n-1|,|s_{\min}^2/n-1|\}$; apply Theorem 12.23 to $s_{\max}$ and a matching lower-tail bound to $s_{\min}$, then expand $(1\pm u)^2-1$. $\square$

Part (1) is the sentence to remember: **at $d=n$ the operator-norm error is $3$, when the truth has norm $1$.** The estimator is not slightly off; it is wrong at the scale of the object being estimated. And this happens with Gaussian data, no outliers, no model misspecification, no heavy tails. It is pure dimension counting: $\hat\Sigma$ has $\binom{d+1}2\approx d^2/2$ free parameters and we gave it $nd$ numbers, so at $d\sim n$ we have about two observations per parameter.

> **Worked Example 12.21 (watching the sample covariance fail).**
> $X_i\sim\mathcal{N}(0,I_d)$, so the truth is the identity. `default_rng(99)`. Ledoit–Wolf shrinks toward $\bar\mu I$ with $\bar\mu=\mathrm{tr}(\hat\Sigma)/d$, using the estimated optimal intensity $\hat\rho$ (Definition 12.27 below).
>
> | $d$ | $n$ | $\gamma$ | $\lambda_{\min}$ | $\lambda_{\max}$ | MP $[a,b]$ | $\|\hat\Sigma-I\|_{\text{op}}$ | $\|\hat\Sigma_{\text{LW}}-I\|_{\text{op}}$ | $\hat\rho$ |
> |---|---|---|---|---|---|---|---|---|
> | 100 | 1000 | 0.10 | 0.4810 | 1.6857 | $[0.4675,1.7325]$ | 0.6857 | 0.0021 | 1.000 |
> | 100 | 200 | 0.50 | 0.0918 | 2.7639 | $[0.0858,2.9142]$ | 1.7639 | 0.0504 | 0.964 |
> | 100 | 100 | 1.00 | $-4\times10^{-16}$ | 3.8597 | $[0,4]$ | 2.8597 | 0.0527 | 0.974 |
> | 200 | 100 | 2.00 | $-2\times10^{-16}$ | 5.7286 | $[0,5.8284]$ | 4.7286 | 0.1092 | 0.975 |
>
> The empirical eigenvalue extremes sit just inside the Marchenko–Pastur support in every row — the theory is not asymptotic decoration, it is what you measure at $n=100$. At $\gamma=1$ and $\gamma=2$ the smallest eigenvalue is numerically zero, so $\hat\Sigma$ is singular and $\hat\Sigma^{-1}$ does not exist; the condition number computed naively comes out around $10^{12}$, which is the reciprocal of machine epsilon rather than a property of the data.
>
> Ledoit–Wolf is dramatically better here, but this experiment is *rigged in its favour*: the true covariance is exactly the shrinkage target, so $\hat\rho\approx1$ is correct and the estimator is nearly perfect. The next example removes the rigging.

**Definition 12.27 (linear shrinkage; Ledoit–Wolf).** Let $\bar\mu:=\mathrm{tr}(\hat\Sigma)/d$ and define, for $\rho\in[0,1]$,
$$\hat\Sigma_\rho\ :=\ (1-\rho)\,\hat\Sigma\ +\ \rho\,\bar\mu\, I_d .$$
The *oracle* shrinkage intensity is the minimizer $\rho^\star$ of the expected squared Frobenius error $\mathbb{E}\|\hat\Sigma_\rho-\Sigma\|_F^2$. The Ledoit–Wolf estimator uses a consistent estimate $\hat\rho$ of $\rho^\star$ computed from the data.

**Proposition 12.28 (the oracle intensity).** Write $\mu := \mathrm{tr}(\Sigma)/d$, $\alpha^2:=\|\Sigma-\mu I\|_F^2$ and $\beta^2:=\mathbb{E}\|\hat\Sigma-\Sigma\|_F^2$. Then
$$\rho^\star = \frac{\beta^2}{\alpha^2+\beta^2},\qquad \mathbb{E}\|\hat\Sigma_{\rho^\star}-\Sigma\|_F^2 = \frac{\alpha^2\beta^2}{\alpha^2+\beta^2}\ \le\ \min(\alpha^2,\beta^2).$$

*Proof.* Treat $\bar\mu$ as $\mu$ (the difference is $O(1/\sqrt{nd})$ and does not affect the leading-order answer). Decompose
$$\hat\Sigma_\rho-\Sigma = (1-\rho)(\hat\Sigma-\Sigma) - \rho(\Sigma-\mu I).$$
The two terms are orthogonal in expectation under the Frobenius inner product, since $\mathbb{E}\langle\hat\Sigma-\Sigma,\ \Sigma-\mu I\rangle_F = \langle\mathbb{E}\hat\Sigma-\Sigma,\ \Sigma-\mu I\rangle_F=0$ by unbiasedness. Hence
$$\mathbb{E}\|\hat\Sigma_\rho-\Sigma\|_F^2 = (1-\rho)^2\beta^2+\rho^2\alpha^2 .$$
This is a strictly convex quadratic in $\rho$; setting the derivative $-2(1-\rho)\beta^2+2\rho\alpha^2$ to zero gives $\rho^\star=\beta^2/(\alpha^2+\beta^2)$, and substituting gives the stated risk. $\square$

Two readings of Proposition 12.28. Mathematically it is the same bias–variance trade that produced James–Stein in Chapter 8: $\beta^2$ is the variance of the unbiased estimator, $\alpha^2$ is the squared bias you incur by collapsing to the target, and the optimum interpolates in proportion. Practically, $\beta^2$ grows like $d^2/n$ while $\alpha^2$ is a fixed property of $\Sigma$, so $\rho^\star\to1$ as $d/n$ grows: **the more high-dimensional your problem, the more you should shrink.** That is the sentence that makes shrinkage non-optional rather than a nicety.

> **Worked Example 12.22 (shrinkage when the target is wrong).**
> Now $\Sigma$ has eigenvalues log-spaced on $[0.2,5]$ (true condition number $25$) in a random orthonormal basis, $d=200$, $n=250$ so $\gamma=0.8$, $20$ independent replications, `default_rng(4242)`.
>
> | quantity | value |
> |---|---|
> | $\|\hat\Sigma-\Sigma\|_{\text{op}}$ | $4.6588$ (sd $0.2679$) |
> | $\|\hat\Sigma_{\text{LW}}-\Sigma\|_{\text{op}}$ | $2.7220$ (sd $0.0498$) |
> | mean estimated $\hat\rho$ | $0.5162$ |
> | median $\mathrm{cond}(\hat\Sigma^{-1}\Sigma)$ | $2.97\times10^{2}$ |
> | median $\mathrm{cond}(\hat\Sigma_{\text{LW}}^{-1}\Sigma)$ | $13.77$ |
> | true $\mathrm{cond}(\Sigma)$ | $25.00$ |
>
> Shrinkage halves the operator-norm error and reduces its variability by a factor of five, even though the shrinkage target ($\bar\mu I$) is now genuinely wrong. The last three rows are the ones that matter for practice. Read $\mathrm{cond}(\hat\Sigma^{-1}\Sigma)$ as "how badly does whitening by $\hat\Sigma$ fail": if $\hat\Sigma=\Sigma$ this quantity is $1$. Sample covariance gives $297$ — whitening with it *creates* a condition number an order of magnitude worse than the original $25$, by inflating the directions where $\hat\Sigma$ happened to underestimate. Shrinkage gives $13.8$, i.e. it genuinely improves conditioning relative to the raw data.
>
> This is the failure mode behind a whole family of practical bugs. If you whiten features using a sample covariance estimated from a batch smaller than the feature count, you are amplifying the noisiest directions by a factor set by MP's lower edge, not by anything in your data.

### 12.7.1 The spiked model: when can PCA see a signal?

Everything so far described the *null* case — a covariance with no structure, or structure treated as a nuisance. But the reason you compute eigenvectors at all is usually to *find* structure: a dominant direction of variation, a factor, a component worth keeping. So the operational question is the converse of Marchenko–Pastur: if the true covariance contains one genuinely large eigenvalue, when does the sample spectrum reveal it, and how well does the sample eigenvector point at the true one? The spiked covariance model answers both questions exactly, and the answer is a sharp phase transition, not a gradual degradation.

The model: $\Sigma = I_d + (\lambda-1)\,vv^\top$ for a unit vector $v$ and $\lambda>1$ — an identity "sea" plus one planted spike of size $\lambda$ along direction $v$. Estimate $\hat\Sigma$ from $n$ Gaussian samples, let $\hat\lambda_1$ be its top eigenvalue and $\hat v$ the corresponding unit eigenvector, and let $\gamma=d/n$ as always.

**Theorem 12.29 (BBP phase transition; eigenvector overlap).** In the spiked model with $d/n\to\gamma\in(0,1]$, almost surely:

1. *(Subcritical.)* If $\lambda\le1+\sqrt\gamma$, then $\hat\lambda_1\to(1+\sqrt\gamma)^2$ — the top sample eigenvalue sticks to the Marchenko–Pastur edge, exactly as if there were no spike — and $|\langle\hat v,v\rangle|^2\to0$: the sample eigenvector is asymptotically *orthogonal* to the truth, carrying no information about it.
2. *(Supercritical.)* If $\lambda>1+\sqrt\gamma$, then
$$\hat\lambda_1\ \to\ \lambda\Big(1+\frac{\gamma}{\lambda-1}\Big)\ >\ (1+\sqrt\gamma)^2,\qquad |\langle\hat v,v\rangle|^2\ \to\ \frac{1-\gamma/(\lambda-1)^2}{1+\gamma/(\lambda-1)}\ \in(0,1).$$

We do not prove this (see Results used but not proved); the statement is what you need. Three readings. First, **detectability has a hard threshold**: a spike of size $\lambda=1+\sqrt\gamma$ is invisible, not merely hard to see — below the threshold the top sample eigenvalue and eigenvector are statistically indistinguishable from pure noise, and no amount of staring at the spectrum helps. At $\gamma=1$ (one sample per dimension) the threshold is $\lambda=2$: a direction with *twice* the average variance is undetectable by PCA. Second, even above the threshold the sample eigenvalue is **biased upward** — it overshoots $\lambda$ by the factor $1+\gamma/(\lambda-1)$ — which is precisely the eigenvalue inflation that nonlinear shrinkage (Common Confusion 12.8) is built to undo. Third, the eigenvector recovers only **partially**: the overlap formula rises from $0$ at the threshold toward $1$ as $\lambda/(1+\sqrt\gamma)$ grows, so "PCA found the component" is a quantitative statement with a computable accuracy, not a binary one.

> **Worked Example 12.23 (watching the phase transition).**
> $d=1000$, $n=2000$ (so $\gamma=0.5$, threshold $1+\sqrt{0.5}=1.7071$, MP edge $(1+\sqrt{0.5})^2=2.9142$), spike along $v=e_1$, $5$ replications per row, `default_rng(2026)`:
>
> | $\lambda$ | $\hat\lambda_1$ observed | predicted | $\vert\langle\hat v,v\rangle\vert^2$ observed | predicted |
> |---|---|---|---|---|
> | 1.20 | 2.9026 | 2.9142 (edge) | 0.005 | 0 |
> | 1.50 | 2.8975 | 2.9142 (edge) | 0.024 | 0 |
> | 2.00 | 2.9905 | 3.0000 | 0.287 | 0.3333 |
> | 3.00 | 3.7815 | 3.7500 | 0.710 | 0.7000 |
> | 5.00 | 5.5794 | 5.6250 | 0.857 | 0.8611 |
>
> The rows tell the whole story. At $\lambda=1.5$ — a $50\%$ variance excess along a genuine direction — the top sample eigenvalue is *below* the null edge and the eigenvector overlap is $2.4\%$: the spike has been swallowed by the bulk. At $\lambda=2$, just above threshold, detection succeeds but the recovered direction explains only a third of the truth. Not until $\lambda=5$ does the eigenvector become substantially correct, and even then the eigenvalue overreports by $12\%$.
>
> Where this bites in ML. (i) *PCA preprocessing:* "keep components above the noise floor" only works for components above $1+\sqrt\gamma$ times the noise, and their reported variances need the de-biasing correction. (ii) *Probing and interpretability:* fitting a direction (a linear probe, a concept vector) from $n$ activations of dimension $d$ inherits exactly this threshold; with $d\sim n$, weak-but-real features are provably invisible to spectral methods. (iii) *The shrinkage caveat made precise:* Common Confusion 12.8 warned that linear shrinkage over-shrinks large eigenvalues; Theorem 12.29 says by how much a large sample eigenvalue should instead be corrected ($\hat\lambda_1\mapsto$ the $\lambda$ solving the displayed equation), which is the core of Ledoit–Wolf's nonlinear shrinkage and of Donoho's optimal-shrinkage line of work.

### 12.7.2 Where this bites in machine learning

- **Whitening and PCA preprocessing.** Whitening by $\hat\Sigma^{-1/2}$ with $d\gtrsim n$ amplifies the smallest sample eigenvalues, which by MP are near-zero *artifacts*. Standard fixes: keep only the top-$k$ components, add a ridge $\hat\Sigma+\epsilon I$, or shrink. Batch normalization sidesteps the issue entirely by whitening only the diagonal, which is why it is robust and why full-covariance "decorrelated BN" variants need careful regularization.
- **Gaussian discriminant analysis / LDA / QDA.** The discriminant depends on $\hat\Sigma^{-1}$. QDA estimates one covariance per class, so the effective $n$ per estimate is the class count — this is why QDA is unusable and LDA marginal at moderate $d$, and why regularized discriminant analysis (a shrinkage estimator by another name) was invented.
- **Mahalanobis distance for out-of-distribution detection.** A popular OOD score is $\min_c(z-\mu_c)^\top\hat\Sigma^{-1}(z-\mu_c)$ on penultimate features, where $d$ can be $2048$ and the per-class $n$ a few hundred. The reported sensitivity of these methods to the choice of shrinkage/ridge parameter is Proposition 12.26 in action, not an implementation detail.
- **Natural gradient and second-order optimizers.** Natural gradient needs $F^{-1}$ where $F$ is the Fisher information, itself a covariance of gradients estimated from a minibatch. With $d$ in the millions and a batch of a few hundred, $\hat F$ has rank at most the batch size. This is why every practical method — K-FAC, Shampoo, Adam's diagonal, SOAP — imposes *structure* (Kronecker factorization, block-diagonality, diagonality) and adds damping $\hat F+\epsilon I$. The damping is not numerical hygiene; it is the shrinkage of Definition 12.27, and its magnitude is doing statistical work.
- **Covariance-based exploration and Thompson sampling.** Linear bandit algorithms maintain $(\lambda I + \sum_t x_tx_t^\top)^{-1}$; the $\lambda I$ is again shrinkage, and the regret bounds depend on it in exactly the way Proposition 12.28 suggests.

> **Common Confusion 12.8.**
> *"The sample covariance is unbiased, so it's the right estimator — any bias I introduce by shrinking makes things worse."*
>
> Unbiasedness is a statement about $\mathbb{E}\hat\Sigma$, i.e. about each entry averaged over datasets. Nothing about it constrains the *spectrum* of any single realization, and the spectrum is what you use. Worked Example 12.21 shows an unbiased estimator of $I_{100}$ from $n=100$ samples whose eigenvalues run from $0$ to $3.86$. Averaging a thousand such matrices would indeed give something close to $I$; you have one.
>
> The deeper point, which is Chapter 8's James–Stein point restated: in high dimension, unbiasedness is a bad target. Proposition 12.28 exhibits a *biased* estimator with strictly smaller risk than the unbiased one for every $\Sigma$ with $\alpha^2,\beta^2>0$, since $\frac{\alpha^2\beta^2}{\alpha^2+\beta^2}<\beta^2$. Dimension is what converts "biased" from a defect into a design choice.
>
> One caveat in the other direction. Linear shrinkage toward $\bar\mu I$ is optimal only within the class of linear combinations of $\hat\Sigma$ and $I$. If $\Sigma$ has a few very large eigenvalues (a factor structure, which is typical for financial returns and for neural network features), pulling everything toward a common mean shrinks the large eigenvalues too much. Nonlinear shrinkage — applying a separate, MP-derived correction to each sample eigenvalue while keeping the sample eigenvectors — does better, and is what Ledoit and Wolf's later work provides.

> **Self-check 12.7.**
> 1. At $d/n=0.25$, what does Proposition 12.26(1) predict for $\|\hat\Sigma-I\|_{\text{op}}$? What is the interval the sample eigenvalues fill?
> 2. Why does unbiasedness of $\hat\Sigma$ fail to imply that $\hat\Sigma^{-1}$ is a good estimator of $\Sigma^{-1}$?
> 3. In Proposition 12.28, what happens to $\rho^\star$ as $n\to\infty$ with $d$ fixed, and why is that reassuring?
> 4. Name two optimizers whose damping term is a shrinkage estimator in disguise.

## 12.8 The double-edged sword

Every result in this chapter is the same result, seen from two sides.

The **statement** is: a smooth function of many independent random variables is nearly deterministic. Theorem 12.5 for the norm, Theorem 12.7 for any Lipschitz function, Theorem 12.12 for convex Lipschitz functions of bounded variables, Theorem 12.19 for the whole spectrum of a random matrix. In each case a quantity that could a priori fluctuate wildly turns out to sit within $O(1)$ of a deterministic value, with Gaussian tails and no dependence on dimension.

**Side one: this is what makes learning possible.** Consider what would happen if it were false.

- The empirical risk $\hat R_n(h)=\frac1n\sum_i\ell(h(X_i),Y_i)$ is an average of $n$ independent terms. Concentration (Hoeffding, Bernstein, Ch. 4) says it is within $O(\sqrt{\log(1/\delta)/n})$ of the true risk $R(h)$. Without concentration, minimizing training loss would tell you nothing about test loss — there would be no such thing as generalization, and no reason to fit models to data at all.
- Uniform convergence over a hypothesis class combines concentration with the union bound, giving $\sup_h|\hat R_n(h)-R(h)|\lesssim\sqrt{\log|\mathcal{H}|/n}$ for finite classes and Rademacher-complexity analogues otherwise (Ch. 4). The $\log$ is cheap precisely because concentration gives *exponential* tails.
- A minibatch gradient is an average of $B$ per-example gradients. SGD works because that average concentrates around the full gradient — the assumption $\mathbb{E}[g_t\mid\mathcal{F}_t]=\nabla F(\theta_t)$ from Chapter 1 gives unbiasedness, and concentration gives the control on fluctuations that convergence proofs need.
- Monte Carlo integration, importance sampling, the reparameterization trick's gradient estimator, and every evaluation metric you compute on a held-out set: all rest on averages concentrating.

**Side two: this is what makes geometry counterintuitive.** The exact same theorem, applied to $\|X-Y\|$ rather than to $\frac1n\sum\ell_i$, says all pairwise distances are nearly equal (Theorem 12.13). So:

- Nearest-neighbour queries lose their meaning for isotropic data, and with them $k$-NN, kernel bandwidth selection, density estimation by local averaging, and anything else that needs "close" to mean something (§12.3).
- Volume is not where the density is (§12.1.3), so mode-seeking and mass-seeking diverge, and MAP is a poor summary of a high-dimensional posterior.
- A random matrix that is correct on average is badly conditioned (§12.5, §12.6), and a covariance estimate that is unbiased is spectrally wrong (§12.7).

The duality is not a coincidence or a cute framing; it is arithmetic. Concentration says a quantity's *relative* fluctuation shrinks like $1/\sqrt{d}$ or $1/\sqrt n$. When that quantity is an estimate you want to trust, small relative fluctuation is exactly the good news. When the quantity is a distance whose *differences* carry your signal, small relative fluctuation means the differences have been crushed relative to the common value — and the signal is in the differences.

The practical resolution, and the thing to carry out of this chapter: **you get to choose which side you are on, by choosing the representation.** Worked Examples 12.11 and 12.12 are the proof. The same ambient dimension $d=10^4$ gives dead contrast for isotropic data and healthy contrast for data on a five-dimensional manifold. Concentration in the useful direction (averages over samples) is guaranteed by $n$; concentration in the harmful direction (distances between points) is governed by intrinsic dimension, which is a property of your features. Representation learning, dimension reduction, and metric learning are all, viewed this way, the engineering of intrinsic dimension — moving your data from the bad side of the sword to the good one.

> **Self-check 12.8.**
> 1. Name one result from this chapter that is good news for learning and one that is bad news for geometry, and explain why they are the same result.
> 2. Concentration makes empirical risk a reliable estimate of true risk. Which quantity in that sentence plays the role of $d$ in §12.2, and which plays the role of $n$?
> 3. Your retrieval system's recall is poor at $d=1536$. Give two diagnostics from this chapter that distinguish "the representation has too high an intrinsic dimension" from "the index is badly tuned."
> 4. Why is it accurate to say that representation learning is the engineering of intrinsic dimension?

## 12.9 Results used but not proved

This table is complete: every result invoked in this chapter without a proof appears here. Results from earlier chapters of this tutorial are cited by name and chapter, per the citation convention; results from the literature get a source.

| Result | Used in | Where to find a proof |
|---|---|---|
| Stirling / asymptotic expansion of $\log\Gamma$ | Proposition 12.1(3), Proposition 12.3 | Any analysis text; Whittaker–Watson §12.31, or Abramowitz–Stegun 6.1.41 |
| Polar-coordinate change of variables in $\mathbb{R}^d$ (Jacobian $r^{d-1}$) | Proposition 12.1 | The change-of-variables formula (Ch. 3) |
| Rotational invariance of $\mathcal{N}(0,\sigma^2I_d)$; $U=Z/\|Z\|$ is uniform on $S^{d-1}$ | Propositions 12.2, 12.3; Lemma 12.16 | The multivariate Gaussian (Ch. 5) |
| Beta representation $\chi^2_a/(\chi^2_a+\chi^2_b)\sim\mathrm{Beta}(a/2,b/2)$ | Proposition 12.2(2) | Ch. 3's change of variables applied to the Gamma density; Casella–Berger Thm 4.3.5 |
| *Sharp* angle tail $\Pr(\vert T\vert\ge s)\le2e^{-ds^2/2}$ (we prove only the weaker $3e^{-(d-1)s^2/8}$) | remark after Proposition 12.2 | K. Ball, *An Elementary Introduction to Modern Convex Geometry*, Lecture 2; Blum–Hopcroft–Kannan §2.5 |
| Closed form $\int_{-1}^1\frac{\sqrt{1-y^2}}{c+2\kappa y}dy=\frac{\pi}{4\kappa^2}(c-\sqrt{c^2-4\kappa^2})$ | Solution A13(a) | Gradshteyn–Ryzhik 3.644; or contour integration |
| Zero sets of non-zero polynomials on $\mathbb{R}^m$ are Lebesgue-null | Solution A17(b) | Standard; e.g. Caron–Traynor, "The zero set of a polynomial," 2005 |
| Chernoff bounding method | Lemma 12.4, Theorem 12.23 | Concentration inequalities (Ch. 4) |
| Sub-Gaussian norm; squaring a sub-Gaussian gives a sub-exponential variable | Theorem 12.6, Theorem 12.13 | Concentration inequalities (Ch. 4); Vershynin §2.7 |
| Bernstein's inequality for sub-exponential summands | Theorem 12.6 | Concentration inequalities (Ch. 4); Vershynin Thm 2.8.1 |
| Sums of independent sub-Gaussians are sub-Gaussian | Theorem 12.13 | Concentration inequalities (Ch. 4) |
| Standard Gaussian tail $\Pr(G\ge r)\le e^{-r^2/2}$ | Theorem 12.23, Step 1 | Concentration inequalities (Ch. 4) |
| Covering number of $S^{d-1}$: $N(\varepsilon)\le(1+2/\varepsilon)^d$ | Theorem 12.23 | Covering numbers (Ch. 4); Vershynin Cor 4.2.13 |
| Gaussian Lipschitz concentration (Borell; Tsirelson–Ibragimov–Sudakov) | Theorem 12.7, Corollary 12.8, Worked Example 12.8 | Boucheron–Lugosi–Massart Thm 5.6; Ledoux *The Concentration of Measure Phenomenon* Ch. 1–2 |
| Gaussian Poincaré inequality $\operatorname{Var}(f(Z))\le\mathbb{E}\|\nabla f\|^2$ | Proposition 12.3 (alternative route) | Boucheron–Lugosi–Massart Thm 3.20 |
| Gaussian isoperimetric inequality | Theorem 12.10 | Ledoux Ch. 2; Bogachev *Gaussian Measures* Thm 4.3.3 |
| Lévy's lemma on the sphere (constants in the exponent vary by source — $d-1$ here, $d-2$ or $cd$ elsewhere; the $\Theta(d)$ rate is what matters) | Corollary 12.11 | Ledoux Ch. 2; Matoušek *Lectures on Discrete Geometry* §14.2 |
| BBP phase transition and eigenvector overlap in the spiked model | Theorem 12.29, Worked Example 12.23 | Baik–Ben Arous–Péché (2005); Paul (2007) for the overlap; Benaych-Georges–Nadakuditi (2011) for the general framework |
| Talagrand's convex distance inequality | Theorem 12.12 | Talagrand (1995); Boucheron–Lugosi–Massart Ch. 8 |
| $\Phi^{-1}(p_c(\cdot))$ is $1/\sigma$-Lipschitz (randomized smoothing certificate) | Worked Example 12.8 | Cohen–Rosenfeld–Kolter (2019), Lemma 1; follows from Theorem 12.10 |
| Marchenko–Pastur law | Theorem 12.19, Propositions 12.25, 12.26 | Bai–Silverstein *Spectral Analysis of Large Dimensional Random Matrices* Thm 3.6 |
| Bai–Yin theorem (no outliers; edge limits) | Theorem 12.20, Proposition 12.26 | Bai–Yin (1993); Bai–Silverstein Thm 5.11 |
| Wigner semicircle law | Theorem 12.21 | Anderson–Guionnet–Zeitouni Thm 2.1.1 |
| Tracy–Widom edge fluctuations (mentioned only) | Remark after Theorem 12.20 | Johnstone (2001) |
| Smallest singular value of a square random matrix, $s_{\min}\asymp n^{-1/2}$ | Worked Example 12.16, Common Confusion 12.6 | Rudelson–Vershynin (2008) |
| Matrix Bernstein inequality | Theorem 12.24 | Tropp, *An Introduction to Matrix Concentration Inequalities* (2015), Thm 6.1.1 |
| Consistency of the Ledoit–Wolf estimate $\hat\rho$ of $\rho^\star$ | Definition 12.27, Worked Examples 12.21, 12.22 | Ledoit–Wolf (2004), Thms 2–3 |
| Optimality of JL: $k=\Omega(\varepsilon^{-2}\log n)$ is necessary | Common Confusion 12.5 | Larsen–Nelson (2017); Alon–Klartag (2017) |
| Bochner's theorem (random Fourier features) | §12.4.3 | Rudin *Fourier Analysis on Groups* §1.4.3 |
| SimHash collision identity $\Pr=1-\Theta/\pi$ | Worked Example 12.15 | Goemans–Williamson (1995) rounding analysis; Charikar (2002) |
| Deep linear networks: all local minima are global | §12.6.2 | Kawaguchi (2016), Thm 2.3; Baldi–Hornik (1989) |
| Strict saddle property; perturbed GD escapes saddles in $O(\log d)$ | §12.6.2, §12.6.3 | Ge–Huang–Jin–Yuan (2015); Jin et al. (2017), Thm 3 |
| Existence of spurious local minima in two-layer ReLU networks | §12.6.2, Common Confusion 12.7 | Safran–Shamir (2018); Yun–Sra–Jadbabaie (2019) |
| NTK-regime global convergence | §12.6.2 | Jacot–Gabriel–Hongler (2018); Du et al. (2019); Allen-Zhu–Li–Song (2019) |
| Deep linear network training-time separation for orthogonal vs Gaussian init | Worked Example 12.18 | Saxe–McClelland–Ganguli (2014) |
| Limiting Jacobian spectra of random deep networks; ReLU cannot achieve isometry | Worked Example 12.18 | Pennington–Schoenholz–Ganguli (2017, 2018) |
| Intrinsic dimension estimators (MLE, TwoNN) and ImageNet estimates | §12.3.2, Common Confusion 12.4 | Levina–Bickel (2004); Facco et al. (2017); Pope et al. (2021) |
| Union bound | Theorems 12.13, 12.17, 12.23 | Proposition 1.8(2) — proved in Ch. 1 |
| Definition of support | §12.3.2 | Definition 1.17 — Ch. 1 |

## 12.10 Chapter summary

1. **Volume abandons the interior, and then collapses.** For any fixed $\varepsilon$, the fraction of a $d$-ball's volume within $\varepsilon$ of the surface is $1-(1-\varepsilon)^d\to1$; at $d=10^4$ a shell of relative thickness $6.9\times10^{-5}$ holds half the volume. Meanwhile $V_d(1)$ itself peaks at $d=5$ and then decays like $e^{-(d/2)\log d}$, and the ball inscribed in the unit cube occupies a fraction $1.9\times10^{-70}$ of it at $d=100$ — so box-based space partitioning cannot approximate ball queries. Nothing is "small" in an absolute sense; what shrinks is volume measured against a fixed unit, and the right response is to rescale radii, not to distrust the arithmetic.

2. **Random directions are nearly orthogonal.** With $T=\cos\Theta$ for independent uniform directions, $T^2\sim\mathrm{Beta}(\tfrac12,\tfrac{d-1}2)$, $\operatorname{Var}(T)=1/d$, and $\Pr(|T|\ge s)\le3e^{-(d-1)s^2/8}$ (proved; the sharp constant is $2e^{-ds^2/2}$). At $d=10^4$ two random directions are within $5^\circ$ of perpendicular with probability $1-2.5\times10^{-18}$. This is the geometric basis of superposition in interpretability and of the effectiveness of random projections.
3. **The Gaussian is a soap bubble, not a blob.** $\|Z\|$ has mean $\approx\sqrt{d-\tfrac12}$ and standard deviation $\to1/\sqrt2$, so the annulus $\sqrt d\pm1$ holds $84.3\%$ of the mass at essentially every dimension while the unit ball at the mode holds $10^{-1285}$ at $d=1000$. Density and probability part company: the mode is in a desert. This is why `slerp` beats linear interpolation in latent space and why the truncation trick trades diversity for fidelity.
4. **Chi-square tails have two regimes.** $\Pr(\chi^2_d\ge d+2\sqrt{dt}+2t)\le e^{-t}$ — sub-Gaussian ($2\sqrt{dt}$) for small deviations, sub-exponential ($2t$) for large ones. The bound is loose by a factor of $6$–$23$ over the useful range, uniformly in $d$, which is exactly the property that makes it usable inside proofs.
5. **Lipschitz concentration is dimension-free.** For $L$-Lipschitz $f$ and $Z\sim\mathcal{N}(0,I_d)$, $\Pr(|f(Z)-\mathbb{E}f(Z)|\ge t)\le2e^{-t^2/(2L^2)}$ with no $d$ anywhere. This is a consequence of the Gaussian isoperimetric inequality, whose content is that half-spaces are the extremal sets. Randomized-smoothing robustness certificates are a direct application, and their $d$-independence is real.
6. **Concentration is about scale, not determinism.** $\operatorname{Var}(\|Z\|)\to\tfrac12$, a constant. What vanishes is the *relative* fluctuation $\mathrm{sd}/\mathbb{E}\approx1/\sqrt{2d}$. A stable gradient norm is therefore not evidence of a stable gradient direction.
7. **Distance contrast dies for isotropic data, at a rate you can compute.** With $n$ points and i.i.d. sub-Gaussian coordinates, all pairwise distances lie within $1\pm O(\sqrt{\log n/d})$ of a common value, so contrast of size $\varepsilon$ survives only while $d\lesssim c^2\log n/\varepsilon^2$, with $c\approx2.7$ measured. Never quote a single threshold dimension: at $n=10^6$ this is $d\approx100$ if you need order-one contrast and $d\approx10^4$ if $10\%$ will do. Measured: at $n=1000$, $d=10^4$, the relative contrast is $0.070$.
8. **Contrast is governed by intrinsic dimension, not ambient dimension — and that part is a theorem.** A bi-Lipschitz embedding changes ambient dimension without changing contrast beyond the factor $\beta/\alpha$ (Proposition 12.15), and a small-ball argument gives $D_{\min}\asymp n^{-1/k}$ with the *intrinsic* $k$ in the exponent, so contrast grows like $n^{1/k}$. That real data has small $k$ — the manifold hypothesis — is an *empirical* claim, not a theorem, and should be measured rather than assumed. Measured: isotropic contrast falls $4.02\to1.04$ as $d$ goes $10\to10^4$, while a $5$-dimensional manifold in the *same* $\mathbb{R}^d$ plateaus at $5.13$; the corresponding $1$-NN errors are $0.477$ (chance) versus $0.118$. Distinguish exactly-low-dimensional support from approximately-low-dimensional mass: real data is the latter, so the support may be everything while the mass is not.
9. **Johnson–Lindenstrauss.** A random Gaussian projection to $k\ge8\varepsilon^{-2}(2\log n+\log\tfrac1\delta)$ dimensions preserves all $\binom n2$ pairwise distances to within $1\pm\varepsilon$ with probability $1-\delta$, with $k$ independent of $d$. The $\varepsilon^{-2}$ comes from sub-Gaussian concentration, the $\log n$ from the union bound, the absence of $d$ from Gaussian rotational invariance. Measured conservatism: $k=256$ sufficed where the bound demanded $1243$.
10. **JL is often vacuous where you wanted it.** At $n=10^6$, $\varepsilon=0.1$ it demands $k=11053$, exceeding typical embedding dimensions. It is a data-oblivious worst-case guarantee, provably optimal in that class, which is precisely why data-aware methods (PCA, learned projections) beat it in practice.
11. **Random matrices have deterministic spectra — and signals face a detection threshold.** Marchenko–Pastur gives the limiting eigenvalue density of $\hat\Sigma$ with support $[(1-\sqrt\gamma)^2,(1+\sqrt\gamma)^2]$, $\gamma=d/n$; Bai–Yin says there are no outliers; the semicircle law covers symmetric matrices. Verified to three decimals at $n=4000$. Conversely, a planted spike of size $\lambda$ in the covariance is detectable by PCA only when $\lambda>1+\sqrt\gamma$ (BBP, Theorem 12.29); below that, the top sample eigenvector is asymptotically orthogonal to the truth, and above it the sample eigenvalue overshoots by the computable factor $1+\gamma/(\lambda-1)$. Measured at $\gamma=0.5$: a spike of $1.5$ is invisible (overlap $0.02$); a spike of $3$ is recovered with overlap $0.71$.
12. **The $\varepsilon$-net argument.** $\|B\|\le C_\varepsilon(\sqrt n+\sqrt d)+K_\varepsilon s$ with probability $1-e^{-s^2/2}$, where $C_{\varepsilon^\star}=3.022$ and $K_{\varepsilon^\star}=1.138$ at $\varepsilon^\star=0.0605$; proved from a covering bound plus a union bound plus Cauchy–Schwarz. The optimal net resolution is $\varepsilon^\star\approx0.06$ — fine, because cardinality enters logarithmically while the discretization penalty enters linearly. Sharp constants come from RMT; robust constants come from nets.
13. **Initialization: the mean gain is not the whole story.** He initialization fixes $\mathbb{E}\|Wh\|^2$, i.e. the mean square singular value, and leaves the spread entirely to Marchenko–Pastur. Measured at width $512$: Gaussian He init has condition number $2156$ with $4.7\%$ of directions attenuated below gain $0.1$; a $\sqrt2$-scaled orthogonal matrix has condition number exactly $1$ with the same mean gain. That gap is dynamical isometry.
14. **Be precise about loss landscapes.** "No bad local minima" is a theorem about deep *linear* networks and about the NTK regime; there are *proofs* that two-layer ReLU networks have spurious local minima. "Most critical points are saddles" is a theorem about isotropic Gaussian random fields, which neural losses are not. What is solid: escaping strict saddles is easy in high dimension (theorem), and trained networks empirically sit at points with a flat bulk, a few large outliers, and negligible negative curvature — measured here as index $0.51$ at init versus $|\lambda_{\min}|/\lambda_{\max}=3\times10^{-4}$ after training.
15. **Sample covariance is unbiased and spectrally wrong.** At $d=n$ with $\Sigma=I$ the operator-norm error tends to $3$ and the matrix is singular. Linear shrinkage $\hat\Sigma_\rho=(1-\rho)\hat\Sigma+\rho\bar\mu I$ with $\rho^\star=\beta^2/(\alpha^2+\beta^2)$ strictly dominates the unbiased estimator, and $\rho^\star\to1$ as $d/n$ grows. Measured at $\gamma=0.8$ with non-identity truth: operator error $4.66\to2.72$, and whitening quality $\mathrm{cond}(\hat\Sigma^{-1}\Sigma)$ improving from $297$ to $13.8$. Every damping term in K-FAC, Shampoo, Adam, and linear-bandit algorithms is this shrinkage in disguise.
16. **The duality.** Concentration is what makes empirical averages reliable — hence generalization, SGD, and Monte Carlo — and simultaneously what makes distances stop discriminating. Which side of the sword you are on is determined by the representation, because the harmful side is governed by intrinsic dimension, which is a property of your features and therefore something you can engineer.

### Callbacks

This chapter discharged three promises made earlier. Chapter 1 established that a deep generative model is a pushforward $P_\theta=(G_\theta)_\#\mathcal{N}(0,I_k)$ whose support has Lebesgue measure zero when $k<d$, and promised that the *practical* consequences of low-dimensional support would be developed later: §12.3.2 delivered that, distinguishing exactly-low-dimensional support (Chapter 1's circle generator) from approximately-low-dimensional mass (real data), and showing quantitatively that intrinsic rather than ambient dimension controls whether nearest-neighbour methods function. Chapter 1 also flagged that the union bound is the source of the $\log|\mathcal{H}|$ in generalization bounds and is tight for disjoint events; Theorem 12.17's $\log n$ is the same mechanism, and Worked Example 12.14's factor-of-five conservatism is precisely the price of pretending strongly correlated pair events are disjoint. Chapter 4 promised that covering numbers would earn their keep beyond Rademacher complexity: Theorem 12.23 is that payoff, with the whole operator-norm bound assembled from a covering number, a union bound, and Cauchy–Schwarz. Chapter 5's variance-propagation derivation of He initialization is completed here by Proposition 12.25 and Worked Example 12.18, which supply the rest of the spectrum. And Chapter 8's James–Stein lesson — that unbiasedness is a poor target in high dimension — recurs verbatim in Proposition 12.28, now for matrices rather than means.

### Before you continue

1. Without computing, say whether the fraction of a $d$-ball's volume within $1\%$ of its surface is larger at $d=100$ or at $d=1000$, and explain why the answer does not depend on which $\varepsilon$ you chose.
2. You draw $z\sim\mathcal{N}(0,I_{1024})$ a million times. Give the approximate range of $\|z\|$ you expect to see, and explain why the probability of ever seeing $\|z\|<25$ in those million draws is negligible. (Quantify "negligible" — the chapter's own discipline forbids saying "never" when you mean a computable small number.)
3. State Theorem 12.7 from memory, then say what is surprising about it in one sentence.
4. A colleague reports that her $k$-NN classifier on $2048$-dimensional CLIP features works well, and cites "the curse of dimensionality" as a reason it should not. Resolve the tension in three sentences, naming the quantity that actually matters and saying which half of the resolution is a theorem and which is an empirical claim.
5. In the Johnson–Lindenstrauss lemma, which of $n$, $d$, $\varepsilon$, $\delta$ does the target dimension $k$ depend on, and in what functional form?
6. The true covariance is $I_{500}$ and you estimate it from $n=1000$ samples. What interval will the sample eigenvalues fill, and what will the condition number be?
7. Explain why He initialization getting the mean gain right does not prevent a Gaussian-initialized square layer from having a condition number in the thousands.
8. Give one claim about deep learning loss landscapes that is a theorem, one that is empirical, and one that is conjecture.
9. State the duality of §12.8 in one sentence, and name the design lever that determines which side of it you experience.

### Looking ahead to Chapter 13

Chapter 13 is where the whole tutorial assembles. It opens with Monte Carlo — the direct application of the concentration results of this chapter to integration — and then Markov chain Monte Carlo, where §12.1's typical-set geometry becomes operationally central: a sampler that is tuned to find the mode is a bad sampler, because the mode is not where the mass is, and Hamiltonian Monte Carlo's advantage over random-walk Metropolis in high dimension is exactly its ability to traverse the typical set rather than diffuse across it. Variational inference then arrives from three directions at once — as a bound on the marginal likelihood, as a KL projection, and as an importance-weighting identity — with the ELBO's decomposition into reconstruction and regularization terms falling out of the KL machinery of Chapter 11. From there the chapter builds variational autoencoders, then score matching and the denoising diffusion models that grew out of it, where Chapter 3's change of variables and this chapter's soap-bubble geometry jointly explain why noise schedules are parameterized the way they are. It closes on uncertainty calibration: what a predicted probability means, how to measure whether it means it, and why deep networks are systematically overconfident — a question that reaches back to the very first chapter's insistence that a probability is a measure of an event, not a number a model happens to output.

# Exercises

## Part A — Theory

**A1 (★).** Fix $\varepsilon\in(0,1)$. Find the smallest $d$ for which at least $99\%$ of the volume of $B_d(1)$ lies outside $B_d(1-\varepsilon)$, as a function of $\varepsilon$, and evaluate it for $\varepsilon=0.1$ and $\varepsilon=0.01$.

**A2 (★).** Using Stirling's formula, show $V_d(1)^{1/d}\sim\sqrt{2\pi e/d}$. Deduce the radius $r_d$ of the ball of volume exactly $1$ in $\mathbb{R}^d$, and evaluate $r_{100}$.

**A3 (★★).** Let $Z\sim\mathcal{N}(0,I_d)$. (a) Prove $\mathbb{E}\|Z\|=\sqrt2\,\Gamma(\tfrac{d+1}2)/\Gamma(\tfrac d2)$. (b) Show $\operatorname{Var}(\|Z\|)=d-(\mathbb{E}\|Z\|)^2$ and that this tends to $\tfrac12$. (c) Conclude that the *relative* standard deviation of $\|Z\|$ is asymptotically $1/\sqrt{2d}$.

**A4 (★★).** Let $T=\cos\Theta$ for two independent uniform directions in $\mathbb{R}^d$. (a) Derive $T^2\sim\mathrm{Beta}(\tfrac12,\tfrac{d-1}2)$ from the Gaussian representation. (b) Using this, compute $\Pr(|T|\le s)$ exactly for $d=3$ and check it against the elementary fact that the projection of a uniform point on $S^2$ onto a fixed axis is uniform on $[-1,1]$ (Archimedes).

**A5 (★★).** Prove the two bounds of Lemma 12.4 in full, filling in the elementary inequalities $e^{2a}\ge1+2a+2a^2$ for $a\ge0$ and $-x-\log(1-x)\ge x^2/2$ for $x\in[0,1)$.

**A6 (★★).** Show that the map $x\mapsto\|Ax\|$ is $\|A\|_{\text{op}}$-Lipschitz. Deduce from Theorem 12.7 that for $Z\sim\mathcal{N}(0,I_d)$ and any fixed matrix $A$, $\operatorname{Var}(\|AZ\|)\le\|A\|_{\text{op}}^2$. Then exhibit an $A$ for which this is tight up to a constant and one for which it is enormously loose, and explain the difference in terms of the spectrum of $A$.

**A7 (★★).** The Gaussian Poincaré inequality states $\operatorname{Var}(f(Z))\le\mathbb{E}\|\nabla f(Z)\|^2$ for differentiable $f$ (Results used but not proved). (a) Apply it to $f(z)=\|z\|$ to recover $\operatorname{Var}(\|Z\|)\le1$. (b) Apply it to $f(z)=\max_iz_i$, computing $\nabla f$ where it exists, and compare the resulting bound with the measured values in Worked Example 12.7. (c) Explain why Poincaré gives a variance bound but not a tail bound, and what extra ingredient Theorem 12.7 needs.

**A8 (★★★).** Prove Theorem 12.13's conclusion under the weaker hypothesis that the coordinates are i.i.d. with mean $0$, variance $\sigma^2$, and finite fourth moment (no sub-Gaussianity). Specifically, show that for *fixed* $n$, $C_n(Q)\to0$ in probability as $d\to\infty$. Where does your argument fail if you try to let $n$ grow with $d$, and what does that tell you about the role of sub-Gaussianity in the version we proved?

**A9 (★★).** Prove Lemma 12.16 with the sharper constant: show that for $\varepsilon\in(0,1)$,
$$\Pr\big(\big|\|Rx\|^2-\|x\|^2\big|\ge\varepsilon\|x\|^2\big)\ \le\ 2\exp\Big(-\frac k2\big(\varepsilon-\log(1+\varepsilon)\big)\Big),$$
and verify numerically at $k=100$, $\varepsilon=0.2$ that this improves on $2e^{-k\varepsilon^2/8}$.

**A10 (★★).** Prove Corollary 12.18 carefully, tracking constants: state exactly which vectors you must include in the JL point set and how many there are, and give the resulting requirement on $k$.

**A11 (★★).** Prove that $S^{d-1}$ admits an $\varepsilon$-net of cardinality at most $(1+2/\varepsilon)^d$ by a volumetric packing argument. (Hint: a maximal $\varepsilon$-separated subset is an $\varepsilon$-net; compare volumes of balls of radius $\varepsilon/2$ centred at its points.)

**A12 (★★★).** Prove Lemma 12.22, then prove the following variant: if $\mathcal{N}$ is an $\varepsilon$-net of $S^{d-1}$ only (no net on $S^{n-1}$), then $\|A\|\le(1-\varepsilon)^{-1}\max_{v\in\mathcal{N}}\|Av\|$. Which version gives a better final bound in Theorem 12.23, and why?

**A13 (★★).** For the Marchenko–Pastur law: (a) verify that $\rho_\gamma$ integrates to $1$ for $\gamma\in(0,1)$; (b) compute $\int x\,\rho_\gamma(x)\,dx$ and confirm it equals $1$; (c) give the condition number $b/a$ as a function of $\gamma$ and evaluate at $\gamma=0.1,0.5,0.8,0.99$; (d) show $b/a\to1$ as $\gamma\to0$ and $\to\infty$ as $\gamma\to1$.

**A14 (★★).** Suppose $\Sigma=I_d$ and $d/n=\gamma$. Using Proposition 12.26(1), find the largest $\gamma$ for which $\|\hat\Sigma-I\|_{\text{op}}\le0.1$. How many samples per dimension does that require? Comment on the implications for estimating a $2048\times2048$ feature covariance.

**A15 (★★).** Derive Proposition 12.28 in full, including the justification that the cross term vanishes. Then show that $\rho^\star=\beta^2/(\alpha^2+\beta^2)$ implies $\rho^\star\to1$ when $\Sigma=I$ exactly, and explain why this is the right behaviour.

**A16 (★★).** Show that $\log\Pr(\|Z\|\le1)=-\tfrac d2\log d+O(d)$ for $Z\sim\mathcal{N}(0,I_d)$, and compare with the exact values in Worked Example 12.5.

**A17 (★★).** Let $Q\in\mathbb{R}^{n\times n}$ be orthogonal and $c>0$. (a) Show that all singular values of $cQ$ equal $c$ and that $\|cQh\|=c\|h\|$ for every $h$. (b) Show that if $M$ has i.i.d. entries drawn from a distribution with a density (i.e. absolutely continuous with respect to Lebesgue measure), then $\Pr(M^\top M=c^2I)=0$ for $n\ge2$. (c) Show that (b) genuinely needs the density hypothesis, by computing $\Pr(M^\top M=2I)$ exactly when $n=2$ and the entries are i.i.d. uniform on $\{-1,+1\}$. What does this tell you about the phrase "a random matrix is never an isometry"?

**A18 (★★★).** *(Matrix Bernstein for covariance; the spiked threshold in numbers.)* Let $X_1,\dots,X_n$ be i.i.d. mean-zero in $\mathbb{R}^d$ with covariance $\Sigma$ and $\|X_i\|^2\le B$ almost surely. (a) Carry through the Matrix Bernstein argument sketched after Theorem 12.24 in full: define the summands, verify the boundedness parameter $K$ and the variance parameter $\sigma^2$, and derive the high-probability bound $\|\hat\Sigma-\Sigma\|\lesssim\sqrt{B\|\Sigma\|\log(d/\delta)/n}+B\log(d/\delta)/n$. (You may use $\mathbb{E}[\|X\|^2XX^\top]\preceq B\,\Sigma$, and should prove it.) (b) Deduce a sufficient sample size $n$ for relative error $\|\hat\Sigma-\Sigma\|\le\varepsilon\|\Sigma\|$, in terms of the effective dimension $B/\|\Sigma\|$, and compare with the Gaussian answer of Proposition 12.26(2). (c) Now the spiked model of Theorem 12.29: at $\gamma=d/n=0.25$, compute the detection threshold, the limiting top sample eigenvalue when the true spike is $\lambda=2$, and the limiting eigenvector overlap $|\langle\hat v,v\rangle|^2$; then find the smallest $\lambda$ for which the overlap exceeds $0.9$.

## Part B — Computational labs

Use a fixed seed and report it. All labs are doable in NumPy/SciPy in under a minute except B8.

**B1 (★).** Reproduce the master table of §12.1.4, and additionally plot the exact angle density $\propto\sin^{d-2}\theta$ for $d\in\{2,10,100,1000\}$ to recreate Figure 12.3. Report, for $d\in\{2,10,100,1000,10000\}$: the shell fraction at $\varepsilon=0.1$, $\log_{10}V_d(1)$, $\log_{10}$ of the inscribed-ball-to-cube ratio, $\Pr(\Theta\in[85^\circ,95^\circ])$, $\mathbb{E}\|Z\|$ versus $\sqrt d$, $\mathrm{sd}(\|Z\|)$, and $\log_{10}\Pr(\|Z\|\le1)$. Use exact special functions, not simulation, and use extended precision (`mpmath`) for the last column at $d\ge1000$.

**B2 (★★).** Verify Gaussian norm concentration. For $d\in\{10,100,1000\}$ and $t\in\{1,2,3\}$, compare (i) the Monte Carlo estimate of $\Pr(|\|Z\|-\sqrt d|\ge t)$, (ii) the exact value from the $\chi^2_d$ CDF, and (iii) the bounds $2e^{-t^2/8}$ (Theorem 12.5) and $2e^{-t^2/2}$ (Corollary 12.8). Comment on which bound is useful at which $t$.

**B3 (★★).** Implement Johnson–Lindenstrauss. Take $n=500$ points in $\mathbb{R}^{2000}$; for $k$ ranging over powers of two from $16$ to $1024$ plus the theoretical $k^\star=\lceil 8\log n/\varepsilon^2\rceil$ at $\varepsilon=0.2$, measure the worst-case, $95$th-percentile, and mean pairwise distance distortion. Then find the smallest $k$ achieving worst-case distortion $\le0.2$ and report the ratio $k^\star/k$. Finally, hold $k=200$ fixed and vary $d\in\{200,1000,5000,20000\}$ to confirm $d$-independence.

**B4 (★★).** Plot empirical singular-value spectra of $n\times d$ Gaussian matrices (entries $\mathcal{N}(0,1/n)$) against the Marchenko–Pastur density for $\gamma\in\{0.1,0.5,1.0\}$, and tabulate observed $s_{\min},s_{\max}$ against $1\mp\sqrt\gamma$. Use $n\ge2000$.

**B5 (★★★).** *The most important lab in the chapter.* Build two datasets in the same ambient dimension $d$: (i) isotropic $\mathcal{N}(0,I_d)$; (ii) a $k=5$ dimensional manifold, e.g. $\Psi(z)=\sqrt2\cos(W^\top z+b)$ with $W\in\mathbb{R}^{5\times d}$ having $\mathcal{N}(0,1/5)$ entries and $b$ uniform on $[0,2\pi)^d$, applied to $z\sim\mathcal{N}(0,I_5)$. For $d\in\{10,100,1000,10000\}$ report (a) mean $D_{\max}/D_{\min}$ over $100$ queries against $n=1000$ references, and (b) $1$-NN test error for a label depending on five informative variables in both cases. Produce the two-curve figure. Then answer: which curve would you expect for a real image-embedding dataset, and how would you check?

**B6 (★★).** Lipschitz concentration. For $f(x)=\max_ix_i$ and $d\in\{10,10^2,10^3,10^4,10^5\}$, estimate $\mathbb{E}f(Z)$ and $\mathrm{sd}(f(Z))$ from $2\times10^4$ draws. Confirm $\mathrm{sd}\le1$ at every $d$ and compare $\mathbb{E}f$ with $\sqrt{2\log d}$. Repeat with $g(x)=\frac1{\sqrt d}\sum_ix_i$ (also $1$-Lipschitz) and explain why its standard deviation behaves completely differently from $f$'s while both obey the same bound.

**B7 (★★).** Covariance estimation. With $\Sigma$ having log-spaced eigenvalues on $[0.2,5]$ in a random basis, $d=200$, and $n\in\{100,250,500,2000\}$, compute over $20$ replications: $\|\hat\Sigma-\Sigma\|_{\text{op}}$, $\|\hat\Sigma_{\text{LW}}-\Sigma\|_{\text{op}}$, the estimated $\hat\rho$, and the whitening diagnostic $\mathrm{cond}(\hat\Sigma^{-1}\Sigma)$ versus $\mathrm{cond}(\hat\Sigma_{\text{LW}}^{-1}\Sigma)$. Plot $\hat\rho$ against $d/n$.

**B8 (★★★).** Hessian spectrum. Implement a one-hidden-layer $\tanh$ network ($d_{\text{in}}=10$, width $20$, so $P=241$) with squared loss on $n=200$ synthetic points from a teacher network plus noise. Compute the exact Hessian by central differences on the analytic gradient. Report the spectrum at random initialization and after training to loss below $10^{-5}$: $\lambda_{\min}$, $\lambda_{\max}$, the index (fraction negative), $|\lambda_{\min}|/\lambda_{\max}$, and the number of eigenvalues above $0.05$. Compare the initialization index with the semicircle prediction of $\tfrac12$, and comment on whether the trained point is better described as a minimum or a saddle.

## Part C — Paper-style derivations

**C1 (★★★).** Prove the Johnson–Lindenstrauss lemma from first principles. Your write-up must (i) state the projection and its normalization and explain why that normalization makes the estimator unbiased; (ii) prove the single-vector concentration bound, identifying *exactly* where sub-Gaussian (or sub-exponential) concentration is used and why the bound does not involve $d$; (iii) apply the union bound, identifying exactly how many events you union over and how this produces $\log n$; (iv) state the final requirement on $k$ with explicit constants; and (v) discuss, with reference to the measured conservatism in Worked Example 12.14, whether the union bound is the dominant source of looseness and what the tightness of the union bound for disjoint events implies here.

**C2 (★★★).** Prove the operator-norm bound $\|B\|\lesssim\sqrt n+\sqrt d$ for a Gaussian matrix via an $\varepsilon$-net argument. Then analyse the trade-off in choosing the net resolution: write the final constant as an explicit function $C(\varepsilon)$, minimize it numerically, and explain in words why the optimum lies at a *fine* net rather than a coarse one. Compare your constant with the sharp asymptotic constant from Bai–Yin and justify, in two or three sentences, why one would ever prefer the net argument.

**C3 (★★★).** Write a critical note, of roughly 800–1200 words, on the claim: *"Deep learning works because high-dimensional loss landscapes have no bad local minima."* Your note must separate (a) what is actually proved, and for which model class, citing specific results; (b) what is empirically observed, and on which architectures and at what scale; (c) what remains conjecture. Address the symmetry structure of neural losses and why it violates the hypotheses of the random-matrix heuristic. Conclude with a precise statement you would be willing to defend in a paper, and one you would not.

# Appendix A — Self-check answers

**Self-check 12.1.**
1. $1-0.95^{50}=1-0.0769=0.9231$, so about $92.3\%$.
2. False as stated. The ball of radius $1$ has small *volume measured in the units $\lambda_d$*, which is a $d$-fold product of length units; comparing $V_{100}$ to $V_3$ is comparing incommensurable quantities. What is genuinely true and useful: the unit ball occupies a vanishing fraction of the unit cube that contains it, and volume concentrates near its surface. Its diameter is still $2$.
3. $\mathbb{E}\|z\|\approx\sqrt{256-0.5}=15.984$ and $\mathrm{sd}\approx0.707$, so roughly $[15.28,16.69]$.
4. $\mathbb{E}|\cos\Theta|\approx\sqrt{2/(\pi\cdot1000)}=0.0252$. Relevance: since random directions are almost orthogonal, a width-$1000$ residual stream can carry far more than $1000$ features, each on its own random direction, with interference of order $0.025$ per pair — this is the superposition hypothesis.

**Self-check 12.2.**
1. $f(x)=\frac1d\sum x_i$ has $\nabla f=\frac1d\mathbf{1}$, so $L=\|\nabla f\|=1/\sqrt d$. Then (12.3) gives $\Pr(|f(Z)|\ge t)\le2e^{-dt^2/2}$. Exactly: $f(Z)\sim\mathcal{N}(0,1/d)$, so the true tail is $2(1-\Phi(t\sqrt d))\le e^{-dt^2/2}$. The bound is right up to the factor $2$ — Lipschitz concentration is *sharp* for linear functions, which is reassuring.
2. Corollary 12.11 is stated on the *unit* sphere, whose diameter is $2$ regardless of $d$; a $1$-Lipschitz function there has range at most $2$, and the concentration scale must therefore shrink as the space grows, giving the $d$ in the exponent. Theorem 12.7 is on $\mathbb{R}^d$ with the standard Gaussian, which lives on a sphere of radius $\sqrt d$; rescaling $S^{d-1}\to\sqrt d\,S^{d-1}$ multiplies Lipschitz constants by $\sqrt d$ and converts $e^{-dt^2/2}$ into $e^{-t^2/2}$. No conflict: they are the same statement in different coordinates.
3. $f(z)=\|z\|$: $1$-Lipschitz with $\mathbb{E}f\approx\sqrt d$. Or $f(z)=\max_iz_i$ with $\mathbb{E}f\approx\sqrt{2\log d}$. Concentration says $f(Z)$ is close to *its own mean*; locating the mean is a separate problem, often the harder one.
4. Convexity is what lets you control the function by its behaviour on a *net* or by a linear lower bound at the median: for convex $f$, the sublevel sets are convex, and Talagrand's convex distance measures how far a point is from a convex set in a coordinate-weighted way. Without convexity a $1$-Lipschitz function can concentrate its sensitivity on a single unlucky coordinate configuration, and the bound genuinely fails.

**Self-check 12.3.**
1. Set $2.7\sqrt{\log n/d}=0.1$ with $\log(10^6)=13.82$: $\sqrt{13.82/d}=0.0370$, so $d=13.82/0.00137=10{,}085$, i.e. roughly $d\approx10^4$. Contrast the same computation at $\varepsilon=1$: $d\approx7.3\times13.82\approx100$. The $\varepsilon^{-2}$ makes the tolerable dimension swing by two orders of magnitude, which is why quoting a single "threshold dimension" is meaningless without saying how much contrast you need.
2. Normalizing to unit norm forces all points onto $S^{d-1}$, so $D_{\max}/D_{\min}\le2/D_{\min}$ and the *squared* distance becomes $2-2\cos\Theta$: distances are now an exact monotone function of the angle. The ratio $D_{\max}/D_{\min}$ typically *increases* because $D_{\min}$ can be small for genuinely similar items while $D_{\max}\le2$. Cosine similarity is more informative because it removes the norm, which for embeddings often encodes frequency or confidence rather than semantics.
3. Exactly low-dimensional support: $P(M)=1$ for a $k$-manifold $M$ with $\lambda_d(M)=0$, so $P$ has no Lebesgue density. Approximately low-dimensional mass: $\mathrm{supp}(P)$ may be all of $\mathbb{R}^d$ but almost all mass is within $\epsilon$ of a $k$-dimensional set. Chapter 1's circle generator $G(t)=(\cos t,\sin t)$ produces the *exact* case.
4. Contrast is a *ratio* of distances, so any global rescaling cancels. Only the *distortion* $\beta/\alpha$ — the failure of $\Psi$ to be a similarity — can change it.

**Self-check 12.4.**
1. The $\log n$ enters at the union bound over $\binom n2<n^2/2$ pairs, which contributes $\log(n^2)=2\log n$ to the exponent. The $\varepsilon^{-2}$ enters in Lemma 12.16, where the failure probability for a single vector is $e^{-k\varepsilon^2/8}$; making that small requires $k\gtrsim\varepsilon^{-2}$.
2. So that $\mathbb{E}\|Rx\|^2=\|x\|^2$. With variance $1$ per entry, $\|Rx\|^2$ would be a $\chi^2_k$ scaled by $\|x\|^2$ with mean $k\|x\|^2$ — the projection would inflate all distances by $\sqrt k$. The $1/\sqrt k$ scaling is exactly the normalization that makes the map an isometry in expectation.
3. $k\ge8\cdot100\cdot(2\log10^4+\log(1/\delta))$, which is at least $14{,}700$ even ignoring $\delta$. Since $d=50$, the theorem prescribes $k\gg d$ and is completely vacuous: the identity map already achieves $\varepsilon=0$. JL is only informative when $\varepsilon^{-2}\log n\ll d$.
4. Because the $\binom n2$ pair events share a single random matrix $R$ and are therefore strongly positively correlated, whereas the union bound implicitly prices them as disjoint — and the union bound is tight exactly for disjoint events.

**Self-check 12.5.**
1. $\gamma=1$, so the support is $[(1-1)^2,(1+1)^2]=[0,4]$: the sample eigenvalues fill all of $[0,4]$ even though every true eigenvalue is $1$.
2. Because the operator norm is a supremum over a *pair* of unit vectors, $\|A\|=\sup_{u,v}\langle u,Av\rangle$, and each of the two suprema must be discretized. Discretizing only $v$ would leave $\|Av\|$ itself as a supremum over $u$ — which is fine, but then the one-net variant of A12 applies and the elementary quantity you bound is $\|Av\|$ (a $\chi$-type variable) rather than $\langle u,Av\rangle$ (a single standard Gaussian).
3. Step 2, over $|\mathcal{M}||\mathcal{N}|\le(1+2/\varepsilon)^{n+d}=e^{L(n+d)}$ pairs. This is where $\sqrt{n+d}$, and hence $\sqrt n+\sqrt d$, comes from.
4. A finer net ($\varepsilon$ smaller) reduces the discretization penalty $(1-2\varepsilon)^{-1}$ toward $1$ but increases the net cardinality, which enters only as $\sqrt{2\log(1+2/\varepsilon)}$ — logarithmically. Since cost grows logarithmically and benefit grows like $(1-2\varepsilon)^{-1}$, which diverges as $\varepsilon\to\tfrac12$, the optimum is pushed toward small $\varepsilon$; numerically $\varepsilon^\star=0.0605$, giving $C=3.022$.

**Self-check 12.6.**
1. It pins down $\mathbb{E}[\frac1{n_{\text{in}}}\|W\|_F^2]$, i.e. the *mean square* singular value $\frac1{n}\sum_is_i^2$. It says nothing about the spread, which Marchenko–Pastur then determines to run over $\big[\sigma\,|\sqrt{n_{\text{in}}}-\sqrt{n_{\text{out}}}|,\ \sigma(\sqrt{n_{\text{in}}}+\sqrt{n_{\text{out}}})\big]$ — from $0$ to twice the target gain when the layer is square.
2. Because the relevant Jacobian is not $W$ but $D W$, where $D$ is the diagonal $0/1$ gating matrix of the ReLU. Even if $W$ is orthogonal, $DW$ is a random row-subsampling of an orthogonal matrix, which has a spread spectrum; the gating introduces multiplicative randomness that no choice of $W$ can cancel. Pennington–Schoenholz–Ganguli make this precise.
3. Because "saddle" implies a *usable* descent direction — an eigenvalue negative enough that a step along it reduces the loss appreciably. Here the most negative eigenvalue is $3\times10^{-4}$ of the largest, which is at the level of the finite-difference error, and $78\%$ of directions are flat to within $0.01$. The geometry is a flat valley, not a pass between two peaks.
4. Examples: "all local minima of a deep linear network are global" (Kawaguchi) — the gap is that real networks have nonlinearities, which is exactly what creates spurious minima (Safran–Shamir); or NTK-regime global convergence — the gap is that the required widths are polynomially enormous and the linearized dynamics provably cannot learn features.

**Self-check 12.7.**
1. $\gamma=0.25$, $\sqrt\gamma=0.5$: error $\to2(0.5)+0.25=1.25$, and the eigenvalues fill $[(1-0.5)^2,(1+0.5)^2]=[0.25,2.25]$. So with four samples per dimension the sample condition number is $9$ where the truth is $1$.
2. Because $\Sigma\mapsto\Sigma^{-1}$ is nonlinear, so $\mathbb{E}[\hat\Sigma^{-1}]\ne(\mathbb{E}\hat\Sigma)^{-1}$; by Jensen-type reasoning the inversion systematically inflates directions where $\hat\Sigma$ underestimates. Worse, if $d\ge n$ then $\hat\Sigma$ is singular and $\hat\Sigma^{-1}$ does not exist at all, while $\mathbb{E}\hat\Sigma=\Sigma$ remains invertible.
3. $\beta^2=\mathbb{E}\|\hat\Sigma-\Sigma\|_F^2\to0$ as $n\to\infty$ with $d$ fixed, while $\alpha^2=\|\Sigma-\mu I\|_F^2$ is a fixed constant. Hence $\rho^\star\to0$ and the estimator reverts to $\hat\Sigma$ — reassuring, because with abundant data the unbiased estimator is the right one and shrinkage should switch itself off.
4. K-FAC and Shampoo (also Adam's $\epsilon$, and the $\lambda I$ in linear-bandit / LinUCB design matrices).

**Self-check 12.8.**
1. Theorem 12.5/Theorem 12.7 (concentration of a Lipschitz function of many independent variables). Good news: applied to $\hat R_n=\frac1n\sum_i\ell_i$, it says the empirical risk is close to the true risk, which is what makes generalization possible. Bad news: applied to $\|X-Y\|$, it says all pairwise distances are close to a common value, which is what kills nearest-neighbour discrimination. Same theorem, different function: in the first case the common value is the thing you want and the fluctuation is the error; in the second the fluctuation *is* the signal and the common value is the nuisance.
2. The sample size $n$ plays the role of $d$: it counts the independent inputs being combined. Concretely, view $\hat R_n$ as a function $F(\ell_1,\dots,\ell_n)=\tfrac1n\sum_i\ell_i$ of $n$ independent per-example losses; its gradient has Euclidean norm $\|\nabla F\|=\tfrac1{\sqrt n}$, so $F$ is $n^{-1/2}$-Lipschitz. A concentration bound of the form (12.3) then gives deviations of order $L=n^{-1/2}$ — which is exactly the $O(n^{-1/2})$ rate of Hoeffding. So "more data" and "higher dimension" are the *same* mechanism, differing only in whether the shrinking Lipschitz constant is something you want (estimation) or something that destroys your signal (distance).
3. (i) Estimate intrinsic dimension directly (TwoNN or Levina–Bickel MLE) and compare with $d$; (ii) plot the histogram of pairwise distances and compute $(D_{\max}-D_{\min})/D_{\text{median}}$ — if this is below roughly $0.2$ the contrast is dead and no index can help. A third, cheap option: compute exact brute-force recall on a subsample. If brute-force $k$-NN is also poor, the problem is the representation; if brute force is good and the index is poor, the problem is the index.
4. Because §12.3 shows that every harmful consequence of dimension — contrast collapse, $k$-NN degradation, the need for enormous sample sizes — is governed by intrinsic dimension (Definition 12.14, Proposition 12.15), not ambient dimension, and because a learned encoder is precisely a map that changes the intrinsic dimension of the represented data. Worked Examples 12.11 and 12.12 exhibit two datasets with identical ambient dimension and opposite behaviour; the only difference is the structure of the map that produced them.

# Solutions to Exercises

## Part A

**A1.** We need $(1-\varepsilon)^d\le0.01$, i.e. $d\log(1-\varepsilon)\le\log0.01$, i.e.
$$d\ \ge\ \frac{\log 100}{-\log(1-\varepsilon)}\ =\ \frac{4.60517}{-\log(1-\varepsilon)} .$$
*Thinking:* the only step that needs care is the sign — $\log(1-\varepsilon)<0$, so dividing flips the inequality. For $\varepsilon=0.1$: $-\log0.9=0.105361$, so $d\ge43.71$, i.e. $d=44$. For $\varepsilon=0.01$: $-\log0.99=0.0100503$, so $d\ge458.2$, i.e. $d=459$. Note the near-perfect $1/\varepsilon$ scaling, since $-\log(1-\varepsilon)\approx\varepsilon$ for small $\varepsilon$: shrinking the shell by a factor of ten costs a factor of ten in dimension, no more.

**A2.** From Proposition 12.1, $\log V_d(1)=\tfrac d2\log\pi-\log\Gamma(\tfrac d2+1)$. Stirling gives $\log\Gamma(m+1)=m\log m-m+O(\log m)$ with $m=d/2$:
$$\log V_d(1)=\frac d2\log\pi-\frac d2\log\frac d2+\frac d2+O(\log d).$$
Divide by $d$:
$$\frac1d\log V_d(1)=\frac12\Big(\log\pi-\log\frac d2+1\Big)+O\Big(\frac{\log d}d\Big)=\frac12\log\frac{2\pi e}{d}+o(1),$$
so $V_d(1)^{1/d}\sim\sqrt{2\pi e/d}$.

For the unit-volume radius: $V_d(r_d)=V_d(1)r_d^d=1$ gives $r_d=V_d(1)^{-1/d}\sim\sqrt{d/(2\pi e)}$. At $d=100$: $\sqrt{100/17.079}=2.420$; the exact value $V_{100}(1)^{-1/100}=10^{39.6256/100}=2.490$. *Thinking:* the asymptotic is about $3\%$ low at $d=100$ because we dropped an $O(\log d/d)$ term; the qualitative message — the unit-volume ball has radius growing like $\sqrt d$ — is what matters.

**A3.** (a) $\|Z\|=(\chi^2_d)^{1/2}$. For $W\sim\chi^2_d$ with density $f(w)=\frac{w^{d/2-1}e^{-w/2}}{2^{d/2}\Gamma(d/2)}$,
$$\mathbb{E}[W^{s}]=\frac1{2^{d/2}\Gamma(d/2)}\int_0^\infty w^{d/2+s-1}e^{-w/2}\,dw = \frac{2^{d/2+s}\Gamma(d/2+s)}{2^{d/2}\Gamma(d/2)}=2^s\frac{\Gamma(d/2+s)}{\Gamma(d/2)},$$
substituting $w=2u$. Put $s=\tfrac12$.

(b) $\operatorname{Var}(\|Z\|)=\mathbb{E}\|Z\|^2-(\mathbb{E}\|Z\|)^2=d-(\mathbb{E}\|Z\|)^2$ since $\mathbb{E}\|Z\|^2=\mathbb{E}\chi^2_d=d$. For the limit, use the ratio asymptotic $\Gamma(m+\tfrac12)/\Gamma(m)=\sqrt m\,(1-\tfrac1{8m}+O(m^{-2}))$ with $m=d/2$:
$$\mathbb{E}\|Z\|=\sqrt2\cdot\sqrt{d/2}\Big(1-\frac1{4d}+O(d^{-2})\Big)=\sqrt d\Big(1-\frac1{4d}+O(d^{-2})\Big),$$
so $(\mathbb{E}\|Z\|)^2=d(1-\tfrac1{2d}+O(d^{-2}))=d-\tfrac12+O(d^{-1})$ and $\operatorname{Var}(\|Z\|)=\tfrac12+O(d^{-1})$. This also explains why $\sqrt{d-\tfrac12}$ was such a good approximation in Proposition 12.3: $\sqrt{d-\tfrac12}=\sqrt d(1-\tfrac1{4d}+O(d^{-2}))$, matching to the order computed.

(c) $\mathrm{sd}/\mathbb{E}\to\sqrt{1/2}/\sqrt d=1/\sqrt{2d}$. *Thinking:* the entire soap-bubble phenomenon is this one line. Absolute width constant, mean growing like $\sqrt d$, ratio therefore vanishing.

**A4.** (a) Write $U=Z/\|Z\|$ with $Z\sim\mathcal{N}(0,I_d)$, and by rotational invariance take the second direction to be $e_1$, so $T=U_1=Z_1/\|Z\|$. Then
$$T^2=\frac{Z_1^2}{Z_1^2+\sum_{j\ge2}Z_j^2}=\frac{G_1}{G_1+G_2},\quad G_1\sim\Gamma(\tfrac12,2)\ \text{indep. of}\ G_2\sim\Gamma(\tfrac{d-1}2,2).$$
The standard Gamma–Beta fact (change of variables $(g_1,g_2)\mapsto(g_1+g_2,\ g_1/(g_1+g_2))$, whose Jacobian factorizes the joint density) gives $G_1/(G_1+G_2)\sim\mathrm{Beta}(\tfrac12,\tfrac{d-1}2)$.

(b) At $d=3$: $T^2\sim\mathrm{Beta}(\tfrac12,1)$, whose CDF is $\Pr(T^2\le u)=\int_0^u\frac{\Gamma(3/2)}{\Gamma(1/2)\Gamma(1)}v^{-1/2}dv=\tfrac12\cdot2u^{1/2}=\sqrt u$. Hence $\Pr(|T|\le s)=\Pr(T^2\le s^2)=s$. So $|T|$ is uniform on $[0,1]$, i.e. $T$ is uniform on $[-1,1]$ — exactly Archimedes' theorem that the projection of the uniform measure on $S^2$ onto an axis is uniform (the "sphere and cylinder" result). *Thinking:* this is a good sanity check on the Beta formula, because $d=3$ is the one dimension where the answer is independently known and pretty.

**A5.** *MGF.* Computed in the text. *Upper tail.* Chernoff and optimization give $\Pr(W\ge d+u)\le\exp(-\tfrac d2h(u/d))$ with $h(x)=x-\log(1+x)$; the optimization is legitimate because $\psi$ is strictly concave on $(0,\tfrac12)$ with $\psi'(0^+)=-u<0$... in fact $\psi'(0)=-(d+u)+d=-u<0$, so the stationary point $s^\star$ found in the text is the unique maximum of $-\psi$, i.e. the minimum of the bound. Substituting $u=2\sqrt{dt}+2t$ and $a=\sqrt{t/d}$ gives $u/d=2a+2a^2$ and we need
$$2a+2a^2-\log(1+2a+2a^2)\ \ge\ 2a^2\quad\Longleftrightarrow\quad 2a\ \ge\ \log(1+2a+2a^2)\quad\Longleftrightarrow\quad e^{2a}\ge1+2a+2a^2 .$$
The last holds because $e^{2a}=\sum_{m\ge0}(2a)^m/m! = 1+2a+2a^2+\tfrac43a^3+\cdots$ and every dropped term is $\ge0$ for $a\ge0$. Hence $\tfrac d2h(u/d)\ge\tfrac d2\cdot2a^2=t$.

*Lower tail.* For $s<0$, $\mathbb{E}e^{sW}=(1-2s)^{-d/2}$ is finite for all $s<0$, and
$$\Pr(W\le d-u)\le e^{-s(d-u)}(1-2s)^{-d/2}.$$
Writing $s=-\sigma$ with $\sigma>0$ and optimizing as before yields $\Pr(W\le d-u)\le\exp\big(-\tfrac d2\big[-\tfrac ud-\log(1-\tfrac ud)\big]\big)$. Set $x=u/d\in[0,1)$. We need $-x-\log(1-x)\ge x^2/2$. Let $\phi(x)=-x-\log(1-x)-x^2/2$; then $\phi(0)=0$ and
$$\phi'(x)=-1+\frac1{1-x}-x=\frac{x}{1-x}-x=\frac{x-x(1-x)}{1-x}=\frac{x^2}{1-x}\ \ge0,$$
so $\phi\ge0$ on $[0,1)$. Taking $u=2\sqrt{dt}$ gives $x=2\sqrt{t/d}$ and $\tfrac d2\cdot\tfrac{x^2}2=\tfrac d2\cdot\tfrac{4t/d}2=t$. *Thinking:* the asymmetry between the tails (why the upper tail needs the extra $+2t$) traces back to $\chi^2$ being right-skewed: the MGF blows up at $s=\tfrac12$, capping how much upper-tail decay you can extract, while the left tail is unobstructed.

**A6.** Lipschitz: $\big|\|Ax\|-\|Ay\|\big|\le\|A(x-y)\|\le\|A\|_{\text{op}}\|x-y\|$ by the reverse triangle inequality and the definition of the operator norm. Theorem 12.7 with $L=\|A\|_{\text{op}}$ gives $\operatorname{Var}(\|AZ\|)\le\|A\|_{\text{op}}^2$.

*Tight case:* $A=e_1e_1^\top$ (rank one, $\|A\|=1$). Then $\|AZ\|=|Z_1|$, whose variance is $1-2/\pi=0.3634$ — within a factor $2.75$ of the bound $1$. *Loose case:* $A=I_d$, $\|A\|=1$, and $\operatorname{Var}(\|Z\|)\to\tfrac12$ — also within a factor $2$. A genuinely loose case needs many equal singular values *and* a large one: take $A=\mathrm{diag}(1,1,\dots,1)$ scaled, or better, $A$ with singular values $(1,\epsilon,\dots,\epsilon)$: then $\|A\|=1$ but $\|AZ\|\approx|Z_1|$ again. The instructive contrast is instead between $A=I_d$ and $A=\sqrt d\,e_1e_1^\top$: both have $\operatorname{Var}(\|AZ\|)$ of order $1$ and $d$ respectively, and the bound $\|A\|^2$ is $1$ and $d$. *Thinking:* the bound depends only on the *largest* singular value, so it is tight when the variability of $\|AZ\|$ is driven by the top singular direction and loose when many directions of comparable size average out. This is the same phenomenon as $\operatorname{Var}(\|Z\|)=\tfrac12<1$: averaging over $d$ equal directions buys you a factor of $2$ but not more, because $\|\cdot\|$ is not linear.

**A7.** (a) $f(z)=\|z\|$ has $\nabla f(z)=z/\|z\|$ for $z\ne0$, so $\|\nabla f\|^2=1$ a.e. and Poincaré gives $\operatorname{Var}(\|Z\|)\le1$. (The single point $z=0$ is Lebesgue-null, so a.e. differentiability suffices.)

(b) $f(z)=\max_iz_i$. Off the null set where the maximum is attained twice, $\nabla f=e_{i^\star}$ with $i^\star=\arg\max$, so $\|\nabla f\|^2=1$ and Poincaré gives $\operatorname{Var}(\max_iZ_i)\le1$, i.e. $\mathrm{sd}\le1$. Measured (Worked Example 12.7): $0.5867,0.4314,0.3515,0.3031$ at $d=10,10^2,10^3,10^4$. The bound holds with room to spare and, notably, does not capture the *decrease* — the true standard deviation of the maximum of $d$ Gaussians decays like $1/\sqrt{2\log d}$, which Poincaré cannot see because it uses only $\|\nabla f\|$ pointwise and not the fact that the argmax is concentrated in a shrinking region.

(c) Poincaré controls the second moment of the deviation; a tail bound requires control of *all* moments (or of the MGF). The standard route is to iterate Poincaré-type arguments into a logarithmic Sobolev inequality, then apply the Herbst argument (integrate the entropy inequality to get an MGF bound) and Chernoff. That chain — log-Sobolev, Herbst, Chernoff — is exactly what a proof of Theorem 12.7 supplies, and it is the reason Theorem 12.7 is a genuinely deeper statement than Poincaré rather than a repackaging.

**A8.** *Fixed $n$, no sub-Gaussianity.* Let $Y=Q-X^{(1)}$ with i.i.d. coordinates $Y_j$, $\mathbb{E}Y_j=0$, $\mathbb{E}Y_j^2=2\sigma^2$, $\mathbb{E}Y_j^4=:m_4<\infty$. Then $\|Y\|^2=\sum_jY_j^2$ is a sum of $d$ i.i.d. terms with mean $2\sigma^2$ and variance $m_4-4\sigma^4<\infty$, so by Chebyshev
$$\Pr\Big(\Big|\frac{\|Y\|^2}{2\sigma^2d}-1\Big|\ge\epsilon\Big)\ \le\ \frac{m_4-4\sigma^4}{4\sigma^4\epsilon^2 d}\ \xrightarrow[d\to\infty]{}\ 0 .$$
For fixed $n$, apply this to each of the $n$ pairs and union-bound: the failure probability is at most $n$ times the above, still $\to0$. On the good event, as in Theorem 12.13, $D_{\max}/D_{\min}\le\sqrt{(1+\epsilon)/(1-\epsilon)}$, so $C_n(Q)\le\epsilon+O(\epsilon^2)$. Since $\epsilon>0$ was arbitrary, $C_n(Q)\to0$ in probability.

*Where it fails for growing $n$.* Chebyshev gives a failure probability decaying only like $1/(\epsilon^2d)$, so the union bound over $n$ events requires $n/(\epsilon^2 d)\to0$, i.e. $n\ll\epsilon^2 d$ — a *polynomial* relation between $n$ and $d$. Sub-Gaussianity replaces $1/(\epsilon^2d)$ by $e^{-c\epsilon^2d}$, so the union bound tolerates $n$ up to $e^{c\epsilon^2 d}$, i.e. requires only $\log n\ll\epsilon^2d$ — a *logarithmic* relation. *Thinking:* this is the general lesson about why anyone bothers with exponential tail bounds. Chebyshev suffices for any fixed number of events; exponential tails are what let the number of events grow exponentially, which is exactly what union-bound arguments over hypothesis classes, over $\binom n2$ pairs, or over $\varepsilon$-nets of size $(1+2/\varepsilon)^d$ all need.

**A9.** As in Lemma 12.16 reduce to $W\sim\chi^2_k$ and the event $\{|W-k|\ge\varepsilon k\}$. From A5's intermediate step, $\Pr(W\ge k(1+\varepsilon))\le\exp(-\tfrac k2h(\varepsilon))$ with $h(\varepsilon)=\varepsilon-\log(1+\varepsilon)$; and by the same Chernoff computation with $s<0$, $\Pr(W\le k(1-\varepsilon))\le\exp(-\tfrac k2(-\varepsilon-\log(1-\varepsilon)))$. Since $-\varepsilon-\log(1-\varepsilon)\ge\varepsilon-\log(1+\varepsilon)$ for $\varepsilon\in(0,1)$ (both equal $\tfrac{\varepsilon^2}2\mp\tfrac{\varepsilon^3}3+\cdots$, and the lower-tail series has all-positive coefficients), the upper tail dominates and the sum of the two is at most $2\exp(-\tfrac k2h(\varepsilon))$.

Numerically at $k=100$, $\varepsilon=0.2$: $h(0.2)=0.2-\log1.2=0.2-0.182322=0.017678$, so the sharp bound is $2e^{-0.8839}=0.8263$. The crude bound is $2e^{-100\cdot0.04/8}=2e^{-0.5}=1.2131$. Improvement factor $1.47$. At $\varepsilon=0.5$, $k=100$: $h=0.5-0.405465=0.094535$, bound $2e^{-4.727}=0.01771$, versus $2e^{-3.125}=0.0879$ — improvement factor $5.0$. *Thinking:* the two bounds agree to leading order in small $\varepsilon$ (since $h(\varepsilon)\approx\varepsilon^2/2$ gives exponent $k\varepsilon^2/4$, versus $k\varepsilon^2/8$ — already a factor of two) and diverge for larger $\varepsilon$, where the crude quadratic form throws away the sub-exponential regime.

**A10.** Apply Theorem 12.17 to the point set $\mathcal{P}=\{0\}\cup\{x_i\}_{i\le n}\cup\{x_i+x_j\}_{i<j}\cup\{x_i-x_j\}_{i<j}$. Its size is $N=1+n+2\binom n2=n^2+1$. Theorem 12.17 with $N$ in place of $n$ requires
$$k\ \ge\ \frac8{\varepsilon^2}\big(2\log(n^2+1)+\log\tfrac1\delta\big)\ \approx\ \frac8{\varepsilon^2}\big(4\log n+\log\tfrac1\delta\big),$$
so the constant merely doubles relative to the distance-only statement: $k=O(\varepsilon^{-2}\log n)$ still. On the good event, (12.6) holds for $u=x_i+x_j$ and $u=x_i-x_j$ (these are differences of points in $\mathcal{P}$ against $0$), so by polarization
$$\big|\langle Rx_i,Rx_j\rangle-\langle x_i,x_j\rangle\big| = \tfrac14\big|(\|R(x_i{+}x_j)\|^2-\|x_i{+}x_j\|^2)-(\|R(x_i{-}x_j)\|^2-\|x_i{-}x_j\|^2)\big|$$
$$\le\ \tfrac\varepsilon4\big(\|x_i+x_j\|^2+\|x_i-x_j\|^2\big)\ =\ \tfrac\varepsilon2\big(\|x_i\|^2+\|x_j\|^2\big),$$
by the parallelogram law. For unit vectors this is $\varepsilon$; in general, since $\tfrac12(\|x_i\|^2+\|x_j\|^2)\ge\|x_i\|\|x_j\|$ by AM–GM, the clean form $\le\varepsilon\|x_i\|\|x_j\|$ requires rescaling to unit vectors first — which is why the statement of Corollary 12.18 is phrased for the normalized version.

**A11.** Let $\mathcal{N}\subseteq S^{d-1}$ be a maximal $\varepsilon$-separated set (pairwise distances $>\varepsilon$); such a set exists by compactness, and maximality forces it to be an $\varepsilon$-net, since any point at distance $>\varepsilon$ from all of $\mathcal{N}$ could be added. The balls $B(x,\varepsilon/2)$, $x\in\mathcal{N}$, are pairwise disjoint and all contained in $B(0,1+\varepsilon/2)$. Comparing volumes and using $\lambda_d(B(0,r))=V_d(1)r^d$:
$$|\mathcal{N}|\cdot(\varepsilon/2)^d\ \le\ (1+\varepsilon/2)^d\quad\Longrightarrow\quad |\mathcal{N}|\ \le\ \Big(\frac{1+\varepsilon/2}{\varepsilon/2}\Big)^d=\Big(1+\frac2\varepsilon\Big)^d .$$
*Thinking:* the $d$-th power is the whole story, and it is why net cardinalities enter final bounds through a logarithm — $\log|\mathcal{N}|\le d\log(1+2/\varepsilon)$ is *linear* in $d$ and only logarithmic in $1/\varepsilon$. That asymmetry is exactly what makes fine nets affordable (see A12/C2).

**A12.** *Lemma 12.22:* proved in the text. *One-net variant:* let $v^\star$ attain $\|A\|=\|Av^\star\|$ and pick $v\in\mathcal{N}$ with $\|v-v^\star\|\le\varepsilon$. Then
$$\|A\|=\|Av^\star\|\le\|Av\|+\|A(v^\star-v)\|\le\|Av\|+\varepsilon\|A\|,$$
so $\|A\|\le(1-\varepsilon)^{-1}\max_{v\in\mathcal{N}}\|Av\|$.

*Which is better?* The one-net version has a smaller penalty, $(1-\varepsilon)^{-1}$ versus $(1-2\varepsilon)^{-1}$, and unions over only $(1+2/\varepsilon)^d$ events rather than $(1+2/\varepsilon)^{n+d}$ — both improvements. But the quantity to be bounded changes from $\langle u,Bv\rangle\sim\mathcal{N}(0,1)$, whose tail is exactly $e^{-r^2/2}$, to $\|Bv\|\sim\chi_n$, whose concentration is around $\sqrt n$ with the $\chi$-tail of Theorem 12.5. Running the one-net argument gives, with $L=\log(1+2/\varepsilon)$,
$$\|B\|\le\frac{\sqrt n+\sqrt{2Ld}+s}{1-\varepsilon}\cdot 1 ,$$
which is also of the form $C(\sqrt n+\sqrt d)$ but with an *asymmetric* constant: the $\sqrt n$ term is paid at constant $1/(1-\varepsilon)$ while the $\sqrt d$ term pays $\sqrt{2L}/(1-\varepsilon)$. For tall matrices ($n\gg d$) this is better than the two-net bound; for square ones the two are comparable. *Thinking:* the general principle is to discretize as few of the suprema as you can get away with, and to make the un-discretized one the one whose concentration you understand best.

**A13.** (a),(b) Substitute $x=1+\gamma-2\sqrt\gamma\cos\theta$... cleaner: put $x = 1+\gamma+2\sqrt\gamma\, y$ with $y\in[-1,1]$, so $b-x=2\sqrt\gamma(1-y)$ and $x-a=2\sqrt\gamma(1+y)$, hence $\sqrt{(b-x)(x-a)}=2\sqrt\gamma\sqrt{1-y^2}$ and $dx=2\sqrt\gamma\,dy$. Then
$$\int_a^b\rho_\gamma = \int_{-1}^1\frac{2\sqrt\gamma\sqrt{1-y^2}}{2\pi\gamma(1+\gamma+2\sqrt\gamma y)}\,2\sqrt\gamma\,dy=\frac2\pi\int_{-1}^1\frac{\sqrt{1-y^2}}{1+\gamma+2\sqrt\gamma y}\,dy .$$
The standard integral is
$$\int_{-1}^1\frac{\sqrt{1-y^2}}{c+2\kappa y}\,dy=\frac{\pi}{4\kappa^2}\Big(c-\sqrt{c^2-4\kappa^2}\Big)\qquad (c>2|\kappa|),$$
which with $c=1+\gamma$ and $\kappa=\sqrt\gamma$ gives $\frac{\pi}{4\gamma}\big((1+\gamma)-|1-\gamma|\big)=\frac{\pi}{4\gamma}\cdot2\gamma=\frac\pi2$ for $\gamma<1$. Hence $\int_a^b\rho_\gamma=\frac2\pi\cdot\frac\pi2=1$. (Verified numerically: the integral equals $\pi/2$ to six decimals at $\gamma=0.1,0.3,0.7$ — a check worth doing, because the constant $\frac1{4\kappa^2}$ is easy to misremember as $\frac1{2\kappa^2}$, which would give the answer $2$ instead of $1$.) For the mean, note $\int x\rho_\gamma(x)dx=\frac1{2\pi\gamma}\int_a^b\sqrt{(b-x)(x-a)}\,dx=\frac1{2\pi\gamma}\cdot\frac\pi8(b-a)^2=\frac{(4\sqrt\gamma)^2}{16\gamma}=1$, using $\int_a^b\sqrt{(b-x)(x-a)}dx=\pi(b-a)^2/8$ (the area of a semicircle of radius $(b-a)/2$).

(c) $b/a=\big(\frac{1+\sqrt\gamma}{1-\sqrt\gamma}\big)^2$: at $\gamma=0.1$, $3.71$; $\gamma=0.5$, $33.97$; $\gamma=0.8$, $322.00$; $\gamma=0.99$, $158402$.

(d) As $\gamma\to0$, $b/a\to1$: with vastly more samples than dimensions the sample covariance is well conditioned. As $\gamma\to1$, $a\to0$ so $b/a\to\infty$. *Thinking:* part (b) is the sanity check that MP is consistent with $\mathbb{E}\hat\Sigma=I$ — the mean of the limiting spectrum is exactly $1$, i.e. the estimator is unbiased *on average over eigenvalues* while being wrong in every individual one. That is Common Confusion 12.8 in a formula.

**A14.** We need $2\sqrt\gamma+\gamma\le0.1$. Solve $\gamma+2\sqrt\gamma-0.1=0$ with $u=\sqrt\gamma$: $u^2+2u-0.1=0$, $u=-1+\sqrt{1.1}=0.048809$, so $\gamma\le0.0023823$, i.e. $n\ge419.8\,d$. For $d=2048$: $n\ge859{,}672$ samples. *Thinking:* this is the number that stops people. To estimate a $2048$-dimensional feature covariance to $10\%$ in operator norm you need close to a million samples — far more than the batch sizes used by any second-order optimizer, and more than the per-class counts available for Mahalanobis OOD detection on ImageNet ($1300$ per class). Every practical method must therefore either impose structure or shrink; there is no third option.

**A15.** Set $\bar\mu\to\mu$ as in the text. Decompose
$$\hat\Sigma_\rho-\Sigma=(1-\rho)\hat\Sigma+\rho\mu I-\Sigma=(1-\rho)(\hat\Sigma-\Sigma)-\rho(\Sigma-\mu I).$$
Take $\|\cdot\|_F^2$ and expectations. The cross term is
$$-2\rho(1-\rho)\,\mathbb{E}\big\langle\hat\Sigma-\Sigma,\ \Sigma-\mu I\big\rangle_F = -2\rho(1-\rho)\big\langle\mathbb{E}\hat\Sigma-\Sigma,\ \Sigma-\mu I\big\rangle_F=0,$$
because $\Sigma-\mu I$ is deterministic (so it passes through the expectation) and $\mathbb{E}\hat\Sigma=\Sigma$. Hence
$$R(\rho):=\mathbb{E}\|\hat\Sigma_\rho-\Sigma\|_F^2=(1-\rho)^2\beta^2+\rho^2\alpha^2 .$$
$R''=2(\alpha^2+\beta^2)>0$, so $R$ is strictly convex; $R'(\rho)=-2(1-\rho)\beta^2+2\rho\alpha^2=0$ gives $\rho^\star=\beta^2/(\alpha^2+\beta^2)\in(0,1)$. Substituting,
$$R(\rho^\star)=\Big(\frac{\alpha^2}{\alpha^2+\beta^2}\Big)^2\beta^2+\Big(\frac{\beta^2}{\alpha^2+\beta^2}\Big)^2\alpha^2=\frac{\alpha^2\beta^2(\alpha^2+\beta^2)}{(\alpha^2+\beta^2)^2}=\frac{\alpha^2\beta^2}{\alpha^2+\beta^2},$$
which is the harmonic-type mean and is $<\min(\alpha^2,\beta^2)$ whenever both are positive. Since $R(0)=\beta^2$ is the risk of $\hat\Sigma$, shrinkage strictly dominates.

If $\Sigma=I$ exactly then $\mu=1$ and $\alpha^2=\|I-I\|_F^2=0$, so $\rho^\star=1$: shrink all the way to the target, which is correct because the target *is* the truth. Worked Example 12.21 measured $\hat\rho\in[0.964,1.000]$ in exactly this situation.

**A16.** $\Pr(\|Z\|\le1)=\int_{B_d(1)}(2\pi)^{-d/2}e^{-\|x\|^2/2}dx$. Bound the integrand between $(2\pi)^{-d/2}e^{-1/2}$ and $(2\pi)^{-d/2}$, so
$$(2\pi)^{-d/2}e^{-1/2}V_d(1)\ \le\ \Pr(\|Z\|\le1)\ \le\ (2\pi)^{-d/2}V_d(1).$$
Taking logs and using A2's $\log V_d(1)=-\tfrac d2\log d+O(d)$:
$$\log\Pr(\|Z\|\le1)=-\frac d2\log(2\pi)-\frac d2\log d+O(d)=-\frac d2\log d+O(d).$$
Check at $d=100$: the formula's leading term is $-50\log100=-230.26$ nats $=-100.0$ in $\log_{10}$; adding $-\tfrac d2\log(2\pi)+\tfrac d2\log\pi+\tfrac d2 = -\tfrac d2\log 2+\tfrac d2$ from the constants gives $\log_{10}$ correction $\approx+\tfrac{100}2(1-0.693)/2.3026=+6.67$, hmm — more carefully, combining exactly: $\log_{10}\Pr\approx\log_{10}V_{100}(1)-50\log_{10}(2\pi)=-39.63-39.90=-79.53$, against the exact $-79.75$. Excellent agreement, and the $0.22$ gap is the $e^{-1/2}$-to-$1$ slack in the integrand bound ($\log_{10}$ of which is $0.217$) — so the crude bracket is essentially exact here. *Thinking:* the two-sided bracket is worth the extra line, because it converts an asymptotic statement into a numerical check, which is how you catch algebra errors in this kind of computation.

**A17.** (a) If $Q$ is orthogonal then $(cQ)^\top(cQ)=c^2Q^\top Q=c^2I$, so every eigenvalue of $(cQ)^\top(cQ)$ is $c^2$ and every singular value of $cQ$ is $c$. Hence $\|cQh\|^2=h^\top(c^2I)h=c^2\|h\|^2$.

(b) The event $\{M^\top M=c^2I\}$ is contained in the single off-diagonal constraint $E:=\{\sum_kM_{k1}M_{k2}=0\}$, which is the zero set of the polynomial $p(M)=\sum_kM_{k1}M_{k2}$ on $\mathbb{R}^{n^2}$. This polynomial is not identically zero for $n\ge2$ (e.g. it equals $1$ at $M_{11}=M_{12}=1$, all other entries $0$). The zero set of a non-zero polynomial on $\mathbb{R}^{n^2}$ is a Lebesgue-null set. If the entries have a joint density — which they do, being i.i.d. with a density — then the law of $M$ is absolutely continuous with respect to $\lambda_{n^2}$, so it assigns probability $0$ to every Lebesgue-null set (this is exactly what $\ll$ means; Chapter 2's Radon–Nikodym language). Hence $\Pr(E)=0$ and a fortiori $\Pr(M^\top M=c^2I)=0$.

(c) With $n=2$ and i.i.d. Rademacher entries there are $2^4=16$ equally likely matrices. The condition $M^\top M=2I$ requires both columns to have squared norm $2$ — automatic, since every entry is $\pm1$ — and the columns to be orthogonal: $M_{11}M_{12}+M_{21}M_{22}=0$. Each summand is $\pm1$, so the sum is $0$ exactly when the two summands have opposite signs, which happens for $8$ of the $16$ sign patterns. Hence
$$\Pr(M^\top M=2I)=\tfrac8{16}=\tfrac12 .$$
(Exhaustive enumeration confirms $8/16$.) These eight matrices are precisely the $2\times2$ Hadamard matrices, and $\tfrac1{\sqrt2}M$ is orthogonal for each of them.

*Thinking, and the moral.* Part (b) is the statement people mean when they say "a random matrix is never an isometry," and it is true for continuous entry distributions — which covers Gaussian and uniform initialization. Part (c) shows the slogan is *false* as a statement about i.i.d. entries in general: for discrete distributions the null-set argument has nothing to bite on, and here the probability is not small but exactly one half. This is worth internalizing beyond the exercise, because the same gap appears whenever a proof says "a nontrivial polynomial constraint, hence measure zero": that inference silently assumes absolute continuity. It is also why orthogonal initialization is implemented by generating a Gaussian matrix and *then* orthogonalizing (QR or SVD) rather than by hunting for a clever entrywise distribution — for continuous entries, orthogonality is a probability-zero event, so it must be imposed, not sampled.

**A18.** (a) Set $Y_i:=\tfrac1n(X_iX_i^\top-\Sigma)$, so $\hat\Sigma-\Sigma=\sum_{i=1}^nY_i$ with the $Y_i$ independent, mean-zero, symmetric. *Boundedness:* $\|X_iX_i^\top\|=\|X_i\|^2\le B$ (the operator norm of a rank-one matrix $xx^\top$ is $\|x\|^2$), and $\|\Sigma\|=\sup_{\|u\|=1}\mathbb{E}\langle X,u\rangle^2\le\mathbb{E}\|X\|^2\le B$ by Cauchy–Schwarz, so $\|Y_i\|\le\tfrac1n(B+\|\Sigma\|)\le\tfrac{2B}n=:K$. *The auxiliary inequality:* for any unit $u$, $u^\top\mathbb{E}[\|X\|^2XX^\top]u=\mathbb{E}[\|X\|^2\langle X,u\rangle^2]\le B\,\mathbb{E}\langle X,u\rangle^2=B\,u^\top\Sigma u$, using $\|X\|^2\le B$ almost surely; since both matrices are symmetric this is exactly $\mathbb{E}[\|X\|^2XX^\top]\preceq B\Sigma$. *Variance:* $\mathbb{E}Y_i^2=\tfrac1{n^2}\big(\mathbb{E}[\|X\|^2XX^\top]-\Sigma^2\big)\preceq\tfrac1{n^2}\mathbb{E}[\|X\|^2XX^\top]\preceq\tfrac B{n^2}\Sigma$ (dropping $-\Sigma^2\preceq0$), so $\sigma^2=\big\|\sum_i\mathbb{E}Y_i^2\big\|\le\tfrac{B\|\Sigma\|}n$. Theorem 12.24 gives $\Pr(\|\hat\Sigma-\Sigma\|\ge t)\le2d\exp\big(\tfrac{-t^2/2}{\sigma^2+Kt/3}\big)$. To invert, set $L:=\log(2d/\delta)$ and claim $t^\star:=\sqrt{2\sigma^2L}+\tfrac23KL$ makes the exponent at most $-L$: indeed $\tfrac12(t^\star)^2=\sigma^2L+\tfrac23KL\sqrt{2\sigma^2L}+\tfrac29K^2L^2$, while $L(\sigma^2+Kt^\star/3)=\sigma^2L+\tfrac13KL\sqrt{2\sigma^2L}+\tfrac29K^2L^2$, and the first quantity exceeds the second by $\tfrac13KL\sqrt{2\sigma^2L}\ge0$. So with probability at least $1-\delta$, substituting $\sigma^2\le B\|\Sigma\|/n$ and $K=2B/n$,
$$\|\hat\Sigma-\Sigma\|\ \le\ \sqrt{\frac{2B\|\Sigma\|\log(2d/\delta)}{n}}\ +\ \frac{4B\log(2d/\delta)}{3n},$$
which is the claimed shape. *Thinking:* the whole computation is bookkeeping once you notice that the rank-one structure makes both parameters computable — $\|xx^\top\|$ is a norm you know, and the fourth-moment matrix is tamed by pulling $\|X\|^2\le B$ out of the expectation.

(b) Demand each term be at most $\tfrac\varepsilon2\|\Sigma\|$. The first requires $n\ge8\varepsilon^{-2}(B/\|\Sigma\|)\log(2d/\delta)$; the second requires $n\ge\tfrac{8}{3}\varepsilon^{-1}(B/\|\Sigma\|)\log(2d/\delta)$, which is weaker for $\varepsilon\le1$. So
$$n\ \gtrsim\ \varepsilon^{-2}\,\frac{B}{\|\Sigma\|}\,\log\frac d\delta$$
samples suffice, with $r:=B/\|\Sigma\|$ playing the role of an effective dimension. Sanity checks: if $X$ is uniform on the sphere of radius $\sqrt d$ (so $\Sigma=I$, $B=d$), then $r=d$ and the requirement is $n\gtrsim\varepsilon^{-2}d\log d$ — the Gaussian answer of Proposition 12.26(2), $n\gtrsim\varepsilon^{-2}d$, up to the $\log d$ that the union-bound prefactor $2d$ costs. But if the mass of $X$ concentrates on a few directions — say $\Sigma$ has one large eigenvalue carrying most of $\mathbb{E}\|X\|^2$ — then $r\ll d$ and Bernstein *beats* the worst-case dimension count. That trade ($\log d$ penalty, structure-adaptive rate, no Gaussianity) is the standard reason to reach for Matrix Bernstein.

(c) At $\gamma=0.25$: threshold $1+\sqrt{0.25}=1.5$. For $\lambda=2$ (supercritical, since $2>1.5$): limiting top eigenvalue $\lambda(1+\gamma/(\lambda-1))=2(1+0.25)=2.5$, and overlap $\dfrac{1-\gamma/(\lambda-1)^2}{1+\gamma/(\lambda-1)}=\dfrac{1-0.25}{1.25}=0.6$. (Simulation check, $d=500$, $n=2000$, `default_rng(5)`: observed $2.444$ and $0.578$ in a single replication — the right order, with finite-$n$ fluctuation.) For the last part, set $m:=\lambda-1$ and solve $\dfrac{1-0.25/m^2}{1+0.25/m}=0.9$, i.e. $0.1m^2-0.225m-0.25=0$, giving $m=\tfrac{2.25+\sqrt{15.0625}}{2}=3.0655$ and hence $\lambda^\star=4.066$. *Thinking:* the striking number is the last one — to recover the direction with $90\%$ overlap you need the signal eigenvalue to be more than *four times* the noise level, at a comfortable four samples per dimension. Eigenvector recovery is much more demanding than eigenvalue detection, and this is the quantitative reason "PCA found the factor" claims deserve an error bar.

## Part B

**B1.** The point of the lab is to use special functions rather than simulation, since the target probabilities reach $10^{-17831}$.

```python
import numpy as np
from scipy.special import gammaln
from scipy.stats import beta
from mpmath import mp, mpf, gammainc, log10 as mlog10
mp.dps = 40

def logvol(d, r=1.0):
    return (d/2)*np.log(np.pi) - gammaln(d/2 + 1) + d*np.log(r)

for d in [2, 10, 100, 1000, 10000]:
    shell  = 1 - 0.9**d
    lv     = logvol(d)/np.log(10)
    lcube  = logvol(d, 0.5)/np.log(10)          # inscribed ball / unit cube
    Enorm  = np.sqrt(2)*np.exp(gammaln((d+1)/2) - gammaln(d/2))
    sdn    = np.sqrt(d - Enorm**2)
    pang   = beta.cdf(np.sin(np.deg2rad(5))**2, 0.5, (d-1)/2)
    lpball = float(mlog10(gammainc(mpf(d)/2, 0, mpf(1)/2, regularized=True)))
    print(d, shell, lv, lcube, pang, Enorm, np.sqrt(d), sdn, lpball)
```
Output reproduces the table of §12.1.4. *Three things to notice while doing this.* First, `np.exp(logvol(d))` underflows to $0.0$ for $d\ge1000$; work in logs throughout. Second, `scipy.stats.chi2.cdf(1, d)` returns exactly $0.0$ for $d\ge400$ or so, which is why the last column needs `mpmath` — a plain float pipeline silently reports "probability zero" for something that is merely $10^{-1285}$, and the difference matters if you ever divide by it. Third, the angle probability is best computed from the exact Beta CDF; Monte Carlo cannot resolve $1-2.5\times10^{-18}$.

**B2.** 
```python
import numpy as np
from scipy.stats import chi2
rng = np.random.default_rng(2024)
for d in [10, 100, 1000]:
    n = 500_000 if d <= 100 else 100_000
    nm = np.linalg.norm(rng.standard_normal((n, d)), axis=1)
    s = np.sqrt(d)
    for t in [1.0, 2.0, 3.0]:
        exact = chi2.cdf(max(s-t,0)**2, d) + chi2.sf((s+t)**2, d)
        print(d, t, np.mean(np.abs(nm-s) >= t), exact,
              2*np.exp(-t*t/8), 2*np.exp(-t*t/2))
```
Results (seed 2024; $n=5\times10^5$ for $d\le100$, $10^5$ for $d=1000$):

| $d$ | $t$ | Monte Carlo | exact | $2e^{-t^2/8}$ | $2e^{-t^2/2}$ |
|---|---|---|---|---|---|
| 10 | 1 | 0.156212 | 0.155706 | 1.7650 | 1.2131 |
| 10 | 2 | 0.003736 | 0.003629 | 1.2131 | 0.2707 |
| 10 | 3 | 0.000034 | 0.000038 | 0.6493 | 0.0222 |
| 100 | 1 | 0.156912 | 0.157180 | 1.7650 | 1.2131 |
| 100 | 2 | 0.004630 | 0.004563 | 1.2131 | 0.2707 |
| 100 | 3 | 0.000022 | 0.000024 | 0.6493 | 0.0222 |
| 1000 | 1 | 0.157690 | 0.157288 | 1.7650 | 1.2131 |
| 1000 | 2 | 0.005080 | 0.004666 | 1.2131 | 0.2707 |
| 1000 | 3 | 0.000010 | 0.000022 | 0.6493 | 0.0222 |

*Discussion.* The exact column is essentially constant down each $t$-block — that is the dimension-freeness of Theorem 12.5, visible in data. Both bounds exceed $1$ (and so are vacuous) at $t=1$; $2e^{-t^2/2}$ becomes informative from $t=2$, while $2e^{-t^2/8}$ needs $t\gtrsim3$ and even at $t=3$ is off by four orders of magnitude. Lesson: the weaker constant we could prove is fine for asymptotic arguments and useless for numerics; when you need numbers, use the exact $\chi^2$ CDF, and use bounds only where you need them to be uniform in $d$ inside a proof. Note also the $d=1000$, $t=3$ row, where Monte Carlo with $10^5$ draws sees only one event and is therefore unreliable — a reminder that simulation cannot verify tail bounds without enormous sample sizes.

**B3.**
```python
import numpy as np
from scipy.spatial.distance import pdist
rng = np.random.default_rng(1234)
n, d, eps = 500, 2000, 0.2
X  = rng.standard_normal((n, d)); D0 = pdist(X)
kstar = int(np.ceil(8*np.log(n)/eps**2))          # 1243
for k in [16, 32, 64, 128, 256, 512, 1024, kstar]:
    R = rng.standard_normal((d, k))/np.sqrt(k)
    r = np.abs(pdist(X @ R)/D0 - 1)
    print(k, r.max(), np.percentile(r, 95), r.mean())
```
The table appears as Worked Example 12.14. Sweeping $k$ finds worst-case distortion $0.2637$ at $k=150$, $0.2286$ at $200$, $0.1999$ at $256$ — so the smallest power-of-two $k$ meeting $\varepsilon=0.2$ is $256$, and $k^\star/k=1243/256=4.86$.

The $d$-independence check with $k=200$ fixed: $d=200\to0.2290$, $d=1000\to0.2625$, $d=5000\to0.2210$, $d=20000\to0.2075$. No trend, only sampling noise across the four independent draws.

*Thinking about what could go wrong.* Two implementation traps. (i) If you forget the $1/\sqrt k$ scaling, all distances inflate by $\approx\sqrt k$ and the "distortion" is enormous; check $\mathbb{E}\|Rx\|^2=\|x\|^2$ first. (ii) If you reuse the *same* $R$ across $k$ values by slicing columns, the results become correlated and the monotone decrease looks cleaner than it should; draw a fresh $R$ per $k$, as above.

**B4.**
```python
import numpy as np
rng = np.random.default_rng(555)
for g in [0.1, 0.5, 1.0]:
    n = 4000; d = int(g*n)
    s = np.linalg.svd(rng.standard_normal((n, d))/np.sqrt(n), compute_uv=False)
    print(g, s.min(), 1-np.sqrt(g), s.max(), 1+np.sqrt(g))
```
The four printed numbers per row are exactly Worked Example 12.16's table, so I do not repeat it. For the density overlay, histogram the eigenvalues of $A^\top A$ with `density=True` and plot
$$\rho_\gamma(x)=\frac{\sqrt{(b-x)(x-a)}}{2\pi\gamma x},\quad a=(1-\sqrt\gamma)^2,\ b=(1+\sqrt\gamma)^2 .$$
At $\gamma=1$ the density diverges like $x^{-1/2}$ at $0$, so the first histogram bin will always look "wrong" — that is the integrable singularity, not a bug. Verified at $n=3000$, seed 2718: $\gamma=0.2$ gives observed $[\lambda_{\min},\lambda_{\max}]=[0.3069,2.1051]$ against theory $[0.3056,2.0944]$; $\gamma=1$ gives $[0.0000,3.9877]$ against $[0,4]$.

**B5.** 
```python
import numpy as np
from scipy.spatial.distance import cdist
rng = np.random.default_rng(11)
n, k = 1000, 5
def contrast(Q, X):
    D = cdist(Q, X); return (D.max(1)/D.min(1)).mean()
for d in [10, 100, 1000, 10000]:
    iso = contrast(rng.standard_normal((100, d)), rng.standard_normal((n, d)))
    W = rng.standard_normal((k, d))/np.sqrt(k); b = rng.uniform(0, 2*np.pi, d)
    emb = lambda Z: np.sqrt(2.0)*np.cos(Z @ W + b)
    man = contrast(emb(rng.standard_normal((100, k))),
                   emb(rng.standard_normal((n, k))))
    print(d, iso, man)
```
Contrast (seed 11): isotropic $4.018,\,1.499,\,1.136,\,1.041$; manifold $8.975,\,5.936,\,5.150,\,5.132$. The $1$-NN error experiment (seed 303, $n_{\text{tr}}=2000$, $n_{\text{te}}=1000$, label $\mathbb{1}\{\sin 2v_1+v_2v_3-v_4+0.5v_5>0\}$): isotropic $0.236\to0.443\to0.482\to0.477$ for $d=10,100,1000,5000$; manifold $0.184\to0.119\to0.127\to0.118$.

*The concluding question.* For a real image-embedding dataset you should expect the manifold-like curve, and the way to check is direct: (i) estimate intrinsic dimension with the TwoNN or Levina–Bickel MLE estimator and compare it with $d$; (ii) plot the histogram of pairwise distances and compute $(D_{\max}-D_{\min})/D_{\text{median}}$ — a value near zero means contrast is dead; (iii) compute the participation ratio $(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ of the feature covariance spectrum, which estimates the effective number of directions carrying variance. All three are a few lines of code, and all three should be run before blaming an ANN index for poor recall.

**B6.** With seed 2024 and $2\times10^4$ draws: $f(x)=\max_ix_i$ gives $(\mathbb{E},\mathrm{sd})=(1.5476,0.5867)$, $(2.5130,0.4314)$, $(3.2455,0.3515)$, $(3.8519,0.3031)$ at $d=10,10^2,10^3,10^4$, against $\sqrt{2\log d}=2.146,3.035,3.717,4.292$. The bound $\mathrm{sd}\le1$ holds at every $d$ with a growing margin.

For $g(x)=\frac1{\sqrt d}\sum_ix_i$: exactly $g(Z)\sim\mathcal{N}(0,1)$ for every $d$, so $\mathbb{E}g=0$ and $\mathrm{sd}(g)=1$ identically — the bound is *attained*. *Why so different.* Both are $1$-Lipschitz, so both are covered by $\mathrm{sd}\le1$; but $g$ is linear, and Theorem 12.7 is sharp for linear functions (the extremal sets in the isoperimetric inequality are half-spaces, i.e. sublevel sets of linear functions). The maximum is far from linear: its gradient is a single coordinate direction that jumps around as $Z$ varies, and that "gradient decorrelation" is what buys the extra factor. The moral: a Lipschitz bound is a worst-case bound over functions, and the worst case is attained by linear functions.

**B7.** Using the estimator
$$\hat\rho=\min\Big(1,\ \frac{\frac1{n^2}\sum_i\|x_ix_i^\top-\hat\Sigma\|_F^2}{\|\hat\Sigma-\bar\mu I\|_F^2}\Big),$$
with $\Sigma$ log-spaced on $[0.2,5]$, $d=200$, $20$ replications, seed 4242, at $n=250$ ($\gamma=0.8$): $\|\hat\Sigma-\Sigma\|_{\text{op}}=4.659\pm0.268$, $\|\hat\Sigma_{\text{LW}}-\Sigma\|_{\text{op}}=2.722\pm0.050$, $\hat\rho=0.516$, median $\mathrm{cond}(\hat\Sigma^{-1}\Sigma)=297$, median $\mathrm{cond}(\hat\Sigma_{\text{LW}}^{-1}\Sigma)=13.77$, against the true $\mathrm{cond}(\Sigma)=25$. As $n$ grows with $d=200$ fixed you should observe $\hat\rho$ decreasing roughly monotonically toward $0$ and the two operator-norm errors converging — the plot of $\hat\rho$ against $d/n$ is close to increasing and concave, consistent with $\rho^\star=\beta^2/(\alpha^2+\beta^2)$ and $\beta^2\propto d^2/n$.

*What to notice.* The variance reduction (sd $0.268\to0.050$) is as valuable as the bias reduction: shrinkage makes the estimator *reproducible*, which is what matters when it sits inside an optimizer that runs thousands of times. And the whitening diagnostic is the one to report, because operator-norm error alone hides the fact that raw $\hat\Sigma$ makes conditioning *worse* than doing nothing.

**B8.** The implementation appears in the solution to nothing else, so here is the shape of it. Parameters $p\in\mathbb{R}^{241}$ pack $(W\in\mathbb{R}^{10\times20},\,b\in\mathbb{R}^{20},\,w\in\mathbb{R}^{20},\,c\in\mathbb{R})$; forward pass $H=\tanh(XW+b)$, prediction $Hw+c$, loss $\frac1{2n}\|r\|^2$ with $r$ the residual. The analytic gradient is
$$\nabla_W=X^\top\Delta,\quad \nabla_b=\Delta^\top\mathbf{1},\quad\nabla_w=H^\top\tfrac rn,\quad\nabla_c=\textstyle\sum_i\tfrac{r_i}n,\qquad \Delta=\big(\tfrac rn w^\top\big)\odot(1-H^2).$$
The Hessian is obtained column by column: $H_{:,i}=(\nabla f(p+\epsilon e_i)-\nabla f(p-\epsilon e_i))/(2\epsilon)$ with $\epsilon=10^{-5}$, then symmetrized as $(H+H^\top)/2$. That is $2P=482$ gradient evaluations, well under a second.

With seed 31415 and $2\times10^5$ full-batch steps at learning rate $1.0$ (final loss $3.05\times10^{-6}$) this reproduces Worked Example 12.20's table exactly; I do not repeat the numbers.

*Interpretation, and the honest version of it.* The initialization index of $0.5104$ matches the semicircle law's prediction of $\tfrac12$ to within Monte Carlo error at $P=241$ — good evidence that the Wigner heuristic describes *random* points in parameter space. The trained point is best described as a flat minimum: $77.6\%$ of eigenvalues have $|\lambda|<0.01$, the $26$ negative eigenvalues are all smaller in magnitude than the finite-difference error scale (which you can estimate by re-running with $\epsilon=10^{-4}$ and comparing), and the $20$ positive outliers match the $20$ hidden units. Two caveats you should state if you report this: one model, one seed, and a finite-difference Hessian whose accuracy near zero is limited — so "the negative eigenvalues are real" is not a claim this experiment can support, whereas "negative curvature is negligible relative to $\lambda_{\max}$" is.

## Part C

**C1.** *Setup and normalization.* Let $R\in\mathbb{R}^{k\times d}$ have i.i.d. $\mathcal{N}(0,1/k)$ entries and define $f(x)=Rx$. For fixed $x$, the $j$-th coordinate of $Rx$ is $\langle r_j,x\rangle$ where $r_j\sim\mathcal{N}(0,I_d/k)$; since a linear functional of a spherical Gaussian is Gaussian with variance $\|x\|^2/k$ (Ch. 5), we get $\langle r_j,x\rangle\sim\mathcal{N}(0,\|x\|^2/k)$ and, by independence of the rows,
$$\|Rx\|^2=\sum_{j=1}^k\langle r_j,x\rangle^2\ \stackrel{d}{=}\ \frac{\|x\|^2}{k}\,\chi^2_k .$$
Hence $\mathbb{E}\|Rx\|^2=\|x\|^2$: the variance $1/k$ per entry is exactly what makes the map an isometry in expectation. With variance $1$ we would have $\mathbb{E}\|Rx\|^2=k\|x\|^2$ and every distance would be inflated by $\sqrt k$.

*Single-vector concentration, and where it comes from.* The event $\{|\|Rx\|^2-\|x\|^2|\ge\varepsilon\|x\|^2\}$ is precisely $\{|W-k|\ge\varepsilon k\}$ for $W\sim\chi^2_k$. This is a deviation of a sum of $k$ i.i.d. $\chi^2_1$ variables. Each $\chi^2_1=G^2$ with $G$ standard Gaussian is *sub-exponential* rather than sub-Gaussian (its MGF exists only for $s<\tfrac12$, as computed in Lemma 12.4), so Bernstein's two-regime behaviour applies: by Lemma 12.4 with $t=k\varepsilon^2/8$,
$$\Pr\big(|W-k|\ge\varepsilon k\big)\ \le\ 2e^{-k\varepsilon^2/8},\qquad\varepsilon\in(0,1).$$
Two features deserve emphasis. (i) **The bound contains no $d$.** This is because $\|Rx\|^2$ depends on $x$ only through $\|x\|$, which is a consequence of the rotational invariance of the Gaussian: replacing $x$ by $Ox$ for orthogonal $O$ leaves the law of $Rx$ unchanged. If the entries of $R$ were, say, i.i.d. uniform on $[-a,a]$, rotational invariance would fail and the argument would need more work (it can be done — this is the sparse-JL literature — but the clean statement really is a Gaussian fact). (ii) **The exponent is quadratic in $\varepsilon$**, and this is the second-moment signature: the fluctuation of $W/k$ around $1$ has standard deviation $\sqrt{2/k}$, so demanding a deviation of $\varepsilon$ means demanding $\varepsilon/\sqrt{2/k}=\varepsilon\sqrt{k/2}$ standard deviations, and a Gaussian-type tail at $z$ standard deviations is $e^{-z^2/2}=e^{-k\varepsilon^2/4}$. Our constant $8$ instead of $4$ is the price of a clean proof.

*Union bound.* Apply the above to each of the $\binom n2$ vectors $x_i-x_j$ (using linearity, $R(x_i-x_j)=Rx_i-Rx_j$). By Proposition 1.8(2),
$$\Pr\Big(\exists\,i<j:\ \big|\|Rx_i-Rx_j\|^2-\|x_i-x_j\|^2\big|>\varepsilon\|x_i-x_j\|^2\Big)\ \le\ \binom n2\,2e^{-k\varepsilon^2/8}\ <\ n^2e^{-k\varepsilon^2/8}.$$
This is where $\log n$ is born: forcing the right side below $\delta$ requires $k\varepsilon^2/8>2\log n+\log(1/\delta)$.

*Final statement.* If
$$k\ \ge\ \frac8{\varepsilon^2}\Big(2\log n+\log\frac1\delta\Big)$$
then with probability at least $1-\delta$ all $\binom n2$ pairwise squared distances are preserved within $1\pm\varepsilon$. In particular $k=O(\varepsilon^{-2}\log n)$, independent of $d$.

*Is the union bound the dominant looseness?* Worked Example 12.14 measured that $k=256$ achieves worst-case distortion $0.2$ where the bound demands $1243$ — a factor $4.86$. Decompose the gap. The single-vector bound is loose by a factor of roughly $2$ in the exponent ($e^{-k\varepsilon^2/8}$ where the truth is closer to $e^{-k\varepsilon^2/4}$), which accounts for a factor $2$ in $k$. The remaining factor $\approx2.4$ is the union bound. And the reason the union bound is loose here is exactly Chapter 1's observation that the union bound is tight for *disjoint* events and progressively looser as events become positively dependent: all $\binom n2$ pair events are functions of the *same* matrix $R$, so if $R$ happens to be a good projection then all pairs succeed together, and if it is bad many fail together. The events are massively positively correlated, so treating them as disjoint over-counts the failure probability. Two further remarks for completeness: the bound cannot be improved by more than constants, since Larsen–Nelson show $k=\Omega(\varepsilon^{-2}\log n)$ is necessary; and the empirical factor of five is *configuration-dependent* — for $n$ points in adversarial position (e.g. an orthonormal simplex) the bound is much closer to tight, which is the sense in which JL is a worst-case theorem.

**C2.** *The bound.* Let $B\in\mathbb{R}^{n\times d}$ have i.i.d. $\mathcal{N}(0,1)$ entries. Fix $\varepsilon\in(0,\tfrac12)$ and take $\varepsilon$-nets $\mathcal{N}\subseteq S^{d-1}$, $\mathcal{M}\subseteq S^{n-1}$, of cardinalities at most $(1+2/\varepsilon)^d$ and $(1+2/\varepsilon)^n$ (A11). Write $L=\log(1+2/\varepsilon)$.

Step 1 (one pair). For unit $u,v$, $\langle u,Bv\rangle=\sum_{i,j}u_iv_jB_{ij}$ is a linear combination of i.i.d. standard Gaussians with coefficient $\ell^2$-norm $\big(\sum_{i,j}u_i^2v_j^2\big)^{1/2}=\|u\|\|v\|=1$, hence exactly $\mathcal{N}(0,1)$, so $\Pr(\langle u,Bv\rangle\ge r)\le e^{-r^2/2}$.

Step 2 (union). Over the $|\mathcal{M}||\mathcal{N}|\le e^{L(n+d)}$ pairs, $\Pr(\max_{u,v}\langle u,Bv\rangle\ge r)\le e^{L(n+d)-r^2/2}$.

Step 3 (choose $r$). Take $r=\sqrt{2L(n+d)}+s$. Since $(\alpha+\beta)^2\ge\alpha^2+\beta^2$ for $\alpha,\beta\ge0$, $r^2\ge2L(n+d)+s^2$, so the bound is $e^{-s^2/2}$.

Step 4 (net penalty). By Lemma 12.22, $\|B\|\le(1-2\varepsilon)^{-1}\max_{u,v}\langle u,Bv\rangle$. Using $\sqrt{n+d}\le\sqrt n+\sqrt d$,
$$\|B\|\ \le\ \underbrace{\frac{\sqrt{2L}}{1-2\varepsilon}}_{C_\varepsilon}\big(\sqrt n+\sqrt d\big)\ +\ \underbrace{\frac{1}{1-2\varepsilon}}_{K_\varepsilon}s\qquad\text{with probability}\ \ge1-e^{-s^2/2}.$$
Keep $K_\varepsilon$ explicit. Rescaling $s$ to make its coefficient $1$ changes the exponent to $-s^2/(2K_\varepsilon^2)$; the two forms are equivalent but only if you carry the change, and the constant is only $1.138$ at the optimum, so there is no reason to hide it.

*The trade-off.* Define
$$C(\varepsilon)\ :=\ \frac{\sqrt{2\log(1+2/\varepsilon)}}{1-2\varepsilon}.$$
Numerically:

| $\varepsilon$ | $(1-2\varepsilon)^{-1}$ | $\sqrt{2\log(1+2/\varepsilon)}$ | $C(\varepsilon)$ |
|---|---|---|---|
| 0.05 | 1.1111 | 2.7253 | 3.0281 |
| 0.0605 | 1.1379 | 2.6558 | 3.0220 |
| 0.10 | 1.2500 | 2.4676 | 3.0845 |
| 0.125 | 1.3333 | 2.3804 | 3.1739 |
| 0.20 | 1.6667 | 2.1899 | 3.6499 |
| 0.25 | 2.0000 | 2.0963 | 4.1926 |
| 0.40 | 5.0000 | 1.8930 | 9.4651 |

The minimum is $C=3.0220$ at $\varepsilon^\star=0.0605$ (bounded scalar minimization).

*Why the optimum is a fine net.* The two costs are structurally asymmetric. Refining the net multiplies its cardinality by a large factor, but cardinality enters the final bound only through $\sqrt{2\log|\mathcal{N}||\mathcal{M}|}$ — a *logarithm*, then a square root, so the penalty for a $100\times$ larger net is a factor $\sqrt{1+\log 100/(L(n+d))}$, which is negligible. Coarsening the net, by contrast, degrades the reconstruction factor $(1-2\varepsilon)^{-1}$, which *diverges* as $\varepsilon\to\tfrac12$. A cost that grows logarithmically against a benefit that diverges: the optimum is pushed hard toward small $\varepsilon$, stopping only because $\log(1+2/\varepsilon)\to\infty$ as $\varepsilon\to0$ too, at rate $\log(1/\varepsilon)$. The balance point, where $d\log C/d\varepsilon=0$, sits near $0.06$. The commonly seen choice $\varepsilon=1/4$ (giving the memorable $C\le2\sqrt{2\log9}\approx4.19$) is chosen for algebraic tidiness, not optimality, and costs $39\%$ in the constant.

*Comparison with Bai–Yin, and why one would use the net anyway.* Bai–Yin (Theorem 12.20) gives the sharp asymptotic $\|B\|/\sqrt n\to1+\sqrt\gamma$, i.e. constant exactly $1$; measured in Worked Example 12.17, $\|B\|/(\sqrt n+\sqrt d)\in[0.963,1.003]$. Our net bound is off by a factor of $3.02$. One prefers the net argument for four reasons: it is *non-asymptotic*, valid at every finite $n,d$ rather than in a limit; it comes with an explicit *tail* $e^{-s^2/2}$, which Bai–Yin does not supply; it is *robust*, extending verbatim to sub-Gaussian entries (replace Step 1's exact Gaussian tail by the sub-Gaussian one), to matrices with independent rows but dependent entries within a row, and to structured ensembles where no limiting spectral law is known; and it is *reusable* — the same four steps prove bounds on empirical covariance operators, on Rademacher processes, and on suprema of random quadratic forms. Sharp constants are for stating results; nets are for proving them.

**C3.** *A critical note on "deep learning works because high-dimensional loss landscapes have no bad local minima."*

The claim conflates three separable propositions: (1) that neural network losses have no local minima with loss substantially above the global minimum; (2) that in high dimension, critical points of generic functions are overwhelmingly saddles rather than minima; and (3) that the absence of bad minima is the *explanation* for the empirical success of gradient-based training. Each has a different status, and the slogan borrows credibility from the best-supported to prop up the least.

*(a) What is proved.* For **deep linear networks** with squared loss, every local minimum is a global minimum (Baldi–Hornik 1989 for one hidden layer; Kawaguchi 2016, Theorem 2.3, in depth). This is a real theorem, and it is about a model class that computes only linear maps, so it cannot be the reason nonlinear networks work — the object of interest has been removed. For **structured non-neural problems** — matrix completion, matrix sensing, phase retrieval, tensor decomposition, dictionary learning — the loss provably satisfies the *strict saddle property*, and perturbed gradient descent escapes all saddles in time polynomial in $1/\epsilon$ and logarithmic in dimension (Ge–Huang–Jin–Yuan 2015; Jin–Ge–Netrapalli–Kakade–Jordan 2017, Theorem 3). These are strong results, and the $\log d$ dependence is genuinely the high-dimensional-geometry contribution: a saddle needs only one escape direction, and in $d$ dimensions there are $d$ candidates, so finding one is cheap. But the objectives are not neural losses. For **wide networks in the NTK regime**, gradient descent provably reaches zero training loss (Jacot–Gabriel–Hongler 2018; Du et al. 2019; Allen-Zhu–Li–Song 2019). The width requirements are polynomial in $n$ and $1/\lambda_{\min}$ of the kernel and are orders of magnitude beyond anything trained; more fundamentally the linearized dynamics provably cannot learn features, so this regime describes a different learning phenomenon than the one we care about.

*In the opposite direction — and this is the part usually omitted — there are proofs that neural losses do have bad minima.* Safran and Shamir (2018) exhibit and count spurious local minima for two-layer ReLU networks fitting a ReLU teacher, with positive probability over initialization; Yun, Sra and Jadbabaie (2019) characterize their existence for one-hidden-layer networks. Any statement of the form "neural networks have no bad local minima" is therefore false as literally stated.

*(b) What is empirical.* Several robust observations, none proved. The Hessian at a trained solution has a large bulk of near-zero eigenvalues, a small number of large positive outliers whose count tracks the number of classes or of active features, and a negative tail of negligible magnitude relative to $\lambda_{\max}$ (Sagun–Evci–Güney–Dauphin–Bottou 2018; Ghorbani–Krishnan–Xiao 2019; Papyan 2019). My own measurement in Worked Example 12.20 reproduces this on a $241$-parameter network: index $0.51$ at initialization, versus $|\lambda_{\min}|/\lambda_{\max}=3\times10^{-4}$ and $78\%$ of directions flat after training. Independently trained solutions are frequently connected by low-loss paths (Garipov et al. 2018; Draxler et al. 2018) and, after accounting for permutation symmetry, sometimes by nearly linear ones (Frankle et al. 2020; Entezari et al. 2022). Loss curves along SGD trajectories are largely monotone with no evidence of trapping. These are observations on specific architectures (mostly ResNets and VGGs on CIFAR/ImageNet, plus small MLPs) at accessible scales; extrapolation to frontier-scale models is untested.

*(c) The random-matrix heuristic, and why its hypotheses fail.* The saddle argument runs: if the Hessian at a critical point resembled a Wigner matrix, its spectrum would follow the semicircle law (Theorem 12.21), symmetric about zero, so all $P$ eigenvalues being positive would have probability exponentially small in $P$. Dauphin et al. (2014) imported this from the physics of Gaussian random fields (Bray–Dean 2007; Fyodorov–Williams 2007). Worked Example 12.19 shows the mechanism and its refinement: shifting the semicircle by $cI$ drives the index from $0.50$ to $0.00$ as $c$ reaches the spectral edge, so low-loss critical points have low index and high-loss ones have index near $\tfrac12$ — the interesting version of the claim.

The hypotheses do not hold for neural losses, for reasons that are not technicalities. Neural losses have enormous exact symmetry groups: permuting hidden units in a layer ($h!$ elements), and for ReLU networks rescaling $W_\ell\to\alpha W_\ell$, $W_{\ell+1}\to\alpha^{-1}W_{\ell+1}$ (a continuous group). Continuous symmetries force the Hessian to have exact zero eigenvalues along the group orbits, so every critical point is a *manifold* of critical points, and the spectrum has an atom at zero of substantial mass — flatly contradicting the absolutely continuous semicircle law. The empirically observed spectrum reflects this: a spike at zero, not a semicircle. Moreover the Hessian has known low-rank-plus-structure form (Gauss–Newton plus a residual term), so its entries are strongly dependent, violating Wigner's independence hypothesis. The semicircle heuristic therefore describes random *initializations* — where my measured index of $0.51$ matches it well — and not trained solutions.

*(d) Conjecture.* That the absence of bad minima is *why* training works. I know of no evidence for the causal claim, and the more plausible picture is different: overparameterization creates a vast, connected, near-zero-loss set, and the scientifically interesting question is which element of that set SGD selects — the implicit-bias question — not whether it can reach one. On this account "no bad minima" is closer to a restatement of overparameterization than to an explanation of generalization, and it says nothing about the quantity we actually care about, which is test error.

*Statements I would defend in a paper.* "In high dimension, strict saddle points are easy to escape, with dimension entering only logarithmically (Jin et al. 2017)." "Empirically, the Hessian at trained solutions of standard image classifiers has a flat bulk, a small number of large positive outliers, and negative curvature negligible relative to the top eigenvalue." "For deep linear networks, all local minima are global (Kawaguchi 2016)."

*Statements I would not.* "Deep networks have no bad local minima." "Most critical points of the loss are saddles." "High-dimensional geometry explains why SGD generalizes." The first is contradicted by Safran–Shamir; the second applies a theorem about isotropic Gaussian fields to an object with large symmetry groups and dependent Hessian entries; the third conflates optimization with generalization, which are separate questions — the loss landscape governs whether you can fit the training set, and it is the implicit bias of the algorithm plus the structure of the data, not the census of critical points, that governs whether fitting it helps.

### References and further reading

**Textbooks and monographs.**

- R. Vershynin, *High-Dimensional Probability: An Introduction with Applications in Data Science*, Cambridge, 2018. The single best companion to this chapter. Ch. 2–3 for sub-Gaussian and norm concentration, Ch. 4 for nets and the operator-norm bound, Ch. 5 for Lipschitz concentration and isoperimetry, Ch. 9 for random matrices. Freely available from the author's page.
- S. Boucheron, G. Lugosi and P. Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence*, Oxford, 2013. The reference for Theorem 12.7 (Thm 5.6), the Gaussian Poincaré inequality (Thm 3.20), the entropy method, and Talagrand's inequalities (Ch. 8).
- M. Ledoux, *The Concentration of Measure Phenomenon*, AMS, 2001. Isoperimetry, the concentration function, Lévy's lemma, and the log-Sobolev route to Theorem 12.7.
- A. Blum, J. Hopcroft and R. Kannan, *Foundations of Data Science*, Cambridge, 2020, Ch. 2. The gentlest correct treatment of the geometry in §12.1, including the Gaussian annulus theorem and Johnson–Lindenstrauss.
- Z. Bai and J. Silverstein, *Spectral Analysis of Large Dimensional Random Matrices*, 2nd ed., Springer, 2010. Marchenko–Pastur (Thm 3.6) and Bai–Yin (Thm 5.11).
- G. Anderson, A. Guionnet and O. Zeitouni, *An Introduction to Random Matrices*, Cambridge, 2010. The semicircle law and the method of moments.
- J. Tropp, *An Introduction to Matrix Concentration Inequalities*, Foundations and Trends in ML, 2015. Matrix Bernstein and its applications.
- M. Wainwright, *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*, Cambridge, 2019. Covariance estimation, sparse recovery, and minimax rates in the $d\sim n$ regime.
- J. Matoušek, *Lectures on Discrete Geometry*, Springer, 2002, Ch. 14–15. Lévy's lemma and the combinatorial side of Johnson–Lindenstrauss.

**Papers.**

*Concentration and dimension reduction.* W. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space," 1984. S. Dasgupta and A. Gupta, "An elementary proof of a theorem of Johnson and Lindenstrauss," 2003 — the proof in §12.4 follows this. K. Larsen and J. Nelson, "Optimality of the Johnson–Lindenstrauss lemma," FOCS 2017. N. Alon and B. Klartag, "Optimal compression of approximate inner products," FOCS 2017. B. Laurent and P. Massart, "Adaptive estimation of a quadratic functional by model selection," Annals of Statistics 2000 — the source of inequality (12.1).

*Distance concentration and intrinsic dimension.* K. Beyer, J. Goldstein, R. Ramakrishnan and U. Shaft, "When is 'nearest neighbor' meaningful?", ICDT 1999 — the original statement of the contrast collapse. C. Aggarwal, A. Hinneburg and D. Keim, "On the surprising behavior of distance metrics in high dimensional space," ICDT 2001. E. Levina and P. Bickel, "Maximum likelihood estimation of intrinsic dimension," NeurIPS 2004. E. Facco, M. d'Errico, A. Rodriguez and A. Laio, "Estimating the intrinsic dimension of datasets by a minimal neighborhood information," Scientific Reports 2017 — the TwoNN estimator. P. Pope, C. Zhu, A. Abdelkader, M. Goldblum and T. Goldstein, "The intrinsic dimension of images and its impact on learning," ICLR 2021 — the ImageNet estimates quoted in §12.3.2.

*Hashing, sketching, random features.* M. Charikar, "Similarity estimation techniques from rounding algorithms," STOC 2002 — SimHash. A. Rahimi and B. Recht, "Random features for large-scale kernel machines," NeurIPS 2007. A. Andoni and P. Indyk, "Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions," FOCS 2006. H. Jégou, M. Douze and C. Schmid, "Product quantization for nearest neighbor search," TPAMI 2011 — the basis of FAISS.

*Random matrices in deep learning.* A. Saxe, J. McClelland and S. Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks," ICLR 2014. J. Pennington, S. Schoenholz and S. Ganguli, "Resurrecting the sigmoid in deep learning through dynamical isometry," NeurIPS 2017, and "The emergence of spectral universality in deep networks," AISTATS 2018. K. He, X. Zhang, S. Ren and J. Sun, "Delving deep into rectifiers," ICCV 2015 — the initialization whose spectrum §12.6.1 completes. J. Pennington and Y. Bahri, "Geometry of neural network loss surfaces via random matrix theory," ICML 2017.

*Loss landscapes — read these together with §12.6.2's labels.* Y. Dauphin, R. Pascanu, C. Gulcehre, K. Cho, S. Ganguli and Y. Bengio, "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization," NeurIPS 2014. K. Kawaguchi, "Deep learning without poor local minima," NeurIPS 2016. R. Ge, F. Huang, C. Jin and Y. Yuan, "Escaping from saddle points," COLT 2015. C. Jin, R. Ge, P. Netrapalli, S. Kakade and M. Jordan, "How to escape saddle points efficiently," ICML 2017. I. Safran and O. Shamir, "Spurious local minima are common in two-layer ReLU neural networks," ICML 2018. C. Yun, S. Sra and A. Jadbabaie, "Small nonlinearities in activation functions create bad local minima," ICLR 2019. L. Sagun, U. Evci, V. Güney, Y. Dauphin and L. Bottou, "Empirical analysis of the Hessian of over-parametrized neural networks," ICLR Workshop 2018. B. Ghorbani, S. Krishnan and Y. Xiao, "An investigation into neural net optimization via Hessian eigenvalue density," ICML 2019. T. Garipov, P. Izmailov, D. Podoprikhin, D. Vetrov and A. Wilson, "Loss surfaces, mode connectivity, and fast ensembling of DNNs," NeurIPS 2018. R. Entezari, H. Sedghi, O. Saukh and B. Neyshabur, "The role of permutation invariance in linear mode connectivity," ICLR 2022.

*Covariance estimation and the spiked model.* J. Baik, G. Ben Arous and S. Péché, "Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices," Annals of Probability 2005 — the BBP transition of Theorem 12.29. D. Paul, "Asymptotics of sample eigenstructure for a large dimensional spiked covariance model," Statistica Sinica 2007 — the eigenvector overlap formula. F. Benaych-Georges and R. Nadakuditi, "The eigenvalues and eigenvectors of finite, low rank perturbations of large random matrices," Advances in Mathematics 2011. M. Gavish and D. Donoho, "The optimal hard threshold for singular values is $4/\sqrt3$," IEEE Trans. IT 2014 — spike de-biasing turned into a usable recipe. O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," Journal of Multivariate Analysis 2004, and "Nonlinear shrinkage estimation of large-dimensional covariance matrices," Annals of Statistics 2012. I. Johnstone, "On the distribution of the largest eigenvalue in principal components analysis," Annals of Statistics 2001. M. Rudelson and R. Vershynin, "The least singular value of a random square matrix is $O(n^{-1/2})$," 2008. J. Martens and R. Grosse, "Optimizing neural networks with Kronecker-factored approximate curvature," ICML 2015 — K-FAC, whose damping is Definition 12.27. V. Gupta, T. Koren and Y. Singer, "Shampoo: preconditioned stochastic tensor optimization," ICML 2018. K. Lee, K. Lee, H. Lee and J. Shin, "A simple unified framework for detecting out-of-distribution samples and adversarial attacks," NeurIPS 2018 — the Mahalanobis OOD score discussed in §12.7.2.

*Robustness and the Lipschitz view.* J. Cohen, E. Rosenfeld and Z. Kolter, "Certified adversarial robustness via randomized smoothing," ICML 2019. A. Raghunathan, J. Steinhardt and P. Liang, "Certified defenses against adversarial examples," ICLR 2018. T. Miyato, T. Kataoka, M. Koyama and Y. Yoshida, "Spectral normalization for generative adversarial networks," ICLR 2018.
