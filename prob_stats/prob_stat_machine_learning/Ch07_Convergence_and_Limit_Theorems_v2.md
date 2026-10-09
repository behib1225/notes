# Chapter 7 — Convergence and Limit Theorems

## Chapter 7 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 7.1 Four modes of convergence | a.s., in probability, in $L^p$, in distribution — and exactly which implies which | Tells you what "the loss converges" can possibly mean; stops you proving the wrong thing |
| 7.2 Borel–Cantelli and a.s. convergence | Summable tails force almost-sure behaviour; independent divergent tails force infinitely-often behaviour | The standard device for upgrading a rate into an almost-sure guarantee (SGD a.s. convergence proofs) |
| 7.3 Laws of large numbers | Sample averages converge to expectations under $\mathbb{E}\lvert X\rvert<\infty$ and nothing more | Empirical risk $\to$ true risk *pointwise*; Monte Carlo consistency; why heavy tails break both |
| 7.4 Characteristic functions | Every distribution has one, it determines the law, and it turns convolution into multiplication | The engine of the CLT; also the tool that makes the multivariate case free |
| 7.5 The Central Limit Theorem | $\sqrt n(\bar X_n-\mu)\to_d\mathcal N(0,\sigma^2)$, with the Lindeberg generalization and the Berry–Esseen rate | Error bars, $A/B$ tests, and an honest account of when $n=30$ is a lie |
| 7.6 Multivariate CLT and the delta method | Push a limit through a smooth map: $\sqrt n(g(\hat\theta)-g(\theta))\to_d\mathcal N(0,\nabla g^\top\Sigma\nabla g)$ | Standard errors for ratios, log-odds, AUC, $R^2$ — anything that is a smooth function of averages |
| 7.7 Slutsky and continuous mapping | You may substitute a consistent estimate of a nuisance parameter without changing the limit | Why $\hat\theta\pm z\,\hat\sigma/\sqrt n$ is legitimate even though $\hat\sigma$ is random |
| 7.8 Empirical processes, briefly | Glivenko–Cantelli, DKW, Donsker: the empirical CDF converges *uniformly*, at rate $n^{-1/2}$ | The bridge from Chapter 4's uniform convergence to Chapter 9's bootstrap |
| 7.9 ML payoffs | SGD noise, the diffusion approximation, bootstrap validity, MLE asymptotics, and the fixed-$n$ caveat | Where the asymptotic story helps, and the three regimes where it silently fails |

---

## 7.0 What "converges" is going to mean

Up to now this book has been about *one* probability space at a time. Chapter 1 built the space, Chapter 2 built the integral, Chapters 3–5 built the objects you integrate, and Chapter 6 let information accumulate along a filtration. Nothing so far has required a *sequence* of random variables to settle down. This chapter is about exactly that, and it is where probability starts paying rent in machine learning, because essentially every quantitative claim an ML paper makes about a finite sample is a claim about a limit dressed up in a finite-$n$ costume.

Here is the difficulty, stated plainly. A random variable is a function on $\Omega$ (Chapter 1's insistence: all the randomness lives in $\omega$, the seed). So a sequence $X_1,X_2,\dots$ is a sequence of *functions*, and sequences of functions can converge in many inequivalent ways — you already met this in analysis, where pointwise, uniform, and $L^1$ convergence of functions are genuinely different. Probability adds one more mode that has no analysis analogue: convergence *in distribution*, which does not ask the functions to get close at all, only their pushforward laws. Four modes, twelve possible implications, and only four of them true. Getting this diagram right is the difference between a proof and a plausible-sounding paragraph.

What you should bring from earlier chapters. From Chapter 1, by number: Definition 1.25 (almost surely), Proposition 1.8 (monotonicity, the union bound, continuity from above and below), Definitions 1.16 and 1.18 (law and CDF), Theorem 1.20 (a measure is determined by its values on a generating $\pi$-system, hence the CDF determines the law), and your own Exercise A9, the first Borel–Cantelli lemma. From Chapter 2: monotone and dominated convergence, Fatou's lemma, the tail formula $\mathbb{E}[X]=\int_0^\infty \mathbb{P}(X>t)\,dt$ for $X\ge0$, and Fubini–Tonelli. From Chapter 3: Jensen's inequality and the moment generating function, including the fact that an MGF can fail to exist. From Chapter 4: Markov, Chebyshev, sub-Gaussianity, and Hoeffding — and, crucially, the *uniform* convergence machinery, because §7.3 will explain exactly why the law of large numbers is not enough on its own. From Chapter 5: the multivariate Gaussian and its characterization by mean and covariance. From Chapter 6: martingales and Azuma's inequality. I will cite all of these by name and chapter and will not re-derive them.

The punchline, stated in advance, is a two-line answer to "how big is $\bar X_n - \mu$?". The law of large numbers says it goes to zero. The central limit theorem says it goes to zero at rate exactly $n^{-1/2}$, with a *universal* limiting shape — Gaussian — that does not remember the parent distribution except through one number, $\sigma$. Berry–Esseen says the approximation error in that shape is itself $O(n^{-1/2})$, with a constant that depends on the third absolute moment. And the delta method says all of this survives being pushed through any smooth function. That chain — LLN, CLT, Berry–Esseen, delta method — is the entire asymptotic toolkit, and once you have it, most of the "$\pm$" symbols in the empirical sections of ML papers become derivable rather than mysterious.

There is a second punchline, less often stated. Every theorem in this chapter is a statement about $n\to\infty$ with everything else held fixed, and that is not the regime modern machine learning lives in. Models have millions of parameters; $p$ grows with $n$; the "samples" in a minibatch are drawn from a training set that is itself a sample; the estimator is the output of a nonconvex optimizer with no closed form. I will be explicit throughout about which assumptions each theorem needs and which of them ML routinely violates. Chapter 12 takes up the $p/n\to\gamma$ regime properly. For now, treat asymptotic normality as a very good default that you must be able to argue for, not assume.

Finally, a word on the shape of §7.1. It is unusually counterexample-heavy, on purpose. The implication diagram among the four modes is small enough to memorize and useless if you memorize it without the counterexamples, because the counterexamples are what tell you *which hypothesis is doing the work* when a proof succeeds. Three of the four counterexamples in §7.1 are built from the same two gadgets — a spike that gets taller as it narrows, and a block that slides along $[0,1]$ — and once you have those two pictures you can reconstruct the entire diagram from scratch on an exam or in a referee report.

---

## 7.1 Four modes of convergence

### 7.1.1 The definitions

Throughout, $X, X_1, X_2, \dots$ are random variables on a common probability space $(\Omega,\mathcal{F},\mathbb{P})$ — with one important exception noted below. Intuition first.

*Almost sure convergence* is the strictest and the most literal: for almost every seed $\omega$, the actual number sequence $X_1(\omega), X_2(\omega),\dots$ converges to the number $X(\omega)$. If you fix the random seed and watch, you see convergence. *Convergence in probability* is weaker and subtler: at each fixed time $n$, the chance of being far from the limit is small — but nothing forbids the sequence from misbehaving at *some* time far in the future, as long as which time it is keeps moving. *Convergence in $L^p$* is about average size: the $p$-th moment of the discrepancy vanishes. It is sensitive to rare-but-enormous excursions in a way the other two are not. *Convergence in distribution* is different in kind: it says nothing about the random variables, only about their laws.

**Definition 7.1 (almost sure convergence).** $X_n \to X$ almost surely, written $X_n \to_{\mathrm{as}} X$, if
$$\mathbb{P}\Big(\big\{\omega\in\Omega : \lim_{n\to\infty}X_n(\omega)=X(\omega)\big\}\Big)=1.$$
The set in question is measurable: it equals $\bigcap_{k\ge1}\bigcup_{N\ge1}\bigcap_{n\ge N}\{|X_n-X|\le 1/k\}$, a countable combination of events, so Definition 1.3 applies. (Countability is load-bearing here exactly as Chapter 1 warned.)

**Definition 7.2 (convergence in probability).** $X_n\to_{\mathrm p} X$ if for every $\varepsilon>0$,
$$\lim_{n\to\infty}\mathbb{P}(|X_n-X|>\varepsilon)=0.$$

**Definition 7.3 (convergence in $L^p$).** For $p\ge1$, $X_n\to_{L^p}X$ if $X_n,X\in L^p(\mathbb{P})$ and
$$\lim_{n\to\infty}\mathbb{E}\big[|X_n-X|^p\big]=0,\qquad\text{equivalently}\qquad \|X_n-X\|_p\to0 .$$
The cases $p=1$ (convergence in mean) and $p=2$ (convergence in mean square) are the ones you will meet.

**Definition 7.4 (convergence in distribution).** Let $F_n$ and $F$ be the CDFs of $X_n$ and $X$ (Definition 1.18). Then $X_n\to_{\mathrm d}X$ if
$$F_n(x)\longrightarrow F(x)\qquad\text{at every }x\text{ where }F\text{ is continuous.}$$
Equivalently — and this is often the more useful form — $\mathbb{E}[f(X_n)]\to\mathbb{E}[f(X)]$ for every bounded continuous $f:\mathbb{R}\to\mathbb{R}$. Because Definition 7.4 involves only the laws $P_{X_n}$ and $P_X$, the random variables need not live on the same probability space at all; one often writes $X_n\to_{\mathrm d}P$ or $X_n\to_{\mathrm d}\mathcal N(0,1)$, naming the limit law rather than a limit variable.

Two remarks on Definition 7.4 that students trip over. First, why exclude discontinuity points of $F$? Because otherwise the definition would fail on examples where it obviously ought to succeed. Take $X_n=1/n$ deterministically, so $P_{X_n}=\delta_{1/n}$ and $X=0$, $P_X=\delta_0$. Then $F_n(x)=\mathbf 1\{x\ge 1/n\}$ and $F(x)=\mathbf 1\{x\ge0\}$. At $x=0$ we have $F_n(0)=0$ for every $n$ but $F(0)=1$: no convergence. Yet $\delta_{1/n}\to\delta_0$ is exactly what any reasonable notion of "the laws converge" should say. The point $x=0$ is the single discontinuity of $F$, and excluding it repairs everything. Second, the "bounded continuous test functions" formulation makes the reason transparent: $\mathbf 1\{x\ge0\}$ is not continuous, so it is not an admissible test function, whereas every bounded continuous $f$ satisfies $f(1/n)\to f(0)$.

**Theorem 7.5 (Portmanteau).** The following are equivalent for probability measures $P_n, P$ on $\mathbb{R}^d$:

1. $\int f\,dP_n \to \int f\,dP$ for all bounded continuous $f$;
2. $\int f\,dP_n \to \int f\,dP$ for all bounded Lipschitz $f$;
3. $\limsup_n P_n(C)\le P(C)$ for all closed $C$;
4. $\liminf_n P_n(G)\ge P(G)$ for all open $G$;
5. $P_n(A)\to P(A)$ for all Borel $A$ with $P(\partial A)=0$.

For $d=1$, each is equivalent to $F_n(x)\to F(x)$ at continuity points of $F$.

*Proof of (1) $\Rightarrow$ (3), which is the implication we use.* Let $C$ be closed and let $C^\delta=\{x:\operatorname{dist}(x,C)<\delta\}$. Define $f_\delta(x)=\max\{0,\,1-\operatorname{dist}(x,C)/\delta\}$, which is bounded, continuous (indeed $1/\delta$-Lipschitz), equal to $1$ on $C$ and $0$ off $C^\delta$. Then
$$P_n(C)\le\int f_\delta\,dP_n\longrightarrow\int f_\delta\,dP\le P(C^\delta).$$
Hence $\limsup_n P_n(C)\le P(C^\delta)$ for every $\delta>0$. Since $C$ is closed, $C^\delta\downarrow C$ as $\delta\downarrow0$, so continuity from above (Proposition 1.8(4), legitimate because $P$ is finite) gives $P(C^\delta)\to P(C)$. $\square$

The remaining equivalences are standard and are recorded in §7.10; the proofs are variations on the same smoothing trick. I will use (5) freely: *$P_n(A)\to P(A)$ whenever the boundary of $A$ is null for the limit*. That single sentence explains the continuity-point caveat in Definition 7.4, since for $A=(-\infty,x]$ we have $\partial A=\{x\}$ and $P(\{x\})=0$ exactly when $F$ is continuous at $x$.

> **Common Confusion 7.1.**
> *"Convergence in distribution must be about the random variables getting close — otherwise why call it convergence?"*
>
> It is not, and this is the single most common error in first courses. Let $Z\sim\mathcal N(0,1)$ and set $X_n := (-1)^n Z$. Every $X_n$ has exactly the standard normal law, because $-Z \overset{d}{=} Z$. So $F_n = \Phi$ for all $n$, and trivially $X_n\to_{\mathrm d}Z$. But $|X_n - Z|$ equals $0$ for even $n$ and $2|Z|$ for odd $n$; it does not converge in probability to anything, since $\mathbb{P}(|X_n-Z|>1)=\mathbb{P}(2|Z|>1)\approx 0.617$ along odd $n$. The sequence's *laws* converge perfectly while the sequence itself oscillates forever.
>
> The right mental model: convergence in distribution is convergence of *histograms*, not of *trajectories*. If someone hands you $X_n\to_{\mathrm d}X$ and you want to conclude anything about $X_n-X$, you need extra structure — most often that $X$ is a constant (Proposition 7.9) or Slutsky (Theorem 7.40).

### 7.1.2 The implications that hold

**Proposition 7.6 (a reformulation of almost sure convergence).** $X_n\to_{\mathrm{as}}X$ if and only if for every $\varepsilon>0$,
$$\lim_{N\to\infty}\ \mathbb{P}\Big(\sup_{n\ge N}|X_n-X|>\varepsilon\Big)=0,$$
if and only if $\mathbb{P}\big(|X_n - X|>\varepsilon \text{ for infinitely many } n\big)=0$ for every $\varepsilon>0$.

*Proof.* Write $A_n(\varepsilon)=\{|X_n-X|>\varepsilon\}$ and $B_N(\varepsilon)=\bigcup_{n\ge N}A_n(\varepsilon)=\{\sup_{n\ge N}|X_n-X|>\varepsilon\}$. The set where $X_n(\omega)\not\to X(\omega)$ is precisely $\bigcup_{k\ge1}\bigcap_{N\ge1}B_N(1/k)$: convergence fails iff some tolerance $1/k$ is exceeded at arbitrarily large times. By countable subadditivity (Proposition 1.8(2)), this union is null iff each $\bigcap_N B_N(1/k)$ is null. For fixed $\varepsilon$, $B_N(\varepsilon)$ decreases in $N$, so by continuity from above (Proposition 1.8(4)) $\mathbb{P}(B_N(\varepsilon))\downarrow\mathbb{P}(\bigcap_N B_N(\varepsilon))=\mathbb{P}(A_n(\varepsilon)\text{ i.o.})$. Both stated conditions therefore say the same thing: $\mathbb{P}(A_n(\varepsilon)\text{ i.o.})=0$ for all $\varepsilon>0$. $\square$

Compare this with Definition 7.2 and the whole distinction snaps into focus. Convergence in probability controls $\mathbb{P}(A_n(\varepsilon))$, one $n$ at a time. Almost sure convergence controls $\mathbb{P}(\bigcup_{n\ge N}A_n(\varepsilon))$, a *whole tail at once*. The union of a tail can be much larger than any single term — that gap is where every counterexample below lives.

**Proposition 7.7 (a.s. $\Rightarrow$ in probability).** If $X_n\to_{\mathrm{as}}X$ then $X_n\to_{\mathrm p}X$.

*Proof.* $\mathbb{P}(A_n(\varepsilon))\le\mathbb{P}(B_n(\varepsilon))\to0$ by monotonicity (Proposition 1.8(1)) and Proposition 7.6. $\square$

**Proposition 7.8 ($L^q\Rightarrow L^p\Rightarrow$ in probability).** Let $1\le p<q<\infty$.
(i) If $X_n\to_{L^q}X$ then $X_n\to_{L^p}X$. (ii) If $X_n\to_{L^p}X$ then $X_n\to_{\mathrm p}X$.

*Proof.* (i) By Jensen's inequality (Ch. 3) applied to the convex function $u\mapsto u^{q/p}$ on $[0,\infty)$ with $u=|X_n-X|^p$, we get $\big(\mathbb{E}|X_n-X|^p\big)^{q/p}\le\mathbb{E}|X_n-X|^q$, i.e. $\|X_n-X\|_p\le\|X_n-X\|_q$ (Lyapunov's inequality). (ii) By Markov's inequality (Ch. 4) applied to $|X_n-X|^p$,
$$\mathbb{P}(|X_n-X|>\varepsilon)=\mathbb{P}(|X_n-X|^p>\varepsilon^p)\le\frac{\mathbb{E}|X_n-X|^p}{\varepsilon^p}\to0. \qquad\square$$

The $p=2$ case of (ii) is worth naming because it is what you will actually use: $\mathbb{P}(|X_n-X|>\varepsilon)\le \mathbb{E}[(X_n-X)^2]/\varepsilon^2$, which is Chebyshev's inequality in disguise.

**Proposition 7.9 (in probability $\Rightarrow$ in distribution).** If $X_n\to_{\mathrm p}X$ then $X_n\to_{\mathrm d}X$.

*Proof.* Fix $x$ with $F$ continuous at $x$, and $\varepsilon>0$. Split on whether $X_n$ is close to $X$:
$$F_n(x)=\mathbb{P}(X_n\le x)\le \mathbb{P}(X\le x+\varepsilon)+\mathbb{P}(|X_n-X|>\varepsilon)=F(x+\varepsilon)+o(1).$$
(If $X_n\le x$ and $|X_n-X|\le\varepsilon$ then $X\le x+\varepsilon$.) Symmetrically,
$$F(x-\varepsilon)=\mathbb{P}(X\le x-\varepsilon)\le\mathbb{P}(X_n\le x)+\mathbb{P}(|X_n-X|>\varepsilon)=F_n(x)+o(1).$$
Therefore $F(x-\varepsilon)\le\liminf_n F_n(x)\le\limsup_n F_n(x)\le F(x+\varepsilon)$. Let $\varepsilon\downarrow0$ and use continuity of $F$ at $x$. $\square$

**Proposition 7.10 (converse when the limit is a constant).** If $X_n\to_{\mathrm d}c$ for a constant $c\in\mathbb{R}$, then $X_n\to_{\mathrm p}c$.

*Proof.* The limit CDF is $F(x)=\mathbf 1\{x\ge c\}$, continuous everywhere except at $c$. For $\varepsilon>0$,
$$\mathbb{P}(|X_n-c|>\varepsilon)\le \mathbb{P}(X_n\le c-\varepsilon)+\mathbb{P}(X_n> c+\varepsilon)= F_n(c-\varepsilon)+\big(1-F_n(c+\varepsilon)\big).$$
Both $c\pm\varepsilon$ are continuity points of $F$, so $F_n(c-\varepsilon)\to F(c-\varepsilon)=0$ and $F_n(c+\varepsilon)\to F(c+\varepsilon)=1$. $\square$

Proposition 7.10 is the workhorse of §7.7. Almost every application of Slutsky boils down to: *this nuisance quantity converges in distribution to a constant, hence in probability to that constant, hence it can be treated as if it were that constant in the limit.*

Assembled, the four true implications are
$$\boxed{\;L^q\ \Rightarrow\ L^p\ \Rightarrow\ \text{in probability}\ \Rightarrow\ \text{in distribution},\qquad \text{a.s.}\ \Rightarrow\ \text{in probability}. \;}$$
Nothing else holds without extra assumptions. Figure 7.1 records this, with the counterexamples attached to the arrows that are *missing*.

<svg viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="30" y="28" width="110" height="42" rx="6"/>
    <rect x="205" y="28" width="90" height="42" rx="6"/>
    <rect x="205" y="152" width="90" height="42" rx="6"/>
    <rect x="380" y="90" width="120" height="42" rx="6"/>
    <rect x="570" y="90" width="120" height="42" rx="6"/>
  </g>
  <g fill="currentColor" font-size="14" text-anchor="middle" font-family="serif">
    <text x="85" y="54">L^q  (q &gt; p)</text>
    <text x="250" y="54">L^p</text>
    <text x="250" y="178">almost sure</text>
    <text x="440" y="116">in probability</text>
    <text x="630" y="116">in distribution</text>
  </g>
  <g stroke="#10b981" stroke-width="2" fill="none" marker-end="url(#ah)">
    <path d="M142 49 L200 49"/>
    <path d="M297 60 L376 98"/>
    <path d="M297 168 L376 128"/>
    <path d="M502 111 L566 111"/>
  </g>
  <defs>
    <marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="#10b981"/>
    </marker>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="#ef4444"/>
    </marker>
    <marker id="ab" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="#3b82f6"/>
    </marker>
  </defs>
  <g fill="#10b981" font-size="11" text-anchor="middle" font-family="sans-serif">
    <text x="171" y="40">Jensen</text>
    <text x="352" y="70">Markov</text>
    <text x="352" y="163">Prop 7.7</text>
    <text x="534" y="103">Prop 7.9</text>
  </g>
  <g stroke="#ef4444" stroke-width="1.6" fill="none" stroke-dasharray="5 4" marker-end="url(#ar)">
    <path d="M566 126 L504 126"/>
    <path d="M376 138 L299 182"/>
    <path d="M376 84 L299 66"/>
    <path d="M243 148 L243 76"/>
    <path d="M257 76 L257 148"/>
  </g>
  <g fill="#ef4444" font-size="11" font-family="sans-serif">
    <text x="512" y="146" text-anchor="start">✗ WE 7.6: X_n = (−1)ⁿZ</text>
    <text x="306" y="150" text-anchor="start">✗ WE 7.1: sliding block</text>
    <text x="306" y="82" text-anchor="start">✗ WE 7.2: moving spike</text>
    <text x="120" y="120" text-anchor="start">✗ WE 7.2 (a.s. ⇏ L^1)</text>
    <text x="120" y="136" text-anchor="start">✗ WE 7.3 (L^1 ⇏ a.s.)</text>
  </g>
  <g stroke="#3b82f6" stroke-width="1.6" fill="none" stroke-dasharray="2 3" marker-end="url(#ab)">
    <path d="M600 136 L470 150 L448 136"/>
  </g>
  <text x="536" y="170" fill="#3b82f6" font-size="11" text-anchor="middle" font-family="sans-serif">only if the limit is a constant (Prop 7.10)</text>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="255">Figure 7.1. The four modes of convergence. Green solid arrows are the only implications that hold in general;</text>
    <text x="20" y="274">each is proved in §7.1.2. Red dashed arrows are FALSE, and are labelled with the worked example that kills them.</text>
    <text x="20" y="293">There is no arrow in either direction between "almost sure" and "L^p": a.s. convergence permits rare huge</text>
    <text x="20" y="312">excursions that blow up the mean, and L^1 convergence permits a small bad set that keeps moving. The blue</text>
    <text x="20" y="331">dotted arrow is the one partial converse: convergence in distribution to a CONSTANT upgrades to convergence</text>
    <text x="20" y="350">in probability, which is what makes Slutsky's theorem (§7.7) work.</text>
  </g>
</svg>

### 7.1.3 The counterexamples

All of these live on $\Omega=[0,1]$ with $\mathcal{F}=\mathcal{B}([0,1])$ and $\mathbb{P}=\lambda$, Lebesgue measure. Think of $\omega$ as a uniform random seed, exactly as Chapter 1 taught.

**Worked Example 7.1 (in probability but not almost surely: the sliding block).** For $n\ge1$ write $n=2^k+j$ with $k\ge0$ and $0\le j<2^k$ (this is unique: $k=\lfloor\log_2 n\rfloor$, $j=n-2^k$). Define
$$X_n(\omega) := \mathbf 1_{I_{k,j}}(\omega),\qquad I_{k,j}:=\big[\,j2^{-k},\,(j+1)2^{-k}\,\big).$$
So $X_1$ is the indicator of $[0,1)$; $X_2,X_3$ indicate the two halves; $X_4,\dots,X_7$ the four quarters; and so on. Then:

*Convergence in probability to $0$.* For $0<\varepsilon<1$, $\mathbb{P}(|X_n|>\varepsilon)=\lambda(I_{k,j})=2^{-k}\le 2/n$, which $\to0$. So $X_n\to_{\mathrm p}0$, and indeed $X_n\to_{L^p}0$ for every $p$, since $\mathbb{E}|X_n|^p=2^{-k}\le2/n$.

*No almost sure convergence, anywhere.* Fix any $\omega\in[0,1)$. For each $k$ there is exactly one $j=j_k(\omega)=\lfloor 2^k\omega\rfloor$ with $\omega\in I_{k,j}$, so $X_{2^k+j_k(\omega)}(\omega)=1$. There are infinitely many such $n$ (one per $k$), and infinitely many $n$ with $X_n(\omega)=0$ (as soon as $2^k\ge2$ there are other blocks). Hence $\limsup_n X_n(\omega)=1$ and $\liminf_n X_n(\omega)=0$ for *every* $\omega\in[0,1)$: the set of convergence is empty, not merely null.

Concretely, at $\omega=0.3$: $\lfloor 2\cdot0.3\rfloor=0$ so $n=2^1+0=2$; $\lfloor4\cdot0.3\rfloor=1$ so $n=5$; $\lfloor8\cdot0.3\rfloor=2$ so $n=10$; $\lfloor16\cdot0.3\rfloor=4$ so $n=20$; $\lfloor32\cdot0.3\rfloor=9$ so $n=41$. The sequence $X_n(0.3)$ equals $1$ at $n=1,2,5,10,20,41,\dots$ and $0$ elsewhere. The spikes get rarer, which is why the *probability* of a spike at a given time goes to zero, but they never stop, which is why the trajectory never settles.

<svg viewBox="0 0 700 330" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g stroke="currentColor" stroke-width="1" fill="none">
    <line x1="40" y1="130" x2="360" y2="130"/>
  </g>
  <g font-size="11" fill="currentColor" font-family="sans-serif">
    <text x="40" y="146" text-anchor="middle">0</text>
    <text x="360" y="146" text-anchor="middle">1</text>
    <text x="136" y="146" text-anchor="middle">0.3</text>
    <text x="200" y="20" text-anchor="middle" font-size="12">support of X_n on Ω = [0,1]</text>
    <text x="24" y="38" text-anchor="end">n=1</text>
    <text x="24" y="58" text-anchor="end">n=2,3</text>
    <text x="24" y="78" text-anchor="end">n=4..7</text>
    <text x="24" y="98" text-anchor="end">n=8..15</text>
    <text x="24" y="118" text-anchor="end">n=16..31</text>
  </g>
  <g stroke="currentColor" stroke-width="1" fill="none">
    <rect x="40" y="30" width="320" height="10"/>
    <rect x="40" y="50" width="160" height="10"/><rect x="200" y="50" width="160" height="10"/>
    <rect x="40" y="70" width="80" height="10"/><rect x="120" y="70" width="80" height="10"/><rect x="200" y="70" width="80" height="10"/><rect x="280" y="70" width="80" height="10"/>
    <rect x="40" y="90" width="40" height="10"/><rect x="80" y="90" width="40" height="10"/><rect x="120" y="90" width="40" height="10"/><rect x="160" y="90" width="40" height="10"/><rect x="200" y="90" width="40" height="10"/><rect x="240" y="90" width="40" height="10"/><rect x="280" y="90" width="40" height="10"/><rect x="320" y="90" width="40" height="10"/>
  </g>
  <g fill="#3b82f6" opacity="0.85">
    <rect x="40" y="30" width="320" height="10"/><rect x="40" y="50" width="160" height="10"/>
    <rect x="120" y="70" width="80" height="10"/><rect x="120" y="90" width="40" height="10"/>
    <rect x="120" y="110" width="20" height="10"/>
  </g>
  <g stroke="currentColor" stroke-width="1" fill="none"><rect x="40" y="110" width="320" height="10"/></g>
  <line x1="136" y1="26" x2="136" y2="134" stroke="#ef4444" stroke-width="1.4" stroke-dasharray="4 3"/>
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="420" y1="120" x2="680" y2="120"/><line x1="420" y1="120" x2="420" y2="35"/>
  </g>
  <g stroke="#ef4444" stroke-width="2.5">
    <line x1="425" y1="120" x2="425" y2="45"/><line x1="430" y1="120" x2="430" y2="45"/>
    <line x1="445" y1="120" x2="445" y2="45"/><line x1="470" y1="120" x2="470" y2="45"/>
    <line x1="520" y1="120" x2="520" y2="45"/><line x1="625" y1="120" x2="625" y2="45"/>
  </g>
  <g font-size="11" fill="currentColor" font-family="sans-serif">
    <text x="414" y="49" text-anchor="end">1</text><text x="414" y="124" text-anchor="end">0</text>
    <text x="550" y="140" text-anchor="middle">n  (1, 2, 5, 10, 20, 41, ...)</text>
    <text x="550" y="26" text-anchor="middle" font-size="12">the trajectory X_n(0.3)</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="180">Figure 7.2. The sliding block of Worked Example 7.1. Left: at each dyadic level the block sweeps across [0,1];</text>
    <text x="20" y="199">blue marks the block that happens to contain ω = 0.3. Right: the resulting trajectory. The probability of a spike</text>
    <text x="20" y="218">at time n is 2^(−⌊log2 n⌋) ≤ 2/n → 0, so X_n → 0 in probability and in every L^p. But EVERY ω is hit once per</text>
    <text x="20" y="237">dyadic level, so X_n(ω) has limsup 1 and liminf 0 for every ω: the set of convergence is empty. Convergence in</text>
    <text x="20" y="256">probability constrains one time at a time; almost sure convergence constrains an entire tail at once, and the</text>
    <text x="20" y="275">union of the tail events here has probability 1 for every N.</text>
  </g>
</svg>

**Worked Example 7.2 (a.s. and in probability but not in $L^1$: the moving spike).** On the same space, let
$$Y_n(\omega):= n\,\mathbf 1_{[0,1/n)}(\omega).$$
For every $\omega>0$ we have $Y_n(\omega)=0$ once $n>1/\omega$, so $Y_n\to0$ pointwise on $(0,1]$, i.e. almost surely (the exceptional set $\{0\}$ is $\lambda$-null). By Proposition 7.7, $Y_n\to_{\mathrm p}0$ as well. But
$$\mathbb{E}[|Y_n|]=n\cdot\tfrac1n=1\quad\text{for every }n,$$
so $Y_n\not\to_{L^1}0$. Raising the spike to $n^{1/p}$ instead breaks $L^p$ for any chosen $p$; raising it to $n^2$ breaks all of them at once with $\mathbb{E}|Y_n|=n\to\infty$.

This example is the reason the dominated convergence theorem (Ch. 2) has a hypothesis. Here $\sup_n Y_n(\omega)=\lfloor 1/\omega\rfloor$-ish, which is not integrable near $0$: $\int_0^1 (1/\omega)\,d\omega=\infty$. No dominating function, no interchange. Every time you see someone swap $\lim$ and $\mathbb{E}$ without comment, this is the picture of what could go wrong: probability mass escaping to infinity in height while its width shrinks.

**Worked Example 7.3 ($L^1$ and in probability but not a.s.).** Reuse the sliding block $X_n$ of Worked Example 7.1: $\mathbb{E}|X_n|=2^{-\lfloor\log_2 n\rfloor}\to0$, so $X_n\to_{L^1}0$, while $X_n(\omega)\not\to0$ for any $\omega$. Combined with Worked Example 7.2, this establishes that neither of "a.s." and "$L^1$" implies the other — the two vertical red marks in Figure 7.1.

**Worked Example 7.4 (a.s. but not in $L^1$, with a probabilistic flavour).** Let $U\sim\mathrm{Unif}(0,1)$ and set $W_n:=e^{n}\mathbf 1\{U<e^{-n}\}$. Then $\mathbb{P}(W_n\ne0)=e^{-n}$ is summable, so by Borel–Cantelli I (Theorem 7.11, next section) $W_n=0$ eventually, almost surely, hence $W_n\to_{\mathrm{as}}0$. Yet $\mathbb{E}[W_n]=1$ for all $n$. This is the same phenomenon as Worked Example 7.2 in probabilistic clothing, and it is exactly the failure mode of importance sampling with a badly matched proposal: the estimator is almost always $0$ and very occasionally astronomically large, so it converges a.s. to the wrong thing while its mean stays put.

**Worked Example 7.5 (in distribution but not in probability, discrete version).** Recall Chapter 1's running example (i): three fair coin flips, $\Omega=\{H,T\}^3$, $X=$ number of heads. Let $X_n:=X$ for odd $n$ and $X_n:=3-X$ for even $n$. Since $X\sim\mathrm{Bin}(3,1/2)$ is symmetric about $3/2$, $3-X\overset{d}{=}X$, so $F_n=F_X$ for every $n$ and $X_n\to_{\mathrm d}X$. But $|X_n-X|=|3-2X|\in\{1,3\}$ along even $n$, so $\mathbb{P}(|X_n-X|\ge1)=1$ for even $n$: no convergence in probability. Note that both $X_n$ and $X$ are defined on the same space here, so this is not a technicality about spaces.

**Worked Example 7.6 (in distribution but not in probability, continuous version).** The Gaussian version from Common Confusion 7.1: $Z\sim\mathcal N(0,1)$, $X_n=(-1)^nZ$. Then $X_n\to_{\mathrm d}\mathcal N(0,1)$, since every $X_n$ is exactly standard normal. Along *odd* $n$ we have $X_n-Z=-2Z$, so $\mathbb{P}(|X_n-Z|>1)=\mathbb{P}(|Z|>1/2)=2\Phi(-0.5)=0.6171$ (four decimals); along even $n$ the difference is identically $0$. The sequence $\mathbb{P}(|X_n-Z|>1)$ therefore oscillates between $0$ and $0.6171$ and does not converge to $0$.

**Worked Example 7.7 (convergence in distribution to the *wrong* variable).** Let $U\sim\mathrm{Unif}(0,1)$ and $X_n:=U$ for all $n$, $X:=1-U$. Then $X\overset{d}{=}U$, so $X_n\to_{\mathrm d}X$ — the sequence is constant and converges in distribution to a variable it is never close to ($|X_n-X|=|2U-1|$, with mean $1/2$). This is worth internalizing: "$X_n\to_{\mathrm d}X$" is a statement about $P_X$ only. Two different variables with the same law are interchangeable as limits in distribution. That is why one writes $X_n\to_{\mathrm d}\mathcal N(0,1)$, naming the law.

> **Common Confusion 7.2.**
> *"If $\mathbb{P}(|X_n - X|>\varepsilon)\to 0$ then surely $X_n(\omega)\to X(\omega)$ for almost every $\omega$ — the bad set is shrinking!"*
>
> The bad set is shrinking, but it is also *moving*. In Worked Example 7.1 the bad set at time $n$ has measure $\le 2/n$, yet its union over $n\ge N$ is all of $[0,1)$ for every $N$. Almost sure convergence needs $\mathbb{P}(\bigcup_{n\ge N}A_n(\varepsilon))\to0$; convergence in probability only gives $\mathbb{P}(A_n(\varepsilon))\to0$. The union bound (Proposition 1.8(2)) closes the gap only when $\sum_n \mathbb{P}(A_n(\varepsilon))<\infty$ — which is exactly Borel–Cantelli I, and exactly what fails here since $\sum_n 2^{-\lfloor\log_2 n\rfloor}=\sum_k 2^k\cdot2^{-k}=\infty$.
>
> The honest summary: *convergence in probability plus a summable rate equals almost sure convergence.* Rates are not decoration; they are the whole content.

### 7.1.4 Two remarks you will need later

**Convergence in distribution is not preserved by addition.** If $X_n\to_{\mathrm d}X$ and $Y_n\to_{\mathrm d}Y$, it is *false* in general that $X_n+Y_n\to_{\mathrm d}X+Y$. Take $Z\sim\mathcal N(0,1)$, $X_n:=Z$ and $Y_n:=(-1)^nZ$. Then $X_n\to_{\mathrm d}\mathcal N(0,1)$ and $Y_n\to_{\mathrm d}\mathcal N(0,1)$, but $X_n+Y_n$ equals $2Z\sim\mathcal N(0,4)$ for even $n$ and $0$ for odd $n$; the laws oscillate between $\mathcal N(0,4)$ and $\delta_0$ and do not converge. The reason is that convergence in distribution knows nothing about the *joint* law of $(X_n,Y_n)$, and $X+Y$ depends on it. Slutsky's theorem (§7.7) is precisely the repair: it holds when one of the two limits is a constant, because then the joint law is asymptotically determined.

**What repairs the missing arrow into $L^1$.** Figure 7.1 shows no arrow from "in probability" to $L^1$, and Worked Example 7.2 is why: mass can escape to infinity in height. There is a clean necessary-and-sufficient repair, worth knowing by name even though we relegate its proof to the exercises. Call a family $\{X_n\}$ *uniformly integrable* (UI) if $\lim_{M\to\infty}\sup_n\mathbb{E}[|X_n|\mathbf 1\{|X_n|>M\}]=0$ — no member of the family hides a non-vanishing amount of expectation above any finite level, uniformly in $n$. The Vitali convergence theorem then says: *if $X_n\to_{\mathrm p}X$, then $X_n\to_{L^1}X$ if and only if $\{X_n\}$ is uniformly integrable.* Boundedness ($|X_n|\le M$ a.s.) is the crudest sufficient condition (Exercise A3); domination by an integrable variable is the next crudest, which is how this connects to the dominated convergence theorem (Ch. 2); Exercise A8 proves the delicate half you will actually use, that UI plus convergence in distribution transfers expectations. The moving spike fails UI in the most literal way possible: $\mathbb{E}[|Y_n|\mathbf 1\{|Y_n|>M\}]=1$ for every $n>M$.

**Uniqueness of limits.** If $X_n\to_{\mathrm p}X$ and $X_n\to_{\mathrm p}X'$ then $X=X'$ a.s.: for any $\varepsilon>0$, $\mathbb{P}(|X-X'|>2\varepsilon)\le\mathbb{P}(|X-X_n|>\varepsilon)+\mathbb{P}(|X_n-X'|>\varepsilon)\to0$, so $\mathbb{P}(|X-X'|>2\varepsilon)=0$ for all $\varepsilon$; take $\varepsilon=1/k$ and use countable subadditivity. Distributional limits are unique only as *laws* (Worked Example 7.7), which is exactly what Theorem 1.20 guarantees: the CDF determines the law, so if $F_n\to F$ and $F_n\to G$ at continuity points then $F=G$ off a countable set, hence everywhere by right-continuity, hence $P=Q$.

> **Self-check 7.1.**
> 1. Give a sequence with $X_n\to_{\mathrm{as}}0$ but $\mathbb{E}[X_n^2]\to\infty$.
> 2. True or false: if $X_n\to_{\mathrm d}X$ and $X$ is a constant, then $X_n\to_{L^1}X$.
> 3. In Worked Example 7.1, compute $\mathbb{P}(\sup_{n\ge 100}|X_n|>1/2)$.
> 4. If $X_n\to_{\mathrm p}X$ and $Y_n\to_{\mathrm p}Y$ on the same space, does $X_n+Y_n\to_{\mathrm p}X+Y$? Why does the counterexample of §7.1.4 not apply?

---

## 7.2 Borel–Cantelli and almost sure convergence

### 7.2.1 The two lemmas

Given events $A_1,A_2,\dots$ in $\mathcal{F}$, write
$$\{A_n \text{ i.o.}\}:=\limsup_n A_n=\bigcap_{N\ge1}\bigcup_{n\ge N}A_n = \{\omega: \omega\in A_n\text{ for infinitely many }n\}.$$
Read it as: *no matter how long you wait, another $A_n$ happens.* Its complement, $\liminf_n A_n^c=\bigcup_N\bigcap_{n\ge N}A_n^c$, is "$A_n$ fails eventually, forever". Proposition 7.6 says almost sure convergence is exactly the statement that $\{|X_n-X|>\varepsilon\}$ fails eventually, for every $\varepsilon$. So the whole subject of a.s. convergence reduces to computing $\mathbb{P}(A_n\text{ i.o.})$, and Borel–Cantelli is the pair of tools for that.

**Theorem 7.11 (Borel–Cantelli I).** If $\sum_{n\ge1}\mathbb{P}(A_n)<\infty$ then $\mathbb{P}(A_n\text{ i.o.})=0$.

*Proof (you proved this as Chapter 1's Exercise A9; here for completeness).* Let $B_N=\bigcup_{n\ge N}A_n$. By countable subadditivity, the union bound of Proposition 1.8(2), $\mathbb{P}(B_N)\le\sum_{n\ge N}\mathbb{P}(A_n)$, which is the tail of a convergent series and hence $\to0$. Since $B_N\downarrow\{A_n\text{ i.o.}\}$, continuity from above (Proposition 1.8(4), valid because $\mathbb{P}$ is finite) gives $\mathbb{P}(A_n\text{ i.o.})=\lim_N\mathbb{P}(B_N)=0$. $\square$

No independence was used. That is what makes BC-I so widely applicable: any time you can bound $\mathbb{P}(A_n)$ by something summable, you get an almost-sure conclusion for free, whatever the dependence structure. Its partner needs independence and gives the reverse conclusion.

**Theorem 7.12 (Borel–Cantelli II).** If $A_1,A_2,\dots$ are independent (Definition 1.23) and $\sum_{n\ge1}\mathbb{P}(A_n)=\infty$, then $\mathbb{P}(A_n\text{ i.o.})=1$.

*Proof.* It suffices to show $\mathbb{P}(\bigcup_{n\ge N}A_n)=1$ for every $N$, since then $\mathbb{P}(A_n\text{ i.o.})=\lim_N \mathbb{P}(B_N)=1$ by continuity from above. Equivalently, $\mathbb{P}(\bigcap_{n\ge N}A_n^c)=0$. For $M>N$, independence of the $A_n^c$ (which follows from independence of the $A_n$: complements of independent events are independent) gives
$$\mathbb{P}\Big(\bigcap_{n=N}^{M}A_n^c\Big)=\prod_{n=N}^{M}\big(1-\mathbb{P}(A_n)\big)\le\prod_{n=N}^{M}e^{-\mathbb{P}(A_n)}=\exp\Big(-\sum_{n=N}^{M}\mathbb{P}(A_n)\Big),$$
using the elementary inequality $1-u\le e^{-u}$, valid for all real $u$ (convexity of $e^{-u}$: its graph lies above the tangent at $u=0$). Since $\sum_n\mathbb{P}(A_n)=\infty$, the tail sum $\sum_{n=N}^{M}\mathbb{P}(A_n)\to\infty$ as $M\to\infty$, so the bound $\to0$. Finally $\bigcap_{n=N}^{M}A_n^c\downarrow\bigcap_{n\ge N}A_n^c$ as $M\to\infty$, so continuity from above gives $\mathbb{P}(\bigcap_{n\ge N}A_n^c)=0$. $\square$

**Corollary 7.13 (a zero–one dichotomy).** For independent events, $\mathbb{P}(A_n\text{ i.o.})\in\{0,1\}$, and which one it is depends only on whether $\sum_n\mathbb{P}(A_n)$ converges. There is no middle ground.

This is a genuinely surprising rigidity, and it is the simplest instance of the zero–one laws (Kolmogorov's, Hewitt–Savage) that pervade probability. Independence forbids "the event happens infinitely often with probability $1/2$".

**Worked Example 7.8 (runs of heads).** Flip a fair coin forever; let $A_n$ be the event that flips $n, n+1,\dots,n+k-1$ are all heads, for a fixed $k$. Then $\mathbb{P}(A_n)=2^{-k}$ and $\sum_n\mathbb{P}(A_n)=\infty$, but the $A_n$ overlap and are *not* independent, so Theorem 7.12 does not apply directly. Repair it by taking a sparse subsequence: let $C_m:=A_{(m-1)k+1}$, the event that block $m$ of $k$ consecutive flips is all heads. The $C_m$ *are* independent (disjoint blocks of independent flips) and $\sum_m\mathbb{P}(C_m)=\sum_m 2^{-k}=\infty$. By BC-II, $\mathbb{P}(C_m\text{ i.o.})=1$, and $\{C_m\text{ i.o.}\}\subseteq\{A_n\text{ i.o.}\}$. Conclusion: for every $k$, a run of $k$ heads occurs infinitely often, almost surely. Intersecting over $k\in\mathbb{N}$ (a countable intersection of probability-one events, Proposition 1.8(2) applied to the complements) gives: almost surely, *every* finite pattern occurs infinitely often. Long stretches of "impossible-looking" data are certain, given enough data. This is the correct rebuttal to "my model can't be right, look at this streak."

**Worked Example 7.9 (BC-I is not tight without independence).** Let $U\sim\mathrm{Unif}(0,1)$ and $A_n:=\{U<1/n\}$. Then $\sum_n\mathbb{P}(A_n)=\sum 1/n=\infty$, yet $\{A_n\text{ i.o.}\}=\{U<1/n \ \forall\text{ infinitely many }n\}=\{U=0\}$, which has probability $0$. The $A_n$ are nested, hence maximally dependent, and BC-II's independence hypothesis is essential, not cosmetic.

### 7.2.2 Upgrading in probability to almost sure

**Proposition 7.14 (summable tails criterion).** If for every $\varepsilon>0$,
$$\sum_{n\ge1}\mathbb{P}(|X_n-X|>\varepsilon)<\infty,$$
then $X_n\to_{\mathrm{as}}X$. It suffices to check this for $\varepsilon=1/k$, $k\in\mathbb{N}$.

*Proof.* Fix $\varepsilon>0$ and apply Theorem 7.11 to $A_n=\{|X_n-X|>\varepsilon\}$: $\mathbb{P}(A_n\text{ i.o.})=0$. By Proposition 7.6 this for all $\varepsilon$ is equivalent to $X_n\to_{\mathrm{as}}X$. For the last sentence, if $\mathbb{P}(A_n(1/k)\text{ i.o.})=0$ for each $k$ then, given arbitrary $\varepsilon>0$, choose $k>1/\varepsilon$; since $A_n(\varepsilon)\subseteq A_n(1/k)$, monotonicity finishes it. $\square$

This is the single most-used tool in the "almost sure convergence of stochastic approximation" literature. The template: prove a *rate* (a bound on $\mathbb{P}(|X_n-X|>\varepsilon)$ that decays fast enough to be summable in $n$), then invoke Proposition 7.14. Chapter 4's Hoeffding bound often supplies the rate directly.

**Worked Example 7.10 (Hoeffding + BC-I gives a.s. convergence of averages).** Let $X_1,X_2,\dots$ be i.i.d. with $X_i\in[0,1]$ and mean $\mu$. Hoeffding's inequality (Ch. 4) gives $\mathbb{P}(|\bar X_n-\mu|>\varepsilon)\le 2e^{-2n\varepsilon^2}$. For fixed $\varepsilon$ this is summable in $n$ (geometric), so Proposition 7.14 yields $\bar X_n\to_{\mathrm{as}}\mu$ — a strong law, for free, for bounded variables. Compare the sheer effort of the general SLLN (Theorem 7.19). Boundedness buys a lot.

Concretely: with $\varepsilon=0.01$, $\sum_{n\ge1}2e^{-2n(0.01)^2}=2\,e^{-0.0002}/(1-e^{-0.0002})=9999.0$ (five significant figures). Finite, though only barely — which correctly signals that $n$ must reach the tens of thousands before $\bar X_n$ is reliably within $0.01$.

**Theorem 7.15 (subsequence principle).** $X_n\to_{\mathrm p}X$ if and only if every subsequence $(X_{n_k})$ has a further subsequence $(X_{n_{k_j}})$ with $X_{n_{k_j}}\to_{\mathrm{as}}X$.

*Proof.* ($\Rightarrow$) Suppose $X_n\to_{\mathrm p}X$ and let $(X_{n_k})$ be a subsequence; it also converges in probability to $X$. Choose indices $n_{k_1}<n_{k_2}<\cdots$ such that $\mathbb{P}(|X_{n_{k_j}}-X|>2^{-j})\le 2^{-j}$; this is possible because for each $j$ the probability $\mathbb{P}(|X_{n_k}-X|>2^{-j})\to0$ as $k\to\infty$. Now fix $\varepsilon>0$ and take $J$ with $2^{-J}<\varepsilon$; for $j\ge J$, $\{|X_{n_{k_j}}-X|>\varepsilon\}\subseteq\{|X_{n_{k_j}}-X|>2^{-j}\}$, so $\sum_j\mathbb{P}(|X_{n_{k_j}}-X|>\varepsilon)\le J+\sum_{j\ge J}2^{-j}<\infty$. Proposition 7.14 gives a.s. convergence along that sub-subsequence.

($\Leftarrow$) Suppose $X_n\not\to_{\mathrm p}X$. Then there exist $\varepsilon,\delta>0$ and a subsequence with $\mathbb{P}(|X_{n_k}-X|>\varepsilon)\ge\delta$ for all $k$. No further subsequence of it can converge a.s. to $X$, because a.s. convergence implies convergence in probability (Proposition 7.7), which would force those probabilities to $0$. $\square$

Theorem 7.15 is more useful than it looks. It lets you transfer any property that is preserved under a.s. limits to convergence in probability, by the "every subsequence has a further subsequence" trick. Two immediate consequences: if $X_n\to_{\mathrm p}X$ and $g$ is continuous then $g(X_n)\to_{\mathrm p}g(X)$ (apply $g$ along the a.s. sub-subsequence); and if $X_n\to_{\mathrm p}X$, $Y_n\to_{\mathrm p}Y$ then $X_nY_n\to_{\mathrm p}XY$. We revisit this as Theorem 7.39.

> **Common Confusion 7.3.**
> *"BC-II says a divergent sum forces infinitely-often. So if my SGD iterates have $\sum_n \mathbb{P}(\|\theta_n-\theta^\star\|>\varepsilon)=\infty$, they must fail infinitely often."*
>
> Only under independence. SGD iterates are the opposite of independent — $\theta_{n+1}$ is a deterministic function of $\theta_n$ and the fresh minibatch. Worked Example 7.9 shows how badly the conclusion can fail without it: nested events have divergent probability sums and probability-zero limsup. When you need a converse to BC-I under dependence, the standard substitutes are the *conditional* Borel–Cantelli lemma (Lévy's extension, which uses the martingale machinery of Ch. 6 and replaces $\sum\mathbb{P}(A_n)$ by $\sum\mathbb{P}(A_n\mid\mathcal{F}_{n-1})$) or a direct martingale argument. Reaching for BC-II on dependent data is a genuine error, not a technicality.

> **Self-check 7.2.**
> 1. Let $A_n$ be independent with $\mathbb{P}(A_n)=1/n$. What is $\mathbb{P}(A_n\text{ i.o.})$? What if $\mathbb{P}(A_n)=1/n^2$?
> 2. Explain in one sentence why BC-I needs no independence but BC-II does.
> 3. If $\mathbb{P}(|X_n|>1/n)\le 1/n^2$, does $X_n\to_{\mathrm{as}}0$?
> 4. Give a sequence converging in probability but with $\sum_n\mathbb{P}(|X_n|>\varepsilon)=\infty$ for some $\varepsilon$, that nevertheless converges almost surely.

---

## 7.3 Laws of large numbers

### 7.3.1 The weak law

Everyone's intuition about averages is the law of large numbers. The mathematical content is that the intuition is correct under an assumption so mild it is nearly always satisfied — and catastrophically wrong when it is not.

Throughout §7.3, $X_1,X_2,\dots$ are i.i.d. with common law $P$, and $S_n:=\sum_{i=1}^n X_i$, $\bar X_n:=S_n/n$.

**Theorem 7.16 (WLLN under a second moment).** If $\mathbb{E}[X_1^2]<\infty$ with $\mu=\mathbb{E}[X_1]$ and $\sigma^2=\operatorname{Var}(X_1)$, then $\bar X_n\to_{\mathrm p}\mu$; quantitatively, for every $\varepsilon>0$,
$$\mathbb{P}(|\bar X_n-\mu|>\varepsilon)\le\frac{\sigma^2}{n\varepsilon^2}.$$

*Proof.* Independence gives $\operatorname{Var}(S_n)=n\sigma^2$ (covariances vanish), so $\operatorname{Var}(\bar X_n)=\sigma^2/n$. Apply Chebyshev's inequality (Ch. 4) to $\bar X_n$. $\square$

Notice what the proof used: *uncorrelatedness*, not independence. Theorem 7.16 holds verbatim for pairwise uncorrelated variables with common mean and bounded variance, a fact worth remembering because it is often exactly what a dependent-data situation gives you.

The $L^2$ assumption is convenient but not necessary. Kolmogorov's version needs only $\mathbb{E}|X_1|<\infty$, and the standard route is *truncation*: chop off the part of $X_i$ that is too large to control, apply the $L^2$ argument to the bounded part, and show the discarded part is negligible.

**Theorem 7.17 (WLLN under a first moment; Khinchin).** If $\mathbb{E}|X_1|<\infty$ and $\mu=\mathbb{E}[X_1]$, then $\bar X_n\to_{\mathrm p}\mu$.

*Proof.* Fix $\varepsilon>0$. For a truncation level $b_n>0$ to be chosen, set
$$Y_i^{(n)}:=X_i\mathbf 1\{|X_i|\le b_n\},\qquad Z_i^{(n)}:=X_i-Y_i^{(n)}=X_i\mathbf 1\{|X_i|>b_n\}.$$
Write $T_n=\sum_{i\le n}Y_i^{(n)}$ and $R_n=\sum_{i\le n}Z_i^{(n)}$, so $S_n=T_n+R_n$.

*Step 1: the tail part is small in $L^1$.* Let $\mu_n:=\mathbb{E}[Y_1^{(n)}]$. Then
$$\mathbb{E}\Big|\tfrac{R_n}{n}\Big|\le\frac1n\sum_{i\le n}\mathbb{E}|Z_i^{(n)}|=\mathbb{E}\big[|X_1|\mathbf 1\{|X_1|>b_n\}\big]=:r(b_n).$$
Because $\mathbb{E}|X_1|<\infty$, the dominated convergence theorem (Ch. 2) with dominating function $|X_1|$ gives $r(b)\to0$ as $b\to\infty$. Also $|\mu-\mu_n|=|\mathbb{E}[X_1\mathbf 1\{|X_1|>b_n\}]|\le r(b_n)\to0$.

*Step 2: the truncated part obeys Chebyshev.* $|Y_1^{(n)}|\le b_n$, so
$$\operatorname{Var}(Y_1^{(n)})\le\mathbb{E}\big[(Y_1^{(n)})^2\big]\le b_n\,\mathbb{E}\big|Y_1^{(n)}\big|\le b_n\,\mathbb{E}|X_1|,$$
whence $\operatorname{Var}(T_n/n)\le b_n\mathbb{E}|X_1|/n$ and
$$\mathbb{P}\big(|T_n/n-\mu_n|>\varepsilon/2\big)\le\frac{4\,b_n\,\mathbb{E}|X_1|}{n\varepsilon^2}.$$

*Step 3: choose $b_n$ and combine.* Take $b_n=\sqrt n$. Then Step 2's bound is $4\mathbb{E}|X_1|/(\sqrt n\varepsilon^2)\to0$, and $r(b_n)\to0$ since $b_n\to\infty$. By Markov's inequality, $\mathbb{P}(|R_n/n|>\varepsilon/4)\le 4r(b_n)/\varepsilon\to0$. Finally, for $n$ large enough that $|\mu-\mu_n|<\varepsilon/4$,
$$\{|\bar X_n-\mu|>\varepsilon\}\subseteq\{|T_n/n-\mu_n|>\varepsilon/2\}\cup\{|R_n/n|>\varepsilon/4\},$$
because $|\bar X_n-\mu|\le|T_n/n-\mu_n|+|\mu_n-\mu|+|R_n/n|$. The union bound (Proposition 1.8(2)) and Steps 1–2 finish the proof. $\square$

The truncation trick — and its accounting, "control the bulk in $L^2$, control the tail in $L^1$" — recurs constantly. It is the same idea as gradient clipping: replace an unbounded quantity by a bounded proxy, prove things about the proxy, and separately bound the damage done by clipping.

### 7.3.2 The strong law

**Theorem 7.18 (SLLN under a fourth moment).** If $\mathbb{E}[X_1^4]<\infty$, then $\bar X_n\to_{\mathrm{as}}\mu$.

*Proof.* Replacing $X_i$ by $X_i-\mu$, assume $\mu=0$. Expand
$$\mathbb{E}[S_n^4]=\sum_{i,j,k,\ell\le n}\mathbb{E}[X_iX_jX_kX_\ell].$$
By independence and $\mathbb{E}[X_i]=0$, a term vanishes unless every index appearing does so at least twice. The surviving patterns are: all four indices equal ($n$ terms, each $\mathbb{E}[X_1^4]$), and two distinct indices each appearing twice. For the latter, choose the unordered pair $\{i,j\}$ in $\binom n2$ ways and the arrangement in $\binom42=6$ ways, giving $6\binom n2 = 3n(n-1)$ terms, each equal to $\mathbb{E}[X_1^2]^2$. Hence
$$\mathbb{E}[S_n^4]=n\,\mathbb{E}[X_1^4]+3n(n-1)\big(\mathbb{E}[X_1^2]\big)^2\le Cn^2,\qquad C:=\mathbb{E}[X_1^4]+3\big(\mathbb{E}[X_1^2]\big)^2,$$
using $\mathbb{E}[X_1^2]^2\le\mathbb{E}[X_1^4]$ (Jensen, Ch. 3) only to see $C<\infty$. By Markov's inequality applied to $S_n^4$,
$$\mathbb{P}\big(|\bar X_n|>\varepsilon\big)=\mathbb{P}\big(S_n^4>n^4\varepsilon^4\big)\le\frac{Cn^2}{n^4\varepsilon^4}=\frac{C}{n^2\varepsilon^4}.$$
This is summable in $n$, so Proposition 7.14 gives $\bar X_n\to_{\mathrm{as}}0$. $\square$

The fourth moment is doing exactly one job: turning the $1/n$ of Chebyshev into the $1/n^2$ that Borel–Cantelli needs. That is the whole idea, and it is worth seeing once, because the general theorem's proof is a much longer route to the same destination.

**Theorem 7.19 (Kolmogorov's SLLN).** Let $X_1,X_2,\dots$ be i.i.d. Then $\bar X_n$ converges almost surely to a finite limit if and only if $\mathbb{E}|X_1|<\infty$, in which case the limit is $\mathbb{E}[X_1]$.

The "if" direction is proved by truncation plus Kolmogorov's maximal inequality and the Kolmogorov three-series theorem, or by Etemadi's elegant argument (which needs only pairwise independence). It is genuinely more work than anything else in this chapter, and it would double the length of §7.3 to reproduce; it is recorded in §7.10 with references. The "only if" direction we can prove right now, and it is more instructive than it looks.

**Proposition 7.20 (infinite mean destroys the average).** If $X_1,X_2,\dots$ are i.i.d. with $\mathbb{E}|X_1|=\infty$, then $\limsup_n|S_n/n|=\infty$ almost surely. In particular $\bar X_n$ does not converge to any finite limit, a.s. or in probability.

*Proof.* By the tail formula (Ch. 2), $\mathbb{E}|X_1|=\int_0^\infty\mathbb{P}(|X_1|>t)\,dt$, and comparing the integral with the sum over unit intervals,
$$\sum_{n\ge1}\mathbb{P}(|X_1|>n)\ \ge\ \int_1^\infty\mathbb{P}(|X_1|>t)\,dt\ =\ \infty$$
(the integrand is non-increasing). More generally, for any $c>0$, $\sum_n \mathbb{P}(|X_1|>cn)=\infty$, by the same comparison applied to $|X_1|/c$.

Fix $c>0$ and let $A_n:=\{|X_n|>cn\}$. These are independent (functions of distinct $X_n$'s, Definition 1.23) and $\sum_n\mathbb{P}(A_n)=\sum_n\mathbb{P}(|X_1|>cn)=\infty$, so BC-II (Theorem 7.12) gives $\mathbb{P}(|X_n|>cn\text{ i.o.})=1$.

Now on that event, note $X_n = S_n - S_{n-1}$, so
$$\frac{|X_n|}{n}\le\frac{|S_n|}{n}+\frac{n-1}{n}\cdot\frac{|S_{n-1}|}{n-1}\le 2\max\Big\{\frac{|S_n|}{n},\frac{|S_{n-1}|}{n-1}\Big\}.$$
Infinitely often $|X_n|/n>c$, hence infinitely often $\max\{|S_n|/n,|S_{n-1}|/(n-1)\}>c/2$, hence $\limsup_n|S_n|/n\ge c/2$ almost surely. Intersecting over $c\in\mathbb{N}$ (countably many probability-one events) gives $\limsup_n|S_n|/n=\infty$ a.s. $\square$

**Worked Example 7.11 (Cauchy sample means never settle).** Let $X_i$ be i.i.d. standard Cauchy, density $\frac{1}{\pi(1+x^2)}$. Then $\mathbb{E}|X_1|=\frac{2}{\pi}\int_0^\infty\frac{x}{1+x^2}dx=\infty$, so Proposition 7.20 applies: $\limsup_n|\bar X_n|=\infty$ a.s. In fact something sharper is true and we will prove it in §7.4: $\bar X_n$ has *exactly the same law* as $X_1$ for every $n$. Averaging a Cauchy sample accomplishes literally nothing.

A simulation makes the failure vivid. With numpy's PCG64 generator seeded at $7$ and $N=10^5$ draws, the running means are

| $n$ | $\bar X_n$ | sample median |
|---:|---:|---:|
| $10$ | $0.7238$ | $0.3832$ |
| $10^2$ | $0.0034$ | $0.0354$ |
| $10^3$ | $1.3140$ | $0.0635$ |
| $10^4$ | $-0.5858$ | $0.0097$ |
| $10^5$ | $0.4404$ | $-0.0117$ |

The mean wanders without shrinking; over $n\in[10^3,10^5]$ the running mean reaches $|\bar X_n|=1.326$. Meanwhile the sample median converges nicely to the true median $0$ — the Cauchy distribution has a perfectly good center, it just cannot be estimated by averaging. In this sample $78$ of the $10^5$ draws exceeded $1000$ in absolute value and the largest was $60154$; a single such draw at $n=10^3$ shifts the mean by $60$.

The ML lesson is not exotic. Ratios of near-zero-mean quantities are Cauchy-like; importance weights $w=p/q$ with a poorly matched proposal have heavy tails; per-example gradients under a heavy-tailed feature distribution do too. When your Monte Carlo estimate refuses to stabilize as you add samples, the first hypothesis should be *the mean does not exist or the variance does not*, not *I need more samples*.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="60" y1="170" x2="580" y2="170"/><line x1="60" y1="20" x2="60" y2="170"/>
  </g>
  <g stroke="currentColor" stroke-width="0.6" fill="none" stroke-dasharray="3 3">
    <line x1="60" y1="95" x2="580" y2="95"/>
  </g>
  <polyline fill="none" stroke="#ef4444" stroke-width="1.4" points="60,95 90,89 108,85 120,87 130,83 138,76 145,80 150,82 155,71 160,68 164,44 168,63 175,57 178,60 183,55 186,60 190,54 194,82 198,73 203,77 208,82 212,80 216,79 220,74 224,77 228,77 232,80 236,71 241,74 245,78 249,81 253,86 257,92 262,94 266,95 270,97 274,84 278,84 283,84 287,84 291,82 295,160 299,151 304,153 308,149 312,136 316,130 321,123 325,116 329,116 333,112 337,100 341,99 346,101 350,31 354,29 358,32 363,57 367,66 371,70 375,71 379,70 384,72 388,79 392,83 396,88 400,90 405,91 409,93 413,94 417,99 421,97 426,104 430,110 434,127 438,126 442,122 447,126 451,124 455,120 459,117 463,103 468,120 472,126 476,116 480,124 484,117 489,113 493,113 497,109 501,102 505,99 510,94 514,84 518,82 522,83 526,114 531,110 535,68 539,69 543,72 547,73 552,77 556,76 560,78"/>
  <polyline fill="none" stroke="#10b981" stroke-width="1.8" points="60,95 90,89 108,83 120,88 130,83 138,81 145,83 150,88 155,83 160,81 164,78 168,81 175,81 178,83 183,78 186,81 190,81 194,81 198,81 203,90 208,94 212,95 216,94 220,92 224,92 228,94 232,92 236,89 241,93 245,93 249,92 253,93 257,93 262,93 266,93 270,94 274,94 278,94 283,95 287,93 291,93 295,94 299,95 304,94 308,94 312,94 316,94 321,94 325,93 329,94 333,95 337,93 341,93 346,93 350,93 354,93 358,92 363,93 367,93 371,93 375,93 379,93 384,93 388,93 392,93 396,93 400,93 405,94 409,93 413,93 417,94 421,94 426,93 430,94 434,94 438,95 442,95 447,95 451,95 455,95 459,95 463,95 468,95 472,95 476,95 480,95 484,95 489,95 493,95 497,95 501,95 505,95 510,95 514,95 518,95 522,95 526,95 531,95 535,95 539,95 543,95 547,95 552,95 556,95 560,95"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.2" stroke-dasharray="4 3" points="60,95 90,89 108,95 120,103 130,105 138,109 145,107 150,99 155,101 160,103 164,100 168,99 175,100 178,100 183,101 186,102 190,107 194,109 198,110 203,112 208,111 212,113 216,112 220,110 224,109 228,107 232,106 236,104 241,103 245,102 249,102 253,102 257,100 262,101 266,101 270,101 274,101 278,101 283,102 287,102 291,99 295,100 299,101 304,101 308,100 312,99 316,99 321,99 325,99 329,100 333,100 337,100 341,100 346,99 350,99 354,99 358,98 363,98 367,98 371,97 375,97 379,97 384,97 388,97 392,96 396,96 400,96 405,96 409,96 413,96 417,96 421,96 426,96 430,96 434,96 438,95 442,96 447,96 451,96 455,96 459,95 463,95 468,95 472,95 476,95 480,95 484,95 489,95 493,95 497,95 501,95 505,95 510,95 514,95 518,95 522,95 526,95 531,95 535,95 539,95 543,95 547,95 552,95 556,95 560,95"/>
  <g font-size="11" fill="currentColor" font-family="sans-serif">
    <text x="54" y="99" text-anchor="end">0</text><text x="54" y="24" text-anchor="end">+2</text><text x="54" y="174" text-anchor="end">−2</text>
    <text x="60" y="188" text-anchor="middle">1</text><text x="160" y="188" text-anchor="middle">10</text><text x="260" y="188" text-anchor="middle">10²</text>
    <text x="360" y="188" text-anchor="middle">10³</text><text x="460" y="188" text-anchor="middle">10⁴</text><text x="560" y="188" text-anchor="middle">10⁵</text>
    <text x="320" y="204" text-anchor="middle">n  (log scale)</text>
  </g>
  <g font-size="11" font-family="sans-serif">
    <text x="596" y="80" fill="#ef4444">Cauchy mean</text>
    <text x="596" y="98" fill="#10b981">Cauchy median</text>
    <text x="596" y="116" fill="#3b82f6">N(0,1) mean</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="232">Figure 7.3. Running statistics of 100,000 i.i.d. draws (numpy PCG64, seed 7). The Cauchy sample MEAN (red)</text>
    <text x="20" y="251">never settles: E|X| = ∞, so by Proposition 7.20 its limsup is infinite. The Cauchy sample MEDIAN (green)</text>
    <text x="20" y="270">converges to 0 without trouble, and a Gaussian sample mean (blue) converges at the familiar n^(−1/2) rate.</text>
    <text x="20" y="289">Heavy tails do not destroy the center of a distribution; they destroy the AVERAGE as a way of finding it.</text>
  </g>
</svg>

### 7.3.3 What the SLLN buys machine learning, and what it does not

Fix a hypothesis $h$ and a loss $\ell$, and let $R(h):=\mathbb{E}_{(X,Y)\sim P}[\ell(h(X),Y)]$ be the true risk and $\hat R_n(h):=\frac1n\sum_{i=1}^n\ell(h(X_i),Y_i)$ the empirical risk on an i.i.d. sample.

**Corollary 7.21 (pointwise consistency of empirical risk).** If $\mathbb{E}|\ell(h(X),Y)|<\infty$ for a *fixed* $h$, then $\hat R_n(h)\to_{\mathrm{as}}R(h)$.

*Proof.* The variables $\ell(h(X_i),Y_i)$ are i.i.d. (measurable functions of i.i.d. pairs, Proposition 1.15) with finite mean; apply Theorem 7.19. $\square$

The word doing all the work is *fixed*. Corollary 7.21 says: for each $h$ separately, there is a null set $N_h$ off which $\hat R_n(h)\to R(h)$. Training picks $\hat h_n=\arg\min_h\hat R_n(h)$, which *depends on the sample*, so it is not a fixed $h$ and the corollary says nothing about $\hat R_n(\hat h_n)$. Worse, the union $\bigcup_{h\in\mathcal H}N_h$ over an uncountable class need not even be measurable, let alone null — this is precisely the countability warning Chapter 1 issued and the reason Chapter 4 built uniform convergence over $\mathcal H$ via covering numbers, VC dimension, and Rademacher complexity.

Here is a two-line proof that pointwise is genuinely not enough. Let $P$ be $\mathrm{Unif}[0,1]$ on $X$ with $Y=0$ always, $\ell(h(x),y)=\mathbf 1\{h(x)\ne y\}$, and let $\mathcal H$ contain, for every finite set $F\subset[0,1]$, the classifier $h_F=\mathbf 1_F$. For each fixed $F$, $R(h_F)=\lambda(F)=0=\lim_n\hat R_n(h_F)$, fine. But $\hat h_n:=h_{\{X_1,\dots,X_n\}}$ has $\hat R_n(\hat h_n)=1$ and $R(\hat h_n)=0$ — a gap of $1$ for every $n$, forever. (Take the loss as $\mathbf 1\{h(x)=y\}$ if you prefer the gap the other way.) The SLLN is true here for every $h$; it simply does not apply to $\hat h_n$.

> **Common Confusion 7.4.**
> *"The SLLN says $\hat R_n\to R$, so my training loss converges to my test loss."*
>
> It says $\hat R_n(h)\to R(h)$ for each fixed $h$, on a null set that depends on $h$. The trained $\hat h_n$ is a function of the data. Generalization requires $\sup_{h\in\mathcal H}|\hat R_n(h)-R(h)|\to0$, a statement about a supremum over a possibly uncountable family — not even automatically a random variable without separability assumptions (Ch. 1's countability theme), and false without a complexity restriction on $\mathcal H$. That is Chapter 4's subject, and §7.8 gives the one case where the uniform statement is free: the empirical CDF.

**Worked Example 7.12 (Monte Carlo consistency, with the rate).** Take Chapter 1's running example (ii): $A=\max(0,Z)$ with $Z\sim\mathcal N(0,1)$, for which $\mathbb{E}[A]=1/\sqrt{2\pi}=0.398942$. Estimate it by $\bar A_n$ from i.i.d. draws. Since $\mathbb{E}[A^2]=\mathbb{E}[Z^2]/2=1/2$, we get $\operatorname{Var}(A)=\tfrac12-\tfrac1{2\pi}=0.340845$ exactly. With numpy PCG64, seed 31415:

| $n$ | $\bar A_n$ | error |
|---:|---:|---:|
| $10$ | $0.480194$ | $+0.081252$ |
| $10^2$ | $0.429689$ | $+0.030746$ |
| $10^4$ | $0.396226$ | $-0.002716$ |
| $10^6$ | $0.399384$ | $+0.000442$ |

The theoretical standard error at $n=10^6$ is $\sqrt{0.340845}/1000=0.000584$, and the observed error $0.000442$ sits comfortably inside it. Note the shape of the error column: $10^{-1}$, $10^{-1.5}$, $10^{-2.5}$, $10^{-3.4}$ — roughly one decimal digit of accuracy per two orders of magnitude of $n$, which is the signature of the $n^{-1/2}$ rate. Recall from Chapter 1 that $A$ has neither a pmf nor a pdf; the SLLN does not care, because it is a statement about the law $P_A$ and its mean, not about densities. This is Failure 2 from Chapter 1 continuing to be a non-issue once you stop insisting on densities.

> **Common Confusion 7.5.**
> *"The law of large numbers says the error goes to zero, so doubling my Monte Carlo sample halves my error."*
>
> No: it divides the error by $\sqrt2\approx1.414$. The LLN gives *consistency*, not a rate; the rate comes from the CLT (§7.5) and is $\sigma/\sqrt n$. To gain one decimal digit you need $100\times$ the samples. This is the single most consequential fact about Monte Carlo in practice: it is why variance reduction (control variates, antithetic sampling, importance sampling, common random numbers) is a whole discipline, and why "just run it longer" stops being a strategy around $n\approx10^6$.

> **Self-check 7.3.**
> 1. Where exactly in the proof of Theorem 7.18 was the fourth moment needed, and what would go wrong with only a second moment?
> 2. State the hypothesis of Theorem 7.16 that can be weakened from independence to something strictly weaker.
> 3. If $X_i$ are i.i.d. with $\mathbb{P}(X_1>t)=t^{-3/2}$ for $t\ge1$, does $\bar X_n$ converge a.s.? Does $\operatorname{Var}(X_1)$ exist?
> 4. Why does Corollary 7.21 fail to give a generalization bound?

---

## 7.4 Characteristic functions

### 7.4.1 Why complex exponentials

Chapter 3 introduced the moment generating function $M_X(t)=\mathbb{E}[e^{tX}]$ and showed how convenient it is: it turns sums of independent variables into products, and its derivatives at $0$ read off moments. It has one fatal defect. The integrand $e^{tX}$ is unbounded, so $M_X(t)$ can be $+\infty$ — and for genuinely heavy-tailed distributions it is $+\infty$ for every $t\ne0$. The Cauchy distribution has no MGF anywhere except $t=0$; neither does the $t_3$; the lognormal has $M(t)=\infty$ for all $t>0$. Any theory built on MGFs has a hole in it exactly where heavy tails live, which in machine learning is exactly where the interesting failures live.

The repair is to replace the real exponential by a complex one. The map $x\mapsto e^{itx}=\cos(tx)+i\sin(tx)$ has modulus $1$ for every real $t$ and $x$. So its expectation always exists, for every distribution, with no assumptions whatsoever. You pay for this by working with complex-valued integrals, which is a small price: the integral of a complex-valued function is defined by integrating real and imaginary parts separately, and every property from Chapter 2 (linearity, dominated convergence, Fubini) transfers immediately, along with the triangle inequality $|\int f\,d\mu|\le\int|f|\,d\mu$.

**Definition 7.22 (characteristic function).** The characteristic function of a random variable $X$ (equivalently, of its law $P_X$) is
$$\varphi_X(t):=\mathbb{E}\big[e^{itX}\big]=\int_{\mathbb{R}}e^{itx}\,dP_X(x),\qquad t\in\mathbb{R}.$$
For a random vector $X$ in $\mathbb{R}^d$, $\varphi_X(t):=\mathbb{E}[e^{i\langle t,X\rangle}]$ for $t\in\mathbb{R}^d$.

If $X$ has a density $p$ with respect to Lebesgue measure, $\varphi_X$ is (a sign convention away from) the Fourier transform of $p$. But Definition 7.22 needs no density — it is defined for $P_X=\delta_0$, for the Chapter 1 running example $A=\max(0,Z)$ which has neither pmf nor pdf, and for the circle law $G_\#\mathrm{Unif}[0,2\pi)$ in $\mathbb{R}^2$ which is Lebesgue-singular. That universality is the point.

**Proposition 7.23 (elementary properties).** For any random variable $X$:

1. $\varphi_X(0)=1$ and $|\varphi_X(t)|\le1$ for all $t$;
2. $\varphi_X(-t)=\overline{\varphi_X(t)}$; in particular $\varphi_X$ is real-valued iff $X\overset{d}{=}-X$ (a symmetric law);
3. $\varphi_X$ is uniformly continuous on $\mathbb{R}$;
4. for constants $a,b$: $\varphi_{aX+b}(t)=e^{itb}\varphi_X(at)$.

*Proof.* (1) $\varphi_X(0)=\mathbb{E}[1]=1$; and $|\varphi_X(t)|=|\mathbb{E}[e^{itX}]|\le\mathbb{E}|e^{itX}|=1$.
(2) $\varphi_X(-t)=\mathbb{E}[e^{-itX}]=\mathbb{E}[\overline{e^{itX}}]=\overline{\mathbb{E}[e^{itX}]}$. If $X\overset{d}{=}-X$ then $\varphi_X(t)=\varphi_{-X}(t)=\varphi_X(-t)=\overline{\varphi_X(t)}$, so $\varphi_X$ is real; conversely if $\varphi_X$ is real then $\varphi_X=\varphi_{-X}$, and Corollary 7.29 below gives $X\overset{d}{=}-X$.
(3) $|\varphi_X(t+h)-\varphi_X(t)|=|\mathbb{E}[e^{itX}(e^{ihX}-1)]|\le\mathbb{E}|e^{ihX}-1|$, a bound independent of $t$. As $h\to0$, $|e^{ihX}-1|\to0$ pointwise and is bounded by the constant $2$, so dominated convergence (Ch. 2) gives $\mathbb{E}|e^{ihX}-1|\to0$.
(4) $\mathbb{E}[e^{it(aX+b)}]=e^{itb}\mathbb{E}[e^{i(ta)X}]$. $\square$

**Proposition 7.24 (independence turns convolution into multiplication).** If $X\perp Y$ then $\varphi_{X+Y}=\varphi_X\varphi_Y$. More generally, if $X_1,\dots,X_n$ are independent then $\varphi_{\sum_i X_i}(t)=\prod_i\varphi_{X_i}(t)$.

*Proof.* $e^{it(X+Y)}=e^{itX}e^{itY}$, and independence gives $\mathbb{E}[e^{itX}e^{itY}]=\mathbb{E}[e^{itX}]\mathbb{E}[e^{itY}]$ — factorization of expectations of products of independent variables, which follows from Definition 1.23 via Fubini (Ch. 2) applied to the product measure $P_X\otimes P_Y$; the integrand is bounded so integrability is automatic. Induct for general $n$. $\square$

This is the whole reason characteristic functions are the right tool for the CLT. A sum of $n$ independent variables has a law given by an $n$-fold convolution, an object nobody wants to touch directly; on the characteristic-function side it is an $n$-th power.

> **Common Confusion 7.6.**
> *"The MGF and the characteristic function are the same object with $t$ replaced by $it$, so why bother with complex numbers?"*
>
> They agree *when both exist*, in the sense that $\varphi_X(t)=M_X(it)$ when $M_X$ is finite on a neighbourhood of $0$ and extends analytically. But that hypothesis is the whole issue. The Cauchy law has $M_X(t)=\infty$ for every $t\ne0$, so there is nothing to substitute into; yet $\varphi_X(t)=e^{-|t|}$ exists, is continuous, and determines the law. And note $e^{-|t|}$ is *not* analytic at $0$ — it has no derivative there — which correctly encodes "no finite mean". A second, sharper point: moments need not determine a law even when all of them are finite. The lognormal distribution has every moment finite, yet there is an infinite family of distinct distributions sharing all of its moments (Heyde's example). The characteristic function never has this problem: Corollary 7.29 says it always determines the law, full stop.

### 7.4.2 Four computations

**Worked Example 7.13 (Bernoulli and binomial).** If $X\sim\mathrm{Bernoulli}(p)$ then
$$\varphi_X(t)=(1-p)e^{it\cdot0}+pe^{it\cdot1}=1-p+pe^{it}.$$
By Proposition 7.24, a $\mathrm{Bin}(n,p)$ variable has $\varphi(t)=(1-p+pe^{it})^n$. Check the modulus: $|1-p+pe^{it}|^2=(1-p+p\cos t)^2+p^2\sin^2t=1-2p(1-p)(1-\cos t)\le1$, with equality exactly at $t\in2\pi\mathbb{Z}$. The equality set is a discrete subgroup of $\mathbb{R}$ — that is the analytic fingerprint of a *lattice* distribution, and it is the reason lattice variables get a worse Berry–Esseen constant in §7.5 (Worked Example 7.20).

**Worked Example 7.14 (Gaussian, by differential equation).** Let $Z\sim\mathcal N(0,1)$ with density $\phi(z)=(2\pi)^{-1/2}e^{-z^2/2}$. Then $\varphi_Z(t)=e^{-t^2/2}$.

*Derivation.* Differentiate under the integral sign: $\frac{\partial}{\partial t}e^{itz}\phi(z)=ize^{itz}\phi(z)$, dominated in modulus by $|z|\phi(z)$, which is integrable, so the interchange is licensed by the dominated convergence theorem (Ch. 2) — this is *precisely* Chapter 1's Failure 3, swapping a derivative and an expectation, now discharged. Hence
$$\varphi_Z'(t)=i\int z\,e^{itz}\phi(z)\,dz.$$
Now use the defining identity $\phi'(z)=-z\phi(z)$ and integrate by parts (boundary terms vanish since $\phi(z)\to0$ faster than any polynomial):
$$i\int ze^{itz}\phi(z)\,dz=-i\int e^{itz}\phi'(z)\,dz=i\int (it)e^{itz}\phi(z)\,dz=-t\,\varphi_Z(t).$$
So $\varphi_Z'=-t\varphi_Z$ with $\varphi_Z(0)=1$. The unique solution is $\varphi_Z(t)=e^{-t^2/2}$. By Proposition 7.23(4), $X\sim\mathcal N(\mu,\sigma^2)$ has $\varphi_X(t)=e^{i\mu t-\sigma^2t^2/2}$.

An immediate corollary, cleaner than any convolution computation: if $X\sim\mathcal N(\mu_1,\sigma_1^2)\perp Y\sim\mathcal N(\mu_2,\sigma_2^2)$ then
$$\varphi_{X+Y}(t)=e^{i\mu_1t-\sigma_1^2t^2/2}e^{i\mu_2t-\sigma_2^2t^2/2}=e^{i(\mu_1+\mu_2)t-(\sigma_1^2+\sigma_2^2)t^2/2},$$
so $X+Y\sim\mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ by uniqueness. The Gaussian family is closed under independent sums, and the proof is two lines.

**Worked Example 7.15 (Poisson).** If $X\sim\mathrm{Poisson}(\lambda)$ then
$$\varphi_X(t)=\sum_{k\ge0}e^{-\lambda}\frac{\lambda^k}{k!}e^{itk}=e^{-\lambda}\sum_{k\ge0}\frac{(\lambda e^{it})^k}{k!}=e^{-\lambda}e^{\lambda e^{it}}=\exp\big(\lambda(e^{it}-1)\big).$$
(The interchange of sum and expectation is Tonelli, Ch. 2, on the counting measure; the terms are absolutely summable.) Hence independent Poissons add: $\varphi_{X+Y}(t)=\exp((\lambda+\mu)(e^{it}-1))$, so $X+Y\sim\mathrm{Poisson}(\lambda+\mu)$. Also, expanding for small $t$: $\varphi_X(t)=\exp(\lambda(it-t^2/2+O(t^3)))=1+i\lambda t-\lambda(1+\lambda)t^2/2+O(t^3)$, from which $\mathbb{E}[X]=\lambda$ and $\mathbb{E}[X^2]=\lambda+\lambda^2$ by Proposition 7.26 below — the standard moments, recovered without a single factorial manipulation.

**Worked Example 7.16 (Cauchy, and why its averages do nothing).** First compute the characteristic function of the *Laplace* density $g(x)=\tfrac12e^{-|x|}$:
$$\varphi_g(t)=\tfrac12\int_0^\infty e^{-x}\big(e^{itx}+e^{-itx}\big)\,dx=\tfrac12\Big(\frac{1}{1-it}+\frac{1}{1+it}\Big)=\frac{1}{1+t^2}.$$
This $\varphi_g$ is integrable over $\mathbb{R}$, so the inversion formula (Theorem 7.28) applies and returns the density:
$$\tfrac12e^{-|x|}=\frac{1}{2\pi}\int_{\mathbb{R}}e^{-itx}\frac{1}{1+t^2}\,dt\quad\Longleftrightarrow\quad e^{-|x|}=\int_{\mathbb{R}}e^{-itx}\,\frac{dt}{\pi(1+t^2)} .$$
The right-hand side is exactly $\mathbb{E}[e^{-ixC}]=\varphi_C(-x)$ for $C$ standard Cauchy. Since the Cauchy law is symmetric, $\varphi_C(-x)=\varphi_C(x)$, and renaming the variable,
$$\boxed{\ \varphi_C(t)=e^{-|t|}.\ }$$
Now the punchline promised in Worked Example 7.11. If $C_1,\dots,C_n$ are i.i.d. standard Cauchy, then by Propositions 7.23(4) and 7.24,
$$\varphi_{\bar C_n}(t)=\prod_{i=1}^n\varphi_{C}(t/n)=\big(e^{-|t|/n}\big)^n=e^{-|t|}=\varphi_C(t).$$
By uniqueness (Corollary 7.29), $\bar C_n\overset{d}{=}C_1$ *exactly*, for every $n$. Averaging $10^6$ Cauchy draws gives you a variable with precisely the same distribution as one draw. Figure 7.3's red curve is not "slow convergence"; it is a stationary process in disguise. Note also that $\varphi_C(t)=e^{-|t|}$ is not differentiable at $t=0$: by Proposition 7.26, a finite mean would force differentiability at $0$, so the corner certifies $\mathbb{E}|C|=\infty$. (This inference runs one way only — see the caution after Proposition 7.26: a smooth $\varphi$ at $0$ does *not* certify a finite mean.) The characteristic function *sees* the missing mean, in the form of a corner.

### 7.4.3 Expansions, inversion, uniqueness, continuity

**Lemma 7.25 (Taylor bound for the complex exponential).** For every $x\in\mathbb{R}$ and integer $n\ge1$,
$$\Big|e^{ix}-\sum_{k=0}^{n}\frac{(ix)^k}{k!}\Big|\ \le\ \min\Big\{\frac{|x|^{n+1}}{(n+1)!},\ \frac{2|x|^{n}}{n!}\Big\}.$$

*Proof.* Start from $\int_0^x e^{is}\,ds=\frac{e^{ix}-1}{i}$, i.e. $e^{ix}=1+i\int_0^x e^{is}\,ds$. Integrating by parts,
$$\int_0^x(x-s)^{n}e^{is}\,ds=\frac{x^{n+1}}{n+1}+\frac{i}{n+1}\int_0^x(x-s)^{n+1}e^{is}\,ds,$$
and an induction on $n$ starting from the displayed identity yields the exact remainder formula
$$e^{ix}=\sum_{k=0}^{n}\frac{(ix)^k}{k!}+\frac{i^{n+1}}{n!}\int_0^x(x-s)^{n}e^{is}\,ds. \tag{7.1}$$
Bounding $|e^{is}|=1$ in (7.1) gives remainder $\le\frac{1}{n!}\cdot\frac{|x|^{n+1}}{n+1}=\frac{|x|^{n+1}}{(n+1)!}$, the first bound.

For the second, subtract from (7.1) the same identity with $n$ replaced by $n-1$; the difference of the two remainder integrals rearranges to
$$\frac{i^{n+1}}{n!}\int_0^x(x-s)^{n}e^{is}\,ds=\frac{i^{n}}{(n-1)!}\int_0^x(x-s)^{n-1}\big(e^{is}-1\big)\,ds .$$
Since $|e^{is}-1|\le2$, the modulus is at most $\frac{2}{(n-1)!}\cdot\frac{|x|^{n}}{n}=\frac{2|x|^{n}}{n!}$. $\square$

The two bounds trade off exactly as you would want: the first is sharp for small $|x|$, the second for large $|x|$. That is what makes Lemma 7.25 the right tool for the Lindeberg method, where $|tX_i/\sqrt n|$ is small for typical $X_i$ but large for the rare big ones.

**Proposition 7.26 (moments and the expansion at $0$).** Suppose $\mathbb{E}|X|^{n}<\infty$. Then $\varphi_X$ is $n$ times continuously differentiable, $\varphi_X^{(k)}(0)=i^k\mathbb{E}[X^k]$ for $k\le n$, and
$$\varphi_X(t)=\sum_{k=0}^{n}\frac{(it)^k\mathbb{E}[X^k]}{k!}+o(|t|^{n})\qquad\text{as }t\to0 .$$

*Proof.* Apply Lemma 7.25 with $x=tX$ and take expectations:
$$\Big|\varphi_X(t)-\sum_{k=0}^{n}\frac{(it)^k\mathbb{E}[X^k]}{k!}\Big|\le\mathbb{E}\Big[\min\Big\{\frac{|tX|^{n+1}}{(n+1)!},\frac{2|tX|^{n}}{n!}\Big\}\Big]=|t|^{n}\,\mathbb{E}\Big[\min\Big\{\frac{|t|\,|X|^{n+1}}{(n+1)!},\frac{2|X|^{n}}{n!}\Big\}\Big].$$
The bracketed random variable is dominated by the integrable $2|X|^{n}/n!$ and tends to $0$ pointwise as $t\to0$; dominated convergence (Ch. 2) gives that the expectation is $o(1)$, hence the whole bound is $o(|t|^n)$. The differentiability claims follow by differentiating under the integral sign, licensed by dominating $|\partial_t^k e^{itX}|=|X|^k\le 1+|X|^n$. $\square$

The case we need in §7.5 is $n=2$ with $\mathbb{E}[X]=0$, $\mathbb{E}[X^2]=1$:
$$\varphi_X(t)=1-\tfrac{t^2}{2}+o(t^2). \tag{7.2}$$
That single line contains the entire CLT.

One caution before moving on, because Proposition 7.26 is a one-way street and students reliably drive it backwards. Finite moments imply smoothness of $\varphi$ at $0$; smoothness of $\varphi$ at $0$ does **not** imply finite moments. There exist symmetric laws — the standard example has density proportional to $1/(x^2\log|x|)$ for $|x|\ge 2$ — with $\mathbb{E}|X|=\infty$ and yet $\varphi$ differentiable at $0$ with $\varphi'(0)=0$ (§7.10 gives references). So the *corner* of the Cauchy's $e^{-|t|}$ at $0$ is a certificate that the mean is missing (the valid contrapositive of Proposition 7.26), but a *smooth* characteristic function at $0$ certifies nothing about moments. If you need $\mathbb{E}[X^2]<\infty$, you must assume it; you cannot read it off the graph of $\varphi$. The correct two-sided statement, for the curious, replaces the ordinary derivative by a symmetrized second difference and is beyond what we need.

Everything downstream is bookkeeping — plus one small lemma that lets us replace a product of $n$ nearly-equal complex numbers by a product we can evaluate.

**Lemma 7.27 (product-difference bound).** Let $a_1,\dots,a_n$ and $b_1,\dots,b_n$ be complex numbers with $|a_j|\le1$ and $|b_j|\le1$ for all $j$. Then
$$\Big|\prod_{j=1}^{n}a_j-\prod_{j=1}^{n}b_j\Big|\ \le\ \sum_{j=1}^{n}|a_j-b_j| .$$

*Proof.* Telescope: writing $A_k=a_1\cdots a_k$ and $B_k=b_1\cdots b_k$,
$$\prod_j a_j-\prod_j b_j=\sum_{k=1}^{n}\big(A_{k}B_{k+1:n}-A_{k-1}B_{k:n}\big),\qquad B_{k:n}:=b_k\cdots b_n,$$
and the $k$-th summand equals $A_{k-1}(a_k-b_k)B_{k+1:n}$, whose modulus is at most $|a_k-b_k|$ because all the other factors have modulus $\le1$. Apply the triangle inequality. $\square$

The lemma is worth pausing on. Naively, an error of size $\epsilon$ in each of $n$ factors could compound multiplicatively; Lemma 7.27 says that when everything sits inside the closed unit disc, errors only *add*. This is why the CLT proof survives an $o(1/n)$ error per factor across $n$ factors.

**Theorem 7.28 (inversion formula).** Let $X$ have characteristic function $\varphi$ and CDF $F$. For any $a<b$ that are continuity points of $F$,
$$F(b)-F(a)=\lim_{T\to\infty}\frac{1}{2\pi}\int_{-T}^{T}\frac{e^{-ita}-e^{-itb}}{it}\,\varphi(t)\,dt .$$
If moreover $\varphi\in L^1(\lambda)$, then $X$ has a bounded continuous density with respect to Lebesgue measure, given by
$$p(x)=\frac{1}{2\pi}\int_{\mathbb{R}}e^{-itx}\varphi(t)\,dt .$$

The proof is a Fubini argument combined with the Dirichlet integral $\int_0^\infty\frac{\sin u}{u}du=\frac\pi2$; it is standard and is listed in §7.10. Note the second statement is a genuine theorem about *existence of densities*: integrability of $\varphi$ is a checkable sufficient condition for a law to be absolutely continuous with respect to Lebesgue measure — a concrete criterion for the Radon–Nikodym question Chapter 2 answered abstractly.

**Corollary 7.29 (uniqueness).** If $\varphi_X=\varphi_Y$ then $P_X=P_Y$.

*Proof.* By Theorem 7.28, $\varphi$ determines $F(b)-F(a)$ for all continuity points $a<b$. Letting $a\to-\infty$ along continuity points determines $F(b)$ at all continuity points of $F$. The discontinuity points of a CDF are countable (Proposition 1.19: a monotone bounded function has at most countably many jumps), so $F$ is determined on a dense set, and right-continuity (Proposition 1.19 again) pins it down everywhere. Theorem 1.20 then gives $P_X=P_Y$, since the CDF determines the law. $\square$

**Theorem 7.30 (Lévy's continuity theorem).** Let $X_1,X_2,\dots$ be random variables with characteristic functions $\varphi_n$.

1. If $X_n\to_{\mathrm d}X$ then $\varphi_n(t)\to\varphi_X(t)$ for every $t$.
2. Conversely, if $\varphi_n(t)\to\psi(t)$ for every $t$ and $\psi$ is continuous at $t=0$, then $\psi$ is the characteristic function of some random variable $X$ and $X_n\to_{\mathrm d}X$.

Direction (1) is immediate: $x\mapsto\cos(tx)$ and $x\mapsto\sin(tx)$ are bounded and continuous, so Definition 7.4's test-function formulation applies. Direction (2) is the deep one; it rests on Helly's selection theorem (every sequence of CDFs has a subsequence converging vaguely to a possibly-defective limit) plus a *tightness* argument showing that continuity of $\psi$ at $0$ forbids mass escaping to $\pm\infty$. It is recorded in §7.10.

The continuity-at-$0$ hypothesis is not decoration. Let $X_n\sim\mathcal N(0,n)$. Then $\varphi_n(t)=e^{-nt^2/2}\to\mathbf 1\{t=0\}=:\psi(t)$ pointwise. But $\psi$ is discontinuous at $0$, and indeed $X_n$ does not converge in distribution to anything: $F_n(x)\to1/2$ for every fixed $x$, and $1/2$ is not a CDF. The mass has escaped to infinity. Tightness is exactly the condition that rules this out, and continuity of the limit at $0$ is a slick way to enforce it.

> **Common Confusion 7.7.**
> *"$\varphi_n\to\psi$ pointwise means the laws converge — that's what pointwise convergence of CDFs would mean."*
>
> It means the laws converge *only if* $\psi$ is continuous at $0$, equivalently only if no mass escapes to infinity. The $\mathcal N(0,n)$ example above shows exactly how it fails. In applications this hypothesis is usually free because you have already identified $\psi$ as a known characteristic function (e.g. $e^{-t^2/2}$), which is continuous everywhere. But when you are proving that *some* limit exists, tightness is a real obligation, and in ML settings where a variance might blow up with $n$ it is the obligation that fails.

> **Self-check 7.4.**
> 1. Compute $\varphi_X$ for $X\sim\mathrm{Unif}[-1,1]$ and verify $\varphi_X(t)=1-t^2/6+o(t^2)$.
> 2. Why is $|\varphi_X(t)|\le1$ always true, whereas $M_X(t)\le1$ is usually false?
> 3. What feature of $\varphi_C(t)=e^{-|t|}$ tells you the Cauchy law has no mean?
> 4. If $\varphi_X$ is real-valued and $\varphi_X(t)=\varphi_X(2t)$ for all $t$, what can you say about $X$?

---

## 7.5 The Central Limit Theorem

### 7.5.1 The classical theorem and its proof

**Theorem 7.31 (i.i.d. Central Limit Theorem).** Let $X_1,X_2,\dots$ be i.i.d. with $\mathbb{E}[X_1]=\mu$ and $0<\sigma^2=\operatorname{Var}(X_1)<\infty$. Then
$$Z_n:=\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}=\frac{S_n-n\mu}{\sigma\sqrt n}\ \longrightarrow_{\mathrm d}\ \mathcal N(0,1).$$

*Proof.* Set $Y_i:=(X_i-\mu)/\sigma$, so the $Y_i$ are i.i.d. with $\mathbb{E}[Y_1]=0$, $\mathbb{E}[Y_1^2]=1$, and $Z_n=n^{-1/2}\sum_{i\le n}Y_i$. Write $\varphi:=\varphi_{Y_1}$.

*Step 1 (factorize).* By Propositions 7.23(4) and 7.24,
$$\varphi_{Z_n}(t)=\prod_{i=1}^{n}\varphi\big(t/\sqrt n\big)=\varphi\big(t/\sqrt n\big)^{n}.$$

*Step 2 (expand one factor).* Since $\mathbb{E}[Y_1^2]<\infty$, equation (7.2) gives $\varphi(s)=1-s^2/2+r(s)$ with $r(s)/s^2\to0$ as $s\to0$. Put $s=t/\sqrt n$ for a fixed $t$: as $n\to\infty$,
$$\varphi\big(t/\sqrt n\big)=1-\frac{t^2}{2n}+r\big(t/\sqrt n\big),\qquad n\,\big|r(t/\sqrt n)\big|=t^2\cdot\frac{|r(t/\sqrt n)|}{(t/\sqrt n)^2}\longrightarrow0 .$$

*Step 3 (multiply the errors, using Lemma 7.27).* Take $a_j:=\varphi(t/\sqrt n)$ and $b_j:=1-\frac{t^2}{2n}$ for $j=1,\dots,n$. We have $|a_j|\le1$ by Proposition 7.23(1), and $|b_j|\le1$ as soon as $n\ge t^2/4$, which holds for all large $n$ since $t$ is fixed. Lemma 7.27 gives
$$\Big|\varphi_{Z_n}(t)-\Big(1-\frac{t^2}{2n}\Big)^{n}\Big|\le n\Big|\varphi\big(t/\sqrt n\big)-\Big(1-\frac{t^2}{2n}\Big)\Big|=n\big|r(t/\sqrt n)\big|\longrightarrow0 .$$

*Step 4 (evaluate the comparison product).* For $n>t^2/2$, $\log(1-\frac{t^2}{2n})=-\frac{t^2}{2n}+O(n^{-2})$, so $n\log(1-\frac{t^2}{2n})\to-\frac{t^2}{2}$ and $(1-\frac{t^2}{2n})^{n}\to e^{-t^2/2}$.

*Step 5 (conclude).* Combining Steps 3 and 4, $\varphi_{Z_n}(t)\to e^{-t^2/2}$ for every $t\in\mathbb{R}$. The limit $e^{-t^2/2}$ is the characteristic function of $\mathcal N(0,1)$ (Worked Example 7.14) and is continuous at $0$. Lévy's continuity theorem (Theorem 7.30(2)) gives $Z_n\to_{\mathrm d}\mathcal N(0,1)$. $\square$

Look at what the proof consumed. Finiteness of $\sigma^2$: used once, in Step 2, to get the second-order expansion (7.2). Independence: used once, in Step 1. Identical distribution: used to write one $\varphi$ instead of $n$ different ones — and §7.5.2 shows that assumption is the easiest to drop. Nothing else. In particular the CLT never touches densities, so it applies verbatim to lattice variables, to Chapter 1's $A=\max(0,Z)$ (neither pmf nor pdf), and to any mixture you like.

**Worked Example 7.17 (a confidence interval for a quantity with no density).** Take $A=\max(0,Z)$ again, $\mu=\mathbb{E}[A]=1/\sqrt{2\pi}=0.398942$, $\sigma^2=\operatorname{Var}(A)=\tfrac12-\tfrac1{2\pi}=0.340845$. Theorem 7.31 says $\sqrt n(\bar A_n-\mu)/\sigma\to_{\mathrm d}\mathcal N(0,1)$, so an asymptotic $95\%$ interval is $\bar A_n\pm1.96\,\sigma/\sqrt n$. At $n=10^4$: half-width $1.96\sqrt{0.340845}/100=0.011443$. The simulated estimate from Worked Example 7.12 was $\bar A_{10^4}=0.396226$, giving $[0.384783,\ 0.407669]$, which does contain $\mu=0.398942$. Chapter 1 flagged $A$ as pathological because it has neither a pmf nor a pdf; Chapter 2 explained why that never mattered; the CLT confirms it, because the CLT reads only $\mu$ and $\sigma$ off the law.

### 7.5.2 Non-identically-distributed summands: Lindeberg–Feller

In practice the summands are rarely identically distributed. Minibatch gradients on a non-i.i.d. data stream, per-round regret increments, per-token losses in a language model — all are independent-ish but with varying scale. Lindeberg's condition is the right generalization, and its content in words is: *no single summand contributes a non-vanishing fraction of the total variance.*

**Theorem 7.32 (Lindeberg–Feller CLT).** For each $n$, let $X_{n,1},\dots,X_{n,k_n}$ be independent with $\mathbb{E}[X_{n,j}]=0$ and $\mathbb{E}[X_{n,j}^2]<\infty$. Put $s_n^2:=\sum_{j=1}^{k_n}\mathbb{E}[X_{n,j}^2]$ and $S_n:=\sum_{j=1}^{k_n}X_{n,j}$. If the **Lindeberg condition** holds — for every $\varepsilon>0$,
$$L_n(\varepsilon):=\frac{1}{s_n^2}\sum_{j=1}^{k_n}\mathbb{E}\Big[X_{n,j}^2\,\mathbf 1\big\{|X_{n,j}|>\varepsilon s_n\big\}\Big]\ \longrightarrow\ 0,$$
then $S_n/s_n\to_{\mathrm d}\mathcal N(0,1)$. Moreover the condition forces $\max_{j\le k_n}\mathbb{E}[X_{n,j}^2]/s_n^2\to0$, and under that "uniform asymptotic negligibility" the Lindeberg condition is also *necessary* (Feller).

The proof is the same four steps as Theorem 7.31, with Lemma 7.25's *two* bounds doing the work that (7.2) did before: on the event $\{|X_{n,j}|\le\varepsilon s_n\}$ use the sharp small-$x$ bound $|x|^3/6$, and on its complement use the crude $2|x|^2/2!$ bound. The Lindeberg quantity is precisely what controls the second piece. §7.10 records a reference.

Reading the condition. $s_n^2$ is the total variance. $\mathbb{E}[X_{n,j}^2\mathbf 1\{|X_{n,j}|>\varepsilon s_n\}]$ is the part of summand $j$'s variance contributed by values exceeding $\varepsilon$ times the total scale. Lindeberg says: after rescaling by the total standard deviation, the summands are *collectively uniformly square-integrable* — the variance is spread out, not concentrated in a few terms or in extreme values. If one term dominates, the sum inherits that term's shape rather than becoming Gaussian.

**Corollary 7.33 (Lyapunov's condition).** If for some $\delta>0$,
$$\frac{1}{s_n^{2+\delta}}\sum_{j=1}^{k_n}\mathbb{E}\big|X_{n,j}\big|^{2+\delta}\longrightarrow0,$$
then the Lindeberg condition holds, hence $S_n/s_n\to_{\mathrm d}\mathcal N(0,1)$.

*Proof.* On $\{|X_{n,j}|>\varepsilon s_n\}$ we have $1\le|X_{n,j}|^\delta/(\varepsilon s_n)^\delta$, so
$$X_{n,j}^2\mathbf 1\{|X_{n,j}|>\varepsilon s_n\}\le\frac{|X_{n,j}|^{2+\delta}}{\varepsilon^{\delta}s_n^{\delta}} .$$
Summing and dividing by $s_n^2$ gives $L_n(\varepsilon)\le\varepsilon^{-\delta}s_n^{-(2+\delta)}\sum_j\mathbb{E}|X_{n,j}|^{2+\delta}\to0$. $\square$

Lyapunov's condition is easier to check because it needs no truncation, only a moment slightly beyond the second. Taking $\delta=1$: bounded third absolute moments plus $s_n\to\infty$ at a decent rate suffices. This is the version you will most often apply in practice.

**A counterexample worth carrying.** Let $X_{n,j}$ be independent with $X_{n,1}\sim\mathrm{Unif}\{-n,n\}$ and $X_{n,j}\sim\mathrm{Unif}\{-1,1\}$ for $2\le j\le n$. Then $s_n^2=n^2+(n-1)\approx n^2$, and the single term $X_{n,1}$ carries essentially all the variance. Indeed $S_n/s_n\approx X_{n,1}/n\sim\mathrm{Unif}\{-1,1\}$, which is as far from Gaussian as a variable can be. Lindeberg's condition fails: for $\varepsilon<1$, $\mathbb{E}[X_{n,1}^2\mathbf 1\{|X_{n,1}|>\varepsilon s_n\}]=n^2$, so $L_n(\varepsilon)\to1$. The picture to keep: *one whale plus many minnows is not Gaussian, it is a whale.*

### 7.5.3 How fast? Berry–Esseen

**Theorem 7.34 (Berry–Esseen).** Let $X_1,\dots,X_n$ be i.i.d. with mean $\mu$, variance $\sigma^2\in(0,\infty)$, and finite third absolute central moment $\rho:=\mathbb{E}|X_1-\mu|^3$. Then, with $Z_n=(S_n-n\mu)/(\sigma\sqrt n)$,
$$\sup_{x\in\mathbb{R}}\big|\mathbb{P}(Z_n\le x)-\Phi(x)\big|\ \le\ \frac{C\rho}{\sigma^3\sqrt n},$$
where $C$ is an absolute constant; the best published value is $C\le0.4748$ (Shevtsova, 2011). The rate $n^{-1/2}$ cannot be improved in general.

Three things to extract. First, the rate is $n^{-1/2}$ — the *same* order as the quantity being approximated, which is why the CLT gives useful but never spectacular accuracy. Second, the constant scales with the *standardized third moment* $\rho/\sigma^3$, a scale-free measure of tail heaviness and asymmetry. Third, the bound is uniform in $x$, which is exactly what you want for confidence intervals but is also its weakness: the absolute error $n^{-1/2}$ swamps a tail probability like $10^{-4}$, so Berry–Esseen says nothing useful about extreme quantiles. (Relative-error statements need large-deviation theory, not the CLT.)

**Worked Example 7.18 (CLT for a rare event: Bernoulli$(0.01)$ at $n=30$).** Take $X_i\sim\mathrm{Bernoulli}(0.01)$, so $\mu=0.01$, $\sigma=\sqrt{0.0099}=0.09950$, $S_{30}\sim\mathrm{Bin}(30,0.01)$ with mean $0.3$ and standard deviation $0.5450$. Exact versus normal:

| quantity | exact | normal approximation | with continuity correction |
|---|---:|---:|---:|
| $\mathbb{P}(S_{30}=0)$ | $0.73970$ | $0.29099$ | $0.64319$ |
| $\mathbb{P}(S_{30}\ge2)$ | $0.03615$ | $0.00091$ | $0.01384$ |
| $\mathbb{P}(S_{30}\ge3)$ | $0.00332$ | $0.00000036$ | — |

The normal approximation is not merely imprecise; it is wrong by a factor of $40$ on $\mathbb{P}(S\ge2)$ and by four orders of magnitude on $\mathbb{P}(S\ge3)$. It also assigns probability $0.291$ to the event $\{S_{30}\le0\}$ interpreted as $\{Z_{30}\le-0.5505\}$, when in fact $S_{30}\le0$ means $S_{30}=0$, an event of probability $0.740$. And the nominal $95\%$ normal interval for $S_{30}$ has lower endpoint $0.3-1.96(0.5450)=-0.768$, a value the count can never take.

The exact supremum distance is $\sup_x|\mathbb{P}(Z_{30}\le x)-\Phi(x)|=0.4487$. For comparison, Berry–Esseen with $\rho/\sigma^3=(p^2+q^2)/\sqrt{pq}=9.8514$ gives the bound $0.4748\times9.8514/\sqrt{30}=0.854$ — correct but vacuous, since any two CDFs are within $1$ of each other. The rule of thumb "$n\ge30$" is not a theorem; the honest rule for a Bernoulli parent is $np(1-p)\gtrsim10$, which here demands $n\gtrsim1010$. Even at $n=1000$ the exact sup-distance is still $0.0830$.

This matters in ML more than it looks. Click-through rates, rare-class recall, catastrophic-failure rates, and per-token error rates for a well-trained model are all small-$p$ Bernoulli means. Reporting $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}$ when $n\hat p<10$ produces intervals that are wrong in width, wrong in position, and sometimes include negative rates. Use the Clopper–Pearson or Wilson interval instead; Chapter 9 returns to this.

**Worked Example 7.19 (CLT under skew: exponential parent at $n=30$).** Let $X_i\sim\mathrm{Exp}(1)$, so $\mu=\sigma=1$, skewness $2$, and $S_n\sim\mathrm{Gamma}(n,1)$ exactly, which lets us compute everything without simulation. At $n=30$:

| $z$ | exact $\mathbb{P}(Z_{30}>z)$ | $1-\Phi(z)$ | ratio |
|---:|---:|---:|---:|
| $1.645$ | $0.05894$ | $0.04998$ | $1.179$ |
| $2.326$ | $0.01704$ | $0.01001$ | $1.703$ |
| $3.000$ | $0.00416$ | $0.00135$ | $3.084$ |

The centre is fine — a nominal $5\%$ one-sided test really rejects about $5.9\%$ of the time — but the far tail is understated by a factor of three. Skewness leaks into the tails first. This is the generic picture for right-skewed losses (latency, cost, exponential-family counts): the mean is well approximated long before the tail is, so $p$-values near $0.05$ are roughly trustworthy at $n=30$ while $p$-values near $0.001$ are not.

**Worked Example 7.20 (measuring the Berry–Esseen rate exactly).** For four parents I computed $D_n:=\sup_x|\mathbb{P}(Z_n\le x)-\Phi(x)|$ *exactly* — binomial CDF for the Bernoulli parents, Gamma CDF for the exponential, and Gil–Pelaez numerical inversion of $\varphi_{Z_n}(t)=(1+|t|/\sqrt n)^{n}e^{-\sqrt n|t|}$ for the $t_3$ parent (the $t_3$ characteristic function is $(1+\sqrt3|t|)e^{-\sqrt3|t|}$). No simulation, no seeds — these are deterministic numerical values, reported to four decimals.

| $n$ | Bern$(0.01)$ | Bern$(0.5)$ | Exp$(1)$ | $t_3$ |
|---:|---:|---:|---:|---:|
| $1$ | $0.5300$ | $0.3413$ | $0.1587$ | $0.0869$ |
| $5$ | $0.5399$ | $0.1726$ | $0.0596$ | $0.0442$ |
| $30$ | $0.4487$ | $0.0722$ | $0.0243$ | $0.0192$ |
| $100$ | $0.2358$ | $0.0398$ | $0.0133$ | $0.0108$ |
| $1000$ | $0.0830$ | $0.0126$ | $0.0042$ | $0.0035$ |
| $10^4$ | $0.0266$ | $0.0040$ | $0.0013$ | $0.0011$ |

Multiply each column by $\sqrt n$ and the pattern becomes transparent:

| $n$ | Bern$(0.01)$ | Bern$(0.5)$ | Exp$(1)$ | $t_3$ |
|---:|---:|---:|---:|---:|
| $30$ | $2.458$ | $0.3956$ | $0.1330$ | $0.1054$ |
| $100$ | $2.358$ | $0.3979$ | $0.1330$ | $0.1075$ |
| $1000$ | $2.626$ | $0.3988$ | $0.1330$ | $0.1092$ |
| $10^4$ | $2.656$ | $0.3989$ | $0.1330$ | $0.1097$ |

Every column stabilizes: $\sqrt n\,D_n\to c$, confirming that $n^{-1/2}$ is the true rate, not just an upper bound. The limiting constants are identifiable. For $\mathrm{Exp}(1)$ the leading Edgeworth correction is $-\frac{\gamma}{6\sqrt n}(x^2-1)\phi(x)$ with skewness $\gamma=2$, whose supremum over $x$ is attained at $x=0$ and equals $\frac{2}{6}\phi(0)=\frac{1}{3\sqrt{2\pi}}=0.13298$ — matching the measured $0.1330$ to four decimals. For $\mathrm{Bern}(0.5)$ the skewness is $0$, so the leading error is not Edgeworth but *lattice*: the CDF of a lattice variable jumps, and half the largest jump is $\tfrac12\binom{n}{n/2}2^{-n}\approx\frac{1}{\sqrt{2\pi n}}$, giving the constant $0.3989=1/\sqrt{2\pi}$ — again matching to four decimals.

Two honest caveats. (a) $t_3$ has $\mathbb{E}|X|^3=\infty$, so Berry–Esseen's bound is literally $+\infty$ and says nothing — yet the measured rate is a clean $n^{-1/2}$ with constant $0.110$. The third-moment hypothesis is sufficient, not necessary; the theorem being silent is not the same as the rate being slow. (b) Conversely, Bern$(0.01)$ has a perfectly finite third moment and the rate *is* $n^{-1/2}$, but the constant $2.66$ is so large that the asymptotic regime does not begin until $n$ is in the thousands. The rate and the constant are different questions, and in applications the constant is usually the one that bites.

<svg viewBox="0 0 700 290" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g stroke="currentColor" stroke-width="1" fill="none">
    <line x1="45" y1="120" x2="225" y2="120"/><line x1="260" y1="120" x2="440" y2="120"/><line x1="475" y1="120" x2="655" y2="120"/>
  </g>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.8" points="45,120 48,120 51,120 54,120 57,120 60,120 63,120 66,120 69,120 72,120 75,120 78,120 81,120 84,120 87,120 90,120 93,120 96,120 99,30 102,30 105,34 108,43.5 111,51.9 114,59.4 117,66.1 120,72 123,77.3 126,82 129,86.2 132,89.9 135,93.2 138,96.2 141,98.8 144,101.1 147,103.2 150,105.1 153,106.7 156,108.2 159,109.5 162,110.6 165,111.7 168,112.6 171,113.4 174,114.1 177,114.8 180,115.3 183,115.9 186,116.3 189,116.7 192,117.1 195,117.4 198,117.7 201,117.9 204,118.2 207,118.4 210,118.6 213,118.7 216,118.9 219,119 222,119.1 225,119.2"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.8" points="260,120 263,120 266,120 269,120 272,120 275,120 278,120 281,120 284,119.8 287,119 290,117 293,113.6 296,109 299,103.3 302,97.1 305,90.9 308,84.9 311,79.6 314,75.1 317,71.7 320,69.3 323,67.9 326,67.6 329,68.1 332,69.4 335,71.3 338,73.6 341,76.3 344,79.3 347,82.4 350,85.5 353,88.6 356,91.6 359,94.5 362,97.2 365,99.7 368,102.1 371,104.2 374,106.2 377,107.9 380,109.5 383,110.9 386,112.1 389,113.2 392,114.2 395,115 398,115.7 401,116.4 404,116.9 407,117.4 410,117.8 413,118.1 416,118.4 419,118.7 422,118.9 425,119.1 428,119.2 431,119.4 434,119.5 437,119.6 440,119.6"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="1.8" points="475,119.9 478,119.9 481,119.7 484,119.5 487,119.2 490,118.7 493,118 496,117.1 499,115.7 502,114 505,111.9 508,109.4 511,106.4 514,103 517,99.4 520,95.5 523,91.5 526,87.5 529,83.8 532,80.3 535,77.3 538,74.9 541,73.1 544,71.9 547,71.5 550,71.7 553,72.6 556,74 559,76 562,78.4 565,81.1 568,84.1 571,87.1 574,90.2 577,93.3 580,96.3 583,99.1 586,101.8 589,104.3 592,106.5 595,108.5 598,110.3 601,111.9 604,113.2 607,114.4 610,115.4 613,116.3 616,117 619,117.6 622,118.1 625,118.5 628,118.8 631,119 634,119.2 637,119.4 640,119.5 643,119.7 646,119.7 649,119.8 652,119.8 655,119.9"/>
  <g fill="none" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="5 4">
    <polyline points="45,119.5 51,119 57,118.1 63,116.6 69,114.3 75,111.1 81,106.7 87,101.2 93,94.8 99,88.1 105,81.7 111,76.4 117,73.1 123,72.2 129,73.8 135,77.8 141,83.4 147,90 153,96.7 159,102.9 165,108.1 171,112.1 177,115.1 183,117.1 189,118.4 195,119.1 201,119.6 207,119.8 213,119.9 219,120 225,120"/>
    <polyline points="260,119.5 266,119 272,118.1 278,116.6 284,114.3 290,111.1 296,106.7 302,101.2 308,94.8 314,88.1 320,81.7 326,76.4 332,73.1 338,72.2 344,73.8 350,77.8 356,83.4 362,90 368,96.7 374,102.9 380,108.1 386,112.1 392,115.1 398,117.1 404,118.4 410,119.1 416,119.6 422,119.8 428,119.9 434,120 440,120"/>
    <polyline points="475,119.5 481,119 487,118.1 493,116.6 499,114.3 505,111.1 511,106.7 517,101.2 523,94.8 529,88.1 535,81.7 541,76.4 547,73.1 553,72.2 559,73.8 565,77.8 571,83.4 577,90 583,96.7 589,102.9 595,108.1 601,112.1 607,115.1 613,117.1 619,118.4 625,119.1 631,119.6 637,119.8 643,119.9 649,120 655,120"/>
  </g>
  <g font-size="12" fill="currentColor" font-family="sans-serif" text-anchor="middle">
    <text x="135" y="22">n = 1</text><text x="350" y="22">n = 5</text><text x="565" y="22">n = 30</text>
    <text x="135" y="136">−3        0        +3</text><text x="350" y="136">−3        0        +3</text><text x="565" y="136">−3        0        +3</text>
  </g>
  <g font-size="11" font-family="sans-serif"><text x="45" y="152" fill="#3b82f6">density of standardized sum</text><text x="300" y="152" fill="#ef4444">N(0,1) reference</text></g>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="184">Figure 7.4. The CLT in action for an Exponential(1) parent, using the EXACT law of the standardized sum</text>
    <text x="20" y="203">(a Gamma(n,1) rescaled), not a simulation. At n = 1 the density is one-sided and maximally skewed; the</text>
    <text x="20" y="222">standardized variable cannot go below −√n = −1. At n = 5 the left edge has retreated to −√5 and the shape is</text>
    <text x="20" y="241">already roughly bell-like but visibly right-skewed. At n = 30 the two curves nearly coincide near the centre;</text>
    <text x="20" y="260">the remaining discrepancy lives in the tails, where the exact upper tail is about 3× the normal one at z = 3</text>
    <text x="20" y="279">(Worked Example 7.19). Convergence happens in the middle first and at the edges last.</text>
  </g>
</svg>

Figure 7.4 tracked one parent across $n$; Figure 7.5 fixes the same three values of $n$ and compares the three parents of Worked Example 7.20 side by side, CDF against CDF, with every curve computed exactly. It is the single picture to hold in mind when someone says "$n=30$ is enough": *which* parent, not *which* $n$, is the question.

<svg viewBox="0 0 700 605" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
<g stroke="currentColor" stroke-width="1" fill="none"><line x1="70" y1="150" x2="250" y2="150"/><line x1="70" y1="30" x2="70" y2="150"/><line x1="285" y1="150" x2="465" y2="150"/><line x1="285" y1="30" x2="285" y2="150"/><line x1="500" y1="150" x2="680" y2="150"/><line x1="500" y1="30" x2="500" y2="150"/><line x1="70" y1="310" x2="250" y2="310"/><line x1="70" y1="190" x2="70" y2="310"/><line x1="285" y1="310" x2="465" y2="310"/><line x1="285" y1="190" x2="285" y2="310"/><line x1="500" y1="310" x2="680" y2="310"/><line x1="500" y1="190" x2="500" y2="310"/><line x1="70" y1="470" x2="250" y2="470"/><line x1="70" y1="350" x2="70" y2="470"/><line x1="285" y1="470" x2="465" y2="470"/><line x1="285" y1="350" x2="285" y2="470"/><line x1="500" y1="470" x2="680" y2="470"/><line x1="500" y1="350" x2="500" y2="470"/></g>
<g fill="none" stroke="#ef4444" stroke-width="1.3" stroke-dasharray="5 4"><polyline points="70.0,150.0 75.6,149.9 81.2,149.9 86.9,149.7 92.5,149.5 98.1,149.0 103.8,148.3 109.4,147.1 115.0,145.2 120.6,142.5 126.2,138.6 131.9,133.6 137.5,127.1 143.1,119.3 148.8,110.3 154.4,100.4 160.0,90.0 165.6,79.6 171.2,69.7 176.9,60.7 182.5,52.9 188.1,46.4 193.8,41.4 199.4,37.5 205.0,34.8 210.6,32.9 216.2,31.7 221.9,31.0 227.5,30.5 233.1,30.3 238.8,30.1 244.4,30.1 250.0,30.0"/><polyline points="285.0,150.0 290.6,149.9 296.2,149.9 301.9,149.7 307.5,149.5 313.1,149.0 318.8,148.3 324.4,147.1 330.0,145.2 335.6,142.5 341.2,138.6 346.9,133.6 352.5,127.1 358.1,119.3 363.8,110.3 369.4,100.4 375.0,90.0 380.6,79.6 386.2,69.7 391.9,60.7 397.5,52.9 403.1,46.4 408.8,41.4 414.4,37.5 420.0,34.8 425.6,32.9 431.2,31.7 436.9,31.0 442.5,30.5 448.1,30.3 453.8,30.1 459.4,30.1 465.0,30.0"/><polyline points="500.0,150.0 505.6,149.9 511.2,149.9 516.9,149.7 522.5,149.5 528.1,149.0 533.8,148.3 539.4,147.1 545.0,145.2 550.6,142.5 556.2,138.6 561.9,133.6 567.5,127.1 573.1,119.3 578.8,110.3 584.4,100.4 590.0,90.0 595.6,79.6 601.2,69.7 606.9,60.7 612.5,52.9 618.1,46.4 623.8,41.4 629.4,37.5 635.0,34.8 640.6,32.9 646.2,31.7 651.9,31.0 657.5,30.5 663.1,30.3 668.8,30.1 674.4,30.1 680.0,30.0"/><polyline points="70.0,310.0 75.6,309.9 81.2,309.9 86.9,309.7 92.5,309.5 98.1,309.0 103.8,308.3 109.4,307.1 115.0,305.2 120.6,302.5 126.2,298.6 131.9,293.6 137.5,287.1 143.1,279.3 148.8,270.3 154.4,260.4 160.0,250.0 165.6,239.6 171.2,229.7 176.9,220.7 182.5,212.9 188.1,206.4 193.8,201.4 199.4,197.5 205.0,194.8 210.6,192.9 216.2,191.7 221.9,191.0 227.5,190.5 233.1,190.3 238.8,190.1 244.4,190.1 250.0,190.0"/><polyline points="285.0,310.0 290.6,309.9 296.2,309.9 301.9,309.7 307.5,309.5 313.1,309.0 318.8,308.3 324.4,307.1 330.0,305.2 335.6,302.5 341.2,298.6 346.9,293.6 352.5,287.1 358.1,279.3 363.8,270.3 369.4,260.4 375.0,250.0 380.6,239.6 386.2,229.7 391.9,220.7 397.5,212.9 403.1,206.4 408.8,201.4 414.4,197.5 420.0,194.8 425.6,192.9 431.2,191.7 436.9,191.0 442.5,190.5 448.1,190.3 453.8,190.1 459.4,190.1 465.0,190.0"/><polyline points="500.0,310.0 505.6,309.9 511.2,309.9 516.9,309.7 522.5,309.5 528.1,309.0 533.8,308.3 539.4,307.1 545.0,305.2 550.6,302.5 556.2,298.6 561.9,293.6 567.5,287.1 573.1,279.3 578.8,270.3 584.4,260.4 590.0,250.0 595.6,239.6 601.2,229.7 606.9,220.7 612.5,212.9 618.1,206.4 623.8,201.4 629.4,197.5 635.0,194.8 640.6,192.9 646.2,191.7 651.9,191.0 657.5,190.5 663.1,190.3 668.8,190.1 674.4,190.1 680.0,190.0"/><polyline points="70.0,470.0 75.6,469.9 81.2,469.9 86.9,469.7 92.5,469.5 98.1,469.0 103.8,468.3 109.4,467.1 115.0,465.2 120.6,462.5 126.2,458.6 131.9,453.6 137.5,447.1 143.1,439.3 148.8,430.3 154.4,420.4 160.0,410.0 165.6,399.6 171.2,389.7 176.9,380.7 182.5,372.9 188.1,366.4 193.8,361.4 199.4,357.5 205.0,354.8 210.6,352.9 216.2,351.7 221.9,351.0 227.5,350.5 233.1,350.3 238.8,350.1 244.4,350.1 250.0,350.0"/><polyline points="285.0,470.0 290.6,469.9 296.2,469.9 301.9,469.7 307.5,469.5 313.1,469.0 318.8,468.3 324.4,467.1 330.0,465.2 335.6,462.5 341.2,458.6 346.9,453.6 352.5,447.1 358.1,439.3 363.8,430.3 369.4,420.4 375.0,410.0 380.6,399.6 386.2,389.7 391.9,380.7 397.5,372.9 403.1,366.4 408.8,361.4 414.4,357.5 420.0,354.8 425.6,352.9 431.2,351.7 436.9,351.0 442.5,350.5 448.1,350.3 453.8,350.1 459.4,350.1 465.0,350.0"/><polyline points="500.0,470.0 505.6,469.9 511.2,469.9 516.9,469.7 522.5,469.5 528.1,469.0 533.8,468.3 539.4,467.1 545.0,465.2 550.6,462.5 556.2,458.6 561.9,453.6 567.5,447.1 573.1,439.3 578.8,430.3 584.4,420.4 590.0,410.0 595.6,399.6 601.2,389.7 606.9,380.7 612.5,372.9 618.1,366.4 623.8,361.4 629.4,357.5 635.0,354.8 640.6,352.9 646.2,351.7 651.9,351.0 657.5,350.5 663.1,350.3 668.8,350.1 674.4,350.1 680.0,350.0"/></g>
<g fill="none" stroke="#3b82f6" stroke-width="1.8"><polyline points="70.0,150.0 157.4,150.0 157.4,31.2 250.0,31.2"/><polyline points="285.0,150.0 369.2,150.0 369.2,35.9 465.0,35.9"/><polyline points="500.0,150.0 575.8,150.0 575.8,61.2 623.0,61.2 623.0,34.3 670.2,34.3 670.2,30.4 680.0,30.4"/><polyline points="70.0,310.0 75.6,310.0 81.2,310.0 86.9,310.0 92.5,310.0 98.1,310.0 103.8,310.0 109.4,310.0 115.0,310.0 120.6,310.0 126.2,310.0 131.9,310.0 137.5,295.9 143.1,275.1 148.8,258.4 154.4,244.9 160.0,234.1 165.6,225.5 171.2,218.5 176.9,212.9 182.5,208.4 188.1,204.8 193.8,201.9 199.4,199.5 205.0,197.7 210.6,196.2 216.2,195.0 221.9,194.0 227.5,193.2 233.1,192.6 238.8,192.1 244.4,191.7 250.0,191.3"/><polyline points="285.0,310.0 290.6,310.0 296.2,310.0 301.9,310.0 307.5,310.0 313.1,310.0 318.8,310.0 324.4,310.0 330.0,309.4 335.6,307.3 341.2,303.0 346.9,296.1 352.5,286.9 358.1,276.3 363.8,265.0 369.4,253.6 375.0,242.9 380.6,233.1 386.2,224.6 391.9,217.3 397.5,211.2 403.1,206.3 408.8,202.4 414.4,199.3 420.0,197.0 425.6,195.1 431.2,193.8 436.9,192.7 442.5,192.0 448.1,191.4 453.8,191.0 459.4,190.7 465.0,190.5"/><polyline points="500.0,310.0 505.6,310.0 511.2,310.0 516.9,310.0 522.5,309.9 528.1,309.7 533.8,309.3 539.4,308.4 545.0,306.7 550.6,303.9 556.2,299.8 561.9,294.1 567.5,286.8 573.1,278.0 578.8,268.1 584.4,257.7 590.0,247.1 595.6,236.9 601.2,227.6 606.9,219.3 612.5,212.3 618.1,206.6 623.8,202.0 629.4,198.5 635.0,195.9 640.6,194.0 646.2,192.7 651.9,191.7 657.5,191.1 663.1,190.7 668.8,190.4 674.4,190.3 680.0,190.2"/><polyline points="70.0,469.5 75.6,469.4 81.2,469.2 86.9,469.0 92.5,468.8 98.1,468.5 103.8,468.1 109.4,467.5 115.0,466.6 120.6,465.4 126.2,463.5 131.9,460.7 137.5,456.4 143.1,449.7 148.8,439.8 154.4,426.2 160.0,410.0 165.6,393.8 171.2,380.2 176.9,370.3 182.5,363.6 188.1,359.3 193.8,356.5 199.4,354.6 205.0,353.4 210.6,352.5 216.2,351.9 221.9,351.5 227.5,351.2 233.1,351.0 238.8,350.8 244.4,350.6 250.0,350.5"/><polyline points="285.0,469.6 290.6,469.5 296.2,469.4 301.9,469.2 307.5,469.0 313.1,468.6 318.8,468.1 324.4,467.3 330.0,466.1 335.6,464.4 341.2,461.7 346.9,457.8 352.5,452.2 358.1,444.6 363.8,434.7 369.4,422.9 375.0,410.0 380.6,397.1 386.2,385.3 391.9,375.4 397.5,367.8 403.1,362.2 408.8,358.3 414.4,355.6 420.0,353.9 425.6,352.7 431.2,351.9 436.9,351.4 442.5,351.0 448.1,350.8 453.8,350.6 459.4,350.5 465.0,350.4"/><polyline points="500.0,469.8 505.6,469.7 511.2,469.6 516.9,469.5 522.5,469.2 528.1,468.8 533.8,468.2 539.4,467.2 545.0,465.7 550.6,463.5 556.2,460.2 561.9,455.6 567.5,449.4 573.1,441.5 578.8,432.1 584.4,421.4 590.0,410.0 595.6,398.6 601.2,387.9 606.9,378.5 612.5,370.6 618.1,364.4 623.8,359.8 629.4,356.5 635.0,354.3 640.6,352.8 646.2,351.8 651.9,351.2 657.5,350.8 663.1,350.5 668.8,350.4 674.4,350.3 680.0,350.2"/></g>
<g font-size="11" fill="currentColor" font-family="sans-serif" text-anchor="middle"><text x="160.0" y="20">n = 1</text><text x="375.0" y="20">n = 5</text><text x="590.0" y="20">n = 30</text><text x="32" y="90.0" transform="rotate(-90 32 90.0)">Bernoulli(0.01)</text><text x="32" y="250.0" transform="rotate(-90 32 250.0)">Exponential(1)</text><text x="32" y="410.0" transform="rotate(-90 32 410.0)">t₃</text><text x="70" y="164" font-size="9">−3.5</text><text x="250" y="164" font-size="9">+3.5</text><text x="285" y="164" font-size="9">−3.5</text><text x="465" y="164" font-size="9">+3.5</text><text x="500" y="164" font-size="9">−3.5</text><text x="680" y="164" font-size="9">+3.5</text><text x="70" y="324" font-size="9">−3.5</text><text x="250" y="324" font-size="9">+3.5</text><text x="285" y="324" font-size="9">−3.5</text><text x="465" y="324" font-size="9">+3.5</text><text x="500" y="324" font-size="9">−3.5</text><text x="680" y="324" font-size="9">+3.5</text><text x="70" y="484" font-size="9">−3.5</text><text x="250" y="484" font-size="9">+3.5</text><text x="285" y="484" font-size="9">−3.5</text><text x="465" y="484" font-size="9">+3.5</text><text x="500" y="484" font-size="9">−3.5</text><text x="680" y="484" font-size="9">+3.5</text></g>
<g font-size="11" font-family="sans-serif"><text x="70" y="500" fill="#3b82f6">exact CDF of Z_n</text><text x="220" y="500" fill="#ef4444">Φ (standard normal)</text></g>
<g fill="currentColor" font-size="12.5" font-family="sans-serif"><text x="20" y="516">Figure 7.5. CLT convergence for three parents at n = 1, 5, 30: exact CDF of the standardized sum Z_n (blue)</text><text x="20" y="535">against Φ (red dashed). No simulation: binomial, Gamma, and Gil–Pelaez inversion of the t₃ characteristic</text><text x="20" y="554">function. Exponential and t₃ are visually Gaussian by n = 30; Bernoulli(0.01) is still a two-step staircase,</text><text x="20" y="573">because S₃₀ is 0 or 1 with probability 0.96 — the sup-distance is 0.4487 (Worked Example 7.20). The lattice</text><text x="20" y="592">structure, not the tail weight, is what resists: t₃ has infinite third moment yet converges fastest here.</text></g></svg>

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="60" y1="200" x2="580" y2="200"/><line x1="60" y1="200" x2="60" y2="20"/>
  </g>
  <polyline fill="none" stroke="#ef4444" stroke-width="2" points="60,35.5 98,35.2 147,35.1 185,35.6 245,39.6 310,55.3 435,80.8 560,108.6"/>
  <polyline fill="none" stroke="#f59e0b" stroke-width="2" points="60,46.3 98,53.9 147,62.9 185,71.2 245,84.2 310,98.8 435,126.9 560,154.9"/>
  <polyline fill="none" stroke="#3b82f6" stroke-width="2" points="60,65 98,77.6 147,88.9 185,97.4 245,110.8 310,125.5 435,153.7 560,182.3"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2" points="60,79.7 98,86.5 147,96.2 185,103.9 245,116.5 310,130.7 435,158.5 560,186.4"/>
  <line x1="60" y1="32.5" x2="560" y2="145" stroke="currentColor" stroke-width="1" stroke-dasharray="4 4"/>
  <g font-size="11" fill="currentColor" font-family="sans-serif">
    <text x="54" y="24" text-anchor="end">1</text><text x="54" y="80" text-anchor="end">0.1</text>
    <text x="54" y="136" text-anchor="end">0.01</text><text x="54" y="192" text-anchor="end">0.001</text>
    <text x="60" y="216" text-anchor="middle">1</text><text x="185" y="216" text-anchor="middle">10</text>
    <text x="310" y="216" text-anchor="middle">100</text><text x="435" y="216" text-anchor="middle">1000</text><text x="560" y="216" text-anchor="middle">10⁴</text>
    <text x="320" y="232" text-anchor="middle">n</text>
    <text x="20" y="112" text-anchor="start" transform="rotate(-90 20 112)">sup_x |P(Z_n ≤ x) − Φ(x)|</text>
  </g>
  <g font-size="11" font-family="sans-serif">
    <text x="592" y="112" fill="#ef4444">Bern(0.01)</text><text x="592" y="158" fill="#f59e0b">Bern(0.5)</text>
    <text x="592" y="186" fill="#3b82f6">Exp(1)</text><text x="592" y="190" fill="#10b981"> </text>
    <text x="592" y="204" fill="#10b981">t₃</text>
    <text x="470" y="148" fill="currentColor">slope −1/2</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="258">Figure 7.6. Exact Kolmogorov distance to the standard normal, log–log. All four parents eventually run</text>
    <text x="20" y="277">parallel to the dashed reference line of slope −1/2, confirming the Berry–Esseen rate. What differs by two</text>
    <text x="20" y="296">orders of magnitude is the INTERCEPT: Bernoulli(0.01) at n = 10⁴ is still less accurate than Exp(1) at n = 30.</text>
    <text x="20" y="315">"n ≥ 30" is a statement about a constant nobody checked, not about a rate.</text>
  </g>
</svg>

**Worked Example 7.21 (how many samples? three answers).** Suppose $X_i\in[0,1]$ i.i.d. and you want $\mathbb{P}(|\bar X_n-\mu|>0.01)\le0.05$. Three tools give three answers.

- *Chebyshev* (Theorem 7.16), using the worst-case variance $\sigma^2\le1/4$: need $\sigma^2/(n\varepsilon^2)\le0.05$, i.e. $n\ge0.25/(0.05\times10^{-4})=50{,}000$.
- *Hoeffding* (Ch. 4): need $2e^{-2n\varepsilon^2}\le0.05$, i.e. $n\ge\ln(40)/(2\times10^{-4})=18{,}445$.
- *CLT* (Theorem 7.31), treating the approximation as exact: need $1.96\sigma/\sqrt n\le0.01$ with $\sigma=1/2$, i.e. $n\ge(1.96\times0.5/0.01)^2=9{,}604$.

The ordering is instructive and typical: Chebyshev is a valid bound and a factor of five too conservative; Hoeffding is valid and about twice too conservative; the CLT number is the smallest but is an *approximation*, not a bound — Berry–Esseen tells you the error in that approximation is at most $0.4748\rho/(\sigma^3\sqrt{9604})\le0.0048\,\rho/\sigma^3$, which for a Bernoulli$(1/2)$ parent ($\rho/\sigma^3=1$) is a negligible $0.005$ but for a Bernoulli$(0.01)$ parent ($\rho/\sigma^3=9.85$) is $0.048$, comparable to the $0.05$ you were trying to control. Use the CLT number when the parent is well behaved and the bound when it is not.

> **Common Confusion 7.8.**
> *"The CLT says the sample mean is approximately normal for $n\ge30$."*
>
> The CLT says a limit holds as $n\to\infty$. It says nothing about any particular $n$; the finite-$n$ statement is Berry–Esseen, and its constant is $\rho/\sigma^3$, which the "$n\ge30$" rule never mentions. Worked Example 7.18 exhibits a parent for which $n=10^4$ is still not enough. Worked Example 7.19 shows that even for a mild parent the tails lag the centre badly. Three practical diagnostics, in increasing order of effort: (a) compute $n\hat p(1-\hat p)$ for binary data and demand $\ge10$; (b) compute the sample standardized third moment $\hat\rho/\hat\sigma^3$ and plug into Berry–Esseen; (c) bootstrap (Ch. 9) and look at the resampling distribution directly. Also note the CLT is a statement about the *sample mean*. For a maximum, a minimum, or an extreme quantile the limit theory is completely different (extreme value theory), and the normal approximation is not merely inaccurate but qualitatively wrong.

> **Self-check 7.5.**
> 1. Where in the proof of Theorem 7.31 is finiteness of the variance used, and what breaks for a Cauchy parent?
> 2. If $\rho/\sigma^3=5$, how large must $n$ be for Berry–Esseen to guarantee a sup-error below $0.01$?
> 3. State the Lindeberg condition in one sentence of plain English.
> 4. Why does the normal approximation understate $\mathbb{P}(Z_n>3)$ for a right-skewed parent?

---

## 7.6 The multivariate CLT and the delta method

### 7.6.1 From one dimension to $d$

Characteristic functions make the multivariate CLT nearly free, because of a device that converts any $d$-dimensional distributional question into a family of one-dimensional ones.

**Theorem 7.35 (Cramér–Wold device).** Let $X_n,X$ be random vectors in $\mathbb{R}^d$. Then
$$X_n\to_{\mathrm d}X\quad\Longleftrightarrow\quad \langle u,X_n\rangle\to_{\mathrm d}\langle u,X\rangle\ \text{ for every }u\in\mathbb{R}^d .$$

*Proof.* ($\Rightarrow$) Directly: for each fixed $t\in\mathbb{R}$, $x\mapsto e^{it\langle u,x\rangle}$ is a bounded continuous function on $\mathbb{R}^d$, so the test-function form of convergence in distribution gives $\varphi_{\langle u,X_n\rangle}(t)\to\varphi_{\langle u,X\rangle}(t)$, and Lévy continuity in one dimension (Theorem 7.30) finishes it. (More generally this is the continuous mapping theorem, Theorem 7.39, whose proof in §7.7 does not use the present result — no circularity.)
($\Leftarrow$) Suppose $\langle u,X_n\rangle\to_{\mathrm d}\langle u,X\rangle$ for each $u$. By Theorem 7.30(1) applied in dimension one at the point $t=1$,
$$\varphi_{X_n}(u)=\mathbb{E}\big[e^{i\langle u,X_n\rangle}\big]=\varphi_{\langle u,X_n\rangle}(1)\longrightarrow\varphi_{\langle u,X\rangle}(1)=\varphi_X(u)$$
for every $u\in\mathbb{R}^d$. The limit $\varphi_X$ is a characteristic function, continuous at $0$, so the multivariate Lévy continuity theorem gives $X_n\to_{\mathrm d}X$. $\square$

The device is memorable as a slogan: *a random vector's law is determined by the laws of all its one-dimensional shadows.* This is not obvious — it is false, for instance, that a vector's law is determined by its $d$ coordinate marginals — and it is exactly the statement that characteristic functions on $\mathbb{R}^d$ are built from linear functionals.

**Theorem 7.36 (multivariate CLT).** Let $X_1,X_2,\dots$ be i.i.d. random vectors in $\mathbb{R}^d$ with $\mathbb{E}\|X_1\|^2<\infty$, mean $\mu:=\mathbb{E}[X_1]$ and covariance $\Sigma:=\operatorname{Cov}(X_1)$. Then
$$\sqrt n\,(\bar X_n-\mu)\ \longrightarrow_{\mathrm d}\ \mathcal N(0,\Sigma).$$

*Proof.* Fix $u\in\mathbb{R}^d$ and set $W_i:=\langle u,X_i-\mu\rangle$, i.i.d. real variables with $\mathbb{E}[W_1]=0$ and $\operatorname{Var}(W_1)=u^\top\Sigma u<\infty$. If $u^\top\Sigma u>0$, Theorem 7.31 gives $n^{-1/2}\sum_i W_i\to_{\mathrm d}\mathcal N(0,u^\top\Sigma u)$. If $u^\top\Sigma u=0$ then $W_1=0$ a.s. and the left side is identically $0=\mathcal N(0,0)$, so the conclusion holds trivially. But $\langle u,\sqrt n(\bar X_n-\mu)\rangle=n^{-1/2}\sum_iW_i$, and by Chapter 5's characterization of the multivariate Gaussian, $\mathcal N(0,u^\top\Sigma u)$ is exactly the law of $\langle u,G\rangle$ for $G\sim\mathcal N(0,\Sigma)$. Theorem 7.35 finishes it. $\square$

Note the hypothesis $\mathbb{E}\|X_1\|^2<\infty$ is equivalent to each coordinate having a finite second moment, since $\|x\|^2=\sum_jx_j^2$. No density, no invertibility of $\Sigma$: a singular $\Sigma$ is fine and produces a degenerate Gaussian supported on a proper subspace — which, as Chapter 1 pointed out for generator pushforwards, has no Lebesgue density in $\mathbb{R}^d$. The CLT does not care.

### 7.6.2 Vector-valued convergence, and the notation $O_p$ and $o_p$

First, a definitional point this book owes you before using estimators $\hat\theta_n\in\mathbb{R}^d$. For random *vectors*, almost sure convergence and convergence in probability are Definitions 7.1 and 7.2 verbatim with $|\cdot|$ replaced by the Euclidean norm $\|\cdot\|$; likewise $L^p$ convergence. Everything in §§7.1–7.2 transfers unchanged, because $\|X_n-X\|$ is a scalar random variable (a continuous function of a random vector, measurable by Proposition 1.15), and because $\|X_n-X\|\to0$ iff every coordinate converges (the norms $\max_j|x_j|$ and $\|x\|$ are equivalent). Convergence in distribution for vectors is the test-function formulation of Definition 7.4 — $\mathbb{E}[f(X_n)]\to\mathbb{E}[f(X)]$ for all bounded continuous $f:\mathbb{R}^d\to\mathbb{R}$ — which is exactly the form the Portmanteau theorem (Theorem 7.5) was already stated in; the one thing that does *not* transfer is the CDF formulation, which becomes awkward in $d>1$ and is best forgotten.

Two abbreviations make the next proofs readable. Write $R_n=o_p(1)$ to mean $R_n\to_{\mathrm p}0$, and $T_n=O_p(1)$ to mean the sequence is *bounded in probability* (tight): for every $\eta>0$ there is $M<\infty$ with $\sup_n\mathbb{P}(\|T_n\|>M)\le\eta$. Two facts we will use:

- If $T_n\to_{\mathrm d}T$ then $T_n=O_p(1)$. (Choose $M$ a continuity point of the law of $\|T\|$ with $\mathbb{P}(\|T\|>M)\le\eta/2$; then $\mathbb{P}(\|T_n\|>M)\to\mathbb{P}(\|T\|>M)$, so it is $\le\eta$ for large $n$, and enlarge $M$ to cover the finitely many small $n$.)
- $o_p(1)\cdot O_p(1)=o_p(1)$. (For $\varepsilon>0$ and $M$ as above, $\mathbb{P}(|R_nT_n|>\varepsilon)\le\mathbb{P}(\|T_n\|>M)+\mathbb{P}(|R_n|>\varepsilon/M)\le\eta+o(1)$; let $\eta\downarrow0$.)

These are exactly the two ingredients Slutsky's theorem packages, and I prove them here so that §7.6 does not depend circularly on §7.7.

### 7.6.3 The delta method

The situation: you have an estimator $\hat\theta_n\in\mathbb{R}^d$ that is asymptotically normal, and you actually care about $g(\hat\theta_n)$ for some smooth $g$ — a ratio, a log-odds, an $R^2$, a difference of transformed rates. The delta method says: linearize $g$ at the truth, and the limit is whatever the linearization does to the Gaussian.

**Theorem 7.37 (delta method).** Let $\hat\theta_n$ be random vectors in $\mathbb{R}^d$, $\theta\in\mathbb{R}^d$ fixed, and $a_n\to\infty$ a deterministic sequence with
$$a_n(\hat\theta_n-\theta)\ \longrightarrow_{\mathrm d}\ T .$$
Let $g:\mathbb{R}^d\to\mathbb{R}^m$ be differentiable at $\theta$, with Jacobian $Dg(\theta)\in\mathbb{R}^{m\times d}$. Then
$$a_n\big(g(\hat\theta_n)-g(\theta)\big)\ \longrightarrow_{\mathrm d}\ Dg(\theta)\,T .$$
In particular, if $a_n=\sqrt n$ and $T\sim\mathcal N(0,\Sigma)$, then $\sqrt n(g(\hat\theta_n)-g(\theta))\to_{\mathrm d}\mathcal N\big(0,\,Dg(\theta)\Sigma Dg(\theta)^\top\big)$, which for scalar $g$ reads
$$\sqrt n\big(g(\hat\theta_n)-g(\theta)\big)\ \longrightarrow_{\mathrm d}\ \mathcal N\big(0,\ \nabla g(\theta)^\top\Sigma\,\nabla g(\theta)\big).$$

*Proof.* Write $T_n:=a_n(\hat\theta_n-\theta)$, so $T_n\to_{\mathrm d}T$ and hence $T_n=O_p(1)$ by §7.6.2. Since $a_n\to\infty$,
$$\hat\theta_n-\theta=T_n/a_n=o_p(1)\cdot O_p(1)=o_p(1),$$
so $\hat\theta_n\to_{\mathrm p}\theta$: the estimator is consistent.

Differentiability at $\theta$ means
$$g(\theta+h)=g(\theta)+Dg(\theta)h+\|h\|\,\psi(h),\qquad \psi(h)\to0\ \text{ as }h\to0,\ \psi(0):=0,$$
which defines $\psi:\mathbb{R}^d\to\mathbb{R}^m$ continuous at $0$. Substituting $h=\hat\theta_n-\theta$ and multiplying by $a_n$,
$$a_n\big(g(\hat\theta_n)-g(\theta)\big)=Dg(\theta)\,T_n+\|T_n\|\,\psi(\hat\theta_n-\theta).$$
For the second term: $\hat\theta_n-\theta\to_{\mathrm p}0$ and $\psi$ is continuous at $0$ with $\psi(0)=0$, so $\psi(\hat\theta_n-\theta)\to_{\mathrm p}0$ (by the continuous-mapping consequence of Theorem 7.15 — pass to a sub-subsequence along which $\hat\theta_n\to\theta$ a.s., where continuity applies pointwise). Since $\|T_n\|=O_p(1)$, the product is $o_p(1)$.

For the first term: $Dg(\theta)T_n\to_{\mathrm d}Dg(\theta)T$ by characteristic functions, self-contained: $\varphi_{Dg(\theta)T_n}(u)=\varphi_{T_n}(Dg(\theta)^\top u)\to\varphi_{T}(Dg(\theta)^\top u)=\varphi_{Dg(\theta)T}(u)$, and Lévy continuity (Theorem 7.30) applies. (Equivalently, linear maps are continuous, so Theorem 7.39 — proved independently in §7.7 — also gives it.)

Finally, a distributional limit plus an $o_p(1)$ perturbation has the same limit: if $A_n\to_{\mathrm d}A$ and $B_n\to_{\mathrm p}0$ then $A_n+B_n\to_{\mathrm d}A$. (Proof: for a continuity point $x$ of $F_A$ and $\varepsilon>0$, $\mathbb{P}(A_n+B_n\le x)\le\mathbb{P}(A_n\le x+\varepsilon)+\mathbb{P}(|B_n|>\varepsilon)$, and symmetrically; take $\limsup$, then $\varepsilon\downarrow0$ using continuity of $F_A$ at $x$. This is the scalar case of Theorem 7.40.) The linear-image statement for $\mathcal N(0,\Sigma)$ is Chapter 5's affine-transformation rule. $\square$

Read the conclusion geometrically. $\nabla g(\theta)^\top\Sigma\nabla g(\theta)$ is the variance of the Gaussian *projected onto the direction of steepest ascent of $g$*, weighted by how steep that ascent is. Where $g$ is flat, the estimator's noise is compressed; where $g$ is steep, it is amplified. That is the whole content, and Figure 7.7 draws it.

<svg viewBox="0 0 700 340" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="70" y1="200" x2="310" y2="200"/><line x1="70" y1="200" x2="70" y2="30"/>
    <line x1="400" y1="200" x2="640" y2="200"/><line x1="400" y1="200" x2="400" y2="30"/>
  </g>
  <path d="M80 190 Q160 175 200 120 Q240 65 300 45" fill="none" stroke="#3b82f6" stroke-width="2"/>
  <line x1="140" y1="168" x2="260" y2="76" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="5 4"/>
  <line x1="200" y1="200" x2="200" y2="120" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3 3"/>
  <line x1="70" y1="120" x2="200" y2="120" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3 3"/>
  <path d="M160 200 Q200 152 240 200" fill="#10b981" opacity="0.35" stroke="none"/>
  <path d="M70 88 Q118 120 70 152" fill="#ef4444" opacity="0.35" stroke="none"/>
  <path d="M430 190 Q470 60 500 60 Q530 60 570 190" fill="none" stroke="#3b82f6" stroke-width="2"/>
  <line x1="440" y1="60" x2="560" y2="60" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="5 4"/>
  <line x1="500" y1="200" x2="500" y2="60" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3 3"/>
  <path d="M460 200 Q500 152 540 200" fill="#10b981" opacity="0.35" stroke="none"/>
  <path d="M400 44 Q436 58 400 74" fill="#ef4444" opacity="0.35" stroke="none"/>
  <g font-size="11" fill="currentColor" font-family="sans-serif">
    <text x="200" y="216" text-anchor="middle">θ</text><text x="500" y="216" text-anchor="middle">θ</text>
    <text x="64" y="124" text-anchor="end">g(θ)</text><text x="394" y="64" text-anchor="end">g(θ)</text>
    <text x="190" y="30" text-anchor="middle" font-size="12">∇g(θ) ≠ 0</text>
    <text x="520" y="30" text-anchor="middle" font-size="12">∇g(θ) = 0</text>
    <text x="316" y="204" text-anchor="start">θ̂</text><text x="646" y="204" text-anchor="start">θ̂</text>
  </g>
  <g font-size="10.5" font-family="sans-serif">
    <text x="248" y="196" fill="#10b981">law of θ̂ₙ</text><text x="548" y="196" fill="#10b981">law of θ̂ₙ</text>
    <text x="90" y="84" fill="#ef4444">law of g(θ̂ₙ)</text><text x="410" y="38" fill="#ef4444">law of g(θ̂ₙ)</text>
    <text x="262" y="72" fill="#f59e0b">tangent</text><text x="566" y="58" fill="#f59e0b">tangent</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="248">Figure 7.7. The delta method is linearization. LEFT: near θ the function g is indistinguishable from its</text>
    <text x="20" y="267">tangent at the n^(−1/2) scale on which θ̂ₙ fluctuates, so the (green) Gaussian law of θ̂ₙ is mapped to a (red)</text>
    <text x="20" y="286">Gaussian scaled by the slope: variance ∇gᵀΣ∇g. RIGHT: when ∇g(θ) = 0 the tangent is flat, the first-order</text>
    <text x="20" y="305">term vanishes, and the FOLD becomes visible — both tails of θ̂ₙ map to the same side of g(θ). The limit is</text>
    <text x="20" y="324">then a chi-square-type law at rate 1/n, not a Gaussian at rate 1/√n (Theorem 7.38).</text>
  </g>
</svg>

**Worked Example 7.22 (variance of a ratio).** Let $(X_i,Y_i)$ be i.i.d. in $\mathbb{R}^2$ with means $\mu_x\ne0$, $\mu_y$, and covariance $\Sigma=\begin{psmallmatrix}\sigma_x^2&\sigma_{xy}\\\sigma_{xy}&\sigma_y^2\end{psmallmatrix}$. Estimate $R:=\mu_y/\mu_x$ by $\hat R_n=\bar Y_n/\bar X_n$. Take $g(a,b)=b/a$, so
$$\nabla g(\mu_x,\mu_y)=\Big(-\frac{\mu_y}{\mu_x^2},\ \frac{1}{\mu_x}\Big)^\top,$$
and Theorems 7.36 and 7.37 give $\sqrt n(\hat R_n-R)\to_{\mathrm d}\mathcal N(0,\tau^2)$ with
$$\tau^2=\nabla g^\top\Sigma\nabla g=\frac{\mu_y^2\sigma_x^2}{\mu_x^4}-\frac{2\mu_y\sigma_{xy}}{\mu_x^3}+\frac{\sigma_y^2}{\mu_x^2}=\frac{1}{\mu_x^2}\Big(\sigma_y^2-2R\,\sigma_{xy}+R^2\sigma_x^2\Big).$$
That last form is worth memorizing: the asymptotic variance of a ratio is the variance of the residual $Y-RX$, divided by $\mu_x^2$. Numerator and denominator errors partially cancel when they are positively correlated, which is why paired designs help.

A numerical check. Let $X\sim\mathrm{Gamma}(4,\text{scale }1/2)$ so $\mu_x=2$, $\sigma_x^2=1$; and $Y=3X+\epsilon$ with $\epsilon\sim\mathcal N(0,1)$ independent, so $\mu_y=6$, $\sigma_y^2=10$, $\sigma_{xy}=3$, $R=3$. Then $\tau^2=\frac14(10-2\cdot3\cdot3+9\cdot1)=\frac14(10-18+9)=0.25$. Simulating $M=200{,}000$ datasets of size $n=200$ (numpy PCG64, seed 31415) gives $n\widehat{\operatorname{Var}}(\hat R_n)=0.250603$ and mean $\hat R_n=3.00007$. Agreement to three decimals.

The warning attached: the delta method is an *asymptotic* statement and $\hat R_n$ has no finite mean at all here for small $n$ (indeed $\bar X_n$ can be near zero when $\mu_x$ is small relative to $\sigma_x/\sqrt n$). If $|\mu_x|/(\sigma_x/\sqrt n)$ is not comfortably large — say below $5$ — the ratio behaves Cauchy-like and $\tau^2$ is meaningless. This is exactly the failure mode of inverse-propensity-weighted estimators with small estimated propensities, and it is why self-normalized and clipped IPW estimators exist.

**Worked Example 7.23 (log-odds, and why we transform).** Let $\hat p_n=S_n/n$ with $S_n\sim\mathrm{Bin}(n,p)$, $0<p<1$. Then $\sqrt n(\hat p_n-p)\to_{\mathrm d}\mathcal N(0,p(1-p))$. For the logit $g(p)=\log\frac{p}{1-p}$ we have $g'(p)=\frac{1}{p(1-p)}$, so
$$\sqrt n\big(\operatorname{logit}\hat p_n-\operatorname{logit}p\big)\ \longrightarrow_{\mathrm d}\ \mathcal N\Big(0,\ \frac{1}{p(1-p)}\Big),$$
since $\big(\frac{1}{p(1-p)}\big)^2p(1-p)=\frac{1}{p(1-p)}$. The standard error of a log-odds is therefore $\widehat{\mathrm{se}}=1/\sqrt{n\hat p(1-\hat p)}=\sqrt{1/S_n+1/(n-S_n)}$ — the formula printed by every logistic regression package. For a log-odds *ratio* comparing two independent groups, the variances add:
$$\widehat{\mathrm{se}}(\log\widehat{\mathrm{OR}})=\sqrt{\frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}},$$
with $a,b,c,d$ the four cell counts of the $2\times2$ table (Exercise C2 derives this).

Simulation check, $M=200{,}000$ replications per cell (numpy PCG64, seed 11235), reporting $\sqrt n\times$ the empirical standard deviation of $\operatorname{logit}\hat p_n$ (replications with $S_n\in\{0,n\}$ discarded, since the logit is then infinite):

| $p$ | $n$ | empirical | theory $1/\sqrt{p(1-p)}$ | ratio | fraction discarded |
|---:|---:|---:|---:|---:|---:|
| $0.1$ | $50$ | $3.7239$ | $3.3333$ | $1.117$ | $0.49\%$ |
| $0.1$ | $200$ | $3.4517$ | $3.3333$ | $1.036$ | $0.00\%$ |
| $0.1$ | $1000$ | $3.3508$ | $3.3333$ | $1.005$ | $0.00\%$ |
| $0.1$ | $10^4$ | $3.3350$ | $3.3333$ | $1.001$ | $0.00\%$ |
| $0.5$ | $50$ | $2.0446$ | $2.0000$ | $1.022$ | $0.00\%$ |
| $0.5$ | $10^4$ | $2.0018$ | $2.0000$ | $1.001$ | $0.00\%$ |

The delta-method variance is accurate to about $2\%$ by $n=200$ at $p=0.5$ and needs roughly $n=1000$ at $p=0.1$ — the same $np(1-p)\gtrsim10$ heuristic as Worked Example 7.18. Note also the discarded fraction: at $p=0.1$, $n=50$, half a percent of samples have $\hat p=0$ and an infinite logit. Asymptotic theory is silent about a quantity that is $+\infty$ with positive probability; in practice one adds the Haldane–Anscombe correction (add $\tfrac12$ to each cell) precisely to avoid this.

**Worked Example 7.24 (correlation, and variance stabilization).** For a bivariate normal sample with correlation $\rho$, one can show $\sqrt n(\hat r_n-\rho)\to_{\mathrm d}\mathcal N(0,(1-\rho^2)^2)$. The asymptotic variance depends on $\rho$, which is inconvenient: the width of your interval depends on the thing you are estimating. Fisher's transform $g(\rho)=\operatorname{arctanh}\rho=\tfrac12\log\frac{1+\rho}{1-\rho}$ has $g'(\rho)=\frac{1}{1-\rho^2}$, so
$$\sqrt n\big(\operatorname{arctanh}\hat r_n-\operatorname{arctanh}\rho\big)\to_{\mathrm d}\mathcal N\Big(0,\ \tfrac{1}{(1-\rho^2)^2}(1-\rho^2)^2\Big)=\mathcal N(0,1).$$
The $\rho$-dependence cancels exactly. This is *variance stabilization*: choose $g$ with $g'=1/\text{(standard deviation function)}$ and the limiting variance becomes constant.

Numbers, $\rho=0.6$, $n=50$, $M=200{,}000$ (numpy PCG64, seed 31415): $\operatorname{sd}(\hat r)=0.09348$ versus the delta-method $(1-\rho^2)/\sqrt n=0.09051$; $\operatorname{sd}(\operatorname{arctanh}\hat r)=0.14592$ versus $1/\sqrt n=0.14142$ and versus Fisher's refined $1/\sqrt{n-3}=0.14586$. Fisher's finite-sample correction is essentially exact. And the payoff is in the shape, not just the scale: the sample skewness of $\hat r$ is $-0.5115$, while that of $\operatorname{arctanh}\hat r$ is $0.0033$. Transforming did not merely stabilize the variance, it removed the skew — so a symmetric normal interval is far more honest on the transformed scale. Build the interval for $\operatorname{arctanh}\rho$ and map the endpoints back with $\tanh$; never build a symmetric interval for $\rho$ directly, which can produce endpoints outside $[-1,1]$.

### 7.6.4 When the gradient vanishes

**Theorem 7.38 (second-order delta method).** Suppose $a_n(\hat\theta_n-\theta)\to_{\mathrm d}T$ in $\mathbb{R}^d$, and $g:\mathbb{R}^d\to\mathbb{R}$ is twice differentiable at $\theta$ with $\nabla g(\theta)=0$ and Hessian $H:=\nabla^2g(\theta)$. Then
$$a_n^2\big(g(\hat\theta_n)-g(\theta)\big)\ \longrightarrow_{\mathrm d}\ \tfrac12\,T^\top H\,T .$$
In particular if $T\sim\mathcal N(0,\Sigma)$ and $d=1$ with $\Sigma=\sigma^2$, the limit is $\tfrac12 g''(\theta)\sigma^2\chi^2_1$.

*Proof.* Second-order differentiability at $\theta$ with $\nabla g(\theta)=0$ gives
$$g(\theta+h)=g(\theta)+\tfrac12h^\top Hh+\|h\|^2\psi(h),\qquad\psi(h)\to0\text{ as }h\to0,\ \psi(0)=0 .$$
Put $T_n=a_n(\hat\theta_n-\theta)$ and $h=\hat\theta_n-\theta$; multiplying by $a_n^2$,
$$a_n^2\big(g(\hat\theta_n)-g(\theta)\big)=\tfrac12T_n^\top HT_n+\|T_n\|^2\psi(\hat\theta_n-\theta).$$
As in Theorem 7.37, $\hat\theta_n\to_{\mathrm p}\theta$, so $\psi(\hat\theta_n-\theta)=o_p(1)$; and $\|T_n\|^2=O_p(1)$ because $T_n=O_p(1)$ and $x\mapsto\|x\|^2$ maps bounded sets to bounded sets. Hence the remainder is $o_p(1)$. The quadratic form $x\mapsto\tfrac12x^\top Hx$ is continuous, so $\tfrac12T_n^\top HT_n\to_{\mathrm d}\tfrac12T^\top HT$ by Theorem 7.39 (proved independently in §7.7). Adding an $o_p(1)$ does not change a distributional limit (proved inside Theorem 7.37). For the last claim, $T=\sigma Z$ with $Z\sim\mathcal N(0,1)$ and $T^2=\sigma^2Z^2\sim\sigma^2\chi^2_1$. $\square$

Three things change when $\nabla g(\theta)=0$. The *rate* improves from $a_n^{-1}$ to $a_n^{-2}$ (typically $1/\sqrt n\to1/n$): a flat $g$ suppresses noise. The *shape* stops being Gaussian and becomes a quadratic form in a Gaussian — a $\chi^2$ when $d=1$, a weighted sum of $\chi^2_1$'s in general. And the limit acquires a *sign*: if $H\prec0$ the limit is non-positive, so $g(\hat\theta_n)$ is systematically below $g(\theta)$. Symmetric confidence intervals are simply wrong in this regime.

**Worked Example 7.25 (the flat point of $p(1-p)$).** Estimate the Bernoulli variance $g(p)=p(1-p)$ by $g(\hat p_n)$. Then $g'(p)=1-2p$, which vanishes at $p=1/2$ — precisely where the variance is maximal. At $p=1/2$: $g''=-2$, $\sigma^2=p(1-p)=1/4$, so Theorem 7.38 gives
$$n\big(\hat p_n(1-\hat p_n)-\tfrac14\big)\ \longrightarrow_{\mathrm d}\ \tfrac12(-2)\cdot\tfrac14\,\chi^2_1=-\tfrac14\chi^2_1 .$$
The limit is *never positive*: the plug-in variance estimate is asymptotically biased downward at the flat point, with limiting mean $-1/4$ and limiting standard deviation $\tfrac14\sqrt2=0.3536$.

Simulation ($M=400{,}000$ per cell, numpy PCG64, seed 11235 continued from Worked Example 7.23):

| $n$ | mean of $T_n$ | sd of $T_n$ | $\mathbb{P}(T_n>-0.05)$ |
|---:|---:|---:|---:|
| $50$ | $-0.2507$ | $0.3502$ | $0.3274$ |
| $200$ | $-0.2491$ | $0.3519$ | $0.3799$ |
| $1000$ | $-0.2502$ | $0.3521$ | $0.3641$ |
| $10^4$ | $-0.2496$ | $0.3525$ | $0.3478$ |
| limit | $-0.2500$ | $0.3536$ | $0.3453$ |

where $T_n:=n(\hat p_n(1-\hat p_n)-1/4)$. Mean and standard deviation lock on immediately; the probability oscillates because $\hat p_n$ is lattice-valued and $T_n$ has atoms whose positions move with $n$ — the same lattice effect that produced the $1/\sqrt{2\pi}$ constant in Worked Example 7.20. First-order delta method would have predicted $\sqrt n(g(\hat p_n)-1/4)\to\mathcal N(0,0)$, i.e. a degenerate limit; that is not wrong, it is just uninformative, and the second-order theorem tells you what to look at instead.

**Worked Example 7.26 ($R^2$ under the null).** In simple linear regression, $R^2=\hat r_n^2$. Applying the delta method to $g(\rho)=\rho^2$ with $g'(\rho)=2\rho$ and Worked Example 7.24's $\sqrt n(\hat r_n-\rho)\to_{\mathrm d}\mathcal N(0,(1-\rho^2)^2)$ gives
$$\sqrt n\,(R^2-\rho^2)\to_{\mathrm d}\mathcal N\big(0,\ 4\rho^2(1-\rho^2)^2\big),\qquad\rho\ne0 .$$
At $\rho=0$ the gradient vanishes and this degenerates. Theorem 7.38 with $g''=2$, $\sigma^2=(1-0)^2=1$ gives
$$n\,R^2\ \longrightarrow_{\mathrm d}\ \tfrac12\cdot2\cdot1\cdot\chi^2_1=\chi^2_1 .$$
Simulation with $\rho=0$, $M=200{,}000$ (numpy PCG64, seed 161803): at $n=50$, $\mathbb{E}[nR^2]=1.0118$ and $\mathbb{P}(nR^2>3.841)=0.0500$; at $n=200$, $\mathbb{E}[nR^2]=1.0032$ and $\mathbb{P}(nR^2>3.841)=0.0503$. The limiting values are $1$ and $0.0500$ (since $3.841=\chi^2_{1,0.95}$).

This has a direct ML reading. On pure noise, $R^2\approx\chi^2_1/n$, so with $n=50$ you will see $R^2>0.077$ about $5\%$ of the time from nothing at all. With $k$ predictors the null becomes $nR^2\to\chi^2_k$, so $\mathbb{E}[R^2]\approx k/n$: reported $R^2$ grows with model size even under a pure-noise generating process. That is the entire justification for adjusted $R^2$, and it is the simplest instance of the overfitting arithmetic that Chapter 8 revisits and Chapter 12 generalizes to $k/n\to\gamma$ bounded away from $0$.

> **Common Confusion 7.9.**
> *"The delta method needs $g$ to be invertible, or at least monotone."*
>
> It needs $g$ to be differentiable *at the single point $\theta$*, and nothing else. Not invertible, not monotone, not smooth anywhere else. What it does need — and this is the hypothesis people actually violate — is $\nabla g(\theta)\ne0$, or else Theorem 7.38 takes over with a different rate and a non-Gaussian limit. It also needs $\theta$ to be an interior point in the relevant sense: applying the delta method to $g(p)=\sqrt p$ at $p=0$, where $g'(0)=\infty$, gives nonsense, and the correct limit there is $n\hat p_n\to_{\mathrm d}\mathrm{Poisson}$-type behaviour, not a Gaussian at rate $\sqrt n$. Boundary cases and flat points are where asymptotic normality quietly fails, and both are common in ML: sparsity means parameters *at* zero, and regularization paths pass through non-differentiable points by design.

> **Self-check 7.6.**
> 1. State the Cramér–Wold device and explain why coordinate marginals alone are not enough.
> 2. Compute the asymptotic variance of $\sqrt n(\log\bar X_n-\log\mu)$ for i.i.d. positive $X_i$ with mean $\mu$, variance $\sigma^2$.
> 3. For $g(p)=p(1-p)$ at $p=0.3$, is the first- or second-order delta method appropriate? What is the asymptotic variance?
> 4. Why does Fisher's $\operatorname{arctanh}$ transform make the limiting variance free of $\rho$?

---

## 7.7 Slutsky and continuous mapping

Two theorems in this section, and they are the two you will use most often in day-to-day statistical work. Both say the same kind of thing: a distributional limit survives being fed into a well-behaved operation.

**Theorem 7.39 (continuous mapping theorem).** Let $h:\mathbb{R}^d\to\mathbb{R}^m$ be Borel measurable and let $D_h$ be its set of discontinuity points.

1. If $X_n\to_{\mathrm{as}}X$ and $\mathbb{P}(X\in D_h)=0$, then $h(X_n)\to_{\mathrm{as}}h(X)$.
2. If $X_n\to_{\mathrm p}X$ and $\mathbb{P}(X\in D_h)=0$, then $h(X_n)\to_{\mathrm p}h(X)$.
3. If $X_n\to_{\mathrm d}X$ and $\mathbb{P}(X\in D_h)=0$, then $h(X_n)\to_{\mathrm d}h(X)$.

*Proof of (1).* On the event $\{X_n\to X\}\cap\{X\notin D_h\}$, which has probability $1$, continuity of $h$ at $X(\omega)$ gives $h(X_n(\omega))\to h(X(\omega))$. (Measurability of $D_h$ is a standard fact: $D_h=\bigcup_{k}\bigcap_{m}\{x: \operatorname{osc}_h(x,1/m)\ge 1/k\}$ with the oscillation, a Borel function.)

*Proof of (2).* Use Theorem 7.15. Let $(n_k)$ be any subsequence. Since $X_{n_k}\to_{\mathrm p}X$, extract a further subsequence with $X_{n_{k_j}}\to_{\mathrm{as}}X$; by part (1), $h(X_{n_{k_j}})\to_{\mathrm{as}}h(X)$. Every subsequence of $(h(X_n))$ thus has a further a.s.-convergent subsequence with the same limit, so Theorem 7.15 gives $h(X_n)\to_{\mathrm p}h(X)$.

*Proof of (3), sketch.* Let $f$ be bounded and continuous; we must show $\mathbb{E}[f(h(X_n))]\to\mathbb{E}[f(h(X))]$, i.e. that $f\circ h$ is an admissible test function. It is bounded, but continuous only off $D_h$. Use Portmanteau (Theorem 7.5) in the form (3): for closed $C$, $\limsup_n\mathbb{P}(h(X_n)\in C)\le\limsup_n\mathbb{P}(X_n\in \overline{h^{-1}(C)})\le\mathbb{P}(X\in\overline{h^{-1}(C)})\le\mathbb{P}(X\in h^{-1}(C)\cup D_h)=\mathbb{P}(h(X)\in C)$, where the third step uses $\overline{h^{-1}(C)}\subseteq h^{-1}(C)\cup D_h$ (a point in the closure that is not in $h^{-1}(C)$ must be a discontinuity of $h$). Apply Portmanteau again to conclude. Alternatively, invoke the Skorokhod representation theorem — every distributional limit can be realized as an a.s. limit on a suitable space — and reduce to part (1). Both routes are recorded in §7.10. $\square$

**Theorem 7.40 (Slutsky).** Suppose $X_n\to_{\mathrm d}X$ and $Y_n\to_{\mathrm p}c$ for a constant $c$. Then, jointly, $(X_n,Y_n)\to_{\mathrm d}(X,c)$, and consequently
$$X_n+Y_n\to_{\mathrm d}X+c,\qquad Y_nX_n\to_{\mathrm d}cX,\qquad X_n/Y_n\to_{\mathrm d}X/c\ \ (c\ne0).$$

*Proof.* It suffices to prove the joint statement $(X_n,Y_n)\to_{\mathrm d}(X,c)$ in $\mathbb{R}^2$; the three consequences then follow from Theorem 7.39(3) applied to $h(x,y)=x+y$, $h(x,y)=xy$, and $h(x,y)=x/y$ respectively. (For the last, $D_h=\{y=0\}$ and the limit law $(X,c)$ puts zero mass there when $c\ne0$.)

By Theorem 7.35 (Cramér–Wold) it is enough to show $uX_n+vY_n\to_{\mathrm d}uX+vc$ for all $u,v\in\mathbb{R}$. Now $uX_n\to_{\mathrm d}uX$ by Theorem 7.39(3), and $vY_n-vc\to_{\mathrm p}0$. So the claim reduces to the lemma already proved inside Theorem 7.37: if $A_n\to_{\mathrm d}A$ and $B_n\to_{\mathrm p}0$ then $A_n+B_n\to_{\mathrm d}A$. For completeness, here it is again in full. Let $x$ be a continuity point of $F_A$ and $\varepsilon>0$. Then
$$\mathbb{P}(A_n+B_n\le x)\le\mathbb{P}(A_n\le x+\varepsilon)+\mathbb{P}(|B_n|>\varepsilon),$$
$$\mathbb{P}(A_n+B_n\le x)\ge\mathbb{P}(A_n\le x-\varepsilon)-\mathbb{P}(|B_n|>\varepsilon).$$
Choosing $\varepsilon$ so that $x\pm\varepsilon$ are also continuity points of $F_A$ (possible for all but countably many $\varepsilon$), taking $\limsup$ and $\liminf$, and then letting $\varepsilon\downarrow0$ using continuity of $F_A$ at $x$, gives $\mathbb{P}(A_n+B_n\le x)\to F_A(x)$. $\square$

One more remark, needed by Chapter 8. The theorem is stated for scalars, but the proof already contains the vector–matrix version: if $X_n\to_{\mathrm d}X$ in $\mathbb{R}^d$ and $Y_n\to_{\mathrm p}C$ for a *constant* vector or matrix $C$, then $(X_n,Y_n)\to_{\mathrm d}(X,C)$ jointly — the Cramér–Wold step (Theorem 7.35) works coordinate-by-coordinate on $(u^\top X_n + \langle V, Y_n\rangle)$ exactly as before — and hence $Y_nX_n\to_{\mathrm d}CX$ by continuous mapping. This matrix form is what Chapter 8's MLE proof uses to multiply an inverse observed-information matrix (converging in probability to the constant $I(\theta_0)^{-1}$) against a score vector obeying a CLT.

The hypothesis that $c$ is a *constant* is essential and is exactly what §7.1.4's counterexample violates. Convergence in distribution says nothing about joint behaviour; a degenerate limit forces the joint behaviour, because $Y_n$ is asymptotically pinned.

**Corollary 7.41 (asymptotically valid confidence intervals).** Let $\hat\theta_n$ satisfy $\sqrt n(\hat\theta_n-\theta)\to_{\mathrm d}\mathcal N(0,\sigma^2)$ with $\sigma^2>0$, and let $\hat\sigma_n$ be any estimator with $\hat\sigma_n\to_{\mathrm p}\sigma$. Then
$$\frac{\sqrt n(\hat\theta_n-\theta)}{\hat\sigma_n}\ \longrightarrow_{\mathrm d}\ \mathcal N(0,1),$$
and consequently $\mathbb{P}\big(\theta\in[\hat\theta_n\pm z_{1-\alpha/2}\hat\sigma_n/\sqrt n]\big)\to1-\alpha$.

*Proof.* $\sigma/\hat\sigma_n\to_{\mathrm p}1$ by Theorem 7.39(2) with $h(y)=\sigma/y$, continuous at $y=\sigma>0$. Apply Theorem 7.40 with $X_n=\sqrt n(\hat\theta_n-\theta)/\sigma\to_{\mathrm d}\mathcal N(0,1)$ and $Y_n=\sigma/\hat\sigma_n\to_{\mathrm p}1$. For the coverage claim, the event is $\{|\sqrt n(\hat\theta_n-\theta)/\hat\sigma_n|\le z_{1-\alpha/2}\}$, whose probability converges to $\Phi(z_{1-\alpha/2})-\Phi(-z_{1-\alpha/2})=1-\alpha$ because $\pm z_{1-\alpha/2}$ are continuity points of $\Phi$. $\square$

This is the licence for every "estimate $\pm$ 1.96 standard errors" in the literature. Read the guarantee carefully, because it promises less than people assume. It says: *for each fixed data-generating distribution satisfying the hypotheses, the coverage probability converges to $1-\alpha$ as $n\to\infty$.* It does not say the coverage is $\ge1-\alpha$ at your $n$. It does not say the convergence is uniform over a class of distributions (it is not, in general, and non-uniformity is the technical heart of post-selection inference problems). And it says nothing at all if $\hat\theta_n$ is the output of a procedure that also selected the model, tuned hyperparameters, or peeked at the data — in that case $\hat\theta_n$ is not the estimator whose asymptotics you computed.

**Worked Example 7.27 (coverage of the plug-in interval, measured).** Corollary 7.41 promises coverage $\to0.95$ for $\bar X_n\pm1.96\,s_n/\sqrt n$ with $s_n$ the sample standard deviation. How fast? $M=200{,}000$ replications per cell, numpy PCG64 seed 2718:

| parent | $n=10$ | $n=30$ | $n=100$ | $n=1000$ |
|---|---:|---:|---:|---:|
| $\mathcal N(0,1)$ | $0.9187$ | $0.9404$ | $0.9485$ | $0.9499$ |
| $t_3$ | $0.9282$ | $0.9463$ | $0.9504$ | $0.9504$ |
| $\mathrm{Exp}(1)$ | $0.8690$ | $0.9175$ | $0.9399$ | $0.9493$ |
| lognormal$(0,1)$ | $0.8070$ | $0.8726$ | $0.9140$ | $0.9441$ |
| $\mathrm{Bern}(0.01)$ | $0.0949$ | $0.2601$ | $0.6336$ | $0.9266$ |

Read this table slowly; it contains most of what an applied statistician knows.

The Gaussian row undershoots at small $n$ for a reason unrelated to the CLT: $s_n$ is a noisy estimate of $\sigma$, and $\sqrt n(\bar X_n-\mu)/s_n$ is exactly $t_{n-1}$-distributed, whose tails are heavier than normal. Using $t_{9,0.975}=2.262$ instead of $1.96$ restores exact $95\%$ coverage here. That is the entire content of the $t$-distribution: a finite-$n$ correction for having estimated $\sigma$, valid under exact normality.

The $t_3$ row does *better* than the Gaussian row at small $n$, which surprises people. Two errors are cancelling: heavy tails make $\bar X_n$ non-normal (pushing coverage down) but also make $s_n$ occasionally huge (pushing intervals wide, coverage up). Do not read this as "heavy tails are fine" — the intervals are wildly variable in width, so the *expected* interval length is poor even though coverage is nominal.

The lognormal row is the honest warning about skew: at $n=100$ you are still $3.6$ percentage points short, and the miscoverage is one-sided (the interval sits too low because $\bar X_n$ is right-skewed).

The Bernoulli$(0.01)$ row is the catastrophe. At $n=10$, coverage is $9.5\%$ against a nominal $95\%$. The reason is stark: with probability $0.99^{10}=0.9044$ all ten observations are $0$, whence $\bar X_n=0$, $s_n=0$, and the interval is the single point $\{0\}$, which excludes $p=0.01$. Coverage is essentially $\mathbb{P}(\text{at least one success})=1-0.99^{10}=0.0956$, matching the measured $0.0949$ to within Monte Carlo error. No amount of appealing to Corollary 7.41 helps: $\hat\sigma_n\to_{\mathrm p}\sigma$ holds, but "eventually" is a long way off. This is exactly the regime of rare-event ML metrics, and the fix is a method designed for it (Wilson, Clopper–Pearson, or a Bayesian interval), not a bigger $z$.

> **Common Confusion 7.10.**
> *"Slutsky says I can plug in any consistent estimate of a nuisance parameter, so plug-in is free."*
>
> Slutsky says the *limit distribution* is unchanged. It says nothing about the finite-$n$ distribution, and the difference is exactly what Worked Example 7.27 measures — several percentage points of coverage at realistic $n$, and a total collapse in the rare-event regime. Three further traps. (a) Slutsky requires $Y_n\to_{\mathrm p}c$ with $c$ *constant*; if your "nuisance" estimate converges to a random limit (as when it is estimated on the same data through a selection step), Slutsky does not apply. (b) Plugging in changes the *variance* of the interval width, which affects power even when coverage is fine. (c) If $\hat\sigma_n$ and $\hat\theta_n$ are correlated — the usual case — the joint fluctuation induces a skew in the studentized statistic which the first-order theory cannot see; this is what higher-order (Edgeworth) corrections and the bootstrap-$t$ of Chapter 9 address.

> **Self-check 7.7.**
> 1. Give a counterexample showing Theorem 7.40 fails if $Y_n\to_{\mathrm d}Y$ with $Y$ non-degenerate.
> 2. Why does Theorem 7.39(3) need $\mathbb{P}(X\in D_h)=0$ rather than just measurability of $h$?
> 3. In Corollary 7.41, what exactly is being claimed to converge, and to what?
> 4. Explain the $9.5\%$ coverage in the Bernoulli$(0.01)$, $n=10$ cell in one sentence.

---

## 7.8 Empirical processes, briefly

Section 7.3 gave the law of large numbers for a fixed function; Chapter 4 gave uniform convergence over a restricted class. There is one class for which the uniform statement is free of any complexity assumption, and it is the class of indicator functions of half-lines. That fact — Glivenko–Cantelli — is the foundation of nonparametric statistics and of the bootstrap.

Let $X_1,\dots,X_n$ be i.i.d. with CDF $F$, and let
$$F_n(x):=\frac1n\sum_{i=1}^n\mathbf 1\{X_i\le x\}$$
be the *empirical CDF*, the CDF of the empirical distribution $\hat P_n=\frac1n\sum_i\delta_{X_i}$ of Chapter 1's notation. For fixed $x$, $nF_n(x)\sim\mathrm{Bin}(n,F(x))$, so the SLLN gives $F_n(x)\to_{\mathrm{as}}F(x)$ and the CLT gives $\sqrt n(F_n(x)-F(x))\to_{\mathrm d}\mathcal N(0,F(x)(1-F(x)))$. The content of this section is that both statements hold *uniformly in $x$*.

**Theorem 7.42 (Glivenko–Cantelli).** $\displaystyle D_n:=\sup_{x\in\mathbb{R}}\big|F_n(x)-F(x)\big|\ \longrightarrow_{\mathrm{as}}\ 0 .$

*Proof.* First, measurability: the supremum is over an uncountable family, which Chapter 1 warned is not automatically a random variable. But $F_n$ and $F$ are both right-continuous, so the supremum over $\mathbb{R}$ equals the supremum over $\mathbb{Q}\cup\{\text{jump points}\}$, a countable set, and Proposition 1.15 applies. This is the "continuity plus separability rescues measurability" pattern from Chapter 1, in its simplest form.

Now fix $k\in\mathbb{N}$ and choose points $-\infty=t_0<t_1<\cdots<t_k=+\infty$ with $F(t_j^-)\le j/k\le F(t_j)$ — such points exist by taking $t_j:=\inf\{x:F(x)\ge j/k\}$ (generalized inverse), and then $F(t_j)-F(t_{j-1}^-)\le 1/k$ for each $j$ by construction. Write $F_n(t^-)$ for the left limit. By the SLLN (Theorem 7.19) applied to the $2(k-1)$ fixed indicator variables $\mathbf 1\{X_i\le t_j\}$ and $\mathbf 1\{X_i<t_j\}$, there is a probability-one event on which
$$F_n(t_j)\to F(t_j)\quad\text{and}\quad F_n(t_j^-)\to F(t_j^-)\qquad\text{for all }j=1,\dots,k-1 .$$
Work on that event. For $x\in[t_{j-1},t_j)$, monotonicity of $F_n$ and $F$ gives
$$F_n(x)-F(x)\le F_n(t_j^-)-F(t_{j-1})\le \big[F_n(t_j^-)-F(t_j^-)\big]+\tfrac1k,$$
$$F_n(x)-F(x)\ge F_n(t_{j-1})-F(t_j^-)\ge\big[F_n(t_{j-1})-F(t_{j-1})\big]-\tfrac1k .$$
Taking the max over the finitely many $j$, $\limsup_nD_n\le1/k$ on that event. Intersecting the events over $k\in\mathbb{N}$ (countably many probability-one events) gives $D_n\to0$ almost surely. $\square$

The proof is a template worth extracting: *a monotone family is controlled by finitely many points, because monotonicity interpolates.* This is a bracketing argument, and it is the crudest member of the family of techniques (covering numbers, chaining, Rademacher complexity) that Chapter 4 developed for general function classes. Half-lines in $\mathbb{R}$ have VC dimension $1$; the uniformity is free because the class is tiny.

**Theorem 7.43 (Dvoretzky–Kiefer–Wolfowitz, with Massart's constant).** For every $n\ge1$ and every $\varepsilon>0$,
$$\mathbb{P}\big(D_n>\varepsilon\big)\ \le\ 2e^{-2n\varepsilon^2}.$$
The constant $2$ in front is sharp (Massart, 1990).

Look at the shape: it is *exactly* Hoeffding's bound for a single fixed $x$ (Ch. 4), with no penalty at all for taking the supremum over uncountably many $x$. That is a remarkable statement — the union bound would have cost a $\log$ factor even over a fine grid — and it is special to this class.

**Worked Example 7.28 (how many samples to know a distribution).** DKW gives a *distribution-free* confidence band: with probability at least $1-\alpha$, $F$ lies entirely within $F_n\pm\varepsilon_n$ where $\varepsilon_n=\sqrt{\log(2/\alpha)/(2n)}$. Sample sizes:

| target | $\alpha$ | required $n$ |
|---|---:|---:|
| $\varepsilon=0.05$ | $0.05$ | $738$ |
| $\varepsilon=0.01$ | $0.05$ | $18{,}445$ |
| $\varepsilon=0.01$ | $0.01$ | $26{,}492$ |

Note what "distribution-free" buys: these numbers hold for *every* $F$ — continuous or not — with no assumptions at all; Massart's inequality does not require continuity. (Continuity of $F$ is needed only for the separate claim below, that the *limit law* of $\sqrt n D_n$ is the same for all $F$.) Note also what it costs: the $n^{-1/2}$ rate means one extra decimal digit of resolution costs $100\times$ the data, and DKW controls the CDF uniformly but says nothing about a density, whose estimation is genuinely harder (rate $n^{-2/5}$ under smoothness).

How conservative is the band? Simulating $\mathbb{E}[D_n]$ for a $\mathrm{Unif}(0,1)$ parent (numpy PCG64, seed 2718; $2000$ replications for $n\le10^4$, $300$ for $n=10^5$):

| $n$ | $\mathbb{E}[D_n]$ | $\sqrt n\,\mathbb{E}[D_n]$ | DKW $95\%$ bound |
|---:|---:|---:|---:|
| $10$ | $0.25843$ | $0.8172$ | $0.4295$ |
| $10^2$ | $0.08444$ | $0.8444$ | $0.1358$ |
| $10^3$ | $0.02742$ | $0.8672$ | $0.0429$ |
| $10^4$ | $0.00868$ | $0.8680$ | $0.0136$ |
| $10^5$ | $0.00282$ | $0.8902$ | $0.0043$ |

The column $\sqrt n\,\mathbb{E}[D_n]$ converges to about $0.87$, and the theoretical limit is $\mathbb{E}\big[\sup_{t}|B^\circ(t)|\big]=\sqrt{\pi/2}\,\log2=0.8687$, where $B^\circ$ is the Brownian bridge — which is Donsker's theorem, next. The DKW $95\%$ bound is roughly $1.5$–$1.6\times$ the mean, which is about right for a bound that must cover the upper tail as well.

**Theorem 7.44 (Donsker).** Let $\mathbb{G}_n(x):=\sqrt n\,(F_n(x)-F(x))$. Then $\mathbb{G}_n$ converges in distribution, as a random element of the space $D[-\infty,\infty]$ of right-continuous functions with left limits (equipped with the Skorokhod topology), to $B^\circ\circ F$, where $B^\circ$ is a standard Brownian bridge on $[0,1]$: a centred Gaussian process with $\operatorname{Cov}(B^\circ(s),B^\circ(t))=s\wedge t-st$.

The finite-dimensional part of Donsker's theorem is nothing more than the multivariate CLT of Theorem 7.36: for fixed $x_1<\cdots<x_k$, the vector $\mathbb{G}_n(x_1),\dots,\mathbb{G}_n(x_k)$ is $\sqrt n$ times a centred average of i.i.d. vectors of indicators, and its covariance is $\operatorname{Cov}(\mathbf 1\{X\le x_i\},\mathbf 1\{X\le x_j\})=F(x_i\wedge x_j)-F(x_i)F(x_j)$ — exactly the bridge covariance after the time change $t=F(x)$. What Donsker adds is *tightness* in function space, upgrading finite-dimensional convergence to convergence of the whole path. That upgrade is the technical content, and it is recorded in §7.10.

Why you should care. Donsker is the reason the bootstrap works (Ch. 9): the bootstrap approximates the law of $\mathbb{G}_n$ by resampling, and the approximation is valid because both converge to the same Gaussian process. It is also the reason that the Kolmogorov–Smirnov statistic $\sqrt nD_n$ has a distribution-free limiting law $\sup_t|B^\circ(t)|$ — the Kolmogorov distribution — independent of $F$, which is what makes the KS test possible at all.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" role="img">
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="60" y1="190" x2="300" y2="190"/><line x1="60" y1="190" x2="60" y2="35"/>
  </g>
  <line x1="60" y1="190" x2="300" y2="40" stroke="currentColor" stroke-width="1.4" stroke-dasharray="5 4"/>
  <g stroke="#10b981" stroke-width="1" fill="none" stroke-dasharray="3 3">
    <line x1="60" y1="144.4" x2="300" y2="40"/><line x1="60" y1="190" x2="300" y2="85.6"/>
  </g>
  <path d="M 60 190 L 102.6 190.0 L 102.6 182.5 L 102.9 182.5 L 102.9 175.0 L 114.3 175.0 L 114.3 167.5 L 126.7 167.5 L 126.7 160.0 L 131.6 160.0 L 131.6 152.5 L 141.3 152.5 L 141.3 145.0 L 145.2 145.0 L 145.2 137.5 L 148.9 137.5 L 148.9 130.0 L 167.6 130.0 L 167.6 122.5 L 172.1 122.5 L 172.1 115.0 L 183.6 115.0 L 183.6 107.5 L 212.6 107.5 L 212.6 100.0 L 213.6 100.0 L 213.6 92.5 L 216.7 92.5 L 216.7 85.0 L 240.7 85.0 L 240.7 77.5 L 249.7 77.5 L 249.7 70.0 L 258.2 70.0 L 258.2 62.5 L 277.2 62.5 L 277.2 55.0 L 280.8 55.0 L 280.8 47.5 L 292.1 47.5 L 292.1 40.0 L 300 40" fill="none" stroke="#3b82f6" stroke-width="1.8"/>
  <g stroke="currentColor" stroke-width="1.2" fill="none">
    <line x1="400" y1="190" x2="650" y2="190"/><line x1="400" y1="190" x2="400" y2="35"/>
  </g>
  <polyline fill="none" stroke="#ef4444" stroke-width="2" points="400,63 452,90 512,119 572,147 632,175"/>
  <polyline fill="none" stroke="#10b981" stroke-width="2" points="400,45 452,73 512,101 572,129 632,157"/>
  <g font-size="11" fill="currentColor" font-family="sans-serif">
    <text x="54" y="44" text-anchor="end">1</text><text x="54" y="194" text-anchor="end">0</text>
    <text x="180" y="208" text-anchor="middle">x   (n = 20 sample)</text>
    <text x="394" y="49" text-anchor="end">0.3</text><text x="394" y="105" text-anchor="end">0.03</text><text x="394" y="161" text-anchor="end">0.003</text>
    <text x="400" y="206" text-anchor="middle">10</text><text x="512" y="206" text-anchor="middle">10³</text><text x="632" y="206" text-anchor="middle">10⁵</text>
    <text x="525" y="222" text-anchor="middle">n</text>
  </g>
  <g font-size="10.5" font-family="sans-serif">
    <text x="206" y="176" fill="#3b82f6">F_n</text><text x="248" y="60" fill="currentColor">F</text>
    <text x="88" y="132" fill="#10b981">DKW 95% band</text>
    <text x="470" y="80" fill="#ef4444">DKW 95% bound</text><text x="450" y="180" fill="#10b981">E[D_n] (simulated)</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-family="sans-serif">
    <text x="20" y="248">Figure 7.8. LEFT: the empirical CDF of 20 uniform draws (seed 2026) against the true F, with the DKW 95%</text>
    <text x="20" y="267">band of half-width 0.3037; the observed sup-distance is 0.1774. RIGHT: E[D_n] and the DKW bound on log–log</text>
    <text x="20" y="286">axes. Both have slope −1/2; the bound sits a constant factor of about 1.5 above the mean, and √n E[D_n] → 0.869.</text>
  </g>
</svg>

> **Self-check 7.8.**
> 1. Why is $\sup_x|F_n(x)-F(x)|$ a random variable even though the supremum is over uncountably many $x$?
> 2. Compare the DKW bound to what the union bound over a grid of $m$ points would give.
> 3. What is the limiting distribution of $\sqrt n\,D_n$, and why does it not depend on $F$?
> 4. Does Glivenko–Cantelli imply that the empirical *density* converges uniformly?

---

## 7.9 ML payoffs

Four places where this chapter's theorems are doing real work in machine learning — and, in each case, the boundary of what they actually license.

### 7.9.1 Why SGD noise is approximately Gaussian

Fix parameters $\theta\in\mathbb{R}^p$ and a training set $\mathcal{D}=\{z_1,\dots,z_N\}$. The full-batch gradient is $g(\theta):=\frac1N\sum_{i=1}^N\nabla_\theta\ell(\theta;z_i)$, and a minibatch of size $B$ drawn uniformly with replacement gives
$$\hat g_B(\theta):=\frac1B\sum_{b=1}^{B}\nabla_\theta\ell(\theta;z_{I_b}),\qquad I_1,\dots,I_B\overset{\text{iid}}{\sim}\mathrm{Unif}\{1,\dots,N\}.$$

**Proposition 7.45 (minibatch gradient CLT).** Fix $\theta$ and $\mathcal{D}$, and let $\Sigma(\theta):=\frac1N\sum_i\big(\nabla\ell(\theta;z_i)-g(\theta)\big)\big(\nabla\ell(\theta;z_i)-g(\theta)\big)^\top$ be the empirical gradient covariance. Then, as $B\to\infty$ with $\theta$ and $\mathcal{D}$ fixed,
$$\sqrt B\,\big(\hat g_B(\theta)-g(\theta)\big)\ \longrightarrow_{\mathrm d}\ \mathcal N\big(0,\ \Sigma(\theta)\big).$$

*Proof.* The summands $\nabla\ell(\theta;z_{I_b})$ are i.i.d. random vectors in $\mathbb{R}^p$ (a finite population sampled with replacement), with mean $g(\theta)$ and covariance $\Sigma(\theta)$. All moments are finite because the population is finite. Theorem 7.36 applies directly. $\square$

Two things are worth saying about this innocuous-looking proposition. First, the randomness is over the *index draw*, not over the data — $\mathcal{D}$ is held fixed, which is exactly Chapter 1's picture of $\omega$ as the seed. Second, the limit is $B\to\infty$, not $N\to\infty$, so the relevant question is whether $B=32$ or $B=256$ is "large", and that is a Berry–Esseen question about $\rho/\sigma^3$ for the *per-example gradient distribution*, not about $B$ in isolation.

**Worked Example 7.29 (how Gaussian is minibatch noise, actually?).** Take a one-dimensional least-squares problem: $N=50{,}000$ training points, $y_i=2x_i+\epsilon_i$ with $\epsilon_i\sim\mathcal N(0,1)$, loss $\ell(w;x,y)=\tfrac12(wx-y)^2$, evaluated at $w=1.5$. The per-example gradient is $\nabla\ell=x(wx-y)$. I ran two versions, one with $x_i\sim\mathcal N(0,1)$ and one with $x_i\sim t_3$, and measured the Kolmogorov distance between the standardized minibatch gradient and $\Phi$ over $M=200{,}000$ minibatches (numpy PCG64, seed 1729).

| features | per-example excess kurtosis | $B=1$ | $B=8$ | $B=32$ | $B=256$ |
|---|---:|---:|---:|---:|---:|
| $x\sim\mathcal N(0,1)$ | $8.9$ | $0.1583$ | $0.0488$ | $0.0270$ | $0.0107$ |
| $x\sim t_3$ | $3243$ | $0.3433$ | $0.2652$ | $0.1930$ | $0.1340$ |

With Gaussian features the noise is already close to normal at $B=32$ and essentially normal at $B=256$ — the numbers track $B^{-1/2}$ neatly ($0.0270\times\sqrt{32}=0.153$, $0.0107\times\sqrt{256}=0.171$). With $t_3$ features it is *not* close at $B=256$, and the decay is visibly slower than $B^{-1/2}$ ($0.1930\times\sqrt{32}=1.09$, $0.1340\times\sqrt{256}=2.14$ — the product is still growing, so the $B^{-1/2}$ regime has not started).

Note that even the "Gaussian features" row has per-example excess kurtosis $8.9$: the gradient $x(wx-y)$ is a *product* of near-Gaussians, hence heavy-tailed even when the inputs are not. This is generic. Gradients in deep networks are products of many factors (activations times backpropagated errors), and their empirical distributions are routinely reported as heavy-tailed, sometimes with estimated tail index below $2$ — in which case Proposition 7.45's variance hypothesis fails outright and the correct limit is an $\alpha$-stable law, not a Gaussian. That is a live empirical debate; treat "SGD noise is Gaussian" as a modelling assumption to be checked, not a theorem.

### 7.9.2 The diffusion approximation, and what it does not say

If one accepts $\hat g_B\approx g+\sqrt{\Sigma/B}\,\xi$ with $\xi\sim\mathcal N(0,I)$, the SGD recursion $\theta_{k+1}=\theta_k-\eta\hat g_B(\theta_k)$ becomes an Euler–Maruyama discretization of the stochastic differential equation
$$d\theta_t=-\nabla F(\theta_t)\,dt+\sqrt{\tfrac{\eta}{B}\,\Sigma(\theta_t)}\;dW_t,$$
with $F$ the full-batch objective. This is the source of a large family of useful heuristics: the *linear scaling rule* (double $B$, double $\eta$, to hold $\eta/B$ fixed and preserve the noise level), the claim that the stationary distribution looks like $\exp(-2F/(\eta/B))$ when $\Sigma\propto I$, and the "flat minima" story in which large $\eta/B$ escapes sharp basins.

Three honest caveats, each traceable to a hypothesis in this chapter.

*The CLT is per-step, the SDE is over many steps.* Proposition 7.45 gives approximate normality of one gradient. Chaining $K$ steps requires the errors not to accumulate, and rigorous statements (weak approximation of order $1$ or $2$ in $\eta$) hold over horizons $K=O(1/\eta)$ with constants depending on the smoothness of $F$ — not over the $10^5$–$10^6$ steps of a real training run.

*Heavy tails break it.* If per-example gradients have infinite variance, there is no $\Sigma$ and the Brownian $dW_t$ should be a stable Lévy process instead. Worked Example 7.29's second row is a toy version of this.

*The covariance is state-dependent and anisotropic.* $\Sigma(\theta)$ is not a multiple of the identity; near a minimum of a least-squares problem it is approximately the Gauss–Newton/Fisher matrix, so the noise is largest in the directions of largest curvature. Analyses that assume isotropic noise are analysing a different process.

### 7.9.3 Bootstrap validity, MLE asymptotics, and the fixed-$n$ caveat

**Bootstrap (Ch. 9).** The bootstrap replaces the unknown $P$ by $\hat P_n$ and resamples. Its asymptotic validity is a two-step argument built entirely out of this chapter: Glivenko–Cantelli (Theorem 7.42) says $\hat P_n$ is uniformly close to $P$; Donsker (Theorem 7.44) says $\sqrt n(\hat P_n-P)$ has a Gaussian-process limit; and the delta method (Theorem 7.37), in its functional form (the *Hadamard*-differentiability version), transfers that limit to any smooth functional $T(\hat P_n)$. The bootstrap distribution of $\sqrt n(T(\hat P_n^*)-T(\hat P_n))$ then converges to the same law as $\sqrt n(T(\hat P_n)-T(P))$. Where the functional is *not* smooth — the maximum, a boundary parameter, the number of nonzero coefficients in a lasso fit — the bootstrap is inconsistent, and this is exactly the $\nabla g(\theta)=0$ / non-differentiability failure of Common Confusion 7.9 in another costume.

**MLE (Ch. 8).** Under regularity conditions, the maximum likelihood estimator satisfies $\sqrt n(\hat\theta_n-\theta_0)\to_{\mathrm d}\mathcal N(0,I(\theta_0)^{-1})$ with $I$ the Fisher information. The proof is a delta-method-shaped argument: Taylor-expand the score equation $0=\nabla\ell_n(\hat\theta_n)$ around $\theta_0$, apply the CLT (Theorem 7.31) to the score $n^{-1/2}\nabla\ell_n(\theta_0)$, apply the LLN (Theorem 7.19) to the observed information $-n^{-1}\nabla^2\ell_n$, and glue them with Slutsky (Theorem 7.40). Chapter 8 does this carefully; you already own all three ingredients.

**The fixed-$n$ caveat, stated bluntly.** Every theorem in this chapter is a statement about a limit with $p$, the model, and the estimation procedure all held fixed. Modern ML violates each of these:

- *$n$ is not large relative to $p$.* Overparameterized models have $p\gg n$. The classical asymptotics assume $p/n\to0$; when $p/n\to\gamma>0$ the sample covariance does not converge to $\Sigma$ (its eigenvalues follow the Marchenko–Pastur law), the MLE need not be consistent, and $\hat\sigma^2$ is biased by a factor $(1-\gamma)$. Chapter 12 takes this up properly.
- *The estimator is data-adaptive.* Architecture search, early stopping, and hyperparameter tuning all use the data, so the final $\hat\theta_n$ is not the estimator whose limit you computed. Corollary 7.41's guarantee evaporates.
- *The data are not i.i.d.* Time series, active learning, and RL all violate independence. Martingale CLTs (which use Chapter 6's machinery) are the right replacement, and their conditions are genuinely harder to check.
- *You care about tails, not the centre.* Berry–Esseen's uniform $n^{-1/2}$ error is useless for a probability of $10^{-4}$. Safety-critical failure rates need concentration bounds (Ch. 4) or large-deviation theory, not the CLT.

**Worked Example 7.30 (an asymptotic statement that is true and useless).** Suppose you measure a model's catastrophic-failure rate $p$ and observe $0$ failures in $n=10{,}000$ trials. The CLT interval is $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}=[0,0]$: you have "proved" the failure rate is exactly zero. Corollary 7.41 is not violated — its promise is about $n\to\infty$ for fixed $p>0$ — but it is vacuous here because $n\hat p=0$. The correct statement is the exact one-sided Clopper–Pearson bound: $\mathbb{P}(\text{0 failures})= (1-p)^n\ge0.05$ iff $p\le1-0.05^{1/n}=2.9953\times10^{-4}$, so the $95\%$ upper confidence bound is $p\le0.0003$, the familiar "rule of three" ($3/n$). Two orders of magnitude of difference between the asymptotic answer and the right one, on a question that matters.

> **Common Confusion 7.11.**
> *"My training loss curve flattened, so by the law of large numbers the model has converged."*
>
> Three distinct things are being conflated. (a) *Optimization* convergence: $\theta_k$ settling, which is a statement about a dependent, non-stationary sequence and needs Chapter 6's martingale tools plus Proposition 7.14, not the LLN. (b) *Statistical* convergence of the empirical risk to the true risk at a fixed $h$: Corollary 7.21, which says nothing about the trained $\hat h_n$. (c) *Generalization*: $\sup_h|\hat R_n(h)-R(h)|$, Chapter 4's business, which requires a complexity restriction. A flat training curve is evidence for (a) alone, and under a state-dependent noise covariance it is not even strong evidence for that — SGD can plateau on a flat region for a long time and then leave it.

> **Self-check 7.9.**
> 1. In Proposition 7.45, what is the limiting variable and which quantity plays the role of "$n$"?
> 2. Why does the linear scaling rule pair $\eta$ with $B$ rather than with $N$?
> 3. Give two hypotheses of Corollary 7.41 that a tuned deep-learning pipeline violates.
> 4. In Worked Example 7.30, why does the CLT interval have zero width, and what is the correct upper bound?

---

## 7.10 Results used but not proved

Results inherited from earlier chapters (dominated and monotone convergence, Fatou, Fubini–Tonelli, the tail formula, Jensen, Markov, Chebyshev, Hoeffding, the multivariate Gaussian's affine-transformation rule, martingales and Azuma) were proved where they were introduced and are cited here by name and chapter only. The table below lists everything used in *this* chapter that is not proved in *any* chapter.

| Result | Used in | Where to find a proof |
|---|---|---|
| Portmanteau theorem, equivalences (2), (4), (5) | §7.1, Theorem 7.39(3) | Billingsley, *Convergence of Probability Measures*, 2nd ed., Thm 2.1; Durrett, *PTE*, 5th ed., Thm 3.2.11 |
| Skorokhod representation theorem | Theorem 7.39(3), alternative route | Billingsley, Thm 6.7; Durrett Thm 3.2.8 |
| Measurability of the discontinuity set $D_h$ of a function | Theorem 7.39 | Billingsley, §M (Appendix); Folland, *Real Analysis*, Prop. 4.32 exercise |
| Kolmogorov's SLLN under $\mathbb{E}\lvert X\rvert<\infty$ (Theorem 7.19, "if" direction) | §7.3, §7.8 | Durrett Thm 2.4.1 (Etemadi's proof); Billingsley, *Probability and Measure*, Thm 22.1 |
| Kolmogorov's maximal inequality; three-series theorem | mentioned in §7.3.2 as the classical route | Durrett Thm 2.5.5, Thm 2.5.8 |
| Inversion formula for characteristic functions (Theorem 7.28) | §7.4, Worked Example 7.16 | Durrett Thm 3.3.11 and Thm 3.3.14; Feller Vol. II, XV.3 |
| Lévy continuity theorem, direction (2) (Theorem 7.30) | Theorem 7.31, Theorem 7.35 | Durrett Thm 3.3.17; Billingsley, *P&M*, Thm 26.3 |
| Helly's selection theorem; Prokhorov's theorem / tightness | inside Theorem 7.30 and Theorem 7.44 | Durrett Thm 3.2.12–3.2.14; Billingsley, *CPM*, Thm 5.1 |
| Multivariate Lévy continuity theorem | Theorem 7.35 | Durrett Thm 3.10.7 |
| Lindeberg–Feller CLT (Theorem 7.32) and Feller's converse | §7.5.2 | Durrett Thm 3.4.10 and Thm 3.4.13; Billingsley, *P&M*, Thm 27.2 |
| Berry–Esseen theorem (Theorem 7.34); constant $C\le0.4748$ | §7.5.3 | Feller Vol. II, XVI.5 (Esseen's smoothing lemma); Shevtsova, arXiv:1111.6554 (2011) for the constant |
| Leading Edgeworth term $-\tfrac{\gamma}{6\sqrt n}(x^2-1)\phi(x)$ | Worked Example 7.20 | Feller Vol. II, XVI.4; Hall, *The Bootstrap and Edgeworth Expansion*, Ch. 2 |
| Esseen's lattice correction: limiting constant $1/\sqrt{2\pi}$ for a symmetric lattice parent | Worked Example 7.20 | Esseen (1945), *Acta Math.* 77; Petrov, *Sums of Independent Random Variables*, Ch. V |
| Characteristic function of $t_\nu$; in particular $\varphi_{t_3}(t)=(1+\sqrt3\lvert t\rvert)e^{-\sqrt3\lvert t\rvert}$ | Worked Example 7.20, Lab B2 | Hurst (1995), *Statistics & Probability Letters* 25; Dreier & Kotz (2002) |
| Gil–Pelaez inversion $F(x)=\tfrac12-\tfrac1\pi\int_0^\infty \operatorname{Im}(e^{-itx}\varphi(t))/t\,dt$ | Worked Example 7.20, Lab B2 | Gil-Pelaez (1951), *Biometrika* 38; Wendel (1961) |
| Heyde's example: distinct laws with identical moments (lognormal) | Common Confusion 7.6 | Heyde (1963), *JRSS-B* 25; Billingsley, *P&M*, Example 30.2 |
| DKW inequality with Massart's constant (Theorem 7.43) | §7.8 | Massart (1990), *Ann. Probab.* 18; van der Vaart & Wellner, §2.14 |
| Donsker's theorem (Theorem 7.44) | §7.8 | Billingsley, *CPM*, Thm 14.3; van der Vaart, *Asymptotic Statistics*, Thm 19.3 |
| Vitali convergence theorem: for $X_n\to_{\mathrm p}X$, $L^1$ convergence $\iff$ uniform integrability | §7.1.4 remark; Exercises A3, A8 prove the directions actually used | Durrett, *PTE*, 5th ed., Thm 4.6.3; Billingsley, *P&M*, Thm 16.14 |
| Student's theorem: under exact normality, $\bar X_n\perp s_n^2$ and $\sqrt n(\bar X_n-\mu)/s_n\sim t_{n-1}$ exactly | Worked Example 7.27 | Casella & Berger, *Statistical Inference*, 2nd ed., Thm 5.3.1; also derived in Chapter 9 of this book |
| Consistency and $\sqrt n$-asymptotics of the sample median (and general sample quantiles) | Worked Example 7.11, Figure 7.3, Lab B3 | van der Vaart, *Asymptotic Statistics*, Cor. 21.5; Chapter 9 of this book sketches the argument |
| Minimax rate $n^{-2/5}$ for density estimation under two derivatives | Worked Example 7.28, Self-check 7.8 | Tsybakov, *Introduction to Nonparametric Estimation*, Ch. 1–2 |
| Existence of laws with $\varphi$ differentiable at $0$ but $\mathbb{E}\lvert X\rvert=\infty$ (density $\propto 1/(x^2\log\lvert x\rvert)$) | caution after Proposition 7.26 | Durrett, *PTE*, 5th ed., §3.3 exercises; Feller Vol. II, XVII.2a |
| $\mathbb{E}\big[\sup_t\lvert B^\circ(t)\rvert\big]=\sqrt{\pi/2}\,\log 2$ | Worked Example 7.28 | Kolmogorov distribution moments; Shorack & Wellner, *Empirical Processes*, Ch. 3 |
| Asymptotic variance $(1-\rho^2)^2$ of the sample correlation under bivariate normality | Worked Example 7.24 | Ferguson, *A Course in Large Sample Theory*, §7; Lehmann, *Elements of Large-Sample Theory*, §5.4 |
| Fisher's $z$-transform and the $n-3$ finite-sample refinement | Worked Example 7.24 | Fisher (1921), *Metron* 1; Lehmann §5.4 |
| Hadamard differentiability and the functional delta method | §7.9.3 | van der Vaart, *Asymptotic Statistics*, Ch. 20 |
| Asymptotic normality of the MLE | §7.9.3 (previewed) | Chapter 8 of this book; van der Vaart Ch. 5 |
| Clopper–Pearson / Wilson intervals and the "rule of three" | Worked Examples 7.18, 7.27, 7.30 | Brown, Cai & DasGupta (2001), *Statistical Science* 16; Louis (1981), *Am. Statistician* 35 |
| Extreme value theory (limits for maxima are not Gaussian) | Common Confusion 7.8 | de Haan & Ferreira, *Extreme Value Theory*, Ch. 1 |
| Weak (SDE) approximation of SGD, order-$1$ and order-$2$ results | §7.9.2 | Li, Tai & E (2019), *JMLR* 20(40); Mandt, Hoffman & Blei (2017), *JMLR* 18(134) |
| Marchenko–Pastur law and $p/n\to\gamma$ asymptotics | §7.9.3 (forward reference) | Chapter 12 of this book; Bai & Silverstein, *Spectral Analysis of Large Dimensional Random Matrices* |
| Martingale CLT (for dependent summands) | §7.9.3 | Hall & Heyde, *Martingale Limit Theory*, Ch. 3; Durrett Thm 8.2.8 |

---

## 7.11 Chapter summary

1. There are four modes of convergence — almost sure, in probability, in $L^p$, in distribution — and exactly four implications among them hold in general: $L^q\Rightarrow L^p\Rightarrow$ probability $\Rightarrow$ distribution, and a.s. $\Rightarrow$ probability. Every other arrow has a counterexample built from a moving spike or a sliding block (Figure 7.1).
2. Convergence in distribution is about *laws*, not about random variables. $X_n\to_{\mathrm d}X$ tells you nothing about $X_n-X$; the one exception is a constant limit, which upgrades to convergence in probability (Proposition 7.10) and is what makes Slutsky work.
3. Convergence in probability constrains one time at a time; almost sure convergence constrains a whole tail at once. The gap is exactly $\sum_n\mathbb{P}(|X_n-X|>\varepsilon)$, and Borel–Cantelli I closes it whenever that sum converges. *A rate turns convergence in probability into almost sure convergence.*
4. Borel–Cantelli II is the converse, and it needs independence. Nested events (Worked Example 7.9) and SGD iterates are both counterexamples to using it carelessly.
5. The strong law holds under $\mathbb{E}|X_1|<\infty$ and fails — dramatically, not marginally — without it: Cauchy sample means have the same law as a single draw, for every $n$ (Worked Example 7.16).
6. The SLLN gives $\hat R_n(h)\to R(h)$ for each *fixed* $h$, with a null set depending on $h$. It gives no generalization bound; uniformity over $\mathcal H$ is a strictly stronger and separately earned statement.
7. Characteristic functions exist for every distribution, always; MGFs do not. $\varphi$ determines the law (Corollary 7.29), turns independent sums into products (Proposition 7.24), and encodes moments in its behaviour at $0$ (Proposition 7.26). The corner of $e^{-|t|}$ at $t=0$ is the Cauchy law's missing mean, visible.
8. The i.i.d. CLT follows from one line, $\varphi(t)=1-t^2/2+o(t^2)$, plus a telescoping bound on products (Lemma 7.27) and Lévy continuity. Finite variance is used exactly once, and nothing else is needed — no densities, no boundedness, no symmetry.
9. Lindeberg–Feller drops identical distribution. Its condition means "no single summand carries a non-vanishing share of the total variance"; one whale plus many minnows is not Gaussian.
10. Berry–Esseen gives the rate $n^{-1/2}$ with constant $C\rho/\sigma^3$. The rate is universal; the constant is not, and it is the constant that decides whether $n=30$ or $n=10^4$ is enough. Measured exactly, $\sqrt n D_n$ tends to $0.133$ for Exp$(1)$ and $2.66$ for Bern$(0.01)$ — a factor of twenty.
11. "$n\ge30$" is not a theorem. For binary data use $np(1-p)\gtrsim10$; for skewed data expect the centre to be accurate long before the tails; for maxima and extreme quantiles the CLT is the wrong theorem entirely.
12. The delta method pushes a limit through a smooth map: $\sqrt n(g(\hat\theta)-g(\theta))\to_{\mathrm d}\mathcal N(0,\nabla g^\top\Sigma\nabla g)$. It needs differentiability at one point and a non-vanishing gradient — nothing more, and nothing less.
13. When $\nabla g(\theta)=0$ the rate improves to $1/n$, the limit becomes a quadratic form in a Gaussian, and it acquires a sign. $nR^2\to_{\mathrm d}\chi^2_1$ under the null is this phenomenon, and it is why $R^2$ inflates with model size.
14. Variance-stabilizing transforms (logit, $\operatorname{arctanh}$, $\log$) are delta-method constructions. They fix not only the variance but usually the skew, which is why intervals should be built on the transformed scale and mapped back.
15. Slutsky licenses substituting a consistent nuisance estimate without changing the limit — asymptotically. Measured coverage of $\bar X_n\pm1.96s_n/\sqrt n$ at $n=100$ ranges from $0.9485$ (Gaussian parent) to $0.6336$ (Bernoulli$(0.01)$ parent). The theorem is true in every one of those cells.
16. Glivenko–Cantelli and DKW give free uniform convergence of the empirical CDF, with no complexity penalty, because half-lines form a tiny class. Donsker upgrades this to a Gaussian-process limit and is the reason the bootstrap works — and the reason it fails for non-smooth functionals.

### Callbacks

This chapter discharged several promissory notes. Chapter 1's insistence that *countability is load-bearing* was paid off twice: in Definition 7.1, where the set of convergence is measurable only because it is a countable combination of events, and in Theorem 7.42, where the supremum over uncountably many $x$ is a random variable only because right-continuity lets us restrict to a countable set. Chapter 1's Failure 3 — swapping a derivative and an expectation — was closed in Chapter 2 by dominated convergence and is *used* here, in Worked Example 7.14, to derive the Gaussian characteristic function; Worked Example 7.2 exhibits the moving spike that shows why the domination hypothesis cannot be dropped. Chapter 1's running example (ii), $A=\max(0,Z)$ with neither pmf nor pdf, reappears in Worked Examples 7.12 and 7.17 with a Monte Carlo estimate and a confidence interval, demonstrating concretely that the LLN and CLT never needed a density. Chapter 1's running example (i), three coin flips, supplies the discrete counterexample of Worked Example 7.5. Chapter 3's warning that the MGF may not exist is discharged in §7.4, where the characteristic function replaces it with no hypotheses at all. Chapter 4's uniform convergence machinery is given its motivation in §7.3.3 and its easiest special case in §7.8. Chapter 5's characterization of the multivariate Gaussian by its one-dimensional projections is precisely what makes the Cramér–Wold proof of Theorem 7.36 work.

### Before you continue

1. State all four modes of convergence and draw the implication diagram from memory, attaching a counterexample to each missing arrow.
2. Why does convergence in probability plus a summable rate give almost sure convergence, and which lemma is doing the work?
3. Why does Borel–Cantelli II require independence when Borel–Cantelli I does not?
4. What exactly does the SLLN assume, and what is the sharp failure when the assumption is dropped?
5. Name two things a characteristic function can do that a moment generating function cannot.
6. Reproduce the five steps of the CLT proof, naming the hypothesis used at each step.
7. What does Berry–Esseen add to the CLT, and what does it *not* tell you about a tail probability of $10^{-4}$?
8. State the delta method, and state what changes when $\nabla g(\theta)=0$.
9. What does "asymptotically valid at level $95\%$" promise, and what does it not promise, at $n=100$?

### Looking ahead to Chapter 8

Chapter 8 turns from limit theorems to *point estimation*: given data and a model, how should you produce a single number, and how good can that number be? It builds the maximum likelihood estimator and proves its asymptotic normality by gluing together three results you now own — the CLT applied to the score, the LLN applied to the observed information, and Slutsky to combine them — with Fisher information $I(\theta)$ emerging as the asymptotic precision. The Cramér–Rao inequality then shows $I(\theta)^{-1}/n$ is a floor on the variance of any unbiased estimator, making the MLE asymptotically optimal. Alongside this runs the exact, finite-sample theory: sufficiency, the factorization criterion, and the Rao–Blackwell theorem, which says conditioning on a sufficient statistic never hurts. The chapter then reframes regularization as a prior, deriving MAP estimation and showing that ridge and lasso penalties are Gaussian and Laplace priors in disguise, and develops the EM algorithm for latent-variable models as a sequence of surrogate maximizations. It closes with an honest treatment of the bias–variance decomposition in the overparameterized regime, where the classical U-shaped curve is not what happens and "more parameters than data" is not automatically fatal.

---

# Exercises

Difficulty: ★ routine, ★★ requires an idea, ★★★ substantial.

## Part A — Theory

**A1 (★).** Suppose $X_n\to_{L^2}X$. Prove that $\mathbb{E}[X_n]\to\mathbb{E}[X]$ and $\operatorname{Var}(X_n)\to\operatorname{Var}(X)$. Then give an example showing that $X_n\to_{\mathrm p}X$ alone implies neither.

**A2 (★).** (a) Construct $X_n$ and $X$ on a *common* probability space with $X_n\to_{\mathrm d}X$ but $\mathbb{P}(|X_n-X|\ge1)=1$ for every $n$. (b) Construct $Y_n$ with $Y_n\to_{\mathrm{as}}0$ but $\mathbb{E}|Y_n|\to\infty$.

**A3 (★).** Suppose $X_n\to_{\mathrm p}X$ and there is $M<\infty$ with $|X_n|\le M$ almost surely for every $n$. Show that $X_n\to_{L^p}X$ for every $p\in[1,\infty)$. Where is boundedness used, and what does Worked Example 7.2 say about dropping it?

**A4 (★★).** Let $X_1,X_2,\dots$ be i.i.d. $\mathrm{Exp}(1)$. Prove that
$$\limsup_{n\to\infty}\frac{X_n}{\log n}=1\qquad\text{almost surely.}$$
(Use both Borel–Cantelli lemmas, and be careful to reduce to countably many thresholds.)

**A5 (★★).** For random variables on a fixed probability space define $d(X,Y):=\mathbb{E}\big[\tfrac{|X-Y|}{1+|X-Y|}\big]$. Show that $d$ is a metric on equivalence classes modulo almost-sure equality, and that $d(X_n,X)\to0$ if and only if $X_n\to_{\mathrm p}X$. Explain what this says about Figure 7.1 (specifically: which of the four modes is *not* induced by a metric on this space?).

**A6 (★★).** Suppose $\varphi_X(t_0)=1$ for some $t_0\ne0$. Prove that $\mathbb{P}\big(X\in\tfrac{2\pi}{t_0}\mathbb{Z}\big)=1$. Deduce that if $|\varphi_X(t)|=1$ for two incommensurable values of $t$ then $X$ is almost surely constant. Relate this to the remark in Worked Example 7.13 about $|1-p+pe^{it}|$.

**A7 (★★).** Show that Lindeberg's condition (Theorem 7.32) implies uniform asymptotic negligibility:
$$\max_{j\le k_n}\frac{\mathbb{E}[X_{n,j}^2]}{s_n^2}\longrightarrow0 .$$
Then verify directly that the "one whale, many minnows" array of §7.5.2 violates this.

**A8 (★★★).** Suppose $X_n\to_{\mathrm d}X$ and $\{X_n\}$ is uniformly integrable, i.e. $\lim_{M\to\infty}\sup_n\mathbb{E}\big[|X_n|\mathbf 1\{|X_n|>M\}\big]=0$. Prove $\mathbb{E}[X_n]\to\mathbb{E}[X]$ (in particular $\mathbb{E}|X|<\infty$). Show by example that uniform integrability cannot be dropped.

**A9 (★★★).** Let $X_1,X_2,\dots$ be i.i.d. standard Cauchy and $S_n=\sum_{i\le n}X_i$. Prove that there exist *no* constants $a_n\in\mathbb{R}$, $b_n>0$ with $(S_n-a_n)/b_n\to_{\mathrm d}\mathcal N(0,1)$. (Work with characteristic functions and compare the moduli at two values of $t$.) Contrast with the fact that $b_n=n$, $a_n=0$ gives a perfectly good — non-Gaussian — limit.

## Part B — Computational labs

Report seeds, sample sizes, and a consistent number of decimals. Where a quantity can be computed exactly, do so rather than simulating.

**B1 (★).** Implement the sliding block $X_n=\mathbf 1_{I_{k,j}}$ of Worked Example 7.1 and the moving spike $Y_n=n\mathbf 1_{[0,1/n)}$ of Worked Example 7.2 on $\Omega=[0,1]$.
(a) Verify numerically that $\mathbb{P}(|X_n|>1/2)=2^{-\lfloor\log_2n\rfloor}$ and evaluate it at $n=1000$.
(b) For $\omega=0.3$, list all $n\le200$ with $X_n(\omega)=1$.
(c) Confirm $\mathbb{E}|Y_n|=1$ for all $n$ while $Y_n\to0$ pointwise on $(0,1]$, and construct a variant with $\mathbb{E}|Y_n|\to\infty$.
(d) Explain, in two sentences, which arrow of Figure 7.1 each part kills.

**B2 (★★).** Measure the CLT convergence rate. For parents $\mathrm{Bernoulli}(0.01)$, $\mathrm{Exp}(1)$, and $t_3$, compute
$$D_n=\sup_x\big|\mathbb{P}(Z_n\le x)-\Phi(x)\big|,\qquad Z_n=\frac{S_n-n\mu}{\sigma\sqrt n},$$
for $n\in\{1,5,30,100,1000,10^4\}$. Use the exact binomial CDF for the first parent and the exact $\mathrm{Gamma}(n,1)$ CDF for the second. For $t_3$, avoid simulation: use $\varphi_{t_3}(t)=(1+\sqrt3|t|)e^{-\sqrt3|t|}$ to get $\varphi_{Z_n}(t)=(1+|t|/\sqrt n)^ne^{-\sqrt n|t|}$ and invert with the Gil–Pelaez formula
$$F(x)=\tfrac12-\tfrac1\pi\int_0^\infty\frac{\operatorname{Im}\big(e^{-itx}\varphi(t)\big)}{t}\,dt .$$
Tabulate $D_n$ and $\sqrt n\,D_n$. Which parents show a stable $\sqrt n D_n$? For $\mathrm{Exp}(1)$, compare the limiting constant to the Edgeworth prediction $\gamma\phi(0)/6$ with $\gamma=2$.

**B3 (★★).** Draw $10^5$ i.i.d. standard Cauchy variates and plot the running mean and running median against $\log n$. Repeat with a $\mathcal N(0,1)$ parent. Report both statistics at $n\in\{10,10^2,10^3,10^4,10^5\}$, and report $\max_{10^3\le n\le10^5}|\bar X_n|$. Then verify the theory of Worked Example 7.16 by checking that the empirical distribution of $\bar X_n$ over $10^4$ independent replications is indistinguishable (by a KS test) from that of a single draw.

**B4 (★★).** Validate the delta method for the logit. For $p\in\{0.1,0.5\}$ and $n\in\{50,200,1000,10^4\}$, simulate $M=200{,}000$ binomial draws, discard degenerate replications, and compare $\sqrt n\times\operatorname{sd}(\operatorname{logit}\hat p_n)$ to the theoretical $1/\sqrt{p(1-p)}$. Report the discarded fraction. At which $n$ does the ratio fall below $1.01$, and how does that compare with the $np(1-p)\ge10$ heuristic?

**B5 (★★★).** Minibatch gradient noise. Generate $N=50{,}000$ points from $y=2x+\epsilon$, $\epsilon\sim\mathcal N(0,1)$, once with $x\sim\mathcal N(0,1)$ and once with $x\sim t_3$. At $w=1.5$, form per-example gradients $g_i=x_i(wx_i-y_i)$. For $B\in\{1,8,32,256\}$ draw $M=200{,}000$ minibatches with replacement, standardize the minibatch mean, and compute its KS distance to $\Phi$. Report the per-example excess kurtosis for each feature distribution. Does the KS distance decay like $B^{-1/2}$ in both cases? What does your answer imply for the diffusion approximation of §7.9.2?

## Part C — Paper-style derivations

**C1 (★★★).** Prove the i.i.d. central limit theorem via characteristic functions. Your write-up must (i) state the theorem with full hypotheses; (ii) give the proof in numbered steps; and (iii) include an explicit list of every analytic fact invoked — including the ones people usually leave silent — with a one-line justification or a citation for each.

**C2 (★★).** Prove the delta method (Theorem 7.37) from first principles, then apply it to derive the standard error of a log-odds ratio. Specifically, let $\hat p_1=a/(a+b)$ and $\hat p_2=c/(c+d)$ be independent binomial proportions from a $2\times2$ table with cell counts $a,b,c,d$. Show
$$\widehat{\operatorname{se}}\big(\log\widehat{\mathrm{OR}}\big)=\sqrt{\tfrac1a+\tfrac1b+\tfrac1c+\tfrac1d},$$
state exactly which hypotheses you used, and explain what goes wrong when any cell count is $0$.

**C3 (★★★).** State and prove Slutsky's theorem. Then use it to justify the interval $\hat\theta_n\pm z_{1-\alpha/2}\hat\sigma_n/\sqrt n$. Your justification must be explicit about (i) what convergence statement is being asserted, (ii) what is held fixed as $n\to\infty$, (iii) what is *not* promised at finite $n$, and (iv) at least two realistic machine-learning situations in which the hypotheses fail. Cite measured evidence from this chapter for (iii).

---

# Appendix A — Self-check answers

**7.1.** (1) $X_n=n\mathbf 1_{[0,1/n)}$ on $([0,1],\lambda)$: $X_n\to0$ pointwise on $(0,1]$ but $\mathbb{E}[X_n^2]=n\to\infty$. (2) False. The same $X_n$ has $X_n\to_{\mathrm d}0$ (a constant) yet $\mathbb{E}|X_n|=1$ for every $n$. Convergence in distribution to a constant gives convergence in probability (Proposition 7.10), never $L^1$. (3) It equals $1$. The indices $n\in\{128,\dots,255\}$ form a complete sweep of the dyadic level $k=7$, whose blocks cover $[0,1)$; so for every $\omega$ there is some $n\ge100$ with $X_n(\omega)=1>1/2$. (4) Yes: $\mathbb{P}(|(X_n+Y_n)-(X+Y)|>\varepsilon)\le\mathbb{P}(|X_n-X|>\varepsilon/2)+\mathbb{P}(|Y_n-Y|>\varepsilon/2)\to0$. The §7.1.4 counterexample does not apply because there the hypotheses were only distributional, so the *joint* law of $(X_n,Y_n)$ was unconstrained; convergence in probability is a statement about the joint law.

**7.2.** (1) $\sum1/n=\infty$ and the events are independent, so $\mathbb{P}(A_n\text{ i.o.})=1$ by Theorem 7.12. With $\mathbb{P}(A_n)=1/n^2$ the sum converges and Theorem 7.11 gives $0$. (2) BC-I only needs an *upper* bound on the probability of a union, which subadditivity supplies for free; BC-II needs a *lower* bound on the probability that at least one of infinitely many events occurs, and without independence the events can pile up on top of each other (Worked Example 7.9). (3) Yes. Given $\varepsilon>0$, for all $n>1/\varepsilon$ we have $\{|X_n|>\varepsilon\}\subseteq\{|X_n|>1/n\}$, so $\sum_n\mathbb{P}(|X_n|>\varepsilon)$ is finite (finitely many terms bounded by $1$, plus a tail bounded by $\sum1/n^2$). Proposition 7.14 applies. (4) $X_n=\mathbf 1\{U<1/n\}$ with $U\sim\mathrm{Unif}(0,1)$: $\sum_n\mathbb{P}(X_n>1/2)=\sum1/n=\infty$, yet $X_n(\omega)\to0$ for every $\omega>0$.

**7.3.** (1) In the Markov step. The fourth moment gives $\mathbb{E}[S_n^4]\le Cn^2$, hence a bound $C/(n^2\varepsilon^4)$, which is summable and feeds Borel–Cantelli. A second moment gives only $\sigma^2/(n\varepsilon^2)$, which is not summable — that is exactly why Theorem 7.16 is a *weak* law. (2) Independence: the proof used only $\operatorname{Var}(S_n)=n\sigma^2$, which holds for pairwise uncorrelated variables. (3) $\mathbb{E}[X]=1+\int_1^\infty t^{-3/2}dt=3<\infty$, so $\bar X_n\to_{\mathrm{as}}3$ by Theorem 7.19. But $\mathbb{E}[X^2]=\int_0^\infty2t\,\mathbb{P}(X>t)\,dt=1+2\int_1^\infty t^{-1/2}dt=\infty$, so there is no variance and no CLT at rate $\sqrt n$. (4) Because the null set in Corollary 7.21 depends on $h$, and $\hat h_n$ is chosen using the data; over an uncountable $\mathcal H$ the union of null sets need not be null, nor even measurable.

**7.4.** (1) $\varphi_X(t)=\frac{1}{2}\int_{-1}^1e^{itx}dx=\frac{\sin t}{t}$, whose expansion is $1-t^2/6+t^4/120-\cdots$; and indeed $\mathbb{E}[X^2]=1/3$, so Proposition 7.26 predicts $1-\tfrac{t^2}{2}\cdot\tfrac13$. (2) $|e^{itx}|=1$ for all real $t,x$, so the integrand is bounded by $1$ regardless of the law. By contrast $e^{tx}$ is unbounded, and Jensen gives $M_X(t)\ge e^{t\mathbb{E}X}$, so $M_X$ typically exceeds $1$ and may be $+\infty$. (3) The corner at $t=0$: $e^{-|t|}$ has one-sided derivatives $\mp1$ and no derivative. Proposition 7.26 says $\varphi'(0)$ exists (and equals $i\mathbb{E}X$) when $\mathbb{E}|X|<\infty$; the corner is the visible signature of $\mathbb{E}|C|=\infty$. (4) Iterating, $\varphi(t)=\varphi(t/2^k)$ for every $k$; letting $k\to\infty$ and using continuity (Proposition 7.23(3)) gives $\varphi(t)=\varphi(0)=1$ for all $t$. Hence $\varphi_X\equiv\varphi_{\delta_0}$, so $X=0$ almost surely by Corollary 7.29.

**7.5.** (1) Step 2, the expansion $\varphi(s)=1-s^2/2+o(s^2)$, which is Proposition 7.26 with $\mathbb{E}[Y^2]<\infty$. For a Cauchy parent $\varphi(t)=e^{-|t|}=1-|t|+O(t^2)$: the leading correction is of order $|t|$, not $t^2$, so the correct normalization is $n$ rather than $\sqrt n$ and the limit is Cauchy, not Gaussian. (2) Need $0.4748\times5/\sqrt n\le0.01$, i.e. $n\ge(237.4)^2=56{,}358.76$, so $n\ge56{,}359$. (3) After dividing by the total standard deviation, no individual summand contributes a non-vanishing share of the total variance through large values. (4) A right-skewed parent puts more mass far above the mean than a Gaussian does; the leading Edgeworth correction $-\frac{\gamma}{6\sqrt n}(x^2-1)\phi(x)$ with $\gamma>0$ is negative for $|x|>1$ in the CDF, i.e. the true upper tail exceeds the normal one (Worked Example 7.19 measures a factor of $3$ at $z=3$, $n=30$).

**7.6.** (1) $X_n\to_{\mathrm d}X$ in $\mathbb{R}^d$ iff $\langle u,X_n\rangle\to_{\mathrm d}\langle u,X\rangle$ for every $u\in\mathbb{R}^d$. Coordinate marginals correspond only to $u=e_1,\dots,e_d$, and those do not determine a joint law: $(Z_1,Z_2)$ independent standard normal and $(Z_1,Z_1)$ have identical marginals and different joint laws. Cramér–Wold needs *all* directions. (2) $g(x)=\log x$, $g'(\mu)=1/\mu$, so the asymptotic variance is $\sigma^2/\mu^2$ — the squared coefficient of variation. (3) $g'(0.3)=1-0.6=0.4\ne0$, so the first-order method applies; asymptotic variance $=(0.4)^2\times0.3\times0.7=0.0336$. (4) Because $g'(\rho)=1/(1-\rho^2)$ is exactly the reciprocal of the asymptotic standard deviation $(1-\rho^2)$, so the product $g'(\rho)^2\times(1-\rho^2)^2=1$ is free of $\rho$. That is the defining recipe for a variance-stabilizing transform.

**7.7.** (1) $X_n=Z$ and $Y_n=(-1)^nZ$ with $Z\sim\mathcal N(0,1)$: both converge in distribution to $\mathcal N(0,1)$, but $X_n+Y_n$ alternates between $2Z\sim\mathcal N(0,4)$ and $0$, so its law does not converge. (2) Because the conclusion is about $\mathbb{E}[f(h(X_n))]$, and $f\circ h$ must be continuous $P_X$-almost everywhere to be an admissible test function. Concretely, $X_n\equiv1/n\to_{\mathrm d}0$ and $h=\mathbf 1\{x>0\}$ give $h(X_n)\equiv1$ but $h(0)=0$: the limit law puts all its mass on the single discontinuity. (3) That the *studentized* statistic $\sqrt n(\hat\theta_n-\theta)/\hat\sigma_n$ converges in distribution to $\mathcal N(0,1)$, hence that the coverage probability converges to $1-\alpha$ — for a fixed data-generating law, as $n\to\infty$. Nothing is claimed at any particular $n$. (4) With probability $0.99^{10}=0.9044$ all ten observations equal $0$, so $\bar X_n=s_n=0$ and the interval degenerates to $\{0\}$, which misses $p=0.01$; coverage is therefore about $1-0.99^{10}=0.0956$, matching the measured $0.0949$.

**7.8.** (1) Because $F_n$ and $F$ are right-continuous and monotone, so the supremum over $\mathbb{R}$ equals the supremum over the countable set $\mathbb{Q}$ together with the (countably many) jump points; a countable supremum of random variables is measurable by Proposition 1.15. (2) Over a grid of $m$ points, Hoeffding plus the union bound gives $2m\,e^{-2n\varepsilon^2}$, plus a separate discretization error for the $x$ between grid points. DKW has no $m$ at all — the supremum over uncountably many $x$ costs nothing. (3) $\sqrt n\,D_n\to_{\mathrm d}\sup_{t\in[0,1]}|B^\circ(t)|$, the Kolmogorov distribution. It is free of $F$ because for continuous $F$ the time change $t=F(x)$ maps the problem to the uniform case, and a supremum is invariant under reparametrization. (4) No. The empirical measure $\hat P_n$ is purely atomic and has no density at all; and even for smoothed estimators, density estimation converges at a slower rate (e.g. $n^{-2/5}$ under two derivatives) and is not distribution-free.

**7.9.** (1) The limit is $\mathcal N(0,\Sigma(\theta))$ where $\Sigma$ is the empirical covariance of per-example gradients; the batch size $B$ plays the role of $n$, with $\theta$ and the training set held fixed. (2) Because the SGD increment is $-\eta\hat g_B$, whose noise covariance is $\eta^2\Sigma/B$; the diffusion coefficient in the SDE therefore depends on $\eta/B$. The dataset size $N$ enters only through $\Sigma$ itself. (3) Any two of: the estimator is data-adaptive (hyperparameters, architecture, early stopping all use the same data, so $\hat\theta_n$ is not the estimator whose asymptotics were computed); $p$ is not fixed and negligible relative to $n$; the data are not i.i.d.; or the quantity of interest is a tail probability rather than a mean. (4) Because $\hat p=0$ makes the plug-in standard error $\sqrt{\hat p(1-\hat p)/n}$ exactly zero. The correct $95\%$ upper bound is $1-0.05^{1/n}=2.9953\times10^{-4}$, essentially the rule of three, $3/n$.

---

# Solutions to Exercises

## Part A

**A1.** *Thinking.* $L^2$ convergence is convergence in a norm, and both the mean and the second moment are continuous functionals for that norm. The whole solution is two applications of a triangle inequality.

For the mean, Jensen (or Cauchy–Schwarz with the constant function $1$) gives
$$\big|\mathbb{E}[X_n]-\mathbb{E}[X]\big|=\big|\mathbb{E}[X_n-X]\big|\le\mathbb{E}|X_n-X|=\|X_n-X\|_1\le\|X_n-X\|_2\to0 .$$
For the second moment, the reverse triangle inequality in $L^2$ gives $\big|\|X_n\|_2-\|X\|_2\big|\le\|X_n-X\|_2\to0$, so $\mathbb{E}[X_n^2]=\|X_n\|_2^2\to\|X\|_2^2=\mathbb{E}[X^2]$. Hence
$$\operatorname{Var}(X_n)=\mathbb{E}[X_n^2]-(\mathbb{E}[X_n])^2\longrightarrow\mathbb{E}[X^2]-(\mathbb{E}[X])^2=\operatorname{Var}(X).$$
*Counterexample for convergence in probability alone.* On $([0,1],\lambda)$ take $X_n=n^2\mathbf 1_{[0,1/n)}$. Then $\mathbb{P}(|X_n|>\varepsilon)=1/n\to0$, so $X_n\to_{\mathrm p}0$; but $\mathbb{E}[X_n]=n\to\infty$ and $\operatorname{Var}(X_n)=n^4/n-n^2=n^3-n^2\to\infty$. The moral is the one Figure 7.1 draws: convergence in probability controls *where* the mass is, not *how heavy* it is.

**A2.** (a) Let $X\sim\mathrm{Unif}\{-1,+1\}$ and set $X_n:=-X$ for every $n$. Since the law is symmetric, $-X\overset{d}{=}X$, so $X_n\to_{\mathrm d}X$ trivially (the laws are equal, not merely convergent). But $|X_n-X|=|{-2X}|=2\ge1$ with probability $1$, for every $n$. This is Worked Example 7.7 in its starkest form: distributional convergence is a statement about $P_X$, and two variables can share a law while never being close.

(b) $Y_n:=n^2\mathbf 1_{[0,1/n)}$ on $([0,1],\lambda)$. For any $\omega>0$, $Y_n(\omega)=0$ once $n>1/\omega$, so $Y_n\to0$ pointwise on $(0,1]$, i.e. almost surely. Yet $\mathbb{E}|Y_n|=n^2\cdot\frac1n=n\to\infty$.

**A3.** *Thinking.* The natural move with a "converges in probability, so split on a good event" problem is always the same: choose $\varepsilon$, split the expectation on $\{|X_n-X|\le\varepsilon\}$ and its complement, use the target tolerance on the good event and the uniform bound on the bad one.

First, $|X|\le M$ almost surely. To see this, take a subsequence with $X_{n_k}\to X$ a.s. (Theorem 7.15); on that event $|X|=\lim_k|X_{n_k}|\le M$. Now fix $\varepsilon>0$ and $p\ge1$. On $\{|X_n-X|>\varepsilon\}$ we have $|X_n-X|\le|X_n|+|X|\le2M$, so
$$\mathbb{E}|X_n-X|^p=\mathbb{E}\big[|X_n-X|^p\mathbf 1\{|X_n-X|\le\varepsilon\}\big]+\mathbb{E}\big[|X_n-X|^p\mathbf 1\{|X_n-X|>\varepsilon\}\big]\le\varepsilon^p+(2M)^p\,\mathbb{P}(|X_n-X|>\varepsilon).$$
Let $n\to\infty$: $\limsup_n\mathbb{E}|X_n-X|^p\le\varepsilon^p$. Since $\varepsilon>0$ was arbitrary, the limit is $0$.

Boundedness is used exactly once, to bound the integrand on the bad event by a constant. Worked Example 7.2 shows this is not a technicality: $Y_n=n\mathbf 1_{[0,1/n)}$ converges in probability (indeed a.s.) to $0$, and the bad event has probability $1/n\to0$, but the integrand there is $n$, and $n\cdot\frac1n=1$ refuses to vanish. The general repair is uniform integrability (Exercise A8), of which boundedness is the crudest sufficient condition.

**A4.** *Thinking.* $\limsup$ statements about i.i.d. sequences are Borel–Cantelli statements: "$\limsup X_n/\log n\ge c$" means "$X_n>c\log n$ infinitely often", and "$\le c$" means "only finitely often". So we need $\sum_n\mathbb{P}(X_n>c\log n)$ for each $c$, and the reduction to countably many $c$ at the end.

For $\mathrm{Exp}(1)$, $\mathbb{P}(X_n>c\log n)=e^{-c\log n}=n^{-c}$.

*Upper bound.* Fix $c>1$. Then $\sum_nn^{-c}<\infty$, so by Borel–Cantelli I (Theorem 7.11) $\mathbb{P}(X_n>c\log n\text{ i.o.})=0$; on the complementary probability-one event, $X_n\le c\log n$ for all large $n$, hence $\limsup_nX_n/\log n\le c$. Intersect over $c\in\{1+1/m:m\in\mathbb{N}\}$, a countable family of probability-one events, to get $\limsup\le1$ a.s.

*Lower bound.* Fix $c<1$. Then $\sum_nn^{-c}=\infty$, and the events $\{X_n>c\log n\}$ are independent because they involve distinct $X_n$ (Definition 1.23). Borel–Cantelli II (Theorem 7.12) gives $\mathbb{P}(X_n>c\log n\text{ i.o.})=1$, hence $\limsup_nX_n/\log n\ge c$ a.s. Intersect over $c\in\{1-1/m\}$ to get $\limsup\ge1$ a.s.

Combining, $\limsup_nX_n/\log n=1$ almost surely. Note the shape of the argument: BC-I gives upper bounds and needs nothing; BC-II gives lower bounds and needs independence; and the countable-intersection step at the end is not optional, because an uncountable intersection of probability-one events can be empty.

**A5.** *Thinking.* The function $\psi(u)=u/(1+u)$ is increasing, bounded by $1$, and subadditive on $[0,\infty)$; those three properties are exactly what a metric proof needs, and boundedness is what makes the expectation always finite.

*Metric axioms.* $d\ge0$ is clear, and $d(X,Y)=0$ forces $\psi(|X-Y|)=0$ a.s. hence $X=Y$ a.s. Symmetry is clear. For the triangle inequality, note $\psi$ is increasing and subadditive: $\psi(u+v)=\frac{u+v}{1+u+v}=\frac{u}{1+u+v}+\frac{v}{1+u+v}\le\psi(u)+\psi(v)$. Applying this with $u=|X-Z|$, $v=|Z-Y|$ and monotonicity,
$$\psi(|X-Y|)\le\psi(|X-Z|+|Z-Y|)\le\psi(|X-Z|)+\psi(|Z-Y|),$$
and taking expectations gives $d(X,Y)\le d(X,Z)+d(Z,Y)$.

*Equivalence with convergence in probability.* Suppose $X_n\to_{\mathrm p}X$. Fix $\varepsilon>0$ and split:
$$d(X_n,X)=\mathbb{E}\big[\psi(|X_n-X|)\mathbf 1\{|X_n-X|\le\varepsilon\}\big]+\mathbb{E}\big[\psi(\cdot)\mathbf 1\{>\varepsilon\}\big]\le\psi(\varepsilon)+\mathbb{P}(|X_n-X|>\varepsilon),$$
using $\psi\le1$. Take $\limsup$, then $\varepsilon\downarrow0$. Conversely, if $d(X_n,X)\to0$ then by Markov's inequality applied to the non-negative variable $\psi(|X_n-X|)$,
$$\mathbb{P}(|X_n-X|>\varepsilon)=\mathbb{P}\big(\psi(|X_n-X|)>\psi(\varepsilon)\big)\le\frac{d(X_n,X)}{\psi(\varepsilon)}\to0,$$
where the first equality uses that $\psi$ is strictly increasing.

*What this says about Figure 7.1.* Convergence in probability is *metrizable*: it is topological convergence in $(d)$. So is $L^p$ convergence (by the $L^p$ norm) and, on a suitable space of laws, convergence in distribution (by the Lévy–Prokhorov metric). Almost sure convergence is the odd one out — it is **not** induced by any topology on this space. The standard proof uses a general fact about topologies: *in any topological space, $x_n\to x$ if and only if every subsequence of $(x_n)$ has a further subsequence converging to $x$* (if $x_n\not\to x$, some neighbourhood of $x$ excludes a whole subsequence, and no further subsequence of that one can converge to $x$). So if a.s. convergence were the convergence of some topology, then "every subsequence has a further subsequence converging a.s." would imply a.s. convergence — and Worked Example 7.1 shows it does not (the sliding block converges in probability, so every subsequence has an a.s.-convergent further subsequence, yet the sequence converges a.s. nowhere).

**A6.** *Thinking.* $\varphi(t_0)=1$ says a complex average of unit-modulus numbers has modulus $1$, which can only happen if they all point the same way. Making that rigorous is a single non-negativity argument.

Taking real parts, $1=\operatorname{Re}\varphi_X(t_0)=\mathbb{E}[\cos(t_0X)]$, so
$$\mathbb{E}\big[1-\cos(t_0X)\big]=0 .$$
The integrand is non-negative, so it vanishes almost surely (Chapter 2: a non-negative function with zero integral is $0$ a.e.). Hence $\cos(t_0X)=1$ a.s., i.e. $t_0X\in2\pi\mathbb{Z}$ a.s., i.e. $X\in\frac{2\pi}{t_0}\mathbb{Z}$ a.s.

*Deduction.* Suppose $|\varphi_X(t_1)|=|\varphi_X(t_2)|=1$ with $t_1/t_2$ irrational. Write $\varphi_X(t_j)=e^{i\alpha_j}$; then $\varphi_{X-\alpha_1/t_1}(t_1)=1$, so by the above $X$ is supported on a coset $a_1+\frac{2\pi}{t_1}\mathbb{Z}$, and similarly on $a_2+\frac{2\pi}{t_2}\mathbb{Z}$. The intersection of two such cosets with incommensurable spacings contains at most one point (two points would force $\frac{2\pi k}{t_1}=\frac{2\pi m}{t_2}$ for integers $k,m$ not both zero, making $t_1/t_2$ rational). So $X$ is a.s. constant.

*Relation to Worked Example 7.13.* There, $|1-p+pe^{it}|=1$ exactly when $\cos t=1$, i.e. $t\in2\pi\mathbb{Z}$ — a lattice of $t$'s, all commensurable, consistent with $X$ being supported on $\{0,1\}\subset\mathbb{Z}$ rather than being constant. The size of the set $\{t:|\varphi(t)|=1\}$ is exactly the dual of the lattice supporting $X$: a discrete subgroup means a lattice law, $\{0\}$ alone means a non-lattice law, and all of $\mathbb{R}$ means a constant.

**A7.** For each $j$, split at $\varepsilon s_n$:
$$\mathbb{E}[X_{n,j}^2]=\mathbb{E}\big[X_{n,j}^2\mathbf 1\{|X_{n,j}|\le\varepsilon s_n\}\big]+\mathbb{E}\big[X_{n,j}^2\mathbf 1\{|X_{n,j}|>\varepsilon s_n\}\big]\le\varepsilon^2s_n^2+\mathbb{E}\big[X_{n,j}^2\mathbf 1\{|X_{n,j}|>\varepsilon s_n\}\big].$$
The second term is at most $s_n^2L_n(\varepsilon)$, because $L_n(\varepsilon)$ is the *sum* over $j$ of exactly these quantities divided by $s_n^2$, and each summand is non-negative. Dividing by $s_n^2$ and maximizing over $j$,
$$\max_{j\le k_n}\frac{\mathbb{E}[X_{n,j}^2]}{s_n^2}\le\varepsilon^2+L_n(\varepsilon).$$
Let $n\to\infty$ (so $L_n(\varepsilon)\to0$), then $\varepsilon\downarrow0$.

*The whale array.* $X_{n,1}\sim\mathrm{Unif}\{-n,n\}$, $X_{n,j}\sim\mathrm{Unif}\{-1,1\}$ for $2\le j\le n$. Then $\mathbb{E}[X_{n,1}^2]=n^2$ and $s_n^2=n^2+(n-1)$, so $\mathbb{E}[X_{n,1}^2]/s_n^2=n^2/(n^2+n-1)\to1$, not $0$. Uniform asymptotic negligibility fails, hence so does Lindeberg's condition, and indeed $S_n/s_n\to_{\mathrm d}\mathrm{Unif}\{-1,1\}$: the sum inherits the whale's shape.

**A8.** *Thinking.* $x\mapsto x$ is continuous but unbounded, so it is not an admissible test function for Definition 7.4. The fix is to truncate it into a bounded continuous function, apply the definition there, and use uniform integrability to control the truncation error *uniformly in $n$* — which is precisely what "uniform" buys.

Define, for $M>0$, the bounded continuous truncation
$$\psi_M(x):=\begin{cases}x,&|x|\le M\\ M\operatorname{sign}(x),&|x|>M.\end{cases}$$
Note $|x-\psi_M(x)|\le|x|\mathbf 1\{|x|>M\}$.

*Step 1: $\mathbb{E}|X|<\infty$.* For each $M$, $|\psi_M|$ is bounded continuous, so $\mathbb{E}|\psi_M(X_n)|\to\mathbb{E}|\psi_M(X)|$. Also $\mathbb{E}|\psi_M(X_n)|\le\mathbb{E}|X_n|\le M+\sup_n\mathbb{E}[|X_n|\mathbf 1\{|X_n|>M\}]=:K<\infty$ for a fixed $M$ large enough that the sup is $\le1$. Hence $\mathbb{E}|\psi_M(X)|\le K$ for every $M$, and monotone convergence as $M\uparrow\infty$ gives $\mathbb{E}|X|\le K<\infty$.

*Step 2: the estimate.* Write
$$\big|\mathbb{E}[X_n]-\mathbb{E}[X]\big|\le\underbrace{\mathbb{E}\big|X_n-\psi_M(X_n)\big|}_{(\mathrm{i})}+\underbrace{\big|\mathbb{E}[\psi_M(X_n)]-\mathbb{E}[\psi_M(X)]\big|}_{(\mathrm{ii})}+\underbrace{\mathbb{E}\big|X-\psi_M(X)\big|}_{(\mathrm{iii})}.$$
Term (i) is $\le\sup_n\mathbb{E}[|X_n|\mathbf 1\{|X_n|>M\}]=:\eta(M)$, which $\to0$ as $M\to\infty$ by uniform integrability. Term (iii) is $\le\mathbb{E}[|X|\mathbf 1\{|X|>M\}]\to0$ by dominated convergence (Ch. 2), using Step 1. Term (ii) $\to0$ as $n\to\infty$ for each fixed $M$, since $\psi_M$ is bounded and continuous. So
$$\limsup_n\big|\mathbb{E}[X_n]-\mathbb{E}[X]\big|\le\eta(M)+\mathbb{E}[|X|\mathbf 1\{|X|>M\}]\xrightarrow[M\to\infty]{}0 .$$

*Uniform integrability cannot be dropped.* $X_n=n\mathbf 1_{[0,1/n)}$ has $X_n\to_{\mathrm d}0$ but $\mathbb{E}[X_n]=1\not\to0$. Here $\mathbb{E}[|X_n|\mathbf 1\{|X_n|>M\}]=1$ for every $n>M$, so the family is not uniformly integrable.

**A9.** *Thinking.* Characteristic functions convert the question into an algebraic identity, and the modulus kills the centering constants — which is the whole trick, since otherwise you would have to track $a_n$.

By Proposition 7.24 and Worked Example 7.16, $\varphi_{S_n}(t)=\varphi_C(t)^n=e^{-n|t|}$. Hence, by Proposition 7.23(4), with $T_n:=(S_n-a_n)/b_n$,
$$\varphi_{T_n}(t)=e^{-ita_n/b_n}\,e^{-n|t|/b_n},\qquad\text{so}\qquad\big|\varphi_{T_n}(t)\big|=e^{-n|t|/b_n}.$$
Suppose $T_n\to_{\mathrm d}\mathcal N(0,1)$. By Lévy's continuity theorem, direction (1) (Theorem 7.30), $\varphi_{T_n}(t)\to e^{-t^2/2}$ for every $t$, hence also $|\varphi_{T_n}(t)|\to e^{-t^2/2}$ (the modulus is continuous). Write $c_n:=n/b_n>0$. Then
$$e^{-c_n|t|}\to e^{-t^2/2}\quad\text{for every }t,\qquad\text{i.e.}\qquad c_n|t|\to t^2/2 .$$
At $t=1$ this forces $c_n\to1/2$. At $t=2$ it forces $2c_n\to2$, i.e. $c_n\to1$. Contradiction. So no such $a_n,b_n$ exist.

*Contrast.* Take $a_n=0$, $b_n=n$. Then $|\varphi_{T_n}(t)|=e^{-|t|}$ and in fact $\varphi_{T_n}(t)=e^{-|t|}$ exactly, so $T_n=\bar C_n\overset{d}{=}C_1$ for every $n$ — a genuine (indeed exact) limit, just not a Gaussian one. This is the $\alpha=1$ case of the stable limit laws: without a finite variance the CLT's normalization $\sqrt n$ and its Gaussian limit are both wrong, and the correct normalization is $n^{1/\alpha}$ with an $\alpha$-stable limit. Every step of Theorem 7.31's proof that could fail, fails here: the expansion (7.2) is unavailable because $\mathbb{E}[C^2]=\infty$.

## Part B

**B1.** (a) At dyadic level $k$ the block has width $2^{-k}$, and $n=2^k+j$ means $k=\lfloor\log_2n\rfloor$, so $\mathbb{P}(|X_n|>1/2)=\lambda(I_{k,j})=2^{-\lfloor\log_2n\rfloor}$. At $n=1000$, $\lfloor\log_21000\rfloor=9$, giving $2^{-9}=0.001953125$.

(b) The spike times for $\omega$ are $n=2^k+\lfloor2^k\omega\rfloor$. For $\omega=0.3$ and $k=0,1,\dots,7$: $n=1,\ 2,\ 5,\ 10,\ 20,\ 41,\ 83,\ 166$. All eight are $\le200$; the next is $k=8$, $n=256+76=332$.

(c) $\mathbb{E}|Y_n|=n\cdot\lambda([0,1/n))=1$ for every $n$, while for each fixed $\omega\in(0,1]$, $Y_n(\omega)=0$ as soon as $n>1/\omega$. The variant $\tilde Y_n=n^2\mathbf 1_{[0,1/n)}$ has $\mathbb{E}|\tilde Y_n|=n\to\infty$ and still converges to $0$ a.s.

(d) Part (a)–(b) kill "in probability $\Rightarrow$ a.s.": the probability of a spike vanishes but every $\omega$ is hit at every dyadic level, so no trajectory converges. Part (c) kills "a.s. $\Rightarrow L^1$": pointwise convergence to $0$ coexists with $\mathbb{E}|Y_n|\equiv1$. Since the sliding block also converges in $L^1$ (its $L^1$ norm is $2^{-k}\to0$) while converging a.s. nowhere, the same two constructions also kill "$L^1\Rightarrow$ a.s.", closing all three missing arrows on the left of Figure 7.1.

**B2.** Exact values (no simulation, hence no seed):

| $n$ | Bern$(0.01)$ | Exp$(1)$ | $t_3$ | $\sqrt n\,D_n$: Bern | Exp | $t_3$ |
|---:|---:|---:|---:|---:|---:|---:|
| $1$ | $0.5300$ | $0.1587$ | $0.0869$ | $0.530$ | $0.159$ | $0.087$ |
| $5$ | $0.5399$ | $0.0596$ | $0.0442$ | $1.207$ | $0.133$ | $0.099$ |
| $30$ | $0.4487$ | $0.0243$ | $0.0192$ | $2.458$ | $0.133$ | $0.105$ |
| $100$ | $0.2358$ | $0.0133$ | $0.0108$ | $2.358$ | $0.133$ | $0.108$ |
| $1000$ | $0.0830$ | $0.0042$ | $0.0035$ | $2.626$ | $0.133$ | $0.109$ |
| $10^4$ | $0.0266$ | $0.0013$ | $0.0011$ | $2.656$ | $0.133$ | $0.110$ |

All three columns of $\sqrt nD_n$ stabilize, so all three parents obey the $n^{-1/2}$ rate. Exp$(1)$ is stable from $n=5$ onwards at $0.1330$; $t_3$ creeps up to $0.1100$; Bern$(0.01)$ is stable only from $n\approx10^3$ and at a value about twenty times larger.

The Edgeworth prediction for a non-lattice parent is that the leading error is $-\frac{\gamma}{6\sqrt n}(x^2-1)\phi(x)$, whose supremum over $x$ is attained at $x=0$ and equals $\frac{\gamma}{6}\phi(0)$. For $\mathrm{Exp}(1)$, $\gamma=2$, giving $\frac{2}{6}\cdot\frac{1}{\sqrt{2\pi}}=0.13298$ — the measured constant to four decimals. Two further checks worth making: (i) Bern$(0.5)$ has $\gamma=0$, so Edgeworth predicts a *faster* rate, yet the measured $\sqrt nD_n\to0.3989=1/\sqrt{2\pi}$; the error there is not Edgeworth but the lattice jump, half of whose largest value is $\approx1/\sqrt{2\pi n}$. (ii) $t_3$ has $\mathbb{E}|X|^3=\infty$, so Berry–Esseen's bound is $+\infty$ and yet the rate is a clean $n^{-1/2}$: the third-moment condition is sufficient, not necessary. A common student error here is to simulate the $t_3$ column; with $M$ replications the Monte Carlo error in a KS statistic is about $0.5/\sqrt M$, which at $M=10^6$ is $5\times10^{-4}$ — larger than $D_{10^4}=0.0011$ itself. Inverting the characteristic function is not a flourish; it is the only way to get the last row right.

**B3.** With numpy PCG64 seed $7$ and $N=10^5$ draws:

| $n$ | Cauchy mean | Cauchy median | $\mathcal N(0,1)$ mean |
|---:|---:|---:|---:|
| $10$ | $0.7238$ | $0.3832$ | $-0.20234$ |
| $10^2$ | $0.0034$ | $0.0354$ | $-0.17293$ |
| $10^3$ | $1.3140$ | $0.0635$ | $-0.07228$ |
| $10^4$ | $-0.5858$ | $0.0097$ | $-0.01232$ |
| $10^5$ | $0.4404$ | $-0.0117$ | $-0.00133$ |

and $\max_{10^3\le n\le10^5}|\bar X_n|=1.326$, attained near $n=1006$. The Gaussian column shrinks by roughly $\sqrt{10}$ per row, as the CLT predicts; the Cauchy mean column does not shrink at all; the Cauchy median column does, at the usual parametric rate.

*Exact-law check.* Worked Example 7.16 proves $\bar C_n\overset{d}{=}C_1$ for every $n$. Simulating $R=10^4$ independent replications of $\bar C_n$ (numpy PCG64, seed 4242) and comparing to the exact standard Cauchy CDF: at $n=10$, one-sample KS statistic $0.0099$ ($p=0.279$); at $n=1000$, $0.0044$ ($p=0.989$). Two-sample comparisons against $10^4$ single draws give $0.0135$ ($p=0.322$) and $0.0077$ ($p=0.928$). No detectable difference, as the theorem requires. Note the logic: a KS test *failing to reject* is weak evidence in general, but here we have a proof and are only checking the code.

**B4.** $M=200{,}000$ per cell, numpy PCG64 seed 11235; replications with $S_n\in\{0,n\}$ discarded.

| $p$ | $n$ | $\sqrt n\,\operatorname{sd}(\operatorname{logit}\hat p_n)$ | theory | ratio | discarded |
|---:|---:|---:|---:|---:|---:|
| $0.1$ | $50$ | $3.7239$ | $3.3333$ | $1.1172$ | $0.4930\%$ |
| $0.1$ | $200$ | $3.4517$ | $3.3333$ | $1.0355$ | $0.0000\%$ |
| $0.1$ | $1000$ | $3.3508$ | $3.3333$ | $1.0053$ | $0.0000\%$ |
| $0.1$ | $10^4$ | $3.3350$ | $3.3333$ | $1.0005$ | $0.0000\%$ |
| $0.5$ | $50$ | $2.0446$ | $2.0000$ | $1.0223$ | $0.0000\%$ |
| $0.5$ | $200$ | $2.0084$ | $2.0000$ | $1.0042$ | $0.0000\%$ |
| $0.5$ | $1000$ | $2.0016$ | $2.0000$ | $1.0008$ | $0.0000\%$ |
| $0.5$ | $10^4$ | $2.0018$ | $2.0000$ | $1.0009$ | $0.0000\%$ |

The ratio drops below $1.01$ at $n=200$ for $p=0.5$ (where $np(1-p)=50$) and at $n=1000$ for $p=0.1$ (where $np(1-p)=90$); at $n=200$, $p=0.1$ we have $np(1-p)=18$ and the ratio is still $1.036$. So the $np(1-p)\ge10$ heuristic is roughly the threshold for "within a few percent", not for "accurate". Note also the reporting trap: the discarded replications are precisely the ones where the estimate is infinite, so the tabulated standard deviation is *conditional on a non-degenerate answer* and understates the true dispersion; at $p=0.1$, $n=50$ the correct summary is that the logit estimator has no finite variance at all.

**B5.** $N=50{,}000$ training points, $M=200{,}000$ minibatches per cell, numpy PCG64 seed 1729, evaluated at $w=1.5$.

| features | per-example grad sd | excess kurtosis | $B=1$ | $B=8$ | $B=32$ | $B=256$ |
|---|---:|---:|---:|---:|---:|---:|
| $x\sim\mathcal N(0,1)$ | $1.203$ | $8.9$ | $0.1583$ | $0.0488$ | $0.0270$ | $0.0107$ |
| $x\sim t_3$ | $7.643$ | $3243.2$ | $0.3433$ | $0.2652$ | $0.1930$ | $0.1340$ |

Multiply each entry by $\sqrt B$. Gaussian features: $0.158,\ 0.138,\ 0.153,\ 0.171$ — roughly constant, so the decay is $B^{-1/2}$ as Berry–Esseen predicts. Heavy-tailed features: $0.343,\ 0.750,\ 1.092,\ 2.144$ — still growing, so the $B^{-1/2}$ regime has not started by $B=256$.

Implications for §7.9.2. The diffusion approximation replaces the minibatch gradient by $g+\sqrt{\Sigma/B}\,\xi$ with $\xi$ Gaussian. Row 1 says this is defensible at ordinary batch sizes when per-example gradients are light-tailed. Row 2 says it is not, and the failure is not a small correction: at $B=256$ the standardized noise is still $0.134$ away in Kolmogorov distance, which is roughly where the Gaussian approximation to $\mathrm{Bin}(30,0.01)$ sat when we called it unusable. Two further observations worth putting in a lab report. First, even the "Gaussian features" row has per-example excess kurtosis $8.9$ — the gradient $x(wx-y)$ is a product of near-Gaussians and hence heavy-tailed even when the inputs are not, so gradient noise is *never* as benign as the feature distribution. Second, if the per-example gradient had infinite variance, $\Sigma$ would not exist and Proposition 7.45 would have no content; the empirical literature on gradient tail indices in deep networks is precisely an investigation of whether that is happening.

## Part C

**C1.** *Statement.* Let $X_1,X_2,\dots$ be independent and identically distributed real random variables on a probability space $(\Omega,\mathcal F,\mathbb P)$, with $\mathbb{E}[X_1]=\mu$ and $\operatorname{Var}(X_1)=\sigma^2\in(0,\infty)$. Set $S_n=\sum_{i\le n}X_i$ and $Z_n=(S_n-n\mu)/(\sigma\sqrt n)$. Then $Z_n\to_{\mathrm d}\mathcal N(0,1)$.

*Proof in steps.* Standardize: $Y_i=(X_i-\mu)/\sigma$, i.i.d. with mean $0$, variance $1$, and $Z_n=n^{-1/2}\sum_iY_i$; write $\varphi=\varphi_{Y_1}$.

1. $\varphi_{Z_n}(t)=\varphi(t/\sqrt n)^n$.
2. $\varphi(s)=1-s^2/2+r(s)$ with $r(s)=o(s^2)$ as $s\to0$; hence $n|r(t/\sqrt n)|\to0$ for fixed $t$.
3. For $n\ge t^2/4$, Lemma 7.27 with $a_j=\varphi(t/\sqrt n)$ and $b_j=1-t^2/(2n)$ gives $|\varphi_{Z_n}(t)-(1-t^2/(2n))^n|\le n|r(t/\sqrt n)|\to0$.
4. $(1-t^2/(2n))^n\to e^{-t^2/2}$.
5. Therefore $\varphi_{Z_n}(t)\to e^{-t^2/2}$ pointwise; the limit is continuous at $0$ and is the characteristic function of $\mathcal N(0,1)$; Lévy continuity gives the conclusion.

*Every analytic fact used.*

| Fact | Where | Justification |
|---|---|---|
| $\mathbb{E}[e^{itX}]$ exists for every law and every real $t$ | throughout | $|e^{itX}|=1$, so the integrand is bounded; Chapter 2's integral of a bounded measurable function on a probability space is finite |
| Complex-valued integration: linearity and $|\int f|\le\int|f|$ | Steps 1–3 | Define by real and imaginary parts; the modulus inequality follows by writing $\int f=re^{i\theta}$ and taking real parts of $e^{-i\theta}\int f$ |
| $\mathbb{E}[\prod_jf_j(X_j)]=\prod_j\mathbb{E}[f_j(X_j)]$ for independent $X_j$ and bounded measurable $f_j$ | Step 1 | Definition 1.23 plus Fubini (Ch. 2) on the product measure |
| $\varphi_{aX}(t)=\varphi_X(at)$ | Step 1 | Proposition 7.23(4) |
| $|e^{ix}-\sum_{k\le2}(ix)^k/k!|\le\min\{|x|^3/6,\,x^2\}$ | Step 2 | Lemma 7.25, proved by iterated integration by parts |
| Dominated convergence to conclude $\mathbb{E}[\min\{|tY|^3/6,\,(tY)^2\}]=o(t^2)$ | Step 2 | Ch. 2, dominating function $Y^2\in L^1$ since $\operatorname{Var}(Y)<\infty$ |
| $|\varphi(t)|\le1$ | Step 3 | Proposition 7.23(1) |
| $|\prod a_j-\prod b_j|\le\sum|a_j-b_j|$ inside the closed unit disc | Step 3 | Lemma 7.27, telescoping |
| $|1-t^2/(2n)|\le1$ for $n\ge t^2/4$ | Step 3 | elementary; note it must be *checked*, since Lemma 7.27 requires it |
| $n\log(1-t^2/(2n))\to-t^2/2$ | Step 4 | $\log(1-u)=-u+O(u^2)$ for $|u|\le1/2$, with $u=t^2/(2n)\to0$ |
| $\varphi_{\mathcal N(0,1)}(t)=e^{-t^2/2}$ | Step 5 | Worked Example 7.14, via differentiation under the integral (licensed by DCT with dominating function $|z|\phi(z)$) and the ODE $\varphi'=-t\varphi$ |
| Lévy continuity theorem, direction (2) | Step 5 | Theorem 7.30; rests on Helly selection and a tightness argument. **Not proved here** — see §7.10 |
| Continuity of the limit at $t=0$ | Step 5 | $e^{-t^2/2}$ is smooth; this hypothesis is not decoration (see the $\mathcal N(0,n)$ example after Theorem 7.30) |

Two silent facts worth naming explicitly, because papers routinely omit them. First, $t$ is *fixed* while $n\to\infty$ in Steps 2–4: the convergence $\varphi_{Z_n}\to e^{-t^2/2}$ is pointwise, not uniform, and Lévy's theorem only needs pointwise. Second, $\sigma^2>0$ is used to divide by $\sigma$; if $\sigma^2=0$ then $X_1$ is a.s. constant and $Z_n$ is undefined.

**C2.** *Proof of the delta method.* This is Theorem 7.37; the proof is reproduced there in full and I will not repeat it, but the four moving parts are worth naming because each is a place people cut corners: (i) $T_n:=a_n(\hat\theta_n-\theta)\to_{\mathrm d}T$ implies $T_n=O_p(1)$; (ii) hence $\hat\theta_n\to_{\mathrm p}\theta$, so the estimator is consistent *as a consequence*, not as an extra assumption; (iii) differentiability at $\theta$ gives $g(\hat\theta_n)-g(\theta)=Dg(\theta)(\hat\theta_n-\theta)+\|\hat\theta_n-\theta\|\psi(\hat\theta_n-\theta)$ with $\psi$ continuous at $0$ and $\psi(0)=0$; (iv) $\|T_n\|\psi(\hat\theta_n-\theta)=O_p(1)o_p(1)=o_p(1)$, and adding an $o_p(1)$ does not change a distributional limit.

*Log-odds ratio.* Let group $1$ have $a$ successes and $b$ failures, $n_1=a+b$; group $2$ have $c$ successes and $d$ failures, $n_2=c+d$; groups independent, with true probabilities $p_1,p_2\in(0,1)$. The parameter is
$$\theta:=\log\mathrm{OR}=\log\frac{p_1/(1-p_1)}{p_2/(1-p_2)}=\operatorname{logit}p_1-\operatorname{logit}p_2 .$$

*Step 1.* By the CLT, $\sqrt{n_k}(\hat p_k-p_k)\to_{\mathrm d}\mathcal N(0,p_k(1-p_k))$ for $k=1,2$.

*Step 2.* Apply the delta method with $g=\operatorname{logit}$, $g'(p)=1/(p(1-p))$:
$$\sqrt{n_k}\big(\operatorname{logit}\hat p_k-\operatorname{logit}p_k\big)\to_{\mathrm d}\mathcal N\Big(0,\ \frac{1}{p_k(1-p_k)}\Big),$$
because $\big(\frac{1}{p(1-p)}\big)^2p(1-p)=\frac{1}{p(1-p)}$.

*Step 3.* The two groups are independent, so the variances of the difference add:
$$\operatorname{Var}\big(\log\widehat{\mathrm{OR}}\big)\approx\frac{1}{n_1p_1(1-p_1)}+\frac{1}{n_2p_2(1-p_2)} .$$

*Step 4.* Substitute the plug-in estimates $\hat p_1=a/n_1$, $\hat p_2=c/n_2$ and simplify:
$$\frac{1}{n_1\hat p_1(1-\hat p_1)}=\frac{1}{n_1\cdot\frac{a}{n_1}\cdot\frac{b}{n_1}}=\frac{n_1}{ab}=\frac{a+b}{ab}=\frac1a+\frac1b,$$
and likewise for the second group. Hence
$$\widehat{\operatorname{se}}\big(\log\widehat{\mathrm{OR}}\big)=\sqrt{\tfrac1a+\tfrac1b+\tfrac1c+\tfrac1d},$$
which is Woolf's formula.

*Hypotheses used.* (i) Two independent binomial samples with fixed $n_1,n_2$ and $p_1,p_2$ strictly inside $(0,1)$ — the *interior* requirement of Common Confusion 7.9. (ii) Finite variance, automatic for a Bernoulli. (iii) Differentiability of $\operatorname{logit}$ at $p_k$, which fails at $0$ and $1$. (iv) Slutsky, to replace $p_k$ by $\hat p_k$ inside the variance formula in Step 4; formally, $\hat p_k\to_{\mathrm p}p_k$ and $p\mapsto1/(p(1-p))$ is continuous on $(0,1)$, so Theorem 7.39(2) and Theorem 7.40 apply.

*Zero cells.* If any of $a,b,c,d$ is $0$, then $\hat p_k\in\{0,1\}$, the logit is $\pm\infty$, and the formula returns $\infty$. This is not a numerical accident: the estimator genuinely has no finite value, and no amount of asymptotic theory repairs it, because the hypothesis $p_k\in(0,1)$ was about the *truth* while the failure is in the *estimate*. Worked Example 7.23 measures how often this happens ($0.49\%$ of replications at $p=0.1$, $n=50$). The standard fixes are the Haldane–Anscombe correction (add $\tfrac12$ to each cell, which also reduces the second-order bias), exact conditional inference (Fisher's exact test / conditional maximum likelihood), or a weakly informative Bayesian prior on the log-odds.

**C3.** *Statement and proof of Slutsky.* Theorem 7.40, proved in §7.7 via Cramér–Wold plus the lemma "$A_n\to_{\mathrm d}A$ and $B_n\to_{\mathrm p}0$ imply $A_n+B_n\to_{\mathrm d}A$". The essential hypothesis is that $Y_n$ converges to a *constant*; §7.1.4 exhibits the failure otherwise ($X_n=Z$, $Y_n=(-1)^nZ$).

*The interval.* Suppose (H1) $\sqrt n(\hat\theta_n-\theta)\to_{\mathrm d}\mathcal N(0,\sigma^2)$ with $\sigma^2\in(0,\infty)$, and (H2) $\hat\sigma_n\to_{\mathrm p}\sigma$. Then $\sigma/\hat\sigma_n\to_{\mathrm p}1$ by the continuous mapping theorem (Theorem 7.39(2), with $y\mapsto\sigma/y$ continuous at $\sigma>0$), and Slutsky gives
$$T_n:=\frac{\sqrt n(\hat\theta_n-\theta)}{\hat\sigma_n}=\underbrace{\frac{\sqrt n(\hat\theta_n-\theta)}{\sigma}}_{\to_{\mathrm d}\ \mathcal N(0,1)}\cdot\underbrace{\frac{\sigma}{\hat\sigma_n}}_{\to_{\mathrm p}1}\ \longrightarrow_{\mathrm d}\ \mathcal N(0,1).$$
Since $\pm z_{1-\alpha/2}$ are continuity points of $\Phi$,
$$\mathbb{P}\Big(\theta\in\big[\hat\theta_n\pm z_{1-\alpha/2}\hat\sigma_n/\sqrt n\big]\Big)=\mathbb{P}\big(|T_n|\le z_{1-\alpha/2}\big)\longrightarrow1-\alpha .$$

*(i) What is asserted.* A limit of coverage probabilities. Nothing else.

*(ii) What is held fixed.* The data-generating distribution $P$, the parameter $\theta=\theta(P)$, the dimension of the model, and the estimation procedure. Only $n$ moves.

*(iii) What is not promised at finite $n$.* Not that coverage is $\ge1-\alpha$, nor even close to it, at your $n$. Not that convergence is uniform over a class of $P$'s — and it is not: for every $n$ there is a $P$ (e.g. Bernoulli with $p$ small enough) for which coverage is arbitrarily bad, which is why "for each fixed $P$" cannot be upgraded to "uniformly in $P$". Not that the interval has the right shape when the sampling distribution is skewed. Worked Example 7.27 measures all three: at $n=100$ the measured coverage is $0.9485$ for a Gaussian parent, $0.9140$ for a lognormal parent, and $0.6336$ for a $\mathrm{Bernoulli}(0.01)$ parent; at $n=10$ the Bernoulli cell reads $0.0949$ against a nominal $0.95$. Every one of those cells satisfies (H1) and (H2).

*(iv) Two realistic ML failures.* (a) *Post-selection inference.* You fit $50$ model variants, pick the best by validation accuracy, and report a confidence interval for its test accuracy. The estimator whose asymptotics you computed was "accuracy of a fixed model", not "accuracy of the argmax over $50$ models"; the selection step makes $\hat\theta_n$ a different, biased functional, and (H1) simply does not hold for it. (b) *Rare-event metrics.* You report a $95\%$ interval for a catastrophic-failure rate with $n\hat p<10$. (H1) and (H2) both hold in the limit, but Worked Example 7.30 shows the finite-$n$ interval can be the single point $\{0\}$; use Clopper–Pearson. A third, worth a sentence: correlated evaluation data (multiple questions from the same document, multiple frames from the same video) violates independence, and the effective sample size is far below $n$, so $\hat\sigma_n$ converges to the wrong constant and (H2) fails outright.

---

### References and further reading

**Textbooks.**

- R. Durrett, *Probability: Theory and Examples*, 5th ed., Cambridge, 2019. Chapters 2 (laws of large numbers, Borel–Cantelli) and 3 (weak convergence, characteristic functions, CLT, Lindeberg–Feller) are the closest match to this chapter's level and coverage; §3.10 has the multivariate versions.
- P. Billingsley, *Probability and Measure*, 3rd ed., Wiley, 1995. Chapters 22 (SLLN), 26–27 (characteristic functions, CLT). Slower and more discursive than Durrett; excellent for a first careful reading.
- P. Billingsley, *Convergence of Probability Measures*, 2nd ed., Wiley, 1999. The reference for Portmanteau, Prokhorov, Skorokhod, and Donsker in general metric spaces.
- W. Feller, *An Introduction to Probability Theory and Its Applications*, Vol. II, 2nd ed., Wiley, 1971. Chapters XV–XVI remain the best source on characteristic functions, Esseen's smoothing lemma, Edgeworth expansions, and the lattice corrections used in Worked Example 7.20.
- A. W. van der Vaart, *Asymptotic Statistics*, Cambridge, 1998. Chapters 2–3 (stochastic convergence, delta method), 19 (empirical processes), 20 (functional delta method), 23 (bootstrap). The single most useful book for the statistical applications in §§7.6–7.9.
- A. W. van der Vaart and J. A. Wellner, *Weak Convergence and Empirical Processes*, Springer, 1996. §2.14 for DKW; the standard reference for the machinery underlying Chapter 4's uniform convergence.
- T. S. Ferguson, *A Course in Large Sample Theory*, Chapman & Hall, 1996. Short, computational, and full of delta-method examples of exactly the kind in §7.6.3.
- E. L. Lehmann, *Elements of Large-Sample Theory*, Springer, 1999. §5.4 for the correlation and Fisher $z$ results used in Worked Example 7.24.
- V. V. Petrov, *Sums of Independent Random Variables*, Springer, 1975. Chapter V for sharp Berry–Esseen and lattice results.
- P. Hall, *The Bootstrap and Edgeworth Expansion*, Springer, 1992. The definitive account of why the bootstrap beats the normal approximation: it captures the first Edgeworth term.
- G. R. Shorack and J. A. Wellner, *Empirical Processes with Applications to Statistics*, SIAM Classics, 2009. For the Brownian bridge functionals in Worked Example 7.28.
- I. A. Ibragimov and Yu. V. Linnik, *Independent and Stationary Sequences of Random Variables*, Wolters-Noordhoff, 1971, for stable laws — the correct limit theory when the CLT's variance hypothesis fails (Exercise A9).

**Papers, including the ML-facing ones.**

- P. Massart, "The tight constant in the Dvoretzky–Kiefer–Wolfowitz inequality," *Ann. Probab.* 18 (1990), 1269–1283.
- I. G. Shevtsova, "On the absolute constants in the Berry–Esseen type inequalities for identically distributed summands," arXiv:1111.6554 (2011). Source of the constant $0.4748$.
- J. Gil-Pelaez, "Note on the inversion theorem," *Biometrika* 38 (1951), 481–482. The formula used to compute the $t_3$ column of Worked Example 7.20 exactly.
- S. Hurst, "The characteristic function of the Student-$t$ distribution," Financial Mathematics Research Report, ANU, 1995. Source of $\varphi_{t_3}(t)=(1+\sqrt3|t|)e^{-\sqrt3|t|}$.
- C. C. Heyde, "On a property of the lognormal distribution," *J. Roy. Statist. Soc. B* 25 (1963), 392–393. The moment-indeterminacy example of Common Confusion 7.6.
- L. D. Brown, T. T. Cai and A. DasGupta, "Interval estimation for a binomial proportion," *Statistical Science* 16 (2001), 101–133. Read this before you next report $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}$; it is the applied companion to Worked Examples 7.18 and 7.27.
- B. Efron, "Bootstrap methods: another look at the jackknife," *Ann. Statist.* 7 (1979), 1–26. The origin of Chapter 9's subject, resting on Theorems 7.42 and 7.44.
- Q. Li, C. Tai and W. E, "Stochastic modified equations and dynamics of stochastic gradient algorithms I: mathematical foundations," *JMLR* 20(40) (2019). The rigorous version of the SDE approximation of §7.9.2, with the horizon restrictions stated honestly.
- S. Mandt, M. D. Hoffman and D. M. Blei, "Stochastic gradient descent as approximate Bayesian inference," *JMLR* 18(134) (2017). The stationary-distribution picture, and its isotropy assumption.
- U. Simsekli, L. Sagun and M. Gurbuzbalaban, "A tail-index analysis of stochastic gradient noise in deep neural networks," *ICML* 2019. Empirical evidence that Proposition 7.45's finite-variance hypothesis may fail in practice; see also the responses in the literature, which is not settled.
- P. Goyal et al., "Accurate, large minibatch SGD: training ImageNet in 1 hour," arXiv:1706.02677 (2017). The linear scaling rule of §7.9.2 in production.
- S. L. Smith, P.-J. Kindermans, C. Ying and Q. V. Le, "Don't decay the learning rate, increase the batch size," *ICLR* 2018. The same $\eta/B$ invariance, used the other way round.
- T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation* 10 (1998), 1895–1923. An early and still-relevant catalogue of how §7.7's hypotheses get violated in ML benchmarking.
- X. Bouthillier et al., "Accounting for variance in machine learning benchmarks," *MLSys* 2021. A modern empirical version of the same warning, with measured variance components.
