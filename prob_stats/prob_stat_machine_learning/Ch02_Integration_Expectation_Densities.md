# Chapter 2 — Integration, Expectation, and Densities

*(Revised edition. Changes from the first draft: the monotone convergence theorem is now proved immediately after the definition of the integral, so no proof relies on a forward reference; total variation distance and $f$-divergences are defined, so the "every $f$-divergence is blind" claim is now a theorem rather than a slogan; Minkowski's and Hölder's inequalities are stated and accounted for; the exact relationship between the Riemann and Lebesgue integrals is stated as a theorem; the law-of-large-numbers discussion of infinite and undefined means is made precise; a caveat on Wasserstein distances is added; the exercise set has a gentler on-ramp; and every self-check now has an answer in Appendix 2.A.)*

## Chapter 2 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 2.1 | Riemann slices the domain; Lebesgue slices the range and *measures* the level sets | One integral covers sums, integrals, and mixtures — one formula for discrete, continuous, and mixed models |
| 2.2 | The three-step construction; the monotone convergence theorem; $L^1$; means that are $+\infty$ or undefined | Why $\mathbb{E}[X]$ can be $+\infty$ or *undefined*, and what each does to the law of large numbers; heavy-tailed gradient noise |
| 2.3 | $\mathbb{E}[X] := \int_\Omega X\,d\mathbb{P}$, and the change-of-variables theorem | $\sum_x g(x)p(x)$, $\int gp\,dx$, and the Monte Carlo estimator are one theorem |
| 2.4 | Linearity, monotonicity, a.e. blindness, Markov | A density value at a point is meaningful only under a continuity convention — which is exactly the convention your normalizing flow relies on |
| 2.5 | Fatou, dominated convergence, uniform integrability (Vitali); mass escapes sideways or upward | Diagnoses VAE variance collapse and every "the limit of my loss is not the loss of my limit" bug |
| 2.6 | Differentiating under the expectation — **Failure 3** | Reparameterization and REINFORCE are two answers to one measure-theoretic question; the score has mean zero |
| 2.7 | Radon–Nikodym — **Failures 1 and 2** | A density is a local exchange rate, not a probability; pdf, pmf, and the ReLU's mixed law are one object |
| 2.8 | Total variation, KL, $f$-divergences, JS — **Failure 4**; Wasserstein, honestly | Why GAN gradients vanish between singular laws, why $W_1$ does not, and what $W_1$ costs you in exchange |
| 2.9 | Change of measure and importance sampling | Off-policy RL, PPO ratio clipping, and the silent variance catastrophe your own error bar cannot detect |
| 2.10 | Product measures, Fubini–Tonelli, i.i.d. data | Every generalization bound is a statement about $P^{\otimes n}$; why losses are sums and minibatch gradients unbiased |
| 2.11 | $L^p$ spaces, Hölder, Minkowski, Cauchy–Schwarz, Jensen; $L^2$ as a Hilbert space | The room in which conditional expectation (Ch. 6) is a projection; Jensen manufactures every variational bound |

## 2.0 What this chapter is for

Chapter 1 built the stage: a probability space $(\Omega,\mathcal{F},\mathbb{P})$ (Definition 1.7), random variables as measurable functions (Definition 1.13), laws as pushforwards $P_X = X_\#\mathbb{P}$ (Definition 1.16), and $\sigma$-algebras as *information* (Definition 1.21). What the stage lacked was an actor: a way to compute a *number* from a random variable. We said what $X$ is; we did not say what $\mathbb{E}[X]$ is, except in the two special cases a first course provides — a sum for discrete variables, a Riemann integral of a density for continuous ones.

This chapter builds the missing machine. There is exactly one integral, $\int_\Omega X\,d\mathbb{P}$, defined for every non-negative measurable function on every measure space, with no case distinction. Sums are a special case (counting measure). Riemann integrals of nice functions are a special case (Lebesgue measure — Theorem 2.1 below says exactly when). The awkward object $A = \max(0,Z)$ from Chapter 1, neither discrete nor continuous, is *not* a special case requiring special treatment: it is another measurable function, integrated by the same definition, and by §2.7 it will have a density like everything else — once we stop insisting that "density" means "density with respect to Lebesgue measure."

You are assumed to bring from Chapter 1, without re-derivation: the axioms and elementary consequences of a measure (Proposition 1.8, especially **continuity from below**, the engine of every convergence theorem below); the closure of measurable functions under countable sups, infs, and limits (Proposition 1.15 — this is what makes the whole construction possible); the uniqueness theorem for measures agreeing on a generating $\pi$-system (Theorem 1.20); Doob–Dynkin (Lemma 1.22); independence (Definition 1.23); and "almost surely" (Definition 1.25). The running examples return: three coin flips with $X$ = number of heads; the ReLU output $A = \max(0,Z)$, $Z\sim\mathcal{N}(0,1)$; and the circle generator $G(t)=(\cos t,\sin t)$, whose image is Lebesgue-null in $\mathbb{R}^2$.

Here is the punchline, stated in advance so you can watch it arrive. **Every quantity in machine learning written as an expectation is an integral against a measure, and almost every subtle bug in the theory is a question about whether two operations commute.** Does the derivative commute with the expectation? (§2.6: sometimes, and the reparameterization trick is a way of arranging that it does.) Does the limit commute with the integral? (§2.5: only if mass cannot escape — and "cannot escape" has an exact name, uniform integrability plus tightness.) Does one order of a double integral commute with the other? (§2.10: yes for non-negative integrands, yes for absolutely integrable ones, and *no* in general.) Can one distribution's density be written with respect to another's? (§2.7: exactly when $\nu\ll\mu$, and this one condition explains why densities exceed $1$, why the ReLU output has a density after all, why importance sampling needs support coverage, and why the GAN loss is flat.)

By the end of the chapter, all four Failures that opened Chapter 1 are closed, and the score card appears in §2.13. Two of them (densities exceeding $1$; the mixed law with neither pmf nor pdf) turn out to be the *same* failure, and both dissolve into one theorem — Radon–Nikodym — plus a change of attitude: a density is a *ratio of two measures*, a local exchange rate between "how much probability" and "how much reference stuff." One (swapping $\nabla$ and $\mathbb{E}$) is a hypothesis-checking problem solved by dominated convergence. And one (the flat GAN loss) is a geometric fact, now provable in full generality (Proposition 2.36): $f$-divergences are computed from density ratios, and between mutually singular measures every density ratio is degenerate, so every $f$-divergence returns the same uninformative constant no matter how far apart the measures sit.

One promise about difficulty. §2.2 is the most abstract material in the chapter: nested definitions, none of which computes anything you care about, plus one hard proof (monotone convergence). It is also short, and it is the last time you will think about the definition, because from §2.3 onward we compute with *theorems about* the integral. Every step is spelled out, and the "standard ladder" proof technique introduced there will be used four times before the chapter ends.

## 2.1 One integral to replace sums and Riemann integrals

### 2.1.1 Why the two toolkits from a first course fail

A first probability course hands you two devices. If $X$ takes values in a countable set with $\mathbb{P}(X=x) = p(x)$, then $\mathbb{E}[g(X)] = \sum_x g(x)p(x)$. If $X$ has a density $p$, then $\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x)p(x)\,dx$, a Riemann integral. Two devices, two proofs of every theorem, and a nagging question: what about a random variable that is neither?

That question is the daily condition of machine learning. Three examples:

*The ReLU output.* $A = \max(0,Z)$ with $Z\sim\mathcal{N}(0,1)$ satisfies $\mathbb{P}(A=0)=\tfrac12$, so $A$ is not continuous (a continuous law assigns probability zero to every point). But $A$ also spreads probability $\tfrac12$ over the uncountable set $(0,\infty)$, so it is not discrete. Chapter 1 could describe it through its CDF, $F_A(a) = \Phi(a)$ for $a\ge0$, but could not integrate against it with either device. Every hidden unit in a ReLU network has exactly this law at initialization, up to scale.

*A deep generative model.* $P_\theta = (G_\theta)_\#\mathcal{N}(0,I_k)$ with $G_\theta:\mathbb{R}^k\to\mathbb{R}^d$, $k<d$. Chapter 1 proved (via the circle example) that the support of such a law is Lebesgue-null, so it has no Lebesgue density at all. Yet we constantly compute expectations under it — a sample from a GAN *is* an evaluation used to estimate an expectation.

*A mixed loss.* A point mass at "no purchase" mixed with a continuous spend distribution; a quantized weight distribution in a low-precision network; a censored measurement. All mixed, none covered by either device.

The Lebesgue integral covers all of these with a single definition — and, surprisingly, it is *easier to prove theorems about* than the Riemann integral: its convergence theorems (§2.5) have hypotheses you can actually check. That is why analysis adopted it a century ago and never went back.

### 2.1.2 The core idea: slice the range, not the domain

The Riemann integral chops the *domain* into thin vertical columns. Over each column $[x_i,x_{i+1}]$ it asks "how tall is $f$ here?", multiplies height by width, and adds up. The columns must be intervals, because "width" means "length of an interval." That is the Riemann integral's fatal limitation: it handles only functions that are nearly constant on small intervals. A function oscillating at every scale — the indicator of the rationals — defeats it.

The Lebesgue integral chops the *range* into thin horizontal bands. At level $t$ it asks: **how much of the domain has $f$ above $t$?** That is a question about the *measure* of the set $\{f>t\}$ — and measure is exactly what Chapter 1 constructed. The set $\{f>t\}$ may be as complicated as any element of $\mathcal{F}$ — a union of a thousand intervals, a Cantor set, the irrationals — because we never need it to be an interval; we only need its measure. And "$\{f>t\}\in\mathcal{F}$ for every $t$" is *precisely* the definition of measurability (Definition 1.13). Measurability is not a technical nuisance attached to random variables; it is the exact hypothesis that makes horizontal slicing possible.

The standard analogy is coin sorting, and we attach it to a real computation so it does not float free. You have a heap of coins and want the total value. *Riemann:* sweep left to right, adding each coin's value to a running total. *Lebesgue:* sort into denominations, count each pile, compute $\sum(\text{value})\times(\text{count})$. Same total; the second works however irregularly the coins are scattered.

Take Chapter 1's running example (i): $\Omega=\{H,T\}^3$ with the uniform measure, $X$ = number of heads. Sweeping the domain means summing over the eight outcomes:
$$\mathbb{E}[X] = \tfrac18(3+2+2+2+1+1+1+0) = \tfrac{12}{8} = 1.5 .$$
Sorting by denomination means grouping outcomes by the value of $X$ — decomposing $\Omega$ into the level sets $\{X=0\},\dots,\{X=3\}$ of sizes $1,3,3,1$:
$$\mathbb{E}[X] = 0\cdot\tfrac18 + 1\cdot\tfrac38 + 2\cdot\tfrac38 + 3\cdot\tfrac18 = 1.5 .$$
Now the truly Lebesgue bookkeeping, with horizontal slices. For integer-valued $X\ge0$, the number of levels $t\in\{1,2,3,\dots\}$ lying below $X(\omega)$ is exactly $X(\omega)$, so summing the measures of the superlevel sets reproduces the mean:
$$\mathbb{P}(X\ge1)+\mathbb{P}(X\ge2)+\mathbb{P}(X\ge3) = \tfrac78+\tfrac48+\tfrac18 = 1.5 .$$
Three bookkeepings, one number. The third is the discrete shadow of the **tail formula** $\mathbb{E}[X] = \int_0^\infty\mathbb{P}(X>t)\,dt$, proved as Proposition 2.48 and used constantly in Chapter 4 to convert tail bounds into moment bounds. Notice what it needs: only the measures $\mathbb{P}(X>t)$ — no density, no pmf, no case distinction. That is the promise of this chapter in one line.

*Figure 2.1 (described).* Left panel: a bumpy non-negative curve with ten vertical rectangles beneath it, each on a small interval of the $x$-axis — the Riemann picture, height times width. Right panel: the same curve with horizontal dashed lines at levels $t$; one band is highlighted, and its superlevel set $\{f>t\}$ appears on the $x$-axis as **two** disjoint segments, because the curve is bimodal. The point: $\{f>t\}$ needs no structure beyond membership in $\mathcal{F}$.

### 2.1.3 Exactly how the two integrals relate

Since a first course leaves the relationship vague, here it is precisely. Both parts are standard real analysis; proofs are in the references (§2.12).

**Theorem 2.1** (Riemann versus Lebesgue). Let $f:[a,b]\to\mathbb{R}$ be bounded.
1. $f$ is Riemann integrable **iff** $f$ is continuous $\lambda$-almost everywhere on $[a,b]$ (Lebesgue's criterion). In that case $f$ is Lebesgue measurable up to a null set, and the two integrals agree.
2. The *improper* Riemann integral is a different object: it is a **limit of integrals**, $\lim_{b\to\infty}\int_0^b$, and a limit can exist through cancellation the Lebesgue integral refuses to perform. Standard example: $\int_0^\infty \frac{\sin x}{x}\,dx = \frac{\pi}{2}$ as an improper Riemann integral, yet $\int_0^\infty|\sin x/x|\,d\lambda = \infty$, so $\sin x/x\notin L^1(\lambda)$ and its Lebesgue integral is **undefined** in the sense of Definition 2.9 below.

So the two theories are not nested; they answer different questions. Lebesgue insists on absolute convergence and receives the convergence theorems in exchange. When a physicist hands you a conditionally convergent integral, the correct reading is "a limit of Lebesgue integrals over bounded regions," and the order of the limit is part of the data.

> **Common Confusion 2.1**
>
> *"Isn't the Lebesgue integral just the Riemann integral with extra steps? Every function I actually integrate is continuous, so who cares?"*
>
> If you only ever integrate continuous functions on bounded intervals, the two agree (Theorem 2.1) and you may use whichever you like. The reason to care is not exotic functions; it is **exotic measures and limits**. Three cash-outs. (1) The Riemann integral is defined against *length* only. It says nothing about $\int X\,d\mathbb{P}$ on an abstract $\Omega$ like $\{H,T\}^3$ or a space of images, and nothing about integrating against a law with an atom. The Lebesgue integral is defined against any measure, so one formula covers sums, integrals, and mixtures. (2) The convergence theorems. If $f_n\to f$ pointwise, the Riemann integrals $\int f_n$ may all exist while "$\int f$" does not even make sense as a Riemann integral; Lebesgue theory guarantees the limit is measurable (Proposition 1.15) and gives checkable conditions for $\int f_n\to\int f$ (§2.5). Every training loss is a limit of empirical averages and every gradient a limit of difference quotients, so you need those conditions. (3) Completeness of $L^2$ (§2.11), false for Riemann-integrable functions, and exactly what Chapter 6 needs to define conditional expectation as a projection. The extra steps buy theorems.

> **Self-check 2.1** *(answers in Appendix 2.A)*
>
> 1. Why does the Riemann construction require its slices of the domain to be intervals, while Lebesgue places no such requirement on $\{f>t\}$?
> 2. In the coin-flip example, which of the three computations of $\mathbb{E}[X]=1.5$ would still work if $\Omega$ were the set of $256\times256$ images and $X$ a network's loss?
> 3. Chapter 1 said measurability of $X$ means $\{X\le a\}\in\mathcal{F}$ for all $a$. Where exactly does the horizontal-slicing picture use that fact?
> 4. Which half of Theorem 2.1 shows the two theories are not nested, and in which direction?

## 2.2 The three-step construction, and the first convergence theorem

Throughout, $(\Omega,\mathcal{F},\mu)$ is a measure space — not necessarily a probability space, since we want counting measure and Lebesgue measure too. Measurable means $\mathcal{F}$-measurable (Definition 1.13), with $\mathbb{R}$ carrying its Borel $\sigma$-algebra. Functions may take values in $[0,\infty]$; the extended real line carries the $\sigma$-algebra generated by the Borel sets together with $\{\pm\infty\}$, and the closure properties of Proposition 1.15 continue to hold.

The construction has three steps, each defined from the previous. Step (a) integrates functions so crude that the integral is a finite sum by fiat. Step (b) integrates every non-negative measurable function by squeezing from below with step-(a) functions. Step (c) handles signs by splitting into positive and negative parts. We then immediately prove the **monotone convergence theorem**, because it needs nothing beyond step (b) and everything after §2.2 needs *it*. (In the first draft, MCT was proved two sections later than it was first used; this ordering removes every forward reference.)

### 2.2.1 Step (a): simple functions

**Definition 2.2** (simple function). A function $s:\Omega\to\mathbb{R}$ is **simple** if it is measurable and takes finitely many values. Equivalently
$$s = \sum_{i=1}^k a_i\mathbb{1}_{A_i},\qquad a_i\in\mathbb{R},\ A_i\in\mathcal{F},$$
and it can always be so written with the $A_i$ *disjoint* and the $a_i$ the distinct values of $s$ — take $A_i = s^{-1}(\{a_i\})$, in $\mathcal{F}$ by measurability. This is the **canonical representation**. Note the resonance with Proposition 1.4: on a finite $\Omega$, a measurable function is exactly one constant on the atoms of $\mathcal{F}$, i.e., a simple function whose $A_i$ are unions of atoms.

**Definition 2.3** (integral of a non-negative simple function). For simple $s\ge0$ with canonical representation $s = \sum_{i=1}^k a_i\mathbb{1}_{A_i}$ ($A_i$ disjoint, $\bigcup A_i = \Omega$),
$$\int_\Omega s\,d\mu := \sum_{i=1}^k a_i\,\mu(A_i)\in[0,\infty],$$
with the standing **convention $0\cdot\infty = 0$**.

The convention is forced, not chosen. Take $\mu=\lambda$ on $\mathbb{R}$ and $s\equiv0$: its canonical representation has the single value $a_1=0$ on $A_1=\mathbb{R}$, and $\lambda(\mathbb{R})=\infty$. The integral of the zero function had better be $0$, so $0\cdot\infty := 0$. The same convention says a function that is huge on a null set contributes nothing — the seed of the a.e. blindness of §2.4.

**Worked Example 2.1** (a simple function by hand). $\Omega=\{H,T\}^3$, $\mathcal{F}=2^\Omega$, $\mathbb{P}$ uniform, $X$ = number of heads. $X$ is simple with canonical representation
$$X = 0\cdot\mathbb{1}_{A_0}+1\cdot\mathbb{1}_{A_1}+2\cdot\mathbb{1}_{A_2}+3\cdot\mathbb{1}_{A_3},\qquad |A_0|,\dots,|A_3| = 1,3,3,1,$$
so Definition 2.3 gives $\int X\,d\mathbb{P} = 0\cdot\tfrac18+1\cdot\tfrac38+2\cdot\tfrac38+3\cdot\tfrac18 = 1.5$. This is not a new computation — it is the *definition*, and it happens to coincide with the elementary formula $\sum_x x\,\mathbb{P}(X=x)$; that coincidence is Corollary 2.12 below. A *non-canonical* representation gives the same number: writing $X = \mathbb{1}_{\{X\ge1\}}+\mathbb{1}_{\{X\ge2\}}+\mathbb{1}_{\{X\ge3\}}$ (check pointwise: at an $\omega$ with two heads, the first two indicators fire) we get $\tfrac78+\tfrac48+\tfrac18 = 1.5$ again. That the answer never depends on the representation is exactly what the next lemma asserts, and it is the first thing that needs proof.

**Lemma 2.4** (the simple integral is well defined, additive, monotone). Let $s,t\ge0$ be simple and $c\ge0$.
1. If $s = \sum_{i=1}^k a_i\mathbb{1}_{A_i} = \sum_{j=1}^m b_j\mathbb{1}_{B_j}$ with each family disjoint and covering $\Omega$, then $\sum_i a_i\mu(A_i) = \sum_j b_j\mu(B_j)$. Hence $\int s\,d\mu$ is unambiguous and computable from *any* disjoint representation.
2. $\int(s+t)\,d\mu = \int s\,d\mu + \int t\,d\mu$ and $\int cs\,d\mu = c\int s\,d\mu$.
3. If $s\le t$ pointwise then $\int s\,d\mu \le \int t\,d\mu$.

*Proof.* (1) The sets $C_{ij} := A_i\cap B_j$ lie in $\mathcal{F}$, are pairwise disjoint, and $A_i = \bigsqcup_j C_{ij}$, $B_j = \bigsqcup_i C_{ij}$. Finite additivity gives $\mu(A_i) = \sum_j\mu(C_{ij})$ and $\mu(B_j) = \sum_i\mu(C_{ij})$. On a non-empty $C_{ij}$, $a_i = s = b_j$. Hence
$$\sum_i a_i\mu(A_i) = \sum_{i,j} a_i\mu(C_{ij}) = \sum_{i,j:\,C_{ij}\ne\emptyset} a_i\mu(C_{ij}) = \sum_{i,j:\,C_{ij}\ne\emptyset} b_j\mu(C_{ij}) = \sum_j b_j\mu(B_j),$$
with empty $C_{ij}$ contributing $0$ on both sides (using $0\cdot\infty=0$ when some $a_i=0$ and $\mu(A_i)=\infty$). All sums are finite sums of terms in $[0,\infty]$, so rearrangement is legitimate.

(2) On the common refinement $\{C_{ij}\}$, $s+t$ is simple with value $a_i+b_j$ on $C_{ij}$; by (1) compute all three integrals on that refinement and split the sum. Homogeneity is immediate.

(3) $t = s+(t-s)$ with $t-s$ simple and non-negative; apply (2) and non-negativity. $\square$

### 2.2.2 Step (b): non-negative measurable functions, and monotone convergence

**Definition 2.5** (integral of a non-negative measurable function). For measurable $f:\Omega\to[0,\infty]$,
$$\int_\Omega f\,d\mu := \sup\Big\{\int_\Omega s\,d\mu : s\text{ simple},\ 0\le s\le f\text{ pointwise}\Big\}\in[0,\infty],$$
and for $E\in\mathcal{F}$, $\int_E f\,d\mu := \int_\Omega f\mathbb{1}_E\,d\mu$.

Read it as: *approximate $f$ from below by staircases and take the best staircase estimate.* The supremum runs over an enormous family, which makes the definition useless for direct computation — you will never evaluate it directly after this section. Its purpose is to exist, to be monotone by construction, and to support the proof of monotone convergence, after which we compute with theorems.

Two things must be checked: consistency with Definition 2.3 when $f$ is itself simple, and that the family of competitors is rich enough to reach $f$. The second is important enough to name.

**Lemma 2.6** (simple approximation). Let $f:\Omega\to[0,\infty]$ be measurable and define
$$s_n := \min\Big(\frac{\lfloor 2^n f\rfloor}{2^n},\,n\Big).$$
Then each $s_n$ is simple, $0\le s_n\le s_{n+1}\le f$, and $s_n(\omega)\to f(\omega)$ for every $\omega$ (where $f(\omega)=\infty$, $s_n(\omega)=n\to\infty$).

*Proof.* $s_n$ takes values in the finite set $\{0,2^{-n},2\cdot2^{-n},\dots,n\}$, and each preimage has the form $f^{-1}([k2^{-n},(k+1)2^{-n}))$ or $f^{-1}([n,\infty])$, in $\mathcal{F}$ by measurability of $f$; so $s_n$ is simple, and clearly $s_n\le f$. Monotonicity in $n$: the dyadic grid at level $n{+}1$ refines the grid at level $n$, so $\lfloor2^{n+1}f\rfloor2^{-(n+1)}\ge\lfloor2^nf\rfloor2^{-n}$, and the truncation level increases. Convergence: if $f(\omega)<\infty$, then for $n>f(\omega)$ the truncation is inactive and $0\le f(\omega)-s_n(\omega)\le2^{-n}\to0$. $\square$

**Proposition 2.7** (consistency and monotonicity). (i) If $f$ is simple and non-negative, Definitions 2.3 and 2.5 agree. (ii) If $0\le f\le g$ are measurable then $\int f\,d\mu\le\int g\,d\mu$.

*Proof.* (i) $f$ is itself a competitor, so the sup is $\ge\int f$ in the sense of Definition 2.3; every competitor $s\le f$ has $\int s\le\int f$ by Lemma 2.4(3), so the sup is $\le$ it. (ii) Every simple $s\le f$ satisfies $s\le g$, so the sup defining $\int g$ runs over a larger family. $\square$

**Worked Example 2.2** (the staircases, computed). $\mu=\lambda$ on $[0,1]$, $f(x)=x$. The approximants of Lemma 2.6 are $s_n(x)=\lfloor2^nx\rfloor/2^n$ (the truncation at $n$ never bites, since $f\le1$): a staircase with $2^n$ steps of width $2^{-n}$ and heights $0,2^{-n},\dots,1-2^{-n}$. Its integral is a finite sum:
$$\int_0^1 s_n\,d\lambda = \sum_{k=0}^{2^n-1}\frac{k}{2^n}\cdot\frac{1}{2^n} = \frac{2^n-1}{2^{n+1}},$$
equal to $0.25,\ 0.375,\ 0.4375,\ 0.46875,\ 0.49951$ for $n=1,2,3,4,10$, increasing to $0.5$. And $0.5$ is the supremum over *all* simple minorants, not just these: any simple $s\le f$ has $\int s\le\int f$ while the staircases already reach $0.5$ in the limit. So $\int_0^1 x\,d\lambda = \tfrac12$, agreeing with first-year calculus, as Theorem 2.1 says it must.

Now the theorem that powers everything downstream. Watch where countable additivity enters.

**Theorem 2.8** (monotone convergence, MCT). Let $f_n:\Omega\to[0,\infty]$ be measurable with $f_n\le f_{n+1}$ pointwise, and let $f := \lim_n f_n$ (which exists pointwise in $[0,\infty]$ and is measurable by Proposition 1.15). Then
$$\int_\Omega f_n\,d\mu\ \uparrow\ \int_\Omega f\,d\mu .$$

*Proof.* Since $f_n\le f_{n+1}\le f$, monotonicity (Proposition 2.7(ii)) makes $\int f_n$ a non-decreasing sequence bounded above by $\int f$, so $L := \lim_n\int f_n$ exists in $[0,\infty]$ with $L\le\int f$.

For the reverse inequality, fix a simple $s = \sum_{i=1}^k a_i\mathbb{1}_{A_i}\le f$ (canonical, $A_i$ disjoint) and fix $c\in(0,1)$. Put
$$E_n := \{\omega: f_n(\omega)\ge c\,s(\omega)\}\in\mathcal{F}.$$
The $E_n$ increase (because $f_n$ does), and $\bigcup_n E_n = \Omega$: where $s(\omega)=0$, $f_n(\omega)\ge0 = cs(\omega)$ for every $n$; where $s(\omega)>0$, $f(\omega)\ge s(\omega)>cs(\omega)$, so some $f_n(\omega)$ eventually exceeds $cs(\omega)$. Now
$$\int f_n\,d\mu\ \ge\ \int f_n\mathbb{1}_{E_n}\,d\mu\ \ge\ \int c\,s\,\mathbb{1}_{E_n}\,d\mu\ =\ c\sum_{i=1}^k a_i\,\mu(A_i\cap E_n),$$
using monotonicity twice and Lemma 2.4 for the simple integral. Let $n\to\infty$: each $A_i\cap E_n\uparrow A_i$, so **continuity from below** (Proposition 1.8(3)) gives $\mu(A_i\cap E_n)\to\mu(A_i)$, whence $L\ge c\sum_i a_i\mu(A_i) = c\int s\,d\mu$. Let $c\uparrow1$: $L\ge\int s\,d\mu$. Take the supremum over simple $s\le f$: $L\ge\int f\,d\mu$ by Definition 2.5. $\square$

*Remark (where the axiom earned its keep).* Countable additivity entered through continuity from below, which Proposition 1.9 identified as *equivalent* to countable additivity given finite additivity. That is the payoff Chapter 1 promised: countable additivity is in the axioms precisely so that this proof works, and this proof is why we can differentiate under expectations, interchange sums and integrals, and (Chapter 6) define conditional expectation.

### 2.2.3 Step (c): signed functions, $L^1$, and the two ways a mean can fail

**Definition 2.9** (the general integral; $L^1$). For measurable $f:\Omega\to[-\infty,\infty]$, write $f^+ := \max(f,0)$ and $f^- := \max(-f,0)$, so $f = f^+-f^-$ and $|f| = f^++f^-$, all measurable. Define
$$\int_\Omega f\,d\mu := \int f^+\,d\mu - \int f^-\,d\mu$$
whenever at least one of the two terms is finite (so that $\infty-\infty$ never arises); say the integral is **defined** in that case, and **undefined** if $\int f^+ = \int f^- = \infty$. Say $f\in L^1(\mu)$ — $f$ is **integrable** — if $\int|f|\,d\mu<\infty$, i.e., both parts are finite; then $\big|\int f\,d\mu\big|\le\int|f|\,d\mu$.

So there are exactly three possibilities for $\int f\,d\mu$: a finite number ($f\in L^1$, up to sign conventions), the value $+\infty$ or $-\infty$ (one part diverges), or **undefined** (both parts diverge). Keep the last two distinct; they behave differently, and the next worked example plus Worked Example 2.8 in §2.5 pin down exactly how.

**Worked Example 2.3** (the Cauchy law: an undefined mean, and what it really does to the LLN). Let $X$ have the standard Cauchy density $p(x) = \frac{1}{\pi(1+x^2)}$. Then
$$\int X^+\,d\mathbb{P} = \int_0^\infty\frac{x}{\pi(1+x^2)}\,dx = \Big[\tfrac{1}{2\pi}\log(1+x^2)\Big]_0^\infty = \infty,$$
and by symmetry $\int X^-\,d\mathbb{P} = \infty$ too. So $\mathbb{E}[X]$ is **undefined** — not zero. The seductive argument "the density is symmetric, so the mean is $0$ by symmetry" contains one false sentence: *"the integral exists, and therefore equals its symmetric value."* Symmetry tells you that **if** the mean existed it would be $0$; it does not make the mean exist. (Compare: the principal value $\lim_{R\to\infty}\int_{-R}^R x\,p(x)\,dx = 0$ does exist — but it is a limit of integrals, not an integral, and it is not stable: shifting the truncation window to $[-R, 2R]$ changes the answer. Definition 2.9 is refusing to let an answer depend on an arbitrary truncation scheme.)

The practical consequence, stated precisely because the vague version causes confusion:

- **Undefined mean (Cauchy):** the strong law of large numbers *fails outright*. The sample mean $\bar X_n$ does not converge to anything — in fact $\bar X_n$ is itself standard Cauchy for every $n$, so averaging a million samples leaves you exactly where one sample left you.
- **Infinite mean (e.g., the Pareto of Worked Example 2.8, where $\int X^-=0$ and $\int X^+=\infty$):** the strong law *holds in the extended sense*: $\bar X_n\to+\infty$ almost surely. The sample mean does converge — to $+\infty$ — so it drifts upward without bound rather than fluctuating forever.

Both situations break the *finite-limit* LLN that underwrites naive Monte Carlo and standard SGD analyses, but they break it differently, and Definition 2.9 is what distinguishes them. Heavy-tailed stochastic gradient noise is not hypothetical: empirical tail-index studies of deep-network gradient noise (Şimşekli et al., 2019; see references) find regimes consistent with infinite variance, where the classical $O(1/\sqrt{n})$ SGD story loses its hypotheses.

**Worked Example 2.4** (counting measure: sums are integrals). Let $\# $ be counting measure on $\mathbb{N} = \{0,1,2,\dots\}$ ($\#(A) = $ number of elements of $A$) and $f:\mathbb{N}\to[0,\infty)$. Then
$$\int_{\mathbb{N}} f\,d\# = \sum_{n=0}^\infty f(n).$$
*Why:* for each $N$, $s_N := f\cdot\mathbb{1}_{\{0,\dots,N\}}$ is simple (finitely many values, zero off a finite set), and Definition 2.3 gives $\int s_N\,d\# = \sum_{n=0}^N f(n)$. The $s_N$ increase to $f$, so MCT (Theorem 2.8) gives $\int f\,d\# = \lim_N\sum_{n\le N}f(n) = \sum_n f(n)$. — Notice we just used MCT for the first time, four lines after proving it; the first draft of this chapter had to defer this argument.

*Cash-out:* if $P$ is a law on $\mathbb{N}$ with pmf $p(n)$, then $p$ is a density with respect to $\#$ (Definition 2.27 will make this official), and $\mathbb{E}[g(X)] = \sum_n g(n)p(n)$ is *literally the same integral* as $\int gp\,d\#$. For $\mathrm{Poisson}(\lambda)$: $\mathbb{E}[X] = \sum_{n\ge0} n e^{-\lambda}\lambda^n/n! = \lambda$. The discrete and continuous formulas from your first course were never two theories; they were one theory with two base measures.

> **Common Confusion 2.2**
>
> *"Why approximate from below? Why not sandwich between lower and upper staircases like Riemann does?"*
>
> Because on an infinite-measure space, upper staircases are useless. Take $f(x) = e^{-x^2}$ on $(\mathbb{R},\lambda)$, whose integral is $\sqrt\pi$. Any simple majorant $s\ge f$ satisfies $s>0$ everywhere (since $f>0$ everywhere); but $s$ takes only finitely many values, so its smallest value $m$ is positive, whence $s\ge m$ on all of $\mathbb{R}$ and $\int s\,d\lambda\ge m\cdot\lambda(\mathbb{R}) = \infty$. The infimum over majorants would return $\infty$ for a function whose integral is $\sqrt\pi$. Approximation from below has no such pathology: minorants may vanish outside a bounded set. The asymmetry is baked into $[0,\infty]$ — you can always afford to undershoot, but overshooting on an infinite-measure set costs infinitely much.

> **Self-check 2.2** *(answers in Appendix 2.A)*
>
> 1. Why must the convention $0\cdot\infty = 0$ be adopted rather than merely chosen? Give the function and measure that force it.
> 2. A student writes $\int_{\mathbb{R}} x\,dP(x) = 0$ for the Cauchy law "by symmetry." Which exact sentence of their argument is false?
> 3. For the Cauchy law and for a Pareto law with $\mathbb{E}[X] = +\infty$: in each case, what does the sequence of sample means $\bar X_n$ actually do as $n\to\infty$?
> 4. Let $f = \mathbb{1}_{\{0\}}$ on $\mathbb{R}$. Compute $\int f\,d\lambda$, $\int f\,d\#$, and $\int f\,d\delta_0$.
> 5. State where the proof of MCT uses countable additivity, and in what disguise.

## 2.3 Expectation, and the theorem that lets you forget $\Omega$

**Definition 2.10** (expectation). Let $(\Omega,\mathcal{F},\mathbb{P})$ be a probability space and $X$ a random variable. Then
$$\mathbb{E}[X] := \int_\Omega X\,d\mathbb{P},$$
defined whenever the right side is (Definition 2.9). $X\in L^1(\mathbb{P})$ means $\mathbb{E}|X|<\infty$; we then say $X$ has a finite mean. For a random vector $X = (X_1,\dots,X_d)$, $\mathbb{E}[X] := (\mathbb{E}[X_1],\dots,\mathbb{E}[X_d])$ componentwise, defined when every component is in $L^1$.

One definition, no cases. Whether $X$ is discrete, continuous, mixed, or pathological, $\mathbb{E}[X]$ means the same thing: integrate the function $X$ against the measure $\mathbb{P}$ on the abstract space $\Omega$.

That is philosophically clean and practically useless, because you never know $\Omega$. In Chapter 1's framing $\omega$ is the random seed, and nobody integrates over the space of random seeds. What you know is the *law* $P_X$ on $\mathbb{R}^d$ — a Gaussian, a categorical, your model's output distribution. The following theorem says the abstract integral can always be replaced by an integral over $\mathbb{R}^d$ against the law. It is the workhorse of the subject; statisticians affectionately call it the *law of the unconscious statistician*, because everyone uses it without noticing there is anything to prove.

**Theorem 2.11** (change of variables / LOTUS). Let $X:\Omega\to\mathbb{R}^d$ be a random vector with law $P_X = X_\#\mathbb{P}$ (Definition 1.16), and let $g:\mathbb{R}^d\to\overline{\mathbb{R}}$ be Borel measurable. Then
$$\int_\Omega g(X(\omega))\,d\mathbb{P}(\omega) = \int_{\mathbb{R}^d} g(x)\,dP_X(x)$$
in the sense that if either side is defined, so is the other, and they are equal. In particular $\mathbb{E}[g(X)] = \int g\,dP_X$, and $g(X)\in L^1(\mathbb{P})\iff g\in L^1(P_X)$.

*Proof, by the standard ladder.* This four-rung argument recurs three more times in the chapter, so watch its shape: **indicators → simple → non-negative → general**.

*Rung 1 — indicators.* Let $g = \mathbb{1}_B$ for Borel $B\subseteq\mathbb{R}^d$. Then $g(X(\omega)) = \mathbb{1}_{X^{-1}(B)}(\omega)$, with $X^{-1}(B)\in\mathcal{F}$ by measurability, and Definition 2.3 gives
$$\int_\Omega\mathbb{1}_{X^{-1}(B)}\,d\mathbb{P} = \mathbb{P}(X^{-1}(B)) = P_X(B) = \int_{\mathbb{R}^d}\mathbb{1}_B\,dP_X,$$
the middle equality being the *definition* of the pushforward. Rung 1 carries all the content; the rest is bookkeeping.

*Rung 2 — simple.* If $g = \sum_{i=1}^k a_i\mathbb{1}_{B_i}$ with $a_i\ge0$, then $g\circ X = \sum_i a_i\mathbb{1}_{X^{-1}(B_i)}$, and Lemma 2.4(2) plus Rung 1 give equality.

*Rung 3 — non-negative.* Let $g\ge0$ be Borel. Take the simple approximants $s_n\uparrow g$ of Lemma 2.6. Then $s_n\circ X\uparrow g\circ X$ pointwise on $\Omega$ (composition preserves pointwise inequalities and limits). MCT (Theorem 2.8) on each side:
$$\int_\Omega g\circ X\,d\mathbb{P} = \lim_n\int_\Omega s_n\circ X\,d\mathbb{P} = \lim_n\int_{\mathbb{R}^d} s_n\,dP_X = \int_{\mathbb{R}^d} g\,dP_X.$$

*Rung 4 — general.* $(g\circ X)^\pm = g^\pm\circ X$; Rung 3 applies to each, so the positive parts agree and the negative parts agree; hence one side is defined iff the other is, and Definition 2.9 gives equality. $\square$

**Corollary 2.12** (the formulas from your first course, and Monte Carlo). Let $g$ be Borel with $g\ge0$ or $g\in L^1(P_X)$.
1. If $P_X$ has pmf $p$ on a countable set $S$ — meaning $P_X = \sum_{x\in S}p(x)\delta_x$ — then $\mathbb{E}[g(X)] = \sum_{x\in S}g(x)p(x)$.
2. If $P_X$ has Lebesgue density $p$ — meaning $dP_X = p\,d\lambda_d$ in the sense of §2.7 — then $\mathbb{E}[g(X)] = \int_{\mathbb{R}^d} g(x)p(x)\,dx$.
3. If $\hat P_n := \frac1n\sum_{i=1}^n\delta_{x_i}$ is the empirical distribution of a dataset $x_1,\dots,x_n$, then $\int g\,d\hat P_n = \frac1n\sum_{i=1}^n g(x_i)$.

*Proof.* (1) By Theorem 2.11 it suffices to integrate $g$ against $\sum_x p(x)\delta_x$. For an indicator this gives $\sum_{x\in B}p(x)$; the ladder extends it (Rung 3 uses Corollary 2.20 below, sums-and-integrals, to interchange the countable sum with the limit — that corollary depends only on MCT and linearity, so no circularity). (2) Is Proposition 2.28(1) once densities are defined; again a ladder argument. (3) Is the case $S = \{x_1,\dots,x_n\}$, $p(x_i) = 1/n$ of (1), allowing repeats. $\square$

Part 3 reframes something you do every day. **A Monte Carlo estimator is not an approximation to an integral; it is exactly an integral, against a different measure.** A minibatch loss $\frac1B\sum_i\ell(f_\theta(x_i),y_i)$ *is* $\int\ell\,d\hat P_B$, exactly. The approximation error lives entirely in the gap between the measures $\hat P_B$ and $P$, not in the integration. That shift of viewpoint is what makes concentration inequalities (Chapter 4) and empirical process theory (Chapter 7) possible: they are statements about how close $\hat P_n$ is to $P$, uniformly over a class of test functions.

**Worked Example 2.5** (the ReLU output, integrated two ways). Let $Z\sim\mathcal{N}(0,1)$, $A = \max(0,Z)$.

*Route 1 — treat $A$ as $g(Z)$ with $g(z) = \max(0,z)$ and integrate against the Gaussian law (Theorem 2.11):*
$$\mathbb{E}[A] = \int_{\mathbb{R}}\max(0,z)\,\varphi(z)\,dz = \int_0^\infty z\frac{e^{-z^2/2}}{\sqrt{2\pi}}\,dz = \frac{1}{\sqrt{2\pi}}\Big[-e^{-z^2/2}\Big]_0^\infty = \frac{1}{\sqrt{2\pi}}\approx0.3989423 .$$
(Numerical quadrature: $0.39894228$, SciPy `quad`, error estimate $1.1\times10^{-8}$.)

*Route 2 — treat $A$ as a random variable in its own right and integrate against its own law $P_A$.* Theorem 2.11 says this must give the same answer, and it does — but notice we cannot use Corollary 2.12(1) (no pmf) or (2) (no Lebesgue density: $P_A(\{0\})=\tfrac12$ while $\lambda(\{0\})=0$). We must fall back on the CDF or, better, wait for §2.7, where $P_A$ acquires an honest density with respect to $\delta_0+\lambda$ and the computation becomes $0\cdot\tfrac12+\int_0^\infty a\varphi(a)\,da = 1/\sqrt{2\pi}$ again.

The pedagogical point: the two routes agree by theorem, and Route 1 is available *right now* because the Gaussian law is friendly. That is why practitioners write $\mathbb{E}_{z\sim\mathcal{N}}[\ell(g_\theta(z))]$ rather than pushing the law forward through $g_\theta$ first. Keeping the expectation on the *latent* side, where the measure is simple, is the same manoeuvre as the reparameterization trick (§2.6), seen from another angle.

**Worked Example 2.6** (Monte Carlo as an integral, with numbers). $X\sim\mathcal{N}(0,1)$, $g(x)=x^2$, so $\mathbb{E}[g(X)] = 1$. Quadrature of $\int x^2\varphi\,dx$ gives $1.0000000$ (error estimate $5.3\times10^{-9}$). Monte Carlo (NumPy `default_rng(0)`):

| $n$ | estimate $\int g\,d\hat P_n$ | standard error |
|---|---|---|
| $10^4$ | $0.996197$ | $0.013984$ |
| $10^6$ | $1.001345$ | $0.001417$ |

Both estimates are *exact* integrals — against $\hat P_{10^4}$ and $\hat P_{10^6}$. The errors $-0.0038$ and $+0.0013$ are the price of replacing $P$ by $\hat P_n$, and they shrink like $n^{-1/2}$ because $\mathrm{Var}(X^2) = 2<\infty$. Hold that hypothesis in mind: §2.9 exhibits an estimator with variance $+\infty$, where this table would lie to you.

> **Common Confusion 2.3**
>
> *"To compute $\mathbb{E}[g(X)]$ don't I need the density of $g(X)$? My textbook computed the density of $Y=X^2$ first."*
>
> You can, and Chapter 3 shows how (Jacobians, plus the extra care needed when $g$ is not injective — $x^2$ being the standard trap). But you almost never *should*, and Theorem 2.11 is the reason: $\mathbb{E}[g(X)] = \int g\,dP_X$ needs only the law of $X$ and the function $g$; finding the law of $g(X)$ is strictly harder work you then throw away. The stakes are large: a neural network loss is a composition $g = \ell\circ f_\theta$, the law of $g(X)$ is intractable, and $\int g\,dP_X$ is estimated by a forward pass on a minibatch. Every training loop is Theorem 2.11, applied in the direction textbooks avoid.

> **Self-check 2.3** *(answers in Appendix 2.A)*
>
> 1. State the four rungs of the standard ladder and identify the rung that carries the content in Theorem 2.11.
> 2. Why can Corollary 2.12(2) not be used to compute $\mathbb{E}[A]$ for $A = \max(0,Z)$?
> 3. In what precise sense is a minibatch average an *exact* integral?
> 4. Theorem 2.11 asserts an equivalence, $g(X)\in L^1(\mathbb{P})\iff g\in L^1(P_X)$, not just an implication. Where in the proof does the equivalence come from?

## 2.4 Properties of the integral, and why it cannot see null sets

**Proposition 2.13** (linearity and monotonicity).
1. If $f,g\ge0$ are measurable and $c\ge0$, then $\int(f+g)\,d\mu = \int f\,d\mu+\int g\,d\mu$ and $\int cf\,d\mu = c\int f\,d\mu$.
2. If $f,g\in L^1(\mu)$ and $a,b\in\mathbb{R}$, then $af+bg\in L^1(\mu)$ and $\int(af+bg)\,d\mu = a\int f\,d\mu+b\int g\,d\mu$.
3. If $f\le g$ and both integrals are defined, then $\int f\,d\mu\le\int g\,d\mu$.

*Proof.* (1) Take $s_n\uparrow f$, $t_n\uparrow g$ from Lemma 2.6; then $s_n+t_n\uparrow f+g$, and $\int(s_n+t_n) = \int s_n+\int t_n$ by Lemma 2.4(2). Apply MCT (Theorem 2.8) to all three sequences. Homogeneity is similar. (2) With $h := af+bg$: $|h|\le|a||f|+|b||g|$, so $\int|h|<\infty$ by (1) and monotonicity, giving $h\in L^1$. For additivity with signs, $h^+-h^- = f^+-f^-+g^+-g^-$ (taking $a=b=1$ first), hence $h^++f^-+g^- = h^-+f^++g^+$, an identity between non-negative functions; integrate with (1) and rearrange — legal because all terms are finite. For $a<0$ use $\int(-f) = -\int f$, immediate from swapping $f^\pm$. (3) In the $L^1$ case apply (2) to $g-f\ge0$; in the non-negative case this is Proposition 2.7(ii). $\square$

**Proposition 2.14** (a.e. blindness). Let $f,g$ be measurable.
1. If $f = g$ $\mu$-a.e. and $\int f\,d\mu$ is defined, then $\int g\,d\mu$ is defined and equal to it.
2. If $f\ge0$ then $\int f\,d\mu = 0\iff f = 0$ $\mu$-a.e.
3. If $f\in L^1(\mu)$ and $N$ is $\mu$-null, then $\int_N f\,d\mu = 0$.

*Proof.* (3) first, for $f\ge0$: every simple $s$ with $0\le s\le f\mathbb{1}_N$ vanishes off $N$, so in its canonical representation each $A_i$ with $a_i>0$ satisfies $A_i\subseteq N$, hence $\mu(A_i)=0$; so $\int s = 0$ and the supremum is $0$. For general $f\in L^1$ apply this to $f^\pm$.

(1) Let $N := \{f\ne g\}$, measurable and null. Decompose $f = f\mathbb{1}_N+f\mathbb{1}_{N^c}$, likewise $g$; the $N^c$ parts are the same function, the $N$ parts integrate to $0$ by (3); Proposition 2.13 finishes it, applied to positive and negative parts separately so no $\infty-\infty$ arises.

(2) ($\Leftarrow$) is (1) with $g=0$. ($\Rightarrow$) Let $A_t := \{f>t\}$ for $t>0$. Then $t\mathbb{1}_{A_t}\le f$, so $t\,\mu(A_t)\le\int f = 0$, giving $\mu(A_t)=0$. Since $\{f>0\} = \bigcup_n A_{1/n}$, countable subadditivity — the union bound, Proposition 1.8(2) — gives $\mu(\{f>0\}) = 0$. $\square$

**Proposition 2.15** (Markov's inequality). If $f\ge0$ is measurable and $t>0$, then $\mu(\{f\ge t\})\le\frac1t\int f\,d\mu$. Probabilistic form: $\mathbb{P}(|X|\ge t)\le\mathbb{E}|X|/t$.

*Proof.* $t\,\mathbb{1}_{\{f\ge t\}}\le f$ pointwise; integrate and use monotonicity. $\square$

That two-line proof is the seed of all of Chapter 4. Applied to $|X-\mathbb{E}X|^2$ it becomes Chebyshev; applied to $e^{s(X-\mathbb{E}X)}$ it becomes Chernoff, thence Hoeffding and the $\sqrt{\log(1/\delta)/n}$ in every generalization bound. It is also usually terrible on its own: if a validation loss has mean $0.04$, Markov gives $\mathbb{P}(\text{loss}\ge1)\le0.04$ — true, and nearly useless when the loss is concentrated near $0.04$. Chapter 4 is the story of what extra hypotheses buy exponentially better bounds.

### 2.4.1 The loud consequence

Proposition 2.14 is the most consequential result in this section and the one most often relegated to a footnote, so here it is loudly:

> **Integration cannot see null sets.** Two functions differing on a set of measure zero have the same integral over every measurable set. Consequently a *density* — an object defined through its integrals, as §2.7 will make precise — is never a single function. It is an equivalence class of functions modulo a.e. equality. There is no such thing as "the value of the density at $x$," full stop. You may specify a representative and evaluate it, but the number you get is a property of your choice of representative — unless an extra convention (continuity, below) pins the representative down.

**Worked Example 2.7** (two densities, one distribution). Let $p = \varphi$ be the standard normal density and define
$$\tilde p(x) := \begin{cases}10^6, & x\in\mathbb{Q},\\ \varphi(x), & x\notin\mathbb{Q}.\end{cases}$$
Then $\tilde p = \varphi$ off $\mathbb{Q}$, and $\lambda(\mathbb{Q})=0$, so by Proposition 2.14(1), $\int_B\tilde p\,d\lambda = \int_B\varphi\,d\lambda$ for every Borel $B$: the two functions define **the same probability measure**. Now ask the questions practitioners ask.

- "What is the density at $x=0$?" — $0.39894$ if you use $\varphi$; $10^6$ if you use $\tilde p$. As posed, the question has no answer.
- "Model 1 assigns density $2.3$ to this image and model 2 assigns $2.1$, so model 1 fits it better." — Taken literally, meaningless: both densities could be modified on a null set (which includes any finite set of images, indeed the entire set of images with rational pixel values) without changing either model as a probability measure.
- "But my normalizing flow prints a number." — It prints the value of the *particular continuous representative* your code computes, and that is fine and useful, for a reason worth spelling out. The representatives used in practice are continuous, and **a continuous function is determined by its values off any null set**. Proof in one breath: the complement of a $\lambda$-null set is dense in $\mathbb{R}$ (a null set contains no interval, since intervals have positive length), and two continuous functions agreeing on a dense set agree everywhere (their difference is continuous and vanishes on a dense set, hence vanishes identically). **The convention rescuing pointwise density comparisons is continuity, not measure theory.** If you compare densities at a point you are relying on a smoothness assumption, and you should know that you are.

What is meaningful *without* any convention is an integral: $P(B) = \int_B p\,d\lambda$ for a set of positive measure, an average log-density over a test set, a KL divergence. Those do not depend on the representative.

> **Common Confusion 2.4**
>
> *"But isn't $p(x)$ just the probability of $x$? My first course said $p(x)\,dx$ is the probability of landing in a tiny interval around $x$, so surely $p(x)$ is a probability per unit length, which is a real physical thing."*
>
> "Probability per unit length" is exactly right — and it is why $p(x)$ is *not* a probability. It is a **rate**: probability divided by reference length. A rate can exceed $1$ (§2.7 computes $39.894$ for $\mathcal{N}(0,0.01^2)$ at its mode) and changes when you change units, as a speed changes between km/h and m/s while the journey does not. The clean statement, proved in §2.7: for continuous $p$, $\lim_{h\to0}\frac{1}{2h}\mathbb{P}(X\in[x-h,x+h]) = p(x)$. Both the limit and the division by $2h$ are load-bearing.

> **Self-check 2.4** *(answers in Appendix 2.A)*
>
> 1. Give two functions on $\mathbb{R}$ differing at uncountably many points with the same integral against $\lambda$ over every Borel set.
> 2. Why is $\int f\,d\mu = 0$ with $f\ge0$ *not* enough to conclude $f\equiv0$?
> 3. Markov applied to a loss with mean $0.04$ gives what bound on $\mathbb{P}(\text{loss}\ge1)$? Why is it likely to be loose?
> 4. Under exactly what additional assumption does "model 1 assigns higher density to $x$ than model 2" become meaningful, and why does that assumption suffice?

## 2.5 The convergence theorems

Almost everything you want to do involves exchanging a limit with an integral. Training is a limit of empirical averages; a gradient is a limit of difference quotients; a series expansion is a limit of partial sums. So "when does $\int\lim f_n = \lim\int f_n$?" is not a technical aside; it is the technical core. MCT (Theorem 2.8) was the first answer; here are the rest.

The guiding intuition: **limits and integrals fail to commute exactly when mass escapes**, and mass has two escape routes. It can run off *horizontally*, keeping its height while sliding out to infinity; or *vertically*, concentrating into an ever taller, ever narrower spike. MCT forbids horizontal escape by only ever letting the functions increase (nothing can leave); dominated convergence forbids vertical escape by nailing an integrable ceiling over the whole sequence; Fatou forbids neither and consequently delivers only an inequality — in the safe direction, which is surprisingly often all you need. At the end of this section, Theorem 2.19 (Vitali) upgrades the intuition to a genuine equivalence: "no vertical escape" and "no horizontal escape" have exact names — uniform integrability and tightness — and together they are not merely sufficient but *necessary and sufficient* for the interchange. The two-escapes picture is a theorem, not a slogan.

**Theorem 2.16** (Fatou's lemma). For any measurable $f_n:\Omega\to[0,\infty]$,
$$\int_\Omega\liminf_n f_n\,d\mu\ \le\ \liminf_n\int_\Omega f_n\,d\mu .$$
No monotonicity, no domination, no convergence required.

*Proof.* Let $g_n := \inf_{m\ge n}f_m$, measurable by Proposition 1.15 (a *countable* inf — Chapter 1's insistence on countability is load-bearing here). Then $g_n\uparrow\liminf_m f_m$, and $g_n\le f_m$ for every $m\ge n$, so $\int g_n\le\inf_{m\ge n}\int f_m$. MCT applies to $(g_n)$:
$$\int\liminf_m f_m\,d\mu = \lim_n\int g_n\,d\mu\ \le\ \lim_n\inf_{m\ge n}\int f_m\,d\mu = \liminf_m\int f_m\,d\mu. \qquad\square$$

**Theorem 2.17** (dominated convergence, DCT). Let $f_n$ be measurable with $f_n\to f$ pointwise $\mu$-a.e., and suppose there is $g\in L^1(\mu)$ with $|f_n|\le g$ $\mu$-a.e. for every $n$. Then $f\in L^1(\mu)$ and
$$\lim_n\int f_n\,d\mu = \int f\,d\mu,\qquad\text{and in fact}\qquad \int|f_n-f|\,d\mu\to0 .$$

*Proof.* By Proposition 2.14 we may modify everything on a null set and assume the hypotheses hold everywhere. $|f|\le g$ gives $f\in L^1$. Apply Fatou to the two non-negative sequences $g+f_n\ge0$ and $g-f_n\ge0$:
$$\int(g+f)\le\liminf\int(g+f_n) = \int g+\liminf\int f_n \implies \int f\le\liminf\int f_n,$$
$$\int(g-f)\le\liminf\int(g-f_n) = \int g-\limsup\int f_n \implies \limsup\int f_n\le\int f.$$
(Linearity, Proposition 2.13(2), applies because $g,f,f_n\in L^1$; subtracting the finite $\int g$ is legal.) So $\limsup\int f_n\le\int f\le\liminf\int f_n$: the limit exists and equals $\int f$. For the final claim, apply the same argument to $h_n := |f_n-f|\le2g$, whose pointwise limit is $0$. $\square$

**Corollary 2.18** (bounded convergence). On a space of **finite total measure** — in particular a probability space — if $X_n\to X$ a.e. and $|X_n|\le M$ a.e. for a constant $M$, then $\int X_n\,d\mu\to\int X\,d\mu$.

*Proof.* DCT with $g\equiv M$, integrable precisely because $\int M\,d\mu = M\mu(\Omega)<\infty$. $\square$

This is the version you reach for most often in practice — bounded losses, clipped rewards, probabilities, indicators — and it is worth noticing that it is **false** on $(\mathbb{R},\lambda)$, where constants are not integrable: $f_n = \mathbb{1}_{[n,n+1]}$ is bounded by $1$ and converges pointwise to $0$, yet $\int f_n\,d\lambda = 1$ for every $n$. Finite total measure is doing real work.

**Theorem 2.19** (Vitali convergence theorem; stated, proved in Chapter 5). Let $\mu$ be a finite measure, $f_n\in L^1(\mu)$, and $f_n\to f$ in measure (in particular, if a.e.). Then $\int|f_n-f|\,d\mu\to0$ **iff** the family $\{f_n\}$ is **uniformly integrable**:
$$\sup_n\int_{\{|f_n|>K\}}|f_n|\,d\mu\ \longrightarrow\ 0\quad\text{as } K\to\infty .$$
On an infinite-measure space one must additionally rule out horizontal escape ("tightness": $\sup_n\int_{E_K^c}|f_n|\,d\mu\to0$ along an exhausting sequence $E_K\uparrow\Omega$ of finite-measure sets).

Read the theorem as the honest version of this section's slogan. Uniform integrability says *no mass survives arbitrarily high in the range, uniformly in $n$* — the exact negation of vertical escape. Tightness says *no mass survives arbitrarily far out in the domain* — the exact negation of horizontal escape. Domination by a single $g\in L^1$ implies both (which is why DCT is a special case), and so does boundedness in $L^p$ for any $p>1$. Chapter 5 proves Vitali and uses uniform integrability as the right hypothesis for convergence of expectations under convergence in distribution; Exercise A3(c) below is the one-sequence special case.

**Corollary 2.20** (interchanging sums and integrals). Let $f_n\ge0$ be measurable. Then $\int\sum_{n=1}^\infty f_n\,d\mu = \sum_{n=1}^\infty\int f_n\,d\mu$ (both sides possibly $+\infty$). If instead $\sum_n\int|f_n|\,d\mu<\infty$, the same identity holds with signs allowed and every term finite.

*Proof.* For $f_n\ge0$, the partial sums $S_N := \sum_{n\le N}f_n$ increase to the full sum; apply finite additivity (Proposition 2.13(1)) and MCT. For the signed case, $g := \sum_n|f_n|$ has $\int g<\infty$ by the first part, and $g$ dominates every partial sum, $|S_N|\le g$; apply DCT. $\square$

**Worked Example 2.8** (MCT in action, and a mean that is $+\infty$). Let $X$ have Lebesgue density $p(x) = x^{-2}$ on $[1,\infty)$ (Pareto, tail index $1$). Let $f_n := \min(X,n)$. Then $f_n\uparrow X$ pointwise, so MCT gives $\mathbb{E}[\min(X,n)]\uparrow\mathbb{E}[X]$. Compute the left side exactly:
$$\mathbb{E}[\min(X,n)] = \int_1^n x\cdot x^{-2}\,dx + n\,\mathbb{P}(X>n) = \log n + n\cdot\tfrac1n = 1+\log n .$$
For $n = 10,10^2,10^3,10^6$: $3.3026,\ 5.6052,\ 7.9078,\ 14.8155$ — increasing without bound, so $\mathbb{E}[X] = +\infty$. Contrast Worked Example 2.3: here the integral is *defined* (the negative part is $0$) and equals $+\infty$, and correspondingly $\bar X_n\to+\infty$ a.s.; there it was *undefined*, and $\bar X_n$ converged to nothing at all.

This is also the mathematics behind a familiar frustration: truncating or clipping a heavy-tailed quantity always yields a finite answer that grows with the clip threshold. If your reported average gradient norm keeps rising as you raise the clipping threshold, you are watching $1+\log n$.

### 2.5.1 The two escapes, in full detail

**Worked Example 2.9** (horizontal escape). On $(\mathbb{R},\mathcal{B},\lambda)$ let $f_n := \mathbb{1}_{[n,n+1]}$. Then $\int f_n\,d\lambda = 1$ for every $n$; $f_n(x)\to0$ for every $x$ (once $n>x$, $x\notin[n,n+1]$); hence $\lim\int f_n = 1\ne0 = \int\lim f_n$.

*Which hypothesis fails where?* MCT: the sequence is not monotone ($f_1(1.5) = 1>0 = f_2(1.5)$). DCT: any $g\ge f_n$ for all $n$ satisfies $g\ge\mathbb{1}_{[1,\infty)}$, so $\int g = \infty$; no integrable dominator exists. Vitali: the family *is* uniformly integrable (it is bounded by $1$), but tightness fails — every $f_n$ beyond index $K$ carries its whole mass outside $E_K = [-K,K]$; the escape is genuinely horizontal. Fatou: applies, and yields $0\le1$ — a *strict* inequality. Fatou is telling the truth, and the truth is strict; that is the honest content of "mass escaped."

**Worked Example 2.10** (vertical escape, and VAE variance collapse). On $((0,1],\mathcal{B},\lambda)$ let $f_n := n\,\mathbb{1}_{(0,1/n]}$. Then each spike has unit area by design: $\int f_n\,d\lambda = n\cdot\tfrac1n = 1$; and $f_n(x)\to0$ for every $x\in(0,1]$ (once $n>1/x$). Again $\lim\int f_n = 1\ne0$, with Fatou strict.

*Which hypothesis fails?* MCT: not monotone. DCT: the smallest candidate ceiling is $g(x) = \sup_n f_n(x) = \lfloor1/x\rfloor$, and since $\lfloor1/x\rfloor>1/x-1$, $\int_0^1 g\,d\lambda\ge\int_0^1(1/x-1)\,dx = \infty$: no integrable dominator. Vitali: the space has finite measure so tightness is automatic; what fails is uniform integrability — $\int_{\{f_n>K\}}f_n\,d\lambda = 1$ for every $n>K$. The escape is genuinely vertical, and Vitali names it.

Now the machine learning content, because this example is not a pathology; it is a training run. Consider a VAE with Gaussian decoder $p_\theta(x|z) = \mathcal{N}(x;\mu_\theta(z),\sigma^2 I_d)$, $\sigma$ *learned*. The reconstruction term of the ELBO is
$$\mathbb{E}_q[\log p_\theta(x|z)] = -\frac d2\log(2\pi\sigma^2) - \frac{\|x-\mu_\theta(z)\|^2}{2\sigma^2}.$$
Suppose the decoder reconstructs a training point exactly, $\mu_\theta(z) = x$: the second term vanishes and the objective is $-\frac d2\log(2\pi\sigma^2)\to+\infty$ as $\sigma\to0$. **The objective is unbounded above.** The optimizer, being an optimizer, finds this: $\sigma$ collapses, the density piles into a spike over the training points, and the reported log-likelihood improves without limit while the model gets worse. This is exactly $f_n = n\mathbb{1}_{(0,1/n]}$ with $n = 1/\sigma$; the reason it is possible at all is §2.7 — a density is a rate, and rates are unbounded. Every practical countermeasure (fixing $\sigma$, lower-bounding it, uniform dequantization noise as in Exercise C3, a discretized likelihood) amounts to installing the missing integrable ceiling.

The mirror-image failure is *posterior collapse*, where the KL term of the ELBO wins instead: $q_\phi(z|x)$ converges to the prior, the latent carries no information about $x$, and the decoder becomes unconditional. There the pathology is not an escaping spike but a genuine optimum of the objective you wrote down; §2.8 explains why reverse-KL objectives are prone to it.

*Figure 2.2 (described).* Left: three unit-area rectangles of height $1$, width $1$, marching right with supports $[1,2],[2,3],[3,4]$ — horizontal escape. Right: three unit-area rectangles anchored at the origin, heights $1,2,4$, widths $1,\tfrac12,\tfrac14$ — vertical escape. Both converge pointwise to $0$ while every integral equals $1$. MCT outlaws the left picture; a dominating ceiling outlaws the right one; Vitali names both; Fatou survives both, strictly.

> **Common Confusion 2.5**
>
> *"If $f_n\to f$ pointwise, surely $\int f_n\to\int f$. Isn't that what convergence means?"*
>
> Pointwise convergence is a statement about each $\omega$ *separately*; the integral is a statement about all of $\Omega$ *at once*; nothing forces compatibility. Worked Examples 2.9 and 2.10 are counterexamples where every point sees $f_n(\omega)\to0$ yet every integral equals $1$. The correct reflex is not "pointwise convergence is worthless" but "pointwise convergence *plus a reason mass cannot escape*" — and Theorem 2.19 says that reflex is exactly right, with "reason" = uniform integrability plus tightness. In practice the reason is usually a bound you already believe: a bounded loss ($0\le\ell\le1$ dominates trivially on a probability space), a Lipschitz condition, a compact parameter set. When none holds — unbounded losses, learned variances, unnormalized energies — expect the pathology, because the pathology is what the mathematics says will happen.

> **Self-check 2.5** *(answers in Appendix 2.A)*
>
> 1. For $f_n = \mathbb{1}_{[n,n+1]}$: which hypothesis of MCT fails, which of DCT fails (with the dominating-function computation), and which half of Vitali's condition fails?
> 2. Fatou gives $\int\liminf\le\liminf\int$. Give one example where the inequality is strict and one where equality holds without monotonicity or domination.
> 3. Why is a bounded loss automatically dominated on a probability space but not on $(\mathbb{R},\lambda)$?
> 4. In the VAE variance-collapse story, which escape occurs, and what plays the role of $n$?

## 2.6 Differentiating under the expectation — closing Failure 3

Every gradient-based learning algorithm computes something of the form
$$\nabla_\theta\,\mathbb{E}[h(\theta,X)]\quad\text{and hopes it equals}\quad\mathbb{E}[\nabla_\theta h(\theta,X)].$$
The hope is the SGD unbiasedness assumption from Chapter 1: $\mathbb{E}[g_t\mid\mathcal{F}_t] = \nabla F(\theta_t)$. It is an interchange of two limits — a derivative is a limit of difference quotients, an expectation an integral — and Failure 3 was precisely the observation that nobody had told you when it is legal. Here is the theorem.

**Theorem 2.21** (differentiation under the integral sign). Let $(\mathcal{X},\mathcal{G},\mu)$ be a measure space, $\Theta\subseteq\mathbb{R}$ an open interval, and $h:\Theta\times\mathcal{X}\to\mathbb{R}$ satisfying:
1. for each $\theta\in\Theta$, $h(\theta,\cdot)\in L^1(\mu)$;
2. for $\mu$-a.e. $x$, the map $\theta\mapsto h(\theta,x)$ is differentiable on all of $\Theta$;
3. **(uniform integrable domination)** there exists $g\in L^1(\mu)$, *not depending on $\theta$*, with
$$|\partial_\theta h(\theta,x)|\le g(x)\qquad\text{for all }\theta\in\Theta\text{ and }\mu\text{-a.e. }x .$$
Then $H(\theta) := \int_{\mathcal{X}}h(\theta,x)\,d\mu(x)$ is differentiable on $\Theta$ and
$$H'(\theta) = \int_{\mathcal{X}}\partial_\theta h(\theta,x)\,d\mu(x).$$
The same holds coordinatewise for open $\Theta\subseteq\mathbb{R}^m$, giving $\nabla_\theta H = \int\nabla_\theta h\,d\mu$.

*Proof.* Fix $\theta$ and any sequence of non-zero reals $t_n\to0$ with $\theta+t_n\in\Theta$. Define
$$\Delta_n(x) := \frac{h(\theta+t_n,x)-h(\theta,x)}{t_n},$$
measurable (difference of $L^1$ functions over a constant), with $\Delta_n(x)\to\partial_\theta h(\theta,x)$ for a.e. $x$ by hypothesis 2. By the **mean value theorem** applied to $s\mapsto h(s,x)$ on the interval between $\theta$ and $\theta+t_n$, there is $\xi_{n,x}$ strictly between them with
$$\Delta_n(x) = \partial_\theta h(\xi_{n,x},x),\qquad\text{hence}\qquad |\Delta_n(x)|\le g(x)$$
by hypothesis 3 — and *this* is where uniformity in $\theta$ is indispensable, since we have no control over where $\xi_{n,x}$ lands. DCT (Theorem 2.17) with dominator $g$ now gives
$$\lim_n\frac{H(\theta+t_n)-H(\theta)}{t_n} = \lim_n\int\Delta_n\,d\mu = \int\partial_\theta h(\theta,x)\,d\mu(x),$$
using linearity for the first equality. The limit is the same along every sequence $t_n\to0$, so $H'(\theta)$ exists and equals it. $\square$

Three remarks on how this gets misused. First, hypothesis 3 must hold with a *single* $g$ valid throughout a neighbourhood of $\theta$; a bound $|\partial_\theta h|\le g_\theta(x)$ that degrades as $\theta$ moves proves nothing (the $\xi_{n,x}$ in the proof lands at unknown parameter values). Second, it is a hypothesis about the *derivative*, not about $h$; bounding $h$ is not enough. Third, in practice one verifies it on a compact $\Theta_0\subseteq\Theta$ and concludes differentiability on the interior of $\Theta_0$ — all one ever needs, since differentiability is local. When a paper says "assume standard regularity conditions," this is what it is assuming, and it is exactly the set of conditions Chapter 8 needs for the Cramér–Rao bound.

**Worked Example 2.11** (a verification you can check line by line). Let $X\sim\mathcal{N}(\mu,1)$ and $H(\mu) := \mathbb{E}[X^2] = \mu^2+1$, so $H'(\mu) = 2\mu$; we recover this from Theorem 2.21 and watch every hypothesis hold. Write the expectation against the *latent* measure: $X = \mu+\varepsilon$, $\varepsilon\sim\mathcal{N}(0,1)$; so $\mathcal{X} = \mathbb{R}$, $d\mu(\varepsilon) = \varphi(\varepsilon)\,d\varepsilon$, $h(\mu,\varepsilon) = (\mu+\varepsilon)^2$.
- Hypothesis 1: $\int(\mu+\varepsilon)^2\varphi\,d\varepsilon = \mu^2+1<\infty$. ✓
- Hypothesis 2: $\mu\mapsto(\mu+\varepsilon)^2$ is a polynomial. ✓
- Hypothesis 3: $\partial_\mu h = 2(\mu+\varepsilon)$, so for $|\mu|\le M$: $|\partial_\mu h|\le2M+2|\varepsilon| =: g(\varepsilon)$, and $\int g\varphi\,d\varepsilon = 2M+2\sqrt{2/\pi} = 2M+1.5958<\infty$. ✓

Hence $H'(\mu) = \mathbb{E}[2(\mu+\varepsilon)] = 2\mu$. Hypothesis 3 held only on bounded intervals — which is fine: differentiability is local.

### 2.6.1 Application (a): the reparameterization trick

Here is the difficulty the trick solves. We want $\nabla_\theta\,\mathbb{E}_{x\sim P_\theta}[f(x)]$, where $\theta$ indexes the *measure*. Theorem 2.21 does not apply as stated: the integrand $f$ does not depend on $\theta$ at all — the parameter sits inside the measure, where derivatives have no meaning.

**Corollary 2.22** (reparameterization gradient). Suppose $P_\theta = (T_\theta)_\#Q$ for a fixed measure $Q$ on a space $\mathcal{E}$ and measurable maps $T_\theta:\mathcal{E}\to\mathbb{R}^d$ — that is, $x = T_\theta(\varepsilon)$ with $\varepsilon\sim Q$ **independent of $\theta$**. Let $f:\mathbb{R}^d\to\mathbb{R}$ and $h(\theta,\varepsilon) := f(T_\theta(\varepsilon))$. If $h$ satisfies the hypotheses of Theorem 2.21 with respect to $Q$, then
$$\nabla_\theta\,\mathbb{E}_{x\sim P_\theta}[f(x)] = \nabla_\theta\,\mathbb{E}_{\varepsilon\sim Q}[f(T_\theta(\varepsilon))] = \mathbb{E}_{\varepsilon\sim Q}[\nabla_\theta f(T_\theta(\varepsilon))].$$

*Proof.* The first equality is change of variables (Theorem 2.11) applied to $P_\theta = (T_\theta)_\#Q$: $\int f\,dP_\theta = \int f\circ T_\theta\,dQ$. The second is Theorem 2.21. $\square$

That two-line proof is the entire content of Kingma and Welling's trick, and the manoeuvre deserves a name, because it is measure-theoretic, not numerical:

> **The reparameterization trick moves $\theta$ out of the measure and into the integrand.** Before: $\int f\,dP_\theta$, with the parameter inside an object that cannot be differentiated. After: $\int f\circ T_\theta\,dQ$, with the parameter inside a function, where the chain rule and Theorem 2.21 live. The identity licensing the move is the change-of-variables theorem — a pushforward integral can be computed upstream.

For the Gaussian case in every VAE, $T_{(\mu,\sigma)}(\varepsilon) = \mu+\sigma\odot\varepsilon$ with $Q = \mathcal{N}(0,I)$, and the gradient becomes $\mathbb{E}_\varepsilon[\nabla_\theta f(\mu_\theta+\sigma_\theta\odot\varepsilon)]$: ordinary backpropagation through a sampled $\varepsilon$. The requirement that $Q$ be $\theta$-free is what fails for discrete latents: there is no fixed noise source with a $\theta$-differentiable map producing a categorical sample (any such map is piecewise constant in $\theta$, with derivative zero a.e. and undefined at the jumps). Hence Gumbel–Softmax relaxations, straight-through estimators, or score-function methods.

### 2.6.2 Application (b): the score-function / REINFORCE estimator

**Proposition 2.23** (score-function estimator). Let $\{P_\theta\}$ have densities $p_\theta = dP_\theta/d\mu$ with respect to a fixed base measure $\mu$ (§2.7), and $f\in L^1(P_\theta)$. Assume:
1. $S := \{x: p_\theta(x)>0\}$ **does not depend on $\theta$** (the support is $\theta$-free);
2. $\theta\mapsto p_\theta(x)$ is differentiable for a.e. $x$, and there is $G\in L^1(\mu)$ with $|f(x)\,\partial_\theta p_\theta(x)|\le G(x)$ for all $\theta$ in a neighbourhood and a.e. $x$.

Then
$$\nabla_\theta\,\mathbb{E}_{P_\theta}[f] = \int f\,\nabla_\theta p_\theta\,d\mu = \mathbb{E}_{P_\theta}\big[f(X)\,\nabla_\theta\log p_\theta(X)\big].$$

*Proof.* The first equality is Theorem 2.21 applied to $h(\theta,x) = f(x)p_\theta(x)$ with dominator $G$. For the second: on $S$, $p_\theta>0$ and the chain rule gives $\nabla_\theta p_\theta = p_\theta\nabla_\theta\log p_\theta$, so
$$\int_S f\,\nabla_\theta p_\theta\,d\mu = \int_S f\,\nabla_\theta\log p_\theta\;p_\theta\,d\mu = \mathbb{E}_{P_\theta}[f\,\nabla_\theta\log p_\theta],$$
the last step converting a $\mu$-integral with density $p_\theta$ into a $P_\theta$-integral (Proposition 2.28(1)). Off $S$ the integrand vanishes for every $\theta$, by hypothesis 1, so nothing is lost. $\square$

*Remark (why hypothesis 1 is not decoration).* If the support depended on $\theta$, there would be a region where $p_\theta$ jumps from $0$ to positive as $\theta$ moves: $\partial_\theta p_\theta$ fails to exist there, and the honest derivative of $\int fp_\theta$ acquires a **boundary term** the score-function formula cannot see. Exercise A8 computes this exactly for $\mathrm{Unif}[0,\theta]$ — where the estimator returns $-\tfrac12$ for a true derivative of $+\tfrac12$: not merely biased but *wrong in sign*, so gradient ascent moves $\theta$ the wrong way. Deterministic policies, hard parameter-dependent action masks, and learned truncation points all violate hypothesis 1, and no amount of sampling fixes the resulting bias.

Contrast the two estimators as *design choices*, not tricks: reparameterization needs a differentiable $T_\theta$ and a $\theta$-free noise source; the score function needs a differentiable density with $\theta$-free support and only *evaluations* of $f$ — so $f$ may be a black box, non-differentiable, even a human rating. Where both apply, reparameterization usually has far lower variance. Quantified next.

**Worked Example 2.12** (variance of the two estimators, computed and simulated). Take $\frac{d}{d\mu}\mathbb{E}_{x\sim\mathcal{N}(\mu,1)}[x^2]$ at $\mu=1$; the truth is $2\mu = 2$.

*Reparameterized.* $x = \mu+\varepsilon$, single-sample estimator $\hat g_{\text{rep}} = 2(\mu+\varepsilon) = 2(1+\varepsilon)$: mean $2$, variance $4\,\mathrm{Var}(\varepsilon) = 4$.

*Score-function.* $\nabla_\mu\log p_\mu(x) = x-\mu = \varepsilon$, so $\hat g_{\text{sf}} = x^2(x-\mu) = (1+\varepsilon)^2\varepsilon$. With $\mathbb{E}\varepsilon = \mathbb{E}\varepsilon^3 = \mathbb{E}\varepsilon^5 = 0$, $\mathbb{E}\varepsilon^2 = 1$, $\mathbb{E}\varepsilon^4 = 3$, $\mathbb{E}\varepsilon^6 = 15$:
$$\mathbb{E}\hat g_{\text{sf}} = \mathbb{E}[\varepsilon+2\varepsilon^2+\varepsilon^3] = 2\ \checkmark,\qquad \mathbb{E}\hat g_{\text{sf}}^2 = \mathbb{E}[(1+\varepsilon)^4\varepsilon^2] = 1+18+15 = 34,$$
so $\mathrm{Var}(\hat g_{\text{sf}}) = 34-4 = 30$. **Variance ratio $30/4 = 7.5$.**

*Simulation* (NumPy `default_rng(42)`, $5000$ replications per batch size):

| batch $B$ | reparam. mean | reparam. var | score mean | score var | ratio |
|---|---|---|---|---|---|
| $1$ | $1.9602$ | $3.9956$ | $1.9209$ | $29.3461$ | $7.34$ |
| $10$ | $1.9989$ | $0.4033$ | $2.0120$ | $3.0901$ | $7.66$ |
| $100$ | $1.9990$ | $0.0401$ | $2.0052$ | $0.3090$ | $7.71$ |

Both unbiased; variances match the predicted $4/B$ and $30/B$; the ratio hovers around $7.5$ (a variance estimate from $5000$ replications carries a few percent of Monte Carlo noise itself). Practical reading: matching the reparameterized gradient's quality requires $\approx7.5\times$ the batch — a $7.5\times$ compute multiplier, in a *benign one-dimensional* problem. In real problems it is worse, because score-function variance grows with the dimension of $\theta$ and the scale of $f$, while the reparameterized variance inherits the smoothness of $f$. Exercise C1 shows a well-chosen baseline reduces $30$ to $14$ — better, still $3.5\times$ worse than reparameterization.

### 2.6.3 Application (c): the score has mean zero

**Proposition 2.24** (zero-mean score). Let $p_\theta = dP_\theta/d\mu$ with $\theta$-free support $S$, suppose $\theta\mapsto p_\theta(x)$ is differentiable for a.e. $x$, and suppose there is $G\in L^1(\mu)$ with $|\nabla_\theta p_\theta(x)|\le G(x)$ for all $\theta$ near $\theta_0$ and a.e. $x$. Then
$$\mathbb{E}_{P_\theta}[\nabla_\theta\log p_\theta(X)] = 0\in\mathbb{R}^m .$$

*Proof.* Differentiate the normalization constraint: $\int_S p_\theta\,d\mu = 1$ for every $\theta$. The right side is constant, so its gradient is $0$; the left side may be differentiated inside the integral by Theorem 2.21 with dominator $G$:
$$0 = \nabla_\theta\int_S p_\theta\,d\mu = \int_S\nabla_\theta p_\theta\,d\mu = \int_S p_\theta\,\nabla_\theta\log p_\theta\,d\mu = \mathbb{E}_{P_\theta}[\nabla_\theta\log p_\theta]. \qquad\square$$

Three sentences, and they underwrite a lot of machinery. (i) The score is a valid *control-variate direction*: since its mean is zero, adding any constant multiple of it to an unbiased estimator preserves unbiasedness — which is exactly why REINFORCE baselines work (Exercise C1). (ii) The Fisher information $I(\theta) := \mathbb{E}[\nabla\log p_\theta\,\nabla\log p_\theta^\top]$ is therefore the *covariance* of the score, not merely its second moment — which is what makes it a natural Riemannian metric on parameter space, the object natural-gradient methods and K-FAC approximate. (iii) The "regularity conditions" in Chapter 8's Cramér–Rao bound are *literally the hypotheses of Theorem 2.21*; when a textbook says "assume we may differentiate under the integral," it means: exhibit the dominating $G$. When the support depends on $\theta$ — $\mathrm{Unif}[0,\theta]$ is the classic case — the score does not have mean zero, Cramér–Rao does not apply, and estimators can "beat" its bound. Not a paradox; a hypothesis violation (Exercise A8(d)).

> **Common Confusion 2.6**
>
> *"The reparameterization trick is a variance-reduction trick, right? Both estimators compute the same gradient; one is just noisier."*
>
> Half right, and the wrong half matters. Both are unbiased for the same quantity, so yes, the difference is variance. But they are not two Monte Carlo schemes for one integral — they are integrals of *different integrands* against *different measures*. Reparameterization integrates $\nabla_\theta(f\circ T_\theta)$ against the fixed noise law $Q$; the score-function estimator integrates $f\cdot\nabla_\theta\log p_\theta$ against $P_\theta$. The first uses the derivative of $f$ — it knows the geometry of how $x$ moves when $\theta$ moves; the second uses only evaluations of $f$ and learns nothing about $f$'s shape. That is why the variance gap is structural, why no clever sampling turns REINFORCE into a reparameterized gradient, and why their applicability conditions differ: reparameterization fails for discrete latents; the score function fails for $\theta$-dependent supports.

> **Self-check 2.6** *(answers in Appendix 2.A)*
>
> 1. In Theorem 2.21, why must the dominating $g$ be independent of $\theta$? Point at the exact step of the proof that breaks otherwise.
> 2. What does the reparameterization trick do to the *location* of $\theta$, and which theorem licenses the move?
> 3. Which hypothesis of Proposition 2.23 fails for $\mathrm{Unif}[0,\theta]$, and what goes wrong as a result?
> 4. Why does $\mathbb{E}[\nabla_\theta\log p_\theta] = 0$ imply that subtracting a constant baseline from $f$ preserves unbiasedness of the score-function estimator?

## 2.7 Radon–Nikodym — closing Failures 1 and 2

### 2.7.1 Absolute continuity: "$\nu$ cannot put mass where $\mu$ sees nothing"

**Definition 2.25** (absolute continuity, singularity, equivalence). Let $\mu,\nu$ be measures on $(\Omega,\mathcal{F})$.
- $\nu$ is **absolutely continuous** with respect to $\mu$, written $\nu\ll\mu$, if $\mu(A) = 0\implies\nu(A) = 0$ for every $A\in\mathcal{F}$.
- $\mu,\nu$ are **mutually singular**, written $\mu\perp\nu$, if there is $A\in\mathcal{F}$ with $\mu(A^c) = 0$ and $\nu(A) = 0$: they live on disjoint sets.
- They are **equivalent** if $\nu\ll\mu$ and $\mu\ll\nu$; then they share null sets, hence share "almost everywhere."

Say $\nu\ll\mu$ aloud as: *$\nu$ cannot put mass where $\mu$ sees nothing.* It is a one-directional implication about null sets, and the direction matters: $\mu$ is the *reference*; it may see more than $\nu$ (Lebesgue measure sees $[2,3]$ where a law supported on $[0,1]$ puts nothing) but never less.

Examples and non-examples, all used later:
- $\delta_0\not\ll\lambda$: $\lambda(\{0\}) = 0$ but $\delta_0(\{0\}) = 1$. **A point mass has no Lebesgue density** — the entire reason Failure 2 was a failure.
- $\lambda\not\ll\delta_0$: $\delta_0([1,2]) = 0$ while $\lambda([1,2]) = 1$. Indeed $\lambda\perp\delta_0$: take $A = \{0\}$ in the definition.
- $\mathcal{N}(0,1)$ and $\lambda$ are **equivalent**, since $\varphi>0$ everywhere. This is why "$\lambda$-a.e." and "almost surely under a Gaussian" mean the same thing — used constantly, rarely stated.
- The circle law $P = G_\#\mathrm{Unif}[0,2\pi)$, $G(t) = (\cos t,\sin t)$: $P\not\ll\lambda_2$, because $S^1$ is Lebesgue-null in $\mathbb{R}^2$ (cover it by $O(1/\epsilon)$ squares of side $\epsilon$, total area $O(\epsilon)$) while $P(S^1) = 1$. **Every generative model whose latent dimension is below the ambient dimension is in this situation** — Failure 4's root cause, as Chapter 1 diagnosed.
- $\hat P_n = \frac1n\sum_i\delta_{x_i}\not\ll\lambda_d$: the empirical distribution of any finite dataset is a sum of atoms. The "data distribution" you actually possess never has a Lebesgue density, whatever the population distribution does.

**Theorem 2.26** (Radon–Nikodym). Let $\mu$ be $\sigma$-finite on $(\Omega,\mathcal{F})$ — meaning $\Omega = \bigcup_n\Omega_n$ with $\mu(\Omega_n)<\infty$ — and let $\nu$ be a $\sigma$-finite measure with $\nu\ll\mu$. Then there exists a measurable $f:\Omega\to[0,\infty)$ with
$$\nu(A) = \int_A f\,d\mu\qquad\text{for every }A\in\mathcal{F},$$
and $f$ is unique up to $\mu$-a.e. equality. Write $f = \dfrac{d\nu}{d\mu}$: the **Radon–Nikodym derivative**, or the **density of $\nu$ with respect to $\mu$**.

We do not prove this. The standard proof (via the Lebesgue decomposition, or von Neumann's Hilbert-space argument using the $L^2$ projection theorem) is a page of real analysis that teaches less per page than anything else here; it is in every reference of §2.12. What we will do is take it completely seriously, because it changes what the word "density" means.

$\sigma$-finiteness is not decoration: counting measure on an *uncountable* set is not $\sigma$-finite, and there $\lambda\ll\#$ holds trivially (the only $\#$-null set is $\emptyset$) while no density $d\lambda/d\#$ exists. But every measure met in practice is $\sigma$-finite: probability measures are finite; $\lambda_d$ is covered by boxes; counting measure on a countable set is $\sigma$-finite.

### 2.7.2 What a density actually is

> **A density is not a probability. It is a ratio of two measures — a local exchange rate.**
>
> $\dfrac{d\nu}{d\mu}(x)$ answers: *per unit of $\mu$-stuff near $x$, how much $\nu$-stuff is there?* Like any exchange rate it can be any non-negative number, it has units ($\nu$-units per $\mu$-unit), and it changes when you change the reference. Therefore **there is no such thing as "the" density of a distribution** — only the density with respect to a reference measure you have chosen, usually silently. Write $p(x)$ and you have made a choice; the choice is almost always $\lambda_d$ for continuous data and counting measure for discrete data, and every formula you know is haunted by it.

**Definition 2.27** (pdf, pmf, and everything else, unified). Let $P$ be a law on $\mathbb{R}^d$ and $\mu$ a $\sigma$-finite reference with $P\ll\mu$. Then $p := dP/d\mu$ is *the density of $P$ with respect to $\mu$*, and $P(B) = \int_B p\,d\mu$ for every Borel $B$.
- $\mu = \lambda_d$: $p$ is what a first course calls the **pdf**.
- $\mu = \#$ on a countable $S$: $p$ is the **pmf**, and $p(x) = P(\{x\})$ genuinely *is* a probability, because $\#(\{x\}) = 1$ — the exchange rate is one-to-one, which is the only reason pmfs are safe to read as probabilities.
- $\mu = \delta_0+\lambda$: $p$ describes a **mixed** law with an atom at $0$ plus a continuous part — the ReLU output, computed below.

**Proposition 2.28** (the calculus of densities). All measures $\sigma$-finite on $(\Omega,\mathcal{F})$.
1. **(Integration against a density.)** If $\nu\ll\mu$ with $f = d\nu/d\mu$, then for measurable $g\ge0$ (or $gf\in L^1(\mu)$): $\int g\,d\nu = \int gf\,d\mu$.
2. **(Chain rule.)** If $\nu\ll\mu\ll\rho$ then $\nu\ll\rho$ and $\frac{d\nu}{d\rho} = \frac{d\nu}{d\mu}\cdot\frac{d\mu}{d\rho}$ $\rho$-a.e.
3. **(Inverse rule.)** If $\mu,\nu$ are equivalent then $\frac{d\mu}{d\nu} = \big(\frac{d\nu}{d\mu}\big)^{-1}$ $\nu$-a.e.
4. **(Additivity.)** If $\nu_1,\nu_2\ll\mu$ then $\frac{d(\nu_1+\nu_2)}{d\mu} = \frac{d\nu_1}{d\mu}+\frac{d\nu_2}{d\mu}$ $\mu$-a.e.

*Proof.* (1) The **standard ladder**, second appearance. Indicators: $\int\mathbb{1}_A\,d\nu = \nu(A) = \int_A f\,d\mu = \int\mathbb{1}_A f\,d\mu$ — Theorem 2.26 verbatim. Simple: linearity. Non-negative: $s_n\uparrow g$ from Lemma 2.6, note $s_nf\uparrow gf$, MCT on both sides. General: split into $g^\pm$.

(2) For $A\in\mathcal{F}$, apply (1) with $g = \mathbb{1}_A\frac{d\nu}{d\mu}$ and the pair $(\mu,\rho)$:
$$\nu(A) = \int_A\frac{d\nu}{d\mu}\,d\mu = \int_A\frac{d\nu}{d\mu}\frac{d\mu}{d\rho}\,d\rho,$$
so the product is *a* density of $\nu$ w.r.t. $\rho$, and uniqueness in Theorem 2.26 makes it *the* density $\rho$-a.e. Also $\rho(A) = 0\Rightarrow\mu(A) = 0\Rightarrow\nu(A) = 0$, so $\nu\ll\rho$.

(3) Take $\rho = \nu$ in (2): $\frac{d\nu}{d\nu} = 1$ $\nu$-a.e. (since $\nu(A) = \int_A1\,d\nu$, plus uniqueness), so $\frac{d\nu}{d\mu}\frac{d\mu}{d\nu} = 1$ $\nu$-a.e., and both factors are positive and finite $\nu$-a.e. because their product is $1$.

(4) Additivity of the integral plus uniqueness. $\square$

Proposition 2.28(1) is the identity you have used since your first course under the name "$\mathbb{E}[g(X)] = \int g(x)p(x)\,dx$." It is now a theorem with hypotheses, valid for *any* base measure — which is what makes one formula cover discrete, continuous, and mixed models. It also completes the proof of Corollary 2.12(2), as promised.

**Theorem 2.29** (Lebesgue decomposition; stated). Let $\mu,\nu$ be $\sigma$-finite on $(\Omega,\mathcal{F})$. Then $\nu$ decomposes uniquely as
$$\nu = \nu_{ac}+\nu_s,\qquad \nu_{ac}\ll\mu,\quad \nu_s\perp\mu .$$
On $\mathbb{R}$ with $\mu = \lambda$, every law therefore splits uniquely into a part with a Lebesgue density, an atomic part (countably many point masses), and a "singular continuous" part — atomless yet carried by a Lebesgue-null set. The third part is invisible in elementary courses but not empty: the Cantor distribution has a continuous CDF, no atoms, and lives on a null set; the circle law is the same phenomenon in $\mathbb{R}^2$, and is what every GAN generator produces. The decomposition also explains why we can always find *some* reference measure: for a law with atoms at $\{a_1,a_2,\dots\}$ plus an absolutely continuous part, $\mu = \lambda+\sum_i\delta_{a_i}$ works.

### 2.7.3 The ReLU output finally gets a density — Failure 2 closed

**Worked Example 2.13** (the density of $A = \max(0,Z)$, explicitly). From Chapter 1: $\mathbb{P}(A = 0) = \tfrac12$ and $F_A(a) = \Phi(a)$ for $a\ge0$. Choose the reference
$$\mu := \delta_0+\lambda\quad\text{on }(\mathbb{R},\mathcal{B}),$$
$\sigma$-finite ($\lambda$ plus a finite measure). *Check $P_A\ll\mu$:* if $\mu(B) = 0$ then $\delta_0(B) = 0$ and $\lambda(B) = 0$, so $0\notin B$ and $B$ is Lebesgue-null; then $P_A(B) = \mathbb{P}(Z\in B\cap(0,\infty)) = \int_{B\cap(0,\infty)}\varphi\,d\lambda = 0$. ✓

*Produce the density.* Define
$$f(a) := \begin{cases}\tfrac12, & a = 0,\\ \varphi(a), & a>0,\\ 0, & a<0,\end{cases}$$
and verify the defining property for arbitrary Borel $B$, splitting the $\mu$-integral by Proposition 2.28(4):
$$\int_B f\,d\mu = \underbrace{f(0)\,\mathbb{1}\{0\in B\}}_{\int_B f\,d\delta_0} + \underbrace{\int_{B\cap(0,\infty)}\varphi\,d\lambda}_{\int_B f\,d\lambda} = \tfrac12\mathbb{1}\{0\in B\}+\mathbb{P}(Z\in B\cap(0,\infty)) = P_A(B).\ \checkmark$$
(In the $\lambda$-integral, the value $f(0)$ is irrelevant because $\lambda(\{0\}) = 0$ — a.e. blindness at work.) So
$$\boxed{\ \frac{dP_A}{d(\delta_0+\lambda)}(a) = \tfrac12\text{ at }a = 0,\qquad = \varphi(a)\text{ on }(0,\infty),\qquad = 0\text{ on }(-\infty,0).\ }$$
**Failure 2 is closed.** The object Chapter 1 could only describe through its CDF has one density, exactly like a Gaussian or a Poisson — just with respect to a reference measure adapted to it. Nothing about $A$ was ever exceptional; the exceptional thing was the unspoken assumption that "density" means "$\lambda$-density."

Recompute $\mathbb{E}[A]$ from the density via Proposition 2.28(1) with $g(a) = a$:
$$\mathbb{E}[A] = \int a f(a)\,d\mu(a) = \underbrace{0\cdot\tfrac12}_{\text{atom}} + \underbrace{\int_0^\infty a\varphi(a)\,da}_{1/\sqrt{2\pi}} = \frac{1}{\sqrt{2\pi}} = 0.3989423,$$
matching Worked Example 2.5. The atom contributes zero to the *mean* only because its location is $0$; it contributes decisively elsewhere: $\mathbb{P}(A\le0) = \tfrac12$, and $\mathbb{E}[e^{-A}] = \tfrac12+\int_0^\infty e^{-a}\varphi(a)\,da$, where dropping the atom costs $0.5$ in absolute error. Monte Carlo verification, $10^6$ samples (`default_rng(1)`): $\hat{\mathbb{E}}[A] = 0.397900$ (SE $0.000583$), $\widehat{\mathbb{P}}(A = 0) = 0.4998$. And the "forgot the atom" computation — renormalizing $\varphi$ to a density on $(0,\infty)$, i.e., using $2\varphi$ — gives $0.7978846 = \sqrt{2/\pi}$, exactly **twice** the correct answer. That factor of two is the most common bug in code handling mixed distributions.

One more lesson from this example. Here the density's value *at the atom* is not conventional: $\mu(\{0\}) = 1>0$, so a.e. uniqueness pins $f(0)$ to exactly $\tfrac12$. Density values are meaningless only at points the reference measure ignores; where the reference has mass, the value is as real as a pmf entry — because it *is* one.

*Figure 2.3 (described).* Three panels, all showing the law of $A$. Against Lebesgue measure: the atom at $0$ cannot be represented ($\lambda(\{0\}) = 0$, $P_A(\{0\}) = \tfrac12$) — no density. Against counting measure: no countable set carries the continuous mass on $(0,\infty)$ — no density. Against $\delta_0+\lambda$: both parts visible, density $\tfrac12$ at the atom and $\varphi(a)$ on $(0,\infty)$. "The" density does not exist; *a* density does.

### 2.7.4 Densities are rates, hence unbounded — Failure 1 closed

If a density were a probability, it would be at most $1$. It is not, and here is the arithmetic. For $X\sim\mathcal{N}(0,\sigma^2)$ the $\lambda$-density peaks at
$$p(0) = \frac{1}{\sigma\sqrt{2\pi}} = \begin{cases}0.3989423, & \sigma = 1,\\ 3.9894228, & \sigma = 0.1,\\ 39.894228, & \sigma = 0.01,\end{cases}$$
unbounded as $\sigma\to0$. Nothing is wrong: $p(0) = 39.894$ says "probability per unit length near $0$ is about $40$," and since the relevant region has length $\approx0.01$, the probability stays below $1$. Formally, for continuous $p$,
$$\mathbb{P}(X\in[x-h,x+h]) = \int_{x-h}^{x+h}p\,d\lambda = 2h\,p(x)+o(h),\qquad\text{so}\qquad p(x) = \lim_{h\to0}\frac{1}{2h}\mathbb{P}(X\in[x-h,x+h]):$$
a probability *divided by a length*. **Failure 1 is closed.** A density exceeding $1$ is exactly as alarming as a speed exceeding $1$: it depends entirely on the units, and Radon–Nikodym is what makes the units explicit.

Which brings us to the units.

**Proposition 2.30** (rescaling shifts log-densities by $d\log c$). Let $X$ be a random vector in $\mathbb{R}^d$ with density $p_X$ w.r.t. $\lambda_d$, let $c>0$, and $Y := cX$. Then
$$p_Y(y) = c^{-d}\,p_X(y/c),\qquad\text{equivalently}\qquad \log p_Y(y) = \log p_X(y/c)-d\log c .$$

*Proof.* Use the scaling property $\lambda_d(cA) = c^d\lambda_d(A)$ (§2.12) and the ladder — third appearance. For $h = \mathbb{1}_A$, $A$ Borel:
$$\int h(x/c)\,d\lambda_d(x) = \lambda_d(\{x: x/c\in A\}) = \lambda_d(cA) = c^d\int h\,d\lambda_d,$$
and by linearity and MCT the identity extends to all measurable $h\ge0$. Now for Borel $B$:
$$\mathbb{P}(Y\in B) = \mathbb{P}(X\in c^{-1}B) = \int\mathbb{1}_{c^{-1}B}(x)\,p_X(x)\,d\lambda_d(x).$$
Apply the identity with $h(u) := \mathbb{1}_{c^{-1}B}(u)p_X(u)$, substituting $x = y/c$: $\int h\,d\lambda_d = c^{-d}\int h(y/c)\,d\lambda_d(y) = \int_B c^{-d}p_X(y/c)\,d\lambda_d(y)$, since $y/c\in c^{-1}B\iff y\in B$. Uniqueness in Theorem 2.26 identifies $p_Y$. $\square$

**Worked Example 2.14** (why two papers report different log-likelihoods for the same model). Images arrive as integers in $\{0,\dots,255\}$ per channel. Some papers model $x\in[0,255]^d$; others $u = x/255\in[0,1]^d$; others $u = x/256$. By Proposition 2.30 with $c = 1/255$,
$$\log p_U(u) = \log p_X(x)+d\log255,\qquad\log255 = 5.5412635\text{ nats}.$$
A model reporting $+3.2$ nats/dim on $[0,1]$-scaled data is reporting $3.2-5.5413 = -2.3413$ nats/dim on $[0,255]$-scaled data. **Same model, same measure, different reference units.** With the $256$ convention the shift is $\log256 = 5.5451774$ nats — exactly $8$ bits, the origin of the field's favourite unit:
$$\text{bits/dim} := \frac{-\log_2 p_X(x)}{d} = \frac{-\log_2 p_U(u)}{d}+8 .$$
The two honest conventions differ by $\log256-\log255 = 0.0039$ nats $= 0.0056$ bits/dim — small, but larger than the margin in many published comparisons. Exercise C3 makes you adjudicate a real version of this dispute. And note the sign: since positive $\log p$ is allowed (densities exceed $1$), bits/dim can in principle be negative, though for well-behaved image models it is not.

> **Common Confusion 2.7**
>
> *"Absolutely continuous, continuous distribution — same thing, right? A continuous random variable is one with a density."*
>
> Three different notions, and conflating them causes real errors. (a) *$P$ has no atoms*: $P(\{x\}) = 0$ for every $x$; equivalently $F_X$ is continuous (jumps of $F_X$ are atoms, Proposition 1.19). This is the informal "continuous random variable." (b) *$P\ll\lambda$*: $P$ has a Lebesgue density. (c) *$F_X$ absolutely continuous as a function* — the classical real-analysis notion, equivalent to (b) on $\mathbb{R}$. Now: (b)$\Rightarrow$(a), since $\lambda(\{x\}) = 0$ forces $P(\{x\}) = 0$. But (a)$\not\Rightarrow$(b): the Cantor distribution has a continuous CDF, no atoms, yet lives on a Lebesgue-null set — the "singular continuous" part of Theorem 2.29 — and the circle law is the same phenomenon in $\mathbb{R}^2$, produced by every GAN generator. *Atomless does not mean has-a-density.* When someone says "assume $X$ is continuous," ask which of (a) or (b) they need; if they are about to write $p(x)$, they need (b).

> **Self-check 2.7** *(answers in Appendix 2.A)*
>
> 1. State $\nu\ll\mu$ in words, and identify which implication is denied in $\delta_0\not\ll\lambda$ and in $\lambda\not\ll\delta_0$.
> 2. Why does $P_A$ for $A = \max(0,Z)$ have a density w.r.t. $\delta_0+\lambda$ but not w.r.t. $\lambda$ or $\#$?
> 3. A colleague's flow assigns density $150$ to a test image, "which is impossible since probabilities are at most 1." Answer them in two sentences.
> 4. Your model's average log-density on $[0,1]$-scaled $3\times32\times32$ images is $+3.2$ nats/dim. What is it on $[0,255]$-scaled data, and how many bits/dim (using the $256$ convention)?

## 2.8 Distances and divergences between measures — closing Failure 4

This section builds the toolkit for comparing two probability measures $P$ and $Q$: total variation, KL, the general $f$-divergence family, Jensen–Shannon, and Wasserstein. The organizing fact is the one §2.7 installed: everything computed from a *density ratio* $dP/dQ$ inherits the yes/no character of absolute continuity, and that inheritance is exactly why GAN training with an $f$-divergence loss can receive no gradient at all.

### 2.8.1 Total variation

**Definition 2.31** (total variation distance). For probability measures $P,Q$ on $(\Omega,\mathcal{F})$,
$$\mathrm{TV}(P,Q) := \sup_{A\in\mathcal{F}}|P(A)-Q(A)|\ \in[0,1].$$
If both have densities $p,q$ with respect to a common $\sigma$-finite $\mu$ (always possible: take $\mu = P+Q$), then
$$\mathrm{TV}(P,Q) = \tfrac12\int|p-q|\,d\mu ,$$
with the supremum attained at $A^\ast = \{p>q\}$.

*Proof of the identity.* For any $A$: $P(A)-Q(A) = \int_A(p-q)\,d\mu\le\int_{A\cap\{p>q\}}(p-q)\,d\mu\le\int(p-q)^+\,d\mu$, with equality at $A = A^\ast$. Since $\int(p-q)\,d\mu = 1-1 = 0$, we have $\int(p-q)^+ = \int(p-q)^- = \tfrac12\int|p-q|$. The same bound applies to $Q(A)-P(A)$ by symmetry. $\square$

TV is the bluntest instrument: the worst-case disagreement over *events*. It is a genuine metric, it is bounded by $1$, and $\mathrm{TV}(P,Q) = 1$ exactly when $P\perp Q$ (take $A$ from the definition of singularity). Two useful anchors for later: Scheffé's lemma (Exercise A6) says a.e. convergence of densities implies TV convergence, and **Pinsker's inequality** (stated in §2.12, used in Chapters 4–5) says $\mathrm{TV}(P,Q)\le\sqrt{\tfrac12 D_{\mathrm{KL}}(P\|Q)}$, so small KL forces small TV — but not conversely.

### 2.8.2 KL divergence, and why $+\infty$ is honest

**Definition 2.32** (KL divergence; cross-entropy). For probability measures $P,Q$ on $(\Omega,\mathcal{F})$,
$$D_{\mathrm{KL}}(P\|Q) := \begin{cases}\displaystyle\int_\Omega\log\frac{dP}{dQ}\,dP = \int_\Omega w\log w\,dQ,\quad w := \frac{dP}{dQ}, & P\ll Q,\\[2mm] +\infty, & P\not\ll Q.\end{cases}$$
If both have densities $p,q$ w.r.t. a common $\mu$, then $w = p/q$ on $\{q>0\}$ ($P$-a.s.) and $D_{\mathrm{KL}}(P\|Q) = \int p\log(p/q)\,d\mu$. The **cross-entropy** of $Q$ relative to $P$ is $H(P,Q) := -\int\log q\,dP$; when $P$ has density $p$ w.r.t. the same $\mu$, $H(P,Q) = H(P)+D_{\mathrm{KL}}(P\|Q)$ with $H(P) := -\int p\log p\,d\mu$ the entropy of $P$ *relative to $\mu$* — a reminder that "differential entropy" is not a property of a distribution alone (§2.7.4).

The second form, $\int w\log w\,dQ$, shows the integral is always defined: $t\log t\ge-1/e$ for $t\ge0$, so the negative part of the integrand is bounded and $\int(w\log w)^-\,dQ\le1/e<\infty$. Hence $D_{\mathrm{KL}}\in[0,+\infty]$ with no undefined case — unlike the Cauchy mean, KL never fails to exist; it merely takes the value $+\infty$.

**Proposition 2.33** (Gibbs' inequality). $D_{\mathrm{KL}}(P\|Q)\ge0$, with equality iff $P = Q$.

*Proof.* If $P\not\ll Q$, nothing to prove. Otherwise let $w = dP/dQ$; note $w>0$ $P$-a.s., since $P(w = 0) = \int_{\{w=0\}}w\,dQ = 0$. Use the tangent-line inequality $\log t\le t-1$ for $t>0$ (equality iff $t = 1$; concavity of $\log$ at $t = 1$) with $t = 1/w$:
$$-\log w\le\frac1w-1\ \ P\text{-a.s.}\implies D_{\mathrm{KL}}(P\|Q) = \int\log w\,dP\ \ge\ 1-\int\frac1w\,dP .$$
By Proposition 2.28(1), $\int\frac1w\,dP = \int\frac1w\,w\,dQ = Q(w>0)\le1$, so $D_{\mathrm{KL}}\ge1-Q(w>0)\ge0$.

Equality forces $Q(w>0) = 1$ and $\log w = 1-1/w$ $P$-a.s., i.e., $w = 1$ $P$-a.s. Then $P(w\ne1) = \int_{\{w\ne1\}}w\,dQ = 0$ gives $Q(\{w\ne1\}\cap\{w>0\}) = 0$; with $Q(w = 0) = 0$ we get $w = 1$ $Q$-a.e., hence $P(A) = \int_A w\,dQ = Q(A)$ for all $A$. $\square$

Now the practical face of $+\infty$. In $K$-class classification with a softmax, the loss on one example is $-\log q_y$, the negative log predicted probability of the true class:

| $q_y$ | $0.5$ | $0.1$ | $10^{-3}$ | $10^{-8}$ | $10^{-15}$ | $0$ |
|---|---|---|---|---|---|---|
| $-\log q_y$ (nats) | $0.6931$ | $2.3026$ | $6.9078$ | $18.4207$ | $34.5388$ | $+\infty$ |

This is not a numerical artefact to paper over; it is Definition 2.32 reporting that a model assigning probability zero to something that happened is infinitely wrong, and no finite amount of subsequent good behaviour compensates. The engineering response is dictated by the mathematics. Computing `log(softmax(z))` in two steps evaluates $\exp$ then $\log$: in float32, $\exp$ of arguments below about $-87$ leaves the normal range (and underflows to exactly $0$ near $-104$, once subnormals are exhausted), producing $-\infty$ or `NaN` downstream. The fused `log_softmax` computes $z_y-\log\sum_k e^{z_k}$ with max-subtraction and stays finite and accurate, because the *difference of logits* is well-conditioned even when the probability is $10^{-30}$. Logit clamping and label smoothing go further: they *bound* the loss — i.e., they replace the true objective with a different, finite one. A deliberate bias, chosen because an unbounded objective has unbounded gradients.

### 2.8.3 Forward versus reverse KL

$D_{\mathrm{KL}}$ is not symmetric, and the asymmetry is its single most consequential property in practice. Fix a target $P$ and a model family $\{Q_\phi\}$.

- **Forward KL, $D_{\mathrm{KL}}(P\|Q_\phi) = \int p\log(p/q_\phi)\,d\mu$.** The integral runs against $P$: wherever $p>0$ and $q_\phi\approx0$, the integrand blows up. The model is punished for *failing to cover* mass: forward KL is **mass-covering** (zero-avoiding). Maximum likelihood is forward KL: $\arg\max_\phi\frac1n\sum_i\log q_\phi(x_i) = \arg\min_\phi D_{\mathrm{KL}}(\hat P_n\|Q_\phi)$.
- **Reverse KL, $D_{\mathrm{KL}}(Q_\phi\|P) = \int q_\phi\log(q_\phi/p)\,d\mu$.** The integral runs against $Q_\phi$: regions where $q_\phi\approx0$ cost nothing, whatever $p$ does there. The model is punished only for putting mass where $P$ has none: reverse KL is **mode-seeking** (zero-forcing). Variational inference minimizes reverse KL, because that is the direction in which the intractable normalizing constant cancels (Exercise C2).

**Worked Example 2.15** (the two fits, computed). Target $P = \tfrac12\mathcal{N}(-3,1)+\tfrac12\mathcal{N}(3,1)$; family $Q = \mathcal{N}(m,s^2)$. Numerical minimization (Nelder–Mead over a fine grid representation, $4\times10^5$ points on $[-20,20]$):

| objective | optimal $m$ | optimal $s$ | value at its own optimum | value at the *other* optimum |
|---|---|---|---|---|
| forward $D_{\mathrm{KL}}(P\|Q)$ | $-0.0000$ | $3.1623$ | $0.4620$ | $7.8601$ |
| reverse $D_{\mathrm{KL}}(Q\|P)$ | $2.9843$ | $1.0234$ | $0.6888$ | $0.9380$ |

The forward fit sits at the midpoint with $s = 3.1623 = \sqrt{10}$ — exactly moment matching ($\mathbb{E}_P[X] = 0$, $\mathrm{Var}_P(X) = 1+9 = 10$); forward KL for an exponential-family model *is* moment matching, a fact Chapter 3 explains via sufficiency. It covers both modes and places substantial density in the empty valley near $0$. The reverse fit collapses onto one mode ($m\approx3$, $s\approx1$) and ignores the other entirely: the mixture places only $0.4797$ of its mass within $\pm2s$ of that fit — the model has quietly decided half the target does not exist. And look at the last column: the mode-seeking solution scores a catastrophic $7.86$ under forward KL, while the mass-covering solution scores only $0.94$ under reverse KL. **Reverse KL is the more forgiving objective — and that is precisely the problem: it is content with an answer that discards half the distribution.**

For VAEs this is not a curiosity. The ELBO's variational gap is $D_{\mathrm{KL}}(q_\phi(z|x)\,\|\,p_\theta(z|x))$ — a reverse KL (Exercise C2 derives the identity). Its zero-forcing behaviour is why learned posteriors are systematically **over-concentrated**: the true posterior may be multimodal or heavy-tailed, and a Gaussian $q_\phi$ is rewarded for picking one narrow region. Downstream: underestimated uncertainty, blurry decodes from the prior, and at the extreme, posterior collapse — where $q_\phi(z|x)$ ignores $x$ altogether, a perfectly comfortable resting place for a reverse-KL objective (Exercise A7(c) exhibits it as a stationary point of the loss you wrote down).

### 2.8.4 $f$-divergences, and the general blindness theorem — Failure 4 closed

The first draft of this chapter kept saying "every $f$-divergence" without defining one. Here is the definition, after which the GAN failure becomes a two-line theorem in full generality rather than a computation about one example.

**Definition 2.34** ($f$-divergence). Let $f:(0,\infty)\to\mathbb{R}$ be convex with $f(1) = 0$; extend it by $f(0) := \lim_{t\downarrow0}f(t)\in(-\infty,+\infty]$ and set $f'(\infty) := \lim_{t\to\infty}f(t)/t\in(-\infty,+\infty]$. For probability measures $P,Q$, pick any $\sigma$-finite $\mu$ dominating both (e.g., $\mu = P+Q$), write $p = dP/d\mu$, $q = dQ/d\mu$, and define
$$D_f(P\|Q) := \int_{\{q>0\}} q\,f\!\Big(\frac pq\Big)\,d\mu\ +\ f'(\infty)\cdot P(q = 0),$$
with the convention $\infty\cdot0 = 0$. The value does not depend on the choice of $\mu$ (Exercise A9). Jensen's inequality (Proposition 2.54 below) gives $D_f\ge f(1) = 0$.

Instances: $f(t) = t\log t$ gives $D_{\mathrm{KL}}(P\|Q)$ (with $f'(\infty) = \infty$, recovering the $+\infty$ convention when $P\not\ll Q$); $f(t) = -\log t$ gives reverse KL; $f(t) = \tfrac12|t-1|$ gives TV; $f(t) = (\sqrt t-1)^2$ gives squared Hellinger; $f(t) = (t-1)^2$ gives $\chi^2$. The Jensen–Shannon divergence, next, is also an $f$-divergence.

**Definition 2.35** (Jensen–Shannon divergence). For $P,Q$ with $M := \tfrac12(P+Q)$,
$$D_{\mathrm{JS}}(P\|Q) := \tfrac12 D_{\mathrm{KL}}(P\|M)+\tfrac12 D_{\mathrm{KL}}(Q\|M).$$
Both $P\ll M$ and $Q\ll M$ always hold, so $D_{\mathrm{JS}}$ is always finite; and $0\le D_{\mathrm{JS}}\le\log2$. The original GAN objective at its inner optimum over the discriminator equals $2D_{\mathrm{JS}}(P_{\text{data}}\|P_\theta)-\log4$, so minimizing the generator loss *is* minimizing $D_{\mathrm{JS}}$ (Goodfellow et al., 2014).

**Proposition 2.36** (mutual singularity pins every $f$-divergence). If $P\perp Q$, then
$$D_f(P\|Q) = f(0)+f'(\infty),$$
a constant depending only on $f$, not on $P$, $Q$, or how far apart they sit. In particular $D_{\mathrm{JS}}(P\|Q) = \log2$ and $\mathrm{TV}(P,Q) = 1$ for *all* mutually singular pairs, and $D_{\mathrm{KL}}(P\|Q) = +\infty$. Consequently, if $\{P_\theta\}$ is a family with $P_\theta\perp Q$ for all $\theta$ in an open set $U$, then $\theta\mapsto D_f(P_\theta\|Q)$ is constant on $U$ and its gradient is identically zero there.

*Proof.* Pick $A$ with $P(A^c) = 0$, $Q(A) = 0$, and take $\mu = P+Q$. Then $q = 0$ $\mu$-a.e. on $A$ and $p = 0$ $\mu$-a.e. on $A^c$; so on $\{q>0\}\subseteq A^c$ the ratio $p/q = 0$ $\mu$-a.e., giving $\int_{\{q>0\}}q\,f(p/q)\,d\mu = f(0)\,Q(\Omega) = f(0)$; and $P(q = 0) = P(A) = 1$ contributes $f'(\infty)$. The gradient claim: a constant function has zero gradient. $\square$

*Check against JS.* JS is $D_f$ for $f(t) = \tfrac t2\log\frac{2t}{t+1}+\tfrac12\log\frac{2}{t+1}$; one computes $f(0) = \tfrac12\log2$ and $f'(\infty) = \tfrac12\log2$, sum $\log2$. ✓ (You can also verify directly, as the first draft did: for $M = \tfrac12(P+Q)$ with $P\perp Q$, $\frac{dP}{dM} = 2\cdot\mathbb{1}_A$ $M$-a.e. — check the defining property: $\int_B2\mathbb{1}_A\,dM = P(A\cap B)+Q(A\cap B) = P(B)$ since $P(B\setminus A)\le P(A^c) = 0$ and $Q(A\cap B) = 0$ — whence $D_{\mathrm{KL}}(P\|M) = \int_A\log2\,dP = \log2$, symmetrically for $Q$.)

**Failure 4 is closed**, and note *how*. Chapter 1 established the root cause: for $G_\theta:\mathbb{R}^k\to\mathbb{R}^d$ with $k<d$, $\mathrm{supp}(P_\theta)$ is Lebesgue-null, and two such laws on different low-dimensional sets are (generically) mutually singular. Proposition 2.36 then says every $f$-divergence between generator and data is pinned to a constant. The discriminator can be perfect, the loss maximally bad, and the gradient exactly zero. Not slow learning — *no signal at all*.

The structural reason, stated once so it generalizes: **an $f$-divergence sees only the density ratio $dP/dQ$**, and between mutually singular measures that ratio is degenerate ($0$ or $\infty$) *everywhere* — degenerate in the same way whether the supports are $10^{-6}$ apart or $10^{6}$ apart, because absolute continuity is a yes/no property that knows nothing about distance. To get a divergence that sees distance, you must put the metric of the underlying space into the definition. That is Wasserstein.

### 2.8.5 Wasserstein distance — the fix, and its price

**Definition 2.37** (Wasserstein-1 distance). For laws $P,Q$ on $\mathbb{R}^d$ with finite first moments, let $\Pi(P,Q)$ be the set of **couplings**: probability measures $\pi$ on $\mathbb{R}^d\times\mathbb{R}^d$ (product $\sigma$-algebra, Definition 2.43) with marginals $P$ and $Q$. Then
$$W_1(P,Q) := \inf_{\pi\in\Pi(P,Q)}\int\|x-y\|\,d\pi(x,y).$$
$\Pi(P,Q)$ is never empty: it contains $P\otimes Q$ (Theorem 2.44). $W_1$ is a metric on laws with finite first moment; the triangle inequality uses a gluing construction found in the references (Villani, §2.12).

**Proposition 2.38** (two parallel segments). Let $P$ be uniform on $S_0 = \{(t,0): t\in[0,1]\}$ and $Q_\alpha$ uniform on $S_\alpha = \{(t,\alpha): t\in[0,1]\}$, $\alpha\ne0$. Then
$$W_1(P,Q_\alpha) = |\alpha|,\qquad\text{while}\qquad D_{\mathrm{JS}}(P\|Q_\alpha) = \log2,\quad D_{\mathrm{KL}}(P\|Q_\alpha) = +\infty,\quad\mathrm{TV} = 1 .$$

*Proof.* $P\perp Q_\alpha$ (take $A = S_0$), so the divergence claims are Proposition 2.36. *Lower bound for $W_1$:* any coupling $\pi$ has $\pi(S_0\times S_\alpha) = 1$ (its marginals give $\pi((S_0\times\mathbb{R}^2)^c) = P(S_0^c) = 0$, likewise for the second factor), and for $x\in S_0$, $y\in S_\alpha$, $\|x-y\|\ge|x_2-y_2| = |\alpha|$; hence every coupling costs $\ge|\alpha|$. *Upper bound:* let $T(t,0) := (t,\alpha)$ and $\pi^\ast := (\mathrm{id},T)_\#P$, the law of $(X,T(X))$: first marginal $P$, second $T_\#P = Q_\alpha$, cost $\|X-T(X)\| = |\alpha|$ identically. $\square$

**Worked Example 2.16** (the divergences, numerically, as singularity is approached). "$+\infty$" invites the suspicion of a convention chosen for tidiness. It is not: it is what the limit does. Smooth both laws in the $y$-coordinate by convolving with $\mathcal{N}(0,h^2)$, which makes them mutually absolutely continuous for every $h>0$; since the $x$-marginals coincide, everything reduces to comparing $\mathcal{N}(0,h^2)$ with $\mathcal{N}(\alpha,h^2)$, so $D_{\mathrm{KL}} = \alpha^2/(2h^2)$ exactly (Exercise A7), and $D_{\mathrm{JS}}$ is computed by quadrature ($2\times10^5$-point trapezoid):

| $h$ | $\alpha$ | $D_{\mathrm{KL}}$ | $D_{\mathrm{JS}}$ | $W_1$ |
|---|---|---|---|---|
| $0.5$ | $0.1$ | $0.0200$ | $0.004975$ | $0.1$ |
| $0.5$ | $1.0$ | $2.0000$ | $0.336831$ | $1.0$ |
| $0.1$ | $0.1$ | $0.5000$ | $0.111421$ | $0.1$ |
| $0.1$ | $0.5$ | $12.5000$ | $0.675943$ | $0.5$ |
| $0.1$ | $1.0$ | $50.0000$ | $0.693146$ | $1.0$ |
| $0.01$ | $0.1$ | $50.0000$ | $0.693146$ | $0.1$ |
| $0.01$ | $0.5$ | $1250.0000$ | $0.693147$ | $0.5$ |
| $0.01$ | $1.0$ | $5000.0000$ | $0.693147$ | $1.0$ |

Read the columns. As $h\to0$, $D_{\mathrm{KL}}\to\infty$: it genuinely diverges, matching Definition 2.32. $D_{\mathrm{JS}}$ saturates at $\log2 = 0.6931472$ and, crucially, becomes **flat in $\alpha$**: at $h = 0.01$, a *tenfold* increase in separation changes $D_{\mathrm{JS}}$ in the seventh decimal. A gradient computed from that signal is numerically zero — Proposition 2.36 arriving in the limit. Meanwhile $W_1 = \alpha$ throughout, with gradient $1$ in the right direction at every separation. That is the argument for Wasserstein GANs in one table. The middle rows also show the *other* fix used in practice: at $h = 0.5$ the smoothed JS *does* vary with $\alpha$ ($0.005\to0.34$), which is why adding instance noise to both real and fake samples restores a usable gradient — it manufactures mutual absolute continuity (Arjovsky–Bottou, 2017).

**What $W_1$ costs you — an honest accounting.** The first draft presented this section as "$W_1$ good, $f$-divergences bad," which oversells. Three caveats, each load-bearing in practice.
1. **Estimation is cursed by dimension.** The empirical $W_1(\hat P_n,P)$ converges at rate $n^{-1/d}$ for $d\ge3$ (up to logs) — exponentially slower than the $n^{-1/2}$ enjoyed by the expectations $f$-divergence estimators are built from. In high-dimensional image space, no feasible sample size estimates $W_1$ accurately.
2. **WGANs do not compute $W_1$.** They optimize the Kantorovich dual $\sup_{\|g\|_{\mathrm{Lip}}\le1}\{\mathbb{E}_P[g]-\mathbb{E}_Q[g]\}$ (§2.12) over a *neural critic* whose Lipschitz constraint is enforced only approximately (weight clipping, gradient penalty, spectral normalization). The trained objective is a biased, family-restricted surrogate for $W_1$ — usually a useful one, but claims that "WGAN minimizes the Wasserstein distance" are aspirational.
3. **Metric dependence.** $W_1$ inherits the geometry of the ground metric $\|\cdot\|$. In raw pixel space, Euclidean distance is a poor proxy for perceptual distance, so "small $W_1$" need not mean "similar images." Choosing the metric is choosing what the divergence can see — the mirror image of $f$-divergences choosing to see only the ratio.
The clean summary: $f$-divergences are statistically easy and geometrically blind; $W_1$ is geometrically aware and statistically hard. Instance noise and diffusion-style smoothing occupy the middle ground: they keep an $f$-divergence but destroy the singularity that blinds it.

*Figure 2.4 (described).* Left: two horizontal unit segments at heights $0$ and $\alpha$, with dashed vertical arrows — every coupling must transport mass a distance $|\alpha|$. Right: the divergences plotted against $\alpha$; $D_{\mathrm{JS}}$ is a horizontal line at $\log2$ (with an isolated value $0$ only at $\alpha = 0$), while $W_1$ is the line of slope $1$ through the origin.

> **Common Confusion 2.8**
>
> *"Infinite KL is just a numerical problem. I'll add $\epsilon$ to the denominator, or clip the ratio, and get on with my life."*
>
> Adding $\epsilon$ changes the objective to a different, finite one; it does not compute the original. Sometimes that is exactly right, and you should say so out loud: label smoothing, logit clamping, PPO ratio clipping, and V-trace truncation are all deliberate replacements of an unbounded quantity by a bounded surrogate, trading bias for controlled variance or gradient magnitude. What is not right is believing you computed $D_{\mathrm{KL}}$. Two traps. (a) When $P\not\ll Q$ the true value is $+\infty$, and your $\epsilon$-regularized answer is a function of $\epsilon$ that drifts as you change it: if your reported KL depends on your clamp threshold, you have measured your clamp. (b) In the GAN case, $\epsilon$ does not fix the flat loss, because the flatness is not underflow — Proposition 2.36 says the exact real number is exactly constant. That is why the successful fixes changed the *divergence* (Wasserstein) or the *supports* (instance noise), not the arithmetic.

> **Self-check 2.8** *(answers in Appendix 2.A)*
>
> 1. Why is $D_{\mathrm{KL}}(P\|Q)$ always defined (possibly $+\infty$) while $\mathbb{E}[X]$ can be undefined?
> 2. Which direction of KL is maximum likelihood, and which is variational inference? Which is mode-seeking?
> 3. Compute $f(0)+f'(\infty)$ for $f(t) = \tfrac12|t-1|$ and confirm it matches $\mathrm{TV}(P,Q)$ for $P\perp Q$.
> 4. Give the one-sentence reason $W_1$ has a useful gradient between mutually singular laws while no $f$-divergence does — and one sentence on what $W_1$ pays for the privilege.

## 2.9 Change of measure and importance sampling

Sometimes you must average under $P$ but can only sample from $Q$: off-policy reinforcement learning (data from a behaviour policy, evaluation of a target policy), covariate shift, rare-event estimation, tempered sampling. Radon–Nikodym solves this in one line.

**Proposition 2.39** (change of measure). Let $P\ll Q$ with $w := dP/dQ$, and let $f$ be measurable with $f\ge0$ or $f\in L^1(P)$. Then
$$\mathbb{E}_P[f] = \mathbb{E}_Q[w\,f].$$
Consequently, for $x_1,\dots,x_n\overset{\text{iid}}{\sim}Q$, the estimator $\hat I_n := \frac1n\sum_{i=1}^n w(x_i)f(x_i)$ is unbiased for $\mathbb{E}_P[f]$.

*Proof.* Proposition 2.28(1) with $g = f$, itself proved by the ladder; unbiasedness is linearity over $n$ identically distributed terms. $\square$

Two warnings; they are the reason this section exists.

**Warning 1: the requirement is $P\ll Q$, not the reverse.** $Q$ must cover $P$'s support. If $P$ puts mass where $Q$ cannot reach, $dP/dQ$ does not exist and the estimator is not merely high-variance — it is *systematically blind* to a region, with a bias no sample size shrinks. In off-policy RL: **the behaviour policy's support must contain the target policy's support**. A deterministic target policy is the extreme case: $P$ is a point mass on one action per state, and if the behaviour policy ever gives that action probability zero somewhere, importance-weighted evaluation is invalid there, forever. Exploration is not a nicety in off-policy methods; it is a hypothesis of the estimator.

**Warning 2: unbiasedness does not imply usefulness.** The estimator can be unbiased with infinite variance, converging to nothing at any usable rate — and, the part practitioners are rarely told, *you cannot detect this from the estimator's own reported uncertainty*.

**Proposition 2.40** (variance and the optimal proposal). In the setting of Proposition 2.39, with $I := \mathbb{E}_P[f]$,
$$\mathrm{Var}_Q(\hat I_n) = \frac1n\big(\mathbb{E}_Q[w^2f^2]-I^2\big) = \frac1n\big(\mathbb{E}_P[wf^2]-I^2\big),$$
finite **iff** $\mathbb{E}_Q[w^2f^2] = \int f^2\frac{(dP/d\mu)^2}{dQ/d\mu}\,d\mu<\infty$ for a common base $\mu$. Among all proposals $Q$ with $P\ll Q$, the variance is minimized by
$$q^\ast\ \propto\ |f|\,p,$$
and if $f$ has constant sign the minimal variance is exactly $0$.

*Proof.* The variance formula is the i.i.d.-average formula, the second form via $\mathbb{E}_Q[w^2f^2] = \mathbb{E}_P[wf^2]$ (Proposition 2.39 applied to $wf^2$). Optimality: with densities w.r.t. $\mu$, Cauchy–Schwarz (Proposition 2.52) on the factorization $|f|p = \big(|f|p/\sqrt q\big)\sqrt q$ gives
$$\Big(\int|f|p\,d\mu\Big)^2\le\int\frac{f^2p^2}{q}\,d\mu\cdot\int q\,d\mu = \mathbb{E}_Q[w^2f^2].$$
The left side is $Q$-free, hence a lower bound; Cauchy–Schwarz is tight iff the factors are proportional, i.e., $q\propto|f|p$. If $f\ge0$, then with $q^\ast = fp/I$ we get $wf = I$ identically — constant, zero variance. $\square$

**Worked Example 2.17** (a silent catastrophe, with numbers). Target $P = \mathcal{N}(0,1)$, integrand $f(x) = x^2$, truth $I = 1$; proposals $Q_\sigma = \mathcal{N}(0,\sigma^2)$. The weight is $w(x) = \varphi(x)/\varphi_\sigma(x)$, and
$$\mathbb{E}_{Q_\sigma}[w^2f^2] = \int x^4\frac{\varphi(x)^2}{\varphi_\sigma(x)}\,dx = \frac{\sigma}{\sqrt{2\pi}}\int x^4e^{-ax^2}\,dx,\qquad a := 1-\frac{1}{2\sigma^2},$$
finite **iff $a>0$, i.e., $\sigma>1/\sqrt2 = 0.7071$**, and then equal to $\frac{3\sigma}{4\sqrt2}a^{-5/2}$ (verified against quadrature at $\sigma = 1.5$: both give $1.4910723$; sanity check at $\sigma = 1$: $a = \tfrac12$, closed form $3 = \mathbb{E}[Z^4]$, as it must be since $w\equiv1$). Here is what $200$ independent trials of $n = 10^5$ samples each (`default_rng(2024)`) actually report:

| $\sigma$ | analytic $\mathrm{Var}(wf)$ | analytic sd of $\hat I_n$ | mean of 200 estimates | across-trial sd | median *reported* SE | coverage of nominal 95% CI |
|---|---|---|---|---|---|---|
| $1.5$ | $0.4911$ | $0.002216$ | $1.0002$ | $0.0023$ | $0.002216$ | $0.970$ |
| $1.0$ | $2.0000$ | $0.004472$ | $1.0003$ | $0.0048$ | $0.004471$ | $0.935$ |
| $0.9$ | $4.2674$ | $0.006533$ | $1.0001$ | $0.0070$ | $0.006477$ | $0.960$ |
| $0.8$ | $17.9568$ | $0.013400$ | $0.9990$ | $0.0125$ | $0.011295$ | $0.950$ |
| $0.72$ | $1607.78$ | $0.126798$ | $0.9963$ | $0.0250$ | $0.019570$ | $0.895$ |
| $0.6$ | $+\infty$ | $+\infty$ | $0.9786$ | $0.0809$ | $0.043450$ | $0.765$ |
| $0.45$ | $+\infty$ | $+\infty$ | $0.8652$ | $0.2370$ | $0.078368$ | $0.430$ |

Four lessons.

*(a) The best proposal is not the target.* At $\sigma = 1.5$ the estimator's sd is $0.00222$ — *half* that of sampling from $P$ itself ($0.00447$). Not a fluke: by Proposition 2.40 the optimum is $q^\ast\propto x^2\varphi(x)$, heavier-tailed than $\varphi$ because $f$ weights the tails. Sampling $q^\ast$ exactly (draw $|x|$ from a $\chi$ distribution with 3 degrees of freedom, attach a random sign) gives $\hat I = 1.0$ with across-sample sd $1.2\times10^{-16}$ — zero to machine precision, as the proposition predicts. Importance sampling is a variance-reduction technique in its own right, not only a workaround for "I can't sample $P$."

*(b) The catastrophe is invisible.* At $\sigma = 0.6$ and $0.45$ the variance is **provably infinite**, yet the median reported SE is a healthy-looking $0.043$ and $0.078$. A single run prints an estimate near $1$ with a $5\%$ error bar and everything looks fine. It is not: the nominal 95% CI covers the truth $76.5\%$ and $43.0\%$ of the time. Even in the *finite*-variance case $\sigma = 0.72$, the reported SE ($0.0196$) understates the true sd ($0.1268$) by a factor of $6$ — the variance lives in weights so rare that $10^5$ samples usually contain none. (Nor have $200$ trials found them: the across-trial sd $0.0250$ is *also* far below the analytic $0.1268$, for the same reason.)

*(c) Unbiasedness is no consolation.* The mean of the $200$ estimates at $\sigma = 0.45$ is $0.8652$, not $1$. The estimator *is* unbiased — the theorem is not wrong — but the distribution of $\hat I_n$ is so right-skewed that $2\times10^7$ total samples have not produced the rare enormous weights carrying the missing mass. Unbiasedness is a statement about an average over infinitely many runs, a situation you are never in.

*(d) Never trust the weight variance; use dedicated diagnostics.* Three, in increasing sophistication:
- **Effective sample size**, $\mathrm{ESS} := (\sum_i w_i)^2/\sum_i w_i^2\in[1,n]$: the number of equally-weighted samples your weighted sample is worth. Medians across the rows above: $83{,}142$ ($\sigma = 1.5$), $57{,}927$ ($0.72$), $16{,}239$ ($0.6$), $2{,}075$ ($0.45$), out of $10^5$. ESS falls off a cliff, unlike the reported SE.
- **Maximum weight share**, $\max_i w_i/\sum_j w_j$: medians $0.00002$, $0.00092$, $0.00381$, $0.01443$ on the same rows. When one sample in $10^5$ carries $1.4\%$ of the total weight, your "average" is a handful of points.
- **PSIS $\hat k$** (Vehtari et al., 2024): fit a generalized Pareto distribution to the largest weights; the shape estimate $\hat k$ is a tail index, with $\hat k\ge\tfrac12$ implying infinite variance and $\hat k\ge1$ infinite mean; the popular practical cutoff is $\hat k>0.7$. Our medians: $0.191$ ($\sigma = 0.9$), $0.345$ ($0.8$), $0.452$ ($0.72$), $0.591$ ($0.6$), $0.735$ ($0.45$). An honest flag about theory versus rule of thumb: at $\sigma = 0.6$ the variance is *infinite* and $\hat k = 0.591$ correctly crosses the theoretical $0.5$ boundary — but not the popular $0.7$ cutoff, so a practitioner following the rule of thumb would have shipped a provably infinite-variance estimator. When the question is variance, use the $0.5$ boundary; rules of thumb are calibrated for typical cases, and the boundary case is where you must think.

This is the mathematics behind two widely used tricks. **PPO** clips the policy ratio $w = \pi_\theta/\pi_{\theta_{\text{old}}}$ to $[1-\epsilon,1+\epsilon]$; **V-trace** (IMPALA) truncates importance weights at $\bar\rho$. Both deliberately introduce bias in exchange for bounded weights, hence bounded variance: they replace an unbounded Radon–Nikodym derivative with a bounded surrogate, giving up the exact identity $\mathbb{E}_P[f] = \mathbb{E}_Q[wf]$ to obtain a finite-variance estimator. A good trade — and recognizable as a trade only once you know the unclipped variance can be $+\infty$.

> **Self-check 2.9** *(answers in Appendix 2.A)*
>
> 1. Which absolute-continuity condition does importance sampling need, and what goes wrong in off-policy RL when it fails?
> 2. Why can the optimal proposal beat sampling from the target itself? What is $q^\ast$ for $f(x) = x^2$, $p = \varphi$?
> 3. One importance-sampling run reports estimate $1.003$, standard error $0.04$. What can you conclude, and what should you compute instead?
> 4. In what precise sense is PPO ratio clipping a bias–variance trade in the language of this section?

## 2.10 Product measures, Fubini–Tonelli, and the law of a dataset

### 2.10.1 The measure a dataset is drawn from

**Definition 2.41** (product $\sigma$-algebra and product measure). For measure spaces $(\Omega_1,\mathcal{F}_1,\mu_1)$, $(\Omega_2,\mathcal{F}_2,\mu_2)$, the **product $\sigma$-algebra** is
$$\mathcal{F}_1\otimes\mathcal{F}_2 := \sigma\big(\{A_1\times A_2: A_i\in\mathcal{F}_i\}\big),$$
generated by the *measurable rectangles*. A **product measure** $\mu_1\otimes\mu_2$ is a measure on it with $(\mu_1\otimes\mu_2)(A_1\times A_2) = \mu_1(A_1)\mu_2(A_2)$. Finite products iterate; write $\mu^{\otimes n}$ for the $n$-fold product of $\mu$ with itself.

**Theorem 2.42** (existence and uniqueness of the product measure). If $\mu_1,\mu_2$ are $\sigma$-finite, a product measure exists and is unique. Existence is not proved here (§2.12); uniqueness is immediate from Chapter 1: the rectangles form a $\pi$-system ($(A_1\times A_2)\cap(B_1\times B_2) = (A_1\cap B_1)\times(A_2\cap B_2)$) generating $\mathcal{F}_1\otimes\mathcal{F}_2$, so Theorem 1.20 applies.

Now the sentence that reframes supervised learning. A dataset of $n$ i.i.d. examples is **one draw from one measure**:
$$(X_1,\dots,X_n)\sim P^{\otimes n}\quad\text{on}\quad(\mathcal{X}^n,\mathcal{G}^{\otimes n}).$$

> **Every generalization bound is a statement about $P^{\otimes n}$.** When a paper says "with probability at least $1-\delta$ over the draw of the training set, $R(\hat h)\le\hat R_n(\hat h)+\varepsilon_n$," the probability is $P^{\otimes n}$ of a subset of $\mathcal{X}^n$ — a set of *datasets*. The randomness is not in the points individually and not in the algorithm; it is in which of the possible datasets you got. Chapter 4's concentration inequalities estimate $P^{\otimes n}(\text{bad dataset})$; Chapter 7's uniform bounds estimate $P^{\otimes n}(\text{some hypothesis in }\mathcal{H}\text{ looks better than it is})$ — where Chapter 1's union bound and its $\log|\mathcal{H}|$ come from.

### 2.10.2 Tonelli and Fubini: the two-step dance

**Theorem 2.43** (Tonelli; stated). Let $\mu_1,\mu_2$ be $\sigma$-finite and $f:\Omega_1\times\Omega_2\to[0,\infty]$ be $\mathcal{F}_1\otimes\mathcal{F}_2$-measurable. Then $\omega_1\mapsto\int f(\omega_1,\omega_2)\,d\mu_2(\omega_2)$ is $\mathcal{F}_1$-measurable (symmetrically in the other variable), and
$$\int_{\Omega_1\times\Omega_2}f\,d(\mu_1\otimes\mu_2) = \int_{\Omega_1}\Big(\int_{\Omega_2}f\,d\mu_2\Big)d\mu_1 = \int_{\Omega_2}\Big(\int_{\Omega_1}f\,d\mu_1\Big)d\mu_2,$$
all three possibly $+\infty$. **No integrability hypothesis beyond non-negativity.** (Proof: the standard ladder once more, with measurability of sections as the technical step; references in §2.12.)

**Theorem 2.44** (Fubini). Let $\mu_1,\mu_2$ be $\sigma$-finite and $f\in L^1(\mu_1\otimes\mu_2)$. Then for $\mu_1$-a.e. $\omega_1$ the section $f(\omega_1,\cdot)\in L^1(\mu_2)$, the map $\omega_1\mapsto\int f(\omega_1,\cdot)\,d\mu_2$ is in $L^1(\mu_1)$, and the three-way equality of Theorem 2.43 holds with all quantities finite.

*Proof from Tonelli.* Apply Tonelli to $|f|$: since $\int|f|\,d(\mu_1\otimes\mu_2)<\infty$, the iterated integrals of $|f|$ are finite, so $\int|f(\omega_1,\cdot)|\,d\mu_2<\infty$ for $\mu_1$-a.e. $\omega_1$ (a non-negative function with finite integral is finite a.e., by Markov). On that full-measure set split $f = f^+-f^-$, apply Tonelli to each (both dominated by $|f|$, hence finite), and subtract — legal because both terms are finite. $\square$

This is the **two-step dance**, the correct working procedure:
1. **Tonelli first**, on $|f|$, to establish $\int|f|<\infty$ — always legal, nothing to check.
2. **Then Fubini**, on $f$, to swap the order.

You cannot skip step 1, because swapping can genuinely change the answer. Standard counterexample: on $[0,1]^2$ with Lebesgue measure, $f(x,y) = \frac{x^2-y^2}{(x^2+y^2)^2}$ has iterated integrals $\pi/4$ and $-\pi/4$. Both exist; they disagree; and Tonelli on $|f|$ would have diagnosed it: $\int|f| = \infty$.

**Proposition 2.45** (independence *is* a product law). Let $X,Y$ be random vectors on a common probability space, valued in $\mathbb{R}^{d_1},\mathbb{R}^{d_2}$. Then
$$X\perp\!\!\!\perp Y\iff P_{(X,Y)} = P_X\otimes P_Y,$$
and in that case $\mathbb{E}[g(X)h(Y)] = \mathbb{E}[g(X)]\,\mathbb{E}[h(Y)]$ whenever $g(X),h(Y)\in L^1$ (or both non-negative).

*Proof.* ($\Rightarrow$) Independence (Definition 1.23) gives $P_{(X,Y)}(A\times B) = P_X(A)P_Y(B) = (P_X\otimes P_Y)(A\times B)$ on all rectangles; the rectangles are a generating $\pi$-system, so Theorem 1.20 upgrades this to equality of measures. ($\Leftarrow$) Read the chain backwards on rectangles. Factorization: Theorem 2.11 applied to the pair $(X,Y)$, then Tonelli/Fubini:
$$\mathbb{E}[g(X)h(Y)] = \int g(x)h(y)\,d(P_X\otimes P_Y) = \int\Big(\int g(x)h(y)\,dP_Y(y)\Big)dP_X(x) = \int g\,dP_X\int h\,dP_Y. \ \square$$

This discharges a Chapter 1 promise: independence, defined there through factorized event probabilities, *means* the joint law is a product measure. And the corollary you use daily: for i.i.d. data with density $p$ w.r.t. $\mu$, the density of $P^{\otimes n}$ w.r.t. $\mu^{\otimes n}$ is $\prod_i p(x_i)$ (check on rectangles, invoke uniqueness), so
$$\log p^{\otimes n}(x_1,\dots,x_n) = \sum_{i=1}^n\log p(x_i).$$
**This is why losses are sums.** Not convention, not convenience: the log-likelihood of a dataset is a sum because the joint density of independent variables is a product, and a logarithm turns products into sums. Everything downstream follows — a sum-structured loss has a sum-structured gradient, and a uniformly random minibatch average is an unbiased estimate of the full-data average, which is the justification of SGD.

**Proposition 2.46** (the tail formula). If $X\ge0$ is a random variable, then
$$\mathbb{E}[X] = \int_0^\infty\mathbb{P}(X>t)\,dt .$$

*Proof.* Let $E := \{(\omega,t)\in\Omega\times(0,\infty): t<X(\omega)\}$, in $\mathcal{F}\otimes\mathcal{B}$ because $(\omega,t)\mapsto X(\omega)-t$ is measurable on the product. For fixed $\omega$: $\int_0^\infty\mathbb{1}_E(\omega,t)\,dt = X(\omega)$. For fixed $t$: $\int_\Omega\mathbb{1}_E\,d\mathbb{P} = \mathbb{P}(X>t)$. Since $\mathbb{1}_E\ge0$, Tonelli equates the iterated integrals with no check:
$$\mathbb{E}[X] = \int_\Omega\int_0^\infty\mathbb{1}_E\,dt\,d\mathbb{P} = \int_0^\infty\int_\Omega\mathbb{1}_E\,d\mathbb{P}\,dt = \int_0^\infty\mathbb{P}(X>t)\,dt. \qquad\square$$

(Non-negativity of $X$ is used where the inner integral computes to $X(\omega)$: for signed $X$ the set $E$ no longer captures $X$, and the correct signed version is $\mathbb{E}[X] = \int_0^\infty\mathbb{P}(X>t)\,dt-\int_0^\infty\mathbb{P}(X<-t)\,dt$, requiring $X\in L^1$.) Flag this result: it is the standard device converting a *tail bound* into a *moment bound*. Chapter 4 uses it repeatedly — given $\mathbb{P}(X>t)\le2e^{-t^2/2\sigma^2}$, integrate to bound $\mathbb{E}[X]$; given $\mathbb{P}(|X|>t)\le Ce^{-ct}$, integrate $pt^{p-1}$ against it to bound $\mathbb{E}|X|^p$. It is also the honest version of §2.1's coin-sorting computation: there we summed $\mathbb{P}(X\ge k)$ over integers and got $1.5$; here the ad hoc argument became a theorem. Numerical check for $A = \max(0,Z)$: $\mathbb{P}(A>t) = 1-\Phi(t)$ for $t\ge0$, and quadrature of $\int_0^\infty(1-\Phi(t))\,dt$ returns $0.39894228 = 1/\sqrt{2\pi} = \mathbb{E}[A]$ to eight digits. Two entirely different computations, one theorem.

**Three interchanges you perform daily**, all Tonelli or Fubini in disguise.

*(a) Minibatch unbiasedness, completed.* Full-data loss $L(\theta) = \frac1n\sum_{i=1}^n\ell(\theta,x_i)$; minibatch $B\subseteq\{1,\dots,n\}$ of size $b$ uniform at random; $\hat L(\theta) = \frac1b\sum_{i\in B}\ell(\theta,x_i)$. Then $\mathbb{E}_B[\hat L(\theta)] = L(\theta)$: interchange the finite sum with the expectation (Proposition 2.13) and use $\mathbb{P}(i\in B) = b/n$. Combined with Theorem 2.21 to move $\nabla_\theta$ inside, this is the complete proof that **the minibatch gradient is unbiased** — the SGD hypothesis Chapter 1 flagged.

*(b) Two independent randomness sources.* Dropout masks, augmentations, and initializations are independent of the data, so the expected loss is an integral against $P_{\text{data}}\otimes P_{\text{alg}}$, and Tonelli says you may average over data first and algorithmic randomness second, or the reverse — provided the integrand is non-negative (a loss: yes) or absolutely integrable. This licenses "dropout approximates an ensemble average," and the reverse reading used in SGD-noise analyses. If your loss is unbounded and possibly non-integrable — an unclipped importance ratio times an unbounded reward — the licence lapses, and the two orders can genuinely differ.

*(c) Sums and integrals.* Corollary 2.20 was itself Tonelli in disguise: counting measure on $\mathbb{N}$ times $\mu$.

> **Self-check 2.10** *(answers in Appendix 2.A)*
>
> 1. In what precise sense is "with probability $1-\delta$ over the draw of the training set" a statement about a single measure?
> 2. State the two-step dance and explain why step 1 can never be skipped.
> 3. Prove in one line that $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ for independent integrable $X,Y$, naming each theorem used.
> 4. Where does the tail formula's proof use $X\ge0$, and what is the correct statement for signed $X$?

## 2.11 $L^p$ spaces, briefly

This section is a bridge, not a treatment: it installs the vocabulary Chapters 4 and 6 need, verifies that the objects deserve their names, and stops.

**Definition 2.47** ($L^p$ and its norm). For $1\le p<\infty$ on a measure space $(\Omega,\mathcal{F},\mu)$:
$$\|f\|_p := \Big(\int|f|^p\,d\mu\Big)^{1/p},\qquad L^p(\mu) := \{f\text{ measurable}: \|f\|_p<\infty\},$$
and $\|f\|_\infty := \inf\{M: |f|\le M\text{ a.e.}\}$, with $L^\infty(\mu)$ accordingly. By Proposition 2.14, $\|f\|_p = 0$ iff $f = 0$ a.e., so $\|\cdot\|_p$ separates points only after we agree that **the elements of $L^p$ are a.e.-equivalence classes**, the same identification §2.4 forced on densities. On a probability space, $\|X\|_p = (\mathbb{E}|X|^p)^{1/p}$; $L^1$ is the finite-mean variables, $L^2$ the finite-variance ones.

For $\|\cdot\|_p$ to be a norm we also need the triangle inequality, which is *not* obvious for $p>1$ and has a name. (The first draft used the word "norm" without accounting for this; the account is two theorems.)

**Theorem 2.48** (Hölder's inequality; stated, proof in §2.12). Let $1\le p,q\le\infty$ with $\frac1p+\frac1q = 1$. If $f\in L^p(\mu)$ and $g\in L^q(\mu)$, then $fg\in L^1(\mu)$ and
$$\int|fg|\,d\mu\le\|f\|_p\,\|g\|_q .$$

**Theorem 2.49** (Minkowski's inequality). For $1\le p\le\infty$ and $f,g\in L^p(\mu)$: $f+g\in L^p(\mu)$ and $\|f+g\|_p\le\|f\|_p+\|g\|_p$.

*Proof for $1<p<\infty$ from Hölder.* First, $|f+g|^p\le2^{p-1}(|f|^p+|g|^p)$ (convexity of $t^p$), so $f+g\in L^p$ and we may assume $\|f+g\|_p>0$. Write $|f+g|^p\le|f|\,|f+g|^{p-1}+|g|\,|f+g|^{p-1}$ and apply Hölder to each term with exponents $(p,q)$, $q = p/(p-1)$, noting $\||f+g|^{p-1}\|_q = \|f+g\|_p^{p-1}$:
$$\|f+g\|_p^p\le(\|f\|_p+\|g\|_p)\,\|f+g\|_p^{p-1};$$
divide by the finite, positive $\|f+g\|_p^{p-1}$. Cases $p = 1,\infty$ are the pointwise triangle inequality plus monotonicity. $\square$

**Proposition 2.50** (Cauchy–Schwarz). If $X,Y\in L^2(\mu)$ then $XY\in L^1(\mu)$ and
$$\Big|\int XY\,d\mu\Big|\le\|X\|_2\|Y\|_2,$$
with equality iff $X$ and $Y$ are proportional a.e. (one being $0$ a.e. counts).

*Proof.* Integrability: $|XY|\le\tfrac12(X^2+Y^2)$ pointwise. If $\|Y\|_2 = 0$ then $Y = 0$ a.e. and both sides vanish. Otherwise, for every $t\in\mathbb{R}$:
$$0\le\int(X-tY)^2\,d\mu = \|X\|_2^2-2t\int XY\,d\mu+t^2\|Y\|_2^2,$$
a quadratic in $t$, non-negative everywhere, hence with non-positive discriminant: $(\int XY)^2\le\|X\|_2^2\|Y\|_2^2$. Equality forces a real root $t_0$, i.e., $\int(X-t_0Y)^2 = 0$, i.e., $X = t_0Y$ a.e. by Proposition 2.14(2). $\square$

(Cauchy–Schwarz is Hölder at $p = q = 2$, but its two-line direct proof is worth owning.)

**Theorem 2.51** ($L^2$ is a Hilbert space). On a probability space, $\langle X,Y\rangle := \mathbb{E}[XY]$ is an inner product on $L^2(\mathbb{P})$ inducing $\|\cdot\|_2$, and $L^2(\mathbb{P})$ is **complete**: every Cauchy sequence converges in $\|\cdot\|_2$ to an element of $L^2$. (Bilinearity and symmetry: Proposition 2.13. Positive-definiteness modulo null sets: Proposition 2.14(2). Finiteness: Cauchy–Schwarz. Completeness — Riesz–Fischer — stated, §2.12.)

Completeness is exactly what fails for the Riemann-integrable functions, and exactly what Chapter 6 needs; here is the one-sentence why. **Chapter 6 defines $\mathbb{E}[X\mid\mathcal{G}]$ as the orthogonal projection of $X$ onto the closed subspace $L^2(\Omega,\mathcal{G},\mathbb{P})\subseteq L^2(\Omega,\mathcal{F},\mathbb{P})$** — the best $\mathcal{G}$-measurable approximation in mean square — and projections onto closed subspaces exist only in complete spaces. That single sentence simultaneously explains why conditional expectation is a "best guess given information," why it is linear, why it satisfies the tower property (a projection composed with a projection onto a smaller subspace), and why regression targets are conditional means: minimizing $\mathbb{E}[(X-g(Z))^2]$ over $\sigma(Z)$-measurable $g(Z)$ is a projection problem, and Doob–Dynkin (Lemma 1.22) identifies the $\sigma(Z)$-measurable variables as exactly the functions of $Z$. **Squared-error regression estimates a conditional expectation because of a projection theorem in a Hilbert space.**

**Proposition 2.52** (Jensen's inequality). Let $X\in L^1(\mathbb{P})$ and $\psi:\mathbb{R}\to\mathbb{R}$ be convex. Then
$$\psi(\mathbb{E}[X])\ \le\ \mathbb{E}[\psi(X)],$$
the right side well defined in $(-\infty,+\infty]$. If $\psi$ is concave the inequality reverses. If $\psi$ is strictly convex, equality holds iff $X$ is a.s. constant. The same statement holds for a random vector $X\in L^1(\mathbb{P};\mathbb{R}^d)$ and convex $\psi:\mathbb{R}^d\to\mathbb{R}$, with the supporting line replaced by a supporting hyperplane.

*Proof.* Let $m := \mathbb{E}[X]$, finite. Convexity supplies a **supporting line** at $m$: a subgradient $c$ with $\psi(x)\ge\psi(m)+c(x-m)$ for all $x$ (in $\mathbb{R}^d$: a subgradient vector $c$ with $\psi(x)\ge\psi(m)+\langle c,x-m\rangle$; such $c$ exists for any convex, hence continuous, $\psi$ on $\mathbb{R}^d$). The right-hand side is integrable (affine in $X$), so $\psi(X)^-$ is integrable and $\mathbb{E}[\psi(X)]$ is well defined, possibly $+\infty$. Monotonicity and linearity give
$$\mathbb{E}[\psi(X)]\ \ge\ \psi(m)+c\,(\mathbb{E}[X]-m) = \psi(m).$$
Equality case: $\mathbb{E}[\psi(X)-\psi(m)-c(X-m)] = 0$ with a non-negative integrand forces $\psi(X) = \psi(m)+c(X-m)$ a.s. (Proposition 2.14(2)); strict convexity confines that to $X = m$ a.s. $\square$

Notice the shape: **one tangent line, then monotonicity** — the same move as Gibbs' inequality (Proposition 2.33), which used $\log t\le t-1$, the supporting line of $\log$ at $1$. Jensen is the workhorse of variational methods: it turns an intractable $\log\int$ into a tractable $\int\log$, which is precisely the step producing the ELBO (Exercise C2) and the dequantization bound (Exercise C3). Two instances to keep loaded, both concave $\log$: $\log\mathbb{E}[W]\ge\mathbb{E}[\log W]$ (a log-of-average dominates an average-of-logs), and $\mathbb{E}[X^2]\ge(\mathbb{E}X)^2$ (variance is non-negative; $\psi(x) = x^2$).

**Proposition 2.53** (norms nest on probability spaces). If $\mathbb{P}$ is a probability measure and $1\le p\le q$, then $\|X\|_p\le\|X\|_q$; in particular $\mathbb{E}|X|\le\sqrt{\mathbb{E}[X^2]}$, so $L^2(\mathbb{P})\subseteq L^1(\mathbb{P})$.

*Proof.* The most-used case $p = 1$, $q = 2$: Cauchy–Schwarz with $Y\equiv1$ gives $\mathbb{E}|X|\le\|X\|_2\|1\|_2 = \|X\|_2$. General case: Jensen with the convex $\psi(t) = t^{q/p}$ applied to $|X|^p$: $(\mathbb{E}|X|^p)^{q/p}\le\mathbb{E}|X|^q$; take $q$-th roots. (Hölder gives it too.) The hypothesis $\mathbb{P}(\Omega) = 1$ is essential: on $(\mathbb{R},\lambda)$ the inclusion fails both ways — $1/(1+|x|)\in L^2\setminus L^1$, while $|x|^{-1/2}\mathbb{1}_{(0,1)}\in L^1\setminus L^2$. $\square$

**Worked Example 2.18** (Cauchy–Schwarz as the correlation bound). For $X,Y\in L^2$ with means $\mu_X,\mu_Y$, apply Proposition 2.50 to the centred variables:
$$|\mathrm{Cov}(X,Y)|\le\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}\implies|\rho(X,Y)|\le1,$$
with equality iff $X-\mu_X = t(Y-\mu_Y)$ a.s.: **correlation $\pm1$ means exact affine dependence almost surely**, not "very strong dependence." The same inequality on the counting-measure space $\{1,\dots,d\}$ gives $|\langle u,v\rangle|\le\|u\|_2\|v\|_2$: cosine similarity between embeddings lies in $[-1,1]$ for exactly the reason correlation does. Two facts you may have learned separately are one theorem at two measures — a small, characteristic dividend of having a single integral.

Chapter 4 needs one more idea from this circle: sub-Gaussian and sub-exponential norms, defined by requiring $\|X\|_p$ to grow no faster than $\sqrt p$ or $p$ (equivalently, $\mathbb{E}[e^{\lambda X}]$ finite near $0$). Those are Orlicz norms; they live in this $L^p$ framework and are what turn Markov's inequality into exponential tail bounds. We stop here.

> **Self-check 2.11** *(answers in Appendix 2.A)*
>
> 1. Why must the elements of $L^p$ be equivalence classes rather than functions?
> 2. Which named inequality makes $\|\cdot\|_p$ a norm, and from which other inequality did we derive it?
> 3. Why does $L^2\subseteq L^1$ hold on a probability space but fail on $(\mathbb{R},\lambda)$?
> 4. Which property of $L^2$ does Chapter 6 need to define conditional expectation, and why?

## 2.12 Results used but not proved

Complete list. Everything else in the chapter is proved above or cited by number from Chapter 1.

| Result | Used in | Where to find a proof |
|---|---|---|
| Riemann–Lebesgue relationship (Theorem 2.1: Lebesgue's criterion; agreement of the integrals) | §2.1.3 | Folland, *Real Analysis*, Thm 2.28 and §2.3; Tao, *Measure Theory*, §1.6 |
| Radon–Nikodym theorem (Theorem 2.26) | §2.7 throughout; §§2.8–2.9 depend on it | Folland §3.2; Durrett, App. A; Billingsley §32 |
| Lebesgue decomposition (Theorem 2.29) | §2.7.2 | Folland §3.2; Billingsley §32 |
| Vitali convergence theorem (Theorem 2.19; proved in Ch. 5) | §2.5 (the honest version of "two escapes") | Folland §6.1 (uniform integrability); Williams, Ch. 13 |
| Existence of product measures (Theorem 2.42; uniqueness proved via Thm 1.20) | §2.10; Definition 2.37 (couplings) | Folland §2.5; Durrett §1.7 |
| Tonelli's theorem (Theorem 2.43) | §2.10; Proposition 2.46; the daily interchanges | Folland §2.5 Thm 2.37; Durrett Thm 1.7.2 |
| Completeness of $L^p$ / Riesz–Fischer (part of Theorem 2.51) | §2.11; needed by Ch. 6 | Folland §6.1; Rudin Thm 3.11 |
| Hölder's inequality (Theorem 2.48) | §2.11 (Minkowski's proof; alternative route to Prop. 2.53) | Folland §6.1 (from Young's inequality, three lines) |
| Pinsker's inequality, $\mathrm{TV}\le\sqrt{D_{\mathrm{KL}}/2}$ | §2.8.1 (mentioned); used in Chs. 4–5 | Tsybakov, *Nonparametric Estimation*, Lemma 2.5; Cover–Thomas §11.6 |
| Kantorovich duality for $W_1$ | §2.8.5 (the WGAN caveat) | Villani, *Optimal Transport*, Ch. 5 |
| $W_1$ empirical convergence rate $n^{-1/d}$ | §2.8.5, caveat 1 | Weed & Bach, *Bernoulli* 2019 |
| Scaling of Lebesgue measure, $\lambda_d(cA) = c^d\lambda_d(A)$ | Proposition 2.30 | Folland §2.6; provable from Thm 1.20 on boxes |
| $\lambda(\mathbb{Q}) = 0$; $\lambda_2(S^1) = 0$ | §2.1, §2.7.1 | Countable additivity plus a covering argument; both sketched in text |
| Existence of Lebesgue measure on $\mathbb{R}^d$ | Everywhere $\lambda$ appears | Ch. 1 (on faith there); Folland §1.5 |
| Mean value theorem | Theorem 2.21 | Any calculus text |
| Gaussian moments $\mathbb{E}\varepsilon^{2k}$; $\int x^4e^{-ax^2}dx = \tfrac34\sqrt\pi\,a^{-5/2}$ | Worked Examples 2.12, 2.17 | Differentiate $\int e^{-ax^2}dx = \sqrt{\pi/a}$ in $a$ |
| PSIS $\hat k$ diagnostic | §2.9(d) | Vehtari, Simpson, Gelman, Yao, Gabry, *JMLR* 2024 |

## 2.13 Chapter summary

1. **One integral.** $\int f\,d\mu$ is built in three steps — simple functions by fiat (Definition 2.3), non-negative functions by supremum over simple minorants (Definition 2.5), signed functions via $f = f^+-f^-$ (Definition 2.9). Sums, Riemann integrals, and mixtures are instances, differing only in the measure; Theorem 2.1 states the exact relationship with the Riemann integral.
2. **Lebesgue slices the range.** "How much of the domain has $f$ above $t$?" is a question about measure — which is why measurability is the exact hypothesis, and why $\mathbb{1}_{\mathbb{Q}}$ integrates to $0$ while its Riemann integral does not exist.
3. **The integral can be $+\infty$, or undefined — and the two differ.** With $\mathbb{E}[X] = +\infty$ (Pareto), $\bar X_n\to+\infty$ a.s.: the LLN holds in the extended sense. With $\mathbb{E}[X]$ undefined (Cauchy), $\bar X_n$ converges to nothing — it is Cauchy for every $n$. "By symmetry the mean is zero" is a false argument, and heavy-tailed gradient noise makes all of this a live issue for SGD theory.
4. **$\mathbb{E}[X] := \int_\Omega X\,d\mathbb{P}$, and change of variables lets you forget $\Omega$.** $\mathbb{E}[g(X)] = \int g\,dP_X$ (Theorem 2.11), proved by the standard ladder — indicators, simple, non-negative, general — which recurs three more times. A minibatch average is an *exact* integral against $\hat P_n$.
5. **Integration cannot see null sets.** Densities are equivalence classes; "the density at $x$" is well defined only under a continuity convention — the convention every normalizing flow silently relies on (§2.4).
6. **Limits and integrals commute unless mass escapes** — sideways ($\mathbb{1}_{[n,n+1]}$) or upward ($n\mathbb{1}_{(0,1/n]}$). MCT forbids the first, DCT's integrable ceiling the second, and Vitali (Theorem 2.19) turns the slogan into an equivalence: tightness + uniform integrability. Fatou survives both escapes, strictly. VAE variance collapse is the upward escape running inside a training loop.
7. **Failure 3 closed.** Differentiation under the expectation is legal given a $\theta$-uniform integrable bound on $\partial_\theta h$ (Theorem 2.21: DCT plus the mean value theorem). "Standard regularity conditions" means: exhibit that bound.
8. **Reparameterization moves $\theta$ out of the measure into the integrand**; the licence is change of variables. REINFORCE keeps $\theta$ in the measure and pays: in the model problem, variances $4$ versus $30$, ratio $7.5$, simulation-confirmed at $7.3$–$7.7$.
9. **The score has mean zero** (Proposition 2.24), by differentiating $\int p_\theta\,d\mu = 1$. This underwrites REINFORCE baselines, Fisher information as a covariance, and Chapter 8's Cramér–Rao regularity — which is literally the hypothesis set of Theorem 2.21. $\theta$-dependent supports break all of it, sign included (Exercise A8).
10. **Failures 1 and 2 closed by Radon–Nikodym.** A density is $d\nu/d\mu$ — a local exchange rate, not a probability. It may exceed $1$ ($\mathcal{N}(0,0.01^2)$ peaks at $39.894$); it shifts by $d\log c$ under rescaling (whence $\log255 = 5.5413$ nats and the $+8$ bits/dim convention); and it exists for the ReLU output against $\delta_0+\lambda$: $\tfrac12$ at the atom, $\varphi(a)$ on $(0,\infty)$.
11. **pdf, pmf, and mixed densities are one object** at three base measures ($\lambda$, $\#$, $\delta_0+\lambda$). "The" density does not exist; a density with respect to a stated reference does.
12. **Jensen (Proposition 2.52) is the engine of variational bounds**: one supporting line plus monotonicity, and the concave-$\log$ case converts $\log\int$ into $\int\log$ — the ELBO and the dequantization bound are each that single step. Relatedly, **$D_{\mathrm{KL}}$ is always defined, possibly $+\infty$**, and the $+\infty$ is what the limit genuinely does (Worked Example 2.16: $\alpha^2/2h^2\to\infty$). Infinite cross-entropy on a confidently wrong prediction is real mathematics; `log_softmax` and clamping are engineering responses, and clamping changes the objective.
13. **Forward KL covers mass; reverse KL seeks modes.** Bimodal target: $(m,s) = (0.0000,3.1623)$ forward — exact moment matching — versus $(2.9843,1.0234)$ reverse, which discards half the target. VI's reverse-KL choice is why VAE posteriors over-concentrate and why posterior collapse is a stationary point of the written loss.
14. **Failure 4 closed, in general.** An $f$-divergence sees only the ratio $dP/dQ$; between mutually singular laws every $f$-divergence equals the constant $f(0)+f'(\infty)$ (Proposition 2.36) — JS pins to $\log2$, TV to $1$, KL to $+\infty$ — so the gradient is exactly zero regardless of distance. $W_1 = |\alpha|$ for parallel segments because it is built from the metric — at the price of $n^{-1/d}$ estimation, an approximately-Lipschitz critic in WGANs, and dependence on the chosen ground metric.
15. **Importance sampling needs $P\ll Q$, and unbiasedness is not usefulness.** For $p = \varphi$, $f = x^2$, $q_\sigma = \mathcal{N}(0,\sigma^2)$: variance finite iff $\sigma>1/\sqrt2$; in the infinite-variance regime a single run still reports a healthy SE, and nominal 95% intervals covered $43\%$ at $\sigma = 0.45$. Use ESS, max-weight share, PSIS $\hat k$ — noting that $\hat k = 0.591$ at $\sigma = 0.6$ passes the popular $0.7$ threshold while the variance is provably infinite.
16. **A dataset is one draw from $P^{\otimes n}$**; independence *means* the joint law is a product measure; hence log-likelihoods are sums and minibatch gradients unbiased. Tonelli swaps freely for non-negative integrands; Fubini needs $\int|f|<\infty$, checked *by* Tonelli — the two-step dance. The tail formula is Tonelli, and it is how Chapter 4 turns tail bounds into moment bounds.

### Callbacks

Chapter 1 left four IOUs; this chapter paid all four (score card below). Three further promises are also discharged. Chapter 1 defined independence through factorizing probabilities and promised a cleaner formulation: Proposition 2.45 delivers it — independence *is* $P_{(X,Y)} = P_X\otimes P_Y$. Chapter 1 argued that countable additivity is the right axiom because it is equivalent to continuity from below, and promised the payoff later: it came in the proof of MCT (Theorem 2.8), where continuity from below is the only measure-theoretic input — hence in every convergence and interchange theorem downstream. Chapter 1 insisted a random variable is a deterministic function with all randomness in $\omega$ and a distribution a pushforward: Theorem 2.11 makes that viewpoint computationally usable, and Corollary 2.22 shows the reparameterization trick is that theorem read right-to-left.

**The Four Failures — final score card.**

| # | Failure (Ch. 1 §1.1) | Status at end of Ch. 1 | Closed by | How |
|---|---|---|---|---|
| 1 | Densities exceed $1$; $\mathbb{P}(X = x) = 0$ for continuous $X$ | OPEN | **§2.7.4** (Thm 2.26, Prop 2.30) | A density is $dP/d\lambda$: probability per unit reference measure, a rate, unbounded. $\mathcal{N}(0,0.01^2)$ peaks at $39.894$. Rescaling by $c$ shifts $\log p$ by $-d\log c$ — hence bits/dim conventions. |
| 2 | $A = \max(0,Z)$ is neither discrete nor continuous | HALF-RESOLVED (CDF only) | **§2.7.3** (WE 2.13, Thm 2.29) | Choose base measure $\delta_0+\lambda$: then $dP_A/d\mu = \tfrac12$ at the atom, $\varphi(a)$ on $(0,\infty)$ — one density, from which $\mathbb{E}[A] = 1/\sqrt{2\pi}$ is recomputed. Lebesgue decomposition says this always works. |
| 3 | Swapping $\nabla$ and $\mathbb{E}$ | OPEN | **§2.6** (Thm 2.21) | DCT plus the mean value theorem, under a $\theta$-uniform integrable bound on $\partial_\theta h$. Applications: reparameterization (Cor 2.22), REINFORCE (Prop 2.23, needs $\theta$-free support), zero-mean score (Prop 2.24). |
| 4 | GAN loss constant (JS $= \log2$ on disjoint supports) | DIAGNOSED (support is $\lambda$-null) | **§2.8.4** (Prop 2.36, Prop 2.38) | Every $f$-divergence equals $f(0)+f'(\infty)$ between mutually singular laws, so $\nabla_\theta D_f\equiv0$: $f$-divergences see only density ratios, and absolute continuity is yes/no. $W_1 = |\alpha|$ because it uses the metric — with its own costs (§2.8.5). |

### Before you continue

Answerable without looking back:

1. Why is measurability exactly the hypothesis the Lebesgue construction needs, and where is it used?
2. Give a function whose Lebesgue integral is $0$ and whose Riemann integral does not exist.
3. What is the difference between an integral equal to $+\infty$ and one that is undefined — including what the sample mean does in each case?
4. State the change-of-variables theorem and name the four rungs of the ladder proving it.
5. Which hypothesis of DCT fails for $f_n = n\mathbb{1}_{(0,1/n]}$, what is the smallest candidate dominating function, and which half of Vitali's condition is violated?
6. What does the reparameterization trick move, from where to where, and under what licence?
7. Why is there no such thing as "the" density of a distribution? What must be specified?
8. Two mutually singular laws are $10^{-6}$ apart; two others are $10^6$ apart. What is $D_f$ in each case for any $f$-divergence, and why does that make GAN training fail?
9. Your importance sampler reports a small standard error. Name two diagnostics that could still reveal it is broken, and the theoretical fact explaining why the standard error was uninformative.
10. Which inequality makes $\|\cdot\|_p$ a norm, and which inequality is the engine of every variational bound?

### Looking ahead to Chapter 3

Chapter 2 gave you one integral and one notion of density; Chapter 3 asks what happens to both when you *transform* a random variable. The centrepiece is the change-of-variables formula with Jacobians — pushing a law through a smooth injective map multiplies the density by $|\det J|^{-1}$ — with careful treatment of non-injective maps (the density becomes a sum over preimage branches; $Y = X^2$ stops being a trap). From there: inverse-transform sampling, which turns the CDF into a sampler; the exponential family and sufficiency, which retroactively explains why forward-KL fitting reduced to moment matching in Worked Example 2.15; normalizing flows, which are the Jacobian formula composed many times and trained by maximum likelihood, their log-determinant terms being Proposition 2.30's $-d\log c$ in general form; and a distribution zoo organized by *when would I reach for this?*

# Exercises

Difficulty: ★ routine consolidation, ★★ requires a real idea, ★★★ paper-level. Solutions follow the exercise statements.

## Part A — Theory

**A0 (★).** *Warm-up on base measures.* Let $\mu := 2\delta_{-1}+3\lambda|_{[0,1]}$ on $(\mathbb{R},\mathcal{B})$ and $f(x) = x^2$. (a) Compute $\int f\,d\mu$ directly from Definitions 2.3–2.5 and Proposition 2.28(4)'s logic (split the measure). (b) Is $\mu$ a probability measure? If not, normalize it and recompute.

**A1 (★).** (a) Let $\mu = \delta_0+\lambda$ on $(\mathbb{R},\mathcal{B})$ and $f(x) = e^{-|x|}$. Compute $\int f\,d\mu$. (b) Prove that for measures $\mu_1,\mu_2$ and constants $c_1,c_2\ge0$: $\int g\,d(c_1\mu_1+c_2\mu_2) = c_1\int g\,d\mu_1+c_2\int g\,d\mu_2$ for all measurable $g\ge0$, stating where the ladder is used. (c) Let $\#$ be counting measure on $\mathbb{Z}$ and $g(k) = 2^{-|k|}$. Compute $\int g\,d\#$ and $\int g\,d(\#|_{\{0,1,2,\dots\}})$; state which is the expectation of a random variable and give the pmf.

**A2 (★★).** (a) Deduce Chebyshev's inequality $\mathbb{P}(|X-\mathbb{E}X|\ge t)\le\mathrm{Var}(X)/t^2$ from Markov (Proposition 2.15). (b) Characterize exactly when Markov is an equality for a given $t>0$, and exhibit a two-point distribution achieving it. (c) A validation loss has mean $0.04$ and lies in $[0,1]$. Compare the Markov bound on $\mathbb{P}(\text{loss}\ge0.5)$ with the best bound you can get knowing additionally $\mathrm{Var} = 0.01$.

**A3 (★★).** Let $X\ge0$ with $\mathbb{E}[X]<\infty$. Prove: (a) $\mathbb{P}(X = \infty) = 0$; (b) if $\mathbb{E}[X] = 0$ then $X = 0$ a.s.; (c) $\mathbb{E}[X\mathbb{1}_{\{X>n\}}]\to0$. Which theorem does (c) need, and why is it false without $\mathbb{E}[X]<\infty$? Relate (c) to the definition of uniform integrability (Theorem 2.19).

**A4 (★★).** (a) Give a sequence $f_n\ge0$ with $\int\liminf f_n\,d\lambda<\liminf\int f_n\,d\lambda$ strictly, and one where equality holds without monotonicity or domination. (b) Prove the *reverse Fatou lemma*: if $f_n\le g$ for all $n$ with $g\in L^1$, then $\limsup_n\int f_n\le\int\limsup_n f_n$. (c) Deduce DCT from (b) and Fatou in two lines.

**A5 (★★).** MCT is stated for increasing sequences. (a) Show by example that a decreasing sequence $f_n\downarrow f$ of non-negative measurable functions can have $\int f_n\not\to\int f$. (b) Prove that if additionally $f_1\in L^1(\mu)$, then $\int f_n\to\int f$. (c) Where is finiteness used, and which Chapter 1 result is the set-measure analogue?

**A6 (★★★).** (*Scheffé's lemma.*) Let $p_n,p$ be probability densities w.r.t. $\mu$ with $p_n\to p$ $\mu$-a.e. Prove $\int|p_n-p|\,d\mu\to0$, i.e., $\mathrm{TV}(P_n,P)\to0$ (Definition 2.31). (Hint: bound $(p-p_n)^+$ by $p$; use DCT and $\int(p_n-p)\,d\mu = 0$.) Explain in one sentence why this says a.e. convergence of densities implies total-variation convergence, and why the converse fails.

**A7 (★★).** Using Proposition 2.28, derive
$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12 .$$
Then (a) verify it vanishes iff the parameters agree; (b) compute the limit as $\sigma_1\to0$ with $\mu_1\ne\mu_2$ fixed and interpret; (c) identify which term is responsible for the "KL vanishing" pathology in VAEs when $q$ matches the prior.

**A8 (★★★).** Let $p_\theta(x) = \frac1\theta\mathbb{1}_{[0,\theta]}(x)$, $\theta>0$, and $f(x) = x$. (a) Compute $\mathbb{E}_\theta[f]$ and $\frac{d}{d\theta}\mathbb{E}_\theta[f]$ exactly. (b) Compute $\mathbb{E}_\theta[f\,\partial_\theta\log p_\theta]$ exactly. (c) They differ; identify the missing term as a boundary contribution and compute it. (d) Which hypothesis of Proposition 2.23 fails, and what does this imply about applying Chapter 8's Cramér–Rao bound to this family?

**A9 (★★).** *Well-definedness of $f$-divergences.* Show that Definition 2.34 does not depend on the dominating measure: if $\mu$ and $\mu'$ both dominate $P$ and $Q$, the two computations of $D_f(P\|Q)$ agree. (Hint: both are dominated by $\mu'' = \mu+\mu'$; use the chain rule, Proposition 2.28(2), and check the $\{q = 0\}$ bookkeeping.) Then verify from the definition that $f(t) = \tfrac12|t-1|$ yields $\mathrm{TV}$.

## Part B — Computational labs

Report every number you produce, with the seed and sample size. Do not quote a number you did not compute. (Reference environment for the solutions: NumPy 2.4.4 / SciPy 1.17.1.)

**B1 (★).** *Change of variables, checked twice.* (a) For $X\sim\mathcal{N}(0,1)$, compute $\mathbb{E}[X^2]$ by (i) quadrature of $\int x^2\varphi\,dx$ and (ii) Monte Carlo with $n = 10^4$ and $10^6$; report estimates and standard errors. (b) Repeat for $A = \max(0,Z)$, computing $\mathbb{E}[A]$ and $\mathbb{E}[A^2]$; for the quadrature you must integrate against $\delta_0+\lambda$, handling the atom explicitly; show that omitting the atom and renormalizing $\varphi$ on $(0,\infty)$ gives exactly twice the correct $\mathbb{E}[A]$. (c) Explain why the atom does not affect $\mathbb{E}[A]$ but does affect $\mathbb{E}[e^{-A}]$, and compute both.

**B2 (★★★).** *Importance sampling and the silent variance catastrophe.* Target $p = \mathcal{N}(0,1)$, $f(x) = x^2$, truth $1$; proposals $q_\sigma = \mathcal{N}(0,\sigma^2)$.
(a) Derive analytically that $\mathrm{Var}_{q_\sigma}(wf)<\infty$ iff $\sigma>1/\sqrt2$, by exhibiting the exponential factor of $p^2/q_\sigma$; obtain the closed-form variance and verify numerically at $\sigma = 1.5$.
(b) Run $200$ trials of $n = 10^5$ for $\sigma\in\{1.5,1.0,0.9,0.8,0.72,0.6,0.45\}$; tabulate the mean of the estimates and the across-trial standard deviation; compare with (a).
(c) Explain why $\sigma = 1.5$ beats sampling from the target itself, derive the optimal proposal, and verify its variance numerically.
(d) Show that in the provably-infinite-variance regime a single run still reports a healthy-looking standard error: report the median reported SE with its 10th/90th percentiles across trials, and the empirical coverage of the nominal 95% interval. Then compute ESS, maximum weight share, and PSIS $\hat k$ per $\sigma$, and say which diagnostics catch the failure. Relate to PPO clipping and V-trace truncation.

**B3 (★★).** *Gradient estimator variance.* For $\frac{d}{d\mu}\mathbb{E}_{\mathcal{N}(\mu,1)}[x^2]$ at $\mu = 1$ (truth $2$): implement the reparameterized and score-function estimators; for $B\in\{1,10,100\}$ and $5000$ replications, report mean and variance of each. Derive per-sample variances analytically and compare. Explain the implied compute multiplier, and why discrete latents cannot use the reparameterized estimator (name two standard workarounds).

**B4 (★★).** *Failure 4, numerically.* $P$ uniform on $\{(t,0): t\in[0,1]\}$, $Q_\alpha$ uniform on $\{(t,\alpha)\}$. (a) Explain why $D_{\mathrm{KL}}(P\|Q_\alpha) = +\infty$ and $D_{\mathrm{JS}} = \log2$ for all $\alpha\ne0$ without computing anything. (b) Smooth both laws in the $y$-coordinate by $\mathcal{N}(0,h^2)$ and compute $D_{\mathrm{KL}}$, $D_{\mathrm{JS}}$, $W_1$ for $h\in\{0.5,0.1,0.01\}$, $\alpha\in\{0.01,0.05,0.1,0.5,1,2\}$. (c) Tabulate all three against $\alpha$ and state the gradient consequence for a generator that must drive $\alpha\to0$.

## Part C — Paper-style derivations

**C1 (★★★).** *Baselines for the score-function estimator.*
(a) Prove that for any constant $b\in\mathbb{R}$, $\hat g_b := (f(X)-b)\nabla_\theta\log p_\theta(X)$ is unbiased for $\nabla_\theta\mathbb{E}_{p_\theta}[f]$, stating the hypotheses.
(b) For $\theta\in\mathbb{R}^m$, taking as criterion the trace of the covariance of $\hat g_b$ (equivalently $\mathbb{E}\|\hat g_b-\nabla_\theta\mathbb{E}[f]\|^2$), show the minimizing constant is
$$b^\ast = \frac{\mathbb{E}[f(X)\,\|\nabla_\theta\log p_\theta(X)\|^2]}{\mathbb{E}[\|\nabla_\theta\log p_\theta(X)\|^2]},$$
**not** $\mathbb{E}[f(X)]$.
(c) Compute $b^\ast$ and $\mathbb{E}[f]$ for the model problem of Worked Example 2.12, and the resulting variances; verify by simulation.
(d) Actor–critic methods use a learned $V(s)\approx\mathbb{E}[\text{return}\mid s]$ — the "wrong" baseline by (b). Give two reasons this is nevertheless the right engineering choice.

**C2 (★★★).** *The ELBO identity.* Let $p_\theta(x,z)$ be a joint density w.r.t. $\mu_X\otimes\mu_Z$, $q(z)$ a density w.r.t. $\mu_Z$, and
$$\mathcal{L}(q,\theta;x) := \mathbb{E}_{z\sim q}[\log p_\theta(x,z)-\log q(z)].$$
(a) Prove $\log p_\theta(x) = \mathcal{L}(q,\theta;x)+D_{\mathrm{KL}}(q\,\|\,p_\theta(\cdot|x))$.
(b) Conclude $\mathcal{L}\le\log p_\theta(x)$, with equality iff $q = p_\theta(\cdot|x)$ $\mu_Z$-a.e.
(c) State where absolute continuity is required, and what happens to each term if $q$ puts mass where the posterior has none, and vice versa.
(d) Explain why the bound is tight in principle but loose in practice for Gaussian $q$ and a multimodal posterior, connecting to Worked Example 2.15.

**C3 (★★★).** *Bits per dimension, dequantization, and a real dispute.* Alice reports her image model achieves $+3.2$ nats/dim on $[0,1]$-scaled data. Bob reports $-2.3$ nats/dim for what he believes is the same architecture on $[0,255]$-scaled data. Carol declares both results buggy, "since log-likelihoods must be negative." (a) Reconcile Alice and Bob using Proposition 2.30, and convert both to bits/dim under the 256-level convention. (b) Adjudicate Carol. (c) Explain in four steps why discrete pixel data has no Lebesgue density, why a continuous model fitted to it can drive the likelihood to $+\infty$, why uniform dequantization yields a valid lower bound on a genuine discrete log-likelihood (via Jensen), and how variational dequantization tightens it.

# Solutions

## Part A

**A0.** *(a)* Split: $\int f\,d\mu = 2\int f\,d\delta_{-1}+3\int_{[0,1]}f\,d\lambda = 2f(-1)+3\int_0^1x^2\,dx = 2\cdot1+3\cdot\tfrac13 = 3$. *(b)* $\mu(\mathbb{R}) = 2+3 = 5\ne1$, so no. Normalizing, $\bar\mu = \mu/5$ and $\int f\,d\bar\mu = 3/5 = 0.6$ — the mean of $X^2$ for a random variable that is $-1$ with probability $2/5$ and uniform on $[0,1]$ with probability $3/5$.

**A1.** *(a)* $\int f\,d\mu = f(0)+\int_{\mathbb{R}}e^{-|x|}\,dx = 1+2 = 3$. *(b)* Indicators: $\int\mathbb{1}_A\,d(c_1\mu_1+c_2\mu_2) = c_1\mu_1(A)+c_2\mu_2(A)$ — the definition of the sum measure. Simple: linearity of finite sums. Non-negative: $s_n\uparrow g$ (Lemma 2.6) and MCT applied on all three spaces simultaneously — this is the rung where the ladder is essential, since the supremum in Definition 2.5 does not obviously split. *(c)* $\int g\,d\# = \sum_{k\in\mathbb{Z}}2^{-|k|} = 1+2\sum_{k\ge1}2^{-k} = 3$; restricted to $\mathbb{N}$: $\sum_{k\ge0}2^{-k} = 2$. Neither total is $1$, so $g$ is not itself a pmf; but $p := g/3$ on $\mathbb{Z}$ (or $g/2$ on $\mathbb{N}$) is one, and then any expectation under it, $\mathbb{E}[h(K)] = \sum_k h(k)p(k) = \int hp\,d\#$, exhibits the pmf as a density with respect to counting measure (Definition 2.27). The two computations above are the normalizing constants of these two pmfs.

**A2.** *(a)* Apply Markov to the non-negative variable $(X-\mathbb{E}X)^2$ at level $t^2$: $\mathbb{P}(|X-\mathbb{E}X|\ge t) = \mathbb{P}((X-\mathbb{E}X)^2\ge t^2)\le\mathrm{Var}(X)/t^2$. *(b)* Tracing the proof, equality requires $t\mathbb{1}_{\{f\ge t\}} = f$ a.e.: $f\in\{0,t\}$ a.e. Two-point achiever: $X = t$ with probability $q$, $X = 0$ with probability $1-q$; then $\mathbb{E}X = qt$ and $\mathbb{P}(X\ge t) = q = \mathbb{E}X/t$. *(c)* Markov: $\mathbb{P}(\ell\ge0.5)\le0.04/0.5 = 0.08$. Chebyshev with $\mathrm{Var} = 0.01$: $\mathbb{P}(\ell\ge0.5)\le\mathbb{P}(|\ell-0.04|\ge0.46)\le0.01/0.46^2 = 0.0473$ — better, and Chapter 4's one-sided refinement (Cantelli) gives $0.01/(0.01+0.46^2) = 0.0451$.

**A3.** *(a)* For every $n$, $n\,\mathbb{P}(X = \infty)\le n\,\mathbb{P}(X\ge n)\le\mathbb{E}[X]$ by Markov, so $\mathbb{P}(X = \infty)\le\mathbb{E}[X]/n\to0$. *(b)* Proposition 2.14(2). *(c)* $X\mathbb{1}_{\{X>n\}}\to X\mathbb{1}_{\{X = \infty\}} = 0$ a.s. (by (a)), dominated by $X\in L^1$: DCT gives the limit $0$. Without $\mathbb{E}[X]<\infty$ it fails — the Pareto of Worked Example 2.8 has $\mathbb{E}[X\mathbb{1}_{\{X>n\}}] = \int_n^\infty x^{-1}dx = \infty$ for every $n$. Relation to uniform integrability: the UI condition of Theorem 2.19 is exactly this statement made *uniform over a family*; a single $L^1$ variable is automatically "uniformly" integrable, which is what (c) proves.

**A4.** *(a)* Strict: $f_n = \mathbb{1}_{[n,n+1]}$ on $(\mathbb{R},\lambda)$, giving $0<1$. (An instructive near-miss: alternating $f_n = \mathbb{1}_{[0,1]}$ and $\tfrac12\mathbb{1}_{[0,2]}$ gives $\liminf f_n = \tfrac12\mathbb{1}_{[0,1]}$ with integral $\tfrac12$ against $\liminf\int f_n = 1$ — still strict, showing oscillation alone can open a gap even with no escaping mass.) Equality without invoking monotonicity or an integrable dominator on all of $(\mathbb{R},\lambda)$: $f_n = (1+\tfrac1n)\mathbb{1}_{[0,1]}$ — not monotone (it decreases), and we cite no domination; $\liminf f_n = \mathbb{1}_{[0,1]}$ and both sides equal $1$. *(b)* Apply Fatou to $g-f_n\ge0$: $\int(g-\limsup f_n) = \int\liminf(g-f_n)\le\liminf\int(g-f_n) = \int g-\limsup\int f_n$; cancel the finite $\int g$. *(c)* With $|f_n|\le g$ and $f_n\to f$ a.e.: Fatou on $f_n+g\ge0$ gives $\int f\le\liminf\int f_n$; reverse Fatou gives $\limsup\int f_n\le\int f$. Sandwich.

**A5.** *(a)* $f_n = \mathbb{1}_{[n,\infty)}$ on $(\mathbb{R},\lambda)$: $f_n\downarrow0$ pointwise, yet $\int f_n = \infty$ for every $n$ while $\int0 = 0$. *(b)* $g_n := f_1-f_n\ge0$ increases to $f_1-f$; MCT gives $\int(f_1-f_n)\uparrow\int(f_1-f)$. Since $f_1\in L^1$ and $0\le f_n\le f_1$, every integral is finite, so $\int(f_1-f_n) = \int f_1-\int f_n$ (Proposition 2.13(2)), likewise for $f$; subtract the finite $\int f_1$. *(c)* Finiteness is used exactly at the subtraction — without it, $\int f_1-\int f_n$ is $\infty-\infty$. The set-measure analogue is continuity from above (Proposition 1.8(4)), which carries the same finiteness hypothesis for the same reason.

**A6.** Let $h_n := (p-p_n)^+$. Two observations do all the work. First, $0\le h_n\le p$ pointwise (as $p_n\ge0$) with $p\in L^1(\mu)$, and $h_n\to0$ $\mu$-a.e.; DCT gives $\int h_n\,d\mu\to0$. Second, both are densities, so $\int(p-p_n)\,d\mu = 0$, i.e., $\int(p-p_n)^+ = \int(p-p_n)^-$. Hence
$$\int|p_n-p|\,d\mu = 2\int h_n\,d\mu\to0,$$
and by Definition 2.31, $\mathrm{TV}(P_n,P) = \tfrac12\int|p_n-p|\,d\mu\to0$: a.e. convergence of densities implies total-variation convergence — a strong conclusion from a weak-looking hypothesis, contrasted in Chapter 5 with weak convergence. The converse fails: take the "typewriter sequence" on $[0,1]$ — intervals $I_n$ cycling through $[0,\tfrac12],[\tfrac12,1],[0,\tfrac14],\dots$ so every $x$ lies in $I_n$ infinitely often while $\lambda(I_n)\to0$ — and $p_n := (1+\mathbb{1}_{I_n})/(1+\lambda(I_n))$. Each $p_n$ is a density; $\int|p_n-1|\,d\lambda = O(\lambda(I_n))\to0$; yet $p_n(x)$ is near $2$ infinitely often for every $x$, so no a.e. convergence. (One always recovers a.e. convergence along a *subsequence* — the most that can be salvaged.)

**A7.** With $\log\frac{p_1}{p_2} = \log\frac{\sigma_2}{\sigma_1}-\frac{(x-\mu_1)^2}{2\sigma_1^2}+\frac{(x-\mu_2)^2}{2\sigma_2^2}$, integrate against $P_1$, using $\mathbb{E}[(x-\mu_1)^2] = \sigma_1^2$ and $\mathbb{E}[(x-\mu_2)^2] = \sigma_1^2+(\mu_1-\mu_2)^2$ (bias–variance split):
$$D_{\mathrm{KL}} = \log\frac{\sigma_2}{\sigma_1}-\frac12+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}.$$
*(a)* Set $u := \sigma_1^2/\sigma_2^2$, $\delta := (\mu_1-\mu_2)^2/(2\sigma_2^2)\ge0$: $D_{\mathrm{KL}} = \tfrac12(u-1-\log u)+\delta$, and $u-1-\log u\ge0$ with equality iff $u = 1$ (the tangent-line inequality of Proposition 2.33 again). So $D_{\mathrm{KL}} = 0$ iff both parameters agree. Sanity: $D_{\mathrm{KL}}(\mathcal{N}(0,1)\|\mathcal{N}(0,4)) = \log2+\tfrac18-\tfrac12 = 0.3181$. *(b)* As $\sigma_1\to0$ with $\mu_1\ne\mu_2$, $-\log\sigma_1\to+\infty$ dominates: $D_{\mathrm{KL}}\to+\infty$. Interpretation: $P_1\to\delta_{\mu_1}$, which is not $\ll\lambda$, hence has infinite KL to any Gaussian — the formula's divergence agreeing with the convention, the same lesson as Worked Example 2.16. *(c)* The combination $\log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2}{2\sigma_2^2}-\tfrac12$ vanishes precisely when $\sigma_1 = \sigma_2$, and the $\delta$ term vanishes when the means agree: an encoder outputting exactly the prior incurs KL exactly $0$ — no penalty whatsoever for ignoring $x$. Posterior collapse, exhibited as a stationary point of the loss you wrote down.

**A8.** *(a)* $\mathbb{E}_\theta[X] = \int_0^\theta\frac x\theta\,dx = \frac\theta2$, so $\frac{d}{d\theta}\mathbb{E}_\theta[X] = \frac12$. *(b)* $\log p_\theta = -\log\theta$ on $[0,\theta]$, so $\partial_\theta\log p_\theta = -1/\theta$ there, and $\mathbb{E}_\theta[X\,\partial_\theta\log p_\theta] = -\frac1\theta\cdot\frac\theta2 = -\frac12$. **Wrong sign**, not merely biased: gradient ascent moves $\theta$ exactly the wrong way. *(c)* Leibniz with a moving boundary:
$$\frac{d}{d\theta}\int_0^\theta\frac x\theta\,dx = \underbrace{\theta\cdot\tfrac1\theta}_{\text{boundary}}+\underbrace{\int_0^\theta\partial_\theta\Big(\frac x\theta\Big)dx}_{\text{interior}} = 1-\frac12 = \frac12 .$$
The score-function formula computes only the interior term $-\tfrac12$; the missing boundary contribution is $+1$. Increasing $\theta$ makes *new values available* — the support grows — and the score, which reports only how the density reweights *existing* values, cannot see this. *(d)* Hypothesis 1 of Proposition 2.23 ($\theta$-free support) fails; consequently Proposition 2.24 fails too: $\mathbb{E}_\theta[\partial_\theta\log p_\theta] = -1/\theta\ne0$. Cramér–Rao's regularity — the hypotheses of Theorem 2.21 — is violated, and the bound does not apply. Not academic: the MLE $\hat\theta = \max_iX_i$ has $\mathrm{Var}(\hat\theta) = \frac{n\theta^2}{(n+1)^2(n+2)} = O(n^{-2})$, beating the $O(n^{-1})$ a naive Cramér–Rao application would seem to forbid. When an estimator "beats Cramér–Rao," look for a $\theta$-dependent support.

**A9.** Let $\mu'' := \mu+\mu'$, which dominates both dominating measures. Write $p'' = dP/d\mu''$, $q'' = dQ/d\mu''$, and $r = d\mu/d\mu''$ (which exists: $\mu\ll\mu''$, both $\sigma$-finite). By the chain rule (Proposition 2.28(2)), $p = dP/d\mu$ satisfies $p'' = p\,r$ $\mu''$-a.e. on $\{r>0\}$, likewise $q'' = q\,r$. On $\{q''>0\}\cap\{r>0\}$ the ratio $p''/q'' = p/q$, and
$$\int_{\{q''>0\}}q''f\Big(\frac{p''}{q''}\Big)d\mu'' = \int_{\{q>0\}}q\,f\Big(\frac pq\Big)r\,d\mu'' = \int_{\{q>0\}}q\,f\Big(\frac pq\Big)d\mu$$
by Proposition 2.28(1). Bookkeeping for the two boundary sets: up to $\mu''$-null sets, $\{q''>0\} = \{q>0\}\cap\{r>0\}$ (since $q'' = qr$), so the two integration regions agree once the factor $r$ is present; and the atom terms match because $P(q''=0) = P(q=0)$ — indeed $\{q''=0\}\setminus\{q=0\}\subseteq\{r=0\}$ up to a $\mu''$-null set, and $P(r=0) = \int_{\{r=0\}}p''\,d\mu'' = \int_{\{r=0\}}pr\,d\mu'' = 0$, while $\{q=0\}\setminus\{q''=0\}$ is $\mu''$-null by the same factorization. Repeating with $\mu'$ in place of $\mu$ shows both computations equal the $\mu''$-computation. For TV: $f(t) = \tfrac12|t-1|$ gives $\int_{\{q>0\}}q\cdot\tfrac12|p/q-1|\,d\mu+\tfrac12P(q = 0) = \tfrac12\int_{\{q>0\}}|p-q|\,d\mu+\tfrac12\int_{\{q = 0\}}p\,d\mu = \tfrac12\int|p-q|\,d\mu = \mathrm{TV}$, using $f'(\infty) = \lim f(t)/t = \tfrac12$ and that $|p-q| = p$ on $\{q = 0\}$.

## Part B

All numbers produced by running the described code with the stated seeds; NumPy 2.4.4 / SciPy 1.17.1.

**B1.** *(a)* Quadrature: $\int x^2\varphi\,dx = 1.0000000$ (`quad`, error estimate $5.3\times10^{-9}$). Monte Carlo, `default_rng(0)`: $n = 10^4$: $0.996197$ (SE $0.013984$); $n = 10^6$: $1.001345$ (SE $0.001417$) — within $0.3$ and $1.0$ standard errors of the truth.

*(b)* Against $\delta_0+\lambda$ with the density of Worked Example 2.13:
$$\mathbb{E}[A] = 0\cdot\tfrac12+\int_0^\infty a\varphi(a)\,da = 0.3989423,\qquad\mathbb{E}[A^2] = 0\cdot\tfrac12+\int_0^\infty a^2\varphi(a)\,da = 0.5000000 .$$
Monte Carlo, `default_rng(1)`: $n = 10^4$: $\hat{\mathbb{E}}[A] = 0.393336$ (SE $0.005782$), $\hat{\mathbb{E}}[A^2] = 0.489002$, $\widehat{\mathbb{P}}(A = 0) = 0.5045$; $n = 10^6$: $0.397900$ (SE $0.000583$), $0.498203$, $0.4998$. Forgetting the atom — using the renormalized density $2\varphi$ on $(0,\infty)$ — gives $\int_0^\infty a\cdot2\varphi(a)\,da = 0.7978846 = \sqrt{2/\pi}$: exactly twice the truth, the classic factor-of-two bug.

*(c)* The atom sits at $a = 0$, so it contributes $0\cdot\tfrac12 = 0$ to $\mathbb{E}[A]$ and $\mathbb{E}[A^2]$: integrands vanishing at $0$ are blind to it. Integrands that do not vanish there are not:
$$\mathbb{E}[e^{-A}] = \tfrac12\cdot e^0+\int_0^\infty e^{-a}\varphi(a)\,da = 0.5+0.2615783 = 0.7615783,$$
the second term having closed form $e^{1/2}(1-\Phi(1)) = 0.2615783$. Monte Carlo ($10^6$, `default_rng(3)`): $0.761510$ (SE $0.000297$). Omitting the atom and renormalizing gives $2\times0.2615783 = 0.5231566$ — absolute error $0.238$, i.e., $31\%$. Lesson: whether an atom bug is visible depends entirely on the integrand, so testing mixed-distribution code only on the mean is not a test.

**B2.** *(a)* With $p = \varphi$, $q_\sigma(x) = \frac1\sigma\varphi(x/\sigma)$:
$$\frac{p(x)^2}{q_\sigma(x)} = \frac{\sigma}{\sqrt{2\pi}}\exp\Big(-x^2\Big(1-\frac{1}{2\sigma^2}\Big)\Big) = \frac{\sigma}{\sqrt{2\pi}}e^{-ax^2},\qquad a := 1-\frac{1}{2\sigma^2}.$$
Multiplying by $f^2 = x^4$ and integrating: convergent iff $a>0$, i.e., $\boxed{\sigma>1/\sqrt2 = 0.7071068}$. Using $\int x^4e^{-ax^2}dx = \tfrac34\sqrt\pi\,a^{-5/2}$:
$$\mathbb{E}_{q_\sigma}[w^2f^2] = \frac{3\sigma}{4\sqrt2}\,a^{-5/2},\qquad\mathrm{Var}(wf) = \frac{3\sigma}{4\sqrt2}\,a^{-5/2}-1 .$$
Checks: $\sigma = 1.5$: closed form $1.4910723$, quadrature over $[-50,50]$: $1.4910723$. $\sigma = 1$: $a = \tfrac12$, closed form $3 = \mathbb{E}[Z^4]$, as it must be since $w\equiv1$.

*(b), (d)* $200$ trials of $n = 10^5$, `default_rng(2024)`: the analytic variance/sd, estimate means, across-trial sd, median reported SE, and coverage are the table of Worked Example 2.17 and are not repeated. What that table omits and (d) asks for — the *spread* of the reported error bar and the diagnostics:

| $\sigma$ | reported SE: q10 / median / q90 | q90/q10 | median ESS (of $10^5$) | median max-$w$ share | median $\hat k$ |
|---|---|---|---|---|---|
| $1.5$ | $0.00221$ / $0.00222$ / $0.00222$ | $1.005$ | $83{,}142$ | $0.00002$ | $-1.803$ |
| $1.0$ | $0.00443$ / $0.00447$ / $0.00451$ | $1.018$ | $100{,}000$ | $0.00001$ | n/a ($w\equiv1$) |
| $0.9$ | $0.00629$ / $0.00648$ / $0.00670$ | $1.065$ | $97{,}219$ | $0.00006$ | $0.191$ |
| $0.8$ | $0.01003$ / $0.01130$ / $0.01413$ | $1.409$ | $83{,}531$ | $0.00030$ | $0.345$ |
| $0.72$ | $0.01526$ / $0.01957$ / $0.03309$ | $2.168$ | $57{,}927$ | $0.00092$ | $0.452$ |
| $0.6$ | $0.02680$ / $0.04345$ / $0.10823$ | $4.039$ | $16{,}239$ | $0.00381$ | $0.591$ |
| $0.45$ | $0.03995$ / $0.07837$ / $0.27671$ | $6.926$ | $2{,}075$ | $0.01443$ | $0.735$ |

Across-trial sd matches the analytic prediction for $\sigma\ge0.9$ ($0.0023$ vs $0.002216$; $0.0048$ vs $0.004472$; $0.0070$ vs $0.006533$) and then decouples: at $\sigma = 0.72$ theory says $0.1268$, $200$ trials show only $0.0250$ — the variance is carried by weights so rare that $2\times10^7$ draws have not sampled them. The fraction of trials within $1\%$ of the truth falls $1.00,\ 0.96,\ 0.85,\ 0.54,\ 0.32,\ 0.10,\ 0.01$ across the rows.

*(c)* $\sigma = 1.5$ beats $\sigma = 1$ (sd $0.00222$ vs $0.00447$) because $f(x) = x^2$ weights the tails: the optimal proposal must be heavier-tailed than the target. By Proposition 2.40 the optimum is $q^\ast\propto x^2\varphi(x)$, already normalized since $\mathbb{E}[Z^2] = 1$. Sampling it exactly ($|x|\sim\chi_3$, random sign; $10^5$ samples, `default_rng(5)`): $\hat I = 1.0$ with across-sample sd $1.2\times10^{-16}$ — zero to machine precision, since $wf = pf/q^\ast\equiv1$.

*(d)* The catastrophe is invisible to the estimator's own error bar: at $\sigma = 0.6,0.45$ the variance is infinite, yet the median reported SE is $0.0435$ and $0.0784$, while nominal-95% coverage is $0.765$ and $0.430$. Note also the *spread* of the reported SE (q90/q10 spanning $4\times$ and $7\times$, versus $1.005$ at $\sigma = 1.5$): an unstable error bar is itself a symptom, since a finite-variance estimator's SE concentrates. Diagnostics: ESS collapses $83{,}142\to2{,}075$ ($40\times$); max-weight share rises $2\times10^{-5}\to1.4\times10^{-2}$; PSIS $\hat k$ rises monotonically $0.191\to0.735$ (fitting a generalized Pareto to the largest $\min(n/5,3\sqrt n)$ weight exceedances, per Vehtari et al.). Honest caveat, worth internalizing: $\hat k\ge0.5$ is the *theoretical* infinite-variance boundary and is correctly crossed at $\sigma = 0.6$ ($\hat k = 0.591$) — but the popular $\hat k>0.7$ rule is **not** crossed there, so a practitioner following the rule of thumb ships a provably infinite-variance estimator. When the question is variance, use $0.5$. Connection to RL: PPO clips $w = \pi_\theta/\pi_{\theta_{\text{old}}}$ to $[1-\epsilon,1+\epsilon]$; V-trace truncates at $\bar\rho$; both bound the Radon–Nikodym derivative, hence the variance, at the cost of bias — exactly the trade this table argues for.

**B3.** Analytic per-sample variances (Worked Example 2.12): reparameterized $\hat g = 2(1+\varepsilon)$: mean $2$, variance $4$. Score-function $\hat g = (1+\varepsilon)^2\varepsilon$: mean $2$, second moment $\mathbb{E}[(1+\varepsilon)^4\varepsilon^2] = 1+6\cdot3+15 = 34$, variance $30$; ratio $7.5$. Simulated means/variances/ratios for $B\in\{1,10,100\}$ (`default_rng(42)`, $5000$ replications) are the table of §2.6.2; variances scale as $1/B$ and the ratio brackets $7.5$. Compute multiplier: matching reparameterized variance requires $\approx7.5\times$ the batch, i.e., $7.5\times$ the forward passes — in a benign one-dimensional problem. Discrete latents cannot use Corollary 2.22 because it needs a $\theta$-free noise source and a $T_\theta$ differentiable in $\theta$: any map producing a categorical sample from fixed noise is piecewise constant in $\theta$ (derivative zero a.e., undefined at jumps). Workarounds: (i) **Gumbel–Softmax / Concrete** relaxations — a temperature-controlled continuous relaxation with differentiable $T_\theta$, biased except in the zero-temperature limit where variance explodes; (ii) **straight-through** estimators — discrete forward pass, surrogate Jacobian backward, biased by construction, justified empirically; (iii) **control variates** for the score-function estimator (NVIL, MuProp, REBAR, RELAX) — unbiased, attacking variance directly.

**B4.** *(a)* $P$ lives on $S_0$, $Q_\alpha$ on $S_\alpha$: disjoint sets of full measure for their own laws, so $P\perp Q_\alpha$. Then $P\not\ll Q_\alpha$ (indeed $Q_\alpha(S_0) = 0$ while $P(S_0) = 1$), so $D_{\mathrm{KL}} = +\infty$ by Definition 2.32, and $D_{\mathrm{JS}} = \log2$ by Proposition 2.36. No integral computed.

*(b)* Smoothing the $y$-coordinate by $\mathcal{N}(0,h^2)$ gives both laws densities on $[0,1]\times\mathbb{R}$; the $x$-marginals coincide, so everything reduces to $\mathcal{N}(0,h^2)$ vs $\mathcal{N}(\alpha,h^2)$: $D_{\mathrm{KL}} = \alpha^2/(2h^2)$ by A7. $D_{\mathrm{JS}}$ by trapezoid quadrature ($2\times10^5$ points):

| $\alpha$ | $h = 0.5$: KL / JS | $h = 0.1$: KL / JS | $h = 0.01$: KL / JS | $W_1$ |
|---|---|---|---|---|
| $0.01$ | $0.0002$ / $0.000050$ | $0.0050$ / $0.001248$ | $0.5000$ / $0.111421$ | $0.01$ |
| $0.05$ | $0.0050$ / $0.001248$ | $0.1250$ / $0.030311$ | $12.5000$ / $0.675943$ | $0.05$ |
| $0.1$ | $0.0200$ / $0.004975$ | $0.5000$ / $0.111421$ | $50.0000$ / $0.693146$ | $0.1$ |
| $0.5$ | $0.5000$ / $0.111421$ | $12.5000$ / $0.675943$ | $1250.0000$ / $0.693147$ | $0.5$ |
| $1.0$ | $2.0000$ / $0.336831$ | $50.0000$ / $0.693146$ | $5000.0000$ / $0.693147$ | $1.0$ |
| $2.0$ | $8.0000$ / $0.632720$ | $200.0000$ / $0.693147$ | $20000.0000$ / $0.693147$ | $2.0$ |

$W_1 = \alpha$ exactly, for every $h$ — provable in two lines rather than estimated. *Upper bound:* the shift $(x,y)\mapsto(x,y+\alpha)$ pushes the smoothed $P$ to the smoothed $Q_\alpha$ and moves every point exactly $\alpha$. *Lower bound:* for any coupling $\pi$, $\int\|(x,y)-(x',y')\|\,d\pi\ge\int|y'-y|\,d\pi\ge|\mathbb{E}_{Q}[y']-\mathbb{E}_P[y]| = \alpha$ (Jensen/triangle inequality for integrals). Upper meets lower.

*(c)* At $h = 0.01$, $D_{\mathrm{JS}}$ agrees to six decimals across $\alpha\in[0.1,2.0]$: a generator that must reduce $\alpha$ receives $\partial_\alpha D_{\mathrm{JS}}\approx0$ — no signal, exactly as Proposition 2.36 predicts in the limit — while $\partial_\alpha W_1 = 1$ uniformly. The middle rows show the practical fix: at $h = 0.5$ the smoothed JS *does* vary with $\alpha$ ($0.00005\to0.63$), which is why instance noise on both real and fake samples restores a usable gradient: it manufactures mutual absolute continuity.

## Part C

**C1.** *(a)* Let $s := \nabla_\theta\log p_\theta(X)$. Under the hypotheses of Proposition 2.23 ($\theta$-free support, integrable domination of $f\,\partial_\theta p_\theta$), $\mathbb{E}[fs] = \nabla_\theta\mathbb{E}_{p_\theta}[f]$; under those of Proposition 2.24 (integrable domination of $\partial_\theta p_\theta$), $\mathbb{E}[s] = 0$. Linearity: $\mathbb{E}[(f-b)s] = \mathbb{E}[fs]-b\,\mathbb{E}[s] = \nabla_\theta\mathbb{E}[f]$ for every constant $b$. Both ingredients are needed, and both are consequences of Theorem 2.21.

*(b)* Since $\mathbb{E}[\hat g_b]$ is $b$-free, minimizing $\mathrm{tr}\,\mathrm{Cov}(\hat g_b)$ is minimizing $\mathbb{E}\|\hat g_b\|^2 = \mathbb{E}[(f-b)^2\|s\|^2]$:
$$\Phi(b) = \mathbb{E}[f^2\|s\|^2]-2b\,\mathbb{E}[f\|s\|^2]+b^2\,\mathbb{E}[\|s\|^2],$$
a convex quadratic; $\Phi'(b^\ast) = 0$ gives $b^\ast = \mathbb{E}[f\|s\|^2]/\mathbb{E}[\|s\|^2]$ — a **score-norm-weighted mean of $f$**, not $\mathbb{E}[f]$. Intuition worth keeping: the baseline should match $f$ where the *score* is large — the regions contributing most gradient variance — not where $f$ is typical.

*(c)* For $p_\mu = \mathcal{N}(\mu,1)$ at $\mu = 1$, $f(x) = x^2$: $s = \varepsilon := x-1\sim\mathcal{N}(0,1)$, so
$$b^\ast = \frac{\mathbb{E}[(1+\varepsilon)^2\varepsilon^2]}{\mathbb{E}[\varepsilon^2]} = \frac{1+0+3}{1} = 4,\qquad\text{whereas } \mathbb{E}[f] = \mu^2+1 = 2 .$$
Single-sample variances: $b = 0$: $30$; $b = 2$: $\mathbb{E}[(\varepsilon^3+2\varepsilon^2-\varepsilon)^2]-4 = (15+12+1-6)-4 = 18$; $b = b^\ast = 4$: $\mathbb{E}[(\varepsilon^3+2\varepsilon^2-3\varepsilon)^2]-4 = (15+12+9-18)-4 = 14$. Simulation (`default_rng(42)`, $5000$ replications, $B = 1$): $29.3461$, $17.5267$, $13.6986$ — matching $30$, $18$, $14$. The naive baseline captures most of the available reduction ($30\to18$); the optimal one adds a further $22\%$ ($18\to14$); reparameterization remains $3.5\times$ better at $4$.

*(d)* Two reasons actor–critic uses $V(s)\approx\mathbb{E}[\text{return}\mid s]$ rather than $b^\ast$. (i) **$b^\ast$ is harder to estimate than the thing it improves**: a ratio of two expectations, both involving $\|\nabla\log\pi_\theta\|^2$ — itself high-variance and changing every step; a noisy $\hat b^\ast$ can *increase* variance, and correlating the baseline with the sampled action destroys the unbiasedness of (a) (the proof needs $b$ independent of the sample). (ii) **The dominant variance in RL is across states, not across the score norm**: returns differ by orders of magnitude between states, so a *state-dependent* $V(s)$ removes far more variance than any constant, optimal or not — and $V$ is needed anyway for bootstrapping and advantage estimation, so it is free. The exact optimum of a clean criterion is not the engineering answer; a suboptimal member of a richer class dominates the optimal member of a poor one.

**C2.** *(a)* Assume $q\ll\mu_Z$ with density $q$, and (discussed in (c)) $q\ll p_\theta(\cdot|x)$. From the definition of the conditional density, $p_\theta(x,z) = p_\theta(x)\,p_\theta(z|x)$; for $z$ in the support of $q$,
$$\log p_\theta(x,z)-\log q(z) = \log p_\theta(x)+\log\frac{p_\theta(z|x)}{q(z)}.$$
Take $\mathbb{E}_{z\sim q}$: the left side is $\mathcal{L}$; on the right, $\log p_\theta(x)$ is constant in $z$ and passes through ($\int1\,dq = 1$); the remaining term is $-\mathbb{E}_q[\log\frac{q}{p_\theta(\cdot|x)}] = -D_{\mathrm{KL}}(q\|p_\theta(\cdot|x))$. Hence $\log p_\theta(x) = \mathcal{L}+D_{\mathrm{KL}}(q\|p_\theta(\cdot|x))$; the only measure-theoretic input is linearity, valid when the pieces are integrable (or read consistently in $[-\infty,\infty]$).

*(b)* $D_{\mathrm{KL}}\ge0$ with equality iff the laws coincide (Proposition 2.33). Since $\log p_\theta(x)$ is $q$-free, $\mathcal{L}\le\log p_\theta(x)$, with equality iff $q = p_\theta(\cdot|x)$ as measures — i.e., densities equal $\mu_Z$-**a.e.**, not everywhere: a consequence of §2.4, and the reason "the variational posterior equals the true posterior" is a statement about equivalence classes.

*(c)* Two distinct absolute-continuity requirements: $q\ll\mu_Z$, so the density $q$ and $\log q$ exist; and $q\ll p_\theta(\cdot|x)$, so $D_{\mathrm{KL}}(q\|p_\theta(\cdot|x))<\infty$. If $q$ puts mass where the posterior has none: $D_{\mathrm{KL}} = +\infty$ *and* $\mathcal{L} = -\infty$ (there $\log p_\theta(x,z) = -\infty$, since $p_\theta(x)>0$ forces $p_\theta(x,z) = 0$ exactly where $p_\theta(z|x) = 0$); the *identity* becomes $\log p_\theta(x) = -\infty+\infty$, indeterminate — and the derivation breaks a step earlier, at the linearity split, whose hypothesis is integrability. What survives is the *inequality* $\mathcal{L}\le\log p_\theta(x)$, trivially ($-\infty\le$ anything); since training relies on the bound, not the identity, the ELBO remains a valid if uninformative objective — in practice the optimizer sees a $-\infty$/`NaN` reconstruction term. If instead the *posterior* puts mass where $q$ has none: reverse KL is entirely unbothered — nothing in $\int q\log(q/p)$ inspects regions where $q = 0$ — the identity stays valid and finite, and the gap is simply large. That asymmetry *is* the mode-seeking behaviour of §2.8.3, seen from inside the ELBO.

*(d)* Tight in principle: $\mathcal{L} = \log p_\theta(x)$ exactly when $q$ is the posterior. In practice $\{q_\phi\}$ is Gaussian and the posterior is not, so the gap is $\min_\phi D_{\mathrm{KL}}(q_\phi\|p_\theta(\cdot|x))>0$ — and by Worked Example 2.15, the minimizer of a reverse KL against a multimodal target is mode-seeking: the reverse-KL fit sat on one mode ($m = 2.9843$, $s = 1.0234$) with residual divergence $0.6888$, while the mass-covering alternative scored $0.9380$. The ELBO gap is not generic slack: it is slack of a specific shape, systematically favouring over-concentrated posteriors. Hence richer $q$ families (flows for $q$, importance-weighted bounds, semi-amortized refinement) improve the *bound* rather than the model — and a rising ELBO can mask a stationary $\log p_\theta(x)$.

**C3.** *(a) Alice and Bob.* By Proposition 2.30 with $c = 1/255$, per-dimension log-densities on the $[0,1]$ scale exceed those on $[0,255]$ by $\log255 = 5.5412635$ nats. Alice's $+3.2$ nats/dim on $[0,1]$ is $3.2-5.5413 = -2.3413$ nats/dim on $[0,255]$; Bob reports $-2.3$. **They agree to the precision reported** (the difference, $0.041$ nats/dim, is below Bob's rounding), so the results are consistent and plausibly the same model. Bits/dim, 256-level convention: Alice $8-3.2/\log2 = 3.3834$; Bob $2.3/\log2 = 3.3182$; the residual $0.065$ bits/dim is rounding plus the $255$-vs-$256$ ambiguity ($\log256-\log255 = 0.0039$ nats $= 0.0056$ bits). **Any comparison at the third decimal of bits/dim is meaningless unless both papers state their scaling and dequantization conventions.**

*(b) Carol* is wrong, and §2.7.4 says why: a density is a Radon–Nikodym derivative $dP/d\lambda_d$ — probability per unit volume — and rates are unbounded above. On $[0,1]$-scaled data the reference volume is small, so densities well above $1$, hence positive log-densities, are expected, not buggy ($\mathcal{N}(0,0.01^2)$ has density $39.894$ at its mode). What *would* be a bug: a positive log-likelihood from a probability *mass* function, or a claim that $\int p\,d\lambda>1$. Carol is, however, right to be suspicious in general; the question worth asking is not "is the number positive?" but "what base measure, what dequantization, and is the objective bounded above?"

*(c) Dequantization, four steps.*
1. **Discrete data has no Lebesgue density.** Pixels take finitely many values, so the data law is $\hat P = \sum_k\pi_k\delta_{x_k}$, atomic, with $\hat P\not\ll\lambda_d$ (Definition 2.25): there is no $d\hat P/d\lambda_d$, and "the likelihood of the data under a continuous model" is not the same object as the model's density evaluated at the data — a fact the notation obscures.
2. **A continuous model fitted to discrete data can drive the likelihood to $+\infty$.** A spike of width $\epsilon$ and height $\propto\epsilon^{-d}$ at each data point makes the average log-density grow like $d\log(1/\epsilon)\to\infty$ — Worked Example 2.10 with $n = 1/\epsilon$, and the VAE collapse of §2.5. A reported likelihood of $+\infty$ says nothing about quality; the objective is unbounded, so maximizing it is ill-posed.
3. **Uniform dequantization yields a valid lower bound.** Let $y := x+u$, $u\sim\mathrm{Unif}[0,1)^d$ independent of the discrete $x$; the law of $y$ has Lebesgue density $p^{\text{cont}}(y) = \hat P(\{\lfloor y\rfloor\})$, constant on unit cubes. For a continuous model $p_\theta$, Jensen (Proposition 2.52, concave $\log$ inside the uniform average) gives
$$\log\mathbb{P}_\theta(x) := \log\int_{[0,1)^d}p_\theta(x+u)\,du\ \ge\ \int_{[0,1)^d}\log p_\theta(x+u)\,du = \mathbb{E}_u[\log p_\theta(x+u)],$$
where $\mathbb{P}_\theta(x)$ is the probability the model assigns to the *discrete* symbol $x$ (its density integrated over the corresponding cube). The right side is what adding uniform noise estimates. Dequantized log-likelihood is therefore a **lower bound on a genuine discrete log-likelihood** — comparable across models and immune to the spike exploit, since the objective averages $\log p_\theta$ over the whole cube.
4. **Variational dequantization tightens it.** Replace $\mathrm{Unif}[0,1)^d$ by a learned $q(u|x)$ on $[0,1)^d$; the same Jensen argument gives $\log\mathbb{P}_\theta(x)\ge\mathbb{E}_{u\sim q}[\log p_\theta(x+u)-\log q(u|x)]$ — an ELBO of exactly C2's form, tight when $q(u|x)\propto p_\theta(x+u)$ on the cube. Uniform dequantization is the case $q = $ uniform, loose precisely when the model's density varies within a cube — which it does for sharp models, so the fix matters most for the best models (Flow++).

*Verdict.* Alice: fine; should state her scaling. Bob: fine, consistent with Alice. Carol: wrong about positivity, right to demand conventions.

# Appendix 2.A — Answers to self-checks

**2.1.** (1) Riemann needs "width," which is only defined for intervals; Lebesgue needs only the *measure* of $\{f>t\}$, defined for every set in $\mathcal{F}$. (2) The first and second (they use only the measure $\mathbb{P}$ and the values of $X$); the third also works since $X\ge0$ — what fails is nothing: all three are measure-theoretic. What fails is any *Riemann* computation, since image space carries no notion of interval. (3) At the moment we form $\{f>t\}$ and ask for its measure: $\{f>t\}\in\mathcal{F}$ for all $t$ is measurability. (4) Part 2, in the direction "improper Riemann exists, Lebesgue undefined" ($\sin x/x$); the other direction (Lebesgue-integrable but not Riemann-integrable) is $\mathbb{1}_{\mathbb{Q}}$.

**2.2.** (1) The zero function on $(\mathbb{R},\lambda)$: its canonical representation is $0\cdot\mathbb{1}_{\mathbb{R}}$ with $\lambda(\mathbb{R}) = \infty$, and its integral must be $0$. (2) "The integral exists (and therefore equals its symmetric value)" — symmetry gives the value only conditional on existence, and both tail integrals are infinite. (3) Cauchy: $\bar X_n$ is standard Cauchy for every $n$ — no convergence of any kind. Pareto with $\mathbb{E}X = +\infty$: $\bar X_n\to+\infty$ a.s. (4) $\int f\,d\lambda = \lambda(\{0\}) = 0$; $\int f\,d\# = \#(\{0\}) = 1$; $\int f\,d\delta_0 = f(0) = 1$. (5) Through continuity from below, $\mu(A_i\cap E_n)\to\mu(A_i)$ — equivalent to countable additivity given finite additivity (Proposition 1.9).

**2.3.** (1) Indicators, simple, non-negative, general; Rung 1 (the pushforward definition) carries the content. (2) $P_A\not\ll\lambda$: $P_A(\{0\}) = \tfrac12>0 = \lambda(\{0\})$, so no Lebesgue density exists. (3) It equals $\int g\,d\hat P_B$ exactly, by Corollary 2.12(3); all approximation error is in $\hat P_B$ vs $P$. (4) From Rung 4: $(g\circ X)^\pm = g^\pm\circ X$ and Rung 3 gives *equality* of each pair of integrals, so finiteness on one side is finiteness on the other.

**2.4.** (1) $\mathbb{1}_C$ and $0$, for $C$ the Cantor set: uncountable, $\lambda(C) = 0$. (2) Because $f$ may be positive on a null set — Proposition 2.14(2) concludes only $f = 0$ *a.e.* (3) $0.04/1 = 0.04$; loose because Markov uses only the mean, and a loss concentrated near $0.04$ is far below $1$ almost always. (4) That both models' code evaluates a *continuous* representative: a continuous function is determined by its values off any null set, so the printed numbers identify the equivalence classes' unique continuous members and comparing them is well defined.

**2.5.** (1) MCT: monotonicity fails ($f_1\not\le f_2$). DCT: any dominator majorizes $\sup_nf_n\ge\mathbb{1}_{[1,\infty)}$, whose integral is $\infty$. Vitali: uniform integrability holds (bounded family), tightness fails. (2) Strict: $f_n = \mathbb{1}_{[n,n+1]}$. Equality: $f_n = (1+\tfrac1n)\mathbb{1}_{[0,1]}$ (or any $L^1$-convergent sequence). (3) Constants are integrable iff $\mu(\Omega)<\infty$; on $(\mathbb{R},\lambda)$, $\int M\,d\lambda = \infty$ for $M>0$. (4) Vertical (a spike of height $\propto1/\sigma$); $n = 1/\sigma$.

**2.6.** (1) The MVT point $\xi_{n,x}$ lands at an uncontrolled parameter value; only a $\theta$-uniform bound covers it, so a $\theta$-dependent $g_\theta$ leaves $|\Delta_n|$ unbounded. (2) It moves $\theta$ from the measure into the integrand; the licence is change of variables (Theorem 2.11). (3) Hypothesis 1 ($\theta$-free support): the derivative acquires a boundary term the estimator misses, producing $-\tfrac12$ instead of $+\tfrac12$ — wrong sign (Exercise A8). (4) $\mathbb{E}[(f-b)s] = \mathbb{E}[fs]-b\,\mathbb{E}[s]$, and the second term vanishes.

**2.7.** (1) "$\nu$ puts no mass where $\mu$ sees nothing," i.e., $\mu(A) = 0\Rightarrow\nu(A) = 0$. $\delta_0\not\ll\lambda$: fails at $A = \{0\}$. $\lambda\not\ll\delta_0$: fails at $A = [1,2]$. (2) $\delta_0+\lambda$ sees both the atom and the continuous part; $\lambda$ misses the atom ($P_A\not\ll\lambda$), $\#$ misses the continuous part (no countable carrier). (3) "A density is a rate — probability per unit volume — not a probability; $150$ per unit volume over a region of volume $\ll1/150$ keeps every probability below $1$." (4) $3.2-\log255 = -2.3413$ nats/dim; $8-3.2/\log2 = 3.3834$ bits/dim.

**2.8.** (1) The negative part of $w\log w$ is bounded by $1/e$, hence integrable against a probability measure; $\mathbb{E}[X]$ can have both parts infinite. (2) Maximum likelihood is forward; variational inference is reverse; reverse is mode-seeking. (3) $f(0) = \tfrac12$, $f'(\infty) = \lim\tfrac{|t-1|}{2t} = \tfrac12$; sum $1 = \mathrm{TV}$ between singular laws. ✓ (4) $W_1$ is built from the ground metric $\|x-y\|$, which varies continuously with separation, while every $f$-divergence sees only the ratio $dP/dQ$, degenerate identically between singular laws; the price is $n^{-1/d}$ estimation, surrogate critics, and metric dependence.

**2.9.** (1) $P\ll Q$ — the behaviour policy must cover the target's support; otherwise the estimator is systematically blind to a region, with bias no data reduces. (2) Because variance depends on $w^2f^2$, not on $Q$ matching $P$; weighting by $|f|$ is optimal: $q^\ast\propto x^2\varphi(x)$ ($|x|\sim\chi_3$ with random sign). (3) Almost nothing: with heavy-tailed weights the reported SE can look healthy while the variance is infinite; compute ESS, max-weight share, and PSIS $\hat k$ (variance question: use the $0.5$ boundary). (4) Clipping replaces the unbounded Radon–Nikodym derivative $w$ by a bounded surrogate: it abandons exact unbiasedness ($\mathbb{E}_Q[wf] = \mathbb{E}_P[f]$) to guarantee finite variance.

**2.10.** (1) The probability is $P^{\otimes n}$ of a set of *datasets* in $\mathcal{X}^n$. (2) Tonelli on $|f|$ first (always legal), then Fubini on $f$; skipping step 1 risks the $\pi/4$ vs $-\pi/4$ counterexample, where both iterated integrals exist and disagree. (3) $\mathbb{E}[XY] = \int xy\,d(P_X\otimes P_Y) = \int x\,dP_X\int y\,dP_Y$ — Proposition 2.45 (independence = product law) for the first equality, Fubini for the second. (4) Where the inner integral computes $\int_0^\infty\mathbb{1}\{t<X(\omega)\}\,dt = X(\omega)$, valid only for $X\ge0$; signed version: $\mathbb{E}[X] = \int_0^\infty\mathbb{P}(X>t)\,dt-\int_0^\infty\mathbb{P}(X<-t)\,dt$ for $X\in L^1$.

**2.11.** (1) $\|f\|_p = 0$ only forces $f = 0$ a.e. (Proposition 2.14), so the "norm" separates points only on equivalence classes. (2) Minkowski (Theorem 2.49), derived from Hölder (Theorem 2.48). (3) Jensen/Cauchy–Schwarz with the constant $1$, integrable only when $\mu(\Omega)<\infty$; on $(\mathbb{R},\lambda)$ both inclusions fail ($1/(1+|x|)$, $|x|^{-1/2}\mathbb{1}_{(0,1)}$). (4) Completeness: conditional expectation is the orthogonal projection onto the closed subspace $L^2(\Omega,\mathcal{G},\mathbb{P})$, and projections onto closed subspaces exist only in complete spaces.

# References and further reading

**Textbooks (measure-theoretic probability and real analysis).**
- G. Folland, *Real Analysis: Modern Techniques and Their Applications*, 2nd ed. Chapters 2 (integration, Fubini–Tonelli), 3 (Radon–Nikodym, Lebesgue decomposition), 6 ($L^p$, Hölder, Minkowski, uniform integrability). The reference for every result cited but not proved.
- R. Durrett, *Probability: Theory and Examples*, 5th ed. §1.4–1.7 build the integral, expectation, and product measures in the probabilist's idiom; the appendix covers Radon–Nikodym.
- P. Billingsley, *Probability and Measure*, 3rd ed. §§15–21, 32. Slower and thorough; §32 is the classical Radon–Nikodym treatment.
- D. Williams, *Probability with Martingales*. Chapters 5–6, 13. Short, sharp, unusually good on the standard ladder as a *technique* (his "standard machine") and on uniform integrability.
- W. Rudin, *Real and Complex Analysis*, 3rd ed. Chapters 1–3, 6. Terse; the place to see $L^p$ completeness done cleanly.
- T. Tao, *An Introduction to Measure Theory*. Free online; the most patient treatment of why the Lebesgue integral is built the way it is, including Lebesgue's criterion (Theorem 2.1).
- C. Villani, *Optimal Transport: Old and New*. Chapters 5–6: Wasserstein distances and Kantorovich duality (§2.8.5).
- T. Cover and J. Thomas, *Elements of Information Theory*, 2nd ed. Chapters 2, 8, 11: KL, differential entropy and its base-measure dependence, Pinsker.
- A. Tsybakov, *Introduction to Nonparametric Estimation*. §2.4: the divergence inequalities (Pinsker et al.) in the form statisticians use.
- J. Weed and F. Bach, "Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance," *Bernoulli* 2019 — the $n^{-1/d}$ rate behind §2.8.5's caveat 1.

**ML papers where this chapter's results do the work.**
- D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," ICLR 2014 — the reparameterization trick (Corollary 2.22) and the ELBO (Exercise C2).
- R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, 1992 — REINFORCE (Proposition 2.23) and baselines (Exercise C1).
- I. Goodfellow et al., "Generative Adversarial Networks," NeurIPS 2014 — the JS divergence at the optimal discriminator (Definition 2.35).
- M. Arjovsky and L. Bottou, "Towards Principled Methods for Training Generative Adversarial Networks," ICLR 2017 — the singular-support diagnosis and instance noise (Proposition 2.36, Worked Example 2.16).
- M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein GAN," ICML 2017 — the $W_1$ alternative and the parallel-segments example (Proposition 2.38); read alongside §2.8.5's caveats.
- I. Gulrajani et al., "Improved Training of Wasserstein GANs," NeurIPS 2017 — the gradient penalty; part of caveat 2's story.
- L. Theis, A. van den Oord, M. Bethge, "A note on the evaluation of generative models," ICLR 2016 — log-likelihood vs sample quality vs bits/dim (Exercise C3).
- L. Dinh, J. Sohl-Dickstein, S. Bengio, "Density estimation using Real NVP," ICLR 2017; D. P. Kingma, P. Dhariwal, "Glow," NeurIPS 2018 — dequantization conventions and bits/dim in practice.
- J. Ho, X. Chen, A. Srinivas, Y. Duan, P. Abbeel, "Flow++," ICML 2019 — variational dequantization (Exercise C3, step 4).
- E. Jang, S. Gu, B. Poole, "Categorical Reparameterization with Gumbel-Softmax," ICLR 2017; C. Maddison, A. Mnih, Y. Teh, "The Concrete Distribution," ICLR 2017 — when Corollary 2.22 does not apply.
- A. Vehtari, D. Simpson, A. Gelman, Y. Yao, J. Gabry, "Pareto Smoothed Importance Sampling," *JMLR* 2024 — the $\hat k$ diagnostic (§2.9).
- J. Schulman et al., "Proximal Policy Optimization Algorithms," 2017; L. Espeholt et al., "IMPALA," ICML 2018 — clipping/truncating Radon–Nikodym derivatives for bounded variance.
- U. Şimşekli, L. Sagun, M. Gürbüzbalaban, "A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks," ICML 2019 — evidence that finite-variance hypotheses behind standard SGD rates can fail (Worked Examples 2.3, 2.8).
- S. Bowman et al., "Generating Sentences from a Continuous Space," CoNLL 2016 — posterior collapse (§2.8.3, Exercise A7(c)).
