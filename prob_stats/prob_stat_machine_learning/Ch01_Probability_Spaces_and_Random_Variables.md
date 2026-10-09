# Probability and Statistics for Machine Learning and Deep Learning
### A self-contained tutorial for upper-year undergraduates

**Chapter 1 — Probability Spaces and Random Variables**

*Revision 3. Changes from revision 2, in decreasing order of importance:*

1. *Corrected a false claim in §1.10.1: non-constancy of a generator does **not** rule out atoms in its output distribution — ReLU generators can (and do) produce atoms. The correct condition is now stated, with a counterexample.*
2. *Corrected the uniqueness statement for Lebesgue measure (§1.6): local finiteness is implied by translation invariance plus normalization, so it is a convenience, not a needed hypothesis. Revision 2 claimed the opposite.*
3. *The existence of a probability space carrying infinitely many independent draws (used for infinite coin flips in §1.13 and the SGD filtration in Exercise C1) is a genuine theorem — the Kolmogorov extension theorem — and is now listed in the debt ledger of §1.14, which previously and wrongly claimed to be complete.*
4. *§1.8.3 no longer claims that the separability argument "covers essentially all of deep learning." It covers losses that are continuous in the parameters; the 0–1 loss of classification is not, and now gets its own honest paragraph.*
5. *Conditional independence (§1.12.1) is now defined only for the case this chapter can actually support (conditioning on the values of a discrete variable), with a forward pointer to Chapter 6 for the general case.*
6. *New material: inverse-transform sampling as a worked example (§1.10.2) and a lab (B6); a concrete singular-continuous distribution (the Cantor distribution, §1.10.4); a remark distinguishing the Borel and Lebesgue σ-algebras (§1.6); a statement of the second Borel–Cantelli lemma next to the first (Exercise A9).*
7. *Editorial: figures renumbered in order of appearance and converted to renderer-independent text diagrams; a broken sentence in Common Confusion 1.5 repaired; the cross-reference for the countable-additivity lemma corrected; two overstatements ("strictly more information", "every set you will ever write down is Borel") replaced by accurate versions.*

---

## Preface: how to use this tutorial

**Who this is for.** You know single-variable and multivariable calculus, linear algebra, and a first course in probability — you have met random variables, expectation, and the normal distribution, and you can compute with them. You now want the probability and statistics that modern machine learning research actually runs on: enough to read a NeurIPS or ICML paper, follow its derivations line by line, and eventually write your own.

**What is different from your first probability course.** A first course teaches you to *compute*. This tutorial teaches you to *reason*, especially in the two regimes where machine learning lives.

The first is **high dimensions**. Your geometric intuition was built in one, two, and three dimensions, and in a thousand dimensions it is not merely imprecise — it is systematically wrong. Nearly all the volume of a high-dimensional ball sits in a thin shell near the surface. Two random directions are almost exactly perpendicular. A Gaussian looks less like a fuzzy blob and more like a soap bubble. Deep learning happens here, and Chapter 12 is devoted to rebuilding your intuition for it.

The second is **randomness as a tool rather than a nuisance**. Stochastic gradient descent, dropout, variational inference, diffusion models, Monte Carlo estimators — these are algorithms whose correctness statements *are* probabilistic statements. To know when they work, and more importantly when they silently fail, you need the machinery.

**A word about the measure theory.** You chose the track that includes measure-theoretic foundations, and I want to set expectations honestly. Measure theory is not a hazing ritual, and it is not about pathological sets invented to annoy undergraduates. It is a language that makes certain questions *sayable* — questions like "my model reports a density, but a density relative to *what*?", "am I allowed to move this derivative inside this expectation?", and "why is my divergence stuck at a constant?" Without the language, these questions cannot even be posed, let alone answered.

But I will not pretend the material is easy. Some pages will need a second reading with paper and pen. What I *can* promise is that every page will be fair: no unexplained leaps, and a concrete example for every abstraction. Where I use a result without proving it, I will say so — and §1.14 collects every such borrowing in one place, so you always know exactly what you are taking on faith. If a passage ever feels effortless, be suspicious; I may have hidden something.

**How this chapter is built.** Five devices recur:

- **Worked Examples** — fully computed, with the reasoning shown. Read these actively: cover the solution and try the first step yourself.
- **Common Confusion** boxes — a wrong belief stated in a student's own voice, then dismantled. Misconceptions are sticky, and naming them works better than quietly explaining around them.
- **Figures** — rendered as plain-text diagrams so they survive any markdown viewer. Every figure has a caption that makes the surrounding text complete without it.
- **Self-check** questions at the end of each section, short enough to do in your head. Answers are collected in Appendix A.
- **Exercises** with full solutions at the end, in three flavours: theory, computational labs, and paper-style derivations.

Budget roughly a week per chapter for a serious pass.

**Notation** (consistent across all chapters):

| Symbol | Meaning |
|---|---|
| $\Omega$, $\omega$ | sample space; a single outcome |
| $\mathcal{F}$, $\mathcal{G}$ | $\sigma$-algebras (collections of events) |
| $\mathbb{P}$, $\mu$, $\nu$ | measures; $\mathbb{P}$ reserved for probability |
| $\lambda$, $\lambda_d$ | Lebesgue measure on $\mathbb{R}$, on $\mathbb{R}^d$ |
| $X$, $Y$, $Z$ | random variables and random vectors |
| $x$, $y$, $z$ | realized values |
| $P_X$ | the distribution (law) of $X$ |
| $\operatorname{supp}(P)$ | the support of $P$ |
| $\mathbb{1}_A$ | indicator function of the set $A$ |
| $A^c$, $A \sqcup B$ | complement; union of *disjoint* sets |
| $X \perp Y$ | $X$ and $Y$ are independent |
| $\otimes$ | product measure or product $\sigma$-algebra |
| $\mathcal{N}(\mu,\Sigma)$ | Gaussian with mean $\mu$, covariance $\Sigma$ |
| $\theta$ | model or distribution parameters |
| $:=$ | "is defined to be" |

**Roadmap.** Thirteen chapters, meant to be read in order.

1. **Probability spaces and random variables** — what probability *is*. *(this chapter)*
2. **Integration, expectation, and densities** — the Lebesgue integral, the convergence theorems, Radon–Nikodym, and what a density really is.
3. **Transformations and distribution families** — change of variables and Jacobians, the exponential family, sufficiency, normalizing flows.
4. **Inequalities and concentration** — Jensen, Markov, Chebyshev, Hoeffding, Bernstein, McDiarmid, sub-Gaussianity; why generalization bounds look the way they do.
5. **Multivariate and Gaussian theory** — covariance as linear algebra, the multivariate Gaussian, conditioning, Schur complements, Gaussian processes.
6. **Conditional expectation and martingales** — projection, filtrations, and the machinery behind SGD convergence proofs.
7. **Convergence and limit theorems** — the four modes of convergence, laws of large numbers, the CLT, the delta method, Slutsky.
8. **Point estimation** — MLE, MAP, bias–variance, Fisher information, Cramér–Rao, asymptotic normality.
9. **Hypothesis testing and confidence sets** — Neyman–Pearson, likelihood ratios, $p$-values and their misuse, multiple comparisons, the bootstrap.
10. **Bayesian inference** — priors, posteriors, conjugacy, predictive distributions, model comparison, regularization as a prior.
11. **Information theory** — entropy, cross-entropy, KL, mutual information, $f$-divergences.
12. **High-dimensional probability** — concentration of measure, random matrices, Johnson–Lindenstrauss.
13. **Monte Carlo, variational inference, and generative models** — importance sampling, MCMC, Langevin dynamics, the ELBO, VAEs, score matching, diffusion, calibration.

---
---

# Chapter 1

## 1.0 What this chapter is about, and why it takes so long to say

Here is the punchline in advance, so you know where we are going.

Your first probability course gave you a working setup: some random thing happens, and probabilities are numbers attached to outcomes. That setup is fine for computing the chance of three heads in five flips. It quietly breaks down the moment you want to do machine learning, and this chapter replaces it with a sturdier one built from three pieces:

$$\big(\underbrace{\Omega}_{\text{what could happen}}, \quad \underbrace{\mathcal{F}}_{\text{what you can ask}}, \quad \underbrace{\mathbb{P}}_{\text{how likely}}\big)$$

Almost all the conceptual work is in the middle piece, which is the one your first course never mentioned. That piece, $\mathcal{F}$, turns out to encode **information** — and once you see that, a surprising amount of machine learning theory becomes readable.

Then we define a random variable, and here the definition will feel backwards at first. A random variable is not a random number. It is an ordinary, deterministic **function**. All the randomness sits in $\Omega$; the random variable merely reads off a feature of whatever happened. This shift of perspective is what lets us say precisely what a generative model *is*.

Let me first convince you the sturdier setup is necessary.

---

## 1.1 Four things that break without this machinery

I want to earn your patience with four concrete failures. Each is a real thing that trips up real researchers, and none can be diagnosed with the tools from a first probability course. Two are resolved in this chapter and two need Chapter 2; I will say which is which, and §1.15 keeps score so you can see exactly what has been paid off.

### Failure 1 — Probabilities greater than one

You have been told that for a continuous random variable $X$ with density $p$, the probability that $X$ lands in $[a,b]$ is $\int_a^b p(x)\,dx$.

Fine. Now answer this: what is $\mathbb{P}(X = 0.5)$?

It is $\int_{0.5}^{0.5} p(x)\,dx = 0$. So *every* individual outcome has probability zero — yet when you run the experiment, some outcome occurs. Something has probability zero and happens anyway.

It gets stranger. A normalizing flow trained on images will happily report a log-likelihood of $+3.2$ for a test image. Positive log-likelihood means likelihood greater than one. How can a probability exceed one?

Both puzzles have the same resolution, and it is not a technicality: **$p(x)$ is not a probability at all.** It is a *rate* — probability per unit volume — and a rate is only meaningful relative to a chosen way of measuring volume. Change your units and every density value changes. This is why rescaling images from $[0,1]$ to $[0,255]$ shifts every reported log-likelihood by a fixed amount, and why "my model achieved higher likelihood" is meaningless until you state your preprocessing. *(Resolved in Chapter 2, via Radon–Nikodym.)*

### Failure 2 — A distribution that is neither discrete nor continuous

Take a single ReLU unit. Its input $Z$ is continuous — say $Z \sim \mathcal{N}(0,1)$ — and its output is $A = \max(0,Z)$.

Is $A$ discrete or continuous?

Neither. About half the time $Z$ is negative and $A$ is *exactly* zero, so $\mathbb{P}(A = 0) = 1/2$: a lump of probability on a single point, which continuous random variables may not have. The other half of the time $A$ spreads smoothly over $(0,\infty)$, taking uncountably many values, which discrete random variables may not do. $A$ has no probability mass function and no probability density function.

Your first course handed you two disjoint toolkits labelled "discrete" and "continuous," and the output of the most common activation function in deep learning belongs to neither. The same problem arises for zero-inflated count models, quantized latents in VQ-VAEs, spike-and-slab priors, and clipped rewards in reinforcement learning. Keep this example in mind throughout: it will return in §1.10 (where the CDF describes it perfectly), and again in §1.10.4 (where the *same* mechanism — a nonlinearity flattening a region to a point — turns out to create atoms in generative models too). *(Half-resolved in §1.10 of this chapter, using CDFs; completed in Chapter 2.)*

### Failure 3 — Moving a derivative inside an expectation

Nearly every gradient in modern machine learning has the shape

$$\nabla_\theta \, \mathbb{E}_{x \sim p_\theta}\big[f(x)\big].$$

Policy gradients, the VAE objective, GAN generator updates — all of this. And in every case we want to write

$$\nabla_\theta \, \mathbb{E}[f] \;\overset{?}{=}\; \mathbb{E}[\nabla_\theta f],$$

pushing the derivative through so we can estimate it from samples. Sometimes this is legitimate and sometimes it is not. When it is not, your estimator is *biased*, your training converges confidently to the wrong answer, and nothing in your loss curve tells you. *(Resolved in Chapter 2, via the Dominated Convergence Theorem.)*

### Failure 4 — A loss function that is constant

The original GAN objective is, up to constants, the Jensen–Shannon divergence between the data distribution and the generator's distribution. Natural images lie on or near a low-dimensional surface inside the enormous space of pixel arrays. A generator mapping a 100-dimensional latent into 3072-dimensional pixel space also produces a low-dimensional surface. Two low-dimensional surfaces in a high-dimensional space generically **do not intersect at all**.

Arjovsky and Bottou showed in 2017 what follows: when the two distributions live on non-overlapping sets, the Jensen–Shannon divergence equals $\log 2$ *exactly*, for every possible generator. A constant. Its gradient is zero. The generator receives no signal about which direction to move, no matter how badly wrong it is.

This is the vanishing-gradient problem that motivated Wasserstein GANs, and stating the diagnosis requires a concept called absolute continuity with no counterpart in calculus-based probability. *(Set up in §1.9 of this chapter, where we see exactly why the generator's distribution has zero volume; completed in Chapter 2.)*

> **The theme.** Every one of these failures is about the same thing: probability is a statement about the *sizes of sets*, and to reason about it you need a theory of size. That theory is measure theory. Let us build it.

---

## 1.2 The first ingredient: the sample space $\Omega$

We start with the easiest piece.

> **Definition 1.1.** The **sample space** $\Omega$ is the set of all possible outcomes of the random experiment. An element $\omega \in \Omega$ is a single **outcome**.

The important word is *all*: the outcomes must be mutually exclusive (exactly one occurs) and exhaustive (at least one occurs). And an outcome must be a *complete* description — everything random about the experiment is settled once you know $\omega$.

Here is the mental image I recommend. **Think of $\omega$ as the random seed.** When you write `rng = np.random.default_rng(42)`, you have fixed $\omega$, and from that moment every "random" quantity in your program is determined. Nature draws $\omega$ once, in secret, and everything else follows deterministically.

| Experiment | $\Omega$ | Size |
|---|---|---|
| One coin flip | $\{H,T\}$ | 2 |
| Three coin flips | $\{H,T\}^3$ | 8 |
| A die roll | $\{1,\ldots,6\}$ | 6 |
| Counting arrivals in an hour | $\{0,1,2,\ldots\}$ | countably infinite |
| A real-valued measurement | $\mathbb{R}$ | uncountable |
| Infinitely many coin flips | $\{H,T\}^{\mathbb{N}}$ | uncountable |
| A $32\times32$ RGB image | $[0,1]^{3072}$ | uncountable |

Notice how fast we leave the comfortable finite world. The last three rows are where all the difficulty of this chapter comes from, and you can already see why: once $\Omega$ is uncountable, "add up the probabilities of the individual outcomes" stops making sense.

**Choosing $\Omega$ is modelling, not mathematics.** For three flips I wrote $\Omega = \{H,T\}^3$, recording the full sequence. I could instead have taken $\Omega = \{0,1,2,3\}$, recording only the number of heads. Both are legitimate; they encode different decisions about what counts as an outcome. The first lets me ask "was the second flip heads?"; the second does not. That is not a defect — it is the point. **Your choice of $\Omega$ determines what questions are askable**, which brings us to the second ingredient.

> **Self-check 1.2**
> (a) Write down $\Omega$ for "roll two dice and record the sum."
> (b) With that $\Omega$, can you ask whether the first die showed a 3?
> (c) Give an $\Omega$ for which you can.

---

## 1.3 The second ingredient: events, and why we need $\sigma$-algebras

### 1.3.1 Events

Individual outcomes are rarely what we care about. We care about *events*.

> **Definition 1.2.** An **event** is a subset $A \subseteq \Omega$. We say $A$ **occurs** if the realized outcome $\omega$ lies in $A$.

The translation between English and set theory is worth writing out, because it is a common early stumbling block:

| English | Set theory |
|---|---|
| "$A$ occurs" | $\omega \in A$ |
| "$A$ does not occur" | $\omega \in A^c$ |
| "$A$ **or** $B$ occurs" | $\omega \in A\cup B$ |
| "$A$ **and** $B$ occur" | $\omega\in A\cap B$ |
| "$A$ and $B$ cannot both occur" | $A\cap B = \emptyset$ |
| "if $A$ occurs then $B$ occurs" | $A \subseteq B$ |
| "something happens" | $\omega\in\Omega$ |

**Worked Example 1.1.** Three coin flips, $\Omega = \{H,T\}^3$. Express these as subsets.

*"The first flip is heads."* $F_1 = \{HHH,HHT,HTH,HTT\}$ — four outcomes, since the last two flips are unconstrained.

*"At least two heads."* $\{HHH,HHT,HTH,THH\}$, four outcomes.

*"First flip heads **and** at least two heads overall."* Intersect: $\{HHH,HHT,HTH\}$.

*"Exactly one head."* $\{HTT,THT,TTH\}$. Note this is "at most one head" minus "no heads" — real questions get built by combining simpler ones, which is why we will want closure under the set operations.

### 1.3.2 The naive plan, and where it fails

We want to assign a probability to every event. The obvious plan: let the events be *all* subsets of $\Omega$, written $2^\Omega$, and assign a number to each.

For finite $\Omega$ this works perfectly. Three flips give $|2^\Omega| = 2^8 = 256$ events, and a fair-coin model assigns $\mathbb{P}(A) = |A|/8$. Done.

For $\Omega = \mathbb{R}$ or $[0,1]$, the plan is **provably impossible**. Not difficult — impossible. There is no way to assign a probability to every subset of $[0,1)$ while keeping the two properties any reasonable notion of "uniformly random" must have. I will prove this in §1.5. The consequence is what matters now:

> We must *choose* a collection of events to assign probabilities to, and it cannot be everything.

That collection is called a $\sigma$-algebra. Rather than think of it as a sad compromise, I want you to see it as carrying real content.

### 1.3.3 The definition

> **Definition 1.3 ($\sigma$-algebra).** A collection $\mathcal{F}$ of subsets of $\Omega$ is a **$\sigma$-algebra** if
> 1. $\Omega \in \mathcal{F}$;
> 2. *(closed under complement)* $A \in \mathcal{F} \implies A^c \in \mathcal{F}$;
> 3. *(closed under countable union)* $A_1,A_2,\ldots \in \mathcal{F} \implies \bigcup_{n=1}^\infty A_n \in \mathcal{F}$.
>
> Then $(\Omega,\mathcal{F})$ is a **measurable space**, and members of $\mathcal{F}$ are **measurable sets**, or **events**.

Three one-line consequences:

- $\emptyset = \Omega^c \in \mathcal{F}$, by (1) and (2).
- Closed under countable intersection: $\bigcap_n A_n = \big(\bigcup_n A_n^c\big)^c$, so apply (2), (3), (2).
- Closed under set difference: $A\setminus B = A \cap B^c$.

So a $\sigma$-algebra is closed under everything you would want to do to events. Axiom (3) covers finite unions too: to union just $A_1,A_2$, take $A_3 = A_4 = \cdots = \emptyset$.

### 1.3.4 The idea that makes it click: a $\sigma$-algebra is information

Here is the single most useful way to read Definition 1.3, and the one that pays off throughout machine learning.

> **A $\sigma$-algebra is the collection of yes/no questions you are able to answer.**

If $A \in \mathcal{F}$, you can determine whether the outcome landed in $A$. If $A\notin\mathcal{F}$, that question is beyond your reach — not because probability is mystically undefined, but because you lack the information to answer it.

Now re-read the axioms. They are not arbitrary; they are *forced*:

1. You can always answer "did the experiment happen at all?" So $\Omega\in\mathcal{F}$.
2. If you can determine whether $A$ happened, you automatically know whether it *didn't*. Same information, opposite answer. So $A^c\in\mathcal{F}$.
3. If you can answer "$A_1$?", "$A_2$?", … then you can answer "did at least one occur?" — check them one at a time. So $\bigcup_n A_n \in \mathcal{F}$.

A **small** $\sigma$-algebra means little information; a **large** one means a lot. Two extremes: $\mathcal{F} = \{\emptyset,\Omega\}$, the **trivial** $\sigma$-algebra, where you know nothing beyond "something happened"; and $\mathcal{F} = 2^\Omega$, where you know everything.

**Worked Example 1.2 (information from a partial report).** Roll a die, $\Omega = \{1,\ldots,6\}$. An assistant rolls it and tells you only the **parity**. What can you determine?

You can answer "was it odd?" ($\{1,3,5\}$) and "was it even?" ($\{2,4,6\}$), plus the two trivial questions. Nothing else. So

$$\mathcal{F}_{\text{par}} = \big\{\emptyset,\ \{1,3,5\},\ \{2,4,6\},\ \Omega\big\}.$$

Let us *verify* rather than assume. (1) $\Omega$ is present ✓. (2) $\emptyset^c = \Omega$ ✓, $\{1,3,5\}^c = \{2,4,6\}$ ✓, and symmetrically ✓. (3) $\{1,3,5\}\cup\{2,4,6\} = \Omega$ ✓; unions involving $\emptyset$ or $\Omega$ stay inside ✓.

Now notice what is *missing*: $\{6\}\notin\mathcal{F}_{\text{par}}$. You cannot answer "was it a six?" — correctly, because parity does not tell you. The $\sigma$-algebra is not an approximation to your information. It *is* your information.

### 1.3.5 On finite sets, a $\sigma$-algebra is exactly a partition

This is the anchor that makes everything concrete.

Look at $\mathcal{F}_{\text{par}}$ again. Its two smallest non-empty members, $\{1,3,5\}$ and $\{2,4,6\}$, are disjoint and cover $\Omega$: a **partition**. And every set in $\mathcal{F}_{\text{par}}$ is a union of partition blocks. This is completely general.

> **Proposition 1.4.** Let $\Omega$ be finite. Then $\sigma$-algebras on $\Omega$ correspond exactly to partitions of $\Omega$. Given a $\sigma$-algebra $\mathcal{F}$, its **atoms** — the smallest non-empty members — partition $\Omega$, and $\mathcal{F}$ consists precisely of all unions of atoms. Conversely, all unions of blocks of any partition form a $\sigma$-algebra.

*Proof.* Given $\mathcal{F}$, define for each $\omega\in\Omega$ its atom $A(\omega) := \bigcap\{A\in\mathcal{F} : \omega\in A\}$ — the smallest event containing $\omega$, an event itself because $\Omega$ is finite, so this is a finite intersection. Two atoms are either equal or disjoint: if $\omega'\in A(\omega)$ then every event containing $\omega'$ but not $\omega$ would, by complementation, put $\omega$ in an event excluding $\omega'$, shrinking $A(\omega)$ below itself — impossible — so $A(\omega') = A(\omega)$. Hence the distinct atoms partition $\Omega$. Every $A\in\mathcal{F}$ is the union of the atoms of its points (each $A(\omega)\subseteq A$ for $\omega\in A$, since $A$ participates in the intersection). Conversely, unions of blocks of a partition are trivially closed under complement and union. ∎

A useful count falls out: **$k$ atoms give exactly $2^k$ events**, one for each subset of atoms. This gives you a sanity check on any finite $\sigma$-algebra you write down.

```
Figure 1.1 — Three σ-algebras on Ω = {1,...,6}, drawn as partitions

  Trivial F = {∅, Ω}          Parity F_par                Full F = 2^Ω
  ┌──────────────────┐        ┌────────┬─────────┐        ┌──┬──┬──┬──┬──┬──┐
  │  1  2  3  4  5  6│        │ 1 3 5  │  2 4 6  │        │1 │2 │3 │4 │5 │6 │
  └──────────────────┘        └────────┴─────────┘        └──┴──┴──┴──┴──┴──┘
  1 atom  · |F| = 2           2 atoms · |F| = 4           6 atoms · |F| = 64

  Coarser partition  ←——————— information ———————→  Finer partition

  An event is measurable exactly when it is a union of whole atoms.
  "Was it a 6?" is unanswerable in the middle picture — it would split
  the atom {2,4,6}.
```

**Worked Example 1.3 (a filtration: information accumulating over time).** Flip a coin three times, $\Omega = \{H,T\}^3$, watching one flip at a time.

*After flip 1.* You know the first coordinate. Atoms: $\{H\!*\!*\}$ and $\{T\!*\!*\}$, each of size 4. So
$$\mathcal{F}_1 = \big\{\emptyset,\ \{HHH,HHT,HTH,HTT\},\ \{THH,THT,TTH,TTT\},\ \Omega\big\},\qquad |\mathcal{F}_1| = 2^2 = 4.$$

*After flip 2.* Four atoms — $HH\!*$, $HT\!*$, $TH\!*$, $TT\!*$ — each of size 2, so $|\mathcal{F}_2| = 2^4 = 16$.

*After flip 3.* Eight singleton atoms, $|\mathcal{F}_3| = 2^8 = 256$. You know everything.

And crucially $\mathcal{F}_1 \subsetneq \mathcal{F}_2 \subsetneq \mathcal{F}_3$. Information only accumulates. An increasing sequence of $\sigma$-algebras is a **filtration**, the standard formalism for "what is known at time $t$."

```
Figure 1.2 — A filtration: the partition of Ω = {H,T}³ refines flip by flip

  After flip 1 (F₁)      After flip 2 (F₂)      After flip 3 (F₃)
  ┌───────────────┐      ┌───────┬───────┐      ┌───┬───┬───┬───┐
  │ HHH HHT HTH HTT│  →  │HHH HHT│HTH HTT│  →   │HHH│HHT│HTH│HTT│
  ├───────────────┤      ├───────┼───────┤      ├───┼───┼───┼───┤
  │ THH THT TTH TTT│     │THH THT│TTH TTT│      │THH│THT│TTH│TTT│
  └───────────────┘      └───────┴───────┘      └───┴───┴───┴───┘
   2 atoms                4 atoms                8 atoms (singletons)

  Each atom splits when a flip is revealed; nothing ever merges — you
  cannot forget. This nesting F₁ ⊂ F₂ ⊂ F₃ is a filtration. In SGD
  analysis, F_t is "everything seen before step t," and "the stochastic
  gradient is unbiased" is the statement E[g_t | F_t] = ∇F(θ_t).
```

**Why an ML researcher should care.** Three places where this is load-bearing.

*Conditional expectation.* Chapter 6 defines $\mathbb{E}[X\mid\mathcal{G}]$ as "the best prediction of $X$ using only the information in $\mathcal{G}$," and it is an orthogonal projection. On a finite space, with $\mathcal{G}$ given by a partition, $\mathbb{E}[X\mid\mathcal{G}]$ is simply the function that averages $X$ over each atom — a picture you can hold in your head, and it generalizes.

*Stochastic gradient descent.* Convergence proofs for SGD, Adam, and temporal-difference learning are martingale arguments over a filtration $\mathcal{F}_1\subseteq\mathcal{F}_2\subseteq\cdots$. The core assumption "the minibatch gradient is unbiased" is $\mathbb{E}[g_t\mid\mathcal{F}_t] = \nabla F(\theta_t)$: *given everything I know at step $t$, the noise averages out.* You cannot state this without conditioning on a $\sigma$-algebra. Exercise C1 makes it precise.

*Fairness and privacy.* "The prediction must not depend on the protected attribute" is a measurability statement: the predictor must be measurable with respect to the $\sigma$-algebra generated by the permitted features.

> **Common Confusion 1.1**
> *"If an event isn't in $\mathcal{F}$, doesn't that mean it can't happen?"*
>
> No. In Worked Example 1.2 the outcome $6$ can certainly happen — the die can land on six. What fails is your *ability to detect* it from the information you were given. Membership in $\mathcal{F}$ is about **detectability**, not possibility.
>
> A useful test: "can happen" is a statement about $\Omega$ (is the outcome in there?); "is measurable" is a statement about $\mathcal{F}$ (can I check?). Keeping these apart will save you real confusion at conditional expectation.

> **Common Confusion 1.2**
> *"Why countable unions? Why not allow arbitrary unions and be done with it?"*
>
> Because arbitrary unions would destroy the enterprise. Every subset of $\mathbb{R}$ is the union of the singletons it contains — an *uncountable* union. So if $\sigma$-algebras were closed under arbitrary unions, any $\sigma$-algebra on $\mathbb{R}$ containing the singletons would automatically be all of $2^{\mathbb{R}}$, which §1.5 shows is unusable.
>
> Countable is exactly the right amount: enough to take limits and do analysis, not enough to reconstruct arbitrary sets from points. A useful heuristic: countable operations are the ones you could actually *carry out*, one step at a time.
>
> This is not a purely abstract point. §1.8.3 shows a genuine machine-learning consequence: the supremum of an *uncountable* family of measurable functions can fail to be measurable, and every generalization bound is a statement about such a supremum.

> **Self-check 1.3**
> (a) Is $\mathcal{A} = \{\emptyset, \{1\}, \{2,3\}, \{1,2,3\}\}$ a $\sigma$-algebra on $\Omega=\{1,2,3\}$? Check each axiom explicitly rather than eyeballing it.
> (b) Is $\mathcal{B} = \{\emptyset, \{1\},\{2\},\{1,2,3\}\}$ a $\sigma$-algebra on the same $\Omega$?
> (c) On $\Omega = \{1,2,3,4\}$, how many events are in the $\sigma$-algebra with atoms $\{1,2\}$, $\{3\}$, $\{4\}$?
> (d) In Worked Example 1.3, is "exactly two heads" in $\mathcal{F}_2$?

---

## 1.4 Building $\sigma$-algebras: generation and the Borel sets

In practice we never write out a $\sigma$-algebra by hand. We say which questions we *care* about and let the axioms supply the rest.

> **Definition 1.5 (generated $\sigma$-algebra).** For any collection $\mathcal{C}$ of subsets of $\Omega$,
> $$\sigma(\mathcal{C}) := \bigcap\{\mathcal{G} : \mathcal{G}\text{ a } \sigma\text{-algebra}, \ \mathcal{C}\subseteq\mathcal{G}\},$$
> the **smallest** $\sigma$-algebra containing $\mathcal{C}$, called the $\sigma$-algebra **generated by** $\mathcal{C}$.

Two things make this well posed, both easy. There is at least one $\sigma$-algebra containing $\mathcal{C}$, namely $2^\Omega$, so the family being intersected is non-empty. And an arbitrary intersection of $\sigma$-algebras is a $\sigma$-algebra (Exercise A2) — because each axiom has the form "if these sets are in, then that set is in," and such conditions survive intersection.

Read $\sigma(\mathcal{C})$ as: **the information you get from being told the answers to all the questions in $\mathcal{C}$, together with everything logically deducible from them.**

**Worked Example 1.4 (generating, step by step).** On $\Omega = \{1,\ldots,6\}$, compute $\sigma(\{\{1\},\{2\}\})$.

Work outward from the generators, closing under the operations until nothing new appears.

- Start: $\{1\},\{2\}$.
- Union: $\{1,2\}$.
- Complements: $\{2,3,4,5,6\}$, $\{1,3,4,5,6\}$, $\{3,4,5,6\}$.
- Add $\emptyset,\Omega$.

Anything new? Intersections give sets already listed — e.g. $\{2,3,4,5,6\}\cap\{1,3,4,5,6\} = \{3,4,5,6\}$ ✓. So
$$\sigma(\{\{1\},\{2\}\}) = \big\{\emptyset,\{1\},\{2\},\{1,2\},\{3,4,5,6\},\{1,3,4,5,6\},\{2,3,4,5,6\},\Omega\big\}.$$

Sanity check via Proposition 1.4: the atoms should be $\{1\},\{2\},\{3,4,5,6\}$ — three atoms, so $2^3 = 8$ events. We listed 8. ✓

Interpretation: you can identify a 1, identify a 2, and otherwise only report "something in $\{3,4,5,6\}$." Exactly the information content of the two questions you were handed.

### 1.4.1 The Borel $\sigma$-algebra

Now the case we actually need. On $\mathbb{R}$, which questions do we care about? At minimum, "is $X$ in this interval?" So we generate from the open sets.

> **Definition 1.6.** The **Borel $\sigma$-algebra** on $\mathbb{R}$ is $\mathcal{B}(\mathbb{R}) := \sigma(\{\text{open subsets of }\mathbb{R}\})$, and its members are **Borel sets**. Similarly $\mathcal{B}(\mathbb{R}^d)$.
>
> **Convention: whenever we write $\mathbb{R}^d$ as a measurable space, we mean $(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))$.** Assumed silently hereafter.

**Worked Example 1.5 (what's in there?).**

*Closed sets:* complements of open sets ✓.
*Single points:* $\{a\} = \bigcap_{n\ge1}\big(a-\tfrac1n,\ a+\tfrac1n\big)$, a countable intersection of open intervals ✓.
*Half-open intervals:* $(a,b] = \bigcap_n \big(a,\ b+\tfrac1n\big)$ ✓.
*The rationals:* $\mathbb{Q} = \bigcup_{q\in\mathbb{Q}}\{q\}$, a countable union of points ✓ — hence also the irrationals, by complement ✓.
*The integers, the Cantor set, the set of normal numbers:* all Borel.

The pattern: **anything built from intervals using countably many unions, intersections, and complements is Borel** — which in practice covers every set that arises in this book.

Two refinements of that statement, one reassuring and one cautionary.

*The reassuring one.* $\mathcal{B}(\mathbb{R})$ is generated by a *countable* family — the rational rays $(-\infty,q]$ suffice (Exercise A3). A cardinality count then gives $|\mathcal{B}(\mathbb{R})| = 2^{\aleph_0}$, while $|2^{\mathbb{R}}| = 2^{2^{\aleph_0}}$. So **almost every subset of $\mathbb{R}$ fails to be Borel** — and yet no explicit finite formula built from intervals and countable operations will ever produce one, because such a construction is itself a countable act. Non-measurable sets are simultaneously ubiquitous and unreachable by direct construction. (The cardinality claim is one of the results listed in §1.14.)

*The cautionary one.* Do not upgrade this to "every set you can describe is Borel." Perfectly natural-looking descriptions can escape: the *projection* of a Borel subset of $\mathbb{R}^2$ onto the $x$-axis — that is, a set of the form $\{x : \exists t,\ (x,t)\in B\}$ — need not be Borel. (Such sets are called *analytic*; they are still "almost measurable" in a sense made precise in descriptive set theory.) Notice the shape of the culprit: an **existential quantifier over a continuum**. Keep that shape in mind — it is exactly the shape of a supremum over an uncountable family, and §1.8.3 shows the same mechanism biting in learning theory. Definable does not imply Borel; *countably constructible* does.

> **Common Confusion 1.3**
> *"Is a $\sigma$-algebra just a partition, like Proposition 1.4 says?"*
>
> Only on finite (and, with care, countable) sample spaces. On $\mathbb{R}$ the analogy breaks, instructively. In $\mathcal{B}(\mathbb{R})$ every singleton $\{a\}$ is measurable, so the "atoms" would have to be single points — but $\mathcal{B}(\mathbb{R})$ is *not* all unions of points, since that would be $2^{\mathbb{R}}$. Uncountable unions of atoms escape the $\sigma$-algebra.
>
> Keep the partition picture as **intuition for what information means**, and Definition 1.3 as the actual definition. For finite or discrete cases, draw the partition; on $\mathbb{R}$, trust the axioms.

> **Self-check 1.4**
> (a) Show $[a,b)$ is Borel.
> (b) Is the set of irrationals in $[0,1]$ Borel?
> (c) True or false: $\sigma(\mathcal{C})$ is the set of all countable unions and complements of members of $\mathcal{C}$. (Careful.)

---

## 1.5 ★ Why we cannot use all subsets: the Vitali construction

*This section justifies the whole framework. Nothing later depends on it, so you may skip it on a first reading — but it is only a page.*

Suppose we want a "uniform distribution on $[0,1)$" assigning a probability to **every** subset. Any honest notion of "uniform" must have:

- **(A) Countable additivity.** The probability of a countable disjoint union is the sum of the probabilities.
- **(B) Translation invariance.** Sliding a set around the circle does not change its probability. Formally, work modulo 1: with $A \oplus t := \{(a+t)\bmod 1 : a\in A\}$, require $\mathbb{P}(A\oplus t) = \mathbb{P}(A)$.

Property (B) is what "uniform" *means*: no location is favoured. Watch these two assumptions destroy each other.

**The construction.** Declare $x\sim y$ when $x - y\in\mathbb{Q}$. This is an equivalence relation on $[0,1)$, partitioning it into classes of the form $(x+\mathbb{Q})\cap[0,1)$ — each countable and dense. There are uncountably many classes (each is countable, and together they cover an uncountable set). Using the Axiom of Choice, pick exactly one representative from each class; call the resulting set $V$.

**Claim 1: the rational translates of $V$ are pairwise disjoint.** Let $q\ne q'$ be rationals in $[0,1)$ and suppose $z\in(V\oplus q)\cap(V\oplus q')$. Then $z = v\oplus q = v'\oplus q'$ for some $v,v'\in V$, so $v - v'\in\mathbb{Q}$, meaning $v,v'$ lie in the same class. But $V$ has one representative per class, so $v = v'$ — and then $q = q'$, contradiction. ✓

**Claim 2: the rational translates cover $[0,1)$.** Take $x\in[0,1)$. It lies in some class, whose chosen representative is some $v\in V$. Then $x - v\in\mathbb{Q}$, and with $q := (x-v)\bmod 1$ we get $x = v\oplus q\in V\oplus q$. ✓

**The contradiction.** List $\mathbb{Q}\cap[0,1) = \{q_1,q_2,\ldots\}$, countable. By Claims 1 and 2,
$$[0,1) = \bigsqcup_{n=1}^\infty (V\oplus q_n),$$
a *countable disjoint* union. If $V$ had probability $\mathbb{P}(V) = p$, then by (A) and (B),
$$1 = \mathbb{P}([0,1)) = \sum_{n=1}^\infty \mathbb{P}(V\oplus q_n) = \sum_{n=1}^\infty p.$$
If $p = 0$ the sum is $0\ne1$. If $p>0$ the sum is $+\infty\ne1$. No third option.

**Conclusion: no such $p$ exists.** Any uniform probability on $[0,1)$ must leave some sets unmeasured.

**What to take from this.** Restricting to a $\sigma$-algebra is not fussiness. Assigning probabilities to all subsets of $[0,1)$ is *impossible*, and a $\sigma$-algebra is the precise statement of which sets *can* be measured. Note also that the Axiom of Choice did all the work: $V$ has no formula or algorithm — which is exactly why you will never trip over one by accident.

---

## 1.6 The third ingredient: measures and probability

Now we can say how big a set is.

> **Definition 1.7 (measure).** Let $(\Omega,\mathcal{F})$ be a measurable space. A function $\mu:\mathcal{F}\to[0,\infty]$ is a **measure** if
> 1. $\mu(\emptyset) = 0$;
> 2. *(countable additivity)* for **pairwise disjoint** $A_1,A_2,\ldots\in\mathcal{F}$,
> $$\mu\Big(\bigsqcup_{n=1}^\infty A_n\Big) = \sum_{n=1}^\infty \mu(A_n).$$
>
> Then $(\Omega,\mathcal{F},\mu)$ is a **measure space**. If $\mu(\Omega)=1$ we call $\mu$ a **probability measure**, write it $\mathbb{P}$, and call $(\Omega,\mathcal{F},\mathbb{P})$ a **probability space**.

A measure is an abstract notion of **size**: length, area, volume, mass, count, or probability, all at once. Countable additivity says the size of a whole equals the sum of the sizes of disjoint pieces, *even for infinitely many pieces*.

**Three measures to keep in mind.**

*Lebesgue measure $\lambda_d$ on $(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))$.* Ordinary length, area, volume: $\lambda_1([a,b]) = b-a$. Precisely: **$\lambda_d$ is the unique measure on $\mathcal{B}(\mathbb{R}^d)$ that is translation invariant and satisfies $\lambda_d([0,1]^d) = 1$.** Existence and uniqueness are a substantial theorem (§1.14). A remark on hypotheses, because textbooks vary: some state uniqueness with an additional assumption of *local finiteness* (finite measure on compact sets). That assumption costs nothing — but it is not logically needed, because it is *implied* by the other two: any compact set is covered by finitely many translates of $[0,1)^d$, and each translate has measure at most $\mu([0,1]^d) = 1$ by monotonicity and translation invariance, so compact sets automatically get finite measure. Note $\lambda_d(\mathbb{R}^d)=\infty$, so Lebesgue measure is *not* a probability measure, but it is the reference against which continuous distributions are measured (Chapter 2). Useful facts: $\lambda_d$ of a single point is $0$, hence any countable set has $\lambda_d = 0$, being a countable union of points.

*Counting measure $\#$ on a countable set $S$ with $\mathcal{F} = 2^S$.* $\#(A) = |A|$. The reference for discrete distributions.

*Dirac measure $\delta_{x_0}$.* $\delta_{x_0}(A) = \mathbb{1}[x_0\in A]$ — a unit point mass. Your training set is made of these: the **empirical distribution**
$$\hat P_n := \frac1n\sum_{i=1}^n \delta_{x_i}$$
is a probability measure putting mass $1/n$ on each data point. Every generalization bound is, at heart, a statement about how far $\hat P_n$ is from the true $P$.

**A terminological caution: "Borel" versus "Lebesgue" measurable.** This book works with the Borel $\sigma$-algebra throughout. When you open Durrett or Billingsley you will meet a slightly larger $\sigma$-algebra, the *Lebesgue* $\sigma$-algebra, obtained from $\mathcal{B}(\mathbb{R}^d)$ by a procedure called **completion**: every subset of a measure-zero Borel set is declared measurable (with measure zero). The Lebesgue $\sigma$-algebra strictly contains the Borel one, and "Lebesgue measurable but not Borel" sets exist. For everything in this tutorial the distinction is invisible — probabilities never change — but knowing the two names refer to different collections will save you confusion when reading other sources. When we say "Lebesgue measure" we mean $\lambda_d$ restricted to Borel sets.

### 1.6.1 Properties you will use constantly

> **Proposition 1.8.** Let $\mu$ be a measure on $(\Omega,\mathcal{F})$ and $A,B,A_n\in\mathcal{F}$.
> 1. *(monotonicity)* $A\subseteq B \implies \mu(A)\le\mu(B)$.
> 2. *(countable subadditivity, the union bound)* $\mu\big(\bigcup_n A_n\big)\le\sum_n\mu(A_n)$ — **no disjointness required**.
> 3. *(continuity from below)* $A_1\subseteq A_2\subseteq\cdots \implies \mu\big(\bigcup_n A_n\big) = \lim_n\mu(A_n)$.
> 4. *(continuity from above)* $A_1\supseteq A_2\supseteq\cdots$ **and** $\mu(A_1)<\infty \implies \mu\big(\bigcap_n A_n\big) = \lim_n\mu(A_n)$.
> 5. *(complement, for probability measures)* $\mathbb{P}(A^c) = 1-\mathbb{P}(A)$.
> 6. *(inclusion–exclusion)* $\mathbb{P}(A\cup B) = \mathbb{P}(A)+\mathbb{P}(B)-\mathbb{P}(A\cap B)$.

**Proofs.** The technique throughout is *disjointification*: turn a general union into a disjoint one so countable additivity applies. Learn this move.

*(1)* $B = A\sqcup(B\setminus A)$, both in $\mathcal{F}$ and disjoint, so $\mu(B) = \mu(A)+\mu(B\setminus A)\ge\mu(A)$ by non-negativity. ∎

*(3)* Set $B_1 := A_1$ and $B_n := A_n\setminus A_{n-1}$. The $B_n$ are disjoint (each excludes everything in $A_{n-1}$, which contains all earlier $B_k$), with $\bigsqcup_{k\le n}B_k = A_n$ and $\bigsqcup_{k\ge1}B_k = \bigcup_n A_n$. So
$$\mu\Big(\bigcup_n A_n\Big) = \sum_{k=1}^\infty\mu(B_k) = \lim_{n}\sum_{k\le n}\mu(B_k) = \lim_n\mu(A_n),$$
using that an infinite sum *is* the limit of partial sums. ∎

*(2)* Same disjointification: $B_n := A_n\setminus(A_1\cup\cdots\cup A_{n-1})$ are disjoint with the same union and $B_n\subseteq A_n$, so $\mu(\bigcup_n A_n) = \sum_n\mu(B_n)\le\sum_n\mu(A_n)$ by monotonicity termwise. ∎

*(4)* Apply (3) to the increasing sets $A_1\setminus A_n$, whose union is $A_1\setminus\bigcap_n A_n$, then subtract from $\mu(A_1)$. Subtraction needs $\mu(A_1)<\infty$. ∎

*(5)* $\Omega = A\sqcup A^c$. *(6)* $A\cup B = A\sqcup(B\setminus A)$ and $B = (A\cap B)\sqcup(B\setminus A)$; subtract. ∎

**Why the finiteness hypothesis in (4) is real.** Take $\mu = \lambda$ on $\mathbb{R}$ and $A_n = [n,\infty)$. These decrease with $\bigcap_n A_n = \emptyset$, so $\lambda(\bigcap_n A_n) = 0$; but $\lambda(A_n) = \infty$ for every $n$. Continuity from above fails. Probability measures are finite, so we never worry — but for general measures, check.

### 1.6.2 Why *countable* additivity is the right axiom

Merely saying finite additivity is "too weak to support limiting arguments" would be hand-waving. Here is the precise statement, and it is three lines.

> **Proposition 1.9.** Let $\mu:\mathcal{F}\to[0,\infty]$ satisfy $\mu(\emptyset)=0$ and be **finitely** additive. Then $\mu$ is countably additive **if and only if** $\mu$ is continuous from below: whenever $A_1\subseteq A_2\subseteq\cdots$ with $A = \bigcup_n A_n$, we have $\mu(A_n)\to\mu(A)$.

*Proof.* ($\Rightarrow$) This is Proposition 1.8(3), just proved. ($\Leftarrow$) Let $A_1,A_2,\ldots$ be disjoint with $S := \bigsqcup_n A_n$, and set $T_n := \bigsqcup_{k\le n}A_k$. The $T_n$ increase to $S$. Finite additivity gives $\mu(T_n) = \sum_{k\le n}\mu(A_k)$, and continuity from below gives
$$\mu(S) = \lim_{n\to\infty}\mu(T_n) = \lim_{n\to\infty}\sum_{k\le n}\mu(A_k) = \sum_{k=1}^\infty \mu(A_k). \qquad \blacksquare$$

So **countable additivity is exactly the assumption that measures behave continuously under increasing limits.** That is why every asymptotic theorem in statistics needs it: the law of large numbers, the CLT, consistency of maximum likelihood, and convergence of SGD are all statements about limits, and without Proposition 1.9 you cannot pass a limit through a probability.

*A concrete illustration of the gap.* Consider trying to define a "uniform distribution on $\mathbb{N}$" by natural density, $d(A) := \lim_{n\to\infty}\frac{|A\cap\{1,\ldots,n\}|}{n}$, on the collection of sets where that limit exists. Then $d$ is finitely additive where defined, $d(\mathbb{N})=1$, and $d(\{k\}) = 0$ for every single $k$. If $d$ were countably additive we would get $1 = d(\mathbb{N}) = \sum_k d(\{k\}) = 0$. It is not, and indeed there is no countably additive uniform probability measure on $\mathbb{N}$ — a fact worth knowing, since "pick a random integer" is a phrase people use casually. (Caveat: $d$ is not defined on all subsets, so it is not a finitely additive measure on $2^{\mathbb{N}}$; it illustrates the phenomenon rather than being a formal counterexample.)

### 1.6.3 The union bound is the most important inequality in learning theory

Property (2) looks like the least interesting item on the list. It is the most useful.

Suppose you have a hypothesis class $\mathcal{H}$ with $m$ candidates, and for each single $h$ you can prove "the chance that $h$'s test error deviates from its training error by more than $\varepsilon$ is at most $\delta_0$." You want a guarantee holding *simultaneously* for all $m$ — because you will *pick* one after looking at the data, and you do not know in advance which.

Let $A_h$ be the bad event for $h$. You want $\mathbb{P}(\bigcup_h A_h)$. These events are neither disjoint nor independent, and you know nothing about their dependence. The union bound needs nothing:
$$\mathbb{P}\Big(\bigcup_{h\in\mathcal{H}}A_h\Big)\le\sum_{h}\mathbb{P}(A_h) = m\delta_0.$$
So for overall confidence $\delta$, take $\delta_0 = \delta/m$. In Chapter 4 this produces the characteristic $\sqrt{\log m}$ factor in every uniform-convergence bound. **That $\log m$ is the price of the union bound**, and it is why bounds worsen as model classes grow.

**Worked Example 1.6 (how loose is it?).** The union bound against the truth in three dependence regimes, $m$ events each of probability $p$:

| $m$, $p$ | union bound $mp$ | if independent | if all identical | if disjoint |
|---|---|---|---|---|
| $m=10$, $p=0.01$ | 0.1000 | 0.0956 | 0.01 | 0.1000 |
| $m=100$, $p=0.01$ | 1.0000 (vacuous) | 0.6340 | 0.01 | 1.0000 |
| $m=1000$, $p=0.001$ | 1.0000 (vacuous) | 0.6323 | 0.001 | 1.0000 |

(For independent events the exact value is $1-(1-p)^m$; check the $m=10$ entry: $1-0.99^{10} = 0.0956$ ✓.)

Four lessons.

*It is tight.* Disjoint events make it an equality, so no better general bound exists.
*It is nearly free for rare events.* At $m=10,p=0.01$: $0.1$ versus a truth of $0.0956$, a 5% overshoot.
*It goes vacuous once $mp\ge1$.* At $m=100$ it says "at most 1" while the truth is $0.63$. This is where naive union-bound theory stops saying anything about large classes — hence chaining, covering numbers, and Rademacher complexity in Chapter 4.
*It can be enormously loose.* When all $m$ events coincide, the truth is $p$ and the bound is $mp$ — off by a factor of $m$. Dependence is what the union bound discards.

> **Common Confusion 1.4**
> *"Doesn't $\mathbb{P}(A)=0$ mean $A$ is impossible?"*
>
> No. Draw $U$ uniformly from $[0,1]$: then $\mathbb{P}(U = 0.3) = 0$, since $\lambda(\{0.3\})=0$. Yet *some* number gets drawn, and whatever it is, it had probability zero.
>
> Read $\mathbb{P}(A)=0$ as "**negligible**," not "impossible." Flip a coin infinitely often: every specific infinite sequence has probability zero, and one occurs. §1.13 gives this its own vocabulary. The converse trap is equally common: $\mathbb{P}(A)=1$ does not mean $A$ is *certain*, only that its failure is negligible.

> **Self-check 1.6**
> (a) If $\mathbb{P}(A)=0.7$ and $\mathbb{P}(B)=0.6$, what is the smallest possible $\mathbb{P}(A\cap B)$?
> (b) A measure on $\{1,2,3\}$ has $\mu(\{1\})=2$, $\mu(\{2\})=0$, $\mu(\{3\})=5$. What is $\mu(\{1,3\})$? Is $\mu$ a probability measure?
> (c) You have 40 hypotheses and want overall confidence 95%. What per-hypothesis confidence does the union bound require?

---

## 1.7 Conditional probability and Bayes' rule

Everything so far has been about a *fixed* state of information. But the central activity in both statistics and machine learning is **updating**: you learn something, and your probabilities change. Here is the elementary version of that operation. (The general version — conditioning on a $\sigma$-algebra rather than an event — is Chapter 6, and §1.7.4 explains why we need it.)

> **Definition 1.10.** Let $B\in\mathcal{F}$ with $\mathbb{P}(B)>0$. For $A\in\mathcal{F}$, the **conditional probability of $A$ given $B$** is
> $$\mathbb{P}(A\mid B) := \frac{\mathbb{P}(A\cap B)}{\mathbb{P}(B)}.$$

The intuition: learning that $B$ occurred means discarding all outcomes outside $B$ and **renormalizing** what remains so the total is 1 again. The numerator asks "how much of $A$ survived?" and the denominator restores the total.

That description suggests conditioning produces a genuinely new probability space, and it does.

> **Proposition 1.11.** Fix $B$ with $\mathbb{P}(B)>0$ and define $\mathbb{P}_B(A) := \mathbb{P}(A\mid B)$. Then $\mathbb{P}_B$ is a probability measure on $(\Omega,\mathcal{F})$.

*Proof.* Non-negativity is clear, and $\mathbb{P}_B(\Omega) = \mathbb{P}(\Omega\cap B)/\mathbb{P}(B) = 1$ ✓. For $\mathbb{P}_B(\emptyset) = 0$: the numerator is $\mathbb{P}(\emptyset)=0$ ✓. For countable additivity, let $A_1,A_2,\ldots$ be disjoint. Then $A_n\cap B$ are also disjoint, with $\big(\bigsqcup_n A_n\big)\cap B = \bigsqcup_n (A_n\cap B)$, so
$$\mathbb{P}_B\Big(\bigsqcup_n A_n\Big) = \frac{\mathbb{P}\big(\bigsqcup_n(A_n\cap B)\big)}{\mathbb{P}(B)} = \frac{\sum_n\mathbb{P}(A_n\cap B)}{\mathbb{P}(B)} = \sum_n\mathbb{P}_B(A_n),$$
using countable additivity of $\mathbb{P}$ and the fact that dividing a convergent series by a positive constant is legal. ∎

This matters more than it looks. It means **every theorem you prove about probability measures applies verbatim to conditional probability.** The union bound holds conditionally, monotonicity holds conditionally, and so on — you never have to re-derive anything.

### 1.7.1 The chain rule and the law of total probability

Rearranging Definition 1.10 gives the **multiplication rule** $\mathbb{P}(A\cap B) = \mathbb{P}(A\mid B)\mathbb{P}(B)$, and iterating gives the **chain rule**:
$$\mathbb{P}(A_1\cap\cdots\cap A_n) = \mathbb{P}(A_1)\,\mathbb{P}(A_2\mid A_1)\,\mathbb{P}(A_3\mid A_1\cap A_2)\cdots\mathbb{P}(A_n\mid A_1\cap\cdots\cap A_{n-1}),$$
valid whenever all the conditioning events have positive probability. This is not a formality — it is the identity behind **autoregressive models**. When a language model factorizes $p(x_1,\ldots,x_T) = \prod_t p(x_t\mid x_{<t})$ and is trained to predict each token from its predecessors, that factorization *is* the chain rule. It holds with no independence assumptions whatsoever, which is precisely why autoregressive modelling is so general.

> **Proposition 1.12.** Let $\{B_1,B_2,\ldots\}$ be a finite or countable partition of $\Omega$ into events with $\mathbb{P}(B_i)>0$. Then for any $A\in\mathcal{F}$:
> $$\textbf{(law of total probability)}\qquad \mathbb{P}(A) = \sum_{i}\mathbb{P}(A\mid B_i)\,\mathbb{P}(B_i),$$
> $$\textbf{(Bayes' rule)}\qquad \mathbb{P}(B_j\mid A) = \frac{\mathbb{P}(A\mid B_j)\mathbb{P}(B_j)}{\sum_i \mathbb{P}(A\mid B_i)\mathbb{P}(B_i)}, \quad \text{provided } \mathbb{P}(A)>0.$$

*Proof.* Since the $B_i$ partition $\Omega$, the sets $A\cap B_i$ are disjoint with union $A$. Countable additivity gives $\mathbb{P}(A) = \sum_i \mathbb{P}(A\cap B_i) = \sum_i\mathbb{P}(A\mid B_i)\mathbb{P}(B_i)$, which is the first claim. For the second, $\mathbb{P}(B_j\mid A) = \mathbb{P}(A\cap B_j)/\mathbb{P}(A)$; expand the numerator by the multiplication rule and the denominator by the first claim. ∎

Note that the law of total probability is just countable additivity in disguise — one more place where the "boring" axiom does the work.

**Terminology to internalize now, because Chapter 10 lives on it.** In Bayes' rule, $\mathbb{P}(B_j)$ is the **prior**, $\mathbb{P}(A\mid B_j)$ is the **likelihood**, $\mathbb{P}(A)$ is the **evidence** or **marginal likelihood**, and $\mathbb{P}(B_j\mid A)$ is the **posterior**. The compressed form worth memorizing is
$$\text{posterior} \ \propto\ \text{likelihood}\times\text{prior},$$
with the evidence acting purely as the normalizing constant that makes the posterior sum to 1. Most of the difficulty in Bayesian machine learning is that this constant is an intractable integral — which is what Chapter 13's variational methods and MCMC exist to circumvent.

### 1.7.2 Base rates: the mistake everyone makes

**Worked Example 1.7 (why a 99%-accurate test can be nearly useless).** A disease affects $0.5\%$ of a population. A test has **sensitivity** $99\%$ (detects the disease when present) and **specificity** $95\%$ (correctly returns negative when absent). You test positive. What is the probability you have the disease?

Most people answer "about 99%." Let us compute. Write $D$ for having the disease, $+$ for a positive test.

$$\mathbb{P}(D) = 0.005,\qquad \mathbb{P}(+\mid D) = 0.99,\qquad \mathbb{P}(+\mid D^c) = 1 - 0.95 = 0.05.$$

By the law of total probability,
$$\mathbb{P}(+) = 0.99\times0.005 + 0.05\times0.995 = 0.00495 + 0.04975 = 0.0547.$$
By Bayes' rule,
$$\mathbb{P}(D\mid +) = \frac{0.00495}{0.0547} = 0.0905.$$

**About 9%.** Not 99%. A simulation of $2\times10^6$ people gives $0.0913$ over the $109{,}415$ positives ✓.

The reason is visible in the arithmetic: the false positives ($0.05\times0.995 = 4.98\%$ of everyone) vastly outnumber the true positives ($0.99\times0.005 = 0.50\%$ of everyone), because there are simply 199 times more healthy people. **When the base rate is low, the false positives from the large healthy group swamp the true positives from the small sick group.**

```
Figure 1.3 — Base rates: 10,000 people; prevalence 0.5%,
             sensitivity 99%, specificity 95%

  Healthy: 9,950
  ├────────────────────────────── 9,452 true negatives ──────────┤▓▓▓┤
                                                        498 false positives

  Diseased: 50  (0.5% — the whole bar is tiny at this scale)
  ├█┤  ≈ 50 true positives (49.5 in expectation)

  Positive tests: 498 + 50 ≈ 548, of which only 50 are genuine.
  P(D | +) ≈ 50/548 ≈ 9%. A 5% false-positive rate applied to a huge
  healthy group produces ten times more positives than a 99% detection
  rate applied to a tiny sick one.
```

Two follow-ups worth doing yourself, with answers so you can check.

*What if you test positive twice, on independent tests?* Then the likelihood becomes $0.99^2$ against $0.05^2$, and
$$\mathbb{P}(D\mid ++) = \frac{0.99^2\times0.005}{0.99^2\times0.005 + 0.05^2\times0.995} = 0.663.$$
From 9% to 66%. Independent evidence compounds multiplicatively in the likelihood, which is exactly why sequential Bayesian updating is powerful — and why the independence assumption needs scrutiny, since correlated tests (same lab error, same confounder) do not compound this way.

*What specificity would be needed for a single positive to be more likely than not?* Solving $\mathbb{P}(D\mid+) = 0.5$ gives specificity $\approx 99.50\%$. To beat a base rate of 1 in 200, your false-positive rate must be pushed below roughly 1 in 200. **Rare events demand extreme specificity**, and no amount of sensitivity substitutes.

**Why this is central to machine learning, not a medical curiosity.** This example is the reason accuracy is a poor metric on imbalanced data — a classifier that always predicts "healthy" achieves 99.5% accuracy here while being useless. It is also why **precision** ($=\mathbb{P}(D\mid+)$, the posterior we just computed) and **recall** ($=\mathbb{P}(+\mid D)$, the sensitivity) are reported separately: they are conditional probabilities *conditioned on different events*, and confusing them is the single most common error in applied ML evaluation. Fraud detection, rare-disease screening, anomaly detection, and content moderation all live in the low-base-rate regime where this arithmetic dominates. Chapter 9 develops the full confusion-matrix vocabulary; the point here is that all of it is Bayes' rule.

### 1.7.3 Independence, previewed

Conditioning gives the cleanest statement of what independence means. Events $A$ and $B$ with $\mathbb{P}(B)>0$ satisfy $\mathbb{P}(A\cap B) = \mathbb{P}(A)\mathbb{P}(B)$ exactly when
$$\mathbb{P}(A\mid B) = \mathbb{P}(A),$$
i.e. **learning $B$ does not change your belief about $A$.** That is the content of independence, and §1.12 develops it properly. Note the product form is the better *definition* because it stays meaningful when $\mathbb{P}(B) = 0$ and is symmetric in $A$ and $B$.

### 1.7.4 The limitation that forces Chapter 6

Definition 1.10 requires $\mathbb{P}(B)>0$. That restriction is fatal for exactly the cases machine learning cares about.

Suppose $X$ is a continuous random variable and you want $\mathbb{P}(Y\in A\mid X = x)$ — the conditional distribution of a label given a real-valued feature. Then $\mathbb{P}(X = x) = 0$ for every $x$, so Definition 1.10 gives $0/0$. And yet "the distribution of $Y$ given $X=x$" is *the* object of supervised learning: a regression model estimates $\mathbb{E}[Y\mid X=x]$, and a classifier estimates $\mathbb{P}(Y=1\mid X=x)$.

So the elementary definition cannot support the central object of the field. The repair is to condition on a **$\sigma$-algebra** instead of an event — $\mathbb{E}[Y\mid\sigma(X)]$ — defining it by the property that it is the $\sigma(X)$-measurable function which integrates correctly against every event in $\sigma(X)$. That is Chapter 6, and this is why §1.3.4's information reading of $\sigma$-algebras was worth the effort: it is the only route to conditioning on continuous variables.

> **Common Confusion 1.5**
> *"Isn't $\mathbb{P}(A\mid B)$ just $\mathbb{P}(A)$ restricted to a smaller world? Why not define a new sample space $\Omega' = B$?"*
>
> You can, and sometimes it is clarifying: $(B, \{A\cap B : A\in\mathcal{F}\}, \mathbb{P}_B)$ is a legitimate probability space. But keeping $\mathbb{P}_B$ on the *original* $(\Omega,\mathcal{F})$ — as Proposition 1.11 does — is more useful, because it lets you compare $\mathbb{P}$ and $\mathbb{P}_B$ as two measures **on the same space**. That comparison is what makes it possible to define a KL divergence between a prior and a posterior (Chapter 11), and it is unavailable if you shrink the space: divergences compare measures with a common domain.

> **Self-check 1.7**
> (a) If $\mathbb{P}(A\mid B) = \mathbb{P}(A)$, must $\mathbb{P}(B\mid A) = \mathbb{P}(B)$? (Assume both events have probability strictly between 0 and 1.)
> (b) Can $\mathbb{P}(A\mid B) > \mathbb{P}(A)$ and $\mathbb{P}(A\mid B^c) > \mathbb{P}(A)$ both hold? (Same assumption.)
> (c) In Worked Example 1.7, what happens to $\mathbb{P}(D\mid+)$ as prevalence $\to 0$ with the test fixed?

---

## 1.8 Random variables are functions

Now the conceptual turn promised in §1.0.

Informally, "a random variable is a number whose value is random." That sentence is a useful crutch and also the reason many students never quite understand what is going on. Here is the correct picture.

**All the randomness lives in $\omega$.** Nature draws a single outcome $\omega\in\Omega$ — the random seed. A random variable is an ordinary, deterministic **function** that reads off some numerical feature of $\omega$. There is nothing random about the function: given the same $\omega$, it always returns the same number.

**Worked Example 1.8 (a random variable, explicitly).** Three coin flips, $\Omega = \{H,T\}^3$. Let $X(\omega) :=$ the number of heads. Then $X$ is the function

| $\omega$ | $HHH$ | $HHT$ | $HTH$ | $THH$ | $HTT$ | $THT$ | $TTH$ | $TTT$ |
|---|---|---|---|---|---|---|---|---|
| $X(\omega)$ | 3 | 2 | 2 | 2 | 1 | 1 | 1 | 0 |

A perfectly ordinary function from an 8-element set to $\{0,1,2,3\}$. What is random is *which column we land in*.

What makes a function a legitimate random variable? We want to ask questions about it — "is $X\le2$?" — and get probabilities. The question "$X\in B$?" corresponds to the event
$$X^{-1}(B) := \{\omega\in\Omega : X(\omega)\in B\}\subseteq\Omega,$$
the **preimage** of $B$. For $\mathbb{P}(X\in B)$ to be defined we need $X^{-1}(B)\in\mathcal{F}$. That is the whole definition.

> **Definition 1.13 (measurable function; random variable).** Let $(\Omega,\mathcal{F})$ and $(S,\mathcal{S})$ be measurable spaces. A function $X:\Omega\to S$ is **measurable** if
> $$X^{-1}(B)\in\mathcal{F}\quad\text{for every } B\in\mathcal{S}.$$
> When $(S,\mathcal{S}) = (\mathbb{R},\mathcal{B}(\mathbb{R}))$, $X$ is a **random variable**; when $(S,\mathcal{S}) = (\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))$, a **random vector**.

In words, and this is the sentence to remember:

> **Measurability says: every question about $X$ must be a question you can answer.**

It is a compatibility condition between the function $X$ and the information $\mathcal{F}$.

```
Figure 1.4 — Measurability: the preimage of every Borel set must be an event

        Ω  (outcomes, with information F)              ℝ
   ┌────────────────────────┐
   │        ┌─────────┐     │        X          ───────[══ B ══]──────►
   │        │ X⁻¹(B)  │     │   ──────────►          a Borel set
   │        └─────────┘     │
   └────────────────────────┘

  X is measurable when, for EVERY Borel B on the right, the preimage
  X⁻¹(B) on the left is a member of F. Intuition: "is X in B?" must be
  a question you can answer. Note that X maps left-to-right, but the
  CONDITION is checked right-to-left. That reversal is the thing to
  get used to.
```

### 1.8.1 A labour-saving shortcut

Checking every Borel set would be hopeless. You never have to.

> **Lemma 1.14.** Suppose $\mathcal{S} = \sigma(\mathcal{C})$ and $X^{-1}(C)\in\mathcal{F}$ for every $C\in\mathcal{C}$. Then $X$ is measurable.

*Proof.* Consider $\mathcal{A} := \{B\subseteq S : X^{-1}(B)\in\mathcal{F}\}$, the sets for which the condition happens to hold. I claim $\mathcal{A}$ is a $\sigma$-algebra. This follows from two preimage identities (Exercise A4):
$$X^{-1}(B^c) = \big(X^{-1}(B)\big)^c,\qquad X^{-1}\Big(\bigcup_n B_n\Big) = \bigcup_n X^{-1}(B_n).$$
Because preimages commute with complements and unions, $\mathcal{A}$ inherits closure under those operations from $\mathcal{F}$; and $S\in\mathcal{A}$ since $X^{-1}(S) = \Omega\in\mathcal{F}$. So $\mathcal{A}$ is a $\sigma$-algebra containing $\mathcal{C}$, and minimality gives $\sigma(\mathcal{C})\subseteq\mathcal{A}$ — that is, $\mathcal{S}\subseteq\mathcal{A}$, which says $X$ is measurable. ∎

That pattern — "the collection of good sets is itself a $\sigma$-algebra and contains the generators, so it contains everything" — recurs throughout measure theory.

**Consequence for real-valued $X$:** since $\mathcal{B}(\mathbb{R}) = \sigma(\{(-\infty,a] : a\in\mathbb{R}\})$, it suffices to check $\{X\le a\}\in\mathcal{F}$ for every $a\in\mathbb{R}$.

**Worked Example 1.9 (using the shortcut).** On $\Omega = \{1,\ldots,6\}$ with $\mathcal{F}_{\text{par}}$:

*Is $X(\omega) = \omega$ measurable?* Check $\{X\le1\} = \{1\}$. Is $\{1\}\in\mathcal{F}_{\text{par}}$? No. So $X$ is **not** measurable — sensible, since knowing only parity you cannot report the exact value.

*Is $Y(\omega) = \mathbb{1}[\omega\text{ even}]$ measurable?* Preimages are $\emptyset$, $\{1,3,5\}$, $\{2,4,6\}$, $\Omega$ — all in $\mathcal{F}_{\text{par}}$ ✓.

Notice the pattern, which is Proposition 1.4 in action: **a function on a finite space is measurable precisely when it is constant on each atom.** $Y$ is constant on $\{1,3,5\}$ and on $\{2,4,6\}$; $X$ is not. When stuck on a finite-space measurability question, draw the atoms and check constancy.

### 1.8.2 Measurability is almost never the practical obstacle

> **Proposition 1.15 (closure properties).** Let $f,g,f_1,f_2,\ldots$ be measurable real-valued functions on $(\Omega,\mathcal{F})$.
> 1. Every continuous $h:\mathbb{R}^n\to\mathbb{R}^m$ is Borel measurable.
> 2. Compositions of measurable functions are measurable.
> 3. $f+g$, $fg$, $\max(f,g)$, $\min(f,g)$, $|f|$, $cf$ are measurable.
> 4. For a **countable** sequence, $\sup_n f_n$, $\inf_n f_n$, $\limsup_n f_n$, $\liminf_n f_n$ are measurable, and so is $\lim_n f_n$ where it exists.
> 5. $\mathbb{1}_A$ is measurable iff $A\in\mathcal{F}$.

*Proof sketches.* (1) For continuous $h$, preimages of open sets are open, and open sets generate the Borel $\sigma$-algebra; apply Lemma 1.14. (2) $(g\circ f)^{-1}(B) = f^{-1}(g^{-1}(B))$: chase the definitions. (3) $(f,g)$ is measurable into $\mathbb{R}^2$ (checked on rectangles, §1.9.3), and $+,\times,\max,\min$ are continuous maps $\mathbb{R}^2\to\mathbb{R}$; compose. (5) Preimages of $\mathbb{1}_A$ are $\emptyset, A, A^c, \Omega$.

Item 4 is the one that distinguishes $\sigma$-algebras from weaker structures. Its proof is a one-liner illustrating why: $\{\sup_n f_n \le a\} = \bigcap_n \{f_n\le a\}$, a **countable** intersection of events, hence an event; then apply Lemma 1.14 with the generators $(-\infty, a]$. The countability is doing real work.

**The practical upshot:** every function appearing in a neural network is measurable. Affine layers, ReLU, sigmoid, softmax, attention, batch normalization, max-pooling, and even non-differentiable operations like rounding, `argmax`, and hard thresholding. You will not accidentally build a non-measurable network. Understanding *why* measurability is required is the point, not checking it case by case.

### 1.8.3 Where measurability *does* bite: uncountable suprema

Item 4 above says *countable* sequence. For uncountable families the result is **false**, and this is not a curiosity — it is a real gap in the foundations of statistical learning theory.

**A counterexample.** Let $V\subseteq[0,1]$ be a non-measurable set (§1.5), and for each $t\in V$ define $f_t := \mathbb{1}_{\{t\}}$ on $([0,1],\mathcal{B}([0,1]))$. Each $f_t$ is measurable, being the indicator of a single point. But
$$\sup_{t\in V} f_t = \mathbb{1}_V,$$
which is *not* measurable, since $V\notin\mathcal{B}([0,1])$. An uncountable supremum of measurable functions escaped measurability. ∎

**Why an ML researcher must care.** Every uniform-convergence generalization bound is a statement about
$$\sup_{h\in\mathcal{H}}\big|\hat R_n(h) - R(h)\big|,$$
where $\hat R_n$ is training error, $R$ is true risk, and $\mathcal{H}$ is the hypothesis class. For a neural network, $\mathcal{H}$ is indexed by a continuum of weight vectors — **uncountable**. So the quantity we want to bound is not automatically a random variable, and writing $\mathbb{P}\big(\sup_{h}|\hat R_n(h)-R(h)| > \varepsilon\big)$ presupposes something that needs justification.

Three standard resolutions, in increasing order of sophistication — and, this time, with an honest statement of what each one does and does not cover.

1. **Finite or countable $\mathcal{H}$.** No problem; the union bound of §1.6.3 applies directly. This is why introductory treatments start with finite classes.

2. **Continuity plus separability.** Suppose the map $w\mapsto \phi(w) := |\hat R_n(h_w) - R(h_w)|$ is **continuous** in the parameters $w$, and the parameter space is separable (has a countable dense subset — true of $\mathbb{R}^p$). Then the supremum over all of $\mathbb{R}^p$ equals the supremum over a countable dense subset (Exercise A12), which *is* measurable by item 4. **When does the continuity hypothesis hold?** For losses that are continuous in the network output — squared error, cross-entropy on soft predictions, hinge loss — composed with a network that is continuous in its weights: then $\hat R_n$ is continuous in $w$, and $R$ is too under mild conditions (dominated convergence, Chapter 2). This covers regression and surrogate-loss classification.

3. **When continuity fails: the 0–1 loss.** The loss most generalization bounds are actually *about* — the misclassification indicator $\mathbb{1}[\operatorname{sign} f_w(x) \ne y]$ — is **not continuous in $w$**: an arbitrarily small weight change can flip a prediction. So resolution 2 does not apply as stated, and this case genuinely needs more: empirical process theory replaces the supremum with a measurable majorant, or verifies *pointwise measurability* / *admissibility* conditions ("image admissible Suslin" is the phrase you will meet) under which a countable subclass still controls the supremum. This is the machinery of van der Vaart and Wellner, and it is what careful papers cite. The important honest summary: **the standard bounds are all fine, but for classification with 0–1 loss the reason is not the simple separability argument, and rigorous treatments must (and do) say more.**

So Common Confusion 1.2's answer — "countable is exactly the right amount" — has a concrete consequence: the countability restriction that makes measure theory coherent is also the reason learning theory needs a measurability argument that introductory accounts almost never state. Chapter 4 returns to this. (Recall also §1.4.1's caution about projections: "$\exists$ over a continuum" and "$\sup$ over a continuum" are the same dangerous shape.)

> **Common Confusion 1.6**
> *"Why call it a random variable if it's just a function? Where did the randomness go?"*
>
> Into $\Omega$, and it stayed there. This reframing is useful for three reasons.
>
> First, it explains how several random variables can be *related*. If $X$ and $Y$ are both functions on the same $\Omega$, they are evaluated at the *same* $\omega$, which is what makes correlation, dependence, and joint distributions meaningful. Two "random numbers" floating free have no relationship; two functions on a shared domain do.
>
> Second, it explains what a random *vector* or random *process* is: a function into $\mathbb{R}^d$, or into a space of paths.
>
> Third, it matches your code. `x = f(rng)` is a deterministic function of generator state. The seed is $\omega$.

> **Self-check 1.8**
> (a) On $\Omega=\{1,\ldots,6\}$ with $\mathcal{F}_{\text{par}}$, is $Z(\omega) = (-1)^\omega$ measurable?
> (b) In Worked Example 1.8, what is $X^{-1}(\{2\})$? What is $X^{-1}([1.5,10])$?
> (c) If $X$ is measurable, must $X^2$ be? Must $\sqrt{|X|}$ be?
> (d) Why is $\{\inf_n f_n < a\} = \bigcup_n\{f_n < a\}$ an event, and where would the argument fail for an uncountable family?

---

## 1.9 Distributions: the pushforward measure

We have $\mathbb{P}$ on $\Omega$ and a random variable $X$ mapping into $\mathbb{R}$. But we almost never care about $\Omega$. We care how $X$ is spread over $\mathbb{R}$. So transport the probability.

> **Definition 1.16 (law / distribution / pushforward).** For a random variable $X$ on $(\Omega,\mathcal{F},\mathbb{P})$, define $P_X$ on $\mathcal{B}(\mathbb{R})$ by
> $$P_X(B) := \mathbb{P}\big(X^{-1}(B)\big) = \mathbb{P}(X\in B).$$
> $P_X$ is the **law**, or **distribution**, of $X$; also written $\mathbb{P}\circ X^{-1}$ or $X_\#\mathbb{P}$, the **pushforward** of $\mathbb{P}$ by $X$. We write $X\sim P_X$.

This is well defined precisely because $X$ is measurable: $X^{-1}(B)\in\mathcal{F}$, so $\mathbb{P}$ applies.

**It really is a probability measure.** $P_X(\mathbb{R}) = \mathbb{P}(X^{-1}(\mathbb{R})) = \mathbb{P}(\Omega) = 1$ ✓. For disjoint $B_1,B_2,\ldots$ the preimages are also disjoint (if $\omega$ were in two, $X(\omega)$ would lie in two disjoint $B$'s), so
$$P_X\Big(\bigsqcup_n B_n\Big) = \mathbb{P}\Big(\bigsqcup_n X^{-1}(B_n)\Big) = \sum_n\mathbb{P}(X^{-1}(B_n)) = \sum_n P_X(B_n).\ ✓$$

The mental image: $\mathbb{P}$ distributes sand over $\Omega$; $X$ picks up each grain and drops it at position $X(\omega)$; $P_X$ describes the resulting pile. Grains sent to the same place merge.

```
Figure 1.5 — Pushforward: X = #heads transports mass from Ω to ℝ

     (Ω, P)  fair coin, each outcome mass 1/8          P_X on ℝ

      HHH ────────────────────────────► 3                     3/8  3/8
      HHT ─┐                                                  ┌─┐  ┌─┐
      HTH ─┼──────────────────────────► 2                     │ │  │ │
      THH ─┘                                             1/8  │ │  │ │  1/8
      HTT ─┐                                             ┌─┐  │ │  │ │  ┌─┐
      THT ─┼──────────────────────────► 1                │ │  │ │  │ │  │ │
      TTH ─┘                                            ─┴─┴──┴─┴──┴─┴──┴─┴─
      TTT ────────────────────────────► 0                 0    1    2    3

  The three outcomes mapping to X = 2 merge into one bar of height 3/8.
  Each level set of X becomes exactly one bar: the pushforward forgets
  everything about ω except what X reports.
```

**Worked Example 1.10 (a discrete pushforward).** Continue Worked Example 1.8 with a fair coin, $\mathbb{P}(\{\omega\}) = 1/8$. Reading the table:
$$P_X(\{0\}) = \tfrac18,\quad P_X(\{1\}) = \tfrac38,\quad P_X(\{2\}) = \tfrac38,\quad P_X(\{3\}) = \tfrac18.$$
This is $\mathrm{Binomial}(3,\tfrac12)$. Note $P_X$ assigns zero to every Borel set missing $\{0,1,2,3\}$: all the mass sits on four points, though $P_X$ is formally a measure on all of $\mathcal{B}(\mathbb{R})$.

**Worked Example 1.11 (a continuous pushforward, two ways).** Let $\Omega = [-1,1]$ with $\mathbb{P}$ uniform ($\mathbb{P}(A) = \lambda(A)/2$) and $X(\omega) = \omega^2$.

*Direct.* For $b\in[0,1]$,
$$P_X([0,b]) = \mathbb{P}(\omega^2\le b) = \mathbb{P}(-\sqrt b\le\omega\le\sqrt b) = \frac{2\sqrt b}{2} = \sqrt b.$$
So $F_X(b) = \sqrt b$ on $[0,1]$, with density $\frac{d}{db}\sqrt b = \frac{1}{2\sqrt b}$ — unbounded as $b\to0^+$ yet integrable. The squaring map crushes the region near $\omega = 0$ into a tiny neighbourhood of $0$, piling probability there.

*Empirical check* ($10^6$ samples): $\hat{\mathbb{P}}(X\le0.25) = 0.4999$ vs $0.5$; $\hat{\mathbb{P}}(X\le0.5) = 0.7072$ vs $0.7071$; $\hat{\mathbb{P}}(X\le0.81) = 0.9002$ vs $0.9$. ✓

Preview: the factor $\frac{1}{2\sqrt b}$ will be revealed in Chapter 3 as a Jacobian — the engine of normalizing flows.

### 1.9.1 Why the pushforward is the central concept

**We can now stop mentioning $\Omega$.** Two experiments with utterly different sample spaces can produce random variables with the same law, and *no statistical procedure can tell them apart* — every probabilistic statement about $X$ alone is a statement about $P_X$. So $\Omega$ is scaffolding: necessary to erect the building, invisible once it stands.

This justifies exactly what practitioners do. You write $X\sim\mathcal{N}(0,1)$ and never specify $\Omega$. When you call `torch.randn(100)`, the underlying $\Omega$ is the pseudorandom generator's state space; you neither know nor care.

**Every deep generative model is a pushforward.** A generator is a measurable map $G_\theta:\mathbb{R}^k\to\mathbb{R}^d$ taking a simple latent $Z\sim\mathcal{N}(0,I_k)$ to $X = G_\theta(Z)$. The model's distribution *is*
$$P_\theta = (G_\theta)_\#\,\mathcal{N}(0,I_k).$$
GANs, VAE decoders, normalizing flows, and diffusion samplers all do this, differing in how $G_\theta$ is parameterized and how mismatch with the data is measured — not in the basic object.

### 1.9.2 Support: saying where a distribution lives

To state the next results I need a word people often use loosely. Here is its actual definition.

> **Definition 1.17 (support).** For a Borel probability measure $P$ on $\mathbb{R}^d$, the **support** is
> $$\operatorname{supp}(P) := \{x\in\mathbb{R}^d : P(B(x,r)) > 0 \text{ for every } r>0\},$$
> where $B(x,r)$ is the open ball of radius $r$. Equivalently, $\operatorname{supp}(P)$ is the smallest **closed** set of full measure.

*Why the two descriptions agree, and why a smallest closed set exists.* The complement of the set defined above is $U := \bigcup\{B(x,r) : P(B(x,r)) = 0\}$, an open set. Because $\mathbb{R}^d$ has a countable base of balls with rational centres and radii, every null ball in that union can be replaced by the null rational balls it contains, so $U$ is in fact a *countable* union of null balls, and $P(U) = 0$ by the union bound — therefore $P(\operatorname{supp}(P)) = 1$. Any closed set $C$ with $P(C) = 1$ must contain $\operatorname{supp}(P)$: a point $x\notin C$ has a ball missing $C$, hence a ball of measure zero, hence $x\notin\operatorname{supp}(P)$. ∎

Notice that countability was again essential — without a countable base, $U$ could be an uncountable union of null sets and needn't be null.

Two cautions, both of which matter for how I use the word later.

**A set of full measure need not contain the support.** Let $P$ be uniform on $[0,1]$ and $N = [0,1]\setminus\{0.5\}$. Then $P(N) = 1$, yet $0.5\in\operatorname{supp}(P)$ and $0.5\notin N$. So "$P(N)=1$" and "$N\supseteq\operatorname{supp}(P)$" are different statements; only the *closed* full-measure sets are guaranteed to contain the support.

**The support is about where mass *isn't*, not where it's concentrated.** $\operatorname{supp}(\mathcal{N}(0,1)) = \mathbb{R}$, even though essentially all the mass is within a few standard deviations of the origin. A Gaussian's support tells you nothing about its shape. This is worth flagging because "the data lies on a low-dimensional manifold" is a claim about the support, and a Gaussian with tiny variance in most directions has *full* support while being *effectively* low-dimensional. The distinction between exactly-low-dimensional support and approximately-low-dimensional mass is precisely what separates the GAN pathology below from the merely-hard case.

### 1.9.3 Random vectors and joint distributions

Definition 1.13 covered random vectors; let us actually use one, since joint distributions are where dependence lives.

First, a convenient fact: $\mathcal{B}(\mathbb{R}^2) = \sigma(\{A\times B : A,B\in\mathcal{B}(\mathbb{R})\})$, the $\sigma$-algebra generated by rectangles (§1.14). Combining this with Lemma 1.14 gives:

> **A map $(X,Y):\Omega\to\mathbb{R}^2$ is measurable if and only if $X$ and $Y$ are each measurable.**

(One direction: $\{X\in A\} = \{(X,Y)\in A\times\mathbb{R}\}$. The other: check the generating rectangles, $\{(X,Y)\in A\times B\} = \{X\in A\}\cap\{Y\in B\}\in\mathcal{F}$.) So "random vector" adds no new measurability burden — but its *law* $P_{(X,Y)}$ on $\mathbb{R}^2$ contains in general more information than the pair $(P_X, P_Y)$: the marginals never determine the joint unless something extra (like independence, or degeneracy of a marginal) pins it down.

**Worked Example 1.12 (a joint law, and a trap).** Roll two fair dice: $\Omega = \{1,\ldots,6\}^2$ with $\mathbb{P}$ uniform, so each outcome has probability $1/36$. Let $X$ be the first die and $Y$ the sum.

*The joint law.* $(X,Y) = (x,y)$ happens exactly when the first die is $x$ and the second is $y - x$, which is possible iff $1\le y-x\le6$. So
$$P_{(X,Y)}(\{(x,y)\}) = \begin{cases}\tfrac1{36}, & 1\le x\le 6 \text{ and } 1\le y-x\le6,\\ 0,&\text{otherwise.}\end{cases}$$
There are $36$ such pairs, total mass $1$ ✓. Note the support is a *diagonal band* in the plane, not a rectangle — geometry that encodes the dependence.

*The marginals.* Summing over $y$: $P_X(\{x\}) = 6/36 = 1/6$, uniform ✓. Summing over $x$: $P_Y(\{y\}) = \#\{x : 1\le y-x\le6\}/36$, giving the familiar triangular distribution $1/36, 2/36,\ldots,6/36,\ldots,1/36$ for $y = 2,\ldots,12$.

*Now the trap.* Check independence at $(x,y) = (1,7)$:
$$\mathbb{P}(X=1, Y=7) = \tfrac1{36} = 0.02778, \qquad \mathbb{P}(X=1)\mathbb{P}(Y=7) = \tfrac16\cdot\tfrac6{36} = \tfrac1{36} = 0.02778.$$
**They agree exactly.** If you checked only this pair you would conclude independence. But check $(1,2)$:
$$\mathbb{P}(X=1,Y=2) = \tfrac1{36} = 0.02778,\qquad \mathbb{P}(X=1)\mathbb{P}(Y=2) = \tfrac16\cdot\tfrac1{36} = \tfrac1{216} = 0.00463.$$
Not equal. $X$ and $Y$ are **dependent** — obviously so, since $Y\ge X+1$ always.

The lesson generalizes and is worth stating loudly: **independence requires the product rule for *every* pair of events, and a coincidental match on one pair proves nothing.** This is the same failure mode as pairwise-versus-joint independence (§1.12), and it is why independence testing is genuinely hard: verifying a condition that must hold universally, from finite data.

**Forward pointer.** Once Chapter 2 gives us product measures, independence acquires its cleanest formulation: $X\perp Y$ if and only if
$$P_{(X,Y)} = P_X\otimes P_Y,$$
i.e. **independence *means* the joint law is a product measure.** In Worked Example 1.12, $P_{(X,Y)}$ is supported on a diagonal band while $P_X\otimes P_Y$ is supported on the full rectangle $\{1,\ldots,6\}\times\{2,\ldots,12\}$ — visibly different measures. Dependence has a shape.

### 1.9.4 Failure 4, set up: generators have zero volume

Now look what the formalism gives us for free. If $k<d$ — as always, since latent dimensions are far smaller than pixel counts — then $P_\theta$ assigns probability 1 to the image $G_\theta(\mathbb{R}^k)$, at most a $k$-dimensional surface in $\mathbb{R}^d$. Such a surface has **Lebesgue measure zero** (for $C^1$ maps; §1.14 and Exercise A11).

**Worked Example 1.13 (a generator's support has zero volume).** Take $k=1$, $d=2$, $G(t) = (\cos t,\sin t)$, $Z\sim\mathcal{N}(0,1)$. Then $P_\theta$ lives entirely on the unit circle, and $\operatorname{supp}(P_\theta)$ is exactly that circle (closed ✓, full measure ✓). Numerically, $200{,}000$ samples all satisfy $\|G(Z)\| = 1$ to six decimals, and **zero** samples land in $\{\|x\|<0.99\}$ — a region containing $98.01\%$ of the unit disc's area.

The circle has zero area in $\mathbb{R}^2$. So there is **no density** $p$ with $P_\theta(A) = \int_A p\,d\lambda_2$: such a formula would give $P_\theta(\text{circle}) = 0$, contradicting $P_\theta(\text{circle}) = 1$.

Two consequences, exactly the ones from §1.1:

- **A GAN generator has no Lebesgue density**, so "the model's likelihood" is not merely hard to compute — it does not exist. This is why GANs are trained adversarially rather than by maximum likelihood, and it is not an engineering choice.
- **Data and model distributions live on different measure-zero sets**, which generically do not intersect. Chapter 2 shows this makes the KL divergence infinite and Jensen–Shannon constant, hence gradient-free.

> **Common Confusion 1.7**
> *"If the generator's distribution has no density, how do papers report likelihoods for normalizing flows?"*
>
> Because flows are built differently on purpose. A normalizing flow uses $k=d$ and an *invertible* $G_\theta$, so it does not collapse dimensions and its pushforward does have a Lebesgue density, computable by change of variables. That architectural constraint is the price of having a likelihood at all (Chapter 3). VAEs take a third route: adding Gaussian noise in the output space thickens the measure-zero surface into something with full-dimensional support — as Lab B4 demonstrates.

> **Self-check 1.9**
> (a) Let $\Omega = \{1,\ldots,6\}$ uniform and $X(\omega) = \omega\bmod3$. Compute $P_X$.
> (b) If $X$ and $Y$ have the same law, must $X(\omega) = Y(\omega)$ for all $\omega$?
> (c) A generator is a $C^1$ map $\mathbb{R}^{100}\to\mathbb{R}^{3072}$. What is $\lambda_{3072}(\operatorname{supp}(P_\theta))$?
> (d) What is $\operatorname{supp}$ of the Binomial$(3,\tfrac12)$ law from Worked Example 1.10?

---

## 1.10 Cumulative distribution functions, and the ReLU resolved

A measure on $\mathcal{B}(\mathbb{R})$ is unwieldy — a function on an enormous collection of sets. It would be convenient to summarize it by a function of a *real variable*.

> **Definition 1.18.** The **cumulative distribution function** of $X$ is $F_X(a) := \mathbb{P}(X\le a) = P_X\big((-\infty,a]\big)$.

> **Proposition 1.19.** $F_X$ is non-decreasing and right-continuous, with $\lim_{a\to-\infty}F_X(a) = 0$ and $\lim_{a\to\infty}F_X(a) = 1$. Conversely, every function with these properties is the CDF of some random variable. *(The converse is proved constructively in §1.10.2 below, using one borrowed ingredient listed in §1.14.)*

*Proof of the listed properties.* Monotonicity is Proposition 1.8(1), since $a\le a'$ gives $(-\infty,a]\subseteq(-\infty,a']$. For right-continuity, take $a_n\downarrow a$; the sets $(-\infty,a_n]$ decrease with intersection $(-\infty,a]$, so continuity from above (Proposition 1.8(4), legitimate as $\mathbb{P}$ is finite) gives $F_X(a_n)\to F_X(a)$. The limits at $\pm\infty$ follow from continuity from below and above applied to $(-\infty,n]$ and $(-\infty,-n]$. ∎

Each analytic property of $F_X$ came from a *measure-theoretic* continuity property — the first payoff of Proposition 1.8.

**Two useful formulas.**
$$\mathbb{P}(a<X\le b) = F_X(b)-F_X(a),\qquad \mathbb{P}(X=a) = F_X(a)-F_X(a^-),$$
where $F_X(a^-)$ is the left limit. So **the jumps of $F_X$ are exactly the atoms of $P_X$**, with heights equal to the point probabilities. Continuous $F_X$ means no atoms.

**Does $F_X$ determine $P_X$?** Yes, and the reason is a theorem I will state rather than leave mysterious.

> **Theorem 1.20 (uniqueness of measures).** Let $\mathcal{C}$ be a collection of subsets of $\Omega$ closed under finite intersections (a **$\pi$-system**), and let $\mu,\nu$ be probability measures on $\sigma(\mathcal{C})$. If $\mu(C) = \nu(C)$ for all $C\in\mathcal{C}$, then $\mu=\nu$ on all of $\sigma(\mathcal{C})$.

This is a consequence of Dynkin's $\pi$–$\lambda$ theorem; we use it as a black box (§1.14). Here $\mathcal{C} = \{(-\infty,a]\}$ is a $\pi$-system, since $(-\infty,a]\cap(-\infty,b] = (-\infty,\min(a,b)]$ ✓, and it generates $\mathcal{B}(\mathbb{R})$. Hence two random variables with the same CDF have the same law. **This is why your first course could define distributions by their CDFs.** It also justifies basing tests on the empirical CDF (Kolmogorov–Smirnov, Chapter 9).

### 1.10.1 Failure 2, half-resolved

**Worked Example 1.14 (the ReLU output).** Let $Z\sim\mathcal{N}(0,1)$, $A = \max(0,Z)$, and write $\Phi,\varphi$ for the standard normal CDF and density.

For $a<0$: $\{A\le a\}$ is empty since $A\ge0$, so $F_A(a) = 0$.
For $a\ge0$: $A\le a$ happens when $Z\le0$ (giving $A=0\le a$) or when $0<Z\le a$ — together exactly $\{Z\le a\}$. So $F_A(a) = \Phi(a)$.

$$F_A(a) = \begin{cases}0,&a<0,\\ \Phi(a),&a\ge0.\end{cases}$$

Read the structure off the graph. At $a=0$ the function jumps from $0$ to $\Phi(0) = \tfrac12$: an **atom of mass $1/2$**, matching $\mathbb{P}(Z\le0)$. For $a>0$ it rises smoothly with slope $\varphi(a)$: a **continuous part** carrying the other half.

*Empirical check* ($10^6$ samples): $\hat{\mathbb{P}}(A=0) = 0.4994$ vs $0.5$; $\hat F_A(1) = 0.8412$ vs $0.8413$; $\hat F_A(2) = 0.9772$ vs $0.9772$ ✓.

```
Figure 1.6 — CDF of A = max(0, Z), Z ~ N(0,1): a jump plus a smooth rise

  F_A(a)
  1.0 ┤                                    ______________
      │                          _________/
      │                   ______/          smooth rise = Φ(a):
      │              ____/                 the continuous half
  0.5 ┤             ●
      │             ┆  ← jump of height 1/2 = P(A = 0), an atom
      │             ┆
  0.0 ┼─────────────○──────────────────────────────► a
                    0

  ○ = open endpoint (value not attained), ● = closed (right-continuity).
  The CDF handles this distribution without complaint, even though A
  has neither a pmf (its mass is not countably supported) nor a pdf
  (it has an atom). Chapter 2 gives the object unifying all cases.
```

So the CDF *describes* $A$ perfectly well. What we still lack is a single kind of object — like a density — that works for discrete, continuous, and mixed distributions alike, so formulas can be written once. That is the Radon–Nikodym derivative, opening Chapter 2.

### 1.10.2 The quantile function, and how samplers actually work

The converse half of Proposition 1.19 — every candidate CDF is realized by some random variable — deserves more than a citation, because its proof is an algorithm you use every day.

> **Definition 1.21 (quantile function / generalized inverse).** For a CDF $F$, define
> $$F^{-1}(u) := \inf\{a\in\mathbb{R} : F(a)\ge u\},\qquad u\in(0,1).$$
> When $F$ is continuous and strictly increasing this is the ordinary inverse; the infimum handles flat stretches (jump over them) and jumps (land on them).

> **Proposition 1.22 (inverse-transform sampling).** Let $U\sim\mathrm{Unif}(0,1)$ and $X := F^{-1}(U)$. Then $X$ has CDF $F$.

*Proof.* The key fact about the generalized inverse is the equivalence
$$F^{-1}(u)\le a \iff u\le F(a)\qquad\text{for all } u\in(0,1),\ a\in\mathbb{R}.$$
($\Leftarrow$: if $u\le F(a)$ then $a$ belongs to the set being infimized, so the inf is $\le a$. $\Rightarrow$: if $F^{-1}(u)\le a$, take $a_n\downarrow F^{-1}(u)$ with $F(a_n)\ge u$; right-continuity of $F$ gives $F(F^{-1}(u))\ge u$, and monotonicity gives $F(a)\ge F(F^{-1}(u))\ge u$.) Therefore
$$\mathbb{P}(X\le a) = \mathbb{P}\big(F^{-1}(U)\le a\big) = \mathbb{P}\big(U\le F(a)\big) = F(a),$$
the last step because $U$ is uniform. ∎

Note exactly where right-continuity of $F$ was used — the property we derived from continuity from above in Proposition 1.19. The one ingredient taken on faith is the *existence of a uniform random variable* on some probability space, i.e. of Lebesgue measure on $[0,1]$; that is the borrowed row in §1.14. Given that single seed, every distribution on $\mathbb{R}$ exists.

**Why you should care beyond the theory.** Inverse-transform sampling is how random number libraries generate exponential, Cauchy, and many other draws: produce a uniform, apply $F^{-1}$. It is also, composed coordinate-wise, the simplest example of the pattern of §1.9.1 — *every sampler is a pushforward of something simple* — and the conceptual ancestor of normalizing flows. Lab B6 has you implement it and watch it handle the ReLU distribution, atom and all: when $U$ lands in $(0, 1/2]$, the flat inverse returns exactly $0$, reproducing the atom.

### 1.10.3 The general picture: Lebesgue's decomposition

Lebesgue's decomposition theorem (proved in Chapter 2) says every distribution on $\mathbb{R}$ splits uniquely into three parts:

- a **discrete** part — atoms, visible as jumps in $F$;
- an **absolutely continuous** part — has a density, visible as smooth increase in $F$;
- a **singular continuous** part — $F$ continuous (no atoms) yet all of its increase happens on a set of Lebesgue measure zero, so there is no density either.

The ReLU output is discrete + absolutely continuous. The third category sounds exotic, so let us build one.

**Worked Example 1.15 (the Cantor distribution — singular continuous, concretely).** Flip a fair coin infinitely often, getting bits $b_1, b_2, \ldots$, and set
$$X := \sum_{n=1}^\infty \frac{2b_n}{3^n}$$
— i.e. write down a random base-3 number whose digits are 0 or 2, each with probability $1/2$, never 1. Then:

*No atoms.* Each specific digit sequence has probability $\lim_m 2^{-m} = 0$, and distinct sequences give distinct points, so $\mathbb{P}(X = x) = 0$ for every $x$: $F_X$ is continuous.

*No density.* $X$ lands with probability 1 in the **Cantor set** $C$ — the numbers in $[0,1]$ with a base-3 expansion avoiding the digit 1. And $\lambda(C) = 0$: at stage $m$, the surviving numbers occupy $2^m$ intervals of length $3^{-m}$, total length $(2/3)^m \to 0$. A density would force $1 = P_X(C) = \int_C p\,d\lambda = 0$, contradiction — the same three-line argument as Worked Example 1.13.

So $F_X$ (the "Cantor function" or "devil's staircase") climbs from 0 to 1 continuously while being flat outside a measure-zero set. Continuous CDF does **not** imply a density exists. Hold onto this example: it is a one-dimensional toy of exactly the phenomenon in the next subsection.

### 1.10.4 What kind of distribution does a deep generator output?

Now we can classify the pushforward $P_\theta = (G_\theta)_\#\nu$ of a generator $G_\theta:\mathbb{R}^k\to\mathbb{R}^d$, $k<d$, $G_\theta$ continuous, $\nu$ the latent distribution (say $\mathcal{N}(0,I_k)$, which has a density on $\mathbb{R}^k$).

*It has no absolutely continuous part of full mass:* §1.9.4 showed the image is Lebesgue-null (for $C^1$ maps), so no Lebesgue density exists.

*Does it have atoms?* Here is where careless reasoning goes wrong, so let us be careful. An atom of $P_\theta$ at $x_0$ means $\nu\big(G_\theta^{-1}(\{x_0\})\big) > 0$: **a set of latent values of positive probability all mapping to the same output.** One might hope that ruling out constant $G_\theta$ rules this out. **It does not.** A map can be non-constant globally while constant on a large region:

> **Counterexample.** Let $G(z) = (\max(0, z_1),\, 0)$ on $\mathbb{R}^2$, with $Z\sim\mathcal{N}(0,I_2)$. Then $G$ is non-constant, yet $G(Z) = (0,0)$ whenever $z_1\le0$ — an event of probability $1/2$. The output distribution has an atom of mass $1/2$ at the origin.

And this is not a contrived map: **it is a ReLU**. Real generators built from ReLU networks are piecewise affine and can be constant on entire polyhedral regions of latent space; when such a region has positive latent probability, the output distribution has an atom. (Whether trained networks actually do this is an empirical question; the mathematics permits it, and Failure 2's mechanism — a nonlinearity flattening a region to a point — is exactly what produces it.)

*The correct statement:*

> **Proposition 1.23.** Let $P_\theta = (G_\theta)_\#\nu$ with $G_\theta:\mathbb{R}^k\to\mathbb{R}^d$ of class $C^1$ and $k<d$. Then $P_\theta$ has no absolutely continuous part: it decomposes into atoms plus a singular continuous part. Moreover, $P_\theta$ has **no atoms** — hence is **purely singular continuous** — if and only if $\nu\big(G_\theta^{-1}(\{x\})\big) = 0$ for every $x\in\mathbb{R}^d$. A convenient sufficient condition: if $\nu$ has a density on $\mathbb{R}^k$ and every level set $G_\theta^{-1}(\{x\})$ has Lebesgue measure zero in $\mathbb{R}^k$ — for instance, if the Jacobian of $G_\theta$ has full rank $k$ at $\nu$-almost every point, so that level sets are lower-dimensional — then there are no atoms.

*Proof.* The absence of an absolutely continuous part is Exercise A11. The atom characterization is just the definition, $P_\theta(\{x\}) = \nu(G_\theta^{-1}(\{x\}))$. For the sufficient condition: a latent density means $\nu\ll\lambda_k$, so $\lambda_k$-null level sets are $\nu$-null. ∎

**Takeaway for reading papers.** A smooth generator with almost-everywhere full-rank Jacobian outputs a singular continuous distribution — a high-dimensional cousin of the Cantor distribution, and the setting of Failure 4. A ReLU generator may additionally place point masses. In *neither* case does a Lebesgue density exist, so "the model's log-likelihood" is undefined; every architecture that reports one has specifically engineered its way around Proposition 1.23 (see Exercise C2(d)).

> **Self-check 1.10**
> (a) Sketch the CDF of a variable equal to $0$ with probability $1/3$ and uniform on $[1,2]$ otherwise.
> (b) From a CDF $F$, how do you read off $\mathbb{P}(X=3)$?
> (c) Can a CDF be left-continuous but not right-continuous? Why does the convention $\{X\le a\}$ rather than $\{X<a\}$ matter?
> (d) $U\sim\mathrm{Unif}(0,1)$. What is the distribution of $-\ln(1-U)$? (Compute its CDF directly; this is inverse-transform sampling in action.)

---

## 1.11 The information carried by a random variable

Now we close the loop between §1.3's information picture and §1.8's functions.

> **Definition 1.24.** For a random variable $X$, $\ \sigma(X) := \{X^{-1}(B) : B\in\mathcal{B}(\mathbb{R})\}\subseteq\mathcal{F}$, the **$\sigma$-algebra generated by $X$**. (It is a $\sigma$-algebra by the preimage identities in Lemma 1.14's proof.)

$\sigma(X)$ is **the information you acquire by observing $X$**.

**Worked Example 1.16.** In Worked Example 1.8 ($X$ = number of heads in three flips), $\sigma(X)$ is determined by its atoms, the level sets of $X$:
$$\{TTT\},\quad\{HTT,THT,TTH\},\quad\{HHT,HTH,THH\},\quad\{HHH\}.$$
Four atoms, so $|\sigma(X)| = 2^4 = 16$ — far smaller than $|\mathcal{F}| = 256$. Learning the *count* is strictly less information than learning the *sequence*: told "$X=2$", you cannot determine which flip was tails. The three outcomes in that atom are indistinguishable to you.

Two facts to carry forward.

**Fact 1.** $X$ is $\mathcal{G}$-measurable $\iff\sigma(X)\subseteq\mathcal{G}$: "$X$ can be computed from the information in $\mathcal{G}$."

> **Lemma 1.25 (Doob–Dynkin).** $Y$ is $\sigma(X)$-measurable $\iff$ $Y = g(X)$ for some measurable $g:\mathbb{R}\to\mathbb{R}$.

This says something philosophically satisfying:

> *"$Y$ is determined by $X$"* and *"$Y$ is a function of $X$"* are the **same statement**.

*Proof for finitely-valued $X$* (the general case is in §1.14). ($\Leftarrow$) If $Y = g(X)$ then $\sigma(Y)\subseteq\sigma(X)$ by Exercise A8(a). ($\Rightarrow$) Let $X$ take values $x_1,\ldots,x_m$. The atoms of $\sigma(X)$ are the level sets $\{X=x_j\}$, so by the constancy-on-atoms observation of Worked Example 1.9, a $\sigma(X)$-measurable $Y$ is constant on each $\{X=x_j\}$, say with value $c_j$. Define $g(x_j) := c_j$ and $g := 0$ elsewhere; then $Y = g(X)$ and $g$ is measurable, being a finitely-valued function differing from $0$ only on a finite set of points. ∎

This is why **supervised learning and conditional expectation are the same subject.** Chapter 6 defines $\mathbb{E}[Y\mid X]$ as the best $\sigma(X)$-measurable approximation to $Y$; Doob–Dynkin says that means the best *function of $X$*; and "find the function of the features that best predicts the label" is exactly what a supervised learner does. Regression is projection onto a $\sigma$-algebra.

> **Self-check 1.11**
> (a) If $Y = X^3$, how do $\sigma(X)$ and $\sigma(Y)$ compare?
> (b) If $Y=X^2$ and $X$ takes values $\pm1,\pm2$, is $\sigma(Y)\subseteq\sigma(X)$? Is the reverse true?
> (c) What is $\sigma(X)$ when $X$ is constant?

---

## 1.12 Independence

> **Definition 1.26.** Sub-$\sigma$-algebras $\mathcal{G}_1,\ldots,\mathcal{G}_n\subseteq\mathcal{F}$ are **independent** if for every choice $A_i\in\mathcal{G}_i$,
> $$\mathbb{P}\Big(\bigcap_{i=1}^n A_i\Big) = \prod_{i=1}^n\mathbb{P}(A_i).$$
> Random variables $X_1,\ldots,X_n$ are **independent** if $\sigma(X_1),\ldots,\sigma(X_n)$ are. An *infinite* family is independent if every finite subfamily is.

Stating it via $\sigma$-algebras makes the meaning clear: **independence means the information in one collection tells you nothing about the answers in another.** By §1.7.3 this is equivalent, when probabilities are positive, to $\mathbb{P}(A\mid B) = \mathbb{P}(A)$: conditioning changes nothing.

Note that Definition 1.26 requires the product identity for **all** sub-collections, since taking $A_i = \Omega$ for the indices you wish to ignore recovers every sub-product. Three cautions follow.

**Caution 1: pairwise independence does not imply joint independence.** Let $X,Y$ be independent fair bits and $Z := X\oplus Y$ (XOR). Every pair is independent — for instance $\mathbb{P}(X=1,Z=1) = \mathbb{P}(X=1,Y=0) = \tfrac14 = \mathbb{P}(X=1)\mathbb{P}(Z=1)$ — yet
$$\mathbb{P}(X=1,Y=1,Z=1) = 0 \ne \tfrac18,$$
since $X=Y=1$ forces $Z=0$. Any two of the three determine the third: maximal dependence, invisible pairwise. The practical lesson is that inspecting a correlation matrix cannot certify independence.

**Caution 2: a coincidental match on one pair proves nothing.** Worked Example 1.12 showed $\mathbb{P}(X=1,Y=7) = \mathbb{P}(X=1)\mathbb{P}(Y=7)$ exactly, for variables that are visibly dependent. Independence is a universally quantified statement, and a single verified instance is not evidence.

**Caution 3: independence is strictly stronger than zero correlation.** Let $X\sim\mathcal{N}(0,1)$, $Y = X^2$. Then $\operatorname{Cov}(X,Y) = \mathbb{E}[X^3]-\mathbb{E}[X]\mathbb{E}[X^2] = 0-0 = 0$, so they are uncorrelated — yet $Y$ is a deterministic function of $X$. Correlation sees only linear structure. This is why mutual information (Chapter 11) is the right general dependence measure, and why "decorrelated" features (PCA, whitening) are not "disentangled" ones.

### 1.12.1 Conditional independence

One more definition, because it is the backbone of probabilistic modelling. I state it here only in the form this chapter can honestly support: conditioning on the *values of a discrete variable*, so that Definition 1.10 applies. (Conditioning on a continuous variable requires the machinery of Chapter 6, for the reason given in §1.7.4.)

> **Definition 1.27.** Events $A$ and $B$ are **conditionally independent given $C$** (with $\mathbb{P}(C)>0$) if
> $$\mathbb{P}(A\cap B\mid C) = \mathbb{P}(A\mid C)\,\mathbb{P}(B\mid C).$$
> For a **discrete** random variable $Z$, random variables $X$ and $Y$ are conditionally independent given $Z$ — written $X\perp Y\mid Z$ — if for every value $z$ with $\mathbb{P}(Z=z)>0$ and all Borel sets $A,B$,
> $$\mathbb{P}(X\in A,\ Y\in B\mid Z=z) = \mathbb{P}(X\in A\mid Z=z)\,\mathbb{P}(Y\in B\mid Z=z).$$
> The general case ($Z$ arbitrary) is defined in Chapter 6 via conditional expectation and agrees with this one when $Z$ is discrete.

Conditional independence is **neither implied by nor implies** ordinary independence — a fact worth internalizing with two examples. Two coin flips are independent but *not* conditionally independent given their sum (knowing the sum is 1, learning the first flip determines the second). Conversely, two symptoms of a single disease may be dependent marginally but conditionally independent given disease status (Exercise A10 constructs this).

This is exactly the structure that **naive Bayes** classifiers assume: features are conditionally independent given the label, so $p(x_1,\ldots,x_d\mid y) = \prod_j p(x_j\mid y)$, which makes the model tractable even though the features are strongly dependent marginally. It is also the content of **d-separation** in graphical models, and the reason latent-variable models are useful: a latent $Z$ can *explain away* dependence among observables, making them conditionally independent given $Z$ even when they are marginally entangled.

### 1.12.2 Where independence enters machine learning

The standing assumption is that your data $(x_1,y_1),\ldots,(x_n,y_n)$ are independent and identically distributed draws from an unknown $P$. That is what makes the log-likelihood a **sum**:
$$\log \mathbb{P}(\text{data}) = \sum_{i=1}^n\log p(x_i,y_i),$$
and *that* is why your loss is an average over the training set, and why a minibatch gradient is an unbiased estimate of the full gradient. Chapter 2 expresses this precisely with product measures.

Know where the assumption fails, because much current research lives there: time series and language (adjacent tokens are dependent — which is why autoregressive models use the chain rule of §1.7.1 instead of an independence assumption), graph data, active learning (you chose which points to label), reinforcement learning (the policy shapes the data distribution), and any deployed system whose predictions influence its future inputs.

> **Self-check 1.12**
> (a) If $A$ and $B$ are independent, are $A^c$ and $B$ independent?
> (b) Can an event be independent of itself?
> (c) If $X\perp Y$, must $f(X)\perp g(Y)$ for measurable $f,g$?
> (d) Give two variables that are conditionally independent given a third but not independent.

---

## 1.13 "Almost surely"

> **Definition 1.28.** A property holds **almost everywhere** ($\mu$-a.e.), or **almost surely** (a.s.) when the measure is a probability, if the set where it fails is contained in a measurable set of measure zero.

So "$X = Y$ a.s." means $\mathbb{P}(X\ne Y) = 0$: the two agree except on a negligible set.

**A.s. equality is the right notion of "the same random variable."** If $X = Y$ a.s. then $P_X = P_Y$ — indeed for any Borel $B$,
$$|P_X(B) - P_Y(B)| = |\mathbb{P}(X\in B) - \mathbb{P}(Y\in B)| \le \mathbb{P}(X\ne Y) = 0,$$
since the two events differ only inside $\{X\ne Y\}$. So no distributional statement can distinguish a.s.-equal variables. (The converse fails: Self-check 1.9(b) gives variables with the same law that are never equal.) A caution, though: a.s. equal variables can generate *different* $\sigma$-algebras — Exercise A6 has an example — so "the same" depends on whether you mean distributionally or informationally.

**An honesty note about infinite sequences.** Twice now — Common Confusion 1.4 and the Cantor construction of §1.10.3 — I have casually said "flip a coin infinitely often." That phrase presumes a probability space $(\{H,T\}^{\mathbb{N}}, \mathcal{F}, \mathbb{P})$ carrying infinitely many independent fair flips, and the *existence* of such a measure is a genuine theorem (the **Kolmogorov extension theorem**, or the Ionescu-Tulcea theorem), not a triviality: one must check that consistently specifying all finite-dimensional probabilities pins down a countably additive measure on the infinite product. We take it on faith here, it is now a row in §1.14, and Exercise C1's infinite sequence of minibatches leans on the same theorem. This is a good example of the ledger discipline: the most innocent-sounding sentence in the chapter ("flip forever") conceals its deepest borrowed theorem.

Three appearances of "almost surely" worth flagging now.

*Densities are only defined up to a.e. equivalence* (Chapter 2). Change a density at finitely many points and it is the same distribution. So "the density at $x$" is not physically meaningful, and comparing two models' density values at a single point is meaningless.

*The ReLU kink is harmless — usually.* ReLU is non-differentiable at $0$, yet nobody worries, because if the pre-activation has a continuous distribution it equals exactly zero with probability zero. Note the hedge: this **fails** under quantization, with initializations producing exact zeros, or with inputs that are exactly zero (padding, sparse features, masked tokens). A measure-zero argument is only as good as the continuity assumption behind it, and that assumption is often false in practice. This is a good example of a correct theoretical argument being misapplied.

*Convergence theorems come in a.s. flavours.* The strong law of large numbers says the sample mean converges to the true mean almost surely — for *almost every* infinite data sequence, not every one. The standard tool for establishing such statements is the Borel–Cantelli lemma, which you will prove in Exercise A9 using nothing but continuity from above and the union bound. Chapter 7 develops this properly.

Finally, keep Common Confusion 1.4 in mind: **a measure-zero set need not be empty, and probability zero is not impossibility.**

---

## 1.14 Results used but not proved

The preface promised no unexplained appeals to theorems. Here is the complete list of what this chapter borrows, so you always know the size of your debt. Knowing which load-bearing facts you have taken on faith is a research skill in itself.

| Result | Used in | Where to find a proof |
|---|---|---|
| **Existence and uniqueness of Lebesgue measure** on $\mathcal{B}(\mathbb{R}^d)$ — the unique translation-invariant measure with $\lambda_d([0,1]^d)=1$ (local finiteness follows, §1.6) | §1.6; existence of $\mathrm{Unif}[0,1]$ in §1.10.2 | Billingsley §12; Durrett Appendix A (Carathéodory extension) |
| **Dynkin's $\pi$–$\lambda$ theorem**, hence Theorem 1.20 (uniqueness on a generating $\pi$-system) | §1.10 | Durrett Thm A.1.5; Williams Ch. A1 |
| **Kolmogorov extension theorem** (equivalently Ionescu-Tulcea): a consistent family of finite-dimensional distributions extends to a measure on the infinite product — needed for "infinitely many independent draws" | §1.13 (infinite coin flips, Cantor construction), Exercise C1 (infinite minibatch sequence) | Durrett §2.1 and Appendix A; Kallenberg Ch. 8 |
| **$\lvert\mathcal{B}(\mathbb{R})\rvert = 2^{\aleph_0}$**; existence of analytic non-Borel sets | §1.4.1 | Transfinite induction on the Borel hierarchy; Srivastava, *A Course on Borel Sets* |
| **Doob–Dynkin (Lemma 1.25) in general** — proved here only for finitely-valued $X$ | §1.11 | Kallenberg, *Foundations of Modern Probability*, Lemma 1.14 (3rd ed. numbering may differ) |
| **$\mathcal{B}(\mathbb{R}^2) = \sigma(\text{rectangles})$** | §1.9.3 | Follows from second countability; Durrett §1.1 |
| **A $C^1$ image of $\mathbb{R}^k$ in $\mathbb{R}^d$ with $k<d$ is $\lambda_d$-null** | §1.9.4, §1.10.4, A11 | Sketched in the solution to A11; any geometric measure theory text (e.g. Federer) |
| **Lebesgue's decomposition theorem** (discrete + absolutely continuous + singular continuous) | §1.10.3–1.10.4 | Chapter 2; Billingsley §31 |
| **Integrating over a null set gives zero** | Worked Example 1.13, A11 | Chapter 2, immediately after the integral is defined |
| **The Axiom of Choice** (to build the Vitali set) | §1.5 | Taken as an axiom |

Everything else in this chapter is proved where it is stated.

---

## 1.15 Chapter summary and score card

### The score card

§1.1 opened with four failures. Here is exactly where each stands.

| Failure | Status after Chapter 1 | Resolved by |
|---|---|---|
| **1.** Densities exceeding 1; probability of a point is 0 | Open. We now know why it is subtle (a density cannot be a probability), but we cannot yet say what it *is*. | Chapter 2, §Radon–Nikodym |
| **2.** ReLU output is neither discrete nor continuous | **Half-resolved.** §1.10 describes it completely via its CDF, with the atom visible as a jump, and §1.10.3 names the three-way classification it belongs to. What is missing is a *single object* (like a density) valid across all cases. | Chapter 2 |
| **3.** Swapping $\nabla$ and $\mathbb{E}$ | Open. We have not yet built the integral. | Chapter 2, §Dominated Convergence |
| **4.** Constant GAN loss | **Diagnosed.** §1.9.4 proves the generator's support has Lebesgue measure zero and hence no density, and §1.10.4 classifies the output law exactly. What remains is showing that this makes JS divergence constant. | Chapter 2 |

Two of four substantially advanced, and — more importantly — you now have the vocabulary in which the remaining two are stated. Chapter 2 closes all four.

### What to remember

1. A **probability space** $(\Omega,\mathcal{F},\mathbb{P})$ has three parts: what could happen, what you can ask, how likely.
2. A **$\sigma$-algebra is information** — the questions you can answer. On finite spaces it is exactly a partition, and its atoms are the finest distinctions available. Finer partition = more information; $k$ atoms gives $2^k$ events.
3. A **filtration** $\mathcal{F}_1\subseteq\mathcal{F}_2\subseteq\cdots$ is information accumulating over time; this is how SGD analysis is set up.
4. **Not every subset can be assigned a probability** (§1.5). The $\sigma$-algebra is not a compromise but the precise statement of what is measurable.
5. **Countable additivity is exactly continuity under increasing limits** (Proposition 1.9) — which is why every asymptotic theorem in statistics needs it.
6. The **union bound** needs no assumptions, is tight for disjoint events, nearly tight for independent rare events, and vacuous once $\sum\mathbb{P}(A_n)\ge1$. It is the source of $\log|\mathcal{H}|$ in generalization bounds.
7. **Conditional probability renormalizes onto an event**, and $\mathbb{P}(\cdot\mid B)$ is itself a probability measure, so every theorem transfers. The **chain rule** is why autoregressive models factorize; **Bayes' rule** is posterior $\propto$ likelihood $\times$ prior; and **base rates dominate** when events are rare, which is why precision and recall are reported separately.
8. A **random variable is a deterministic function** on $\Omega$; all randomness lives in $\omega$. Measurability says every question about $X$ is answerable, and on finite spaces it means "constant on each atom."
9. **Countability is load-bearing.** Suprema over countable families stay measurable; over uncountable families they need not. This is a real gap in uniform-convergence arguments: continuity-plus-separability closes it for continuous losses, while the 0–1 loss genuinely needs empirical-process conditions.
10. A **distribution is a pushforward** $P_X = \mathbb{P}\circ X^{-1}$. Once you have it, $\Omega$ can be forgotten.
11. The **support** is the smallest closed set of full measure — about where mass *isn't*, not where it's concentrated.
12. **Joint laws carry more than marginals.** Independence *means* the joint law is a product measure, and a coincidental product identity on one pair of events proves nothing.
13. **CDFs determine distributions** (Theorem 1.20); jumps are atoms; and the quantile function $F^{-1}$ turns a uniform draw into any distribution — the germ of every sampler.
14. Every distribution on $\mathbb{R}$ is **discrete + absolutely continuous + singular continuous**. The Cantor distribution is a genuine third-kind example; a smooth low-dimensional generator's output is its high-dimensional cousin, and a **ReLU generator can additionally produce atoms** — non-constancy is not enough to rule them out.
15. **Every deep generative model is a pushforward** of a simple latent. When $k<d$ the support has Lebesgue measure zero, so no density exists — the root cause of GAN pathologies.
16. $\sigma(X)$ is the information in $X$; by **Doob–Dynkin**, "determined by $X$" = "a function of $X$." This bridges to conditional expectation and hence supervised learning.
17. **Independence** means informational irrelevance; **conditional independence** is neither weaker nor stronger, and is the backbone of naive Bayes, graphical models, and latent-variable models.
18. Even "flip a coin forever" hides a theorem (**Kolmogorov extension**). Track your debts; §1.14 shows how.

### Before you continue

Answer without looking back:

1. Define a $\sigma$-algebra in one sentence, without the word "closed."
2. On a finite sample space, what are the atoms, and what do they mean?
3. Why is countable additivity, rather than finite additivity, the axiom?
4. A test is 99% sensitive and 95% specific for a disease with prevalence 0.5%. Roughly what is $\mathbb{P}(\text{disease}\mid\text{positive})$, and why is it not near 99%?
5. Why is a random variable a function, and where is the randomness?
6. What is a pushforward, and why does it let us stop talking about $\Omega$?
7. Why does a smooth generator with a 100-dimensional latent have no density on $\mathbb{R}^{3072}$? Can its output distribution have atoms, and what would produce them?
8. Why might $\sup_{h\in\mathcal{H}}|\hat R_n(h)-R(h)|$ fail to be a random variable, what rescues it for continuous losses, and why is the 0–1 loss harder?
9. How do you turn a uniform random number into a sample from an arbitrary CDF $F$?
10. Give an event of probability zero that can nonetheless occur.

### Looking ahead to Chapter 2

Chapter 2 builds the integral. We construct the Lebesgue integral from scratch — with worked computations, not just a definition — and use it to define expectation once, for all distributions at once. Then come the three convergence theorems governing when a limit and an integral may be exchanged, which is what licenses or forbids differentiating through an expectation (Failure 3). Finally the Radon–Nikodym theorem reveals what a density actually is, unifying pmfs and pdfs into a single object and closing Failures 1, 2, and 4.

---

## 1.16 Corrections from revision 2

For transparency, and because knowing what kinds of errors creep into technical writing is useful:

1. **§1.10 (now §1.10.4) claimed that a *non-constant* generator produces no atoms.** False, and the counterexample is not exotic: $G(z) = (\max(0,z_1), 0)$ is non-constant yet places mass $1/2$ at a single point — and it is literally a ReLU. The revised Proposition 1.23 states the correct condition (level sets of $\nu$-measure zero) and a usable sufficient condition (a.e. full-rank Jacobian). The error is instructive: "non-constant" rules out only the *globally* degenerate case, while atoms come from *local* flatness — the very mechanism this chapter's own Failure 2 is about. When a hedge is added under time pressure, check it against the examples already in the document.
2. **§1.6 claimed local finiteness was a needed hypothesis for the uniqueness of Lebesgue measure.** It is implied by translation invariance plus the normalization $\lambda_d([0,1]^d)=1$ (cover any compact set by finitely many unit-cube translates). Revision 2's "fix" of revision 1 was an overcorrection; the moral is that corrections need the same scrutiny as first drafts.
3. **§1.14 claimed completeness while the chapter silently used the Kolmogorov extension theorem** every time it said "flip a coin infinitely often" or "the infinite sequence of minibatches." Now listed, with a note in §1.13.
4. **§1.8.3 claimed the separability argument "covers essentially all of deep learning."** It covers losses continuous in the parameters. The 0–1 loss of classification — the loss most generalization bounds are about — is not continuous in the parameters, and needs the pointwise-measurability machinery of empirical process theory. The section now says so.
5. Smaller repairs: conditional independence is now defined only for discrete conditioning variables (the general case belongs to Chapter 6); "the joint law contains *strictly* more information than the marginals" weakened to "in general more" (a degenerate marginal makes them equivalent); "every set you will ever write down is Borel" replaced by an accurate statement with the analytic-set caveat, which also foreshadows §1.8.3; a truncated sentence in Common Confusion 1.5 completed; figures renumbered in order of appearance and converted to plain-text so they render everywhere; the cross-reference to the countable-additivity lemma corrected.

---
---

# Exercises

Attempt these before reading solutions; the solutions explain the *thinking* and are much less useful without a genuine attempt. Difficulty: ★ routine, ★★ needs an idea, ★★★ challenging.

## Part A — Theory

**A1 (★).** Let $\Omega = \{1,2,3,4\}$. (a) Write out $\sigma(\{\{1,2\}\})$ and its atoms. (b) Write out $\sigma(\{\{1\},\{2\}\})$ and its atoms. (c) Describe the information each represents and say which is finer.

**A2 (★).** (a) Prove the intersection of an arbitrary family of $\sigma$-algebras on $\Omega$ is a $\sigma$-algebra. (b) Show by counterexample that the *union* of two need not be, and interpret the failure informationally.

**A3 (★★).** Prove $\mathcal{B}(\mathbb{R}) = \sigma(\{(-\infty,q] : q\in\mathbb{Q}\})$. Explain why countability of $\mathbb{Q}$ matters and what it implies about how many Borel sets there are.

**A4 (★).** Prove the two preimage identities of Lemma 1.14, then explain in a paragraph how they yield the lemma.

**A5 (★★).** *(Bayes and base rates.)* A spam filter flags 98% of spam and misflags 2% of legitimate mail. Suppose 30% of incoming mail is spam. (a) Compute $\mathbb{P}(\text{spam}\mid\text{flagged})$. (b) Now suppose you deploy in an enterprise where only 2% of mail is spam. Recompute. (c) Your product manager wants precision above 90% at the enterprise base rate. What false-positive rate is required, holding sensitivity at 98%? (d) Explain why "our filter is 98% accurate" is not a useful claim, referencing which conditional probability it describes.

**A6 (★★).** Let $\Omega = [0,1]$ with Borel sets and $\mathbb{P} = \lambda$, and $X(\omega) = \mathbb{1}[\omega\in\mathbb{Q}]$. (a) Show $X$ is measurable. (b) Compute $P_X$. (c) Compute $\sigma(X)$ and describe the information. (d) Exhibit a random variable $Y$ with $X = Y$ a.s. but $\sigma(Y)\ne\sigma(X)$, and explain what this shows about the two senses of "the same random variable."

**A7 (★★).** Let $F$ be a CDF. (a) Show $F$ has at most countably many discontinuities. (b) Show $\mathbb{P}(X\in D)$, with $D$ the discontinuity set, can be any value in $[0,1]$, exhibiting examples achieving $0$, $1$, and $1/2$.

**A8 (★★).** Let $X$ be a random variable and $g$ measurable. (a) Show $\sigma(g(X))\subseteq\sigma(X)$. (b) Give a $g$ making the inclusion strict, and one making it equality. (c) Characterize which $g$ give equality when $X$ takes finitely many values.

**A9 (★★★).** *(First Borel–Cantelli lemma.)* Let $A_1,A_2,\ldots$ be events with $\sum_n\mathbb{P}(A_n)<\infty$. Prove $\mathbb{P}(\text{infinitely many }A_n\text{ occur}) = 0$. *(Hint: "infinitely many occur" $=\bigcap_m\bigcup_{n\ge m}A_n$. Use continuity from above and the union bound.)* Then explain why this is the natural tool for almost-sure convergence statements. *(For orientation, not for proof here: the **second** Borel–Cantelli lemma says that if the $A_n$ are **independent** and $\sum_n\mathbb{P}(A_n)=\infty$, then infinitely many occur with probability 1 — so for independent events the dichotomy is complete. Chapter 7 proves it.)*

**A10 (★★).** Construct three random variables pairwise but not jointly independent, verifying all required identities. Then construct two uncorrelated but dependent variables, and verify. Finally, give two variables that are conditionally independent given a third but not independent.

**A11 (★★★).** Let $G:\mathbb{R}^k\to\mathbb{R}^d$ be $C^1$ with $k<d$, and $Z$ any distribution on $\mathbb{R}^k$. Argue that $P = G_\#(\text{law of }Z)$ satisfies $P(N)=1$ for a Borel $N$ with $\lambda_d(N)=0$, and conclude $P$ has no density w.r.t. $\lambda_d$. *(You may assume a $C^1$ image of $\mathbb{R}^k$ in $\mathbb{R}^d$ is $\lambda_d$-null when $k<d$; a proof sketch is in the solution.)*

**A12 (★★).** *(Uncountable suprema.)* (a) Verify that $\{\sup_n f_n \le a\} = \bigcap_n\{f_n\le a\}$ for a countable family, and explain why this proves Proposition 1.15(4). (b) Using a non-measurable $V\subseteq[0,1]$, construct an uncountable family of measurable functions whose supremum is non-measurable. (c) Let $\phi:\mathbb{R}^p\to\mathbb{R}$ be continuous. Prove $\sup_{w\in\mathbb{R}^p}\phi(w) = \sup_{w\in Q}\phi(w)$ for any countable dense $Q\subseteq\mathbb{R}^p$, and conclude the supremum is measurable when $\phi(w) = |\hat R_n(h_w) - R(h_w)|$ is continuous in $w$. (d) Exhibit a simple one-parameter classifier for which $w\mapsto\hat R_n(h_w)$ under the 0–1 loss is *not* continuous, and explain why (c) therefore does not apply directly to classification error.

**A13 (★★).** *(Atoms of pushforwards.)* Let $P = G_\#\nu$ for measurable $G:\mathbb{R}^k\to\mathbb{R}^d$. (a) Show $P$ has an atom at $x_0$ iff $\nu(G^{-1}(\{x_0\}))>0$. (b) Give a non-constant continuous $G$ and absolutely continuous $\nu$ for which $P$ has an atom. (c) Show that if $\nu\ll\lambda_k$ and every level set of $G$ has $\lambda_k$-measure zero, then $P$ has no atoms. (d) Explain in one paragraph why ReLU networks are the practically relevant case of (b).

## Part B — Computational labs

Use NumPy. Report numbers and interpret them in one or two sentences — the interpretation is the point.

**B1 (★).** Verify Worked Example 1.11. Sample $10^6$ points $\omega\sim\mathrm{Unif}[-1,1]$, set $y=\omega^2$, compare $\hat{\mathbb{P}}(Y\le b)$ to $\sqrt b$ for $b\in\{0.25,0.5,0.81\}$. Histogram $y$ and overlay $\frac1{2\sqrt b}$. Explain why the density blows up near $0$ while total probability stays 1.

**B2 (★★).** Verify Worked Example 1.14. Sample $10^6$ values $z\sim\mathcal{N}(0,1)$, set $a = \max(0,z)$, estimate $\mathbb{P}(A=0)$, $F_A(t)$ for $t\in\{0.5,1,2\}$, and $\mathbb{E}[A]$ (true value $1/\sqrt{2\pi}$; you are only checking, not deriving). Plot the empirical CDF and identify the jump.

**B3 (★★).** Union-bound tightness. For $m$ events of probability $p$, compute the true union probability under independence, identity, and disjointness, and compare to $mp$, for $(m,p)\in\{(10,0.01),(100,0.01),(1000,0.001)\}$. Then plot (true)/(bound) against $mp$ for independent events, $mp\in[0.01,5]$. Where does the bound die, and what does that mean for large hypothesis classes?

**B4 (★★).** Demonstrate Worked Example 1.13. Sample $2\times10^5$ values $t\sim\mathcal{N}(0,1)$, map through $G(t) = (\cos t,\sin t)$. (a) Verify $\|G(t)\|=1$ throughout. (b) Estimate the fraction in $\{\|x\|<0.99\}$ and compare with that region's share of the disc's *area*. (c) Add noise, $\tilde G(t) = G(t)+\varepsilon$, $\varepsilon\sim\mathcal{N}(0,\sigma^2I_2)$, and repeat (b) for $\sigma\in\{0.001,0.01,0.1\}$. Explain what this says about why VAEs add output noise. (d) Now use $G_2(t) = (\max(0,t), 0)$ instead and estimate $\hat{\mathbb{P}}(G_2(t) = (0,0))$. Which part of §1.10.4 have you just demonstrated?

**B5 (★★).** *(Base rates by simulation.)* Simulate $2\times10^6$ individuals with prevalence $0.5\%$, sensitivity $99\%$, specificity $95\%$. (a) Estimate $\mathbb{P}(D\mid+)$ and compare to the analytic $0.0905$. (b) Plot $\mathbb{P}(D\mid+)$ against prevalence on a log scale from $10^{-5}$ to $0.5$. (c) Plot $\mathbb{P}(D\mid+)$ against specificity from $0.90$ to $0.9999$ at fixed prevalence $0.005$, and mark where it crosses $0.5$. (d) A colleague proposes raising sensitivity from 99% to 99.9%. Quantify the improvement in precision and compare it to raising specificity from 95% to 99%. Which investment is better, and why?

**B6 (★★).** *(Inverse-transform sampling.)* (a) Implement $F^{-1}$ for the exponential distribution ($F(a) = 1-e^{-a}$ for $a\ge0$), draw $10^6$ samples via $F^{-1}(U)$, and compare the empirical mean and $\hat F(1)$ to the true values $1$ and $1-e^{-1}\approx0.632$. (b) Implement the generalized inverse for the ReLU CDF of Worked Example 1.14 — $F_A^{-1}(u) = 0$ for $u\le1/2$ and $\Phi^{-1}(u)$ for $u>1/2$ — sample $10^6$ values, and check that you reproduce $\hat{\mathbb{P}}(A=0)\approx0.5$ and $\hat F_A(1)\approx0.841$. (c) Explain, in terms of flat stretches and jumps of $F$, why the generalized inverse maps an interval of $u$-values to the atom and thereby recreates it.

## Part C — Paper-style derivations

Write these as appendix lemmas: statement, proof, and a paragraph on why it matters.

**C1.** *(Formalizing the SGD setup.)* Let $\theta_1$ be fixed and $\theta_{t+1} = \theta_t-\eta_t g_t$, where $g_t$ is computed from a minibatch $B_t$ drawn independently each step. (You may assume a probability space carrying the infinite i.i.d. sequence $B_1, B_2, \ldots$ exists — Kolmogorov extension, §1.14.) (a) Define a filtration $\{\mathcal{F}_t\}$ making precise "everything known just before step $t$," and verify it increases. (b) State as a measurability claim that $\theta_t$ is determined by the history, and prove it. (c) Write the unbiasedness assumption as a conditional expectation statement and explain what it asserts. (d) Explain why $\mathbb{E}[g_t] = \nabla F(\theta_t)$ *without* conditioning is not the right assumption, attending to the fact that $\theta_t$ is random.

**C2.** *(Why GANs cannot be trained by maximum likelihood.)* Let $P_{\text{data}}$ be a distribution on $\mathbb{R}^d$, $P_\theta = (G_\theta)_\#\mathcal{N}(0,I_k)$, $G_\theta$ $C^1$, $k<d$. (a) Prove $P_\theta$ has no density w.r.t. $\lambda_d$. (b) Explain precisely what goes wrong in writing $\sum_i\log p_\theta(x_i)$. (c) If $P_{\text{data}}$ is supported on a $k'$-dimensional surface with $k'<d$, argue the supports are generically disjoint and say what happens to any divergence built from density ratios. (d) Describe the three architectural responses used by normalizing flows, VAEs, and Wasserstein GANs.

---
---

# Appendix A — Self-check answers

**1.2** (a) $\Omega = \{2,3,\ldots,12\}$. (b) No — the sum $7$ is consistent with any first die, and $\Omega$ records nothing else. (c) $\Omega = \{1,\ldots,6\}^2$; then "first die 3" is $\{(3,j):j=1,\ldots,6\}$.

**1.3** (a) **Yes.** Check: $\Omega = \{1,2,3\}$ present ✓. Complements: $\emptyset^c = \Omega$ ✓, $\{1\}^c = \{2,3\}$ ✓, $\{2,3\}^c = \{1\}$ ✓, $\Omega^c = \emptyset$ ✓. Unions: $\{1\}\cup\{2,3\} = \Omega$ ✓, and unions with $\emptyset$ or $\Omega$ stay inside ✓. Atoms are $\{1\}$ and $\{2,3\}$, so $2^2 = 4$ members — matching the count ✓. (b) **No.** $\{1\}\cup\{2\} = \{1,2\}\notin\mathcal{B}$, so closure under union fails. (Also $\{1\}^c = \{2,3\}\notin\mathcal{B}$.) Note $\mathcal{B}$ has the right *size* for a $\sigma$-algebra with two atoms, which is why you must check the axioms rather than count. (c) Three atoms, $2^3 = 8$. (d) **No.** "Exactly two heads" $=\{HHT,HTH,THH\}$ contains $HHT$ but not $HHH$, splitting the atom $HH\!*$. Sensible: after two flips you cannot know the total.

**1.4** (a) $[a,b) = [a,\infty)\cap(-\infty,b)$, an intersection of a closed and an open set, both Borel. (b) Yes: $\mathbb{Q}\cap[0,1]$ is countable hence Borel; take the complement within $[0,1]$. (c) **False.** $\sigma(\mathcal{C})$ generally requires *iterating* the operations — unions, then complements, then unions of those, transfinitely; one pass is not enough. This is why Definition 1.5 works "from above," as an intersection, rather than "from below" by construction.

**1.6** (a) $\mathbb{P}(A\cap B)\ge\mathbb{P}(A)+\mathbb{P}(B)-1 = 0.3$, from inclusion–exclusion and $\mathbb{P}(A\cup B)\le1$. (b) $\mu(\{1,3\}) = 7$ by additivity; not a probability measure since $\mu(\Omega) = 7$. ($\mu(\{2\})=0$ is perfectly legal.) (c) $\delta_0 = 0.05/40 = 0.00125$, i.e. 99.875% per hypothesis.

**1.7** (a) Yes. $\mathbb{P}(A\mid B) = \mathbb{P}(A)$ is equivalent to $\mathbb{P}(A\cap B) = \mathbb{P}(A)\mathbb{P}(B)$, which is symmetric; divide by $\mathbb{P}(A)$ instead. (b) No. By the law of total probability, $\mathbb{P}(A)$ is a weighted average of $\mathbb{P}(A\mid B)$ and $\mathbb{P}(A\mid B^c)$ with positive weights, so it lies between them; both cannot exceed it. (c) It $\to0$. With sensitivity and specificity fixed, false positives from the healthy majority eventually swamp true positives entirely.

**1.8** (a) $Z = -1$ on $\{1,3,5\}$, $+1$ on $\{2,4,6\}$ — constant on each atom, so **yes**. (b) $X^{-1}(\{2\}) = \{HHT,HTH,THH\}$; $X^{-1}([1.5,10]) = \{X\in\{2,3\}\} = \{HHT,HTH,THH,HHH\}$. (c) Both yes: $x\mapsto x^2$ and $x\mapsto\sqrt{|x|}$ are continuous hence Borel, and compositions of measurable maps are measurable. (d) It is a **countable** union of events, hence an event by axiom (3). For an uncountable family the union is uncountable, and axiom (3) does not apply — see A12(b) for an actual failure.

**1.9** (a) $X$ is $1$ on $\{1,4\}$, $2$ on $\{2,5\}$, $0$ on $\{3,6\}$, so $P_X$ is uniform on $\{0,1,2\}$. (b) **No.** Same law $\ne$ same function. Take $\Omega = \{H,T\}$ uniform, $X = \mathbb{1}[H]$, $Y = \mathbb{1}[T]$: both Bernoulli$(1/2)$, never equal. (c) Zero — the support is the closure of a $C^1$ image of $\mathbb{R}^{100}$, which is $\lambda_{3072}$-null (closure adds only a boundary that is itself contained in limits of null compact images; more simply, the *image* already carries probability 1 and is null, and that is all the no-density argument needs). (d) $\{0,1,2,3\}$: a finite set, which is closed and has full measure, and no smaller closed set does.

**1.10** (a) $0$ for $a<0$; jumps to $1/3$ at $a=0$; flat at $1/3$ on $[0,1)$; rises linearly from $1/3$ to $1$ across $[1,2]$; then $1$. (b) $\mathbb{P}(X=3) = F(3)-F(3^-)$, the jump height. (c) No — right-continuity is forced by continuity from above (Proposition 1.19). Defining $F(a) = \mathbb{P}(X<a)$ would give a left-continuous function instead; the convention matters because the jump then sits on the other side, and $\mathbb{P}(a<X\le b) = F(b)-F(a)$ would change form. (d) For $a\ge0$: $\mathbb{P}(-\ln(1-U)\le a) = \mathbb{P}(U\le1-e^{-a}) = 1-e^{-a}$. Exponential with rate 1 — you have just derived the standard exponential sampler.

**1.11** (a) Equal: $x\mapsto x^3$ is a measurable bijection with measurable inverse, so each is a function of the other. (b) $\sigma(Y)\subseteq\sigma(X)$ ✓ since $Y$ is a function of $X$. Reverse fails: $Y$ cannot distinguish $X=1$ from $X=-1$, so $\{X=1\}\notin\sigma(Y)$. (c) The trivial $\sigma$-algebra $\{\emptyset,\Omega\}$: a constant tells you nothing.

**1.12** (a) Yes: $\mathbb{P}(A^c\cap B) = \mathbb{P}(B)-\mathbb{P}(A\cap B) = \mathbb{P}(B)(1-\mathbb{P}(A)) = \mathbb{P}(A^c)\mathbb{P}(B)$. (b) Only if $\mathbb{P}(A)\in\{0,1\}$, since independence with itself means $\mathbb{P}(A) = \mathbb{P}(A)^2$. (c) Yes: $\sigma(f(X))\subseteq\sigma(X)$ and $\sigma(g(Y))\subseteq\sigma(Y)$, and sub-$\sigma$-algebras of independent $\sigma$-algebras are independent. (d) Let $Z\sim$ Bernoulli$(1/2)$ be a disease indicator and let $X,Y$ be two symptoms, each equal to $Z$ with probability $0.9$ independently given $Z$. Then $X\perp Y\mid Z$ by construction, but marginally they are positively correlated, since both track $Z$. (Exercise A10 verifies the numbers.)

---

# Solutions

## Part A

**A1.** (a) $\sigma(\{\{1,2\}\}) = \{\emptyset,\{1,2\},\{3,4\},\Omega\}$; atoms $\{1,2\},\{3,4\}$. (b) $\sigma(\{\{1\},\{2\}\}) = $ all unions of the atoms $\{1\},\{2\},\{3,4\}$ — eight sets. (c) The first answers only "was the outcome in $\{1,2\}$?"; the second additionally distinguishes 1 from 2 but still cannot separate 3 from 4. The second is strictly finer: every set of (a) is a union of (b)'s atoms.

**A2.** (a) Let $\{\mathcal{G}_i\}_{i\in I}$ be $\sigma$-algebras and $\mathcal{G} = \bigcap_i\mathcal{G}_i$. $\Omega\in\mathcal{G}$ since $\Omega\in$ each. If $A\in\mathcal{G}$, then $A\in\mathcal{G}_i$ for every $i$, so $A^c\in\mathcal{G}_i$ for every $i$, so $A^c\in\mathcal{G}$. Same argument verbatim for countable unions. (The proof works for arbitrary $I$, uncountable included — we are intersecting the *collections*, not the sets.) (b) On $\Omega = \{1,2,3\}$: $\mathcal{G}_1 = \{\emptyset,\{1\},\{2,3\},\Omega\}$ and $\mathcal{G}_2 = \{\emptyset,\{2\},\{1,3\},\Omega\}$. Their union contains $\{1\}$ and $\{2\}$ but not $\{1\}\cup\{2\}$. Informationally: knowing the answers to two separate questionnaires is *more* information than either alone — combining information requires generating, $\sigma(\mathcal{G}_1\cup\mathcal{G}_2)$, not uniting.

**A3.** Let $\mathcal{R} = \sigma(\{(-\infty,q]:q\in\mathbb{Q}\})$. ($\subseteq$ direction of the interesting inclusion) Every open interval is in $\mathcal{R}$: for $a<b$,
$$(a,b) = \Big(\bigcup_{q\in\mathbb{Q},\,q<b}(-\infty,q]\Big)\setminus\Big(\bigcap_{q\in\mathbb{Q},\,q>a}(-\infty,q]\Big)$$
— more transparently, $(-\infty,b) = \bigcup_{q<b}(-\infty,q]$ and $(-\infty,a] = \bigcap_{q>a}(-\infty,q]$ with $q$ rational (density of $\mathbb{Q}$ makes both exact), so $(a,b) = (-\infty,b)\setminus(-\infty,a]\in\mathcal{R}$. Every open $U\subseteq\mathbb{R}$ is a countable union of open intervals with rational endpoints (take, for each rational $q \in U$, a rational-radius interval around it inside $U$; countably many suffice and their union is $U$), so $U\in\mathcal{R}$, hence $\mathcal{B}(\mathbb{R})\subseteq\mathcal{R}$. Conversely each $(-\infty,q]$ is closed, hence Borel, so $\mathcal{R}\subseteq\mathcal{B}(\mathbb{R})$. ∎ Countability matters twice: the unions/intersections above must be countable to stay inside a $\sigma$-algebra, and a countable generating family is what makes the transfinite counting argument give $|\mathcal{B}(\mathbb{R})| = 2^{\aleph_0}$ — strictly fewer than $2^{2^{\aleph_0}}$ subsets, so non-Borel sets must exist even before Vitali exhibits a non-*Lebesgue-measurable* one.

**A4.** $\omega\in X^{-1}(B^c) \iff X(\omega)\in B^c\iff X(\omega)\notin B\iff\omega\notin X^{-1}(B)\iff\omega\in(X^{-1}(B))^c$. And $\omega\in X^{-1}(\bigcup_n B_n)\iff X(\omega)\in\text{some }B_n\iff\omega\in\text{some }X^{-1}(B_n)$. ∎ These identities say preimage is a homomorphism of the set operations. Hence "the sets whose preimages are events" is closed under complement and countable union and contains $S$ — a $\sigma$-algebra. If it contains the generators, minimality of $\sigma(\mathcal{C})$ forces it to contain all of $\mathcal{S}$: measurability checked on generators is measurability, which is Lemma 1.14.

**A5.** Write $S$ = spam, $F$ = flagged; $\mathbb{P}(F\mid S) = 0.98$, $\mathbb{P}(F\mid S^c) = 0.02$.
(a) Prior $0.3$: $\mathbb{P}(S\mid F) = \dfrac{0.98\times0.3}{0.98\times0.3+0.02\times0.7} = \dfrac{0.294}{0.308} = \mathbf{0.9545}$.
(b) Prior $0.02$: $\mathbb{P}(S\mid F) = \dfrac{0.98\times0.02}{0.98\times0.02+0.02\times0.98} = \dfrac{0.0196}{0.0392} = \mathbf{0.500}$. The same filter drops from 95% precision to a coin flip — nothing changed but the base rate.
(c) Require $\dfrac{0.98\times0.02}{0.98\times0.02+\alpha\times0.98}\ge0.9$, i.e. $\dfrac{0.02}{0.02+\alpha}\ge0.9$, giving $\alpha\le0.02/9 = \mathbf{0.00222}$: a nine-fold reduction in false positives.
(d) "98% accurate" conflates $\mathbb{P}(F\mid S)$ and $\mathbb{P}(F^c\mid S^c)$ (or worse, averages them by the base rate). What a user experiences when a message is flagged is $\mathbb{P}(S\mid F)$ — the *reverse* conditional — which depends critically on prevalence, as (a) vs (b) shows. Any accuracy claim silent on base rates is uninterpretable.

**A6.** (a) Preimages of $X$: $\emptyset$, $\mathbb{Q}\cap[0,1]$, its complement, $[0,1]$ — all Borel (countable set, complement of one). Formally, for Borel $B$: the preimage depends only on whether $B$ contains $0$ and whether it contains $1$. (b) $\mathbb{P}(X=1) = \lambda(\mathbb{Q}\cap[0,1]) = 0$, so $P_X = \delta_0$. (c) $\sigma(X) = \{\emptyset, \mathbb{Q}\cap[0,1], [0,1]\setminus\mathbb{Q}, [0,1]\}$ — the ability to answer exactly one question: "was the draw rational?" (d) Take $Y \equiv 0$. Then $\{X\ne Y\} = \mathbb{Q}\cap[0,1]$ has probability 0, so $X = Y$ a.s.; but $\sigma(Y)$ is trivial while $\sigma(X)$ has four members. Distributionally $X$ and $Y$ are identical (both have law $\delta_0$); informationally they differ on a null set that the $\sigma$-algebra still sees. "Same random variable" is a.s.-equality for every probabilistic purpose, but generated information is not invariant under a.s. modification.

**A7.** (a) The jump at a discontinuity $a$ is $j(a) = F(a)-F(a^-)>0$. For each $n$, the set $D_n = \{a: j(a)>1/n\}$ has at most $n$ elements — distinct jumps are disjoint increments of a function increasing from 0 to 1, so their sum is $\le1$. Then $D = \bigcup_n D_n$ is a countable union of finite sets: countable. ∎ (b) $\mathbb{P}(X\in D) = \sum_{a\in D}j(a)$ is the total atomic mass. Continuous distribution: $0$. Any discrete distribution: $1$. The ReLU output of Worked Example 1.14: $D = \{0\}$ and $\mathbb{P}(X\in D) = 1/2$. ∎

**A8.** (a) A question about $g(X)$ is a question about $X$: $\{g(X)\in B\} = \{X\in g^{-1}(B)\}$, and $g^{-1}(B)$ is Borel by measurability of $g$, so the event is in $\sigma(X)$. (b) Strict: $g(x) = x^2$ with $X$ symmetric — the sign is lost. Equality: $g(x) = x^3$, or any measurable injection with measurable inverse on the range of $X$. (c) For finitely-valued $X$, equality holds iff $g$ is injective on the values $X$ actually takes: injectivity means each level set of $g(X)$ equals a level set of $X$, so the atoms coincide; conversely if $g(x_i) = g(x_j)$ for two attained values with (wlog) $\{X=x_i\}$ nonempty of positive probability or not — regardless, $\{X = x_i\}$ is an event of $\sigma(X)$ that cannot be built from level sets of $g(X)$, since those merge $x_i$ and $x_j$.

**A9.** Let $A := \{\text{infinitely many }A_n\} = \bigcap_{m\ge1}\underbrace{\bigcup_{n\ge m}A_n}_{U_m}$. (An outcome is in every $U_m$ iff for every $m$ some later $A_n$ contains it — iff it is in infinitely many.) The $U_m$ decrease, and $\mathbb{P}(U_1)\le1<\infty$, so continuity from above gives $\mathbb{P}(A) = \lim_m\mathbb{P}(U_m)$. By the union bound, $\mathbb{P}(U_m)\le\sum_{n\ge m}\mathbb{P}(A_n)$, the tail of a convergent series, which $\to0$. Hence $\mathbb{P}(A) = 0$. ∎ Why it is the natural tool: "$X_n\to X$ a.s." is exactly "$\mathbb{P}(|X_n - X|>\varepsilon\text{ infinitely often}) = 0$ for every $\varepsilon>0$." Borel–Cantelli converts that into the checkable analytic condition $\sum_n\mathbb{P}(|X_n-X|>\varepsilon)<\infty$ — probability bounds in, almost-sure statement out, no independence needed. (The second lemma, quoted in the problem, shows the summability condition is also *sharp* for independent events.) Notice the proof used exactly two tools from this chapter, continuity from above and the union bound, and nothing else.

**A10.** *Pairwise not joint:* $X,Y$ independent fair bits, $Z = X\oplus Y$. $Z$ is a fair bit: $\mathbb{P}(Z=1) = \mathbb{P}(X\ne Y) = 1/2$ ✓. $X,Z$ independent: $\mathbb{P}(X=1,Z=1) = \mathbb{P}(X=1,Y=0) = 1/4 = \mathbb{P}(X=1)\mathbb{P}(Z=1)$ ✓, and the other three cells similarly each equal $1/4$ ✓; by symmetry $Y,Z$ too, and $X,Y$ by assumption. Jointly: $\mathbb{P}(X=1,Y=1,Z=1) = 0\ne1/8$ ✗. ∎ *Uncorrelated dependent:* $X\sim\mathcal{N}(0,1)$, $Y=X^2$: $\operatorname{Cov} = \mathbb{E}[X^3] = 0$ by symmetry, yet $Y = X^2$ exactly. (Discrete alternative: $X$ uniform on $\{-1,0,1\}$, $Y = \mathbb{1}[X=0]$; $\mathbb{E}[XY] = 0 = \mathbb{E}[X]\mathbb{E}[Y]$, but $\mathbb{P}(X=0,Y=0)=0\ne\frac13\cdot\frac23$.) *Conditionally independent, not independent:* disease $Z\sim$Bern$(1/2)$; given $Z$, symptoms $X,Y$ independent Bern$(0.9)$ if $Z=1$, Bern$(0.1)$ if $Z=0$. Conditional independence holds by construction. Marginally: $\mathbb{P}(X=1) = \mathbb{P}(Y=1) = \frac12(0.9)+\frac12(0.1) = 0.5$, while $\mathbb{P}(X=1,Y=1) = \frac12(0.81)+\frac12(0.01) = 0.41\ne0.25$. Dependent. ∎

**A11.** Let $M = G(\mathbb{R}^k)$. *Sketch that $\lambda_d(M) = 0$ for $C^1$ $G$, $k<d$:* Write $\mathbb{R}^k = \bigcup_j K_j$ with $K_j = [-j,j]^k$ compact — countably many, so it suffices (union bound) that each $G(K_j)$ is null. On $K_j$, $G$ is Lipschitz with some constant $L$ ($C^1$ on a compact set). Cover $K_j$ by $N\sim(2j/\varepsilon)^k$ cubes of side $\varepsilon$; each image sits inside a ball of radius $L\varepsilon\sqrt k$, of $d$-volume $c_d(L\varepsilon\sqrt k)^d$. Total volume of the cover: $\lesssim \varepsilon^{-k}\cdot\varepsilon^d = \varepsilon^{d-k}\to0$ as $\varepsilon\to0$, using $d>k$. So $\lambda_d(G(K_j)) = 0$. (Images of compact sets are compact hence Borel; $M$ is a countable union of these, hence Borel and null — a convenient bonus that sidesteps worrying about measurability of $M$ itself.) Now $P(M) = \mathbb{P}(G(Z)\in M) = 1$ trivially. If $P$ had a density $p$, then $1 = P(M) = \int_M p\,d\lambda_d = 0$ since $\lambda_d(M) = 0$ — contradiction. ∎ *Why it matters:* this is the precise statement of "a GAN has no likelihood." The argument needs nothing about the architecture except smoothness and $k<d$, so no training procedure and no clever parameterization escapes it within this architecture class.

**A12.** (a) $\omega$ satisfies $\sup_n f_n(\omega)\le a$ iff every $f_n(\omega)\le a$: the sup of a set is $\le a$ iff $a$ upper-bounds the set. Each $\{f_n\le a\}\in\mathcal{F}$, and the intersection is countable, hence in $\mathcal{F}$ by axioms (2)+(3). Since sets $\{sup\le a\}$ for all $a$ generate what is needed (Lemma 1.14 with generators $(-\infty,a]$), $\sup_n f_n$ is measurable. For $\inf$, $\limsup$, $\liminf$: combinations of countable sups and infs. (b) With $V$ non-measurable, $f_t = \mathbb{1}_{\{t\}}$ for $t\in V$: each measurable; $\sup_{t\in V}f_t = \mathbb{1}_V$ non-measurable, as in §1.8.3. (c) $\sup_Q\phi\le\sup_{\mathbb{R}^p}\phi$ is trivial. For the reverse, take any $w$ and $q_n\to w$ with $q_n\in Q$ (density); continuity gives $\phi(q_n)\to\phi(w)$, so $\sup_Q\phi\ge\phi(w)$; take sup over $w$. Hence the two sups coincide, and the right-hand one is a countable supremum of random variables (each $\phi(q)$ measurable in the underlying data randomness), hence measurable. (d) Take $h_w(x) = \operatorname{sign}(x - w)$ on one feature, 0–1 loss, and any finite dataset. As $w$ crosses a data point $x_i$, the empirical risk jumps by $\pm1/n$: $w\mapsto\hat R_n(h_w)$ is a step function, not continuous. So (c)'s hypothesis fails, and the measurability of $\sup_w|\hat R_n - R|$ for classification cannot be established by this route; empirical-process arguments (pointwise measurability of the class $\{(x,y)\mapsto\mathbb{1}[h_w(x)\ne y]\}$, or Suslin conditions) are what careful treatments use. In *this* one-dimensional example one can hand-verify that suprema over rational $w$ suffice — the step function is right- or left-continuous between data points — which is precisely a pointwise-measurability argument in miniature.

**A13.** (a) By definition of pushforward, $P(\{x_0\}) = \nu(G^{-1}(\{x_0\}))$; an atom means this is positive. (b) $G(z) = (\max(0,z_1),0)$, $\nu = \mathcal{N}(0,I_2)$: continuous, non-constant, and $G^{-1}(\{(0,0)\}) = \{z_1\le0\}$ has $\nu$-measure $1/2$. (c) If every level set $L_x = G^{-1}(\{x\})$ has $\lambda_k(L_x) = 0$ and $\nu\ll\lambda_k$ (which is what having a density means — Chapter 2 makes "$\ll$" official, but "$\nu(N) = \int_N p\,d\lambda_k = 0$ when $\lambda_k(N)=0$" is all we use), then $\nu(L_x) = 0$ for every $x$: no atoms. (d) A ReLU network is piecewise affine: latent space is partitioned into polyhedral cells, on each of which $G$ is affine. If on some cell the affine map is constant — all ReLUs feeding the output are "off," or weights annihilate the cell's directions — then the whole cell maps to one point, and if the cell has positive latent probability, that point is an atom. This is the mechanism of (b) writ large, and it is why Proposition 1.23 hedges where revision 2 did not.

## Part B

Representative results (seed-dependent in the third decimal).

**B1.**
```python
import numpy as np
rng = np.random.default_rng(0)
w = rng.uniform(-1, 1, 10**6); y = w**2
for b in (0.25, 0.5, 0.81):
    print(b, (y <= b).mean(), np.sqrt(b))
```
Output: $0.4999$ vs $0.5$; $0.7072$ vs $0.7071$; $0.9002$ vs $0.9$. Histogram hugs $1/(2\sqrt b)$, spiking near 0. *Interpretation:* squaring compresses $[-\delta, \delta]$ (input length $2\delta$) into $[0,\delta^2]$ (output length $\delta^2$) — mass conserved, length shrunk, so the density (mass per length) blows up like $1/(2\sqrt b)$; the spike is integrable ($\int_0^1 \frac{db}{2\sqrt b} = 1$), so total probability is fine. Densities are rates, not probabilities — Failure 1 in miniature.

**B2.** $\hat{\mathbb{P}}(A=0) = 0.4994$; $\hat F_A(0.5) = 0.6908$ vs $\Phi(0.5) = 0.6915$; $\hat F_A(1) = 0.8412$ vs $0.8413$; $\hat F_A(2) = 0.9772$ ✓; $\widehat{\mathbb{E}}[A] = 0.3994$ vs $1/\sqrt{2\pi} = 0.3989$. The empirical CDF shows a vertical riser of height $\approx0.5$ at $0$ then a smooth Gaussian-shaped climb. *Interpretation:* one variable, both behaviours; the CDF accommodates the mixture without special cases, exactly as §1.10.1 claims. Note the mean is finite and well-defined despite the absence of both pmf and pdf — expectation belongs to the measure, not to a density (Chapter 2).

**B3.** Table matches Worked Example 1.6. The ratio (true)/(bound) for independent events is $\approx0.95$ at $mp = 0.1$, $\approx0.63$ at $mp = 1$, $\to (1-e^{-mp})/(mp)\to0$ as $mp$ grows (for small $p$, the truth is $1-(1-p)^m\approx1-e^{-mp}$). *Interpretation:* the bound is essentially free while $mp\ll1$ and worthless beyond $mp\approx1$: with per-hypothesis failure probability $\delta_0$, a class of size $m\gg1/\delta_0$ gets no guarantee. To cover big classes you either shrink $\delta_0$ exponentially (costing sample size logarithmically — the $\log m$) or abandon the union bound for covering/chaining arguments: Chapter 4's agenda in one plot.

**B4.** (a) $\max\big|\|G(t)\|-1\big| < 10^{-6}$ ✓. (b) $0$ of $200{,}000$ samples in $\{\|x\|<0.99\}$, vs $98.01\%$ of the disc's area. (c) With $\sigma = 0.001$: still $\approx0$ inside; $\sigma = 0.01$: $\approx0.4\%$; $\sigma = 0.1$: $\approx34\%$ — noise thickens the circle into an annulus of positive area, and mass leaks inward. *Interpretation:* output noise converts a measure-zero support into full-dimensional support, which is exactly why a VAE's Gaussian observation model gives it a well-defined likelihood while a GAN has none. The price is blur: the likelihood exists because the model literally is data + noise. (d) $\hat{\mathbb{P}}(G_2 = (0,0)) = 0.4997\approx1/2$: an atom, produced by a non-constant map — Proposition 1.23's counterexample, live. A histogram of the first coordinate shows the §1.10 ReLU picture embedded in the plane.

**B5.** (a) $0.0913$ vs analytic $0.0905$ (the simulation's $109{,}415$ positives make the standard error $\approx0.0009$; consistent ✓). (b) Log-linear falloff: precision $\approx$ prevalence$/(\text{prevalence} + \alpha)$ for small prevalence, where $\alpha = 0.05$ — at prevalence $10^{-4}$, precision $\approx0.002$. (c) Crosses $0.5$ at specificity $\approx0.995$, confirming §1.7.2. (d) Sensitivity $0.99\to0.999$: precision moves $0.0905\to0.0913$ — a gain of $0.0008$. Specificity $0.95\to0.99$: precision $0.0905\to0.3322$ — a gain of $0.24$, three hundred times larger. *Interpretation:* the positive pool is dominated by false positives from the healthy majority, so shrinking the false-positive rate attacks the denominator where it is fat; sensitivity attacks a numerator already near its ceiling ($\le0.005$ of the population). At low base rates, invest in specificity — a conclusion invisible to anyone reasoning about "accuracy."

**B6.** (a) `x = -np.log(1 - u)`: empirical mean $1.0002$ vs $1$; $\hat F(1) = 0.6320$ vs $0.6321$ ✓. (b) `x = np.where(u <= 0.5, 0.0, ndtri(u))` (with `scipy.special.ndtri`): $\hat{\mathbb{P}}(X=0) = 0.5001$; $\hat F(1) = 0.8409$ vs $0.8413$ ✓ — the sampler reproduces the ReLU law without ever touching a Gaussian sample or a ReLU. (c) A *jump* of $F$ (an atom) is a flat stretch of $F^{-1}$: every $u\in(0,\frac12]$ maps to the same output $0$, so an interval of uniform mass lands on one point — mass concentrates, atom reproduced. Dually, a *flat stretch* of $F$ (a gap in the support) is a jump of $F^{-1}$: no $u$ lands inside the gap. The generalized inverse is a dictionary between features of $F$ and features of the law, and reading it fluently is most of Chapter 3's change-of-variables intuition.

## Part C

**C1.** *(a)* Let $\mathcal{F}_1 := \{\emptyset,\Omega\}$ (before anything is drawn, only $\theta_1$ — a constant — is known) and for $t\ge2$, $\mathcal{F}_t := \sigma(B_1,\ldots,B_{t-1})$: the information in the first $t-1$ minibatches. Increasing since each generating family nests ✓. *(b)* Claim: $\theta_t$ is $\mathcal{F}_t$-measurable. Induction. Base: $\theta_1$ constant, measurable w.r.t. anything. Step: $\theta_{t+1} = \theta_t - \eta_t\,g(\theta_t, B_t)$ where $g$ is a measurable function of (parameters, batch) — a finite composition of the network's measurable operations. By induction $\theta_t$ is $\mathcal{F}_t$-measurable $\subseteq\mathcal{F}_{t+1}$-measurable, $B_t$ is $\mathcal{F}_{t+1}$-measurable by construction, and measurable functions of measurable inputs are measurable (Proposition 1.15). ∎ Interpretation: *the parameter trajectory is determined by the data seen so far* — Doob–Dynkin makes "determined by" literal: $\theta_t = \varphi_t(B_1,\ldots,B_{t-1})$ for a measurable $\varphi_t$. *(c)* Unbiasedness: $\mathbb{E}[g_t\mid\mathcal{F}_t] = \nabla F(\theta_t)$ a.s. — *averaging over only the fresh randomness $B_t$, with the past frozen, the gradient estimate is exact at the current (random!) iterate.* *(d)* The unconditional statement compares apples to oranges: $\mathbb{E}[g_t]$ averages over the randomness in $\theta_t$ as well, giving $\mathbb{E}[\nabla F(\theta_t)]$ at best — the gradient averaged over wherever the trajectory might be, which is not the gradient anywhere in particular. Convergence proofs proceed step by step from the current iterate ("conditionally on the present, the next step improves in expectation"), and that is intrinsically a conditional statement; it is also why the tools are martingale tools (Chapter 6). Formally: without conditioning one cannot even write the one-step descent inequality $\mathbb{E}[F(\theta_{t+1})\mid\mathcal{F}_t]\le F(\theta_t) - \eta_t\|\nabla F(\theta_t)\|^2 + O(\eta_t^2)$, whose *conditional* left side is where the analysis lives.

**C2.** *(a)* Exercise A11 verbatim: the image of $\mathbb{R}^k$ under a $C^1$ map is $\lambda_d$-null for $k<d$; $P_\theta$ gives it probability 1; a density would integrate to 0 over it. ∎ *(b)* Maximum likelihood requires evaluating a density $p_\theta(x_i)$ at each data point. There is no density, so the objective is not merely intractable — the symbol $p_\theta(x_i)$ has no referent. (And the naive limit picture makes it worse: for $x_i$ off the support, shrinking balls give likelihood $0$, i.e. $\log$-likelihood $-\infty$; on the support, the ratio $P_\theta(B(x_i,r))/\lambda_d(B(x_i,r))$ blows up. Nothing finite and nonzero is available.) *(c)* Two smooth surfaces of dimensions $k, k' < d$ in general position intersect in dimension $k + k' - d$, which is negative when $k + k' < d$ — the generic intersection is empty; and even when nonempty it is lower-dimensional, hence null for both laws. With disjoint (up to null sets) supports, any divergence defined through density ratios or shared reference measures saturates: $\mathrm{KL}(P_{\text{data}}\Vert P_\theta) = \infty$, $\mathrm{JS} = \log 2$, total variation $= 1$ — *constants*, whatever $\theta$. A constant objective has zero gradient a.e.: the generator gets no training signal however wrong it is. This is Failure 4, now fully diagnosed (Arjovsky & Bottou, 2017); the last step — computing those divergence values — is Chapter 2's. *(d)* Three escapes. **Normalizing flows** set $k = d$ and constrain $G_\theta$ to be invertible with tractable Jacobian: the pushforward then *has* a density, computable by change of variables (Chapter 3), so maximum likelihood works; the price is architectural rigidity and full-dimensional latents. **VAEs** append output noise, $x = G_\theta(z)+\varepsilon$: the convolution with a full-dimensional Gaussian has a density (Lab B4(c) is the picture); the price is blur and that one optimizes a bound (the ELBO, Chapter 13) rather than the likelihood. **Wasserstein GANs** change the yardstick: the $W_1$ distance is defined by transport cost, not density ratios, so it is finite and typically well-behaved for disjoint supports — moving the model surface toward the data surface reduces it continuously; the price is estimating $W_1$ through a critic and enforcing its Lipschitz constraint (Chapter 11 touches $f$-divergences vs metrics). *(Postscript: diffusion models dissolve the problem from the other end — the forward noising process gives every intermediate distribution full support, so score matching has densities to work with at every noise level; by the time this matters, Chapter 13 will have built the tools.)* One paragraph of measure theory — "a $C^1$ image of a lower-dimensional space is null" — thus explains the likelihood-freeness of GANs, the blur of VAEs, the invertibility constraint of flows, and the design of the WGAN objective. That is the argument for learning this material.

---

# References

- R. Durrett, *Probability: Theory and Examples*, 5th ed., Cambridge University Press, 2019. (The standard graduate text; Appendix A has Carathéodory and Kolmogorov extension.)
- D. Williams, *Probability with Martingales*, Cambridge University Press, 1991. (Short, sharp; the best companion for Chapters 1–2 and 6.)
- P. Billingsley, *Probability and Measure*, 3rd ed., Wiley, 1995. (Encyclopedic reference for the measure-theoretic borrowings of §1.14.)
- S. Resnick, *A Probability Path*, Birkhäuser, 1999. (Gentlest of the graduate texts; good for a second pass.)
- O. Kallenberg, *Foundations of Modern Probability*, 3rd ed., Springer, 2021. (Doob–Dynkin in full generality; for reference, not first reading.)
- S. M. Srivastava, *A Course on Borel Sets*, Springer, 1998. (Analytic sets and why "definable" ≠ Borel, §1.4.1.)
- A. W. van der Vaart and J. A. Wellner, *Weak Convergence and Empirical Processes*, Springer, 1996. (The measurability machinery for uncountable suprema, §1.8.3.)
- S. Shalev-Shwartz and S. Ben-David, *Understanding Machine Learning*, Cambridge University Press, 2014. (Finite-class bounds and the union-bound-to-uniform-convergence pipeline of §1.6.3.)
- M. Arjovsky and L. Bottou, "Towards Principled Methods for Training Generative Adversarial Networks," ICLR 2017. (Failure 4: disjoint supports and constant divergences.)
- M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein Generative Adversarial Networks," ICML 2017. (The metric-based escape of C2(d).)
- D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," ICLR 2014. (The output-noise escape of C2(d).)
- L. Bottou, F. E. Curtis, and J. Nocedal, "Optimization Methods for Large-Scale Machine Learning," *SIAM Review* 60(2), 2018. (The filtration-based SGD formalism of C1.)

*End of Chapter 1. Chapter 2: Integration, Expectation, and Densities.*
