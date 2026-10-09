# Chapter 6 — Conditional Expectation and Martingales

## Chapter 6 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 6.1 Why elementary conditioning is not enough | $\mathbb{P}(A\mid B)$ dies when $\mathbb{P}(B)=0$, and $\{X=x\}$ is null for continuous $X$ | Regression and classification condition on events of probability zero *by definition*; we need a theory that survives it |
| 6.2 Definition and existence | $\mathbb{E}[X\mid\mathcal{G}]$ is a $\mathcal{G}$-measurable random variable matching $X$'s integral on every $G\in\mathcal{G}$; it exists by Radon–Nikodym | Makes "the model's prediction given the features" a well-defined object even with continuous inputs |
| 6.3 Conditional expectation as projection | In $L^2$, $\mathbb{E}[X\mid\mathcal{G}]$ is the orthogonal projection onto $\mathcal{G}$-measurable functions | **Regression is projection.** The Bayes-optimal predictor under squared loss is the conditional mean; under absolute loss the median; under 0–1 loss the mode; under log loss (cross-entropy) the entire conditional distribution |
| 6.4 Properties | Linearity, monotonicity, tower, pull-out, independence, irrelevance of independent information ("freezing"), conditional Jensen, conditional DCT | The algebra every derivation in ML silently uses: importance weights, REINFORCE, Rao–Blackwellisation, EM |
| 6.5 Conditional variance | Law of total variance, and bias–variance as its corollary | Bias–variance is a *conditioning identity*, not folklore; irreducible noise is $\mathbb{E}[\operatorname{Var}(Y\mid X)]$ |
| 6.6 Conditional distributions | Regular conditional probability; $p(y\mid x)=p(x,y)/p(x)$ as a special case | Justifies every "$p(y\mid x)$" in a paper; Gaussian conditioning (Ch. 5) is one instance; the Borel paradox is the warning label |
| 6.7 Martingales | Filtrations, adapted/predictable processes, stopping times, optional stopping | SGD trajectories, sequential testing, likelihood ratios, gambling strategies that "cannot lose" |
| 6.8 Inequalities and convergence | Doob's maximal inequality, martingale convergence, uniform integrability and Lévy's upward theorem, Azuma–Hoeffding | Concentration for *sequentially revealed* data; McDiarmid's proof; anytime-valid confidence sequences |
| 6.9 Capstone: SGD convergence | Full proof under $L$-smoothness, conditional unbiasedness, bounded conditional variance, Robbins–Monro steps | The theorem behind every learning-rate schedule — and an exact account of which assumption each real-world failure breaks |

## 6.0 What this chapter is for

Chapter 1 made a claim that has been waiting five chapters for payment: *a $\sigma$-algebra is information*. We used it as a slogan. We drew the picture of a filtration $\mathcal{F}_1\subseteq\mathcal{F}_2\subseteq\cdots$ as knowledge accumulating over time, and we wrote down, without being able to define it, the assumption that makes stochastic gradient descent work: $\mathbb{E}[g_t\mid\mathcal{F}_t]=\nabla F(\theta_t)$. This chapter defines that object, proves it exists, proves it is essentially unique, derives its algebra, and then uses it to prove that SGD converges. By the end you will have cashed the slogan in full.

Here is the difficulty in one line. Chapter 1's Definition 1.10 defines $\mathbb{P}(A\mid B)=\mathbb{P}(A\cap B)/\mathbb{P}(B)$ and requires $\mathbb{P}(B)>0$. But the central objects of supervised learning are $\mathbb{E}[Y\mid X=x]$ and $\mathbb{P}(Y=1\mid X=x)$ for a feature vector $X$ with a continuous distribution — and for such $X$, $\mathbb{P}(X=x)=0$ for every single $x$. Every regression function, every classifier's output, every "$p(y\mid x)$" printed in a paper is nominally a $0/0$. Practitioners divide anyway and get correct answers; the reason they get correct answers is the content of Sections 6.2 and 6.6, and the reason they *sometimes* get incoherent answers (the Borel paradox, §6.6.4) is the content of the same sections.

The move that fixes it is the one Chapter 1 set up. Do not condition on the *event* $\{X=x\}$; condition on the *$\sigma$-algebra* $\sigma(X)$, all at once, for all $x$ simultaneously. The output is not a number but a random variable: a $\sigma(X)$-measurable function on $\Omega$ which, by Doob–Dynkin (Lemma 1.22), is automatically of the form $g(X)$ for a measurable $g$. That $g$ is the regression function. It is characterised not pointwise but by an averaging property — it integrates the same as $X$ over every event you can describe using $\mathcal{G}$ — and it exists because of the Radon–Nikodym theorem from Chapter 2. Individual values $g(x)$ are only defined up to null sets, which is exactly why every theorem in this chapter ends in "a.s."

**What you are assumed to bring.** From Chapter 2: the Lebesgue integral and its convergence theorems (monotone convergence, Fatou, dominated convergence), the Radon–Nikodym theorem for $\sigma$-finite measures with $\nu\ll\mu$, the completeness of $L^p(\mu)$ and in particular that $L^2$ is a Hilbert space with the orthogonal projection theorem for closed subspaces, and Fubini–Tonelli. From Chapter 4: Jensen's inequality and the bounded-differences (McDiarmid) inequality, which we finally *prove* here. From Chapter 5: the conditioning formula for a jointly Gaussian vector, $\mathbb{E}[Y\mid X=x]=\mu_Y+\Sigma_{YX}\Sigma_{XX}^{-1}(x-\mu_X)$, which we will re-derive in this chapter's language as a worked instance. From Chapter 1 by number: Definitions 1.3 ($\sigma$-algebra), 1.10 (elementary conditional probability), 1.21 ($\sigma(X)$), 1.23 (independence), 1.25 (a.s.); Propositions 1.4 (finite $\sigma$-algebras $=$ partitions), 1.11 ($\mathbb{P}(\cdot\mid B)$ is a probability measure), 1.15 (closure properties of measurability); and above all Lemma 1.22, Doob–Dynkin, which is the conceptual key to the whole chapter.

**The punchline, stated in advance.** Three sentences. *(1)* Conditional expectation is orthogonal projection onto the subspace of functions you are allowed to compute, so regression is geometry and the choice of loss is the choice of which posterior summary you extract. *(2)* Every identity in ML that looks like an accounting trick — the tower property, the law of total variance, the bias–variance decomposition, the unbiasedness of a minibatch gradient — is one theorem, applied at different depths. *(3)* A martingale is a process whose conditional expectation given the past is its present value, and this single property is strong enough to prove concentration inequalities and to prove that SGD converges; when SGD fails in practice, it is because a specific, nameable hypothesis of that proof has been broken, and this chapter names them.

A warning about difficulty, honestly given. Section 6.2's definition will feel unmotivated for about ten minutes. It defines an object by a property it satisfies rather than by a formula, which is unfamiliar if your probability so far has been formula-driven. Push through to Worked Example 6.2, where the abstract definition is computed on three coin flips and turns out to say something completely obvious: *average over each atom*. Everything after that is elaboration of that picture.

## 6.1 Why elementary conditioning is not enough

### 6.1.1 The object ML actually wants

Fix a probability space $(\Omega,\mathcal{F},\mathbb{P})$ carrying a feature vector $X:\Omega\to\mathbb{R}^d$ and a label $Y:\Omega\to\mathbb{R}$. Supervised learning is the business of producing a function $\hat f$ with $\hat f(X)\approx Y$. Before you can ask whether a neural network is close to the best possible $\hat f$, you have to say what the best possible $\hat f$ *is*. Every textbook answers: the regression function
$$
f^\star(x)\;=\;\mathbb{E}[Y\mid X=x],
$$
and for binary labels, $\eta(x)=\mathbb{P}(Y=1\mid X=x)$, the thing a sigmoid output layer is trained to approximate. These are the targets. Cross-entropy training, calibration curves, "the model is overconfident", the Bayes error rate — all of it is talk about $\eta$.

Now apply Definition 1.10 literally. It says
$$
\mathbb{P}(Y=1\mid X=x)\;=\;\frac{\mathbb{P}(\{Y=1\}\cap\{X=x\})}{\mathbb{P}(X=x)},
$$
and it explicitly requires the denominator to be positive. If $X$ has a Lebesgue density — the standing assumption for continuous features, and the model in every paper that writes $p(x)$ — then $\mathbb{P}(X=x)=0$ for every $x\in\mathbb{R}^d$. Both numerator and denominator vanish. The definition does not merely give an unhelpful answer; it gives no answer at all.

This is not a corner case. It is the *typical* case in modern ML: image pixels, embeddings, log-returns, sensor readings. Chapter 1 promised that "probability zero is not impossible", and here that slogan acquires a bill: events of probability zero are precisely the ones we most want to condition on.

### 6.1.2 The natural repair, and why it is not enough on its own

The obvious repair is to condition on a thickened event and take a limit:
$$
\mathbb{P}(Y\in A\mid X=x)\;\stackrel{?}{=}\;\lim_{h\downarrow0}\mathbb{P}\bigl(Y\in A\;\bigm|\;\|X-x\|\le h\bigr).
$$
Each term on the right is legitimate: for small $h$ the ball still has positive probability, so Definition 1.10 applies. This is a decent instinct, and in nice situations the limit exists and equals what we want. It is also, in disguise, what $k$-nearest-neighbours does.

**Worked Example 6.1 (the limit exists and is computable).**
Let $(U_1,U_2)$ be independent $\mathrm{Uniform}(0,1)$ and set $X=\max(U_1,U_2)$, $Y=\min(U_1,U_2)$. The pair $(X,Y)$ is uniform on the triangle $T=\{0<y<x<1\}$, whose area is $1/2$, so the joint density is $p(x,y)=2\cdot\mathbf{1}_T(x,y)$ with respect to $\lambda_2$.

Take $x=0.3$ and $A=(0,0.1]$. For small $h>0$,
$$
\mathbb{P}\bigl(Y\le 0.1,\;|X-0.3|\le h\bigr)=\int_{0.3-h}^{0.3+h}\!\!\int_0^{0.1}2\,dy\,dx=2\cdot 2h\cdot 0.1=0.4h,
$$
$$
\mathbb{P}\bigl(|X-0.3|\le h\bigr)=\int_{0.3-h}^{0.3+h}\!\!\int_0^{x}2\,dy\,dx=\int_{0.3-h}^{0.3+h}2x\,dx=2\cdot 2h\cdot 0.3=1.2h .
$$
The ratio is $0.4h/1.2h=1/3$ for every $h<0.2$ — the $h$'s cancel exactly, so the limit is $1/3$ and we did not even need to take a limit. The answer is what a density calculation would give: $Y\mid X=0.3$ is uniform on $(0,0.3)$, so $\mathbb{P}(Y\le 0.1\mid X=0.3)=0.1/0.3=1/3$. Section 6.6 will prove this is not a coincidence.

So why not simply *define* conditional probability by such a limit? Three reasons, in increasing order of seriousness.

*First, existence.* The limit need not exist. Nothing in the definition of a joint law forces the ratio to settle down as $h\downarrow0$; for joint laws without densities it routinely oscillates.

*Second, and fatally, the answer depends on how you shrink.* "The event $\{X=x\}$" does not determine a limiting procedure. Shrinking with balls, with thin rectangles, or along a curve through $x$ can give different limits — even for the innocent-looking bivariate normal, once you reparametrise. This is the **Borel paradox**, and we return to it with a concrete instance in §6.6.4. A definition that depends on an arbitrary choice made outside the model is not a definition.

*Third, it answers one question at a time.* The limit gives you a number for one $x$ and one $A$. What ML needs is a *function* $x\mapsto f^\star(x)$ that is measurable — otherwise $f^\star(X)$ is not a random variable, $\mathbb{E}[(Y-f^\star(X))^2]$ is not defined, and the whole apparatus of risk fails to get started. Chapter 1 hammered on countability being load-bearing for exactly this reason: assembling uncountably many separately-defined numbers into a measurable function is not automatic. Doing it one $x$ at a time gives you no measurability at all.

> **Common Confusion 6.1.**
> *"But surely $\mathbb{P}(Y\in A\mid X=x)$ just means the limit of $\mathbb{P}(Y\in A\mid X\in[x-h,x+h])$. Every physics course conditions on measure-zero events and nothing bad happens. Isn't this measure-theoretic fussing?"*
>
> Two things are true at once, and keeping them apart is the whole point. **True:** when $(X,Y)$ has a joint density that is continuous and positive near $x$, the limit exists, is independent of the shrinking scheme, and equals $p(x,y)/p_X(x)$. Practitioners divide densities and are right. **Also true:** the limit is not what *defines* the object, and when the regularity fails the limiting recipe returns different answers for different shrinking schemes, with no principled way to choose. The measure-theoretic definition (§6.2) never mentions a limit. It pins down the conditional expectation by an *averaging* property that holds for all events simultaneously, then §6.6 shows the density formula is a theorem about that object. The fussing buys you: a definition that always applies, automatic measurability in $x$, and a precise statement of what "$p(y\mid x)$" means when the naive limit misbehaves — namely, a version defined up to a $\mathbb{P}_X$-null set.

### 6.1.3 The plan

We will define $\mathbb{E}[X\mid\mathcal{G}]$ for a sub-$\sigma$-algebra $\mathcal{G}\subseteq\mathcal{F}$, not $\mathbb{E}[X\mid \text{an event}]$. The two guiding intuitions:

1. **$\mathcal{G}$ is what you are allowed to see.** By Proposition 1.4, on a finite $\Omega$ a $\sigma$-algebra is a partition, and its atoms are the finest distinctions available to an observer holding $\mathcal{G}$. Such an observer cannot report $X(\omega)$ — that requires knowing $\omega$ exactly — but can report the average of $X$ over the atom containing $\omega$. That average *is* the conditional expectation.
2. **Best guess $=$ correct averages.** If someone hands you a candidate summary $Y$ for $X$, the test is: for every question $G$ you can ask using $\mathcal{G}$ alone, does $Y$ have the same total as $X$ on $G$? If yes for all $G\in\mathcal{G}$, then $Y$ has extracted everything $\mathcal{G}$ can say about $X$ and added nothing.

Property (2) turns into the defining equation $\int_G Y\,d\mathbb{P}=\int_G X\,d\mathbb{P}$ for all $G\in\mathcal{G}$. Property (1) turns into the requirement that $Y$ be $\mathcal{G}$-measurable. Existence will come from Radon–Nikodym; measurability in $x$ will come for free from Doob–Dynkin; and the price we pay is that $Y$ is only determined up to a null set.

> **Self-check 6.1.**
> 1. Why is $\mathbb{P}(X=x)=0$ for all $x$ whenever $X$ has a Lebesgue density? (One sentence, citing a Chapter 2 fact.)
> 2. In Worked Example 6.1, compute $\mathbb{P}(Y\le 0.1\mid |X-0.5|\le h)$ for small $h$, and check it against $0.1/0.5$.
> 3. Give the three reasons the limiting definition is unsatisfactory, in one clause each.
> 4. Chapter 1's random variable $A=\max(0,Z)$ has $\mathbb{P}(A=0)=1/2$. Is $\mathbb{E}[Z\mid A=0]$ legitimate under Definition 1.10? What about $\mathbb{E}[Z\mid A=1]$?

## 6.2 Definition and existence

### 6.2.1 The definition

**Definition 6.1 (conditional expectation).**
Let $(\Omega,\mathcal{F},\mathbb{P})$ be a probability space, let $\mathcal{G}\subseteq\mathcal{F}$ be a sub-$\sigma$-algebra, and let $X\in L^1(\mathbb{P})$, i.e. $\mathbb{E}|X|<\infty$. A random variable $Y$ is called *a version of the conditional expectation of $X$ given $\mathcal{G}$*, written $Y=\mathbb{E}[X\mid\mathcal{G}]$ a.s., if

&nbsp;&nbsp;**(CE1)** $Y$ is $\mathcal{G}$-measurable;
&nbsp;&nbsp;**(CE2)** $Y\in L^1(\mathbb{P})$;
&nbsp;&nbsp;**(CE3)** $\displaystyle\int_G Y\,d\mathbb{P}=\int_G X\,d\mathbb{P}$ for every $G\in\mathcal{G}$.

Read the three conditions as a job description. (CE1): *you may only use information in $\mathcal{G}$* — no peeking at $\omega$ beyond what $\mathcal{G}$ resolves. (CE2): the answer is an honest integrable random variable. (CE3): *your summary must have the right totals on every event you can describe*. Taking $G=\Omega$ in (CE3) already gives $\mathbb{E}[Y]=\mathbb{E}[X]$, the crudest form of the tower property.

Notice what is *not* in the definition: no formula, no limit, no pointwise prescription. This is a characterisation-by-property, like defining the Radon–Nikodym derivative or the transpose of a matrix. Such definitions are worth the initial discomfort because they make proofs mechanical: to show that some concrete $Y$ *is* the conditional expectation, you check three boxes.

**Theorem 6.2 (existence and a.s. uniqueness).**
Let $X\in L^1(\Omega,\mathcal{F},\mathbb{P})$ and $\mathcal{G}\subseteq\mathcal{F}$ a sub-$\sigma$-algebra. Then a version of $\mathbb{E}[X\mid\mathcal{G}]$ exists, and any two versions agree $\mathbb{P}$-a.s.

*Proof.* **Existence, $X\ge0$ first.** Define a set function on the measurable space $(\Omega,\mathcal{G})$ by
$$
\nu(G)\;:=\;\int_G X\,d\mathbb{P},\qquad G\in\mathcal{G}.
$$
Then $\nu$ is a measure: $\nu(\varnothing)=0$, and if $G=\bigsqcup_n G_n$ with $G_n\in\mathcal{G}$ disjoint, then $X\mathbf{1}_G=\sum_n X\mathbf{1}_{G_n}$ pointwise with non-negative terms, so countable additivity follows from the monotone convergence theorem (Ch. 2) applied to the partial sums. It is a *finite* measure because $\nu(\Omega)=\mathbb{E}[X]\le\mathbb{E}|X|<\infty$.

Let $\mathbb{P}|_{\mathcal{G}}$ denote the restriction of $\mathbb{P}$ to $\mathcal{G}$; it is a probability measure, hence $\sigma$-finite. If $G\in\mathcal{G}$ has $\mathbb{P}(G)=0$ then $X\mathbf{1}_G=0$ a.s., so $\nu(G)=0$. Thus $\nu\ll\mathbb{P}|_{\mathcal{G}}$, and both measures live on the *same* $\sigma$-algebra $\mathcal{G}$ — this is the crux, and it is why the resulting density is automatically $\mathcal{G}$-measurable. The Radon–Nikodym theorem (Ch. 2) supplies a $\mathcal{G}$-measurable $Y\ge0$ with
$$
\nu(G)=\int_G Y\,d\mathbb{P}|_{\mathcal{G}}=\int_G Y\,d\mathbb{P}\qquad\text{for all }G\in\mathcal{G},
$$
the last equality because $Y\mathbf{1}_G$ is $\mathcal{G}$-measurable and $\mathbb{P}|_{\mathcal{G}}$ agrees with $\mathbb{P}$ on $\mathcal{G}$. That is (CE1) and (CE3); (CE2) follows from $G=\Omega$: $\mathbb{E}[Y]=\mathbb{E}[X]<\infty$ and $Y\ge0$.

**General $X$.** Write $X=X^+-X^-$ with $X^\pm\ge0$ integrable, obtain $Y^\pm$ as above, and set $Y=Y^+-Y^-$, which is finite a.s. because $\mathbb{E}[Y^\pm]=\mathbb{E}[X^\pm]<\infty$. Conditions (CE1)–(CE3) are preserved by this difference.

**Uniqueness.** Suppose $Y,Y'$ both satisfy (CE1)–(CE3). Fix $\varepsilon>0$ and let $G_\varepsilon:=\{Y-Y'\ge\varepsilon\}$. Since $Y,Y'$ are $\mathcal{G}$-measurable, $G_\varepsilon\in\mathcal{G}$ (Proposition 1.15). Subtracting the two instances of (CE3),
$$
0=\int_{G_\varepsilon}(Y-Y')\,d\mathbb{P}\;\ge\;\varepsilon\,\mathbb{P}(G_\varepsilon),
$$
so $\mathbb{P}(G_\varepsilon)=0$. Taking $\varepsilon=1/k$ and using countable subadditivity (Proposition 1.8(2), the union bound) over $k\in\mathbb{N}$ gives $\mathbb{P}(Y>Y')=0$. By symmetry $\mathbb{P}(Y<Y')=0$, hence $Y=Y'$ a.s. $\blacksquare$

Four remarks on the proof, each of which pays off later.

*The proof is Radon–Nikodym and nothing else.* Chapter 2 introduced $d\nu/d\mu$ to give a single notion of "density" covering discrete, continuous and mixed cases at once (that was Failure 1 of the Four Failures). Here the same theorem manufactures conditional expectation. If you ever wonder why measure theory earns its keep in ML, this is a good exhibit: one abstract theorem, two apparently unrelated payoffs.

*Uniqueness is only a.s., and this is not a technicality.* The conditional expectation is an equivalence class of random variables, not a function. Any two versions differ on a null set, and a null set can be chosen to contain any particular point you care about. This is exactly why we cannot speak of "the value of $\mathbb{E}[Y\mid X]$ at $X=x$" for a single $x$ without further work (§6.6 does that work) and why every statement below carries "a.s."

*There is a second road to existence, and it is worth knowing it exists.* One can instead define $\mathbb{E}[X\mid\mathcal{G}]$ first for $X\in L^2$ as an orthogonal projection in Hilbert space (exactly the content of §6.3), and then extend to $X\in L^1$ by truncating, using monotonicity, and passing to the limit. That route trades Radon–Nikodym for the projection theorem. The fact that two unrelated-looking pillars of Chapter 2 each independently manufacture the same object is itself evidence that the three conditions (CE1)–(CE3) carve out something canonical, and it is the deeper reason a characterisation-by-property was the right kind of definition.

*Integrability was assumed, not proved.* If $X\ge0$ but $\mathbb{E}[X]=\infty$ the same construction still yields a $\mathcal{G}$-measurable $Y$ with values in $[0,\infty]$ satisfying (CE3); this extended conditional expectation is occasionally useful and we allow it silently for non-negative $X$.

**Definition 6.3 (conditioning on random variables and events).**
For a random vector $W$ (on the same space) define $\mathbb{E}[X\mid W]:=\mathbb{E}[X\mid\sigma(W)]$, where $\sigma(W)$ is as in Definition 1.21. For $A\in\mathcal{F}$ define the *conditional probability* $\mathbb{P}(A\mid\mathcal{G}):=\mathbb{E}[\mathbf{1}_A\mid\mathcal{G}]$, and $\mathbb{P}(A\mid W):=\mathbb{P}(A\mid\sigma(W))$.

Because $\mathbb{E}[X\mid W]$ is $\sigma(W)$-measurable, **Doob–Dynkin (Lemma 1.22)** applies: there is a measurable $g$ with $\mathbb{E}[X\mid W]=g(W)$ a.s. We write $\mathbb{E}[X\mid W=w]:=g(w)$. The function $g$ is determined only up to $\mathbb{P}_W$-null sets, so "$\mathbb{E}[X\mid W=w]$" is meaningful for $\mathbb{P}_W$-almost every $w$, not for every $w$. **This is the sentence that makes the regression function legitimate.** The target $f^\star(x)=\mathbb{E}[Y\mid X=x]$ of §6.1 exists, is measurable in $x$, and is unique up to changes on sets of feature-probability zero.

*Random vectors.* For $X=(X^{(1)},\dots,X^{(d)})$ with each coordinate in $L^1$, define $\mathbb{E}[X\mid\mathcal{G}]$ coordinatewise: it is the vector whose $i$-th entry is $\mathbb{E}[X^{(i)}\mid\mathcal{G}]$. Linearity, the tower property, pull-out (with scalar or matrix-valued $\mathcal{G}$-measurable $Z$), and the independence properties of Theorem 6.8 all extend immediately, because each is a coordinatewise statement. We use this silently in the SGD proof of §6.9, where gradients are vectors and the conditional bias–variance split is applied per coordinate and summed.

### 6.2.2 The finite case: averaging over atoms

Abstraction pays only if it reduces to the obvious thing in the obvious case. It does.

**Proposition 6.4 (conditional expectation on a partition).**
Suppose $\mathcal{G}=\sigma(\{A_1,\dots,A_k\})$ where $A_1,\dots,A_k$ partition $\Omega$ (by Proposition 1.4 every $\sigma$-algebra with finitely many sets is of this form, the $A_i$ being its atoms). Assume $\mathbb{P}(A_i)>0$ for all $i$. Then for $X\in L^1$,
$$
\mathbb{E}[X\mid\mathcal{G}](\omega)\;=\;\sum_{i=1}^k\Bigl(\frac{1}{\mathbb{P}(A_i)}\int_{A_i}X\,d\mathbb{P}\Bigr)\mathbf{1}_{A_i}(\omega)\;=\;\sum_{i=1}^k\mathbb{E}[X\mid A_i]\,\mathbf{1}_{A_i}(\omega)\quad\text{a.s.},
$$
where $\mathbb{E}[X\mid A_i]$ is the elementary conditional expectation with respect to the measure $\mathbb{P}(\cdot\mid A_i)$ of Definition 1.10.

*Proof.* Call the right-hand side $Y$. It is constant on each atom, hence $\mathcal{G}$-measurable by Proposition 1.4; it takes finitely many finite values, hence is integrable. For (CE3) it suffices to check $G=A_j$ for each $j$, because every $G\in\mathcal{G}$ is a disjoint union of atoms and both sides of (CE3) are additive. And
$$
\int_{A_j}Y\,d\mathbb{P}=\frac{1}{\mathbb{P}(A_j)}\Bigl(\int_{A_j}X\,d\mathbb{P}\Bigr)\mathbb{P}(A_j)=\int_{A_j}X\,d\mathbb{P}. \qquad\blacksquare
$$

So on a finite partition, conditional expectation is *literally* the operation "replace $X$ by its average over the atom you are in". Everything in this chapter is a generalisation of that sentence. Figure 6.1 is the picture to keep.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A function X on the sample space and its conditional expectation, constant on each atom of a partition">
  <line x1="70" y1="320" x2="660" y2="320" stroke="currentColor" stroke-width="1.5"/>
  <line x1="70" y1="320" x2="70" y2="50" stroke="currentColor" stroke-width="1.5"/>
  <text x="30" y="55" fill="currentColor" font-size="13">value</text>
  <text x="600" y="340" fill="currentColor" font-size="13">ω ∈ Ω</text>
  <g stroke="currentColor" stroke-dasharray="4 4" stroke-width="1" opacity="0.6">
    <line x1="215" y1="320" x2="215" y2="60"/>
    <line x1="360" y1="320" x2="360" y2="60"/>
    <line x1="505" y1="320" x2="505" y2="60"/>
  </g>
  <polyline fill="none" stroke="currentColor" stroke-width="2" points="70,150 80,143 89,138 99,135 109,133 118,133 128,134 138,136 147,138 157,142 167,146 176,150 186,154 196,158 205,161 215,164 225,166 234,168 244,169 254,169 263,168 273,166 283,164 292,162 302,160 312,157 321,155 331,153 341,152 350,152 360,153 370,154 379,158 389,162 399,168 408,175 418,183 428,193 437,203 447,214 457,225 466,237 476,249 486,260 495,271 505,280 515,288 524,295 534,300 544,303 553,304 563,303 573,300 582,295 592,288 602,279 611,268 621,256 631,242 640,228 650,213"/>
  <g stroke="#3b82f6" stroke-width="3.5" fill="none">
    <line x1="70" y1="144" x2="215" y2="144"/>
    <line x1="215" y1="161" x2="360" y2="161"/>
    <line x1="360" y1="205" x2="505" y2="205"/>
    <line x1="505" y1="280" x2="650" y2="280"/>
  </g>
  <line x1="70" y1="197" x2="660" y2="197" stroke="#10b981" stroke-width="2" stroke-dasharray="8 5"/>
  <text x="668" y="201" fill="#10b981" font-size="12">E[X]</text>
  <text x="120" y="132" fill="currentColor" font-size="13">X(ω)</text>
  <text x="230" y="155" fill="#3b82f6" font-size="13">E[X | G]</text>
  <g fill="currentColor" font-size="12" text-anchor="middle">
    <text x="142" y="340">atom A₁</text>
    <text x="287" y="340">atom A₂</text>
    <text x="432" y="340">atom A₃</text>
    <text x="577" y="340">atom A₄</text>
  </g>
  <text x="70" y="372" fill="currentColor" font-size="13">Figure 6.1 — Conditioning on a finite σ-algebra flattens X within each atom. The blue step function is</text>
  <text x="70" y="389" fill="currentColor" font-size="13">E[X | G]: on each atom its height is the average of X there, so the signed areas above and below the curve cancel</text>
  <text x="70" y="406" fill="currentColor" font-size="13">atom by atom. Coarsening G to {∅, Ω} collapses all four steps onto the single green line E[X].</text>
</svg>

**Worked Example 6.2 (three coin flips — Chapter 1's running example, finally conditioned).**
Recall $\Omega=\{H,T\}^3$ with the uniform measure ($\mathbb{P}(\{\omega\})=1/8$) and $X(\omega)=$ number of heads. Let $\mathcal{F}_1=\sigma(\text{first flip})$ and $\mathcal{F}_2=\sigma(\text{first two flips})$, so $\mathcal{F}_1\subset\mathcal{F}_2\subset\mathcal{F}=2^\Omega$ is a filtration: information accumulating flip by flip.

$\mathcal{F}_1$ has two atoms, $H{*}{*}$ and $T{*}{*}$, each of probability $1/2$ and containing four outcomes. Averaging $X$ over them (computed exactly by enumeration):
$$
\mathbb{E}[X\mid\mathcal{F}_1]=\begin{cases}2.0 & \text{on }\{\text{first}=H\},\\[2pt] 1.0 & \text{on }\{\text{first}=T\}.\end{cases}
$$
Sanity check by hand: given the first flip is a head, you already have one head and expect $2\times\tfrac12=1$ more, total $2$. $\mathcal{F}_2$ has four atoms:
$$
\mathbb{E}[X\mid\mathcal{F}_2]=2.5\ \text{on }HH{*},\quad 1.5\ \text{on }HT{*},\quad 1.5\ \text{on }TH{*},\quad 0.5\ \text{on }TT{*}.
$$
Now iterate. Averaging $\mathbb{E}[X\mid\mathcal{F}_2]$ over the two atoms of $\mathcal{F}_1$: on $\{\text{first}=H\}$ we average $2.5$ and $1.5$ with equal weights to get $2.0$; on $\{\text{first}=T\}$, $1.5$ and $0.5$ give $1.0$. So
$$
\mathbb{E}\bigl[\mathbb{E}[X\mid\mathcal{F}_2]\;\bigm|\;\mathcal{F}_1\bigr]=\mathbb{E}[X\mid\mathcal{F}_1],
$$
which is the tower property (Theorem 6.8(c)) verified numerically before we prove it. Averaging once more, over $\Omega$: $\tfrac12(2.0)+\tfrac12(1.0)=1.5=\mathbb{E}[X]$. The sequence $M_0=1.5$, $M_1=\mathbb{E}[X\mid\mathcal{F}_1]$, $M_2=\mathbb{E}[X\mid\mathcal{F}_2]$, $M_3=X$ is our first martingale (§6.7), a *Doob martingale*: partial information about a fixed quantity, refined step by step.

**Worked Example 6.3 (two dice — a conditional expectation that is not constant).**
Chapter 1's example (iv): $X$ is the first die, $Y$ the sum of two fair dice. Here $\sigma(X)$ has six atoms $\{X=i\}$, each of probability $1/6$, and on $\{X=i\}$ the sum is $i+D_2$ with $D_2$ uniform on $\{1,\dots,6\}$ and independent of the atom. Hence
$$
\mathbb{E}[Y\mid X]=X+3.5,\qquad\text{i.e. }\ \mathbb{E}[Y\mid X=i]=i+3.5\in\{4.5,5.5,6.5,7.5,8.5,9.5\}.
$$
Doob–Dynkin's $g$ is here explicit: $g(i)=i+3.5$. Tower: $\mathbb{E}[X+3.5]=3.5+3.5=7=\mathbb{E}[Y]$. ✓

Two lessons. First, $\mathbb{E}[Y\mid X]$ is a *random variable* — it varies with $\omega$ through $X$ — whereas $\mathbb{E}[Y\mid X=3]=6.5$ is a number. Second, this example was used in Chapter 1 to show that $\mathbb{P}(X=1,Y=7)=\mathbb{P}(X=1)\mathbb{P}(Y=7)$ exactly ($1/36$ on both sides) even though $X$ and $Y$ are dependent. Conditional expectation exposes the dependence immediately: if $X\perp Y$ then $\mathbb{E}[Y\mid X]$ would be the constant $\mathbb{E}[Y]=7$ (Theorem 6.8(e)); it is not, so they are dependent. Checking one factorised pair proves nothing; checking $\mathbb{E}[Y\mid X]$ proves a lot.

**Worked Example 6.4 (the two extremes, and independence).**
(i) $\mathcal{G}=\{\varnothing,\Omega\}$, the trivial $\sigma$-algebra (you know nothing). A $\mathcal{G}$-measurable function is constant; (CE3) with $G=\Omega$ forces the constant to be $\mathbb{E}[X]$. So $\mathbb{E}[X\mid\{\varnothing,\Omega\}]=\mathbb{E}[X]$.
(ii) $\mathcal{G}=\mathcal{F}$, or more generally $X$ is $\mathcal{G}$-measurable (you know everything relevant). Then $Y=X$ satisfies (CE1)–(CE3) trivially, so $\mathbb{E}[X\mid\mathcal{G}]=X$ a.s.
(iii) If $\sigma(X)$ is independent of $\mathcal{G}$ then for $G\in\mathcal{G}$, $\int_G X\,d\mathbb{P}=\mathbb{E}[X\mathbf{1}_G]=\mathbb{E}[X]\mathbb{P}(G)=\int_G\mathbb{E}[X]\,d\mathbb{P}$, so the constant $\mathbb{E}[X]$ is a version: independence means the information is useless. Conditional expectation therefore interpolates between "know nothing" ($\mathbb{E}[X]$) and "know everything" ($X$), and the size of $\operatorname{Var}(\mathbb{E}[X\mid\mathcal{G}])$ measures how much $\mathcal{G}$ actually tells you about $X$ — a statement made precise in §6.5.

**Worked Example 6.5 (Chapter 1's $A=\max(0,Z)$, conditioned on its sign).**
Let $Z\sim N(0,1)$ and $A=\max(0,Z)$ — the random variable that is neither discrete nor continuous, with $\mathbb{P}(A=0)=1/2$ and $\mathbb{E}[A]=1/\sqrt{2\pi}=0.3989$. This is the ReLU of a standard normal pre-activation, and its conditional structure is worth seeing explicitly.

Let $S=\mathbf{1}\{Z>0\}$ and $\mathcal{G}=\sigma(S)$, a two-atom $\sigma$-algebra with $\mathbb{P}(S=1)=\mathbb{P}(S=0)=1/2$. By Proposition 6.4,
$$
\mathbb{E}[A\mid\mathcal{G}]=\mathbb{E}[A\mid Z>0]\mathbf{1}\{Z>0\}+\mathbb{E}[A\mid Z\le0]\mathbf{1}\{Z\le0\}.
$$
On $\{Z\le0\}$, $A\equiv0$, so that term is $0$. On $\{Z>0\}$, $A=Z$ and the half-normal mean is
$$
\mathbb{E}[Z\mid Z>0]=\frac{\int_0^\infty z\,\varphi(z)\,dz}{1/2}=\frac{\varphi(0)}{1/2}=2\varphi(0)=\sqrt{2/\pi}=0.797885 ,
$$
using $\int_0^\infty z\varphi(z)dz=\varphi(0)=1/\sqrt{2\pi}$. So $\mathbb{E}[A\mid\mathcal{G}]$ takes the value $0.797885$ with probability $1/2$ and $0$ with probability $1/2$. Tower check:
$$
\mathbb{E}\bigl[\mathbb{E}[A\mid\mathcal{G}]\bigr]=\tfrac12(0.797885)+\tfrac12(0)=0.398942=\frac{1}{\sqrt{2\pi}}=\mathbb{E}[A].\ \checkmark
$$
(All figures computed exactly and reported to six decimals.) Notice that conditioning on one bit of information — the sign — decomposed a random variable that has *neither a pmf nor a pdf* into two pieces, each of which is elementary: a point mass and a half-normal. That is Failure 2 of the Four Failures being handled from a new direction; Chapter 2 dispatched it with Radon–Nikodym, and here the same object is being dissected by conditioning.

> **Common Confusion 6.2.**
> *"Isn't $\mathbb{E}[X\mid\mathcal{G}]$ a number? Conditional expectation in my first course was $\mathbb{E}[X\mid A]=\sum_x x\,\mathbb{P}(X=x\mid A)$, which is definitely a number."*
>
> $\mathbb{E}[X\mid A]$ for a *single event* $A$ with $\mathbb{P}(A)>0$ is a number. $\mathbb{E}[X\mid\mathcal{G}]$ for a *$\sigma$-algebra* is a random variable: it assembles all those numbers into one function of $\omega$. Proposition 6.4 shows exactly how — the random variable equals $\mathbb{E}[X\mid A_i]$ on atom $A_i$. The upgrade from number to random variable is what makes composition possible: you can condition a conditional expectation again (the tower property), you can multiply it by other random variables, you can plug it into a loss. You could not do any of that with a number, and it is precisely composition that the SGD proof in §6.9 needs, since there we condition at every time step on a growing filtration.

> **Common Confusion 6.3.**
> *"If $\mathbb{E}[X\mid\mathcal{G}]$ is only defined up to a null set, isn't it useless? I want to know its value at my data point $x_0$."*
>
> You want $g(x_0)$ where $g$ is the Doob–Dynkin function. Different versions of $g$ can disagree at $x_0$, and one can always cook up a version that is, say, $10^6$ at $x_0$, provided $\mathbb{P}_X(\{x_0\})=0$. So "the value at $x_0$" is genuinely not determined by the joint law. What *is* determined: $g$ up to $\mathbb{P}_X$-null sets, hence every integral $\int g\,dQ$ for $Q\ll\mathbb{P}_X$, hence every risk, every average prediction over a test set drawn from $\mathbb{P}_X$, and every a.s. statement. In practice you break the tie with a *regularity assumption*: if you additionally know $g$ is continuous, then the continuous version is unique (two continuous functions agreeing $\mathbb{P}_X$-a.e. agree on $\operatorname{supp}(\mathbb{P}_X)$, by Definition 1.17). This is one reason smoothness assumptions on the regression function are not merely for convenience: without them, pointwise statements about $f^\star$ are not even well posed. It is also a reason to be careful about model predictions far off the data manifold — there, the "true" conditional expectation is not just hard to estimate, it is undefined.

> **Self-check 6.2.**
> 1. Which of (CE1), (CE2), (CE3) fails for the candidate $Y\equiv\mathbb{E}[X]$ when $\mathcal{G}=\sigma(X)$ and $X$ is non-constant?
> 2. In Worked Example 6.2 compute $\mathbb{E}[X\mid\mathcal{F}_2]$ on the atom $TH{*}$ directly from the definition, by integrating over that atom.
> 3. Where in the proof of Theorem 6.2 is the assumption $\mathcal{G}\subseteq\mathcal{F}$ used? Where is $\sigma$-finiteness used?
> 4. True or false: if $Y$ is a version of $\mathbb{E}[X\mid\mathcal{G}]$ and $Y'=Y$ except on a set of probability $0$, then $Y'$ is also a version.

## 6.3 Conditional expectation as projection

### 6.3.1 The geometry

Chapter 2 established that $L^2(\Omega,\mathcal{F},\mathbb{P})$ is a Hilbert space with inner product $\langle U,V\rangle=\mathbb{E}[UV]$ and norm $\|U\|_2=(\mathbb{E}[U^2])^{1/2}$, and that a closed subspace admits an orthogonal projection: every vector splits uniquely into a component in the subspace and a component perpendicular to it. That single theorem, applied to the subspace of things you are allowed to compute, is conditional expectation.

**Theorem 6.5 ($\mathbb{E}[\cdot\mid\mathcal{G}]$ is orthogonal projection).**
Let $X\in L^2(\Omega,\mathcal{F},\mathbb{P})$ and $\mathcal{G}\subseteq\mathcal{F}$. Write $\mathcal{H}_{\mathcal{G}}:=L^2(\Omega,\mathcal{G},\mathbb{P})$, viewed as a subspace of $L^2(\Omega,\mathcal{F},\mathbb{P})$. Then:

1. $\mathcal{H}_{\mathcal{G}}$ is a closed subspace;
2. the orthogonal projection $\Pi_{\mathcal{G}}X$ of $X$ onto $\mathcal{H}_{\mathcal{G}}$ is a version of $\mathbb{E}[X\mid\mathcal{G}]$;
3. consequently $\mathbb{E}[X\mid\mathcal{G}]$ solves $\min\{\mathbb{E}[(X-Y)^2]:Y\in\mathcal{H}_{\mathcal{G}}\}$, and the minimiser is a.s. unique;
4. **Pythagoras / excess-risk identity:** for every $Y\in\mathcal{H}_{\mathcal{G}}$,
$$
\mathbb{E}\bigl[(X-Y)^2\bigr]=\underbrace{\mathbb{E}\bigl[(X-\mathbb{E}[X\mid\mathcal{G}])^2\bigr]}_{\text{irreducible given }\mathcal{G}}+\underbrace{\mathbb{E}\bigl[(\mathbb{E}[X\mid\mathcal{G}]-Y)^2\bigr]}_{\text{excess}} .
$$

*Proof.* (1) $\mathcal{H}_{\mathcal{G}}$ is a linear subspace since sums and scalar multiples of $\mathcal{G}$-measurable functions are $\mathcal{G}$-measurable (Proposition 1.15). It is complete: $L^2(\Omega,\mathcal{G},\mathbb{P}|_{\mathcal{G}})$ is complete by the Riesz–Fischer theorem (Ch. 2), and an $L^2(\mathcal{G})$-Cauchy sequence has an $L^2(\mathcal{G})$ limit, which is in particular its $L^2(\mathcal{F})$ limit. A complete subspace of a metric space is closed.

(2) By the projection theorem (Ch. 2), $R:=X-\Pi_{\mathcal{G}}X$ satisfies $\langle R,V\rangle=0$ for every $V\in\mathcal{H}_{\mathcal{G}}$. Fix $G\in\mathcal{G}$; then $\mathbf{1}_G\in\mathcal{H}_{\mathcal{G}}$ because $\mathbb{P}$ is finite, so
$$
0=\langle R,\mathbf{1}_G\rangle=\mathbb{E}\bigl[(X-\Pi_{\mathcal{G}}X)\mathbf{1}_G\bigr]\iff\int_G\Pi_{\mathcal{G}}X\,d\mathbb{P}=\int_G X\,d\mathbb{P},
$$
which is (CE3). (CE1) holds by construction. (CE2): on a probability space $L^2\subseteq L^1$ by Cauchy–Schwarz, $\mathbb{E}|U|\le\|U\|_2$. So $\Pi_{\mathcal{G}}X$ satisfies Definition 6.1, and by the a.s. uniqueness in Theorem 6.2 it *is* the conditional expectation.

(3)–(4) For $Y\in\mathcal{H}_{\mathcal{G}}$ write $X-Y=R+(\Pi_{\mathcal{G}}X-Y)$ with $R\perp(\Pi_{\mathcal{G}}X-Y)$ since the second term lies in $\mathcal{H}_{\mathcal{G}}$. Expanding the square and dropping the vanishing cross term gives (4); the right side is minimised exactly when $\mathbb{E}[(\Pi_{\mathcal{G}}X-Y)^2]=0$, i.e. $Y=\Pi_{\mathcal{G}}X$ a.s. $\blacksquare$

<svg viewBox="0 0 720 420" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Orthogonal projection of X onto the subspace of G-measurable square-integrable functions">
  <polygon points="60,300 480,300 660,210 240,210" fill="currentColor" opacity="0.07" stroke="currentColor" stroke-width="1.2"/>
  <text x="500" y="290" fill="currentColor" font-size="13">H_G = L²(Ω, G, P)  —  all predictors you can compute from G</text>
  <circle cx="300" cy="90" r="5" fill="#3b82f6"/>
  <text x="308" y="86" fill="#3b82f6" font-size="14">X   (the target, F-measurable)</text>
  <circle cx="300" cy="258" r="5" fill="#10b981"/>
  <text x="252" y="285" fill="#10b981" font-size="14">E[X | G]</text>
  <line x1="300" y1="90" x2="300" y2="258" stroke="#ef4444" stroke-width="2.5"/>
  <text x="308" y="180" fill="#ef4444" font-size="13">residual X − E[X | G]</text>
  <text x="308" y="197" fill="#ef4444" font-size="13">⊥ every G-measurable V</text>
  <polyline points="300,244 314,244 314,258" fill="none" stroke="#ef4444" stroke-width="1.5"/>
  <circle cx="470" cy="240" r="4.5" fill="currentColor"/>
  <text x="478" y="236" fill="currentColor" font-size="13">Y  (any other G-measurable predictor)</text>
  <line x1="300" y1="90" x2="470" y2="240" stroke="currentColor" stroke-width="1.6" stroke-dasharray="5 4"/>
  <line x1="300" y1="258" x2="470" y2="240" stroke="#10b981" stroke-width="1.8" stroke-dasharray="5 4"/>
  <text x="330" y="330" fill="currentColor" font-size="13">‖X − Y‖² = ‖X − E[X|G]‖² + ‖E[X|G] − Y‖²   (Pythagoras = excess-risk identity)</text>
  <text x="60" y="372" fill="currentColor" font-size="13">Figure 6.2 — Conditional expectation is the foot of the perpendicular from X to the plane of G-measurable</text>
  <text x="60" y="389" fill="currentColor" font-size="13">predictors. Any competitor Y sits in the same plane, so its squared error exceeds the minimum by exactly the</text>
  <text x="60" y="406" fill="currentColor" font-size="13">squared distance from E[X|G] to Y. "Learning" is the search for that foot; the red leg is the Bayes error.</text>
</svg>

### 6.3.2 Regression is projection

Specialise to $\mathcal{G}=\sigma(X)$ with $X$ the features and $Y$ the label, and add Doob–Dynkin. A $\sigma(X)$-measurable random variable is exactly $g(X)$ for measurable $g$ (Lemma 1.22), so $\mathcal{H}_{\sigma(X)}$ is the set of all measurable functions of the features with finite second moment — *every* possible predictor, linear or not, deep or shallow, that uses only the features. We get:

**Corollary 6.6 (the Bayes-optimal predictor under squared loss).**
Let $\mathbb{E}[Y^2]<\infty$ and define $f^\star(x):=\mathbb{E}[Y\mid X=x]$ (a Doob–Dynkin version). Then for every measurable $f$ with $\mathbb{E}[f(X)^2]<\infty$,
$$
\underbrace{\mathbb{E}\bigl[(Y-f(X))^2\bigr]}_{\text{risk }R(f)}=\underbrace{\mathbb{E}\bigl[(Y-f^\star(X))^2\bigr]}_{\text{Bayes risk}}+\underbrace{\mathbb{E}\bigl[(f^\star(X)-f(X))^2\bigr]}_{\text{excess risk}\;=\;\|f-f^\star\|^2_{L^2(\mathbb{P}_X)}},
$$
so $f^\star$ minimises the squared-error risk, uniquely up to $\mathbb{P}_X$-null sets.

This is the theorem that licenses the entire practice of least-squares regression. Say it plainly: **fitting a network by minimising mean squared error is an attempt to compute an orthogonal projection, and the thing it is trying to hit is the conditional mean.** Three consequences worth internalising.

*The excess risk is an $L^2(\mathbb{P}_X)$ distance.* Not an $L^\infty$ distance, and not a distance on all of $\mathbb{R}^d$: the metric is weighted by where the features actually live. A model that is wildly wrong on a region of feature space with $\mathbb{P}_X$-measure $10^{-6}$ has excess risk at most $10^{-6}\times(\text{error there})^2$. This is why test-set MSE is blind to catastrophic off-distribution behaviour, and why distribution shift is not a detail: change $\mathbb{P}_X$ and you change the norm in which you were optimal.

*The Bayes risk does not depend on the model class.* It is $\mathbb{E}[\operatorname{Var}(Y\mid X)]$ (§6.5), a property of the data-generating process. No architecture reduces it. Reported MSE floors in benchmark papers are floors for this reason. A measured test MSE *below* a claimed Bayes risk therefore means one of three things, in rough order of likelihood: the claimed Bayes risk was estimated too high, the test set is small enough that the gap is sampling noise, or there is leakage between train and test. It does not mean the Bayes bound was beaten.

*Everything above holds for vector-valued $Y$ coordinatewise*, so the same statement covers multi-output regression, and with $Y=\mathbf{1}\{\text{class}=k\}$ it covers the probabilistic output of a classifier trained with squared loss: $f^\star_k(x)=\mathbb{P}(\text{class}=k\mid X=x)$.

**Worked Example 6.6 (verifying the projection numerically).**
Let $X\sim N(0,1)$ and $Y=X^2+\varepsilon$ with $\varepsilon\sim N(0,0.5^2)$ independent. Then $f^\star(x)=x^2$ and the Bayes risk is $\operatorname{Var}(\varepsilon)=0.25$. Perturb: take $f=f^\star+c\,h$ for various $h$ and $c$. Corollary 6.6 predicts excess risk exactly $c^2\mathbb{E}[h(X)^2]$. Simulation with $n=2\times10^6$ samples, seed 6 (NumPy `default_rng(6)`):

| predictor | MSE | excess | predicted excess |
|---|---|---|---|
| $f^\star(x)=x^2$ | 0.24993 | — | 0 |
| $x^2+0.1$ | 0.25988 | 0.00995 | $0.1^2\cdot1=0.01$ |
| $x^2+0.3$ | 0.33977 | 0.08984 | $0.09$ |
| $x^2+0.1x$ | 0.25994 | 0.01001 | $0.01^{\vphantom{2}}\cdot\mathbb{E}[X^2]=0.01$ |
| $x^2+0.3\sin 3x$ | 0.29528 | 0.04535 | $0.09\cdot\tfrac{1-e^{-18}}{2}=0.045$ |

Every perturbation makes things worse, by the predicted amount, in whichever direction you move. That is what "foot of the perpendicular" means operationally.

**Worked Example 6.7 (the best *linear* predictor is not the conditional mean).**
Same $(X,Y)$. Compute $\operatorname{Cov}(X,Y)=\mathbb{E}[X^3]=0$: $X$ and $Y$ are uncorrelated. The best linear predictor is therefore the constant $\mathbb{E}[Y]=1$, with risk $\operatorname{Var}(Y)=\operatorname{Var}(X^2)+0.25=2+0.25=2.25$. The conditional mean has risk $0.25$. Simulation ($n=4\times10^6$, seed 31): OLS returns intercept $1.0006$, slope $0.0002$, MSE $2.2502$; the conditional mean gives MSE $0.2501$. **Nine times the risk**, purely from restricting the projection to a two-dimensional subspace.

Linear regression is the projection onto $\operatorname{span}\{1,X\}$, not onto $\mathcal{H}_{\sigma(X)}$. Both are projections; the difference is the subspace. The whole of representation learning can be read as an attempt to enlarge the subspace enough that its projection is close to $f^\star$ while remaining estimable from $n$ samples — Chapter 8's bias–complexity trade-off in geometric clothing. Note also the by-product: $X$ and $Y$ here are uncorrelated yet emphatically dependent, which is Chapter 1's lesson about independence sharpened.

### 6.3.3 Other losses, other summaries

Squared loss gives the mean. That is a fact about squared loss, not about prediction. The general pattern: *choosing a loss is choosing which summary of the conditional distribution you will estimate.*

**Proposition 6.7 (Bayes predictors for absolute and 0–1 loss).**
Let $(X,Y)$ be a random pair.

(a) *(Absolute loss $\to$ conditional median.)* Suppose $\mathbb{E}|Y|<\infty$. For $\mathbb{P}_X$-a.e. $x$, any minimiser over $a\in\mathbb{R}$ of $a\mapsto\mathbb{E}[|Y-a|\mid X=x]$ is a median of the conditional distribution of $Y$ given $X=x$, and conversely every such median is a minimiser. Hence $f_{\mathrm{med}}(x)=\operatorname{med}(Y\mid X=x)$ minimises $\mathbb{E}|Y-f(X)|$.

(b) *(Pinball loss $\to$ conditional quantile.)* For $\tau\in(0,1)$ and $\rho_\tau(u)=u(\tau-\mathbf{1}\{u<0\})$, any minimiser of $\mathbb{E}[\rho_\tau(Y-a)\mid X=x]$ is a $\tau$-quantile of the conditional law.

(c) *(0–1 loss $\to$ conditional mode.)* For $Y\in\{0,1\}$ and $\eta(x)=\mathbb{P}(Y=1\mid X=x)$, the classifier $h^\star(x)=\mathbf{1}\{\eta(x)\ge 1/2\}$ minimises $\mathbb{P}(h(X)\ne Y)$ among all measurable $h:\mathbb{R}^d\to\{0,1\}$, and for any $h$,
$$
\mathbb{P}(h(X)\ne Y)-\mathbb{P}(h^\star(X)\ne Y)=\mathbb{E}\bigl[\,|2\eta(X)-1|\,\mathbf{1}\{h(X)\ne h^\star(X)\}\,\bigr].
$$
The multiclass version is identical in structure: for $Y\in\{1,\dots,K\}$ with $\eta_k(x)=\mathbb{P}(Y=k\mid X=x)$, the Bayes classifier is $h^\star(x)=\arg\max_k\eta_k(x)$ (any measurable tie-breaking rule), because at each $x$ one is choosing which of the numbers $1-\eta_k(x)$ to pay.

(d) *(Log loss $\to$ the conditional distribution itself.)* Let $Y\in\{1,\dots,K\}$ and let the predictor output a probability vector $q(x)=(q_1(x),\dots,q_K(x))$. The log-loss (cross-entropy) risk is $\mathbb{E}\bigl[-\log q_Y(X)\bigr]$. Its unique minimiser over measurable probability-vector-valued $q$ is the true conditional distribution, $q_k^\star(x)=\eta_k(x)$ for $\mathbb{P}_X$-a.e. $x$, and the excess risk of any $q$ is exactly the average KL divergence to the truth:
$$
\mathbb{E}\bigl[-\log q_Y(X)\bigr]-\mathbb{E}\bigl[-\log \eta_Y(X)\bigr]\;=\;\mathbb{E}_X\Bigl[D_{\mathrm{KL}}\bigl(\eta(X)\,\big\|\,q(X)\bigr)\Bigr]\;\ge\;0 .
$$

*Proof.* (a) Work with the conditional law $\mu_x$ of $Y$ given $X=x$ (which exists by §6.6; for now read the argument with $\mu_x$ any probability measure with $\int|y|d\mu_x<\infty$). For $a<b$ and any $y$, using $\frac{d}{dt}|y-t|=2\mathbf{1}\{y<t\}-1$ for $t\neq y$,
$$
|y-b|-|y-a|=\int_a^b\bigl(2\mathbf{1}\{y<t\}-1\bigr)\,dt .
$$
The integrand is bounded, so Fubini (Ch. 2) permits integration against $\mu_x$:
$$
\psi(b)-\psi(a)=\int_a^b\bigl(2F_x(t^-)-1\bigr)dt,\qquad \psi(a):=\int|y-a|\,\mu_x(dy),
$$
where $F_x$ is the conditional CDF. The integrand $2F_x(t^-)-1$ is non-decreasing in $t$, so $\psi$ is convex, and it is non-increasing while $F_x(t^-)<1/2$ and non-decreasing once $F_x(t^-)>1/2$. Hence the minimisers of $\psi$ are exactly the points $m$ with $F_x(m^-)\le 1/2\le F_x(m)$ — the medians. Choosing a measurable version $f_{\mathrm{med}}$ (e.g. the smallest median, which is measurable in $x$) and integrating over $x$ gives the global statement, since the risk decomposes as $\mathbb{E}_X[\psi_X(f(X))]$ by the tower property.

(b) Identical, replacing the integrand by $\tau-\mathbf{1}\{y<t\}$; minimisers are the $t$ with $F_x(t^-)\le\tau\le F_x(t)$.

(c) Condition on $X$: $\mathbb{P}(h(X)\ne Y)=\mathbb{E}\bigl[\eta(X)\mathbf{1}\{h(X)=0\}+(1-\eta(X))\mathbf{1}\{h(X)=1\}\bigr]$. For each $x$ we choose between the two numbers $\eta(x)$ and $1-\eta(x)$; picking the smaller means predicting $1$ iff $\eta(x)\ge1/2$. The difference between the chosen and optimal value at $x$ is $|2\eta(x)-1|$ when $h(x)\ne h^\star(x)$ and $0$ otherwise; take expectations. The multiclass case is the same one-line comparison at each $x$.

(d) Condition on $X=x$ (tower property again). The conditional expected loss of the vector $q=q(x)$ is $-\sum_k\eta_k(x)\log q_k$, and
$$
-\sum_k\eta_k\log q_k\;+\;\sum_k\eta_k\log\eta_k\;=\;\sum_k\eta_k\log\frac{\eta_k}{q_k}\;=\;D_{\mathrm{KL}}(\eta\,\|\,q),
$$
which is $\ge0$ with equality iff $q=\eta$: by Jensen's inequality (Ch. 4) applied to the strictly convex $-\log$ under the probability vector $\eta$, $\ \sum_k\eta_k\log\frac{\eta_k}{q_k}=\sum_k\eta_k\bigl(-\log\tfrac{q_k}{\eta_k}\bigr)\ge-\log\sum_k\eta_k\tfrac{q_k}{\eta_k}=-\log\sum_kq_k\ge0$, with equality forcing $q_k/\eta_k$ constant, hence $q=\eta$ (Gibbs' inequality). Minimising pointwise in $x$ and integrating over $\mathbb{P}_X$ gives both the identification of the minimiser and the excess-risk formula. $\blacksquare$

Part (d) completes the pattern of this section, and it is the one you meet most often in deep learning. Squared loss extracts one number from the conditional law (the mean); absolute loss another (the median); pinball a quantile; 0–1 a mode. **Log loss extracts everything: the Bayes-optimal output under cross-entropy is the conditional distribution itself**, and the price of being wrong is measured in KL. This is the precise sense in which a softmax classifier trained with cross-entropy is a *density estimator* of $p(y\mid x)$, and it is why calibration is even a meaningful demand: the training objective's optimum is calibrated by construction, so miscalibration is entirely a statement about the gap between the trained network and the Bayes optimum (finite data, restricted architecture, optimisation error) — not about the loss. Losses with this property, that their Bayes optimum is the true conditional law, are called *strictly proper scoring rules*; log loss and the (multiclass) Brier/squared loss are the two standard examples, and 0–1 loss is emphatically not one, which is why "accuracy went up" and "the predicted probabilities got better" are logically independent claims.

Part (c)'s identity is the standard entry point to classification theory: the excess 0–1 risk is small if disagreement happens only where $\eta$ is near $1/2$, which is why margin conditions (Tsybakov noise conditions) control classification rates.

**Worked Example 6.8 (mean versus median, and why your loss is a modelling choice).**
Let $X\sim\mathrm{Uniform}(0,1)$ and, given $X=x$, let $Y\sim\mathrm{Exponential}$ with mean $1+x$. Then
$$
f^\star(x)=\mathbb{E}[Y\mid X=x]=1+x,\qquad f_{\mathrm{med}}(x)=(1+x)\ln 2=0.693147(1+x).
$$
The median predictor is about $31\%$ below the mean predictor everywhere — a systematic, not random, difference, driven by skew. Simulation ($n=10^6$, seed 7):

| predictor | MSE | MAE |
|---|---|---|
| $f^\star(x)=1+x$ (conditional mean) | **2.3298** | 1.1030 |
| $f_{\mathrm{med}}(x)=0.693147(1+x)$ | 2.5473 | **1.0384** |

Each predictor wins under its own loss and loses under the other. There is no contradiction and no "better model": they estimate different functionals of the same conditional law. In practice this shows up whenever someone trains with MSE and evaluates with MAE (or MAPE) and is puzzled that the model "over-predicts". It is not over-predicting; it is reporting a mean where a median was wanted. Right-skewed targets — delivery times, click counts, incomes, latencies — are where the gap bites hardest.

> **Common Confusion 6.4.**
> *"Regression means fitting a line. Conditional expectation is a fancy name for the regression line, isn't it?"*
>
> Two different projections. $\mathbb{E}[Y\mid X]$ is the projection onto *all* measurable functions of $X$; the regression line is the projection onto the two-dimensional subspace $\operatorname{span}\{1,X\}$. They coincide exactly when $f^\star$ happens to be affine — true for jointly Gaussian $(X,Y)$ (Ch. 5, and Worked Example 6.14 below), which is why the Gaussian case is so seductive and so misleading as a mental default. Worked Example 6.7 exhibits a pair with $\operatorname{Cov}(X,Y)=0$ and $f^\star(x)=x^2$: the line is flat and useless while the true conditional mean is a parabola. The failure of correlation-based reasoning on such data is not a small-sample problem; it is a subspace problem, and no amount of data fixes it.

> **Self-check 6.3.**
> 1. Why is $\mathcal{H}_{\mathcal{G}}$ closed, and where would the argument fail if we tried the same with $L^1$?
> 2. In Corollary 6.6, what is the Bayes risk if $Y=g(X)$ exactly, with no noise?
> 3. Give an example of a conditional law for which the median predictor is not unique. Does this contradict Proposition 6.7?
> 4. A colleague trains with MSE and reports that predictions are systematically higher than typical outcomes. Give the one-line diagnosis and the one-line fix.

## 6.4 The algebra of conditional expectation

Everything in this section is proved the same way: to show two random variables are a.s. equal, show one of them satisfies the three defining conditions for the other's conditional expectation, then invoke a.s. uniqueness (Theorem 6.2). For (d) we need the *standard machine* (the "ladder"): prove the claim for indicators, extend by linearity to simple functions, extend to non-negative measurable functions by monotone convergence, and finish by splitting into positive and negative parts. You have seen this ladder in Chapter 2 when the Lebesgue integral was built; it is the same ladder.

**Theorem 6.8 (basic properties).** Let $X,X_1,X_2\in L^1(\mathbb{P})$, $a,b\in\mathbb{R}$, and $\mathcal{H}\subseteq\mathcal{G}\subseteq\mathcal{F}$ sub-$\sigma$-algebras. All statements hold a.s.

&nbsp;&nbsp;(a) **Linearity.** $\mathbb{E}[aX_1+bX_2\mid\mathcal{G}]=a\,\mathbb{E}[X_1\mid\mathcal{G}]+b\,\mathbb{E}[X_2\mid\mathcal{G}]$.
&nbsp;&nbsp;(b) **Monotonicity.** $X_1\le X_2$ a.s. $\Rightarrow \mathbb{E}[X_1\mid\mathcal{G}]\le\mathbb{E}[X_2\mid\mathcal{G}]$. In particular $|\mathbb{E}[X\mid\mathcal{G}]|\le\mathbb{E}[|X|\mid\mathcal{G}]$.
&nbsp;&nbsp;(c) **Tower property.** $\mathbb{E}\bigl[\mathbb{E}[X\mid\mathcal{G}]\mid\mathcal{H}\bigr]=\mathbb{E}[X\mid\mathcal{H}]$; with $\mathcal{H}=\{\varnothing,\Omega\}$ this is the law of total expectation $\mathbb{E}\bigl[\mathbb{E}[X\mid\mathcal{G}]\bigr]=\mathbb{E}[X]$.
&nbsp;&nbsp;(d) **Taking out what is known.** If $Z$ is $\mathcal{G}$-measurable and $ZX\in L^1$, then $\mathbb{E}[ZX\mid\mathcal{G}]=Z\,\mathbb{E}[X\mid\mathcal{G}]$.
&nbsp;&nbsp;(e) **Independence kills conditioning.** If $\sigma(X)\perp\mathcal{G}$ then $\mathbb{E}[X\mid\mathcal{G}]=\mathbb{E}[X]$.
&nbsp;&nbsp;(f) **Irrelevance of independent information.** If $\mathcal{H}'\subseteq\mathcal{F}$ is independent of $\sigma(X)\vee\mathcal{G}$ (the smallest $\sigma$-algebra containing both), then
$$
\mathbb{E}\bigl[X\mid\sigma(\mathcal{G}\cup\mathcal{H}')\bigr]=\mathbb{E}[X\mid\mathcal{G}]\quad\text{a.s.}
$$
Adding information that is independent of *everything relevant* — of $X$ **and** of what you already know — changes nothing. Note the hypothesis carefully: it is independence from the *pair*, not merely from $X$. A statement one might hope for instead — that $\mathcal{H}'\perp\sigma(X)$ alone suffices — is false; see the caution after the proof.

*Proof.* (a) The right-hand side is $\mathcal{G}$-measurable and integrable; for $G\in\mathcal{G}$, $\int_G(a\mathbb{E}[X_1|\mathcal{G}]+b\mathbb{E}[X_2|\mathcal{G}])=a\int_GX_1+b\int_GX_2=\int_G(aX_1+bX_2)$ by linearity of the integral.

(b) Let $D:=\mathbb{E}[X_1|\mathcal{G}]-\mathbb{E}[X_2|\mathcal{G}]$ and $G:=\{D\ge\varepsilon\}\in\mathcal{G}$. Then $\varepsilon\mathbb{P}(G)\le\int_G D\,d\mathbb{P}=\int_G(X_1-X_2)d\mathbb{P}\le0$, so $\mathbb{P}(G)=0$; let $\varepsilon\downarrow0$ along $1/k$ and use the union bound. The second claim follows from $\pm X\le|X|$ and (a).

(c) The candidate $\mathbb{E}[X\mid\mathcal{H}]$ is $\mathcal{H}$-measurable and integrable. For $H\in\mathcal{H}$ we have $H\in\mathcal{G}$ too (since $\mathcal{H}\subseteq\mathcal{G}$), so
$$
\int_H\mathbb{E}[X\mid\mathcal{H}]\,d\mathbb{P}=\int_H X\,d\mathbb{P}=\int_H\mathbb{E}[X\mid\mathcal{G}]\,d\mathbb{P},
$$
the first equality by (CE3) for $\mathcal{H}$ and the second by (CE3) for $\mathcal{G}$ applied to the set $H$. That is exactly (CE3) for $\mathbb{E}[\,\cdot\mid\mathcal{H}]$ applied to the random variable $\mathbb{E}[X\mid\mathcal{G}]$. **Read the middle step again: it is the whole proof.** The tower property holds because a coarse event is also a fine event.

(d) Ladder. *Step 1: $Z=\mathbf{1}_{G_0}$, $G_0\in\mathcal{G}$.* The candidate $\mathbf{1}_{G_0}\mathbb{E}[X|\mathcal{G}]$ is $\mathcal{G}$-measurable and integrable; for $G\in\mathcal{G}$,
$$
\int_G\mathbf{1}_{G_0}\mathbb{E}[X|\mathcal{G}]\,d\mathbb{P}=\int_{G\cap G_0}\mathbb{E}[X|\mathcal{G}]\,d\mathbb{P}=\int_{G\cap G_0}X\,d\mathbb{P}=\int_G\mathbf{1}_{G_0}X\,d\mathbb{P},
$$
using $G\cap G_0\in\mathcal{G}$. *Step 2:* linearity extends this to simple $\mathcal{G}$-measurable $Z$. *Step 3:* for $Z\ge0$ take simple $Z_n\uparrow Z$ and $X\ge0$; monotone convergence applies to both sides of (CE3) (on the conditional side, note $Z_n\mathbb{E}[X|\mathcal{G}]\uparrow Z\mathbb{E}[X|\mathcal{G}]$ since $\mathbb{E}[X|\mathcal{G}]\ge0$ by (b)). *Step 4:* split $Z=Z^+-Z^-$, $X=X^+-X^-$ and combine, the integrability hypothesis $ZX\in L^1$ ensuring all four pieces are finite.

(e) Done in Worked Example 6.4(iii). $\blacksquare$

*Proof of (f).* The candidate $Y:=\mathbb{E}[X\mid\mathcal{G}]$ is $\sigma(\mathcal{G}\cup\mathcal{H}')$-measurable (it is already $\mathcal{G}$-measurable) and integrable, so only (CE3) needs checking, and here Chapter 1's pattern returns: check it on a *generating $\pi$-system* and let uniqueness of measures (Theorem 1.20) do the rest. The sets $\{G\cap H': G\in\mathcal{G},\,H'\in\mathcal{H}'\}$ form a $\pi$-system generating $\sigma(\mathcal{G}\cup\mathcal{H}')$. For such a set, using $\mathcal{H}'\perp\sigma(X)\vee\mathcal{G}$ (so $\mathbf{1}_{H'}$ is independent of both $X\mathbf{1}_G$ and $Y\mathbf{1}_G$):
$$
\int_{G\cap H'}X\,d\mathbb{P}=\mathbb{E}[X\mathbf{1}_G]\,\mathbb{P}(H')=\mathbb{E}[Y\mathbf{1}_G]\,\mathbb{P}(H')=\int_{G\cap H'}Y\,d\mathbb{P},
$$
the middle equality by (CE3) for $\mathcal{G}$. The two finite measures $E\mapsto\int_EX^+\!+Y^-\,d\mathbb{P}$ and $E\mapsto\int_EX^-\!+Y^+\,d\mathbb{P}$ agree on this $\pi$-system and on $\Omega$, hence on all of $\sigma(\mathcal{G}\cup\mathcal{H}')$ by Theorem 1.20, which is (CE3) in full. $\blacksquare$

One consequence of (f) is used so often that it deserves a name, the **freezing lemma** (stated in prose to keep the numbering clean; (f) itself is the step that legitimises the bias–variance proof of Corollary 6.14, and the freezing lemma is the step behind the Gaussian kernel of Worked Example 6.14): if $W\perp X$ (as random elements) and $h$ is measurable with $h(X,W)\in L^1$, then
$$
\mathbb{E}\bigl[h(X,W)\mid X\bigr]=\varphi(X)\ \text{ a.s.},\qquad \varphi(x):=\mathbb{E}[h(x,W)].
$$
*Freeze the variable you know; average the one you don't.* Proof by the ladder: for $h(x,w)=\mathbf{1}_A(x)\mathbf{1}_B(w)$ both sides equal $\mathbf{1}_A(X)\,\mathbb{P}(W\in B)$ by pull-out and (e); rectangles $A\times B$ form a $\pi$-system generating the product $\sigma$-algebra, so Theorem 1.20 extends the identity to indicators of all product-measurable sets; linearity extends it to simple $h$, monotone convergence (Ch. 2, on both sides, using Proposition 6.11(a) on the left) to non-negative $h$, and positive/negative parts to integrable $h$. Measurability of $\varphi$ is Fubini–Tonelli (Ch. 2).

*Caution: independence is fragile under enlargement.* Independence is not inherited by conditioning, and this is exactly why (f) demands independence from the *pair* $\sigma(X)\vee\mathcal{G}$ rather than from $X$ alone. $\sigma(X)\perp\mathcal{G}$ gives $\mathbb{E}[X\mid\mathcal{G}]=\mathbb{E}[X]$, but if $\mathcal{G}$ is enlarged by a third source of information, all bets are off: with $U,V$ independent fair coins and $W=U\oplus V$, we have $U\perp W$, yet $\mathbb{E}[U\mid W,V]=U$. Check (f)'s hypothesis against this example: $\sigma(V)$ is independent of $\sigma(U)$ and of $\sigma(W)$ separately, but *not* of $\sigma(U)\vee\sigma(W)$ — indeed $V$ is a function of $(U,W)$ — so (f) does not apply, and correctly so. Chapter 1's Definition 1.24 warned that conditional independence neither implies nor is implied by independence; this is the same phenomenon (Berkson's paradox / explaining away) in conditional-expectation form. In ML it is the reason that "feature $A$ is uncorrelated with the label" is worthless as a feature-selection criterion once other features are in the model.

**Worked Example 6.9 (pull-out, numerically).**
Three coin flips again, $X=\#$heads, $\mathcal{F}_1=\sigma(\text{first flip})$, $Z=\mathbf{1}\{\text{first}=H\}$ (which is $\mathcal{F}_1$-measurable). From Worked Example 6.2, $\mathbb{E}[X\mid\mathcal{F}_1]=2$ on $\{Z=1\}$ and $1$ on $\{Z=0\}$. Then $Z\mathbb{E}[X\mid\mathcal{F}_1]$ equals $2$ on $\{Z=1\}$ and $0$ on $\{Z=0\}$. Direct computation of $\mathbb{E}[ZX\mid\mathcal{F}_1]$: on the atom $\{Z=1\}$, $ZX=X$ and its average over that atom is $2$; on $\{Z=0\}$, $ZX\equiv0$. Same function. ✓ And $\mathbb{E}[ZX]=\tfrac12\cdot2=1$, matching $\mathbb{E}[X\mathbf{1}\{\text{first}=H\}]=\tfrac18(3+2+2+1)=1$. ✓

Property (d) is the workhorse of ML derivations. Three instances:

*Importance sampling / off-policy correction.* If $w$ is $\mathcal{G}$-measurable (e.g. a ratio computed from the state) then $\mathbb{E}[w\,r\mid\mathcal{G}]=w\,\mathbb{E}[r\mid\mathcal{G}]$, which is why importance weights pass through conditional expectations untouched.

*Baselines in REINFORCE.* For a score-function estimator with baseline $b(s)$ measurable w.r.t. the state, $\mathbb{E}[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\mid s]=b(s)\,\mathbb{E}[\nabla_\theta\log\pi_\theta(a\mid s)\mid s]=b(s)\cdot0=0$. Pull-out plus the fact that the score has conditional mean zero is the entire proof that baselines do not bias the gradient — and by §6.5 they strictly reduce variance when correlated with the return.

*Rao–Blackwellisation.* Replacing an unbiased estimator $T$ by $\mathbb{E}[T\mid\mathcal{G}]$ keeps it unbiased (tower) and reduces variance (law of total variance, §6.5). Every "analytically integrate out the discrete latent instead of sampling it" trick in variational inference is this.

### 6.4.1 Conditional Jensen and the conditional convergence theorems

**Theorem 6.9 (conditional Jensen).** Let $\varphi:\mathbb{R}\to\mathbb{R}$ be convex and suppose $X,\varphi(X)\in L^1$. Then a.s.
$$
\varphi\bigl(\mathbb{E}[X\mid\mathcal{G}]\bigr)\;\le\;\mathbb{E}\bigl[\varphi(X)\mid\mathcal{G}\bigr].
$$

*Proof.* A convex $\varphi$ is the supremum of its supporting lines, and one may take a *countable* family: there exist $(a_n,b_n)_{n\in\mathbb{N}}$ with $\varphi(x)=\sup_n(a_nx+b_n)$ for all $x$ (take rational tangency points and use continuity of convex functions on the interior of their domain). For each $n$, monotonicity and linearity give
$$
\mathbb{E}[\varphi(X)\mid\mathcal{G}]\ \ge\ \mathbb{E}[a_nX+b_n\mid\mathcal{G}]=a_n\mathbb{E}[X\mid\mathcal{G}]+b_n\quad\text{a.s.}
$$
Each holds off a null set $N_n$; the union $\bigcup_nN_n$ is null *because the family is countable* (Chapter 1's refrain: countability is load-bearing). Off that union, take the supremum over $n$ to get $\mathbb{E}[\varphi(X)|\mathcal{G}]\ge\sup_n(a_n\mathbb{E}[X|\mathcal{G}]+b_n)=\varphi(\mathbb{E}[X|\mathcal{G}])$. $\blacksquare$

**Corollary 6.10 ($L^p$ contraction).** For $p\ge1$ and $X\in L^p$, $\bigl\|\mathbb{E}[X\mid\mathcal{G}]\bigr\|_p\le\|X\|_p$. In particular conditioning never increases variance-scale magnitude, and $\mathbb{E}[\cdot\mid\mathcal{G}]$ is a contraction on $L^p$.

*Proof.* Apply Theorem 6.9 with $\varphi(x)=|x|^p$ and take expectations, using the tower property: $\mathbb{E}\bigl|\mathbb{E}[X|\mathcal{G}]\bigr|^p\le\mathbb{E}\bigl[\mathbb{E}[|X|^p|\mathcal{G}]\bigr]=\mathbb{E}|X|^p$. $\blacksquare$

**Proposition 6.11 (conditional MCT, Fatou, DCT).** Let $X_n$ be random variables.
(a) If $0\le X_n\uparrow X$ a.s. with $X\in L^1$, then $\mathbb{E}[X_n\mid\mathcal{G}]\uparrow\mathbb{E}[X\mid\mathcal{G}]$ a.s.
(b) If $X_n\ge0$, then $\mathbb{E}[\liminf_nX_n\mid\mathcal{G}]\le\liminf_n\mathbb{E}[X_n\mid\mathcal{G}]$ a.s.
(c) If $X_n\to X$ a.s. and $|X_n|\le W$ a.s. with $W\in L^1$, then $\mathbb{E}[X_n\mid\mathcal{G}]\to\mathbb{E}[X\mid\mathcal{G}]$ a.s. and in $L^1$.

*Proof sketch.* (a) By monotonicity the sequence $Y_n:=\mathbb{E}[X_n|\mathcal{G}]$ is a.s. non-decreasing; let $Y:=\lim_nY_n$ (which is $\mathcal{G}$-measurable, being a countable limit — Proposition 1.15). For $G\in\mathcal{G}$, MCT (Ch. 2) applied twice gives $\int_GY=\lim_n\int_GY_n=\lim_n\int_GX_n=\int_GX$, so $Y$ satisfies (CE1)–(CE3). (b) Apply (a) to $\inf_{m\ge n}X_m\uparrow\liminf X_n$ and use monotonicity. (c) Apply (b) to $W\pm X_n\ge0$ exactly as in the classical proof of dominated convergence. $\blacksquare$

Proposition 6.11(c) is the conditional form of the tool that closed Failure 3 of the Four Failures — swapping $\nabla$ and $\mathbb{E}$. In §6.9 we will need to swap a limit with a *conditional* expectation, and (c) is what licenses it.

**Worked Example 6.10 (conditional Jensen with numbers, and why it is not an equality).**
Coin flips again, $\varphi(x)=x^2$. On the atom $\{\text{first}=H\}$ the four outcomes give $X\in\{3,2,2,1\}$, so $\mathbb{E}[X^2\mid\mathcal{F}_1]=\tfrac14(9+4+4+1)=4.5$ while $\bigl(\mathbb{E}[X\mid\mathcal{F}_1]\bigr)^2=2^2=4.0$. On $\{\text{first}=T\}$: $X\in\{2,1,1,0\}$, giving $1.5$ versus $1.0$. The gap, $0.5$ on each atom, is precisely the conditional variance (§6.5). Jensen is an equality if and only if $X$ is a.s. $\mathcal{G}$-measurable (for strictly convex $\varphi$) — i.e. exactly when conditioning loses nothing.

> **Common Confusion 6.5.**
> *"The tower property says $\mathbb{E}[\mathbb{E}[X\mid\mathcal{G}]]=\mathbb{E}[X]$. So conditioning washes out and doesn't change anything?"*
>
> It changes everything except the *mean*. The tower property says the conditional expectation is an unbiased refinement: on average it is the same, but as a random variable it carries structure. Worked Example 6.2: $\mathbb{E}[X\mid\mathcal{F}_1]$ takes values $2$ and $1$; its mean is $1.5=\mathbb{E}[X]$; but its variance is $0.25$, and that variance is the amount of $X$'s uncertainty that the first flip explains. The entire content of §6.5 — and of the bias–variance decomposition, and of Rao–Blackwellisation, and of the claim that a good feature "explains variance" — lives in the gap between "same mean" and "different random variable". A conditional expectation is what you learn; the tower property is only the promise that learning it does not bias you.

> **Self-check 6.4.**
> 1. Prove $\mathbb{E}[X\mid\mathcal{G}]=X$ a.s. when $X$ is $\mathcal{G}$-measurable, using (d) with $Z=X$ and $X$ replaced by $1$.
> 2. Which hypothesis of (d) fails if $Z=\mathbb{E}[X\mid\mathcal{G}]$ and $X$ is Cauchy-distributed?
> 3. Show, from the tower property, that $\operatorname{Cov}\bigl(X-\mathbb{E}[X\mid\mathcal{G}],\;V\bigr)=0$ for every bounded $\mathcal{G}$-measurable $V$.
> 4. Give a one-line proof that $\mathbb{E}[X\mid\mathcal{G}]$ has smaller-or-equal second moment than $X$.

## 6.5 Conditional variance, and where bias–variance comes from

### 6.5.1 The law of total variance

**Definition 6.12 (conditional variance).** For $X\in L^2$ and $\mathcal{G}\subseteq\mathcal{F}$,
$$
\operatorname{Var}(X\mid\mathcal{G}):=\mathbb{E}\bigl[(X-\mathbb{E}[X\mid\mathcal{G}])^2\mid\mathcal{G}\bigr]=\mathbb{E}[X^2\mid\mathcal{G}]-\bigl(\mathbb{E}[X\mid\mathcal{G}]\bigr)^2\quad\text{a.s.}
$$
(The second equality is the usual expansion, valid by linearity and pull-out, since $\mathbb{E}[X\mid\mathcal{G}]$ is $\mathcal{G}$-measurable.) By conditional Jensen it is a.s. non-negative. Like $\mathbb{E}[X\mid\mathcal{G}]$, it is a *random variable*.

**Theorem 6.13 (law of total variance).** For $X\in L^2$ and any $\mathcal{G}\subseteq\mathcal{F}$,
$$
\operatorname{Var}(X)=\underbrace{\mathbb{E}\bigl[\operatorname{Var}(X\mid\mathcal{G})\bigr]}_{\text{unexplained: noise within atoms}}+\underbrace{\operatorname{Var}\bigl(\mathbb{E}[X\mid\mathcal{G}]\bigr)}_{\text{explained: spread across atoms}} .
$$

*Proof.* Write $M:=\mathbb{E}[X\mid\mathcal{G}]$ and $R:=X-M$. First, $\mathbb{E}[R\mid\mathcal{G}]=0$ a.s. by linearity and $\mathbb{E}[M\mid\mathcal{G}]=M$ (Worked Example 6.4(ii)), hence $\mathbb{E}[R]=0$ by the tower property and $\mathbb{E}[M]=\mathbb{E}[X]$. Next, $R$ and $M$ are uncorrelated:
$$
\mathbb{E}[RM]=\mathbb{E}\bigl[\mathbb{E}[RM\mid\mathcal{G}]\bigr]=\mathbb{E}\bigl[M\,\mathbb{E}[R\mid\mathcal{G}]\bigr]=0,
$$
using the tower property and then pull-out (Theorem 6.8(c),(d)) — this is the same orthogonality as Figure 6.2, now in probabilistic dress. Therefore
$$
\operatorname{Var}(X)=\operatorname{Var}(M+R)=\operatorname{Var}(M)+\operatorname{Var}(R)+2\operatorname{Cov}(M,R)=\operatorname{Var}(M)+\mathbb{E}[R^2].
$$
Finally $\mathbb{E}[R^2]=\mathbb{E}\bigl[\mathbb{E}[R^2\mid\mathcal{G}]\bigr]=\mathbb{E}[\operatorname{Var}(X\mid\mathcal{G})]$. $\blacksquare$

Both terms are non-negative, so **conditioning never increases the variance of your best guess**: $\operatorname{Var}(\mathbb{E}[X\mid\mathcal{G}])\le\operatorname{Var}(X)$, with equality iff $X$ is a.s. $\mathcal{G}$-measurable. That inequality is the Rao–Blackwell theorem in one line, and it is why "integrate out what you can, sample only what you must" is sound advice in variational inference, in policy-gradient estimators, and in Monte Carlo generally.

**Worked Example 6.11 (total variance on the three running examples).** All figures exact.

*Three coin flips, $X=\#$heads, $\mathcal{G}=\mathcal{F}_1=\sigma(\text{first flip})$.* $\operatorname{Var}(X)=3\cdot\tfrac14=0.75$. Conditional variance is $0.5$ on each atom (two remaining fair flips), so $\mathbb{E}[\operatorname{Var}(X\mid\mathcal{F}_1)]=0.5$. The conditional mean takes values $2,1$ with probability $\tfrac12$ each, so $\operatorname{Var}(\mathbb{E}[X\mid\mathcal{F}_1])=0.25$. Total $0.5+0.25=0.75$ ✓. Knowing the first flip explains one third of the variance — as it must, since it is one of three independent flips.

*Two dice, $Y=$ sum, $X=$ first die.* $\operatorname{Var}(Y)=2\cdot\tfrac{35}{12}=5.83333$. Given $X$, $Y=X+D_2$ so $\operatorname{Var}(Y\mid X)=\tfrac{35}{12}=2.91667$ (constant), and $\mathbb{E}[Y\mid X]=X+3.5$ has variance $\tfrac{35}{12}=2.91667$. Total $5.83333$ ✓ — an exact 50/50 split, as symmetry demands.

*ReLU of a Gaussian, $A=\max(0,Z)$, $\mathcal{G}=\sigma(\mathbf{1}\{Z>0\})$.* $\operatorname{Var}(A)=\tfrac12-\tfrac{1}{2\pi}=0.340845$. On $\{Z\le0\}$ the conditional variance is $0$; on $\{Z>0\}$ it is the half-normal variance $1-2/\pi=0.363380$. So $\mathbb{E}[\operatorname{Var}(A\mid\mathcal{G})]=\tfrac12(1-2/\pi)=0.181690$, and $\mathbb{E}[A\mid\mathcal{G}]\in\{0,\sqrt{2/\pi}\}$ equiprobably gives $\operatorname{Var}=\tfrac14\cdot\tfrac2\pi=0.159155$. Sum: $0.340845$ ✓. One bit of information — the sign — explains $46.7\%$ of the variance of a ReLU unit's output.

<svg viewBox="0 0 720 360" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Stacked bars showing the law of total variance for three examples">
  <g font-size="13" fill="currentColor">
    <text x="20" y="60">3 coin flips</text><text x="20" y="76" font-size="11">Var = 0.750</text>
    <text x="20" y="130">2 dice</text><text x="20" y="146" font-size="11">Var = 5.833</text>
    <text x="20" y="200">ReLU max(0,Z)</text><text x="20" y="216" font-size="11">Var = 0.341</text>
  </g>
  <g>
    <rect x="180" y="38" width="267" height="30" fill="#ef4444" opacity="0.75"/>
    <rect x="447" y="38" width="133" height="30" fill="#3b82f6" opacity="0.85"/>
    <text x="200" y="58" fill="currentColor" font-size="12">E[Var(X|G)] = 0.500</text>
    <text x="455" y="58" fill="currentColor" font-size="12">Var(E[X|G]) = 0.250</text>
  </g>
  <g>
    <rect x="180" y="108" width="200" height="30" fill="#ef4444" opacity="0.75"/>
    <rect x="380" y="108" width="200" height="30" fill="#3b82f6" opacity="0.85"/>
    <text x="200" y="128" fill="currentColor" font-size="12">2.917</text>
    <text x="400" y="128" fill="currentColor" font-size="12">2.917</text>
  </g>
  <g>
    <rect x="180" y="178" width="213" height="30" fill="#ef4444" opacity="0.75"/>
    <rect x="393" y="178" width="187" height="30" fill="#3b82f6" opacity="0.85"/>
    <text x="200" y="198" fill="currentColor" font-size="12">0.1817</text>
    <text x="410" y="198" fill="currentColor" font-size="12">0.1592</text>
  </g>
  <line x1="180" y1="30" x2="180" y2="220" stroke="currentColor" stroke-width="1"/>
  <line x1="580" y1="30" x2="580" y2="220" stroke="currentColor" stroke-width="1"/>
  <text x="180" y="248" fill="#ef4444" font-size="12">◼ unexplained (within atoms) = irreducible noise given G</text>
  <text x="180" y="266" fill="#3b82f6" font-size="12">◼ explained (across atoms) = what G tells you about X</text>
  <text x="20" y="300" fill="currentColor" font-size="13">Figure 6.3 — The law of total variance as a partition of each bar; every bar is rescaled to the same length so</text>
  <text x="20" y="317" fill="currentColor" font-size="13">the proportions are comparable. Red is E[Var(X|G)], the Bayes error you cannot remove by better prediction;</text>
  <text x="20" y="334" fill="currentColor" font-size="13">blue is Var(E[X|G]), the variance a feature explains. Adding features moves area from red to blue, never back.</text>
</svg>

### 6.5.2 Bias–variance is a conditioning identity

Now the ML corollary. The set-up: a training set $D=\bigl((X_1,Y_1),\dots,(X_n,Y_n)\bigr)$ drawn i.i.d. from the data law, a learning algorithm producing a predictor $\hat f_D$, and a fresh test point $(X,Y)$ independent of $D$ with the same law. Two independent sources of randomness — the training draw and the test draw — and the decomposition is nothing but conditioning on one and then the other.

**Corollary 6.14 (bias–variance decomposition).** Assume all quantities are square-integrable and $(X,Y)\perp D$. Write $f^\star(x)=\mathbb{E}[Y\mid X=x]$ and $\bar f(x):=\mathbb{E}_D[\hat f_D(x)]$, the average predictor over training sets. Then for $\mathbb{P}_X$-a.e. $x$,
$$
\mathbb{E}_{D,Y}\bigl[(Y-\hat f_D(x))^2\;\bigm|\;X=x\bigr]
=\underbrace{\operatorname{Var}(Y\mid X=x)}_{\text{noise}}
+\underbrace{\bigl(\bar f(x)-f^\star(x)\bigr)^2}_{\text{bias}^2}
+\underbrace{\operatorname{Var}_D\bigl(\hat f_D(x)\bigr)}_{\text{variance}},
$$
and integrating over $x\sim\mathbb{P}_X$ gives the same decomposition for the expected risk.

*Proof.* Condition on $D$ first. Given $D$ (hence given $\hat f_D$) and $X=x$, the predictor is a constant $c=\hat f_D(x)$, and by Theorem 6.5(4) applied to the conditional law of $Y$ given $X=x$,
$$
\mathbb{E}\bigl[(Y-c)^2\mid X=x,D\bigr]=\operatorname{Var}(Y\mid X=x)+\bigl(f^\star(x)-c\bigr)^2 .
$$
(Here we used $(X,Y)\perp D$ through Theorem 6.8(f): $\sigma(D)$ is independent of $\sigma(Y)\vee\sigma(X)$, so conditioning additionally on $D$ does not change the conditional expectation of $Y$ — or of $(Y-c)^2$ for fixed $c$ — given $X$. Note that (e) alone would *not* suffice here; this is precisely the enlargement situation the caution after Theorem 6.8 warned about, and (f)'s stronger hypothesis is what makes the step legal.) Take $\mathbb{E}_D$ of both sides — the tower property — and apply the elementary identity $\mathbb{E}[(W-a)^2]=\operatorname{Var}(W)+(\mathbb{E}W-a)^2$ with $W=\hat f_D(x)$, $a=f^\star(x)$, which is itself Theorem 6.13 with $\mathcal{G}$ trivial. $\blacksquare$

Three points of hygiene that papers routinely blur.

*The noise term is $\mathbb{E}[\operatorname{Var}(Y\mid X)]$ after averaging over $x$* — a property of the joint law only. It is the Bayes risk of Corollary 6.6. If your labels are deterministic functions of the features, it is zero, and a model can in principle reach zero test MSE.

*Bias and variance are defined relative to the randomness in $D$*, not to anything about the model class per se. "Variance" here is variance over resampling the training set — which is why bagging (averaging over bootstrap resamples) attacks precisely this term and leaves bias roughly alone.

*The decomposition is exact for squared loss and essentially only for squared loss.* Analogues for 0–1 loss and cross-entropy exist but are not clean orthogonal splits; treating "bias–variance" as a universal law of learning is a category error. What *is* universal is the underlying identity: orthogonal projection plus the tower property.

**Worked Example 6.12 (bias–variance, measured).**
Truth: $f(x)=\sin(2\pi x)$ on $[0,1]$ with $Y=f(x)+\varepsilon$, $\varepsilon\sim N(0,0.30^2)$, so noise $=0.09$. Training sets: $n=20$ points on a fixed grid $x_i\in[0.05,0.95]$, fresh labels each replication; $R=4000$ replications; seed 2024 (`default_rng(2024)`). Fit ordinary least-squares polynomials of degree $d$; evaluate on a grid of 101 points and average the three terms over that grid.

| degree $d$ | bias$^2$ | variance | noise | total |
|---|---|---|---|---|
| 1 | 0.1384 | 0.0088 | 0.0900 | 0.2372 |
| 3 | 0.0022 | 0.0166 | 0.0900 | 0.1087 |
| 9 | <0.0001 | 0.0428 | 0.0900 | 0.1328 |
| 15 | 0.0001 | 0.8592 | 0.0900 | 0.9493 |

Read the columns. Bias collapses between $d=1$ and $d=3$ (a cubic already approximates one period of a sine well) and is negligible thereafter. Variance grows monotonically and then explodes as $d$ approaches $n-1=19$, where the design matrix becomes ill-conditioned. Total error is U-shaped with a minimum at $d=3$: the textbook picture, reproduced honestly, with the noise floor of $0.09$ visible as the number the other columns cannot go below.

> **Common Confusion 6.6.**
> *"So more parameters always means more variance, and the test error curve is always U-shaped. Doesn't that contradict the fact that a 100-million-parameter network interpolating the training data generalises fine?"*
>
> The decomposition in Corollary 6.14 is an identity — it cannot be contradicted. What can be, and has been, contradicted is the *folk claim* that variance increases monotonically with parameter count. In the classical regime (least squares with $d<n$, no regularisation) it does, and Worked Example 6.12 shows it. Past the interpolation threshold $d\approx n$, the picture changes: with minimum-norm interpolation, variance can *decrease* again as $d$ grows, producing the double-descent curve. The reason is that the estimator changes character — among the infinitely many interpolants, the algorithm's implicit bias picks a smooth one, which is a form of regularisation that strengthens with over-parameterisation. So: the identity is safe; the story attached to it is not. Chapter 8 treats this properly. Meanwhile, do not use "bias–variance" as an argument that a big model must overfit; use it as an accounting scheme and measure the terms, as we just did.

> **Self-check 6.5.**
> 1. Prove $\operatorname{Var}(\mathbb{E}[X\mid\mathcal{G}])\le\operatorname{Var}(X)$ in one line and state when equality holds.
> 2. In Worked Example 6.11, what fraction of $\operatorname{Var}(X)$ does $\mathcal{F}_2$ (two flips) explain? Guess first, then verify.
> 3. Which term of Corollary 6.14 does bagging target? Which does early stopping target?
> 4. If $\operatorname{Var}(Y\mid X=x)=0$ for $\mathbb{P}_X$-a.e. $x$, what does that say about $Y$?

## 6.6 Conditional distributions

### 6.6.1 What is still missing

We can now condition any single integrable random variable. But a phrase like "the conditional distribution of $Y$ given $X=x$" asks for more: a whole probability measure for each $x$, and papers use it constantly — $p(y\mid x)$ in a likelihood, $q(z\mid x)$ in a VAE encoder, $\pi(a\mid s)$ in a policy, the transition kernel of a Markov chain.

The naive route is to define $\mu_x(B):=\mathbb{P}(Y\in B\mid X=x)$ using Definition 6.3 for each fixed $B$. The trouble is the order of quantifiers. For fixed $B$ we get a function of $x$ defined up to a $\mathbb{P}_X$-null set $N_B$. Countable additivity in $B$ then holds only outside $\bigcup_k N_{B_k}$ for each countable family — and there are uncountably many countable families of Borel sets. Nothing so far guarantees a *single* null set outside which $B\mapsto\mu_x(B)$ is a genuine measure. This is Chapter 1's countability warning striking again, and this time it is not a nuisance we can dodge: without a fix, "sample from $p(y\mid x)$" would be meaningless.

**Definition 6.15 (probability kernel; regular conditional distribution).**
Let $(\mathcal{X},\mathcal{A})$ and $(\mathcal{Y},\mathcal{B})$ be measurable spaces. A *probability kernel* from $\mathcal{X}$ to $\mathcal{Y}$ is a map $\kappa:\mathcal{X}\times\mathcal{B}\to[0,1]$ such that
(i) for each fixed $x$, $B\mapsto\kappa(x,B)$ is a probability measure on $(\mathcal{Y},\mathcal{B})$; and
(ii) for each fixed $B$, $x\mapsto\kappa(x,B)$ is $\mathcal{A}$-measurable.
Given random elements $X\in\mathcal{X}$, $Y\in\mathcal{Y}$ on a common probability space, a kernel $\kappa$ is a *regular conditional distribution (r.c.d.) of $Y$ given $X$* if
$$
\kappa(X,B)=\mathbb{P}(Y\in B\mid X)\quad\text{a.s., for every }B\in\mathcal{B}.
$$
We then write $\mathbb{P}(Y\in B\mid X=x)=\kappa(x,B)$ and $\mu_x:=\kappa(x,\cdot)$.

Condition (i) is the "single null set" demand; (ii) is what makes $\int f(y)\kappa(\cdot,dy)$ a measurable function of $x$, so that expectations can be iterated.

**Theorem 6.16 (existence of regular conditional distributions; cited, not proved).**
If $\mathcal{Y}$ is a Polish space (a complete separable metric space) with its Borel $\sigma$-algebra — e.g. $\mathbb{R}^d$, any Borel subset thereof, $\mathbb{R}^\infty$, or a space of continuous functions — then for any random elements $X,Y$ as above an r.c.d. of $Y$ given $X$ exists, and any two agree for $\mathbb{P}_X$-a.e. $x$ (as measures).

A sketch of the reason, so it is not a black box: on $\mathcal{Y}=\mathbb{R}$ one builds candidate conditional CDFs $F(q\mid x):=\mathbb{P}(Y\le q\mid X=x)$ for *rational* $q$ only — a countable family, so a single null set suffices — repairs monotonicity and right-continuity on that null set, and extends to all reals by right-continuous interpolation, invoking Proposition 1.19's converse (every such $F$ is a CDF) and Theorem 1.20 (a CDF determines a law). General Polish spaces reduce to $\mathbb{R}$ by a Borel isomorphism. The countable dense set is doing the work, exactly as in Chapter 1, where measurability only ever had to be checked on a generating family.

The practical upshot: **on the spaces ML actually uses, conditional distributions exist and can be manipulated as ordinary distributions.** Pathologies live on exotic spaces you will not meet. But the a.e. caveat never goes away: $\mu_x$ is pinned down only for $\mathbb{P}_X$-almost every $x$.

Two consequences we will use freely:

*Disintegration.* For bounded measurable $h$, $\ \mathbb{E}[h(X,Y)]=\int_{\mathcal{X}}\Bigl(\int_{\mathcal{Y}}h(x,y)\,\mu_x(dy)\Bigr)\mathbb{P}_X(dx)$, and $\mathbb{E}[h(X,Y)\mid X]=\int h(X,y)\mu_X(dy)$ a.s. (Prove it by the ladder: indicators of rectangles, then $\pi$–$\lambda$ via Theorem 1.20, then simple functions, then MCT.) This is a conditional Fubini, and it is the formal content of "first draw $x$, then draw $y$ from $p(\cdot\mid x)$" — the generative-model narrative.

*Conditional expectations are integrals against $\mu_x$.* In particular $f^\star(x)=\int y\,\mu_x(dy)$ whenever $Y\in L^1$, which is the formula everyone writes down.

### 6.6.2 Densities: recovering $p(y\mid x)=p(x,y)/p(x)$

**Proposition 6.17 (conditional density).**
Suppose $(X,Y)$ takes values in $\mathbb{R}^d\times\mathbb{R}^m$ and has a joint density $p(x,y)$ with respect to $\lambda_d\otimes\lambda_m$. Let $p_X(x):=\int p(x,y)\,dy$, the marginal density (measurable by Tonelli, Ch. 2). Define
$$
p(y\mid x):=\begin{cases}\dfrac{p(x,y)}{p_X(x)},& p_X(x)\in(0,\infty),\\[6pt] q_0(y),&\text{otherwise,}\end{cases}
$$
where $q_0$ is an arbitrary fixed probability density. Then $\kappa(x,B):=\int_Bp(y\mid x)\,dy$ is an r.c.d. of $Y$ given $X$. Consequently $\mathbb{E}[Y\mid X=x]=\int y\,p(y\mid x)\,dy$ for $\mathbb{P}_X$-a.e. $x$ with $\mathbb{E}|Y|<\infty$.

*Proof.* First, $\{p_X=0\}$ has $\mathbb{P}_X$-probability $\int_{\{p_X=0\}}p_X\,dx=0$, and $\{p_X=\infty\}$ is Lebesgue-null since $p_X\in L^1(\lambda_d)$; so the arbitrary choice on the exceptional set is invisible. Property (i) of Definition 6.15: for $p_X(x)\in(0,\infty)$, $\kappa(x,\cdot)$ is a measure by countable additivity of the integral and $\kappa(x,\mathbb{R}^m)=p_X(x)/p_X(x)=1$. Property (ii): $x\mapsto\int_Bp(x,y)dy$ is measurable by Tonelli, and quotients of measurable functions are measurable (Proposition 1.15).

For the defining property, we must show $\kappa(X,B)$ satisfies (CE1)–(CE3) for $\mathbb{E}[\mathbf{1}\{Y\in B\}\mid X]$. (CE1) holds since $\kappa(\cdot,B)$ is measurable and we compose with $X$; (CE2) since $0\le\kappa\le1$. For (CE3), a generic $G\in\sigma(X)$ is $\{X\in A\}$ with $A$ Borel (Definition 1.21), and by Tonelli,
$$
\int_{\{X\in A\}}\kappa(X,B)\,d\mathbb{P}=\int_A\kappa(x,B)\,p_X(x)\,dx=\int_A\!\int_B\frac{p(x,y)}{p_X(x)}p_X(x)\,dy\,dx=\int_A\!\int_Bp(x,y)\,dy\,dx,
$$
which is $\mathbb{P}(X\in A,\,Y\in B)=\int_{\{X\in A\}}\mathbf{1}\{Y\in B\}\,d\mathbb{P}$. The final claim follows from disintegration applied to $h(x,y)=y$. $\blacksquare$

So the formula everyone divides by is a theorem, valid under a stated hypothesis (a joint density exists), with an explicit statement of what happens where the denominator vanishes (nothing, a.s.). Note also what the proposition does *not* require: no continuity, no positivity, no limits.

**Worked Example 6.13 (the triangle, revisited and completed).**
Return to Worked Example 6.1: $(X,Y)$ uniform on $T=\{0<y<x<1\}$, $p(x,y)=2\mathbf{1}_T$. Then $p_X(x)=\int_0^x2\,dy=2x$ on $(0,1)$, so
$$
p(y\mid x)=\frac{2}{2x}\mathbf{1}\{0<y<x\}=\frac1x\mathbf{1}_{(0,x)}(y):\qquad Y\mid X=x\ \sim\ \mathrm{Uniform}(0,x).
$$
Hence $f^\star(x)=\mathbb{E}[Y\mid X=x]=x/2$ and $\operatorname{Var}(Y\mid X=x)=x^2/12$, and $\mathbb{P}(Y\le0.1\mid X=0.3)=1/3$ as the limiting computation found. Monte Carlo check ($n=4\times10^6$ pairs, seed 12, conditioning band $|X-x_0|<0.005$):

| $x_0$ | MC $\mathbb{E}[Y\mid X\approx x_0]$ | theory $x_0/2$ | MC var | theory $x_0^2/12$ | band count |
|---|---|---|---|---|---|
| 0.3 | 0.1500 | 0.1500 | 0.00748 | 0.00750 | 23 795 |
| 0.7 | 0.3501 | 0.3500 | 0.04082 | 0.04083 | 56 295 |

Also worth noting: the law of total variance here reads $\operatorname{Var}(Y)=\mathbb{E}[X^2/12]+\operatorname{Var}(X/2)$. With $\mathbb{E}[X^2]=\int_0^1x^2\cdot2x\,dx=1/2$ and $\operatorname{Var}(X)=1/2-(2/3)^2=1/18$, this is $1/24+1/72=1/18=\operatorname{Var}(Y)$ ✓ (by symmetry $Y=\min$ has the same variance as $X=\max$).

**Worked Example 6.14 (Gaussian conditioning as an r.c.d. — reconciling with Chapter 5).**
Let $(X,Y)$ be jointly Gaussian with $\mathbb{E}X=\mathbb{E}Y=0$, $\sigma_X=1$, $\sigma_Y=2$, $\rho=0.6$. Chapter 5's conditioning formula gives $Y\mid X=x\sim N\bigl(\rho\frac{\sigma_Y}{\sigma_X}x,\ \sigma_Y^2(1-\rho^2)\bigr)=N(1.2x,\ 2.56)$. Let us see it as an r.c.d. and verify it three ways.

*Algebraically.* Write $Y=\rho\frac{\sigma_Y}{\sigma_X}X+W$ with $W:=Y-1.2X$. Then $\operatorname{Cov}(X,W)=\rho\sigma_X\sigma_Y-1.2\sigma_X^2=1.2-1.2=0$, and since $(X,W)$ is a linear image of a Gaussian vector it is jointly Gaussian, so zero covariance gives *independence* (a Gaussian-only luxury, Ch. 5). Therefore $\kappa(x,\cdot):=N(1.2x,\operatorname{Var}(W))$ is a kernel — measurable in $x$, a probability measure for each $x$ — and for Borel $B$,
$$
\mathbb{P}(Y\in B\mid X)=\mathbb{P}(1.2X+W\in B\mid X)=\kappa(X,B)\ \text{a.s.},
$$
by the freezing lemma after Theorem 6.8: with $h(x,w)=\mathbf{1}_B(1.2x+w)$ and $W\perp X$, $\ \mathbb{E}[h(X,W)\mid X]=\varphi(X)$ where $\varphi(x)=\mathbb{P}(1.2x+W\in B)=\kappa(x,B)$. $\operatorname{Var}(W)=\sigma_Y^2(1-\rho^2)=4(0.64)=2.56$.

*Via Proposition 6.17.* Dividing the bivariate normal density by the $N(0,1)$ marginal and completing the square gives the same $N(1.2x,2.56)$; the algebra is Chapter 5's.

*Numerically* ($n=4\times10^6$, seed 11, band $|X-x_0|<0.01$):

| $x_0$ | MC mean | theory $1.2x_0$ | MC var | theory $2.56$ | band count |
|---|---|---|---|---|---|
| $-1.0$ | $-1.2091$ | $-1.2000$ | 2.5580 | 2.56 | 19 362 |
| $0.5$ | $0.5965$ | $0.6000$ | 2.5794 | 2.56 | 28 223 |
| $2.0$ | $2.3702$ | $2.4000$ | 2.6082 | 2.56 | 4 304 |

The $x_0=2$ row is off by $0.03$, and the reason is instructive: the band has positive width and the density of $X$ decreases steeply there, so the band's occupants are skewed toward smaller $x$, dragging the conditional mean down. Finite bands estimate a weighted average of $f^\star$ over the band, not $f^\star(x_0)$ — the same bias that afflicts kernel regression with too wide a bandwidth, appearing here in miniature. It is also a reminder of why the limiting definition of §6.1 is delicate: the answer you get depends on how you shrink, and only in the limit (and only under regularity) does the dependence vanish.

Since $f^\star(x)=1.2x$ is affine here, the best predictor coincides with the best linear predictor — the coincidence flagged in Common Confusion 6.4 that makes Gaussian intuition treacherous. Compare Worked Example 6.7, where the same $\rho=0$ would have suggested "no relationship".

### 6.6.3 Conditional independence, formalised

Definition 1.24 defined conditional independence for events. With kernels we can now state it for random variables in the form used in graphical models: $X\perp Y\mid Z$ iff for all bounded measurable $f,g$,
$$
\mathbb{E}[f(X)g(Y)\mid Z]=\mathbb{E}[f(X)\mid Z]\,\mathbb{E}[g(Y)\mid Z]\quad\text{a.s.},
$$
equivalently iff $\mathbb{E}[f(X)\mid Y,Z]=\mathbb{E}[f(X)\mid Z]$ a.s. for all bounded $f$ — "once you know $Z$, $Y$ tells you nothing more about $X$". The second form is the one to remember: conditional independence is a statement that a conditional expectation *does not depend on the extra variable*. Every d-separation argument, every Markov-property manipulation, and the entire factorisation $p(x_{1:n})=\prod_ip(x_i\mid \mathrm{pa}(i))$ is this identity applied repeatedly.

### 6.6.4 The Borel paradox, concretely

> **Common Confusion 6.7.**
> *"Fine, but once we have $p(y\mid x)$, conditioning on a measure-zero event is safe. Why does anyone still call it a paradox?"*
>
> Because the conditional law depends on **which random variable you conditioned on**, not merely on which event occurred. The same null event, described by two different random variables, has two different — and both correct — conditional laws attached to it. The computation below makes this exact: describing $\{X=Y\}$ as $\{X-Y=0\}$ gives $\mathbb{E}[Y^2\mid\cdot]=0.5$; describing it as $\{X/Y=1\}$ gives $1$. What is illegitimate is not conditioning on a null event; it is the phrase "given that $X=Y$", full stop, with no conditioning variable named.

Here is the computation, which you can check yourself. Let $X,Y$ be i.i.d. $N(0,1)$ and consider the event $\{X=Y\}$, described two ways.

*Way 1:* $\{U=0\}$ where $U:=X-Y$. Since $(U,Y)$ is jointly Gaussian with $\operatorname{Cov}(U,Y)=-1$, $\operatorname{Var}(U)=2$, Chapter 5's conditioning formula gives $Y\mid U=0\sim N(0,\tfrac12)$, so $\mathbb{E}[Y^2\mid U=0]=0.5$.

*Way 2:* $\{V=1\}$ where $V:=X/Y$. Change variables $(x,y)\mapsto(v,y)$ with $x=vy$ and Jacobian $|y|$ (change-of-variables formula, Ch. 3): the joint density of $(V,Y)$ is $\frac{1}{2\pi}e^{-(v^2y^2+y^2)/2}|y|$. At $v=1$ this is proportional to $|y|e^{-y^2}$, which normalises exactly, so $p(y\mid V=1)=|y|e^{-y^2}$ and $\mathbb{E}[Y^2\mid V=1]=\int y^2|y|e^{-y^2}dy=1$.

**Same event, two answers: $0.5$ and $1$.** Monte Carlo confirms both ($n=4\times10^7$, seed 41, bands of half-width $0.01$): conditioning on $|X-Y|<0.01$ gives $\mathbb{E}[Y^2]=0.4966$ over 225 397 hits; conditioning on $|X/Y-1|<0.01$ gives $0.9954$ over 126 927 hits. Nothing is broken. Definition 6.15 conditions on a *random variable* — on the whole $\sigma$-algebra $\sigma(U)$ or $\sigma(V)$, which are genuinely different — and each answer is correct for its own $\sigma$-algebra. The bands make the mechanism visible: $\{|X-Y|<\epsilon\}$ is a strip of constant width, while $\{|X/Y-1|<\epsilon\}$ is a wedge that widens as $|y|$ grows, so it over-samples large $|y|$. Figure 6.4 shows why the two answers differ, and it is worth looking at before reading on: the disagreement is entirely about the *shape* of the approximating region.

Where this bites in ML: any time you condition on a constraint surface — normalising an embedding to the unit sphere and then "conditioning on the direction", constraining a latent to a manifold, or imposing $g(\theta)=0$ in a Bayesian model — the resulting conditional law depends on the parametrisation of the constraint. The fix is always the same: say which random variable you are conditioning on, and if a limit is intended, say which limiting scheme.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two approximations to the event X equals Y: a constant-width strip and a wedge that widens with the magnitude of y">
  <g stroke="currentColor" stroke-width="1.2">
    <line x1="60" y1="290" x2="320" y2="290"/><line x1="190" y1="60" x2="190" y2="290"/>
    <line x1="400" y1="290" x2="660" y2="290"/><line x1="530" y1="60" x2="530" y2="290"/>
  </g>
  <g fill="currentColor" font-size="11">
    <text x="322" y="294">x</text><text x="182" y="54">y</text>
    <text x="662" y="294">x</text><text x="522" y="54">y</text>
  </g>
  <polygon points="72,290 84,290 308,66 296,66" fill="#3b82f6" opacity="0.28"/>
  <line x1="66" y1="290" x2="314" y2="66" stroke="#3b82f6" stroke-width="1.6"/>
  <polygon points="416,290 440,290 644,66 620,66" fill="#10b981" opacity="0.28"/>
  <polygon points="416,290 440,290 644,66 620,66" fill="none" stroke="none"/>
  <polygon points="530,178 644,66 620,66 530,155" fill="#10b981" opacity="0.28"/>
  <polygon points="530,178 440,290 416,290 530,201" fill="#10b981" opacity="0.28"/>
  <line x1="406" y1="290" x2="654" y2="66" stroke="#10b981" stroke-width="1.6"/>
  <g font-size="12" text-anchor="middle">
    <text x="190" y="316" fill="#3b82f6">condition on U = X − Y</text>
    <text x="190" y="333" fill="#3b82f6">band { |X − Y| &lt; ε } : constant width</text>
    <text x="530" y="316" fill="#10b981">condition on V = X / Y</text>
    <text x="530" y="333" fill="#10b981">band { |X/Y − 1| &lt; ε } : width ∝ |y|</text>
  </g>
  <text x="200" y="100" fill="currentColor" font-size="11">E[Y² | ·] = 0.5</text>
  <text x="540" y="100" fill="currentColor" font-size="11">E[Y² | ·] = 1</text>
  <text x="60" y="360" fill="currentColor" font-size="13">Figure 6.4 — The Borel paradox. Both shaded regions shrink to the same line {x = y} as ε → 0, but the wedge on</text>
  <text x="60" y="377" fill="currentColor" font-size="13">the right pinches at the origin and flares far from it, so it keeps proportionally more of the large-|y| outcomes.</text>
  <text x="60" y="394" fill="currentColor" font-size="13">Different shapes, different limits: 0.5 versus 1. The event is the same; the conditioning σ-algebra is not.</text>
</svg>

> **Self-check 6.6.**
> 1. Why can we not simply define $\mu_x(B)=\mathbb{P}(Y\in B\mid X=x)$ for each $B$ separately and be done?
> 2. In Proposition 6.17, why does the arbitrary choice of $q_0$ not matter?
> 3. For the triangle example, compute $\mathbb{E}[X\mid Y=y]$ (note: the *other* direction) and check it against $\operatorname{Var}$ bookkeeping.
> 4. State the Borel paradox in one sentence, in a way that a colleague who has not read this section would not mishear as "conditioning on null events is illegal".

## 6.7 Martingales

### 6.7.1 Information over time

Chapter 1 §1.3.4 introduced filtrations informally: a $\sigma$-algebra is information, so a growing family of $\sigma$-algebras is information accumulating. Now we can do something with them.

**Definition 6.18 (filtration; adapted; predictable).**
A *filtration* on $(\Omega,\mathcal{F},\mathbb{P})$ is an increasing family $\mathbb{F}=(\mathcal{F}_n)_{n\ge0}$ of sub-$\sigma$-algebras, $\mathcal{F}_0\subseteq\mathcal{F}_1\subseteq\cdots\subseteq\mathcal{F}$. A process $(Z_n)_{n\ge0}$ is *adapted* if $Z_n$ is $\mathcal{F}_n$-measurable for every $n$ ("$Z_n$ is known at time $n$"), and *predictable* if $Z_n$ is $\mathcal{F}_{n-1}$-measurable for every $n\ge1$ ("$Z_n$ is decided *before* time $n$'s randomness is revealed"). The *natural filtration* of a process $(X_n)$ is $\mathcal{F}_n=\sigma(X_0,\dots,X_n)$.

The distinction between adapted and predictable is the difference between a bet you place after seeing the card and one you place before. In SGD, the iterate $\theta_t$ is $\mathcal{F}_t$-measurable and the step size $\eta_t$ is predictable (chosen from past information); the stochastic gradient $g_t$ is $\mathcal{F}_{t+1}$-measurable but not $\mathcal{F}_t$-measurable — it is exactly the new information at step $t$. Keeping this straight is 80% of §6.9.

**Definition 6.19 (martingale, submartingale, supermartingale).**
An adapted process $(M_n)$ with $\mathbb{E}|M_n|<\infty$ for all $n$ is a *martingale* (w.r.t. $\mathbb{F}$ and $\mathbb{P}$) if
$$
\mathbb{E}[M_{n+1}\mid\mathcal{F}_n]=M_n\quad\text{a.s. for all }n\ge0 .
$$
It is a *submartingale* if $\ge$ holds and a *supermartingale* if $\le$ holds.

In words: *given everything you know now, the best forecast of tomorrow is today's value.* A submartingale drifts up (a favourable game), a supermartingale drifts down. The unfortunate naming — "super" means it decreases — is universal; the mnemonic is that a supermartingale is *above* where it is going.

Iterating the tower property gives $\mathbb{E}[M_m\mid\mathcal{F}_n]=M_n$ for all $m\ge n$, and in particular $\mathbb{E}[M_n]=\mathbb{E}[M_0]$ for a martingale, $\mathbb{E}[M_n]\ge\mathbb{E}[M_0]$ for a submartingale.

**Proposition 6.20 (three constructions).**
(a) *(Random walk.)* If $\xi_1,\xi_2,\dots$ are independent with $\mathbb{E}|\xi_i|<\infty$ and $\mathbb{E}\xi_i=0$, and $\mathcal{F}_n=\sigma(\xi_1,\dots,\xi_n)$, then $M_n=\sum_{i\le n}\xi_i$ is a martingale. If instead $\mathbb{E}\xi_i\le0$, it is a supermartingale.
(b) *(Doob martingale.)* For $X\in L^1$ and any filtration, $M_n:=\mathbb{E}[X\mid\mathcal{F}_n]$ is a martingale.
(c) *(Convex functions.)* If $(M_n)$ is a martingale and $\varphi$ is convex with $\mathbb{E}|\varphi(M_n)|<\infty$, then $(\varphi(M_n))$ is a submartingale; in particular $(|M_n|)$ and $(M_n^2)$ are (the latter when $M_n\in L^2$).

*Proof.* (a) $\mathbb{E}[M_{n+1}\mid\mathcal{F}_n]=M_n+\mathbb{E}[\xi_{n+1}\mid\mathcal{F}_n]=M_n+\mathbb{E}[\xi_{n+1}]=M_n$, using $M_n$ $\mathcal{F}_n$-measurable and Theorem 6.8(e). (b) $\mathbb{E}[M_{n+1}\mid\mathcal{F}_n]=\mathbb{E}\bigl[\mathbb{E}[X\mid\mathcal{F}_{n+1}]\mid\mathcal{F}_n\bigr]=\mathbb{E}[X\mid\mathcal{F}_n]=M_n$ by the tower property, since $\mathcal{F}_n\subseteq\mathcal{F}_{n+1}$. Integrability from Corollary 6.10. (c) Conditional Jensen: $\mathbb{E}[\varphi(M_{n+1})\mid\mathcal{F}_n]\ge\varphi(\mathbb{E}[M_{n+1}\mid\mathcal{F}_n])=\varphi(M_n)$. $\blacksquare$

Part (b) deserves emphasis: **any** integrable random variable, revealed gradually through any filtration, generates a martingale. Worked Example 6.2 was one ($X=\#$heads revealed flip by flip). Section 6.8 turns this observation into a concentration inequality for arbitrary functions of independent data, which is where McDiarmid comes from.

<svg viewBox="0 0 720 420" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A binary lattice of a symmetric random walk showing that the conditional expectation of the future equals the present value">
  <g stroke="currentColor" stroke-width="1" opacity="0.45" fill="none">
    <line x1="100" y1="200" x2="250" y2="160"/><line x1="100" y1="200" x2="250" y2="240"/>
    <line x1="250" y1="160" x2="400" y2="120"/><line x1="250" y1="160" x2="400" y2="200"/>
    <line x1="250" y1="240" x2="400" y2="200"/><line x1="250" y1="240" x2="400" y2="280"/>
    <line x1="400" y1="120" x2="550" y2="80"/><line x1="400" y1="120" x2="550" y2="160"/>
    <line x1="400" y1="200" x2="550" y2="160"/><line x1="400" y1="200" x2="550" y2="240"/>
    <line x1="400" y1="280" x2="550" y2="240"/><line x1="400" y1="280" x2="550" y2="320"/>
  </g>
  <polyline points="100,200 250,160 400,200 550,160" fill="none" stroke="#3b82f6" stroke-width="3"/>
  <g fill="currentColor">
    <circle cx="100" cy="200" r="5"/><circle cx="250" cy="160" r="5"/><circle cx="250" cy="240" r="5"/>
    <circle cx="400" cy="120" r="5"/><circle cx="400" cy="200" r="5"/><circle cx="400" cy="280" r="5"/>
    <circle cx="550" cy="80" r="5"/><circle cx="550" cy="160" r="5"/><circle cx="550" cy="240" r="5"/><circle cx="550" cy="320" r="5"/>
  </g>
  <g fill="currentColor" font-size="12">
    <text x="82" y="192">0</text><text x="258" y="152">+1</text><text x="258" y="256">−1</text>
    <text x="408" y="114">+2</text><text x="408" y="194">0</text><text x="408" y="296">−2</text>
    <text x="562" y="84">+3</text><text x="562" y="164">+1</text><text x="562" y="244">−1</text><text x="562" y="324">−3</text>
  </g>
  <g fill="currentColor" font-size="12" text-anchor="middle">
    <text x="100" y="360">F₀ (nothing)</text><text x="250" y="360">F₁</text><text x="400" y="360">F₂</text><text x="550" y="360">F₃</text>
  </g>
  <text x="270" y="115" fill="#10b981" font-size="12">E[M₃ | F₁] = ½(+3) + ... = +1 = M₁</text>
  <text x="20" y="392" fill="currentColor" font-size="13">Figure 6.5 — A symmetric random walk on its natural filtration. From any node, the average of the reachable</text>
  <text x="20" y="409" fill="currentColor" font-size="13">values at any later time equals the node's own value: that is the martingale property. The blue path is one</text>
  <text x="20" y="426" fill="currentColor" font-size="13">realisation; the martingale property constrains averages over branches, not the path itself, which wanders freely.</text>
</svg>

**Worked Example 6.15 (three martingales you will meet).**
(i) *Symmetric random walk.* $\xi_i=\pm1$ fair; $M_n=\sum\xi_i$. Fair game.
(ii) *Product of mean-one variables.* $\xi_i>0$ independent with $\mathbb{E}\xi_i=1$; $M_n=\prod_{i\le n}\xi_i$. Then $\mathbb{E}[M_{n+1}\mid\mathcal{F}_n]=M_n\mathbb{E}[\xi_{n+1}]=M_n$ by pull-out and independence. Multiplicative wealth under fair odds; also the "exponential martingale" $M_n=\exp(\lambda S_n-n\psi(\lambda))$ with $\psi$ the cumulant generating function, which is the engine of Chapter 4's Chernoff method and of Azuma below.
(iii) *Doob martingale of a statistic.* $X=f(\xi_1,\dots,\xi_N)$ for independent $\xi_i$; $M_n=\mathbb{E}[X\mid\xi_1,\dots,\xi_n]$ interpolates from $M_0=\mathbb{E}[X]$ to $M_N=X$. For $f=$ test error of a model trained on the sample, this is exactly "how the expected performance updates as data arrives".

**Worked Example 6.16 (the likelihood ratio is a martingale, and it goes to zero).**
Let $X_1,X_2,\dots$ be i.i.d. with density $p$ (the truth), and let $q$ be a competing density with $q\ll p$. Define
$$
L_n:=\prod_{i=1}^n\frac{q(X_i)}{p(X_i)} .
$$
Since $\mathbb{E}_p[q(X_i)/p(X_i)]=\int\frac{q}{p}p\,dx=\int q\,dx=1$, Worked Example 6.15(ii) applies: $(L_n)$ is a martingale under $\mathbb{P}=p^{\otimes\infty}$ with $\mathbb{E}[L_n]=1$ for all $n$. This single fact is the basis of *anytime-valid* inference: by Markov's inequality $\mathbb{P}(\sup_nL_n\ge1/\alpha)\le\alpha$ (using Theorem 6.24 below), so "reject when $L_n\ge1/\alpha$" controls type-I error *no matter when you look* — the modern e-value/e-process machinery, and the principled fix for peeking at A/B tests.

Yet $L_n\to0$ a.s. whenever $q\neq p$: $\frac1n\log L_n\to-D_{\mathrm{KL}}(P\|Q)<0$ by the law of large numbers (Ch. 7). A martingale with constant mean $1$ that converges a.s. to $0$ — the mass escapes to rarer and rarer events with larger and larger values. Simulation with $p=N(0,1)$, $q=N(0.5,1)$, $R=200\,000$ replicates, seed 15:

| $n$ | sample mean of $L_n$ | median | $\mathbb{P}(L_n>10)$ | max observed |
|---|---|---|---|---|
| 1 | 0.9991 | 0.8822 | 0.00000 | 7.49 |
| 10 | 0.9913 | 0.2869 | 0.01206 | $2.81\times10^2$ |
| 50 | 0.8909 | $1.916\times10^{-3}$ | 0.00788 | $1.36\times10^4$ |

The theoretical mean is exactly $1$ at every $n$; by $n=50$ the *sample* mean over 200 000 draws has drifted to $0.89$, because the mean is carried by events of probability $\sim10^{-6}$ that the sample has not yet seen enough of. Here $D_{\mathrm{KL}}(P\|Q)=0.125$ per observation, so the median decays like $e^{-0.125n}$: at $n=50$, $e^{-6.25}=1.9\times10^{-3}$, matching the observed median to two digits. Keep this example in mind whenever you estimate an expectation of a likelihood ratio, an importance weight, or a PPO ratio by Monte Carlo: **the sample mean of a heavy-tailed positive martingale is systematically below its true mean until the sample is enormous.**

### 6.7.2 Stopping times and optional stopping

**Definition 6.21 (stopping time).** A random variable $\tau:\Omega\to\{0,1,2,\dots\}\cup\{\infty\}$ is a *stopping time* for $\mathbb{F}$ if $\{\tau\le n\}\in\mathcal{F}_n$ for every $n$ (equivalently $\{\tau=n\}\in\mathcal{F}_n$).

The condition says: *you can tell whether you have stopped using only information available at the time.* "The first $n$ with $M_n\ge10$" is a stopping time. "The last time before $N$ that $M_n$ was at its maximum" is not, and neither is "the epoch at which validation loss was lowest, chosen after training finishes". Early stopping on a validation curve, done honestly (stop when the criterion fires), is a stopping time; done retrospectively (rewind to the best epoch) it is not — and that difference is exactly why the retrospective procedure needs a held-out set to remain unbiased.

**Proposition 6.22 (you cannot beat a fair game; stopped martingales).**
Let $(M_n)$ be a martingale, $(H_n)_{n\ge1}$ predictable and bounded, and define the *martingale transform* $(H\cdot M)_n:=\sum_{k=1}^nH_k(M_k-M_{k-1})$, $(H\cdot M)_0=0$. Then $(H\cdot M)$ is a martingale. If instead $(M_n)$ is a submartingale (resp. supermartingale) and additionally $H_n\ge0$, then $(H\cdot M)$ is a submartingale (resp. supermartingale). Consequently, for any stopping time $\tau$, the stopped process $M^\tau_n:=M_{n\wedge\tau}$ is a martingale/submartingale/supermartingale whenever $(M_n)$ is, and in the martingale case $\mathbb{E}[M_{n\wedge\tau}]=\mathbb{E}[M_0]$ for all $n$.

*Proof.* $(H\cdot M)_n$ is adapted and integrable ($H$ bounded). Since $H_{n+1}$ is $\mathcal{F}_n$-measurable, pull-out gives
$$
\mathbb{E}[(H\cdot M)_{n+1}-(H\cdot M)_n\mid\mathcal{F}_n]=H_{n+1}\,\mathbb{E}[M_{n+1}-M_n\mid\mathcal{F}_n]=0 .
$$
In the submartingale case the last conditional expectation is $\ge0$ rather than $=0$, and multiplying by $H_{n+1}\ge0$ preserves the sign (monotonicity, Theorem 6.8(b)); the supermartingale case is symmetric. For the stopped process, take $H_k=\mathbf{1}\{\tau\ge k\}=1-\mathbf{1}\{\tau\le k-1\}$, which is $\mathcal{F}_{k-1}$-measurable (Definition 6.21), bounded, and non-negative. Then $(H\cdot M)_n=M_{n\wedge\tau}-M_0$. $\blacksquare$

Read the first statement as: *no betting strategy based on the past changes the expected outcome of a fair game.* This is the mathematical core of the efficient-market intuition, and of the fact that no data-dependent-but-non-anticipating reweighting of an unbiased gradient estimator introduces bias.

**Theorem 6.23 (optional stopping).** Let $(M_n)$ be a martingale and $\tau$ a stopping time. Then $\mathbb{E}[M_\tau]=\mathbb{E}[M_0]$ provided any one of:
&nbsp;&nbsp;(i) $\tau\le N$ a.s. for some constant $N$;
&nbsp;&nbsp;(ii) $\tau<\infty$ a.s. and the stopped process $(M_{n\wedge\tau})_n$ is uniformly bounded by a constant;
&nbsp;&nbsp;(iii) $\mathbb{E}[\tau]<\infty$ and the increments are bounded: $|M_n-M_{n-1}|\le c$ a.s. for all $n$ and some constant $c$.
For a supermartingale, $\mathbb{E}[M_\tau]\le\mathbb{E}[M_0]$ under the same conditions; for a submartingale, $\mathbb{E}[M_\tau]\ge\mathbb{E}[M_0]$. Moreover — and this is the form Doob's maximal inequality will use — for a *submartingale* and a *bounded* stopping time $\tau\le n$,
$$
\mathbb{E}[M_0]\ \le\ \mathbb{E}[M_\tau]\ \le\ \mathbb{E}[M_n]:
$$
stopping a favourable game early cannot beat playing to the end.

*Proof.* By Proposition 6.22, $\mathbb{E}[M_{n\wedge\tau}]=\mathbb{E}[M_0]$ for every $n$. It remains to pass to the limit in $n$, and each hypothesis supplies a different licence.
(i) Take $n=N$: $M_{N\wedge\tau}=M_\tau$ exactly.
(ii) $M_{n\wedge\tau}\to M_\tau$ a.s. (since $\tau<\infty$), with a constant dominating function; apply dominated convergence (Ch. 2).
(iii) Write $|M_{n\wedge\tau}-M_0|=\bigl|\sum_{k=1}^{n\wedge\tau}(M_k-M_{k-1})\bigr|\le c\,\tau$, and $\mathbb{E}[c\tau]<\infty$ by hypothesis, so $c\tau$ is an integrable dominating function; again DCT.

For the submartingale sandwich with $\tau\le n$: the left inequality is Proposition 6.22 (the stopped process is a submartingale, so $\mathbb{E}[M_{n\wedge\tau}]\ge\mathbb{E}[M_0]$, and $M_{n\wedge\tau}=M_\tau$). For the right, write $M_n-M_\tau=\sum_{k=1}^n\mathbf{1}\{\tau<k\}(M_k-M_{k-1})$; each $\mathbf{1}\{\tau<k\}=\mathbf{1}\{\tau\le k-1\}$ is $\mathcal{F}_{k-1}$-measurable and non-negative, so by pull-out and the submartingale property each term has non-negative expectation: $\mathbb{E}\bigl[\mathbf{1}\{\tau<k\}(M_k-M_{k-1})\bigr]=\mathbb{E}\bigl[\mathbf{1}\{\tau<k\}\,\mathbb{E}[M_k-M_{k-1}\mid\mathcal{F}_{k-1}]\bigr]\ge0$. Summing gives $\mathbb{E}[M_n]\ge\mathbb{E}[M_\tau]$. The supermartingale and martingale statements follow by applying this to $-M$ and to $\pm M$. $\blacksquare$

**Worked Example 6.17 (gambler's ruin: optional stopping used correctly).**
Symmetric random walk $M_n$ started at $M_0=a=5$, stopped at $\tau=\inf\{n:M_n\in\{0,10\}\}$. Increments are bounded by $1$ and one can show $\mathbb{E}[\tau]<\infty$ (the walk hits a boundary within any block of 10 consecutive steps with probability $\ge2^{-10}$, so $\tau$ has a geometric tail), so hypothesis (iii) applies:
$$
5=\mathbb{E}[M_0]=\mathbb{E}[M_\tau]=10\,\mathbb{P}(\text{hit }10)+0\cdot\mathbb{P}(\text{hit }0)\implies\mathbb{P}(\text{hit }10)=\tfrac12 .
$$
For the duration, apply optional stopping to the martingale $M_n^2-n$ (a martingale because $\mathbb{E}[M_{n+1}^2\mid\mathcal{F}_n]=M_n^2+1$): $\ \mathbb{E}[M_\tau^2]-\mathbb{E}[\tau]=25$, and $\mathbb{E}[M_\tau^2]=100\cdot\tfrac12=50$, giving $\mathbb{E}[\tau]=25=a(N-a)$. Simulation ($R=500\,000$ paths, seed 14): $\mathbb{P}(\text{hit }10)=0.4996$, $\mathbb{E}[\tau]=24.995$, longest observed run $279$ steps. ✓

**Worked Example 6.18 (the doubling strategy: optional stopping abused).**
Bet $1$ on a fair coin; if you lose, double the stake; stop at your first win. Let $\tau$ be the round of the first head, $\tau\sim\mathrm{Geometric}(1/2)$, and let $M_n$ be your net wealth after $n$ rounds ($M_0=0$). Losing rounds $1,\dots,k-1$ costs $1+2+\cdots+2^{k-2}=2^{k-1}-1$, and the win at round $k$ pays $2^{k-1}$; so $M_\tau=+1$ **with probability one**.

$(M_n)$ is a martingale (each round is a fair bet with a predictable stake), $\tau<\infty$ a.s., and $\mathbb{E}[\tau]=2<\infty$. Yet
$$
\mathbb{E}[M_\tau]=1\ \neq\ 0=\mathbb{E}[M_0].
$$
Which hypothesis failed? Not (i): $\tau$ is unbounded, so (i) never applied. Not $\mathbb{E}\tau<\infty$. It is the *boundedness* clauses: the stopped process is not uniformly bounded (before the first win, wealth is $-(2^{k-1}-1)$, unbounded below), and the increments $\pm2^{n-1}$ are not bounded by a constant, so (ii) and (iii) both fail. The martingale property is intact; the interchange of limit and expectation is not — which is Failure 3 of the Four Failures in a new costume, and the reason Chapter 2's dominated convergence theorem keeps reappearing.

Simulation, seed 13, $N=10^6$ plays: $\mathbb{E}[\tau]=2.0002$, and $M_\tau=1$ in every single play. Now truncate at $n$ rounds ("I only have so much capital"), so the stopping time is $\tau\wedge n$, which is bounded — hypothesis (i) — and the theorem must hold:

| cap $n$ | $\mathbb{P}(\tau>n)$ (theory $2^{-n}$) | loss if you hit the cap | $\mathbb{E}[M_{\tau\wedge n}]$, theory |
|---|---|---|---|
| 5 | $0.031250$ | $-31$ | $0$ |
| 10 | $0.000977$ | $-1023$ | $0$ |
| 20 | $9.537\times10^{-7}$ | $-1\,048\,575$ | $0$ |

The books balance exactly: $\ (1-2^{-n})\cdot(+1)+2^{-n}\cdot(-(2^n-1))=0$. Monte Carlo at cap $20$ with $N=2\times10^7$ plays (seed 23) found only $13$ paths that hit the cap and returned $\mathbb{E}[M_{\tau\wedge20}]\approx+0.31843$ — a *positive* empirical mean for a quantity whose true mean is $0$, because the compensating catastrophe has probability $10^{-6}$ and magnitude $10^6$. **This is what a strategy with a hidden tail looks like from inside the data.** Substitute "sell deep out-of-the-money options", "martingale position sizing", or "a reward hack that works until it doesn't" and the arithmetic is unchanged.

> **Common Confusion 6.8.**
> *"A martingale is a fair game, so my expected profit is zero no matter what I do — but the doubling strategy wins with probability 1. Which is it?"*
>
> Both, and they are compatible because "wins with probability 1" and "has positive expectation" are different claims, joined only under an integrability hypothesis. Proposition 6.22 says no non-anticipating strategy changes the expectation *at any fixed time* $n$ — that part is airtight. Theorem 6.23 says the same at a random time $\tau$ *only if* one of (i)–(iii) holds. The doubling strategy satisfies none, and what it actually buys is: a small gain with probability $1-2^{-n}$ against a ruinous loss with probability $2^{-n}$, exactly balanced. The moment you have finite capital (always), your $\tau$ is truncated and the expectation is $0$ again. The general lesson for ML: whenever you see an estimator that seems to be free money — an unbiased estimator with unbounded weights, a "no-regret" claim without a boundedness assumption, a bound that holds "eventually" without a rate — look for the missing domination hypothesis. It is where the risk went.

> **Self-check 6.7.**
> 1. Is $\tau=\inf\{n:M_n=\max_{k\le 100}M_k\}$ a stopping time? Why or why not?
> 2. Show that if $(M_n)$ is a martingale then $\mathbb{E}[M_m\mid\mathcal{F}_n]=M_n$ for all $m>n$.
> 3. In Worked Example 6.18, compute $\mathbb{E}[M_{\tau\wedge3}]$ by hand from the four possible outcomes.
> 4. Why is $(L_n)$ of Worked Example 6.16 a martingale under $p$ but not under $q$? What is it under $q$?

## 6.8 Martingale inequalities and convergence

Martingales are useful because the single defining identity implies strong quantitative control: over the *whole path* (maximal inequalities), in the *limit* (convergence theorems), and in the *tails* (Azuma–Hoeffding). This section proves the ones we need and states the rest.

### 6.8.1 Maximal inequalities

**Theorem 6.24 (Doob's maximal inequality).** Let $(X_n)$ be a non-negative submartingale and $X_n^*:=\max_{0\le k\le n}X_k$. Then for every $\lambda>0$,
$$
\lambda\,\mathbb{P}(X_n^*\ge\lambda)\ \le\ \mathbb{E}\bigl[X_n\mathbf{1}\{X_n^*\ge\lambda\}\bigr]\ \le\ \mathbb{E}[X_n].
$$

*Proof.* Let $\tau:=\inf\{k\le n:X_k\ge\lambda\}$, with $\tau=n$ if no such $k$ exists; $\tau$ is a bounded stopping time. On the event $A:=\{X_n^*\ge\lambda\}$ we have $X_\tau\ge\lambda$. Since $(X_k)$ is a submartingale and $\tau\le n$ is a bounded stopping time, the submartingale sandwich in Theorem 6.23 gives $\mathbb{E}[X_n]\ge\mathbb{E}[X_\tau]$. Split:
$$
\mathbb{E}[X_\tau]=\mathbb{E}[X_\tau\mathbf{1}_A]+\mathbb{E}[X_\tau\mathbf{1}_{A^c}]\ \ge\ \lambda\mathbb{P}(A)+\mathbb{E}[X_n\mathbf{1}_{A^c}],
$$
using $X_\tau=X_n$ on $A^c$. Rearranging, $\lambda\mathbb{P}(A)\le\mathbb{E}[X_\tau]-\mathbb{E}[X_n\mathbf{1}_{A^c}]\le\mathbb{E}[X_n]-\mathbb{E}[X_n\mathbf{1}_{A^c}]=\mathbb{E}[X_n\mathbf{1}_A]$. $\blacksquare$

Compare Markov's inequality (Ch. 4), which bounds $\mathbb{P}(X_n\ge\lambda)$ at a *single* time. Doob's controls the running maximum at the same price — a free upgrade from pointwise to uniform-over-time, and the reason martingale methods dominate sequential analysis.

Applied to the non-negative martingale $(L_n)$ of Worked Example 6.16 with $\mathbb{E}[L_n]=1$, and letting $n\uparrow\infty$ with continuity from below (Proposition 1.8(3)):
$$
\mathbb{P}\Bigl(\sup_{n\ge0}L_n\ge\tfrac1\alpha\Bigr)\le\alpha .
$$
This is **Ville's inequality**, the sequential analogue of "reject at level $\alpha$", and it is what makes an e-process valid under optional stopping: you may monitor $L_n$ continuously and stop whenever you like, and your type-I error is still at most $\alpha$. In an ML context, this is the correct tool for early-stopping an A/B test or a model-comparison experiment.

**Theorem 6.25 (Doob's $L^p$ inequality; stated, not proved).** For $p>1$ and a non-negative submartingale $(X_n)$,
$$
\bigl\|X_n^*\bigr\|_p\ \le\ \frac{p}{p-1}\,\|X_n\|_p .
$$
In particular $\mathbb{E}[(\max_{k\le n}|M_k|)^2]\le4\,\mathbb{E}[M_n^2]$ for a martingale $(M_n)$ — the maximum of a martingale is, in $L^2$, no worse than a constant factor times its endpoint. (Proof: the layer-cake formula plus Theorem 6.24 and Hölder; see the references.)

### 6.8.2 Convergence

One definition is needed first, because it is exactly the hypothesis that separates "converges" from "converges with its expectation".

**Uniform integrability.** A family $(Z_i)_{i\in I}$ of random variables is *uniformly integrable* (UI) if
$$
\lim_{K\to\infty}\ \sup_{i\in I}\ \mathbb{E}\bigl[|Z_i|\,\mathbf{1}\{|Z_i|>K\}\bigr]=0 .
$$
In words: the mass sitting in the far tail can be made small *simultaneously for the whole family*. A single integrable random variable is UI (dominated convergence); a family dominated by one integrable $W$ is UI; a family bounded in $L^p$ for some $p>1$ is UI. What is **not** UI is a family whose mass escapes to ever-larger values on ever-smaller sets — precisely the behaviour of Worked Example 6.16, where $L_n$ has mean $1$ carried by events of probability $\to0$. The reason UI is the right condition is a theorem of Vitali (Ch. 2): $Z_n\to Z$ a.s. and $(Z_n)$ UI together give $\mathbb{E}[Z_n]\to\mathbb{E}[Z]$, and without UI that conclusion can fail even though everything converges pointwise.

With this language in hand, we can retroactively state the *most general* sufficient condition for optional stopping (Theorem 6.23), which subsumes (i)–(iii): **(iv)** $\tau<\infty$ a.s. and the stopped process $(M_{n\wedge\tau})_n$ is uniformly integrable. Indeed $M_{n\wedge\tau}\to M_\tau$ a.s., and Vitali then gives $\mathbb{E}[M_{n\wedge\tau}]\to\mathbb{E}[M_\tau]$, while Proposition 6.22 keeps $\mathbb{E}[M_{n\wedge\tau}]=\mathbb{E}[M_0]$ for every $n$. Each of (i)–(iii) is just a checkable certificate of (iv): a bounded family is UI, and a family dominated by an integrable $c\tau$ is UI. Reading the doubling strategy (Worked Example 6.18) through this lens: what that strategy manufactures is precisely a stopped process that is *not* uniformly integrable — the losing tail carries mass to $-\infty$ — and (iv) is the single hypothesis it violates.

**Theorem 6.26 (martingale convergence; stated, with sketch).**
(a) Let $(M_n)$ be a supermartingale with $\sup_n\mathbb{E}|M_n|<\infty$ (e.g. any non-negative supermartingale, or any $L^2$-bounded martingale). Then $M_n\to M_\infty$ a.s. for some $M_\infty$ with $\mathbb{E}|M_\infty|<\infty$.
(b) Convergence also holds in $L^1$ if and only if $(M_n)_n$ is uniformly integrable. If $(M_n)$ is moreover a **martingale** with $\sup_n\mathbb{E}[M_n^2]<\infty$, convergence holds in $L^2$ as well — this part is genuinely a martingale fact (its proof uses the orthogonality of martingale increments, $\mathbb{E}[(M_m-M_n)^2]=\sum_{k=n+1}^m\mathbb{E}[D_k^2]$, which fails for general supermartingales), so do not apply it outside the martingale case.
(c) **(Lévy's upward theorem.)** If $X\in L^1$ and $M_n=\mathbb{E}[X\mid\mathcal{F}_n]$ is the Doob martingale of Proposition 6.20(b), then $(M_n)$ *is* uniformly integrable, and
$$
\mathbb{E}[X\mid\mathcal{F}_n]\ \longrightarrow\ \mathbb{E}[X\mid\mathcal{F}_\infty]\qquad\text{a.s. and in }L^1,\qquad \mathcal{F}_\infty:=\sigma\Bigl(\bigcup_n\mathcal{F}_n\Bigr).
$$
In particular, if $X$ is $\mathcal{F}_\infty$-measurable then $\mathbb{E}[X\mid\mathcal{F}_n]\to X$.

Part (c) is what makes Proposition 6.20(b) more than a formal curiosity: **accumulating information recovers the target.** In Worked Example 6.2 the Doob martingale reached $X$ after three flips because the filtration exhausted $\mathcal{F}$; part (c) says the same thing in the limit for infinite filtrations. It is also the honest version of the intuition behind Corollary 6.29 — the Doob martingale of a statistic really does interpolate from $\mathbb{E}[f]$ to $f$ — and it is why Bayesian posteriors based on accumulating data converge to the truth when the truth is identifiable from the data (the parameter must be $\mathcal{F}_\infty$-measurable; when it is not, they converge to something coarser, which is the failure mode behind non-identifiable models). There is a mirror-image *downward* theorem for *decreasing* families of $\sigma$-algebras ("backward martingales"), which we do not need here but flag for Chapter 7: it yields the slickest known proof of the strong law of large numbers, by conditioning the sample mean on the exchangeable information that remains as $n\to\infty$.

*Sketch.* The obstruction to a.s. convergence is oscillation. Fix rationals $a<b$ and let $U_n[a,b]$ count the *upcrossings* of $[a,b]$ by $M_0,\dots,M_n$. Doob's upcrossing inequality — proved by betting $1$ each time the process is below $a$ and holding until it exceeds $b$, a predictable strategy, then applying Proposition 6.22 — gives $(b-a)\mathbb{E}[U_n[a,b]]\le\mathbb{E}[(M_n-a)^-]\le|a|+\sup_n\mathbb{E}|M_n|$. Hence $\mathbb{E}[U_\infty[a,b]]<\infty$, so $U_\infty[a,b]<\infty$ a.s. Taking a *countable* union over rational pairs $a<b$ (countability again) shows that a.s. the path cannot oscillate across any nondegenerate interval infinitely often, so $\lim_nM_n$ exists in $[-\infty,\infty]$; Fatou (Ch. 2) bounds $\mathbb{E}|M_\infty|\le\sup_n\mathbb{E}|M_n|<\infty$, so the limit is a.s. finite. $\blacksquare$

Worked Example 6.16 is the standard cautionary case, and it is precisely a failure of uniform integrability: $L_n\ge0$ is a martingale, so by (a) it converges a.s.; its limit is $0$; but $\mathbb{E}[L_n]=1\not\to0=\mathbb{E}[L_\infty]$. Check the definition against the table there — at $n=50$ the mass sits on events of probability $\sim10^{-6}$ carrying values up to $1.36\times10^4$, so $\mathbb{E}[L_n\mathbf{1}\{L_n>K\}]$ stays near $1$ no matter how large $K$ is taken, uniformly in $n$. Contrast part (c): $L_n$ is a martingale but *not* the Doob martingale of any fixed integrable $X$, which is exactly why Lévy's theorem does not apply to it. The two examples together delimit what martingale convergence promises.  A.s. convergence does **not** give convergence of expectations without uniform integrability. Every time you see "the estimator converges almost surely" in a paper, this example is the reason to ask whether the expectation converges too.

### 6.8.3 Azuma–Hoeffding, and the debt to Chapter 4

Chapter 4 proved Hoeffding's inequality for sums of independent bounded variables and *stated* McDiarmid's bounded-differences inequality. We now pay that debt. The key is that independence was never essential — what mattered was that each increment has conditional mean zero and bounded conditional range.

**Lemma 6.27 (conditional Hoeffding lemma).** Let $\mathcal{H}\subseteq\mathcal{F}$ and let $D$ be a random variable with $\mathbb{E}[D\mid\mathcal{H}]=0$ and $A\le D\le B$ a.s., where $A,B$ are $\mathcal{H}$-measurable with $B-A\le c$ a.s. for a constant $c$. Then for all $\lambda\in\mathbb{R}$,
$$
\mathbb{E}\bigl[e^{\lambda D}\mid\mathcal{H}\bigr]\ \le\ \exp\Bigl(\frac{\lambda^2c^2}{8}\Bigr)\quad\text{a.s.}
$$

*Proof.* By convexity of $d\mapsto e^{\lambda d}$ on $[A,B]$,
$$
e^{\lambda D}\ \le\ \frac{B-D}{B-A}e^{\lambda A}+\frac{D-A}{B-A}e^{\lambda B}\quad\text{a.s.}
$$
Take $\mathbb{E}[\cdot\mid\mathcal{H}]$. The coefficients involve only $\mathcal{H}$-measurable quantities, so pull-out (Theorem 6.8(d)) applies, and $\mathbb{E}[D\mid\mathcal{H}]=0$ leaves
$$
\mathbb{E}[e^{\lambda D}\mid\mathcal{H}]\le\frac{B}{B-A}e^{\lambda A}-\frac{A}{B-A}e^{\lambda B}=:e^{\phi(u)},\qquad u:=\lambda(B-A),
$$
where, writing $\theta=-A/(B-A)\in[0,1]$, $\phi(u)=-\theta u+\log(1-\theta+\theta e^{u})$. Calculus: $\phi(0)=\phi'(0)=0$ and $\phi''(u)=\frac{\theta e^u(1-\theta)}{(1-\theta+\theta e^u)^2}\le\frac14$ (an $x(1-x)$ bound). Taylor with remainder gives $\phi(u)\le u^2/8=\lambda^2(B-A)^2/8\le\lambda^2c^2/8$. All steps hold pointwise a.s. in $\omega$, and measurability of each side is inherited from that of $A,B$. $\blacksquare$

**Theorem 6.28 (Azuma–Hoeffding).** Let $(M_n)_{n\ge0}$ be a martingale with respect to $(\mathcal{F}_n)$ and suppose the increments $D_k:=M_k-M_{k-1}$ satisfy $A_k\le D_k\le A_k+c_k$ a.s. for some $\mathcal{F}_{k-1}$-measurable $A_k$ and constants $c_k$. Then for all $t>0$,
$$
\mathbb{P}(M_n-M_0\ge t)\ \le\ \exp\Bigl(-\frac{2t^2}{\sum_{k=1}^nc_k^2}\Bigr),
$$
and the same bound holds for $\mathbb{P}(M_n-M_0\le-t)$; a union bound gives twice that for the two-sided event. In the common special case $|D_k|\le b_k$ (take $A_k=-b_k$, $c_k=2b_k$) this reads $\exp\bigl(-t^2/(2\sum_kb_k^2)\bigr)$.

*Proof.* Chernoff, exactly as in Chapter 4, but peeling off increments with the tower property instead of independence. For $\lambda>0$, let $S_n:=M_n-M_0=\sum_{k\le n}D_k$. Then
$$
\mathbb{E}\bigl[e^{\lambda S_n}\bigr]=\mathbb{E}\Bigl[e^{\lambda S_{n-1}}\,\mathbb{E}\bigl[e^{\lambda D_n}\mid\mathcal{F}_{n-1}\bigr]\Bigr]\ \le\ e^{\lambda^2c_n^2/8}\,\mathbb{E}\bigl[e^{\lambda S_{n-1}}\bigr],
$$
using the tower property, pull-out ($e^{\lambda S_{n-1}}$ is $\mathcal{F}_{n-1}$-measurable), and Lemma 6.27. Induct: $\mathbb{E}[e^{\lambda S_n}]\le\exp\bigl(\lambda^2\sum_kc_k^2/8\bigr)$. Markov's inequality gives $\mathbb{P}(S_n\ge t)\le\exp\bigl(-\lambda t+\lambda^2\sum_kc_k^2/8\bigr)$, and optimising over $\lambda$ (take $\lambda=4t/\sum_kc_k^2$) yields the claim. Apply to $(-M_n)$ for the lower tail. $\blacksquare$

**Corollary 6.29 (McDiarmid / bounded differences — the Chapter 4 debt, paid).**
Let $\xi_1,\dots,\xi_n$ be independent (not necessarily identically distributed) random elements and let $f$ satisfy the bounded-differences condition
$$
\bigl|f(x_1,\dots,x_k,\dots,x_n)-f(x_1,\dots,x_k',\dots,x_n)\bigr|\le c_k\qquad\text{for all }x,\;x_k' ,
$$
with $\mathbb{E}|f(\xi)|<\infty$. Then for $t>0$, $\ \mathbb{P}\bigl(f(\xi)-\mathbb{E}f(\xi)\ge t\bigr)\le\exp\bigl(-2t^2/\sum_kc_k^2\bigr)$.

*Proof.* Let $\mathcal{F}_k=\sigma(\xi_1,\dots,\xi_k)$ and $M_k=\mathbb{E}[f(\xi)\mid\mathcal{F}_k]$, the Doob martingale of Proposition 6.20(b), so $M_0=\mathbb{E}f(\xi)$ and $M_n=f(\xi)$. By independence and disintegration (§6.6), $M_k=g_k(\xi_1,\dots,\xi_k)$ with $g_k(x_{1:k})=\mathbb{E}[f(x_{1:k},\xi_{k+1:n})]$. Define, for fixed $\xi_{1:k-1}$,
$$
A_k:=\inf_{x}\bigl\{g_k(\xi_{1:k-1},x)\bigr\}-M_{k-1},\qquad B_k:=\sup_{x}\bigl\{g_k(\xi_{1:k-1},x)\bigr\}-M_{k-1},
$$
which are $\mathcal{F}_{k-1}$-measurable, satisfy $A_k\le D_k\le B_k$, and — because the bounded-differences property survives averaging over $\xi_{k+1:n}$ — obey $B_k-A_k\le c_k$. (Measurability of the inf/sup requires a mild regularity assumption such as separability of the range space; for $\xi_k$ taking values in a Polish space one uses a countable dense set, Chapter 1's countability point once more.) Theorem 6.28 applies. $\blacksquare$

Notice what this buys beyond Chapter 4's Hoeffding: $f$ need not be a sum, and — in Theorem 6.28 — **the increments need not be independent at all**, only conditionally centred and conditionally bounded. That is precisely the situation in online learning, in bandit algorithms (where the arm you pull depends on everything before), and in SGD (where $\theta_t$ depends on all previous gradients).

**Worked Example 6.19 (how loose is Azuma?).**
Take $M_n=\sum_{i\le n}\xi_i$ with $\xi_i=\pm1$ fair, $n=100$; increments satisfy $|D_k|\le1$, so Theorem 6.28 gives $\mathbb{P}(M_n\ge t)\le e^{-t^2/200}$. Simulation with $R=2\times10^6$ replicates, seed 16:

| $t$ | empirical $\mathbb{P}(M_{100}\ge t)$ | Azuma bound | normal approximation $1-\Phi(t/10)$ |
|---|---|---|---|
| 10 | 0.184393 | 0.606531 | 0.158655 |
| 20 | 0.028647 | 0.135335 | 0.022750 |
| 30 | 0.001793 | 0.011109 | 0.001350 |

Azuma is correct but loose by a factor of roughly $3$–$6$ here; the normal approximation is much closer (and with a continuity correction, $1-\Phi(0.9)=0.1841$, it nails the $t=10$ row). This is the usual trade: Azuma is *non-asymptotic and assumption-light* — no independence, no distributional form, valid at every $n$ — while the CLT (Ch. 7) is sharp but asymptotic and gives no finite-$n$ guarantee. Generalisation bounds use Azuma-type inequalities for exactly this reason and inherit exactly this looseness, which is one honest explanation for why bounds of the form "test error $\le$ train error $+\,$something" are often vacuous numerically while still being the right qualitative statement.

**Worked Example 6.20 (concentration for sequentially revealed data).**
Suppose an online algorithm processes examples one at a time; at step $k$ it emits a prediction based on $\xi_{1:k-1}$ and suffers loss $\ell_k\in[0,1]$. The losses are *not* independent — later losses depend on earlier ones through the algorithm's state — so Chapter 4's Hoeffding does not apply. But $D_k:=\ell_k-\mathbb{E}[\ell_k\mid\mathcal{F}_{k-1}]$ has conditional mean zero and conditional range $\le1$, so $M_n=\sum_{k\le n}D_k$ is a martingale satisfying Theorem 6.28 with $c_k=1$:
$$
\mathbb{P}\Bigl(\frac1n\sum_{k=1}^n\ell_k-\frac1n\sum_{k=1}^n\mathbb{E}[\ell_k\mid\mathcal{F}_{k-1}]\ \ge\ \varepsilon\Bigr)\le e^{-2n\varepsilon^2}.
$$
With $n=1000$ and $\delta=0.05$: $\varepsilon=\sqrt{\log(1/\delta)/(2n)}=0.0387$. So the realised average loss is within $0.039$ of the average *conditional expected* loss with probability $0.95$ — a statement about a quantity that is itself random, which is the correct form for adaptive settings and the basis of online-to-batch conversion.

> **Self-check 6.8.**
> 1. Why does Theorem 6.24 require non-negativity? Find a martingale where the conclusion fails without it.
> 2. In Ville's inequality, which Chapter 1 proposition licenses passing from $\max_{n\le N}$ to $\sup_n$?
> 3. In Theorem 6.28, what do you gain by allowing $A_k$ to be $\mathcal{F}_{k-1}$-measurable rather than constant?
> 4. Chapter 4 proved Hoeffding for independent variables. Name two hypotheses Azuma drops and one it keeps.
> 5. Using the table in Worked Example 6.16, argue directly from the definition that $(L_n)$ is not uniformly integrable. Which part of Theorem 6.26 does this block, and which part still applies?

## 6.9 Capstone: a full convergence proof for SGD

Chapter 1's Exercise C1 asked you to set up the filtration for stochastic gradient descent and left the analysis for "later". This is later.

### 6.9.1 The setting and the assumptions

We minimise $F:\mathbb{R}^d\to\mathbb{R}$ — think of $F(\theta)=\mathbb{E}_{(X,Y)\sim\mathbb{P}}[\ell(\theta;X,Y)]$, the population risk, or the empirical risk $\frac1n\sum_i\ell(\theta;x_i,y_i)$. The algorithm is
$$
\theta_{t+1}=\theta_t-\eta_t g_t,\qquad t=0,1,2,\dots
$$
with $\theta_0$ deterministic. Define the filtration $\mathcal{F}_t:=\sigma(\theta_0,g_0,g_1,\dots,g_{t-1})$, so that $\mathcal{F}_0=\{\varnothing,\Omega\}$ (nothing is known before we start) and $\theta_t$ is $\mathcal{F}_t$-measurable — the iterate is a deterministic function of the past gradients. **This is the filtration; everything else follows from getting it right.** The stochastic gradient $g_t$ is *not* $\mathcal{F}_t$-measurable: it is the fresh randomness of step $t$ (a minibatch draw, a dropout mask, a data-augmentation sample).

The assumptions, numbered for later reference:

**(A1) Smoothness.** $F$ is differentiable and $\nabla F$ is $L$-Lipschitz: $\|\nabla F(\theta)-\nabla F(\theta')\|\le L\|\theta-\theta'\|$. Consequently (a standard calculus fact, proved by integrating along the segment) the *descent lemma* holds:
$$
F(\theta')\le F(\theta)+\langle\nabla F(\theta),\theta'-\theta\rangle+\frac{L}{2}\|\theta'-\theta\|^2\quad\text{for all }\theta,\theta'.
$$
**(A2) Lower bound.** $F_\star:=\inf_\theta F(\theta)>-\infty$.
**(A3) Conditional unbiasedness.** $\mathbb{E}[g_t\mid\mathcal{F}_t]=\nabla F(\theta_t)$ a.s. for every $t$.
**(A4) Bounded conditional variance.** $\mathbb{E}\bigl[\|g_t-\nabla F(\theta_t)\|^2\mid\mathcal{F}_t\bigr]\le\sigma^2$ a.s.
**(A5) Robbins–Monro step sizes.** $\eta_t>0$ is deterministic (or, more generally, $\mathcal{F}_t$-measurable and bounded — i.e. *predictable*), with $\eta_t\le1/L$, $\sum_{t}\eta_t=\infty$ and $\sum_t\eta_t^2<\infty$.

(A3) is the statement Chapter 1 could not yet parse. Read it now: *conditional on the entire history, the gradient estimate points, on average, in the true direction.* It is exactly what fresh i.i.d. minibatch sampling delivers, and §6.9.4 shows what goes wrong without it.

### 6.9.2 The theorem

**Theorem 6.30 (SGD on smooth non-convex objectives).** Under (A1)–(A5), with $\Gamma_T:=\sum_{t=0}^{T-1}\eta_t$,
$$
\min_{0\le t<T}\mathbb{E}\bigl\|\nabla F(\theta_t)\bigr\|^2\ \le\ \frac{\sum_{t<T}\eta_t\,\mathbb{E}\|\nabla F(\theta_t)\|^2}{\Gamma_T}\ \le\ \frac{2\bigl(F(\theta_0)-F_\star\bigr)+L\sigma^2\sum_{t<T}\eta_t^2}{\Gamma_T}\ \xrightarrow[T\to\infty]{}\ 0 .
$$
In particular $\liminf_{t\to\infty}\mathbb{E}\|\nabla F(\theta_t)\|^2=0$: the iterates visit approximately stationary points. (Strictly, the displayed bound forces $\min_{t<T}\to0$, which gives the $\liminf$ claim unless some *single* iterate has $\mathbb{E}\|\nabla F(\theta_t)\|^2=0$ exactly — in which case that iterate is already stationary in expectation and the conclusion is stronger, not weaker. We will not mention the case again, but it is the kind of hypothesis-checking this chapter asks you to make habitual.) Moreover the conclusion $\to0$ holds whenever $\Gamma_T\to\infty$ and $\bigl(\sum_{t<T}\eta_t^2\bigr)/\Gamma_T\to0$, which is *weaker* than (A5)'s $\sum\eta_t^2<\infty$.

*Proof.* **Step 1 (descent inequality).** Apply (A1) with $\theta=\theta_t$, $\theta'=\theta_{t+1}=\theta_t-\eta_tg_t$:
$$
F(\theta_{t+1})\ \le\ F(\theta_t)-\eta_t\langle\nabla F(\theta_t),g_t\rangle+\frac{L\eta_t^2}{2}\|g_t\|^2 .
$$
This is a pathwise inequality — no probability yet.

**Step 2 (condition on the past).** Take $\mathbb{E}[\,\cdot\mid\mathcal{F}_t]$. Because $\theta_t$ is $\mathcal{F}_t$-measurable, so are $F(\theta_t)$, $\nabla F(\theta_t)$ and $\eta_t$; by pull-out (Theorem 6.8(d)) and (A3),
$$
\mathbb{E}\bigl[\langle\nabla F(\theta_t),g_t\rangle\mid\mathcal{F}_t\bigr]=\bigl\langle\nabla F(\theta_t),\mathbb{E}[g_t\mid\mathcal{F}_t]\bigr\rangle=\|\nabla F(\theta_t)\|^2 .
$$
For the quadratic term, the conditional bias–variance split (Definition 6.12, applied coordinatewise and summed) plus (A3)–(A4) give
$$
\mathbb{E}\bigl[\|g_t\|^2\mid\mathcal{F}_t\bigr]=\bigl\|\mathbb{E}[g_t\mid\mathcal{F}_t]\bigr\|^2+\mathbb{E}\bigl[\|g_t-\nabla F(\theta_t)\|^2\mid\mathcal{F}_t\bigr]\le\|\nabla F(\theta_t)\|^2+\sigma^2 .
$$
Substituting,
$$
\mathbb{E}[F(\theta_{t+1})\mid\mathcal{F}_t]\ \le\ F(\theta_t)-\eta_t\Bigl(1-\frac{L\eta_t}{2}\Bigr)\|\nabla F(\theta_t)\|^2+\frac{L\eta_t^2\sigma^2}{2}\ \le\ F(\theta_t)-\frac{\eta_t}{2}\|\nabla F(\theta_t)\|^2+\frac{L\sigma^2}{2}\eta_t^2,
$$
where the last step used $\eta_t\le1/L$ from (A5), so $1-L\eta_t/2\ge1/2$. **Read this line as the whole story of SGD:** expected progress is a gain proportional to $\eta_t\|\nabla F\|^2$ minus a noise penalty proportional to $\eta_t^2\sigma^2$. Small steps make the penalty negligible relative to the gain; that is the entire art of step-size selection.

**Step 3 (telescope).** Take total expectations (tower property) and sum for $t=0,\dots,T-1$; the $F$ terms telescope:
$$
\frac12\sum_{t<T}\eta_t\,\mathbb{E}\|\nabla F(\theta_t)\|^2\ \le\ \mathbb{E}[F(\theta_0)]-\mathbb{E}[F(\theta_T)]+\frac{L\sigma^2}{2}\sum_{t<T}\eta_t^2\ \le\ F(\theta_0)-F_\star+\frac{L\sigma^2}{2}\sum_{t<T}\eta_t^2,
$$
using (A2). Divide by $\Gamma_T/2$. A weighted average dominates its minimum, giving the left inequality. Under (A5) the numerator is bounded (as $\sum\eta_t^2<\infty$) while $\Gamma_T\to\infty$, so the bound tends to $0$. $\blacksquare$

Three things to notice. *The proof never used convexity* — only smoothness and a lower bound; that is why it is the standard workhorse for deep learning, where convexity is absent. *It concludes about gradients, not about function values or iterates*: without convexity you cannot conclude $\theta_t\to\theta_\star$, and SGD may drift along a flat valley forever. *The noise enters only through $\sigma^2$ multiplying $\sum\eta_t^2$*, which is the formal version of "smaller steps average out the noise".

**Theorem 6.31 (almost-sure version; Robbins–Siegmund route, stated).** Under (A1)–(A5), $F(\theta_t)$ converges a.s. to a finite random limit, $\sum_t\eta_t\|\nabla F(\theta_t)\|^2<\infty$ a.s., and consequently $\liminf_{t}\|\nabla F(\theta_t)\|=0$ a.s.

*Why it is true, in one paragraph.* Set $Y_t:=F(\theta_t)-F_\star+\frac{L\sigma^2}{2}\sum_{s\ge t}\eta_s^2$, which is well defined because $\sum\eta_s^2<\infty$. Step 2 rearranges to
$$
\mathbb{E}[Y_{t+1}\mid\mathcal{F}_t]\ \le\ Y_t-\frac{\eta_t}{2}\|\nabla F(\theta_t)\|^2\ \le\ Y_t ,
$$
so $(Y_t)$ is a **non-negative supermartingale**. By Theorem 6.26 it converges a.s. to a finite limit; hence $F(\theta_t)$ does too. Summing the middle inequality and taking expectations bounds $\mathbb{E}\sum_t\eta_t\|\nabla F(\theta_t)\|^2<\infty$, so the sum is finite a.s. Since $\sum\eta_t=\infty$, the summand's factor $\|\nabla F(\theta_t)\|^2$ cannot stay bounded away from $0$. (The general statement — the Robbins–Siegmund almost-supermartingale lemma — is in the references; we use only the non-negative-supermartingale case, which Theorem 6.26 covers.) $\blacksquare$

This is the cleanest illustration of the chapter's thesis: **SGD converges because a suitably shifted version of the loss along the trajectory is a supermartingale.** The optimisation statement *is* a martingale statement.

**Corollary 6.32 (convex case: function values converge).** Assume (A3), (A5) with $\sum\eta_t^2<\infty$, that $F$ is convex with a minimiser $\theta_\star$, and $\mathbb{E}[\|g_t\|^2\mid\mathcal{F}_t]\le G^2$. Let $\bar\theta_T:=\Gamma_T^{-1}\sum_{t<T}\eta_t\theta_t$. Then
$$
\mathbb{E}\bigl[F(\bar\theta_T)\bigr]-F(\theta_\star)\ \le\ \frac{\|\theta_0-\theta_\star\|^2+G^2\sum_{t<T}\eta_t^2}{2\,\Gamma_T}\ \xrightarrow[T\to\infty]{}0 .
$$

*Proof.* Expand the squared distance and condition:
$$
\mathbb{E}\bigl[\|\theta_{t+1}-\theta_\star\|^2\mid\mathcal{F}_t\bigr]=\|\theta_t-\theta_\star\|^2-2\eta_t\langle\nabla F(\theta_t),\theta_t-\theta_\star\rangle+\eta_t^2\mathbb{E}[\|g_t\|^2\mid\mathcal{F}_t],
$$
using (A3) and pull-out on the cross term. Convexity gives $\langle\nabla F(\theta_t),\theta_t-\theta_\star\rangle\ge F(\theta_t)-F(\theta_\star)$. Take total expectations, multiply by nothing, telescope, and divide by $2\Gamma_T$ to bound the $\eta$-weighted average of $\mathbb{E}[F(\theta_t)]-F(\theta_\star)$; Jensen's inequality (Ch. 4) moves the average inside $F$. $\blacksquare$

### 6.9.3 What the theorem predicts, and what actually happens

**Worked Example 6.21 (three schedules on a quadratic).**
Take $F(\theta)=\tfrac12\theta^\top A\theta$ with $A=\operatorname{diag}(1,10)$ (so $L=10$, condition number $10$), $\theta_0=(2,2)$, and $g_t=\nabla F(\theta_t)+\xi_t$ with $\xi_t\sim N(0,I_2)$ i.i.d., so (A3) holds and $\sigma^2=2$. Fifty runs (seeds $1000,\dots,1049$), $T=10^5$ steps; the table reports the mean of $F(\theta_t)$ across runs.

One point of hypothesis-checking before the table — the kind this chapter keeps insisting on. The classical "$\eta_t=1/t$" schedule, quoted raw, **violates (A5)** for $t\le9$: it exceeds $1/L=0.1$ there, and at $t=1$ the update map on the stiff coordinate has factor $1-\eta_1a_2=1-10=-9$, which is *expansive* — the first few steps blow the iterate up before the decay rescues it. (A5)'s bound $\eta_t\le1/L$ is not decoration; it is what makes Step 2's bracket positive. So we run the schedule the theorem actually licenses, $\eta_t=\min(1/L,\,1/t)$: constant at $0.1$ for the first nine steps, then $1/t$. Asymptotically nothing changes; the early trajectory does.

| schedule | $t=10$ | $10^2$ | $10^3$ | $10^4$ | $10^5$ | satisfies (A5)? |
|---|---|---|---|---|---|---|
| $\eta=0.05$ (constant) | $7.15\times10^{-1}$ | $3.74\times10^{-2}$ | $2.64\times10^{-2}$ | $2.48\times10^{-2}$ | $2.53\times10^{-2}$ | no: $\sum\eta_t^2=\infty$, $(\sum\eta^2)/\Gamma_T\not\to0$ |
| $\eta_t=\min(0.1,\,1/t)$ | $2.78\times10^{-1}$ | $1.01\times10^{-2}$ | $8.99\times10^{-4}$ | $8.58\times10^{-5}$ | $8.40\times10^{-6}$ | yes |
| $\eta_t=0.1/\sqrt t$ | $7.10\times10^{-1}$ | $4.87\times10^{-2}$ | $2.04\times10^{-3}$ | $5.72\times10^{-4}$ | $1.38\times10^{-4}$ | no ($\sum\eta_t^2=\infty$) but $(\sum\eta_t^2)/\Gamma_T\to0$ |

Every number matches the theory, including the failures.

*Constant step.* The bound's numerator grows like $T\eta^2$ and its denominator like $T\eta$, so it plateaus at $O(\eta L\sigma^2)$ — no convergence to zero, only to a noise ball. The measured plateau is $2.5\times10^{-2}$; the exact stationary value for this Ornstein–Uhlenbeck-like recursion is $\frac12\sum_ia_i\frac{\eta^2}{1-(1-\eta a_i)^2}=0.02949$. A tighter Monte Carlo run (4000 seeds, $T=2\times10^4$, seed 77) gives $\mathbb{E}[F(\theta_\infty)]=0.02933\pm0.00048$ — agreement to within one standard error. Constant-step SGD does not converge; it equilibrates, and the equilibrium loss is proportional to $\eta$ (measured: $0.00503$ at $\eta=0.01$, $0.01087$ at $\eta=0.02$, versus theory $0.00514$ and $0.01061$). *This is why learning-rate decay is not a heuristic.*

*$\eta_t=\min(0.1,1/t)$.* Satisfies (A5) exactly (the cap is what makes that true), and the loss decays like $1/t$: from $8.99\times10^{-4}$ to $8.58\times10^{-5}$ to $8.40\times10^{-6}$ across successive decades — a clean factor of ten each time.

*$\eta_t=0.1/\sqrt t$.* Violates the letter of (A5) ($\sum\eta_t^2\asymp\log T$) but satisfies the weaker sufficient condition in Theorem 6.30, since $(\sum_{t<T}\eta_t^2)/\Gamma_T=O(\log T/\sqrt T)\to0$. And indeed it converges, more slowly. **The theorem's weaker hypothesis is the one that matters in practice** — a good illustration of why one should read the proof and not only the assumption list: $\sqrt{t}$-decay and cosine schedules are used constantly and are *not* covered by the classical Robbins–Monro conditions as usually quoted.

<svg viewBox="0 0 720 400" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Log-log plot of mean loss versus iteration for three step-size schedules">
  <line x1="90" y1="310" x2="640" y2="310" stroke="currentColor" stroke-width="1.5"/>
  <line x1="90" y1="50" x2="90" y2="310" stroke="currentColor" stroke-width="1.5"/>
  <g fill="currentColor" font-size="11" text-anchor="middle">
    <text x="90" y="328">10¹</text><text x="220" y="328">10²</text><text x="350" y="328">10³</text><text x="480" y="328">10⁴</text><text x="610" y="328">10⁵</text>
    <text x="365" y="348">iteration t</text>
  </g>
  <g fill="currentColor" font-size="11" text-anchor="end">
    <text x="84" y="64">10⁰</text><text x="84" y="147">10⁻²</text><text x="84" y="231">10⁻⁴</text><text x="84" y="314">10⁻⁶</text>
  </g>
  <text x="24" y="40" fill="currentColor" font-size="11">mean F(θ_t)</text>
  <line x1="90" y1="128" x2="640" y2="128" stroke="#ef4444" stroke-width="1" stroke-dasharray="6 4"/>
  <text x="404" y="122" fill="#ef4444" font-size="11">exact stationary E[F] for η = 0.05: 0.0295</text>
  <polyline points="90,70 220,123 350,130 480,131 610,131" fill="none" stroke="#ef4444" stroke-width="2.5"/>
  <polyline points="90,87 220,147 350,190 480,233 610,275" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
  <polyline points="90,70 220,119 350,176 480,199 610,225" fill="none" stroke="#10b981" stroke-width="2.5"/>
  <text x="470" y="146" fill="#ef4444" font-size="12">η = 0.05 constant</text>
  <text x="470" y="266" fill="#3b82f6" font-size="12">η_t = min(0.1, 1/t)</text>
  <text x="470" y="216" fill="#10b981" font-size="12">η_t = 0.1/√t</text>
  <text x="24" y="372" fill="currentColor" font-size="13">Figure 6.6 — Mean loss over 50 runs of SGD on a quadratic (A = diag(1,10), σ² = 2), log–log axes, plotted from</text>
  <text x="24" y="389" fill="currentColor" font-size="13">the measured values. Constant steps flatten onto the dashed line — the exact stationary mean loss 0.0295, which</text>
  <text x="24" y="406" fill="currentColor" font-size="13">is proportional to η and does not decrease with T; min(0.1, 1/t) decays like 1/t; 0.1/√t converges more slowly.</text>
</svg>

### 6.9.4 Where the proof breaks — one assumption at a time

This is the part worth memorising, because each item below is a real failure mode with a name.

**(a) Unconditional unbiasedness is not enough (completing Chapter 1's Exercise C1(d)).**
Suppose we weaken (A3) to $\mathbb{E}[g_t]=\mathbb{E}[\nabla F(\theta_t)]$ — unbiased *on average over everything*. Step 2 collapses immediately: the manipulation
$$
\mathbb{E}\bigl[\langle\nabla F(\theta_t),g_t\rangle\mid\mathcal{F}_t\bigr]=\bigl\langle\nabla F(\theta_t),\mathbb{E}[g_t\mid\mathcal{F}_t]\bigr\rangle
$$
requires pulling out the $\mathcal{F}_t$-measurable factor $\nabla F(\theta_t)$, and what remains is the *conditional* mean of $g_t$. With only an unconditional guarantee, $\mathbb{E}[\langle\nabla F(\theta_t),g_t\rangle]$ is a covariance we cannot sign: the noise may be correlated with the current iterate.

Concretely: draw $\zeta=\pm1$ once at time $0$, and set $g_t=\nabla F(\theta_t)+b\zeta$. The noise has unconditional mean $0$ at every step (so $\mathbb{E}[g_t]=\mathbb{E}[\nabla F(\theta_t)]$ ✓), but $\zeta$ is $\mathcal{F}_t$-measurable for $t\ge1$, so $\mathbb{E}[g_t\mid\mathcal{F}_t]=\nabla F(\theta_t)+b\zeta\ne\nabla F(\theta_t)$. For $F(\theta)=\theta^2/2$ the recursion converges to $\theta_\infty=-b\zeta$, not $0$. Simulation ($b=0.5$, $\eta=0.05$, $T=2\times10^4$, 2000 runs, seed 99): $|\theta_T|=0.5000$ in every run, mean $F(\theta_T)=0.1250$, exactly $b^2/2$. The average over runs of $\theta_T$ is $\approx0$ — the estimator looks unbiased in aggregate — while **every individual run converges to the wrong point.** No amount of averaging over runs helps a practitioner who gets one run.

Real instances of the same structure: a data-loading bug that fixes the shuffling seed per worker; a fixed augmentation applied identically each epoch (verified numerically in the remark to Exercise C3); a stale gradient in asynchronous training that depends on an earlier iterate. The predicted symptom — and it is a prediction of the algebra above, not an anecdote — is a loss that plateaus above the achievable minimum and, crucially, does *not* improve when you shrink the step size, because the obstruction is a bias rather than a noise floor. That test distinguishes this failure from failure (b) below.

**(b) Heavy-tailed noise: (A4) fails.**
If $\mathbb{E}[\|g_t-\nabla F(\theta_t)\|^2\mid\mathcal{F}_t]=\infty$, Step 2's bound on $\mathbb{E}[\|g_t\|^2\mid\mathcal{F}_t]$ is vacuous and the descent inequality carries an infinite penalty. Chapter 2's Cauchy example — a distribution with no mean, let alone a variance — is the canonical case, and empirically gradient-noise distributions in transformer training have been reported to be heavy-tailed.

**Worked Example 6.22 (Cauchy gradient noise).**
Same quadratic, same (A5)-compliant schedule $\eta_t=\min(0.1,1/t)$, $T=2\times10^4$, 50 seeds (0–49), but $\xi_t$ now i.i.d. standard Cauchy per coordinate:

| noise | median $F(\theta_T)$ | mean $F(\theta_T)$ | max $F(\theta_T)$ |
|---|---|---|---|
| Gaussian ($\sigma^2=2$) | $2.63\times10^{-5}$ | $3.64\times10^{-5}$ | $1.76\times10^{-4}$ |
| Cauchy | $1.09$ | $1.04\times10^{3}$ | $5.07\times10^{4}$ |

With Cauchy noise, $52\%$ of runs end with $F>1$, and the worst run ends *above* the initial $F(\theta_0)=22$ by three orders of magnitude — a single enormous gradient can throw the iterate arbitrarily far, and the $1/t$ steps are by then too small to walk back. With a constant step $\eta=0.05$ and $T=10^5$ the picture is the same (median $F=1.33$, mean $43.4$, max $776$). The median stalls around $1$: typical runs *look* like they are converging slowly, while the mean is dominated by catastrophes — the same statistical signature as Worked Example 6.18's doubling strategy and Worked Example 6.16's likelihood ratio. **Heavy tails do not announce themselves in the median.**

The standard fix is *gradient clipping*: replace $g_t$ by $g_t\min(1,c/\|g_t\|)$. This restores (A4) — the clipped gradient is bounded — at the cost of (A3), since clipping biases the estimate. The modern analyses of clipped SGD are therefore not corollaries of Theorem 6.30; they trade a variance assumption for a bias term that must be controlled separately.

**(c) Step sizes that depend on $g_t$: (A5)'s predictability fails.**
Adam, AdaGrad and RMSProp choose $\eta_t$ using the *current* gradient. Then $\eta_t$ is $\mathcal{F}_{t+1}$-measurable but not $\mathcal{F}_t$-measurable, and pull-out in Step 2 is illegal: $\mathbb{E}[\eta_tg_t\mid\mathcal{F}_t]\ne\eta_t\nabla F(\theta_t)$, because $\eta_t$ and $g_t$ are correlated by construction (a large gradient produces a small step). This is not a pedantic objection — it is the reason adaptive-method convergence proofs are substantially harder than Theorem 6.30, why the original Adam proof was found to be flawed, and why analyses typically either use $\eta_t$ built from gradients up to $t-1$ only (restoring predictability) or introduce a decorrelation argument.

**(d) Correlated minibatches: (A3) fails.**
Sampling without replacement within an epoch (standard practice!) makes $g_t$ depend on which examples were already used, so $\mathbb{E}[g_t\mid\mathcal{F}_t]$ is the average gradient over the *remaining* examples, not over the full dataset. Random reshuffling has its own analysis and, pleasantly, is often *better* than i.i.d. sampling; the point is that Theorem 6.30 does not cover it. Worse offenders: curriculum ordering, sorting by length, temporally correlated replay buffers in RL, and any pipeline where consecutive batches share examples.

**(e) A subtler point: (A1)'s $L$ is global.** Deep networks are not globally $L$-smooth; the local curvature along the trajectory can grow sharply. There is an empirical literature reporting that the top Hessian eigenvalue along full-batch gradient-descent trajectories rises until it sits near $2/\eta$ and then hovers there (the "edge of stability"); we have not reproduced that here and report it as a reference claim rather than a verified one. Theorem 6.30's requirement $\eta_t\le1/L$ then has no fixed meaning, and the theorem's guarantee is best read as a statement about a region rather than about all of parameter space.

> **Self-check 6.9.**
> 1. In Step 2, exactly which property of conditional expectation is used to justify $\mathbb{E}[\langle\nabla F(\theta_t),g_t\rangle\mid\mathcal{F}_t]=\|\nabla F(\theta_t)\|^2$? Name both properties involved.
> 2. Why does Theorem 6.30 conclude something about $\min_{t<T}$ rather than about $\theta_T$?
> 3. For constant $\eta$, what does the bound in Theorem 6.30 converge to as $T\to\infty$, and does the experiment agree?
> 4. Which assumption does each of the following violate: (a) sorting the training set by label; (b) Adam; (c) a fixed augmentation seed per example; (d) unclipped gradients with heavy-tailed noise?

## 6.10 Results used but not proved

| Result | Used in | Where to find a proof |
|---|---|---|
| Radon–Nikodym theorem ($\nu\ll\mu$, $\mu$ $\sigma$-finite, $\nu$ finite) | Theorem 6.2 (existence of $\mathbb{E}[X\mid\mathcal{G}]$) | Ch. 2; Folland, *Real Analysis* §3.2; Durrett §A.4 |
| Monotone convergence theorem | Theorem 6.2, Proposition 6.11(a), disintegration | Ch. 2 |
| Dominated convergence theorem | Proposition 6.11(c), Theorem 6.23(ii)–(iii) | Ch. 2 |
| Fatou's lemma | Theorem 6.26(a) (finiteness of the limit) | Ch. 2 |
| Vitali convergence theorem (a.s. convergence $+$ uniform integrability $\Rightarrow$ $L^1$ convergence) | Theorem 6.26(b); the UI discussion preceding it | Ch. 2; Williams §13.7 |
| Lévy's upward theorem | Theorem 6.26(c) | Williams §14.2; Durrett §4.6 |
| Fubini–Tonelli | Propositions 6.7(a), 6.17; disintegration | Ch. 2 |
| Completeness of $L^2$ (Riesz–Fischer) and the orthogonal projection theorem for closed subspaces of a Hilbert space | Theorem 6.5 | Ch. 2; Brezis, *Functional Analysis* Ch. 5 |
| Cauchy–Schwarz inequality ($L^2\subseteq L^1$ on a probability space) | Theorem 6.5(2) | Ch. 2 |
| Markov's inequality | Theorem 6.28, Ville's inequality | Ch. 4 |
| Jensen's inequality (unconditional) | Corollary 6.32 | Ch. 4 |
| Change-of-variables formula for densities | §6.6.4 (Borel paradox) | Ch. 3 |
| Gaussian conditioning formula for jointly Gaussian vectors | Worked Example 6.14; Common Confusion 6.7 | Ch. 5 |
| A convex function on $\mathbb{R}$ is the supremum of a *countable* family of affine minorants | Theorem 6.9 (conditional Jensen) | Rockafellar, *Convex Analysis* Thm 12.1; or construct from rational tangency points |
| Descent lemma: $L$-Lipschitz gradient $\Rightarrow F(\theta')\le F(\theta)+\langle\nabla F(\theta),\theta'-\theta\rangle+\frac L2\|\theta'-\theta\|^2$ | Assumption (A1), Theorem 6.30 Step 1 | Nesterov, *Lectures on Convex Optimization* Lemma 1.2.3 |
| Existence and a.e.-uniqueness of regular conditional distributions on Polish spaces | Theorem 6.16, disintegration, Corollary 6.29 | Durrett §4.1.3; Kallenberg, *Foundations* Thm 6.3; Çınlar Ch. IV |
| Borel isomorphism of uncountable Polish spaces with $\mathbb{R}$ | sketch of Theorem 6.16 | Kechris, *Classical Descriptive Set Theory* Thm 15.6 |
| Converse of Proposition 1.19 (every non-decreasing right-continuous $F$ with the right limits is a CDF) | sketch of Theorem 6.16 | Ch. 1 (taken on faith there); Billingsley §14 |
| Doob's $L^p$ maximal inequality | Theorem 6.25 (stated) | Williams §14.11; Durrett §4.4 |
| Doob's upcrossing inequality | Theorem 6.26(a) (sketched, not proved) | Williams §11.1–11.5; Durrett §4.2 |
| $L^2$ convergence of $L^2$-bounded martingales (via orthogonality of increments) | Theorem 6.26(b), $L^2$ clause | Williams §12.1; Durrett §4.4 |
| Robbins–Siegmund almost-supermartingale lemma (general form) | Theorem 6.31 (we use only the non-negative supermartingale case) | Robbins & Siegmund (1971); Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming* Prop. 4.2 |
| Measurable selection of a quantile (the smallest median is measurable in $x$) | Proposition 6.7(a) | van der Vaart, *Asymptotic Statistics* §21.1 |
| Measurability of $\inf_x g_k(\cdot,x)$ over an uncountable index set under separability | Corollary 6.29 (McDiarmid) | Boucheron–Lugosi–Massart §6.1; Ch. 1's countability discussion |
| Strong law of large numbers (used only to assert $\frac1n\log L_n\to-D_{\mathrm{KL}}$) | Worked Example 6.16 | Ch. 7 (forward reference) |
| Stationary variance of the linear recursion $\theta_{t+1}=(1-\eta a)\theta_t-\eta\xi_t$ | Worked Example 6.21's theoretical column | derived in Exercise B4 of this chapter |

## 6.11 Chapter summary

1. **Elementary conditioning is not enough.** $\mathbb{P}(A\mid B)$ needs $\mathbb{P}(B)>0$, but the objects ML estimates — $\mathbb{E}[Y\mid X=x]$, $\mathbb{P}(Y=1\mid X=x)$ — condition on events of probability zero whenever the features are continuous.
2. **Condition on a $\sigma$-algebra, not an event.** $\mathbb{E}[X\mid\mathcal{G}]$ is defined by three requirements: $\mathcal{G}$-measurable, integrable, and matching $X$'s integral on every $G\in\mathcal{G}$. The output is a random variable, not a number.
3. **Existence is Radon–Nikodym.** Define $\nu(G)=\int_GX\,d\mathbb{P}$ on $\mathcal{G}$, note $\nu\ll\mathbb{P}|_{\mathcal{G}}$, and take the derivative. Uniqueness is only almost sure, which is why every statement in the theory carries "a.s." and why pointwise values $\mathbb{E}[Y\mid X=x]$ are defined only for $\mathbb{P}_X$-a.e. $x$.
4. **On a finite partition it is just averaging over atoms** (Proposition 6.4, Figure 6.1). Everything else is a generalisation of that sentence.
5. **Doob–Dynkin turns $\sigma(X)$-measurability into "a function of $X$".** That is what makes the regression function $f^\star$ a bona fide measurable function and $f^\star(X)$ a random variable.
6. **In $L^2$, conditional expectation is orthogonal projection** onto the closed subspace of $\mathcal{G}$-measurable functions. Hence: *regression is projection*, the Bayes-optimal predictor under squared loss is the conditional mean, and the excess risk of any predictor is its squared $L^2(\mathbb{P}_X)$ distance to $f^\star$.
7. **The loss you choose selects the posterior summary you estimate:** squared loss $\to$ conditional mean, absolute loss $\to$ conditional median, pinball loss $\to$ conditional quantile, 0–1 loss $\to$ conditional mode, and log loss (cross-entropy) $\to$ the *entire* conditional distribution, with excess risk equal to the average KL divergence to the truth. Disagreements between MSE-trained and MAE-evaluated models are not bugs, and a cross-entropy-trained classifier is, at its optimum, a calibrated estimator of $p(y\mid x)$.
8. **Seven properties do all the work:** linearity, monotonicity, tower, pull-out ("taking out what is known"), independence kills conditioning, irrelevance of independent information (Theorem 6.8(f), whose freezing-lemma corollary — freeze what you know, average what you don't — is the step behind both the Gaussian kernel of §6.6 and the bias–variance proof), and conditional Jensen. The tower property holds because a coarse event is also a fine event; pull-out is proved by the standard ladder; (f) is a $\pi$–$\lambda$ argument and needs independence from the *pair*, not from $X$ alone.
9. **The law of total variance splits uncertainty into unexplained plus explained**, and the bias–variance decomposition is a corollary obtained by conditioning first on the training set and then on the test point. The noise term is $\mathbb{E}[\operatorname{Var}(Y\mid X)]$ — the Bayes risk — and no model reduces it.
10. **Regular conditional distributions exist on Polish spaces**, which legitimises writing $p(y\mid x)$; the density formula $p(y\mid x)=p(x,y)/p_X(x)$ is a theorem (Proposition 6.17), valid a.e., not a definition.
11. **The Borel paradox is real and computable:** conditioning on $\{X=Y\}$ via $X-Y$ gives $\mathbb{E}[Y^2]=0.5$; via $X/Y$ it gives $1$. Always name the conditioning random variable.
12. **A martingale is a process whose conditional expectation given the past is its present value.** Doob martingales $\mathbb{E}[X\mid\mathcal{F}_n]$ arise whenever a fixed quantity is revealed gradually; likelihood ratios are martingales under the null; convex functions of martingales are submartingales.
13. **No non-anticipating strategy beats a fair game** (martingale transforms), and optional stopping preserves the mean *only* under a boundedness hypothesis. The doubling strategy wins with probability one and still has mean zero once truncated — the missing hypothesis is where the risk hides.
14. **Doob's maximal inequality upgrades pointwise tail bounds to uniform-over-time ones** at no extra cost, giving Ville's inequality and anytime-valid testing; the martingale convergence theorem gives a.s. limits but not convergence of expectations without uniform integrability. Convergence needs two separate hypotheses: an $L^1$ bound gives *almost-sure* convergence, while passing to expectations additionally requires uniform integrability — which the Doob martingale of a fixed integrable $X$ always has (Lévy's upward theorem, so $\mathbb{E}[X\mid\mathcal{F}_n]\to\mathbb{E}[X\mid\mathcal{F}_\infty]$), and which the likelihood ratio never has.
15. **Azuma–Hoeffding drops independence**, requiring only conditionally centred, conditionally bounded increments; applying it to a Doob martingale proves McDiarmid, discharging Chapter 4's debt, and it is the right tool for adaptively collected data.
16. **SGD converges because a shifted loss along the trajectory is a supermartingale.** Under $L$-smoothness, conditional unbiasedness, bounded conditional variance and Robbins–Monro steps, $\min_{t<T}\mathbb{E}\|\nabla F(\theta_t)\|^2\to0$. Constant steps equilibrate at a noise floor $\propto\eta$; the proof breaks in identifiable ways under unconditional-only unbiasedness, heavy tails, gradient-dependent step sizes, and correlated minibatches.

### Callbacks

This chapter discharged the largest promise Chapter 1 made: that a $\sigma$-algebra is information, and that $\mathbb{E}[g_t\mid\mathcal{F}_t]=\nabla F(\theta_t)$ would eventually mean something. It also completed Chapter 1's Exercise C1 in full, including part (d)'s question about unconditional unbiasedness, which turns out to have a one-line answer (pull-out fails) and a concrete counterexample in which every run converges to the wrong point. Chapter 1's running examples were all reused and extended: the three coin flips became a filtration and a Doob martingale; the two dice showed that $\mathbb{E}[Y\mid X]=X+3.5$ is non-constant, settling the dependence question that a single factorised pair could not; $A=\max(0,Z)$ was decomposed by conditioning on its sign, with the tower property reproducing $\mathbb{E}[A]=1/\sqrt{2\pi}$ exactly. Chapter 2's Radon–Nikodym theorem, introduced there to unify densities, turned out to be the engine of existence for conditional expectation; Chapter 2's dominated convergence theorem reappeared as the hypothesis whose absence makes the doubling strategy "work"; Chapter 2's Cauchy example returned as the heavy-tailed gradient noise that breaks SGD. Chapter 4's McDiarmid inequality, stated there without proof, is now proved (Corollary 6.29). Chapter 5's Gaussian conditioning formula was re-derived as a regular conditional distribution and used to expose the Borel paradox. Chapter 2's KL divergence reappeared with a new job: it is exactly the excess risk of cross-entropy training (Proposition 6.7(d)), which is why 'minimise cross-entropy' and 'estimate the conditional distribution' are the same instruction. Chapter 1's insistence that countability is load-bearing was needed four separate times: in conditional Jensen, in the construction of regular conditional distributions, in the upcrossing argument, and in the measurability of suprema in McDiarmid's proof.

### Before you continue

1. State the three defining conditions of $\mathbb{E}[X\mid\mathcal{G}]$ without looking, and say which one forces "you may only use information in $\mathcal{G}$".
2. Why is conditional expectation unique only up to null sets, and what practical consequence does that have for the phrase "the model's prediction at $x_0$"?
3. Give the two-sentence argument that the conditional mean minimises squared error, naming the geometric fact you are using.
4. Write the law of total variance and say which term is the Bayes risk in a regression problem.
5. What extra property does a *regular* conditional distribution have that the family $\{\mathbb{P}(Y\in B\mid X)\}_B$ does not automatically have, and why is that property not free?
6. Define a stopping time, and explain why "the epoch with the lowest validation loss, chosen after training" is not one.
7. State optional stopping with all three sufficient hypotheses, then explain in one sentence which one the doubling strategy violates.
8. In the SGD proof, at which single line is assumption (A3) used, and what would break if it held only unconditionally?
9. Why does constant-step SGD not converge, and what does the theorem's bound converge to instead?

### Looking ahead to Chapter 7

Chapter 7 turns from conditioning to limits: the asymptotic toolkit. We will first sort out the *modes of convergence* — almost surely, in probability, in $L^p$, in distribution — and prove which implies which (the martingale convergence theorem you just met is an a.s. statement, and Worked Example 6.16 already showed why a.s. convergence need not give convergence of means). Then come the limit theorems: the weak and strong laws of large numbers, which say that empirical averages — and hence the empirical risk $\hat R_n$ of Chapter 4 — converge to their population counterparts; characteristic functions as the Fourier-analytic device that makes convergence in distribution tractable, together with Lévy's continuity theorem; the central limit theorem, which explains why $\sqrt n(\hat\theta_n-\theta)$ is asymptotically normal and why error bars have the shape they do; and the two workhorses that let you propagate those limits through transformations and combinations — the delta method and Slutsky's theorem. The payoff is practical: confidence intervals, standard errors, hypothesis tests, and a principled account of why the bootstrap works.

# Exercises

Difficulty: ★ routine consolidation · ★★ requires an idea · ★★★ a small research-grade argument.

## Part A — Theory

**A1 (★) Tower needs nesting.** Let $\Omega=\{1,2,3,4\}$ with the uniform measure and $X(\omega)=\omega$. Let $\mathcal{G}=\sigma(\{1,2\},\{3,4\})$ and $\mathcal{H}=\sigma(\{1,3\},\{2,4\})$. Compute $\mathbb{E}[X\mid\mathcal{G}]$, $\mathbb{E}[X\mid\mathcal{H}]$ and $\mathbb{E}\bigl[\mathbb{E}[X\mid\mathcal{G}]\mid\mathcal{H}\bigr]$. Which hypothesis of Theorem 6.8(c) fails, and what is the general moral?

**A2 (★) The other direction for two dice.** With $X$ the first die and $Y$ the sum: compute $\mathbb{E}[X\mid Y]$ (note the reversal), verify the tower property, and verify the law of total variance. Explain the symmetry you find between this and Worked Example 6.11.

**A3 (★★) Mean independence sits strictly between independence and uncorrelatedness.** Prove: (i) $X\perp Y\Rightarrow\mathbb{E}[Y\mid X]=\mathbb{E}[Y]$; (ii) $\mathbb{E}[Y\mid X]=\mathbb{E}[Y]\Rightarrow\operatorname{Cov}(X,Y)=0$ (assume square-integrability); (iii) neither converse holds — give a counterexample for each, and identify which of your examples is Worked Example 6.7.

**A4 (★★) Conditional Chebyshev.** Prove that for $X\in L^2$ and $t>0$,
$\mathbb{P}\bigl(|X-\mathbb{E}[X\mid\mathcal{G}]|\ge t\mid\mathcal{G}\bigr)\le\operatorname{Var}(X\mid\mathcal{G})/t^2$ a.s.
Then deduce an unconditional bound by the tower property, and explain why the conditional version is strictly more informative for a heteroscedastic regression problem.

**A5 (★★) When conditioning changes nothing.** Prove that $\mathbb{E}[X\mid\mathcal{G}]=X$ a.s. if and only if $X$ agrees a.s. with some $\mathcal{G}$-measurable random variable. (Careful: $X$ itself need not be $\mathcal{G}$-measurable on a null set. State where completeness of the $\sigma$-algebra is convenient.)

**A6 (★★) Biased gambler's ruin.** Let $S_n$ be a random walk with steps $+1$ w.p. $p\ne1/2$ and $-1$ w.p. $q=1-p$, started at $S_0=a\in\{0,\dots,N\}$, and $\tau$ the exit time from $(0,N)$. Show that $M_n:=(q/p)^{S_n}$ is a martingale, verify the hypotheses of optional stopping, and deduce $\mathbb{P}(S_\tau=N)=\frac{1-(q/p)^a}{1-(q/p)^N}$. Then find a second martingale that yields $\mathbb{E}[\tau]$.

**A7 (★★★) From upcrossings to convergence.** Take as given Doob's upcrossing inequality: for a supermartingale $(M_n)$ and $a<b$, $(b-a)\mathbb{E}[U_n[a,b]]\le\mathbb{E}[(M_n-a)^-]$. Prove Theorem 6.26(a): if $\sup_n\mathbb{E}|M_n|<\infty$ then $M_n$ converges a.s. to an integrable limit. Be explicit about where countability enters and where Fatou is used.

**A8 (★★★) Conditional expectation as an operator.** On $H=L^2(\Omega,\mathcal{F},\mathbb{P})$ define $T_{\mathcal{G}}X:=\mathbb{E}[X\mid\mathcal{G}]$. Prove: (i) $T_{\mathcal{G}}$ is linear and bounded with $\|T_{\mathcal{G}}\|=1$; (ii) $T_{\mathcal{G}}^2=T_{\mathcal{G}}$; (iii) $T_{\mathcal{G}}$ is self-adjoint, i.e. $\mathbb{E}[X\,T_{\mathcal{G}}Y]=\mathbb{E}[(T_{\mathcal{G}}X)Y]$; (iv) conclude from (ii)–(iii) that $T_{\mathcal{G}}$ is an orthogonal projection; (v) show $T_{\mathcal{H}}T_{\mathcal{G}}=T_{\mathcal{H}}$ when $\mathcal{H}\subseteq\mathcal{G}$, and use A1 to show $T_{\mathcal{G}}T_{\mathcal{H}}\ne T_{\mathcal{H}}$ in general — i.e. these projections do not commute.

**A9 (★★) Proper scoring rules.** A loss $\ell(q,y)$ for probability forecasts $q\in\Delta_{K-1}$ (the probability simplex) is a *strictly proper scoring rule* if, for every distribution $\eta$ on $\{1,\dots,K\}$, the map $q\mapsto\sum_k\eta_k\,\ell(q,k)$ is uniquely minimised at $q=\eta$. (i) Show the log loss $\ell(q,y)=-\log q_y$ is strictly proper (Proposition 6.7(d)). (ii) Show the Brier score $\ell(q,y)=\sum_k(q_k-\mathbf{1}\{y=k\})^2$ is strictly proper. (iii) Show the 0–1 loss on the induced classifier, $\ell(q,y)=\mathbf{1}\{\arg\max_kq_k\ne y\}$, is *not* strictly proper, and explain in one sentence what this means for reading a model's reported accuracy as evidence about its probability estimates.

## Part B — Computational labs

Report seeds and sample sizes. Where a theoretical value is available, print it next to the empirical one.

**B1 (★) Tower and pull-out on a table.** Take $X\in\{0,1\}$, $Y\in\{0,1,2\}$ with joint pmf
$p(0,0)=0.10$, $p(0,1)=0.20$, $p(0,2)=0.10$, $p(1,0)=0.30$, $p(1,1)=0.20$, $p(1,2)=0.10$.
(a) Compute $\mathbb{E}[Y\mid X]$ exactly and verify $\mathbb{E}[\mathbb{E}[Y\mid X]]=\mathbb{E}[Y]$.
(b) With $Z=\mathbf{1}\{X=1\}$, verify $\mathbb{E}[ZY\mid X]=Z\,\mathbb{E}[Y\mid X]$ pointwise.
(c) Verify the law of total variance numerically.
(d) Repeat (a)–(c) by Monte Carlo with $10^6$ draws and report the agreement to four decimals.

**B2 (★★) The projection is a minimum.** With $X\sim N(0,1)$ and $Y=X^2+0.5\varepsilon$, $\varepsilon\sim N(0,1)$: estimate the risk of $f^\star(x)=x^2$ and of $f^\star+c\,h$ for $h\in\{1,x,\sin3x,x^3\}$ and $c\in\{0.05,0.1,0.3\}$. Check the excess against the prediction $c^2\mathbb{E}[h(X)^2]$. Then estimate the risk of the best *linear* predictor and explain the factor-of-nine gap.

**B3 (★★) Doubling strategy.** Simulate the doubling strategy of Worked Example 6.18. (a) Confirm $M_\tau=1$ always and estimate $\mathbb{E}[\tau]$. (b) For caps $n\in\{5,10,20\}$ estimate $\mathbb{E}[M_{\tau\wedge n}]$ and compare to the theoretical $0$; report how many simulated paths hit the cap. (c) Explain why your estimate at $n=20$ is positive even with $10^6$ paths, and compute how many paths you would need for a standard error below $0.01$.

**B4 (★★★) Step-size schedules.** On $F(\theta)=\frac12\theta^\top A\theta$ with $A=\operatorname{diag}(1,10)$, $\sigma^2=2$ Gaussian gradient noise, run SGD from $\theta_0=(2,2)$ with $\eta$ constant, $\eta_t=\min(1/L,\,c/t)$, and $\eta_t=c/\sqrt t$ (with $c\le1/L$ so the cap never binds in the third case). Before running anything, check each schedule against every clause of (A5), including $\eta_t\le1/L$ — one of the "obvious" schedules fails it for small $t$. (a) Plot mean $F(\theta_t)$ on log–log axes over $T=10^5$. (b) Derive the stationary variance for a constant step: show that for a one-dimensional recursion $\theta_{t+1}=(1-\eta a)\theta_t-\eta\xi_t$ with $|1-\eta a|<1$ and $\operatorname{Var}(\xi)=1$, $\operatorname{Var}(\theta_\infty)=\frac{\eta^2}{1-(1-\eta a)^2}$, and hence $\mathbb{E}[F(\theta_\infty)]=\frac12\sum_ia_i\frac{\eta^2}{1-(1-\eta a_i)^2}$. Compare with simulation for $\eta\in\{0.01,0.02,0.05\}$. (c) Which schedules satisfy (A5)? Which satisfy the weaker condition of Theorem 6.30? Does the empirical behaviour track the distinction?

**B5 (★★) Heavy tails and clipping.** Repeat B4 with standard Cauchy gradient noise. (a) Report median, mean and max of $F(\theta_T)$ over 50 seeds and explain why you must report the median. (b) Add gradient clipping at threshold $c\in\{1,10\}$ and show it restores convergence. (c) Argue which of (A3), (A4) clipping repairs and which it breaks, and design a diagnostic that would distinguish "heavy-tailed noise" from "learning rate too large" in a real training run.

**B6 (★★) Bias–variance by simulation.** Reproduce Worked Example 6.12 and extend it: (a) add ridge regularisation $\lambda\in\{0,10^{-4},10^{-2}\}$ at degree $9$ and show the effect on each term; (b) push the degree past the interpolation threshold ($d\ge n$) using the minimum-norm least-squares solution (`numpy.linalg.lstsq`) and plot total error versus $d$ — comment on whether you see double descent and on how sensitive it is to the design.

## Part C — Paper-style derivations

**C1 (★★★) Existence of conditional expectation.** Write a self-contained proof of Theorem 6.2 for $X\in L^1$. Requirements: (i) state precisely the version of the Radon–Nikodym theorem you invoke, including the $\sigma$-finiteness hypothesis, and verify it for the pair $(\nu,\mathbb{P}|_{\mathcal{G}})$; (ii) explain why the resulting density is automatically $\mathcal{G}$-measurable and why this fails if one takes the derivative on $\mathcal{F}$ instead; (iii) handle signed $X$; (iv) prove a.s. uniqueness, and exhibit two distinct versions of $\mathbb{E}[Y\mid X]$ for a continuous $X$ to show that uniqueness cannot be improved.

**C2 (★★) Total variance and bias–variance.** (a) Prove the law of total variance, being explicit about which conditional-expectation properties each step uses. (b) Derive the bias–variance decomposition for prediction error from it, stating the independence assumption between training and test draws and where it is used. (c) State clearly which term is the Bayes risk, and give one sentence on why the decomposition is exact for squared loss but not for 0–1 loss.

**C3 (★★★) An SGD convergence lemma, appendix style.** Write the complete convergence proof for SGD as it would appear in a paper's appendix: number the assumptions (A1)–(A5), state the lemma with explicit constants, and give the proof in numbered steps. Then add a remark identifying, for each of the following, which numbered assumption it violates and what the observable symptom would be: (a) heavy-tailed gradient noise; (b) step sizes computed from the current gradient (Adam-style); (c) minibatches sampled without replacement within an epoch; (d) a fixed augmentation seed per example. Finally, state and prove the constant-step corollary: with $\eta_t\equiv\eta\le1/L$,
$$
\frac1T\sum_{t<T}\mathbb{E}\|\nabla F(\theta_t)\|^2\le\frac{2(F(\theta_0)-F_\star)}{\eta T}+L\eta\sigma^2 ,
$$
and interpret the two terms.

# Appendix A — Self-check answers

**Self-check 6.1.**
1. A singleton $\{x\}$ has $\lambda_d(\{x\})=0$, and the integral of any function over a Lebesgue-null set is $0$ (Ch. 2); so $\mathbb{P}(X=x)=\int_{\{x\}}p\,d\lambda_d=0$.
2. Numerator $\mathbb{P}(Y\le0.1,|X-0.5|\le h)=2\cdot2h\cdot0.1=0.4h$; denominator $\int_{0.5-h}^{0.5+h}2x\,dx=2h\cdot 2\cdot 0.5=2h$. Ratio $0.2=0.1/0.5$ ✓ (uniform on $(0,0.5)$).
3. The limit may fail to exist; it depends on the shrinking scheme (Borel paradox); and it produces one number at a time, with no guarantee that $x\mapsto$ answer is measurable.
4. $\mathbb{E}[Z\mid A=0]$ is legitimate: $\mathbb{P}(A=0)=1/2>0$, and it equals $\mathbb{E}[Z\mid Z\le0]=-\sqrt{2/\pi}\approx-0.797885$. $\mathbb{E}[Z\mid A=1]$ is not: $\mathbb{P}(A=1)=0$.

**Self-check 6.2.**
1. (CE1) and (CE2) hold — constants are measurable with respect to every $\sigma$-algebra. (CE3) fails: take $G=\{X>\mathbb{E}[X]\}\in\sigma(X)$, which has positive probability when $X$ is non-constant; then $\int_GX\,d\mathbb{P}>\mathbb{E}[X]\mathbb{P}(G)=\int_G\mathbb{E}[X]d\mathbb{P}$.
2. The atom is $\{THH,THT\}$ with probability $1/4$; $\int_{\text{atom}}X\,d\mathbb{P}=\frac18(2+1)=\frac38$; dividing by $\frac14$ gives $1.5$ ✓.
3. $\mathcal{G}\subseteq\mathcal{F}$ is used so that $\int_GX\,d\mathbb{P}$ makes sense for $G\in\mathcal{G}$ (we need $G\in\mathcal{F}$) and so that $\mathbb{P}|_{\mathcal{G}}$ is defined. $\sigma$-finiteness is needed to apply Radon–Nikodym; $\mathbb{P}|_{\mathcal{G}}$ is a probability measure, hence finite, hence $\sigma$-finite.
4. True *provided* $Y'$ is still $\mathcal{G}$-measurable. Modifying on a null set does not change any integral, so (CE3) survives; but if the null set is not in $\mathcal{G}$, the modified function may fail (CE1). If $(\Omega,\mathcal{G},\mathbb{P})$ is complete, the caveat disappears.

**Self-check 6.3.**
1. $L^2(\Omega,\mathcal{G},\mathbb{P})$ is complete (Riesz–Fischer), and a complete subspace of a metric space is closed. The argument would not deliver a *projection* in $L^1$: $L^1$ has no inner product, so there is no orthogonality and no projection theorem (the nearest point in $L^1$ need not be unique — it is the conditional median, not the mean).
2. Zero: $\operatorname{Var}(Y\mid X)=0$ a.s.
3. $Y\mid X=x$ uniform on $\{0,1\}$: every $m\in[0,1]$ is a median and all minimise $\mathbb{E}|Y-a|$. No contradiction — Proposition 6.7 identifies the minimiser *set* with the median set.
4. Diagnosis: MSE estimates the conditional mean, which lies above the median for right-skewed targets. Fix: train with absolute (or pinball) loss if a median/quantile is what you want to report.

**Self-check 6.4.**
1. $X$ is $\mathcal{G}$-measurable, so pull-out with $Z=X$ applied to the constant random variable $1$ gives $\mathbb{E}[X\cdot1\mid\mathcal{G}]=X\,\mathbb{E}[1\mid\mathcal{G}]=X$.
2. The hypothesis $ZX\in L^1$ — indeed for a Cauchy $X$ even $X\in L^1$ fails, so $\mathbb{E}[X\mid\mathcal{G}]$ is not defined to begin with.
3. $\mathbb{E}[X-\mathbb{E}[X\mid\mathcal{G}]]=0$ by the tower property, so the covariance is $\mathbb{E}[(X-\mathbb{E}[X\mid\mathcal{G}])V]=\mathbb{E}\bigl[V\,\mathbb{E}[X-\mathbb{E}[X\mid\mathcal{G}]\mid\mathcal{G}]\bigr]=0$, using tower then pull-out.
4. Conditional Jensen with $\varphi(x)=x^2$ then tower: $\mathbb{E}\bigl[(\mathbb{E}[X\mid\mathcal{G}])^2\bigr]\le\mathbb{E}\bigl[\mathbb{E}[X^2\mid\mathcal{G}]\bigr]=\mathbb{E}[X^2]$.

**Self-check 6.5.**
1. $\operatorname{Var}(\mathbb{E}[X\mid\mathcal{G}])=\operatorname{Var}(X)-\mathbb{E}[\operatorname{Var}(X\mid\mathcal{G})]\le\operatorname{Var}(X)$; equality iff $\operatorname{Var}(X\mid\mathcal{G})=0$ a.s., i.e. iff $X$ is a.s. $\mathcal{G}$-measurable.
2. Two thirds. $\mathbb{E}[X\mid\mathcal{F}_2]$ takes values $2.5,1.5,1.5,0.5$ with equal probabilities; mean $1.5$; variance $\frac14(1+0+0+1)=0.5$, and $0.5/0.75=2/3$ ✓ — two of three independent flips.
3. Bagging targets the variance term (averaging over resampled training sets). Early stopping also mainly reduces variance, at the price of some bias: it restricts how far the fit can move from its initialisation.
4. $Y$ agrees a.s. with a measurable function of $X$, namely $f^\star(X)$; the Bayes risk is $0$ and perfect prediction is possible in principle.

**Self-check 6.6.**
1. Because the null set on which the defining property can fail depends on $B$, and there are uncountably many $B$; countable additivity for *all* countable families simultaneously is not implied.
2. $\{p_X=0\}$ has $\mathbb{P}_X$-probability $0$ and $\{p_X=\infty\}$ is Lebesgue-null, so any two choices of $q_0$ give kernels that agree for $\mathbb{P}_X$-a.e. $x$ — which is all the uniqueness statement claims.
3. $p_Y(y)=\int_y^12\,dx=2(1-y)$, so $X\mid Y=y\sim\mathrm{Uniform}(y,1)$, $\mathbb{E}[X\mid Y=y]=(1+y)/2$ and $\operatorname{Var}(X\mid Y=y)=(1-y)^2/12$. Check: $\mathbb{E}[X]=\mathbb{E}[(1+Y)/2]=(1+\frac13)/2=\frac23$ ✓.
4. "A conditional law is attached to a conditioning *random variable*, not to an event: $\{X=Y\}$ described via $X-Y$ and via $X/Y$ are the same event but different $\sigma$-algebras, and they give different — both correct — answers."

**Self-check 6.7.**
1. No. Deciding at time $n$ whether $M_n$ equals $\max_{k\le100}M_k$ requires knowing the path up to time $100$, so $\{\tau\le n\}\notin\mathcal{F}_n$ in general.
2. Induction on $m$: $\mathbb{E}[M_m\mid\mathcal{F}_n]=\mathbb{E}\bigl[\mathbb{E}[M_m\mid\mathcal{F}_{m-1}]\mid\mathcal{F}_n\bigr]=\mathbb{E}[M_{m-1}\mid\mathcal{F}_n]$ by the tower property (legal since $\mathcal{F}_n\subseteq\mathcal{F}_{m-1}$), and iterate down to $M_n$.
3. Win at round 1, 2 or 3 (probabilities $\frac12,\frac14,\frac18$) gives $+1$; no win in three rounds (probability $\frac18$) gives $-(1+2+4)=-7$. Mean $=\frac78-\frac78=0$ ✓.
4. Under $p$, $\mathbb{E}_p[q(X)/p(X)]=\int q=1$, giving a martingale. Under $q$, $\mathbb{E}_q[q(X)/p(X)]=\int q^2/p=1+\chi^2(Q\|P)\ge1$ with equality iff $P=Q$; so under $q$ it is a submartingale (it drifts up — as it should, since the data now favour $q$).

**Self-check 6.8.**
1. Non-negativity is used in the last step $\mathbb{E}[X_n\mathbf{1}_A]\le\mathbb{E}[X_n]$. Counterexample without it: the symmetric random walk $M_n$ has $\mathbb{E}[M_n]=0$ but $\mathbb{P}(\max_{k\le n}M_k\ge\lambda)>0$, so $\lambda\mathbb{P}(\cdot)\le\mathbb{E}[M_n]=0$ is false.
2. Continuity from below, Proposition 1.8(3), applied to the increasing events $\{\max_{n\le N}L_n\ge1/\alpha\}$.
3. It lets the *location* of the increment's range depend on the past; only the *width* $c_k$ must be a constant. This is what makes McDiarmid's sharper constant available ($e^{-2t^2/\sum c_k^2}$ rather than $e^{-t^2/2\sum b_k^2}$) and what makes the inequality usable when the conditional mean shifts around.
4. Drops: independence of the increments, and any knowledge of their distributions. Keeps: bounded increments (a bounded conditional range), and adds conditional centring — the martingale property itself.
5. Uniform integrability requires $\sup_n\mathbb{E}[L_n\mathbf{1}\{L_n>K\}]\to0$ as $K\to\infty$. But $\mathbb{E}[L_n]=1$ for every $n$ while the median at $n=50$ is $1.9\times10^{-3}$ and the observed maximum is $1.36\times10^4$: essentially the entire unit of mass has migrated into the far tail. Formally, for any fixed $K$ one can choose $n$ large enough that $\mathbb{P}(L_n\le K)$ is as close to $1$ as desired while $\mathbb{E}[L_n\mathbf{1}\{L_n\le K\}]\to0$, forcing $\mathbb{E}[L_n\mathbf{1}\{L_n>K\}]\to1$. So the supremum over $n$ does not vanish. This blocks Theorem 6.26(b) — no $L^1$ convergence, hence no convergence of expectations — while 6.26(a) still applies and does give $L_n\to0$ a.s.

**Self-check 6.9.**
1. Pull-out (Theorem 6.8(d)), because $\nabla F(\theta_t)$ is $\mathcal{F}_t$-measurable, followed by assumption (A3).
2. Because only a weighted *average* of $\mathbb{E}\|\nabla F(\theta_t)\|^2$ is controlled. Nothing forces the sequence to be monotone, so the final iterate need not be the best one; with noise it typically is not.
3. It converges to $L\eta\sigma^2$ (see Exercise C3's corollary): a floor proportional to $\eta$. The experiment agrees in scaling — measured stationary $\mathbb{E}[F]$ was $0.00503$, $0.01087$, $0.02933$ for $\eta=0.01,0.02,0.05$, close to linear in $\eta$ and matching the exact theory $0.00514$, $0.01061$, $0.02949$.
4. (a) (A3) — the conditional mean of the minibatch gradient is not $\nabla F$; (b) (A5) — $\eta_t$ is not predictable; (c) (A3) again — the noise is a fixed function of the example, so it does not re-randomise and its conditional mean is not zero; (d) (A4) — infinite conditional variance.

# Solutions to Exercises

## Part A

**A1.** *Thinking:* the tower property is not a symmetric statement — it needs one $\sigma$-algebra inside the other. Here neither contains the other, so we should expect it to fail, and the computation shows how.

$\mathbb{E}[X\mid\mathcal{G}]$ averages over $\{1,2\}$ and $\{3,4\}$: it equals $1.5$ on $\{1,2\}$ and $3.5$ on $\{3,4\}$. $\mathbb{E}[X\mid\mathcal{H}]$ averages over $\{1,3\}$ and $\{2,4\}$: $2$ on $\{1,3\}$, $3$ on $\{2,4\}$. Now condition the first on $\mathcal{H}$: on $\{1,3\}$ we average $1.5$ (at $\omega=1$) and $3.5$ (at $\omega=3$) to get $2.5$; on $\{2,4\}$, likewise $2.5$. So
$$
\mathbb{E}\bigl[\mathbb{E}[X\mid\mathcal{G}]\mid\mathcal{H}\bigr]\equiv2.5=\mathbb{E}[X]\ \ne\ \mathbb{E}[X\mid\mathcal{H}].
$$
The failed hypothesis is $\mathcal{H}\subseteq\mathcal{G}$. Moral: iterated conditioning collapses to the coarser $\sigma$-algebra only along a *chain*. Two non-nested information sets can jointly destroy information — here $\mathcal{G}$ records the pair $\{$low, high$\}$ and $\mathcal{H}$ records parity, and composing them retains nothing. In ML terms: summarising by feature $A$ and then summarising that summary by feature $B$ is not the same as summarising by $B$, which is why sequential feature aggregation pipelines can lose signal that either feature alone would have kept.

**A2.** By symmetry of the two dice, given the sum $Y=y$ the two dice are exchangeable, so $\mathbb{E}[X\mid Y]=Y/2$. (Formally: $\mathbb{E}[D_1\mid Y]=\mathbb{E}[D_2\mid Y]$ because $(D_1,D_2)$ and $(D_2,D_1)$ have the same law and $Y$ is symmetric, and their sum is $Y$.) Check with $y=5$: the four equally likely pairs are $(1,4),(2,3),(3,2),(4,1)$, and the average first coordinate is $2.5=5/2$ ✓.

Tower: $\mathbb{E}[Y/2]=7/2=3.5=\mathbb{E}[X]$ ✓. Total variance: $\operatorname{Var}(X)=35/12=2.9167$; $\operatorname{Var}(\mathbb{E}[X\mid Y])=\operatorname{Var}(Y)/4=(35/6)/4=35/24=1.4583$; hence $\mathbb{E}[\operatorname{Var}(X\mid Y)]=35/12-35/24=35/24=1.4583$. Exactly half the variance of $X$ is explained by $Y$ — the mirror image of Worked Example 6.11, where exactly half the variance of $Y$ was explained by $X$. The symmetry is not an accident: $\operatorname{Var}(\mathbb{E}[X\mid Y])/\operatorname{Var}(X)=\operatorname{Var}(\mathbb{E}[Y\mid X])/\operatorname{Var}(Y)=1/2$ here because both conditional means are affine and the squared correlation $\rho^2=1/2$ governs both directions.

**A3.** (i) If $X\perp Y$ then $\sigma(Y)\perp\sigma(X)$, so Theorem 6.8(e) gives $\mathbb{E}[Y\mid X]=\mathbb{E}[Y]$.
(ii) $\operatorname{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}X\mathbb{E}Y=\mathbb{E}\bigl[X\,\mathbb{E}[Y\mid X]\bigr]-\mathbb{E}X\mathbb{E}Y$ (tower then pull-out) $=\mathbb{E}[Y]\mathbb{E}[X]-\mathbb{E}X\mathbb{E}Y=0$.
(iii) *Mean independence does not imply independence:* let $N\sim N(0,1)$ and let $\varepsilon$ be Rademacher ($\pm1$ with probability $\tfrac12$ each) independent of $N$. Set $X:=\varepsilon|N|$ and $Y:=|N|$. Then $X=\varepsilon Y$ with $\varepsilon\perp\sigma(Y)$, so by pull-out and independence $\mathbb{E}[X\mid Y]=Y\,\mathbb{E}[\varepsilon]=0=\mathbb{E}[X]$: $X$ is mean-independent of $Y$. But $|X|=Y$ exactly, so $X$ and $Y$ are about as dependent as two random variables can be — knowing $Y$ pins down $|X|$ completely. (Incidentally $X\sim N(0,1)$ as well, by symmetry of the normal.) *Uncorrelated does not imply mean independent:* Worked Example 6.7, $X\sim N(0,1)$, $Y=X^2+\varepsilon$: $\operatorname{Cov}(X,Y)=\mathbb{E}[X^3]=0$ but $\mathbb{E}[Y\mid X]=X^2\ne\mathbb{E}[Y]=1$. So independence $\Rightarrow$ mean independence $\Rightarrow$ uncorrelated, both strictly.

**A4.** Apply Markov's inequality to the conditional measure. Rigorously, let $R:=X-\mathbb{E}[X\mid\mathcal{G}]$ and $A:=\{|R|\ge t\}$. Then $t^2\mathbf{1}_A\le R^2$ pointwise, so by monotonicity and linearity of conditional expectation,
$$
t^2\,\mathbb{P}(A\mid\mathcal{G})=\mathbb{E}[t^2\mathbf{1}_A\mid\mathcal{G}]\le\mathbb{E}[R^2\mid\mathcal{G}]=\operatorname{Var}(X\mid\mathcal{G})\quad\text{a.s.}
$$
Taking expectations and using the tower property gives $\mathbb{P}(|R|\ge t)\le\mathbb{E}[\operatorname{Var}(X\mid\mathcal{G})]/t^2$. The conditional version is stronger for heteroscedastic problems because it gives a *different* bound in each region of feature space: where $\operatorname{Var}(Y\mid X=x)$ is small the prediction interval is narrow, and averaging that away — as the unconditional bound does — produces intervals that are simultaneously too wide in quiet regions and too narrow in noisy ones. This is exactly the argument for heteroscedastic regression heads that predict a variance alongside a mean.

**A5.** ($\Leftarrow$) Suppose $X=X'$ a.s. with $X'$ $\mathcal{G}$-measurable. Then $X'$ satisfies (CE1)–(CE3) for $X$ (integrals are unchanged by null modifications), so $\mathbb{E}[X\mid\mathcal{G}]=X'=X$ a.s.
($\Rightarrow$) If $\mathbb{E}[X\mid\mathcal{G}]=X$ a.s., then $X'=\mathbb{E}[X\mid\mathcal{G}]$ is a $\mathcal{G}$-measurable random variable agreeing with $X$ a.s.
The a.s. hedging is unavoidable: $X$ itself may fail to be $\mathcal{G}$-measurable on a null set (e.g. modify a $\mathcal{G}$-measurable function at one point of a continuous distribution). If $\mathcal{G}$ is $\mathbb{P}$-complete, every a.s. modification of a $\mathcal{G}$-measurable function is again $\mathcal{G}$-measurable and the statement simplifies to "$X$ is $\mathcal{G}$-measurable".

**A6.** *Thinking:* find a function $\phi$ with $\mathbb{E}[\phi(S_{n+1})\mid\mathcal{F}_n]=\phi(S_n)$; for a walk this means $p\phi(s+1)+q\phi(s-1)=\phi(s)$, and $\phi(s)=r^s$ works iff $pr+q/r=1$, i.e. $r=q/p$ (or $r=1$, the trivial solution).

With $r=q/p$: $\mathbb{E}[r^{S_{n+1}}\mid\mathcal{F}_n]=r^{S_n}(pr+qr^{-1})=r^{S_n}(q+p)=r^{S_n}$ ✓, using pull-out and independence of the next step. The stopped process takes values in $[\min(1,r^N),\max(1,r^N)]$, so it is uniformly bounded, and $\tau<\infty$ a.s. (the walk exits any window of width $N$ within $N$ consecutive same-direction steps, an event of probability $\ge\min(p,q)^N$ in each block, so $\tau$ has a geometric tail). Theorem 6.23(ii) applies:
$$
r^a=\mathbb{E}[r^{S_\tau}]=\mathbb{P}(S_\tau=N)r^N+\bigl(1-\mathbb{P}(S_\tau=N)\bigr)\implies\mathbb{P}(S_\tau=N)=\frac{1-r^a}{1-r^N}.
$$
For the duration use $M_n:=S_n-(p-q)n$, a martingale since $\mathbb{E}[S_{n+1}-S_n\mid\mathcal{F}_n]=p-q$. Increments are bounded by $1+|p-q|\le2$ and $\mathbb{E}\tau<\infty$ (geometric tail), so Theorem 6.23(iii) gives $a=\mathbb{E}[S_\tau]-(p-q)\mathbb{E}[\tau]$, i.e. $\mathbb{E}[\tau]=\bigl(N\,\mathbb{P}(S_\tau=N)-a\bigr)/(p-q)$.

**A7.** Fix rationals $a<b$ and let $U_\infty[a,b]:=\lim_nU_n[a,b]$ (monotone in $n$). By the upcrossing inequality and monotone convergence,
$$
(b-a)\mathbb{E}[U_\infty[a,b]]=\lim_n(b-a)\mathbb{E}[U_n[a,b]]\le\sup_n\mathbb{E}[(M_n-a)^-]\le|a|+\sup_n\mathbb{E}|M_n|<\infty,
$$
so $U_\infty[a,b]<\infty$ a.s.; call the exceptional set $N_{a,b}$. Let $N:=\bigcup_{a<b\text{ rational}}N_{a,b}$, a **countable** union of null sets, hence null. Off $N$: if $\liminf M_n<\limsup M_n$, pick rationals $a<b$ strictly between them; then the path crosses $[a,b]$ upward infinitely often, contradicting $U_\infty[a,b]<\infty$. Hence $\lim_nM_n$ exists in $[-\infty,\infty]$ a.s. Finally Fatou (Ch. 2) gives $\mathbb{E}|M_\infty|=\mathbb{E}[\liminf|M_n|]\le\liminf\mathbb{E}|M_n|\le\sup_n\mathbb{E}|M_n|<\infty$, so $M_\infty$ is a.s. finite and integrable. Countability is what lets us use one null set for all pairs at once; Fatou is what converts an $L^1$ bound along the sequence into integrability of the limit.

**A8.** (i) Linearity is Theorem 6.8(a); $\|T_{\mathcal{G}}X\|_2\le\|X\|_2$ is Corollary 6.10 with $p=2$; taking $X=1$ (which is $\mathcal{G}$-measurable) gives $T_{\mathcal{G}}X=X$, so $\|T_{\mathcal{G}}\|=1$.
(ii) $T_{\mathcal{G}}X$ is $\mathcal{G}$-measurable, so $T_{\mathcal{G}}(T_{\mathcal{G}}X)=T_{\mathcal{G}}X$ (Worked Example 6.4(ii)).
(iii) $\mathbb{E}[X\,T_{\mathcal{G}}Y]=\mathbb{E}\bigl[\mathbb{E}[X\,T_{\mathcal{G}}Y\mid\mathcal{G}]\bigr]=\mathbb{E}\bigl[(T_{\mathcal{G}}Y)(T_{\mathcal{G}}X)\bigr]$, by tower then pull-out; the last expression is symmetric in $X,Y$, so it also equals $\mathbb{E}[(T_{\mathcal{G}}X)Y]$.
(iv) A bounded, idempotent, self-adjoint operator on a Hilbert space is the orthogonal projection onto its range: for $X\in H$ write $X=T X+(I-T)X$; the second term is orthogonal to the range because $\langle (I-T)X,TY\rangle=\langle X,TY\rangle-\langle TX,TY\rangle=\langle X,TY\rangle-\langle X,T^2Y\rangle=0$.
(v) $T_{\mathcal{H}}T_{\mathcal{G}}=T_{\mathcal{H}}$ for $\mathcal{H}\subseteq\mathcal{G}$ is the tower property. A1 shows the identity fails without nesting: there $T_{\mathcal{H}}T_{\mathcal{G}}X\equiv2.5$ while $T_{\mathcal{H}}X$ takes values $2,3$. *A caution:* in A1's symmetric example the two compositions happen to be equal (both collapse to $\mathbb{E}[X]$), so that example does **not** prove non-commutativity. For genuine non-commutativity, weight the four points as $(0.4,0.1,0.2,0.3)$ and keep $X(\omega)=\omega$, $\mathcal{G}=\sigma(\{1,2\},\{3,4\})$, $\mathcal{H}=\sigma(\{1,3\},\{2,4\})$. Then (computed exactly)
$$
T_{\mathcal{H}}T_{\mathcal{G}}X=(2,3,2,3),\qquad T_{\mathcal{G}}T_{\mathcal{H}}X=(2.0333,2.0333,2.7667,2.7667),
$$
which differ, and neither equals $T_{\mathcal{H}}X=(1.6667,3.5,1.6667,3.5)$ nor $\mathbb{E}[X]=2.4$. Non-commuting projections are the reason alternating conditional-mean fitting (backfitting in additive models, or coordinate-wise imputation) has to be *iterated* rather than applied once.

**A9.** *Thinking:* in every case, condition on $X=x$ (tower property) so the problem becomes: given a fixed distribution $\eta$ over classes, which reported $q$ minimises the $\eta$-average loss? "Strictly proper" means honesty is the unique optimum.

(i) Done in the proof of Proposition 6.7(d): the excess $\eta$-average loss of $q$ over $\eta$ is $D_{\mathrm{KL}}(\eta\|q)$, which is $\ge0$ with equality iff $q=\eta$ by Gibbs' inequality.

(ii) The $\eta$-average Brier score is $\sum_k\mathbb{E}_\eta(q_k-\mathbf{1}\{Y=k\})^2=\sum_k\bigl[(q_k-\eta_k)^2+\eta_k(1-\eta_k)\bigr]$, by the elementary decomposition $\mathbb{E}[(c-Z)^2]=(c-\mathbb{E}Z)^2+\operatorname{Var}(Z)$ applied to $Z=\mathbf{1}\{Y=k\}$. The second sum does not depend on $q$; the first is uniquely minimised (to $0$) at $q=\eta$. So Brier is strictly proper, and its excess is the squared Euclidean distance $\|q-\eta\|^2$ — the same geometry as §6.3, now on the simplex.

(iii) The $\eta$-average 0–1 loss of $q$ depends on $q$ only through $\arg\max_kq_k$. Any $q$ whose largest coordinate sits at $\arg\max_k\eta_k$ achieves the minimum — e.g. with $\eta=(0.6,0.4)$, both $q=(0.6,0.4)$ and $q=(0.99,0.01)$ are optimal — so the minimiser is a large set, not the point $\eta$. Consequence: accuracy is invariant under wild distortions of the reported probabilities on the correct side of the decision boundary, so high accuracy is *no* evidence that the probabilities are calibrated; if you need calibrated $p(y\mid x)$, evaluate with a strictly proper score (log loss or Brier), not accuracy.

## Part B

**B1.** All quantities are exact rationals; computed values below were produced by direct enumeration.
(a) $\mathbb{P}(X=0)=0.4$, $\mathbb{P}(X=1)=0.6$; $\mathbb{E}[Y\mid X=0]=0.4/0.4=1$, $\mathbb{E}[Y\mid X=1]=0.4/0.6=0.666667$. Tower: $0.4(1)+0.6(0.666667)=0.8=\mathbb{E}[Y]$ ✓ (directly, $\mathbb{P}(Y=1)=0.4$, $\mathbb{P}(Y=2)=0.2$ give $0.8$).
(b) On $\{X=0\}$: $Z=0$ so both sides are $0$. On $\{X=1\}$: $Z=1$ so both sides are $\mathbb{E}[Y\mid X=1]=0.666667$.
(c) $\operatorname{Var}(Y)=1.2-0.64=0.56$; $\operatorname{Var}(Y\mid X=0)=1.5-1=0.5$, $\operatorname{Var}(Y\mid X=1)=1.0-0.444444=0.555556$, so $\mathbb{E}[\operatorname{Var}(Y\mid X)]=0.533333$ and $\operatorname{Var}(\mathbb{E}[Y\mid X])=0.4\cdot0.6\cdot(1-\tfrac23)^2=0.026667$. Sum $=0.560000$ ✓.
(d) With $10^6$ draws you should reproduce each number to about $\pm0.001$; the point of the exercise is to see that the Monte Carlo error, not the identity, is what moves.

**B2.** Excess risk must equal $c^2\mathbb{E}[h(X)^2]$ by Corollary 6.6. Measured ($n=2\times10^6$, seed 6; base MSE $0.24993$):

| $h$ | $\mathbb{E}[h(X)^2]$ | $c=0.1$ excess | $c=0.3$ excess |
|---|---|---|---|
| $1$ | 1 | 0.00995 (theory 0.01) | 0.08984 (0.09) |
| $x$ | 1 | 0.01001 (0.01) | 0.09006 (0.09) |
| $\sin3x$ | $(1-e^{-18})/2=0.5$ | 0.00511 (0.005) | 0.04535 (0.045) |
| $x^3$ | 15 | 0.14948 (0.15) | 1.34717 (1.35) |

(For $x^3$ at $c=0.05$: measured $0.03729$, theory $0.0375$.) The best linear predictor has risk $2.2502$ versus $0.2501$: the factor of nine is $\operatorname{Var}(Y)/\text{Bayes risk}=2.25/0.25$, and it arises because $\operatorname{Cov}(X,Y)=\mathbb{E}[X^3]=0$ forces the fitted line to be flat, so the linear model captures none of the $X^2$ signal. This is a *subspace* limitation, not a sample-size limitation.

**B3.** (a) $M_\tau=+1$ in $10^6/10^6$ plays; $\hat{\mathbb{E}}[\tau]=2.0002$ (seed 13), theory $2$.
(b) With caps: at $n=5$, $\hat{\mathbb{E}}[M_{\tau\wedge5}]=+0.0010$ with $31\,218$ paths hitting the cap (theory $\mathbb{P}=2^{-5}=0.031250$); at $n=10$, $+0.0159$ with $961$ cap hits (theory $0.000977$); at $n=20$ with $10^6$ paths, $+1.0000$ and *zero* cap hits.
(c) At $n=20$ the compensating loss is $-1\,048\,575$ with probability $2^{-20}\approx9.54\times10^{-7}$; in $10^6$ paths you expect about one, and with none you measure exactly $+1$. Running $2\times10^7$ paths (seed 23) found $13$ cap hits and gave $\hat{\mathbb{E}}[M_{\tau\wedge20}]=+0.31843$ — still far from $0$. The variance of a single play is $\operatorname{Var}=\mathbb{E}[M^2]=2^{-20}(2^{20}-1)^2+(1-2^{-20})\approx2^{20}$, so the standard error is $\approx2^{10}/\sqrt N$, and $\text{s.e.}<0.01$ needs $N>(1024/0.01)^2\approx10^{10}$ plays. **A quantity whose mean requires $10^{10}$ samples to estimate to two decimals is one you should reason about analytically, not empirically** — the practical lesson for any heavy-tailed estimator.

**B4.** (a) See Figure 6.6 and Worked Example 6.21.
(b) *Derivation.* For $\theta_{t+1}=(1-\eta a)\theta_t-\eta\xi_t$ with $\xi_t$ independent of $\theta_t$, mean $0$, variance $1$: $v_{t+1}=(1-\eta a)^2v_t+\eta^2$. If $|1-\eta a|<1$ this converges to the fixed point $v_\infty=\eta^2/(1-(1-\eta a)^2)$. Summing $\frac12a_iv_\infty^{(i)}$ over coordinates gives the stated formula. For $A=\operatorname{diag}(1,10)$, $\eta=0.05$: coordinate $1$ contributes $\frac12\cdot1\cdot\frac{0.0025}{1-0.9025}=0.012821$; coordinate $2$ contributes $\frac12\cdot10\cdot\frac{0.0025}{1-0.25}=0.016667$; total $0.029487$. Simulation (4000 seeds, $T=2\times10^4$, seed 77) gives $0.02933\pm0.00048$ ✓; at $\eta=0.01$ and $0.02$: measured $0.00503$, $0.01087$ versus theory $0.00514$, $0.01061$.
(c) Only the capped schedule $\eta_t=\min(1/L,c/t)$ satisfies (A5) as stated — note that the raw $c/t$ with $c=1$ violates the clause $\eta_t\le1/L$ for $t<cL$, and on this problem the first step is even expansive on the stiff coordinate ($1-\eta_1a_2=-9$); the cap is what restores the hypothesis, at no asymptotic cost. $\eta_t=c/\sqrt t$ satisfies the weaker sufficient condition of Theorem 6.30 because $(\sum_{t<T}\eta_t^2)/\sum_{t<T}\eta_t=O(\log T/\sqrt T)\to0$; constant $\eta$ satisfies neither and indeed plateaus. Empirically: $1/t$ decays like $1/t$, $1/\sqrt t$ decays but more slowly ($1.38\times10^{-4}$ at $T=10^5$), constant $\eta$ flat at $\approx0.025$–$0.029$. Theory and experiment agree on all three, including the negative result.

**B5.** (a) With standard Cauchy noise and $\eta_t=\min(0.1,1/t)$ ($T=2\times10^4$, seeds 0–49): median $F=1.09$, mean $1.04\times10^3$, max $5.07\times10^4$. You must report the median because the mean is dominated by one or two runs — the sample mean of a heavy-tailed quantity is itself heavy-tailed and is not a summary of typical behaviour.
(b) Clipping the gradient to norm $c$ restores convergence dramatically (same seeds and horizon):

| clipping | median $F(\theta_T)$ | mean | max |
|---|---|---|---|
| none | $1.09$ | $1.04\times10^{3}$ | $5.07\times10^{4}$ |
| $c=10$ | $4.04\times10^{-4}$ | $5.63\times10^{-4}$ | $2.14\times10^{-3}$ |
| $c=1$ | $1.53\times10^{-2}$ | $1.55\times10^{-2}$ | $2.78\times10^{-2}$ |

Note $c=1$ is *worse* than $c=10$: over-aggressive clipping bites into legitimate large gradients early in training and slows progress. That trade-off is the bias that clipping introduces.
(c) Clipping repairs (A4) — the clipped gradient has norm $\le c$, so its conditional variance is at most $c^2$ — and breaks (A3), since $\mathbb{E}[\text{clip}(g_t)\mid\mathcal{F}_t]\ne\nabla F(\theta_t)$ in general (clipping shrinks toward the origin asymmetrically when the distribution is skewed). A useful diagnostic: log the *empirical distribution* of $\|g_t\|$ per step, not just its mean. Heavy tails show up as a stable median with a max that grows like a power of the number of steps (and a Hill-type tail-index estimate below $2$); "learning rate too large" shows up as the whole distribution — median included — drifting upward together with the loss.

**B6.** (a) Ridge at degree $9$ ($R=2000$ replications, seed 2024, basis: powers of $2x-1$): $\lambda=0$ gives bias$^2<0.0001$, variance $0.0425$, total $0.1325$; $\lambda=10^{-4}$ gives $<0.0001$, $0.0308$, $0.1208$; $\lambda=10^{-2}$ gives $0.0021$, $0.0214$, $0.1135$. Regularisation trades a small bias increase for a larger variance decrease — visible in the columns, which is the whole argument for ridge in one table.
(b) Past the interpolation threshold, *the answer depends on the basis*, which is itself the lesson. With a raw monomial basis the minimum-norm interpolant is numerically catastrophic ($d=25$: total error $1.7\times10^5$) and no second descent appears. With an orthonormal Legendre basis on $[-1,1]$ (600 replications, seed 7, $n=20$):

| $d$ | 3 | 9 | 15 | 19 | 20 | 25 | 40 | 80 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| total error | 0.1085 | 0.1311 | 0.898 | $1.56\times10^4$ | $1.53\times10^4$ | 1.914 | **0.1772** | 0.4898 | 0.5480 |

That is a textbook double-descent curve: a classical U up to $d\approx n$, a spike at the interpolation threshold $d=n=20$, and a second descent to $0.1772$ at $d=40$ — here still worse than the classical optimum $0.1085$ at $d=3$, and rising again at very large $d$ as the min-norm interpolant becomes over-smoothed. The honest summary: double descent is real and reproducible, but its location and depth depend on the parametrisation, because "minimum norm" means minimum norm *in the chosen coordinates*.

## Part C

**C1.** *Strategy.* Conditional expectation is a density in disguise. The map $G\mapsto\int_GX\,d\mathbb{P}$ is a signed measure on $\mathcal{G}$ that vanishes wherever $\mathbb{P}$ does; Radon–Nikodym converts "vanishes wherever $\mathbb{P}$ does" into "is an integral against a density", and that density is the object we want. Everything else is bookkeeping.

*(i) The theorem invoked.* **Radon–Nikodym (Ch. 2).** Let $(\Omega,\mathcal{G})$ be a measurable space, $\mu$ a $\sigma$-finite non-negative measure on $\mathcal{G}$, and $\nu$ a finite signed measure on $\mathcal{G}$ with $\nu\ll\mu$ (i.e. $\mu(G)=0\Rightarrow\nu(G)=0$). Then there is a $\mu$-integrable, $\mathcal{G}$-measurable $f$ with $\nu(G)=\int_Gf\,d\mu$ for all $G\in\mathcal{G}$, unique up to $\mu$-null sets.

Take $\mu:=\mathbb{P}|_{\mathcal{G}}$, the restriction of $\mathbb{P}$ to $\mathcal{G}$. It is a probability measure on $\mathcal{G}$ (countable additivity is inherited; $\mu(\Omega)=1$), hence finite, hence $\sigma$-finite — the hypothesis is free here, which is exactly why the construction works for arbitrary $\mathcal{G}$ and would need care if $\mathbb{P}$ were replaced by, say, Lebesgue measure on $\mathbb{R}$.

For $X\ge0$ in $L^1$ define $\nu(G):=\int_GX\,d\mathbb{P}$ for $G\in\mathcal{G}$. Then $\nu$ is a finite measure on $\mathcal{G}$: $\nu(\Omega)=\mathbb{E}[X]<\infty$, and countable additivity follows from the monotone convergence theorem applied to $\sum_k\mathbf{1}_{G_k}X$ for disjoint $G_k$. Absolute continuity: if $G\in\mathcal{G}$ has $\mu(G)=\mathbb{P}(G)=0$ then $\mathbf{1}_GX=0$ a.s., so $\nu(G)=0$. Radon–Nikodym supplies $Y:=d\nu/d\mu$.

*(ii) Why the density is $\mathcal{G}$-measurable.* Because Radon–Nikodym is applied *on the measurable space $(\Omega,\mathcal{G})$*: it returns a function measurable with respect to the $\sigma$-algebra on which the two measures live. This is the entire content of property (CE1), and it is obtained for free rather than imposed. If instead one formed $\nu$ and $\mathbb{P}$ on all of $\mathcal{F}$ and took $d\nu/d\mathbb{P}$ there, the answer would be $X$ itself — $\mathcal{F}$-measurable, and useless: it is not a summary of $X$ given $\mathcal{G}$, it *is* $X$. The restriction to $\mathcal{G}$ is not a technicality; it is where the coarsening happens.

*(iii) Signed $X$.* Write $X=X^+-X^-$ with $X^\pm\ge0$ and $\mathbb{E}[X^\pm]\le\mathbb{E}|X|<\infty$. Apply the above to each part to get $\mathcal{G}$-measurable $Y^\pm\ge0$ with $\int_GY^\pm d\mathbb{P}=\int_GX^\pm d\mathbb{P}$, and set $Y:=Y^+-Y^-$. Both $Y^\pm$ are integrable (take $G=\Omega$), so $Y$ is well defined a.s. and integrable, and $\int_GY\,d\mathbb{P}=\int_GX\,d\mathbb{P}$ for all $G\in\mathcal{G}$ by linearity. Thus (CE1)–(CE3) hold. (One may alternatively apply the signed version of Radon–Nikodym directly, via the Jordan decomposition of $\nu$ — the same argument, packaged.)

*(iv) Uniqueness.* Suppose $Y_1,Y_2$ both satisfy (CE1)–(CE3). Let $G:=\{Y_1-Y_2\ge\varepsilon\}$, which lies in $\mathcal{G}$ since both are $\mathcal{G}$-measurable. Subtracting the two instances of (CE3),
$$
0=\int_G(Y_1-Y_2)\,d\mathbb{P}\ \ge\ \varepsilon\,\mathbb{P}(G),
$$
so $\mathbb{P}(Y_1-Y_2\ge\varepsilon)=0$ for every $\varepsilon>0$; taking $\varepsilon=1/n$ and a countable union gives $\mathbb{P}(Y_1>Y_2)=0$, and by symmetry $Y_1=Y_2$ a.s.

*Two distinct versions.* Let $X\sim\mathrm{Uniform}(0,1)$ and $Y=X$. Then $g_1(x)=x$ and
$$
g_2(x)=\begin{cases}x,&x\ne\tfrac12,\\ 17,&x=\tfrac12,\end{cases}
$$
both give versions of $\mathbb{E}[Y\mid X]$: $g_1(X)$ and $g_2(X)$ are $\sigma(X)$-measurable and agree except on $\{X=1/2\}$, an event of probability $0$, so they have identical integrals over every $G\in\sigma(X)$. They differ at a point, so uniqueness cannot be improved beyond "a.s.". This matters in practice: a fitted regression function is only ever pinned down $\mathbb{P}_X$-almost everywhere, so two models that disagree on a measure-zero set of inputs are *the same estimator* as far as any risk is concerned — and conversely, a test set that concentrates on such a set measures nothing.

**C2.** *(a) Law of total variance.* Write $m:=\mathbb{E}[X\mid\mathcal{G}]$. By definition $\operatorname{Var}(X\mid\mathcal{G})=\mathbb{E}[X^2\mid\mathcal{G}]-m^2$ a.s. — this uses **linearity** (Theorem 6.8(a)) to expand $\mathbb{E}[(X-m)^2\mid\mathcal{G}]$ and **pull-out** (6.8(d)) twice to extract the $\mathcal{G}$-measurable $m$ from the cross term and the square. Take expectations and use the **tower property** (6.8(c)) on the first term:
$$
\mathbb{E}[\operatorname{Var}(X\mid\mathcal{G})]=\mathbb{E}[X^2]-\mathbb{E}[m^2].
$$
Separately, $\operatorname{Var}(m)=\mathbb{E}[m^2]-(\mathbb{E}m)^2=\mathbb{E}[m^2]-(\mathbb{E}X)^2$, the last step again by the **tower property**. Adding gives $\mathbb{E}[X^2]-(\mathbb{E}X)^2=\operatorname{Var}(X)$. Integrability of $X^2$ is needed throughout (Theorem 6.13's hypothesis $X\in L^2$); without it $\operatorname{Var}(X\mid\mathcal{G})$ need not be integrable and the identity can read $\infty=\infty$.

*(b) Bias–variance.* Fix a test point $x$. Let $D$ denote the training sample and $\hat f_D$ the fitted predictor; let $(x,Y)$ be a fresh draw with $Y=f^\star(x)+\varepsilon$, $\mathbb{E}[\varepsilon]=0$, $\operatorname{Var}(\varepsilon)=\sigma^2(x)$. **The independence assumption is that $(x,Y)$ is drawn independently of $D$**, and it is used at exactly one place: to assert $\mathbb{E}[\varepsilon\,(f^\star(x)-\hat f_D(x))]=\mathbb{E}[\varepsilon]\,\mathbb{E}[f^\star(x)-\hat f_D(x)]=0$, i.e. that the test noise does not correlate with the fitted function. Then
$$
\mathbb{E}_{D,\varepsilon}\bigl[(Y-\hat f_D(x))^2\bigr]=\underbrace{\sigma^2(x)}_{\text{Bayes risk}}+\underbrace{\bigl(f^\star(x)-\mathbb{E}_D\hat f_D(x)\bigr)^2}_{\text{bias}^2}+\underbrace{\operatorname{Var}_D(\hat f_D(x))}_{\text{variance}},
$$
*Be precise about which identity does which job here*, since it is easy to blur them. The **first** term is produced by the law of total variance: $\sigma^2(x)=\operatorname{Var}(Y\mid X=x)$, and averaging over $x$ gives the Bayes risk $\mathbb{E}[\operatorname{Var}(Y\mid X)]$, which is exactly the term total variance isolates as unexplainable by $X$. The **last two** terms come from a different and more elementary identity, the mean-square decomposition $\mathbb{E}[(Z-c)^2]=(\mathbb{E}Z-c)^2+\operatorname{Var}(Z)$ applied to $Z=\hat f_D(x)$ and $c=f^\star(x)$ — no conditioning is involved. So the three-way split is total variance *composed with* a bias–variance split of the estimator, not a single application of Theorem 6.13. If the test point is *not* independent of $D$ (leakage: the same example, or a near-duplicate, appears in both), the cross term is nonzero and negative, and the measured error is optimistically biased. That is the arithmetic behind every deduplication pass.

*(c)* The first term, $\sigma^2(x)$, is the Bayes risk: it is the irreducible error of the conditional mean itself and no estimator can go below it. The decomposition is exact for squared loss because squared loss is a squared Hilbert-space norm, so orthogonality gives an exact Pythagorean split (Theorem 6.5). For 0–1 loss there is no inner product and no orthogonality; the various "bias–variance decompositions" for classification are definitional constructions that do not add up in the same clean way, and error can even *decrease* when variance increases, because a noisy vote can cross the decision boundary in the right direction.

**C3.** *Appendix-style statement.*

> **Assumptions.** (A1) $F:\mathbb{R}^d\to\mathbb{R}$ is differentiable with $L$-Lipschitz gradient. (A2) $F_\star:=\inf F>-\infty$. (A3) $\mathbb{E}[g_t\mid\mathcal{F}_t]=\nabla F(\theta_t)$ a.s. (A4) $\mathbb{E}[\|g_t-\nabla F(\theta_t)\|^2\mid\mathcal{F}_t]\le\sigma^2$ a.s. (A5) $\eta_t$ is $\mathcal{F}_t$-measurable with $0<\eta_t\le1/L$.
>
> **Lemma.** Under (A1)–(A5), the iterates $\theta_{t+1}=\theta_t-\eta_tg_t$ satisfy, for every $T\ge1$,
> $$\mathbb{E}\Bigl[\sum_{t<T}\eta_t\|\nabla F(\theta_t)\|^2\Bigr]\ \le\ 2\bigl(F(\theta_0)-F_\star\bigr)+L\sigma^2\,\mathbb{E}\Bigl[\sum_{t<T}\eta_t^2\Bigr].$$

*Proof.*
**Step 1 (descent lemma).** By (A1), $F(\theta_{t+1})\le F(\theta_t)+\langle\nabla F(\theta_t),\theta_{t+1}-\theta_t\rangle+\frac L2\|\theta_{t+1}-\theta_t\|^2$. Substituting $\theta_{t+1}-\theta_t=-\eta_tg_t$,
$$
F(\theta_{t+1})\le F(\theta_t)-\eta_t\langle\nabla F(\theta_t),g_t\rangle+\tfrac{L}{2}\eta_t^2\|g_t\|^2 .
$$
**Step 2 (condition on the past).** Take $\mathbb{E}[\cdot\mid\mathcal{F}_t]$. Both $\nabla F(\theta_t)$ and $\eta_t$ are $\mathcal{F}_t$-measurable by (A5), so pull-out (Theorem 6.8(d)) applies to each, and (A3) gives $\mathbb{E}[\langle\nabla F(\theta_t),g_t\rangle\mid\mathcal{F}_t]=\|\nabla F(\theta_t)\|^2$. For the quadratic term, expand $\|g_t\|^2=\|\nabla F(\theta_t)\|^2+2\langle\nabla F(\theta_t),g_t-\nabla F(\theta_t)\rangle+\|g_t-\nabla F(\theta_t)\|^2$ and condition: the cross term vanishes by (A3) and pull-out, and the last is $\le\sigma^2$ by (A4). Hence
$$
\mathbb{E}[F(\theta_{t+1})\mid\mathcal{F}_t]\le F(\theta_t)-\eta_t\Bigl(1-\tfrac{L\eta_t}{2}\Bigr)\|\nabla F(\theta_t)\|^2+\tfrac{L}{2}\eta_t^2\sigma^2 .
$$
Since $\eta_t\le1/L$, the bracket is $\ge\frac12$, giving
$$
\mathbb{E}[F(\theta_{t+1})\mid\mathcal{F}_t]\le F(\theta_t)-\tfrac{\eta_t}{2}\|\nabla F(\theta_t)\|^2+\tfrac{L}{2}\eta_t^2\sigma^2 .\tag{$\ast$}
$$
**Step 3 (telescope).** Take total expectations in $(\ast)$ using the tower property, rearrange, and sum over $t=0,\dots,T-1$; the $\mathbb{E}[F(\theta_t)]$ terms telescope:
$$
\tfrac12\mathbb{E}\Bigl[\sum_{t<T}\eta_t\|\nabla F(\theta_t)\|^2\Bigr]\le F(\theta_0)-\mathbb{E}[F(\theta_T)]+\tfrac{L\sigma^2}{2}\mathbb{E}\Bigl[\sum_{t<T}\eta_t^2\Bigr]\le F(\theta_0)-F_\star+\tfrac{L\sigma^2}{2}\mathbb{E}\Bigl[\sum_{t<T}\eta_t^2\Bigr],
$$
the last step by (A2). Multiply by $2$. $\blacksquare$

*Remark (what breaks, and how you would see it).*
- **(a) Heavy-tailed gradient noise** violates **(A4)**: the conditional second moment is infinite, so Step 2's bound on $\mathbb{E}[\|g_t\|^2\mid\mathcal{F}_t]$ is vacuous. *Symptom:* the median loss trajectory looks fine while the mean across seeds is dominated by a handful of divergent runs, and the running max of $\|g_t\|$ grows without settling. In the experiment of Worked Example 6.22, median $F=1.09$ against a mean of $1.04\times10^3$.
- **(b) Adam-style step sizes** violate **(A5)**: $\eta_t$ computed from $g_t$ is $\mathcal{F}_{t+1}$-measurable, not $\mathcal{F}_t$-measurable, so it cannot be pulled out before conditioning and $\mathbb{E}[\eta_t\langle\nabla F(\theta_t),g_t\rangle\mid\mathcal{F}_t]\ne\eta_t\|\nabla F(\theta_t)\|^2$ — the step size and the noise are correlated. *Symptom:* a systematic drift that no amount of averaging removes; historically, this is the gap the original Adam proof missed and the counterexamples of the "On the Convergence of Adam" literature exploit.
- **(c) Minibatches without replacement within an epoch** violate **(A3)**: once part of the epoch is consumed, the remaining examples are no longer a fresh uniform draw, so the conditional mean of $g_t$ is a re-weighted gradient, not $\nabla F(\theta_t)$. *Symptom:* one would expect within-epoch structure, and in the extreme (sorted or curriculum-ordered data) a bias that persists; the mild case is often *helpful* in practice, which is why random reshuffling has its own analysis rather than being a corollary of the theorem above. Treat the specific symptom as an expectation from the algebra, not a measurement — we have not simulated it.
- **(d) A fixed augmentation seed per example** violates **(A3)**: the "noise" is now a deterministic function of which example was drawn, so revisiting an example reproduces the same perturbation exactly; conditionally on the history it does not re-randomise and its conditional mean is not zero relative to the intended objective. *Symptom:* the run converges — to the wrong thing — and no amount of further training or step-size decay repairs it, because the error is a bias, not a noise floor. **Verified.** Take $F(\theta)=\theta^2/2$, $n=50$ examples, and give example $i$ a frozen perturbation $b_i\sim N(0,1)$ drawn once; the stochastic gradient on a draw of example $i$ is $g=\theta+b_i$. Then $\mathbb{E}[g\mid\mathcal{F}_t]=\theta_t+\bar b$ where $\bar b$ is the empirical mean of the frozen perturbations, so (A3) fails and the recursion is driven to $-\bar b$ rather than to $0$. With $\eta_t=1/(t+10)$, $T=2\times10^5$ and 200 seeds (7000–7199): every run satisfied $|\theta_T+\bar b|\le6.0\times10^{-3}$, mean $|\theta_T|=0.12367$ against mean $|\bar b|=0.12370$, and mean $F(\theta_T)=0.012270$ against the predicted $\mathbb{E}[\bar b^2]/2=0.012274$. The identical experiment with a *freshly drawn* perturbation at every step gives mean $|\theta_T|=0.00179$ and mean $F=2\times10^{-6}$. Same noise distribution, same schedule, same code path except for one line — and a bias of $10^{-1}$ versus $10^{-3}$. This is (a) of §6.9.4 in applied dress: unconditional unbiasedness without conditional unbiasedness. Note also what the experiment shows about *diagnosis*: the residual error shrinks with more examples (as $\bar b\to0$ at rate $n^{-1/2}$) but not with more steps, which is the signature to look for.

*Corollary (constant step).* Let $\eta_t\equiv\eta\le1/L$. Then $\sum_{t<T}\eta_t=\eta T$ and $\sum_{t<T}\eta_t^2=\eta^2T$, so the Lemma reads $\eta\,\mathbb{E}[\sum_{t<T}\|\nabla F(\theta_t)\|^2]\le2(F(\theta_0)-F_\star)+L\sigma^2\eta^2T$. Dividing by $\eta T$:
$$
\frac1T\sum_{t<T}\mathbb{E}\|\nabla F(\theta_t)\|^2\le\frac{2(F(\theta_0)-F_\star)}{\eta T}+L\eta\sigma^2 .\qquad\blacksquare
$$
*Interpretation.* The first term is **optimisation progress**: how fast the initial suboptimality is burned off. It decays like $1/T$ and is *helped* by a large $\eta$. The second is the **noise floor**: it does not decay with $T$ at all and is *hurt* by a large $\eta$. Every constant-step run therefore has two phases — a transient governed by the first term, then a plateau at height $\propto\eta\sigma^2$ — and the plateau is what the experiment in Worked Example 6.21 measured directly ($0.00503$, $0.01087$, $0.02933$ for $\eta=0.01,0.02,0.05$: linear in $\eta$, as promised). Optimising the bound over $\eta$ gives $\eta\asymp\sqrt{(F(\theta_0)-F_\star)/(L\sigma^2T)}$ and a rate of $O(1/\sqrt T)$, the standard non-convex SGD rate — and it also explains the practice of *decaying* $\eta$ on a plateau: you are lowering the floor once the first term has stopped mattering.

### References and further reading

**Textbooks.** Williams, *Probability with Martingales* (CUP, 1991) is the closest companion to this chapter: conditional expectation via Radon–Nikodym, martingales, optional stopping, upcrossings and convergence, all in the same order and with the same emphasis. Durrett, *Probability: Theory and Examples* (5th ed., CUP, 2019), Ch. 4, covers the same ground with more examples and includes regular conditional distributions. Billingsley, *Probability and Measure* (3rd ed., Wiley, 1995) is the reference for the measure-theoretic underpinnings, including the Borel paradox discussion. Kallenberg, *Foundations of Modern Probability* (3rd ed., Springer, 2021) and Çınlar, *Probability and Stochastics* (Springer, 2011) are where to look for existence of regular conditional distributions on Polish spaces and the Borel isomorphism theorem at full strength. For the analysis debts: Folland, *Real Analysis* (2nd ed., Wiley, 1999) for Radon–Nikodym, Fubini–Tonelli and the convergence theorems; Brezis, *Functional Analysis, Sobolev Spaces and PDE* (Springer, 2011) for the projection theorem and Riesz–Fischer; Rockafellar, *Convex Analysis* (Princeton, 1970) for the countable family of affine minorants behind conditional Jensen; Nesterov, *Lectures on Convex Optimization* (2nd ed., Springer, 2018) for the descent lemma. Kechris, *Classical Descriptive Set Theory* (Springer, 1995) is the source for the standard Borel space facts. For statistical uses, van der Vaart, *Asymptotic Statistics* (CUP, 1998); for concentration, Boucheron, Lugosi and Massart, *Concentration Inequalities* (OUP, 2013), which treats Azuma–Hoeffding and McDiarmid in exactly the martingale framing used in §6.8; Wainwright, *High-Dimensional Statistics* (CUP, 2019) for the same material aimed at modern statistics. Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning* (2nd ed., Springer, 2009) for the bias–variance decomposition and ridge regression as used in §6.5 and Exercise B6. Cesa-Bianchi and Lugosi, *Prediction, Learning, and Games* (CUP, 2006) for the online-learning applications of Azuma sketched in Worked Example 6.20.

**Papers.** Robbins and Monro, "A stochastic approximation method" (*Annals of Mathematical Statistics*, 1951) is the origin of the step-size conditions in (A5). Robbins and Siegmund, "A convergence theorem for non negative almost supermartingales and some applications" (1971) is the exact tool used in Theorem 6.31. Ghadimi and Lan, "Stochastic first-order and zeroth-order methods for nonconvex stochastic programming" (*SIAM J. Optimization*, 2013) is the modern reference for the non-convex rate proved in Theorem 6.30 and its corollary. Bottou, Curtis and Nocedal, "Optimization methods for large-scale machine learning" (*SIAM Review*, 2018) is the best single survey of this material for an ML audience, and states the assumptions in the same style. Reddi, Kale and Kumar, "On the convergence of Adam and beyond" (ICLR 2018) supplies the counterexamples behind remark (b) of Exercise C3. On heavy tails: Simsekli, Sagun and Gurbuzbalaban, "A tail-index analysis of stochastic gradient noise in deep neural networks" (ICML 2019) for the empirical case that (A4) fails in practice, and Zhang, He, Sra and Jadbabaie, "Why gradient clipping accelerates training" (ICLR 2020) for the analysis of the fix explored in Exercise B5. Belkin, Hsu, Ma and Mandal, "Reconciling modern machine learning practice and the bias–variance trade-off" (*PNAS*, 2019) is the double-descent reference for Common Confusion 6.6 and Exercise B6. McDiarmid, "On the method of bounded differences" (*Surveys in Combinatorics*, 1989) is the original of Corollary 6.29. Finally, for Ville's inequality as the foundation of anytime-valid inference — the modern descendant of §6.8 — see Ramdas, Grünwald, Vovk and Shafer, "Game-theoretic statistics and safe anytime-valid inference" (*Statistical Science*, 2023).
