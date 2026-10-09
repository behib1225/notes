# Chapter 10 — Bayesian Inference

## Chapter 10 at a glance

| Section | Idea | ML payoff |
|---|---|---|
| 10.1 The Bayesian setup | Parameters are random variables; the posterior is a conditional law, and updating is coherent and order-free | Explains why streaming/online Bayesian updates are safe, and why the *posterior predictive* — not the fitted parameter — is the object you deploy |
| 10.2 Conjugacy | Closed-form posteriors for the standard models, all instances of one exponential-family theorem; hyperparameters are pseudo-counts | Label smoothing, additive smoothing in language models, count-based bandits, and every "add $\alpha$ to the numerator" heuristic you have ever seen |
| 10.3 Choosing priors | Flat is not neutral: a prior is a measure, and measures transform with Jacobians. Jeffreys priors, improper priors, and honest ML practice | Weight decay is a Gaussian prior with $\lambda=\sigma^2/\tau^2$; explains why "uninformative" priors on log-variances silently encode strong beliefs |
| 10.4 Hierarchical models | Partial pooling interpolates between one estimate per group and one estimate for all; shrinkage falls out of the algebra | Per-user/per-group/per-task estimates with few observations each: recommender cold start, A/B testing at scale, meta-learning |
| 10.5 Model comparison | The marginal likelihood is a distribution over *datasets*, so flexibility is paid for automatically; BIC is a Laplace approximation | Where model-selection scores come from, why evidence is prior-sensitive in a way cross-validation is not, and what WAIC/PSIS-LOO actually estimate |
| 10.6 Bayesian linear regression | The whole pipeline in closed form: posterior, predictive, evidence | The posterior mean *is* ridge; the predictive variance expresses the epistemic uncertainty a ridge point estimate cannot |
| 10.7 Approximate inference | Laplace approximation and its failure modes; a bridge to MCMC and VI | Cheapest possible uncertainty retrofit for a trained network |
| 10.8 Bayesian deep learning | Honest survey: Laplace, MC dropout, ensembles, SWAG, last-layer methods | Calibrated uncertainty for selective prediction, active learning, and OOD detection — usually the real goal, not a faithful posterior |
| 10.9 Bayesian decision theory | The Bayes action minimizes posterior expected loss; your loss picks the summary | Squared loss $\to$ mean, absolute $\to$ median, 0-1 $\to$ mode; asymmetric costs $\to$ quantile thresholds in screening, fraud, and inventory |

## 10.0 Orientation: what changes when parameters become random variables

Everything in this chapter follows from one move. Up to now, $\theta$ has been a fixed but unknown label on a family of distributions: we wrote $P_\theta$, formed a likelihood, maximized it, and asked how the maximizer fluctuates over repeated draws of the data. Chapter 8 studied estimators as functions of the sample; Chapter 9 studied tests and confidence sets whose probability statements are about the *sampling* mechanism with $\theta$ held fixed. The Bayesian move is to put $\theta$ on the same footing as the data: to build a single probability space carrying both, so that $\theta$ is a random variable in the sense of Definition 1.13 and $(\theta, X_1,\dots,X_n)$ has a joint law. Once you have a joint law, "what do I believe about $\theta$ having seen the data" stops being a philosophical question and becomes a computation: it is the conditional law of $\theta$ given the data, which is exactly the object Chapter 6 built.

That is the whole conceptual content. Everything else is technique for computing the conditional law, plus honesty about the modelling choices it exposes. But do not let the smallness of the move fool you: it changes which statements are meaningful. A frequentist confidence interval is a random set with a coverage guarantee over repeated experiments; a Bayesian credible interval is a set with specified posterior probability for the data you actually saw. Both are useful, they answer different questions, and they are not interchangeable — a point we will make precisely rather than tribally.

You are assumed to bring five things from earlier chapters, and I will use them by name rather than re-deriving them. From Chapter 2: the Radon–Nikodym theorem, densities with respect to a stated base measure, and the fact that "the density of $\theta$" is meaningless until you name the measure it is a density against; also KL divergence and the ELBO. From Chapter 3: the change-of-variables formula with its Jacobian factor, and the exponential-family form $p(x\mid\theta)=h(x)\exp(\langle \eta(\theta), T(x)\rangle - A(\theta))$ with its conjugacy structure. From Chapter 5: Gaussian conditioning, completing the square in a quadratic form, and Schur complements — Bayesian linear regression is a two-line consequence of these. From Chapter 6: conditional expectation as an $L^2$ projection, and the general fact that the minimizer of a posterior expected loss depends on the loss. From Chapter 8: MAP estimation and its relationship to penalized likelihood; from Chapter 9: Fisher information, hypothesis testing, and confidence intervals. From Chapter 1, by number: conditional probability (Definition 1.10), that conditioning yields a genuine probability measure (Proposition 1.11), the law of total probability and Bayes' rule with the fixed terminology prior / likelihood / evidence / posterior (Proposition 1.12), and conditional independence (Definition 1.24).

Here is the punchline, stated in advance so that you can watch it arrive. **Bayesian inference is one formula plus a great deal of bookkeeping about base measures.** The formula is $\text{posterior} \propto \text{likelihood} \times \text{prior}$. The bookkeeping is where the content lives, and it produces four results worth the price of admission. First, conjugate priors are not a bag of tricks but a single theorem about exponential families, and the theorem tells you that a prior's hyperparameters are *pseudo-counts* — imaginary data you are pretending to have seen. Second, "flat" is not a property of a belief but of a coordinate system, because a prior is a measure and measures pick up Jacobians; the Jeffreys prior is the unique fix. Third, the marginal likelihood is a normalized distribution over datasets, so a model that can explain everything must explain each thing weakly, and Occam's razor requires no extra penalty term. Fourth, the posterior is not an answer; it is an input to a decision. Squared loss returns its mean, absolute loss its median, 0-1 loss its mode, and asymmetric loss a quantile — so the question "what is my estimate of $\theta$" is incomplete until you say what an error costs.

A word on what this chapter is *not*. It is not an argument that you should be a Bayesian. Large parts of modern ML are unapologetically point-estimate machinery, and for good reason: a 70-billion-parameter posterior is not something anyone computes. What this chapter gives you is the ability to read the enormous Bayesian literature that ML draws on — variational autoencoders, Bayesian optimization, Thompson sampling, uncertainty quantification, hierarchical models in causal inference and A/B testing — and to recognize when a paper's "regularizer" is a prior in disguise, which is more often than papers say.

Two operating notes. Throughout, when I write $p(\theta)$ or $p(x\mid\theta)$ I mean a density with respect to a base measure that I will always name: Lebesgue measure $\lambda_d$ on $\mathbb{R}^d$, counting measure $\#$ on a discrete set, or a product of these. Chapter 2 earned us the right to be casual about which, but not the right to leave it unstated. And every number in this chapter was computed by running code, with seeds and sample sizes reported; where I quote a simulation, you can reproduce it.

## 10.1 The Bayesian setup

### 10.1.1 One probability space for parameters and data

Start with the picture in words. You have a family of candidate data-generating mechanisms indexed by $\theta$, and you have some opinion about which mechanisms are plausible before you look at the data. Bayes says: build a machine that first draws $\theta$ from your opinion, then draws data from the mechanism $\theta$ names. That machine has a joint distribution over $(\theta, X)$. Conditioning it on the observed $X$ turns your before-opinion into an after-opinion. That is all.

Formally, we need $\theta$ to live on a measurable space and the likelihood to be jointly measurable, so that the joint law exists.

**Definition 10.1 (Bayesian model).** Let $(\Theta,\mathcal{T})$ be a measurable space of parameters and $(\mathcal{X},\mathcal{A})$ a measurable space of observations. A *Bayesian model* consists of

1. a *prior* probability measure $\Pi$ on $(\Theta,\mathcal{T})$;
2. a *statistical model*: a family $\{P_\theta:\theta\in\Theta\}$ of probability measures on $(\mathcal{X},\mathcal{A})$ such that $\theta\mapsto P_\theta(A)$ is $\mathcal{T}$-measurable for every $A\in\mathcal{A}$ (that is, a *probability kernel* from $\Theta$ to $\mathcal{X}$).

The kernel condition is exactly what is needed for the joint measure
$$
\mathbb{P}(\mathrm{d}\theta,\mathrm{d}x) \;=\; \Pi(\mathrm{d}\theta)\,P_\theta(\mathrm{d}x)
$$
to be well defined on the product space $(\Theta\times\mathcal{X},\ \mathcal{T}\otimes\mathcal{A})$; the construction is the standard one for building a measure from a marginal and a kernel (Ch. 4, product measures and Fubini). Under $\mathbb{P}$, the coordinate maps $\theta$ and $X$ are random variables on one space, and $P_\theta$ is the conditional law of $X$ given $\theta$.

In practice all our models are *dominated*: there is a $\sigma$-finite base measure $\mu$ on $\mathcal{X}$ with $P_\theta \ll \mu$ for all $\theta$, and we write $p(x\mid\theta) := \mathrm{d}P_\theta/\mathrm{d}\mu\,(x)$ for the likelihood, chosen jointly measurable in $(\theta,x)$. Similarly $\Pi$ usually has a density $\pi(\theta)$ against Lebesgue or counting measure on $\Theta$. Radon–Nikodym (Ch. 2) is what licenses all of this notation, and Chapter 2's warning applies: $p$ and $\pi$ are only meaningful relative to their stated base measures.

**Definition 10.2 (evidence and posterior).** Given data $x\in\mathcal{X}$, define the *evidence* (or *marginal likelihood*)
$$
Z(x) \;:=\; \int_\Theta p(x\mid\theta)\,\Pi(\mathrm{d}\theta),
$$
and, whenever $0<Z(x)<\infty$, define the *posterior* measure $\Pi(\cdot\mid x)$ on $(\Theta,\mathcal{T})$ by
$$
\Pi(B\mid x) \;:=\; \frac{1}{Z(x)}\int_B p(x\mid\theta)\,\Pi(\mathrm{d}\theta),\qquad B\in\mathcal{T}.
$$
If $\Pi$ has density $\pi$ then the posterior has density $\pi(\theta\mid x) = p(x\mid\theta)\pi(\theta)/Z(x)$, the statement $\text{posterior}\propto\text{likelihood}\times\text{prior}$ of Proposition 1.12.

**Proposition 10.3 (the posterior is a probability measure, and exists almost surely).** Fix a Bayesian model as above with dominating measure $\mu$.

(i) For each $x$ with $0<Z(x)<\infty$, $\Pi(\cdot\mid x)$ is a probability measure on $(\Theta,\mathcal{T})$, and $\Pi(\cdot\mid x)\ll\Pi$ with $\frac{\mathrm{d}\Pi(\cdot\mid x)}{\mathrm{d}\Pi}(\theta) = p(x\mid\theta)/Z(x)$.

(ii) $Z$ is $\mathcal{A}$-measurable, $Z<\infty$ $\mu$-a.e., and the marginal law of $X$ under $\mathbb{P}$ is $m(\mathrm{d}x) = Z(x)\mu(\mathrm{d}x)$. Consequently $Z(X)\in(0,\infty)$ almost surely, so the posterior is defined for almost every dataset you could actually observe.

(iii) $\Pi(\cdot\mid x)$ is a version of the conditional distribution of $\theta$ given $X=x$: for every bounded measurable $f$, $\mathbb{E}[f(\theta)\mid X] = \int f\,\mathrm{d}\Pi(\cdot\mid X)$ a.s.

*Proof.* (i) Countable additivity of $B\mapsto \int_B p(x\mid\theta)\Pi(\mathrm{d}\theta)$ is monotone convergence applied to partial sums, and total mass $1$ is the definition of $Z(x)$. Absolute continuity and the stated derivative are immediate: if $\Pi(B)=0$ then the integral over $B$ vanishes. (ii) Measurability of $Z$ follows from joint measurability of $p$ and Tonelli. For the marginal, take $A\in\mathcal{A}$; then by Tonelli
$$
\mathbb{P}(X\in A)=\int_\Theta P_\theta(A)\,\Pi(\mathrm{d}\theta)=\int_\Theta\int_A p(x\mid\theta)\mu(\mathrm{d}x)\Pi(\mathrm{d}\theta)=\int_A Z(x)\,\mu(\mathrm{d}x).
$$
Taking $A=\mathcal{X}$ gives $\int Z\,\mathrm{d}\mu=1$, so $Z<\infty$ $\mu$-a.e. and $\{Z=0\}$ has $m$-measure zero. (iii) We must check the defining property of conditional expectation: for bounded measurable $g$ on $\mathcal{X}$,
$$
\mathbb{E}\Big[g(X)\int f\,\mathrm{d}\Pi(\cdot\mid X)\Big]
=\int_{\mathcal{X}} g(x)\frac{\int f(\theta)p(x\mid\theta)\Pi(\mathrm{d}\theta)}{Z(x)}Z(x)\mu(\mathrm{d}x)
=\int\!\!\int g(x)f(\theta)p(x\mid\theta)\mu(\mathrm{d}x)\Pi(\mathrm{d}\theta),
$$
which is $\mathbb{E}[g(X)f(\theta)]$ by Tonelli. Since $\sigma(X)$-measurable functions are exactly the $g(X)$ (Doob–Dynkin, Lemma 1.22), this is the required identity. $\blacksquare$

Part (ii) deserves a moment. It says you cannot be unlucky in a way that destroys the posterior: the set of datasets with $Z(x)=0$ or $Z(x)=\infty$ is null under the very model you are using. It does *not* say the posterior is well behaved if your prior is improper (not a probability measure), because then $Z$ need not integrate to anything; §10.3 handles that case separately.

> **Common Confusion 10.1.** *"Isn't the posterior just the likelihood, renormalized? The prior only rescales things, and rescaling doesn't change shape."*
>
> It rescales *pointwise in $\theta$*, which changes shape completely. Multiplying a function by another function is not multiplying it by a constant. Take a coin with $s=13$ heads in $n=20$ flips. The normalized likelihood is $\mathrm{Beta}(14,8)$, with mean $0.6364$. Under a $\mathrm{Beta}(10,10)$ prior the posterior is $\mathrm{Beta}(23,17)$ with mean $0.5750$ and a 95% credible interval $[0.4210,0.7219]$, versus $[0.4303,0.8189]$ for the normalized likelihood: a different centre, and an interval $23\%$ narrower (width $0.301$ against $0.389$). The prior contributes *information*, which is precisely a reweighting in $\theta$. The confusion is worth taking seriously, though, because it has a correct cousin: the posterior depends on the data only through the likelihood function up to a $\theta$-free factor. Two experiments with proportional likelihoods give the same posterior — the *likelihood principle* — which is exactly where Bayesian and frequentist analyses start to part company, since a frequentist $p$-value depends on the sample space of outcomes not observed.

### 10.1.2 Sequential updating, and why order does not matter

The most practically important structural fact about Bayesian inference is that it composes. If you process data in batches, updating your belief after each batch, you land in exactly the same place as if you had waited and processed everything at once — and you land there regardless of the order of the batches.

**Theorem 10.4 (coherence and order-invariance of updating).** Let $X_1,\dots,X_n$ be conditionally i.i.d. given $\theta$, with likelihood $p(x\mid\theta)$, and suppose $0<Z<\infty$ for the data at hand. Write $D_k=(x_1,\dots,x_k)$. Then:

(i) *(Chain rule / recursive updating.)* $\Pi(\mathrm{d}\theta\mid D_k)$ is the posterior obtained from prior $\Pi(\mathrm{d}\theta\mid D_{k-1})$ and the single observation $x_k$.

(ii) *(Order invariance.)* $\Pi(\mathrm{d}\theta\mid x_1,\dots,x_n)$ is unchanged by any permutation of $x_1,\dots,x_n$.

(iii) *(Factorized evidence.)* $Z(D_n)=\prod_{k=1}^n p(x_k\mid D_{k-1})$, where $p(x_k\mid D_{k-1})=\int p(x_k\mid\theta)\Pi(\mathrm{d}\theta\mid D_{k-1})$ is the one-step-ahead predictive density.

*Proof.* Conditional independence given $\theta$ (Definition 1.24) gives $p(D_n\mid\theta)=\prod_{k\le n}p(x_k\mid\theta)$. For (i), start from the definition and factor the likelihood:
$$
\pi(\theta\mid D_k)\;\propto\;\Big[\prod_{j\le k}p(x_j\mid\theta)\Big]\pi(\theta)
\;\propto\;p(x_k\mid\theta)\underbrace{\Big[\prod_{j\le k-1}p(x_j\mid\theta)\pi(\theta)\Big]}_{\propto\ \pi(\theta\mid D_{k-1})},
$$
and since both sides are probability densities in $\theta$, proportionality plus normalization gives equality. (ii) is immediate: the product $\prod_k p(x_k\mid\theta)$ is a product of scalars, hence symmetric in its factors, and the prior does not see the data at all. For (iii), telescope: by (i), $Z(D_k)/Z(D_{k-1}) = \int p(x_k\mid\theta)\pi(\theta\mid D_{k-1})\mathrm{d}\theta$, because
$$
Z(D_k)=\int \prod_{j\le k} p(x_j\mid\theta)\pi(\theta)\,\mathrm{d}\theta = Z(D_{k-1})\int p(x_k\mid\theta)\,\pi(\theta\mid D_{k-1})\,\mathrm{d}\theta .
$$
Multiplying these ratios for $k=1,\dots,n$ with $Z(D_0)=1$ gives the claim. $\blacksquare$

Part (ii) is doing real work in applications. It is why a streaming Bayesian system — a bandit algorithm updating after each impression, a Kalman filter, an online conjugate model in a recommender — is not making an approximation when it never revisits old data: the sufficient summary of the past is the current posterior. Part (iii) will return in §10.5: the evidence is a *prequential* score, the product of honest one-step-ahead predictions, which is the deepest reason it behaves like a cross-validation criterion.

One caution before you over-generalize the theorem: the hypothesis is doing exactly as much work as it says. For a model that is *not* conditionally i.i.d. — an autoregression with likelihood $p(x_k\mid x_{1:k-1},\theta)$, a hidden Markov model, any time series — parts (i) and (iii) survive verbatim with $p(x_k\mid\theta)$ replaced by the sequential factor $p(x_k\mid x_{1:k-1},\theta)$, because the chain rule of probability needs no symmetry. Part (ii) does not survive: the joint likelihood of an AR(1) model is not a symmetric function of the observations, and shuffling a time series before updating destroys the very dependence the model is about. Order-invariance needs the joint likelihood to be symmetric in the data, for which conditional i.i.d. is sufficient and exchangeability (next paragraph) is the exact condition. Streaming updates are always safe; *shuffled* updates are safe only for exchangeable models.

There is a practical corollary that is easy to miss and worth a great deal in applied work: **the posterior does not depend on why you stopped collecting data.** The likelihood of $D_n$ is $\prod_k p(x_k\mid\theta)$ whatever rule determined $n$ — a fixed budget, a deadline, a sequential test, or a decision to peek every morning and stop when the result looked good — provided the stopping rule depends only on the observed data and not on $\theta$ directly. Such a rule contributes a factor that is a function of $D_n$ alone, which cancels in the normalization. Frequentist error rates do not have this property: a $p$-value depends on the sample space of outcomes you might have obtained, so optional stopping inflates type-I error, which is why A/B testing platforms need sequential-testing corrections. A Bayesian analysis of the same experiment needs none. This is the operational form of the likelihood principle raised in Common Confusion 10.1, and it is the most under-appreciated practical advantage of the framework.

Order-invariance also invites a question we should answer rather than dodge: *why is there a prior at all?* The honest answer is not "because we like priors" but a theorem. If your beliefs about an infinite sequence of observations are *exchangeable* — invariant under permutation, which is exactly the symmetry Theorem 10.4(ii) exploits — then you are already committed to a prior, whether or not you write one down.

**Theorem 10.5 (de Finetti representation; stated, not proved).** Let $(X_1,X_2,\dots)$ be an infinite sequence of random variables taking values in a Polish space $\mathcal{X}$, whose joint law is *exchangeable*: for every $n$ and every permutation $\varsigma$ of $\{1,\dots,n\}$, $(X_1,\dots,X_n)$ and $(X_{\varsigma(1)},\dots,X_{\varsigma(n)})$ have the same law. Then there exists a unique probability measure $\Pi$ on the space of probability measures on $\mathcal{X}$ such that
$$
\mathbb{P}(X_1\in A_1,\dots,X_n\in A_n)=\int \prod_{i=1}^n P(A_i)\ \Pi(\mathrm{d}P)
$$
for all $n$ and all measurable $A_1,\dots,A_n$. Equivalently: the $X_i$ are conditionally i.i.d. given a random measure $P$, whose law is $\Pi$.

Read the display slowly, because it is the justification for everything in this chapter. Its left side contains no parameter and no prior — it is a statement about your beliefs concerning observables. Its right side is *exactly* Definition 10.1: a prior $\Pi$ over mechanisms, and data conditionally i.i.d. given the mechanism drawn. So a Bayesian model is not an extra assumption layered on top of exchangeability; it is what exchangeability *is*, in disguise. Someone who says "I have no prior, I only assume my observations are exchangeable" has asserted the existence of one and declined to say which.

Two limitations, stated plainly. The theorem needs an *infinite* exchangeable sequence; for finite sequences only an approximate version holds, with error of order $n/N$ when $n$ of $N$ items are observed. And the guaranteed prior lives on the space of *all* probability measures on $\mathcal{X}$, which is infinite-dimensional. It does not hand you a $\mathrm{Beta}(2,2)$; the choice of a finite-dimensional model is a separate, unjustified-by-de-Finetti modelling commitment. The theorem tells you a prior is unavoidable, not which one is right.

**Worked Example 10.1 (order invariance, checked).** Prior $\mathrm{Beta}(2,2)$, data $13$ heads and $7$ tails. Processing one flip at a time with the update $(a,b)\mapsto(a+x,b+1-x)$, three random permutations of the sequence (seed 10) all terminate at $\mathrm{Beta}(15,9)$. Posterior mean $15/24=0.6250$, posterior mode $14/22=0.6364$, posterior sd $0.0968$, and the central 95% credible interval is $[0.4273,0.8029]$. The MLE is $0.6500$. Notice the ordering: prior mean $0.5 <$ posterior mean $0.625 <$ MLE $0.65$; the posterior mean is a convex combination, as Proposition 10.11 will show in general.

Verifying (iii) on the same example: summing the logs of the twenty one-step predictive probabilities $p(x_k\mid D_{k-1})$ along the sequence $\text{HHH}\dots\text{TTT}$ gives $-14.0190920131$, and the closed form $\log\big(B(15,9)/B(2,2)\big)$ gives $-14.0190920131$. They agree to ten decimals, as they must.

### 10.1.3 The posterior predictive is the object you deploy

A posterior over parameters is rarely what you want. You want to predict a new observation $\tilde X$. Averaging the model over your uncertainty about $\theta$ gives the right thing.

**Definition 10.6 (posterior predictive).** Let $\tilde X$ be a new observation, conditionally independent of $D$ given $\theta$, with the same likelihood. The *posterior predictive distribution* is
$$
p(\tilde x\mid D)\;=\;\int_\Theta p(\tilde x\mid\theta)\,\Pi(\mathrm{d}\theta\mid D).
$$

**Proposition 10.7 (predictive moments; over-dispersion).** With $\mu(\theta):=\mathbb{E}[\tilde X\mid\theta]$ and $v(\theta):=\mathrm{Var}(\tilde X\mid\theta)$, assuming these are integrable under the posterior,
$$
\mathbb{E}[\tilde X\mid D]=\mathbb{E}[\mu(\theta)\mid D],\qquad
\mathrm{Var}(\tilde X\mid D)=\underbrace{\mathbb{E}[v(\theta)\mid D]}_{\text{aleatoric}}+\underbrace{\mathrm{Var}(\mu(\theta)\mid D)}_{\text{epistemic}} .
$$
In particular, if $v(\theta)\equiv v$ does not depend on $\theta$, the predictive variance strictly exceeds the plug-in variance $v$ whenever $\mu(\theta)$ is non-degenerate under the posterior.

*Proof.* Both statements are the tower property and the law of total variance from Chapter 6, applied under the posterior measure $\Pi(\cdot\mid D)$ rather than the prior; Proposition 1.11 guarantees that all theorems about probability measures apply verbatim to a conditional one. $\blacksquare$

This decomposition is the single most-cited formula in the ML uncertainty literature, usually under the names *aleatoric* (irreducible noise, $\mathbb{E}[v(\theta)\mid D]$) and *epistemic* (reducible by more data, $\mathrm{Var}(\mu(\theta)\mid D)$) uncertainty. Note what the labels do and do not mean: the split is relative to a fixed model class. Model misspecification lands in neither bucket.

**Worked Example 10.2 (predictive vs plug-in, five more flips).** Continuing Worked Example 10.1 with posterior $\mathrm{Beta}(15,9)$: the probability the next flip is heads is the posterior mean, $0.6250$. For $m=5$ further flips, the predictive is Beta-Binomial with mean $3.1250$ and variance $1.3594$ (sd $1.1659$); the plug-in $\mathrm{Bin}(5,0.625)$ has variance $1.1719$ (sd $1.0825$). The predictive variance is $16.0\%$ larger, exactly the epistemic term of Proposition 10.7. The tails are where it bites:

| $k$ heads in 5 | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Beta-Binomial | 0.0131 | 0.0755 | 0.2015 | 0.3114 | 0.2802 | 0.1183 |
| plug-in Binomial | 0.0074 | 0.0618 | 0.2060 | 0.3433 | 0.2861 | 0.0954 |

The plug-in model assigns probability $0.0954$ to five heads in a row; integrating over parameter uncertainty raises this to $0.1183$, a $24\%$ increase. Plug-in predictions are systematically overconfident in the tails, and tails are where costly decisions live.

### 10.1.4 Frequentist and Bayesian, and where they meet

The two frameworks answer different questions with the same algebra, and it is worth being blunt about the difference. A 95% *confidence* interval $C(X)$ satisfies $P_\theta(\theta\in C(X))\ge 0.95$ for every $\theta$: the randomness is in $X$, the guarantee is over hypothetical repetitions, and for your one dataset the interval either contains $\theta$ or does not. A 95% *credible* interval $B$ satisfies $\Pi(\theta\in B\mid X=x)=0.95$: the randomness is in $\theta$, and the statement is about the dataset you have. Neither is a substitute for the other, and the Bayesian statement is purchased with a prior.

Since we will report intervals constantly, it is worth fixing the definition and the two standard constructions, because they behave differently under reparameterization and the difference matters in §10.7.

**Definition 10.8 (credible set; equal-tailed and HPD intervals).** Let $\theta$ be real-valued with posterior $\Pi(\cdot\mid D)$ and let $\alpha\in(0,1)$. A set $B\in\mathcal{B}(\mathbb{R})$ is a *$(1-\alpha)$ credible set* if $\Pi(\theta\in B\mid D)\ge1-\alpha$. Two canonical choices:

1. the *equal-tailed* (or central) interval $[q_{\alpha/2},\,q_{1-\alpha/2}]$, where $q_u$ is a posterior $u$-quantile;
2. the *highest posterior density* (HPD) region $\{\theta:\pi(\theta\mid D)\ge c_\alpha\}$, with $c_\alpha$ chosen so the mass is $1-\alpha$ — the shortest such region when the density is unimodal.

Unless stated otherwise, every interval in this chapter is equal-tailed.

That last sentence is not housekeeping. Quantiles are *equivariant* under strictly increasing maps: if $\phi=g(\theta)$ with $g$ increasing, then $q_u(\phi)=g(q_u(\theta))$, so you may compute an equal-tailed interval in whichever coordinates are convenient and transform the endpoints. Densities are *not* equivariant — they pick up the Jacobian of Proposition 10.18 — so an HPD region computed for $\phi$ and mapped back is generally not the HPD region for $\theta$. Worked Example 10.23 exploits the equal-tailed property to fix a bad approximation by working on the logit scale; the same trick applied to HPD intervals would be wrong.

> **Common Confusion 10.2.** *"So a 95% credible interval is a 95% confidence interval — same numbers, different words?"*
>
> Sometimes nearly, often not, and never by definition. In Worked Example 10.1 the credible interval is $[0.4273,0.8029]$ while the Wald confidence interval $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}$ is $[0.4410,0.8590]$: different centre, different width. Under a flat prior in a Gaussian location model with known variance they coincide exactly, which is why the confusion survives — the textbook example is the degenerate case. The two objects have genuinely different guarantees: a credible interval has no coverage guarantee for fixed $\theta$ (choose a silly prior and it will fail badly), and a confidence interval has no probability interpretation for your particular sample. If a paper reports "95% CI" for a Bayesian model, look at whether it is a posterior quantile interval or a coverage-calibrated set; they are frequently conflated.

The frameworks do converge, and the theorem that says so is important enough to state even though its proof is beyond us here.

**Theorem 10.9 (Bernstein–von Mises; stated, not proved).** Let $\{P_\theta:\theta\in\Theta\}$, $\Theta\subseteq\mathbb{R}^d$ open, be a differentiable-in-quadratic-mean model with nonsingular Fisher information $I(\theta_0)$ (Ch. 9), let the model be identifiable with a consistent test separating $\theta_0$ from complements of its neighbourhoods, and let the prior $\Pi$ have a density that is positive and continuous at $\theta_0$. If $X_1,\dots,X_n\overset{iid}{\sim}P_{\theta_0}$, then the posterior of $\sqrt{n}(\theta-\hat\theta_n)$, where $\hat\theta_n$ is an efficient estimator, converges in total variation, in $P_{\theta_0}$-probability, to $N(0, I(\theta_0)^{-1})$.

The practical content: with enough data, in fixed finite dimension, the posterior forgets the prior, becomes Gaussian, and its credible sets are asymptotically confidence sets. The fine print is where the ML relevance lies. "Fixed finite dimension" excludes the regime where $d$ grows with $n$ — that is, essentially all of deep learning. "Positive density at $\theta_0$" excludes priors that assign zero mass near the truth, e.g. a spike-at-zero sparsity prior when the truth is not sparse. And on the boundary of the parameter space, or with non-identifiable models (permutation symmetry in mixtures, in neural networks), the theorem simply fails.

There is one more piece of fine print, and for ML it is the most important one: Theorem 10.9 assumes the data really came from some $P_{\theta_0}$ in the model. Remove that assumption — as honesty requires, since every model you will ever fit is misspecified — and the situation is this. Under regularity conditions, the posterior still concentrates, but at the *pseudo-true* parameter $\theta^\star=\arg\min_\theta D_{\mathrm{KL}}(P_{\text{true}}\,\|\,P_\theta)$, the KL-closest point of the model to the truth (Chapter 2's divergence, now picking the target of inference); a misspecified Bernstein–von Mises theorem then gives a Gaussian limit centred at $\theta^\star$, but with the *wrong width*: the posterior's covariance is built from the model's curvature, while the actual sampling fluctuation of the estimator is governed by the robust "sandwich" covariance $I^{-1}JI^{-1}$ (with $I$ the model's expected curvature at $\theta^\star$ and $J$ the true variance of the score there), and the two agree — $J=I$, the information identity of Chapter 9 — only when the model is correct. The upshot is stated plainly: under misspecification the posterior still gives you a sensible point (the best approximation in your model class, in the same KL sense in which maximum likelihood does), but its credible sets can over- or under-cover by arbitrary factors, and no amount of data repairs this — more data makes the intervals narrower around the wrong width. This is why "the model fits worse than its own posterior expects" is a detectable symptom (posterior predictive checks, §10.3.5's habit applied after the fit as well as before), and it is one root of the tempering practices we will meet in §10.8.

**Worked Example 10.3 (Bernstein–von Mises, numerically).** Bernoulli data with $\hat p=0.65$ held fixed as $n$ grows, prior $\mathrm{Beta}(2,2)$. The theorem predicts posterior sd $\approx 1/\sqrt{nI(\hat p)}=\sqrt{\hat p(1-\hat p)/n}$ and posterior mean $\approx$ MLE.

| $n$ | posterior mean | MLE | $\lvert$diff$\rvert$ | posterior sd | $1/\sqrt{nI(\hat p)}$ | ratio |
|---|---|---|---|---|---|---|
| 20 | 0.625000 | 0.65 | $2.5\times10^{-2}$ | 0.096825 | 0.106654 | 0.9078 |
| 200 | 0.647059 | 0.65 | $2.9\times10^{-3}$ | 0.033377 | 0.033727 | 0.9896 |
| 2 000 | 0.649701 | 0.65 | $3.0\times10^{-4}$ | 0.010654 | 0.010665 | 0.9989 |
| 20 000 | 0.649970 | 0.65 | $3.0\times10^{-5}$ | 0.003372 | 0.003373 | 0.9999 |

The prior's pull on the mean decays like $1/n$ (here exactly $(a_0+b_0)(\,0.5-\hat p)/(a_0+b_0+n)$), and the sd ratio approaches 1. At $n=20$ the discrepancy is 9%, which is the honest reason not to trust asymptotics on small samples: the asymptotics are correct and also irrelevant at the sample sizes where you needed a prior in the first place.

> **Self-check 10.1.**
> 1. Where in the proof of Theorem 10.4 was conditional independence given $\theta$ used, and where was it *not* needed?
> 2. Proposition 10.3(ii) says $Z(X)<\infty$ almost surely. Under which measure, and why does that qualifier matter?
> 3. Give a model in which the epistemic term of Proposition 10.7 is zero for all $n$ even though $\theta$ is unknown. (Hint: what if $\mu(\theta)$ is constant in $\theta$?)
> 4. Bernstein–von Mises says the prior washes out. Name two features of modern deep networks that break its hypotheses.

## 10.2 Conjugacy

### 10.2.1 What conjugacy is, and why it is not a coincidence

The integral defining $Z(x)$ is usually intractable. Conjugacy is the special situation in which it is not: the prior is chosen from a parametric family that the likelihood maps into itself, so updating becomes arithmetic on hyperparameters. Historically these looked like a list of lucky pairs — Beta with Bernoulli, Gamma with Poisson, Gaussian with Gaussian. Theorem 10.16 will show they are one construction applied to different exponential families, and that construction hands us a genuinely useful intuition: **the prior's hyperparameters are counts of fictitious data.** Once you see that, "add-one smoothing", "label smoothing", "pseudo-count exploration bonuses" and "prior sample size" all become the same idea.

Because parameterizations differ between textbooks and silently break formulas, here are the ones this chapter uses throughout. Every density is with respect to $\lambda_1$ (or $\lambda_d$ on the simplex) unless the support is discrete, in which case counting measure.

| distribution | density / pmf $\propto$ | support | mean |
|---|---|---|---|
| $\mathrm{Beta}(a,b)$ | $\theta^{a-1}(1-\theta)^{b-1}$ | $(0,1)$ | $a/(a+b)$ |
| $\mathrm{Dirichlet}(\alpha_1,\dots,\alpha_K)$ | $\prod_k\theta_k^{\alpha_k-1}$ | simplex $\Delta_{K-1}$ | $\alpha_k/\alpha_0$, $\alpha_0=\sum_j\alpha_j$ |
| $\mathrm{Gamma}(\alpha,\beta)$ (shape, **rate**) | $\lambda^{\alpha-1}e^{-\beta\lambda}$ | $(0,\infty)$ | $\alpha/\beta$ |
| $\mathrm{Inv\text{-}Gamma}(\alpha,\beta)$ | $(\sigma^2)^{-\alpha-1}e^{-\beta/\sigma^2}$ | $(0,\infty)$ | $\beta/(\alpha-1)$, $\alpha>1$ |
| $t_\nu(m,s^2)$ | $\big[1+\tfrac{(\theta-m)^2}{\nu s^2}\big]^{-(\nu+1)/2}$ | $\mathbb{R}$ | $m$ ($\nu>1$); variance $s^2\tfrac{\nu}{\nu-2}$ |
| $\mathrm{Laplace}(0,b)$ | $e^{-|\theta|/b}$ | $\mathbb{R}$ | $0$; variance $2b^2$ |

The shape–rate convention for the Gamma is the one that makes Proposition 10.15's update additive; SciPy's `scale` is $1/\beta$.

**Definition 10.10 (conjugate family).** Let $\{P_\theta\}$ be a statistical model. A family $\mathcal{F}=\{\Pi_\xi:\xi\in\Xi\}$ of priors on $\Theta$ is *conjugate* to the model if for every $\xi\in\Xi$ and every $x$ in a set of full measure, the posterior $\Pi_\xi(\cdot\mid x)$ again belongs to $\mathcal{F}$; that is, there is an update map $\xi\mapsto \xi'(\xi,x)$ with $\Pi_\xi(\cdot\mid x)=\Pi_{\xi'}$.

Two cautions before we compute. First, conjugacy is a property of the *pair* (family, model), not of the prior alone. Second, the definition is trivially satisfiable — take $\mathcal{F}$ to be all probability measures — so the content is always "conjugate *and* finite-dimensional *and* the update map is cheap."

### 10.2.2 Beta–Bernoulli and Beta–Binomial

**Proposition 10.11 (Beta–Bernoulli).** Let $X_1,\dots,X_n\mid\theta \overset{iid}{\sim}\mathrm{Bernoulli}(\theta)$ with $\theta\sim\mathrm{Beta}(a_0,b_0)$, $a_0,b_0>0$, and let $s=\sum_i x_i$. Then

(i) $\theta\mid D \sim \mathrm{Beta}(a_n,b_n)$ with $a_n=a_0+s$, $b_n=b_0+n-s$;

(ii) the posterior mean is a convex combination of the prior mean and the MLE,
$$
\mathbb{E}[\theta\mid D]=\frac{a_0+s}{a_0+b_0+n}=\frac{a_0+b_0}{a_0+b_0+n}\cdot\frac{a_0}{a_0+b_0}+\frac{n}{a_0+b_0+n}\cdot\frac{s}{n};
$$

(iii) the evidence is $Z(D)=B(a_n,b_n)/B(a_0,b_0)$ (with respect to counting measure on $\{0,1\}^n$);

(iv) the posterior predictive for $m$ further trials is Beta-Binomial:
$$
p(\tilde s = k\mid D)=\binom{m}{k}\frac{B(a_n+k,\ b_n+m-k)}{B(a_n,b_n)},\qquad
\mathbb{P}(\tilde X_{n+1}=1\mid D)=\frac{a_n}{a_n+b_n}.
$$

*Proof.* The $\mathrm{Beta}(a,b)$ density is $\theta^{a-1}(1-\theta)^{b-1}/B(a,b)$ on $(0,1)$ with respect to $\lambda_1$. Multiply by the likelihood $\theta^{s}(1-\theta)^{n-s}$:
$$
\pi(\theta\mid D)\ \propto\ \theta^{a_0+s-1}(1-\theta)^{b_0+n-s-1},
$$
which is the $\mathrm{Beta}(a_n,b_n)$ kernel; since both sides integrate to 1, (i) follows and the normalizing constant must be $B(a_n,b_n)$. This also gives (iii): $Z(D)=\int\theta^{a_0+s-1}(1-\theta)^{b_0+n-s-1}\mathrm{d}\theta/B(a_0,b_0)$. For (ii), $\mathbb{E}[\mathrm{Beta}(a,b)]=a/(a+b)$; split the numerator $a_0+s$ and divide. For (iv), integrate the Binomial likelihood against the posterior,
$$
\int_0^1\binom{m}{k}\theta^k(1-\theta)^{m-k}\frac{\theta^{a_n-1}(1-\theta)^{b_n-1}}{B(a_n,b_n)}\mathrm{d}\theta=\binom{m}{k}\frac{B(a_n+k,b_n+m-k)}{B(a_n,b_n)},
$$
and set $m=1,k=1$ for the special case. $\blacksquare$

Read (ii) again: the prior acts through $a_0+b_0$, a *sample size*. A $\mathrm{Beta}(2,2)$ prior contributes as much as four coin flips (two heads, two tails); a $\mathrm{Beta}(50,50)$ prior as much as a hundred. This is the pseudo-count reading, and it is the fastest way to check whether a prior you have written down is as mild as you think.

**Worked Example 10.4 (prior sensitivity, $n=20$, $s=13$).** All entries computed exactly from Proposition 10.11.

| prior | interpretation | posterior mean | 95% credible interval |
|---|---|---|---|
| $\mathrm{Beta}(1,1)$ | uniform, 2 pseudo-flips | 0.6364 | $[0.4303,\,0.8189]$ |
| $\mathrm{Beta}(0.5,0.5)$ | Jeffreys, 1 pseudo-flip | 0.6429 | $[0.4322,\,0.8277]$ |
| $\mathrm{Beta}(2,2)$ | mild, 4 pseudo-flips | 0.6250 | $[0.4273,\,0.8029]$ |
| $\mathrm{Beta}(10,10)$ | "probably fair", 20 pseudo-flips | 0.5750 | $[0.4210,\,0.7219]$ |
| $\mathrm{Beta}(1,9)$ | "probably rare", 10 pseudo-flips, mean 0.1 | 0.4667 | $[0.2945,\,0.6431]$ |
| $\mathrm{Beta}(50,50)$ | dogmatic, 100 pseudo-flips | 0.5250 | $[0.4358,\,0.6134]$ |

With $n=20$ real observations, priors carrying $\le 4$ pseudo-flips move the posterior mean by at most $0.011$ from the uniform answer, while the dogmatic prior moves it by $0.111$ and shrinks the interval to less than half the width. Two lessons for practice: report this table, and note that "the prior barely matters" is a claim about the ratio of pseudo-counts to real counts, not a general truth.

Figure 10.1 shows the same mechanism dynamically, with data arriving in batches.

<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="sans-serif">
  <line x1="60.00" y1="300.00" x2="730.00" y2="300.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="60.00" y1="300.00" x2="60.00" y2="30.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="60.00" y1="300.00" x2="60.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="60.00" y="320.00" font-size="12" fill="currentColor" text-anchor="middle" font-weight="normal">0.0</text>
  <line x1="194.00" y1="300.00" x2="194.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="194.00" y="320.00" font-size="12" fill="currentColor" text-anchor="middle" font-weight="normal">0.2</text>
  <line x1="328.00" y1="300.00" x2="328.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="328.00" y="320.00" font-size="12" fill="currentColor" text-anchor="middle" font-weight="normal">0.4</text>
  <line x1="462.00" y1="300.00" x2="462.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="462.00" y="320.00" font-size="12" fill="currentColor" text-anchor="middle" font-weight="normal">0.6</text>
  <line x1="596.00" y1="300.00" x2="596.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="596.00" y="320.00" font-size="12" fill="currentColor" text-anchor="middle" font-weight="normal">0.8</text>
  <line x1="730.00" y1="300.00" x2="730.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="730.00" y="320.00" font-size="12" fill="currentColor" text-anchor="middle" font-weight="normal">1.0</text>
  <text x="395.00" y="344.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">theta  (probability of heads)</text>
  <text x="52.00" y="36.00" font-size="12" fill="currentColor" text-anchor="end" font-weight="normal">density</text>
  <polyline points="60.7,299.8 63.7,298.9 66.8,298.0 69.8,297.2 72.9,296.3 75.9,295.5 79.0,294.6 82.0,293.8 85.1,293.0 88.1,292.1 91.2,291.3 94.3,290.5 97.3,289.7 100.4,288.9 103.4,288.2 106.5,287.4 109.5,286.6 112.6,285.9 115.6,285.1 118.7,284.4 121.7,283.7 124.8,282.9 127.8,282.2 130.9,281.5 133.9,280.8 137.0,280.1 140.1,279.5 143.1,278.8 146.2,278.1 149.2,277.5 152.3,276.8 155.3,276.2 158.4,275.5 161.4,274.9 164.5,274.3 167.5,273.7 170.6,273.1 173.6,272.5 176.7,271.9 179.7,271.3 182.8,270.8 185.9,270.2 188.9,269.7 192.0,269.1 195.0,268.6 198.1,268.1 201.1,267.5 204.2,267.0 207.2,266.5 210.3,266.0 213.3,265.5 216.4,265.1 219.4,264.6 222.5,264.1 225.5,263.7 228.6,263.2 231.7,262.8 234.7,262.4 237.8,261.9 240.8,261.5 243.9,261.1 246.9,260.7 250.0,260.3 253.0,260.0 256.1,259.6 259.1,259.2 262.2,258.9 265.2,258.5 268.3,258.2 271.3,257.8 274.4,257.5 277.5,257.2 280.5,256.9 283.6,256.6 286.6,256.3 289.7,256.0 292.7,255.7 295.8,255.5 298.8,255.2 301.9,255.0 304.9,254.7 308.0,254.5 311.0,254.2 314.1,254.0 317.1,253.8 320.2,253.6 323.2,253.4 326.3,253.2 329.4,253.1 332.4,252.9 335.5,252.7 338.5,252.6 341.6,252.4 344.6,252.3 347.7,252.2 350.7,252.0 353.8,251.9 356.8,251.8 359.9,251.7 362.9,251.6 366.0,251.5 369.0,251.5 372.1,251.4 375.2,251.4 378.2,251.3 381.3,251.3 384.3,251.2 387.4,251.2 390.4,251.2 393.5,251.2 396.5,251.2 399.6,251.2 402.6,251.2 405.7,251.2 408.7,251.3 411.8,251.3 414.8,251.4 417.9,251.4 421.0,251.5 424.0,251.5 427.1,251.6 430.1,251.7 433.2,251.8 436.2,251.9 439.3,252.0 442.3,252.2 445.4,252.3 448.4,252.4 451.5,252.6 454.5,252.7 457.6,252.9 460.6,253.1 463.7,253.2 466.8,253.4 469.8,253.6 472.9,253.8 475.9,254.0 479.0,254.2 482.0,254.5 485.1,254.7 488.1,255.0 491.2,255.2 494.2,255.5 497.3,255.7 500.3,256.0 503.4,256.3 506.4,256.6 509.5,256.9 512.5,257.2 515.6,257.5 518.7,257.8 521.7,258.2 524.8,258.5 527.8,258.9 530.9,259.2 533.9,259.6 537.0,260.0 540.0,260.3 543.1,260.7 546.1,261.1 549.2,261.5 552.2,261.9 555.3,262.4 558.3,262.8 561.4,263.2 564.5,263.7 567.5,264.1 570.6,264.6 573.6,265.1 576.7,265.5 579.7,266.0 582.8,266.5 585.8,267.0 588.9,267.5 591.9,268.1 595.0,268.6 598.0,269.1 601.1,269.7 604.1,270.2 607.2,270.8 610.3,271.3 613.3,271.9 616.4,272.5 619.4,273.1 622.5,273.7 625.5,274.3 628.6,274.9 631.6,275.5 634.7,276.2 637.7,276.8 640.8,277.5 643.8,278.1 646.9,278.8 649.9,279.5 653.0,280.1 656.1,280.8 659.1,281.5 662.2,282.2 665.2,282.9 668.3,283.7 671.3,284.4 674.4,285.1 677.4,285.9 680.5,286.6 683.5,287.4 686.6,288.2 689.6,288.9 692.7,289.7 695.7,290.5 698.8,291.3 701.9,292.1 704.9,293.0 708.0,293.8 711.0,294.6 714.1,295.5 717.1,296.3 720.2,297.2 723.2,298.0 726.3,298.9 729.3,299.8" fill="none" stroke="currentColor" stroke-width="2.2" stroke-dasharray="5,4"/>
  <polyline points="60.7,300.0 63.7,300.0 66.8,300.0 69.8,300.0 72.9,300.0 75.9,300.0 79.0,300.0 82.0,300.0 85.1,300.0 88.1,300.0 91.2,299.9 94.3,299.9 97.3,299.9 100.4,299.9 103.4,299.8 106.5,299.8 109.5,299.8 112.6,299.7 115.6,299.7 118.7,299.6 121.7,299.5 124.8,299.5 127.8,299.4 130.9,299.3 133.9,299.2 137.0,299.1 140.1,299.0 143.1,298.9 146.2,298.8 149.2,298.7 152.3,298.5 155.3,298.4 158.4,298.2 161.4,298.1 164.5,297.9 167.5,297.7 170.6,297.6 173.6,297.4 176.7,297.2 179.7,296.9 182.8,296.7 185.9,296.5 188.9,296.3 192.0,296.0 195.0,295.7 198.1,295.5 201.1,295.2 204.2,294.9 207.2,294.6 210.3,294.3 213.3,294.0 216.4,293.7 219.4,293.3 222.5,293.0 225.5,292.6 228.6,292.2 231.7,291.9 234.7,291.5 237.8,291.1 240.8,290.7 243.9,290.2 246.9,289.8 250.0,289.4 253.0,288.9 256.1,288.5 259.1,288.0 262.2,287.5 265.2,287.0 268.3,286.5 271.3,286.0 274.4,285.5 277.5,285.0 280.5,284.4 283.6,283.9 286.6,283.3 289.7,282.8 292.7,282.2 295.8,281.6 298.8,281.0 301.9,280.4 304.9,279.8 308.0,279.2 311.0,278.6 314.1,278.0 317.1,277.3 320.2,276.7 323.2,276.0 326.3,275.4 329.4,274.7 332.4,274.0 335.5,273.4 338.5,272.7 341.6,272.0 344.6,271.3 347.7,270.6 350.7,269.9 353.8,269.2 356.8,268.5 359.9,267.8 362.9,267.0 366.0,266.3 369.0,265.6 372.1,264.9 375.2,264.1 378.2,263.4 381.3,262.7 384.3,261.9 387.4,261.2 390.4,260.4 393.5,259.7 396.5,258.9 399.6,258.2 402.6,257.5 405.7,256.7 408.7,256.0 411.8,255.3 414.8,254.5 417.9,253.8 421.0,253.1 424.0,252.3 427.1,251.6 430.1,250.9 433.2,250.2 436.2,249.5 439.3,248.8 442.3,248.1 445.4,247.4 448.4,246.7 451.5,246.0 454.5,245.4 457.6,244.7 460.6,244.0 463.7,243.4 466.8,242.8 469.8,242.2 472.9,241.5 475.9,241.0 479.0,240.4 482.0,239.8 485.1,239.2 488.1,238.7 491.2,238.2 494.2,237.6 497.3,237.1 500.3,236.7 503.4,236.2 506.4,235.7 509.5,235.3 512.5,234.9 515.6,234.5 518.7,234.1 521.7,233.8 524.8,233.4 527.8,233.1 530.9,232.8 533.9,232.6 537.0,232.3 540.0,232.1 543.1,231.9 546.1,231.8 549.2,231.6 552.2,231.5 555.3,231.4 558.3,231.4 561.4,231.4 564.5,231.4 567.5,231.4 570.6,231.5 573.6,231.6 576.7,231.7 579.7,231.9 582.8,232.1 585.8,232.3 588.9,232.6 591.9,232.9 595.0,233.2 598.0,233.6 601.1,234.0 604.1,234.5 607.2,235.0 610.3,235.6 613.3,236.2 616.4,236.8 619.4,237.5 622.5,238.2 625.5,239.0 628.6,239.8 631.6,240.6 634.7,241.6 637.7,242.5 640.8,243.5 643.8,244.6 646.9,245.7 649.9,246.9 653.0,248.1 656.1,249.4 659.1,250.8 662.2,252.2 665.2,253.6 668.3,255.1 671.3,256.7 674.4,258.3 677.4,260.0 680.5,261.8 683.5,263.6 686.6,265.5 689.6,267.5 692.7,269.5 695.7,271.6 698.8,273.7 701.9,276.0 704.9,278.3 708.0,280.6 711.0,283.1 714.1,285.6 717.1,288.2 720.2,290.9 723.2,293.6 726.3,296.4 729.3,299.4" fill="none" stroke="#f59e0b" stroke-width="1.8"/>
  <polyline points="60.7,300.0 63.7,300.0 66.8,300.0 69.8,300.0 72.9,300.0 75.9,300.0 79.0,300.0 82.0,300.0 85.1,300.0 88.1,300.0 91.2,300.0 94.3,300.0 97.3,300.0 100.4,300.0 103.4,300.0 106.5,300.0 109.5,300.0 112.6,300.0 115.6,300.0 118.7,300.0 121.7,300.0 124.8,300.0 127.8,300.0 130.9,299.9 133.9,299.9 137.0,299.9 140.1,299.9 143.1,299.9 146.2,299.9 149.2,299.8 152.3,299.8 155.3,299.8 158.4,299.7 161.4,299.7 164.5,299.6 167.5,299.6 170.6,299.5 173.6,299.5 176.7,299.4 179.7,299.3 182.8,299.2 185.9,299.2 188.9,299.1 192.0,299.0 195.0,298.8 198.1,298.7 201.1,298.6 204.2,298.4 207.2,298.3 210.3,298.1 213.3,298.0 216.4,297.8 219.4,297.6 222.5,297.4 225.5,297.1 228.6,296.9 231.7,296.7 234.7,296.4 237.8,296.1 240.8,295.8 243.9,295.5 246.9,295.2 250.0,294.9 253.0,294.5 256.1,294.1 259.1,293.7 262.2,293.3 265.2,292.9 268.3,292.5 271.3,292.0 274.4,291.5 277.5,291.0 280.5,290.5 283.6,290.0 286.6,289.4 289.7,288.8 292.7,288.2 295.8,287.6 298.8,287.0 301.9,286.3 304.9,285.6 308.0,284.9 311.0,284.2 314.1,283.5 317.1,282.7 320.2,281.9 323.2,281.1 326.3,280.3 329.4,279.5 332.4,278.6 335.5,277.7 338.5,276.8 341.6,275.9 344.6,275.0 347.7,274.0 350.7,273.0 353.8,272.1 356.8,271.1 359.9,270.0 362.9,269.0 366.0,267.9 369.0,266.9 372.1,265.8 375.2,264.7 378.2,263.6 381.3,262.5 384.3,261.3 387.4,260.2 390.4,259.0 393.5,257.9 396.5,256.7 399.6,255.5 402.6,254.3 405.7,253.2 408.7,252.0 411.8,250.8 414.8,249.6 417.9,248.4 421.0,247.2 424.0,246.0 427.1,244.8 430.1,243.7 433.2,242.5 436.2,241.3 439.3,240.2 442.3,239.0 445.4,237.9 448.4,236.8 451.5,235.7 454.5,234.6 457.6,233.5 460.6,232.4 463.7,231.4 466.8,230.4 469.8,229.4 472.9,228.4 475.9,227.5 479.0,226.6 482.0,225.7 485.1,224.9 488.1,224.1 491.2,223.3 494.2,222.6 497.3,221.9 500.3,221.2 503.4,220.6 506.4,220.0 509.5,219.5 512.5,219.0 515.6,218.6 518.7,218.2 521.7,217.9 524.8,217.6 527.8,217.4 530.9,217.2 533.9,217.1 537.0,217.0 540.0,217.0 543.1,217.1 546.1,217.2 549.2,217.4 552.2,217.6 555.3,217.9 558.3,218.3 561.4,218.7 564.5,219.2 567.5,219.8 570.6,220.4 573.6,221.1 576.7,221.9 579.7,222.7 582.8,223.7 585.8,224.6 588.9,225.7 591.9,226.8 595.0,227.9 598.0,229.2 601.1,230.5 604.1,231.8 607.2,233.3 610.3,234.7 613.3,236.3 616.4,237.9 619.4,239.6 622.5,241.3 625.5,243.0 628.6,244.9 631.6,246.7 634.7,248.6 637.7,250.6 640.8,252.6 643.8,254.6 646.9,256.6 649.9,258.7 653.0,260.8 656.1,262.9 659.1,265.0 662.2,267.1 665.2,269.3 668.3,271.4 671.3,273.5 674.4,275.6 677.4,277.6 680.5,279.7 683.5,281.6 686.6,283.6 689.6,285.5 692.7,287.3 695.7,289.0 698.8,290.7 701.9,292.2 704.9,293.7 708.0,295.0 711.0,296.2 714.1,297.3 717.1,298.2 720.2,298.9 723.2,299.5 726.3,299.8 729.3,300.0" fill="none" stroke="#ef4444" stroke-width="1.8"/>
  <polyline points="60.7,300.0 63.7,300.0 66.8,300.0 69.8,300.0 72.9,300.0 75.9,300.0 79.0,300.0 82.0,300.0 85.1,300.0 88.1,300.0 91.2,300.0 94.3,300.0 97.3,300.0 100.4,300.0 103.4,300.0 106.5,300.0 109.5,300.0 112.6,300.0 115.6,300.0 118.7,300.0 121.7,300.0 124.8,300.0 127.8,300.0 130.9,300.0 133.9,300.0 137.0,300.0 140.1,300.0 143.1,300.0 146.2,300.0 149.2,300.0 152.3,300.0 155.3,300.0 158.4,300.0 161.4,300.0 164.5,300.0 167.5,300.0 170.6,300.0 173.6,300.0 176.7,300.0 179.7,300.0 182.8,300.0 185.9,300.0 188.9,300.0 192.0,300.0 195.0,300.0 198.1,300.0 201.1,300.0 204.2,300.0 207.2,300.0 210.3,300.0 213.3,300.0 216.4,300.0 219.4,300.0 222.5,300.0 225.5,300.0 228.6,300.0 231.7,300.0 234.7,300.0 237.8,300.0 240.8,300.0 243.9,299.9 246.9,299.9 250.0,299.9 253.0,299.9 256.1,299.9 259.1,299.9 262.2,299.8 265.2,299.8 268.3,299.8 271.3,299.7 274.4,299.7 277.5,299.6 280.5,299.6 283.6,299.5 286.6,299.4 289.7,299.3 292.7,299.2 295.8,299.0 298.8,298.9 301.9,298.7 304.9,298.5 308.0,298.3 311.0,298.1 314.1,297.8 317.1,297.5 320.2,297.2 323.2,296.8 326.3,296.4 329.4,296.0 332.4,295.5 335.5,294.9 338.5,294.3 341.6,293.7 344.6,293.0 347.7,292.2 350.7,291.3 353.8,290.4 356.8,289.4 359.9,288.4 362.9,287.2 366.0,286.0 369.0,284.7 372.1,283.3 375.2,281.8 378.2,280.2 381.3,278.5 384.3,276.7 387.4,274.8 390.4,272.8 393.5,270.7 396.5,268.4 399.6,266.1 402.6,263.7 405.7,261.2 408.7,258.5 411.8,255.8 414.8,252.9 417.9,250.0 421.0,247.0 424.0,243.9 427.1,240.7 430.1,237.5 433.2,234.2 436.2,230.8 439.3,227.4 442.3,223.9 445.4,220.5 448.4,217.0 451.5,213.5 454.5,210.0 457.6,206.6 460.6,203.2 463.7,199.8 466.8,196.5 469.8,193.3 472.9,190.2 475.9,187.2 479.0,184.3 482.0,181.6 485.1,179.1 488.1,176.7 491.2,174.5 494.2,172.5 497.3,170.7 500.3,169.1 503.4,167.8 506.4,166.8 509.5,166.0 512.5,165.4 515.6,165.2 518.7,165.2 521.7,165.5 524.8,166.1 527.8,167.0 530.9,168.2 533.9,169.7 537.0,171.4 540.0,173.5 543.1,175.8 546.1,178.3 549.2,181.1 552.2,184.2 555.3,187.4 558.3,190.9 561.4,194.5 564.5,198.3 567.5,202.3 570.6,206.4 573.6,210.6 576.7,214.9 579.7,219.2 582.8,223.6 585.8,228.0 588.9,232.4 591.9,236.8 595.0,241.1 598.0,245.3 601.1,249.5 604.1,253.5 607.2,257.4 610.3,261.2 613.3,264.8 616.4,268.3 619.4,271.5 622.5,274.6 625.5,277.5 628.6,280.2 631.6,282.7 634.7,284.9 637.7,287.0 640.8,288.9 643.8,290.6 646.9,292.1 649.9,293.4 653.0,294.6 656.1,295.6 659.1,296.5 662.2,297.2 665.2,297.8 668.3,298.3 671.3,298.7 674.4,299.1 677.4,299.3 680.5,299.5 683.5,299.7 686.6,299.8 689.6,299.9 692.7,299.9 695.7,299.9 698.8,300.0 701.9,300.0 704.9,300.0 708.0,300.0 711.0,300.0 714.1,300.0 717.1,300.0 720.2,300.0 723.2,300.0 726.3,300.0 729.3,300.0" fill="none" stroke="#3b82f6" stroke-width="1.8"/>
  <polyline points="60.7,300.0 63.7,300.0 66.8,300.0 69.8,300.0 72.9,300.0 75.9,300.0 79.0,300.0 82.0,300.0 85.1,300.0 88.1,300.0 91.2,300.0 94.3,300.0 97.3,300.0 100.4,300.0 103.4,300.0 106.5,300.0 109.5,300.0 112.6,300.0 115.6,300.0 118.7,300.0 121.7,300.0 124.8,300.0 127.8,300.0 130.9,300.0 133.9,300.0 137.0,300.0 140.1,300.0 143.1,300.0 146.2,300.0 149.2,300.0 152.3,300.0 155.3,300.0 158.4,300.0 161.4,300.0 164.5,300.0 167.5,300.0 170.6,300.0 173.6,300.0 176.7,300.0 179.7,300.0 182.8,300.0 185.9,300.0 188.9,300.0 192.0,300.0 195.0,300.0 198.1,300.0 201.1,300.0 204.2,300.0 207.2,300.0 210.3,300.0 213.3,300.0 216.4,300.0 219.4,300.0 222.5,300.0 225.5,300.0 228.6,300.0 231.7,300.0 234.7,300.0 237.8,300.0 240.8,300.0 243.9,300.0 246.9,300.0 250.0,300.0 253.0,300.0 256.1,300.0 259.1,300.0 262.2,300.0 265.2,300.0 268.3,300.0 271.3,300.0 274.4,300.0 277.5,300.0 280.5,300.0 283.6,300.0 286.6,300.0 289.7,300.0 292.7,300.0 295.8,300.0 298.8,300.0 301.9,300.0 304.9,300.0 308.0,300.0 311.0,300.0 314.1,300.0 317.1,300.0 320.2,300.0 323.2,300.0 326.3,300.0 329.4,300.0 332.4,300.0 335.5,300.0 338.5,300.0 341.6,300.0 344.6,300.0 347.7,300.0 350.7,299.9 353.8,299.9 356.8,299.9 359.9,299.8 362.9,299.8 366.0,299.7 369.0,299.6 372.1,299.5 375.2,299.3 378.2,299.1 381.3,298.8 384.3,298.4 387.4,298.0 390.4,297.4 393.5,296.7 396.5,295.8 399.6,294.7 402.6,293.3 405.7,291.7 408.7,289.8 411.8,287.5 414.8,284.7 417.9,281.5 421.0,277.8 424.0,273.5 427.1,268.6 430.1,263.0 433.2,256.7 436.2,249.7 439.3,241.9 442.3,233.4 445.4,224.1 448.4,214.0 451.5,203.3 454.5,192.0 457.6,180.1 460.6,167.8 463.7,155.3 466.8,142.6 469.8,130.0 472.9,117.6 475.9,105.6 479.0,94.3 482.0,83.9 485.1,74.6 488.1,66.5 491.2,59.8 494.2,54.8 497.3,51.5 500.3,50.0 503.4,50.4 506.4,52.7 509.5,56.9 512.5,62.8 515.6,70.5 518.7,79.7 521.7,90.3 524.8,102.0 527.8,114.7 530.9,128.0 533.9,141.8 537.0,155.8 540.0,169.8 543.1,183.5 546.1,196.8 549.2,209.4 552.2,221.3 555.3,232.4 558.3,242.5 561.4,251.6 564.5,259.7 567.5,266.9 570.6,273.1 573.6,278.4 576.7,282.9 579.7,286.6 582.8,289.6 585.8,292.1 588.9,294.1 591.9,295.6 595.0,296.8 598.0,297.7 601.1,298.4 604.1,298.9 607.2,299.2 610.3,299.5 613.3,299.7 616.4,299.8 619.4,299.9 622.5,299.9 625.5,300.0 628.6,300.0 631.6,300.0 634.7,300.0 637.7,300.0 640.8,300.0 643.8,300.0 646.9,300.0 649.9,300.0 653.0,300.0 656.1,300.0 659.1,300.0 662.2,300.0 665.2,300.0 668.3,300.0 671.3,300.0 674.4,300.0 677.4,300.0 680.5,300.0 683.5,300.0 686.6,300.0 689.6,300.0 692.7,300.0 695.7,300.0 698.8,300.0 701.9,300.0 704.9,300.0 708.0,300.0 711.0,300.0 714.1,300.0 717.1,300.0 720.2,300.0 723.2,300.0 726.3,300.0 729.3,300.0" fill="none" stroke="#10b981" stroke-width="2.2"/>
  <line x1="540.00" y1="34.00" x2="570.00" y2="34.00" stroke="currentColor" stroke-width="2.2" stroke-dasharray="5,4"/>
  <text x="578.00" y="38.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">prior  Beta(2,2)</text>
  <line x1="540.00" y1="52.00" x2="570.00" y2="52.00" stroke="#f59e0b" stroke-width="2.2"/>
  <text x="578.00" y="56.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">n=2, s=2  ->  Beta(4,2)</text>
  <line x1="540.00" y1="70.00" x2="570.00" y2="70.00" stroke="#ef4444" stroke-width="2.2"/>
  <text x="578.00" y="74.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">n=5, s=4  ->  Beta(6,3)</text>
  <line x1="540.00" y1="88.00" x2="570.00" y2="88.00" stroke="#3b82f6" stroke-width="2.2"/>
  <text x="578.00" y="92.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">n=20, s=14  ->  Beta(16,8)</text>
  <line x1="540.00" y1="106.00" x2="570.00" y2="106.00" stroke="#10b981" stroke-width="2.2"/>
  <text x="578.00" y="110.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">n=80, s=53  ->  Beta(55,29)</text>
  <line x1="495.50" y1="300.00" x2="495.50" y2="70.00" stroke="currentColor" stroke-width="1.0" stroke-dasharray="2,3"/>
  <text x="495.50" y="64.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">true theta = 0.65</text>
  <text x="380.0" y="380" font-size="13" fill="currentColor" text-anchor="middle">Figure 10.1  Sequential Beta-Bernoulli updating, prior Beta(2,2), data simulated with</text>
  <text x="380.0" y="397" font-size="13" fill="currentColor" text-anchor="middle">theta = 0.65 (seed 2). Each new batch of coin flips concentrates the posterior; the prior's</text>
  <text x="380.0" y="414" font-size="13" fill="currentColor" text-anchor="middle">influence shrinks like 4/(4+n), and by n = 80 the posterior is nearly a Gaussian bump of</text>
  <text x="380.0" y="431" font-size="13" fill="currentColor" text-anchor="middle">width about 0.05 around 0.65. Posterior means: 0.500, 0.667, 0.667, 0.667, 0.655.</text>
</svg>


### 10.2.3 Dirichlet–Categorical: where smoothing comes from

**Proposition 10.12 (Dirichlet–Categorical).** Let $X_1,\dots,X_n\mid\theta\overset{iid}{\sim}\mathrm{Categorical}(\theta)$ on $\{1,\dots,K\}$, $\theta\in\Delta_{K-1}$, with $\theta\sim\mathrm{Dirichlet}(\alpha_1,\dots,\alpha_K)$, $\alpha_k>0$, and let $c_k=\#\{i:x_i=k\}$. Then $\theta\mid D\sim\mathrm{Dirichlet}(\alpha_1+c_1,\dots,\alpha_K+c_K)$, and
$$
\mathbb{P}(\tilde X=k\mid D)=\frac{\alpha_k+c_k}{\alpha_0+n},\qquad \alpha_0:=\sum_j\alpha_j .
$$
Marginally, $\theta_k\mid D\sim\mathrm{Beta}(\alpha_k+c_k,\ \alpha_0+n-\alpha_k-c_k)$.

*Proof.* The Dirichlet density on the simplex is $\propto\prod_k\theta_k^{\alpha_k-1}$ and the likelihood is $\prod_k\theta_k^{c_k}$, so the posterior is $\propto\prod_k\theta_k^{\alpha_k+c_k-1}$: a Dirichlet with the stated parameters. The predictive is $\mathbb{E}[\theta_k\mid D]$, and the Dirichlet mean is $\alpha_k/\alpha_0$ componentwise; the marginal is the standard aggregation property of the Dirichlet (any partition of coordinates is Dirichlet with summed parameters, here $\{k\}$ against the rest). $\blacksquare$

**Worked Example 10.5 (additive smoothing is a posterior predictive).** Four categories, counts $c=(12,3,0,5)$, $n=20$. The MLE is $(0.60,0.15,0.00,0.25)$ and assigns probability *zero* to category 3, so a single occurrence of category 3 in a test set drives cross-entropy to $+\infty$. With $\mathrm{Dirichlet}(1,1,1,1)$ the posterior is $\mathrm{Dirichlet}(13,4,1,6)$ and the predictive is
$$
(0.541667,\ 0.166667,\ 0.041667,\ 0.250000),
$$
which is Laplace's add-one rule. With $\mathrm{Dirichlet}(0.5,\dots,0.5)$ the predictive is $(0.568182,0.159091,0.022727,0.250000)$ — the unseen category gets $0.0227$ instead of $0.0417$. The posterior sd of $\theta_1$ is $0.0997$, so with $n=20$ we genuinely do not know $\theta_1$ to better than $\pm0.2$.

Every additive-smoothing constant you have ever tuned is a Dirichlet hyperparameter, i.e. a pseudo-count, and the "$\alpha_0$ pseudo-observations" reading tells you how to set it: $\alpha_k=\alpha_0 q_k$ pulls the predictive toward a base distribution $q$ with strength $\alpha_0/(\alpha_0+n)$. Label smoothing in classification is the same operation applied to the *target* rather than to the estimated distribution, and Kneser–Ney and other backoff schemes in language modelling are hierarchical versions (§10.4) of it.

### 10.2.4 Gaussian–Gaussian, known variance

**Proposition 10.13 (Gaussian mean, known variance).** Let $X_1,\dots,X_n\mid\mu\overset{iid}{\sim}N(\mu,\sigma^2)$ with $\sigma^2$ known, and $\mu\sim N(\mu_0,\tau^2)$. Then $\mu\mid D\sim N(\mu_n,\tau_n^2)$ with
$$
\frac{1}{\tau_n^2}=\frac{1}{\tau^2}+\frac{n}{\sigma^2},\qquad
\mu_n=\tau_n^2\Big(\frac{\mu_0}{\tau^2}+\frac{n\bar x}{\sigma^2}\Big),
$$
and the posterior predictive is $\tilde X\mid D\sim N(\mu_n,\ \tau_n^2+\sigma^2)$.

*Proof.* Work with $\log$ densities and keep only $\mu$-dependent terms:
$$
-\frac{1}{2}\Big[\frac{\sum_i(x_i-\mu)^2}{\sigma^2}+\frac{(\mu-\mu_0)^2}{\tau^2}\Big]
=-\frac{1}{2}\Big[\Big(\frac{n}{\sigma^2}+\frac{1}{\tau^2}\Big)\mu^2-2\Big(\frac{n\bar x}{\sigma^2}+\frac{\mu_0}{\tau^2}\Big)\mu\Big]+\text{const}.
$$
Completing the square (Ch. 5) gives a quadratic $-\frac{1}{2\tau_n^2}(\mu-\mu_n)^2$ with the stated $\tau_n^2,\mu_n$; a density proportional to $\exp$ of that is $N(\mu_n,\tau_n^2)$. For the predictive, $\tilde X=\mu+\varepsilon$ with $\varepsilon\sim N(0,\sigma^2)$ independent of $\mu$ given $D$, and the sum of independent Gaussians is Gaussian with summed variances (Ch. 5). $\blacksquare$

Precisions add. That single sentence is worth memorizing; it is why Kalman filters, Gaussian belief propagation, and the "natural-gradient as precision-weighted average" reading of second-order optimization all look alike.

**Worked Example 10.6 (precision-weighted averaging).** $\sigma^2=4$ known, prior $N(0,1)$, and $n=10$ observations with $\bar x=1.5$. Posterior precision $1+10/4=3.5$, posterior mean $(0+2.5\cdot1.5)/3.5=1.0714$, posterior sd $0.5345$. The data carry weight $(n/\sigma^2)/3.5=0.7143$ and the prior $0.2857$. The predictive sd is $\sqrt{1/3.5+4}=2.0702$, only $3.5\%$ larger than the noise sd $2$: with $n=10$, epistemic uncertainty is already a minor part of predictive uncertainty here.

| $n$ | weight on data | posterior sd |
|---|---|---|
| 1 | 0.2000 | 0.8944 |
| 10 | 0.7143 | 0.5345 |
| 100 | 0.9615 | 0.1961 |
| 1000 | 0.9960 | 0.0631 |

The prior is worth $\sigma^2/\tau^2=4$ observations. Again: a pseudo-count.

### 10.2.5 Unknown variance, and where Student-$t$ comes from

**Proposition 10.14 (Normal-Inverse-Gamma).** Let $X_1,\dots,X_n\mid(\mu,\sigma^2)\overset{iid}{\sim}N(\mu,\sigma^2)$ and put the *joint* prior
$$
\sigma^2\sim\mathrm{Inv\text{-}Gamma}(\alpha_0,\beta_0),\qquad \mu\mid\sigma^2\sim N(\mu_0,\ \sigma^2/\kappa_0),
$$
written $\mathrm{NIG}(\mu_0,\kappa_0,\alpha_0,\beta_0)$. Here $\mathrm{Inv\text{-}Gamma}(\alpha,\beta)$ has density $\propto (\sigma^2)^{-\alpha-1}e^{-\beta/\sigma^2}$, mean $\beta/(\alpha-1)$ for $\alpha>1$; equivalently $1/\sigma^2\sim\mathrm{Gamma}(\alpha,\beta)$ in shape–rate form. Note that the prior on $\mu$ is scaled by $\sigma^2$: this coupling is what makes the pair conjugate, and Exercise A.4 shows conjugacy is lost without it. With $\bar x$ the sample mean and $S=\sum_i(x_i-\bar x)^2$, the posterior is $\mathrm{NIG}(\mu_n,\kappa_n,\alpha_n,\beta_n)$ with
$$
\kappa_n=\kappa_0+n,\quad \mu_n=\frac{\kappa_0\mu_0+n\bar x}{\kappa_n},\quad
\alpha_n=\alpha_0+\frac n2,\quad
\beta_n=\beta_0+\frac{S}{2}+\frac{\kappa_0 n(\bar x-\mu_0)^2}{2\kappa_n}.
$$
Moreover the marginal posterior of $\mu$ and the posterior predictive are Student-$t$:
$$
\mu\mid D\sim t_{2\alpha_n}\!\Big(\mu_n,\ \frac{\beta_n}{\alpha_n\kappa_n}\Big),\qquad
\tilde X\mid D\sim t_{2\alpha_n}\!\Big(\mu_n,\ \frac{\beta_n(\kappa_n+1)}{\alpha_n\kappa_n}\Big),
$$
where $t_\nu(m,s^2)$ denotes location $m$, squared scale $s^2$, $\nu$ degrees of freedom.

*Proof sketch (full algebra in the solution to Exercise A.4).* Write the joint density as $(\sigma^2)^{-\alpha_0-1-1/2}\exp\{-[\beta_0+\kappa_0(\mu-\mu_0)^2/2]/\sigma^2\}$ times the likelihood $(\sigma^2)^{-n/2}\exp\{-[S+n(\mu-\bar x)^2]/2\sigma^2\}$. Inside the exponent, $\kappa_0(\mu-\mu_0)^2+n(\mu-\bar x)^2=\kappa_n(\mu-\mu_n)^2+\kappa_0n(\bar x-\mu_0)^2/\kappa_n$ by completing the square, which yields the stated updates by matching the NIG form. Integrating out $\sigma^2$ from an NIG gives a $t$: with $u=1/\sigma^2$ one recognizes a Gamma integral, $\int_0^\infty u^{\alpha_n-1/2}e^{-u[\beta_n+\kappa_n(\mu-\mu_n)^2/2]}\mathrm{d}u\propto[\beta_n+\kappa_n(\mu-\mu_n)^2/2]^{-(\alpha_n+1/2)}$, which is the $t_{2\alpha_n}$ kernel with the stated scale. The predictive follows the same route with the extra observation's variance $\sigma^2(1+1/\kappa_n)$. $\blacksquare$

**Worked Example 10.7 (six measurements, and why $t$ instead of $z$).** Data $x=(2.1,1.8,2.5,2.0,2.3,1.7)$, so $n=6$, $\bar x=2.066667$, $S=0.453333$, $s^2=0.090667$. Prior $\mathrm{NIG}(\mu_0{=}2,\kappa_0{=}1,\alpha_0{=}2,\beta_0{=}0.5)$. The updates give $\mu_n=2.057143$, $\kappa_n=7$, $\alpha_n=5$, $\beta_n=0.728571$, hence $\mathbb{E}[\sigma^2\mid D]=\beta_n/(\alpha_n-1)=0.182143$ (so $\mathbb{E}[\sigma^2]^{1/2}=0.4268$). The marginal posterior for $\mu$ is $t_{10}(2.057143,\ \text{scale }0.144279)$ with 95% credible interval $[1.7357,\,2.3786]$. The predictive is $t_{10}(2.057143,\ \text{scale }0.408082)$ with 95% interval $[1.1479,\,2.9664]$, width $1.8185$; using a Gaussian with the same scale would give width $1.5997$, i.e. $12.0\%$ too narrow. In the tails the gap is larger: $P(|T_{10}|>3)=0.013344$ against $0.002700$ for a Gaussian, a factor of $4.9$.

That factor is the entire reason robust ML losses exist. When you fit a Gaussian likelihood with a plug-in variance you are asserting that a $3\sigma$ residual happens $0.27\%$ of the time; integrating over your uncertainty about $\sigma^2$ says $1.3\%$. Heavy-tailed predictive distributions arise *automatically* from parameter uncertainty, without positing heavy-tailed noise.

### 10.2.6 Gamma–Poisson and the Negative Binomial

**Proposition 10.15 (Gamma–Poisson).** Let $X_1,\dots,X_n\mid\lambda\overset{iid}{\sim}\mathrm{Poisson}(\lambda)$ with $\lambda\sim\mathrm{Gamma}(\alpha_0,\beta_0)$ (shape, rate). With $t=\sum_ix_i$, $\lambda\mid D\sim\mathrm{Gamma}(\alpha_0+t,\ \beta_0+n)$, and the predictive for one new count is Negative Binomial:
$$
p(\tilde x=k\mid D)=\frac{\Gamma(k+\alpha_n)}{k!\,\Gamma(\alpha_n)}\Big(\frac{\beta_n}{\beta_n+1}\Big)^{\alpha_n}\Big(\frac{1}{\beta_n+1}\Big)^{k},
$$
with mean $\alpha_n/\beta_n$ and variance $\frac{\alpha_n}{\beta_n}\big(1+\frac1{\beta_n}\big)$. (The $\Gamma$-ratio is written rather than a binomial coefficient because $\alpha_n=\alpha_0+t$ need not be an integer; when it is, the two agree.)

*Proof.* $\pi(\lambda\mid D)\propto\lambda^{\alpha_0-1}e^{-\beta_0\lambda}\cdot\lambda^{t}e^{-n\lambda}=\lambda^{\alpha_0+t-1}e^{-(\beta_0+n)\lambda}$, the Gamma kernel. For the predictive, integrate: $\int_0^\infty\frac{\lambda^k e^{-\lambda}}{k!}\frac{\beta_n^{\alpha_n}}{\Gamma(\alpha_n)}\lambda^{\alpha_n-1}e^{-\beta_n\lambda}\mathrm{d}\lambda=\frac{\Gamma(k+\alpha_n)}{k!\,\Gamma(\alpha_n)}\frac{\beta_n^{\alpha_n}}{(1+\beta_n)^{k+\alpha_n}}$, which is the stated NB pmf. Its moments follow from Proposition 10.7: mean $\mathbb{E}[\lambda\mid D]=\alpha_n/\beta_n$ and variance $\mathbb{E}[\lambda\mid D]+\mathrm{Var}(\lambda\mid D)=\alpha_n/\beta_n+\alpha_n/\beta_n^2$. $\blacksquare$

**Worked Example 10.8 (over-dispersion, exactly).** Prior $\mathrm{Gamma}(2,1)$, counts $(1,0,3,2,1)$ so $n=5$, $t=7$. Posterior $\mathrm{Gamma}(9,6)$: mean $1.5$, sd $0.5$; the MLE is $1.4$. Predictive $\mathrm{NB}$ with mean $1.5$ and variance $1.75$, versus $\mathrm{Poisson}(1.5)$ with variance $1.5$ — an over-dispersion factor of $1+1/\beta_n=7/6=1.1667$.

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| predictive NB | 0.2497 | 0.3211 | 0.2293 | 0.1201 | 0.0515 | 0.0191 |
| plug-in Poisson(1.5) | 0.2231 | 0.3347 | 0.2510 | 0.1255 | 0.0471 | 0.0141 |

$P(\tilde X\ge5)$ is $0.0282$ under the predictive and $0.0186$ under the plug-in: the plug-in misses a third of that tail mass, or equivalently the correct predictive tail is $52\%$ heavier. The Poisson-to-Negative-Binomial upgrade that count-regression practitioners reach for when they see over-dispersion is, in the Bayesian reading, not a new model but the correct predictive distribution of the old one.

### 10.2.7 One theorem behind all of them

**Theorem 10.16 (conjugate prior for an exponential family; hyperparameters as pseudo-counts).** Let the model be a natural exponential family in canonical form (Ch. 3),
$$
p(x\mid\eta)=h(x)\exp\big(\langle\eta,T(x)\rangle-A(\eta)\big),\qquad \eta\in\mathcal{N}\subseteq\mathbb{R}^k,
$$
with $A$ the log-partition function and $\mathcal{N}$ the natural parameter space. For $\nu>0$ and $\chi\in\mathbb{R}^k$ define
$$
\pi_{\chi,\nu}(\eta)\;:=\;\frac{1}{C(\chi,\nu)}\exp\big(\langle\eta,\chi\rangle-\nu A(\eta)\big)\mathbf{1}\{\eta\in\mathcal{N}\},\qquad
C(\chi,\nu):=\int_{\mathcal{N}}e^{\langle\eta,\chi\rangle-\nu A(\eta)}\mathrm{d}\eta,
$$
and suppose $(\chi,\nu)$ is such that $C(\chi,\nu)<\infty$. Then:

(i) *(Conjugacy.)* Given $x_1,\dots,x_n$ i.i.d. from the family, the posterior is $\pi_{\chi',\nu'}$ with
$$
\chi'=\chi+\sum_{i=1}^n T(x_i),\qquad \nu'=\nu+n .
$$
So $\nu$ is a *pseudo-sample-size* and $\chi$ is a vector of *pseudo-sufficient-statistics*: the prior behaves exactly like $\nu$ previously observed data points whose sufficient statistics summed to $\chi$.

(ii) *(Prior mean of the sufficient statistic.)* If $(\chi,\nu)$ is in the interior of the region where $C<\infty$, then
$$
\mathbb{E}\big[\nabla A(\eta)\big]=\frac{\chi}{\nu},
$$
i.e. the prior asserts a mean sufficient statistic of $\chi/\nu$, and the posterior asserts $\frac{\chi+\sum_iT(x_i)}{\nu+n}$ — a convex combination of the prior guess $\chi/\nu$ and the empirical average $\frac1n\sum_iT(x_i)$ with weights $\nu/(\nu+n)$ and $n/(\nu+n)$.

(iii) *(Evidence.)* $Z(D)=\Big(\prod_i h(x_i)\Big)\dfrac{C(\chi+\sum_iT(x_i),\ \nu+n)}{C(\chi,\nu)}$.

*Proof.* (i) Multiply:
$$
\pi(\eta\mid D)\propto e^{\langle\eta,\chi\rangle-\nu A(\eta)}\prod_i h(x_i)e^{\langle\eta,T(x_i)\rangle-A(\eta)}
\propto \exp\Big(\big\langle\eta,\ \chi+\textstyle\sum_iT(x_i)\big\rangle-(\nu+n)A(\eta)\Big),
$$
which is $\pi_{\chi',\nu'}$ up to normalization; the $h(x_i)$ do not involve $\eta$. Normalizing gives (iii) as well, since the constant that makes the left side a density is $Z(D)$ and the one on the right is $C(\chi',\nu')$.

(ii) Differentiate $\log C$ in $\chi$: since $C(\chi,\nu)=\int e^{\langle\eta,\chi\rangle-\nu A(\eta)}\mathrm{d}\eta$ and the integrand is dominated locally, differentiation under the integral sign is licensed by the Dominated Convergence Theorem (Ch. 2) on the interior, giving $\nabla_\chi\log C=\mathbb{E}[\eta]$ and $\partial_\nu\log C=-\mathbb{E}[A(\eta)]$. Now integrate the identity $\nabla_\eta\big(e^{\langle\eta,\chi\rangle-\nu A(\eta)}\big)=(\chi-\nu\nabla A(\eta))e^{\langle\eta,\chi\rangle-\nu A(\eta)}$ over $\mathcal{N}$. If $(\chi,\nu)$ is interior then the integrand vanishes on the boundary of $\mathcal{N}$ (this is exactly the integrability condition), so the left side integrates to zero by the divergence theorem, giving $\chi-\nu\,\mathbb{E}[\nabla A(\eta)]=0$. $\blacksquare$

**Worked Example 10.9 (reading the pseudo-counts off each family).** Apply Theorem 10.16 to the four cases above.

| model | $T(x)$ | conjugate prior | $\chi$ | $\nu$ | pseudo-data reading |
|---|---|---|---|---|---|
| Bernoulli | $x$ | $\mathrm{Beta}(a_0,b_0)$ | $a_0$ | $a_0+b_0$ | $a_0$ heads out of $a_0+b_0$ flips |
| Categorical | one-hot $x$ | $\mathrm{Dirichlet}(\alpha)$ | $\alpha$ | $\alpha_0$ | $\alpha_k$ occurrences of class $k$ |
| Poisson | $x$ | $\mathrm{Gamma}(\alpha_0,\beta_0)$ | $\alpha_0$ | $\beta_0$ | $\alpha_0$ events in $\beta_0$ exposure units |
| $N(\mu,\sigma^2)$, $\sigma^2$ known | $x/\sigma^2$ | $N(\mu_0,\sigma^2/\kappa_0)$ | $\kappa_0\mu_0/\sigma^2$ | $\kappa_0$ | $\kappa_0$ observations with mean $\mu_0$ |
| $N(\mu,\sigma^2)$, both unknown | $(x,x^2)$ | $\mathrm{NIG}(\mu_0,\kappa_0,\alpha_0,\beta_0)$ | — | $\kappa_0$ (mean), $2\alpha_0$ (variance) | $\kappa_0$ pseudo-obs for $\mu$, $2\alpha_0$ for $\sigma^2$ |

Concretely with Worked Example 10.7's prior: $\kappa_0=1$ means the prior on $\mu$ is worth one measurement, and $\alpha_0=2$ means the prior on $\sigma^2$ is worth four. Against $n=6$ real measurements that is not negligible, which is why the posterior mean $2.0571$ sits noticeably below $\bar x=2.0667$.

Theorem 10.16(ii) also explains the "convex combination" structure we kept observing: it is not a series of coincidences but one statement about mean sufficient statistics.

**Corollary 10.17 (predictive distributions are over-dispersed).** Write $\hat\theta:=\mathbb{E}[\theta\mid D]$ and let $\tilde X$ have posterior predictive variance $V:=\mathrm{Var}(\tilde X\mid D)$, to be compared with the plug-in variance $v(\hat\theta)$.

(i) If $v(\theta)\equiv v$ is constant in $\theta$, then $V=v+\mathrm{Var}(\mu(\theta)\mid D)>v(\hat\theta)=v$ unless $\mu(\theta)$ is posterior-a.s. constant.

(ii) If $v$ is convex, then $V\ge v(\hat\theta)+\mathrm{Var}(\mu(\theta)\mid D)\ge v(\hat\theta)$, with strict inequality unless both $v$ is affine on the posterior's support and $\mu(\theta)$ is posterior-a.s. constant.

(iii) If $v$ is concave the two effects compete and "the predictive is always wider" is false. The sharpest instance is a *single* Bernoulli trial: with $\mu(\theta)=\theta$ and $v(\theta)=\theta(1-\theta)$,
$$
\mathbb{E}[v(\theta)\mid D]+\mathrm{Var}(\theta\mid D)=\big(\hat\theta-\mathbb{E}[\theta^2\mid D]\big)+\big(\mathbb{E}[\theta^2\mid D]-\hat\theta^2\big)=\hat\theta(1-\hat\theta)=v(\hat\theta),
$$
exactly — the concavity loss cancels the epistemic gain, which it must, since the predictive for one trial *is* $\mathrm{Bernoulli}(\hat\theta)$. Over-dispersion reappears as soon as you predict $m\ge2$ trials, where the Beta-Binomial-to-Binomial variance ratio is $\frac{a_n+b_n+m}{a_n+b_n+1}$ — equal to $1$ at $m=1$ and $29/25=1.16$ at $m=5$, $a_n+b_n=24$, reproducing Worked Example 10.2.

*Proof.* (i) is Proposition 10.7. (ii) is Jensen's inequality (Ch. 2), $\mathbb{E}[v(\theta)\mid D]\ge v(\hat\theta)$, plus nonnegativity of the epistemic term. (iii) Jensen reverses for concave $v$, so $\mathbb{E}[v(\theta)\mid D]\le v(\hat\theta)$ and the sign of $V-v(\hat\theta)$ depends on which term is larger; the two displays are direct computations, the second from Proposition 10.11(iv). $\blacksquare$

> **Common Confusion 10.3.** *"Conjugate priors are chosen because they're the right priors, aren't they? Otherwise why would every textbook use them?"*
>
> They are chosen because the integral has a closed form. That is a computational virtue, not an epistemic one. A $\mathrm{Beta}(a,b)$ prior is unimodal on $(0,1)$ and cannot express "the coin is either strongly biased to heads or strongly biased to tails, but certainly not fair" — a bimodal belief that is perfectly reasonable for a trick coin, and that no Beta can represent. The mechanical form of Theorem 10.16 makes the trade-off visible: conjugate priors are exactly the family whose log-density is *linear in the natural parameter plus a multiple of $A(\eta)$*, which is a strong shape restriction. In modern practice MCMC and variational inference (Ch. 13) remove the need for conjugacy, and the honest use of conjugate models is (a) as building blocks inside larger samplers, (b) as fast approximations in latency-critical systems such as Thompson sampling in production bandits, and (c) as sanity checks — you should always test an approximate inference algorithm on a conjugate model where the truth is known.

> **Self-check 10.2.**
> 1. A colleague uses $\mathrm{Beta}(100,100)$ as a "weak" prior on a click-through rate and has $n=200$ impressions. Quantify the objection in one sentence.
> 2. In Proposition 10.15, what happens to the over-dispersion factor $1+1/\beta_n$ as $n\to\infty$, and what does that say about the predictive?
> 3. Theorem 10.16 requires $C(\chi,\nu)<\infty$. Where in the Beta case does that condition show up?
> 4. Why does the Student-$t$ appear in Proposition 10.14 but not in Proposition 10.13?

## 10.3 Choosing priors

### 10.3.1 Three honest categories

Priors come in three flavours, and it helps to name them plainly.

An *informative* prior encodes real knowledge: previous experiments, physical constraints, the fact that click-through rates in your product have historically lived between 1% and 5%. It is the least fashionable and often the most defensible choice, because it is checkable — you can say where the numbers came from.

A *weakly informative* prior encodes only scale and shape: "this coefficient is probably between $-10$ and $10$, and definitely not $10^6$." It regularizes without pretending to know the answer. In ML this is the overwhelmingly common case; $N(0,\tau^2 I)$ on weights is a weakly informative prior whose only claim is that weights are $O(\tau)$.

An "*uninformative*" prior claims to add no information. The scare quotes are the point of this section: there is no such thing, and the reason is measure-theoretic rather than philosophical. A prior is a measure on $\Theta$. Measures are not invariant under reparameterization — they pick up Jacobians (Ch. 3) — so uniformity is a property of a coordinate system, not of a belief.

Whatever flavour you choose, one rule is absolute, and it is worth naming before any recipes. **Cromwell's rule (a remark, not a numbered result).** If the prior assigns an event probability zero, so does every posterior: $\Pi(A)=0$ implies $\Pi(A\mid D)=0$ for every dataset $D$, because the posterior $\Pi(\cdot\mid D)$ is absolutely continuous with respect to the prior — its density $p(D\mid\theta)/Z(D)$ multiplies prior mass and cannot create mass where there is none. (One line from Definition 10.2, but the consequences are permanent.) Zero is therefore not a strong opinion; it is an unrecoverable one, and no volume of contrary evidence will move it. The rule cuts in both directions in this chapter: it is why the spike-and-slab prior of §10.3.5 must place a genuine atom at zero if you ever want the posterior to *say* zero, and it is why a prior with a hard support restriction — a truncated prior that excludes the truth — defeats Bernstein–von Mises no matter how large $n$ grows (the failure exhibited numerically in the solution to Exercise A.2). Reserve probability zero for logical impossibilities, and use small-but-positive for everything you merely disbelieve.

### 10.3.2 A flat prior is not flat after you change variables

**Proposition 10.18 (priors transform as measures).** Let $\Theta\subseteq\mathbb{R}^d$ be open, let $g:\Theta\to\Phi\subseteq\mathbb{R}^d$ be a $C^1$ diffeomorphism, and let $\theta$ have prior density $\pi_\theta$ with respect to $\lambda_d$. Then $\phi:=g(\theta)$ has prior density
$$
\pi_\phi(\phi)=\pi_\theta\big(g^{-1}(\phi)\big)\,\big|\det J_{g^{-1}}(\phi)\big|,
\qquad J_{g^{-1}}(\phi)=\frac{\partial g^{-1}}{\partial\phi}(\phi),
$$
and consequently: (i) $\pi_\theta\equiv c$ constant does *not* imply $\pi_\phi$ constant, unless $|\det J_{g^{-1}}|$ is constant; (ii) the induced prior probability of any *event* is unchanged, $\Pi_\phi(g(B))=\Pi_\theta(B)$.

*Proof.* This is exactly the change-of-variables formula for pushforward measures (Ch. 3) applied to $\Pi_\phi=g_\#\Pi_\theta$; (ii) is the definition of pushforward (Definition 1.16). (i) follows by inspection of the displayed formula. $\blacksquare$

Part (ii) is the reassuring half: nothing incoherent happens. Your beliefs about *events* are coordinate-free. Part (i) is the trap: "I chose a flat prior so as not to favour any value" is a statement about a density, and densities are not coordinate-free. The moment you reparameterize — and ML reparameterizes constantly, working with logits instead of probabilities, log-variances instead of variances, unconstrained matrices instead of covariances — a flat prior turns into an opinionated one.

**Worked Example 10.10 (flat on $\theta$ versus flat on the log-odds).** Let $\theta\sim\mathrm{Uniform}(0,1)$ and $\phi=\mathrm{logit}(\theta)=\log\frac{\theta}{1-\theta}$. Then $\theta=g^{-1}(\phi)=\sigma(\phi)$ and $\mathrm{d}\theta/\mathrm{d}\phi=\sigma(\phi)(1-\sigma(\phi))$, so
$$
\pi_\phi(\phi)=\frac{e^{-\phi}}{(1+e^{-\phi})^2},
$$
the standard logistic density. Its value at $\phi=0$ is $0.250000$; at $\phi=3$ it is $0.045177$; at $\phi=6$ it is $0.002467$ — a factor of $101$ between the middle and $\phi=6$. Under this induced prior, $P(|\phi|\le1)=0.462117$ and $P(-6\le\phi\le-3)=0.044953$. "Flat on $\theta$" is therefore a strong statement on the logit scale: it says extreme log-odds are very unlikely.

Run it the other way. Take $\phi\sim\mathrm{Uniform}(-6,6)$, a common "vague prior on the logit". Then
$$
P(\theta<0.1)=P\big(\phi<\mathrm{logit}(0.1)\big)=\frac{\log(1/9)+6}{12}=0.316898,
$$
whereas $\mathrm{Uniform}(0,1)$ on $\theta$ gives $0.100000$. Two priors, each "uninformative" in its own coordinates, disagree by a factor of $3.2$ about a perfectly concrete event.

The same phenomenon appears in every variance parameterization you will meet. If $\sigma\sim\mathrm{Uniform}(0,1)$ then $v=\sigma^2$ has density $1/(2\sqrt v)$, which puts mass $0.1$ on $v<0.01$ and diverges at $0$; a flat prior on $\log\sigma$ is scale-invariant but improper; a flat prior on the precision $1/\sigma^2$ is yet another distinct claim. In a variational autoencoder the reported "unit Gaussian prior on the latent" combined with a learned decoder is a prior on *functions* whose flatness in weight space says nothing at all about flatness in function space, which is one reason VAE priors are so often mismatched to the aggregate posterior.

Figure 10.2 makes the transformation concrete.

<svg viewBox="0 0 760 420" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="sans-serif">
  <line x1="55.00" y1="270.00" x2="340.00" y2="270.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="55.00" y1="270.00" x2="55.00" y2="60.00" stroke="currentColor" stroke-width="1.2"/>
  <polyline points="55.0,160.0 340.0,160.0" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
  <line x1="55.00" y1="160.00" x2="55.00" y2="270.00" stroke="#3b82f6" stroke-width="1.2"/>
  <line x1="340.00" y1="160.00" x2="340.00" y2="270.00" stroke="#3b82f6" stroke-width="1.2"/>
  <text x="197.50" y="38.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">prior on theta:  Uniform(0,1)</text>
  <line x1="55.00" y1="270.00" x2="55.00" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="55.00" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.00</text>
  <line x1="126.25" y1="270.00" x2="126.25" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="126.25" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.25</text>
  <line x1="197.50" y1="270.00" x2="197.50" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="197.50" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.50</text>
  <line x1="268.75" y1="270.00" x2="268.75" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="268.75" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.75</text>
  <line x1="340.00" y1="270.00" x2="340.00" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="340.00" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">1.00</text>
  <text x="197.50" y="310.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">theta</text>
  <text x="49.00" y="160.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">1</text>
  <rect x="55.00" y="160.00" width="28.50" height="110.00" fill="#f59e0b" opacity="0.30"/>
  <text x="89.50" y="182.00" font-size="11" fill="#f59e0b" text-anchor="start" font-weight="normal">mass 0.100</text>
  <line x1="420.00" y1="270.00" x2="730.00" y2="270.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="420.00" y1="270.00" x2="420.00" y2="60.00" stroke="currentColor" stroke-width="1.2"/>
  <polyline points="420.0,268.3 421.4,268.2 422.8,268.1 424.2,268.0 425.7,267.9 427.1,267.7 428.5,267.6 429.9,267.5 431.3,267.3 432.7,267.2 434.2,267.0 435.6,266.9 437.0,266.7 438.4,266.5 439.8,266.3 441.2,266.1 442.6,265.9 444.1,265.7 445.5,265.4 446.9,265.2 448.3,264.9 449.7,264.6 451.1,264.3 452.6,264.0 454.0,263.7 455.4,263.3 456.8,262.9 458.2,262.5 459.6,262.1 461.1,261.7 462.5,261.2 463.9,260.8 465.3,260.3 466.7,259.7 468.1,259.2 469.5,258.6 471.0,258.0 472.4,257.3 473.8,256.6 475.2,255.9 476.6,255.1 478.0,254.3 479.5,253.5 480.9,252.6 482.3,251.7 483.7,250.7 485.1,249.7 486.5,248.6 487.9,247.5 489.4,246.3 490.8,245.1 492.2,243.8 493.6,242.4 495.0,241.0 496.4,239.5 497.9,238.0 499.3,236.4 500.7,234.7 502.1,232.9 503.5,231.1 504.9,229.1 506.3,227.1 507.8,225.1 509.2,222.9 510.6,220.6 512.0,218.3 513.4,215.9 514.8,213.4 516.3,210.8 517.7,208.1 519.1,205.3 520.5,202.5 521.9,199.5 523.3,196.5 524.7,193.4 526.2,190.2 527.6,186.9 529.0,183.6 530.4,180.2 531.8,176.8 533.2,173.2 534.7,169.7 536.1,166.1 537.5,162.4 538.9,158.7 540.3,155.1 541.7,151.4 543.2,147.7 544.6,144.0 546.0,140.4 547.4,136.8 548.8,133.2 550.2,129.8 551.6,126.4 553.1,123.1 554.5,120.0 555.9,116.9 557.3,114.0 558.7,111.3 560.1,108.7 561.6,106.3 563.0,104.2 564.4,102.2 565.8,100.4 567.2,98.9 568.6,97.6 570.0,96.6 571.5,95.8 572.9,95.3 574.3,95.0 575.7,95.0 577.1,95.3 578.5,95.8 580.0,96.6 581.4,97.6 582.8,98.9 584.2,100.4 585.6,102.2 587.0,104.2 588.4,106.3 589.9,108.7 591.3,111.3 592.7,114.0 594.1,116.9 595.5,120.0 596.9,123.1 598.4,126.4 599.8,129.8 601.2,133.2 602.6,136.8 604.0,140.4 605.4,144.0 606.8,147.7 608.3,151.4 609.7,155.1 611.1,158.7 612.5,162.4 613.9,166.1 615.3,169.7 616.8,173.2 618.2,176.8 619.6,180.2 621.0,183.6 622.4,186.9 623.8,190.2 625.3,193.4 626.7,196.5 628.1,199.5 629.5,202.5 630.9,205.3 632.3,208.1 633.7,210.8 635.2,213.4 636.6,215.9 638.0,218.3 639.4,220.6 640.8,222.9 642.2,225.1 643.7,227.1 645.1,229.1 646.5,231.1 647.9,232.9 649.3,234.7 650.7,236.4 652.1,238.0 653.6,239.5 655.0,241.0 656.4,242.4 657.8,243.8 659.2,245.1 660.6,246.3 662.1,247.5 663.5,248.6 664.9,249.7 666.3,250.7 667.7,251.7 669.1,252.6 670.5,253.5 672.0,254.3 673.4,255.1 674.8,255.9 676.2,256.6 677.6,257.3 679.0,258.0 680.5,258.6 681.9,259.2 683.3,259.7 684.7,260.3 686.1,260.8 687.5,261.2 688.9,261.7 690.4,262.1 691.8,262.5 693.2,262.9 694.6,263.3 696.0,263.7 697.4,264.0 698.9,264.3 700.3,264.6 701.7,264.9 703.1,265.2 704.5,265.4 705.9,265.7 707.4,265.9 708.8,266.1 710.2,266.3 711.6,266.5 713.0,266.7 714.4,266.9 715.8,267.0 717.3,267.2 718.7,267.3 720.1,267.5 721.5,267.6 722.9,267.7 724.3,267.9 725.8,268.0 727.2,268.1 728.6,268.2 730.0,268.3" fill="none" stroke="#3b82f6" stroke-width="2.5"/>
  <polyline points="420.0,211.7 730.0,211.7" fill="none" stroke="#10b981" stroke-width="2.0" stroke-dasharray="6,4"/>
  <text x="726.00" y="205.69" font-size="11" fill="#10b981" text-anchor="end" font-weight="normal">flat on phi (1/12)</text>
  <rect x="420.00" y="60.00" width="98.24" height="210.00" fill="#f59e0b" opacity="0.30"/>
  <text x="524.24" y="76.00" font-size="11" fill="#f59e0b" text-anchor="start" font-weight="normal">same event, theta &lt; 0.1</text>
  <text x="524.24" y="92.00" font-size="11" fill="#f59e0b" text-anchor="start" font-weight="normal">mass 0.100 (not 0.317)</text>
  <line x1="420.00" y1="270.00" x2="420.00" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="420.00" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">-6</text>
  <line x1="497.50" y1="270.00" x2="497.50" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="497.50" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">-3</text>
  <line x1="575.00" y1="270.00" x2="575.00" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="575.00" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0</text>
  <line x1="652.50" y1="270.00" x2="652.50" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="652.50" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">3</text>
  <line x1="730.00" y1="270.00" x2="730.00" y2="275.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="730.00" y="289.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">6</text>
  <text x="575.00" y="310.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">phi = log(theta/(1-theta))</text>
  <text x="575.00" y="38.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">induced prior on phi: logistic density</text>
  <path d="M 352 165 Q 385 140 408 165" fill="none" stroke="currentColor" stroke-width="1.4"/>
  <text x="380.00" y="132.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">phi = logit(theta)</text>
  <text x="380.0" y="350" font-size="13" fill="currentColor" text-anchor="middle">Figure 10.2  A flat prior is a claim, not the absence of one. Uniform(0,1) on theta pushes</text>
  <text x="380.0" y="367" font-size="13" fill="currentColor" text-anchor="middle">forward (Ch. 3 change of variables) to the logistic density on the log-odds scale phi, which is</text>
  <text x="380.0" y="384" font-size="13" fill="currentColor" text-anchor="middle">far from flat: it puts density 0.250 at phi = 0 and 0.0025 at phi = 6. Conversely a flat prior on</text>
  <text x="380.0" y="401" font-size="13" fill="currentColor" text-anchor="middle">phi over [-6,6] assigns probability 0.317 to theta < 0.1, where Uniform(0,1) assigns 0.100.</text>
</svg>


### 10.3.3 The Jeffreys prior and its invariance

If flatness is coordinate-dependent, is there a rule for choosing a prior that is *not*? Jeffreys' answer: build the prior out of the model's own geometry. Fisher information (Ch. 9) transforms like a metric tensor, and the square root of the determinant of a metric is exactly the volume element it induces — so using it as a density gives a coordinate-free recipe.

**Definition 10.19 (Jeffreys prior).** Let $\{p(x\mid\theta)\}$, $\theta\in\Theta\subseteq\mathbb{R}^d$ open, be a model with Fisher information matrix
$$
I(\theta)=\mathbb{E}_\theta\big[\nabla_\theta\log p(X\mid\theta)\,\nabla_\theta\log p(X\mid\theta)^{\!\top}\big],
$$
assumed finite, positive definite, and continuous. The *Jeffreys prior* is the (possibly improper) measure with density
$$
\pi_J(\theta)\ \propto\ \sqrt{\det I(\theta)}
$$
with respect to $\lambda_d$.

Two conventions to fix, since both trip people up. $I(\theta)$ here is the *single-observation* information; for $n$ i.i.d. observations the information is $nI(\theta)$, so $\sqrt{\det(nI)}=n^{d/2}\sqrt{\det I}$ differs only by a $\theta$-free constant and defines the same prior. And $\pi_J$ is defined only up to a positive constant, which is why it is allowed to be improper: what the recipe pins down is the *shape*, not the total mass.

**Theorem 10.20 (reparameterization invariance of the Jeffreys prior).** Let $g:\Theta\to\Phi$ be a $C^1$ diffeomorphism and $\phi=g(\theta)$. Let $I_\theta$ and $I_\phi$ be the Fisher informations of the same model in the two parameterizations. Then
$$
I_\theta(\theta)=J^{\!\top} I_\phi\big(g(\theta)\big)J,\qquad J:=\frac{\partial g}{\partial\theta}(\theta),
$$
and consequently the Jeffreys prior computed in $\phi$-coordinates and pushed forward to $\theta$-coordinates equals the Jeffreys prior computed directly in $\theta$-coordinates. In symbols, $g_\#\pi_J^{(\theta)}=\pi_J^{(\phi)}$.

*Proof.* Write $\ell_\theta(x)=\log p(x\mid\theta)$ and $\tilde\ell_\phi(x)=\log p(x\mid g^{-1}(\phi))$; these are the same function of $x$ at corresponding parameter values, so by the chain rule $\nabla_\theta\ell=J^{\!\top}\nabla_\phi\tilde\ell$. Substituting into the definition of $I_\theta$,
$$
I_\theta(\theta)=\mathbb{E}\big[J^{\!\top}\nabla_\phi\tilde\ell\,\nabla_\phi\tilde\ell^{\!\top}J\big]=J^{\!\top}I_\phi(\phi)J,
$$
since $J$ is non-random. Taking determinants and square roots,
$$
\sqrt{\det I_\theta(\theta)}=|\det J|\,\sqrt{\det I_\phi(g(\theta))}. \tag{10.1}
$$
Now apply Proposition 10.18 to the density $\pi^{(\phi)}_J(\phi)\propto\sqrt{\det I_\phi(\phi)}$: the density it induces on $\theta$ is
$$
\pi^{(\phi)}_J(g(\theta))\,|\det J| \ \propto\ \sqrt{\det I_\phi(g(\theta))}\,|\det J|\ =\ \sqrt{\det I_\theta(\theta)}\ \propto\ \pi_J^{(\theta)}(\theta),
$$
where the middle equality is (10.1). Since both are densities of the same total mass convention, the measures agree. $\blacksquare$

The mechanism deserves one sentence of emphasis, because it is the whole trick: the Jacobian factor $|\det J|$ that ruins flat priors is *exactly* the factor that $\sqrt{\det I}$ produces, so the two cancel. Jeffreys priors are the unique (up to constants) priors built pointwise from the Fisher metric with this property.

**Worked Example 10.11 (Jeffreys for a Bernoulli, and what it does with zero successes).** For $\mathrm{Bernoulli}(\theta)$, $\ell=x\log\theta+(1-x)\log(1-\theta)$, so $\partial_\theta\ell=\frac{x}{\theta}-\frac{1-x}{1-\theta}$ and
$$
I(\theta)=\mathbb{E}\Big[\Big(\frac{X-\theta}{\theta(1-\theta)}\Big)^2\Big]=\frac{\theta(1-\theta)}{\theta^2(1-\theta)^2}=\frac{1}{\theta(1-\theta)}.
$$
Hence $\pi_J(\theta)\propto\theta^{-1/2}(1-\theta)^{-1/2}$, which is $\mathrm{Beta}(\tfrac12,\tfrac12)$ — proper, with pseudo-count $\nu=1$. Now suppose you observe $n=10$ trials with $s=0$ successes, the regime where prior choice actually matters:

| prior | posterior | posterior mean | $P(\theta>0.2\mid D)$ |
|---|---|---|---|
| $\mathrm{Beta}(1,1)$, uniform | $\mathrm{Beta}(1,11)$ | 0.083333 | 0.085899 |
| $\mathrm{Beta}(\tfrac12,\tfrac12)$, Jeffreys | $\mathrm{Beta}(0.5,10.5)$ | 0.045455 | 0.032387 |
| $\mathrm{Beta}(0,0)$, Haldane | $\mathrm{Beta}(0,10)$ | — improper — | undefined |
| MLE | — | 0.000000 | — |

Two defensible "objective" answers to "how likely is a failure rate above 20% after ten clean trials" differ by a factor of $2.7$ ($3.2\%$ against $8.6\%$), and a third "objective" choice produces no answer at all. This is why safety-critical reporting of rare-event rates should always state the prior.

An honest caveat closes the topic, because Theorem 10.20 can leave the impression that Jeffreys priors are the solved end of the story, and they are not. The invariance argument is exactly as strong in $d$ dimensions as in one — the proof above never used $d=1$ — but the *statistical performance* of the recipe degrades as parameters multiply. For $N(\mu,\sigma^2)$ with both parameters unknown, the joint Jeffreys prior works out to $\propto\sigma^{-2}$ (Exercise C.2(c)), and it yields marginal inferences for $\sigma^2$ that most statisticians reject in favour of the "independence Jeffreys" prior $\propto\sigma^{-1}$ — a choice Jeffreys himself endorsed, at the price of the very invariance that motivated the construction. The degradation is not cosmetic: in models with many nuisance parameters, the joint Jeffreys prior can produce *inconsistent* inference. The classic instance is the Neyman–Scott problem — pairs $X_{i1},X_{i2}\sim N(\mu_i,\sigma^2)$ with a fresh mean per pair, so the nuisance dimension grows with $n$ — where the Jeffreys-prior posterior for $\sigma^2$, like the MLE, converges to $\sigma^2/2$: half the true value, forever. The modern repair is the *reference prior* program of Bernardo and Berger, which maximizes the data's expected information gain and treats parameters of interest and nuisance parameters asymmetrically; it recovers Jeffreys in one dimension and departs from it beyond. The working rule this leaves you with: trust Jeffreys as a default for one or two parameters, treat it as one candidate among several beyond that, and never confuse "invariant" with "harmless" — the invariance is to your choice of coordinates, not to the model's difficulty.

### 10.3.4 Improper priors: when you can get away with it

Jeffreys priors are frequently improper: $\pi(\theta)\ge0$ measurable with $\int\pi\,\mathrm{d}\theta=\infty$. Such a $\pi$ is not a probability measure, so Proposition 10.3 does not apply and Definition 10.2's $Z$ can be infinite. The Bayesian machinery can still be run formally, and often the result is a genuine posterior.

**Proposition 10.21 (propriety of the posterior).** Let $\pi\ge0$ be a measurable, possibly non-integrable prior density and let $Z(x)=\int p(x\mid\theta)\pi(\theta)\mathrm{d}\theta$. If $0<Z(x)<\infty$ then $\pi(\theta\mid x):=p(x\mid\theta)\pi(\theta)/Z(x)$ is a probability density, and it is the unique such normalization; if $Z(x)=\infty$ there is no posterior, and no amount of algebra creates one. Moreover:

(i) *(Gaussian location.)* For $X_1,\dots,X_n\overset{iid}{\sim}N(\mu,\sigma^2)$ with $\sigma^2$ known and $\pi(\mu)\equiv1$ on $\mathbb{R}$, $Z<\infty$ for every $n\ge1$ and $\mu\mid D\sim N(\bar x,\sigma^2/n)$.

(ii) *(Haldane–Bernoulli.)* For Bernoulli data with $\pi(\theta)\propto\theta^{-1}(1-\theta)^{-1}$, the posterior is $\mathrm{Beta}(s,n-s)$, which is proper if and only if $1\le s\le n-1$. With $s=0$ or $s=n$, $Z=\infty$.

*Proof.* The first claim is arithmetic. (i) $Z=\int(2\pi\sigma^2)^{-n/2}\exp(-\sum_i(x_i-\mu)^2/2\sigma^2)\mathrm{d}\mu$; completing the square as in Proposition 10.13 gives $\exp(-\sum_i(x_i-\bar x)^2/2\sigma^2)\cdot(2\pi\sigma^2)^{-n/2}\sqrt{2\pi\sigma^2/n}<\infty$, and the $\mu$-dependence is $\exp(-n(\mu-\bar x)^2/2\sigma^2)$, i.e. $N(\bar x,\sigma^2/n)$. (ii) The posterior kernel is $\theta^{s-1}(1-\theta)^{n-s-1}$, whose integral is $B(s,n-s)$, finite iff $s>0$ and $n-s>0$. $\blacksquare$

The moral is operational: an improper prior is a limit you must check, dataset by dataset. Hierarchical models (§10.4) are where this bites hardest — the flat prior on a group-level variance $\tau^2$ that looks harmless produces an improper posterior in the classic one-way random effects model, a fact discovered repeatedly and painfully. When a sampler "fails to converge" on a hierarchical model with a $\mathrm{Uniform}(0,10^6)$ prior on a scale parameter, the model, not the sampler, is often the problem.

### 10.3.5 Priors in machine learning, honestly

**Proposition 10.22 (weight decay is a Gaussian prior; $\ell_1$ is a Laplace prior).** Consider a model with likelihood $p(y\mid x,\theta)$ and let $D=\{(x_i,y_i)\}_{i=1}^n$.

(i) If $y_i\mid x_i,\theta\sim N(f_\theta(x_i),\sigma^2)$ and $\theta\sim N(0,\tau^2I)$, then
$$
\arg\max_\theta \pi(\theta\mid D)=\arg\min_\theta\ \sum_{i=1}^n\big(y_i-f_\theta(x_i)\big)^2+\lambda\|\theta\|_2^2,\qquad \boxed{\lambda=\frac{\sigma^2}{\tau^2}}.
$$

(ii) If instead $\theta_j\overset{iid}{\sim}\mathrm{Laplace}(0,b)$, the MAP objective is $\sum_i(y_i-f_\theta(x_i))^2+\lambda_1\|\theta\|_1$ with $\lambda_1=2\sigma^2/b$.

(iii) For a classification likelihood (softmax cross-entropy) with $\theta\sim N(0,\tau^2I)$, the MAP objective is $\sum_i\mathrm{CE}(y_i,f_\theta(x_i))+\frac{1}{2\tau^2}\|\theta\|_2^2$; if the training loss is averaged rather than summed, the weight-decay coefficient is $\frac{1}{2n\tau^2}$.

*Proof.* Take $-\log$ of $p(D\mid\theta)\pi(\theta)$ and drop $\theta$-free terms. In (i), $-\log p(D\mid\theta)=\frac{1}{2\sigma^2}\sum_i(y_i-f_\theta(x_i))^2+\text{const}$ and $-\log\pi(\theta)=\frac{1}{2\tau^2}\|\theta\|_2^2+\text{const}$; multiplying the whole objective by $2\sigma^2$ (which does not move the argmin) gives the stated form with $\lambda=\sigma^2/\tau^2$. Same computation with $-\log\pi(\theta)=\|\theta\|_1/b+\text{const}$ for (ii), and directly for (iii). $\blacksquare$

**Worked Example 10.12 (translating your weight decay into a prior).** Suppose you train with mean-squared error averaged over $n=1000$ examples and weight decay coefficient $c$, i.e. objective $\frac1n\sum_i(y_i-f_\theta(x_i))^2+c\|\theta\|^2$. Multiplying by $n$, the effective summed penalty is $\lambda=nc$, so by Proposition 10.22(i) the implied prior sd is $\tau=\sigma/\sqrt{nc}$. With residual noise $\sigma=1$ and $c=10^{-4}$: $\tau=1/\sqrt{0.1}=3.162$. With $c=10^{-2}$: $\tau=0.316$. Conversely, a belief that weights are $O(3)$ corresponds to $\lambda=\sigma^2/\tau^2=1/9=0.111$, i.e. $c=1.1\times10^{-4}$ at $n=1000$. It is worth doing this arithmetic once for a model you actually train: most default weight-decay values correspond to priors nobody would defend if stated out loud, precisely because the mapping involves $n$ and $\sigma$, which change between projects while the default does not.

Three further remarks, offered without varnish.

First, **most ML priors are chosen for tractability, and papers rarely admit it.** $N(0,\tau^2I)$ on the weights of a ReLU network is not a considered belief about functions; it is the only prior that keeps the algebra and the sampler cheap. Its induced distribution over functions depends violently on depth, width, and activation — this is the whole content of the neural-tangent and infinite-width literatures — and is essentially never inspected. When a paper says "we place a standard Gaussian prior on the weights", read "we regularize".

Second, **sparsity priors do not do what MAP suggests**, a point we prove in Proposition 10.37: the Laplace prior's MAP is exactly sparse, but its posterior mean is not, and the posterior assigns probability zero to any coefficient being exactly zero. If you want genuine sparsity in a Bayesian sense you need a spike-and-slab prior with an atom at zero, which is a different object (and, since the atom means the prior is not absolutely continuous, one where Chapter 2's careful base-measure bookkeeping earns its keep).

Third, **priors are checkable.** The prior predictive $p(x)=\int p(x\mid\theta)\pi(\theta)\mathrm{d}\theta$ is a distribution over datasets you can simulate from. If your "weakly informative" prior generates click-through rates of 0.97 or house prices of $-\$10^6$, you have learned something before touching the data. This *prior predictive check* costs a few lines of code and is the single highest-value habit in this chapter. Since recommending it without performing it would be poor form, here it is performed.

**Worked Example 10.13 (a prior predictive check that fails, and one that passes).** You are about to estimate a click-through rate from $n=50$ impressions and must choose a prior on $\theta$. Do not reason about $\theta$; reason about the *data the prior predicts*. Under $\theta\sim\mathrm{Beta}(a_0,b_0)$ the prior predictive number of clicks is Beta-Binomial$(50,a_0,b_0)$ (Proposition 10.11(iv) with $n=0$), so every row below is exactly computable.

| prior | prior 95% interval for $\theta$ | $\mathbb{E}[\text{clicks}]$ | $P(\text{rate}>0.25)$ | $P(\text{rate}>0.5)$ |
|---|---|---|---|---|
| $\mathrm{Beta}(1,1)$ "uniform" | $[0.025,\ 0.975]$ | 25.0 | 0.7451 | 0.4902 |
| $\mathrm{Beta}(0.5,0.5)$ Jeffreys | $[0.0015,\ 0.9985]$ | 25.0 | 0.6648 | 0.4937 |
| $\mathrm{Beta}(2,20)$ | $[0.012,\ 0.238]$ | 4.55 | 0.0349 | 0.0001 |
| $\mathrm{Beta}(20,200)$ | $[0.057,\ 0.132]$ | 4.55 | 0.0016 | 0.0000 |

Read the first row as a prediction and it is absurd: the "uninformative" uniform prior says that before seeing data you think it is a coin flip — probability $0.49$ — whether more than *half* your impressions will convert. No one working on click-through rates believes that; typical rates are a few percent. The Jeffreys prior is worse on this criterion, because it piles mass at both extremes. $\mathrm{Beta}(2,20)$ predicts about $4.6$ clicks in 50 with a plausible range, and is genuinely weakly informative: it rules out the absurd without pinning down the answer. $\mathrm{Beta}(20,200)$ predicts the same mean but with a prior interval of width $0.075$ — that is a strong claim, appropriate only if you really do have historical data worth 220 impressions.

Two things to take from this. First, the check is *cheap and decisive*: four lines of code turned an abstract argument about objectivity into a falsifiable prediction, and one candidate prior failed obviously. Second, the check is stated in the units of the problem — clicks out of 50 — which is why you can ask a domain expert whether the prediction is reasonable, and cannot ask them whether $a_0=2$ is reasonable. Do this before every analysis in this chapter, including the ones where you think the prior does not matter.

> **Common Confusion 10.4.** *"I used a uniform prior, so my analysis is objective — the data speak for themselves."*
>
> Uniform on *what*? Worked Example 10.10 exhibits two uniform priors that disagree by a factor of $3.2$ about the event $\{\theta<0.1\}$. Neither is more objective; they are different beliefs in different coordinates. Worse, in high dimensions "uniform" becomes actively strange: a uniform prior on the $d$-ball puts almost all its mass near the boundary at radius $\approx1$, so a "flat prior on weights in a box" asserts that the weight vector is large with probability near one. If you want a defensible answer to the objectivity question, the options are (a) state an informative prior and its provenance, (b) use the Jeffreys prior and accept that it is model-dependent and often improper, or (c) do a sensitivity analysis over a range of priors and report the range, as in Worked Examples 10.4 and 10.11. Option (d), asserting objectivity, is not available.

> **Self-check 10.3.**
> 1. You put a flat prior on $\log\sigma$. Is the induced prior on $\sigma^2$ flat? Compute its density.
> 2. In the proof of Theorem 10.20, exactly where would the argument fail if we used $\det I(\theta)$ instead of $\sqrt{\det I(\theta)}$?
> 3. Give a dataset for which the Haldane prior yields no posterior, and explain in words what went wrong.
> 4. Your training objective averages the loss over the batch and uses weight decay $c=10^{-3}$ with $n=50{,}000$ and $\sigma=0.5$. What prior sd on the weights are you asserting?

## 10.4 Hierarchical models and shrinkage

### 10.4.1 The problem partial pooling solves

You are estimating click-through rates for eight ad variants. One variant has 50 impressions, another 2000. Two extreme strategies are available. *No pooling*: estimate each rate by its own empirical frequency. This is unbiased and, for the 50-impression variant, useless — the standard error is about 0.06 on a quantity near 0.2. *Complete pooling*: estimate one rate for all variants. This has tiny variance and is wrong if the variants genuinely differ.

Partial pooling is the interpolation, and the Bayesian formulation produces it automatically rather than as a heuristic: model the group-level parameters as themselves drawn from a common distribution whose parameters you also do not know. Each group's estimate then borrows strength from the others, in an amount determined by how similar the groups appear to be and by how much data each group has.

**Definition 10.23 (hierarchical model).** A *hierarchical* (or *multilevel*) Bayesian model specifies a chain of conditional distributions
$$
\psi\sim\Pi_{\text{hyper}},\qquad \theta_1,\dots,\theta_J\mid\psi\overset{iid}{\sim}\Pi_{\psi},\qquad
D_j\mid\theta_j\sim P_{\theta_j}\ \ (j=1,\dots,J),
$$
with $D_1,\dots,D_J$ conditionally independent given $(\theta_1,\dots,\theta_J)$. The $\theta_j$ are *group-level parameters*, $\psi$ are *hyperparameters*, and $\Pi_\psi$ is the *population distribution*.

The dependence structure is worth naming with Chapter 1's vocabulary: the $\theta_j$ are conditionally independent given $\psi$ (Definition 1.24) but *not* independent marginally. That is precisely the channel through which group 1's data informs group 2's estimate: data on $\theta_1$ updates $\psi$, and $\psi$ ties all the $\theta_j$ together. If you had fixed $\psi$ in advance, no borrowing would occur.

### 10.4.2 Shrinkage in the Gaussian case, derived

**Theorem 10.24 (Gaussian hierarchical model).** Suppose group $j$ yields a summary $y_j$ with known sampling variance $s_j^2$,
$$
y_j\mid\theta_j\sim N(\theta_j,s_j^2)\ \text{ independently},\qquad
\theta_j\mid\mu,\tau^2\overset{iid}{\sim}N(\mu,\tau^2),
$$
with $\mu,\tau^2$ known for now. Then

(i) marginally $y_j\sim N(\mu,\ \tau^2+s_j^2)$, independently across $j$;

(ii) the posteriors are independent across $j$ with
$$
\theta_j\mid y\ \sim\ N\Big((1-B_j)\,y_j+B_j\,\mu,\ \ (1-B_j)\,s_j^2\Big),
\qquad B_j:=\frac{s_j^2}{s_j^2+\tau^2}\in(0,1);
$$

(iii) if instead $\mu$ carries a flat prior on $\mathbb{R}$ (and $\tau^2$ is known), the posterior for $\mu$ is $N(\hat\mu,\ V)$ with
$$
\hat\mu=\frac{\sum_j w_jy_j}{\sum_jw_j},\quad w_j=\frac{1}{\tau^2+s_j^2},\quad V=\Big(\sum_jw_j\Big)^{-1},
$$
and the marginal posterior of $\theta_j$ is $N\big((1-B_j)y_j+B_j\hat\mu,\ (1-B_j)s_j^2+B_j^2V\big)$.

*Proof.* (i) $y_j=\theta_j+\varepsilon_j$ with independent $\theta_j\sim N(\mu,\tau^2)$ and $\varepsilon_j\sim N(0,s_j^2)$; sums of independent Gaussians are Gaussian with added variances (Ch. 5). Independence across $j$ holds because the pairs $(\theta_j,\varepsilon_j)$ are independent across $j$.

(ii) Conditional on $(\mu,\tau^2)$ the model factorizes over $j$, and each factor is exactly Proposition 10.13 with "prior" $N(\mu,\tau^2)$ and one observation of variance $s_j^2$. Posterior precision $1/\tau^2+1/s_j^2$, so posterior variance $\frac{\tau^2s_j^2}{\tau^2+s_j^2}=(1-B_j)s_j^2$, and posterior mean
$$
\frac{\mu/\tau^2+y_j/s_j^2}{1/\tau^2+1/s_j^2}=\frac{s_j^2\mu+\tau^2y_j}{s_j^2+\tau^2}=B_j\mu+(1-B_j)y_j.
$$

(iii) Using (i), the likelihood for $\mu$ after marginalizing the $\theta_j$ is $\prod_j N(y_j;\mu,\tau^2+s_j^2)$, so with a flat prior the log-posterior is $-\frac12\sum_jw_j(y_j-\mu)^2+\text{const}$; completing the square gives $N(\hat\mu,V)$. For the last display, write $\theta_j=B_j\mu+(1-B_j)y_j+\eta_j$ with $\eta_j\sim N(0,(1-B_j)s_j^2)$ independent of $\mu$ given $y$ — this is (ii) applied conditionally on $\mu$ — and use the law of total variance (Ch. 6): $\mathrm{Var}(\theta_j\mid y)=\mathbb{E}[(1-B_j)s_j^2]+\mathrm{Var}(B_j\mu\mid y)=(1-B_j)s_j^2+B_j^2V$. $\blacksquare$

**Corollary 10.25 (the shrinkage factor, and James–Stein).** In Theorem 10.24, group $j$'s estimate is pulled toward the population mean by the fraction
$$
B_j=\frac{s_j^2}{s_j^2+\tau^2}=\frac{1/n_j}{1/n_j+\tau^2/\sigma^2}\quad\text{when }s_j^2=\sigma^2/n_j,
$$
so: (a) $B_j\to0$ as $n_j\to\infty$ (big groups are trusted); (b) $B_j\to1$ as $\tau^2\to0$ (homogeneous population $\Rightarrow$ complete pooling); (c) $B_j\to0$ as $\tau^2\to\infty$ (unrelated groups $\Rightarrow$ no pooling). Moreover, in the equal-variance case $s_j^2\equiv s^2$ with $\mu=0$ known, replacing $\tau^2$ by the moment estimate implied by $\sum_jy_j^2$ and using the shrinkage factor
$$
\hat B=\frac{(J-2)s^2}{\sum_j y_j^2}
$$
gives exactly the James–Stein estimator $\hat\theta=(1-\hat B)y$ of Chapter 8, which dominates the MLE $y$ in total mean squared error for $J\ge3$.

*Proof.* (a)–(c) are limits in the displayed formula. For the last claim: by Theorem 10.24(i), $y_j\sim N(0,s^2+\tau^2)$ i.i.d., so $\sum_jy_j^2/(s^2+\tau^2)\sim\chi^2_J$ and $\mathbb{E}\big[(J-2)/\sum_jy_j^2\big]=1/(s^2+\tau^2)$ using $\mathbb{E}[1/\chi^2_J]=1/(J-2)$. Hence $\hat B$ is an unbiased estimator of $B=s^2/(s^2+\tau^2)$, and substituting it into the Bayes rule $(1-B)y$ reproduces James–Stein verbatim. Its dominance was proved in Chapter 8. $\blacksquare$

This is the cleanest available answer to "why does James–Stein work?", a question that looks like a paradox in the frequentist framing and like bookkeeping in the Bayesian one. James–Stein is an empirical-Bayes estimator for a Gaussian hierarchical model, and its dominance is the statement that the hierarchical model's shrinkage is a better bet than no shrinkage even when you do not believe the hierarchical model.

### 10.4.3 Empirical Bayes

In practice $\mu$ and $\tau^2$ are unknown. Two routes: put priors on them and do full Bayes (Chapter 13's machinery), or estimate them from the marginal likelihood and plug in.

**Definition 10.26 (empirical Bayes / type-II maximum likelihood).** In a hierarchical model, the *empirical Bayes* estimate of the hyperparameters is
$$
\hat\psi\in\arg\max_\psi\ p(y\mid\psi)=\arg\max_\psi\int p(y\mid\theta)\,\Pi_\psi(\mathrm{d}\theta),
$$
after which group-level inference proceeds using $\Pi(\theta_j\mid y,\hat\psi)$. In the Gaussian case of Theorem 10.24 this reads
$$
(\hat\mu,\hat\tau^2)=\arg\max\ -\frac12\sum_{j=1}^J\Big[\log\big(2\pi(\tau^2+s_j^2)\big)+\frac{(y_j-\mu)^2}{\tau^2+s_j^2}\Big].
$$

Empirical Bayes is maximizing the same marginal likelihood that §10.5 will use for model comparison, and the same object that §10.6 maximizes to select the ridge penalty. It has one systematic flaw worth stating: it treats $\hat\psi$ as known, so it *understates* uncertainty, and it does so most where you can least afford it — when $J$ is small, $\hat\tau^2$ is poorly determined, and the estimate $\hat\tau^2=0$ (complete pooling) occurs with non-negligible probability, producing implausibly confident group estimates. Full Bayes over $\tau^2$ fixes this at the cost of an integral.

Figure 10.3 shows what partial pooling does to the eight-variant example, and Worked Example 10.14 supplies the numbers.

<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="sans-serif">
  <line x1="90.00" y1="70.00" x2="700.00" y2="70.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="90.00" y1="250.00" x2="700.00" y2="250.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="80.00" y="75.00" font-size="12" fill="currentColor" text-anchor="end" font-weight="normal">no pooling</text>
  <text x="80.00" y="255.00" font-size="12" fill="currentColor" text-anchor="end" font-weight="normal">partial pooling</text>
  <line x1="125.88" y1="250.00" x2="125.88" y2="255.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="125.88" y="269.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.12</text>
  <line x1="269.41" y1="250.00" x2="269.41" y2="255.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="269.41" y="269.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.16</text>
  <line x1="412.94" y1="250.00" x2="412.94" y2="255.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="412.94" y="269.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.20</text>
  <line x1="556.47" y1="250.00" x2="556.47" y2="255.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="556.47" y="269.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.24</text>
  <line x1="700.00" y1="250.00" x2="700.00" y2="255.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="700.00" y="269.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.28</text>
  <text x="395.00" y="292.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">estimated click-through rate</text>
  <line x1="432.93" y1="42.00" x2="432.93" y2="258.00" stroke="#10b981" stroke-width="1.6" stroke-dasharray="5,4"/>
  <text x="432.93" y="36.00" font-size="11" fill="#10b981" text-anchor="middle" font-weight="normal">grand mean 0.2056</text>
  <line x1="197.65" y1="70.00" x2="411.83" y2="250.00" stroke="#ef4444" stroke-width="1.6"/>
  <circle cx="197.65" cy="70.00" r="4.5" fill="#ef4444"/>
  <circle cx="411.83" cy="250.00" r="4.5" fill="#ef4444"/>
  <text x="197.65" y="60.00" font-size="9" fill="#ef4444" text-anchor="middle" font-weight="normal">n=50</text>
  <line x1="628.24" y1="70.00" x2="444.28" y2="250.00" stroke="#ef4444" stroke-width="1.6"/>
  <circle cx="628.24" cy="70.00" r="4.5" fill="#ef4444"/>
  <circle cx="444.28" cy="250.00" r="4.5" fill="#ef4444"/>
  <text x="628.24" y="60.00" font-size="9" fill="#ef4444" text-anchor="middle" font-weight="normal">n=50</text>
  <line x1="412.94" y1="70.00" x2="430.35" y2="250.00" stroke="#ef4444" stroke-width="1.6"/>
  <circle cx="412.94" cy="70.00" r="4.5" fill="#ef4444"/>
  <circle cx="430.35" cy="250.00" r="4.5" fill="#ef4444"/>
  <text x="412.94" y="60.00" font-size="9" fill="#ef4444" text-anchor="middle" font-weight="normal">n=100</text>
  <line x1="377.06" y1="70.00" x2="425.47" y2="250.00" stroke="#ef4444" stroke-width="1.6"/>
  <circle cx="377.06" cy="70.00" r="4.5" fill="#ef4444"/>
  <circle cx="425.47" cy="250.00" r="4.5" fill="#ef4444"/>
  <text x="377.06" y="60.00" font-size="9" fill="#ef4444" text-anchor="middle" font-weight="normal">n=100</text>
  <line x1="323.24" y1="70.00" x2="389.41" y2="250.00" stroke="#f59e0b" stroke-width="1.6"/>
  <circle cx="323.24" cy="70.00" r="4.5" fill="#f59e0b"/>
  <circle cx="389.41" cy="250.00" r="4.5" fill="#f59e0b"/>
  <text x="323.24" y="60.00" font-size="9" fill="#f59e0b" text-anchor="middle" font-weight="normal">n=400</text>
  <line x1="350.15" y1="70.00" x2="400.74" y2="250.00" stroke="#f59e0b" stroke-width="1.6"/>
  <circle cx="350.15" cy="70.00" r="4.5" fill="#f59e0b"/>
  <circle cx="400.74" cy="250.00" r="4.5" fill="#f59e0b"/>
  <text x="350.15" y="60.00" font-size="9" fill="#f59e0b" text-anchor="middle" font-weight="normal">n=400</text>
  <line x1="497.26" y1="70.00" x2="480.04" y2="250.00" stroke="#3b82f6" stroke-width="1.6"/>
  <circle cx="497.26" cy="70.00" r="4.5" fill="#3b82f6"/>
  <circle cx="480.04" cy="250.00" r="4.5" fill="#3b82f6"/>
  <text x="497.26" y="60.00" font-size="9" fill="#3b82f6" text-anchor="middle" font-weight="normal">n=2000</text>
  <line x1="499.06" y1="70.00" x2="481.33" y2="250.00" stroke="#3b82f6" stroke-width="1.6"/>
  <circle cx="499.06" cy="70.00" r="4.5" fill="#3b82f6"/>
  <circle cx="481.33" cy="250.00" r="4.5" fill="#3b82f6"/>
  <text x="499.06" y="60.00" font-size="9" fill="#3b82f6" text-anchor="middle" font-weight="normal">n=2000</text>
  <line x1="90.00" y1="308.00" x2="116.00" y2="308.00" stroke="#ef4444" stroke-width="2.2"/>
  <text x="124.00" y="312.00" font-size="11" fill="#ef4444" text-anchor="start" font-weight="normal">n = 50-100  (shrink 0.87-0.94)</text>
  <line x1="90.00" y1="324.00" x2="116.00" y2="324.00" stroke="#f59e0b" stroke-width="2.2"/>
  <text x="124.00" y="328.00" font-size="11" fill="#f59e0b" text-anchor="start" font-weight="normal">n = 400  (shrink 0.61)</text>
  <line x1="90.00" y1="340.00" x2="116.00" y2="340.00" stroke="#3b82f6" stroke-width="2.2"/>
  <text x="124.00" y="344.00" font-size="11" fill="#3b82f6" text-anchor="start" font-weight="normal">n = 2000  (shrink 0.27)</text>
  <text x="380.0" y="386" font-size="13" fill="currentColor" text-anchor="middle">Figure 10.3  Partial pooling in the Gaussian hierarchical model (8 ad variants, seed 10).</text>
  <text x="380.0" y="403" font-size="13" fill="currentColor" text-anchor="middle">Each group's raw rate is pulled toward the estimated grand mean by a factor B_j = s_j^2/(s_j^2 + tau^2):</text>
  <text x="380.0" y="420" font-size="13" fill="currentColor" text-anchor="middle">small groups move a long way, the two n = 2000 groups barely move. Mean squared error against the</text>
  <text x="380.0" y="437" font-size="13" fill="currentColor" text-anchor="middle">true rates falls from 1.316e-3 (no pooling) to 6.553e-4 (partial pooling), a 50.2% reduction.</text>
</svg>


**Worked Example 10.14 (eight ad variants; partial pooling halves the error).** Simulated with `numpy` default RNG, seed 10. True rates $\theta_j\sim N(0.20,0.04^2)$, impressions $n_j$ as shown, clicks $\sim\mathrm{Bin}(n_j,\theta_j)$. We use the Gaussian approximation of Theorem 10.24 with $y_j=$ observed rate and $s_j^2=y_j(1-y_j)/n_j$.

| $j$ | $n_j$ | clicks | $y_j$ | $s_j$ | true $\theta_j$ | $B_j$ | partial-pooled | posterior sd |
|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 7 | 0.1400 | 0.0491 | 0.1559 | 0.9103 | 0.1997 | 0.0147 |
| 2 | 50 | 13 | 0.2600 | 0.0620 | 0.1710 | 0.9419 | 0.2087 | 0.0150 |
| 3 | 100 | 20 | 0.2000 | 0.0400 | 0.1687 | 0.8708 | 0.2049 | 0.0144 |
| 4 | 100 | 19 | 0.1900 | 0.0392 | 0.2107 | 0.8664 | 0.2035 | 0.0143 |
| 5 | 400 | 70 | 0.1750 | 0.0190 | 0.1901 | 0.6033 | 0.1934 | 0.0120 |
| 6 | 400 | 73 | 0.1825 | 0.0193 | 0.2051 | 0.6112 | 0.1966 | 0.0120 |
| 7 | 2000 | 447 | 0.2235 | 0.0093 | 0.2337 | 0.2678 | 0.2187 | 0.0080 |
| 8 | 2000 | 448 | 0.2240 | 0.0093 | 0.2343 | 0.2681 | 0.2191 | 0.0080 |

One honest caveat before reading the table, because it is visible in the numbers. Theorem 10.24 assumes the sampling variances $s_j^2$ are *known*, but here they are estimated as $y_j(1-y_j)/n_j$ from the same data. Groups 1 and 2 have identical $n_j=50$, yet $s_1=0.0491$ and $s_2=0.0620$ purely because $y_2$ is closer to $1/2$ — so group 2 is judged noisier and shrunk harder ($B_2=0.9419$ against $B_1=0.9103$). That is an artifact of the plug-in variance, not a feature of partial pooling. It is mild here and it is the standard applied shortcut, but the principled alternatives are to model the counts directly as $\mathrm{Bin}(n_j,\theta_j)$ with a Beta or logit-normal population distribution and do full Bayes (Ch. 13), or at minimum to use a variance-stabilizing transform such as $\arcsin\sqrt{y_j}$, whose sampling variance $1/(4n_j)$ does not depend on $y_j$.

Empirical Bayes (Definition 10.26, Nelder–Mead on the marginal log-likelihood) gives $\hat\mu=0.205571$ and $\hat\tau^2=2.3730\times10^{-4}$, i.e. $\hat\tau=0.0154$. Note that $\hat\tau$ under-estimates the true $0.04$ here; with $J=8$ groups, $\tau$ is barely identified, exactly the caveat under Definition 10.26.

Mean squared error against the true $\theta_j$:

| estimator | MSE | RMSE |
|---|---|---|
| no pooling ($y_j$) | $1.3156\times10^{-3}$ | 0.03627 |
| complete pooling (pooled rate $0.215098$) | $1.1334\times10^{-3}$ | 0.03367 |
| partial pooling | $6.5529\times10^{-4}$ | 0.02560 |

Partial pooling cuts MSE by $50.2\%$ relative to no pooling. Look at where the gain comes from: group 2 observed $0.26$ against a truth of $0.171$ — an unlucky 13 clicks out of 50 — contributing $79.2\times10^{-4}$ to the no-pooling error, which partial pooling reduces to $14.2\times10^{-4}$. The price is paid in group 1, whose luck ran the other way: its error rises from $2.5\times10^{-4}$ to $19.2\times10^{-4}$. Shrinkage trades a small increase in error on groups whose noise happened to be favourable for a large decrease on groups whose noise was extreme. Since extremeness is common among small groups, the trade is profitable on average.

"On average" should be checked, not asserted. Repeating the entire experiment (new $\theta_j$, new clicks, re-fit EB each time) over 500 datasets with seed 11:

| estimator | mean MSE over 500 replications |
|---|---|
| no pooling | $1.2835\times10^{-3}$ |
| complete pooling | $1.8444\times10^{-3}$ |
| partial pooling | $7.6901\times10^{-4}$ |

Partial pooling beats no pooling in $76.0\%$ of replications and complete pooling in $92.0\%$; averaged over replications it reduces MSE by $40.1\%$ relative to no pooling. Notice also that complete pooling is *worse* than no pooling on average here — the groups really do differ — which is the point of interpolating rather than choosing a side.

This machinery is everywhere in applied ML, usually unlabelled. Per-user or per-item effects in a recommender with a shared population prior are a hierarchical model; so are the "empirical priors" in count-based exploration bonuses; so are multi-task and meta-learning setups where a shared initialization plays the role of $\mu$ and the task-specific fine-tuning plays the role of $\theta_j$ (the connection is exact for the linear-Gaussian case, and heuristic beyond it). Federated averaging with a proximal term to the global model is partial pooling with a fixed, hand-chosen $B_j$.

> **Common Confusion 10.5.** *"Shrinkage introduces bias on purpose. Surely the unbiased estimate is the honest one?"*
>
> Unbiasedness is a property of one estimator for one parameter under repeated sampling; it is not a synonym for correctness, and it is not free. In Worked Example 10.14 the unbiased no-pooling estimates have $1.32\times10^{-3}$ MSE and the biased partial-pooled estimates have $6.55\times10^{-4}$. If you will be judged on squared error — and virtually every deployed system is judged on something like squared error or log-loss, not on bias — the biased estimator is the better estimator. The deeper point, which Chapter 8 proved via James–Stein and Corollary 10.25 re-derives, is that when you estimate many parameters at once, the unbiased choice is *inadmissible*: there is an estimator that is at least as good for every true parameter value and strictly better for some. The honest summary is not "bias is bad" but "bias and variance are both costs, and shrinkage buys variance reduction cheaply when the groups are similar and each group's data is thin."
>
> One real caveat, though, in the other direction: shrinkage is not free for *individual* groups. If your application must protect a specific small group from being pulled toward the population — a fairness-relevant subgroup, say, or a safety-critical unit — then a per-group guarantee is what you need, and average MSE improvement does not provide it. Report both.

> **Self-check 10.4.**
> 1. In Theorem 10.24, what happens to $B_j$ when $s_j^2=\tau^2$? Interpret in words.
> 2. Why are the $\theta_j$ dependent under the marginal (prior) distribution, and why does that matter for borrowing strength?
> 3. Empirical Bayes returned $\hat\tau^2=2.373\times10^{-4}$ when the truth was $1.6\times10^{-3}$. Name the consequence for the reported posterior sds.
> 4. Give a situation in which complete pooling beats partial pooling.

## 10.5 Model comparison

### 10.5.1 The evidence is a distribution over datasets

Suppose you have two models $M_0,M_1$ — different likelihoods, different parameter spaces, different priors. Put a prior over models, $P(M_0),P(M_1)$, and apply Bayes' rule (Proposition 1.12) one level up.

**Definition 10.27 (marginal likelihood, Bayes factor, posterior odds).** For model $M$ with parameter space $\Theta_M$, prior $\Pi_M$ and likelihood $p(D\mid\theta,M)$, the *marginal likelihood* (or *evidence*) is
$$
p(D\mid M)=\int_{\Theta_M}p(D\mid\theta,M)\,\Pi_M(\mathrm{d}\theta).
$$
The *Bayes factor* of $M_1$ against $M_0$ is $\mathrm{BF}_{10}:=\dfrac{p(D\mid M_1)}{p(D\mid M_0)}$, and
$$
\underbrace{\frac{P(M_1\mid D)}{P(M_0\mid D)}}_{\text{posterior odds}}=\mathrm{BF}_{10}\times\underbrace{\frac{P(M_1)}{P(M_0)}}_{\text{prior odds}}.
$$

Everything interesting follows from one observation, which I want to state before any algebra because it is the intuition: **$p(\cdot\mid M)$ is a probability distribution over datasets.** It integrates (or sums) to one over the sample space. A model that can fit many different datasets must therefore assign small probability to each of them. This is not a penalty bolted on; it is conservation of probability.

**Proposition 10.28 (evidence as a prequential score; automatic Occam).** For any Bayesian model and data $D=(x_1,\dots,x_n)$:

(i) $\displaystyle \log p(D\mid M)=\sum_{k=1}^{n}\log p(x_k\mid x_1,\dots,x_{k-1},M)$: the log-evidence is the total log-loss of the model's honest sequential one-step-ahead predictions.

(ii) $\displaystyle \sum_{D'\in\mathcal{X}^n} p(D'\mid M)=1$ (counting measure; $\int$ in the dominated case).

*Proof.* (i) is Theorem 10.4(iii) with logs. (ii) is Proposition 10.3(ii): the evidence is the density of the marginal law of the data, which is a probability measure. $\blacksquare$

Part (ii) is the whole of Occam's razor, and it is worth saying why it needs no theorem beyond conservation of mass. A model that assigns appreciable probability to many qualitatively different datasets must assign small probability to each; a model that bets on few must assign more to those few. So if the observed $D$ is one of the datasets the rigid model was betting on, the rigid model wins. That is an accounting identity and an intuition, *not* a general inequality: whether $\mathrm{BF}_{10}$ exceeds one depends entirely on where $D$ falls, and Worked Example 10.15 shows the verdict flipping as $s$ moves from $13$ to $16$ on the same twenty flips. "The simpler model always wins" is false, and so is its converse.

Part (i) is the deepest of the three and deserves unpacking. It says the marginal likelihood is not "the likelihood at the best-fitting parameter with a fudge factor for complexity". It is a *held-out* predictive score, accumulated over the whole dataset, in which every observation is predicted using only its predecessors. A model with a flexible parameter space starts out predicting badly (its early one-step predictions are diffuse) and improves as it learns; a rigid model predicts moderately well immediately and never improves. The evidence adds up the whole learning curve. That is why complexity is penalized without a penalty term, and it is also why the evidence is closer in spirit to cross-validation than to a maximized likelihood.

Figure 10.4 shows this with the smallest possible example: twenty coin flips.

<svg viewBox="0 0 760 440" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="sans-serif">
  <line x1="70.00" y1="300.00" x2="720.00" y2="300.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="70.00" y1="300.00" x2="70.00" y2="50.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="65.00" y1="300.00" x2="70.00" y2="300.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="304.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0.00</text>
  <line x1="65.00" y1="237.50" x2="70.00" y2="237.50" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="241.50" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0.05</text>
  <line x1="65.00" y1="175.00" x2="70.00" y2="175.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="179.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0.10</text>
  <line x1="65.00" y1="112.50" x2="70.00" y2="112.50" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="116.50" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0.15</text>
  <line x1="65.00" y1="50.00" x2="70.00" y2="50.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="54.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0.20</text>
  <rect x="73.71" y="300.00" width="11.76" height="0.00" fill="#3b82f6" opacity="0.85"/>
  <rect x="85.48" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="85.48" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0</text>
  <rect x="104.67" y="299.98" width="11.76" height="0.02" fill="#3b82f6" opacity="0.85"/>
  <rect x="116.43" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="135.62" y="299.77" width="11.76" height="0.23" fill="#3b82f6" opacity="0.85"/>
  <rect x="147.38" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="147.38" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">2</text>
  <rect x="166.57" y="298.64" width="11.76" height="1.36" fill="#3b82f6" opacity="0.85"/>
  <rect x="178.33" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="197.52" y="294.22" width="11.76" height="5.78" fill="#3b82f6" opacity="0.85"/>
  <rect x="209.29" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="209.29" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">4</text>
  <rect x="228.48" y="281.52" width="11.76" height="18.48" fill="#3b82f6" opacity="0.85"/>
  <rect x="240.24" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="259.43" y="253.79" width="11.76" height="46.21" fill="#3b82f6" opacity="0.85"/>
  <rect x="271.19" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="271.19" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">6</text>
  <rect x="290.38" y="207.59" width="11.76" height="92.41" fill="#3b82f6" opacity="0.85"/>
  <rect x="302.14" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="321.33" y="149.83" width="11.76" height="150.17" fill="#3b82f6" opacity="0.85"/>
  <rect x="333.10" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="333.10" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">8</text>
  <rect x="352.29" y="99.78" width="11.76" height="200.22" fill="#3b82f6" opacity="0.85"/>
  <rect x="364.05" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="383.24" y="79.75" width="11.76" height="220.25" fill="#3b82f6" opacity="0.85"/>
  <rect x="395.00" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="395.00" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">10</text>
  <rect x="414.19" y="99.78" width="11.76" height="200.22" fill="#3b82f6" opacity="0.85"/>
  <rect x="425.95" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="445.14" y="149.83" width="11.76" height="150.17" fill="#3b82f6" opacity="0.85"/>
  <rect x="456.90" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="456.90" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">12</text>
  <rect x="476.10" y="207.59" width="11.76" height="92.41" fill="#3b82f6" opacity="0.85"/>
  <rect x="487.86" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="507.05" y="253.79" width="11.76" height="46.21" fill="#3b82f6" opacity="0.85"/>
  <rect x="518.81" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="518.81" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">14</text>
  <rect x="538.00" y="281.52" width="11.76" height="18.48" fill="#3b82f6" opacity="0.85"/>
  <rect x="549.76" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="568.95" y="294.22" width="11.76" height="5.78" fill="#3b82f6" opacity="0.85"/>
  <rect x="580.71" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="580.71" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">16</text>
  <rect x="599.90" y="298.64" width="11.76" height="1.36" fill="#3b82f6" opacity="0.85"/>
  <rect x="611.67" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="630.86" y="299.77" width="11.76" height="0.23" fill="#3b82f6" opacity="0.85"/>
  <rect x="642.62" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="642.62" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">18</text>
  <rect x="661.81" y="299.98" width="11.76" height="0.02" fill="#3b82f6" opacity="0.85"/>
  <rect x="673.57" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <rect x="692.76" y="300.00" width="11.76" height="0.00" fill="#3b82f6" opacity="0.85"/>
  <rect x="704.52" y="240.48" width="11.76" height="59.52" fill="#f59e0b" opacity="0.85"/>
  <text x="704.52" y="318.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">20</text>
  <text x="395.00" y="340.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">s = number of heads in 20 flips (the data)</text>
  <text x="24.00" y="32.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">p(s | model)</text>
  <line x1="78.00" y1="56.00" x2="104.00" y2="56.00" stroke="#3b82f6" stroke-width="8.0"/>
  <text x="112.00" y="60.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">M0: theta = 1/2 exactly (no free parameter)</text>
  <line x1="78.00" y1="76.00" x2="104.00" y2="76.00" stroke="#f59e0b" stroke-width="8.0"/>
  <text x="112.00" y="80.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">M1: theta ~ Uniform(0,1) (one free parameter)</text>
  <line x1="487.86" y1="201.59" x2="487.86" y2="108.00" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="3,3"/>
  <text x="495.86" y="108.00" font-size="11" fill="#ef4444" text-anchor="start" font-weight="normal">s=13: BF10 = 0.644, the SIMPLE model wins</text>
  <line x1="580.71" y1="234.48" x2="580.71" y2="128.00" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="3,3"/>
  <text x="588.71" y="128.00" font-size="11" fill="#ef4444" text-anchor="start" font-weight="normal">s=16: BF10 = 10.3, the flexible model wins</text>
  <text x="380.0" y="380" font-size="13" fill="currentColor" text-anchor="middle">Figure 10.4  The marginal likelihood is a probability distribution over datasets, so it must sum to 1.</text>
  <text x="380.0" y="397" font-size="13" fill="currentColor" text-anchor="middle">M1 can explain any s but spreads its mass evenly (1/21 = 0.0476 each); M0 can only explain s near 10</text>
  <text x="380.0" y="414" font-size="13" fill="currentColor" text-anchor="middle">but concentrates 0.0739 there. Occam's razor is automatic: flexibility is paid for in spread, not by any</text>
  <text x="380.0" y="431" font-size="13" fill="currentColor" text-anchor="middle">added penalty term. Only data that M0 explains badly (s >= 15 or s <= 5) buys M1 its extra parameter.</text>
</svg>


**Worked Example 10.15 (Occam's razor on twenty coin flips).** Compare $M_0$: $\theta=1/2$ exactly, no free parameter, against $M_1$: $\theta\sim\mathrm{Uniform}(0,1)$, one free parameter. Then
$$
p(s\mid M_0)=\binom{20}{s}2^{-20},\qquad
p(s\mid M_1)=\binom{20}{s}B(s+1,21-s)=\frac{1}{21}\ \text{ for every }s,
$$
where the second identity is Proposition 10.11(iv) with $a_n=b_n=1$, $m=20$. So $M_1$'s predictive distribution over the 21 possible datasets is exactly uniform: it can explain anything, at $1/21=0.047619$ apiece. $M_0$ concentrates: $p(10\mid M_0)=0.176197$, $p(13\mid M_0)=0.073929$, $p(16\mid M_0)=0.004621$.

| observed $s$ | $\mathrm{BF}_{10}$ | $\log_{10}\mathrm{BF}_{10}$ | verdict |
|---|---|---|---|
| 10 | 0.2703 | $-0.568$ | simple model favoured 3.7:1 |
| 13 | 0.6441 | $-0.191$ | simple model favoured 1.6:1 |
| 16 | 10.306 | $+1.013$ | flexible model favoured 10:1 |
| 20 | 49 932 | $+4.698$ | flexible model overwhelmingly favoured |

At $s=13$ the MLE is $0.65$, the likelihood ratio $\sup_\theta p(D\mid\theta)/p(D\mid\tfrac12)=2.49$ favours the free-parameter model, and yet the Bayes factor favours the *fixed* model, $1.55$ to $1$. The extra parameter has to pay for itself, and at $s=13$ it does not. This is Occam's razor as arithmetic. (Verification: the analytic evidence $B(14,8)=6.1428080004\times10^{-7}$ matches numerical quadrature of $\int_0^1\theta^{13}(1-\theta)^7\mathrm{d}\theta$ to ten significant figures.)

### 10.5.2 The evidence is prior-sensitive in a way likelihoods are not

The same example exposes the central practical hazard. Widen or narrow $M_1$'s prior and the Bayes factor moves, even though the likelihood is untouched:

| $M_1$ prior | $p(D\mid M_1)$ | $\mathrm{BF}_{10}$ at $s=13$ |
|---|---|---|
| $\mathrm{Beta}(0.5,0.5)$ | $4.1879\times10^{-7}$ | 0.4391 |
| $\mathrm{Beta}(1,1)$ | $6.1428\times10^{-7}$ | 0.6441 |
| $\mathrm{Beta}(2,2)$ | $8.1580\times10^{-7}$ | 0.8554 |
| $\mathrm{Beta}(5,5)$ | $1.0116\times10^{-6}$ | 1.0608 |
| $\mathrm{Beta}(20,20)$ | $1.0548\times10^{-6}$ | 1.1060 |
| $\mathrm{Beta}(100,100)$ | $9.8708\times10^{-7}$ | 1.0350 |

The verdict flips from $2.3{:}1$ against $M_1$ to $1.1{:}1$ for it purely through the prior. Note the non-monotonicity: as $\mathrm{Beta}(a,a)$ concentrates, $M_1$ converges to $M_0$ and $\mathrm{BF}_{10}\to1$ from above. And there is no "vague" limit in which the Bayes factor becomes prior-free. Here the parameter is confined to $[0,1]$, so the damage is bounded; but replace $M_1$'s prior by $\mathrm{Uniform}(-T,T)$ on an unbounded parameter and its evidence falls like $1/T$, so $\mathrm{BF}_{10}\to0$ as $T\to\infty$ and the flexible model is rejected however well it fits. Since an improper prior is that limit, a Bayes factor computed with one is not a number at all: it depends on an arbitrary constant that does not cancel between models of different dimension. This is Lindley's paradox in embryo, and it means:

> **Bayes factors with improper or arbitrarily diffuse priors are meaningless.**

Posterior *estimation* is fairly robust to diffuse priors (Worked Example 10.4), because there the prior enters through a weight that tends to zero as $n$ grows. Model *comparison* is not, because there the prior's spread enters multiplicatively and never washes out. This asymmetry is the single most common error in applied Bayesian model selection.

### 10.5.3 The Jeffreys–Lindley paradox

**Proposition 10.29 (Jeffreys–Lindley).** Let $\bar X\mid\mu\sim N(\mu,\sigma^2/n)$ with $\sigma^2$ known, and compare $M_0:\mu=0$ against $M_1:\mu\sim N(0,\tau^2)$. Write $z=\sqrt n\,\bar x/\sigma$ and $r=n\tau^2/\sigma^2$. Then
$$
\mathrm{BF}_{01}=\frac{p(\bar x\mid M_0)}{p(\bar x\mid M_1)}=\sqrt{1+r}\ \exp\Big(-\frac{z^2}{2}\cdot\frac{r}{1+r}\Big).
$$
Consequently, for any fixed $z$ (hence any fixed two-sided $p$-value $2\Phi(-|z|)$), $\mathrm{BF}_{01}\to\infty$ as $n\to\infty$: the frequentist test rejects $M_0$ at a fixed level while the Bayes factor becomes arbitrarily decisive *in favour of* $M_0$.

*Proof.* Under $M_0$, $\bar X\sim N(0,\sigma^2/n)$; under $M_1$, marginally $\bar X\sim N(0,\sigma^2/n+\tau^2)$ by the same convolution argument as Theorem 10.24(i). So
$$
\mathrm{BF}_{01}=\frac{(2\pi\sigma^2/n)^{-1/2}\exp\{-\bar x^2n/2\sigma^2\}}{(2\pi(\sigma^2/n+\tau^2))^{-1/2}\exp\{-\bar x^2/(2(\sigma^2/n+\tau^2))\}}
=\sqrt{1+r}\,\exp\Big(-\frac{z^2}{2}+\frac{z^2}{2(1+r)}\Big),
$$
using $\bar x^2/(\sigma^2/n+\tau^2)=z^2/(1+r)$. Combine the exponents. For fixed $z$, $r\to\infty$ with $n$, so the exponential tends to $e^{-z^2/2}$ while $\sqrt{1+r}\to\infty$. $\blacksquare$

**Worked Example 10.16 (a $p$-value of exactly 0.05 at four sample sizes).** Fix $\sigma=1$, $\tau=1$, and choose $\bar x$ so that $z=1.959964$ exactly, i.e. $p=0.0500$ every time. Assume equal prior model probabilities.

| $n$ | $\bar x$ | $p$-value | $\mathrm{BF}_{01}$ | $P(M_0\mid D)$ |
|---|---|---|---|---|
| 100 | 0.195996 | 0.0500 | 1.50 | 0.600 |
| 1 000 | 0.061980 | 0.0500 | 4.64 | 0.823 |
| 10 000 | 0.019600 | 0.0500 | 14.65 | 0.936 |
| 100 000 | 0.006198 | 0.0500 | 46.33 | 0.979 |

At $n=10^4$ a frequentist rejects $\mu=0$ at the 5% level while a Bayesian with a unit-scale prior assigns the null probability $0.936$. (Independent check: evaluating $m_0$ and $m_1$ by numerical quadrature at $n=10^4$ gives $\mathrm{BF}_{01}=14.653552$, matching the closed form to six decimals.) And the answer depends on $\tau$:

| $\tau$ | 0.1 | 0.5 | 1.0 | 5.0 | 20.0 |
|---|---|---|---|---|---|
| $\mathrm{BF}_{01}$ at $n=10^4$ | 1.50 | 7.33 | 14.65 | 73.25 | 293.0 |

How should you feel about this? Both parties are computing correctly, and they are answering different questions. The $p$-value asks: *if $\mu$ were exactly zero, how surprising is a deviation this large?* Answer: mildly, 5%. The Bayes factor asks: *which of two models predicted this dataset better?* $M_1$ spread its predictions over a range of width $\tau=1$, so it wasted almost all its mass on values of $\bar x$ far from $0.0196$; $M_0$ predicted $\bar x$ within $\pm 0.02$ and was nearly right. On the *prediction* question the null wins easily. The paradox dissolves once you notice that $\mu=0$ exactly is usually a straw man: in most applications the honest $M_0$ is $|\mu|<\delta$ for some scientifically negligible $\delta$, and formulating it that way removes the pathology. The lesson for ML: a "statistically significant" improvement at $n=10^6$ may be a decisive *loss* on predictive grounds, which is why effect sizes and predictive scores, not $p$-values, are the right currency for large-$n$ comparisons.

### 10.5.4 Laplace approximation, BIC, AIC, WAIC, PSIS-LOO

The evidence is an integral, and mostly not a tractable one. The cheapest approximation expands the log-posterior around its mode.

**Theorem 10.30 (Laplace approximation to the evidence).** Let $\Theta\subseteq\mathbb{R}^d$ be open and let $q(\theta):=p(D\mid\theta)\pi(\theta)=\exp\{n g_n(\theta)\}$ for data $D=(x_1,\dots,x_n)$ i.i.d., where $g_n\to g$ uniformly on compacts with $g$ three times continuously differentiable. Assume: $q$ has a unique maximizer $\hat\theta$ in the interior of $\Theta$, with $\hat\theta\to\theta_0$ interior; $H:=-\nabla^2\log q(\hat\theta)$ satisfies $H=nH_1+o(n)$ with $H_1\succ0$; the third derivatives of $g_n$ are bounded uniformly in $n$ on a neighbourhood of $\theta_0$; and $\sup_{\|\theta-\theta_0\|>\delta}g_n(\theta)<g(\theta_0)-c(\delta)$ for every $\delta>0$ and some $c(\delta)>0$. Define
$$
\log Z_{\mathrm{Lap}}:=\log q(\hat\theta)+\frac d2\log(2\pi)-\frac12\log\det H .
$$
Then $\log Z=\log Z_{\mathrm{Lap}}+O(n^{-1})$.

The four hypotheses are not decoration: they are, in order, "there is one mode and it is not on a boundary", "the curvature grows like $n$ and is non-degenerate", "the cubic remainder is controlled", and "no mass hides far from the mode". Every failure mode in §10.7.1 is the violation of one of them.

*Proof.* Write $\log q(\theta)=\log q(\hat\theta)-\frac12(\theta-\hat\theta)^{\!\top}H(\theta-\hat\theta)+R(\theta)$ with $R(\theta)=O(n\|\theta-\hat\theta\|^3)$ by Taylor's theorem with remainder, the gradient vanishing because $\hat\theta$ is an interior maximum; the constant in $R$ is uniform by the third-derivative hypothesis. Then
$$
Z=q(\hat\theta)\int e^{-\frac12(\theta-\hat\theta)^\top H(\theta-\hat\theta)}e^{R(\theta)}\mathrm{d}\theta .
$$
Substituting $u=\sqrt n(\theta-\hat\theta)$ in the i.i.d. case makes the cubic remainder $O(n^{-1/2})$ uniformly on $\|u\|\le n^{1/6}$, so $e^{R}=1+O(n^{-1/2})$ there and the Gaussian integral $\int e^{-\frac12u^\top H_1u}\mathrm{d}u=(2\pi)^{d/2}(\det H_1)^{-1/2}$ applies; the tail region $\|u\|>n^{1/6}$ contributes exponentially little. Taking logs and re-expressing in terms of $H=nH_1$ gives the display; the odd-order term integrates to zero against the symmetric Gaussian, which is why the correction is $O(n^{-1})$ and not $O(n^{-1/2})$. $\blacksquare$

**Corollary 10.31 (BIC).** In the i.i.d. setting with $\hat\theta\to\theta_0$ and $H=nI(\theta_0)(1+o_P(1))$, and with a prior density continuous and positive at $\theta_0$,
$$
\log Z=\ell(\hat\theta)-\frac d2\log n+O_P(1),
\qquad \mathrm{BIC}:=-2\ell(\hat\theta)+d\log n\approx-2\log Z .
$$
*Proof.* Substitute $\log\det H=d\log n+\log\det I(\theta_0)+o_P(1)$ into Theorem 10.30 and absorb $\log\pi(\hat\theta)$, $\frac d2\log2\pi$, and $-\frac12\log\det I(\theta_0)$ into the $O_P(1)$ term. $\blacksquare$

So BIC is the Laplace approximation with the prior and the curvature scale thrown away. That is a substantial amputation: the discarded terms are $O(1)$, hence do *not* vanish as $n\to\infty$, so BIC differences do not converge to log Bayes-factor differences. What BIC does inherit is consistency for model *selection* when the true model is in the candidate set, because the $\frac d2\log n$ term dominates $O(1)$ errors.

**Worked Example 10.17 (how good are Laplace and BIC?).** Binomial model, uniform prior, $\hat p=0.65$ held fixed as $n$ grows. Exact $\log Z=\log B(1+s,1+n-s)$.

| $n$ | exact $\log Z$ | Laplace error | BIC-style error |
|---|---|---|---|
| 20 | $-14.3028$ | $+0.034651$ | $-0.143985$ |
| 200 | $-131.9634$ | $+0.003573$ | $-0.175063$ |
| 2 000 | $-1298.5155$ | $+0.000358$ | $-0.178278$ |
| 20 000 | $-12953.7059$ | $+0.000036$ | $-0.178600$ |
| 200 000 | $-129495.2522$ | $+0.000004$ | $-0.178632$ |

The Laplace error falls like $1/n$ exactly as Theorem 10.30 predicts (each tenfold increase in $n$ divides it by ten). The BIC error converges to the constant $-0.1786$ and stays there — visible confirmation that BIC is off by $O(1)$ forever. For comparing models whose log-evidences differ by less than $0.2$, BIC is therefore useless; for comparing models whose dimensions differ, it is usually fine.

**AIC is not an approximation to the evidence at all.** $\mathrm{AIC}=-2\ell(\hat\theta)+2d$ estimates the expected out-of-sample log predictive density of the *plug-in* model, derived from a bias correction on the training log-likelihood; its target is predictive accuracy at fixed $n$, not the marginal likelihood. Hence AIC and BIC disagree systematically (BIC penalizes by $\frac d2\log n$, AIC by $d$), and the disagreement is not a bug in either: they estimate different quantities. Do not average them.

**Definition 10.32 (elpd, WAIC, PSIS-LOO).** For a model with posterior $\Pi(\cdot\mid D)$, the *expected log pointwise predictive density* for new data is
$$
\mathrm{elpd}=\sum_{i=1}^{n}\mathbb{E}_{\tilde x_i\sim P_{\text{true}}}\big[\log p(\tilde x_i\mid D)\big].
$$
Two standard estimators from posterior draws $\theta^{(1)},\dots,\theta^{(S)}$:
$$
\widehat{\mathrm{WAIC}}=\sum_{i=1}^n\Big[\log\frac1S\sum_s p(x_i\mid\theta^{(s)})-\widehat{\mathrm{Var}}_s\big(\log p(x_i\mid\theta^{(s)})\big)\Big],
$$
$$
\widehat{\mathrm{elpd}}_{\text{LOO}}=\sum_{i=1}^n\log \hat p(x_i\mid D_{-i}),\qquad
\hat p(x_i\mid D_{-i})=\frac{\sum_{s} w_i^{(s)}\,p(x_i\mid\theta^{(s)})}{\sum_{s}w_i^{(s)}} .
$$
The second estimator reuses the *full*-data posterior draws to approximate each leave-one-out posterior by importance sampling. The correct raw importance ratio is
$$
r_i^{(s)}=\frac{\pi(\theta^{(s)}\mid D_{-i})}{\pi(\theta^{(s)}\mid D)}\ \propto\ \frac{1}{p(x_i\mid\theta^{(s)})},
$$
because dropping $x_i$ divides the posterior density by that one likelihood factor; the $\propto$ hides an $s$-free constant that cancels in the self-normalized ratio above. These ratios are heavy-tailed — the dropped point is exactly the one the full posterior fits *too* well — so one replaces $r_i^{(s)}$ by smoothed weights $w_i^{(s)}$ obtained by fitting a generalized Pareto distribution to the largest ratios. That is *Pareto-smoothed importance sampling*, and its shape parameter estimate $\hat k$ (Ch. 2's importance-sampling diagnostic) is the reason to prefer it to plain importance sampling: $\hat k>0.7$ signals ratios with infinite variance, and the estimate for that $i$ should be discarded or computed by refitting.

The practical hierarchy in modern applied work: use PSIS-LOO with its $\hat k$ diagnostic when you can draw from the posterior, WAIC when LOO is unstable, BIC when you have only a maximum-likelihood fit and the models differ substantially in dimension, and marginal likelihood only when you can defend your priors. In deep learning all of these are typically replaced by a held-out set, which is the honest choice when $d\gg n$ and no asymptotic expansion applies.

> **Common Confusion 10.6.** *"The marginal likelihood is just the likelihood at the best parameters, so of course the bigger model always wins — that's why we need BIC's penalty."*
>
> Two errors in one sentence. First, $p(D\mid M)=\int p(D\mid\theta)\pi(\theta)\mathrm{d}\theta$ is an *average*, not a maximum, and averaging a sharply peaked likelihood against a diffuse prior gives something far smaller than the peak: in Worked Example 10.15 the maximized likelihood for $M_1$ is $0.65^{13}0.35^{7}=2.3788\times10^{-6}$ while its evidence is $6.1428\times10^{-7}$, smaller by a factor of $3.87$. Second, precisely because of this, the bigger model does *not* always win — at $s=13$ it loses. BIC's penalty is not an addition to the marginal likelihood; it is an approximation *of* it (Corollary 10.31). If you find yourself adding a complexity penalty to a marginal likelihood, you are double-counting.
>
> The mirror-image error is also common: "the evidence is basically cross-validation, so I can use a vague prior." Proposition 10.28(i) does make the evidence a prequential predictive score, but it is the score of the model *including its prior*, and §10.5.2 shows a vague prior degrades that score without bound. Cross-validation scores the posterior predictive after seeing $n-1$ points; the evidence scores predictions made from as little as zero points. They differ most exactly where the prior matters most.

> **Self-check 10.5.**
> 1. Why does $p(s\mid M_1)=1/21$ for all $s$ in Worked Example 10.15? Name the proposition and the mechanism.
> 2. In Proposition 10.29, what happens to $\mathrm{BF}_{01}$ as $\tau\to0$ with $n,z$ fixed, and why is that the right answer?
> 3. The BIC error in Worked Example 10.17 converges to $-0.1786$ rather than to $0$. Which discarded terms of Theorem 10.30 does that constant contain?
> 4. Your colleague reports a Bayes factor of 30 in favour of a model whose prior was $\mathrm{Uniform}(-10^6,10^6)$ on a coefficient. What one question do you ask?

## 10.6 Bayesian linear regression, fully derived

### 10.6.1 Setup and posterior

This section is the chapter's showcase: a model where every object we have defined — posterior, predictive, evidence, MAP, shrinkage — is available in closed form, and where the Bayesian answer connects to a method you already use.

Fix a feature map $\varphi:\mathcal{X}\to\mathbb{R}^d$ and write $\Phi\in\mathbb{R}^{n\times d}$ for the design matrix with rows $\varphi(x_i)^{\!\top}$. The model is
$$
y\mid\theta\ \sim\ N(\Phi\theta,\ \sigma^2I_n),\qquad \theta\ \sim\ N(0,\ \tau^2I_d),
$$
with $\sigma^2>0$ and $\tau^2>0$ known for now (§10.6.3 estimates them). The features may be arbitrary — polynomials, random Fourier features, the penultimate layer of a trained network — because linearity is required only in $\theta$.

**Theorem 10.33 (posterior).** With the model above,
$$
\theta\mid y\ \sim\ N(m,\ \Sigma),\qquad
\Sigma=\Big(\frac{\Phi^{\!\top}\Phi}{\sigma^2}+\frac{I_d}{\tau^2}\Big)^{-1},\qquad
m=\Sigma\,\frac{\Phi^{\!\top}y}{\sigma^2}.
$$

*Proof.* Up to $\theta$-free additive constants,
$$
\log p(\theta\mid y)=-\frac{1}{2\sigma^2}\|y-\Phi\theta\|^2-\frac{1}{2\tau^2}\|\theta\|^2
=-\frac12\theta^{\!\top}\underbrace{\Big(\frac{\Phi^{\!\top}\Phi}{\sigma^2}+\frac{I}{\tau^2}\Big)}_{=:A}\theta+\theta^{\!\top}\frac{\Phi^{\!\top}y}{\sigma^2}+\text{const}.
$$
$A$ is positive definite ($\Phi^{\!\top}\Phi\succeq0$ and $I/\tau^2\succ0$), so completing the square (Ch. 5) gives $-\frac12(\theta-A^{-1}\Phi^{\!\top}y/\sigma^2)^{\!\top}A(\theta-\cdots)+\text{const}$. A density proportional to the exponential of that quadratic is the stated Gaussian, by uniqueness of the Gaussian's canonical form. $\blacksquare$

**Corollary 10.34 (the posterior mean is the ridge estimator).** With $\lambda:=\sigma^2/\tau^2$,
$$
m=\big(\Phi^{\!\top}\Phi+\lambda I_d\big)^{-1}\Phi^{\!\top}y=\hat\theta^{\text{ridge}}_\lambda ,
$$
and $m$ is also the MAP estimate, since the posterior is Gaussian (mean = mode). In particular ridge regression *is* Bayesian linear regression's posterior mean, and the regularization strength is a ratio of variances: noise variance over prior variance.

*Proof.* Multiply the expression for $\Sigma^{-1}$ by $\sigma^2$: $\sigma^2A=\Phi^{\!\top}\Phi+\lambda I$. Then $m=A^{-1}\Phi^{\!\top}y/\sigma^2=(\sigma^2A)^{-1}\Phi^{\!\top}y=(\Phi^{\!\top}\Phi+\lambda I)^{-1}\Phi^{\!\top}y$. Gaussian mean equals mode by symmetry. $\blacksquare$

This identity is more informative than it looks. It tells you what $\lambda$ *means*: $\lambda$ large says either that the data are noisy ($\sigma^2$ large) or that you expected small coefficients ($\tau^2$ small), and those are different statements with the same consequence for the point estimate but different consequences for the predictive variance. It also tells you what ridge cannot express, which is the subject of §10.6.2.

**Worked Example 10.18 (ridge equality, checked to machine precision).** Data: $n=12$ points, $x_i\sim\mathrm{Uniform}(-1,1)$ sorted, $y_i=1+2x_i-1.5x_i^2+\varepsilon_i$ with $\varepsilon_i\sim N(0,0.25^2)$, seed 10. Features: cubic polynomial basis $\varphi(x)=(1,x,x^2,x^3)$, so $d=4$. Take $\sigma=0.25$ (known) and $\tau^2=1$, hence $\lambda=\sigma^2/\tau^2=0.0625$.

| quantity | value |
|---|---|
| posterior mean $m$ | $(0.737955,\ 1.878058,\ -1.150220,\ 0.127966)$ |
| ridge $(\Phi^{\!\top}\Phi+0.0625I)^{-1}\Phi^{\!\top}y$ | $(0.737955,\ 1.878058,\ -1.150220,\ 0.127966)$ |
| $\max_j$ absolute difference | $1.78\times10^{-15}$ (machine precision; the exact digit depends on the linear-algebra routine) |
| OLS ($\lambda=0$) | $(0.782847,\ 2.037462,\ -1.249765,\ -0.079815)$ |
| posterior sd | $(0.127308,\ 0.294124,\ 0.312342,\ 0.526745)$ |

The agreement is to machine precision, as Corollary 10.34 requires. Two things to notice beyond the identity. The cubic coefficient's posterior is $0.128\pm0.527$: the data are entirely consistent with there being no cubic term, and also with a substantial one. And OLS puts the cubic coefficient at $-0.080$ while the posterior mean puts it at $+0.128$ — a sign flip produced by a shrinkage of $\lambda=0.0625$, which should discourage you from reading signs off weakly identified coefficients in any regularized model.

### 10.6.2 The posterior predictive, and what a point estimate cannot say

**Proposition 10.35 (posterior predictive and its variance decomposition).** For a new input with features $\varphi_*:=\varphi(x_*)$, the posterior predictive of $y_*=\varphi_*^{\!\top}\theta+\varepsilon$, $\varepsilon\sim N(0,\sigma^2)$ independent, is
$$
y_*\mid y\ \sim\ N\big(\varphi_*^{\!\top}m,\ \ \underbrace{\varphi_*^{\!\top}\Sigma\varphi_*}_{\text{epistemic}}+\underbrace{\sigma^2}_{\text{aleatoric}}\big).
$$

*Proof.* Given $y$, $\theta\sim N(m,\Sigma)$, so $\varphi_*^{\!\top}\theta\sim N(\varphi_*^{\!\top}m,\varphi_*^{\!\top}\Sigma\varphi_*)$ (affine images of Gaussians are Gaussian, Ch. 5); adding independent $N(0,\sigma^2)$ noise adds variances. The decomposition is Proposition 10.7 with $\mu(\theta)=\varphi_*^{\!\top}\theta$ and $v(\theta)\equiv\sigma^2$. $\blacksquare$

The epistemic term $\varphi_*^{\!\top}\Sigma\varphi_*$ is the whole payoff of this section. It is small where you have data and large where you do not, and it does this without being told where the data are — the information is in $\Sigma$.

**Worked Example 10.19 (the two variance components, tabulated).** Same fit as Worked Example 10.18; $\sigma^2=0.0625$ throughout.

| $x_*$ | predictive mean | epistemic var | aleatoric var | total sd | 95% predictive interval |
|---|---|---|---|---|---|
| $-1.00$ | $-2.4183$ | 0.216670 | 0.0625 | 0.5284 | $[-3.4539,\ -1.3827]$ |
| $-0.50$ | $-0.5046$ | 0.014281 | 0.0625 | 0.2771 | $[-1.0477,\ \ 0.0385]$ |
| $0.00$ | $0.7380$ | 0.016207 | 0.0625 | 0.2805 | $[\ \ 0.1881,\ \ 1.2878]$ |
| $0.50$ | $1.4054$ | 0.015811 | 0.0625 | 0.2798 | $[\ \ 0.8569,\ \ 1.9539]$ |
| $1.00$ | $1.5938$ | 0.055870 | 0.0625 | 0.3440 | $[\ \ 0.9194,\ \ 2.2681]$ |
| $1.50$ | $1.3989$ | 1.462845 | 0.0625 | 1.2350 | $[-1.0217,\ \ 3.8196]$ |

In the middle of the data the epistemic term is $0.0162$, a quarter of the noise term: the model knows the function about as well as the noise permits. Half a unit outside the data range it is $1.4628$, twenty-three times the noise. The point estimate at $x_*=1.5$ is $1.40$ in both cases; only the Bayesian version reports that this number is worth $\pm2.4$.

This is not an academic nicety. Extrapolation-aware uncertainty is exactly what active learning and Bayesian optimization exploit (query where $\varphi_*^{\!\top}\Sigma\varphi_*$ is large), what safe deployment needs (abstain where it is large), and what a ridge point estimate structurally cannot provide. Figure 10.5 shows the bands.

<svg viewBox="0 0 760 470" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="sans-serif">
  <polygon points="70.0,211.3 73.6,211.9 77.3,212.5 80.9,213.0 84.5,213.4 88.2,213.8 91.8,214.2 95.4,214.4 99.1,214.7 102.7,214.8 106.3,215.0 109.9,215.0 113.6,215.0 117.2,215.0 120.8,214.9 124.5,214.8 128.1,214.6 131.7,214.3 135.4,214.0 139.0,213.7 142.6,213.3 146.3,212.8 149.9,212.4 153.5,211.8 157.2,211.2 160.8,210.6 164.4,209.9 168.0,209.2 171.7,208.4 175.3,207.5 178.9,206.7 182.6,205.7 186.2,204.7 189.8,203.7 193.5,202.6 197.1,201.5 200.7,200.3 204.4,199.1 208.0,197.8 211.6,196.5 215.3,195.1 218.9,193.7 222.5,192.2 226.1,190.7 229.8,189.2 233.4,187.6 237.0,186.0 240.7,184.4 244.3,182.7 247.9,181.1 251.6,179.4 255.2,177.7 258.8,175.9 262.5,174.2 266.1,172.5 269.7,170.8 273.4,169.1 277.0,167.4 280.6,165.7 284.2,164.0 287.9,162.4 291.5,160.7 295.1,159.1 298.8,157.5 302.4,155.9 306.0,154.4 309.7,152.8 313.3,151.3 316.9,149.9 320.6,148.4 324.2,147.0 327.8,145.6 331.5,144.2 335.1,142.9 338.7,141.5 342.3,140.2 346.0,139.0 349.6,137.7 353.2,136.5 356.9,135.3 360.5,134.1 364.1,133.0 367.8,131.8 371.4,130.7 375.0,129.6 378.7,128.6 382.3,127.5 385.9,126.5 389.6,125.5 393.2,124.5 396.8,123.5 400.4,122.6 404.1,121.7 407.7,120.8 411.3,119.9 415.0,119.0 418.6,118.1 422.2,117.3 425.9,116.5 429.5,115.6 433.1,114.9 436.8,114.1 440.4,113.3 444.0,112.6 447.7,111.9 451.3,111.2 454.9,110.5 458.5,109.8 462.2,109.2 465.8,108.6 469.4,108.0 473.1,107.4 476.7,106.8 480.3,106.3 484.0,105.7 487.6,105.2 491.2,104.8 494.9,104.3 498.5,103.9 502.1,103.4 505.8,103.0 509.4,102.7 513.0,102.3 516.6,102.0 520.3,101.7 523.9,101.4 527.5,101.1 531.2,100.8 534.8,100.6 538.4,100.3 542.1,100.1 545.7,99.9 549.3,99.7 553.0,99.5 556.6,99.2 560.2,99.0 563.9,98.8 567.5,98.6 571.1,98.4 574.7,98.1 578.4,97.8 582.0,97.5 585.6,97.2 589.3,96.8 592.9,96.4 596.5,95.9 600.2,95.4 603.8,94.9 607.4,94.2 611.1,93.6 614.7,92.8 618.3,92.1 622.0,91.2 625.6,90.3 629.2,89.3 632.8,88.3 636.5,87.2 640.1,86.0 643.7,84.8 647.4,83.5 651.0,82.2 654.6,80.8 658.3,79.3 661.9,77.7 665.5,76.1 669.2,74.5 672.8,72.7 676.4,70.9 680.1,69.1 683.7,67.1 687.3,65.1 690.9,63.1 694.6,60.9 698.2,58.7 701.8,56.5 705.5,54.1 709.1,51.7 712.7,49.2 716.4,46.7 720.0,44.1 720.0,197.8 716.4,194.4 712.7,191.2 709.1,188.0 705.5,184.9 701.8,181.9 698.2,179.1 694.6,176.3 690.9,173.5 687.3,170.9 683.7,168.4 680.1,165.9 676.4,163.6 672.8,161.3 669.2,159.1 665.5,157.0 661.9,155.0 658.3,153.1 654.6,151.2 651.0,149.5 647.4,147.8 643.7,146.3 640.1,144.8 636.5,143.4 632.8,142.1 629.2,140.9 625.6,139.8 622.0,138.7 618.3,137.8 614.7,136.9 611.1,136.1 607.4,135.5 603.8,134.8 600.2,134.3 596.5,133.9 592.9,133.5 589.3,133.2 585.6,133.0 582.0,132.8 578.4,132.7 574.7,132.6 571.1,132.6 567.5,132.7 563.9,132.8 560.2,132.9 556.6,133.1 553.0,133.3 549.3,133.5 545.7,133.8 542.1,134.0 538.4,134.3 534.8,134.7 531.2,135.0 527.5,135.4 523.9,135.7 520.3,136.1 516.6,136.5 513.0,137.0 509.4,137.4 505.8,137.8 502.1,138.3 498.5,138.8 494.9,139.2 491.2,139.7 487.6,140.2 484.0,140.8 480.3,141.3 476.7,141.8 473.1,142.4 469.4,143.0 465.8,143.6 462.2,144.2 458.5,144.8 454.9,145.5 451.3,146.1 447.7,146.8 444.0,147.5 440.4,148.2 436.8,149.0 433.1,149.7 429.5,150.5 425.9,151.3 422.2,152.1 418.6,153.0 415.0,153.8 411.3,154.7 407.7,155.6 404.1,156.5 400.4,157.5 396.8,158.4 393.2,159.4 389.6,160.5 385.9,161.5 382.3,162.5 378.7,163.6 375.0,164.7 371.4,165.8 367.8,167.0 364.1,168.1 360.5,169.3 356.9,170.5 353.2,171.7 349.6,173.0 346.0,174.2 342.3,175.5 338.7,176.8 335.1,178.1 331.5,179.4 327.8,180.7 324.2,182.1 320.6,183.5 316.9,184.9 313.3,186.3 309.7,187.8 306.0,189.2 302.4,190.7 298.8,192.2 295.1,193.7 291.5,195.3 287.9,196.9 284.2,198.5 280.6,200.1 277.0,201.8 273.4,203.5 269.7,205.2 266.1,207.0 262.5,208.8 258.8,210.7 255.2,212.6 251.6,214.6 247.9,216.7 244.3,218.8 240.7,221.0 237.0,223.3 233.4,225.6 229.8,228.0 226.1,230.5 222.5,233.1 218.9,235.8 215.3,238.6 211.6,241.5 208.0,244.5 204.4,247.5 200.7,250.7 197.1,254.0 193.5,257.4 189.8,260.9 186.2,264.4 182.6,268.1 178.9,271.9 175.3,275.8 171.7,279.8 168.0,283.9 164.4,288.1 160.8,292.4 157.2,296.8 153.5,301.3 149.9,305.9 146.3,310.7 142.6,315.5 139.0,320.0 135.4,320.0 131.7,320.0 128.1,320.0 124.5,320.0 120.8,320.0 117.2,320.0 113.6,320.0 109.9,320.0 106.3,320.0 102.7,320.0 99.1,320.0 95.4,320.0 91.8,320.0 88.2,320.0 84.5,320.0 80.9,320.0 77.3,320.0 73.6,320.0 70.0,320.0" fill="#3b82f6" opacity="0.16"/>
  <polygon points="70.0,212.4 73.6,213.0 77.3,213.6 80.9,214.2 84.5,214.7 88.2,215.1 91.8,215.5 95.4,215.8 99.1,216.1 102.7,216.4 106.3,216.5 109.9,216.7 113.6,216.7 117.2,216.8 120.8,216.8 124.5,216.7 128.1,216.6 131.7,216.4 135.4,216.2 139.0,216.0 142.6,215.7 146.3,215.4 149.9,215.0 153.5,214.6 157.2,214.1 160.8,213.7 164.4,213.1 168.0,212.5 171.7,211.9 175.3,211.3 178.9,210.6 182.6,209.9 186.2,209.1 189.8,208.3 193.5,207.5 197.1,206.6 200.7,205.7 204.4,204.7 208.0,203.8 211.6,202.7 215.3,201.7 218.9,200.6 222.5,199.4 226.1,198.2 229.8,197.0 233.4,195.7 237.0,194.4 240.7,193.1 244.3,191.7 247.9,190.2 251.6,188.7 255.2,187.2 258.8,185.6 262.5,184.0 266.1,182.3 269.7,180.6 273.4,178.9 277.0,177.2 280.6,175.5 284.2,173.8 287.9,172.1 291.5,170.5 295.1,168.8 298.8,167.2 302.4,165.5 306.0,163.9 309.7,162.4 313.3,160.8 316.9,159.3 320.6,157.8 324.2,156.4 327.8,155.0 331.5,153.6 335.1,152.2 338.7,150.9 342.3,149.6 346.0,148.3 349.6,147.1 353.2,145.9 356.9,144.7 360.5,143.5 364.1,142.4 367.8,141.2 371.4,140.2 375.0,139.1 378.7,138.0 382.3,137.0 385.9,136.0 389.6,135.0 393.2,134.0 396.8,133.1 400.4,132.2 404.1,131.2 407.7,130.3 411.3,129.4 415.0,128.6 418.6,127.7 422.2,126.9 425.9,126.0 429.5,125.2 433.1,124.4 436.8,123.6 440.4,122.9 444.0,122.1 447.7,121.4 451.3,120.7 454.9,120.0 458.5,119.3 462.2,118.7 465.8,118.0 469.4,117.4 473.1,116.8 476.7,116.3 480.3,115.7 484.0,115.2 487.6,114.7 491.2,114.3 494.9,113.8 498.5,113.4 502.1,113.0 505.8,112.7 509.4,112.3 513.0,112.0 516.6,111.7 520.3,111.5 523.9,111.2 527.5,111.0 531.2,110.8 534.8,110.6 538.4,110.4 542.1,110.3 545.7,110.1 549.3,109.9 553.0,109.7 556.6,109.5 560.2,109.3 563.9,109.0 567.5,108.7 571.1,108.3 574.7,107.9 578.4,107.4 582.0,106.8 585.6,106.2 589.3,105.5 592.9,104.8 596.5,104.0 600.2,103.2 603.8,102.3 607.4,101.3 611.1,100.3 614.7,99.3 618.3,98.2 622.0,97.0 625.6,95.8 629.2,94.6 632.8,93.3 636.5,91.9 640.1,90.5 643.7,89.0 647.4,87.5 651.0,86.0 654.6,84.4 658.3,82.7 661.9,81.0 665.5,79.2 669.2,77.4 672.8,75.5 676.4,73.6 680.1,71.6 683.7,69.6 687.3,67.5 690.9,65.3 694.6,63.1 698.2,60.8 701.8,58.4 705.5,56.0 709.1,53.5 712.7,51.0 716.4,48.3 720.0,45.7 720.0,196.2 716.4,192.8 712.7,189.4 709.1,186.2 705.5,183.1 701.8,180.0 698.2,177.0 694.6,174.1 690.9,171.3 687.3,168.6 683.7,165.9 680.1,163.4 676.4,160.9 672.8,158.5 669.2,156.1 665.5,153.9 661.9,151.7 658.3,149.6 654.6,147.6 651.0,145.7 647.4,143.8 643.7,142.1 640.1,140.3 636.5,138.7 632.8,137.2 629.2,135.7 625.6,134.3 622.0,132.9 618.3,131.7 614.7,130.5 611.1,129.4 607.4,128.4 603.8,127.4 600.2,126.6 596.5,125.8 592.9,125.1 589.3,124.5 585.6,123.9 582.0,123.5 578.4,123.1 574.7,122.9 571.1,122.7 567.5,122.6 563.9,122.6 560.2,122.7 556.6,122.8 553.0,123.0 549.3,123.2 545.7,123.5 542.1,123.9 538.4,124.2 534.8,124.6 531.2,125.0 527.5,125.4 523.9,125.9 520.3,126.3 516.6,126.8 513.0,127.3 509.4,127.7 505.8,128.2 502.1,128.7 498.5,129.2 494.9,129.7 491.2,130.2 487.6,130.8 484.0,131.3 480.3,131.8 476.7,132.4 473.1,132.9 469.4,133.5 465.8,134.1 462.2,134.7 458.5,135.3 454.9,136.0 451.3,136.6 447.7,137.3 444.0,138.0 440.4,138.7 436.8,139.4 433.1,140.2 429.5,140.9 425.9,141.7 422.2,142.5 418.6,143.4 415.0,144.2 411.3,145.1 407.7,146.0 404.1,147.0 400.4,147.9 396.8,148.9 393.2,149.9 389.6,150.9 385.9,152.0 382.3,153.1 378.7,154.2 375.0,155.3 371.4,156.4 367.8,157.6 364.1,158.7 360.5,159.9 356.9,161.1 353.2,162.4 349.6,163.6 346.0,164.9 342.3,166.1 338.7,167.4 335.1,168.7 331.5,170.0 327.8,171.4 324.2,172.7 320.6,174.1 316.9,175.4 313.3,176.8 309.7,178.2 306.0,179.6 302.4,181.1 298.8,182.5 295.1,184.0 291.5,185.5 287.9,187.1 284.2,188.6 280.6,190.2 277.0,191.9 273.4,193.6 269.7,195.4 266.1,197.2 262.5,199.1 258.8,201.1 255.2,203.2 251.6,205.3 247.9,207.6 244.3,209.9 240.7,212.3 237.0,214.9 233.4,217.5 229.8,220.2 226.1,223.0 222.5,226.0 218.9,229.0 215.3,232.1 211.6,235.2 208.0,238.5 204.4,241.9 200.7,245.3 197.1,248.9 193.5,252.5 189.8,256.3 186.2,260.1 182.6,264.0 178.9,268.0 175.3,272.1 171.7,276.2 168.0,280.5 164.4,284.9 160.8,289.3 157.2,293.9 153.5,298.5 149.9,303.3 146.3,308.1 142.6,313.1 139.0,318.1 135.4,320.0 131.7,320.0 128.1,320.0 124.5,320.0 120.8,320.0 117.2,320.0 113.6,320.0 109.9,320.0 106.3,320.0 102.7,320.0 99.1,320.0 95.4,320.0 91.8,320.0 88.2,320.0 84.5,320.0 80.9,320.0 77.3,320.0 73.6,320.0 70.0,320.0" fill="#10b981" opacity="0.28"/>
  <line x1="70.00" y1="164.44" x2="720.00" y2="164.44" stroke="currentColor" stroke-width="0.8" stroke-dasharray="2,4"/>
  <line x1="70.00" y1="320.00" x2="720.00" y2="320.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="70.00" y1="320.00" x2="70.00" y2="40.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="70.00" y1="320.00" x2="70.00" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="70.00" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">-1.5</text>
  <line x1="178.33" y1="320.00" x2="178.33" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="178.33" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">-1.0</text>
  <line x1="286.67" y1="320.00" x2="286.67" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="286.67" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">-0.5</text>
  <line x1="395.00" y1="320.00" x2="395.00" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="395.00" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.0</text>
  <line x1="503.33" y1="320.00" x2="503.33" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="503.33" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.5</text>
  <line x1="611.67" y1="320.00" x2="611.67" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="611.67" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">1.0</text>
  <line x1="720.00" y1="320.00" x2="720.00" y2="325.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="720.00" y="339.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">1.5</text>
  <line x1="65.00" y1="288.89" x2="70.00" y2="288.89" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="292.89" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">-4</text>
  <line x1="65.00" y1="226.67" x2="70.00" y2="226.67" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="230.67" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">-2</text>
  <line x1="65.00" y1="164.44" x2="70.00" y2="164.44" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="168.44" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0</text>
  <line x1="65.00" y1="102.22" x2="70.00" y2="102.22" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="106.22" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">2</text>
  <line x1="65.00" y1="40.00" x2="70.00" y2="40.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="61.00" y="44.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">4</text>
  <polyline points="70.0,323.1 73.6,319.9 77.3,316.7 80.9,313.5 84.5,310.4 88.2,307.3 91.8,304.2 95.4,301.2 99.1,298.2 102.7,295.2 106.3,292.3 109.9,289.4 113.6,286.5 117.2,283.6 120.8,280.8 124.5,278.0 128.1,275.2 131.7,272.5 135.4,269.7 139.0,267.0 142.6,264.4 146.3,261.8 149.9,259.1 153.5,256.6 157.2,254.0 160.8,251.5 164.4,249.0 168.0,246.5 171.7,244.1 175.3,241.7 178.9,239.3 182.6,236.9 186.2,234.6 189.8,232.3 193.5,230.0 197.1,227.7 200.7,225.5 204.4,223.3 208.0,221.1 211.6,219.0 215.3,216.9 218.9,214.8 222.5,212.7 226.1,210.6 229.8,208.6 233.4,206.6 237.0,204.6 240.7,202.7 244.3,200.8 247.9,198.9 251.6,197.0 255.2,195.2 258.8,193.3 262.5,191.5 266.1,189.8 269.7,188.0 273.4,186.3 277.0,184.6 280.6,182.9 284.2,181.2 287.9,179.6 291.5,178.0 295.1,176.4 298.8,174.8 302.4,173.3 306.0,171.8 309.7,170.3 313.3,168.8 316.9,167.4 320.6,165.9 324.2,164.5 327.8,163.2 331.5,161.8 335.1,160.5 338.7,159.1 342.3,157.9 346.0,156.6 349.6,155.3 353.2,154.1 356.9,152.9 360.5,151.7 364.1,150.5 367.8,149.4 371.4,148.3 375.0,147.2 378.7,146.1 382.3,145.0 385.9,144.0 389.6,143.0 393.2,142.0 396.8,141.0 400.4,140.0 404.1,139.1 407.7,138.2 411.3,137.3 415.0,136.4 418.6,135.5 422.2,134.7 425.9,133.9 429.5,133.1 433.1,132.3 436.8,131.5 440.4,130.8 444.0,130.1 447.7,129.3 451.3,128.7 454.9,128.0 458.5,127.3 462.2,126.7 465.8,126.1 469.4,125.5 473.1,124.9 476.7,124.3 480.3,123.8 484.0,123.3 487.6,122.7 491.2,122.2 494.9,121.8 498.5,121.3 502.1,120.9 505.8,120.4 509.4,120.0 513.0,119.6 516.6,119.3 520.3,118.9 523.9,118.6 527.5,118.2 531.2,117.9 534.8,117.6 538.4,117.3 542.1,117.1 545.7,116.8 549.3,116.6 553.0,116.4 556.6,116.2 560.2,116.0 563.9,115.8 567.5,115.6 571.1,115.5 574.7,115.4 578.4,115.3 582.0,115.2 585.6,115.1 589.3,115.0 592.9,114.9 596.5,114.9 600.2,114.9 603.8,114.8 607.4,114.8 611.1,114.9 614.7,114.9 618.3,114.9 622.0,115.0 625.6,115.0 629.2,115.1 632.8,115.2 636.5,115.3 640.1,115.4 643.7,115.5 647.4,115.7 651.0,115.8 654.6,116.0 658.3,116.2 661.9,116.4 665.5,116.6 669.2,116.8 672.8,117.0 676.4,117.2 680.1,117.5 683.7,117.7 687.3,118.0 690.9,118.3 694.6,118.6 698.2,118.9 701.8,119.2 705.5,119.5 709.1,119.9 712.7,120.2 716.4,120.6 720.0,120.9" fill="none" stroke="#3b82f6" stroke-width="2.4"/>
  <polyline points="70.0,320.0 73.6,320.0 77.3,320.0 80.9,320.0 84.5,320.0 88.2,320.0 91.8,320.0 95.4,320.0 99.1,320.0 102.7,320.0 106.3,320.0 109.9,320.0 113.6,320.0 117.2,320.0 120.8,320.0 124.5,320.0 128.1,320.0 131.7,320.0 135.4,320.0 139.0,320.0 142.6,320.0 146.3,320.0 149.9,320.0 153.5,320.0 157.2,320.0 160.8,320.0 164.4,320.0 168.0,320.0 171.7,320.0 175.3,320.0 178.9,320.0 182.6,320.0 186.2,320.0 189.8,320.0 193.5,320.0 197.1,320.0 200.7,320.0 204.4,320.0 208.0,320.0 211.6,320.0 215.3,320.0 218.9,291.1 222.5,264.9 226.1,244.3 229.8,228.4 233.4,216.4 237.0,207.6 240.7,201.4 244.3,197.3 247.9,194.8 251.6,193.5 255.2,193.0 258.8,193.1 262.5,193.4 266.1,193.9 269.7,194.4 273.4,194.7 277.0,194.7 280.6,194.4 284.2,193.7 287.9,192.7 291.5,191.3 295.1,189.5 298.8,187.4 302.4,185.0 306.0,182.3 309.7,179.4 313.3,176.3 316.9,173.1 320.6,169.9 324.2,166.7 327.8,163.5 331.5,160.4 335.1,157.4 338.7,154.7 342.3,152.1 346.0,149.8 349.6,147.7 353.2,145.9 356.9,144.4 360.5,143.2 364.1,142.3 367.8,141.7 371.4,141.3 375.0,141.3 378.7,141.4 382.3,141.7 385.9,142.3 389.6,142.9 393.2,143.7 396.8,144.5 400.4,145.4 404.1,146.2 407.7,147.0 411.3,147.6 415.0,148.2 418.6,148.6 422.2,148.7 425.9,148.6 429.5,148.3 433.1,147.7 436.8,146.8 440.4,145.5 444.0,144.0 447.7,142.1 451.3,139.9 454.9,137.5 458.5,134.8 462.2,131.8 465.8,128.6 469.4,125.3 473.1,121.9 476.7,118.4 480.3,114.9 484.0,111.5 487.6,108.3 491.2,105.2 494.9,102.5 498.5,100.1 502.1,98.2 505.8,96.8 509.4,96.0 513.0,95.9 516.6,96.4 520.3,97.7 523.9,99.9 527.5,102.8 531.2,106.5 534.8,110.9 538.4,116.1 542.1,122.0 545.7,128.3 549.3,135.1 553.0,142.0 556.6,148.8 560.2,155.3 563.9,161.0 567.5,165.6 571.1,168.6 574.7,169.5 578.4,167.5 582.0,161.9 585.6,152.0 589.3,136.7 592.9,115.1 596.5,86.0 600.2,48.0 603.8,40.0 607.4,40.0 611.1,40.0 614.7,40.0 618.3,40.0 622.0,40.0 625.6,40.0 629.2,40.0 632.8,40.0 636.5,40.0 640.1,40.0 643.7,40.0 647.4,40.0 651.0,40.0 654.6,40.0 658.3,40.0 661.9,40.0 665.5,40.0 669.2,40.0 672.8,40.0 676.4,40.0 680.1,40.0 683.7,40.0 687.3,40.0 690.9,40.0 694.6,40.0 698.2,40.0 701.8,40.0 705.5,40.0 709.1,40.0 712.7,40.0 716.4,40.0 720.0,40.0" fill="none" stroke="#ef4444" stroke-width="2.0" stroke-dasharray="6,4"/>
  <circle cx="237.23" cy="206.06" r="4" fill="currentColor"/>
  <circle cx="243.02" cy="200.34" r="4" fill="currentColor"/>
  <circle cx="268.33" cy="193.22" r="4" fill="currentColor"/>
  <circle cx="324.89" cy="167.18" r="4" fill="currentColor"/>
  <circle cx="362.72" cy="141.26" r="4" fill="currentColor"/>
  <circle cx="400.55" cy="146.17" r="4" fill="currentColor"/>
  <circle cx="476.92" cy="117.86" r="4" fill="currentColor"/>
  <circle cx="535.98" cy="109.29" r="4" fill="currentColor"/>
  <circle cx="537.33" cy="120.26" r="4" fill="currentColor"/>
  <circle cx="543.09" cy="121.26" r="4" fill="currentColor"/>
  <circle cx="592.60" cy="121.32" r="4" fill="currentColor"/>
  <circle cx="593.00" cy="110.40" r="4" fill="currentColor"/>
  <line x1="80.00" y1="50.00" x2="106.00" y2="50.00" stroke="#3b82f6" stroke-width="2.4"/>
  <text x="114.00" y="54.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">posterior mean = ridge, lambda = 0.0625</text>
  <line x1="80.00" y1="70.00" x2="106.00" y2="70.00" stroke="#ef4444" stroke-width="2.0" stroke-dasharray="6,4"/>
  <text x="114.00" y="74.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">degree-7 least squares (no prior)</text>
  <rect x="80.00" y="85.00" width="26" height="10" fill="#10b981" opacity="0.28"/>
  <text x="114.00" y="94.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">+/- 2 sd, parameter uncertainty only</text>
  <rect x="80.00" y="105.00" width="26" height="10" fill="#3b82f6" opacity="0.16"/>
  <text x="114.00" y="114.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">+/- 2 sd, full predictive (parameters + noise)</text>
  <line x1="611.67" y1="40.00" x2="611.67" y2="320.00" stroke="currentColor" stroke-width="0.8" stroke-dasharray="2,4"/>
  <text x="616.00" y="312.00" font-size="10" fill="currentColor" text-anchor="start" font-weight="normal">edge of data</text>
  <text x="380.0" y="400" font-size="13" fill="currentColor" text-anchor="middle">Figure 10.5  Bayesian cubic regression, n = 12, sigma = 0.25 known, prior N(0, I) (seed 10).</text>
  <text x="380.0" y="417" font-size="13" fill="currentColor" text-anchor="middle">Inside the data the two bands are nearly parallel and noise dominates (parameter variance 0.016</text>
  <text x="380.0" y="434" font-size="13" fill="currentColor" text-anchor="middle">vs noise 0.0625 at x = 0); outside it, parameter variance explodes (1.463 at x = 1.5) and the band</text>
  <text x="380.0" y="451" font-size="13" fill="currentColor" text-anchor="middle">flares. The degree-7 least-squares fit has no way to say that it does not know.</text>
</svg>


### 10.6.3 Evidence: choosing $\lambda$ without a validation set

**Proposition 10.36 (evidence for Bayesian linear regression).** Under the model of §10.6.1,
$$
y\ \sim\ N\big(0,\ \sigma^2I_n+\tau^2\Phi\Phi^{\!\top}\big),
$$
so with $A=\Phi^{\!\top}\Phi/\sigma^2+I_d/\tau^2$ and $m=A^{-1}\Phi^{\!\top}y/\sigma^2$,
$$
\log p(y)= -\frac{n}{2}\log(2\pi\sigma^2)-\frac{d}{2}\log\tau^2-\frac12\log\det A
-\underbrace{\frac{\|y-\Phi m\|^2}{2\sigma^2}}_{\text{fit}}-\underbrace{\frac{\|m\|^2}{2\tau^2}}_{\text{prior fit}} .
$$

*Proof.* Marginally $y=\Phi\theta+\varepsilon$ with independent $\theta\sim N(0,\tau^2I)$ and $\varepsilon\sim N(0,\sigma^2I)$, so $y$ is a sum of independent Gaussian vectors, giving mean $0$ and covariance $C=\sigma^2I+\tau^2\Phi\Phi^{\!\top}$ (Ch. 5). For the second form, use Sylvester's determinant identity,
$$
\det C=\sigma^{2n}\det\Big(I_n+\frac{\tau^2}{\sigma^2}\Phi\Phi^{\!\top}\Big)=\sigma^{2n}\det\Big(I_d+\frac{\tau^2}{\sigma^2}\Phi^{\!\top}\Phi\Big)=\sigma^{2n}\tau^{2d}\det A,
$$
and, for the quadratic form, the standard "completing the square in the joint density then integrating" identity $y^{\!\top}C^{-1}y=\|y-\Phi m\|^2/\sigma^2+\|m\|^2/\tau^2$, which follows from Theorem 10.33 by writing $\log p(y)=\log p(y\mid\theta)+\log\pi(\theta)-\log\pi(\theta\mid y)$ and evaluating at $\theta=m$. (Numerically verified: for the data of Worked Example 10.18, $y^{\!\top}C^{-1}y=11.396396380522$ from both expressions, and $\log\det C=-19.935558456617$ from both.) $\blacksquare$

Read the four terms. $-\|y-\Phi m\|^2/2\sigma^2$ rewards fit. $-\|m\|^2/2\tau^2$ penalizes coefficients that strain the prior. $-\frac d2\log\tau^2-\frac12\log\det A$ is the Occam factor: it is the log-ratio of posterior volume to prior volume, and it punishes a prior so wide that the posterior occupies a tiny fraction of it. Maximizing over $\tau^2$ therefore balances flexibility against fit *using the training data only*.

**Worked Example 10.20 (evidence maximization; and evidence selects the right polynomial degree).** Same data as Worked Example 10.18, $\sigma=0.25$ known.

| $\tau^2$ | $\lambda=\sigma^2/\tau^2$ | $\log p(y)$ |
|---|---|---|
| 0.01 | 6.2500 | $-71.8178$ |
| 0.10 | 0.6250 | $-19.9507$ |
| 1.00 | 0.0625 | $-6.7577$ |
| 10.0 | 0.0063 | $-8.5078$ |
| 100. | 0.0006 | $-12.8001$ |

The maximum (bounded scalar optimization on $\log\tau^2$) is at $\tau^2=1.556526$, i.e. $\lambda^*=0.040154$, with $\log p(y)=-6.560551$. The resulting posterior mean is $(0.753958,\ 1.924454,\ -1.187983,\ 0.072747)$, and it again matches ridge at that exact $\lambda$ to machine precision. Note the shape of the curve: sharply bad for small $\tau^2$ (the prior fights the data) and mildly bad for large $\tau^2$ (the Occam factor bites). That asymmetry is generic, and it is why evidence maximization tends to err toward over-regularization rather than under.

Now hold $\tau^2$ free and vary the polynomial degree, i.e. compare models:

| degree | $d$ | $\log p(y)$ at $\tau^2=1$ | $\tau^2$ optimized | best $\log p(y)$ | OLS residual SS |
|---|---|---|---|---|---|
| 0 | 1 | $-117.4576$ | 0.3380 | $-117.2496$ | 15.0330 |
| 1 | 2 | $-12.7115$ | 1.5861 | $-12.5895$ | — |
| 2 | 3 | $-6.1462$ | 1.9949 | $\mathbf{-5.7298}$ | 0.3487 |
| 3 | 4 | $-6.7577$ | 1.5565 | $-6.5606$ | 0.3478 |
| 4 | 5 | $-6.8671$ | 1.3006 | $-6.7955$ | — |
| 5 | 6 | $-7.1338$ | 1.2582 | $-7.0784$ | — |
| 6 | 7 | $-7.2975$ | 1.1996 | $-7.2631$ | — |
| 7 | 8 | $-7.4362$ | 1.1822 | $-7.4070$ | 0.0957 |

The evidence peaks at degree 2, which is the true model order. Compare the last column: residual sum of squares decreases monotonically with degree (0.3487 at degree 2, 0.0957 at degree 7), so maximum likelihood would choose degree 7 and overfit. The evidence declines gently past degree 2 rather than falling off a cliff — with $n=12$ there is not enough data to rule out mild extra curvature — and that gentleness is itself informative: it says "degree 2 or 3, probably not 7."

In ML practice this machinery appears as *evidence maximization* or *type-II ML* for kernel and Gaussian-process hyperparameters, where the same formula with $\Phi\Phi^{\!\top}$ replaced by a kernel matrix $K$ is the standard objective. It is also the mechanism behind automatic relevance determination: give each feature its own $\tau_j^2$ and maximize the evidence; irrelevant features get $\tau_j^2\to0$ and are pruned.

### 10.6.4 The $\ell_1$/Laplace case: sparse MAP, dense posterior

Swap the Gaussian prior for a Laplace one and MAP estimation becomes the lasso. The posterior, however, behaves quite differently from what the lasso's output suggests.

**Proposition 10.37 (Laplace prior: soft-thresholded MAP, non-sparse posterior mean).** Consider the scalar case $y\mid\theta\sim N(\theta,s^2)$ with prior density $\pi(\theta)=\frac{1}{2b}e^{-|\theta|/b}$ on $\mathbb{R}$. Then:

(i) the MAP estimate is the soft-threshold $\hat\theta_{\text{MAP}}=\mathrm{sign}(y)\max\big(|y|-\tfrac{s^2}{b},\,0\big)$, which is exactly zero whenever $|y|\le s^2/b$;

(ii) the posterior is absolutely continuous with respect to $\lambda_1$, hence $\Pi(\theta=0\mid y)=0$;

(iii) for $y\ne0$, $\mathrm{sign}\big(\mathbb{E}[\theta\mid y]\big)=\mathrm{sign}(y)$ and $\mathbb{E}[\theta\mid y]\ne0$.

*Proof.* (i) Minimize $g(\theta)=\frac{(y-\theta)^2}{2s^2}+\frac{|\theta|}{b}$. On $\theta>0$, $g'=\frac{\theta-y}{s^2}+\frac1b$, vanishing at $\theta=y-s^2/b$, which is admissible iff $y>s^2/b$; symmetrically on $\theta<0$. Otherwise the minimum is at the kink $\theta=0$, where the subdifferential $[-1/b,1/b]$ contains $y/s^2$ iff $|y|\le s^2/b$. Since $g$ is strictly convex the stationary point is the global minimum.

(ii) The posterior density $\propto e^{-(y-\theta)^2/2s^2}e^{-|\theta|/b}$ is a bounded measurable function times $\lambda_1$; singletons are $\lambda_1$-null, so they are posterior-null. (Radon–Nikodym, Ch. 2.)

(iii) Let $f(\theta)$ denote the unnormalized posterior density and take $y>0$. For $\theta>0$,
$$
\frac{f(\theta)}{f(-\theta)}=\exp\Big(\frac{-(y-\theta)^2+(y+\theta)^2}{2s^2}\Big)=\exp\Big(\frac{2y\theta}{s^2}\Big)>1 .
$$
Hence, with $Z=\int f$,
$$
\mathbb{E}[\theta\mid y]=\frac1Z\int_0^\infty\theta\big(f(\theta)-f(-\theta)\big)\mathrm{d}\theta>0,
$$
because the integrand is strictly positive for $\theta>0$. The case $y<0$ is symmetric. $\blacksquare$

**Worked Example 10.21 (MAP says zero, the posterior does not).** Take $s=1$, $b=1$, so the threshold is $s^2/b=1$. Posterior summaries by numerical quadrature:

| $y$ | threshold | MAP | posterior mean | posterior median |
|---|---|---|---|---|
| 0.60 | 1.00 | 0.000000 | 0.291106 | 0.233330 |
| 1.50 | 1.00 | 0.500000 | 0.805627 | 0.732452 |
| 3.00 | 1.00 | 2.000000 | 2.025812 | 2.012500 |
| 0.60 ($b=0.25$) | 4.00 | 0.000000 | 0.059285 | 0.034515 |

At $y=0.6$ the lasso reports the coefficient is zero; the posterior mean is $0.291$ and the posterior probability that the coefficient is exactly zero is $0$. At $y=1.5$ the lasso reports $0.5$ and the posterior mean is $0.806$ — the MAP is biased toward zero by roughly the threshold, while the mean is not. The gap closes for large signals ($y=3$: MAP $2.000$, mean $2.026$).

So what is the lasso's zero *saying*? Not "this coefficient is zero" but "the posterior mode is at the kink", which happens whenever the data do not overcome the prior's pull by more than $s^2/b$. That is a statement about a mode of a density, and modes of densities in high dimensions are not where the mass is — a point worth remembering whenever you see a MAP estimate presented as a summary of a posterior.

> **Common Confusion 10.7.** *"The lasso set that coefficient to zero, so the model says the feature has no effect. And since ridge equals the Bayesian posterior mean anyway, Bayes adds nothing here."*
>
> Both halves need work. On sparsity: Proposition 10.37(ii) says the posterior assigns probability zero to the event $\{\theta=0\}$, so no amount of data will make the posterior *say* the coefficient is zero under a Laplace prior. Sparse MAP is a computational and interpretational convenience — and a genuinely good one for feature selection — but it is not a posterior statement, and confidence intervals or $p$-values read off a lasso fit without accounting for the selection are not valid. If you want a posterior probability of exclusion, you need a prior with an atom at zero (spike-and-slab), and then Chapter 2's absolute-continuity bookkeeping is doing real work: the posterior is a mixture of an atom and a continuous part.
>
> On "Bayes adds nothing": Corollary 10.34 says the *point estimates* coincide. Worked Example 10.19 shows what does not: at $x_*=1.5$ both methods predict $1.40$, but only one of them reports a predictive sd of $1.24$ rather than $0.25$. Ridge cannot distinguish "I predict 1.40 because I have seen data here" from "I predict 1.40 because my prior pulled me here." Nor does ridge give you Proposition 10.36's evidence, which selected the right polynomial degree in Worked Example 10.20 without a validation set. The point estimate is where the two frameworks agree; everything you would actually use uncertainty for is where they do not.

> **Self-check 10.6.**
> 1. In Corollary 10.34, what happens to $m$ as $\tau^2\to\infty$, and when is that limit well defined?
> 2. Why does $\varphi_*^{\!\top}\Sigma\varphi_*$ grow so fast at $x_*=1.5$ for a cubic basis? (Think about $\varphi_*$ itself.)
> 3. In Proposition 10.36, which term prevents the evidence from increasing without bound as $\tau^2\to\infty$?
> 4. Give the one-line reason a lasso zero is not the statement "$\theta=0$ with high posterior probability."

## 10.7 Approximate inference, previewed

### 10.7.1 The Laplace approximation

Outside conjugate families the posterior is an unnormalized function you can evaluate but not integrate. The cheapest useful approximation is to pretend it is Gaussian, with mean at the mode and covariance from the curvature there. If you already have a trained model, you already have the mode; the extra cost is one Hessian.

**Theorem 10.38 (Laplace approximation to a posterior).** Let $q(\theta)=p(D\mid\theta)\pi(\theta)$ satisfy the hypotheses of Theorem 10.30, with mode $\hat\theta$ interior and $H=-\nabla^2\log q(\hat\theta)\succ0$. Define $\tilde\Pi:=N(\hat\theta,H^{-1})$. Then

(i) $\log\pi(\theta\mid D)=\log\tilde\pi(\theta)+\big(\log Z_{\text{Lap}}-\log Z\big)+R(\theta)$, where $R(\theta)=O(\|\theta-\hat\theta\|^3)$ and $Z_{\text{Lap}}=q(\hat\theta)(2\pi)^{d/2}(\det H)^{-1/2}$;

(ii) in the i.i.d. setting with $n$ observations, on the region $\|\theta-\hat\theta\|\le n^{-1/2}\log n$ the log-density error is $O(n^{-1/2}\log^3 n)$, and $\log Z=\log Z_{\text{Lap}}+O(n^{-1})$;

(iii) $\tilde\Pi$ and the Bernstein–von Mises limit (Theorem 10.9) agree to leading order, since $H=nI(\theta_0)(1+o_P(1))$.

*Proof.* (i) is Taylor's theorem applied to $\log q$ at its mode, exactly as in the proof of Theorem 10.30; the constant is the ratio of normalizers. (ii) follows by substituting $\theta=\hat\theta+u/\sqrt n$ and bounding the cubic term, with the evidence statement quoted from Theorem 10.30. (iii) is the identification of the Hessian of the negative log-likelihood at the MLE with $n$ times observed Fisher information, which converges to $nI(\theta_0)$ by the law of large numbers. $\blacksquare$

The failure modes are as important as the theorem, and all of them are visible in the statement:

1. **Skewness is invisible to a Gaussian.** The approximation is symmetric by construction, so it misplaces mass whenever the posterior is skewed, which is the norm for scale parameters, rates near zero, and logistic coefficients with modest data.
2. **The mode may be a bad summary.** In high dimensions the mode of a density can be far from where the mass is (the Gaussian annulus phenomenon), and for multimodal posteriors — mixtures, neural networks with permutation symmetries — Laplace captures one mode and pretends the others do not exist.
3. **$H$ may not be positive definite,** or may be enormous: for a network with $d=10^8$ the Hessian is out of reach, and the standard fixes (diagonal, Kronecker-factored, or last-layer-only) are approximations *of the approximation*.
4. **Constraints and boundaries break it.** If $\hat\theta$ sits on the boundary of $\Theta$ (a variance estimated as zero, a probability estimated as one), the expansion is invalid. Reparameterizing to an unconstrained space usually fixes this and often improves accuracy dramatically.

**Worked Example 10.22 (Laplace versus exact for a logistic coefficient).** One-dimensional logistic regression, $P(y=1\mid x,\theta)=\sigma(\theta x)$, prior $\theta\sim N(0,2^2)$. Data: $x=(-2,-1,-0.5,0.5,1,2)$, $y=(0,0,1,1,1,1)$. Everything below is computed by bounded optimization and adaptive quadrature on $[-30,30]$.

| quantity | Laplace | exact |
|---|---|---|
| centre | $\hat\theta_{\text{MAP}}=1.500369$ | mean $1.858382$, median $1.745797$ |
| sd | $H^{-1/2}=0.990957$ ($H=1.018335$) | $1.057422$ |
| 2.5% quantile | $-0.441870$ | $0.110258$ |
| 97.5% quantile | $3.442608$ | $4.223451$ |
| $\log$ evidence | $-3.007225$ | $-2.989766$ (error $-0.017$) |

The sd is nearly right and the interval is badly wrong: the true posterior is right-skewed (mean $1.86$ exceeds mode $1.50$), so the symmetric approximation puts $2.5\%$ of its mass below $-0.44$ when the exact posterior puts $2.5\%$ below $+0.11$. A Laplace-based analysis would conclude that a negative coefficient is quite possible; the exact posterior says it is not.

The predictive is a second lesson. At $x_*=0.75$:

| method | $P(y_*=1\mid x_*)$ |
|---|---|
| exact posterior average | $0.774730$ |
| plug-in at MAP, $\sigma(\hat\theta x_*)$ | $0.754966$ |
| Laplace, averaged (Monte Carlo, $2\times10^6$ draws, seed 0) | $0.732759$ |
| Laplace, probit approximation $\sigma\!\big(\hat\theta x_*/\sqrt{1+\tfrac\pi8 s_*^2}\big)$ | $0.734986$ |

Note what happened: averaging over the Laplace posterior moved the prediction *away* from the exact answer. Marginalization pulls binary predictions toward $0.5$, which is the right direction when the mode is the posterior's centre; here the mode is below the centre, so the correction compounds an error rather than fixing it. "Bayesian averaging always improves calibration" is therefore false as stated — it improves calibration when the approximate posterior is roughly right.

Finally, the boundary failure mode, concretely. Replace the data with the separable set $x=(-2,-1,1,2)$, $y=(0,0,1,1)$, where the MLE does not exist ($\theta\to\infty$). The prior is now the only thing keeping the MAP finite:

| prior sd | 2 | 10 | 100 |
|---|---|---|---|
| MAP | 1.7174 | 3.9453 | 7.8441 |

The MAP diverges logarithmically as the prior widens, the curvature at the mode goes to zero, and the Laplace sd blows up while the posterior itself remains heavy-tailed but perfectly proper. Any pipeline that reports "MAP $\pm$ Laplace sd" on nearly separable data is reporting an artifact of the prior. Nearly separable data is the normal condition for a well-trained deep classifier on its training set.

**Worked Example 10.23 (choose your coordinates before you approximate).** Take the exact posterior $\mathrm{Beta}(15,9)$ from Worked Example 10.1, whose mode is $0.636364$ and true sd $0.096825$. Laplace on the $\theta$ scale: $H=(a-1)/\hat\theta^2+(b-1)/(1-\hat\theta)^2=95.071429$, so $\text{sd}=0.102559$ ($6\%$ too large) and the 95% interval is $[0.4354,\ 0.8374]$ against the exact $[0.4273,\ 0.8029]$ — the upper end overshoots by $0.035$, and for $a$ or $b$ near 1 the interval can even leave $[0,1]$. Redo it on the logit scale $\phi=\log\frac{\theta}{1-\theta}$, remembering the Jacobian (Proposition 10.18): the mode is $\phi=0.510826$ with curvature giving sd $0.421637$, and transforming the interval endpoints back gives $[0.4218,\ 0.7920]$ — closer to exact at both ends, and guaranteed to stay inside $(0,1)$. The general rule: apply Laplace in the coordinates where the posterior is most nearly Gaussian, which usually means unconstrained ones (log for scales, logit for probabilities, Cholesky factors for covariances).

### 10.7.2 What comes next

Two families of methods dominate when Laplace is not enough, and Chapter 13 develops both.

*Markov chain Monte Carlo* constructs a Markov chain whose stationary distribution is the posterior, and estimates posterior expectations by ergodic averages. Chapter 13 covers Metropolis–Hastings, Gibbs sampling, Hamiltonian Monte Carlo and NUTS, the geometry that makes HMC efficient in high dimensions, and the diagnostics ($\hat R$, effective sample size, divergences) that tell you when to distrust the output. MCMC is asymptotically exact and expensive.

*Variational inference* posits a tractable family $\mathcal{Q}$ and picks $q\in\mathcal{Q}$ minimizing $D_{\mathrm{KL}}(q\,\|\,\Pi(\cdot\mid D))$, equivalently maximizing the ELBO that Chapter 2 introduced. Chapter 13 covers mean-field and structured families, stochastic and black-box variational inference, the reparameterization gradient (which needs Chapter 3's change of variables), amortized inference and the VAE, and the systematic biases of the reverse-KL objective — most notably variance under-estimation. VI is cheap, scalable, and biased.

> **Self-check 10.7.**
> 1. Theorem 10.38 needs $H\succ0$. Give two situations in a deep network where it fails.
> 2. In Worked Example 10.22, the Laplace sd was within 7% of exact but the credible interval was badly wrong. What property of the exact posterior does sd fail to capture?
> 3. Why does reparameterizing to the logit scale improve the Laplace approximation of a Beta posterior?
> 4. You have a trained network and one hour of compute. What does Laplace buy you that a validation set does not?

## 10.8 Bayesian deep learning

### 10.8.1 The honest situation

A neural network's posterior is a distribution over $10^7$ to $10^{11}$ parameters with an enormous number of symmetry-induced modes (permute hidden units within a layer; rescale weights across a ReLU boundary), and it is not remotely Gaussian. No one computes it. What people compute are objects that (a) are cheap, and (b) produce predictive distributions with useful properties. It is worth being explicit that this changes the goal: the practical target is **calibrated, useful predictive uncertainty**, not fidelity to the posterior. A method can be a poor approximation to the posterior and an excellent uncertainty estimator, and vice versa.

A striking empirical fact makes the point sharper, and you will meet it in papers under the name *cold posteriors*. When approximate posteriors over network weights are tempered — the posterior density raised to a power $1/T$ — practitioners repeatedly find that $T<1$, i.e. a posterior *sharper* than Bayes' rule prescribes, gives better test accuracy and log-loss than $T=1$ (Wenzel et al., 2020). If the Bayesian posterior were the right target, $T=1$ should be best; that it often is not is direct evidence that something in the pipeline is misspecified — candidate culprits include priors chosen for tractability rather than belief (§10.3.5), likelihoods that ignore data augmentation and curation, and the approximation itself — and it connects back to §10.1.4's warning that a posterior under misspecification is faithfully quantifying uncertainty about the wrong object. The debate over which culprit dominates is unresolved; what you should take from it is operational: in deep learning, "more Bayesian" is not automatically "better calibrated", and every method below must be judged by predictive metrics, not by fidelity to a posterior nobody can check.

**Definition 10.39 (calibration and ECE).** Let $\hat p(\cdot\mid x)$ be a probabilistic classifier, $\hat y(x):=\arg\max_k\hat p(k\mid x)$ its prediction, and $C:=\max_k\hat p(k\mid X)$ its confidence, a random variable in $[0,1]$. The classifier is *calibrated* if
$$
\mathbb{E}\big[\mathbf{1}\{Y=\hat y(X)\}\ \big|\ C\big]=C\qquad\text{almost surely.}
$$
The conditional expectation is the Chapter 6 object, so this is a statement about a version of $\mathbb{E}[\cdot\mid\sigma(C)]$ and is therefore only defined up to null sets — which is exactly why one cannot write the naive "$P(Y=\hat y(X)\mid C=c)=c$ for every $c$": when $C$ has a continuous distribution, $\{C=c\}$ is null for every $c$ and that expression conditions on nothing. Equivalently, calibration says $\mathbb{E}[(\mathbf{1}\{Y=\hat y(X)\}-C)h(C)]=0$ for every bounded measurable $h$, which is the form an estimator actually checks. Taking $h$ to be indicators of bins $B_1,\dots,B_M$ partitioning $[0,1]$ gives the *expected calibration error*,
$$
\mathrm{ECE}=\sum_{m=1}^M\frac{|B_m|}{n}\big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\big| ,
$$
a consistent estimator of a binned version of the discrepancy as $M\to\infty$ slowly with $n$ — and note that ECE at fixed $M$ can be zero for an uncalibrated classifier whose errors cancel within bins, which is why the bin count must be reported alongside it. Calibration is a property of the marginal predictive distribution only; it says nothing about sharpness (a constant predictor of the base rate is perfectly calibrated and useless), which is why calibration is always reported alongside a proper scoring rule such as log-loss or Brier score. Chapter 13 treats calibration, temperature scaling, and proper scoring rules properly.

### 10.8.2 Methods, with what is derivable and what is not

**Laplace for networks.** Apply Theorem 10.38 at a trained weight vector, with $H$ approximated by the generalized Gauss–Newton or Fisher matrix and factorized (diagonal, or Kronecker-factored per layer). The *last-layer* variant treats only the final linear layer as random, which makes the model exactly the Bayesian linear regression of §10.6 with $\varphi$ = the learned penultimate features: Theorem 10.33, Proposition 10.35 and Proposition 10.36 apply verbatim, including evidence-based tuning of the prior precision. This is the cheapest principled option and, empirically, a strong baseline. Its weakness is the one Worked Example 10.22 exhibited: at a well-fit mode on nearly separable data, curvature-based uncertainty is small and dominated by the prior.

**MC dropout.** Keep dropout on at test time and average $T$ stochastic forward passes. The argument for calling this Bayesian: dropout training can be written as variational inference with a variational family $q$ that is a mixture of two Gaussians per weight (a spike near zero and a spike near the learned value), and the dropout objective is then the ELBO up to constants. The criticisms are substantive and you should know them. (i) The variational family's "posterior" does not concentrate as $n\to\infty$ — the dropout rate is fixed, so the implied uncertainty does not shrink with more data, which no posterior does. (ii) The KL term in the claimed ELBO is improper or requires a singular prior, so the correspondence is formal rather than exact. (iii) Empirically the predictive variance is controlled mostly by the dropout rate, a quantity chosen for regularization, so the "uncertainty" is a tuning knob. Use MC dropout as a cheap ensemble-flavoured smoother; do not claim it estimates a posterior.

**Deep ensembles.** Train $K$ networks from different random initializations and average their predictive distributions. This is *not* Bayesian — there is no prior, and the ensemble members are modes found by an optimizer, not posterior samples — and it is nevertheless the strongest general-purpose baseline for calibration and out-of-distribution robustness in most published comparisons. There is a partial Bayesian reading (the members approximate distinct posterior modes, so the ensemble is a crude multimodal approximation that Laplace and mean-field VI structurally cannot produce), and it explains the empirical pattern: ensembles win where multimodality matters, and $K=5$ typically captures most of the gain. The cost is $K\times$ training and $K\times$ inference.

**SWAG** fits a Gaussian to the trajectory of SGD iterates in the tail of training (mean = the SWA average, covariance = a low-rank plus diagonal estimate from iterate deviations), then samples from it. The motivation is that SGD with constant learning rate explores a region whose scale is related to posterior width; this is exactly true for a quadratic loss with isotropic gradient noise and approximately true otherwise. It costs one training run plus storage.

**Where this leaves you.** For a practitioner: use a proper scoring rule on held-out data to compare methods, use deep ensembles when you can afford them, last-layer Laplace or SWAG when you cannot, and treat any claim of "the posterior" over network weights with suspicion. For a reader of papers: check whether reported uncertainty improvements survive temperature scaling on a validation set, because a large fraction of "Bayesian deep learning improves calibration" results are recoverable by that one-parameter fix.

> **Common Confusion 10.8.** *"I turned dropout on at test time and averaged 50 forward passes, so those are posterior samples and my error bars are Bayesian."*
>
> They are samples from a variational family whose spread is set by the dropout rate you chose for regularization, not by the data. Here is the diagnostic that settles it: increase $n$ by a factor of 100 and retrain. A genuine posterior's width shrinks like $n^{-1/2}$ (Theorem 10.9); MC dropout's predictive spread on a fixed input does not systematically shrink at all, because the dropout rate did not change. Similarly, the epistemic/aleatoric decomposition of Proposition 10.7 computed from dropout samples inherits this problem: the "epistemic" part is a function of your regularization hyperparameter. None of this makes MC dropout useless — it is cheap, it often improves log-loss over a single deterministic pass, and it is a reasonable smoother — but the words "posterior sample" should not be used for it, and reviewers increasingly ask.

> **Self-check 10.8.**
> 1. Why does last-layer Laplace let you reuse Theorem 10.33 exactly?
> 2. Give the one-sentence argument that MC dropout's uncertainty does not behave like a posterior's.
> 3. A model has ECE $0.01$ and accuracy $0.62$; another has ECE $0.08$ and accuracy $0.91$. Which is better calibrated, and which would you deploy?
> 4. Why do deep ensembles capture something that mean-field VI and Laplace structurally cannot?

## 10.9 Bayesian decision theory

### 10.9.1 The posterior is not the answer; it is an input

Nobody deploys a posterior. You deploy an action: a number to report, a dose to give, a threshold to alert at, an item to recommend. Decision theory is the layer that turns a posterior into an action, and it makes precise something that is usually left implicit: **the question "what is my estimate of $\theta$" has no answer until you say what an error costs.**

**Definition 10.40 (loss, posterior risk, Bayes action).** Let $\mathcal{A}$ be a set of available actions and $L:\Theta\times\mathcal{A}\to[0,\infty)$ a measurable *loss*. Given data $D$, the *posterior expected loss* (or posterior risk) of action $a$ is
$$
\rho(a\mid D):=\mathbb{E}\big[L(\theta,a)\ \big|\ D\big]=\int_\Theta L(\theta,a)\,\Pi(\mathrm{d}\theta\mid D),
$$
and a *Bayes action* is any $a^\star\in\arg\min_{a\in\mathcal{A}}\rho(a\mid D)$. The *Bayes risk* is $\rho(a^\star\mid D)$. Averaging the loss of a decision rule $\delta:\mathcal{X}\to\mathcal{A}$ over both $\theta$ and data gives the *integrated risk* $r(\delta)=\mathbb{E}[L(\theta,\delta(X))]$, and a rule that chooses a Bayes action for every $x$ minimizes $r$ — the reason Bayes rules are the natural inhabitants of admissibility theory in Chapter 8.

**Theorem 10.41 (your loss selects your summary).** Let $\theta$ be real-valued with posterior $\Pi(\cdot\mid D)$ and $\mathcal{A}=\mathbb{R}$.

(i) *(Squared loss $\to$ mean.)* If $L(\theta,a)=(\theta-a)^2$ and $\mathbb{E}[\theta^2\mid D]<\infty$, the unique Bayes action is $a^\star=\mathbb{E}[\theta\mid D]$.

(ii) *(Absolute loss $\to$ median.)* If $L(\theta,a)=|\theta-a|$ and $\mathbb{E}|\theta|<\infty$, the Bayes actions are exactly the posterior medians, i.e. all $a$ with $\Pi(\theta\le a\mid D)\ge\frac12$ and $\Pi(\theta\ge a\mid D)\ge\frac12$.

(iii) *(0-1 loss $\to$ mode.)* If $\Theta$ is countable and $L(\theta,a)=\mathbf{1}\{a\ne\theta\}$, the Bayes actions are the posterior modes. If $\Theta=\mathbb{R}$ with continuous posterior density $\pi(\cdot\mid D)$ and $L_\varepsilon(\theta,a)=\mathbf{1}\{|\theta-a|>\varepsilon\}$, then any sequence of $L_\varepsilon$-Bayes actions accumulates only at maximizers of $\pi(\cdot\mid D)$ as $\varepsilon\downarrow0$.

*Proof.* (i) Expand: $\rho(a)=\mathbb{E}[\theta^2\mid D]-2a\mathbb{E}[\theta\mid D]+a^2$, a strictly convex parabola in $a$ minimized at $a=\mathbb{E}[\theta\mid D]$. (This is the $L^2$-projection characterization of conditional expectation from Chapter 6, specialized to constants.)

(ii) $\rho(a)=\mathbb{E}|\theta-a|$ is finite and convex in $a$ (a supremum-free average of convex functions; convexity passes through integration). For $a<a'$,
$$
\rho(a')-\rho(a)=\int\big(|\theta-a'|-|\theta-a|\big)\Pi(\mathrm{d}\theta\mid D),
$$
and the integrand equals $(a'-a)$ for $\theta\le a$, $-(a'-a)$ for $\theta\ge a'$, and lies between for $\theta\in(a,a')$. Letting $a'\downarrow a$ gives the right derivative $\rho'_+(a)=\Pi(\theta\le a\mid D)-\Pi(\theta>a\mid D)$, and similarly $\rho'_-(a)=\Pi(\theta<a\mid D)-\Pi(\theta\ge a\mid D)$. A convex function is minimized exactly where $\rho'_-(a)\le0\le\rho'_+(a)$, which rearranges to the two median inequalities.

(iii) In the countable case $\rho(a)=1-\Pi(\theta=a\mid D)$, minimized by maximizing the posterior mass at $a$. In the continuous case $\rho_\varepsilon(a)=1-\Pi(|\theta-a|\le\varepsilon\mid D)$, so minimizing $\rho_\varepsilon$ maximizes $\int_{a-\varepsilon}^{a+\varepsilon}\pi(t\mid D)\mathrm{d}t=2\varepsilon\pi(a\mid D)+o(\varepsilon)$ by continuity of $\pi$; hence any limit point of minimizers maximizes $\pi(\cdot\mid D)$. $\blacksquare$

Part (iii) is the honest status of MAP estimation: it is the Bayes action for a loss that treats every wrong answer as equally bad, no matter how nearly right — a loss almost nobody actually has. That is one reason to be suspicious of MAP as a summary, alongside Proposition 10.37's sparsity subtlety and the high-dimensional mode-versus-mass problem of §10.7.

The three summaries genuinely differ whenever the posterior is skewed. Take the $\mathrm{Gamma}(9,6)$ posterior of Worked Example 10.8: mean $1.500000$, median $1.444825$, mode $1.333333$. Reporting "the rate is $1.33$" versus "$1.50$" is a $12.5\%$ difference produced entirely by the choice of loss, not by the data.

### 10.9.2 Asymmetric losses give quantiles

Symmetric losses are the exception in applications. Under-ordering inventory costs a lost sale; over-ordering costs storage. Missing a fraudulent transaction costs the fraud amount; flagging a legitimate one costs customer annoyance. Under-predicting a delivery time annoys a customer; over-predicting loses the sale.

**Proposition 10.42 (asymmetric linear loss $\to$ posterior quantile).** Let $c_u,c_o>0$ and
$$
L(\theta,a)=c_u(\theta-a)_++c_o(a-\theta)_+ ,
$$
so under-shooting costs $c_u$ per unit and over-shooting $c_o$ per unit. If $\mathbb{E}|\theta|<\infty$ then the Bayes actions are exactly the posterior quantiles at level
$$
q^\star=\frac{c_u}{c_u+c_o},
$$
i.e. all $a$ with $\Pi(\theta\le a\mid D)\ge q^\star$ and $\Pi(\theta\ge a\mid D)\ge1-q^\star$.

*Proof.* As in Theorem 10.41(ii), $\rho$ is convex with right derivative
$$
\rho'_+(a)=c_o\,\Pi(\theta\le a\mid D)-c_u\,\Pi(\theta>a\mid D)=(c_o+c_u)\Pi(\theta\le a\mid D)-c_u,
$$
and analogously for $\rho'_-$. Setting $\rho'_-(a)\le0\le\rho'_+(a)$ gives $\Pi(\theta<a\mid D)\le q^\star\le\Pi(\theta\le a\mid D)$, the quantile condition. Taking $c_u=c_o$ recovers the median. $\blacksquare$

This is precisely the quantile-regression check loss, and it explains why quantile regression is the right tool for asymmetric costs: you are not "modelling the quantile" as an end in itself, you are computing a Bayes action.

**Worked Example 10.24 (an inventory decision, with the predictive doing real work).** Daily demand $\sim\mathrm{Poisson}(\lambda)$, prior $\lambda\sim\mathrm{Gamma}(2,1)$, and five days of data totalling $100$ units. Posterior $\lambda\mid D\sim\mathrm{Gamma}(102,6)$: mean $17.000$, sd $1.683$ (the MLE is $20.0$; the prior's one unit of pseudo-exposure and shape 2 pull it down noticeably because $\beta_0=1$ against $n=5$ days is not negligible). The predictive for tomorrow is Negative Binomial with $r=102$, $p=6/7$: mean $17.000$, sd $4.4535$, versus $\mathrm{Poisson}(17)$'s sd $4.1231$.

Stocking cost: $c_u=4$ per unit short, $c_o=1$ per unit unsold. Proposition 10.42 says order the $q^\star=4/5$ quantile.

| level $q$ | 0.50 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| predictive (NB) quantile | 17 | **21** | 23 | 25 | 28 |
| plug-in Poisson(17) quantile | 17 | **20** | 22 | 24 | 27 |

Expected loss under the correct predictive, evaluated exactly by summing over $k=0,\dots,79$:

| order $a$ | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|
| $\rho(a\mid D)$ | 6.9019 | 6.5237 | **6.4736** | 6.6960 | 7.1359 |

The Bayes action is $a^\star=21$, matching the NB $0.8$-quantile. Using the plug-in Poisson quantile ($a=20$) costs $6.5237-6.4736=0.0501$ per day, about $0.8\%$ of the daily loss — small here, and worth noticing that it grows with the cost ratio and with parameter uncertainty. The predictive tail is where the difference lives: $P(\text{demand}>25)$ is $0.0356$ under the predictive and $0.0252$ under the plug-in: the plug-in misses $29\%$ of that tail mass.

### 10.9.3 Binary decisions: the threshold is a cost ratio, not a probability

**Corollary 10.43 (optimal decision threshold).** Let $\theta\in\{0,1\}$ (disease present or not), $\mathcal{A}=\{\text{act},\text{don't act}\}$, and losses $L(1,\text{don't act})=c_{FN}$, $L(0,\text{act})=c_{FP}$, $L(1,\text{act})=L(0,\text{don't act})=0$. Write $p:=\Pi(\theta=1\mid D)$. Then
$$
\rho(\text{act})=(1-p)c_{FP},\qquad \rho(\text{don't act})=p\,c_{FN},
$$
so acting is optimal iff
$$
p\ >\ \frac{c_{FP}}{c_{FP}+c_{FN}}\qquad\Longleftrightarrow\qquad \frac{p}{1-p}>\frac{c_{FP}}{c_{FN}}.
$$

*Proof.* Direct evaluation of Definition 10.40's two expectations, then compare. $\blacksquare$

Three consequences worth stating out loud. The famous threshold $0.5$ is the Bayes threshold *only* when $c_{FP}=c_{FN}$; there is nothing natural about it otherwise. The threshold depends on the loss alone and the posterior on the data alone, so the two are cleanly separable — you can retune a deployed threshold when costs change without retraining. And a classifier trained with a symmetric loss and then thresholded at $0.5$ is making an implicit claim of equal costs that is almost always false in the applications where classifiers matter.

Figure 10.6 draws the two lines and their crossing.

<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto" font-family="sans-serif">
  <line x1="80.00" y1="300.00" x2="700.00" y2="300.00" stroke="currentColor" stroke-width="1.2"/>
  <line x1="80.00" y1="300.00" x2="80.00" y2="50.00" stroke="currentColor" stroke-width="1.2"/>
  <polyline points="80.0,287.5 92.7,287.8 105.3,288.0 118.0,288.3 130.6,288.5 143.3,288.8 155.9,289.0 168.6,289.3 181.2,289.5 193.9,289.8 206.5,290.1 219.2,290.3 231.8,290.6 244.5,290.8 257.1,291.1 269.8,291.3 282.4,291.6 295.1,291.8 307.8,292.1 320.4,292.3 333.1,292.6 345.7,292.9 358.4,293.1 371.0,293.4 383.7,293.6 396.3,293.9 409.0,294.1 421.6,294.4 434.3,294.6 446.9,294.9 459.6,295.2 472.2,295.4 484.9,295.7 497.6,295.9 510.2,296.2 522.9,296.4 535.5,296.7 548.2,296.9 560.8,297.2 573.5,297.4 586.1,297.7 598.8,298.0 611.4,298.2 624.1,298.5 636.7,298.7 649.4,299.0 662.0,299.2 674.7,299.5 687.3,299.7 700.0,300.0" fill="none" stroke="#3b82f6" stroke-width="2.2"/>
  <polyline points="80.0,300.0 83.1,298.7 86.2,297.5 89.3,296.2 92.5,295.0 95.6,293.7 98.7,292.5 101.8,291.2 104.9,289.9 108.0,288.7 111.2,287.4 114.3,286.2 117.4,284.9 120.5,283.7 123.6,282.4 126.7,281.2 129.8,279.9 133.0,278.6 136.1,277.4 139.2,276.1 142.3,274.9 145.4,273.6 148.5,272.4 151.7,271.1 154.8,269.8 157.9,268.6 161.0,267.3 164.1,266.1 167.2,264.8 170.4,263.6 173.5,262.3 176.6,261.1 179.7,259.8 182.8,258.5 185.9,257.3 189.0,256.0 192.2,254.8 195.3,253.5 198.4,252.3 201.5,251.0 204.6,249.7 207.7,248.5 210.9,247.2 214.0,246.0 217.1,244.7 220.2,243.5 223.3,242.2 226.4,241.0 229.5,239.7 232.7,238.4 235.8,237.2 238.9,235.9 242.0,234.7 245.1,233.4 248.2,232.2 251.4,230.9 254.5,229.6 257.6,228.4 260.7,227.1 263.8,225.9 266.9,224.6 270.1,223.4 273.2,222.1 276.3,220.9 279.4,219.6 282.5,218.3 285.6,217.1 288.7,215.8 291.9,214.6 295.0,213.3 298.1,212.1 301.2,210.8 304.3,209.5 307.4,208.3 310.6,207.0 313.7,205.8 316.8,204.5 319.9,203.3 323.0,202.0 326.1,200.8 329.2,199.5 332.4,198.2 335.5,197.0 338.6,195.7 341.7,194.5 344.8,193.2 347.9,192.0 351.1,190.7 354.2,189.4 357.3,188.2 360.4,186.9 363.5,185.7 366.6,184.4 369.7,183.2 372.9,181.9 376.0,180.7 379.1,179.4 382.2,178.1 385.3,176.9 388.4,175.6 391.6,174.4 394.7,173.1 397.8,171.9 400.9,170.6 404.0,169.3 407.1,168.1 410.3,166.8 413.4,165.6 416.5,164.3 419.6,163.1 422.7,161.8 425.8,160.6 428.9,159.3 432.1,158.0 435.2,156.8 438.3,155.5 441.4,154.3 444.5,153.0 447.6,151.8 450.8,150.5 453.9,149.2 457.0,148.0 460.1,146.7 463.2,145.5 466.3,144.2 469.4,143.0 472.6,141.7 475.7,140.5 478.8,139.2 481.9,137.9 485.0,136.7 488.1,135.4 491.3,134.2 494.4,132.9 497.5,131.7 500.6,130.4 503.7,129.1 506.8,127.9 509.9,126.6 513.1,125.4 516.2,124.1 519.3,122.9 522.4,121.6 525.5,120.4 528.6,119.1 531.8,117.8 534.9,116.6 538.0,115.3 541.1,114.1 544.2,112.8 547.3,111.6 550.5,110.3 553.6,109.0 556.7,107.8 559.8,106.5 562.9,105.3 566.0,104.0 569.1,102.8 572.3,101.5 575.4,100.3 578.5,99.0 581.6,97.7 584.7,96.5 587.8,95.2 591.0,94.0 594.1,92.7 597.2,91.5 600.3,90.2 603.4,88.9 606.5,87.7 609.6,86.4 612.8,85.2 615.9,83.9 619.0,82.7 622.1,81.4 625.2,80.2 628.3,78.9 631.5,77.6 634.6,76.4 637.7,75.1 640.8,73.9 643.9,72.6 647.0,71.4 650.2,70.1 653.3,68.8 656.4,67.6 659.5,66.3 662.6,65.1 665.7,63.8 668.8,62.6 672.0,61.3 675.1,60.1 678.2,58.8 681.3,57.5 684.4,56.3 687.5,55.0 690.7,53.8 693.8,52.5 696.9,51.3 700.0,50.0" fill="none" stroke="#ef4444" stroke-width="2.2"/>
  <line x1="109.52" y1="300.00" x2="109.52" y2="70.00" stroke="#10b981" stroke-width="1.6" stroke-dasharray="5,4"/>
  <text x="115.52" y="84.00" font-size="11" fill="#10b981" text-anchor="start" font-weight="normal">threshold c_FP/(c_FP+c_FN) = 0.0476</text>
  <line x1="136.11" y1="300.00" x2="136.11" y2="120.00" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="5,4"/>
  <text x="142.11" y="134.00" font-size="11" fill="#f59e0b" text-anchor="start" font-weight="normal">P(D | +) = 0.0905  ->  treat</text>
  <line x1="80.00" y1="300.00" x2="80.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="80.00" y="319.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.0</text>
  <line x1="204.00" y1="300.00" x2="204.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="204.00" y="319.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.2</text>
  <line x1="328.00" y1="300.00" x2="328.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="328.00" y="319.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.4</text>
  <line x1="452.00" y1="300.00" x2="452.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="452.00" y="319.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.6</text>
  <line x1="576.00" y1="300.00" x2="576.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="576.00" y="319.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">0.8</text>
  <line x1="700.00" y1="300.00" x2="700.00" y2="305.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="700.00" y="319.00" font-size="11" fill="currentColor" text-anchor="middle" font-weight="normal">1.0</text>
  <line x1="75.00" y1="300.00" x2="80.00" y2="300.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="71.00" y="304.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">0</text>
  <line x1="75.00" y1="237.50" x2="80.00" y2="237.50" stroke="currentColor" stroke-width="1.2"/>
  <text x="71.00" y="241.50" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">5</text>
  <line x1="75.00" y1="175.00" x2="80.00" y2="175.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="71.00" y="179.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">10</text>
  <line x1="75.00" y1="112.50" x2="80.00" y2="112.50" stroke="currentColor" stroke-width="1.2"/>
  <text x="71.00" y="116.50" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">15</text>
  <line x1="75.00" y1="50.00" x2="80.00" y2="50.00" stroke="currentColor" stroke-width="1.2"/>
  <text x="71.00" y="54.00" font-size="11" fill="currentColor" text-anchor="end" font-weight="normal">20</text>
  <text x="390.00" y="342.00" font-size="13" fill="currentColor" text-anchor="middle" font-weight="normal">posterior probability of disease, P(D | data)</text>
  <text x="28.00" y="34.00" font-size="12" fill="currentColor" text-anchor="start" font-weight="normal">posterior expected loss</text>
  <text x="421.00" y="284.38" font-size="12" fill="#3b82f6" text-anchor="start" font-weight="normal">do not treat: (1-p) x c_FP</text>
  <text x="266.00" y="258.75" font-size="12" fill="#ef4444" text-anchor="start" font-weight="normal">treat: p x c_FN</text>
  <text x="380.0" y="380" font-size="13" fill="currentColor" text-anchor="middle">Figure 10.6  The Bayes decision rule for the Chapter 1 screening example (prevalence 0.5%,</text>
  <text x="380.0" y="397" font-size="13" fill="currentColor" text-anchor="middle">sensitivity 99%, specificity 95%, so P(D | +) = 0.0905). Expected loss is linear in the posterior;</text>
  <text x="380.0" y="414" font-size="13" fill="currentColor" text-anchor="middle">the two lines cross at c_FP/(c_FP + c_FN). With c_FN = 20 the threshold is 0.0476 and we treat;</text>
  <text x="380.0" y="431" font-size="13" fill="currentColor" text-anchor="middle">with c_FN = 10 the threshold is 0.0909 and we do not. The posterior alone decides nothing.</text>
</svg>


**Worked Example 10.25 (the Chapter 1 screening test, now as a decision).** Recall the running example: prevalence $0.5\%$, sensitivity $99\%$, specificity $95\%$. Bayes' rule (Proposition 1.12) gives
$$
P(D\mid+)=\frac{0.005\times0.99}{0.005\times0.99+0.995\times0.05}=\frac{0.00495}{0.05470}=0.090494,
$$
the $9.05\%$ computed in Chapter 1. Chapter 1 stopped there, with the observation that base rates dominate. Now finish the job: is $9.05\%$ enough to act on?

| $c_{FN}:c_{FP}$ | threshold $c_{FP}/(c_{FP}+c_{FN})$ | decision at $p=0.0905$ |
|---|---|---|
| $1:1$ | 0.500000 | do not treat |
| $10:1$ | 0.090909 | do not treat (just barely) |
| $20:1$ | 0.047619 | treat |
| $100:1$ | 0.009901 | treat |

The break-even cost ratio is $(1-p)/p=10.0505$. At $c_{FN}=20$, $c_{FP}=1$: $\rho(\text{treat})=(1-p)\cdot1=0.909506$ and $\rho(\text{don't})=p\cdot20=1.809872$, so treating is better by a factor of $2.0$. At $c_{FN}=10$ the two are within $1\%$ of each other and the decision is genuinely close — which is useful information in itself, because it tells you that refining the estimate of the cost ratio would change the decision, while refining the posterior by a percentage point would not.

The sensitivity to prevalence is worth one more line. Holding $c_{FN}=10$, the decision flips from "don't treat" to "treat" when prevalence exceeds $0.005025$ — a change of half a percent *relative*. A screening policy tuned in a population with $0.5\%$ prevalence is therefore on a knife-edge and will silently change its recommendation when applied to a slightly different population. This is the decision-theoretic form of the base-rate warning from Chapter 1, and it is why deployed thresholds must be revalidated per population rather than shipped as constants.

Two ML-specific notes to close.

First, the standard practice of picking a threshold to maximize $F_1$ or Youden's $J$ on a validation set is a decision rule with an *implicit* loss, and the implied cost ratio depends on the base rate of the validation set. If your validation set was rebalanced (a very common preprocessing step) then the implied costs bear no relation to deployment. Corollary 10.43 gives you the alternative: state $c_{FP}$ and $c_{FN}$, use the deployment base rate, compute the threshold. It is one line and it is auditable.

Second, all of this presumes calibrated posteriors, and a deep classifier's softmax output is typically not calibrated (Definition 10.39). Corollary 10.43 applied to an over-confident score gives a threshold that looks principled and is not. The order of operations is: calibrate, then threshold, then monitor both.

> **Self-check 10.9.**
> 1. A posterior is right-skewed. Order the mean, median, and mode, and say which loss produces each.
> 2. For the loss of Proposition 10.42 with $c_u=9,c_o=1$, what quantile do you report, and why is it not the mean?
> 3. In Corollary 10.43, if $c_{FN}=c_{FP}$, what is the threshold? What does that tell you about the ubiquitous $0.5$ cutoff?
> 4. Why can the threshold be retuned after deployment without retraining the model?

## 10.10 Results used but not proved

This table is meant to be complete: if this chapter used a result it did not prove, it is listed here.

| Result | Used in | Where to find a proof |
|---|---|---|
| Radon–Nikodym theorem; densities w.r.t. a stated base measure | Definitions 10.1–10.2; Proposition 10.37(ii) | Ch. 2 of this tutorial |
| Fubini–Tonelli theorem | Proposition 10.3; Theorem 10.16(iii) | Ch. 4 |
| Existence of the joint law from a prior and a probability kernel | Definition 10.1 | Ch. 4 (product measures); Kallenberg, *Foundations of Modern Probability*, Thm 8.4 |
| Dominated Convergence Theorem (differentiation under the integral sign) | Theorem 10.16(ii) | Ch. 2 |
| Jensen's inequality | Corollary 10.17 | Ch. 2 |
| Change-of-variables / pushforward density with Jacobian factor | Proposition 10.18; Theorem 10.20; Worked Example 10.23 | Ch. 3 |
| Exponential-family canonical form, sufficiency, and $\nabla A(\eta)=\mathbb{E}_\eta[T(X)]$ | Theorem 10.16 | Ch. 3 |
| Gaussian identities: affine images, sums of independents, completing the square, conditioning | Propositions 10.13, 10.29, 10.35, 10.36; Theorems 10.24, 10.33 | Ch. 5 |
| Uniqueness of the Gaussian canonical form (density $\propto\exp$ of a negative-definite quadratic $\Rightarrow$ Gaussian) | Proposition 10.13; Theorems 10.24(iii), 10.33 | Ch. 5 |
| Conditional expectation: tower property, $L^2$ projection, law of total variance | Proposition 10.7; Theorem 10.24(iii); Theorem 10.41(i); Definition 10.39 | Ch. 6 |
| Law of large numbers (observed information $\to$ Fisher information) | Theorem 10.38(iii); Corollary 10.31 | Ch. 7 |
| MAP as penalized likelihood; admissibility; James–Stein dominance | Proposition 10.22; Corollary 10.25 | Ch. 8 |
| Fisher information: definition, regularity conditions, transformation rule as a metric | Definition 10.19; Theorem 10.20; Corollary 10.31 | Ch. 9 |
| Pareto-smoothed importance sampling and the $\hat k$ diagnostic | Definition 10.32 | Ch. 2; Vehtari, Simpson, Gelman, Yao & Gabry (2024), *JMLR* 25:72 |
| de Finetti representation theorem | Theorem 10.5; §10.1.2 | Kallenberg, *Foundations of Modern Probability*, Thm 11.10; Schervish, *Theory of Statistics*, §1.5 |
| Bernstein–von Mises theorem | Theorem 10.9; §10.7.1; Common Confusion 10.8 | van der Vaart, *Asymptotic Statistics*, Thm 10.1 |
| Posterior concentration at the KL-closest point, and the misspecified BvM with sandwich covariance (stated) | §10.1.4 | Kleijn & van der Vaart (2012), *Electron. J. Statist.* 6:354–381; Berk (1966), *Ann. Math. Statist.* 37:51–58 |
| Inconsistency of the joint Jeffreys prior in the Neyman–Scott problem (stated) | §10.3.3 | Bernardo & Smith, *Bayesian Theory*, §5.6.2; Neyman & Scott (1948), *Econometrica* 16:1–32 |
| Sylvester's determinant identity $\det(I_n+AB)=\det(I_d+BA)$ | Proposition 10.36 | Horn & Johnson, *Matrix Analysis*, §0.8.5 |
| Divergence theorem (boundary term vanishes for interior hyperparameters) | Theorem 10.16(ii) | any multivariable analysis text, e.g. Rudin, *Principles*, Ch. 10 |
| Taylor's theorem with remainder | Theorems 10.30, 10.38 | Ch. 2 appendix / standard calculus |
| Convexity is preserved by integration | Theorem 10.41(ii); Proposition 10.42 | standard; Boyd & Vandenberghe §3.2 |
| Beta and Gamma function identities, $B(a,b)=\Gamma(a)\Gamma(b)/\Gamma(a+b)$ | Propositions 10.11, 10.14, 10.15 | standard; Abramowitz & Stegun Ch. 6 |
| Dirichlet aggregation: sub-vector marginals are Dirichlet/Beta | Proposition 10.12 | Gelman et al., *BDA3*, App. A |
| Gamma-integral identity yielding the Student-$t$ from a Normal-Inverse-Gamma | Proposition 10.14 | Murphy, *Machine Learning: A Probabilistic Perspective*, §4.6.3 |
| $\mathbb{E}[1/\chi^2_J]=1/(J-2)$ for $J\ge3$ | Corollary 10.25 | Casella & Berger, *Statistical Inference*, §5.3 (also Exercise A.5 here) |
| AIC as a bias-corrected estimate of out-of-sample log predictive density | §10.5.4 | Burnham & Anderson, *Model Selection and Multimodel Inference*, Ch. 2 |
| Consistency of the binned ECE estimator | Definition 10.39 | Guo et al. (2017); Kumar, Liang & Ma (2019), *NeurIPS* |
| Variance-stabilizing property of $\arcsin\sqrt{\cdot}$ for binomial proportions | Worked Example 10.14 | Casella & Berger, *Statistical Inference*, §5.5 |
| Impropriety of the posterior under a flat prior on a hierarchical variance | §10.3.4 | Gelman (2006), *Bayesian Analysis* 1(3):515–534 |
| Dropout training as variational inference (stated and criticized, not proved) | §10.8.2 | Gal & Ghahramani (2016), *ICML*; critiques in Osband (2016), *NeurIPS Workshop* |
| Neural-network weight-space symmetries (permutation, ReLU rescaling) | §10.8.1 | Hecht-Nielsen (1990); Dinh et al. (2017), *ICML* |

## 10.11 Chapter summary

1. **A Bayesian model is a prior plus a probability kernel**, and it defines one joint law over parameters and data (Definition 10.1). The posterior is a conditional law — the object Chapter 6 built — and Proposition 10.3 shows it exists for almost every dataset the model can produce. You do not get to opt out of having a prior: de Finetti (Theorem 10.5) says that if your beliefs about observables are exchangeable, a prior already exists, and declining to name it is not the same as not having one.
2. **Updating is coherent and order-free** (Theorem 10.4) — with the hypothesis said out loud: order-freeness needs the joint likelihood to be symmetric in the data (conditional i.i.d. suffices; time-series models keep sequential updating but lose shuffling). Batch, streaming, and shuffled processing agree exactly for exchangeable models; the current posterior is a sufficient summary of the past, which is what licenses online Bayesian systems. The same factorization means the posterior does not depend on *why you stopped*, so optional stopping and peeking are harmless to a Bayesian analysis and are not harmless to a $p$-value. **The evidence is the product of one-step-ahead predictive probabilities** (Theorem 10.4(iii)), verified numerically to ten decimals in Worked Example 10.1. This single identity explains both the Occam behaviour of §10.5 and the resemblance of evidence to cross-validation.
3. **Predict with the posterior predictive, not the plug-in.** Its variance splits into aleatoric and epistemic parts (Proposition 10.7), and it is strictly wider than the plug-in (Corollary 10.17): $16\%$ wider on five coin flips, a $24\%$ higher probability of five heads, $4.9\times$ more mass beyond three sigma in the six-measurement Gaussian example.
4. **All standard conjugate pairs are one theorem** (Theorem 10.16), and its content is that hyperparameters are pseudo-counts: $a_0+b_0$ flips, $\alpha_0$ class occurrences, $\beta_0$ exposure units, $\kappa_0$ observations. Say your prior's pseudo-count out loud and you will usually discover whether it is as mild as you claimed.
5. **Additive smoothing, label smoothing, and count-based exploration bonuses are posterior predictives** of Dirichlet–Categorical models (Worked Example 10.5). The unseen category's probability $0.0417$ was not a hack; it was $\mathbb{E}[\theta_3\mid D]$.
6. **"Flat" is a property of coordinates, not of belief.** Priors are measures and transform with Jacobians (Proposition 10.18); a uniform prior on $\theta$ is the logistic density on the log-odds scale, and the two "uninformative" choices disagree by a factor of $3.2$ about $\{\theta<0.1\}$. **The Jeffreys prior $\propto\sqrt{\det I(\theta)}$ is the unique fix**, and it works because Fisher information transforms as a metric so that $\sqrt{\det I}$ produces exactly the Jacobian that flat priors are missing (Theorem 10.20).
7. **Improper priors are a limit you must check per dataset** (Proposition 10.21). The Haldane prior gives no posterior at all when $s\in\{0,n\}$, and flat priors on hierarchical variances are a known source of improper posteriors.
8. **Weight decay is a Gaussian prior with $\lambda=\sigma^2/\tau^2$** (Proposition 10.22), and the translation involves $n$ and $\sigma$, so a default weight-decay constant corresponds to a different prior in every project.
9. **Hierarchical models interpolate between one estimate per group and one for all**, with shrinkage factor $B_j=s_j^2/(s_j^2+\tau^2)$ (Theorem 10.24). On the eight-variant example this cut MSE by $50.2\%$ on the displayed dataset and $40.1\%$ averaged over 500 replications, and it re-derives James–Stein as empirical Bayes (Corollary 10.25).
10. **The marginal likelihood is a distribution over datasets**, so Occam's razor needs no penalty term: a uniform-prior coin model spends $1/21$ on each of 21 outcomes while the fixed-$\theta$ model spends $0.0739$ on $s=13$, and the Bayes factor favours the simpler model at $s=13$ despite an MLE of $0.65$.
11. **Model comparison is far more prior-sensitive than estimation.** The same data flip the verdict from $2.3{:}1$ against the flexible model to $1.1{:}1$ for it, and Jeffreys–Lindley (Proposition 10.29) shows $p=0.05$ coexisting with $P(M_0\mid D)=0.936$ at $n=10^4$. Never compute a Bayes factor with a diffuse or improper prior.
12. **BIC is the Laplace approximation with the prior and curvature thrown away**, so its error converges to a nonzero constant ($-0.1786$ in Worked Example 10.17) while Laplace's error decays like $1/n$. AIC estimates something else entirely; do not mix them.
13. **Bayesian linear regression's posterior mean is exactly ridge** with $\lambda=\sigma^2/\tau^2$ (Corollary 10.34, verified to $1.8\times10^{-15}$), and its predictive variance is what ridge cannot express: $0.0162$ inside the data and $1.4628$ half a unit outside it. Evidence maximization then picks $\lambda$ and even the polynomial degree from training data alone.
14. **A sparse MAP is not a sparse posterior** (Proposition 10.37): with a Laplace prior the posterior gives probability zero to $\{\theta=0\}$, and at $y=0.6$ the lasso says $0$ while the posterior mean is $0.291$.
15. **The Laplace approximation is cheap and symmetric**, so it fails exactly where posteriors are skewed: it put $2.5\%$ of its mass below $-0.44$ for a logistic coefficient whose exact $2.5\%$ point was $+0.11$. Reparameterize to unconstrained coordinates before approximating. **In deep learning the goal is calibrated predictive uncertainty, not a faithful posterior.** Deep ensembles are not Bayesian and are usually the strongest baseline; MC dropout's spread is set by a regularization hyperparameter and does not shrink with $n$.
16. **Your loss chooses your summary** (Theorem 10.41): squared $\to$ mean, absolute $\to$ median, 0-1 $\to$ mode, asymmetric linear $\to$ quantile (Proposition 10.42). On a $\mathrm{Gamma}(9,6)$ posterior those are $1.500$, $1.445$, and $1.333$. **Binary decision thresholds are cost ratios, not probabilities** (Corollary 10.43). At $P(D\mid+)=0.0905$ the break-even ratio is $10.05$: treat at $20{:}1$, do not at $10{:}1$. The $0.5$ cutoff asserts equal costs.

### Callbacks

This chapter discharged several promissory notes. Chapter 2's Radon–Nikodym machinery finally earned its keep in the "density with respect to what?" bookkeeping that made Proposition 10.18 and Proposition 10.37(ii) precise rather than vague, and Chapter 2's KL/ELBO material was set up for Chapter 13 rather than used here. Chapter 3's change-of-variables formula, promised as the tool that would eventually explain "why is there a Jacobian in my prior", delivered exactly that in Theorem 10.20. Chapter 5's completing-the-square identities were promised as the engine of Bayesian linear algebra and produced Theorem 10.24 and Theorem 10.33 in a few lines each. Chapter 6 promised that conditional expectation would reappear as "the answer to a decision problem"; Theorem 10.41 is that statement. Chapter 8 left James–Stein as a technical marvel with no intuition attached, and Corollary 10.25 supplied the intuition: it is empirical Bayes in a Gaussian hierarchical model. Chapter 8's MAP-equals-penalized-likelihood observation was upgraded to a precise dictionary in Proposition 10.22, including the $\lambda=\sigma^2/\tau^2$ identification. Chapter 9's tests and confidence intervals were promised a Bayesian counterpart to compare against; Common Confusion 10.2 and the Jeffreys–Lindley analysis of §10.5.3 provide the comparison, fairly and with the disagreement quantified. And Chapter 1's disease-screening example, left at $P(D\mid+)=9.05\%$ with the moral "base rates dominate", was finished in Worked Example 10.25: the posterior alone decides nothing, and the missing ingredient was the cost ratio.

### Before you continue

Answer these without looking back. If you cannot, reread the section named.

1. State the definition of the posterior as a measure, and say exactly what can go wrong if the prior is improper. (§10.1, §10.3.4)
2. Explain in one sentence why the posterior does not depend on the order of the data, and identify where conditional independence given $\theta$ enters. (§10.1.2)
3. A colleague proposes $\mathrm{Beta}(30,10)$ as a "mild" prior on a conversion rate and has 40 observations. Quantify the objection using pseudo-counts. (§10.2)
4. Why is a uniform prior on a probability *not* uninformative on the log-odds scale? Give the induced density. (§10.3.2)
5. Write down the shrinkage factor in a Gaussian hierarchical model and say what happens to it in the three limits $n_j\to\infty$, $\tau^2\to0$, $\tau^2\to\infty$. (§10.4.2)
6. Explain why a model with more parameters does not automatically have higher marginal likelihood, using the phrase "distribution over datasets". (§10.5.1)
7. Give the regularization parameter of ridge regression in terms of the Bayesian model's variances, and state one thing the Bayesian version reports that ridge cannot. (§10.6)
8. Your posterior for a delay is right-skewed. You must report a single number, and being late costs four times as much as being early. What do you report? (§10.9.2)
9. Name two hypotheses of Bernstein–von Mises that fail for deep networks. (§10.1.4)

### Looking ahead to Chapter 11

Chapter 11 turns to information theory, and it will feel like a change of subject for about two pages before it becomes obvious that it is not. We will define entropy and cross-entropy and show that the loss function you have been minimizing in every classification experiment is a KL divergence between an empirical distribution and a model, so that "minimize cross-entropy" and "maximize likelihood" and "minimize $D_{\mathrm{KL}}(\hat P_n\|P_\theta)$" are three names for one optimization. From there: mutual information and its role in representation learning and feature selection; the general family of $f$-divergences, of which KL, reverse KL, total variation, Hellinger, and Jensen–Shannon are members; and their *variational representations*, which are what let a divergence be estimated by training a discriminator — the mathematical content of GANs, of contrastive learning objectives, and of mutual-information estimators. We will then meet optimal transport and the Wasserstein distances, which behave well precisely where the $f$-divergences fail: on distributions with disjoint supports, the situation Chapter 1 diagnosed for generative models with low-dimensional latents and Chapter 2 revisited with the constant-JS-loss problem. The chapter's organizing question is the one every practitioner should be able to answer about their own training script: *why is your loss function what it is, and what geometry on distributions does it implicitly assume?*

# Exercises

Difficulty: ★ routine consolidation, ★★ requires an idea, ★★★ paper-level.

## Part A — Theory

**A.1 (★) Elicitation.** You believe a conversion rate has mean $0.3$ and standard deviation $0.1$. Find the $\mathrm{Beta}(a,b)$ with those moments, state its pseudo-count, and say how many real observations are needed before the data dominate the prior by 4:1.

**A.2 (★★) How fast does the prior wash out?** For the Beta–Bernoulli model, prove
$$
\mathbb{E}[\theta\mid D]-\hat\theta_{\text{MLE}}=\frac{(a_0+b_0)\big(m_0-\hat\theta_{\text{MLE}}\big)}{a_0+b_0+n},\qquad m_0=\frac{a_0}{a_0+b_0},
$$
and verify it reproduces the $|{\rm diff}|$ column of Worked Example 10.3 at $n=20$ and $n=2000$. Then state precisely which hypothesis of Bernstein–von Mises this computation illustrates.

**A.3 (★★) Jeffreys priors and propriety.** Compute the Jeffreys prior for (a) $\mathrm{Poisson}(\lambda)$, $\lambda>0$; (b) $\mathrm{Exponential}$ with rate $\lambda$; (c) $N(\mu,\sigma^2)$ with both unknown, in the $(\mu,\sigma)$ parameterization. For each, say whether the prior is proper and whether the posterior is proper after $n\ge1$ observations. In (a), contrast the behaviour at $t=\sum x_i=0$ with the Haldane–Bernoulli failure of Proposition 10.21(ii).

**A.4 (★★★) Normal-Inverse-Gamma, in full.** Complete the proof of Proposition 10.14: (i) derive the four hyperparameter updates by completing the square in $\mu$; (ii) marginalize $\sigma^2$ to obtain the Student-$t$ posterior for $\mu$, identifying scale and degrees of freedom; (iii) obtain the posterior predictive. State where you used the fact that the prior on $\mu$ was scaled by $\sigma^2$, and what breaks if you use an independent prior $\mu\sim N(\mu_0,\tau^2)$ instead.

**A.5 (★★) Finishing James–Stein.** Prove $\mathbb{E}[1/\chi^2_J]=1/(J-2)$ for $J\ge3$, then complete the proof of Corollary 10.25 by showing $\hat B=(J-2)s^2/\sum_jy_j^2$ is unbiased for $B=s^2/(s^2+\tau^2)$ under the hierarchical model. Why does unbiasedness of $\hat B$ *not* imply that $(1-\hat B)y$ is unbiased for $\theta$?

**A.6 (★★) When is the predictive not over-dispersed?** In Proposition 10.7, characterize exactly when $\mathrm{Var}(\tilde X\mid D)=v(\hat\theta)$ with $\hat\theta=\mathbb{E}[\theta\mid D]$, for a model with $\theta$-independent conditional variance. Give a nontrivial example (i.e. one where $\theta$ is genuinely unknown) in which the epistemic term vanishes.

**A.7 (★★) Jeffreys–Lindley, both limits.** Derive the $\mathrm{BF}_{01}$ formula of Proposition 10.29 from scratch, then evaluate $\lim_{\tau\to0}$ and $\lim_{\tau\to\infty}$ at fixed $n,z$. Interpret each limit as a statement about model comparison, and use them to explain why an "uninformative" prior is not a safe default for a Bayes factor.

**A.8 (★★★) Spike-and-slab: the prior that can say zero.** Let $y\mid\theta\sim N(\theta,s^2)$ and $\pi=w\,\delta_0+(1-w)N(0,\tau^2)$.
(i) Identify the base measure with respect to which $\pi$ has a density, and explain why Proposition 10.37(ii) does not apply.
(ii) Show $\Pi(\theta=0\mid y)=\dfrac{w\,\phi(y;0,s^2)}{w\,\phi(y;0,s^2)+(1-w)\,\phi(y;0,s^2+\tau^2)}$ where $\phi(\cdot;0,v)$ is the $N(0,v)$ density.
(iii) With $s=\tau=1$, tabulate $\Pi(\theta=0\mid y)$ and $\mathbb{E}[\theta\mid y]$ for $w\in\{0.5,0.9\}$ and $y\in\{0.6,1.5,3.0\}$, and compare to Worked Example 10.21.

**A.9 (★) 0-1 loss and its pathologies.** Prove Theorem 10.41(iii) in the countable case. Then give a discrete posterior on $\{0,1,\dots,10\}$ whose mode is $0$ while $\Pi(\theta\ge5\mid D)>0.7$, and say in one sentence what this shows about MAP reporting.

## Part B — Computational labs

Report seeds and sample sizes. Numbers below are from the reference implementation in **Appendix B**, which you should attempt only after your own; if you use the same seeds and library versions your figures should match to the stated precision, and the exact identities (B.1(c), B.2(a), B.4(b)) must match to machine precision regardless.

**B.1 (★) Sequential updating and prior sensitivity.**
(a) Simulate 80 Bernoulli$(0.65)$ draws with `numpy` default RNG seed 2 and plot the $\mathrm{Beta}(2,2)$-prior posterior after $0,2,5,20,80$ observations. You should obtain posterior means $0.500,0.667,0.667,0.667,0.655$ (Figure 10.1).
(b) With $n=20$, $s=13$ fixed, reproduce the prior-sensitivity table of Worked Example 10.4 for the six priors listed, including 95% credible intervals.
(c) Update one observation at a time in three random orders and confirm you land on $\mathrm{Beta}(15,9)$ each time. Then confirm $\sum_k\log p(x_k\mid D_{k-1})=\log\big(B(15,9)/B(2,2)\big)=-14.0190920131$.

**B.2 (★★) Bayesian linear regression, ridge, and bands.** Use $n=12$, $x_i\sim\mathrm{Uniform}(-1,1)$ sorted, $y_i=1+2x_i-1.5x_i^2+\varepsilon_i$, $\varepsilon_i\sim N(0,0.25^2)$, seed 10, cubic basis.
(a) Compute the posterior of Theorem 10.33 with $\tau^2=1$ and verify $\|m-\hat\theta^{\text{ridge}}_{0.0625}\|_\infty<10^{-12}$.
(b) Plot the predictive mean with $\pm2$ sd bands for parameter-only and full predictive variance on $x\in[-1.5,1.5]$; reproduce the variance table of Worked Example 10.19.
(c) Maximize the evidence of Proposition 10.36 over $\tau^2$; you should find $\tau^2=1.5565$, $\lambda^*=0.040154$, $\log p(y)=-6.5606$. Verify that the posterior mean at $\lambda^*$ again equals ridge at $\lambda^*$.
(d) Compute the evidence for polynomial degrees $0$–$7$ with $\tau^2$ optimized per degree, and compare the ranking to the OLS residual sum of squares. Which criterion recovers the true degree?

**B.3 (★★) Hierarchical shrinkage.** With $J=8$, $n_j=(50,50,100,100,400,400,2000,2000)$, $\theta_j\sim N(0.20,0.04^2)$, clicks $\sim\mathrm{Bin}(n_j,\theta_j)$, seed 10:
(a) fit empirical Bayes (Definition 10.26) and reproduce $\hat\mu=0.2056$, $\hat\tau^2=2.373\times10^{-4}$;
(b) compute MSE for no pooling, complete pooling, and partial pooling against the true $\theta_j$ ($1.3156\times10^{-3}$, $1.1334\times10^{-3}$, $6.5529\times10^{-4}$);
(c) repeat the whole experiment 500 times (seed 11) and report mean MSEs and the fraction of replications in which partial pooling wins;
(d) modify the design so that all $n_j=2000$ and explain what happens to the shrinkage factors and to the MSE advantage.

**B.4 (★★) A marginal likelihood, two ways, and its prior sensitivity.** Data $x=(2.1,1.8,2.5,2.0,2.3,1.7)$, model $N(\mu,\sigma^2)$ with $\sigma^2=0.09$ known, prior $\mu\sim N(2,\tau^2)$.
(a) Compute $\log p(x)$ analytically using $x\sim N(\mu_0\mathbf{1},\ \sigma^2I+\tau^2\mathbf{1}\mathbf{1}^{\!\top})$ at $\tau^2=0.25$; you should get $-2.25253845$.
(b) Compute the same quantity by one-dimensional quadrature over $\mu$ and report the absolute difference (reference: $4.4\times10^{-16}$).
(c) Evaluate the evidence at $\tau^2\in\{0.01,0.25,1,100\}$ and explain the monotonicity you see in terms of the Occam factor.
(d) Explain why part (c) makes Bayes factors dangerous but does not make posterior means dangerous.

**B.5 (★★★) Laplace versus exact, and the separability trap.** One-dimensional logistic regression, prior $N(0,2^2)$, data $x=(-2,-1,-0.5,0.5,1,2)$, $y=(0,0,1,1,1,1)$.
(a) Find the MAP and the Laplace sd; reproduce $1.500369$ and $0.990957$.
(b) By quadrature on $[-30,30]$, compute the exact posterior mean, sd, and $2.5/50/97.5$ percentiles ($1.858382$, $1.057422$, $0.110258$, $1.745797$, $4.223451$).
(c) Compare the exact predictive at $x_*=0.75$ ($0.774730$) with the MAP plug-in ($0.754966$) and the Laplace average ($0.732759$ by Monte Carlo with $2\times10^6$ draws, seed 0). Explain the direction of the Laplace error.
(d) Replace the data with $x=(-2,-1,1,2)$, $y=(0,0,1,1)$ and compute the MAP for prior sds $2,10,100$ ($1.7174$, $3.9453$, $7.8441$). What does this imply about reporting Laplace error bars for a well-fit deep classifier?

## Part C — Paper-style derivations

**C.1 (★★★) The general exponential-family conjugate prior.** Let $p(x\mid\eta)=h(x)\exp(\langle\eta,T(x)\rangle-A(\eta))$ be a natural exponential family with $\mathcal{N}$ open.
(a) Prove Theorem 10.16(i) and (iii).
(b) Prove Theorem 10.16(ii), stating carefully the condition under which the boundary term vanishes.
(c) Specialize to Bernoulli, Poisson, and the Gaussian with known variance, exhibiting $(\chi,\nu)$ in terms of the usual hyperparameters, and verify the pseudo-count reading in each case.
(d) Explain, with a formula, why the posterior mean of $\nabla A(\eta)$ — not of $\eta$ — is the quantity with the clean convex-combination form, and why this matters when someone reports "the posterior mean of the logit".

**C.2 (★★★) Jeffreys priors from first principles.**
(a) Derive the transformation rule $I_\theta=J^{\!\top}I_\phi J$ for Fisher information under a $C^1$ diffeomorphism, stating the regularity you need.
(b) Prove Theorem 10.20, exhibiting explicitly the cancellation between the Jacobian of the change of measure and the Jacobian in $\sqrt{\det I}$.
(c) Compute the Jeffreys prior for $N(\mu,\sigma^2)$ with both parameters unknown and show it is $\propto\sigma^{-2}$ in the $(\mu,\sigma)$ parameterization. Then explain why applied Bayesians usually prefer the "independence Jeffreys" prior $\propto\sigma^{-1}$ and what invariance is sacrificed.
(d) Give a model in which the Jeffreys prior is improper *and* the posterior is improper for some datasets, and state the practical diagnostic you would run before trusting a sampler on such a model.

**C.3 (★★★) Bayesian linear regression, end to end.** With the model of §10.6.1:
(a) derive the posterior (Theorem 10.33) by completing the square, being explicit about why $A\succ0$;
(b) derive the posterior predictive at a new $\varphi_*$ and decompose its variance;
(c) derive the evidence in both the marginal-Gaussian and the "fit + Occam factor" forms of Proposition 10.36, proving the determinant and quadratic-form identities you use;
(d) prove that the posterior mean equals the ridge estimator and identify $\lambda$ exactly;
(e) explain, in terms your (b) makes precise, what the predictive variance expresses that the ridge point estimate structurally cannot, and give a deployment decision that depends on the difference;
(f) state what changes if $\sigma^2$ is unknown with an Inverse-Gamma prior (name the resulting predictive family; you may cite Proposition 10.14).

# Appendix A — Self-check answers

**Self-check 10.1.**
1. Conditional independence given $\theta$ was used to factor $p(D_n\mid\theta)=\prod_kp(x_k\mid\theta)$, which is what makes the recursion (i) and the telescoping (iii) work. Order-invariance (ii) needs only that the joint likelihood is a *symmetric* function of the observations; exchangeability suffices, and conditional i.i.d. is one way to get it. So (ii) is the weaker statement.
2. Under the *marginal law of the data implied by the model*, $m(\mathrm{d}x)=Z(x)\mu(\mathrm{d}x)$. The qualifier matters because the statement is circular-sounding but not circular: it guarantees nothing about datasets generated by a different mechanism than your model. If your model is badly misspecified you can observe data in a set your model considers null, and then "almost surely" gives no protection.
3. Any model where the mean does not depend on the unknown parameter. Example: $X\mid\sigma^2\sim N(0,\sigma^2)$ with $\sigma^2$ unknown. Then $\mu(\theta)\equiv0$, so $\mathrm{Var}(\mu(\theta)\mid D)=0$ for every $n$, and all predictive variance is the aleatoric term $\mathbb{E}[\sigma^2\mid D]$. Note the uncertainty about $\sigma^2$ has not disappeared — it shows up inside the aleatoric term. The epistemic/aleatoric labels are about the *mean function*, not about knowledge in general.
4. Any two of: the dimension grows with $n$ (so "fixed $d$" fails); the model is non-identifiable because of permutation and rescaling symmetries, so the posterior is multimodal and there is no single $\theta_0$; the MAP typically sits at or near a boundary of the region where the loss is flat (interpolation regimes); efficient estimators and nonsingular Fisher information are unavailable because $I(\theta)$ is singular (flat directions).

**Self-check 10.2.**
1. $\mathrm{Beta}(100,100)$ carries $200$ pseudo-observations, so with $n=200$ real impressions the prior gets exactly half the weight in the posterior mean, and it forces that half toward $0.5$ — a wildly implausible click-through rate. It is a dogmatic prior, not a weak one.
2. $\beta_n=\beta_0+n\to\infty$, so $1+1/\beta_n\to1$: the predictive converges to the plug-in Poisson. Over-dispersion is entirely a finite-sample effect of parameter uncertainty here, which also tells you that persistent over-dispersion in real count data is evidence of model misspecification (heterogeneity), not just estimation uncertainty.
3. $C(\chi,\nu)<\infty$ becomes $B(a_0,b_0)<\infty$, i.e. $a_0>0$ and $b_0>0$. The Haldane choice $a_0=b_0=0$ violates it, which is exactly the impropriety of Proposition 10.21(ii).
4. Because in Proposition 10.13 the variance is known, so the posterior for $\mu$ is Gaussian; in Proposition 10.14 we must integrate over an unknown $\sigma^2$, and a Gaussian whose variance is mixed over an Inverse-Gamma is a Student-$t$. The $t$'s heavy tails encode uncertainty about the scale.

**Self-check 10.3.**
1. No. If $\pi(\log\sigma)\propto1$ then $\pi(\sigma)\propto1/\sigma$ and, with $v=\sigma^2$, $\pi(v)\propto1/v$ — strongly peaked at zero (and improper at both ends). A flat prior on $\log\sigma$ is the scale-invariant choice, and it is emphatically not flat on the variance.
2. The cancellation in the last display. With $\det I$ instead of $\sqrt{\det I}$ we would get a factor $(\det J)^2$ from the transformation rule but only $|\det J|$ from the change of measure, leaving a residual $|\det J|$: not invariant. The square root is forced.
3. Bernoulli data with $s=0$ (or $s=n$): the posterior kernel $\theta^{-1}(1-\theta)^{n-1}$ has a non-integrable singularity at $0$. What went wrong: the Haldane prior places infinite mass arbitrarily close to $\theta=0$, and data consisting of only failures never contradicts $\theta\approx0$, so nothing tames the singularity.
4. Effective summed penalty $\lambda=nc=50{,}000\times10^{-3}=50$, so $\tau=\sigma/\sqrt\lambda=0.5/\sqrt{50}=0.0707$. You are asserting that every weight is within about $\pm0.14$ of zero with 95% prior probability — a much stronger claim than most people intend.

**Self-check 10.4.**
1. $B_j=1/2$: the group estimate sits exactly halfway between its own data and the population mean. In words, one group's worth of data is exactly as informative as the population-level knowledge about where that group should be.
2. Marginally, $\mathrm{Cov}(\theta_i,\theta_j)=\mathrm{Var}(\mu)>0$ for $i\ne j$ once $\mu$ is uncertain, because both are pulled by the same $\mu$. That dependence is the channel for borrowing strength: without it, $p(\theta_j\mid \text{all data})=p(\theta_j\mid D_j)$ and no pooling could occur.
3. Under-estimating $\tau^2$ makes the $B_j$ too large: too much shrinkage, and posterior sds that are too small (they are computed as if $\tau^2$ were known). Both errors point the same way — over-confidence. Full Bayes over $\tau^2$ or a hierarchical prior on it is the fix.
4. When the groups genuinely do not differ ($\tau^2=0$), or when $J$ is small and the per-group data are so thin that $\hat\tau^2$ cannot be distinguished from zero. Complete pooling then has the same mean and lower variance. In Worked Example 10.14's replication study complete pooling lost on average, because the true $\tau=0.04$ was large relative to the sampling sds of the large groups.

**Self-check 10.5.**
1. Proposition 10.11(iv) with $a_n=b_n=1$, $m=20$: the Beta-Binomial with uniform parameters is the discrete uniform. The mechanism: a uniform prior over $\theta$ makes every count equally likely a priori, which is exactly the maximum-spread predictive distribution — the model has committed to nothing.
2. $\mathrm{BF}_{01}\to1$: as $\tau\to0$, $M_1$ becomes indistinguishable from $M_0$, so no data can discriminate them. That is the right answer and it is a good sanity check on the formula.
3. It contains $\log\pi(\hat\theta)+\frac{d}{2}\log2\pi-\frac12\log\det I(\theta_0)$, i.e. the prior density at the mode and the curvature scale. Here $d=1$, $\pi\equiv1$ on $[0,1]$, $I(0.65)=1/(0.65\cdot0.35)=4.396$, giving $\frac12\log2\pi-\frac12\log4.396=0.9189-0.7403=0.1786$ — matching the observed $0.1786$ to the accuracy of the remaining $O(1/n)$ terms.
4. "What is the prior's scale, and how does the Bayes factor change if you halve or double it?" A $\mathrm{Uniform}(-10^6,10^6)$ prior makes the Bayes factor essentially a report of the prior's width (§10.5.2), so a factor of 30 is uninterpretable without a sensitivity analysis.

**Self-check 10.6.**
1. $m\to(\Phi^{\!\top}\Phi)^{-1}\Phi^{\!\top}y$, the OLS estimator, which requires $\Phi^{\!\top}\Phi$ invertible, i.e. $\mathrm{rank}(\Phi)=d$ (so $n\ge d$ and no collinearity). When $\Phi$ is rank-deficient the limit does not exist, which is precisely the case where the prior is doing indispensable work.
2. Because $\varphi_*=(1,1.5,2.25,3.375)$ has large high-order entries, and $\Sigma$ has its largest variances in exactly those coordinates (posterior sd $0.527$ for the cubic coefficient). The quadratic form $\varphi_*^{\!\top}\Sigma\varphi_*$ multiplies large uncertainties by large feature values, so extrapolation error grows polynomially in the distance outside the data.
3. $-\frac d2\log\tau^2-\frac12\log\det A$. As $\tau^2\to\infty$, $\det A\to\det(\Phi^{\!\top}\Phi/\sigma^2)$ is bounded while $-\frac d2\log\tau^2\to-\infty$: the Occam factor penalizes a prior so wide that the posterior is a vanishing fraction of it.
4. Because the posterior is absolutely continuous with respect to Lebesgue measure, so $\Pi(\theta=0\mid y)=0$; the lasso's zero is the location of a mode of that continuous density, not an atom of probability.

**Self-check 10.7.**
1. Any two of: the mode is a saddle or lies in a flat direction so $H$ is singular (very common at convergence with over-parameterized networks); the "mode" is only an SGD stopping point and not a stationary point at all; the loss has exactly-flat directions from weight-space symmetries; the Hessian has negative eigenvalues at the point reached, in which case Gauss–Newton or Fisher approximations are used precisely because they are positive semidefinite by construction.
2. Skewness. Standard deviation is a second moment and says nothing about asymmetry; the exact posterior's mass is shifted right of the mode, so a symmetric interval centred at the mode misplaces both endpoints in the same direction.
3. Because the Beta posterior on $(0,1)$ is bounded and skewed near the ends, whereas its logit transform is defined on all of $\mathbb{R}$, is much closer to symmetric, and has no boundary to violate — so the quadratic Taylor expansion is accurate over a wider region, and the resulting interval automatically respects $0<\theta<1$.
4. Uncertainty *per input*: a validation set gives you an aggregate error rate, whereas Laplace gives you $\varphi_*^{\!\top}\Sigma\varphi_*$-style input-dependent variance, which is what you need to abstain on a specific case, to choose the next query in active learning, or to flag extrapolation.

**Self-check 10.8.**
1. Because freezing all but the last layer makes the model exactly linear in the remaining parameters, with feature map $\varphi(x)$ = the penultimate-layer activations. Theorem 10.33's derivation used linearity in $\theta$ only, so it applies verbatim (and so do Proposition 10.35, Proposition 10.36, and Corollary 10.34).
2. Its predictive spread is controlled by the dropout rate, a fixed regularization hyperparameter, so it does not shrink as $n\to\infty$ — whereas any genuine posterior's width shrinks like $n^{-1/2}$.
3. The first is better calibrated (ECE $0.01$ versus $0.08$); you would deploy the second, after temperature-scaling it. Calibration without sharpness is worthless: predicting the base rate for everything achieves ECE $\approx0$ and accuracy at chance. Report a proper scoring rule (log-loss or Brier) as the primary metric and calibration as a diagnostic.
4. Multimodality. Both mean-field VI and Laplace place a single unimodal (indeed, Gaussian) approximation around one point, so they cannot represent distinct basins; independently initialized ensemble members land in different basins, giving predictive diversity that no single-mode approximation can produce.

**Self-check 10.9.**
1. For a right-skewed posterior: mode $<$ median $<$ mean. 0-1 loss gives the mode, absolute loss the median, squared loss the mean (Theorem 10.41).
2. The $q^\star=9/10$ quantile. It is not the mean because the loss is asymmetric: under-shooting costs nine times as much as over-shooting, so the optimal action is pushed far into the upper tail. Only for the symmetric squared loss does the mean appear, and only for $c_u=c_o$ does the median.
3. Threshold $=\frac12$. It tells you that thresholding a probability at $0.5$ is an assertion that false positives and false negatives cost the same — a claim that is false in essentially every application where classification matters, and one that is usually made silently.
4. Because Corollary 10.43 factorizes the decision into a posterior (a function of data and model only) and a threshold (a function of costs only). Changing costs changes only the threshold, so no refitting is needed — provided the posterior is calibrated, which is the assumption to monitor.

# Solutions to Exercises

## Part A

### A.1 Elicitation

*Thinking.* We need to invert two moment formulas. For $\mathrm{Beta}(a,b)$ with $k=a+b$: mean $m=a/k$ and variance $m(1-m)/(k+1)$. The variance formula is the useful one because it isolates $k$: the shape of a Beta is controlled by its total pseudo-count, and the mean only says where it sits.

From $\mathrm{Var}=m(1-m)/(k+1)$ we get
$$
k=\frac{m(1-m)}{\mathrm{Var}}-1=\frac{0.3\times0.7}{0.01}-1=21-1=20,
$$
so $a=km=6$ and $b=k(1-m)=14$. Check: $\mathrm{sd}=\sqrt{6\cdot14/(20^2\cdot21)}=\sqrt{84/8400}=0.100000$. ✓

Pseudo-count: $a+b=20$, i.e. this prior is worth twenty observations, of which six were conversions.

"Data dominate 4:1" means the data weight in the posterior mean (Proposition 10.11(ii)) is four times the prior weight: $n/(k+n)=4\cdot k/(k+n)$, i.e. $n=4k=80$. So you need 80 real observations before this prior is a minority partner at that ratio. That is a lot, and it is the honest consequence of claiming a prior sd as tight as $0.1$.

### A.2 How fast does the prior wash out?

*Thinking.* Both quantities are ratios with the same denominator, so just put them over a common denominator and see what survives.

$$
\mathbb{E}[\theta\mid D]-\frac sn=\frac{a_0+s}{a_0+b_0+n}-\frac sn
=\frac{n(a_0+s)-s(a_0+b_0+n)}{n(a_0+b_0+n)}
=\frac{na_0-s(a_0+b_0)}{n(a_0+b_0+n)}.
$$
Factor $(a_0+b_0)$ out of the numerator: $na_0-s(a_0+b_0)=n(a_0+b_0)\big(\frac{a_0}{a_0+b_0}-\frac sn\big)=n(a_0+b_0)(m_0-\hat\theta)$. Cancelling $n$,
$$
\mathbb{E}[\theta\mid D]-\hat\theta=\frac{(a_0+b_0)(m_0-\hat\theta)}{a_0+b_0+n}. \qquad\blacksquare
$$

Verification with $a_0=b_0=2$ (so $m_0=0.5$, $a_0+b_0=4$) and $\hat\theta=0.65$:

- $n=20$: $4(0.5-0.65)/24=-0.6/24=-0.025000$. Worked Example 10.3 reports $|{\rm diff}|=2.5\times10^{-2}$. ✓
- $n=2000$: $4(-0.15)/2004=-2.994\times10^{-4}$. Reported: $3.0\times10^{-4}$. ✓

The formula shows the bias is exactly $O(n^{-1})$ with constant $(a_0+b_0)(m_0-\hat\theta)$. This illustrates the Bernstein–von Mises hypothesis that the prior density be *positive and continuous at $\theta_0$*: the prior enters only through a fixed pseudo-count, which is swamped by $n$. Where the hypothesis fails — a prior with zero density at the truth, e.g. $\mathrm{Beta}(a_0,b_0)$ conditioned to $[0,0.5]$ when $\theta_0=0.65$ — the difference does *not* go to zero, because the posterior is confined to a region excluding the truth no matter how large $n$ is.

### A.3 Jeffreys priors and propriety

*Thinking.* In each case: write the log-likelihood for one observation, differentiate twice (or square the score), take the expectation, then take $\sqrt{\det}$. Then multiply by the likelihood for $n$ observations and check integrability.

**(a) Poisson.** $\ell=x\log\lambda-\lambda-\log x!$, so $\partial_\lambda\ell=x/\lambda-1$ and $\partial^2_\lambda\ell=-x/\lambda^2$; hence $I(\lambda)=\mathbb{E}[X]/\lambda^2=1/\lambda$. So
$$
\pi_J(\lambda)\propto\lambda^{-1/2},
$$
improper on $(0,\infty)$ (both ends: $\int_0^1\lambda^{-1/2}<\infty$ but $\int_1^\infty\lambda^{-1/2}=\infty$). Posterior kernel with $t=\sum x_i$: $\lambda^{t-1/2}e^{-n\lambda}$, i.e. $\mathrm{Gamma}(t+\tfrac12,n)$, proper for every $t\ge0$ and $n\ge1$ because the shape $t+\frac12>0$ and the rate $n>0$.

Contrast with Haldane–Bernoulli: there the failure at $s=0$ came from a *non-integrable singularity at the boundary* $\theta=0$ created by the $\theta^{-1}$ factor. Here the Jeffreys exponent is $-\frac12>-1$, so the singularity at $\lambda=0$ is integrable and even $t=0$ gives a proper posterior ($\mathrm{Gamma}(0.5,n)$, posterior mean $0.5/n$; with $n=10$, $0.050000$). The moral: the exponent matters, not the mere fact of impropriety.

**(b) Exponential, rate parameterization.** $\ell=\log\lambda-\lambda x$, $\partial_\lambda\ell=1/\lambda-x$, $\partial^2_\lambda\ell=-1/\lambda^2$, so $I(\lambda)=1/\lambda^2$ and
$$
\pi_J(\lambda)\propto1/\lambda,
$$
the scale-invariant prior (improper). Posterior kernel: $\lambda^{n-1}e^{-\lambda\sum x_i}$, i.e. $\mathrm{Gamma}(n,\sum x_i)$: proper for $n\ge1$ (assuming $\sum x_i>0$, which holds a.s.).

**(c) Gaussian, both unknown, $(\mu,\sigma)$ coordinates.** $\ell=-\log\sigma-\frac{(x-\mu)^2}{2\sigma^2}+\text{const}$. Then $\partial_\mu\ell=(x-\mu)/\sigma^2$ and $\partial_\sigma\ell=-1/\sigma+(x-\mu)^2/\sigma^3$. Taking expectations of the outer product with $\mathbb{E}[(X-\mu)^2]=\sigma^2$, $\mathbb{E}[(X-\mu)^3]=0$, $\mathbb{E}[(X-\mu)^4]=3\sigma^4$:
$$
I(\mu,\sigma)=\begin{pmatrix}1/\sigma^2&0\\0&2/\sigma^2\end{pmatrix},
\qquad \sqrt{\det I}=\frac{\sqrt2}{\sigma^2},
$$
so $\pi_J\propto\sigma^{-2}$, improper. The posterior kernel after $n$ observations is $\sigma^{-n-2}\exp(-[S+n(\bar x-\mu)^2]/2\sigma^2)$, which integrates over $\mu\in\mathbb{R}$ and $\sigma\in(0,\infty)$ to a finite value provided $n\ge2$ and $S>0$; with $n=1$ (so $S=0$) the $\sigma$-integral diverges. So: proper for $n\ge2$, improper for $n=1$ — a concrete example of "check propriety per dataset".

### A.4 Normal-Inverse-Gamma, in full

*Thinking.* Everything reduces to one algebraic identity for the $\mu$-quadratic. Do that identity first, then the $\sigma^2$-integral is a Gamma integral in disguise.

**(i) The updates.** Prior times likelihood, in terms of $u=1/\sigma^2$ for clarity:
$$
q\propto u^{\alpha_0+\frac12-1}e^{-u\beta_0}e^{-u\kappa_0(\mu-\mu_0)^2/2}\cdot u^{n/2}e^{-u[S+n(\bar x-\mu)^2]/2}.
$$
The key identity: for any $\kappa_0,n>0$,
$$
\kappa_0(\mu-\mu_0)^2+n(\mu-\bar x)^2=\kappa_n(\mu-\mu_n)^2+\frac{\kappa_0n(\bar x-\mu_0)^2}{\kappa_n},
\quad \kappa_n=\kappa_0+n,\ \mu_n=\frac{\kappa_0\mu_0+n\bar x}{\kappa_n}.
$$
*Proof of the identity:* expand both sides as quadratics in $\mu$. The $\mu^2$ coefficients are $\kappa_0+n=\kappa_n$ on both sides; the $\mu$ coefficients are $-2(\kappa_0\mu_0+n\bar x)=-2\kappa_n\mu_n$ on both sides; equating constants gives $\kappa_0\mu_0^2+n\bar x^2-\kappa_n\mu_n^2$, and substituting $\mu_n$ and simplifying yields $\kappa_0n(\bar x-\mu_0)^2/\kappa_n$. (The cleanest route to the constant term: it must be the value of the left side at $\mu=\mu_n$, which is a two-line computation.)

Substituting, $q\propto u^{\alpha_n+\frac12-1}\exp\{-u[\beta_n+\kappa_n(\mu-\mu_n)^2/2]\}$ with $\alpha_n=\alpha_0+n/2$ and $\beta_n=\beta_0+S/2+\kappa_0n(\bar x-\mu_0)^2/(2\kappa_n)$: exactly the $\mathrm{NIG}(\mu_n,\kappa_n,\alpha_n,\beta_n)$ kernel. ✓

**(ii) Marginalizing $\sigma^2$.** Integrate $u$ out:
$$
\pi(\mu\mid D)\propto\int_0^\infty u^{\alpha_n-\frac12}e^{-u[\beta_n+\kappa_n(\mu-\mu_n)^2/2]}\mathrm{d}u
=\Gamma(\alpha_n+\tfrac12)\Big[\beta_n+\frac{\kappa_n(\mu-\mu_n)^2}{2}\Big]^{-(\alpha_n+\frac12)},
$$
using $\int_0^\infty u^{c-1}e^{-ub}\mathrm{d}u=\Gamma(c)b^{-c}$ with $c=\alpha_n+\frac12$. Rewrite the bracket as $\beta_n\big[1+\frac{(\mu-\mu_n)^2}{2\beta_n/\kappa_n}\big]$ and compare with the $t_\nu(m,s^2)$ kernel $\big[1+\frac{(\mu-m)^2}{\nu s^2}\big]^{-(\nu+1)/2}$: matching exponents gives $\nu=2\alpha_n$, and matching the quadratic's scale gives $\nu s^2=2\beta_n/\kappa_n$, i.e. $s^2=\beta_n/(\alpha_n\kappa_n)$. ✓

**(iii) Predictive.** Given $(\mu,\sigma^2)$, $\tilde X\sim N(\mu,\sigma^2)$; given $\sigma^2$ and $D$, $\mu\sim N(\mu_n,\sigma^2/\kappa_n)$, so $\tilde X\mid\sigma^2,D\sim N(\mu_n,\sigma^2(1+1/\kappa_n))$. Now the same Gamma integral with $\kappa_n$ replaced by $\kappa_n/(1+1/\kappa_n)=\kappa_n^2/(\kappa_n+1)$ gives $t_{2\alpha_n}$ with $s^2=\beta_n(\kappa_n+1)/(\alpha_n\kappa_n)$. ✓

**Where the $\sigma^2$-scaling was used, and what breaks.** The prior $\mu\mid\sigma^2\sim N(\mu_0,\sigma^2/\kappa_0)$ contributes $u^{1/2}e^{-u\kappa_0(\mu-\mu_0)^2/2}$, whose $u$-power and exponent combine with the likelihood's into a *single* Gamma kernel. With an independent prior $\mu\sim N(\mu_0,\tau^2)$ the exponent contains $-(\mu-\mu_0)^2/2\tau^2$ with *no* factor of $u$, so $u$ and $\mu$ no longer separate: the posterior is not NIG, the marginal for $\mu$ is not a $t$, and there is no closed form. This is a good illustration that conjugacy is a property of the pair (prior, likelihood) and can be destroyed by a change that looks innocuous — and, in this instance, by a change that many would call *more* natural.

### A.5 Finishing James–Stein

*Thinking.* $\chi^2_J$ is $\mathrm{Gamma}(J/2,\ \text{rate }1/2)$; reciprocal moments of a Gamma are one-line integrals.

$$
\mathbb{E}\Big[\frac1X\Big]=\int_0^\infty\frac1x\frac{(1/2)^{J/2}}{\Gamma(J/2)}x^{J/2-1}e^{-x/2}\mathrm{d}x
=\frac{(1/2)^{J/2}}{\Gamma(J/2)}\Gamma\big(\tfrac J2-1\big)\big(\tfrac12\big)^{-(J/2-1)}
=\frac{1/2}{J/2-1}=\frac{1}{J-2},
$$
finite iff $J/2-1>0$, i.e. $J\ge3$. ✓

Now the unbiasedness. Under the hierarchical model with $\mu=0$, $s_j^2\equiv s^2$: marginally $y_j\sim N(0,s^2+\tau^2)$ i.i.d. (Theorem 10.24(i)), so $W:=\sum_jy_j^2/(s^2+\tau^2)\sim\chi^2_J$ and
$$
\mathbb{E}[\hat B]=(J-2)s^2\,\mathbb{E}\Big[\frac1{\sum_jy_j^2}\Big]=\frac{(J-2)s^2}{s^2+\tau^2}\mathbb{E}\Big[\frac1W\Big]=\frac{s^2}{s^2+\tau^2}=B. \qquad\blacksquare
$$

**Why $(1-\hat B)y$ is not unbiased for $\theta$.** Because $\hat B$ and $y$ are dependent — $\hat B$ is a function of the same $y$'s — so $\mathbb{E}[(1-\hat B)y_j]\ne(1-\mathbb{E}[\hat B])\mathbb{E}[y_j]$. Unbiasedness does not survive multiplication of dependent quantities. And of course we do not want it to: the whole point of Corollary 10.25 and Common Confusion 10.5 is that James–Stein is deliberately biased and better in MSE.

### A.6 When is the predictive not over-dispersed?

With $v(\theta)\equiv v$ constant, Proposition 10.7 gives $\mathrm{Var}(\tilde X\mid D)=v+\mathrm{Var}(\mu(\theta)\mid D)$, so equality with the plug-in variance $v$ holds **iff $\mathrm{Var}(\mu(\theta)\mid D)=0$, i.e. iff $\mu(\theta)$ is posterior-a.s. constant.**

A nontrivial example: $X\mid\theta\sim N(0,1)$ where $\theta$ indexes something the mean does not depend on — say $\theta$ is the label of a nuisance covariate's coefficient that happens to be multiplied by a zero feature, or more cleanly, $X\mid\theta\sim N(0,1)$ for every $\theta$ in a model where $\theta$ affects only a *different* observable $Y\mid\theta\sim N(\theta,1)$. Then predictions about $X$ carry no epistemic uncertainty even though $\theta$ is unknown and the posterior for $\theta$ is non-degenerate.

A subtler example worth internalizing: a two-point posterior on $\theta\in\{-1,+1\}$ with $X\mid\theta\sim N(\theta^2,1)$. Here $\mu(\theta)=\theta^2\equiv1$, so the epistemic term is zero even though the posterior is maximally uncertain about $\theta$. The lesson: epistemic uncertainty is uncertainty about the *predictive mean*, not about the parameter. Papers that equate "posterior variance of the weights" with epistemic uncertainty are making an unstated assumption of identifiability.

### A.7 Jeffreys–Lindley, both limits

The derivation is Proposition 10.29's proof; reproduce it and then take limits with $r=n\tau^2/\sigma^2$:
$$
\mathrm{BF}_{01}=\sqrt{1+r}\exp\Big(-\frac{z^2}{2}\frac{r}{1+r}\Big).
$$

**$\tau\to0$ ($r\to0$):** $\mathrm{BF}_{01}\to1$. Interpretation: $M_1$ has shrunk onto $M_0$; the two models make the same predictions, so no data can distinguish them, and the Bayes factor correctly refuses to.

**$\tau\to\infty$ ($r\to\infty$):** $\sqrt{1+r}\to\infty$ while the exponential $\to e^{-z^2/2}$, a fixed positive number, so $\mathrm{BF}_{01}\to\infty$: the null wins by an arbitrarily large factor *for any fixed data*. Interpretation: $M_1$ has spread its predictive mass over an infinite range, so it assigns essentially zero probability to any particular observed $\bar x$, and it is beaten by any model that makes a definite prediction — even a wrong one.

**Why "uninformative" is unsafe here.** The second limit shows that a diffuse prior does not converge to a neutral answer; it converges to *automatic acceptance of the null*, with the rate controlled entirely by the prior's width. So the Bayes factor for a point null against a diffuse alternative is a report about $\tau$, not about the data. Practical consequences: use priors whose scale you can defend (unit-information priors, or priors elicited on effect size), never improper priors on parameters that appear in only one of the models, and always report a sensitivity curve as in Worked Example 10.16.

### A.8 Spike-and-slab

**(i) Base measure.** $\pi=w\delta_0+(1-w)N(0,\tau^2)$ has a density with respect to $\nu=\delta_0+\lambda_1$: namely $\mathrm{d}\pi/\mathrm{d}\nu(\theta)=w\mathbf{1}\{\theta=0\}+(1-w)\phi(\theta;0,\tau^2)$. It is *not* absolutely continuous with respect to $\lambda_1$, because $\pi(\{0\})=w>0$ while $\lambda_1(\{0\})=0$. This is exactly why Proposition 10.37(ii)'s argument fails: that argument required the posterior to be dominated by $\lambda_1$, which forced singletons to be null.

**(ii) The posterior mass at zero.** The posterior of a mixture prior is the mixture of the component posteriors weighted by component evidences. Formally, for $B\ni0$,
$$
\Pi(\theta=0\mid y)=\frac{w\,p(y\mid\theta=0)}{w\,p(y\mid\theta=0)+(1-w)\int p(y\mid\theta)\phi(\theta;0,\tau^2)\mathrm{d}\theta}
=\frac{w\,\phi(y;0,s^2)}{w\,\phi(y;0,s^2)+(1-w)\,\phi(y;0,s^2+\tau^2)},
$$
where the slab's evidence is the convolution $N(0,s^2+\tau^2)$ (Proposition 10.13's predictive, or Theorem 10.24(i)). Conditional on the slab, the posterior is $N\big(\frac{\tau^2}{s^2+\tau^2}y,\ \frac{s^2\tau^2}{s^2+\tau^2}\big)$, so
$$
\mathbb{E}[\theta\mid y]=\big(1-\Pi(\theta=0\mid y)\big)\cdot\frac{\tau^2}{s^2+\tau^2}y .
$$

**(iii) Numbers** ($s=\tau=1$, so the slab shrinks by $\tfrac12$):

| $w$ | $y$ | $\Pi(\theta=0\mid y)$ | slab mean | $\mathbb{E}[\theta\mid y]$ |
|---|---|---|---|---|
| 0.5 | 0.6 | 0.563794 | 0.300000 | 0.130862 |
| 0.5 | 1.5 | 0.446227 | 0.750000 | 0.415330 |
| 0.5 | 3.0 | 0.129721 | 1.500000 | 1.305418 |
| 0.9 | 0.6 | 0.920839 | 0.300000 | 0.023748 |
| 0.9 | 1.5 | 0.878819 | 0.750000 | 0.090885 |
| 0.9 | 3.0 | 0.572926 | 1.500000 | 0.640611 |

Compare Worked Example 10.21, where the Laplace prior at $y=0.6$ gave posterior mean $0.291106$ and $\Pi(\theta=0\mid y)=0$. The spike-and-slab model can *report* a probability of exclusion — $0.564$ at $w=0.5$, $0.921$ at $w=0.9$ — which is the quantity a scientist actually wants when asking "is this coefficient zero?" The price is a non-conjugate, combinatorially structured posterior ($2^d$ inclusion patterns in $d$ dimensions), which is why $\ell_1$ remains the workhorse and spike-and-slab the reference standard.

### A.9 0-1 loss and its pathologies

**Countable case.** With $L(\theta,a)=\mathbf{1}\{a\ne\theta\}$,
$$
\rho(a\mid D)=\mathbb{E}[\mathbf{1}\{a\ne\theta\}\mid D]=1-\Pi(\theta=a\mid D),
$$
so minimizing $\rho$ is maximizing the posterior mass at $a$: the Bayes actions are exactly the posterior modes. $\blacksquare$

**A mode that is a terrible summary.** Put $\Pi(\theta=0\mid D)=0.24$ and $\Pi(\theta=k\mid D)=0.19$ for $k\in\{6,7,8,9\}$ (total $0.24+0.76=1$). Then the unique mode is $\theta=0$, yet
$$
\Pi(\theta\ge5\mid D)=0.76>0.7,\qquad \mathbb{E}[\theta\mid D]=0.19(6+7+8+9)=5.70,\qquad \text{median}=7.
$$
So the 0-1 Bayes action is $0$ while the squared-loss action is $5.70$ and the absolute-loss action is $7$. The construction generalizes: spread any total mass $1-p$ thinly over many values far from a spike of height $p$, and the mode can be made arbitrarily unrepresentative while remaining the strict maximizer.

What this shows: the mode can be an arbitrarily bad summary when the posterior is multimodal or has a spike at an isolated value, because 0-1 loss ignores *how* wrong you are. Reporting a MAP estimate without the posterior spread is therefore not a neutral choice of summary; it is a claim that near-misses are worthless.

## Part B

Reference values below were produced with NumPy 2.4.4 / SciPy 1.17.1 using `numpy.random.default_rng` with the stated seeds. If your RNG differs, your streams will differ; the *identities* (a-parts of B.1, B.2, B.4) must still hold to machine precision, and the *qualitative* conclusions must still hold.

### B.1 Sequential updating and prior sensitivity

**(a)** With prior $\mathrm{Beta}(2,2)$ and seed 2, the running $(n,s)$ pairs are $(0,0),(2,2),(5,4),(20,14),(80,53)$, giving posteriors $\mathrm{Beta}(2,2)$, $\mathrm{Beta}(4,2)$, $\mathrm{Beta}(6,3)$, $\mathrm{Beta}(16,10)$, $\mathrm{Beta}(55,31)$ and means $0.500,0.667,0.667,0.667,0.655$. These are the five curves of Figure 10.1. Look at the *widths*, not the means: the means barely move (the first flips landed near the truth) while the widths contract by roughly $\sqrt{80/2}=6.3$ from the second panel to the last.

**(b)** With $n=20$, $s=13$:

| Prior | Pseudo-count | Posterior mean | 95% CI |
|---|---|---|---|
| $\mathrm{Beta}(1,1)$ | 2 | $0.6364$ | $[0.4303,0.8189]$ |
| $\mathrm{Beta}(0.5,0.5)$ | 1 | $0.6429$ | $[0.4322,0.8277]$ |
| $\mathrm{Beta}(2,2)$ | 4 | $0.6250$ | $[0.4273,0.8029]$ |
| $\mathrm{Beta}(10,10)$ | 20 | $0.5750$ | $[0.4210,0.7219]$ |
| $\mathrm{Beta}(1,9)$ | 10 | $0.4667$ | $[0.2945,0.6431]$ |
| $\mathrm{Beta}(50,50)$ | 100 | $0.5250$ | $[0.4358,0.6134]$ |

The four priors with pseudo-count $\le20$ give posterior means spanning $0.575$–$0.643$, a range small against the posterior sd ($\approx0.097$); the two opinionated ones behave differently, with $\mathrm{Beta}(1,9)$ pushing the mean below $\tfrac12$ and $\mathrm{Beta}(50,50)$ producing an interval that excludes the MLE. Prior sensitivity is not a yes/no property but a comparison of pseudo-count to $n$.

**(c)** Three permutations (seed 10) each terminate at $\mathrm{Beta}(15,9)$, as Theorem 10.4(ii) requires — and note the check is exact in integer arithmetic, not approximate, because updating only ever adds $1$ to $a$ or to $b$. The telescoping check gives
$$
\sum_{k=1}^{20}\log p(x_k\mid D_{k-1})=-14.0190920131=\log\frac{B(15,9)}{B(2,2)},
$$
agreeing to all ten reported decimals. This is Theorem 10.4(iii) numerically, and it is the identity worth remembering: the evidence is a *prequential* score, so a model with high marginal likelihood is exactly a model that would have predicted the data well online, having seen only the past.

### B.2 Bayesian linear regression, ridge, and bands

**(a)** With $\sigma^2=0.0625$, $\tau^2=1$, so $\lambda=\sigma^2/\tau^2=0.0625$:
$$
m=(0.737955,\ 1.878058,\ -1.150220,\ 0.127966)=\hat\theta^{\text{ridge}}_{0.0625},\qquad \|m-\hat\theta^{\text{ridge}}\|_\infty=1.776\times10^{-15}.
$$
That is floating-point equality, which is the point: Corollary 10.34 is an identity, not an approximation. For comparison, OLS gives $(0.782847,2.037462,-1.249765,-0.079815)$ — note the cubic coefficient, $-0.0798$ under OLS versus $+0.1280$ under the posterior mean; the prior has pulled the least-determined coefficient hardest, which is exactly what the posterior sds ($0.127,0.294,0.312,0.527$) predict.

**(b)** Predictive variance at six test points ($\sigma^2=0.0625$ throughout):

| $x_*$ | mean | $\varphi_*^{\!\top}\Sigma\varphi_*$ | total var | 95% PI |
|---|---|---|---|---|
| $-1.00$ | $-2.418289$ | $0.216670$ | $0.279170$ | $[-3.4539,-1.3827]$ |
| $-0.50$ | $-0.504625$ | $0.014281$ | $0.076781$ | $[-1.0477,\ 0.0385]$ |
| $0.00$ | $0.737955$ | $0.016207$ | $0.078707$ | $[0.1881,1.2878]$ |
| $0.50$ | $1.405424$ | $0.015811$ | $0.078311$ | $[0.8569,1.9539]$ |
| $1.00$ | $1.593758$ | $0.055870$ | $0.118370$ | $[0.9194,2.2681]$ |
| $1.50$ | $1.398931$ | $1.462845$ | $1.525345$ | $[-1.0217,3.8196]$ |

Inside the data the epistemic term is about a quarter of the noise term; half a unit outside it is $23$ times the noise term. Plotting both bands makes the structural point visible: the parameter-only band is a narrow ribbon that flares at the edges, and the full band is that ribbon inflated by a constant $\sigma^2$. Only the first can shrink with more data.

**(c)** Maximizing Proposition 10.36's expression over $\tau^2$ gives $\tau^{2\star}=1.556526$, hence $\lambda^\star=\sigma^2/\tau^{2\star}=0.040154$ and $\log p(y)=-6.560551$ (versus $-6.757681$ at $\tau^2=1$). The posterior mean there, $(0.753958,1.924454,-1.187983,0.072747)$, again equals ridge at $\lambda^\star$ to machine precision. The identity holds at every $\lambda$, so it holds at the selected one — which is why "tune $\lambda$ by evidence maximization" and "tune $\lambda$ by cross-validation" are two ways of choosing the same estimator, differing only in the criterion.

**(d)** Degrees $0$–$7$, with $\tau^2$ optimized per degree:

| degree | $\log p(y)$ at $\tau^2=1$ | $\tau^{2\star}$ | best $\log p(y)$ | OLS residual SS |
|---|---|---|---|---|
| 0 | $-117.457587$ | $0.3380$ | $-117.249553$ | $15.032976$ |
| 1 | $-12.711501$ | $1.5861$ | $-12.589485$ | — |
| **2** | $-6.146151$ | $1.9949$ | $\mathbf{-5.729786}$ | $0.348716$ |
| 3 | $-6.757681$ | $1.5565$ | $-6.560551$ | $0.347808$ |
| 4 | $-6.867068$ | $1.3006$ | $-6.795462$ | — |
| 5 | $-7.133806$ | $1.2582$ | $-7.078423$ | — |
| 6 | $-7.297456$ | $1.1996$ | $-7.263093$ | — |
| 7 | $-7.436223$ | $1.1822$ | $-7.406955$ | $0.095746$ |

The evidence peaks at degree $2$ — the true generating degree — and then declines monotonically. Residual sum of squares does the opposite: it falls from $0.348716$ (degree 2) to $0.347808$ (degree 3) to $0.095746$ (degree 7), so it "recovers" degree 7 and would recover degree 11 if we let it. The comparison is the cleanest available demonstration of §10.5.1: RSS measures fit to the observed dataset, while the evidence measures how much probability the model assigned to that dataset *before seeing it*, and a degree-7 model must spread the same total probability over a far larger space of possible datasets. The difference between degree 2 and degree 3 is instructive too: the extra parameter buys a fit improvement of $0.0009$ in RSS and costs $0.83$ in log evidence.

### B.3 Hierarchical shrinkage

**(a)** Maximizing the marginal likelihood $\prod_j N(y_j;\mu,\tau^2+s_j^2)$ over $(\mu,\log\tau^2)$ gives $\hat\mu=0.205571$ and $\hat\tau^2=2.3730\times10^{-4}$, i.e. $\hat\tau=0.015405$. The true $\tau$ was $0.04$. Empirical Bayes has under-estimated the between-group spread by a factor of $2.6$, which is normal behaviour with $J=8$ groups and is the reason Self-check 10.4.3 warns that empirical Bayes is systematically over-confident: the $B_j$ come out too large and the posterior sds too small.

**(b)** Shrinkage factors $B_j=(0.9103,0.9419,0.8708,0.8664,0.6033,0.6112,0.2678,0.2681)$: the two $n_j=50$ groups are pulled $91\%$–$94\%$ of the way to $\hat\mu$, the $n_j=2000$ groups only $27\%$. MSEs against the true $\theta_j$:
$$
\text{no pooling } 1.31563\times10^{-3},\quad
\text{complete } 1.13336\times10^{-3},\quad
\text{partial } 6.5529\times10^{-4},
$$
a $50.19\%$ reduction over no pooling. The per-group breakdown ($\times10^{-4}$) shows how it was earned: no pooling's error is dominated by one group (the $n_j=50$ variant that observed $13/50=0.26$, squared error $79.2$), and partial pooling cuts that to $14.2$ while *increasing* error in several well-measured groups. That is shrinkage working as advertised — it trades small losses everywhere for one large rescue.

**(c)** Over 500 replications (seed 11): mean MSE $1.28347\times10^{-3}$ (no pooling), $1.84440\times10^{-3}$ (complete), $7.6901\times10^{-4}$ (partial), a $40.1\%$ reduction. Partial pooling beats no pooling in $76.0\%$ of replications and complete pooling in $92.0\%$. Two things to notice. Complete pooling is *worse* than no pooling on average, because $\tau=0.04$ is large relative to the big groups' standard errors ($0.0093$) — so the dataset in (b), where complete pooling looked good, was not representative. And partial pooling wins on average but not always: a $76\%$ win rate is what "better in expectation" looks like, so a single-dataset comparison is nearly uninformative.

**(d)** With all $n_j=2000$, the standard errors are all $\approx0.0093$, so $s_j^2\approx8.7\times10^{-5}$ against a true $\tau^2=1.6\times10^{-3}$: the shrinkage factors collapse to $\bar B\approx0.075$. Over 500 replications the mean MSEs become $7.9805\times10^{-5}$ (no pooling), $1.41623\times10^{-3}$ (complete), $7.7805\times10^{-5}$ (partial) — a $2.5\%$ reduction, with partial pooling winning only $57.8\%$ of the time. The reason is structural, not numerical: hierarchical modelling helps in proportion to how much of each group's estimate is noise. When every group is precisely measured, there is nothing to borrow and $B_j\to0$ correctly turns the machinery off. The corollary for practice is that partial pooling pays off in the *unbalanced* regime — the long tail of small groups, new users, rare segments — which is also the regime where per-group estimates are reported most confidently and least reliably.

### B.4 A marginal likelihood, two ways

**(a)** Marginally $x\sim N(\mu_0\mathbf{1},\ \sigma^2I_6+\tau^2\mathbf{1}\mathbf{1}^{\!\top})$ — the prior is a scalar $\mu$ added to every coordinate, so it induces an equicorrelated covariance. At $\tau^2=0.25$: $\log p(x)=-2.25253845$. (Posterior: mean $2.062893$, sd $0.118958$.)

**(b)** Quadrature over $\mu$ of $\prod_i\phi(x_i;\mu,\sigma^2)\phi(\mu;2,\tau^2)$ gives the same value with absolute difference $4.44\times10^{-16}$. The two routes exercise different intuitions — the analytic form says "the evidence is a density evaluated at the data", the quadrature says "the evidence is an average of likelihoods over the prior" — and disagreement between them catches sign and factor-of-two errors.

**(c)** $\tau^2\in\{0.01,0.25,1,100\}$ gives $\log p(x)=-1.152615,\ -2.252538,\ -2.917799,\ -5.210848$: monotonically decreasing. The interpretation is the Occam factor of §10.5.1. The data have $\bar x=2.0667$ and the prior is centred at exactly $2$, so a *tight* prior is a correct, confident prediction and is rewarded; every widening of the prior spreads the same unit of probability over a larger set of datasets that did not occur. The monotonicity here is an artifact of the prior being well-centred: had we set $\mu_0=1$, the tight prior would be badly wrong and the ordering would reverse over part of the range. Evidence rewards confident predictions *that come true*.

**(d)** Bayes factors are ratios of these numbers, and (c) shows a single model's evidence moving by $4.06$ in log units — a factor of $58$ — under a change of prior scale that nobody would think of as a change of model. So a Bayes factor computed with an arbitrary or improper prior scale is reporting the scale. Posterior means are different: they involve the prior through the ratio $\tau^2/(\tau^2+\sigma^2/n)$, which converges to $1$ as $\tau^2$ grows and as $n$ grows, so the posterior mean over the same sweep moves only from $2.026667$ to $2.066657$, a range of $0.04$ against a posterior sd of $0.12$. This asymmetry — estimation robust, model comparison fragile — is the single most important practical fact in §10.5, and it is why "use a vague prior" is acceptable advice for the first task and malpractice for the second.

### B.5 Laplace versus exact, and the separability trap

**(a)** With prior $N(0,4)$, the MAP is $\hat\theta=1.500369$ and the negative log-posterior Hessian is $H=1.018335$, giving Laplace sd $H^{-1/2}=0.990957$ and interval $[-0.441870,3.442608]$.

**(b)** Quadrature on $[-30,30]$: exact posterior mean $1.858382$, sd $1.057422$, percentiles $0.110258$ / $1.745797$ / $4.223451$.

The comparison is the whole exercise. The sds agree well ($0.991$ versus $1.057$, a $6\%$ error) and the intervals do not agree at all: Laplace puts its $2.5\%$ point at $-0.442$, while the exact posterior's is $+0.110$. Laplace is asserting appreciable probability that the coefficient is negative — that the relationship runs the other way — when the exact posterior essentially excludes it. What Laplace has missed is skewness, which is a third moment and therefore invisible to a method that matches only the mode and the curvature. The skew is not incidental here: with six points and a nearly separable configuration, the likelihood decays slowly to the right (large positive coefficients remain compatible with the data) and fast to the left, so the exact posterior has a long right tail, mean $1.858$ well above mode $1.500$, and a $97.5\%$ point at $4.22$ rather than $3.44$.

**(c)** Predictive at $x_*=0.75$: exact $0.774730$, MAP plug-in $0.754966$, Laplace posterior average $0.732759$ (Monte Carlo, $2\times10^6$ draws, seed 0; the probit approximation gives $0.734986$, agreeing with the Monte Carlo as it should since both integrate the same Gaussian).

The Laplace average is *further* from the exact answer than the plug-in — worse than doing nothing. Two effects, both pointing the same way. First, averaging $\sigma(0.75\theta)$ over any symmetric distribution centred at $\hat\theta$ pulls the result toward $0.5$, because $\sigma$ is concave on the positive side; this moves $0.755\to0.733$. Second, the exact posterior's mass is shifted right of the mode, which moves the exact predictive *up* to $0.775$. Laplace makes the first correction and misses the second. The general lesson: a symmetric approximation to a skewed posterior does not merely add noise to your predictions, it biases them in a predictable direction, and the bias can exceed the error of the point estimate you were trying to improve on. (The evidence is better behaved: exact $-2.989766$ versus Laplace $-3.007225$, error $-0.017$, consistent with Theorem 10.30's $O(1/n)$ claim even at $n=6$.)

**(d)** With the separable data $x=(-2,-1,1,2)$, $y=(0,0,1,1)$, the MAP is $1.7174$, $3.9453$, $7.8441$ for prior sds $2$, $10$, $100$ — diverging as the prior widens, because the likelihood alone is maximized only in the limit $\theta\to\infty$ and the prior is the sole thing holding the estimate finite.

The implication for deep learning is uncomfortable and worth stating plainly. A modern classifier that interpolates its training data is in exactly this regime: the log-likelihood has no interior maximum, so the location of the "mode" is determined by weight decay and by where the optimizer stopped, and the curvature there is whatever the regularizer supplies. Laplace error bars in that setting are error bars around an arbitrary point, scaled by a hyperparameter. They can still be useful — last-layer Laplace with a *tuned* prior precision is a respectable calibration method (§10.8.1) — but the tuning is doing the work, and the correct description is "a calibration method with a Bayesian derivation", not "the posterior".

## Part C

### C.1 The general exponential-family conjugate prior

**(a)** *Conjugacy.* The likelihood of $n$ i.i.d. observations is
$$
p(x_{1:n}\mid\eta)=\Big(\prod_ih(x_i)\Big)\exp\Big(\Big\langle\eta,\sum_iT(x_i)\Big\rangle-nA(\eta)\Big),
$$
so multiplying by $\pi_{\chi,\nu}(\eta)\propto\exp(\langle\eta,\chi\rangle-\nu A(\eta))$ and discarding the $\eta$-free factor $\prod_ih(x_i)$,
$$
\pi(\eta\mid x_{1:n})\ \propto\ \exp\Big(\Big\langle\eta,\ \chi+\sum_iT(x_i)\Big\rangle-(\nu+n)A(\eta)\Big)\ =\ \text{the }\pi_{\chi_n,\nu_n}\text{ kernel},
$$
with $\chi_n=\chi+\sum_iT(x_i)$ and $\nu_n=\nu+n$. Both sides are probability densities in $\eta$ (the left by construction, the right once normalized by $C(\chi_n,\nu_n)$, which is finite because $\nu_n>\nu$ and $\chi_n$ remains in the finiteness region — see (b)), so proportionality plus normalization gives equality. Note where the work is: the two exponents are both *affine in the pair* $(\eta,-A(\eta))$, and affine functions add. Conjugacy is that one observation.

*Evidence.* The constant that normalizes the left side is $Z(x_{1:n})/\prod_ih(x_i)$ and the constant that normalizes the right is $C(\chi_n,\nu_n)^{-1}$ against the prior's $C(\chi,\nu)^{-1}$, so
$$
Z(x_{1:n})=\Big(\prod_ih(x_i)\Big)\frac{C(\chi_n,\nu_n)}{C(\chi,\nu)},
$$
which is Theorem 10.16(iii). Every conjugate evidence formula in §10.2 — $B(a_n,b_n)/B(a_0,b_0)$ and its relatives — is this ratio.

*Pseudo-counts.* The update adds $n$ to $\nu$ and $\sum_iT(x_i)$ to $\chi$, exactly as $n$ further real observations with those sufficient statistics would. So $(\chi,\nu)$ is not merely *analogous* to data; it enters the arithmetic in the identical slot. Two priors with the same $(\chi,\nu)$ are indistinguishable, and a prior with $\nu$ comparable to your sample size is doing half the inference.

**(b)** *Boundary condition.* Differentiate the normalizer $C(\chi,\nu)^{-1}=\int_{\mathcal{N}}\exp(\langle\eta,\chi\rangle-\nu A(\eta))\mathrm{d}\eta$ with respect to $\chi$, justified by dominated convergence on compact subsets of the open set $\mathcal{N}$. This yields $\mathbb{E}[\eta]$-type identities; the identity we want, $\mathbb{E}[\nabla A(\eta)]=\chi/\nu$, comes from integrating $\nabla\cdot$ of the vector field $\eta\mapsto\exp(\langle\eta,\chi\rangle-\nu A(\eta))$ over $\mathcal{N}$ and applying the divergence theorem:
$$
0=\int_{\partial\mathcal{N}}\!\!\exp(\langle\eta,\chi\rangle-\nu A(\eta))\,n\cdot\mathrm{d}S
=\int_{\mathcal{N}}\big(\chi-\nu\nabla A(\eta)\big)\exp(\langle\eta,\chi\rangle-\nu A(\eta))\,\mathrm{d}\eta .
$$
The boundary term vanishes provided the integrand tends to $0$ at $\partial\mathcal{N}$ and at infinity — which is exactly the condition $C(\chi,\nu)<\infty$ *together with* $\chi/\nu$ lying in the interior of the convex hull of the support of $T$. When $\chi/\nu$ sits on the boundary of that hull, the prior is improper and the identity fails; the Haldane prior ($a_0=b_0=0$, so $\chi/\nu=0/0$) and the "zero successes" failure of Proposition 10.21(ii) are the canonical example. Practically: the condition you must check is that your pseudo-counts describe a *possible* dataset, strictly inside the range of achievable sufficient statistics.

**(c)** Specializations, with $\theta$ the usual parameter and $\eta$ the natural one:

| Family | $T(x)$ | $\eta$ | $A(\eta)$ | $(\chi,\nu)$ | Usual hyperparameters |
|---|---|---|---|---|---|
| Bernoulli | $x$ | $\log\frac{\theta}{1-\theta}$ | $\log(1+e^\eta)$ | $(a_0,\ a_0+b_0)$ | $\mathrm{Beta}(a_0,b_0)$ |
| Poisson | $x$ | $\log\lambda$ | $e^\eta$ | $(\alpha_0,\ \beta_0)$ | $\mathrm{Gamma}(\alpha_0,\beta_0)$ |
| $N(\mu,\sigma^2)$, $\sigma^2$ known | $x/\sigma^2$ | $\mu$ | $\mu^2/2\sigma^2$ | $(\kappa_0\mu_0/\sigma^2,\ \kappa_0)$ | $N(\mu_0,\sigma^2/\kappa_0)$ |

Check the pseudo-count reading in each: $\chi/\nu$ is $a_0/(a_0+b_0)$, the prior mean success rate over $a_0+b_0$ pretend flips; $\alpha_0/\beta_0$, the prior mean rate as $\alpha_0$ pretend events in $\beta_0$ pretend exposure units; and $\mu_0$, a pretend sample mean of $\kappa_0$ pretend observations. In all three the update $\chi_n=\chi+\sum T(x_i)$, $\nu_n=\nu+n$ is literally "add the real data to the pretend data".

**(d)** The clean statement is about the *mean parameter*, not the natural one. From (b) applied to the posterior,
$$
\mathbb{E}\big[\nabla A(\eta)\mid D\big]=\frac{\chi_n}{\nu_n}=\frac{\chi+\sum_iT(x_i)}{\nu+n}
=\frac{\nu}{\nu+n}\cdot\frac{\chi}{\nu}+\frac{n}{\nu+n}\cdot\overline{T},
$$
an exact convex combination of prior mean and sufficient-statistic average with weights $\nu:n$. No such identity holds for $\mathbb{E}[\eta\mid D]$, because $\nabla A$ is nonlinear and expectation does not commute with it. This is not pedantry. "The posterior mean of the logit" and "the logit of the posterior mean probability" differ, and by Jensen's inequality they differ in a known direction; a team that averages logits across bootstrap replicates or ensemble members and then applies $\sigma$ is computing neither the posterior mean probability nor the posterior mean logit's transform, and will find its probabilities systematically pushed away from $0.5$ relative to $\mathbb{E}[\theta\mid D]$. Report the summary that your loss asks for (Theorem 10.41), and compute it on the scale you report it.

### C.2 Jeffreys priors from first principles

**(a)** *Transformation rule.* Let $\theta=g(\phi)$ with $g$ a $C^1$ diffeomorphism and $J=\partial\theta/\partial\phi$. Assume the model is regular in the sense of Chapter 9: the score has mean zero and $I$ exists and is finite. Since the likelihood is the same function of the data under either coordinate, $\ell_\phi(\phi)=\ell_\theta(g(\phi))$, so by the chain rule the scores satisfy $\nabla_\phi\ell=J^{\!\top}\nabla_\theta\ell$. Taking the covariance of both sides,
$$
I_\phi(\phi)=\mathbb{E}\big[J^{\!\top}\nabla_\theta\ell\,\nabla_\theta\ell^{\!\top}J\big]=J^{\!\top}I_\theta(\theta)J .
$$
(Only the score identity is needed, not the second-derivative form, which is why no extra smoothness beyond $C^1$ in $\phi$ is required.) Hence $\det I_\phi=(\det J)^2\det I_\theta$ and
$$
\sqrt{\det I_\phi(\phi)}=|\det J|\sqrt{\det I_\theta(g(\phi))}. \tag{$\ast$}
$$

**(b)** *Invariance.* Two things transform, and the content of the theorem is that they cancel. First, a *measure* transforms by the change-of-variables formula (Ch. 3): if $\pi_\theta$ is a density in $\theta$, the induced density in $\phi$ is
$$
\tilde\pi_\phi(\phi)=\pi_\theta(g(\phi))\,|\det J| .
$$
Second, the *Jeffreys recipe applied in the $\phi$ coordinates* gives, by $(\ast)$,
$$
\pi_\phi^J(\phi)\propto\sqrt{\det I_\phi(\phi)}=|\det J|\sqrt{\det I_\theta(g(\phi))}\propto|\det J|\,\pi^J_\theta(g(\phi))=\tilde\pi^J_\phi(\phi).
$$
So "transform the Jeffreys prior" and "compute the Jeffreys prior in the new coordinates" agree. $\blacksquare$ The cancellation is exact because the measure contributes one factor of $|\det J|$ and $\sqrt{\det I}$ contributes exactly one as well — which is why the *square root* is not a cosmetic choice: $\det I$ would contribute $(\det J)^2$ and leave a residual $|\det J|$, and $(\det I)^{1/4}$ would leave $|\det J|^{-1/2}$.

**(c)** From A.3(c), $I(\mu,\sigma)=\mathrm{diag}(\sigma^{-2},2\sigma^{-2})$ so $\pi^J(\mu,\sigma)\propto\sqrt{2}\,\sigma^{-2}\propto\sigma^{-2}$.

Applied practice usually prefers $\pi(\mu,\sigma)\propto\sigma^{-1}$, the "independence Jeffreys" prior, obtained by computing the Jeffreys prior for $\mu$ with $\sigma$ *fixed* (giving a constant) and for $\sigma$ with $\mu$ fixed (giving $\sigma^{-1}$) and multiplying. Reasons: it yields posterior inferences for $\mu$ that coincide exactly with the classical $t$ interval with $n-1$ degrees of freedom, whereas $\sigma^{-2}$ gives $n$ degrees of freedom; and it generalizes to the regression and hierarchical settings without the pathologies that the full multivariate Jeffreys prior develops as the number of parameters grows (the "many means" problem, where full Jeffreys becomes increasingly informative about $\sigma$).

What is sacrificed is the invariance just proved. The independence-Jeffreys construction depends on the chosen factorization into $(\mu,\sigma)$ blocks, so reparameterizing to, say, $(\mu/\sigma,\log\sigma)$ and repeating the recipe gives a different prior. This is a real cost and it is the reason the "objective Bayes" literature (reference priors, in Bernardo's sense) exists: it seeks a construction that keeps invariance while respecting the parameter of interest / nuisance distinction.

**(d)** *Improper prior and improper posterior.* The one-way normal random-effects model with a flat prior on the between-group variance: $y_j\mid\theta_j\sim N(\theta_j,s^2)$, $\theta_j\mid\tau\sim N(\mu,\tau^2)$, with $\pi(\mu,\tau)\propto1$. The marginal likelihood for $\tau$ is bounded and positive as $\tau\to0$, so $\int_0^\infty$ of it against a flat prior can converge — but the closely related choice $\pi(\tau^2)\propto1/\tau^2$ (the naive "Jeffreys-like" scale prior applied to a *hierarchical* variance) gives a posterior with a non-integrable singularity at $\tau^2=0$, and the posterior is improper for every dataset. See Gelman (2006).

The practical diagnostic: an improper posterior does not announce itself, because MCMC on an improper target still produces samples — they simply drift. So (i) prove propriety analytically for your model class, or use a proper weakly informative prior (half-$t$ or half-Cauchy on $\tau$, which is the standard recommendation); (ii) run chains from widely dispersed inits for far longer than you think necessary and watch whether the posterior mean of $\log\tau$ *stabilizes* or continues to drift downward with chain length; (iii) check $\hat R$ and effective sample size for $\log\tau$ specifically, not just for the parameters you care about. Drift in a variance component's log-scale, worsening as you sample longer, is the signature.

### C.3 Bayesian linear regression, end to end

**(a)** *Posterior.* Keeping only $\theta$-dependent terms in the log-posterior,
$$
-\frac{\|y-\Phi\theta\|^2}{2\sigma^2}-\frac{\|\theta\|^2}{2\tau^2}
=-\frac12\theta^{\!\top}\underbrace{\Big(\frac{\Phi^{\!\top}\Phi}{\sigma^2}+\frac{I}{\tau^2}\Big)}_{A}\theta+\theta^{\!\top}\frac{\Phi^{\!\top}y}{\sigma^2}+\text{const},
$$
and for any symmetric $A\succ0$ and vector $c$, $-\frac12\theta^{\!\top}A\theta+\theta^{\!\top}c=-\frac12(\theta-A^{-1}c)^{\!\top}A(\theta-A^{-1}c)+\frac12c^{\!\top}A^{-1}c$ — verify by expanding the right side. With $c=\Phi^{\!\top}y/\sigma^2$ the $\theta$-dependence is a negative-definite quadratic centred at $m=A^{-1}c$, and by uniqueness of the Gaussian canonical form (Ch. 5) the posterior is $N(m,A^{-1})$.

The one point deserving emphasis is why $A=\sigma^{-2}\Phi^{\!\top}\Phi+\tau^{-2}I\succ0$: for any $u\ne0$, $u^{\!\top}Au=\sigma^{-2}\|\Phi u\|^2+\tau^{-2}\|u\|^2\ge\tau^{-2}\|u\|^2>0$. The prior term alone guarantees positive definiteness, with no rank condition on $\Phi$ — so the posterior is a well-defined Gaussian even when $n<d$ and $\Phi^{\!\top}\Phi$ is singular. That single inequality is the entire technical difference between Bayesian linear regression and OLS, and it is the reason the Bayesian version has an answer in the over-parameterized regime.

**(b)** Proposition 10.35. Given the posterior $\theta\mid y\sim N(m,\Sigma)$, $\varphi_*^{\!\top}\theta$ is a scalar affine image, so $\varphi_*^{\!\top}\theta\mid y\sim N(\varphi_*^{\!\top}m,\ \varphi_*^{\!\top}\Sigma\varphi_*)$, and adding independent noise $\varepsilon_*\sim N(0,\sigma^2)$ gives
$$
y_*\mid y\ \sim\ N\big(\varphi_*^{\!\top}m,\ \underbrace{\varphi_*^{\!\top}\Sigma\varphi_*}_{\text{epistemic}}+\underbrace{\sigma^2}_{\text{aleatoric}}\big).
$$
This is Proposition 10.7's general decomposition made concrete: $\mathrm{Var}(\mu(\theta)\mid D)=\varphi_*^{\!\top}\Sigma\varphi_*$ and $\mathbb{E}[v(\theta)\mid D]=\sigma^2$.

**(c)** Both forms. *Marginal-Gaussian form:* $y=\Phi\theta+\varepsilon$ with $\theta\sim N(0,\tau^2I)$ and $\varepsilon\sim N(0,\sigma^2I)$ independent, so $y$ is an affine image of a Gaussian and
$$
y\sim N(0,\ C),\qquad C=\tau^2\Phi\Phi^{\!\top}+\sigma^2I_n .
$$
*Fit-plus-Occam form:* $\log p(y)=-\frac{1}{2\sigma^2}\|y-\Phi m\|^2-\frac{1}{2\tau^2}\|m\|^2-\frac12\log\det A-\frac d2\log\tau^2-\frac n2\log2\pi\sigma^2$.

The two identities needed to pass between them are
$$
\det C=\sigma^{2n}\tau^{2d}\det A,\qquad y^{\!\top}C^{-1}y=\frac{\|y-\Phi m\|^2}{\sigma^2}+\frac{\|m\|^2}{\tau^2}.
$$
*First identity.* $\det C=\det(\sigma^2I_n+\tau^2\Phi\Phi^{\!\top})=\sigma^{2n}\det(I_n+\frac{\tau^2}{\sigma^2}\Phi\Phi^{\!\top})=\sigma^{2n}\det(I_d+\frac{\tau^2}{\sigma^2}\Phi^{\!\top}\Phi)$ by Sylvester's identity, and $I_d+\frac{\tau^2}{\sigma^2}\Phi^{\!\top}\Phi=\tau^2(\tau^{-2}I+\sigma^{-2}\Phi^{\!\top}\Phi)=\tau^2A$, giving $\sigma^{2n}\tau^{2d}\det A$. *Second identity.* Both sides equal $-2\log$ of the same normalized quantity: integrate $\exp(-\frac{1}{2\sigma^2}\|y-\Phi\theta\|^2-\frac{1}{2\tau^2}\|\theta\|^2)$ over $\theta$ by completing the square, which produces the right-hand side plus $\det A$ terms, and compare with the marginal density. (Numerically, at $\tau^2=1$ on the B.2 dataset: $\log\det C=-19.935558456617$ and $y^{\!\top}C^{-1}y=11.396396380522$ by both routes.)

**(d)** Corollary 10.34. The posterior mean is $m=A^{-1}\sigma^{-2}\Phi^{\!\top}y$; multiplying numerator and denominator by $\sigma^2$,
$$
m=\big(\Phi^{\!\top}\Phi+\tfrac{\sigma^2}{\tau^2}I\big)^{-1}\Phi^{\!\top}y=\hat\theta^{\text{ridge}}_\lambda,\qquad \boxed{\lambda=\frac{\sigma^2}{\tau^2}}.
$$
$\lambda$ is a *ratio of variances*, dimensionless in the sense that it is unchanged if $y$ and the prior scale are both rescaled — which is why a ridge penalty transferred between datasets with different noise levels is a different prior each time (Proposition 10.22, Worked Example 10.12).

**(e)** *What the predictive variance expresses that ridge cannot.* Ridge returns a vector; its output at $\varphi_*$ is a number. The posterior returns a distribution, and (b) shows its width at $\varphi_*$ is $\varphi_*^{\!\top}\Sigma\varphi_*+\sigma^2$, an *input-dependent* quantity that measures how well the training design constrains the model in the direction $\varphi_*$. Ridge has no object of this type: $\lambda$ is a scalar, and no function of the point estimate can distinguish "this input resembles the training data" from "this input is an extrapolation", because the estimate is the same vector in both cases. On the B.2 data the epistemic term is $0.0162$ at $x_*=0$ and $1.4628$ at $x_*=1.5$ — a factor of $90$ — while the ridge prediction is a perfectly confident-looking number at both.

A deployment decision that depends on the difference: an *abstention* or hand-off policy. Serve the model's prediction when total predictive sd is below a threshold and route to a human (or to a fallback rule) otherwise. On this dataset, a threshold of $0.5$ serves $x_*\in[-1,1]$ and abstains at $x_*=1.5$. Two more: active learning, which must query the input maximizing $\varphi_*^{\!\top}\Sigma\varphi_*$ and therefore cannot be run at all from a ridge fit; and any decision under the asymmetric losses of §10.9, whose Bayes action is a *quantile* of the predictive and so requires its spread, not merely its centre. This is the concrete answer to "why bother being Bayesian if the point estimate is identical": the point estimate is identical, and every decision that depends on more than the point estimate is not.

**(f)** With $\sigma^2$ unknown and $\sigma^2\sim\mathrm{Inv\text{-}Gamma}(\alpha_0,\beta_0)$, $\theta\mid\sigma^2\sim N(0,\sigma^2\Lambda_0^{-1})$ — note the prior covariance must scale with $\sigma^2$ for conjugacy, exactly as in A.4 — the posterior is Normal-Inverse-Gamma, the marginal posterior for $\theta$ is multivariate Student-$t$ with $2\alpha_n$ degrees of freedom, and the posterior predictive at $\varphi_*$ is univariate Student-$t$ with $2\alpha_n$ degrees of freedom, location $\varphi_*^{\!\top}m$ and squared scale $\frac{\beta_n}{\alpha_n}(1+\varphi_*^{\!\top}\Lambda_n^{-1}\varphi_*)$ (Proposition 10.14 with $\varphi_*^{\!\top}\Lambda_n^{-1}\varphi_*$ in place of $1/\kappa_n$). Practically: the bands widen, and they widen most where $n$ is small — which is the honest behaviour, since with $\sigma^2$ estimated from few residuals the noise level itself is uncertain. The plug-in habit of substituting $\hat\sigma^2$ and using Gaussian bands understates predictive intervals by the same mechanism that makes $z$-intervals too narrow at small $n$ (Worked Example 10.7: $13\%$ too narrow at $n=6$).

# Appendix B — Reference implementations

Terse but complete implementations of the five labs, so that a reader working alone can check numbers rather than guess at them. NumPy 2.4.4, SciPy 1.17.1. Every RNG is `numpy.random.default_rng` with the seed shown; if your library versions differ, the *identities* must still hold to machine precision even if simulated streams do not match.

```python
import numpy as np
from scipy import stats, special, optimize, integrate

# ---- B.1  Beta-Bernoulli: sequential updating, prior sensitivity, evidence ----
rng = np.random.default_rng(2)
x = rng.binomial(1, 0.65, 80)
for n in [0, 2, 5, 20, 80]:                      # posterior means 0.500 ... 0.655
    a, b = 2 + x[:n].sum(), 2 + n - x[:n].sum()
    print(n, a, b, a / (a + b))

n, s = 20, 13
for a0, b0 in [(1,1), (.5,.5), (2,2), (10,10), (1,9), (50,50)]:
    a, b = a0 + s, b0 + n - s
    print((a0,b0), a/(a+b), stats.beta.ppf([.025,.975], a, b))

# evidence two ways: telescoped one-step predictives vs closed form
seq = np.r_[np.ones(13), np.zeros(7)]
a, b, tot = 2.0, 2.0, 0.0
for xi in seq:
    p = a / (a + b)                              # one-step predictive
    tot += np.log(p if xi else 1 - p)
    a, b = a + xi, b + 1 - xi
print(tot, np.log(special.beta(15, 9) / special.beta(2, 2)))   # -14.0190920131

# ---- B.2  Bayesian linear regression = ridge; bands; evidence ----
rng = np.random.default_rng(10)
xs = np.sort(rng.uniform(-1, 1, 12))
ys = 1 + 2*xs - 1.5*xs**2 + rng.normal(0, .25, 12)
sig2, deg = .25**2, 3
Phi = np.vander(xs, deg + 1, increasing=True)

def posterior(Phi, y, tau2, sig2):
    A = Phi.T @ Phi / sig2 + np.eye(Phi.shape[1]) / tau2
    S = np.linalg.inv(A)
    return S @ Phi.T @ y / sig2, S, A

m, S, A = posterior(Phi, ys, 1.0, sig2)
lam = sig2 / 1.0
ridge = np.linalg.solve(Phi.T @ Phi + lam*np.eye(deg+1), Phi.T @ ys)
print(np.abs(m - ridge).max())                   # ~1e-15 (machine precision)

for xstar in [-1, -.5, 0, .5, 1, 1.5]:           # predictive bands
    f = np.vander([xstar], deg+1, increasing=True)[0]
    ep = f @ S @ f
    print(xstar, f @ m, ep, ep + sig2)

def log_evidence(Phi, y, tau2, sig2):
    n, d = Phi.shape
    m, S, A = posterior(Phi, y, tau2, sig2)
    return (-n/2*np.log(2*np.pi*sig2) - d/2*np.log(tau2)
            - .5*np.linalg.slogdet(A)[1]
            - np.sum((y - Phi @ m)**2)/(2*sig2) - m @ m/(2*tau2))

r = optimize.minimize_scalar(lambda lt: -log_evidence(Phi, ys, np.exp(lt), sig2),
                             bounds=(-8, 8), method="bounded",
                             options=dict(xatol=1e-12))
print(np.exp(r.x), sig2/np.exp(r.x), -r.fun)     # 1.5565, 0.040154, -6.560551

for d in range(8):                               # degree selection
    P = np.vander(xs, d + 1, increasing=True)
    rr = optimize.minimize_scalar(lambda lt: -log_evidence(P, ys, np.exp(lt), sig2),
                                  bounds=(-8, 8), method="bounded")
    print(d, -rr.fun)                            # peaks at d = 2

# ---- B.3  Hierarchical shrinkage / empirical Bayes ----
nj = np.array([50, 50, 100, 100, 400, 400, 2000, 2000])
def one_dataset(rng):
    th = rng.normal(.20, .04, len(nj))
    cl = rng.binomial(nj, th)
    y = cl / nj
    return th, y, np.maximum(y*(1-y)/nj, 1e-12), cl

def emp_bayes(y, s2):
    nl = lambda p: .5*np.sum(np.log(2*np.pi*(np.exp(p[1]) + s2))
                             + (y - p[0])**2/(np.exp(p[1]) + s2))
    r = optimize.minimize(nl, [y.mean(), np.log(1e-3)], method="Nelder-Mead",
                          options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    return r.x[0], np.exp(r.x[1])

th, y, s2, cl = one_dataset(np.random.default_rng(10))
mu, t2 = emp_bayes(y, s2)                        # 0.205571, 2.3730e-4
B = s2/(t2 + s2)
part = B*mu + (1 - B)*y
pool = cl.sum()/nj.sum()
mse = lambda e: np.mean((e - th)**2)
print(mu, t2, B, mse(y), mse(np.full(8, pool)), mse(part))

rng2 = np.random.default_rng(11)                 # 500 replications
out = []
for _ in range(500):
    th, y, s2, cl = one_dataset(rng2)
    mu, t2 = emp_bayes(y, s2)
    B = s2/(t2 + s2); part = B*mu + (1-B)*y; pool = cl.sum()/nj.sum()
    out.append([mse(y), mse(np.full(8, pool)), mse(part)])
out = np.array(out)
print(out.mean(0), (out[:,2] < out[:,0]).mean())  # 40.1% reduction, 76.0% wins

# ---- B.4  A marginal likelihood, two ways ----
xd = np.array([2.1, 1.8, 2.5, 2.0, 2.3, 1.7]); sg2, mu0 = .09, 2.0
for tau2 in [.01, .25, 1., 100.]:
    C = sg2*np.eye(6) + tau2*np.ones((6, 6))
    ana = stats.multivariate_normal.logpdf(xd, np.full(6, mu0), C)
    num = np.log(integrate.quad(
        lambda m: np.exp(np.sum(stats.norm.logpdf(xd, m, np.sqrt(sg2))))
                  * stats.norm.pdf(m, mu0, np.sqrt(tau2)), -20, 20)[0])
    print(tau2, ana, abs(ana - num))             # -2.25253845 at tau2 = .25

# ---- B.5  Laplace vs exact for a logistic coefficient ----
xl = np.array([-2, -1, -.5, .5, 1, 2]); yl = np.array([0, 0, 1, 1, 1, 1])
def neg_log_post(t, prior_sd=2.0):
    z = t*xl
    return -(np.sum(yl*z - np.logaddexp(0, z)) + stats.norm.logpdf(t, 0, prior_sd))

opt = optimize.minimize_scalar(neg_log_post, bounds=(-20, 20), method="bounded",
                               options=dict(xatol=1e-13))
map_ = opt.x
eps = 1e-5                                        # H by central differences
h = (neg_log_post(map_+eps) - 2*neg_log_post(map_) + neg_log_post(map_-eps))/eps**2
print(map_, h, h**-0.5)                           # 1.500369, 1.018335, 0.990957

Zq = integrate.quad(lambda t: np.exp(-neg_log_post(t)), -30, 30)[0]
mean = integrate.quad(lambda t: t*np.exp(-neg_log_post(t)), -30, 30)[0]/Zq
print(mean, np.log(Zq))                           # 1.858382, -2.989766
```

Three habits this code is meant to instil. First, compute every identity twice by different routes and compare — the `abs(ana - num)` line in B.4 and the `np.abs(m - ridge).max()` line in B.2 are the checks that catch the errors you did not anticipate. Second, put the optimizer on $\log\tau^2$, never on $\tau^2$: the parameter is positive and the log-scale is where the objective is nearly quadratic, which is the same reparameterization lesson as Worked Example 10.23. Third, report a seed with every simulated number, so that "I could not reproduce your table" becomes a statement about code rather than about luck.

### References and further reading

#### Textbooks and monographs

- **A. Gelman, J. Carlin, H. Stern, D. Dunson, A. Vehtari, D. Rubin, *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013.** The standard applied reference, and the right next stop after this chapter. Ch. 5 for hierarchical models (§10.4), Ch. 7 for WAIC and LOO, App. A for the conjugate distributions of §10.2. Its stance on priors — weakly informative by default, prior predictive checks always — is the one adopted here.
- **A. Gelman, "Prior distributions for variance parameters in hierarchical models", *Bayesian Analysis* 1(3):515–534, 2006.** Short; the source for the propriety warnings of §10.3.4 and C.2(d). Read it before putting a flat prior on a variance component.
- **C. Robert, *The Bayesian Choice*, 2nd ed., Springer, 2007.** §10.9 done properly: admissibility, minimaxity, Bayes rules versus frequentist optimality; its Ch. 2 goes deeper on Jeffreys and reference priors than §10.3.
- **J. Berger, *Statistical Decision Theory and Bayesian Analysis*, 2nd ed., Springer, 1985.** The most careful account of loss functions, Bayes risk, and prior robustness, with quantitative sensitivity bounds of the kind §10.5.2 only sketches.
- **J. Bernardo and A. Smith, *Bayesian Theory*, Wiley, 1994.** Foundations: exchangeability, coherence, and the reference-prior program C.2(c) alludes to.
- **A. van der Vaart, *Asymptotic Statistics*, Cambridge, 1998, Chapter 10.** The proof of Bernstein–von Mises (Theorem 10.9), with hypotheses stated precisely enough that you can see which ones deep networks violate.
- **C. Rasmussen and C. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.** Chs. 2 and 5 are the sequel to §10.6: Bayesian linear regression in an infinite-dimensional feature space, with evidence maximization used exactly as in Worked Example 10.20.
- **K. Murphy, *Probabilistic Machine Learning: Advanced Topics*, MIT Press, 2023.** The current single-volume ML-facing treatment; good on the parts of §10.7–§10.8 previewed here and on the algebra of Proposition 10.14.
- **D. MacKay, *Information Theory, Inference, and Learning Algorithms*, Cambridge, 2003.** Chs. 27–28: the Laplace approximation and the Occam-factor reading of the evidence, in the form that made both standard in ML.
- **G. Casella and R. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002.** For the distributional facts used without proof, including the reciprocal-$\chi^2$ moment of A.5.

#### Papers

- **R. Neal, *Bayesian Learning for Neural Networks*, PhD thesis, Toronto, 1995.** The origin of Bayesian neural networks and of HMC in this setting; still the clearest statement of what a posterior over weights is supposed to mean.
- **D. MacKay, "Bayesian interpolation", *Neural Computation* 4(3):415–447, 1992.** Evidence maximization for regularization strength; ancestor of Worked Example 10.20 and of the "type-II ML" vocabulary.
- **M. Tipping, "Sparse Bayesian learning and the relevance vector machine", *JMLR* 1:211–244, 2001.** Automatic relevance determination: per-coefficient prior variances chosen by evidence maximization, giving genuine sparsity where Proposition 10.37's Laplace prior does not.
- **R. Kass and A. Raftery, "Bayes factors", *JASA* 90(430):773–795, 1995.** The reference for §10.5: prior sensitivity, and the sense in which BIC approximates a Bayes factor.
- **A. Vehtari, A. Gelman and J. Gabry, "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC", *Statistics and Computing* 27:1413–1432, 2017**, and **A. Vehtari, D. Simpson, A. Gelman, Y. Yao and J. Gabry, "Pareto smoothed importance sampling", *JMLR* 25(72), 2024.** Definition 10.32 and the $\hat k$ diagnostic; cite the second for when PSIS-LOO can be trusted.
- **Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation", *ICML*, 2016.** The MC dropout argument of §10.8.2. Read alongside **I. Osband, "Risk versus uncertainty in deep learning", *NeurIPS Bayesian Deep Learning Workshop*, 2016**, which is the source of the criticism that dropout's predictive spread does not contract with $n$.
- **B. Lakshminarayanan, A. Pritzel and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles", *NeurIPS*, 2017.** Deep ensembles: not Bayesian, usually the strongest baseline, and the benchmark any §10.8 method must beat.
- **W. Maddox, P. Izmailov, T. Garipov, D. Vetrov and A. Wilson, "A simple baseline for Bayesian uncertainty in deep learning", *NeurIPS*, 2019.** SWAG: a Gaussian fitted to the SGD iterate trajectory.
- **E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer and P. Hennig, "Laplace Redux — effortless Bayesian deep learning", *NeurIPS*, 2021**, and **A. Kristiadi, M. Hein and P. Hennig, "Being Bayesian, even just a bit, fixes overconfidence in ReLU networks", *ICML*, 2020.** The modern case for last-layer Laplace (§10.8.1), and the result that a little weight-space uncertainty already removes ReLU networks' pathological confidence far from the data.
- **C. Guo, G. Pleiss, Y. Sun and K. Weinberger, "On calibration of modern neural networks", *ICML*, 2017.** Definition 10.39 in practice: ECE, reliability diagrams, temperature scaling — the baseline Self-check 10.8.3 asks you to apply.
- **A. Wilson and P. Izmailov, "Bayesian deep learning and a probabilistic perspective of generalization", *NeurIPS*, 2020.** Why marginalization rather than posterior fidelity is what matters in deep learning, and why ensembles are a crude approximation to it.
- **F. Wenzel, K. Roth, B. Veeling, J. Świątkowski, L. Tran, S. Mandt, J. Snoek, T. Salimans, R. Jenatton and S. Nowozin, "How good is the Bayes posterior in deep neural networks really?", *ICML*, 2020.** The cold-posterior effect of §10.8.1: tempered posteriors with $T<1$ outperforming $T=1$, and the misspecification diagnosis.
- **B. Kleijn and A. van der Vaart, "The Bernstein–von-Mises theorem under misspecification", *Electronic Journal of Statistics* 6:354–381, 2012.** The precise version of §10.1.4's warning: concentration at the KL-closest point with model-curvature width, versus the sandwich covariance that governs actual sampling fluctuation.
- **C. Stein, "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution", *Proc. Third Berkeley Symposium*, 1956**, and **W. James and C. Stein, "Estimation with quadratic loss", *Proc. Fourth Berkeley Symposium*, 1961.** The originals behind Corollary 10.25; read once, to see how surprising the result was before empirical Bayes explained it.
