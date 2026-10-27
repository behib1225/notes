# ML system design flashcards

310 active-recall cards from *System Design for Machine Learning, Deep Learning, and Agentic AI*. Same content as the study page and the Anki file.

**How to use:** read the question, answer from memory (out loud or on paper), then check the answer. Mark the cards you miss and revisit them first next time.

Card types: **Concept** (definitions and trade-offs), **Formula** (recall the expression), **Calculation** (work it out before reading the answer), **Scenario** (apply a concept to a situation).

Hardware figures needed for a calculation are given in the question; they are approximate and change with new hardware.

## Contents

- [Ch 1–2 Foundations and the design framework](#ch-12-foundations-and-the-design-framework) (8 cards)
- [Ch 3 Distributed-systems essentials](#ch-3-distributed-systems-essentials) (13 cards)
- [Ch 4 Problem framing](#ch-4-problem-framing) (10 cards)
- [Ch 5 Data and labels](#ch-5-data-and-labels) (15 cards)
- [Ch 6 Features and feature stores](#ch-6-features-and-feature-stores) (11 cards)
- [Ch 7 Modeling choices](#ch-7-modeling-choices) (16 cards)
- [Ch 8 Evaluation and experiments](#ch-8-evaluation-and-experiments) (25 cards)
- [Ch 9 Serving and deployment](#ch-9-serving-and-deployment) (20 cards)
- [Ch 10 Monitoring, drift and MLOps](#ch-10-monitoring-drift-and-mlops) (15 cards)
- [Ch 11 Distributed training](#ch-11-distributed-training) (19 cards)
- [Ch 12 LLM inference and serving](#ch-12-llm-inference-and-serving) (25 cards)
- [Ch 13 Fine-tuning, alignment and RL](#ch-13-fine-tuning-alignment-and-rl) (15 cards)
- [Ch 14 Retrieval-augmented generation](#ch-14-retrieval-augmented-generation) (23 cards)
- [Ch 15 Agentic AI](#ch-15-agentic-ai) (37 cards)
- [Ch 16 LLMOps, safety and cost](#ch-16-llmops-safety-and-cost) (12 cards)
- [Case studies](#case-studies) (32 cards)
- [Labs and capstones](#labs-and-capstones) (9 cards)
- [Interview playbook](#interview-playbook) (5 cards)

---

## Ch 1–2 Foundations and the design framework

**1. In a production ML system, which part is usually the smallest?**  
*Concept*

> The model code. Data collection, validation, feature pipelines, serving and monitoring dominate effort and failures (Sculley et al., 2015).

**2. List the 8 steps of the ML system design framework in order.**  
*Concept*

> Requirements → Framing → Data → Features → Model → Evaluation → Serving → Monitoring. Monitoring feeds back into data.

**3. Give one functional and three non-functional requirements for a feed ranker.**  
*Concept*

> Functional: rank 500 videos for the home feed. Non-functional: e.g., 50k QPS, p99 < 200 ms, new items visible within hours (also cost, privacy, fairness, safety).

**4. Why state a non-ML baseline in the framing step?**  
*Concept*

> The model must beat it, it is the fallback when the model fails, and it can ship quickly to start collecting training data.

**5. “You have two weeks and one engineer.” What do you build first?**  
*Scenario*

> Logging and the data pipeline, a heuristic baseline, and an evaluation set. Everything later depends on them, and the baseline may already deliver value.

**6. For LLM and agent systems, what replaces the “features” step?**  
*Concept*

> Context engineering: the system prompt, retrieved documents, tool definitions and memory that go into the prompt.

**7. For LLM and agent systems, what replaces the “model choice” step?**  
*Concept*

> Choosing the autonomy level (single call, workflow, agent, multi-agent) and model routing between small and large models.

**8. What is the most common way candidates fail an ML design interview?**  
*Concept*

> Jumping to the model (“I’d use a transformer”) before defining the prediction target, scale and label source.


---

## Ch 3 Distributed-systems essentials

**9. Why do interviewers ask for p99 latency instead of the average?**  
*Concept*

> In fan-out systems the page waits for the slowest backend call, so a backend’s tail latency becomes the user’s typical latency (Dean & Barroso, “The Tail at Scale”).

**10. A request fans out to 20 independent backend calls. Each exceeds its p99 with probability 1%. What fraction of requests hit at least one slow call?**  
*Calculation*

> 1 − 0.99²⁰ ≈ 18%. *With 100 calls it is ≈ 63%. Correlated slowness can make the real figure better or worse, so measure it.*

**11. What is a hedged request?**  
*Concept*

> Sending a duplicate request after a short delay and using whichever response arrives first, to cut tail latency.

**12. Three estimation shortcuts: seconds per day, QPS per 1M requests/day, peak-to-average ratio.**  
*Formula*

> 1 day ≈ 86,400 s ≈ 10⁵ s; 1M requests/day ≈ 12 QPS; peak ≈ 2–5× average.

**13. 300M requests/day. Average QPS, and peak QPS at 3× average?**  
*Calculation*

> 3×10⁸ ÷ 86,400 ≈ 3,500 QPS average (≈ 3,000 with the 10⁵ shortcut); ≈ 10,000 QPS peak.

**14. Order from fastest to slowest: datacenter round trip, DRAM reference, cross-continent round trip, NVMe SSD 4 KB random read.**  
*Concept*

> DRAM (~100 ns) < SSD random read (~20–100 µs) < datacenter round trip (~0.5 ms) < US–Europe round trip (~70–100 ms). *Memorize ratios, not digits; long-distance round trips are bounded by the speed of light in fibre.*

**15. After a back-of-envelope estimate, what should you name for the interviewer?**  
*Concept*

> The dominant factor. Ranking cost = requests × candidates × FLOPs per candidate, so the first lever is fewer candidates — the reason multi-stage ranking exists.

**16. 100M user embeddings, 128 dimensions, float32. Storage? And in float16?**  
*Calculation*

> 10⁸ × 128 × 4 B ≈ 51 GB; ≈ 26 GB in float16.

**17. Replication vs sharding.**  
*Concept*

> Replication copies data to several machines for availability and read throughput. Sharding splits data across machines when it does not fit on one.

**18. During a network partition, does ML serving usually prefer consistency or availability?**  
*Concept*

> Availability: a slightly stale feature value is better than a failed request.

**19. Why might you compute a feature in a daily batch job even though streaming is fresher?**  
*Concept*

> Streaming is costlier and harder (backfills, late events, exactly-once). Use it only for features whose value decays quickly, such as transaction velocity.

**20. What does it mean for an operation to be idempotent?**  
*Concept*

> Doing it twice has the same effect as doing it once, so clients can safely retry.

**21. What is graceful degradation in ML serving?**  
*Concept*

> When a component fails, serve a worse but valid result — e.g., cached or popularity-based results when the ranker times out.


---

## Ch 4 Problem framing

**22. Why is choosing the objective the most consequential design decision?**  
*Concept*

> The model maximizes it relentlessly, including its loopholes: optimizing clicks yields clickbait; optimizing watch time can promote compulsive use.

**23. Typical multi-signal feed score.**  
*Formula*

> score = w₁·P(click) + w₂·E[watch time] + w₃·P(like) − w₄·P(report), with weights tuned in online experiments.

**24. Pointwise vs pairwise vs listwise learning to rank.**  
*Concept*

> Pointwise scores each item independently; pairwise learns which of two items is better (RankNet); listwise optimizes the whole list’s quality (ListNet, LambdaMART).

**25. When should you not use ML?**  
*Concept*

> When a simple stable rule suffices (e.g., a legal rule), there are no labels, errors must be legally explainable, or maintenance costs exceed the value. Many systems are hybrid: rules for hard constraints, ML for the fuzzy middle.

**26. Define uplift (conditional average treatment effect).**  
*Formula*

> τ(x) = P(Y=1 | x, treated) − P(Y=1 | x, not treated): the change in outcome caused by the action.

**27. Which of the four uplift segments should receive a coupon?**  
*Concept*

> Only persuadables (buy only with the coupon). Not sure things (buy anyway), lost causes (never buy) or sleeping dogs (contact makes them leave).

**28. T-learner vs S-learner.**  
*Concept*

> T-learner: one model on treated, one on control, subtract predictions. S-learner: one model with the treatment flag as a feature, predicted with the flag on and off.

**29. Why is a randomized pilot preferred for training uplift models?**  
*Concept*

> Observational data confuses “whom we chose to treat” with “what the treatment does”; observational methods rely on the unverifiable no-unmeasured-confounders assumption.

**30. How do you evaluate an uplift model?**  
*Concept*

> Qini or uplift curves on a randomized holdout: rank by predicted uplift and measure the incremental outcome from treating the top k versus a random k.

**31. A churn model (AUC 0.85) targets the top 5% with retention offers; a randomized test shows almost no effect. Why, and what instead?**  
*Scenario*

> It ranks churn risk, not the offer’s effect: many targets are lost causes or sleeping dogs. Run a randomized pilot, train an uplift model, target highest predicted uplift.


---

## Ch 5 Data and labels

**32. Main weakness of each label source: explicit, implicit, human annotation, delayed natural, weak/programmatic.**  
*Concept*

> Explicit: sparse. Implicit: noisy and biased by what was shown. Human: expensive, labelers disagree. Delayed: training data is always old. Weak: noisy — validate against a gold set.

**33. Three mitigations for position bias in click labels.**  
*Concept*

> Use position as a training feature and set it to a constant at serving; inverse propensity weighting; randomize a small traffic slice to estimate the bias.

**34. What is selection (exposure) bias and its standard mitigation?**  
*Concept*

> You only get labels for items the current system chose to show. Reserve exploration traffic for random or uncertain items.

**35. What is a feedback loop in ML systems?**  
*Concept*

> Today’s model decides tomorrow’s training data, which can amplify popularity and bias over time.

**36. What counts as a negative example for a click model?**  
*Concept*

> An impression without a click. An item that was never shown is not a negative — the user had no chance to click it.

**37. What are hard negatives used for in retrieval training?**  
*Concept*

> Items similar to the positive but not engaged with; they teach finer distinctions than random negatives.

**38. Probability correction after keeping negatives at rate w.**  
*Formula*

> q = p / (p + (1 − p)/w), where p is the model output (He et al., 2014).

**39. Negatives were kept at 10% (w = 0.1). The model predicts p = 0.5. Calibrated probability?**  
*Calculation*

> q = 0.5 / (0.5 + 0.5/0.1) = 0.5 / 5.5 ≈ 0.091.

**40. Which metrics suit heavy class imbalance?**  
*Concept*

> PR-AUC and operating-point metrics such as recall at fixed precision — not accuracy.

**41. How should you split data for a production model that evolves over time?**  
*Concept*

> By time (train on the past, test on the future), by group (e.g., user) to measure generalization to new entities, and with a gap before the test period when labels are delayed.

**42. Define data leakage and give two examples.**  
*Concept*

> Training features that contain information unavailable at prediction time. E.g., “account closed” as a churn feature; a user’s average spend computed with future transactions.

**43. What is point-in-time correctness?**  
*Concept*

> Each training example’s features use only data available strictly before that example’s timestamp. Feature-store point-in-time joins enforce it.

**44. A churn model has 0.97 AUC offline and is no better than random online. What do you check first?**  
*Scenario*

> Leakage — a suspiciously high offline AUC is itself a warning sign. Then training–serving skew.

**45. Most common cause of silent model degradation, and the defense?**  
*Concept*

> Unannounced upstream data changes (units, casing, dropped events). Automated validation of schema, ranges, null rates and distributions.

**46. What is reject inference in credit models?**  
*Concept*

> Outcomes exist only for applicants the old policy approved, so the model cannot learn from rejected applicants (selection bias).


---

## Ch 6 Features and feature stores

**47. What does UICX stand for?**  
*Concept*

> User, Item, Context and Cross (interaction) features.

**48. How do you represent high-cardinality IDs such as user ID or zip code?**  
*Concept*

> Learned embeddings or the hashing trick.

**49. Why is the hashing trick acceptable despite collisions?**  
*Concept*

> With enough buckets frequent values rarely collide, and rare values carry little signal. You gain fixed memory, no vocabulary, and natural handling of unseen values.

**50. How do you handle heavy-tailed numeric features (counts, prices)?**  
*Concept*

> Log transform or bucketize.

**51. How should missing values be represented?**  
*Concept*

> An explicit missing indicator plus an imputed value — missingness is often informative.

**52. Batch, streaming and on-demand features: one example and freshness for each.**  
*Concept*

> Batch: 30-day average watch time (hours–1 day). Streaming: clicks in the last 5 minutes (seconds). On-demand: user–restaurant distance now (instant).

**53. Three causes of training–serving skew.**  
*Concept*

> Separate codebases computing a feature differently, different timing (end-of-day vs live values), different missing-value handling.

**54. Most reliable fix for training–serving skew.**  
*Concept*

> Log the feature values used at serving and train on those logs (“log-and-wait”), with shared feature definitions.

**55. What four things does a feature store provide?**  
*Concept*

> A shared registry of definitions, an offline store with point-in-time joins, a low-latency online store, and pipelines that materialize features online.

**56. Two roles of embeddings in ML systems.**  
*Concept*

> Input features for downstream models, and keys for nearest-neighbour retrieval.

**57. A velocity feature is exact in training (SQL) but its serving counter (stream job) lags by 2 minutes. Consequence?**  
*Scenario*

> Serving undercounts during bursts — exactly when fraud spikes — so the model under-reacts. Train on logged served values and alert on stream lag.


---

## Ch 7 Modeling choices

**58. The standard model progression in interviews.**  
*Concept*

> Heuristic → logistic regression → GBDT → deep models → foundation models, each step justified by a concrete limitation.

**59. When is GBDT preferred over a deep model?**  
*Concept*

> Mostly tabular data with engineered features, small-to-medium datasets, need for interpretability and fast iteration, no GPU serving.

**60. When do deep models beat GBDT?**  
*Concept*

> Huge data, very high-cardinality IDs needing embeddings, raw multimodal inputs, sequence features, multi-task objectives.

**61. Core idea of Wide & Deep.**  
*Concept*

> A linear “wide” part memorizes specific feature combinations; a deep part generalizes.

**62. Structure of DLRM.**  
*Concept*

> Embedding tables for sparse features, MLPs for dense features, and explicit pairwise feature interactions.

**63. Two-tower vs cross-encoder: the trade-off.**  
*Concept*

> Two-tower: user and item meet only in a dot product, so item vectors are precomputed for ANN retrieval over millions — less expressive. Cross-encoder: features interact in every layer — more accurate but affordable only for hundreds to thousands of candidates.

**64. What bias do in-batch negatives cause, and how is it corrected?**  
*Concept*

> Popular items appear as negatives more often, so they are over-penalized. The logQ correction subtracts each item’s log sampling probability from its logit (Yi et al., 2019).

**65. Loss for a p90 delivery-time prediction?**  
*Concept*

> Quantile (pinball) loss with τ = 0.9.

**66. Loss family for training retrieval embeddings?**  
*Concept*

> Softmax / sampled softmax / InfoNCE (contrastive).

**67. How does LambdaMART weight errors?**  
*Concept*

> Each pairwise gradient is scaled by |ΔNDCG| from swapping the pair, so errors near the top count more; the gradients train boosted trees.

**68. What problem does MMoE address in multi-task ranking?**  
*Concept*

> Negative transfer between conflicting tasks, by using task-specific gates over shared experts.

**69. Define calibration and name two post-hoc fixes.**  
*Concept*

> Among examples predicted 0.3, about 30% are positive. Fixes: Platt scaling, isotonic regression (also temperature scaling). Measure with reliability diagrams and ECE.

**70. Four ways to make a model cheap enough to serve.**  
*Concept*

> Distillation, quantization, pruning, and caching/precomputation (e.g., item embeddings offline).

**71. How does a new item get recommended before any interactions?**  
*Concept*

> Content features (text/image embeddings, creator features) give it a meaningful embedding immediately; reserve exploration traffic for new items.

**72. How do you handle a brand-new user?**  
*Concept*

> Use context (location, device, signup answers), popularity and quick exploration; update the user representation within the session.

**73. What is target attention (Deep Interest Network)?**  
*Concept*

> The candidate item acts as the query attending over the user’s history, so the model focuses on the history relevant to that candidate.


---

## Ch 8 Evaluation and experiments

**74. Precision and recall.**  
*Formula*

> Precision = TP / (TP + FP). Recall = TP / (TP + FN).

**75. What does ROC-AUC measure?**  
*Concept*

> The probability that a random positive is scored above a random negative.

**76. Why can ROC-AUC look good but be misleading at a 0.05% positive rate?**  
*Concept*

> It is dominated by easy negatives: a tiny false-positive rate over millions of negatives still produces far more false positives than true positives. PR-AUC exposes this.

**77. What is normalized entropy (Facebook ads)?**  
*Concept*

> Log loss divided by the entropy of the background CTR, making log loss comparable across base rates.

**78. When is MRR the right ranking metric?**  
*Concept*

> When only the first relevant result matters (e.g., question answering). MRR = mean of 1 / rank of the first relevant item.

**79. DCG@K and NDCG@K.**  
*Formula*

> DCG@K = Σᵢ (2^relᵢ − 1) / log₂(i + 1). NDCG@K = DCG@K / IDCG@K.

**80. Which items must IDCG be computed from?**  
*Concept*

> All relevant items known for the query, including those the model failed to return; otherwise low recall goes unpunished.

**81. Model ranks two items with relevance [1, 3]; these are all relevant items. NDCG@2?**  
*Calculation*

> DCG = 1/1 + 7/log₂3 ≈ 5.42. IDCG (order [3, 1]) = 7 + 1/log₂3 ≈ 7.63. NDCG ≈ 0.71.

**82. Which offline metric fits each stage: retrieval, ranking, rare events, calibration?**  
*Concept*

> Recall@K; NDCG or AUC; PR-AUC or operating-point metrics; log loss.

**83. A/B sample size per group (α = 0.05 two-sided, power 0.8).**  
*Formula*

> n ≈ 16σ² / δ². For a per-user binary metric, σ² = p(1 − p). *16 ≈ 2 × (1.96 + 0.84)².*

**84. Per-user conversion is 10%. You need to detect a +0.5 percentage-point absolute lift. Users per group?**  
*Calculation*

> σ² = 0.1 × 0.9 = 0.09; n ≈ 16 × 0.09 / 0.005² ≈ 57,600.

**85. You need 300k users per group, 2 groups, and 40k new eligible users arrive per day. How long do you run the test?**  
*Scenario*

> 600k ÷ 40k = 15 days, rounded up to full weekly cycles: 21 days. Don’t stop early on a significant peek.

**86. Why randomize an A/B test by user rather than by request?**  
*Concept*

> Each user gets one consistent experience and user-level metrics like retention are attributable. Variance must then also be computed per user.

**87. Why is p(1 − p)/impressions wrong for CTR in a user-randomized test?**  
*Concept*

> Impressions from one user are correlated, so the true variance is larger; the naive formula gives false “significant” results. Use the delta method or bootstrap over users.

**88. What is peeking and why is it harmful?**  
*Concept*

> Repeatedly checking p-values and stopping when significant; it inflates false positives. Use a fixed horizon or sequential testing.

**89. How do you handle network interference in marketplace or social A/B tests?**  
*Concept*

> Cluster randomization (by city or social cluster) or switchback experiments that alternate treatment over time windows.

**90. What is CUPED?**  
*Concept*

> Variance reduction that uses pre-experiment data as a covariate, cutting the required sample size (Deng et al., 2013).

**91. What are guardrail metrics?**  
*Concept*

> Metrics that must not degrade even if the primary metric improves, e.g., latency, crash rate, unsubscribes, revenue.

**92. Interleaving: strength and limitation.**  
*Concept*

> Merges two rankers’ results into one list for the same user; far more sensitive to ranking preference than A/B, but does not measure business metrics.

**93. How does Thompson sampling choose a variant?**  
*Concept*

> Sample each variant’s success rate from its posterior and play the highest sample, so uncertain variants still get tried.

**94. What must be logged to evaluate a new policy off-policy?**  
*Concept*

> The propensities — the probability with which the old policy showed each item — for inverse propensity scoring.

**95. Demographic parity vs equal opportunity vs equalized odds.**  
*Concept*

> Equal positive-decision rates; equal true-positive rates; equal true-positive and false-positive rates across groups.

**96. With different base rates, which two fairness criteria generally cannot both hold?**  
*Concept*

> Calibration within groups and equal error rates, except in degenerate cases such as a perfect classifier (Kleinberg et al., 2016; Chouldechova, 2017).

**97. Four reasons offline gains fail to show up online.**  
*Concept*

> Exposure bias in logged data, proxy objective ≠ true goal, training–serving skew or latency regressions, and users adapting to the new system.

**98. NDCG +4%, CTR +2%, but 7-day retention −0.3% (significant). Launch?**  
*Scenario*

> No. Investigate clickbait (dwell time, hides), diversity collapse, and latency regressions; add satisfaction signals to the objective.


---

## Ch 9 Serving and deployment

**99. Batch vs online prediction trade-off.**  
*Concept*

> Batch: near-zero-latency lookups but stale and wasted on inactive users. Online: fresh context but must fit the latency budget. Hybrid is common.

**100. Stages of the multi-stage funnel and what each optimizes.**  
*Concept*

> Retrieval (recall, ~10³–10⁴) → light ranker → heavy ranker (precision/NDCG, ~10²) → re-ranking (diversity, filters, business rules) → ~10–50 shown.

**101. Why union several candidate sources?**  
*Concept*

> Each captures a different signal (learned preference, follows, trending, fresh content); together they raise recall, add robustness and enable exploration.

**102. Maximal Marginal Relevance (MMR) selection rule.**  
*Formula*

> Greedily pick the item maximizing λ·relevance − (1 − λ)·max similarity to items already picked.

**103. HNSW in one line, with its tuning knobs.**  
*Concept*

> Multi-layer proximity graph searched greedily; excellent recall/latency, memory-hungry. Knobs: M (links per node), efSearch.

**104. IVF in one line, with its tuning knobs.**  
*Concept*

> k-means clusters vectors into lists; search only the nprobe closest lists. Knobs: nlist, nprobe.

**105. What does product quantization (PQ) do?**  
*Concept*

> Splits each vector into sub-vectors and stores the ID of each sub-vector’s nearest centroid, compressing vectors to tens of bytes.

**106. 1B vectors, 256-dim float32 vs PQ with 32 one-byte codes. Memory of each?**  
*Calculation*

> ≈ 1.02 TB raw vs ≈ 32 GB PQ (32× smaller).

**107. How do you recover recall lost to PQ compression?**  
*Concept*

> Re-score the top candidates with full-precision vectors.

**108. Why must query and index use the same embedding-model version?**  
*Concept*

> Different versions produce incompatible vector spaces; re-index when the model changes.

**109. Shadow deployment vs canary vs blue-green.**  
*Concept*

> Shadow: copy of traffic, outputs unused. Canary: gradual traffic ramp with automatic rollback. Blue-green: two full environments with instant switch and rollback.

**110. Canary release vs A/B test.**  
*Concept*

> A canary checks safety (errors, latency) during a gradual ramp. An A/B test measures causal business impact with fixed allocation and duration.

**111. Dynamic batching: what it trades.**  
*Concept*

> Waits a few milliseconds to group requests: higher GPU throughput for a small latency cost.

**112. Why prefer safetensors over pickle checkpoints?**  
*Concept*

> Pickle files can execute arbitrary code when loaded; safetensors cannot.

**113. Four serving reliability patterns.**  
*Concept*

> Timeouts with fallbacks, circuit breakers, feature defaults identical to training, multi-region deployment.

**114. Which autoscaling signals suit GPU inference?**  
*Concept*

> Queue depth, in-flight requests per replica, p95 latency or TTFT, and KV-cache utilization — not CPU utilization.

**115. Why can’t autoscaling absorb a sudden LLM traffic spike, and the mitigations?**  
*Concept*

> A new replica needs a GPU node, a multi-GB image and a weight load — minutes. Keep warm spare replicas, scale predictively before known peaks, cache weights locally, pre-pull images.

**116. What is NVIDIA MIG used for?**  
*Concept*

> Partitioning one A100/H100-class GPU into up to 7 isolated slices so small models don’t waste a whole GPU.

**117. When should a model-server readiness probe report “ready”?**  
*Concept*

> Only after weights are loaded and a warm-up request succeeds.

**118. Where are spot/preemptible GPUs appropriate?**  
*Concept*

> Interruption-tolerant work such as batch inference or checkpointed training — never as the only capacity for a latency-critical service.


---

## Ch 10 Monitoring, drift and MLOps

**119. Covariate vs label vs concept drift.**  
*Concept*

> Covariate: P(X) changes. Label: P(Y) changes. Concept: P(Y|X) changes — the same input now means something different.

**120. The seven monitoring layers, bottom to top.**  
*Concept*

> System health → data quality → feature distributions → prediction distribution → model performance → business metrics → fairness/slices.

**121. Why monitor the prediction distribution?**  
*Concept*

> It is available immediately while labels may take weeks; sudden shifts often reveal a broken feature or bad deploy first.

**122. Population Stability Index and its usual thresholds.**  
*Formula*

> PSI = Σ (aᵢ − eᵢ) ln(aᵢ / eᵢ). < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant (conventions, not laws).

**123. Two bins. Training proportions [0.5, 0.5], serving [0.7, 0.3]. PSI and verdict?**  
*Calculation*

> 0.2·ln 1.4 + (−0.2)·ln 0.6 ≈ 0.067 + 0.102 ≈ 0.17 → moderate shift.

**124. Caveat when using statistical tests for drift on millions of samples.**  
*Concept*

> Tiny, irrelevant shifts become “significant”; judge effect size and focus on features the model relies on.

**125. Three retraining strategies, and the step all of them need.**  
*Concept*

> Scheduled, triggered, continual/online — all through an automated evaluation gate before promotion.

**126. Risks of continual (online) learning.**  
*Concept*

> Instability, catastrophic forgetting, and data poisoning.

**127. How do you keep unbiased labels when the model’s actions hide outcomes?**  
*Concept*

> Keep a small randomized or un-intervened holdout.

**128. Reproducibility checklist for a model version.**  
*Concept*

> Code commit, data snapshot, feature definitions version, hyperparameters, random seeds, container image, evaluation report.

**129. Data poisoning: what it is and two defenses.**  
*Concept*

> Injecting crafted training data or labels to degrade a model or plant a backdoor. Defenses: data provenance, caps on any one source’s influence, anomaly detection on labels, a trusted evaluation set.

**130. Evasion attack and defenses.**  
*Concept*

> Small deliberate input changes that flip predictions (misspellings, mimicry). Adversarial training, features costly to fake, fast retraining.

**131. Model extraction and defenses.**  
*Concept*

> Querying an API to train a copy. Rate limits, returning labels instead of full probabilities, detecting systematic query patterns.

**132. What is a membership inference attack?**  
*Concept*

> Determining whether a specific record was in the training set. Defenses: data minimization, deduplication, differential privacy.

**133. A spam filter’s flag rate fell from 8% to 3% while precision stayed stable. Labels lag one week. Hypotheses?**  
*Scenario*

> Recall dropped because spammers adapted, a feature stopped updating, or less spam is being sent. Check feature freshness, human-review a sample of unflagged mail, watch user spam reports.


---

## Ch 11 Distributed training

**134. Bytes per parameter for mixed-precision Adam training, itemized.**  
*Formula*

> 16: BF16 weights 2 + BF16 gradients 2 + FP32 master weights 4 + Adam first moment 4 + second moment 4. Activations are extra.

**135. 13B model, mixed-precision Adam, ZeRO-3 over 16 GPUs. Model-state memory per GPU?**  
*Calculation*

> 13×10⁹ × 16 B = 208 GB total ÷ 16 = 13 GB per GPU.

**136. What does activation checkpointing cost and save?**  
*Concept*

> Recomputes discarded activations in the backward pass: about one extra forward pass (~30% more compute) for a large activation-memory saving.

**137. What is gradient accumulation?**  
*Concept*

> Running several micro-batches before each optimizer step to simulate a large batch with little memory.

**138. Training and inference compute for a dense transformer.**  
*Formula*

> Training ≈ 6·N·D FLOPs (N params, D tokens). Inference ≈ 2·N FLOPs per generated token.

**139. 13B model on 1T tokens, 512 GPUs at 400 TFLOP/s effective each. Wall-clock?**  
*Calculation*

> 6 × 1.3×10¹⁰ × 10¹² = 7.8×10²² FLOPs ÷ (512 × 4×10¹⁴) ≈ 3.8×10⁵ s ≈ 4.4 days of pure compute.

**140. What is MFU, and a typical value for large training jobs?**  
*Concept*

> Model FLOPs Utilization = achieved FLOP/s ÷ hardware peak; roughly 35–50%.

**141. Chinchilla rule and why modern models exceed it.**  
*Concept*

> ≈ 20 tokens per parameter is compute-optimal. Models are trained on far more because a smaller model trained longer is cheaper to serve.

**142. What do ZeRO-1, ZeRO-2 and ZeRO-3 shard?**  
*Concept*

> ZeRO-1: optimizer states. ZeRO-2: + gradients. ZeRO-3 / FSDP: + parameters (gathered before use, freed after).

**143. How do data-parallel GPUs synchronize, and how often?**  
*Concept*

> An all-reduce averages gradients once per step.

**144. Why is tensor parallelism usually kept within one node?**  
*Concept*

> It communicates activations in every layer, needing NVLink-class bandwidth; inter-node links are several times slower.

**145. Pipeline bubble fraction (GPipe-style).**  
*Formula*

> (p − 1) / (m + p − 1) for p stages and m micro-batches.

**146. 8 pipeline stages, 24 micro-batches. Bubble fraction?**  
*Calculation*

> 7 / 31 ≈ 23% idle.

**147. Sequence parallelism vs context parallelism.**  
*Concept*

> SP shards LayerNorm/dropout activations along the sequence within a TP group. CP splits a whole long sequence, including attention, across GPUs (e.g., ring attention).

**148. Rule of thumb for placing parallelism dimensions.**  
*Concept*

> Put the most communication-hungry parallelism on the fastest links: TP within a node, PP across a few nodes, DP/FSDP across the rest; EP for MoE experts.

**149. How are recommender models with terabytes of embedding tables trained?**  
*Concept*

> Shard tables model-parallel, replicate the dense MLP with data parallelism, exchange embeddings with all-to-all, update only touched rows. The bottleneck is memory and all-to-all bandwidth.

**150. Why is BF16 usually preferred over FP16 for training?**  
*Concept*

> BF16 has FP32’s exponent range, so gradients don’t underflow and loss scaling is usually unnecessary.

**151. Four fault-tolerance practices for very large training runs.**  
*Concept*

> Frequent sharded asynchronous checkpoints, automatic restart, health checks that evict bad nodes, data loaders that resume at the exact sample.

**152. What is MinHash used for in pretraining data pipelines?**  
*Concept*

> Estimating document similarity from compact hash signatures to find near-duplicates without comparing every pair.


---

## Ch 12 LLM inference and serving

**153. Prefill vs decode: bottleneck and latency metric of each.**  
*Concept*

> Prefill is compute-bound and sets time to first token (TTFT). Decode is memory-bandwidth-bound and sets time per output token (TPOT).

**154. Roofline argument for why decode is memory-bound.**  
*Concept*

> An H100 offers ~300 FLOPs per byte of bandwidth; batch-1 decode does ~1 FLOP per byte read, so compute sits mostly idle.

**155. Lower bound on one decode step.**  
*Formula*

> t ≈ (weight bytes + KV-cache bytes of all sequences in the batch) ÷ memory bandwidth. Real systems reach ~50–80% of this bound.

**156. 70B model in BF16, TP = 2 on GPUs with 3.35 TB/s each, batch 1. Minimum time per token?**  
*Calculation*

> Each GPU reads 70 GB: 70×10⁹ ÷ 3.35×10¹² ≈ 21 ms per token (≤ ~48 tokens/s).

**157. KV-cache bytes per token.**  
*Formula*

> 2 × layers × KV heads × head dimension × bytes per value (the 2 is for K and V).

**158. 40 layers, 8 KV heads, head dim 128, FP8 KV cache. Bytes per token?**  
*Calculation*

> 2 × 40 × 8 × 128 × 1 = 81,920 B ≈ 80 KiB.

**159. A model needs 320 KB of KV per token. Memory for one 64K-token request?**  
*Calculation*

> 65,536 × 327,680 B ≈ 21 GB.

**160. MQA vs GQA vs MLA.**  
*Concept*

> MQA: all query heads share one K/V head. GQA: groups of query heads share K/V heads. MLA: K/V compressed into a low-rank latent.

**161. Static vs continuous batching.**  
*Concept*

> Static waits for the whole batch to finish. Continuous lets sequences leave and join at every decode step, so short requests don’t wait for long ones (Orca).

**162. What is chunked prefill?**  
*Concept*

> Splitting a long prompt’s prefill into chunks interleaved with decode steps so it doesn’t stall other users’ token streams.

**163. What problem does PagedAttention solve?**  
*Concept*

> KV-cache fragmentation and over-reservation: the cache lives in fixed-size blocks like OS pages, allowing larger batches and prefix sharing.

**164. How should prompts be ordered to benefit from prefix caching?**  
*Concept*

> Stable content (instructions, tool definitions, shared documents) first; variable content last.

**165. What does FlashAttention reduce, and what doesn’t it?**  
*Concept*

> It cuts memory traffic and memory use by tiling, never writing the n×n attention matrix to HBM; it is exact. It does not remove the quadratic FLOPs.

**166. Why does weight-only INT4 speed up decode but barely prefill?**  
*Concept*

> Decode is bandwidth-bound and INT4 cuts bytes read ~4×. Prefill is compute-bound and still multiplies in higher precision; it needs low-precision compute such as FP8.

**167. Why doesn’t speculative decoding change output quality?**  
*Concept*

> The target model verifies every draft token with a rejection-sampling rule, so outputs follow the target model’s distribution exactly.

**168. When does speculative decoding help, and when not?**  
*Concept*

> At small-to-moderate batch sizes, where decode leaves compute idle. At large batches that slack disappears and gains shrink or reverse — it is a latency tool.

**169. Define goodput.**  
*Concept*

> Throughput of requests that meet the latency SLO.

**170. End-to-end generation latency.**  
*Formula*

> ≈ TTFT + TPOT × (output tokens − 1).

**171. What is the main benefit of prefill/decode disaggregation?**  
*Concept*

> Predictable latency: compute-bound prefill and bandwidth-bound decode stop interfering. Cost: KV-cache transfer between pools.

**172. Tensor vs pipeline parallelism at inference: which lowers per-token latency?**  
*Concept*

> Tensor parallelism (each GPU reads only part of the weights). Pipeline parallelism helps fit the model but doesn’t lower per-token latency.

**173. MoE: what scales with total parameters and what with active parameters?**  
*Concept*

> Memory scales with total (all experts resident); compute per token with active. At large batches, bandwidth per step approaches the full model.

**174. MoE with 1T total and 32B active parameters, FP8 weights. Weight memory, and FLOPs per token?**  
*Calculation*

> ≈ 1 TB of weights; ≈ 2 × 32×10⁹ = 64 GFLOPs per token.

**175. How do reasoning models change capacity planning?**  
*Concept*

> Thousands of thinking tokens per request make the load decode- and KV-heavy. Plan on total generated tokens; use routing and thinking budgets.

**176. Visible answers stay at 500 tokens but each request adds 4,500 thinking tokens. Decode work grows by?**  
*Calculation*

> 500 → 5,000 tokens: about 10×, slightly more because longer sequences reduce batch concurrency.

**177. Prefill of a 2,000-token prompt on a 70B model at 400 TFLOP/s effective. Time?**  
*Calculation*

> 2 × 7×10¹⁰ × 2,000 = 2.8×10¹⁴ FLOPs ÷ 4×10¹⁴ ≈ 0.7 s.


---

## Ch 13 Fine-tuning, alignment and RL

**178. The adaptation ladder, cheapest first.**  
*Concept*

> Prompting → RAG (missing knowledge) → fine-tuning (missing behaviour, or cost) → continued pretraining → pretraining from scratch.

**179. When fine-tune and when use RAG?**  
*Concept*

> RAG for knowledge that is fresh, private or must be cited. Fine-tuning for format, style, specialized behaviour, or a cheaper small model.

**180. LoRA update and trainable parameter count for a d × k matrix.**  
*Formula*

> ΔW = BA, B ∈ ℝ^(d×r), A ∈ ℝ^(r×k). Trainable params = r(d + k).

**181. LoRA with r = 8 on a 4096 × 11008 MLP matrix. Trainable parameters?**  
*Calculation*

> 8 × (4096 + 11008) = 120,832 (vs ~45M in the full matrix).

**182. What does QLoRA add to LoRA?**  
*Concept*

> Keeps the frozen base in 4-bit NF4 and trains LoRA adapters in higher precision; it fine-tuned a 65B model on one 48 GB GPU.

**183. What is multi-LoRA serving?**  
*Concept*

> One base model in GPU memory with many small adapters swapped per request — cheap per-customer customization.

**184. Order of a typical post-training pipeline.**  
*Concept*

> SFT → preference optimization (RLHF-PPO or DPO) → RL with verifiable rewards → capability, safety and regression evals.

**185. How does DPO differ from RLHF-PPO?**  
*Concept*

> DPO optimizes the policy directly on preference pairs with a classification-style loss — no separate reward model or RL loop.

**186. Why does RLHF use a KL penalty to the reference model?**  
*Concept*

> The reward model is an imperfect proxy; the penalty keeps the policy near the SFT model, limiting reward hacking and drift.

**187. How does GRPO compute advantages?**  
*Concept*

> By normalizing rewards among several attempts at the same task, so no value network is needed.

**188. Telltale sign of reward hacking during RL training.**  
*Concept*

> Training reward rises while held-out task success stays flat.

**189. Defenses against reward hacking for a coding agent rewarded by passing tests.**  
*Concept*

> Read-only or restored test files, hidden tests, checks that diffs don’t touch tests, trajectory review, and choosing checkpoints by held-out success.

**190. What usually dominates wall-clock time in RL post-training?**  
*Concept*

> Rollout generation: long traces, and each step waits for its slowest rollouts (stragglers).

**191. Synchronous vs asynchronous RL pipelines.**  
*Concept*

> Synchronous is strictly on-policy but GPUs idle in turn. Asynchronous overlaps generation and training with slightly stale weights; bound staleness and apply importance-weighting corrections.

**192. What must every fine-tuned model be checked for besides its target task?**  
*Concept*

> Regressions (catastrophic forgetting) on general and safety evaluations.


---

## Ch 14 Retrieval-augmented generation

**193. Four things RAG provides that a bare LLM doesn’t.**  
*Concept*

> Fresh knowledge (update the index, not the model), private knowledge, citations, and permission-aware access.

**194. Where do most RAG quality problems originate?**  
*Concept*

> In retrieval — and many retrieval problems are parsing and chunking problems.

**195. Chunk-size trade-off.**  
*Concept*

> Small chunks retrieve precisely but lose context; large chunks keep context but dilute embeddings and cost more tokens. Tune on your eval set.

**196. What is parent–child (small-to-big) chunking?**  
*Concept*

> Index small chunks for precise matching, but return their larger parent section to the LLM.

**197. What is contextual chunking?**  
*Concept*

> Prepending a short generated description of where the chunk sits in its document before embedding and indexing it.

**198. What do Matryoshka embeddings allow?**  
*Concept*

> Truncating vectors (e.g., 1024 → 256 dims) for cheaper storage with modest quality loss.

**199. What does changing the embedding model require?**  
*Concept*

> Re-embedding the whole corpus — plan index versioning and backfills.

**200. Dense retrieval vs BM25: what each misses.**  
*Concept*

> Dense misses exact rare terms (error codes, SKUs, names). BM25 misses paraphrases. Hybrid search uses both.

**201. Reciprocal Rank Fusion.**  
*Formula*

> RRF(d) = Σ over rankers of 1 / (k + rank(d)), k ≈ 60. It needs only ranks, no score normalization.

**202. RRF (k = 60) for a document ranked 2nd by BM25 and 5th by dense.**  
*Calculation*

> 1/62 + 1/65 ≈ 0.0161 + 0.0154 ≈ 0.0315.

**203. Typical retrieve-then-rerank setup in RAG.**  
*Concept*

> Retrieve 50–200 candidates for recall, rerank with a cross-encoder reading query and chunk together, keep the top 5–10.

**204. What is HyDE?**  
*Concept*

> The LLM writes a hypothetical answer, which is embedded and used as the search query, because answers resemble documents more than questions do.

**205. Why is conversational query rewriting essential in multi-turn RAG?**  
*Concept*

> Follow-ups like “what about contractors?” lack context; rewriting with chat history produces a standalone query.

**206. Where should the most relevant chunks go in the context, and why?**  
*Concept*

> First (or at the edges): models use the middle of long contexts less reliably (“Lost in the Middle”).

**207. Where must document permissions be enforced in RAG?**  
*Concept*

> Deterministically in the retrieval query (pre-filter by the user’s groups). Anything in the context can leak into the output.

**208. How should RAG treat instructions found inside retrieved documents?**  
*Concept*

> As untrusted data, never as instructions (indirect prompt injection).

**209. What kind of question does GraphRAG help with?**  
*Concept*

> Global “sensemaking” questions across a corpus (main themes across all reports), which chunk retrieval handles poorly.

**210. When does long context beat RAG, and when does RAG win?**  
*Concept*

> Long context (plus prompt caching) suits a small corpus queried rarely. RAG wins for large or changing corpora, permissions, and per-query cost or latency.

**211. RAG debugging: right chunk retrieved but ranked low. Fix?**  
*Concept*

> Add or tune a reranker, or pass more chunks (larger k).

**212. RAG debugging: right chunk never retrieved. Fixes?**  
*Concept*

> Parsing, chunking, embedding model, hybrid search, query rewriting.

**213. RAG debugging: right chunk in context but wrong answer. Fixes?**  
*Concept*

> Prompt, model choice, context ordering, less noise in the context.

**214. What should a RAG golden set contain?**  
*Concept*

> 200–1,000 real user questions with reference answers and the chunk IDs that should be retrieved, including unanswerable questions (correct behaviour: decline).

**215. What does a faithfulness (groundedness) score measure?**  
*Concept*

> Whether every claim in the answer is supported by the retrieved context; scored with an LLM judge or NLI model.


---

## Ch 15 Agentic AI

**216. Working definition of an agent.**  
*Concept*

> An LLM using tools in a loop, directing its own process based on feedback from the environment.

**217. First principle of agent design.**  
*Concept*

> Use the least autonomy that solves the problem: single call → workflow → agent → multi-agent, climbing only when the simpler option fails.

**218. Workflow vs agent.**  
*Concept*

> Workflow: developer-defined code paths — predictable, testable, cheaper. Agent: the LLM decides steps and tools dynamically — flexible, harder to test and bound.

**219. Seven components of an agent.**  
*Concept*

> Model, instructions, tools, context/memory, orchestration loop, environment, guardrails.

**220. Four safety features every agent loop needs.**  
*Concept*

> A step budget, unknown-tool handling, tool errors returned to the model, human approval for consequential actions.

**221. Why prefer task-level tools over thin API wrappers?**  
*Concept*

> Fewer, more capable tools mean fewer steps, and fewer steps mean fewer compounding errors.

**222. Agent–computer interface (tool design) principles besides granularity.**  
*Concept*

> Clear descriptions, strict typed schemas with enums, concise paginated outputs, actionable error messages, idempotency and least privilege.

**223. Prompt chaining.**  
*Concept*

> A fixed sequence of LLM calls, each using the previous output, with programmatic checks (gates) between steps.

**224. Routing pattern.**  
*Concept*

> Classify the input and send it to a specialized prompt, model or tool chain (e.g., easy queries to a small model).

**225. Parallelization: sectioning vs voting.**  
*Concept*

> Sectioning splits a task into independent subtasks run at once. Voting runs the same task several times and aggregates.

**226. Orchestrator–workers vs parallelization.**  
*Concept*

> In orchestrator–workers the subtasks aren’t known in advance; a central LLM decides them dynamically, delegates, and synthesizes.

**227. Evaluator–optimizer pattern.**  
*Concept*

> One LLM generates, another critiques against criteria, looping until good enough.

**228. ReAct vs plan-and-execute.**  
*Concept*

> ReAct interleaves reasoning, actions and observations step by step. Plan-and-execute writes an explicit (approvable) plan first, then executes, re-planning on failure.

**229. Most useful kind of evaluator inside an agent loop.**  
*Concept*

> An objective one: tests, compilers, schema validators, constraint checks, simulators — stronger than an LLM judging its own work.

**230. What is “context rot”?**  
*Concept*

> As the context fills with history and tool output, the model becomes slower, costlier and worse at finding what matters.

**231. Context compaction.**  
*Concept*

> Near the limit, summarize older turns — keeping decisions, open questions and key facts — and drop raw tool outputs.

**232. Just-in-time retrieval for agents.**  
*Concept*

> Keep lightweight references (paths, IDs, URLs) and load content only when needed.

**233. How do sub-agents help context management?**  
*Concept*

> A sub-task runs in a fresh, clean context and returns only a condensed result.

**234. Episodic vs semantic vs procedural memory.**  
*Concept*

> Episodic: past interactions and outcomes. Semantic: facts about the user or world. Procedural: how to do things (playbooks, skills).

**235. Five memory design questions an interviewer probes.**  
*Concept*

> What gets written, when, how it’s retrieved (similarity + recency + importance), how it’s corrected or deleted, and privacy scoping per user/tenant.

**236. When do multi-agent systems help?**  
*Concept*

> Broad, parallelizable tasks; sub-tasks needing separate clean contexts; sub-tasks needing different tools or permissions.

**237. When do multi-agent systems hurt?**  
*Concept*

> Tightly coupled tasks such as writing one coherent piece of code: agents lose shared context and make conflicting assumptions. They also multiply token cost.

**238. MCP vs A2A.**  
*Concept*

> MCP connects an agent or app to tools and data. A2A connects agents to other agents.

**239. Three primitives an MCP server exposes.**  
*Concept*

> Tools (actions), resources (readable data), prompts (reusable templates).

**240. The lethal trifecta.**  
*Concept*

> Access to private data + exposure to untrusted content + ability to communicate externally. Together they enable exfiltration; remove at least one.

**241. Direct vs indirect prompt injection.**  
*Concept*

> Direct: the user tries to override instructions. Indirect: instructions hidden in content the agent reads (web page, email, tool result).

**242. Idea behind the dual-LLM / CaMeL pattern.**  
*Concept*

> A privileged planner never sees untrusted content; a quarantined model processes it, and its outputs can’t trigger tools unchecked.

**243. Why put “refunds over $200 need approval” in the tool code, not the prompt?**  
*Concept*

> Code runs deterministically and can’t be talked around by injection or model error; it is testable and auditable.

**244. Defense-in-depth layers for agents.**  
*Concept*

> Least privilege, sandboxing with egress allowlists, human approval for irreversible actions, deterministic policy checks, input/output classifiers, audit logs, budgets.

**245. Each of 10 agent steps succeeds independently with p = 0.97. Task success?**  
*Calculation*

> 0.97¹⁰ ≈ 0.74.

**246. Common agent failure modes.**  
*Concept*

> Looping on a failing call, premature success claims, hallucinated tool arguments, losing the goal in long contexts, over-eager out-of-scope actions.

**247. What is durable execution for long-running agents?**  
*Concept*

> Persist state after each step, make steps idempotent, resume from the last checkpoint; run long tasks asynchronously with a task ID.

**248. pass@k vs pass^k for per-attempt success p.**  
*Formula*

> pass@k = 1 − (1 − p)ᵏ (at least one succeeds). pass^k = pᵏ (all succeed — consistency).

**249. p = 0.9 per attempt, k = 5. pass@5 and pass^5?**  
*Calculation*

> pass@5 = 1 − 0.1⁵ ≈ 0.99999. pass^5 = 0.9⁵ ≈ 0.59.

**250. Which metric matters for a customer-facing agent: pass@k or pass^k?**  
*Concept*

> pass^k — every customer interaction must work, so consistency is what counts.

**251. Outcome vs trajectory evaluation of agents.**  
*Concept*

> Outcome: was the end state correct (tests pass, right DB record)? Trajectory: was the path reasonable — right tools, no unsafe actions or wasted steps?

**252. What should an agent trace contain?**  
*Concept*

> A tree of spans: each LLM call (prompt, response, tokens, latency, model version) and each tool call (arguments, result, errors).


---

## Ch 16 LLMOps, safety and cost

**253. Four steps of evaluation-driven development.**  
*Concept*

> Build an eval set early (every bug becomes a test), define graders, run evals in CI on every prompt/model/retrieval change, monitor sampled production traffic.

**254. Grader preference order.**  
*Concept*

> Code-based (exact match, schema validity, unit tests) whenever possible; LLM-as-judge for open-ended quality; human review to calibrate judges.

**255. Four LLM-as-judge biases and fixes.**  
*Concept*

> Position (judge both orders), verbosity (rubric penalizes padding), self-preference (different judge family), inconsistency (low temperature, explicit rubric, reasoning before score).

**256. How do you validate an LLM judge?**  
*Concept*

> Compare it with a few hundred human labels and report agreement (e.g., Cohen’s kappa); re-validate when the judge model changes.

**257. Input vs output guardrails.**  
*Concept*

> Input: injection/jailbreak classifiers, PII redaction, topic filters, rate limits. Output: moderation, PII leakage checks, structured-output validation, groundedness and business-rule checks.

**258. Layers of hallucination mitigation.**  
*Concept*

> Grounding with citations, validated citations, permitting “I don’t know”, lower temperature, tools for arithmetic/dates/lookups, a checker pass for high stakes.

**259. Cost per LLM request.**  
*Formula*

> Σ over all calls of (input tokens × input price + output tokens × output price). Thinking tokens are usually billed as output.

**260. 3 calls per task, each 10,000 input and 500 output tokens, at $3/M input and $15/M output. Cost per task?**  
*Calculation*

> Input 30,000 × $3/M = $0.09; output 1,500 × $15/M = $0.0225; total ≈ $0.11.

**261. Biggest levers to cut LLM app cost.**  
*Concept*

> Model routing/cascades, prompt caching, response caching, trimming context, limiting output and reasoning length, batch APIs for offline work.

**262. Why is a semantic cache risky for personalized answers?**  
*Concept*

> Similar-looking queries can differ in meaning (“cancel” vs “don’t cancel”) and answers may contain another user’s data. Use only for non-personalized FAQs with high thresholds.

**263. Four reliability practices for LLM apps.**  
*Concept*

> Retries with exponential backoff and jitter, provider/model fallbacks validated by evals, pinned dated model versions, per-user quotas.

**264. What does an interviewer expect on regulation for an LLM use case?**  
*Concept*

> Recognize high-risk domains (hiring, credit, healthcare) and involve legal/compliance; add human oversight for consequential decisions.


---

## Case studies

**265. CS1 Video feed: what objective should the ranker optimize?**  
*Scenario*

> A weighted mix of P(click), E[watch time], P(like), P(share) minus P(dislike), with weights tuned online against long-term retention.

**266. CS1: why a light ranker between ~2,000 retrieved and ~500 heavy-ranked candidates?**  
*Scenario*

> Heavy-ranker cost scales with candidate count; scoring 2,000 instead of 500 costs ~4× more.

**267. CS2 Ad CTR: offline AUC improved, revenue dropped. Most likely cause?**  
*Scenario*

> Miscalibration: the auction multiplies pCTR by bids, and AUC ignores calibration. Check log loss and calibration ratio (Σ predicted ÷ Σ actual clicks ≈ 1) per segment.

**268. CS3 Fraud: expected-cost decline rule.**  
*Formula*

> Decline if P(fraud) × amount × c_fraud > (1 − P) × c_false-decline, with a middle band for step-up authentication.

**269. CS3: which label problem do declined transactions create?**  
*Scenario*

> Selection bias: declined transactions never get a fraud label. Keep a loss-capped exploration slice or a review sample.

**270. CS3: what happens if the fraud model times out on the payment path?**  
*Scenario*

> Fall back to a rules-only decision within the ~100 ms budget.

**271. CS4 Product search: why not train the ranker only on clicks?**  
*Scenario*

> Clicks are position-biased and reward catchy, cheap items. Anchor on human relevance labels and use purchase signals with bias correction.

**272. CS4: how do you handle the ambiguous query “apple”?**  
*Scenario*

> A query category classifier gives a distribution over intents; blend results, personalize by history and context, and diversify the first page.

**273. CS5 Moderation: order of the detection cascade.**  
*Scenario*

> Hash matching for known content → cheap unimodal classifiers → multimodal model on the uncertain fraction → auto-remove, human review band, or publish.

**274. CS5: how should the human review queue be prioritized?**  
*Scenario*

> By P(violation) × severity × expected reach, because harm scales with views and reviewer capacity is the binding constraint.

**275. CS5: north-star integrity metric for moderation.**  
*Concept*

> Prevalence: the fraction of views that were violating content, measured by sampling and labeling.

**276. CS6 Enterprise RAG: a user loses access to a folder. How does the assistant stop using it?**  
*Scenario*

> Permission events update ACL metadata on chunks (no re-embedding); retrieval filters by the user’s current groups; a reconciliation job catches missed events.

**277. CS6: two documents give conflicting policies. What should the answer do?**  
*Scenario*

> Prefer the most recent authoritative source, mention the conflict with both citations, and flag it to the document owners.

**278. CS7 70B chat serving: how is the batch size chosen?**  
*Scenario*

> From the per-user speed SLO and memory headroom: batch ≈ 128 keeps ≥ 25 tokens/s per user and ~25% memory free; 192 would leave only ~9% free.

**279. CS7: biggest cost lever and roughly what it saves.**  
*Scenario*

> Prefix caching with session-affinity routing: prefill compute drops from ~2,450 to ~600 GPUs’ worth.

**280. CS7: first token takes 8 s at peak. Four causes to check.**  
*Scenario*

> Queueing (capacity/autoscaling), long prompts monopolizing prefill, prefix-cache misses from routing, KV pressure causing preemption.

**281. CS8 Support agent: where does the customer’s identity come from?**  
*Scenario*

> The authenticated session, injected by the system into tools — never from the conversation or the LLM.

**282. CS8: which online metric reveals false “resolutions”?**  
*Scenario*

> Re-contact rate within 7 days.

**283. CS8: safe rollout sequence for a support agent.**  
*Scenario*

> Copilot drafts for humans → autonomous read-only intents → write actions with low limits and confirmations on a small traffic share, with a kill switch.

**284. CS9 Deep research: why is orchestrator–workers justified?**  
*Scenario*

> The task is breadth-first and parallelizable, and the material gathered exceeds one context window.

**285. CS9: subagents keep researching the same sub-topic. Fix?**  
*Scenario*

> Explicit, disjoint task specs with scope boundaries, a shared plan store checked before spawning, and source deduplication.

**286. CS10: newsvendor optimal order quantity.**  
*Formula*

> Order the demand quantile c_u / (c_u + c_o): c_u = cost per unit short, c_o = cost per unit left over.

**287. CS10: a product has $4 margin lost per unit short and $1 cost per unsold unit. Which quantile to order?**  
*Calculation*

> 4 / (4 + 1) = 0.8 → the 80th percentile of forecast demand.

**288. CS10: why can training on raw sales cause more stockouts?**  
*Scenario*

> Stockouts censor demand (zero sales while demand existed), so the model under-forecasts items that sell out. Mask or impute stockout periods.

**289. CS10: why WAPE rather than MAPE?**  
*Concept*

> MAPE is undefined when actual sales are zero and treats over- and under-forecasts asymmetrically. WAPE = Σ|y − ŷ| / Σy handles zeros and weights by volume.

**290. Pinball (quantile) loss for quantile τ.**  
*Formula*

> L = max(τ(y − ŷ), (τ − 1)(y − ŷ)): under-prediction weighted τ, over-prediction 1 − τ; minimized at the τ-quantile.

**291. CS10: correct evaluation protocol for forecasts.**  
*Concept*

> Rolling-origin backtests: train to day T, forecast ahead, move T forward, repeat across seasons. Never random-split time series.

**292. CS11 Voice agent: what makes latency the dominant constraint?**  
*Concept*

> Human turn gaps are ~200 ms; a multi-second pause feels broken. Measure voice-to-voice latency at the phone line.

**293. CS11: usually the biggest tunable piece of voice latency?**  
*Concept*

> Endpointing — deciding the caller has finished. Use semantic turn detection instead of a fixed silence timeout.

**294. CS11: the caller interrupts the agent. Which step is easy to forget?**  
*Scenario*

> Truncating conversation history to what the caller actually heard, after stopping playback.

**295. CS12 Reasoning model: does routing only hard requests to unlimited reasoning meet “cost per solved problem ≤ 2×”?**  
*Scenario*

> No: ~4× per request, so ≥ 2.8× per solved problem even at 100% accuracy. Add a small model for easy requests and a ~2,000-token thinking budget (≈ 1.9× per request).

**296. CS12: when is best-of-n sampling justified?**  
*Concept*

> Only as an escalation when an objective verifier exists (tests, answer checks), with its n× cost counted.


---

## Labs and capstones

**297. Capstone A (People You May Know): dominant candidate source and why it is computed offline.**  
*Concept*

> Friends-of-friends; ~300 × 300 = 90,000 candidates per user is too costly per request, so a batch graph job keeps the top-K.

**298. Adamic–Adar index.**  
*Formula*

> Sum over mutual friends of 1 / log(that friend’s number of friends): a mutual friend with few friends is stronger evidence.

**299. Capstone B (coding agent): why a single agent rather than multi-agent?**  
*Concept*

> Coding is tightly coupled, so shared context matters; use sub-agents only for read-only exploration.

**300. Capstone B: the only action that leaves the coding agent’s sandbox.**  
*Concept*

> Opening a draft pull request — humans review before merge; no secrets in the sandbox, egress allowlisted.

**301. Capstone C (visual search): main evaluation risk and how to address it.**  
*Concept*

> The domain gap between studio catalog photos and real user photos; evaluate recall@10 on labeled real user photos.

**302. Capstone D (bank build vs buy): decision-matrix dimensions.**  
*Concept*

> Quality on real tasks, data governance, cost at projected volume, latency, control (version pinning), engineering effort.

**303. Lab 4: why does the injection test pass even though the scripted model was fooled?**  
*Concept*

> The approval rule lives in the issue_refund tool code, and identity comes from the session — not from the prompt.

**304. Lab 4: an attacker splits one large refund into many small ones. Fix?**  
*Scenario*

> Enforce a cumulative refund limit per order or per customer in the tool layer.

**305. Lab 1: gradient of in-batch softmax cross-entropy with respect to the score matrix S = UVᵀ − log q.**  
*Formula*

> P − I (softmax probabilities minus the identity); user gradient (P − I)V, item gradient (P − I)ᵀU.


---

## Interview playbook

**306. The interviewer says “you decide” about scale. What do you do?**  
*Concept*

> State an assumption with a number and move on (“I’ll assume 100M DAU and p99 of 200 ms”).

**307. How should each major design choice be presented?**  
*Concept*

> The option you pick, one alternative, and why — tied to a constraint.

**308. Precision or recall: cancer-screening triage vs auto-removing posts?**  
*Concept*

> Screening: recall (follow-up tests catch false positives). Auto-removal: precision (send the uncertain middle to humans).

**309. Self-grading rubric: which result means you should redo the case?**  
*Concept*

> A total below 14 of 24, or any score of 1 in Data or Evaluation.

**310. How do you close an interview strongly?**  
*Concept*

> A 30-second summary, the top risks and how you’d monitor them, and next iterations.
