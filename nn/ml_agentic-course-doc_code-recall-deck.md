# System design recall deck

260 flashcards for the ML and agentic AI system design crash course, checked 9 November 2026: 179 recall, 57 scenario and 24 calculation cards across 15 modules.

## How to use this file

Cover each answer, say or write your own, then compare. Mark the cards you miss and revisit them after 1, 3 and 7 days. Cards marked **time-sensitive** state facts that change (specification versions, package names, regulations, OWASP lists); re-check them before relying on them in an interview. Each card ends with its course section and a stable ID that matches the interactive deck and the Anki CSV.

## Contents

- [Module 0: Interview framework](#module-0-interview-framework) (13 cards)
- [Module 1: Problem framing and metrics](#module-1-problem-framing-and-metrics) (15 cards)
- [Module 2: Data, features, training](#module-2-data-features-training) (14 cards)
- [Module 3: Serving](#module-3-serving) (12 cards)
- [Module 4: MLOps and experimentation](#module-4-mlops-and-experimentation) (16 cards)
- [Module 5: Classic case studies](#module-5-classic-case-studies) (15 cards)
- [Module 6: LLM fundamentals and adaptation](#module-6-llm-fundamentals-and-adaptation) (24 cards)
- [Module 7: RAG](#module-7-rag) (17 cards)
- [Module 8: Agents and tools](#module-8-agents-and-tools) (22 cards)
- [Module 9: Protocols and frameworks](#module-9-protocols-and-frameworks) (28 cards)
- [Module 10: Context, memory, state](#module-10-context-memory-state) (14 cards)
- [Module 11: Evaluation](#module-11-evaluation) (18 cards)
- [Module 12: Security and identity](#module-12-security-and-identity) (30 cards)
- [Module 13: LLMOps](#module-13-llmops) (13 cards)
- [Module 14: Agentic case studies](#module-14-agentic-case-studies) (9 cards)

## Module 0: Interview framework

**0.1 (Recall)** CRISP-DEF is this course’s own mnemonic, not an industry standard. What do its letters stand for?

> Clarify, Requirements, Inputs (data and labels), System sketch, Pick models, Deep-dive, Evaluate, Fail-safes. In an interview, walk through the steps rather than naming the acronym.

*Course section: The framework: 8 steps in 45 minutes. ID: `m0-crispdef`*

**0.2 (Recall)** In a 45-minute design round, which step gets the most time, and about how much?

> The deep-dive, about 12 minutes. It is where the interviewer probes depth and seniority.

*Course section: The framework: 8 steps in 45 minutes. ID: `m0-deepdive`*

**0.3 (Recall)** What is the most common opening mistake in a design interview?

> Drawing architecture before agreeing on the goal, users, scale and success metric.

*Course section: Common mistakes. ID: `m0-first-mistake`*

**0.4 (Scenario)** Asked to design an internal Q&A bot, a candidate opens with five cooperating agents. What is the stronger opening?

> RAG with one model call, or one agent with a search tool, measured on an eval set. Add agents only if the data shows a need: multi-agent systems add cost, latency and failure points.

*Course section: Common mistakes. ID: `m0-qa-bot`*

**0.5 (Calculation)** One million requests per day is roughly how many queries per second on average?

> About 12 (1,000,000 ÷ 86,400 ≈ 11.6). Plan capacity for peaks 3 to 10 times higher.

*Course section: Back-of-envelope numbers. ID: `m0-qps`*

**0.6 (Calculation)** Roughly how many tokens is 1,000 English words?

> About 1,300: a token is roughly 0.75 words, or about 4 characters. Tokenizers differ, so count with yours when it matters.

*Course section: Back-of-envelope numbers. ID: `m0-tokens`*

**0.7 (Calculation)** How big is one 1,536-dimension float32 embedding, and 10 million of them?

> 6 KB each (1,536 × 4 bytes), so about 60 GB for 10 million, before index overhead.

*Course section: Back-of-envelope numbers. ID: `m0-embedding`*

**0.8 (Calculation)** Each model call takes 1 s to the first token and then streams 500 tokens at 100 tokens per second. How long do 10 sequential calls take?

> About 60 s: (1 + 500 ÷ 100) × 10. Output speed varies widely by model and hardware, from tens to over a thousand tokens per second, so measure yours.

*Course section: Back-of-envelope numbers. ID: `m0-sequential`*

**0.9 (Calculation)** 2 million conversations a month, 6 turns each, 3,000 input and 300 output tokens per turn, at $3 and $15 per million tokens. Monthly model cost?

> About $162,000: 36 billion input tokens ($108,000) plus 3.6 billion output tokens ($54,000). Input dominates, so cache the stable prompt prefix first. Prices are illustrative.

*Course section: Exercises (E0.3). ID: `m0-cost`*

**0.10 (Recall)** In coding rounds that allow AI assistants, what is still being graded?

> What the assistant can’t do for you: decomposing the problem, writing specs and tests first, reviewing generated code critically, catching subtle bugs and explaining trade-offs.

*Course section: Why it matters. ID: `m0-ai-coding`*

**0.11 (Recall)** On the failure-modes part of a design interview, what separates a strong answer from a weak one?

> Raising failure modes proactively (drift, outages, abuse, prompt injection, cost blow-ups) instead of waiting to be asked.

*Course section: Interview prep: scoring rubric. ID: `m0-failures`*

**0.12 (Recall)** What does a strong answer on evaluation include?

> An eval set with graders and sound statistics, online monitoring, and an A/B test or staged rollout plan, not just “we would test it”.

*Course section: Interview prep: scoring rubric. ID: `m0-eval-answer`*

**0.13 (Recall)** How should a behavioral interview answer be structured?

> STAR: the Situation, your Task, the Actions you took, and the measurable Result.

*Course section: Interview prep: behavioral questions. ID: `m0-star`*

## Module 1: Problem framing and metrics

**1.1 (Recall)** When should you not use machine learning?

> When rules can express the logic, data is scarce, or every error is catastrophic and must be fully explainable. Always propose a non-ML baseline first.

*Course section: Core concepts. ID: `m1-no-ml`*

**1.2 (Recall)** What is the framing chain from business goal to metrics?

> Business objective → product decision → ML task → label → features → offline metric → online metric.

*Course section: Core concepts. ID: `m1-chain`*

**1.3 (Recall)** What is label leakage? Give an example.

> A feature that encodes information unavailable at prediction time, such as using “refund issued” to predict fraud, or averages computed with future transactions.

*Course section: Core concepts. ID: `m1-leakage`*

**1.4 (Recall)** Why is ROC-AUC misleading when positives are rare?

> Its false-positive rate divides by the huge number of negatives, so many false alarms still look tiny. Report PR-AUC, precision and recall at the threshold, or recall at a fixed false-positive rate.

*Course section: Metrics cheat sheet. ID: `m1-rocauc`*

**1.5 (Recall)** Which metric fits a retrieval (candidate generation) stage?

> Recall@k with a large k (for example 500). Later stages rerank, so coverage matters most.

*Course section: Metrics cheat sheet. ID: `m1-retrieval-metric`*

**1.6 (Recall)** What does NDCG reward?

> Putting highly relevant items near the top: graded relevance discounted by the logarithm of the position, normalized by the ideal ordering.

*Course section: Metrics cheat sheet. ID: `m1-ndcg`*

**1.7 (Recall)** What is a guardrail metric?

> A metric that must not get worse when the primary metric improves, such as latency, complaints, diversity, fairness or cost.

*Course section: Metrics cheat sheet. ID: `m1-guardrail`*

**1.8 (Recall)** Why do offline wins sometimes fail online?

> Offline logs contain only what the old system showed (exposure bias), and offline metrics are proxies. Confirm with an A/B test.

*Course section: Metrics cheat sheet. ID: `m1-offline-gap`*

**1.9 (Calculation)** How much downtime per month does 99.9% availability allow?

> About 43 minutes (0.1% of a 30-day month is 43.2 minutes).

*Course section: Non-functional requirements checklist. ID: `m1-availability`*

**1.10 (Recall)** How should you split data for a model that predicts the future?

> By time: train on earlier periods and test on later ones, with features computed only from data before each cutoff. Never shuffle rows randomly.

*Course section: Exercises (E1.4). ID: `m1-time-split`*

**1.11 (Scenario)** Frame “show a more accurate delivery ETA” as an ML task. What is the label, and the main offline metric?

> Regression (or quantile regression) on actual delivery time. Track MAE plus the late rate, because being late hurts more than being early.

*Course section: Exercises (E1.2). ID: `m1-eta`*

**1.12 (Recall)** When does calibration matter more than ranking quality?

> When the probability itself drives a decision or a price, such as ad auctions (bid × predicted click-through rate) or expected-cost thresholds. Measure it with log loss, Brier score or a calibration curve.

*Course section: Metrics cheat sheet. ID: `m1-calibration-when`*

**1.13 (Scenario)** Your model ranks well, but its predicted probabilities are 20% too high on average. How do you fix it?

> Fit a calibration layer, such as Platt scaling or isotonic regression, on recent held-out data, and check the correction for any negative downsampling used in training.

*Course section: Mock interview 1. ID: `m1-calibration-fix`*

**1.14 (Scenario)** For a churn model, a teammate proposes a random 80/20 split of customer-month rows. What is wrong?

> The same customer appears in many rows, so future behavior leaks into training. Split by time instead.

*Course section: Exercises (E1.4). ID: `m1-churn-split`*

**1.15 (Scenario)** A new feed ranker raises clicks 3%, but 4-week retention falls. Ship it?

> No: a guardrail regressed, which suggests it promotes clickbait. Revisit the objective, for example add watch-time or satisfaction labels.

*Course section: Exercises (E1.3). ID: `m1-clicks-retention`*

## Module 2: Data, features, training

**2.1 (Recall)** What is point-in-time correctness?

> Each training row uses only feature values that were available at the event time (minus pipeline lag), so training matches what serving will see.

*Course section: Core concepts. ID: `m2-pit`*

**2.2 (Recall)** What is training-serving skew, and how do you prevent it?

> The model sees different feature values online than in training. Use one shared feature definition, log served features and train on them, and monitor online vs offline distributions.

*Course section: Core concepts. ID: `m2-skew`*

**2.3 (Scenario)** Which kind of feature pipeline does “transactions on this card in the last 10 minutes” need?

> Streaming (for example Kafka with Flink), because the value must be minutes fresh; a daily batch job would be stale.

*Course section: Exercises (E2.2). ID: `m2-streaming`*

**2.4 (Scenario)** “The shipping address matches the billing address on this order” is what kind of feature?

> On-demand: computed at request time from the request itself.

*Course section: Exercises (E2.2). ID: `m2-on-demand`*

**2.5 (Recall)** What are a feature store’s two core jobs?

> Point-in-time correct training data from the offline store, and low-latency lookups of the latest values from the online store.

*Course section: Core concepts. ID: `m2-feature-store`*

**2.6 (Recall)** What does pandas merge_asof with direction=“backward” guarantee in a training join?

> Each event joins the latest feature row at or before its timestamp, never a later one. A tolerance can mark features that are too stale as missing.

*Course section: Code: a point-in-time correct training set. ID: `m2-merge-asof`*

**2.7 (Recall)** What does active learning do?

> Sends the examples the model is least certain about (or the most diverse) to annotators first, reaching a target accuracy with fewer labels than random sampling.

*Course section: Labeling pipelines and active learning. ID: `m2-active`*

**2.8 (Recall)** How do you use an LLM to speed up labeling safely?

> The LLM pre-labels, humans verify a sample or the uncertain cases, and you measure LLM–human agreement before trusting the labels.

*Course section: Labeling pipelines and active learning. ID: `m2-llm-labels`*

**2.9 (Recall)** How do you find ambiguous labeling guidelines?

> Measure inter-annotator agreement (for example Cohen’s kappa) and read the cases where annotators disagree.

*Course section: Labeling pipelines and active learning. ID: `m2-guidelines`*

**2.10 (Calculation)** Estimate the compute to fine-tune an 8B-parameter model on 100 million tokens.

> ≈ 6 × 8×10⁹ × 10⁸ = 4.8×10¹⁸ FLOPs. At about 4×10¹⁴ effective FLOP/s on an H100 (roughly 40% of peak), that is about 3 to 4 GPU-hours.

*Course section: Estimating training cost. ID: `m2-flops`*

**2.11 (Calculation)** How much memory does full fine-tuning of an 8B model with Adam in mixed precision need, before activations?

> About 128 GB: roughly 16 bytes per parameter for weights, gradients and optimizer states. LoRA or QLoRA cut this enough to fit on one GPU.

*Course section: Estimating training cost. ID: `m2-ft-memory`*

**2.12 (Recall)** DDP vs FSDP?

> DDP: every GPU holds a full model copy and gradients are averaged. FSDP (or DeepSpeed ZeRO): parameters, gradients and optimizer states are sharded across GPUs, for models too big for one GPU.

*Course section: Core concepts. ID: `m2-ddp-fsdp`*

**2.13 (Scenario)** A model is 15% worse in its first production week than offline. What is the first diagnostic?

> Compare served feature values with training values, feature by feature, to find skew or broken inputs. Then replay production requests through the offline pipeline and diff the predictions.

*Course section: Exercises (E2.4). ID: `m2-skew-debug`*

**2.14 (Scenario)** In a point-in-time join, the latest feature snapshot is three days older than the event. How should it be treated?

> As missing if it exceeds the staleness you allow (merge_asof tolerance), with an explicit default and a “feature missing” indicator applied the same way online.

*Course section: Exercises (E2.3). ID: `m2-stale-feature`*

## Module 3: Serving

**3.1 (Recall)** Name the four inference modes.

> Batch (precompute on a schedule), online synchronous (request and response), streaming asynchronous (score events from a queue), and hybrid (precompute heavy parts, finish online).

*Course section: Core concepts. ID: `m3-modes`*

**3.2 (Recall)** Why do recommenders and search use a multi-stage funnel?

> Scoring millions of items with a heavy model is too slow. Cheap retrieval keeps recall high on a small set; heavier models then rank only the survivors.

*Course section: Core concepts. ID: `m3-funnel`*

**3.3 (Recall)** What is the first latency optimization to try, before touching the model?

> Do less work: cache results, precompute in batch, and cut the number of candidates scored.

*Course section: Core concepts. ID: `m3-first-latency`*

**3.4 (Scenario)** 50 million new product listings a day need a policy check, and sellers don’t wait on the response. Which inference mode?

> Streaming asynchronous: a queue and an autoscaled consumer pool (about 580 listings per second on average), with uncertain cases sent to human review.

*Course section: Exercises (E3.4). ID: `m3-listings`*

**3.5 (Recall)** Shadow vs canary deployment?

> Shadow: the new model scores live traffic but its output isn’t used. Canary: a small share of real traffic (1 to 5%) gets the new model, ramped up if healthy.

*Course section: Safe deployment patterns. ID: `m3-shadow-canary`*

**3.6 (Recall)** What should a fraud check do when its model times out?

> Fall back to a conservative rule (or the previous model), not approve or block everything.

*Course section: Fallbacks. ID: `m3-fallback`*

**3.7 (Recall)** Why can’t a request timeout alone protect a Python model server?

> A timed-out thread keeps running because Python can’t kill threads. Bound in-flight predictions with a semaphore released when the work really finishes, and shed extra load to a fallback.

*Course section: Code: a production-shaped prediction service. ID: `m3-thread`*

**3.8 (Recall)** How should you route users to a canary model?

> By a stable hash of a user ID (for example SHA-256). Not random(), and not Python’s built-in hash() of a string, which is salted per process.

*Course section: Exercises (E3.3). ID: `m3-canary-hash`*

**3.9 (Recall)** Why monitor p95 and p99 latency rather than the mean?

> Tail latency is what users feel and what breaks timeouts; the mean hides slow requests.

*Course section: Core concepts. ID: `m3-tail`*

**3.10 (Recall)** What does quantization trade?

> Lower-precision weights (and sometimes KV cache), such as FP8, INT8, INT4 or FP4, for less memory and higher throughput, at a small quality cost you must verify on your eval set.

*Course section: Core concepts. ID: `m3-quantization`*

**3.11 (Recall)** What is dynamic batching in a model server?

> The server groups concurrent requests into one GPU batch for a few milliseconds, raising throughput at a small latency cost.

*Course section: Core concepts. ID: `m3-dynamic-batching`*

**3.12 (Scenario)** Ranking takes 180 ms at p95 for 1,000 candidates, and the budget is 60 ms. First move?

> Cut the candidate count (for example to 300 through better retrieval), since ranking cost is roughly linear in candidates. Then distill, quantize or compile the ranker.

*Course section: Exercises (E3.2). ID: `m3-ranker-budget`*

## Module 4: MLOps and experimentation

**4.1 (Scenario)** An upstream feed starts sending nulls. Which monitoring layer should catch it first?

> The data layer: schema, null-rate and distribution checks fire before model or business metrics move.

*Course section: Four monitoring layers. ID: `m4-null-feed`*

**4.2 (Scenario)** A marketing campaign brings many first-time users from a new country. Which type of drift is this?

> Covariate (data) drift: P(x) changes. It can be detected without labels.

*Course section: Types of drift. ID: `m4-covariate`*

**4.3 (Scenario)** A coordinated attack raises the fraud rate from 0.2% to 1%. Which type of drift is this?

> Prior (label) drift: P(y) changes.

*Course section: Types of drift. ID: `m4-prior`*

**4.4 (Scenario)** Fraudsters switch to aged, legitimate-looking accounts, so old signals stop working. Which type of drift is this?

> Concept drift: P(y given x) changes, so the same features now mean something else. Detecting it usually needs labels.

*Course section: Types of drift. ID: `m4-concept`*

**4.5 (Recall)** What are the usual PSI thresholds?

> Below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant drift.

*Course section: Drift statistics. ID: `m4-psi`*

**4.6 (Recall)** Why can a KS test flag drift that doesn’t matter?

> With large samples, tiny irrelevant shifts become statistically significant. Alert on effect size and on features the model relies on.

*Course section: Drift statistics. ID: `m4-ks`*

**4.7 (Recall)** How do you monitor a model whose labels arrive months later?

> Use proxies: prediction and approval-rate drift, early outcomes (for example 30-day delinquency), review decisions and human-labeled samples, tracked by cohort.

*Course section: Label delay and proxies. ID: `m4-label-delay`*

**4.8 (Recall)** How do you limit a model’s feedback loop on its own training data?

> Reserve a small exploration slice, log propensities for counterfactual evaluation, and get labels for some blocked or unshown cases through review.

*Course section: Feedback loops. ID: `m4-feedback`*

**4.9 (Calculation)** If you halve the minimum detectable effect, what happens to an A/B test’s sample size?

> It roughly quadruples, because sample size scales with 1/δ².

*Course section: A/B testing essentials. ID: `m4-mde`*

**4.10 (Calculation)** Baseline conversion is 4%, and you want to detect +0.4 points (absolute) at α = 0.05 and 80% power. About how many users per arm?

> About 37,700: n ≈ 2 × (1.96 + 0.84)² × 0.04 × 0.96 ÷ 0.004².

*Course section: Code: PSI drift check and A/B sample size. ID: `m4-sample-size`*

**4.11 (Recall)** What is sample ratio mismatch?

> The observed traffic split differs from the planned one (for example 52/48 instead of 50/50). It signals a bug, so don’t trust the results.

*Course section: A/B testing essentials. ID: `m4-srm`*

**4.12 (Recall)** What is a novelty effect, and how do you detect it?

> An early lift that fades as users get used to a change. Check the lift by days since each user entered the experiment.

*Course section: A/B testing essentials. ID: `m4-novelty`*

**4.13 (Recall)** What does CUPED do?

> Uses each user’s pre-experiment behavior as a covariate to reduce variance, so experiments need fewer users.

*Course section: A/B testing essentials. ID: `m4-cuped`*

**4.14 (Recall)** Why is interleaving useful for ranking tests?

> It mixes results from both rankers into one list and sees which gets the clicks, detecting preferences with far fewer users than an A/B test.

*Course section: A/B testing essentials. ID: `m4-interleaving`*

**4.15 (Calculation)** You need 56,500 users per arm and get 20,000 eligible users a day, split 50/50. How long should the test run?

> About 6 days of traffic (113,000 users). Run 7 to cover a full weekly cycle.

*Course section: Exercises (E4.2). ID: `m4-ab-days`*

**4.16 (Scenario)** A fraud model’s most important feature shows PSI 0.31 this week. What next?

> Significant drift: first rule out an upstream data bug, then segment the shift and check score and approval-rate drift. Retrain or recalibrate only if the change is real.

*Course section: Drift statistics. ID: `m4-psi-alert`*

## Module 5: Classic case studies

**5.1 (Recall)** What is a two-tower model?

> Separate user and item encoders trained so their dot product predicts engagement. Item vectors are precomputed and searched with approximate nearest neighbor.

*Course section: Case study A: recommendations. ID: `m5-two-tower`*

**5.2 (Recall)** What is position bias, and how is it handled?

> Higher positions get more clicks regardless of relevance. Train with position as a feature fixed to one value at serving, or use inverse-propensity weighting.

*Course section: Case study A: recommendations. ID: `m5-position-bias`*

**5.3 (Scenario)** How do you recommend a video uploaded five minutes ago, with no interaction history?

> Use content embeddings (text, audio, visual) so the item tower can place it, plus a reserved exploration slot to collect early engagement.

*Course section: Case study A: recommendations. ID: `m5-cold-start`*

**5.4 (Recall)** What is the strong standard model for search learning-to-rank on tabular features?

> LambdaMART: gradient-boosted trees, for example in LightGBM or XGBoost.

*Course section: Case study B: search ranking. ID: `m5-lambdamart`*

**5.5 (Recall)** Why does fraud detection use both rules and a model?

> Rules handle known patterns and can be updated instantly by analysts; gradient-boosted trees score everything else quickly on heterogeneous tabular features.

*Course section: Case study C: fraud detection. ID: `m5-rules-model`*

**5.6 (Calculation)** A missed fraud costs the amount A and a false decline costs C. Above what fraud probability should you decline? Compute it for A = $40 and C = $15.

> Decline when p × A > (1 − p) × C, so p > C ÷ (A + C) = 15 ÷ 55 ≈ 0.27. For a $400 transaction it falls to about 0.036.

*Course section: Exercises (E5.2). ID: `m5-threshold`*

**5.7 (Recall)** Why are fraud labels biased?

> You never see outcomes for declined transactions (selection bias). Use a small randomized holdout or review-based labels.

*Course section: Case study C: fraud detection. ID: `m5-selection-bias`*

**5.8 (Recall)** Why is training a demand forecast on raw sales dangerous?

> Sales are censored demand: on stockout days they understate demand, which teaches the model to under-forecast exactly the products that sell out.

*Course section: Case study D: demand forecasting. ID: `m5-censored`*

**5.9 (Recall)** Why use WAPE instead of MAPE in retail forecasting?

> MAPE explodes on low-volume series. WAPE (total absolute error divided by total sales) is robust.

*Course section: Case study D: demand forecasting. ID: `m5-wape`*

**5.10 (Recall)** How do you validate a forecasting model?

> Rolling-origin backtesting: train up to date T, forecast the next horizon, slide T forward and repeat across several periods, including holidays.

*Course section: Case study D: demand forecasting. ID: `m5-backtest`*

**5.11 (Recall)** What is prevalence in content moderation?

> The share of content views that are violating, measured by labeling a random sample. It reflects what users actually experience.

*Course section: Case study E: content moderation. ID: `m5-prevalence`*

**5.12 (Scenario)** Where in a content-moderation cascade should an expensive vision-language model sit?

> After hash matching and cheap classifiers, applied only to the uncertain or high-severity few percent; humans review what remains uncertain.

*Course section: Case study E: content moderation. ID: `m5-vlm-placement`*

**5.13 (Calculation)** 3% of 20 million daily posts go to a vision-language model, at 1,500 input and 100 output tokens each, priced $1 and $5 per million. Daily cost?

> About $1,200: 600,000 calls; $900 for input plus $300 for output. Prices are illustrative.

*Course section: Exercises (E5.6). ID: `m5-vlm-cost`*

**5.14 (Scenario)** Click logs favor top-ranked results. How do you build a less biased offline evaluation for search?

> Label a stratified sample of queries (head, torso, tail) with graded relevance from humans, or from LLM raters calibrated against humans, and compute NDCG@10 per segment.

*Course section: Exercises (E5.3). ID: `m5-search-eval`*

**5.15 (Scenario)** A new advertiser has no click history. How do you predict its click-through rate?

> Back off to features that generalize (category, creative text and image embeddings, landing-page quality), use priors from similar advertisers, and reserve some exploration traffic.

*Course section: Mock interview 1. ID: `m5-new-advertiser`*

## Module 6: LLM fundamentals and adaptation

**6.1 (Recall)** What do prefill and decode determine?

> Prefill processes the input in parallel and sets time to first token (TTFT). Decode generates tokens one at a time and sets time per output token (TPOT).

*Course section: Core concepts. ID: `m6-prefill-decode`*

**6.2 (Recall)** What is the KV cache memory formula?

> 2 × layers × KV heads × head dimension × bytes per value, per token. An 8B model with 32 layers, 8 KV heads and head dimension 128 at 16-bit uses about 128 KB per token.

*Course section: Core concepts: KV cache. ID: `m6-kv-formula`*

**6.3 (Calculation)** KV cache for one 32,000-token request on a 70B-class model with 80 layers, 8 KV heads, head dimension 128, FP16?

> About 10.5 GB: 2 × 80 × 8 × 128 × 2 bytes ≈ 320 KB per token, times 32,000 tokens.

*Course section: Exercises (E6.2). ID: `m6-kv-70b`*

**6.4 (Calculation)** How much GPU memory do the weights of a 70B-parameter model need at FP16, FP8 and 4-bit?

> About 140 GB, 70 GB and 35 GB (2, 1 and 0.5 bytes per parameter), before KV cache and runtime overhead.

*Course section: Mock interview 3. ID: `m6-weights`*

**6.5 (Recall)** When does the standard KV cache formula overestimate memory?

> For models with multi-head latent attention, sliding-window layers, or hybrid linear-attention or state-space layers. Compute it from the model’s actual config.

*Course section: Core concepts: KV cache. ID: `m6-kv-caveat`*

**6.6 (Recall)** What does continuous batching do?

> Adds and removes requests from the running batch at every decode step, which greatly increases GPU throughput.

*Course section: Serving engine techniques. ID: `m6-continuous-batching`*

**6.7 (Recall)** What problem does PagedAttention solve?

> KV cache fragmentation: the cache is stored in fixed-size blocks, like operating-system memory pages, so more requests fit on a GPU.

*Course section: Serving engine techniques. ID: `m6-paged`*

**6.8 (Recall)** What is speculative decoding?

> A small draft model proposes several tokens that the large model verifies in one pass, lowering TPOT. Standard verification keeps the large model’s output distribution.

*Course section: Serving engine techniques. ID: `m6-speculative`*

**6.9 (Recall)** What is prefill/decode disaggregation?

> Running compute-bound prefill and memory-bound decode on separate GPU pools and moving the KV cache between them. llm-d and NVIDIA Dynamo offer it on top of engines such as vLLM and SGLang.

*Course section: Serving engine techniques. ID: `m6-disaggregation`*

**6.10 (Recall)** How should you order a prompt for prompt caching?

> Stable content first (system prompt, tool definitions, documents), variable content last. Anything that changes early, even a timestamp, breaks the cache for everything after it.

*Course section: Cost levers. ID: `m6-cache-order`*

**6.11 (Recall)** When does prompt caching actually save money?

> When a prefix is reused several times within the cache lifetime. Reads are discounted, but some providers charge extra for cache writes, and lifetimes vary: minutes for some automatic caches, an hour or more for explicitly configured caches, which may also be billed for storage.

*Course section: Cost levers. Time-sensitive: checked 9 November 2026. ID: `m6-cache-economics`*

**6.12 (Recall)** What do provider batch APIs trade for their discount?

> Latency: requests complete asynchronously, typically within about 24 hours, at roughly half price.

*Course section: Cost levers. Time-sensitive: checked 9 November 2026. ID: `m6-batch-api`*

**6.13 (Recall)** If you use schema-constrained structured outputs, should you still validate?

> Yes. Validate with Pydantic and retry with the error message; schema support and strictness differ between providers.

*Course section: Structured outputs. ID: `m6-structured`*

**6.14 (Scenario)** Should you fine-tune a model on 5,000 policy documents so the chatbot knows them?

> Usually no: fine-tuning injects facts unreliably, needs retraining on every change, and can’t cite sources or enforce permissions. Use RAG; fine-tune for format, style, narrow skills or distillation.

*Course section: Exercises (E6.4). ID: `m6-finetune-facts`*

**6.15 (Scenario)** 70% of an agent’s frontier-model calls are routing and field extraction, and costs are too high. Which adaptation fits?

> Distillation: fine-tune a small model on the frontier model’s outputs for those steps (often supervised fine-tuning with LoRA), verify on your eval set, then canary.

*Course section: Post-training and adapting models for agents. ID: `m6-distill`*

**6.16 (Scenario)** You have a coding agent and unit tests that can grade its work. Which post-training method uses that signal?

> Reinforcement fine-tuning with verifiable rewards. Watch for reward hacking, such as the model editing the tests.

*Course section: Post-training and adapting models for agents. ID: `m6-rft`*

**6.17 (Recall)** What data does preference optimization, such as DPO, need?

> Pairs of a better and a worse response to the same prompt.

*Course section: Post-training and adapting models for agents. ID: `m6-dpo`*

**6.18 (Recall)** What is reward hacking in reinforcement fine-tuning?

> The model satisfies the grader without solving the task, for example by editing the tests instead of fixing the code.

*Course section: Post-training and adapting models for agents. ID: `m6-reward-hacking`*

**6.19 (Recall)** When should you use high reasoning effort?

> For planning and hard multi-step decisions. Use low effort for routing and extraction, because thinking tokens add latency and cost.

*Course section: Core concepts. ID: `m6-reasoning`*

**6.20 (Recall)** When is self-hosting open-weights models usually cheaper than an API?

> At high, steady volume with good GPU utilization, or when data must stay in-house. At low or bursty volume an API is usually cheaper once engineering and on-call time are counted.

*Course section: API vs self-hosting. ID: `m6-self-host`*

**6.21 (Scenario)** You add 50,000 tokens of retrieved documents to every prompt. Which latency metric rises most, and what helps?

> Time to first token, because prefill grows. Prefix caching helps if the same documents repeat within the cache lifetime; otherwise retrieve less.

*Course section: Exercises (E6.1). ID: `m6-ttft-docs`*

**6.22 (Scenario)** A prompt starts with today’s date, followed by 17,000 tokens of fixed instructions and tools. Why is the cache hit rate near zero?

> The changing date at the start breaks the cached prefix for everything after it. Move the date and other variable content to the end.

*Course section: Exercises (E6.3). ID: `m6-cache-bust`*

**6.23 (Calculation)** A request needs 10.5 GB of KV cache at FP16. With an FP8 KV cache, how many such requests fit in 40 GB of free memory?

> About 7 (about 5.25 GB each), up from about 3 at FP16.

*Course section: Exercises (E6.2). ID: `m6-kv-fp8`*

**6.24 (Calculation)** 1,000 sessions each hold about 3,000 active tokens, at 160 KB per token (FP8 KV cache, 70B-class model). Total KV cache?

> About 480 GB (3 million tokens × 160 KB), on top of roughly 70 GB of FP8 weights per replica: several 8-GPU nodes. Validate with a load test.

*Course section: Mock interview 3. ID: `m6-fleet-kv`*

## Module 7: RAG

**7.1 (Recall)** Why use hybrid retrieval?

> BM25 catches exact rare terms such as error codes and SKUs; dense retrieval catches paraphrases. Fusing both covers both failure modes.

*Course section: Core concepts. ID: `m7-hybrid`*

**7.2 (Recall)** What is the reciprocal rank fusion (RRF) formula?

> score(d) = Σ 1 ÷ (k + rank_i(d)) across rankers, with k usually 60.

*Course section: Core concepts. ID: `m7-rrf`*

**7.3 (Recall)** What does a cross-encoder reranker do?

> Reads the query and chunk together and scores relevance far more accurately than embedding similarity. Apply it to the top ~50 candidates and keep 5 to 10.

*Course section: Core concepts. ID: `m7-rerank`*

**7.4 (Recall)** Where must document permissions be enforced in RAG?

> At retrieval time, with metadata filters tied to the user’s identity. Never by asking the model to hide content.

*Course section: Core concepts. ID: `m7-acl`*

**7.5 (Recall)** When in the indexing pipeline are document permissions captured?

> During enrichment: each chunk stores its source’s access-control list, so the query path can filter by the user’s identity.

*Course section: Core concepts. ID: `m7-acl-index`*

**7.6 (Recall)** HNSW vs IVF-PQ?

> HNSW: a graph index with high recall that uses a lot of memory; the common default. IVF with product quantization: compressed and cheaper at billions of vectors, with lower recall.

*Course section: Core concepts. ID: `m7-hnsw`*

**7.7 (Recall)** What is parent-document retrieval?

> Retrieve small, precise chunks, but return their larger parent section to the model so it gets enough context.

*Course section: Core concepts. ID: `m7-parent`*

**7.8 (Recall)** What are contextual chunk headers?

> The document title and section path, or a one-line summary, prepended to each chunk before embedding so the chunk keeps its context.

*Course section: Core concepts. ID: `m7-context-headers`*

**7.9 (Recall)** What happens when you change embedding models?

> You must re-embed the whole corpus, so version the index and run old and new side by side during migration.

*Course section: Core concepts. ID: `m7-reembed`*

**7.10 (Recall)** When could long context replace RAG?

> For a small, stable corpus (up to a few hundred thousand tokens) with prompt caching. It is simpler, but costs more per query and can be weaker on very long inputs.

*Course section: Core concepts. ID: `m7-long-context`*

**7.11 (Scenario)** The right document exists, but the answer says it can’t be found. Likely cause and first fixes?

> A retrieval miss. Add hybrid search, query rewriting, contextual chunk headers or better chunking, and measure recall@k.

*Course section: RAG failure modes and fixes. ID: `m7-miss`*

**7.12 (Recall)** How do you evaluate the retrieval half of RAG?

> Recall@k against a golden set of questions labeled with their correct sources (did a needed source appear in the top k?), plus MRR or NDCG for ranking quality.

*Course section: Exercises (E7.4). ID: `m7-retrieval-eval`*

**7.13 (Recall)** What does a faithfulness (groundedness) metric check?

> Whether every claim in the answer is supported by the retrieved context, regardless of whether it is true in the world.

*Course section: RAG failure modes and fixes. ID: `m7-faithfulness`*

**7.14 (Calculation)** Raw memory for 20 million 1,024-dimension float32 vectors?

> About 82 GB (4 KB each), plus HNSW graph overhead. int8 quantization cuts the vectors about 4×.

*Course section: Exercises (E7.3). ID: `m7-vector-memory`*

**7.15 (Scenario)** Users search for error code “E-4012”, and dense retrieval returns unrelated VPN articles. Fix?

> Add BM25 (hybrid retrieval): exact rare tokens are what lexical search catches and embeddings blur.

*Course section: Exercises (E7.1). ID: `m7-error-code`*

**7.16 (Scenario)** The bot cites an outdated version of a policy. Fix?

> Deduplicate, store version and date metadata, delete superseded chunks when a document is updated, and boost recent versions.

*Course section: RAG failure modes and fixes. ID: `m7-stale-policy`*

**7.17 (Scenario)** A contractor sees salary-band content in an answer. Root cause and fix?

> Permissions weren’t filtered at retrieval. Sync access-control lists into chunk metadata, filter by the user’s groups on every query, and test it in CI.

*Course section: RAG failure modes and fixes. ID: `m7-acl-leak`*

## Module 8: Agents and tools

**8.1 (Recall)** What is an agent?

> An LLM in a loop that chooses actions (tool calls), observes the results, and decides when it is done.

*Course section: Core concepts. ID: `m8-agent`*

**8.2 (Recall)** Who executes an agent’s tool calls?

> Never the model itself: it only emits a structured request. Your runtime executes client-side tools; hosted (server-side) tools, such as a provider’s built-in web search or code execution, run on the provider’s platform.

*Course section: Core concepts. ID: `m8-who-executes`*

**8.3 (Recall)** Workflow vs agent?

> A workflow runs LLM calls along predefined code paths; an agent lets the model direct its own steps. Use a workflow when the steps are known.

*Course section: Core concepts. ID: `m8-workflow-agent`*

**8.4 (Scenario)** Support tickets need different handling for billing, technical and refund issues. Which workflow pattern?

> Routing: classify the input, then send it to a specialized prompt, model or workflow.

*Course section: Workflow patterns. ID: `m8-routing`*

**8.5 (Scenario)** A contract must be checked against 10 independent policies quickly. Which workflow pattern?

> Parallelization (sectioning): run the 10 checks concurrently and combine the results.

*Course section: Workflow patterns. ID: `m8-parallel`*

**8.6 (Scenario)** A code change will touch an unknown set of files. Which workflow pattern?

> Orchestrator-workers: a lead model decides the subtasks at run time, delegates them, and merges the results.

*Course section: Workflow patterns. ID: `m8-orchestrator`*

**8.7 (Scenario)** Generated SQL must pass a test suite, and iterating helps. Which workflow pattern?

> Evaluator-optimizer: one step generates, another evaluates against the criteria, and the loop repeats until it passes.

*Course section: Workflow patterns. ID: `m8-evaluator`*

**8.8 (Scenario)** Draft marketing copy, then translate it, with a check in between. Which workflow pattern?

> Prompt chaining: a fixed sequence of calls, each using the previous output, with checks between steps.

*Course section: Workflow patterns. ID: `m8-chaining`*

**8.9 (Scenario)** “Investigate why last night’s data pipeline failed and propose a fix.” Workflow or agent?

> Agent: the investigation path depends on what the logs and code reveal.

*Course section: Exercises (E8.2). ID: `m8-investigate`*

**8.10 (Recall)** ReAct vs plan-and-execute?

> ReAct interleaves reasoning and tool calls step by step. Plan-and-execute writes a plan first, executes it and re-plans on failure; it suits long tasks and human review of the plan.

*Course section: Core concepts. ID: `m8-react`*

**8.11 (Recall)** Why prefer one schedule_meeting tool over list_calendars + get_free_busy + create_event?

> Fewer, task-level tools mean fewer steps, fewer chances to chain calls wrongly, and less context spent on tool definitions.

*Course section: Tool design. ID: `m8-task-tools`*

**8.12 (Recall)** What should a tool’s error message contain?

> What went wrong and how to fix the call, so the model can recover. For example: “Order not found; ask the user to confirm the ID.”

*Course section: Tool design. ID: `m8-tool-errors`*

**8.13 (Recall)** Why paginate or truncate tool outputs?

> Large outputs flood the context window, raising cost and latency and degrading the model’s attention.

*Course section: Tool design. ID: `m8-tool-output`*

**8.14 (Recall)** Which stop conditions should every agent have?

> A step limit, a token or dollar budget, a wall-clock limit and loop detection, returning a partial result with an explanation when one is hit.

*Course section: Stop conditions and budgets. ID: `m8-stops`*

**8.15 (Recall)** When should an agent require human approval?

> For irreversible, financial, external-communication or privileged actions, showing the exact tool call and its arguments.

*Course section: Human in the loop. ID: `m8-approval`*

**8.16 (Recall)** When is a multi-agent architecture worth it?

> When the task is broad, parallelizable, exceeds one context window and justifies the cost. Anthropic reported that its multi-agent research system used about 15 times the tokens of a chat.

*Course section: Multi-agent systems. ID: `m8-multi-agent`*

**8.17 (Recall)** What is an idempotency key for?

> A unique ID per side-effecting operation, so a retried call doesn’t repeat it (for example, refunding twice).

*Course section: Code: a framework-free agent loop. ID: `m8-idempotency`*

**8.18 (Recall)** Who should supply the customer’s identity to a tool: the model or the runtime?

> The runtime. The backend then refuses other customers’ data no matter what the model asks.

*Course section: Code: a framework-free agent loop. ID: `m8-identity`*

**8.19 (Recall)** How do you give an agent hundreds of tools without flooding its context?

> Tool search (load relevant tool definitions on demand), namespaced MCP servers, and skills loaded only when relevant.

*Course section: Specialized agents and techniques. ID: `m8-tool-search`*

**8.20 (Recall)** What is an Agent Skill?

> A folder with a SKILL.md of instructions plus scripts and resources, loaded only when relevant (progressive disclosure). Anthropic published the format openly in late 2025.

*Course section: Specialized agents and techniques. ID: `m8-skills`*

**8.21 (Scenario)** An agent keeps calling search with the same arguments. What should the runtime do?

> Detect the repeat, return an error telling the model to use earlier results or finish, and enforce the step budget.

*Course section: Code: a framework-free agent loop. ID: `m8-loop`*

**8.22 (Scenario)** A support agent wants to issue a $120 refund. What happens before money moves?

> The policy layer checks the user’s scope and the amount limit, a human approves the exact call, and the backend executes it with an idempotency key.

*Course section: Code: a framework-free agent loop. ID: `m8-refund-flow`*

## Module 9: Protocols and frameworks

**9.1 (Recall)** What problem does MCP solve?

> It is one open protocol for connecting AI applications to tools, resources and prompts, so an integration built once works with any MCP-capable host instead of N × M custom integrations.

*Course section: MCP: Model Context Protocol. ID: `m9-mcp-why`*

**9.2 (Recall)** Who decides when an MCP tool is used?

> The model (tools are model-controlled).

*Course section: MCP: Model Context Protocol. ID: `m9-tools-control`*

**9.3 (Recall)** Who decides when an MCP resource is loaded into context?

> The host application (resources are application-controlled).

*Course section: MCP: Model Context Protocol. ID: `m9-resources-control`*

**9.4 (Recall)** Who decides when an MCP prompt is used?

> The user, for example by picking a slash command (prompts are user-controlled).

*Course section: MCP: Model Context Protocol. ID: `m9-prompts-control`*

**9.5 (Recall)** What is the current MCP specification version, and which did it replace?

> 2026-07-28, which replaced 2025-11-25. The specification is versioned by date.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-version`*

**9.6 (Recall)** Who created MCP, and who governs it now?

> Anthropic created it in November 2024. It is now governed under the Linux Foundation’s Agentic AI Foundation, formed in December 2025.

*Course section: MCP: Model Context Protocol. Time-sensitive: checked 9 November 2026. ID: `m9-governance`*

**9.7 (Recall)** What is the headline change in MCP 2026-07-28?

> A stateless core: no initialize handshake and no Mcp-Session-Id. Every request carries its version and capabilities, so any replica behind a plain load balancer can serve it.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-stateless`*

**9.8 (Recall)** What are MCP Multi Round-Trip Requests (MRTR)?

> A tool that needs user input returns “input required”; the client asks the user and retries the call with the answers. They replace server-initiated elicitation, sampling and roots requests.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-mrtr`*

**9.9 (Recall)** What are the Mcp-Method and Mcp-Name headers for?

> They let gateways, rate limiters and firewalls route and authorize requests without parsing JSON bodies.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-headers`*

**9.10 (Recall)** Which MCP features did the 2026-07-28 revision deprecate, and with what notice?

> Roots, sampling and protocol-level logging, plus the legacy HTTP+SSE transport, with at least a 12-month deprecation window.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-deprecations`*

**9.11 (Recall)** In MCP authorization, what replaces Dynamic Client Registration?

> Client ID Metadata Documents (CIMD). Dynamic Client Registration still works but is formally deprecated.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-dcr`*

**9.12 (Recall)** In MCP authorization, what role does a remote MCP server play?

> An OAuth-protected resource server: clients obtain access tokens from an authorization server and present them on each request.

*Course section: Agent identity and authorization. ID: `m9-auth-role`*

**9.13 (Recall)** What must an MCP client validate before redeeming an authorization code, and why?

> The iss (issuer) parameter, per RFC 9207, to close authorization-server mix-up attacks.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-rfc9207`*

**9.14 (Recall)** What is the MCP Tasks extension for?

> Long-running work: a request becomes a task the client can poll (tasks/get) and update, instead of holding a connection open.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-tasks`*

**9.15 (Recall)** What are MCP Apps?

> An official MCP extension that lets a server provide interactive user interfaces, such as forms or charts, that the host application renders.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-apps`*

**9.16 (Recall)** What do ttlMs and cacheScope on MCP list results do?

> Tell clients how long and how widely they may cache tool, prompt and resource lists, cutting refetches and keeping prompt caches stable.

*Course section: MCP: current version. Time-sensitive: checked 9 November 2026. ID: `m9-cache-hints`*

**9.17 (Recall)** What was FastMCP renamed to in the MCP Python SDK v2?

> MCPServer, imported with: from mcp.server import MCPServer.

*Course section: Code 1: expose tools as an MCP server. Time-sensitive: checked 9 November 2026. ID: `m9-sdk-rename`*

**9.18 (Recall)** In the MCP Python SDK v2, how does a tool return structured content?

> Annotate its return type with a Pydantic model or TypedDict. A bare dict is sent only as text, with no output schema.

*Course section: Code 1: expose tools as an MCP server. Time-sensitive: checked 9 November 2026. ID: `m9-sdk-typed`*

**9.19 (Recall)** In the MCP Python SDK v2, how does a tool report an error the model can read?

> Raise ToolError with a helpful message. Other exceptions are reported only as a generic “Error executing tool” message.

*Course section: Code 1: expose tools as an MCP server. Time-sensitive: checked 9 November 2026. ID: `m9-tool-error`*

**9.20 (Recall)** What is an A2A Agent Card, and where is it conventionally served?

> A JSON document describing an agent’s skills, interfaces (URL plus protocol binding), capabilities and security requirements, served at /.well-known/agent-card.json.

*Course section: A2A: Agent2Agent protocol. Time-sensitive: checked 9 November 2026. ID: `m9-agent-card`*

**9.21 (Recall)** Which A2A task state means the remote agent needs more information from the caller?

> input-required (or auth-required when it needs credentials).

*Course section: A2A: Agent2Agent protocol. ID: `m9-a2a-input`*

**9.22 (Recall)** Which protocol bindings does A2A v1.0 define?

> JSON-RPC, gRPC and HTTP/REST, with the protobuf definition as the canonical source.

*Course section: A2A: Agent2Agent protocol. Time-sensitive: checked 9 November 2026. ID: `m9-a2a-bindings`*

**9.23 (Recall)** Why can A2A Agent Cards be signed?

> So a client can verify that the card came from the claimed provider and wasn’t tampered with.

*Course section: A2A: Agent2Agent protocol. ID: `m9-a2a-signed`*

**9.24 (Recall)** MCP vs A2A in one line?

> MCP connects an agent to tools and data (vertical); A2A connects independent agents across systems or organizations (horizontal).

*Course section: A2A: Agent2Agent protocol. ID: `m9-mcp-vs-a2a`*

**9.25 (Scenario)** Your planner and coder agents run in one service you own. Do you need A2A between them?

> Usually no: direct calls or shared state are faster and simpler. A2A pays off across ownership or trust boundaries.

*Course section: Exercises (E9.5). ID: `m9-a2a-inhouse`*

**9.26 (Recall)** What is the main pitfall when a LangGraph run resumes after interrupt()?

> The interrupted node re-runs from the top, so code before interrupt() runs twice. Put side effects after it, make them idempotent, and avoid parallel tool calls in approval steps.

*Course section: Code 2: LangGraph with checkpointing. ID: `m9-langgraph-resume`*

**9.27 (Scenario)** A company has 40 internal systems to expose to agents. What is the architecture in one sentence?

> Task-level MCP servers per domain behind an MCP gateway with per-user OAuth, per-tool authorization, rate limits, audit logs and an allowlisted server registry.

*Course section: Exercises (E9.4). ID: `m9-forty-systems`*

**9.28 (Scenario)** A tool on a stateless MCP server needs the user’s confirmation mid-call. Which mechanism?

> A Multi Round-Trip Request: the tool returns “input required”, the client asks the user, and the call is retried with the answer.

*Course section: MCP: current version. ID: `m9-confirm-midcall`*

## Module 10: Context, memory, state

**10.1 (Recall)** What are the four context-engineering strategies?

> Write (persist outside the window), Select (pull in only what is needed), Compress (summarize or clear old content), Isolate (split work across subagents).

*Course section: Core concepts. ID: `m10-strategies`*

**10.2 (Scenario)** Running test-fixing in a subagent that returns only a summary is which context strategy?

> Isolate: the subagent explores in its own context window and hands back a condensed result.

*Course section: Core concepts. ID: `m10-isolate`*

**10.3 (Scenario)** Keeping a progress notes file (plan, done, next) that survives compaction is which context strategy?

> Write: information is persisted outside the context window.

*Course section: Core concepts. ID: `m10-write`*

**10.4 (Recall)** What must context compaction never do?

> Separate a tool result from the assistant message that called it; chat APIs reject orphaned tool results. Cut on turn boundaries.

*Course section: Core concepts. ID: `m10-compaction`*

**10.5 (Recall)** What is just-in-time context?

> Giving the agent tools to list, search and open content so it loads only what it needs, instead of pasting everything up front.

*Course section: Core concepts. ID: `m10-jit`*

**10.6 (Scenario)** “The user prefers metric units” is which type of memory?

> Semantic memory: a long-term fact or preference about the user or domain.

*Course section: Memory types. ID: `m10-semantic`*

**10.7 (Scenario)** “Last time, the deploy failed because of a missing migration” is which type of memory?

> Episodic memory: a record of a past interaction and its outcome.

*Course section: Memory types. ID: `m10-episodic`*

**10.8 (Recall)** Where does procedural memory (how to do things) live in an agent system?

> In instructions, skills and playbooks, and eventually in fine-tuned weights.

*Course section: Memory types. ID: `m10-procedural`*

**10.9 (Recall)** What is memory poisoning, and how do you prevent it?

> Malicious content written into long-term memory that steers future sessions. Validate memory writes, and never let untrusted tool output write memory unchecked.

*Course section: Memory write policy. ID: `m10-poisoning`*

**10.10 (Recall)** What does durable execution give an agent?

> Each step is checkpointed and recorded, so after a crash the run resumes from the last step without re-running completed side effects.

*Course section: State and durable execution. ID: `m10-durable`*

**10.11 (Scenario)** An agent crashes at step 37 of 50 after sending 3 emails. What prevents duplicate emails on restart?

> Durable execution replays completed steps from recorded results, and the email tool uses idempotency keys, so any retry is deduplicated.

*Course section: Exercises (E10.5). ID: `m10-crash`*

**10.12 (Recall)** What is the saga pattern?

> For multi-step side effects, each step has a compensating action (for example, cancel the hotel if the flight booking fails) to undo partial work.

*Course section: State and durable execution. ID: `m10-saga`*

**10.13 (Recall)** How should a long-running background agent be exposed to users?

> Accept the job and return a job ID, run it on a worker queue with durable state, stream or notify progress, and let users intervene.

*Course section: Sync vs async agents. ID: `m10-async`*

**10.14 (Scenario)** A coding agent fails after 90 minutes because test logs fill its context. Name two fixes.

> Return only failing test names and the first lines of each failure (with a handle to fetch more), and clear old tool results once acted on. A progress notes file also helps.

*Course section: Exercises (E10.3). ID: `m10-test-logs`*

## Module 11: Evaluation

**11.1 (Recall)** Why grade an agent’s outcome state rather than its exact trajectory?

> There are usually many valid paths. Checking the final state (database rows, files, passing tests) is robust; over-specified trajectories make evals brittle.

*Course section: Core concepts. ID: `m11-outcome`*

**11.2 (Recall)** Which kind of grader should you reach for first?

> Code-based graders (exact match, schema validation, tests, state checks): fast, cheap and deterministic. Use LLM judges for what code can’t check.

*Course section: Graders. ID: `m11-code-first`*

**11.3 (Recall)** What is position bias in pairwise LLM judging, and the fix?

> The judge prefers whichever answer is shown first. Randomize or swap the order and average.

*Course section: Graders. ID: `m11-position-bias`*

**11.4 (Recall)** Besides position bias, name two LLM-judge biases.

> Verbosity bias (preferring longer answers) and self-preference (favoring outputs from its own model family).

*Course section: Graders. ID: `m11-other-biases`*

**11.5 (Recall)** How do you check whether an LLM judge can be trusted?

> Compare it with human labels and report raw agreement plus Cohen’s kappa, which corrects for chance agreement.

*Course section: Comparing two systems. ID: `m11-kappa`*

**11.6 (Recall)** pass@k vs pass^k?

> pass@k: at least one of k attempts succeeds, which measures capability. pass^k: all k succeed, which measures reliability.

*Course section: Non-determinism and reliability. ID: `m11-pass-k`*

**11.7 (Recall)** How do you estimate pass^k correctly?

> Run n trials with n larger than k, count c successes, and use C(c, k) ÷ C(n, k). With n = k it collapses to “did every trial pass”.

*Course section: Non-determinism and reliability. ID: `m11-estimator`*

**11.8 (Calculation)** An agent succeeds 70% of the time per trial. What is its pass^5?

> About 17% (0.7⁵ ≈ 0.168).

*Course section: Non-determinism and reliability. ID: `m11-pass5`*

**11.9 (Calculation)** A case passes 7 of 10 trials. Estimate pass^3.

> C(7, 3) ÷ C(10, 3) = 35 ÷ 120 ≈ 0.29.

*Course section: Code: a minimal eval harness. ID: `m11-pass3-from10`*

**11.10 (Calculation)** An agent succeeds 90% of the time per independent trial. What are pass^3 and pass@3?

> pass^3 = 0.9³ = 0.729; pass@3 = 1 − 0.1³ = 0.999.

*Course section: Exercises (E11.2). ID: `m11-90`*

**11.11 (Recall)** How do you tell whether version B really beats version A on the same eval cases?

> A paired bootstrap: resample the cases many times, recompute the score difference, and check whether the 95% interval excludes zero.

*Course section: Comparing two systems. ID: `m11-bootstrap`*

**11.12 (Recall)** Why compare two systems on the same cases (paired) rather than as independent samples?

> Case difficulty varies far more than system quality; pairing removes that variance, giving a narrower interval.

*Course section: Comparing two systems. ID: `m11-pairing`*

**11.13 (Recall)** Where do good eval cases come from?

> Real production traces, support tickets, domain experts, reviewed synthetic cases and adversarial cases. Every production bug becomes a regression case.

*Course section: Core concepts. ID: `m11-sources`*

**11.14 (Recall)** Which benchmark introduced the pass^k metric?

> τ-bench, which tests tool-using agents with simulated users; τ²-bench is its successor.

*Course section: Public benchmarks. ID: `m11-tau`*

**11.15 (Recall)** Why are public benchmarks not enough to evaluate your agent?

> They compare models, not your system, and they saturate or leak into training data. Your own task-specific eval decides.

*Course section: Public benchmarks. ID: `m11-public`*

**11.16 (Scenario)** Your LLM judge agrees with human labels 72% of the time. What do you do before trusting it?

> Read the disagreements, turn the rubric into specific binary criteria, require reasoning before the verdict, give it reference answers, and re-measure agreement and kappa on held-out labels.

*Course section: Exercises (E11.3). ID: `m11-judge-72`*

**11.17 (Scenario)** A newer model is available for your agent. What must happen before you switch?

> It must match or beat the current one on the offline suite (multiple trials, paired comparison, cost, latency, safety cases), then pass a shadow run and a canary A/B test, with instant rollback ready.

*Course section: Exercises (E11.5). ID: `m11-new-model`*

**11.18 (Calculation)** Version A passes 140 of 200 cases; B passes 152 of the same 200 (16 fixed, 4 broken). Is B better?

> Very likely: a paired bootstrap gives +6 points with a 95% interval of about +1.5 to +10.5 points, which excludes zero.

*Course section: Exercises (E11.6). ID: `m11-a-vs-b`*

## Module 12: Security and identity

**12.1 (Recall)** Direct vs indirect prompt injection?

> Direct: the user tries to override the instructions. Indirect: instructions hidden in content the agent reads, such as web pages, emails, documents or tool output.

*Course section: Core concepts. ID: `m12-injection`*

**12.2 (Recall)** What is the lethal trifecta?

> Private data + untrusted content + external communication. An agent with all three can be tricked into exfiltrating data; break at least one leg.

*Course section: The lethal trifecta. ID: `m12-trifecta`*

**12.3 (Scenario)** An email assistant must read incoming mail and the user’s mailbox. Which leg of the lethal trifecta do you remove?

> External communication: no sending or arbitrary HTTP while summarizing, no remote images, and any reply needs explicit user confirmation of the exact message and recipient.

*Course section: Exercises (E12.1). ID: `m12-email-trifecta`*

**12.4 (Recall)** Why isn’t a warning in the system prompt a security control?

> It is probabilistic and fails some fraction of the time at scale. Real controls hold even when the model is fooled: scopes, sandboxes, egress allowlists, approvals.

*Course section: Exercises (E12.5). ID: `m12-system-prompt`*

**12.5 (Recall)** OWASP Top 10 for Agentic Applications, ASI01: which risk, and the main defense?

> Agent goal hijack: content the agent reads redirects its objective. Treat retrieved content as untrusted and constrain what the agent may do regardless of its context.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi01`*

**12.6 (Recall)** OWASP agentic ASI02: which risk, and the main defense?

> Tool misuse and exploitation. Scope tools to least agency and validate parameters at run time.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi02`*

**12.7 (Recall)** OWASP agentic ASI03: which risk, and the main defense?

> Identity and privilege abuse. Give each agent its own identity and short-lived, task-scoped credentials.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi03`*

**12.8 (Recall)** OWASP agentic ASI04: which risk, and the main defense?

> Agentic supply chain vulnerabilities (MCP servers, packages, prompts). Pin and verify components and keep an inventory of agent components.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi04`*

**12.9 (Recall)** OWASP agentic ASI05: which risk, and the main defense?

> Unexpected code execution. Run code in sandboxes with deny-by-default network egress.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi05`*

**12.10 (Recall)** OWASP agentic ASI06: which risk, and the main defense?

> Memory and context poisoning. Validate memory writes, scope memory per user, and make it inspectable.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi06`*

**12.11 (Recall)** OWASP agentic ASI07: which risk, and the main defense?

> Insecure inter-agent communication. Use mutual authentication, signed messages and delegation allowlists.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi07`*

**12.12 (Recall)** OWASP agentic ASI08: which risk, and the main defense?

> Cascading failures. Isolate blast radius, add circuit breakers, and separate development from production.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi08`*

**12.13 (Recall)** OWASP agentic ASI09: which risk, and the main defense?

> Human-agent trust exploitation, such as approvals based on the agent’s own summary. Confirmations must show the raw action and arguments.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi09`*

**12.14 (Recall)** OWASP agentic ASI10: which risk, and the main defense?

> Rogue agents. Monitor behavior against a baseline, give every agent an owner, and keep a tested kill switch.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-asi10`*

**12.15 (Recall)** What is LLM01 in the OWASP Top 10 for LLM Applications (2025)?

> Prompt injection.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-llm01`*

**12.16 (Recall)** What is “improper output handling” (OWASP LLM05)?

> Passing model output to downstream systems, such as browsers, shells or SQL, without validation or escaping.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-llm05`*

**12.17 (Recall)** What is “excessive agency” (OWASP LLM06)?

> Giving an LLM application more functionality, permissions or autonomy than it needs, so a manipulated model can do real damage.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-llm06`*

**12.18 (Recall)** What is “unbounded consumption” (OWASP LLM10)?

> Uncontrolled resource use (tokens, compute, money) that enables cost blow-ups or denial of service. Defend with rate limits, budgets and loop detection.

*Course section: OWASP lists to know by name. Time-sensitive: checked 9 November 2026. ID: `m12-llm10`*

**12.19 (Recall)** What is least agency?

> Giving an agent only the minimum tools, scopes and autonomy its task needs.

*Course section: Defense in depth for agents. ID: `m12-least-agency`*

**12.20 (Recall)** What is token passthrough, and why does MCP forbid it?

> Forwarding a token issued for your service to a third-party server. It enables confused-deputy attacks and token theft; obtain a token issued for that server instead.

*Course section: Agent identity and authorization. ID: `m12-passthrough`*

**12.21 (Recall)** What does OAuth 2.0 Token Exchange (RFC 8693) do for agents?

> It swaps the user’s token for a narrower, audience-restricted token, so the agent can call downstream APIs on the user’s behalf with least privilege.

*Course section: Agent identity and authorization. ID: `m12-rfc8693`*

**12.22 (Recall)** How can an agent leak data through the output it renders?

> Markdown images or links pointing to an attacker’s URL can carry data in the URL. Block model-generated remote images and allowlist egress.

*Course section: Defense in depth for agents. ID: `m12-exfil`*

**12.23 (Recall)** What is the dual-LLM pattern?

> A privileged planner never sees untrusted content; a quarantined model processes it and returns only constrained, typed values. Google DeepMind’s CaMeL develops the idea further.

*Course section: Defense in depth for agents. ID: `m12-dual-llm`*

**12.24 (Recall)** Containers vs microVMs for running agent-generated code?

> Containers share the host kernel, so isolation is weaker; microVMs (such as Firecracker) or user-space kernels (such as gVisor) add a stronger boundary at some startup and runtime cost. Either way: no secrets inside, resource limits, an egress allowlist.

*Course section: Defense in depth for agents. ID: `m12-sandbox`*

**12.25 (Recall)** Since when have the EU AI Act’s prohibited practices applied?

> February 2025.

*Course section: Governance and regulation. Time-sensitive: checked 9 November 2026. ID: `m12-eu-prohibited`*

**12.26 (Recall)** Since when have EU AI Act obligations for general-purpose AI models applied?

> August 2025.

*Course section: Governance and regulation. Time-sensitive: checked 9 November 2026. ID: `m12-eu-gpai`*

**12.27 (Recall)** When do the EU AI Act’s Article 50 transparency obligations apply, and what is the one exception?

> From 2 August 2026 (for example, telling people they are talking to AI and labeling deepfakes). Exception: generative AI systems already on the market before that date have until 2 December 2026 to add machine-readable marking (watermarking) of outputs.

*Course section: Governance and regulation. Time-sensitive: checked 9 November 2026. ID: `m12-eu-art50`*

**12.28 (Recall)** After the AI Omnibus, when do the EU AI Act’s high-risk obligations apply?

> 2 December 2027 for standalone high-risk systems (Annex III) and 2 August 2028 for AI embedded in regulated products (Annex I).

*Course section: Governance and regulation. Time-sensitive: checked 9 November 2026. ID: `m12-eu-high-risk`*

**12.29 (Scenario)** A coding agent runs commands on customer repositories. Name two essential controls.

> An ephemeral sandbox with egress limited to package registries and the git host, and a short-lived token for one repository that can open a pull request but never merge.

*Course section: Exercises (E12.4). ID: `m12-coding-sandbox`*

**12.30 (Scenario)** A sales agent updates a CRM through a third-party MCP server. May it forward the user’s token for your app?

> No: that is token passthrough. Obtain a token issued for that MCP server, scoped to the CRM fields needed, and have the user confirm writes.

*Course section: Exercises (E12.6). ID: `m12-crm-token`*

## Module 13: LLMOps

**13.1 (Recall)** What should one agent trace contain?

> One span per LLM call, tool call, retrieval and guardrail check, with redacted inputs and outputs, model and prompt version, tokens, cost, latency and errors.

*Course section: Core concepts. ID: `m13-trace`*

**13.2 (Recall)** Which OpenTelemetry GenAI attributes record the model and token usage?

> gen_ai.request.model, gen_ai.usage.input_tokens and gen_ai.usage.output_tokens. The GenAI conventions are still evolving.

*Course section: Core concepts. Time-sensitive: checked 9 November 2026. ID: `m13-otel`*

**13.3 (Recall)** Why put an AI gateway between teams and model providers?

> So keys, budgets, rate limits, retries, failover, logging and redaction are implemented once, and every call is attributable to a team and task.

*Course section: The AI gateway. ID: `m13-gateway`*

**13.4 (Recall)** Which LLM API errors should you retry?

> 429 (honoring Retry-After), 5xx, timeouts and connection errors.

*Course section: Reliability patterns. ID: `m13-retry`*

**13.5 (Scenario)** A request fails with HTTP 400. Retry, fall back to another provider, or neither?

> Neither: the request itself is invalid, so another provider will reject it too. Fix the request.

*Course section: Code: a resilient, traced LLM call. ID: `m13-400`*

**13.6 (Recall)** Why add jitter to exponential backoff?

> So that many clients that failed together don’t retry in synchronized waves (a thundering herd).

*Course section: Reliability patterns. ID: `m13-jitter`*

**13.7 (Recall)** Why should a retry wrapper have an overall deadline?

> Per-attempt timeouts plus backoff across several providers can otherwise add up to minutes.

*Course section: Code: a resilient, traced LLM call. ID: `m13-deadline`*

**13.8 (Calculation)** An agent runs 9 LLM calls of 3 s and 4 independent tool calls of 4 s, all sequentially (43 s). How much does running the tools in parallel save?

> About 12 s: the tools take about 4 s together instead of 16 s, so the run drops to about 31 s.

*Course section: Exercises (E13.3). ID: `m13-parallel`*

**13.9 (Calculation)** Of those 9 LLM calls (3 s each), 5 are routing and extraction. Moving them to a small model at about 1 s each saves how much?

> About 10 s: LLM time drops from 27 s to 4 × 3 + 5 × 1 = 17 s. Verify quality on the eval set.

*Course section: Exercises (E13.3). ID: `m13-small-models`*

**13.10 (Recall)** What should be versioned together as one release?

> The prompt, a pinned model version, tool descriptions, retrieval data and the framework version. Record the bundle version on every trace.

*Course section: Releases and change management. ID: `m13-bundle`*

**13.11 (Recall)** Why pin a dated model snapshot in production rather than a floating alias?

> An alias can change behavior without your release; a pinned version changes only when you ship it through your evals.

*Course section: Releases and change management. ID: `m13-pin`*

**13.12 (Scenario)** A customer reports the agent wrongly said their order was cancelled. First step?

> Find the trace and walk its spans: was the tool data wrong, did the model misread it, or did compaction drop the status? Then replay it with the same versions.

*Course section: Exercises (E13.2). ID: `m13-wrong-answer`*

**13.13 (Scenario)** Your primary model provider returns HTTP 429 during a launch. What should happen?

> Retry with backoff and jitter, honoring Retry-After, then fail over to a fallback provider within an overall deadline, and degrade gracefully if everything fails.

*Course section: Reliability patterns. ID: `m13-429-launch`*

## Module 14: Agentic case studies

**14.1 (Recall)** Why is a customer-support agent usually a single agent, not a multi-agent system?

> Conversations are sequential, tightly coupled and latency-sensitive. One agent with good tools and routing is faster, cheaper and easier to evaluate.

*Course section: Case study A: customer support agent. ID: `m14-support-single`*

**14.2 (Recall)** Where must a support agent’s refund limits live?

> In the backend that executes refunds, not in the prompt.

*Course section: Case study A: customer support agent. ID: `m14-refund-limits`*

**14.3 (Recall)** Why does a deep-research agent justify a multi-agent design?

> The work is broad, parallel and exceeds one context window. Subagents research in isolation and return condensed briefs, followed by a citation-verification pass.

*Course section: Case study B: deep-research agent. ID: `m14-research`*

**14.4 (Recall)** What fixes most text-to-SQL failures?

> A semantic layer with curated metric definitions, because most failures come from ambiguous business definitions, not SQL syntax.

*Course section: Case study C: data-analyst agent. ID: `m14-semantic-layer`*

**14.5 (Recall)** How should a text-to-SQL agent be evaluated?

> By execution accuracy: run the generated and reference queries and compare the result sets, not the SQL strings.

*Course section: Case study C: data-analyst agent. ID: `m14-execution-accuracy`*

**14.6 (Scenario)** Insurance claims arrive as emails and PDFs, and the steps are known. Agent or workflow?

> A workflow: classification, schema extraction, validation rules and confidence-based human review are cheaper and auditable.

*Course section: Case study D: claims intake. ID: `m14-claims`*

**14.7 (Recall)** What limit applies to an AI recruiting assistant?

> Decision support only, never automatic rejection: employment decisions are high-risk under the EU AI Act and covered by anti-discrimination law.

*Course section: Exercises (E14.3). ID: `m14-recruiting`*

**14.8 (Scenario)** A travel agent finds that the flight price changed between search and booking. What should happen?

> The hold step re-quotes; if the price moved beyond a threshold, show the new price and ask again. Never book at a price the user didn’t confirm.

*Course section: Mock interview 2. ID: `m14-price-change`*

**14.9 (Scenario)** Resolution rate is 62% and leadership wants 80% in four weeks. What do you do first?

> Label the root causes of a sample of unresolved conversations (missing tool, retrieval miss, policy gap, model error, user wanted a human) and fix the biggest buckets first.

*Course section: Exercises (E14.4). ID: `m14-resolution`*
