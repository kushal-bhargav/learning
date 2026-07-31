# 05 — Feedback / RL Spec (Scoped to a Contextual Bandit)

## Why scoped down
The source brainstorm proposes full RL over recipient satisfaction/novelty/emotional-value. For a 2–6 page Creative-AI submission with a small user study, a **contextual bandit** is the honest, defensible scope: it is a real online-learning method, it maps cleanly onto the "one decision per session" structure of gifting (no long-horizon credit assignment needed), and it can be evaluated rigorously with the sample sizes actually achievable. Full RL/RLHF is named explicitly as future work in the paper, not attempted at MVP scale.

## Problem formulation
- **Context** `x`: relationship_type, closeness_score, occasion formality, `agency_slider` default suggested by Relationship Analysis Agent.
- **Action** `a`: a discrete bucket over (recommendation category, agency_slider bucket ∈ {low, mid, high}, style archetype).
- **Reward** `r`: post-session human rating (1–5) mapped to [0,1], optionally combined with implicit signals (did the human `accept` most stages without heavy editing → higher implied satisfaction; heavy `regenerate` usage → lower).

## Algorithm
**LinUCB** (linear upper-confidence-bound contextual bandit) — simple, sample-efficient, well-understood, easy to implement in ~50–100 lines without extra infra.
```
for each session:
    observe context x
    for each candidate action a:
        p(a) = x^T θ_a + α * sqrt(x^T A_a^{-1} x)
    choose a* = argmax p(a)
    observe reward r after session
    update A_{a*} += x x^T ; b_{a*} += r * x
```
- `α` (exploration weight) tuned small given limited sessions expected during the study (favor exploitation once a handful of sessions are logged; don't waste a 12–20 person study on random exploration).

## What it updates
- The **default** `agency_slider` value proposed at session start per (relationship_type, closeness) bucket.
- The **default** style archetype fed as `human_style_ref` when the human hasn't specified one.
- It does **not** touch GAN weights directly (that would require expensive retraining) — it only changes which *conditioning inputs* are proposed, keeping the loop cheap and fast enough to demo live.

## Logging
Every session logs `(x, a, r)` triples to `experiments/bandit_log.jsonl`. Provide a small offline script (`eval/bandit_offline_eval.py`) that replays logged sessions to report regret/reward-over-time even if online updates during the live study are minimal.

## Evaluation of the bandit component
- Report **cumulative reward over session index** compared to a fixed non-adaptive baseline (always agency_slider=0.5) — even a small, non-significant improvement is reportable as a trend, and the honest comparison itself is a contribution (matches evenhanded, non-overclaiming tone required for a credible submission).

## Future work (state in paper, do not implement)
- Full RLHF-style reward modeling over open-ended recipient feedback text.
- Multi-step RL over the whole gifting relationship history (repeat gifting across occasions/years).
