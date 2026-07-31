# 06 — Evaluation Spec

## Two evaluation tracks

### Track A: Generative quality (quantitative, automatic)
Covered in detail in `03_gan_model_spec.md` § Evaluation of the GAN itself. Summary of what must be reported in the paper:
- FID / KID vs. held-out real split.
- CLIPScore for conditioning fidelity.
- LPIPS-based interpolation smoothness along the agency-slider path.
- LPIPS-based intra-condition diversity (mode-collapse check).
Report these as a small table + 1 figure (interpolation strip image: `t=0..1` samples side by side).

### Track B: Agency / human study (small-n, qualitative + light quantitative)
This is the track-specific contribution — measuring **perceived agency**, not just satisfaction.

**Design**: within-subjects, 3 conditions per participant (order randomized):
1. **AI-autonomous**: agency_slider fixed at 1.0, no human edits allowed, agent runs straight through.
2. **Human-only baseline**: participant manually picks style/message with no AI generation (control condition — a plain form/checklist UI).
3. **Negotiated hybrid**: full Agency Console — accept/edit/regenerate/delegate at every stage.

**Participants**: n ≥ 12–20 (state actual n honestly; small-n qualitative framing is acceptable for a Creative AI track poster, do not inflate claims).

**Measures** (5-point Likert unless noted):
- Perceived **authorship** ("This gift feels like it came from me")
- Perceived **control** ("I felt in control of the outcome")
- **Satisfaction** with the resulting artifact
- **Novelty/surprise**
- **Trust** in the system's suggestions
- Open-ended: "Describe, in your own words, who or what made the creative decisions in this gift."
- Behavioral log (free, no self-report needed): count of accept/edit/regenerate/delegate actions per session (from `stage_log`).

**Analysis**:
- Report means/medians + spread per condition per measure; use paired Wilcoxon signed-rank (non-parametric, appropriate for small-n Likert data) between hybrid vs. each baseline.
- Report effect sizes even when not significant; be explicit about the small-n limitation — do not p-hack or overclaim.
- Thematically code the open-ended authorship question (2–4 short recurring themes) — this is the qualitative heart of the Agency contribution.

## Ablations (if time allows, ordered by priority)
1. With vs. without Memory Graph conditioning (generic style vs. memory-conditioned style) — CLIPScore + a short subjective preference question.
2. With vs. without bandit-personalized defaults (fixed agency_slider=0.5 baseline vs. learned default) — cumulative reward comparison (spec 05).
3. With vs. without Relationship Analysis Agent gating (does relationship-aware tone guidance change acceptance rates in `stage_log`?).

## Reporting discipline
- Never claim statistical significance without the actual test and n reported.
- Always report both the quantitative generative metrics *and* the human-agency measures — a Creative AI reviewer will care more about the latter, but an ML reviewer needs the former to trust the system is real.
- Include failure cases (a bad GAN sample, a rejected agent suggestion) — a short "limitations & failure modes" subsection is expected and strengthens credibility.
