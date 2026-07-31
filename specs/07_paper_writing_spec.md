# 07 — Paper Writing Spec (NeurIPS 2026 Creative AI Track)

## Submission facts to respect (verify against the live call before submitting — dates/details can change)
- Theme: **Agency**.
- Length: **2–6 pages, excluding references**. Use the official Creative-AI-track template.
- Track is **non-archival** in 2026: accepted work goes to OpenReview + onsite poster / large-screen presentation, not the main proceedings — write for a poster-viewing audience skimming in ~2 minutes as much as for a careful reader.
- At least one author must register and attend in person if accepted.
- Track has historically accepted both research papers and artwork/system descriptions — lean into the working demo, don't over-formalize into a dry ML paper.

## Required content (from the call's own submission checklist — confirm current wording each cycle)
- Description of the work and **the roles of AI and ML** — this maps directly to `01_architecture_spec.md` + `03_gan_model_spec.md`: be concrete about what the GAN does vs. what the LLM agents do vs. what stays human.
- Description of **how the theme (Agency) is addressed** — this is the Agency Ledger / agency-slider / 3-condition study, front and center, not an afterthought.
- Short **author biographies**, including references to relevant prior work.

## Suggested section structure (fit to page budget; drop sections if over 6 pages)
1. **Title + abstract** (abstract states the agency question, the GAN-based mechanism, and the headline human-study finding in ≤150 words).
2. **Introduction** — open with the agency question (see `00_project_overview.md`), not with "gift recommendation is an unsolved problem."
3. **Related work** (brief, 1 short paragraph each, no exhaustive survey — cite RecSys, generative recommendation, conversational recommendation, human–AI co-creation, and the one existing AI-gift-recommendation consumer-behavior study; this project's own literature scan is the source list to draw from).
4. **System / Method** — pipeline diagram (from spec 01), the MemoryGAN conditioning + agency-slider mechanism (spec 03) as the centerpiece figure, brief mention of the bandit (spec 05).
5. **How Agency is instrumented** (can be folded into Method or its own short section) — the Agency Ledger concept, the accept/edit/regenerate/delegate actions.
6. **Evaluation** — generative metrics table + human study results (spec 06), with the interpolation-strip figure.
7. **Limitations & failure modes** — small dataset, small-n study, simulated delivery, bandit not full RL, GAN not SOTA fidelity. State plainly.
8. **Ethical considerations** — synthetic/consented data only, no real private data, no deceptive personalization, transparency of AI involvement to gift recipients as a design principle.
9. **Author bios** (required).
10. **References** (excluded from page count).

## Tone guidance
- Confident about the *framing contribution* (agency-instrumented pipeline + evaluation protocol), modest about *generative fidelity claims*.
- Prefer one strong, well-explained figure over five cluttered ones — poster/large-screen viewing rewards a single clear diagram.
- Write the abstract and section 1 last, once real results exist — don't lock in claims before the GAN/study are run.

## Deliverables checklist
- [ ] Draft in official template, page count checked (excluding refs).
- [ ] Pipeline diagram (can reuse/adapt `01_architecture_spec.md` ASCII diagram as a real figure).
- [ ] Interpolation-strip figure from the GAN (spec 03).
- [ ] Results table (Track A metrics) + results figure/table (Track B study).
- [ ] Author bios drafted.
- [ ] Poster derived from the same figures (spec 08 mentions demo asset; poster is a separate static deliverable — same visual language, less text).
