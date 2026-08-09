# 3-Condition Session Protocol

## Study Design

Use a within-subjects design. Each participant completes all three conditions. Randomize condition order per participant to reduce order effects.

Recommended participant count: 12 to 20. Report the actual n honestly.

Conditions:

- `ai_autonomous`: agency slider fixed at 1.0. No human edits. The agent pipeline runs through to completion.
- `human_only`: participant manually selects style, gift concept, and message using a plain form or checklist. No AI generation is shown.
- `negotiated_hybrid`: full Agency Console. The participant can accept, edit, regenerate, or delegate at every stage.

## Setup

1. Assign a participant id, for example `P01`.
2. Randomize the order of the three conditions.
3. Use the same synthetic giver, recipient, occasion, and budget across all three conditions for that participant.
4. Confirm the participant has read and accepted the consent script.
5. Remind the participant that they are evaluating the interaction, not trying to produce a perfect gift.

## Shared Task Prompt

Facilitator says:

Create a gift concept for the provided synthetic recipient and occasion. Try to make something that feels personal and appropriate. After each session, answer the short questionnaire based on how that condition felt.

## Condition Procedure

### AI-Autonomous

1. Open the AI-autonomous version or configure the live console with `agency_slider=1.0`.
2. Do not allow participant edits, regeneration, or manual stage choices.
3. Run the system through recipient profiling, relationship analysis, recommendation, creative generation, greeting/story, and delivery planning.
4. Show the final gift artifact and Agency Ledger.
5. Export or store the stage log.
6. Administer the questionnaire.

### Human-Only Baseline

1. Open the human-only baseline form or checklist.
2. Ask the participant to manually choose or write:
   - gift category
   - visual style or motif
   - short message or greeting
   - delivery format
3. Do not show AI-generated recommendations or generated images.
4. Show a simple final summary of their choices.
5. Record the final choices and any available timing or interaction notes.
6. Administer the questionnaire.

### Negotiated Hybrid

1. Open the full Agency Console.
2. For each stage, the participant may choose Accept, Edit, Regenerate, or Delegate rest to AI.
3. On Creative Generation, allow use of the Agency Slider. Regenerate only after a pause or explicit regenerate action.
4. Do not auto-advance unless the participant explicitly delegates.
5. Show the final Agency Ledger timeline.
6. Export or store the stage log.
7. Administer the questionnaire.

## Counterbalancing

Use a balanced or randomized order. A simple rotation for 12 participants:

| Participant group | Condition order |
|---|---|
| 1 | ai_autonomous, human_only, negotiated_hybrid |
| 2 | ai_autonomous, negotiated_hybrid, human_only |
| 3 | human_only, ai_autonomous, negotiated_hybrid |
| 4 | human_only, negotiated_hybrid, ai_autonomous |
| 5 | negotiated_hybrid, ai_autonomous, human_only |
| 6 | negotiated_hybrid, human_only, ai_autonomous |

Repeat the rotation as needed.

## Data to Record

For each participant and condition:

- participant_id
- condition
- condition_order_index
- persona_id
- occasion_id
- final artifact reference or summary
- authorship, control, satisfaction, novelty, trust ratings
- open_authorship response
- accept_count, edit_count, regenerate_count, delegate_count when available
- optional notes on confusion, delays, or failures

## Facilitator Neutrality

Do not imply that any condition is expected to be better. Do not explain the research hypothesis until the participant has completed all three sessions.
