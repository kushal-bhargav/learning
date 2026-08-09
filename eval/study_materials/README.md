# GMGI Human Study Materials

This folder contains draft materials for the Track B agency study described in `specs/06_evaluation_spec.md`.

Files:

- `consent_script.md`: facilitator script for informed consent.
- `session_protocol.md`: 3-condition within-subjects protocol.
- `questionnaire.md`: post-condition Likert items and open-ended authorship prompt.
- `authorship_coding_guide.md`: lightweight qualitative coding guide for the authorship responses.
- `analyze_wilcoxon.py`: paired Wilcoxon signed-rank analysis for collected CSV data.

Expected CSV shape for analysis:

```csv
participant_id,condition,authorship,control,satisfaction,novelty,trust,open_authorship,accept_count,edit_count,regenerate_count,delegate_count
P01,ai_autonomous,2,1,3,4,2,"Mostly the system decided.",0,0,0,6
P01,human_only,5,5,4,2,4,"I chose the pieces myself.",0,0,0,0
P01,negotiated_hybrid,4,4,5,4,4,"I guided it and the AI filled in details.",2,1,1,2
```

Condition ids must be exactly:

- `ai_autonomous`
- `human_only`
- `negotiated_hybrid`

Likert measures are expected on a 1 to 5 scale. Behavioral counts are optional but recommended.
