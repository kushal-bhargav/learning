# 02 â€” Memory Graph & Data Spec

## Purpose
Turn scattered signals about a relationship (dates, photos, chat snippets, stated preferences) into a structured graph that (a) conditions the GAN's generator, (b) grounds the LLM reasoning agents, and (c) gives the paper a clean "relationship-aware, memory-grounded" story distinct from plain RecSys.

## Node types
| Node | Key attributes |
|---|---|
| `Person` | id, display_name, role (`giver`\|`recipient`), demographic tags (optional, self-reported only) |
| `Relationship` | id, person_a, person_b, type (partner/parent/friend/colleague/...), closeness_score (1â€“5, self-reported) |
| `Event` | id, date, type (birthday/anniversary/graduation/...), participants[] |
| `Memory` | id, modality (`photo`\|`text`\|`audio-caption`), embedding (vector), timestamp, associated Event/Person, emotion_tag (enum: joy, nostalgia, gratitude, humor, comfort, other) |
| `Preference` | id, person_id, category (hobby/color/brand/food/...), value, confidence (0â€“1), source (stated vs inferred) |
| `Occasion` | id, name, date, budget_hint, formality |
| `GiftArtifact` | id, produced_by (agent id), type (image/text/bundle), agency_slider value at generation time |

## Edge types
`RELATES_TO(Person,Person,via Relationship)`, `PARTICIPATED_IN(Person,Event)`, `RECALLS(Memory,Event|Person)`, `PREFERS(Person,Preference)`, `INTENDED_FOR(GiftArtifact,Occasion)`, `GENERATED_FROM(GiftArtifact,Memory[])`.

## Construction pipeline
1. **Ingest** raw inputs (see `01_architecture_spec.md` Â§ Ingestion Layer).
2. **Extract** structured facts from free text via an LLM extraction prompt (structured JSON output: candidate Preferences, Events, emotion tags) â€” always mark `source=inferred` and `confidence<1` for anything not explicitly stated by the user.
3. **Embed** photos (CLIP image encoder) and text snippets (sentence embedding model); store as `Memory.embedding`.
4. **Link** extracted facts to existing nodes by fuzzy name/date match; create new nodes otherwise.
5. **Summarize** on demand: `embed_context(person_id, occasion_id)` returns a pooled embedding (mean or attention-weighted over relevant `Memory` + `Preference` embeddings) â€” this pooled vector is the GAN's primary conditioning input (`03_gan_model_spec.md`).

## Storage & API (MVP)
- Implementation: `networkx.MultiDiGraph`, persisted as JSON on disk per demo "household"/session.
- Embedding models: OpenCLIP `ViT-B-32` with `openai` pretrained weights for images, and `sentence-transformers/clip-ViT-B-32-multilingual-v1` for text. Both emit 512-dimensional vectors in the original CLIP ViT-B/32 space, so photo and text memories can be pooled without an untrained projection.
- `context_embedding` mean-pools all available, dimensionally compatible `Memory.embedding` and `Preference.embedding` vectors in the selected subgraph. Preferences without a stored embedding are encoded from `"{category}: {value}"` when a text encoder is configured. The method raises a clear error when no embeddings are available or dimensions disagree.
- Minimal API surface:
  - `graph.add_node(node_type, **attrs) -> node_id`
  - `graph.add_edge(src, dst, edge_type, **attrs)`
  - `graph.subgraph_for(person_id, occasion_id=None) -> nx.MultiDiGraph`
  - `graph.context_embedding(person_id, occasion_id) -> np.ndarray`
  - `graph.to_json()` / `graph.from_json()`

## Data ethics & scope guardrails
- **Use only synthetic personas or explicitly consented volunteer data** for the demo and paper. Do not scrape real people's social media.
- No storage of sensitive categories (health, religion, political affiliation) even if volunteered â€” filter these out at ingestion.
- All `Preference`/`emotion_tag` inference is probabilistic and shown to the human with its `confidence` â€” never presented as ground truth (ties directly into the Agency framing: inferred facts are proposals, not decisions).

## Test fixtures
Provide 3â€“5 synthetic personas (e.g., "long-distance partners," "parent-teen," "close coworkers") with hand-authored memories, so agents and GAN conditioning can be developed and demoed without needing real user data before the study.

