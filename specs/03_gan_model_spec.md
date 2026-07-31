# 03 Ã¢â‚¬â€ Generative Core: Conditional GAN Spec ("MemoryGAN")

> This is the load-bearing technical component of the project. The explicit constraint is: **train with a GAN architecture** (not a diffusion model, not GAN-free LLM image generation). Everything below is scoped to be trainable on a single consumer/cloud GPU in a few days, which matches Creative-AI-track norms (small, reproducible, artist-usable systems) rather than large-lab compute.

## Task
Given a **conditioning vector** derived from the Memory Graph (relationship type, closeness, pooled memory/preference embedding, occasion, emotion tag) plus an **agency-slider** value, generate a personalized 2D visual artifact: a stylized illustration / greeting-card motif / gift-wrap pattern. This is one output among several (visual), paired with the LLM-generated text (spec 04).

## Why GAN (and which GAN family)
GANs remain the right tool here because (a) single forward-pass inference is fast enough for a live poster demo (no multi-step denoising loop), (b) small-data + strong augmentation training is well studied for GANs (StyleGAN2-ADA), and (c) GANs expose a clean, continuous, interpretable latent/style space, which is exactly the control surface the "Agency" framing needs (interpolating between human-specified and AI-inferred style is a latent-space operation).

**Base architecture: StyleGAN2 with Adaptive Discriminator Augmentation (ADA)**, generator capacity reduced for small-dataset / limited-compute training (channel multiplier reduced, e.g., `channel_base=16384` instead of `32768`; resolution capped at 256Ãƒâ€”256 or 512Ãƒâ€”512 depending on compute budget).


### Implemented MVP architecture (source of truth)
- `src/gan/configs/model.json` defines the real architecture: `z_dim=512`, `context_dim=512`, relationship/emotion/occasion one-hot widths `8/6/8`, projected conditioning width `512`, `w_dim=512`, and an 8-layer mapping network.
- Generator resolution is 256x256 with `channel_base=16384` and `channel_max=512`. It starts from a learned 4x4 constant, uses two modulated/demodulated 3x3 convolutions per higher-resolution block, learned per-layer noise, and skip-summed ToRGB outputs.
- The discriminator uses residual downsampling blocks, minibatch standard deviation, and a projection-conditioning term built from the same conditioning fields.
- For portability and testability, up/downsampling uses PyTorch bilinear interpolation and average pooling rather than StyleGAN2's fused CUDA `upfirdn2d` kernels. This is the only material implementation deviation from the reference StyleGAN2 resampling path; it avoids a custom CUDA build on the target Windows/consumer-GPU environment.
- ADA is the documented transform subset actually implemented: x-flip (pixel blitting), integer translation (geometric), brightness, and contrast (color). Probability is updated from the real-logit sign statistic with target `0.6`, interval `4`, and `ada_kimg=500`.
- Actual regularization values are `r1_gamma=0.8`, path-length weight `2.0` every 4 steps with decay `0.01`, and CLIP consistency weight `0.1`. CLIP consistency uses differentiable OpenCLIP ViT-B/32 image features and frozen text features.
Fallback/alternative if training instability or compute is tighter: **Lightweight-GAN** (skip-layer excitation, single discriminator scale) Ã¢â‚¬â€ trains in hours on a single GPU with a few thousand images; document whichever is actually used in `experiments/` configs, keep this spec's architecture description in sync with the real choice.

## Conditioning mechanism
- Mapping network `M: z, c Ã¢â€ â€™ w` where `z ~ N(0,I)` (style noise) and `c` is the **conditioning embedding**:
  `c = concat[ context_embedding (from Memory Graph), relationship_type_onehot, emotion_tag_onehot, occasion_onehot ]`, projected through a small MLP to the mapping network's input dimensionality.
- `w` modulates convolution weights via AdaIN/weight-demodulation exactly as in StyleGAN2, per-layer, so both coarse (composition/motif) and fine (color/texture) style are conditionable.
- **Agency-slider interpolation**: at inference time, compute two style codes Ã¢â‚¬â€
  `w_human` from a user-chosen style tag/reference (low-agency, human-directed) and
  `w_ai` from the pooled memory-graph embedding alone (high-agency, AI-inferred) Ã¢â‚¬â€
  and generate from `w_final = slerp(w_human, w_ai, agency_slider)`. This makes the agency dial a literal, auditable operation on the model, not just a UI gimmick Ã¢â‚¬â€ worth a figure in the paper.

## Losses
- **Adversarial loss**: non-saturating logistic loss (`log(sigmoid(D(x)))` form), as in StyleGAN2.
- **R1 gradient penalty** on the discriminator (real-data regularization), standard StyleGAN2 recipe (`r1_gamma` tuned per dataset resolution).
- **Path length regularization** on the generator (encourages a smoother, more interpolatable `w`-space Ã¢â‚¬â€ directly useful for the agency-slider interpolation above).
- **Conditioning consistency loss**: an auxiliary CLIP-similarity term between the generated image and a text description built from the conditioning attributes (e.g., "a warm, nostalgic illustration for a close friend's birthday"), weighted low (e.g., 0.1Ãƒâ€”) so it nudges rather than dominates Ã¢â‚¬â€ prevents the model from ignoring `c` under small-data regimes.
- **ADA**: adaptive augmentation pipeline (StyleGAN2-ADA) Ã¢â‚¬â€ pixel blitting, geometric, color transforms Ã¢â‚¬â€ with the standard `p`-adjustment heuristic (target real-vs-fake discriminator sign ~0.6).

## Data
- Recommended dataset(s) for the demo: a small curated set (2kÃ¢â‚¬â€œ10k images) of illustration/greeting-card-style art with permissive licenses (e.g., public-domain illustration collections, or an internally commissioned small style set) Ã¢â‚¬â€ **do not train on scraped copyrighted artwork**; document exact source and license in `experiments/DATA_CARD.md`.
- Weak conditioning labels (relationship/occasion/emotion) attached via metadata where available, or synthetically assigned during a "conditioning pretext" phase if the art dataset itself is unlabeled (cluster images by CLIP embedding, map clusters to pseudo-emotion-tags) Ã¢â‚¬â€ document this shortcut plainly as a limitation in the paper.
- Held-out validation split (10%) for FID/KID tracking during training.

## Training recipe (starting point Ã¢â‚¬â€ tune empirically, log actual values used)
| Hyperparameter | Starting value |
|---|---|
| Resolution | 256Ãƒâ€”256 (bump to 512 only if compute allows) |
| Batch size | 16Ã¢â‚¬â€œ32 (gradient-accumulate if VRAM-limited) |
| Learning rate (G, D) | 0.0025 (Adam, ÃŽÂ²1=0, ÃŽÂ²2=0.99) Ã¢â‚¬â€ StyleGAN2 default scaled for smaller batch |
| R1 gamma | 0.5Ã¢â‚¬â€œ1.0 for 256px (scale with resolutionÃ‚Â²) |
| ADA target | 0.6 |
| Path length weight | 2.0, applied every 4 steps (lazy regularization) |
| CLIP consistency weight | 0.1 |
| Training length | until FID plateaus on val split; log curve, don't just pick a fixed step count |
| Mixed precision | fp16/bf16 if GPU supports it, to fit resolution/batch on 8Ã¢â‚¬â€œ16GB VRAM |


### Training implementation and smoke run
- `src/gan/configs/train.json` implements the full recipe: 256px, batch 16, Adam learning rates `0.0025`, betas `(0, 0.99)`, `r1_gamma=0.8`, ADA target `0.6`, path-length weight `2.0` every 4 steps, and CLIP consistency weight `0.1`. It evaluates canonical Inception-v3 FID/KID every 1,000 steps and supports FID-plateau early stopping.
- `src/gan/configs/train_smoke.json` is intentionally compute-reduced for pipeline validation: 16px, batch 4, 8 training images, 4 validation images, 200 steps, `z/w/condition=64`, two mapping layers, `channel_base=256`, and `channel_max=64`. Optimizer and regularization values remain the recipe values.
- The completed CPU smoke run is `experiments/run-002`: 200/200 steps, checkpoint/sample/logs written, AMP requested but correctly inactive because CUDA was unavailable. Its held-out metric backend is explicitly `openclip_vit_b32`, yielding feature-space Frechet distance `0.362440` and polynomial KID `0.000834` on four samples. These are pipeline smoke diagnostics, not canonical Inception FID/KID and must not be reported as paper results.
- Canonical full-run FID/KID remains configured through TorchMetrics/torch-fidelity with Inception-v3 features. The smoke fallback was necessary because the official Inception checkpoint could not be downloaded in the current restricted environment; metric backend identity is stored beside every metric record.
## Evaluation of the GAN itself (feeds `06_evaluation_spec.md`)
- **FID / KID** vs. held-out real images, tracked over training.
- **CLIPScore** between generated image and its conditioning-derived text description (checks conditioning fidelity, not just realism).
- **Style-agency interpolation smoothness**: perceptual distance (LPIPS) sampled along the `slerp(w_human, w_ai, t)` path for `tÃ¢Ë†Ë†{0,0.25,0.5,0.75,1}` should vary smoothly/monotonically Ã¢â‚¬â€ a quantitative check that the "agency slider" is a meaningful latent operation and not noise.
- **Diversity**: pairwise LPIPS across samples from the same conditioning vector with different `z`, to confirm the model isn't mode-collapsing per-condition.

## Interfaces
```python
class MemoryGAN:
    def load(checkpoint_path: str) -> "MemoryGAN": ...
    def generate(
        context_embedding: np.ndarray,   # from Memory Graph
        relationship_type: str,
        emotion_tag: str,
        occasion: str,
        agency_slider: float,            # 0..1
        human_style_ref: Optional[np.ndarray] = None,  # required if agency_slider < 1
        seed: Optional[int] = None,
    ) -> PIL.Image: ...
```

## Explicit scope limits (state plainly in the paper)
- Small dataset Ã¢â€¡â€™ limited stylistic range; this is a proof-of-concept generative core, not a general-purpose illustration model.
- No claim of beating diffusion-model image quality Ã¢â‚¬â€ the GAN is chosen for **inference speed + interpretable, interpolatable latent control**, which is the actual point of the contribution.
