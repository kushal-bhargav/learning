# GMGI GAN Dataset Card

## Status

This card documents the data pipeline and the current pipeline-validation sample. GAN training has not started. The 24-image sample validates acquisition, preprocessing, pseudo-labeling, and splitting; it is not large enough for GAN training. A future training snapshot must contain 2,000-10,000 manually reviewed images and update this card.

## Source selection

- **Source:** Art Institute of Chicago Open Access Collection.
- **Acquisition:** Art Institute public API search plus its IIIF Image API.
- **Selection queries:** `postcard`, `greeting card`, `illustration`, and `decorative print`, all configured in `src/gan/configs/data_pipeline.json`.
- **Eligibility gate:** every retained API record must have `is_public_domain=true` and a non-empty `image_id`.
- **Download behavior:** images are fetched serially at the museum-recommended 843-pixel IIIF size, with a one-second delay between downloads.
- **Why selected:** the source provides an auditable public API, explicit per-object public-domain status, stable object metadata, and documented CC0 image access. This is safer and more reproducible than scraping contemporary art.

## License and permitted use

The Art Institute of Chicago offers eligible Open Access images under the **Creative Commons Zero (CC0) Public Domain Designation**. Each manifest row retains the object page, IIIF image URL, public-domain flag, license string, source query, and source-image SHA-256 checksum.

- Open Access images: https://www.artic.edu/open-access/open-access-images
- Image licensing: https://www.artic.edu/image-licensing
- Public API and IIIF documentation: https://api.artic.edu/docs/
- CC0 legal tool: https://creativecommons.org/publicdomain/zero/1.0/

Users remain responsible for non-copyright rights that may apply to a particular use. Metadata availability alone is never treated as image permission; the public-domain flag and image identifier are mandatory.

## Intended use

The future full dataset is intended to train and evaluate a small conditional GAN that creates decorative, illustration-like gift artifacts for the GMGI research prototype. It is not intended for artist imitation, biometric analysis, identity inference, or production deployment.

## Processing

1. Query only public-domain artworks and preserve source metadata and checksums.
2. Decode as RGB, center-crop to a square, and resize to **256 x 256** with Lanczos resampling; save lossless PNG files.
3. Encode images using real OpenCLIP **ViT-B/32 with OpenAI weights**.
4. L2-normalize embeddings and cluster them with deterministic k-means++ (`seed=2026`, six clusters).
5. Encode configured emotion descriptions with the same CLIP model. Mean-center and scale prompt similarities across centroids to remove prompt-wide similarity bias, then assign each cluster its highest relative pseudo-emotion label.
6. Create a deterministic cluster-stratified train/validation split (`seed=2026`, target validation fraction 0.10).
7. Write JSONL metadata, a NumPy embedding matrix, an execution report, and a labeled contact sheet.

All source selection, preprocessing, model, clustering, splitting, and preview settings are config-driven. Source images, processed images, and embeddings are ignored by Git.

## Current pipeline-validation sample

- Records: **24**
- Train: **20**
- Validation: **4**
- Resolution: **256 x 256 RGB**
- CLIP model: **ViT-B-32, OpenAI weights**
- Clusters: **6**
- Cluster labels: `0=nostalgia`, `1=joy`, `2=nostalgia`, `3=joy`, `4=joy`, `5=joy`
- Training started: **No**

The 4/24 validation share is larger than 10% because the pilot uses a cluster-stratified split that preserves at least one training example per cluster where possible. The full dataset should converge much more closely to 90/10.

## Labels and limitations

The museum records do not provide GMGI relationship, occasion, or emotion labels. `pseudo_emotion_tag` is a weak CLIP-derived label, not a curator annotation or a claim about an artwork's true meaning. Cluster IDs and labels may change with the data snapshot, model, prompts, cluster count, or seed.

This pilot is visibly broad museum imagery rather than a clean greeting-card corpus. Search ranking introduces selection bias; historical and geographic coverage is uneven; center crops can remove context; and CLIP carries known representation biases. The contact sheet must be manually reviewed before scaling or training. A production-quality acquisition should add explicit visual relevance filtering and record rejection reasons.

## Reproducibility artifacts

- Configuration: `src/gan/configs/data_pipeline.json`
- Run report: `experiments/data_pipeline_run.json`
- Local manifest: `data/gan/metadata.jsonl`
- Local embeddings: `data/gan/clip_embeddings.npy`
- Preview: `experiments/sample_clustered_images.jpg`
- Pipeline command: `python -m src.gan.data_pipeline --config src/gan/configs/data_pipeline.json`