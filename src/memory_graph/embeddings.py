from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class OpenClipImageEncoder:
    """Lazy OpenCLIP ViT-B/32 image encoder using original OpenAI weights."""

    model_name = "ViT-B-32"
    pretrained = "openai"
    dimension = 512

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self._preprocess = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    def encode(self, image: str | Path | Image.Image) -> np.ndarray:
        import torch

        self._load()
        assert self._model is not None and self._preprocess is not None
        source = Image.open(image) if isinstance(image, (str, Path)) else image
        rgb = source.convert("RGB")
        try:
            batch = self._preprocess(rgb).unsqueeze(0).to(self.device)
        finally:
            rgb.close()
            if source is not image:
                source.close()
        with torch.inference_mode():
            vector = self._model.encode_image(batch)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector[0].detach().cpu().numpy().astype(np.float32)


class SentenceTextEncoder:
    """Lazy multilingual sentence encoder aligned with CLIP ViT-B/32."""

    model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    dimension = 512

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, text: str | Sequence[str]) -> np.ndarray:
        self._load()
        assert self._model is not None
        vector = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vector, dtype=np.float32)
