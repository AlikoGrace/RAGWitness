from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None  # None = auto
    normalize: bool = True
    show_progress_bar: bool = False

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dim(self) -> int:
        # SentenceTransformers exposes embedding dimension in a few ways; this one is stable
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Returns shape: (n, dim) float32.
        normalize=True makes cosine similarity stable + comparable across runs.
        """
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        return vecs

    def fingerprint(self) -> Dict[str, Any]:
        """
        Creates a compact fingerprint you can log in runs/config/environment.
        """
        # Model card name is already in model_name. We add dim + platform for reproducibility notes.
        fp = {
            "embedder_type": "sentence_transformers",
            "model_name": self.model_name,
            "dim": self.dim,
            "normalize": self.normalize,
            "show_progress_bar": self.show_progress_bar,
            "platform": platform.platform(),
        }
        # Optional: a simple hash of the fingerprint itself (not the weights)
        fp_str = repr(sorted(fp.items())).encode("utf-8")
        fp["fingerprint_sha256"] = hashlib.sha256(fp_str).hexdigest()
        return fp
