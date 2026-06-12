"""Macenko H&E stain normalization (self-contained NumPy implementation).

Why this exists: the v1 model showed a ~9-point OOF→LB gap, the signature of stain /
scanner / lab color shift between train and test (Tellez et al. 2019, arXiv:1902.06543).
Because the backbones are FROZEN, we cannot make them stain-invariant by augmentation
(that needs training the net). The frozen-feature lever is *normalization*: map every
patch onto a single canonical H&E color appearance so the frozen net sees consistent
color regardless of the source slide.

Algorithm (Macenko et al. 2009, "A method for normalizing histology slides..."):
  1. RGB → optical density  OD = -log((I+1)/Io).
  2. Drop near-white background pixels (OD below `beta`).
  3. Top-2 eigenvectors of the OD covariance span the "stain plane"; the robust angular
     extremes (alpha / 100-alpha percentiles) give this image's H and E stain vectors.
  4. Solve for per-pixel stain concentrations, rescale them so each stain's 99th-pct
     concentration matches a fixed reference, and recompose against a canonical reference
     stain matrix → a stain-normalized RGB image.

We normalize to a fixed canonical reference (the widely-used HERef/maxCRef constants) by
default, which is more robust than fitting an arbitrary single patch; `fit()` can instead
derive the reference from a chosen patch if desired. Pure NumPy + vectorized, so it runs
fast inside DataLoader workers and has no third-party stain dependency.
"""

from __future__ import annotations

import numpy as np

# Canonical reference H&E stain matrix (3x2: rows RGB, cols H/E) and the reference
# max stain concentrations — the standard constants used across Macenko implementations.
_HEREF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]], dtype=np.float64)
_MAXCREF = np.array([1.9705, 1.0308], dtype=np.float64)


class MacenkoNormalizer:
    """Normalize H&E patches to a canonical stain appearance (Macenko 2009)."""

    def __init__(
        self,
        Io: int = 240,
        beta: float = 0.15,
        alpha: float = 1.0,
        HERef: np.ndarray | None = None,
        maxCRef: np.ndarray | None = None,
    ) -> None:
        """Configure the normalizer.

        Args:
            Io: Transmitted-light intensity for the OD transform (240 is standard).
            beta: OD threshold below which a pixel is treated as background and dropped.
            alpha: Percentile (and 100-alpha) used to find robust stain-vector extremes.
            HERef: Reference 3x2 stain matrix to normalize toward (defaults to canonical).
            maxCRef: Reference per-stain max concentrations (defaults to canonical).
        """
        self.Io = Io
        self.beta = beta
        self.alpha = alpha
        self.HERef = _HEREF.copy() if HERef is None else np.asarray(HERef, dtype=np.float64)
        self.maxCRef = _MAXCREF.copy() if maxCRef is None else np.asarray(maxCRef, dtype=np.float64)

    def _stain_matrix_and_conc(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Estimate this image's stain matrix and per-pixel concentrations.

        Args:
            img: HxWx3 uint8 RGB image.

        Returns:
            (HE, C): HE is the 3x2 stain matrix; C is the 2xN concentration matrix over
            all pixels (background included), ready to be rescaled and recomposed.

        Raises:
            ValueError: if too few tissue pixels remain to estimate stains robustly.
        """
        # 1) Optical density for every pixel (flattened to N x 3).
        flat = img.reshape(-1, 3).astype(np.float64)
        OD = -np.log((flat + 1.0) / self.Io)

        # 2) Keep tissue pixels (drop near-white background where all OD channels small).
        tissue = OD[~np.any(OD < self.beta, axis=1)]
        if tissue.shape[0] < 100:
            raise ValueError("too few tissue pixels for stain estimation")

        # 3) Stain plane = top-2 eigenvectors of the tissue OD covariance.
        _, eigvecs = np.linalg.eigh(np.cov(tissue.T))
        plane = eigvecs[:, 1:3]  # the two largest eigenvalues (eigh returns ascending)

        # 4) Project onto the plane and take robust angular extremes as the two stains.
        proj = tissue @ plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        a_min, a_max = np.percentile(angles, self.alpha), np.percentile(angles, 100 - self.alpha)
        v_min = plane @ np.array([np.cos(a_min), np.sin(a_min)])
        v_max = plane @ np.array([np.cos(a_max), np.sin(a_max)])
        # Order so the first column is hematoxylin (the vector with larger R component).
        HE = np.column_stack((v_min, v_max) if v_min[0] > v_max[0] else (v_max, v_min))

        # 5) Per-pixel concentrations C (2 x N) via least squares: OD = HE @ C.
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        return HE, C

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Stain-normalize one patch; return the original unchanged on any failure.

        Args:
            img: HxWx3 uint8 RGB image.

        Returns:
            HxWx3 uint8 RGB image normalized to the canonical H&E appearance. If stain
            estimation fails (e.g. a near-blank patch), the input is returned as-is so the
            pipeline never crashes on a degenerate patch.
        """
        try:
            h, w, _ = img.shape
            _, C = self._stain_matrix_and_conc(img)

            # Rescale concentrations so each stain's 99th percentile matches the reference.
            maxC = np.percentile(C, 99, axis=1)
            maxC = np.where(maxC <= 0, 1.0, maxC)  # guard against degenerate stains
            C = C * (self.maxCRef / maxC)[:, None]

            # Recompose against the canonical reference stain matrix.
            out = self.Io * np.exp(-self.HERef @ C)
            out = np.clip(out, 0, 255).T.reshape(h, w, 3)
            return out.astype(np.uint8)
        except Exception:
            return img

    def fit(self, ref_img: np.ndarray) -> "MacenkoNormalizer":
        """Optionally derive the reference stain matrix / max concentrations from a patch.

        Args:
            ref_img: HxWx3 uint8 RGB reference patch.

        Returns:
            self, with HERef/maxCRef replaced by the reference patch's estimates. Falls
            back silently to the canonical constants if estimation fails.
        """
        try:
            HE, C = self._stain_matrix_and_conc(ref_img)
            self.HERef = HE
            maxC = np.percentile(C, 99, axis=1)
            self.maxCRef = np.where(maxC <= 0, 1.0, maxC)
        except Exception:
            pass
        return self
