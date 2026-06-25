import numpy as np
from pyFM.functional import FunctionalMapping
import os
from pathlib import Path
from tqdm.auto import tqdm
import vtk
import os
import numpy as np
from pathlib import Path
from PynamicMesh.utils.tools import  mesh_mat2object

try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False



class CustomFunctionalMapping(FunctionalMapping):
    """
    Functional map wrapper with:
      - Combined descriptor support (WKS + HKS)
    """

    def _set_descriptors(self, descr1: np.ndarray, descr2: np.ndarray) -> None:
        self.descr1 = descr1
        self.descr2 = descr2
        self.A = descr1
        self.B = descr2

    def preprocess(
        self,
        K=(10, 10),
        n_descr=100,
        descr_type="WKS",
        landmarks=None,
        subsample_step=1,
        k_process=None,
        verbose=False,
        **kwargs,
    ):
        """
        Preprocess the meshes for functional map fitting.

        Supported descr_type:
          - "WKS"
          - "HKS"
          - "WKS+HKS" or "HKS+WKS" (Combined)
        """

        required_k1 = max(K[0], k_process if k_process else 100)
        required_k2 = max(K[1], k_process if k_process else 100)

        if self.mesh1.eigenvalues is None or len(self.mesh1.eigenvalues) < required_k1:
            self.mesh1.process(k=required_k1)
        if self.mesh2.eigenvalues is None or len(self.mesh2.eigenvalues) < required_k2:
            self.mesh2.process(k=required_k2)

        combined = isinstance(descr_type, str) and descr_type.upper() in {"WKS+HKS", "HKS+WKS"}

        if combined:
            super().preprocess(
                K=K,
                n_descr=n_descr,
                descr_type="WKS",
                landmarks=landmarks,
                subsample_step=subsample_step,
                k_process=required_k1, 
                verbose=verbose,
                **kwargs,
            )
            descr1_wks = self.descr1.copy()
            descr2_wks = self.descr2.copy()

            super().preprocess(
                K=K,
                n_descr=n_descr,
                descr_type="HKS",
                landmarks=landmarks,
                subsample_step=subsample_step,
                k_process=required_k1,
                verbose=verbose,
                **kwargs,
            )
            descr1_hks = self.descr1.copy()
            descr2_hks = self.descr2.copy()

            def normalize_descriptor(desc):
                desc_gpu = xp.asarray(desc) if GPU_AVAILABLE else desc
                mean = desc_gpu.mean(axis=0)
                std = desc_gpu.std(axis=0) + 1e-8
                norm_gpu = (desc_gpu - mean) / std
                return norm_gpu.get() if GPU_AVAILABLE else norm_gpu

            descr1_wks = normalize_descriptor(descr1_wks)
            descr2_wks = normalize_descriptor(descr2_wks)
            
            descr1_hks = normalize_descriptor(descr1_hks) 
            descr2_hks = normalize_descriptor(descr2_hks)

            self._set_descriptors(
                np.hstack([descr1_wks, descr1_hks]),
                np.hstack([descr2_wks, descr2_hks]),
            )
        else:
            super().preprocess(
                K=K,
                n_descr=n_descr,
                descr_type=descr_type,
                landmarks=landmarks,
                subsample_step=subsample_step,
                k_process=required_k1,
                verbose=verbose,
                **kwargs,
            )
            self._set_descriptors(self.descr1, self.descr2)

        return self